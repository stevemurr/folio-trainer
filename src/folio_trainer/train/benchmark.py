"""Benchmark orchestration for policy and model variants."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from folio_trainer.backtest.baselines import (
    equal_weight_strategy,
    inverse_volatility_strategy,
    min_variance_strategy,
    previous_hold_strategy,
    teacher_replay_strategy,
)
from folio_trainer.backtest.data_utils import (
    align_returns_to_dates,
    build_covariance_lookup,
    build_feature_matrix_by_date,
    build_returns_by_date,
)
from folio_trainer.backtest.metrics import compute_metrics, metrics_to_dict
from folio_trainer.backtest.simulator import BacktestResult, simulate
from folio_trainer.config.loader import apply_strategy_profile
from folio_trainer.config.schema import PipelineConfig
from folio_trainer.models.dataset import PreparedDataset, prepare_dataset, stack_by_date
from folio_trainer.models.factory import create_direct_weight_model
from folio_trainer.models.utility_scorer import UtilityScorer
from folio_trainer.train.train_utility import (
    build_shared_feature_table,
    prepare_utility_dataset,
)

logger = logging.getLogger(__name__)


def run_benchmark_suite(
    prices: pl.DataFrame,
    asset_features: pl.DataFrame,
    labels: pl.DataFrame,
    splits: pl.DataFrame,
    ticker_order: list[str],
    config: PipelineConfig,
    market_features: pl.DataFrame | None = None,
    cross_asset_features: pl.DataFrame | None = None,
    candidates: pl.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Train learned variants and benchmark them against corrected baselines."""
    dataset = prepare_dataset(
        asset_features=asset_features,
        labels=labels,
        splits=splits,
        ticker_order=ticker_order,
        market_features=market_features,
        cross_asset_features=cross_asset_features,
        calendar_dates=sorted(prices["date"].unique().to_list()),
    )

    _, returns_by_date = build_returns_by_date(prices, ticker_order)
    test_returns = align_returns_to_dates(
        returns_by_date,
        dataset.test_prediction_dates,
        len(ticker_order),
    )

    baseline_results = _run_baselines(
        dataset=dataset,
        asset_features=asset_features,
        labels=labels,
        returns_by_date=returns_by_date,
        config=config,
    )

    learned_results: dict[str, dict] = {}
    for variant in _default_variants():
        variant_cfg = apply_strategy_profile(config, variant["strategy"])
        variant_cfg.model.direct_weight_model.kind = variant["kind"]
        variant_result = _train_and_evaluate_variant(
            name=variant["name"],
            dataset=dataset,
            test_returns=test_returns,
            config=variant_cfg,
            output_dir=Path(output_dir) / "variants" if output_dir else None,
        )
        learned_results[variant["name"]] = variant_result

    if candidates is not None and len(candidates) > 0:
        utility_result = _train_and_evaluate_utility_variant(
            dataset=dataset,
            candidates=candidates,
            market_features=market_features,
            cross_asset_features=cross_asset_features,
            test_returns=test_returns,
            config=apply_strategy_profile(config, "neutral"),
            output_dir=Path(output_dir) / "variants" / "utility_rerank_neutral" if output_dir else None,
        )
        if utility_result is not None:
            learned_results["utility_rerank_neutral"] = utility_result

    result = {
        "baselines": {name: payload["metrics"] for name, payload in baseline_results.items()},
        "learned_variants": {name: payload["metrics"] for name, payload in learned_results.items()},
        "benchmark_table": _build_benchmark_table(baseline_results, learned_results),
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "benchmark_results.json").write_text(json.dumps(result, indent=2, default=str))
        (out / "benchmark_report.md").write_text(_render_benchmark_report(result["benchmark_table"]))

    return result


def _default_variants() -> list[dict[str, str]]:
    return [
        {"name": "gbm_neutral", "strategy": "neutral", "kind": "gbm"},
        {"name": "gbm_aggressive", "strategy": "aggressive", "kind": "gbm"},
        {"name": "gbm_conservative", "strategy": "conservative", "kind": "gbm"},
        {"name": "rank_gbm_neutral", "strategy": "neutral", "kind": "rank_gbm"},
        {"name": "linear_neutral", "strategy": "neutral", "kind": "linear"},
    ]


def _run_baselines(
    dataset: PreparedDataset,
    asset_features: pl.DataFrame,
    labels: pl.DataFrame,
    returns_by_date: dict,
    config: PipelineConfig,
) -> dict[str, dict]:
    signal_dates = dataset.test_dates
    execution_dates = dataset.test_prediction_dates
    n_assets = len(dataset.ticker_names)
    n_days = len(signal_dates)
    test_returns = align_returns_to_dates(returns_by_date, execution_dates, n_assets)

    rolling_vol = build_feature_matrix_by_date(
        asset_features,
        dataset.ticker_names,
        signal_dates,
        "vol_20",
        default_value=1.0,
    )
    cov_lookup = build_covariance_lookup(signal_dates, returns_by_date, window=60)

    strategies = {
        "equal_weight": equal_weight_strategy(n_assets, n_days),
        "previous_hold": previous_hold_strategy(n_assets, n_days, asset_returns=test_returns),
        "inverse_volatility": inverse_volatility_strategy(n_assets, n_days, rolling_vol=rolling_vol),
        "min_variance": min_variance_strategy(
            n_assets,
            n_days,
            rolling_cov_fn=lambda idx: cov_lookup.get(signal_dates[idx]),
        ),
        "teacher_replay": teacher_replay_strategy(n_days, labels, signal_dates),
    }

    results: dict[str, dict] = {}
    for name, weights in strategies.items():
        bt = simulate(
            weight_signals=weights,
            asset_returns=test_returns,
            dates=execution_dates,
            execution_config=config.execution,
            cost_config=config.cost_model,
            ticker_names=dataset.ticker_names,
        )
        results[name] = {
            "backtest": bt,
            "metrics": metrics_to_dict(compute_metrics(bt, use_net=True)),
        }
    return results


def _train_and_evaluate_variant(
    name: str,
    dataset: PreparedDataset,
    test_returns: np.ndarray,
    config: PipelineConfig,
    output_dir: Path | None = None,
) -> dict:
    model = create_direct_weight_model(config.model.direct_weight_model)
    train_meta = model.train(
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        w_train=dataset.w_train,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
        w_val=dataset.w_val,
        feature_names=dataset.feature_names,
        train_date_indices=dataset.train_date_indices,
        val_date_indices=dataset.val_date_indices,
        target_weights_train=dataset.target_weight_train,
        target_weights_val=dataset.target_weight_val,
    )

    test_weights = model.predict_weights(dataset.X_test, dataset.test_date_indices)
    weight_matrix = stack_by_date(test_weights, dataset.test_date_indices, len(dataset.ticker_names))
    bt = simulate(
        weight_signals=weight_matrix,
        asset_returns=test_returns,
        dates=dataset.test_prediction_dates,
        execution_config=config.execution,
        cost_config=config.cost_model,
        ticker_names=dataset.ticker_names,
    )
    metrics = metrics_to_dict(compute_metrics(bt, use_net=True))
    importance = model.feature_importance()

    if output_dir:
        run_dir = output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        model.save(run_dir / "model.bin")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        (run_dir / "top_features.json").write_text(
            json.dumps(
                sorted(importance.items(), key=lambda item: -item[1])[:20],
                indent=2,
                default=str,
            )
        )

    return {
        "train_meta": train_meta,
        "backtest": bt,
        "metrics": metrics,
        "top_features": sorted(importance.items(), key=lambda item: -item[1])[:10],
    }


def _train_and_evaluate_utility_variant(
    dataset: PreparedDataset,
    candidates: pl.DataFrame,
    market_features: pl.DataFrame | None,
    cross_asset_features: pl.DataFrame | None,
    test_returns: np.ndarray,
    config: PipelineConfig,
    output_dir: Path | None = None,
) -> dict | None:
    shared_features = build_shared_feature_table(market_features, cross_asset_features)
    if len(shared_features) == 0:
        return None

    train_cands = candidates.filter(pl.col("asof_date").is_in(dataset.train_dates))
    val_cands = candidates.filter(pl.col("asof_date").is_in(dataset.val_dates))
    test_cands = candidates.filter(pl.col("asof_date").is_in(dataset.test_dates))
    if len(train_cands) == 0 or len(test_cands) == 0:
        return None

    train_state, train_weights, train_obj, _, state_feature_names = prepare_utility_dataset(
        train_cands,
        shared_features,
    )
    val_state, val_weights, val_obj, val_groups, _ = prepare_utility_dataset(
        val_cands,
        shared_features,
    )

    model = UtilityScorer()
    train_meta = model.train(
        train_state,
        train_weights,
        train_obj,
        val_state,
        val_weights,
        val_obj,
    )
    val_metrics = model.evaluate(val_state, val_weights, val_obj, val_groups)

    joined_test = test_cands.join(shared_features, on="asof_date", how="left").sort(["asof_date", "rank"])
    test_state, test_weights_arr, _, _, _ = prepare_utility_dataset(joined_test, shared_features)
    preds = model.predict(test_state, test_weights_arr)

    best_by_date: dict = {}
    for row, pred in zip(joined_test.iter_rows(named=True), preds, strict=False):
        date = row["asof_date"]
        if date not in best_by_date or pred > best_by_date[date][0]:
            best_by_date[date] = (float(pred), np.asarray(json.loads(row["weights_json"]), dtype=np.float32))

    weight_matrix = np.vstack([best_by_date[date][1] for date in dataset.test_dates if date in best_by_date])
    if len(weight_matrix) != len(dataset.test_prediction_dates):
        return None

    bt = simulate(
        weight_signals=weight_matrix,
        asset_returns=test_returns,
        dates=dataset.test_prediction_dates,
        execution_config=config.execution,
        cost_config=config.cost_model,
        ticker_names=dataset.ticker_names,
    )
    metrics = metrics_to_dict(compute_metrics(bt, use_net=True))

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(output_dir / "utility_scorer.bin")
        (output_dir / "utility_metrics.json").write_text(
            json.dumps({"train_meta": train_meta, "val_metrics": val_metrics, "test_metrics": metrics}, indent=2)
        )
        (output_dir / "utility_feature_manifest.json").write_text(
            json.dumps({"state_feature_names": state_feature_names}, indent=2)
        )

    return {
        "train_meta": train_meta,
        "validation_metrics": val_metrics,
        "backtest": bt,
        "metrics": metrics,
    }


def _build_benchmark_table(
    baseline_results: dict[str, dict],
    learned_results: dict[str, dict],
) -> list[dict[str, float | str]]:
    rows = []
    for bucket in (baseline_results, learned_results):
        for name, payload in bucket.items():
            metrics = payload["metrics"]
            rows.append(
                {
                    "name": name,
                    "Sharpe": metrics["Sharpe"],
                    "CAGR": metrics["CAGR"],
                    "Max DD": metrics["Max DD"],
                    "Avg Turnover": metrics["Avg Turnover"],
                    "Avg Active Pos": metrics["Avg Active Pos"],
                    "HHI": metrics["HHI"],
                }
            )
    rows.sort(key=lambda row: row["Sharpe"], reverse=True)
    return rows


def _render_benchmark_report(rows: list[dict[str, float | str]]) -> str:
    lines = ["# Benchmark Report", "", "| Strategy | Sharpe | CAGR | Max DD | Avg Turnover | Avg Active Pos | HHI |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            "| {name} | {Sharpe:.4f} | {CAGR:.2%} | {Max DD:.2%} | {Avg Turnover:.2%} | {Avg Active Pos:.2f} | {HHI:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines)
