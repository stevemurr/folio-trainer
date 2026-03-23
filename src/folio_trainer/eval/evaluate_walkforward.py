"""Walk-forward evaluation with per-fold retraining (spec 2.7)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from folio_trainer.backtest.metrics import compute_metrics, metrics_to_dict
from folio_trainer.backtest.simulator import simulate
from folio_trainer.config.schema import PipelineConfig
from folio_trainer.models.dataset import stack_by_date
from folio_trainer.models.factory import create_direct_weight_model

logger = logging.getLogger(__name__)


def evaluate_walkforward(
    asset_features: pl.DataFrame,
    labels: pl.DataFrame,
    returns_by_date: dict,
    splits: pl.DataFrame,
    config: PipelineConfig,
    ticker_order: list[str],
    market_features: pl.DataFrame | None = None,
    cross_asset_features: pl.DataFrame | None = None,
    calendar_dates: list | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Run walk-forward evaluation with per-fold retraining.

    Parameters
    ----------
    asset_features
        Full asset feature table.
    labels
        Full teacher label table.
    returns_by_date
        Dict mapping date -> (n_assets,) returns array.
    splits
        Walk-forward split metadata.
    config
        Pipeline config.
    ticker_order
        Ordered list of tickers.
    output_dir
        Directory for output.

    Returns
    -------
    dict with per-fold and aggregate metrics.
    """
    from folio_trainer.models.dataset import prepare_dataset

    # Find walk-forward folds
    fold_names = set()
    for name in splits["split_name"].to_list():
        if name.startswith("walkforward_"):
            fold_id = "_".join(name.split("_")[:2])
            fold_names.add(fold_id)

    fold_names = sorted(fold_names)
    if not fold_names:
        logger.warning("No walk-forward folds found in splits.")
        return {"folds": [], "aggregate": {}}

    fold_results = []

    for fold_id in fold_names:
        logger.info("Processing fold %s...", fold_id)

        # Build fold-specific splits table
        fold_splits = splits.filter(
            pl.col("split_name").str.starts_with(fold_id)
        ).with_columns(
            pl.col("split_name").str.replace(f"{fold_id}_", "").alias("split_name")
        )

        if len(fold_splits.filter(pl.col("split_name") == "train")) == 0:
            continue

        try:
            dataset = prepare_dataset(
                asset_features=asset_features,
                labels=labels,
                splits=fold_splits,
                ticker_order=ticker_order,
                market_features=market_features,
                cross_asset_features=cross_asset_features,
                calendar_dates=calendar_dates,
            )
        except Exception as e:
            logger.warning("Failed to prepare dataset for fold %s: %s", fold_id, e)
            continue

        model = create_direct_weight_model(config.model.direct_weight_model)
        try:
            model.train(
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
        except Exception as e:
            logger.warning("Training failed for fold %s: %s", fold_id, e)
            continue

        test_weights = model.predict_weights(dataset.X_test, dataset.test_date_indices)
        n_assets = len(ticker_order)
        weight_matrix = stack_by_date(test_weights, dataset.test_date_indices, n_assets)
        n_dates = len(dataset.test_prediction_dates)

        test_returns = np.zeros((n_dates, n_assets))
        for i, d in enumerate(dataset.test_prediction_dates):
            if d in returns_by_date:
                test_returns[i] = returns_by_date[d]

        bt_result = simulate(
            weight_signals=weight_matrix,
            asset_returns=test_returns,
            dates=dataset.test_prediction_dates,
            execution_config=config.execution,
            cost_config=config.cost_model,
        )
        metrics = compute_metrics(bt_result, use_net=True)

        fold_results.append({
            "fold": fold_id,
            "test_start": str(dataset.test_prediction_dates[0]) if dataset.test_prediction_dates else None,
            "test_end": str(dataset.test_prediction_dates[-1]) if dataset.test_prediction_dates else None,
            "n_test_dates": n_dates,
            "metrics": metrics_to_dict(metrics),
        })

        logger.info("Fold %s: Sharpe=%.4f, CAGR=%.4f", fold_id, metrics.sharpe, metrics.cagr)

    # Aggregate metrics across folds
    aggregate = {}
    if fold_results:
        metric_keys = fold_results[0]["metrics"].keys()
        for key in metric_keys:
            values = [f["metrics"][key] for f in fold_results]
            aggregate[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    result = {
        "folds": fold_results,
        "aggregate": aggregate,
        "n_folds": len(fold_results),
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "walkforward_results.json").write_text(json.dumps(result, indent=2, default=str))

    return result
