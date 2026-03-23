"""Train a direct-weight model and run corrected research benchmarks.

Usage:
    .venv/bin/python scripts/run_training.py
    .venv/bin/python scripts/run_training.py --strategy aggressive --model-kind linear
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import polars as pl

from folio_trainer.backtest.data_utils import align_returns_to_dates, build_returns_by_date
from folio_trainer.config.loader import load_config
from folio_trainer.eval.evaluate_holdout import evaluate_holdout
from folio_trainer.models.dataset import prepare_dataset
from folio_trainer.models.factory import load_direct_weight_model
from folio_trainer.train.benchmark import run_benchmark_suite
from folio_trainer.train.train_direct import train_direct_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--data-dir", default="data", help="Data directory.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory.")
    parser.add_argument("--run-id", default=None, help="Optional run id.")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["aggressive", "neutral", "conservative"],
        help="Override the named strategy profile for the primary model.",
    )
    parser.add_argument(
        "--model-kind",
        default=None,
        choices=["gbm", "rank_gbm", "linear"],
        help="Override the primary model kind.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Train and evaluate only the primary run without the benchmark suite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)

    cfg = load_config(
        args.config,
        overrides={"data_dir": str(data_dir), "artifacts_dir": str(artifacts_dir)},
        strategy=args.strategy,
    )
    if args.model_kind is not None:
        cfg.model.direct_weight_model.kind = args.model_kind

    logger.info("Loading datasets from %s", data_dir)
    prices = pl.read_parquet(data_dir / "bronze" / "prices_daily.parquet")
    asset_features = pl.read_parquet(data_dir / "features" / "features_asset_daily.parquet")
    labels = pl.read_parquet(data_dir / "labels" / "teacher_labels.parquet")
    splits = pl.read_parquet(data_dir / "splits" / "dataset_splits.parquet")
    manifest = json.loads((data_dir / "labels" / "label_manifest.json").read_text())
    ticker_order = manifest["ticker_order"]

    market_features = _try_read(data_dir / "features" / "features_market_daily.parquet")
    cross_asset_features = _try_read(data_dir / "features" / "features_cross_asset_daily.parquet")
    candidates = _try_read(data_dir / "labels" / "teacher_candidates.parquet")

    static_splits = splits.filter(~pl.col("split_name").str.starts_with("walkforward"))
    calendar_dates = sorted(prices["date"].unique().to_list())
    dataset = prepare_dataset(
        asset_features=asset_features,
        labels=labels,
        splits=static_splits,
        ticker_order=ticker_order,
        market_features=market_features,
        cross_asset_features=cross_asset_features,
        calendar_dates=calendar_dates,
    )

    _, returns_by_date = build_returns_by_date(prices, ticker_order)
    val_returns = align_returns_to_dates(
        returns_by_date,
        dataset.val_prediction_dates,
        len(ticker_order),
    )
    test_returns = align_returns_to_dates(
        returns_by_date,
        dataset.test_prediction_dates,
        len(ticker_order),
    )

    run_meta = train_direct_model(
        dataset=dataset,
        config=cfg,
        val_returns=val_returns,
        val_dates_list=dataset.val_prediction_dates,
        run_id=args.run_id,
        artifacts_dir=artifacts_dir,
    )
    run_dir = artifacts_dir / "runs" / run_meta["run_id"]
    model = load_direct_weight_model(run_dir / "model.bin")
    holdout = evaluate_holdout(
        model=model,
        dataset=dataset,
        test_returns=test_returns,
        config=cfg,
        output_dir=run_dir,
    )

    benchmark = None
    if not args.skip_benchmark:
        logger.info("Running benchmark suite...")
        benchmark = run_benchmark_suite(
            prices=prices,
            asset_features=asset_features,
            labels=labels,
            splits=static_splits,
            ticker_order=ticker_order,
            config=cfg,
            market_features=market_features,
            cross_asset_features=cross_asset_features,
            candidates=candidates,
            output_dir=run_dir / "benchmark",
        )

    _print_summary(run_dir, run_meta, holdout, benchmark)


def _print_summary(run_dir: Path, run_meta: dict, holdout: dict, benchmark: dict | None) -> None:
    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Run ID:         {run_meta['run_id']}")
    print(f"Model kind:     {run_meta['model_kind']}")
    print(f"Target mode:    {run_meta.get('target_mode', 'unknown')}")
    print(f"Features:       {run_meta['n_features']}")
    print(f"Best iteration: {run_meta.get('best_iteration')}")
    print(f"Artifacts:      {run_dir}")
    print("")
    print(f"{'Metric':<18} {'Holdout Net':>12} {'Holdout Gross':>14}")
    print("-" * 48)
    for metric in ["Sharpe", "CAGR", "Ann. Vol", "Max DD", "Avg Turnover", "HHI"]:
        net = holdout["model_metrics_net"][metric]
        gross = holdout["model_metrics_gross"][metric]
        if metric in {"CAGR", "Ann. Vol", "Max DD", "Avg Turnover"}:
            print(f"{metric:<18} {net:>11.2%} {gross:>13.2%}")
        else:
            print(f"{metric:<18} {net:>12.4f} {gross:>14.4f}")

    if benchmark is not None:
        top_row = benchmark["benchmark_table"][0]
        print("")
        print(f"Top benchmark:  {top_row['name']} (Sharpe {top_row['Sharpe']:.4f})")
        print(f"Benchmark dir:  {run_dir / 'benchmark'}")


def _try_read(path: Path) -> pl.DataFrame | None:
    if path.exists():
        return pl.read_parquet(path)
    return None


if __name__ == "__main__":
    main()
