"""Holdout test evaluation (spec 2.7)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from folio_trainer.backtest.metrics import PortfolioMetrics, compute_metrics, metrics_to_dict
from folio_trainer.backtest.simulator import BacktestResult, simulate
from folio_trainer.config.schema import PipelineConfig
from folio_trainer.models.dataset import PreparedDataset
from folio_trainer.models.direct_weight_gbm import DirectWeightGBM
from folio_trainer.models.losses import batch_kl_divergence

logger = logging.getLogger(__name__)


def evaluate_holdout(
    model: DirectWeightGBM,
    dataset: PreparedDataset,
    test_returns: np.ndarray,
    config: PipelineConfig,
    baseline_results: dict[str, BacktestResult] | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Evaluate model on holdout test split.

    Parameters
    ----------
    model
        Trained model.
    dataset
        Prepared dataset.
    test_returns
        (n_test_dates, n_assets) test period returns.
    config
        Pipeline config.
    baseline_results
        Dict of baseline name -> BacktestResult for comparison.
    output_dir
        Directory to write evaluation report.

    Returns
    -------
    dict with portfolio metrics, prediction metrics, and comparisons.
    """
    n_assets = len(dataset.ticker_names)

    # Predict weights on test set
    test_weights = model.predict_weights(dataset.X_test, dataset.test_date_indices)

    unique_test_dates = sorted(set(dataset.test_dates))
    n_dates = min(len(unique_test_dates), test_returns.shape[0])
    weight_matrix = np.zeros((n_dates, n_assets))

    for i, d_idx in enumerate(np.unique(dataset.test_date_indices)[:n_dates]):
        mask = dataset.test_date_indices == d_idx
        w = test_weights[mask]
        if len(w) >= n_assets:
            weight_matrix[i] = w[:n_assets]

    # Run backtest
    bt_result = simulate(
        weight_signals=weight_matrix,
        asset_returns=test_returns[:n_dates],
        dates=unique_test_dates[:n_dates],
        execution_config=config.execution,
        cost_config=config.cost_model,
        ticker_names=dataset.ticker_names,
    )

    # Compute metrics (gross and net)
    metrics_net = compute_metrics(bt_result, use_net=True)
    metrics_gross = compute_metrics(bt_result, use_net=False)

    # Prediction metrics
    pred_metrics = _compute_prediction_metrics(
        test_weights, dataset.y_test, dataset.test_date_indices, n_assets,
    )

    # Baseline comparison
    comparisons = {}
    if baseline_results:
        for name, bl_result in baseline_results.items():
            bl_metrics = compute_metrics(bl_result, use_net=True)
            comparisons[name] = metrics_to_dict(bl_metrics)

    result = {
        "model_metrics_net": metrics_to_dict(metrics_net),
        "model_metrics_gross": metrics_to_dict(metrics_gross),
        "prediction_metrics": pred_metrics,
        "baseline_comparisons": comparisons,
        "n_test_dates": n_dates,
    }

    # Write report
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_report(result, out / "eval_report.md")
        (out / "eval_results.json").write_text(json.dumps(result, indent=2, default=str))

        # Save test predictions
        import polars as pl
        pred_df = pl.DataFrame({
            "date": unique_test_dates[:n_dates],
            **{f"w_{t}": weight_matrix[:, j] for j, t in enumerate(dataset.ticker_names)},
        })
        pred_df.write_parquet(out / "predictions_test.parquet")

    return result


def _compute_prediction_metrics(
    pred_weights: np.ndarray,
    target_weights: np.ndarray,
    date_indices: np.ndarray,
    n_assets: int,
) -> dict:
    """Compute prediction-level metrics (spec 2.7.2)."""
    unique_dates = np.unique(date_indices)
    n_dates = len(unique_dates)

    pred_matrix = np.zeros((n_dates, n_assets))
    target_matrix = np.zeros((n_dates, n_assets))

    for i, d_idx in enumerate(unique_dates):
        mask = date_indices == d_idx
        p = pred_weights[mask]
        t = target_weights[mask]
        if len(p) >= n_assets:
            pred_matrix[i] = p[:n_assets]
            target_matrix[i] = t[:n_assets]

    # KL divergence
    kl = batch_kl_divergence(pred_matrix, target_matrix)

    # MAE and RMSE
    mae = float(np.mean(np.abs(pred_matrix - target_matrix)))
    rmse = float(np.sqrt(np.mean((pred_matrix - target_matrix) ** 2)))

    # Weight stability (day-to-day L1 change)
    if n_dates > 1:
        daily_changes = np.sum(np.abs(np.diff(pred_matrix, axis=0)), axis=1)
        stability = float(np.mean(daily_changes))
    else:
        stability = 0.0

    return {
        "kl_divergence": kl,
        "mae": mae,
        "rmse": rmse,
        "weight_stability": stability,
    }


def _write_report(results: dict, path: Path) -> None:
    """Write evaluation report as markdown."""
    lines = ["# Holdout Evaluation Report\n"]

    lines.append("## Model Performance (Net of Costs)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in results["model_metrics_net"].items():
        lines.append(f"| {k} | {v:.4f} |")

    lines.append("\n## Model Performance (Gross)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in results["model_metrics_gross"].items():
        lines.append(f"| {k} | {v:.4f} |")

    lines.append("\n## Prediction Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in results["prediction_metrics"].items():
        lines.append(f"| {k} | {v:.6f} |")

    if results.get("baseline_comparisons"):
        lines.append("\n## Baseline Comparisons\n")
        baselines = results["baseline_comparisons"]
        headers = list(next(iter(baselines.values())).keys()) if baselines else []
        lines.append("| Strategy | " + " | ".join(headers) + " |")
        lines.append("|----------|" + "|".join(["-------"] * len(headers)) + "|")

        lines.append("| **Model (net)** | " + " | ".join(
            f"{results['model_metrics_net'].get(h, 0):.4f}" for h in headers
        ) + " |")

        for name, metrics in baselines.items():
            lines.append(f"| {name} | " + " | ".join(
                f"{metrics.get(h, 0):.4f}" for h in headers
            ) + " |")

    path.write_text("\n".join(lines))
    logger.info("Report written to %s", path)
