"""Training loop for the direct weight prediction model (spec 2.5)."""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from pathlib import Path

import numpy as np
import yaml

from folio_trainer.backtest.metrics import compute_metrics
from folio_trainer.backtest.simulator import simulate
from folio_trainer.config.schema import PipelineConfig
from folio_trainer.models.dataset import PreparedDataset, save_preprocessing
from folio_trainer.models.direct_weight_gbm import DirectWeightGBM

logger = logging.getLogger(__name__)


def train_direct_model(
    dataset: PreparedDataset,
    config: PipelineConfig,
    val_returns: np.ndarray | None = None,
    val_dates_list: list | None = None,
    run_id: str | None = None,
    artifacts_dir: str | Path = "artifacts",
) -> dict:
    """Train the direct weight GBM model.

    Parameters
    ----------
    dataset
        Prepared dataset with train/val/test splits.
    config
        Full pipeline config.
    val_returns
        (n_val_days, n_assets) returns for validation backtest.
        If provided, model selection uses validation net Sharpe.
    val_dates_list
        List of validation dates for backtest.
    run_id
        Custom run identifier. Auto-generated if None.
    artifacts_dir
        Directory for saving model artifacts.

    Returns
    -------
    dict with run metadata, metrics, and artifact paths.
    """
    run_id = run_id or f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(artifacts_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_config = config.model.direct_weight_model
    model = DirectWeightGBM(model_config)

    logger.info("Training GBM model (run_id=%s)...", run_id)

    # Train
    train_meta = model.train(
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        w_train=dataset.w_train,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
        w_val=dataset.w_val,
        feature_names=dataset.feature_names,
    )

    logger.info("Training complete. Best iteration: %d", train_meta.get("best_iteration", -1))

    # Compute validation metrics
    val_metrics = {}
    if val_returns is not None and val_dates_list is not None:
        val_weights = model.predict_weights(dataset.X_val, dataset.val_date_indices)

        # Reshape to (n_dates, n_assets) for backtest
        n_assets = len(dataset.ticker_names)
        unique_val_dates = sorted(set(dataset.val_dates))
        n_val_dates = len(unique_val_dates)
        weight_matrix = np.zeros((n_val_dates, n_assets))

        for i, d_idx in enumerate(np.unique(dataset.val_date_indices)):
            mask = dataset.val_date_indices == d_idx
            weight_matrix[i] = val_weights[mask][:n_assets]

        # Run backtest on validation
        bt_result = simulate(
            weight_signals=weight_matrix,
            asset_returns=val_returns[:n_val_dates],
            dates=unique_val_dates[:n_val_dates],
            execution_config=config.execution,
            cost_config=config.cost_model,
        )
        metrics = compute_metrics(bt_result, use_net=True)
        val_metrics = {
            "validation_net_sharpe": metrics.sharpe,
            "validation_max_drawdown": metrics.max_drawdown,
            "validation_turnover": metrics.avg_turnover,
        }
        logger.info("Validation Sharpe: %.4f, MaxDD: %.4f", metrics.sharpe, metrics.max_drawdown)

    # Feature importance
    importance = model.feature_importance()
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:20]

    # Save artifacts
    model.save(run_dir / "model.bin")
    save_preprocessing(dataset.preprocessing, run_dir / "preprocessor.pkl")

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)

    # Save feature manifest
    manifest = {
        "feature_names": dataset.feature_names,
        "n_features": len(dataset.feature_names),
        "ticker_order": dataset.ticker_names,
    }
    (run_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Save run metadata
    run_meta = {
        "run_id": run_id,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_kind": model_config.kind,
        "best_iteration": train_meta.get("best_iteration"),
        "n_train_rows": len(dataset.X_train),
        "n_val_rows": len(dataset.X_val),
        "n_features": len(dataset.feature_names),
        "random_seed": config.random_seed,
        "val_metrics": val_metrics,
        "top_features": top_features[:10],
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    logger.info("Artifacts saved to %s", run_dir)
    return run_meta
