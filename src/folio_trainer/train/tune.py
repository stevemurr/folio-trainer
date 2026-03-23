"""Hyperparameter tuning with Optuna (spec 2.5.2)."""

from __future__ import annotations

import logging

import numpy as np
import optuna

from folio_trainer.backtest.metrics import compute_metrics
from folio_trainer.backtest.simulator import simulate
from folio_trainer.config.schema import DirectWeightModelConfig, ExecutionConfig, CostModelConfig
from folio_trainer.models.dataset import PreparedDataset, stack_by_date
from folio_trainer.models.factory import create_direct_weight_model

logger = logging.getLogger(__name__)


def tune_direct_model(
    dataset: PreparedDataset,
    val_returns: np.ndarray,
    val_dates: list,
    execution_config: ExecutionConfig,
    cost_config: CostModelConfig,
    n_trials: int = 50,
    random_seed: int = 42,
) -> dict:
    """Tune hyperparameters for the GBM direct weight model.

    Uses Optuna with time-series-safe validation (already split).
    Objective: maximize validation net Sharpe.

    Parameters
    ----------
    dataset
        Prepared dataset.
    val_returns
        (n_val_dates, n_assets) validation period returns.
    val_dates
        Validation date list.
    execution_config
        Execution policy config.
    cost_config
        Cost model config.
    n_trials
        Number of Optuna trials.
    random_seed
        Random seed.

    Returns
    -------
    dict with best_params and best_sharpe.
    """

    def objective(trial: optuna.Trial) -> float:
        model_config = DirectWeightModelConfig(
            kind="gbm",
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            num_leaves=trial.suggest_int("num_leaves", 16, 128),
            n_estimators=500,  # early stopping handles this
            temperature=trial.suggest_float("temperature", 0.1, 5.0),
            l2_weight_change_penalty=trial.suggest_float("l2_weight_change", 0.001, 0.1, log=True),
            use_confidence_weights=True,
        )

        model = create_direct_weight_model(model_config)

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
            logger.warning("Trial %d failed during training: %s", trial.number, e)
            return float("-inf")

        val_weights = model.predict_weights(dataset.X_val, dataset.val_date_indices)
        weight_matrix = stack_by_date(
            val_weights,
            dataset.val_date_indices,
            len(dataset.ticker_names),
        )
        n_dates = min(len(dataset.val_prediction_dates), val_returns.shape[0])
        weight_matrix = weight_matrix[:n_dates]

        bt_result = simulate(
            weight_signals=weight_matrix,
            asset_returns=val_returns[:n_dates],
            dates=dataset.val_prediction_dates[:n_dates],
            execution_config=execution_config,
            cost_config=cost_config,
        )
        metrics = compute_metrics(bt_result, use_net=True)

        return metrics.sharpe

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(
        "Best trial: Sharpe=%.4f, params=%s",
        study.best_value,
        study.best_params,
    )

    return {
        "best_params": study.best_params,
        "best_sharpe": study.best_value,
        "n_trials": n_trials,
    }
