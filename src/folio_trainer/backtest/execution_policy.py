"""Execution policy for backtests (spec 2.6.1)."""

from __future__ import annotations

import numpy as np

from folio_trainer.config.schema import ExecutionConfig


def apply_execution_policy(
    pred_target: np.ndarray,
    prev_live_weights: np.ndarray,
    config: ExecutionConfig,
) -> tuple[np.ndarray, bool]:
    """Apply rebalance band and partial rebalance to target weights.

    Parameters
    ----------
    pred_target
        Model's predicted target weights (n_assets,).
    prev_live_weights
        Current live portfolio weights (n_assets,).
    config
        Execution policy configuration.

    Returns
    -------
    exec_weights
        Executed portfolio weights after applying policy.
    traded
        Whether a trade was executed.
    """
    l1_change = 0.5 * np.sum(np.abs(pred_target - prev_live_weights))

    if l1_change < config.rebalance_band:
        return prev_live_weights.copy(), False

    # Partial rebalance toward target
    exec_weights = (
        prev_live_weights
        + config.partial_rebalance_alpha * (pred_target - prev_live_weights)
    )

    # Ensure on simplex
    exec_weights = np.maximum(exec_weights, 0.0)
    total = exec_weights.sum()
    if total > 0:
        exec_weights = exec_weights / total

    return exec_weights, True


def apply_drift(
    weights: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """Apply market drift to portfolio weights based on realized returns.

    Parameters
    ----------
    weights
        Current portfolio weights (n_assets,).
    returns
        Next-period simple returns (n_assets,).

    Returns
    -------
    drifted_weights
        Weights after market drift.
    """
    drifted = weights * (1 + returns)
    total = drifted.sum()
    if total > 0:
        return drifted / total
    return weights.copy()
