"""Baseline strategy implementations (spec 2.2)."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from folio_trainer.labels.candidate_generation import _min_variance_weights


def equal_weight_strategy(
    n_assets: int,
    n_days: int,
    **kwargs,
) -> np.ndarray:
    """Equal weight: 1/N for all assets."""
    return np.full((n_days, n_assets), 1.0 / n_assets)


def previous_hold_strategy(
    n_assets: int,
    n_days: int,
    initial_weights: np.ndarray | None = None,
    asset_returns: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Previous-live-weights hold: never rebalance, just drift."""
    weights = np.zeros((n_days, n_assets))
    w = initial_weights if initial_weights is not None else np.full(n_assets, 1.0 / n_assets)
    for t in range(n_days):
        weights[t] = w.copy()
        if asset_returns is not None:
            drifted = w * (1 + asset_returns[t])
            total = drifted.sum()
            if total > 0:
                w = drifted / total
    return weights


def inverse_volatility_strategy(
    n_assets: int,
    n_days: int,
    rolling_vol: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Inverse volatility / risk parity approximation."""
    weights = np.zeros((n_days, n_assets))
    for t in range(n_days):
        vol = np.maximum(rolling_vol[t], 1e-8)
        inv_vol = 1.0 / vol
        weights[t] = inv_vol / inv_vol.sum()
    return weights


def min_variance_strategy(
    n_assets: int,
    n_days: int,
    rolling_cov_fn: callable,
    **kwargs,
) -> np.ndarray:
    """Minimum variance approximation using rolling covariance."""
    weights = np.zeros((n_days, n_assets))
    for t in range(n_days):
        cov = rolling_cov_fn(t)
        if cov is not None:
            weights[t] = _min_variance_weights(cov)
        else:
            weights[t] = np.full(n_assets, 1.0 / n_assets)
    return weights


def teacher_replay_strategy(
    n_days: int,
    teacher_labels: pl.DataFrame,
    trading_dates: list,
    **kwargs,
) -> np.ndarray:
    """Replay teacher hard-target labels (oracle upper bound)."""
    date_to_weights = {}
    for row in teacher_labels.iter_rows(named=True):
        w = json.loads(row["hard_target_weights_json"])
        date_to_weights[row["asof_date"]] = np.array(w)

    n_assets = len(next(iter(date_to_weights.values()))) if date_to_weights else 1
    weights = np.zeros((n_days, n_assets))

    for t, date in enumerate(trading_dates[:n_days]):
        if date in date_to_weights:
            weights[t] = date_to_weights[date]
        elif t > 0:
            weights[t] = weights[t - 1]
        else:
            weights[t] = np.full(n_assets, 1.0 / n_assets)

    return weights
