"""Loss functions for weight prediction models (spec 2.3.3)."""

from __future__ import annotations

import numpy as np


def kl_divergence(pred_weights: np.ndarray, target_weights: np.ndarray, eps: float = 1e-8) -> float:
    """KL divergence: KL(target || pred).

    Parameters
    ----------
    pred_weights
        (n_assets,) predicted weight vector.
    target_weights
        (n_assets,) target weight vector.
    eps
        Small constant to avoid log(0).

    Returns
    -------
    KL divergence value.
    """
    p = np.clip(target_weights, eps, 1.0)
    q = np.clip(pred_weights, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def cross_entropy(pred_weights: np.ndarray, target_weights: np.ndarray, eps: float = 1e-8) -> float:
    """Cross-entropy: -sum(target * log(pred)).

    Parameters
    ----------
    pred_weights
        (n_assets,) predicted weight vector.
    target_weights
        (n_assets,) target weight vector.
    eps
        Small constant.

    Returns
    -------
    Cross-entropy value.
    """
    q = np.clip(pred_weights, eps, 1.0)
    return float(-np.sum(target_weights * np.log(q)))


def weight_change_penalty(
    current_weights: np.ndarray,
    previous_weights: np.ndarray,
) -> float:
    """L2 penalty on day-to-day weight changes."""
    return float(np.sum((current_weights - previous_weights) ** 2))


def batch_kl_divergence(
    pred_weights: np.ndarray,
    target_weights: np.ndarray,
    sample_weights: np.ndarray | None = None,
    eps: float = 1e-8,
) -> float:
    """Batch KL divergence across multiple date cross-sections.

    Parameters
    ----------
    pred_weights
        (n_dates, n_assets) predicted weights.
    target_weights
        (n_dates, n_assets) target weights.
    sample_weights
        (n_dates,) per-sample confidence weights. If None, uniform.
    eps
        Small constant.

    Returns
    -------
    Weighted average KL divergence.
    """
    p = np.clip(target_weights, eps, 1.0)
    q = np.clip(pred_weights, eps, 1.0)
    kl_per_sample = np.sum(p * np.log(p / q), axis=1)  # (n_dates,)

    if sample_weights is not None:
        return float(np.average(kl_per_sample, weights=sample_weights))
    return float(np.mean(kl_per_sample))
