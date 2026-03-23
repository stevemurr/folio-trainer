"""Soft-target distillation from top-K candidates (spec 1.8.7)."""

from __future__ import annotations

import numpy as np


def distill_top_k(
    top_k_weights: np.ndarray,
    top_k_objectives: np.ndarray,
    temperature: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Distill soft target weights from top-K candidates.

    Parameters
    ----------
    top_k_weights
        (K, n_assets) weight vectors of top-K candidates.
    top_k_objectives
        (K,) objective scores of top-K candidates.
    temperature
        Softmax temperature for blending candidates.

    Returns
    -------
    soft_target
        (n_assets,) blended weight vector.
    confidence
        Teacher confidence = best_objective - worst_in_top_k.
    """
    k = len(top_k_objectives)
    if k == 0:
        raise ValueError("No candidates to distill.")

    if k == 1:
        return top_k_weights[0].copy(), 0.0

    # Softmax over objectives
    scaled = top_k_objectives / max(temperature, 1e-10)
    # Numerical stability: subtract max
    scaled = scaled - scaled.max()
    exp_scaled = np.exp(scaled)
    probs = exp_scaled / exp_scaled.sum()  # (K,)

    # Weighted average of weight vectors
    soft_target = probs @ top_k_weights  # (n_assets,)

    # Ensure on simplex
    soft_target = np.maximum(soft_target, 0.0)
    total = soft_target.sum()
    if total > 0:
        soft_target = soft_target / total

    # Confidence: spread between best and worst in top-K
    confidence = float(top_k_objectives[0] - top_k_objectives[-1])

    return soft_target, confidence


def masked_softmax(
    scores: np.ndarray,
    mask: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert raw scores to weights via masked softmax.

    Shared utility used by both label distillation and model inference.

    Parameters
    ----------
    scores
        (n_assets,) raw scores.
    mask
        (n_assets,) binary mask (1 = available, 0 = unavailable).
    temperature
        Softmax temperature.

    Returns
    -------
    weights
        (n_assets,) weight vector on the simplex.
    """
    masked_scores = np.where(mask > 0, scores / max(temperature, 1e-10), -1e9)
    masked_scores = masked_scores - masked_scores.max()
    exp_scores = np.exp(masked_scores) * mask
    total = exp_scores.sum()
    if total > 0:
        return exp_scores / total
    # Fallback: equal weight among available assets
    n_available = mask.sum()
    if n_available > 0:
        return mask / n_available
    return np.zeros_like(scores)
