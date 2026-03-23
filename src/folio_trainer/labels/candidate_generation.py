"""Candidate portfolio generation (spec 1.8.3)."""

from __future__ import annotations

import numpy as np

from folio_trainer.config.schema import CandidateSearchConfig


def generate_candidates(
    n_assets: int,
    weight_caps: np.ndarray,
    prev_live_weights: np.ndarray,
    prev_target_weights: np.ndarray | None,
    rolling_vol: np.ndarray | None,
    rolling_cov: np.ndarray | None,
    config: CandidateSearchConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate candidate portfolio weight vectors.

    Parameters
    ----------
    n_assets
        Number of assets in universe.
    weight_caps
        Maximum weight per asset (n_assets,).
    prev_live_weights
        Current live weights (n_assets,).
    prev_target_weights
        Previous target weights, or None for first day.
    rolling_vol
        Rolling volatility per asset, for inverse-vol and min-var candidates.
    rolling_cov
        Rolling covariance matrix (n_assets, n_assets) for min-var candidate.
    config
        Candidate search configuration.
    rng
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Candidate weight matrix (n_candidates, n_assets), each row on the simplex.
    """
    candidates: list[np.ndarray] = []

    # 1. Deterministic candidates
    for cand_type in config.deterministic_candidates:
        w = _make_deterministic(
            cand_type, n_assets, prev_live_weights, prev_target_weights,
            rolling_vol, rolling_cov,
        )
        if w is not None:
            w = project_to_simplex(w, weight_caps)
            candidates.append(w)

    # 2. Sparse Dirichlet random candidates
    # Instead of spreading weights across all assets, randomly select a subset
    # of K assets per candidate and allocate weights only among those.
    sparse_k = min(config.sparse_k_assets, n_assets)
    n_per_alpha = config.dirichlet_candidates_per_day // len(config.dirichlet_alpha_mix)
    for alpha_val in config.dirichlet_alpha_mix:
        for _ in range(n_per_alpha):
            chosen = rng.choice(n_assets, size=sparse_k, replace=False)
            alpha = np.full(sparse_k, alpha_val)
            sub_weights = rng.dirichlet(alpha)
            w = np.zeros(n_assets)
            w[chosen] = sub_weights
            candidates.append(project_to_simplex(w, weight_caps))

    # 3. Local perturbations around top deterministic seeds
    det_candidates = candidates[: len(config.deterministic_candidates)]
    n_seeds = min(5, len(det_candidates))
    for seed_w in det_candidates[:n_seeds]:
        for _ in range(config.local_perturbations_per_seed):
            noise = rng.normal(0, 0.02, size=n_assets)
            perturbed = seed_w + noise
            perturbed = project_to_simplex(perturbed, weight_caps)
            candidates.append(perturbed)

    return np.array(candidates)


def _make_deterministic(
    cand_type: str,
    n_assets: int,
    prev_live: np.ndarray,
    prev_target: np.ndarray | None,
    rolling_vol: np.ndarray | None,
    rolling_cov: np.ndarray | None,
) -> np.ndarray | None:
    """Create a deterministic candidate portfolio."""
    if cand_type == "equal_weight":
        return np.full(n_assets, 1.0 / n_assets)

    if cand_type == "cash_only":
        w = np.zeros(n_assets)
        w[-1] = 1.0  # CASH is last asset by convention
        return w

    if cand_type == "prev_live":
        return prev_live.copy()

    if cand_type == "prev_target":
        if prev_target is not None:
            return prev_target.copy()
        return prev_live.copy()

    if cand_type == "inverse_vol":
        if rolling_vol is not None:
            vol = np.maximum(rolling_vol, 1e-8)
            inv_vol = 1.0 / vol
            return inv_vol / inv_vol.sum()
        return np.full(n_assets, 1.0 / n_assets)

    if cand_type == "min_variance":
        if rolling_cov is not None:
            return _min_variance_weights(rolling_cov)
        return np.full(n_assets, 1.0 / n_assets)

    return None


def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Approximate minimum variance portfolio using closed-form solution."""
    try:
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-6  # regularize
        inv_cov = np.linalg.inv(cov_reg)
        ones = np.ones(cov.shape[0])
        w = inv_cov @ ones
        w = w / w.sum()
        # Clip negative weights for long-only
        w = np.maximum(w, 0.0)
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.full(len(w), 1.0 / len(w))
        return w
    except np.linalg.LinAlgError:
        return np.full(cov.shape[0], 1.0 / cov.shape[0])


def project_to_simplex(w: np.ndarray, weight_caps: np.ndarray, max_iter: int = 10) -> np.ndarray:
    """Project weights onto the constrained simplex.

    Constraints: w >= 0, sum(w) = 1, w <= weight_caps.
    Uses iterative clipping and renormalization.
    """
    w = np.maximum(w, 0.0)

    for _ in range(max_iter):
        # Clip to caps
        excess = np.maximum(w - weight_caps, 0.0)
        if excess.sum() < 1e-10:
            break
        w = np.minimum(w, weight_caps)
        # Redistribute excess to uncapped assets
        uncapped_mask = w < weight_caps - 1e-10
        if uncapped_mask.sum() == 0:
            break
        redistribute = excess.sum() / uncapped_mask.sum()
        w[uncapped_mask] += redistribute

    # Final normalization
    total = w.sum()
    if total > 0:
        w = w / total
    else:
        w = np.full(len(w), 1.0 / len(w))

    # One more cap clip + renorm
    w = np.minimum(w, weight_caps)
    w = w / w.sum()

    return w
