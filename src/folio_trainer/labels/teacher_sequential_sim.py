"""Sequential teacher label simulation (spec 1.8.6).

This is the path-dependent core: each day's prev_live_weights depends on
the prior day's chosen target and market drift.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from folio_trainer.config.schema import (
    CandidateSearchConfig,
    CostModelConfig,
    ExecutionConfig,
    TeacherObjectiveConfig,
)
from folio_trainer.labels.candidate_generation import generate_candidates
from folio_trainer.labels.distill_labels import distill_top_k
from folio_trainer.labels.teacher_scoring import score_candidates

logger = logging.getLogger(__name__)


def run_sequential_simulation(
    trading_dates: list[dt.date],
    returns_matrix: np.ndarray,
    rf_returns: np.ndarray,
    adv20_matrix: np.ndarray,
    rolling_vol_matrix: np.ndarray | None,
    rolling_cov_fn: callable | None,
    weight_caps: np.ndarray,
    n_assets: int,
    horizon: int,
    candidate_config: CandidateSearchConfig,
    objective_config: TeacherObjectiveConfig,
    cost_config: CostModelConfig,
    execution_config: ExecutionConfig,
    random_seed: int = 42,
    portfolio_value: float = 1_000_000.0,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the sequential teacher label simulation.

    Parameters
    ----------
    trading_dates
        Sorted list of trading dates.
    returns_matrix
        (n_dates, n_assets) daily simple returns aligned to trading_dates.
    rf_returns
        (n_dates,) daily risk-free returns.
    adv20_matrix
        (n_dates, n_assets) 20-day average dollar volume.
    rolling_vol_matrix
        (n_dates, n_assets) rolling volatility, or None.
    rolling_cov_fn
        Callable(date_idx) -> (n_assets, n_assets) covariance matrix, or None.
    weight_caps
        (n_assets,) max weight per asset.
    n_assets
        Number of assets.
    horizon
        Label horizon in trading days.
    candidate_config, objective_config, cost_config, execution_config
        Configuration objects.
    random_seed
        Random seed for reproducibility.
    portfolio_value
        Notional portfolio value for cost estimation.
    checkpoint_dir
        Directory to save/load checkpoints.
    resume
        If True, attempt to resume from last checkpoint.

    Returns
    -------
    teacher_labels
        DataFrame with distilled labels per date.
    teacher_candidates
        DataFrame with all scored candidates per date.
    """
    rng = np.random.default_rng(random_seed)
    n_dates = len(trading_dates)

    # Initialize state
    start_idx = 0
    prev_live_weights = np.full(n_assets, 1.0 / n_assets)
    prev_target_weights = None

    # Attempt resume
    if resume and checkpoint_dir:
        state = _load_checkpoint(checkpoint_dir)
        if state is not None:
            start_idx = state["last_completed_idx"] + 1
            prev_live_weights = np.array(state["prev_live_weights"])
            prev_target_weights = np.array(state["prev_target_weights"]) if state.get("prev_target_weights") else None
            logger.info("Resuming from date index %d (%s)", start_idx, trading_dates[start_idx])

    label_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for i in range(start_idx, n_dates):
        date = trading_dates[i]

        # Check if we have enough forward data for the horizon
        if i + horizon >= n_dates:
            logger.info("Stopping at %s: insufficient forward data for horizon %d.", date, horizon)
            break

        # Forward returns for scoring
        forward_rets = returns_matrix[i + 1 : i + 1 + horizon]  # (horizon, n_assets)
        rf_fwd = rf_returns[i + 1 : i + 1 + horizon]  # (horizon,)

        # Current ADV and vol
        adv20 = adv20_matrix[i] if adv20_matrix is not None else np.full(n_assets, 1e8)
        rolling_vol = rolling_vol_matrix[i] if rolling_vol_matrix is not None else None
        rolling_cov = rolling_cov_fn(i) if rolling_cov_fn is not None else None

        # Generate candidates
        candidates = generate_candidates(
            n_assets=n_assets,
            weight_caps=weight_caps,
            prev_live_weights=prev_live_weights,
            prev_target_weights=prev_target_weights,
            rolling_vol=rolling_vol,
            rolling_cov=rolling_cov,
            config=candidate_config,
            rng=rng,
        )

        # Score candidates
        scores = score_candidates(
            candidates=candidates,
            forward_returns=forward_rets,
            rf_returns=rf_fwd,
            prev_live_weights=prev_live_weights,
            adv20=adv20,
            portfolio_value=portfolio_value,
            objective_config=objective_config,
            cost_config=cost_config,
        )

        # Rank candidates
        ranking = np.argsort(-scores["objective_total"])

        # Store candidate rows (top candidates for the record)
        for rank_pos, cand_idx in enumerate(ranking[:candidate_config.top_k * 2]):
            candidate_rows.append({
                "asof_date": date,
                "prediction_date": trading_dates[i + 1],
                "horizon_h": horizon,
                "candidate_id": f"{date}_{cand_idx}",
                "candidate_type": _get_candidate_type(cand_idx, len(candidate_config.deterministic_candidates)),
                "weights_json": json.dumps(candidates[cand_idx].tolist()),
                "objective_total": float(scores["objective_total"][cand_idx]),
                "objective_sharpe": float(scores["objective_sharpe"][cand_idx]),
                "objective_return": float(scores["objective_return"][cand_idx]),
                "objective_vol": float(scores["objective_vol"][cand_idx]),
                "turnover": float(scores["turnover"][cand_idx]),
                "est_cost": float(scores["est_cost"][cand_idx]),
                "concentration_hhi": float(scores["concentration_hhi"][cand_idx]),
                "rank": rank_pos,
                "rank_pct": rank_pos / len(ranking),
            })

        # Distill labels from top-K
        top_k_indices = ranking[:candidate_config.top_k]
        top_k_weights = candidates[top_k_indices]
        top_k_objectives = scores["objective_total"][top_k_indices]

        hard_target = candidates[ranking[0]]
        soft_target, confidence = distill_top_k(
            top_k_weights, top_k_objectives,
            temperature=candidate_config.distillation_temperature,
        )

        label_rows.append({
            "asof_date": date,
            "prediction_date": trading_dates[i + 1],
            "horizon_h": horizon,
            "prev_live_weights_json": json.dumps(prev_live_weights.tolist()),
            "prev_target_weights_json": json.dumps(
                (prev_target_weights if prev_target_weights is not None else prev_live_weights).tolist()
            ),
            "hard_target_weights_json": json.dumps(hard_target.tolist()),
            "soft_target_weights_json": json.dumps(soft_target.tolist()),
            "teacher_confidence": float(confidence),
            "best_objective": float(scores["objective_total"][ranking[0]]),
            "topk_mean_objective": float(np.mean(top_k_objectives)),
            "num_candidates": len(candidates),
        })

        # Update state: apply trade and drift
        prev_target_weights = hard_target.copy()

        # Partial rebalance execution
        l1_change = 0.5 * np.sum(np.abs(hard_target - prev_live_weights))
        if l1_change >= execution_config.rebalance_band:
            exec_weights = (
                prev_live_weights
                + execution_config.partial_rebalance_alpha * (hard_target - prev_live_weights)
            )
            # Normalize
            exec_weights = np.maximum(exec_weights, 0.0)
            exec_weights = exec_weights / exec_weights.sum()
        else:
            exec_weights = prev_live_weights.copy()

        # Apply market drift for next day
        next_day_returns = returns_matrix[i + 1]  # (n_assets,)
        drifted = exec_weights * (1 + next_day_returns)
        total = drifted.sum()
        if total > 0:
            prev_live_weights = drifted / total
        else:
            prev_live_weights = exec_weights.copy()

        # Checkpoint periodically
        if checkpoint_dir and (i % 50 == 0 or i == n_dates - 1):
            _save_checkpoint(checkpoint_dir, i, prev_live_weights, prev_target_weights)

        if (i - start_idx) % 100 == 0:
            logger.info(
                "Processed %d/%d dates (current: %s, best_obj: %.4f)",
                i - start_idx + 1, n_dates - start_idx, date,
                scores["objective_total"][ranking[0]],
            )

    teacher_labels = pl.DataFrame(label_rows) if label_rows else pl.DataFrame()
    teacher_candidates = pl.DataFrame(candidate_rows) if candidate_rows else pl.DataFrame()

    return teacher_labels, teacher_candidates


def _get_candidate_type(cand_idx: int, n_deterministic: int) -> str:
    """Classify candidate by index into generation method."""
    if cand_idx < n_deterministic:
        return "deterministic"
    return "random"


def _save_checkpoint(
    checkpoint_dir: str | Path, idx: int,
    prev_live: np.ndarray, prev_target: np.ndarray | None,
) -> None:
    """Save simulation state to JSON checkpoint."""
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "last_completed_idx": idx,
        "prev_live_weights": prev_live.tolist(),
        "prev_target_weights": prev_target.tolist() if prev_target is not None else None,
    }
    (path / "checkpoint.json").write_text(json.dumps(state))


def _load_checkpoint(checkpoint_dir: str | Path) -> dict | None:
    """Load simulation state from checkpoint."""
    path = Path(checkpoint_dir) / "checkpoint.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, KeyError):
        logger.warning("Invalid checkpoint at %s, starting fresh.", path)
        return None
