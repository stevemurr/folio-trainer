"""Training loop for the utility scorer model."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from folio_trainer.models.utility_scorer import UtilityScorer

logger = logging.getLogger(__name__)

CANDIDATE_STATE_FEATURES = (
    "turnover",
    "est_cost",
    "concentration_hhi",
    "candidate_cash_weight",
    "candidate_max_weight",
    "candidate_active_positions",
)


def build_shared_feature_table(
    market_features: pl.DataFrame | None = None,
    cross_asset_features: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Join shared market/cross-asset features on asof_date."""
    frames = [f for f in (market_features, cross_asset_features) if f is not None and len(f) > 0]
    if not frames:
        return pl.DataFrame(schema={"asof_date": pl.Date})

    shared = frames[0]
    for frame in frames[1:]:
        shared = shared.join(frame, on="asof_date", how="outer_coalesce")
    return shared.sort("asof_date")


def prepare_utility_dataset(
    candidates: pl.DataFrame,
    shared_features: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build utility-model tensors from current-state features and candidates."""
    if len(candidates) == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            [],
        )

    feature_cols = [c for c in shared_features.columns if c != "asof_date"]
    joined = candidates.join(shared_features, on="asof_date", how="left")

    rows_state: list[list[float]] = []
    rows_weights: list[np.ndarray] = []
    objectives: list[float] = []
    group_indices: list[int] = []
    date_map: dict = {}

    for row in joined.iter_rows(named=True):
        asof_date = row["asof_date"]
        if asof_date not in date_map:
            date_map[asof_date] = len(date_map)

        weights = np.asarray(json.loads(row["weights_json"]), dtype=np.float32)
        candidate_state = [
            float(row.get("turnover") or 0.0),
            float(row.get("est_cost") or 0.0),
            float(row.get("concentration_hhi") or 0.0),
            float(weights[-1]) if len(weights) else 0.0,
            float(weights.max()) if len(weights) else 0.0,
            float(np.sum(weights > 0.01)),
        ]
        shared_state = [
            float(row.get(col) or 0.0)
            for col in feature_cols
        ]

        rows_state.append(shared_state + candidate_state)
        rows_weights.append(weights)
        objectives.append(float(row["objective_total"]))
        group_indices.append(date_map[asof_date])

    return (
        np.asarray(rows_state, dtype=np.float32),
        np.asarray(rows_weights, dtype=np.float32),
        np.asarray(objectives, dtype=np.float32),
        np.asarray(group_indices, dtype=np.int32),
        feature_cols + list(CANDIDATE_STATE_FEATURES),
    )


def train_utility_model(
    candidates: pl.DataFrame,
    shared_features: pl.DataFrame,
    splits: pl.DataFrame,
    output_dir: str | Path,
) -> dict:
    """Train the utility scorer model."""
    from folio_trainer.splits.make_splits import get_split_dates

    train_start, train_end = get_split_dates(splits, "train")
    val_start, val_end = get_split_dates(splits, "val")

    train_cands = candidates.filter(
        (pl.col("asof_date") >= train_start) & (pl.col("asof_date") <= train_end)
    )
    val_cands = candidates.filter(
        (pl.col("asof_date") >= val_start) & (pl.col("asof_date") <= val_end)
    )

    train_state, train_weights, train_obj, train_groups, state_feature_names = prepare_utility_dataset(
        train_cands, shared_features
    )
    val_state, val_weights, val_obj, val_groups, _ = prepare_utility_dataset(
        val_cands, shared_features
    )

    if len(train_obj) == 0:
        logger.warning("No training candidates found.")
        return {"error": "No training data"}

    model = UtilityScorer()
    train_meta = model.train(
        train_state,
        train_weights,
        train_obj,
        val_state,
        val_weights,
        val_obj,
    )

    eval_metrics = model.evaluate(val_state, val_weights, val_obj, val_groups)
    logger.info(
        "Utility scorer: Spearman=%.4f, Kendall=%.4f",
        eval_metrics["spearman"],
        eval_metrics["kendall"],
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "utility_scorer.bin")
    (out / "utility_feature_manifest.json").write_text(
        json.dumps({"state_feature_names": state_feature_names}, indent=2)
    )

    results = {**train_meta, **eval_metrics}
    (out / "utility_scorer_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results
