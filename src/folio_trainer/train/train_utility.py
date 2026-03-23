"""Training loop for the utility scorer model."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from folio_trainer.models.utility_scorer import UtilityScorer

logger = logging.getLogger(__name__)


def train_utility_model(
    candidates: pl.DataFrame,
    features: pl.DataFrame,
    splits: pl.DataFrame,
    ticker_order: list[str],
    output_dir: str | Path,
) -> dict:
    """Train the utility scorer model.

    Parameters
    ----------
    candidates
        Teacher candidates table with weights_json and objective_total.
    features
        Market feature table keyed by asof_date.
    splits
        Split metadata.
    ticker_order
        Asset order for weight vector alignment.
    output_dir
        Directory for artifacts.

    Returns
    -------
    dict with training results and ranking metrics.
    """
    from folio_trainer.splits.make_splits import get_split_dates

    train_start, train_end = get_split_dates(splits, "train")
    val_start, val_end = get_split_dates(splits, "val")

    train_cands = candidates.filter(
        (pl.col("asof_date") >= train_start) & (pl.col("asof_date") <= train_end)
    )
    val_cands = candidates.filter(
        (pl.col("asof_date") >= val_start) & (pl.col("asof_date") <= val_end)
    )

    def _prepare(cands: pl.DataFrame):
        weights_list = []
        objectives = []
        state_features = []
        group_indices = []

        date_map = {}
        for row in cands.iter_rows(named=True):
            date = row["asof_date"]
            if date not in date_map:
                date_map[date] = len(date_map)

            w = np.array(json.loads(row["weights_json"]))
            weights_list.append(w)
            objectives.append(row["objective_total"])
            group_indices.append(date_map[date])

            # Simple state features: objective components
            state_features.append([
                row.get("objective_sharpe", 0),
                row.get("turnover", 0),
                row.get("est_cost", 0),
                row.get("concentration_hhi", 0),
            ])

        return (
            np.array(state_features, dtype=np.float32),
            np.array(weights_list, dtype=np.float32),
            np.array(objectives, dtype=np.float32),
            np.array(group_indices, dtype=np.int32),
        )

    train_state, train_weights, train_obj, train_groups = _prepare(train_cands)
    val_state, val_weights, val_obj, val_groups = _prepare(val_cands)

    if len(train_obj) == 0:
        logger.warning("No training candidates found.")
        return {"error": "No training data"}

    model = UtilityScorer()
    train_meta = model.train(
        train_state, train_weights, train_obj,
        val_state, val_weights, val_obj,
    )

    # Evaluate
    eval_metrics = model.evaluate(val_state, val_weights, val_obj, val_groups)
    logger.info("Utility scorer: Spearman=%.4f, Kendall=%.4f", eval_metrics["spearman"], eval_metrics["kendall"])

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "utility_scorer.bin")
    results = {**train_meta, **eval_metrics}
    (out / "utility_scorer_results.json").write_text(json.dumps(results, indent=2, default=str))

    return results
