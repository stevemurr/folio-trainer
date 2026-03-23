"""Dataset preparation: join features + labels, fit preprocessing on train only."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class PreparedDataset:
    """Model-ready dataset with train/val/test splits."""

    X_train: np.ndarray
    y_train: np.ndarray  # soft target weights per asset
    w_train: np.ndarray  # confidence weights
    X_val: np.ndarray
    y_val: np.ndarray
    w_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    w_test: np.ndarray
    feature_names: list[str]
    ticker_names: list[str]
    train_dates: list
    val_dates: list
    test_dates: list
    # Per-row metadata for cross-sectional grouping
    train_date_indices: np.ndarray  # maps each row to its date index
    val_date_indices: np.ndarray
    test_date_indices: np.ndarray
    preprocessing: dict = field(default_factory=dict)


def prepare_dataset(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    splits: pl.DataFrame,
    ticker_order: list[str],
) -> PreparedDataset:
    """Join features and labels, split, and preprocess.

    Parameters
    ----------
    features
        Asset feature table with (asof_date, ticker) key.
    labels
        Teacher labels with (asof_date) key and weight JSON columns.
    splits
        Split metadata table.
    ticker_order
        Ordered list of tickers (defines column order in weight vectors).

    Returns
    -------
    PreparedDataset
    """
    # Parse label weight vectors
    label_rows = []
    for row in labels.iter_rows(named=True):
        soft_w = json.loads(row["soft_target_weights_json"])
        conf = row.get("teacher_confidence", 1.0)
        label_rows.append({
            "asof_date": row["asof_date"],
            "confidence": conf,
            **{f"target_{ticker}": soft_w[i] for i, ticker in enumerate(ticker_order)},
        })
    label_df = pl.DataFrame(label_rows)

    # Join features with labels on asof_date
    merged = features.join(label_df, on="asof_date", how="inner")

    # Get split boundaries
    from folio_trainer.splits.make_splits import get_split_dates

    train_start, train_end = get_split_dates(splits, "train")
    val_start, val_end = get_split_dates(splits, "val")
    test_start, test_end = get_split_dates(splits, "test")

    train_df = merged.filter(
        (pl.col("asof_date") >= train_start) & (pl.col("asof_date") <= train_end)
    )
    val_df = merged.filter(
        (pl.col("asof_date") >= val_start) & (pl.col("asof_date") <= val_end)
    )
    test_df = merged.filter(
        (pl.col("asof_date") >= test_start) & (pl.col("asof_date") <= test_end)
    )

    # Identify feature columns (exclude keys, targets, metadata)
    exclude_cols = {"asof_date", "ticker", "confidence", "missing_feature_count"}
    target_cols = {f"target_{t}" for t in ticker_order}
    exclude_cols.update(target_cols)
    feature_cols = [c for c in merged.columns if c not in exclude_cols]

    # Extract arrays
    def _extract(df: pl.DataFrame):
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        # For each row (one per ticker per date), the target is the weight for that ticker
        # But we need to know which ticker this row is for
        tickers = df["ticker"].to_list()
        y = np.array([
            df[f"target_{t}"][i] if f"target_{t}" in df.columns else 0.0
            for i, t in enumerate(tickers)
        ], dtype=np.float32)
        w = df["confidence"].to_numpy().astype(np.float32) if "confidence" in df.columns else np.ones(len(df), dtype=np.float32)
        dates = df["asof_date"].to_list()
        # Date indices for grouping
        unique_dates = sorted(set(dates))
        date_map = {d: i for i, d in enumerate(unique_dates)}
        date_indices = np.array([date_map[d] for d in dates])
        return X, y, w, unique_dates, date_indices

    X_train, y_train, w_train, train_dates, train_di = _extract(train_df)
    X_val, y_val, w_val, val_dates, val_di = _extract(val_df)
    X_test, y_test, w_test, test_dates, test_di = _extract(test_df)

    # Fit preprocessing on train only
    preprocessing = {}

    # Impute NaN with train median
    train_medians = np.nanmedian(X_train, axis=0)
    train_medians = np.nan_to_num(train_medians, nan=0.0)
    preprocessing["medians"] = train_medians

    for X in (X_train, X_val, X_test):
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = train_medians[j]

    # Standardize (z-score) using train statistics
    train_means = np.mean(X_train, axis=0)
    train_stds = np.std(X_train, axis=0)
    train_stds = np.where(train_stds < 1e-8, 1.0, train_stds)
    preprocessing["means"] = train_means
    preprocessing["stds"] = train_stds

    X_train = (X_train - train_means) / train_stds
    X_val = (X_val - train_means) / train_stds
    X_test = (X_test - train_means) / train_stds

    # Clip confidence weights
    w_train = np.clip(w_train, 0.1, 10.0)
    w_val = np.clip(w_val, 0.1, 10.0)
    w_test = np.clip(w_test, 0.1, 10.0)

    return PreparedDataset(
        X_train=X_train, y_train=y_train, w_train=w_train,
        X_val=X_val, y_val=y_val, w_val=w_val,
        X_test=X_test, y_test=y_test, w_test=w_test,
        feature_names=feature_cols,
        ticker_names=ticker_order,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        train_date_indices=train_di,
        val_date_indices=val_di,
        test_date_indices=test_di,
        preprocessing=preprocessing,
    )


def save_preprocessing(preprocessing: dict, path: str | Path) -> None:
    """Save preprocessing artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessing, f)


def load_preprocessing(path: str | Path) -> dict:
    """Load preprocessing artifacts."""
    with open(path, "rb") as f:
        return pickle.load(f)
