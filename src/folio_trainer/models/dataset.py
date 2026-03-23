"""Dataset preparation for direct-weight models."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
import datetime as dt

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

PORTFOLIO_STATE_FEATURES = (
    "prev_live_weight",
    "prev_target_weight",
    "delta_from_prev_target",
)


@dataclass
class PreparedDataset:
    """Model-ready dataset with train/val/test splits."""

    X_train: np.ndarray
    y_train: np.ndarray  # transformed training target
    target_weight_train: np.ndarray  # raw teacher weight per asset row
    w_train: np.ndarray  # confidence weights
    X_val: np.ndarray
    y_val: np.ndarray
    target_weight_val: np.ndarray
    w_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    target_weight_test: np.ndarray
    w_test: np.ndarray
    feature_names: list[str]
    ticker_names: list[str]
    target_mode: str
    train_dates: list[dt.date]  # unique as-of dates
    val_dates: list[dt.date]
    test_dates: list[dt.date]
    train_prediction_dates: list[dt.date]
    val_prediction_dates: list[dt.date]
    test_prediction_dates: list[dt.date]
    train_date_indices: np.ndarray  # maps each row to its as-of date index
    val_date_indices: np.ndarray
    test_date_indices: np.ndarray
    preprocessing: dict = field(default_factory=dict)


def build_state_feature_frame(
    labels: pl.DataFrame,
    ticker_order: list[str],
) -> pl.DataFrame:
    """Materialize sequential portfolio-state features per (asof_date, ticker)."""
    if len(labels) == 0:
        return pl.DataFrame(
            schema={
                "asof_date": pl.Date,
                "ticker": pl.Utf8,
                "prev_live_weight": pl.Float64,
                "prev_target_weight": pl.Float64,
                "delta_from_prev_target": pl.Float64,
            }
        )

    rows: list[dict] = []
    n_assets = len(ticker_order)
    last_hard_target = np.full(n_assets, 1.0 / max(n_assets, 1))

    for row in labels.sort("asof_date").iter_rows(named=True):
        prev_live = _coerce_weight_vector(row.get("prev_live_weights_json"), n_assets)

        if "prev_target_weights_json" in row and row.get("prev_target_weights_json") is not None:
            prev_target = _coerce_weight_vector(row["prev_target_weights_json"], n_assets)
        else:
            prev_target = last_hard_target.copy()

        for idx, ticker in enumerate(ticker_order):
            rows.append(
                {
                    "asof_date": row["asof_date"],
                    "ticker": ticker,
                    "prev_live_weight": float(prev_live[idx]),
                    "prev_target_weight": float(prev_target[idx]),
                    "delta_from_prev_target": float(prev_live[idx] - prev_target[idx]),
                }
            )

        hard_target_raw = row.get("hard_target_weights_json")
        if hard_target_raw is not None:
            last_hard_target = _coerce_weight_vector(hard_target_raw, n_assets)

    return pl.DataFrame(rows)


def build_feature_table(
    asset_features: pl.DataFrame,
    labels: pl.DataFrame,
    ticker_order: list[str],
    market_features: pl.DataFrame | None = None,
    cross_asset_features: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build the full model feature table keyed by (asof_date, ticker)."""
    if len(asset_features) == 0:
        return asset_features

    asset_base = asset_features.drop(
        [c for c in (*PORTFOLIO_STATE_FEATURES, "missing_feature_count") if c in asset_features.columns]
    )
    scaffold = pl.DataFrame(
        {"asof_date": asset_base["asof_date"].unique().sort().to_list()}
    ).join(
        pl.DataFrame({"ticker": ticker_order}),
        how="cross",
    )
    result = scaffold.join(asset_base, on=["asof_date", "ticker"], how="left")

    if market_features is not None and len(market_features) > 0:
        result = result.join(market_features, on="asof_date", how="left")

    if cross_asset_features is not None and len(cross_asset_features) > 0:
        result = result.join(cross_asset_features, on="asof_date", how="left")

    state_features = build_state_feature_frame(labels, ticker_order)
    result = result.join(state_features, on=["asof_date", "ticker"], how="left")

    feature_cols = [
        c
        for c in result.columns
        if c not in {"asof_date", "ticker", "prediction_date", "missing_feature_count"}
    ]
    result = result.with_columns(
        pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int32) for c in feature_cols]).alias(
            "missing_feature_count"
        )
    )

    ticker_rank = {ticker: idx for idx, ticker in enumerate(ticker_order)}
    return result.with_columns(
        pl.col("ticker").replace_strict(ticker_rank).alias("_ticker_idx")
    ).sort(["asof_date", "_ticker_idx"]).drop("_ticker_idx")


def build_target_table(
    labels: pl.DataFrame,
    ticker_order: list[str],
    target_mode: str = "log_relative",
) -> pl.DataFrame:
    """Expand label vectors to a long target table keyed by (asof_date, ticker)."""
    n_assets = len(ticker_order)
    eqw = 1.0 / max(n_assets, 1)
    rows: list[dict] = []

    for row in labels.iter_rows(named=True):
        soft_target = _coerce_weight_vector(row["soft_target_weights_json"], n_assets)
        confidence = float(row.get("teacher_confidence", 1.0))
        prediction_date = row.get("prediction_date")
        for idx, ticker in enumerate(ticker_order):
            target_weight = float(soft_target[idx])
            if target_mode == "log_relative":
                target_score = float(np.log(max(target_weight, 1e-8) / eqw))
            elif target_mode == "raw_weight":
                target_score = target_weight
            else:
                msg = f"Unknown target_mode '{target_mode}'."
                raise ValueError(msg)

            rows.append(
                {
                    "asof_date": row["asof_date"],
                    "prediction_date": prediction_date,
                    "ticker": ticker,
                    "target_weight": target_weight,
                    "target_score": target_score,
                    "confidence": confidence,
                }
            )

    result = pl.DataFrame(rows)
    ticker_rank = {ticker: idx for idx, ticker in enumerate(ticker_order)}
    return result.with_columns(
        pl.col("ticker").replace_strict(ticker_rank).alias("_ticker_idx")
    ).sort(["asof_date", "_ticker_idx"]).drop("_ticker_idx")


def prepare_dataset(
    asset_features: pl.DataFrame,
    labels: pl.DataFrame,
    splits: pl.DataFrame,
    ticker_order: list[str],
    market_features: pl.DataFrame | None = None,
    cross_asset_features: pl.DataFrame | None = None,
    calendar_dates: list[dt.date] | None = None,
    target_mode: str = "log_relative",
) -> PreparedDataset:
    """Build and preprocess a model-ready dataset.

    Parameters
    ----------
    asset_features
        Asset feature table keyed by (asof_date, ticker).
    labels
        Teacher label table with vector weight columns.
    splits
        Split metadata table.
    ticker_order
        Ordered list of tickers (defines column order in weight vectors).
    market_features
        Shared market feature table keyed by asof_date.
    cross_asset_features
        Shared cross-asset feature table keyed by asof_date.
    calendar_dates
        Trading calendar dates used to resolve prediction_date when labels do
        not already contain it.
    target_mode
        Training target representation. Defaults to ``log_relative``.

    Returns
    -------
    PreparedDataset
    """
    feature_table = build_feature_table(
        asset_features=asset_features,
        labels=labels,
        ticker_order=ticker_order,
        market_features=market_features,
        cross_asset_features=cross_asset_features,
    )
    target_table = build_target_table(labels, ticker_order, target_mode=target_mode)

    prediction_map = _build_prediction_date_frame(labels, feature_table, calendar_dates)
    merged = (
        feature_table.join(target_table, on=["asof_date", "ticker"], how="inner")
        .join(prediction_map, on="asof_date", how="left", suffix="_resolved")
        .with_columns(
            pl.coalesce("prediction_date", "prediction_date_resolved").alias("prediction_date")
        )
    )
    if "prediction_date_resolved" in merged.columns:
        merged = merged.drop("prediction_date_resolved")
    merged = merged.filter(pl.col("prediction_date").is_not_null())

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

    exclude_cols = {
        "asof_date",
        "prediction_date",
        "ticker",
        "confidence",
        "target_score",
        "target_weight",
        "missing_feature_count",
    }
    feature_cols = [c for c in merged.columns if c not in exclude_cols]

    def _extract(df: pl.DataFrame):
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y = df["target_score"].to_numpy().astype(np.float32)
        target_weight = df["target_weight"].to_numpy().astype(np.float32)
        w = (
            df["confidence"].to_numpy().astype(np.float32)
            if "confidence" in df.columns
            else np.ones(len(df), dtype=np.float32)
        )
        dates = df["asof_date"].to_list()
        prediction_dates = (
            df.select(["asof_date", "prediction_date"])
            .unique()
            .sort("asof_date")["prediction_date"]
            .to_list()
        )
        unique_dates = df["asof_date"].unique().sort().to_list()
        date_map = {d: i for i, d in enumerate(unique_dates)}
        date_indices = np.array([date_map[d] for d in dates])
        return X, y, target_weight, w, unique_dates, prediction_dates, date_indices

    (
        X_train,
        y_train,
        target_weight_train,
        w_train,
        train_dates,
        train_prediction_dates,
        train_di,
    ) = _extract(train_df)
    (
        X_val,
        y_val,
        target_weight_val,
        w_val,
        val_dates,
        val_prediction_dates,
        val_di,
    ) = _extract(val_df)
    (
        X_test,
        y_test,
        target_weight_test,
        w_test,
        test_dates,
        test_prediction_dates,
        test_di,
    ) = _extract(test_df)

    preprocessing = {}

    train_medians = np.nanmedian(X_train, axis=0)
    train_medians = np.nan_to_num(train_medians, nan=0.0)
    preprocessing["medians"] = train_medians

    for X in (X_train, X_val, X_test):
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = train_medians[j]

    train_means = np.mean(X_train, axis=0)
    train_stds = np.std(X_train, axis=0)
    train_stds = np.where(train_stds < 1e-8, 1.0, train_stds)
    preprocessing["means"] = train_means
    preprocessing["stds"] = train_stds

    X_train = (X_train - train_means) / train_stds
    X_val = (X_val - train_means) / train_stds
    X_test = (X_test - train_means) / train_stds

    w_train = np.clip(w_train, 0.1, 10.0)
    w_val = np.clip(w_val, 0.1, 10.0)
    w_test = np.clip(w_test, 0.1, 10.0)

    return PreparedDataset(
        X_train=X_train,
        y_train=y_train,
        target_weight_train=target_weight_train,
        w_train=w_train,
        X_val=X_val,
        y_val=y_val,
        target_weight_val=target_weight_val,
        w_val=w_val,
        X_test=X_test,
        y_test=y_test,
        target_weight_test=target_weight_test,
        w_test=w_test,
        feature_names=feature_cols,
        ticker_names=ticker_order,
        target_mode=target_mode,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        train_prediction_dates=train_prediction_dates,
        val_prediction_dates=val_prediction_dates,
        test_prediction_dates=test_prediction_dates,
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


def stack_by_date(
    values: np.ndarray,
    date_indices: np.ndarray,
    n_assets: int,
) -> np.ndarray:
    """Reshape row-wise per-asset values into a date x asset matrix."""
    unique_dates = np.unique(date_indices)
    matrix = np.zeros((len(unique_dates), n_assets), dtype=np.float32)
    for idx, date_idx in enumerate(unique_dates):
        mask = date_indices == date_idx
        date_values = values[mask]
        if len(date_values) != n_assets:
            msg = (
                f"Expected {n_assets} rows for date group {date_idx}, "
                f"got {len(date_values)}."
            )
            raise ValueError(msg)
        matrix[idx] = date_values
    return matrix


def _coerce_weight_vector(raw: str | list[float] | None, n_assets: int) -> np.ndarray:
    """Parse a serialized weight vector and align it to the model universe."""
    if raw is None:
        return np.full(n_assets, 1.0 / max(n_assets, 1), dtype=np.float64)
    values = json.loads(raw) if isinstance(raw, str) else raw
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] < n_assets:
        arr = np.pad(arr, (0, n_assets - arr.shape[0]))
    elif arr.shape[0] > n_assets:
        arr = arr[:n_assets]
    return arr


def _build_prediction_date_frame(
    labels: pl.DataFrame,
    feature_table: pl.DataFrame,
    calendar_dates: list[dt.date] | None,
) -> pl.DataFrame:
    """Resolve prediction dates from labels or the next available trading day."""
    if "prediction_date" in labels.columns:
        return labels.select(["asof_date", "prediction_date"]).unique().sort("asof_date")

    dates = calendar_dates or feature_table["asof_date"].unique().sort().to_list()
    mapping_rows = []
    for idx, asof_date in enumerate(dates[:-1]):
        mapping_rows.append(
            {
                "asof_date": asof_date,
                "prediction_date": dates[idx + 1],
            }
        )
    return pl.DataFrame(mapping_rows)
