"""Shared return-alignment helpers for training and evaluation."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl


def build_returns_by_date(
    prices: pl.DataFrame,
    ticker_order: list[str],
) -> tuple[list[dt.date], dict[dt.date, np.ndarray]]:
    """Build a date-indexed return lookup aligned to the model universe."""
    if len(prices) == 0:
        return [], {}

    price_wide = prices.pivot(on="ticker", index="date", values="adj_close").sort("date")
    dates = price_wide["date"].to_list()

    price_cols: list[np.ndarray] = []
    for ticker in ticker_order:
        if ticker == "CASH":
            price_cols.append(np.ones(len(dates), dtype=np.float64))
            continue

        if ticker in price_wide.columns:
            price_cols.append(
                price_wide[ticker].fill_null(strategy="forward").fill_null(strategy="backward").to_numpy()
            )
        else:
            price_cols.append(np.ones(len(dates), dtype=np.float64))

    price_matrix = np.column_stack(price_cols)
    returns = np.zeros_like(price_matrix)
    returns[1:] = price_matrix[1:] / price_matrix[:-1] - 1
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    return dates, {date: returns[idx] for idx, date in enumerate(dates)}


def align_returns_to_dates(
    returns_by_date: dict[dt.date, np.ndarray],
    target_dates: list[dt.date],
    n_assets: int,
) -> np.ndarray:
    """Align a return lookup to a requested ordered date list."""
    aligned = np.zeros((len(target_dates), n_assets), dtype=np.float64)
    for idx, date in enumerate(target_dates):
        if date in returns_by_date:
            aligned[idx] = returns_by_date[date]
    return aligned


def build_feature_matrix_by_date(
    feature_table: pl.DataFrame,
    ticker_order: list[str],
    dates: list[dt.date],
    value_col: str,
    default_value: float = 0.0,
) -> np.ndarray:
    """Pivot a long per-asset feature column into a date x asset matrix."""
    if len(dates) == 0:
        return np.zeros((0, len(ticker_order)), dtype=np.float64)

    if len(feature_table) == 0 or value_col not in feature_table.columns:
        return np.full((len(dates), len(ticker_order)), default_value, dtype=np.float64)

    values = (
        feature_table.select(["asof_date", "ticker", value_col])
        .pivot(on="ticker", index="asof_date", values=value_col)
        .sort("asof_date")
    )
    value_by_date: dict[dt.date, np.ndarray] = {}
    for row in values.iter_rows(named=True):
        vector = np.full(len(ticker_order), default_value, dtype=np.float64)
        for idx, ticker in enumerate(ticker_order):
            vector[idx] = row.get(ticker, default_value)
        vector = np.nan_to_num(vector, nan=default_value, posinf=default_value, neginf=default_value)
        value_by_date[row["asof_date"]] = vector

    matrix = np.full((len(dates), len(ticker_order)), default_value, dtype=np.float64)
    for idx, date in enumerate(dates):
        if date in value_by_date:
            matrix[idx] = value_by_date[date]
    return matrix


def build_covariance_lookup(
    ordered_dates: list[dt.date],
    returns_by_date: dict[dt.date, np.ndarray],
    window: int = 60,
) -> dict[dt.date, np.ndarray | None]:
    """Build a trailing covariance lookup keyed by as-of date."""
    if not ordered_dates:
        return {}

    n_assets = len(next(iter(returns_by_date.values()))) if returns_by_date else 0
    lookup: dict[dt.date, np.ndarray | None] = {}

    returns_matrix = np.array(
        [returns_by_date.get(date, np.zeros(n_assets, dtype=np.float64)) for date in ordered_dates],
        dtype=np.float64,
    )

    for idx, date in enumerate(ordered_dates):
        start = max(1, idx - window + 1)
        window_slice = returns_matrix[start : idx + 1]
        if len(window_slice) < 2:
            lookup[date] = None
            continue
        lookup[date] = np.cov(window_slice.T)

    return lookup
