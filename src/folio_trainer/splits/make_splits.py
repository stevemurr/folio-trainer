"""Train/val/test split generation with purge buffers (spec 1.9)."""

from __future__ import annotations

import datetime as dt
import logging

import polars as pl

from folio_trainer.config.schema import SplitsConfig
from folio_trainer.data.calendars import get_trading_days

logger = logging.getLogger(__name__)


def make_static_splits(
    trading_dates: list[dt.date],
    config: SplitsConfig,
) -> pl.DataFrame:
    """Generate static train/val/test splits with purge buffers.

    Parameters
    ----------
    trading_dates
        Sorted list of all trading dates.
    config
        Split configuration.

    Returns
    -------
    pl.DataFrame
        dataset_splits table with split_name, start_date, end_date,
        purge_start, purge_end columns.
    """
    n = len(trading_dates)
    train_end_idx = int(n * config.train_frac) - 1
    val_end_idx = train_end_idx + int(n * config.val_frac)
    # test gets the rest

    purge = config.purge_days

    splits = []

    # Train
    splits.append({
        "split_name": "train",
        "start_date": trading_dates[0],
        "end_date": trading_dates[train_end_idx],
        "purge_start": trading_dates[train_end_idx],
        "purge_end": trading_dates[min(train_end_idx + purge, n - 1)],
    })

    # Validation (starts after purge)
    val_start_idx = train_end_idx + purge + 1
    if val_start_idx >= n:
        logger.warning("Not enough dates for validation split after purge.")
    else:
        val_end_actual = min(val_end_idx, n - 1)
        splits.append({
            "split_name": "val",
            "start_date": trading_dates[val_start_idx],
            "end_date": trading_dates[val_end_actual],
            "purge_start": trading_dates[val_end_actual],
            "purge_end": trading_dates[min(val_end_actual + purge, n - 1)],
        })

        # Test (starts after val purge)
        test_start_idx = val_end_actual + purge + 1
        if test_start_idx >= n:
            logger.warning("Not enough dates for test split after purge.")
        else:
            splits.append({
                "split_name": "test",
                "start_date": trading_dates[test_start_idx],
                "end_date": trading_dates[n - 1],
                "purge_start": trading_dates[n - 1],
                "purge_end": trading_dates[n - 1],
            })

    return pl.DataFrame(splits)


def make_walkforward_splits(
    trading_dates: list[dt.date],
    config: SplitsConfig,
    exchange: str = "XNYS",
) -> pl.DataFrame:
    """Generate walk-forward rolling splits.

    Parameters
    ----------
    trading_dates
        Sorted list of all trading dates.
    config
        Split configuration.
    exchange
        Exchange calendar to use.

    Returns
    -------
    pl.DataFrame
        dataset_splits table with walk-forward fold entries.
    """
    approx_days_per_year = 252
    train_days = config.walkforward_train_years * approx_days_per_year
    val_days = config.walkforward_val_years * approx_days_per_year
    test_days = config.walkforward_test_years * approx_days_per_year
    purge = config.purge_days
    step_days = test_days  # Roll forward by test window size

    n = len(trading_dates)
    fold_num = 0
    splits = []
    offset = 0

    while offset + train_days + purge + val_days + purge + test_days <= n:
        train_start = offset
        train_end = offset + train_days - 1
        val_start = train_end + purge + 1
        val_end = val_start + val_days - 1
        test_start = val_end + purge + 1
        test_end = test_start + test_days - 1

        if test_end >= n:
            break

        fold_name = f"walkforward_{fold_num:02d}"

        # Train split
        splits.append({
            "split_name": f"{fold_name}_train",
            "start_date": trading_dates[train_start],
            "end_date": trading_dates[train_end],
            "purge_start": trading_dates[train_end],
            "purge_end": trading_dates[min(train_end + purge, n - 1)],
        })

        # Val split
        splits.append({
            "split_name": f"{fold_name}_val",
            "start_date": trading_dates[val_start],
            "end_date": trading_dates[val_end],
            "purge_start": trading_dates[val_end],
            "purge_end": trading_dates[min(val_end + purge, n - 1)],
        })

        # Test split
        splits.append({
            "split_name": f"{fold_name}_test",
            "start_date": trading_dates[test_start],
            "end_date": trading_dates[test_end],
            "purge_start": trading_dates[test_end],
            "purge_end": trading_dates[min(test_end + purge, n - 1)],
        })

        fold_num += 1
        offset += step_days

    logger.info("Generated %d walk-forward folds.", fold_num)
    return pl.DataFrame(splits) if splits else pl.DataFrame()


def get_split_dates(
    splits_df: pl.DataFrame,
    split_name: str,
) -> tuple[dt.date, dt.date]:
    """Get start and end dates for a named split."""
    row = splits_df.filter(pl.col("split_name") == split_name)
    if len(row) == 0:
        msg = f"Split '{split_name}' not found."
        raise ValueError(msg)
    return row["start_date"][0], row["end_date"][0]
