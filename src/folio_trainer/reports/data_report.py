"""Data quality and leakage audit reports (spec 1.10)."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def generate_data_report(
    data_dir: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """Generate a comprehensive data quality report.

    Returns the report as a markdown string and optionally writes to file.
    """
    data_dir = Path(data_dir)
    sections: list[str] = []
    sections.append("# Data Quality Report\n")
    sections.append(f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}\n")

    # Universe summary
    sections.append(_check_prices(data_dir))
    sections.append(_check_features(data_dir))
    sections.append(_check_labels(data_dir))
    sections.append(_check_splits(data_dir))

    report = "\n---\n\n".join(sections)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        logger.info("Report written to %s", out)

    return report


def _check_prices(data_dir: Path) -> str:
    """Ingestion checks for prices (spec 1.10.1)."""
    lines = ["## Price Data Checks\n"]
    price_file = data_dir / "bronze" / "prices_daily.parquet"

    if not price_file.exists():
        lines.append("- **SKIP**: No price data found.\n")
        return "\n".join(lines)

    df = pl.read_parquet(price_file)
    n_rows = len(df)
    n_tickers = df["ticker"].n_unique()
    date_range = f"{df['date'].min()} to {df['date'].max()}"
    lines.append(f"- Rows: {n_rows:,}")
    lines.append(f"- Tickers: {n_tickers}")
    lines.append(f"- Date range: {date_range}")

    # Duplicate check
    dupes = df.group_by(["date", "ticker"]).len().filter(pl.col("len") > 1)
    if len(dupes) > 0:
        lines.append(f"- **FAIL**: {len(dupes)} duplicate (date, ticker) rows")
    else:
        lines.append("- **PASS**: No duplicate (date, ticker) rows")

    # Missing adj_close
    null_adj = df.filter(pl.col("adj_close").is_null())
    if len(null_adj) > 0:
        lines.append(f"- **WARN**: {len(null_adj)} rows with null adj_close")
    else:
        lines.append("- **PASS**: No null adj_close values")

    return "\n".join(lines)


def _check_features(data_dir: Path) -> str:
    """Feature table checks."""
    lines = ["## Feature Table Checks\n"]
    feature_file = data_dir / "features" / "features_asset_daily.parquet"

    if not feature_file.exists():
        lines.append("- **SKIP**: No feature table found.\n")
        return "\n".join(lines)

    df = pl.read_parquet(feature_file)
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Columns: {len(df.columns)}")

    # Missingness summary
    total_cells = len(df) * len(df.columns)
    null_cells = sum(df[col].null_count() for col in df.columns)
    miss_pct = null_cells / max(total_cells, 1) * 100
    lines.append(f"- Overall missingness: {miss_pct:.1f}%")

    # Per-column missingness (top 10)
    col_miss = [(col, df[col].null_count()) for col in df.columns]
    col_miss.sort(key=lambda x: -x[1])
    lines.append("- Top missing columns:")
    for col, count in col_miss[:10]:
        if count > 0:
            lines.append(f"  - {col}: {count:,} ({count / len(df) * 100:.1f}%)")

    return "\n".join(lines)


def _check_labels(data_dir: Path) -> str:
    """Label checks (spec 1.10.3)."""
    lines = ["## Label Checks\n"]
    label_file = data_dir / "labels" / "teacher_labels.parquet"

    if not label_file.exists():
        lines.append("- **SKIP**: No label table found.\n")
        return "\n".join(lines)

    df = pl.read_parquet(label_file)
    lines.append(f"- Label rows: {len(df):,}")

    # Weight sum check
    import json as _json

    errors = 0
    neg_weight_count = 0
    for row in df.iter_rows(named=True):
        hard = np.array(_json.loads(row["hard_target_weights_json"]))
        if abs(hard.sum() - 1.0) > 1e-6:
            errors += 1
        if (hard < -1e-8).any():
            neg_weight_count += 1

        soft = np.array(_json.loads(row["soft_target_weights_json"]))
        if abs(soft.sum() - 1.0) > 1e-6:
            errors += 1
        if (soft < -1e-8).any():
            neg_weight_count += 1

    if errors == 0:
        lines.append("- **PASS**: All target weights sum to 1 (within 1e-6)")
    else:
        lines.append(f"- **FAIL**: {errors} weight vectors don't sum to 1")

    if neg_weight_count == 0:
        lines.append("- **PASS**: No negative weights")
    else:
        lines.append(f"- **FAIL**: {neg_weight_count} weight vectors have negative weights")

    # Confidence stats
    if "teacher_confidence" in df.columns:
        conf = df["teacher_confidence"]
        lines.append(f"- Confidence: mean={conf.mean():.4f}, min={conf.min():.4f}, max={conf.max():.4f}")

    return "\n".join(lines)


def _check_splits(data_dir: Path) -> str:
    """Split summary."""
    lines = ["## Split Summary\n"]
    split_file = data_dir / "splits" / "dataset_splits.parquet"

    if not split_file.exists():
        lines.append("- **SKIP**: No split metadata found.\n")
        return "\n".join(lines)

    df = pl.read_parquet(split_file)
    for row in df.iter_rows(named=True):
        lines.append(
            f"- **{row['split_name']}**: {row['start_date']} to {row['end_date']} "
            f"(purge: {row['purge_start']} to {row['purge_end']})"
        )

    return "\n".join(lines)
