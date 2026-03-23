"""Data quality and research audit reports."""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def generate_data_report(
    data_dir: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """Generate a comprehensive data quality report."""
    data_dir = Path(data_dir)
    sections: list[str] = []
    sections.append("# Data Quality Report\n")
    sections.append(f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}\n")

    sections.append(_check_prices(data_dir))
    sections.append(_check_feature_blocks(data_dir))
    sections.append(_check_features(data_dir))
    sections.append(_check_labels(data_dir))
    sections.append(_check_candidates(data_dir))
    sections.append(_check_splits(data_dir))

    report = "\n---\n\n".join(sections)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        logger.info("Report written to %s", out)

    return report


def _check_prices(data_dir: Path) -> str:
    lines = ["## Price Data Checks\n"]
    price_file = data_dir / "bronze" / "prices_daily.parquet"

    if not price_file.exists():
        lines.append("- **SKIP**: No price data found.\n")
        return "\n".join(lines)

    df = pl.read_parquet(price_file)
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Tickers: {df['ticker'].n_unique()}")
    lines.append(f"- Date range: {df['date'].min()} to {df['date'].max()}")

    dupes = df.group_by(["date", "ticker"]).len().filter(pl.col("len") > 1)
    lines.append(
        f"- {'**PASS**' if len(dupes) == 0 else '**FAIL**'}: duplicate (date, ticker) rows = {len(dupes)}"
    )

    null_adj = df.filter(pl.col("adj_close").is_null())
    lines.append(
        f"- {'**PASS**' if len(null_adj) == 0 else '**WARN**'}: null adj_close rows = {len(null_adj)}"
    )
    return "\n".join(lines)


def _check_feature_blocks(data_dir: Path) -> str:
    lines = ["## Feature Block Coverage\n"]
    feature_dir = data_dir / "features"
    for name in (
        "features_asset_daily.parquet",
        "features_market_daily.parquet",
        "features_cross_asset_daily.parquet",
    ):
        path = feature_dir / name
        label = name.replace(".parquet", "")
        if not path.exists():
            lines.append(f"- **MISSING**: {label}")
            continue
        df = pl.read_parquet(path)
        date_col = "asof_date" if "asof_date" in df.columns else "date"
        lines.append(
            f"- **{label}**: rows={len(df):,}, cols={len(df.columns)}, "
            f"range={df[date_col].min()} to {df[date_col].max()}"
        )
    return "\n".join(lines)


def _check_features(data_dir: Path) -> str:
    lines = ["## Joined Feature Health\n"]
    asset_path = data_dir / "features" / "features_asset_daily.parquet"

    if not asset_path.exists():
        lines.append("- **SKIP**: No asset feature table found.\n")
        return "\n".join(lines)

    asset_df = pl.read_parquet(asset_path)
    lines.append(f"- Asset feature rows: {len(asset_df):,}")
    lines.append(f"- Asset feature columns: {len(asset_df.columns)}")

    total_cells = len(asset_df) * len(asset_df.columns)
    null_cells = sum(asset_df[col].null_count() for col in asset_df.columns)
    lines.append(f"- Overall asset-feature missingness: {null_cells / max(total_cells, 1) * 100:.2f}%")

    state_cols = [c for c in ("prev_live_weight", "prev_target_weight", "delta_from_prev_target") if c in asset_df.columns]
    if state_cols:
        for col in state_cols:
            nulls = asset_df[col].null_count()
            lines.append(f"- State feature `{col}` null rows: {nulls:,}")

    top_missing = (
        asset_df.select([pl.all().null_count()])
        .transpose(include_header=True, header_name="column", column_names=["nulls"])
        .sort("nulls", descending=True)
        .head(10)
        .to_dicts()
    )
    lines.append("- Top missing columns:")
    for row in top_missing:
        if row["nulls"] > 0:
            lines.append(f"  - {row['column']}: {row['nulls']:,}")
    return "\n".join(lines)


def _check_labels(data_dir: Path) -> str:
    lines = ["## Label Checks\n"]
    label_file = data_dir / "labels" / "teacher_labels.parquet"

    if not label_file.exists():
        lines.append("- **SKIP**: No label table found.\n")
        return "\n".join(lines)

    df = pl.read_parquet(label_file).sort("asof_date")
    lines.append(f"- Label rows: {len(df):,}")

    errors = 0
    neg_weight_count = 0
    active_positions = []
    max_weights = []
    cash_weights = []

    for row in df.iter_rows(named=True):
        hard = np.asarray(json.loads(row["hard_target_weights_json"]), dtype=float)
        soft = np.asarray(json.loads(row["soft_target_weights_json"]), dtype=float)

        if abs(hard.sum() - 1.0) > 1e-6 or abs(soft.sum() - 1.0) > 1e-6:
            errors += 1
        if (hard < -1e-8).any() or (soft < -1e-8).any():
            neg_weight_count += 1

        active_positions.append(int(np.sum(soft > 0.01)))
        max_weights.append(float(soft.max()))
        cash_weights.append(float(soft[-1]))

    lines.append(
        f"- {'**PASS**' if errors == 0 else '**FAIL**'}: weight vectors summing to 1 violations = {errors}"
    )
    lines.append(
        f"- {'**PASS**' if neg_weight_count == 0 else '**FAIL**'}: negative-weight vectors = {neg_weight_count}"
    )

    if "teacher_confidence" in df.columns:
        conf = df["teacher_confidence"]
        lines.append(
            f"- Confidence: mean={conf.mean():.4f}, median={conf.median():.4f}, min={conf.min():.4f}, max={conf.max():.4f}"
        )

    lines.append(
        f"- Soft target concentration: active>1% mean={np.mean(active_positions):.2f}, "
        f"median={np.median(active_positions):.1f}, max weight mean={np.mean(max_weights):.2%}, "
        f"cash weight mean={np.mean(cash_weights):.2%}"
    )
    return "\n".join(lines)


def _check_candidates(data_dir: Path) -> str:
    lines = ["## Candidate Search Audit\n"]
    candidate_file = data_dir / "labels" / "teacher_candidates.parquet"

    if not candidate_file.exists():
        lines.append("- **SKIP**: No candidate table found.\n")
        return "\n".join(lines)

    df = pl.read_parquet(candidate_file)
    lines.append(f"- Candidate rows: {len(df):,}")

    by_type = df.group_by("candidate_type").len().sort("len", descending=True).to_dicts()
    best_by_type = (
        df.filter(pl.col("rank") == 0)
        .group_by("candidate_type")
        .len()
        .sort("len", descending=True)
        .to_dicts()
    )

    lines.append("- Candidate counts by type:")
    for row in by_type:
        lines.append(f"  - {row['candidate_type']}: {row['len']:,}")

    lines.append("- Best-candidate win counts by type:")
    for row in best_by_type:
        lines.append(f"  - {row['candidate_type']}: {row['len']:,}")

    objective_stats = (
        df.group_by("candidate_type")
        .agg(
            pl.col("objective_total").mean().alias("mean_objective"),
            pl.col("objective_total").median().alias("median_objective"),
        )
        .sort("mean_objective", descending=True)
        .to_dicts()
    )
    lines.append("- Objective totals by type:")
    for row in objective_stats:
        lines.append(
            f"  - {row['candidate_type']}: mean={row['mean_objective']:.4f}, "
            f"median={row['median_objective']:.4f}"
        )
    return "\n".join(lines)


def _check_splits(data_dir: Path) -> str:
    lines = ["## Split Coverage\n"]
    split_file = data_dir / "splits" / "dataset_splits.parquet"

    if not split_file.exists():
        lines.append("- **SKIP**: No split metadata found.\n")
        return "\n".join(lines)

    df = pl.read_parquet(split_file).sort("split_name")
    static = df.filter(~pl.col("split_name").str.starts_with("walkforward"))
    walkforward = df.filter(pl.col("split_name").str.starts_with("walkforward"))

    lines.append(f"- Static split count: {len(static)}")
    lines.append(f"- Walk-forward split count: {len(walkforward)}")

    for row in static.iter_rows(named=True):
        lines.append(
            f"- **{row['split_name']}**: {row['start_date']} to {row['end_date']} "
            f"(purge: {row['purge_start']} to {row['purge_end']})"
        )

    if len(walkforward) > 0:
        fold_ids = sorted({"_".join(name.split("_")[:2]) for name in walkforward["split_name"].to_list()})
        lines.append(f"- Walk-forward folds: {', '.join(fold_ids)}")
    return "\n".join(lines)
