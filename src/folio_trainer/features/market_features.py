"""Market/regime features (spec 1.7.3)."""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def compute_market_features(
    treasury_yields: pl.DataFrame | None = None,
    ff_factors: pl.DataFrame | None = None,
    macro_series: pl.DataFrame | None = None,
    vix_series: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute market-level regime features.

    All inputs are optional; features are computed for whatever data is available.

    Returns
    -------
    pl.DataFrame
        Keyed by asof_date with market/regime feature columns.
    """
    frames: list[pl.DataFrame] = []

    if vix_series is not None and len(vix_series) > 0:
        frames.append(_compute_vix_features(vix_series))

    if treasury_yields is not None and len(treasury_yields) > 0:
        frames.append(_compute_treasury_features(treasury_yields))

    if ff_factors is not None and len(ff_factors) > 0:
        frames.append(_compute_ff_features(ff_factors))

    if macro_series is not None and len(macro_series) > 0:
        frames.append(_compute_macro_features(macro_series))

    if not frames:
        return pl.DataFrame(schema={"asof_date": pl.Date})

    # Join all on asof_date
    result = frames[0]
    for f in frames[1:]:
        result = result.join(f, on="asof_date", how="outer_coalesce")

    return result.sort("asof_date")


def _compute_vix_features(vix: pl.DataFrame) -> pl.DataFrame:
    """VIX level and changes."""
    # Expect columns: date, value (or series_id + value)
    if "series_id" in vix.columns:
        vix = vix.filter(pl.col("series_id") == "VIXCLS")

    date_col = "date" if "date" in vix.columns else "asof_date"
    df = vix.select([pl.col(date_col).alias("asof_date"), pl.col("value").alias("vix")]).sort(
        "asof_date"
    )

    df = df.with_columns(
        (pl.col("vix") - pl.col("vix").shift(5)).alias("vix_chg_5"),
        (pl.col("vix") - pl.col("vix").shift(20)).alias("vix_chg_20"),
        (pl.col("vix") / pl.col("vix").shift(5) - 1).alias("vix_pct_chg_5"),
        (pl.col("vix") / pl.col("vix").shift(20) - 1).alias("vix_pct_chg_20"),
    )

    return df


def _compute_treasury_features(treasury: pl.DataFrame) -> pl.DataFrame:
    """Treasury yields and slope terms."""
    # Pivot from long to wide format
    wide = treasury.pivot(on="maturity", index="date", values="yield_pct")
    wide = wide.rename({"date": "asof_date"})

    # Rename columns with treasury_ prefix
    rename_map = {}
    for col in wide.columns:
        if col != "asof_date":
            rename_map[col] = f"tsy_{col}"
    wide = wide.rename(rename_map)

    # Add slope terms
    slope_exprs = []
    if "tsy_10y" in wide.columns and "tsy_2y" in wide.columns:
        slope_exprs.append((pl.col("tsy_10y") - pl.col("tsy_2y")).alias("tsy_slope_10y_2y"))
    if "tsy_10y" in wide.columns and "tsy_3m" in wide.columns:
        slope_exprs.append((pl.col("tsy_10y") - pl.col("tsy_3m")).alias("tsy_slope_10y_3m"))

    if slope_exprs:
        wide = wide.with_columns(slope_exprs)

    return wide.sort("asof_date")


def _compute_ff_features(ff: pl.DataFrame) -> pl.DataFrame:
    """Fama-French factor features."""
    date_col = "date" if "date" in ff.columns else "asof_date"
    df = ff.rename({date_col: "asof_date"})

    # Standardize column names
    rename = {}
    for col in df.columns:
        if col == "asof_date":
            continue
        clean = col.lower().replace("-", "_").replace(" ", "_")
        if clean != col:
            rename[col] = clean
    if rename:
        df = df.rename(rename)

    # Prefix with ff_
    rename2 = {}
    for col in df.columns:
        if col not in ("asof_date", "source_ts", "ingested_at", "source_name", "source_version"):
            if not col.startswith("ff_"):
                rename2[col] = f"ff_{col}"
    if rename2:
        df = df.rename(rename2)

    # Drop metadata columns
    drop_cols = [c for c in ("source_ts", "ingested_at", "source_name", "source_version") if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    return df.sort("asof_date")


def _compute_macro_features(macro: pl.DataFrame) -> pl.DataFrame:
    """Forward-filled macro features from FRED/ALFRED."""
    if "series_id" not in macro.columns:
        return pl.DataFrame(schema={"asof_date": pl.Date})

    date_col = "date" if "date" in macro.columns else "asof_date"

    # Pivot series to wide format, forward-fill
    wide = (
        macro.select([pl.col(date_col).alias("asof_date"), "series_id", "value"])
        .pivot(on="series_id", index="asof_date", values="value")
        .sort("asof_date")
    )

    # Forward-fill all macro columns (data released infrequently)
    macro_cols = [c for c in wide.columns if c != "asof_date"]
    wide = wide.with_columns([pl.col(c).forward_fill().alias(c) for c in macro_cols])

    # Prefix with macro_
    rename = {c: f"macro_{c.lower()}" for c in macro_cols}
    wide = wide.rename(rename)

    return wide
