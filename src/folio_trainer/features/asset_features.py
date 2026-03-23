"""Per-asset price/volume feature computation (spec 1.7.1)."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from folio_trainer.config.schema import FeaturesConfig

logger = logging.getLogger(__name__)


def compute_asset_features(
    prices: pl.DataFrame,
    config: FeaturesConfig,
) -> pl.DataFrame:
    """Compute all per-ticker rolling features from price data.

    Parameters
    ----------
    prices
        DataFrame with columns: date, ticker, open, high, low, close, adj_close,
        volume, dollar_volume. Sorted by (ticker, date).
    config
        Feature configuration with window sizes.

    Returns
    -------
    pl.DataFrame
        Features keyed by (asof_date, ticker) with feature_cutoff_ts and
        missing_feature_count columns.
    """
    # Ensure sorted
    df = prices.sort(["ticker", "date"])

    # Compute log returns from adj_close
    df = df.with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker"))
        .log()
        .alias("log_ret"),
        (pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1).alias(
            "simple_ret"
        ),
    )

    feature_exprs: list[pl.Expr] = []

    # --- Returns ---
    for w in config.return_windows:
        if w == 1:
            feature_exprs.append(pl.col("simple_ret").alias("ret_1"))
        else:
            feature_exprs.append(
                (pl.col("adj_close") / pl.col("adj_close").shift(w).over("ticker") - 1).alias(
                    f"ret_{w}"
                )
            )

    # --- Realized volatility ---
    for w in config.vol_windows:
        feature_exprs.append(
            pl.col("log_ret")
            .rolling_std(window_size=w)
            .over("ticker")
            .mul(np.sqrt(252))
            .alias(f"vol_{w}")
        )

    # --- Downside volatility (only negative returns) ---
    df = df.with_columns(
        pl.when(pl.col("log_ret") < 0).then(pl.col("log_ret")).otherwise(0.0).alias("neg_ret")
    )
    for w in [20, 60]:
        feature_exprs.append(
            pl.col("neg_ret")
            .rolling_std(window_size=w)
            .over("ticker")
            .mul(np.sqrt(252))
            .alias(f"downvol_{w}")
        )

    # --- Drawdown ---
    for w in config.dd_windows:
        feature_exprs.append(
            (
                pl.col("adj_close")
                / pl.col("adj_close").rolling_max(window_size=w).over("ticker")
                - 1
            ).alias(f"dd_{w}")
        )

    # --- Momentum / reversal ---
    for w in [20, 60, 120]:
        feature_exprs.append(
            (pl.col("adj_close") / pl.col("adj_close").shift(w).over("ticker") - 1).alias(
                f"mom_{w}"
            )
        )
    feature_exprs.append(
        (pl.col("adj_close") / pl.col("adj_close").shift(5).over("ticker") - 1)
        .mul(-1)
        .alias("rev_5")
    )

    # --- Liquidity ---
    for w in config.liquidity_windows:
        feature_exprs.append(
            pl.col("dollar_volume")
            .rolling_mean(window_size=w)
            .over("ticker")
            .alias(f"dollar_vol_{w}")
        )

    # Amihud illiquidity (|return| / dollar_volume)
    df = df.with_columns(
        (pl.col("simple_ret").abs() / pl.col("dollar_volume").clip(lower_bound=1.0)).alias(
            "amihud_daily"
        )
    )
    feature_exprs.append(
        pl.col("amihud_daily")
        .rolling_mean(window_size=20)
        .over("ticker")
        .alias("amihud_20")
    )

    # --- Volume shock z-scores ---
    feature_exprs.append(
        (
            (
                pl.col("volume")
                - pl.col("volume").rolling_mean(window_size=20).over("ticker")
            )
            / pl.col("volume").rolling_std(window_size=20).over("ticker").clip(lower_bound=1.0)
        ).alias("volume_zscore_20")
    )
    feature_exprs.append(
        (
            (
                pl.col("dollar_volume")
                - pl.col("dollar_volume").rolling_mean(window_size=20).over("ticker")
            )
            / pl.col("dollar_volume")
            .rolling_std(window_size=20)
            .over("ticker")
            .clip(lower_bound=1.0)
        ).alias("dollar_vol_zscore_20")
    )

    # Apply all feature expressions
    df = df.with_columns(feature_exprs)

    # Portfolio context placeholders (populated during label generation)
    df = df.with_columns(
        pl.lit(None).cast(pl.Float64).alias("prev_live_weight"),
        pl.lit(None).cast(pl.Float64).alias("prev_target_weight"),
        pl.lit(None).cast(pl.Float64).alias("delta_from_prev_target"),
    )

    # Rename date to asof_date for feature table
    df = df.rename({"date": "asof_date"})

    # Count missing features per row
    feature_cols = [
        c
        for c in df.columns
        if c not in ("asof_date", "ticker", "open", "high", "low", "close", "adj_close",
                      "volume", "dollar_volume", "source_ts", "log_ret", "simple_ret",
                      "neg_ret", "amihud_daily", "ingested_at", "source_name", "source_version")
    ]
    df = df.with_columns(
        pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int32) for c in feature_cols]).alias(
            "missing_feature_count"
        )
    )

    # Select only feature columns + keys
    keep_cols = ["asof_date", "ticker"] + feature_cols + ["missing_feature_count"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df.select(keep_cols)


def compute_beta_and_correlation(
    prices: pl.DataFrame,
    config: FeaturesConfig,
) -> pl.DataFrame:
    """Compute rolling beta and correlation to equal-weight universe index.

    This requires cross-sectional data so is separated from per-ticker features.
    """
    df = prices.sort(["ticker", "date"])

    # Compute simple returns
    df = df.with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1).alias(
            "simple_ret"
        )
    )

    # Equal-weight universe return per date
    eqw_ret = (
        df.group_by("date")
        .agg(pl.col("simple_ret").mean().alias("eqw_ret"))
        .sort("date")
    )

    df = df.join(eqw_ret, on="date", how="left")

    results = []
    for ticker in df["ticker"].unique().to_list():
        ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")

        feature_exprs = []
        for w in config.beta_windows:
            # Rolling beta = cov(r_i, r_m) / var(r_m)
            feature_exprs.extend([
                _rolling_beta(ticker_df, w).alias(f"beta_eqw_{w}"),
            ])

        for w in config.corr_windows:
            feature_exprs.extend([
                _rolling_corr(ticker_df, w).alias(f"corr_eqw_{w}"),
            ])

        ticker_result = ticker_df.select(["date", "ticker"]).with_columns(feature_exprs)
        results.append(ticker_result)

    if not results:
        return pl.DataFrame()

    return pl.concat(results).rename({"date": "asof_date"})


def _rolling_beta(df: pl.DataFrame, window: int) -> pl.Expr:
    """Compute rolling beta as cov(asset, market) / var(market)."""
    # Using rolling covariance / rolling variance
    cov = pl.col("simple_ret").rolling_map(
        lambda s: np.cov(s, df["eqw_ret"].to_numpy()[-len(s):])[0, 1]
        if len(s) >= 2 else None,
        window_size=window,
    )
    var = pl.col("eqw_ret").rolling_var(window_size=window)
    return cov / var.clip(lower_bound=1e-10)


def _rolling_corr(df: pl.DataFrame, window: int) -> pl.Expr:
    """Compute rolling correlation between asset and market returns."""
    return pl.col("simple_ret").rolling_map(
        lambda s: np.corrcoef(s, df["eqw_ret"].to_numpy()[-len(s):])[0, 1]
        if len(s) >= 2 else None,
        window_size=window,
    )
