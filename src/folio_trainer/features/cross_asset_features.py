"""Cross-asset structure features (spec 1.7.2)."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from folio_trainer.config.schema import FeaturesConfig

logger = logging.getLogger(__name__)


def compute_cross_asset_features(
    prices: pl.DataFrame,
    config: FeaturesConfig,
) -> pl.DataFrame:
    """Compute cross-sectional features from the full universe return matrix.

    Parameters
    ----------
    prices
        DataFrame with date, ticker, adj_close columns.
    config
        Feature configuration.

    Returns
    -------
    pl.DataFrame
        Keyed by asof_date with cross-asset feature columns.
    """
    df = prices.sort(["ticker", "date"])

    # Compute simple returns
    df = df.with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1).alias(
            "simple_ret"
        )
    )

    # Pivot to wide format: rows=dates, columns=tickers, values=returns
    ret_wide = df.pivot(on="ticker", index="date", values="simple_ret").sort("date")

    dates = ret_wide["date"].to_list()
    ticker_cols = [c for c in ret_wide.columns if c != "date"]
    ret_matrix = ret_wide.select(ticker_cols).to_numpy()  # (n_dates, n_tickers)

    # Equal-weight universe return
    eqw_returns = np.nanmean(ret_matrix, axis=1)

    results: list[dict] = []

    for i, date in enumerate(dates):
        row: dict = {"asof_date": date}

        row["eqw_universe_ret"] = float(eqw_returns[i]) if not np.isnan(eqw_returns[i]) else None

        # Cross-sectional features for different windows
        for window in [20, 60]:
            if i < window:
                # Not enough history
                row[f"avg_pairwise_corr_{window}"] = None
                row[f"avg_vol_{window}"] = None
                for pc in range(config.pca_components):
                    row[f"pca_var_ratio_{pc}_{window}"] = None
                continue

            window_rets = ret_matrix[i - window + 1 : i + 1]  # (window, n_tickers)

            # Remove tickers with all NaN in window
            valid_mask = ~np.all(np.isnan(window_rets), axis=0)
            valid_rets = window_rets[:, valid_mask]

            if valid_rets.shape[1] < 2:
                row[f"avg_pairwise_corr_{window}"] = None
                row[f"avg_vol_{window}"] = None
                for pc in range(config.pca_components):
                    row[f"pca_var_ratio_{pc}_{window}"] = None
                continue

            # Fill NaN with 0 for covariance computation
            clean_rets = np.nan_to_num(valid_rets, nan=0.0)

            # Average pairwise correlation
            corr_matrix = np.corrcoef(clean_rets.T)
            n = corr_matrix.shape[0]
            if n > 1:
                # Extract upper triangle (excluding diagonal)
                upper_tri = corr_matrix[np.triu_indices(n, k=1)]
                row[f"avg_pairwise_corr_{window}"] = float(np.nanmean(upper_tri))
            else:
                row[f"avg_pairwise_corr_{window}"] = None

            # Average volatility
            vols = np.nanstd(clean_rets, axis=0) * np.sqrt(252)
            row[f"avg_vol_{window}"] = float(np.mean(vols))

            # PCA on correlation matrix
            try:
                eigenvalues = np.linalg.eigvalsh(corr_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]  # descending
                total_var = np.sum(eigenvalues)
                for pc in range(config.pca_components):
                    if pc < len(eigenvalues) and total_var > 0:
                        row[f"pca_var_ratio_{pc}_{window}"] = float(
                            eigenvalues[pc] / total_var
                        )
                    else:
                        row[f"pca_var_ratio_{pc}_{window}"] = None
            except np.linalg.LinAlgError:
                for pc in range(config.pca_components):
                    row[f"pca_var_ratio_{pc}_{window}"] = None

        # Concentration of recent winners/losers (20-day momentum breadth)
        if i >= 20:
            mom_20 = ret_matrix[i] if i > 0 else np.zeros(len(ticker_cols))
            # Use cumulative return over 20 days
            cum_20 = np.nanprod(1 + ret_matrix[max(0, i - 19) : i + 1], axis=0) - 1
            valid = ~np.isnan(cum_20)
            if np.sum(valid) >= 4:
                sorted_mom = np.sort(cum_20[valid])[::-1]
                n_valid = len(sorted_mom)
                top_q = max(1, n_valid // 4)
                row["breadth_top_quartile_20"] = float(np.mean(sorted_mom[:top_q]))
                row["breadth_bottom_quartile_20"] = float(np.mean(sorted_mom[-top_q:]))
                row["breadth_spread_20"] = float(
                    np.mean(sorted_mom[:top_q]) - np.mean(sorted_mom[-top_q:])
                )
            else:
                row["breadth_top_quartile_20"] = None
                row["breadth_bottom_quartile_20"] = None
                row["breadth_spread_20"] = None
        else:
            row["breadth_top_quartile_20"] = None
            row["breadth_bottom_quartile_20"] = None
            row["breadth_spread_20"] = None

        results.append(row)

    return pl.DataFrame(results).with_columns(pl.col("asof_date").cast(pl.Date))
