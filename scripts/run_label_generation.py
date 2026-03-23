"""Run teacher label generation on the S&P 500 dataset.

Usage: python scripts/run_label_generation.py [--fast]

With --fast, uses reduced candidate counts for quick iteration.
"""

import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

from folio_trainer.config.loader import load_config
from folio_trainer.data.calendars import get_trading_days
from folio_trainer.data.universe import build_universe, get_weight_caps
from folio_trainer.labels.teacher_sequential_sim import run_sequential_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FAST_MODE = "--fast" in sys.argv


def main():
    data_dir = Path("data")
    cfg = load_config()

    # Load prices
    logger.info("Loading price data...")
    prices = pl.read_parquet(data_dir / "bronze" / "prices_daily.parquet")

    # Get tickers that have enough history
    ticker_counts = prices.group_by("ticker").agg(pl.len().alias("n_days"))
    min_days = 252  # at least 1 year
    valid_tickers = ticker_counts.filter(pl.col("n_days") >= min_days)["ticker"].to_list()
    logger.info("Tickers with >= %d days of data: %d", min_days, len(valid_tickers))

    # Filter prices to valid tickers
    prices = prices.filter(pl.col("ticker").is_in(valid_tickers))

    # Build universe config with actual tickers
    ticker_order = sorted(valid_tickers) + ["CASH"]  # CASH as last asset
    n_assets = len(ticker_order)
    logger.info("Universe: %d assets (%d equities + CASH)", n_assets, n_assets - 1)

    # Build weight caps
    max_weight = cfg.universe.max_single_name_weight
    weight_caps = np.array([max_weight] * (n_assets - 1) + [1.0])  # CASH uncapped

    # Pivot prices to wide format for returns matrix
    logger.info("Building returns matrix...")
    price_wide = prices.pivot(on="ticker", index="date", values="adj_close").sort("date")
    trading_dates = [d for d in price_wide["date"].to_list()]

    # Align to only dates where we have data
    equity_tickers = [t for t in ticker_order if t != "CASH"]
    # Ensure columns exist
    available_tickers = [t for t in equity_tickers if t in price_wide.columns]
    missing_tickers = set(equity_tickers) - set(available_tickers)
    if missing_tickers:
        logger.warning("Tickers missing from price pivot: %s", missing_tickers)

    ticker_order = available_tickers + ["CASH"]
    n_assets = len(ticker_order)
    weight_caps = np.array([max_weight] * (n_assets - 1) + [1.0])

    # Extract price matrix and compute returns
    price_matrix = price_wide.select(available_tickers).to_numpy()  # (n_dates, n_equities)

    # Add CASH column (constant price = 1, i.e., 0 return)
    cash_prices = np.ones((price_matrix.shape[0], 1))
    price_matrix_full = np.concatenate([price_matrix, cash_prices], axis=1)

    # Simple returns
    returns_matrix = np.zeros_like(price_matrix_full)
    returns_matrix[1:] = price_matrix_full[1:] / price_matrix_full[:-1] - 1
    # Handle NaN/inf in returns
    returns_matrix = np.nan_to_num(returns_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Risk-free returns (use 0 for now, could load from FF data)
    rf_returns = np.zeros(len(trading_dates))

    # ADV20 (20-day average dollar volume)
    logger.info("Computing ADV20...")
    dv_wide = prices.pivot(on="ticker", index="date", values="dollar_volume").sort("date")
    dv_matrix = dv_wide.select(available_tickers).to_numpy()
    dv_matrix = np.nan_to_num(dv_matrix, nan=0.0)

    # Rolling 20-day mean
    adv20_matrix = np.zeros_like(dv_matrix)
    for i in range(len(dv_matrix)):
        start = max(0, i - 19)
        adv20_matrix[i] = np.mean(dv_matrix[start : i + 1], axis=0)

    # Add CASH column (high ADV to avoid cost penalty)
    cash_adv = np.full((adv20_matrix.shape[0], 1), 1e12)
    adv20_matrix = np.concatenate([adv20_matrix, cash_adv], axis=1)

    # Rolling volatility (20-day)
    logger.info("Computing rolling volatility...")
    vol_matrix = np.zeros_like(returns_matrix)
    for i in range(20, len(returns_matrix)):
        vol_matrix[i] = np.nanstd(returns_matrix[i - 19 : i + 1], axis=0) * np.sqrt(252)
    vol_matrix[:20] = vol_matrix[20]  # backfill

    # Configure candidate search
    candidate_config = cfg.candidate_search.model_copy()
    if FAST_MODE:
        candidate_config.dirichlet_candidates_per_day = 500
        candidate_config.local_perturbations_per_seed = 10
        logger.info("FAST MODE: reduced to %d Dirichlet candidates/day",
                     candidate_config.dirichlet_candidates_per_day)
    else:
        # Still reduce from 5000 for the first full run — 500 tickers makes this expensive
        candidate_config.dirichlet_candidates_per_day = 1000
        candidate_config.local_perturbations_per_seed = 20
        logger.info("Using %d Dirichlet candidates/day",
                     candidate_config.dirichlet_candidates_per_day)

    horizon = cfg.horizons.primary  # 20 days

    # Only generate labels for dates where we have enough history AND forward data
    start_idx = 120  # need lookback for features
    n_eligible = len(trading_dates) - start_idx - horizon
    logger.info(
        "Generating labels for %d dates (horizon=%d, start=%s, end=%s)",
        n_eligible, horizon,
        trading_dates[start_idx], trading_dates[start_idx + n_eligible - 1],
    )

    # Run sequential simulation
    checkpoint_dir = data_dir / "labels" / "checkpoints"
    teacher_labels, teacher_candidates = run_sequential_simulation(
        trading_dates=trading_dates[start_idx:],
        returns_matrix=returns_matrix[start_idx:],
        rf_returns=rf_returns[start_idx:],
        adv20_matrix=adv20_matrix[start_idx:],
        rolling_vol_matrix=vol_matrix[start_idx:],
        rolling_cov_fn=None,  # Skip covariance for speed
        weight_caps=weight_caps,
        n_assets=n_assets,
        horizon=horizon,
        candidate_config=candidate_config,
        objective_config=cfg.teacher_objective,
        cost_config=cfg.cost_model,
        execution_config=cfg.execution,
        random_seed=cfg.random_seed,
        checkpoint_dir=str(checkpoint_dir),
        resume=True,
    )

    # Save results
    label_dir = data_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    if len(teacher_labels) > 0:
        teacher_labels.write_parquet(label_dir / "teacher_labels.parquet")
        logger.info("Saved %d teacher labels -> %s", len(teacher_labels), label_dir / "teacher_labels.parquet")
    else:
        logger.warning("No teacher labels generated!")

    if len(teacher_candidates) > 0:
        teacher_candidates.write_parquet(label_dir / "teacher_candidates.parquet")
        logger.info("Saved %d candidate records -> %s", len(teacher_candidates), label_dir / "teacher_candidates.parquet")

    # Save ticker order for later use
    manifest = {
        "ticker_order": ticker_order,
        "n_assets": n_assets,
        "horizon": horizon,
        "n_label_dates": len(teacher_labels) if len(teacher_labels) > 0 else 0,
        "fast_mode": FAST_MODE,
        "candidate_config": {
            "dirichlet_per_day": candidate_config.dirichlet_candidates_per_day,
            "local_perturbations": candidate_config.local_perturbations_per_seed,
        },
    }
    (label_dir / "label_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Done!")


if __name__ == "__main__":
    main()
