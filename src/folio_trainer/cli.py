"""CLI entry points for folio-trainer pipeline."""

from __future__ import annotations

from pathlib import Path

import click

from folio_trainer.config.loader import load_config


@click.group()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML config file. Uses defaults if not provided.",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="data",
    help="Root directory for data storage.",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["aggressive", "neutral", "conservative"], case_sensitive=False),
    default=None,
    help="Named strategy profile for risk/return tradeoff. Overrides teacher lambdas and temperatures.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, data_dir: str, strategy: str | None) -> None:
    """folio-trainer: Point-in-time daily predictive asset allocation pipeline."""
    ctx.ensure_object(dict)
    cfg = load_config(config_path, overrides={"data_dir": data_dir}, strategy=strategy)
    ctx.obj["config"] = cfg
    ctx.obj["data_dir"] = Path(data_dir)


# ---------------------------------------------------------------------------
# Part 1 — Data preparation commands
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--source", type=str, default=None, help="Specific source to ingest (prices, sec, fred, treasury, ff). Default: all.")
@click.option("--prices-file", type=click.Path(exists=True), default=None, help="Path to user-supplied OHLCV CSV/Parquet.")
@click.option("--fred-api-key", envvar="FRED_API_KEY", default=None, help="FRED API key (or set FRED_API_KEY env var).")
@click.option("--force", is_flag=True, help="Overwrite existing data.")
@click.pass_context
def ingest(ctx: click.Context, source: str | None, prices_file: str | None, fred_api_key: str | None, force: bool) -> None:
    """Ingest raw data from configured sources."""
    import logging

    from folio_trainer.data.ingest_prices import ingest_prices
    from folio_trainer.data.ingest_treasury import ingest_treasury_yields
    from folio_trainer.data.ingest_ff import ingest_ff_factors
    from folio_trainer.data.ingest_fred import ingest_fred
    from folio_trainer.data.normalize import normalize_all

    logging.basicConfig(level=logging.INFO)
    data_dir = ctx.obj["data_dir"]
    sources = [source] if source else ["prices", "treasury", "ff", "fred"]

    for src in sources:
        if src == "prices":
            if not prices_file:
                click.echo("Skipping prices: --prices-file not provided.")
                continue
            ingest_prices(prices_file, data_dir, force=force)
        elif src == "treasury":
            ingest_treasury_yields(data_dir, force=force)
        elif src == "ff":
            ingest_ff_factors(data_dir, force=force)
        elif src == "fred":
            if not fred_api_key:
                click.echo("Skipping FRED: --fred-api-key not provided.")
                continue
            ingest_fred(fred_api_key, data_dir, force=force)
        elif src == "sec":
            click.echo("SEC ingestion requires CIK mapping. Use build-features pipeline.")
        else:
            click.echo(f"Unknown source: {src}")

    # Normalize all available raw data to bronze
    normalize_all(data_dir)
    click.echo("Ingestion complete.")


@cli.command("build-features")
@click.option("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD).")
@click.option("--end-date", type=str, default=None, help="End date (YYYY-MM-DD).")
@click.pass_context
def build_features(ctx: click.Context, start_date: str | None, end_date: str | None) -> None:
    """Compute feature tables (asset, market, cross-asset)."""
    import logging
    import polars as pl
    from folio_trainer.features.asset_features import compute_asset_features
    from folio_trainer.features.market_features import compute_market_features
    from folio_trainer.features.cross_asset_features import compute_cross_asset_features

    logging.basicConfig(level=logging.INFO)
    cfg = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]

    # Load bronze prices
    prices_file = data_dir / "bronze" / "prices_daily.parquet"
    if not prices_file.exists():
        click.echo("Error: No bronze prices found. Run 'ingest' first.")
        return
    prices = pl.read_parquet(prices_file)

    # Asset features
    click.echo("Computing asset features...")
    asset_feats = compute_asset_features(prices, cfg.features)
    feat_dir = data_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    asset_feats.write_parquet(feat_dir / "features_asset_daily.parquet")

    # Market features (load available data)
    click.echo("Computing market features...")
    treasury = _try_read(data_dir / "bronze" / "treasury_yields.parquet")
    ff = _try_read(data_dir / "bronze" / "ff_factors_daily.parquet")
    macro = _try_read(data_dir / "bronze" / "macro_series_daily.parquet")
    vix = macro.filter(pl.col("series_id") == "VIXCLS") if macro is not None and "series_id" in macro.columns else None
    market_feats = compute_market_features(treasury, ff, macro, vix)
    if len(market_feats) > 0:
        market_feats.write_parquet(feat_dir / "features_market_daily.parquet")

    # Cross-asset features
    click.echo("Computing cross-asset features...")
    cross_feats = compute_cross_asset_features(prices, cfg.features)
    if len(cross_feats) > 0:
        cross_feats.write_parquet(feat_dir / "features_cross_asset_daily.parquet")

    click.echo("Feature computation complete.")


@cli.command("build-labels")
@click.option("--start-date", type=str, default=None)
@click.option("--end-date", type=str, default=None)
@click.option("--resume", is_flag=True, help="Resume from last checkpoint.")
@click.pass_context
def build_labels(ctx: click.Context, start_date: str | None, end_date: str | None, resume: bool) -> None:
    """Generate teacher candidates and distill labels."""
    import logging
    logging.basicConfig(level=logging.INFO)
    cfg = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    click.echo(f"[build-labels] Would run sequential simulation for horizon={cfg.horizons.primary}")
    click.echo("This command requires price data and features. See docs for full pipeline.")


@cli.command("make-splits")
@click.pass_context
def make_splits(ctx: click.Context) -> None:
    """Generate train/val/test splits with purge buffers."""
    import logging
    import polars as pl
    from folio_trainer.splits.make_splits import make_static_splits, make_walkforward_splits
    from folio_trainer.data.calendars import get_trading_days

    logging.basicConfig(level=logging.INFO)
    cfg = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]

    # Get date range from price data
    prices_file = data_dir / "bronze" / "prices_daily.parquet"
    if not prices_file.exists():
        click.echo("Error: No bronze prices found. Run 'ingest' first.")
        return
    prices = pl.read_parquet(prices_file)
    trading_dates = sorted(prices["date"].unique().to_list())

    # Static splits
    static = make_static_splits(trading_dates, cfg.splits)
    click.echo(f"Static splits: {len(static)} entries")

    # Walk-forward splits
    wf = make_walkforward_splits(trading_dates, cfg.splits)
    click.echo(f"Walk-forward splits: {len(wf)} entries")

    # Combine and save
    all_splits = pl.concat([static, wf]) if len(wf) > 0 else static
    split_dir = data_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    all_splits.write_parquet(split_dir / "dataset_splits.parquet")
    click.echo(f"Splits saved to {split_dir / 'dataset_splits.parquet'}")


@cli.command("run-data-report")
@click.pass_context
def run_data_report(ctx: click.Context) -> None:
    """Generate data quality and leakage audit reports."""
    import logging
    from folio_trainer.reports.data_report import generate_data_report

    logging.basicConfig(level=logging.INFO)
    data_dir = ctx.obj["data_dir"]
    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = generate_data_report(data_dir, report_dir / "data_quality_report.md")
    click.echo(f"Report generated: {report_dir / 'data_quality_report.md'}")


# ---------------------------------------------------------------------------
# Part 2 — Model commands
# ---------------------------------------------------------------------------


@cli.command("train-direct")
@click.option("--run-id", type=str, default=None, help="Custom run identifier.")
@click.pass_context
def train_direct(ctx: click.Context, run_id: str | None) -> None:
    """Train the direct weight prediction model."""
    import logging
    logging.basicConfig(level=logging.INFO)
    cfg = ctx.obj["config"]
    click.echo(f"[train-direct] run_id={run_id}")
    click.echo("Requires prepared dataset. See train_direct.py for programmatic usage.")


@cli.command("train-utility")
@click.option("--run-id", type=str, default=None)
@click.pass_context
def train_utility(ctx: click.Context, run_id: str | None) -> None:
    """Train the optional utility scorer model."""
    import logging
    logging.basicConfig(level=logging.INFO)
    cfg = ctx.obj["config"]
    if not cfg.model.utility_model.enabled:
        click.echo("Utility model disabled in config.")
        return
    click.echo(f"[train-utility] run_id={run_id}")
    click.echo("Requires candidate data. See train_utility.py for programmatic usage.")


@cli.command("eval-holdout")
@click.option("--run-id", type=str, required=True, help="Run ID of the trained model.")
@click.pass_context
def eval_holdout(ctx: click.Context, run_id: str) -> None:
    """Evaluate model on holdout test split."""
    import logging
    logging.basicConfig(level=logging.INFO)
    click.echo(f"[eval-holdout] run_id={run_id}")
    click.echo("See evaluate_holdout.py for programmatic usage.")


@cli.command("eval-walkforward")
@click.option("--run-id", type=str, required=True)
@click.pass_context
def eval_walkforward(ctx: click.Context, run_id: str) -> None:
    """Evaluate model with walk-forward retraining."""
    import logging
    logging.basicConfig(level=logging.INFO)
    click.echo(f"[eval-walkforward] run_id={run_id}")
    click.echo("See evaluate_walkforward.py for programmatic usage.")


@cli.command("backtest-baselines")
@click.pass_context
def backtest_baselines(ctx: click.Context) -> None:
    """Run all baseline strategy backtests."""
    import logging
    logging.basicConfig(level=logging.INFO)
    click.echo("[backtest-baselines]")
    click.echo("Requires price data and splits. See baselines.py for programmatic usage.")


@cli.command("predict")
@click.option("--tickers", type=str, default=None, help="Comma-separated tickers (default: all from model).")
@click.option("--date", "pred_date", type=str, default=None, help="Prediction date YYYY-MM-DD (default: latest in data).")
@click.option("--run-id", type=str, default=None, help="Model run ID (default: latest run).")
@click.option("--top", type=int, default=None, help="Show only top N holdings.")
@click.pass_context
def predict(ctx: click.Context, tickers: str | None, pred_date: str | None, run_id: str | None, top: int | None) -> None:
    """Predict optimal portfolio allocation for given tickers and date."""
    import datetime as dt
    import json
    import logging
    import pickle

    import numpy as np
    import polars as pl

    from folio_trainer.features.asset_features import compute_asset_features
    from folio_trainer.models.direct_weight_gbm import DirectWeightGBM

    logging.basicConfig(level=logging.WARNING)
    cfg = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    artifacts_dir = Path(cfg.artifacts_dir)

    # --- Resolve model run ---
    if run_id is None:
        runs_dir = artifacts_dir / "runs"
        if not runs_dir.exists():
            click.echo("Error: No model runs found in artifacts/runs/")
            return
        runs = sorted(runs_dir.iterdir())
        if not runs:
            click.echo("Error: No model runs found.")
            return
        run_dir = runs[-1]
        run_id = run_dir.name
    else:
        run_dir = artifacts_dir / "runs" / run_id

    if not run_dir.exists():
        click.echo(f"Error: Run directory not found: {run_dir}")
        return

    # --- Load model, preprocessor, manifest ---
    model = DirectWeightGBM.load(run_dir / "model.bin")
    with open(run_dir / "preprocessor.pkl", "rb") as f:
        preproc = pickle.load(f)
    manifest = json.loads((run_dir / "feature_manifest.json").read_text())
    feature_names = manifest["feature_names"]
    model_tickers = [t for t in manifest["ticker_order"] if t != "CASH"]

    # --- Resolve tickers ---
    if tickers is not None:
        requested = [t.strip().upper() for t in tickers.split(",")]
    else:
        requested = model_tickers

    # --- Load price data ---
    prices_file = data_dir / "bronze" / "prices_daily.parquet"
    if not prices_file.exists():
        click.echo("Error: No price data found. Run 'ingest' first.")
        return
    prices = pl.read_parquet(prices_file)

    # Filter to requested tickers
    available = set(prices["ticker"].unique().to_list())
    missing = [t for t in requested if t not in available]
    if missing:
        click.echo(f"Warning: Tickers not in price data (skipped): {', '.join(missing)}")
    use_tickers = [t for t in requested if t in available]
    if not use_tickers:
        click.echo("Error: No valid tickers.")
        return

    prices = prices.filter(pl.col("ticker").is_in(use_tickers))

    # --- Resolve date ---
    all_dates = sorted(prices["date"].unique().to_list())
    if pred_date is not None:
        target_date = dt.date.fromisoformat(pred_date)
    else:
        target_date = all_dates[-1]

    if target_date not in all_dates:
        # Find the closest date <= target
        earlier = [d for d in all_dates if d <= target_date]
        if not earlier:
            click.echo(f"Error: No price data on or before {target_date}")
            return
        target_date = earlier[-1]

    # --- Compute features ---
    asset_feats = compute_asset_features(prices, cfg.features)
    target_feats = asset_feats.filter(pl.col("asof_date") == target_date)

    if len(target_feats) == 0:
        click.echo(f"Error: No features computed for {target_date}. Need enough price history.")
        return

    # Align tickers to requested order
    feat_tickers = target_feats["ticker"].to_list()

    # Fill portfolio context features with equal-weight defaults
    n = len(feat_tickers)
    eqw = 1.0 / n if n > 0 else 0.0
    target_feats = target_feats.with_columns(
        pl.lit(eqw).alias("prev_live_weight"),
        pl.lit(eqw).alias("prev_target_weight"),
        pl.lit(0.0).alias("delta_from_prev_target"),
    )

    # Extract feature matrix in the correct column order
    available_features = [f for f in feature_names if f in target_feats.columns]
    missing_features = [f for f in feature_names if f not in target_feats.columns]
    X = target_feats.select(available_features).to_numpy().astype(np.float32)

    # Add zero columns for any missing features
    if missing_features:
        zeros = np.zeros((X.shape[0], len(missing_features)), dtype=np.float32)
        # Rebuild in correct order
        full_X = np.zeros((X.shape[0], len(feature_names)), dtype=np.float32)
        for i, fname in enumerate(feature_names):
            if fname in available_features:
                col_idx = available_features.index(fname)
                full_X[:, i] = X[:, col_idx]
        X = full_X

    # --- Preprocess ---
    medians = preproc["medians"]
    means = preproc["means"]
    stds = preproc["stds"]

    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if j < len(medians):
            X[mask, j] = medians[j]

    X = (X - means[:X.shape[1]]) / stds[:X.shape[1]]

    # --- Predict ---
    date_indices = np.zeros(len(feat_tickers), dtype=np.int64)
    weights = model.predict_weights(X, date_indices)

    # --- Output ---
    results = sorted(zip(feat_tickers, weights), key=lambda x: -x[1])
    if top is not None:
        display_results = results[:top]
    else:
        display_results = results

    strategy_label = f", strategy: {cfg.strategy}" if cfg.strategy else ""
    click.echo(f"\nPredicted allocation for {target_date} (model: {run_id}{strategy_label})")
    click.echo(f"{'Ticker':<10} {'Weight':>10}")
    click.echo("-" * 22)
    for ticker, w in display_results:
        click.echo(f"{ticker:<10} {w:>9.2%}")
    click.echo("-" * 22)
    click.echo(f"{'Total':<10} {sum(w for _, w in results):>9.2%}")

    # Summary stats
    top10_wt = sum(w for _, w in results[:10])
    active = sum(1 for _, w in results if w > 0.01)
    hhi = sum(w**2 for _, w in results)
    click.echo(f"\nTop 10:      {top10_wt:>7.2%}")
    click.echo(f"Active (>1%): {active:>5}")
    click.echo(f"HHI:          {hhi:>7.4f}")


def _try_read(path: Path):
    """Try to read a Parquet file, return None if not found."""
    import polars as pl
    if path.exists():
        return pl.read_parquet(path)
    return None
