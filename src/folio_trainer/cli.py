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
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, data_dir: str) -> None:
    """folio-trainer: Point-in-time daily predictive asset allocation pipeline."""
    ctx.ensure_object(dict)
    cfg = load_config(config_path, overrides={"data_dir": data_dir})
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


def _try_read(path: Path):
    """Try to read a Parquet file, return None if not found."""
    import polars as pl
    if path.exists():
        return pl.read_parquet(path)
    return None
