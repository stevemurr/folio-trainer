# folio-trainer

`folio-trainer` is a point-in-time daily asset allocation training pipeline. It ingests market data, builds feature tables, generates teacher portfolio labels, trains a direct-weight model, evaluates it on holdout data, and saves artifacts for later prediction.

This repository currently uses a split workflow:

1. CLI commands handle data prep, reporting, and prediction.
2. `scripts/run_label_generation.py` handles teacher label generation.
3. `scripts/run_training.py` handles end-to-end model training, holdout evaluation, and benchmark runs.

For the detailed design spec, see [`docs/allocation_model_spec.md`](docs/allocation_model_spec.md). For the default config, see [`docs/allocation_model_default_config.yaml`](docs/allocation_model_default_config.yaml).

## Quick Start

Run the commands below from the repository root.

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

### 2. Provide price data

Recommended: bring your own CSV or Parquet with these required columns:

```text
date,ticker,open,high,low,close,adj_close,volume
```

Optional: generate a sample S&P 500 price file with the helper script:

```bash
python -m pip install yfinance
python scripts/download_sp500_prices.py
```

That script writes `data/raw/prices/prices_daily.parquet`, which can be passed directly into the ingest step below.

### 3. Ingest and normalize data

```bash
folio-trainer ingest --prices-file data/raw/prices/prices_daily.parquet
```

What this does now:

1. Copies your price file into `data/raw/prices/`.
2. Normalizes prices into `data/bronze/prices_daily.parquet`.
3. Also attempts to fetch Treasury and Fama-French data.
4. Fetches FRED macro data only if `FRED_API_KEY` is set or `--fred-api-key` is passed.

Example with FRED enabled:

```bash
export FRED_API_KEY=your_key
folio-trainer ingest --prices-file data/raw/prices/prices_daily.parquet \
  --fred-api-key "$FRED_API_KEY"
```

### 4. Build features

```bash
folio-trainer build-features
```

This writes:

```text
data/features/features_asset_daily.parquet
data/features/features_market_daily.parquet
data/features/features_cross_asset_daily.parquet
```

### 5. Generate teacher labels

Current working entry point:

```bash
python scripts/run_label_generation.py --fast
```

Use `--fast` for iteration. Drop it for a fuller run.

This writes:

```text
data/labels/teacher_labels.parquet
data/labels/teacher_candidates.parquet
data/labels/label_manifest.json
```

### 6. Make splits and run the data report

```bash
folio-trainer make-splits
folio-trainer run-data-report
```

This writes:

```text
data/splits/dataset_splits.parquet
data/reports/data_quality_report.md
```

### 7. Train and evaluate

Current working entry point:

```bash
python scripts/run_training.py
```

Useful variants:

```bash
python scripts/run_training.py --skip-benchmark
python scripts/run_training.py --strategy aggressive
python scripts/run_training.py --model-kind linear
```

A successful run writes artifacts under `artifacts/runs/<run_id>/`, including:

```text
model.bin
preprocessor.pkl
config.yaml
feature_manifest.json
run_metadata.json
eval_report.md
eval_results.json
predictions_test.parquet
benchmark/
```

### 8. Predict a portfolio from a saved run

```bash
folio-trainer predict --top 20
```

Or target a specific run and date:

```bash
folio-trainer predict --run-id <run_id> --date 2025-12-31 --top 20
```

If `--run-id` is omitted, the CLI uses the latest run in `artifacts/runs/`.

## Current Flow

At a high level, the repo works like this:

1. Price and macro data are stored under `data/raw/` and normalized into `data/bronze/`.
2. Feature builders create asset-level, market-level, and cross-asset daily feature tables.
3. The teacher simulation searches candidate portfolios and distills them into soft target weights.
4. Split generation creates static and walk-forward date boundaries.
5. Training builds a per-asset supervised dataset, fits a direct-weight model, backtests it on holdout dates, and compares it with baseline policies.
6. Prediction reloads a saved model and preprocessor, recomputes features for a target date, and outputs portfolio weights on the simplex.

## Current Entry Points

Use these commands for the working flow today:

- `folio-trainer ingest`
- `folio-trainer build-features`
- `folio-trainer make-splits`
- `folio-trainer run-data-report`
- `folio-trainer predict`
- `python scripts/run_label_generation.py`
- `python scripts/run_training.py`

The following CLI commands exist but are still lightweight wrappers or placeholders relative to the script-based flow:

- `folio-trainer build-labels`
- `folio-trainer train-direct`
- `folio-trainer train-utility`
- `folio-trainer eval-holdout`
- `folio-trainer eval-walkforward`
- `folio-trainer backtest-baselines`

## Repo Layout

```text
src/folio_trainer/    Core pipeline modules
scripts/              Script entry points for label generation and training
docs/                 Spec and default config
data/                 Local datasets, feature tables, labels, splits, reports
artifacts/            Saved model runs and evaluation outputs
tests/                Unit tests
```
