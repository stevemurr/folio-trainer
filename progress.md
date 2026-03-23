# folio-trainer Implementation Progress

## Status: All 8 Phases Complete

| Phase | Description | Status | Modules |
|-------|-------------|--------|---------|
| 1 | Project Skeleton & Config | Complete | pyproject.toml, config/schema.py, config/loader.py, data/calendars.py, data/universe.py, cli.py |
| 2 | Data Ingestion & Normalization | Complete | ingest_prices.py, ingest_sec.py, ingest_fred.py, ingest_treasury.py, ingest_ff.py, normalize.py |
| 3 | Point-in-Time Joins & Features | Complete | point_in_time_join.py, asset_features.py, market_features.py, cross_asset_features.py |
| 4 | Teacher Label Generation | Complete | cost_model.py, candidate_generation.py, teacher_scoring.py, teacher_sequential_sim.py, distill_labels.py |
| 5 | Splits, Baselines & Backtest | Complete | make_splits.py, execution_policy.py, simulator.py, metrics.py, baselines.py, data_report.py |
| 6 | Direct Weight Model Training | Complete | dataset.py, losses.py, direct_weight_gbm.py, train_direct.py, tune.py |
| 7 | Evaluation & Robustness | Complete | evaluate_holdout.py, evaluate_walkforward.py, robustness.py, plots.py |
| 8 | Optional Models & CLI Wiring | Complete | direct_weight_nn.py, utility_scorer.py, train_utility.py, CLI fully wired |

## Verification
- 36 modules all import successfully
- 18 unit tests passing (config, calendar, universe)
- 10 CLI commands operational (`folio-trainer --help`)
- Smoke-tested: backtest simulator, candidate generation, metrics computation

## Architecture
- **36 Python modules** across 10 subpackages
- **Polars** primary DataFrame library
- **Pydantic v2** config validation
- **Click** CLI with 10 commands
- **LightGBM** primary model, **PyTorch** optional
- **Optuna** hyperparameter tuning
- Point-in-time joins with lookahead assertion
- Path-dependent sequential teacher label simulation with checkpoint/resume

## Next Steps
1. Provide price data and run `folio-trainer ingest --prices-file <path>`
2. Run `folio-trainer build-features` to compute feature tables
3. Run `folio-trainer make-splits` to generate train/val/test splits
4. Use programmatic API for label generation and model training
