"""Train the GBM direct weight model and run holdout evaluation.

Usage: python scripts/run_training.py
"""

import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    data_dir = Path("data")
    artifacts_dir = Path("artifacts")

    from folio_trainer.config.loader import load_config
    cfg = load_config()

    # -------------------------------------------------------------------------
    # 1. Load all data
    # -------------------------------------------------------------------------
    logger.info("Loading data...")
    prices = pl.read_parquet(data_dir / "bronze" / "prices_daily.parquet")
    asset_features = pl.read_parquet(data_dir / "features" / "features_asset_daily.parquet")
    labels = pl.read_parquet(data_dir / "labels" / "teacher_labels.parquet")
    splits = pl.read_parquet(data_dir / "splits" / "dataset_splits.parquet")
    manifest = json.loads((data_dir / "labels" / "label_manifest.json").read_text())
    ticker_order = manifest["ticker_order"]
    n_assets = manifest["n_assets"]

    logger.info("Assets: %d, Label dates: %d, Feature rows: %d",
                n_assets, len(labels), len(asset_features))

    # -------------------------------------------------------------------------
    # 2. Prepare dataset (join features + labels, split, preprocess)
    # -------------------------------------------------------------------------
    logger.info("Preparing dataset...")

    # Only use static splits for training
    static_splits = splits.filter(~pl.col("split_name").str.starts_with("walkforward"))

    from folio_trainer.splits.make_splits import get_split_dates
    train_start, train_end = get_split_dates(static_splits, "train")
    val_start, val_end = get_split_dates(static_splits, "val")
    test_start, test_end = get_split_dates(static_splits, "test")

    logger.info("Train: %s to %s", train_start, train_end)
    logger.info("Val:   %s to %s", val_start, val_end)
    logger.info("Test:  %s to %s", test_start, test_end)

    # Parse label weights into per-ticker targets
    logger.info("Parsing label weights...")
    label_rows = []
    for row in labels.iter_rows(named=True):
        soft_w = json.loads(row["soft_target_weights_json"])
        conf = row.get("teacher_confidence", 1.0)
        entry = {"asof_date": row["asof_date"], "confidence": conf}
        for i, ticker in enumerate(ticker_order):
            entry[f"target_{ticker}"] = soft_w[i] if i < len(soft_w) else 0.0
        label_rows.append(entry)
    label_df = pl.DataFrame(label_rows)

    # Join features with labels
    logger.info("Joining features with labels...")
    merged = asset_features.join(label_df, on="asof_date", how="inner")
    logger.info("Merged rows: %d", len(merged))

    # Split
    train_df = merged.filter(
        (pl.col("asof_date") >= train_start) & (pl.col("asof_date") <= train_end)
    )
    val_df = merged.filter(
        (pl.col("asof_date") >= val_start) & (pl.col("asof_date") <= val_end)
    )
    test_df = merged.filter(
        (pl.col("asof_date") >= test_start) & (pl.col("asof_date") <= test_end)
    )
    logger.info("Train: %d rows, Val: %d rows, Test: %d rows",
                len(train_df), len(val_df), len(test_df))

    # Identify feature columns
    exclude = {"asof_date", "ticker", "confidence", "missing_feature_count"}
    target_cols = {f"target_{t}" for t in ticker_order}
    exclude.update(target_cols)
    feature_cols = [c for c in merged.columns if c not in exclude]
    logger.info("Feature columns: %d", len(feature_cols))

    # Extract arrays
    # Key insight: with 469 assets, absolute soft-target weights are ~0.002 each,
    # giving the GBM almost no signal. Instead, train on log-relative weight vs
    # equal weight: log(w_target / w_equal). This centers the target around 0
    # with meaningful variance. At inference, exp(score) feeds into softmax.
    eqw = 1.0 / n_assets

    def extract(df):
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        tickers = df["ticker"].to_list()
        ticker_to_idx = {t: i for i, t in enumerate(ticker_order)}
        # Target = log(soft_target_weight / equal_weight)
        y = np.array([
            np.log(max(df[f"target_{t}"][i], 1e-8) / eqw)
            if f"target_{t}" in df.columns and t in ticker_to_idx
            else 0.0
            for i, t in enumerate(tickers)
        ], dtype=np.float32)
        w = df["confidence"].to_numpy().astype(np.float32) if "confidence" in df.columns else np.ones(len(df), dtype=np.float32)
        dates = df["asof_date"].to_list()
        unique_dates = sorted(set(dates))
        date_map = {d: i for i, d in enumerate(unique_dates)}
        date_indices = np.array([date_map[d] for d in dates])
        return X, y, w, unique_dates, date_indices

    X_train, y_train, w_train, train_dates, train_di = extract(train_df)
    X_val, y_val, w_val, val_dates, val_di = extract(val_df)
    X_test, y_test, w_test, test_dates, test_di = extract(test_df)

    # Impute NaN with train median, standardize
    logger.info("Preprocessing (impute + standardize on train)...")
    train_medians = np.nanmedian(X_train, axis=0)
    train_medians = np.nan_to_num(train_medians, nan=0.0)
    for X in (X_train, X_val, X_test):
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = train_medians[j]

    train_means = np.mean(X_train, axis=0)
    train_stds = np.std(X_train, axis=0)
    train_stds = np.where(train_stds < 1e-8, 1.0, train_stds)
    X_train = (X_train - train_means) / train_stds
    X_val = (X_val - train_means) / train_stds
    X_test = (X_test - train_means) / train_stds

    # Clip confidence weights
    w_train = np.clip(w_train, 0.1, 10.0)
    w_val = np.clip(w_val, 0.1, 10.0)

    # -------------------------------------------------------------------------
    # 3. Train model
    # -------------------------------------------------------------------------
    logger.info("Training GBM model...")
    from folio_trainer.models.direct_weight_gbm import DirectWeightGBM

    model = DirectWeightGBM(cfg.model.direct_weight_model)
    train_meta = model.train(
        X_train=X_train, y_train=y_train, w_train=w_train,
        X_val=X_val, y_val=y_val, w_val=w_val,
        feature_names=feature_cols,
    )
    logger.info("Best iteration: %d", train_meta.get("best_iteration", -1))

    # -------------------------------------------------------------------------
    # 4. Predict on all splits
    # -------------------------------------------------------------------------
    logger.info("Predicting weights...")
    val_pred_weights = model.predict_weights(X_val, val_di)
    test_pred_weights = model.predict_weights(X_test, test_di)

    # -------------------------------------------------------------------------
    # 5. Build returns matrices for backtest
    # -------------------------------------------------------------------------
    logger.info("Building returns matrices for backtest...")
    price_wide = prices.pivot(on="ticker", index="date", values="adj_close").sort("date")
    available_tickers = [t for t in ticker_order if t != "CASH" and t in price_wide.columns]

    price_mat = price_wide.select(available_tickers).to_numpy()
    cash_col = np.ones((price_mat.shape[0], 1))
    price_full = np.concatenate([price_mat, cash_col], axis=1)
    ret_full = np.zeros_like(price_full)
    ret_full[1:] = price_full[1:] / price_full[:-1] - 1
    ret_full = np.nan_to_num(ret_full, nan=0.0, posinf=0.0, neginf=0.0)

    all_dates = [d for d in price_wide["date"].to_list()]
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    def build_returns(split_dates):
        n = len(split_dates)
        ret = np.zeros((n, n_assets))
        for i, d in enumerate(split_dates):
            if d in date_to_idx:
                ret[i] = ret_full[date_to_idx[d]]
        return ret

    val_returns = build_returns(val_dates)
    test_returns = build_returns(test_dates)

    # -------------------------------------------------------------------------
    # 6. Run backtests
    # -------------------------------------------------------------------------
    logger.info("Running backtests...")
    from folio_trainer.backtest.simulator import simulate
    from folio_trainer.backtest.metrics import compute_metrics, metrics_to_dict
    from folio_trainer.backtest.baselines import equal_weight_strategy, inverse_volatility_strategy

    def reshape_weights(pred_w, date_indices, n_dates):
        """Reshape per-row weights to (n_dates, n_assets) matrix."""
        mat = np.zeros((n_dates, n_assets))
        for i, d_idx in enumerate(np.unique(date_indices)):
            mask = date_indices == d_idx
            w = pred_w[mask]
            if len(w) >= n_assets:
                mat[i] = w[:n_assets]
            elif len(w) > 0:
                mat[i, :len(w)] = w
                mat[i] = mat[i] / max(mat[i].sum(), 1e-10)
        return mat

    # Model weights
    val_weight_mat = reshape_weights(val_pred_weights, val_di, len(val_dates))
    test_weight_mat = reshape_weights(test_pred_weights, test_di, len(test_dates))

    # Baselines
    eqw_val = equal_weight_strategy(n_assets, len(val_dates))
    eqw_test = equal_weight_strategy(n_assets, len(test_dates))

    results = {}
    for name, weights, returns, dates in [
        ("model_val", val_weight_mat, val_returns, val_dates),
        ("model_test", test_weight_mat, test_returns, test_dates),
        ("eqw_val", eqw_val, val_returns, val_dates),
        ("eqw_test", eqw_test, test_returns, test_dates),
    ]:
        bt = simulate(weights, returns, dates, cfg.execution, cfg.cost_model)
        m = compute_metrics(bt, use_net=True)
        results[name] = metrics_to_dict(m)
        logger.info("%s: Sharpe=%.3f, CAGR=%.2f%%, MaxDD=%.2f%%",
                    name, m.sharpe, m.cagr * 100, m.max_drawdown * 100)

    # -------------------------------------------------------------------------
    # 7. Feature importance
    # -------------------------------------------------------------------------
    importance = model.feature_importance()
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:20]
    logger.info("Top 10 features:")
    for fname, fval in top_features[:10]:
        logger.info("  %s: %.0f", fname, fval)

    # -------------------------------------------------------------------------
    # 8. Save artifacts
    # -------------------------------------------------------------------------
    run_id = f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model.save(run_dir / "model.bin")

    import pickle
    with open(run_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump({"medians": train_medians, "means": train_means, "stds": train_stds}, f)

    import yaml
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False)

    (run_dir / "feature_manifest.json").write_text(json.dumps({
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "ticker_order": ticker_order,
    }, indent=2))

    (run_dir / "results.json").write_text(json.dumps({
        "run_id": run_id,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "best_iteration": train_meta.get("best_iteration"),
        "n_train_rows": len(X_train),
        "n_val_rows": len(X_val),
        "n_test_rows": len(X_test),
        "n_features": len(feature_cols),
        "metrics": results,
        "top_features": top_features,
    }, indent=2, default=str))

    # Save test predictions
    pred_df = pl.DataFrame({
        "date": test_dates,
        **{f"w_{ticker_order[j]}": test_weight_mat[:, j] for j in range(n_assets)},
    })
    pred_df.write_parquet(run_dir / "predictions_test.parquet")

    logger.info("Artifacts saved to %s", run_dir)

    # -------------------------------------------------------------------------
    # 9. Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nRun ID: {run_id}")
    print(f"Features: {len(feature_cols)}")
    print(f"Best iteration: {train_meta.get('best_iteration')}")
    print(f"\n{'Metric':<20} {'Model Val':>12} {'EQW Val':>12} {'Model Test':>12} {'EQW Test':>12}")
    print("-" * 68)
    for metric in ["Sharpe", "CAGR", "Ann. Vol", "Max DD", "Avg Turnover", "HHI"]:
        vals = [results[k].get(metric, 0) for k in ["model_val", "eqw_val", "model_test", "eqw_test"]]
        if metric in ("CAGR", "Ann. Vol", "Max DD", "Avg Turnover"):
            print(f"{metric:<20} {vals[0]:>11.2%} {vals[1]:>11.2%} {vals[2]:>11.2%} {vals[3]:>11.2%}")
        else:
            print(f"{metric:<20} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f} {vals[3]:>12.4f}")

    print(f"\nTop 5 features:")
    for fname, fval in top_features[:5]:
        print(f"  {fname}: {fval:.0f}")
    print(f"\nArtifacts: {run_dir}")


if __name__ == "__main__":
    main()
