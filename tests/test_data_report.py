"""Tests for the expanded research data report."""

from __future__ import annotations

import datetime as dt
import json

import polars as pl

from folio_trainer.reports.data_report import generate_data_report


def test_data_report_includes_research_audit_sections(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "bronze").mkdir(parents=True)
    (data_dir / "features").mkdir(parents=True)
    (data_dir / "labels").mkdir(parents=True)
    (data_dir / "splits").mkdir(parents=True)

    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]

    pl.DataFrame(
        {
            "date": [dates[0], dates[1]],
            "ticker": ["AAA", "AAA"],
            "adj_close": [100.0, 101.0],
        }
    ).write_parquet(data_dir / "bronze" / "prices_daily.parquet")

    pl.DataFrame(
        {
            "asof_date": [dates[0], dates[1]],
            "ticker": ["AAA", "AAA"],
            "ret_1": [0.0, 0.01],
            "prev_live_weight": [0.5, 0.5],
            "prev_target_weight": [0.5, 0.4],
            "delta_from_prev_target": [0.0, 0.1],
            "missing_feature_count": [0, 0],
        }
    ).write_parquet(data_dir / "features" / "features_asset_daily.parquet")
    pl.DataFrame({"asof_date": dates, "vix": [15.0, 16.0]}).write_parquet(
        data_dir / "features" / "features_market_daily.parquet"
    )
    pl.DataFrame({"asof_date": dates, "avg_pairwise_corr_20": [0.2, 0.3]}).write_parquet(
        data_dir / "features" / "features_cross_asset_daily.parquet"
    )

    pl.DataFrame(
        {
            "asof_date": [dates[0]],
            "prediction_date": [dates[1]],
            "horizon_h": [20],
            "prev_live_weights_json": [json.dumps([0.5, 0.5])],
            "hard_target_weights_json": [json.dumps([0.7, 0.3])],
            "soft_target_weights_json": [json.dumps([0.6, 0.4])],
            "teacher_confidence": [0.5],
            "best_objective": [1.0],
        }
    ).write_parquet(data_dir / "labels" / "teacher_labels.parquet")

    pl.DataFrame(
        {
            "asof_date": [dates[0], dates[0]],
            "prediction_date": [dates[1], dates[1]],
            "horizon_h": [20, 20],
            "candidate_id": ["d0", "d1"],
            "candidate_type": ["deterministic", "random"],
            "weights_json": [json.dumps([0.7, 0.3]), json.dumps([0.4, 0.6])],
            "objective_total": [1.0, 0.8],
            "objective_sharpe": [1.0, 0.8],
            "objective_return": [0.2, 0.1],
            "objective_vol": [0.1, 0.2],
            "turnover": [0.1, 0.2],
            "est_cost": [1.0, 2.0],
            "concentration_hhi": [0.58, 0.52],
            "rank": [0, 1],
        }
    ).write_parquet(data_dir / "labels" / "teacher_candidates.parquet")

    pl.DataFrame(
        {
            "split_name": ["train", "val", "test"],
            "start_date": [dates[0], dates[1], dates[1]],
            "end_date": [dates[0], dates[1], dates[1]],
            "purge_start": [dates[0], dates[1], dates[1]],
            "purge_end": [dates[0], dates[1], dates[1]],
        }
    ).write_parquet(data_dir / "splits" / "dataset_splits.parquet")

    report = generate_data_report(data_dir)

    assert "## Feature Block Coverage" in report
    assert "Soft target concentration" in report
    assert "Best-candidate win counts by type" in report
    assert "## Split Coverage" in report
