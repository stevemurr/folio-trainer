"""Tests for the unified dataset builder."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from folio_trainer.models.dataset import build_feature_table, prepare_dataset, stack_by_date


def test_prepare_dataset_joins_feature_blocks_and_state_features():
    ticker_order = ["AAA", "BBB", "CASH"]
    dates = [
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
        dt.date(2024, 1, 4),
        dt.date(2024, 1, 5),
    ]

    asset_rows = []
    for day_idx, date in enumerate(dates):
        for ticker_idx, ticker in enumerate(ticker_order):
            asset_rows.append(
                {
                    "asof_date": date,
                    "ticker": ticker,
                    "ret_1": float(day_idx + ticker_idx),
                    "vol_20": float(0.1 + ticker_idx),
                    "prev_live_weight": None,
                    "prev_target_weight": None,
                    "delta_from_prev_target": None,
                    "missing_feature_count": 3,
                }
            )
    asset_features = pl.DataFrame(asset_rows)

    market_features = pl.DataFrame(
        {"asof_date": dates, "vix": [15.0, 16.0, 17.0, 18.0]}
    )
    cross_features = pl.DataFrame(
        {"asof_date": dates, "avg_pairwise_corr_20": [0.2, 0.25, 0.3, 0.35]}
    )

    labels = pl.DataFrame(
        {
            "asof_date": dates[:-1],
            "prediction_date": dates[1:],
            "horizon_h": [20, 20, 20],
            "prev_live_weights_json": [
                json.dumps([1 / 3, 1 / 3, 1 / 3]),
                json.dumps([0.55, 0.25, 0.20]),
                json.dumps([0.20, 0.55, 0.25]),
            ],
            "hard_target_weights_json": [
                json.dumps([0.60, 0.20, 0.20]),
                json.dumps([0.20, 0.60, 0.20]),
                json.dumps([0.20, 0.20, 0.60]),
            ],
            "soft_target_weights_json": [
                json.dumps([0.50, 0.30, 0.20]),
                json.dumps([0.25, 0.55, 0.20]),
                json.dumps([0.20, 0.25, 0.55]),
            ],
            "teacher_confidence": [0.9, 0.8, 0.7],
            "best_objective": [1.0, 1.1, 1.2],
        }
    )

    feature_table = build_feature_table(
        asset_features=asset_features,
        labels=labels,
        ticker_order=ticker_order,
        market_features=market_features,
        cross_asset_features=cross_features,
    )
    state_subset = feature_table.filter(pl.col("asof_date").is_in(dates[:-1]))
    assert state_subset["prev_live_weight"].null_count() == 0
    assert state_subset["prev_target_weight"].null_count() == 0
    assert state_subset["delta_from_prev_target"].null_count() == 0

    splits = pl.DataFrame(
        {
            "split_name": ["train", "val", "test"],
            "start_date": [dates[0], dates[1], dates[2]],
            "end_date": [dates[0], dates[1], dates[2]],
            "purge_start": [dates[0], dates[1], dates[2]],
            "purge_end": [dates[0], dates[1], dates[2]],
        }
    )

    dataset = prepare_dataset(
        asset_features=asset_features,
        labels=labels,
        splits=splits,
        ticker_order=ticker_order,
        market_features=market_features,
        cross_asset_features=cross_features,
        calendar_dates=dates,
    )

    assert "vix" in dataset.feature_names
    assert "avg_pairwise_corr_20" in dataset.feature_names
    assert "prev_live_weight" in dataset.feature_names
    assert dataset.train_prediction_dates == [dates[1]]
    assert dataset.val_prediction_dates == [dates[2]]
    assert dataset.test_prediction_dates == [dates[3]]

    expected = np.log(0.50 / (1 / 3))
    assert dataset.y_train[0] == pytest.approx(expected)
    assert dataset.target_weight_train[0] == pytest.approx(0.50)

    train_target_matrix = stack_by_date(
        dataset.target_weight_train,
        dataset.train_date_indices,
        len(ticker_order),
    )
    assert train_target_matrix.shape == (1, 3)
    assert train_target_matrix[0].tolist() == pytest.approx([0.50, 0.30, 0.20])
