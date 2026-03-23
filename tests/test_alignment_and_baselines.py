"""Tests for return alignment and baseline regressions."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from folio_trainer.backtest.baselines import equal_weight_strategy, previous_hold_strategy
from folio_trainer.backtest.data_utils import align_returns_to_dates, build_returns_by_date
from folio_trainer.backtest.metrics import compute_metrics
from folio_trainer.backtest.simulator import simulate
from folio_trainer.config.schema import CostModelConfig, ExecutionConfig


def test_align_returns_uses_prediction_dates():
    prices = pl.DataFrame(
        {
            "date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)],
            "ticker": ["AAA", "AAA", "AAA"],
            "adj_close": [100.0, 110.0, 121.0],
        }
    )
    _, returns_by_date = build_returns_by_date(prices, ["AAA", "CASH"])

    aligned = align_returns_to_dates(
        returns_by_date,
        [dt.date(2024, 1, 4)],
        2,
    )

    assert aligned.shape == (1, 2)
    assert aligned[0, 0] == pytest.approx(0.10)
    assert aligned[0, 1] == pytest.approx(0.0)


def test_baseline_metric_regression_fixed_slice():
    returns = np.array(
        [
            [0.01, -0.02],
            [0.03, 0.01],
            [-0.01, 0.02],
            [0.00, 0.01],
        ],
        dtype=float,
    )
    dates = [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(len(returns))]

    execution_config = ExecutionConfig(rebalance_band=0.0, partial_rebalance_alpha=1.0)
    cost_config = CostModelConfig(
        commission_bps=0.0,
        regulatory_bps=0.0,
        spread_bps_proxy_default=0.0,
        impact_coeff=0.0,
        liquidity_buckets=[],
    )

    eqw_bt = simulate(
        equal_weight_strategy(2, len(returns)),
        returns,
        dates,
        execution_config,
        cost_config,
    )
    hold_bt = simulate(
        previous_hold_strategy(2, len(returns), asset_returns=returns),
        returns,
        dates,
        execution_config,
        cost_config,
    )

    eqw_metrics = compute_metrics(eqw_bt, use_net=True)
    hold_metrics = compute_metrics(hold_bt, use_net=True)

    assert eqw_metrics.sharpe == pytest.approx(11.114378604524228)
    assert eqw_metrics.total_return == pytest.approx(0.025074372499999775)
    assert eqw_metrics.avg_turnover == pytest.approx(0.004975583948422235)

    assert hold_metrics.sharpe == pytest.approx(10.902292894568998)
    assert hold_metrics.total_return == pytest.approx(0.02479447999999973)
    assert hold_metrics.avg_turnover == pytest.approx(4.85722573273506e-17)
