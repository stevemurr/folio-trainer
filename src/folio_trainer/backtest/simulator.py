"""Backtest simulator (spec 2.6)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from folio_trainer.backtest.execution_policy import apply_drift, apply_execution_policy
from folio_trainer.config.schema import CostModelConfig, ExecutionConfig
from folio_trainer.labels.cost_model import estimate_cost


@dataclass
class BacktestResult:
    """Container for backtest output."""

    dates: list[dt.date]
    daily_returns_gross: np.ndarray  # (n_days,)
    daily_returns_net: np.ndarray  # (n_days,)
    daily_weights: np.ndarray  # (n_days, n_assets)
    daily_turnover: np.ndarray  # (n_days,)
    daily_costs_bps: np.ndarray  # (n_days,)
    trade_flags: np.ndarray  # (n_days,) boolean
    ticker_names: list[str] = field(default_factory=list)


def simulate(
    weight_signals: np.ndarray,
    asset_returns: np.ndarray,
    dates: list[dt.date],
    execution_config: ExecutionConfig,
    cost_config: CostModelConfig,
    adv20: np.ndarray | None = None,
    rf_returns: np.ndarray | None = None,
    portfolio_value: float = 1_000_000.0,
    ticker_names: list[str] | None = None,
    initial_weights: np.ndarray | None = None,
) -> BacktestResult:
    """Run a daily backtest simulation.

    Parameters
    ----------
    weight_signals
        (n_days, n_assets) target weight signals for each day.
    asset_returns
        (n_days, n_assets) daily simple returns.
    dates
        List of dates corresponding to rows.
    execution_config
        Execution policy parameters.
    cost_config
        Cost model parameters.
    adv20
        (n_days, n_assets) or (n_assets,) average dollar volume.
        If 1D, constant across days.
    rf_returns
        (n_days,) risk-free returns. If None, assumed 0.
    portfolio_value
        Notional value for cost estimation.
    ticker_names
        Asset names for labeling.
    initial_weights
        Starting weights. If None, uses equal weight.

    Returns
    -------
    BacktestResult
        Complete backtest results.
    """
    n_days, n_assets = asset_returns.shape

    if rf_returns is None:
        rf_returns = np.zeros(n_days)

    if adv20 is None:
        adv20_daily = np.full((n_days, n_assets), 1e8)
    elif adv20.ndim == 1:
        adv20_daily = np.tile(adv20, (n_days, 1))
    else:
        adv20_daily = adv20

    # State
    if initial_weights is not None:
        live_weights = initial_weights.copy()
    else:
        live_weights = np.full(n_assets, 1.0 / n_assets)

    daily_returns_gross = np.zeros(n_days)
    daily_returns_net = np.zeros(n_days)
    daily_weights = np.zeros((n_days, n_assets))
    daily_turnover = np.zeros(n_days)
    daily_costs = np.zeros(n_days)
    trade_flags = np.zeros(n_days, dtype=bool)

    for t in range(n_days):
        # Apply execution policy
        target = weight_signals[t]
        exec_weights, traded = apply_execution_policy(target, live_weights, execution_config)
        trade_flags[t] = traded

        # Compute turnover
        daily_turnover[t] = 0.5 * np.sum(np.abs(exec_weights - live_weights))

        # Compute cost if traded
        if traded:
            cost_bps, _ = estimate_cost(
                exec_weights, live_weights, adv20_daily[t], portfolio_value, cost_config
            )
            daily_costs[t] = cost_bps
        else:
            exec_weights = live_weights.copy()

        daily_weights[t] = exec_weights

        # Compute portfolio return
        port_return = np.sum(exec_weights * asset_returns[t])
        daily_returns_gross[t] = port_return
        daily_returns_net[t] = port_return - daily_costs[t] / 10000  # bps to fraction

        # Drift weights to next day
        live_weights = apply_drift(exec_weights, asset_returns[t])

    return BacktestResult(
        dates=dates,
        daily_returns_gross=daily_returns_gross,
        daily_returns_net=daily_returns_net,
        daily_weights=daily_weights,
        daily_turnover=daily_turnover,
        daily_costs_bps=daily_costs,
        trade_flags=trade_flags,
        ticker_names=ticker_names or [],
    )
