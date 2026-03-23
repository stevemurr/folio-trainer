"""Portfolio performance metrics (spec 2.7.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from folio_trainer.backtest.simulator import BacktestResult


@dataclass
class PortfolioMetrics:
    """Computed performance metrics for a backtest."""

    cagr: float
    annualized_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    avg_turnover: float
    avg_active_positions: float
    avg_cash_weight: float
    hhi: float
    total_return: float
    n_trades: int
    n_days: int


def compute_metrics(
    result: BacktestResult,
    use_net: bool = True,
    rf_annual: float = 0.0,
    cash_index: int | None = None,
) -> PortfolioMetrics:
    """Compute standard portfolio metrics from backtest results.

    Parameters
    ----------
    result
        BacktestResult from the simulator.
    use_net
        If True, use net-of-cost returns; else gross.
    rf_annual
        Annualized risk-free rate for Sharpe/Sortino.
    cash_index
        Index of cash asset in weight vector (for avg cash weight).

    Returns
    -------
    PortfolioMetrics
    """
    returns = result.daily_returns_net if use_net else result.daily_returns_gross
    n = len(returns)

    if n == 0:
        return PortfolioMetrics(
            cagr=0, annualized_vol=0, sharpe=0, sortino=0,
            max_drawdown=0, calmar=0, avg_turnover=0,
            avg_active_positions=0, avg_cash_weight=0, hhi=0,
            total_return=0, n_trades=0, n_days=0,
        )

    # Total return
    cumulative = np.prod(1 + returns)
    total_return = cumulative - 1

    # CAGR
    years = n / 252
    cagr = cumulative ** (1 / max(years, 1e-6)) - 1 if cumulative > 0 else -1.0

    # Annualized vol
    ann_vol = np.std(returns) * np.sqrt(252)

    # Daily risk-free
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1

    # Sharpe
    excess = returns - rf_daily
    sharpe = np.mean(excess) / max(np.std(excess), 1e-10) * np.sqrt(252)

    # Sortino (downside deviation)
    downside = np.minimum(excess, 0)
    downside_std = np.sqrt(np.mean(downside ** 2))
    sortino = np.mean(excess) / max(downside_std, 1e-10) * np.sqrt(252)

    # Max drawdown
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / running_max - 1
    max_dd = float(np.min(drawdowns))

    # Calmar
    calmar = cagr / max(abs(max_dd), 1e-10)

    # Turnover
    avg_turnover = float(np.mean(result.daily_turnover))

    # Active positions (weight > 1%)
    active = np.sum(result.daily_weights > 0.01, axis=1)
    avg_active = float(np.mean(active))

    # Cash weight
    if cash_index is not None:
        avg_cash = float(np.mean(result.daily_weights[:, cash_index]))
    else:
        avg_cash = 0.0

    # HHI (average concentration)
    daily_hhi = np.sum(result.daily_weights ** 2, axis=1)
    avg_hhi = float(np.mean(daily_hhi))

    return PortfolioMetrics(
        cagr=cagr,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        avg_turnover=avg_turnover,
        avg_active_positions=avg_active,
        avg_cash_weight=avg_cash,
        hhi=avg_hhi,
        total_return=total_return,
        n_trades=int(np.sum(result.trade_flags)),
        n_days=n,
    )


def metrics_to_dict(m: PortfolioMetrics) -> dict[str, float]:
    """Convert PortfolioMetrics to a flat dict for reporting."""
    return {
        "CAGR": m.cagr,
        "Ann. Vol": m.annualized_vol,
        "Sharpe": m.sharpe,
        "Sortino": m.sortino,
        "Max DD": m.max_drawdown,
        "Calmar": m.calmar,
        "Avg Turnover": m.avg_turnover,
        "Avg Active Pos": m.avg_active_positions,
        "Avg Cash Wt": m.avg_cash_weight,
        "HHI": m.hhi,
        "Total Return": m.total_return,
        "N Trades": m.n_trades,
        "N Days": m.n_days,
    }
