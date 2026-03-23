"""Teacher candidate scoring with forward returns (spec 1.8.4)."""

from __future__ import annotations

import numpy as np

from folio_trainer.config.schema import CostModelConfig, TeacherObjectiveConfig
from folio_trainer.labels.cost_model import estimate_cost


def score_candidates(
    candidates: np.ndarray,
    forward_returns: np.ndarray,
    rf_returns: np.ndarray,
    prev_live_weights: np.ndarray,
    adv20: np.ndarray,
    portfolio_value: float,
    objective_config: TeacherObjectiveConfig,
    cost_config: CostModelConfig,
) -> dict[str, np.ndarray]:
    """Score candidate portfolios using realized forward returns.

    Parameters
    ----------
    candidates
        (n_candidates, n_assets) weight matrix.
    forward_returns
        (horizon, n_assets) daily asset returns over the forward horizon.
    rf_returns
        (horizon,) daily risk-free returns over the horizon.
    prev_live_weights
        (n_assets,) current portfolio weights.
    adv20
        (n_assets,) 20-day average dollar volume.
    portfolio_value
        Portfolio notional value (for cost estimation).
    objective_config
        Teacher objective parameters.
    cost_config
        Cost model parameters.

    Returns
    -------
    dict with keys:
        - objective_total: (n_candidates,)
        - objective_sharpe: (n_candidates,)
        - objective_return: (n_candidates,)
        - objective_vol: (n_candidates,)
        - turnover: (n_candidates,)
        - est_cost: (n_candidates,)
        - concentration_hhi: (n_candidates,)
    """
    n_candidates = candidates.shape[0]
    horizon = forward_returns.shape[0]

    # Portfolio daily returns: (n_candidates, horizon)
    port_returns = candidates @ forward_returns.T  # (n_cand, horizon)

    # Excess returns over risk-free
    excess_returns = port_returns - rf_returns[np.newaxis, :]  # (n_cand, horizon)

    # Sharpe ratio over horizon
    mean_excess = np.mean(excess_returns, axis=1)
    std_returns = np.std(port_returns, axis=1)
    sharpe = mean_excess / (std_returns + objective_config.epsilon)

    # Annualized return
    ann_return = np.mean(port_returns, axis=1) * 252

    # Annualized vol
    ann_vol = std_returns * np.sqrt(252)

    # Turnover
    turnover = 0.5 * np.sum(np.abs(candidates - prev_live_weights[np.newaxis, :]), axis=1)

    # Cost
    est_costs = np.zeros(n_candidates)
    for i in range(n_candidates):
        cost_bps, _ = estimate_cost(candidates[i], prev_live_weights, adv20, portfolio_value, cost_config)
        est_costs[i] = cost_bps

    # Concentration (HHI)
    hhi = np.sum(candidates ** 2, axis=1)

    # Total objective
    objective_total = (
        sharpe
        - objective_config.lambda_turnover * turnover
        - objective_config.lambda_cost * est_costs / 10000  # convert bps to fraction
        - objective_config.lambda_concentration * hhi
    )

    return {
        "objective_total": objective_total,
        "objective_sharpe": sharpe,
        "objective_return": ann_return,
        "objective_vol": ann_vol,
        "turnover": turnover,
        "est_cost": est_costs,
        "concentration_hhi": hhi,
    }
