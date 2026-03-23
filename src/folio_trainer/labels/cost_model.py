"""Trading cost estimation model (spec 1.8.5)."""

from __future__ import annotations

import numpy as np

from folio_trainer.config.schema import CostModelConfig


def estimate_cost(
    target_weights: np.ndarray,
    prev_weights: np.ndarray,
    adv20: np.ndarray,
    portfolio_value: float,
    config: CostModelConfig,
) -> tuple[float, np.ndarray]:
    """Estimate total trading cost in bps for a rebalance.

    Parameters
    ----------
    target_weights
        Target portfolio weights (n_assets,).
    prev_weights
        Current portfolio weights (n_assets,).
    adv20
        20-day average dollar volume per asset (n_assets,).
    portfolio_value
        Total portfolio notional value.
    config
        Cost model configuration.

    Returns
    -------
    total_cost_bps
        Weighted average cost in basis points.
    per_asset_cost_bps
        Cost per asset in basis points (n_assets,).
    """
    trade_delta = np.abs(target_weights - prev_weights)
    trade_notional = trade_delta * portfolio_value

    # Look up spread proxy per asset from liquidity buckets
    spread_bps = np.array([config.get_spread_bps(float(a)) for a in adv20])

    # Clamp ADV to floor
    adv_clamped = np.maximum(adv20, config.adv_floor)

    # Market impact
    impact_bps = config.impact_coeff * np.sqrt(
        np.divide(trade_notional, adv_clamped, where=adv_clamped > 0, out=np.zeros_like(trade_notional))
    )

    # Total per-asset cost
    per_asset_cost_bps = (
        config.commission_bps + config.regulatory_bps + spread_bps + impact_bps
    )

    # Weight by trade delta to get portfolio-level cost
    total_cost_bps = float(np.sum(per_asset_cost_bps * trade_delta))

    return total_cost_bps, per_asset_cost_bps
