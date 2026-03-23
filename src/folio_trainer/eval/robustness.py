"""Robustness and stress tests (spec 2.7.3)."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np

from folio_trainer.backtest.metrics import compute_metrics, metrics_to_dict
from folio_trainer.backtest.simulator import BacktestResult, simulate
from folio_trainer.config.schema import CostModelConfig, ExecutionConfig, PipelineConfig

logger = logging.getLogger(__name__)


def run_robustness_tests(
    weight_signals: np.ndarray,
    asset_returns: np.ndarray,
    dates: list,
    config: PipelineConfig,
    vix_values: np.ndarray | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Run robustness and stress tests.

    Parameters
    ----------
    weight_signals
        (n_days, n_assets) model weight predictions.
    asset_returns
        (n_days, n_assets) test period returns.
    dates
        List of test dates.
    config
        Pipeline config.
    vix_values
        (n_days,) VIX values for regime analysis. Optional.
    output_dir
        Output directory.

    Returns
    -------
    dict with all robustness test results.
    """
    results = {}

    # 1. Transaction cost stress tests
    results["cost_stress"] = _cost_stress_test(
        weight_signals, asset_returns, dates, config
    )

    # 2. VIX regime split
    if vix_values is not None:
        results["vix_regime"] = _vix_regime_test(
            weight_signals, asset_returns, dates, config, vix_values
        )

    # 3. First half vs second half
    results["half_split"] = _half_split_test(
        weight_signals, asset_returns, dates, config
    )

    # 4. Sensitivity to rebalance band and partial alpha
    results["sensitivity"] = _sensitivity_test(
        weight_signals, asset_returns, dates, config
    )

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "robustness_results.json").write_text(json.dumps(results, indent=2, default=str))

    return results


def _cost_stress_test(
    weights: np.ndarray, returns: np.ndarray, dates: list, config: PipelineConfig
) -> dict:
    """Test with 0.5x, 1.0x, 2.0x transaction costs."""
    results = {}
    for multiplier in [0.5, 1.0, 2.0]:
        cost_cfg = config.cost_model.model_copy()
        cost_cfg.commission_bps *= multiplier
        cost_cfg.spread_bps_proxy_default *= multiplier
        cost_cfg.impact_coeff *= multiplier
        for bucket in cost_cfg.liquidity_buckets:
            bucket.spread_bps_proxy *= multiplier

        bt = simulate(weights, returns, dates, config.execution, cost_cfg)
        m = compute_metrics(bt, use_net=True)
        results[f"{multiplier}x"] = metrics_to_dict(m)

    return results


def _vix_regime_test(
    weights: np.ndarray, returns: np.ndarray, dates: list,
    config: PipelineConfig, vix: np.ndarray,
) -> dict:
    """Split test period by VIX regime."""
    median_vix = float(np.nanmedian(vix))
    high_mask = vix >= median_vix
    low_mask = ~high_mask

    results = {"median_vix": median_vix}
    for name, mask in [("high_vix", high_mask), ("low_vix", low_mask)]:
        idx = np.where(mask)[0]
        if len(idx) < 5:
            results[name] = None
            continue
        bt = simulate(
            weights[idx], returns[idx],
            [dates[i] for i in idx],
            config.execution, config.cost_model,
        )
        m = compute_metrics(bt, use_net=True)
        results[name] = metrics_to_dict(m)

    return results


def _half_split_test(
    weights: np.ndarray, returns: np.ndarray, dates: list, config: PipelineConfig
) -> dict:
    """First half vs second half of test period."""
    mid = len(dates) // 2
    results = {}
    for name, sl in [("first_half", slice(0, mid)), ("second_half", slice(mid, None))]:
        bt = simulate(
            weights[sl], returns[sl], dates[sl],
            config.execution, config.cost_model,
        )
        m = compute_metrics(bt, use_net=True)
        results[name] = metrics_to_dict(m)
    return results


def _sensitivity_test(
    weights: np.ndarray, returns: np.ndarray, dates: list, config: PipelineConfig
) -> dict:
    """Sensitivity to rebalance band and partial trade alpha."""
    results = {}

    for band in [0.01, 0.02, 0.05, 0.10]:
        exec_cfg = config.execution.model_copy()
        exec_cfg.rebalance_band = band
        bt = simulate(weights, returns, dates, exec_cfg, config.cost_model)
        m = compute_metrics(bt, use_net=True)
        results[f"band_{band}"] = {"sharpe": m.sharpe, "turnover": m.avg_turnover}

    for alpha in [0.25, 0.50, 0.75, 1.0]:
        exec_cfg = config.execution.model_copy()
        exec_cfg.partial_rebalance_alpha = alpha
        bt = simulate(weights, returns, dates, exec_cfg, config.cost_model)
        m = compute_metrics(bt, use_net=True)
        results[f"alpha_{alpha}"] = {"sharpe": m.sharpe, "turnover": m.avg_turnover}

    return results
