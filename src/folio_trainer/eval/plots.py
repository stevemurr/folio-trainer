"""Evaluation plots (spec 2.8)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from folio_trainer.backtest.simulator import BacktestResult

logger = logging.getLogger(__name__)


def plot_cumulative_returns(
    results: dict[str, BacktestResult],
    output_path: str | Path,
    title: str = "Cumulative Returns",
    use_net: bool = True,
) -> None:
    """Plot cumulative return curves for multiple strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, result in results.items():
        returns = result.daily_returns_net if use_net else result.daily_returns_gross
        cum = np.cumprod(1 + returns)
        ax.plot(result.dates, cum, label=name, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_weight_allocation(
    result: BacktestResult,
    output_path: str | Path,
    title: str = "Portfolio Weight Allocation",
) -> None:
    """Stacked area chart of portfolio weights over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        result.dates,
        result.daily_weights.T,
        labels=result.ticker_names or [f"Asset {i}" for i in range(result.daily_weights.shape[1])],
        alpha=0.8,
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_drawdown(
    result: BacktestResult,
    output_path: str | Path,
    title: str = "Drawdown",
    use_net: bool = True,
) -> None:
    """Plot drawdown over time."""
    returns = result.daily_returns_net if use_net else result.daily_returns_gross
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum / running_max - 1

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(result.dates, drawdown, 0, alpha=0.4, color="red")
    ax.plot(result.dates, drawdown, color="red", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_turnover(
    result: BacktestResult,
    output_path: str | Path,
    title: str = "Daily Turnover",
) -> None:
    """Plot daily turnover."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(result.dates, result.daily_turnover, alpha=0.6, width=1.0)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_rolling_sharpe(
    result: BacktestResult,
    output_path: str | Path,
    window: int = 60,
    title: str = "Rolling Sharpe Ratio",
    use_net: bool = True,
) -> None:
    """Plot rolling Sharpe ratio."""
    returns = result.daily_returns_net if use_net else result.daily_returns_gross

    if len(returns) < window:
        logger.warning("Not enough data for rolling Sharpe with window=%d", window)
        return

    rolling_mean = np.convolve(returns, np.ones(window) / window, mode="valid")
    rolling_std = np.array([
        np.std(returns[i : i + window]) for i in range(len(returns) - window + 1)
    ])
    rolling_sharpe = rolling_mean / np.maximum(rolling_std, 1e-10) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result.dates[window - 1 :], rolling_sharpe, linewidth=1.0)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"{title} ({window}-day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_feature_importance(
    importance: dict[str, float],
    output_path: str | Path,
    top_n: int = 20,
    title: str = "Feature Importance",
) -> None:
    """Bar chart of top feature importances."""
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:top_n]
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(names)), values, align="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def generate_all_plots(
    model_result: BacktestResult,
    baseline_results: dict[str, BacktestResult],
    feature_importance: dict[str, float],
    output_dir: str | Path,
) -> None:
    """Generate all standard evaluation plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Cumulative returns comparison
    all_results = {"Model": model_result, **baseline_results}
    plot_cumulative_returns(all_results, out / "cumulative_returns.png")

    # Model-specific plots
    plot_weight_allocation(model_result, out / "weight_allocation.png")
    plot_drawdown(model_result, out / "drawdown.png")
    plot_turnover(model_result, out / "turnover.png")
    plot_rolling_sharpe(model_result, out / "rolling_sharpe.png")

    # Feature importance
    if feature_importance:
        plot_feature_importance(feature_importance, out / "feature_importance.png")

    logger.info("All plots saved to %s", out)
