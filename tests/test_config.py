"""Tests for config schema and loader."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from folio_trainer.config.loader import load_config
from folio_trainer.config.schema import (
    BUILTIN_STRATEGY_PROFILES,
    PipelineConfig,
    SplitsConfig,
    StrategyProfileConfig,
    UniverseConfig,
)


def test_default_config_loads():
    """Loading config with no args should produce valid defaults."""
    cfg = load_config()
    assert isinstance(cfg, PipelineConfig)
    assert cfg.project_name == "allocation_model_v1"
    assert cfg.random_seed == 42


def test_config_universe_defaults():
    cfg = load_config()
    assert cfg.universe.include_cash is True
    assert cfg.universe.cash_ticker == "CASH"
    assert cfg.universe.max_single_name_weight == 0.25
    assert cfg.universe.long_only is True
    assert cfg.universe.leverage_allowed is False


def test_config_horizons():
    cfg = load_config()
    assert cfg.horizons.primary == 20
    assert set(cfg.horizons.all_horizons) == {5, 20, 60}


def test_config_splits_validation():
    """Splits fractions must sum to 1."""
    with pytest.raises(ValidationError):
        SplitsConfig(train_frac=0.5, val_frac=0.1, test_frac=0.1)


def test_config_splits_valid():
    s = SplitsConfig(train_frac=0.7, val_frac=0.15, test_frac=0.15)
    assert abs(s.train_frac + s.val_frac + s.test_frac - 1.0) < 1e-6


def test_config_cost_model_spread_lookup():
    cfg = load_config()
    # Low ADV -> high spread
    assert cfg.cost_model.get_spread_bps(500_000) == 10.0
    # Medium ADV
    assert cfg.cost_model.get_spread_bps(5_000_000) == 5.0
    # High ADV
    assert cfg.cost_model.get_spread_bps(100_000_000) == 2.0


def test_config_overrides():
    cfg = load_config(overrides={"random_seed": 99, "universe": {"tickers": ["SPY"]}})
    assert cfg.random_seed == 99
    assert cfg.universe.tickers == ["SPY"]
    # Other defaults preserved
    assert cfg.universe.include_cash is True


def test_config_candidate_search():
    cfg = load_config()
    assert cfg.candidate_search.dirichlet_candidates_per_day == 5000
    assert len(cfg.candidate_search.dirichlet_alpha_mix) == 4
    assert cfg.candidate_search.top_k == 5
    assert cfg.candidate_search.distillation_temperature == 0.10
    assert cfg.candidate_search.sparse_k_assets == 30


# ---------------------------------------------------------------------------
# Strategy profile tests
# ---------------------------------------------------------------------------


def test_strategy_none_backward_compatible():
    """No strategy set should produce identical defaults."""
    cfg = load_config()
    assert cfg.strategy is None
    assert cfg.teacher_objective.lambda_turnover == 0.20
    assert cfg.teacher_objective.lambda_cost == 1.0
    assert cfg.teacher_objective.lambda_concentration == 0.0
    assert cfg.candidate_search.distillation_temperature == 0.10
    assert cfg.model.direct_weight_model.temperature == 0.30


def test_builtin_strategy_aggressive():
    cfg = load_config(strategy="aggressive")
    assert cfg.strategy == "aggressive"
    assert cfg.teacher_objective.lambda_turnover == 0.05
    assert cfg.teacher_objective.lambda_cost == 0.30
    assert cfg.teacher_objective.lambda_concentration == 0.0
    assert cfg.candidate_search.distillation_temperature == 0.05
    assert cfg.model.direct_weight_model.temperature == 0.20
    assert cfg.execution.rebalance_band == 0.0025
    assert cfg.execution.partial_rebalance_alpha == 0.75


def test_builtin_strategy_conservative():
    cfg = load_config(strategy="conservative")
    assert cfg.strategy == "conservative"
    assert cfg.teacher_objective.lambda_turnover == 0.50
    assert cfg.teacher_objective.lambda_cost == 2.0
    assert cfg.teacher_objective.lambda_concentration == 0.30
    assert cfg.candidate_search.distillation_temperature == 0.15
    assert cfg.model.direct_weight_model.temperature == 0.40
    assert cfg.execution.rebalance_band == 0.02
    assert cfg.execution.partial_rebalance_alpha == 0.25


def test_strategy_neutral_matches_defaults():
    """Neutral profile should produce the same values as no strategy."""
    cfg_none = load_config()
    cfg_neutral = load_config(strategy="neutral")
    assert cfg_neutral.strategy == "neutral"
    assert cfg_neutral.teacher_objective.lambda_turnover == cfg_none.teacher_objective.lambda_turnover
    assert cfg_neutral.teacher_objective.lambda_cost == cfg_none.teacher_objective.lambda_cost
    assert cfg_neutral.teacher_objective.lambda_concentration == cfg_none.teacher_objective.lambda_concentration
    assert cfg_neutral.candidate_search.distillation_temperature == cfg_none.candidate_search.distillation_temperature
    assert cfg_neutral.model.direct_weight_model.temperature == cfg_none.model.direct_weight_model.temperature
    assert cfg_neutral.execution.rebalance_band == cfg_none.execution.rebalance_band
    assert cfg_neutral.execution.partial_rebalance_alpha == cfg_none.execution.partial_rebalance_alpha


def test_explicit_override_wins_over_profile():
    """User-provided overrides should take precedence over strategy profile values."""
    cfg = load_config(
        strategy="aggressive",
        overrides={"teacher_objective": {"lambda_turnover": 0.35}},
    )
    # Explicit override wins
    assert cfg.teacher_objective.lambda_turnover == 0.35
    # Other values come from aggressive profile
    assert cfg.teacher_objective.lambda_cost == 0.30


def test_custom_profile():
    """Custom profiles defined via overrides should work."""
    cfg = load_config(overrides={
        "strategy": "my_strat",
        "custom_profiles": {
            "my_strat": {
                "lambda_turnover": 0.12,
                "lambda_cost": 0.75,
                "inference_temperature": 0.25,
                "rebalance_band": 0.03,
            },
        },
    })
    assert cfg.strategy == "my_strat"
    assert cfg.teacher_objective.lambda_turnover == 0.12
    assert cfg.teacher_objective.lambda_cost == 0.75
    assert cfg.model.direct_weight_model.temperature == 0.25
    assert cfg.execution.rebalance_band == 0.03
    # Unset profile fields keep defaults
    assert cfg.teacher_objective.lambda_concentration == 0.0


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy 'nonexistent'"):
        load_config(strategy="nonexistent")
