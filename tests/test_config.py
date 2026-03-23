"""Tests for config schema and loader."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from folio_trainer.config.loader import load_config
from folio_trainer.config.schema import PipelineConfig, SplitsConfig, UniverseConfig


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
    assert cfg.candidate_search.top_k == 20
