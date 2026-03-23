"""Load and validate pipeline configuration from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from folio_trainer.config.schema import (
    BUILTIN_STRATEGY_PROFILES,
    PipelineConfig,
    StrategyProfileConfig,
)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "docs" / "allocation_model_default_config.yaml"


def resolve_strategy_profile(
    name: str,
    custom_profiles: dict | None = None,
) -> dict:
    """Look up a strategy profile by name and return a nested config dict.

    The returned dict mirrors the PipelineConfig structure so it can be
    deep-merged between defaults and user overrides.
    """
    profile: StrategyProfileConfig | None = None

    # Check custom profiles first, then builtins
    if custom_profiles:
        raw = custom_profiles.get(name)
        if raw is not None:
            profile = (
                raw
                if isinstance(raw, StrategyProfileConfig)
                else StrategyProfileConfig.model_validate(raw)
            )

    if profile is None:
        profile = BUILTIN_STRATEGY_PROFILES.get(name)

    if profile is None:
        available = sorted(
            set(list(BUILTIN_STRATEGY_PROFILES.keys()) + list((custom_profiles or {}).keys()))
        )
        msg = f"Unknown strategy '{name}'. Available: {available}"
        raise ValueError(msg)

    # Map profile fields to their nested config paths
    result: dict = {}
    if profile.lambda_turnover is not None:
        result.setdefault("teacher_objective", {})["lambda_turnover"] = profile.lambda_turnover
    if profile.lambda_cost is not None:
        result.setdefault("teacher_objective", {})["lambda_cost"] = profile.lambda_cost
    if profile.lambda_concentration is not None:
        result.setdefault("teacher_objective", {})["lambda_concentration"] = profile.lambda_concentration
    if profile.distillation_temperature is not None:
        result.setdefault("candidate_search", {})["distillation_temperature"] = profile.distillation_temperature
    if profile.inference_temperature is not None:
        result.setdefault("model", {}).setdefault("direct_weight_model", {})[
            "temperature"
        ] = profile.inference_temperature

    return result


def load_config(
    path: str | Path | None = None,
    overrides: dict | None = None,
    strategy: str | None = None,
) -> PipelineConfig:
    """Load config from YAML, merge with defaults, validate via Pydantic.

    Parameters
    ----------
    path
        Path to a YAML config file. If None, uses built-in defaults.
    overrides
        Dict of overrides merged on top of the loaded YAML.
    strategy
        Named strategy profile. Overrides teacher lambdas and temperatures.
        CLI flag takes precedence over YAML ``strategy`` field.

    Returns
    -------
    PipelineConfig
        Validated configuration object.
    """
    # Start with defaults
    defaults = {}
    if _DEFAULT_CONFIG_PATH.exists():
        defaults = yaml.safe_load(_DEFAULT_CONFIG_PATH.read_text()) or {}

    # Layer user config on top
    user_cfg = {}
    if path is not None:
        raw = Path(path).read_text()
        user_cfg = yaml.safe_load(raw) or {}

    # Resolve strategy: CLI flag > overrides > user YAML > None
    effective_strategy = (
        strategy
        or (overrides or {}).get("strategy")
        or user_cfg.get("strategy")
    )

    if effective_strategy:
        # Collect custom profiles from all sources
        custom_profiles = {
            **user_cfg.get("custom_profiles", {}),
            **(overrides or {}).get("custom_profiles", {}),
        }
        profile_overrides = resolve_strategy_profile(effective_strategy, custom_profiles)
        # Merge order: defaults → profile → user YAML → CLI overrides
        merged = _deep_merge(defaults, profile_overrides)
        merged = _deep_merge(merged, user_cfg)
    else:
        merged = _deep_merge(defaults, user_cfg)

    if overrides:
        merged = _deep_merge(merged, overrides)

    # Ensure the resolved strategy name is in the final config
    if effective_strategy:
        merged["strategy"] = effective_strategy

    return PipelineConfig.model_validate(merged)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base, returning a new dict."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
