"""Load and validate pipeline configuration from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from folio_trainer.config.schema import PipelineConfig

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "docs" / "allocation_model_default_config.yaml"


def load_config(
    path: str | Path | None = None,
    overrides: dict | None = None,
) -> PipelineConfig:
    """Load config from YAML, merge with defaults, validate via Pydantic.

    Parameters
    ----------
    path
        Path to a YAML config file. If None, uses built-in defaults.
    overrides
        Dict of overrides merged on top of the loaded YAML.

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

    merged = _deep_merge(defaults, user_cfg)
    if overrides:
        merged = _deep_merge(merged, overrides)

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
