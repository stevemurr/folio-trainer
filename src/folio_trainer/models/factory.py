"""Factory helpers for direct-weight model variants."""

from __future__ import annotations

import pickle
from pathlib import Path

from folio_trainer.config.schema import DirectWeightModelConfig
from folio_trainer.models.direct_weight_gbm import DirectWeightGBM
from folio_trainer.models.direct_weight_linear import DirectWeightLinear


def create_direct_weight_model(config: DirectWeightModelConfig):
    """Instantiate a direct-weight model from config."""
    if config.kind in {"gbm", "rank_gbm"}:
        return DirectWeightGBM(config)
    if config.kind == "linear":
        return DirectWeightLinear(config)
    msg = f"Unsupported direct weight model kind '{config.kind}'."
    raise ValueError(msg)


def load_direct_weight_model(path: str | Path):
    """Load a persisted direct-weight model regardless of concrete type."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    config = payload["config"]
    if config.kind in {"gbm", "rank_gbm"}:
        return DirectWeightGBM.load(path)
    if config.kind == "linear":
        return DirectWeightLinear.load(path)
    msg = f"Unsupported persisted direct weight model kind '{config.kind}'."
    raise ValueError(msg)
