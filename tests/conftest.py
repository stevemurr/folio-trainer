"""Shared test fixtures."""

from __future__ import annotations

import datetime as dt

import pytest

from folio_trainer.config.schema import PipelineConfig, UniverseConfig


@pytest.fixture
def default_config() -> PipelineConfig:
    """Return a PipelineConfig with test defaults."""
    return PipelineConfig(
        universe=UniverseConfig(
            tickers=["AAPL", "GOOG", "MSFT"],
            include_cash=True,
        ),
    )


@pytest.fixture
def sample_tickers() -> list[str]:
    return ["AAPL", "GOOG", "MSFT"]


@pytest.fixture
def sample_date_range() -> tuple[dt.date, dt.date]:
    return dt.date(2023, 1, 1), dt.date(2023, 12, 31)
