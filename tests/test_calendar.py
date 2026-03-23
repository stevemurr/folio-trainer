"""Tests for calendar and universe modules."""

from __future__ import annotations

import datetime as dt

from folio_trainer.data.calendars import (
    build_schedule,
    cutoff_ts,
    get_trading_days,
    next_trading_day,
    previous_trading_day,
)
from folio_trainer.data.universe import build_universe, get_active_tickers, get_weight_caps
from folio_trainer.config.schema import UniverseConfig


class TestCalendar:
    def test_trading_days_excludes_weekends(self):
        # 2024-01-06 is a Saturday, 2024-01-07 is a Sunday
        days = get_trading_days(dt.date(2024, 1, 5), dt.date(2024, 1, 8))
        dates_set = set(days)
        assert dt.date(2024, 1, 6) not in dates_set  # Saturday
        assert dt.date(2024, 1, 7) not in dates_set  # Sunday
        assert dt.date(2024, 1, 5) in dates_set  # Friday
        assert dt.date(2024, 1, 8) in dates_set  # Monday

    def test_trading_days_excludes_holidays(self):
        # 2024-01-01 is New Year's Day (NYSE closed)
        days = get_trading_days(dt.date(2023, 12, 29), dt.date(2024, 1, 3))
        dates_set = set(days)
        assert dt.date(2024, 1, 1) not in dates_set

    def test_cutoff_ts_timezone(self):
        ts = cutoff_ts(dt.date(2024, 3, 15))
        assert ts.hour == 18
        assert ts.minute == 0
        assert str(ts.tzinfo) == "America/New_York"

    def test_next_trading_day_over_weekend(self):
        # Friday 2024-01-05 -> Monday 2024-01-08
        nxt = next_trading_day(dt.date(2024, 1, 5))
        assert nxt == dt.date(2024, 1, 8)

    def test_previous_trading_day(self):
        # Monday 2024-01-08 -> Friday 2024-01-05
        prev = previous_trading_day(dt.date(2024, 1, 8))
        assert prev == dt.date(2024, 1, 5)

    def test_build_schedule(self):
        sched = build_schedule(dt.date(2024, 1, 2), dt.date(2024, 1, 5))
        assert len(sched) == 4  # Tue-Fri
        assert set(sched.columns) == {"trade_date", "cutoff_ts", "prediction_date"}
        # Each prediction_date should be after trade_date
        for row in sched.iter_rows(named=True):
            assert row["prediction_date"] > row["trade_date"]


class TestUniverse:
    def test_build_universe_with_cash(self):
        cfg = UniverseConfig(tickers=["AAPL", "GOOG"], include_cash=True)
        uni = build_universe(cfg)
        assert len(uni) == 3  # AAPL, GOOG, CASH
        tickers = uni["ticker"].to_list()
        assert "CASH" in tickers
        cash_row = uni.filter(uni["is_cash"])
        assert len(cash_row) == 1
        assert cash_row["weight_cap"][0] == 1.0

    def test_build_universe_without_cash(self):
        cfg = UniverseConfig(tickers=["AAPL", "GOOG"], include_cash=False)
        uni = build_universe(cfg)
        assert len(uni) == 2
        assert "CASH" not in uni["ticker"].to_list()

    def test_weight_caps(self):
        cfg = UniverseConfig(tickers=["AAPL"], include_cash=True, max_single_name_weight=0.30)
        uni = build_universe(cfg)
        caps = get_weight_caps(uni)
        assert caps["AAPL"] == 0.30
        assert caps["CASH"] == 1.0

    def test_active_tickers(self):
        cfg = UniverseConfig(tickers=["AAPL", "GOOG"], include_cash=True)
        uni = build_universe(cfg, start_date=dt.date(2020, 1, 1))
        active = get_active_tickers(uni, dt.date(2023, 6, 15))
        assert len(active) == 3
        active_no_cash = get_active_tickers(uni, dt.date(2023, 6, 15), include_cash=False)
        assert len(active_no_cash) == 2
        assert "CASH" not in active_no_cash
