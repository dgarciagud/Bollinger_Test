import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from vol_spillover.clean import filter_spreads, detect_and_neutralize_rolls, filter_holidays_and_weekends, get_trading_date


def test_filter_spreads():
    dates = pd.date_range("2024-01-01 00:00:00", periods=10, freq="1min", tz="UTC")
    bids = np.ones(10) * 100.0
    asks = np.ones(10) * 100.1

    # Inverted spread at index 2
    bids[2] = 100.2
    asks[2] = 100.0

    # Extreme spread at index 5
    asks[5] = 150.0

    df = pd.DataFrame({"bid": bids, "ask": asks, "mid": (bids + asks) / 2.0}, index=dates)
    filtered = filter_spreads(df, k=5.0)

    assert dates[2] not in filtered.index
    assert dates[5] not in filtered.index


def test_detect_and_neutralize_rolls():
    dates = pd.date_range("2024-01-25 00:00:00", periods=50, freq="1min", tz="UTC")
    mid = np.ones(50) * 75.0
    mid[25:] = 85.0

    df = pd.DataFrame({"mid": mid}, index=dates)
    df_clean, rolls = detect_and_neutralize_rolls(df, "lightcmdusd")

    assert len(rolls) == 1
    assert rolls[0]["instrument"] == "lightcmdusd"
    assert df_clean.iloc[25]["mid"] == df_clean.iloc[24]["mid"]


def test_filter_holidays_and_weekends():
    dates = pd.date_range("2024-12-24 00:00:00", periods=5*1440, freq="1min", tz="UTC")
    df = pd.DataFrame({"mid": np.ones(len(dates))}, index=dates)

    config = {
        "cleaning": {
            "exclude_dates_mmdd": ["12-24", "12-25"],
            "min_coverage_pct": 50.0,
            "day_boundary_utc": "22:00"
        }
    }

    df_filtered = filter_holidays_and_weekends(df, config)
    mmdd = df_filtered.index.strftime("%m-%d").unique()
    assert "12-24" not in mmdd
    assert "12-25" not in mmdd
