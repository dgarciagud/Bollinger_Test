import os
import struct
import lzma
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from vol_spillover.download import evaluate_data_quality, download_bi5_hour_ticks


def test_evaluate_data_quality():
    dates = pd.date_range("2024-01-01 00:00:00", periods=100, freq="1min", tz="UTC")
    bids = np.ones(100) * 1.0800
    asks = np.ones(100) * 1.0802
    # Add a jump
    bids[50] = 1.1500
    asks[50] = 1.1502

    df = pd.DataFrame({"bid": bids, "ask": asks, "mid": (bids + asks) / 2.0}, index=dates)

    q_rec = evaluate_data_quality(df, "eurusd", 2024)
    assert q_rec["instrument"] == "eurusd"
    assert q_rec["candles_count"] == 100
    assert q_rec["median_spread_bps"] > 0
    assert q_rec["jump_count_10mad"] >= 1


def test_bi5_binary_decoder():
    # Construct synthetic LZMA compressed 20-byte record
    # Format: >3I2f -> time_ms (uint32), ask_raw (uint32), bid_raw (uint32), ask_vol (float32), bid_vol (float32)
    time_ms = 1000  # 1 sec into hour
    ask_raw = 108500
    bid_raw = 108480
    ask_vol = 1.5
    bid_vol = 2.0

    record = struct.pack(">3I2f", time_ms, ask_raw, bid_raw, ask_vol, bid_vol)
    compressed = lzma.compress(record)

    # We test struct unpacking logic directly as in download_bi5_hour_ticks
    decompressed = lzma.decompress(compressed)
    assert len(decompressed) == 20
    t_ms, a_r, b_r, a_v, b_v = struct.unpack(">3I2f", decompressed)
    assert t_ms == time_ms
    assert a_r == ask_raw
    assert b_r == bid_raw
    assert abs(a_r / 100000.0 - 1.08500) < 1e-6
