import pytest
import pandas as pd
import numpy as np
from vol_spillover.realized import compute_subsampled_rv, compute_bipower_variation, compute_abnormal_flags, evaluate_stationarity


def test_subsampled_rv_brownian_motion():
    np.random.seed(42)
    true_daily_var = 0.0001
    dt_std = np.sqrt(true_daily_var / 1440.0)

    rvs = []
    for _ in range(200):
        returns = np.random.normal(0, dt_std, 1440)
        price_path = 100.0 * np.exp(np.cumsum(returns))
        rv = compute_subsampled_rv(pd.Series(price_path))
        rvs.append(rv)

    mean_rv = np.mean(rvs)
    bias_pct = abs(mean_rv - true_daily_var) / true_daily_var * 100.0
    assert bias_pct < 2.0, f"Bias {bias_pct:.2f}% exceeds 2%"


def test_compute_bipower_variation():
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, 1440)
    price_path = 100.0 * np.exp(np.cumsum(returns))
    bv = compute_bipower_variation(pd.Series(price_path))
    assert not np.isnan(bv)
    assert bv > 0


def test_evaluate_stationarity():
    np.random.seed(42)
    stationary_series = pd.Series(np.random.normal(0, 1, 100))
    res = evaluate_stationarity(stationary_series)
    assert "adf_pvalue" in res
    assert res["stationary"] == True


def test_no_lookahead_abnormal_flags():
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    data = np.ones(300)
    data[299] = 10.0

    log_rv_df = pd.DataFrame({"asset_A": data}, index=dates)
    flags = compute_abnormal_flags(log_rv_df, percentile=95, window=50)

    assert flags.iloc[299]["asset_A"] == True
    assert flags.iloc[298]["asset_A"] == False
