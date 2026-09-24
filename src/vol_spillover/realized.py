import os
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, kpss
from vol_spillover.clean import get_trading_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vol_spillover.realized")


def compute_subsampled_rv(m1_prices: pd.Series) -> float:
    if len(m1_prices) < 10:
        return np.nan

    log_prices = np.log(m1_prices.values)
    rv_sum = 0.0

    for offset in range(5):
        sub_grid = log_prices[offset::5]
        if len(sub_grid) > 1:
            returns = np.diff(sub_grid)
            rv_sum += np.sum(returns ** 2)

    return rv_sum / 5.0


def compute_bipower_variation(m1_prices: pd.Series) -> float:
    if len(m1_prices) < 10:
        return np.nan

    log_prices = np.log(m1_prices[::5].values)
    if len(log_prices) < 3:
        return np.nan

    returns = np.diff(log_prices)
    abs_ret = np.abs(returns)
    N = len(returns)
    if N < 2:
        return np.nan

    bv = (np.pi / 2.0) * (N / (N - 1)) * np.sum(abs_ret[1:] * abs_ret[:-1])
    return float(bv)


def evaluate_stationarity(series: pd.Series) -> dict:
    clean_s = series.dropna()
    if len(clean_s) < 20:
        return {"adf_stat": np.nan, "adf_pvalue": np.nan, "kpss_stat": np.nan, "kpss_pvalue": np.nan, "stationary": False}

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_res = adfuller(clean_s, autolag='AIC')
            adf_p = float(adf_res[1])
    except Exception:
        adf_p = np.nan

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_res = kpss(clean_s, regression='c', nlags='auto')
            kpss_p = float(kpss_res[1])
    except Exception:
        kpss_p = np.nan

    is_stat = (adf_p < 0.05) if not np.isnan(adf_p) else False
    return {
        "adf_pvalue": round(adf_p, 4) if not np.isnan(adf_p) else np.nan,
        "kpss_pvalue": round(kpss_p, 4) if not np.isnan(kpss_p) else np.nan,
        "stationary": is_stat
    }


def compute_abnormal_flags(log_rv_df: pd.DataFrame, percentile: float = 95.0, window: int = 250) -> pd.DataFrame:
    abnormal_flags = pd.DataFrame(index=log_rv_df.index, columns=log_rv_df.columns, dtype=bool)
    min_p = min(window, 30)

    for col in log_rv_df.columns:
        s = log_rv_df[col]
        rolling_p95 = s.shift(1).rolling(window=window, min_periods=min_p).quantile(percentile / 100.0)
        abnormal_flags[col] = (s > rolling_p95).fillna(False)

    return abnormal_flags


def run_realized(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    clean_dir = config.get("paths", {}).get("clean_dir", "data/clean")
    features_dir = config.get("paths", {}).get("features_dir", "data/features")
    reports_dir = config.get("paths", {}).get("reports_dir", "reports")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    panel_file = os.path.join(clean_dir, "m1_panel.parquet")
    if not os.path.exists(panel_file):
        logger.error(f"Panel file not found: {panel_file}. Run clean first.")
        return

    panel_m1 = pd.read_parquet(panel_file)
    boundary_time = config.get("cleaning", {}).get("day_boundary_utc", "22:00")

    panel_m1['trading_date'] = get_trading_date(panel_m1.index.to_series(), boundary_time)

    rv_dict = {}
    bv_dict = {}
    jump_dict = {}
    log_rv_dict = {}

    assets = [c for c in panel_m1.columns if c != 'trading_date']

    for asset in assets:
        logger.info(f"Computing Realized Volatility for {asset}...")
        asset_series = panel_m1[[asset, 'trading_date']].dropna()

        daily_rv = {}
        daily_bv = {}
        daily_jumps = {}
        daily_log_rv = {}

        for t_date, group in asset_series.groupby('trading_date'):
            prices = group[asset]
            rv = compute_subsampled_rv(prices)
            bv = compute_bipower_variation(prices)

            if not np.isnan(rv) and rv > 0:
                jump = max(rv - (bv if not np.isnan(bv) else rv), 0.0)
                annualized_log_rv = float(np.log(rv * 252.0))

                daily_rv[t_date] = rv
                daily_bv[t_date] = bv
                daily_jumps[t_date] = jump
                daily_log_rv[t_date] = annualized_log_rv

        rv_dict[asset] = pd.Series(daily_rv)
        bv_dict[asset] = pd.Series(daily_bv)
        jump_dict[asset] = pd.Series(daily_jumps)
        log_rv_dict[asset] = pd.Series(daily_log_rv)

    rv_df = pd.DataFrame(rv_dict)
    bv_df = pd.DataFrame(bv_dict)
    jump_df = pd.DataFrame(jump_dict)
    log_rv_df = pd.DataFrame(log_rv_dict)

    rv_df.index = pd.to_datetime(rv_df.index)
    bv_df.index = pd.to_datetime(bv_df.index)
    jump_df.index = pd.to_datetime(jump_df.index)
    log_rv_df.index = pd.to_datetime(log_rv_df.index)

    rv_df = rv_df.sort_index()
    bv_df = bv_df.sort_index()
    jump_df = jump_df.sort_index()
    log_rv_df = log_rv_df.sort_index()

    stat_records = []
    for col in log_rv_df.columns:
        res = evaluate_stationarity(log_rv_df[col])
        res["asset"] = col
        stat_records.append(res)

    df_stat = pd.DataFrame(stat_records)
    df_stat.to_csv(os.path.join(reports_dir, "stationarity_tests.csv"), index=False)
    logger.info(f"Stationarity test results saved to {os.path.join(reports_dir, 'stationarity_tests.csv')}")

    p95 = config.get("realized", {}).get("abnormal_percentile", 95)
    win = config.get("realized", {}).get("abnormal_rolling_window", 250)
    abnormal_df = compute_abnormal_flags(log_rv_df, percentile=p95, window=win)

    log_rv_df.to_parquet(os.path.join(features_dir, "log_rv.parquet"))
    bv_df.to_parquet(os.path.join(features_dir, "bv.parquet"))
    jump_df.to_parquet(os.path.join(features_dir, "jumps.parquet"))
    abnormal_df.to_parquet(os.path.join(features_dir, "abnormal_flags.parquet"))

    logger.info(f"Realized Volatility features saved to {features_dir} ({len(log_rv_df)} daily observations)")
    return log_rv_df, abnormal_df


if __name__ == "__main__":
    run_realized()
