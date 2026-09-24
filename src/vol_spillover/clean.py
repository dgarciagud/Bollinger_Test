import os
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vol_spillover.clean")


def get_trading_date(timestamps: pd.Series, boundary_time: str = "22:00") -> pd.Series:
    boundary_hour = int(boundary_time.split(":")[0])
    shifted = timestamps - pd.Timedelta(hours=boundary_hour)
    return shifted.dt.date


def filter_spreads(df: pd.DataFrame, k: float = 10.0) -> pd.DataFrame:
    if df.empty or 'bid' not in df or 'ask' not in df:
        return df

    valid = df[df['ask'] > df['bid']].copy()
    if valid.empty:
        return valid

    spread = valid['ask'] - valid['bid']
    rolling_med = spread.rolling(window=20*1440, min_periods=5).median()
    rolling_med = rolling_med.fillna(spread.median())

    clean_mask = spread <= (k * rolling_med)
    return valid[clean_mask]


def detect_and_neutralize_rolls(df: pd.DataFrame, inst_id: str) -> tuple[pd.DataFrame, list]:
    detected_rolls = []
    if df.empty or len(df) < 10:
        return df, detected_rolls

    futures_keywords = ['light', 'brent', 'cmd', 'bond', 'bund', 'ust']
    is_futures = any(kw in inst_id.lower() for kw in futures_keywords)
    if not is_futures:
        return df, detected_rolls

    df = df.copy()
    log_ret = np.log(df['mid'] / df['mid'].shift(1))

    dates = df.index
    month_boundary_mask = (dates.day >= 20) | (dates.day <= 5)

    med_ret = log_ret.median()
    mad = (log_ret - med_ret).abs().median()
    scale = max(1.4826 * mad, 1e-4)
    z_score = (log_ret - med_ret).abs() / scale

    roll_mask = (z_score > 15) & month_boundary_mask
    roll_timestamps = df.index[roll_mask]

    for ts in roll_timestamps:
        detected_rolls.append({
            "instrument": inst_id,
            "timestamp": str(ts),
            "prev_mid": float(df.loc[ts, 'mid']),
            "z_score": float(z_score.loc[ts])
        })
        idx_loc = df.index.get_loc(ts)
        if isinstance(idx_loc, int) and idx_loc > 0:
            df.iloc[idx_loc, df.columns.get_loc('mid')] = df.iloc[idx_loc - 1, df.columns.get_loc('mid')]

    return df, detected_rolls


def filter_holidays_and_weekends(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df = df[df.index.dayofweek < 5]

    exclude_dates = config.get("cleaning", {}).get("exclude_dates_mmdd", ["12-24", "12-25", "12-26", "12-31", "01-01"])
    mmdd_str = df.index.strftime("%m-%d")
    holiday_mask = ~mmdd_str.isin(exclude_dates)
    df = df[holiday_mask]

    min_cov = config.get("cleaning", {}).get("min_coverage_pct", 50.0)
    boundary_time = config.get("cleaning", {}).get("day_boundary_utc", "22:00")

    df['trading_date'] = get_trading_date(df.index.to_series(), boundary_time)
    daily_counts = df.groupby('trading_date')['mid'].transform('count')

    min_count = 1440 * (min_cov / 100.0)
    df = df[daily_counts >= min_count].drop(columns=['trading_date'])

    return df


def clean_single_instrument(parquet_path: str, inst_id: str, config: dict) -> tuple[pd.DataFrame, list]:
    if not os.path.exists(parquet_path):
        logger.warning(f"File not found: {parquet_path}")
        return pd.DataFrame(), []

    df = pd.read_parquet(parquet_path)
    if df.empty:
        return pd.DataFrame(), []

    k = config.get("cleaning", {}).get("spread_multiplier_k", 10.0)
    df = filter_spreads(df, k=k)

    df, rolls = detect_and_neutralize_rolls(df, inst_id)

    df = filter_holidays_and_weekends(df, config)

    return df, rolls


def run_clean(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config.get("paths", {}).get("raw_dir", "data/raw")
    clean_dir = config.get("paths", {}).get("clean_dir", "data/clean")
    reports_dir = config.get("paths", {}).get("reports_dir", "reports")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    instruments = config.get("instruments", [])
    all_rolls = []
    cleaned_dict = {}

    for inst in instruments:
        inst_id = inst["id"]
        parquet_in = os.path.join(raw_dir, f"{inst_id}_m1.parquet")
        logger.info(f"Cleaning {inst_id}...")

        df_clean, rolls = clean_single_instrument(parquet_in, inst_id, config)
        if not df_clean.empty:
            out_file = os.path.join(clean_dir, f"{inst_id}_clean.parquet")
            df_clean.to_parquet(out_file)
            cleaned_dict[inst_id] = df_clean['mid']
            logger.info(f"Cleaned {inst_id}: {len(df_clean)} rows saved to {out_file}")
            if rolls:
                all_rolls.extend(rolls)

    df_rolls = pd.DataFrame(all_rolls)
    roll_csv = os.path.join(reports_dir, "roll_dates.csv")
    df_rolls.to_csv(roll_csv, index=False)
    logger.info(f"Roll dates report saved to {roll_csv}")

    if cleaned_dict:
        panel_df = pd.DataFrame(cleaned_dict)
        panel_file = os.path.join(clean_dir, "m1_panel.parquet")
        panel_df.to_parquet(panel_file)
        logger.info(f"Aligned M1 panel saved to {panel_file} ({len(panel_df)} rows, {len(panel_df.columns)} assets)")

    return cleaned_dict


if __name__ == "__main__":
    run_clean()
