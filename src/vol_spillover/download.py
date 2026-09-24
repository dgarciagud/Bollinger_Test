import os
import sys
import time
import glob
import logging
import struct
import lzma
import urllib.request
from datetime import datetime, timedelta, date
import yaml
import pandas as pd
import numpy as np
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vol_spillover.download")


def get_date_range(config: dict):
    history_days = config.get("download", {}).get("history_days", 365)
    end_dt = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=history_days)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def download_dukascopy_node(inst_id: str, date_from: str, date_to: str, price_type: str, out_dir: str, retries: int = 5, retry_pause_ms: int = 500) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "npx", "dukascopy-node",
        "-i", inst_id,
        "-from", date_from,
        "-to", date_to,
        "-t", "m1",
        "-p", price_type,
        "-f", "csv",
        "-dir", out_dir,
        "-r", str(retries),
        "-rp", str(retry_pause_ms),
        "-s"
    ]

    for attempt in range(1, retries + 1):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if res.returncode == 0 and "File saved" in res.stdout:
                pattern = os.path.join(out_dir, f"{inst_id}-m1-{price_type}-*.csv")
                files = glob.glob(pattern)
                if files:
                    latest_file = max(files, key=os.path.getmtime)
                    return latest_file
            logger.warning(f"Attempt {attempt}/{retries} failed for {inst_id} ({price_type}): {res.stderr.strip() or res.stdout.strip()}")
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{retries} error for {inst_id} ({price_type}): {e}")
        time.sleep((retry_pause_ms / 1000.0) * (2 ** (attempt - 1)))
    return ""


def download_bi5_hour_ticks(inst_id: str, dt_hour: datetime, scale_factor: float = 100000.0):
    month_0idx = dt_hour.month - 1
    url = f"https://datafeed.dukascopy.com/datafeed/{inst_id.upper()}/{dt_hour.year}/{month_0idx:02d}/{dt_hour.day:02d}/{dt_hour.hour:02d}h_ticks.bi5"

    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            compressed_data = resp.read()
    except Exception:
        return pd.DataFrame()

    if not compressed_data:
        return pd.DataFrame()

    try:
        decompressed = lzma.decompress(compressed_data)
    except Exception:
        return pd.DataFrame()

    record_size = 20
    n_records = len(decompressed) // record_size
    records = []

    base_time = datetime(dt_hour.year, dt_hour.month, dt_hour.day, dt_hour.hour, tzinfo=None)
    for i in range(n_records):
        chunk = decompressed[i * record_size : (i + 1) * record_size]
        time_ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(">3I2f", chunk)
        timestamp = base_time + timedelta(milliseconds=time_ms)
        ask = ask_raw / scale_factor
        bid = bid_raw / scale_factor
        records.append({"timestamp": timestamp, "bid": bid, "ask": ask})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.set_index("timestamp", inplace=True)
    df_m1 = df.resample("1min").last().dropna()
    return df_m1


def parse_dukascopy_csv(filepath: str) -> pd.DataFrame:
    if not filepath or not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return pd.DataFrame()

        time_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()][0]
        df['timestamp'] = pd.to_datetime(df[time_col], utc=True)

        price_col = 'close' if 'close' in df.columns else ('price' if 'price' in df.columns else df.columns[1])
        df['price'] = df[price_col].astype(float)

        df = df[['timestamp', 'price']].sort_values('timestamp').drop_duplicates('timestamp')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error parsing CSV {filepath}: {e}")
        return pd.DataFrame()


def evaluate_data_quality(df_merged: pd.DataFrame, inst_id: str, year: int) -> dict:
    if df_merged.empty:
        return {
            "instrument": inst_id,
            "year": year,
            "candles_count": 0,
            "expected_coverage_pct": 0.0,
            "low_coverage_days": 0,
            "median_spread_bps": np.nan,
            "p99_spread_bps": np.nan,
            "jump_count_10mad": 0,
            "flat_sequences_gt_30m": 0
        }

    candles_count = len(df_merged)
    total_days = max((df_merged.index[-1] - df_merged.index[0]).days + 1, 1)
    expected_minutes = max(total_days * 1440 * (5/7), 1)
    coverage_pct = round(100.0 * candles_count / expected_minutes, 2)

    daily_counts = df_merged.resample('1D').count()['mid']
    low_coverage_days = int((daily_counts[daily_counts > 0] < 720).sum())

    spread_bps = 10000.0 * (df_merged['ask'] - df_merged['bid']) / df_merged['mid']
    median_spread_bps = round(float(spread_bps.median()), 2)
    p99_spread_bps = round(float(spread_bps.quantile(0.99)), 2)

    log_ret = np.log(df_merged['mid'] / df_merged['mid'].shift(1)).dropna()
    if len(log_ret) > 0:
        median_ret = log_ret.median()
        mad = (log_ret - median_ret).abs().median()
        scale = max(1.4826 * mad, 1e-4)
        robust_z = (log_ret - median_ret).abs() / scale
        jump_count = int((robust_z > 10).sum())
    else:
        jump_count = 0

    price_diff = (df_merged['mid'].diff() == 0).astype(int)
    flat_runs = price_diff.groupby((price_diff != price_diff.shift()).cumsum()).cumsum()
    flat_seq_count = int((flat_runs == 30).sum())

    return {
        "instrument": inst_id,
        "year": year,
        "candles_count": candles_count,
        "expected_coverage_pct": coverage_pct,
        "low_coverage_days": low_coverage_days,
        "median_spread_bps": median_spread_bps,
        "p99_spread_bps": p99_spread_bps,
        "jump_count_10mad": jump_count,
        "flat_sequences_gt_30m": flat_seq_count
    }


def run_download(config_path: str = "config/config.yaml", force_bi5: bool = False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    date_from, date_to = get_date_range(config)
    logger.info(f"Downloading M1 data from {date_from} to {date_to}...")

    raw_dir = config.get("paths", {}).get("raw_dir", "data/raw")
    reports_dir = config.get("paths", {}).get("reports_dir", "reports")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    instruments = config.get("instruments", [])
    quality_records = []

    for inst in instruments:
        inst_id = inst["id"]
        inst_name = inst.get("name", inst_id)
        logger.info(f"Processing instrument: {inst_name} ({inst_id})")

        out_inst_dir = os.path.join(raw_dir, inst_id)
        os.makedirs(out_inst_dir, exist_ok=True)

        parquet_out = os.path.join(raw_dir, f"{inst_id}_m1.parquet")

        if os.path.exists(parquet_out):
            logger.info(f"Cache hit for {inst_id}: {parquet_out}")
            df_existing = pd.read_parquet(parquet_out)
            q_rec = evaluate_data_quality(df_existing, inst_id, datetime.now().year)
            quality_records.append(q_rec)
            continue

        df_bid = pd.DataFrame()
        df_ask = pd.DataFrame()

        if not force_bi5:
            bid_csv = download_dukascopy_node(inst_id, date_from, date_to, "bid", out_inst_dir)
            ask_csv = download_dukascopy_node(inst_id, date_from, date_to, "ask", out_inst_dir)

            df_bid = parse_dukascopy_csv(bid_csv)
            df_ask = parse_dukascopy_csv(ask_csv)

        if (df_bid.empty or df_ask.empty) and force_bi5:
            logger.info(f"Attempting fallback .bi5 download for {inst_id}...")
            dt_start = datetime.strptime(date_from, "%Y-%m-%d")
            dt_end = datetime.strptime(date_to, "%Y-%m-%d")
            curr = dt_start
            bid_list, ask_list = [], []
            scale = 1000.0 if "jpy" in inst_id.lower() or "idx" in inst_id.lower() or "cmd" in inst_id.lower() else 100000.0
            while curr <= dt_end:
                df_hr = download_bi5_hour_ticks(inst_id, curr, scale_factor=scale)
                if not df_hr.empty:
                    bid_list.append(df_hr[['bid']].rename(columns={'bid': 'price'}))
                    ask_list.append(df_hr[['ask']].rename(columns={'ask': 'price'}))
                curr += timedelta(hours=1)
            if bid_list and ask_list:
                df_bid = pd.concat(bid_list)
                df_ask = pd.concat(ask_list)

        if df_bid.empty or df_ask.empty:
            logger.warning(f"Failed to fetch data for instrument {inst_id}. Skipping.")
            q_rec = evaluate_data_quality(pd.DataFrame(), inst_id, datetime.now().year)
            quality_records.append(q_rec)
            continue

        df_merged = pd.merge(
            df_bid.rename(columns={'price': 'bid'}),
            df_ask.rename(columns={'price': 'ask'}),
            left_index=True, right_index=True, how='inner'
        )
        df_merged['mid'] = (df_merged['bid'] + df_merged['ask']) / 2.0

        if not df_merged.empty:
            df_merged.to_parquet(parquet_out)
            logger.info(f"Saved {len(df_merged)} rows to {parquet_out}")

        q_rec = evaluate_data_quality(df_merged, inst_id, datetime.now().year)
        quality_records.append(q_rec)

    df_quality = pd.DataFrame(quality_records)
    quality_csv = os.path.join(reports_dir, "data_quality.csv")
    df_quality.to_csv(quality_csv, index=False)
    logger.info(f"Data quality report saved to {quality_csv}")
    return df_quality


if __name__ == "__main__":
    run_download()
