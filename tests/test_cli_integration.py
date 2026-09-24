import os
import pytest
import yaml
import pandas as pd
import numpy as np
from typer.testing import CliRunner
from vol_spillover.cli import app

runner = CliRunner()


def test_cli_full_pipeline_synthetic(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    clean_dir = tmp_path / "data" / "clean"
    features_dir = tmp_path / "data" / "features"
    reports_dir = tmp_path / "reports"

    raw_dir.mkdir(parents=True)
    clean_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    config = {
        "download": {"history_days": 30},
        "cleaning": {"spread_multiplier_k": 10, "min_coverage_pct": 50, "exclude_dates_mmdd": [], "day_boundary_utc": "22:00"},
        "realized": {"subsample_minutes": 5, "abnormal_percentile": 95, "abnormal_rolling_window": 10},
        "connect": {"var_max_lag": 2, "forecast_horizon": 5, "dy_rolling_window": 5},
        "paths": {
            "raw_dir": str(raw_dir),
            "clean_dir": str(clean_dir),
            "features_dir": str(features_dir),
            "reports_dir": str(reports_dir)
        },
        "instruments": [
            {"id": "usa500idxusd", "name": "S&P 500", "role": "receptor"},
            {"id": "lightcmdusd", "name": "WTI", "role": "candidato"},
            {"id": "eurusd", "name": "EURUSD", "role": "candidato"}
        ]
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Generate synthetic M1 parquet files for 15 days
    dates = pd.date_range("2024-01-01 00:00:00", periods=15*1440, freq="1min", tz="UTC")
    for inst in ["usa500idxusd", "lightcmdusd", "eurusd"]:
        bids = 100.0 + np.cumsum(np.random.normal(0, 0.01, len(dates)))
        asks = bids + 0.02
        df_raw = pd.DataFrame({"bid": bids, "ask": asks, "mid": (bids + asks) / 2.0}, index=dates)
        df_raw.to_parquet(raw_dir / f"{inst}_m1.parquet")

    # Run clean, realized, connect, report via CLI runner
    res_clean = runner.invoke(app, ["clean", "--config", str(config_path)])
    assert res_clean.exit_code == 0

    res_realized = runner.invoke(app, ["realized", "--config", str(config_path)])
    assert res_realized.exit_code == 0

    res_connect = runner.invoke(app, ["connect", "--method", "all", "--config", str(config_path)])
    assert res_connect.exit_code == 0

    res_report = runner.invoke(app, ["report", "--config", str(config_path)])
    assert res_report.exit_code == 0

    # Verify generated outputs
    assert os.path.exists(reports_dir / "summary.md")
    assert os.path.exists(reports_dir / "tci_overlay.png")
