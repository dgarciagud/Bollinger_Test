import os
import pytest
import pandas as pd
import numpy as np
from vol_spillover.events import analyze_events
from vol_spillover.plots import generate_plots
from vol_spillover.summary import generate_summary_md


def test_analyze_events_and_reporting(tmp_path):
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    log_rv_df = pd.DataFrame({
        "usa500idxusd": np.random.normal(0, 1, 50),
        "lightcmdusd": np.random.normal(0, 1, 50),
        "eurusd": np.random.normal(0, 1, 50)
    }, index=dates)

    abnormal_flags = pd.DataFrame(False, index=dates, columns=log_rv_df.columns)
    abnormal_flags.iloc[20, 1] = True # lightcmdusd abnormal vol on day 20

    dy_npdc = {
        "dates": dates[20:],
        "npdc_raw": {("lightcmdusd", "usa500idxusd"): np.ones(30) * 5.0}
    }

    config = {
        "paths": {"reports_dir": str(tmp_path)},
        "instruments": [
            {"id": "usa500idxusd", "role": "receptor"},
            {"id": "lightcmdusd", "role": "candidato"},
            {"id": "eurusd", "role": "candidato"}
        ]
    }

    df_events, top10 = analyze_events(log_rv_df, abnormal_flags, dy_npdc, config)
    assert os.path.exists(os.path.join(str(tmp_path), "events_response.csv"))
    assert os.path.exists(os.path.join(str(tmp_path), "top_episodes.csv"))

    # Test summary generation
    dy_res = {"tci": pd.Series(np.ones(30)*20, index=dates[20:])}
    q_res = {
        0.50: {"net": pd.DataFrame({"lightcmdusd": [5.0]*30}, index=dates[20:])},
        0.95: {"net": pd.DataFrame({"lightcmdusd": [10.0]*30}, index=dates[20:])}
    }

    summary_text = generate_summary_md(dy_res, q_res, {}, {}, config)
    assert "Informe de Transmisión de Volatilidad" in summary_text
    assert os.path.exists(os.path.join(str(tmp_path), "summary.md"))
