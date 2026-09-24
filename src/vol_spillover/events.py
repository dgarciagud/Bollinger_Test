import os
import logging
import yaml
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vol_spillover.events")


def analyze_events(
    log_rv_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    dy_npdc: dict,
    config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reports_dir = config.get("paths", {}).get("reports_dir", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    instruments = config.get("instruments", [])
    index_ids = [inst["id"] for inst in instruments if inst.get("role") == "receptor"]
    candidate_ids = [inst["id"] for inst in instruments if inst.get("role") == "candidato"]

    # Filter to available columns
    index_ids = [i for i in index_ids if i in log_rv_df.columns]
    candidate_ids = [c for c in candidate_ids if c in log_rv_df.columns]

    event_records = []

    for candidate in candidate_ids:
        cand_flags = abnormal_df[candidate]
        # Event days: candidate flag is True at t
        event_days = cand_flags[cand_flags].index

        for t_day in event_days:
            idx_loc = log_rv_df.index.get_loc(t_day)
            if idx_loc < 1 or idx_loc + 10 >= len(log_rv_df):
                continue

            # Check indices did not have abnormal vol at t-1
            t_prev = log_rv_df.index[idx_loc - 1]
            indices_tranquil = True
            for idx in index_ids:
                if abnormal_df.loc[t_prev, idx]:
                    indices_tranquil = False
                    break

            if not indices_tranquil:
                continue

            # Measure log RV response of indices from t to t+10
            for target_idx in index_ids:
                base_val = log_rv_df.loc[t_prev, target_idx]
                resp_vec = [log_rv_df.iloc[idx_loc + h][target_idx] - base_val for h in range(11)]

                event_records.append({
                    "candidate": candidate,
                    "target_index": target_idx,
                    "event_date": str(t_day.date()),
                    **{f"h_{h}": resp_vec[h] for h in range(11)}
                })

    df_events = pd.DataFrame(event_records)
    df_events.to_csv(os.path.join(reports_dir, "events_response.csv"), index=False)

    # Top episodes ranking by monthly NPDC transmission
    episode_records = []
    if "dates" in dy_npdc:
        dates = dy_npdc["dates"]
        npdc_raw = dy_npdc.get("npdc_raw", {})

        for i, dt in enumerate(dates):
            for cand in candidate_ids:
                for idx in index_ids:
                    key = (cand, idx)
                    val = npdc_raw.get(key, [0]*len(dates))[i] if key in npdc_raw and i < len(npdc_raw[key]) else 0.0

                    episode_records.append({
                        "date": str(dt.date()),
                        "candidate": cand,
                        "target_index": idx,
                        "npdc_to_index": val
                    })

    df_episodes = pd.DataFrame(episode_records)
    if not df_episodes.empty:
        # Sort top 10 episodes by NPDC
        top10 = df_episodes.sort_values(by="npdc_to_index", ascending=False).head(10)
        top10.to_csv(os.path.join(reports_dir, "top_episodes.csv"), index=False)
    else:
        top10 = pd.DataFrame()

    logger.info(f"Event analysis completed. Output saved to {reports_dir}")
    return df_events, top10
