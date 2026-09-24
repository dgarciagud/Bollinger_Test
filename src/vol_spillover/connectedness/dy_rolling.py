import numpy as np
import pandas as pd
from vol_spillover.var_utils import (
    select_var_lag, estimate_var_ols, compute_ma_coefficients,
    compute_gfevd, compute_connectedness_metrics
)


def run_rolling_dy(log_rv_df: pd.DataFrame, window: int = 100, H: int = 10, max_lag: int = 5) -> dict:
    T, N = log_rv_df.shape
    asset_names = list(log_rv_df.columns)

    if T < window:
        raise ValueError(f"Data length {T} is smaller than rolling window {window}")

    tci_series = []
    to_dict = {asset: [] for asset in asset_names}
    from_dict = {asset: [] for asset in asset_names}
    net_dict = {asset: [] for asset in asset_names}
    npdc_dict = {} # (candidate, receptor) -> list of values

    dates = log_rv_df.index[window:]

    for i in range(window, T):
        sub_df = log_rv_df.iloc[i - window : i].dropna()
        if len(sub_df) < window * 0.8:
            continue

        lag = select_var_lag(sub_df, max_lag=max_lag)
        A_list, sigma_u, _ = estimate_var_ols(sub_df, lag=lag)
        Phi = compute_ma_coefficients(A_list, H=H)
        gfevd = compute_gfevd(Phi, sigma_u, H=H)
        metrics = compute_connectedness_metrics(gfevd, asset_names)

        tci_series.append(metrics["tci"])
        for asset in asset_names:
            to_dict[asset].append(metrics["to"][asset])
            from_dict[asset].append(metrics["from"][asset])
            net_dict[asset].append(metrics["net"][asset])

        npdc_df = metrics["npdc"]
        for c1 in asset_names:
            for c2 in asset_names:
                key = (c1, c2)
                if key not in npdc_dict:
                    npdc_dict[key] = []
                npdc_dict[key].append(npdc_df.loc[c1, c2])

    df_tci = pd.Series(tci_series, index=dates[:len(tci_series)], name="TCI")
    df_to = pd.DataFrame(to_dict, index=dates[:len(tci_series)])
    df_from = pd.DataFrame(from_dict, index=dates[:len(tci_series)])
    df_net = pd.DataFrame(net_dict, index=dates[:len(tci_series)])

    return {
        "tci": df_tci,
        "to": df_to,
        "from": df_from,
        "net": df_net,
        "npdc_raw": npdc_dict,
        "dates": dates[:len(tci_series)]
    }
