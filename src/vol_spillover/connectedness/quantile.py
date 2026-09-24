import logging
import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
from vol_spillover.var_utils import (
    compute_ma_coefficients, compute_gfevd, compute_connectedness_metrics
)

logger = logging.getLogger("vol_spillover.quantile")


def estimate_qvar_equation(df: pd.DataFrame, tau: float = 0.5, lag: int = 1):
    N = df.shape[1]
    Y = df.values
    T = len(Y)

    X_list = []
    for l in range(1, lag + 1):
        X_list.append(Y[lag - l: T - l])
    X = np.column_stack(X_list)
    # Intercept
    X_full = np.column_stack([np.ones(len(X)), X])
    Y_target = Y[lag:]

    B_mat = np.zeros((N, N * lag))
    resids = np.zeros_like(Y_target)

    for i in range(N):
        y_i = Y_target[:, i]
        try:
            mod = QuantReg(y_i, X_full)
            res = mod.fit(q=tau, max_iter=2000)
            B_i = res.params[1:]
            resids[:, i] = y_i - X_full @ res.params
        except Exception as e:
            # Fallback to OLS if quantile regression fails
            beta, _, _, _ = np.linalg.lstsq(X_full, y_i, rcond=None)
            B_i = beta[1:]
            resids[:, i] = y_i - X_full @ beta

        B_mat[i, :] = B_i

    A_list = []
    for l in range(lag):
        A_list.append(B_mat[:, l * N : (l + 1) * N])

    sigma_u = (resids.T @ resids) / max(len(Y_target) - (N * lag + 1), 1)
    return A_list, sigma_u


def run_quantile_connectedness(
    log_rv_df: pd.DataFrame,
    quantiles: list[float] = [0.05, 0.50, 0.95],
    window: int = 100,
    H: int = 10,
    lag: int = 1
) -> dict:
    T, N = log_rv_df.shape
    asset_names = list(log_rv_df.columns)

    if T < window:
        window = min(T, 50)

    results_by_tau = {}

    for q in quantiles:
        logger.info(f"Running Quantile VAR for tau = {q}...")
        tci_series = []
        to_dict = {asset: [] for asset in asset_names}
        from_dict = {asset: [] for asset in asset_names}
        net_dict = {asset: [] for asset in asset_names}
        npdc_dict = {}

        dates = log_rv_df.index[window:]

        for i in range(window, T):
            sub_df = log_rv_df.iloc[i - window : i].dropna()
            if len(sub_df) < max(20, lag + 5):
                continue

            # Check if effective tail observations in window are too low for q=0.95
            if q >= 0.9 and len(sub_df) * (1 - q) < 5:
                logger.debug(f"Warning: sparse tail observations ({len(sub_df)*(1-q):.1f}) for tau={q} at index {i}")

            A_list, sigma_u = estimate_qvar_equation(sub_df, tau=q, lag=lag)
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

        df_tci = pd.Series(tci_series, index=dates[:len(tci_series)], name=f"TCI_q{int(q*100)}")
        df_to = pd.DataFrame(to_dict, index=dates[:len(tci_series)])
        df_from = pd.DataFrame(from_dict, index=dates[:len(tci_series)])
        df_net = pd.DataFrame(net_dict, index=dates[:len(tci_series)])

        results_by_tau[q] = {
            "tci": df_tci,
            "to": df_to,
            "from": df_from,
            "net": df_net,
            "npdc_raw": npdc_dict,
            "dates": dates[:len(tci_series)]
        }

    return results_by_tau
