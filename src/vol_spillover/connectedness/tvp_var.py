import numpy as np
import pandas as pd
from vol_spillover.var_utils import (
    estimate_var_ols, compute_ma_coefficients,
    compute_gfevd, compute_connectedness_metrics
)


def run_tvp_var(
    log_rv_df: pd.DataFrame,
    lag: int = 1,
    H: int = 10,
    kappa1: float = 0.99,
    kappa2: float = 0.99,
    training_window: int = 100
) -> dict:
    Y = log_rv_df.values
    T, N = Y.shape
    asset_names = list(log_rv_df.columns)

    if T <= training_window:
        training_window = min(T // 2, 50)

    # Prior initialization from OLS on training window
    init_df = log_rv_df.iloc[:training_window]
    A_list_init, sigma_u_init, _ = estimate_var_ols(init_df, lag=lag)

    # Flatten coefficient matrix B (shape (N*lag, N))
    B_t = np.column_stack(A_list_init) # shape (N, N*lag) -> transpose later
    B_flat = B_t.flatten() # size N*N*lag
    P_t = np.eye(len(B_flat)) * 0.1 # state covariance
    sigma_t = sigma_u_init.copy() # observation noise covariance

    tci_series = []
    to_dict = {asset: [] for asset in asset_names}
    from_dict = {asset: [] for asset in asset_names}
    net_dict = {asset: [] for asset in asset_names}

    dates = log_rv_df.index[training_window:]

    for t in range(training_window, T):
        y_t = Y[t]

        # Build regressor vector x_t from previous lags
        x_t_list = []
        for l in range(1, lag + 1):
            x_t_list.append(Y[t - l])
        x_t = np.concatenate(x_t_list) # size N*lag

        # Kronecker design matrix X_t = I_N (x) x_t' (shape N, N*N*lag)
        X_t = np.kron(np.eye(N), x_t)

        # 1. State prediction & covariance inflation via forgetting factor kappa1
        P_t_pred = P_t / kappa1

        # 2. Measurement prediction & error
        y_pred = X_t @ B_flat
        e_t = y_t - y_pred

        # 3. Innovation covariance & Kalman gain
        F_t = X_t @ P_t_pred @ X_t.T + sigma_t
        try:
            K_t = P_t_pred @ X_t.T @ np.linalg.inv(F_t)
        except np.linalg.LinAlgError:
            K_t = P_t_pred @ X_t.T @ np.linalg.pinv(F_t)

        # 4. State update
        B_flat = B_flat + K_t @ e_t
        P_t = P_t_pred - K_t @ X_t @ P_t_pred

        # 5. Time-varying covariance update via forgetting factor kappa2
        e_t_col = e_t.reshape(-1, 1)
        sigma_t = kappa2 * sigma_t + (1.0 - kappa2) * (e_t_col @ e_t_col.T)

        # Reconstruct A_list for GFEVD
        B_mat = B_flat.reshape(N, N * lag)
        A_list_t = []
        for l in range(lag):
            A_list_t.append(B_mat[:, l * N : (l + 1) * N])

        # Compute GFEVD
        Phi_t = compute_ma_coefficients(A_list_t, H=H)
        gfevd_t = compute_gfevd(Phi_t, sigma_t, H=H)
        metrics = compute_connectedness_metrics(gfevd_t, asset_names)

        tci_series.append(metrics["tci"])
        for asset in asset_names:
            to_dict[asset].append(metrics["to"][asset])
            from_dict[asset].append(metrics["from"][asset])
            net_dict[asset].append(metrics["net"][asset])

    df_tci = pd.Series(tci_series, index=dates[:len(tci_series)], name="TCI")
    df_to = pd.DataFrame(to_dict, index=dates[:len(tci_series)])
    df_from = pd.DataFrame(from_dict, index=dates[:len(tci_series)])
    df_net = pd.DataFrame(net_dict, index=dates[:len(tci_series)])

    return {
        "tci": df_tci,
        "to": df_to,
        "from": df_from,
        "net": df_net,
        "dates": dates[:len(tci_series)]
    }
