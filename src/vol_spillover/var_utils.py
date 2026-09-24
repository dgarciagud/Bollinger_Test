import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def select_var_lag(df: pd.DataFrame, max_lag: int = 5) -> int:
    if len(df) <= max_lag + 5:
        return 1
    model = VAR(df)
    try:
        results = model.select_order(maxlag=max_lag)
        bic_lag = results.bic
        return int(bic_lag) if bic_lag and bic_lag > 0 else 1
    except Exception:
        return 1


def estimate_var_ols(df: pd.DataFrame, lag: int = 1):
    N = df.shape[1]
    Y = df.values
    T = len(Y)

    X_list = []
    for l in range(1, lag + 1):
        X_list.append(Y[lag - l: T - l])
    X = np.column_stack(X_list)
    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])
    Y_target = Y[lag:]

    # OLS estimation
    beta, _, _, _ = np.linalg.lstsq(X, Y_target, rcond=None)
    intercept = beta[0, :]
    coefs = beta[1:, :].T # shape (N, N*lag)

    resids = Y_target - X @ beta
    sigma_u = (resids.T @ resids) / max(len(Y_target) - (N * lag + 1), 1)

    # Reshape coefs into list of matrices [A1, A2, ..., Ap] each (N, N)
    A_list = []
    for l in range(lag):
        A_list.append(coefs[:, l * N : (l + 1) * N])

    return A_list, sigma_u, resids


def compute_ma_coefficients(A_list: list[np.ndarray], H: int) -> list[np.ndarray]:
    p = len(A_list)
    N = A_list[0].shape[0]

    Phi = [np.eye(N)] # Phi_0 = I
    for h in range(1, H):
        Phi_h = np.zeros((N, N))
        for j in range(1, min(h, p) + 1):
            Phi_h += Phi[h - j] @ A_list[j - 1]
        Phi.append(Phi_h)
    return Phi


def compute_gfevd(Phi: list[np.ndarray], sigma_u: np.ndarray, H: int) -> np.ndarray:
    N = sigma_u.shape[0]
    sigma_diag = np.diag(sigma_u)

    # Unnormalized GFEVD: Theta_tilde[i, j]
    theta_tilde = np.zeros((N, N))

    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1.0

        # Denominator for row i: sum over h of (e_i' Phi_h Sigma Phi_h' e_i)
        denom_i = 0.0
        for h in range(H):
            Phi_h = Phi[h]
            denom_i += e_i @ Phi_h @ sigma_u @ Phi_h.T @ e_i

        for j in range(N):
            if sigma_diag[j] <= 0 or denom_i <= 0:
                theta_tilde[i, j] = 0.0
                continue
            e_j = np.zeros(N)
            e_j[j] = 1.0

            numer_ij = 0.0
            for h in range(H):
                Phi_h = Phi[h]
                term = e_i @ Phi_h @ sigma_u @ e_j
                numer_ij += term ** 2

            theta_tilde[i, j] = (1.0 / sigma_diag[j]) * numer_ij / denom_i

    # Row normalization
    row_sums = theta_tilde.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    theta_norm = theta_tilde / row_sums
    return theta_norm


def compute_connectedness_metrics(gfevd: np.ndarray, asset_names: list[str]) -> dict:
    N = gfevd.shape[0]

    # Off-diagonal mask
    off_diag_mask = ~np.eye(N, dtype=bool)

    # TCI
    tci = (gfevd[off_diag_mask].sum() / N) * 100.0

    # TO_i: sum of column j excluding diagonal
    to_i = (gfevd.sum(axis=0) - np.diag(gfevd)) * 100.0

    # FROM_i: sum of row i excluding diagonal
    from_i = (gfevd.sum(axis=1) - np.diag(gfevd)) * 100.0

    # NET_i = TO_i - FROM_i
    net_i = to_i - from_i

    # NPDC_ij matrix: NPDC[i, j] = theta[j, i] - theta[i, j]
    npdc = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            npdc[i, j] = (gfevd[j, i] - gfevd[i, j]) * 100.0

    return {
        "gfevd": pd.DataFrame(gfevd, index=asset_names, columns=asset_names),
        "tci": float(tci),
        "to": pd.Series(to_i, index=asset_names),
        "from": pd.Series(from_i, index=asset_names),
        "net": pd.Series(net_i, index=asset_names),
        "npdc": pd.DataFrame(npdc, index=asset_names, columns=asset_names)
    }
