import logging
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from vol_spillover.var_utils import (
    select_var_lag, estimate_var_ols, compute_connectedness_metrics
)

logger = logging.getLogger("vol_spillover.frequency")


def compute_spectral_gfevd_bands(
    A_list: list[np.ndarray],
    sigma_u: np.ndarray,
    freq_bands: dict = None,
    n_freqs: int = 500
) -> dict:
    if freq_bands is None:
        freq_bands = {
            "short": (1, 5),     # 1-5 days: freq in [2*pi/5, pi]
            "medium": (5, 20),   # 5-20 days: freq in [2*pi/20, 2*pi/5]
            "long": (20, 1000)   # >20 days: freq in [0, 2*pi/20]
        }

    N = sigma_u.shape[0]
    p = len(A_list)
    sigma_diag = np.diag(sigma_u)

    # Grid of frequencies from 0 to pi
    w_grid = np.linspace(0.0001, np.pi, n_freqs)

    # Store Psi(w) for all w in w_grid
    psi_grid = []
    for w in w_grid:
        poly = np.eye(N, dtype=complex)
        for l in range(1, p + 1):
            poly -= A_list[l - 1] * np.exp(-1j * w * l)
        try:
            psi_w = np.linalg.inv(poly)
        except np.linalg.LinAlgError:
            psi_w = np.linalg.pinv(poly)
        psi_grid.append(psi_w)

    band_gfevd = {}

    for band_name, (d_low, d_high) in freq_bands.items():
        # Frequency bounds: omega = 2*pi / period
        w_high = min(2.0 * np.pi / d_low, np.pi)
        w_low = max(2.0 * np.pi / d_high, 0.0)

        mask = (w_grid >= w_low) & (w_grid <= w_high)
        if not np.any(mask):
            continue

        w_sub = w_grid[mask]
        idx_sub = np.where(mask)[0]

        theta_band_tilde = np.zeros((N, N))

        for i in range(N):
            e_i = np.zeros(N)
            e_i[i] = 1.0

            # Integrand for denominator
            denom_integrand = []
            for idx in idx_sub:
                psi_w = psi_grid[idx]
                val = e_i @ (psi_w @ sigma_u @ psi_w.conj().T) @ e_i
                denom_integrand.append(np.real(val))

            denom_val = simpson(y=denom_integrand, x=w_sub) if len(w_sub) > 1 else denom_integrand[0]

            for j in range(N):
                e_j = np.zeros(N)
                e_j[j] = 1.0

                numer_integrand = []
                for idx in idx_sub:
                    psi_w = psi_grid[idx]
                    val = (e_i @ psi_w @ sigma_u @ e_j) ** 2
                    numer_integrand.append(np.real(val))

                numer_val = simpson(y=numer_integrand, x=w_sub) if len(w_sub) > 1 else numer_integrand[0]

                if sigma_diag[j] > 0 and denom_val > 0:
                    theta_band_tilde[i, j] = (1.0 / sigma_diag[j]) * numer_val / denom_val
                else:
                    theta_band_tilde[i, j] = 0.0

        # Row normalization
        row_sums = theta_band_tilde.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        theta_norm = theta_band_tilde / row_sums

        band_gfevd[band_name] = theta_norm

    return band_gfevd


def run_frequency_connectedness(
    log_rv_df: pd.DataFrame,
    freq_bands: dict = None,
    lag: int = 1
) -> dict:
    asset_names = list(log_rv_df.columns)

    A_list, sigma_u, _ = estimate_var_ols(log_rv_df, lag=lag)
    band_gfevd = compute_spectral_gfevd_bands(A_list, sigma_u, freq_bands=freq_bands)

    freq_results = {}
    for band_name, gfevd_mat in band_gfevd.items():
        metrics = compute_connectedness_metrics(gfevd_mat, asset_names)
        freq_results[band_name] = metrics

    return freq_results
