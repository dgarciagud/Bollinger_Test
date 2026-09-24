import pytest
import numpy as np
import pandas as pd
from vol_spillover.var_utils import (
    estimate_var_ols, compute_ma_coefficients,
    compute_gfevd, compute_connectedness_metrics
)
from vol_spillover.connectedness.dy_rolling import run_rolling_dy
from vol_spillover.connectedness.tvp_var import run_tvp_var
from vol_spillover.connectedness.quantile import run_quantile_connectedness, estimate_qvar_equation
from vol_spillover.connectedness.frequency import run_frequency_connectedness


def test_synthetic_var1_spillover():
    np.random.seed(42)
    T = 500
    N = 4

    # A_matrix: A (var 0) transmits to B (var 1) and C (var 2). No transmission to A.
    A1 = np.array([
        [0.5, 0.0, 0.0, 0.0],  # A depends only on lagged A
        [0.4, 0.3, 0.0, 0.0],  # B depends on lagged A and lagged B
        [0.4, 0.0, 0.3, 0.0],  # C depends on lagged A and lagged C
        [0.0, 0.0, 0.0, 0.4]   # D independent
    ])

    # Simulate Y
    Y = np.zeros((T, N))
    errors = np.random.normal(0, 0.1, size=(T, N))
    for t in range(1, T):
        Y[t] = A1 @ Y[t - 1] + errors[t]

    asset_names = ["A", "B", "C", "D"]
    df = pd.DataFrame(Y, columns=asset_names)

    A_list_est, sigma_u_est, _ = estimate_var_ols(df, lag=1)
    Phi = compute_ma_coefficients(A_list_est, H=10)
    gfevd = compute_gfevd(Phi, sigma_u_est, H=10)
    metrics = compute_connectedness_metrics(gfevd, asset_names)

    # 1. Normalized GFEVD rows sum to 1 within 1e-6 tolerance
    row_sums = gfevd.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(N), atol=1e-6)

    # 2. Sum of NET across all variables equals 0
    net_sum = metrics["net"].sum()
    np.testing.assert_allclose(net_sum, 0.0, atol=1e-5)

    # 3. Variable A is a net transmitter (NET_A > 0)
    assert metrics["net"]["A"] > 0, f"NET_A should be positive, got {metrics['net']['A']}"

    # 4. NPDC A -> B > 0 and NPDC A -> C > 0
    npdc = metrics["npdc"]
    assert npdc.loc["A", "B"] > 0, f"NPDC A->B should be > 0, got {npdc.loc['A', 'B']}"
    assert npdc.loc["A", "C"] > 0, f"NPDC A->C should be > 0, got {npdc.loc['A', 'C']}"


def test_tvp_var_convergence_to_static_var():
    np.random.seed(42)
    T = 200
    N = 3

    A1 = np.array([
        [0.4, 0.1, 0.0],
        [0.0, 0.4, 0.1],
        [0.1, 0.0, 0.4]
    ])

    Y = np.zeros((T, N))
    errors = np.random.normal(0, 0.1, size=(T, N))
    for t in range(1, T):
        Y[t] = A1 @ Y[t - 1] + errors[t]

    df = pd.DataFrame(Y, columns=["X1", "X2", "X3"])

    # Static VAR metrics
    A_list_est, sigma_u_est, _ = estimate_var_ols(df, lag=1)
    Phi = compute_ma_coefficients(A_list_est, H=10)
    gfevd_static = compute_gfevd(Phi, sigma_u_est, H=10)
    metrics_static = compute_connectedness_metrics(gfevd_static, ["X1", "X2", "X3"])

    # TVP-VAR with kappa1=0.9999, kappa2=0.9999 (almost static)
    tvp_res = run_tvp_var(df, lag=1, H=10, kappa1=0.9999, kappa2=0.9999, training_window=50)
    tci_tvp_final = tvp_res["tci"].iloc[-1]

    # Difference in TCI should be small
    diff = abs(metrics_static["tci"] - tci_tvp_final)
    assert diff < 5.0, f"Difference between TVP and static TCI too large: {diff:.2f}"


def test_quantile_tau_05_vs_ols():
    np.random.seed(42)
    T = 200
    N = 3

    Y = np.random.normal(0, 1, size=(T, N))
    df = pd.DataFrame(Y, columns=["A", "B", "C"])

    A_ols, sigma_ols = estimate_var_ols(df, lag=1)[:2]
    A_q, sigma_q = estimate_qvar_equation(df, tau=0.5, lag=1)

    # QVAR at tau=0.5 should be close to OLS for Gaussian data
    np.testing.assert_allclose(A_q[0], A_ols[0], atol=0.2)


def test_frequency_connectedness():
    np.random.seed(42)
    T = 200
    N = 3
    df = pd.DataFrame(np.random.normal(0, 1, (T, N)), columns=["A", "B", "C"])

    freq_res = run_frequency_connectedness(df, lag=1)
    assert "short" in freq_res
    assert "medium" in freq_res
    assert "long" in freq_res
    assert "tci" in freq_res["short"]
