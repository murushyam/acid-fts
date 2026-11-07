"""
ACID-FTS Framework - Phase 2: VMD-Fuzzy Residual Learning (ICPDA)
=================================================================
Implements Algorithm 2 and Equations (7)-(13) of the manuscript:
1. Extracts training residuals r_t = y_t - y_hat_base(t) from the Phase 1 RF predictor.
2. Decomposes r_t into K = 5 orthogonal Intrinsic Mode Functions (IMFs) {u_k(t)}_{k=1}^K
   via Variational Mode Decomposition (VMD) ADMM spectral updates (Eq. 8-10).
3. Fuzzifies each IMF u_k(t) into n = 5 linguistic triangular fuzzy sets
   {VL, L, M, H, VH} using empirical quantile centers c_1 < ... < c_5 (Eq. 11).
4. Trains K mode-specific Fuzzy Random Forest regressors F_k(mu_k(t), X_t) (Eq. 12)
   and reconstructs domain-constrained forecasts y_hat_P2 = max(0, y_hat_base + r_hat) (Eq. 13).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import config
from utils.metrics import calculate_metrics


def vmd_decompose_1d(signal, K=config.VMD_K_MODES, alpha=config.VMD_ALPHA, tau=config.VMD_TAU, tol=config.VMD_TOL):
    """
    Equations (8)-(10): Decomposes a 1D residual signal r(t) into K narrow-band
    Intrinsic Mode Functions (IMFs) via Variational Mode Decomposition (ADMM).
    Uses `vmdpy.VMD` when installed, or an exact frequency-domain Wiener-filter
    ADMM implementation otherwise.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n_len = len(signal)

    try:
        from vmdpy import VMD
        u, _, _ = VMD(signal, alpha, tau, K, 0, 1, tol)
        # Ensure output shape matches (K, n_len)
        if u.shape[1] < n_len:
            pad_w = n_len - u.shape[1]
            u = np.pad(u, ((0, 0), (0, pad_w)), mode="edge")
        return u[:, :n_len]
    except Exception:
        # Exact spectral ADMM Wiener filter decomposition over K frequency bands
        f_hat = np.fft.rfft(signal)
        freqs = np.linspace(0.0, 0.5, len(f_hat))
        omega_k = np.linspace(0.02, 0.42, K)
        u_hat = np.zeros((K, len(f_hat)), dtype=np.complex128)
        lambda_hat = np.zeros(len(f_hat), dtype=np.complex128)

        for _ in range(60):
            u_hat_prev = u_hat.copy()
            for k in range(K):
                sum_other = np.sum(u_hat, axis=0) - u_hat[k]
                numer = f_hat - sum_other + lambda_hat / 2.0
                denom = 1.0 + 2.0 * alpha * ((freqs - omega_k[k]) ** 2)
                u_hat[k] = numer / denom
                power = np.abs(u_hat[k]) ** 2
                if np.sum(power) > 1e-12:
                    omega_k[k] = np.sum(freqs * power) / np.sum(power)

            lambda_hat = lambda_hat + tau * (f_hat - np.sum(u_hat, axis=0))
            rel_diff = np.sum(np.abs(u_hat - u_hat_prev) ** 2) / (np.sum(np.abs(u_hat_prev) ** 2) + 1e-12)
            if rel_diff < tol:
                break

        modes = np.array([np.fft.irfft(u_hat[k], n=n_len) for k in range(K)])
        return modes


def compute_triangular_fuzzy_memberships(mode_series, centers=None):
    """
    Equation (11): Maps a scalar mode series u_k(t) into an n = 5 fuzzy membership
    degree vector mu_k(t) = [mu_A1, ..., mu_A5]^T using overlapping asymmetric
    triangular membership functions with quantile centers c_1 < c_2 < c_3 < c_4 < c_5.
    """
    u = np.asarray(mode_series, dtype=np.float64)
    if centers is None:
        quantiles = np.linspace(0.05, 0.95, config.FUZZY_N_SETS)
        centers = np.quantile(u, quantiles)
        # Guarantee strictly increasing centers even on low-variance modes
        for j in range(1, len(centers)):
            if centers[j] <= centers[j - 1]:
                centers[j] = centers[j - 1] + 1e-4

    n_sets = len(centers)
    mu = np.zeros((len(u), n_sets), dtype=np.float64)

    for j in range(n_sets):
        c_j = centers[j]
        c_left = centers[j - 1] if j > 0 else c_j - (centers[1] - c_j + 1e-4)
        c_right = centers[j + 1] if j < n_sets - 1 else c_j + (c_j - centers[-2] + 1e-4)

        left_slope = (u - c_left) / max(c_j - c_left, 1e-6)
        right_slope = (c_right - u) / max(c_right - c_j, 1e-6)

        if j == 0:
            # Left boundary shoulder
            mu[:, j] = np.where(u <= c_j, 1.0, np.clip(right_slope, 0.0, 1.0))
        elif j == n_sets - 1:
            # Right boundary shoulder
            mu[:, j] = np.where(u >= c_j, 1.0, np.clip(left_slope, 0.0, 1.0))
        else:
            mu[:, j] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    return mu, centers


def run_phase_2(data_splits, p1_preds, lag):
    """
    Executes Phase 2 (Algorithm 2): VMD-Fuzzy Residual Learning (ICPDA).
    """
    print(f"\n[Phase 2] Running VMD-Fuzzy Residual Learning / ICPDA (Lag={lag}d)...")

    X_train, y_train = data_splits["X_train"], data_splits["y_train"]
    X_cal, y_cal = data_splits["X_cal"], data_splits["y_cal"]
    X_val, y_val = data_splits["X_val"], data_splits["y_val"]
    X_future, y_future = data_splits["X_future"], data_splits["y_future"]

    X_tr_flat = X_train.reshape(len(X_train), -1)
    X_cal_flat = X_cal.reshape(len(X_cal), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_fut_flat = X_future.reshape(len(X_future), -1)

    base_tr = p1_preds["Train_RF"]
    base_cal = p1_preds["Cal_RF"]
    base_val = p1_preds["RF"]
    base_fut = p1_preds["Future_RF"]

    n_targets = y_train.shape[1]
    pred_p2_cal = np.zeros_like(y_cal)
    pred_p2_val = np.zeros_like(y_val)
    pred_p2_fut = np.zeros_like(y_future)

    for t_idx in range(n_targets):
        # Equation (7): Training residual series r_t = y_t - y_hat_base(t)
        res_tr = y_train[:, t_idx] - base_tr[:, t_idx]

        # Equations (8)-(10): Decompose training residual into K = 5 IMFs
        imfs_tr = vmd_decompose_1d(res_tr, K=config.VMD_K_MODES)

        # Estimate IMF states for evaluation splits using a causal linear filter from X_flat
        r_hat_cal = np.zeros(len(X_cal_flat))
        r_hat_val = np.zeros(len(X_val_flat))
        r_hat_fut = np.zeros(len(X_fut_flat))

        for k in range(config.VMD_K_MODES):
            u_k_tr = imfs_tr[k]
            mu_tr, centers_k = compute_triangular_fuzzy_memberships(u_k_tr)

            # Predict mode amplitude u_k from meteorological lookback features X_flat to avoid future leakage
            mode_proxy = RandomForestRegressor(n_estimators=40, max_depth=8, random_state=config.RANDOM_SEED + k, n_jobs=-1)
            mode_proxy.fit(X_tr_flat, u_k_tr)

            u_k_cal_est = mode_proxy.predict(X_cal_flat)
            u_k_val_est = mode_proxy.predict(X_val_flat)
            u_k_fut_est = mode_proxy.predict(X_fut_flat)

            mu_cal, _ = compute_triangular_fuzzy_memberships(u_k_cal_est, centers=centers_k)
            mu_val, _ = compute_triangular_fuzzy_memberships(u_k_val_est, centers=centers_k)
            mu_fut, _ = compute_triangular_fuzzy_memberships(u_k_fut_est, centers=centers_k)

            # Equation (12): Mode-specific Fuzzy Random Forest F_k(mu_k(t), X_t)
            fuzzy_rf = RandomForestRegressor(n_estimators=60, max_depth=10, random_state=config.RANDOM_SEED + 10 + k, n_jobs=-1)
            fuzzy_rf.fit(np.hstack([mu_tr, X_tr_flat]), u_k_tr)

            r_hat_cal += fuzzy_rf.predict(np.hstack([mu_cal, X_cal_flat]))
            r_hat_val += fuzzy_rf.predict(np.hstack([mu_val, X_val_flat]))
            r_hat_fut += fuzzy_rf.predict(np.hstack([mu_fut, X_fut_flat]))

        # Equation (13): Reconstruct residual-corrected forecast with non-negativity projection
        pred_p2_cal[:, t_idx] = np.maximum(0.0, base_cal[:, t_idx] + r_hat_cal)
        pred_p2_val[:, t_idx] = np.maximum(0.0, base_val[:, t_idx] + r_hat_val)
        pred_p2_fut[:, t_idx] = np.maximum(0.0, base_fut[:, t_idx] + r_hat_fut)

    # Integer rounding for Fire Count
    pred_p2_cal[:, 1] = np.round(pred_p2_cal[:, 1])
    pred_p2_val[:, 1] = np.round(pred_p2_val[:, 1])
    pred_p2_fut[:, 1] = np.round(pred_p2_fut[:, 1])

    m_res = calculate_metrics(y_val, pred_p2_val)
    print(f"  Phase 2 VMD-Fuzzy (ICPDA) Holdout MAE: {m_res['MAE']:.4f}")

    metrics_rows = []
    for t_idx, t_col in enumerate(config.TARGET_COLS):
        metrics_rows.append({
            "Phase": "Phase_2_VMD_Fuzzy",
            "Model": "ICPDA_VMD_Fuzzy",
            "Target": t_col,
            "Lag": lag,
            "MAE": m_res["MAE_Raw"][t_idx],
            "RMSE": m_res["RMSE_Raw"][t_idx],
            "LinEx": m_res["LinEx_Raw"][t_idx],
        })

    preds_map = {
        "FuzzyVMD": pred_p2_val,
        "Cal_FuzzyVMD": pred_p2_cal,
        "Future_FuzzyVMD": pred_p2_fut,
    }
    return preds_map, pd.DataFrame(metrics_rows)

