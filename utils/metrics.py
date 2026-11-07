"""
ACID-FTS Framework - Deterministic & Probabilistic Evaluation Metrics
=====================================================================
Implements all evaluation metrics defined in the manuscript:
- Root Mean Square Error (RMSE, Eq. 5)
- Mean Absolute Error (MAE, Eq. 6)
- Scale-Normalized Asymmetric LinEx Fire Penalty Loss (L_asym, Eq. 21)
- Prediction Interval Coverage Probability (PICP, Eq. 33)
- Prediction Interval Normalized Average Width (PINAW, Eq. 34)
- Average Coverage Error (ACE, Eq. 35)
"""

import numpy as np
import config


def compute_normalized_linex_loss(y_true, y_pred, b=config.LINEX_ASYMMETRY_B):
    """
    Equation (21): Scale-Normalized Linear-Exponential (LinEx) Asymmetric Loss.
    Penalizes catastrophic under-prediction (e_norm > 0) exponentially while
    penalizing false-alarm over-prediction (e_norm < 0) linearly.
    Normalized by target standard deviation sigma_Y to prevent overflow across
    large burned-area spikes.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sigma_y = max(float(np.std(y_true)), 1e-6)
    e_norm = np.clip((y_true - y_pred) / sigma_y, -10.0, 10.0)
    loss = (2.0 / (b ** 2)) * (np.exp(b * e_norm) - b * e_norm - 1.0)
    return float(np.mean(loss))


def compute_conformal_metrics(y_true, y_lower, y_upper, alpha=config.CONFORMAL_ALPHA):
    """
    Equations (33)-(35): Computes PICP, PINAW, and ACE for bounded conformal
    prediction intervals [L_{t+L}, U_{t+L}].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_lower = np.asarray(y_lower, dtype=np.float64)
    y_upper = np.asarray(y_upper, dtype=np.float64)

    covered = (y_true >= y_lower) & (y_true <= y_upper)
    picp = float(np.mean(covered))

    y_range = max(float(np.max(y_true) - np.min(y_true)), 1e-6)
    pinaw = float(np.mean(y_upper - y_lower) / y_range)
    ace = picp - (1.0 - alpha)

    return {
        "PICP": picp,
        "PINAW": pinaw,
        "ACE": ace,
    }


def calculate_metrics(y_true, y_pred):
    """
    Computes target-specific and aggregate MAE, RMSE, and normalized LinEx asymmetric loss.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]

    mae_raw = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse_raw = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    linex_raw = np.array([
        compute_normalized_linex_loss(y_true[:, i], y_pred[:, i])
        for i in range(y_true.shape[1])
    ])

    return {
        "MAE": float(np.mean(mae_raw)),
        "RMSE": float(np.mean(rmse_raw)),
        "LinEx": float(np.mean(linex_raw)),
        "MAE_Raw": mae_raw,
        "RMSE_Raw": rmse_raw,
        "LinEx_Raw": linex_raw,
    }

