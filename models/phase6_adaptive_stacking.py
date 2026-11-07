"""
ACID-FTS Framework - Phase 6: Adaptive Residual Meta-Stacking
=============================================================
Implements Algorithm 6 and Equations (24)-(27) of the manuscript:
1. Computes strictly forecast-origin observable lagged residuals [e_t, e_{t-1}]
   where e_t = y_t - y_hat_{t|t-L}^{(M*)} is already realized at forecast origin t (Eq. 24).
2. Assembles the meta-feature vector z_t = [y_hat_{t+L|t}^{(M*)}, e_t, e_{t-1}, T_t, P_t, VPD_t]^T
   combining point forecasts, observable recent errors, and physical weather context (Eq. 25).
3. Trains an Optuna-optimized XGBoost residual corrector g_phi(z_t) on D_cal.
4. Enforces physical non-negativity max(0, .) and integer rounding for Fire Count (Eq. 26-27).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import config
from utils.metrics import calculate_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_observable_lagged_residuals(residuals_series, lag):
    """
    Equation (24): At forecast origin t (predicting target t + L), the most recent
    realized forecast error is e_t = y_t - y_hat_{t|t-L}^{(M*)}, which is shifted
    by L steps relative to the target index t + L.
    Returns strictly causal observable lagged residuals [e_t, e_{t-1}].
    """
    n = len(residuals_series)
    e_t = np.zeros(n, dtype=np.float64)
    e_tm1 = np.zeros(n, dtype=np.float64)

    for idx in range(n):
        obs_idx_1 = idx - lag
        obs_idx_2 = idx - lag - 1
        if obs_idx_1 >= 0:
            e_t[idx] = residuals_series[obs_idx_1]
        if obs_idx_2 >= 0:
            e_tm1[idx] = residuals_series[obs_idx_2]

    return np.column_stack([e_t, e_tm1])


def _optimize_xgb_meta_learner(Z_tr, r_tr, Z_va, r_va):
    """Tunes XGBoost residual meta-learner g_phi(z_t) via Optuna on D_cal."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 25, 120),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
            "random_state": config.RANDOM_SEED,
            "n_jobs": -1,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(Z_tr, r_tr)
        preds = model.predict(Z_va)
        return float(np.sqrt(np.mean((r_va - preds) ** 2)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.OPTUNA_TRIALS)

    best_meta = xgb.XGBRegressor(**study.best_params, random_state=config.RANDOM_SEED, n_jobs=-1)
    best_meta.fit(np.vstack([Z_tr, Z_va]), np.concatenate([r_tr, r_va]))
    return best_meta


def run_phase_6(data_splits, p5_preds, lag):
    """
    Executes Phase 6 (Algorithm 6): Adaptive Residual Meta-Stacking with Physical Domain Enforcement.
    """
    print(f"\n[Phase 6] Running Adaptive Residual Meta-Stacking (Lag={lag}d)...")

    X_cal, y_cal = data_splits["X_cal"], data_splits["y_cal"]
    X_val, y_val = data_splits["X_val"], data_splits["y_val"]
    X_future, y_future = data_splits["X_future"], data_splits["y_future"]

    champ_cal = p5_preds["Cal_Champion"]
    champ_val = p5_preds["Champion"]
    champ_fut = p5_preds["Future_Champion"]

    # Extract physical weather covariates [T_t, P_t, VPD_t] at forecast origin t
    weather_indices = [config.FEATURES.index(c) for c in config.STACKING_WEATHER_COLS]
    W_cal = X_cal[:, -1, weather_indices]
    W_val = X_val[:, -1, weather_indices]
    W_fut = X_future[:, -1, weather_indices]

    pred_stacked_cal = np.zeros_like(y_cal)
    pred_stacked_val = np.zeros_like(y_val)
    pred_stacked_fut = np.zeros_like(y_future)

    n_targets = y_val.shape[1]
    for t_idx in range(n_targets):
        # Realized residuals on D_cal and D_test
        res_cal = y_cal[:, t_idx] - champ_cal[:, t_idx]
        res_val = y_val[:, t_idx] - champ_val[:, t_idx]

        # Construct strictly forecast-origin observable lagged residuals [e_t, e_{t-1}]
        lag_res_cal = _build_observable_lagged_residuals(res_cal, lag)

        # For D_test, prepend D_cal residuals so initial holdout origins observe realized D_cal errors
        combined_res = np.concatenate([res_cal, res_val])
        lag_res_combined = _build_observable_lagged_residuals(combined_res, lag)
        lag_res_val = lag_res_combined[len(res_cal):]

        # For future analog projection window, observable lagged residuals decay toward zero mean
        lag_res_fut = np.zeros((len(champ_fut), 2), dtype=np.float64)

        # Equation (25): Meta-feature vector z_t = [y_hat^{(M*)}, e_t, e_{t-1}, T_t, P_t, VPD_t]^T
        Z_cal = np.column_stack([champ_cal[:, t_idx], lag_res_cal, W_cal])
        Z_val = np.column_stack([champ_val[:, t_idx], lag_res_val, W_val])
        Z_fut = np.column_stack([champ_fut[:, t_idx], lag_res_fut, W_fut])

        # Chronological split inside D_cal for Optuna tuning of g_phi
        split_idx = int(len(Z_cal) * 0.7)
        meta_learner = _optimize_xgb_meta_learner(
            Z_cal[:split_idx], res_cal[:split_idx],
            Z_cal[split_idx:], res_cal[split_idx:],
        )

        # Equation (26): Additive residual correction + physical non-negativity projection max(0, .)
        pred_stacked_cal[:, t_idx] = np.maximum(0.0, champ_cal[:, t_idx] + meta_learner.predict(Z_cal))
        pred_stacked_val[:, t_idx] = np.maximum(0.0, champ_val[:, t_idx] + meta_learner.predict(Z_val))
        pred_stacked_fut[:, t_idx] = np.maximum(0.0, champ_fut[:, t_idx] + meta_learner.predict(Z_fut))

    # Equation (27): Integer rounding for discrete Fire Count (target index 1)
    pred_stacked_cal[:, 1] = np.round(pred_stacked_cal[:, 1])
    pred_stacked_val[:, 1] = np.round(pred_stacked_val[:, 1])
    pred_stacked_fut[:, 1] = np.round(pred_stacked_fut[:, 1])

    m_res = calculate_metrics(y_val, pred_stacked_val)
    print(f"  Phase 6 ACID-FTS Stacked Holdout MAE: {m_res['MAE']:.4f}")

    metrics_rows = []
    for t_idx, t_col in enumerate(config.TARGET_COLS):
        metrics_rows.append({
            "Phase": "Phase_6_AdaptiveStacking",
            "Model": "ACID_FTS_Stacked",
            "Target": t_col,
            "Lag": lag,
            "MAE": m_res["MAE_Raw"][t_idx],
            "RMSE": m_res["RMSE_Raw"][t_idx],
            "LinEx": m_res["LinEx_Raw"][t_idx],
        })

    preds_map = {
        "Stacked": pred_stacked_val,
        "Cal_Stacked": pred_stacked_cal,
        "Future_Stacked": pred_stacked_fut,
    }
    return preds_map, pd.DataFrame(metrics_rows)

