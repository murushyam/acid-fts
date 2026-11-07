"""
ACID-FTS Framework - Phase 5: Unified Multi-Target Candidate & Dynamic Best Model Routing
=========================================================================================
Implements Algorithm 5 and Equations (22)-(23) of the manuscript:
1. Evaluates the unified multi-target joint model (P5_joint) across all three fire
   regime variables simultaneously.
2. Evaluates all candidate architectures M in {LR, RF, LSTM, TFT, Ensemble, FuzzyVMD,
   Causal_S, Causal, CausalAttention, P5_joint} strictly on the calibration split D_cal.
3. Selects the target-and-horizon specific champion model via the routing operator:
   M*(Y, L) = argmin_{M in M} RMSE_cal(M, Y, L) (Eq. 22).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import config
from utils.metrics import calculate_metrics


def run_phase_5(data_splits, all_preds, lag):
    """
    Executes Phase 5 (Algorithm 5): Evaluates P5_joint and performs zero-leakage
    Dynamic Best Model Routing M*(Y, L) using D_cal RMSE.
    """
    print(f"\n[Phase 5] Running Unified Multi-Target Candidate & Dynamic Routing (Lag={lag}d)...")

    X_train, y_train = data_splits["X_train"], data_splits["y_train"]
    X_cal, y_cal = data_splits["X_cal"], data_splits["y_cal"]
    X_val, y_val = data_splits["X_val"], data_splits["y_val"]
    X_future, y_future = data_splits["X_future"], data_splits["y_future"]

    X_tr_flat = X_train.reshape(len(X_train), -1)
    X_cal_flat = X_cal.reshape(len(X_cal), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_fut_flat = X_future.reshape(len(X_future), -1)

    # 1. Unified Multi-Target Candidate (P5_joint): Simultaneous 3-output regressor
    joint_model = ExtraTreesRegressor(n_estimators=100, max_depth=14, random_state=config.RANDOM_SEED, n_jobs=-1)
    joint_model.fit(X_tr_flat, y_train)

    pred_joint_cal = np.maximum(0.0, joint_model.predict(X_cal_flat))
    pred_joint_val = np.maximum(0.0, joint_model.predict(X_val_flat))
    pred_joint_fut = np.maximum(0.0, joint_model.predict(X_fut_flat))

    for arr in [pred_joint_cal, pred_joint_val, pred_joint_fut]:
        arr[:, 1] = np.round(arr[:, 1])

    all_preds["P5_joint"] = pred_joint_val
    all_preds["Cal_P5_joint"] = pred_joint_cal
    all_preds["Future_P5_joint"] = pred_joint_fut

    # 2. Dynamic Best Model Routing M*(Y, L) selected strictly on D_cal (Eq. 22)
    candidate_names = [
        "LR",
        "RF",
        "LSTM",
        "TFT",
        "Ensemble",
        "FuzzyVMD",
        "Causal_S",
        "Causal",
        "CausalAttention",
        "P5_joint",
    ]

    champion_cal = np.zeros_like(y_cal)
    champion_val = np.zeros_like(y_val)
    champion_fut = np.zeros_like(y_future)
    routing_selection = {}

    for t_idx, t_col in enumerate(config.TARGET_COLS):
        best_rmse_cal = float("inf")
        best_name = "RF"

        for name in candidate_names:
            cal_key = f"Cal_{name}"
            if cal_key in all_preds and name in all_preds:
                rmse_cal = float(np.sqrt(np.mean((y_cal[:, t_idx] - all_preds[cal_key][:, t_idx]) ** 2)))
                if rmse_cal < best_rmse_cal:
                    best_rmse_cal = rmse_cal
                    best_name = name

        routing_selection[t_col] = best_name
        champion_cal[:, t_idx] = all_preds[f"Cal_{best_name}"][:, t_idx]
        champion_val[:, t_idx] = all_preds[best_name][:, t_idx]
        champion_fut[:, t_idx] = all_preds[f"Future_{best_name}"][:, t_idx]
        print(f"  Routed Champion M*({t_col}, Lag={lag}d): {best_name} (Cal RMSE={best_rmse_cal:.4f})")

    m_joint = calculate_metrics(y_val, pred_joint_val)
    m_champ = calculate_metrics(y_val, champion_val)

    metrics_rows = []
    for model_label, m_res in [("P5_Joint_Candidate", m_joint), ("P5_Dynamic_Routed_M*", m_champ)]:
        for t_idx, t_col in enumerate(config.TARGET_COLS):
            metrics_rows.append({
                "Phase": "Phase_5_DynamicRouting",
                "Model": model_label,
                "Target": t_col,
                "Lag": lag,
                "MAE": m_res["MAE_Raw"][t_idx],
                "RMSE": m_res["RMSE_Raw"][t_idx],
                "LinEx": m_res["LinEx_Raw"][t_idx],
            })

    preds_map = {
        "P5_joint": pred_joint_val,
        "Champion": champion_val,
        "Cal_Champion": champion_cal,
        "Future_Champion": champion_fut,
        "routing_selection": routing_selection,
    }
    return preds_map, pd.DataFrame(metrics_rows)

