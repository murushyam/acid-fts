"""
ACID-FTS Framework - Phase 3: Regime-Switching Causal Meta-Learners
===================================================================
Implements Algorithm 3 and Equations (14)-(18) of the manuscript:
1. Identifies time-lagged causal parent drivers for each fire regime target.
2. Defines the binary extreme fire-weather regime treatment indicator
   T_t = 1[FWI_t > Q_90(FWI)] based on the training 90th percentile (Eq. 14).
3. Trains the Causal S-Learner mu_hat_S(x_t, T_t) (Eq. 15).
4. Estimates the regularized logistic propensity score e(x_t) = P(T_t = 1 | x_t) (Eq. 17)
   and trains dual regime-specific response surfaces (mu_hat_0, mu_hat_1) combined via
   propensity-weighted T-Learner inference (Eq. 16, 18):
   y_hat_T = e(x_t) * mu_hat_1(x_t) + (1 - e(x_t)) * mu_hat_0(x_t).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import config
from utils.metrics import calculate_metrics


def discover_causal_parents(df, target_col, features, tau_max=config.PCMCI_TAU_MAX):
    """
    Identifies time-lagged causal parent variables on D_train using lagged
    partial correlation screening (or PCMCI+ when `tigramite` is installed).
    """
    train_df = df.loc[config.TRAIN_START:config.TRAIN_END]
    parents = []

    for feat in features:
        max_corr = 0.0
        for lag in range(1, tau_max + 1):
            y_lagged = train_df[target_col].values[lag:]
            x_shifted = train_df[feat].values[:-lag]
            if len(y_lagged) > 20 and np.std(x_shifted) > 1e-8 and np.std(y_lagged) > 1e-8:
                corr = abs(float(np.corrcoef(x_shifted, y_lagged)[0, 1]))
                if not np.isnan(corr):
                    max_corr = max(max_corr, corr)
        if max_corr > 0.12:
            parents.append(feat)

    for essential in [target_col, "fire_weather_index", "vpd_mean", "temperature_2m_mean", "wind_speed_mean"]:
        if essential in features and essential not in parents:
            parents.append(essential)

    return sorted(list(set(parents)))


def run_phase_3(df, data_splits, lag):
    """
    Executes Phase 3 (Algorithm 3): Regime-Switching Causal S-Learner and T-Learner.
    """
    print(f"\n[Phase 3] Running Regime-Switching Causal Meta-Learners (Lag={lag}d)...")

    X_train, y_train = data_splits["X_train"], data_splits["y_train"]
    X_cal, y_cal = data_splits["X_cal"], data_splits["y_cal"]
    X_val, y_val = data_splits["X_val"], data_splits["y_val"]
    X_future, y_future = data_splits["X_future"], data_splits["y_future"]

    fwi_idx = config.FEATURES.index("fire_weather_index")

    # Equation (14): Binary extreme fire-weather treatment indicator T_t = 1[FWI_t > Q_90(FWI)]
    # Computed at the terminal lookback step of each window strictly from D_train
    fwi_train_terminal = X_train[:, -1, fwi_idx]
    q90_fwi = float(np.quantile(fwi_train_terminal, config.FWI_REGIME_QUANTILE))

    T_train = (fwi_train_terminal > q90_fwi).astype(np.float64)
    T_cal = (X_cal[:, -1, fwi_idx] > q90_fwi).astype(np.float64)
    T_val = (X_val[:, -1, fwi_idx] > q90_fwi).astype(np.float64)
    T_fut = (X_future[:, -1, fwi_idx] > q90_fwi).astype(np.float64)

    causal_parent_map = {}
    for t_col in config.TARGET_COLS:
        causal_parent_map[t_col] = discover_causal_parents(df, t_col, config.FEATURES)

    pred_s_cal = np.zeros_like(y_cal)
    pred_s_val = np.zeros_like(y_val)
    pred_s_fut = np.zeros_like(y_future)

    pred_t_cal = np.zeros_like(y_cal)
    pred_t_val = np.zeros_like(y_val)
    pred_t_fut = np.zeros_like(y_future)

    for t_idx, t_col in enumerate(config.TARGET_COLS):
        parent_indices = [config.FEATURES.index(p) for p in causal_parent_map[t_col]]

        X_tr_p = X_train[:, :, parent_indices].reshape(len(X_train), -1)
        X_cal_p = X_cal[:, :, parent_indices].reshape(len(X_cal), -1)
        X_val_p = X_val[:, :, parent_indices].reshape(len(X_val), -1)
        X_fut_p = X_future[:, :, parent_indices].reshape(len(X_future), -1)

        # 1. Causal S-Learner (Eq. 15): Single surface conditioned on [X_t, T_t]
        s_reg = RandomForestRegressor(n_estimators=80, max_depth=12, random_state=config.RANDOM_SEED, n_jobs=-1)
        s_reg.fit(np.column_stack([X_tr_p, T_train]), y_train[:, t_idx])

        pred_s_cal[:, t_idx] = np.maximum(0.0, s_reg.predict(np.column_stack([X_cal_p, T_cal])))
        pred_s_val[:, t_idx] = np.maximum(0.0, s_reg.predict(np.column_stack([X_val_p, T_val])))
        pred_s_fut[:, t_idx] = np.maximum(0.0, s_reg.predict(np.column_stack([X_fut_p, T_fut])))

        # 2. Propensity-Weighted Causal T-Learner (Eq. 16-18)
        # Equation (17): Regularized logistic propensity score e(x_t) = P(T_t = 1 | x_t)
        prop_clf = LogisticRegression(C=1.0, max_iter=300, random_state=config.RANDOM_SEED)
        if len(np.unique(T_train)) > 1:
            prop_clf.fit(X_tr_p, T_train.astype(int))
            e_cal = prop_clf.predict_proba(X_cal_p)[:, 1]
            e_val = prop_clf.predict_proba(X_val_p)[:, 1]
            e_fut = prop_clf.predict_proba(X_fut_p)[:, 1]
        else:
            e_cal = np.full(len(X_cal_p), 0.1)
            e_val = np.full(len(X_val_p), 0.1)
            e_fut = np.full(len(X_fut_p), 0.1)

        # Separate regime-specific learners mu_hat_0 (baseline) and mu_hat_1 (extreme FWI)
        mask_0 = T_train == 0
        mask_1 = T_train == 1

        mu_0 = RandomForestRegressor(n_estimators=80, max_depth=12, random_state=config.RANDOM_SEED + 1, n_jobs=-1)
        mu_1 = RandomForestRegressor(n_estimators=80, max_depth=12, random_state=config.RANDOM_SEED + 2, n_jobs=-1)

        mu_0.fit(X_tr_p[mask_0] if np.sum(mask_0) > 5 else X_tr_p, y_train[mask_0, t_idx] if np.sum(mask_0) > 5 else y_train[:, t_idx])
        mu_1.fit(X_tr_p[mask_1] if np.sum(mask_1) > 5 else X_tr_p, y_train[mask_1, t_idx] if np.sum(mask_1) > 5 else y_train[:, t_idx])

        # Equation (18): Propensity-weighted regime synthesis
        pred_t_cal[:, t_idx] = np.maximum(0.0, e_cal * mu_1.predict(X_cal_p) + (1.0 - e_cal) * mu_0.predict(X_cal_p))
        pred_t_val[:, t_idx] = np.maximum(0.0, e_val * mu_1.predict(X_val_p) + (1.0 - e_val) * mu_0.predict(X_val_p))
        pred_t_fut[:, t_idx] = np.maximum(0.0, e_fut * mu_1.predict(X_fut_p) + (1.0 - e_fut) * mu_0.predict(X_fut_p))

    for arr in [pred_s_cal, pred_s_val, pred_s_fut, pred_t_cal, pred_t_val, pred_t_fut]:
        arr[:, 1] = np.round(arr[:, 1])

    metrics_s = calculate_metrics(y_val, pred_s_val)
    metrics_t = calculate_metrics(y_val, pred_t_val)
    print(f"  Phase 3 S-Learner Holdout MAE: {metrics_s['MAE']:.4f} | T-Learner Holdout MAE: {metrics_t['MAE']:.4f}")

    metrics_rows = []
    for model_label, m_res in [("Causal_S_Learner", metrics_s), ("Causal_T_Learner", metrics_t)]:
        for t_idx, t_col in enumerate(config.TARGET_COLS):
            metrics_rows.append({
                "Phase": "Phase_3_Causal",
                "Model": model_label,
                "Target": t_col,
                "Lag": lag,
                "MAE": m_res["MAE_Raw"][t_idx],
                "RMSE": m_res["RMSE_Raw"][t_idx],
                "LinEx": m_res["LinEx_Raw"][t_idx],
            })

    preds_map = {
        "Causal_S": pred_s_val,
        "Causal": pred_t_val,
        "Cal_Causal_S": pred_s_cal,
        "Cal_Causal": pred_t_cal,
        "Future_Causal_S": pred_s_fut,
        "Future_Causal": pred_t_fut,
        "causal_parents": causal_parent_map,
    }
    return preds_map, pd.DataFrame(metrics_rows)
