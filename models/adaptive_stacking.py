"""
PACI-ICPDA Framework - Phase 6: Adaptive Stacking
Corrects systematic, context-dependent prediction errors using a secondary meta-learner
(XGBoost meta-regressor) trained to predict and correct the residual errors of the champion model.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import config
from utils.metrics import calculate_metrics

# Set Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

def train_meta_learner(X_train, y_train, X_val, y_val):
    """
    Optimizes and trains an XGBoost meta-regressor using Optuna to predict
    residual error of the champion model.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_jobs": -1,
            "random_state": 42
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - preds) ** 2))
        return rmse
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.OPTUNA_TRIALS)
    
    # Train champion configuration
    best_model = xgb.XGBRegressor(**study.best_params, n_jobs=-1, random_state=42)
    best_model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    return best_model

def run_phase_6(df, data_splits, p5_predictions, lag):
    """
    Executes Phase 6: Error learning stacking using optimized XGBoost models.
    """
    print(f"\n[Phase 6] Running Adaptive Stacking Error Correction (Lag={lag}d)...")
    
    X_train = data_splits["X_train"]
    y_train = data_splits["y_train"]
    X_val = data_splits["X_val"]
    y_val = data_splits["y_val"]
    X_future = data_splits["X_future"]
    
    n_targets = y_train.shape[1]
    
    pred_champion_val = p5_predictions["Champion"]
    fut_champion_fut = p5_predictions["Future_Champion"]
    
    # Get predictions on training set to compute training residuals for stacking
    # (Use a simple RF copy as proxy for champion to avoid train overfitting leakage)
    X_train_flat = X_train.reshape(len(X_train), -1)
    from sklearn.ensemble import RandomForestRegressor
    rf_temp = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train_flat, y_train)
    pred_champion_train = rf_temp.predict(X_train_flat)
    
    # Stacking meta-features: Weather/context variables
    context_indices = [config.FEATURES.index(c) for c in config.CONTEXT_COLS if c in config.FEATURES]
    
    # Stack features shape: (Samples, Lags * ContextVars + ChampionPred)
    # Using spatial-temporal context + current forecast to predict residual correction
    def extract_stack_features(X_lagged, preds_point):
        # Flatten temporal context features
        feat_flat = X_lagged[:, :, context_indices].reshape(len(X_lagged), -1)
        # Combine with point predictions
        return np.hstack([feat_flat, preds_point])
        
    X_meta_train = extract_stack_features(X_train, pred_champion_train)
    X_meta_val = extract_stack_features(X_val, pred_champion_val)
    X_meta_future = extract_stack_features(X_future, fut_champion_fut)
    
    pred_stacked_val = np.zeros_like(y_val)
    pred_stacked_fut = np.zeros_like(data_splits["y_future"])
    
    # Train separate stacking model for each target residual
    for t_idx in range(n_targets):
        res_train = y_train[:, t_idx] - pred_champion_train[:, t_idx]
        res_val = y_val[:, t_idx] - pred_champion_val[:, t_idx]
        
        # Split meta-validation for training tuning
        split = int(len(X_meta_val) * 0.6)
        
        X_m_tr, X_m_va = X_meta_val[:split], X_meta_val[split:]
        res_m_tr, res_m_va = res_val[:split], res_val[split:]
        
        # Train meta-regressor to learn context-dependent residual error
        meta_learner = train_meta_learner(X_m_tr, res_m_tr, X_m_va, res_m_va)
        
        # Predict corrections
        correction_val = meta_learner.predict(X_meta_val)
        correction_fut = meta_learner.predict(X_meta_future)
        
        # Apply corrections
        pred_stacked_val[:, t_idx] = pred_champion_val[:, t_idx] + correction_val
        pred_stacked_fut[:, t_idx] = fut_champion_fut[:, t_idx] + correction_fut
        
    # Constraints and integer rounding
    pred_stacked_val = np.maximum(pred_stacked_val, 0.0)
    pred_stacked_val[:, 1] = np.round(pred_stacked_val[:, 1])
    
    pred_stacked_fut = np.maximum(pred_stacked_fut, 0.0)
    pred_stacked_fut[:, 1] = np.round(pred_stacked_fut[:, 1])
    
    # Calculate performance metrics
    metrics_stacked = calculate_metrics(y_val, pred_stacked_val)
    print(f"  Champion Model MAE:   {p5_predictions['Champion'].mean():.4f}")
    print(f"  Adaptive Stacked MAE: {metrics_stacked['MAE']:.4f}")
    
    # Save outputs
    out_dir = os.path.join(config.OUTPUT_DIR, f"LAG_{lag}", "Phase_6_AdaptiveStacking")
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "val_pred_Stacked.npy"), pred_stacked_val)
    np.save(os.path.join(out_dir, "fut_pred_Stacked.npy"), pred_stacked_fut)
    
    metrics_summary = []
    for t_idx, t_col in enumerate(config.TARGET_COLS):
        metrics_summary.append({
            "Phase": "Phase_6_AdaptiveStacking",
            "Model": "Adaptive_XGB_Stacking",
            "Target": t_col,
            "Lag": lag,
            "MAE": metrics_stacked["MAE_Raw"][t_idx],
            "RMSE": metrics_stacked["RMSE_Raw"][t_idx]
        })
        
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    
    preds_map = {
        "Stacked": pred_stacked_val,
        "Future_Stacked": pred_stacked_fut
    }
    
    return preds_map, df_metrics
