"""
PACI-ICPDA Framework - Phase 1: Baseline Ensemble Prediction
Establishes baseline performance bounds using classical statistical methods,
machine learning regressors, and deep learning architectures to build a multi-model benchmark.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
import config
from utils.metrics import calculate_metrics

def train_lstm(X_train, y_train, X_val, y_val, lag, n_features, n_targets):
    """Trains a simple deep learning LSTM model."""
    inp = Input(shape=(lag, n_features))
    x = LSTM(32, activation="relu", return_sequences=False)(inp)
    x = Dense(16, activation="relu")(x)
    out = Dense(n_targets)(x)
    
    model = Model(inp, out)
    model.compile(optimizer=Adam(learning_rate=0.005), loss="mse")
    
    # Train with early validation monitoring
    model.fit(
        X_train, y_train, 
        epochs=config.LSTM_EPOCHS, 
        batch_size=config.BATCH_SIZE, 
        validation_data=(X_val, y_val),
        verbose=0
    )
    return model

def run_phase_1(data_splits, lag):
    """
    Executes Phase 1: Establishes base predictions and computes metrics.
    """
    print(f"\n[Phase 1] Running Baseline Models (Lag={lag}d)...")
    
    X_train = data_splits["X_train"]
    y_train = data_splits["y_train"]
    X_val = data_splits["X_val"]
    y_val = data_splits["y_val"]
    X_future = data_splits["X_future"]
    
    n_samples_train, lag_steps, n_features = X_train.shape
    n_targets = y_train.shape[1]
    
    # Flatten inputs for non-sequential classical ML models
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_future_flat = X_future.reshape(len(X_future), -1)
    
    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_flat, y_train)
    pred_lr = lr.predict(X_val_flat)
    fut_lr = lr.predict(X_future_flat)
    
    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, y_train)
    pred_rf = rf.predict(X_val_flat)
    fut_rf = rf.predict(X_future_flat)
    
    # 3. Deep Learning LSTM
    lstm = train_lstm(X_train, y_train, X_val, y_val, lag, n_features, n_targets)
    pred_lstm = lstm.predict(X_val)
    fut_lstm = lstm.predict(X_future)
    
    # Ensure physical limits (predictions should not be negative)
    pred_lr = np.maximum(pred_lr, 0.0)
    pred_rf = np.maximum(pred_rf, 0.0)
    pred_lstm = np.maximum(pred_lstm, 0.0)
    
    fut_lr = np.maximum(fut_lr, 0.0)
    fut_rf = np.maximum(fut_rf, 0.0)
    fut_lstm = np.maximum(fut_lstm, 0.0)
    
    # Integer rounding for Fire Counts (assumed index 1 in TARGET_COLS)
    for preds in [pred_lr, pred_rf, pred_lstm, fut_lr, fut_rf, fut_lstm]:
        preds[:, 1] = np.round(preds[:, 1])
        
    # 4. Multi-Model Ensemble Average
    pred_ens = (pred_lr + pred_rf + pred_lstm) / 3.0
    fut_ens = (fut_lr + fut_rf + fut_lstm) / 3.0
    pred_ens[:, 1] = np.round(pred_ens[:, 1])
    fut_ens[:, 1] = np.round(fut_ens[:, 1])
    
    # Compile performance reports
    metrics_lr = calculate_metrics(y_val, pred_lr)
    metrics_rf = calculate_metrics(y_val, pred_rf)
    metrics_lstm = calculate_metrics(y_val, pred_lstm)
    metrics_ens = calculate_metrics(y_val, pred_ens)
    
    print(f"  Linear Regression MAE: {metrics_lr['MAE']:.4f}")
    print(f"  Random Forest MAE:     {metrics_rf['MAE']:.4f}")
    print(f"  Deep LSTM MAE:         {metrics_lstm['MAE']:.4f}")
    print(f"  Baseline Ensemble MAE: {metrics_ens['MAE']:.4f}")
    
    # Save outputs
    out_dir = os.path.join(config.OUTPUT_DIR, f"LAG_{lag}", "Phase_1_Baselines")
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "val_pred_LR.npy"), pred_lr)
    np.save(os.path.join(out_dir, "val_pred_RF.npy"), pred_rf)
    np.save(os.path.join(out_dir, "val_pred_LSTM.npy"), pred_lstm)
    np.save(os.path.join(out_dir, "val_pred_Ensemble.npy"), pred_ens)
    
    np.save(os.path.join(out_dir, "fut_pred_Ensemble.npy"), fut_ens)
    
    # Return predictions map and metrics
    preds_map = {
        "LR": pred_lr,
        "RF": pred_rf,
        "LSTM": pred_lstm,
        "Ensemble": pred_ens,
        "Future_Ensemble": fut_ens
    }
    
    metrics_summary = []
    for model_name, met in [("LR", metrics_lr), ("RF", metrics_rf), ("LSTM", metrics_lstm), ("Ensemble", metrics_ens)]:
        for t_idx, t_col in enumerate(config.TARGET_COLS):
            metrics_summary.append({
                "Phase": "Phase_1_Baseline",
                "Model": model_name,
                "Target": t_col,
                "Lag": lag,
                "MAE": met["MAE_Raw"][t_idx],
                "RMSE": met["RMSE_Raw"][t_idx]
            })
            
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    
    return preds_map, df_metrics
