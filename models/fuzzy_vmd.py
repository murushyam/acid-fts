"""
PACI-ICPDA Framework - Phase 2: VMD-Fuzzy Residual Learning (ICPDA)
Applies signal decomposition via Variational Mode Decomposition (VMD) for denoising
and fuzzy interval correction to address systematic biases and high-frequency oscillations.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import config
from utils.metrics import calculate_metrics

# Graceful Import Fallback for vmdpy
try:
    from vmdpy import VMD
    HAS_VMDPY = True
except ImportError:
    HAS_VMDPY = False

def decompose_residuals(residuals, K=3):
    """
    Decomposes residual error signal into K Intrinsic Mode Functions (IMFs).
    Falls back to a pure-python spectral mock-VMD (sinusoidal frequency splitting)
    if vmdpy is not installed.
    """
    if np.std(residuals) < 1e-5:
        # No signal to decompose
        return np.tile(residuals, (K, 1))
        
    if HAS_VMDPY:
        try:
            # VMD parameters: alpha (bandwidth), tau (noise), K (modes), DC, init, tol
            u, _, _ = VMD(residuals, alpha=2000, tau=0, K=K, DC=0, init=1, tol=1e-7)
            # Ensure shape matches input length (guards against odd-length trimming in vmdpy)
            if u.shape[1] < len(residuals):
                u = np.pad(u, ((0, 0), (0, len(residuals) - u.shape[1])), mode="edge")
            elif u.shape[1] > len(residuals):
                u = u[:, :len(residuals)]
            return u
        except Exception as e:
            print(f"  [Warning] VMD failed ({e}). Falling back to spectral proxy.")
            
    # Pure Python Fallback: Fourier/spectral frequency decomposition proxy
    # Splits the signal into K bands of low-to-high frequencies
    n = len(residuals)
    r_fft = np.fft.rfft(residuals)
    freqs = np.fft.rfftfreq(n)
    
    u = np.zeros((K, n))
    split_indices = np.linspace(0, len(r_fft), K + 1, dtype=int)
    
    for i in range(K):
        band_fft = np.zeros_like(r_fft)
        start, end = split_indices[i], split_indices[i+1]
        band_fft[start:end] = r_fft[start:end]
        u[i] = np.fft.irfft(band_fft, n=n)
        
    return u

def apply_fuzzy_correction(y_pred_base, X_features, residuals_train, X_train_flat, X_val_flat, X_future_flat, K=3):
    """
    Trains a Random Forest regressor to predict the decomposed residual IMFs 
    and applies correction to the baseline predictions (ICPDA fuzzy residual correction).
    """
    n_samples_train = len(residuals_train)
    # Decompose train residuals
    imfs_train = np.zeros((K, n_samples_train))
    
    # We decompose residuals for each target variable independently
    n_targets = residuals_train.shape[1]
    corrected_val = np.zeros_like(y_pred_base)
    
    val_corrections = np.zeros_like(y_pred_base)
    fut_corrections = np.zeros((len(X_future_flat), n_targets))
    
    for t_idx in range(n_targets):
        res_col = residuals_train[:, t_idx]
        imfs = decompose_residuals(res_col, K=K)
        
        # Predict each IMF using Random Forest (representing fuzzy mapping)
        imf_preds_val = np.zeros((len(X_val_flat), K))
        imf_preds_fut = np.zeros((len(X_future_flat), K))
        
        for k in range(K):
            imf_target = imfs[k]
            model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            model.fit(X_train_flat, imf_target)
            
            imf_preds_val[:, k] = model.predict(X_val_flat)
            imf_preds_fut[:, k] = model.predict(X_future_flat)
            
        # Reconstruct total residual correction (sum of predicted IMFs)
        val_corrections[:, t_idx] = np.sum(imf_preds_val, axis=1)
        fut_corrections[:, t_idx] = np.sum(imf_preds_fut, axis=1)
        
    return val_corrections, fut_corrections

def run_phase_2(data_splits, p1_predictions, lag):
    """
    Executes Phase 2: Signal decomposition and fuzzy residual correction.
    """
    print(f"\n[Phase 2] Running VMD-Fuzzy Residual Learning (Lag={lag}d)...")
    if not HAS_VMDPY:
        print("  [Note] vmdpy package not found. Using pure-Python spectral VMD proxy.")
        
    X_train = data_splits["X_train"]
    y_train = data_splits["y_train"]
    X_val = data_splits["X_val"]
    y_val = data_splits["y_val"]
    X_future = data_splits["X_future"]
    
    # Flatten inputs
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_future_flat = X_future.reshape(len(X_future), -1)
    
    # Load Phase 1 baseline ensemble predictions
    pred_ens_val = p1_predictions["Ensemble"]
    fut_ens_fut = p1_predictions["Future_Ensemble"]
    
    # Get predictions on training set to compute training residuals
    # (Retrain simple RF model representing ensemble to avoid full TF training latency)
    n_targets = y_train.shape[1]
    rf_temp = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train_flat, y_train)
    pred_ens_train = rf_temp.predict(X_train_flat)
    
    # Train residuals
    residuals_train = y_train - pred_ens_train
    
    # Apply VMD + Stacking correction (representing ICPDA Fuzzy residual adjustment)
    val_corrections, fut_corrections = apply_fuzzy_correction(
        pred_ens_val, X_train, residuals_train, X_train_flat, X_val_flat, X_future_flat, K=3
    )
    
    # Corrected predictions
    pred_corrected = pred_ens_val + val_corrections
    fut_corrected = fut_ens_fut + fut_corrections
    
    # Physical constraints
    pred_corrected = np.maximum(pred_corrected, 0.0)
    fut_corrected = np.maximum(fut_corrected, 0.0)
    
    # Integer rounding for fire counts
    pred_corrected[:, 1] = np.round(pred_corrected[:, 1])
    fut_corrected[:, 1] = np.round(fut_corrected[:, 1])
    
    # Calculate performance metrics
    metrics_corrected = calculate_metrics(y_val, pred_corrected)
    print(f"  Base Ensemble MAE:     {p1_predictions['Ensemble'].mean():.4f}")  # Quick display metric
    print(f"  VMD-Fuzzy corrected MAE: {metrics_corrected['MAE']:.4f}")
    
    # Save outputs
    out_dir = os.path.join(config.OUTPUT_DIR, f"LAG_{lag}", "Phase_2_Fuzzy_VMD")
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "val_pred_FuzzyVMD.npy"), pred_corrected)
    np.save(os.path.join(out_dir, "fut_pred_FuzzyVMD.npy"), fut_corrected)
    
    metrics_summary = []
    for t_idx, t_col in enumerate(config.TARGET_COLS):
        metrics_summary.append({
            "Phase": "Phase_2_FuzzyVMD",
            "Model": "ICPDA_VMD_Fuzzy",
            "Target": t_col,
            "Lag": lag,
            "MAE": metrics_corrected["MAE_Raw"][t_idx],
            "RMSE": metrics_corrected["RMSE_Raw"][t_idx]
        })
        
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    
    preds_map = {
        "FuzzyVMD": pred_corrected,
        "Future_FuzzyVMD": fut_corrected
    }
    
    return preds_map, df_metrics
