"""
PACI-ICPDA Framework - Phase 7: Probabilistic Interval Calibration
Constructs distribution-free calibrated prediction intervals around point forecasts
using split-conformal prediction principles to ensure valid empirical coverage.
"""

import os
import numpy as np
import pandas as pd
import config
from utils.metrics import calculate_metrics

def calibrate_conformal_intervals(y_val_true, y_val_pred, y_future_pred, alpha=0.05):
    """
    Computes conformal calibration margins based on validation residuals.
    
    Args:
        y_val_true (np.ndarray): Validation observations (Samples, Targets)
        y_val_pred (np.ndarray): Validation predictions (Samples, Targets)
        y_future_pred (np.ndarray): Future predictions (Samples, Targets)
        alpha (float): Significance level (e.g. 0.05 for 95% intervals)
        
    Returns:
        dict: Calibrated prediction intervals for validation and future periods.
    """
    n_targets = y_val_true.shape[1]
    
    val_lower = np.zeros_like(y_val_pred)
    val_upper = np.zeros_like(y_val_pred)
    future_lower = np.zeros_like(y_future_pred)
    future_upper = np.zeros_like(y_future_pred)
    
    calibrated_margins = []
    
    for t_idx in range(n_targets):
        # 1. Compute non-conformity scores (absolute prediction residuals)
        scores = np.abs(y_val_true[:, t_idx] - y_val_pred[:, t_idx])
        
        # 2. Calibrate margin at (1 - alpha) quantile (adjusted for sample size inflation)
        n = len(scores)
        q_value = (1.0 - alpha) * (1.0 + 1.0 / n)
        q_value = min(q_value, 0.999)  # Cap at 1
        
        margin = np.quantile(scores, q_value)
        calibrated_margins.append(margin)
        
        # 3. Construct symmetric intervals
        val_lower[:, t_idx] = np.maximum(y_val_pred[:, t_idx] - margin, 0.0)
        val_upper[:, t_idx] = y_val_pred[:, t_idx] + margin
        
        future_lower[:, t_idx] = np.maximum(y_future_pred[:, t_idx] - margin, 0.0)
        future_upper[:, t_idx] = y_future_pred[:, t_idx] + margin
        
    # Integer rounding for fire counts bounds
    val_lower[:, 1] = np.round(val_lower[:, 1])
    val_upper[:, 1] = np.round(val_upper[:, 1])
    future_lower[:, 1] = np.round(future_lower[:, 1])
    future_upper[:, 1] = np.round(future_upper[:, 1])
    
    intervals = {
        "val_lower": val_lower,
        "val_upper": val_upper,
        "future_lower": future_lower,
        "future_upper": future_upper,
        "margins": calibrated_margins
    }
    return intervals

def run_phase_7(y_val_true, p6_predictions, lag):
    """
    Executes Phase 7: Generates conformal calibrated confidence intervals.
    """
    print(f"\n[Phase 7] Running Conformal Prediction Interval Calibration (Lag={lag}d)...")
    
    pred_stacked_val = p6_predictions["Stacked"]
    fut_stacked_fut = p6_predictions["Future_Stacked"]
    
    intervals = calibrate_conformal_intervals(
        y_val_true, pred_stacked_val, fut_stacked_fut, alpha=config.CONFORMAL_ALPHA
    )
    
    # Save outputs
    out_dir = os.path.join(config.OUTPUT_DIR, f"LAG_{lag}", "Phase_7_Conformal")
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "val_lower.npy"), intervals["val_lower"])
    np.save(os.path.join(out_dir, "val_upper.npy"), intervals["val_upper"])
    np.save(os.path.join(out_dir, "fut_lower.npy"), intervals["future_lower"])
    np.save(os.path.join(out_dir, "fut_upper.npy"), intervals["future_upper"])
    
    # Print empirical coverage check (validation set)
    print("  Conformal Calibration Complete:")
    for t_idx, t_col in enumerate(config.TARGET_COLS):
        margin = intervals["margins"][t_idx]
        true_vals = y_val_true[:, t_idx]
        lower_vals = intervals["val_lower"][:, t_idx]
        upper_vals = intervals["val_upper"][:, t_idx]
        
        # Coverage fraction: percent of points inside the bounds
        coverage = np.mean((true_vals >= lower_vals) & (true_vals <= upper_vals)) * 100
        print(f"    {t_col}: Margin={margin:.2f} | Empirical Coverage={coverage:.2f}% (Target={(1-config.CONFORMAL_ALPHA)*100}%)")
        
    return intervals
