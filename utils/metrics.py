"""
PACI-ICPDA Framework - Metrics Calculator
Computes forecasting performance metrics (MAE, RMSE, MAPE) for target variables.
"""

import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Computes absolute and relative metrics between predictions and ground truths.
    
    Args:
        y_true (np.ndarray): Ground truth target array (Samples, Targets)
        y_pred (np.ndarray): Model forecast array (Samples, Targets)
        
    Returns:
        dict: Summary of global and target-specific metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle single-target inputs
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        
    n_samples, n_targets = y_true.shape
    
    # Target-specific lists
    rmse_list = []
    mae_list = []
    mape_list = []
    
    for i in range(n_targets):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((true_col - pred_col) ** 2))
        rmse_list.append(rmse)
        
        # Calculate MAE
        mae = np.mean(np.abs(true_col - pred_col))
        mae_list.append(mae)
        
        # Calculate MAPE (handling zero-division)
        mask = true_col != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((true_col[mask] - pred_col[mask]) / true_col[mask])) * 100
        else:
            mape = 0.0
        mape_list.append(mape)
        
    global_rmse = np.mean(rmse_list)
    global_mae = np.mean(mae_list)
    global_mape = np.mean(mape_list)
    
    results = {
        "RMSE": global_rmse,
        "MAE": global_mae,
        "MAPE": global_mape,
        "RMSE_Raw": rmse_list,
        "MAE_Raw": mae_list,
        "MAPE_Raw": mape_list
    }
    return results
