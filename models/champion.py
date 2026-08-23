"""
PACI-ICPDA Framework - Phase 5: Target-Specific Champion Assembly
Performs automated model selection by evaluating candidate models on the validation
set and selecting the best configuration (the champion) for each target variable.
"""

import os
import numpy as np
import pandas as pd
import config
from utils.metrics import calculate_metrics

def run_phase_5(y_val, y_future, all_val_preds, all_fut_preds, lag):
    """
    Executes Phase 5: Auto-selects the champion model for each target variable.
    """
    print(f"\n[Phase 5] Running Target-Specific Champion Assembly (Lag={lag}d)...")
    
    n_targets = y_val.shape[1]
    
    # Models to compare
    candidate_names = ["LR", "RF", "LSTM", "Ensemble", "FuzzyVMD", "Causal", "CausalAttention"]
    
    champion_pred_val = np.zeros_like(y_val)
    champion_pred_fut = np.zeros_like(y_future)
    champion_names = []
    
    # Select best model for each target variable independently
    for t_idx, t_col in enumerate(config.TARGET_COLS):
        best_mae = float("inf")
        best_model_name = None
        best_val = None
        best_fut = None
        
        for name in candidate_names:
            if name in all_val_preds:
                pred_val_col = all_val_preds[name][:, t_idx]
                mae = np.mean(np.abs(y_val[:, t_idx] - pred_val_col))
                
                if mae < best_mae:
                    best_mae = mae
                    best_model_name = name
                    best_val = all_val_preds[name][:, t_idx]
                    best_fut = all_fut_preds[f"Future_{name}"][:, t_idx]
                    
        champion_names.append(best_model_name)
        champion_pred_val[:, t_idx] = best_val
        champion_pred_fut[:, t_idx] = best_fut
        print(f"  Champion for {t_col}: {best_model_name} (MAE={best_mae:.4f})")
        
    # Calculate performance metrics
    metrics_champion = calculate_metrics(y_val, champion_pred_val)
    
    # Save outputs
    out_dir = os.path.join(config.OUTPUT_DIR, f"LAG_{lag}", "Phase_5_Champion")
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "val_pred_Champion.npy"), champion_pred_val)
    np.save(os.path.join(out_dir, "fut_pred_Champion.npy"), champion_pred_fut)
    
    # Save selection info
    selection_info = {t_col: name for t_col, name in zip(config.TARGET_COLS, champion_names)}
    import json
    with open(os.path.join(out_dir, "champion_models.json"), "w") as f:
        json.dump(selection_info, f, indent=4)
        
    metrics_summary = []
    for t_idx, t_col in enumerate(config.TARGET_COLS):
        metrics_summary.append({
            "Phase": "Phase_5_Champion",
            "Model": f"Champion_{selection_info[t_col]}",
            "Target": t_col,
            "Lag": lag,
            "MAE": metrics_champion["MAE_Raw"][t_idx],
            "RMSE": metrics_champion["RMSE_Raw"][t_idx]
        })
        
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    
    preds_map = {
        "Champion": champion_pred_val,
        "Future_Champion": champion_pred_fut,
        "champion_names": selection_info
    }
    
    return preds_map, df_metrics
