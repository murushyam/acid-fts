"""
PACI-ICPDA Framework - Phase 9: Statistical Validation
Conducts non-parametric hypothesis tests (Wilcoxon signed-rank test) to verify 
that the predictive improvements achieved across the different phases are statistically significant.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import config

def run_phase_9(y_val_true, p1_predictions, p6_predictions, lag):
    """
    Executes Phase 9: Non-parametric hypothesis significance testing.
    Compares the standard LSTM baseline against the Phase 6 Stacked champion.
    """
    print(f"\n[Phase 9] Running Statistical Validation (Lag={lag}d)...")
    
    pred_lstm = p1_predictions["LSTM"]
    pred_stacked = p6_predictions["Stacked"]
    
    n_targets = y_val_true.shape[1]
    
    validation_results = []
    
    for t_idx, t_col in enumerate(config.TARGET_COLS):
        true_col = y_val_true[:, t_idx]
        lstm_col = pred_lstm[:, t_idx]
        stacked_col = pred_stacked[:, t_idx]
        
        # Absolute errors
        error_lstm = np.abs(true_col - lstm_col)
        error_stacked = np.abs(true_col - stacked_col)
        
        # Compute differences in absolute errors
        diff = error_lstm - error_stacked
        
        # Check if errors are identical (Wilcoxon requires non-zero differences)
        if np.all(diff == 0):
            stat, p_val = 0.0, 1.0
            sig = False
        else:
            try:
                # Wilcoxon signed-rank test (two-sided)
                stat, p_val = wilcoxon(error_lstm, error_stacked, alternative="two-sided")
                sig = p_val < 0.05
            except Exception as e:
                print(f"    [Warning] Wilcoxon test failed for {t_col} ({e}).")
                stat, p_val = 0.0, 1.0
                sig = False
                
        # Mean Absolute Errors
        mae_lstm = np.mean(error_lstm)
        mae_stacked = np.mean(error_stacked)
        improvement = (mae_lstm - mae_stacked) / mae_lstm * 100 if mae_lstm > 0 else 0.0
        
        validation_results.append({
            "Target": t_col,
            "Lag": lag,
            "LSTM_MAE": mae_lstm,
            "Stacked_MAE": mae_stacked,
            "Improvement_%": improvement,
            "Wilcoxon_Stat": stat,
            "p-value": p_val,
            "Significant_p<0.05": sig
        })
        
        print(f"    {t_col}: LSTM MAE={mae_lstm:.4f} | Stacked MAE={mae_stacked:.4f} | Improvement={improvement:.2f}% | p-val={p_val:.4e} (Sig: {sig})")
        
    df_val = pd.DataFrame(validation_results)
    
    out_dir = os.path.join(config.OUTPUT_DIR, f"LAG_{lag}", "Phase_9_Validation")
    os.makedirs(out_dir, exist_ok=True)
    df_val.to_csv(os.path.join(out_dir, "wilcoxon_significance_results.csv"), index=False)
    
    # Save LaTeX table code snippet for IEEE manuscript insertion
    latex_table = df_val.to_latex(index=False, float_format="%.4g")
    with open(os.path.join(out_dir, "wilcoxon_table.tex"), "w") as f:
        f.write(latex_table)
        
    print(f"  ✅ Statistical validation reports saved to {out_dir}")
    return df_val
