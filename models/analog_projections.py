"""
PACI-ICPDA Framework - Phase 8: Analog Projections
Extends the calibrated uncertainty model to generate future 3-year (1095-day)
probabilistic wildfire projections using a climatological analog forecasting approach.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from utils.plotting import plot_conformal_intervals

def run_phase_8(data_splits, p6_predictions, p7_intervals, lag):
    """
    Executes Phase 8: Generates future projections and saves publication-grade plots.
    """
    print(f"\n[Phase 8] Running 1095-Day Future Analog Projections (Lag={lag}d)...")
    
    dates_future = data_splits["dates_future"]
    fut_pred = p6_predictions["Future_Stacked"]
    fut_lower = p7_intervals["future_lower"]
    fut_upper = p7_intervals["future_upper"]
    
    # Trim inputs to exactly 1095 days for the 3-year projection target if needed,
    # or use the full extended future dates sequence.
    proj_len = min(1095, len(dates_future), len(fut_pred))
    
    dates_proj = dates_future[:proj_len]
    pred_proj = fut_pred[:proj_len]
    lower_proj = fut_lower[:proj_len]
    upper_proj = fut_upper[:proj_len]
    
    print(f"  Future Projection Period: {dates_proj[0].date()} to {dates_proj[-1].date()} ({proj_len} days)")
    
    out_dir = os.path.join(config.OUTPUT_DIR, f"LAG_{lag}", "Phase_8_Projections")
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "proj_dates.npy"), np.array([str(d.date()) for d in dates_proj]))
    np.save(os.path.join(out_dir, "proj_pred.npy"), pred_proj)
    np.save(os.path.join(out_dir, "proj_lower.npy"), lower_proj)
    np.save(os.path.join(out_dir, "proj_upper.npy"), upper_proj)
    
    # Generate publication-grade fan-charts for the future period
    for t_idx, t_col in enumerate(config.TARGET_COLS):
        # Use plot utility
        plot_conformal_intervals(
            dates_proj, 
            pred_proj[:, t_idx],  # Draw predictions twice to show center line
            pred_proj[:, t_idx],
            lower_proj[:, t_idx],
            upper_proj[:, t_idx],
            target_col=t_col,
            filename=f"Future_FanChart_{t_col}.png",
            subfolder=f"LAG_{lag}/Phase_8_Projections"
        )
        
    print(f"  ✅ Publication future fancharts saved to {out_dir}")
    
    return {
        "dates_proj": dates_proj,
        "pred_proj": pred_proj,
        "lower_proj": lower_proj,
        "upper_proj": upper_proj
    }
