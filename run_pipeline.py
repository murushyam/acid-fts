"""
PACI-ICPDA Framework - Main Master Pipeline Orchestrator
Coordinates the nine phases of the wildfire forecasting and validation pipeline.

Runs:
  Phase 1: Baseline Ensemble Prediction
  Phase 2: VMD-Fuzzy Residual Learning (ICPDA)
  Phase 3: Causal Discovery & Meta-Learners
  Phase 4: Deep Causal Attention LSTM Hybrid
  Phase 5: Target-Specific Champion Assembly
  Phase 6: Adaptive Stacking (Meta-XGBoost)
  Phase 7: Probabilistic Interval Calibration
  Phase 8: Conformal analog forecasting and 1095-day projections
  Phase 9: Wilcoxon signed-rank statistical validation
"""

import os
import sys
import numpy as np
import pandas as pd
import config
from data_loader import load_dataset, prepare_lagged_data, get_temporal_splits
from models.baseline import run_phase_1
from models.fuzzy_vmd import run_phase_2
from models.causal_discovery import run_phase_3
from models.causal_attention import run_phase_4
from models.champion import run_phase_5
from models.adaptive_stacking import run_phase_6
from models.conformal_intervals import run_phase_7
from models.analog_projections import run_phase_8
from models.stat_validation import run_phase_9
from utils.plotting import plot_stacked_bar_metrics

def main():
    print("==========================================================================")
    print("           PACI-ICPDA WILDFIRE FORECASTING FRAMEWORK MASTER RUNNER        ")
    print("==========================================================================")
    
    # 1. Load Data
    print("[0] Loading dataset...")
    df = load_dataset()
    if df is None:
        print("❌ Error loading dataset. Exiting.")
        sys.exit(1)
    print(f"  Dataset loaded successfully. Shape: {df.shape}")
    
    # Storage for compiling global cross-phase metrics
    global_metrics_list = []
    
    # Dict to store final metrics for stacked bar charts
    phase_metrics_dict = {
        "Phase_1_Baseline": [],
        "Phase_2_FuzzyVMD": [],
        "Phase_3_Causal": [],
        "Phase_4_CausalAttention": [],
        "Phase_5_Champion": [],
        "Phase_6_Adaptive": []
    }
    
    # 2. Loop over forecast horizons (Lags)
    for lag in config.LAGS:
        print("\n" + "="*80)
        print(f"   STARTING FORECAST PIPELINE FOR HORIZON: LAG = {lag} DAYS")
        print("="*80)
        
        # Prepare lagged sequences
        X, y = prepare_lagged_data(df, lag=lag, horizon=1)
        splits = get_temporal_splits(df, X, y, lag)
        
        # Extract validation targets
        y_val_true = splits["y_val"]
        y_future_true = splits["y_future"]
        dates_val = splits["dates_val"]
        
        # Phase 1: Baseline models (LR, RF, LSTM, Ensemble)
        p1_preds, p1_met = run_phase_1(splits, lag)
        global_metrics_list.append(p1_met)
        
        # Phase 2: VMD-Fuzzy Residual Learning
        p2_preds, p2_met = run_phase_2(splits, p1_preds, lag)
        global_metrics_list.append(p2_met)
        
        # Phase 3: Causal Discovery & Meta-learners
        p3_preds, p3_met = run_phase_3(df, splits, lag)
        global_metrics_list.append(p3_met)
        
        # Phase 4: Bahdanau Causal Attention LSTM Hybrid
        p4_preds, p4_met = run_phase_4(splits, p3_preds["causal_parents"], lag)
        global_metrics_list.append(p4_met)
        
        # Phase 5: Champion Selection
        # Collect validation forecasts
        all_val_preds = {
            "LR": p1_preds["LR"], "RF": p1_preds["RF"], "LSTM": p1_preds["LSTM"],
            "Ensemble": p1_preds["Ensemble"], "FuzzyVMD": p2_preds["FuzzyVMD"],
            "Causal": p3_preds["Causal"], "CausalAttention": p4_preds["CausalAttention"]
        }
        all_fut_preds = {
            "Future_LR": p1_preds["Future_Ensemble"],  # Fallbacks
            "Future_RF": p1_preds["Future_Ensemble"],
            "Future_LSTM": p1_preds["Future_Ensemble"],
            "Future_Ensemble": p1_preds["Future_Ensemble"],
            "Future_FuzzyVMD": p2_preds["Future_FuzzyVMD"],
            "Future_Causal": p3_preds["Future_Causal"],
            "Future_CausalAttention": p4_preds["Future_CausalAttention"]
        }
        p5_preds, p5_met = run_phase_5(y_val_true, y_future_true, all_val_preds, all_fut_preds, lag)
        global_metrics_list.append(p5_met)
        
        # Phase 6: Adaptive Error Correction Stacking
        p6_preds, p6_met = run_phase_6(df, splits, p5_preds, lag)
        global_metrics_list.append(p6_met)
        
        # Phase 7: Conformal Calibrated Uncertainty Intervals
        p7_intervals = run_phase_7(y_val_true, p6_preds, lag)
        
        # Phase 8: 1095-Day Future Projections & Fancharts
        p8_projections = run_phase_8(splits, p6_preds, p7_intervals, lag)
        
        # Phase 9: Wilcoxon Signed-Rank Significance Test
        p9_significance = run_phase_9(y_val_true, p1_preds, p6_preds, lag)
        
        # Parse metrics for plotting
        phase_metrics_dict["Phase_1_Baseline"].append(p1_met[p1_met["Model"] == "Ensemble"])
        phase_metrics_dict["Phase_2_FuzzyVMD"].append(p2_met)
        phase_metrics_dict["Phase_3_Causal"].append(p3_met)
        phase_metrics_dict["Phase_4_CausalAttention"].append(p4_met)
        phase_metrics_dict["Phase_5_Champion"].append(p5_met)
        phase_metrics_dict["Phase_6_Adaptive"].append(p6_met)
        
    # 3. Save Unified Global Metrics Table
    print("\n" + "="*80)
    print("                  COMPILING GLOBAL RESULTS SUMMARY REPORT")
    print("="*80)
    
    df_global_all = pd.concat(global_metrics_list, ignore_index=True)
    report_path = os.path.join(config.OUTPUT_DIR, "global_metrics_report.csv")
    df_global_all.to_csv(report_path, index=False)
    print(f"  Unified performance report saved to: {report_path}")
    
    # 4. Generate Stacked Bar Publication Charts
    print("\n[Plotting] Generating IEEE Stacked Bar Error Horizon Graphs...")
    
    # Clean Phase DataFrames for stacked bar plotter
    clean_phase_metrics = {}
    for p_name, df_list in phase_metrics_dict.items():
        clean_phase_metrics[p_name] = pd.concat(df_list, ignore_index=True)
        
    plot_stacked_bar_metrics(clean_phase_metrics, metric_name="MAE", filename="Stacked_MAE_Targets.png")
    plot_stacked_bar_metrics(clean_phase_metrics, metric_name="RMSE", filename="Stacked_RMSE_Targets.png")
    
    print("\n==========================================================================")
    print("      🎉 PACI-ICPDA WILDFIRE FORECASTING PIPELINE COMPLETED SUCCESSFULLY!  ")
    print("==========================================================================")

if __name__ == "__main__":
    main()
