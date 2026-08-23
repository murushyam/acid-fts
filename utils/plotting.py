"""
PACI-ICPDA Framework - Visualization Utilities
Implements IEEE-standard, high-resolution scientific plotting modules.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import config

# Apply Central Style
config.apply_plot_style()

def plot_forecast_comparison(dates, y_true, y_pred, target_cols, filename="forecast_comparison.png", subfolder="plots"):
    """
    Plots multi-target point forecasts comparison against observed data.
    """
    n_targets = len(target_cols)
    fig, axs = plt.subplots(n_targets, 1, figsize=(10, 3 * n_targets), sharex=True)
    
    if n_targets == 1:
        axs = [axs]
        
    for idx, var in enumerate(target_cols):
        ax = axs[idx]
        t_name = config.TARGET_DISPLAY_NAMES.get(var, var)
        
        ax.plot(dates, y_true[:, idx], label="Observed", color="black", alpha=0.8, linewidth=1.2)
        ax.plot(dates, y_pred[:, idx], label="PACI-ICPDA Forecast", color="#E76F51", alpha=0.9, linestyle="--")
        
        ax.set_ylabel(t_name)
        ax.legend(loc="upper right")
        
    plt.xlabel("Date")
    plt.suptitle("Multi-Target Wildfire Forecasting Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    out_dir = os.path.join(config.OUTPUT_DIR, subfolder)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved forecast comparison: {filename}")

def plot_stacked_bar_metrics(phase_metrics, metric_name="MAE", filename="stacked_bar_metrics.png", subfolder="plots"):
    """
    Creates a publication-grade horizontal stacked bar plot representing
    cumulative errors across forecast horizons (lags) for different phases.
    
    Args:
        phase_metrics (dict): Dict of phase -> DataFrame containing MAE/RMSE per target and lag.
        metric_name (str): 'MAE' or 'RMSE' to plot.
    """
    # Exclude unwanted parameters from layout
    lags = config.LAGS
    phases = list(phase_metrics.keys())
    targets = config.TARGET_COLS
    
    # 1 figure per target with 1 panel
    for t_col in targets:
        t_label = config.TARGET_DISPLAY_NAMES.get(t_col, t_col)
        
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        # Color palette showing sequential degradation
        colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
        
        y_pos = np.arange(len(phases))
        
        # Cumulative segment builder
        left_accum = np.zeros(len(phases))
        
        for lag_idx, lag in enumerate(lags):
            lengths = []
            for phase in phases:
                df = phase_metrics[phase]
                # Filter for target and lag
                sub = df[(df["Target"] == t_col) & (df["Lag"] == lag)]
                if len(sub) > 0:
                    lengths.append(sub[metric_name].values[0])
                else:
                    lengths.append(0.0)
                    
            lengths = np.array(lengths)
            bars = ax.barh(y_pos, lengths, left=left_accum, label=f"t+{lag}d", color=colors[lag_idx % len(colors)], edgecolor="white", height=0.65)
            
            # Embed values inside segments
            for bar_idx, bar in enumerate(bars):
                w = bar.get_width()
                if w > 0.05 * np.max(left_accum + lengths):  # Only show if segment is wide enough
                    x_c = bar.get_x() + w / 2
                    y_c = bar.get_y() + bar.get_height() / 2
                    txt = ax.text(x_c, y_c, f"{w:.2f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
                    txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="black")])
                    
            left_accum += lengths
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(phases)
        ax.set_title(f"{metric_name} Cumulative Error Profile - {t_label}", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Cumulative {metric_name}")
        ax.legend(title="Horizon", loc="lower right")
        
        # Add labels (a) panel indicator
        ax.text(-0.1, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="right")
        
        plt.tight_layout()
        out_dir = os.path.join(config.OUTPUT_DIR, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"Stacked_{metric_name}_{t_col}.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
    print(f"  [Plot] Saved stacked bar plots for {metric_name}")

def plot_conformal_intervals(dates, y_true, y_pred, y_lower, y_upper, target_col, filename="conformal_intervals.png", subfolder="plots"):
    """
    Plots forecasts with 95% conformal prediction calibration intervals.
    """
    plt.figure(figsize=(9.5, 5))
    t_label = config.TARGET_DISPLAY_NAMES.get(target_col, target_col)
    
    plt.plot(dates, y_true, label="Observed", color="black", linewidth=1.1, alpha=0.8)
    plt.plot(dates, y_pred, label="Point Forecast", color="#2A9D8F", linewidth=1.3)
    plt.fill_between(dates, y_lower, y_upper, color="#2A9D8F", alpha=0.2, label="95% Conformal Interval")
    
    plt.ylabel(t_label)
    plt.xlabel("Date")
    plt.title(f"Conformal Calibration Prediction Intervals - {t_label}", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left")
    
    out_dir = os.path.join(config.OUTPUT_DIR, subfolder)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved conformal intervals: {filename}")
