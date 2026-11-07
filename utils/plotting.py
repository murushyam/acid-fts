"""
ACID-FTS Framework - Scientific Visualization Utilities
=======================================================
Provides IEEE publication-grade plotting functions for:
1. Grouped bar charts of multi-horizon RMSE & MAE across model phases
2. Conformal prediction fan charts (95% calibrated bounds)
3. 1095-day climatological analog multi-year projections
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import config

config.apply_plot_style()


def plot_grouped_horizon_bars(df_metrics, metric_name="RMSE", subfolder="plots"):
    """
    Generates grouped bar charts comparing forecasting architectures across
    prediction horizons L in {1, 2, 7, 14, 30} days for each fire target.
    """
    out_dir = os.path.join(config.OUTPUT_DIR, subfolder)
    os.makedirs(out_dir, exist_ok=True)

    lags = sorted(df_metrics["Lag"].unique())
    models = list(dict.fromkeys(df_metrics["Model"].tolist()))
    palette = ["#264653", "#2A9D8F", "#457B9D", "#E9C46A", "#F4A261", "#E76F51", "#8E44AD", "#C0392B"]

    for t_col in config.TARGET_COLS:
        t_label = config.TARGET_DISPLAY_NAMES.get(t_col, t_col)
        fig, ax = plt.subplots(figsize=(10, 4.8))

        x = np.arange(len(lags))
        bar_w = 0.8 / max(len(models), 1)

        for m_idx, model_name in enumerate(models):
            vals = []
            for lag in lags:
                sub = df_metrics[(df_metrics["Target"] == t_col) & (df_metrics["Lag"] == lag) & (df_metrics["Model"] == model_name)]
                vals.append(float(sub[metric_name].iloc[0]) if len(sub) > 0 else 0.0)

            offset = (m_idx - (len(models) - 1) / 2.0) * bar_w
            ax.bar(
                x + offset,
                vals,
                width=bar_w,
                label=model_name,
                color=palette[m_idx % len(palette)],
                edgecolor="black",
                linewidth=0.4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Lag {l}d" for l in lags])
        ax.set_ylabel(f"{metric_name} ({t_label})")
        ax.set_xlabel("Forecast Horizon")
        ax.set_title(f"Multi-Horizon {metric_name} Comparison - {t_label}", fontweight="bold")
        ax.legend(ncol=3, loc="upper left", frameon=True)

        plt.tight_layout()
        fname = os.path.join(out_dir, f"GroupedBar_{metric_name}_{t_col}.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_conformal_intervals(dates, y_true, y_pred, y_lower, y_upper, target_col, filename="conformal_intervals.png", subfolder="plots"):
    """
    Plots point forecasts alongside non-negative 95% split-conformal prediction intervals.
    """
    out_dir = os.path.join(config.OUTPUT_DIR, subfolder)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    t_label = config.TARGET_DISPLAY_NAMES.get(target_col, target_col)

    ax.plot(dates, y_true, label="Observed", color="black", linewidth=1.0, alpha=0.85)
    ax.plot(dates, y_pred, label="ACID-FTS Point Forecast", color="#C0392B", linewidth=1.2)
    ax.fill_between(dates, y_lower, y_upper, color="#E76F51", alpha=0.25, label="95% Bounded Conformal Interval")

    ax.set_ylabel(t_label)
    ax.set_xlabel("Date")
    ax.set_title(f"ACID-FTS Bounded Conformal Calibration - {t_label}", fontweight="bold")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

