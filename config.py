"""
PACI-ICPDA Framework - Configuration Settings
Physics-based Adaptive Causal Informed Fuzzy Time Series prediction based on the
Improved Cumulative Probabilistic Distribution Approach.

This file defines the training/validation date ranges, feature columns,
target variables, model hyperparameters, and visualization aesthetics.
"""

import os
import matplotlib.pyplot as plt

# Central Workspace Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_demo")

# Create directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datasets
DATA_FILE = os.path.join(DATA_DIR, "sample_data.h5")
DATA_FILE_ERA5 = os.path.join(DATA_DIR, "sample_data_era5.h5")

# Targets & Variable Names
TARGET_COLS = ["fire_weather_index", "firms_fire_mask", "modis_burn_date"]
TARGET_DISPLAY_NAMES = {
    "fire_weather_index": "Fire Weather Index (FWI)",
    "firms_fire_mask": "Fire Count (pixels)",
    "modis_burn_date": "Burnt Area (km²)"
}

# Features List (Weather drivers + historical fire lags)
FEATURES = [
    "fire_weather_index", "drought_code", "build_up_index", "initial_spread_index",
    "fine_fuel_moisture_code", "duff_moisture_code", "keetch_byram_drought_index",
    "fire_danger_index", "burning_index", "energy_release_component",
    "firms_fire_mask", "modis_burn_date",
    "temperature_2m_mean", "wind_speed_mean", "specific_humidity_mean", 
    "vpd_mean", "relative_humidity_mean", "soil_moisture_mean", 
    "total_precipitation_sum", "UINFDR", "HDWI"
]
STATIC_FEATURES = ["elevation", "slope", "aspect", "dist_road", "rural_population"]

# Causal/Weather Drivers for Stacking Context
CONTEXT_COLS = ["temperature_2m_mean", "total_precipitation_sum", "vpd_mean", "wind_speed_mean"]

# Forecast Horizons (Lag steps in days)
LAGS = [1, 7, 14, 30]

# Date Splits (Training / Validation / Future Projections)
TRAIN_START = "2019-01-01"
TRAIN_END = "2022-12-31"     
VAL_START = "2023-01-01"     
VAL_END = "2025-06-30"       
TEST_START = "2023-01-01"    
TEST_END = "2025-06-30"
FUTURE_START = "2025-07-01"  
FUTURE_END = "2028-12-31"    # For 1095-day future projection simulation

# Model Training Parameters
N_RUNS = 1                  # Stability loop iterations
OPTUNA_TRIALS = 3           # Low count for demo/testing, scale up for production
LSTM_EPOCHS = 10
BATCH_SIZE = 32
CONFORMAL_ALPHA = 0.05      # 95% prediction interval coverage probability (PICP)

# PCMCI Causal Discovery Settings
PCMCI_ALPHA = 0.05
PCMCI_TAU_MAX = 7

# =======================================================
# IEEE TRANSACTION PLOT STYLING STYLE
# =======================================================
PLOT_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (9, 5.5),
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.5
}

def apply_plot_style():
    """Apply standard IEEE publication-grade style to matplotlib."""
    for key, value in PLOT_PARAMS.items():
        try:
            plt.rcParams[key] = value
        except Exception:
            pass
