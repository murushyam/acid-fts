"""
ACID-FTS Framework - Central Configuration Module
=================================================
Adaptive Causal-Informed and Domain-Constrained Fuzzy Time Series (ACID-FTS):
A Multi-Horizon Forecasting and Uncertainty Quantification Framework for Forest Fire Regimes.

Defines spatial study regions (Bandhavgarh, Similipal, Chamba), strict chronological
partitions (D_train, D_cal, D_test), the 26 core multi-sensor variables (expanded
with orthogonal aspect encodings), forecast horizons L in {1, 2, 7, 14, 30} days,
and phase-specific hyperparameters.
"""

import os

# -----------------------------------------------------------------------------
# 1. Directory & Regional Data Cube Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

REGIONS = {
    "bandhavgarh": {
        "name": "Bandhavgarh Tiger Reserve",
        "climate": "Tropical Dry Deciduous (Central India)",
        "bbox": (23.50, 24.00, 80.75, 81.25),
        "h5_file": os.path.join(DATA_DIR, "Bandhavgarh_DataCube.h5"),
        "test_end": "2024-07-31",
    },
    "similipal": {
        "name": "Similipal Biosphere Reserve",
        "climate": "Humid Sub-Tropical Moist Deciduous (Eastern India)",
        "bbox": (21.50, 22.25, 86.00, 86.75),
        "h5_file": os.path.join(DATA_DIR, "Similipal_DataCube.h5"),
        "test_end": "2024-07-31",
    },
    "chamba": {
        "name": "Chamba Himalayan Region",
        "climate": "Western Himalayan Montane Coniferous",
        "bbox": (32.25, 33.00, 75.75, 76.75),
        "h5_file": os.path.join(DATA_DIR, "Chamba_DataCube.h5"),
        "test_end": "2025-08-01",
    },
}

DEFAULT_REGION = "bandhavgarh"
H5_FILE_PATH = REGIONS[DEFAULT_REGION]["h5_file"]

# -----------------------------------------------------------------------------
# 2. Strict Chronological Data Splits (Zero Temporal Leakage)
# -----------------------------------------------------------------------------
TRAIN_START = "2019-01-01"
TRAIN_END = "2022-06-30"      # D_train: Base statistical, VMD-Fuzzy, causal & deep model fitting
CAL_START = "2022-07-01"
CAL_END = "2023-06-30"        # D_cal:   Phase 5 routing, Phase 6 stacking & Phase 7 conformal calibration
TEST_START = "2023-07-01"
TEST_END = "2025-06-30"       # D_test:  Default/fallback held-out prospective evaluation partition

# Regional held-out prospective test end dates (matching Table I and Section II.A)
REGIONAL_TEST_ENDS = {
    "bandhavgarh": "2024-07-31",   # 13 months (2023-07-01 to 2024-07-31)
    "similipal": "2024-07-31",     # 13 months (2023-07-01 to 2024-07-31)
    "chamba": "2025-08-01",        # 25 months (2023-07-01 to 2025-08-01)
}


def get_test_end(region=None):
    """Returns the regional test partition end date, defaulting to TEST_END."""
    if region and str(region).lower() in REGIONAL_TEST_ENDS:
        return REGIONAL_TEST_ENDS[str(region).lower()]
    return TEST_END

# Climatological Analog Projection Window (1,095 days / 3 years)
PROJ_START = "2025-07-01"
PROJ_END = "2028-06-29"

# -----------------------------------------------------------------------------
# 3. Multi-Target Variables & 26 Core Multi-Sensor Features
# -----------------------------------------------------------------------------
TARGET_COLS = [
    "fire_weather_index",
    "firms_fire_mask",
    "modis_burn_date",
]

TARGET_DISPLAY_NAMES = {
    "fire_weather_index": "Fire Weather Index (FWI)",
    "firms_fire_mask": "Fire Count (pixels)",
    "modis_burn_date": "Burnt Area (km²)",
}

# 21 Dynamic Fire-Weather/Meteorological/Target Features + 6 Static Descriptors (including Northness & Eastness) totaling 27 features
DYNAMIC_FEATURES = [
    "fire_weather_index",
    "drought_code",
    "build_up_index",
    "initial_spread_index",
    "fine_fuel_moisture_code",
    "duff_moisture_code",
    "keetch_byram_drought_index",
    "fire_danger_index",
    "burning_index",
    "energy_release_component",
    "firms_fire_mask",
    "modis_burn_date",
    "temperature_2m_mean",
    "wind_speed_mean",
    "specific_humidity_mean",
    "vpd_mean",
    "relative_humidity_mean",
    "soil_moisture_mean",
    "total_precipitation_sum",
    "UINFDR",
    "HDWI",
]

STATIC_FEATURES = [
    "elevation",
    "slope",
    "northness",          # cos(Aspect) circular encoding (Eq. 1)
    "eastness",           # sin(Aspect) circular encoding (Eq. 1)
    "dist_road",
    "rural_population",
]

FEATURES = DYNAMIC_FEATURES + STATIC_FEATURES

# Physical meteorological covariates used in Phase 6 Adaptive Residual Meta-Stacking (Eq. 25)
STACKING_WEATHER_COLS = [
    "temperature_2m_mean",
    "total_precipitation_sum",
    "vpd_mean",
]

# -----------------------------------------------------------------------------
# 4. Multi-Horizon Forecast Lags & Model Hyperparameters
# -----------------------------------------------------------------------------
LAGS = [1, 2, 7, 14, 30]
LOOKBACK_WINDOW = 30

# Phase 1: Baseline & Ensemble Hyperparameters
LSTM_EPOCHS = 15
BATCH_SIZE = 32
OPTUNA_TRIALS = 10

# Phase 2: VMD-Fuzzy Residual Learning (ICPDA) Hyperparameters
VMD_K_MODES = 5            # Number of Intrinsic Mode Functions (IMFs)
VMD_ALPHA = 2000.0         # Quadratic penalty bandwidth constraint
VMD_TAU = 0.0              # Dual ascent noise-tolerance step
VMD_TOL = 1e-6             # ADMM convergence tolerance
FUZZY_N_SETS = 5           # Number of triangular fuzzy linguistic sets {VL, L, M, H, VH}

# Phase 3: Regime-Switching Causal Meta-Learner Hyperparameters
FWI_REGIME_QUANTILE = 0.90  # 90th percentile threshold for extreme fire-weather regime T_t
PCMCI_ALPHA = 0.05          # Significance threshold for conditional independence tests
PCMCI_TAU_MAX = 7           # Maximum causal time lag (days)

# Phase 4: Deep Causal Attention Hybrid Hyperparameters
ATTENTION_HIDDEN_DIM = 32   # LSTM encoder hidden state dimension d_h
ATTENTION_UNITS = 16        # Bahdanau additive alignment dimension d_a

# Phase 7: Bounded Conformal Prediction & Loss Hyperparameters
CONFORMAL_ALPHA = 0.05      # Significance level alpha = 0.05 (95% nominal coverage)
LINEX_ASYMMETRY_B = 1.0     # Scale-normalized LinEx under-prediction penalty parameter b > 0
RANDOM_SEED = 42


def apply_plot_style():
    """Applies IEEE publication-grade Matplotlib styling."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
