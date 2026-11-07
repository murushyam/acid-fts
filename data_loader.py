"""
ACID-FTS Framework - Spatiotemporal Data Cube Loader & Preprocessor
===================================================================
Implements circular topographic aspect encoding (Eq. 1), strict training-only
z-score standardization (Eq. 2), and multi-horizon sliding-window tensor
construction across D_train, D_cal, D_test, and 1095-day analog projections.
"""

import os
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config
from generate_sample_data import generate_regional_datacube


def encode_circular_aspect(aspect_deg):
    """
    Equation (1): Decomposes azimuth aspect angle (degrees) into continuous
    meridional (Northness) and zonal (Eastness) components to eliminate
    the 0 deg / 360 deg boundary discontinuity.
    """
    rad = np.deg2rad(aspect_deg)
    northness = np.cos(rad)
    eastness = np.sin(rad)
    return northness, eastness


def load_datacube(region_key=config.DEFAULT_REGION, h5_path=None):
    """
    Loads the 26-feature daily spatiotemporal DataFrame for the specified region.
    Automatically synthesizes a sample HDF5 cube if the file is not present locally.
    """
    if h5_path is None:
        h5_path = config.REGIONS[region_key]["h5_file"]

    if not os.path.exists(h5_path):
        print(f"  [DataLoader] HDF5 file not found at {h5_path}. Generating synthetic data cube...")
        generate_regional_datacube(region_key=region_key, h5_path=h5_path)

    with h5py.File(h5_path, "r") as h5f:
        data = h5f["features"][:]
        raw_dates = h5f["dates"][:]
        dates = pd.to_datetime([d.decode("utf-8") if isinstance(d, bytes) else str(d) for d in raw_dates])

    df = pd.DataFrame(data, index=dates, columns=config.FEATURES)
    df.index.name = "date"

    # Forward/backward fill isolated missing sensor days only within causal history
    df = df.ffill().bfill()
    return df


def create_multi_horizon_splits(df, lag, lookback=None):
    """
    Constructs strictly partitioned chronological tensors (D_train, D_cal, D_test, D_proj)
    for forecast lead time L = `lag` days using a lookback window W.

    Equation (2): Feature standardization parameters (mu_train, sigma_train) are
    estimated strictly on D_train (2004-2018) to guarantee zero temporal leakage.
    """
    if lookback is None:
        lookback = max(lag, 7)

    # Fit StandardScaler strictly on D_train predictors
    train_df = df.loc[config.TRAIN_START:config.TRAIN_END]
    scaler_X = StandardScaler()
    scaler_X.fit(train_df[config.FEATURES].values)

    X_scaled_all = scaler_X.transform(df[config.FEATURES].values)
    y_raw_all = df[config.TARGET_COLS].values
    dates_all = df.index

    X_windows, y_targets, target_dates, origin_indices = [], [], [], []
    n_total = len(df)

    for idx in range(lookback, n_total - lag + 1):
        target_idx = idx + lag - 1
        X_windows.append(X_scaled_all[idx - lookback:idx, :])
        y_targets.append(y_raw_all[target_idx, :])
        target_dates.append(dates_all[target_idx])
        origin_indices.append(idx - 1)

    X_windows = np.array(X_windows, dtype=np.float32)
    y_targets = np.array(y_targets, dtype=np.float32)
    target_dates = pd.DatetimeIndex(target_dates)
    origin_indices = np.array(origin_indices, dtype=np.int64)

    # Strict chronological partition masks based on target forecast date
    train_mask = (target_dates >= config.TRAIN_START) & (target_dates <= config.TRAIN_END)
    cal_mask = (target_dates >= config.CAL_START) & (target_dates <= config.CAL_END)
    test_mask = (target_dates >= config.TEST_START) & (target_dates <= config.TEST_END)
    proj_mask = (target_dates >= config.PROJ_START) & (target_dates <= config.PROJ_END)

    # Combine D_cal and D_test when reporting validation/holdout evaluation or expose both
    return {
        "X_train": X_windows[train_mask],
        "y_train": y_targets[train_mask],
        "dates_train": target_dates[train_mask],
        "origins_train": origin_indices[train_mask],
        "X_cal": X_windows[cal_mask],
        "y_cal": y_targets[cal_mask],
        "dates_cal": target_dates[cal_mask],
        "origins_cal": origin_indices[cal_mask],
        "X_val": X_windows[test_mask],
        "y_val": y_targets[test_mask],
        "dates_val": target_dates[test_mask],
        "origins_val": origin_indices[test_mask],
        "X_future": X_windows[proj_mask],
        "y_future": y_targets[proj_mask],
        "dates_future": target_dates[proj_mask],
        "origins_future": origin_indices[proj_mask],
        "scaler_X": scaler_X,
        "lookback": lookback,
        "lag": lag,
    }

