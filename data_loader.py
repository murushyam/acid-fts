"""
PACI-ICPDA Framework - Standardized Data Loader
Loads and pre-processes spatial/regional wildfire prediction datasets from HDF5 format.
Includes an automatic fallback to synthetic sample data if no file is found.
"""

import os
import h5py
import numpy as np
import pandas as pd
import config
from generate_sample_data import generate_lightweight_datasets

def load_dataset():
    """
    Loads target fire variables and weather features, aligns them temporally,
    calculates physical features (HDWI, UINFDR), and handles temporal analogies.
    
    If datasets do not exist, automatically triggers the synthetic generator.
    """
    # Trigger fallback if data files are missing
    if not os.path.exists(config.DATA_FILE) or not os.path.exists(config.DATA_FILE_ERA5):
        print("[DataLoader] Datasets not found. Triggering synthetic generator...")
        generate_lightweight_datasets(verbose=False)
        
    try:
        # 1. Load Primary targets HDF5
        with h5py.File(config.DATA_FILE, "r") as f:
            data = f["data"][:]
            times = [t.decode("utf-8") for t in f["time"][:]]
            variables = [v.decode("utf-8").strip() for v in f["variable"][:]]
            
        time_index = pd.to_datetime(times)
        regional_data = {}
        
        # Spatial processing
        for idx, var in enumerate(variables):
            if var == "firms_fire_mask":
                # Fire count is the sum of pixels with active fires (threshold > 0)
                fire_counts = np.zeros(len(times))
                for t in range(len(times)):
                    fire_counts[t] = np.nansum(data[t, idx, :, :] > 0)
                regional_data[var] = fire_counts
            else:
                # Other continuous variables use the spatial mean
                regional_data[var] = np.nanmean(data[:, idx, :, :], axis=(1, 2))
                
        df = pd.DataFrame(regional_data, index=time_index)
        
        # 2. Load ERA5 Meteorological HDF5
        with h5py.File(config.DATA_FILE_ERA5, "r") as f_era:
            data_era = f_era["data"][:]
            times_era = [t.decode("utf-8") for t in f_era["time"][:]]
            vars_era = [v.decode("utf-8").strip() for v in f_era["variable"][:]]
            
        regional_data_era = {
            var: np.nanmean(data_era[:, idx, :, :], axis=(1, 2)) 
            for idx, var in enumerate(vars_era)
        }
        df_era = pd.DataFrame(regional_data_era, index=pd.to_datetime(times_era))
        
        # Temporal alignment (inner join + group averages for duplicates)
        df = df.groupby(level=0).mean()
        df_era = df_era.groupby(level=0).mean()
        df = df.join(df_era, how="inner")
        
        # 3. Climatological Future Analog Extension (Phase 8 Analogs)
        last_date = df.index[-1]
        future_end_date = pd.Timestamp(config.FUTURE_END)
        
        if last_date < future_end_date:
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), future_end_date, freq="D")
            future_df = pd.DataFrame(index=future_dates, columns=df.columns)
            
            # Map future years back to historical baseline analogs
            year_mapping = {2025: 2021, 2026: 2022, 2027: 2019, 2028: 2020}
            feature_cols = [col for col in df.columns if col not in config.TARGET_COLS]
            
            for fd in future_dates:
                analog_year = year_mapping.get(fd.year, 2021)
                try:
                    src_date = pd.Timestamp(year=analog_year, month=fd.month, day=fd.day)
                    if src_date in df.index:
                        future_df.loc[fd, feature_cols] = df.loc[src_date, feature_cols].values
                except Exception:
                    pass
            
            df = pd.concat([df, future_df]).ffill().bfill()
            
        df.index.name = "Date"
        
        # 4. Feature Engineering
        # Hot Dry Windy Index (HDWI) = Wind Speed * Vapor Pressure Deficit
        if "wind_speed_mean" in df.columns and "vpd_mean" in df.columns:
            df["HDWI"] = df["wind_speed_mean"] * df["vpd_mean"]
        else:
            df["HDWI"] = 0.0
            
        # Universal Fire Danger Rating proxy (UINFDR)
        if "fire_danger_index" in df.columns:
            df["UINFDR"] = df["fire_danger_index"].copy()
        else:
            df["UINFDR"] = 0.0
            
        # Handle residual NaNs
        df = df.ffill().bfill().fillna(0.0)
        return df
        
    except Exception as e:
        print(f"[DataLoader] Critical error loading data: {e}")
        return None

def prepare_lagged_data(df, lag, horizon=1):
    """
    Prepares features (X) and targets (y) arrays based on forecast lag steps.
    Returns:
        X (np.ndarray): Shape (Samples, Lag, Features)
        y (np.ndarray): Shape (Samples, Targets)
    """
    X, y = [], []
    features_data = df[config.FEATURES].values
    targets_data = df[config.TARGET_COLS].values
    
    for i in range(len(df) - lag - horizon + 1):
        X.append(features_data[i : i + lag])
        y.append(targets_data[i + lag + horizon - 1])
        
    return np.array(X), np.array(y)

def split_static_dynamic(X):
    """
    Splits the 3D feature array into sequential dynamic (meteorological/fire)
    and static (topography/distance) features for Transformer-based architectures.
    """
    static_indices = [config.FEATURES.index(c) for c in config.STATIC_FEATURES if c in config.FEATURES]
    dynamic_indices = [i for i in range(len(config.FEATURES)) if i not in static_indices]
    
    X_dyn = X[:, :, dynamic_indices]
    X_stat = X[:, -1, static_indices]  # last observed step for static features
    return X_dyn, X_stat

def get_temporal_splits(df, X, y, lag):
    """
    Returns data splits for training, validation, and future analog periods.
    """
    def get_index(date_str):
        ts = pd.Timestamp(date_str)
        return df.index.get_indexer([ts], method="nearest")[0]
        
    tr_start = get_index(config.TRAIN_START)
    tr_end = get_index(config.TRAIN_END) + 1
    va_start = get_index(config.VAL_START)
    va_end = get_index(config.VAL_END) + 1
    fut_start = get_index(config.FUTURE_START)
    fut_end = min(get_index(config.FUTURE_END) + 1, len(df))
    
    # Map raw indices to shifted lagged arrays
    train_s, train_e = max(0, tr_start - lag), max(0, tr_end - lag)
    val_s, val_e = max(0, va_start - lag), max(0, va_end - lag)
    fut_s, fut_e = max(0, fut_start - lag), max(0, fut_end - lag)
    
    splits = {
        "X_train": X[train_s:train_e],
        "y_train": y[train_s:train_e],
        "X_val": X[val_s:val_e],
        "y_val": y[val_s:val_e],
        "X_future": X[fut_s:fut_e],
        "y_future": y[fut_s:fut_e],
        "dates_train": df.index[train_s + lag : train_e + lag],
        "dates_val": df.index[val_s + lag : val_e + lag],
        "dates_future": df.index[fut_s + lag : fut_e + lag]
    }
    return splits
