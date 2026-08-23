"""
PACI-ICPDA Framework - Synthetic Data Generator
Generates lightweight sample HDF5 data files to make the pipeline out-of-the-box runnable.
"""

import os
import h5py
import numpy as np
import pandas as pd
import config

def generate_lightweight_datasets(verbose=True):
    """
    Generates two small, standard-compliant HDF5 files:
      1. sample_data.h5 (Primary targets: fire_weather_index, firms_fire_mask, modis_burn_date)
      2. sample_data_era5.h5 (ERA5 covariates: temperature, wind, humidity, precipitation, etc.)
    
    Spans from TRAIN_START to VAL_END (approx 2373 days) with a 2x2 spatial grid.
    """
    if verbose:
        print("==================================================")
        print("GENERATING SYNTHETIC LIGHTWEIGHT HDF5 DATASETS...")
        print("==================================================")
        
    dates = pd.date_range(start=config.TRAIN_START, end=config.VAL_END, freq="D")
    n_days = len(dates)
    height, width = 2, 2  # 2x2 grid to represent spatial dimensions
    
    # Primary Targets file variables
    target_variables = [
        "fire_weather_index", "drought_code", "build_up_index", "initial_spread_index",
        "fine_fuel_moisture_code", "duff_moisture_code", "keetch_byram_drought_index",
        "fire_danger_index", "burning_index", "energy_release_component",
        "firms_fire_mask", "modis_burn_date"
    ]
    
    # ERA5 weather variables
    era5_variables = [
        "temperature_2m_mean", "wind_speed_mean", "specific_humidity_mean", 
        "vpd_mean", "relative_humidity_mean", "soil_moisture_mean", 
        "total_precipitation_sum"
    ]
    
    # Ensure config data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # 1. Generate sample_data.h5
    if verbose:
        print(f"Creating primary target data at: {config.DATA_FILE}")
    np.random.seed(42)
    
    # Generate semi-realistic synthetic series
    # Using sinusoidal functions + random noise to mock seasonal patterns
    with h5py.File(config.DATA_FILE, "w") as f:
        # Create byte-encoded times and variables
        time_bytes = np.array([str(d.date()).encode("utf-8") for d in dates])
        var_bytes = np.array([v.encode("utf-8") for v in target_variables])
        
        # Initialize synthetic values
        # Shape: (time, variables, height, width)
        data_arr = np.zeros((n_days, len(target_variables), height, width), dtype=np.float32)
        
        for t in range(n_days):
            season = np.sin(2 * np.pi * t / 365.0)  # seasonal index
            for v_idx, var in enumerate(target_variables):
                if var == "fire_weather_index":
                    val = 15.0 + 10.0 * season + np.random.normal(0, 2.0)
                    data_arr[t, v_idx, :, :] = np.clip(val, 0, 50)
                elif var == "firms_fire_mask":
                    # Sparsely distributed count integers
                    val = 2.0 + 3.0 * season + np.random.normal(0, 1.5)
                    # Create sparse fire counts (approx 15% fire occurrence)
                    if np.random.random() > 0.85:
                        data_arr[t, v_idx, :, :] = np.random.randint(1, 10, size=(height, width))
                    else:
                        data_arr[t, v_idx, :, :] = 0
                elif var == "modis_burn_date":
                    # Sparse area burned values
                    if np.random.random() > 0.88:
                        data_arr[t, v_idx, :, :] = np.random.uniform(0.1, 5.0, size=(height, width))
                    else:
                        data_arr[t, v_idx, :, :] = 0.0
                else:
                    # General indices
                    val = 50.0 + 30.0 * season + np.random.normal(0, 5.0)
                    data_arr[t, v_idx, :, :] = np.clip(val, 0, 200)
                    
        f.create_dataset("data", data=data_arr, compression="gzip")
        f.create_dataset("time", data=time_bytes)
        f.create_dataset("variable", data=var_bytes)
        
    # 2. Generate sample_data_era5.h5
    if verbose:
        print(f"Creating meteorological data at: {config.DATA_FILE_ERA5}")
        
    with h5py.File(config.DATA_FILE_ERA5, "w") as f_era:
        time_bytes = np.array([str(d.date()).encode("utf-8") for d in dates])
        var_bytes = np.array([v.encode("utf-8") for v in era5_variables])
        
        data_arr = np.zeros((n_days, len(era5_variables), height, width), dtype=np.float32)
        
        for t in range(n_days):
            season = np.sin(2 * np.pi * t / 365.0)
            for v_idx, var in enumerate(era5_variables):
                if var == "temperature_2m_mean":
                    val = 295.0 + 15.0 * season + np.random.normal(0, 3.0)  # Kelvin
                    data_arr[t, v_idx, :, :] = val
                elif var == "wind_speed_mean":
                    val = 2.5 + 1.2 * season + np.random.normal(0, 0.5)
                    data_arr[t, v_idx, :, :] = np.clip(val, 0.5, 8.0)
                elif var == "vpd_mean":
                    val = 1.2 + 0.8 * season + np.random.normal(0, 0.3)     # kPa
                    data_arr[t, v_idx, :, :] = np.clip(val, 0.05, 5.0)
                elif var == "total_precipitation_sum":
                    # Precipitation is zero-inflated
                    if np.random.random() > 0.7:
                        data_arr[t, v_idx, :, :] = np.random.exponential(10.0, size=(height, width))
                    else:
                        data_arr[t, v_idx, :, :] = 0.0
                else:
                    # Humidity / Moisture indices
                    val = 0.5 + 0.3 * season + np.random.normal(0, 0.1)
                    data_arr[t, v_idx, :, :] = np.clip(val, 0.0, 1.0)
                    
        f_era.create_dataset("data", data=data_arr, compression="gzip")
        f_era.create_dataset("time", data=time_bytes)
        f_era.create_dataset("variable", data=var_bytes)
        
    if verbose:
        print("✅ Sample datasets successfully created under data/ directory!")

if __name__ == "__main__":
    generate_lightweight_datasets()
