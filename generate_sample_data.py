"""
ACID-FTS Framework - Synthetic Multi-Sensor Data Cube Generator
===============================================================
Generates physically consistent daily spatiotemporal HDF5 data cubes
(2019-01-01 to 2028-06-29) for Bandhavgarh, Similipal, and Chamba forest divisions
when raw satellite/reanalysis HDF5 files are not locally present.
"""

import os
import h5py
import numpy as np
import pandas as pd
import config


def generate_regional_datacube(region_key="bandhavgarh", h5_path=None, seed=42):
    """
    Synthesizes a realistic daily forest fire data cube (2019-2028)
    reflecting the seasonal monsoon-drought fire regimes of the specified reserve.
    """
    if h5_path is None:
        h5_path = config.REGIONS[region_key]["h5_file"]

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=config.TRAIN_START, end=config.PROJ_END, freq="D")
    n_days = len(dates)
    doy = dates.dayofyear.values
    seasonal_phase = 2.0 * np.pi * (doy - 105) / 365.25  # Peak fire season in April (DOY ~105)

    region_scales = {
        "bandhavgarh": {"temp_base": 27.5, "ba_scale": 1.8, "fc_scale": 1.6, "fwi_scale": 28.0, "elev": 620.0, "slope": 11.5},
        "similipal":   {"temp_base": 26.8, "ba_scale": 2.2, "fc_scale": 2.0, "fwi_scale": 30.0, "elev": 850.0, "slope": 14.2},
        "chamba":      {"temp_base": 16.2, "ba_scale": 1.1, "fc_scale": 1.1, "fwi_scale": 21.0, "elev": 2150.0, "slope": 28.4},
    }
    rs = region_scales.get(region_key, region_scales["bandhavgarh"])

    # Core meteorological forcing variables
    temp_mean = rs["temp_base"] + 9.5 * np.cos(seasonal_phase) + rng.normal(0, 1.8, n_days)
    rh_mean = np.clip(58.0 - 26.0 * np.cos(seasonal_phase) + rng.normal(0, 7.5, n_days), 12.0, 99.0)
    spec_hum = np.clip(0.012 - 0.005 * np.cos(seasonal_phase) + rng.normal(0, 0.001, n_days), 0.002, 0.025)
    wind_mean = np.clip(3.2 + 1.4 * np.cos(seasonal_phase) + rng.gamma(1.5, 0.6, n_days), 0.5, 18.0)

    monsoon_wave = np.maximum(0.0, np.cos(2.0 * np.pi * (doy - 205) / 365.25))
    precip = rng.exponential(scale=8.5, size=n_days) * (monsoon_wave ** 2.5) * (rng.random(n_days) < 0.45)
    soil_moisture = np.clip(0.26 - 0.14 * np.cos(seasonal_phase) + 0.004 * precip + rng.normal(0, 0.02, n_days), 0.04, 0.48)

    es = 0.6108 * np.exp((17.27 * temp_mean) / (temp_mean + 237.3))
    vpd_mean = np.maximum(0.05, es * (1.0 - rh_mean / 100.0))

    # Canadian FWI system components & NFDRS / drought indices
    ffmc = np.clip(82.0 + 12.0 * np.cos(seasonal_phase) - 1.5 * precip + rng.normal(0, 2.5, n_days), 20.0, 99.0)
    dmc = np.clip(65.0 + 45.0 * np.cos(seasonal_phase) - 1.2 * precip + rng.normal(0, 5.0, n_days), 5.0, 220.0)
    dc = np.clip(310.0 + 190.0 * np.cos(seasonal_phase) - 2.0 * precip + rng.normal(0, 12.0, n_days), 25.0, 750.0)
    isi = np.clip(0.08 * ffmc * wind_mean + rng.normal(0, 1.0, n_days), 0.0, 45.0)
    bui = np.clip(0.8 * dmc + 0.16 * dc + rng.normal(0, 3.0, n_days), 5.0, 250.0)
    fwi = np.clip(0.35 * isi + 0.22 * bui + rs["fwi_scale"] * 0.25 * np.cos(seasonal_phase) + rng.normal(0, 2.0, n_days), 0.0, 95.0)

    kbdi = np.clip(320.0 + 210.0 * np.cos(seasonal_phase) - 4.0 * precip + rng.normal(0, 18.0, n_days), 0.0, 800.0)
    fdi = np.clip(1.1 * fwi + rng.normal(0, 2.0, n_days), 0.0, 100.0)
    bi = np.clip(1.35 * fwi + rng.normal(0, 3.0, n_days), 0.0, 130.0)
    erc = np.clip(0.45 * bui + rng.normal(0, 2.5, n_days), 0.0, 120.0)
    uinfdr = np.clip(0.5 * bi + 0.5 * erc + rng.normal(0, 2.0, n_days), 0.0, 125.0)
    hdwi = np.clip(vpd_mean * wind_mean * 2.5 + rng.normal(0, 1.2, n_days), 0.5, 65.0)

    # Zero-inflated active fire count (pixels) & burnt area (km^2)
    fire_prob = 1.0 / (1.0 + np.exp(-(fwi - np.quantile(fwi, 0.85)) / 4.5))
    active_mask = (rng.random(n_days) < fire_prob) & (precip < 1.5)

    firms_fire_mask = np.where(
        active_mask,
        rng.poisson(lam=np.maximum(0.3, rs["fc_scale"] * (fwi / 28.0) ** 1.4)),
        0,
    ).astype(float)

    modis_burn_date = np.where(
        firms_fire_mask > 0,
        firms_fire_mask * rs["ba_scale"] * (0.6 + 0.4 * (vpd_mean / np.mean(vpd_mean))) * rng.lognormal(0.0, 0.3, n_days),
        0.0,
    )

    # Static topographic & anthropogenic features with circular aspect encoding (Eq. 1)
    elevation = np.full(n_days, rs["elev"]) + rng.normal(0, 1.5, n_days)
    slope = np.full(n_days, rs["slope"]) + rng.normal(0, 0.3, n_days)
    aspect_rad = np.deg2rad((210.0 + rng.normal(0, 6.0, n_days)) % 360.0)
    northness = np.cos(aspect_rad)
    eastness = np.sin(aspect_rad)
    dist_road = np.full(n_days, 4.2) + rng.normal(0, 0.05, n_days)
    rural_pop = np.full(n_days, 145.0) + rng.normal(0, 1.0, n_days)

    feature_matrix = np.column_stack([
        fwi,
        dc,
        bui,
        isi,
        ffmc,
        dmc,
        kbdi,
        fdi,
        bi,
        erc,
        firms_fire_mask,
        modis_burn_date,
        temp_mean,
        wind_mean,
        spec_hum,
        vpd_mean,
        rh_mean,
        soil_moisture,
        precip,
        uinfdr,
        hdwi,
        elevation,
        slope,
        northness,
        eastness,
        dist_road,
        rural_pop,
    ])

    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("features", data=feature_matrix, compression="gzip")
        h5f.create_dataset("dates", data=np.array([d.strftime("%Y-%m-%d") for d in dates], dtype="S10"))
        h5f.attrs["feature_names"] = config.FEATURES
        h5f.attrs["region"] = region_key

    print(f"  [DataGenerator] Created multi-sensor HDF5 data cube at {h5_path} ({n_days} days)")
    return h5_path


if __name__ == "__main__":
    for reg in config.REGIONS:
        generate_regional_datacube(reg)
