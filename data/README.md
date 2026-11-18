# ACID-FTS Cleaned Multi-Horizon Wildfire Benchmark Dataset (Bandhavgarh, Similipal & Chamba, 2019–2025)

Cleaned, unit-verified, analysis-ready daily spatiotemporal datasets supporting the study:

> **Adaptive Causal-Informed and Domain-Constrained Fuzzy Time Series Forecasting for Multi-Horizon Forest Fire Danger and Extremes (ACID-FTS)**  
> **Authors**: M. S. Shyam Sunder, Sibarama Panigrahi, and Bhishma Tyagi  
> **Affiliation**: National Institute of Technology Rourkela, Odisha, India  
> **Source Code Repository**: [https://github.com/murushyam/acid-fts](https://github.com/murushyam/acid-fts)

---

## 1. Dataset Overview & Regional Coverage

This repository provides standardized, zero-leakage daily benchmark datasets (27 variables: 21 dynamic meteorological/fire-weather/satellite variables + 6 static topographic and anthropogenic descriptors) across three bioclimatically distinct Indian forest reserves:

| Study Region | Bioclimatic Zone | Bounding Box | Observational Window | Holdout Test Partition (`D_test`) | Files Provided |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bandhavgarh Tiger Reserve** | Tropical Dry Deciduous (Central India) | `23.50°N–24.00°N, 80.75°E–81.25°E` | `2019-01-01` to `2024-07-31` | `2023-07-01` to `2024-07-31` | `Bandhavgarh_DataCube.h5`<br>`Bandhavgarh_Cleaned_Daily.csv` |
| **Similipal Biosphere Reserve** | Humid Sub-Tropical Moist Deciduous (Eastern India) | `21.50°N–22.25°N, 86.00°E–86.75°E` | `2019-01-01` to `2024-07-31` | `2023-07-01` to `2024-07-31` | `Similipal_DataCube.h5`<br>`Similipal_Cleaned_Daily.csv` |
| **Chamba Himalayan Division** | Western Himalayan Montane Coniferous | `32.25°N–33.00°N, 75.75°E–76.75°E` | `2019-01-01` to `2025-08-01` | `2023-07-01` to `2025-08-01` | `Chamba_DataCube.h5`<br>`Chamba_Cleaned_Daily.csv` |

---

## 2. Chronological Partitioning Protocol

Each CSV file includes an explicit `partition` column enforcing zero temporal data leakage across the seven phases of ACID-FTS:

- **`D_train` (`2019-01-01` to `2022-06-30`)**: Training partition used for feature standardization (`z`-score), VMD–Fuzzy interval clustering, causal parent discovery, propensity score estimation, and base model fitting (Phases 1–4).
- **`D_cal` (`2022-07-01` to `2023-06-30`)**: Held-out calibration partition used for Phase 5 dynamic model routing, Phase 6 forecast-origin residual meta-stacking training, and Phase 7 split-conformal quantile estimation.
- **`D_test` (`2023-07-01` to `2024-07-31` / `2025-08-01`)**: Strictly unseen prospective holdout evaluation partition.
- **`D_proj` (up to `2028-06-29`)**: 1,095-day climatological analog projection window used in Phase 7 long-term risk outlook evaluation.

---

## 3. File Formats & Direct Usage with `acid-fts`

- **HDF5 Data Cubes (`*_DataCube.h5`)**: Contain compressed `float32` array `features` of shape `(N_days, 27)`, UTF-8 `dates`, and `feature_names`. Drop directly into the `data/` directory of [`acid-fts`](https://github.com/murushyam/acid-fts).
- **CSV Files (`*_Cleaned_Daily.csv`)**: Human-readable daily tables indexed by `date` with `partition` and all 27 variables.
- **Metadata & Verification**:
  - `Data_Dictionary_and_Metadata.csv`: Complete variable definitions, physical units, and satellite/reanalysis provenance.
  - `Unit_Verification_Summary.csv`: Summary statistics (`Min`, `Mean`, `Std`, `Max`, `Non_Zero_Days`, `Zero_Fraction_Pct`) for the three target variables (`fire_weather_index`, `firms_fire_mask`, `modis_burn_date`).

---

## 4. License

Released under the **Creative Commons Attribution 4.0 International (CC-BY 4.0)** license.
