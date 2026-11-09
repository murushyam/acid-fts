# ACID-FTS: Adaptive Causal-Informed and Domain-Constrained Fuzzy Time Series

Official Python implementation of **"Adaptive Causal-Informed and Domain-Constrained Fuzzy Time Series Forecasting for Multi-Horizon Forest Fire Danger and Extremes (ACID-FTS)"**.

---

## Overview

Wildfire activity across tropical deciduous and montane forest ecosystems is governed by non-stationary, zero-inflated interactions among atmospheric aridity, fuel moisture desiccation, and terrain complexity across multi-week lead times. **ACID-FTS** is a seven-phase forecasting and uncertainty quantification framework that integrates variational mode decomposition, fuzzy interval residual learning, regime-switching causal meta-learners, deep temporal attention, dynamic model routing, forecast-origin residual meta-stacking, and physically bounded split-conformal calibration.

---

## Seven-Phase Pipeline Architecture

| Phase | Module | Methodological Role |
| :--- | :--- | :--- |
| **Phase 1** | `models/phase1_baseline.py` | **Baseline Ensemble Prediction**: Evaluates statistical baselines (Naive, Seasonal Naive, Moving Average, ARIMA) alongside Ridge Regression, Optuna-tuned Random Forest, Multi-layer LSTM, and Temporal Fusion Transformer (TFT), combining the top four multivariate learners into an equal-weight ensemble. |
| **Phase 2** | `models/phase2_vmd_fuzzy.py` | **VMD–Fuzzy Residual Learning (ICPDA)**: Decomposes Phase 1 training residuals into `K = 5` band-limited Intrinsic Mode Functions (IMFs) via Variational Mode Decomposition (VMD), fuzzifies each mode across `n = 5` triangular quantile intervals, and trains mode-specific Fuzzy Random Forest regressors. |
| **Phase 3** | `models/phase3_causal_metalearners.py` | **Regime-Switching Causal Meta-Learners**: Defines binary extreme fire-weather treatments at the 90th training percentile of the Canadian Fire Weather Index (`FWI`), estimates logistic propensity scores, and trains decoupled Causal S-Learner and propensity-weighted Causal T-Learner estimators. |
| **Phase 4** | `models/phase4_causal_attention.py` | **Deep Causal Attention Hybrid**: Encodes 30-day lookback sequences (`W = 30`) through a causal LSTM with Bahdanau additive attention and refines the concatenated context vector and terminal state via an XGBoost regressor. |
| **Phase 5** | `models/phase5_dynamic_routing.py` | **Candidate Assembly & Dynamic Routing**: Benchmarks a joint multi-target hybrid network against single-target candidates and routes each target variable and lead horizon (`L` in `{1, 2, 7, 14, 30}` days) to its optimal architecture on the held-out calibration partition. |
| **Phase 6** | `models/phase6_adaptive_stacking.py` | **Adaptive Residual Meta-Stacking**: Trains a secondary XGBoost error corrector conditioned strictly on forecast-origin realized residuals and contemporaneous weather covariates (`Temperature`, `Precipitation`, `VPD`), enforcing non-negativity and discrete integer rounding for fire counts. |
| **Phase 7** | `models/phase7_conformal_calibration.py` | **Bounded Conformal Calibration & Projections**: Fits residual distributions (KDE, Skew-Normal, Gaussian) via Kolmogorov–Smirnov testing, constructs non-negative split-conformal prediction intervals (`60%`, `80%`, `90%`, `95%` CI), generates 1,095-day CMIP6 analog projections, and runs paired Wilcoxon signed-rank tests. |

---

## Core Mathematical Formulations

### 1. Circular Topographic Aspect Decomposition

To prevent discontinuity artifacts at `0° / 360°` during spatial aggregation, terrain aspect (in degrees) is decomposed into orthogonal directional components:

$$
\text{Northness} = \cos\left(\frac{\theta \pi}{180}\right), \qquad \text{Eastness} = \sin\left(\frac{\theta \pi}{180}\right)
$$

### 2. Scale-Normalized Asymmetric LinEx Loss

To penalize missed wildfire outbreaks exponentially while penalizing false alarms linearly across heterogeneous target scales, prediction errors are normalized by the training target standard deviation and clipped to `[-10, 10]`:

$$
\mathcal{L}_{\text{asym}}(y_i, \hat{y}_i; b) = \frac{2}{b^2} \left( \exp(b u_i) - b u_i - 1 \right), \qquad u_i = \text{clip}\left(\frac{y_i - \hat{y}_i}{\sigma_Y}, \; -10, \; 10\right)
$$

### 3. Propensity-Weighted Causal T-Learner

Extreme fire-weather regimes are defined at the 90th training percentile of the Canadian Fire Weather Index (`FWI`), and logistic propensity scores weight the regime-specific response surfaces via the law of total expectation:

$$
T_t = \mathbf{1}\left[\text{FWI}_t > Q_{90}(\text{FWI})\right], \qquad e(\mathbf{x}_t) = P(T_t = 1 \mid \mathbf{x}_t)
$$

$$
\hat{y}_{t+L}^{\text{T}} = e(\mathbf{x}_t) \, f_1(\tilde{\mathbf{x}}_t) + \left(1 - e(\mathbf{x}_t)\right) f_0(\tilde{\mathbf{x}}_t)
$$

### 4. Strictly Causal Residual Meta-Stacking & Domain Projection

At forecast origin `t` for lead horizon `L`, secondary residual correction conditions strictly on realized forecast-origin errors observable at time `t`, followed by non-negative domain projection (and nearest-integer rounding for discrete active fire pixel counts):

$$
e_{t \mid t-L} = y_t - \hat{y}_{t \mid t-L}^*
$$

$$
\hat{e}_{t+L \mid t} = f_{\text{XGB}}\left(\left[\hat{y}_{t+L \mid t}^*, \; e_{t \mid t-L}, \; e_{t-1 \mid t-L-1}, \; T_t, \; P_t, \; \text{VPD}_t\right]\right)
$$

$$
\hat{y}_{t+L}^{\text{P6}} = \max\left(0, \; \hat{y}_{t+L \mid t}^* + \hat{e}_{t+L \mid t}\right)
$$

### 5. Bounded Split-Conformal Prediction Intervals

Using absolute residual non-conformity scores on the held-out calibration partition, finite-sample conformal quantiles at miscoverage rate `α` are projected onto the non-negative physical support `[0, ∞)`:

$$
S_i = \left| y_i - \hat{y}_i^{\text{P6}} \right|, \qquad \mathcal{C}_{t+L}^{(\alpha)} = \left[ \max\left(0, \; \hat{y}_{t+L}^{\text{P6}} - \hat{q}_\alpha\right), \; \hat{y}_{t+L}^{\text{P6}} + \hat{q}_\alpha \right]
$$

---

## Repository Structure

```text
acid-fts/
├── config.py                              # Study regions, chronological splits, 27 features, and hyperparameters
├── data_loader.py                         # Aspect encoding, zero-leakage standardization, sliding-window tensors
├── generate_sample_data.py                # Synthetic 27-feature HDF5 data cube generator for reproducible testing
├── run_pipeline.py                        # Master CLI orchestrator for Phases 1-7
├── requirements.txt                       # Python package dependencies
├── LICENSE                                # MIT License
├── models/
│   ├── __init__.py
│   ├── phase1_baseline.py                 # Phase 1: Statistical, ML, DL (LSTM/TFT) & 4-Model Ensemble
│   ├── phase2_vmd_fuzzy.py                # Phase 2: VMD ADMM decomposition + 5-set triangular Fuzzy RF (ICPDA)
│   ├── phase3_causal_metalearners.py      # Phase 3: FWI regime switching, propensity scores, S- & T-Learners
│   ├── phase4_causal_attention.py         # Phase 4: Bahdanau query attention LSTM + XGBoost hybrid
│   ├── phase5_dynamic_routing.py          # Phase 5: Joint multi-target candidate & dynamic routing
│   ├── phase6_adaptive_stacking.py        # Phase 6: Forecast-origin lagged residual XGBoost meta-stacking
│   └── phase7_conformal_calibration.py    # Phase 7: KS diagnostics, bounded conformal intervals, 1095d analogs
├── tests/
│   └── test_pipeline.py                   # 17 unit tests verifying mathematical fidelity & zero temporal leakage
└── utils/
    ├── __init__.py
    ├── metrics.py                         # RMSE, MAE, LinEx loss, PICP, PINAW, ACE, and POD/FAR/F1/CSI metrics
    └── plotting.py                        # Grouped horizon bar charts & conformal fan chart utilities
```

---

## Installation & Usage

```bash
git clone https://github.com/murushyam/acid-fts.git
cd acid-fts
pip install -r requirements.txt
```

### Run the Full 7-Phase Pipeline

```bash
# Run across all 5 forecast horizons (1, 2, 7, 14, 30 days) for Bandhavgarh Tiger Reserve
python run_pipeline.py --region bandhavgarh --lags 1 2 7 14 30

# Run for Similipal Biosphere Reserve or Chamba Himalayan Forest Division
python run_pipeline.py --region similipal --lags 1 2 7 14 30
python run_pipeline.py --region chamba --lags 1 2 7 14 30

# Run the automated unit test suite (17 unit tests)
python tests/test_pipeline.py
```

> **Note on Data Cubes**: If regional HDF5 data cubes (`data/*_DataCube_2004_2026.h5`) are not present locally, `data_loader.py` automatically invokes `generate_sample_data.py` to synthesize a physically consistent 27-feature daily spatiotemporal dataset so the full 7-phase pipeline can be verified out-of-the-box.

---

## Chronological Partitioning Protocol

To guarantee zero temporal leakage across all seven phases, the historical record across the three Indian forest reserves (Bandhavgarh Tiger Reserve, Similipal Biosphere Reserve, and Chamba Western Himalaya) is partitioned into three strictly disjoint chronological splits:

| Partition | Date Range | Role in ACID-FTS |
| :--- | :--- | :--- |
| **Training (`D_train`)** | `2019-01-01` to `2022-06-30` | Z-score standardization, VMD–Fuzzy interval centroids, causal parent discovery, propensity scores, and base model fitting (Phases 1–4) |
| **Calibration (`D_cal`)** | `2022-07-01` to `2023-06-30` | Out-of-sample Phase 5 model routing, Phase 6 residual meta-stacking training, and Phase 7 conformal quantile calibration |
| **Holdout Test (`D_test`)** | `2023-07-01` to `2024-07-31` (Bandhavgarh, Similipal)<br>`2023-07-01` to `2025-08-01` (Chamba) | Prospective out-of-sample deterministic (`RMSE`, `MAE`, `LinEx`, `POD`, `FAR`, `CSI`) and probabilistic (`PICP`, `PINAW`, `ACE`, `Wilcoxon`) evaluation |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
