# ACID-FTS: Adaptive Causal-Informed and Domain-Constrained Fuzzy Time Series

Official Python implementation of **"ACID-FTS: A Multi-Horizon Forecasting and Uncertainty Quantification Framework for Forest Fire Regimes"**.

---

## Overview

Wildfire activity across tropical monsoon and montane forest ecosystems arises from non-stationary, zero-inflated interactions among atmospheric aridity, canopy fuel moisture, and terrain complexity across sub-seasonal lead times. **ACID-FTS** (*Adaptive Causal-Informed and Domain-Constrained Fuzzy Time Series*) is a seven-phase forecasting and uncertainty quantification architecture that integrates spectral-fuzzy decomposition, regime-switching causal inference, deep temporal attention, dynamic multi-horizon model routing, adaptive residual meta-stacking, and physically bounded split-conformal calibration.

### Key Architectural Phases

1. **Phase 1 — Multi-Paradigm Baseline & Ensemble Modeling (`models/phase1_baseline.py`)**:
   - Statistical persistence and autoregressive baselines: Naive, Seasonal Naive ($P=365$), Moving Average ($k=7$), and $\text{ARIMA}(5,1,0)$.
   - Machine learning & deep sequence architectures: Ridge-regularized Linear Regression, Optuna-tuned Random Forest, Multi-layer LSTM, Temporal Fusion Transformer (TFT), and a 4-model equal-weight ensemble.
2. **Phase 2 — VMD–Fuzzy Residual Learning / ICPDA (`models/phase2_vmd_fuzzy.py`)**:
   - Decomposes base training residuals $r_t = y_t - \hat{y}_{\text{base},t}$ into $K=5$ orthogonal Intrinsic Mode Functions (IMFs) via Variational Mode Decomposition (ADMM spectral optimization).
   - Fuzzifies each mode $u_k(t)$ into $n=5$ overlapping triangular linguistic sets $\{\text{VL, L, M, H, VH}\}$ parameterized by empirical quantile centers $c_1 < \dots < c_5$ and trains mode-specific Fuzzy Random Forest regressors.
3. **Phase 3 — Regime-Switching Causal Meta-Learners (`models/phase3_causal_metalearners.py`)**:
   - Identifies time-lagged causal parent drivers and defines an extreme fire-weather regime treatment indicator $T_t = \mathbf{1}[\text{FWI}_t > Q_{90}(\text{FWI})]$.
   - Implements both a single-surface Causal S-Learner $\hat{\mu}_S(\mathbf{x}_t, T_t)$ and a propensity-weighted Causal T-Learner $\hat{y}_{\text{T-Learner}} = e(\mathbf{x}_t)\hat{\mu}_1(\mathbf{x}_t) + (1 - e(\mathbf{x}_t))\hat{\mu}_0(\mathbf{x}_t)$ weighted by regularized logistic propensity scores $e(\mathbf{x}_t) = P(T_t = 1 \mid \mathbf{x}_t)$.
4. **Phase 4 — Deep Causal Attention Hybrid (`models/phase4_causal_attention.py`)**:
   - Encodes causal parent lookback windows via an LSTM encoder into hidden states $\{\mathbf{h}_1, \dots, \mathbf{h}_W\}$ and computes Bahdanau additive alignment weights conditioned on the terminal query state $\mathbf{h}_W$:
     $$s_i = \mathbf{v}_a^\top \tanh(\mathbf{W}_a \mathbf{h}_i + \mathbf{U}_a \mathbf{h}_W), \quad \mathbf{c}_t = \sum_{i=1}^W \alpha_i \mathbf{h}_i$$
   - Refines the concatenated representation $[\mathbf{c}_t; \mathbf{h}_W; \mathbf{x}_t]$ via gradient-boosted regression trees (XGBoost) evaluated under a scale-normalized LinEx asymmetric loss $\mathcal{L}_{\text{asym}}$.
5. **Phase 5 — Unified Multi-Target Candidate & Dynamic Best Model Routing (`models/phase5_dynamic_routing.py`)**:
   - Evaluates all candidate architectures strictly on the calibration split $\mathcal{D}_{\text{cal}}$ and dynamically routes each target-horizon pair $(Y, L)$ via $M^*(Y, L) = \arg\min_{M \in \mathcal{M}} \text{RMSE}_{\text{cal}}(M, Y, L)$.
6. **Phase 6 — Adaptive Residual Meta-Stacking (`models/phase6_adaptive_stacking.py`)**:
   - Conditions an Optuna-tuned XGBoost residual meta-regressor $g_\phi(\mathbf{z}_t)$ on strictly forecast-origin observable lagged errors $[e_t, e_{t-1}]$ (where $e_t = y_t - \hat{y}_{t \mid t-L}^{(M^*)}$) and contemporaneous physical weather covariates $[T_t, P_t, \text{VPD}_t]$, followed by non-negativity projection $\max(0, \cdot)$ and integer rounding for discrete Fire Count.
7. **Phase 7 — Bounded Conformal Prediction & 1,095-Day Analog Projections (`models/phase7_conformal_calibration.py`)**:
   - Performs Kolmogorov–Smirnov ($D_{\text{KS}}$) residual distribution diagnostics across Gaussian KDE, Skew-Normal, and Normal fits, computes finite-sample split-conformal quantiles $\hat{q}_\alpha$ on $\mathcal{D}_{\text{cal}}$, constructs non-negative prediction intervals $\mathcal{C}_{t+L}^{(\alpha)} = [\max(0, \hat{y}_{t+L} - \hat{q}_\alpha), \hat{y}_{t+L} + \hat{q}_\alpha]$, evaluates Prediction Interval Coverage Probability ($\text{PICP}$), Prediction Interval Normalized Average Width ($\text{PINAW}$), and Average Coverage Error ($\text{ACE} = \text{PICP} - (1 - \alpha)$), generates 1,095-day climatological analog projections with monthly-to-daily phenological disaggregation, and conducts paired Wilcoxon signed-rank tests.

---

## Repository Structure

```text
acid-fts/
├── config.py                              # Study regions, chronological splits, 26 features, and hyperparameters
├── data_loader.py                         # Aspect encoding, zero-leakage standardization, sliding-window tensors
├── generate_sample_data.py                # Synthetic 26-feature HDF5 data cube generator for reproducible testing
├── run_pipeline.py                        # Master CLI orchestrator for Phases 1-7
├── requirements.txt                       # Python package dependencies
├── LICENSE                                # MIT License
├── models/
│   ├── __init__.py
│   ├── phase1_baseline.py                 # Phase 1: Statistical, ML, DL (LSTM/TFT) & 4-Model Ensemble
│   ├── phase2_vmd_fuzzy.py                # Phase 2: VMD ADMM decomposition + 5-set triangular Fuzzy RF (ICPDA)
│   ├── phase3_causal_metalearners.py      # Phase 3: FWI regime switching, propensity scores, S- & T-Learners
│   ├── phase4_causal_attention.py         # Phase 4: Bahdanau query attention LSTM + XGBoost hybrid
│   ├── phase5_dynamic_routing.py          # Phase 5: Joint multi-target candidate & dynamic routing M*(Y,L)
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

## Installation

```bash
git clone https://github.com/murushyam/acid-fts.git
cd acid-fts
pip install -r requirements.txt
```

---

## Usage

### Run the Full 7-Phase Pipeline

```bash
# Run across all 5 forecast horizons (1, 2, 7, 14, 30 days) for Bandhavgarh Tiger Reserve
python run_pipeline.py --region bandhavgarh --lags 1 2 7 14 30

# Run for Similipal Biosphere Reserve or Chamba Himalayan Forest Division
python run_pipeline.py --region similipal --lags 1 2 7 14 30
python run_pipeline.py --region chamba --lags 1 2 7 14 30

# Run the automated unit test suite
python tests/test_pipeline.py
```

> **Note on Data Cubes**: If the regional HDF5 data cubes (`data/*_DataCube_2004_2026.h5`) are not present locally, `data_loader.py` automatically invokes `generate_sample_data.py` to synthesize a physically consistent 26-feature daily spatiotemporal dataset so the full 7-phase pipeline can be verified out-of-the-box.

---

## Chronological Partitioning Protocol

To guarantee zero temporal leakage across all seven phases, the multi-year historical record across the three Indian forest reserves (Bandhavgarh Tiger Reserve, Similipal Biosphere Reserve, and Chamba Himalayan region) is partitioned into three strictly disjoint chronological blocks:

| Partition | Date Range | Role in ACID-FTS |
| :--- | :--- | :--- |
| **Training ($\mathcal{D}_{\text{train}}$)** | `2019-01-01` – `2022-06-30` | Z-score standardization, VMD–Fuzzy centers, causal parent discovery, propensity scores, base & deep model training |
| **Calibration ($\mathcal{D}_{\text{cal}}$)** | `2022-07-01` – `2023-06-30` | Optuna hyperparameter tuning, Phase 5 dynamic routing $M^*(Y,L)$, Phase 6 meta-learner training, Phase 7 conformal quantile $\hat{q}_\alpha$ |
| **Holdout Test ($\mathcal{D}_{\text{test}}$)** | `2023-07-01` – `2024-07-31` (Bandhavgarh, Similipal); `2023-07-01` – `2025-08-01` (Chamba) | Strictly out-of-sample deterministic (RMSE, MAE, $\mathcal{L}_{\text{asym}}$, POD, FAR, CSI) and probabilistic (PICP, PINAW, ACE, Wilcoxon) evaluation |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
