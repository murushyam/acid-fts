# PACI-ICPDA Wildfire Forecasting Framework

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/framework-PACI--ICPDA-orange.svg)]()

This repository implements the **Physics-based Adaptive Causal Informed Fuzzy Time Series prediction based on the Improved Cumulative Probabilistic Distribution Approach (PACI-ICPDA)** wildfire forecasting pipeline. 

PACI-ICPDA is a modular, multi-phase mathematical forecasting framework designed to predict critical fire weather variables—specifically the **Fire Weather Index (FWI)**, **Active Fire Count**, and **Burnt Area**—across multiple look-ahead forecast horizons (lags of 1, 7, 14, and 30 days).

---

## 9-Phase Master Pipeline Architecture

The framework coordinates nine distinct forecasting and validation phases:

```
[Input Data (HDF5)] -> [Phase 1: Baselines] -> [Phase 2: VMD-Fuzzy] -> [Phase 3: Causal PCMCI]
                             |                      |                     |
                             v                      v                     v
[Phase 4: Causal Attention] -> [Phase 5: Champion Selection] -> [Phase 6: Error Stacking]
                                                                          |
                                                                          v
[Phase 9: Wilcoxon Testing] <- [Phase 8: Future Projections] <- [Phase 7: Conformal Intervals]
```

1. **Phase 1: Baseline Ensemble Prediction** – Establishes performance bounds using classical models (Linear Regression), machine learning (Random Forest), and deep learning (LSTM) benchmarks.
2. **Phase 2: VMD-Fuzzy Residual Learning (ICPDA)** – Decomposes prediction residuals via Variational Mode Decomposition (VMD) to filter out high-frequency noise and applies fuzzy interval corrections.
3. **Phase 3: Causal Meta-Learners** – Applies PCMCI+ causal discovery to identify time-lagged causal parent weather drivers and uses causal treatment effect S-Learners to block spurious correlations.
4. **Phase 4: Deep Causal Attention Hybrid** – Restricts the input space of an LSTM network to the identified causal parents and applies a Bahdanau attention mechanism to score historical steps.
5. **Phase 5: Target-Specific Champion Assembly** – Evaluates candidate configurations on a validation set and selects the optimal model for each target variable.
6. **Phase 6: Adaptive Stacking** – Predicts and corrects residual errors of the champion model using a secondary XGBoost meta-regressor trained on weather context drivers.
7. **Phase 7: Conformal Interval Calibration** – Constructs distribution-free calibrated prediction intervals using split-conformal inference to guarantee coverage probability.
8. **Phase 8: Conformal analog forecasting and 1095-day projections** – Generates 3-year future probabilistic projections using a climatological analog method.
9. **Phase 9: Statistical Validation** – Conducts non-parametric Wilcoxon signed-rank tests to confirm the statistical significance of Phase 6 improvements over the baseline LSTM.

---

## Project Structure

```
PACI_ICPDA_Framework/
│
├── config.py                 # Central configurations (hyperparameters, paths, splits)
├── data_loader.py            # Standardized loading from HDF5/CSV + auto-fallback
├── generate_sample_data.py   # Lightweight synthetic HDF5 data generator
├── run_pipeline.py           # Main orchestrator running Phase 1 to Phase 9
│
├── models/                   # Modular implementations of pipeline phases
│   ├── __init__.py
│   ├── baseline.py           # Phase 1: LR, RF, LSTM models
│   ├── fuzzy_vmd.py          # Phase 2: VMD signal decomposition + fuzzy logic
│   ├── causal_discovery.py   # Phase 3: PCMCI+ (with correlation proxy fallback)
│   ├── causal_attention.py   # Phase 4: Bahdanau Attention Causal LSTM Hybrid
│   ├── champion.py           # Phase 5: Automated champion assembly selection
│   ├── adaptive_stacking.py  # Phase 6: XGBoost error stacking corrector
│   ├── conformal_intervals.py # Phase 7: Split-conformal prediction intervals
│   ├── analog_projections.py # Phase 8: 1095-day future projections
│   └── stat_validation.py    # Phase 9: Wilcoxon signed-rank significance testing
│
├── utils/                    # Shared utilities
│   ├── __init__.py
│   ├── metrics.py            # MAE, RMSE, MAPE calculations
│   └── plotting.py           # IEEE-grade plotting (fancharts, stacked bar graphs)
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Installation & Getting Started

### 1. Environment Setup

Create a conda environment and install the required dependencies:

```bash
# Create and activate environment
conda create -n PACI_ICPDA python=3.9
conda activate PACI_ICPDA

# Install packages
pip install -r requirements.txt
```

### 2. Out-of-the-Box Running (Synthetic Mode)

Running large-scale regional datasets (multi-GB HDF5 files) can be resource-intensive. To make testing immediate and portable, the codebase contains a **synthetic data generator** (`generate_sample_data.py`).

If you run the pipeline without putting real data files in the `data/` folder, the framework will **automatically** generate a lightweight 2x2 spatial dataset (`sample_data.h5` and `sample_data_era5.h5`) and run the complete 9-phase sequence:

```bash
# Run the complete 9-phase pipeline
python run_pipeline.py
```

*Note: The demo runs in just a few minutes using the sample data, outputting all forecasting arrays, Wilcoxon tables, and publication-ready fancharts.*

---

## Expected Outputs

All outputs are saved to the `output_demo/` folder, structured by look-ahead horizon:
- **`output_demo/global_metrics_report.csv`**: Combined performance report of all models across all targets and lags.
- **`output_demo/plots/`**: 
  - **`Stacked_MAE_Targets.png`**: Stacked bar chart showing cumulative MAE degradation over lag horizons.
  - **`Stacked_RMSE_Targets.png`**: Stacked bar chart showing cumulative RMSE.
- **`output_demo/LAG_[1|7|14|30]/`**:
  - **`Phase_8_Projections/Future_FanChart_[variable].png`**: Probabilistic forecast fan charts with 95% conformal prediction bands.
  - **`Phase_9_Validation/wilcoxon_significance_results.csv`**: Significance test output.
  - **`Phase_9_Validation/wilcoxon_table.tex`**: Ready-to-copy LaTeX table code for paper insertion.

---

## How to Upload to GitHub

To push this codebase to your GitHub repository (`https://github.com/murushyam/PACI-ICPDA`):

1. **Initialize Git** in the codebase folder:
   ```bash
   cd PACI_ICPDA_Framework
   git init
   ```

2. **Add a `.gitignore`** to exclude generated data, outputs, and cached python binaries:
   ```bash
   echo "data/" >> .gitignore
   echo "output_demo/" >> .gitignore
   echo "__pycache__/" >> .gitignore
   echo "*.npy" >> .gitignore
   echo ".ipynb_checkpoints/" >> .gitignore
   ```

3. **Commit** the files:
   ```bash
   git add .
   git commit -m "Initial commit: PACI-ICPDA Wildfire Forecasting Framework"
   ```

4. **Link** to your remote GitHub repository and **push**:
   ```bash
   git branch -M main
   git remote add origin https://github.com/murushyam/PACI-ICPDA.git
   git push -u origin main
   ```
