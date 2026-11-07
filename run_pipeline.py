"""
ACID-FTS Framework - Master 7-Phase Pipeline Orchestrator
=========================================================
Executes the complete 7-Phase ACID-FTS (Adaptive Causal-Informed and Domain-Constrained
Fuzzy Time Series) workflow across multi-horizon lead times L in {1, 2, 7, 14, 30} days
and biogeographic forest divisions (Bandhavgarh, Similipal, Chamba).

Usage:
    python run_pipeline.py --region bandhavgarh --lags 1 2 7 14 30
"""

import argparse
import os
import time
import pandas as pd
import config
from data_loader import load_datacube, create_multi_horizon_splits
from models.phase1_baseline import run_phase_1
from models.phase2_vmd_fuzzy import run_phase_2
from models.phase3_causal_metalearners import run_phase_3
from models.phase4_causal_attention import run_phase_4
from models.phase5_dynamic_routing import run_phase_5
from models.phase6_adaptive_stacking import run_phase_6
from models.phase7_conformal_calibration import run_phase_7
from utils.plotting import plot_grouped_horizon_bars, plot_conformal_intervals


def run_acid_fts_pipeline(region_key=config.DEFAULT_REGION, lags=None, save_outputs=False):
    """
    Executes Phases 1 through 7 of the ACID-FTS architecture sequentially
    without temporal data leakage.
    """
    if lags is None:
        lags = config.LAGS

    region_info = config.REGIONS[region_key]
    print("=" * 80)
    print(f"ACID-FTS 7-Phase Multi-Horizon Forecasting Framework")
    print(f"Region: {region_info['name']} ({region_info['climate']})")
    print(f"Horizons: {lags} days | Targets: {config.TARGET_COLS}")
    print("=" * 80)

    start_time = time.time()
    df = load_datacube(region_key=region_key)

    all_metrics_list = []
    all_conformal_list = []
    all_wilcoxon_list = []

    for lag in lags:
        print(f"\n>>> Processing Forecast Horizon L = t+{lag}d <<<")
        splits = create_multi_horizon_splits(df, lag=lag)

        # Phase 1: Baseline Statistical, ML, DL & 4-Model Ensemble
        p1_preds, df_m1 = run_phase_1(splits, lag)

        # Phase 2: VMD-Fuzzy Residual Learning (ICPDA)
        p2_preds, df_m2 = run_phase_2(splits, p1_preds, lag)

        # Phase 3: Regime-Switching Causal Meta-Learners (S-Learner & Propensity T-Learner)
        p3_preds, df_m3 = run_phase_3(df, splits, lag)

        # Phase 4: Deep Causal Attention Hybrid (Bahdanau Query Attention + XGBoost)
        p4_preds, df_m4 = run_phase_4(splits, p3_preds["causal_parents"], lag)

        # Merge candidate predictions for Phase 5 Dynamic Routing
        candidate_pool = {}
        for d in (p1_preds, p2_preds, p3_preds, p4_preds):
            candidate_pool.update(d)

        # Phase 5: Unified Multi-Target Candidate & Dynamic Best Model Routing M*(Y, L)
        p5_preds, df_m5 = run_phase_5(splits, candidate_pool, lag)

        # Phase 6: Adaptive Residual Meta-Stacking with Physical Domain Enforcement
        p6_preds, df_m6 = run_phase_6(splits, p5_preds, lag)

        # Phase 7: Bounded Conformal Calibration, 1095-Day Analog Projections & Wilcoxon Test
        p7_out = run_phase_7(df, splits, p1_preds, p6_preds, lag)

        df_lag_metrics = pd.concat([df_m1, df_m2, df_m3, df_m4, df_m5, df_m6], ignore_index=True)
        all_metrics_list.append(df_lag_metrics)

        diag_df = p7_out["conformal"]["diagnostics"].copy()
        diag_df["Lag"] = lag
        all_conformal_list.append(diag_df)
        all_wilcoxon_list.append(p7_out["wilcoxon"])

        if save_outputs:
            out_dir = os.path.join(config.OUTPUT_DIR, region_key)
            os.makedirs(out_dir, exist_ok=True)
            for t_idx, t_col in enumerate(config.TARGET_COLS):
                plot_conformal_intervals(
                    splits["dates_val"],
                    splits["y_val"][:, t_idx],
                    p6_preds["Stacked"][:, t_idx],
                    p7_out["conformal"]["val_lower"][:, t_idx],
                    p7_out["conformal"]["val_upper"][:, t_idx],
                    target_col=t_col,
                    filename=f"Conformal_Holdout_Lag{lag}_{t_col}.png",
                    subfolder=f"{region_key}/plots",
                )

    df_all_metrics = pd.concat(all_metrics_list, ignore_index=True)
    df_all_conformal = pd.concat(all_conformal_list, ignore_index=True)
    df_all_wilcoxon = pd.concat(all_wilcoxon_list, ignore_index=True)

    if save_outputs:
        out_dir = os.path.join(config.OUTPUT_DIR, region_key)
        os.makedirs(out_dir, exist_ok=True)
        df_all_metrics.to_csv(os.path.join(out_dir, "Publication_Metrics_PhaseComparison.csv"), index=False)
        df_all_conformal.to_csv(os.path.join(out_dir, "Conformal_Calibration_Diagnostics.csv"), index=False)
        df_all_wilcoxon.to_csv(os.path.join(out_dir, "Wilcoxon_Significance_Summary.csv"), index=False)
        plot_grouped_horizon_bars(df_all_metrics, metric_name="RMSE", subfolder=f"{region_key}/plots")
        plot_grouped_horizon_bars(df_all_metrics, metric_name="MAE", subfolder=f"{region_key}/plots")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ACID-FTS Pipeline Complete in {elapsed:.2f}s.")
    print("=" * 80)
    return {
        "metrics": df_all_metrics,
        "conformal": df_all_conformal,
        "wilcoxon": df_all_wilcoxon,
    }


def main():
    parser = argparse.ArgumentParser(description="ACID-FTS 7-Phase Forest Fire Forecasting Pipeline")
    parser.add_argument(
        "--region",
        type=str,
        default=config.DEFAULT_REGION,
        choices=list(config.REGIONS.keys()),
        help="Target forest division (bandhavgarh, similipal, or chamba)",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=config.LAGS,
        help="Forecast lead times in days (default: 1 2 7 14 30)",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Persist CSV tables and PNG figures to local output/ directory (ignored by git)",
    )
    args = parser.parse_args()
    run_acid_fts_pipeline(region_key=args.region, lags=args.lags, save_outputs=args.save_outputs)


if __name__ == "__main__":
    main()

