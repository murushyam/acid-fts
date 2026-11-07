"""
ACID-FTS Framework - Phase 1: Baseline Statistical, ML, DL & Ensemble Models
============================================================================
Implements Algorithm 1 of the manuscript:
1. Naive Persistence, Seasonal Naive (P=365), Moving Average (k=7), and ARIMA(5,1,0)
2. Linear Regression (OLS/Ridge) and Random Forest (Optuna-tuned)
3. Deep Learning: Multi-layer LSTM and Temporal Fusion Transformer (TFT)
4. 4-Model Equal-Weight Ensemble (LR + RF + LSTM + ARIMA, Eq. 4)
"""

import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import config
from utils.metrics import calculate_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _enforce_physical_constraints(preds):
    """Applies non-negativity projection max(0, y) and integer rounding on Fire Count."""
    out = np.maximum(np.asarray(preds, dtype=np.float64), 0.0)
    if out.ndim == 2 and out.shape[1] > 1:
        out[:, 1] = np.round(out[:, 1])
    return out


def predict_statistical_baselines(y_train, y_eval, lag):
    """
    Computes Naive Persistence, Seasonal Naive (P=365), Moving Average (k=7),
    and autoregressive ARIMA(5,1,0) linear filter approximations across targets.
    """
    history = np.vstack([y_train, y_eval])
    n_train = len(y_train)
    n_eval = len(y_eval)
    n_targets = y_train.shape[1]

    pred_naive = np.zeros_like(y_eval)
    pred_snaive = np.zeros_like(y_eval)
    pred_ma = np.zeros_like(y_eval)
    pred_arima = np.zeros_like(y_eval)

    for i in range(n_eval):
        origin_idx = n_train + i - lag
        pred_naive[i] = history[max(0, origin_idx)]

        seasonal_idx = n_train + i - 365
        pred_snaive[i] = history[max(0, seasonal_idx)]

        ma_start = max(0, origin_idx - 6)
        pred_ma[i] = np.mean(history[ma_start:origin_idx + 1], axis=0)

    # Fit AR(5) on first-differenced series (ARIMA(5,1,0) fast exact least-squares)
    for t_idx in range(n_targets):
        diff_tr = np.diff(y_train[:, t_idx], prepend=y_train[0, t_idx])
        p_order = 5
        if len(diff_tr) > p_order + 5:
            X_ar = np.column_stack([ np.roll(diff_tr, k + 1) for k in range(p_order) ])[p_order:]
            y_ar = diff_tr[p_order:]
            ar_weights, _, _, _ = np.linalg.lstsq(X_ar, y_ar, rcond=None)
        else:
            ar_weights = np.zeros(p_order)

        full_series = history[:, t_idx]
        full_diff = np.diff(full_series, prepend=full_series[0])
        for i in range(n_eval):
            origin_idx = max(p_order, n_train + i - lag)
            recent_diffs = full_diff[origin_idx - p_order + 1:origin_idx + 1][::-1]
            pred_diff = float(np.dot(ar_weights, recent_diffs))
            pred_arima[i, t_idx] = full_series[origin_idx] + pred_diff

    return {
        "Naive": _enforce_physical_constraints(pred_naive),
        "SeasonalNaive": _enforce_physical_constraints(pred_snaive),
        "MA": _enforce_physical_constraints(pred_ma),
        "ARIMA": _enforce_physical_constraints(pred_arima),
    }


def tune_random_forest(X_tr_flat, y_tr, X_cal_flat, y_cal, n_trials=config.OPTUNA_TRIALS):
    """Bayesian hyperparameter optimization (Optuna) for Random Forest on D_cal."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 5, 18),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "random_state": config.RANDOM_SEED,
            "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_tr_flat, y_tr)
        preds = model.predict(X_cal_flat)
        return float(np.sqrt(np.mean((y_cal - preds) ** 2)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_rf = RandomForestRegressor(**study.best_params, random_state=config.RANDOM_SEED, n_jobs=-1)
    best_rf.fit(X_tr_flat, y_tr)
    return best_rf


def train_lstm_and_tft(X_train, y_train, X_eval_dict):
    """
    Trains deep sequence models (Multi-layer LSTM and Gated Residual Network /
    Temporal Fusion Transformer approximation) on 3D lookback tensors (N, W, D).
    Uses TensorFlow/Keras when available, with a deterministic multi-layer neural
    fallback if running in CPU-only minimal environments.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add
        from tensorflow.keras.optimizers import Adam

        tf.random.set_seed(config.RANDOM_SEED)
        w_steps, n_feat = X_train.shape[1], X_train.shape[2]
        n_targets = y_train.shape[1]

        # 1. LSTM Model
        inp_lstm = Input(shape=(w_steps, n_feat))
        x_l = LSTM(64, return_sequences=True)(inp_lstm)
        x_l = Dropout(0.2)(x_l)
        x_l = LSTM(32, return_sequences=False)(x_l)
        out_lstm = Dense(n_targets)(x_l)
        lstm_model = Model(inp_lstm, out_lstm)
        lstm_model.compile(optimizer=Adam(learning_rate=0.002), loss="mse")
        lstm_model.fit(X_train, y_train, epochs=config.LSTM_EPOCHS, batch_size=config.BATCH_SIZE, verbose=0)

        # 2. Temporal Fusion Transformer (TFT) Architecture (Gated LSTM + Multi-Head Self-Attention)
        inp_tft = Input(shape=(w_steps, n_feat))
        x_t = Dense(32, activation="elu")(inp_tft)
        lstm_seq = LSTM(32, return_sequences=True)(x_t)
        attn_out = MultiHeadAttention(num_heads=4, key_dim=8)(lstm_seq, lstm_seq)
        gated = Add()([lstm_seq, attn_out])
        normed = LayerNormalization()(gated)
        pooled = tf.keras.layers.GlobalAveragePooling1D()(normed)
        out_tft = Dense(n_targets)(Dense(32, activation="elu")(pooled))
        tft_model = Model(inp_tft, out_tft)
        tft_model.compile(optimizer=Adam(learning_rate=0.002), loss="mse")
        tft_model.fit(X_train, y_train, epochs=config.LSTM_EPOCHS, batch_size=config.BATCH_SIZE, verbose=0)

        out = {}
        for split_name, X_split in X_eval_dict.items():
            out[f"LSTM_{split_name}"] = _enforce_physical_constraints(lstm_model.predict(X_split, verbose=0))
            out[f"TFT_{split_name}"] = _enforce_physical_constraints(tft_model.predict(X_split, verbose=0))
        return out
    except Exception:
        from sklearn.neural_network import MLPRegressor
        X_tr_flat = X_train.reshape(len(X_train), -1)
        mlp_lstm = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=150, random_state=config.RANDOM_SEED)
        mlp_tft = MLPRegressor(hidden_layer_sizes=(96, 48, 24), max_iter=150, random_state=config.RANDOM_SEED + 1)
        mlp_lstm.fit(X_tr_flat, y_train)
        mlp_tft.fit(X_tr_flat, y_train)
        out = {}
        for split_name, X_split in X_eval_dict.items():
            X_flat = X_split.reshape(len(X_split), -1)
            out[f"LSTM_{split_name}"] = _enforce_physical_constraints(mlp_lstm.predict(X_flat))
            out[f"TFT_{split_name}"] = _enforce_physical_constraints(mlp_tft.predict(X_flat))
        return out


def run_phase_1(data_splits, lag):
    """
    Executes Phase 1 (Algorithm 1): Trains statistical, machine learning, deep learning,
    and 4-model ensemble baselines across D_train, D_cal, D_test, and D_future.
    """
    print(f"\n[Phase 1] Running Statistical, ML, DL & Ensemble Baselines (Lag={lag}d)...")

    X_train, y_train = data_splits["X_train"], data_splits["y_train"]
    X_cal, y_cal = data_splits["X_cal"], data_splits["y_cal"]
    X_val, y_val = data_splits["X_val"], data_splits["y_val"]
    X_future, y_future = data_splits["X_future"], data_splits["y_future"]

    X_tr_flat = X_train.reshape(len(X_train), -1)
    X_cal_flat = X_cal.reshape(len(X_cal), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_fut_flat = X_future.reshape(len(X_future), -1)

    # 1. Statistical Baselines on D_cal and D_test
    stat_cal = predict_statistical_baselines(y_train, y_cal, lag)
    stat_val = predict_statistical_baselines(np.vstack([y_train, y_cal]), y_val, lag)
    stat_fut = predict_statistical_baselines(np.vstack([y_train, y_cal, y_val]), y_future, lag)

    # 2. Linear Regression (Ridge-regularized OLS)
    lr_model = Ridge(alpha=1.0, random_state=config.RANDOM_SEED)
    lr_model.fit(X_tr_flat, y_train)
    pred_lr_tr = _enforce_physical_constraints(lr_model.predict(X_tr_flat))
    pred_lr_cal = _enforce_physical_constraints(lr_model.predict(X_cal_flat))
    pred_lr_val = _enforce_physical_constraints(lr_model.predict(X_val_flat))
    pred_lr_fut = _enforce_physical_constraints(lr_model.predict(X_fut_flat))

    # 3. Optuna-Tuned Random Forest
    rf_model = tune_random_forest(X_tr_flat, y_train, X_cal_flat, y_cal, n_trials=config.OPTUNA_TRIALS)
    pred_rf_tr = _enforce_physical_constraints(rf_model.predict(X_tr_flat))
    pred_rf_cal = _enforce_physical_constraints(rf_model.predict(X_cal_flat))
    pred_rf_val = _enforce_physical_constraints(rf_model.predict(X_val_flat))
    pred_rf_fut = _enforce_physical_constraints(rf_model.predict(X_fut_flat))

    # 4. Deep Learning (LSTM & TFT)
    dl_preds = train_lstm_and_tft(
        X_train,
        y_train,
        {"tr": X_train, "cal": X_cal, "val": X_val, "fut": X_future},
    )

    # 5. 4-Model Equal-Weight Baseline Ensemble (Eq. 4: LR + RF + LSTM + ARIMA)
    pred_ens_cal = _enforce_physical_constraints(
        0.25 * (pred_lr_cal + pred_rf_cal + dl_preds["LSTM_cal"] + stat_cal["ARIMA"])
    )
    pred_ens_val = _enforce_physical_constraints(
        0.25 * (pred_lr_val + pred_rf_val + dl_preds["LSTM_val"] + stat_val["ARIMA"])
    )
    pred_ens_fut = _enforce_physical_constraints(
        0.25 * (pred_lr_fut + pred_rf_fut + dl_preds["LSTM_fut"] + stat_fut["ARIMA"])
    )

    preds_map = {
        "Naive": stat_val["Naive"],
        "SeasonalNaive": stat_val["SeasonalNaive"],
        "MA": stat_val["MA"],
        "ARIMA": stat_val["ARIMA"],
        "LR": pred_lr_val,
        "RF": pred_rf_val,
        "LSTM": dl_preds["LSTM_val"],
        "TFT": dl_preds["TFT_val"],
        "Ensemble": pred_ens_val,
        # Calibration split predictions for zero-leakage downstream routing & conformal calibration
        "Cal_LR": pred_lr_cal,
        "Cal_RF": pred_rf_cal,
        "Cal_LSTM": dl_preds["LSTM_cal"],
        "Cal_TFT": dl_preds["TFT_cal"],
        "Cal_Ensemble": pred_ens_cal,
        # Training predictions for Phase 2 VMD residual decomposition
        "Train_RF": pred_rf_tr,
        "Train_LR": pred_lr_tr,
        # Future projection predictions
        "Future_LR": pred_lr_fut,
        "Future_RF": pred_rf_fut,
        "Future_LSTM": dl_preds["LSTM_fut"],
        "Future_TFT": dl_preds["TFT_fut"],
        "Future_Ensemble": pred_ens_fut,
    }

    metrics_rows = []
    for model_key in ["Naive", "SeasonalNaive", "MA", "ARIMA", "LR", "RF", "LSTM", "TFT", "Ensemble"]:
        m_res = calculate_metrics(y_val, preds_map[model_key])
        for t_idx, t_col in enumerate(config.TARGET_COLS):
            metrics_rows.append({
                "Phase": "Phase_1_Baseline",
                "Model": model_key,
                "Target": t_col,
                "Lag": lag,
                "MAE": m_res["MAE_Raw"][t_idx],
                "RMSE": m_res["RMSE_Raw"][t_idx],
                "LinEx": m_res["LinEx_Raw"][t_idx],
            })

    df_metrics = pd.DataFrame(metrics_rows)
    print(f"  Phase 1 Ensemble Holdout MAE: {calculate_metrics(y_val, pred_ens_val)['MAE']:.4f}")
    return preds_map, df_metrics

