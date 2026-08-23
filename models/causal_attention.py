"""
PACI-ICPDA Framework - Phase 4: Deep Causal Attention Hybrid
Restricts the input space of an LSTM network to identified causal parent features
and incorporates a Bahdanau-style attention mechanism.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Layer
from tensorflow.keras.optimizers import Adam
import config
from utils.metrics import calculate_metrics

class BahdanauAttention(Layer):
    """
    Bahdanau (additive) attention mechanism to score importance of
    historical time-steps dynamically based on causal inputs.
    """
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W1 = self.add_weight(name="attention_w1", shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        self.W2 = self.add_weight(name="attention_w2", shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        self.V = self.add_weight(name="attention_v", shape=(self.units, 1), initializer="glorot_uniform", trainable=True)
        super(BahdanauAttention, self).build(input_shape)
        
    def call(self, x):
        # Calculate attention scores
        # Score shape: (batch_size, time_steps, 1)
        score = tf.tanh(tf.tensordot(x, self.W1, axes=1) + tf.tensordot(x, self.W2, axes=1))
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.V, axes=1), axis=1)
        
        # Context vector shape: (batch_size, features)
        context_vector = x * attention_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
        
    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({"units": self.units})
        return config

def build_causal_attention_model(lag, n_features):
    """Compiles the Bahdanau Attention LSTM hybrid network."""
    inp = Input(shape=(lag, n_features))
    
    # LSTM layer to process temporal features
    lstm_out = LSTM(32, return_sequences=True)(inp)
    
    # Attention context
    context, weights = BahdanauAttention(units=16)(lstm_out)
    
    # Fully-connected layers
    x = Dense(16, activation="relu")(context)
    out = Dense(1)(x)
    
    model = Model(inp, out)
    model.compile(optimizer=Adam(learning_rate=0.005), loss="mse")
    return model

def run_phase_4(data_splits, causal_parents, lag):
    """
    Executes Phase 4: Restricts features to PCMCI causal parents and trains the attention hybrid.
    """
    print(f"\n[Phase 4] Running Deep Causal Attention Hybrid (Lag={lag}d)...")
    
    X_train = data_splits["X_train"]
    y_train = data_splits["y_train"]
    X_val = data_splits["X_val"]
    y_val = data_splits["y_val"]
    X_future = data_splits["X_future"]
    
    n_targets = y_train.shape[1]
    
    pred_attention_val = np.zeros_like(y_val)
    pred_attention_fut = np.zeros_like(data_splits["y_future"])
    
    # Train target-specific networks restricted to their respective causal parents
    for t_idx, var in enumerate(config.TARGET_COLS):
        parents = causal_parents[var]
        parent_indices = [config.FEATURES.index(p) for p in parents]
        
        X_tr_filtered = X_train[:, :, parent_indices]
        X_va_filtered = X_val[:, :, parent_indices]
        X_fu_filtered = X_future[:, :, parent_indices]
        
        model = build_causal_attention_model(lag, len(parent_indices))
        
        # Train with early validation monitoring
        model.fit(
            X_tr_filtered, y_train[:, t_idx],
            epochs=config.LSTM_EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_data=(X_va_filtered, y_val[:, t_idx]),
            verbose=0
        )
        
        # Predict
        pred_attention_val[:, t_idx] = model.predict(X_va_filtered).squeeze()
        pred_attention_fut[:, t_idx] = model.predict(X_fu_filtered).squeeze()
        
    # Apply constraints and integer rounding
    pred_attention_val = np.maximum(pred_attention_val, 0.0)
    pred_attention_val[:, 1] = np.round(pred_attention_val[:, 1])
    
    pred_attention_fut = np.maximum(pred_attention_fut, 0.0)
    pred_attention_fut[:, 1] = np.round(pred_attention_fut[:, 1])
    
    # Calculate performance metrics
    metrics_attention = calculate_metrics(y_val, pred_attention_val)
    print(f"  Causal Attention LSTM MAE: {metrics_attention['MAE']:.4f}")
    
    # Save outputs
    out_dir = os.path.join(config.OUTPUT_DIR, f"LAG_{lag}", "Phase_4_CausalAttention")
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "val_pred_CausalAttention.npy"), pred_attention_val)
    np.save(os.path.join(out_dir, "fut_pred_CausalAttention.npy"), pred_attention_fut)
    
    metrics_summary = []
    for t_idx, t_col in enumerate(config.TARGET_COLS):
        metrics_summary.append({
            "Phase": "Phase_4_CausalAttention",
            "Model": "Causal_Attention_LSTM",
            "Target": t_col,
            "Lag": lag,
            "MAE": metrics_attention["MAE_Raw"][t_idx],
            "RMSE": metrics_attention["RMSE_Raw"][t_idx]
        })
        
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    
    preds_map = {
        "CausalAttention": pred_attention_val,
        "Future_CausalAttention": pred_attention_fut
    }
    
    return preds_map, df_metrics
