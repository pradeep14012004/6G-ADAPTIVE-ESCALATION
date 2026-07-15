"""
modules/predictor.py
LSTM traffic predictor: build, train, evaluate, and inference.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import (
    LSTM_MODEL_PATH, SEQ_LEN,
    LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS, DROPOUT_RATE,
    BATCH_SIZE, MAX_EPOCHS, PATIENCE,
)


# ── Model definition ───────────────────────────────────────────────────────

def build_lstm() -> Sequential:
    """Stacked LSTM as specified in the research architecture."""
    model = Sequential([
        Input(shape=(SEQ_LEN, 1)),
        LSTM(LSTM_UNITS_1, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS_2, return_sequences=False),
        Dense(DENSE_UNITS, activation="relu"),
        Dense(1),
    ], name="AdaptiveEscalation_LSTM")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ── Training ───────────────────────────────────────────────────────────────

def train_lstm(X_tr, y_tr, X_val, y_val) -> tuple:
    """Train LSTM with callbacks; returns (model, history)."""
    model = build_lstm()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(str(LSTM_MODEL_PATH), save_best_only=True, monitor="val_loss"),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_lstm(model, X_te, y_te) -> dict:
    """Return RMSE, MAE, MAPE, R² on test set."""
    y_pred = model.predict(X_te, verbose=0).flatten()
    rmse   = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    mae    = float(mean_absolute_error(y_te, y_pred))
    mape   = float(np.mean(np.abs((y_te - y_pred) / (np.abs(y_te) + 1e-8))) * 100)
    r2     = float(r2_score(y_te, y_pred))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "y_pred": y_pred}


# ── Inference ──────────────────────────────────────────────────────────────

def load_lstm() -> Sequential:
    return tf.keras.models.load_model(str(LSTM_MODEL_PATH))


def predict_next(model, window: np.ndarray) -> float:
    """Predict next traffic value from a (SEQ_LEN,) window."""
    x = window.reshape(1, SEQ_LEN, 1)
    return float(model.predict(x, verbose=0)[0, 0])


def predict_horizon(model, seed_window: np.ndarray, steps: int) -> np.ndarray:
    """Auto-regressive multi-step forecast."""
    window = seed_window.copy().tolist()
    preds  = []
    for _ in range(steps):
        x   = np.array(window[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
        val = float(model.predict(x, verbose=0)[0, 0])
        preds.append(val)
        window.append(val)
    return np.array(preds)
