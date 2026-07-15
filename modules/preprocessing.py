"""
modules/preprocessing.py
Data generation, cleaning, feature engineering, and scaling.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from config import (
    DATA_PATH, SCALER_PATH, N_SAMPLES, RANDOM_SEED, SEQ_LEN, TEST_SPLIT
)


# ── Synthetic dataset ──────────────────────────────────────────────────────

def generate_dataset(n: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate realistic synthetic healthcare network traffic dataset."""
    rng = np.random.default_rng(seed)
    t   = np.arange(n)

    # Base traffic: diurnal pattern + weekly seasonality
    diurnal  = 30 * np.sin(2 * np.pi * t / 144)          # 10-min intervals → 1 day
    weekly   = 10 * np.sin(2 * np.pi * t / (144 * 7))
    trend    = 0.005 * t
    noise    = rng.normal(0, 5, n)
    traffic  = 60 + diurnal + weekly + trend + noise
    traffic  = np.clip(traffic, 5, 200)

    # Inject anomalies (~5 %)
    anomaly_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    traffic[anomaly_idx] += rng.uniform(40, 80, len(anomaly_idx))
    traffic = np.clip(traffic, 5, 200)

    latency     = 1 + 0.05 * traffic + rng.normal(0, 0.5, n)
    bandwidth   = 0.8 * traffic + rng.normal(0, 3, n)
    packet_loss = np.clip(0.001 * traffic + rng.normal(0, 0.05, n), 0, 1)
    reliability = np.clip(100 - packet_loss * 10, 90, 100)
    jitter      = 0.1 * latency + rng.normal(0, 0.1, n)

    device_types = rng.choice(
        ["Remote Surgery", "ICU Monitoring", "Wearable Devices", "Emergency Ambulance"],
        size=n, p=[0.15, 0.35, 0.35, 0.15]
    )

    df = pd.DataFrame({
        "timestamp":   pd.date_range("2024-01-01", periods=n, freq="10min"),
        "traffic_mbps": traffic,
        "latency_ms":   latency,
        "bandwidth_mbps": bandwidth,
        "packet_loss":  packet_loss,
        "reliability":  reliability,
        "jitter_ms":    jitter,
        "device_type":  device_types,
        "anomaly_label": 0,
    })
    df.loc[anomaly_idx, "anomaly_label"] = 1
    return df


# ── Cleaning ───────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and type casting."""
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.reset_index(drop=True)


# ── Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal and rolling features."""
    df = df.copy()
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_peak"]     = ((df["hour"] >= 8) & (df["hour"] <= 20)).astype(int)

    for w in [6, 12, 24]:
        df[f"traffic_roll_mean_{w}"] = (
            df["traffic_mbps"].rolling(w, min_periods=1).mean()
        )
        df[f"traffic_roll_std_{w}"] = (
            df["traffic_mbps"].rolling(w, min_periods=1).std().fillna(0)
        )

    df["traffic_diff1"] = df["traffic_mbps"].diff().fillna(0)
    df["traffic_diff2"] = df["traffic_diff1"].diff().fillna(0)
    return df


# ── Scaling ────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "traffic_mbps", "latency_ms", "bandwidth_mbps",
    "packet_loss", "reliability", "jitter_ms",
    "hour", "day_of_week", "is_peak",
    "traffic_roll_mean_6", "traffic_roll_std_6",
    "traffic_roll_mean_12", "traffic_roll_std_12",
    "traffic_roll_mean_24", "traffic_roll_std_24",
    "traffic_diff1", "traffic_diff2",
]


def scale_features(df: pd.DataFrame, fit: bool = True):
    """Scale features; fit=True trains a new scaler, False loads saved."""
    if fit:
        scaler = MinMaxScaler()
        df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    return df, scaler


# ── Sequence builder ───────────────────────────────────────────────────────

def build_sequences(series: np.ndarray, seq_len: int = SEQ_LEN):
    """Convert 1-D array to (X, y) supervised sequences."""
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i: i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X)[..., np.newaxis], np.array(y)


# ── Train/test split ───────────────────────────────────────────────────────

def train_test_split_ts(X: np.ndarray, y: np.ndarray, test_ratio: float = TEST_SPLIT):
    """Chronological split (no shuffle)."""
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


# ── Pipeline ───────────────────────────────────────────────────────────────

def run_pipeline(save: bool = True):
    """End-to-end preprocessing pipeline."""
    df = generate_dataset()
    df = clean_data(df)
    df = engineer_features(df)
    df, scaler = scale_features(df, fit=True)
    if save:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    traffic = df["traffic_mbps"].values
    X, y    = build_sequences(traffic)
    X_tr, X_te, y_tr, y_te = train_test_split_ts(X, y)
    return df, scaler, X_tr, X_te, y_tr, y_te
