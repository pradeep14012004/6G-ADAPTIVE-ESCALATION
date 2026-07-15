"""
config.py — Central configuration for Adaptive Escalation 6G Framework
IEEE Software Architecture: All tunable parameters in one place.
No hardcoded values in source modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

DATA_PATH          = DATA_DIR  / "healthcare_network.csv"
LSTM_MODEL_PATH    = MODEL_DIR / "lstm_model.keras"
ISO_FOREST_PATH    = MODEL_DIR / "isolation_forest.pkl"
SCALER_PATH        = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH     = MODEL_DIR / "optimal_threshold.json"

# ── Dataset generation ─────────────────────────────────────────────────────
N_SAMPLES        = 5000
RANDOM_SEED      = 42

# ── LSTM ───────────────────────────────────────────────────────────────────
SEQ_LEN          = 20          # look-back window
LSTM_UNITS_1     = 64
LSTM_UNITS_2     = 32
DENSE_UNITS      = 16
DROPOUT_RATE     = 0.2
BATCH_SIZE       = 32
MAX_EPOCHS       = 100
PATIENCE         = 10          # EarlyStopping patience
TEST_SPLIT       = 0.2

# ── Isolation Forest ───────────────────────────────────────────────────────
IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS  = 200
IF_MAX_SAMPLES   = "auto"

# ── Threshold optimisation ─────────────────────────────────────────────────
THRESHOLD_SEARCH_LOW  = 0.40
THRESHOLD_SEARCH_HIGH = 0.95
THRESHOLD_STEPS       = 56     # granularity of grid search
DEFAULT_THRESHOLD     = 0.65   # fallback if file not found

# ── Adaptive Escalation Score weights ─────────────────────────────────────
W_ERROR   = 0.5   # prediction error weight
W_ANOMALY = 0.3   # anomaly score weight
W_QOS     = 0.2   # QoS risk weight

# ── Confidence decay ───────────────────────────────────────────────────────
LAMBDA_DECAY = 1.0   # C = exp(-lambda * normalised_error)

# ── Resource allocation ────────────────────────────────────────────────────
ALPHA_RESOURCE = 1.2   # R = alpha * predicted_traffic

# ── QoS weights ───────────────────────────────────────────────────────────
W1_BW   = 0.4
W2_REL  = 0.35
W3_LAT  = 0.25

# ── Cost function weights ──────────────────────────────────────────────────
ALPHA_COST  = 0.30   # latency
BETA_COST   = 0.25   # energy
GAMMA_COST  = 0.30   # LLM calls
DELTA_COST  = 0.15   # resource waste

# ── QoS profiles (intent → requirements) ──────────────────────────────────
QOS_PROFILES = {
    "Remote Surgery": {
        "latency_ms":    1.0,
        "reliability":   99.999,
        "bandwidth_mbps": 150.0,
        "priority":      "Critical",
    },
    "ICU Monitoring": {
        "latency_ms":    5.0,
        "reliability":   99.99,
        "bandwidth_mbps": 80.0,
        "priority":      "High",
    },
    "Wearable Devices": {
        "latency_ms":    20.0,
        "reliability":   99.9,
        "bandwidth_mbps": 20.0,
        "priority":      "Medium",
    },
    "Emergency Ambulance": {
        "latency_ms":    2.0,
        "reliability":   99.999,
        "bandwidth_mbps": 100.0,
        "priority":      "Critical",
    },
}

# ── Groq LLM ───────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama3-8b-8192")
GROQ_TIMEOUT  = 30   # seconds

# ── Streamlit UI defaults ──────────────────────────────────────────────────
DEFAULT_DEVICE    = "ICU Monitoring"
DEFAULT_TRAFFIC   = 60.0
DEFAULT_PRED_WIN  = 10
DEFAULT_EMERGENCY = False
