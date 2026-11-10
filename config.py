"""Central configuration for the 6G adaptive-escalation research prototype."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
DATA_PATH = DATA_DIR / "healthcare_network.csv"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.keras"
ISO_FOREST_PATH = MODEL_DIR / "isolation_forest.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "optimal_threshold.json"

# Synthetic experiment defaults. Report actual generated sample count in every run.
N_SAMPLES = 10_000
RANDOM_SEED = 42
SEQ_LEN = 20
LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS = 64, 32, 16
DROPOUT_RATE, BATCH_SIZE, MAX_EPOCHS, PATIENCE = 0.2, 32, 100, 10
TEST_SPLIT, VALIDATION_SPLIT = 0.2, 0.1
TRAFFIC_SCALE_MBPS = 200.0

IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = "auto"
THRESHOLD_SEARCH_LOW, THRESHOLD_SEARCH_HIGH, THRESHOLD_STEPS = 0.05, 0.95, 91
DEFAULT_THRESHOLD = 0.55
W_ERROR, W_ANOMALY, W_QOS = 0.5, 0.3, 0.2
LAMBDA_DECAY = 1.0
ALPHA_RESOURCE = 1.2
W1_BW, W2_REL, W3_LAT = 0.4, 0.35, 0.25
ALPHA_COST, BETA_COST, GAMMA_COST, DELTA_COST = 0.30, 0.25, 0.30, 0.15

QOS_PROFILES = {
    "Remote Surgery": {"latency_ms": 1.0, "reliability": 99.999, "bandwidth_mbps": 150.0, "priority": "Critical"},
    "ICU Monitoring": {"latency_ms": 5.0, "reliability": 99.99, "bandwidth_mbps": 80.0, "priority": "High"},
    "Wearable Devices": {"latency_ms": 20.0, "reliability": 99.9, "bandwidth_mbps": 20.0, "priority": "Medium"},
    "Emergency Ambulance": {"latency_ms": 2.0, "reliability": 99.999, "bandwidth_mbps": 100.0, "priority": "Critical"},
}
MQTT_BROKER, MQTT_PORT = "localhost", 1883
MQTT_ANOMALY_ALERT = 0.6
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TIMEOUT = 30
DEFAULT_DEVICE, DEFAULT_TRAFFIC, DEFAULT_PRED_WIN, DEFAULT_EMERGENCY = "ICU Monitoring", 60.0, 10, False
