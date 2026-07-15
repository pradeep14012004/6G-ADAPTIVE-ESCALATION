"""
modules/anomaly_detector.py
Isolation Forest anomaly detection with normalised score output.
"""

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from config import (
    ISO_FOREST_PATH, IF_CONTAMINATION, IF_N_ESTIMATORS,
    IF_MAX_SAMPLES, RANDOM_SEED,
)


# ── Training ───────────────────────────────────────────────────────────────

def train_isolation_forest(X: np.ndarray) -> IsolationForest:
    """Fit Isolation Forest and persist model."""
    clf = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        max_samples=IF_MAX_SAMPLES,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X)
    ISO_FOREST_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, ISO_FOREST_PATH)
    return clf


# ── Inference ──────────────────────────────────────────────────────────────

def load_isolation_forest() -> IsolationForest:
    return joblib.load(ISO_FOREST_PATH)


def get_anomaly_score(clf: IsolationForest, x: np.ndarray) -> float:
    """
    Return normalised anomaly score in [0, 1].
    Isolation Forest decision_function returns negative scores for anomalies;
    we invert and normalise so that 1 = most anomalous.
    """
    raw = clf.decision_function(x.reshape(1, -1))[0]   # higher = more normal
    # Clip to a reasonable range then invert
    score = float(np.clip(-raw, 0, 1))
    return score


def get_anomaly_label(clf: IsolationForest, x: np.ndarray) -> int:
    """Return 1 if anomaly, 0 if normal."""
    pred = clf.predict(x.reshape(1, -1))[0]
    return int(pred == -1)


def batch_scores(clf: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Vectorised anomaly scores for an array of feature vectors."""
    raw = clf.decision_function(X)
    return np.clip(-raw, 0, 1)


def batch_labels(clf: IsolationForest, X: np.ndarray) -> np.ndarray:
    return (clf.predict(X) == -1).astype(int)
