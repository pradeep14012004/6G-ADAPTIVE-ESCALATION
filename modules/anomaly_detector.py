"""Isolation Forest with training-distribution score calibration."""
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from config import ISO_FOREST_PATH, IF_CONTAMINATION, IF_N_ESTIMATORS, IF_MAX_SAMPLES, RANDOM_SEED

def train_isolation_forest(X: np.ndarray) -> IsolationForest:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 10 or not np.isfinite(X).all():
        raise ValueError("X must be a finite 2-D training matrix with at least 10 rows")
    clf = IsolationForest(n_estimators=IF_N_ESTIMATORS, contamination=IF_CONTAMINATION,
                          max_samples=IF_MAX_SAMPLES, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X)
    # Persist empirical score bounds; decision_function is higher for normal points.
    train_scores = clf.decision_function(X)
    clf.score_lower_ = float(np.quantile(train_scores, 0.01))
    clf.score_upper_ = float(np.quantile(train_scores, 0.99))
    ISO_FOREST_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, ISO_FOREST_PATH)
    return clf

def load_isolation_forest() -> IsolationForest:
    return joblib.load(ISO_FOREST_PATH)

def get_anomaly_score(clf: IsolationForest, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(1, -1)
    raw = float(clf.decision_function(x)[0])
    low = float(getattr(clf, "score_lower_", -0.5))
    high = float(getattr(clf, "score_upper_", 0.5))
    if not np.isfinite(raw) or high <= low:
        raise ValueError("invalid Isolation Forest score calibration")
    return float(np.clip((high - raw) / (high - low), 0.0, 1.0))

def get_anomaly_label(clf: IsolationForest, x: np.ndarray) -> int:
    return int(clf.predict(np.asarray(x).reshape(1, -1))[0] == -1)

def batch_scores(clf: IsolationForest, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    raw = clf.decision_function(X)
    low = float(getattr(clf, "score_lower_", -0.5))
    high = float(getattr(clf, "score_upper_", 0.5))
    if high <= low:
        raise ValueError("invalid Isolation Forest score calibration")
    return np.clip((high - raw) / (high - low), 0.0, 1.0)

def batch_labels(clf: IsolationForest, X: np.ndarray) -> np.ndarray:
    return (clf.predict(np.asarray(X, dtype=float)) == -1).astype(int)
