"""Calibrated error/confidence and adaptive escalation score utilities."""
import numpy as np
from config import LAMBDA_DECAY, W_ERROR, W_ANOMALY, W_QOS, TRAFFIC_SCALE_MBPS

def normalise_error(actual: float, predicted: float, max_val: float = TRAFFIC_SCALE_MBPS) -> float:
    """Absolute forecast error normalized by a documented traffic scale (Mbps)."""
    if not np.isfinite(actual) or not np.isfinite(predicted):
        raise ValueError("actual and predicted traffic must be finite")
    if max_val <= 0:
        raise ValueError("max_val must be positive")
    return float(np.clip(abs(float(actual) - float(predicted)) / max_val, 0.0, 1.0))

def confidence_score(norm_error: float, lam: float = LAMBDA_DECAY) -> float:
    if not 0 <= norm_error <= 1 or lam < 0:
        raise ValueError("norm_error must be in [0,1] and lambda non-negative")
    return float(np.exp(-lam * norm_error))

def escalation_score(norm_error: float, anomaly_score: float, qos_risk: float,
                     w_e: float = W_ERROR, w_a: float = W_ANOMALY, w_q: float = W_QOS) -> float:
    values = np.asarray([norm_error, anomaly_score, qos_risk], dtype=float)
    weights = np.asarray([w_e, w_a, w_q], dtype=float)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("all escalation inputs must be finite values in [0,1]")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("escalation weights must be non-negative and sum to 1")
    return float(np.dot(values, weights))

def should_invoke_llm(score: float, threshold: float) -> bool:
    if not np.isfinite(score) or not 0 <= score <= 1 or not 0 <= threshold <= 1:
        raise ValueError("score and threshold must be in [0,1]")
    return bool(score >= threshold)
