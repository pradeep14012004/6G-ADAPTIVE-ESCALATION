"""
modules/confidence.py
Confidence score and multi-factor Adaptive Escalation Score.

Escalation Score:  S = 0.5*E_t + 0.3*A_t + 0.2*Q_t
Confidence:        C = exp(-lambda * E_t)
"""

import numpy as np
from config import LAMBDA_DECAY, W_ERROR, W_ANOMALY, W_QOS


def normalise_error(actual: float, predicted: float, max_val: float = 1.0) -> float:
    """Normalised absolute prediction error in [0, 1]."""
    return float(np.clip(abs(actual - predicted) / (max_val + 1e-8), 0, 1))


def confidence_score(norm_error: float, lam: float = LAMBDA_DECAY) -> float:
    """C = exp(-lambda * E_t)  →  high confidence when error is small."""
    return float(np.exp(-lam * norm_error))


def escalation_score(
    norm_error: float,
    anomaly_score: float,
    qos_risk: float,
    w_e: float = W_ERROR,
    w_a: float = W_ANOMALY,
    w_q: float = W_QOS,
) -> float:
    """
    Multi-factor Adaptive Escalation Score.
    S = w_e*E_t + w_a*A_t + w_q*Q_t  ∈ [0, 1]
    """
    s = w_e * norm_error + w_a * anomaly_score + w_q * qos_risk
    return float(np.clip(s, 0, 1))


def should_invoke_llm(score: float, threshold: float) -> bool:
    """Trigger LLM when S > τ."""
    return score > threshold
