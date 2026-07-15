"""
modules/escalation.py
Adaptive Escalation Engine — orchestrates the full decision pipeline.
"""

import numpy as np
from modules.confidence import normalise_error, confidence_score, escalation_score, should_invoke_llm
from modules.intent import compute_qos_risk
from modules.qos import compute_qos, default_slice
from modules.llm import invoke_llm
from config import DEFAULT_THRESHOLD


def run_escalation(
    actual_traffic: float,
    predicted_traffic: float,
    anomaly_score: float,
    latency_ms: float,
    bandwidth_mbps: float,
    reliability: float,
    device_type: str,
    threshold: float = DEFAULT_THRESHOLD,
    emergency: bool = False,
) -> dict:
    """
    Full adaptive escalation pipeline for one time-step.

    Returns a result dict with all intermediate values and the final
    slice recommendation.
    """
    # ── Step 1: Prediction error & confidence ──────────────────────────────
    norm_err   = normalise_error(actual_traffic, predicted_traffic, max_val=1.0)
    confidence = confidence_score(norm_err)

    # ── Step 2: QoS risk ───────────────────────────────────────────────────
    qos_risk = compute_qos_risk(device_type, latency_ms, bandwidth_mbps, reliability)

    # ── Step 3: Multi-factor escalation score ──────────────────────────────
    esc_score = escalation_score(norm_err, anomaly_score, qos_risk)

    # Emergency override always triggers LLM
    invoke = should_invoke_llm(esc_score, threshold) or emergency

    # ── Step 4: QoS score ──────────────────────────────────────────────────
    qos_score = compute_qos(bandwidth_mbps, reliability, latency_ms, device_type)

    # ── Step 5: Slice decision ─────────────────────────────────────────────
    if invoke:
        context = {
            "device_type":      device_type,
            "traffic_mbps":     actual_traffic,
            "predicted_mbps":   predicted_traffic,
            "pred_error":       norm_err,
            "confidence":       confidence,
            "anomaly_score":    anomaly_score,
            "escalation_score": esc_score,
            "latency_ms":       latency_ms,
            "bandwidth_mbps":   bandwidth_mbps,
            "reliability":      reliability,
            "current_slice":    device_type,
            "qos_score":        qos_score,
            "emergency":        emergency,
        }
        slice_rec = invoke_llm(context)
    else:
        slice_rec = default_slice(device_type)

    return {
        "actual_traffic":    actual_traffic,
        "predicted_traffic": predicted_traffic,
        "norm_error":        norm_err,
        "confidence":        confidence,
        "anomaly_score":     anomaly_score,
        "qos_risk":          qos_risk,
        "escalation_score":  esc_score,
        "qos_score":         qos_score,
        "llm_invoked":       invoke,
        "slice":             slice_rec,
        "threshold":         threshold,
        "emergency":         emergency,
    }
