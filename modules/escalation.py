"""Single-source-of-truth adaptive escalation decision pipeline."""
from modules.confidence import normalise_error, confidence_score, escalation_score, should_invoke_llm
from modules.intent import compute_qos_risk
from modules.qos import compute_qos, default_slice
from modules.llm import invoke_llm
from config import DEFAULT_THRESHOLD, TRAFFIC_SCALE_MBPS

def run_escalation(actual_traffic: float, predicted_traffic: float, anomaly_score: float,
                   latency_ms: float, bandwidth_mbps: float, reliability: float,
                   device_type: str, threshold: float = DEFAULT_THRESHOLD,
                   emergency: bool = False) -> dict:
    # The online decision can only use a forecast generated before observing actual_traffic.
    # actual_traffic is accepted for retrospective evaluation; callers must not use its
    # resulting error to decide the same time-step in a real-time deployment.
    norm_err = normalise_error(actual_traffic, predicted_traffic, TRAFFIC_SCALE_MBPS)
    confidence = confidence_score(norm_err)
    qos_risk = compute_qos_risk(device_type, latency_ms, bandwidth_mbps, reliability)
    esc_score = escalation_score(norm_err, anomaly_score, qos_risk)
    invoke = bool(emergency or should_invoke_llm(esc_score, threshold))
    qos_score = compute_qos(bandwidth_mbps, reliability, latency_ms, device_type)
    if invoke:
        context = {"device_type": device_type, "traffic_mbps": actual_traffic,
                   "predicted_mbps": predicted_traffic, "pred_error": norm_err,
                   "confidence": confidence, "anomaly_score": anomaly_score,
                   "escalation_score": esc_score, "latency_ms": latency_ms,
                   "bandwidth_mbps": bandwidth_mbps, "reliability": reliability,
                   "current_slice": device_type, "qos_score": qos_score, "emergency": emergency}
        slice_rec = invoke_llm(context)
    else:
        slice_rec = default_slice(device_type)
    return {"actual_traffic": actual_traffic, "predicted_traffic": predicted_traffic,
            "norm_error": norm_err, "confidence": confidence, "anomaly_score": anomaly_score,
            "qos_risk": qos_risk, "escalation_score": esc_score, "qos_score": qos_score,
            "llm_invoked": invoke, "slice": slice_rec, "threshold": threshold,
            "emergency": emergency}
