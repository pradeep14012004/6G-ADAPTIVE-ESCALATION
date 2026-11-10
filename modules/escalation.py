"""Single-source-of-truth adaptive escalation decision pipeline."""
from modules.confidence import normalise_error, confidence_score, escalation_score, should_invoke_llm
from modules.intent import compute_qos_risk
from modules.qos import compute_qos, default_slice
from modules.llm import invoke_llm
from config import DEFAULT_THRESHOLD, TRAFFIC_SCALE_MBPS

def run_escalation(actual_traffic: float, predicted_traffic: float, anomaly_score: float,
                   latency_ms: float, bandwidth_mbps: float, reliability: float,
                   device_type: str, threshold: float = DEFAULT_THRESHOLD,
                   emergency: bool = False, prior_norm_error: float = 0.0) -> dict:
    """Make a causal online decision from the previous forecast residual.

    actual_traffic is used only to calculate an evaluation residual after the
    decision. A live controller cannot know the current observation before
    choosing its action. prior_norm_error must be computed from a forecast
    whose target has already been observed at an earlier time step.
    """
    if not 0 <= prior_norm_error <= 1:
        raise ValueError("prior_norm_error must be in [0,1]")
    observed_error = normalise_error(actual_traffic, predicted_traffic, TRAFFIC_SCALE_MBPS)
    confidence = confidence_score(prior_norm_error)
    qos_risk = compute_qos_risk(device_type, latency_ms, bandwidth_mbps, reliability)
    esc_score = escalation_score(prior_norm_error, anomaly_score, qos_risk)
    invoke = bool(emergency or should_invoke_llm(esc_score, threshold))
    qos_score = compute_qos(bandwidth_mbps, reliability, latency_ms, device_type)
    if invoke:
        context = {"device_type": device_type, "traffic_mbps": actual_traffic,
                   "predicted_mbps": predicted_traffic, "pred_error": prior_norm_error,
                   "confidence": confidence, "anomaly_score": anomaly_score,
                   "escalation_score": esc_score, "latency_ms": latency_ms,
                   "bandwidth_mbps": bandwidth_mbps, "reliability": reliability,
                   "current_slice": device_type, "qos_score": qos_score, "emergency": emergency}
        slice_rec = invoke_llm(context)
    else:
        slice_rec = default_slice(device_type)
    return {"actual_traffic": actual_traffic, "predicted_traffic": predicted_traffic,
            "observed_error_for_evaluation_only": observed_error,
            "prior_norm_error_used_for_decision": prior_norm_error,
            "confidence": confidence, "anomaly_score": anomaly_score,
            "qos_risk": qos_risk, "escalation_score": esc_score, "qos_score": qos_score,
            "llm_invoked": invoke, "slice": slice_rec, "threshold": threshold,
            "emergency": emergency}
