"""Deterministic safety-first slice policy with one configurable escalation threshold."""
import time
from config import QOS_PROFILES, DEFAULT_THRESHOLD, MQTT_ANOMALY_ALERT

_isolated: dict[str, float] = {}
ISOLATE_RECOVER_SEC = 20
LLM_THRESHOLD = DEFAULT_THRESHOLD
ANOMALY_THRESH = MQTT_ANOMALY_ALERT

def evaluate(device_id: str, device_type: str, priority: int, bandwidth: float,
             latency: float, predicted_bw: float, anomaly_score: float, esc_score: float,
             congested: bool, total_usage: float, network_capacity: float = 500.0,
             emergency: bool = False) -> dict:
    if network_capacity <= 0:
        raise ValueError("network_capacity must be positive")
    for name, value in (("anomaly_score", anomaly_score), ("esc_score", esc_score)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    profile = QOS_PROFILES.get(device_type, QOS_PROFILES["ICU Monitoring"])
    utilization = max(0.0, total_usage) / network_capacity
    allocation = {"allocated_bandwidth": profile["bandwidth_mbps"],
                  "allocated_latency": profile["latency_ms"], "priority": profile["priority"]}
    action, severity, reason = "NO_ACTION", "LOW", "Normal operation"
    invoke_llm = bool(emergency)

    if device_id in _isolated:
        if anomaly_score < ANOMALY_THRESH and time.time() - _isolated[device_id] >= ISOLATE_RECOVER_SEC:
            del _isolated[device_id]
            return {"action": "RECOVER_DEVICE", "severity": "LOW",
                    "reason": f"{device_id} anomaly cleared — restoring normal allocation",
                    "invoke_llm": False, "allocation": allocation}
        if anomaly_score >= ANOMALY_THRESH:
            return {"action": "ISOLATE_DEVICE", "severity": "CRITICAL",
                    "reason": f"Persistent anomaly score {anomaly_score:.3f} on {device_id}",
                    "invoke_llm": True, "allocation": {
                        "allocated_bandwidth": max(profile["bandwidth_mbps"] * 0.2, 1.0),
                        "allocated_latency": profile["latency_ms"] * 2, "priority": "Low"}}

    # Safety actions take precedence over ordinary congestion policies.
    if emergency or (device_type == "Emergency Ambulance" and bandwidth > 20):
        action, severity, reason = "RESERVE_CAPACITY", "CRITICAL", "Emergency traffic — reserve URLLC capacity"
        allocation = {"allocated_bandwidth": profile["bandwidth_mbps"],
                      "allocated_latency": profile["latency_ms"], "priority": "Critical"}
        invoke_llm = invoke_llm or emergency
    elif anomaly_score >= ANOMALY_THRESH:
        _isolated[device_id] = time.time()
        action, severity, reason, invoke_llm = "ISOLATE_DEVICE", "CRITICAL", f"Anomaly score {anomaly_score:.3f} — isolate device", True
        allocation = {"allocated_bandwidth": max(profile["bandwidth_mbps"] * 0.2, 1.0),
                      "allocated_latency": profile["latency_ms"] * 2, "priority": "Low"}
    elif device_type == "Remote Surgery" and predicted_bw > bandwidth * 1.15:
        action, severity, reason = "RESERVE_CAPACITY", "HIGH", f"Surgery traffic forecast rising to {predicted_bw:.1f} Mbps"
        allocation = {"allocated_bandwidth": min(max(predicted_bw * 1.2, profile["bandwidth_mbps"]), 200.0),
                      "allocated_latency": profile["latency_ms"], "priority": "Critical"}
    elif device_type == "ICU Monitoring" and latency > profile["latency_ms"]:
        action, severity, reason = "INCREASE_PRIORITY", "HIGH", f"ICU latency {latency:.1f} ms exceeds target"
    elif congested and device_type == "Wearable Devices":
        action, severity, reason = "THROTTLE", "MEDIUM", f"Network utilization {utilization:.1%}; throttle non-critical wearable traffic"
        allocation = {"allocated_bandwidth": min(max(bandwidth * 0.3, 1.0), 3.0),
                      "allocated_latency": profile["latency_ms"] * 2, "priority": "Low"}

    if esc_score >= LLM_THRESHOLD and congested and priority >= 6 and action not in {"ISOLATE_DEVICE", "RESERVE_CAPACITY"}:
        invoke_llm = True
        action, severity, reason = "LLM_ESCALATION", "CRITICAL", f"Escalation score {esc_score:.3f} exceeds threshold under congestion"
    return {"action": action, "severity": severity, "reason": reason,
            "invoke_llm": invoke_llm, "allocation": allocation}
