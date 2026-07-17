"""
modules/policy_engine.py
Rule-based decisions for routine cases.
LLM reserved for esc > 0.9 AND congestion conflict.
"""
import numpy as np
from config import QOS_PROFILES

LLM_THRESHOLD  = 0.9
ANOMALY_THRESH = 0.6

# Track which devices are currently isolated so we can auto-recover
_isolated: dict[str, float] = {}   # device_id → timestamp of isolation
ISOLATE_RECOVER_SEC = 20


def evaluate(
    device_id: str,
    device_type: str,
    priority: int,
    bandwidth: float,
    latency: float,
    predicted_bw: float,
    anomaly_score: float,
    esc_score: float,
    congested: bool,
    total_usage: float,
    network_capacity: float = 500.0,
) -> dict:
    import time
    profile     = QOS_PROFILES.get(device_type, QOS_PROFILES["ICU Monitoring"])
    utilization = total_usage / network_capacity

    action     = "NO_ACTION"
    severity   = "LOW"
    reason     = "Normal operation"
    invoke_llm = False
    allocation = {
        "allocated_bandwidth": profile["bandwidth_mbps"],
        "allocated_latency":   profile["latency_ms"],
        "priority":            profile["priority"],
    }

    # ── Auto-recovery: device was isolated, anomaly cleared ───────────────
    if device_id in _isolated and anomaly_score < ANOMALY_THRESH:
        elapsed = time.time() - _isolated[device_id]
        if elapsed > ISOLATE_RECOVER_SEC:
            del _isolated[device_id]
            action   = "RECOVER_DEVICE"
            severity = "LOW"
            reason   = f"{device_id} anomaly cleared — restoring normal allocation"
            allocation = {
                "allocated_bandwidth": profile["bandwidth_mbps"],
                "allocated_latency":   profile["latency_ms"],
                "priority":            profile["priority"],
            }
            return {"action": action, "severity": severity, "reason": reason,
                    "invoke_llm": False, "allocation": allocation}

    # ── Rule 1: Emergency device ───────────────────────────────────────────
    if device_type == "Emergency Ambulance" and bandwidth > 20:
        action     = "RESERVE_CAPACITY"
        severity   = "CRITICAL"
        reason     = "Ambulance emergency burst — reserving URLLC capacity"
        allocation = {"allocated_bandwidth": 100.0, "allocated_latency": 2.0, "priority": "Critical"}

    # ── Rule 2: Surgery traffic rising ────────────────────────────────────
    elif device_type == "Remote Surgery" and predicted_bw > bandwidth * 1.15:
        action     = "RESERVE_CAPACITY"
        severity   = "HIGH"
        reason     = f"Surgery traffic rising to {predicted_bw:.1f} Mbps — pre-allocating"
        allocation = {"allocated_bandwidth": min(predicted_bw * 1.2, 150.0),
                      "allocated_latency": 1.0, "priority": "Critical"}

    # ── Rule 3: ICU latency spike ─────────────────────────────────────────
    elif device_type == "ICU Monitoring" and latency > 5:
        action     = "INCREASE_PRIORITY"
        severity   = "HIGH"
        reason     = f"ICU latency {latency:.1f}ms exceeds 5ms threshold"
        allocation = {"allocated_bandwidth": 80.0, "allocated_latency": 5.0, "priority": "High"}

    # ── Rule 4: Network congested — throttle wearables ────────────────────
    elif congested and device_type == "Wearable Devices":
        action     = "THROTTLE"
        severity   = "MEDIUM"
        reason     = f"Network at {utilization*100:.0f}% — throttling wearables"
        allocation = {"allocated_bandwidth": min(bandwidth * 0.3, 3.0),
                      "allocated_latency": 40.0, "priority": "Low"}

    # ── Rule 5: Anomaly — isolate device, call LLM immediately ────────────
    elif anomaly_score > ANOMALY_THRESH:
        _isolated[device_id] = time.time()
        action     = "ISOLATE_DEVICE"
        severity   = "CRITICAL"
        reason     = f"Anomaly score {anomaly_score:.3f} on {device_id} — isolating"
        invoke_llm = True   # always call LLM for anomalies
        # Cap bandwidth to 20% of normal, slow publish, drop priority
        allocation = {
            "allocated_bandwidth": max(profile["bandwidth_mbps"] * 0.2, 1.0),
            "allocated_latency":   profile["latency_ms"] * 2,
            "priority":            "Low",
        }

    # ── Rule 6: Escalate to LLM — high esc + congestion ──────────────────
    if esc_score > LLM_THRESHOLD and congested and priority >= 6:
        invoke_llm = True
        action     = "LLM_ESCALATION"
        severity   = "CRITICAL"
        reason     = (f"esc={esc_score:.3f} > {LLM_THRESHOLD} with congestion "
                      f"({utilization*100:.0f}%) — escalating to LLM")

    return {
        "action":     action,
        "severity":   severity,
        "reason":     reason,
        "invoke_llm": invoke_llm,
        "allocation": allocation,
    }
