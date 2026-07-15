"""
modules/qos.py
QoS score computation and slice state management.
"""

from config import W1_BW, W2_REL, W3_LAT, QOS_PROFILES


def compute_qos(
    bandwidth: float,
    reliability: float,
    latency: float,
    device_type: str,
    w1: float = W1_BW,
    w2: float = W2_REL,
    w3: float = W3_LAT,
) -> float:
    """
    QoS = w1*(BW/BW_req) + w2*(Rel/Rel_req) - w3*(Lat/Lat_req)
    Clamped to [0, 1].
    """
    profile = QOS_PROFILES.get(device_type, QOS_PROFILES["ICU Monitoring"])
    bw_req  = profile["bandwidth_mbps"]
    rel_req = profile["reliability"]
    lat_req = profile["latency_ms"]

    score = (
        w1 * (bandwidth  / (bw_req  + 1e-8))
        + w2 * (reliability / (rel_req + 1e-8))
        - w3 * (latency    / (lat_req + 1e-8))
    )
    return float(max(0.0, min(score, 2.0)))   # allow >1 to show over-provisioning


def default_slice(device_type: str) -> dict:
    """Return default slice parameters from QoS profile."""
    p = QOS_PROFILES.get(device_type, QOS_PROFILES["ICU Monitoring"])
    return {
        "slice":     device_type,
        "bandwidth": p["bandwidth_mbps"],
        "latency":   p["latency_ms"],
        "priority":  p["priority"],
        "reason":    "Default profile allocation",
    }
