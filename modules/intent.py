"""
modules/intent.py
Maps user-selected device type to a QoS profile and computes QoS risk.
"""

from config import QOS_PROFILES


def get_qos_profile(device_type: str) -> dict:
    """Return QoS requirements for the given device type."""
    return QOS_PROFILES.get(device_type, QOS_PROFILES["ICU Monitoring"])


def compute_qos_risk(
    device_type: str,
    current_latency: float,
    current_bandwidth: float,
    current_reliability: float,
) -> float:
    """
    Normalised QoS risk in [0, 1].
    Risk increases when current metrics fall below required thresholds.
    """
    profile = get_qos_profile(device_type)

    lat_risk = max(0.0, (current_latency - profile["latency_ms"])   / (profile["latency_ms"]   + 1e-8))
    bw_risk  = max(0.0, (profile["bandwidth_mbps"] - current_bandwidth) / (profile["bandwidth_mbps"] + 1e-8))
    rel_risk = max(0.0, (profile["reliability"] - current_reliability)  / (profile["reliability"]    + 1e-8))

    risk = (lat_risk + bw_risk + rel_risk) / 3.0
    return float(min(risk, 1.0))
