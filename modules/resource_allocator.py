"""
modules/resource_allocator.py
Dynamic resource allocation based on predicted traffic and slice recommendation.
"""

import numpy as np
from config import ALPHA_RESOURCE, QOS_PROFILES


def allocate_resources(
    predicted_traffic: float,
    slice_rec: dict,
    llm_invoked: bool,
    device_type: str,
) -> dict:
    """
    R = alpha * predicted_traffic  (baseline)
    If LLM was invoked, use LLM-recommended bandwidth/latency/priority.
    """
    profile = QOS_PROFILES.get(device_type, QOS_PROFILES["ICU Monitoring"])

    if llm_invoked:
        bandwidth = float(slice_rec.get("bandwidth", profile["bandwidth_mbps"]))
        latency   = float(slice_rec.get("latency",   profile["latency_ms"]))
        priority  = slice_rec.get("priority", profile["priority"])
    else:
        bandwidth = float(np.clip(ALPHA_RESOURCE * predicted_traffic,
                                  profile["bandwidth_mbps"] * 0.5,
                                  profile["bandwidth_mbps"] * 2.0))
        latency   = profile["latency_ms"]
        priority  = profile["priority"]

    utilisation = float(np.clip(predicted_traffic / (bandwidth + 1e-8), 0, 1))

    return {
        "allocated_bandwidth": bandwidth,
        "allocated_latency":   latency,
        "priority":            priority,
        "utilisation":         utilisation,
        "resource_waste":      max(0.0, bandwidth - predicted_traffic),
    }
