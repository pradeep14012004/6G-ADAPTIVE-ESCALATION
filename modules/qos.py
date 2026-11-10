"""QoS SLA metrics; score is bounded [0,1], with explicit SLA compliance."""
import numpy as np
from config import W1_BW, W2_REL, W3_LAT, QOS_PROFILES

def sla_violations(bandwidth: float, reliability: float, latency: float, device_type: str) -> dict:
    p=QOS_PROFILES.get(device_type,QOS_PROFILES["ICU Monitoring"])
    return {"bandwidth_violation":bool(bandwidth < p["bandwidth_mbps"]),
            "reliability_violation":bool(reliability < p["reliability"]),
            "latency_violation":bool(latency > p["latency_ms"])}

def compute_qos(bandwidth: float, reliability: float, latency: float, device_type: str,
                w1: float=W1_BW,w2: float=W2_REL,w3: float=W3_LAT) -> float:
    weights=np.asarray([w1,w2,w3],dtype=float)
    values=np.asarray([bandwidth,reliability,latency],dtype=float)
    if not np.isfinite(values).all() or np.any(values<0): raise ValueError("QoS measurements must be finite and non-negative")
    if np.any(weights<0) or not np.isclose(weights.sum(),1): raise ValueError("QoS weights must be non-negative and sum to 1")
    p=QOS_PROFILES.get(device_type,QOS_PROFILES["ICU Monitoring"])
    bw=float(np.clip(bandwidth/p["bandwidth_mbps"],0,1))
    rel=float(np.clip(reliability/p["reliability"],0,1))
    lat=float(np.clip(p["latency_ms"]/(latency+1e-12),0,1))
    return float(w1*bw+w2*rel+w3*lat)

def default_slice(device_type: str) -> dict:
    p=QOS_PROFILES.get(device_type,QOS_PROFILES["ICU Monitoring"])
    return {"slice":device_type,"bandwidth":p["bandwidth_mbps"],"latency":p["latency_ms"],
            "priority":p["priority"],"reason":"Default QoS profile allocation"}
