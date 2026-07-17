import random

def _maybe_anomaly(metrics: dict) -> dict:
    if random.random() < 0.02:
        kind = random.choice(["spike", "packet_loss", "latency", "disconnect"])
        if kind == "spike":
            metrics["bandwidth"] *= random.uniform(3, 5)
        elif kind == "packet_loss":
            metrics["packet_loss"] = random.uniform(0.1, 0.4)
        elif kind == "latency":
            metrics["latency"] *= random.uniform(5, 12)
        elif kind == "disconnect":
            metrics["status"] = "DISCONNECTED"
            metrics["bandwidth"] = 0.0
    return metrics

def icu_pattern() -> dict:
    return _maybe_anomaly({"bandwidth": random.uniform(15, 25), "latency": random.uniform(1, 3),
                            "packet_loss": random.uniform(0, 0.001), "signal_strength": random.randint(-60, -45), "status": "ACTIVE"})

def surgery_pattern() -> dict:
    return _maybe_anomaly({"bandwidth": random.uniform(80, 120), "latency": random.uniform(0.5, 1.0),
                            "packet_loss": random.uniform(0, 0.0001), "signal_strength": random.randint(-50, -40), "status": "ACTIVE"})

def ambulance_pattern() -> dict:
    emergency = random.random() < 0.1
    return _maybe_anomaly({"bandwidth": random.uniform(40, 80) if emergency else random.uniform(0, 5),
                            "latency": random.uniform(2, 5), "packet_loss": random.uniform(0, 0.005),
                            "signal_strength": random.randint(-75, -55), "status": "EMERGENCY" if emergency else "IDLE"})

def wearable_pattern() -> dict:
    return _maybe_anomaly({"bandwidth": random.uniform(1, 10), "latency": random.uniform(10, 40),
                            "packet_loss": random.uniform(0, 0.02), "signal_strength": random.randint(-80, -60), "status": "ACTIVE"})

PATTERN_MAP = {
    "Remote Surgery":      surgery_pattern,
    "ICU Monitoring":      icu_pattern,
    "Wearable Devices":    wearable_pattern,
    "Emergency Ambulance": ambulance_pattern,
}
