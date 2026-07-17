"""
modules/network_manager.py
Maintains slice capacity, applies policy decisions, sends control commands.
"""
import time
import threading
import json
import paho.mqtt.client as mqtt
from config import MQTT_BROKER, MQTT_PORT

NETWORK_CAPACITY_MBPS = 300.0   # total simulated network capacity

# 6G slice definitions
SLICES = {
    "URLLC": {"capacity": 200.0, "used": 0.0},   # Ultra-Reliable Low Latency (Surgery, Ambulance)
    "eMBB":  {"capacity": 200.0, "used": 0.0},   # Enhanced Mobile Broadband (ICU)
    "mMTC":  {"capacity": 100.0, "used": 0.0},   # Massive Machine Type (Wearables)
}

DEVICE_TYPE_TO_SLICE = {
    "Remote Surgery":      "URLLC",
    "Emergency Ambulance": "URLLC",
    "ICU Monitoring":      "eMBB",
    "Wearable Devices":    "mMTC",
}

_lock    = threading.Lock()
_history: list[dict] = []   # action log
_client: mqtt.Client | None = None


def _get_client() -> mqtt.Client:
    global _client
    if _client is None:
        _client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        _client.connect(MQTT_BROKER, MQTT_PORT)
        _client.loop_start()
    return _client


# ── Capacity tracking ──────────────────────────────────────────────────────

def update_usage(device_type: str, bandwidth: float):
    slice_name = DEVICE_TYPE_TO_SLICE.get(device_type, "mMTC")
    with _lock:
        SLICES[slice_name]["used"] = bandwidth   # latest sample per device type


def get_slice_state() -> dict:
    with _lock:
        return {k: dict(v) for k, v in SLICES.items()}


def get_total_usage() -> float:
    with _lock:
        return sum(s["used"] for s in SLICES.values())


def is_congested() -> bool:
    return get_total_usage() > NETWORK_CAPACITY_MBPS * 0.85


# ── Control command dispatcher ─────────────────────────────────────────────

def send_control(device_id: str, command: dict, reason: str = ""):
    topic   = f"hospital/control/{device_id}"
    payload = json.dumps({**command, "timestamp": int(time.time())})
    _get_client().publish(topic, payload)

    entry = {
        "timestamp":  time.strftime("%H:%M:%S"),
        "device_id":  device_id,
        "command":    command,
        "reason":     reason,
    }
    with _lock:
        _history.append(entry)
    print(f"[NET_MGR] → {device_id}: {command} | {reason}")


def get_action_log(n: int = 50) -> list[dict]:
    with _lock:
        return list(_history)[-n:]


# ── Policy application ─────────────────────────────────────────────────────

def apply_allocation(device_id: str, device_type: str, allocation: dict, reason: str = ""):
    """Translate resource allocator output into a control command and send it."""
    cmd = {}

    bw = allocation.get("allocated_bandwidth")
    if bw is not None:
        cmd["max_bandwidth"] = round(bw, 1)

    priority = allocation.get("priority", "")
    if priority in ("Critical", "High"):
        cmd["publish_interval"] = 200    # faster publish = higher priority
        cmd["priority"]         = 10 if priority == "Critical" else 8
    elif priority == "Low":
        cmd["publish_interval"] = 2000
        cmd["throttle"]         = True
        cmd["priority"]         = 2
    elif priority == "Medium":
        cmd["publish_interval"] = 1000

    if cmd:
        send_control(device_id, cmd, reason)
