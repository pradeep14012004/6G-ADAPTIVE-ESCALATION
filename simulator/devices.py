import time
import threading
from simulator.traffic_patterns import PATTERN_MAP

PRIORITY_MAP = {
    "Remote Surgery":      10,
    "ICU Monitoring":      8,
    "Emergency Ambulance": 6,
    "Wearable Devices":    2,
}

# State machine transitions
STATES      = ["ACTIVE", "THROTTLED", "RECOVERING"]
RECOVER_SEC = 15   # seconds before THROTTLED → RECOVERING
ACTIVE_SEC  = 10   # seconds before RECOVERING → ACTIVE


class MedicalDevice:
    def __init__(self, device_id: str, device_type: str):
        self.device_id          = device_id
        self.device_type        = device_type
        self.priority           = PRIORITY_MAP[device_type]
        self.current_bandwidth  = 10.0
        self.latency            = 2.0
        self.packet_loss        = 0.01
        self.signal_strength    = -55
        self.status             = "ACTIVE"
        self.publish_interval   = 1.0
        self.max_bandwidth      = None   # None = unrestricted
        self._state             = "ACTIVE"
        self._state_since       = time.time()
        self._lock              = threading.Lock()

    # ── State machine ──────────────────────────────────────────────────────

    def _tick_state(self):
        now = time.time()
        with self._lock:
            if self._state == "THROTTLED" and now - self._state_since > RECOVER_SEC:
                self._state       = "RECOVERING"
                self._state_since = now
                self.publish_interval = max(0.5, self.publish_interval * 0.5)
            elif self._state == "RECOVERING" and now - self._state_since > ACTIVE_SEC:
                self._state           = "ACTIVE"
                self._state_since     = now
                self.publish_interval = 1.0
                self.max_bandwidth    = None

    def get_state(self) -> str:
        return self._state

    # ── Update metrics from traffic pattern ───────────────────────────────

    def update_state(self):
        self._tick_state()
        m = PATTERN_MAP[self.device_type]()
        with self._lock:
            bw = m["bandwidth"]
            if self.max_bandwidth is not None:
                bw = min(bw, self.max_bandwidth)
            self.current_bandwidth = bw
            self.latency           = m["latency"]
            self.packet_loss       = m["packet_loss"]
            self.signal_strength   = m["signal_strength"]
            # Only override status if not externally set to EMERGENCY/DISCONNECTED
            if self.status not in ("EMERGENCY", "DISCONNECTED"):
                self.status = self._state if self._state != "ACTIVE" else m["status"]

    # ── React to control command from AI ──────────────────────────────────

    def apply_control(self, cmd: dict):
        with self._lock:
            if "publish_interval" in cmd:
                self.publish_interval = cmd["publish_interval"] / 1000.0
            if "max_bandwidth" in cmd:
                self.max_bandwidth = float(cmd["max_bandwidth"])
            if "priority" in cmd:
                self.priority = int(cmd["priority"])
            if cmd.get("throttle"):
                self._state       = "THROTTLED"
                self._state_since = time.time()
            if cmd.get("emergency"):
                self.status           = "EMERGENCY"
                self.publish_interval = 0.2   # fast publish during emergency

    # ── Telemetry payload ─────────────────────────────────────────────────

    def telemetry(self) -> dict:
        return {
            "device_id":       self.device_id,
            "device_type":     self.device_type,
            "priority":        self.priority,
            "bandwidth":       round(self.current_bandwidth, 3),
            "latency":         round(self.latency, 3),
            "packet_loss":     round(self.packet_loss, 4),
            "signal_strength": self.signal_strength,
            "status":          self.status,
            "device_state":    self._state,
            "timestamp":       int(time.time()),
        }
