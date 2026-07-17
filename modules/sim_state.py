import threading
from collections import deque

WINDOW = 20

class SimState:
    def __init__(self):
        self._lock    = threading.Lock()
        self._devices: dict[str, dict]   = {}
        self._history: dict[str, deque]  = {}
        self._events:  deque             = deque(maxlen=100)

    def update(self, telemetry: dict):
        did = telemetry["device_id"]
        with self._lock:
            self._devices[did] = telemetry
            if did not in self._history:
                self._history[did] = deque(maxlen=WINDOW)
            self._history[did].append(telemetry)

    def get_all(self) -> dict:
        with self._lock:
            return dict(self._devices)

    def get_history(self, device_id: str) -> list:
        with self._lock:
            return list(self._history.get(device_id, []))

    def add_event(self, msg: str):
        import time
        with self._lock:
            self._events.append(f"{time.strftime('%H:%M:%S')} — {msg}")

    def get_events(self, n: int = 30) -> list:
        with self._lock:
            return list(self._events)[-n:]

# Singleton shared across subscriber + app.py
sim_state = SimState()
