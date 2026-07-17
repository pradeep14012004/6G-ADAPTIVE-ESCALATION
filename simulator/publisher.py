"""
Run from project root:  python -m simulator.publisher
"""
import json, time, threading, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import paho.mqtt.client as mqtt
from simulator.devices import MedicalDevice
from config import MQTT_BROKER, MQTT_PORT

BROKER       = MQTT_BROKER
PORT         = MQTT_PORT
TOPIC_FMT    = "hospital/device/{}"
CONTROL_TOPIC = "hospital/control/#"

ROSTER = [
    ("ICU Monitoring",      5),
    ("Remote Surgery",      3),
    ("Emergency Ambulance", 4),
    ("Wearable Devices",    8),
]

# Global device registry so control messages can find devices by ID
_device_registry: dict[str, MedicalDevice] = {}


def build_devices() -> list[MedicalDevice]:
    devices = []
    for dtype, count in ROSTER:
        prefix = dtype.replace(" ", "")[:6]
        for i in range(1, count + 1):
            d = MedicalDevice(f"{prefix}_{i:02d}", dtype)
            devices.append(d)
            _device_registry[d.device_id] = d
    return devices


def _on_control(client, userdata, msg):
    """React to AI control commands — update device behaviour immediately."""
    try:
        did = msg.topic.split("/")[-1]
        cmd = json.loads(msg.payload.decode())
        if did in _device_registry:
            _device_registry[did].apply_control(cmd)
            print(f"[CTRL] {did} ← {cmd}")
    except Exception as e:
        print(f"[CTRL ERROR] {e}")


def _publish_loop(client: mqtt.Client, device: MedicalDevice):
    while True:
        device.update_state()
        client.publish(TOPIC_FMT.format(device.device_id),
                       json.dumps(device.telemetry()))
        time.sleep(device.publish_interval)


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = _on_control
    client.connect(BROKER, PORT)
    client.subscribe(CONTROL_TOPIC)
    client.loop_start()

    devices = build_devices()
    print(f"[PUBLISHER] {len(devices)} virtual devices started — listening for control on {CONTROL_TOPIC}")

    for d in devices:
        threading.Thread(target=_publish_loop, args=(client, d), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[PUBLISHER] Stopped.")
        client.loop_stop()


if __name__ == "__main__":
    main()
