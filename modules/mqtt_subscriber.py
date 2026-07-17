"""
Closed-loop MQTT subscriber.
Flow: telemetry → LSTM → IsolationForest → PolicyEngine → NetworkManager → control command → device
"""
import json
import threading
import time
import numpy as np
import paho.mqtt.client as mqtt

from modules.sim_state          import sim_state
from modules.confidence         import normalise_error, escalation_score
from modules.anomaly_detector   import get_anomaly_score
from modules.predictor          import predict_next
from modules.llm                import invoke_llm
from modules.resource_allocator import allocate_resources
from modules import policy_engine, network_manager
from modules.storage            import (
    init_db, save_telemetry, save_prediction,
    save_anomaly, save_action, save_llm_call,
)
from config import (
    SEQ_LEN, DEFAULT_THRESHOLD,
    MQTT_BROKER, MQTT_PORT, MQTT_ANOMALY_ALERT,
)

BROKER   = MQTT_BROKER
PORT     = MQTT_PORT
TOPIC    = "hospital/device/#"

_lstm = None
_iso  = None
_tau  = DEFAULT_THRESHOLD

REQUIRED = {"device_id", "device_type", "bandwidth", "latency", "packet_loss", "signal_strength"}


def _build_feature_vector(history: list[dict]) -> np.ndarray:
    bw  = np.array([s["bandwidth"]   for s in history], dtype=float)
    lat = np.array([s["latency"]     for s in history], dtype=float)
    pl  = np.array([s["packet_loss"] for s in history], dtype=float)

    now         = time.localtime()
    hour        = float(now.tm_hour)
    day_of_week = float(now.tm_wday)
    is_peak     = float(8 <= now.tm_hour <= 20)

    roll6_mean  = float(bw[-6:].mean())  if len(bw) >= 6  else float(bw.mean())
    roll6_std   = float(bw[-6:].std())   if len(bw) >= 6  else 0.0
    roll12_mean = float(bw[-12:].mean()) if len(bw) >= 12 else float(bw.mean())
    roll12_std  = float(bw[-12:].std())  if len(bw) >= 12 else 0.0
    roll24_mean = float(bw[-24:].mean()) if len(bw) >= 24 else float(bw.mean())
    roll24_std  = float(bw[-24:].std())  if len(bw) >= 24 else 0.0
    diff1       = float(bw[-1] - bw[-2]) if len(bw) >= 2 else 0.0
    diff2       = float((bw[-1] - bw[-2]) - (bw[-2] - bw[-3])) if len(bw) >= 3 else 0.0

    return np.array([
        bw[-1], lat[-1], bw[-1] * 0.8,
        pl[-1], float(np.clip(100 - pl[-1] * 10, 90, 100)), lat[-1] * 0.1,
        hour, day_of_week, is_peak,
        roll6_mean, roll6_std, roll12_mean, roll12_std,
        roll24_mean, roll24_std, diff1, diff2,
    ], dtype=float)


def _process(telemetry: dict):
    did   = telemetry["device_id"]
    dtype = telemetry["device_type"]
    bw    = telemetry["bandwidth"]

    sim_state.update(telemetry)
    save_telemetry(telemetry)

    # Update network manager usage
    network_manager.update_usage(dtype, bw)

    history = sim_state.get_history(did)
    if len(history) < SEQ_LEN:
        return

    # ── LSTM prediction ────────────────────────────────────────────────────
    bw_window = np.array([s["bandwidth"] for s in history], dtype=float)
    predicted = predict_next(_lstm, bw_window)
    save_prediction(did, predicted, bw)

    # ── Isolation Forest ───────────────────────────────────────────────────
    x_feat  = _build_feature_vector(history)
    a_score = get_anomaly_score(_iso, x_feat)

    norm_err = normalise_error(bw, predicted)
    qos_risk = float(np.clip(telemetry["packet_loss"] * 10, 0, 1))
    esc      = escalation_score(norm_err, a_score, qos_risk)

    if a_score > MQTT_ANOMALY_ALERT:
        save_anomaly(did, a_score, esc)

    # ── Policy Engine ──────────────────────────────────────────────────────
    congested   = network_manager.is_congested()
    total_usage = network_manager.get_total_usage()

    policy = policy_engine.evaluate(
        device_id=did, device_type=dtype,
        priority=telemetry["priority"],
        bandwidth=bw, latency=telemetry["latency"],
        predicted_bw=predicted,
        anomaly_score=a_score, esc_score=esc,
        congested=congested, total_usage=total_usage,
    )

    # ── LLM — only for exceptional cases ──────────────────────────────────
    llm_response = None
    allocation   = policy["allocation"]

    if policy["invoke_llm"]:
        context = {
            "device_type": dtype, "traffic_mbps": bw,
            "predicted_mbps": predicted, "pred_error": norm_err,
            "confidence": float(np.exp(-norm_err)),
            "anomaly_score": a_score, "escalation_score": esc,
            "latency_ms": telemetry["latency"], "bandwidth_mbps": bw,
            "reliability": float(np.clip(1 - telemetry["packet_loss"] * 10, 0, 1) * 100),
            "current_slice": dtype, "qos_score": float(np.clip(1 - qos_risk, 0, 1)),
            "emergency": telemetry["status"] == "EMERGENCY",
        }
        try:
            llm_response = invoke_llm(context)
            allocation   = {
                "allocated_bandwidth": llm_response.get("bandwidth", allocation["allocated_bandwidth"]),
                "allocated_latency":   llm_response.get("latency",   allocation["allocated_latency"]),
                "priority":            llm_response.get("priority",  allocation["priority"]),
            }
            save_llm_call(did, llm_response)
            sim_state.add_event(
                f"🤖 LLM {did} → {llm_response.get('slice','?')} "
                f"BW={allocation['allocated_bandwidth']:.0f}Mbps | {llm_response.get('reason','')}"
            )
        except Exception as e:
            sim_state.add_event(f"⚠️ LLM failed {did}: {e}")

    # ── Send control command back to device ───────────────────────────────
    if policy["action"] not in ("NO_ACTION", "RAISE_ALERT"):
        network_manager.apply_allocation(did, dtype, allocation, policy["reason"])
        save_action(did, policy["action"], policy["severity"], policy["reason"])
        sim_state.add_event(
            f"{'🔴' if policy['severity']=='CRITICAL' else '⚡' if policy['severity']=='HIGH' else '🟡'} "
            f"{policy['action']} on {did} — {policy['reason']}"
        )

    # ── Store enriched state ───────────────────────────────────────────────
    sim_state.update({
        **telemetry,
        "predicted_bw":        round(predicted, 3),
        "anomaly_score":       round(a_score, 4),
        "escalation_score":    round(esc, 4),
        "llm_needed":          policy["invoke_llm"],
        "llm_response":        llm_response,
        "allocated_bandwidth": allocation["allocated_bandwidth"],
        "allocated_latency":   allocation["allocated_latency"],
        "policy_action":       policy["action"],
        "congested":           congested,
        "total_network_usage": round(total_usage, 1),
    })

    if a_score > MQTT_ANOMALY_ALERT:
        sim_state.add_event(f"🔴 ANOMALY {did} (score={a_score:.3f})")
    elif esc > _tau and not policy["invoke_llm"]:
        sim_state.add_event(f"⚡ Escalation {did} (esc={esc:.3f})")


def _on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        if REQUIRED.issubset(data.keys()):
            _process(data)
    except Exception as e:
        print(f"[SUBSCRIBER] Error: {e}")


def start(lstm_model, iso_clf, tau: float = DEFAULT_THRESHOLD):
    global _lstm, _iso, _tau
    _lstm, _iso, _tau = lstm_model, iso_clf, tau
    init_db()

    def _run():
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_message = _on_message
        try:
            client.connect(BROKER, PORT)
            client.subscribe(TOPIC)
            client.loop_forever()
        except Exception as e:
            print(f"[SUBSCRIBER] Could not connect: {e}")

    threading.Thread(target=_run, daemon=True).start()
    print("[SUBSCRIBER] Closed-loop backend started")
