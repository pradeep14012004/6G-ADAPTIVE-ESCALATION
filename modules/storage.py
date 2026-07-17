"""
modules/storage.py
SQLite persistence for telemetry, predictions, anomalies, actions, llm_calls.
"""
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path("data/network.db")
_lock   = threading.Lock()


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with _lock:
        con = _conn()
        con.executescript("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER, device_id TEXT, device_type TEXT,
            bandwidth REAL, latency REAL, packet_loss REAL,
            signal_strength REAL, status TEXT, device_state TEXT
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER, device_id TEXT,
            predicted_bw REAL, actual_bw REAL, error REAL
        );
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER, device_id TEXT,
            anomaly_score REAL, escalation_score REAL
        );
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER, device_id TEXT,
            action TEXT, severity TEXT, reason TEXT
        );
        CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER, device_id TEXT,
            slice TEXT, bandwidth REAL, latency REAL,
            priority TEXT, reason TEXT
        );
        """)
        con.commit()
        con.close()


def save_telemetry(t: dict):
    with _lock:
        con = _conn()
        con.execute(
            "INSERT INTO telemetry (ts,device_id,device_type,bandwidth,latency,"
            "packet_loss,signal_strength,status,device_state) VALUES (?,?,?,?,?,?,?,?,?)",
            (t.get("timestamp", int(time.time())), t["device_id"], t["device_type"],
             t["bandwidth"], t["latency"], t["packet_loss"],
             t.get("signal_strength", 0), t.get("status", ""), t.get("device_state", "ACTIVE"))
        )
        con.commit(); con.close()


def save_prediction(device_id: str, predicted: float, actual: float):
    with _lock:
        con = _conn()
        con.execute(
            "INSERT INTO predictions (ts,device_id,predicted_bw,actual_bw,error) VALUES (?,?,?,?,?)",
            (int(time.time()), device_id, predicted, actual, abs(predicted - actual))
        )
        con.commit(); con.close()


def save_anomaly(device_id: str, anomaly_score: float, esc_score: float):
    with _lock:
        con = _conn()
        con.execute(
            "INSERT INTO anomalies (ts,device_id,anomaly_score,escalation_score) VALUES (?,?,?,?)",
            (int(time.time()), device_id, anomaly_score, esc_score)
        )
        con.commit(); con.close()


def save_action(device_id: str, action: str, severity: str, reason: str):
    with _lock:
        con = _conn()
        con.execute(
            "INSERT INTO actions (ts,device_id,action,severity,reason) VALUES (?,?,?,?,?)",
            (int(time.time()), device_id, action, severity, reason)
        )
        con.commit(); con.close()


def save_llm_call(device_id: str, llm_response: dict):
    with _lock:
        con = _conn()
        con.execute(
            "INSERT INTO llm_calls (ts,device_id,slice,bandwidth,latency,priority,reason) "
            "VALUES (?,?,?,?,?,?,?)",
            (int(time.time()), device_id,
             llm_response.get("slice", ""),
             llm_response.get("bandwidth", 0),
             llm_response.get("latency", 0),
             llm_response.get("priority", ""),
             llm_response.get("reason", ""))
        )
        con.commit(); con.close()


def query(sql: str, params: tuple = ()) -> list[dict]:
    with _lock:
        con = _conn()
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
        con.close()
        return [dict(r) for r in rows]
