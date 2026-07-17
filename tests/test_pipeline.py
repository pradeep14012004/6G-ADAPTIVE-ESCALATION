"""
tests/test_pipeline.py
Run: python -m pytest tests/test_pipeline.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import pytest
from unittest.mock import patch

from modules import policy_engine, network_manager
from modules.policy_engine import _isolated


# ── helpers ────────────────────────────────────────────────────────────────

def _eval(device_id="ICU_01", device_type="ICU Monitoring", priority=8,
          bandwidth=20.0, latency=2.0, predicted_bw=21.0,
          anomaly_score=0.0, esc_score=0.3,
          congested=False, total_usage=50.0):
    _isolated.clear()
    return policy_engine.evaluate(
        device_id=device_id, device_type=device_type, priority=priority,
        bandwidth=bandwidth, latency=latency, predicted_bw=predicted_bw,
        anomaly_score=anomaly_score, esc_score=esc_score,
        congested=congested, total_usage=total_usage,
    )


def _reset_slices():
    for s in network_manager.SLICES.values():
        s["used"] = 0.0


# ══════════════════════════════════════════════════════════════════════════
# Policy Engine — Rule Tests
# ══════════════════════════════════════════════════════════════════════════

class TestPolicyRules:

    def test_normal_no_action(self):
        assert _eval()["action"] == "NO_ACTION"

    def test_ambulance_emergency_reserves_capacity(self):
        r = _eval(device_type="Emergency Ambulance", priority=6, bandwidth=50.0)
        assert r["action"] == "RESERVE_CAPACITY"
        assert r["severity"] == "CRITICAL"
        assert r["allocation"]["allocated_bandwidth"] == 100.0

    def test_ambulance_idle_no_action(self):
        assert _eval(device_type="Emergency Ambulance", priority=6, bandwidth=3.0)["action"] == "NO_ACTION"

    def test_surgery_rising_reserves_capacity(self):
        r = _eval(device_type="Remote Surgery", priority=10, bandwidth=80.0, predicted_bw=95.0)
        assert r["action"] == "RESERVE_CAPACITY"
        assert r["allocation"]["priority"] == "Critical"

    def test_surgery_stable_no_action(self):
        assert _eval(device_type="Remote Surgery", priority=10, bandwidth=80.0, predicted_bw=81.0)["action"] == "NO_ACTION"

    def test_icu_high_latency_increases_priority(self):
        r = _eval(device_type="ICU Monitoring", latency=8.0)
        assert r["action"] == "INCREASE_PRIORITY"
        assert r["allocation"]["allocated_bandwidth"] == 80.0

    def test_icu_normal_latency_no_action(self):
        assert _eval(device_type="ICU Monitoring", latency=3.0)["action"] == "NO_ACTION"

    def test_wearable_throttled_when_congested(self):
        r = _eval(device_type="Wearable Devices", priority=2, bandwidth=8.0,
                  congested=True, total_usage=110.0)
        assert r["action"] == "THROTTLE"
        assert r["allocation"]["priority"] == "Low"
        assert r["allocation"]["allocated_bandwidth"] <= 3.0

    def test_wearable_not_throttled_without_congestion(self):
        assert _eval(device_type="Wearable Devices", priority=2, bandwidth=8.0)["action"] == "NO_ACTION"


# ══════════════════════════════════════════════════════════════════════════
# Anomaly Isolation Tests
# ══════════════════════════════════════════════════════════════════════════

class TestAnomalyIsolation:

    def test_anomaly_triggers_isolate(self):
        _isolated.clear()
        r = policy_engine.evaluate(
            device_id="ICU_01", device_type="ICU Monitoring", priority=8,
            bandwidth=20.0, latency=2.0, predicted_bw=21.0,
            anomaly_score=0.85, esc_score=0.5, congested=False, total_usage=50.0,
        )
        assert r["action"] == "ISOLATE_DEVICE"
        assert r["severity"] == "CRITICAL"
        assert r["invoke_llm"] is True

    def test_isolation_caps_bandwidth(self):
        _isolated.clear()
        r = policy_engine.evaluate(
            device_id="ICU_02", device_type="ICU Monitoring", priority=8,
            bandwidth=20.0, latency=2.0, predicted_bw=21.0,
            anomaly_score=0.9, esc_score=0.5, congested=False, total_usage=50.0,
        )
        # ICU profile bandwidth = 80 Mbps, 20% = 16 Mbps
        assert r["allocation"]["allocated_bandwidth"] <= 16.1
        assert r["allocation"]["priority"] == "Low"

    def test_isolated_device_added_to_registry(self):
        _isolated.clear()
        policy_engine.evaluate(
            device_id="ICU_03", device_type="ICU Monitoring", priority=8,
            bandwidth=20.0, latency=2.0, predicted_bw=21.0,
            anomaly_score=0.75, esc_score=0.5, congested=False, total_usage=50.0,
        )
        assert "ICU_03" in _isolated

    def test_recovery_after_timeout(self):
        _isolated.clear()
        _isolated["ICU_04"] = time.time() - 25   # isolated 25s ago
        r = policy_engine.evaluate(
            device_id="ICU_04", device_type="ICU Monitoring", priority=8,
            bandwidth=20.0, latency=2.0, predicted_bw=21.0,
            anomaly_score=0.1, esc_score=0.2, congested=False, total_usage=50.0,
        )
        assert r["action"] == "RECOVER_DEVICE"
        assert "ICU_04" not in _isolated

    def test_no_recovery_before_timeout(self):
        _isolated.clear()
        _isolated["ICU_05"] = time.time() - 5    # only 5s ago
        r = policy_engine.evaluate(
            device_id="ICU_05", device_type="ICU Monitoring", priority=8,
            bandwidth=20.0, latency=2.0, predicted_bw=21.0,
            anomaly_score=0.1, esc_score=0.2, congested=False, total_usage=50.0,
        )
        assert r["action"] != "RECOVER_DEVICE"
        assert "ICU_05" in _isolated


# ══════════════════════════════════════════════════════════════════════════
# LLM Invocation Tests
# ══════════════════════════════════════════════════════════════════════════

class TestLLMInvocation:

    def test_llm_called_for_anomaly(self):
        _isolated.clear()
        r = policy_engine.evaluate(
            device_id="Surgery_01", device_type="Remote Surgery", priority=10,
            bandwidth=100.0, latency=1.0, predicted_bw=102.0,
            anomaly_score=0.8, esc_score=0.5, congested=False, total_usage=50.0,
        )
        assert r["invoke_llm"] is True

    def test_llm_called_for_high_esc_and_congestion(self):
        _isolated.clear()
        r = policy_engine.evaluate(
            device_id="ICU_01", device_type="ICU Monitoring", priority=8,
            bandwidth=20.0, latency=2.0, predicted_bw=21.0,
            anomaly_score=0.0, esc_score=0.95, congested=True, total_usage=110.0,
        )
        assert r["invoke_llm"] is True
        assert r["action"] == "LLM_ESCALATION"

    def test_llm_not_called_low_esc_no_anomaly(self):
        assert _eval(anomaly_score=0.0, esc_score=0.5)["invoke_llm"] is False

    def test_llm_not_called_high_esc_without_congestion(self):
        _isolated.clear()
        r = policy_engine.evaluate(
            device_id="ICU_01", device_type="ICU Monitoring", priority=8,
            bandwidth=20.0, latency=2.0, predicted_bw=21.0,
            anomaly_score=0.0, esc_score=0.95, congested=False, total_usage=50.0,
        )
        assert r["invoke_llm"] is False

    def test_llm_fallback_returns_valid_structure(self):
        from modules.llm import _rule_based_fallback
        result = _rule_based_fallback({"predicted_mbps": 50, "emergency": False})
        for key in ("slice", "bandwidth", "latency", "priority", "reason"):
            assert key in result

    def test_llm_fallback_emergency_mode(self):
        from modules.llm import _rule_based_fallback
        result = _rule_based_fallback({"predicted_mbps": 50, "emergency": True})
        assert result["priority"] == "Critical"
        assert result["slice"] == "Emergency"

    def test_llm_mock_response_used_for_allocation(self):
        mock_resp = {"slice": "Emergency", "bandwidth": 180.0,
                     "latency": 1.0, "priority": "Critical", "reason": "test"}
        with patch("modules.llm.invoke_llm", return_value=mock_resp) as mock_llm:
            from modules.llm import invoke_llm
            result = invoke_llm({"device_type": "Remote Surgery"})
            assert result["bandwidth"] == 180.0
            mock_llm.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════
# Network Manager Tests
# ══════════════════════════════════════════════════════════════════════════

class TestNetworkManager:

    def setup_method(self):
        _reset_slices()

    def test_usage_update_correct_slice(self):
        network_manager.update_usage("ICU Monitoring", 40.0)
        assert network_manager.SLICES["eMBB"]["used"] == 40.0

    def test_surgery_maps_to_urllc(self):
        network_manager.update_usage("Remote Surgery", 90.0)
        assert network_manager.SLICES["URLLC"]["used"] == 90.0

    def test_wearable_maps_to_mmtc(self):
        network_manager.update_usage("Wearable Devices", 5.0)
        assert network_manager.SLICES["mMTC"]["used"] == 5.0

    def test_total_usage_sum(self):
        network_manager.update_usage("ICU Monitoring",   30.0)
        network_manager.update_usage("Remote Surgery",   50.0)
        network_manager.update_usage("Wearable Devices", 10.0)
        assert network_manager.get_total_usage() == 90.0

    def test_not_congested_below_threshold(self):
        network_manager.update_usage("ICU Monitoring", 10.0)
        assert network_manager.is_congested() is False

    def test_congested_above_85_percent(self):
        # 125 * 0.85 = 106.25
        network_manager.update_usage("ICU Monitoring",   50.0)
        network_manager.update_usage("Remote Surgery",   50.0)
        network_manager.update_usage("Wearable Devices", 20.0)
        assert network_manager.is_congested() is True

    def test_apply_allocation_low_priority_throttles(self):
        with patch("modules.network_manager.send_control") as mock_send:
            network_manager.apply_allocation(
                "Wearab_01", "Wearable Devices",
                {"allocated_bandwidth": 2.0, "allocated_latency": 40.0, "priority": "Low"},
                "throttle test"
            )
            mock_send.assert_called_once()
            cmd = mock_send.call_args[0][1]
            assert cmd["max_bandwidth"] == 2.0
            assert cmd["throttle"] is True
            assert cmd["publish_interval"] == 2000

    def test_apply_allocation_critical_priority_fast_publish(self):
        with patch("modules.network_manager.send_control") as mock_send:
            network_manager.apply_allocation(
                "Surgery_01", "Remote Surgery",
                {"allocated_bandwidth": 150.0, "allocated_latency": 1.0, "priority": "Critical"},
                "surgery spike"
            )
            cmd = mock_send.call_args[0][1]
            assert cmd["publish_interval"] == 200
            assert cmd["priority"] == 10


# ══════════════════════════════════════════════════════════════════════════
# End-to-End Control Flow Tests
# ══════════════════════════════════════════════════════════════════════════

class TestControlFlow:

    def test_anomaly_triggers_control_command(self):
        _isolated.clear()
        _reset_slices()
        with patch("modules.network_manager.send_control") as mock_send:
            policy = policy_engine.evaluate(
                device_id="ICU_99", device_type="ICU Monitoring", priority=8,
                bandwidth=20.0, latency=2.0, predicted_bw=21.0,
                anomaly_score=0.85, esc_score=0.6, congested=False, total_usage=50.0,
            )
            assert policy["action"] == "ISOLATE_DEVICE"
            assert policy["invoke_llm"] is True
            network_manager.apply_allocation("ICU_99", "ICU Monitoring",
                                             policy["allocation"], policy["reason"])
            mock_send.assert_called_once()
            cmd = mock_send.call_args[0][1]
            assert cmd["throttle"] is True
            assert cmd["max_bandwidth"] <= 16.1

    def test_recovery_restores_full_allocation(self):
        _isolated.clear()
        _isolated["ICU_99"] = time.time() - 25
        _reset_slices()
        with patch("modules.network_manager.send_control") as mock_send:
            policy = policy_engine.evaluate(
                device_id="ICU_99", device_type="ICU Monitoring", priority=8,
                bandwidth=20.0, latency=2.0, predicted_bw=21.0,
                anomaly_score=0.1, esc_score=0.2, congested=False, total_usage=50.0,
            )
            assert policy["action"] == "RECOVER_DEVICE"
            network_manager.apply_allocation("ICU_99", "ICU Monitoring",
                                             policy["allocation"], policy["reason"])
            cmd = mock_send.call_args[0][1]
            assert cmd["max_bandwidth"] == 80.0   # full ICU bandwidth restored

    def test_congestion_throttles_wearable_and_sends_command(self):
        _isolated.clear()
        _reset_slices()
        with patch("modules.network_manager.send_control") as mock_send:
            policy = policy_engine.evaluate(
                device_id="Wearab_01", device_type="Wearable Devices", priority=2,
                bandwidth=8.0, latency=20.0, predicted_bw=8.5,
                anomaly_score=0.0, esc_score=0.4, congested=True, total_usage=110.0,
            )
            assert policy["action"] == "THROTTLE"
            network_manager.apply_allocation("Wearab_01", "Wearable Devices",
                                             policy["allocation"], policy["reason"])
            mock_send.assert_called_once()
            cmd = mock_send.call_args[0][1]
            assert cmd["throttle"] is True
