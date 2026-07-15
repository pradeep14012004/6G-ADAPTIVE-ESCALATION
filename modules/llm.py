"""
modules/llm.py
Groq LLM integration — invoked only when escalation score exceeds threshold.
Returns structured JSON slice recommendation.
"""

import json
import os
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Export it as an environment variable."
            )
        _client = Groq(api_key=key)
    return _client


# ── Prompt builder ─────────────────────────────────────────────────────────

def _build_prompt(context: dict) -> str:
    return f"""You are an AI-native 6G network slice controller for a healthcare network.

Network State:
- Device Type      : {context['device_type']}
- Current Traffic  : {context['traffic_mbps']:.2f} Mbps
- Predicted Traffic: {context['predicted_mbps']:.2f} Mbps
- Prediction Error : {context['pred_error']:.4f} (normalised)
- Confidence Score : {context['confidence']:.4f}
- Anomaly Score    : {context['anomaly_score']:.4f}
- Escalation Score : {context['escalation_score']:.4f}
- Current Latency  : {context['latency_ms']:.2f} ms
- Current Bandwidth: {context['bandwidth_mbps']:.2f} Mbps
- Current Reliability: {context['reliability']:.3f} %
- Current Slice    : {context['current_slice']}
- QoS Score        : {context['qos_score']:.4f}
- Emergency Mode   : {context['emergency']}

Task: Recommend an optimal network slice reconfiguration.

IMPORTANT: Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.

JSON schema:
{{
  "slice": "<slice name>",
  "bandwidth": <number in Mbps>,
  "latency": <number in ms>,
  "priority": "<Critical|High|Medium|Low>",
  "reason": "<one sentence justification>"
}}"""


# ── LLM call ───────────────────────────────────────────────────────────────

def invoke_llm(context: dict) -> dict:
    """
    Call Groq LLM and return parsed slice recommendation.
    Falls back to a rule-based recommendation on any failure.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": _build_prompt(context)}],
            temperature=0.1,
            max_tokens=256,
            timeout=GROQ_TIMEOUT,
        )
        raw = response.choices[0].message.content.strip()
        # Extract JSON even if model wraps it in markdown
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)

    except Exception as exc:
        # Graceful fallback — never crash the pipeline
        return _rule_based_fallback(context, str(exc))


def _rule_based_fallback(context: dict, reason: str = "") -> dict:
    """Deterministic fallback when LLM is unavailable."""
    traffic = context.get("predicted_mbps", context.get("traffic_mbps", 60))
    if traffic > 120 or context.get("emergency", False):
        return {"slice": "Emergency", "bandwidth": 180, "latency": 1,
                "priority": "Critical", "reason": f"Fallback: high traffic. {reason}"}
    if traffic > 80:
        return {"slice": "High-Priority", "bandwidth": 120, "latency": 3,
                "priority": "High", "reason": f"Fallback: elevated traffic. {reason}"}
    return {"slice": "Standard", "bandwidth": 60, "latency": 10,
            "priority": "Medium", "reason": f"Fallback: normal traffic. {reason}"}
