"""Transparent strategy summaries. Estimated quantities are labelled as proxies."""
import numpy as np
from config import ALPHA_COST, BETA_COST, GAMMA_COST, DELTA_COST

def compute_cost(latency: float, energy: float, llm_calls: int, resource_waste: float, n_steps: int = 1) -> float:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    vals = [latency, energy, llm_calls, resource_waste]
    if not np.isfinite(vals).all() or min(vals) < 0:
        raise ValueError("cost inputs must be finite and non-negative")
    lat_norm = np.clip(latency / 100.0, 0, 1)
    eng_norm = np.clip(energy / 100.0, 0, 1)
    llm_norm = np.clip(llm_calls / n_steps, 0, 1)
    waste_norm = np.clip(resource_waste / 200.0, 0, 1)
    return float(ALPHA_COST*lat_norm + BETA_COST*eng_norm + GAMMA_COST*llm_norm + DELTA_COST*waste_norm)

def energy_model(bandwidth: float, llm_invoked: bool) -> float:
    """Illustrative proxy only; not measured hardware/API energy."""
    return max(0.0, 0.5 * float(bandwidth)) + (15.0 if llm_invoked else 0.0)

def compare_strategies(results: list[dict]) -> dict:
    """Compare decision counts and QoS outputs; never fabricate per-strategy latency."""
    n = len(results)
    if not n:
        return {}
    proposed_calls = int(sum(bool(r["llm_invoked"]) for r in results))
    qos = [float(r.get("qos_score", 0.0)) for r in results]
    alloc_bw = [float(r.get("allocated_bandwidth", r.get("slice", {}).get("bandwidth", 0.0))) for r in results]
    traffic = [float(r.get("predicted_traffic", 0.0)) for r in results]
    waste = float(np.mean([max(0.0, b-t) for b,t in zip(alloc_bw, traffic)]))
    return {
        "Always-LLM": {"llm_calls": n, "llm_call_rate": 1.0, "avg_qos_score_observed": float(np.mean(qos))},
        "LSTM-Only": {"llm_calls": 0, "llm_call_rate": 0.0, "avg_qos_score_observed": float(np.mean(qos))},
        "Proposed Framework": {"llm_calls": proposed_calls, "llm_call_rate": proposed_calls/n,
                               "avg_qos_score_observed": float(np.mean(qos)), "avg_bandwidth_waste_mbps": waste,
                               "energy_proxy_only": float(np.mean([energy_model(b, False) for b in alloc_bw]))}
    }
