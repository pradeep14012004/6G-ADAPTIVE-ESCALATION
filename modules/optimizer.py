"""
modules/optimizer.py
Cost function and performance comparison across three strategies.
"""

import numpy as np
from config import ALPHA_COST, BETA_COST, GAMMA_COST, DELTA_COST


def compute_cost(
    latency: float,
    energy: float,
    llm_calls: int,
    resource_waste: float,
    n_steps: int = 1,
) -> float:
    """
    Cost = alpha*Latency + beta*Energy + gamma*(LLMCalls/n) + delta*ResourceWaste
    All terms normalised to [0,1] range before weighting.
    """
    lat_norm   = np.clip(latency / 100.0, 0, 1)
    eng_norm   = np.clip(energy  / 100.0, 0, 1)
    llm_norm   = np.clip(llm_calls / max(n_steps, 1), 0, 1)
    waste_norm = np.clip(resource_waste / 200.0, 0, 1)

    return float(
        ALPHA_COST * lat_norm
        + BETA_COST  * eng_norm
        + GAMMA_COST * llm_norm
        + DELTA_COST * waste_norm
    )


def energy_model(bandwidth: float, llm_invoked: bool) -> float:
    """Simple linear energy proxy (Joules per interval)."""
    base   = 0.5 * bandwidth
    llm_e  = 15.0 if llm_invoked else 0.0
    return base + llm_e


def compare_strategies(results: list[dict]) -> dict:
    """
    Compare three strategies over a list of escalation result dicts:
      1. Always-LLM
      2. LSTM-Only (no LLM, no escalation)
      3. Proposed Adaptive Escalation
    Returns summary metrics for each strategy.
    """
    n = len(results)
    if n == 0:
        return {}

    proposed_llm_calls = sum(r["llm_invoked"] for r in results)

    def _metrics(llm_calls_per_step, use_llm_bw=False):
        latencies, energies, wastes, qos_scores = [], [], [], []
        for r in results:
            alloc_bw  = r["slice"].get("bandwidth", 80)
            lat       = r["slice"].get("latency", 10)
            energy    = energy_model(alloc_bw, use_llm_bw)
            waste     = max(0.0, alloc_bw - r["predicted_traffic"])
            latencies.append(lat)
            energies.append(energy)
            wastes.append(waste)
            qos_scores.append(r["qos_score"])
        total_llm = int(llm_calls_per_step * n)
        cost = compute_cost(
            np.mean(latencies), np.mean(energies),
            total_llm, np.mean(wastes), n
        )
        return {
            "avg_latency":    float(np.mean(latencies)),
            "avg_energy":     float(np.mean(energies)),
            "llm_calls":      total_llm,
            "avg_qos":        float(np.mean(qos_scores)),
            "avg_waste":      float(np.mean(wastes)),
            "cost":           cost,
        }

    return {
        "Always-LLM":          _metrics(1.0, use_llm_bw=True),
        "LSTM-Only":           _metrics(0.0, use_llm_bw=False),
        "Proposed Framework":  _metrics(proposed_llm_calls / n, use_llm_bw=False),
    }
