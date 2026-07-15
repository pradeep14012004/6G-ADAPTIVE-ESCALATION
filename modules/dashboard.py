"""
modules/dashboard.py
Plotly figure factories used by both the Streamlit app and notebooks.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


PALETTE = {
    "actual":     "#00B4D8",
    "predicted":  "#F77F00",
    "anomaly":    "#E63946",
    "confidence": "#2DC653",
    "escalation": "#9B5DE5",
    "llm":        "#FF006E",
    "qos":        "#06D6A0",
    "cost":       "#FFB703",
}


def fig_traffic_prediction(
    timestamps, actual: np.ndarray, predicted: np.ndarray, title: str = "Traffic Prediction"
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=actual,    name="Actual",    line=dict(color=PALETTE["actual"])))
    fig.add_trace(go.Scatter(x=timestamps, y=predicted, name="Predicted", line=dict(color=PALETTE["predicted"], dash="dash")))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Traffic (Mbps)",
                      template="plotly_dark", height=350)
    return fig


def fig_anomaly_scores(timestamps, scores: np.ndarray, labels: np.ndarray) -> go.Figure:
    colors = [PALETTE["anomaly"] if l == 1 else PALETTE["actual"] for l in labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=timestamps, y=scores, marker_color=colors, name="Anomaly Score"))
    fig.update_layout(title="Isolation Forest Anomaly Scores",
                      xaxis_title="Time", yaxis_title="Score",
                      template="plotly_dark", height=300)
    return fig


def fig_confidence_gauge(confidence: float, escalation: float, threshold: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "indicator"}, {"type": "indicator"}]])
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=round(confidence * 100, 1),
        title={"text": "Confidence (%)"},
        gauge={"axis": {"range": [0, 100]},
               "bar":  {"color": PALETTE["confidence"]},
               "steps": [{"range": [0, 40], "color": "#E63946"},
                         {"range": [40, 70], "color": "#FFB703"},
                         {"range": [70, 100], "color": "#2DC653"}]},
    ), row=1, col=1)
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=round(escalation * 100, 1),
        delta={"reference": threshold * 100, "increasing": {"color": PALETTE["anomaly"]}},
        title={"text": "Escalation Score (%)"},
        gauge={"axis": {"range": [0, 100]},
               "bar":  {"color": PALETTE["escalation"]},
               "threshold": {"line": {"color": "white", "width": 3},
                             "thickness": 0.75, "value": threshold * 100}},
    ), row=1, col=2)
    fig.update_layout(template="plotly_dark", height=280)
    return fig


def fig_llm_invocations(timestamps, llm_flags: list[bool]) -> go.Figure:
    y = [1 if f else 0 for f in llm_flags]
    colors = [PALETTE["llm"] if f else PALETTE["actual"] for f in llm_flags]
    fig = go.Figure(go.Bar(x=timestamps, y=y, marker_color=colors, name="LLM Invoked"))
    fig.update_layout(title="LLM Invocation Events",
                      xaxis_title="Time", yaxis_title="Invoked (1=Yes)",
                      template="plotly_dark", height=250)
    return fig


def fig_qos_dashboard(timestamps, qos_scores: np.ndarray, bandwidth: np.ndarray,
                      latency: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("QoS Score", "Bandwidth (Mbps)", "Latency (ms)"))
    fig.add_trace(go.Scatter(x=timestamps, y=qos_scores, fill="tozeroy",
                             line=dict(color=PALETTE["qos"]), name="QoS"), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=bandwidth,
                             line=dict(color=PALETTE["actual"]), name="BW"), row=2, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=latency,
                             line=dict(color=PALETTE["cost"]), name="Latency"), row=3, col=1)
    fig.update_layout(template="plotly_dark", height=500, showlegend=False)
    return fig


def fig_cost_comparison(comparison: dict) -> go.Figure:
    strategies = list(comparison.keys())
    metrics    = ["avg_latency", "avg_energy", "llm_calls", "cost"]
    labels     = ["Avg Latency (ms)", "Avg Energy (J)", "LLM Calls", "Cost"]

    fig = make_subplots(rows=2, cols=2, subplot_titles=labels)
    positions = [(1,1),(1,2),(2,1),(2,2)]
    colors    = [PALETTE["cost"], PALETTE["qos"], PALETTE["llm"], PALETTE["escalation"]]

    for (r, c), metric, color in zip(positions, metrics, colors):
        vals = [comparison[s][metric] for s in strategies]
        fig.add_trace(go.Bar(x=strategies, y=vals, marker_color=color, showlegend=False),
                      row=r, col=c)
    fig.update_layout(template="plotly_dark", height=450,
                      title="Strategy Performance Comparison")
    return fig


def fig_slice_allocation(slice_rec: dict) -> go.Figure:
    categories = ["Bandwidth", "Latency", "Priority Score"]
    priority_map = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    values = [
        slice_rec.get("bandwidth", 0),
        slice_rec.get("latency", 0),
        priority_map.get(slice_rec.get("priority", "Medium"), 2),
    ]
    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=[PALETTE["actual"], PALETTE["cost"], PALETTE["llm"]],
    ))
    fig.update_layout(title=f"Slice: {slice_rec.get('slice','N/A')}",
                      template="plotly_dark", height=280)
    return fig
