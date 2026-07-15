"""
app.py — Streamlit Dashboard
Adaptive Escalation Framework for Intent-Aware AI-Native Network Slicing in 6G Healthcare Networks
"""

import json
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Escalation 6G",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Imports (after page config) -------------------------------------------
from config import (
    LSTM_MODEL_PATH, ISO_FOREST_PATH, SCALER_PATH, THRESHOLD_PATH,
    DEFAULT_THRESHOLD, DEFAULT_DEVICE, DEFAULT_TRAFFIC,
    DEFAULT_PRED_WIN, DEFAULT_EMERGENCY, QOS_PROFILES, SEQ_LEN,
)
from modules.predictor        import load_lstm, predict_horizon
from modules.anomaly_detector import load_isolation_forest, get_anomaly_score
from modules.confidence       import normalise_error, confidence_score, escalation_score
from modules.intent           import get_qos_profile, compute_qos_risk
from modules.qos              import compute_qos, default_slice
from modules.escalation       import run_escalation
from modules.resource_allocator import allocate_resources
from modules.optimizer        import compare_strategies, compute_cost, energy_model
from modules.dashboard        import (
    fig_traffic_prediction, fig_anomaly_scores, fig_confidence_gauge,
    fig_llm_invocations, fig_qos_dashboard, fig_cost_comparison, fig_slice_allocation,
)
from modules.preprocessing    import (
    generate_dataset, clean_data, engineer_features, scale_features, FEATURE_COLS,
)


# ── Model loader (cached) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    lstm = load_lstm()
    iso  = load_isolation_forest()
    try:
        with open(THRESHOLD_PATH) as f:
            tau = json.load(f)["optimal_threshold"]
    except Exception:
        tau = DEFAULT_THRESHOLD
    return lstm, iso, tau


@st.cache_data(show_spinner="Generating dataset…")
def get_demo_data():
    df = generate_dataset()
    df = clean_data(df)
    df = engineer_features(df)
    df, _ = scale_features(df, fit=False)
    return df


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=64)
    st.title("⚙️ Control Panel")
    st.markdown("---")

    device_type = st.selectbox(
        "🏥 Device / Intent Type",
        list(QOS_PROFILES.keys()),
        index=list(QOS_PROFILES.keys()).index(DEFAULT_DEVICE),
    )
    traffic_input = st.slider(
        "📶 Current Traffic (Mbps)", 5.0, 200.0, DEFAULT_TRAFFIC, 1.0
    )
    pred_window = st.slider(
        "🔭 Prediction Horizon (steps)", 1, 30, DEFAULT_PRED_WIN
    )
    threshold = st.slider(
        "τ Escalation Threshold", 0.40, 0.95, DEFAULT_THRESHOLD, 0.01
    )
    emergency = st.toggle("🚨 Emergency Mode", value=DEFAULT_EMERGENCY)

    st.markdown("---")
    st.markdown("**QoS Requirements**")
    profile = get_qos_profile(device_type)
    st.metric("Latency",     f"{profile['latency_ms']} ms")
    st.metric("Bandwidth",   f"{profile['bandwidth_mbps']} Mbps")
    st.metric("Reliability", f"{profile['reliability']} %")
    st.metric("Priority",    profile["priority"])

    st.markdown("---")
    run_btn = st.button("▶ Run Framework", type="primary", use_container_width=True)


# ── Header ─────────────────────────────────────────────────────────────────

st.title("🏥 Adaptive Escalation Framework")
st.caption(
    "Intent-Aware AI-Native Network Slicing · 6G Healthcare Networks · "
    "LSTM + Isolation Forest + Groq LLM"
)
st.markdown("---")

# ── Load models ────────────────────────────────────────────────────────────

models_ok = (
    LSTM_MODEL_PATH.exists()
    and ISO_FOREST_PATH.exists()
    and SCALER_PATH.exists()
)

if not models_ok:
    st.warning(
        "⚠️ Trained models not found. "
        "Run Notebooks 1–4 first, or click **Bootstrap Demo** below."
    )
    if st.button("🚀 Bootstrap Demo (train models now)"):
        with st.spinner("Training LSTM & Isolation Forest — this takes ~2 min…"):
            from modules.preprocessing import run_pipeline
            from modules.predictor import train_lstm
            from modules.anomaly_detector import train_isolation_forest

            df, scaler, X_tr, X_te, y_tr, y_te = run_pipeline(save=True)
            val_split = int(len(X_tr) * 0.9)
            model, _ = train_lstm(X_tr[:val_split], y_tr[:val_split],
                                  X_tr[val_split:], y_tr[val_split:])
            feat_cols = [c for c in FEATURE_COLS if c in df.columns]
            train_isolation_forest(df[feat_cols].values)

            # Auto-compute threshold
            from modules.anomaly_detector import batch_scores
            from modules.confidence import escalation_score
            X_feat = df[feat_cols].values
            a_sc   = batch_scores(load_isolation_forest(), X_feat)
            traffic = df["traffic_mbps"].values
            pe     = np.clip(np.abs(np.diff(traffic, prepend=traffic[0])) / (traffic.max() + 1e-8), 0, 1)
            qr     = np.clip(df["packet_loss"].values if "packet_loss" in df.columns else np.zeros(len(df)), 0, 1)
            esc_sc = np.array([escalation_score(pe[i], a_sc[i], qr[i]) for i in range(len(df))])
            y_true = df["anomaly_label"].values
            from sklearn.metrics import f1_score
            best_tau, best_f1 = DEFAULT_THRESHOLD, 0
            for tau_c in np.linspace(0.4, 0.95, 56):
                f1 = f1_score(y_true, (esc_sc > tau_c).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_tau = f1, float(tau_c)
            THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(THRESHOLD_PATH, "w") as fp:
                json.dump({"optimal_threshold": best_tau, "f1": best_f1}, fp)
        st.success("✅ Models trained! Refresh the page.")
    st.stop()

lstm_model, iso_clf, opt_tau = load_models()

# ── Single-step inference ──────────────────────────────────────────────────

try:
    df_demo = get_demo_data()
    feat_cols = [c for c in FEATURE_COLS if c in df_demo.columns]
    traffic_series = df_demo["traffic_mbps"].values
    seed_window    = traffic_series[-SEQ_LEN:]
    pred_series    = predict_horizon(lstm_model, seed_window, pred_window)
    predicted_now  = float(pred_series[0])
except Exception as e:
    st.error(f"Prediction error: {e}")
    predicted_now = traffic_input
    pred_series   = np.full(pred_window, traffic_input)

# Anomaly score for current feature vector
try:
    x_feat    = df_demo[feat_cols].values[-1]
    a_score   = get_anomaly_score(iso_clf, x_feat)
except Exception:
    a_score   = 0.0

# Escalation
norm_err   = normalise_error(traffic_input, predicted_now)
conf       = confidence_score(norm_err)
qos_risk   = compute_qos_risk(device_type, profile["latency_ms"],
                               traffic_input * 0.8, profile["reliability"])
esc        = escalation_score(norm_err, a_score, qos_risk)
llm_needed = (esc > threshold) or emergency

# QoS score
qos_score  = compute_qos(traffic_input * 0.8, profile["reliability"],
                          profile["latency_ms"], device_type)

# Slice decision
if run_btn and llm_needed:
    from modules.llm import invoke_llm
    context = {
        "device_type": device_type, "traffic_mbps": traffic_input,
        "predicted_mbps": predicted_now, "pred_error": norm_err,
        "confidence": conf, "anomaly_score": a_score,
        "escalation_score": esc, "latency_ms": profile["latency_ms"],
        "bandwidth_mbps": traffic_input * 0.8,
        "reliability": profile["reliability"],
        "current_slice": device_type, "qos_score": qos_score,
        "emergency": emergency,
    }
    with st.spinner("🤖 Invoking Groq LLM…"):
        slice_rec = invoke_llm(context)
else:
    slice_rec = default_slice(device_type)

alloc = allocate_resources(predicted_now, slice_rec, llm_needed, device_type)

# ── KPI Row ────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📶 Traffic",       f"{traffic_input:.1f} Mbps")
c2.metric("🔮 Predicted",     f"{predicted_now:.3f} Mbps")
c3.metric("🎯 Confidence",    f"{conf*100:.1f} %")
c4.metric("⚡ Esc. Score",    f"{esc:.3f}",
          delta=f"τ={threshold:.2f}", delta_color="inverse")
c5.metric("🤖 LLM Invoked",   "YES 🔴" if llm_needed else "NO 🟢")

st.markdown("---")

# ── Row 1: Traffic + Anomaly ───────────────────────────────────────────────

col_l, col_r = st.columns(2)

with col_l:
    st.subheader("📈 Traffic Prediction")
    hist_len = min(100, len(traffic_series))
    hist     = traffic_series[-hist_len:]
    ts_hist  = list(range(-hist_len, 0))
    ts_pred  = list(range(0, pred_window))
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=ts_hist, y=hist, name="Historical",
                               line=dict(color="#00B4D8")))
    fig_t.add_trace(go.Scatter(x=ts_pred, y=pred_series, name="Forecast",
                               line=dict(color="#F77F00", dash="dash")))
    fig_t.add_vline(x=0, line_dash="dot", line_color="white")
    fig_t.update_layout(template="plotly_dark", height=320,
                        xaxis_title="Steps", yaxis_title="Traffic (norm.)")
    st.plotly_chart(fig_t, use_container_width=True)

with col_r:
    st.subheader("🔍 Isolation Forest Scores")
    sample_n = min(200, len(df_demo))
    from modules.anomaly_detector import batch_scores, batch_labels
    a_sc_arr = batch_scores(iso_clf, df_demo[feat_cols].values[-sample_n:])
    a_lb_arr = batch_labels(iso_clf, df_demo[feat_cols].values[-sample_n:])
    st.plotly_chart(
        fig_anomaly_scores(list(range(sample_n)), a_sc_arr, a_lb_arr),
        use_container_width=True,
    )

# ── Row 2: Confidence Gauge + LLM Indicator ───────────────────────────────

col_g, col_llm = st.columns(2)

with col_g:
    st.subheader("🎯 Confidence & Escalation")
    st.plotly_chart(
        fig_confidence_gauge(conf, esc, threshold),
        use_container_width=True,
    )

with col_llm:
    st.subheader("🤖 LLM Decision")
    if llm_needed:
        st.error("### 🔴 LLM INVOKED")
        st.json(slice_rec)
    else:
        st.success("### 🟢 LLM NOT REQUIRED")
        st.info(f"Using default profile: **{device_type}**")
        st.json(slice_rec)

# ── Row 3: QoS Dashboard ──────────────────────────────────────────────────

st.subheader("📊 QoS Dashboard")
qos_hist = np.clip(
    np.random.normal(qos_score, 0.05, 100), 0, 2
)  # simulated history for display
bw_hist  = np.clip(np.random.normal(alloc["allocated_bandwidth"], 5, 100), 0, 200)
lat_hist = np.clip(np.random.normal(alloc["allocated_latency"], 0.5, 100), 0.1, 50)
st.plotly_chart(
    fig_qos_dashboard(list(range(100)), qos_hist, bw_hist, lat_hist),
    use_container_width=True,
)

# ── Row 4: Slice Allocation + Cost ────────────────────────────────────────

col_sl, col_cost = st.columns(2)

with col_sl:
    st.subheader("🔧 Slice Allocation")
    st.plotly_chart(fig_slice_allocation(slice_rec), use_container_width=True)
    st.markdown(f"**Reason:** {slice_rec.get('reason','—')}")
    st.metric("Resource Utilisation", f"{alloc['utilisation']*100:.1f} %")
    st.metric("Resource Waste",       f"{alloc['resource_waste']:.1f} Mbps")

with col_cost:
    st.subheader("💰 Cost Function")
    energy = energy_model(alloc["allocated_bandwidth"], llm_needed)
    cost   = compute_cost(alloc["allocated_latency"], energy,
                          int(llm_needed), alloc["resource_waste"])
    st.metric("Cost", f"{cost:.4f}")
    st.metric("Energy", f"{energy:.2f} J")

    # Simulated comparison
    sim_results = []
    for _ in range(50):
        t = float(np.random.uniform(20, 150))
        p = t + float(np.random.normal(0, 5))
        a = float(np.random.uniform(0, 0.6))
        r = run_escalation(t, p, a, profile["latency_ms"],
                           t * 0.8, profile["reliability"],
                           device_type, threshold, False)
        al = allocate_resources(p, r["slice"], r["llm_invoked"], device_type)
        r.update(al)
        sim_results.append(r)

    comparison = compare_strategies(sim_results)
    st.plotly_chart(fig_cost_comparison(comparison), use_container_width=True)

# ── Row 5: Performance Comparison Table ───────────────────────────────────

st.subheader("📋 Strategy Performance Comparison")
import pandas as pd
comp_df = pd.DataFrame(comparison).T.reset_index().rename(columns={"index": "Strategy"})
comp_df = comp_df.round(4)
st.dataframe(comp_df, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Adaptive Escalation Framework · 6G Healthcare Network Slicing · "
    "LSTM + Isolation Forest + Groq LLM · IEEE Research Prototype"
)
