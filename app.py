"""
app.py — Streamlit Dashboard
Adaptive Escalation Framework for Intent-Aware AI-Native Network Slicing in 6G Healthcare Networks
"""

import json
import time
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

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
from modules.sim_state        import sim_state
from modules.mqtt_subscriber  import start as start_subscriber


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

tab1, tab2 = st.tabs(["📊 Framework Analysis", "🔴 Live Simulator"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — original framework analysis (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

with tab1:

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

    # ── Single-step inference ──────────────────────────────────────────────

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

    try:
        x_feat  = df_demo[feat_cols].values[-1]
        a_score = get_anomaly_score(iso_clf, x_feat)
    except Exception:
        a_score = 0.0

    norm_err  = normalise_error(traffic_input, predicted_now)
    conf      = confidence_score(norm_err)
    qos_risk  = compute_qos_risk(device_type, profile["latency_ms"],
                                  traffic_input * 0.8, profile["reliability"])
    esc       = escalation_score(norm_err, a_score, qos_risk)
    llm_needed = (esc > threshold) or emergency
    qos_score  = compute_qos(traffic_input * 0.8, profile["reliability"],
                              profile["latency_ms"], device_type)

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

    # ── KPI Row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📶 Traffic",     f"{traffic_input:.1f} Mbps")
    c2.metric("🔮 Predicted",   f"{predicted_now:.3f} Mbps")
    c3.metric("🎯 Confidence",  f"{conf*100:.1f} %")
    c4.metric("⚡ Esc. Score",  f"{esc:.3f}",
              delta=f"τ={threshold:.2f}", delta_color="inverse")
    c5.metric("🤖 LLM Invoked", "YES 🔴" if llm_needed else "NO 🟢")

    st.markdown("---")

    # ── Row 1: Traffic + Anomaly ──────────────────────────────────────────
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

    # ── Row 2: Confidence Gauge + LLM ────────────────────────────────────
    col_g, col_llm = st.columns(2)

    with col_g:
        st.subheader("🎯 Confidence & Escalation")
        st.plotly_chart(fig_confidence_gauge(conf, esc, threshold), use_container_width=True)

    with col_llm:
        st.subheader("🤖 LLM Decision")
        if llm_needed:
            st.error("### 🔴 LLM INVOKED")
            st.json(slice_rec)
        else:
            st.success("### 🟢 LLM NOT REQUIRED")
            st.info(f"Using default profile: **{device_type}**")
            st.json(slice_rec)

    # ── Row 3: QoS Dashboard ──────────────────────────────────────────────
    st.subheader("📊 QoS Dashboard")
    qos_hist = np.clip(np.random.normal(qos_score, 0.05, 100), 0, 2)
    bw_hist  = np.clip(np.random.normal(alloc["allocated_bandwidth"], 5, 100), 0, 200)
    lat_hist = np.clip(np.random.normal(alloc["allocated_latency"], 0.5, 100), 0.1, 50)
    st.plotly_chart(
        fig_qos_dashboard(list(range(100)), qos_hist, bw_hist, lat_hist),
        use_container_width=True,
    )

    # ── Row 4: Slice Allocation + Cost ────────────────────────────────────
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
        st.metric("Cost",   f"{cost:.4f}")
        st.metric("Energy", f"{energy:.2f} J")

        sim_results = []
        for _ in range(50):
            t  = float(np.random.uniform(20, 150))
            p  = t + float(np.random.normal(0, 5))
            a  = float(np.random.uniform(0, 0.6))
            r  = run_escalation(t, p, a, profile["latency_ms"],
                                t * 0.8, profile["reliability"],
                                device_type, threshold, False)
            al = allocate_resources(p, r["slice"], r["llm_invoked"], device_type)
            r.update(al)
            sim_results.append(r)

        comparison = compare_strategies(sim_results)
        st.plotly_chart(fig_cost_comparison(comparison), use_container_width=True)

    # ── Row 5: Performance Comparison Table ───────────────────────────────
    st.subheader("📋 Strategy Performance Comparison")
    comp_df = pd.DataFrame(comparison).T.reset_index().rename(columns={"index": "Strategy"})
    st.dataframe(comp_df.round(4), use_container_width=True)

    st.markdown("---")
    st.caption(
        "Adaptive Escalation Framework · 6G Healthcare Network Slicing · "
        "LSTM + Isolation Forest + Groq LLM · IEEE Research Prototype"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Simulator (MQTT virtual devices → existing models)
# ══════════════════════════════════════════════════════════════════════════════

with tab2:

    st.subheader("🔴 Live Virtual Hospital Network Simulator")
    st.caption("20 virtual devices · MQTT · LSTM + Isolation Forest + Policy Engine · Closed-loop control")

    # ── Start MQTT subscriber once ────────────────────────────────────────
    if "subscriber_started" not in st.session_state:
        try:
            _lstm, _iso, _tau = load_models()
            start_subscriber(_lstm, _iso, _tau)
            st.session_state.subscriber_started = True
        except Exception as e:
            st.error(f"Could not start subscriber: {e}")

    # ── Instructions ──────────────────────────────────────────────────────
    with st.expander("▶ How to start the simulator", expanded=False):
        st.code(
            "# Terminal 1\nmosquitto\n\n# Terminal 2\npython -m simulator.publisher",
            language="bash",
        )

    # ── Failure Injection ─────────────────────────────────────────────────
    st.subheader("💥 Failure Injection")
    fi1, fi2, fi3, fi4 = st.columns(4)
    inject_icu       = fi1.button("🏥 ICU Failure")
    inject_ddos      = fi2.button("💀 Wearable Flood")
    inject_ambulance = fi3.button("🚑 Ambulance Emergency")
    inject_surgery   = fi4.button("🔪 Surgery Spike")

    from modules.network_manager import send_control
    devices_now = sim_state.get_all()

    if inject_icu:
        for did, d in devices_now.items():
            if d["device_type"] == "ICU Monitoring":
                send_control(did, {"publish_interval": 200, "max_bandwidth": 0.1}, "ICU Failure injected")
        sim_state.add_event("💥 INJECTED: ICU Failure — all ICU devices throttled")

    if inject_ddos:
        for did, d in devices_now.items():
            if d["device_type"] == "Wearable Devices":
                send_control(did, {"publish_interval": 100, "max_bandwidth": 50}, "DDoS flood injected")
        sim_state.add_event("💥 INJECTED: Wearable Flood — bandwidth spike on all wearables")

    if inject_ambulance:
        for did, d in devices_now.items():
            if d["device_type"] == "Emergency Ambulance":
                send_control(did, {"emergency": True, "publish_interval": 200}, "Emergency injected")
        sim_state.add_event("💥 INJECTED: Ambulance Emergency — all ambulances activated")

    if inject_surgery:
        for did, d in devices_now.items():
            if d["device_type"] == "Remote Surgery":
                send_control(did, {"max_bandwidth": 150, "publish_interval": 200}, "Surgery spike injected")
        sim_state.add_event("💥 INJECTED: Surgery Spike — max bandwidth on all surgery devices")

    st.markdown("---")
    auto_refresh = st.toggle("🔄 Auto-refresh (1 s)", value=True)
    placeholder  = st.empty()

    def render_live():
        devices = sim_state.get_all()

        if not devices:
            placeholder.warning("⏳ Waiting for telemetry… Start Mosquitto and simulator/publisher.py")
            return

        rows = list(devices.values())
        df   = pd.DataFrame(rows)

        active    = int((df["status"] != "DISCONNECTED").sum())
        total_bw  = float(df["bandwidth"].sum())
        avg_lat   = float(df["latency"].mean())
        anomalies = int(df["anomaly_score"].gt(0.6).sum()) if "anomaly_score" in df.columns else 0
        escalated = int(df["llm_needed"].sum())            if "llm_needed"    in df.columns else 0
        congested = bool(df["congested"].any())            if "congested"     in df.columns else False
        net_usage = float(df["total_network_usage"].max()) if "total_network_usage" in df.columns else total_bw

        from modules.network_manager import get_slice_state, get_action_log, NETWORK_CAPACITY_MBPS
        slice_state = get_slice_state()
        action_log  = get_action_log(10)

        with placeholder.container():
            # KPIs
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Total Devices",    len(devices))
            k2.metric("Active",           active)
            k3.metric("Total BW (Mbps)",  f"{total_bw:.1f}")
            k4.metric("Avg Latency (ms)", f"{avg_lat:.2f}")
            k5.metric("🔴 Anomalies",     anomalies)
            k6.metric("⚡ Escalations",   escalated)

            util_pct = min(net_usage / NETWORK_CAPACITY_MBPS, 1.0)
            st.markdown(f"**Network Utilisation: {net_usage:.0f} / {NETWORK_CAPACITY_MBPS:.0f} Mbps ({'🔴 CONGESTED' if congested else '🟢 OK'})**")
            st.progress(util_pct)
            st.markdown("---")

            # Slice state
            st.subheader("🌐 Network Slice Allocation")
            sc1, sc2, sc3 = st.columns(3)
            for col, (sname, sdata) in zip([sc1, sc2, sc3], slice_state.items()):
                col.metric(sname, f"{sdata['used']:.1f} / {sdata['capacity']:.0f} Mbps")
                col.progress(min(sdata["used"] / max(sdata["capacity"], 1), 1.0))
            st.markdown("---")

            # Device table
            st.subheader("📋 Device Status")
            cols = ["device_id", "device_type", "bandwidth", "latency", "packet_loss", "priority", "status"]
            for c in ["device_state","predicted_bw","anomaly_score","escalation_score",
                      "llm_needed","allocated_bandwidth","allocated_latency","policy_action"]:
                if c in df.columns: cols.append(c)
            tbl = df[cols].sort_values("priority", ascending=False).reset_index(drop=True)
            st.dataframe(tbl, use_container_width=True, height=300)

            llm_rows = [d for d in rows if d.get("llm_response")]
            if llm_rows:
                st.subheader("🤖 LLM Slice Decisions")
                for d in llm_rows:
                    with st.expander(f"{d['device_id']} — {d['llm_response'].get('slice','?')} [{d['llm_response'].get('priority','?')}]"):
                        st.json(d["llm_response"])
            st.markdown("---")

            # AI Impact Metrics
            st.subheader("🎯 AI Impact Metrics")
            m1, m2, m3, m4 = st.columns(4)
            throttled  = int((df["device_state"] == "THROTTLED").sum())  if "device_state" in df.columns else 0
            recovering = int((df["device_state"] == "RECOVERING").sum()) if "device_state" in df.columns else 0
            m1.metric("Devices Throttled",  throttled)
            m2.metric("Devices Recovering", recovering)
            m3.metric("Policy Actions",     len(action_log))
            m4.metric("LLM Calls",          int(df["llm_needed"].sum()) if "llm_needed" in df.columns else 0)
            if action_log:
                st.dataframe(pd.DataFrame(action_log), use_container_width=True, height=200)
            st.markdown("---")

            # Charts
            st.subheader("📈 Live Metrics by Device Type")
            c1, c2 = st.columns(2)
            with c1:
                type_bw = df.groupby("device_type")["bandwidth"].sum().reset_index()
                fig_bw  = px.bar(type_bw, x="device_type", y="bandwidth", color="device_type",
                                 template="plotly_dark", labels={"bandwidth": "Total BW (Mbps)"})
                fig_bw.update_layout(height=280, showlegend=False, margin=dict(t=20, b=20))
                st.plotly_chart(fig_bw, use_container_width=True)
            with c2:
                if "anomaly_score" in df.columns:
                    fig_a = px.scatter(df, x="device_id", y="anomaly_score", color="device_type",
                                       template="plotly_dark", labels={"anomaly_score": "Anomaly Score"})
                    fig_a.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="alert threshold")
                    fig_a.update_layout(height=280, margin=dict(t=20, b=20))
                    st.plotly_chart(fig_a, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                fig_lat = px.box(df, x="device_type", y="latency", color="device_type",
                                 template="plotly_dark", labels={"latency": "Latency (ms)"})
                fig_lat.update_layout(height=260, showlegend=False, margin=dict(t=20, b=20))
                st.plotly_chart(fig_lat, use_container_width=True)
            with c4:
                if "allocated_bandwidth" in df.columns:
                    fig_alloc = px.bar(df.dropna(subset=["allocated_bandwidth"]),
                                       x="device_id", y=["bandwidth","allocated_bandwidth"],
                                       barmode="group", template="plotly_dark", labels={"value": "Mbps"})
                    fig_alloc.update_layout(height=260, margin=dict(t=20, b=20), xaxis_tickangle=-45)
                    st.plotly_chart(fig_alloc, use_container_width=True)
            st.markdown("---")

            # Event Timeline
            st.subheader("🕐 Event Timeline")
            events = sim_state.get_events(25)
            if events:
                for e in reversed(events):
                    st.text(e)
            else:
                st.info("No events yet.")

    render_live()

    if auto_refresh:
        time.sleep(1)
        st.rerun()
