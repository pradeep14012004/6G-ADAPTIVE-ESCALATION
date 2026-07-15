# Adaptive Escalation Framework for Intent-Aware AI-Native Network Slicing in 6G Healthcare Networks

> B.Tech Final Year Project · IEEE Research Prototype

---

## Architecture

```
Traffic Data
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  LSTM Traffic Predictor  →  D̂ = f_LSTM(D)                  │
│  Isolation Forest        →  Anomaly Score A_t               │
│  Confidence Module       →  C = exp(-λ·E_t)                 │
│                                                             │
│  Escalation Score:  S = 0.5·E_t + 0.3·A_t + 0.2·Q_t       │
│                                                             │
│  if S > τ  →  Invoke Groq LLM  →  JSON Slice Rec.          │
│  else      →  Default QoS Profile                           │
│                                                             │
│  Resource Allocator  →  R = α · D̂                          │
│  QoS Evaluator       →  w1·BW + w2·Rel - w3·Lat            │
│  Cost Function       →  α·Lat + β·E + γ·LLM + δ·Waste      │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
AdaptiveEscalation6G/
├── notebooks/
│   ├── 1_Data_Preprocessing.ipynb
│   ├── 2_LSTM_Traffic_Prediction.ipynb
│   ├── 3_IsolationForest_Training.ipynb
│   ├── 4_Threshold_Optimization.ipynb
│   └── 5_Adaptive_Escalation_Framework.ipynb
├── modules/
│   ├── preprocessing.py      # Data pipeline
│   ├── predictor.py          # LSTM model
│   ├── anomaly_detector.py   # Isolation Forest
│   ├── confidence.py         # Escalation score
│   ├── intent.py             # QoS profiles & risk
│   ├── qos.py                # QoS computation
│   ├── escalation.py         # Main orchestrator
│   ├── llm.py                # Groq LLM integration
│   ├── resource_allocator.py # Dynamic allocation
│   ├── optimizer.py          # Cost function & comparison
│   └── dashboard.py          # Plotly figure factories
├── models/                   # Saved models (generated)
├── data/                     # Dataset (generated)
├── app.py                    # Streamlit dashboard
├── config.py                 # All parameters
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Groq API key

```bash
# Windows
set GROQ_API_KEY=your_key_here

# Linux / macOS
export GROQ_API_KEY=your_key_here
```

### 3. Run notebooks in order

```
notebooks/1_Data_Preprocessing.ipynb
notebooks/2_LSTM_Traffic_Prediction.ipynb
notebooks/3_IsolationForest_Training.ipynb
notebooks/4_Threshold_Optimization.ipynb
notebooks/5_Adaptive_Escalation_Framework.ipynb
```

### 4. Launch dashboard

```bash
streamlit run app.py
```

---

## Key Equations

| Symbol | Formula | Description |
|--------|---------|-------------|
| D̂ | f_LSTM(D) | LSTM traffic prediction |
| E_t | \|D - D̂\| / D_max | Normalised prediction error |
| C | exp(-λ·E_t) | Confidence score |
| A_t | -decision_function(x) | Normalised anomaly score |
| Q_t | f(latency, bandwidth, reliability) | QoS risk |
| S | 0.5·E_t + 0.3·A_t + 0.2·Q_t | **Adaptive Escalation Score** |
| QoS | w1·BW/BW_req + w2·Rel/Rel_req - w3·Lat/Lat_req | QoS metric |
| Cost | α·Lat + β·Energy + γ·LLMCalls + δ·Waste | Optimisation objective |

## QoS Profiles

| Device | Latency | Bandwidth | Reliability |
|--------|---------|-----------|-------------|
| Remote Surgery | 1 ms | 150 Mbps | 99.999% |
| ICU Monitoring | 5 ms | 80 Mbps | 99.99% |
| Wearable Devices | 20 ms | 20 Mbps | 99.9% |
| Emergency Ambulance | 2 ms | 100 Mbps | 99.999% |

## LLM Integration

The Groq LLM is invoked **only when S > τ** (adaptive escalation).
It returns structured JSON — never plain English:

```json
{
  "slice": "Emergency",
  "bandwidth": 180,
  "latency": 1,
  "priority": "Critical",
  "reason": "Traffic spike with high anomaly score detected"
}
```

This minimises unnecessary LLM calls while ensuring expert-level
reconfiguration when the network state is uncertain or risky.
