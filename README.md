# 6G Adaptive Escalation for Healthcare Network Slicing

An AI-driven research prototype for adaptive healthcare network slicing using traffic prediction, anomaly detection and selective LLM reasoning.

## Overview

The system uses lightweight ML models for routine network conditions and escalates only anomalous or low-confidence cases to an LLM. This is intended to reduce unnecessary LLM calls while supporting QoS-aware resource allocation.

## Pipeline

```text
Healthcare / IoT Telemetry
          |
          v
   Preprocessing
      /       \
     v         v
   LSTM    Isolation Forest
     |         |
     +----+----+
          |
          v
    Confidence + QoS
          |
          v
 Adaptive Escalation
      /       \
 Normal       Risk / Low Confidence
   |                 |
   v                 v
Local Decision       LLM Reasoning
      \               /
       +------v------+
       Slice Reconfiguration
              |
              v
     Resource Allocation
              |
              v
        QoS Monitoring
              |
              v
        Streamlit UI
```

## Key Features

- LSTM-based traffic prediction
- Isolation Forest anomaly detection
- Prediction-confidence estimation
- Selective LLM escalation
- QoS-aware resource allocation
- MQTT-based healthcare-device simulation
- Streamlit/Plotly monitoring dashboard

## Dataset

The prototype uses a synthetic healthcare network dataset containing approximately **10,000 records and 35 features**. Features include application type, intent, priority, slice type, latency, throughput, packet loss, jitter, bandwidth utilization, connected devices and traffic measurements.

The dataset is synthetic and should not be presented as real hospital or telecom traffic.

## Reported Evaluation

The research evaluation reported:

- **92.7–96.8% traffic prediction accuracy**
- **5–10% selective LLM invocation**

These figures are experimental results from the prototype and should not be interpreted as production 6G performance.

## Technology Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy / Pandas
- Streamlit / Plotly
- MQTT / Paho MQTT / Mosquitto
- Groq API
- Jupyter / Google Colab

## Project Structure

```text
6G-ADAPTIVE-ESCALATION/
├── app.py
├── config.py
├── requirements.txt
├── .env.example
├── modules/
├── notebooks/
├── simulator/
└── tests/
```

## Setup

```bash
git clone https://github.com/pradeep14012004/6G-ADAPTIVE-ESCALATION.git
cd 6G-ADAPTIVE-ESCALATION

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\\Scripts\\activate

pip install -r requirements.txt
```

Configure the required LLM credentials in a local `.env` file. **Never commit API keys.**

Run the dashboard:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest
```

## Limitations

This is a software research prototype. It does not provide validation on real 6G infrastructure, live hospital traffic, physical network hardware or a telecom testbed.

## Future Work

- 5G/6G testbed integration
- Realistic network traces
- SDN/NFV orchestration
- Edge deployment
- Reinforcement-learning-based slice optimization
- Real IoMT integration
- Multi-domain network slicing

## Research Areas

**AI/ML · 6G Networks · Network Slicing · Healthcare IoT · LLM-based Networking · Intelligent Network Management**
