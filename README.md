# 6G Adaptive Escalation for Healthcare Network Slicing

Research prototype exploring selective escalation from lightweight predictive and rule-based control to an LLM-assisted decision layer in a **synthetic healthcare-network simulation**.

> **Research integrity notice:** this repository is a simulation/prototype. It is not connected to a real hospital, carrier network, or 6G testbed. Synthetic results must not be described as measured clinical or production 6G performance. Previously reported accuracy, latency, energy, and LLM-call figures are preliminary and must be regenerated using the corrected protocol before being quoted in a paper.

## Research question

Can a causal, confidence- and QoS-aware escalation policy reduce LLM invocation while preserving service-level agreement (SLA) compliance under controlled healthcare traffic scenarios?

## Corrected research safeguards

- modules/preprocessing.py: rolling features are shifted so a row's rolling statistics use past observations; scaler fit is restricted to the chronological training interval.
- modules/confidence.py: forecast error is normalized by an explicit Mbps scale.
- modules/anomaly_detector.py: Isolation Forest scores are mapped using bounds learned from training scores; they are not calibrated probabilities.
- modules/escalation.py: current-step observed traffic error is evaluation-only. Online decisions use prior_norm_error from an already-observed forecast residual.
- modules/policy_engine.py: anomaly isolation and emergency actions take precedence; the escalation threshold is configured centrally.
- modules/qos.py: QoS score is bounded to [0,1], with individual SLA violations exposed separately.
- modules/threshold_optimizer.py: threshold selection is validation-only; the test set must remain untouched.
- modules/evaluation.py: provides forecasting, anomaly, and policy metrics without inventing latency or energy measurements.
- modules/optimizer.py: energy is explicitly an illustrative proxy; strategy summaries do not claim fair measured baseline comparisons.

## Data

Default synthetic generation is 10,000 time steps at 10-minute intervals, with four healthcare application classes and configurable injected anomalies. This is a controllable simulation, not a validated model of hospital or telecom traffic. Do not describe it as a real-world dataset.

For external validity, add independent public network traces and/or a documented packet-level simulation. Keep training, validation, and test periods separate and publish seeds, scenario parameters, and split boundaries.

## Reproducibility and tests

Install dependencies from requirements.txt, then run: python -m pytest tests/ -v

Run notebooks from top to bottom in a clean environment. The notebooks are exploratory; modules/ is the canonical implementation.

## Required experiment design before publication

Compare independently executed policies under identical traffic traces, network capacity, and initial conditions: Always-LLM; local rules/no LLM; LSTM forecast plus deterministic policy; anomaly-only escalation; proposed causal adaptive escalation.

Report MAE/RMSE for forecasting; precision, recall, F1, ROC-AUC and PR-AUC for anomaly detection; LLM-call rate; measured median/p95 decision latency; SLA compliance by service class; bandwidth utilization and resource waste. Measure actual wall-clock/API latency. Do not call a formula-based energy proxy measured energy.

Include multiple random seeds, anomaly rates, offered-load levels, capacity settings, and confidence intervals. Tune thresholds on validation only. Use the held-out test set once for final results.

## Limitations

- Synthetic data only unless an external trace is explicitly added and documented.
- No physical 5G/6G RAN, SDN/NFV, edge hardware, or hospital deployment validation.
- Isolation Forest scores are anomaly ranking scores, not probabilities.
- LLM outputs are not safety-certified; deterministic emergency policies must remain authoritative.
- SLA targets are simulation assumptions and should be sourced/justified in a publication.

## Project structure

app.py; config.py; modules/; notebooks/; simulator/; tests/

## Publication claims

Do not reuse historical performance numbers in an abstract or paper until corrected experiments have been rerun and independently checked. Distinguish simulation outputs, software/API timings, and physical network measurements.
