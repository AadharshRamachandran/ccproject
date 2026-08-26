# Detailed Rationale and Evaluation of Novelty Extensions

This document provides a comprehensive explanation of the four architectural novelties implemented in this project over the Vu, Tran, and Kim (2022) baseline, including their theoretical rationales, code structures, and empirical outcomes from the ablation study.

---

## 1. Validation-Tuned Adaptive Burst Detection (`tuned_adaptive`)

### The Problem in the Paper
The baseline paper utilizes an online Z-score detector (Algorithm 1) to identify workload spikes. However, it hardcodes the parameters: `threshold = 5.0` and `influence = 0.5`. There is no validation or mathematical proof that these parameters generalize to workloads with different characteristics (e.g., FIFA World Cup vs. Wikipedia traces).

### Our Novelty Extension
We implement a **validation-tuned, volatility-adaptive burst detector**:
1. **Validation Optimization:** Before evaluation, the script reserves a validation slice of the training data. It runs a parameter search to find optimal values for `threshold`, `influence`, and `minimum_threshold` that minimize total resource cost subject to the paper's target SLO threshold (3.68%).
2. **Volatility Adaptivity:** Instead of a static threshold, the detector computes a rolling coefficient of variation ($CV = \sigma / \mu$). When volatility is low, the threshold decreases to catch subtle spikes early. During high-volatility noise, the threshold increases to prevent false-alarm over-provisioning.

* **Code Reference:** [`Novelty/src/burst.py`](file:///c:/Users/HP/OneDrive/Desktop/Project/Cloud%20computing%20project/Novelty/src/burst.py) (see `AdaptiveBurstDetector` and `tune_burst_parameters`)

### Ablation Findings
On the FIFA World Cup trace, the tuned burst policy selected conservative parameters to prioritize SLO safety. This increased the resource cost (from **126.9** to **139.4**) without providing a proportional SLO improvement. This indicates that the paper's default parameters (5.0, 0.5) were already highly optimized for the World Cup trace's specific spike density.

---

## 2. Uncertainty-Aware Quantile Provisioning (`quantile_bilstm`)

### The Problem in the Paper
The baseline paper uses a standard deterministic Bi-LSTM, which outputs a single point prediction ($y_{t+1}$). During bursts, the scaler has no concept of prediction uncertainty and blindly triggers a hardcoded backup rule: scaling up to the maximum allowable CPU limit (e.g., 950 millicores) for safety. This causes heavy over-provisioning when point forecasts are uncertain.

### Our Novelty Extension
We replace the point forecaster with a **probabilistic Quantile Bi-LSTM**:
1. **Quantile Forecasting:** The neural network is trained using Pinball Loss to output three quantiles: P10, P50 (median), and P90.
2. **Gated Uncertainty Scaling:** 
   * **Normal state:** The scaler targets the P50 forecast using the standard DTR performance model.
   * **Burst state:** The scaler dynamically switches to the P90 forecast. The workload prediction is boosted proportionally to the *prediction interval width* ($P90 - P10$), adjusting the CPU allocation depending on how uncertain the model is.

* **Code Reference:** [`Novelty/src/forecast.py`](file:///c:/Users/HP/OneDrive/Desktop/Project/Cloud%20computing%20project/Novelty/src/forecast.py) (see `QuantileBiLSTMForecaster`) and [`Novelty/src/scaler.py`](file:///c:/Users/HP/OneDrive/Desktop/Project/Cloud%20computing%20project/Novelty/src/scaler.py) (see `UncertaintyAwareProvisioner`)

### Ablation Findings
This novelty was highly successful. By incorporating P90 safety buffers dynamically instead of using a rigid maximum-capacity fallback, SLO violations were cut from **1.597%** to **0.625%** (a **60% reduction in violations**), while keeping the resource cost practically identical (**124.7** vs. **126.9**).

---

## 3. Pod In-Place Resource Resizing (`inplace`)

### The Problem in the Paper
In 2022, dynamically changing container CPU allocations (vertical scaling) required a rolling update. The Kubernetes API destroyed the old Pod and deployed a new one. This created a resource overlap period (Eq. 4 in the paper) where both pods ran concurrently, incurring high temporary cost overhead and potential replication latency.

### Our Novelty Extension
With the release of Kubernetes v1.35+, we implement an **In-Place Pod Resize** backend:
1. **Symbolic Patching:** The vertical resize is done by issuing a `PATCH` request to the Pod's `spec.containers.resources` subresource.
2. **Zero-Overlap Actuation:** Because the container is resized in-place without restart or destruction (`resizePolicy: NotRequired` for CPU), the old/new deployment overlap cost is mathematically **zero**.

* **Code Reference:** [`Novelty/src/kubernetes_executor.py`](file:///c:/Users/HP/OneDrive/Desktop/Project/Cloud%20computing%20project/Novelty/src/kubernetes_executor.py) (see `InPlaceResizeExecutor`)

### Ablation Findings
This is a strict Pareto improvement. Switching from `rolling` to `inplace` execution eliminated the rolling overlap cost completely (reducing it from **7.38** to **0.00**), while slightly improving the average SLO violation rate (from **1.597%** to **0.833%**) due to the faster actuation speed of API patches.

---

## 4. Adaptive Monitoring Cadence (`adaptive`)

### The Problem in the Paper
The baseline paper uses a fixed monitoring interval of 60 seconds. Polling Prometheus every 60 seconds is wasteful during flat, stable traffic periods, but may be too slow to catch transient spikes during bursty periods.

### Our Novelty Extension
We implement a **workload-adaptive monitoring interval**:
1. **Dynamic Scaling:** The interval scales between a minimum of 15 seconds (high volatility/bursts) and a maximum of 180 seconds (quiescent traffic).
2. **Volative and Width Gates:** The interval size is updated using both the rolling load coefficient of variation ($CV$) and the forecaster's predicted relative uncertainty interval width.

* **Code Reference:** [`Novelty/src/monitoring.py`](file:///c:/Users/HP/OneDrive/Desktop/Project/Cloud%20computing%20project/Novelty/src/monitoring.py) (see `AdaptiveMonitoringInterval`)

### Ablation Findings
Adaptive monitoring reduced the total number of controller execution cycles by **~40%** (from **1440** monitoring runs to **876** runs) with negligible impact on the SLO violation rate, proving that variable-cadence monitoring is an effective way to lower API and controller overhead.

---