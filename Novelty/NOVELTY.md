# Detailed Rationale and Evaluation of Novelty Extensions

This document provides a comprehensive explanation of the four architectural novelties implemented in this project over the Vu, Tran, and Kim (2022) baseline, including their theoretical rationales, code structures, and empirical outcomes from the ablation study.

## Corrected full-trace run (supersedes earlier results below)

The full factorial run is `Novelty/results/factorial_full_trace_seed42/`: 125,310 WorldCup98 samples, chronological 80/20 development/held-out split, 25,062 held-out ticks, seed 42, 10 epochs, and the synthetic performance fixture. It contains Paper Hybrid, all four isolated novelties, and all 11 combinations. Paper Hybrid recorded 0.01596% SLO violations, 13.102% mean utilization, and 903.638 resource-cost units. The all-four arm recorded 0.00399% SLO violations, 13.113% utilization, and 902.328 cost units. Its paired 2,000-draw block-bootstrap SLO interval reaches zero, so the study does not establish a reliable SLO improvement. The emulator and resize accounting are synthetic; these are not measured cluster results. See the run report, summary, paired effects, and metadata in that folder for all configurations.

N2 and N4 have lower SLO point estimates; N1 and N4 have lower cost point estimates. No novelty has paired 95% confidence intervals below zero on both outcomes. The synthetic capacity formula and closed response-time curve are not independently measured. N1/N4 cost changes are not proof of real performance; in-place resize is assigned zero overlap without modeling resize readiness or transient CPU mismatch. The collected performance file has only nine rows and is not suitable for calibration. Use the detailed report and paired block-bootstrap intervals in the results folder; do not cite earlier percentages in this document as current results.

---

## 1. Validation-Tuned Adaptive Burst Detection (`tuned_adaptive`)

### The Problem in the Paper
The baseline paper utilizes an online Z-score detector (Algorithm 1) to identify workload spikes. However, it hardcodes the parameters: `threshold = 5.0` and `influence = 0.5`. There is no validation or mathematical proof that these parameters generalize to workloads with different characteristics (e.g., FIFA World Cup vs. Wikipedia traces).

### Our Novelty Extension
We implement a **validation-tuned, volatility-adaptive burst detector**:
1. **Residual Signal:** At each one-minute observation, the detector receives the signed one-step forecast residual (observed demand minus predicted demand). Positive errors indicate underprediction, the direction that can cause an SLO miss.
2. **Validation Optimization:** Before evaluation, the script reserves a validation slice of the training data. It tunes detector parameters using a validation SLO target strictly below the Paper Hybrid validation rate and scores resource cost including rolling overlap.
3. **Volatility Adaptivity:** The residual window's standard deviation is scaled by its mean absolute magnitude, which remains stable when signed residuals average near zero. Candidate thresholds and persistence gates are selected on validation only.

* **Code Reference:** [`Novelty/src/burst.py`](file:///c:/Users/HP/OneDrive/Desktop/Project/Cloud%20computing%20project/Novelty/src/burst.py) (see `AdaptiveBurstDetector` and `tune_burst_parameters`)

### Ablation Findings
In the corrected full-trace run, N1 had 0.016% SLO violations (equal to Paper Hybrid) and cost 902.503 vs 903.638. The paired cost interval was [$-$1.639,$-$0.676] within this single synthetic run; it still does not establish overall dominance because SLO did not improve and the fixture is not measured cluster data.

---

## 2. Uncertainty-Aware Quantile Provisioning (`quantile_bilstm`)

### The Problem in the Paper
The baseline paper uses a standard deterministic Bi-LSTM, which outputs a single point prediction ($y_{t+1}$). During bursts, the scaler has no concept of prediction uncertainty and blindly triggers a hardcoded backup rule: scaling up to the maximum allowable CPU limit (e.g., 950 millicores) for safety. This causes heavy over-provisioning when point forecasts are uncertain.

### Our Novelty Extension
We replace the point forecaster with a **probabilistic Quantile Bi-LSTM**:
1. **Residual Quantile Forecasting:** The network predicts P10/P50/P90 next-step residuals around the last observed value using Pinball Loss and a persistence skip connection.
2. **Gated Uncertainty Scaling:** 
   * **Normal state:** The scaler uses a validation-selected interpolation gain across the P10/P50/P90 interval. The corrected run selected -0.75 (toward P10); this setting reduced the SLO point estimate slightly but did not lower resource cost.
   * **Burst state:** The scaler uses a width-aware high-CPU path, adjusting workload and CPU allocation according to the calibrated interval width.

* **Code Reference:** [`Novelty/src/forecast.py`](file:///c:/Users/HP/OneDrive/Desktop/Project/Cloud%20computing%20project/Novelty/src/forecast.py) (see `QuantileBiLSTMForecaster`) and [`Novelty/src/scaler.py`](file:///c:/Users/HP/OneDrive/Desktop/Project/Cloud%20computing%20project/Novelty/src/scaler.py) (see `UncertaintyAwareProvisioner`)

### Ablation Findings
In the corrected run, validation selected safety gain **-0.75**. Held-out SLO violations were 0.008% vs 0.016% for Paper Hybrid, while cost was 903.641 vs 903.638. Quantile forecast MAE was 1.894 vs 2.419 for Bi-LSTM, and empirical P10--P90 coverage was 90.52%. It improved forecast error and the SLO point estimate, but not cost; paired intervals include zero.

---

## 3. Adaptive Monitoring Cadence (`adaptive`)

### The Problem in the Paper
The baseline paper uses a fixed monitoring interval of 60 seconds. Polling Prometheus every 60 seconds is wasteful during flat, stable traffic periods, but may be too slow to catch transient spikes during bursty periods.

### Our Novelty Extension
We implement a **workload-adaptive monitoring interval**:
1. **Dynamic Scaling:** The interval scales between 15 seconds and a validation-selected maximum (120 seconds in the corrected run).
2. **Volative and Width Gates:** The interval size is updated using both the rolling load coefficient of variation ($CV$) and the forecaster's predicted relative uncertainty interval width.

* **Code Reference:** [`Novelty/src/monitoring.py`](file:///c:/Users/HP/OneDrive/Desktop/Project/Cloud%20computing%20project/Novelty/src/monitoring.py) (see `AdaptiveMonitoringInterval`)

### Ablation Findings
On the corrected holdout, monitor cycles fell from 25,062 to 24,106 and realized average interval was 62.4 seconds. Cost changed from 903.638 to 903.580, while SLO stayed at 0.016%. The paired cost interval includes zero. At 60-second trace resolution, shorter scheduled intervals cannot produce more frequent observations.

## 4. Pod In-Place Resource Resizing (`inplace`)

### The Problem in the Paper
In 2022, dynamically changing container CPU allocations (vertical scaling) required a rolling update. The Kubernetes API destroyed the old Pod and deployed a new one. This created a resource overlap period (Eq. 4 in the paper) where both pods ran concurrently, incurring high temporary cost overhead and potential replication latency.

### Our Novelty Extension
With the release of Kubernetes v1.35+, we implement an **In-Place Pod Resize** backend:
1. **Symbolic Patching:** The vertical resize is done by issuing a `PATCH` request to the Pod's `spec.containers.resources` subresource.
2. **In-Place API Path:** The executor requests a Pod resize through Kubernetes' resize subresource. The emulator currently charges no rolling overlap on this path; this is an accounting assumption, not proof of zero live resize latency or resource overlap.

* **Code Reference:** [`Novelty/src/kubernetes_executor.py`](file:///c:/Users/HP/OneDrive/Desktop/Project/Cloud%20computing%20project/Novelty/src/kubernetes_executor.py) (see `InPlaceResizeExecutor`)

### Ablation Findings
In the corrected run, SLO violations were 0.012% vs 0.016% for Paper Hybrid. In-place cost was 903.063 vs 903.638 and modeled rolling overlap was zero. The paired cost interval was [$-$0.836,$-$0.328], while the SLO interval reached zero. This small cost difference is an accounting assumption: resize delay, readiness, and transient CPU mismatch are not represented. It is not evidence that live in-place resizing improves both SLO and cost.

---

## Controlled run and interpretation

The earlier 1,440-tick results in saved reports were produced by a shorter trace window and are superseded by the corrected full-trace run above. Results across those different windows or training settings must not be combined. No current run supports the claim that all four standalone novelties beat Paper Hybrid on both SLO and cost.

---
