import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import statistics
import time
import requests
import threading
import math
from collections import deque
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import uvicorn
# kubernetes client imported dynamically inside helper to make package dependency optional

# Ensure the root and Paper folders are on the python path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Paper'))

# Imports from Paper and Novelty modules
from Paper.src.burst import OnlineBurstDetector
from Paper.src.scaler import HybridProvisioner, ScalingDecision
from Paper.src.performance import PerformanceModel
from Paper.src.monitor import query, prometheus_qps
from Novelty.src.forecast import WorkloadForecaster, QuantileBiLSTMForecaster
from Novelty.src.burst import AdaptiveBurstDetector, BurstParameters
from Novelty.src.scaler import UncertaintyAwareProvisioner
from Novelty.src.monitoring import AdaptiveMonitoringInterval
from Paper.src.data import load_workload_csv

app = FastAPI(title="Autoscaling Real-time Comparison Dashboard")

# ---------------------------------------------------------------------------
# Request Tracker for Live Locust Traffic
# ---------------------------------------------------------------------------
class RequestTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0
        self.last_reset = time.time()
        
    def increment(self):
        with self.lock:
            self.count += 1
            
    def get_qps_and_reset(self):
        with self.lock:
            now = time.time()
            elapsed = max(now - self.last_reset, 0.01)
            qps = self.count / elapsed
            self.count = 0
            self.last_reset = now
            return qps

tracker = RequestTracker()

# ---------------------------------------------------------------------------
# Global State & Model Cache
# ---------------------------------------------------------------------------
class SimulationState:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.current_tick = 0
        self.history = deque(maxlen=120)  # for forecast lookback
        self.detector_fixed = OnlineBurstDetector(window=10, threshold=5.0, influence=0.5)
        self.detector_adaptive = AdaptiveBurstDetector(window=10, parameters=BurstParameters())
        self.replicas = 1
        self.cpu = 600
        self.pending_actions = []  # items like (tick_due, decision, 'rolling')
        
        # Cumulative stats
        self.total_cost = 0.0
        self.overlap_cost = 0.0
        self.api_cycles = 0
        self.slo_violations = 0
        self.ticks_count = 0
        self.next_monitor_tick = 0
        self.last_monitor_interval = 60
        self.last_forecast_mid = 0.0
        self.last_forecast_interval = (0.0, 0.0, 0.0)
        self.last_predicted_replicas = 1
        self.active_burst = False
        
        # Shift-alignment variables for comparing prediction with reality at same tick
        self.next_forecast_qps = None
        self.next_forecast_p10 = None
        self.next_forecast_p90 = None
        self.next_predicted_replicas = 1
        
        # Historical records for charts
        self.chart_ticks = []
        self.chart_observed = []
        self.chart_predicted = []
        self.chart_p10 = []
        self.chart_p90 = []
        
        # Pod comparison fields (as requested by user)
        self.chart_replicas = []            # Actual active replicas (deployed)
        self.chart_predicted_replicas = []  # Replicas predicted/selected by Autoscaler
        self.chart_optimal_replicas = []    # Replicas optimally needed (from Perf model on actual QPS)
        
        self.chart_cpu = []
        self.chart_utilization = []
        self.chart_latency = []
        self.chart_burst = []
        self.chart_interval = []
        self.chart_cost = []

# Load models and handle fallbacks gracefully
try:
    print("Loading Paper Workload Forecaster...", flush=True)
    bilstm_model = WorkloadForecaster.load(str(ROOT / 'Paper' / 'models' / 'bilstm.pt'))
except Exception as e:
    print(f"Warning: Failed to load Paper BiLSTM model: {e}. Using dummy.", flush=True)
    bilstm_model = WorkloadForecaster()

try:
    print("Loading Novelty Quantile Forecaster...", flush=True)
    quantile_bilstm_model = QuantileBiLSTMForecaster.load(str(ROOT / 'Novelty' / 'models' / 'quantile_bilstm.pt'))
except Exception as e:
    print(f"Warning: Failed to load Novelty Quantile BiLSTM model: {e}. Using dummy.", flush=True)
    quantile_bilstm_model = QuantileBiLSTMForecaster()

try:
    print("Loading Performance Model...", flush=True)
    perf_model = PerformanceModel.load(str(ROOT / 'Paper' / 'models' / 'performance.joblib'))
except Exception as e:
    print(f"Warning: Failed to load Performance model: {e}. Fitting fallback.", flush=True)
    perf_model = PerformanceModel()
    rows = []
    for cpu in (600, 700, 800, 900, 950):
        for rate in range(50, 2001, 50):
            replicas = max(1, int(np.ceil(rate / (cpu * .42))))
            rows.append((rate, cpu, 350, replicas, min(.99, rate / (replicas * cpu * .42))))
    frame = pd.DataFrame(rows, columns=['request_rate', 'cpu_millicores', 'slo_ms', 'replicas', 'utilization'])
    perf_model.fit(frame)

# Instantiate provisioners
provisioner_fixed = HybridProvisioner(perf_model)
provisioner_uncertainty = UncertaintyAwareProvisioner(perf_model)
interval_policy = AdaptiveMonitoringInterval(base_seconds=60)

# Load World Cup trace
print("Loading World Cup workload trace...", flush=True)
trace_path = ROOT / 'Novelty' / 'data' / 'processed' / 'worldcup98_minute.csv'
if not trace_path.exists():
    trace_path = ROOT / 'Paper' / 'data' / 'processed' / 'worldcup98_minute.csv'

if trace_path.exists():
    try:
        raw_wc = np.asarray(load_workload_csv(str(trace_path)), dtype=float)
        cut = int(.8 * len(raw_wc))
        scale = 1800.0 / max(raw_wc[:cut].max(), 1e-9)
        wc_raw = raw_wc[cut:].tolist()
        wc_scaled = (raw_wc[cut:] * scale).tolist()
        print(f"Loaded trace: {len(wc_scaled)} test points.", flush=True)
        
        # Calibrate components on training/validation slice using trace_path with fallback
        calibrate_raw = np.asarray(load_workload_csv(str(trace_path)), dtype=float)
        calibrate_cut = int(.8 * len(calibrate_raw))
        calibrate_scale = 1800.0 / max(calibrate_raw[:calibrate_cut].max(), 1e-9)
        calibrate_demand = calibrate_raw * calibrate_scale
        calibrate_train = calibrate_demand[:calibrate_cut]
        validation_size = max(20, len(calibrate_train) // 5)
        forecast_train = calibrate_train[:-validation_size]
        validation = calibrate_train[-validation_size:]
        
        print("Calibrating Uncertainty Provisioner and Adaptive Monitoring...", flush=True)
        running = list(forecast_train)
        val_intervals = []
        for val_val in validation:
            # Convert QPS to Requests/Minute for pre-trained model scaling compatibility
            running_rpm = [v * 60.0 for v in running]
            low_rpm, mid_rpm, high_rpm = quantile_bilstm_model.predict_interval(running_rpm)
            val_intervals.append((low_rpm / 60.0, mid_rpm / 60.0, high_rpm / 60.0))
            running.append(float(val_val))
            
        widths = [provisioner_uncertainty._relative_width(interval) for interval in val_intervals]
        provisioner_uncertainty.set_interval_calibration(widths)
        interval_policy.calibrate(validation, forecast_widths=widths, high_cv_percentile=0.60)
        print(f"Calibration finished. Max relative width: {provisioner_uncertainty.max_relative_width:.3f}", flush=True)
    except Exception as e:
        print(f"Warning: Calibration or trace loading failed: {e}. Using synthetic.", flush=True)
        wc_raw = []
        wc_scaled = []
else:
    print("Trace CSV not found. Simulation will fall back to synthetic waveforms.", flush=True)
    wc_raw = []
    wc_scaled = []

# Generate synthetic trace as fallback or option
def generate_synthetic():
    n_points = 500
    t = np.arange(n_points)
    # Sinusoidal base load
    base_load = 400 + 300 * np.sin(t * (2 * np.pi / 150))
    # Noise
    noise = np.random.normal(0, 30, n_points)
    raw = (base_load + noise).clip(min=10)
    # Add bursts
    for idx in range(80, n_points, 180):
        for duration in range(12):
            if idx + duration < n_points:
                raw[idx + duration] += 900 + 200 * np.sin(duration * (np.pi / 12))
    scale = 1800.0 / max(raw.max(), 1e-9)
    scaled = (raw * scale).tolist()
    return raw.tolist(), scaled

syn_raw, syn_scaled = generate_synthetic()

# ---------------------------------------------------------------------------
# Direct Kubernetes API Fallback
# ---------------------------------------------------------------------------
def get_live_k8s_status(ns, deploy):
    try:
        from kubernetes import client, config as k8s_config
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()
        
        apps_v1 = client.AppsV1Api()
        deployment = apps_v1.read_namespaced_deployment(deploy, ns)
        
        # Get active replicas
        actual_replicas = deployment.status.ready_replicas or deployment.status.replicas or 1
        
        # Get request CPU millicores
        container = deployment.spec.template.spec.containers[0]
        cpu_req = container.resources.requests.get('cpu') if container.resources and container.resources.requests else "600m"
        
        if isinstance(cpu_req, str) and cpu_req.endswith('m'):
            cpu_millicores = int(cpu_req[:-1])
        else:
            cpu_millicores = int(float(cpu_req) * 1000) if cpu_req else 600
            
        return actual_replicas, cpu_millicores
    except ImportError:
        # Silently return None so the dashboard can fall back to Prometheus
        return None
    except Exception as e:
        print(f"Live Mode fallback: Direct Kubernetes API query failed: {e}")
        return None

# ---------------------------------------------------------------------------
# Simulation Server Logic
# ---------------------------------------------------------------------------
state = SimulationState()

# Simulation & Live Configurations
config = {
    "mode": "simulation",        # "simulation" or "live_kubernetes"
    "source": "worldcup",        # "worldcup" or "synthetic" or "locust" or "live_locust"
    "novelty1_burst": True,       # Adaptive Burst Detection
    "novelty2_quantile": True,    # Quantile Forecasting & Uncertainty Scaler
    "novelty3_inplace": True,     # Pod In-Place Resource Resizing
    "novelty4_monitoring": True,  # Adaptive Monitoring Cadence
    "locust_manual_qps": 500.0,   # User-specified workload in Locust mode
    "simulation_speed_ms": 500,   # Tick duration in milliseconds
    "is_playing": False,
    "prometheus_url": "http://localhost:9090",
    "controller_url": "http://localhost:8080",
    "k8s_namespace": "hybrid-autoscaling",
    "k8s_deployment": "house-price"
}

# Emulated application performance response time
def simulate_latency(demand, replicas, cpu):
    utilization = demand / max(replicas * cpu * 0.42, 1e-9)
    # Emulates the application's latency profile: 80ms baseline + cubic growth past utilization limits
    response_ms = 80.0 + 180.0 * min(utilization, 1.5) ** 3
    return min(utilization, 1.5), response_ms

# Cost equations from Vu, Tran, and Kim (2022)
def cost_eq_2(replicas, cpu, price, seconds):
    return replicas * cpu * price * seconds

def cost_eq_4(old_replicas, old_cpu, replicas, cpu, price, overlap_seconds):
    return (old_replicas * old_cpu + replicas * cpu) * price * overlap_seconds

def cost_eq_3(old_replicas, old_cpu, replicas, cpu, price, interval_seconds, overlap_seconds):
    return cost_eq_4(old_replicas, old_cpu, replicas, cpu, price, overlap_seconds) + cost_eq_2(replicas, cpu, price, interval_seconds - overlap_seconds)

def get_workload_point(tick):
    if config["source"] == "live_locust":
        qps = tracker.get_qps_and_reset()
        return qps, qps
    elif config["source"] == "locust":
        raw = config["locust_manual_qps"]
        return raw, raw
    elif config["source"] == "worldcup":
        if len(wc_scaled) > 0:
            idx = tick % len(wc_scaled)
            return wc_raw[idx], wc_scaled[idx]
        else:
            idx = tick % len(syn_scaled)
            return syn_raw[idx], syn_scaled[idx]
    else:
        idx = tick % len(syn_scaled)
        return syn_raw[idx], syn_scaled[idx]

def run_simulation_step():
    # 1. Fetch workload point
    raw_qps, qps = get_workload_point(state.current_tick)
    state.history.append(float(qps))
    
    tick = state.current_tick
    
    # Retrieve forecast made at T-1 for the current tick T (shift alignment)
    if tick < 10:
        predicted_now = None
        low_now = None
        high_now = None
        predicted_replicas_now = None
    else:
        predicted_now = state.next_forecast_qps
        low_now = state.next_forecast_p10
        high_now = state.next_forecast_p90
        predicted_replicas_now = state.next_predicted_replicas

    interval_seconds = 60
    overlap_seconds = 12
    price = 1e-6 # Resource cost per CPU millicore per second
    
    # 2. Check pending scaling actions (Rolling update latency)
    transition = None
    due = [item for item in state.pending_actions if item[0] <= tick]
    state.pending_actions = [item for item in state.pending_actions if item[0] > tick]
    
    if due:
        _, decision, kind = due[-1]
        old_replicas, old_cpu = state.replicas, state.cpu
        state.replicas, state.cpu = decision.replicas, decision.cpu_millicores
        transition = (old_replicas, old_cpu, kind)
        
    # 3. Determine if monitoring is due
    monitored = (tick >= state.next_monitor_tick)
    burst = state.active_burst
    
    low = mid = high = float(qps)
    chosen_interval = state.last_monitor_interval
    predicted_replicas = state.last_predicted_replicas
    
    if monitored:
        state.api_cycles += 1
        
        # 4. Workload Forecasting (only active from tick 9 / history length 10 onwards)
        if len(state.history) >= 10:
            if config["novelty2_quantile"]:
                history_rpm = [v * 60.0 for v in state.history]
                low_rpm, mid_rpm, high_rpm = quantile_bilstm_model.predict_interval(history_rpm)
                low = low_rpm / 60.0
                mid = mid_rpm / 60.0
                high = high_rpm / 60.0
            else:
                history_rpm = [v * 60.0 for v in state.history]
                mid_rpm = bilstm_model.predict(history_rpm)
                mid = mid_rpm / 60.0
                low = high = mid
                
            # 5. Burst Detection
            if config["novelty1_burst"]:
                burst = state.detector_adaptive.update(mid)
            else:
                burst = state.detector_fixed.update(mid)
                
            state.active_burst = burst
            
            # 6. Scaling Decision
            if config["novelty2_quantile"]:
                decision = provisioner_uncertainty.decide(mid, burst=burst, interval=(low, mid, high))
            else:
                decision = provisioner_fixed.decide(mid, burst=burst)
                
            predicted_replicas = decision.replicas
            
            # 7. Apply scaling decision (Novelty 3: Pod In-Place Resize vs Rolling Update)
            if config["novelty3_inplace"]:
                old_replicas, old_cpu = state.replicas, state.cpu
                state.replicas, state.cpu = decision.replicas, decision.cpu_millicores
                if (old_replicas, old_cpu) != (state.replicas, state.cpu):
                    transition = (old_replicas, old_cpu, 'inplace')
            else:
                has_change = (decision.cpu_millicores != state.cpu or decision.replicas != state.replicas)
                if has_change:
                    state.pending_actions.append((tick + 1, decision, 'rolling'))
                    
            # 8. Adaptive monitoring cadence (Novelty 4)
            if config["novelty4_monitoring"]:
                forecast_width = provisioner_uncertainty._relative_width((low, mid, high)) if config["novelty2_quantile"] else 0.0
                chosen_interval = interval_policy.update(qps, burst, forecast_width)
            else:
                chosen_interval = 60
        else:
            # First 9 ticks: No prediction or autoscaling decisions
            mid = None
            low = high = None
            predicted_replicas = 1
            burst = False
            chosen_interval = 60
            
        state.last_forecast_mid = mid
        state.last_forecast_interval = (low, mid, high)
        state.last_predicted_replicas = predicted_replicas
        state.last_monitor_interval = chosen_interval
        state.next_monitor_tick = tick + max(1, int(np.ceil(chosen_interval / 60.0)))
    else:
        mid = state.last_forecast_mid
        low, mid, high = state.last_forecast_interval
        burst = state.active_burst
        predicted_replicas = state.last_predicted_replicas
        
    # 9. Performance Metric Calculations & Cost Accounting
    utilization, response_ms = simulate_latency(qps, state.replicas, state.cpu)
    
    # Calculate optimal replicas actually needed based on DTR for the actual workload load
    try:
        optimal_replicas, _ = perf_model.predict(qps, state.cpu, 350)
    except Exception:
        optimal_replicas = 1
        
    # Calculate costs
    current_cost = 0.0
    current_overlap = 0.0
    
    if transition and transition[2] == 'rolling' and transition[1] != state.cpu:
        current_cost = cost_eq_3(transition[0], transition[1], state.replicas, state.cpu, price, 60.0, overlap_seconds)
        current_overlap = cost_eq_4(transition[0], transition[1], state.replicas, state.cpu, price, overlap_seconds)
    else:
        current_cost = cost_eq_2(state.replicas, state.cpu, price, 60.0)
        current_overlap = 0.0
        
    state.total_cost += current_cost
    state.overlap_cost += current_overlap
    state.ticks_count += 1
    
    is_violation = response_ms > 350.0
    if is_violation:
        state.slo_violations += 1
        
    # Save predictions for the next tick T+1
    state.next_forecast_qps = mid
    state.next_forecast_p10 = low
    state.next_forecast_p90 = high
    state.next_predicted_replicas = predicted_replicas

    # 10. Record records for UI charts (using shift-aligned values, guard None)
    state.chart_ticks.append(tick)
    state.chart_observed.append(round(raw_qps, 1))
    state.chart_predicted.append(round(predicted_now, 1) if predicted_now is not None else None)
    state.chart_p10.append(round(low_now, 1) if low_now is not None else None)
    state.chart_p90.append(round(high_now, 1) if high_now is not None else None)

    state.chart_replicas.append(state.replicas)
    state.chart_predicted_replicas.append(predicted_replicas_now)
    state.chart_optimal_replicas.append(optimal_replicas)

    state.chart_cpu.append(state.cpu)
    state.chart_utilization.append(round(utilization * 100, 1))
    state.chart_latency.append(round(response_ms, 1))
    state.chart_burst.append(1 if burst else 0)
    state.chart_interval.append(chosen_interval)
    state.chart_cost.append(round(state.total_cost, 4))

    state.current_tick += 1

    avg_interval = 60.0
    if len(state.chart_interval) > 0:
        avg_interval = sum(state.chart_interval) / len(state.chart_interval)

    return {
        "tick": tick,
        "observed_qps": round(raw_qps, 1),
        "predicted_qps": round(predicted_now, 1) if predicted_now is not None else None,
        "p10": round(low_now, 1) if low_now is not None else None,
        "p90": round(high_now, 1) if high_now is not None else None,
        "burst": burst,
        "replicas": state.replicas,
        "predicted_replicas": predicted_replicas_now,
        "optimal_replicas": optimal_replicas,
        "cpu_millicores": state.cpu,
        "utilization": round(utilization * 100, 1),
        "latency_ms": round(response_ms, 1),
        "slo_violation": is_violation,
        "monitoring_interval": chosen_interval,
        "monitored": monitored,
        "total_cost": round(state.total_cost, 4),
        "overlap_cost": round(state.overlap_cost, 4),
        "api_cycles": state.api_cycles,
        "slo_violation_rate": round((state.slo_violations / max(1, state.ticks_count)) * 100, 2),
        "average_interval": round(avg_interval, 1)
    }

def run_live_kubernetes_step():
    prom_url = config["prometheus_url"]
    ctrl_url = config["controller_url"]
    ns = config["k8s_namespace"]
    svc = "house-price" # default service label name
    deploy = config["k8s_deployment"]
    
    # Try querying direct Kubernetes API first (much faster and highly reliable fallback)
    k8s_status = get_live_k8s_status(ns, deploy)
    
    # 1. Query Prometheus for actual QPS and Latency
    try:
        obs_qps = prometheus_qps(prom_url, ns, svc)
    except Exception as e:
        print(f"Live Mode: Prometheus QPS query failed: {e}")
        obs_qps = 0.0
        
    # Retrieve forecast made at T-1 for the current tick T (shift alignment)
    if state.current_tick < 10:
        predicted_now = None
        low_now = None
        high_now = None
        predicted_replicas_now = None
    else:
        predicted_now = state.next_forecast_qps
        low_now = state.next_forecast_p10
        high_now = state.next_forecast_p90
        predicted_replicas_now = state.next_predicted_replicas if state.next_predicted_replicas is not None else 1
        
    try:
        # P95 response latency
        lat_expr = f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{namespace="{ns}",service="{svc}"}}[1m])) by (le))'
        lat_val = query(prom_url, lat_expr)
        latency_ms = float(lat_val) * 1000.0 if lat_val > 0 else 80.0
    except Exception as e:
        print(f"Live Mode: Prometheus Latency query failed: {e}")
        latency_ms = 80.0
        
    # Get replicas and cpu allocation
    if k8s_status is not None:
        actual_replicas, cpu_millicores_source = k8s_status
    else:
        # Fall back to Prometheus queries for replicas
        try:
            rep_expr = f'kube_deployment_status_replicas_available{{namespace="{ns}",deployment="{deploy}"}}'
            rep_val = query(prom_url, rep_expr)
            actual_replicas = int(round(rep_val)) if rep_val > 0 else 1
        except Exception as e:
            print(f"Live Mode: Prometheus Replicas query failed: {e}")
            actual_replicas = 1
        cpu_millicores_source = 600
        
    # 2. Query the live Autoscaling Controller endpoint for active replicas and CPU
    try:
        resp = requests.get(f"{ctrl_url}/status", timeout=2)
        if resp.status_code == 200:
            ctrl_data = resp.json()
            predicted_replicas = ctrl_data.get("replicas", actual_replicas)
            cpu_millicores = ctrl_data.get("cpu_millicores", cpu_millicores_source)
            raw_util = ctrl_data.get("utilization", 0.0)
            utilization = (raw_util * 100.0) if raw_util is not None else 0.0
        else:
            predicted_replicas = actual_replicas
            cpu_millicores = cpu_millicores_source
            utilization = 0.0
    except Exception as e:
        print(f"Live Mode: Autoscaler controller /status unreachable: {e}")
        predicted_replicas = actual_replicas
        cpu_millicores = cpu_millicores_source
        utilization = 0.0

    # 3. Workload Forecasting (run locally in dashboard on observed history with corrected units scaling)
    state.history.append(float(obs_qps))
    
    if len(state.history) >= 10:
        if config["novelty2_quantile"]:
            history_rpm = [v * 60.0 for v in state.history]
            low_rpm, mid_rpm, high_rpm = quantile_bilstm_model.predict_interval(history_rpm)
            p10 = low_rpm / 60.0
            predicted_qps = mid_rpm / 60.0
            p90 = high_rpm / 60.0
        else:
            history_rpm = [v * 60.0 for v in state.history]
            mid_rpm = bilstm_model.predict(history_rpm)
            predicted_qps = mid_rpm / 60.0
            p10 = p90 = predicted_qps
            
        # Burst Detection
        if config["novelty1_burst"]:
            burst = state.detector_adaptive.update(predicted_qps)
        else:
            burst = state.detector_fixed.update(predicted_qps)
    else:
        predicted_qps = None
        p10 = p90 = None
        burst = False
        
    # Compute optimal replicas needed based on performance model
    try:
        optimal_replicas, _ = perf_model.predict(obs_qps, cpu_millicores, 350)
    except Exception:
        optimal_replicas = 1

    # Cost accounting
    price = 1e-6
    state.total_cost += cost_eq_2(actual_replicas, cpu_millicores, price, 60.0)

    tick = state.current_tick

    # Save predictions for the next tick T+1
    state.next_forecast_qps = predicted_qps
    state.next_forecast_p10 = p10
    state.next_forecast_p90 = p90
    state.next_predicted_replicas = predicted_replicas

    # Record records for UI charts (using shift-aligned values, guard None with 0)
    state.chart_ticks.append(tick)
    state.chart_observed.append(round(obs_qps, 1))
    state.chart_predicted.append(round(predicted_now, 1) if predicted_now is not None else None)
    state.chart_p10.append(round(low_now, 1) if low_now is not None else None)
    state.chart_p90.append(round(high_now, 1) if high_now is not None else None)

    state.chart_replicas.append(actual_replicas)
    state.chart_predicted_replicas.append(predicted_replicas_now)
    state.chart_optimal_replicas.append(optimal_replicas)

    state.chart_cpu.append(cpu_millicores)
    state.chart_utilization.append(round(utilization, 1))
    state.chart_latency.append(round(latency_ms, 1))
    state.chart_burst.append(1 if burst else 0)
    state.chart_interval.append(60)
    state.chart_cost.append(round(state.total_cost, 4))

    state.ticks_count += 1
    if latency_ms > 350.0:
        state.slo_violations += 1

    state.current_tick += 1

    return {
        "tick": tick,
        "observed_qps": round(obs_qps, 1),
        "predicted_qps": round(predicted_now, 1) if predicted_now is not None else None,
        "p10": round(low_now, 1) if low_now is not None else None,
        "p90": round(high_now, 1) if high_now is not None else None,
        "burst": burst,
        "replicas": actual_replicas,
        "predicted_replicas": predicted_replicas_now,
        "optimal_replicas": optimal_replicas,
        "cpu_millicores": cpu_millicores,
        "utilization": round(utilization, 1),
        "latency_ms": round(latency_ms, 1),
        "slo_violation": latency_ms > 350.0,
        "monitoring_interval": 60,
        "monitored": True,
        "total_cost": round(state.total_cost, 4),
        "overlap_cost": 0.0,
        "api_cycles": state.ticks_count,
        "slo_violation_rate": round((state.slo_violations / max(1, state.ticks_count)) * 100, 2),
        "average_interval": 60.0
    }

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Autoscaling Real-time Comparison Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.1.0/dist/chartjs-plugin-annotation.min.js"></script>
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            darkBg: '#0f172a',
                            panelBg: '#1e293b',
                            accentBlue: '#3b82f6',
                            accentGreen: '#10b981',
                            accentRed: '#ef4444',
                            accentOrange: '#f97316'
                        }
                    }
                }
            }
        </script>
        <style>
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #0f172a; }
            ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #475569; }
        </style>
    </head>
    <body class="bg-darkBg text-slate-100 font-sans min-h-screen flex flex-col">
        <!-- Top Navigation -->
        <header class="bg-panelBg border-b border-slate-800 px-6 py-4 flex items-center justify-between shadow-lg">
            <div class="flex items-center space-x-3">
                <div class="bg-accentBlue p-2 rounded-lg text-white font-bold text-xl shadow">
                    AGY
                </div>
                <div>
                    <h1 class="text-lg font-bold tracking-tight text-white">Autoscaler Real-Time Comparison</h1>
                    <p class="text-xs text-slate-400 font-medium">Pods Predicted vs. Pods Needed Comparison (Prometheus + Kubernetes)</p>
                </div>
            </div>
            
            <!-- Mode Switcher -->
            <div class="flex items-center bg-slate-900 border border-slate-700 p-0.5 rounded-lg">
                <button id="mode-sim" onclick="setMode('simulation')" class="px-4 py-1.5 rounded-md text-xs font-semibold bg-accentBlue text-white shadow transition">
                    Simulation Mode
                </button>
                <button id="mode-live" onclick="setMode('live_kubernetes')" class="px-4 py-1.5 rounded-md text-xs font-semibold text-slate-400 hover:text-slate-200 transition">
                    Live Cluster Mode
                </button>
            </div>
        </header>

        <!-- Main Dashboard Layout -->
        <main class="flex-1 flex flex-col md:flex-row p-6 gap-6 overflow-hidden">
            <!-- Sidebar: Config & Controls -->
            <section class="w-full md:w-80 bg-panelBg rounded-xl p-5 border border-slate-800 flex flex-col space-y-5 shadow-md flex-shrink-0">
                <!-- Preset Options (Simulation Mode only) -->
                <div id="sim-presets-group">
                    <h2 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2.5 border-b border-slate-800 pb-1">Autoscaler Preset</h2>
                    <select id="preset-selector" onchange="applyPreset()" class="w-full bg-slate-900 border border-slate-700 text-slate-200 rounded px-2.5 py-1.5 text-xs focus:outline-none focus:border-accentBlue mb-4">
                        <option value="novelty" selected>Novelty Full (Quantile + Adaptive)</option>
                        <option value="paper">Paper Baseline (BiLSTM + Rolling)</option>
                        <option value="custom">Custom Configuration</option>
                    </select>

                    <h2 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 border-b border-slate-800 pb-1">Enabled Novelties</h2>
                    <div class="space-y-3">
                        <label class="flex items-start space-x-2.5 cursor-pointer">
                            <input type="checkbox" id="toggle-n1" checked onchange="configChanged('custom')" class="mt-0.5 h-4 w-4 text-accentBlue border-slate-700 rounded bg-slate-900">
                            <div>
                                <span class="text-xs font-medium text-slate-200">N1: Adaptive Burst</span>
                                <p class="text-[10px] text-slate-400">Volatility-tuned adaptive threshold</p>
                            </div>
                        </label>
                        <label class="flex items-start space-x-2.5 cursor-pointer">
                            <input type="checkbox" id="toggle-n2" checked onchange="configChanged('custom')" class="mt-0.5 h-4 w-4 text-accentBlue border-slate-700 rounded bg-slate-900">
                            <div>
                                <span class="text-xs font-medium text-slate-200">N2: Quantile Scaling</span>
                                <p class="text-[10px] text-slate-400">P10/P50/P90 interval safety buffer</p>
                            </div>
                        </label>
                        <label class="flex items-start space-x-2.5 cursor-pointer">
                            <input type="checkbox" id="toggle-n3" checked onchange="configChanged('custom')" class="mt-0.5 h-4 w-4 text-accentBlue border-slate-700 rounded bg-slate-900">
                            <div>
                                <span class="text-xs font-medium text-slate-200">N3: In-Place Resize</span>
                                <p class="text-[10px] text-slate-400">Zero deployment overlap CPU patch</p>
                            </div>
                        </label>
                        <label class="flex items-start space-x-2.5 cursor-pointer">
                            <input type="checkbox" id="toggle-n4" checked onchange="configChanged('custom')" class="mt-0.5 h-4 w-4 text-accentBlue border-slate-700 rounded bg-slate-900">
                            <div>
                                <span class="text-xs font-medium text-slate-200">N4: Adaptive monitoring</span>
                                <p class="text-[10px] text-slate-400">Interval scales based on CV/width</p>
                            </div>
                        </label>
                    </div>
                </div>

                <!-- Live Cluster Configurations (Live Mode only) -->
                <div id="live-config-group" class="hidden space-y-4">
                    <h2 class="text-xs font-bold uppercase tracking-wider text-slate-400 border-b border-slate-800 pb-1">Cluster Settings</h2>
                    <div>
                        <label class="text-[10px] text-slate-400 block mb-1">Prometheus API URL</label>
                        <input type="text" id="live-prom-url" value="http://localhost:9090" onchange="updateLiveConfig()" class="w-full bg-slate-900 border border-slate-700 text-slate-200 rounded px-2.5 py-1 text-xs focus:outline-none focus:border-accentBlue">
                    </div>
                    <div>
                        <label class="text-[10px] text-slate-400 block mb-1">Controller Status URL</label>
                        <input type="text" id="live-ctrl-url" value="http://localhost:8080" onchange="updateLiveConfig()" class="w-full bg-slate-900 border border-slate-700 text-slate-200 rounded px-2.5 py-1 text-xs focus:outline-none focus:border-accentBlue">
                        <p class="text-[9px] text-slate-400 mt-0.5">Points to the active autoscaler API</p>
                    </div>
                    <div class="grid grid-cols-2 gap-2">
                        <div>
                            <label class="text-[10px] text-slate-400 block mb-1">Namespace</label>
                            <input type="text" id="live-ns" value="hybrid-autoscaling" onchange="updateLiveConfig()" class="w-full bg-slate-900 border border-slate-700 text-slate-200 rounded px-2 py-1 text-xs focus:outline-none">
                        </div>
                        <div>
                            <label class="text-[10px] text-slate-400 block mb-1">Deployment</label>
                            <input type="text" id="live-deploy" value="house-price" onchange="updateLiveConfig()" class="w-full bg-slate-900 border border-slate-700 text-slate-200 rounded px-2 py-1 text-xs focus:outline-none">
                        </div>
                    </div>
                </div>

                <!-- Workload settings -->
                <div id="sim-source-group">
                    <h2 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 border-b border-slate-800 pb-1">Workload Pattern</h2>
                    <div class="space-y-3">
                        <select id="source-selector" onchange="updateSource()" class="w-full bg-slate-900 border border-slate-700 text-slate-200 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-accentBlue">
                            <option value="worldcup">FIFA World Cup 98 Test Set</option>
                            <option value="synthetic">Synthetic Sinusoidal + Spikes</option>
                            <option value="locust">Interactive Locust Slider</option>
                            <option value="live_locust">Live Locust Traffic (Port 8500)</option>
                        </select>
                        <div id="locust-container" class="hidden bg-slate-900/50 border border-slate-800 p-2.5 rounded-lg space-y-1.5">
                            <div class="flex justify-between text-[10px]">
                                <span class="text-slate-400">Locust Traffic (QPS):</span>
                                <span class="font-bold text-accentBlue" id="locust-qps-value">500</span>
                            </div>
                            <input type="range" id="locust-slider" min="10" max="2000" value="500" oninput="updateLocustQps(this.value)" class="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-accentBlue">
                        </div>
                    </div>
                </div>

                <!-- Simulation Status Indicator -->
                <div class="bg-slate-900/70 border border-slate-800/80 p-3 rounded-lg space-y-1">
                    <div class="flex justify-between text-xs">
                        <span class="text-slate-400">Simulation Status:</span>
                        <span id="lbl-status" class="font-bold text-accentGreen">Idle</span>
                    </div>
                    <div class="flex justify-between text-xs">
                        <span class="text-slate-400">Autoscaling Mode:</span>
                        <span id="lbl-mode" class="font-bold text-accentBlue text-right">Simulation</span>
                    </div>
                </div>

                <!-- Controls -->
                <div class="flex-1 flex flex-col justify-end pt-3 space-y-3">
                    <div class="grid grid-cols-2 gap-2">
                        <button id="btn-play" onclick="togglePlay()" class="bg-accentBlue hover:bg-blue-600 active:scale-95 transition text-white text-xs font-semibold py-2 px-3 rounded shadow flex items-center justify-center space-x-1">
                            <span id="play-icon">▶</span> <span id="play-text">Play</span>
                        </button>
                        <button id="btn-step" onclick="stepSimulation()" class="bg-slate-700 hover:bg-slate-600 active:scale-95 transition text-slate-200 text-xs font-semibold py-2 px-3 rounded border border-slate-600 flex items-center justify-center space-x-1">
                            <span>⏯</span> <span>Step</span>
                        </button>
                    </div>
                    <button id="btn-reset" onclick="resetSimulation()" class="w-full bg-slate-900 hover:bg-slate-800 active:scale-95 transition text-slate-400 hover:text-white border border-slate-800 text-[10px] font-semibold py-1 rounded text-center">
                        Reset Data
                    </button>
                    <div id="sim-speed-group">
                        <div class="flex justify-between text-[10px] mb-1">
                            <span class="text-slate-400">Poll Speed / Interval:</span>
                            <span class="text-slate-250 font-semibold" id="speed-label">500ms</span>
                        </div>
                        <input type="range" id="speed-slider" min="100" max="5000" step="100" value="500" onchange="updateSpeed(this.value)" class="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-slate-400">
                    </div>
                </div>
            </section>

            <!-- Dashboard Stats & Visualizations -->
            <section class="flex-1 flex flex-col space-y-5 overflow-y-auto pr-1">
                <!-- KPI Panels -->
                <div class="grid grid-cols-2 lg:grid-cols-5 gap-4">
                    <!-- Workload -->
                    <div class="bg-panelBg rounded-xl p-4 border border-slate-800 flex flex-col justify-between shadow">
                        <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wide">Workload Load</span>
                        <div class="flex items-baseline space-x-1.5 mt-1.5">
                            <span class="text-2xl font-black text-white" id="stat-obs-qps">0.0</span>
                            <span class="text-xs text-slate-400">QPS</span>
                        </div>
                        <div class="text-[9px] text-slate-450 mt-1" id="stat-forecast-info">Warming forecaster...</div>
                    </div>

                    <!-- Pod Count (Live vs Predicted vs Needed) -->
                    <div class="bg-panelBg rounded-xl p-4 border border-slate-800 flex flex-col justify-between shadow">
                        <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wide">Replicas comparison</span>
                        <div class="flex items-baseline space-x-1.5 mt-1.5">
                            <span class="text-2xl font-black text-white" id="stat-pods">1</span>
                            <span class="text-xs text-slate-400">Active</span>
                            <span class="text-xs text-accentBlue" id="stat-predicted-pods">/ 1 Pred</span>
                        </div>
                        <div class="text-[9px] text-slate-400 mt-1" id="stat-needed-pods">Needed (Optimal): 1 Pod</div>
                    </div>

                    <!-- Latency & SLO -->
                    <div class="bg-panelBg rounded-xl p-4 border border-slate-800 flex flex-col justify-between shadow">
                        <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wide">Latency & SLO</span>
                        <div class="flex items-baseline space-x-1.5 mt-1.5">
                            <span class="text-2xl font-black text-white" id="stat-latency">80.0</span>
                            <span class="text-xs text-slate-400">ms</span>
                            <span class="text-[10px] px-1.5 py-0.5 rounded font-black text-white bg-accentGreen ml-1" id="stat-slo-badge">OK</span>
                        </div>
                        <div class="text-[9px] text-slate-400 mt-1">
                            <span id="stat-util">Utilization: 0.0%</span> | Cadence: <span id="rate-cadence">60.0s</span>
                        </div>
                    </div>

                    <!-- Allocation Costs -->
                    <div class="bg-panelBg rounded-xl p-4 border border-slate-800 flex flex-col justify-between shadow">
                        <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wide">Allocated CPU Size</span>
                        <div class="flex items-baseline space-x-1 mt-1.5">
                            <span class="text-2xl font-black text-white" id="stat-cpu">600m</span>
                            <span class="text-xs text-slate-455 ml-1">per Pod</span>
                        </div>
                        <div class="text-[9px] text-slate-400 mt-1" id="stat-total-cores">Total allocation: 0.6 Cores</div>
                    </div>

                    <!-- SLO Violation Rate -->
                    <div class="bg-panelBg rounded-xl p-4 border border-slate-800 flex flex-col justify-between shadow">
                        <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wide">Cost & Violations</span>
                        <div class="flex items-baseline space-x-1.5 mt-1.5">
                            <span class="text-2xl font-black text-white" id="rate-violations">0.0%</span>
                            <span class="text-[10px] text-slate-400">violations</span>
                        </div>
                        <div class="text-[9px] text-slate-400 mt-1">
                            <span id="stat-cost-summary">Total Cost: $0.0000</span> | <span id="rate-ticks">0</span> ticks
                        </div>
                    </div>
                </div>

                <!-- Fig 6 Chart 1: Resource per pod (CPU millicores) -->
                <div class="bg-panelBg p-4 rounded-xl border border-slate-800 shadow flex flex-col h-[220px]">
                    <div class="flex justify-between items-center mb-2">
                        <h3 class="text-xs font-bold text-slate-300">Resource per Pod (minicore CPU)</h3>
                        <div class="flex items-center space-x-3 text-[10px] text-slate-400">
                            <span class="flex items-center"><span class="h-1 w-4 bg-[#f59e0b] block mr-1.5"></span>Hybrid Scaling (Proposed)</span>
                            <span class="flex items-center"><span class="h-1 w-4 bg-[#3b82f6] block mr-1.5"></span>Proactive HPA</span>
                            <span class="flex items-center"><span class="h-1 w-4 bg-[#10b981] block mr-1.5"></span>Reactive HPA</span>
                        </div>
                    </div>
                    <div class="flex-1 min-h-0 relative">
                        <canvas id="chart-cpu-per-pod"></canvas>
                    </div>
                </div>

                <!-- Fig 6 Chart 2: Number of pods -->
                <div class="bg-panelBg p-4 rounded-xl border border-slate-800 shadow flex flex-col h-[220px]">
                    <div class="flex justify-between items-center mb-2">
                        <h3 class="text-xs font-bold text-slate-300">Number of Pods</h3>
                        <div class="flex items-center space-x-3 text-[10px] text-slate-400">
                            <span class="flex items-center"><span class="h-1 w-4 bg-[#f59e0b] block mr-1.5"></span>Hybrid (Active)</span>
                            <span class="flex items-center"><span class="h-1 w-4 bg-[#3b82f6] block mr-1.5"></span>Predicted</span>
                            <span class="flex items-center"><span class="h-1 w-4 bg-[#10b981] block mr-1.5"></span>Optimal</span>
                        </div>
                    </div>
                    <div class="flex-1 min-h-0 relative">
                        <canvas id="chart-num-pods"></canvas>
                    </div>
                </div>

                <!-- Fig 6 Chart 3: Avg pod utilization -->
                <div class="bg-panelBg p-4 rounded-xl border border-slate-800 shadow flex flex-col h-[220px]">
                    <div class="flex justify-between items-center mb-2">
                        <h3 class="text-xs font-bold text-slate-300">Avg Pod Utilization (%)</h3>
                        <div class="flex items-center space-x-3 text-[10px] text-slate-400">
                            <span class="flex items-center"><span class="h-1 w-4 bg-[#f59e0b] block mr-1.5"></span>Hybrid Utilization</span>
                            <span class="flex items-center"><span class="h-1 w-4 bg-[#10b981] block mr-1.5"></span>QPS Observed</span>
                            <span class="flex items-center"><span class="h-1 w-4 bg-[#3b82f6] block mr-1.5"></span>QPS Forecast</span>
                        </div>
                    </div>
                    <div class="flex-1 min-h-0 relative">
                        <canvas id="chart-utilization"></canvas>
                    </div>
                </div>

                <!-- Fig 6 Chart 4: Avg response time with QoS constraint -->
                <div class="bg-panelBg p-4 rounded-xl border border-slate-800 shadow flex flex-col h-[220px]">
                    <div class="flex justify-between items-center mb-2">
                        <h3 class="text-xs font-bold text-slate-300">Avg Response Time (ms)</h3>
                        <div class="flex items-center space-x-3 text-[10px] text-slate-400">
                            <span class="flex items-center"><span class="h-1 w-4 bg-[#f59e0b] block mr-1.5"></span>Latency</span>
                            <span class="flex items-center"><span class="h-1 w-4 border-t-2 border-dashed border-[#ef4444] mr-1.5"></span>QoS Constraint (350ms)</span>
                        </div>
                    </div>
                    <div class="flex-1 min-h-0 relative">
                        <canvas id="chart-response-time"></canvas>
                    </div>
                </div>
            </section>

        </main>

        <script>
            let playInterval = null;
            let chartCpu, chartPods, chartUtil, chartLatency;
            const maxChartDataPoints = 80;
            const SLO_MS = 350;

            document.addEventListener("DOMContentLoaded", () => {
                initializeCharts();
                fetchStatus();
            });

            
            function setMode(mode) {
                const btnSim = document.getElementById("mode-sim");
                const btnLive = document.getElementById("mode-live");
                const simPresetsGroup = document.getElementById("sim-presets-group");
                const simSourceGroup = document.getElementById("sim-source-group");
                const simSpeedGroup = document.getElementById("sim-speed-group");
                const liveConfigGroup = document.getElementById("live-config-group");
                const lblMode = document.getElementById("lbl-mode");
                
                pauseLoop();
                
                if (mode === "simulation") {
                    btnSim.className = "px-4 py-1.5 rounded-md text-xs font-semibold bg-accentBlue text-white shadow transition";
                    btnLive.className = "px-4 py-1.5 rounded-md text-xs font-semibold text-slate-400 hover:text-slate-200 transition";
                    simPresetsGroup.classList.remove("hidden");
                    simSourceGroup.classList.remove("hidden");
                    simSpeedGroup.classList.remove("hidden");
                    liveConfigGroup.classList.add("hidden");
                    lblMode.innerText = "Simulation";
                    lblMode.className = "font-bold text-accentBlue text-right";
                    
                    const src = document.getElementById("source-selector").value;
                    if (src === "live_locust") {
                        document.getElementById("speed-slider").value = 1000;
                        document.getElementById("speed-label").innerText = "1000ms";
                    } else {
                        document.getElementById("speed-slider").value = 500;
                        document.getElementById("speed-label").innerText = "500ms";
                    }
                } else {
                    btnLive.className = "px-4 py-1.5 rounded-md text-xs font-semibold bg-accentBlue text-white shadow transition";
                    btnSim.className = "px-4 py-1.5 rounded-md text-xs font-semibold text-slate-400 hover:text-slate-200 transition";
                    simPresetsGroup.classList.add("hidden");
                    simSourceGroup.classList.add("hidden");
                    simSpeedGroup.classList.remove("hidden");
                    liveConfigGroup.classList.remove("hidden");
                    lblMode.innerText = "Live Kubernetes";
                    lblMode.className = "font-bold text-accentGreen text-right";
                    
                    document.getElementById("speed-slider").value = 5000;
                    document.getElementById("speed-label").innerText = "5.0s Interval";
                }
                
                fetch("/api/set_mode?mode=" + mode, { method: "POST" })
                .then(res => res.json())
                .then(() => {
                    resetSimulation();
                });
            }
            
            function updateLiveConfig() {
                const data = {
                    "prometheus_url": document.getElementById("live-prom-url").value,
                    "controller_url": document.getElementById("live-ctrl-url").value,
                    "k8s_namespace": document.getElementById("live-ns").value,
                    "k8s_deployment": document.getElementById("live-deploy").value
                };
                
                fetch("/api/update_live_config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });
            }
            
            function initializeCharts() {
                const gridColor  = 'rgba(51,65,85,0.35)';
                const tickStyle  = { color: '#94a3b8', font: { size: 9 } };
                const baseOpts   = {
                    responsive: true, maintainAspectRatio: false,
                    animation: false,
                    plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
                    scales: {
                        x: { grid: { color: gridColor }, ticks: { ...tickStyle, maxTicksLimit: 15 },
                             title: { display: true, text: 'Time interval', color: '#64748b', font: { size: 9 } } }
                    }
                };

                // ── Chart 1: Resource per Pod (CPU millicores) ──────────────────
                chartCpu = new Chart(
                    document.getElementById('chart-cpu-per-pod').getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [
                            { label: 'Hybrid CPU/pod', data: [], borderColor: '#f59e0b',
                              borderWidth: 2, pointRadius: 0, stepped: true },
                            { label: 'Proactive (fixed)', data: [], borderColor: '#3b82f6',
                              borderWidth: 1.5, pointRadius: 0, borderDash: [6, 3],
                              // Proactive HPA keeps CPU fixed — drawn as a flat reference
                              stepped: true },
                            { label: 'Reactive HPA', data: [], borderColor: '#10b981',
                              borderWidth: 1.5, pointRadius: 0, borderDash: [2, 3], stepped: true }
                        ]
                    },
                    options: { ...baseOpts,
                        scales: { ...baseOpts.scales,
                            y: { grid: { color: gridColor }, ticks: { ...tickStyle },
                                 title: { display: true, text: 'resource per pod (millicores)', color: '#94a3b8', font: { size: 9 } },
                                 min: 0 }
                        }
                    }
                });

                // ── Chart 2: Number of pods ──────────────────────────────────────
                chartPods = new Chart(
                    document.getElementById('chart-num-pods').getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [
                            { label: 'Active pods (Hybrid)', data: [], borderColor: '#f59e0b',
                              borderWidth: 2, pointRadius: 2, pointHoverRadius: 4,
                              stepped: true },
                            { label: 'Predicted pods', data: [], borderColor: '#3b82f6',
                              borderWidth: 1.5, pointRadius: 0, borderDash: [5, 3],
                              stepped: true, spanGaps: false },
                            { label: 'Optimal pods', data: [], borderColor: '#10b981',
                              borderWidth: 1.5, pointRadius: 0, borderDash: [2, 3], stepped: true }
                        ]
                    },
                    options: { ...baseOpts,
                        scales: { ...baseOpts.scales,
                            y: { grid: { color: gridColor },
                                 ticks: { ...tickStyle, stepSize: 1, precision: 0 },
                                 title: { display: true, text: 'Number of pods', color: '#94a3b8', font: { size: 9 } },
                                 min: 0 }
                        }
                    }
                });

                // ── Chart 3: Avg pod utilization (%) ────────────────────────────
                // Also overlays QPS observed and forecast on a right-hand y-axis
                // so you can see workload shape driving utilization (like Fig 6 row 3).
                chartUtil = new Chart(
                    document.getElementById('chart-utilization').getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [
                            { label: 'Utilization %', data: [], borderColor: '#f59e0b',
                              borderWidth: 2, pointRadius: 0, tension: 0.1, yAxisID: 'y' },
                            { label: 'QPS Observed', data: [], borderColor: '#10b981',
                              borderWidth: 1.2, pointRadius: 0, tension: 0.15,
                              borderDash: [4, 2], yAxisID: 'y1' },
                            { label: 'QPS Forecast', data: [], borderColor: '#3b82f6',
                              borderWidth: 1.2, pointRadius: 0, tension: 0.15,
                              borderDash: [2, 3], yAxisID: 'y1', spanGaps: false }
                        ]
                    },
                    options: { ...baseOpts,
                        scales: { ...baseOpts.scales,
                            y:  { grid: { color: gridColor }, ticks: { ...tickStyle, color: '#f59e0b' },
                                  title: { display: true, text: 'Avg pod utilization (%)', color: '#f59e0b', font: { size: 9 } },
                                  min: 0, max: 100 },
                            y1: { position: 'right', grid: { drawOnChartArea: false },
                                  ticks: { ...tickStyle }, title: { display: true, text: 'QPS', color: '#94a3b8', font: { size: 9 } },
                                  min: 0 }
                        }
                    }
                });

                // ── Chart 4: Avg response time (ms) with QoS constraint ──────────
                chartLatency = new Chart(
                    document.getElementById('chart-response-time').getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [
                            { label: 'Avg response time (ms)', data: [], borderColor: '#f59e0b',
                              borderWidth: 2, pointRadius: 0, tension: 0.1, fill: false }
                        ]
                    },
                    options: { ...baseOpts,
                        plugins: { ...baseOpts.plugins,
                            annotation: {
                                annotations: {
                                    sloLine: {
                                        type: 'line', yMin: SLO_MS, yMax: SLO_MS,
                                        borderColor: '#ef4444', borderWidth: 1.5,
                                        borderDash: [6, 4],
                                        label: { content: 'QoS constraint', enabled: true,
                                                 position: 'end', color: '#ef4444',
                                                 font: { size: 9 } }
                                    }
                                }
                            }
                        },
                        scales: { ...baseOpts.scales,
                            y: { grid: { color: gridColor }, ticks: { ...tickStyle },
                                 title: { display: true, text: 'Avg response time (ms)', color: '#94a3b8', font: { size: 9 } },
                                 min: 0, suggestedMax: 600 }
                        }
                    }
                });
            }

            function _trim(chart) {
                while (chart.data.labels.length > maxChartDataPoints) {
                    chart.data.labels.shift();
                    chart.data.datasets.forEach(d => d.data.shift());
                }
            }

            function appendChartPoints(latest) {
                const t = latest.tick;

                // Chart 1 — CPU per pod
                // "Proactive HPA" baseline = fixed 950 m (same as paper's flat blue line)
                // "Reactive HPA"  baseline = 600 m (never adjusts, reactive reference)
                chartCpu.data.labels.push(t);
                chartCpu.data.datasets[0].data.push(latest.cpu_millicores);   // Hybrid
                chartCpu.data.datasets[1].data.push(950);                      // Proactive (fixed)
                chartCpu.data.datasets[2].data.push(600);                      // Reactive (fixed)
                _trim(chartCpu);
                chartCpu.update('none');

                // Chart 2 — Number of pods
                chartPods.data.labels.push(t);
                chartPods.data.datasets[0].data.push(latest.replicas);
                chartPods.data.datasets[1].data.push(latest.predicted_replicas);
                chartPods.data.datasets[2].data.push(latest.optimal_replicas);
                _trim(chartPods);
                chartPods.update('none');

                // Chart 3 — Avg pod utilization (%) + QPS overlay
                chartUtil.data.labels.push(t);
                chartUtil.data.datasets[0].data.push(latest.utilization);
                chartUtil.data.datasets[1].data.push(latest.observed_qps);
                chartUtil.data.datasets[2].data.push(latest.predicted_qps);
                _trim(chartUtil);
                chartUtil.update('none');

                // Chart 4 — Avg response time (ms)
                chartLatency.data.labels.push(t);
                chartLatency.data.datasets[0].data.push(latest.latency_ms);
                _trim(chartLatency);
                chartLatency.update('none');
            }



            
            function applyPreset() {
                const preset = document.getElementById("preset-selector").value;
                const toggleN1 = document.getElementById("toggle-n1");
                const toggleN2 = document.getElementById("toggle-n2");
                const toggleN3 = document.getElementById("toggle-n3");
                const toggleN4 = document.getElementById("toggle-n4");
                
                if (preset === "paper") {
                    toggleN1.checked = false;
                    toggleN2.checked = false;
                    toggleN3.checked = false;
                    toggleN4.checked = false;
                } else if (preset === "novelty") {
                    toggleN1.checked = true;
                    toggleN2.checked = true;
                    toggleN3.checked = true;
                    toggleN4.checked = true;
                }
                
                configChanged(preset);
            }
            
            function configChanged(presetValue) {
                if (presetValue === undefined || presetValue === "custom") {
                    presetValue = "custom";
                    document.getElementById("preset-selector").value = "custom";
                }
                
                const data = {
                    "novelty1_burst": document.getElementById("toggle-n1").checked,
                    "novelty2_quantile": document.getElementById("toggle-n2").checked,
                    "novelty3_inplace": document.getElementById("toggle-n3").checked,
                    "novelty4_monitoring": document.getElementById("toggle-n4").checked,
                };
                
                fetch("/api/update_config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });
            }
            
            function updateSource() {
                const src = document.getElementById("source-selector").value;
                const locustContainer = document.getElementById("locust-container");
                if (src === "locust") {
                    locustContainer.classList.remove("hidden");
                } else {
                    locustContainer.classList.add("hidden");
                }
                
                if (src === "live_locust") {
                    document.getElementById("speed-slider").value = 1000;
                    document.getElementById("speed-label").innerText = "1000ms";
                }
                
                fetch("/api/update_source?source=" + src, { method: "POST" })
                .then(() => {
                    resetSimulation();
                });
            }
            
            function updateLocustQps(val) {
                document.getElementById("locust-qps-value").innerText = val;
                fetch("/api/update_locust_qps?qps=" + val, { method: "POST" });
            }
            
            function updateSpeed(val) {
                const isLive = document.getElementById("lbl-mode").innerText.includes("Live");
                document.getElementById("speed-label").innerText = isLive ? (val/1000).toFixed(1) + "s Interval" : val + "ms";
                fetch("/api/update_speed?speed_ms=" + val, { method: "POST" })
                .then(() => {
                    if (playInterval) {
                        pauseLoop();
                        playLoop(parseInt(val));
                    }
                });
            }
            
            function stepSimulation() {
                const isLive = document.getElementById("lbl-mode").innerText.includes("Live");
                const endpoint = isLive ? "/api/live_step" : "/api/step";
                
                fetch(endpoint, { method: "POST" })
                .then(res => res.json())
                .then(data => {
                    updateDashboard(data);
                });
            }
            
            function togglePlay() {
                const speed = parseInt(document.getElementById("speed-slider").value);
                if (playInterval) {
                    pauseLoop();
                } else {
                    playLoop(speed);
                }
            }
            
            function playLoop(speedMs) {
                document.getElementById("play-icon").innerText = "⏸";
                document.getElementById("play-text").innerText = "Pause";
                document.getElementById("btn-play").className = "bg-accentOrange hover:bg-orange-650 active:scale-95 transition text-white text-xs font-semibold py-2 px-3 rounded shadow flex items-center justify-center space-x-1";
                document.getElementById("lbl-status").innerText = "Running";
                document.getElementById("lbl-status").className = "font-bold text-accentOrange";
                
                fetch("/api/set_play_state?playing=true", { method: "POST" });
                
                playInterval = setInterval(() => {
                    stepSimulation();
                }, speedMs);
            }
            
            function pauseLoop() {
                if (playInterval) {
                    clearInterval(playInterval);
                    playInterval = null;
                }
                document.getElementById("play-icon").innerText = "▶";
                document.getElementById("play-text").innerText = "Play";
                document.getElementById("btn-play").className = "bg-accentBlue hover:bg-blue-600 active:scale-95 transition text-white text-xs font-semibold py-2 px-3 rounded shadow flex items-center justify-center space-x-1";
                document.getElementById("lbl-status").innerText = "Paused";
                document.getElementById("lbl-status").className = "font-bold text-accentGreen";
                
                fetch("/api/set_play_state?playing=false", { method: "POST" });
            }
            
            function resetSimulation() {
                pauseLoop();
                fetch("/api/reset", { method: "POST" })
                .then(res => res.json())
                .then(data => {
                    podsCompareChart.data.labels = [];
                    podsCompareChart.data.datasets.forEach(d => d.data = []);
                    podsCompareChart.update();
                    
                    qpsChart.data.labels = [];
                    qpsChart.data.datasets.forEach(d => d.data = []);
                    qpsChart.update();
                    
                    utilLatChart.data.labels = [];
                    utilLatChart.data.datasets.forEach(d => d.data = []);
                    utilLatChart.update();
                    
                    updateDashboard(data);
                    document.getElementById("lbl-status").innerText = "Idle";
                    document.getElementById("lbl-status").className = "font-bold text-accentGreen";
                });
            }
            
            function fetchStatus() {
                fetch("/api/status")
                .then(res => res.json())
                .then(data => {
                    document.getElementById("toggle-n1").checked = data.config.novelty1_burst;
                    document.getElementById("toggle-n2").checked = data.config.novelty2_quantile;
                    document.getElementById("toggle-n3").checked = data.config.novelty3_inplace;
                    document.getElementById("toggle-n4").checked = data.config.novelty4_monitoring;
                    document.getElementById("source-selector").value = data.config.source;
                    document.getElementById("speed-slider").value = data.config.simulation_speed_ms;
                    
                    const isLive = data.config.mode === "live_kubernetes";
                    document.getElementById("speed-label").innerText = isLive ? (data.config.simulation_speed_ms/1000).toFixed(1) + "s Interval" : data.config.simulation_speed_ms + "ms";
                    
                    if (data.config.source === "locust") {
                        document.getElementById("locust-container").classList.remove("hidden");
                        document.getElementById("locust-slider").value = data.config.locust_manual_qps;
                        document.getElementById("locust-qps-value").innerText = data.config.locust_manual_qps;
                    }
                    
                    document.getElementById("live-prom-url").value = data.config.prometheus_url;
                    document.getElementById("live-ctrl-url").value = data.config.controller_url;
                    document.getElementById("live-ns").value = data.config.k8s_namespace;
                    document.getElementById("live-deploy").value = data.config.k8s_deployment;
                    
                    if (isLive) {
                        setMode("live_kubernetes");
                    } else {
                        setMode("simulation");
                    }
                    
                    updateDashboard(data.latest);
                    
                    if (data.config.is_playing) {
                        playLoop(data.config.simulation_speed_ms);
                    }
                });
            }
            
            function updateDashboard(latest) {
                if (!latest || latest.tick === undefined) {
                    document.getElementById("stat-obs-qps").innerText = "0.0";
                    document.getElementById("stat-pods").innerText = "1";
                    document.getElementById("stat-predicted-pods").innerText = "/ 1 Pred";
                    document.getElementById("stat-needed-pods").innerText = "Needed (Optimal): 1 Pod";
                    document.getElementById("stat-cpu").innerText = "600m";
                    document.getElementById("stat-total-cores").innerText = "Total allocation: 0.6 Cores";
                    document.getElementById("stat-latency").innerText = "80.0";
                    document.getElementById("stat-util").innerText = "Utilization: 0.0%";
                    document.getElementById("stat-cost-summary").innerText = "Total Cost: $0.0000";
                    document.getElementById("rate-violations").innerText = "0.0%";
                    document.getElementById("rate-cadence").innerText = "60.0s";
                    document.getElementById("rate-ticks").innerText = "0";
                    document.getElementById("stat-forecast-info").innerText = "Warming forecaster...";
                    return;
                }
                
                document.getElementById("stat-obs-qps").innerText = latest.observed_qps.toFixed(1);
                document.getElementById("stat-pods").innerText = latest.replicas ?? "—";
                document.getElementById("stat-predicted-pods").innerText = latest.predicted_replicas != null ? "/ " + latest.predicted_replicas + " Pred" : "/ — Pred";
                document.getElementById("stat-needed-pods").innerText = "Needed (Optimal): " + (latest.optimal_replicas ?? "—") + " Pod" + ((latest.optimal_replicas ?? 0) > 1 ? "s" : "");
                document.getElementById("stat-cpu").innerText = latest.cpu_millicores + "m";
                document.getElementById("stat-total-cores").innerText = "Total allocation: " + ((latest.replicas * latest.cpu_millicores) / 1000).toFixed(2) + " Cores";
                document.getElementById("stat-latency").innerText = latest.latency_ms.toFixed(1);
                document.getElementById("stat-util").innerText = "Utilization: " + latest.utilization.toFixed(1) + "%";
                document.getElementById("stat-cost-summary").innerText = "Total Cost: $" + latest.total_cost.toFixed(4);
                
                document.getElementById("rate-violations").innerText = latest.slo_violation_rate.toFixed(1) + "%";
                document.getElementById("rate-cadence").innerText = latest.average_interval.toFixed(1) + "s";
                document.getElementById("rate-ticks").innerText = latest.tick + 1;
                
                const badge = document.getElementById("stat-slo-badge");
                if (latest.slo_violation) {
                    badge.innerText = "MISS";
                    badge.className = "text-xs px-1.5 py-0.5 rounded font-black text-white bg-accentRed";
                } else {
                    badge.innerText = "OK";
                    badge.className = "text-xs px-1.5 py-0.5 rounded font-black text-white bg-accentGreen";
                }
                
                // Burst detection display - show prominent visual indicator
                const forecastInfo = document.getElementById("stat-forecast-info");
                if (latest.predicted_qps === null) {
                    forecastInfo.innerHTML = `<span class="font-bold text-slate-500">⏳ Warming up (${latest.tick + 1}/10 ticks)...</span>`;
                } else if (latest.burst) {
                    forecastInfo.innerHTML = `<span class="font-bold text-accentRed animate-pulse">💥 BURST DETECTED — Scaling up!</span><br>` +
                        `Forecast: <b>${latest.predicted_qps.toFixed(1)}</b> QPS`;
                } else {
                    forecastInfo.innerHTML = `<span class="font-bold text-accentGreen">🟢 Stable Workload</span><br>` +
                        `Forecast: ${latest.predicted_qps.toFixed(1)} QPS`;
                }
                
                appendChartPoints(latest);
            }
            


        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/status")
def get_status():
    latest_step = {}
    if state.ticks_count > 0:
        latest_step = {
            "tick": state.chart_ticks[-1],
            "observed_qps": state.chart_observed[-1],
            "predicted_qps": state.chart_predicted[-1],
            "p10": state.chart_p10[-1],
            "p90": state.chart_p90[-1],
            "burst": bool(state.chart_burst[-1]),
            "replicas": state.chart_replicas[-1],
            "predicted_replicas": state.chart_predicted_replicas[-1],
            "optimal_replicas": state.chart_optimal_replicas[-1],
            "cpu_millicores": state.chart_cpu[-1],
            "utilization": state.chart_utilization[-1],
            "latency_ms": state.chart_latency[-1],
            "slo_violation": bool(state.chart_latency[-1] > 350.0),
            "monitoring_interval": state.chart_interval[-1],
            "total_cost": state.chart_cost[-1],
            "overlap_cost": state.overlap_cost,
            "api_cycles": state.api_cycles,
            "slo_violation_rate": (state.slo_violations / max(1, state.ticks_count)) * 100.0,
            "average_interval": sum(state.chart_interval) / max(1, len(state.chart_interval))
        }
        
    return JSONResponse(content={
        "config": config,
        "chart_ticks": list(state.chart_ticks)[-maxChartDataPoints:] if state.chart_ticks else [],
        "chart_observed": list(state.chart_observed)[-maxChartDataPoints:] if state.chart_observed else [],
        "chart_predicted": list(state.chart_predicted)[-maxChartDataPoints:] if state.chart_predicted else [],
        "chart_p10": list(state.chart_p10)[-maxChartDataPoints:] if state.chart_p10 else [],
        "chart_p90": list(state.chart_p90)[-maxChartDataPoints:] if state.chart_p90 else [],
        "chart_replicas": list(state.chart_replicas)[-maxChartDataPoints:] if state.chart_replicas else [],
        "chart_predicted_replicas": list(state.chart_predicted_replicas)[-maxChartDataPoints:] if state.chart_predicted_replicas else [],
        "chart_optimal_replicas": list(state.chart_optimal_replicas)[-maxChartDataPoints:] if state.chart_optimal_replicas else [],
        "chart_cpu": list(state.chart_cpu)[-maxChartDataPoints:] if state.chart_cpu else [],
        "chart_utilization": list(state.chart_utilization)[-maxChartDataPoints:] if state.chart_utilization else [],
        "chart_latency": list(state.chart_latency)[-maxChartDataPoints:] if state.chart_latency else [],
        "chart_burst": list(state.chart_burst)[-maxChartDataPoints:] if state.chart_burst else [],
        "chart_interval": list(state.chart_interval)[-maxChartDataPoints:] if state.chart_interval else [],
        "chart_cost": list(state.chart_cost)[-maxChartDataPoints:] if state.chart_cost else [],
        "latest": latest_step
    })

@app.post("/api/set_mode")
def set_mode(mode: str):
    if mode in ("simulation", "live_kubernetes"):
        config["mode"] = mode
        if mode == "live_kubernetes":
            config["source"] = "live_locust"
    return {"status": "ok"}

@app.post("/api/update_config")
def update_config(data: dict):
    config["novelty1_burst"] = data.get("novelty1_burst", config["novelty1_burst"])
    config["novelty2_quantile"] = data.get("novelty2_quantile", config["novelty2_quantile"])
    config["novelty3_inplace"] = data.get("novelty3_inplace", config["novelty3_inplace"])
    config["novelty4_monitoring"] = data.get("novelty4_monitoring", config["novelty4_monitoring"])
    return {"status": "ok"}

@app.post("/api/update_live_config")
def update_live_config(data: dict):
    config["prometheus_url"] = data.get("prometheus_url", config["prometheus_url"])
    config["controller_url"] = data.get("controller_url", config["controller_url"])
    config["k8s_namespace"] = data.get("k8s_namespace", config["k8s_namespace"])
    config["k8s_deployment"] = data.get("k8s_deployment", config["k8s_deployment"])
    return {"status": "ok"}

@app.post("/api/update_source")
def update_source(source: str):
    if source in ("worldcup", "synthetic", "locust", "live_locust"):
        config["source"] = source
    return {"status": "ok"}

@app.post("/api/update_locust_qps")
def update_locust_qps(qps: float):
    config["locust_manual_qps"] = qps
    return {"status": "ok"}

@app.post("/api/update_speed")
def update_speed(speed_ms: int):
    config["simulation_speed_ms"] = speed_ms
    return {"status": "ok"}

@app.post("/api/set_play_state")
def set_play_state(playing: bool):
    config["is_playing"] = playing
    return {"status": "ok"}

@app.post("/api/step")
def step_sim():
    return run_simulation_step()

@app.post("/api/live_step")
def live_step():
    return run_live_kubernetes_step()

@app.post("/api/reset")
def reset_sim():
    state.reset()
    return {}

# ---------------------------------------------------------------------------
# Real Locust target API implementation (handles port conflicts)
# ---------------------------------------------------------------------------
@app.get("/predict")
def predict_endpoint(area: float = 1800, rooms: int = 3, age: int = 10):
    # Track the request count to compute live QPS
    tracker.increment()
    return {"estimated_price": round(50000 + area * 160 + rooms * 22000 - age * 900, 2)}

@app.get("/healthz")
def healthz_endpoint():
    return {"status": "ok"}

@app.get("/metrics")
def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    maxChartDataPoints = 50
    # Warm up history with starting values to speed up forecaster start
    raw_init, scaled_init = get_workload_point(0)
    for _ in range(9):
        state.history.append(float(scaled_init))
        
    print("Starting dashboard server on http://127.0.0.1:8500...", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8500)
