"""Measured-DTR hybrid autoscaler with runtime-selectable paper/novelty policies."""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from Novelty.src.burst import AdaptiveBurstDetector, BurstParameters, OnlineBurstDetector
from Novelty.src.forecast import FORECASTERS
from Novelty.src.kubernetes_executor import InPlaceResizeExecutor, RollingUpdateExecutor
from Novelty.src.monitor import prometheus_qps
from Novelty.src.monitoring import AdaptiveMonitoringInterval
from Novelty.src.performance import PERFORMANCE_MODELS
from Novelty.src.scaler import HybridProvisioner, UncertaintyAwareProvisioner

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("novelty-controller")
app = FastAPI(title="Measured-DTR Predictive Hybrid Autoscaler")

ROOT = Path(__file__).resolve().parents[2]
namespace = os.getenv("NAMESPACE", "hybrid-autoscaling")
deployment_name = os.getenv("DEPLOYMENT", "house-price")
container_name = os.getenv("CONTAINER", "app")
apply_changes = os.getenv("APPLY_CHANGES", "false").lower() == "true"
fixed_interval = max(5, int(os.getenv("INTERVAL_SECONDS", "60")))
slo_ms = float(os.getenv("SLO_MS", "350"))
policy_calibration_path = Path(os.getenv(
    "POLICY_CALIBRATION", str(ROOT / "Novelty" / "models" / "policy_calibration.json")))
try:
    default_calibration = json.loads(policy_calibration_path.read_text(encoding="utf-8"))
except FileNotFoundError:
    default_calibration = {}

performance_kind = os.getenv("PERFORMANCE_MODEL_KIND", "dtr").lower()
performance_path = Path(os.getenv(
    "PERFORMANCE_MODEL", str(ROOT / "Novelty" / "models" / "performance.joblib")))
performance = PERFORMANCE_MODELS[performance_kind].load(str(performance_path))

# Load per-trace forecast artifacts so both public traces are tested with
# forecasters trained on that same trace, while every arm shares the measured DTR.
models_dir = Path(os.getenv("MODELS_DIR", str(ROOT / "Novelty" / "models")))
forecasters: dict[str, dict[str, Any]] = {"worldcup98": {}, "azurefunctions2019": {}}
calibrations: dict[str, dict[str, Any]] = {}
for trace in forecasters:
    trace_dir = models_dir / trace
    try:
        calibrations[trace] = json.loads((trace_dir / "policy_calibration.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        calibrations[trace] = default_calibration
    for kind in ("bilstm", "quantile_bilstm"):
        env_name = f"{trace.upper()}_{kind.upper()}_MODEL"
        candidate = Path(os.getenv(env_name, str(trace_dir / f"{kind}.pt")))
        if not candidate.is_file():
            candidate = Path(os.getenv(f"{kind.upper()}_MODEL", str(models_dir / f"{kind}.pt")))
        if not candidate.is_file() and trace == "worldcup98" and kind == "bilstm":
            candidate = ROOT / "Paper" / "models" / "bilstm.pt"
        try:
            if candidate.is_file():
                forecasters[trace][kind] = FORECASTERS[kind].load(str(candidate))
            else:
                log.warning("Missing %s forecast model for %s: %s", kind, trace, candidate)
        except Exception as exc:
            log.warning("Could not load %s from %s: %s", kind, candidate, exc)

policy_lock = threading.RLock()
history: deque[float] = deque(maxlen=120)
recent: deque[dict[str, Any]] = deque(maxlen=120)
latest: dict[str, Any] = {"status": "starting"}
current_policy = {
    "name": "paper_hybrid",
    "n1_burst": False,
    "n2_quantile": False,
    "n3_adaptive_monitoring": False,
    "n4_inplace_resize": False,
    "trace": "worldcup98",
}
forecast = None
detector = None
provisioner = None
executor = None
interval_policy = None


def _configure_components(policy: dict[str, Any]) -> None:
    global forecast, detector, provisioner, executor, interval_policy, current_policy, apply_changes, policy_calibration
    with policy_lock:
        if policy.get("apply_changes") is not None:
            apply_changes = bool(policy["apply_changes"])
        trace = str(policy.get("trace", current_policy.get("trace", "worldcup98")))
        if trace not in forecasters:
            raise HTTPException(422, "trace must be worldcup98 or azurefunctions2019")
        policy_calibration = calibrations[trace]
        next_policy = {
            "name": str(policy.get("name", "custom")),
            "n1_burst": bool(policy.get("n1_burst", False)),
            "n2_quantile": bool(policy.get("n2_quantile", False)),
            "n3_adaptive_monitoring": bool(policy.get("n3_adaptive_monitoring", False)),
            "n4_inplace_resize": bool(policy.get("n4_inplace_resize", False)),
            "trace": trace,
        }
        kind = "quantile_bilstm" if next_policy["n2_quantile"] else "bilstm"
        if kind not in forecasters[trace]:
            raise HTTPException(503, f"Required {trace} forecaster {kind} is unavailable; train the models first")
        forecast = forecasters[trace][kind]
        burst_parameters = policy_calibration.get("burst_parameters", {})
        allowed_burst_fields = BurstParameters.__dataclass_fields__
        tuned_burst_parameters = BurstParameters(**{
            key: value for key, value in burst_parameters.items()
            if key in allowed_burst_fields
        })
        detector = (AdaptiveBurstDetector(parameters=tuned_burst_parameters)
                    if next_policy["n1_burst"] else OnlineBurstDetector())
        if next_policy["n2_quantile"] or next_policy["n4_inplace_resize"]:
            # The measured calibration supports 450, 600, 800, and 950m CPU.
            # Let cost-aware novelty policies consider the validated 450m point;
            # keep the paper baseline's resource choices unchanged.
            resource_levels = ((450, 600, 800, 950)
                               if next_policy["n4_inplace_resize"]
                               else (600, 700, 800, 900, 950))
            provisioner = UncertaintyAwareProvisioner(
                performance, resources=resource_levels,
                cost_aware=next_policy["n4_inplace_resize"])
        else:
            provisioner = HybridProvisioner(performance)
        if isinstance(provisioner, UncertaintyAwareProvisioner):
            provisioner.interval_safety_gain = float(policy_calibration.get("quantile_safety_gain", 0.0))
            if policy_calibration.get("quantile_max_relative_width"):
                provisioner.max_relative_width = float(policy_calibration["quantile_max_relative_width"])
        interval_policy = AdaptiveMonitoringInterval(
            base_seconds=fixed_interval,
            minimum_seconds=int(policy_calibration.get("monitor_minimum_seconds", 15)),
            maximum_seconds=int(policy_calibration.get("monitor_maximum_seconds", 120)),
            high_cv_percentile=float(policy_calibration.get("monitor_high_cv_percentile", .40)),
        )
        if "monitor_low_cv" in policy_calibration:
            interval_policy.low_cv = float(policy_calibration["monitor_low_cv"])
        if "monitor_high_cv" in policy_calibration:
            interval_policy.high_cv = float(policy_calibration["monitor_high_cv"])
        if apply_changes:
            executor = (InPlaceResizeExecutor(namespace, deployment_name, container_name)
                        if next_policy["n4_inplace_resize"]
                        else RollingUpdateExecutor(namespace, deployment_name))
        else:
            executor = None
        history.clear()
        recent.clear()
        latest.clear()
        latest.update({"status": "warming", "policy": next_policy, "samples": 0})
        current_policy = next_policy


_configure_components(current_policy)


class PolicyRequest(BaseModel):
    name: str = "custom"
    n1_burst: bool = False
    n2_quantile: bool = False
    n3_adaptive_monitoring: bool = False
    n4_inplace_resize: bool = False
    trace: str = "worldcup98"
    apply_changes: bool | None = None
    initial_history: list[float] | None = None


def process(qps: float) -> dict[str, Any]:
    global latest
    observed = float(qps)
    if observed < 0:
        raise ValueError("Observed request rate cannot be negative")
    with policy_lock:
        history.append(observed)
        if forecast is None or len(history) < forecast.lookback:
            result = {"status": "warming", "samples": len(history), "policy": dict(current_policy)}
            latest = result
            return result
        values = list(history)
        # Paper Algorithm 1 starts with the k most recent workload observations.
        # A fresh runtime policy has no detector state, so initialize it from
        # the same measured history before the first forecast decision.
        if len(detector.values) == 0:
            detector.warm(values[-detector.values.maxlen:])
        if hasattr(forecast, "predict_interval"):
            low, predicted, high = map(float, forecast.predict_interval(values))
        else:
            predicted = float(forecast.predict(values))
            low = high = predicted
        burst = bool(detector.update(predicted))
        interval = (low, predicted, high) if current_policy["n2_quantile"] else None
        decision = provisioner.decide(predicted, burst=burst, interval=interval)
        monitoring_seconds = (interval_policy.update(
            observed, burst,
            (high - low) / max(abs(predicted), 1e-9) if current_policy["n2_quantile"] else 0.0,
        ) if current_policy["n3_adaptive_monitoring"] else fixed_interval)
        result = {
            "status": "ready", "observed_qps": observed, "predicted_qps": predicted,
            "p10": low, "p90": high, "burst": burst,
            "desired_replicas": int(decision.replicas),
            "desired_cpu_millicores": int(decision.cpu_millicores),
            "predicted_utilization": float(decision.utilization),
            "monitoring_interval_seconds": int(monitoring_seconds),
            "slo_ms": slo_ms, "policy": dict(current_policy),
            "performance_model": "measured_dtr", "actuation_enabled": bool(apply_changes),
        }
        if executor:
            result["execution"] = executor.execute(decision)
        latest = result
        recent.append({"timestamp": time.time(), **result})
        return result


def monitor_loop() -> None:
    url = os.getenv("PROMETHEUS_URL", "http://kube-prometheus-stack-prometheus.monitoring.svc:9090")
    service = os.getenv("SERVICE", "house-price")
    while True:
        delay = fixed_interval
        try:
            qps = prometheus_qps(url, namespace, service)
            result = process(qps)
            delay = int(result.get("monitoring_interval_seconds", fixed_interval))
            log.info("Prometheus observation/controller decision: %s", result)
        except Exception:
            log.exception("Prometheus/controller cycle failed")
        time.sleep(max(5, delay))


@app.on_event("startup")
def startup() -> None:
    if os.getenv("MONITOR_ENABLED", "true").lower() == "true":
        threading.Thread(target=monitor_loop, daemon=True, name="prometheus-monitor").start()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "performance_model": "measured_dtr"}


@app.get("/status")
def status() -> dict[str, Any]:
    with policy_lock:
        return {**latest, "policy": dict(current_policy), "actuation_enabled": bool(apply_changes),
                "performance_model": "measured_dtr", "recent": list(recent)[-20:]}


@app.post("/configure")
def configure(request: PolicyRequest) -> dict[str, Any]:
    payload = request.model_dump()
    initial_history = payload.pop("initial_history", None)
    _configure_components(payload)
    if initial_history is not None:
        if forecast is None or len(initial_history) != forecast.lookback:
            raise HTTPException(422, f"initial_history must contain exactly {forecast.lookback if forecast else 10} points")
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0
               for value in initial_history):
            raise HTTPException(422, "initial_history must contain non-negative numeric values")
        with policy_lock:
            history.extend(float(value) for value in initial_history)
    return {"status": "configured", "policy": dict(current_policy),
            "actuation_enabled": bool(apply_changes)}


@app.post("/observe/{qps}")
def observe(qps: float) -> dict[str, Any]:
    """Inject a measured QPS point for controller diagnostics; not an app metric."""
    return process(qps)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return {"status": "ok", "performance_model": "measured_dtr",
            "policy": dict(current_policy), "history_samples": len(history)}
