"""Replay a sampled World Cup/Azure profile against the deployed app with Locust.

Profiles contain ``time_seconds,target_qps``. The custom load shape maintains a
bounded number of closed-loop users and uses measured request time to pace them;
the actual achieved rate is always reported from Prometheus, never assumed to
equal the requested profile.
"""
from __future__ import annotations

import csv
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from locust import HttpUser, LoadTestShape, events, task


PROFILE_PATH = Path(os.environ["AUTOSCALER_TRACE_PROFILE"])
TICK_SECONDS = max(5, int(os.getenv("AUTOSCALER_TRACE_TICK_SECONDS", "15")))
EXPECTED_RESPONSE_SECONDS = max(.05, float(os.getenv("AUTOSCALER_EXPECTED_RESPONSE_SECONDS", ".12")))
MAX_USERS = max(1, int(os.getenv("AUTOSCALER_MAX_USERS", "30")))
REQUEST_LOG_PATH = os.getenv("LOCUST_REQUEST_LOG")
_request_log_lock = threading.Lock()


@events.request.add_listener
def record_request(request_type, name, response_time, response_length, exception,
                   context, start_time=None, url=None, response=None, **kwargs):
    """Persist one row per completed Locust request for exact request SLO rates."""
    if not REQUEST_LOG_PATH:
        return
    status_code = getattr(response, "status_code", "")
    failed = bool(exception) or (isinstance(status_code, int) and status_code >= 500)
    with _request_log_lock:
        path = Path(REQUEST_LOG_PATH)
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if new_file:
                writer.writerow(("timestamp", "method", "name", "response_time_ms",
                                 "status_code", "failed", "exception"))
            writer.writerow((datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                             request_type, name, f"{float(response_time):.6f}",
                             status_code, int(failed), str(exception or "")))

with PROFILE_PATH.open(newline="", encoding="utf-8") as stream:
    PROFILE = [
        (float(row["time_seconds"]), max(0.0, float(row["target_qps"])))
        for row in csv.DictReader(stream)
    ]
if not PROFILE:
    raise ValueError(f"Trace profile is empty: {PROFILE_PATH}")

TARGET_QPS = 0.0


class TraceReplayUser(HttpUser):
    host = os.getenv("LOCUST_HOST", "http://127.0.0.1:8000")

    def wait_time(self) -> float:
        rate = max(TARGET_QPS, 0.01)
        users = max(1, int(getattr(self.environment.runner, "user_count", 1)))
        desired_period = users / rate
        return max(0.0, desired_period - getattr(self, "last_request_seconds", .05))

    @task
    def predict(self) -> None:
        # Relative URL preserves the Kubernetes API service-proxy prefix used
        # by the live runner, while still resolving to /predict for a root URL.
        with self.client.get("predict?area=2500&rooms=4&age=8", name="/predict",
                             catch_response=True) as response:
            self.last_request_seconds = max(.001, response.elapsed.total_seconds())
            if response.status_code >= 500:
                response.failure(f"HTTP {response.status_code}")


class TraceReplayShape(LoadTestShape):
    """Piecewise-constant, chronological, bounded-QPS trace profile."""

    def tick(self):
        global TARGET_QPS
        run_time = self.get_run_time()
        index = int(run_time // TICK_SECONDS)
        if index >= len(PROFILE):
            return None
        TARGET_QPS = PROFILE[index][1]
        if TARGET_QPS <= 0:
            return (0, 1)
        users = min(MAX_USERS, max(1, int(math.ceil(TARGET_QPS * EXPECTED_RESPONSE_SECONDS))))
        # A bounded step avoids churn when adjacent trace points are similar.
        spawn_rate = max(1, int(math.ceil(users / min(TICK_SECONDS, 5))))
        return users, spawn_rate
