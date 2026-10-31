"""Run a matched Paper Hybrid + 15-arm novelty factorial on live Kubernetes.

Start Prometheus/controller/app port-forwards first and run the dashboard. Every
arm drives a trace-shaped Locust profile, uses the measured DTR in the in-cluster
controller, and records Prometheus samples separately from simulator outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
ALL_POLICIES = ["paper_hybrid", "N1", "N2", "N3", "N4", "N1+N2", "N1+N3", "N1+N4",
                "N2+N3", "N2+N4", "N3+N4", "N1+N2+N3", "N1+N2+N4",
                "N1+N3+N4", "N2+N3+N4", "N1+N2+N3+N4"]


def api(base: str, path: str, body: dict | None = None, timeout: int = 180) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = Request(base.rstrip("/") + path, data=data, method="POST" if data else "GET",
                      headers={"Content-Type": "application/json"} if data else {})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(f"Dashboard request {path} failed: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", default="http://127.0.0.1:8000")
    parser.add_argument("--traces", nargs="+", choices=("worldcup98", "azurefunctions2019"),
                        default=["worldcup98", "azurefunctions2019"])
    parser.add_argument("--policies", nargs="+", choices=ALL_POLICIES, default=ALL_POLICIES)
    parser.add_argument("--max-points", type=int, default=75,
                        help="Number of contiguous one-minute test points to replay")
    parser.add_argument("--segment-start", type=int, default=None,
                        help="Absolute source-row index for the first held-out point; requires a matching segment-trained forecaster")
    parser.add_argument("--segment-train-points", type=int, default=300,
                        help="Development points immediately before --segment-start")
    parser.add_argument("--paper-worldcup-first-interval", type=int, default=1,
                        help="One-based interval number represented by the first WorldCup point; paper burst zone is intervals 27-54")
    parser.add_argument("--tick-seconds", type=int, default=60,
                        help="Locust profile duration per point; 60 s matches the default controller cadence")
    parser.add_argument("--max-qps", type=float, default=20.0,
                        help="Scale selected development-segment peak to this measured-DTR workload (QPS)")
    parser.add_argument("--max-users", type=int, default=30)
    parser.add_argument("--apply-changes", action="store_true",
                        help="Allow the controller to change the benchmark Deployment during each arm")
    parser.add_argument("--dry-run", action="store_true", help="Write the experiment plan without starting workloads")
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()
    if args.tick_seconds < 15:
        raise SystemExit("Use --tick-seconds >= 15 to align with the Prometheus scrape interval.")
    if not 5 <= args.max_points <= 1440:
        raise SystemExit("--max-points must be between 5 and 1440.")

    suite_id = datetime.now().strftime("suite_%Y%m%d_%H%M%S")
    suite_dir = ROOT / "Novelty" / "results" / "live_kubernetes" / suite_id
    suite_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "suite_id": suite_id, "measurement_type": "live_kubernetes_locust_prometheus",
        "traces": args.traces, "policies": args.policies,
        "max_points": args.max_points, "tick_seconds": args.tick_seconds,
        "segment_start": args.segment_start,
        "segment_train_points": args.segment_train_points,
        "paper_worldcup_burst_intervals_1based": [27, 54],
        "paper_worldcup_first_interval": args.paper_worldcup_first_interval,
        "max_qps": args.max_qps, "max_users": args.max_users,
        "apply_changes": args.apply_changes,
        "arms": len(args.traces) * len(args.policies),
        "profile_seconds_per_arm": args.max_points * args.tick_seconds,
        "expected_total_hours": round(len(args.traces) * len(args.policies) *
                                       args.max_points * args.tick_seconds / 3600, 2),
    }
    (suite_dir / "suite_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (suite_dir / "suite_metadata.json").write_text(
        json.dumps({**plan, "status": "running", "completed_arms": 0}, indent=2),
        encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    rows = []

    def handle_interrupt(_signum, _frame):
        try:
            api(args.dashboard, "/api/live/stop", {})
        except Exception:
            pass
        (suite_dir / "suite_metadata.json").write_text(json.dumps({
            **plan, "status": "aborted", "completed_arms": len(rows),
            "abort_reason": "Interrupted by the operator; the active Locust arm was stopped.",
        }, indent=2), encoding="utf-8")
        print("Suite aborted cleanly; partial arms remain marked non-comparable.", flush=True)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handle_interrupt)
    for trace in args.traces:
        for policy in args.policies:
            print(f"Starting {trace} / {policy} ...", flush=True)
            try:
                started = api(args.dashboard, "/api/live/start", {
                    "trace": trace, "policy": policy, "max_points": args.max_points,
                    "tick_seconds": args.tick_seconds, "max_qps": args.max_qps,
                    "max_users": args.max_users, "apply_changes": args.apply_changes,
                    "segment_start": args.segment_start,
                    "segment_train_points": args.segment_train_points,
                })
            except Exception as exc:
                (suite_dir / "suite_metadata.json").write_text(json.dumps({
                    **plan, "status": "aborted", "completed_arms": len(rows),
                    "abort_reason": f"Could not start {trace}/{policy}: {exc}",
                }, indent=2), encoding="utf-8")
                raise
            run_id = started["run_id"]
            deadline = time.time() + started["duration_seconds"] + 180
            while time.time() < deadline:
                state = api(args.dashboard, "/api/live/status")
                if state.get("run_id") == run_id and state.get("state") in ("completed", "failed", "stopped"):
                    break
                if state.get("run_id") != run_id and state.get("state") == "telemetry_error":
                    print("Telemetry collection error:", state.get("error"), flush=True)
                time.sleep(max(2, args.poll_seconds))
            else:
                api(args.dashboard, "/api/live/stop", {})
                raise TimeoutError(f"Live arm {run_id} did not finish within expected run time")
            run_dir = Path(started["run_dir"])
            sample_file = run_dir / "prometheus_samples.csv"
            samples = list(csv.DictReader(sample_file.open(encoding="utf-8"))) if sample_file.exists() else []
            numeric = lambda key: [float(row[key]) for row in samples
                                   if row.get(key) not in (None, "")]
            p95s = numeric("request_p95_ms")
            qps_values = numeric("request_rate_qps")
            cpu_values = numeric("cpu_utilization_pct")
            slo_values = numeric("request_slo_violation_pct")
            request_events_file = run_dir / "locust_request_events.csv"
            request_events = (list(csv.DictReader(request_events_file.open(encoding="utf-8")))
                              if request_events_file.exists() else [])
            event_failures = sum(row.get("failed") == "1" for row in request_events)
            transport_errors = sum(
                bool(row.get("exception")) and row.get("status_code") in (None, "", "0")
                for row in request_events)
            event_violations = sum(
                row.get("failed") == "1" or
                (row.get("response_time_ms") not in (None, "") and
                 float(row["response_time_ms"]) > 350.0)
                for row in request_events
            )
            # The paper reports interval-based outcomes in the burst detector's
            # ON/OFF zones. Preserve per-request rates as a separate measure.
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            run_origin = datetime.fromisoformat(metadata["started_at"].replace("Z", "+00:00"))

            def tick_index(timestamp):
                return int((timestamp - run_origin).total_seconds() // args.tick_seconds)

            timed_events = []
            for event in request_events:
                try:
                    timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                    timed_events.append((tick_index(timestamp), event))
                except (KeyError, TypeError, ValueError):
                    continue
            interval_events: dict[int, list[dict]] = {}
            for index, event in timed_events:
                if 0 <= index < args.max_points:
                    interval_events.setdefault(index, []).append(event)
            interval_violations = {}
            for index, events in interval_events.items():
                latencies = [float(event["response_time_ms"]) for event in events
                             if event.get("response_time_ms") not in (None, "")]
                failed = any(event.get("failed") == "1" for event in events)
                interval_violations[index] = bool(
                    failed or (latencies and sum(latencies) / len(latencies) > 350.0))
            observed_intervals = set(interval_violations)
            burst_samples: dict[int, tuple[datetime, bool]] = {}
            utilization_samples: dict[int, tuple[datetime, float]] = {}
            for row in samples:
                if not row.get("timestamp"):
                    continue
                try:
                    timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                    index = tick_index(timestamp)
                    if not 0 <= index < args.max_points:
                        continue
                    raw_burst = str(row.get("controller_burst", "")).lower()
                    if row.get("controller_status") == "ready" and raw_burst in ("true", "false"):
                        prior = burst_samples.get(index)
                        if prior is None or timestamp > prior[0]:
                            burst_samples[index] = (timestamp, raw_burst == "true")
                    raw_util = row.get("cpu_utilization_pct")
                    if raw_util not in (None, ""):
                        prior = utilization_samples.get(index)
                        if prior is None or timestamp > prior[0]:
                            utilization_samples[index] = (timestamp, float(raw_util))
                except (TypeError, ValueError):
                    continue
            # Apply the paper's fixed FIFA burst zone (intervals 27-54) to
            # every policy. Do not let each arm's own burst detector define
            # the evaluation cohorts; that would make the comparison biased.
            paper_intervals = {
                i: args.paper_worldcup_first_interval + i
                for i in range(args.max_points)
            }
            burst_on = {i for i, paper_i in paper_intervals.items()
                        if 27 <= paper_i <= 54}
            burst_off = set(paper_intervals) - burst_on

            def interval_violation_pct(indexes):
                covered = observed_intervals & indexes
                return (100.0 * sum(interval_violations[i] for i in covered) / len(covered)
                        if covered else None)

            def mean_utilization(indexes):
                values = [utilization_samples[i][1] for i in indexes & set(utilization_samples)]
                return sum(values) / len(values) if values else None

            decision_periods = sum(row.get("controller_status") == "ready" for row in samples)
            # Integrate the actual CPU requested by all matching app pods over
            # Prometheus sample timestamps. This is a measured resource-cost
            # proxy (CPU-core-seconds), including rolling-update overlap.
            requested = [(datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).timestamp(),
                          float(row["cpu_requested_cores"]))
                         for row in samples
                         if row.get("timestamp") and row.get("cpu_requested_cores") not in (None, "")]
            cpu_core_seconds = sum(
                max(0.0, min(30.0, requested[i][0] - requested[i - 1][0])) *
                (requested[i][1] + requested[i - 1][1]) / 2.0
                for i in range(1, len(requested)))
            requested_core_mean = (sum(cores for _, cores in requested) / len(requested)
                                   if requested else None)
            result = {
                "run_id": run_id, "trace": trace, "policy": policy,
                "state": state.get("state"), "apply_changes": args.apply_changes,
                "profile_points": args.max_points, "locust_tick_seconds": args.tick_seconds,
                "prometheus_samples": len(samples),
                "controller_ready_sample_periods": decision_periods,
                "locust_completed_requests": len(request_events),
                "locust_failed_requests": event_failures,
                "locust_transport_errors": transport_errors,
                "request_slo_violations": event_violations,
                "request_slo_violation_pct": (100 * event_violations / len(request_events)
                                               if request_events else None),
                "request_slo_measurement_source": "Locust per-request response time; failure counts as a violation",
                "paper_interval_slo_violation_pct": interval_violation_pct(set(range(args.max_points))),
                "paper_fifa_burst_on_slo_violation_pct": (
                    interval_violation_pct(burst_on) if trace == "worldcup98" else None),
                "paper_fifa_burst_off_slo_violation_pct": (
                    interval_violation_pct(burst_off) if trace == "worldcup98" else None),
                "burst_on_interval_count": len(observed_intervals & burst_on),
                "burst_off_interval_count": len(observed_intervals & burst_off),
                "burst_on_profile_interval_count": len(burst_on),
                "burst_off_profile_interval_count": len(burst_off),
                "mean_cpu_utilization_burst_on_pct": mean_utilization(burst_on),
                "mean_cpu_utilization_burst_off_pct": mean_utilization(burst_off),
                "paper_interval_slo_definition": (
                    "60-second mean response time > 350 ms or any failed request; FIFA ON is fixed paper intervals 27-54 inclusive, all other selected intervals are OFF"),
                "paper_interval_numbering": "one-based interval numbers from the paper workload timeline",
                "paper_first_interval_number": args.paper_worldcup_first_interval if trace == "worldcup98" else None,
                "paper_burst_on_interval_numbers": sorted(paper_intervals[i] for i in burst_on) if trace == "worldcup98" else [],
                "paper_burst_off_interval_numbers": sorted(paper_intervals[i] for i in burst_off) if trace == "worldcup98" else [],
                "paper_interval_count_observed": len(observed_intervals),
                "matched_policy_comparison_valid": (args.max_points >= 30 and
                                                     state.get("state") == "completed" and
                                                     decision_periods > 0 and bool(request_events) and
                                                     transport_errors == 0),
                "paper_burst_zone_fully_covered": (
                    trace == "worldcup98" and args.paper_worldcup_first_interval <= 27 and
                    args.paper_worldcup_first_interval + args.max_points - 1 >= 54),
                "paper_fifa_figure_window_approx_intervals": 72 if trace == "worldcup98" else None,
                "measured_cpu_request_core_seconds": cpu_core_seconds,
                "mean_requested_cpu_cores": requested_core_mean,
                "resource_cost_definition": "Prometheus CPU requests integrated over sampled time; includes overlapping matching pods",
                "mean_measured_qps": sum(qps_values) / len(qps_values) if qps_values else None,
                "mean_request_p95_ms": sum(p95s) / len(p95s) if p95s else None,
                "max_request_p95_ms": max(p95s) if p95s else None,
                "fraction_sampled_periods_over_slo_pct": (100 * sum(v > 350 for v in p95s) / len(p95s)
                                                            if p95s else None),
                "mean_prometheus_rolling_slo_violation_pct": sum(slo_values) / len(slo_values) if slo_values else None,
                "mean_cpu_utilization_pct": sum(cpu_values) / len(cpu_values) if cpu_values else None,
                "run_directory": str(run_dir.relative_to(ROOT)),
            }
            rows.append(result)
            with (suite_dir / "suite_summary.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (suite_dir / "suite_metadata.json").write_text(json.dumps({
                **plan, "status": "running", "completed_arms": len(rows),
                "current_arm": f"{trace}/{policy}",
            }, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)

    (suite_dir / "suite_metadata.json").write_text(json.dumps({**plan, "status": "completed"}, indent=2),
                                                    encoding="utf-8")
    embedded_rows = json.dumps(rows).replace("</", "<\\/")
    report = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Live Kubernetes factorial results</title><script src='https://cdn.jsdelivr.net/npm/chart.js'></script><style>body{{font:15px system-ui;background:#0b1220;color:#e2e8f0;margin:2rem}}section{{background:#111c30;padding:1rem;margin:1rem 0;border-radius:12px;height:320px}}canvas{{max-height:260px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.45rem;border-bottom:1px solid #334155;text-align:left}}.note{{color:#fbbf24}}</style></head><body><h1>Live Kubernetes / Locust / Prometheus results</h1><p class='note'>These are measured benchmark observations. CPU request is a resource proxy, not cloud billing. One repetition per policy is descriptive; do not claim statistical significance from this suite.</p><label>Trace: <select id='trace' onchange='draw()'><option value='worldcup98'>FIFA World Cup 1998</option><option value='azurefunctions2019'>Azure Functions 2019</option></select></label><section><canvas id='p95'></canvas></section><section><canvas id='slo'></canvas></section><section><canvas id='cpu'></canvas></section><h2>Raw summary</h2><table id='table'></table><script>const rows={embedded_rows};const keys=[['p95','Mean measured request P95 (ms)','mean_request_p95_ms'],['slo','Request-based SLO violation (%)','request_slo_violation_pct'],['cpu','Mean container CPU utilization (%)','mean_cpu_utilization_pct']];const charts={{}};function draw(){{const trace=document.getElementById('trace').value;const data=rows.filter(r=>r.trace===trace);for(const [id,title,key] of keys){{if(charts[id])charts[id].destroy();charts[id]=new Chart(document.getElementById(id),{{type:'bar',data:{{labels:data.map(r=>r.policy),datasets:[{{label:title,data:data.map(r=>r[key]),backgroundColor:'#38bdf8'}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{title:{{display:true,text:title,color:'#e2e8f0'}}}},scales:{{x:{{ticks:{{color:'#cbd5e1'}}}},y:{{beginAtZero:true,ticks:{{color:'#cbd5e1'}}}}}}}}}})}}const cols=Object.keys(rows[0]||{{}});document.getElementById('table').innerHTML='<tr>'+cols.map(k=>'<th>'+k+'</th>').join('')+'</tr>'+data.map(r=>'<tr>'+cols.map(k=>'<td>'+((r[k]??'—'))+'</td>').join('')+'</tr>').join('')}}draw();</script></body></html>"""
    (suite_dir / "suite_report.html").write_text(report, encoding="utf-8")
    print(f"Live experiment complete: {suite_dir / 'suite_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
