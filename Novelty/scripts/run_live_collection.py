"""Collect real Kubernetes/Prometheus performance rows using a stepped Locust load."""
from __future__ import annotations

import argparse
import urllib.request
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="Novelty/data/collected/performance_live.csv")
    parser.add_argument("--prometheus", default="http://127.0.0.1:9090")
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--cpu-levels", nargs="+", type=int, default=[600, 800, 950])
    parser.add_argument("--replica-levels", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--users", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--app-port", type=int, default=8000)
    args = parser.parse_args()
    out = str((ROOT / args.out).resolve())
    locustfile = str(ROOT / "Paper" / "locustfile.py")
    logdir = ROOT / "tmp"
    logdir.mkdir(exist_ok=True)

    # Use fixed replica counts for the calibration matrix so the model learns
    # how response/utilization change with replicas, rather than one HPA state.
    run(["kubectl", "delete", "hpa", "house-price", "-n", "hybrid-autoscaling", "--ignore-not-found=true"])

    for cpu in args.cpu_levels:
        run([
            "kubectl", "set", "resources", "deployment/house-price",
            "-n", "hybrid-autoscaling", "-c", "app",
            f"--requests=cpu={cpu}m,memory=128Mi",
            f"--limits=cpu={cpu}m,memory=256Mi",
        ])
        run(["kubectl", "rollout", "status", "deployment/house-price", "-n",
             "hybrid-autoscaling", "--timeout=120s"])
        for replicas in args.replica_levels:
            run(["kubectl", "scale", "deployment/house-price", "-n", "hybrid-autoscaling",
                 f"--replicas={replicas}"])
            run(["kubectl", "rollout", "status", "deployment/house-price", "-n",
                 "hybrid-autoscaling", "--timeout=120s"])
            pfout = (logdir / "app_portforward.out").open("w")
            pferr = (logdir / "app_portforward.err").open("w")
            portforward = subprocess.Popen([
                "kubectl", "-n", "hybrid-autoscaling", "port-forward",
                "service/house-price", f"{args.app_port}:80",
            ], cwd=ROOT, stdout=pfout, stderr=pferr)
            try:
                for _ in range(30):
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{args.app_port}/healthz", timeout=2) as response:
                            if response.status == 200:
                                break
                    except Exception:
                        time.sleep(1)
                else:
                    raise RuntimeError("App port-forward did not become healthy")
                for users in args.users:
                    print(f"Collecting CPU={cpu}m, replicas={replicas}, Locust users={users}", flush=True)
                    command = [
                        sys.executable, "-m", "locust", "-f", locustfile, "--headless",
                        "-H", f"http://127.0.0.1:{args.app_port}", "-u", str(users), "-r", str(users),
                        "-t", f"{args.duration}s", "--only-summary",
                    ]
                    with (logdir / "locust_campaign.out").open("w") as stdout, \
                         (logdir / "locust_campaign.err").open("w") as stderr:
                        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
                        try:
                            time.sleep(args.warmup)
                            run([
                                sys.executable, "Novelty/scripts/collect_performance_data.py",
                                "--prometheus", args.prometheus, "--cpu", str(cpu),
                                "--slo", "350", "--out", out,
                                "--samples", str(args.samples), "--interval", str(args.interval),
                            ])
                        finally:
                            try:
                                process.wait(timeout=max(1, args.duration - args.warmup + 30))
                            except subprocess.TimeoutExpired:
                                process.terminate()
                                process.wait(timeout=10)
                        if process.returncode not in (0, None):
                            raise RuntimeError(f"Locust failed; see {logdir / 'locust_campaign.err'}")
            finally:
                portforward.terminate()
                try:
                    portforward.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    portforward.kill()
                pfout.close()
                pferr.close()
    run(["kubectl", "autoscale", "deployment/house-price", "-n", "hybrid-autoscaling",
         "--cpu-percent=70", "--min=1", "--max=5"])
    print(f"Collection complete: {out}", flush=True)


if __name__ == "__main__":
    main()
