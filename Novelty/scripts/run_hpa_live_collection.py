"""Collect HPA-decided replica counts under stepped live Locust traffic."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="Novelty/data/collected/performance_live_hpa.csv")
    parser.add_argument("--prometheus", default="http://127.0.0.1:9090")
    parser.add_argument("--cpu-levels", nargs="+", type=int, default=[600, 800, 950])
    parser.add_argument("--users", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--wait-ranges", nargs="+", default=None,
                        help="Optional Locust min:max wait ranges in seconds, e.g. 0.1:0.2 0.5:1 1:2")
    parser.add_argument("--duration", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--app-port", type=int, default=8000)
    args = parser.parse_args()

    out = str((ROOT / args.out).resolve())
    (ROOT / "Novelty" / "data" / "collected").mkdir(parents=True, exist_ok=True)
    logdir = ROOT / "tmp"
    logdir.mkdir(exist_ok=True)
    locustfile = str(ROOT / ("Novelty/locustfile_calibration.py" if args.wait_ranges else "Paper/locustfile.py"))

    for cpu in args.cpu_levels:
        # A fresh HPA per CPU tier resets scaling history; ascending load stages
        # let the HPA's actual scale-up policy determine each replica label.
        run(["kubectl", "delete", "hpa", "house-price", "-n", "hybrid-autoscaling", "--ignore-not-found=true"])
        run(["kubectl", "scale", "deployment/house-price", "-n", "hybrid-autoscaling", "--replicas=1"])
        run([
            "kubectl", "set", "resources", "deployment/house-price",
            "-n", "hybrid-autoscaling", "-c", "app",
            f"--requests=cpu={cpu}m,memory=128Mi",
            f"--limits=cpu={cpu}m,memory=256Mi",
        ])
        run(["kubectl", "rollout", "status", "deployment/house-price", "-n",
             "hybrid-autoscaling", "--timeout=120s"])
        run(["kubectl", "autoscale", "deployment/house-price", "-n", "hybrid-autoscaling",
             "--cpu-percent=70", "--min=1", "--max=5"])

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

            stages = [(1, None)] if args.wait_ranges else [(user, None) for user in args.users]
            if args.wait_ranges:
                stages = []
                for wait_range in args.wait_ranges:
                    parts = wait_range.split(":", maxsplit=1)
                    if len(parts) != 2:
                        raise ValueError(f"Invalid wait range {wait_range!r}; expected min:max seconds")
                    low, high = map(float, parts)
                    if low <= 0 or high < low:
                        raise ValueError(f"Invalid wait range {wait_range!r}")
                    stages.append((1, (low, high)))
            for users, wait_range in stages:
                label = f"users={users}" if wait_range is None else f"users={users}, wait={wait_range[0]}:{wait_range[1]}s"
                print(f"HPA data collection: CPU={cpu}m, {label}", flush=True)
                command = [
                    sys.executable, "-m", "locust", "-f", locustfile, "--headless",
                    "-H", f"http://127.0.0.1:{args.app_port}", "-u", str(users), "-r", str(users),
                    "-t", f"{args.duration}s", "--only-summary",
                ]
                env = None
                if wait_range is not None:
                    import os
                    env = os.environ.copy()
                    env["LOCUST_WAIT_MIN"], env["LOCUST_WAIT_MAX"] = map(str, wait_range)
                with (logdir / "locust_campaign.out").open("w") as stdout, \
                     (logdir / "locust_campaign.err").open("w") as stderr:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, env=env)
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

    print(f"HPA collection complete: {out}", flush=True)


if __name__ == "__main__":
    main()
