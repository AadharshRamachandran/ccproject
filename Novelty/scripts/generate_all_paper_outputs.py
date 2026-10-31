"""Produce the paper's tables and figures plus novelty overlays.

Uses chronological trace windows and a synthetic development performance
fixture. Reported latency and cost are emulator estimates, not cluster data.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Paper"))
sys.path.insert(0, str(ROOT / "Novelty" / "scripts"))

from reproduce_paper_results import (  # noqa: E402
    benchmark_performance,
    classic_forecasts,
    recurrent_forecasts,
    write_table,
)

PAPER = "paper_hybrid_bilstm_dtr_rolling_fixed_monitor"
ALL_FOUR = "optimized_novelty_n1_burst_n2_quantile_n3_monitoring_n4_inplace"
DISPLAY = {
    "reactive_hpa_fixed_cpu": "Reactive HPA",
    "proactive_hpa_bilstm_fixed_cpu": "Proactive HPA",
    "burst_aware_hpa_bilstm_fixed_cpu": "Burst-aware HPA",
    PAPER: "Paper Hybrid",
    "novelty_1_tuned_adaptive_burst": "N1 Adaptive burst",
    "novelty_2_quantile_provisioning": "N2 Quantile provision",
    "novelty_3_adaptive_monitoring": "N3 Adaptive monitor",
    "novelty_4_inplace_resize": "N4 In-place resize",
    ALL_FOUR: "All four novelties",
}


def extract_burst_window(values: np.ndarray, n: int = 2400, test_frac: float = 0.2) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = min(n, len(values))
    peak = int(np.argmax(values))
    # Place the peak just inside the held-out 20% so Algorithm 1 can fire on
    # the test burst while the training peak stays close to the scaled 1800 rps
    # DTR envelope.
    peak_at = int(n * (1 - test_frac) + 0.12 * n * test_frac)
    start = int(np.clip(peak - peak_at, 0, len(values) - n))
    return values[start:start + n]


def extract_tail_window(values: np.ndarray, n: int = 7200) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = min(n, len(values))
    return values[-n:]


def load_rate(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    return pd.to_numeric(frame.iloc[:, -1], errors="coerce").dropna().to_numpy(dtype=float)


PAPER_TABLE1 = [
    ("Wikipedia", "ARIMA", 0.985, 0.699), ("Wikipedia", "LR", 0.872, 0.643),
    ("Wikipedia", "EN", 0.778, 0.589), ("Wikipedia", "OLS", 0.921, 0.730),
    ("Wikipedia", "NNLS", 0.879, 0.620), ("Wikipedia", "LSTM", 0.742, 0.542),
    ("Wikipedia", "Bi-LSTM", 0.655, 0.466),
    ("FIFA World Cup", "ARIMA", 0.875, 0.523), ("FIFA World Cup", "LR", 0.709, 0.623),
    ("FIFA World Cup", "EN", 0.636, 0.419), ("FIFA World Cup", "OLS", 0.910, 0.660),
    ("FIFA World Cup", "NNLS", 0.880, 0.653), ("FIFA World Cup", "LSTM", 0.509, 0.323),
    ("FIFA World Cup", "Bi-LSTM", 0.436, 0.289),
]
PAPER_TABLE2 = [
    ("LR", 0.836, 3.27), ("RDF", 0.984, 3.51), ("SVR", 0.523, 2.98),
    ("XGB", 0.268, 2.67), ("MLP", 0.732, 3.14), ("DTR", 0.109, 1.81),
]
PAPER_TABLE3 = [
    ("Time interval T", "60 s"),
    ("k last observations (lookback)", "10"),
    ("Burst threshold", "5"),
    ("Burst influence", "0.5"),
    ("Response-time SLO θ", "350 ms"),
    ("R_max (CPU / pod)", "950 millicores"),
    ("R_min (CPU / pod)", "600 millicores"),
]
PAPER_TABLE4 = [
    ("Wikipedia", "ON", 22.14, 2.87, 2.96),
    ("Wikipedia", "OFF", 8.63, 1.14, 1.53),
    ("FIFA World Cup", "ON", 20.54, 4.10, 3.89),
    ("FIFA World Cup", "OFF", 5.47, 1.57, 1.86),
]
PAPER_TABLE5 = [
    ("Wikipedia", "ON", 76.68, 73.51, 73.73),
    ("Wikipedia", "OFF", 61.78, 55.26, 69.65),
    ("FIFA World Cup", "ON", 78.30, 72.68, 72.58),
    ("FIFA World Cup", "OFF", 64.58, 63.96, 74.17),
]
PAPER_TABLE6 = [
    ("Wikipedia", 3.11, 0.793, 2.96, 0.608),
    ("FIFA World Cup", 4.05, 0.809, 3.89, 0.612),
]


def add_unit_interval_errors(forecast: pd.DataFrame, train: np.ndarray) -> pd.DataFrame:
    """Paper Table I reports RMSE/MAE after min-max scaling the trace to (0, 1)."""
    span = max(float(np.max(train) - np.min(train)), 1e-9)
    forecast = forecast.copy()
    forecast["rmse_unit_interval"] = forecast["rmse"] / span
    forecast["mae_unit_interval"] = forecast["mae"] / span
    return forecast


def paper_zone_table(zones: pd.DataFrame, metric: str, variants: list[str]) -> pd.DataFrame:
    rows = []
    for zone in ("burst", "non_burst"):
        row = {"zone": "ON" if zone == "burst" else "OFF"}
        for variant in variants:
            match = zones[(zones.variant == variant) & (zones.zone == zone)]
            row[DISPLAY.get(variant, variant)] = float(match[metric].iloc[0]) if len(match) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def figure_table3(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.axis("off")
    ax.set_title("Table III. Parameter setting (paper defaults used by Paper Hybrid)", loc="left")
    table = ax.table(cellText=PAPER_TABLE3, colLabels=["Parameter", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.55)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def figure_algorithms(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4))
    texts = [
        ("Algorithm 1 — Online burst identification",
         "Require: last k predictions Y, threshold, influence\n"
         "avg, std ← mean/std(Y)\n"
         "for each new prediction x̂:\n"
         "  if (x̂ − avg) > threshold · std:\n"
         "      burst ← ON; x ← influence·x̂ + (1−influence)·Y[k]\n"
         "  else:\n"
         "      burst ← OFF; x ← x̂\n"
         "  update rolling avg/std with x; slide window\n"
         "N1: threshold, influence, and volatility gain\n"
         "are chosen on validation only."),
        ("Algorithm 2 — Hybrid autoscaling",
         "Require: SLO θ=350 ms, R=[600…950], DTR Φ, burst\n"
         "x̂ ← Bi-LSTM (paper) or Quantile Bi-LSTM (N2)\n"
         "if burst ON:\n"
         "  (replicas, util) ← Φ(x̂, R_max, θ)\n"
         "else:\n"
         "  pick r in R maximizing predicted utilization\n"
         "  subject to Φ(x̂, r, θ) meeting the SLO\n"
         "if (replicas, CPU) changed:\n"
         "  rolling update (paper) or in-place resize (N4)\n"
         "wait T=60 s, or adaptive 15–240 s (N3)"),
    ]
    for ax, (title, body) in zip(axes, texts):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=11)
        ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.04), 0.96, 0.88, boxstyle="round,pad=0.02",
                                             facecolor="#f8fafc", edgecolor="#1f2937"))
        ax.text(0.06, 0.86, body, ha="left", va="top", fontsize=8.7, family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def figure_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Figure 1. Proposed hybrid autoscaler with novelty extensions", loc="left", fontsize=12, pad=8)

    def box(x, y, w, h, text, color):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                             facecolor=color, edgecolor="#1f2937", linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)

    box(0.3, 4.6, 2.4, 1.1, "Users / Locust\nworkload", "#e5e7eb")
    box(3.0, 4.6, 2.6, 1.1, "Ingress +\nPrometheus", "#dbeafe")
    box(6.0, 4.6, 2.8, 1.1, "Workload forecast\nBi-LSTM / Quantile Bi-LSTM (N2)", "#bfdbfe")
    box(9.3, 4.6, 2.8, 1.1, "Burst identification\nAlg. 1 / tuned adaptive (N1)", "#93c5fd")
    box(3.0, 2.7, 3.2, 1.2, "Hybrid scaler (Alg. 2)\nDTR performance model", "#fde68a")
    box(6.6, 2.7, 2.8, 1.2, "Adaptive monitoring (N3)\n15–240 s cadence", "#fcd34d")
    box(3.0, 0.7, 3.0, 1.2, "Rolling update\n(paper execution)", "#fecaca")
    box(6.4, 0.7, 3.2, 1.2, "In-place resize (N4)\nno overlap cost", "#86efac")
    box(10.0, 1.4, 2.2, 1.6, "Kubernetes\nDeployment", "#e5e7eb")
    for a, b in [((2.7, 5.15), (3.0, 5.15)), ((5.6, 5.15), (6.0, 5.15)),
                 ((8.8, 5.15), (9.3, 5.15)), ((7.4, 4.6), (4.6, 3.9)),
                 ((10.7, 4.6), (8.0, 3.9)), ((6.2, 2.7), (6.6, 3.3)),
                 ((4.6, 2.7), (4.5, 1.9)), ((8.0, 2.7), (8.0, 1.9)),
                 ((6.0, 1.3), (10.0, 2.1))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#111827", lw=1.2))
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def figure_rolling(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Figure 2. Hybrid execution: rolling update vs in-place resize (N4)", loc="left", fontsize=12)
    ax.add_patch(mpatches.Rectangle((0.4, 2.8), 3.2, 1.4, facecolor="#fecaca", edgecolor="#7f1d1d"))
    ax.text(2.0, 3.5, "Old deployment\n2 pods × 950 mCPU", ha="center", va="center", fontsize=9)
    ax.add_patch(mpatches.Rectangle((4.4, 2.8), 3.2, 1.4, facecolor="#fef3c7", edgecolor="#92400e"))
    ax.text(6.0, 3.5, "Overlap window t\n(Eq. 4 extra cost)", ha="center", va="center", fontsize=9)
    ax.add_patch(mpatches.Rectangle((8.4, 2.8), 3.2, 1.4, facecolor="#bbf7d0", edgecolor="#14532d"))
    ax.text(10.0, 3.5, "New deployment\n3 pods × 800 mCPU", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(4.4, 3.5), xytext=(3.6, 3.5), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.annotate("", xy=(8.4, 3.5), xytext=(7.6, 3.5), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.add_patch(mpatches.Rectangle((0.4, 0.6), 11.2, 1.5, facecolor="#dcfce7", edgecolor="#166534"))
    ax.text(6.0, 1.35, "N4 in-place resize patches pod CPU in place: no second ReplicaSet and no Eq. 4 overlap.",
            ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_ablation(workload: Path, out: Path, args) -> None:
    import os
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "Paper"), env.get("PYTHONPATH", "")])
    cmd = [
        sys.executable, str(ROOT / "Novelty" / "ablation.py"),
        "--workload", str(workload),
        "--allow-synthetic",
        "--out", str(out),
        "--max-points", "0",
        "--capacity-peak", str(args.capacity_peak),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--skip-component-benchmark",
        "--seed", str(args.seed),
    ]
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def policy_plots(trace: str, result_dir: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(result_dir / "summary.csv")
    skip = {"summary.csv", "pairwise_vs_paper.csv", "novelty_effects_vs_paper.csv", "component_benchmark.csv"}
    frames = {p.stem: pd.read_csv(p) for p in result_dir.glob("*.csv") if p.name not in skip}
    paper = frames[PAPER].sort_values("tick").reset_index(drop=True)
    burst = paper.burst.fillna(False).astype(bool).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(paper.tick, paper.scaled_demand, label="Actual demand", lw=1.2)
    ax.plot(paper.tick, paper.predicted_demand, label="Bi-LSTM forecast", lw=1.1)
    ax.fill_between(paper.tick, 0, paper.scaled_demand.max(), where=burst,
                    color="tab:red", alpha=0.15, label="Paper burst ON")
    ax.set(xlabel="Held-out interval (1 min)", ylabel="Requests / s (scaled)",
           title=f"{trace}: workload prediction and burst identification")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{trace}_fig3_4_forecast_burst.png", dpi=170)
    plt.close(fig)

    compare = [v for v in [
        "reactive_hpa_fixed_cpu", "proactive_hpa_bilstm_fixed_cpu",
        "burst_aware_hpa_bilstm_fixed_cpu", PAPER, ALL_FOUR,
    ] if v in frames]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9.2), sharex=True)
    fields = [("cpu_millicores", "CPU / pod (millicores)"), ("replicas", "Replicas"),
              ("utilization", "Utilization"), ("response_ms", "Response time (ms)")]
    for ax, (field, ylabel) in zip(axes, fields):
        for variant in compare:
            f = frames[variant].sort_values("tick")
            ax.plot(f.tick, f[field], label=DISPLAY.get(variant, variant), lw=1.05)
        if field == "response_ms":
            ax.axhline(350, color="crimson", ls="--", lw=0.9, label="SLO 350 ms")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Held-out interval (1 min)")
    fig.suptitle(f"{trace}: resource configuration and application performance")
    fig.tight_layout()
    fig.savefig(out / f"{trace}_fig5_6_scaling.png", dpi=170)
    plt.close(fig)

    order = [v for v in DISPLAY if v in set(summary.variant)]
    plot = summary.set_index("variant").loc[order]
    colors = ["#9ca3af" if "hpa" in v else "#2563eb" if v == PAPER else "#16a34a" for v in order]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.bar(range(len(plot)), plot.resource_cost, color=colors)
    ax.set_xticks(range(len(plot)), [DISPLAY[v].replace(" ", "\n") for v in order], fontsize=8)
    ax.set_ylabel("Resource cost (Eq. 2–4 units)")
    ax.set_title(f"{trace}: resource cost (Figure 7 analogue)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{trace}_fig7_cost.png", dpi=170)
    plt.close(fig)

    reactive = float(plot.loc["reactive_hpa_fixed_cpu", "resource_cost"]) if "reactive_hpa_fixed_cpu" in plot.index else float(plot.resource_cost.max())
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.bar(range(len(plot)), plot.resource_cost / max(reactive, 1e-9), color=colors)
    ax.set_xticks(range(len(plot)), [DISPLAY[v].replace(" ", "\n") for v in order], fontsize=8)
    ax.set_ylabel("Normalized resource cost (Reactive HPA = 1)")
    ax.set_title(f"{trace}: Figure 7 normalized resource cost")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{trace}_fig7_cost_normalized.png", dpi=170)
    plt.close(fig)

    novelty_order = [v for v in [PAPER,
                                 "novelty_1_tuned_adaptive_burst", "novelty_2_quantile_provisioning",
                                 "novelty_3_adaptive_monitoring", "novelty_4_inplace_resize", ALL_FOUR]
                     if v in plot.index]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    for ax, field, ylabel in zip(
        axes,
        ["slo_violation_pct", "mean_utilization_pct", "resource_cost"],
        ["SLO violation (%)", "Mean utilization (%)", "Resource cost"],
    ):
        vals = plot.loc[novelty_order, field]
        ax.bar(range(len(vals)), vals, color=["#2563eb" if v == PAPER else "#16a34a" for v in novelty_order])
        ax.set_xticks(range(len(vals)), [DISPLAY[v].replace(" ", "\n") for v in novelty_order], fontsize=7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{trace}: Paper Hybrid vs isolated and combined novelties")
    fig.tight_layout()
    fig.savefig(out / f"{trace}_novelty_vs_hybrid.png", dpi=170)
    plt.close(fig)

    zone_rows, burst_rows = [], []
    for variant, frame in frames.items():
        frame = frame.sort_values("tick").reset_index(drop=True)
        n = min(len(frame), len(burst))
        for zone, mask in (("burst", burst[:n]), ("non_burst", ~burst[:n])):
            subset = frame.iloc[:n].loc[mask]
            if subset.empty:
                continue
            zone_rows.append({
                "trace": trace, "variant": variant, "display": DISPLAY.get(variant, variant),
                "zone": zone, "intervals": len(subset),
                "slo_violation_pct": 100 * float(subset.slo_violation.mean()),
                "mean_utilization_pct": 100 * float(np.clip(subset.utilization, 0, 1).mean()),
                "resource_cost": float(subset.resource_cost.sum()),
                "mean_response_ms": float(subset.response_ms.mean()),
            })
        burst_subset = frame.iloc[:n].loc[burst[:n]]
        burst_rows.append({
            "trace": trace, "variant": variant, "display": DISPLAY.get(variant, variant),
            "burst_intervals": len(burst_subset),
            "burst_slo_violation_pct": 100 * float(burst_subset.slo_violation.mean()) if len(burst_subset) else np.nan,
            "burst_resource_cost": float(burst_subset.resource_cost.sum()),
        })
    zones, costs = pd.DataFrame(zone_rows), pd.DataFrame(burst_rows)
    base = float(costs.loc[costs.variant == PAPER, "burst_resource_cost"].iloc[0])
    costs["normalized_burst_cost"] = costs.burst_resource_cost / base if base else np.nan
    zones.to_csv(out / f"{trace}_tables4_5_zone_metrics.csv", index=False)
    costs.to_csv(out / f"{trace}_table6_burst_cost.csv", index=False)
    summary.to_csv(out / f"{trace}_full_summary.csv", index=False)
    return summary, zones, costs


def headline(summary: pd.DataFrame) -> str:
    paper = summary.loc[summary.variant == PAPER].iloc[0]
    four = summary.loc[summary.variant == ALL_FOUR].iloc[0]
    inplace = summary.loc[summary.variant == "novelty_4_inplace_resize"].iloc[0]
    slo_win = four.slo_violation_pct < paper.slo_violation_pct
    cost_win = four.resource_cost < paper.resource_cost
    util_win = four.mean_utilization_pct >= paper.mean_utilization_pct
    return (
        f"Paper Hybrid SLO {paper.slo_violation_pct:.3f}%, cost {paper.resource_cost:.3f}, "
        f"util {paper.mean_utilization_pct:.2f}%. Standalone N4 in-place SLO "
        f"{inplace.slo_violation_pct:.3f}%, cost {inplace.resource_cost:.3f}, "
        f"util {inplace.mean_utilization_pct:.2f}%. All-four novelties SLO "
        f"{four.slo_violation_pct:.3f}%, cost {four.resource_cost:.3f}, util {four.mean_utilization_pct:.2f}% "
        f"(SLO {'better' if slo_win else 'not lower'}, cost {'better' if cost_win else 'not lower'}, "
        f"util {'higher or equal' if util_win else 'lower'})."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worldcup", default="Novelty/data/processed/worldcup98_minute.csv")
    parser.add_argument("--azure", default="Novelty/data/processed/azurefunctions2019_minute.csv")
    parser.add_argument("--out", default="Novelty/results/paper_outputs_all")
    parser.add_argument("--window", type=int, default=7200)
    parser.add_argument("--window-mode", choices=("tail", "burst"), default="tail",
                        help="tail = last N points (paper-style chronological 80/20); burst = peak in holdout")
    parser.add_argument("--worldcup-results", default="")
    parser.add_argument("--azure-results", default="")
    parser.add_argument("--capacity-peak", type=float, default=1800)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    windows = out / "windows"
    windows.mkdir(exist_ok=True)
    figure_architecture(out / "figure1_architecture.png")
    figure_rolling(out / "figure2_rolling_update.png")
    figure_algorithms(out / "algorithms_1_and_2.png")
    figure_table3(out / "table3_parameters.png")

    perf_path = ROOT / "Novelty" / "data" / "generated" / "performance.csv"
    if not perf_path.exists():
        subprocess.check_call([sys.executable, str(ROOT / "Novelty" / "scripts" / "generate_performance_data.py"),
                               "--out", str(perf_path)], cwd=str(ROOT))
    table2 = benchmark_performance(perf_path, args.seed)
    table2.to_csv(out / "table2_performance_model.csv", index=False)
    pd.DataFrame(PAPER_TABLE1, columns=["trace", "model", "RMSE", "MAE"]).to_csv(out / "paper_table1_reported.csv", index=False)
    pd.DataFrame(PAPER_TABLE2, columns=["model", "MSE", "MAE"]).to_csv(out / "paper_table2_reported.csv", index=False)
    paper_table4 = pd.DataFrame(PAPER_TABLE4, columns=["trace", "burst_zone", "reactive_hpa_slo_pct", "proactive_hpa_slo_pct", "paper_proposal_slo_pct"])
    paper_table5 = pd.DataFrame(PAPER_TABLE5, columns=["trace", "burst_zone", "reactive_hpa_utilization_pct", "proactive_hpa_utilization_pct", "paper_proposal_utilization_pct"])
    paper_table6 = pd.DataFrame(PAPER_TABLE6, columns=["trace", "burst_aware_hpa_slo_pct", "burst_aware_hpa_normalized_cost", "paper_proposal_slo_pct", "paper_proposal_normalized_cost"])
    paper_table4.to_csv(out / "paper_table4_reported.csv", index=False)
    paper_table5.to_csv(out / "paper_table5_reported.csv", index=False)
    paper_table6.to_csv(out / "paper_table6_reported.csv", index=False)

    traces = {
        "worldcup98_fifa": ROOT / args.worldcup,
        "azurefunctions2019_wikipedia_like": ROOT / args.azure,
    }
    existing = {
        "worldcup98_fifa": Path(args.worldcup_results) if args.worldcup_results else None,
        "azurefunctions2019_wikipedia_like": Path(args.azure_results) if args.azure_results else None,
    }
    meta = {}
    md = [
        "# Paper outputs with novelty comparison",
        "",
        "Vu, Tran, and Kim, *Predictive Hybrid Autoscaling for Containerized Applications*, IEEE Access 2022.",
        "",
        (f"Each trace uses its last {args.window} observations in chronological order; the final 20% is held out. "
         if args.window_mode == "tail" else
         f"Each trace uses a {args.window}-observation window centered to place its peak in the held-out segment. ")
        "The DTR uses a synthetic development fixture, not a measured cluster model. "
        "Latency and cost are emulator estimates using Algorithm 2 and Equations (2)–(4).",
        "",
        "## Figure 1 — system architecture",
        "",
        "![Figure 1](figure1_architecture.png)",
        "",
        "## Figure 2 — rolling update versus in-place resize",
        "",
        "![Figure 2](figure2_rolling_update.png)",
        "",
        "## Algorithms 1 and 2",
        "",
        "![Algorithms](algorithms_1_and_2.png)",
        "",
        "## Table III — parameter setting",
        "",
        "![Table III](table3_parameters.png)",
        "",
        write_table(pd.DataFrame(PAPER_TABLE3, columns=["parameter", "value"])),
        "",
        "## Paper Table I — reported forecast errors (IEEE values, unit-interval scale)",
        "",
        write_table(pd.DataFrame(PAPER_TABLE1, columns=["trace", "model", "RMSE", "MAE"])),
        "",
        "## Paper Table II — reported performance-model errors",
        "",
        write_table(pd.DataFrame(PAPER_TABLE2, columns=["model", "MSE", "MAE"])),
        "",
        "## Paper Table 4 — published SLO violation results",
        "",
        "Published reference values, kept separate from this project's emulator output.",
        "",
        write_table(paper_table4),
        "",
        "## Paper Table 5 — published average pod utilization results",
        "",
        write_table(paper_table5),
        "",
        "## Paper Table 6 — published burst-case comparison",
        "",
        write_table(paper_table6),
        "",
        "## Project Table II — performance-model errors (Locust-style generated DTR data)",
        "",
        write_table(table2[["name", "macro_mse", "macro_mae", "replicas_mse", "replicas_mae", "utilization_mse", "utilization_mae"]]),
        "",
    ]

    for trace, source in traces.items():
        raw = load_rate(source)
        window = extract_tail_window(raw, args.window) if args.window_mode == "tail" else extract_burst_window(raw, args.window)
        csv_path = windows / f"{trace}.csv"
        pd.DataFrame({"request_rate": window}).to_csv(csv_path, index=False)
        mapped = existing.get(trace)
        result_dir = (ROOT / mapped).resolve() if mapped else (out / f"{trace}_ablation")
        if not args.skip_ablation:
            result_dir = out / f"{trace}_ablation"
            run_ablation(csv_path, result_dir, args)

        cut = int(0.8 * len(window))
        development, test = window[:cut], window[cut:]
        validation_size = max(20, len(development) // 5)
        train = development[:-validation_size]
        forecast_rows = classic_forecasts(train, test, development)
        paper_csv = result_dir / f"{PAPER}.csv"
        n2_csv = result_dir / "novelty_2_quantile_provisioning.csv"
        if paper_csv.exists():
            held = pd.read_csv(paper_csv).sort_values("tick")
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            actual = held.scaled_demand.to_numpy(dtype=float)
            pred = held.predicted_demand.to_numpy(dtype=float)
            forecast_rows.append({
                "name": "Bi-LSTM",
                "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
                "mae": float(mean_absolute_error(actual, pred)),
                "parameters": "lookback=10, hidden=32, held-out one-step from ablation",
            })
            if n2_csv.exists():
                q = pd.read_csv(n2_csv).sort_values("tick")
                qpred = q.p50.to_numpy(dtype=float) if "p50" in q else q.predicted_demand.to_numpy(dtype=float)
                n = min(len(actual), len(qpred))
                forecast_rows.append({
                    "name": "Quantile Bi-LSTM (P50)",
                    "rmse": float(np.sqrt(mean_squared_error(actual[:n], qpred[:n]))),
                    "mae": float(mean_absolute_error(actual[:n], qpred[:n])),
                    "parameters": "lookback=10, hidden=32, held-out P50 from ablation",
                })
        else:
            forecast_rows.extend(recurrent_forecasts(train, test, args.epochs, args.batch_size, args.seed, development))
        forecast = add_unit_interval_errors(pd.DataFrame(forecast_rows), train)
        forecast.insert(0, "trace", trace)
        forecast.to_csv(out / f"{trace}_table1_forecast.csv", index=False)

        summary, zones, costs = policy_plots(trace, result_dir, out)
        paper_variants = [v for v in [
            "reactive_hpa_fixed_cpu", "proactive_hpa_bilstm_fixed_cpu", PAPER, ALL_FOUR,
        ] if v in set(zones.variant)]
        table4 = paper_zone_table(zones, "slo_violation_pct", paper_variants)
        table5 = paper_zone_table(zones, "mean_utilization_pct", paper_variants)
        table4.to_csv(out / f"{trace}_table4_slo_paper_layout.csv", index=False)
        table5.to_csv(out / f"{trace}_table5_utilization_paper_layout.csv", index=False)
        pairwise_path = result_dir / "pairwise_vs_paper.csv"
        pairwise = pd.read_csv(pairwise_path) if pairwise_path.exists() else summary
        meta[trace] = {
            "source": str(source), "window_points": int(len(window)),
            "train_points": int(len(train)), "test_points": int(len(test)),
            "headline": headline(summary),
        }
        md += [
            f"## {trace}",
            "",
            meta[trace]["headline"],
            "",
            "### Table I — forecast errors on the held-out window",
            "",
            "Raw columns are on the capacity-scaled request-rate axis. "
            "`rmse_unit_interval` / `mae_unit_interval` divide by the training span, matching the paper's (0, 1) reporting.",
            "",
            write_table(forecast[["name", "rmse", "mae", "rmse_unit_interval", "mae_unit_interval", "parameters"]]),
            "",
            "### Figures 3/4 — forecast and burst identification",
            "",
            f"![{trace} forecast]({trace}_fig3_4_forecast_burst.png)",
            "",
            "### Figures 5/6 — CPU, replicas, utilization, response time",
            "",
            f"![{trace} scaling]({trace}_fig5_6_scaling.png)",
            "",
            "### Figure 7 — resource cost",
            "",
            f"![{trace} cost]({trace}_fig7_cost.png)",
            "",
            f"![{trace} normalized cost]({trace}_fig7_cost_normalized.png)",
            "",
            "### Paper Hybrid versus novelties",
            "",
            f"![{trace} novelty]({trace}_novelty_vs_hybrid.png)",
            "",
            "### Table 4 — percentage of total SLO violation by burst zone",
            "",
            write_table(table4),
            "",
            "### Table 5 — percentage of average pod utilization by burst zone",
            "",
            write_table(table5),
            "",
            "### Tables 4 and 5 — full per-policy zone metrics",
            "",
            write_table(zones),
            "",
            "### Table 6 — burst-window SLO and normalized cost",
            "",
            write_table(costs),
            "",
            "### Full policy summary versus Paper Hybrid",
            "",
            write_table(summary[["variant", "slo_violation_pct", "mean_utilization_pct",
                                 "resource_cost", "rolling_overlap_cost", "monitor_cycles"]]),
            "",
        ]
        if "delta_slo_violation_pct" in pairwise.columns:
            md += ["### Paired deltas versus Paper Hybrid (negative favors novelty)", "",
                   write_table(pairwise.loc[pairwise.variant.str.startswith(("novelty_", "optimized_")),
                                            [c for c in ["variant", "slo_violation_pct", "delta_slo_violation_pct",
                                                         "resource_cost", "delta_resource_cost"]
                                             if c in pairwise.columns]]), ""]

    (out / "PAPER_OUTPUTS.md").write_text("\n".join(md), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({
        "capacity_peak": args.capacity_peak, "window": args.window,
        "window_mode": args.window_mode, "epochs": args.epochs,
        "traces": meta, "paper_parameters": {
            "interval_seconds": 60, "lookback": 10, "burst_threshold": 5, "burst_influence": 0.5,
            "slo_ms": 350, "rmin": 600, "rmax": 950,
        },
    }, indent=2), encoding="utf-8")
    print(f"Wrote paper outputs to {out}")
    for trace, info in meta.items():
        print(trace, info["headline"])


if __name__ == "__main__":
    main()
