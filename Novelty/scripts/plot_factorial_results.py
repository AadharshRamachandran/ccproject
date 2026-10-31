"""Plot all-arm simulator output for one or more factorial result folders."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def short_name(name: str) -> str:
    if name == "paper_hybrid_bilstm_dtr_rolling_fixed_monitor":
        return "Paper Hybrid"
    name = name.replace("optimized_novelty_", "")
    name = name.replace("novelty_", "N")
    return name.replace("_", "+")


def plot_folder(folder: Path) -> None:
    summary = pd.read_csv(folder / "summary.csv")
    # Keep only the shared paper reference and the full 2^4 novelty factorial.
    mask = summary.variant.eq("paper_hybrid_bilstm_dtr_rolling_fixed_monitor") | summary.variant.str.startswith(("novelty_", "optimized_novelty_"))
    data = summary.loc[mask].copy()
    labels = [short_name(v) for v in data.variant]
    colors = ["#1f77b4" if v.startswith("paper_hybrid") else "#d97706" for v in data.variant]

    fig, axes = plt.subplots(3, 1, figsize=(15, 14), sharex=True, constrained_layout=True)
    axes[0].bar(range(len(data)), data.slo_violation_pct, color=colors)
    axes[0].set_ylabel("SLO violations (%)")
    axes[0].set_ylim(0, max(float(data.slo_violation_pct.max()) * 1.2, 0.1))
    if float(data.slo_violation_pct.max()) == 0:
        for i in range(len(data)):
            axes[0].text(i, .008, "0", ha="center", va="bottom", fontsize=7)
    axes[0].set_title("Offline simulator estimates — Paper-method Hybrid vs novelty factorial")
    axes[1].bar(range(len(data)), data.resource_cost, color=colors)
    axes[1].set_ylabel("Modeled resource cost")
    axes[2].bar(range(len(data)), data.mean_utilization_pct, color=colors)
    axes[2].set_ylabel("Mean utilization (%)")
    axes[2].set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.25)
    path = folder / "all_arms_metrics.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    for folder in args.results:
        plot_folder(folder)
        print(folder / "all_arms_metrics.png")


if __name__ == "__main__":
    main()
