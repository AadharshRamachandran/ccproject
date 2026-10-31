"""Train the published Paper Hybrid Bi-LSTM and normalized DTR artifacts."""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import numpy as np
import random
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data import chronological_split, load_workload_csv
from src.forecast import WorkloadForecaster
from src.performance import PerformanceModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--performance", required=True)
    parser.add_argument("--models", default="Paper/models")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.models, exist_ok=True)

    workload = load_workload_csv(args.workload)
    train, test = chronological_split(workload, train_fraction=0.8)
    forecaster = WorkloadForecaster(lookback=10, hidden=32)
    forecaster.fit(train, epochs=args.epochs, batch_size=args.batch_size)
    forecaster.save(os.path.join(args.models, "bilstm.pt"))

    performance_data = pd.read_csv(args.performance)
    selection = PerformanceModel.compare_regressors(performance_data, seed=args.seed)
    # The published method selected DTR; all novelty arms load this same
    # artifact so their comparison changes policy only, not calibration.
    performance = PerformanceModel(seed=args.seed).fit(performance_data)
    performance.save(os.path.join(args.models, "performance.joblib"))
    report = {
        "paper_method": "Predictive Hybrid Autoscaling, Vu et al., IEEE Access 2022",
        "workload_split": "chronological 80/20",
        "workload_train_points": len(train),
        "workload_heldout_points": len(test),
        "forecaster": {"type": "Bi-LSTM", "lookback": 10,
                       "hidden_units_per_direction": 32,
                       "epochs": args.epochs, "batch_size": args.batch_size,
                       "normalization": "training-range MinMax to [0, 1]"},
        "performance_model_comparison": selection,
        "controller_performance_model": performance.training_report,
        "paper_specified_controller": {
            "burst_window": 10, "burst_threshold": 5.0,
            "burst_influence": 0.5, "slo_ms": 350,
            "cpu_candidates_millicores": [600, 700, 800, 900, 950],
            "monitor_interval_seconds": 60,
            "execution": "rolling deployment replacement on CPU change; horizontal scale otherwise",
        },
        "reproduction_limits": [
            "The paper's original performance calibration table is not included; the supplied table is used.",
            "The paper does not specify all Bi-LSTM architecture/training hyperparameters; hidden size 32, Adam 0.005, and batch settings are implementation defaults, not verified paper values.",
            "The paper's original Kubernetes cluster and trace subsets are unavailable, so this artifact training is not a reproduction of the reported experimental measurements.",
        ],
    }
    with open(os.path.join(args.models, "paper_training_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
