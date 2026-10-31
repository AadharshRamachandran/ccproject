"""Train the live Paper Hybrid and novelty forecasters on one selected trace.

This script uses the paper's chronological 80/20 split, makes local scaling
explicit, and trains the canonical Paper Bi-LSTM plus its N2 quantile extension.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Paper"))

from Novelty.src.burst import tune_burst_parameters
from Novelty.src.forecast import QuantileBiLSTMForecaster, WorkloadForecaster
from Novelty.src.monitoring import AdaptiveMonitoringInterval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", default=None,
                        help="Trace CSV; defaults to Novelty/data/processed/<trace>_minute.csv")
    parser.add_argument("--trace", default="worldcup98")
    parser.add_argument("--test-start-index", type=int, default=None,
                        help="Optional custom segment start; omit to use chronological 80/20 split")
    parser.add_argument("--training-points", type=int, default=300,
                        help="Training window size when --test-start-index is supplied")
    parser.add_argument("--test-points", type=int, default=75)
    parser.add_argument("--training-peak-qps", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="Novelty/models/paper_live_segments")
    args = parser.parse_args()

    if args.trace not in {"worldcup98", "azurefunctions2019"}:
        parser.error("--trace must be worldcup98 or azurefunctions2019")
    workload_path = (ROOT / args.workload if args.workload else
                     ROOT / "Novelty" / "data" / "processed" / f"{args.trace}_minute.csv")
    if not workload_path.is_file():
        parser.error(f"Workload trace does not exist: {workload_path}")
    frame = pd.read_csv(workload_path)
    key = "request_rate" if "request_rate" in frame.columns else frame.columns[-1]
    source = pd.to_numeric(frame[key], errors="coerce").fillna(0).clip(lower=0).to_numpy(float) / 60.0
    if args.test_start_index is None:
        first = 0
        test_start = int(len(source) * 0.8)
        last = test_start + args.test_points
        if last > len(source):
            parser.error("The chronological held-out trace has fewer than --test-points rows")
        development = source[:test_start]
        heldout = source[test_start:last]
        profile_mode = "chronological_80_20_split"
    else:
        first = args.test_start_index - args.training_points
        last = args.test_start_index + args.test_points
        if first < 0 or last > len(source):
            parser.error("The selected training/test segment is outside the source trace")
        development = source[first:args.test_start_index]
        heldout = source[args.test_start_index:last]
        profile_mode = "custom_contiguous_segment"
    peak = float(np.max(development))
    if peak <= 0:
        parser.error("Selected development segment has no positive demand")
    scale = args.training_peak_qps / peak
    development = development * scale
    heldout = heldout * scale

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    # The final 20% of the training split is validation-only for novelty
    # calibration. The Paper Hybrid forecaster is fitted on the full training
    # split, matching the paper's chronological 80/20 workload split.
    # Keep novelty-only tuning on a trailing validation sample inside the
    # training split. Cap it for large request traces so the adaptive-policy
    # search does not become the dominant experiment.
    validation_size = min(max(30, len(development) // 5), 300)
    fit_values = development[:-validation_size]
    validation = development[-validation_size:]

    tuning_forecast = WorkloadForecaster(lookback=10, hidden=32)
    tuning_forecast.fit(fit_values, epochs=args.epochs, batch_size=512, progress_every=20,
                        window_stride=1, seed=args.seed, label="burst-tuning-bilstm")
    validation_predictions = tuning_forecast.predict_many(
        fit_values[-10:], validation, batch_size=256)
    validation_actuals = validation

    risk_threshold = float(np.quantile(validation_actuals, 0.85))

    def burst_score(_predictions, flags, actuals):
        risky = np.asarray(actuals, dtype=float) >= risk_threshold
        detected = np.asarray(flags, dtype=bool)
        false_negative = int(np.sum(risky & ~detected))
        false_positive = int(np.sum(~risky & detected))
        return 10.0 * false_negative + false_positive

    burst_parameters, burst_validation_score = tune_burst_parameters(
        validation_predictions.tolist(), validation_actuals.tolist(), burst_score,
        thresholds=(0.25, 0.5, 1.0, 2.0, 3.0, 5.0),
        influences=(0.25, 0.5, 0.75), adaptive=(False, True), folds=3,
        consecutive_exceedances=(1, 2), magnitude_floors=(0.0,),
        forecast_error_margins=(0.0,), volatility_gains=(0.5, 1.0),
        min_thresholds=(0.25, 0.5, 1.0), signals=validation_predictions.tolist())

    tuning_quantile = QuantileBiLSTMForecaster(lookback=10, hidden=32, dropout=0.15)
    tuning_quantile.fit(fit_values, epochs=args.epochs, batch_size=512, progress_every=20,
                        window_stride=1, seed=args.seed, label="quantile-tuning-bilstm")
    bands = tuning_quantile.predict_intervals_many(fit_values[-10:], validation)
    widths = (bands[:, 2] - bands[:, 0]) / np.maximum(np.abs(bands[:, 1]), 1e-9)
    max_relative_width = max(float(np.quantile(widths, .99)), 1e-9)

    # Final forecasters are both trained on the complete same 300-point
    # development data. That ensures policy arms differ by novelty only.
    paper_forecast = WorkloadForecaster(lookback=10, hidden=32)
    paper_forecast.fit(development, epochs=args.epochs, batch_size=512, progress_every=20,
                       window_stride=1, seed=args.seed, label="paper-bilstm")
    paper_quantile = QuantileBiLSTMForecaster(lookback=10, hidden=32, dropout=0.15)
    paper_quantile.fit(development, epochs=args.epochs, batch_size=512, progress_every=20,
                       window_stride=1, seed=args.seed, label="n2-quantile-bilstm")

    monitor = AdaptiveMonitoringInterval(base_seconds=60, minimum_seconds=15,
                                         maximum_seconds=120, high_cv_percentile=.40)
    monitor.calibrate(validation.tolist(), high_cv_percentile=.40)
    trace_dir = ROOT / args.output / args.trace
    trace_dir.mkdir(parents=True, exist_ok=True)
    paper_forecast.save(str(trace_dir / "bilstm.pt"))
    paper_quantile.save(str(trace_dir / "quantile_bilstm.pt"))
    calibration = {
        "trace": args.trace,
        "workload_unit": "qps",
        "workload_source": str(workload_path.relative_to(ROOT)),
        "source_unit": "requests_per_minute",
        "workload_split": {
            "method": profile_mode,
            "test_start_source_index": (test_start if args.test_start_index is None
                                        else args.test_start_index),
            "training_start_source_index": first,
            "training_points": len(development),
            "test_points": args.test_points,
            "training_peak_qps_before_scale": peak,
            "target_training_peak_qps": args.training_peak_qps,
            "scale_factor": scale,
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "lookback": 10,
        "burst_parameters": {
            "threshold": burst_parameters.threshold,
            "influence": burst_parameters.influence,
            "volatility_adaptive": burst_parameters.volatility_adaptive,
            "volatility_gain": burst_parameters.volatility_gain,
            "min_threshold": burst_parameters.min_threshold,
            "consecutive_exceedances": burst_parameters.consecutive_exceedances,
            "magnitude_floor": burst_parameters.magnitude_floor,
            "forecast_error_margin": burst_parameters.forecast_error_margin,
        },
        "burst_validation_score": burst_validation_score,
        "burst_validation_risk_threshold_qps": risk_threshold,
        "quantile_safety_gain": 0.5,
        "quantile_max_relative_width": max_relative_width,
        "monitor_minimum_seconds": 15,
        "monitor_maximum_seconds": 120,
        "monitor_high_cv_percentile": .40,
        "monitor_low_cv": monitor.low_cv,
        "monitor_high_cv": monitor.high_cv,
        "validation_points": validation_size,
        "test_profile_qps_min": float(np.min(heldout)),
        "test_profile_qps_max": float(np.max(heldout)),
        "note": f"Canonical Paper Bi-LSTM trained on the {profile_mode} workload split; novelty tuning uses only a validation tail inside training. The original trace subset, exact scaling, calibration table, and trained weights are not published.",
    }
    (trace_dir / "policy_calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps({**calibration, "output_directory": str(trace_dir)}, indent=2))


if __name__ == "__main__":
    main()
