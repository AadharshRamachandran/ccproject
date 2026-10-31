"""Filter measured Kubernetes rows, validate by held-out load run, and train both DTRs."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Paper"))
from Novelty.src.performance import PerformanceModel

FEATURES = ["request_rate", "cpu_millicores", "slo_ms"]
TARGETS = ["replicas", "utilization"]
REQUIRED = FEATURES + TARGETS + ["response_ms"]


def load_measurements(paths: list[Path], samples_per_run: int) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    sources = []
    group_offset = 0
    for path in paths:
        frame = pd.read_csv(path)
        missing = sorted(set(REQUIRED) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        explicit_groups = (frame["measurement_run_id"].astype(str)
                           if "measurement_run_id" in frame.columns else None)
        frame = frame[REQUIRED].apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        # Keep all samples from an explicitly identified calibration run in one
        # validation group to prevent temporal leakage across train and test.
        if explicit_groups is not None:
            groups = explicit_groups.loc[frame.index].astype(str).reset_index(drop=True)
            frame = frame.reset_index(drop=True)
            codes, _ = pd.factorize(groups, sort=True)
            frame["_run_group"] = group_offset + codes
            group_offset += int(codes.max() + 1) if len(codes) else 0
        else:
            frame = frame.reset_index(drop=True)
            # Legacy collection emits contiguous samples per stable Locust load condition.
            frame["_run_group"] = group_offset + np.arange(len(frame)) // samples_per_run
            group_offset += int(np.ceil(len(frame) / samples_per_run))
        frame["_source"] = str(path.relative_to(ROOT))
        frames.append(frame)
        sources.append(str(path.relative_to(ROOT)))
    return pd.concat(frames, ignore_index=True), sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=[
        "Novelty/data/collected/performance_live_hpa.csv",
        "Novelty/data/collected/performance_live_hpa_extra.csv",
        "Novelty/data/collected/performance_live_hpa_more.csv",
        "Novelty/data/collected/performance_live_hpa_lowrate.csv",
        "Novelty/data/collected/performance_live_cost_450m.csv",
    ])
    parser.add_argument("--slo-max-ms", type=float, default=350)
    parser.add_argument("--samples-per-run", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-qps", type=float, default=0.5)
    args = parser.parse_args()

    paths = [(ROOT / item).resolve() for item in args.input]
    raw, sources = load_measurements(paths, args.samples_per_run)
    clean = raw.loc[
        (raw.request_rate >= args.min_qps)
        & (raw.cpu_millicores > 0)
        & (raw.slo_ms > 0)
        & (raw.replicas >= 1)
        & (raw.utilization >= 0)
        & (raw.utilization <= 1)
        & (raw.response_ms > 0)
        & (raw.response_ms <= np.minimum(raw.slo_ms, args.slo_max_ms))
    ].copy().reset_index(drop=True)
    if len(clean) < 20 or clean.cpu_millicores.nunique() < 3 or clean.request_rate.nunique() < 5:
        raise ValueError(
            "Need at least 20 in-SLO rows, 3 CPU levels, and 5 distinct request rates; "
            f"got rows={len(clean)}, CPU levels={clean.cpu_millicores.nunique()}, "
            f"request rates={clean.request_rate.nunique()}."
        )
    if clean._run_group.nunique() < 5:
        raise ValueError("Need at least five distinct measured load runs for held-out validation.")

    x = clean[FEATURES]
    y = clean[TARGETS]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(x, y, groups=clean._run_group))
    heldout = PerformanceModel().fit(clean.iloc[train_idx])
    predictions = heldout.predict_frame(x.iloc[test_idx])
    actual = y.iloc[test_idx].to_numpy()
    metrics = {}
    for column_idx, name in enumerate(TARGETS):
        metrics[name] = {
            "mae": float(mean_absolute_error(actual[:, column_idx], predictions[:, column_idx])),
            "rmse": float(np.sqrt(mean_squared_error(actual[:, column_idx], predictions[:, column_idx]))),
            "r2": float(r2_score(actual[:, column_idx], predictions[:, column_idx]))
            if len(test_idx) > 1 else None,
        }

    out_data = ROOT / "Novelty" / "data" / "collected"
    out_data.mkdir(parents=True, exist_ok=True)
    dataset_path = out_data / "performance_measured_inslo.csv"
    # Preserve load-run groups so downstream model comparisons can hold out
    # entire measurement conditions instead of leaking near-identical samples.
    clean[REQUIRED].assign(measurement_run_id=clean["_run_group"].astype(int)).to_csv(
        dataset_path, index=False
    )

    novelty_dir = ROOT / "Novelty" / "models"
    paper_dir = ROOT / "Paper" / "models"
    novelty_dir.mkdir(exist_ok=True)
    paper_dir.mkdir(exist_ok=True)
    model = PerformanceModel().fit(clean)
    novelty_model = novelty_dir / "performance.joblib"
    model.save(novelty_model)
    paper_model = paper_dir / "performance.joblib"
    shutil.copy2(novelty_model, paper_model)

    report = {
        "data_origin": ("Measured Kubernetes house-price service with Locust and Prometheus. Historical "
                        "calibration rows used HPA-selected replicas; the added 450m CPU calibration "
                        "used two fixed replicas on an Azure development segment."),
        "sources": sources,
        "filter": {
            "response_ms_at_or_below": "min(slo_ms, slo_max_ms)",
            "slo_max_ms": args.slo_max_ms,
            "minimum_request_rate": args.min_qps,
            "invalid_rows_removed": int(len(raw) - len(clean)),
        },
        "rows": int(len(clean)),
        "heldout_validation": {
            "method": ("80/20 GroupShuffleSplit by measurement run; explicit run IDs stay together, "
                       f"and legacy rows use contiguous groups of {args.samples_per_run}"),
            "random_seed": args.seed,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_load_runs": int(clean.iloc[train_idx]._run_group.nunique()),
            "test_load_runs": int(clean.iloc[test_idx]._run_group.nunique()),
            "targets": metrics,
        },
        "paper_model_comparison": PerformanceModel.compare_regressors(clean, seed=args.seed),
        "paper_selection_note": (
            "The published method compares LR, RDF, SVR, XGB, MLP, and DTR after "
            "80/20 normalized holdout evaluation. This implementation evaluates "
            "the available families on this project's measured table and fits DTR "
            "for every baseline/novelty arm, following the paper's reported choice."
        ),
        "coverage": {
            "cpu_millicores": sorted(int(v) for v in clean.cpu_millicores.unique()),
            "replicas": sorted(int(v) for v in clean.replicas.unique()),
            "request_rate_min": float(clean.request_rate.min()),
            "request_rate_max": float(clean.request_rate.max()),
            "distinct_request_rates": int(clean.request_rate.nunique()),
            "p95_response_ms_min": float(clean.response_ms.min()),
            "p95_response_ms_max": float(clean.response_ms.max()),
            "all_p95_within_slo": bool((clean.response_ms <= clean.slo_ms).all()),
        },
        "artifacts": {
            "training_data": str(dataset_path.relative_to(ROOT)),
            "novelty_dtr": str(novelty_model.relative_to(ROOT)),
            "paper_dtr": str(paper_model.relative_to(ROOT)),
        },
        "limitations": [
            "This is measured calibration data from the local Minikube benchmark service, not production Kubernetes traces.",
            "The offline trace ablation still estimates latency and SLO with its simulator; using this DTR does not make those simulated outcomes live measurements.",
            "The benchmark has one application and a limited CPU/request-rate envelope; extrapolation outside the recorded range is unvalidated.",
        ],
    }
    report_path = novelty_dir / "performance_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
