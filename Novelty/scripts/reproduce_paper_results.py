"""Generate paper-aligned component, policy, and figure outputs from this project.

The script keeps paper-reported figures separate from project reproductions. It
uses grouped holdouts for measured DTR data and the saved trace ablation rows for
policy tables/plots.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Paper"))

PAPER = "paper_hybrid_bilstm_dtr_rolling_fixed_monitor"
LOOKBACK = 10


def lagged(values: np.ndarray, lookback: int = LOOKBACK):
    values = np.asarray(values, dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(values, lookback + 1)
    return windows[:, :-1], windows[:, -1]


def forecast_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def classic_forecasts(train, test, history_seed=None):
    from statsmodels.tsa.arima.model import ARIMA

    rows = []
    x, y = lagged(train)
    history = list(np.asarray(train if history_seed is None else history_seed, dtype=float))
    models = {
        "LR": LinearRegression(),
        "EN": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000),
        "OLS": LinearRegression(),
        "NNLS": LinearRegression(positive=True),
    }
    for name, model in models.items():
        model.fit(x, y)
        predictions = []
        for actual in test:
            prediction = float(model.predict(np.asarray(history[-LOOKBACK:]).reshape(1, -1))[0])
            predictions.append(max(0., prediction))
            history.append(float(actual))
        rows.append({"name": name, **forecast_metrics(test, predictions), "parameters": f"lag={LOOKBACK}"})
        history = list(np.asarray(train if history_seed is None else history_seed, dtype=float))

    # The paper does not give the tuned ARIMA order. Use a documented AR(1)
    # walk-forward reference, updating its state with each newly observed point.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ARIMA(np.asarray(train, dtype=float), order=(1, 0, 0), trend="c").fit()
        if history_seed is not None and len(history_seed) > len(train):
            result = result.append(np.asarray(history_seed[len(train):], dtype=float), refit=False)
        predictions = []
        for actual in test:
            prediction = float(np.asarray(result.forecast(1)).reshape(-1)[0])
            predictions.append(max(0., prediction))
            result = result.append([float(actual)], refit=False)
    rows.append({"name": "ARIMA", **forecast_metrics(test, predictions), "parameters": "ARIMA(1,0,0), expanding state"})
    return rows


def recurrent_forecasts(train, test, epochs, batch_size, seed=42, history_seed=None):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from Novelty.src.forecast import WorkloadForecaster, QuantileBiLSTMForecaster

    torch.set_num_threads(4)
    all_rows = []
    # Paper LSTM baseline, implemented with the same lookback, scaling, and
    # training split as the project's Bi-LSTM.
    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.out = nn.Linear(32, 1)

        def forward(self, x):
            return self.out(self.lstm(x)[0][:, -1, :])

    values = np.asarray(train, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 1e-9)
    normalized = (values - low) / span
    xx, yy = lagged(normalized)
    rng = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(xx, dtype=torch.float32).unsqueeze(-1),
                      torch.tensor(yy, dtype=torch.float32).unsqueeze(-1)),
        batch_size=batch_size, shuffle=True, generator=rng,
    )
    torch.manual_seed(seed)
    net = LSTMNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=.005)
    loss_fn = nn.MSELoss()
    net.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = loss_fn(net(bx), by)
            loss.backward()
            optimizer.step()
    hist = list(np.asarray(train if history_seed is None else history_seed, dtype=float))
    predictions = []
    net.eval()
    for actual in test:
        window = (np.asarray(hist[-LOOKBACK:], dtype=np.float32) - low) / span
        with torch.no_grad():
            pred = float(net(torch.tensor(window).view(1, LOOKBACK, 1)).item()) * span + low
        predictions.append(max(0., pred))
        hist.append(float(actual))
    all_rows.append({"name": "LSTM", **forecast_metrics(test, predictions), "parameters": f"lookback={LOOKBACK}, hidden=32, epochs={epochs}"})

    for name, factory in (("Bi-LSTM", WorkloadForecaster), ("Quantile Bi-LSTM (P50)", QuantileBiLSTMForecaster)):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = factory()
        model.fit(train, epochs=epochs, batch_size=batch_size, window_stride=1,
                  progress_every=0, label=name, seed=seed)
        hist = list(np.asarray(train if history_seed is None else history_seed, dtype=float))
        predictions = []
        for actual in test:
            if hasattr(model, "predict_interval"):
                pred = model.predict_interval(hist)[1]
            else:
                pred = model.predict(hist)
            predictions.append(float(pred))
            hist.append(float(actual))
        all_rows.append({"name": name, **forecast_metrics(test, predictions), "parameters": f"lookback={LOOKBACK}, hidden=32, epochs={epochs}"})
    return all_rows


def benchmark_performance(path: Path, seed=42):
    frame = pd.read_csv(path)
    required = ["request_rate", "cpu_millicores", "slo_ms", "replicas", "utilization"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Measured DTR table lacks fields: {missing}; retrain with the latest train_measured_dtr.py")
    X = frame[["request_rate", "cpu_millicores", "slo_ms"]].to_numpy(float)
    y = frame[["replicas", "utilization"]].to_numpy(float)
    if "measurement_run_id" in frame:
        groups = frame.measurement_run_id.to_numpy()
        split = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(X, y, groups))
        split_method = "80/20 GroupShuffleSplit by Locust load condition"
    else:
        groups = np.arange(len(frame)) // 3
        split = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(X, y, groups))
        split_method = "80/20 inferred 3-row run groups (legacy table)"
    train_idx, test_idx = split
    from xgboost import XGBRegressor

    estimators = {
        "LR": LinearRegression(),
        "RDF": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=seed, n_jobs=4),
        "SVR": make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=10, epsilon=.05, kernel="rbf"))),
        "XGB": MultiOutputRegressor(XGBRegressor(n_estimators=180, max_depth=3, learning_rate=.05,
                                                  subsample=.9, colsample_bytree=.9,
                                                  objective="reg:squarederror", n_jobs=4,
                                                  random_state=seed, verbosity=0)),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                                             early_stopping=True, random_state=seed)),
        "DTR": DecisionTreeRegressor(random_state=42, min_samples_leaf=2),
    }
    rows = []
    for name, model in estimators.items():
        model.fit(X[train_idx], y[train_idx])
        pred = np.asarray(model.predict(X[test_idx]), dtype=float)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        errors = y[test_idx] - pred
        rows.append({
            "name": name,
            "macro_mse": float(np.mean(np.mean(errors ** 2, axis=0))),
            "macro_mae": float(np.mean(np.mean(np.abs(errors), axis=0))),
            "replicas_mse": float(mean_squared_error(y[test_idx, 0], pred[:, 0])),
            "replicas_mae": float(mean_absolute_error(y[test_idx, 0], pred[:, 0])),
            "utilization_mse": float(mean_squared_error(y[test_idx, 1], pred[:, 1])),
            "utilization_mae": float(mean_absolute_error(y[test_idx, 1], pred[:, 1])),
            "train_rows": len(train_idx), "test_rows": len(test_idx),
            "split": split_method,
        })
    return pd.DataFrame(rows)


def write_table(df):
    if df.empty:
        return "_No rows._"
    data = df.copy().replace({np.nan: ""})
    for col in data.select_dtypes(include="number"):
        data[col] = data[col].map(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x))
    header = "| " + " | ".join(map(str, data.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(data.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in data.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *body])


def policy_outputs(trace_name, result_dir: Path, out: Path):
    import matplotlib.pyplot as plt

    summary = pd.read_csv(result_dir / "summary.csv")
    frames = {p.stem: pd.read_csv(p) for p in result_dir.glob("*.csv")
              if p.name not in {"summary.csv", "pairwise_vs_paper.csv", "novelty_effects_vs_paper.csv"}}
    if PAPER not in frames:
        raise FileNotFoundError(f"Missing Paper Hybrid detail CSV in {result_dir}")
    paper = frames[PAPER].sort_values("tick").reset_index(drop=True)
    common_burst = paper.burst.fillna(False).astype(bool).to_numpy()
    zone_rows, burst_cost_rows = [], []
    for variant, frame in frames.items():
        frame = frame.sort_values("tick").reset_index(drop=True)
        n = min(len(frame), len(common_burst))
        b = common_burst[:n]
        for zone, mask in (("burst", b), ("non_burst", ~b)):
            subset = frame.iloc[:n].loc[mask]
            if len(subset) == 0:
                continue
            zone_rows.append({
                "trace": trace_name, "variant": variant, "zone": zone, "intervals": len(subset),
                "slo_violation_pct": 100 * float(subset.slo_violation.astype(float).mean()),
                "mean_utilization_pct": 100 * float(subset.utilization.mean()),
                "resource_cost": float(subset.resource_cost.sum()),
                "mean_response_ms": float(subset.response_ms.mean()),
            })
        burst_subset = frame.iloc[:n].loc[b]
        burst_cost_rows.append({
            "trace": trace_name, "variant": variant, "burst_intervals": len(burst_subset),
            "burst_slo_violation_pct": 100 * float(burst_subset.slo_violation.astype(float).mean()) if len(burst_subset) else np.nan,
            "burst_resource_cost": float(burst_subset.resource_cost.sum()),
        })
    zones = pd.DataFrame(zone_rows)
    costs = pd.DataFrame(burst_cost_rows)
    base_cost = float(costs.loc[costs.variant == PAPER, "burst_resource_cost"].iloc[0])
    costs["burst_cost_vs_paper_pct"] = np.where(base_cost > 0, 100 * (costs.burst_resource_cost / base_cost - 1), np.nan)
    zones.to_csv(out / f"{trace_name}_burst_zone_metrics.csv", index=False)
    costs.to_csv(out / f"{trace_name}_burst_cost_metrics.csv", index=False)

    # Figure 3/4 analogue: trace demand and paper Bi-LSTM prediction, with
    # burst periods from the paper detector.
    fig, ax = plt.subplots(figsize=(12, 4))
    ticks = paper.tick.to_numpy()
    ax.plot(ticks, paper.scaled_demand, label="Actual scaled demand", linewidth=1)
    ax.plot(ticks, paper.predicted_demand, label="Bi-LSTM forecast", linewidth=1)
    ax.fill_between(ticks, 0, np.maximum(paper.scaled_demand.max(), 1), where=common_burst,
                    color="tab:red", alpha=.12, label="Paper burst flag")
    ax.set(title=f"{trace_name}: demand forecast and burst identification", xlabel="Held-out minute interval", ylabel="Requests/second (scaled)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{trace_name}_forecast_burst.png", dpi=170)
    plt.close(fig)

    # Figure 5/6 analogue: CPU, replicas, utilization, and modeled latency.
    compare = ["reactive_hpa_fixed_cpu", "proactive_hpa_bilstm_fixed_cpu", PAPER,
               "novelty_4_inplace_resize"]
    compare = [v for v in compare if v in frames]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fields = [("cpu_millicores", "CPU per pod (millicores)"), ("replicas", "Replicas"),
              ("utilization", "Predicted/estimated utilization"), ("response_ms", "Modeled response (ms)")]
    for ax, (field, ylabel) in zip(axes, fields):
        for variant in compare:
            f = frames[variant].sort_values("tick")
            ax.plot(f.tick, f[field], label=variant, linewidth=.9)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Held-out minute interval")
    fig.suptitle(f"{trace_name}: autoscaling policy comparison")
    fig.tight_layout()
    fig.savefig(out / f"{trace_name}_scaling_comparison.png", dpi=170)
    plt.close(fig)

    # Figure 7 analogue with explicit relative units; paper's original plot is
    # normalized to its own workload-cost range, not comparable across traces.
    keep = [v for v in ["reactive_hpa_fixed_cpu", "proactive_hpa_bilstm_fixed_cpu", PAPER,
                         "novelty_1_tuned_adaptive_burst", "novelty_2_quantile_provisioning",
                         "novelty_3_adaptive_monitoring", "novelty_4_inplace_resize"] if v in set(summary.variant)]
    plot = summary.set_index("variant").loc[keep]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(plot)), plot.resource_cost, color=["#667085" if v == PAPER else "#3b82f6" for v in keep])
    ax.set_xticks(range(len(plot)), [v.replace("_", "\n") for v in keep], fontsize=7)
    ax.set_ylabel("Estimated resource cost (simulator units)")
    ax.set_title(f"{trace_name}: resource cost by policy")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / f"{trace_name}_resource_cost.png", dpi=170)
    plt.close(fig)

    return summary, zones, costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance", default="Novelty/data/collected/performance_measured_inslo.csv")
    parser.add_argument("--worldcup", default="Novelty/data/processed/worldcup98_minute.csv")
    parser.add_argument("--azure", default="Novelty/data/processed/azurefunctions2019_minute.csv")
    parser.add_argument("--worldcup-results", default="Novelty/results/worldcup98_measured_dtr_1440")
    parser.add_argument("--azure-results", default="Novelty/results/azurefunctions2019_measured_dtr_1440")
    parser.add_argument("--out", default="Novelty/results/paper_full_outputs_measured_dtr")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-points", type=int, default=1440)
    parser.add_argument("--reuse-existing-forecasts", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    from Paper.src.data import load_workload_csv

    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    perf = benchmark_performance((ROOT / args.performance).resolve(), args.seed)
    perf.to_csv(out / "table2_performance_model_errors.csv", index=False)

    paper_table1 = pd.DataFrame([
        {"trace": "Wikipedia", "model": m, "RMSE": r, "MAE": a, "source": "IEEE paper Table 1"}
        for m, r, a in [("ARIMA", .985, .699), ("LR", .872, .643), ("EN", .778, .589), ("OLS", .921, .730), ("NNLS", .879, .620), ("LSTM", .742, .542), ("Bi-LSTM", .655, .466)]
    ] + [
        {"trace": "FIFA World Cup", "model": m, "RMSE": r, "MAE": a, "source": "IEEE paper Table 1"}
        for m, r, a in [("ARIMA", .875, .523), ("LR", .709, .623), ("EN", .636, .419), ("OLS", .910, .660), ("NNLS", .880, .653), ("LSTM", .509, .323), ("Bi-LSTM", .436, .289)]
    ])
    paper_table1.to_csv(out / "paper_reported_table1_forecasting.csv", index=False)
    paper_table2 = pd.DataFrame([
        ("LR", .836, 3.27), ("RDF", .984, 3.51), ("SVR", .523, 2.98),
        ("XGB", .268, 2.67), ("MLP", .732, 3.14), ("DTR", .109, 1.81),
    ], columns=["model", "MSE", "MAE"])
    paper_table2.to_csv(out / "paper_reported_table2_performance_model.csv", index=False)

    traces = {}
    for trace, workload_path, result_path in (
        ("worldcup98", args.worldcup, args.worldcup_results),
        ("azurefunctions2019", args.azure, args.azure_results),
    ):
        values = np.asarray(load_workload_csv(str((ROOT / workload_path).resolve())), dtype=float)
        if args.max_points > 0:
            values = values[-args.max_points:]
        cut = int(.8 * len(values))
        development, test = values[:cut], values[cut:]
        validation_size = max(20, len(development) // 5)
        model_train, history_seed = development[:-validation_size], development
        forecast_path = out / f"{trace}_table1_forecast_errors.csv"
        if args.reuse_existing_forecasts and forecast_path.exists():
            forecast = pd.read_csv(forecast_path)
        else:
            rows = classic_forecasts(model_train, test, history_seed)
            rows.extend(recurrent_forecasts(model_train, test, args.epochs, args.batch_size, args.seed, history_seed))
            forecast = pd.DataFrame(rows)
            forecast.insert(0, "trace", trace)
            forecast["train_points"] = len(model_train)
            forecast["validation_points"] = validation_size
            forecast["test_points"] = len(test)
            forecast.to_csv(forecast_path, index=False)
        summary, zones, burst_costs = policy_outputs(trace, (ROOT / result_path).resolve(), out)
        traces[trace] = {"points": len(values), "training_points": len(model_train),
                         "validation_points": validation_size, "test_points": len(test),
                         "paper_hybrid_slo_pct": float(summary.loc[summary.variant == PAPER, "slo_violation_pct"].iloc[0]),
                         "paper_hybrid_cost": float(summary.loc[summary.variant == PAPER, "resource_cost"].iloc[0]),
                         "paper_hybrid_utilization_pct": float(summary.loc[summary.variant == PAPER, "mean_utilization_pct"].iloc[0]),
                         "novelty_variants": int(summary.variant.str.startswith(("novelty_", "optimized_novelty_")).sum())}

    # Paper-reported policy tables and burst-aware-HPA table are preserved as
    # reference data. The project tables below are distinct remeasurements.
    paper_t45 = pd.DataFrame([
        ("Wikipedia", "burst", "SLO violation (%)", 22.14, 2.87, 2.96),
        ("Wikipedia", "non-burst", "SLO violation (%)", 8.63, 1.14, 1.53),
        ("FIFA", "burst", "SLO violation (%)", 20.54, 4.10, 3.89),
        ("FIFA", "non-burst", "SLO violation (%)", 5.47, 1.57, 1.86),
        ("Wikipedia", "burst", "Pod utilization (%)", 76.68, 73.51, 73.73),
        ("Wikipedia", "non-burst", "Pod utilization (%)", 61.78, 55.26, 69.65),
        ("FIFA", "burst", "Pod utilization (%)", 78.30, 72.68, 72.58),
        ("FIFA", "non-burst", "Pod utilization (%)", 64.58, 63.96, 74.17),
    ], columns=["trace", "zone", "metric", "Reactive HPA", "Proactive HPA", "Paper Hybrid"])
    paper_t45.to_csv(out / "paper_reported_tables4_5_policy_metrics.csv", index=False)
    paper_t6 = pd.DataFrame([
        ("Wikipedia", 3.11, .793, 2.96, .608),
        ("FIFA", 4.05, .809, 3.89, .612),
    ], columns=["trace", "burst_aware_hpa_slo_pct", "burst_aware_hpa_normalized_cost", "paper_hybrid_slo_pct", "paper_hybrid_normalized_cost"])
    paper_t6.to_csv(out / "paper_reported_table6_burst_comparison.csv", index=False)

    md = [
        "# Paper outputs reproduced with this project",
        "",
        "Paper: Dinh-Dai Vu, Minh-Ngoc Tran, and Younghan Kim, ‘Predictive Hybrid Autoscaling for Containerized Applications,’ IEEE Access 10 (2022), DOI 10.1109/ACCESS.2022.3214985.",
        "",
        "## Scope and interpretation",
        "",
        "The paper reports Tables 1–6 and Figures 3–7. This project can reproduce the methods and outputs where its workload, forecast models, measured performance data, and policy traces support them. The IEEE values are retained as references; project remeasurements are kept separate because the paper uses Wikipedia and FIFA traces, while this project has a World Cup trace and an Azure Functions trace. The measured DTR table comes from a local house-price Kubernetes app, not either trace application.",
        "",
        "The paper’s Table 6 burst-aware HPA is not implemented as the exact cited [21] method in this repository. The project burst-cost tables therefore compare its implemented policies using the paper Hybrid detector’s burst windows and must not be described as a reproduction of Table 6’s baseline.",
        "",
        "## Paper Table 1 — reported forecast errors",
        "",
        write_table(paper_table1),
        "",
        "## Project Table 1 — held-out forecast errors",
        "",
        "All forecasts use the chronological 80/20 split and one-step teacher-forced predictions; classic models use 10 lags, ARIMA uses ARIMA(1,0,0), recurrent models use 10 lags and the requested training epochs. These are project remeasurements, not the paper's values.",
        "",
    ]
    for trace in traces:
        df = pd.read_csv(out / f"{trace}_table1_forecast_errors.csv")
        md += [f"### {trace}", "", write_table(df[["name", "rmse", "mae", "train_points", "test_points", "parameters"]]), ""]
    md += ["## Paper Table 2 — reported performance-model errors", "", write_table(paper_table2), "",
           "## Project Table 2 — measured DTR model comparison", "",
           "Models are tested on held-out measured load groups; metrics are reported separately for replica count and utilization. The paper’s MSE/MAE were calculated on its own generated benchmark data and normalization, so numeric values are not directly comparable.", "",
           write_table(perf[["name", "macro_mse", "macro_mae", "replicas_mse", "replicas_mae", "utilization_mse", "utilization_mae", "train_rows", "test_rows"]]), "",
           "## Paper Tables 4 and 5 — reported policy metrics", "", write_table(paper_t45), "",
           "## Project Tables 4 and 5 — per-zone policy metrics", "",
           "Burst/non-burst zones use the Paper Hybrid's detector flags as a common partition for every method. SLO and utilization outcomes are simulator estimates from the saved held-out trace evaluation.", ""]
    for trace in traces:
        df = pd.read_csv(out / f"{trace}_burst_zone_metrics.csv")
        md += [f"### {trace}", "", write_table(df), ""]
    md += ["## Paper Table 6 — reported burst-aware comparison", "", write_table(paper_t6), "",
           "## Project burst-window cost and SLO comparison", "",
           "Cost is summed over the Paper Hybrid burst windows and normalized relative to the project's Paper Hybrid cost (baseline = 0% change). This is not an exact replication of Table 6 because the cited burst-aware HPA implementation/data is unavailable.", ""]
    for trace in traces:
        df = pd.read_csv(out / f"{trace}_burst_cost_metrics.csv")
        md += [f"### {trace}", "", write_table(df), ""]
    md += ["## Full factorial ablation — all configurations", "",
           "These are the complete outputs from the paired Paper Hybrid and novelty evaluation, including all 11 combinations of two or more novelties.", ""]
    for trace, result_path in (("worldcup98", args.worldcup_results), ("azurefunctions2019", args.azure_results)):
        full = pd.read_csv((ROOT / result_path / "summary.csv").resolve())
        full.to_csv(out / f"{trace}_full_factorial_summary.csv", index=False)
        columns = [c for c in ["variant", "intervals", "slo_violation_pct", "slo_ci95_low_pct",
                               "slo_ci95_high_pct", "mean_utilization_pct", "resource_cost",
                               "cost_ci95_low", "cost_ci95_high", "rolling_overlap_cost", "monitor_cycles"]
                   if c in full.columns]
        md += [f"### {trace}", "", write_table(full[columns]), ""]
    md += ["## Figures", "", "The generated forecast/burst plot, policy scaling plot, and cost plot are analogues of paper Figures 3–7 using each project's held-out trace. `Figure 1` and `Figure 2` are architecture/workflow diagrams, not empirical outputs.", ""]
    for trace in traces:
        md += [f"### {trace}", "", f"![Forecast and burst]({trace}_forecast_burst.png)", "",
               f"![Scaling comparison]({trace}_scaling_comparison.png)", "",
               f"![Resource cost]({trace}_resource_cost.png)", ""]
    md += ["## Can the novelty be declared better?", "",
           "No measured claim that all novelties beat Paper Hybrid is supported by these outputs. Paper Hybrid has 0% simulated violations on both held-out traces, so no policy can reduce that rate. Several novelty arms also have higher modeled cost; isolated in-place resize has a lower cost point estimate on Azure but its paired interval includes zero, while World Cup shows no cost reduction. See the per-arm paired results in each input results directory. These are trace-simulator outcomes, not live production SLO or cost measurements.", ""]
    (out / "paper_output_comparison.md").write_text("\n".join(md), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"traces": traces, "performance_data": str((ROOT / args.performance).resolve()),
                                                              "paper_parameters": {"interval_seconds": 60, "lookback": 10,
                                                                                   "burst_window": 10, "burst_threshold": 5,
                                                                                   "burst_influence": .5, "slo_ms": 350,
                                                                                   "min_cpu_millicores": 600, "max_cpu_millicores": 950,
                                                                                   "train_fraction": .8},
                                                              "project_dtr_model_comparison_split": "80/20 grouped by load condition"}, indent=2), encoding="utf-8")
    print(f"Wrote paper-aligned tables, figures, and report to {out}")


if __name__ == "__main__":
    main()
