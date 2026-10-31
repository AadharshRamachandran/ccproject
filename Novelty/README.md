# Predictive Hybrid Autoscaling — Novelty Extension

This directory extends the paper baseline with four deployable improvements while retaining the Bi-LSTM and decision-tree-regressor (DTR) reference path. Only components suitable for policy evaluation are included.

Read [NOVELTY.md](NOVELTY.md) for the rationale, safeguards, and evaluation criteria for all four novelties.

## Included configurations

| Component | Paper baseline | Novelty choices |
| --- | --- | --- |
| Workload forecast | Bi-LSTM | Quantile Bi-LSTM (P10/P50/P90) |
| Performance model | DTR | DTR only |
| N1 burst policy | Fixed Z-score | Validation-tuned, volatility-adaptive Z-score |
| N2 provisioning | Bi-LSTM point forecast + DTR | Quantile Bi-LSTM interval + risk-aware DTR provisioning |
| N3 monitoring | 60-second fixed interval | Workload-adaptive interval |
| N4 CPU update | Rolling deployment | In-place Pod resize |

## Setup

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\Novelty\requirements.txt
$env:PYTHONPATH = "$PWD;$PWD\Paper"
```

Prepare the World Cup workload trace (clone its repository first if needed):

```powershell
git clone https://github.com/nimamahmoudi/worldcup98-dataset .\worldcup98-dataset
python .\Novelty\scripts\prepare_worldcup98.py --input .\worldcup98-dataset\invocation_count.csv --out .\Novelty\data\processed\worldcup98_minute.csv
python .\Novelty\scripts\generate_performance_data.py --out .\Novelty\data\generated\performance.csv
```

The generated performance file is a development fixture. Keep live measurements in `Novelty/data/collected/performance.csv`; do not mix them with `data/generated/` or with the offline ablation result folders.

The canonical baseline implementation is in `Paper/src/`: Paper Hybrid uses the published fixed burst detector, normalized multi-output DTR provisioner, 60-second monitoring interval, and rolling update executor. `Novelty/controller/main.py` starts in this same no-novelty baseline policy and enables N1–N4 only through explicit flags. The offline ablation now calls the canonical Paper Hybrid provisioner for its reference decisions. Every policy arm loads the same measured DTR; it does not substitute the novelty's quantile model unless N2 is enabled. The measured DTR was trained from the Locust/Prometheus calibration rows described by `Novelty/models/performance_training_report.json`. Trace-specific BiLSTM and quantile-BiLSTM forecasters are in `Novelty/models/worldcup98/` and `Novelty/models/azurefunctions2019/`. See `Paper/IMPLEMENTATION_FIDELITY.md` for limits: unavailable original paper datasets and unspecified model parameters prevent a literal numerical replication.

## Trace replay and live Kubernetes dashboard

The root dashboard keeps offline simulator estimates separate from live readings. Select FIFA World Cup 1998 or Azure Functions 2019 at `/`; both use a chronological 80/20 split. The `/live` view sends a scaled, held-out trace profile to the deployed benchmark using Locust and reads request rate, P95 latency, SLO violations, container CPU and ready replicas from Prometheus. Trace-profile replay is real Kubernetes traffic, but it is still a benchmark generated from a public trace, not the original production services.

### Start the live stack

The local Prometheus installation must discover the included ServiceMonitor. From separate terminals, start the app and Prometheus port-forwards after applying the manifests. The dashboard itself runs on the host and does not need a port-forward.

```powershell
docker build -t hybrid-house-price:latest .\Novelty\app
docker build -t novelty-hybrid-controller:latest -f .\Novelty\controller\Dockerfile .
kubectl apply -f .\Novelty\k8s\namespace.yaml
kubectl apply -f .\Novelty\k8s\app.yaml
kubectl apply -f .\Novelty\k8s\servicemonitor.yaml
kubectl apply -f .\Novelty\k8s\controller.yaml
kubectl -n hybrid-autoscaling get pods,svc
```

In separate terminals:

```powershell
kubectl -n hybrid-autoscaling port-forward service/house-price 8001:80
kubectl -n monitoring port-forward service/kube-prometheus-stack-prometheus 9090:9090
kubectl -n hybrid-autoscaling port-forward service/novelty-hybrid-controller 8080:8080
```

Run the dashboard from the repository root:

```powershell
$env:DASHBOARD_APP_URL = "http://127.0.0.1:8001"
python -m uvicorn Novelty.app.web_dashboard:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/live`. Before each live arm the page checks the app, controller, Prometheus API, Kubernetes Deployment, and HPA state. With actuation enabled it restores the benchmark to one 600m replica and removes stale rolling-update generations before starting the selected arm. The checkbox explicitly opts in to changes to the `hybrid-autoscaling/house-price` benchmark Deployment. Leave it unchecked for read-only controller recommendations.

### Run the full matched factorial

The runner uses one measured DTR for both trace-specific forecasters and loops through Paper Hybrid, N1–N4, and all eleven combinations on both traces. Each arm gets the same sampled held-out profile and its own raw run directory.

```powershell
python .\Novelty\scripts\run_live_trace_experiments.py `
  --max-points 30 --tick-seconds 60 --apply-changes
```

The default full run takes roughly 16 hours of Locust profile time, plus deployment resets and warm-up. For a short wiring check, reduce `--max-points` (minimum 10); the first ten monitoring cycles warm the forecaster and are not full-policy observations. Prometheus scrapes every 15 seconds and the default paper monitoring interval is 60 seconds. Increase `--tick-seconds` for more stable P95/SLO estimates.

Each run is saved under `Novelty/results/live_kubernetes/` with `run_metadata.json`, `prometheus_samples.csv`, a per-request `locust_request_events.csv`, Locust logs, and request-history CSVs. The SLO violation percentage is computed from individual Locust requests: response time above 350 ms or a failed request counts as a violation. The app exports an exact 350 ms Prometheus histogram bucket for live charts. The live record also reports measured request rate, request P95, CPU use/request ratio, ready replicas, controller recommendations and DTR-envelope coverage. CPU allocation-seconds are a resource proxy; they are not billed cloud cost. These observations are live benchmark measurements and must remain separate from the trace simulator results. Runs shorter than 30 points are wiring checks and are marked invalid for matched policy comparison because the controller's 10-observation forecast warm-up leaves too little post-warm-up evaluation.

N4 requires Kubernetes 1.35+ with working Pod resize support. Runs that request N4 fail preflight on older clusters. The dashboard also refuses to start if an HPA controls the benchmark Deployment, so it cannot silently compete with the hybrid controller.

## Train, test, and evaluate

### Reproduce the paper's tables and figures

Generate the reported paper tables (forecasting, performance-model errors, parameters, SLO/utilization zones, and burst-case cost), Figures 1–7, novelty overlays, per-policy CSVs, and a full four-factor ablation. The default protocol uses the final chronological workload window and a 20% held-out suffix:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\Paper"
python .\Novelty\scripts\generate_all_paper_outputs.py `
  --out .\Novelty\results\paper_outputs_complete `
  --window 7200 --window-mode tail `
  --capacity-peak 1800 --epochs 10 --batch-size 512 `
  --bootstrap-samples 200 --seed 42
```

Open `Novelty/results/paper_outputs_complete/PAPER_OUTPUTS.md` for the combined report. Its `windows/` folder contains the exact trace slices used, and each trace's `*_ablation/` folder contains the policy time series and paired comparisons. Tables 1 and 2 include both the paper-reported values and project-computed values where applicable. The Figure 1–7 images and raw tables are saved beside the report.

The run uses a synthetic performance fixture for offline development. Its latency and cost are simulator estimates; they do not establish real-cluster superiority. The combined novelty's full-trace result in `Novelty/results/factorial_full_trace_seed42/` has lower SLO and cost point estimates than Paper Hybrid, but the SLO paired confidence interval touches zero and the in-place cost advantage depends on simulated resize accounting. Validate with matched Kubernetes/Locust runs before making performance claims.

The generator evaluates Paper Hybrid, the standard comparison baselines, each of the four isolated novelties, and all 11 multi-novelty combinations. It uses a chronological tail window by default; `--window-mode burst` is an optional diagnostic and should not be mixed with chronological results.

```powershell
python .\Novelty\scripts\train_novelty.py --workload .\Novelty\data\processed\worldcup98_minute.csv --performance .\Novelty\data\generated\performance.csv --models .\Novelty\models --epochs 80 --batch-size 512
python -m unittest discover -s Novelty\tests -v
python .\Novelty\ablation.py --workload .\Novelty\data\processed\worldcup98_minute.csv --performance .\Novelty\data\collected\performance_measured_inslo.csv --out .\Novelty\results\worldcup98_measured_dtr_1440 --max-points 1440 --capacity-peak 20 --epochs 80 --batch-size 256
```

For a quick smoke run:

```powershell
python .\Novelty\ablation.py --workload .\Novelty\data\processed\worldcup98_minute.csv --performance .\Novelty\data\generated\performance.csv --out .\tmp\smoke --max-points 200 --epochs 1 --batch-size 64 --bootstrap-samples 20 --only reactive,paper_hybrid
```

The ablation includes Paper Hybrid, four isolated novelty arms, and all 11 multi-novelty combinations (the full 2^4 novelty factorial). N1 detects positive one-step forecast residuals; N2 is quantile provisioning; N3 is adaptive monitoring; N4 is in-place resize. It uses the full trace by default (`--max-points 0`). Quantile safety gain and adaptive-monitoring cadence are selected on validation and then held fixed across arms. The runner writes per-configuration CSVs, a summary, paired effects, run metadata, and `report.md`.

Historical offline result `Novelty/results/factorial_full_trace_seed42/` used a synthetic performance fixture; do not interpret it as the current measured-DTR result. The current 1,440-point measured-DTR results are in `Novelty/results/worldcup98_measured_dtr_1440/` and `Novelty/results/azurefunctions2019_measured_dtr_1440/`. Both use the same measured, in-SLO calibration data and `--capacity-peak 20` requests/second, within the observed calibration envelope. The paper simulator still estimates trace latency/cost: these are not live cluster SLO measurements. Paper Hybrid has 0% simulated SLO violations on both held-out trace windows, so no arm can lower its violation rate in these runs. The DTR grouped holdout scores are still weak (negative utilization R²); treat its recommendations as provisional and do not extrapolate outside the measured range. In Azure the isolated in-place arm has a 1.05% lower cost point estimate, but its paired 95% cost interval includes zero. There is no statistically supported joint SLO-and-cost improvement in this evaluation. See each result directory's `report.md`, `summary.csv`, and `pairwise_vs_paper.csv`.

## Deploy to Kubernetes

```powershell
docker build -t novelty-hybrid-controller:latest -f .\Novelty\controller\Dockerfile .
docker build -t hybrid-house-price:latest .\Novelty\app
kubectl apply -f .\Novelty\k8s\namespace.yaml
kubectl apply -f .\Novelty\k8s\app.yaml
kubectl apply -f .\Novelty\k8s\controller.yaml
```

The controller starts with `APPLY_CHANGES=false`; inspect `/status` before enabling live changes. In-place resizing needs Kubernetes 1.35+ Linux nodes and the included `pods/resize` RBAC permission.
