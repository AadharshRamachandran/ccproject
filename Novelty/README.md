# Predictive Hybrid Autoscaling — Novelty Extension

This directory extends the paper baseline with four deployable improvements while retaining the Bi-LSTM and decision-tree-regressor (DTR) reference path. Only components suitable for policy evaluation are included.

Read [NOVELTY.md](NOVELTY.md) for the rationale, safeguards, and evaluation criteria for all four novelties.

## Included configurations

| Component | Paper baseline | Novelty choices |
| --- | --- | --- |
| Workload forecast | Bi-LSTM | Quantile Bi-LSTM (P10/P50/P90) |
| Performance model | DTR | DTR only |
| Burst policy | Fixed Z-score | Validation-tuned, volatility-adaptive Z-score |
| CPU update | Rolling deployment | In-place Pod resize |
| Monitoring | 60-second fixed interval | Workload-adaptive interval |

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

## Real Kubernetes, Locust, and Prometheus validation

The ablation is an offline trace evaluation. It does not send traffic to Kubernetes and must be reported separately from real measurements. Use the following process to validate the paper baseline and each isolated novelty in a live cluster.

### 1. Build and deploy

```powershell
docker build -t novelty-hybrid-controller:latest -f .\Novelty\controller\Dockerfile .
docker build -t hybrid-house-price:latest .\Novelty\app
kubectl apply -f .\Novelty\k8s\namespace.yaml
kubectl apply -f .\Novelty\k8s\app.yaml
kubectl apply -f .\Novelty\k8s\servicemonitor.yaml
kubectl -n hybrid-autoscaling get pods,svc
```

For local testing, run these in separate terminals:

```powershell
kubectl -n hybrid-autoscaling port-forward service/house-price 8000:80
kubectl -n monitoring port-forward service/kube-prometheus-stack-prometheus 9090:9090
```

### 2. Create a real performance dataset

For every CPU level (`600`, `700`, `800`, `900`, `950` millicores) and several steady Locust loads, start Locust and collect Prometheus samples during the stable portion of the run.

```powershell
locust -f .\Novelty\locustfile.py --headless -u 30 -r 5 -t 8m --host http://localhost:8000

python .\Novelty\scripts\collect_performance_data.py `
  --prometheus http://localhost:9090 `
  --cpu 600 `
  --samples 5 --interval 60 `
  --out .\Novelty\data\collected\performance.csv
```

Repeat the same load schedule at every CPU level. The collector appends measured QPS, P95 response time, replicas, and CPU utilisation. Remove rows above the 350 ms SLO before training the DTR.

### 3. Train using only live data

```powershell
python .\Novelty\scripts\train_novelty.py `
  --workload .\Novelty\data\processed\worldcup98_minute.csv `
  --performance .\Novelty\data\collected\performance.csv `
  --models .\Novelty\models --epochs 80 --batch-size 512
```

### 4. Run controlled live comparisons

Apply the controller with `APPLY_CHANGES=false` first and inspect `/status`.

```powershell
kubectl apply -f .\Novelty\k8s\controller.yaml
kubectl -n hybrid-autoscaling port-forward service/novelty-hybrid-controller 8080:8080
```

For each run, change only one setting from the paper baseline, replay the identical Locust profile, and save a separate Prometheus export or observation log.

| Run | Changed setting |
| --- | --- |
| Paper baseline | `FORECASTER=bilstm`, `BURST_POLICY=paper_fixed`, `EXECUTION_BACKEND=rolling`, `ADAPTIVE_MONITORING=false` |
| Novelty 1 | `BURST_POLICY=tuned_adaptive` |
| Novelty 2 | `FORECASTER=quantile_bilstm` |
| Novelty 3 | `EXECUTION_BACKEND=inplace` |
| Novelty 4 | `ADAPTIVE_MONITORING=true` |

Enable `APPLY_CHANGES=true` only after confirming Prometheus connectivity and controller status. For every run, record P95 latency/SLO violations, mean utilisation, replica and CPU requests, resource cost, resize outcomes, restarts, and monitor cycles. These are the real-world results; do not combine them numerically with the offline ablation CSVs.

## Train, test, and evaluate

```powershell
python .\Novelty\scripts\train_novelty.py --workload .\Novelty\data\processed\worldcup98_minute.csv --performance .\Novelty\data\generated\performance.csv --models .\Novelty\models --epochs 80 --batch-size 512
python -m unittest discover -s Novelty\tests -v
python .\Novelty\ablation.py --workload .\Novelty\data\processed\worldcup98_minute.csv --performance .\Novelty\data\generated\performance.csv --out .\Novelty\results\worldcup98 --capacity-peak 1800 --epochs 80 --batch-size 512
```

For a quick smoke run:

```powershell
python .\Novelty\ablation.py --workload .\Novelty\data\processed\worldcup98_minute.csv --performance .\Novelty\data\generated\performance.csv --out .\tmp\smoke --max-points 200 --epochs 1 --batch-size 64 --bootstrap-samples 20 --only reactive,paper_hybrid
```

The ablation runs seven configurations: reactive and proactive controls, the paper hybrid, and one isolated experiment for each novelty. It writes per-configuration CSVs, `summary.csv`, `pairwise_vs_paper.csv`, `novelty_effects_vs_paper.csv`, and `report.md`. The effects file reports each novelty's increase or decrease relative to the paper hybrid; report cost and utilisation alongside SLO violation.

## Deploy to Kubernetes

```powershell
docker build -t novelty-hybrid-controller:latest -f .\Novelty\controller\Dockerfile .
docker build -t hybrid-house-price:latest .\Novelty\app
kubectl apply -f .\Novelty\k8s\namespace.yaml
kubectl apply -f .\Novelty\k8s\app.yaml
kubectl apply -f .\Novelty\k8s\controller.yaml
```

The controller starts with `APPLY_CHANGES=false`; inspect `/status` before enabling live changes. In-place resizing needs Kubernetes 1.35+ Linux nodes and the included `pods/resize` RBAC permission.
