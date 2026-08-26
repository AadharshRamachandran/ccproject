# Predictive Hybrid Autoscaling — Paper Baseline

Reproducible implementation of the paper baseline: a Bi-LSTM workload forecast, fixed Z-score burst detection, a decision-tree performance model, and a rolling-update Kubernetes executor. The novelty extension is in [../Novelty](../Novelty/README.md).

## Setup

Run from the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\Paper\requirements.txt
$env:PYTHONPATH = "$PWD;$PWD\Paper"
```

Prepare data and create a development performance fixture:

```powershell
python .\Paper\scripts\prepare_worldcup98.py --input .\worldcup98-dataset\invocation_count.csv --out .\Paper\data\processed\worldcup98_minute.csv
python .\Paper\scripts\generate_performance_data.py --out .\Paper\data\generated\performance.csv
python .\Paper\scripts\train.py --workload .\Paper\data\processed\worldcup98_minute.csv --performance .\Paper\data\generated\performance.csv --models .\Paper\models
python -m unittest discover -s Paper\tests -v
```

The generated performance data is for pipeline development only. Do not mix it with live measurements or use it for deployment conclusions.

## Real Kubernetes, Locust, and Prometheus validation

Use this workflow to build an application-specific performance dataset and validate the paper controller. Store live measurements under `Paper/data/collected/`; this is intentionally separate from `data/generated/` and from ablation result folders.

### 1. Deploy the application and monitoring stack

Build and apply the manifests below. Ensure Prometheus scrapes the application's `/metrics` endpoint through the supplied ServiceMonitor before collecting data.

```powershell
docker build -t hybrid-house-price:latest .\Paper\app
kubectl apply -f .\Paper\k8s\namespace.yaml
kubectl apply -f .\Paper\k8s\app.yaml
kubectl apply -f .\Paper\k8s\servicemonitor.yaml
kubectl -n hybrid-autoscaling get pods,svc
```

For a local cluster, expose the application and Prometheus in separate terminals:

```powershell
kubectl -n hybrid-autoscaling port-forward service/house-price 8000:80
kubectl -n monitoring port-forward service/kube-prometheus-stack-prometheus 9090:9090
```

### 2. Collect real performance measurements

For each CPU request (`600`, `700`, `800`, `900`, and `950` millicores) and several stable Locust user counts, run Locust in one terminal. In a second terminal, collect five one-minute Prometheus samples while the load is stable.

```powershell
locust -f .\Paper\locustfile.py --headless -u 30 -r 5 -t 8m --host http://localhost:8000

python .\Paper\scripts\collect_performance_data.py `
  --prometheus http://localhost:9090 `
  --cpu 600 `
  --samples 5 --interval 60 `
  --out .\Paper\data\collected\performance.csv
```

Repeat with the same sequence of loads for every CPU level. The collector appends rows with `request_rate,cpu_millicores,slo_ms,replicas,utilization,response_ms`. Inspect the CSV and exclude rows where `response_ms` exceeds the 350 ms SLO before training.

### 3. Train and validate with measured data

```powershell
python .\Paper\scripts\train.py `
  --workload .\Paper\data\processed\worldcup98_minute.csv `
  --performance .\Paper\data\collected\performance.csv `
  --models .\Paper\models

docker build -t paper-hybrid-controller:latest -f .\Paper\controller\Dockerfile .
kubectl apply -f .\Paper\k8s\controller.yaml
```

Keep `APPLY_CHANGES=false` until the controller `/status` output and Prometheus queries are correct. Then enable it only for a controlled experiment, replay the same Locust scenario, and record P95 latency, SLO violations, utilisation, replica count, CPU requests, and scaling events.

## Kubernetes deployment

```powershell
docker build -t paper-hybrid-controller:latest -f .\Paper\controller\Dockerfile .
docker build -t hybrid-house-price:latest .\Paper\app
kubectl apply -f .\Paper\k8s\namespace.yaml
kubectl apply -f .\Paper\k8s\app.yaml
kubectl apply -f .\Paper\k8s\controller.yaml
```

Set `APPLY_CHANGES=true` only after checking the controller `/status` output and validating the deployment in a controlled environment.
