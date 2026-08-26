# Predictive Hybrid Autoscaling — Reproducing and Extending Vu, Tran, and Kim (2022)

This repository contains the complete implementation of a **Predictive Hybrid Autoscaling** system for containerized applications. It reproduces the control baseline from the paper **"Stochastic Resource Provisioning for Containerized Multi-Tier Web Services in Clouds"** (Vu, Tran, and Kim, 2022) and introduces four major novel architectural extensions evaluated via a systematic ablation study.

---

## 📁 Repository Directory Structure

```text
├── README.md                      # Main project guide (this file)
├── Paper/                         # Baseline control implementation
│   ├── app/                       # House Price Prediction Benchmark Application
│   ├── controller/                # Controller (Prometheus query -> Decision -> Actuate)
│   ├── k8s/                       # Kubernetes deployment manifests
│   ├── scripts/                   # Telemetry collection, trace preparation, and training
│   ├── src/                       # Core python modules (forecaster, performance, scaler, executor)
│   ├── tests/                     # Unit test suite (8/8 passing)
│   └── PROJECT_FLOW.md            # Detailed end-to-end data/control flow of Paper baseline
│
├── Novelty/                       # Extended novelty implementation & ablation
│   ├── app/                       # Benchmark application (mirrors Paper/app)
│   ├── controller/                # Controller extending baseline with novelty environments
│   ├── k8s/                       # Kubernetes manifests supporting Pod In-Place Resize
│   ├── scripts/                   # Novelty training, prepare trace, and components benchmark
│   ├── src/                       # Novelty components (GPs, Quantiles, In-Place executor)
│   ├── tests/                     # Novelty-specific unit tests (8/8 passing)
│   ├── ablation.py                # Top-level ablation study runner
│   └── PROJECT_FLOW.md            # Detailed novelty extensions and ablation hypotheses
│
└── worldcup98-dataset/            # FIFA World Cup 1998 request trace repository
```

---

## 🛠️ Prerequisites & Local Setup

### 1. Requirements
* **Operating System:** Windows 10/11 with PowerShell (Administrator recommended for Docker/Minikube)
* **Python Version:** Python 3.12 (standard CPython distribution)
* **Containerization:** Docker Desktop with Kubernetes enabled (or Minikube)
* **Command-line tools:** `kubectl`, `helm`
* **Optional:** `locust` (for live workload generation)

### 2. Environment Setup
From the root of the project, initialize a single shared Python virtual environment:

```powershell
# Create the virtual environment
py -3.12 -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements (shared packages: PyTorch CPU, Scikit-Learn, Pandas, FastAPI, Uvicorn, etc.)
pip install --no-cache-dir --timeout 120 --retries 5 -r .\Paper\requirements.txt
```

### 3. Setting PYTHONPATH
To ensure Python imports resolve correctly across folders, set the `PYTHONPATH` whenever starting a new PowerShell session:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\Paper"
```

---

## 🚀 End-to-End Workflow

### Step 1 — Prepare the Workload Trace
Convert the raw FIFA World Cup 1998 counts to a minute-level workload trace:

```powershell
# Convert trace for Paper
python .\Paper\scripts\prepare_worldcup98.py `
  --input .\worldcup98-dataset\invocation_count.csv `
  --out .\Paper\data\processed\worldcup98_minute.csv

# Convert trace for Novelty
python .\Novelty\scripts\prepare_worldcup98.py `
  --input .\worldcup98-dataset\invocation_count.csv `
  --out .\Novelty\data\processed\worldcup98_minute.csv
```

### Step 2 — Benchmark Component Quality
Evaluate the quality of forecaster and performance models on held-out test data:

```powershell
python .\Novelty\scripts\benchmark_components.py `
  --workload .\Novelty\data\processed\worldcup98_minute.csv `
  --performance .\Novelty\data\generated\performance.csv `
  --skip-forecasters
```

### Step 3 — Train Models
Train the workload forecasting models and application performance models:

```powershell
# Train Paper baseline models
python .\Paper\scripts\train.py `
  --workload .\Paper\data\processed\worldcup98_minute.csv `
  --performance .\Paper\data\generated\performance.csv

# Train Novelty models (Quantile LSTM, PatchTST, Gaussian Processes)
python .\Novelty\scripts\train_novelty.py `
  --workload .\Novelty\data\processed\worldcup98_minute.csv `
  --performance .\Novelty\data\generated\performance.csv `
  --models .\Novelty\models
```

### Step 4 — Run Ablation Study
Run the emulator-based ablation study to compare every novelty configuration against baseline benchmarks:

```powershell
python .\Novelty\ablation.py `
  --workload .\Novelty\data\processed\worldcup98_minute.csv `
  --performance .\Novelty\data\generated\performance.csv `
  --out .\Novelty\results\worldcup98 `
  --capacity-peak 1800 `
  --skip-forecasters
```

---

## 📖 Sub-Folder Navigation Guides

For detailed setup instructions, Kubernetes manifest deployment commands, Prometheus configuration via Helm, and telemetry collection workflows, refer to the guides below:

* **[Paper Baseline Guide (Control)](file:///c:/Users/HP/OneDrive/Desktop/Project/Cloud%20computing%20project/Paper/README.md):** Explains how to replicate the paper's default replacement-deployment rolling updates, DTR models, and 60-second polling loops.
* **[Novelty Extension Guide](file:///c:/Users/HP/OneDrive/Desktop/Project/Cloud%20computing%20project/Novelty/README.md):** Explains the implementation details of all 4 novelties, in-place container resizing (requires Kubernetes 1.35+), adaptive monitoring cadence, and GP models.
