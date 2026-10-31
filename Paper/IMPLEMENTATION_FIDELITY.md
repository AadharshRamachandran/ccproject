# Paper Hybrid implementation and reproduction limits

The `Paper/` package is the canonical implementation of the controller in Vu,
Tran, and Kim (IEEE Access, 2022). `Novelty/` imports the same baseline
forecaster, fixed burst detector, DTR performance model, and hybrid provisioner.
Each novelty is enabled as a separate policy flag; a no-novelty run uses the
Paper Hybrid configuration.

## Controller behavior implemented from the paper

- Workload is forecast one step ahead from 10 prior observations using a
  bidirectional LSTM; the trace is split chronologically 80/20 and the model
  normalizes on the training range.
- Algorithm 1 uses a 10-point moving history, threshold 5, and influence 0.5.
  The detector is initialized from the last available measured workload points.
- The performance function maps request rate, CPU millicores, and SLO to
  replicas and average utilization using a normalized multi-output DTR.
  Supplied calibration rows above the SLO are excluded when `response_ms` is
  present. The comparison utility evaluates LR, RDF, SVR, XGB when installed,
  MLP, and DTR on a seeded 80/20 holdout; the controller uses the DTR reported
  as the paper's selected model.
- Algorithm 2 uses a 350 ms SLO and CPU choices of 600, 700, 800, 900, and
  950 mCPU. On a burst it uses 950 mCPU and the DTR replica recommendation;
  otherwise it chooses the CPU/replica candidate with maximum predicted
  utilization.
- Monitoring uses a 60-second fixed interval. Replica-only changes scale the
  active Deployment. A CPU change creates a second Deployment with the new
  resources, leaves the old one serving until the new one is ready, and then
  removes the old generation.
- Novelty arms retain the same Paper Hybrid DTR and provisioner unless the
  novelty explicitly changes that component. N1 changes burst detection, N2
  adds quantile-aware provisioning, N3 changes monitoring cadence, and N4
  selects in-place resizing. Their factorial combinations use the same shared
  baseline.

## What cannot be reproduced exactly from the supplied paper and repository

The article does not publish its original performance calibration rows, exact
trace subsets/scaling factors, full Bi-LSTM layer widths/training settings, or
the exact random split seed. The paper mentions a two-direction network and
reports a DTR winner, but those missing artifacts prevent bit-for-bit or
numerically identical reproduction. The code uses the project defaults
(32 hidden units per direction, Adam at 0.005, and seed 42) and the available
calibration table. `paper_training_report.json` records those choices and
candidate model metrics. Results from this project must be described as a
paper-method implementation on the supplied data, not as the paper's original
measured results.

The public algorithm text also describes both updating the moving baseline
with the influence-filtered value and appending the forecast to history. The
implementation follows the standard influence-filtered moving-window
detector described in the surrounding prose, which prevents a burst sample
from distorting later thresholds.
