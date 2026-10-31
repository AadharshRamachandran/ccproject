# Azure Functions 2019 workload used for N1/N3 cross-trace evaluation

- Source: [Microsoft Azure Public Dataset, Azure Functions Trace 2019](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md)
- Data collection: July 2019; the release provides 14 daily files with per-function invocation counts for each of 1,440 minutes.
- Derivation: `scripts/prepare_azurefunctions2019.py` sums the per-minute counts across sampled functions for each day and concatenates the days in order. `azurefunctions2019_minute.csv` therefore contains 20,160 chronological minute-level total invocation counts.
- Evaluation: the 1,440-point cross-trace run uses a chronological 80/20 development/held-out split and scales the selected trace window's development peak to 20 requests/second, matching the measured calibration envelope. This is a trace-shape comparison, not a direct comparison of raw request magnitudes.
- Performance model: measured from the local Minikube house-price service using Locust and Prometheus, with the Kubernetes CPU HPA selecting replica counts. The same calibrated DTR is used for World Cup and Azure trace runs. Azure latency and resource-cost measurements are not present in the Azure trace.
- Interpretation: the offline simulator still estimates latency and cost; the measured DTR does not make those trace-level outcomes live cluster measurements. See `Novelty/data/collected/performance_training_report.json` for calibration coverage and held-out metrics.
- License and citation: Microsoft provides this data under CC-BY Attribution. Cite Mohammad Shahrad et al., “Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider,” USENIX ATC 2020.
