# Paper reproduction and novelty ablation

All latency/cost values are emulator estimates; validate finalists with the Kubernetes/Locust experiment.

Capacity calibration peak: `1800` requests per evaluation interval.

## Top configurations (SLO first, cost second)

| variant | forecaster_policy_eligible | performance_policy_eligible | slo_violation_pct | mean_utilization_pct | resource_cost | rolling_overlap_cost | monitor_cycles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reactive_hpa_fixed_cpu | True | True | 2.500 | 57.156 | 10.887 | 0.000 | 40 |
| paper_hybrid_bilstm_dtr_rolling_fixed_monitor | True | True | 85.000 | 100.000 | 4.312 | 0.118 | 40 |

A `False` eligibility flag means the corresponding component did not meet the held-out quality gate in `component_benchmark.csv`; treat its end-to-end row as diagnostic rather than a validated policy result.

## Paper-hybrid reference

- SLO violation: 85.000
- Resource cost: 4.312
- Mean utilization: 100.000%

See `pairwise_vs_paper.csv` for absolute deltas against the paper hybrid.