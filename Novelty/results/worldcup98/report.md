# Paper reproduction and novelty ablation

All latency/cost values are emulator estimates; validate finalists with the Kubernetes/Locust experiment.

Capacity calibration peak: `1800` requests per evaluation interval.

## Top configurations (SLO first, cost second)

| variant | forecaster_policy_eligible | performance_policy_eligible | slo_violation_pct | mean_utilization_pct | resource_cost | rolling_overlap_cost | monitor_cycles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| proactive_hpa_bilstm_fixed_cpu | True | True | 0.000 | 47.570 | 202.236 | 0.000 | 1440 |
| reactive_hpa_fixed_cpu | True | True | 0.069 | 50.553 | 190.608 | 0.000 | 1440 |
| novelty_3_inplace_resize | True | True | 1.181 | 77.095 | 121.806 | 0.000 | 1440 |
| novelty_4_stacked_quantile_with_adaptive_cadence | True | True | 1.250 | 74.152 | 132.293 | 10.714 | 1302 |
| novelty_2_quantile_provisioning | True | True | 1.250 | 74.236 | 132.684 | 11.808 | 1440 |
| paper_hybrid_bilstm_dtr_rolling_fixed_monitor | True | True | 1.944 | 77.026 | 127.404 | 11.280 | 1440 |
| novelty_1_tuned_adaptive_burst | True | True | 1.944 | 76.954 | 127.616 | 11.405 | 1440 |
| novelty_4_adaptive_monitoring | True | True | 2.153 | 77.039 | 125.344 | 7.178 | 805 |

A `False` eligibility flag means the corresponding component did not meet the held-out quality gate in `component_benchmark.csv`; treat its end-to-end row as diagnostic rather than a validated policy result.

## Paper-hybrid reference

- SLO violation: 1.944
- Resource cost: 127.404
- Mean utilization: 77.026%

See `pairwise_vs_paper.csv` for absolute deltas against the paper hybrid.