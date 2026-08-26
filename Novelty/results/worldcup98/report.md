# Paper reproduction and novelty ablation

All latency/cost values are emulator estimates; validate finalists with the Kubernetes/Locust experiment.

Capacity calibration peak: `1800` requests per evaluation interval.

## Top configurations (SLO first, cost second)

| variant | forecaster_policy_eligible | performance_policy_eligible | slo_violation_pct | mean_utilization_pct | resource_cost | rolling_overlap_cost | monitor_cycles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| proactive_hpa_bilstm_fixed_cpu | True | True | 0.000 | 46.422 | 207.081 | 0.000 | 1440 |
| reactive_hpa_fixed_cpu | True | True | 0.069 | 50.553 | 190.608 | 0.000 | 1440 |
| novelty_3_inplace_resize | True | True | 0.694 | 75.816 | 124.056 | 0.000 | 1440 |
| paper_hybrid_bilstm_dtr_rolling_fixed_monitor | True | True | 1.528 | 75.766 | 129.226 | 10.423 | 1440 |
| novelty_1_tuned_adaptive_burst | True | True | 1.528 | 75.677 | 129.520 | 10.639 | 1440 |
| novelty_4_stacked_quantile_with_adaptive_cadence | True | True | 1.667 | 76.124 | 128.393 | 10.054 | 1336 |
| novelty_2_quantile_provisioning | True | True | 1.667 | 76.187 | 128.666 | 10.817 | 1440 |
| novelty_4_adaptive_monitoring | True | True | 1.736 | 75.840 | 127.330 | 6.859 | 805 |

A `False` eligibility flag means the corresponding component did not meet the held-out quality gate in `component_benchmark.csv`; treat its end-to-end row as diagnostic rather than a validated policy result.

## Paper-hybrid reference

- SLO violation: 1.528
- Resource cost: 129.226
- Mean utilization: 75.766%

See `pairwise_vs_paper.csv` for absolute deltas against the paper hybrid.