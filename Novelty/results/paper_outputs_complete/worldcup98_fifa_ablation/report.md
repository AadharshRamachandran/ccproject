# Paper reproduction and novelty ablation

All latency/cost values are emulator estimates; validate finalists with the Kubernetes/Locust experiment.

Performance data: `synthetic development fixture`. Synthetic development fixture; not live cluster measurements. Forecaster epochs: `10`. Seed: `42`. Paired block-bootstrap draws: `200`.

Capacity calibration peak: `1800` requests per evaluation interval.

## Top configurations (SLO first, cost second)

| variant | forecaster_policy_eligible | performance_policy_eligible | slo_violation_pct | mean_utilization_pct | resource_cost | rolling_overlap_cost | monitor_cycles | mean_monitor_interval_seconds | quantile_p10_p90_coverage_pct | forecast_mae_scaled | quantile_interval_safety_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| novelty_4_inplace_resize | None | None | 0.000 | 60.646 | 155.520 | 0.000 | 1440 | 60.000 | nan | 266.296 | nan |
| optimized_novelty_n1_burst_n4_inplace | None | None | 0.000 | 60.646 | 155.520 | 0.000 | 1440 | 60.000 | nan | 266.296 | nan |
| optimized_novelty_n3_monitoring_n4_inplace | None | None | 0.000 | 60.646 | 155.520 | 0.000 | 1126 | 76.747 | nan | 266.296 | nan |
| optimized_novelty_n1_burst_n3_monitoring_n4_inplace | None | None | 0.000 | 60.646 | 155.520 | 0.000 | 1126 | 76.747 | nan | 266.296 | nan |
| proactive_hpa_bilstm_fixed_cpu | None | None | 0.000 | 38.329 | 246.240 | 0.000 | 1440 | 60.000 | nan | 266.296 | nan |
| burst_aware_hpa_bilstm_fixed_cpu | None | None | 0.000 | 38.329 | 246.240 | 0.000 | 1440 | 60.000 | nan | 266.296 | nan |
| novelty_3_adaptive_monitoring | None | None | 0.069 | 60.667 | 160.358 | 9.835 | 1126 | 76.747 | nan | 266.296 | nan |
| optimized_novelty_n1_burst_n3_monitoring | None | None | 0.069 | 60.667 | 160.358 | 9.835 | 1126 | 76.747 | nan | 266.296 | nan |
| paper_hybrid_bilstm_dtr_rolling_fixed_monitor | None | None | 0.069 | 60.667 | 161.071 | 11.261 | 1440 | 60.000 | nan | 266.296 | nan |
| novelty_1_tuned_adaptive_burst | None | None | 0.069 | 60.667 | 161.071 | 11.261 | 1440 | 60.000 | nan | 266.296 | nan |
| reactive_hpa_fixed_cpu | None | None | 0.069 | 50.553 | 190.608 | 0.000 | 1440 | 60.000 | nan | 266.296 | nan |
| optimized_novelty_n1_burst_n2_quantile_n4_inplace | None | None | 0.347 | 69.509 | 135.570 | 0.000 | 1440 | 60.000 | 88.542 | 89.542 | 1.000 |
| optimized_novelty_n2_quantile_n4_inplace | None | None | 0.347 | 69.504 | 135.588 | 0.000 | 1440 | 60.000 | 88.542 | 89.542 | 1.000 |
| optimized_novelty_n1_burst_n2_quantile_n3_monitoring_n4_inplace | None | None | 0.347 | 69.394 | 135.846 | 0.000 | 1412 | 61.191 | 88.542 | 89.542 | 1.000 |
| optimized_novelty_n2_quantile_n3_monitoring_n4_inplace | None | None | 0.347 | 69.390 | 135.864 | 0.000 | 1412 | 61.191 | 88.542 | 89.542 | 1.000 |
| optimized_novelty_n2_quantile_n3_monitoring | None | None | 0.486 | 66.491 | 152.748 | 21.690 | 1412 | 61.191 | 88.542 | 89.542 | 1.500 |
| optimized_novelty_n1_burst_n2_quantile_n3_monitoring | None | None | 0.486 | 66.494 | 152.774 | 21.779 | 1412 | 61.191 | 88.542 | 89.542 | 1.500 |
| novelty_2_quantile_provisioning | None | None | 0.486 | 66.645 | 152.803 | 22.520 | 1440 | 60.000 | 88.542 | 89.542 | 1.500 |
| optimized_novelty_n1_burst_n2_quantile | None | None | 0.486 | 66.647 | 152.830 | 22.609 | 1440 | 60.000 | 88.542 | 89.542 | 1.500 |

Novelty SLO point estimates range from 0.000% to 0.486% (Paper Hybrid: 0.069%). No arm has paired 95% intervals below zero for both SLO and cost; the data do not establish a joint improvement.
Small cost differences are not evidence of a reliable gain unless the paired confidence interval excludes zero.

## Paper-hybrid reference

- SLO violation: 0.069%
- Resource cost: 161.071
- Mean utilization: 60.667%

## Paired block-bootstrap differences vs Paper Hybrid

Negative deltas favor the novelty. Intervals resample the same chronological blocks in both arms.

| Novelty | SLO delta (pp), 95% CI | Cost delta, 95% CI |
| --- | ---: | ---: |
| novelty_1_tuned_adaptive_burst | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| novelty_2_quantile_provisioning | +0.417 [+0.139, +0.766] | -8.268 [-10.603, -5.356] |
| novelty_3_adaptive_monitoring | +0.000 [+0.000, +0.000] | -0.713 [-1.081, -0.388] |
| novelty_4_inplace_resize | -0.069 [-0.208, +0.000] | -5.551 [-6.351, -4.881] |
| optimized_novelty_n1_burst_n2_quantile | +0.417 [+0.139, +0.694] | -8.242 [-10.386, -5.952] |
| optimized_novelty_n1_burst_n3_monitoring | +0.000 [+0.000, +0.000] | -0.713 [-1.146, -0.346] |
| optimized_novelty_n1_burst_n4_inplace | -0.069 [-0.208, +0.000] | -5.551 [-6.417, -4.924] |
| optimized_novelty_n2_quantile_n3_monitoring | +0.417 [+0.139, +0.764] | -8.323 [-10.491, -5.982] |
| optimized_novelty_n2_quantile_n4_inplace | +0.278 [-0.069, +0.625] | -25.483 [-26.957, -23.718] |
| optimized_novelty_n3_monitoring_n4_inplace | -0.069 [-0.208, +0.000] | -5.551 [-6.221, -4.946] |
| optimized_novelty_n1_burst_n2_quantile_n3_monitoring | +0.417 [+0.139, +0.764] | -8.297 [-10.299, -5.955] |
| optimized_novelty_n1_burst_n2_quantile_n4_inplace | +0.278 [+0.000, +0.694] | -25.501 [-26.995, -23.569] |
| optimized_novelty_n1_burst_n3_monitoring_n4_inplace | -0.069 [-0.208, +0.000] | -5.551 [-6.416, -4.989] |
| optimized_novelty_n2_quantile_n3_monitoring_n4_inplace | +0.278 [-0.069, +0.625] | -25.207 [-27.203, -23.361] |
| optimized_novelty_n1_burst_n2_quantile_n3_monitoring_n4_inplace | +0.278 [+0.000, +0.625] | -25.225 [-27.002, -23.467] |

## Multi-novelty combinations versus Paper Hybrid

Each row names the novelty factors included in that configuration. These are combined arms, not isolated one-factor effects. Every comparison uses the same held-out trace and emulator assumptions.

| Combination | SLO violations | SLO delta (pp) | Mean utilization | Resource cost | Cost delta (%) | Rolling overlap cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| optimized_novelty_n1_burst_n2_quantile | 0.486% | +0.417 | 66.647% | 152.830 | -5.12% | 22.609 |
| optimized_novelty_n1_burst_n3_monitoring | 0.069% | +0.000 | 60.667% | 160.358 | -0.44% | 9.835 |
| optimized_novelty_n1_burst_n4_inplace | 0.000% | -0.069 | 60.646% | 155.520 | -3.45% | 0.000 |
| optimized_novelty_n2_quantile_n3_monitoring | 0.486% | +0.417 | 66.491% | 152.748 | -5.17% | 21.690 |
| optimized_novelty_n2_quantile_n4_inplace | 0.347% | +0.278 | 69.504% | 135.588 | -15.82% | 0.000 |
| optimized_novelty_n3_monitoring_n4_inplace | 0.000% | -0.069 | 60.646% | 155.520 | -3.45% | 0.000 |
| optimized_novelty_n1_burst_n2_quantile_n3_monitoring | 0.486% | +0.417 | 66.494% | 152.774 | -5.15% | 21.779 |
| optimized_novelty_n1_burst_n2_quantile_n4_inplace | 0.347% | +0.278 | 69.509% | 135.570 | -15.83% | 0.000 |
| optimized_novelty_n1_burst_n3_monitoring_n4_inplace | 0.000% | -0.069 | 60.646% | 155.520 | -3.45% | 0.000 |
| optimized_novelty_n2_quantile_n3_monitoring_n4_inplace | 0.347% | +0.278 | 69.390% | 135.864 | -15.65% | 0.000 |
| optimized_novelty_n1_burst_n2_quantile_n3_monitoring_n4_inplace | 0.347% | +0.278 | 69.394% | 135.846 | -15.66% | 0.000 |

## Interpretation limits

The performance table is a synthetic development fixture; it is not measured cluster data. The in-place arm removes rolling overlap cost by construction and does not model PodResize latency, readiness delay, or transient CPU mismatch. Treat its small cost delta as a simulator artifact until validated against Kubernetes/Locust measurements. Component eligibility was not benchmarked in this run.

See `pairwise_vs_paper.csv` for point effects and paired confidence intervals.