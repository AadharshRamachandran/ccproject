# Paper reproduction and novelty ablation

All latency/cost values are emulator estimates; validate finalists with the Kubernetes/Locust experiment.

Performance data: `synthetic development fixture`. Synthetic development fixture; not live cluster measurements. Forecaster epochs: `10`. Seed: `42`. Paired block-bootstrap draws: `200`.

Capacity calibration peak: `1800` requests per evaluation interval.

## Top configurations (SLO first, cost second)

| variant | forecaster_policy_eligible | performance_policy_eligible | slo_violation_pct | mean_utilization_pct | resource_cost | rolling_overlap_cost | monitor_cycles | mean_monitor_interval_seconds | quantile_p10_p90_coverage_pct | forecast_mae_scaled | quantile_interval_safety_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proactive_hpa_bilstm_fixed_cpu | None | None | 0.000 | 67.796 | 410.400 | 0.000 | 1440 | 60.000 | nan | 88.470 | nan |
| burst_aware_hpa_bilstm_fixed_cpu | None | None | 0.000 | 67.434 | 413.022 | 0.000 | 1440 | 60.000 | nan | 88.470 | nan |
| reactive_hpa_fixed_cpu | None | None | 0.069 | 62.572 | 447.051 | 0.000 | 1440 | 60.000 | nan | 88.470 | nan |
| optimized_novelty_n2_quantile_n3_monitoring_n4_inplace | None | None | 0.139 | 92.064 | 302.106 | 0.000 | 958 | 90.219 | 55.278 | 57.988 | 2.000 |
| optimized_novelty_n2_quantile_n3_monitoring | None | None | 0.139 | 92.134 | 322.235 | 41.324 | 958 | 90.219 | 55.278 | 57.988 | 2.000 |
| optimized_novelty_n1_burst_n2_quantile_n3_monitoring | None | None | 0.139 | 86.871 | 349.425 | 51.016 | 1156 | 74.753 | 55.278 | 57.988 | 2.000 |
| optimized_novelty_n2_quantile_n4_inplace | None | None | 0.208 | 92.362 | 301.137 | 0.000 | 1440 | 60.000 | 55.278 | 57.988 | 2.000 |
| optimized_novelty_n1_burst_n2_quantile_n3_monitoring_n4_inplace | None | None | 0.208 | 86.718 | 324.774 | 0.000 | 1156 | 74.753 | 55.278 | 57.988 | 2.000 |
| novelty_2_quantile_provisioning | None | None | 0.208 | 92.430 | 330.787 | 60.510 | 1440 | 60.000 | 55.278 | 57.988 | 2.000 |
| optimized_novelty_n1_burst_n2_quantile_n4_inplace | None | None | 0.208 | 84.923 | 332.742 | 0.000 | 1440 | 60.000 | 55.278 | 57.988 | 2.000 |
| optimized_novelty_n1_burst_n2_quantile | None | None | 0.208 | 85.082 | 363.449 | 63.128 | 1440 | 60.000 | 55.278 | 57.988 | 2.000 |
| optimized_novelty_n1_burst_n4_inplace | None | None | 4.931 | 99.431 | 267.300 | 0.000 | 1440 | 60.000 | nan | 88.470 | nan |
| novelty_4_inplace_resize | None | None | 4.931 | 99.399 | 267.432 | 0.000 | 1440 | 60.000 | nan | 88.470 | nan |
| optimized_novelty_n3_monitoring_n4_inplace | None | None | 5.069 | 99.485 | 266.820 | 0.000 | 616 | 140.390 | nan | 88.470 | nan |
| optimized_novelty_n1_burst_n3_monitoring_n4_inplace | None | None | 5.069 | 99.478 | 266.856 | 0.000 | 616 | 140.390 | nan | 88.470 | nan |
| novelty_3_adaptive_monitoring | None | None | 5.486 | 99.448 | 270.631 | 7.622 | 616 | 140.390 | nan | 88.470 | nan |
| optimized_novelty_n1_burst_n3_monitoring | None | None | 5.486 | 99.440 | 270.713 | 7.714 | 616 | 140.390 | nan | 88.470 | nan |
| novelty_1_tuned_adaptive_burst | None | None | 5.764 | 99.352 | 273.931 | 13.262 | 1440 | 60.000 | nan | 88.470 | nan |
| paper_hybrid_bilstm_dtr_rolling_fixed_monitor | None | None | 5.764 | 99.320 | 274.018 | 13.171 | 1440 | 60.000 | nan | 88.470 | nan |

Novelty SLO point estimates range from 0.139% to 5.764% (Paper Hybrid: 5.764%). At least one arm has paired intervals below zero on both outcomes.
Small cost differences are not evidence of a reliable gain unless the paired confidence interval excludes zero.

## Paper-hybrid reference

- SLO violation: 5.764%
- Resource cost: 274.018
- Mean utilization: 99.320%

## Paired block-bootstrap differences vs Paper Hybrid

Negative deltas favor the novelty. Intervals resample the same chronological blocks in both arms.

| Novelty | SLO delta (pp), 95% CI | Cost delta, 95% CI |
| --- | ---: | ---: |
| novelty_1_tuned_adaptive_burst | +0.000 [+0.000, +0.000] | -0.086 [-0.312, +0.137] |
| novelty_2_quantile_provisioning | -5.556 [-7.500, -3.611] | +56.770 [+54.461, +59.478] |
| novelty_3_adaptive_monitoring | -0.278 [-0.698, +0.139] | -3.386 [-4.734, -2.367] |
| novelty_4_inplace_resize | -0.833 [-1.319, -0.276] | -6.586 [-7.686, -5.613] |
| optimized_novelty_n1_burst_n2_quantile | -5.556 [-7.365, -3.958] | +89.432 [+84.730, +94.013] |
| optimized_novelty_n1_burst_n3_monitoring | -0.278 [-0.764, +0.071] | -3.305 [-4.567, -2.116] |
| optimized_novelty_n1_burst_n4_inplace | -0.833 [-1.462, -0.345] | -6.718 [-7.771, -5.531] |
| optimized_novelty_n2_quantile_n3_monitoring | -5.625 [-7.571, -3.679] | +48.217 [+45.977, +50.865] |
| optimized_novelty_n2_quantile_n4_inplace | -5.556 [-7.988, -3.750] | +27.119 [+25.163, +28.936] |
| optimized_novelty_n3_monitoring_n4_inplace | -0.694 [-1.321, -0.139] | -7.198 [-8.372, -5.812] |
| optimized_novelty_n1_burst_n2_quantile_n3_monitoring | -5.625 [-7.571, -3.679] | +75.407 [+71.627, +79.529] |
| optimized_novelty_n1_burst_n2_quantile_n4_inplace | -5.556 [-7.502, -3.957] | +58.724 [+55.115, +63.193] |
| optimized_novelty_n1_burst_n3_monitoring_n4_inplace | -0.694 [-1.319, -0.069] | -7.162 [-8.318, -5.728] |
| optimized_novelty_n2_quantile_n3_monitoring_n4_inplace | -5.625 [-7.571, -3.818] | +28.088 [+26.391, +29.767] |
| optimized_novelty_n1_burst_n2_quantile_n3_monitoring_n4_inplace | -5.556 [-7.714, -3.819] | +50.756 [+46.921, +54.438] |

## Multi-novelty combinations versus Paper Hybrid

Each row names the novelty factors included in that configuration. These are combined arms, not isolated one-factor effects. Every comparison uses the same held-out trace and emulator assumptions.

| Combination | SLO violations | SLO delta (pp) | Mean utilization | Resource cost | Cost delta (%) | Rolling overlap cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| optimized_novelty_n1_burst_n2_quantile | 0.208% | -5.556 | 85.082% | 363.449 | +32.64% | 63.128 |
| optimized_novelty_n1_burst_n3_monitoring | 5.486% | -0.278 | 99.440% | 270.713 | -1.21% | 7.714 |
| optimized_novelty_n1_burst_n4_inplace | 4.931% | -0.833 | 99.431% | 267.300 | -2.45% | 0.000 |
| optimized_novelty_n2_quantile_n3_monitoring | 0.139% | -5.625 | 92.134% | 322.235 | +17.60% | 41.324 |
| optimized_novelty_n2_quantile_n4_inplace | 0.208% | -5.556 | 92.362% | 301.137 | +9.90% | 0.000 |
| optimized_novelty_n3_monitoring_n4_inplace | 5.069% | -0.694 | 99.485% | 266.820 | -2.63% | 0.000 |
| optimized_novelty_n1_burst_n2_quantile_n3_monitoring | 0.139% | -5.625 | 86.871% | 349.425 | +27.52% | 51.016 |
| optimized_novelty_n1_burst_n2_quantile_n4_inplace | 0.208% | -5.556 | 84.923% | 332.742 | +21.43% | 0.000 |
| optimized_novelty_n1_burst_n3_monitoring_n4_inplace | 5.069% | -0.694 | 99.478% | 266.856 | -2.61% | 0.000 |
| optimized_novelty_n2_quantile_n3_monitoring_n4_inplace | 0.139% | -5.625 | 92.064% | 302.106 | +10.25% | 0.000 |
| optimized_novelty_n1_burst_n2_quantile_n3_monitoring_n4_inplace | 0.208% | -5.556 | 86.718% | 324.774 | +18.52% | 0.000 |

## Interpretation limits

The performance table is a synthetic development fixture; it is not measured cluster data. The in-place arm removes rolling overlap cost by construction and does not model PodResize latency, readiness delay, or transient CPU mismatch. Treat its small cost delta as a simulator artifact until validated against Kubernetes/Locust measurements. Component eligibility was not benchmarked in this run.

See `pairwise_vs_paper.csv` for point effects and paired confidence intervals.