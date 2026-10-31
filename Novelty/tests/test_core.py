"""Novelty-local smoke tests; Paper/tests remains the baseline test suite."""
import unittest

import numpy as np

from Novelty.src.burst import AdaptiveBurstDetector, BurstParameters
from Novelty.src.hpa import ReactiveHPA
from Novelty.src.scaler import HybridProvisioner, UncertaintyAwareProvisioner


class Model:
    def predict(self, w, c, s):
        return max(1, round(w / c * 1000)), c / 1000


class NoveltyCoreTests(unittest.TestCase):
    def test_adaptive_burst_fires_on_first_exceedance(self):
        detector = AdaptiveBurstDetector(parameters=BurstParameters(threshold=2, consecutive_exceedances=1))
        detector.warm([10] * 10)
        self.assertTrue(detector.update(100))

    def test_hybrid_algorithm_is_shared_with_paper(self):
        self.assertEqual(HybridProvisioner(Model()).decide(1000, True).cpu_millicores, 950)

    def test_interval_inflation_requires_a_burst(self):
        provisioner = UncertaintyAwareProvisioner(Model(), max_relative_width=0.5)
        low, mid, high = 900.0, 1000.0, 1500.0
        decision = provisioner.decide(mid, burst=True, interval=(low, mid, high))
        self.assertGreater(decision.cpu_millicores, 600)

    def test_ordinary_interval_uses_paper_style_dtr_choice(self):
        provisioner = UncertaintyAwareProvisioner(Model(), max_relative_width=1.0)
        decision = provisioner.decide(1000, burst=False, interval=(990, 1000, 1010))
        self.assertEqual(decision.cpu_millicores, 950)

    def test_nonburst_quantile_interval_adds_calibrated_upper_safety_margin(self):
        provisioner = UncertaintyAwareProvisioner(Model(), interval_safety_gain=0.5)
        point = provisioner.decide(1000, burst=False)
        interval = provisioner.decide(1000, burst=False, interval=(900, 1000, 3000))
        self.assertGreater(interval.replicas, point.replicas)

    def test_negative_quantile_gain_uses_lower_quantile_for_cost_search(self):
        class CapturingModel:
            def __init__(self): self.workloads=[]
            def predict(self, workload, cpu, slo):
                self.workloads.append(workload)
                return 1, 0.5
        model=CapturingModel()
        provisioner=UncertaintyAwareProvisioner(model, interval_safety_gain=-1.0)
        provisioner.decide(200, burst=False, interval=(100, 200, 300))
        self.assertEqual(model.workloads, [100.0] * len(provisioner.resources))

    def test_reactive_hpa_scale_down_cooldown(self):
        hpa = ReactiveHPA(scale_down_cooldown=3, stabilization_window=1)
        first = hpa.decide(2000)
        self.assertGreater(first.replicas, 1)
        second = hpa.decide(100)
        self.assertEqual(second.replicas, first.replicas - 1)
        held = hpa.decide(100)
        self.assertEqual(held.replicas, second.replicas)


if __name__ == '__main__':
    unittest.main()
