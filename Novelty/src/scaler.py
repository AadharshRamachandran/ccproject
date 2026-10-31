"""Paper Algorithm 2 plus burst-gated quantile-aware provisioning."""
from __future__ import annotations

from Paper.src.scaler import HybridProvisioner, ScalingDecision


class UncertaintyAwareProvisioner:
    """Blend calibrated upper quantiles into provisioning decisions.

    Ordinary ticks use a configurable fraction of the P50-to-P90 safety margin;
    a detected burst retains the width-aware emergency path.  This lets
    uncertainty influence scale decisions before the fixed burst detector fires.
    """

    def __init__(
        self,
        performance_model,
        resources=(600, 700, 800, 900, 950),
        slo_ms=350,
        max_relative_width=1.0,
        interval_safety_gain=0.5,
        cost_aware=False,
        utilization_floor=0.60,
        utilization_ceiling=0.85,
    ):
        self.performance = performance_model
        self.resources = tuple(resources)
        self.slo = slo_ms
        self.max_relative_width = max_relative_width
        self.interval_safety_gain = float(interval_safety_gain)
        # N4 adds a cost-aware resource choice on top of the paper controller.
        # The DTR remains the source of predicted replicas/utilization; actual
        # request SLO measurements remain the acceptance criterion.
        self.cost_aware = bool(cost_aware)
        self.utilization_floor = float(utilization_floor)
        self.utilization_ceiling = float(utilization_ceiling)

    def _predict(self, workload, cpu):
        replicas, util = self.performance.predict(workload, cpu, self.slo)
        return replicas, util, 0.0, 0.0

    def set_interval_calibration(self, widths, saturation_quantile=0.99):
        """Set the burst-response saturation width from historical intervals."""
        widths = sorted(float(width) for width in widths if width >= 0)
        if not widths:
            return
        def quantile(q):
            return widths[min(len(widths) - 1, round((len(widths) - 1) * q))]
        self.max_relative_width = max(quantile(saturation_quantile), 1e-9)

    def _relative_width(self, interval):
        low, mid, high = interval
        return (float(high) - float(low)) / max(abs(float(mid)), 1e-9)

    def _cpu_from_width(self, relative_width):
        scale = min(1.0, max(0.0, relative_width / self.max_relative_width))
        index = int(round(scale * (len(self.resources) - 1)))
        return self.resources[index]

    def _workload_from_interval(self, interval):
        low, mid, high = interval
        width = self._relative_width(interval)
        blend = min(1.0, width / self.max_relative_width)
        return float(mid) + blend * (float(high) - float(mid))

    def _baseline_decision(self, workload, burst=False):
        if self.cost_aware:
            return self._cost_aware_decision(workload)
        if burst:
            cpu = max(self.resources)
            replicas, util, _rs, _us = self._predict(workload, cpu)
            return ScalingDecision(replicas, cpu, util, True)
        candidates = []
        for cpu in self.resources:
            replicas, util, _rs, _us = self._predict(workload, cpu)
            candidates.append((util, replicas, cpu))
        util, replicas, cpu = max(candidates, key=lambda item: item[0])
        return ScalingDecision(replicas, cpu, util, False)

    def _cost_aware_decision(self, workload):
        """Minimize predicted requested CPU within a bounded utilization band."""
        candidates = []
        for cpu in self.resources:
            replicas, util, _rs, _us = self._predict(workload, cpu)
            candidates.append((replicas * cpu, -util, replicas, cpu, util))

        in_band = [item for item in candidates
                   if self.utilization_floor <= item[4] <= self.utilization_ceiling]
        below_ceiling = [item for item in candidates
                         if item[4] <= self.utilization_ceiling]
        # Prefer the least requested CPU that meets the utilization floor; use
        # the upper bound as a predicted headroom guard. If no candidate fits,
        # retain the best feasible choice below the ceiling. If the measured
        # model predicts every candidate above the ceiling, use the original
        # paper safety decision instead of relaxing the guard.
        selected = min(in_band or below_ceiling, default=None)
        if selected is None:
            cpu = max(self.resources)
            replicas, util, _rs, _us = self._predict(workload, cpu)
            return ScalingDecision(replicas, cpu, util, True)
        _cores, _neg_util, replicas, cpu, util = selected
        return ScalingDecision(replicas, cpu, util, False)

    def decide(self, workload, burst=False, interval=None):
        if burst and interval is not None:
            relative_width = self._relative_width(interval)
            adjusted = self._workload_from_interval(interval)
            if self.cost_aware:
                return self._cost_aware_decision(adjusted)
            interval_cpu = self._cpu_from_width(relative_width)
            replicas, util, _rs, _us = self._predict(adjusted, interval_cpu)
            return ScalingDecision(replicas, interval_cpu, util, True)
        if interval is not None:
            lower, median, upper = interval
            if self.interval_safety_gain >= 0:
                adjusted = float(median) + self.interval_safety_gain * max(0.0, float(upper) - float(median))
            else:
                # Negative gains interpolate toward the lower quantile. This
                # lets validation find the cheapest risk level that still
                # improves held-out SLO, rather than only testing extra safety.
                adjusted = float(median) + self.interval_safety_gain * max(0.0, float(median) - float(lower))
            return self._baseline_decision(adjusted, burst=False)
        return self._baseline_decision(workload, burst)


__all__ = ['HybridProvisioner', 'ScalingDecision', 'UncertaintyAwareProvisioner']
