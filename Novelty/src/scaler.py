"""Paper Algorithm 2 plus burst-gated quantile-aware provisioning."""
from __future__ import annotations

from Paper.src.scaler import HybridProvisioner, ScalingDecision


class UncertaintyAwareProvisioner:
    """Use quantile information to size a response after a burst is detected.

    Ordinary ticks use the same DTR-optimal resource choice as the paper.  This
    prevents a routinely available quantile interval from becoming perpetual
    over-provisioning. Interval width never independently triggers scaling.
    """

    def __init__(
        self,
        performance_model,
        resources=(600, 700, 800, 900, 950),
        slo_ms=350,
        max_relative_width=1.0,
    ):
        self.performance = performance_model
        self.resources = tuple(resources)
        self.slo = slo_ms
        self.max_relative_width = max_relative_width

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

    def decide(self, workload, burst=False, interval=None):
        if burst and interval is not None:
            relative_width = self._relative_width(interval)
            adjusted = self._workload_from_interval(interval)
            interval_cpu = self._cpu_from_width(relative_width)
            replicas, util, _rs, _us = self._predict(adjusted, interval_cpu)
            return ScalingDecision(replicas, interval_cpu, util, True)
        return self._baseline_decision(workload, burst)


__all__ = ['HybridProvisioner', 'ScalingDecision', 'UncertaintyAwareProvisioner']
