from dataclasses import dataclass

@dataclass(frozen=True)
class ScalingDecision:
    replicas: int
    cpu_millicores: int
    utilization: float
    burst: bool

class HybridProvisioner:
    """Algorithm 2: rmax for bursts; otherwise maximize predicted utilization."""
    def __init__(self, performance_model, resources=(600,700,800,900,950), slo_ms=350):
        self.performance, self.resources, self.slo = performance_model, tuple(resources), slo_ms
    def decide(self, workload, burst=False, interval=None):
        # Accepts an optional `interval` for API compatibility with
        # UncertaintyAwareProvisioner.decide(..., interval=...), but the paper
        # hybrid provisioner does not use interval information and therefore
        # ignores it.
        if burst:
            replicas, util = self.performance.predict(workload, max(self.resources), self.slo)
            return ScalingDecision(replicas, max(self.resources), util, True)
        candidates = []
        for cpu in self.resources:
            replicas, util = self.performance.predict(workload, cpu, self.slo)
            candidates.append((util, replicas, cpu))
        util, replicas, cpu = max(candidates, key=lambda item: item[0])
        return ScalingDecision(replicas, cpu, util, False)
