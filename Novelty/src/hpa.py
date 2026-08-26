"""Reactive HPA baseline with stabilization, cooldown, and dead-zone."""
from __future__ import annotations

from collections import deque

import numpy as np

from Paper.src.scaler import ScalingDecision


class ReactiveHPA:
    """CPU-style HPA with scale-up stabilization and scale-down cooldown.

    Mirrors the paper's headline reactive-HPA weakness: slow to scale down,
    tolerant of small demand changes, and delayed by one monitoring interval.
    """

    def __init__(
        self,
        cpu=950,
        target=0.70,
        tolerance=0.10,
        stabilization_window=3,
        scale_down_cooldown=5,
        max_scale_down_step=1,
        per_pod_capacity=0.42,
    ):
        self.cpu = cpu
        self.target = target
        self.tolerance = tolerance
        self.stabilization_window = stabilization_window
        self.scale_down_cooldown = scale_down_cooldown
        self.max_scale_down_step = max(1, int(max_scale_down_step))
        self.per_pod_capacity = per_pod_capacity
        self.replicas = 1
        self.desired_history = deque(maxlen=stabilization_window)
        self.cooldown_remaining = 0

    def _capacity(self, replicas):
        return replicas * self.cpu * self.per_pod_capacity

    def decide(self, demand):
        demand = float(demand)
        desired = max(1, int(np.ceil(demand / (self.cpu * self.per_pod_capacity * self.target))))
        self.desired_history.append(desired)

        if len(self.desired_history) >= self.stabilization_window:
            stabilized = max(self.desired_history)
        else:
            stabilized = desired

        current_capacity = self._capacity(self.replicas)
        # A dead-zone suppresses only small recommendation changes. The prior
        # condition returned for every lower demand, accidentally preventing
        # all scale-downs and turning the baseline into a permanent max scale.
        relative_change = abs(stabilized - self.replicas) / max(self.replicas, 1)
        if relative_change <= self.tolerance:
            util = min(1.0, demand / max(current_capacity, 1e-9))
            return ScalingDecision(self.replicas, self.cpu, util, False)

        if stabilized > self.replicas:
            self.replicas = stabilized
            self.cooldown_remaining = 0
        elif stabilized < self.replicas:
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            else:
                # Deliberately remove capacity gradually.  This makes the
                # cooldown effective even when a low sample would otherwise
                # recommend a one-replica deployment immediately.
                self.replicas = max(stabilized, self.replicas - self.max_scale_down_step)
                self.cooldown_remaining = self.scale_down_cooldown

        util = min(1.0, demand / max(self._capacity(self.replicas), 1e-9))
        return ScalingDecision(self.replicas, self.cpu, util, False)
