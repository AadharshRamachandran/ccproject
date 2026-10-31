from collections import deque
import statistics

class OnlineBurstDetector:
    """Algorithm 1 from the paper: moving mean/std burst detector."""
    def __init__(self, window=10, threshold=5.0, influence=0.5):
        if window < 2:
            raise ValueError("window must be at least 2")
        self.values = deque(maxlen=window)
        self.threshold = float(threshold)
        self.influence = float(influence)
        self.mean = 0.0
        self.std = 0.0

    def _refresh_baseline(self, values):
        values = list(values)
        self.mean = statistics.fmean(values) if values else 0.0
        self.std = statistics.stdev(values) if len(values) > 1 else 0.0

    def warm(self, values):
        self.values.clear()
        self.values.extend(float(value) for value in values)
        self._refresh_baseline(self.values)

    def update(self, predicted):
        predicted = float(predicted)
        if len(self.values) < 2:
            self.values.append(predicted)
            self._refresh_baseline(self.values)
            return False
        # Algorithm 1 compares against its rolling mean/std, influence-filters
        # the value used to update that baseline, and appends the unfiltered
        # forecast to Y (lines 7-17 of the paper pseudocode).
        is_burst = predicted - self.mean > self.threshold * max(self.std, 1e-9)
        filtered = (self.influence * predicted +
                    (1.0 - self.influence) * self.values[-1]) if is_burst else predicted
        next_baseline = list(self.values)[1:] + [filtered]
        self.values.append(predicted)
        self._refresh_baseline(next_baseline)
        return is_burst
