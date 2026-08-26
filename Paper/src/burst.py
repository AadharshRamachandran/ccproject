from collections import deque
import statistics

class OnlineBurstDetector:
    """Algorithm 1 from the paper: moving mean/std burst detector."""
    def __init__(self, window=10, threshold=5.0, influence=0.5):
        self.values = deque(maxlen=window); self.threshold = threshold; self.influence = influence
    def warm(self, values):
        for value in values: self.values.append(float(value))
    def update(self, predicted):
        if len(self.values) < 2: self.values.append(float(predicted)); return False
        mean = statistics.fmean(self.values); std = statistics.stdev(self.values) or 1e-9
        is_burst = predicted - mean > self.threshold * std
        filtered = self.influence * predicted + (1 - self.influence) * self.values[-1] if is_burst else predicted
        self.values.append(filtered)
        return is_burst
