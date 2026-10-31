"""Workload-adaptive monitoring intervals with validation-only calibration."""
from __future__ import annotations

from collections import deque
import statistics


class AdaptiveMonitoringInterval:
    def __init__(self, minimum_seconds=15, maximum_seconds=240, base_seconds=60,
                 window=10, high_cv=.30, low_cv=.05, high_cv_percentile=.75):
        if minimum_seconds <= 0 or maximum_seconds < minimum_seconds:
            raise ValueError('monitoring interval bounds must be positive and ordered')
        self.minimum = int(minimum_seconds)
        self.maximum = int(maximum_seconds)
        self.base = int(base_seconds)
        self.high_cv = float(high_cv)
        self.low_cv = float(low_cv)
        self.high_cv_percentile = float(high_cv_percentile)
        self.high_forecast_width, self.low_forecast_width = .6, .2
        self.values = deque(maxlen=window)
        self.forecast_widths = deque(maxlen=window)

    @staticmethod
    def _cv(values):
        if len(values) < 3:
            return 0.0
        return statistics.stdev(values) / max(abs(statistics.fmean(values)), 1e-9)

    def history_cv(self):
        """Volatility from already observed samples; never includes the current tick."""
        return self._cv(list(self.values))

    def calibrate(self, values, forecast_widths=None, high_cv_percentile=None):
        """Set thresholds from a prior chronological validation segment."""
        if high_cv_percentile is not None:
            self.high_cv_percentile = float(high_cv_percentile)
        values = list(map(float, values))
        cvs = []
        for end in range(3, len(values) + 1):
            window = values[max(0, end - self.values.maxlen):end]
            cvs.append(self._cv(window))
        if cvs:
            cvs.sort()
            self.low_cv = cvs[len(cvs) // 4]
            index = max(0, min(len(cvs) - 1,
                               int((len(cvs) - 1) * self.high_cv_percentile)))
            self.high_cv = max(self.low_cv + 1e-6, cvs[index])
        if forecast_widths:
            widths = sorted(float(width) for width in forecast_widths if width >= 0)
            if widths:
                self.low_forecast_width = widths[len(widths) // 4]
                index = max(0, min(len(widths) - 1,
                                   int((len(widths) - 1) * self.high_cv_percentile)))
                self.high_forecast_width = max(
                    self.low_forecast_width + 1e-6, widths[index])

    def _urgency_percentile(self, cv, forecast_width):
        load_urgency = max(0.0, min(1.0,
            (cv - self.low_cv) / (self.high_cv - self.low_cv)))
        width_urgency = max(0.0, min(1.0,
            (forecast_width - self.low_forecast_width) /
            (self.high_forecast_width - self.low_forecast_width)))
        return max(load_urgency, width_urgency)

    def update(self, value, burst=False, forecast_relative_width=0.0):
        """Observe a completed sample and return the interval for the next poll."""
        self.values.append(float(value))
        if burst:
            return self.minimum
        if len(self.values) < 3:
            return self.base
        cv = self._cv(list(self.values))
        urgency = self._urgency_percentile(cv, float(forecast_relative_width))
        if urgency >= .75:
            return self.minimum
        if urgency <= .25:
            return self.maximum
        return int(round(self.maximum - (self.maximum - self.minimum) * urgency))
