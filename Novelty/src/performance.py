"""Validated wrapper around the Paper Hybrid DTR performance model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from functools import lru_cache
from Paper.src.performance import PerformanceModel as _PaperPerformanceModel


class PerformanceModel(_PaperPerformanceModel):
    """Keep the paper estimator while rejecting malformed calibration tables."""

    required = ('request_rate', 'cpu_millicores', 'slo_ms', 'replicas', 'utilization')
    _cached_predict = lru_cache(maxsize=250_000)(_PaperPerformanceModel.predict)

    def fit(self, frame):
        missing = [name for name in self.required if name not in frame.columns]
        if missing:
            raise ValueError(f'Performance table is missing required columns: {missing}')
        clean = frame.loc[:, self.required].apply(pd.to_numeric, errors='coerce')
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        clean = clean[(clean.request_rate > 0) & (clean.cpu_millicores > 0) &
                      (clean.slo_ms > 0) & (clean.replicas >= 1) &
                      (clean.utilization >= 0) & (clean.utilization <= 1)]
        if len(clean) < 20:
            raise ValueError(f'Only {len(clean)} valid performance rows; need at least 20.')
        if clean.cpu_millicores.nunique() < 3 or clean.request_rate.nunique() < 5:
            raise ValueError('Performance calibration must cover at least 3 CPU levels and 5 request-rate levels.')
        return super().fit(clean)

    def predict(self, workload, cpu, slo_ms):
        # The same forecasts are replayed across many policy ablations. Cache
        # deterministic DTR lookups to keep validation searches tractable.
        return self._cached_predict(float(workload), int(cpu), float(slo_ms))


PERFORMANCE_MODELS = {'dtr': PerformanceModel}
