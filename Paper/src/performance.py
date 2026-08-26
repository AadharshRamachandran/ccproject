from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

class PerformanceModel:
    """phi(workload, CPU millicores, SLO ms) -> (replicas, utilization)."""
    def __init__(self): self.model = DecisionTreeRegressor(random_state=42, min_samples_leaf=2)
    def fit(self, frame):
        X=frame[['request_rate','cpu_millicores','slo_ms']]; y=frame[['replicas','utilization']]; self.model.fit(X,y)
        return self
    def predict(self, workload, cpu, slo_ms):
        replicas, utilization=self.model.predict(pd.DataFrame([[workload,cpu,slo_ms]],columns=['request_rate','cpu_millicores','slo_ms']))[0]
        return max(1, int(np.ceil(replicas))), float(np.clip(utilization,0,1))
    def save(self,path): joblib.dump(self.model,path)
    @classmethod
    def load(cls,path): obj=cls(); obj.model=joblib.load(path); return obj
