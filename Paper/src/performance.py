"""Paper performance model: normalized multi-output DTR.

The paper defines phi((request rate, CPU, SLO) -> (replicas, utilization)),
normalizes the collected table, and selects a DTR after comparing regression
families on an 80/20 holdout. The original table and exact split seed are not
published with this project, so the comparison is reproduced on the supplied
table and the published DTR is used for the controller.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


FEATURES = ["request_rate", "cpu_millicores", "slo_ms"]
TARGETS = ["replicas", "utilization"]


def _candidate_models(seed: int) -> dict:
    """Regression families named by the paper; XGBoost is optional."""
    models = {
        "LR": LinearRegression(),
        "RDF": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
        "SVR": MultiOutputRegressor(SVR()),
        "MLP": MLPRegressor(random_state=seed, max_iter=1000),
        "DTR": DecisionTreeRegressor(random_state=seed),
    }
    try:
        from xgboost import XGBRegressor
        models["XGB"] = MultiOutputRegressor(
            XGBRegressor(n_estimators=100, random_state=seed, n_jobs=1, verbosity=0)
        )
    except ImportError:
        pass
    return models


class PerformanceModel:
    """Normalized multi-output DTR used by the Paper Hybrid provisioner."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.model = DecisionTreeRegressor(random_state=self.seed)
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.model_name = "DTR"
        self.training_report = {}

    @staticmethod
    def _clean(frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in FEATURES + TARGETS if c not in frame.columns]
        if missing:
            raise ValueError(f"Performance table is missing columns: {missing}")
        data = frame[FEATURES + TARGETS].apply(pd.to_numeric, errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        data = data[(data.request_rate >= 0) & (data.cpu_millicores > 0) &
                    (data.slo_ms > 0) & (data.replicas >= 1) &
                    (data.utilization >= 0) & (data.utilization <= 1)]
        # The paper excludes calibration observations whose response time
        # exceeds the configured QoS/SLO threshold before model fitting.
        if "response_ms" in frame.columns:
            response = pd.to_numeric(frame.loc[data.index, "response_ms"], errors="coerce")
            data = data.loc[response.notna() &
                            (response <= np.minimum(data.slo_ms, 350.0))]
        if len(data) < 10:
            raise ValueError(f"Need at least 10 valid performance rows; got {len(data)}")
        return data.reset_index(drop=True)

    @classmethod
    def compare_regressors(cls, frame: pd.DataFrame, seed: int = 42,
                           test_size: float = 0.2) -> dict:
        """Evaluate the paper's listed model families on a normalized holdout.

        Returns test MSE/MAE averaged over the two output dimensions. A fixed
        seed makes this project's comparison repeatable; the paper's split seed
        and exact dataset are not available, so its original scores cannot be
        claimed or recreated from this function.
        """
        data = cls._clean(frame)
        idx_train, idx_test = train_test_split(
            np.arange(len(data)), test_size=test_size, random_state=seed, shuffle=True
        )
        x = data[FEATURES].to_numpy(dtype=float)
        y = data[TARGETS].to_numpy(dtype=float)
        sx, sy = MinMaxScaler(), MinMaxScaler()
        x_train = sx.fit_transform(x[idx_train])
        x_test = sx.transform(x[idx_test])
        y_train = sy.fit_transform(y[idx_train])
        y_test = sy.transform(y[idx_test])
        rows = {}
        for name, estimator in _candidate_models(seed).items():
            try:
                estimator.fit(x_train, y_train)
                pred = np.asarray(estimator.predict(x_test), dtype=float).reshape(y_test.shape)
                rows[name] = {
                    "test_mse_normalized": float(mean_squared_error(y_test, pred)),
                    "test_mae_normalized": float(mean_absolute_error(y_test, pred)),
                    "test_rows": int(len(idx_test)),
                }
            except (ValueError, FloatingPointError) as exc:
                rows[name] = {"error": str(exc)}
        return {"method": "random 80/20 holdout; MinMax scaling fitted on train split",
                "seed": int(seed), "train_rows": int(len(idx_train)),
                "test_rows": int(len(idx_test)), "models": rows,
                "controller_model": "DTR"}

    def fit(self, frame: pd.DataFrame):
        data = self._clean(frame)
        self.x_scaler = MinMaxScaler().fit(data[FEATURES])
        self.y_scaler = MinMaxScaler().fit(data[TARGETS])
        x = self.x_scaler.transform(data[FEATURES])
        y = self.y_scaler.transform(data[TARGETS])
        # The paper reports DTR as the selected model. Fit that model for every
        # Paper Hybrid and novelty arm so only policy changes across ablations.
        self.model = DecisionTreeRegressor(random_state=self.seed)
        self.model.fit(x, y)
        self.model_name = "DTR"
        self.training_report = {
            "model": self.model_name,
            "rows": int(len(data)),
            "features": FEATURES,
            "targets": TARGETS,
            "normalization": "MinMaxScaler to [0, 1] for features and targets",
        }
        return self

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[FEATURES].apply(pd.to_numeric, errors="raise")
        normalized = self.model.predict(self.x_scaler.transform(x))
        return self.y_scaler.inverse_transform(np.asarray(normalized).reshape(-1, len(TARGETS)))

    def predict(self, workload, cpu, slo_ms):
        if self.x_scaler is None or self.y_scaler is None:
            return self._legacy_predict(workload, cpu, slo_ms)
        values = self.predict_frame(pd.DataFrame(
            [[workload, cpu, slo_ms]], columns=FEATURES
        ))[0]
        replicas, utilization = values
        return max(1, int(np.ceil(replicas))), float(np.clip(utilization, 0, 1))

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        saved = joblib.load(path)
        # Backward compatibility with the old project artifact that stored
        # only an unnormalized sklearn estimator.
        if isinstance(saved, cls):
            return saved
        obj = cls()
        obj.model = saved
        obj.x_scaler = None
        obj.y_scaler = None
        obj.model_name = "DTR (legacy unnormalized artifact)"
        return obj

    def _legacy_predict(self, workload, cpu, slo_ms):
        replicas, utilization = self.model.predict(pd.DataFrame(
            [[workload, cpu, slo_ms]], columns=FEATURES
        ))[0]
        return max(1, int(np.ceil(replicas))), float(np.clip(utilization, 0, 1))
