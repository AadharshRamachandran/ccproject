"""Held-out component benchmarks (Table 1 / Table 2 style)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from Novelty.src.forecast import QuantileBiLSTMForecaster, WorkloadForecaster
from Paper.src.performance import PerformanceModel

FORECASTERS = {
    'bilstm': WorkloadForecaster,
    'quantile_bilstm': QuantileBiLSTMForecaster,
}
PERFORMANCE_MODELS = {
    'dtr': PerformanceModel,
}


def benchmark_forecasters(train, test, epochs, batch_size, window_stride=1, models_dir=None, progress_every=5):
    rows = []
    history = list(train)
    for name, factory in FORECASTERS.items():
        model = factory()
        fit_epochs = epochs
        
        loaded = False
        if models_dir:
            import os
            model_path = os.path.join(models_dir, f'{name}.pt')
            if os.path.exists(model_path):
                print(f"Loading pre-trained {name} from {model_path}...", flush=True)
                model = factory.load(model_path)
                loaded = True
        
        if not loaded:
            print(f"Training {name} from scratch for {fit_epochs} epochs...", flush=True)
            model.fit(train, epochs=fit_epochs, batch_size=batch_size, window_stride=window_stride, progress_every=progress_every, label=name)

        actuals, preds = [], []
        running = list(history)
        for actual in test:
            if hasattr(model, 'predict_interval'):
                _low, mid, _high = model.predict_interval(running)
            else:
                mid = model.predict(running)
            actuals.append(float(actual))
            preds.append(float(mid))
            running.append(float(actual))
        actuals = np.asarray(actuals)
        preds = np.asarray(preds)
        rows.append({
            'component': 'forecaster',
            'name': name,
            'rmse': float(np.sqrt(mean_squared_error(actuals, preds))),
            'mae': float(mean_absolute_error(actuals, preds)),
            'train_points': len(train),
            'test_points': len(test),
            'epochs': 0 if loaded else fit_epochs,
        })
    table = pd.DataFrame(rows)
    baseline_mae = float(table.loc[table['name'] == 'bilstm', 'mae'].iloc[0])
    table['mae_vs_bilstm'] = table.mae / max(baseline_mae, 1e-9)
    # Components above this threshold stay in the ablation as diagnostic arms,
    # but are labelled experimental rather than interpreted as policy results.
    table['policy_eligible'] = table.mae_vs_bilstm <= 1.25
    return table


def benchmark_performance_models(frame, test_size=0.2, random_state=42):
    train, test = train_test_split(frame, test_size=test_size, random_state=random_state)
    rows = []
    for name, factory in PERFORMANCE_MODELS.items():
        model = factory()
        model.fit(train)
        actual_replicas = test['replicas'].to_numpy(dtype=float)
        actual_util = test['utilization'].to_numpy(dtype=float)
        pred_replicas, pred_util = [], []
        replica_std, util_std = [], []
        for _, row in test.iterrows():
            replicas, util = model.predict(row.request_rate, row.cpu_millicores, row.slo_ms)
            pred_replicas.append(replicas)
            pred_util.append(util)
            if hasattr(model, 'predict_with_uncertainty'):
                _r, _u, rs, us = model.predict_with_uncertainty(row.request_rate, row.cpu_millicores, row.slo_ms)
                replica_std.append(rs); util_std.append(us)
        pred_replicas = np.asarray(pred_replicas, dtype=float)
        pred_util = np.asarray(pred_util, dtype=float)
        result={
            'component': 'performance_model',
            'name': name,
            'replicas_mse': float(mean_squared_error(actual_replicas, pred_replicas)),
            'replicas_mae': float(mean_absolute_error(actual_replicas, pred_replicas)),
            'utilization_mse': float(mean_squared_error(actual_util, pred_util)),
            'utilization_mae': float(mean_absolute_error(actual_util, pred_util)),
            'train_points': len(train),
            'test_points': len(test),
            'replicas_interval95_coverage': (float(np.mean(np.abs(actual_replicas-pred_replicas) <= 1.96*np.asarray(replica_std))) if replica_std else np.nan),
            'utilization_interval95_coverage': (float(np.mean(np.abs(actual_util-pred_util) <= 1.96*np.asarray(util_std))) if util_std else np.nan),
        }
        rows.append(result)
    table=pd.DataFrame(rows)
    baseline=table.loc[table['name']=='dtr'].iloc[0]
    table['replicas_mae_vs_dtr']=table.replicas_mae/max(float(baseline.replicas_mae),1e-9)
    table['utilization_mae_vs_dtr']=table.utilization_mae/max(float(baseline.utilization_mae),1e-9)
    table['policy_eligible']=(table.replicas_mae_vs_dtr<=1.25)&(table.utilization_mae_vs_dtr<=1.25)
    return table
