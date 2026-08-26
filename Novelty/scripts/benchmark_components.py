"""CLI for held-out component benchmarks before policy ablation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Paper'))

from src.data import load_workload_csv
from Novelty.src.benchmark import benchmark_forecasters, benchmark_performance_models


def write_report(forecast_table, performance_table, out):
    def _table(frame):
        header = ' | '.join(frame.columns)
        divider = ' | '.join(['---'] * len(frame.columns))
        body = [' | '.join(str(value) for value in row) for row in frame.itertuples(index=False)]
        return '\n'.join([f'| {header} |', f'| {divider} |', *[f'| {line} |' for line in body]])

    lines = [
        '# Component quality benchmarks',
        '',
        'Run this before the policy ablation. Components that fail here should not',
        'be interpreted as policy failures in `summary.csv`.',
        '',
        '## Table 1 — Forecaster held-out quality',
        '',
        _table(forecast_table),
        '',
        '## Table 2 — Performance-model held-out quality',
        '',
        _table(performance_table),
    ]
    (out / 'component_benchmark.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Benchmark forecasters and performance models on held-out data.')
    parser.add_argument('--workload', required=True)
    parser.add_argument('--performance', required=True)
    parser.add_argument('--out', default='Novelty/results/worldcup98')
    parser.add_argument('--models', default='Novelty/models')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--skip-forecasters', action='store_true', help='Skip training and load pre-trained forecaster models.')
    parser.add_argument('--window-stride', type=int, default=5)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--max-train-points', type=int, default=12000, help='Newest points from chronological training split.')
    parser.add_argument('--progress-every', type=int, default=5)
    args = parser.parse_args()

    if args.torch_threads:
        import torch
        torch.set_num_threads(args.torch_threads)

    import numpy as np
    values = np.asarray(load_workload_csv(args.workload), dtype=float)
    cut = int(0.8 * len(values))
    train, test = values[:cut], values[cut:]
    
    train = train[-args.max_train_points:] if args.max_train_points else train

    models_dir = args.models if args.skip_forecasters else None
    forecast_table = benchmark_forecasters(
        train, test, args.epochs, args.batch_size,
        window_stride=args.window_stride, models_dir=models_dir,
        progress_every=args.progress_every
    )
    performance_table = benchmark_performance_models(pd.read_csv(args.performance))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([forecast_table, performance_table], ignore_index=True)
    combined.to_csv(out / 'component_benchmark.csv', index=False)
    write_report(forecast_table, performance_table, out)
    print(combined.to_string(index=False))


if __name__ == '__main__':
    main()
