"""Train deployable novelty models using the same chronological workload split."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'Paper'))
from src.data import chronological_split, load_workload_csv
from Novelty.src.forecast import QuantileBiLSTMForecaster, WorkloadForecaster
from Novelty.src.performance import PerformanceModel

p=argparse.ArgumentParser(description='Train novelty artifacts with practical World Cup trace defaults.')
p.add_argument('--workload',required=True); p.add_argument('--performance',required=True); p.add_argument('--models',default='Novelty/models')
p.add_argument('--epochs',type=int,default=80); p.add_argument('--batch-size',type=int,default=256)
p.add_argument('--max-train-points',type=int,default=12000,help='Newest points from the chronological 80%% training split; 0 keeps all points.')
p.add_argument('--window-stride',type=int,default=5,help='Keep every Nth overlapping training window; 1 uses every window.')
p.add_argument('--progress-every',type=int,default=5,help='Print loss every N epochs; 0 disables epoch progress.')
p.add_argument('--torch-threads',type=int,default=4,help='CPU threads for PyTorch; 0 keeps PyTorch default.')
p.add_argument('--seed',type=int,default=42,help='Random seed for reproducible training; None disables seeding.')
p.add_argument('--seed-sweep',type=int,default=0,help='Train N different seeds (e.g., --seed-sweep 5); 0 trains only --seed.')
p.add_argument('--skip-forecasters',action='store_true',help='Only train and save the DTR performance model; reuse existing .pt files.')
args=p.parse_args()
if args.batch_size < 32: p.error('--batch-size must be at least 32; tiny batches make CPU training very slow.')
if args.max_train_points < 0: p.error('--max-train-points must be non-negative.')
if args.window_stride < 1: p.error('--window-stride must be at least 1.')
if args.torch_threads < 0: p.error('--torch-threads must be non-negative.')
if args.seed_sweep < 0: p.error('--seed-sweep must be non-negative.')
if args.torch_threads:
    import torch
    torch.set_num_threads(args.torch_threads)
os.makedirs(args.models,exist_ok=True); full_train,test=chronological_split(load_workload_csv(args.workload))
train=full_train[-args.max_train_points:] if args.max_train_points else full_train
if len(train) <= 10: p.error('The selected training window must contain more than 10 observations.')
if not args.skip_forecasters:
    print(f'Training on {len(train):,} newest points from {len(full_train):,} chronological training points; '
         f'{len(test):,} points remain held out. window_stride={args.window_stride}, batch_size={args.batch_size}.', flush=True)
    seeds=range(args.seed, args.seed+args.seed_sweep) if args.seed_sweep>0 else [args.seed]
    for seed_idx,seed in enumerate(seeds):
        seed_suffix=f'_seed{seed}' if args.seed_sweep>0 or args.seed else ''
        for name,model in [('bilstm',WorkloadForecaster()),('quantile_bilstm',QuantileBiLSTMForecaster())]:
            fit_epochs=args.epochs
            print(f'Starting {name}{seed_suffix} ({fit_epochs} epochs)...', flush=True)
            model.fit(train,epochs=fit_epochs,batch_size=args.batch_size,window_stride=args.window_stride,progress_every=args.progress_every,label=f'{name}{seed_suffix}',seed=seed)
            model.save(os.path.join(args.models,f'{name}{seed_suffix}.pt'))
            print(f'Saved {name}{seed_suffix}.pt', flush=True)
else:
    print('Skipping forecasters; reusing existing .pt artifacts.', flush=True)
frame=pd.read_csv(args.performance)
print('Training DTR performance model...', flush=True)
PerformanceModel().fit(frame).save(os.path.join(args.models,'performance.joblib'))
print(f'Completed novelty training. Artifacts: {args.models}', flush=True)
