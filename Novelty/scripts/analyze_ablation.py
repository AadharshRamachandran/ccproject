"""Add paired chronological block-bootstrap effects to saved ablation runs."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Novelty.ablation import PAPER_NAME, _bootstrap, write_report

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',required=True)
parser.add_argument('--bootstrap-samples',type=int,default=2000)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--capacity-peak',type=float,default=1800)
parser.add_argument('--epochs',type=int,default=None)
parser.add_argument('--performance-source',default='see run metadata; specify measured or synthetic source')
parser.add_argument('--reuse-existing-paired',action='store_true',help='Reuse confidence intervals already stored in summary.csv.')
args=parser.parse_args()
out=Path(args.results)
summary=pd.read_csv(out/'summary.csv')
reference=pd.read_csv(out/f'{PAPER_NAME}.csv')
rng=np.random.default_rng(args.seed)
paired_columns=('paired_slo_delta_ci95_low_pp','paired_slo_delta_ci95_high_pp',
                'paired_cost_delta_ci95_low','paired_cost_delta_ci95_high')
if not args.reuse_existing_paired or not set(paired_columns).issubset(summary.columns):
    for column in paired_columns: summary[column]=np.nan
    for idx,row in summary.iterrows():
        detail=pd.read_csv(out/f'{row.variant}.csv')
        if len(detail)!=len(reference):
            raise ValueError(f'Unaligned traces for {row.variant}: {len(detail)} vs {len(reference)}')
        sd=detail.slo_violation.astype(float).to_numpy()-reference.slo_violation.astype(float).to_numpy()
        cd=detail.resource_cost.astype(float).to_numpy()-reference.resource_cost.astype(float).to_numpy()
        slo_ci=_bootstrap(sd,np.mean,args.bootstrap_samples,rng)
        cost_ci=_bootstrap(cd,np.sum,args.bootstrap_samples,rng)
        summary.loc[idx,'paired_slo_delta_ci95_low_pp']=100*slo_ci[0]
        summary.loc[idx,'paired_slo_delta_ci95_high_pp']=100*slo_ci[1]
        summary.loc[idx,'paired_cost_delta_ci95_low']=cost_ci[0]
        summary.loc[idx,'paired_cost_delta_ci95_high']=cost_ci[1]
summary.to_csv(out/'summary.csv',index=False)
paper=summary.loc[summary.variant==PAPER_NAME].iloc[0]
summary['delta_slo_violation_pct']=summary.slo_violation_pct-paper.slo_violation_pct
summary['delta_resource_cost']=summary.resource_cost-paper.resource_cost
summary['delta_utilization_pct']=summary.mean_utilization_pct-paper.mean_utilization_pct
summary.to_csv(out/'pairwise_vs_paper.csv',index=False)
summary.loc[summary.variant.str.startswith(('novelty_','optimized_novelty_'))].to_csv(out/'novelty_effects_vs_paper.csv',index=False)
write_report(summary,out,args.capacity_peak,performance_source=args.performance_source,
             epochs=args.epochs,seed=args.seed,pairwise=summary,
             bootstrap_samples=args.bootstrap_samples)
print(f'Wrote paired block-bootstrap effects for {len(summary)} configurations to {out}')
