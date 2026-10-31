"""Prepare nimamahmoudi/worldcup98-dataset's invocation_count.csv for this project."""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data import worldcup98_to_workload


parser=argparse.ArgumentParser(description='Aggregate World Cup 98 invocation counts into a chronological workload CSV.')
parser.add_argument('--input',required=True,help='Path to cloned worldcup98-dataset/invocation_count.csv')
parser.add_argument('--out',default='data/processed/worldcup98_minute.csv')
parser.add_argument('--start',help='Inclusive timestamp, e.g. 1998-06-30 08:00:00')
parser.add_argument('--end',help='Inclusive timestamp, e.g. 1998-07-01 08:00:00')
parser.add_argument('--bucket',default='min',help='Pandas time bucket; use min for the paper-aligned 60 s interval')
parser.add_argument('--scale-max',type=float,help='Optional peak normalization for a capacity-limited local cluster')
args=parser.parse_args()
input_path=Path(args.input)
if input_path.is_dir():
    candidates=[input_path/'invocation_count.csv',input_path/'worldcup98-dataset'/'invocation_count.csv']
    input_path=next((candidate for candidate in candidates if candidate.is_file()),input_path)
if not input_path.is_file():
    parser.error(f'--input must name the World Cup CSV file (or its containing directory); not found: {input_path}')
workload=worldcup98_to_workload(pd.read_csv(input_path),args.bucket,args.start,args.end,args.scale_max)
os.makedirs(os.path.dirname(args.out) or '.',exist_ok=True); workload.to_csv(args.out,index=False)
print(f'Wrote {len(workload)} {args.bucket} observations to {args.out}; peak={workload.request_rate.max():.3f}')
