import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data import load_workload_csv, chronological_split
from src.forecast import WorkloadForecaster
from src.performance import PerformanceModel
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument('--workload',required=True); p.add_argument('--performance',required=True); p.add_argument('--models',default='models'); p.add_argument('--epochs',type=int,default=20,help='Bi-LSTM epochs; 20 is a practical default for the full World Cup trace'); p.add_argument('--batch-size',type=int,default=512); args=p.parse_args()
os.makedirs(args.models,exist_ok=True)
train,test=chronological_split(load_workload_csv(args.workload)); forecaster=WorkloadForecaster(); forecaster.fit(train,epochs=args.epochs,batch_size=args.batch_size); forecaster.save(os.path.join(args.models,'bilstm.pt'))
model=PerformanceModel(); model.fit(pd.read_csv(args.performance)); model.save(os.path.join(args.models,'performance.joblib'))
print(f'Trained Bi-LSTM for {args.epochs} epochs on {len(train)} points; held out {len(test)} points. Saved models in {args.models}.')
