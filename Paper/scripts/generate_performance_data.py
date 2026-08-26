"""Development-only synthetic stand-in. Production data must be recorded from Locust/Kubernetes as documented."""
import argparse, numpy as np, pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--out',default='data/generated/performance.csv'); args=p.parse_args(); rng=np.random.default_rng(42); rows=[]
for cpu in (600,700,800,900,950):
 for rate in range(50,2001,50):
  capacity=cpu*.42; replicas=max(1,int(np.ceil(rate/capacity))); utilization=min(.99, rate/(replicas*capacity)); response=80+180*utilization**3+rng.normal(0,6)
  if response<=350: rows.append([rate,cpu,350,replicas,utilization,response])
frame=pd.DataFrame(rows,columns=['request_rate','cpu_millicores','slo_ms','replicas','utilization','response_ms']); import os; os.makedirs(os.path.dirname(args.out),exist_ok=True); frame.to_csv(args.out,index=False); print(args.out)
