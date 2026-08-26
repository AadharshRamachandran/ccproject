"""Records one measured row per interval while Locust drives a stable workload."""
import sys, pathlib
import argparse, csv, os, time
# Ensure project root is on sys.path so the local "src" package can be imported when running this script directly
# scripts/ is one level below the project root, so use parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.monitor import query, prometheus_qps
p=argparse.ArgumentParser(); p.add_argument('--prometheus',required=True); p.add_argument('--namespace',default='hybrid-autoscaling'); p.add_argument('--service',default='house-price'); p.add_argument('--cpu',type=int,required=True); p.add_argument('--slo',type=int,default=350); p.add_argument('--out',default='data/generated/performance.csv'); p.add_argument('--samples',type=int,default=5); p.add_argument('--interval',type=int,default=60); args=p.parse_args()
os.makedirs(os.path.dirname(args.out),exist_ok=True); exists=os.path.exists(args.out)
with open(args.out,'a',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=['request_rate','cpu_millicores','slo_ms','replicas','utilization','response_ms']);
 if not exists: writer.writeheader()
 for _ in range(args.samples):
  qps=prometheus_qps(args.prometheus,args.namespace,args.service)
  response=1000*query(args.prometheus,f'histogram_quantile(0.95,sum(rate(http_request_duration_seconds_bucket{{namespace="{args.namespace}",service="{args.service}"}}[1m])) by (le))')
  replicas=query(args.prometheus,f'kube_deployment_status_replicas_available{{namespace="{args.namespace}",deployment="house-price"}}')
  utilization=query(args.prometheus,f'avg(rate(container_cpu_usage_seconds_total{{namespace="{args.namespace}",container="app"}}[1m])) / ({args.cpu}/1000)')
  writer.writerow({'request_rate':qps,'cpu_millicores':args.cpu,'slo_ms':args.slo,'replicas':max(1,round(replicas)),'utilization':max(0,min(1,utilization)),'response_ms':response}); stream.flush(); time.sleep(args.interval)
