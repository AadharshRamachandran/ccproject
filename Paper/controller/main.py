import os, time, logging
from collections import deque
from threading import Lock, Thread
from fastapi import FastAPI
from src.burst import OnlineBurstDetector
from src.forecast import WorkloadForecaster
from src.performance import PerformanceModel
from src.scaler import HybridProvisioner
from src.kubernetes_executor import RollingUpdateExecutor, InPlaceResizeExecutor
from src.monitor import prometheus_qps

logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO')); log=logging.getLogger('hybrid-controller')
app=FastAPI(title='Predictive Hybrid Autoscaler')
forecast=WorkloadForecaster.load(os.getenv('FORECAST_MODEL','models/bilstm.pt'))
performance=PerformanceModel.load(os.getenv('PERFORMANCE_MODEL','models/performance.joblib'))
detector=OnlineBurstDetector(); provisioner=HybridProvisioner(performance); history=deque(maxlen=120); lock=Lock(); latest={}
namespace=os.getenv('NAMESPACE','hybrid-autoscaling'); deployment=os.getenv('DEPLOYMENT','house-price')
if os.getenv('APPLY_CHANGES','false').lower()=='true':
    executor = (InPlaceResizeExecutor(namespace,deployment) if os.getenv('EXECUTION_BACKEND','rolling').lower()=='inplace'
                else RollingUpdateExecutor(namespace,deployment))
else: executor=None

def process(qps):
    """Observe q(t), then calculate and optionally apply a decision for q(t+1)."""
    with lock:
        history.append(float(qps))
        if len(history)<forecast.lookback: return {'status':'warming','samples':len(history)}
        predicted=forecast.predict(list(history)); burst=detector.update(predicted); decision=provisioner.decide(predicted,burst)
        result={'status':'ready','observed_qps':qps,'predicted_qps':predicted,'burst':burst,'replicas':decision.replicas,'cpu_millicores':decision.cpu_millicores,'utilization':decision.utilization}
        if executor: result['execution']=executor.execute(decision)
        latest.clear(); latest.update(result); return result

def monitor_loop():
    url=os.getenv('PROMETHEUS_URL','http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090'); service=os.getenv('SERVICE','house-price'); interval=int(os.getenv('INTERVAL_SECONDS','60'))
    adaptive = None
    if os.getenv('ADAPTIVE_MONITORING','false').lower()=='true':
        from Novelty.src.monitoring import AdaptiveMonitoringInterval
        adaptive=AdaptiveMonitoringInterval(base_seconds=interval)
    while True:
        try:
            result=process(prometheus_qps(url,namespace,service)); log.info('decision %s',result)
            if adaptive and result.get('status')=='ready': interval=adaptive.update(result['observed_qps'],result['burst'])
        except Exception: log.exception('monitor cycle failed')
        time.sleep(interval)

@app.on_event('startup')
def startup():
    if os.getenv('MONITOR_ENABLED','true').lower()=='true': Thread(target=monitor_loop,daemon=True).start()
@app.post('/observe/{qps}')
def observe(qps:float): return process(qps)
@app.get('/status')
def status(): return latest or {'status':'starting'}
