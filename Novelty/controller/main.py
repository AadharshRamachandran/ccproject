"""Deployable novelty controller; Paper/controller remains the baseline controller."""
import logging, os, time
from collections import deque
from threading import Lock, Thread
from fastapi import FastAPI
from Novelty.src.burst import AdaptiveBurstDetector, BurstParameters, OnlineBurstDetector
from Novelty.src.forecast import FORECASTERS
from Novelty.src.kubernetes_executor import InPlaceResizeExecutor, RollingUpdateExecutor
from Novelty.src.monitor import prometheus_qps
from Novelty.src.performance import PERFORMANCE_MODELS
from Novelty.src.scaler import HybridProvisioner, UncertaintyAwareProvisioner
from Novelty.src.monitoring import AdaptiveMonitoringInterval

logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO')); log=logging.getLogger('novelty-controller')
app=FastAPI(title='Novel Hybrid Autoscaler')
forecast_kind=os.getenv('FORECASTER','quantile_bilstm').lower(); model_path=os.getenv('FORECAST_MODEL',f'Novelty/models/{forecast_kind}.pt')
forecast=FORECASTERS[forecast_kind].load(model_path)
performance_kind=os.getenv('PERFORMANCE_MODEL_KIND','dtr').lower(); performance_path=os.getenv('PERFORMANCE_MODEL', 'Novelty/models/performance.joblib')
performance=PERFORMANCE_MODELS[performance_kind].load(performance_path)
burst_kind=os.getenv('BURST_POLICY','tuned_adaptive').lower(); detector=AdaptiveBurstDetector(parameters=BurstParameters()) if burst_kind=='tuned_adaptive' else OnlineBurstDetector()
provisioner=UncertaintyAwareProvisioner(performance); history=deque(maxlen=120); lock=Lock(); latest={}
namespace=os.getenv('NAMESPACE','hybrid-autoscaling'); deployment=os.getenv('DEPLOYMENT','house-price')
executor=None
if os.getenv('APPLY_CHANGES','false').lower()=='true': executor=InPlaceResizeExecutor(namespace,deployment) if os.getenv('EXECUTION_BACKEND','inplace').lower()=='inplace' else RollingUpdateExecutor(namespace,deployment)
interval_policy=AdaptiveMonitoringInterval(base_seconds=int(os.getenv('INTERVAL_SECONDS','60')))

def process(qps):
    with lock:
        history.append(float(qps))
        if len(history)<forecast.lookback: return {'status':'warming','samples':len(history)}
        if hasattr(forecast,'predict_interval'): low,predicted,high=forecast.predict_interval(list(history))
        else: predicted=forecast.predict(list(history)); low=high=predicted
        burst=detector.update(predicted)
        interval=(low,predicted,high) if hasattr(forecast,'predict_interval') else None
        decision=provisioner.decide(predicted,burst=burst,interval=interval)
        result={'status':'ready','observed_qps':qps,'predicted_qps':predicted,'p10':low,'p90':high,'burst':burst,'requested_qps':predicted,'replicas':decision.replicas,'cpu_millicores':decision.cpu_millicores,'utilization':decision.utilization}
        if executor: result['execution']=executor.execute(decision)
        latest.clear(); latest.update(result); return result

def monitor_loop():
    url=os.getenv('PROMETHEUS_URL','http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090'); service=os.getenv('SERVICE','house-price'); interval=int(os.getenv('INTERVAL_SECONDS','60')); adaptive=os.getenv('ADAPTIVE_MONITORING','true').lower()=='true'
    while True:
        try:
            result=process(prometheus_qps(url,namespace,service)); log.info('decision %s',result)
            if adaptive and result.get('status')=='ready':
                width=(result['p90']-result['p10'])/max(abs(result['predicted_qps']),1e-9)
                interval=interval_policy.update(result['observed_qps'],result['burst'],width)
        except Exception: log.exception('monitor cycle failed')
        time.sleep(interval)

@app.on_event('startup')
def startup():
    if os.getenv('MONITOR_ENABLED','true').lower()=='true': Thread(target=monitor_loop,daemon=True).start()
@app.post('/observe/{qps}')
def observe(qps:float): return process(qps)
@app.get('/status')
def status(): return latest or {'status':'starting'}
