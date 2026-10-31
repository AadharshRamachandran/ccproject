"""Same CPU-bound benchmark application as Paper/app for fair experiments."""
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import math, time
app=FastAPI(title='CPU-bound house-price benchmark')
requests=Counter('http_requests_total','Requests received',['method','route','status_code'])
# Include the exact 350 ms SLO boundary so Prometheus can count compliant
# requests directly with the le="0.35" bucket in live_dashboard.py.
latency=Histogram('http_request_duration_seconds','Request duration',['method','route','status_code'],
                  buckets=(.005,.01,.025,.05,.075,.1,.25,.35,.5,.75,1.0,2.5,5.0,7.5,10.0))
@app.middleware('http')
async def instrument(request:Request,call_next):
    started=time.perf_counter(); response=await call_next(request); labels=(request.method,request.url.path,str(response.status_code))
    latency.labels(*labels).observe(time.perf_counter()-started); requests.labels(*labels).inc(); return response
@app.get('/healthz')
def health(): return {'status':'ok'}
@app.get('/metrics')
def metrics(): return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)
@app.get('/predict')
def predict(area:float=1800, rooms:int=3, age:int=10):
    value=0.
    for i in range(120_000): value += math.sin(i*.0001)*math.sqrt((area+i)%1000+1)
    return {'estimated_price':round(50000+area*160+rooms*22000-age*900+value,2)}
