import requests
def prometheus_qps(base_url, namespace, service):
    query=f'sum(rate(http_requests_total{{namespace="{namespace}",service="{service}"}}[1m]))'
    payload=requests.get(f'{base_url}/api/v1/query',params={'query':query},timeout=10).json(); rows=payload.get('data',{}).get('result',[])
    return float(rows[0]['value'][1]) if rows else 0.0

def query(base_url, expression):
    payload=requests.get(f'{base_url}/api/v1/query',params={'query':expression},timeout=15).json()
    rows=payload.get('data',{}).get('result',[]); return float(rows[0]['value'][1]) if rows else 0.0
