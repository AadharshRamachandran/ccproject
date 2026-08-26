"""Kubernetes execution backends: 2022 rolling updates and in-place resize."""
from __future__ import annotations
import copy, time
from kubernetes import client, config


def _load_config():
    try: config.load_incluster_config()
    except config.ConfigException: config.load_kube_config()


class InPlaceResizeExecutor:
    """Kubernetes >=1.35 backend using the Pod ``resize`` subresource.

    Deployments cannot themselves be resized in place: their *running Pods* are
    patched individually.  Importantly, this does not patch the Deployment's Pod
    template: that would trigger a replacement rollout and invalidate the
    experiment. CPU changes use the default ``NotRequired`` resize policy and
    therefore do not require blue/green overlap.
    Pending/deferred resize conditions are reported to the controller instead of
    pretending a request was applied.
    """
    def __init__(self, namespace, deployment, container_name=None):
        _load_config(); self.namespace,self.deployment=namespace,deployment
        self.container_name=container_name; self.apps=client.AppsV1Api(); self.core=client.CoreV1Api()

    def _body(self, pod, cpu):
        resources=[]
        for container in pod.spec.containers:
            requests=dict(container.resources.requests or {}) if container.resources else {}
            limits=dict(container.resources.limits or {}) if container.resources else {}
            requests['cpu']=cpu; limits['cpu']=cpu
            resources.append({'name':container.name,'resources':{'requests':requests,'limits':limits}})
        return {'spec':{'containers':resources}}

    def execute(self, decision):
        target=f'{decision.cpu_millicores}m'
        deployment=self.apps.read_namespaced_deployment(self.deployment,self.namespace)
        # Patching only /scale does not create a new ReplicaSet. New replicas use
        # the template's original size and are resized on their next ready cycle.
        self.apps.patch_namespaced_deployment_scale(self.deployment,self.namespace,{'spec':{'replicas':decision.replicas}})
        selector=','.join(f'{key}={value}' for key,value in deployment.spec.selector.match_labels.items())
        pods=self.core.list_namespaced_pod(self.namespace,label_selector=selector).items
        resized=[]
        for pod in pods:
            if pod.metadata.deletion_timestamp is not None or pod.status.phase != 'Running': continue
            # Generated Python clients expose this endpoint from Kubernetes 1.35;
            # use call_api so older clients fail with an explicit compatibility error.
            resized_pod=self.core.api_client.call_api('/api/v1/namespaces/{namespace}/pods/{name}/resize','PATCH',
                path_params={'namespace':self.namespace,'name':pod.metadata.name},
                body=self._body(pod,target), header_params={'Content-Type':'application/merge-patch+json'},
                response_type='V1Pod', auth_settings=['BearerToken'], _return_http_data_only=True)
            conditions=getattr(getattr(resized_pod,'status',None),'conditions',None) or []
            resize_conditions=[{'type':item.type,'reason':getattr(item,'reason',None),'message':getattr(item,'message',None)}
                               for item in conditions if item.type.startswith('PodResize')]
            resized.append({'pod':pod.metadata.name,'resize_conditions':resize_conditions})
        return {'action':'in-place-hybrid','deployment':self.deployment,'resized_pods':resized,
                'overlap_seconds':0, 'target_cpu':target}

class RollingUpdateExecutor:
    def __init__(self, namespace, deployment, timeout_seconds=180):
        _load_config()
        self.namespace,self.deployment,self.timeout=namespace,deployment,timeout_seconds; self.api=client.AppsV1Api()
    def execute(self, decision):
        old=self.api.read_namespaced_deployment(self.deployment,self.namespace)
        container=old.spec.template.spec.containers[0]
        current=container.resources.requests.get('cpu') if container.resources and container.resources.requests else None
        target=f'{decision.cpu_millicores}m'
        if current==target:
            self.api.patch_namespaced_deployment_scale(self.deployment,self.namespace,{'spec':{'replicas':decision.replicas}})
            return {'action':'horizontal','deployment':self.deployment}
        # Paper Fig. 2: new Deployment retains Service labels, becomes ready, then old is deleted.
        body=copy.deepcopy(old); body.metadata.resource_version=None; body.metadata.uid=None; body.metadata.creation_timestamp=None
        body.metadata.name=f'{self.deployment}-cpu-{decision.cpu_millicores}-{int(time.time())}'; body.metadata.generate_name=None
        body.spec.replicas=decision.replicas
        for item in body.spec.template.spec.containers:
            item.resources.requests['cpu']=target; item.resources.limits['cpu']=target
        created=self.api.create_namespaced_deployment(self.namespace,body)
        deadline=time.time()+self.timeout
        while time.time()<deadline:
            status=self.api.read_namespaced_deployment(created.metadata.name,self.namespace).status
            if (status.available_replicas or 0)>=decision.replicas:
                self.api.delete_namespaced_deployment(self.deployment,self.namespace)
                self.deployment=created.metadata.name
                return {'action':'rolling-hybrid','deployment':created.metadata.name}
            time.sleep(2)
        self.api.delete_namespaced_deployment(created.metadata.name,self.namespace)
        raise TimeoutError('new deployment did not become ready; old deployment retained')
