"""Kubernetes execution backends: 2022 rolling updates and in-place resize."""
from __future__ import annotations
import copy, hashlib, threading, time
from kubernetes import client, config


def _load_config():
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


class InPlaceResizeExecutor:
    """Kubernetes >=1.35 backend using the Pod ``resize`` subresource.

    Deployments cannot themselves be resized in place: their *running Pods* are
    patched individually.  Importantly, this does not patch the Deployment's Pod
    template: that would trigger a replacement rollout and invalidate the
    experiment. CPU changes use the default ``NotRequired`` resize policy and
    therefore do not require blue/green overlap.

    Pending/deferred resize conditions are reported to the controller instead of
    pretending a request was applied.

    Design note: if the resize call for one pod raises (e.g. the pod terminated
    between listing and patching) we log the error per-pod and continue, so that
    a single bad pod does not abort the entire decision.

    Patch-type note: the resize subresource is sent as
    ``application/strategic-merge-patch+json`` (NOT plain JSON Merge Patch).
    Kubernetes uses ``name`` as the strategic-merge-patch key for the
    ``containers`` list, so only the ``resources`` sub-field of the named
    container is updated; all other container fields (image, ports, env, …)
    are left untouched.  Plain ``application/merge-patch+json`` would replace
    the *entire* containers array with the partial object, causing the API
    server to reject the request for missing required fields (image, etc.).
    """

    # Strategic-merge-patch is the only correct type for patching named list
    # elements (containers) on a Pod without supplying the full container spec.
    _CONTENT_TYPE = 'application/strategic-merge-patch+json'

    def __init__(self, namespace, deployment, container_name=None):
        _load_config()
        self.namespace      = namespace
        self.deployment     = deployment
        self.container_name = container_name
        self.apps           = client.AppsV1Api()
        self.core           = client.CoreV1Api()

    def _body(self, pod, cpu):
        """Build a strategic-merge-patch body that updates only the CPU
        requests/limits for every container in the pod.

        Because this is sent as strategic-merge-patch+json, Kubernetes merges
        the containers list by ``name`` key and only touches the ``resources``
        sub-object — image and all other fields are not affected.
        """
        containers = []
        for container in pod.spec.containers:
            requests = dict(container.resources.requests or {}) if container.resources else {}
            limits   = dict(container.resources.limits   or {}) if container.resources else {}
            requests['cpu'] = cpu
            limits['cpu']   = cpu
            containers.append({
                'name': container.name,
                'resources': {'requests': requests, 'limits': limits},
            })
        return {'spec': {'containers': containers}}

    def execute(self, decision):
        target     = f'{decision.cpu_millicores}m'
        deployment = self.apps.read_namespaced_deployment(self.deployment, self.namespace)
        # Patching /scale only does not create a new ReplicaSet. New replicas
        # use the template's original CPU request and are resized on their next
        # ready monitoring cycle (one-cycle transient under-provisioning window).
        self.apps.patch_namespaced_deployment_scale(
            self.deployment, self.namespace, {'spec': {'replicas': decision.replicas}})

        selector = ','.join(
            f'{k}={v}' for k, v in deployment.spec.selector.match_labels.items())
        pods    = self.core.list_namespaced_pod(self.namespace, label_selector=selector).items
        resized = []
        for pod in pods:
            if pod.metadata.deletion_timestamp is not None or pod.status.phase != 'Running':
                continue
            try:
                resized_pod = self.core.api_client.call_api(
                    '/api/v1/namespaces/{namespace}/pods/{name}/resize', 'PATCH',
                    path_params={'namespace': self.namespace, 'name': pod.metadata.name},
                    body=self._body(pod, target),
                    # Strategic-merge-patch: Kubernetes merges containers by the
                    # 'name' key, so only 'resources' is touched per container.
                    header_params={'Content-Type': self._CONTENT_TYPE},
                    response_type='V1Pod', auth_settings=['BearerToken'],
                    _return_http_data_only=True)
                conditions = (getattr(resized_pod.status, 'conditions', None) or [])
                resize_conditions = [
                    {'type': c.type, 'reason': getattr(c, 'reason', None),
                     'message': getattr(c, 'message', None)}
                    for c in conditions if c.type.startswith('PodResize')]
                resized.append({'pod': pod.metadata.name,
                                'resize_conditions': resize_conditions})
            except Exception as exc:
                # Per-pod isolation: one failed resize does not abort the rest.
                resized.append({'pod': pod.metadata.name, 'error': str(exc)})

        return {'action': 'in-place-hybrid', 'deployment': self.deployment,
                'resized_pods': resized, 'overlap_seconds': 0, 'target_cpu': target}


class RollingUpdateExecutor:
    """Paper Fig. 2 rolling-update executor.

    Bug fix (selector overlap): the new Deployment is given a unique label
    ``cpu-gen: <hash>`` in *both* its own ``spec.selector.matchLabels`` and its
    pod-template ``metadata.labels``.  This prevents Kubernetes from having two
    Deployment controllers fight over the same ReplicaSets/Pods.  The Service's
    selector is left on ``app: <name>`` (the broader label that both generations
    share), so it keeps load-balancing across old and new pods during the overlap
    window exactly as described in the paper.

    Design note (non-blocking): the readiness-polling loop runs in a daemon
    thread so that ``execute()`` returns immediately.  This keeps HTTP workers
    free if the executor is ever called via the ``/observe/{qps}`` endpoint.

    Thread-safety note: ``self.deployment`` is the single mutable shared field.
    All reads and writes go through ``self._lock`` so that a new ``execute()``
    call arriving while a background cleanup thread is still running always sees
    a consistent view of which Deployment generation is currently active.
    """

    def __init__(self, namespace, deployment, timeout_seconds=180):
        _load_config()
        self.namespace  = namespace
        self.timeout    = timeout_seconds
        self.api        = client.AppsV1Api()
        # Protect self.deployment: written by background cleanup thread,
        # read by the next execute() call on the controller thread.
        self._lock      = threading.Lock()
        self._deployment = deployment   # always access via property below

    # ------------------------------------------------------------------
    # Thread-safe deployment name accessor
    # ------------------------------------------------------------------

    @property
    def deployment(self) -> str:
        with self._lock:
            return self._deployment

    @deployment.setter
    def deployment(self, value: str) -> None:
        with self._lock:
            self._deployment = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gen_label(cpu_millicores: int) -> str:
        """Short stable hash used as a per-generation discriminator label."""
        raw = f'{cpu_millicores}-{time.time()}'.encode()
        return hashlib.sha1(raw).hexdigest()[:8]

    def _poll_and_cleanup(self, new_name: str, old_name: str,
                          required_replicas: int) -> None:
        """Background thread: wait for the new Deployment to be ready, then
        delete the old one.  On timeout, delete the new Deployment instead and
        leave the old one running.

        ``self.deployment`` is updated via the thread-safe property setter so
        the next ``execute()`` call on the controller thread always reads the
        current active generation name.
        """
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                status = self.api.read_namespaced_deployment(
                    new_name, self.namespace).status
                if (status.available_replicas or 0) >= required_replicas:
                    # Isolate the delete so that old_name already being gone
                    # (e.g. a parallel scaling decision already removed it)
                    # does not prevent us from marking new_name as active.
                    try:
                        self.api.delete_namespaced_deployment(old_name, self.namespace)
                    except Exception:
                        pass
                    self.deployment = new_name   # thread-safe property setter
                    return
            except Exception:
                pass
            time.sleep(2)
        # Timed out — roll back by deleting the new Deployment; old stays active
        try:
            self.api.delete_namespaced_deployment(new_name, self.namespace)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, decision):
        # Read current active Deployment name under the lock so we don't race
        # with an in-progress background cleanup thread.
        current_name = self.deployment   # thread-safe property getter

        old       = self.api.read_namespaced_deployment(current_name, self.namespace)
        container = old.spec.template.spec.containers[0]
        current   = (container.resources.requests.get('cpu')
                     if container.resources and container.resources.requests else None)
        target    = f'{decision.cpu_millicores}m'

        # Pure horizontal scale — no new Deployment needed
        if current == target:
            self.api.patch_namespaced_deployment_scale(
                current_name, self.namespace, {'spec': {'replicas': decision.replicas}})
            return {'action': 'horizontal', 'deployment': current_name}

        # ---------------------------------------------------------------
        # Give the new Deployment a unique generation label so its selector
        # does NOT overlap with the old Deployment's selector.
        # ---------------------------------------------------------------
        gen  = self._gen_label(decision.cpu_millicores)
        body = copy.deepcopy(old)

        # Clear server-managed metadata so create_namespaced_deployment works
        body.metadata.resource_version   = None
        body.metadata.uid                = None
        body.metadata.creation_timestamp = None
        body.metadata.generate_name      = None

        new_name           = f'{current_name}-cpu-{decision.cpu_millicores}-{gen}'
        body.metadata.name = new_name

        # Add the discriminator to selector AND pod-template labels
        body.spec.selector.match_labels['cpu-gen']       = gen
        body.spec.template.metadata.labels['cpu-gen']    = gen

        # Update replica count and CPU resources
        body.spec.replicas = decision.replicas
        for item in body.spec.template.spec.containers:
            item.resources.requests['cpu'] = target
            item.resources.limits['cpu']   = target

        self.api.create_namespaced_deployment(self.namespace, body)

        # Non-blocking: poll in background thread so HTTP workers stay free.
        # Pass old_name as a local snapshot so the thread doesn't need to
        # re-read self.deployment (which may change under a future execute()).
        t = threading.Thread(
            target=self._poll_and_cleanup,
            args=(new_name, current_name, decision.replicas),
            daemon=True)
        t.start()

        return {'action': 'rolling-hybrid', 'deployment': new_name,
                'gen_label': gen, 'async_cleanup': True}
