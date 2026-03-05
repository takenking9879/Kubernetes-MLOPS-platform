"""
RunPod and Tailscale helpers for the GPU training DAG.

Provides:
  - RunPodClient  : provision / terminate / poll RunPod GPU pods via GraphQL API
  - generate_tailscale_ephemeral_key : create a one-use Tailscale auth key
  - get_head_pod_ip    : resolve Ray head pod IP from K8s for local Docker PoC
  - wait_for_ray_worker: poll Ray GCS until the GPU worker appears in the cluster
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ─── RunPod ──────────────────────────────────────────────────────────────────

_RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"

_MUTATION_CREATE_POD = """
mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
    desiredStatus
    runtime {
      uptimeInSeconds
      ports {
        ip
        isIpPublic
        privatePort
        publicPort
        type
      }
    }
  }
}
"""

_MUTATION_TERMINATE_POD = """
mutation TerminatePod($input: PodTerminateInput!) {
  podTerminate(input: $input)
}
"""

_QUERY_POD_STATUS = """
query PodStatus($podId: String!) {
  pod(input: { podId: $podId }) {
    id
    name
    desiredStatus
    runtime {
      uptimeInSeconds
    }
  }
}
"""


class RunPodClient:
    """Thin wrapper around the RunPod GraphQL API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ["RUNPOD_API_KEY"]
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})

    def _gql(self, query: str, variables: dict) -> dict:
        resp = self._session.post(
            _RUNPOD_GRAPHQL,
            json={"query": query, "variables": variables},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        if "errors" in body:
            raise RuntimeError(f"RunPod GraphQL error: {body['errors']}")
        return body["data"]

    def provision(
        self,
        *,
        image: str,
        gpu_type: str,
        ray_head_address: str,
        tailscale_authkey: str,
        num_cpus: int = 10,
        container_disk_gb: int = 30,
    ) -> str:
        """Provision a GPU pod on RunPod.

        Returns the pod ID string.
        """
        data = self._gql(
            _MUTATION_CREATE_POD,
            {
                "input": {
                    "imageName": image,
                    "gpuTypeId": gpu_type,
                    "containerDiskInGb": container_disk_gb,
                    "volumeInGb": 0,
                    "ports": "10001/tcp",
                    "env": [
                        {"key": "TAILSCALE_AUTHKEY",  "value": tailscale_authkey},
                        {"key": "RAY_HEAD_ADDRESS",   "value": ray_head_address},
                        {"key": "NUM_CPUS",           "value": str(num_cpus)},
                        {"key": "PYTHONPATH",
                         "value": "/home/ray/app/repo:/home/ray/app/repo/src"},
                    ],
                }
            },
        )
        pod_id: str = data["podFindAndDeployOnDemand"]["id"]
        logger.info("Provisioned RunPod pod %s (gpu_type=%s)", pod_id, gpu_type)
        return pod_id

    def terminate(self, pod_id: str) -> None:
        """Terminate a RunPod pod by ID. Idempotent — swallows 'not found' errors."""
        try:
            self._gql(_MUTATION_TERMINATE_POD, {"input": {"podId": pod_id}})
            logger.info("Terminated RunPod pod %s", pod_id)
        except Exception as exc:
            if "not found" in str(exc).lower() or "404" in str(exc):
                logger.debug("RunPod pod %s already gone", pod_id)
                return
            raise

    def get_status(self, pod_id: str) -> str:
        """Return desiredStatus string: 'RUNNING', 'EXITED', 'PENDING', etc."""
        data = self._gql(_QUERY_POD_STATUS, {"podId": pod_id})
        pod = data.get("pod") or {}
        return pod.get("desiredStatus", "UNKNOWN")

    def wait_for_running(self, pod_id: str, timeout: int = 300, interval: int = 10) -> None:
        """Block until pod desiredStatus == 'RUNNING' or timeout is reached."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.get_status(pod_id)
            logger.info("RunPod pod %s status: %s", pod_id, status)
            if status == "RUNNING":
                return
            if status in ("EXITED", "TERMINATED", "FAILED"):
                raise RuntimeError(
                    f"RunPod pod {pod_id} reached terminal state {status} before becoming RUNNING"
                )
            time.sleep(interval)
        raise TimeoutError(
            f"RunPod pod {pod_id} did not reach RUNNING within {timeout}s"
        )


# ─── Tailscale ───────────────────────────────────────────────────────────────

_TAILSCALE_KEYS_URL = "https://api.tailscale.com/api/v2/tailnet/{tailnet}/keys"


def generate_tailscale_ephemeral_key(
    tailnet: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tags: list[str] | None = None,
    expiry_seconds: int = 86400,
) -> str:
    """Generate a one-use Tailscale ephemeral auth key via the Tailscale OAuth API.

    The key is ephemeral (device removed from tailnet when it disconnects) and
    non-reusable (can only be used to authenticate one device).

    Required env vars (or pass explicitly):
      TAILSCALE_TAILNET        — e.g. "your-org.github" or "-" for personal tailnet
      TAILSCALE_CLIENT_ID      — OAuth client ID from admin console
      TAILSCALE_CLIENT_SECRET  — OAuth client secret

    Args:
        tailnet:        Tailnet name. Defaults to TAILSCALE_TAILNET env var.
        client_id:      OAuth client ID. Defaults to TAILSCALE_CLIENT_ID env var.
        client_secret:  OAuth client secret. Defaults to TAILSCALE_CLIENT_SECRET env var.
        tags:           ACL tags to apply to the device (e.g. ["tag:runpod-worker"]).
        expiry_seconds: Key TTL in seconds (default: 24 hours).

    Returns:
        Tailscale auth key string.
    """
    tailnet = tailnet or os.environ["TAILSCALE_TAILNET"]
    client_id = client_id or os.environ["TAILSCALE_CLIENT_ID"]
    client_secret = client_secret or os.environ["TAILSCALE_CLIENT_SECRET"]
    tags = tags or ["tag:runpod-worker"]

    # Get an OAuth access token first
    token_resp = requests.post(
        "https://api.tailscale.com/api/v2/oauth/token",
        data={"client_id": client_id, "client_secret": client_secret},
        timeout=15,
    )
    token_resp.raise_for_status()
    access_token: str = token_resp.json()["access_token"]

    # Create the ephemeral auth key
    key_resp = requests.post(
        _TAILSCALE_KEYS_URL.format(tailnet=tailnet),
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "capabilities": {
                "devices": {
                    "create": {
                        "reusable": False,
                        "ephemeral": True,
                        "tags": tags,
                    }
                }
            },
            "expirySeconds": expiry_seconds,
        },
        timeout=15,
    )
    key_resp.raise_for_status()
    auth_key: str = key_resp.json()["key"]
    logger.info("Generated Tailscale ephemeral key (ttl=%ds, tags=%s)", expiry_seconds, tags)
    return auth_key


# ─── K8s helpers ─────────────────────────────────────────────────────────────


def get_head_pod_ip(namespace: str = "ray") -> str:
    """Return the Ray head pod IP inside the K8s cluster.

    Used in Phase 0 (local Docker PoC) where the Docker container
    can reach pod IPs directly via --network=host.
    """
    from kubernetes import client as _k8s_client, config as _k8s_config

    try:
        _k8s_config.load_incluster_config()
    except Exception:
        _k8s_config.load_kube_config()

    v1 = _k8s_client.CoreV1Api()
    pods = v1.list_namespaced_pod(
        namespace=namespace,
        label_selector="ray.io/node-type=head",
    )
    if not pods.items:
        raise RuntimeError("No Ray head pod found in namespace '%s'" % namespace)
    pod = pods.items[0]
    ip: str = pod.status.pod_ip
    if not ip:
        raise RuntimeError("Ray head pod has no IP yet — is it still starting?")
    return ip


def wait_for_ray_worker(
    head_address: str,
    num_gpus_expected: int = 1,
    timeout: int = 240,
    interval: int = 10,
) -> None:
    """Poll Ray GCS until at least `num_gpus_expected` GPU resources are available.

    This ensures the external GPU worker has fully joined before the RayJob
    entrypoint tries to schedule GPU tasks.

    Args:
        head_address:      Ray GCS address, e.g. "10.244.x.x:6379"
        num_gpus_expected: Minimum GPU count to wait for.
        timeout:           Seconds before giving up.
        interval:          Polling interval in seconds.
    """
    import ray as _ray

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if not _ray.is_initialized():
                _ray.init(address=f"ray://{head_address}", ignore_reinit_error=True)

            resources = _ray.cluster_resources()
            gpus_available = resources.get("GPU", 0)
            logger.info(
                "Cluster resources: GPU=%.0f (waiting for %.0f)",
                gpus_available,
                num_gpus_expected,
            )
            if gpus_available >= num_gpus_expected:
                _ray.shutdown()
                logger.info("GPU worker is ready.")
                return
        except Exception as exc:
            logger.debug("Ray not reachable yet: %s", exc)
        finally:
            try:
                _ray.shutdown()
            except Exception:
                pass
        time.sleep(interval)

    raise TimeoutError(
        f"GPU worker did not appear in Ray cluster at {head_address} within {timeout}s"
    )
