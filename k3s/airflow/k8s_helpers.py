"""
Shared Kubernetes helpers for MLOps DAGs.

Provides idempotent delete functions for SparkApplication and RayJob resources,
and a helper to generate RFC-1123 compliant Kubernetes resource names from run IDs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def delete_spark_app(client, namespace: str, name: str) -> None:
    """Delete a SparkApplication by name. Idempotent: swallows 404, re-raises on other errors."""
    try:
        client.delete_namespaced_custom_object(
            group="spark.apache.org",
            version="v1",
            namespace=namespace,
            plural="sparkapplications",
            name=name,
        )
        logger.info("Deleted SparkApplication %s in namespace %s", name, namespace)
    except Exception as e:
        if "404" in str(e) or "Not Found" in str(e):
            logger.debug("SparkApplication %s not found (already deleted or never created)", name)
            return
        logger.error("Failed to delete SparkApplication %s: %s", name, e)
        raise


def delete_ray_job(client, namespace: str, name: str) -> None:
    """Delete a RayJob by name. Idempotent: swallows 404, re-raises on other errors."""
    try:
        client.delete_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=namespace,
            plural="rayjobs",
            name=name,
        )
        logger.info("Deleted RayJob %s in namespace %s", name, namespace)
    except Exception as e:
        if "404" in str(e) or "Not Found" in str(e):
            logger.debug("RayJob %s not found (already deleted or never created)", name)
            return
        logger.error("Failed to delete RayJob %s: %s", name, e)
        raise


def patch_ray_service_config(
    client,
    namespace: str,
    ray_service_name: str,
    params_s3_path: str,
) -> None:
    """Patch a RayService's serveConfigV2 to inject PARAMS_SERVING_S3_PATH.

    Strategy: GET the current RayService, parse the serveConfigV2 YAML string,
    inject PARAMS_SERVING_S3_PATH into every application's runtime_env.env_vars,
    re-serialize, and MERGE-patch spec.serveConfigV2. This triggers KubeRay to
    perform a rolling reload of Ray Serve deployments.

    Idempotent: patching the same value twice leaves the resource unchanged.

    Args:
        client:           kubernetes.client.CustomObjectsApi instance
        namespace:        K8s namespace where the RayService lives (e.g. "ray")
        ray_service_name: Name of the RayService resource (e.g. "model-serving")
        params_s3_path:   Full S3 URI of params_serving.yaml to inject
    """
    import copy
    import yaml as _yaml

    obj = client.get_namespaced_custom_object(
        group="ray.io",
        version="v1",
        namespace=namespace,
        plural="rayservices",
        name=ray_service_name,
    )

    serve_config_str = obj.get("spec", {}).get("serveConfigV2", "")
    serve_config = _yaml.safe_load(serve_config_str) or {}
    updated = copy.deepcopy(serve_config)

    for app in updated.get("applications", []):
        runtime_env = app.setdefault("runtime_env", {})
        env_vars = runtime_env.setdefault("env_vars", {})
        env_vars["PARAMS_SERVING_S3_PATH"] = params_s3_path

    patch_body = {"spec": {"serveConfigV2": _yaml.dump(updated, default_flow_style=False)}}

    client.patch_namespaced_custom_object(
        group="ray.io",
        version="v1",
        namespace=namespace,
        plural="rayservices",
        name=ray_service_name,
        body=patch_body,
    )
    logger.info(
        "Patched RayService %s/%s: PARAMS_SERVING_S3_PATH=%s",
        namespace, ray_service_name, params_s3_path,
    )


def k8s_name(run_id: str, prefix: str, max_len: int = 63) -> str:
    """Generate an RFC-1123 compliant Kubernetes resource name from a run ID.

    Replaces underscores with hyphens, lowercases, and truncates to max_len.
    When truncation is needed, the unique trailing hash segment is preserved
    so that distinct run IDs always produce distinct resource names.

    Args:
        run_id:   Run ID (e.g. "pre-network_traffic-20260302T143022Z-a4f91c")
        prefix:   Short resource type prefix (e.g. "spark-preprocess")
        max_len:  Maximum length (default 63 per RFC 1123).
                  RayJob names must be ≤ 47 — pass max_len=47 for those.

    Returns:
        A valid Kubernetes name string.
    """
    safe = run_id.replace("_", "-").lower()
    full = f"{prefix}-{safe}"
    if len(full) <= max_len:
        return full

    # Preserve the unique trailing hash (last hyphen-delimited segment) so
    # that different run IDs never collide after truncation.
    suffix = safe.rsplit("-", 1)[-1]          # e.g. "b929e3"
    available = max_len - len(prefix) - len(suffix) - 2   # 2 hyphens
    middle = safe[:available].rstrip("-")
    return f"{prefix}-{middle}-{suffix}"
