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


def k8s_name(run_id: str, prefix: str, max_len: int = 63) -> str:
    """Generate an RFC-1123 compliant Kubernetes resource name from a run ID.

    Replaces underscores with hyphens, lowercases, and truncates to max_len.

    Args:
        run_id:   Run ID (e.g. "pre-network_traffic-20260302T143022Z-a4f91c")
        prefix:   Short resource type prefix (e.g. "spark-preprocess")
        max_len:  Maximum length (default 63 per RFC 1123)

    Returns:
        A valid Kubernetes name string.
    """
    safe = run_id.replace("_", "-").lower()
    return f"{prefix}-{safe}"[:max_len]
