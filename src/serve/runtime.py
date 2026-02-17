import os
import time
from typing import Any, Dict, Optional, Literal

import yaml
from ray.util.metrics import Counter, Histogram

from src.serve.adapters import AdapterFactory, ModelAdapter, normalize_payload
from src.serve.registry import MLflowRegistry, ModelArtifact
from src.utils.logger import create_logger


class ModelRuntime:
    def __init__(
        self,
        name: str,
        variant: Literal["stable", "canary"],
        registry: MLflowRegistry,
        registry_name: str,
        default_alias: str,
        input_dim_fallback: int,
        num_classes: int,
        dsl_path: str,
    ):
        self._logger = create_logger(name)
        self._variant = variant
        self._registry = registry
        self._registry_name = registry_name
        self._default_alias = default_alias
        self._input_dim = self._resolve_input_dim(dsl_path, input_dim_fallback)
        self._num_classes = num_classes
        self._adapter: Optional[ModelAdapter] = None
        self._artifact: Optional[ModelArtifact] = None

        self._requests_total = Counter(
            "serve_infer_requests_total",
            description="Total inference requests",
            tag_keys=("application", "variant", "framework"),
        )
        self._errors_total = Counter(
            "serve_infer_errors_total",
            description="Total inference errors",
            tag_keys=("application", "variant", "framework"),
        )
        self._latency_ms = Histogram(
            "serve_infer_latency_ms",
            description="Inference latency (ms)",
            boundaries=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000],
            tag_keys=("application", "variant", "framework"),
        )

    def _resolve_input_dim(self, dsl_path: str, fallback: int) -> int:
        try:
            path = os.path.join("/home/ray/", dsl_path.lstrip("/"))
            if not os.path.exists(path):
                path = dsl_path

            with open(path, "r") as f:
                dsl = yaml.safe_load(f)

            final_features = dsl.get("final_features", {})
            if not isinstance(final_features, dict):
                raise ValueError("DSL final_features must be an object with key 'features'")

            dim = len(final_features.get("features", []))

            self._logger.info("Input dimension resolved from DSL: %d", dim)
            return dim
        except Exception as e:
            self._logger.warning("DSL dimension resolution failed, using fallback %d: %s", fallback, e)
            return fallback

    def load(self, alias_override: Optional[str] = None) -> None:
        alias = alias_override or self._default_alias
        artifact = self._registry.load_by_alias(self._registry_name, alias)

        self._artifact = artifact
        self._adapter = AdapterFactory.create(artifact.framework, artifact.model)

        self._logger.info(
            "Runtime loaded (%s): %s@%s v%s (%s)",
            self._variant,
            artifact.registry_name,
            artifact.alias,
            artifact.version,
            artifact.framework.value,
        )

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._adapter is None or self._artifact is None:
            raise RuntimeError("Model not initialized")

        tags = {
            "application": "inference",
            "variant": self._variant,
            "framework": self._artifact.framework.value,
        }

        self._requests_total.inc(1, tags=tags)
        started = time.perf_counter()

        try:
            matrix = normalize_payload(payload)
            if len(matrix[0]) != self._input_dim:
                raise ValueError(
                    f"Matrix dimension mismatch. Expected {self._input_dim}, got {len(matrix[0])}"
                )
            result = self._adapter.predict(matrix)
        except Exception:
            self._errors_total.inc(1, tags=tags)
            raise
        finally:
            latency = (time.perf_counter() - started) * 1000.0
            self._latency_ms.observe(latency, tags=tags)

        result["latency_ms"] = latency
        result["model"] = {
            "variant": self._variant,
            "framework": self._artifact.framework.value,
            "registry": self._artifact.registry_name,
            "alias": self._artifact.alias,
            "version": self._artifact.version,
        }
        return result
