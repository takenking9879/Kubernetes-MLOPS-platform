import os
import time
from typing import Any, Dict, List, Optional, Literal

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
        self._num_classes = num_classes
        self._adapter: Optional[ModelAdapter] = None
        self._artifact: Optional[ModelArtifact] = None

        # Resolve DSL metadata (input_dim + ordered feature list)
        self._input_dim, self._final_features = self._resolve_dsl_meta(
            dsl_path, input_dim_fallback
        )

        # NumPy executor for online (raw-payload) inference.
        # Injected via set_executor() by app.py when online mode is enabled.
        self._executor = None

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

    def set_executor(self, executor) -> None:
        """
        Inject a NumpyPipelineExecutor for online (raw-payload) inference.

        Called by app.py after PipelineArtifactLoader resolves the executor
        from the MLflow → Iceberg → S3 chain.  Called again on webhook-triggered
        reconfigure so the executor tracks the current champion model's artifacts.
        """
        self._executor = executor
        self._logger.info(
            "NumpyPipelineExecutor set (%d features).",
            len(executor.final_features) if executor is not None else 0,
        )

    def _resolve_dsl_meta(self, dsl_path: str, fallback: int):
        """Return (input_dim, ordered_feature_list) from the DSL YAML."""
        try:
            path = os.path.join("/home/ray/", dsl_path.lstrip("/"))
            if not os.path.exists(path):
                path = dsl_path

            with open(path, "r") as f:
                dsl = yaml.safe_load(f)

            ff = dsl.get("final_features", {})
            if not isinstance(ff, dict):
                raise ValueError("DSL final_features must be an object with key 'features'")

            features: List[str] = list(ff.get("features", []))
            self._logger.info("Input dimension resolved from DSL: %d", len(features))
            return len(features), features
        except Exception as e:
            self._logger.warning(
                "DSL meta resolution failed, using fallback dim=%d: %s", fallback, e
            )
            return fallback, []

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
            matrix = self._build_matrix(payload)
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

    def _build_matrix(self, payload: Dict[str, Any]) -> List[List[Any]]:
        """
        Convert a request payload to a feature matrix.

        Two payload formats are supported:

        1. Pre-processed (offline/batch Spark path):
               {"data": [[f1, f2, ..., f14]]}
           Spark ran the DSL pipeline and sends the 14 final features directly.
           Passed through without any preprocessing.

        2. Schema-converted event (online path):
               {"raw": {"timestamp": 1735691403, "event_id": "...",
                        "src_port": 12345, "dst_port": 80, ...}}
           Spark ran kafka_to_schema_features (schema conversion only, no DSL).
           The NumpyPipelineExecutor runs the full DSL preprocessing here in Ray Serve.
        """
        if "raw" in payload:
            if self._executor is None:
                raise RuntimeError(
                    "Raw-payload inference is unavailable: NumpyPipelineExecutor not loaded. "
                    "Ensure kuberay.serving.online=true and pipeline artifacts resolved from S3."
                )
            # Spark already did kafka_to_schema_features (schema conversion).
            # The dict is already flat and typed — pass directly to the executor.
            feature_vec = self._executor.transform_to_vector(payload["raw"])
            return [feature_vec]

        # Default: pre-processed numeric matrix from Spark DSL pipeline
        return normalize_payload(payload)
