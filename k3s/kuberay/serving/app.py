import os
import random
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import boto3
import yaml

from ray import serve
from starlette.requests import Request

from ray.util.metrics import Counter, Histogram

from src.utils.logger import create_logger
from src.utils.baseclass import BaseUtils

@dataclass(frozen=True)
class ModelSpec:
    framework: str
    model_key: str
    artifacts_key: str


def load_params() -> Dict[str, Any]:
    """Load common parameters from params.yaml."""
    params_path = os.getenv("PARAMS_PATH", "/home/ray/app/repo/k3s/params.yaml")
    try:
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                return yaml.safe_load(f)
    except Exception:
        pass
    return {}


class S3Store:
    def __init__(self, *, bucket: str, endpoint_url: Optional[str] = None):
        self._logger = create_logger("S3Store")
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-2"),
            endpoint_url=endpoint_url or os.getenv("S3_ENDPOINT_URL") or None,
        )

    def download_to_tmp(self, *, key: str, filename: str) -> str:
        local_path = os.path.join(tempfile.gettempdir(), filename)
        self._logger.info("Downloading s3://%s/%s -> %s", self._bucket, key, local_path)
        self._client.download_file(self._bucket, key, local_path)
        return local_path


def _normalize_payload(payload: Any) -> List[Dict[str, Any]]:
    """Normalize request payload into a strict matrix.

    Contract: Ray Serve is schema-agnostic and does NOT accept column names.
    Spark guarantees ordering and types upstream.

    Accepted payloads:
    - {"data": [[...], [...]]}
    - {"data": [...]}  (single row)
    """
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    if "data" not in payload:
        raise ValueError("Missing 'data' field")

    data = payload["data"]
    if isinstance(data, list) and (len(data) == 0 or not isinstance(data[0], (list, tuple))):
        # single row vector
        data = [data]

    if not isinstance(data, list) or any(not isinstance(r, (list, tuple)) for r in data):
        raise ValueError("'data' must be a list of lists (matrix) or a single list (row)")

    lengths = {len(r) for r in data}
    if len(lengths) != 1:
        raise ValueError(f"All rows must have the same length. Got lengths={sorted(lengths)}")

    return [list(r) for r in data]


class ModelAdapter(Protocol):
    def predict(self, data: List[List[Any]]) -> Dict[str, Any]:
        ...


class XGBoostAdapter:
    def __init__(self, model_path: str):
        from src.serve.xgboost import XGBoostHandler

        self._handler = XGBoostHandler(model_path)

    def predict(self, data: List[List[Any]]) -> Dict[str, Any]:
        if not isinstance(data, list) or any(not isinstance(r, list) for r in data):
            raise TypeError("data must be List[List]")
        return self._handler.predict(data)


class PyTorchAdapter:
    def __init__(self, model_path: str, *, input_dim: int, num_classes: int):
        from src.serve.pytorch import PyTorchHandler

        self._handler = PyTorchHandler(model_path, input_dim=input_dim, num_classes=num_classes)

    def predict(self, data: List[List[Any]]) -> Dict[str, Any]:
        if not isinstance(data, list) or any(not isinstance(r, list) for r in data):
            raise TypeError("data must be List[List]")
        return self._handler.predict(data)


class ModelFactory:
    @staticmethod
    def count_input_dim_from_dsl(params_path: Optional[str] = None) -> int:
        """Count input dim (categorical + numerical) from DSL referenced in params.

        This uses `BaseUtils.load_params()` so we don't duplicate YAML loading logic.
        """
        logger = create_logger("ModelFactory")
        params_file = params_path or os.getenv("PARAMS_PATH", "/home/ray/app/repo/k3s/params.yaml")
        utils = BaseUtils(logger, params_file)
        try:
            params = utils.load_params()
            dsl_path = params.get('spark', {}).get('preprocessing', {}).get(
                'dsl_path', '/app/repo/k3s/spark/preprocess/dsl_001.yaml'
            )
            # Resolve under /home/ray so relative paths used in cluster match container layout
            dsl_path = os.path.join('/home/ray/', dsl_path.lstrip('/'))
            with open(dsl_path, 'r') as f:
                doc = yaml.safe_load(f)
            final_features = doc.get('pipeline', {}).get('final_features', {})
            categorical = final_features.get('categorical', []) or []
            numerical = final_features.get('numerical', []) or []
            return int(len(categorical) + len(numerical))
        except Exception:
            # Fallback to explicit input_dim in params or default 14
            try:
                cfg = params.get('kuberay', {}).get('model', {}) if 'params' in locals() else {}
                return int(cfg.get('input_dim', 14))
            except Exception:
                return 14

    @staticmethod
    def create_adapter(framework: str, *, model_path: str, params: Dict[str, Any] = None) -> ModelAdapter:
        fw = framework.strip().lower()
        if fw == "xgboost":
            return XGBoostAdapter(model_path)
        if fw == "pytorch":
            model_cfg = (params or {}).get("kuberay", {}).get("model", {})
            if model_cfg.get('dsl_count_dim'):
                input_dim = ModelFactory.count_input_dim_from_dsl()
            else:
                input_dim = int(model_cfg.get("input_dim", 14))
            num_classes = int(model_cfg.get("num_classes"))
            return PyTorchAdapter(model_path, input_dim=input_dim, num_classes=num_classes)
        raise ValueError(f"Unsupported MODEL_FRAMEWORK={framework!r}")


class _ModelRuntime:
    """Shared (non-deployment) runtime.

    IMPORTANT: Do not subclass a class decorated by @serve.deployment.
    Ray wraps deployment classes, and Python inheritance breaks with that wrapper.
    """

    def __init__(self, *, name: str, variant: str):
        self._logger = create_logger(name)
        self._variant = variant
        self._store: Optional[S3Store] = None
        self._adapter: Optional[ModelAdapter] = None
        self._spec: Optional[ModelSpec] = None

        # Custom metrics exported via Ray's Prometheus endpoint (metrics-export-port).
        # These let us break down traffic by model variant/framework (stable/canary, xgboost/pytorch).
        self._requests_total = Counter(
            "serve_infer_requests_total",
            description="Total inference requests handled by a model deployment",
            tag_keys=("application", "variant", "framework"),
        )
        self._errors_total = Counter(
            "serve_infer_errors_total",
            description="Total inference errors raised by a model deployment",
            tag_keys=("application", "variant", "framework"),
        )
        self._latency_ms = Histogram(
            "serve_infer_latency_ms",
            description="End-to-end latency inside the model deployment (ms)",
            boundaries=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000],
            tag_keys=("application", "variant", "framework"),
        )

    def _load_from_config(self, config: Dict[str, Any]) -> None:
        params = load_params()
        model_cfg = params.get("kuberay", {}).get("model", {})
        serving_cfg = params.get("kuberay", {}).get("serving", {})

        bucket = os.getenv("S3_BUCKET_NAME", "k8s-mlops-platform-bucket")
        
        # Priority: Config > serving.framework > model.framework > ENV > default
        framework = str(
            config.get(
                "framework", 
                serving_cfg.get(
                    "framework", 
                    model_cfg.get("framework", os.getenv("MODEL_FRAMEWORK", "xgboost"))
                )
            )
        )
        model_key = str(config.get("model_key", os.getenv("MODEL_KEY", f"v1/models/model_{framework}.pkl")))
        artifacts_key = str(
            config.get(
                "artifacts_key",
                os.getenv("ARTIFACTS_KEY", "v1/artifacts/pipeline_model.json"),
            )
        )
        self._spec = ModelSpec(framework=framework, model_key=model_key, artifacts_key=artifacts_key)

        self._store = S3Store(bucket=bucket)
        model_path = self._store.download_to_tmp(
            key=self._spec.model_key,
            filename=f"{self._variant}_{framework}.pkl",
        )

        # NOTE: preprocessing lives in Spark. Serving is pure inference only.
        self._adapter = ModelFactory.create_adapter(framework, model_path=model_path, params=params)

        self._logger.info("Model loaded (%s): %s", self._variant, self._spec)

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._adapter is None or self._spec is None:
            raise RuntimeError("Model not initialized")

        tags = {
            "application": "inference",
            "variant": self._variant,
            "framework": self._spec.framework,
        }

        self._requests_total.inc(1, tags=tags)

        started = time.perf_counter()
        try:
            matrix = _normalize_payload(payload)
            result = self._adapter.predict(matrix)
        except Exception:
            self._errors_total.inc(1, tags=tags)
            raise
        finally:
            self._latency_ms.observe((time.perf_counter() - started) * 1000.0, tags=tags)

        result["latency_ms"] = (time.perf_counter() - started) * 1000.0
        result["model"] = {
            "variant": self._variant,
            "framework": self._spec.framework,
            "model_key": self._spec.model_key,
        }
        return result


@serve.deployment(name="StableModel")
class StableModel:
    def __init__(self):
        self._rt = _ModelRuntime(name="StableModel", variant="stable")

    def reconfigure(self, config: Dict[str, Any]) -> None:
        try:
            self._rt._load_from_config(config)
        except Exception as e:
            create_logger("StableModel").error("reconfigure failed: %s", str(e), exc_info=True)
            raise

    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._rt.predict(payload)
        except Exception as e:
            create_logger("StableModel").error("predict failed: %s", str(e), exc_info=True)
            raise


@serve.deployment(name="CanaryModel")
class CanaryModel:
    def __init__(self):
        self._rt = _ModelRuntime(name="CanaryModel", variant="canary")

    def reconfigure(self, config: Dict[str, Any]) -> None:
        try:
            self._rt._load_from_config(config)
        except Exception as e:
            create_logger("CanaryModel").error("reconfigure failed: %s", str(e), exc_info=True)
            raise

    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._rt.predict(payload)
        except Exception as e:
            create_logger("CanaryModel").error("predict failed: %s", str(e), exc_info=True)
            raise


@serve.deployment(name="ModelRouter")
class ModelRouter:
    def __init__(self, stable, canary):
        self._logger = create_logger("ModelRouter")
        self._stable = stable
        self._canary = canary
        self._canary_probability = 0.0

    def reconfigure(self, config: Dict[str, Any]) -> None:
        params = load_params()
        serving_cfg = params.get("kuberay", {}).get("serving", {})
        
        # Priority: Config > serving.canary_probability > default
        p = float(config.get("canary_probability", serving_cfg.get("canary_probability", 0.0)))
        self._canary_probability = max(0.0, min(1.0, p))
        self._logger.info("Router configured: canary_probability=%s", self._canary_probability)

    async def __call__(self, request: Request):
        if request.url.path.endswith("/healthz"):
            return {"status": "ok"}

        payload = await request.json()

        use_canary = random.random() < self._canary_probability
        handle = self._canary if use_canary else self._stable
        # Delegate prediction to the chosen model deployment.
        return await handle.predict.remote(payload)


# Serve application graph.
# Note: deployment names are pinned above to match serveConfigV2 updates.
deployment_graph = ModelRouter.bind(StableModel.bind(), CanaryModel.bind())
