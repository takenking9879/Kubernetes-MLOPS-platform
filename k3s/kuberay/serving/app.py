import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import yaml

from ray import serve
from starlette.requests import Request

from ray.util.metrics import Counter, Histogram

from src.utils.logger import create_logger
from src.utils.baseclass import BaseUtils


@dataclass(frozen=True)
class ModelSpec:
    framework: Optional[str]
    registry_name: str
    alias: str
    version: Optional[str] = None


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


def _normalize_payload(payload: Any) -> List[List[Any]]:
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


# ──────────────────────────────────────────────────────────────────────────────
# MLflow Model Registry loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_from_registry(
    *,
    tracking_uri: str,
    registry_name: str,
    alias: str,
    framework: Optional[str],
    logger,
):
    """Load a model from MLflow Model Registry using an alias (MLflow 3.x).

    Returns ``(model_object, model_version_str, resolved_framework)``.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Resolve alias → version
    mv = client.get_model_version_by_alias(registry_name, alias)
    version = str(mv.version)
    model_uri = f"models:/{registry_name}@{alias}"

    resolved_framework = (framework or "").strip().lower() or None
    if not resolved_framework:
        mv_tags = getattr(mv, "tags", {}) or {}
        resolved_framework = str(mv_tags.get("framework", "")).strip().lower() or None

    if not resolved_framework and getattr(mv, "run_id", None):
        run = client.get_run(mv.run_id)
        run_tags = getattr(run.data, "tags", {}) or {}
        run_params = getattr(run.data, "params", {}) or {}
        resolved_framework = (
            str(run_tags.get("framework", "")).strip().lower()
            or str(run_params.get("framework", "")).strip().lower()
            or None
        )

    if not resolved_framework:
        raise RuntimeError(
            "Unable to resolve model framework from configuration or MLflow metadata. "
            "Set model version tag 'framework' (e.g. xgboost, pytorch)."
        )

    logger.info(
        "Loading model from MLflow: %s @%s (v%s, framework=%s)",
        registry_name, alias, version, resolved_framework,
    )

    if resolved_framework == "xgboost":
        import mlflow.xgboost
        model = mlflow.xgboost.load_model(model_uri)
    elif resolved_framework == "pytorch":
        import mlflow.pytorch
        model = mlflow.pytorch.load_model(model_uri)
    else:
        raise ValueError(f"Unsupported framework for MLflow loading: {resolved_framework!r}")

    logger.info("Model loaded successfully: %s @%s v%s", registry_name, alias, version)
    return model, version, resolved_framework


class ModelAdapter(Protocol):
    def predict(self, data: List[List[Any]]) -> Dict[str, Any]:
        ...


class XGBoostAdapter:
    """Wraps an XGBoost Booster for serving predictions."""

    def __init__(self, model):
        import xgboost as xgb
        import numpy as np

        self._model = model
        self._xgb = xgb
        self._np = np

    def predict(self, data: List[List[Any]]) -> Dict[str, Any]:
        if not isinstance(data, list) or any(not isinstance(r, list) for r in data):
            raise TypeError("data must be List[List]")
        import pandas as pd

        df = pd.DataFrame(data)
        # Map feature names when dimensions match
        if hasattr(self._model, "feature_names") and self._model.feature_names:
            if len(df.columns) == len(self._model.feature_names):
                first_col = df.columns[0]
                is_int_col = isinstance(first_col, int) or (
                    isinstance(first_col, str) and first_col.isdigit()
                )
                if is_int_col:
                    df.columns = self._model.feature_names

        dmatrix = self._xgb.DMatrix(df)
        probs = self._model.predict(dmatrix)
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            predictions = self._np.argmax(probs, axis=1)
        else:
            predictions = (probs > 0.5).astype(int)
        return {"predictions": predictions.tolist(), "probabilities": probs.tolist()}


class PyTorchAdapter:
    """Wraps a PyTorch nn.Module for serving predictions."""

    def __init__(self, model):
        import torch

        self._model = model
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def predict(self, data: List[List[Any]]) -> Dict[str, Any]:
        if not isinstance(data, list) or any(not isinstance(r, list) for r in data):
            raise TypeError("data must be List[List]")

        tensor_data = self._torch.tensor(data, dtype=self._torch.float32).to(self._device)
        with self._torch.no_grad():
            outputs = self._model(tensor_data)
            probs = self._torch.softmax(outputs, dim=1)
            predictions = self._torch.argmax(probs, dim=1)
        return {
            "predictions": predictions.cpu().numpy().tolist(),
            "probabilities": probs.cpu().numpy().tolist(),
        }


def _create_adapter(framework: str, model) -> ModelAdapter:
    """Wrap a loaded model object into the serving adapter."""
    fw = framework.strip().lower()
    if fw == "xgboost":
        return XGBoostAdapter(model)
    if fw == "pytorch":
        return PyTorchAdapter(model)
    raise ValueError(f"Unsupported framework={framework!r}")


class _ModelRuntime:
    """Shared (non-deployment) runtime.

    Loads models from **MLflow Model Registry** using aliases (MLflow 3.x).
    Each variant (stable / canary) maps to an alias (champion / challenger).

    IMPORTANT: Do not subclass a class decorated by @serve.deployment.
    Ray wraps deployment classes, and Python inheritance breaks with that wrapper.
    """

    def __init__(self, *, name: str, variant: str):
        self._logger = create_logger(name)
        self._variant = variant
        self._adapter: Optional[ModelAdapter] = None
        self._spec: Optional[ModelSpec] = None

        # Custom metrics exported via Ray's Prometheus endpoint (metrics-export-port).
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

    def _load_from_config(
        self,
        config: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load model from MLflow Model Registry.

                Resolution behavior: registry name and alias come from config, but
                `framework` is resolved from MLflow metadata (model version tag or run).
        """
        params = load_params()
        model_cfg = params.get("kuberay", {}).get("model", {})
        serving_cfg = params.get("kuberay", {}).get("serving", {})
        overrides = overrides or {}


        tracking_uri = (
            overrides.get("mlflow_tracking_uri")
            or model_cfg.get("mlflow_tracking_uri")
            or os.getenv("MLFLOW_TRACKING_URI", "http://my-mlflow")
        )

        registry_name = (
            overrides.get("mlflow_registry_model_name")
            or model_cfg.get("mlflow_registry_model_name")
            or serving_cfg.get("mlflow_registry_model_name")
        )
        if not registry_name:
            raise RuntimeError(
                "Missing required configuration: "
                "kuberay.model.mlflow_registry_model_name in params.yaml"
            )

        default_alias = "challenger" if self._variant == "canary" else "champion"
        alias = (
            overrides.get("alias")
            or (serving_cfg.get("alias") if self._variant != "canary" else None)
            or default_alias
        )

        # Always resolve framework from MLflow metadata; do not rely on params.yaml
        model, version, resolved_framework = _load_model_from_registry(
            tracking_uri=tracking_uri,
            registry_name=registry_name,
            alias=alias,
            framework=None,
            logger=self._logger,
        )

        self._spec = ModelSpec(
            framework=resolved_framework,
            registry_name=registry_name,
            alias=alias,
            version=version,
        )
        self._adapter = _create_adapter(resolved_framework, model)

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
            "registry": self._spec.registry_name,
            "alias": self._spec.alias,
            "version": self._spec.version,
        }
        return result


@serve.deployment(name="StableModel")
class StableModel:
    def __init__(self):
        self._rt = _ModelRuntime(name="StableModel", variant="stable")
        try:
            # Ensure the model is loaded on startup even if reconfigure isn't invoked.
            self._rt._load_from_config({})
        except Exception as e:
            create_logger("StableModel").error("initial load failed: %s", str(e), exc_info=True)
            raise

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
        try:
            params = load_params()
            canary_cfg = params.get("kuberay", {}).get("canary", {})
            if canary_cfg:
                self._rt._load_from_config({}, overrides=canary_cfg)
            else:
                self._rt._load_from_config({})
        except Exception as e:
            create_logger("CanaryModel").error("initial load failed: %s", str(e), exc_info=True)
            raise

    def reconfigure(self, config: Dict[str, Any]) -> None:
        try:
            params = load_params()
            canary_cfg = params.get("kuberay", {}).get("canary", {})
            # pass overrides so canary can use a different framework/model_key
            if canary_cfg:
                self._rt._load_from_config(config, overrides=canary_cfg)
            else:
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
        canary_cfg = params.get("kuberay", {}).get("canary", {})
        # Resolution priority for canary probability:
        # 1. explicit `canary_probability` in `config` passed to reconfigure
        # 2. if `config` sets `canary: true` -> use `canary.canary_probability` from params or config
        # 3. if `serving.canary` is true -> use `kuberay.canary.canary_probability` from params
        # 4. default = 0.0
        if "canary_probability" in config:
            p = float(config.get("canary_probability", 0.0))
        elif "canary" in config:
            # honor explicit enable/disable in config; if enabled, allow override probability
            if bool(config.get("canary")):
                p = float(config.get("canary_probability", canary_cfg.get("canary_probability", 0.0)))
            else:
                p = 0.0
        elif bool(serving_cfg.get("canary", False)):
            p = float(canary_cfg.get("canary_probability", 0.0))
        else:
            p = 0.0
        self._canary_probability = max(0.0, min(1.0, p))
        self._logger.info("Router configured: canary_probability=%s", self._canary_probability)

    async def __call__(self, request: Request):
        if request.url.path.endswith("/healthz"):
            return {"status": "ok"}

        payload = await request.json()

        use_canary = random.random() < self._canary_probability
        # Prefer canary when selected, but gracefully fallback to stable if canary fails.
        if use_canary and self._canary is not None:
            try:
                return await self._canary.predict.remote(payload)
            except Exception as e:
                self._logger.warning("Canary prediction failed, falling back to stable: %s", str(e))

        # Stable is the safe fallback and should exist.
        try:
            return await self._stable.predict.remote(payload)
        except Exception:
            self._logger.exception("Stable model prediction failed")
            raise


# Serve application graph.
# Note: deployment names are pinned above to match serveConfigV2 updates.
deployment_graph = ModelRouter.bind(StableModel.bind(), CanaryModel.bind())
