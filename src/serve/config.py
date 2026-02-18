import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import yaml


class Framework(str, Enum):
    XGBOOST = "xgboost"
    PYTORCH = "pytorch"


@dataclass(frozen=True)
class ModelConfig:
    tracking_uri: str
    registry_name: str
    default_alias: str
    dsl_path: str
    input_dim_fallback: int
    num_classes: int


@dataclass(frozen=True)
class WebhookConfig:
    public_base_url: str
    path: str
    name: str
    max_timestamp_age_seconds: int

    @property
    def url(self) -> str:
        return f"{self.public_base_url.rstrip('/')}/{self.path.lstrip('/')}"

    @property
    def secret(self) -> str:
        secret = os.getenv("MLFLOW_WEBHOOK_SECRET")
        if not secret:
            raise RuntimeError(
                "Webhook secret MUST be set via environment variable MLFLOW_WEBHOOK_SECRET. "
                "It cannot be read from params.yaml."
            )
        return secret


@dataclass(frozen=True)
class CanaryConfig:
    alias: str
    probability: float


@dataclass(frozen=True)
class ServingConfig:
    model: ModelConfig
    webhook: WebhookConfig
    canary: Optional[CanaryConfig]
    canary_enabled: bool


class ConfigLoader:
    _instance: Optional[ServingConfig] = None

    @classmethod
    def load(cls) -> ServingConfig:
        if cls._instance is not None:
            return cls._instance

        params_path = os.getenv("PARAMS_PATH", "/home/ray/app/repo/k3s/params.yaml")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"params.yaml not found at {params_path}")

        with open(params_path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("params.yaml root must be a mapping")

        kuberay = data.get("kuberay")
        spark = data.get("spark", {})
        if not isinstance(kuberay, dict):
            raise RuntimeError("Missing required section: kuberay")

        model_cfg = kuberay.get("model")
        serving_cfg = kuberay.get("serving")
        canary_cfg = kuberay.get("canary")

        if not isinstance(model_cfg, dict):
            raise RuntimeError("Missing required section: kuberay.model")
        if not isinstance(serving_cfg, dict):
            raise RuntimeError("Missing required section: kuberay.serving")

        canary_enabled = bool(serving_cfg.get("canary", False))
        dsl_path = spark.get("preprocessing", {}).get(
            "dsl_path", "/app/repo/k3s/spark/preprocess/dsl_001.yaml"
        )

        model = ModelConfig(
            tracking_uri=cls._require_str(model_cfg, "mlflow_tracking_uri", "kuberay.model"),
            registry_name=cls._require_str(model_cfg, "mlflow_registry_model_name", "kuberay.model"),
            default_alias=cls._get_str(serving_cfg, "alias", "champion"),
            dsl_path=dsl_path,
            input_dim_fallback=cls._get_int(model_cfg, "input_dim", 14),
            num_classes=cls._get_int(model_cfg, "num_classes", 6),
        )

        webhook = WebhookConfig(
            public_base_url=cls._require_str(serving_cfg, "webhook_public_base_url", "kuberay.serving"),
            path=cls._get_str(serving_cfg, "webhook_path", "/infer/webhook"),
            name=cls._get_str(serving_cfg, "webhook_name", f"rayserve-{model.registry_name}-webhook"),
            max_timestamp_age_seconds=cls._get_int(serving_cfg, "webhook_max_timestamp_age_seconds", 300),
        )

        canary = None
        if isinstance(canary_cfg, dict):
            canary = CanaryConfig(
                alias=cls._get_str(canary_cfg, "alias", "challenger"),
                probability=max(0.0, min(1.0, float(canary_cfg.get("canary_probability", 0.0)))),
            )

        cls._instance = ServingConfig(
            model=model,
            webhook=webhook,
            canary=canary,
            canary_enabled=canary_enabled,
        )
        return cls._instance

    @staticmethod
    def _require_str(cfg: Dict[str, Any], key: str, path: str) -> str:
        value = cfg.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Missing required configuration: {path}.{key}")
        return value.strip()

    @staticmethod
    def _get_str(cfg: Dict[str, Any], key: str, default: str) -> str:
        value = cfg.get(key, default)
        return str(value).strip() if value else default

    @staticmethod
    def _get_int(cfg: Dict[str, Any], key: str, default: int) -> int:
        value = cfg.get(key, default)
        return int(value) if value is not None else default
