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

    @staticmethod
    def _build_overlay_s3_client() -> tuple[object, Dict[str, Any]]:
        import boto3
        from botocore.config import Config as _BotoConfig

        access_key = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
        secret_key = (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
        session_token = (os.getenv("AWS_SESSION_TOKEN") or "").strip()
        region = (
            (os.getenv("AWS_REGION") or "").strip()
            or (os.getenv("AWS_DEFAULT_REGION") or "").strip()
            or "us-east-1"
        )
        endpoint_url = (
            (os.getenv("AWS_S3_ENDPOINT_URL") or "").strip()
            or (os.getenv("AWS_ENDPOINT_URL") or "").strip()
        )

        kwargs: Dict[str, Any] = {
            "region_name": region,
        }
        if access_key:
            kwargs["aws_access_key_id"] = access_key
        if secret_key:
            kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token
        if endpoint_url:
            # Path-style addressing is safer for S3-compatible endpoints.
            kwargs["endpoint_url"] = endpoint_url
            kwargs["config"] = _BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            )

        return boto3.client("s3", **kwargs), {
            "region": region,
            "endpoint_url": endpoint_url,
            "has_access_key": bool(access_key),
            "has_secret_key": bool(secret_key),
            "has_session_token": bool(session_token),
        }

    @classmethod
    def _load_serving_overlay(cls) -> Dict[str, Any]:
        """Load optional params_serving.yaml providing serving.* and canary.* overrides.

        Checked sources in priority order:
          1. PARAMS_SERVING_PATH — local file path
          2. PARAMS_SERVING_S3_PATH — downloads from S3 to a temp file

        Returns the parsed dict (with root-level ``serving`` and ``canary`` keys),
        or {} if not configured.

        Used when PARAMS_PATH points to params_training.yaml (which has kuberay.model
        but no kuberay.serving / kuberay.canary).
        """
        local_path = os.getenv("PARAMS_SERVING_PATH")
        if not local_path:
            s3_uri = os.getenv("PARAMS_SERVING_S3_PATH")
            if s3_uri:
                import tempfile
                from urllib.parse import urlparse as _urlparse

                parsed = _urlparse(s3_uri)
                if parsed.scheme != "s3" or not parsed.netloc or not parsed.path or parsed.path == "/":
                    raise RuntimeError(
                        "PARAMS_SERVING_S3_PATH must be a valid s3://bucket/key URI "
                        f"(got: {s3_uri!r})"
                    )

                bucket = parsed.netloc
                key = parsed.path.lstrip("/")
                client, client_meta = cls._build_overlay_s3_client()

                with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
                    local_tmp = tmp.name

                endpoint = client_meta["endpoint_url"] or "<aws-default>"
                print(
                    "[config] Loading serving overlay from "
                    f"s3://{bucket}/{key} "
                    f"(region={client_meta['region']}, endpoint={endpoint}, "
                    f"has_access_key={client_meta['has_access_key']}, "
                    f"has_session_token={client_meta['has_session_token']})"
                )

                try:
                    client.download_file(bucket, key, local_tmp)
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to download PARAMS_SERVING_S3_PATH "
                        f"s3://{bucket}/{key} (region={client_meta['region']}, endpoint={endpoint}). "
                        "Check AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN, "
                        "AWS_REGION/AWS_DEFAULT_REGION, and AWS_S3_ENDPOINT_URL."
                    ) from exc

                local_path = local_tmp
        if not local_path or not os.path.exists(local_path):
            return {}
        with open(local_path) as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}

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

        # Optionally merge config.yaml (static infra config) as fallback values.
        # config.yaml lives next to params.yaml (same directory).
        static_cfg: Dict[str, Any] = {}
        config_path = os.path.join(os.path.dirname(params_path), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as fc:
                static_cfg = yaml.safe_load(fc) or {}

        # Build kuberay section: prefer params.yaml, fall back to config.yaml values.
        kuberay = data.get("kuberay")
        if not isinstance(kuberay, dict):
            # New-format params.yaml (generated by UI) — build kuberay from model + serving sections.
            # Root-level serving.* / canary.* (present in params_serving.yaml) take priority
            # over config.yaml defaults.
            model_section = data.get("model", {})
            root_serving = data.get("serving", {}) or {}
            root_canary  = data.get("canary", {}) or {}
            kuberay = {
                "model": {
                    "mlflow_tracking_uri": (
                        model_section.get("mlflow_tracking_uri")
                        or static_cfg.get("mlflow", {}).get("tracking_uri", "")
                    ),
                    "mlflow_registry_model_name": model_section.get("registry_model_name", ""),
                    "input_dim": model_section.get("input_dim", 14),
                    "num_classes": model_section.get("num_classes", 6),
                },
                "serving": root_serving or {
                    "alias": static_cfg.get("serving", {}).get("alias", "champion"),
                    "canary": False,
                    "webhook_public_base_url": static_cfg.get("serving", {}).get("webhook_base_url", ""),
                    "webhook_path": static_cfg.get("serving", {}).get("webhook_path", "/infer/webhook"),
                    "webhook_name": static_cfg.get("serving", {}).get("webhook_name", ""),
                    "webhook_max_timestamp_age_seconds": static_cfg.get("serving", {}).get(
                        "webhook_max_timestamp_age_seconds", 300
                    ),
                },
                "canary": root_canary or {
                    "alias": static_cfg.get("serving", {}).get("canary_alias", "challenger"),
                    "canary_probability": static_cfg.get("serving", {}).get("canary_probability", 0.10),
                },
            }

        spark = data.get("spark", {})
        if not isinstance(kuberay, dict):
            raise RuntimeError("Missing required section: kuberay")

        model_cfg = kuberay.get("model")
        serving_cfg = kuberay.get("serving")
        canary_cfg = kuberay.get("canary")

        # Serving overlay: fills kuberay.serving / kuberay.canary when PARAMS_PATH points to
        # params_training.yaml (which has kuberay.model but no kuberay.serving/kuberay.canary).
        # Priority: kuberay.* in primary file → PARAMS_SERVING_PATH/S3 overlay → config.yaml.
        serving_overlay = cls._load_serving_overlay()

        if not isinstance(serving_cfg, dict):
            overlay = serving_overlay.get("serving", {})
            serving_cfg = overlay if isinstance(overlay, dict) and overlay else {
                "alias": static_cfg.get("serving", {}).get("alias", "champion"),
                "canary": False,
                "webhook_public_base_url": static_cfg.get("serving", {}).get("webhook_base_url", ""),
                "webhook_path": static_cfg.get("serving", {}).get("webhook_path", "/infer/webhook"),
                "webhook_name": static_cfg.get("serving", {}).get("webhook_name", ""),
                "webhook_max_timestamp_age_seconds": static_cfg.get("serving", {}).get(
                    "webhook_max_timestamp_age_seconds", 300
                ),
            }

        if not isinstance(canary_cfg, dict):
            overlay = serving_overlay.get("canary", {})
            canary_cfg = overlay if isinstance(overlay, dict) and overlay else {
                "alias": static_cfg.get("serving", {}).get("canary_alias", "challenger"),
                "canary_probability": static_cfg.get("serving", {}).get("canary_probability", 0.10),
            }

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
