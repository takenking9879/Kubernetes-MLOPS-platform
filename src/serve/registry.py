from dataclasses import dataclass
from typing import Any, Optional

from src.serve.config import Framework, WebhookConfig


@dataclass(frozen=True)
class ModelArtifact:
    model: Any
    version: str
    framework: Framework
    registry_name: str
    alias: str


class MLflowRegistry:
    def __init__(self, tracking_uri: str, logger):
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient()
        self._logger = logger

    def load_by_alias(self, registry_name: str, alias: str) -> ModelArtifact:
        mv = self._client.get_model_version_by_alias(registry_name, alias)
        version = str(mv.version)

        framework = self._resolve_framework(mv)
        if not framework:
            raise RuntimeError(
                f"Cannot resolve framework for model {registry_name}@{alias} v{version}. "
                f"Set model version tag 'framework' to one of: {[f.value for f in Framework]}"
            )

        model_uri = f"models:/{registry_name}@{alias}"
        model = self._load_model_object(model_uri, framework)

        self._logger.info(
            "Loaded model: %s@%s v%s (%s)",
            registry_name,
            alias,
            version,
            framework.value,
        )

        return ModelArtifact(
            model=model,
            version=version,
            framework=framework,
            registry_name=registry_name,
            alias=alias,
        )

    def resolve_alias_version(self, registry_name: str, alias: str) -> str:
        mv = self._client.get_model_version_by_alias(registry_name, alias)
        return str(mv.version)

    def _resolve_framework(self, model_version) -> Optional[Framework]:
        mv_tags = getattr(model_version, "tags", {}) or {}
        fw_str = str(mv_tags.get("framework", "")).strip().lower()
        if not fw_str:
            return None
        try:
            return Framework(fw_str)
        except ValueError:
            return None

    def _load_model_object(self, model_uri: str, framework: Framework):
        if framework == Framework.XGBOOST:
            import mlflow.xgboost

            return mlflow.xgboost.load_model(model_uri)
        if framework == Framework.PYTORCH:
            import mlflow.pytorch

            return mlflow.pytorch.load_model(model_uri)
        raise ValueError(f"Unsupported framework: {framework}")

    def ensure_webhook(self, webhook_config: WebhookConfig, registry_name: str, alias: str) -> None:
        preferred_events = ["MODEL_VERSION_ALIASED"]
        fallback_events = ["MODEL_VERSION_ALIASED"]
        description = f"Ray Serve sync for {registry_name}@{alias}"

        existing = self._find_webhook_by_name(webhook_config.name)

        def _upsert(events):
            if existing is None:
                created = self._client.create_webhook(
                    name=webhook_config.name,
                    url=webhook_config.url,
                    events=events,
                    description=description,
                    secret=webhook_config.secret,
                )
                self._logger.info(
                    "Created webhook: id=%s name=%s",
                    getattr(created, "webhook_id", "unknown"),
                    webhook_config.name,
                )
                return

            current_events = sorted(getattr(existing, "events", []) or [])
            if (
                getattr(existing, "url", None) != webhook_config.url
                or current_events != sorted(events)
                or getattr(existing, "status", "ACTIVE") != "ACTIVE"
            ):
                self._client.update_webhook(
                    webhook_id=existing.webhook_id,
                    url=webhook_config.url,
                    events=events,
                    description=description,
                    status="ACTIVE",
                )
                self._logger.info("Updated webhook: id=%s", existing.webhook_id)
            else:
                self._logger.info("Webhook already configured: id=%s", existing.webhook_id)

        try:
            _upsert(preferred_events)
        except Exception as e:
            self._logger.warning(
                "Preferred webhook events unsupported or failed (%s). Falling back to legacy events.",
                e,
            )
            _upsert(fallback_events)

    def _find_webhook_by_name(self, name: str):
        page_token = None
        while True:
            page = (
                self._client.list_webhooks(max_results=100, page_token=page_token)
                if page_token
                else self._client.list_webhooks(max_results=100)
            )
            for webhook in page:
                if getattr(webhook, "name", None) == name:
                    return webhook
            page_token = getattr(page, "next_page_token", None)
            if not page_token:
                break
        return None
