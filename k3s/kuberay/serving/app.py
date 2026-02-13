from typing import Dict

from ray import serve
from starlette.requests import Request

from src.serve.config import ConfigLoader
from src.serve.registry import MLflowRegistry
from src.serve.router import TrafficRouter
from src.serve.runtime import ModelRuntime
from src.serve.webhooks import MLflowAliasWebhookHandler
from src.utils.logger import create_logger


@serve.deployment(name="StableModel")
class StableModel:
    def __init__(self):
        config = ConfigLoader.load()
        logger = create_logger("StableModel")
        registry = MLflowRegistry(config.model.tracking_uri, logger)
        self._rt = ModelRuntime(
            name="StableModel",
            variant="stable",
            registry=registry,
            registry_name=config.model.registry_name,
            default_alias=config.model.default_alias,
            input_dim_fallback=config.model.input_dim_fallback,
            num_classes=config.model.num_classes,
            dsl_path=config.model.dsl_path,
        )
        self._rt.load()

    def reconfigure(self, config: Dict[str, str]) -> None:
        alias = config.get("alias") if config else None
        self._rt.load(alias_override=alias)

    async def predict(self, payload: Dict[str, object]):
        return self._rt.predict(payload)


@serve.deployment(name="CanaryModel")
class CanaryModel:
    def __init__(self):
        config = ConfigLoader.load()
        logger = create_logger("CanaryModel")
        registry = MLflowRegistry(config.model.tracking_uri, logger)
        self._rt = ModelRuntime(
            name="CanaryModel",
            variant="canary",
            registry=registry,
            registry_name=config.model.registry_name,
            default_alias=config.model.default_alias,
            input_dim_fallback=config.model.input_dim_fallback,
            num_classes=config.model.num_classes,
            dsl_path=config.model.dsl_path,
        )

        alias = config.canary.alias if config.canary else "challenger"
        self._rt.load(alias_override=alias)

    def reconfigure(self, config: Dict[str, str]) -> None:
        alias = config.get("alias") if config else None
        if alias is None:
            serving_config = ConfigLoader.load()
            alias = serving_config.canary.alias if serving_config.canary else "challenger"
        self._rt.load(alias_override=alias)

    async def predict(self, payload: Dict[str, object]):
        return self._rt.predict(payload)


@serve.deployment(name="ModelRouter")
class ModelRouter:
    def __init__(self, stable, canary):
        self._logger = create_logger("ModelRouter")
        self._stable = stable
        self._canary = canary

        config = ConfigLoader.load()
        canary_probability = (
            config.canary.probability if config.canary_enabled and config.canary else 0.0
        )

        self._traffic_router = TrafficRouter(
            stable_handle=self._stable,
            canary_handle=self._canary,
            canary_probability=canary_probability,
        )
        self._webhook_handler = MLflowAliasWebhookHandler(self._stable, self._logger)

        registry = MLflowRegistry(config.model.tracking_uri, self._logger)
        try:
            registry.ensure_webhook(config.webhook, config.model.registry_name, config.model.default_alias)
        except Exception as e:
            self._logger.warning(
                "Could not ensure MLflow webhook (this is expected if using HTTP and MLflow requires HTTPS): %s",
                e,
            )

    def reconfigure(self, config: Dict[str, object]) -> None:
        serving_config = ConfigLoader.load()
        p = serving_config.canary.probability if serving_config.canary_enabled and serving_config.canary else 0.0
        self._traffic_router.set_canary_probability(p)
        self._logger.info(
            "Router reconfigured: canary_probability=%.2f",
            self._traffic_router.canary_probability,
        )

    async def __call__(self, request: Request):
        if request.url.path == "/webhook":
            return await self._webhook_handler.handle(request)

        return await self._traffic_router.route(request)


deployment_graph = ModelRouter.bind(StableModel.bind(), CanaryModel.bind())
