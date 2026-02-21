import os
from typing import Dict

from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.serve.config import ConfigLoader
from src.serve.registry import MLflowRegistry
from src.serve.router import TrafficRouter
from src.serve.runtime import ModelRuntime
from src.serve.webhooks import MLflowAliasWebhookHandler
from src.utils.logger import create_logger


def _load_executor(loader, registry_name: str, alias: str, logger):
    """Resolve and load NumpyPipelineExecutor via PipelineArtifactLoader."""
    try:
        return loader.load_executor(registry_name, alias)
    except Exception as e:
        logger.error(
            "Failed to load NumpyPipelineExecutor for %s@%s: %s",
            registry_name, alias, e, exc_info=True,
        )
        raise


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

        self._registry_name = config.model.registry_name
        self._default_alias = config.model.default_alias

        # Always load NumpyPipelineExecutor: Ray Serve runs DSL preprocessing + inference.
        from src.serve.pipeline_loader import PipelineArtifactLoader
        params_path  = os.getenv("PARAMS_PATH", "/home/ray/app/repo/k3s/params.yaml")
        self._loader = PipelineArtifactLoader(params_path, logger)
        executor     = _load_executor(self._loader, self._registry_name, self._default_alias, logger)
        self._rt.set_executor(executor)

    def reconfigure(self, config: Dict[str, str]) -> None:
        alias = config.get("alias") if config else None
        self._rt.load(alias_override=alias)

        # Reload executor: the new model version may point to different pipeline artifacts.
        executor = _load_executor(
            self._loader,
            self._registry_name,
            alias or self._default_alias,
            self._rt._logger,
        )
        self._rt.set_executor(executor)

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

        canary_alias = config.canary.alias if config.canary else "challenger"
        self._rt.load(alias_override=canary_alias)

        self._registry_name = config.model.registry_name
        self._canary_alias  = canary_alias

        # Always load NumpyPipelineExecutor: Ray Serve runs DSL preprocessing + inference.
        from src.serve.pipeline_loader import PipelineArtifactLoader
        params_path  = os.getenv("PARAMS_PATH", "/home/ray/app/repo/k3s/params.yaml")
        self._loader = PipelineArtifactLoader(params_path, logger)
        executor     = _load_executor(self._loader, self._registry_name, self._canary_alias, logger)
        self._rt.set_executor(executor)

    def reconfigure(self, config: Dict[str, str]) -> None:
        alias = config.get("alias") if config else None
        if alias is None:
            serving_config = ConfigLoader.load()
            alias = serving_config.canary.alias if serving_config.canary else "challenger"
        self._rt.load(alias_override=alias)

        executor = _load_executor(
            self._loader,
            self._registry_name,
            alias,
            self._rt._logger,
        )
        self._rt.set_executor(executor)

    async def predict(self, payload: Dict[str, object]):
        return self._rt.predict(payload)


@serve.deployment(name="ModelRouter")
class ModelRouter:
    def __init__(self, stable, canary):
        self._logger = create_logger("ModelRouter")
        self._stable = stable
        self._canary = canary

        config = ConfigLoader.load()
        self._registry_name  = config.model.registry_name
        self._default_alias  = config.model.default_alias
        self._webhook_max_age = config.webhook.max_timestamp_age_seconds
        canary_probability   = (
            config.canary.probability if config.canary_enabled and config.canary else 0.0
        )

        self._traffic_router = TrafficRouter(
            stable_handle=self._stable,
            canary_handle=self._canary,
            canary_probability=canary_probability,
        )
        self._webhook_handler = MLflowAliasWebhookHandler(self._stable, self._logger)

        # Register the MLflow webhook so model + executor reload automatically on alias change.
        registry = MLflowRegistry(config.model.tracking_uri, self._logger)
        try:
            registry.ensure_webhook(
                config.webhook,
                config.model.registry_name,
                config.model.default_alias,
            )
        except Exception as e:
            self._logger.warning(
                "Could not ensure MLflow webhook (expected if MLflow requires HTTPS): %s", e
            )

    def reconfigure(self, config: Dict[str, object]) -> None:
        serving_config = ConfigLoader.load()
        p = (
            serving_config.canary.probability
            if serving_config.canary_enabled and serving_config.canary
            else 0.0
        )
        self._traffic_router.set_canary_probability(p)
        self._logger.info(
            "Router reconfigured: canary_probability=%.2f",
            self._traffic_router.canary_probability,
        )

    async def _handle_canary_control(self, request: Request) -> JSONResponse:
        """
        Canary progression endpoint — called by Airflow to ramp up traffic progressively.

        POST /infer/canary  {"canary_probability": 0.25}
            → Updates the traffic split immediately (0.0 = all stable, 1.0 = all canary).
              Airflow calls this repeatedly to step from initial% → 100%.

        POST /infer/canary  {"action": "reload", "alias": "challenger"}
            → Reloads the CanaryModel from the given alias (default: "challenger").
              Call this when a new challenger version is registered in MLflow before
              starting a new progression cycle.

        GET /infer/canary
            → Returns current canary_probability (useful for Airflow sensors).
        """
        if request.method == "GET":
            return JSONResponse({
                "status": "ok",
                "canary_probability": self._traffic_router.canary_probability,
            })

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"status": "error", "reason": "invalid_json"},
                status_code=400,
            )

        if "canary_probability" in body:
            try:
                p = float(body["canary_probability"])
            except (TypeError, ValueError):
                return JSONResponse(
                    {"status": "error", "reason": "canary_probability must be a float"},
                    status_code=400,
                )
            self._traffic_router.set_canary_probability(p)
            self._logger.info(
                "Canary probability updated: %.2f (Airflow)",
                self._traffic_router.canary_probability,
            )
            return JSONResponse({
                "status": "ok",
                "canary_probability": self._traffic_router.canary_probability,
            })

        if body.get("action") == "reload":
            alias = body.get("alias", "challenger")
            await self._canary.reconfigure.remote({"alias": alias})
            self._logger.info("Canary reload triggered: alias=%s (Airflow)", alias)
            return JSONResponse({"status": "ok", "action": "reload", "alias": alias})

        return JSONResponse(
            {"status": "error", "reason": "Provide 'canary_probability' (float) or 'action: reload'"},
            status_code=400,
        )

    async def __call__(self, request: Request):
        path = request.url.path.rstrip("/")

        if path.endswith("/canary"):
            return await self._handle_canary_control(request)

        if request.method == "GET":
            return {"status": "ok", "route": "/infer", "message": "Use POST with JSON payload."}

        if path.endswith("/webhook"):
            secret = ""
            try:
                secret = ConfigLoader.load().webhook.secret
            except Exception as e:
                self._logger.warning("Webhook secret unavailable: %s", e)
            return await self._webhook_handler.handle(
                request,
                expected_registry_name=self._registry_name,
                expected_alias=self._default_alias,
                secret=secret,
                max_age_seconds=self._webhook_max_age,
            )

        try:
            payload = await request.json()
        except Exception:
            try:
                print("[ModelRouter] Payload is not JSON, attempting to read raw body or form data...")  # --- IGNORE ---
                body = await request.body()
                payload = {"body": body.decode("utf-8", errors="ignore")} if body else {}
            except Exception:
                payload = {}

        return await self._traffic_router.route(payload)


deployment_graph = ModelRouter.bind(StableModel.bind(), CanaryModel.bind())
