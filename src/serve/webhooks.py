import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

from starlette.responses import JSONResponse


def verify_webhook_signature(
    payload_bytes: bytes,
    signature_header: Optional[str],
    delivery_id_header: Optional[str],
    timestamp_header: Optional[str],
    secret: str,
    max_age_seconds: int,
) -> bool:
    """Verify MLflow webhook HMAC signature (v1) and timestamp freshness."""
    if not signature_header or not delivery_id_header or not timestamp_header:
        return False

    try:
        webhook_ts = int(timestamp_header)
        age = int(time.time()) - webhook_ts
        if age < 0 or age > max_age_seconds:
            return False
    except (ValueError, TypeError):
        return False

    if not signature_header.startswith("v1,"):
        return False

    received_sig_b64 = signature_header.split(",", 1)[1]
    signed_content = f"{delivery_id_header}.{timestamp_header}.{payload_bytes.decode('utf-8')}".encode("utf-8")
    expected_sig = hmac.new(secret.encode("utf-8"), signed_content, hashlib.sha256).digest()
    expected_sig_b64 = base64.b64encode(expected_sig).decode("utf-8")

    return hmac.compare_digest(received_sig_b64, expected_sig_b64)


class MLflowAliasWebhookHandler:
    """Handle MLflow alias webhook events and trigger stable reloads."""

    def __init__(self, stable_handle, logger):
        self._stable = stable_handle
        self._logger = logger

    async def handle(
        self,
        request,
        *,
        expected_registry_name: str,
        expected_alias: str,
        secret: str,
        max_age_seconds: int,
    ) -> JSONResponse:
        payload_bytes = await request.body()

        verified = verify_webhook_signature(
            payload_bytes=payload_bytes,
            signature_header=request.headers.get("x-mlflow-signature"),
            delivery_id_header=request.headers.get("x-mlflow-delivery-id"),
            timestamp_header=request.headers.get("x-mlflow-timestamp"),
            secret=secret,
            max_age_seconds=max_age_seconds,
        )

        if not verified:
            self._logger.warning("Rejected webhook: invalid signature or timestamp")
            return JSONResponse(
                {"status": "rejected", "reason": "invalid_signature"},
                status_code=401,
            )

        webhook_body = json.loads(payload_bytes.decode("utf-8"))
        entity = webhook_body.get("entity")
        action = webhook_body.get("action")
        data = webhook_body.get("data", {}) or {}

        if entity != "model_version_alias":
            return JSONResponse(
                {"status": "ignored", "reason": "unsupported_entity"},
                status_code=202,
            )

        if action not in {"created", "updated", "set"}:
            return JSONResponse(
                {"status": "ignored", "reason": "unsupported_action"},
                status_code=202,
            )

        model_name = str(data.get("name", ""))
        alias = str(data.get("alias", ""))
        version = str(data.get("version", ""))

        if model_name != expected_registry_name:
            return JSONResponse(
                {"status": "ignored", "reason": "different_model"},
                status_code=202,
            )

        if alias != expected_alias:
            return JSONResponse(
                {"status": "ignored", "reason": "different_alias"},
                status_code=202,
            )

        await self._stable.reconfigure.remote({"alias": alias})

        self._logger.info(
            "Stable model reloaded: %s@%s v%s",
            model_name,
            alias,
            version,
        )

        return JSONResponse(
            {
                "status": "ok",
                "event": f"{entity}.{action}",
                "model_name": model_name,
                "alias": alias,
                "version": version,
            },
            status_code=200,
        )
