import base64
import hashlib
import hmac
import json
import time
from typing import Optional

from starlette.responses import JSONResponse


def verify_webhook_signature(
    payload_bytes: bytes,
    signature_header: Optional[str],
    timestamp_header: Optional[str],
    delivery_id_header: Optional[str],
    secret: str,
    max_age_seconds: int,
) -> bool:
    """Verify MLflow webhook HMAC-SHA256 signature and optional timestamp freshness.

    MLflow signs the raw request body with HMAC-SHA256 and sends:
        X-Mlflow-Signature: sha256=<hex_digest> (old)
        or
        X-Mlflow-Signature: v1,<base64_encoded_signature> (new)
        X-Mlflow-Timestamp: <unix_ms> or <unix_s>
        X-Mlflow-Delivery-Id: <uuid>
    """
    if not signature_header:
        return False

    # Optional timestamp freshness check (skipped if header absent).
    if timestamp_header:
        try:
            webhook_ts = int(timestamp_header)
            # MLflow sends timestamps in milliseconds; normalize to seconds.
            if webhook_ts > 1_000_000_000_000:
                webhook_ts = webhook_ts // 1000
            age = int(time.time()) - webhook_ts
            if age < 0 or age > max_age_seconds:
                return False
        except (ValueError, TypeError):
            return False

    if signature_header.startswith("v1,"):
        if not timestamp_header or not delivery_id_header:
            return False
        received_b64 = signature_header[len("v1,"):]
        payload_str = payload_bytes.decode("utf-8")
        signed_content = f"{delivery_id_header}.{timestamp_header}.{payload_str}"
        expected_signature = hmac.new(
            secret.encode("utf-8"), signed_content.encode("utf-8"), hashlib.sha256
        ).digest()
        expected_b64 = base64.b64encode(expected_signature).decode("utf-8")
        return hmac.compare_digest(received_b64, expected_b64)

    if signature_header.startswith("sha256="):
        received_hex = signature_header[len("sha256="):]
        expected_hex = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
        return hmac.compare_digest(received_hex, expected_hex)

    return False


class MLflowAliasWebhookHandler:
    """Handle MLflow MODEL_VERSION_ALIASED webhook events and trigger stable reloads."""

    def __init__(self, on_alias_set, logger):
        self._on_alias_set = on_alias_set
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
            timestamp_header=request.headers.get("x-mlflow-timestamp"),
            delivery_id_header=request.headers.get("x-mlflow-delivery-id"),
            secret=secret,
            max_age_seconds=max_age_seconds,
        )

        if not verified:
            self._logger.warning(
                "Rejected webhook: invalid signature or timestamp "
                "(sig=%s, ts=%s)",
                request.headers.get("x-mlflow-signature"),
                request.headers.get("x-mlflow-timestamp"),
            )
            return JSONResponse(
                {"status": "rejected", "reason": "invalid_signature"},
                status_code=401,
            )

        webhook_body = json.loads(payload_bytes.decode("utf-8"))

        # Real MLflow payload: {"entity": "model_version_alias", "action": "created", "data": {...}}
        entity = webhook_body.get("entity", "")
        action = webhook_body.get("action", "")
        data = webhook_body.get("data", {}) or {}

        if entity != "model_version_alias" or action not in ("created", "deleted"):
            return JSONResponse(
                {"status": "ignored", "reason": "unsupported_event_type", "entity": entity, "action": action},
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

        await self._on_alias_set(alias)

        self._logger.info(
            "Stable model reloaded: %s@%s v%s",
            model_name,
            alias,
            version,
        )

        return JSONResponse(
            {
                "status": "ok",
                "entity": entity,
                "action": action,
                "model_name": model_name,
                "alias": alias,
                "version": version,
            },
            status_code=200,
        )
