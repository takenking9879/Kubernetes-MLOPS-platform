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
    secret: str,
    max_age_seconds: int,
) -> bool:
    """Verify MLflow webhook HMAC-SHA256 signature and optional timestamp freshness.

    MLflow signs the raw request body with HMAC-SHA256 and sends:
        X-Mlflow-Signature: sha256=<hex_digest>
        X-Mlflow-Timestamp: <unix_ms>   (optional — present in newer MLflow versions)
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

    if not signature_header.startswith("sha256="):
        return False

    received_hex = signature_header[len("sha256="):]
    expected_hex = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()

    return hmac.compare_digest(received_hex, expected_hex)


class MLflowAliasWebhookHandler:
    """Handle MLflow MODEL_VERSION_ALIASED webhook events and trigger stable reloads."""

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
            timestamp_header=request.headers.get("x-mlflow-timestamp"),
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

        # Real MLflow payload: {"event_type": "MODEL_VERSION_ALIASED", "data": {...}}
        event_type = webhook_body.get("event_type", "")
        data = webhook_body.get("data", {}) or {}

        if event_type != "MODEL_VERSION_ALIASED":
            return JSONResponse(
                {"status": "ignored", "reason": "unsupported_event_type", "event_type": event_type},
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
                "event_type": event_type,
                "model_name": model_name,
                "alias": alias,
                "version": version,
            },
            status_code=200,
        )
