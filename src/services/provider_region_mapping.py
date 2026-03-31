"""Provider region mapping helpers for deterministic SkyPilot placement.

This module normalizes provider-native location identifiers into
SkyPilot-compatible infra paths.

Key behaviors:
- RunPod:
  - Canonical zones like "CA-MTL-1" -> runpod/CA/CA-MTL-1
  - Canonical region codes like "CA" -> runpod/CA
  - Alias-prefixed IDs like "EUR-IS-1" -> runpod/IS (region pinned, zone omitted)
- AWS:
  - Regions like "us-east-1" -> aws/us-east-1
  - AZs like "us-east-1a" -> aws/us-east-1
- Vast:
  - SkyPilot does not support region/zone in infra path -> vast
"""
from __future__ import annotations

from dataclasses import dataclass
import re


RUNPOD_SKYPILOT_REGION_CODES: frozenset[str] = frozenset(
    {"CA", "US", "CZ", "IS", "NL", "SE", "RO", "NO"}
)

# Prefixes observed in provider APIs that may prepend a canonical country code.
# Example: EUR-IS-1 -> canonical SkyPilot region IS.
_RUNPOD_ALIAS_PREFIXES: frozenset[str] = frozenset(
    {"EU", "EUR", "NA", "AP", "APAC", "AS", "AUS", "GLOBAL"}
)

_AWS_REGION_RE = re.compile(r"^[a-z]{2}-[a-z0-9-]+-\d+$")
_AWS_ZONE_RE = re.compile(r"^[a-z]{2}-[a-z0-9-]+-\d+[a-z]$")


@dataclass(frozen=True)
class ProviderRegionMapping:
    provider: str
    provider_region_id: str
    skypilot_region: str
    skypilot_zone: str
    skypilot_infra: str
    resolved: bool


def normalize_provider(provider: str) -> str:
    return str(provider or "").strip().lower()


def _build_infra(provider: str, region: str, zone: str) -> str:
    p = normalize_provider(provider)

    if p == "runpod":
        if region and zone:
            return f"runpod/{region}/{zone}"
        if region:
            return f"runpod/{region}"
        return "runpod"

    if p == "aws":
        return f"aws/{region}" if region else "aws"

    if p == "vast":
        return "vast"

    if p in {"gcp", "azure"}:
        return f"{p}/{region}" if region else p

    return f"{p}/{region}" if region else p


def _map_runpod(provider_region_id: str) -> tuple[str, str, bool]:
    raw = str(provider_region_id or "").strip().upper()
    if not raw:
        return "", "", True

    if raw in RUNPOD_SKYPILOT_REGION_CODES:
        return raw, "", True

    parts = [p for p in raw.split("-") if p]
    if not parts:
        return "", "", False

    # Canonical zone format: <CC>-... where CC is a known RunPod region code.
    first = parts[0]
    if first in RUNPOD_SKYPILOT_REGION_CODES:
        zone = raw if len(parts) >= 2 else ""
        return first, zone, True

    # Alias-prefixed format: <ALIAS>-<CC>-... (e.g., EUR-IS-1).
    # We only keep the canonical region for deterministic placement.
    if len(parts) >= 2 and first in _RUNPOD_ALIAS_PREFIXES and parts[1] in RUNPOD_SKYPILOT_REGION_CODES:
        return parts[1], "", True

    return "", "", False


def _map_aws(provider_region_id: str) -> tuple[str, str, bool]:
    raw = str(provider_region_id or "").strip().lower()
    if not raw:
        return "", "", True

    if _AWS_ZONE_RE.match(raw):
        # us-east-1a -> region us-east-1
        return raw[:-1], raw, True

    if _AWS_REGION_RE.match(raw):
        return raw, "", True

    return "", "", False


def normalize_provider_region(provider: str, provider_region_id: str) -> ProviderRegionMapping:
    p = normalize_provider(provider)
    raw = str(provider_region_id or "").strip()

    if p == "runpod":
        region, zone, resolved = _map_runpod(raw)
        infra = _build_infra("runpod", region, zone)
        return ProviderRegionMapping(
            provider="runpod",
            provider_region_id=raw,
            skypilot_region=region,
            skypilot_zone=zone,
            skypilot_infra=infra,
            resolved=resolved,
        )

    if p == "aws":
        region, zone, resolved = _map_aws(raw)
        infra = _build_infra("aws", region, zone)
        return ProviderRegionMapping(
            provider="aws",
            provider_region_id=raw,
            skypilot_region=region,
            skypilot_zone=zone,
            skypilot_infra=infra,
            resolved=resolved,
        )

    if p == "vast":
        # SkyPilot infra for Vast is cloud-only. Keep provider region for UI only.
        return ProviderRegionMapping(
            provider="vast",
            provider_region_id=raw,
            skypilot_region="",
            skypilot_zone="",
            skypilot_infra="vast",
            resolved=True,
        )

    # Generic providers: pass-through region token only.
    generic_region = raw.lower() if raw else ""
    return ProviderRegionMapping(
        provider=p,
        provider_region_id=raw,
        skypilot_region=generic_region,
        skypilot_zone="",
        skypilot_infra=_build_infra(p, generic_region, ""),
        resolved=bool(raw) or p in {"gcp", "azure"},
    )


def build_infra_from_provider_region(provider: str, provider_region_id: str) -> str | None:
    mapping = normalize_provider_region(provider, provider_region_id)

    # If user/provider supplied a location but we cannot normalize it, fail fast.
    if str(provider_region_id or "").strip() and not mapping.resolved:
        return None

    return mapping.skypilot_infra


def normalized_region_code(provider: str, provider_region_id: str) -> str:
    """Return canonical region code used by SkyPilot routing/filtering."""
    return normalize_provider_region(provider, provider_region_id).skypilot_region
