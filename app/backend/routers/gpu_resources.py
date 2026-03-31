"""
app/backend/routers/gpu_resources.py

GPU resource availability endpoints (Phase 2).

Endpoints:
  GET  /api/v2/gpu-resources/catalog          - unified offer list (spot-available first)
  POST /api/v2/gpu-resources/select           - ranked any_of list from constraints
  GET  /api/v2/gpu-resources/llm-catalog      - static LLM model → GPU recommendations
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

# Ensure project root is on sys.path so `src.services` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from src.services.gpu_catalog import GPUCatalogService, GPUOffer  # noqa: E402
from src.services.gpu_selector import (  # noqa: E402
    GPUSelectorService,
    ResourceConstraints,
)
from src.services.gpu_catalog import (  # noqa: E402
    _RUNPOD_TO_SKYPILOT,
    _infer_skypilot_id,
)
from src.services.provider_region_mapping import (  # noqa: E402
    RUNPOD_SKYPILOT_REGION_CODES,
    normalize_provider_region,
)
from services.resource_constraints import normalize_gpu_type  # noqa: E402

router = APIRouter(prefix="/api/v2/gpu-resources", tags=["gpu-resources"])

# Module-level singletons (shared cache across requests)
_catalog_svc = GPUCatalogService()
_selector_svc = GPUSelectorService()
_RUNPOD_REGIONS_TTL_SECONDS = 300
_runpod_regions_cache: list[dict[str, Any]] = []
_runpod_regions_cache_ts: float = 0.0
_RUNPOD_AVAILABILITY_TTL_SECONDS = 20
_runpod_availability_cache: dict[
    tuple[tuple[str, ...], tuple[str, ...]],
    tuple[float, list[dict[str, Any]]],
] = {}

# ── Static LLM catalog ────────────────────────────────────────────────────────
_LLM_CATALOG: list[dict] = [
    {
        "model_id": "meta-llama/Llama-3.1-8B",
        "vram_gb": 16,
        "min_gpus": 1,
        "recommended_gpu": "A100",
    },
    {
        "model_id": "Qwen/Qwen2.5-7B",
        "vram_gb": 14,
        "min_gpus": 1,
        "recommended_gpu": "A40",
    },
    {
        "model_id": "meta-llama/Llama-3.1-70B",
        "vram_gb": 140,
        "min_gpus": 4,
        "recommended_gpu": "A100-80GB",
    },
    {
        "model_id": "deepseek-ai/DeepSeek-R1-8B",
        "vram_gb": 16,
        "min_gpus": 1,
        "recommended_gpu": "A100",
    },
]


# ── Request / Response models ─────────────────────────────────────────────────


class ResourceConstraintsRequest(BaseModel):
    providers: list[str] = Field(default_factory=lambda: ["runpod"])
    gpu_types: list[str] | None = None
    min_vram_gb: float = 0
    max_price_per_hour: float = 9999
    prefer_spot: bool = True
    require_infiniband: bool = False
    preferred_regions: list[str] = Field(default_factory=list)
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    job_type: str = "tabular"


class GPUSelectResponse(BaseModel):
    any_of: list[dict]
    spot_entries: int
    ondemand_entries: int
    estimated_cost_spot: float | None
    estimated_cost_ondemand: float | None


class RunPodAvailabilityRequest(BaseModel):
    gpu_types: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)


def _fallback_runpod_regions() -> list[dict[str, Any]]:
    return [
        {
            "provider": "runpod",
            "provider_region_id": code,
            "name": code,
            "location": code,
            "skypilot_region": code,
            "skypilot_zone": "",
            "skypilot_infra": f"runpod/{code}",
        }
        for code in sorted(RUNPOD_SKYPILOT_REGION_CODES)
    ]


def _get_runpod_supported_regions() -> list[dict[str, Any]]:
    global _runpod_regions_cache_ts
    global _runpod_regions_cache

    if _runpod_regions_cache and (time.time() - _runpod_regions_cache_ts) < _RUNPOD_REGIONS_TTL_SECONDS:
        return _runpod_regions_cache

    api_key = os.getenv("RUNPOD_API_KEY", "")
    if not api_key:
        _runpod_regions_cache = _fallback_runpod_regions()
        _runpod_regions_cache_ts = time.time()
        return _runpod_regions_cache

    try:
        import runpod  # type: ignore[import]
        from runpod.api import graphql as _gql  # type: ignore[import]

        runpod.api_key = api_key
        result = _gql.run_graphql_query("{ dataCenters { id name location } }")
        datacenters = (result.get("data", {}) or {}).get("dataCenters", []) or []

        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for dc in datacenters:
            dc_id = str((dc or {}).get("id") or "").strip()
            if not dc_id:
                continue
            mapping = normalize_provider_region("runpod", dc_id)
            if not mapping.resolved:
                continue
            if mapping.skypilot_region not in RUNPOD_SKYPILOT_REGION_CODES:
                continue
            if dc_id in seen:
                continue
            seen.add(dc_id)
            items.append(
                {
                    "provider": "runpod",
                    "provider_region_id": dc_id,
                    "name": str((dc or {}).get("name") or dc_id),
                    "location": str((dc or {}).get("location") or ""),
                    "skypilot_region": mapping.skypilot_region,
                    "skypilot_zone": mapping.skypilot_zone,
                    "skypilot_infra": mapping.skypilot_infra,
                }
            )

        items.sort(key=lambda x: (x["skypilot_region"], x["provider_region_id"]))
        _runpod_regions_cache = items or _fallback_runpod_regions()
        _runpod_regions_cache_ts = time.time()
        return _runpod_regions_cache
    except Exception:
        _runpod_regions_cache = _fallback_runpod_regions()
        _runpod_regions_cache_ts = time.time()
        return _runpod_regions_cache


def _build_runpod_availability_matrix_query(
    region_ids: list[str],
) -> tuple[str, dict[str, str]]:
    """Build one GraphQL query that asks lowestPrice for all regions via aliases."""
    alias_to_region: dict[str, str] = {}
    region_blocks: list[str] = []
    for idx, region_id in enumerate(region_ids):
        safe_region_id = region_id.replace('"', "").strip()
        alias = f"r{idx}"
        alias_to_region[alias] = safe_region_id
        region_blocks.append(
            f"{alias}: lowestPrice(input: {{ "
            "secureCloud: true, "
            "gpuCount: 1, "
            "minVcpuCount: 2, "
            "minMemoryInGb: 8, "
            "minDisk: 0, "
            f"dataCenterId: \"{safe_region_id}\", "
            "globalNetwork: false, "
            "compliance: null "
            "}) { stockStatus availableGpuCounts maxUnreservedGpuCount }"
        )
    query = "query { gpuTypes { id displayName " + " ".join(region_blocks) + " } }"
    return query, alias_to_region


def _aggregate_runpod_availability_rows(
    gpu_rows: list[dict[str, Any]],
    selected_gpu_types: set[str],
    alias_to_region: dict[str, str],
) -> list[dict[str, Any]]:
    """Aggregate (gpu_type, region) availability rows from GraphQL matrix response."""
    region_mapping: dict[str, Any] = {}
    for region_id in alias_to_region.values():
        mapping = normalize_provider_region("runpod", region_id)
        if mapping.resolved:
            region_mapping[region_id] = mapping

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for gpu in gpu_rows:
        gpu_id = str((gpu or {}).get("id") or "")
        display_name = str((gpu or {}).get("displayName") or gpu_id)
        skypilot_gpu = _RUNPOD_TO_SKYPILOT.get(gpu_id) or _infer_skypilot_id(display_name)
        gpu_type = normalize_gpu_type(skypilot_gpu or display_name)
        if gpu_type not in selected_gpu_types:
            continue

        for alias, region_id in alias_to_region.items():
            mapping = region_mapping.get(region_id)
            if mapping is None:
                continue

            lowest = (gpu or {}).get(alias)
            raw_counts = list((lowest or {}).get("availableGpuCounts") or []) if lowest else []
            counts: list[int] = []
            for value in raw_counts:
                try:
                    parsed = int(value)
                except Exception:
                    continue
                if parsed >= 0:
                    counts.append(parsed)

            # Deduplicate/normalize count vectors from aliased responses.
            counts = sorted(set(counts))
            available = len(counts) > 0
            max_available = int(max(counts)) if counts else 0

            key = (gpu_type, region_id)
            existing = rows_by_key.get(key)
            if existing is None:
                rows_by_key[key] = {
                    "gpu_type": gpu_type,
                    "provider": "runpod",
                    "provider_region_id": region_id,
                    "skypilot_region": mapping.skypilot_region,
                    "skypilot_zone": mapping.skypilot_zone,
                    "skypilot_infra": mapping.skypilot_infra,
                    "available": available,
                    "available_counts": counts,
                    "max_available": max_available,
                }
                continue

            merged_counts = sorted(set(existing.get("available_counts", []) + counts))
            existing["available_counts"] = merged_counts
            existing["available"] = bool(existing.get("available", False) or available)
            existing["max_available"] = max(int(existing.get("max_available", 0)), max_available)

    rows = list(rows_by_key.values())
    rows.sort(key=lambda x: (x["gpu_type"], x["provider_region_id"]))
    return rows


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/catalog")
def get_gpu_catalog(
    providers: str | None = Query(
        default=None,
        description="Comma-separated provider list: runpod,vast,aws,gcp,azure",
    ),
    min_vram: float = Query(default=0, description="Minimum VRAM in GB"),
) -> list[dict]:
    """
    Return GPU offers across providers.
    Spot-available entries appear first; within each group sorted by price.
    """
    provider_list = (
        [p.strip() for p in providers.split(",") if p.strip()]
        if providers
        else None
    )
    offers = _catalog_svc.query_availability(
        providers=provider_list,
        min_vram_gb=min_vram,
    )
    # Spot-available first, then by effective price
    offers.sort(
        key=lambda o: (not o.spot_available, o.price_spot or o.price_on_demand)
    )
    return [_offer_to_dict(o) for o in offers]


@router.post("/select", response_model=GPUSelectResponse)
def select_gpu_resources(body: ResourceConstraintsRequest) -> GPUSelectResponse:
    """
    Given resource constraints, return a ranked `any_of` SkyPilot list.
    Spot entries appear before on-demand entries of equivalent GPU type.
    """
    constraints = ResourceConstraints(
        providers=body.providers,
        gpu_types=body.gpu_types,
        min_vram_gb=body.min_vram_gb,
        max_price_per_hour=body.max_price_per_hour,
        prefer_spot=body.prefer_spot,
        require_infiniband=body.require_infiniband,
        preferred_regions=body.preferred_regions,
        num_nodes=body.num_nodes,
        num_gpus_per_node=body.num_gpus_per_node,
        job_type=body.job_type,
    )
    result = _selector_svc.select_providers(constraints)
    return GPUSelectResponse(
        any_of=result.any_of,
        spot_entries=result.spot_entries,
        ondemand_entries=result.ondemand_entries,
        estimated_cost_spot=result.estimated_cost_spot,
        estimated_cost_ondemand=result.estimated_cost_ondemand,
    )


@router.get("/runpod/regions")
def get_runpod_regions() -> list[dict[str, Any]]:
    """Return RunPod datacenter IDs that can be mapped to SkyPilot regions."""
    return _get_runpod_supported_regions()


@router.post("/runpod/availability")
def get_runpod_region_availability(body: RunPodAvailabilityRequest) -> list[dict[str, Any]]:
    """Check RunPod availability for selected GPU types on selected supported regions only."""
    gpu_types = [normalize_gpu_type(g) for g in (body.gpu_types or []) if str(g).strip()]
    region_ids = [str(r).strip().upper() for r in (body.regions or []) if str(r).strip()]

    if not gpu_types or not region_ids:
        return []

    supported = {r["provider_region_id"].upper() for r in _get_runpod_supported_regions()}
    region_ids = sorted({r for r in region_ids if r in supported})
    if not region_ids:
        return []

    cache_key = (tuple(sorted(set(gpu_types))), tuple(region_ids))
    cached = _runpod_availability_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _RUNPOD_AVAILABILITY_TTL_SECONDS:
        return cached[1]

    api_key = os.getenv("RUNPOD_API_KEY", "")
    if not api_key:
        return []

    try:
        import runpod  # type: ignore[import]
        from runpod.api import graphql as _gql  # type: ignore[import]

        runpod.api_key = api_key
        selected = {normalize_gpu_type(g) for g in gpu_types}
        query, alias_to_region = _build_runpod_availability_matrix_query(region_ids)
        result = _gql.run_graphql_query(query)
        gpu_rows = (result.get("data", {}) or {}).get("gpuTypes", []) or []
        rows = _aggregate_runpod_availability_rows(gpu_rows, selected, alias_to_region)

        _runpod_availability_cache[cache_key] = (time.time(), rows)
        return rows
    except Exception:
        return []


@router.get("/llm-catalog")
def get_llm_catalog() -> list[dict]:
    """Static catalog of LLM models with minimum GPU/VRAM requirements."""
    return _LLM_CATALOG


# ── Serialisation helper ──────────────────────────────────────────────────────


def _offer_to_dict(o: GPUOffer) -> dict:
    return {
        "provider": o.provider,
        "gpu_type": o.gpu_type,
        "gpu_count": o.gpu_count,
        "vram_gb": o.vram_gb,
        "vcpus": o.vcpus,
        "ram_gb": o.ram_gb,
        "price_on_demand": o.price_on_demand,
        "price_spot": o.price_spot,
        "spot_available": o.spot_available,
        "available_count": o.available_count,
        "region": o.region,
        "infiniband": o.infiniband,
        "skypilot_supported": o.skypilot_supported,
        "skypilot_accelerator": o.skypilot_accelerator,
        "skypilot_cloud": o.skypilot_cloud,
        "provider_region_id": o.provider_region_id,
        "skypilot_region": o.skypilot_region,
        "skypilot_zone": o.skypilot_zone,
        "skypilot_infra": o.skypilot_infra,
    }
