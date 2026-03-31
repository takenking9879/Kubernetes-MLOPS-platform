"""
app/backend/routers/gpu_resources.py

GPU resource availability endpoints (Phase 2).

Endpoints:
  GET  /api/v2/gpu-resources/catalog          - unified offer list (spot-available first)
  POST /api/v2/gpu-resources/select           - ranked any_of list from constraints
  GET  /api/v2/gpu-resources/llm-catalog      - static LLM model → GPU recommendations
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

# Ensure project root is on sys.path so `src.services` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.gpu_catalog import GPUCatalogService, GPUOffer  # noqa: E402
from src.services.gpu_selector import (  # noqa: E402
    GPUSelectorService,
    ResourceConstraints,
)

router = APIRouter(prefix="/api/v2/gpu-resources", tags=["gpu-resources"])

# Module-level singletons (shared cache across requests)
_catalog_svc = GPUCatalogService()
_selector_svc = GPUSelectorService()

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


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/catalog")
async def get_gpu_catalog(
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
async def select_gpu_resources(body: ResourceConstraintsRequest) -> GPUSelectResponse:
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


@router.get("/llm-catalog")
async def get_llm_catalog() -> list[dict]:
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
