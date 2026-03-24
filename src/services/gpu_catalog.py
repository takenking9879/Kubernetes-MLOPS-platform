"""
src/services/gpu_catalog.py

Unified GPU availability catalog: RunPod (real-time) + Vast.ai (real-time) +
SkyPilot offline catalog (AWS/GCP/Azure).

Each provider query is independent — failure returns [] without raising.
Results are cached for TTL_SECONDS (60 s) to avoid hammering provider APIs.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── RunPod GPU ID → SkyPilot accelerator ID ──────────────────────────────────
_RUNPOD_TO_SKYPILOT: dict[str, str] = {
    "NVIDIA GeForce RTX 4090": "RTX4090",
    "NVIDIA GeForce RTX 3090": "RTX3090",
    "NVIDIA GeForce RTX 3090 Ti": "RTX3090",
    "NVIDIA GeForce RTX 3080": "RTX3080",
    "NVIDIA GeForce RTX 3070": "RTX3070",
    "NVIDIA GeForce RTX 3060": "RTX3060",
    "NVIDIA RTX 2000 Ada Generation": "RTX2000-Ada",
    "NVIDIA RTX A4000": "RTXA4000",
    "NVIDIA RTX A5000": "RTXA5000",
    "NVIDIA RTX A6000": "RTXA6000",
    "NVIDIA A100 80GB PCIe": "A100-80GB",
    "NVIDIA A100-SXM4-80GB": "A100-80GB",
    "NVIDIA A40": "A40",
    "NVIDIA A10G": "A10G",
    "NVIDIA H100 80GB HBM3": "H100",
    "NVIDIA H100 NVL": "H100-NVL",
    "NVIDIA H100 PCIe": "H100",
    "NVIDIA L40S": "L40S",
    "NVIDIA L40": "L40",
    "NVIDIA L4": "L4",
    "NVIDIA V100 32GB": "V100-32GB",
    "Tesla V100-SXM2-16GB": "V100",
    "NVIDIA Tesla V100 16GB": "V100",
}

# RunPod GPU IDs that expose NVLink / InfiniBand cluster networking.
# RunPod "Secure Cloud" SXM pods have NVLink; PCIe pods do not.
_RUNPOD_INFINIBAND_IDS: frozenset[str] = frozenset({
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",      # H100 SXM NVLink pods on RunPod secure cloud
    "NVIDIA H100 NVL",
})

# AWS instance types with EFA (InfiniBand-equivalent, 100–400 Gbps)
_AWS_EFA_PREFIXES: tuple[str, ...] = ("p4d", "p4de", "p3dn", "p5", "trn1n")

# Vast.ai: treat offers with ≥ 25 Gbps interconnect as InfiniBand-capable.
# The Vast.ai API exposes `inet_up` (upload Mbps); cluster interconnect is not
# separately reported, so we use a conservative threshold for identification.
_VAST_IB_INET_MBPS_THRESHOLD: float = 25_000.0


@dataclass
class GPUOffer:
    provider: str           # "runpod" | "vast" | "aws" | "gcp" | "azure"
    gpu_type: str           # SkyPilot accelerator ID, e.g. "RTX4090"
    gpu_count: int
    vram_gb: float
    vcpus: int
    ram_gb: float
    price_on_demand: float
    price_spot: float | None        # None = no spot option on this provider
    spot_available: bool            # True = spot stock > 0 right now
    available_count: int            # total units available (best-effort)
    region: str
    infiniband: bool
    skypilot_accelerator: str       # e.g. "RTX4090:1"  (for YAML any_of)
    skypilot_cloud: str             # e.g. "runpod"


class GPUCatalogService:
    TTL_SECONDS = 60

    def __init__(self) -> None:
        self._cache: dict[str, list[GPUOffer]] = {}
        self._cache_ts: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def query_availability(
        self,
        providers: list[str] | None = None,
        min_vram_gb: float = 0,
        gpu_types: list[str] | None = None,
    ) -> list[GPUOffer]:
        """Return GPU offers across providers, filtered by constraints."""
        active = set(providers) if providers else {"runpod", "vast", "aws", "gcp", "azure"}

        all_offers: list[GPUOffer] = []
        if "runpod" in active:
            all_offers.extend(self._query_runpod())
        if "vast" in active:
            all_offers.extend(self._query_vast())
        sky_clouds = active & {"aws", "gcp", "azure", "lambda"}
        if sky_clouds:
            all_offers.extend(
                o for o in self._query_skypilot() if o.provider in sky_clouds
            )

        if min_vram_gb > 0:
            all_offers = [o for o in all_offers if o.vram_gb >= min_vram_gb]
        if gpu_types:
            lower = {g.lower() for g in gpu_types}
            all_offers = [o for o in all_offers if o.gpu_type.lower() in lower]

        return all_offers

    # ── Provider queries ──────────────────────────────────────────────────────

    def _query_runpod(self) -> list[GPUOffer]:
        key = "runpod"
        if self._is_cache_valid(key):
            return self._cache[key]
        try:
            import runpod  # type: ignore[import]
            gpus: list[dict[str, Any]] = runpod.get_gpus()
            offers: list[GPUOffer] = []
            for gpu in gpus:
                gpu_id = gpu.get("id", "")
                skypilot_id = _RUNPOD_TO_SKYPILOT.get(gpu_id) or _infer_skypilot_id(
                    gpu.get("displayName", "")
                )
                if not skypilot_id:
                    continue

                vram = float(gpu.get("memoryInGb", 0))
                # Community cloud is cheaper; fall back to secure if absent
                on_demand = float(
                    gpu.get("communityPrice") or gpu.get("securePrice") or 0.0
                )
                spot_raw = gpu.get("communitySpotPrice") or gpu.get("secureSpotPrice")
                spot = float(spot_raw) if spot_raw else None
                spot_available = bool(spot and spot > 0)

                offers.append(
                    GPUOffer(
                        provider="runpod",
                        gpu_type=skypilot_id,
                        gpu_count=1,
                        vram_gb=vram,
                        vcpus=0,  # RunPod doesn't expose per-GPU vCPU count
                        ram_gb=0,
                        price_on_demand=on_demand,
                        price_spot=spot,
                        spot_available=spot_available,
                        available_count=int(gpu.get("maxGpuCount", 1)),
                        region="runpod",
                        infiniband=gpu_id in _RUNPOD_INFINIBAND_IDS,
                        skypilot_accelerator=f"{skypilot_id}:1",
                        skypilot_cloud="runpod",
                    )
                )
            self._set_cache(key, offers)
            return offers
        except Exception as exc:
            logger.warning("RunPod catalog query failed: %s", exc)
            return []

    def _query_vast(self) -> list[GPUOffer]:
        """Query Vast.ai for cheapest offer per GPU type (best-effort)."""
        key = "vast"
        if self._is_cache_valid(key):
            return self._cache[key]
        try:
            from vastai import VastAI  # type: ignore[import]
            api_key = os.getenv("VAST_API_KEY", "")
            if not api_key:
                return []
            client = VastAI(api_key=api_key)
            # Search for single-GPU instances with decent reliability
            raw_offers = client.search_offers(
                query="reliability>0.95 num_gpus=1 rentable=true",
                limit=100,
            ) or []
            # Keep cheapest offer per GPU type
            best: dict[str, dict[str, Any]] = {}
            for raw in raw_offers:
                gpu_name = raw.get("gpu_name", "")
                price = float(raw.get("dph_total", 0))
                if gpu_name not in best or price < float(best[gpu_name].get("dph_total", 99)):
                    best[gpu_name] = raw

            offers: list[GPUOffer] = []
            for gpu_name, raw in best.items():
                skypilot_id = _RUNPOD_TO_SKYPILOT.get(gpu_name) or _infer_skypilot_id(gpu_name)
                if not skypilot_id:
                    continue
                vram = float(raw.get("gpu_ram", 0)) / 1024  # MB → GB
                on_demand = float(raw.get("dph_total", 0))
                # Vast.ai doesn't expose a dedicated IB flag; use upload
                # bandwidth as a proxy for high-speed interconnect.
                inet_up_mbps = float(raw.get("inet_up", 0) or 0)
                has_ib = inet_up_mbps >= _VAST_IB_INET_MBPS_THRESHOLD
                offers.append(
                    GPUOffer(
                        provider="vast",
                        gpu_type=skypilot_id,
                        gpu_count=1,
                        vram_gb=vram,
                        vcpus=int(raw.get("cpu_cores_effective", 0)),
                        ram_gb=float(raw.get("cpu_ram", 0)) / 1024,
                        price_on_demand=on_demand,
                        price_spot=round(on_demand * 0.7, 4),  # Vast spot ~30% cheaper
                        spot_available=True,
                        available_count=1,
                        region=str(raw.get("datacenter", "vast")),
                        infiniband=has_ib,
                        skypilot_accelerator=f"{skypilot_id}:1",
                        skypilot_cloud="vast",
                    )
                )
            self._set_cache(key, offers)
            return offers
        except Exception as exc:
            logger.warning("Vast.ai catalog query failed: %s", exc)
            return []

    def _query_skypilot(self) -> list[GPUOffer]:
        """Query SkyPilot's offline catalog for AWS/GCP/Azure GPUs."""
        key = "skypilot"
        if self._is_cache_valid(key):
            return self._cache[key]
        try:
            import sky  # type: ignore[import]
            catalog: dict[str, list[Any]] = sky.list_accelerators(gpus_only=True)
            offers: list[GPUOffer] = []
            for gpu_name, instances in catalog.items():
                for inst in instances:
                    cloud_raw = str(inst.cloud).lower() if inst.cloud else ""
                    # Strip module prefix: "clouds.aws" → "aws"
                    cloud = cloud_raw.split(".")[-1]
                    if cloud not in ("aws", "gcp", "azure", "lambda"):
                        continue
                    on_demand = float(inst.price or 0)
                    spot = float(inst.spot_price) if getattr(inst, "spot_price", None) else None
                    count = int(inst.accelerator_count or 1)
                    offers.append(
                        GPUOffer(
                            provider=cloud,
                            gpu_type=gpu_name,
                            gpu_count=count,
                            vram_gb=float(getattr(inst, "device_memory", 0) or 0),
                            vcpus=int(getattr(inst, "cpu_count", 0) or 0),
                            ram_gb=float(getattr(inst, "memory", 0) or 0),
                            price_on_demand=on_demand,
                            price_spot=spot,
                            spot_available=bool(spot and spot > 0),
                            available_count=0,  # static catalog — no real-time count
                            region=str(getattr(inst, "region", "") or ""),
                            infiniband=_aws_has_efa(
                                getattr(inst, "instance_type", "") or "", cloud
                            ),
                            skypilot_accelerator=f"{gpu_name}:{count}",
                            skypilot_cloud=cloud,
                        )
                    )
            self._set_cache(key, offers)
            return offers
        except Exception as exc:
            logger.warning("SkyPilot catalog query failed: %s", exc)
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_cache_valid(self, key: str) -> bool:
        return (
            key in self._cache
            and (time.time() - self._cache_ts.get(key, 0)) < self.TTL_SECONDS
        )

    def _set_cache(self, key: str, offers: list[GPUOffer]) -> None:
        self._cache[key] = offers
        self._cache_ts[key] = time.time()


# ── Module-level helpers ──────────────────────────────────────────────────────

def _infer_skypilot_id(display_name: str) -> str:
    """Best-effort: map a free-form GPU name to a SkyPilot accelerator ID."""
    n = display_name.upper().replace(" ", "").replace("-", "").replace("_", "")
    if "RTX4090" in n:
        return "RTX4090"
    if "RTX3090TI" in n:
        return "RTX3090"
    if "RTX3090" in n:
        return "RTX3090"
    if "RTX3080" in n:
        return "RTX3080"
    if "RTX3070" in n:
        return "RTX3070"
    if "RTX2000ADA" in n or "2000ADAGENERATION" in n:
        return "RTX2000-Ada"
    if "A100" in n and "80" in n:
        return "A100-80GB"
    if "A100" in n:
        return "A100"
    if "A40" in n:
        return "A40"
    if "A10G" in n:
        return "A10G"
    if "H100NVL" in n:
        return "H100-NVL"
    if "H100" in n:
        return "H100"
    if "L40S" in n:
        return "L40S"
    if "L40" in n:
        return "L40"
    if "L4" in n and "L40" not in n:
        return "L4"
    if "V100" in n and "32" in n:
        return "V100-32GB"
    if "V100" in n:
        return "V100"
    if "A6000" in n:
        return "RTXA6000"
    if "A5000" in n:
        return "RTXA5000"
    if "A4000" in n:
        return "RTXA4000"
    return ""


def _aws_has_efa(instance_type: str, cloud: str) -> bool:
    if cloud != "aws":
        return False
    return any(instance_type.startswith(p) for p in _AWS_EFA_PREFIXES)
