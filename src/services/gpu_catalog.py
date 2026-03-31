"""
src/services/gpu_catalog.py

Unified GPU availability catalog: RunPod (real-time) + Vast.ai (real-time) +
AWS via SkyPilot offline catalog + boto3 spot prices.

Each provider query is independent — failure returns [] without raising.
Results are cached: provider offers for TTL_SECONDS (60 s), SkyPilot catalog
for 5 minutes (slow subprocess call).

Architecture (Espacio Y):
  gpu_type  = SkyPilot canonical name if skypilot_supported=True, else native
  skypilot_supported = True only when SkyPilot's offline catalog confirms the
                       GPU is launchable on that provider (via _sky_supported_names).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .provider_region_mapping import normalize_provider_region
from .runpod_adapter import RunPodAdapter

# Load .env from project root (src/services/ → src/ → project root)
# This is a no-op in K8s pods (env-secret injects vars directly).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

# ── SkyPilot venv paths ───────────────────────────────────────────────────────
_SKY_PYTHON = Path("/opt/sky-venv/bin/python")
_SKY_SCRIPT = Path(__file__).with_name("_sky_catalog_query.py")

# AWS regions where GPU instances are commonly available
_AWS_ML_REGIONS = [
    "us-east-1", "us-east-2", "us-west-2",
    "eu-west-1", "ap-northeast-1", "ap-southeast-1",
]

# ── Hardcoded fallback GPU sets (used when SkyPilot is not installed) ─────────
# Derived from `sky show-gpus --infra {cloud} -a` with SkyPilot 0.12.0.
# These are updated when new GPU types are added to the SkyPilot catalogs.
_SKY_RUNPOD_KNOWN: frozenset[str] = frozenset({
    "A100-80GB", "A100-80GB-SXM", "A40", "B200", "H100", "H100-NVL",
    "H100-SXM", "H200-SXM", "L4", "L40", "L40S", "MI300X", "NVIDIA-H200-NVL",
    "RTX2000-Ada", "RTX3090", "RTX4000-Ada", "RTX4090", "RTX5090", "RTX6000-Ada",
    "RTXA4000", "RTXA4500", "RTXA5000", "RTXA6000", "RTXPRO4500", "RTXPRO6000",
    "RTXPRO6000-WK",
})
_SKY_VAST_KNOWN: frozenset[str] = frozenset({
    "A100", "H100", "H200", "L40S", "RTX3060", "RTX3090", "RTX4060", "RTX4070",
    "RTX4090", "RTX5070", "RTX5090", "RTX5880-Ada", "RTX6000-Ada",
    "RTXA5000", "RTXA6000", "RTXPRO6000WS", "V100",
})
_SKY_AWS_KNOWN: frozenset[str] = frozenset({
    "A100", "A100-80GB", "A10G", "B200", "B300", "H100", "H200",
    "L4", "L40S", "RTXPRO6000", "T4", "V100", "V100-32GB",
})

# ── RunPod GPU displayName/id → SkyPilot accelerator ID ──────────────────────
# Keys are the `id` field returned by RunPod's GraphQL API (often = displayName).
_RUNPOD_TO_SKYPILOT: dict[str, str] = {
    # Consumer
    "NVIDIA GeForce RTX 5090":          "RTX5090",
    "NVIDIA GeForce RTX 4090":          "RTX4090",
    "NVIDIA GeForce RTX 3090 Ti":       "RTX3090",
    "NVIDIA GeForce RTX 3090":          "RTX3090",
    "NVIDIA GeForce RTX 3080":          "RTX3080",
    "NVIDIA GeForce RTX 3070":          "RTX3070",
    "NVIDIA GeForce RTX 3060":          "RTX3060",
    # Workstation / Ada
    "NVIDIA RTX 2000 Ada Generation":   "RTX2000-Ada",
    "NVIDIA RTX 4000 Ada Generation":   "RTX4000-Ada",
    "NVIDIA RTX 6000 Ada Generation":   "RTX6000-Ada",
    "NVIDIA RTX A4000":                 "RTXA4000",
    "NVIDIA RTX A4500":                 "RTXA4500",
    "NVIDIA RTX A5000":                 "RTXA5000",
    "NVIDIA RTX A6000":                 "RTXA6000",
    # Data center — A series
    "NVIDIA A100 80GB PCIe":            "A100-80GB",
    "NVIDIA A100-SXM4-80GB":            "A100-80GB-SXM",
    "NVIDIA A40":                       "A40",
    "NVIDIA A10G":                      "A10G",
    # H100 series
    "NVIDIA H100 80GB HBM3":            "H100-SXM",
    "NVIDIA H100 NVL":                  "H100-NVL",
    "NVIDIA H100 PCIe":                 "H100",
    # H200 series (SkyPilot canonical: H200-SXM for RunPod, H200 for Vast)
    "NVIDIA H200":                      "H200-SXM",
    "NVIDIA H200 NVL":                  "NVIDIA-H200-NVL",
    # B series
    "NVIDIA B200":                      "B200",
    "NVIDIA B300 SXM6 AC":              "B300",
    # L series
    "NVIDIA L40S":                      "L40S",
    "NVIDIA L40":                       "L40",
    "NVIDIA L4":                        "L4",
    # V series
    "NVIDIA V100 32GB":                 "V100-32GB",
    "Tesla V100-SXM2-16GB":             "V100",
    "NVIDIA Tesla V100 16GB":           "V100",
    # AMD
    "AMD Instinct MI300X":              "MI300X",
}

# RunPod GPU IDs with NVLink / high-speed interconnect
_RUNPOD_INFINIBAND_IDS: frozenset[str] = frozenset({
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
    "NVIDIA H200",
    "NVIDIA B200",
})

# AWS instance type prefixes with EFA (InfiniBand-equivalent, 100–400 Gbps)
_AWS_EFA_PREFIXES: tuple[str, ...] = ("p4d", "p4de", "p3dn", "p5", "p6", "trn1n")

# Vast.ai: treat offers with ≥ 25 Gbps upload as high-speed interconnect.
_VAST_IB_INET_MBPS_THRESHOLD: float = 25_000.0


@dataclass
class GPUOffer:
    provider: str           # "runpod" | "vast" | "aws"
    gpu_type: str           # SkyPilot canonical name if supported, else native name
    gpu_count: int
    vram_gb: float
    vcpus: int
    ram_gb: float
    price_on_demand: float
    price_spot: float | None        # None = no spot option on this provider
    spot_available: bool            # True = spot stock available right now
    available_count: int            # total units available (best-effort)
    region: str
    infiniband: bool
    skypilot_supported: bool        # True = SkyPilot can launch this GPU
    skypilot_accelerator: str       # e.g. "RTX4090:1" — empty if not supported
    skypilot_cloud: str             # e.g. "runpod"
    provider_region_id: str = ""
    skypilot_region: str = ""
    skypilot_zone: str = ""
    skypilot_infra: str = ""


class GPUCatalogService:
    TTL_SECONDS = 60

    def __init__(self) -> None:
        self._cache: dict[str, list[GPUOffer]] = {}
        self._cache_ts: dict[str, float] = {}
        # SkyPilot catalog has a longer TTL (subprocess is slow)
        self._sky_catalog_cache: dict | None = None
        self._sky_catalog_ts: float = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def query_availability(
        self,
        providers: list[str] | None = None,
        min_vram_gb: float = 0,
        gpu_types: list[str] | None = None,
    ) -> list[GPUOffer]:
        """Return GPU offers across providers, filtered by constraints."""
        active = set(providers) if providers else {"runpod", "vast", "aws"}

        # Fetch SkyPilot catalog once — shared by all provider queries.
        sky_catalog = self._get_sky_catalog()

        all_offers: list[GPUOffer] = []
        if "runpod" in active:
            all_offers.extend(self._query_runpod(sky_catalog))
        if "vast" in active:
            all_offers.extend(self._query_vast(sky_catalog))
        if "aws" in active:
            all_offers.extend(self._query_aws(sky_catalog))

        if min_vram_gb > 0:
            all_offers = [o for o in all_offers if o.vram_gb >= min_vram_gb]
        if gpu_types:
            lower = {g.lower() for g in gpu_types}
            all_offers = [o for o in all_offers if o.gpu_type.lower() in lower]

        return all_offers

    # ── SkyPilot catalog (unified, all three providers) ───────────────────────

    def _get_sky_catalog(self) -> dict:
        """
        Return SkyPilot's offline GPU catalog for aws+runpod+vast (5-min cache).

        Result: { gpu_name: [{cloud, region, price, spot_price,
                               accelerator_count, device_memory,
                               cpu_count, memory, instance_type}] }

        Tries in order:
          1. Subprocess to /opt/sky-venv (Docker/K8s with SkyPilot venv)
          2. Direct sky import (local dev environment)
          3. Returns {} → callers fall back to _static_aws_catalog()
        """
        if (
            self._sky_catalog_cache is not None
            and (time.time() - self._sky_catalog_ts) < 300
        ):
            return self._sky_catalog_cache

        catalog: dict = {}

        # 1. Subprocess to isolated SkyPilot venv
        if _SKY_PYTHON.exists() and _SKY_SCRIPT.exists():
            try:
                r = subprocess.run(
                    [str(_SKY_PYTHON), str(_SKY_SCRIPT)],
                    capture_output=True, text=True, timeout=60,
                )
                if r.returncode == 0 and r.stdout:
                    catalog = json.loads(r.stdout)
                    logger.info(
                        "SkyPilot catalog loaded via venv subprocess (%d GPU types)",
                        len(catalog),
                    )
            except Exception as exc:
                logger.warning("SkyPilot venv subprocess failed: %s", exc)

        # 2. Direct import (local dev with sky installed in PATH)
        if not catalog:
            try:
                import math
                import sky  # type: ignore[import]

                raw = sky.list_accelerators(
                    gpus_only=True, clouds=["aws", "runpod", "vast"]
                )
                for gpu_name, insts in raw.items():
                    rows = []
                    for i in insts:
                        cloud = str(i.cloud or "").lower().split(".")[-1]
                        if cloud not in {"aws", "runpod", "vast"}:
                            continue
                        rows.append({
                            "cloud":             cloud,
                            "region":            str(getattr(i, "region", "") or ""),
                            "price":             _f_none(getattr(i, "price", 0)) or 0.0,
                            "spot_price":        _f_none(getattr(i, "spot_price", None)),
                            "accelerator_count": int(getattr(i, "accelerator_count", 1) or 1),
                            "device_memory":     _f_none(getattr(i, "device_memory", 0)) or 0.0,
                            "cpu_count":         int(getattr(i, "cpu_count", 0) or 0),
                            "memory":            _f_none(getattr(i, "memory", 0)) or 0.0,
                            "instance_type":     str(getattr(i, "instance_type", "") or ""),
                        })
                    if rows:
                        catalog[gpu_name] = rows
                if catalog:
                    logger.info(
                        "SkyPilot catalog loaded via direct import (%d GPU types)",
                        len(catalog),
                    )
            except ImportError:
                pass
            except Exception as exc:
                logger.warning("SkyPilot direct import failed: %s", exc)

        self._sky_catalog_cache = catalog
        self._sky_catalog_ts = time.time()
        return catalog

    # ── Provider queries ──────────────────────────────────────────────────────

    def _query_runpod(self, sky_catalog: dict) -> list[GPUOffer]:
        cache_key = "runpod"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        try:
            import runpod  # type: ignore[import]
            # env-secret injects the key after Python imports — set explicitly.
            runpod.api_key = os.getenv("RUNPOD_API_KEY", "")
            if not runpod.api_key:
                logger.warning("RUNPOD_API_KEY not set — skipping RunPod catalog")
                return []
            # Use a single global query for pricing + availability.
            from runpod.api import graphql as _gql  # type: ignore[import]
            result = _gql.run_graphql_query(
                "{ gpuTypes { id displayName memoryInGb "
                "communityPrice securePrice communitySpotPrice secureSpotPrice "
                "lowestPrice(input: { gpuCount: 1, secureCloud: true, "
                "minVcpuCount: 2, minMemoryInGb: 8 }) { "
                "stockStatus availableGpuCounts maxUnreservedGpuCount "
                "uninterruptablePrice minimumBidPrice } } }"
            )
            gpus: list[dict[str, Any]] = (
                result.get("data", {}).get("gpuTypes", []) or []
            )

            sky_runpod = _sky_supported_names(sky_catalog, "runpod")
            offers: list[GPUOffer] = []
            for gpu in gpus:
                gpu_id = str(gpu.get("id") or "")
                display_name = gpu.get("displayName", "") or gpu_id
                if not gpu_id or gpu_id.lower() == "unknown":
                    continue

                # Map to SkyPilot canonical name; validate against catalog.
                skypilot_id = _RUNPOD_TO_SKYPILOT.get(gpu_id) or _infer_skypilot_id(display_name)
                if skypilot_id and skypilot_id in sky_runpod:
                    sky_supported = True
                    gpu_type = skypilot_id
                else:
                    sky_supported = False
                    gpu_type = display_name  # show native name when unsupported

                vram = float(gpu.get("memoryInGb", 0))
                lowest_price = gpu.get("lowestPrice")
                on_demand, spot = RunPodAdapter.extract_price(lowest_price, gpu)
                available = RunPodAdapter.is_available(lowest_price, gpu_count=1)
                count = RunPodAdapter.available_count(lowest_price)
                mapping = normalize_provider_region("runpod", "")

                offers.append(
                    GPUOffer(
                        provider="runpod",
                        gpu_type=gpu_type,
                        gpu_count=1,
                        vram_gb=vram,
                        vcpus=0,
                        ram_gb=0,
                        price_on_demand=on_demand,
                        price_spot=spot,
                        spot_available=available,
                        available_count=count,
                        region="",
                        infiniband=gpu_id in _RUNPOD_INFINIBAND_IDS,
                        skypilot_supported=sky_supported,
                        skypilot_accelerator=f"{gpu_type}:1" if sky_supported else "",
                        skypilot_cloud="runpod",
                        provider_region_id="",
                        skypilot_region=mapping.skypilot_region,
                        skypilot_zone=mapping.skypilot_zone,
                        skypilot_infra=mapping.skypilot_infra,
                    )
                )
            offers = _dedupe_runpod_offers(offers)
            self._set_cache(cache_key, offers)
            return offers
        except Exception as exc:
            logger.warning("RunPod catalog query failed: %s", exc)
            return []

    def _query_vast(self, sky_catalog: dict) -> list[GPUOffer]:
        """Query Vast.ai for cheapest offer per GPU type (best-effort)."""
        cache_key = "vast"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        try:
            # vastai reads ~/.vast_ai/ at import; K8s pods may have HOME=/nonexistent.
            import pathlib
            _home = os.environ.get("HOME", "")
            if not _home or not pathlib.Path(_home).is_dir():
                os.environ["HOME"] = "/tmp"
            from vastai import VastAI  # type: ignore[import]
            api_key = os.getenv("VAST_API_KEY", "") or os.getenv("VASTAI_API_KEY", "")
            if not api_key:
                return []
            client = VastAI(api_key=api_key)
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

            sky_vast = _sky_supported_names(sky_catalog, "vast")
            offers: list[GPUOffer] = []
            for gpu_name, raw in best.items():
                if not gpu_name:
                    continue

                skypilot_id = _RUNPOD_TO_SKYPILOT.get(gpu_name) or _infer_skypilot_id(gpu_name)
                if skypilot_id and skypilot_id in sky_vast:
                    sky_supported = True
                    gpu_type = skypilot_id
                else:
                    # Vast uses bare canonical names ("H100", "A100") while inference
                    # may return RunPod-biased variant-suffixed names ("H100-SXM",
                    # "A100-80GB"). Try exact match on the base component only.
                    base = skypilot_id.split("-")[0] if skypilot_id else ""
                    base_candidates = [k for k in sky_vast if k == base]
                    if len(base_candidates) == 1:
                        sky_supported = True
                        gpu_type = base_candidates[0]
                    else:
                        sky_supported = False
                        gpu_type = gpu_name

                vram = float(raw.get("gpu_ram", 0)) / 1024  # MB → GB
                on_demand = float(raw.get("dph_total", 0))
                inet_up_mbps = float(raw.get("inet_up", 0) or 0)
                has_ib = inet_up_mbps >= _VAST_IB_INET_MBPS_THRESHOLD
                provider_region = str(raw.get("datacenter", ""))
                mapping = normalize_provider_region("vast", provider_region)

                offers.append(
                    GPUOffer(
                        provider="vast",
                        gpu_type=gpu_type,
                        gpu_count=1,
                        vram_gb=vram,
                        vcpus=int(raw.get("cpu_cores_effective", 0)),
                        ram_gb=float(raw.get("cpu_ram", 0)) / 1024,
                        price_on_demand=on_demand,
                        price_spot=None,        # Vast.ai has NO spot pricing
                        spot_available=False,   # confirmed by SkyPilot catalog
                        available_count=1,
                        region=provider_region,
                        infiniband=has_ib,
                        skypilot_supported=sky_supported,
                        skypilot_accelerator=f"{gpu_type}:1" if sky_supported else "",
                        skypilot_cloud="vast",
                        provider_region_id=provider_region,
                        skypilot_region=mapping.skypilot_region,
                        skypilot_zone=mapping.skypilot_zone,
                        skypilot_infra=mapping.skypilot_infra,
                    )
                )
            self._set_cache(cache_key, offers)
            return offers
        except Exception as exc:
            logger.warning("Vast.ai catalog query failed: %s", exc)
            return []

    def _query_aws(self, sky_catalog: dict) -> list[GPUOffer]:
        """Query AWS GPU offers using SkyPilot catalog + boto3 real-time spot prices."""
        cache_key = "aws"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        try:
            aws_entries: dict[str, list[dict]] = {
                gpu_name: [r for r in rows if r.get("cloud") == "aws"]
                for gpu_name, rows in sky_catalog.items()
                if any(r.get("cloud") == "aws" for r in rows)
            }

            if not aws_entries:
                logger.info("SkyPilot AWS catalog empty — using static fallback")
                offers = _static_aws_catalog()
                self._set_cache(cache_key, offers)
                return offers

            # Collect unique instance types for spot price lookup
            instance_types: list[str] = list({
                r["instance_type"]
                for rows in aws_entries.values()
                for r in rows
                if r.get("instance_type")
            })

            # Fetch real-time spot prices from EC2
            spot_prices = self._get_aws_spot_prices(instance_types)

            offers: list[GPUOffer] = []
            seen: set[tuple[str, str]] = set()
            for gpu_name, rows in aws_entries.items():
                for row in rows:
                    region = row.get("region", "")
                    dedup_key = (gpu_name, region)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    instance_type = row.get("instance_type", "")
                    on_demand = float(row.get("price", 0))
                    # Real-time spot preferred; fall back to SkyPilot catalog spot
                    spot = spot_prices.get((instance_type, region)) or _f_none(
                        row.get("spot_price")
                    )
                    count = int(row.get("accelerator_count", 1) or 1)
                    mapping = normalize_provider_region("aws", region)

                    offers.append(
                        GPUOffer(
                            provider="aws",
                            gpu_type=gpu_name,
                            gpu_count=count,
                            vram_gb=float(row.get("device_memory", 0)),
                            vcpus=int(row.get("cpu_count", 0)),
                            ram_gb=float(row.get("memory", 0)),
                            price_on_demand=on_demand,
                            price_spot=spot,
                            spot_available=bool(spot and spot > 0),
                            available_count=-1,  # sentinel: AWS availability not tracked
                            region=region,
                            infiniband=_aws_has_efa(instance_type, "aws"),
                            skypilot_supported=True,  # AWS always from SkyPilot catalog
                            skypilot_accelerator=f"{gpu_name}:{count}",
                            skypilot_cloud="aws",
                            provider_region_id=region,
                            skypilot_region=mapping.skypilot_region,
                            skypilot_zone=mapping.skypilot_zone,
                            skypilot_infra=mapping.skypilot_infra,
                        )
                    )

            self._set_cache(cache_key, offers)
            return offers
        except Exception as exc:
            logger.warning("AWS catalog query failed (%s) — using static fallback", exc)
            offers = _static_aws_catalog()
            self._set_cache(cache_key, offers)
            return offers

    def _get_aws_spot_prices(
        self, instance_types: list[str]
    ) -> dict[tuple[str, str], float]:
        """Fetch current AWS spot prices via boto3. Returns {(instance_type, region): price}."""
        spot_prices: dict[tuple[str, str], float] = {}
        if not instance_types:
            return spot_prices
        try:
            import boto3  # type: ignore[import]
            batch = instance_types[:50]  # EC2 API limit per call
            for region in _AWS_ML_REGIONS:
                try:
                    ec2 = boto3.client("ec2", region_name=region)
                    paginator = ec2.get_paginator("describe_spot_price_history")
                    for page in paginator.paginate(
                        InstanceTypes=batch,
                        ProductDescriptions=["Linux/UNIX"],
                        StartTime=datetime.now(timezone.utc),
                    ):
                        for item in page["SpotPriceHistory"]:
                            itype = item["InstanceType"]
                            price = float(item["SpotPrice"])
                            k = (itype, region)
                            if k not in spot_prices or price < spot_prices[k]:
                                spot_prices[k] = price
                except Exception as exc:
                    logger.debug("Spot price fetch failed for %s: %s", region, exc)
        except ImportError:
            logger.debug("boto3 not available — skipping AWS spot prices")
        return spot_prices

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _is_cache_valid(self, key: str) -> bool:
        return (
            key in self._cache
            and (time.time() - self._cache_ts.get(key, 0)) < self.TTL_SECONDS
        )

    def _set_cache(self, key: str, offers: list[GPUOffer]) -> None:
        self._cache[key] = offers
        self._cache_ts[key] = time.time()


# ── Module-level helpers ──────────────────────────────────────────────────────


def _sky_supported_names(catalog: dict, cloud: str) -> set[str]:
    """Return GPU names SkyPilot supports for a cloud. Falls back to hardcoded
    sets when the catalog is unavailable (e.g. sky not installed in the pod)."""
    if catalog:
        return {
            gpu_name
            for gpu_name, rows in catalog.items()
            if any(r.get("cloud") == cloud for r in rows)
        }
    # Sky not available — use hardcoded sets from the last known catalog.
    if cloud == "runpod":
        return set(_SKY_RUNPOD_KNOWN)
    if cloud == "vast":
        return set(_SKY_VAST_KNOWN)
    if cloud == "aws":
        return set(_SKY_AWS_KNOWN)
    return set()


def _f_none(v: Any) -> float | None:
    """Parse float; return None on NaN/None/error."""
    if v is None:
        return None
    try:
        import math
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _dedupe_runpod_offers(offers: list[GPUOffer]) -> list[GPUOffer]:
    """Keep a single best RunPod offer per canonical GPU type.

    RunPod may return multiple raw GPU IDs that map to the same SkyPilot
    accelerator name. For the pricing table we keep one row per gpu_type,
    preferring: in-stock offers first, then lower effective price.
    """
    best_by_gpu: dict[str, GPUOffer] = {}
    for offer in offers:
        current = best_by_gpu.get(offer.gpu_type)
        if current is None or _runpod_offer_rank(offer) < _runpod_offer_rank(current):
            best_by_gpu[offer.gpu_type] = offer
    return list(best_by_gpu.values())


def _runpod_offer_rank(offer: GPUOffer) -> tuple[int, float, float, int]:
    """Sort key for selecting the best representative RunPod offer."""
    effective_price = (
        float(offer.price_spot)
        if offer.price_spot is not None and offer.price_spot > 0
        else float(offer.price_on_demand)
    )
    return (
        0 if offer.available_count > 0 else 1,
        effective_price,
        float(offer.price_on_demand),
        -int(offer.available_count),
    )


def _infer_skypilot_id(display_name: str) -> str:
    """Best-effort: map a free-form GPU name to a SkyPilot accelerator ID."""
    n = display_name.upper().replace(" ", "").replace("-", "").replace("_", "")
    if "RTX5090" in n:
        return "RTX5090"
    if "RTX5880" in n:
        return "RTX5880-Ada"
    if "RTX5070" in n:
        return "RTX5070"
    if "RTX4090" in n:
        return "RTX4090"
    if "RTX4070" in n:
        return "RTX4070"
    if "RTX4060" in n:
        return "RTX4060"
    if "RTX3090TI" in n:
        return "RTX3090"
    if "RTX3090" in n:
        return "RTX3090"
    if "RTX3080" in n:
        return "RTX3080"
    if "RTX3070" in n:
        return "RTX3070"
    if "RTX3060" in n:
        return "RTX3060"
    if "RTX2000ADA" in n or "2000ADAGENERATION" in n:
        return "RTX2000-Ada"
    if "RTX4000ADA" in n or "4000ADAGENERATION" in n:
        return "RTX4000-Ada"
    if "RTX6000ADA" in n or "6000ADAGENERATION" in n:
        return "RTX6000-Ada"
    if "RTXPRO6000WS" in n:
        return "RTXPRO6000WS"
    if "RTXPRO6000WK" in n:
        return "RTXPRO6000-WK"
    if "RTXPRO6000" in n:
        return "RTXPRO6000"
    if "RTXPRO4500" in n:
        return "RTXPRO4500"
    if "A100" in n and "80" in n and "SXM" in n:
        return "A100-80GB-SXM"
    if "A100" in n and "80" in n:
        return "A100-80GB"
    if "A100" in n and "SXM" in n:
        return "A100-80GB-SXM"
    if "A100" in n:
        return "A100"
    # A4500 / A4000 BEFORE A40 — prevents "RTXA4000" → "A40" false match
    if "A4500" in n:
        return "RTXA4500"
    if "A4000" in n:
        return "RTXA4000"
    if "A40" in n:
        return "A40"
    if "A10G" in n:
        return "A10G"
    if "B300" in n:
        return "B300"
    if "B200" in n:
        return "B200"
    if "H200SXM" in n or ("H200" in n and "SXM" in n):
        return "H200-SXM"
    if "H200NVL" in n:
        return "H200"
    if "H200" in n:
        return "H200-SXM"  # RunPod H200 is H200-SXM in SkyPilot
    if "H100NVL" in n:
        return "H100-NVL"
    if "H100SXM" in n or ("H100" in n and "SXM" in n):
        return "H100-SXM"
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
    if "MI300X" in n:
        return "MI300X"
    return ""


# Static AWS GPU catalog — fallback when SkyPilot is unavailable.
# Prices are us-east-1 on-demand (2025). Spot ≈ 70% of on-demand.
_AWS_STATIC: list[tuple[str, int, float, float, int, str, bool]] = [
    # (gpu_type, vram_gb, od_price, spot_price, count, instance, efa)
    ("T4",          16,   0.526,  0.158,  1, "g4dn.xlarge",       False),
    ("T4",          16,   1.686,  0.506,  4, "g4dn.12xlarge",     False),
    ("A10G",        24,   1.006,  0.344,  1, "g5.xlarge",         False),
    ("A10G",        24,   1.212,  0.320,  1, "g5.2xlarge",        False),
    ("A10G",        24,   7.228,  2.168,  4, "g5.24xlarge",       False),
    ("A10G",        24,  16.288,  4.987,  8, "g5.48xlarge",       False),
    ("V100",        16,   3.060,  0.918,  1, "p3.2xlarge",        False),
    ("V100",        16,  12.240,  3.672,  4, "p3.8xlarge",        False),
    ("V100",        16,  24.480,  7.344,  8, "p3.16xlarge",       False),
    ("V100-32GB",   32,  31.212,  9.364,  8, "p3dn.24xlarge",     True),
    ("A100",        40,  21.958, 10.302,  8, "p4d.24xlarge",      True),   # 40 GB PCIe
    ("A100-80GB",   80,  27.447, 15.532,  8, "p4de.24xlarge",     True),   # 80 GB PCIe
    ("L4",          22,   0.805,  0.217,  1, "g6.xlarge",         False),
    ("L4",          22,  13.350,  3.367,  8, "g6.48xlarge",       False),
    ("L40S",        45,   1.861,  0.525,  1, "g6e.xlarge",        False),
    ("L40S",        45,  30.131,  5.103,  8, "g6e.48xlarge",      False),
    ("H100",        80,   6.880,  2.966,  1, "p5.4xlarge",        True),
    ("H100",        80,  55.040, 10.009,  8, "p5.48xlarge",       True),
    ("H200",       141,  63.296,  9.868,  8, "p5en.48xlarge",     True),
    ("B200",       179, 113.933, 11.393,  8, "p6-b200.48xlarge",  True),
    ("RTXPRO6000",  96,   3.363,  0.904,  1, "g7e.2xlarge",       False),
    ("RTXPRO6000",  96,  33.144,  4.036,  8, "g7e.48xlarge",      False),
]


def _static_aws_catalog() -> list[GPUOffer]:
    offers = []
    for gpu_type, vram_gb, od, spot, count, instance, efa in _AWS_STATIC:
        mapping = normalize_provider_region("aws", "us-east-1")
        offers.append(
            GPUOffer(
                provider="aws",
                gpu_type=gpu_type,
                gpu_count=count,
                vram_gb=float(vram_gb),
                vcpus=0,
                ram_gb=0,
                price_on_demand=od,
                price_spot=spot,
                spot_available=True,
                available_count=-1,  # sentinel: AWS availability not tracked
                region="us-east-1",
                infiniband=efa,
                skypilot_supported=True,
                skypilot_accelerator=f"{gpu_type}:{count}",
                skypilot_cloud="aws",
                provider_region_id="us-east-1",
                skypilot_region=mapping.skypilot_region,
                skypilot_zone=mapping.skypilot_zone,
                skypilot_infra=mapping.skypilot_infra,
            )
        )
    return offers


def _aws_has_efa(instance_type: str, cloud: str) -> bool:
    if cloud != "aws":
        return False
    return any(instance_type.startswith(p) for p in _AWS_EFA_PREFIXES)
