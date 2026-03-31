from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.gpu_catalog import GPUOffer, _dedupe_runpod_offers


def _make_offer(
    gpu_type: str,
    *,
    available_count: int,
    price_on_demand: float,
    price_spot: float | None,
) -> GPUOffer:
    return GPUOffer(
        provider="runpod",
        gpu_type=gpu_type,
        gpu_count=1,
        vram_gb=16.0,
        vcpus=0,
        ram_gb=0.0,
        price_on_demand=price_on_demand,
        price_spot=price_spot,
        spot_available=available_count > 0,
        available_count=available_count,
        region="",
        infiniband=False,
        skypilot_supported=True,
        skypilot_accelerator=f"{gpu_type}:1",
        skypilot_cloud="runpod",
        provider_region_id="",
        skypilot_region="",
        skypilot_zone="",
        skypilot_infra="runpod",
    )


def test_dedupe_runpod_offers_prefers_in_stock_entry():
    offers = [
        _make_offer("RTX4000-Ada", available_count=0, price_on_demand=0.20, price_spot=0.09),
        _make_offer("RTX4000-Ada", available_count=3, price_on_demand=0.26, price_spot=0.19),
    ]

    deduped = _dedupe_runpod_offers(offers)

    assert len(deduped) == 1
    assert deduped[0].gpu_type == "RTX4000-Ada"
    assert deduped[0].available_count == 3


def test_dedupe_runpod_offers_keeps_lowest_price_when_stock_ties():
    offers = [
        _make_offer("RTX2000-Ada", available_count=0, price_on_demand=0.50, price_spot=0.25),
        _make_offer("RTX2000-Ada", available_count=0, price_on_demand=0.44, price_spot=0.21),
        _make_offer("L4", available_count=2, price_on_demand=0.39, price_spot=0.22),
    ]

    deduped = _dedupe_runpod_offers(offers)

    picked = {offer.gpu_type: offer for offer in deduped}
    assert set(picked) == {"RTX2000-Ada", "L4"}
    assert picked["RTX2000-Ada"].price_spot == 0.21
    assert picked["L4"].available_count == 2
