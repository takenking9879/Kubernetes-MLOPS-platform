"""
tests/test_gpu_selector.py

Unit tests for GPUSelectorService (Phase 2).

These tests use synthetic GPUOffer lists — no external API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to sys.path so src.services is importable without install
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.gpu_catalog import GPUOffer
from src.services.gpu_selector import GPUSelectorService, ResourceConstraints

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_offer(
    provider: str = "runpod",
    gpu_type: str = "RTX4090",
    vram_gb: float = 24.0,
    price_on_demand: float = 0.59,
    price_spot: float | None = 0.20,
    spot_available: bool = True,
    infiniband: bool = False,
    region: str = "CA-MTL-1",
) -> GPUOffer:
    return GPUOffer(
        provider=provider,
        gpu_type=gpu_type,
        gpu_count=1,
        vram_gb=vram_gb,
        vcpus=0,
        ram_gb=0,
        price_on_demand=price_on_demand,
        price_spot=price_spot,
        spot_available=spot_available,
        available_count=4,
        region=region,
        infiniband=infiniband,
        skypilot_supported=True,
        skypilot_accelerator=f"{gpu_type}:1",
        skypilot_cloud=provider,
    )


SVC = GPUSelectorService()


# ── Tests: spot ordering ───────────────────────────────────────────────────────


def test_spot_entries_before_ondemand():
    """Spot entries must always appear before on-demand in the any_of list."""
    offers = [
        _make_offer(gpu_type="RTX4090", price_spot=0.20, price_on_demand=0.59),
        _make_offer(gpu_type="A40", price_spot=0.30, price_on_demand=0.79),
    ]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"], prefer_spot=True),
        offers=offers,
    )
    spot_indices = [i for i, e in enumerate(result.any_of) if e["use_spot"] is True]
    od_indices = [i for i, e in enumerate(result.any_of) if e["use_spot"] is False]
    assert spot_indices, "Expected at least one spot entry"
    assert od_indices, "Expected at least one on-demand entry"
    assert max(spot_indices) < min(od_indices), (
        f"All spot entries ({spot_indices}) must precede all on-demand entries ({od_indices})"
    )


def test_spot_count_matches():
    """spot_entries + ondemand_entries == len(any_of)."""
    offers = [
        _make_offer(gpu_type="RTX4090", price_spot=0.20),
        _make_offer(gpu_type="A40", price_spot=0.30),
        _make_offer(gpu_type="RTX3090", price_spot=None),  # no spot
    ]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"]), offers=offers
    )
    assert result.spot_entries + result.ondemand_entries == len(result.any_of)


def test_no_spot_when_prefer_spot_false():
    """When prefer_spot=False, no spot entries should appear."""
    offers = [_make_offer(gpu_type="RTX4090", price_spot=0.20)]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"], prefer_spot=False),
        offers=offers,
    )
    assert result.spot_entries == 0
    assert all(e["use_spot"] is False for e in result.any_of)


def test_offer_without_spot_has_no_spot_entry():
    """GPUOffer with price_spot=None produces only an on-demand entry."""
    offers = [_make_offer(gpu_type="RTX4090", price_spot=None)]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"], prefer_spot=True),
        offers=offers,
    )
    assert result.spot_entries == 0
    assert len(result.any_of) == 1
    assert result.any_of[0]["use_spot"] is False


# ── Tests: InfiniBand filtering ────────────────────────────────────────────────


def test_infiniband_hard_exclude():
    """When require_infiniband=True, GPUs without InfiniBand are excluded."""
    offers = [
        _make_offer(gpu_type="RTX4090", infiniband=False),
        _make_offer(gpu_type="A100-80GB", infiniband=True),
    ]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"], require_infiniband=True),
        offers=offers,
    )
    clouds = {e["accelerators"].split(":")[0] for e in result.any_of}
    assert "RTX4090" not in clouds, "Non-InfiniBand GPU must be excluded"
    assert "A100-80GB" in clouds, "InfiniBand GPU must be included"


def test_infiniband_not_required_includes_all():
    """When require_infiniband=False, both GPU types are included."""
    offers = [
        _make_offer(gpu_type="RTX4090", infiniband=False),
        _make_offer(gpu_type="A100-80GB", infiniband=True),
    ]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"], require_infiniband=False),
        offers=offers,
    )
    gpu_types = {e["accelerators"].split(":")[0] for e in result.any_of}
    assert "RTX4090" in gpu_types
    assert "A100-80GB" in gpu_types


# ── Tests: price filtering ─────────────────────────────────────────────────────


def test_max_price_excludes_expensive_entries():
    """Entries (spot or on-demand) exceeding max_price_per_hour are excluded."""
    offers = [
        _make_offer(gpu_type="RTX4090", price_spot=0.20, price_on_demand=0.59),
        _make_offer(gpu_type="H100", price_spot=2.50, price_on_demand=5.00),
    ]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"], max_price_per_hour=1.00),
        offers=offers,
    )
    for entry in result.any_of:
        assert "H100" not in entry["accelerators"], (
            "H100 exceeds max_price_per_hour=1.00 and must be excluded"
        )


# ── Tests: VRAM filtering ──────────────────────────────────────────────────────


def test_min_vram_filters_small_gpus():
    """Offers below min_vram_gb are filtered out before scoring."""
    offers = [
        _make_offer(gpu_type="RTX3060", vram_gb=12.0),
        _make_offer(gpu_type="A100-80GB", vram_gb=80.0),
    ]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"], min_vram_gb=24.0),
        offers=offers,
    )
    gpu_types = {e["accelerators"].split(":")[0] for e in result.any_of}
    assert "RTX3060" not in gpu_types
    assert "A100-80GB" in gpu_types


# ── Tests: cost estimates ──────────────────────────────────────────────────────


def test_estimated_costs_populated():
    """estimated_cost_spot and estimated_cost_ondemand are non-None when offers exist."""
    offers = [_make_offer(gpu_type="RTX4090", price_spot=0.20, price_on_demand=0.59)]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"]), offers=offers
    )
    assert result.estimated_cost_spot == pytest.approx(0.20)
    assert result.estimated_cost_ondemand == pytest.approx(0.59)


def test_empty_offers_returns_empty():
    """Empty offer list produces empty any_of and None cost estimates."""
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"]), offers=[]
    )
    assert result.any_of == []
    assert result.spot_entries == 0
    assert result.ondemand_entries == 0
    assert result.estimated_cost_spot is None
    assert result.estimated_cost_ondemand is None


# ── Tests: deduplication ───────────────────────────────────────────────────────


def test_duplicate_provider_gpu_deduplicated():
    """Two offers for the same (provider, gpu_type) keep only the cheapest."""
    offers = [
        _make_offer(gpu_type="RTX4090", price_on_demand=0.59, price_spot=0.20),
        _make_offer(gpu_type="RTX4090", price_on_demand=0.45, price_spot=0.18),
    ]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"]), offers=offers
    )
    rtx_entries = [e for e in result.any_of if "RTX4090" in e["accelerators"]]
    # At most one spot + one on-demand entry for RTX4090
    assert len(rtx_entries) <= 2
    spot_entries = [e for e in rtx_entries if e["use_spot"]]
    od_entries = [e for e in rtx_entries if not e["use_spot"]]
    assert len(spot_entries) <= 1
    assert len(od_entries) <= 1


# ── Tests: num_gpus_per_node ───────────────────────────────────────────────────


def test_num_gpus_per_node_reflected_in_accelerators():
    """num_gpus_per_node=4 produces 'RTX4090:4' in the accelerators field."""
    offers = [_make_offer(gpu_type="RTX4090")]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"], num_gpus_per_node=4),
        offers=offers,
    )
    assert any("RTX4090:4" in e["accelerators"] for e in result.any_of)


# ── Tests: provider filter ─────────────────────────────────────────────────────


def test_provider_filter_excludes_other_providers():
    """Offers from providers not in constraints.providers are excluded."""
    offers = [
        _make_offer(provider="runpod", gpu_type="RTX4090"),
        _make_offer(provider="vast", gpu_type="RTX3090"),
    ]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"]), offers=offers
    )
    clouds = {e["infra"].split("/")[0] for e in result.any_of}
    assert "vast" not in clouds
    assert "runpod" in clouds


def test_runpod_preferred_region_filters_entries():
    """Preferred RunPod regions keep only matching region candidates."""
    offers = [
        _make_offer(provider="runpod", gpu_type="RTX4090", region="CA-MTL-1"),
        _make_offer(provider="runpod", gpu_type="RTX4090", region="US-TX-3", price_on_demand=0.50),
    ]
    result = SVC.select_providers(
        ResourceConstraints(providers=["runpod"], preferred_regions=["CA-MTL-1"]),
        offers=offers,
    )
    assert result.any_of
    assert all(e["infra"].startswith("runpod/CA") for e in result.any_of)
