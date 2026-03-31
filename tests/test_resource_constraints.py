from __future__ import annotations

import sys
from pathlib import Path


_BACKEND = Path(__file__).resolve().parent.parent / "app" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.resource_constraints import build_any_of_from_constraints


def test_strict_gpu_types_without_region_only_selected_gpus():
    constraints = {
        "providers": ["runpod"],
        "gpu_types": ["RTX 2000 Ada", "RTX 4090"],
        "preferred_regions": [],
        "num_gpus_per_node": 1,
        "prefer_spot": True,
    }

    any_of = build_any_of_from_constraints(constraints)

    assert any_of
    assert all(entry["infra"] == "runpod" for entry in any_of)
    accels = {entry["accelerators"] for entry in any_of}
    assert accels == {"RTX2000-Ada:1", "RTX4090:1"}


def test_strict_gpu_types_with_region_only_selected_region():
    constraints = {
        "providers": ["runpod"],
        "gpu_types": ["RTX 2000 Ada", "RTX 4090"],
        "preferred_regions": ["CA-MTL-1"],
        "num_gpus_per_node": 1,
        "prefer_spot": True,
    }

    any_of = build_any_of_from_constraints(constraints)

    assert any_of
    assert all(entry["infra"] == "runpod/CA/CA-MTL-1" for entry in any_of)
    accels = {entry["accelerators"] for entry in any_of}
    assert accels == {"RTX2000-Ada:1", "RTX4090:1"}


def test_strict_gpu_types_with_alias_prefixed_runpod_region_maps_to_canonical_region():
    constraints = {
        "providers": ["runpod"],
        "gpu_types": ["RTX 2000 Ada"],
        "preferred_regions": ["EUR-IS-1"],
        "num_gpus_per_node": 1,
        "prefer_spot": True,
    }

    any_of = build_any_of_from_constraints(constraints)

    assert any_of
    assert all(entry["infra"] == "runpod/IS" for entry in any_of)
    assert {entry["accelerators"] for entry in any_of} == {"RTX2000-Ada:1"}


def test_strict_gpu_types_with_invalid_region_returns_empty_any_of():
    constraints = {
        "providers": ["runpod"],
        "gpu_types": ["RTX 2000 Ada"],
        "preferred_regions": ["INVALID-REGION"],
        "num_gpus_per_node": 1,
        "prefer_spot": True,
    }

    any_of = build_any_of_from_constraints(constraints)

    assert any_of == []


def test_manual_gpu_fallbacks_preserved_and_normalized():
    constraints = {
        "providers": ["runpod"],
        "gpu_fallbacks": [
            {
                "infra": "runpod/CA/CA-MTL-1",
                "accelerators": "RTX 2000 Ada:1",
                "use_spot": True,
            },
            {
                "infra": "runpod/US/US-TX-3",
                "accelerators": "RTX4090:1",
                "use_spot": False,
            },
        ],
        "num_gpus_per_node": 1,
    }

    any_of = build_any_of_from_constraints(constraints)

    assert any_of == [
        {
            "infra": "runpod/CA/CA-MTL-1",
            "accelerators": "RTX2000-Ada:1",
            "use_spot": True,
        },
        {
            "infra": "runpod/US/US-TX-3",
            "accelerators": "RTX4090:1",
            "use_spot": False,
        },
    ]
