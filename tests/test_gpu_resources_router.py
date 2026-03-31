from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.backend.routers.gpu_resources import (
    _aggregate_runpod_availability_rows,
    _build_runpod_availability_matrix_query,
)


def test_build_runpod_availability_matrix_query_has_alias_per_region():
    query, alias_to_region = _build_runpod_availability_matrix_query(
        ["EU-RO-1", "EUR-IS-1"],
    )

    assert alias_to_region == {
        "r0": "EU-RO-1",
        "r1": "EUR-IS-1",
    }
    assert 'r0: lowestPrice(input: {' in query
    assert 'r1: lowestPrice(input: {' in query
    assert 'dataCenterId: "EU-RO-1"' in query
    assert 'dataCenterId: "EUR-IS-1"' in query


def test_aggregate_runpod_availability_rows_merges_duplicate_gpu_ids():
    gpu_rows = [
        {
            "id": "NVIDIA RTX 2000 Ada Generation",
            "displayName": "NVIDIA RTX 2000 Ada Generation",
            "r0": {"availableGpuCounts": [1, 2]},
            "r1": {"availableGpuCounts": []},
        },
        {
            "id": "NVIDIA RTX 2000 Ada Generation",
            "displayName": "NVIDIA RTX 2000 Ada Generation",
            "r0": {"availableGpuCounts": [4]},
            "r1": {"availableGpuCounts": [1]},
        },
    ]
    alias_to_region = {
        "r0": "EU-RO-1",
        "r1": "EUR-IS-1",
    }

    rows = _aggregate_runpod_availability_rows(
        gpu_rows=gpu_rows,
        selected_gpu_types={"RTX2000-Ada"},
        alias_to_region=alias_to_region,
    )

    assert len(rows) == 2

    by_region = {row["provider_region_id"]: row for row in rows}
    ro = by_region["EU-RO-1"]
    is_region = by_region["EUR-IS-1"]

    assert ro["gpu_type"] == "RTX2000-Ada"
    assert ro["available"] is True
    assert ro["available_counts"] == [1, 2, 4]
    assert ro["max_available"] == 4

    assert is_region["gpu_type"] == "RTX2000-Ada"
    assert is_region["available"] is True
    assert is_region["available_counts"] == [1]
    assert is_region["max_available"] == 1


def test_aggregate_runpod_availability_rows_falls_back_when_counts_missing():
    gpu_rows = [
        {
            "id": "NVIDIA A40",
            "displayName": "NVIDIA A40",
            "r0": {
                "stockStatus": "High",
                "availableGpuCounts": [],
                "maxUnreservedGpuCount": 3,
            },
        }
    ]
    alias_to_region = {"r0": "CA-MTL-1"}

    rows = _aggregate_runpod_availability_rows(
        gpu_rows=gpu_rows,
        selected_gpu_types={"A40"},
        alias_to_region=alias_to_region,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["gpu_type"] == "A40"
    assert row["available"] is True
    assert row["available_counts"] == [3]
    assert row["max_available"] == 3
