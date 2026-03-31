from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.runpod_adapter import RunPodAdapter


def test_is_available_true_when_available_counts_non_empty():
    lowest_price = {
        "stockStatus": "OUT_OF_STOCK",
        "availableGpuCounts": [2, 4],
        "maxUnreservedGpuCount": 0,
    }
    assert RunPodAdapter.is_available(lowest_price, gpu_count=1) is True


def test_is_available_false_when_available_counts_empty():
    lowest_price = {
        "stockStatus": "Available",
        "availableGpuCounts": [],
        "maxUnreservedGpuCount": 8,
    }
    assert RunPodAdapter.is_available(lowest_price, gpu_count=1) is False


def test_available_count_uses_max_available_counts():
    lowest_price = {
        "availableGpuCounts": [1, 2, 8],
        "maxUnreservedGpuCount": 1,
    }
    assert RunPodAdapter.available_count(lowest_price) == 8


def test_available_count_zero_when_empty_or_missing():
    assert RunPodAdapter.available_count({"availableGpuCounts": []}) == 0
    assert RunPodAdapter.available_count(None) == 0
