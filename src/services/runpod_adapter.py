"""
src/services/runpod_adapter.py

Adapter that normalises RunPod GraphQL `lowestPrice` payloads into the
fields expected by GPUOffer. All methods are pure functions (no I/O).

A GPU is AVAILABLE when any of these indicate capacity:
    1) availableGpuCounts has entries,
    2) maxUnreservedGpuCount > 0,
    3) stockStatus reports an available state.

AWS-backed GPUs returned by RunPod's API may appear out-of-stock due to
RunPod marketplace limitations — not real AWS capacity. The separate
_query_aws() path is authoritative for AWS; RunPod data must not override it.
"""
from __future__ import annotations

class RunPodAdapter:
    """Stateless adapter: converts raw RunPod API dicts → GPUOffer fields."""

    _AVAILABLE_STOCK_STATUSES = {
        "available",
        "high",
        "medium",
        "low",
        "in_stock",
        "limited",
    }
    _UNAVAILABLE_STOCK_STATUSES = {
        "out_of_stock",
        "unavailable",
        "none",
    }

    @staticmethod
    def _to_non_negative_int(value: object) -> int:
        try:
            parsed = int(value)
        except Exception:
            return 0
        return parsed if parsed > 0 else 0

    @staticmethod
    def _extract_available_counts(lowest_price: dict | None) -> list[int]:
        if not lowest_price:
            return []
        raw_counts = lowest_price.get("availableGpuCounts") or []
        counts: list[int] = []
        for value in raw_counts:
            parsed = RunPodAdapter._to_non_negative_int(value)
            if parsed > 0:
                counts.append(parsed)
        return sorted(set(counts))

    @staticmethod
    def _is_stock_status_available(lowest_price: dict | None) -> bool:
        if not lowest_price:
            return False
        status = str(lowest_price.get("stockStatus") or "").strip().lower()
        status = status.replace(" ", "_")
        if not status:
            return False
        if status in RunPodAdapter._UNAVAILABLE_STOCK_STATUSES:
            return False
        if status in RunPodAdapter._AVAILABLE_STOCK_STATUSES:
            return True
        return False

    @staticmethod
    def is_available(lowest_price: dict | None, gpu_count: int) -> bool:
        """
        Return True when RunPod signals capacity through counts, max-unreserved
        or stockStatus.
        """
        if not lowest_price:
            return False

        counts = RunPodAdapter._extract_available_counts(lowest_price)
        if counts:
            return True

        max_unreserved = RunPodAdapter._to_non_negative_int(
            lowest_price.get("maxUnreservedGpuCount")
        )
        needed = gpu_count if gpu_count > 0 else 1
        if max_unreserved >= needed:
            return True

        return RunPodAdapter._is_stock_status_available(lowest_price)

    @staticmethod
    def extract_price(
        lowest_price: dict | None,
        gpu: dict,
    ) -> tuple[float, float | None]:
        """
        Return (on_demand, spot) $/hr per GPU.

        Priority:
          on_demand: lowestPrice.uninterruptablePrice → gpu.securePrice → gpu.communityPrice → 0.0
          spot:      lowestPrice.minimumBidPrice → gpu.communitySpotPrice → gpu.secureSpotPrice → None
        """
        lp = lowest_price or {}
        od_raw = (
            lp.get("uninterruptablePrice")
            or gpu.get("securePrice")
            or gpu.get("communityPrice")
        )
        spot_raw = (
            lp.get("minimumBidPrice")
            or gpu.get("communitySpotPrice")
            or gpu.get("secureSpotPrice")
        )
        return (
            float(od_raw) if od_raw is not None else 0.0,
            float(spot_raw) if spot_raw is not None else None,
        )

    @staticmethod
    def available_count(lowest_price: dict | None) -> int:
        """
        Return an estimated available quantity for RunPod.

        Priority:
          1) max(availableGpuCounts)
          2) maxUnreservedGpuCount
          3) 1 if stockStatus says available but count is not provided
        """
        if not lowest_price:
            return 0

        counts = RunPodAdapter._extract_available_counts(lowest_price)
        max_from_counts = int(max(counts)) if counts else 0

        max_unreserved = RunPodAdapter._to_non_negative_int(
            lowest_price.get("maxUnreservedGpuCount")
        )
        max_available = max(max_from_counts, max_unreserved)
        if max_available > 0:
            return max_available

        if RunPodAdapter._is_stock_status_available(lowest_price):
            return 1

        return 0
