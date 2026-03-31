"""
src/services/runpod_adapter.py

Adapter that normalises RunPod GraphQL `lowestPrice` payloads into the
fields expected by GPUOffer. All methods are pure functions (no I/O).

A GPU is AVAILABLE for a given gpu_count only if ALL four conditions hold:
  1. lowestPrice exists (not None)
  2. stockStatus is "Available" or "Low"
  3. maxUnreservedGpuCount >= gpu_count
  4. gpu_count is listed in availableGpuCounts

AWS-backed GPUs returned by RunPod's API may appear out-of-stock due to
RunPod marketplace limitations — not real AWS capacity. The separate
_query_aws() path is authoritative for AWS; RunPod data must not override it.
"""
from __future__ import annotations

_AVAILABLE_STATUSES: frozenset[str] = frozenset({"Available", "Low"})


class RunPodAdapter:
    """Stateless adapter: converts raw RunPod API dicts → GPUOffer fields."""

    @staticmethod
    def is_available(lowest_price: dict | None, gpu_count: int) -> bool:
        """
        Return True only when ALL four availability conditions are satisfied:
          1. lowestPrice exists
          2. stockStatus is "Available" or "Low"
          3. maxUnreservedGpuCount >= gpu_count
          4. gpu_count is in availableGpuCounts
        """
        if not lowest_price:
            return False
        if (lowest_price.get("stockStatus") or "") not in _AVAILABLE_STATUSES:
            return False
        max_unreserved = lowest_price.get("maxUnreservedGpuCount")
        if max_unreserved is None or max_unreserved < gpu_count:
            return False
        available_counts = lowest_price.get("availableGpuCounts") or []
        return gpu_count in available_counts

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
        Return maxUnreservedGpuCount from lowestPrice, or 0 if absent.
        0 means confirmed no stock right now for RunPod.
        """
        if not lowest_price:
            return 0
        val = lowest_price.get("maxUnreservedGpuCount")
        return int(val) if val is not None else 0
