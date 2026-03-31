"""
src/services/runpod_adapter.py

Adapter that normalises RunPod GraphQL `lowestPrice` payloads into the
fields expected by GPUOffer. All methods are pure functions (no I/O).

A GPU is AVAILABLE when RunPod reports at least one available quantity
for that GPU in `availableGpuCounts`.

AWS-backed GPUs returned by RunPod's API may appear out-of-stock due to
RunPod marketplace limitations — not real AWS capacity. The separate
_query_aws() path is authoritative for AWS; RunPod data must not override it.
"""
from __future__ import annotations

class RunPodAdapter:
    """Stateless adapter: converts raw RunPod API dicts → GPUOffer fields."""

    @staticmethod
    def is_available(lowest_price: dict | None, gpu_count: int) -> bool:
        """
        Return True when RunPod reports at least one available quantity.

        User-facing rule for this project:
          - no stock only when availableGpuCounts == []
        """
        if not lowest_price:
            return False
        available_counts = lowest_price.get("availableGpuCounts") or []
        return len(available_counts) > 0

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
        Return the largest available quantity from availableGpuCounts.
        0 means confirmed no stock right now for RunPod.
        """
        if not lowest_price:
            return 0
        counts = lowest_price.get("availableGpuCounts") or []
        if not counts:
            return 0
        try:
            return int(max(counts))
        except Exception:
            return 0
