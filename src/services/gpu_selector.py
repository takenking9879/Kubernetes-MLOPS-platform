"""
src/services/gpu_selector.py

Spot-first GPU selector: takes a list of GPUOffers and ResourceConstraints,
then produces a ranked SkyPilot `any_of` list where spot entries always
precede on-demand entries of equivalent cost.

Scoring rules (ascending = cheapest first):
  - Spot entries:     price_spot  * region_bonus * ib_bonus
  - On-demand entries: price_spot * ON_DEMAND_PENALTY * region_bonus * ib_bonus
    (ON_DEMAND_PENALTY ensures on-demand sorts after all spot options)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .gpu_catalog import GPUCatalogService, GPUOffer

# On-demand multiplier — keeps on-demand entries ranked after all spot options
_ON_DEMAND_PENALTY = 2.5
_REGION_BONUS = 0.95      # score multiplier for preferred regions
_IB_BONUS = 0.85          # score multiplier when InfiniBand is required and available


@dataclass
class ResourceConstraints:
    providers: list[str] = field(default_factory=lambda: ["runpod"])
    gpu_types: list[str] | None = None   # None = any type (cheapest wins)
    min_vram_gb: float = 0
    max_price_per_hour: float = 9999
    prefer_spot: bool = True             # Default True — spot-first
    require_infiniband: bool = False
    preferred_regions: list[str] = field(default_factory=list)
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    job_type: str = "tabular"            # "tabular" | "llm"


@dataclass
class GPUSelectResult:
    any_of: list[dict]
    spot_entries: int
    ondemand_entries: int
    estimated_cost_spot: float | None
    estimated_cost_ondemand: float | None


class GPUSelectorService:
    def select_providers(
        self,
        constraints: ResourceConstraints,
        offers: list[GPUOffer] | None = None,
    ) -> GPUSelectResult:
        """
        Return a GPUSelectResult whose `any_of` list is ordered spot-first.

        If `offers` is None, fetches from GPUCatalogService live.
        Pass a pre-fetched list to avoid redundant API calls.
        """
        if offers is None:
            catalog = GPUCatalogService()
            offers = catalog.query_availability(
                providers=constraints.providers,
                min_vram_gb=constraints.min_vram_gb,
                gpu_types=constraints.gpu_types,
            )

        # ── 1. Hard-filter ────────────────────────────────────────────────────
        filtered: list[GPUOffer] = []
        for o in offers:
            if o.provider not in constraints.providers:
                continue
            if o.vram_gb < constraints.min_vram_gb:
                continue
            if constraints.require_infiniband and not o.infiniband:
                continue
            if constraints.gpu_types and o.gpu_type not in constraints.gpu_types:
                continue
            filtered.append(o)

        # ── 2. Deduplicate: keep cheapest on-demand offer per (provider, gpu_type) ─
        best: dict[tuple[str, str], GPUOffer] = {}
        for o in filtered:
            k = (o.skypilot_cloud, o.gpu_type)
            if k not in best or o.price_on_demand < best[k].price_on_demand:
                best[k] = o
        unique: list[GPUOffer] = list(best.values())

        # ── 3. Score and build (spot, on-demand) entry pairs ─────────────────
        scored: list[tuple[float, dict, bool]] = []  # (score, yaml_entry, is_spot)
        for o in unique:
            r_bonus = _REGION_BONUS if o.region in constraints.preferred_regions else 1.0
            ib_bonus = (
                _IB_BONUS if constraints.require_infiniband and o.infiniband else 1.0
            )
            gpus_str = f"{o.gpu_type}:{constraints.num_gpus_per_node}"

            # Spot entry (if spot pricing exists and prefer_spot is True)
            if o.price_spot is not None and constraints.prefer_spot:
                spot_score = o.price_spot * r_bonus * ib_bonus
                if o.price_spot <= constraints.max_price_per_hour:
                    scored.append(
                        (
                            spot_score,
                            {
                                "cloud": o.skypilot_cloud,
                                "accelerators": gpus_str,
                                "use_spot": True,
                            },
                            True,
                        )
                    )

            # On-demand entry
            od_score = o.price_on_demand * _ON_DEMAND_PENALTY * r_bonus * ib_bonus
            if o.price_on_demand <= constraints.max_price_per_hour:
                scored.append(
                    (
                        od_score,
                        {
                            "cloud": o.skypilot_cloud,
                            "accelerators": gpus_str,
                            "use_spot": False,
                        },
                        False,
                    )
                )

        # Sort ascending → spot entries naturally precede on-demand (lower score)
        scored.sort(key=lambda x: x[0])

        any_of = [entry for _, entry, _ in scored]
        spot_entries = sum(1 for _, _, is_spot in scored if is_spot)
        ondemand_entries = sum(1 for _, _, is_spot in scored if not is_spot)

        # Cost estimates: cheapest available spot and on-demand
        spot_offers = [o for o in unique if o.price_spot is not None]
        est_spot = min((o.price_spot for o in spot_offers), default=None)  # type: ignore[type-var]
        est_od = min((o.price_on_demand for o in unique), default=None)

        return GPUSelectResult(
            any_of=any_of,
            spot_entries=spot_entries,
            ondemand_entries=ondemand_entries,
            estimated_cost_spot=est_spot,
            estimated_cost_ondemand=est_od,
        )
