"""
src/services/gpu_selector.py

Spot-first GPU selector: takes a list of GPUOffers and ResourceConstraints,
then produces a ranked SkyPilot `any_of` list where spot entries always
precede on-demand entries of equivalent cost.

Scoring rules (ascending = cheapest first):
  - Spot entries:      price_spot  * region_bonus * ib_bonus
  - On-demand entries: price_on_demand * ON_DEMAND_PENALTY * region_bonus * ib_bonus
    (ON_DEMAND_PENALTY ensures on-demand sorts after all spot options)

Notes on SkyPilot field names:
  - `infra` is the recommended field (replaces deprecated `cloud`).
    Schema: <cloud>/<region>/<zone> or k8s/<context>.
  - Region in infra path is provider-dependent:
      AWS:    infra: "aws/us-east-1"    (SkyPilot respects region)
      RunPod: infra: "runpod/CA/CA-MTL-1"  (only if user specifies a zone)
      Vast:   infra: "vast"             (SkyPilot does not support region for Vast)
  - Vast.ai has no spot pricing; only on-demand entries are generated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .gpu_catalog import GPUCatalogService, GPUOffer

# On-demand multiplier — keeps on-demand entries ranked after all spot options
_ON_DEMAND_PENALTY = 2.5
_REGION_BONUS = 0.95      # score multiplier for preferred regions
_IB_BONUS = 0.85          # score multiplier when InfiniBand is required and available

# Valid RunPod region codes (2-letter country prefix of zone names)
_RUNPOD_REGION_CODES: frozenset[str] = frozenset(
    {"CA", "US", "CZ", "IS", "NL", "SE", "RO", "NO"}
)


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


def _build_infra(
    skypilot_cloud: str,
    region: str,
    preferred_regions: list[str],
) -> str:
    """
    Build the SkyPilot `infra` path for an any_of entry.

    Provider-specific behaviour:
    - AWS:    "aws/us-east-1" when region is known; "aws" otherwise.
    - RunPod: "runpod/CA/CA-MTL-1" when the user specifies a valid RunPod zone
              in preferred_regions (zone.split("-")[0] gives the region code).
              Falls back to "runpod" when no zone matches.
    - Vast:   Always "vast" — SkyPilot does not support region for Vast.
    """
    if skypilot_cloud == "aws":
        if region and region not in ("aws", ""):
            return f"aws/{region}"
        return "aws"
    elif skypilot_cloud == "runpod":
        # Try to match a user-supplied RunPod zone (e.g. "CA-MTL-1", "US-TX-3")
        for zone in (preferred_regions or []):
            zone = zone.strip().upper()
            runpod_region = zone.split("-")[0]
            if runpod_region in _RUNPOD_REGION_CODES:
                return f"runpod/{runpod_region}/{zone}"
        return "runpod"
    else:
        # Vast.ai — region not supported in infra path
        return "vast"


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
            if not o.skypilot_supported:
                continue  # only SkyPilot-launchable GPUs in any_of
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
            infra = _build_infra(o.skypilot_cloud, o.region, constraints.preferred_regions)

            # Spot entry — only when spot is available AND user prefers spot
            if o.spot_available and o.price_spot is not None and constraints.prefer_spot:
                spot_score = o.price_spot * r_bonus * ib_bonus
                if o.price_spot <= constraints.max_price_per_hour:
                    scored.append(
                        (
                            spot_score,
                            {
                                "infra": infra,
                                "accelerators": gpus_str,
                                "use_spot": True,
                            },
                            True,
                        )
                    )

            # On-demand entry — always included
            od_score = o.price_on_demand * _ON_DEMAND_PENALTY * r_bonus * ib_bonus
            if o.price_on_demand <= constraints.max_price_per_hour:
                scored.append(
                    (
                        od_score,
                        {
                            "infra": infra,
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
        spot_offers = [o for o in unique if o.price_spot is not None and o.spot_available]
        est_spot = min((o.price_spot for o in spot_offers), default=None)  # type: ignore[type-var]
        est_od = min((o.price_on_demand for o in unique), default=None)

        return GPUSelectResult(
            any_of=any_of,
            spot_entries=spot_entries,
            ondemand_entries=ondemand_entries,
            estimated_cost_spot=est_spot,
            estimated_cost_ondemand=est_od,
        )
