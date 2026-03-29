# src/services/ — Key Elements

## GPUOffer (dataclass)
File: `src/services/gpu_catalog.py`

Fields:
- `provider`: "runpod" | "vast" | "aws"
- `gpu_type`: str — SkyPilot canonical name if `skypilot_supported=True`; native provider name otherwise
- `gpu_count`: int
- `vram_gb`: float
- `vcpus`: int (0 if provider doesn't expose)
- `ram_gb`: float
- `price_on_demand`: float
- `price_spot`: float | None (None for Vast.ai — no spot pricing)
- `spot_available`: bool (False for Vast.ai)
- `available_count`: int
- `region`: str (empty for RunPod — API doesn't expose per-GPU datacenter)
- `infiniband`: bool
- `skypilot_supported`: bool — True if SkyPilot catalog confirms launchability
- `skypilot_accelerator`: str (empty if `skypilot_supported=False`)
- `skypilot_cloud`: str ("runpod" | "vast" | "aws")

---

## GPUCatalogService
Function/Class: `GPUCatalogService`
File: `src/services/gpu_catalog.py`

Does:
- `query_availability(providers, min_vram_gb, gpu_types)` → `List[GPUOffer]`
- Calls `_get_sky_catalog()` once and passes to all provider methods
- 60-second TTL cache per provider; 5-minute TTL for SkyPilot catalog

Provider methods:
- `_query_runpod(sky_catalog)` — GraphQL API; includes ALL GPUs (skypilot_supported=False for unmapped)
- `_query_vast(sky_catalog)` — REST API; all GPUs; spot=None (Vast has no spot)
- `_query_aws(sky_catalog)` — SkyPilot catalog + boto3 `describe_spot_price_history`; always skypilot_supported=True

SkyPilot catalog retrieval (`_get_sky_catalog()`):
1. Subprocess: `/opt/sky-venv/bin/python _sky_catalog_query.py` → JSON
2. Direct import: `sky.list_accelerators(all=True, clouds=["aws","runpod","vast"])` (dev fallback)
3. Empty dict (static AWS fallback)

Helper: `_sky_supported_names(catalog, cloud)` → `set[str]` of GPU names SkyPilot supports for a cloud

InfiniBand detection:
- RunPod: `_RUNPOD_INFINIBAND_IDS` frozenset (A100-SXM4, H100, H200, B200)
- Vast: `inet_up ≥ 25 Gbps`
- AWS: instance type prefix (p4d, p4de, p3dn, p5, p6, trn1n)

Vast spot pricing: **always None / False** — confirmed no spot pricing in SkyPilot catalog for Vast.

RunPod region: **always empty** — GraphQL API doesn't expose per-GPU datacenter availability.

---

## GPUSelectorService
Function/Class: `GPUSelectorService`
File: `src/services/gpu_selector.py`

Does:
- `select_providers(constraints, offers)` → `GPUSelectResult`
- Hard-filter: provider, skypilot_supported=True, vram_gb, infiniband, gpu_types
- Deduplicate: keep cheapest on-demand per (skypilot_cloud, gpu_type)
- Score: `spot * r_bonus * ib_bonus` vs `on_demand * 2.5 * r_bonus * ib_bonus`
- Sort ascending → spot entries naturally precede on-demand

Constants: `_ON_DEMAND_PENALTY = 2.5`, `_REGION_BONUS = 0.95`, `_IB_BONUS = 0.85`

`_build_infra(skypilot_cloud, region, preferred_regions)` → str:
- AWS: `"aws/{region}"` if region known, else `"aws"`
- RunPod: `"runpod/{region_code}/{zone}"` if user supplies zone in preferred_regions (e.g., CA-MTL-1 → runpod/CA/CA-MTL-1); else `"runpod"`
- Vast: always `"vast"` (SkyPilot doesn't support region for Vast)

Valid RunPod region codes: CA, US, CZ, IS, NL, SE, RO, NO

---

## _sky_catalog_query.py
File: `src/services/_sky_catalog_query.py`

Subprocess helper executed inside `/opt/sky-venv/bin/python`.
Output: JSON to stdout `{gpu_name: [{cloud, region, price, spot_price, accelerator_count, device_memory, cpu_count, memory, instance_type}]}`
Uses `sky.list_accelerators(gpus_only=True, all=True, clouds=["aws","runpod","vast"])`.
`all=True` — includes both COMMON and OTHER GPUs (RTX4090, RTXA4000, MI300X, etc.).
