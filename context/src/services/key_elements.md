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
  - `-1` = not tracked / unlimited (AWS live query and static fallback)
  - `0` = confirmed no stock right now (RunPod: `maxUnreservedGpuCount=0` or `lowestPrice` null)
  - `>0` = `maxUnreservedGpuCount` from RunPod `lowestPrice` (real-time); Vast = `1` (offer present)
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
- `_query_runpod(sky_catalog)` — GraphQL API including `lowestPrice(input: {gpuCount:1, secureCloud:true, minVcpuCount:2, minMemoryInGb:8})`. Delegates availability, pricing, and count to `RunPodAdapter`. `spot_available` reflects real-time stock (all 4 conditions), not just price presence.
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

## RunPodAdapter
File: `src/services/runpod_adapter.py`

Stateless adapter that normalises RunPod GraphQL `lowestPrice` payloads. Pure functions, no I/O.

Methods:
- `is_available(lowest_price, gpu_count) -> bool`
  Returns `True` only when ALL four conditions hold:
  `lowestPrice` not null, `stockStatus` ∈ `{"Available","Low"}`,
  `maxUnreservedGpuCount >= gpu_count`, `gpu_count` ∈ `availableGpuCounts`.
- `extract_price(lowest_price, gpu) -> tuple[float, float | None]`
  Returns `(on_demand, spot)`. Prefers `lowestPrice.uninterruptablePrice` / `minimumBidPrice`;
  falls back to GPU-level `securePrice` / `communityPrice` / spot fields.
- `available_count(lowest_price) -> int`
  Returns `maxUnreservedGpuCount` or `0` if absent.

Used by: `gpu_catalog.py:_query_runpod()`

AWS isolation: RunPod's `lowestPrice` for AWS-backed GPUs may return null (RunPod doesn't have real AWS inventory). This correctly produces `available_count=0` for those RunPod-marketplace rows. The separate `_query_aws()` path is authoritative for AWS and is unaffected.

---

## _sky_catalog_query.py
File: `src/services/_sky_catalog_query.py`

Subprocess helper executed inside `/opt/sky-venv/bin/python`.
Output: JSON to stdout `{gpu_name: [{cloud, region, price, spot_price, accelerator_count, device_memory, cpu_count, memory, instance_type}]}`
Uses `sky.list_accelerators(gpus_only=True, all=True, clouds=["aws","runpod","vast"])`.
`all=True` — includes both COMMON and OTHER GPUs (RTX4090, RTXA4000, MI300X, etc.).

---

## SkyPilot API Server Architecture (as of 2026-03-30)

**Deployment**: Helm chart (`skypilot/skypilot --version 0.12.0`) in `skypilot` namespace.
**Values file**: `k3s/sky/helm/values.yaml`
**Deployment steps**: `k3s/sky/skypilot_steps.txt`
**Image**: `takenking9879/sky-runner:0.12.0` (custom image with `docker-entrypoint.sh` for credential setup)

**Why**: previously each Airflow KPO pod started its own ephemeral local API server.
A single transient SSH failure at the first `refresh=True` poll caused all subsequent
polls to use stale cached STARTING state, and the job appeared stuck indefinitely.
The persistent API server with consolidation mode eliminates SSH-based polling entirely.

**Consolidation mode** (`jobs.controller.consolidation_mode: true`):
- Jobs-controller thread runs inside the API server pod (in-process)
- No separate cloud controller VM is provisioned
- `sky.jobs.queue(refresh=True)` is a local in-process query — no SSH, no `returncode: 255`

**Client connection** (all sky-runner KPO pods):
- `SKYPILOT_API_SERVER_ENDPOINT=http://skypilot:<password>@skypilot-api-server.skypilot.svc.cluster.local`
- Added to `env-secret` in `airflow` namespace; all KPO pods inherit it automatically via `env_from=_aws_env_from()`
- No code changes to DAGs or `sky_runner.py` — SDK picks up the env var transparently

**Prometheus metrics**: `http://skypilot-api-server.skypilot.svc.cluster.local:46580/api/metrics`
Scraped by existing Prometheus in `monitoring` namespace (scrape job in `prometheus-stack.yaml`).

**Cloud credentials for API server**: `sky-cloud-credentials` secret in `skypilot` namespace
(same values as `env-secret` in `airflow` namespace, created separately).
