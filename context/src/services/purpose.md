# src/services/ — Purpose

### Purpose
GPU orchestration layer. Aggregates real-time GPU offers from RunPod, Vast.ai, and AWS; validates SkyPilot support; ranks offers cost-optimally for SkyPilot job submission.

### When to use
- Adding or modifying a GPU provider
- Fixing GPU pricing data (spot, on-demand)
- Changing SkyPilot `any_of` generation logic
- Debugging `skypilot_supported` flag

### When not to use
- API endpoint for GPU data → `app/backend/routers/gpu_resources.py`
- SkyPilot YAML content → `k3s/sky/`

### Physical layout
```
src/services/
  gpu_catalog.py          ← GPUCatalogService: query RunPod + Vast + AWS; GPUOffer dataclass
  gpu_selector.py         ← GPUSelectorService: rank offers, build SkyPilot any_of list
  _sky_catalog_query.py   ← Subprocess helper: runs inside /opt/sky-venv; outputs JSON catalog
```
