# src/services/ — Routing

- Add GPU provider → `gpu_catalog.py:_query_{provider}()` + add to `query_availability()` routing
- Fix GPU name mapping (RunPod/Vast → SkyPilot canonical) → `gpu_catalog.py:_RUNPOD_TO_SKYPILOT` dict or `_infer_skypilot_id()`
- Fix IB detection for a provider → `gpu_catalog.py:_RUNPOD_INFINIBAND_IDS` or Vast `inet_up` logic
- Change spot/on-demand scoring weights → `gpu_selector.py` constants (`_ON_DEMAND_PENALTY`, `_REGION_BONUS`, `_IB_BONUS`)
- Change `infra` path format for a provider → `gpu_selector.py:_build_infra()`
- Fix Vast spot pricing assumption → `gpu_catalog.py:_query_vast()` (always `price_spot=None, spot_available=False`)
- Add new SkyPilot clouds to catalog subprocess → `_sky_catalog_query.py:VALID_CLOUDS`
- Extend SkyPilot catalog TTL → `gpu_catalog.py:_sky_catalog_cache` TTL constant
- Change `skypilot_supported` validation logic → `gpu_catalog.py:_sky_supported_names()` + provider query methods
