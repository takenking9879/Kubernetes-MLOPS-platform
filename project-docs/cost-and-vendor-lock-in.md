# Cost And Vendor Lock-In

## Status
- `working`: architecture includes local-first + multi-provider options
- `planned`: deeper automated cost optimization and placement policies

## Positioning
- Run small/iterative workloads locally where possible.
- Burst heavier workloads to external providers when needed.
- Keep orchestration/provider interfaces explicit to avoid single-vendor coupling.

## Decisions
- Kubernetes portability + SkyPilot provider abstraction are core anti-lock-in mechanisms.
- S3-based paths are currently primary; alternative object-storage direction remains future work.

## Trade-Offs
- Portability introduces extra abstraction layers and integration testing overhead.
- Cross-provider data movement can create egress and latency penalties.

## Evidence Pointers
- `app/backend/services/orchestration_selector.py`
- `app/backend/services/job_builder.py`
- `src/services/gpu_catalog.py`
- `src/services/gpu_selector.py`
- `k3s/config.yaml`
