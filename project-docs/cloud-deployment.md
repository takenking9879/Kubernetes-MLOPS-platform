# Cloud Deployment Direction

## Status
- `partial/planned`: provider paths exist in code and templates, but full cloud production posture is still evolving

## Current Direction
- Airflow + SkyPilot are used to route jobs to local or external providers.
- GPU catalog and selector services include RunPod/Vast/AWS-aware logic.
- Provider-specific YAML templates are maintained for training and serving variants.

## Gaps
- Full infra bootstrap automation for persistent cloud environments.
- Standardized production deployment guardrails and SRE workflows.

## Trade-Offs
- Multi-provider optionality reduces lock-in risk.
- Provider differences increase testing matrix and operational complexity.

## Evidence Pointers
- `app/backend/services/job_builder.py`
- `app/backend/services/orchestration_selector.py`
- `src/services/gpu_catalog.py`
- `k3s/sky/*.yaml`
