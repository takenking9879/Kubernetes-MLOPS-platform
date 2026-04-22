# Multi-Cloud YAML Strategy

## Current State

SkyPilot training YAMLs are provider-specific:

| File | Provider | Resources |
|------|----------|-----------|
| `ray-gpu-training-runpod.yaml` | RunPod | `ordered:` list with `image_id` |
| `ray-gpu-training-vast.yaml` | Vast.ai | `ordered:` list |
| `ray-gpu-training-local.yaml` | Local Docker (SSH Node Pool) | `infra: ssh/local-gpu` |
| `ray-gpu-multinode-aws.yaml` | AWS multi-node | `ordered:` list |

Routing: `training_dag_skypilot.py → _yaml_map[(kind, provider)]`

## Why Provider-Specific

Each provider has different base images and `setup:` cost:
- **Local**: `--system-site-packages` venv (~30 s); torch+CUDA+Ray already in image
- **RunPod/Vast**: full pip install from base image (~5-8 min)

The `run:` section is **identical** across all providers.

## Planned Unification

Long-term: shared `run:` via YAML anchors. `setup:` stays provider-specific.
