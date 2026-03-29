## 2026-03-29 — SkyPilot RunPod controller and credential bootstrap hardening
- Updated SkyPilot DAG KubernetesPodOperator tasks to run image ENTRYPOINT and pass runner command via task arguments.
- Increased `training_pipeline_skypilot` submit/poll pod resources and made them tunable (`SKY_SUBMIT_*`, `SKY_POLL_*`).
- Added runtime safeguard in `k3s/sky/docker-entrypoint.sh` to merge `~/.sky/config.yaml` and force managed jobs controller `disk_size` to a RunPod-compatible value (default 30 GB, max 40 GB).
- Added Vast credential alias support (`VAST_API_KEY` and `VASTAI_API_KEY`) in runner entrypoint and `src/services/gpu_catalog.py`.
- Updated context docs: `context/k3s/key_elements.md` and `context/k3s/mismatches.md`.
