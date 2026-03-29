## 2026-03-29 — Single-pod SkyPilot lifecycle + GPU manual fallback list
- **Single-pod lifecycle**: `training_pipeline_skypilot` and `llm_training_pipeline` DAGs now use a single KubernetesPodOperator (`run-training` / `run-llm` commands) that handles submit → poll → cancel-on-failure in one pod, eliminating cross-pod SkyPilot API server state fragmentation and cold-start overhead. A safety-net cleanup task (trigger_rule=ONE_FAILED) still tears down the jobs-controller VM if Airflow kills the main pod.
- **New sky_runner.py commands**: `run-training`, `run-llm`. Legacy split-phase commands kept for debugging.
- **GPU fallback list** (`gpu_fallbacks`): new `GPUFallbackEntry[]` field on `ResourceConstraints`; when set, `_load_task()` injects entries directly into SkyPilot `any_of`, bypassing the catalog auto-selector.
- **GPUResourceSelector**: added mode toggle (Auto / Manual). Manual mode shows an ordered list of GPU options with infra + accelerator + spot toggle + up/down reorder arrows. First entry = highest priority.
- Updated context: `context/k3s/key_elements.md`, `context/app/front/key_elements.md`.

## 2026-03-29 — SkyPilot RunPod controller and credential bootstrap hardening
- Updated SkyPilot DAG KubernetesPodOperator tasks to run image ENTRYPOINT and pass runner command via task arguments.
- Increased `training_pipeline_skypilot` submit/poll pod resources and made them tunable (`SKY_SUBMIT_*`, `SKY_POLL_*`).
- Added runtime safeguard in `k3s/sky/docker-entrypoint.sh` to merge `~/.sky/config.yaml` and force managed jobs controller `disk_size` to a RunPod-compatible value (default 30 GB, max 40 GB).
- Added Vast credential alias support (`VAST_API_KEY` and `VASTAI_API_KEY`) in runner entrypoint and `src/services/gpu_catalog.py`.
- Updated context docs: `context/k3s/key_elements.md` and `context/k3s/mismatches.md`.

## 2026-03-29 — SkyPilot polling stability in Airflow KPO pods
- Updated `k3s/sky/sky_runner.py` polling and launch-visibility checks to use `sky.jobs.queue(version=2)`.
- Added normalization for queue v2 records and handled "No in-progress managed jobs." as an empty queue state.
- Enabled `all_users=True` in polling path to avoid user-hash visibility mismatches between submit/poll pods with non-shared `~/.sky` state.

## 2026-03-29 — Auto teardown of jobs controller on DAG failure
- Added `cleanup_skypilot_controller_on_failure` task to `training_pipeline_skypilot` with `TriggerRule.ONE_FAILED`.
- Added `cleanup-failed-training-controller` command in `k3s/sky/sky_runner.py` to cancel the failed managed job (best effort), check for remaining active managed jobs, and tear down `sky-jobs-controller-*` only when safe.
