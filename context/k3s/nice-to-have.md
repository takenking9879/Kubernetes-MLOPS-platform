# k3s/ — Nice-to-Have

- **Kafka connector lifecycle DAG** — add `stop_kafka_connector` task (or separate `teardown_kafka_connector` DAG) to cleanly stop streaming connectors on re-deploy
- **Retraining trigger DAG** — watch Prometheus drift alert → trigger `preprocessing_pipeline` + `training_pipeline_skypilot` automatically
- **GPU catalog warm-up** — 5-min scheduled DAG calling `GPUCatalogService.query_availability()` for all providers; keeps cache warm during peak usage
- **Merge `training_pipeline` + `training_pipeline_skypilot`** — unify into single DAG with a `use_skypilot` flag; reduces maintenance burden
- **Delete legacy files** — see `context/k3s/legacy.md`; 8 files safe to delete
- **Normalize DAG state outputs** — add a shared `_normalize_state(raw_state) -> Literal["RUNNING","SUCCEEDED","FAILED"]` in `k8s_helpers.py`; referenced by all polling tasks and backend status endpoints
- **Centralize YAML routing** — move `(kind, provider) → yaml_file` routing table to a single location (e.g., `k3s/sky/manifest.yaml`) so `job_builder.py` and `sky_runner.py` both read from it
