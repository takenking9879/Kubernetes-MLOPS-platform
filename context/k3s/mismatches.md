# k3s/ — Mismatches

- **Two training DAGs (`training_pipeline` vs `training_pipeline_skypilot`)**: Both are ACTIVE. `training_pipeline` uses KubeRay locally; `training_pipeline_skypilot` uses cloud GPU. Backend routes between them via `?skypilot=true`. Any change to training params schema must be reflected in BOTH DAGs.

- **`ml_pipeline` vs `full_ml_pipeline`**: Both chain preprocessing + training. `full_ml_pipeline` is code-level composition; `ml_pipeline` is the UI-triggered version with flexible mode flags. Risk of behavioral divergence over time.

- **SkyPilot YAML routing in two places**: `app/backend/services/job_builder.py` AND `k3s/airflow/dags/sky_runner.py` both contain routing tables mapping `(kind, provider)` → YAML filename. If a new YAML is added, both must be updated.

- **`params.yaml` still present but unused**: `k3s/params.yaml` is listed as "developer reference only" but could confuse future developers who treat it as authoritative. Comment the header clearly or move to `docs/`.

- **Kafka connector polling terminal states**: `serving_dag.py:poll_spark_connector_running` polls for `RUNNING` state; if Spark Operator returns a new state name in a future version, this will silently hang.

- **`sky-runner-pod.yaml` vs DAG runtime spec**: Airflow KubernetesPodOperator does not read `sky-runner-pod.yaml` at runtime; resources/env/command come from DAG arguments. This causes confusion when users tune memory in the YAML but submit/poll pods still use limits defined in `training_dag_skypilot.py`.
