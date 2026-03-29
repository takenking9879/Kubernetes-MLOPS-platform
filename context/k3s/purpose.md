# k3s/ — Purpose

### Purpose
Kubernetes orchestration layer. Contains Airflow DAGs, SkyPilot YAMLs, Spark manifests, KubeRay manifests, MLflow config, and infrastructure Helm values. This is where all ML workflow execution is defined.

### When to use
- Adding or modifying an Airflow DAG
- Adding or modifying a SkyPilot job YAML
- Changing K8s resource manifests (SparkApplication, RayJob, RayService)
- Changing infrastructure configuration (Helm values, RBAC)

### When not to use
- API trigger logic → `app/backend/routers/`
- DAG parameter generation → `app/backend/services/job_builder.py`
- ML logic that runs inside the pods → `src/`

### Physical layout
```
k3s/
  airflow/
    dags/               ← 9 active DAGs + 1 empty stub
    k8s_helpers.py      ← K8s resource helpers (idempotent delete, name generation)
    Dockerfile          ← Airflow custom image
    airflow_values.yaml ← Helm values for Airflow
    requirements.txt    ← Python packages for Airflow image
    airflow-rbac.yaml   ← RBAC for Airflow scheduler
  sky/                  ← SkyPilot YAML templates (9 active, 5 legacy)
  spark/
    spark-application.yaml         ← Base SparkApplication (preprocessing)
    preprocess/dsl_001.yaml        ← DSL YAML (reference, overridden per run)
    ingestion/                     ← Ingestion SparkApplication
    inference/                     ← Kafka streaming SparkApplication
  kuberay/
    kuberay-job.yaml               ← Base RayJob (local training)
    serving/rayservice-model-serving.yaml ← RayService (production serving)
    nvidia-*.yaml                  ← NVIDIA device plugin / runtime class
  params.yaml           ← LEGACY: developer reference only, not used at runtime
  config.yaml           ← Static infra config (used by serving config loader)
  mlflow/               ← MLflow K8s deployment
```
