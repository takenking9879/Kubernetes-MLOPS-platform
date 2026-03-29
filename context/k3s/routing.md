# k3s/ — Routing

## If the task is...

- Add a new Airflow DAG → `k3s/airflow/dags/{dag_id}.py`; use PythonOperator for Spark/Ray, KubernetesPodOperator for SkyPilot; add backend trigger endpoint
- Change preprocessing Spark job → `k3s/spark/spark-application.yaml` + `k3s/airflow/dags/preprocessing_dag.py`
- Change local training (KubeRay) → `k3s/kuberay/kuberay-job.yaml` + `k3s/airflow/dags/training_dag.py`
- Change SkyPilot GPU training → `k3s/sky/ray-gpu-training-{provider}.yaml` + `k3s/airflow/dags/training_dag_skypilot.py`
- Change LLM training → `k3s/sky/ray-llm-training-{provider}.yaml` + `k3s/airflow/dags/llm_training_dag.py`
- Change serving deployment (Ray Serve) → `k3s/airflow/dags/serving_dag.py` + `k3s/kuberay/serving/rayservice-model-serving.yaml`
- Change vLLM serving → `k3s/sky/vllm-serving-{provider}.yaml` or `ray-vllm-multinode-serving.yaml` + `k3s/airflow/dags/ray_vllm_serving_dag.py`
- Change MLflow promotion logic → `k3s/airflow/dags/serving_dag.py:promote_to_champion` or `promotion_dag.py`
- Change K8s resource cleanup → `k3s/airflow/k8s_helpers.py`
- Add SkyPilot YAML → `k3s/sky/` (follow naming: `ray-{purpose}-{provider}.yaml`) + update routing table in `job_builder.py` + `sky_runner.py`
- Change Airflow Helm config → `k3s/airflow/airflow_values.yaml`
- Change Airflow RBAC → `k3s/airflow/airflow-rbac.yaml`
- Change Airflow Python deps → `k3s/airflow/requirements.txt` + `k3s/airflow/Dockerfile`
