# k3s/ — Missing Features

- **Spark Kafka connector lifecycle management** — `serving_pipeline` starts the connector but never stops it on re-deploy or cleanup. Need a `stop_kafka_connector` task or separate cleanup DAG.
- **Retraining trigger DAG** — no DAG that responds to drift alerts (Prometheus) and automatically re-triggers `preprocessing_pipeline` + `training_pipeline_skypilot`
- **GPU catalog pre-warm DAG** — no scheduled DAG to call `GPUCatalogService.query_availability()` and warm the cache before user requests
- **Cost tracking / billing DAG** — no DAG that reports GPU cost per job run to a budget tracker
- **Serving config history** — no DAG or mechanism to archive past serving configs from S3
- **DAG log export** — no mechanism to export Airflow task logs to a persistent store (e.g., S3) for post-mortem analysis
