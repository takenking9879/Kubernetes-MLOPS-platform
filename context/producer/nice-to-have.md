# producer/ — Nice-to-Have

- **K8s CronJob or Deployment** for the producer so it can run continuously in-cluster
- **CLI args** for trend (`--trend normal|data_drift|concept_drift`) and rate (`--rate 100`)
- **Auto-drift scheduler** — automatically switch trend after N messages; useful for end-to-end drift detection testing
- **Prometheus metrics** — publish `messages_sent_total`, `messages_failed_total` for producer health monitoring
