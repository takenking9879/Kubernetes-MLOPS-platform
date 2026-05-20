# Observability

## Status
- `working`: in-cluster metrics scraping and dashboards
- `partial`: cross-boundary external worker observability remains an active area

## Design
- Prometheus stack scrapes Spark, Ray, Kafka, kube-state, node-exporter, and SkyPilot API metrics.
- Grafana is provisioned with Prometheus datasource and dashboard config maps.
- Training and inference code paths expose custom metrics endpoints.

![Observability Path](./images/observability-path.png)

## External Worker Consideration
SkyPilot workers outside cluster boundaries are harder to scrape directly; outbound metrics shipping patterns are used/planned for centralized visibility.

## Trade-Offs
- Single-cluster scraping is simpler, but hybrid local+external workloads require additional remote-write and access controls.

## Evidence Pointers
- `k3s/monitoring/prometheus/prometheus-stack.yaml`
- `k3s/monitoring/grafana/grafana.yaml`
- `k3s/monitoring/grafana/dashboards/*`
- `k3s/spark/inference/kafka_main.py`
- `k3s/kuberay/main.py`
