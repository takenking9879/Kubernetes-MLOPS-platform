# GPU Metrics, Grafana Cloud, and Airflow LocalExecutor

## Status

| Feature | Status |
|---------|--------|
| Airflow LocalExecutor | **Already enabled** (`k3s/airflow/airflow_values.yaml` line 1) |
| GPU exporter on SkyPilot VMs | Runs automatically — `k3s/sky/gpu_exporter.py` started in every GPU YAML `run:` block |
| Grafana Cloud remote_write | Optional — set env vars (see below) |
| In-cluster Prometheus scraping | Requires network access to VM external IPs (not available from Docker Desktop cluster) |

---

## GPU Exporter

Every SkyPilot GPU YAML (`ray-gpu-training.yaml`, `ray-gpu-multinode-aws.yaml`, `ray-llm-training.yaml`) starts `k3s/sky/gpu_exporter.py` as a background process before the main training script.

**Metrics exposed on `:9400`:**

| Metric | Labels | Description |
|--------|--------|-------------|
| `skypilot_gpu_utilization_pct` | `gpu`, `cluster`, `is_spot` | GPU utilisation % |
| `skypilot_gpu_memory_used_mb` | `gpu`, `cluster` | GPU memory used MB |
| `skypilot_gpu_memory_total_mb` | `gpu`, `cluster` | GPU total memory MB |

**Label values** are read from env vars injected by Airflow:
- `cluster` ← `SKY_CLUSTER_NAME`
- `is_spot` ← `USE_SPOT` (`"true"` / `"false"`)

---

## Grafana Cloud remote_write (optional)

SkyPilot VMs run on external cloud IPs not reachable from the in-cluster Prometheus. To see GPU metrics in Grafana, use Grafana Cloud's remote_write endpoint (works over HTTPS from anywhere).

### 1 — Create a free Grafana Cloud account

Go to [grafana.com](https://grafana.com) → **Start for free** → note your **instance ID** and **remote_write URL**.

### 2 — Set Airflow Variables

Add these in the Airflow UI under **Admin → Variables** (or as K8s secrets):

| Key | Value |
|-----|-------|
| `GRAFANA_REMOTE_WRITE_URL` | `https://prometheus-prod-XX-prod-XX.grafana.net/api/prom/push` |
| `GRAFANA_INSTANCE_ID` | Your numeric instance ID |
| `GRAFANA_API_KEY` | A Grafana Cloud API key with `metrics:write` scope |
| `SKY_CLUSTER_NAME` | (injected per-run by DAG from the generated cluster name) |

### 3 — Airflow DAG injection

In `training_dag_skypilot.py`, add these to `task.update_envs()`:

```python
task.update_envs({
    ...
    "SKY_CLUSTER_NAME":         cluster_name,
    "GRAFANA_REMOTE_WRITE_URL": Variable.get("GRAFANA_REMOTE_WRITE_URL", default_var=""),
    "GRAFANA_INSTANCE_ID":      Variable.get("GRAFANA_INSTANCE_ID", default_var=""),
    "GRAFANA_API_KEY":          Variable.get("GRAFANA_API_KEY", default_var=""),
})
```

When `GRAFANA_REMOTE_WRITE_URL` is non-empty **and** `grafana-agent` binary is present (installed in `setup:` automatically), `write_grafana_config.py` generates `/tmp/grafana-agent-config.yaml` and grafana-agent ships metrics to Grafana Cloud.

### 4 — In-cluster Prometheus remote_write (for existing Ray/K8s metrics)

To merge in-cluster Ray and Kubernetes metrics into the same Grafana Cloud workspace, add a `remote_write` block to the in-cluster Prometheus ConfigMap:

```yaml
# kubectl edit cm prometheus-server -n monitoring
remote_write:
  - url: "https://prometheus-prod-XX.grafana.net/api/prom/push"
    basic_auth:
      username: "<GRAFANA_INSTANCE_ID>"
      password: "<GRAFANA_API_KEY>"
```

### 5 — Import the dashboard

The dashboard JSON is at `k3s/grafana/dashboards/gpu-training-skypilot.json`.

In Grafana Cloud: **Dashboards → Import → Upload JSON file**.

Panels:
- GPU utilisation % (per GPU, per cluster, spot/on-demand labelled)
- GPU memory used vs total
- Average utilisation stat + memory utilisation %
- Active spot cluster count
- Spot vs on-demand GPU ratio gauge
- Spot vs on-demand utilisation time series (stacked)
- Ray training loss (from in-cluster Ray Prometheus metrics)

---

## Airflow LocalExecutor

`LocalExecutor` is already set in `k3s/airflow/airflow_values.yaml`:

```yaml
executor: "LocalExecutor"
```

This allows **multiple DAGs to run concurrently** (e.g. a preprocessing run and a training run at the same time). PostgreSQL is also already configured in the same file (`postgresql:` block) — LocalExecutor requires a real database (not SQLite).

**Verify after deployment:**

```bash
kubectl exec -n airflow deploy/airflow-scheduler -- \
  airflow config get-value core executor
# Expected: LocalExecutor
```

**Run two DAGs concurrently (smoke test):**

```bash
# Trigger both in quick succession; they should run in parallel in the UI
airflow dags trigger preprocessing_pipeline
airflow dags trigger training_pipeline
```

---

## Cost Tracking

After each tabular training run, `k3s/kuberay/main.py` logs these MLflow tags:

| Tag | Value |
|-----|-------|
| `estimated_cost_usd` | `GPU_PRICE_PER_HOUR × elapsed_hours` |
| `instance_type` | `"spot"` or `"on_demand"` |
| `gpu_price_per_hour` | Raw price from `GPU_PRICE_PER_HOUR` env var |

The Airflow DAG injects `GPU_PRICE_PER_HOUR` from `GPUSelectorService.select_providers()` — the `estimated_cost_spot` field of the winning offer. If `GPU_PRICE_PER_HOUR=0` (default), cost tags are skipped.

LLM training (`k3s/kuberay/llm_main.py`) logs the same tags at job completion.
