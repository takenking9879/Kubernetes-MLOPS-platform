"""Write /tmp/grafana-agent-config.yaml from env vars.

Called from SkyPilot run blocks before starting grafana-agent:

    python3 k3s/sky/write_grafana_config.py
    grafana-agent --config.file=/tmp/grafana-agent-config.yaml &

Required env vars:
    GRAFANA_REMOTE_WRITE_URL  — Grafana Cloud remote_write endpoint
    GRAFANA_INSTANCE_ID       — Grafana Cloud instance / user ID
    GRAFANA_API_KEY           — Grafana Cloud API key (used as password)

Optional env vars:
    SKY_CLUSTER_NAME          — cluster label on exported metrics
    USE_SPOT                  — is_spot label ("true" / "false")
    GPU_EXPORTER_PORT         — port the gpu_exporter.py listens on (default 9400)
"""
from __future__ import annotations

import os
import sys

url = os.getenv("GRAFANA_REMOTE_WRITE_URL", "")
instance_id = os.getenv("GRAFANA_INSTANCE_ID", "")
api_key = os.getenv("GRAFANA_API_KEY", "")
cluster = os.getenv("SKY_CLUSTER_NAME", "unknown")
is_spot = os.getenv("USE_SPOT", "false")
exporter_port = os.getenv("GPU_EXPORTER_PORT", "9400")

if not url:
    print("[write_grafana_config] GRAFANA_REMOTE_WRITE_URL is empty — skipping", flush=True)
    sys.exit(0)

config = f"""\
metrics:
  global:
    scrape_interval: 15s
    remote_write:
      - url: {url}
        basic_auth:
          username: {instance_id}
          password: {api_key}
  configs:
    - name: gpu
      scrape_configs:
        - job_name: skypilot-gpu
          static_configs:
            - targets:
                - "localhost:{exporter_port}"
              labels:
                cluster: {cluster}
                is_spot: {is_spot}
"""

out = "/tmp/grafana-agent-config.yaml"
with open(out, "w") as fh:
    fh.write(config)

print(f"[write_grafana_config] wrote {out}  (cluster={cluster}  url={url})", flush=True)
