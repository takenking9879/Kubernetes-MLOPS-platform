"""Generate /tmp/grafana-agent-config.yaml for outbound metrics shipping.

Called from SkyPilot run blocks before starting grafana-agent:

  python3 k3s/sky/write_grafana_config.py
  grafana-agent --config.file=/tmp/grafana-agent-config.yaml &

Preferred env vars:
  METRICS_REMOTE_WRITE_URL
  METRICS_REMOTE_WRITE_USERNAME
  METRICS_REMOTE_WRITE_PASSWORD

Legacy aliases (still supported):
  GRAFANA_REMOTE_WRITE_URL
  GRAFANA_INSTANCE_ID
  GRAFANA_API_KEY

Optional labels/envs:
  SKY_CLUSTER_NAME
  SKY_PROVIDER
  TRAIN_RUN_ID
  PREPROCESS_RUN_ID
  MODEL_TYPE
  METRICS_SOURCE
  USE_SPOT
  PROMETHEUS_PORT     (default 8002)
  GPU_EXPORTER_PORT   (default 9400)
"""
from __future__ import annotations

import os
import sys

import yaml


def _env(primary: str, legacy: str) -> str:
    return (os.getenv(primary) or os.getenv(legacy) or "").strip()


url = _env("METRICS_REMOTE_WRITE_URL", "GRAFANA_REMOTE_WRITE_URL")
username = _env("METRICS_REMOTE_WRITE_USERNAME", "GRAFANA_INSTANCE_ID")
password = _env("METRICS_REMOTE_WRITE_PASSWORD", "GRAFANA_API_KEY")

cluster = os.getenv("SKY_CLUSTER_NAME", "unknown").strip()
provider = os.getenv("SKY_PROVIDER", "unknown").strip()
train_run_id = os.getenv("TRAIN_RUN_ID", "").strip()
preprocess_run_id = os.getenv("PREPROCESS_RUN_ID", "").strip()
model_type = os.getenv("MODEL_TYPE", "unknown").strip()
metrics_source = os.getenv("METRICS_SOURCE", "skypilot_vm").strip()
is_spot = os.getenv("USE_SPOT", "false").strip()

training_exporter_port = os.getenv("PROMETHEUS_PORT", "8002").strip()
gpu_exporter_port = os.getenv("GPU_EXPORTER_PORT", "9400").strip()

if not url:
    print("[write_grafana_config] remote_write URL is empty - skipping", flush=True)
    sys.exit(0)

# Reachability pre-check: fail fast with a clear message instead of letting
# Grafana Agent silently retry every 15 s. Non-blocking — we still write the
# config file so Agent starts; the warning tells ops to fix
# METRICS_REMOTE_WRITE_URL.
import socket as _socket
import urllib.parse as _urlparse
_parsed = _urlparse.urlparse(url)
_host = _parsed.hostname
_scheme = (_parsed.scheme or "http").lower()
_default_port = 443 if _scheme == "https" else 80
_port = _parsed.port or _default_port

if not _host:
    print(
        "[write_grafana_config] WARNING: METRICS_REMOTE_WRITE_URL has no hostname. "
        "Metrics will NOT be shipped until this URL is fixed.",
        flush=True,
    )
else:
    try:
        with _socket.create_connection((_host, _port), timeout=3):
            pass
    except (_socket.gaierror, OSError, ValueError) as _dns_err:
        print(
            f"[write_grafana_config] WARNING: remote_write URL unreachable: {_dns_err}. "
            f"Metrics will NOT be shipped. Update METRICS_REMOTE_WRITE_URL in .env "
            f"to a publicly accessible host (e.g. ngrok tunnel for local Pushgateway).",
            flush=True,
        )

remote_write = {"url": url}
if username and password:
    remote_write["basic_auth"] = {
        "username": username,
        "password": password,
    }
elif username or password:
    print(
        "[write_grafana_config] partial remote_write auth config detected "
        "(missing username or password) - sending without basic_auth",
        flush=True,
    )

shared_labels = {
    "cluster_name": cluster,
    "provider": provider,
    "run_id": train_run_id,
    "preprocess_run_id": preprocess_run_id,
    "model_type": model_type,
    "source": metrics_source,
    "is_spot": is_spot,
}
shared_labels = {k: v for k, v in shared_labels.items() if v}

config = {
    "metrics": {
        "global": {
            "scrape_interval": "15s",
            "remote_write": [remote_write],
        },
        "configs": [
            {
                "name": "training-and-gpu",
                "scrape_configs": [
                    {
                        "job_name": "skypilot-training",
                        "static_configs": [
                            {
                                "targets": [f"localhost:{training_exporter_port}"],
                                "labels": {
                                    **shared_labels,
                                    "stream": "training",
                                },
                            }
                        ],
                    },
                    {
                        "job_name": "skypilot-gpu",
                        "static_configs": [
                            {
                                "targets": [f"localhost:{gpu_exporter_port}"],
                                "labels": {
                                    **shared_labels,
                                    "stream": "gpu",
                                },
                            }
                        ],
                    },
                ],
            }
        ],
    }
}

out = "/tmp/grafana-agent-config.yaml"
with open(out, "w", encoding="utf-8") as fh:
    yaml.safe_dump(config, fh, sort_keys=False)

print(
    f"[write_grafana_config] wrote {out} "
    f"(cluster={cluster} provider={provider} url={url})",
    flush=True,
)
