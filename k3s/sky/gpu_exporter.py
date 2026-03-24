"""Lightweight GPU metrics exporter.

Exposes GPU utilisation and memory metrics via prometheus_client HTTP server
on port 9400 (or GPU_EXPORTER_PORT env var). Intended to run as a background
process on every SkyPilot VM:

    python3 k3s/sky/gpu_exporter.py &

Metrics exported:
  skypilot_gpu_utilization_pct{gpu, cluster, is_spot}
  skypilot_gpu_memory_used_mb{gpu, cluster}
  skypilot_gpu_memory_total_mb{gpu, cluster}

Labels:
  cluster   — SKY_CLUSTER_NAME env var (injected by Airflow)
  is_spot   — USE_SPOT env var ("true" / "false")
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from prometheus_client import Gauge, start_http_server

CLUSTER = os.getenv("SKY_CLUSTER_NAME") or os.getenv("SKYPILOT_CLUSTER_NAME", "unknown")
IS_SPOT = os.getenv("USE_SPOT", "false")
PORT = int(os.getenv("GPU_EXPORTER_PORT", "9400"))
INTERVAL_S = int(os.getenv("GPU_EXPORTER_INTERVAL", "15"))

gpu_util = Gauge(
    "skypilot_gpu_utilization_pct",
    "GPU utilisation %",
    ["gpu", "cluster", "is_spot"],
)
gpu_mem_used = Gauge(
    "skypilot_gpu_memory_used_mb",
    "GPU memory used MB",
    ["gpu", "cluster"],
)
gpu_mem_total = Gauge(
    "skypilot_gpu_memory_total_mb",
    "GPU total memory MB",
    ["gpu", "cluster"],
)


def _scrape() -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    for i, line in enumerate(result.stdout.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpu_util.labels(gpu=str(i), cluster=CLUSTER, is_spot=IS_SPOT).set(
                float(parts[0])
            )
            gpu_mem_used.labels(gpu=str(i), cluster=CLUSTER).set(float(parts[1]))
            gpu_mem_total.labels(gpu=str(i), cluster=CLUSTER).set(float(parts[2]))


if __name__ == "__main__":
    start_http_server(PORT)
    print(
        f"[gpu_exporter] :{PORT}  cluster={CLUSTER}  is_spot={IS_SPOT}",
        flush=True,
    )
    while True:
        try:
            _scrape()
        except Exception as exc:  # noqa: BLE001
            print(f"[gpu_exporter] warn: {exc}", file=sys.stderr, flush=True)
        time.sleep(INTERVAL_S)
