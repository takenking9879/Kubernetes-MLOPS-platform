"""Wait for all Ray worker GPUs to register on the head node.

Run on the head node after `ray start --head`. Exits 0 when all expected
GPUs are visible in the Ray cluster, exits 1 on timeout (10 min).

Env vars read:
    SKYPILOT_NUM_NODES          — total nodes in the cluster (default 2)
    SKYPILOT_NUM_GPUS_PER_NODE  — GPUs per worker node (default 8)
"""
from __future__ import annotations

import os
import sys
import time

import ray

num_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", 2))
gpus_per_node = int(os.environ.get("SKYPILOT_NUM_GPUS_PER_NODE", 8))
expected_gpus = (num_nodes - 1) * gpus_per_node

ray.init(address="127.0.0.1:6379", ignore_reinit_error=True)
print(f"Waiting for {expected_gpus} GPUs across {num_nodes - 1} worker node(s)...", flush=True)

for attempt in range(60):
    available = ray.cluster_resources().get("GPU", 0)
    print(f"  [{attempt * 10}s] {available}/{expected_gpus} GPUs registered", flush=True)
    if available >= expected_gpus:
        print("All worker GPUs registered — starting training", flush=True)
        sys.exit(0)
    time.sleep(10)

print("ERROR: Timed out waiting for worker GPUs", file=sys.stderr, flush=True)
sys.exit(1)
