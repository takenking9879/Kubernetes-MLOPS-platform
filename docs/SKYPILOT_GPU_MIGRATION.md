# Migrating `kuberay-job-gpu.yaml` to SkyPilot

**Scope:** Replace the GPU execution layer (Virtual Kubelet → RunPod → Ray worker) with SkyPilot.
Nothing else changes: the training script, Docker image, and storage patterns stay the same.

**Assumptions:**
- Registry image is already pushed to Docker Hub as `takenking9879/ray-cluster-gpu:2.53.0`
- RunPod API key is available
- AWS credentials are available (S3 read/write)
- SkyPilot is installed in the environment where jobs are launched (local machine or Airflow worker)

---

## 1. Conceptual Mapping

### KubeRay RayJob → SkyPilot Task

| KubeRay concept | SkyPilot equivalent | Notes |
|-----------------|---------------------|-------|
| `RayJob` CRD | `sky.yaml` task file | Declarative job definition |
| `headGroupSpec` | Node 0 (`SKYPILOT_NODE_RANK == 0`) | Runs `ray start --head` + training script |
| `workerGroupSpec` | Nodes 1..N (`SKYPILOT_NODE_RANK > 0`) | Run `ray start --address=<head>` |
| `image:` on pod spec | `image_id: docker:<image>` | Same Docker image, no registry change |
| `resources.limits.nvidia.com/gpu` | `accelerators: A100:1` | SkyPilot resolves GPU type to cloud SKU |
| `nodeSelector` + `tolerations` | `cloud: runpod` in resources | No virtual node required |
| ConfigMap (script mount) | `file_mounts:` or baked into image | Sync from local path or S3 |
| `kubectl apply` | `sky launch` | Single command to provision + run |
| `kubectl logs` | `sky logs` | Streams from all nodes |
| `kubectl delete rayjob` | `sky down` | Terminates and releases VMs |

### What is no longer needed

| Old component | Replaced by |
|---------------|-------------|
| Virtual Kubelet binary (`vk-ml`) | SkyPilot handles provisioning directly |
| `vk-ml-secrets` Kubernetes Secret | SkyPilot config file with RunPod API key |
| Tailscale daemon in Docker image | SkyPilot manages SSH between nodes internally |
| `ray-gpu-test-head` ClusterIP Service | SkyPilot injects `SKYPILOT_NODE_IPS` env var |
| `nodeSelector: kubernetes.io/hostname: vk-ml-runpod` | Removed — no virtual node |
| Tailscale auth key rotation, ephemeral keys, MagicDNS sync | Gone entirely |

---

## 2. Minimal Working SkyPilot YAML

Create this file at `k3s/sky/ray-gpu-job.yaml`.

### Single-node variant (1 GPU, simpler)

Use this when the job fits on one machine. The training script runs on the head node itself
and dispatches Ray tasks to the local GPU.

```yaml
# k3s/sky/ray-gpu-job.yaml
name: ray-gpu-job

resources:
  cloud: runpod
  accelerators: A100:1        # Change to RTX4090:1 for cheaper option on RunPod
  use_spot: false             # Set true for interruptible (cheaper, may be evicted)
  disk_size: 50               # GB — matches --runpod-disk-gb 50 from old vk-ml config

# Use the same image that was used in kuberay-job-gpu.yaml
# Must be pullable by the RunPod VM — Docker Hub is fine
image_id: docker:takenking9879/ray-cluster-gpu:2.53.0

# Sync the training script into the VM
# If the script is baked into the image already, remove this block
file_mounts:
  /home/ray/gpu_test_main.py: k3s/kuberay/gpu/gpu_test_main.py

envs:
  # Override at launch time: sky launch --env AWS_ACCESS_KEY_ID=...
  AWS_ACCESS_KEY_ID: ""
  AWS_SECRET_ACCESS_KEY: ""
  AWS_DEFAULT_REGION: us-east-1

setup: |
  # Nothing to install — the Docker image already has all dependencies.
  # Add pip installs here only if you need to override something in the image.
  echo "Setup complete"

run: |
  # Start Ray head on this single node, claiming the GPU
  ray start --head --num-gpus=1 --num-cpus=4 --dashboard-host=0.0.0.0

  # Wait briefly for the Ray daemon to be ready
  sleep 5

  # Run the training script
  python3 /home/ray/gpu_test_main.py
```

### Multi-node variant (head + 1 GPU worker, mirrors the original architecture)

Use this when you want to preserve the exact head/worker split from `kuberay-job-gpu.yaml`.
The head has no GPU; the worker has 1 GPU. Matches the original:
`headGroupSpec.num-gpus: 0` / `workerGroupSpec.num-gpus: 1`.

```yaml
# k3s/sky/ray-gpu-job-multinode.yaml
name: ray-gpu-job-multinode

num_nodes: 2   # Node 0 = head (no GPU), Node 1 = worker (1 GPU)

resources:
  cloud: runpod
  accelerators: A100:1   # Applied to every node — see note below
  disk_size: 50

image_id: docker:takenking9879/ray-cluster-gpu:2.53.0

file_mounts:
  /home/ray/gpu_test_main.py: k3s/kuberay/gpu/gpu_test_main.py

envs:
  AWS_ACCESS_KEY_ID: ""
  AWS_SECRET_ACCESS_KEY: ""
  AWS_DEFAULT_REGION: us-east-1

setup: |
  echo "Setup complete"

run: |
  # SkyPilot provides:
  #   SKYPILOT_NODE_RANK   — 0 for head, 1+ for workers
  #   SKYPILOT_NODE_IPS    — newline-separated list of all node IPs (head is first)
  #   SKYPILOT_NUM_NODES   — total node count

  HEAD_IP=$(echo "$SKYPILOT_NODE_IPS" | head -1)

  if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    # Head node: no GPU allocated to Ray head (mirrors num-gpus: 0)
    ray start --head --port=6379 --num-gpus=0 --num-cpus=1 \
              --dashboard-host=0.0.0.0

    # Wait for the worker to join (polls every 5s, max 120s)
    echo "Waiting for GPU worker to register..."
    for i in $(seq 1 24); do
      GPUS=$(python3 -c "import ray; ray.init(address='auto',ignore_reinit_error=True); r=ray.cluster_resources(); print(r.get('GPU',0))" 2>/dev/null || echo 0)
      if [ "$GPUS" != "0" ] && [ -n "$GPUS" ]; then
        echo "GPU worker joined with $GPUS GPU(s)"
        break
      fi
      echo "  step $i/24 — waiting..."
      sleep 5
    done

    # Run the training entrypoint
    python3 /home/ray/gpu_test_main.py

  else
    # Worker node: connect to head, expose 1 GPU
    sleep 10   # give head a moment to start GCS
    ray start --address="$HEAD_IP:6379" \
              --num-gpus=1 \
              --num-cpus=2 \
              --node-ip-address=$(hostname -I | awk '{print $1}') \
              --block
  fi
```

> **Note on GPU allocation per node in multi-node mode:**
> SkyPilot applies the same `resources:` block to every node. In the multi-node YAML above,
> both nodes are provisioned with a GPU, but Ray head is started with `--num-gpus=0` so Ray
> does not schedule GPU tasks on it. This mirrors the original KubeRay setup exactly.
> If you want to avoid paying for the head node's GPU, use SkyPilot's `any_of` resource spec
> or provision head separately — for a single-worker job, the single-node variant is simpler.

---

## 3. Running the Job

### Launch

```bash
# Provision VM(s) on RunPod and run the job
# Terminates the cluster automatically after the job finishes (idle_minutes=0)
sky launch k3s/sky/ray-gpu-job.yaml \
  --cluster-name ray-gpu-test \
  --idle-minutes-to-autostop 5 \
  --yes

# Pass AWS credentials without hardcoding them in the YAML
sky launch k3s/sky/ray-gpu-job.yaml \
  --cluster-name ray-gpu-test \
  --idle-minutes-to-autostop 5 \
  --env AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id) \
  --env AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key) \
  --yes
```

`sky launch` does:
1. Queries RunPod for GPU availability at the requested spec
2. Provisions the VM(s) via RunPod API
3. SSHs in, runs the `setup:` block
4. Syncs `file_mounts:`
5. Executes the `run:` block
6. Streams logs back to your terminal
7. Auto-terminates after idle timeout

### Stream logs

```bash
# Follow logs from all nodes (both head and worker in multinode)
sky logs ray-gpu-test --follow

# Logs from a previous run (by job ID)
sky logs ray-gpu-test 1
```

### Check cluster status

```bash
sky status                       # List all active clusters
sky status ray-gpu-test          # Detail for this cluster
```

### Re-run on an existing warm cluster (skip provisioning)

```bash
sky exec ray-gpu-test k3s/sky/ray-gpu-job.yaml
```

Use `sky exec` during development to avoid paying for repeated VM provisioning.

### Terminate

```bash
# Terminate immediately
sky down ray-gpu-test

# Terminate all clusters
sky down --all
```

`sky down` calls the RunPod API to delete the pod(s). Unlike the old approach, there is no
kubectl delete + vk-ml reconcile loop to wait for.

---

## 4. Using RunPod

### How SkyPilot connects to RunPod

SkyPilot uses the RunPod REST API to provision pods, the same API that your Virtual Kubelet
was calling internally. The difference is that SkyPilot manages the full lifecycle and SSH
access itself — you do not need a Virtual Kubelet process running in your cluster.

Connectivity:
- SkyPilot provisions an SSH key pair at first use
- RunPod VMs get the public key injected at creation
- All further communication (file sync, log streaming, job execution) goes over SSH
- No Tailscale, no WireGuard, no VPN required for single-node jobs
- For multi-node: SkyPilot sets up SSH between nodes itself using the same key pair

### Required credentials

```bash
# Install SkyPilot with RunPod support
pip install "skypilot[runpod]"

# Verify RunPod is recognised
sky check
```

Create or edit `~/.sky/config.yaml`:

```yaml
# ~/.sky/config.yaml
runpod:
  api_key: rpa_XXXXXXXXXXXXXXXXXXXX   # Your RunPod API key
```

Verify the connection:

```bash
sky check runpod
# Expected: RunPod: enabled
```

### GPU type naming on RunPod

SkyPilot uses standardised short names. Common mappings:

| SkyPilot accelerator name | RunPod GPU |
|---------------------------|------------|
| `A100-80GB:1` | NVIDIA A100 80GB PCIe |
| `A100:1` | NVIDIA A100 40GB |
| `RTX4090:1` | NVIDIA GeForce RTX 4090 |
| `RTX3090:1` | NVIDIA GeForce RTX 3090 |
| `A40:1` | NVIDIA A40 |
| `H100:1` | NVIDIA H100 (if available on RunPod) |

To see what is currently available and priced:

```bash
sky show-gpus --cloud runpod
```

### Limitations (RunPod-specific)

- **No spot/interruptible instances on SECURE cloud** — `use_spot: true` requires `COMMUNITY`
  cloud type on RunPod. SkyPilot maps `use_spot` to RunPod's interruptible flag.
- **No persistent volumes** — RunPod pod storage is ephemeral. Write results to S3 before exit.
- **Image pull time** — `ray-cluster-gpu:2.53.0` is a large image. First launch on a cold
  RunPod datacenter node takes 3–10 minutes. `sky exec` reuses the warm VM and skips this.
- **Datacenter availability** — if the requested GPU is unavailable, RunPod returns an error.
  Add a fallback: `cloud: [runpod, lambda]` and SkyPilot will try the next provider.

---

## 5. Data Access

### Reading data (S3 → VM)

Option A — Pass credentials via env vars and use boto3 in the training script (already in
`requirements_gpu.txt`):

```bash
sky launch k3s/sky/ray-gpu-job.yaml \
  --env AWS_ACCESS_KEY_ID=... \
  --env AWS_SECRET_ACCESS_KEY=... \
  --env S3_DATA_PATH=s3://my-bucket/datasets/network_traffic/
```

The training script reads:

```python
import boto3, os
s3 = boto3.client("s3")
s3.download_file("my-bucket", "datasets/network_traffic/train.parquet", "/tmp/train.parquet")
```

Option B — Sync via `file_mounts` (SkyPilot downloads before `run:` starts):

```yaml
file_mounts:
  /data/train.parquet: s3://my-bucket/datasets/network_traffic/train.parquet
```

This requires SkyPilot to have AWS credentials in its own config (preferred for S3 → VM sync):

```yaml
# ~/.sky/config.yaml
aws:
  access_key_id: ...
  secret_access_key: ...
```

### Writing results (VM → S3)

Write from the training script directly using boto3, or at the end of the `run:` block:

```yaml
run: |
  ray start --head --num-gpus=1
  python3 /home/ray/gpu_test_main.py

  # Upload model artifact after training finishes
  aws s3 cp /tmp/model.pkl s3://my-bucket/runs/training/<run-id>/model.pkl
```

Set `AWS_DEFAULT_REGION` in `envs:` to avoid region resolution errors.

### Keeping credentials out of the YAML

Never hardcode credentials in `k3s/sky/ray-gpu-job.yaml`. Always pass at launch time via
`--env` or store them in `~/.sky/config.yaml` (which is not committed to the repo).

For production, use IAM instance profiles (AWS) or RunPod secrets injection.

---

## 6. Differences vs the Old Approach

### What the old approach required

```
Virtual Kubelet process  (custom Go binary, vk-ml)
  ├── Running in kube-system namespace
  ├── Calling RunPod REST API
  ├── Generating ephemeral Tailscale keys per worker
  └── Syncing MagicDNS records to Tailscale API

Tailscale on K3s node
  ├── Subnet router advertising pod + service CIDRs
  └── Routing Ray GCS traffic (port 6379) from worker to ClusterIP

Tailscale in Docker image
  ├── tailscaled in userspace mode
  ├── tailscale up on container start
  └── Waiting for TS_IP before ray start

Ray head in Kubernetes
  ├── ClusterIP Service (ray-gpu-test-head)
  ├── Dependent on MagicDNS record being correct
  └── Workers must resolve ray-head.<tailnet>.ts.net

Worker bootstrap sequence (inside RunPod container)
  1. tailscaled --tun=userspace-networking
  2. tailscale up --authkey=...
  3. Wait for Tailscale IP
  4. ray health-check --address ray-head...ts.net:6379
  5. ray start --address=... --node-ip-address=$TS_IP
```

### What SkyPilot requires

```
SkyPilot CLI or SDK (pip package, runs anywhere)
  ├── ~/.sky/config.yaml  (cloud credentials)
  └── k3s/sky/ray-gpu-job.yaml  (job definition)
```

That is the entire operational surface.

### Component-by-component comparison

| Old component | Status | Reason |
|---------------|--------|--------|
| Virtual Kubelet (`vk-ml`) | **Removed** | SkyPilot provisions VMs directly |
| `vk-ml-secrets` K8s Secret | **Removed** | SkyPilot uses local config file |
| Tailscale on K3s node | **Removed** | No cross-boundary network needed |
| Tailscale in Docker image | **Removed** | SkyPilot uses SSH, not Tailscale |
| Tailscale ephemeral key rotation | **Removed** | Not applicable |
| MagicDNS sync logic | **Removed** | Not applicable |
| `ray-gpu-test-head` ClusterIP Service | **Removed** | Head is on a SkyPilot VM |
| Ray head in Kubernetes | **Moved to SkyPilot VM** | Head runs in the `run:` block |
| KubeRay operator | **No longer needed for this job** | No `RayJob` CRD submitted |
| RunPod API key | **Still needed** | Now in `~/.sky/config.yaml` |
| Docker image | **Unchanged** | Same image, same registry |
| Training script | **Unchanged** | `gpu_test_main.py` unchanged |

### Tailscale removal note

Tailscale was needed because Ray workers on RunPod had to reach the Ray head's ClusterIP
inside Kubernetes. With SkyPilot, head and workers are co-located on the same provider
network (or SkyPilot handles inter-node SSH). There is no cross-cluster boundary.

You can remove Tailscale from the Docker image in a follow-up. The `tailscale install` step
in `k3s/kuberay/gpu/Dockerfile` is now dead code. Keep it for now if you want the image to
remain compatible with the old setup during transition.

---

## 7. Common Pitfalls

### Image not pullable from RunPod

**Symptom:** Job fails at setup with `Error response from daemon: pull access denied`.

**Cause:** Docker Hub rate limits, private image, or wrong tag.

**Fix:**
- Confirm the image is public: `docker pull takenking9879/ray-cluster-gpu:2.53.0`
- If private, push credentials to RunPod via `~/.sky/config.yaml` or use a public registry
- Confirm the tag exists: `docker manifest inspect takenking9879/ray-cluster-gpu:2.53.0`

---

### GPU type unavailable on RunPod

**Symptom:** `sky launch` fails with `No resources found` or `ResourcesUnavailableError`.

**Fix 1:** Check availability before launching:
```bash
sky show-gpus --cloud runpod
```

**Fix 2:** Add a fallback in the resources spec:
```yaml
resources:
  any_of:
    - cloud: runpod
      accelerators: A100:1
    - cloud: runpod
      accelerators: RTX4090:1
    - cloud: lambda
      accelerators: A100:1
```

**Fix 3:** Use `--retry-until-up` to keep retrying until a GPU is available:
```bash
sky launch k3s/sky/ray-gpu-job.yaml --retry-until-up --yes
```

---

### Ray head not reachable from worker (multi-node)

**Symptom:** Worker prints `ConnectionError` or `Failed to connect to GCS` on `ray start`.

**Cause:** Race condition — head node's Ray daemon not ready when worker tries to connect.

**Fix:** The `sleep 10` in the worker block handles most cases. For slow VM starts, increase it.
Check that the head node's firewall allows inbound on port 6379 (SkyPilot opens this via
security group rules automatically on AWS/GCP; verify RunPod's network policy if this occurs).

Verify once the cluster is up:
```bash
sky ssh ray-gpu-test -- "ray status"
```

---

### `ray.init(address='auto')` fails

**Symptom:** Training script exits with `ConnectionError: Could not connect to Ray`.

**Cause:** `ray start --head` was not called before the script, or Ray daemon crashed.

**Fix:** Add an explicit readiness check in the `run:` block:
```bash
run: |
  ray start --head --num-gpus=1
  # Wait until GCS is accepting connections
  python3 -c "
  import ray, time
  for _ in range(12):
      try: ray.init(address='auto', ignore_reinit_error=True); break
      except: time.sleep(5)
  "
  python3 /home/ray/gpu_test_main.py
```

---

### CUDA not available inside the container

**Symptom:** `torch.cuda.is_available()` returns `False`; training script exits with
`RuntimeError: Error: CUDA not available on this worker`.

**Cause 1:** GPU not attached to the VM — confirm with `nvidia-smi` in the `run:` block:
```bash
run: |
  nvidia-smi
  ray start --head --num-gpus=1
  python3 /home/ray/gpu_test_main.py
```

**Cause 2:** Image CUDA version mismatch. The image uses `cu128` (CUDA 12.8).
RunPod nodes run driver versions from `~515` upward. Confirm with `nvidia-smi` output.
If the driver is older than CUDA 12.8 requires, switch to a `cu118` or `cu121` build.

**Cause 3:** Docker `--gpus all` not passed. SkyPilot passes this automatically when
`accelerators:` is set. If you run the image manually for testing, remember `--gpus all`.

---

### File mounts not synced

**Symptom:** `python3 /home/ray/gpu_test_main.py: No such file or directory`.

**Cause:** `file_mounts:` path is relative to where `sky launch` is run.

**Fix:** Run `sky launch` from the repo root, or use an absolute path:
```yaml
file_mounts:
  /home/ray/gpu_test_main.py: ./k3s/kuberay/gpu/gpu_test_main.py
```

Or bake the script into the Docker image and remove `file_mounts:` entirely.

---

### Cluster left running (cost leak)

**Symptom:** RunPod charges accumulate after the job finishes.

**Fix:** Always set `--idle-minutes-to-autostop`:
```bash
sky launch k3s/sky/ray-gpu-job.yaml --idle-minutes-to-autostop 5 --yes
```

Or in the YAML:
```yaml
# Not a standard SkyPilot YAML key — pass at CLI or set a global default:
# sky autostop ray-gpu-test --idle-minutes 5
```

Set a global default in `~/.sky/config.yaml`:
```yaml
jobs:
  controller:
    resources:
      cloud: runpod
autostop:
  idle_minutes: 10
```

Audit running clusters before ending a work session:
```bash
sky status
sky down --all   # if nothing should be running
```

---

### Multi-node: training script runs on both nodes

**Symptom:** `gpu_test_main.py` runs twice — once on head, once on worker.

**Cause:** The `run:` block executes on every node by default.

**Fix:** Guard the entrypoint with `SKYPILOT_NODE_RANK`:
```bash
run: |
  if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    ray start --head ...
    python3 /home/ray/gpu_test_main.py
  else
    ray start --address=... --block
  fi
```
This is already done correctly in the multi-node YAML in section 2.

---

## Appendix: File Structure

```
k3s/
├── kuberay/
│   ├── kuberay-job-gpu.yaml          ← old (keep for reference)
│   └── gpu/
│       ├── Dockerfile                ← unchanged (Tailscale still in image, harmless)
│       ├── requirements_gpu.txt      ← unchanged
│       └── gpu_test_main.py          ← unchanged (extracted from ConfigMap if needed)
└── sky/
    ├── ray-gpu-job.yaml              ← NEW: single-node SkyPilot job
    └── ray-gpu-job-multinode.yaml    ← NEW: multi-node SkyPilot job (head + worker)
```

The training script `gpu_test_main.py` is currently embedded in a ConfigMap inside
`kuberay-job-gpu.yaml`. Extract it to `k3s/kuberay/gpu/gpu_test_main.py` as a standalone
file so `file_mounts:` in the SkyPilot YAML can reference it directly.
