# Phase 0 PoC — Local Docker GPU Worker joined to K8s Ray Cluster

Validate that a Docker container with GPU access can join the Ray head pod
running inside Kubernetes and execute GPU training tasks.

---

## Prerequisites

- Docker Desktop with `--gpus all` working (`docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`)
- Kubernetes cluster running with Ray head pod active in namespace `ray`
- GPU image built: `ray-cluster-gpu:2.53.0`

---

## Step 1 — Apply the head-only RayJob

```bash
kubectl apply -f k3s/kuberay/kuberay-job-gpu.yaml
```

Wait for the head pod to be Running:

```bash
kubectl get pods -n ray -w
```

---

## Step 2 — Get the head pod IP

```bash
HEAD_IP=$(kubectl get pod -n ray -l ray.io/node-type=head \
  -o jsonpath='{.items[0].status.podIP}')
echo "Head IP: $HEAD_IP"
```

---

## Step 3 — Get the worker's advertise IP

The worker must advertise an IP that the head pod can reach back to
(for object store communication). On Docker Desktop, this is the VM host IP:

```bash
HOST_IP=$(docker run --rm --network=host alpine \
  ip route get 8.8.8.8 | awk '{print $7; exit}')
echo "Worker will advertise: $HOST_IP"
```

---

## Step 4 — Start the GPU worker container

Open a **dedicated terminal** (this process blocks until the Ray head disconnects):

```bash
docker run --rm \
  --gpus all \
  --network=host \
  -e PYTHONPATH=/home/ray/app/repo:/home/ray/app/repo/src \
  ray-cluster-gpu:2.53.0 \
  ray start \
    --address="${HEAD_IP}:6379" \
    --num-gpus=1 \
    --num-cpus=10 \
    --node-ip-address="${HOST_IP}" \
    --min-worker-port=20000 \
    --max-worker-port=20020 \
    --block
```

Expected output: `Ray runtime started.`

---

## Step 5 — Verify the GPU worker joined

In another terminal:

```bash
HEAD_POD=$(kubectl get pods -n ray -l ray.io/node-type=head \
  -o jsonpath='{.items[0].metadata.name}')

kubectl exec -it $HEAD_POD -n ray -- ray status
```

Expected: `GPU: 1.0` visible under cluster resources.

---

## Step 6 — Run a GPU smoke test

```bash
kubectl exec -it $HEAD_POD -n ray -- python3 -c "
import ray
ray.init(address='auto')

@ray.remote(num_gpus=1)
def check_gpu():
    import torch
    return torch.cuda.get_device_name(0)

print(ray.get(check_gpu.remote()))
"
```

Expected: `NVIDIA GeForce RTX 4070` (or your GPU model).

---

## Step 7 — Trigger a full training job via Airflow

Trigger `training_pipeline_gpu` DAG with:

```json
{
  "train_run_id":          "train-poc-001",
  "preprocess_run_id":     "<your-preprocess-run-id>",
  "dataset":               "network_traffic",
  "processed_table":       "<your-iceberg-table>",
  "dsl_s3_path":           "s3://...",
  "train_params_s3_path":  "s3://...",
  "model_type":            "pytorch",
  "use_runpod":            false
}
```

`use_runpod: false` skips RunPod provisioning. The DAG submits the RayJob
and the external Docker worker handles all GPU tasks.

---

## Cleanup

Stop the Docker worker with `Ctrl+C` (or `docker stop` the container).
The RayJob is deleted automatically by the `poll_ray_job_gpu` finally block.
To delete manually:

```bash
kubectl delete rayjob -n ray --all
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Worker can't connect to 6379 | Wrong HEAD_IP | Re-check `kubectl get pod ... -o jsonpath='{.status.podIP}'` |
| `GPU: 0` in `ray status` | Missed `--num-gpus=1` | Re-run docker command with the flag |
| Object store timeout | Head can't reach HOST_IP:20000-20020 | Confirm HOST_IP is the VM host IP, not a Docker bridge |
| `torch.cuda` not available | Image built without CUDA deps | Rebuild `ray-cluster-gpu:2.53.0` with `requirements_gpu.txt` |
| RayJob stuck in PENDING | Head pod not Running yet | Wait; check `kubectl get pods -n ray` |
