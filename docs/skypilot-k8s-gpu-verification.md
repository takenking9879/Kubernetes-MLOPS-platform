# SkyPilot K8s GPU Support Verification Guide

Run these steps in order to verify GPU support is working.

## B0 - Cluster GPU-enabled

```bash
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, capacity: .status.capacity}'
```

- Must show `nvidia.com/gpu` with GPU count
- If missing: install Nvidia GPU operator

## B1 - GPU pod runs

```bash
kubectl apply -f https://raw.githubusercontent.com/skypilot-org/skypilot/master/tests/kubernetes/gpu_test_pod.yaml
kubectl get pod skygputest
kubectl logs skygputest
# Should print nvidia-smi output
kubectl delete -f https://raw.githubusercontent.com/skypilot-org/skypilot/master/tests/kubernetes/gpu_test_pod.yaml
```

If pod is pending: ensure `nvidia.com/gpu` resources available.
If logs show `nvidia-smi: command not found`: nvidia runtime not set as default.

## B2 - Nodes labeled correctly

```bash
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, labels: .metadata.labels}'
```

- Look for `skypilot.co/accelerator` label showing GPU type
- GKE: nodes auto-labeled with `cloud.google.com/gke-accelerator`

## B3 - SkyPilot sees GPUs

```bash
sky check
# Should show no GPU warnings
sky gpus list --infra k8s
```

## B4 - GPU task works

```bash
# Replace <gpu-type> with GPU from B3 output
sky launch -y -c mygpucluster --infra k8s --gpu <gpu-type>:1 -- "nvidia-smi"
sky down -y mygpucluster
```

### Verified Local K3S Training Path

The local training task in `k3s/sky/ray-gpu-training-k8s.yaml` was validated with:

- Custom image: `takenking9879/ray-train:2.53.0`
- Runtime user: `ray`
- Python selected from image: `/home/ray/anaconda3/bin/python`
- Pip selected from image: `/home/ray/anaconda3/bin/pip`
- Python version: `3.11.11`
- Torch runtime: `2.10.0+cu128`
- CUDA runtime reported by torch: `12.8`
- GPU detected in pod: `NVIDIA GeForce RTX 4070 Laptop GPU`

Observed setup/run evidence from Airflow logs:

```text
=== ENV DEBUG ===
ray
/home/ray/anaconda3/bin/python
Python 3.11.11
/home/ray/anaconda3/bin/pip
pip 24.3.1 from /home/ray/anaconda3/lib/python3.11/site-packages/pip (python 3.11)
/home/ray

=== PYTHON CHECK ===
/home/ray/anaconda3/bin/python

=== GPU CHECK ===
NVIDIA-SMI 580.108
Driver Version: 581.83
CUDA Version: 13.0

[setup] Accepted: /home/ray/anaconda3/bin/python (python=3.11.11, torch+cuda OK)
[setup] Selected image Python: /home/ray/anaconda3/bin/python (version=3.11.11)
[setup] Import smoke test passed.
[setup] runtime torch=2.10.0+cu128 cuda=12.8
[setup] GPU visible: NVIDIA GeForce RTX 4070 Laptop GPU
Setup complete (K3S local GPU pod).

[run] runtime python=3.11.11 exe=/home/ray/anaconda3/bin/python
[run] runtime torch=2.10.0+cu128 cuda_runtime=12.8
[run] runtime cuda_available=True
[run] GPU exporter started
```

Interpretation:

- SkyPilot Kubernetes pod successfully used the image Python, not a separate fallback runtime.
- `python` and `pip` were aligned to the same Anaconda environment inside the image.
- The custom image already contained the required torch+CUDA stack for local GPU execution.
- The pod saw the RTX 4070 correctly and entered the training run phase successfully.

## Quick Check

```bash
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, capacity: .status.capacity}'
sky gpus list --infra k8s
```

## Common Issues

| Issue | Fix |
|-------|-----|
| `nvidia.com/gpu` not found | Install Nvidia GPU operator |
| nvidia-smi not in pod | Set nvidia runtime as default |
| Missing `skypilot.co/accelerator` label | Label nodes manually |
