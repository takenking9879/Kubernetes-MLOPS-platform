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