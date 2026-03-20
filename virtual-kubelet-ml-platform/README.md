# virtual-kubelet-ml-platform

A minimal Virtual Kubelet provider written in Go that bridges KubeRay worker pods
to external GPU compute backends (RunPod, local Docker). Designed as a debuggable
Proof of Concept: run the full scheduling loop locally without spending money on GPUs,
then switch to RunPod by changing one flag.

---

## Origin — what changed from `k8s-runpod-kubelet/`

The existing `k8s-runpod-kubelet/` folder was used for initial research. This project
is a clean rewrite that keeps only the useful parts.

| Component | Decision | Reason |
|-----------|----------|--------|
| `virtual-kubelet v1.9.0` | **Reused** | Correct framework, stable |
| RunPod REST endpoint (`rest.runpod.io/v1`) | **Reused** | Works, no changes needed |
| Status translation (STARTING/RUNNING/EXITED) | **Reused, simplified** | Same mapping |
| Distroless Dockerfile + RBAC pattern | **Reused** | Good practice |
| GPU Conduit / heartbeat dependency | **Removed** | Commercial SaaS, not needed |
| GraphQL API path | **Removed** | Unreliable in production (GRAPHQL_VALIDATION_FAILED) |
| Hardcoded node capacity | **Removed** | Replaced with configurable flags |
| Complex 3-tier cleanup goroutines | **Removed** | Replaced with simple reconcile loop |
| Sentry integration | **Removed** | Not needed for POC |
| `ComputeProvider` interface + LocalProvider | **New** | Clean abstraction; Docker debug mode |
| `WrapArgsWithGCSWait` | **New** | Replaces KubeRay initContainer that VK skips |
| K8s DNS → NodePort rewriting | **New** | Required for Docker Desktop networking |

---

## File-by-file guide

### `cmd/vk-ml/main.go`

Bootstrap entry point. Builds the Kubernetes client **first** (before selecting the compute
provider) because `LocalProvider` needs the client to look up NodePort services for DNS
rewriting. Then selects the compute backend (`runpod` or `local`), creates the VK `Provider`,
registers the virtual node in Kubernetes, and starts the `NodeController` and `PodController`
from the virtual-kubelet framework. Blocks until `SIGINT`/`SIGTERM`, then shuts down cleanly.

### `pkg/config/config.go`

Loads all configuration from CLI flags and environment variables. Key flags:
`--compute-provider` (`runpod`|`local`), `--node-name` (name of the virtual node in K8s),
`--namespace` (which namespace to watch for pods), `--reconcile-interval` (how often to poll
instance statuses). The RunPod API key is always loaded from the `RUNPOD_API_KEY` environment
variable — never from a flag — to avoid accidental exposure in shell history or logs.

### `pkg/compute/types.go`

Defines the internal contract between the VK provider and any compute backend:

- **`InstanceSpec`** — minimal spec extracted from a K8s Pod: image, command, args, env, CPU,
  memory, GPU count, ports. Only what a GPU provider needs to launch a container.
- **`Instance`** — running or completed compute instance: ID, status, exit code, public IP,
  port mappings (`{"6379": 10341}`).
- **`InstanceStatus`** — canonical lifecycle states: `STARTING`, `RUNNING`, `EXITED`,
  `TERMINATED`, `NOT_FOUND`, `UNKNOWN`.
- **`ComputeProvider`** — the interface all backends implement (`CreateInstance`, `DeleteInstance`,
  `GetInstance`, `ListInstances`). Adding a new provider only requires implementing this interface.

### `pkg/compute/local/provider.go`

Docker-based debug provider. Launches containers with:
- `--network host` — container shares the host's network stack (required for Ray workers on WSL2)
- `--shm-size=5gb` — shared memory for Ray's object store
- `--gpus all` — when `InstanceSpec.GPU > 0`

**Key feature: `rewriteK8sAddresses`**. Docker Desktop runs Kubernetes inside a VM, so pod IPs
and ClusterIPs are not routable from the WSL2 host. NodePort services are reachable at
`127.0.0.1:<nodePort>`. Before launching the container, `rewriteK8sAddresses` scans all env
values and the full args shell string for `*.svc.cluster.local:PORT` patterns. For each match,
it looks up the corresponding NodePort via the K8s API and replaces the address with
`127.0.0.1:<nodePort>`. It also injects `--add-host <fqdn>:127.0.0.1` entries so that c-ares
(used by Ray's gRPC layer, which may bypass `/etc/hosts`) also resolves correctly.

### `pkg/compute/runpod/client.go`

Thin HTTP client for the RunPod REST API (`rest.runpod.io/v1`). Only REST is used — the
GraphQL path is excluded because it has known reliability issues. Key struct: `PodResponse`,
where `PublicIp` is a **top-level field** (not inside `machine{}`), and `PortMappings` is
`map[string]int` mapping internal container ports to their external public ports
(e.g. `{"6379": 10341}` — Ray GCS reachable at `PublicIp:10341`, not at `PublicIp:6379`).

### `pkg/compute/runpod/provider.go`

Implements `ComputeProvider` using RunPod's REST API. Translates `InstanceSpec` into a
`CreatePodRequest` where `minVCPUPerGPU` and `minRAMPerGPU` are per-GPU ratios (not totals).
Contains `wrapArgsForRunPod`: injects `--node-ip-address=$MYIP` into the `ray start` command
by prepending `MYIP=$(curl -s https://api.ipify.org)` to the container's shell script. This is
necessary because RunPod containers are assigned a public IP, and the Ray head needs to know
that IP to dispatch tasks back to the worker. Without it, Ray might auto-detect an internal
IP unreachable from the cluster.

### `pkg/provider/translator.go`

`TranslatePod(*v1.Pod) → InstanceSpec`. Only the first container is processed. Resources
prefer `Requests`, fall back to `Limits`. Default ports exposed on all Ray workers:
`6379/tcp` (GCS), `10001/tcp` (Ray client), `8265/http` (dashboard — HTTP proxy OK for
browsers), `8002/tcp` (Prometheus metrics). Env extraction handles direct `Value` entries only
(no `SecretKeyRef` for this POC — those values are silently skipped).

### `pkg/provider/wrapper.go`

`WrapArgsWithGCSWait(args []string, timeoutSec int) []string` — the GCS wait logic.

KubeRay injects an initContainer named `wait-gcs-ready` into every worker pod. This container
loops using `ray health-check --address "$RAY_ADDRESS"` until the Ray GCS responds, then exits
so the main container starts. The Virtual Kubelet provider skips initContainers entirely (only
`pod.Spec.Containers[0]` is launched). `WrapArgsWithGCSWait` replicates the exact same wait
script by prepending it to `args[0]` (the KubeRay-generated shell string).

The script uses `$RAY_ADDRESS` (always set by KubeRay, already rewritten to `127.0.0.1:30379`
by `LocalProvider.rewriteK8sAddresses`). Behaviour mirrors KubeRay exactly:
- First 120 s: health-check output suppressed (less noise during normal startup)
- After 120 s: full output shown (for diagnosing slow starts)
- After `timeoutSec` (default 300): `exit 1` to avoid hanging indefinitely

### `pkg/provider/status.go`

Translates a `compute.Instance` into a Kubernetes `v1.PodStatus`:

| Instance status | K8s PodPhase | Notes |
|----------------|--------------|-------|
| `STARTING` | `Pending` | Container creating |
| `RUNNING` | `Running` | Ready condition true |
| `EXITED` (code 0) | `Succeeded` | |
| `EXITED` (code ≠ 0) | `Failed` | |
| `TERMINATED` | `Succeeded` | Gracefully stopped |
| `NOT_FOUND` | `Failed` | Instance gone |

`PodIP` is set to `Instance.PublicIP` (RunPod's pod-level public IPv4). `PortMappingAnnotations`
writes `vk-ml/port-<internal>=<external>` annotations so external ports are discoverable without
reading RunPod's API directly (e.g. `vk-ml/port-6379=10341`, `vk-ml/public-ip=100.65.0.119`).

### `pkg/provider/provider.go`

Core VK provider. Implements three interfaces from the virtual-kubelet framework:

- **`PodLifecycleHandler`** — `CreatePod`, `UpdatePod`, `DeletePod`, `GetPod`, `GetPodStatus`,
  `GetPods`. `CreatePod` calls `TranslatePod` → `WrapArgsWithGCSWait` → `compute.CreateInstance`
  and stores the pod→instanceID mapping in an in-memory map.
- **`NodeProvider`** — `Ping` (always nil), `NotifyNodeStatus` (pushes node status every 30 s).
  `BuildNode` returns a `*v1.Node` with configurable CPU/memory/GPU capacity and the taint
  `virtual-kubelet.io/provider=runpod:NoSchedule`.
- **`PodNotifier`** — `NotifyPods` registers the callback the `PodController` uses for
  push-based status updates, then starts the background `reconcileLoop`.

The `reconcileLoop` runs every `--reconcile-interval` (default 15 s): for each tracked pod,
it calls `compute.GetInstance`, translates the status with `BuildPodStatus`, and pushes the
updated pod to the `PodController` via `notifyFunc`. This drives the `Pending → Running →
Succeeded/Failed` transitions visible in `kubectl get pods`.

### `deploy/vk-ml.yaml`

All-in-one Kubernetes manifest: `ClusterRole` (minimal RBAC — no wildcard verbs on secrets),
`ServiceAccount`, `ClusterRoleBinding`, a `Secret` placeholder for `RUNPOD_API_KEY`, and a
`Deployment` with liveness/readiness probes on port 8080.

### `Dockerfile`

Multi-stage build: `golang:1.24` compiles the binary with `CGO_ENABLED=0`, then it's copied
into `gcr.io/distroless/static:nonroot` (no shell, ~2 MB, runs as non-root UID 65532).

---

## Full execution flow

```
1.  kubectl apply -f k3s/kuberay/kuberay-job-gpu.yaml
    → KubeRay Operator creates RayCluster
    → Launches head pod on docker-desktop node (real K8s)

2.  KubeRay generates worker Pod (v1.Pod):
      metadata.name: ray-<job>-<cluster>-gpu-group-worker-<id>
      spec.nodeSelector: kubernetes.io/hostname: vk-ml-local
      spec.tolerations: [virtual-kubelet.io/provider, nvidia.com/gpu]
      spec.containers[0].env:
        RAY_ADDRESS = <HEAD_SVC>.<NS>.svc.cluster.local:6379
        FQ_RAY_IP  = <HEAD_SVC>.<NS>.svc.cluster.local
        RAY_IP     = <HEAD_SVC>
      spec.containers[0].args[0]:
        "ulimit -n 65536; ray start --address=<HEAD_SVC>...:6379 --block ..."

3.  kube-scheduler assigns pod to virtual node vk-ml-local

4.  VK PodController detects pod via filtered informer
    (field-selector: spec.nodeName=vk-ml-local)
    → calls Provider.CreatePod(ctx, pod)

5.  Provider.CreatePod:
      a. TranslatePod  → InstanceSpec {image, args[0], env, CPU=2, MemGB=8, GPU=1}
      b. WrapArgsWithGCSWait → prepends GCS wait loop (uses $RAY_ADDRESS) to args[0]
      c. compute.CreateInstance(ctx, spec)

── LocalProvider path ──────────────────────────────────────────────────────────

6a. LocalProvider.rewriteK8sAddresses:
      - finds *.svc.cluster.local:6379 in env AND inside args[0] (incl. wait loop)
      - K8s API: lists NodePort services in namespace → finds port 6379 → nodePort 30379
      - rewrites: RAY_ADDRESS env    = "127.0.0.1:30379"
      - rewrites: every match in args[0]            = "127.0.0.1:30379"
                  (this includes the address inside the GCS wait loop)
      - adds --add-host entries for c-ares gRPC DNS resolver

    docker run -d --rm \
      --name <pod-name> --label vk-ml=true \
      --network host --shm-size=5gb \
      [--gpus all] \
      --add-host <fqdn>:127.0.0.1 \
      -e RAY_ADDRESS=127.0.0.1:30379 \
      -e FQ_RAY_IP=127.0.0.1 \
      ... \
      ray-cluster-gpu:2.53.0 \
      /bin/bash -c -- "<wait-loop>\nulimit -n 65536; ray start --address=127.0.0.1:30379 --block ..."

── RunPodProvider path ─────────────────────────────────────────────────────────

6b. RunPodProvider.wrapArgsForRunPod:
      - prepends: MYIP=$(curl -s https://api.ipify.org || hostname -I | awk '{print $1}')
      - injects:  --node-ip-address=$MYIP into ray start

    POST https://rest.runpod.io/v1/pods {
      gpuCount: 1, minVCPUPerGPU: 2, minRAMPerGPU: 8,
      ports: ["6379/tcp","10001/tcp","8265/http","8002/tcp"],
      env: {RAY_ADDRESS: "<HEAD_SVC>:6379", ...},
      dockerEntrypoint: ["/bin/bash","-c","--"],
      dockerStartCmd: ["MYIP=$(curl...)\n<wait-loop>\nulimit -n 65536; ray start ..."]
    }

────────────────────────────────────────────────────────────────────────────────

7.  Container starts and executes:

      # GCS wait loop ($RAY_ADDRESS = 127.0.0.1:30379 for LocalProvider)
      SECONDS=0
      while true; do
        if (( SECONDS <= 120 )); then
          if ray health-check --address "$RAY_ADDRESS" > /dev/null 2>&1; then
            echo "GCS is ready."; break
          fi
          echo "$SECONDS seconds elapsed: Waiting for GCS to be ready."
        else
          ...  # verbose output after 120s
        fi
        sleep 5
      done

      # Ray worker starts (GCS is confirmed ready)
      ulimit -n 65536
      ray start --address=127.0.0.1:30379 --block ...

      → Worker registers in Ray cluster ✓

8.  Ray cluster resources updated:
      node:192.168.65.3  (Docker Desktop VM node IP — normal, reachable via NodePort)
      GPU: 1.0
      CPU: 2.0 (available to Ray scheduler)

9.  reconcileLoop (every 15 s, background goroutine):
      LocalProvider.GetInstance(containerID)
        → docker inspect → State.Status="running" → StatusRunning
      BuildPodStatus(pod, inst)
        → PodRunning, PodIP="192.168.65.3"
      notifyFunc(podCopy)
        → PodController → PATCH /api/v1/namespaces/ray/pods/<name>/status
        → kubectl get pods shows: 1/1 Running ✓

10. RayJob Python entrypoint (running in head pod) submits GPU task:
      @ray.remote(num_gpus=1)
      def train_gpu_task(): ...

      result = ray.get(train_gpu_task.remote())
      → Ray scheduler places task on worker GPU ✓
```

---

## Networking — Docker Desktop / WSL2

When Kubernetes runs inside the Docker Desktop VM, pod IPs and ClusterIPs are not routable
from the WSL2 host. All traffic must go through NodePort services via `127.0.0.1:<nodePort>`
(Windows localhost proxy → Docker Desktop VM → pod).

### Required companion service: `ray-gpu-test-head`

Defined in `k3s/kuberay/kuberay-job-gpu.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ray-gpu-test-head
  namespace: ray
spec:
  type: NodePort
  ports:
    - name: gcs
      port: 6379
      targetPort: 6379
      nodePort: 30379   # reachable at 127.0.0.1:30379 from WSL2
    - name: dashboard
      port: 8265
      targetPort: 8265
      nodePort: 30265
  selector:
    ray.io/node-type: head
    # DO NOT add ray.io/cluster here.
    # KubeRay appends a random suffix to each RayJob run (e.g. -wm9mw, -7pcfr).
    # A hardcoded cluster name causes the service to have no endpoints after
    # kubectl delete + kubectl apply.
```

### How `rewriteK8sAddresses` works

1. Regex scans env values and the full `args[0]` shell string for `<svc>.<ns>.svc.cluster.local[:port]`
2. For each match: calls `clientset.CoreV1().Services(ns).Get(svcName)` to find the NodePort.
   If the service is headless (no NodePort), falls back to listing all NodePort services in the
   namespace and picking the one that exposes the right port.
3. Replaces the hostname+port with `127.0.0.1:<nodePort>` everywhere (env AND args string,
   including inside the GCS wait loop that was prepended by `WrapArgsWithGCSWait`).
4. Adds `--add-host <fqdn>:127.0.0.1` for each resolved hostname. This is needed because Ray's
   gRPC layer uses c-ares for DNS, which may query the container's DNS server (`192.168.65.7`,
   Docker Desktop's DNS) instead of reading `/etc/hosts`. Docker Desktop's DNS resolves K8s
   FQDNs to `192.168.65.3` (the VM node IP), which is unreachable from WSL2 without going
   through the NodePort proxy.

### Worker node IP

The Ray worker running with `--network host` auto-detects its IP as `192.168.65.3` (the Docker
Desktop node IP). This is normal — `192.168.65.3:30379` is reachable from within the Docker
Desktop VM network, and the NodePort at that address routes to the Ray head pod.

---

## Networking — RunPod

- TCP ports use `supportPublicIp: true` → RunPod assigns a `publicIp` at the pod level (NOT
  inside `machine{}`).
- `portMappings` maps internal container port → external public port:
  `{"6379": 10341}` means Ray GCS is reachable at `publicIp:10341`, not at `publicIp:6379`.
- `wrapArgsForRunPod` injects `--node-ip-address=$MYIP` (discovered via `curl api.ipify.org`)
  so the Ray head can route tasks back to the worker's public IP.
- External port mappings are surfaced as pod annotations: `vk-ml/port-6379=10341`,
  `vk-ml/public-ip=100.65.0.119`.
- The Ray head must be exposed as `NodePort` or `LoadBalancer` for RunPod workers to connect.

---

## Quick start

### Prerequisites

- Go 1.24+
- Docker (daemon running)
- `kubectl` configured against a local cluster (k3s, Docker Desktop K8s, kind)

### Local debug mode (Docker Desktop)

```bash
# 1. Start the virtual kubelet
KUBECONFIG=~/.kube/config go run ./cmd/vk-ml \
  --compute-provider=local \
  --node-name=vk-ml-local \
  --namespace=ray

# 2. Verify the virtual node appears in the cluster
kubectl get nodes
# NAME           STATUS   ROLES   AGE
# docker-desktop Ready    master  ...
# vk-ml-local    Ready    agent   5s   ← virtual node

# 3. Apply the RayJob (includes NodePort service with correct selector)
kubectl apply -f ../k3s/kuberay/kuberay-job-gpu.yaml

# 4. Watch the Docker worker — GCS wait loop then ray start
docker logs $(docker ps --filter label=vk-ml=true -q) -f
# 0 seconds elapsed: Waiting for GCS to be ready.
# 5 seconds elapsed: Waiting for GCS to be ready.
# GCS is ready.
# ...
# Ray runtime started.

# 5. Verify worker joined the Ray cluster
kubectl exec -n ray -l ray.io/node-type=head -c ray-head -- \
  python3 -c "import ray; ray.init(address='auto',ignore_reinit_error=True); print(ray.cluster_resources())"
# {'GPU': 1.0, 'CPU': 3.0, 'node:192.168.65.3': 1.0, ...}

# 6. Full delete+apply cycle (regression test)
kubectl delete -f ../k3s/kuberay/kuberay-job-gpu.yaml
kubectl apply  -f ../k3s/kuberay/kuberay-job-gpu.yaml
kubectl get endpoints ray-gpu-test-head -n ray   # must auto-populate once head is Ready
```

### RunPod mode

```bash
export RUNPOD_API_KEY=your_key_here

KUBECONFIG=~/.kube/config go run ./cmd/vk-ml \
  --compute-provider=runpod \
  --node-name=vk-ml-runpod \
  --namespace=ray

kubectl apply -f ../k3s/kuberay/kuberay-job-gpu.yaml
kubectl get pods -n ray -w   # Pending → Running as RunPod provisions the GPU
```

---

## Configuration reference

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--compute-provider` | `local` | Backend: `runpod` or `local` (Docker) |
| `--node-name` | `vk-ml-runpod` | Name of the virtual node in Kubernetes |
| `--namespace` | `ray` | Namespace to watch for worker pods |
| `--node-cpu` | `1000` | Advertised CPU capacity (K8s quantity) |
| `--node-memory` | `10Ti` | Advertised memory capacity |
| `--node-gpu` | `100` | Advertised `nvidia.com/gpu` capacity |
| `--reconcile-interval` | `15s` | How often to poll instance statuses |
| `--kubeconfig` | `~/.kube/config` | Path to kubeconfig (empty = in-cluster) |
| `--health-addr` | `:8080` | Listen address for `/healthz` and `/readyz` |

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RUNPOD_API_KEY` | For `runpod` provider | RunPod API key (never pass as a flag) |

---

## KubeRay worker group configuration

For KubeRay to schedule worker pods to the virtual node, add `nodeSelector` and `tolerations`
to the worker group spec in your `RayCluster` or `RayJob`:

```yaml
workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 1
    minReplicas: 0
    maxReplicas: 4
    rayStartParams:
      num-gpus: "1"
      num-cpus: "2"
    template:
      spec:
        # Target the virtual node by its hostname label.
        # Change to match the --node-name flag when starting vk-ml.
        nodeSelector:
          kubernetes.io/hostname: vk-ml-local
        tolerations:
          # Required to schedule onto the virtual node.
          - key: virtual-kubelet.io/provider
            operator: Exists
            effect: NoSchedule
          # Required if real GPU nodes also have this taint.
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
        containers:
          - name: ray-worker
            image: ray-cluster-gpu:2.53.0
            resources:
              requests:
                cpu: "2"
                memory: "8Gi"
                nvidia.com/gpu: "1"
              limits:
                nvidia.com/gpu: "1"
```

---

## Adding a new compute provider

1. Create `pkg/compute/<name>/provider.go` implementing the `ComputeProvider` interface:

```go
type ComputeProvider interface {
    CreateInstance(ctx context.Context, spec InstanceSpec) (*Instance, error)
    DeleteInstance(ctx context.Context, id string) error
    GetInstance(ctx context.Context, id string) (*Instance, error)
    ListInstances(ctx context.Context) ([]Instance, error)
}
```

2. Add a `case` to the `switch cfg.ComputeProvider` block in `cmd/vk-ml/main.go`:

```go
case "lambda":
    cp = lambdapkg.NewLambdaProvider(cfg.LambdaAPIKey)
```

3. Add any new config fields to `pkg/config/config.go`.

No other files need to change. The GCS wait logic, status translation, and VK provider
interfaces are all provider-agnostic.
