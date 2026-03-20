# Virtual Kubelet + RunPod + Tailscale

End-to-end setup guide for launching Ray GPU workers on RunPod through a Virtual Kubelet provider, using Tailscale as the private network layer.

---

## Architecture

```
Kubernetes (k3s)
├── vk-ml  (Virtual Kubelet — kube-system)
│     ↓  schedules worker pods → RunPod API
├── Ray head pod  (namespace: ray)
│     └── ClusterIP service ray-gpu-test-head :6379
│
└── Tailscale subnet router (K3s node)
      advertises  10.42.0.0/16  (pod CIDR)
      advertises  10.43.0.0/16  (service CIDR)

Tailscale tailnet
├── K3s node          100.x.x.x   ← subnet router
└── RunPod worker     100.y.y.y   ← ephemeral, auto-removed on exit
      tailscaled (userspace mode, no CAP_NET_ADMIN)
      ray start --address=ray-head.<tailnet>.ts.net:6379
                --node-ip-address=100.y.y.y
```

All Ray traffic (GCS registration, raylet, object store) flows over WireGuard.
No public TCP ports are opened on RunPod containers.

---

## Prerequisites

| Component | Version |
|-----------|---------|
| k3s | ≥ 1.28 |
| KubeRay operator | ≥ 1.1 |
| Ray | 2.53.0 |
| Go | 1.24 (to build vk-ml) |
| Docker | any recent (to build worker image) |
| Tailscale account | any plan with MagicDNS enabled |

---

## Step 1 — Tailscale: install and configure on the K3s node

### 1.1 Install Tailscale

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

### 1.2 Authenticate and enable subnet routing

Find your pod and service CIDRs first:

```bash
# Pod CIDR (default k3s: 10.42.0.0/16)
kubectl get nodes -o jsonpath='{.items[0].spec.podCIDR}'
#Pod CIDR (default docker-desktop kubeadm: 10.1.0.0/16)

# Service CIDR (look for --service-cidr in k3s args, default: 10.43.0.0/16)
ps aux | grep k3s | grep -o 'service-cidr=[^ ]*'
# Service CIDR (default k3s: 10.96.0.0/12)
kubectl get svc kubernetes -o jsonpath='{.spec.clusterIP}'
```

Enable subnet routing and accept routes from other tailnet peers:

```bash
tailscale up \
  --advertise-routes=10.42.0.0/16,10.43.0.0/16 \
  --accept-routes

#(Docker Desktop kubeadm)
tailscale up \
  --advertise-routes=10.1.0.0/16,10.96.0.0/12 \
  --accept-routes
```

### 1.3 Approve routes in Tailscale admin console

1. Go to [admin.tailscale.com/machines](https://admin.tailscale.com/machines)
2. Find the K3s node → click the three-dot menu → **Edit route settings**
3. Enable both advertised routes (`10.42.0.0/16` and `10.43.0.0/16`)

### 1.4 Enable MagicDNS

In [admin.tailscale.com/dns](https://admin.tailscale.com/dns):

- **MagicDNS** → toggle on
- Note your **tailnet name** (shown on the DNS page, e.g. `example.ts.net`)

---

## Step 2 — Tailscale: generate API keys

You need two Tailscale keys. Generate them at [admin.tailscale.com/settings/keys](https://admin.tailscale.com/settings/keys).

### Auth key (`TAILSCALE_AUTH_KEY`)

Used as a static fallback when the API key is unavailable.

- **Reusable**: yes (or no — each worker will consume one use)
- **Ephemeral**: yes (devices removed automatically on disconnect)
- **Pre-authorized**: yes

Copy the key (`tskey-auth-...`).

### API key (`TAILSCALE_API_KEY`)

Enables automatic MagicDNS record management and per-worker single-use key generation.

- At [admin.tailscale.com/settings/keys](https://admin.tailscale.com/settings/keys) → **Generate API access token**
- Scope required: **read + write** (for devices and DNS)

Copy the key (`tskey-api-...`).

> **When both keys are present**: vk-ml mints a fresh ephemeral key per RunPod
> container at launch time (5-minute TTL, single-use). The static auth key is only
> used as fallback if the API call fails.

---

## Step 3 — RunPod: obtain API key

1. Log in to [runpod.io](https://www.runpod.io)
2. Settings → API Keys → **+ API Key**
3. Copy the key (`rpa_...`)

---

## Step 4 — Build the worker image

The worker image must include both Ray and Tailscale.

```bash
cd k3s/kuberay/gpu/

docker build -t ray-cluster-gpu:2.53.0 .

# Verify Tailscale is present
docker run --rm ray-cluster-gpu:2.53.0 tailscale version
```

Push to a registry that RunPod can pull from (Docker Hub, GHCR, etc.):

```bash
docker tag ray-cluster-gpu:2.53.0 takenking9879/ray-cluster-gpu:2.53.0
docker push takenking9879/ray-cluster-gpu:2.53.0
```

Update `image:` in `k3s/kuberay/kuberay-job-gpu.yaml` (both head and worker) to the pushed image.

---

## Step 5 — Create the Kubernetes secret

All secrets are stored in `vk-ml-secrets` in the `kube-system` namespace.

```bash
kubectl create secret generic vk-ml-secrets \
  --from-literal=RUNPOD_API_KEY=rpa_...            \
  --from-literal=TAILSCALE_AUTH_KEY=tskey-auth-... \
  --from-literal=TAILSCALE_API_KEY=tskey-api-...   \
  -n kube-system
```

Verify:

```bash
kubectl get secret vk-ml-secrets -n kube-system
```

> `TAILSCALE_AUTH_KEY` and `TAILSCALE_API_KEY` are both marked `optional: true`
> in the Deployment. If you only have a static auth key, omit `TAILSCALE_API_KEY`.
> Automatic DNS sync and per-worker key generation will be skipped.

---

## Step 6 — Configure and deploy vk-ml

### 6.1 Edit `deploy/vk-ml.yaml`

Replace the placeholder tailnet name in the `--ray-head-dns-record` flag:

```yaml
- --ray-head-dns-record=ray-head.REPLACE_TAILNET_NAME.ts.net
```

Your tailnet name is visible in [admin.tailscale.com/dns](https://admin.tailscale.com/dns) (e.g. `tail1234ab.ts.net` or `example.ts.net`).

The full args section should look like:

```yaml
args:
  - --compute-provider=runpod
  - --node-name=vk-ml-runpod
  - --namespace=ray
  - --reconcile-interval=15s
  - --node-cpu=1000
  - --node-memory=10Ti
  - --node-gpu=100
  - --tailscale-enabled=true
  - --tailnet-name=-
  - --ray-head-dns-record=ray-head.example.ts.net   # ← your tailnet name
  - --ray-head-service=ray-gpu-test-head
  # RunPod pod parameters
  - --runpod-gpu-type=NVIDIA RTX 2000 Ada Generation
  - --runpod-gpu-priority=availability
  - --runpod-cloud-type=SECURE
  - --runpod-disk-gb=50
  - --runpod-interruptible=false
```

`--ray-head-tailscale-addr` is automatically derived as `ray-head.example.ts.net:6379`.

### 6.2 Build and load the vk-ml binary

```bash
cd virtual-kubelet-ml-platform/

# For in-cluster deployment (distroless image):
docker build -t vk-ml:latest .
# Load into k3s:
docker save vk-ml:latest | sudo k3s ctr images import -

# For local development (run outside cluster):
go run ./cmd/vk-ml \
  --compute-provider=runpod \
  --node-name=vk-ml-runpod \
  --namespace=ray \
  --tailscale-enabled=true \
  --tailnet-name=- \
  --ray-head-dns-record=ray-head.example.ts.net \
  --ray-head-service=ray-gpu-test-head
```

### 6.3 Apply the manifest

```bash
kubectl apply -f virtual-kubelet-ml-platform/deploy/vk-ml.yaml
```

### 6.4 Verify the virtual node appears

```bash
kubectl get nodes
# Expected: vk-ml-runpod   Ready   agent   ...
```

### 6.5 Check startup logs

```bash
kubectl logs -n kube-system deployment/vk-ml -f
```

On a successful Tailscale DNS sync you will see:

```
level=INFO msg="Tailscale DNS record synced" record=ray-head.example.ts.net clusterIP=10.43.87.45
level=INFO msg="Using RunPod compute provider (Tailscale mode)" head-addr=ray-head.example.ts.net:6379 ephemeral-keys=true
level=INFO msg="Virtual node registered" node=vk-ml-runpod
level=INFO msg="PodController ready — virtual node is active" node=vk-ml-runpod
```

---

## Step 7 — Apply the RayJob

```bash
kubectl apply -f k3s/kuberay/kuberay-job-gpu.yaml
```

Watch pod lifecycle:

```bash
kubectl get pods -n ray -w
```

Expected sequence:

```
ray-gpu-test-job-xxxxx-head      Pending → Running
ray-gpu-test-job-xxxxx-worker-0  Pending → Running  (via vk-ml → RunPod)
```

---

## Step 8 — Verify connectivity

### 8.1 Tailscale peer appears on K3s node

```bash
tailscale status
# The RunPod worker should appear as a connected peer (100.y.y.y)
```

### 8.2 Ray worker registered

```bash
kubectl logs -n ray -l ray.io/node-type=head --tail=50 | grep -E "Added|node"
```

Or exec into the head pod:

```bash
kubectl exec -n ray <head-pod-name> -- ray status
```

Expected output includes the RunPod worker as a connected node with GPU resources.

### 8.3 GPU task executed

```bash
kubectl logs -n ray <head-pod-name> | tail -30
# Should show: "Training finished on GPU: NVIDIA GeForce RTX 4090"
```

---

## Configuration reference

### vk-ml flags (Tailscale-related)

| Flag | Default | Description |
|------|---------|-------------|
| `--tailscale-enabled` | `false` | Enable Tailscale mode (replaces public IP) |
| `--tailnet-name` | `-` | Tailscale org name in API URLs. `-` means the tailnet that owns `TAILSCALE_API_KEY` |
| `--ray-head-service` | `ray-gpu-test-head` | K8s service name whose ClusterIP is synced to MagicDNS |
| `--ray-head-dns-record` | `""` | Full MagicDNS hostname, e.g. `ray-head.example.ts.net`. vk-ml creates/updates the A record at startup |
| `--ray-head-tailscale-addr` | auto | `<dns-record>:6379` derived from `--ray-head-dns-record`. Set explicitly only if GCS port differs from 6379 |

### vk-ml flags (RunPod pod parameters)

These flags control every `POST /pods` request sent to the RunPod REST API. They are applied uniformly to all worker pods created by this vk-ml instance.

| Flag | Default | Description |
|------|---------|-------------|
| `--runpod-gpu-type` | `NVIDIA RTX 2000 Ada Generation` | GPU type ID to request. Must match the exact RunPod GPU type string (visible in the RunPod console or `GET /gpu-types`). Set to `""` to let RunPod auto-select by availability/cost |
| `--runpod-gpu-priority` | `availability` | GPU selection priority: `availability` (maximize chance of getting a pod) or `price` (prefer cheapest option) |
| `--runpod-cloud-type` | `SECURE` | Cloud type: `SECURE` (dedicated datacenter hardware) or `COMMUNITY` (community cloud, cheaper but shared) |
| `--runpod-disk-gb` | `50` | Container disk size in GB |
| `--runpod-interruptible` | `false` | Allow spot/interruptible instances. Cheaper but may be terminated mid-training |
| `--runpod-datacenter-ids` | `""` | Comma-separated RunPod data center IDs to restrict allocation to (e.g. `US-TX-3,EU-RO-1`). Empty = all data centers eligible |
| `--runpod-min-disk-mbps` | `0` | Minimum disk bandwidth in MB/s. `0` = no constraint |
| `--runpod-min-download-mbps` | `0` | Minimum network download speed in Mbps. `0` = no constraint |

> **GPU type IDs** must match exactly the strings used by RunPod (e.g. `"NVIDIA RTX 2000 Ada Generation"`, `"NVIDIA GeForce RTX 4090"`). You can find valid IDs in the RunPod web console when creating a pod manually, or via the RunPod API (`GET /gpu-types`). If the requested GPU type is unavailable in your selected data centers, RunPod will return an error — consider setting `--runpod-gpu-priority=availability` or broadening `--runpod-datacenter-ids`.

### vk-ml environment variables (credentials — never flags)

| Variable | Required | Description |
|----------|----------|-------------|
| `RUNPOD_API_KEY` | yes | RunPod REST API key (`rpa_...`) |
| `TAILSCALE_AUTH_KEY` | if no API key | Static Tailscale auth key (`tskey-auth-...`). Used as fallback when `TAILSCALE_API_KEY` is absent or ephemeral key generation fails |
| `TAILSCALE_API_KEY` | recommended | Tailscale management API key (`tskey-api-...`). Enables automatic MagicDNS sync and per-worker ephemeral key generation |

---

## How automatic DNS sync works

At vk-ml startup (before any worker is created):

1. vk-ml calls the K8s API: `GET /api/v1/namespaces/ray/services/ray-gpu-test-head`
2. Extracts `spec.clusterIP` (e.g. `10.43.87.45`)
3. Calls Tailscale API: `GET /api/v2/tailnet/-/dns/records` to list existing records
4. If a record for `ray-head.example.ts.net` already points to `10.43.87.45` → no-op
5. If stale (wrong IP) → `DELETE /api/v2/tailnet/-/dns/records/{id}`
6. Creates: `POST /api/v2/tailnet/-/dns/records` `{ "name": "ray-head.example.ts.net", "type": "A", "value": "10.43.87.45" }`

Workers joining the tailnet with `--accept-dns=true` will immediately resolve `ray-head.example.ts.net` to `10.43.87.45`. Packets to that IP are routed through the K3s subnet router (port 6379 → kube-proxy → Ray head pod).

---

## How per-worker ephemeral keys work

At each `CreateInstance` call (one per RunPod container):

1. vk-ml calls `POST /api/v2/tailnet/-/keys` with `reusable=false`, `ephemeral=true`, `preauthorized=true`, `expirySeconds=300`
2. Tailscale returns a fresh `tskey-auth-...` key
3. vk-ml injects this key as `TAILSCALE_AUTH_KEY` in the RunPod container env
4. The container's `tailscale up --authkey=...` consumes the key and registers as a new tailnet device
5. When the container exits, Tailscale removes the device from the admin console automatically

If key generation fails (API error, network issue), the static `TAILSCALE_AUTH_KEY` is used as fallback and the worker is still created.

---

## Worker bootstrap sequence

Inside each RunPod container, the startup script runs in this order:

```
1. tailscaled --tun=userspace-networking --socket=/tmp/tailscaled.sock &
   (userspace mode — no CAP_NET_ADMIN or /dev/net/tun required)

2. Wait for daemon socket (max 30 s)

3. tailscale up --authkey=$TAILSCALE_AUTH_KEY
               --hostname=$WORKER_NAME
               --accept-routes        ← enables subnet routing (ClusterIP reachability)
               --accept-dns=true      ← resolves ray-head.<tailnet>.ts.net
               # ephemeral behavior comes from the key type, not a CLI flag

4. Wait for Tailscale IP (TS_IP, max 60 s)

5. GCS wait loop: ray health-check --address ray-head.example.ts.net:6379
   (retries every 5 s until the Ray head is ready or timeout)

6. ray start --address=ray-head.example.ts.net:6379
             --node-ip-address=$TS_IP   ← so the head can connect back through tailnet
             --num-gpus=1
             --block
```

---

## Troubleshooting

### Worker pod stuck in Pending

```bash
kubectl describe pod -n ray <worker-pod-name>
```

The pod is scheduled to `vk-ml-runpod` (virtual node). If it stays Pending, check vk-ml logs:

```bash
kubectl logs -n kube-system deployment/vk-ml
```

### Worker created on RunPod but Tailscale never connects

Check RunPod console for container logs. Common causes:

- `TAILSCALE_AUTH_KEY` is expired or already used (single-use key consumed)
- `tailscaled` socket timeout: the container may not have internet access (RunPod network issue)
- Ephemeral key generation failed and the static key was already used

Generate a fresh reusable auth key and recreate the secret.

### Ray worker connects but head cannot reach it

```bash
# On K3s node — verify the RunPod worker is a Tailscale peer
tailscale status | grep 100.

# Ping the worker Tailscale IP from the K3s node
tailscale ping 100.y.y.y
```

If ping fails, the worker may be behind symmetric NAT and relying on DERP. Check [admin.tailscale.com/machines](https://admin.tailscale.com/machines) — the worker should appear with a relay indicator. DERP adds ~10–50 ms latency but is fully functional for Ray.

### MagicDNS record not created

vk-ml logs will show a warning:

```
level=WARN msg="Tailscale DNS sync: could not update DNS record" ...
```

Workers fall back to using `--ray-head-tailscale-addr` directly. If that flag is also missing they will fail to connect. Manual fix:

```bash
# Find ClusterIP
kubectl get svc ray-gpu-test-head -n ray -o jsonpath='{.spec.clusterIP}'

# Add DNS record manually in Tailscale admin console:
# admin.tailscale.com → DNS → Extra records → Add
# Name: ray-head.example.ts.net   Type: A   Value: 10.43.87.45
```

### Tailscale subnet routes not working

Verify the K3s node is advertising routes and they are approved:

```bash
# On K3s node
tailscale status --json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['Self']['AllowedIPs'])"

# Expected to include: 10.42.0.0/16, 10.43.0.0/16
```

If routes are not approved, re-approve them in [admin.tailscale.com/machines](https://admin.tailscale.com/machines).

### Verify subnet routing from a worker (manual test)

```bash
# On the K3s node, simulate what a worker does:
# 1. Find the Ray head ClusterIP
CIP=$(kubectl get svc ray-gpu-test-head -n ray -o jsonpath='{.spec.clusterIP}')

# 2. Test connectivity (should reach Ray GCS)
nc -zv $CIP 6379
```

---

## Teardown

```bash
# Delete the RayJob (also terminates RunPod containers via vk-ml reconcile loop)
kubectl delete -f k3s/kuberay/kuberay-job-gpu.yaml

# Remove vk-ml
kubectl delete -f virtual-kubelet-ml-platform/deploy/vk-ml.yaml

# Remove the secret
kubectl delete secret vk-ml-secrets -n kube-system

# Remove tailnet subnet routes (optional)
tailscale up --advertise-routes="" --accept-routes=false
```

RunPod containers are automatically terminated when vk-ml receives the pod deletion event and calls the RunPod `DELETE /pods/{id}` API. Tailscale nodes registered with ephemeral keys disappear from the admin console within a few minutes of disconnecting.
