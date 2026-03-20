package runpod

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/ml-platform/vk-ml/pkg/compute"
)

// RunPodProviderConfig holds RunPod-specific pod creation parameters that are
// applied to every CreatePodRequest. Zero values fall back to the same defaults
// the provider used before this config struct was introduced.
type RunPodProviderConfig struct {
	// GPU selection
	GpuTypeIDs      []string // e.g. ["NVIDIA RTX 2000 Ada Generation"]; nil = RunPod auto-select
	GpuTypePriority string   // "availability" or "price"; empty = RunPod default

	// Compute location
	DataCenterIDs []string // restrict to specific data centers; nil = all eligible

	// Resource constraints (0 = omit field, no constraint)
	ContainerDiskGB      int // default 50 when 0
	MinDiskBandwidthMBps int
	MinDownloadMbps      int

	// Cost / reliability
	CloudType     string // "SECURE" or "COMMUNITY"; default "SECURE" when empty
	Interruptible bool   // allow spot/preemptible instances; default false
}

// instanceRecord tracks the RunPod pod ID for a given Kubernetes pod name.
type instanceRecord struct {
	runpodID string
	instance compute.Instance
}

// RunPodProvider implements compute.ComputeProvider using the RunPod REST API.
// It translates InstanceSpec → RunPod CreatePodRequest and maps RunPod pod
// statuses back to the canonical compute.InstanceStatus values.
type RunPodProvider struct {
	client *Client
	podCfg RunPodProviderConfig

	// Tailscale networking fields (zero-value = disabled, legacy public-IP mode).
	tailscaleEnabled bool
	tailscaleAuthKey string          // static fallback key; used when syncer is nil or key generation fails
	tailscaleAddr    string          // MagicDNS name:port of Ray head, e.g. "ray-head.ts.net:6379"
	syncer           *TailscaleSyncer // nil when TAILSCALE_API_KEY is not configured

	mu        sync.Mutex
	instances map[string]instanceRecord // key: podName
}

// NewRunPodProvider creates a RunPodProvider in legacy public-IP mode.
func NewRunPodProvider(apiKey string, podCfg RunPodProviderConfig) *RunPodProvider {
	return &RunPodProvider{
		client:    NewClient(apiKey),
		podCfg:    podCfg,
		instances: make(map[string]instanceRecord),
	}
}

// NewRunPodProviderWithTailscale creates a RunPodProvider that configures workers
// to join a Tailscale tailnet and connect to the Ray head via a private Tailscale
// address instead of public IPs. Workers discover the head via MagicDNS (tsAddr).
//
// syncer may be nil. When non-nil, CreateInstance calls syncer.GenerateEphemeralKey
// to mint a fresh single-use auth key per worker instead of using the static tsAuthKey.
// This avoids injecting a long-lived shared secret into every RunPod container.
// If key generation fails the call falls back to tsAuthKey rather than failing.
//
// The K3s node must be configured as a Tailscale subnet router advertising the
// pod and service CIDRs so workers can reach ClusterIPs directly after joining.
func NewRunPodProviderWithTailscale(apiKey, tsAuthKey, tsAddr string, syncer *TailscaleSyncer, podCfg RunPodProviderConfig) *RunPodProvider {
	return &RunPodProvider{
		client:           NewClient(apiKey),
		podCfg:           podCfg,
		instances:        make(map[string]instanceRecord),
		tailscaleEnabled: true,
		tailscaleAuthKey: tsAuthKey,
		tailscaleAddr:    tsAddr,
		syncer:           syncer,
	}
}

// CreateInstance translates the InstanceSpec to a RunPod CreatePodRequest and
// launches the pod via the RunPod REST API.
//
// GPU resource mapping:
//   - GpuCount      = spec.GPU  (at least 1)
//   - MinVCPUPerGPU = spec.CPU / gpuCount  (vCPUs per GPU, not total)
//   - MinRAMPerGPU  = spec.MemoryGB / gpuCount  (GB RAM per GPU, not total)
//
// In Tailscale mode (tailscaleEnabled=true):
//   - K8s service DNS names in env vars and args are rewritten to the Tailscale
//     MagicDNS address so workers can reach the Ray head after joining the tailnet.
//   - The startup args are wrapped with a Tailscale bootstrap block that starts
//     tailscaled in userspace mode, joins the tailnet, and injects --node-ip-address
//     using the worker's own Tailscale IP (replacing the legacy ipify public-IP approach).
//   - No public ports are opened; SupportPublicIp is set to false.
//
// In legacy mode (tailscaleEnabled=false):
//   - wrapArgsForRunPod injects --node-ip-address via curl api.ipify.org.
//   - SupportPublicIp=true is required for TCP inbound on Community Cloud.
func (p *RunPodProvider) CreateInstance(ctx context.Context, spec compute.InstanceSpec) (*compute.Instance, error) {
	gpuCount := maxInt(spec.GPU, 1)

	var wrappedArgs  []string
	var env          map[string]string
	ports := []string{} // FIX: never nil
	supportPublicIP := false

	if p.tailscaleEnabled {
		// Rewrite *.svc.cluster.local DNS names in env vars and args to the
		// Tailscale MagicDNS address. This must happen before wrapArgsWithTailscale
		// so both $RAY_ADDRESS (used by the GCS wait loop) and the hardcoded address
		// in the ray start command point to the Tailscale-reachable head.
		env  = rewriteK8sAddressesForTailscale(spec.Env, p.tailscaleAddr)
		args := rewriteK8sArgsForTailscale(spec.Args, p.tailscaleAddr)

		// Prefer a fresh per-worker ephemeral key (single-use, auto-expires, auto-removed
		// from tailnet admin on container exit) over the static shared key.
		// Fall back to the static key if the API call fails — worker creation still proceeds.
		authKey := p.tailscaleAuthKey
		if p.syncer != nil {
			if key, err := p.syncer.GenerateEphemeralKey(ctx); err == nil {
				authKey = key
			}
			// If key generation fails we intentionally continue with the static key.
			// The caller (main.go / reconcileLoop) will surface errors via logs;
			// here we avoid blocking a GPU worker launch over a Tailscale API hiccup.
		}

		env["TAILSCALE_AUTH_KEY"] = authKey
		env["WORKER_NAME"]        = sanitizeName(spec.PodName)
		wrappedArgs = wrapArgsWithTailscale(args)
		// No public ports: all Ray traffic flows over the Tailscale WireGuard tunnel.
	} else {
		env             = spec.Env
		wrappedArgs     = wrapArgsForRunPod(spec.Args) // legacy: ipify public-IP injection
		
		if spec.Ports != nil {
        ports = spec.Ports
		} else {
			ports = []string{}
		}

		supportPublicIP = true                          // Community Cloud: required for TCP inbound
	}

	diskGB := p.podCfg.ContainerDiskGB
	if diskGB == 0 {
		diskGB = 50 // preserve original default
	}
	cloudType := p.podCfg.CloudType
	if cloudType == "" {
		cloudType = "SECURE" // preserve original default
	}

	req := CreatePodRequest{
		Name:                 sanitizeName(spec.PodName),
		ImageName:            spec.Image,
		GpuCount:             gpuCount,
		GpuTypeIds:           p.podCfg.GpuTypeIDs,
		GpuTypePriority:      p.podCfg.GpuTypePriority,
		MinVCPUPerGPU:        maxInt(spec.CPU/gpuCount, 1),
		MinRAMPerGPU:         maxInt(spec.MemoryGB/gpuCount, 4),
		ContainerDiskInGb:    diskGB,
		DataCenterIds:        p.podCfg.DataCenterIDs,
		Interruptible:        p.podCfg.Interruptible,
		MinDiskBandwidthMBps: p.podCfg.MinDiskBandwidthMBps,
		MinDownloadMbps:      p.podCfg.MinDownloadMbps,
		Env:                  env,
		Ports:                ports,           // nil in Tailscale mode
		DockerEntrypoint:     spec.Command,
		DockerStartCmd:       wrappedArgs,
		SupportPublicIp:      supportPublicIP, // false in Tailscale mode
		CloudType:            cloudType,
	}

	resp, err := p.client.CreatePod(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("runpod create pod %q: %w", spec.PodName, err)
	}

	inst := compute.Instance{
		ID:        resp.ID,
		PodName:   spec.PodName,
		Namespace: spec.Namespace,
		Status:    compute.StatusStarting,
		StartedAt: time.Now(),
	}

	p.mu.Lock()
	p.instances[spec.PodName] = instanceRecord{runpodID: resp.ID, instance: inst}
	p.mu.Unlock()

	return &inst, nil
}

// DeleteInstance stops the RunPod pod identified by id (RunPod pod ID, not pod name).
func (p *RunPodProvider) DeleteInstance(ctx context.Context, id string) error {
	if err := p.client.StopPod(ctx, id); err != nil {
		// Treat "not found" as success — already gone.
		if strings.Contains(err.Error(), "404") || strings.Contains(err.Error(), "not found") {
			return nil
		}
		return fmt.Errorf("runpod stop pod %s: %w", id, err)
	}
	return nil
}

// GetInstance fetches the current state of a RunPod pod and translates it
// to a compute.Instance. Returns StatusNotFound if the pod no longer exists.
func (p *RunPodProvider) GetInstance(ctx context.Context, id string) (*compute.Instance, error) {
	resp, err := p.client.GetPod(ctx, id)
	if err != nil {
		if strings.Contains(err.Error(), "404") {
			return &compute.Instance{ID: id, Status: compute.StatusNotFound}, nil
		}
		return nil, fmt.Errorf("runpod get pod %s: %w", id, err)
	}
	return translatePodResponse(resp), nil
}

// ListInstances returns the current state of all tracked RunPod instances.
func (p *RunPodProvider) ListInstances(ctx context.Context) ([]compute.Instance, error) {
	p.mu.Lock()
	records := make([]instanceRecord, 0, len(p.instances))
	for _, r := range p.instances {
		records = append(records, r)
	}
	p.mu.Unlock()

	result := make([]compute.Instance, 0, len(records))
	for _, r := range records {
		inst, err := p.GetInstance(ctx, r.runpodID)
		if err != nil {
			continue
		}
		inst.PodName = r.instance.PodName
		inst.Namespace = r.instance.Namespace
		result = append(result, *inst)
	}
	return result, nil
}

// ── translation ───────────────────────────────────────────────────────────────

// translatePodResponse converts a RunPod API response to a compute.Instance.
//
// Networking fields:
//   - PublicIp (top-level) → Instance.PublicIP  (available once RUNNING)
//   - PortMappings         → Instance.PortMappings  ({"6379": 10341, ...})
//
// The external port for Ray GCS is: Instance.PublicIP:Instance.PortMappings["6379"]
func translatePodResponse(r *PodResponse) *compute.Instance {
	inst := &compute.Instance{
		ID:           r.ID,
		PublicIP:     r.PublicIp,     // top-level field, populated when RUNNING
		PortMappings: r.PortMappings, // nil until pod is RUNNING and ports are mapped
	}

	// RunPod returns both DesiredStatus and CurrentStatus.
	// CurrentStatus reflects actual state; fall back to DesiredStatus if empty.
	status := r.CurrentStatus
	if status == "" {
		status = r.DesiredStatus
	}

	switch strings.ToUpper(status) {
	case "RUNNING":
		inst.Status = compute.StatusRunning
	case "STARTING", "CREATED":
		inst.Status = compute.StatusStarting
	case "EXITED":
		inst.Status = compute.StatusExited
		if r.Runtime != nil && r.Runtime.Container != nil {
			inst.ExitCode = r.Runtime.Container.ExitCode
		}
	case "TERMINATED", "TERMINATING":
		inst.Status = compute.StatusTerminated
	case "":
		inst.Status = compute.StatusStarting // just created, status not set yet
	default:
		inst.Status = compute.StatusUnknown
	}

	return inst
}

// ── helpers ───────────────────────────────────────────────────────────────────

// wrapArgsForRunPod modifies the Ray start command to dynamically inject
// --node-ip-address using the container's own public IP.
//
// Problem: KubeRay generates pod args like:
//
//	["ulimit -n 65536; ray start --address=HEAD:6379 --block"]
//
// But the Ray head needs to reach back to this worker using its public IP,
// not the container's internal IP. RunPod assigns a public IP only after
// the pod starts, so we can't know it at creation time.
//
// Solution: inject a shell snippet that discovers the public IP at container
// startup and injects it into the ray start command:
//
//	MYIP=$(curl -s https://api.ipify.org || hostname -I | awk '{print $1}')
//	ray start --address=HEAD:6379 --node-ip-address=$MYIP --block
func wrapArgsForRunPod(args []string) []string {
	if len(args) == 0 {
		return args
	}

	// args[0] holds the shell command string (KubeRay uses /bin/bash -c -- convention)
	original := args[0]

	// Only inject if this looks like a ray start command.
	if !strings.Contains(original, "ray start") {
		return args
	}

	ipDiscovery := `MYIP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || hostname -I | awk '{print $1}'); `
	wrapped := ipDiscovery + strings.Replace(original, "ray start ", "ray start --node-ip-address=$MYIP ", 1)

	result := make([]string, len(args))
	copy(result, args)
	result[0] = wrapped
	return result
}

// wrapArgsWithTailscale prepends a Tailscale bootstrap block to the startup
// script in args[0], replacing the legacy public-IP approach (wrapArgsForRunPod).
//
// Execution order inside the RunPod container after this wrapping:
//  1. Tailscale bootstrap: start tailscaled (userspace mode) → join tailnet → wait for TS_IP
//  2. GCS wait loop (already prepended by WrapArgsWithGCSWait): ray health-check using
//     the rewritten $RAY_ADDRESS (Tailscale MagicDNS name, not K8s svc DNS)
//  3. ray start --address=<tailscale-addr> --node-ip-address=$TS_IP --block
//
// TAILSCALE_AUTH_KEY and WORKER_NAME are injected as container env vars by CreateInstance.
// --accept-routes enables subnet routing so workers can reach K8s ClusterIPs via the
// K3s node's Tailscale subnet advertisement (10.42.0.0/16, 10.43.0.0/16).
// Ephemeral behavior (auto-remove on disconnect) is governed by the auth key type
// (generated via Tailscale API with "ephemeral":true), NOT by a --ephemeral CLI flag.
func wrapArgsWithTailscale(args []string) []string {
	if len(args) == 0 || !strings.Contains(args[0], "ray start") {
		return args
	}

	tailscaleBlock := `
# ── Tailscale bootstrap ──────────────────────────────────────────────────────
# Userspace networking mode: no CAP_NET_ADMIN or /dev/net/tun required.
# TAILSCALE_AUTH_KEY and WORKER_NAME are injected by the VK provider.
export TAILSCALE_SOCKET=/tmp/tailscaled.sock

tailscaled --tun=userspace-networking --socket="$TAILSCALE_SOCKET" &
TS_DAEMON_PID=$!

# Wait for the daemon socket to appear (max 30 s).
for i in $(seq 1 30); do
  [ -S "$TAILSCALE_SOCKET" ] && break
  echo "[tailscale] Waiting for daemon... ($i/30)"; sleep 1
done
[ -S "$TAILSCALE_SOCKET" ] || { echo "ERROR: tailscaled socket not ready"; exit 1; }

# Join the tailnet.
# --accept-routes: enables K8s ClusterIP reachability via K3s subnet router.
tailscale --socket="$TAILSCALE_SOCKET" up \
  --authkey="$TAILSCALE_AUTH_KEY" \
  --hostname="${WORKER_NAME:-runpod-worker}" \
  --accept-routes \
  --accept-dns=true

# Wait for Tailscale IP assignment (max 60 s).
TS_IP=""
for i in $(seq 1 60); do
  TS_IP=$(tailscale --socket="$TAILSCALE_SOCKET" ip -4 2>/dev/null | head -1)
  [ -n "$TS_IP" ] && break
  echo "[tailscale] Waiting for IP... ($i/60)"; sleep 1
done
[ -n "$TS_IP" ] || { echo "ERROR: No Tailscale IP after 60s"; exit 1; }
echo "[tailscale] IP=$TS_IP — connected to tailnet"
# ── End Tailscale bootstrap ───────────────────────────────────────────────────
`
	// Inject --node-ip-address=$TS_IP so the Ray head can reach this worker
	// back through the tailnet. $TS_IP is exported by the bootstrap above.
	modified := strings.Replace(args[0], "ray start ", "ray start --node-ip-address=$TS_IP ", 1)

	result := make([]string, len(args))
	copy(result, args)
	result[0] = tailscaleBlock + modified
	return result
}

// k8sDNSPatternRP matches Kubernetes service DNS names with an optional :PORT suffix.
// Format: <service>.<namespace>.svc.cluster.local[:<port>]
// Used to rewrite K8s-internal addresses to Tailscale-reachable addresses.
var k8sDNSPatternRP = regexp.MustCompile(
	`[a-z0-9][a-z0-9-]*\.[a-z0-9][a-z0-9-]*\.svc\.cluster\.local(?::\d+)?`,
)

// rewriteK8sAddressesForTailscale returns a new env map with all K8s service DNS
// names replaced by the Tailscale MagicDNS address (addr).
// The input map is not modified.
func rewriteK8sAddressesForTailscale(env map[string]string, addr string) map[string]string {
	out := make(map[string]string, len(env))
	for k, v := range env {
		out[k] = k8sDNSPatternRP.ReplaceAllString(v, addr)
	}
	return out
}

// rewriteK8sArgsForTailscale returns a new args slice with all K8s service DNS
// names (including those embedded in shell scripts in args[0]) replaced by addr.
func rewriteK8sArgsForTailscale(args []string, addr string) []string {
	out := make([]string, len(args))
	for i, a := range args {
		out[i] = k8sDNSPatternRP.ReplaceAllString(a, addr)
	}
	return out
}

// sanitizeName lowercases and truncates to 63 characters to satisfy both
// Kubernetes naming rules and RunPod's name length limit.
func sanitizeName(name string) string {
	s := strings.ToLower(name)
	// Replace characters not allowed in RunPod names
	s = strings.NewReplacer("_", "-", ".", "-").Replace(s)
	if len(s) > 63 {
		s = s[:63]
	}
	// Trim trailing hyphens
	s = strings.TrimRight(s, "-")
	return s
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
