// Package config loads all configuration from CLI flags and environment variables.
package config

import (
	"flag"
	"os"
	"time"
)

// Config holds all runtime configuration for the virtual kubelet.
type Config struct {
	// ComputeProvider selects the backend: "runpod" or "local".
	ComputeProvider string

	// RunPod credentials — loaded from the RUNPOD_API_KEY environment variable.
	// Required when ComputeProvider is "runpod".
	RunPodAPIKey string

	// Tailscale networking for RunPod workers.
	// When TailscaleEnabled is true, workers join a private tailnet instead of
	// using public IPs and open TCP ports.
	TailscaleEnabled     bool
	RayHeadTailscaleAddr string // e.g. "ray-head.tailnet.ts.net:6379"; auto-derived from RayHeadDNSRecord if empty
	TailscaleAuthKey     string // loaded from TAILSCALE_AUTH_KEY env var (never a flag); fallback when API key absent

	// Tailscale management API — enables automatic DNS sync and per-worker ephemeral keys.
	// All three must be set together; if TailscaleAPIKey is empty the features are skipped
	// and the static TailscaleAuthKey is used instead.
	TailscaleAPIKey    string // loaded from TAILSCALE_API_KEY env var (never a flag)
	TailnetName        string // --tailnet-name: Tailscale org name, e.g. "example@github" or "-"
	RayHeadServiceName string // --ray-head-service: K8s service whose ClusterIP is synced to DNS
	RayHeadDNSRecord   string // --ray-head-dns-record: full hostname, e.g. "ray-head.example.ts.net"

	// Kubernetes client configuration.
	// Leave empty to use in-cluster ServiceAccount credentials.
	Kubeconfig string
	// Namespace is the Kubernetes namespace to watch for pods.
	// KubeRay typically creates worker pods in the same namespace as the RayCluster.
	Namespace string

	// Virtual node identity and advertised capacity.
	NodeName   string // registered node name in Kubernetes
	NodeCPU    string // total CPU capacity (Kubernetes quantity string, e.g. "1000")
	NodeMemory string // total memory capacity (e.g. "10Ti")
	NodeGPU    string // total nvidia.com/gpu capacity (e.g. "100")

	// ReconcileInterval controls how often the provider polls instance statuses
	// and pushes pod status updates to Kubernetes.
	ReconcileInterval time.Duration

	// HealthAddr is the listen address for /healthz and /readyz HTTP endpoints.
	HealthAddr string

	// RunPod pod creation parameters.
	// These are passed as-is to the RunPod REST API for every worker pod created.
	// Zero/empty values fall back to the same defaults as before these flags existed.
	RunPodGPUType         string // --runpod-gpu-type
	RunPodGPUPriority     string // --runpod-gpu-priority
	RunPodDataCenterIDs   string // --runpod-datacenter-ids (comma-separated)
	RunPodCloudType       string // --runpod-cloud-type
	RunPodDiskGB          int    // --runpod-disk-gb
	RunPodInterruptible   bool   // --runpod-interruptible
	RunPodMinDiskMBps     int    // --runpod-min-disk-mbps
	RunPodMinDownloadMbps int    // --runpod-min-download-mbps
}

// Load parses CLI flags and reads environment variables, returning the
// populated Config. flag.Parse() is called internally.
func Load() *Config {
	cfg := &Config{}

	flag.StringVar(&cfg.ComputeProvider, "compute-provider", "local",
		"Compute backend to use: 'runpod' or 'local' (Docker)")
	flag.StringVar(&cfg.Kubeconfig, "kubeconfig", defaultKubeconfig(),
		"Path to kubeconfig file. Leave empty for in-cluster mode.")
	flag.StringVar(&cfg.Namespace, "namespace", "ray",
		"Kubernetes namespace to watch for worker pods")
	flag.StringVar(&cfg.NodeName, "node-name", "vk-ml-runpod",
		"Name of the virtual node registered in Kubernetes")
	flag.StringVar(&cfg.NodeCPU, "node-cpu", "1000",
		"Total CPU capacity advertised by the virtual node (Kubernetes quantity)")
	flag.StringVar(&cfg.NodeMemory, "node-memory", "10Ti",
		"Total memory capacity advertised by the virtual node")
	flag.StringVar(&cfg.NodeGPU, "node-gpu", "100",
		"Total nvidia.com/gpu capacity advertised by the virtual node")
	flag.DurationVar(&cfg.ReconcileInterval, "reconcile-interval", 15*time.Second,
		"How often to poll instance statuses and push pod status updates")
	flag.StringVar(&cfg.HealthAddr, "health-addr", ":8080",
		"Listen address for /healthz and /readyz health endpoints")
	flag.BoolVar(&cfg.TailscaleEnabled, "tailscale-enabled", false,
		"Enable Tailscale networking for RunPod workers (replaces public IP + open ports)")
	flag.StringVar(&cfg.RayHeadTailscaleAddr, "ray-head-tailscale-addr", "",
		"Tailscale MagicDNS name:port for Ray head, e.g. ray-head.ts.net:6379. Auto-derived from --ray-head-dns-record if empty.")
	flag.StringVar(&cfg.TailnetName, "tailnet-name", "-",
		`Tailscale org name used in API URLs, e.g. "example@github". Use "-" for the default tailnet of the API key.`)
	flag.StringVar(&cfg.RayHeadServiceName, "ray-head-service", "ray-gpu-test-head",
		"Kubernetes service name whose ClusterIP is synced to the Tailscale DNS record at startup.")
	flag.StringVar(&cfg.RayHeadDNSRecord, "ray-head-dns-record", "",
		"Full Tailscale MagicDNS hostname for the Ray head, e.g. ray-head.example.ts.net. "+
			"When set, vk-ml creates/updates an A record pointing to the service ClusterIP. "+
			"Also used to derive --ray-head-tailscale-addr when that flag is omitted.")

	// ── RunPod pod creation parameters ───────────────────────────────────────
	flag.StringVar(&cfg.RunPodGPUType, "runpod-gpu-type", "NVIDIA RTX 2000 Ada Generation",
		`RunPod GPU type ID to request (e.g. "NVIDIA RTX 2000 Ada Generation"). `+
			`Set to "" to let RunPod auto-select by cost/availability.`)
	flag.StringVar(&cfg.RunPodGPUPriority, "runpod-gpu-priority", "availability",
		`GPU selection priority: "availability" (default) or "price".`)
	flag.StringVar(&cfg.RunPodDataCenterIDs, "runpod-datacenter-ids", "",
		`Comma-separated RunPod data center IDs to restrict to (e.g. "US-TX-3,EU-RO-1"). `+
			`"" = all data centers eligible.`)
	flag.StringVar(&cfg.RunPodCloudType, "runpod-cloud-type", "SECURE",
		`RunPod cloud type: "SECURE" (default) or "COMMUNITY".`)
	flag.IntVar(&cfg.RunPodDiskGB, "runpod-disk-gb", 50,
		`Container disk size in GB.`)
	flag.BoolVar(&cfg.RunPodInterruptible, "runpod-interruptible", false,
		`Allow spot/interruptible RunPod instances (cheaper, may be preempted).`)
	flag.IntVar(&cfg.RunPodMinDiskMBps, "runpod-min-disk-mbps", 0,
		`Minimum disk bandwidth in MB/s. 0 = no constraint.`)
	flag.IntVar(&cfg.RunPodMinDownloadMbps, "runpod-min-download-mbps", 0,
		`Minimum download speed in Mbps. 0 = no constraint.`)

	flag.Parse()

	// Credentials are always loaded from environment variables (not flags)
	// so they are never accidentally exposed in shell history or logs.
	cfg.RunPodAPIKey     = os.Getenv("RUNPOD_API_KEY")
	cfg.TailscaleAuthKey = os.Getenv("TAILSCALE_AUTH_KEY")
	cfg.TailscaleAPIKey  = os.Getenv("TAILSCALE_API_KEY")

	return cfg
}

func defaultKubeconfig() string {
	// In-cluster mode: no kubeconfig needed (uses pod ServiceAccount).
	// Local mode: use standard ~/.kube/config location.
	if h := os.Getenv("HOME"); h != "" {
		return h + "/.kube/config"
	}
	return ""
}
