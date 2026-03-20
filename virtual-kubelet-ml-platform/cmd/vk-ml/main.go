// Command vk-ml is a Virtual Kubelet provider that bridges Kubernetes pod
// scheduling to external GPU compute backends (RunPod, local Docker).
//
// Startup sequence:
//  1. Load config from flags + environment variables
//  2. Build the Kubernetes client (needed by LocalProvider for DNS rewriting)
//  3. Instantiate the selected compute provider (RunPod or Local/Docker)
//  4. Register (or update) the virtual node object in Kubernetes
//  5. Set up pod and standard informers
//  6. Start the NodeController (manages node heartbeat and status in K8s)
//  7. Start the PodController (watches pods assigned to this node, calls provider)
//  8. Block until SIGINT / SIGTERM, then shut down gracefully
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"

	"github.com/virtual-kubelet/virtual-kubelet/node"

	localpkg "github.com/ml-platform/vk-ml/pkg/compute/local"
	runpodpkg "github.com/ml-platform/vk-ml/pkg/compute/runpod"
	"github.com/ml-platform/vk-ml/pkg/compute"
	"github.com/ml-platform/vk-ml/pkg/config"
	"github.com/ml-platform/vk-ml/pkg/provider"
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}))

	cfg := config.Load()
	log.Info("Starting vk-ml",
		"provider", cfg.ComputeProvider,
		"node", cfg.NodeName,
		"namespace", cfg.Namespace,
	)

	// ── 1. Build Kubernetes client ────────────────────────────────────────────
	// Built before the compute provider because LocalProvider needs it for
	// K8s service DNS → NodePort rewriting (Docker Desktop networking).
	restCfg, err := clientcmd.BuildConfigFromFlags("", cfg.Kubeconfig)
	if err != nil {
		log.Error("Failed to build Kubernetes client config", "err", err)
		os.Exit(1)
	}
	clientset, err := kubernetes.NewForConfig(restCfg)
	if err != nil {
		log.Error("Failed to create Kubernetes clientset", "err", err)
		os.Exit(1)
	}

	// ── 2. Select compute provider ────────────────────────────────────────────

	// Build RunPod pod config from CLI flags.
	// dataCenterIDs: parse comma-separated list; empty string → nil (all data centers).
	var dcIDs []string
	if cfg.RunPodDataCenterIDs != "" {
		for _, id := range strings.Split(cfg.RunPodDataCenterIDs, ",") {
			if s := strings.TrimSpace(id); s != "" {
				dcIDs = append(dcIDs, s)
			}
		}
	}
	// gpuTypeIDs: single-element slice when set; nil means RunPod auto-select.
	var gpuTypeIDs []string
	if cfg.RunPodGPUType != "" {
		gpuTypeIDs = []string{cfg.RunPodGPUType}
	}
	podCfg := runpodpkg.RunPodProviderConfig{
		GpuTypeIDs:           gpuTypeIDs,
		GpuTypePriority:      cfg.RunPodGPUPriority,
		DataCenterIDs:        dcIDs,
		ContainerDiskGB:      cfg.RunPodDiskGB,
		MinDiskBandwidthMBps: cfg.RunPodMinDiskMBps,
		MinDownloadMbps:      cfg.RunPodMinDownloadMbps,
		CloudType:            cfg.RunPodCloudType,
		Interruptible:        cfg.RunPodInterruptible,
	}

	var cp compute.ComputeProvider
	switch cfg.ComputeProvider {
	case "runpod":
		if cfg.RunPodAPIKey == "" {
			log.Error("RUNPOD_API_KEY environment variable must be set for the runpod provider")
			os.Exit(1)
		}
		if cfg.TailscaleEnabled {
			// At least one of static key or API key must be available so workers
			// can authenticate to the tailnet.
			if cfg.TailscaleAuthKey == "" && cfg.TailscaleAPIKey == "" {
				log.Error("set TAILSCALE_AUTH_KEY (static) or TAILSCALE_API_KEY (per-worker ephemeral) when --tailscale-enabled=true")
				os.Exit(1)
			}

			// Auto-derive --ray-head-tailscale-addr from --ray-head-dns-record when
			// not explicitly provided, appending the standard Ray GCS port.
			if cfg.RayHeadTailscaleAddr == "" && cfg.RayHeadDNSRecord != "" {
				cfg.RayHeadTailscaleAddr = cfg.RayHeadDNSRecord + ":6379"
				log.Info("Derived ray-head-tailscale-addr from dns record",
					"addr", cfg.RayHeadTailscaleAddr)
			}
			if cfg.RayHeadTailscaleAddr == "" {
				log.Error("set --ray-head-tailscale-addr or --ray-head-dns-record when --tailscale-enabled=true")
				os.Exit(1)
			}

			// Build the Tailscale syncer when the management API key is available.
			// The syncer handles:
			//   1. DNS sync (startup): creates/updates a MagicDNS A record pointing
			//      to the Ray head service ClusterIP, so workers can resolve it by name.
			//   2. Per-worker ephemeral keys (CreateInstance): mints a fresh single-use
			//      key for each RunPod container instead of sharing a static secret.
			// Both features degrade gracefully: DNS failures are logged as warnings and
			// the static TAILSCALE_AUTH_KEY is used as fallback for key generation.
			var syncer *runpodpkg.TailscaleSyncer
			if cfg.TailscaleAPIKey != "" && cfg.RayHeadDNSRecord != "" {
				syncer = runpodpkg.NewTailscaleSyncer(
					cfg.TailscaleAPIKey, cfg.TailnetName, cfg.RayHeadDNSRecord,
				)

				// DNS sync: resolve the Ray head ClusterIP and upsert the MagicDNS record.
				// Non-fatal: if this fails workers can still use the addr from the flag,
				// but the DNS name may be stale or missing until the next restart.
				// Use a short-lived context — the signal context is created later.
				syncCtx, syncCancel := context.WithTimeout(context.Background(), 30*time.Second)
				clusterIP, err := runpodpkg.GetServiceClusterIP(
					syncCtx, clientset, cfg.Namespace, cfg.RayHeadServiceName,
				)
				if err != nil {
					log.Warn("Tailscale DNS sync: could not resolve service ClusterIP",
						"service", cfg.RayHeadServiceName, "err", err)
				} else if err := syncer.SyncDNSRecord(syncCtx, clusterIP); err != nil {
					log.Warn("Tailscale DNS sync: could not update DNS record",
						"record", cfg.RayHeadDNSRecord, "ip", clusterIP, "err", err)
				} else {
					log.Info("Tailscale DNS record synced",
						"record", cfg.RayHeadDNSRecord, "clusterIP", clusterIP)
				}
				syncCancel()
			}

			cp = runpodpkg.NewRunPodProviderWithTailscale(
				cfg.RunPodAPIKey, cfg.TailscaleAuthKey, cfg.RayHeadTailscaleAddr, syncer, podCfg,
			)
			log.Info("Using RunPod compute provider (Tailscale mode)",
				"head-addr", cfg.RayHeadTailscaleAddr,
				"ephemeral-keys", syncer != nil,
				"gpu-type", cfg.RunPodGPUType)
		} else {
			cp = runpodpkg.NewRunPodProvider(cfg.RunPodAPIKey, podCfg)
			log.Info("Using RunPod compute provider (legacy public-IP mode)",
				"gpu-type", cfg.RunPodGPUType)
		}
	case "local":
		// Pass the clientset so the LocalProvider can rewrite K8s service DNS
		// names (*.svc.cluster.local) to 127.0.0.1:<nodePort>.
		// This is required on Docker Desktop / kind where pod IPs and ClusterIPs
		// are not routable from the host but NodePort services are (via localhost).
		// On bare-metal k3s where pod IPs are directly routable, pass nil instead.
		cp = localpkg.NewLocalProvider(clientset)
		log.Info("Using Local (Docker) compute provider",
			"dns-rewrite", "enabled (K8s service DNS → NodePort)")
	default:
		log.Error("Unknown compute provider", "value", cfg.ComputeProvider,
			"valid", "runpod, local")
		os.Exit(1)
	}

	// ── 3. Build the VK provider ──────────────────────────────────────────────
	prov := provider.NewProvider(cp, provider.NodeConfig{
		NodeName: cfg.NodeName,
		CPU:      cfg.NodeCPU,
		Memory:   cfg.NodeMemory,
		GPU:      cfg.NodeGPU,
	}, cfg.ReconcileInterval, log)

	// ── 4. Register / update the virtual node in Kubernetes ──────────────────
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	nodeObj := prov.BuildNode()
	if err := upsertNode(ctx, clientset.CoreV1().Nodes(), nodeObj, log); err != nil {
		log.Error("Failed to register virtual node", "node", cfg.NodeName, "err", err)
		os.Exit(1)
	}
	log.Info("Virtual node registered", "node", cfg.NodeName)

	// ── 5. Set up informers ───────────────────────────────────────────────────
	// Pod informer: filtered to only watch pods scheduled to this virtual node.
	podInformerFactory := informers.NewSharedInformerFactoryWithOptions(
		clientset,
		0,
		informers.WithTweakListOptions(func(opts *metav1.ListOptions) {
			opts.FieldSelector = fields.OneTermEqualSelector(
				"spec.nodeName", cfg.NodeName,
			).String()
		}),
	)

	// Standard informer: used by the PodController to resolve Secrets and ConfigMaps.
	stdInformerFactory := informers.NewSharedInformerFactory(clientset, 0)

	// ── 6. Event recorder ─────────────────────────────────────────────────────
	broadcaster := record.NewBroadcaster()
	broadcaster.StartLogging(func(format string, args ...any) {
		log.Debug(fmt.Sprintf(format, args...))
	})
	broadcaster.StartRecordingToSink(&corev1client.EventSinkImpl{
		Interface: clientset.CoreV1().Events(""),
	})
	recorder := broadcaster.NewRecorder(scheme.Scheme, corev1.EventSource{
		Component: "vk-ml",
	})

	// ── 7. Node controller ────────────────────────────────────────────────────
	var nodeControllerOpts []node.NodeControllerOpt
	if _, err := clientset.Discovery().ServerResourcesForGroupVersion("coordination.k8s.io/v1"); err == nil {
		nodeControllerOpts = append(nodeControllerOpts,
			node.WithNodeEnableLeaseV1(
				clientset.CoordinationV1().Leases(corev1.NamespaceNodeLease),
				int32(30),
			),
		)
	}

	nc, err := node.NewNodeController(
		prov,
		nodeObj,
		clientset.CoreV1().Nodes(),
		nodeControllerOpts...,
	)
	if err != nil {
		log.Error("Failed to create NodeController", "err", err)
		os.Exit(1)
	}

	// ── 8. Pod controller ─────────────────────────────────────────────────────
	pc, err := node.NewPodController(node.PodControllerConfig{
		PodClient:         clientset.CoreV1(),
		PodInformer:       podInformerFactory.Core().V1().Pods(),
		EventRecorder:     recorder,
		Provider:          prov,
		SecretInformer:    stdInformerFactory.Core().V1().Secrets(),
		ConfigMapInformer: stdInformerFactory.Core().V1().ConfigMaps(),
		ServiceInformer:   stdInformerFactory.Core().V1().Services(),
	})
	if err != nil {
		log.Error("Failed to create PodController", "err", err)
		os.Exit(1)
	}

	// ── 9. Start informers ────────────────────────────────────────────────────
	podInformerFactory.Start(ctx.Done())
	stdInformerFactory.Start(ctx.Done())

	// ── 10. Run controllers ───────────────────────────────────────────────────
	go func() {
		if err := nc.Run(ctx); err != nil && ctx.Err() == nil {
			log.Error("NodeController stopped unexpectedly", "err", err)
			cancel()
		}
	}()

	go func() {
		if err := pc.Run(ctx, 1); err != nil && ctx.Err() == nil {
			log.Error("PodController stopped unexpectedly", "err", err)
			cancel()
		}
	}()

	select {
	case <-nc.Ready():
	case <-ctx.Done():
		log.Info("Shutdown before NodeController became ready")
		return
	}
	log.Info("NodeController ready")

	select {
	case <-pc.Ready():
	case <-ctx.Done():
		log.Info("Shutdown before PodController became ready")
		return
	}
	log.Info("PodController ready — virtual node is active", "node", cfg.NodeName)

	<-ctx.Done()
	log.Info("Received shutdown signal, stopping")
	broadcaster.Shutdown()
}

// upsertNode creates the virtual node in Kubernetes, or updates it if it
// already exists (e.g. after a provider restart).
func upsertNode(ctx context.Context, nodes corev1client.NodeInterface, nodeObj *corev1.Node, log *slog.Logger) error {
	existing, err := nodes.Get(ctx, nodeObj.Name, metav1.GetOptions{})
	if k8serrors.IsNotFound(err) {
		_, err = nodes.Create(ctx, nodeObj, metav1.CreateOptions{})
		return err
	}
	if err != nil {
		return fmt.Errorf("get node %s: %w", nodeObj.Name, err)
	}

	nodeObj.ResourceVersion = existing.ResourceVersion
	_, err = nodes.Update(ctx, nodeObj, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("update node %s: %w", nodeObj.Name, err)
	}
	log.Debug("Virtual node already exists, updated", "node", nodeObj.Name)
	return nil
}
