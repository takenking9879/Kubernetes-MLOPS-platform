package provider

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/virtual-kubelet/virtual-kubelet/node/api"

	"github.com/ml-platform/vk-ml/pkg/compute"
)

// podRecord holds the Kubernetes pod object and the provider-assigned instance ID.
type podRecord struct {
	pod        *corev1.Pod
	instanceID string // provider-specific ID (Docker container ID or RunPod pod ID)
}

// NodeConfig holds the virtual node's identity and advertised capacity.
// Capacity values are Kubernetes resource quantity strings (e.g. "1000", "10Ti").
type NodeConfig struct {
	NodeName string
	CPU      string // total vCPU capacity (e.g. "1000")
	Memory   string // total memory capacity (e.g. "10Ti")
	GPU      string // total nvidia.com/gpu capacity (e.g. "100")
}

// Provider implements:
//   - node.PodLifecycleHandler (CreatePod, UpdatePod, DeletePod, GetPod, GetPodStatus, GetPods)
//   - node.NodeProvider        (Ping, NotifyNodeStatus)
//   - node.PodNotifier         (NotifyPods) — enables push-based status updates
type Provider struct {
	compute           compute.ComputeProvider
	nodeName          string
	logger            *slog.Logger
	reconcileInterval time.Duration

	mu         sync.RWMutex
	pods       map[string]podRecord // key: "namespace/name"

	notifyMu   sync.Mutex
	notifyFunc func(*corev1.Pod) // registered by PodController via NotifyPods

	// advertised node capacity
	nodeCPU    resource.Quantity
	nodeMemory resource.Quantity
	nodeGPU    resource.Quantity
}

// NewProvider creates a Provider that delegates instance management to cp.
func NewProvider(cp compute.ComputeProvider, cfg NodeConfig, reconcileInterval time.Duration, logger *slog.Logger) *Provider {
	return &Provider{
		compute:           cp,
		nodeName:          cfg.NodeName,
		logger:            logger,
		reconcileInterval: reconcileInterval,
		pods:              make(map[string]podRecord),
		nodeCPU:           resource.MustParse(cfg.CPU),
		nodeMemory:        resource.MustParse(cfg.Memory),
		nodeGPU:           resource.MustParse(cfg.GPU),
	}
}

// ── PodLifecycleHandler ───────────────────────────────────────────────────────

// CreatePod is called by the PodController when a pod is assigned to this node.
// It translates the pod spec to an InstanceSpec and launches the compute instance.
func (p *Provider) CreatePod(ctx context.Context, pod *corev1.Pod) error {
	key := podKey(pod)
	p.logger.Info("CreatePod", "pod", key, "image", pod.Spec.Containers[0].Image)

	spec := TranslatePod(pod)

	// Inject GCS wait loop before ray start.
	// Mirrors the initContainer "wait-gcs-ready" that KubeRay injects but VK skips.
	spec.Args = WrapArgsWithGCSWait(spec.Args, 300)

	inst, err := p.compute.CreateInstance(ctx, spec)
	if err != nil {
		return fmt.Errorf("CreatePod %s: %w", key, err)
	}

	p.mu.Lock()
	p.pods[key] = podRecord{pod: pod.DeepCopy(), instanceID: inst.ID}
	p.mu.Unlock()

	p.logger.Info("CreatePod: instance launched", "pod", key, "instanceID", inst.ID)
	return nil
}

// UpdatePod is called when a pod spec changes. For this POC, running instances
// are not reconfigured — the update only refreshes the cached pod object.
func (p *Provider) UpdatePod(_ context.Context, pod *corev1.Pod) error {
	key := podKey(pod)
	p.mu.Lock()
	if rec, ok := p.pods[key]; ok {
		rec.pod = pod.DeepCopy()
		p.pods[key] = rec
	}
	p.mu.Unlock()
	return nil
}

// DeletePod is called when a pod is deleted from Kubernetes.
// It terminates the corresponding compute instance.
func (p *Provider) DeletePod(ctx context.Context, pod *corev1.Pod) error {
	key := podKey(pod)
	p.logger.Info("DeletePod", "pod", key)

	p.mu.Lock()
	rec, ok := p.pods[key]
	if ok {
		delete(p.pods, key)
	}
	p.mu.Unlock()

	if !ok {
		return nil // already removed
	}

	if err := p.compute.DeleteInstance(ctx, rec.instanceID); err != nil {
		// Log but don't fail — the pod is being deleted regardless.
		p.logger.Warn("DeletePod: terminate instance error", "pod", key,
			"instanceID", rec.instanceID, "err", err)
	}
	return nil
}

// GetPod returns the cached pod object for the given namespace/name.
func (p *Provider) GetPod(_ context.Context, namespace, name string) (*corev1.Pod, error) {
	key := namespace + "/" + name
	p.mu.RLock()
	rec, ok := p.pods[key]
	p.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("pod %s not found", key)
	}
	return rec.pod.DeepCopy(), nil
}

// GetPodStatus fetches the current instance state from the compute provider
// and translates it into a Kubernetes PodStatus.
func (p *Provider) GetPodStatus(ctx context.Context, namespace, name string) (*corev1.PodStatus, error) {
	key := namespace + "/" + name

	p.mu.RLock()
	rec, ok := p.pods[key]
	p.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("pod %s not found", key)
	}

	inst, err := p.compute.GetInstance(ctx, rec.instanceID)
	if err != nil {
		return nil, fmt.Errorf("GetPodStatus %s: %w", key, err)
	}

	return BuildPodStatus(rec.pod, inst), nil
}

// GetPods returns all pods currently tracked by this provider.
func (p *Provider) GetPods(_ context.Context) ([]*corev1.Pod, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	pods := make([]*corev1.Pod, 0, len(p.pods))
	for _, rec := range p.pods {
		pods = append(pods, rec.pod.DeepCopy())
	}
	return pods, nil
}

// ── PodNotifier ───────────────────────────────────────────────────────────────

// NotifyPods registers the callback the VK PodController uses to receive
// asynchronous pod status updates. This is called once at startup.
// It immediately starts the status reconciliation background loop.
func (p *Provider) NotifyPods(ctx context.Context, f func(*corev1.Pod)) {
	p.notifyMu.Lock()
	p.notifyFunc = f
	p.notifyMu.Unlock()

	go p.reconcileLoop(ctx)
}

// reconcileLoop polls all tracked instances at the configured interval and
// pushes pod status changes via the notifyFunc callback.
// This drives the Pending → Running → Succeeded/Failed transitions in Kubernetes.
func (p *Provider) reconcileLoop(ctx context.Context) {
	ticker := time.NewTicker(p.reconcileInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.reconcileAll(ctx)
		}
	}
}

func (p *Provider) reconcileAll(ctx context.Context) {
	p.mu.RLock()
	records := make([]podRecord, 0, len(p.pods))
	for _, rec := range p.pods {
		records = append(records, rec)
	}
	p.mu.RUnlock()

	for _, rec := range records {
		inst, err := p.compute.GetInstance(ctx, rec.instanceID)
		if err != nil {
			p.logger.Warn("reconcile: get instance failed",
				"instanceID", rec.instanceID, "err", err)
			continue
		}

		podCopy := rec.pod.DeepCopy()
		podCopy.Status = *BuildPodStatus(rec.pod, inst)

		// Enrich pod annotations with port mapping information (RunPod only).
		// This allows operators to discover the external TCP port for each service.
		if ann := PortMappingAnnotations(inst.PortMappings); len(ann) > 0 {
			if podCopy.Annotations == nil {
				podCopy.Annotations = make(map[string]string)
			}
			for k, v := range ann {
				podCopy.Annotations[k] = v
			}
			// Also store the RunPod instance ID for debugging.
			podCopy.Annotations["vk-ml/instance-id"] = inst.ID
			if inst.PublicIP != "" {
				podCopy.Annotations["vk-ml/public-ip"] = inst.PublicIP
			}
		}

		p.notifyMu.Lock()
		if p.notifyFunc != nil {
			p.notifyFunc(podCopy)
		}
		p.notifyMu.Unlock()
	}
}

// ── NodeProvider ──────────────────────────────────────────────────────────────

// Ping reports whether the provider is reachable. Used by the NodeController
// for liveness checks. Always returns nil for this POC.
func (p *Provider) Ping(_ context.Context) error {
	return nil
}

// NotifyNodeStatus registers the callback the NodeController uses to receive
// node status updates. It starts a background goroutine that pushes the node
// status every 30 seconds to keep the node object fresh in Kubernetes.
func (p *Provider) NotifyNodeStatus(ctx context.Context, f func(*corev1.Node)) {
	// Send the initial status immediately so the node becomes Ready fast.
	f(p.buildNode())

	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				f(p.buildNode())
			}
		}
	}()
}

// BuildNode creates the *corev1.Node object that represents this virtual node
// in Kubernetes. Called by main.go once at startup to register the node.
func (p *Provider) BuildNode() *corev1.Node {
	return p.buildNode()
}

func (p *Provider) buildNode() *corev1.Node {
	now := metav1.NewTime(time.Now())
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: p.nodeName,
			Labels: map[string]string{
				"type":                    "virtual-kubelet",
				"kubernetes.io/role":      "agent",
				"kubernetes.io/hostname":  p.nodeName,
				"beta.kubernetes.io/os":   "linux",
				"vk-ml/provider":          "true",
			},
			Annotations: map[string]string{
				"node.alpha.kubernetes.io/ttl": "0",
			},
		},
		Spec: corev1.NodeSpec{
			// This taint prevents regular pods from being scheduled here.
			// KubeRay worker groups must add a matching toleration.
			Taints: []corev1.Taint{
				{
					Key:    "virtual-kubelet.io/provider",
					Value:  "runpod",
					Effect: corev1.TaintEffectNoSchedule,
				},
			},
		},
		Status: corev1.NodeStatus{
			Capacity: corev1.ResourceList{
				corev1.ResourceCPU:    p.nodeCPU,
				corev1.ResourceMemory: p.nodeMemory,
				"nvidia.com/gpu":      p.nodeGPU,
				corev1.ResourcePods:   resource.MustParse("500"),
			},
			Allocatable: corev1.ResourceList{
				corev1.ResourceCPU:    p.nodeCPU,
				corev1.ResourceMemory: p.nodeMemory,
				"nvidia.com/gpu":      p.nodeGPU,
				corev1.ResourcePods:   resource.MustParse("500"),
			},
			Conditions: []corev1.NodeCondition{
				{
					Type:               corev1.NodeReady,
					Status:             corev1.ConditionTrue,
					LastHeartbeatTime:  now,
					LastTransitionTime: now,
					Reason:             "KubeletReady",
					Message:            "vk-ml node is ready",
				},
				{
					Type:               corev1.NodeMemoryPressure,
					Status:             corev1.ConditionFalse,
					LastHeartbeatTime:  now,
					LastTransitionTime: now,
				},
				{
					Type:               corev1.NodeDiskPressure,
					Status:             corev1.ConditionFalse,
					LastHeartbeatTime:  now,
					LastTransitionTime: now,
				},
				{
					Type:               corev1.NodePIDPressure,
					Status:             corev1.ConditionFalse,
					LastHeartbeatTime:  now,
					LastTransitionTime: now,
				},
			},
			NodeInfo: corev1.NodeSystemInfo{
				OperatingSystem:         "linux",
				Architecture:            "amd64",
				KubeletVersion:          "v1.32.0",
				ContainerRuntimeVersion: "docker://24.0.0",
			},
			Addresses: []corev1.NodeAddress{
				{Type: corev1.NodeHostName, Address: p.nodeName},
			},
		},
	}
}

// ── Unimplemented stubs ───────────────────────────────────────────────────────
// The VK API server requires these methods but they are not meaningful for
// external compute providers in this POC.

// GetContainerLogs returns an error — RunPod does not expose log streaming via API.
func (p *Provider) GetContainerLogs(_ context.Context, _, _, _ string, _ api.ContainerLogOpts) (io.ReadCloser, error) {
	return nil, fmt.Errorf("GetContainerLogs: not supported by external compute providers")
}

// RunInContainer returns an error — exec into RunPod containers is not supported.
func (p *Provider) RunInContainer(_ context.Context, _, _, _ string, _ []string, _ api.AttachIO) error {
	return fmt.Errorf("RunInContainer: not supported by external compute providers")
}

// AttachToContainer returns an error — not supported.
func (p *Provider) AttachToContainer(_ context.Context, _, _, _ string, _ api.AttachIO) error {
	return fmt.Errorf("AttachToContainer: not supported by external compute providers")
}

// ── helpers ───────────────────────────────────────────────────────────────────

func podKey(pod *corev1.Pod) string {
	return pod.Namespace + "/" + pod.Name
}
