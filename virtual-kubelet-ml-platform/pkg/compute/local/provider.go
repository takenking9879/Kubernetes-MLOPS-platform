// Package local implements a ComputeProvider that runs containers locally
// via Docker. This is the primary debug path: it mirrors what
// k3s/kuberay/connect_gpu_worker.sh does manually, but driven by the VK
// provider interface so the full Kubernetes scheduling loop can be tested
// without spending money on RunPod.
//
// # Docker networking for Kubernetes-in-Docker (Docker Desktop / kind)
//
// When Kubernetes runs inside a VM (Docker Desktop on Windows/macOS, kind),
// pod IPs and ClusterIPs are NOT routable from the host. Docker containers
// launched with --network host share the host's network stack, which also
// cannot reach those addresses.
//
// The reachable path is:  container (host net) → 127.0.0.1:<nodePort> → K8s VM → pod
//
// The LocalProvider handles this automatically: when a Kubernetes clientset is
// provided, it scans container env vars and args for *.svc.cluster.local DNS
// names and rewrites them to 127.0.0.1:<nodePort> by looking up NodePort
// services in the same namespace. This allows Ray workers to connect to the
// Ray head without manual port-forwarding or /etc/hosts hacks.
//
// On bare-metal k3s (pod IPs are directly routable), pass nil for the clientset.
package local

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"github.com/ml-platform/vk-ml/pkg/compute"
)

// LocalProvider runs compute instances as local Docker containers.
type LocalProvider struct {
	// clientset is optional. When set, K8s service DNS names in env/args are
	// rewritten to 127.0.0.1:<nodePort> before launching containers.
	// Pass nil on bare-metal k3s where pod IPs are directly routable.
	clientset kubernetes.Interface

	mu        sync.Mutex
	instances map[string]string // podName → containerID
}

// NewLocalProvider creates a LocalProvider.
//   - clientset: pass a live K8s client to enable DNS→NodePort rewriting
//     (required for Docker Desktop / kind environments).
//     Pass nil to disable rewriting (bare-metal k3s).
func NewLocalProvider(clientset kubernetes.Interface) *LocalProvider {
	return &LocalProvider{
		clientset: clientset,
		instances: make(map[string]string),
	}
}

// CreateInstance runs the pod as a local Docker container.
//
// Equivalent shell command (Docker Desktop mode):
//
//	docker run -d --rm \
//	  --name <podName> \
//	  --label vk-ml=true \
//	  --network host \
//	  --shm-size=5gb \
//	  [--gpus all] \
//	  [--add-host <k8s-hostname>:127.0.0.1 ...] \
//	  -e KEY=REWRITTEN_VALUE ... \
//	  <image> <command...> <rewritten-args...>
func (p *LocalProvider) CreateInstance(ctx context.Context, spec compute.InstanceSpec) (*compute.Instance, error) {
	// Rewrite K8s service DNS names → 127.0.0.1:<nodePort> for Docker Desktop.
	resolvedEnv, resolvedCmd, resolvedArgs, extraHosts := p.rewriteK8sAddresses(ctx, spec)

	args := []string{
		"run", "-d", "--rm",
		"--name", spec.PodName,
		"--label", "vk-ml=true",
		"--label", "vk-ml-pod=" + spec.PodName,
		"--network", "host",
		"--shm-size=5gb",
	}

	if spec.GPU > 0 {
		args = append(args, "--gpus", "all")
	}

	// --add-host entries so that DNS lookups inside the container (beyond what
	// we already rewrote in env/args) also resolve to 127.0.0.1.
	for _, h := range extraHosts {
		args = append(args, "--add-host", h)
	}

	for k, v := range resolvedEnv {
		args = append(args, "-e", k+"="+v)
	}

	args = append(args, spec.Image)
	args = append(args, resolvedCmd...)
	args = append(args, resolvedArgs...)

	cmd := exec.CommandContext(ctx, "docker", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("docker run: %w\noutput: %s", err, string(out))
	}

	containerID := strings.TrimSpace(string(out))

	p.mu.Lock()
	p.instances[spec.PodName] = containerID
	p.mu.Unlock()

	return &compute.Instance{
		ID:        containerID,
		PodName:   spec.PodName,
		Namespace: spec.Namespace,
		Status:    compute.StatusStarting,
		StartedAt: time.Now(),
	}, nil
}

// DeleteInstance stops the named Docker container.
func (p *LocalProvider) DeleteInstance(ctx context.Context, id string) error {
	cmd := exec.CommandContext(ctx, "docker", "stop", id)
	_ = cmd.Run() // ignore: container may already be gone (--rm flag)

	p.mu.Lock()
	defer p.mu.Unlock()
	for name, cid := range p.instances {
		if cid == id {
			delete(p.instances, name)
			break
		}
	}
	return nil
}

// GetInstance queries Docker for the current state of a container.
func (p *LocalProvider) GetInstance(ctx context.Context, id string) (*compute.Instance, error) {
	cmd := exec.CommandContext(ctx, "docker", "inspect",
		"--format", `{"Status":"{{.State.Status}}","ExitCode":{{.State.ExitCode}}}`,
		id,
	)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	if err := cmd.Run(); err != nil {
		return &compute.Instance{ID: id, Status: compute.StatusNotFound}, nil
	}

	var state struct {
		Status   string `json:"Status"`
		ExitCode int    `json:"ExitCode"`
	}
	if err := json.Unmarshal(buf.Bytes(), &state); err != nil {
		return nil, fmt.Errorf("docker inspect parse: %w (raw: %s)", err, buf.String())
	}

	inst := &compute.Instance{ID: id}
	switch state.Status {
	case "running":
		inst.Status = compute.StatusRunning
	case "created":
		inst.Status = compute.StatusStarting
	case "exited":
		inst.ExitCode = state.ExitCode
		inst.Status = compute.StatusExited
	case "dead":
		inst.ExitCode = 1
		inst.Status = compute.StatusExited
	default:
		inst.Status = compute.StatusUnknown
	}
	return inst, nil
}

// ListInstances returns all instances tracked by this provider.
func (p *LocalProvider) ListInstances(ctx context.Context) ([]compute.Instance, error) {
	p.mu.Lock()
	snapshot := make(map[string]string, len(p.instances))
	for k, v := range p.instances {
		snapshot[k] = v
	}
	p.mu.Unlock()

	result := make([]compute.Instance, 0, len(snapshot))
	for podName, cid := range snapshot {
		inst, err := p.GetInstance(ctx, cid)
		if err != nil {
			continue
		}
		inst.PodName = podName
		result = append(result, *inst)
	}
	return result, nil
}

// ── K8s DNS → NodePort rewriting ─────────────────────────────────────────────

// k8sDNSPattern matches Kubernetes service DNS names with an optional port.
// Captures: (1) service name, (2) namespace, (3) port (may be empty)
// Examples:
//   - ray-head-svc.ray.svc.cluster.local:6379  → [ray-head-svc, ray, 6379]
//   - ray-head-svc.ray.svc.cluster.local        → [ray-head-svc, ray, ""]
var k8sDNSPattern = regexp.MustCompile(
	`([a-z0-9][a-z0-9-]*)\.([a-z0-9][a-z0-9-]*)\.svc\.cluster\.local(?::(\d+))?`,
)

// rewriteK8sAddresses scans env, command, and args for K8s service DNS names
// and replaces them with 127.0.0.1:<nodePort>. It also returns a list of
// --add-host strings (hostname:127.0.0.1) for resolved hostnames so that any
// in-container DNS lookups also resolve correctly.
func (p *LocalProvider) rewriteK8sAddresses(
	ctx context.Context,
	spec compute.InstanceSpec,
) (env map[string]string, cmd []string, args []string, extraHosts []string) {
	if p.clientset == nil {
		// Bare-metal mode — no rewriting needed.
		return spec.Env, spec.Command, spec.Args, nil
	}

	resolvedHosts := make(map[string]bool)

	rewrite := func(s string) string {
		return k8sDNSPattern.ReplaceAllStringFunc(s, func(match string) string {
			groups := k8sDNSPattern.FindStringSubmatch(match)
			if len(groups) < 3 {
				return match
			}
			svcName, namespace, portStr := groups[1], groups[2], groups[3]

			nodePort, err := p.findNodePort(ctx, svcName, namespace, portStr)
			if err != nil || nodePort == 0 {
				return match // can't resolve: leave unchanged
			}

			hostname := svcName + "." + namespace + ".svc.cluster.local"
			resolvedHosts[hostname] = true

			if portStr != "" {
				return fmt.Sprintf("127.0.0.1:%d", nodePort)
			}
			return "127.0.0.1"
		})
	}

	env = make(map[string]string, len(spec.Env))
	for k, v := range spec.Env {
		env[k] = rewrite(v)
	}

	cmd = make([]string, len(spec.Command))
	for i, c := range spec.Command {
		cmd[i] = rewrite(c)
	}
	args = make([]string, len(spec.Args))
	for i, a := range spec.Args {
		args[i] = rewrite(a)
	}

	for host := range resolvedHosts {
		extraHosts = append(extraHosts, host+":127.0.0.1")
	}
	return env, cmd, args, extraHosts
}

// findNodePort looks up a NodePort for the given service+namespace+port.
//
// Strategy:
//  1. Try the exact service name first (works for ClusterIP+NodePort services).
//  2. If the service is headless (no NodePort), scan all NodePort services in
//     the namespace and return the first one exposing the requested port.
//     This handles KubeRay's pattern where the worker uses the headless
//     *-head-svc name but a companion NodePort service covers the same port.
func (p *LocalProvider) findNodePort(ctx context.Context, svcName, namespace, portStr string) (int32, error) {
	svc, err := p.clientset.CoreV1().Services(namespace).Get(ctx, svcName, metav1.GetOptions{})
	if err == nil {
		if np := nodePortForPort(svc, portStr); np > 0 {
			return np, nil
		}
	}

	// Headless or missing — scan all NodePort services in namespace.
	list, err := p.clientset.CoreV1().Services(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return 0, fmt.Errorf("list services in %s: %w", namespace, err)
	}
	for _, s := range list.Items {
		if s.Spec.Type != corev1.ServiceTypeNodePort {
			continue
		}
		if np := nodePortForPort(&s, portStr); np > 0 {
			return np, nil
		}
	}
	return 0, fmt.Errorf("no NodePort found for %s/%s port %s", namespace, svcName, portStr)
}

// nodePortForPort returns the NodePort for the service port matching portStr.
// If portStr is empty, returns the first NodePort found.
func nodePortForPort(svc *corev1.Service, portStr string) int32 {
	for _, p := range svc.Spec.Ports {
		if p.NodePort == 0 {
			continue
		}
		if portStr == "" {
			return p.NodePort
		}
		if strconv.Itoa(int(p.Port)) == portStr ||
			strconv.Itoa(int(p.TargetPort.IntVal)) == portStr {
			return p.NodePort
		}
	}
	return 0
}
