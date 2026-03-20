// Package provider implements the Virtual Kubelet PodLifecycleHandler and
// NodeProvider interfaces, bridging Kubernetes pod scheduling to any
// compute.ComputeProvider backend.
package provider

import (
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"

	"github.com/ml-platform/vk-ml/pkg/compute"
)

// defaultRayPorts are exposed on every Ray worker instance.
//
// Port protocol rules (RunPod):
//   - "tcp"  → direct TCP port mapping; requires supportPublicIp:true.
//              The external port differs from the internal port (see portMappings).
//   - "http" → routed through RunPod's HTTPS reverse proxy.
//              Suitable for browser-accessible dashboards, NOT for Ray's binary protocol.
//
// Ray GCS (6379), client (10001), and metrics (8002) must be TCP.
// Ray dashboard (8265) can be HTTP for browser access.
//
// Note: RunPod does not support port ranges in the ports array.
// Each port must be declared individually.
var defaultRayPorts = []string{
	"6379/tcp",  // Ray GCS server — head ↔ worker (binary protocol, must be TCP)
	"10001/tcp", // Ray client server (TCP)
	"8265/http", // Ray dashboard (HTTP proxy is fine for browser)
	"8002/tcp",  // Prometheus metrics
}

// TranslatePod converts a Kubernetes Pod spec into the minimal InstanceSpec
// needed to launch a Ray worker on any compute provider.
//
// Design decisions:
//   - Only the first container is processed (single-container assumption for POC)
//   - initContainers, volumes, volumeMounts, affinity, tolerations are ignored
//   - Only direct env var values are extracted (no SecretKeyRef for POC)
//   - Resources: prefer Requests, fall back to Limits
//   - Ports: use defaultRayPorts unless the container declares its own ports
func TranslatePod(pod *corev1.Pod) compute.InstanceSpec {
	spec := compute.InstanceSpec{
		PodName:   pod.Name,
		Namespace: pod.Namespace,
		PodUID:    string(pod.UID),
		Ports:     defaultRayPorts,
	}

	if len(pod.Spec.Containers) == 0 {
		return spec
	}

	c := pod.Spec.Containers[0]
	spec.Image = c.Image
	spec.Command = c.Command
	spec.Args = c.Args
	spec.Env = extractEnv(c.Env)

	// Resources: prefer Requests, fall back to Limits.
	reqs := c.Resources.Requests
	if len(reqs) == 0 {
		reqs = c.Resources.Limits
	}
	spec.CPU = cpuCores(reqs)
	spec.MemoryGB = memoryGB(reqs)
	spec.GPU = gpuCount(reqs)

	// If the container declares explicit ports, use those instead of the defaults.
	// This allows non-Ray workloads to declare their own port requirements.
	if len(c.Ports) > 0 {
		spec.Ports = containerPortsToStrings(c.Ports)
	}

	return spec
}

// ── helpers ───────────────────────────────────────────────────────────────────

// extractEnv converts []corev1.EnvVar to map[string]string.
// Only direct Value entries are supported.
// ValueFrom (SecretKeyRef, ConfigMapKeyRef) is silently skipped for this POC.
func extractEnv(envs []corev1.EnvVar) map[string]string {
	m := make(map[string]string, len(envs))
	for _, e := range envs {
		if e.Value != "" {
			m[e.Name] = e.Value
		}
	}
	return m
}

// cpuCores reads the CPU resource quantity and returns total cores (rounded up).
// Example: "2500m" → 3
func cpuCores(reqs corev1.ResourceList) int {
	q, ok := reqs[corev1.ResourceCPU]
	if !ok {
		return 0
	}
	// MilliValue gives milli-cores; divide and round up.
	return int((q.MilliValue() + 999) / 1000)
}

// memoryGB reads the memory resource quantity and returns total GiB (rounded up).
// Example: "8Gi" → 8, "10Gi" → 10, "1500Mi" → 2
func memoryGB(reqs corev1.ResourceList) int {
	q, ok := reqs[corev1.ResourceMemory]
	if !ok {
		return 0
	}
	bytesVal := q.Value()
	// ceil(bytes / 1Gi)
	const gib = int64(1 << 30)
	return int((bytesVal + gib - 1) / gib)
}

// gpuCount reads the nvidia.com/gpu resource quantity and returns the integer count.
func gpuCount(reqs corev1.ResourceList) int {
	q, ok := reqs["nvidia.com/gpu"]
	if !ok {
		return 0
	}
	v, _ := strconv.Atoi(q.String())
	return v
}

// containerPortsToStrings converts []corev1.ContainerPort to RunPod port strings.
// HTTP ports (well-known web ports) are classified as "http"; all others as "tcp".
func containerPortsToStrings(ports []corev1.ContainerPort) []string {
	result := make([]string, 0, len(ports))
	for _, p := range ports {
		proto := "tcp"
		if isHTTPPort(int(p.ContainerPort)) {
			proto = "http"
		}
		if strings.EqualFold(string(p.Protocol), "udp") {
			proto = "udp"
		}
		result = append(result, strconv.Itoa(int(p.ContainerPort))+"/"+proto)
	}
	return result
}

// isHTTPPort returns true for ports conventionally served over HTTP/HTTPS
// that should use RunPod's HTTPS reverse proxy rather than direct TCP mapping.
func isHTTPPort(port int) bool {
	switch port {
	case 80, 443, 3000, 5000, 8080, 8088, 8265, 8888, 9000:
		return true
	}
	return false
}
