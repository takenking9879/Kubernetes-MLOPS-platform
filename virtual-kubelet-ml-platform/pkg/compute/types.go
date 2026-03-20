// Package compute defines the abstraction layer between the Virtual Kubelet
// provider and any external GPU compute backend (RunPod, local Docker, etc.).
package compute

import (
	"context"
	"time"
)

// InstanceSpec is the minimal normalized spec extracted from a Kubernetes Pod.
// It carries only the fields a GPU compute provider needs to launch a container.
// Fields like volumes, affinity, tolerations, and probes are intentionally omitted
// for this POC — only single-container Ray worker pods are targeted.
type InstanceSpec struct {
	// Kubernetes identity — preserved for tracking and logging.
	PodName   string
	Namespace string
	PodUID    string

	// Container definition
	Image   string
	Command []string          // maps to dockerEntrypoint
	Args    []string          // maps to dockerStartCmd
	Env     map[string]string // direct key=value pairs only (no SecretKeyRef for POC)
	// Ports in "port/protocol" format, e.g. ["6379/tcp", "8265/http"].
	// These are the ports the provider will expose on the compute instance.
	Ports []string

	// Resource requests (prefer Requests, fall back to Limits)
	CPU      int // total vCPU cores
	MemoryGB int // total RAM in GiB
	GPU      int // nvidia.com/gpu count
}

// InstanceStatus represents the lifecycle state of a compute instance.
// Each provider maps its own status strings to these canonical values.
type InstanceStatus string

const (
	StatusStarting   InstanceStatus = "STARTING"
	StatusRunning    InstanceStatus = "RUNNING"
	StatusExited     InstanceStatus = "EXITED"
	StatusTerminated InstanceStatus = "TERMINATED"
	StatusNotFound   InstanceStatus = "NOT_FOUND"
	StatusUnknown    InstanceStatus = "UNKNOWN"
)

// Instance is a running (or completed) compute instance tracked by the provider.
type Instance struct {
	ID        string
	PodName   string
	Namespace string
	Status    InstanceStatus
	ExitCode  int

	// PublicIP is the instance's public IPv4 address once running.
	// For RunPod: this is the top-level "publicIp" field from the pod response.
	// For LocalProvider: empty (--network host is used, no separate IP).
	PublicIP string

	// PortMappings maps an internal container port (as string key) to the
	// externally reachable public port (int value).
	// Example: {"6379": 10341} means Ray GCS is reachable at PublicIP:10341.
	// This is populated after the instance is RUNNING and has been assigned a
	// public IP by the provider.
	PortMappings map[string]int

	StartedAt time.Time
}

// ComputeProvider is the single interface all backend GPU providers implement.
// Implementing a new provider (Lambda Labs, Vast.ai, etc.) only requires
// satisfying these four methods.
type ComputeProvider interface {
	// CreateInstance launches a new compute instance from the given spec.
	// Returns an Instance with at least the provider-assigned ID set.
	CreateInstance(ctx context.Context, spec InstanceSpec) (*Instance, error)

	// DeleteInstance stops and removes the instance identified by id.
	// Implementations should treat "already gone" as a non-error.
	DeleteInstance(ctx context.Context, id string) error

	// GetInstance returns the current state of an instance, including status,
	// exit code, public IP, and port mappings (once available).
	GetInstance(ctx context.Context, id string) (*Instance, error)

	// ListInstances returns all instances currently tracked by the provider.
	ListInstances(ctx context.Context) ([]Instance, error)
}
