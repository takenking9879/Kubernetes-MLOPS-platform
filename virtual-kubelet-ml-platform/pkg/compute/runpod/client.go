// Package runpod implements a ComputeProvider backed by RunPod's REST API.
// Only the REST API is used — the GraphQL path is excluded because it has
// known reliability issues (GRAPHQL_VALIDATION_FAILED) in production.
//
// API reference: https://docs.runpod.io/api-reference/pods
package runpod

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const baseURL = "https://rest.runpod.io/v1"

// Client is a thin HTTP client for the RunPod REST API.
// It handles authentication, JSON marshaling, and error surfacing.
type Client struct {
	apiKey     string
	httpClient *http.Client
}

// NewClient creates a RunPod API client authenticated with the given API key.
func NewClient(apiKey string) *Client {
	return &Client{
		apiKey:     apiKey,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

// ── Request / Response types ──────────────────────────────────────────────────

// CreatePodRequest is the body for POST /pods.
// Field mapping from InstanceSpec:
//
//	Image     → ImageName
//	GPU       → GpuCount
//	CPU/GPU   → MinVCPUPerGPU  (vCPUs per GPU, not total)
//	Memory/GPU→ MinRAMPerGPU   (GB per GPU, not total)
//	Command   → DockerEntrypoint
//	Args      → DockerStartCmd
//	Env       → Env
//	Ports     → Ports           ([]string, e.g. ["6379/tcp","8265/http"])
type CreatePodRequest struct {
	Name     string `json:"name"`
	ImageName string `json:"imageName"`
	GpuCount  int    `json:"gpuCount"`
	// GpuTypeIds pins the GPU model(s) RunPod may allocate, in preference order.
	// Example: []string{"NVIDIA RTX 2000 Ada Generation"}
	// When nil/empty the field is omitted and RunPod auto-selects by cost/availability.
	GpuTypeIds      []string `json:"gpuTypeIds,omitempty"`
	// GpuTypePriority controls how RunPod ranks matching GPU types: "availability" or "price".
	// Omitted when empty (RunPod default applies).
	GpuTypePriority string   `json:"gpuTypePriority,omitempty"`
	MinVCPUPerGPU   int      `json:"minVCPUPerGPU"` // vCPU per GPU (for GPU pods)
	MinRAMPerGPU    int      `json:"minRAMPerGPU"`  // GB RAM per GPU
	ContainerDiskInGb int    `json:"containerDiskInGb"`
	// DataCenterIds restricts allocation to specific RunPod data centers.
	// Omitted when nil/empty (all data centers eligible).
	DataCenterIds []string `json:"dataCenterIds,omitempty"`
	// Interruptible allows spot/preemptible instances (cheaper, may be terminated early).
	// Omitted when false (standard on-demand).
	Interruptible bool `json:"interruptible,omitempty"`
	// MinDiskBandwidthMBps sets a minimum disk throughput requirement in MB/s.
	// Omitted when 0 (no constraint).
	MinDiskBandwidthMBps int `json:"minDiskBandwidthMBps,omitempty"`
	// MinDownloadMbps sets a minimum network download speed requirement in Mbps.
	// Omitted when 0 (no constraint).
	MinDownloadMbps int               `json:"minDownloadMbps,omitempty"`
	Env             map[string]string `json:"env"`
	// Ports is an array of "port/protocol" strings.
	// Protocol must be "tcp" (binary) or "http" (RunPod HTTPS proxy).
	// TCP ports require supportPublicIp:true for inbound connectivity.
	// Example: ["6379/tcp", "8265/http"]
	Ports            []string `json:"ports"`
	DockerEntrypoint []string `json:"dockerEntrypoint"`
	DockerStartCmd   []string `json:"dockerStartCmd"`
	// SupportPublicIp requests a public IPv4 for Community Cloud pods.
	// Required for TCP inbound connectivity (e.g. Ray head → worker on port 6379).
	SupportPublicIp bool `json:"supportPublicIp"`
	// CloudType selects "SECURE" or "COMMUNITY".
	// SECURE Cloud uses globalNetworking for pod-to-pod DNS; COMMUNITY uses public IPs.
	CloudType string `json:"cloudType"`
}

// PodResponse is the shape returned by GET /pods/{id} and POST /pods.
//
// Key networking fields:
//   - PublicIp:     the pod's public IPv4, available once the pod is RUNNING.
//                  This is a top-level field, NOT inside machine{}.
//   - PortMappings: maps internal port (string key) → external public port (int).
//                  e.g. {"6379": 10341} means Ray GCS is accessible at PublicIp:10341.
//                  The external port is NOT the same as the internal port.
type PodResponse struct {
	ID            string         `json:"id"`
	Name          string         `json:"name"`
	DesiredStatus string         `json:"desiredStatus"`
	CurrentStatus string         `json:"currentStatus"`
	PublicIp      string         `json:"publicIp"`       // top-level, populated when RUNNING
	PortMappings  map[string]int `json:"portMappings"`   // {"6379": 10341, "8265": 10342}
	Runtime       *struct {
		Container *struct {
			ExitCode int    `json:"exitCode"`
			Message  string `json:"message"`
		} `json:"container"`
	} `json:"runtime"`
	Machine *struct {
		GpuTypeId  string `json:"gpuTypeId"`
		Location   string `json:"location"`
		DataCenter string `json:"dataCenterId"`
	} `json:"machine"`
}

// ── API methods ───────────────────────────────────────────────────────────────

// CreatePod sends POST /pods and returns the created pod response.
func (c *Client) CreatePod(ctx context.Context, req CreatePodRequest) (*PodResponse, error) {
	return c.postPod(ctx, "/pods", req)
}

// GetPod sends GET /pods/{id} and returns the current pod state.
func (c *Client) GetPod(ctx context.Context, id string) (*PodResponse, error) {
	return c.getPod(ctx, "/pods/"+id)
}

// StopPod sends POST /pods/{id}/stop to terminate the pod.
func (c *Client) StopPod(ctx context.Context, id string) error {
	resp, err := c.do(ctx, http.MethodPost, "/pods/"+id+"/stop", nil)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

// ── internal helpers ──────────────────────────────────────────────────────────

func (c *Client) postPod(ctx context.Context, path string, payload any) (*PodResponse, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	resp, err := c.do(ctx, http.MethodPost, path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return decodePod(resp.Body)
}

func (c *Client) getPod(ctx context.Context, path string) (*PodResponse, error) {
	resp, err := c.do(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return decodePod(resp.Body)
}

func decodePod(r io.Reader) (*PodResponse, error) {
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	var pod PodResponse
	if err := json.Unmarshal(raw, &pod); err != nil {
		return nil, fmt.Errorf("decode pod response: %w (body: %s)", err, raw)
	}
	return &pod, nil
}

func (c *Client) do(ctx context.Context, method, path string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, baseURL+path, body)
	if err != nil {
		return nil, fmt.Errorf("build request %s %s: %w", method, path, err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP %s %s: %w", method, path, err)
	}
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("HTTP %d from %s %s: %s", resp.StatusCode, method, path, b)
	}
	return resp, nil
}
