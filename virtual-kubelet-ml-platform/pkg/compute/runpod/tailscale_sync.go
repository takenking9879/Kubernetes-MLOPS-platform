package runpod

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const tailscaleAPIBase = "https://api.tailscale.com"

// TailscaleSyncer automates two tasks that previously required manual steps:
//
//  1. DNS sync: at vk-ml startup it queries the K8s service ClusterIP and
//     creates/updates a Tailscale MagicDNS custom A record, so workers can
//     resolve the Ray head by name without hardcoding an IP in the YAML.
//
//  2. Per-worker ephemeral keys: instead of sharing a single static auth key
//     across all workers (which can't be rotated without a restart), each call
//     to CreateInstance generates a fresh single-use ephemeral key via the
//     Tailscale API. The key expires in 5 minutes and the node is auto-removed
//     from the tailnet admin when the RunPod container terminates.
//
// Both features are optional — if TailscaleAPIKey is empty the syncer is nil
// and the provider falls back to the static TAILSCALE_AUTH_KEY.
//
// # Architecture note on one-tailscaled-per-worker
//
// Running tailscaled inside every RunPod container is the correct design for
// this Ray architecture, not a scalability problem. Here is why:
//
// Ray requires BIDIRECTIONAL connectivity between the head and each worker:
//   - Worker → Head  (port 6379): initial registration via GCS server.
//   - Head → Worker  (raylet port range): the head connects back to start the
//     raylet once the worker has registered its IP and port.
//
// Without Tailscale on the worker the only IP available is the RunPod private
// container IP, which is behind NAT and unreachable from the K3s node.
//
// Alternative designs that avoid per-container tailscaled all fall short:
//   - K3s subnet router alone: routes Head→ClusterIP, but cannot route
//     Head→RunPod private IP (different network, behind NAT).
//   - Single gateway node: solves Worker→Head direction only; the head still
//     cannot initiate connections back to workers behind the gateway.
//   - Public IP (legacy mode): removes the NAT issue but reintroduces open
//     TCP ports and public exposure — the problem Tailscale was meant to solve.
//
// The per-worker overhead is small: ~50–100 MB RAM and ~30 s startup time
// for a tailscaled process that runs in userspace mode (no kernel TUN needed).
// For GPU workloads with ≥8 GB RAM per pod this is negligible.
//
// The practical improvement offered here is generating per-worker ephemeral
// keys (GenerateEphemeralKey) so no long-lived shared secret is injected into
// containers, and expired workers are automatically pruned from the admin console.
type TailscaleSyncer struct {
	apiKey    string // Tailscale management API key (read:write scope)
	tailnet   string // tailnet org name, e.g. "example@github" or "-" for default
	dnsRecord string // full DNS name to maintain, e.g. "ray-head.example.ts.net"

	httpClient *http.Client
}

// NewTailscaleSyncer creates a TailscaleSyncer.
//
// tailnet is the Tailscale organization identifier used in API URLs.
// Use "-" to target the tailnet that owns apiKey (works for most single-tailnet setups).
// Use the explicit name (e.g. "example@github") when managing a specific org.
//
// dnsRecord is the full hostname that will be created/maintained as an A record,
// e.g. "ray-head.example.ts.net". Workers resolve this name after joining the tailnet.
func NewTailscaleSyncer(apiKey, tailnet, dnsRecord string) *TailscaleSyncer {
	return &TailscaleSyncer{
		apiKey:     apiKey,
		tailnet:    tailnet,
		dnsRecord:  dnsRecord,
		httpClient: &http.Client{Timeout: 15 * time.Second},
	}
}

// GetServiceClusterIP returns the ClusterIP of a Kubernetes service.
// Returns an error if the service does not exist or has no ClusterIP (headless).
// This is a standalone function so callers can use it independently of SyncDNSRecord.
func GetServiceClusterIP(ctx context.Context, client kubernetes.Interface, namespace, name string) (string, error) {
	svc, err := client.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("get service %s/%s: %w", namespace, name, err)
	}
	if svc.Spec.ClusterIP == "" || svc.Spec.ClusterIP == "None" {
		return "", fmt.Errorf("service %s/%s has no ClusterIP (headless or not yet assigned)", namespace, name)
	}
	return svc.Spec.ClusterIP, nil
}

// SyncDNSRecord creates or updates the Tailscale MagicDNS custom A record
// (s.dnsRecord → ip). It is idempotent: if a record already exists with the
// correct value it is left unchanged; stale records with the same name but a
// different IP are deleted first.
//
// Call this once at vk-ml startup before workers are created, so the MagicDNS
// name resolves correctly from the moment workers join the tailnet.
func (s *TailscaleSyncer) SyncDNSRecord(ctx context.Context, ip string) error {
	existing, err := s.listDNSRecords(ctx)
	if err != nil {
		return fmt.Errorf("list tailscale dns records: %w", err)
	}

	for _, r := range existing {
		if r.Name != s.dnsRecord {
			continue
		}
		if r.Value == ip {
			// Record already points to the correct IP — nothing to do.
			return nil
		}
		// Stale record: wrong IP. Delete it before creating a fresh one.
		if err := s.deleteDNSRecord(ctx, r.ID); err != nil {
			return fmt.Errorf("delete stale dns record %s: %w", r.ID, err)
		}
	}

	return s.createDNSRecord(ctx, ip)
}

// GenerateEphemeralKey calls the Tailscale keys API to mint a fresh single-use
// pre-authorised ephemeral auth key for one RunPod worker.
//
// Properties of the generated key:
//   - reusable=false  — can only be used once (by this worker).
//   - ephemeral=true  — the tailnet device is removed automatically when the
//     container disconnects (keeps the admin console clean).
//   - preauthorized=true — no manual approval required in the admin console.
//   - expires in 5 minutes — must be consumed quickly; prevents accumulation of
//     leaked keys.
//
// Returns (key, nil) on success. Callers should fall back to the static
// TailscaleAuthKey if this call fails rather than blocking worker creation.
func (s *TailscaleSyncer) GenerateEphemeralKey(ctx context.Context) (string, error) {
	url := fmt.Sprintf("%s/api/v2/tailnet/%s/keys", tailscaleAPIBase, s.tailnet)

	body, err := json.Marshal(map[string]any{
		"capabilities": map[string]any{
			"devices": map[string]any{
				"create": map[string]any{
					"reusable":      false,
					"ephemeral":     true,
					"preauthorized": true,
				},
			},
		},
		"expirySeconds": 300, // 5-minute window to use the key
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+s.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("tailscale generate key: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return "", fmt.Errorf("tailscale generate key: HTTP %d", resp.StatusCode)
	}

	var result struct {
		Key string `json:"key"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode key response: %w", err)
	}
	if result.Key == "" {
		return "", fmt.Errorf("tailscale generate key: empty key in response")
	}
	return result.Key, nil
}

// ── internal DNS helpers ──────────────────────────────────────────────────────

// tsDNSRecord is the JSON shape used by the Tailscale DNS records API.
type tsDNSRecord struct {
	ID    string `json:"id,omitempty"`
	Name  string `json:"name"`
	Type  string `json:"type"`
	Value string `json:"value"`
}

func (s *TailscaleSyncer) listDNSRecords(ctx context.Context) ([]tsDNSRecord, error) {
	url := fmt.Sprintf("%s/api/v2/tailnet/%s/dns/records", tailscaleAPIBase, s.tailnet)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+s.apiKey)

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("GET dns records: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET dns records: HTTP %d", resp.StatusCode)
	}

	// The API returns { "dns": [ {...}, ... ] }
	var body struct {
		DNS []tsDNSRecord `json:"dns"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("decode dns records response: %w", err)
	}
	return body.DNS, nil
}

func (s *TailscaleSyncer) createDNSRecord(ctx context.Context, ip string) error {
	url := fmt.Sprintf("%s/api/v2/tailnet/%s/dns/records", tailscaleAPIBase, s.tailnet)

	body, err := json.Marshal(tsDNSRecord{Name: s.dnsRecord, Type: "A", Value: ip})
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("POST dns record: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("POST dns record: HTTP %d", resp.StatusCode)
	}
	return nil
}

func (s *TailscaleSyncer) deleteDNSRecord(ctx context.Context, id string) error {
	url := fmt.Sprintf("%s/api/v2/tailnet/%s/dns/records/%s", tailscaleAPIBase, s.tailnet, id)

	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.apiKey)

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("DELETE dns record %s: %w", id, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		return fmt.Errorf("DELETE dns record %s: HTTP %d", id, resp.StatusCode)
	}
	return nil
}
