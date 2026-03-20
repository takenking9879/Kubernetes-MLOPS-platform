package provider

import (
	"fmt"
	"strings"
)

// WrapArgsWithGCSWait prepends a GCS health-check loop to args[0].
//
// Mirrors the initContainer "wait-gcs-ready" that KubeRay automatically injects
// into every worker pod. The Virtual Kubelet provider skips initContainers entirely
// (only pod.Spec.Containers[0] is launched), so this function replicates the same
// wait logic inside the main container's startup command.
//
// # How it works
//
// KubeRay always generates the worker container's startup script as a single shell
// string in args[0], for example:
//
//	"ulimit -n 65536; ray start --address=<SVC>:6379 --block ..."
//
// This function prepends a GCS readiness loop to that string. The loop uses the
// $RAY_ADDRESS environment variable, which KubeRay always sets to "HOST:PORT" of
// the Ray head's GCS service. If the LocalProvider is in use, $RAY_ADDRESS has
// already been rewritten from the Kubernetes service FQDN to "127.0.0.1:30379"
// (NodePort) before the container is started.
//
// # Script behaviour (mirrors the KubeRay initContainer exactly)
//
//   - First 120 s: health-check output suppressed (quiet during normal startup)
//   - After 120 s: full health-check output shown (useful for diagnosing slow starts)
//   - After timeoutSec: exit 1 (prevents containers from hanging indefinitely)
//
// If args[0] does not contain "ray start", or $RAY_ADDRESS would be empty,
// args is returned unchanged. timeoutSec ≤ 0 defaults to 300.
func WrapArgsWithGCSWait(args []string, timeoutSec int) []string {
	if len(args) == 0 || !strings.Contains(args[0], "ray start") {
		return args
	}
	if timeoutSec <= 0 {
		timeoutSec = 300
	}

	// The wait script is identical to KubeRay's initContainer "wait-gcs-ready",
	// but reads the GCS address from $RAY_ADDRESS instead of a hardcoded FQDN.
	// $RAY_ADDRESS is always set by KubeRay (format "HOST:PORT") and has already
	// been rewritten to "127.0.0.1:30379" by LocalProvider.rewriteK8sAddresses
	// before this container is launched.
	waitLoop := fmt.Sprintf(
		"SECONDS=0\n"+
			"while true; do\n"+
			"  if (( SECONDS <= 120 )); then\n"+
			"    if ray health-check --address \"$RAY_ADDRESS\" > /dev/null 2>&1; then\n"+
			"      echo \"GCS is ready.\"; break\n"+
			"    fi\n"+
			"    echo \"$SECONDS seconds elapsed: Waiting for GCS to be ready.\"\n"+
			"  else\n"+
			"    if ray health-check --address \"$RAY_ADDRESS\"; then\n"+
			"      echo \"GCS is ready. Any error messages above can be safely ignored.\"; break\n"+
			"    fi\n"+
			"    echo \"$SECONDS seconds elapsed: Still waiting for GCS to be ready.\"\n"+
			"  fi\n"+
			"  if (( SECONDS > %d )); then\n"+
			"    echo \"Timeout: GCS at $RAY_ADDRESS not available after %d seconds. Exiting.\"\n"+
			"    exit 1\n"+
			"  fi\n"+
			"  sleep 5\n"+
			"done\n",
		timeoutSec, timeoutSec,
	)

	result := make([]string, len(args))
	copy(result, args)
	result[0] = waitLoop + args[0]
	return result
}
