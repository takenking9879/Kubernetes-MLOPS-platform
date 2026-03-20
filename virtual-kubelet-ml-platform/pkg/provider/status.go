package provider

import (
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/ml-platform/vk-ml/pkg/compute"
)

// BuildPodStatus translates a compute.Instance into a Kubernetes PodStatus.
//
// Networking fields populated:
//   - PodIP  = Instance.PublicIP  (RunPod public IPv4, or empty for LocalProvider)
//   - HostIP = Instance.PublicIP
//
// Port mapping annotation:
//   When Instance.PortMappings is non-empty, the pod annotations are enriched
//   with "vk-ml/port-<internal>=<external>" entries so users can discover the
//   actual public port for each service. For example:
//     vk-ml/port-6379 = "10341"
//   means Ray GCS is reachable at PodIP:10341 (not at PodIP:6379).
func BuildPodStatus(pod *corev1.Pod, inst *compute.Instance) *corev1.PodStatus {
	now := metav1.NewTime(time.Now())

	containerName := ""
	containerImage := ""
	if len(pod.Spec.Containers) > 0 {
		containerName = pod.Spec.Containers[0].Name
		containerImage = pod.Spec.Containers[0].Image
	}

	phase, reason, containerState := translateInstanceStatus(inst)

	status := &corev1.PodStatus{
		Phase:  phase,
		HostIP: inst.PublicIP,
		PodIP:  inst.PublicIP,
		Reason: reason,
		Conditions: []corev1.PodCondition{
			{
				Type:               corev1.PodScheduled,
				Status:             corev1.ConditionTrue,
				LastTransitionTime: now,
			},
			{
				Type:               corev1.PodReady,
				Status:             readyCondition(phase),
				LastTransitionTime: now,
			},
			{
				Type:               corev1.ContainersReady,
				Status:             readyCondition(phase),
				LastTransitionTime: now,
			},
		},
	}

	if containerName != "" {
		status.ContainerStatuses = []corev1.ContainerStatus{
			{
				Name:         containerName,
				Image:        containerImage,
				Ready:        phase == corev1.PodRunning,
				RestartCount: 0,
				State:        containerState,
			},
		}
	}

	return status
}

// PortMappingAnnotations returns a map of pod annotation key→value pairs that
// expose the RunPod external port for each internal port.
//
// Example output:
//
//	{"vk-ml/port-6379": "10341", "vk-ml/port-8265": "10342"}
//
// These annotations let the Ray head (or operators) discover the actual TCP
// port to use when connecting to this worker from outside RunPod.
func PortMappingAnnotations(portMappings map[string]int) map[string]string {
	if len(portMappings) == 0 {
		return nil
	}
	ann := make(map[string]string, len(portMappings))
	for internal, external := range portMappings {
		ann[fmt.Sprintf("vk-ml/port-%s", internal)] = fmt.Sprintf("%d", external)
	}
	return ann
}

// ── internal translation ──────────────────────────────────────────────────────

func translateInstanceStatus(inst *compute.Instance) (corev1.PodPhase, string, corev1.ContainerState) {
	now := metav1.NewTime(time.Now())

	switch inst.Status {
	case compute.StatusStarting:
		return corev1.PodPending, "ContainerCreating", corev1.ContainerState{
			Waiting: &corev1.ContainerStateWaiting{
				Reason:  "ContainerCreating",
				Message: "Waiting for compute instance to start",
			},
		}

	case compute.StatusRunning:
		startedAt := metav1.NewTime(inst.StartedAt)
		return corev1.PodRunning, "Running", corev1.ContainerState{
			Running: &corev1.ContainerStateRunning{StartedAt: startedAt},
		}

	case compute.StatusExited:
		if inst.ExitCode == 0 {
			return corev1.PodSucceeded, "Completed", corev1.ContainerState{
				Terminated: &corev1.ContainerStateTerminated{
					ExitCode:   int32(inst.ExitCode),
					FinishedAt: now,
					Reason:     "Completed",
				},
			}
		}
		return corev1.PodFailed, "Error", corev1.ContainerState{
			Terminated: &corev1.ContainerStateTerminated{
				ExitCode:   int32(inst.ExitCode),
				FinishedAt: now,
				Reason:     "Error",
			},
		}

	case compute.StatusTerminated:
		return corev1.PodSucceeded, "Terminated", corev1.ContainerState{
			Terminated: &corev1.ContainerStateTerminated{
				ExitCode:   0,
				FinishedAt: now,
				Reason:     "Terminated",
			},
		}

	case compute.StatusNotFound:
		return corev1.PodFailed, "InstanceNotFound", corev1.ContainerState{
			Terminated: &corev1.ContainerStateTerminated{
				ExitCode: 1,
				Reason:   "InstanceNotFound",
				Message:  "Compute instance no longer exists",
			},
		}

	default: // StatusUnknown
		return corev1.PodUnknown, string(inst.Status), corev1.ContainerState{
			Waiting: &corev1.ContainerStateWaiting{
				Reason: "Unknown",
			},
		}
	}
}

func readyCondition(phase corev1.PodPhase) corev1.ConditionStatus {
	if phase == corev1.PodRunning {
		return corev1.ConditionTrue
	}
	return corev1.ConditionFalse
}
