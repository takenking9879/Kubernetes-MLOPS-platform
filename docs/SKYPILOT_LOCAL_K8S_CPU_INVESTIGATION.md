# SkyPilot Local K8s CPU Saturation Investigation

## Status

- State: active investigation
- Last updated: 2026-05-10
- Environment: local `k3s` on WSL2, 24 host CPUs
- Goal: keep only the strongest evidence and the next useful validation steps

## Problem

Running SkyPilot jobs on local Kubernetes can drive host CPU to roughly `80-100%`, making the machine sluggish:

- typing lag
- slow shells and apps
- long SkyPilot/Ray startup
- Airflow instability during hot windows

This happens even when the intended workload budget is around `8-12` CPUs.

## Important Context

- SkyPilot is deployed through `k3s/sky/helm/`.
- Jobs are launched through the normal stack: UI -> backend -> Airflow -> `sky launch`.
- Local Conda SkyPilot code is for inspection only; the real runtime is inside Kubernetes pods.
- Two Ray clusters are intentional in this setup:
  - SkyPilot Ray for orchestration/runtime
  - user Ray for compute
- Therefore, "two Ray clusters exist" is not itself a bug

## Runtime Under Inspection

- SkyPilot package path:
  - `/home/jorge/miniconda3/envs/ml-platform/lib/python3.12/site-packages/sky`
- SkyPilot version:
  - `0.12.0`

## Executive Summary

The strongest current explanation is additive overload, not a single bad process:

1. SkyPilot worker pods are being created with `requests.cpu` but without `limits.cpu`, so the pod is not CPU-capped at the cgroup level.
2. The worker pod still sees all `24` host CPUs (`nproc=24`, `cpu.max=max`), which makes host-core-based behavior possible inside the pod.
3. SkyPilot has real orchestration overhead outside the job pod:
   - API server
   - job controllers
   - polling/provisioning logic
4. Airflow also consumes meaningful CPU during launches.
5. During the worst hot window, all of these stacked on the same WSL2 machine:
   - user workload
   - user Ray
   - SkyPilot internal Ray and wrappers
   - SkyPilot API/controllers
   - Airflow
   - k3s/containerd

The current evidence does not support blaming only the training code.

## Confirmed Findings

### 1. Worker pods are not CPU hard-limited

Strong static evidence in SkyPilot `0.12.0`:

- `sky/templates/kubernetes-ray.yml.j2`
  - `SKYPILOT_POD_CPU_CORE_LIMIT` is populated from `requests.cpu`
- `sky/clouds/kubernetes.py`
  - pod resource limits are only emitted when `kubernetes.set_pod_resource_limits` is enabled

Strong live evidence from multiple SkyPilot worker pods:

- `requests.cpu=8`
- no `limits.cpu`
- `SKYPILOT_POD_CPU_CORE_LIMIT=8`
- `cpu.max = max 100000`
- `nproc = 24`
- `nproc --all = 24`
- QoS class `Burstable`

Meaning:

- SkyPilot is exposing an 8-CPU budget to the workload, but Kubernetes is not enforcing an 8-CPU cgroup cap.
- On a local single-node cluster, that makes host saturation plausible.

### 2. The local app path assumes SkyPilot CPU config is a real limit

Relevant local files:

- `k3s/airflow/dags/training_dag_skypilot.py`
- `k3s/sky/sky_runner.py`

Observed behavior:

- the app computes a CPU budget and passes it into `resources.cpus`
- the current runtime evidence shows that this becomes a request, not a hard CPU limit, unless SkyPilot limits are explicitly enabled

Meaning:

- the application path likely assumes "8 CPUs" means "cannot use more than 8 CPUs"
- that assumption is false in the current runtime

### 3. SkyPilot has host-core-based orchestration behavior

Static evidence:

- `sky/utils/subprocess_utils.py`
  - Kubernetes parallelism uses `os.cpu_count()` with a multiplier
- `sky/provision/kubernetes/instance.py`
  - uses that parallel thread count
- On a 24-core host this resolves to `92` threads

Static evidence of polling:

- `sky/provision/kubernetes/instance.py`
  - repeated readiness/scheduling polling around `0.5-1s`
- `sky/provision/kubernetes/utils.py`
  - can scan pods across namespaces

Meaning:

- SkyPilot orchestration is not free
- on local k3s, orchestration CPU directly competes with the workload on the same machine

### 4. SkyPilot API server is a real contributor, not just a thin HTTP layer

Relevant config/code:

- `k3s/sky/helm/values.yaml`
  - `apiService.useDeployMode: false`
  - `jobs.controller.consolidation_mode: true`
- runtime code:
  - `sky/server/server.py`
  - `sky/server/config.py`
  - `sky/server/daemons.py`

Live idle baseline:

- API pod approximately `24m CPU`, `1370Mi` memory
- pod has explicit `requests.cpu=4` and `limits.cpu=4`
- multiple executor and `sky.jobs.controller` processes are alive even while idle

Live hot-window evidence:

- API pod rose to about `0.85-1.27` cores
- API pod memory rose to about `5Gi`
- CPU was spread across many `sky.jobs.controller` processes plus executors

Meaning:

- SkyPilot infrastructure outside the worker pod is materially consuming host resources during launches/runs

### 5. Airflow is also part of the overload path

Live hot-window evidence:

- DAG processor reached about `1.1` cores
- triggerer reached about `2.1` cores
- scheduler reached about `1.9` cores in the failure window

Airflow failure evidence:

- `my-airflow-scheduler-0` became `2/3` ready
- scheduler restarted
- headless service `my-airflow-scheduler` had no endpoints during the error
- scheduler container had startup/liveness probe timeouts
- exit code `137`
- scheduler pod QoS class was `BestEffort`

Meaning:

- the Airflow log-read failure was not a separate mystery
- it was a downstream symptom of the same shared-node overload

### 6. The idle cluster is not the problem by itself

Idle baseline with cluster up but no new run:

- node about `913m CPU`
- node about `3%` CPU
- SkyPilot API pod about `24m CPU`

Meaning:

- `k3s` merely being up is not enough to reproduce the slowdown
- the problem emerges during launch/setup/run activity

## Observed Runtime Flow

### `sky launch`

1. UI triggers backend.
2. Backend triggers Airflow DAG.
3. Airflow builds SkyPilot config and calls `sky launch`.
4. SkyPilot API provisions the Kubernetes worker pod.
5. SkyPilot polls for scheduling/readiness.
6. Worker pod starts SkyPilot bootstrap/runtime.
7. User setup runs.
8. User Ray / user workload run.

### Runtime stages seen in live logs

The worker logs clearly showed three phases:

1. SkyPilot bootstrap/runtime setup
2. user-defined setup
3. user run start

Useful because CPU rose before the actual training code was fully running.

## Process Map

### Outside the worker pod

- SkyPilot API server
- `sky.jobs.controller` processes
- SkyPilot executors
- Airflow scheduler / triggerer / DAG processor
- k3s control-plane/container runtime

### Inside the worker pod

- SkyPilot bootstrap shell and wrappers
- SkyPilot internal Ray
- user Ray
- user setup logic
- user workload

## Live Evidence That Matters

### Worker pod during repeated live runs

Strong repeated evidence across multiple runs:

- worker pod created with `requests.cpu=8`
- no `limits.cpu`
- pod cgroup showed `cpu.max=max`
- pod still saw `24` CPUs

This is the most important confirmed finding.

### Early launch / setup

Repeated evidence:

- host/node CPU increased well before the user workload was fully running
- worker pod used only part of the total node CPU during setup
- SkyPilot API and Airflow were also active

Meaning:

- the slowdown cannot be explained by "only the training process"

### Worst hot window

Representative hot-window numbers:

- node about `11.5` cores used
- worker pod about `4.0-4.6` cores
- SkyPilot API pod about `0.85-1.27` cores
- Airflow scheduler/triggerer/DAG processor together added several more cores

Meaning:

- the worker pod was significant, but it was not the only source of pressure

### Worker process evidence in the hot window

Strongest observed signals:

- user workload `k3s/kuberay/main.py` about `95% CPU`
- many user-Ray `ray::IDLE` workers each still burning substantial CPU
- user Ray control-plane processes active
- SkyPilot internal Ray and SkyPilot wrappers also still alive

Meaning:

- by the hot window, real compute activity was present
- but the total machine slowdown was still broader than just the main training process

## What Is Strongly Supported Now

### High confidence

- SkyPilot worker pods are not CPU hard-limited in the current setup.
- The worker pod sees host CPU count, not an enforced reduced CPU count.
- SkyPilot infrastructure outside the worker pod consumes noticeable CPU and memory during runs.
- Airflow shares in the overload and can become unstable on the same machine.
- The local WSL2 slowdown is additive and cluster-wide, not a single-process story.

### Medium confidence

- SkyPilot orchestration parallelism/polling is a meaningful contributor on local k3s.
- User Ray startup/worker behavior may still be using host-visible CPU in ways that worsen startup and hot-window pressure.

### Not supported as a standalone root cause

- "Two Ray clusters exist, therefore that is the bug."
- "The training code alone explains the workstation slowdown."

## Highest-Priority Tests

### 1. Enable real CPU limits for SkyPilot worker pods

Most important control test.

Target:

- enable `kubernetes.set_pod_resource_limits: true` in the SkyPilot config used by the Helm deployment

Question it answers:

- does the problem drop sharply once the worker pod is actually cgroup-capped?

### 2. Compare `sky launch` vs `sky jobs launch`

Question it answers:

- how much extra overhead comes from managed-jobs/controller paths?

### 3. Compare `4` vs `8` vs `12` CPUs after real limits are enabled

Question it answers:

- does host CPU scale with the true pod cap, or does orchestration remain disproportionately expensive?

### 4. Inspect why user-Ray `ray::IDLE` workers are burning CPU

Question it answers:

- is user Ray itself spinning/busy in a way that materially adds to the problem after launch?

## Candidate Changes

### Config changes

1. Enable SkyPilot pod resource limits.
2. Give Airflow scheduler real resource requests/limits instead of `BestEffort`.
3. Re-check WSL2 resource allocation if Windows-side pressure is also visible.

### SkyPilot code changes to test locally

1. Reduce Kubernetes orchestration parallelism.
2. Replace host-core-derived thread counts with cgroup-aware CPU detection.
3. Audit polling frequency in provisioning paths.
4. Audit any startup logic that keys off host-visible CPU count.

## Useful Commands For The Next Session

### Pod resources

```bash
kubectl get pod -n skypilot <job-pod> -o jsonpath='{range .spec.containers[*]}{.name}{" req="}{.resources.requests.cpu}{"/"}{.resources.requests.memory}{" lim="}{.resources.limits.cpu}{"/"}{.resources.limits.memory}{"\n"}{end}'
```

### Cgroup CPU visibility

```bash
kubectl exec -n skypilot <job-pod> -- bash -lc 'echo "cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"; echo "nproc=$(nproc)"; echo "nproc_all=$(nproc --all)"'
```

### Cluster CPU split

```bash
kubectl top nodes
kubectl top pod -A --containers | egrep "skypilot|airflow|ray"
```

### Worker hot processes

```bash
kubectl exec -n skypilot <job-pod> -- bash -lc 'ps -eo pid,ppid,pcpu,pmem,comm,args --sort=-pcpu | head -n 50'
```

### API hot processes

```bash
kubectl exec -n skypilot deploy/my-skypilot-api-server -- bash -lc 'ps -eo pid,ppid,pcpu,pmem,comm,args --sort=-pcpu | head -n 50'
```

### Ray status

```bash
kubectl exec -n skypilot <job-pod> -- bash -lc 'ss -ltnp | egrep ":6379|:6380|:8265|:8266"; RAY_ADDRESS=127.0.0.1:6379 ray status; RAY_ADDRESS=127.0.0.1:6380 ray status'
```

## Provisional Conclusion

The best current conclusion is:

- SkyPilot is launching worker pods that are budgeted by request but not CPU-capped by Kubernetes.
- Those worker pods still see all `24` host CPUs.
- On local WSL2 `k3s`, that uncapped worker pod runs alongside non-trivial SkyPilot orchestration and Airflow overhead on the same machine.
- The resulting slowdown is therefore a combined local-cluster problem, not evidence that only the training code is bad.

The next decision point should be driven by one control experiment:

- rerun the same workload after enabling real SkyPilot pod CPU limits

If host saturation drops sharply, missing cgroup enforcement is the primary issue.
If it does not, the next suspect is orchestration and Ray-side overhead on the shared local node.
