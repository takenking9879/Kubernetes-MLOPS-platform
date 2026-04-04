# SkyPilot Controller Race Fix Plan

## Goal
Stabilize SkyPilot managed jobs in Kubernetes API-server consolidation mode so that back-to-back submissions do not trigger cross-controller cancellation/state corruption.

## Root Cause Summary
1. Multiple controller processes are expected (memory-based parallelism), but controller liveness used strict process create_time equality.
2. Controller launch path could record PID metadata that later drifted from observed process create_time, causing false dead detection.
3. False dead detection triggered repeated controller respawns and overlapping ownership over the same managed job IDs.
4. Overlap produced state-machine conflicts (row-count=0 transitions, cancellation by another process, FAILED_CONTROLLER).
5. Independently, RunPod cluster info could miss internal_ip, raising FetchClusterInfoError and amplifying recovery churn.

## Production Fix Design
1. Controller launch integrity
- Force controller command to run with exec to preserve stable PID identity from launch to runtime.
- Guard optional activation script so missing runtime env does not perturb launch path.

2. Controller liveness hardening
- Replace strict create_time equality with bounded drift tolerance.
- Keep stale/reused PID protection by still failing when drift exceeds threshold.

3. RunPod metadata resilience
- Fallback when internal_ip is absent:
  - internal_ip <- internal_ip or external_ip
  - external_ip <- external_ip or internal_ip
  - ssh_port <- 22 default if omitted

4. Delivery mechanism
- Apply via Helm apiService.preDeployHook patching installed SkyPilot package paths at pod startup.
- Ensure patch is idempotent and emits explicit patch labels in startup logs.

## Validation Protocol (Required)
1. Deploy patched API server pod.
2. Confirm startup log contains all patch labels.
3. Submit same YAML twice from active API pod:
- sky jobs launch /tmp/skytest.yaml -y
- sky jobs launch /tmp/skytest.yaml -y
4. Verify acceptance conditions:
- Two distinct managed job IDs are created.
- Each job log has exactly one controller ownership line.
- No signatures in job logs:
  - RequestCancelled
  - ManagedJobStatusError
  - Failed to set the task
  - AssertionError
  - FetchClusterInfoError (from internal_ip missing)
- scheduler_get_alive == wanted controllers.

## Rollout Plan
1. Apply Helm values update with the patch set.
2. Roll API deployment and ensure new pod becomes Ready.
3. Run the double-submit validation protocol.
4. Observe for 24h under normal load.
5. If stable, keep patch while upstream fix is unavailable.

## Rollback Plan
1. Remove added patch blocks from Helm values.
2. Helm upgrade to previous behavior.
3. Restart API server deployment.
4. Re-run one control submission to verify baseline behavior.

## Long-Term Exit Criteria
1. Upstream SkyPilot release includes equivalent fixes.
2. Remove startup monkey patch blocks.
3. Pin image/chart version to fixed upstream release.
4. Keep the double-submit regression test as a standard smoke test.
