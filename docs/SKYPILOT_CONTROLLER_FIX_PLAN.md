# SkyPilot Controller and Launch Stability Fix Plan

## Goal
Stabilize SkyPilot managed jobs in Kubernetes API-server consolidation mode so that:
1. back-to-back submissions do not trigger cross-controller cancellation/state corruption
2. launch workflows do not wait indefinitely in the "Launching on ..." phase

## Root Cause Summary
1. Multiple controller processes are expected (memory-based parallelism), but controller liveness used strict process create_time equality.
2. Controller launch path could record PID metadata that later drifted from observed process create_time, causing false dead detection.
3. False dead detection triggered repeated controller respawns and overlapping ownership over the same managed job IDs.
4. Overlap produced state-machine conflicts (row-count=0 transitions, cancellation by another process, FAILED_CONTROLLER).
5. In consolidated mode, recovery flow awaited sdk.launch/sdk.stream_and_get without a bounded timeout, so provider-side stalls could look indefinitely stuck at "Launching on ...".
6. Independently, RunPod cluster info could miss internal_ip, raising FetchClusterInfoError and amplifying recovery churn.

## Production Fix Design
1. Controller launch integrity
- Force controller command to run with exec to preserve stable PID identity from launch to runtime.
- Guard optional activation script so missing runtime env does not perturb launch path.

2. Controller liveness hardening
- Replace strict create_time equality with bounded drift tolerance.
- Keep stale/reused PID protection by still failing when drift exceeds threshold.

3. Managed-job launch timeout safeguards
- Wrap sdk.launch with asyncio.wait_for using SKY_JOBS_LAUNCH_TIMEOUT_SECONDS (default 900s).
- Wrap sdk.stream_and_get with asyncio.wait_for using the same timeout budget.
- Keep failover/retry semantics, but prevent indefinite wait in a single launch attempt.

4. RunPod metadata resilience
- Fallback when internal_ip is absent:
  - internal_ip <- internal_ip or external_ip
  - external_ip <- external_ip or internal_ip
  - ssh_port <- 22 default if omitted

5. Delivery mechanism
- Apply via Helm apiService.preDeployHook patching installed SkyPilot package paths at pod startup.
- Ensure patch is idempotent and emits explicit patch labels in startup logs.

Current patch labels expected at startup:
- recovery_strategy.skip_nested_api_start
- recovery_strategy.launch_timeout_start
- recovery_strategy.launch_timeout_end
- recovery_strategy.stream_and_get_timeout
- scheduler.controller_launch_exec_and_activate_guard
- jobs_utils.controller_liveness_create_time_tolerance
- runpod.instance_cluster_info_ip_fallback

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

5. Verify launch-timeout behavior:
- Under constrained provider capacity, logs should show explicit ResourcesUnavailableError/retry/timeout progression.
- No single managed-job launch attempt should block forever in sdk.launch/sdk.stream_and_get.

## Rollout Plan
1. Apply Helm values update with the patch set.
2. Roll API deployment and ensure new pod becomes Ready.
3. Run the double-submit validation protocol.
4. Optionally tune SKY_JOBS_LAUNCH_TIMEOUT_SECONDS for your environment.
5. Observe for 24h under normal load.
6. If stable, keep patch while upstream fix is unavailable.

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
5. Keep one launch-timeout regression test to prevent reintroducing indefinite launch waits.

## Known Limitations
1. "Launching on ..." is emitted before provider allocation is confirmed; it indicates attempt start, not successful capacity allocation.
2. User-facing logs may still appear quiet between retries due to upstream log-stream filtering behavior.
3. Additional observability (instance request sent, provider request-id, readiness heartbeat) is recommended but not part of this patch set.
