# Resource Allocation Logic (SkyPilot + Ray)

This document explains how workers, CPUs, GPUs, and tuning concurrency are decided in the current pipeline.

## 1) High-level strategy

### Single-node mode
Condition: `NUM_NODES == 1`

- `NUM_WORKERS = NUM_GPUS`
- `NUM_WORKERS_TUNE = 1` (one worker per trial)
- `MAX_CONCURRENT_TRIALS = NUM_WORKERS`
- `GPUS_PER_WORKER = 1`
- `GPUS_PER_WORKER_TUNE = GPUS_PER_WORKER`

Interpretation:
- If there is 1 GPU, only 1 concurrent trial.
- If there are 2 GPUs, 2 concurrent trials.
- Final training uses all GPUs through multiple workers (1 GPU each).

### Multi-node mode
Condition: `NUM_NODES > 1`

- `NUM_WORKERS = NUM_NODES`
- `NUM_WORKERS_TUNE = 1` (one worker per trial)
- `MAX_CONCURRENT_TRIALS = NUM_WORKERS` (same as node count)
- `GPUS_PER_WORKER = NUM_GPUS_PER_NODE`
- `GPUS_PER_WORKER_TUNE = GPUS_PER_WORKER`

Interpretation:
- 1 worker maps to 1 node.
- Each worker uses all GPUs on its node.
- Tuning runs 1 trial per node concurrently.

## 2) CPU allocation formula

CPU per worker is computed with:

- `N = total CPUs visible on node` (`nproc` fallback `getconf _NPROCESSORS_ONLN`)
- `M = workers used in the formula context`

Function:

- If `N % M == 0`: `f(N, M) = (N // M) - 1`
- Else: `f(N, M) = N // M`
- Clamp: `f(N, M) >= 1`

This guarantees equal CPUs per worker and keeps leftover CPU.

### How M is chosen

- Single-node templates: `M = NUM_WORKERS`
- Multi-node template: one worker per node for CPU sizing, so `M = 1`

Then:

- `CPUS_PER_WORKER = f(N, M)`
- `CPUS_PER_WORKER_TUNE = CPUS_PER_WORKER`

## 3) Why tuning and training can match cluster demand

Under this strategy:

- Training GPU demand:
  - `GPU_train = NUM_WORKERS * GPUS_PER_WORKER`
- Tuning GPU demand:
  - `GPU_tune = MAX_CONCURRENT_TRIALS * NUM_WORKERS_TUNE * GPUS_PER_WORKER_TUNE`

Given the configured values, `GPU_tune == GPU_train`.

Same logic for CPU demand:

- `CPU_train = NUM_WORKERS * CPUS_PER_WORKER`
- `CPU_tune = MAX_CONCURRENT_TRIALS * NUM_WORKERS_TUNE * CPUS_PER_WORKER_TUNE`

Given the configured values, `CPU_tune == CPU_train`.

## 4) Source of truth and precedence

1. Airflow computes and injects:
   - `NUM_WORKERS`
   - `NUM_WORKERS_TUNE`
   - `MAX_CONCURRENT_TRIALS`

2. Sky YAML runtime blocks apply fallbacks only if vars are missing.

3. Ray trainer/tuner read env vars and build `ScalingConfig`:
   - Training reads: `NUM_WORKERS`, `CPUS_PER_WORKER`, `GPUS_PER_WORKER`
   - Tuning reads: `NUM_WORKERS_TUNE`, `CPUS_PER_WORKER_TUNE`, `GPUS_PER_WORKER_TUNE`, `MAX_CONCURRENT_TRIALS`

## 5) Implementation map

- Airflow worker strategy:
  - `k3s/airflow/dags/training_dag_skypilot.py`
- Single-node runtime fallback:
  - `k3s/sky/ray-gpu-training-runpod.yaml`
  - `k3s/sky/ray-gpu-training-vast.yaml`
- Multi-node runtime fallback:
  - `k3s/sky/ray-gpu-multinode-aws.yaml`
- Ray training scaling config:
  - `src/pipeline/base_trainer.py`
- Ray tuning scaling config:
  - `src/pipeline/base_tuner.py`

## 6) Runtime validation in logs

Look for these lines:

1. Airflow plan line:
- `GPU resource plan: ... training workers=... tune workers_per_trial=... concurrent_trials=...`

2. Sky runtime line:
- `Ray params: NUM_WORKERS=... NUM_WORKERS_TUNE=... MAX_CONCURRENT_TRIALS=... GPUS_PER_WORKER=... CPUS_PER_WORKER=...`

3. Ray Train line:
- `Attempting to start training worker group of size X with the following resources: ...`

4. Worker creation lines:
- `Started training worker group of size X`
- `world_rank=...` (count ranks to verify worker count)

5. Per-worker CPU threads:
- `[pytorch_utils] Worker using Y CPU thread(s)`

## 7) Practical examples

### Example A: single-node, 2 GPUs, 24 CPUs

- `NUM_WORKERS = 2`
- `NUM_WORKERS_TUNE = 1`
- `MAX_CONCURRENT_TRIALS = 2`
- `GPUS_PER_WORKER = 1`
- `CPUS_PER_WORKER = f(24, 2) = 11`

Demands:
- Training: `2 * 1 = 2 GPU`, `2 * 11 = 22 CPU`
- Tuning: `2 * 1 * 1 = 2 GPU`, `2 * 1 * 11 = 22 CPU`

### Example B: multi-node, 3 nodes, 4 GPUs/node, 24 CPUs/node

- `NUM_WORKERS = 3`
- `NUM_WORKERS_TUNE = 1`
- `MAX_CONCURRENT_TRIALS = 3`
- `GPUS_PER_WORKER = 4`
- `CPUS_PER_WORKER = f(24, 1) = 23`

Demands:
- Training: `3 * 4 = 12 GPU`, `3 * 23 = 69 CPU`
- Tuning: `3 * 1 * 4 = 12 GPU`, `3 * 1 * 23 = 69 CPU`
