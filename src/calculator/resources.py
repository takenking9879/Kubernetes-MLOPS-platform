import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceConfig:
    num_workers: int
    cpus_per_worker: int
    cpus_head: int = 1

    @classmethod
    def from_env(cls) -> "ResourceConfig":
        return cls(
            num_workers=int(os.getenv("NUM_WORKERS", "2")),
            cpus_per_worker=int(os.getenv("CPUS_PER_WORKER", "2")),
            cpus_head=int(os.getenv("CPUS_HEAD", "1")),
        )

    def total_cpus_per_trial(self) -> int:
        return self.cpus_head + self.num_workers * self.cpus_per_worker

    def total_worker_cpus(self) -> int:
        return self.num_workers * self.cpus_per_worker


@dataclass(frozen=True)
class GPUResourceConfig:
    """GPU-aware resource configuration for SkyPilot training jobs.

    Single-node tuning: 1 GPU per trial, MAX_CONCURRENT_TRIALS = total_gpus.
    Multi-node tuning:  gpus_per_node workers per trial (DDP), MAX_CONCURRENT_TRIALS = num_nodes.
    Training (both):    all GPUs participate via DDP, NUM_WORKERS = total_gpus.
    """
    num_gpus_per_node: int
    num_nodes: int = 1

    @classmethod
    def from_env(cls) -> "GPUResourceConfig":
        return cls(
            num_gpus_per_node=int(os.getenv("SKYPILOT_NUM_GPUS_PER_NODE", "1")),
            num_nodes=int(os.getenv("SKYPILOT_NUM_NODES", "1")),
        )

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.num_gpus_per_node

    def num_workers_training(self) -> int:
        """DDP workers for final training — one per GPU across all nodes."""
        return self.total_gpus

    def num_workers_tune(self) -> int:
        """Workers per trial:
        - multi-node: gpus_per_node (DDP within the node allocated to each trial)
        - single-node: 1 (each trial gets exactly 1 GPU)
        """
        return self.num_gpus_per_node if self.num_nodes > 1 else 1

    def max_concurrent_trials(self) -> int:
        """Parallel trials:
        - multi-node: num_nodes (1 trial per node)
        - single-node: total_gpus (1 trial per GPU)
        """
        return self.num_nodes if self.num_nodes > 1 else self.total_gpus

    def validate_tuning_budget(self) -> bool:
        """Check: concurrent_trials × workers_per_trial × 1 GPU <= total_gpus.
        By construction this is always True, but useful for unit tests."""
        return self.max_concurrent_trials() * self.num_workers_tune() <= self.total_gpus
