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
