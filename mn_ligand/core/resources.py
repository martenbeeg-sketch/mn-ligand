"""MN-Ligand compatibility facade for the shared compute scheduler library."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from mn_compute_scheduler.resources import (
    AdmissionDecision,
    AdmissionRequest,
    CPULease,
    FileLease,
    GPUCapacity,
    MemoryLease,
    ResourceSnapshot,
    _memory_capacity,
    acquire_cpu_lease,
    acquire_first_gpu_lease,
    acquire_gpu_lease,
    acquire_job_claim,
    acquire_memory_lease,
    active_resource_reservations,
    assess_resource_admission,
    capture_resource_snapshot,
    discover_gpu_capacity,
    discover_gpu_ids,
    gpu_ids_from_command,
    select_gpu_in_command,
    shared_state_dir,
    worker_state_dir as _worker_state_dir,
)
from mn_compute_scheduler.resources import cpu_pool_capacity as _shared_cpu_pool_capacity

from mn_ligand.runtime import runs_root


def worker_state_dir(runs_dir: Path | None = None) -> Path:
    """Keep the historical app-local path for callers that request it explicitly."""
    return _worker_state_dir(runs_dir or runs_root())


def cpu_pool_capacity(host_cpu_threads: int | None = None) -> int:
    """Return the host-wide pool capacity, shared with MN-Protein-Design."""
    return _shared_cpu_pool_capacity(host_cpu_threads)


__all__ = [
    "AdmissionDecision",
    "AdmissionRequest",
    "CPULease",
    "FileLease",
    "GPUCapacity",
    "MemoryLease",
    "ResourceSnapshot",
    "_memory_capacity",
    "acquire_cpu_lease",
    "acquire_first_gpu_lease",
    "acquire_gpu_lease",
    "acquire_job_claim",
    "acquire_memory_lease",
    "active_resource_reservations",
    "assess_resource_admission",
    "capture_resource_snapshot",
    "cpu_pool_capacity",
    "discover_gpu_capacity",
    "discover_gpu_ids",
    "gpu_ids_from_command",
    "select_gpu_in_command",
    "shared_state_dir",
    "worker_state_dir",
]
