from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mn_ligand.core.resources import (
    AdmissionRequest,
    FileLease,
    GPUCapacity,
    ResourceSnapshot,
    acquire_cpu_lease,
    acquire_gpu_lease,
    assess_resource_admission,
    gpu_ids_from_command,
    select_gpu_in_command,
)


def test_gpu_command_parsing_and_selection() -> None:
    command = ["docker", "run", "--gpus", "all", "image:latest"]

    assert gpu_ids_from_command(command) is None
    assert select_gpu_in_command(command, 1)[3] == "device=1"
    assert gpu_ids_from_command(["docker", "run", "--gpus", "device=1,2", "image"]) == (1, 2)


def test_per_gpu_leases_allow_independent_devices(tmp_path: Path) -> None:
    first = acquire_gpu_lease(0, run_id="run-a", worker_id="worker-a", runs_dir=tmp_path)
    second = acquire_gpu_lease(1, run_id="run-b", worker_id="worker-b", runs_dir=tmp_path)
    blocked = acquire_gpu_lease(0, run_id="run-c", worker_id="worker-c", runs_dir=tmp_path)

    assert first is not None
    assert second is not None
    assert blocked is None
    assert first.release()
    assert second.release()


def test_shared_cpu_pool_reserves_exact_slots_and_prevents_oversubscription(
    tmp_path: Path,
) -> None:
    first = acquire_cpu_lease(
        12,
        run_id="run-a",
        worker_id="worker-a",
        runs_dir=tmp_path,
        capacity=16,
    )
    blocked = acquire_cpu_lease(
        5,
        run_id="run-b",
        worker_id="worker-b",
        runs_dir=tmp_path,
        capacity=16,
    )
    second = acquire_cpu_lease(
        4,
        run_id="run-c",
        worker_id="worker-c",
        runs_dir=tmp_path,
        capacity=16,
    )

    assert first is not None and first.threads == 12
    assert blocked is None
    assert second is not None and second.threads == 4
    assert set(first.slot_ids).isdisjoint(second.slot_ids)
    assert len(list((tmp_path / ".worker" / "locks").glob("cpu-[0-9]*.json"))) == 16
    assert first.release()
    assert second.release()


def test_shared_cpu_pool_recovers_stale_dead_slot(tmp_path: Path) -> None:
    lock_dir = tmp_path / ".worker" / "locks"
    lock_dir.mkdir(parents=True)
    (lock_dir / "cpu-0.json").write_text(
        json.dumps(
            {
                "run_id": "old-run",
                "owner_id": "old-worker",
                "pid": 99999999,
                "cpu_slot": 0,
                "heartbeat_at": (
                    datetime.now(timezone.utc) - timedelta(hours=1)
                ).isoformat(),
            }
        )
    )

    lease = acquire_cpu_lease(
        1,
        run_id="new-run",
        worker_id="new-worker",
        runs_dir=tmp_path,
        capacity=2,
        stale_after_seconds=1,
    )

    assert lease is not None and lease.slot_ids == (0,)
    assert json.loads((lock_dir / "cpu-0.json").read_text())["run_id"] == "new-run"
    assert lease.release()


def test_lease_release_requires_matching_owner(tmp_path: Path) -> None:
    path = tmp_path / "lease.json"
    lease = FileLease.acquire(path, run_id="run-a", owner_id="worker-a")
    assert lease is not None
    payload = json.loads(path.read_text())
    payload["owner_id"] = "worker-b"
    path.write_text(json.dumps(payload))

    assert lease.release() is False
    assert path.exists()


def test_stale_dead_process_lease_is_recovered(tmp_path: Path) -> None:
    path = tmp_path / "lease.json"
    stale = {
        "run_id": "old-run",
        "owner_id": "old-worker",
        "pid": 99999999,
        "heartbeat_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
    }
    path.write_text(json.dumps(stale))

    lease = FileLease.acquire(
        path,
        run_id="new-run",
        owner_id="new-worker",
        stale_after_seconds=1,
    )

    assert lease is not None
    assert json.loads(path.read_text())["run_id"] == "new-run"
    assert lease.release()


def test_resource_admission_filters_gpu_by_free_vram() -> None:
    snapshot = ResourceSnapshot(
        cpu_threads_total=16,
        ram_available_gb=48,
        ram_total_gb=64,
        scratch_free_gb=200,
        scratch_total_gb=500,
        gpus=(
            GPUCapacity(0, free_vram_gb=6, total_vram_gb=24),
            GPUCapacity(1, free_vram_gb=18, total_vram_gb=24),
        ),
    )

    decision = assess_resource_admission(
        AdmissionRequest(gpu=True, min_vram_gb=16, cpu_threads=8, ram_gb=32, scratch_gb=20),
        snapshot,
        candidate_gpu_ids=(0, 1),
    )

    assert decision.allowed is True
    assert decision.eligible_gpu_ids == (1,)


def test_resource_admission_distinguishes_waiting_from_impossible() -> None:
    snapshot = ResourceSnapshot(
        cpu_threads_total=8,
        ram_available_gb=6,
        ram_total_gb=32,
        scratch_free_gb=4,
        scratch_total_gb=100,
        gpus=(GPUCapacity(0, free_vram_gb=4, total_vram_gb=24),),
    )

    waiting = assess_resource_admission(
        AdmissionRequest(gpu=True, min_vram_gb=8, cpu_threads=4, ram_gb=16, scratch_gb=20),
        snapshot,
        candidate_gpu_ids=(0,),
    )
    impossible = assess_resource_admission(
        AdmissionRequest(gpu=True, min_vram_gb=32, cpu_threads=12, ram_gb=64, scratch_gb=120),
        snapshot,
        candidate_gpu_ids=(0,),
    )

    assert waiting.allowed is False and waiting.permanent is False
    assert any("RAM" in reason for reason in waiting.reasons)
    assert any("scratch" in reason for reason in waiting.reasons)
    assert any("GPU 0" in reason for reason in waiting.reasons)
    assert impossible.allowed is False and impossible.permanent is True
    assert any("CPU" in reason for reason in impossible.reasons)
    assert any("capacity is insufficient" in reason for reason in impossible.reasons)


def test_admission_request_rejects_ambiguous_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="boolean"):
        AdmissionRequest.from_dict({"gpu": "false"})
    with pytest.raises(ValueError, match="finite"):
        AdmissionRequest.from_dict({"ram_gb": "nan"})
