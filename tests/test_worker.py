from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from mn_ligand.app.pages.common import queue_gpu_job
from mn_ligand.core.worker import (
    WorkerConfig,
    _cleanup_docker_container,
    _prepare_docker_command,
    iter_queued_jobs,
    run_worker_once,
)
from mn_ligand.core.resources import (
    GPUCapacity,
    ResourceSnapshot,
    acquire_cpu_lease,
)


def _snapshot(
    *,
    cpu: int = 16,
    ram_available: float = 64,
    ram_total: float = 128,
    scratch_free: float = 500,
    scratch_total: float = 1000,
    gpus: tuple[GPUCapacity, ...] = (),
) -> ResourceSnapshot:
    return ResourceSnapshot(
        cpu_threads_total=cpu,
        ram_available_gb=ram_available,
        ram_total_gb=ram_total,
        scratch_free_gb=scratch_free,
        scratch_total_gb=scratch_total,
        gpus=gpus,
    )


def _queue(
    runs_dir: Path,
    run_id: str,
    command: list[str],
    *,
    resources: dict[str, object] | None = None,
) -> Path:
    run_dir = runs_dir / "fixture-jobs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "workflow": "fixture",
                "status": "queued",
                "queued_at": f"2026-01-01T00:00:0{run_id[-1]}+00:00",
                "queued_command": command,
                "resources": resources or {"gpu": False},
            }
        )
    )
    return run_dir


def test_worker_executes_oldest_job_and_captures_logs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    newer = _queue(tmp_path, "run-2", [sys.executable, "-c", "print('newer')"])
    older = _queue(tmp_path, "run-1", [sys.executable, "-c", "print('native output')"])
    config = WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05)

    result = run_worker_once(config)

    assert result is not None and result["run_id"] == "run-1"
    metadata = json.loads((older / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["returncode"] == 0
    assert "native output" in (older / "stdout.log").read_text()
    assert not (older / ".worker-claim.json").exists()
    assert json.loads((newer / "metadata.json").read_text())["status"] == "queued"
    heartbeat = json.loads(
        (tmp_path / ".worker" / "workers" / f"{config.worker_id}.json").read_text()
    )
    assert heartbeat["state"] == "idle"
    assert heartbeat["gpu_ids"] == []


def test_worker_waits_for_declared_dependencies(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    upstream = _queue(tmp_path, "run-1", [sys.executable, "-c", "print('upstream')"])
    downstream = _queue(tmp_path, "run-2", [sys.executable, "-c", "print('downstream')"])
    metadata_path = downstream / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["depends_on_run_ids"] = ["run-1"]
    metadata_path.write_text(json.dumps(metadata))
    config = WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05)

    first = run_worker_once(config)
    second = run_worker_once(config)

    assert first is not None and first["run_id"] == upstream.name
    assert second is not None and second["run_id"] == downstream.name


def test_worker_runs_higher_priority_job_before_fifo_order(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    ordinary = _queue(tmp_path, "run-1", [sys.executable, "-c", "print('ordinary')"])
    priority = _queue(tmp_path, "run-2", [sys.executable, "-c", "print('msa')"])
    metadata_path = priority / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["queue_priority"] = 100
    metadata_path.write_text(json.dumps(metadata))

    result = run_worker_once(
        WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05)
    )

    assert result is not None and result["run_id"] == priority.name
    assert json.loads((ordinary / "metadata.json").read_text())["status"] == "queued"


def test_worker_blocks_child_when_dependency_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    upstream = _queue(tmp_path, "run-1", [sys.executable, "-c", "raise SystemExit(2)"])
    downstream = _queue(tmp_path, "run-2", [sys.executable, "-c", "print('must not run')"])
    metadata_path = downstream / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["depends_on_run_ids"] = ["run-1"]
    metadata_path.write_text(json.dumps(metadata))
    config = WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05)

    failed = run_worker_once(config)
    assert failed is not None and failed["run_id"] == upstream.name
    assert run_worker_once(config) is None
    blocked = json.loads(metadata_path.read_text())
    assert blocked["status"] == "blocked"
    assert blocked["blocked_by_run_ids"] == ["run-1"]


def test_worker_executes_ordered_command_sequence_under_one_claim(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    run_dir = _queue(tmp_path, "run-sequence", [sys.executable, "-c", "print('first')"])
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["queued_commands"] = [
        [sys.executable, "-c", "print('first')"],
        [sys.executable, "-c", "print('second')"],
    ]
    metadata_path.write_text(json.dumps(metadata))

    result = run_worker_once(
        WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05)
    )

    assert result is not None and result["status"] == "completed"
    assert "first" in (run_dir / "stdout.log").read_text()
    assert "second" in (run_dir / "stdout.log").read_text()
    completed = json.loads(metadata_path.read_text())
    assert len(completed["executed_commands"]) == 2
    assert completed["command_count"] == 2
    assert not (run_dir / ".worker-claim.json").exists()


def test_worker_assigns_requested_gpu_one_and_releases_lease(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    run_dir = _queue(
        tmp_path,
        "gpu-1",
        [sys.executable, "-c", "print('gpu fixture')"],
        resources={"gpu": True, "gpu_ids": [1]},
    )
    config = WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(0, 1), heartbeat_seconds=0.05)

    result = run_worker_once(config)

    assert result is not None and result["gpu_id"] == 1
    assert json.loads((run_dir / "metadata.json").read_text())["selected_gpu"] == 1
    assert not (tmp_path / ".worker" / "locks" / "gpu-1.json").exists()


def test_dedicated_gpu_worker_skips_cpu_jobs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    cpu_run = _queue(
        tmp_path,
        "cpu-1",
        [sys.executable, "-c", "print('cpu')"],
        resources={"gpu": False},
    )
    gpu_run = _queue(
        tmp_path,
        "gpu-2",
        [sys.executable, "-c", "print('gpu')"],
        resources={"gpu": True, "gpu_ids": [0]},
    )
    monkeypatch.setattr(
        "mn_ligand.core.worker.capture_resource_snapshot",
        lambda _path: _snapshot(
            gpus=(GPUCapacity(0, free_vram_gb=24, total_vram_gb=24),)
        ),
    )

    result = run_worker_once(
        WorkerConfig.create(
            runs_dir=tmp_path,
            gpu_ids=(0,),
            job_class="gpu",
            heartbeat_seconds=0.05,
        )
    )

    assert result is not None and result["run_id"] == gpu_run.name
    assert json.loads((cpu_run / "metadata.json").read_text())["status"] == "queued"


def test_dedicated_cpu_worker_skips_gpu_jobs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    gpu_run = _queue(
        tmp_path,
        "gpu-1",
        [sys.executable, "-c", "print('gpu')"],
        resources={"gpu": True, "gpu_ids": [0]},
    )
    cpu_run = _queue(
        tmp_path,
        "cpu-2",
        [sys.executable, "-c", "print('cpu')"],
        resources={"gpu": False},
    )

    config = WorkerConfig.create(
        runs_dir=tmp_path,
        gpu_ids=(0,),
        job_class="cpu",
        heartbeat_seconds=0.05,
    )
    result = run_worker_once(config)

    assert result is not None and result["run_id"] == cpu_run.name
    assert json.loads((gpu_run / "metadata.json").read_text())["status"] == "queued"
    heartbeat = json.loads(
        (
            tmp_path
            / ".worker"
            / "workers"
            / f"{config.worker_id}.json"
        ).read_text()
    )
    assert heartbeat["job_class"] == "cpu"
    assert heartbeat["gpu_ids"] == []


def test_worker_gpu_scope_cannot_be_overridden_by_job(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    run_dir = _queue(
        tmp_path,
        "gpu-2",
        [sys.executable, "-c", "print('must not run')"],
        resources={"gpu": True, "gpu_ids": [1]},
    )
    gpu_zero_worker = WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(0,), heartbeat_seconds=0.05)

    result = run_worker_once(gpu_zero_worker)

    assert result is None
    assert json.loads((run_dir / "metadata.json").read_text())["status"] == "queued"
    assert not (run_dir / ".worker-claim.json").exists()


def test_worker_fails_malformed_gpu_request_without_stranding_claim(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    run_dir = _queue(
        tmp_path,
        "gpu-3",
        [sys.executable, "-c", "print('must not run')"],
        resources={"gpu": True, "gpu_ids": ["invalid"]},
    )
    config = WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(0, 1), heartbeat_seconds=0.05)

    result = run_worker_once(config)

    assert result is None
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert "Invalid resource request" in metadata["error"]
    assert metadata["admission"]["status"] == "rejected"
    assert not (run_dir / ".worker-claim.json").exists()


def test_worker_rejects_new_schema_job_with_absolute_host_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    run_dir = _queue(
        tmp_path,
        "portable-1",
        [sys.executable, "-c", "print('must not run')"],
    )
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["portability_schema_version"] = 1
    metadata_path.write_text(json.dumps(metadata))
    (run_dir / "input.json").write_text(
        json.dumps({"source_path": "/home/old-machine/source.pdb"})
    )

    result = run_worker_once(
        WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05)
    )

    assert result is None
    rejected = json.loads(metadata_path.read_text())
    assert rejected["status"] == "failed"
    assert rejected["admission"]["status"] == "rejected"
    assert "portability validation failed" in rejected["error"].lower()
    assert not (run_dir / ".worker-claim.json").exists()


def test_worker_marks_nonzero_process_failed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    run_dir = _queue(tmp_path, "run-3", [sys.executable, "-c", "raise SystemExit(3)"])
    config = WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05)

    result = run_worker_once(config)

    assert result is not None and result["status"] == "failed"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["returncode"] == 3
    assert metadata["status"] == "failed"


def test_worker_honors_native_result_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    run_dir = _queue(tmp_path, "run-4", [sys.executable, "-c", "print('done')"])
    (run_dir / "result.json").write_text(json.dumps({"success": False, "error": "native validation failed"}))
    config = WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05)

    result = run_worker_once(config)

    assert result is not None and result["status"] == "failed"


def test_queue_scanner_ignores_invalid_commands(tmp_path: Path) -> None:
    _queue(tmp_path, "run-5", [])

    assert iter_queued_jobs(tmp_path) == []


def test_legacy_queue_records_explicit_gpu_request(tmp_path: Path) -> None:
    run_dir = tmp_path / "fixture-jobs" / "legacy-1"
    command = ["docker", "run", "--gpus", "device=1", "fixture:latest"]

    queue_gpu_job(run_dir, "fixture", "legacy-1", command)

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "queued"
    assert metadata["resources"] == {"gpu": True, "gpu_ids": [1]}
    assert metadata["queued_at"]


def test_worker_honors_cancellation_request(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    run_dir = _queue(tmp_path, "run-6", [sys.executable, "long-running-fixture"])
    config = WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05)

    class FakeProcess:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return int(self.returncode or 0)

    def fake_popen(*_: object, **__: object) -> Any:
        return FakeProcess()

    def request_cancellation(_: float) -> None:
        metadata_path = run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["cancellation_requested"] = True
        metadata_path.write_text(json.dumps(metadata))

    result = run_worker_once(config, popen=fake_popen, sleep=request_cancellation)

    assert result is not None and result["status"] == "cancelled"
    assert json.loads((run_dir / "metadata.json").read_text())["status"] == "cancelled"


def test_worker_enforces_runtime_budget_and_records_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    run_dir = _queue(
        tmp_path,
        "run-timeout",
        [sys.executable, "long-running-fixture"],
    )
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["max_runtime_seconds"] = 5
    metadata_path.write_text(json.dumps(metadata))
    clock = [100.0]

    class FakeProcess:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return int(self.returncode or 0)

    def advance_clock(_: float) -> None:
        clock[0] += 6.0

    monkeypatch.setattr(
        "mn_ligand.core.worker.time.monotonic", lambda: clock[0]
    )
    result = run_worker_once(
        WorkerConfig.create(
            runs_dir=tmp_path,
            gpu_ids=(),
            heartbeat_seconds=0.05,
        ),
        popen=lambda *_args, **_kwargs: FakeProcess(),
        sleep=advance_clock,
    )

    assert result is not None and result["status"] == "failed"
    assert result["runtime_seconds"] == 6.0
    completed = json.loads(metadata_path.read_text())
    assert completed["timed_out"] is True
    assert completed["termination_reason"] == "runtime_budget_exceeded"
    assert completed["max_runtime_seconds"] == 5
    assert completed["runtime_deadline_at"]
    native_result = json.loads((run_dir / "result.json").read_text())
    assert native_result["success"] is False
    assert native_result["timed_out"] is True
    assert native_result["termination_reason"] == "runtime_budget_exceeded"
    assert not (run_dir / ".worker-claim.json").exists()


def test_worker_service_interrupt_fails_run_and_releases_claim(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    run_dir = _queue(tmp_path, "run-interrupted", [sys.executable, "long-running-fixture"])
    config = WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05)

    class FakeProcess:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -2

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return int(self.returncode or 0)

    with pytest.raises(KeyboardInterrupt):
        run_worker_once(
            config,
            popen=lambda *_args, **_kwargs: FakeProcess(),
            sleep=lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["returncode"] == -2
    assert "retry this preserved run" in metadata["error"]
    assert metadata["interrupted_at"]
    assert not (run_dir / ".worker-claim.json").exists()


def test_worker_adds_run_local_cidfile_and_force_removes_interrupted_container(
    tmp_path: Path, monkeypatch
) -> None:
    command, cidfile = _prepare_docker_command(
        ["docker", "run", "--rm", "fixture:latest"], tmp_path, 2
    )
    assert cidfile == tmp_path / ".worker-container-2.cid"
    assert command[:4] == ["docker", "run", "--cidfile", str(cidfile)]
    cidfile.write_text("a" * 64)
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object):
        calls.append(argv)
        return __import__("subprocess").CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr("mn_ligand.core.worker.subprocess.run", fake_run)
    assert _cleanup_docker_container(command, cidfile, force=True) == ""
    assert calls == [["docker", "rm", "--force", "a" * 64]]
    assert not cidfile.exists()


def test_worker_leaves_temporarily_underprovisioned_job_queued(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    run_dir = _queue(
        tmp_path,
        "waiting-1",
        [sys.executable, "-c", "print('must wait')"],
        resources={"gpu": False, "cpu_threads": 2, "ram_gb": 16, "scratch_gb": 4},
    )
    runnable = _queue(
        tmp_path,
        "waiting-2",
        [sys.executable, "-c", "print('smaller job admitted')"],
        resources={"gpu": False, "cpu_threads": 2, "ram_gb": 4, "scratch_gb": 2},
    )
    monkeypatch.setattr(
        "mn_ligand.core.worker.capture_resource_snapshot",
        lambda _path: _snapshot(ram_available=8),
    )

    result = run_worker_once(WorkerConfig.create(runs_dir=tmp_path, gpu_ids=()))

    assert result is not None and result["run_id"] == runnable.name
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "queued"
    assert metadata["admission"]["status"] == "waiting"
    assert "RAM" in metadata["admission"]["reasons"][0]
    assert not (run_dir / ".worker-claim.json").exists()


def test_worker_waits_for_shared_cpu_slots_then_releases_reservation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("MN_LIGAND_CPU_PROCESS_LIMIT", "4")
    run_dir = _queue(
        tmp_path,
        "cpu-pool-1",
        [sys.executable, "-c", "print('shared pool admitted')"],
        resources={"gpu": False, "cpu_threads": 2},
    )
    held = acquire_cpu_lease(
        3,
        run_id="already-running",
        worker_id="other-worker",
        runs_dir=tmp_path,
        capacity=4,
    )
    assert held is not None
    config = WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05)

    assert run_worker_once(config) is None
    waiting = json.loads((run_dir / "metadata.json").read_text())
    assert waiting["status"] == "queued"
    assert waiting["admission"]["status"] == "waiting"
    assert "shared 4-thread CPU pool" in waiting["admission"]["reasons"][0]
    assert held.release()

    result = run_worker_once(config)

    assert result is not None and result["status"] == "completed"
    assert result["cpu_threads"] == 2
    completed = json.loads((run_dir / "metadata.json").read_text())
    assert completed["reserved_cpu_threads"] == 2
    assert completed["cpu_pool_capacity"] == 4
    assert not list((tmp_path / ".worker" / "locks").glob("cpu-[0-9]*.json"))


def test_gpu_worker_uses_the_same_shared_cpu_pool(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("MN_LIGAND_CPU_PROCESS_LIMIT", "4")
    run_dir = _queue(
        tmp_path,
        "gpu-cpu-pool-1",
        [sys.executable, "-c", "print('gpu job with cpu lease')"],
        resources={"gpu": True, "gpu_ids": [0], "cpu_threads": 1},
    )
    held = acquire_cpu_lease(
        4,
        run_id="cpu-owner",
        worker_id="cpu-worker",
        runs_dir=tmp_path,
        capacity=4,
    )
    assert held is not None
    monkeypatch.setattr(
        "mn_ligand.core.worker.capture_resource_snapshot",
        lambda _path: _snapshot(
            gpus=(GPUCapacity(0, free_vram_gb=24, total_vram_gb=24),)
        ),
    )
    config = WorkerConfig.create(
        runs_dir=tmp_path,
        gpu_ids=(0,),
        job_class="gpu",
        heartbeat_seconds=0.05,
    )

    assert run_worker_once(config) is None
    assert not (tmp_path / ".worker" / "locks" / "gpu-0.json").exists()
    waiting = json.loads((run_dir / "metadata.json").read_text())
    assert "shared 4-thread CPU pool" in waiting["admission"]["reasons"][0]
    assert held.release()

    result = run_worker_once(config)

    assert result is not None and result["gpu_id"] == 0
    assert result["cpu_threads"] == 1
    assert not list((tmp_path / ".worker" / "locks").glob("cpu-[0-9]*.json"))


def test_worker_rejects_impossible_request_and_continues_queue(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    impossible = _queue(
        tmp_path,
        "impossible-1",
        [sys.executable, "-c", "print('must not run')"],
        resources={"gpu": False, "cpu_threads": 64},
    )
    runnable = _queue(
        tmp_path,
        "runnable-2",
        [sys.executable, "-c", "print('admitted')"],
        resources={"gpu": False, "cpu_threads": 2},
    )
    monkeypatch.setattr(
        "mn_ligand.core.worker.capture_resource_snapshot",
        lambda _path: _snapshot(cpu=8),
    )

    result = run_worker_once(WorkerConfig.create(runs_dir=tmp_path, gpu_ids=()))

    assert result is not None and result["run_id"] == runnable.name
    rejected = json.loads((impossible / "metadata.json").read_text())
    assert rejected["status"] == "failed"
    assert rejected["admission"]["status"] == "rejected"
    assert "CPU" in rejected["error"]


def test_worker_selects_gpu_that_meets_vram_request(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    run_dir = _queue(
        tmp_path,
        "vram-1",
        [sys.executable, "-c", "print('gpu admitted')"],
        resources={"gpu": True, "gpu_ids": [0, 1], "min_vram_gb": 12},
    )
    monkeypatch.setattr(
        "mn_ligand.core.worker.capture_resource_snapshot",
        lambda _path: _snapshot(
            gpus=(
                GPUCapacity(0, free_vram_gb=8, total_vram_gb=24),
                GPUCapacity(1, free_vram_gb=16, total_vram_gb=24),
            )
        ),
    )

    result = run_worker_once(
        WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(0, 1), heartbeat_seconds=0.05)
    )

    assert result is not None and result["gpu_id"] == 1
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["admission"]["status"] == "admitted"
    assert metadata["admission"]["eligible_gpu_ids"] == [1]
