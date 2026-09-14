from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mn_ligand.core.job_control import (
    cancellation_eligibility,
    create_job_retry,
    request_job_cancellation,
    retry_eligibility,
)
from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.worker import WorkerConfig, run_worker_once


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _worker_job(
    runs_dir: Path,
    run_id: str,
    *,
    status: str,
    finalizer: str = "md_mmgbsa",
    gpu: bool = False,
) -> JobRecord:
    run_dir = runs_dir / "md-mmgbsa" / run_id
    run_dir.mkdir(parents=True)
    command = [sys.executable, "fixture-command", str(run_dir)]
    _write_json(
        run_dir / "metadata.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "job_code": run_id[-5:].upper(),
            "job_type": "md_mmgbsa",
            "workflow": "md_mmgbsa",
            "status": status,
            "worker_finalizer": finalizer,
            "queued_command": command,
            "resources": {"gpu": gpu, "gpu_ids": [0]} if gpu else {"gpu": False},
            "gpu_queued": gpu,
            "created_at": "2026-07-22T00:00:00+00:00",
        },
    )
    _write_json(run_dir / "input.json", {"source_production_run_id": "production-1"})
    _write_json(run_dir / "source_input.json", {"trajectory": "/source/production.dcd"})
    _write_json(
        run_dir / "source_result.json",
        {"success": True, "md_result": {"output_files": {}}},
    )
    _write_json(
        run_dir / "command.json",
        {"schema_version": 1, "argv": command, "commands": [command]},
    )
    return JobRecord.load(run_dir, task_group="md-mmgbsa")


def test_queued_cancellation_is_immediate_and_worker_does_not_execute(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    job = _worker_job(tmp_path, "queued-00001", status="queued")
    original_input = (job.run_dir / "input.json").read_bytes()

    cancelled = request_job_cancellation(job, requested_by="test")

    assert cancelled.status == "cancelled"
    assert cancelled.metadata["cancellation_requested"] is True
    assert cancelled.metadata["cancellation_requested_by"] == "test"
    assert (job.run_dir / "input.json").read_bytes() == original_input
    assert run_worker_once(WorkerConfig.create(runs_dir=tmp_path, gpu_ids=())) is None


def test_running_gpu_cancellation_terminates_process_and_releases_lease(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    job = _worker_job(tmp_path, "running-00002", status="queued", gpu=True)
    metadata = json.loads((job.run_dir / "metadata.json").read_text())
    metadata["queued_command"] = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "device=0",
        "fixture:latest",
    ]
    _write_json(job.run_dir / "metadata.json", metadata)

    class Process:
        returncode: int | None = None
        terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return int(self.returncode or 0)

    process = Process()

    def request_cancel(_seconds: float) -> None:
        current = JobRecord.load(job.run_dir, task_group=job.task_group)
        assert current.status == "running"
        request_job_cancellation(current, requested_by="test")

    result = run_worker_once(
        WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(0,), heartbeat_seconds=0.05),
        popen=lambda *_args, **_kwargs: process,
        sleep=request_cancel,
    )

    assert result is not None and result["status"] == "cancelled"
    assert process.terminated is True
    completed = JobRecord.load(job.run_dir, task_group=job.task_group)
    assert completed.status == "cancelled"
    assert completed.metadata["cancelled_at"]
    assert not (tmp_path / ".worker" / "locks" / "gpu-0.json").exists()
    assert not (job.run_dir / ".worker-claim.json").exists()
    assert not (job.run_dir / "native_result.json").exists()


def test_retry_is_new_immutable_job_and_can_complete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    original = _worker_job(tmp_path, "failed-00003", status="failed")
    _write_json(original.run_dir / "result.json", {"success": False, "error": "fixture"})
    (original.run_dir / "stdout.log").write_text("old log")
    original_snapshot = {
        path.relative_to(original.run_dir): path.read_bytes()
        for path in original.run_dir.rglob("*")
        if path.is_file()
    }

    retry = create_job_retry(original, requested_by="test")

    assert retry.run_id != original.run_id
    assert retry.status == "queued"
    assert retry.metadata["retry_of_run_id"] == original.run_id
    assert retry.metadata["retry_root_run_id"] == original.run_id
    assert retry.metadata["retry_attempt"] == 1
    assert retry.metadata["retry_inputs"] == [
        "input.json",
        "source_input.json",
        "source_result.json",
    ]
    assert str(retry.run_dir) in retry.metadata["queued_command"]
    assert str(original.run_dir) not in retry.metadata["queued_command"]
    assert not (retry.run_dir / "result.json").exists()
    assert not (retry.run_dir / "stdout.log").exists()
    assert {
        path.relative_to(original.run_dir): path.read_bytes()
        for path in original.run_dir.rglob("*")
        if path.is_file()
    } == original_snapshot

    second_retry = create_job_retry(original, requested_by="test")
    assert second_retry.metadata["retry_attempt"] == 2
    second_metadata = json.loads((second_retry.run_dir / "metadata.json").read_text())
    second_metadata["status"] = "cancelled"
    _write_json(second_retry.run_dir / "metadata.json", second_metadata)

    class ImmediateProcess:
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    def fake_popen(_command: list[str], **_kwargs: Any) -> ImmediateProcess:
        _write_json(
            retry.run_dir / "result.json",
            {
                "success": True,
                "mmgbsa": {
                    "status": "success",
                    "method": "fixture",
                    "delta": {"delta_g_bind_total_kj_mol": -1.0},
                },
            },
        )
        return ImmediateProcess()

    result = run_worker_once(
        WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05),
        popen=fake_popen,
        sleep=lambda _: None,
    )

    assert result is not None and result["run_id"] == retry.run_id
    assert result["status"] == "completed"
    assert JobRecord.load(retry.run_dir, task_group=retry.task_group).status == "completed"


def test_workflow_managed_child_is_not_safely_retryable(tmp_path: Path) -> None:
    job = _worker_job(tmp_path, "failed-00004", status="failed")
    metadata = json.loads((job.run_dir / "metadata.json").read_text())
    metadata["workflow_id"] = "workflow-1"
    _write_json(job.run_dir / "metadata.json", metadata)

    eligibility = retry_eligibility(JobRecord.load(job.run_dir, task_group=job.task_group))

    assert eligibility.allowed is False
    assert "parent-aware" in eligibility.reason


def test_alphafold_retry_preserves_inputs_and_processed_data(tmp_path: Path) -> None:
    job = _worker_job(
        tmp_path,
        "failed-af3-00005",
        status="failed",
        finalizer="alphafold3_refolding",
        gpu=True,
    )
    (job.run_dir / "inputs").mkdir()
    (job.run_dir / "data").mkdir()
    _write_json(job.run_dir / "inputs" / "compound.json", {"modelSeeds": [1]})
    _write_json(job.run_dir / "data" / "compound.json", {"modelSeeds": [1]})

    retry = create_job_retry(
        JobRecord.load(job.run_dir, task_group=job.task_group),
        requested_by="test",
    )

    assert retry.metadata["retry_inputs"] == ["input.json", "inputs", "data"]
    assert (retry.run_dir / "inputs" / "compound.json").is_file()
    assert (retry.run_dir / "data" / "compound.json").is_file()


def test_legacy_running_record_without_worker_command_is_not_cancellable(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy" / "legacy-running"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "metadata.json",
        {"schema_version": 1, "run_id": run_dir.name, "status": "running"},
    )

    job = JobRecord.load(run_dir, task_group="legacy")

    eligibility = cancellation_eligibility(job)
    assert eligibility.allowed is False
    assert "worker-owned command" in eligibility.reason
