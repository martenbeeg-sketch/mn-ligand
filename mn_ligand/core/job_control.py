from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mn_ligand.core.artifacts import write_artifact_manifest
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code


CANCELLABLE_STATES = frozenset({"queued", "preparing", "running", "paused"})
RETRYABLE_STATES = frozenset({"failed", "cancelled"})
RETRYABLE_FINALIZERS = frozenset(
    {
        "pocket_detection",
        "docking_campaign",
        "openvs_docking",
        "alphafold3_refolding",
        "boltz2_refolding",
        "nesso_affinity",
        "posebusters_validation",
        "md_mmgbsa",
    }
)

_RUNTIME_METADATA_KEYS = frozenset(
    {
        "run_id",
        "job_code",
        "status",
        "error",
        "queued_at",
        "started_at",
        "completed_at",
        "updated_at",
        "returncode",
        "worker_id",
        "worker_pid",
        "worker_heartbeat_at",
        "selected_gpu",
        "executed_command",
        "executed_commands",
        "command_index",
        "command_count",
        "stdout_tail",
        "stderr_tail",
        "cancellation_requested",
        "cancellation_requested_at",
        "cancellation_requested_by",
        "cancelled_at",
        "finalizer_error",
        "orchestration_error",
    }
)


@dataclass(frozen=True)
class JobActionEligibility:
    allowed: bool
    reason: str = ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def cancellation_eligibility(job: JobRecord) -> JobActionEligibility:
    if job.task_group == "workflows":
        return JobActionEligibility(False, "Cancel individual runnable children, not the workflow record.")
    if job.status not in CANCELLABLE_STATES:
        return JobActionEligibility(False, f"Jobs in state {job.status!r} cannot be cancelled.")
    if not _commands(job.metadata):
        return JobActionEligibility(False, "This record has no valid worker-owned command to cancel.")
    return JobActionEligibility(True)


def request_job_cancellation(job: JobRecord, *, requested_by: str = "ui") -> JobRecord:
    eligibility = cancellation_eligibility(job)
    if not eligibility.allowed:
        raise ValueError(eligibility.reason)
    metadata_path = job.run_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    current = JobRecord.load(job.run_dir, task_group=job.task_group)
    eligibility = cancellation_eligibility(current)
    if not eligibility.allowed:
        raise ValueError(eligibility.reason)
    now = _utc_now_iso()
    metadata.update(
        {
            "cancellation_requested": True,
            "cancellation_requested_at": now,
            "cancellation_requested_by": requested_by,
            "updated_at": now,
        }
    )
    if current.status in {"queued", "paused"}:
        metadata.update({"status": "cancelled", "cancelled_at": now, "completed_at": now})
    _write_json(metadata_path, metadata)
    return JobRecord.load(job.run_dir, task_group=job.task_group)


def _commands(metadata: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    raw = metadata.get("queued_commands")
    if not isinstance(raw, list) or not raw:
        raw = [metadata.get("queued_command")]
    if any(not isinstance(command, list) or not command for command in raw):
        return ()
    if any(not isinstance(value, (str, int, float)) for command in raw for value in command):
        return ()
    return tuple(tuple(str(value) for value in command) for command in raw)


def _retry_inputs(job: JobRecord, commands: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    finalizer = str(job.metadata.get("worker_finalizer") or "")
    if finalizer == "pocket_detection":
        return ("input.json", "runner_input.json")
    if finalizer == "docking_campaign":
        return (
            ("input.json", "input", "prepared")
            if (job.run_dir / "prepared").is_dir()
            else ("input.json", "input")
        )
    if finalizer == "openvs_docking":
        return ("input.json", "input")
    if finalizer == "alphafold3_refolding":
        # Preserve both the reusable model inputs and any already-processed
        # data.  An inference-only AF3 command has one command, but an MSA-first
        # retry still needs ``inputs`` so its dependency barrier can hydrate a
        # fresh ``data`` directory.  Choosing one directory from command count
        # alone can produce a valid-looking run with zero fold jobs.
        staged = tuple(
            relative
            for relative in ("inputs", "data")
            if (job.run_dir / relative).is_dir()
        )
        return ("input.json", *staged)
    if finalizer == "boltz2_refolding":
        return (
            ("input.json", "inputs", "msa")
            if (job.run_dir / "msa").is_dir()
            else ("input.json", "inputs")
        )
    if finalizer == "nesso_affinity":
        return ("input.json", "inputs")
    if finalizer == "posebusters_validation":
        return ("input.json", "input")
    if finalizer == "md_mmgbsa":
        return ("input.json", "source_input.json", "source_result.json")
    return ()


def retry_eligibility(
    job: JobRecord, *, allow_completed_replay: bool = False
) -> JobActionEligibility:
    allowed_states = (
        RETRYABLE_STATES | {"completed"}
        if allow_completed_replay
        else RETRYABLE_STATES
    )
    if job.status not in allowed_states:
        return JobActionEligibility(False, "Only failed or cancelled jobs can be retried.")
    if job.workflow_id or job.workflow_parent_run_id:
        return JobActionEligibility(
            False,
            "Workflow-managed children require parent-aware replacement and are not safely retryable yet.",
        )
    finalizer = str(job.metadata.get("worker_finalizer") or "")
    if finalizer not in RETRYABLE_FINALIZERS:
        return JobActionEligibility(False, "This job type has no verified immutable retry contract.")
    commands = _commands(job.metadata)
    if not commands:
        return JobActionEligibility(False, "The original queued command is missing or invalid.")
    required = _retry_inputs(job, commands)
    missing = [relative for relative in required if not (job.run_dir / relative).exists()]
    if missing:
        return JobActionEligibility(False, "Missing staged retry input(s): " + ", ".join(missing))
    return JobActionEligibility(True)


def _rewrite_value(value: Any, source_dir: Path, retry_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_value(item, source_dir, retry_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_value(item, source_dir, retry_dir) for item in value]
    if not isinstance(value, str):
        return value
    return value.replace(str(source_dir.resolve()), str(retry_dir.resolve()))


def _copy_retry_input(source_dir: Path, retry_dir: Path, relative: str) -> None:
    source = source_dir / relative
    target = retry_dir / relative
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _next_retry_attempt(job: JobRecord, root_run_id: str) -> int:
    attempts = [int(job.metadata.get("retry_attempt") or 0)]
    for path in job.run_dir.parent.iterdir():
        if not path.is_dir() or path == job.run_dir:
            continue
        metadata = _read_json(path / "metadata.json")
        if str(metadata.get("retry_root_run_id") or "") == root_run_id:
            attempts.append(int(metadata.get("retry_attempt") or 0))
    return max(attempts, default=0) + 1


def create_job_retry(
    job: JobRecord,
    *,
    requested_by: str = "ui",
    allow_completed_replay: bool = False,
) -> JobRecord:
    current = JobRecord.load(job.run_dir, task_group=job.task_group)
    eligibility = retry_eligibility(
        current, allow_completed_replay=allow_completed_replay
    )
    if not eligibility.allowed:
        raise ValueError(eligibility.reason)
    commands = _commands(current.metadata)
    run_id = str(uuid4())
    retry_dir = current.run_dir.parent / run_id
    retry_dir.mkdir(parents=True, exist_ok=False)
    try:
        copied_inputs = _retry_inputs(current, commands)
        for relative in copied_inputs:
            _copy_retry_input(current.run_dir, retry_dir, relative)
        for path in retry_dir.rglob("*.json"):
            payload = _read_json(path)
            if payload:
                _write_json(path, _rewrite_value(payload, current.run_dir, retry_dir))
        rewritten_commands = tuple(
            tuple(str(_rewrite_value(value, current.run_dir, retry_dir)) for value in command)
            for command in commands
        )
        metadata = {
            key: value
            for key, value in current.metadata.items()
            if key not in _RUNTIME_METADATA_KEYS
        }
        now = _utc_now_iso()
        root_run_id = str(current.metadata.get("retry_root_run_id") or current.run_id)
        attempt = _next_retry_attempt(current, root_run_id)
        metadata.update(
            {
                "schema_version": int(current.metadata.get("schema_version") or JOB_SCHEMA_VERSION),
                "run_id": run_id,
                "job_code": short_job_code(run_id),
                "status": "queued",
                "retry_of_run_id": current.run_id,
                "retry_root_run_id": root_run_id,
                "retry_attempt": attempt,
                "retry_requested_by": requested_by,
                "retry_source_status": current.status,
                "retry_inputs": list(copied_inputs),
                "queued_at": now,
                "created_at": now,
                "updated_at": now,
                "queued_command": list(rewritten_commands[0]),
            }
        )
        if len(rewritten_commands) > 1 or "queued_commands" in current.metadata:
            metadata["queued_commands"] = [list(command) for command in rewritten_commands]
        else:
            metadata.pop("queued_commands", None)
        _write_json(retry_dir / "metadata.json", metadata)
        write_artifact_manifest(retry_dir, [])
        command_record = _read_json(current.run_dir / "command.json")
        if command_record:
            command_record = _rewrite_value(command_record, current.run_dir, retry_dir)
            command_record.update(
                {
                    "created_at": now,
                    "retry_of_run_id": current.run_id,
                    "retry_attempt": attempt,
                    "argv": list(rewritten_commands[0]),
                    "commands": [list(command) for command in rewritten_commands],
                }
            )
            _write_json(retry_dir / "command.json", command_record)
        return JobRecord.load(retry_dir, task_group=current.task_group)
    except Exception:
        shutil.rmtree(retry_dir, ignore_errors=True)
        raise
