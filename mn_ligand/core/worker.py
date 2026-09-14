from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

from mn_ligand.core.resources import (
    AdmissionDecision,
    AdmissionRequest,
    CPULease,
    FileLease,
    ResourceSnapshot,
    acquire_cpu_lease,
    acquire_first_gpu_lease,
    acquire_job_claim,
    assess_resource_admission,
    capture_resource_snapshot,
    cpu_pool_capacity,
    discover_gpu_ids,
    gpu_ids_from_command,
    select_gpu_in_command,
)
from mn_ligand.runtime import runs_root


def _record_worker_state(
    config: WorkerConfig,
    state: str,
    *,
    run_id: str = "",
    selected_gpu: int | None = None,
) -> None:
    from mn_ligand.core.worker_health import record_worker_heartbeat

    record_worker_heartbeat(
        config.runs_dir,
        worker_id=config.worker_id,
        gpu_ids=config.gpu_ids,
        job_class=config.job_class,
        state=state,
        run_id=run_id,
        selected_gpu=selected_gpu,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


_METADATA_CACHE: dict[Path, tuple[tuple[int, int, int], dict[str, Any]]] = {}


def _read_cached_metadata(path: Path) -> dict[str, Any]:
    """Avoid reparsing every historical run record on every worker poll."""
    try:
        stat = path.stat()
    except OSError:
        _METADATA_CACHE.pop(path, None)
        return {}
    fingerprint = (int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _METADATA_CACHE.get(path)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    payload = _read_json(path)
    _METADATA_CACHE[path] = (fingerprint, payload)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _tail(path: Path, limit: int = 8000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode(errors="replace")
    except OSError:
        return ""


def _prepare_docker_command(
    command: list[str], run_dir: Path, command_index: int
) -> tuple[list[str], Path | None]:
    if len(command) < 2 or Path(command[0]).name != "docker" or command[1] != "run":
        return command, None
    if "--cidfile" in command:
        return command, None
    cidfile = run_dir / f".worker-container-{command_index}.cid"
    cidfile.unlink(missing_ok=True)
    return [*command[:2], "--cidfile", str(cidfile), *command[2:]], cidfile


def _cleanup_docker_container(
    command: list[str], cidfile: Path | None, *, force: bool
) -> str:
    if cidfile is None:
        return ""
    try:
        container_id = cidfile.read_text().strip()
    except OSError:
        container_id = ""
    error = ""
    if force and re.fullmatch(r"[0-9a-fA-F]{12,64}", container_id):
        completed = subprocess.run(
            [command[0], "rm", "--force", container_id],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if completed.returncode != 0 and "No such container" not in str(completed.stderr):
            detail = str(completed.stderr or completed.stdout or "unknown Docker error").strip()
            error = f"Docker container cleanup failed: {detail}"
    cidfile.unlink(missing_ok=True)
    return error


@dataclass(frozen=True)
class QueuedJob:
    run_dir: Path
    metadata: dict[str, Any]
    commands: tuple[tuple[str, ...], ...]
    queued_order: tuple[str, float]

    @property
    def command(self) -> tuple[str, ...]:
        return self.commands[0]

    @property
    def run_id(self) -> str:
        return str(self.metadata.get("run_id") or self.run_dir.name)

    @property
    def workflow(self) -> str:
        return str(self.metadata.get("workflow") or self.metadata.get("workflow_key") or "unknown")

    @property
    def needs_gpu(self) -> bool:
        resources = self.metadata.get("resources")
        resource_gpu = resources.get("gpu") if isinstance(resources, dict) else None
        return bool(
            self.metadata.get("gpu_queued")
            or resource_gpu
            or any("--gpus" in command for command in self.commands)
        )


@dataclass(frozen=True)
class WorkerConfig:
    runs_dir: Path
    worker_id: str
    gpu_ids: tuple[int, ...]
    job_class: str = "mixed"
    heartbeat_seconds: float = 2.0
    stale_after_seconds: float = 120.0
    cancellation_grace_seconds: float = 10.0

    @classmethod
    def create(
        cls,
        *,
        runs_dir: Path | None = None,
        worker_id: str = "",
        gpu_ids: Sequence[int] | None = None,
        job_class: str = "mixed",
        heartbeat_seconds: float = 2.0,
        stale_after_seconds: float = 120.0,
    ) -> WorkerConfig:
        normalized_job_class = str(job_class).strip().lower()
        if normalized_job_class not in {"cpu", "gpu", "mixed"}:
            raise ValueError("Worker job class must be cpu, gpu, or mixed")
        selected_gpu_ids = tuple(gpu_ids) if gpu_ids is not None else discover_gpu_ids()
        if normalized_job_class == "cpu":
            selected_gpu_ids = ()
        return cls(
            runs_dir=(runs_dir or runs_root()).resolve(),
            worker_id=worker_id or f"worker-{os.getpid()}-{uuid4().hex[:8]}",
            gpu_ids=selected_gpu_ids,
            job_class=normalized_job_class,
            heartbeat_seconds=max(0.05, float(heartbeat_seconds)),
            stale_after_seconds=max(1.0, float(stale_after_seconds)),
        )


def iter_queued_jobs(runs_dir: Path) -> list[QueuedJob]:
    jobs: list[QueuedJob] = []
    if not runs_dir.is_dir():
        return jobs
    # Run records have the stable ``<task-group>/<run-id>/metadata.json``
    # layout.  Do not recursively descend into multi-gigabyte run output and
    # backup trees merely to find queue records.
    for metadata_path in runs_dir.glob("*/*/metadata.json"):
        if any(part.startswith(".") for part in metadata_path.relative_to(runs_dir).parts):
            continue
        metadata = _read_cached_metadata(metadata_path)
        raw_commands = metadata.get("queued_commands")
        if not isinstance(raw_commands, list) or not raw_commands:
            raw_commands = [metadata.get("queued_command")]
        if str(metadata.get("status") or "") != "queued":
            continue
        if any(not isinstance(command, list) or not command for command in raw_commands):
            continue
        if any(
            not isinstance(value, (str, int, float))
            for command in raw_commands
            for value in command
        ):
            continue
        try:
            modified = metadata_path.stat().st_mtime
        except OSError:
            modified = 0.0
        queued_at = str(metadata.get("queued_at") or metadata.get("created_at") or "")
        jobs.append(
            QueuedJob(
                run_dir=metadata_path.parent,
                metadata=metadata,
                commands=tuple(
                    tuple(str(value) for value in command) for command in raw_commands
                ),
                queued_order=(queued_at, modified),
            )
        )
    return sorted(
        jobs,
        key=lambda job: (
            -int(job.metadata.get("queue_priority") or 0),
            job.queued_order,
        ),
    )


def _requested_gpu_ids(job: QueuedJob, available_gpu_ids: tuple[int, ...]) -> tuple[int, ...]:
    requested: tuple[int, ...] | None = None
    resources = job.metadata.get("resources")
    if isinstance(resources, dict):
        configured = resources.get("gpu_ids")
        if isinstance(configured, list) and configured:
            ids = tuple(int(value) for value in configured)
            if any(value < 0 for value in ids):
                raise ValueError("GPU IDs cannot be negative")
            requested = tuple(dict.fromkeys(ids))
    if requested is None:
        for key in ("selected_gpu", "gpu_id", "gpu_device"):
            value = job.metadata.get(key)
            if str(value).strip().isdigit():
                requested = (int(value),)
                break
    if requested is None:
        requested = gpu_ids_from_command(job.command)
    if requested is None:
        return available_gpu_ids
    allowed = set(available_gpu_ids)
    return tuple(gpu_id for gpu_id in requested if gpu_id in allowed)


def _record_admission(
    job: QueuedJob,
    *,
    request: AdmissionRequest | None,
    snapshot: ResourceSnapshot,
    decision: AdmissionDecision,
    status: str,
) -> None:
    metadata_path = job.run_dir / "metadata.json"
    metadata = _read_json(metadata_path) or dict(job.metadata)
    metadata["admission"] = {
        "status": status,
        "checked_at": _utc_now_iso(),
        "reasons": list(decision.reasons),
        "request": request.to_dict() if request is not None else dict(metadata.get("resources") or {}),
        "snapshot": snapshot.to_dict(),
        "eligible_gpu_ids": list(decision.eligible_gpu_ids),
    }
    metadata["updated_at"] = _utc_now_iso()
    if status == "rejected":
        metadata.update(
            {
                "status": "failed",
                "error": "Resource admission rejected: " + "; ".join(decision.reasons),
                "completed_at": _utc_now_iso(),
            }
        )
    _write_json(metadata_path, metadata)


def _job_statuses(runs_dir: Path) -> dict[str, str]:
    """Return immutable run IDs and their current states for dependency gating."""
    statuses: dict[str, str] = {}
    if not runs_dir.is_dir():
        return statuses
    for metadata_path in runs_dir.glob("*/*/metadata.json"):
        metadata = _read_cached_metadata(metadata_path)
        run_id = str(metadata.get("run_id") or metadata_path.parent.name)
        if run_id:
            statuses[run_id] = str(metadata.get("status") or "")
    return statuses


def _dependency_state(
    job: QueuedJob, statuses: dict[str, str]
) -> tuple[str, tuple[str, ...]]:
    dependencies = tuple(
        dict.fromkeys(
            str(value)
            for value in (job.metadata.get("depends_on_run_ids") or ())
            if str(value)
        )
    )
    if not dependencies:
        return "ready", ()
    failed = tuple(
        run_id
        for run_id in dependencies
        if statuses.get(run_id) in {"failed", "cancelled", "blocked"}
    )
    if failed:
        return "blocked", failed
    waiting = tuple(
        run_id for run_id in dependencies if statuses.get(run_id) != "completed"
    )
    return ("waiting", waiting) if waiting else ("ready", ())


def _block_dependency_job(job: QueuedJob, failed_dependencies: tuple[str, ...]) -> None:
    metadata_path = job.run_dir / "metadata.json"
    metadata = _read_json(metadata_path) or dict(job.metadata)
    now = _utc_now_iso()
    metadata.update(
        {
            "status": "blocked",
            "updated_at": now,
            "completed_at": now,
            "blocked_by_run_ids": list(failed_dependencies),
            "error": (
                "A required upstream job did not complete successfully: "
                + ", ".join(failed_dependencies)
            ),
        }
    )
    _write_json(metadata_path, metadata)


def _claim_runnable_job(
    config: WorkerConfig,
) -> tuple[QueuedJob, FileLease, CPULease, int | None, FileLease | None] | None:
    queued_jobs = iter_queued_jobs(config.runs_dir)
    if not queued_jobs:
        return None
    statuses = _job_statuses(config.runs_dir)
    snapshot = capture_resource_snapshot(config.runs_dir)
    for job in queued_jobs:
        dependency_state, dependency_ids = _dependency_state(job, statuses)
        if dependency_state == "waiting":
            continue
        if dependency_state == "blocked":
            claim = acquire_job_claim(
                job.run_dir,
                run_id=job.run_id,
                worker_id=config.worker_id,
                stale_after_seconds=config.stale_after_seconds,
            )
            if claim is not None:
                _block_dependency_job(job, dependency_ids)
                claim.release()
            continue
        try:
            resources = job.metadata.get("resources")
            request = AdmissionRequest.from_dict(
                dict(resources) if isinstance(resources, dict) else {},
                gpu_default=job.needs_gpu,
            )
            if config.job_class == "gpu" and not request.gpu:
                continue
            if config.job_class == "cpu" and request.gpu:
                continue
            requested = _requested_gpu_ids(job, config.gpu_ids) if request.gpu else ()
            requested_cpu_threads = max(1, int(request.cpu_threads))
            shared_cpu_capacity = cpu_pool_capacity(snapshot.cpu_threads_total)
            if requested_cpu_threads > shared_cpu_capacity:
                decision = AdmissionDecision(
                    False,
                    permanent=True,
                    reasons=(
                        f"requires {requested_cpu_threads} CPU threads; shared CPU "
                        f"pool capacity is {shared_cpu_capacity}",
                    ),
                )
            else:
                decision = assess_resource_admission(
                    request,
                    snapshot,
                    candidate_gpu_ids=requested,
                )
        except (TypeError, ValueError) as exc:
            claim = acquire_job_claim(
                job.run_dir,
                run_id=job.run_id,
                worker_id=config.worker_id,
                stale_after_seconds=config.stale_after_seconds,
            )
            if claim is not None:
                _record_admission(
                    job,
                    request=None,
                    snapshot=snapshot,
                    decision=AdmissionDecision(
                        False,
                        permanent=True,
                        reasons=(f"Invalid resource request: {exc}",),
                    ),
                    status="rejected",
                )
                claim.release()
            continue
        if not decision.allowed:
            claim = acquire_job_claim(
                job.run_dir,
                run_id=job.run_id,
                worker_id=config.worker_id,
                stale_after_seconds=config.stale_after_seconds,
            )
            if claim is not None:
                _record_admission(
                    job,
                    request=request,
                    snapshot=snapshot,
                    decision=decision,
                    status="rejected" if decision.permanent else "waiting",
                )
                claim.release()
            continue
        claim = acquire_job_claim(
            job.run_dir,
            run_id=job.run_id,
            worker_id=config.worker_id,
            stale_after_seconds=config.stale_after_seconds,
        )
        if claim is None:
            continue
        cpu_lease = acquire_cpu_lease(
            requested_cpu_threads,
            run_id=job.run_id,
            worker_id=config.worker_id,
            workflow=job.workflow,
            runs_dir=config.runs_dir,
            capacity=shared_cpu_capacity,
            stale_after_seconds=config.stale_after_seconds,
        )
        if cpu_lease is None:
            _record_admission(
                job,
                request=request,
                snapshot=snapshot,
                decision=AdmissionDecision(
                    False,
                    reasons=(
                        f"waiting for {requested_cpu_threads} free slot(s) in the "
                        f"shared {shared_cpu_capacity}-thread CPU pool",
                    ),
                ),
                status="waiting",
            )
            claim.release()
            continue
        if not request.gpu:
            _record_admission(
                job,
                request=request,
                snapshot=snapshot,
                decision=decision,
                status="admitted",
            )
            return job, claim, cpu_lease, None, None
        if (config.runs_dir / ".gpu_job.lock").exists():
            _record_admission(
                job,
                request=request,
                snapshot=snapshot,
                decision=AdmissionDecision(False, reasons=("legacy GPU lock is active",)),
                status="waiting",
            )
            cpu_lease.release()
            claim.release()
            continue
        selected = acquire_first_gpu_lease(
            decision.eligible_gpu_ids,
            run_id=job.run_id,
            worker_id=config.worker_id,
            workflow=job.workflow,
            runs_dir=config.runs_dir,
            stale_after_seconds=config.stale_after_seconds,
        )
        if selected is None:
            _record_admission(
                job,
                request=request,
                snapshot=snapshot,
                decision=AdmissionDecision(False, reasons=("eligible GPU lease is currently busy",)),
                status="waiting",
            )
            cpu_lease.release()
            claim.release()
            continue
        gpu_id, gpu_lease = selected
        _record_admission(
            job,
            request=request,
            snapshot=snapshot,
            decision=decision,
            status="admitted",
        )
        return job, claim, cpu_lease, gpu_id, gpu_lease
    return None


def _advance_workflows(metadata: dict[str, Any], metadata_path: Path) -> None:
    try:
        from mn_ligand.workflows.md_simulation import advance_md_workflows
        from mn_ligand.workflows.redocking import advance_redocking_benchmarks

        advance_md_workflows()
        advance_redocking_benchmarks()
    except Exception as exc:
        latest = _read_json(metadata_path) or metadata
        latest["orchestration_error"] = str(exc)
        _write_json(metadata_path, latest)


def _advance_pending_workflows(config: WorkerConfig) -> None:
    """Activate workflow children before scanning the queue.

    This lets a restarted worker resume an MD workflow without requiring a
    Streamlit page render to mutate workflow state.  All worker processes call
    this hook, so coordinate and throttle the global scan rather than making
    every idle GPU worker rescan every workflow on each polling cycle.
    """
    coordinator_dir = runs_root() / ".worker"
    coordinator_dir.mkdir(parents=True, exist_ok=True)
    lock_path = coordinator_dir / "workflow-orchestration.lock"
    state_path = coordinator_dir / "workflow-orchestration.json"

    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state = _read_json(state_path)
        try:
            last_scan = float(state.get("last_scan_epoch") or 0.0)
        except (TypeError, ValueError):
            last_scan = 0.0
        # A CPU worker is the normal coordinator.  GPU-only installations still
        # get a periodic recovery scan, but do not burn CPU while waiting for
        # GPU work that is unrelated to the queued analysis jobs.
        scan_interval = 60.0 if config.job_class == "gpu" else 5.0
        if time.time() - last_scan < scan_interval:
            return

        from mn_ligand.workflows.md_simulation import advance_md_workflows
        from mn_ligand.workflows.redocking import advance_redocking_benchmarks

        advance_md_workflows()
        advance_redocking_benchmarks()
        completed_at = time.time()
        _write_json(
            state_path,
            {
                "last_scan_epoch": completed_at,
                "updated_at": _utc_now_iso(),
                "worker_id": config.worker_id,
            },
        )


def _run_finalizer(run_dir: Path, metadata: dict[str, Any], returncode: int) -> None:
    finalizer = str(metadata.get("worker_finalizer") or "")
    if not finalizer:
        return
    if finalizer == "pocket_detection":
        from mn_ligand.workflows.pocket_detection import finalize_pocket_detection_job

        finalize_pocket_detection_job(run_dir, returncode=returncode)
        return
    if finalizer == "docking_campaign":
        from mn_ligand.workflows.docking import finalize_docking_campaign_job

        finalize_docking_campaign_job(run_dir, returncode=returncode)
        return
    if finalizer == "gnina_rescoring":
        from mn_ligand.workflows.rescoring import finalize_gnina_rescoring_job

        finalize_gnina_rescoring_job(run_dir, returncode=returncode)
        return
    if finalizer == "boltzina_rescoring":
        from mn_ligand.workflows.rescoring import finalize_boltzina_rescoring_job

        finalize_boltzina_rescoring_job(run_dir, returncode=returncode)
        return
    if finalizer == "openvs_docking":
        from mn_ligand.workflows.openvs import finalize_openvs_docking_job

        finalize_openvs_docking_job(run_dir, returncode=returncode)
        return
    if finalizer == "alphafold3_refolding":
        from mn_ligand.workflows.refolding import finalize_alphafold3_refolding_job

        finalize_alphafold3_refolding_job(run_dir, returncode=returncode)
        return
    if finalizer == "alphafold3_msa":
        from mn_ligand.workflows.refolding import finalize_alphafold3_msa_job

        finalize_alphafold3_msa_job(run_dir, returncode=returncode)
        return
    if finalizer == "boltz2_refolding":
        from mn_ligand.workflows.refolding import finalize_boltz2_refolding_job

        finalize_boltz2_refolding_job(run_dir, returncode=returncode)
        return
    if finalizer == "nesso_affinity":
        from mn_ligand.workflows.refolding import finalize_nesso_affinity_job

        finalize_nesso_affinity_job(run_dir, returncode=returncode)
        return
    if finalizer == "posebusters_validation":
        from mn_ligand.workflows.pose_validation import (
            finalize_pose_validation_job,
        )

        finalize_pose_validation_job(run_dir, returncode=returncode)
        return
    if finalizer == "pose_similarity":
        from mn_ligand.workflows.pose_similarity import finalize_pose_similarity_job

        finalize_pose_similarity_job(run_dir, returncode=returncode)
        return
    if finalizer == "interaction_analysis":
        from mn_ligand.workflows.interaction_analysis import (
            finalize_interaction_analysis_job,
        )

        finalize_interaction_analysis_job(run_dir, returncode=returncode)
        return
    if finalizer == "molecule_generation":
        from mn_ligand.workflows.generative_design import (
            finalize_generation_job,
        )

        finalize_generation_job(run_dir, returncode=returncode)
        return
    if finalizer == "molecule_qualification":
        from mn_ligand.workflows.molecule_qualification import (
            finalize_molecule_qualification_job,
        )

        finalize_molecule_qualification_job(
            run_dir,
            returncode=returncode,
        )
        return
    if finalizer == "md_job":
        from mn_ligand.workflows.md_simulation import finalize_md_worker_job

        finalize_md_worker_job(run_dir, returncode=returncode)
        return
    if finalizer == "md_mmgbsa":
        from mn_ligand.workflows.md_simulation import finalize_mmgbsa_analysis_job

        finalize_mmgbsa_analysis_job(run_dir, returncode=returncode)
        return
    raise ValueError(f"Unknown worker finalizer: {finalizer}")


def execute_job(
    job: QueuedJob,
    *,
    config: WorkerConfig,
    claim: FileLease,
    cpu_lease: CPULease,
    gpu_id: int | None,
    gpu_lease: FileLease | None,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    metadata_path = job.run_dir / "metadata.json"
    stdout_path = job.run_dir / "stdout.log"
    stderr_path = job.run_dir / "stderr.log"
    commands = [list(command) for command in job.commands]
    if job.metadata.get("msa_preparation_required"):
        try:
            from mn_ligand.workflows.refolding import prepare_msa_dependent_commands

            commands = prepare_msa_dependent_commands(
                job.run_dir, job.metadata, commands
            )
            if not commands:
                raise RuntimeError("MSA preparation removed every downstream command")
        except Exception as exc:
            if gpu_lease is not None:
                gpu_lease.release()
            cpu_lease.release()
            claim.release()
            now = _utc_now_iso()
            metadata = _read_json(metadata_path) or dict(job.metadata)
            metadata.update(
                {
                    "status": "failed",
                    "error": f"MSA dependency materialization failed: {exc}",
                    "updated_at": now,
                    "completed_at": now,
                }
            )
            _write_json(metadata_path, metadata)
            _record_worker_state(config, "idle")
            return {
                "run_id": job.run_id,
                "workflow": job.workflow,
                "status": "failed",
                "returncode": None,
                "gpu_id": gpu_id,
            }
    if gpu_id is not None:
        commands = [select_gpu_in_command(command, gpu_id) for command in commands]
    prepared = [
        _prepare_docker_command(command, job.run_dir, index)
        for index, command in enumerate(commands)
    ]
    commands = [command for command, _cidfile in prepared]
    container_cidfiles = [cidfile for _command, cidfile in prepared]
    metadata = _read_json(metadata_path) or dict(job.metadata)
    if str(metadata.get("status") or "") == "cancelled" or bool(
        metadata.get("cancellation_requested")
    ):
        if gpu_lease is not None:
            gpu_lease.release()
        cpu_lease.release()
        claim.release()
        now = _utc_now_iso()
        metadata.update(
            {
                "status": "cancelled",
                "cancelled_at": metadata.get("cancelled_at") or now,
                "completed_at": metadata.get("completed_at") or now,
                "updated_at": now,
            }
        )
        _write_json(metadata_path, metadata)
        _record_worker_state(config, "idle")
        return {
            "run_id": job.run_id,
            "workflow": job.workflow,
            "status": "cancelled",
            "returncode": None,
            "gpu_id": gpu_id,
        }
    now = _utc_now_iso()
    metadata.update(
        {
            "status": "running",
            "worker_id": config.worker_id,
            "worker_pid": os.getpid(),
            "worker_heartbeat_at": now,
            "started_at": metadata.get("started_at") or now,
            "selected_gpu": gpu_id,
            "reserved_cpu_threads": cpu_lease.threads,
            "reserved_cpu_slots": list(cpu_lease.slot_ids),
            "cpu_pool_capacity": cpu_lease.capacity,
            "executed_command": commands[0],
            "executed_commands": commands,
        }
    )
    try:
        max_runtime_seconds = max(
            0.0, float(metadata.get("max_runtime_seconds") or 0)
        )
    except (TypeError, ValueError):
        max_runtime_seconds = 0.0
    execution_started_monotonic = time.monotonic()
    if max_runtime_seconds:
        metadata["runtime_budget_started_at"] = now
        metadata["runtime_deadline_at"] = (
            datetime.now(timezone.utc)
            + timedelta(seconds=max_runtime_seconds)
        ).isoformat()
    _write_json(metadata_path, metadata)
    _record_worker_state(config, "running", run_id=job.run_id, selected_gpu=gpu_id)
    returncode: int | None = None
    cancelled = False
    timed_out = False
    interrupted = False
    error = ""
    process: subprocess.Popen[Any] | None = None
    active_command: list[str] | None = None
    active_cidfile: Path | None = None
    try:
        with stdout_path.open("a") as stdout_handle, stderr_path.open("a") as stderr_handle:
            for command_index, command in enumerate(commands):
                if (
                    max_runtime_seconds
                    and time.monotonic() - execution_started_monotonic
                    >= max_runtime_seconds
                ):
                    timed_out = True
                    returncode = -9
                    error = (
                        "Maximum runtime budget exceeded before the next "
                        "command could start."
                    )
                    break
                active_command = command
                active_cidfile = container_cidfiles[command_index]
                metadata = _read_json(metadata_path) or metadata
                metadata["command_index"] = command_index
                metadata["command_count"] = len(commands)
                _write_json(metadata_path, metadata)
                try:
                    process = popen(
                        command,
                        cwd=str(job.run_dir),
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        text=True,
                        start_new_session=True,
                    )
                except OSError as exc:
                    error = str(exc)
                    stderr_handle.write(error + "\n")
                    returncode = -1
                    break
                while process.poll() is None:
                    sleep(config.heartbeat_seconds)
                    claim.heartbeat()
                    cpu_lease.heartbeat()
                    if gpu_lease is not None:
                        gpu_lease.heartbeat()
                    latest = _read_json(metadata_path)
                    if str(latest.get("status") or "") == "cancelled" or bool(
                        latest.get("cancellation_requested")
                    ):
                        cancelled = True
                        # Native CPU jobs can create process pools.  They run
                        # in their own session so cancellation never leaves
                        # expensive orphaned children behind.
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=config.cancellation_grace_seconds)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        break
                    if (
                        max_runtime_seconds
                        and time.monotonic() - execution_started_monotonic
                        >= max_runtime_seconds
                    ):
                        timed_out = True
                        error = (
                            "Maximum runtime budget of "
                            f"{max_runtime_seconds:g} seconds exceeded."
                        )
                        # Docker inference must not remain detached after the
                        # client is terminated. Remove the recorded container
                        # first; this stops its GPU workload immediately.
                        cleanup_error = _cleanup_docker_container(
                            command, active_cidfile, force=True
                        )
                        active_cidfile = None
                        if cleanup_error:
                            error = f"{error} {cleanup_error}".strip()
                        if process.poll() is None:
                            process.terminate()
                        try:
                            process.wait(
                                timeout=config.cancellation_grace_seconds
                            )
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        break
                    latest["worker_heartbeat_at"] = _utc_now_iso()
                    _write_json(metadata_path, latest)
                    _record_worker_state(
                        config, "running", run_id=job.run_id, selected_gpu=gpu_id
                    )
                returncode = int(process.wait())
                process = None
                cleanup_error = _cleanup_docker_container(
                    command,
                    active_cidfile,
                    force=cancelled or timed_out or returncode != 0,
                )
                if cleanup_error:
                    error = f"{error} {cleanup_error}".strip()
                active_command = None
                active_cidfile = None
                if cancelled or timed_out or returncode != 0:
                    break
    except KeyboardInterrupt:
        interrupted = True
        error = "Worker service stopped during execution; retry this preserved run if appropriate."
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=config.cancellation_grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process is not None:
            returncode = int(process.wait())
        elif returncode is None:
            returncode = -2
        if active_command is not None:
            cleanup_error = _cleanup_docker_container(active_command, active_cidfile, force=True)
            if cleanup_error:
                error = f"{error} {cleanup_error}".strip()
    finally:
        if gpu_lease is not None:
            gpu_lease.release()
        cpu_lease.release()
        claim.release()

    metadata = _read_json(metadata_path) or metadata
    finalizer_error = ""
    if not cancelled and not interrupted:
        try:
            _run_finalizer(job.run_dir, metadata, int(returncode if returncode is not None else -1))
        except Exception as exc:
            finalizer_error = f"Worker finalizer failed: {exc}"
            result_payload = {"success": False, "error": finalizer_error, "returncode": returncode}
            _write_json(job.run_dir / "result.json", result_payload)
            metadata = _read_json(metadata_path) or metadata
            metadata["finalizer_error"] = finalizer_error
            _write_json(metadata_path, metadata)
    metadata = _read_json(metadata_path) or metadata
    result = _read_json(job.run_dir / "result.json")
    if timed_out:
        result.update(
            {
                "success": False,
                "timed_out": True,
                "termination_reason": "runtime_budget_exceeded",
                "max_runtime_seconds": max_runtime_seconds,
                "error": error,
            }
        )
        _write_json(job.run_dir / "result.json", result)
    native_success = result.get("success") if result else None
    if cancelled:
        status = "cancelled"
    elif interrupted:
        status = "failed"
    elif native_success is True or (returncode == 0 and native_success is not False):
        status = "completed"
    else:
        status = "failed"
    completed_at = _utc_now_iso()
    runtime_seconds = max(
        0.0, time.monotonic() - execution_started_monotonic
    )
    metadata.update(
        {
            "status": status,
            "returncode": returncode,
            "completed_at": completed_at,
            "updated_at": completed_at,
            "worker_heartbeat_at": completed_at,
            "runtime_seconds": runtime_seconds,
            "stdout_tail": _tail(stdout_path),
            "stderr_tail": _tail(stderr_path),
        }
    )
    if cancelled:
        metadata["cancelled_at"] = metadata.get("cancelled_at") or completed_at
    if timed_out:
        metadata.update(
            {
                "timed_out": True,
                "timed_out_at": completed_at,
                "termination_reason": "runtime_budget_exceeded",
            }
        )
    if interrupted:
        metadata["interrupted_at"] = completed_at
    if error or finalizer_error:
        metadata["error"] = finalizer_error or error
    _write_json(metadata_path, metadata)
    if status == "completed" and metadata.get("requested_repetitions"):
        try:
            from mn_ligand.core.campaign_extensions import apply_pending_extension

            apply_pending_extension(job.run_dir)
            metadata = _read_json(metadata_path) or metadata
            status = str(metadata.get("status") or status)
        except Exception as exc:
            metadata = _read_json(metadata_path) or metadata
            metadata["extension_error"] = str(exc)
            _write_json(metadata_path, metadata)
    _advance_workflows(metadata, metadata_path)
    _record_worker_state(
        config,
        "stopping" if interrupted else "idle",
        run_id=job.run_id if interrupted else "",
        selected_gpu=gpu_id if interrupted else None,
    )
    worker_result = {
        "run_id": job.run_id,
        "workflow": job.workflow,
        "status": status,
        "returncode": returncode,
        "gpu_id": gpu_id,
        "cpu_threads": cpu_lease.threads,
        "runtime_seconds": runtime_seconds,
    }
    if interrupted:
        raise KeyboardInterrupt
    return worker_result


def run_worker_once(
    config: WorkerConfig,
    *,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any] | None:
    _advance_pending_workflows(config)
    claimed = _claim_runnable_job(config)
    if claimed is None:
        return None
    job, claim, cpu_lease, gpu_id, gpu_lease = claimed
    return execute_job(
        job,
        config=config,
        claim=claim,
        cpu_lease=cpu_lease,
        gpu_id=gpu_id,
        gpu_lease=gpu_lease,
        popen=popen,
        sleep=sleep,
    )


def run_worker_until_idle(config: WorkerConfig, *, max_jobs: int | None = None) -> dict[str, Any] | None:
    completed: list[dict[str, Any]] = []
    while max_jobs is None or len(completed) < max_jobs:
        result = run_worker_once(config)
        if result is None:
            break
        completed.append(result)
    if not completed:
        return None
    return {"count": len(completed), "last": completed[-1], "runs": completed}


def serve_worker(config: WorkerConfig, *, poll_seconds: float = 2.0) -> None:
    _record_worker_state(config, "idle")
    try:
        while True:
            result = run_worker_once(config)
            if result is None:
                _record_worker_state(config, "idle")
                time.sleep(max(0.1, poll_seconds))
    finally:
        _record_worker_state(config, "stopped")
