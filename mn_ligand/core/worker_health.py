from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

from mn_ligand.core.resources import cpu_pool_capacity
from mn_ligand.runtime import runs_root


HealthRunner = Callable[..., subprocess.CompletedProcess[str]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _worker_filename(worker_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", worker_id).strip("._") or "worker"


def worker_heartbeat_path(runs_dir: Path, worker_id: str) -> Path:
    return runs_dir / ".worker" / "workers" / f"{_worker_filename(worker_id)}.json"


def record_worker_heartbeat(
    runs_dir: Path,
    *,
    worker_id: str,
    gpu_ids: Sequence[int],
    job_class: str = "mixed",
    state: str,
    run_id: str = "",
    selected_gpu: int | None = None,
) -> Path:
    path = worker_heartbeat_path(runs_dir, worker_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "worker_id": worker_id,
        "pid": os.getpid(),
        "gpu_ids": [int(value) for value in gpu_ids],
        "job_class": str(job_class),
        "state": state,
        "run_id": run_id,
        "selected_gpu": selected_gpu,
        "heartbeat_at": _utc_now_iso(),
    }
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
    return path


def _age_seconds(value: object) -> float | None:
    try:
        timestamp = datetime.fromisoformat(str(value))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())


def _systemd_properties(
    unit: str, *, runner: HealthRunner = subprocess.run
) -> tuple[dict[str, str], str]:
    command = [
        "systemctl",
        "--user",
        "show",
        unit,
        "--no-pager",
        "--property=LoadState,ActiveState,SubState,MainPID,UnitFileState",
    ]
    try:
        completed = runner(command, capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, str(exc)
    properties = {}
    for line in str(completed.stdout or "").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    if completed.returncode == 0:
        return properties, ""
    return properties, str(completed.stderr or completed.stdout or "systemctl unavailable").strip()


@dataclass(frozen=True)
class WorkerServiceHealth:
    gpu_id: int
    unit: str
    enabled: bool
    active: bool
    service_state: str
    pid: int | None
    worker_id: str
    worker_state: str
    current_run_id: str
    heartbeat_at: str
    heartbeat_age_seconds: float | None
    heartbeat_stale: bool
    lease_run_id: str
    worker_kind: str = "gpu"
    error: str = ""

    def to_row(self) -> dict[str, object]:
        heartbeat_age = self.heartbeat_age_seconds
        return {
            "resource": f"GPU {self.gpu_id}" if self.worker_kind == "gpu" else "CPU",
            "service": self.service_state,
            "enabled": self.enabled,
            "PID": self.pid,
            "worker": self.worker_id,
            "worker_state": self.worker_state,
            "current_run": self.current_run_id or self.lease_run_id,
            "heartbeat_age_s": round(heartbeat_age, 1) if heartbeat_age is not None else None,
            "heartbeat_stale": self.heartbeat_stale,
            "message": self.error,
        }


@dataclass(frozen=True)
class WorkerHealthSnapshot:
    services: tuple[WorkerServiceHealth, ...]
    queued_jobs: int
    active_leases: int
    cpu_services: tuple[WorkerServiceHealth, ...] = ()
    cpu_pool_capacity: int = 0
    cpu_slots_leased: int = 0


def inspect_worker_health(
    gpu_ids: Sequence[int] = (0, 1),
    *,
    run_dir: Path | None = None,
    runner: HealthRunner = subprocess.run,
    stale_after_seconds: float = 15.0,
) -> WorkerHealthSnapshot:
    from mn_ligand.core.worker import iter_queued_jobs

    root = (run_dir or runs_root()).resolve()
    queued_jobs = len(iter_queued_jobs(root))
    cpu_capacity = cpu_pool_capacity()
    cpu_slots_leased = len(
        tuple((root / ".worker" / "locks").glob("cpu-[0-9]*.json"))
    )
    services: list[WorkerServiceHealth] = []
    active_leases = 0
    for gpu_id_value in gpu_ids:
        gpu_id = int(gpu_id_value)
        unit = f"mn-ligand-worker@{gpu_id}.service"
        properties, error = _systemd_properties(unit, runner=runner)
        worker_id = f"mn-ligand-gpu-{gpu_id}"
        heartbeat = _read_json(worker_heartbeat_path(root, worker_id))
        lease = _read_json(root / ".worker" / "locks" / f"gpu-{gpu_id}.json")
        if lease:
            active_leases += 1
        heartbeat_age = _age_seconds(heartbeat.get("heartbeat_at"))
        active = properties.get("ActiveState") == "active"
        substate = properties.get("SubState") or "unavailable"
        service_state = (
            f"{properties.get('ActiveState')}/{substate}"
            if properties.get("ActiveState")
            else substate
        )
        raw_pid = str(properties.get("MainPID") or "")
        services.append(
            WorkerServiceHealth(
                gpu_id=gpu_id,
                unit=unit,
                enabled=properties.get("UnitFileState") in {"enabled", "enabled-runtime"},
                active=active,
                service_state=service_state,
                pid=int(raw_pid) if raw_pid.isdigit() and int(raw_pid) > 0 else None,
                worker_id=str(heartbeat.get("worker_id") or worker_id),
                worker_state=str(heartbeat.get("state") or ("starting" if active else "unknown")),
                current_run_id=str(heartbeat.get("run_id") or ""),
                heartbeat_at=str(heartbeat.get("heartbeat_at") or ""),
                heartbeat_age_seconds=heartbeat_age,
                heartbeat_stale=heartbeat_age is None or heartbeat_age > stale_after_seconds,
                lease_run_id=str(lease.get("run_id") or ""),
                worker_kind="gpu",
                error=error,
            )
        )
    cpu_unit = "mn-ligand-cpu-worker.service"
    cpu_properties, cpu_error = _systemd_properties(cpu_unit, runner=runner)
    cpu_worker_id = "mn-ligand-cpu-0"
    cpu_heartbeat = _read_json(worker_heartbeat_path(root, cpu_worker_id))
    cpu_heartbeat_age = _age_seconds(cpu_heartbeat.get("heartbeat_at"))
    cpu_active = cpu_properties.get("ActiveState") == "active"
    cpu_substate = cpu_properties.get("SubState") or "unavailable"
    cpu_service_state = (
        f"{cpu_properties.get('ActiveState')}/{cpu_substate}"
        if cpu_properties.get("ActiveState")
        else cpu_substate
    )
    cpu_raw_pid = str(cpu_properties.get("MainPID") or "")
    cpu_service = WorkerServiceHealth(
        gpu_id=-1,
        unit=cpu_unit,
        enabled=cpu_properties.get("UnitFileState") in {"enabled", "enabled-runtime"},
        active=cpu_active,
        service_state=cpu_service_state,
        pid=(
            int(cpu_raw_pid)
            if cpu_raw_pid.isdigit() and int(cpu_raw_pid) > 0
            else None
        ),
        worker_id=str(cpu_heartbeat.get("worker_id") or cpu_worker_id),
        worker_state=str(
            cpu_heartbeat.get("state") or ("starting" if cpu_active else "unknown")
        ),
        current_run_id=str(cpu_heartbeat.get("run_id") or ""),
        heartbeat_at=str(cpu_heartbeat.get("heartbeat_at") or ""),
        heartbeat_age_seconds=cpu_heartbeat_age,
        heartbeat_stale=(
            cpu_heartbeat_age is None or cpu_heartbeat_age > stale_after_seconds
        ),
        lease_run_id="",
        worker_kind="cpu",
        error=cpu_error,
    )
    return WorkerHealthSnapshot(
        services=tuple(services),
        queued_jobs=queued_jobs,
        active_leases=active_leases,
        cpu_services=(cpu_service,),
        cpu_pool_capacity=cpu_capacity,
        cpu_slots_leased=cpu_slots_leased,
    )
