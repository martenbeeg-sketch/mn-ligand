from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

from mn_ligand.runtime import PROJECT_DIR, app_home, temporary_root


SERVICE_TEMPLATE_NAME = "mn-ligand-worker@.service"
CPU_SERVICE_NAME = "mn-ligand-cpu-worker.service"
DEFAULT_GPU_IDS = (0, 1)


class WorkerServiceError(RuntimeError):
    """Raised when a worker service management command fails."""


@dataclass(frozen=True)
class ServiceCommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


ServiceRunner = Callable[..., subprocess.CompletedProcess[str]]


def parse_gpu_ids(value: str, *, default: Sequence[int] = DEFAULT_GPU_IDS) -> tuple[int, ...]:
    text = value.strip()
    if not text:
        return tuple(default)
    parts = tuple(part.strip() for part in text.split(",") if part.strip())
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError("GPU IDs must be comma-separated non-negative integers")
    return tuple(dict.fromkeys(int(part) for part in parts))


def service_instance_name(gpu_id: int) -> str:
    if gpu_id < 0:
        raise ValueError("GPU IDs cannot be negative")
    return f"mn-ligand-worker@{gpu_id}.service"


def user_unit_dir() -> Path:
    configured = os.getenv("XDG_CONFIG_HOME")
    config_home = Path(configured).expanduser() if configured else Path.home() / ".config"
    return config_home / "systemd" / "user"


def _systemd_quote(value: str | Path) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_worker_service(
    *,
    python_executable: str | Path = sys.executable,
    project_dir: str | Path = PROJECT_DIR,
    runtime_home: str | Path | None = None,
    tmp_dir: str | Path | None = None,
) -> str:
    resolved_python = Path(python_executable).expanduser().absolute()
    resolved_project = Path(project_dir).expanduser().resolve()
    resolved_home = Path(runtime_home or app_home()).expanduser().resolve()
    resolved_tmp = Path(tmp_dir or temporary_root()).expanduser().resolve()
    path_value = ":".join(
        dict.fromkeys(
            (
                str(resolved_python.parent),
                "/usr/local/sbin",
                "/usr/local/bin",
                "/usr/sbin",
                "/usr/bin",
                "/sbin",
                "/bin",
            )
        )
    )
    exec_start = " ".join(
        (
            _systemd_quote(resolved_python),
            "-m",
            "mn_ligand.cli",
            "worker",
            "--gpu-ids",
            "%i",
            "--worker-id",
            "mn-ligand-gpu-%i",
            "--job-class",
            "gpu",
        )
    )
    return "\n".join(
        (
            "[Unit]",
            "Description=mn-ligand durable worker for GPU %i",
            "Documentation=file:" + str(resolved_project / "README.md"),
            "Requires=docker.service",
            "After=docker.service",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={resolved_project}",
            f"Environment={_systemd_quote(f'MN_LIGAND_APP_HOME={resolved_home}')}",
            f"Environment={_systemd_quote(f'MN_LIGAND_TMP_DIR={resolved_tmp}')}",
            f"Environment={_systemd_quote(f'TMPDIR={resolved_tmp}')}",
            f"Environment={_systemd_quote(f'PATH={path_value}')}",
            'Environment="PYTHONUNBUFFERED=1"',
            f"ExecStart={exec_start}",
            "Restart=on-failure",
            "RestartSec=5s",
            "TimeoutStopSec=30s",
            "KillMode=mixed",
            "KillSignal=SIGINT",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        )
    )


def render_cpu_worker_service(
    *,
    python_executable: str | Path = sys.executable,
    project_dir: str | Path = PROJECT_DIR,
    runtime_home: str | Path | None = None,
    tmp_dir: str | Path | None = None,
) -> str:
    resolved_python = Path(python_executable).expanduser().absolute()
    resolved_project = Path(project_dir).expanduser().resolve()
    resolved_home = Path(runtime_home or app_home()).expanduser().resolve()
    resolved_tmp = Path(tmp_dir or temporary_root()).expanduser().resolve()
    path_value = ":".join(
        dict.fromkeys(
            (
                str(resolved_python.parent),
                "/usr/local/sbin",
                "/usr/local/bin",
                "/usr/sbin",
                "/usr/bin",
                "/sbin",
                "/bin",
            )
        )
    )
    exec_start = " ".join(
        (
            _systemd_quote(resolved_python),
            "-m",
            "mn_ligand.cli",
            "worker",
            "--worker-id",
            "mn-ligand-cpu-0",
            "--job-class",
            "cpu",
        )
    )
    return "\n".join(
        (
            "[Unit]",
            "Description=mn-ligand durable CPU worker",
            "Documentation=file:" + str(resolved_project / "README.md"),
            "Requires=docker.service",
            "After=docker.service",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={resolved_project}",
            f"Environment={_systemd_quote(f'MN_LIGAND_APP_HOME={resolved_home}')}",
            f"Environment={_systemd_quote(f'MN_LIGAND_TMP_DIR={resolved_tmp}')}",
            f"Environment={_systemd_quote(f'TMPDIR={resolved_tmp}')}",
            f"Environment={_systemd_quote(f'PATH={path_value}')}",
            'Environment="PYTHONUNBUFFERED=1"',
            f"ExecStart={exec_start}",
            "Restart=on-failure",
            "RestartSec=5s",
            "TimeoutStopSec=30s",
            "KillMode=mixed",
            "KillSignal=SIGINT",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        )
    )


def _run(
    command: Sequence[str],
    *,
    runner: ServiceRunner = subprocess.run,
    check: bool = True,
) -> ServiceCommandResult:
    completed = runner(list(command), capture_output=True, text=True, check=False)
    result = ServiceCommandResult(
        command=tuple(str(value) for value in command),
        returncode=int(completed.returncode),
        stdout=str(completed.stdout or ""),
        stderr=str(completed.stderr or ""),
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown systemd error"
        raise WorkerServiceError(f"{' '.join(result.command)} failed: {detail}")
    return result


def install_worker_service(
    gpu_ids: Sequence[int] = DEFAULT_GPU_IDS,
    *,
    start: bool = True,
    unit_dir: Path | None = None,
    python_executable: str | Path = sys.executable,
    project_dir: str | Path = PROJECT_DIR,
    runtime_home: str | Path | None = None,
    tmp_dir: str | Path | None = None,
    runner: ServiceRunner = subprocess.run,
) -> Path:
    selected = tuple(dict.fromkeys(int(value) for value in gpu_ids))
    if not selected or any(value < 0 for value in selected):
        raise ValueError("At least one non-negative GPU ID is required")
    destination_dir = unit_dir or user_unit_dir()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / SERVICE_TEMPLATE_NAME
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        render_worker_service(
            python_executable=python_executable,
            project_dir=project_dir,
            runtime_home=runtime_home,
            tmp_dir=tmp_dir,
        )
    )
    temporary.replace(destination)
    cpu_destination = destination_dir / CPU_SERVICE_NAME
    cpu_temporary = cpu_destination.with_name(
        f".{cpu_destination.name}.{uuid4().hex}.tmp"
    )
    cpu_temporary.write_text(
        render_cpu_worker_service(
            python_executable=python_executable,
            project_dir=project_dir,
            runtime_home=runtime_home,
            tmp_dir=tmp_dir,
        )
    )
    cpu_temporary.replace(cpu_destination)
    _run(("systemctl", "--user", "daemon-reload"), runner=runner)
    instances = tuple(service_instance_name(value) for value in selected)
    action = ("enable", "--now") if start else ("enable",)
    _run(
        ("systemctl", "--user", *action, *instances, CPU_SERVICE_NAME),
        runner=runner,
    )
    return destination


def manage_worker_services(
    action: str,
    gpu_ids: Sequence[int] = DEFAULT_GPU_IDS,
    *,
    runner: ServiceRunner = subprocess.run,
    check: bool = True,
) -> ServiceCommandResult:
    if action not in {"start", "stop", "restart", "status"}:
        raise ValueError(f"Unsupported worker service action: {action}")
    instances = tuple(service_instance_name(int(value)) for value in gpu_ids)
    if not instances:
        raise ValueError("At least one GPU ID is required")
    arguments = ("--no-pager", "--full") if action == "status" else ()
    return _run(
        ("systemctl", "--user", action, *arguments, *instances, CPU_SERVICE_NAME),
        runner=runner,
        check=check,
    )


def uninstall_worker_service(
    gpu_ids: Sequence[int] = DEFAULT_GPU_IDS,
    *,
    unit_dir: Path | None = None,
    runner: ServiceRunner = subprocess.run,
) -> Path:
    instances = tuple(service_instance_name(int(value)) for value in gpu_ids)
    _run(
        ("systemctl", "--user", "disable", "--now", *instances, CPU_SERVICE_NAME),
        runner=runner,
        check=False,
    )
    destination = (unit_dir or user_unit_dir()) / SERVICE_TEMPLATE_NAME
    destination.unlink(missing_ok=True)
    ((unit_dir or user_unit_dir()) / CPU_SERVICE_NAME).unlink(missing_ok=True)
    _run(("systemctl", "--user", "daemon-reload"), runner=runner)
    return destination
