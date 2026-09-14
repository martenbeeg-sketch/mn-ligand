from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from mn_ligand.core.manifests import ToolManifest, load_tool_registry


_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def docker_is_rootless() -> bool:
    docker_host = os.getenv("DOCKER_HOST", "").lower()
    rootless_socket = Path(f"/run/user/{os.getuid()}/docker.sock")
    return "rootless" in docker_host or "/run/user/" in docker_host or rootless_socket.exists()


@dataclass(frozen=True)
class DockerMount:
    host_path: Path
    container_path: str
    read_only: bool = False

    def __post_init__(self) -> None:
        host = self.host_path.expanduser().resolve()
        container = PurePosixPath(self.container_path)
        if not container.is_absolute() or ".." in container.parts:
            raise ValueError(f"Container mount path must be absolute and traversal-free: {self.container_path}")
        object.__setattr__(self, "host_path", host)
        object.__setattr__(self, "container_path", container.as_posix())

    def argument(self) -> str:
        suffix = ":ro" if self.read_only else ""
        return f"{self.host_path}:{self.container_path}{suffix}"


@dataclass(frozen=True)
class DockerRunSpec:
    tool: ToolManifest
    command: tuple[str, ...]
    mounts: tuple[DockerMount, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    gpu_enabled: bool | None = None
    gpu_devices: tuple[int, ...] | None = None
    workdir: str = ""
    shm_size: str = ""
    use_host_user: bool = True

    def __post_init__(self) -> None:
        if not self.command or any(not str(value) for value in self.command):
            raise ValueError("Native command cannot be empty")
        destinations = [mount.container_path for mount in self.mounts]
        if len(destinations) != len(set(destinations)):
            raise ValueError("Docker mounts must use unique container paths")
        for key in self.environment:
            if not _ENVIRONMENT_KEY.fullmatch(str(key)):
                raise ValueError(f"Invalid environment variable name: {key!r}")
        if self.workdir:
            workdir = PurePosixPath(self.workdir)
            if not workdir.is_absolute() or ".." in workdir.parts:
                raise ValueError("Docker workdir must be absolute and traversal-free")
        if not self.requests_gpu and self.gpu_devices is not None:
            raise ValueError(f"CPU tool {self.tool.tool_id!r} cannot request GPU devices")
        if self.gpu_devices is not None:
            if not self.gpu_devices or any(device < 0 for device in self.gpu_devices):
                raise ValueError("GPU devices must be a non-empty tuple of non-negative IDs")
            if len(self.gpu_devices) != len(set(self.gpu_devices)):
                raise ValueError("GPU device IDs must be unique")

    @property
    def requests_gpu(self) -> bool:
        return self.tool.resources.gpu if self.gpu_enabled is None else bool(self.gpu_enabled)


def build_docker_command(spec: DockerRunSpec, *, rootless: bool | None = None) -> list[str]:
    command = ["docker", "run", "--rm"]
    if spec.requests_gpu:
        request = (
            "all"
            if spec.gpu_devices is None
            else f"device={','.join(str(device) for device in spec.gpu_devices)}"
        )
        command.extend(["--gpus", request])
    if spec.shm_size:
        command.extend(["--shm-size", spec.shm_size])
    is_rootless = docker_is_rootless() if rootless is None else rootless
    if spec.use_host_user and not is_rootless:
        command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    for mount in spec.mounts:
        command.extend(["-v", mount.argument()])
    for key, value in sorted(spec.environment.items()):
        command.extend(["-e", f"{key}={value}"])
    if spec.workdir:
        command.extend(["-w", spec.workdir])
    command.append(spec.tool.image)
    command.extend(str(value) for value in spec.command)
    return command


def registered_tool(tool_id: str, *, image: str = "") -> ToolManifest:
    tool = load_tool_registry().get(tool_id)
    override = str(image).strip()
    return replace(tool, image=override) if override else tool


def write_command_record(
    run_dir: Path,
    spec: DockerRunSpec,
    command: Sequence[str],
    *,
    selected_gpu_ids: Sequence[int] = (),
) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool_id": spec.tool.tool_id,
        "tool_version": spec.tool.version,
        "image": spec.tool.image,
        "image_digest": spec.tool.image_digest,
        "resources": spec.tool.resources.to_dict(),
        "selected_gpu_ids": [int(value) for value in selected_gpu_ids],
        "argv": [str(value) for value in command],
    }
    target = run_dir / "command.json"
    temporary = run_dir / ".command.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    return target


def write_registered_command_record(
    run_dir: Path,
    *,
    tool_id: str,
    commands: Sequence[Sequence[str]],
    image: str = "",
    selected_gpu_ids: Sequence[int] = (),
    filename: str = "command.json",
) -> Path:
    normalized = [[str(value) for value in command] for command in commands]
    if not normalized or any(not command for command in normalized):
        raise ValueError("At least one non-empty command is required")
    tool = registered_tool(tool_id, image=image)
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("Command record filename must be a run-local JSON filename")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool_id": tool.tool_id,
        "tool_version": tool.version,
        "integration_status": tool.integration_status,
        "image": tool.image,
        "image_digest": tool.image_digest,
        "resources": tool.resources.to_dict(),
        "selected_gpu_ids": [int(value) for value in selected_gpu_ids],
        "argv": normalized[0],
        "commands": normalized,
    }
    target = run_dir / filename
    temporary = run_dir / f".{filename}.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    return target
