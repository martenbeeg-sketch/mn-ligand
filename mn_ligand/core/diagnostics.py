from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from mn_ligand.core.manifests import ToolRegistry, load_tool_registry
from mn_ligand.runtime import input_root, library_root, reference_root, runs_root, temporary_root


DiagnosticRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class DiagnosticResult:
    check_id: str
    label: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"pass", "warn", "fail"}:
            raise ValueError(f"Unsupported diagnostic status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "label": self.label,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def _run_command(
    command: Sequence[str], runner: DiagnosticRunner, *, timeout: int = 15
) -> subprocess.CompletedProcess[str]:
    return runner(list(command), capture_output=True, text=True, check=False, timeout=timeout)


def _path_result(check_id: str, label: str, path: Path, *, required: bool) -> DiagnosticResult:
    if not path.is_dir():
        return DiagnosticResult(
            check_id, label, "fail" if required else "warn", f"Directory is missing: {path}", {"path": str(path)}
        )
    readable = os.access(path, os.R_OK)
    writable = os.access(path, os.W_OK)
    status = "pass" if readable and writable else ("fail" if required else "warn")
    return DiagnosticResult(
        check_id,
        label,
        status,
        "Directory is readable and writable" if status == "pass" else "Directory permissions are incomplete",
        {"path": str(path), "readable": readable, "writable": writable},
    )


def run_diagnostics(
    *,
    registry: ToolRegistry | None = None,
    include_images: bool = True,
    runner: DiagnosticRunner = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> list[DiagnosticResult]:
    registry = registry or load_tool_registry()
    results = [
        DiagnosticResult("registry", "Tool registry", "pass", f"Loaded {len(registry.tools)} tool manifests"),
        _path_result("runs", "Results and jobs", runs_root(create=False), required=True),
        _path_result("references", "Reference files", reference_root(create=False), required=False),
        _path_result("libraries", "Compound libraries", library_root(create=False), required=False),
        _path_result("inputs", "Imported inputs", input_root(create=False), required=False),
        _path_result("temporary", "Temporary files", temporary_root(create=False), required=True),
    ]

    docker_path = which("docker")
    docker_available = False
    if not docker_path:
        results.append(DiagnosticResult("docker", "Docker Engine", "fail", "docker executable was not found"))
    else:
        try:
            process = _run_command([docker_path, "version", "--format", "{{.Server.Version}}"], runner)
        except (OSError, subprocess.TimeoutExpired) as exc:
            results.append(DiagnosticResult("docker", "Docker Engine", "fail", f"Docker check failed: {exc}"))
        else:
            docker_available = process.returncode == 0
            message = (process.stdout if docker_available else process.stderr).strip() or "No version output"
            results.append(
                DiagnosticResult("docker", "Docker Engine", "pass" if docker_available else "fail", message)
            )

    gpu_tools = tuple(tool for tool in registry.tools if tool.resources.gpu)
    nvidia_path = which("nvidia-smi")
    if gpu_tools and not nvidia_path:
        results.append(DiagnosticResult("nvidia", "NVIDIA GPUs", "fail", "nvidia-smi was not found"))
    elif gpu_tools and nvidia_path:
        query = [
            nvidia_path,
            "--query-gpu=index,name,memory.total,compute_cap,driver_version",
            "--format=csv,noheader",
        ]
        try:
            process = _run_command(query, runner)
        except (OSError, subprocess.TimeoutExpired) as exc:
            results.append(DiagnosticResult("nvidia", "NVIDIA GPUs", "fail", f"GPU check failed: {exc}"))
        else:
            output = process.stdout.strip()
            results.append(
                DiagnosticResult(
                    "nvidia",
                    "NVIDIA GPUs",
                    "pass" if process.returncode == 0 and output else "fail",
                    output or process.stderr.strip() or "No GPUs reported",
                )
            )

    reference_base = reference_root(create=False)
    image_checks: dict[str, tuple[bool, str]] = {}
    for tool in registry.tools:
        readiness_status = (
            "pass" if tool.integration_status in {"validated", "implemented"} else "warn"
        )
        results.append(
            DiagnosticResult(
                f"integration:{tool.tool_id}",
                f"{tool.name} integration",
                readiness_status,
                (
                    f"{tool.integration_status}: "
                    f"{tool.status_notes or 'No status notes recorded'}"
                    + (
                        f" Evidence records: {len(tool.validation_evidence)}."
                        if tool.validation_evidence
                        else ""
                    )
                ),
                {
                    "validation_evidence_count": len(tool.validation_evidence),
                    "validation_job_ids": [
                        job_id
                        for evidence in tool.validation_evidence
                        for job_id in evidence.job_ids
                    ],
                },
            )
        )
        if not tool.image_digest:
            results.append(
                DiagnosticResult(
                    f"digest:{tool.tool_id}", tool.name, "warn", f"Image digest is not pinned for {tool.image}"
                )
            )
        else:
            results.append(
                DiagnosticResult(
                    f"digest:{tool.tool_id}",
                    tool.name,
                    "pass",
                    f"Image digest recorded: {tool.image_digest}",
                    {"image": tool.image, "image_digest": tool.image_digest},
                )
            )
        for requirement in tool.references:
            candidate = reference_base / requirement.path
            exists = candidate.exists()
            results.append(
                DiagnosticResult(
                    f"reference:{tool.tool_id}:{requirement.path}",
                    f"{tool.name} reference",
                    "pass" if exists else ("fail" if requirement.required else "warn"),
                    f"Available: {requirement.path}" if exists else f"Missing: {requirement.path}",
                    {"path": str(candidate)},
                )
            )
        if include_images:
            if not docker_available:
                results.append(
                    DiagnosticResult(
                        f"image:{tool.tool_id}", tool.name, "warn", "Image check skipped because Docker is unavailable"
                    )
                )
                continue
            cached = image_checks.get(tool.image)
            if cached is None:
                try:
                    process = _run_command(tool.resolved_healthcheck(), runner)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    cached = (False, f"Image check failed: {exc}")
                else:
                    cached = (
                        process.returncode == 0,
                        f"Image available: {tool.image}"
                        if process.returncode == 0
                        else (process.stderr.strip() or f"Image unavailable: {tool.image}"),
                    )
                image_checks[tool.image] = cached
            available, message = cached
            results.append(
                DiagnosticResult(
                    f"image:{tool.tool_id}", tool.name, "pass" if available else "fail", message
                )
            )
    return results


def diagnostics_exit_code(results: Sequence[DiagnosticResult]) -> int:
    return 1 if any(result.status == "fail" for result in results) else 0
