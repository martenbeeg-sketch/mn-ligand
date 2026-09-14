from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from mn_ligand.core.artifacts import ArtifactManifest, load_artifact_manifest


JOB_SCHEMA_VERSION = 1
VALID_JOB_STATES = frozenset(
    {"queued", "preparing", "running", "paused", "completed", "failed", "cancelled", "blocked", "unknown"}
)


def short_job_code(run_id: str) -> str:
    suffix = run_id.rsplit("-", 1)[-1]
    compact = "".join(character for character in suffix.upper() if character.isalnum())
    if len(compact) >= 5:
        return compact[:5]
    fallback = "".join(character for character in run_id.upper() if character.isalnum())
    return (compact + fallback)[:5] or "JOB00"


def display_job_code(metadata_code: object, run_id: str) -> str:
    code = str(metadata_code or "").strip().upper()
    if len(code) == 5 and code.isalnum():
        return code
    return short_job_code(run_id)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


@dataclass(frozen=True)
class JobRecord:
    run_id: str
    task_group: str
    run_dir: Path
    status: str
    job_type: str = ""
    workflow: str = ""
    tool: str = ""
    parent_run_id: str = ""
    workflow_id: str = ""
    workflow_parent_run_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    artifact_manifest: ArtifactManifest | None = None
    warnings: tuple[str, ...] = ()
    schema_version: int = JOB_SCHEMA_VERSION

    @classmethod
    def load(
        cls,
        run_dir: Path,
        *,
        task_group: str | None = None,
        load_result: bool = True,
        load_artifacts: bool = True,
        validate_artifacts: bool = True,
    ) -> JobRecord:
        metadata = _read_json(run_dir / "metadata.json")
        result = _read_json(run_dir / "result.json") if load_result else {}
        warnings: list[str] = []
        if not metadata:
            warnings.append("Missing or unreadable metadata.json")
        status = str(metadata.get("status") or ("completed" if result else "unknown")).lower()
        if status not in VALID_JOB_STATES:
            warnings.append(f"Unrecognized job status: {status}")
            status = "unknown"
        if status == "failed":
            warnings.append(str(result.get("error") or metadata.get("error") or "Job failed"))
        result_warning = str(result.get("warning") or "").strip()
        if result_warning:
            warnings.append(result_warning)
        group = task_group or run_dir.parent.name
        artifact_manifest: ArtifactManifest | None = None
        if load_artifacts:
            try:
                artifact_manifest = load_artifact_manifest(
                    run_dir, task_group=group
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                artifact_manifest = ArtifactManifest(
                    run_id=run_dir.name, source="unavailable"
                )
                warnings.append(f"Artifact manifest error: {exc}")
        if validate_artifacts and artifact_manifest is not None:
            missing_artifacts = [
                artifact.path
                for artifact in artifact_manifest.artifacts
                if artifact.resolve(run_dir, must_exist=True) is None
            ]
            if missing_artifacts:
                warnings.append(
                    f"Missing {len(missing_artifacts)} declared artifact(s)"
                )
        return cls(
            run_id=str(metadata.get("run_id") or run_dir.name),
            task_group=group,
            run_dir=run_dir.resolve(),
            status=status,
            job_type=str(metadata.get("job_type") or group),
            workflow=str(metadata.get("workflow") or metadata.get("workflow_key") or ""),
            tool=str(metadata.get("engine") or metadata.get("tool") or metadata.get("source") or ""),
            parent_run_id=str(
                metadata.get("parent_run_id")
                or metadata.get("source_structure_run_id")
                or metadata.get("structure_run_id")
                or ""
            ),
            workflow_id=str(metadata.get("workflow_id") or ""),
            workflow_parent_run_id=str(metadata.get("workflow_parent_run_id") or ""),
            created_at=str(metadata.get("created_at") or ""),
            updated_at=str(metadata.get("updated_at") or ""),
            completed_at=str(metadata.get("completed_at") or ""),
            metadata=metadata,
            result=result,
            artifact_manifest=artifact_manifest,
            warnings=tuple(warnings),
            schema_version=int(metadata.get("schema_version") or 0),
        )


def iter_job_records(
    runs_dir: Path,
    *,
    task_groups: Iterable[str] | None = None,
    load_result: bool = True,
    load_artifacts: bool = True,
    validate_artifacts: bool = True,
) -> list[JobRecord]:
    if task_groups is None:
        groups = sorted(path.name for path in runs_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    else:
        groups = list(task_groups)

    records: list[JobRecord] = []
    for group in groups:
        group_dir = runs_dir / group
        if not group_dir.is_dir():
            continue
        for run_dir in group_dir.iterdir():
            if run_dir.is_dir() and not run_dir.name.startswith("."):
                records.append(
                    JobRecord.load(
                        run_dir,
                        task_group=group,
                        load_result=load_result,
                        load_artifacts=load_artifacts,
                        validate_artifacts=validate_artifacts,
                    )
                )
    return sorted(
        records,
        key=lambda record: (record.created_at, record.run_dir.stat().st_mtime),
        reverse=True,
    )
