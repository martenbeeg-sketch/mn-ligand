from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from uuid import uuid4


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_MANIFEST_NAME = "artifacts.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_artifact_path(value: str | Path) -> str:
    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or raw == ".":
        raise ValueError(f"Artifact path must be run-relative: {value!s}")
    return path.as_posix()


def _artifact_id(artifact_type: str, relative_path: str) -> str:
    digest = hashlib.sha256(f"{artifact_type}\0{relative_path}".encode()).hexdigest()[:16]
    return f"artifact-{digest}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    run_id: str
    artifact_type: str
    path: str
    artifact_id: str = ""
    role: str = ""
    label: str = ""
    media_type: str = ""
    size_bytes: int | None = None
    sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        relative_path = _relative_artifact_path(self.path)
        if not self.run_id.strip():
            raise ValueError("Artifact run_id cannot be empty")
        if not self.artifact_type.strip():
            raise ValueError("Artifact type cannot be empty")
        object.__setattr__(self, "path", relative_path)
        if not self.artifact_id:
            object.__setattr__(self, "artifact_id", _artifact_id(self.artifact_type, relative_path))

    @classmethod
    def from_path(
        cls,
        run_dir: Path,
        path: Path,
        artifact_type: str,
        *,
        role: str = "",
        label: str = "",
        metadata: dict[str, Any] | None = None,
        checksum: bool = True,
    ) -> ArtifactRef:
        resolved_run_dir = run_dir.resolve()
        resolved_path = path.resolve()
        try:
            relative_path = resolved_path.relative_to(resolved_run_dir).as_posix()
        except ValueError as exc:
            raise ValueError(f"Artifact is outside its run directory: {path}") from exc
        if not resolved_path.is_file():
            raise FileNotFoundError(resolved_path)
        media_type = mimetypes.guess_type(resolved_path.name)[0] or "application/octet-stream"
        return cls(
            run_id=run_dir.name,
            artifact_type=artifact_type,
            path=relative_path,
            role=role,
            label=label or resolved_path.name,
            media_type=media_type,
            size_bytes=resolved_path.stat().st_size,
            sha256=_sha256(resolved_path) if checksum else "",
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactRef:
        return cls(
            run_id=str(payload.get("run_id") or ""),
            artifact_type=str(payload.get("artifact_type") or ""),
            path=str(payload.get("path") or ""),
            artifact_id=str(payload.get("artifact_id") or payload.get("id") or ""),
            role=str(payload.get("role") or ""),
            label=str(payload.get("label") or ""),
            media_type=str(payload.get("media_type") or ""),
            size_bytes=payload.get("size_bytes"),
            sha256=str(payload.get("sha256") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )

    def resolve(self, run_dir: Path, *, must_exist: bool = False) -> Path | None:
        candidate = (run_dir.resolve() / self.path).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Artifact escapes its run directory: {self.path}") from exc
        if must_exist and not candidate.is_file():
            return None
        return candidate

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": "run_artifact",
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "artifact_type": self.artifact_type,
            "path": self.path,
        }
        for key, value in (
            ("role", self.role),
            ("label", self.label),
            ("media_type", self.media_type),
            ("size_bytes", self.size_bytes),
            ("sha256", self.sha256),
            ("metadata", self.metadata),
        ):
            if value not in ("", None, {}):
                payload[key] = value
        return payload


@dataclass(frozen=True)
class ArtifactManifest:
    run_id: str
    artifacts: tuple[ArtifactRef, ...] = ()
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    generated_at: str = field(default_factory=_utc_now_iso)
    source: str = "native"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactManifest:
        version = int(payload.get("schema_version") or 0)
        if version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact schema version: {version}")
        run_id = str(payload.get("run_id") or "")
        artifacts = tuple(ArtifactRef.from_dict(item) for item in payload.get("artifacts") or [])
        if any(artifact.run_id != run_id for artifact in artifacts):
            raise ValueError("Artifact run_id does not match its manifest")
        return cls(
            run_id=run_id,
            artifacts=artifacts,
            schema_version=version,
            generated_at=str(payload.get("generated_at") or ""),
            source=str(payload.get("source") or "native"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "artifact_manifest",
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "source": self.source,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    def by_type(self, artifact_type: str) -> tuple[ArtifactRef, ...]:
        return tuple(item for item in self.artifacts if item.artifact_type == artifact_type)

    def write(self, run_dir: Path) -> Path:
        if run_dir.name != self.run_id:
            raise ValueError("Manifest run_id does not match run directory")
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / ARTIFACT_MANIFEST_NAME
        temporary = run_dir / f".{ARTIFACT_MANIFEST_NAME}.{uuid4().hex}.tmp"
        temporary.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        temporary.replace(target)
        return target


_STRUCTURE_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("*_protein_refined.pdb", "prepared_receptor", "receptor"),
    ("*_complex_refined.pdb", "prepared_complex", "complex"),
    ("*_ligand_refined.sdf", "prepared_ligand_set", "ligand"),
    ("*_ligand_raw.sdf", "compound_set", "raw_ligand"),
    ("*_ligand_ref.smi", "compound_set", "reference_smiles"),
    ("repair_report.json", "repair_report", "report"),
    ("work/results/*_out.pdbqt", "pose_set", "docked_poses"),
    ("work/results/*_out.sdf", "pose_set", "docked_poses"),
    ("work/results/*_out.pdb", "pose_set", "docked_poses"),
)


def infer_legacy_artifacts(run_dir: Path, *, task_group: str = "") -> ArtifactManifest:
    artifacts: list[ArtifactRef] = []
    patterns = _STRUCTURE_PATTERNS if task_group in {"", "structure-jobs"} else ()
    seen: set[Path] = set()
    for pattern, artifact_type, role in patterns:
        for path in sorted(run_dir.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir,
                    path,
                    artifact_type,
                    role=role,
                    metadata={"legacy_inferred": True},
                )
            )
    return ArtifactManifest(run_id=run_dir.name, artifacts=tuple(artifacts), source="legacy_inferred")


def load_artifact_manifest(
    run_dir: Path,
    *,
    task_group: str = "",
    infer_legacy: bool = True,
) -> ArtifactManifest:
    manifest_path = run_dir / ARTIFACT_MANIFEST_NAME
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text())
        return ArtifactManifest.from_dict(payload)
    if infer_legacy:
        return infer_legacy_artifacts(run_dir, task_group=task_group)
    return ArtifactManifest(run_id=run_dir.name)


def write_artifact_manifest(run_dir: Path, artifacts: Iterable[ArtifactRef]) -> ArtifactManifest:
    manifest = ArtifactManifest(run_id=run_dir.name, artifacts=tuple(artifacts))
    manifest.write(run_dir)
    return manifest


def write_structure_artifact_manifest(run_dir: Path) -> ArtifactManifest:
    inferred = infer_legacy_artifacts(run_dir, task_group="structure-jobs")
    manifest = ArtifactManifest(run_id=run_dir.name, artifacts=inferred.artifacts)
    manifest.write(run_dir)
    return manifest
