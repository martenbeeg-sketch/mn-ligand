from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from mn_ligand.core.artifacts import load_artifact_manifest
from mn_ligand.core.portable_paths import resolve_stored_path
from mn_ligand.runtime import app_home, library_root, reference_root, runs_root


CONTAINER_PREFIXES = (
    "/app/",
    "/boltz-context/",
    "/cache/",
    "/data/",
    "/input/",
    "/models/",
    "/mn-ligand/",
    "/opt/",
    "/output/",
    "/prepared-system/",
    "/reference/",
    "/references/",
    "/source/",
    "/tmp/",
    "/work/",
    "/workspace/",
)
PROVENANCE_FILENAMES = frozenset(
    {"command.json", "command_history.json", "native_result.json"}
)
OPERATIONAL_KEY_PARTS = (
    "cache",
    "checkpoint",
    "directory",
    "filename",
    "hparams",
    "model_path",
    "path",
    "repository",
    "run_dir",
)
PATH_METADATA_FILENAMES = frozenset(
    {
        "artifacts.json",
        "benchmark_dataset.json",
        "command.json",
        "input.json",
        "metadata.json",
        "native_result.json",
        "result.json",
        "selection.json",
        "source_input.json",
        "source_result.json",
        "workflow.json",
    }
)
EXPORT_MANIFEST_NAME = "portable-export.json"
EXPORT_REPORT_NAME = "migration-report.json"
EXPORT_SCHEMA_VERSION = 1
JOB_PORTABILITY_SCHEMA_VERSION = 1
PORTABLE_PREFIXES = ("runs:///", "reference:///", "library:///", "app:///")


class PortabilityError(RuntimeError):
    """Raised when a safe portable export or verification cannot complete."""


def _valid_portable_uri(raw: str) -> bool:
    parsed = urlsplit(raw)
    if parsed.scheme not in {"runs", "reference", "library", "app"}:
        return False
    joined = "/".join(part for part in (parsed.netloc, parsed.path) if part)
    relative = PurePosixPath(joined.lstrip("/"))
    return bool(relative.parts) and ".." not in relative.parts


def validate_job_portability(run_dir: Path) -> dict[str, Any]:
    """Check one job record for machine-bound operational paths.

    This is intentionally read-only. Commands and native/provenance output are
    retained verbatim, while operational path fields in canonical job JSON must
    be relative, use a managed portable URI, or refer to a container mount.
    """
    job_dir = Path(run_dir).expanduser().resolve()
    errors: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for json_path in sorted(job_dir.glob("*.json")):
        if json_path.name not in PATH_METADATA_FILENAMES:
            continue
        counts["json_files"] += 1
        try:
            payload = json.loads(json_path.read_text())
        except (OSError, TypeError, ValueError) as exc:
            errors.append({"file": json_path.name, "key": "", "error": f"unreadable JSON: {exc}"})
            continue
        for key, raw in _strings(payload):
            if not _operational_key(key) or _provenance_key(key):
                continue
            if json_path.name in PROVENANCE_FILENAMES:
                continue
            if raw.startswith(PORTABLE_PREFIXES):
                counts["portable_paths"] += 1
                if not _valid_portable_uri(raw):
                    errors.append(
                        {"file": json_path.name, "key": key, "error": f"invalid portable path: {raw}"}
                    )
                continue
            if not raw.startswith("/"):
                if raw:
                    counts["relative_paths"] += 1
                continue
            if raw.startswith(CONTAINER_PREFIXES):
                counts["container_paths"] += 1
                continue
            counts["absolute_host_paths"] += 1
            errors.append(
                {
                    "file": json_path.name,
                    "key": key,
                    "error": f"absolute host path is not portable: {raw}",
                }
            )
    return {
        "schema_version": JOB_PORTABILITY_SCHEMA_VERSION,
        "run_dir": str(job_dir),
        "valid": not errors,
        "counts": dict(sorted(counts.items())),
        "errors": errors,
    }


def assert_job_portable(run_dir: Path) -> None:
    """Raise when a new-schema job contains a machine-bound path."""
    report = validate_job_portability(run_dir)
    if report["valid"]:
        return
    first = report["errors"][0]
    raise PortabilityError(
        f"Job portability validation failed for {report['run_dir']}: "
        f"{first['file']} {first.get('key') or '<value>'}: {first['error']}"
    )


@dataclass(frozen=True)
class _ExportRoot:
    scheme: str
    source: Path
    destination: Path


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _active_job_paths(runs: Path) -> list[Path]:
    active: list[Path] = []
    for metadata_path in runs.glob("*/*/metadata.json"):
        try:
            payload = json.loads(metadata_path.read_text())
        except (OSError, TypeError, ValueError):
            continue
        if str(payload.get("status") or "").lower() in {
            "queued",
            "preparing",
            "running",
            "paused",
        }:
            active.append(metadata_path)
    return active


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _operational_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in OPERATIONAL_KEY_PARTS)


def _provenance_key(key: str) -> bool:
    normalized = key.lower()
    return "command" in normalized or key in {
        "argv",
        "executable",
        "queued_command",
    }


def _managed_source_path(raw: str, roots: tuple[_ExportRoot, ...]) -> Path | None:
    recorded = Path(raw).expanduser()
    if recorded.exists():
        return recorded.resolve()
    normalized = raw.replace("\\", "/")
    markers = {
        "runs": ("/workdir/runs/", "/.ovo-home/workdir/runs/"),
        "reference": ("/reference_files/",),
        "library": ("/libraries/",),
    }
    for root in roots:
        for marker in markers.get(root.scheme, ()):
            if marker not in normalized:
                continue
            relative = normalized.split(marker, 1)[1].lstrip("/")
            if not relative or ".." in Path(relative).parts:
                return None
            candidate = root.source / relative
            return candidate.resolve() if candidate.exists() else None
    return None


def _portable_export_value(
    raw: str,
    roots: tuple[_ExportRoot, ...],
) -> tuple[str | None, Path | None, Path | None]:
    source = _managed_source_path(raw, roots)
    if source is None:
        return None, None, None
    for root in roots:
        if not _is_relative_to(source, root.source):
            continue
        relative = source.relative_to(root.source)
        destination = root.destination / relative
        encoded = f"{root.scheme}:///{relative.as_posix()}"
        return encoded, source, destination
    return None, source, None


def _rewrite_export_payload(
    value: Any,
    *,
    roots: tuple[_ExportRoot, ...],
    report: dict[str, Any],
    relative_file: str,
    key: str = "",
    provenance: bool = False,
) -> Any:
    inherited_provenance = provenance or _provenance_key(key)
    if isinstance(value, dict):
        return {
            child_key: _rewrite_export_payload(
                child,
                roots=roots,
                report=report,
                relative_file=relative_file,
                key=str(child_key),
                provenance=inherited_provenance,
            )
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_export_payload(
                child,
                roots=roots,
                report=report,
                relative_file=relative_file,
                key=key,
                provenance=inherited_provenance,
            )
            for child in value
        ]
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or inherited_provenance
        or not _operational_key(key)
    ):
        return value
    encoded, source, destination = _portable_export_value(value, roots)
    if encoded is None or source is None or destination is None:
        if value.startswith(CONTAINER_PREFIXES):
            return value
        report["unresolved"].append(
            {"file": relative_file, "key": key, "recorded": value}
        )
        return value
    if not destination.exists():
        report["unresolved"].append(
            {
                "file": relative_file,
                "key": key,
                "recorded": value,
                "reason": "resolved source was not copied",
            }
        )
        return value
    report["rewritten"].append(
        {
            "file": relative_file,
            "key": key,
            "recorded": value,
            "portable": encoded,
        }
    )
    return encoded


def _strings(value: Any, key: str = ""):
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _strings(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child, key)
    elif isinstance(value, str):
        yield key, value


def audit_runtime_portability(root: Path | None = None) -> dict[str, Any]:
    """Inspect runtime JSON without writing to the runtime tree."""
    run_root = Path(root or runs_root(create=False)).expanduser().resolve()
    counts: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = {
        "relocatable": [],
        "unresolved": [],
        "external": [],
    }
    reference = reference_root(create=False).resolve()
    resolution_cache: dict[str, tuple[str, str]] = {}

    json_paths: list[Path] = []
    if run_root.is_dir():
        for directory, _, filenames in os.walk(run_root):
            for filename in filenames:
                if not filename.endswith(".json"):
                    continue
                counts["json_files"] += 1
                if (
                    filename in PATH_METADATA_FILENAMES
                    or filename.endswith(".pre-gpu-rebalance.json")
                ):
                    json_paths.append(Path(directory) / filename)
    counts["path_metadata_files"] = len(json_paths)

    for json_path in json_paths:
        try:
            raw_json = json_path.read_text()
        except (OSError, ValueError, TypeError):
            counts["unreadable_json_files"] += 1
            continue
        if '"/' not in raw_json:
            continue
        try:
            payload = json.loads(raw_json)
        except (ValueError, TypeError):
            counts["unreadable_json_files"] += 1
            continue
        for key, raw in _strings(payload):
            if len(raw) > 4096 or not raw.startswith("/"):
                continue
            counts["absolute_values"] += 1
            key_counts[key or "<list item>"] += 1
            file_counts[json_path.name] += 1
            if raw.startswith(CONTAINER_PREFIXES):
                counts["container_paths"] += 1
                continue
            normalized_key = key.lower()
            if (
                json_path.name in PROVENANCE_FILENAMES
                or "command" in normalized_key
                or key in {"argv", "executable", "queued_command"}
            ):
                counts["provenance_paths"] += 1
                continue
            if not any(part in normalized_key for part in OPERATIONAL_KEY_PARTS):
                counts["embedded_text_paths"] += 1
                continue
            cached = resolution_cache.get(raw)
            if cached is None:
                recorded = Path(raw).expanduser()
                try:
                    recorded_exists = recorded.exists()
                except OSError:
                    recorded_exists = False
                if recorded_exists:
                    try:
                        recorded.resolve().relative_to(reference)
                    except ValueError:
                        category = "current"
                    else:
                        category = "reference"
                    resolved_text = str(recorded)
                else:
                    relocated = resolve_stored_path(raw, must_exist=True)
                    if relocated is not None:
                        category = "relocatable"
                        resolved_text = str(relocated)
                    else:
                        managed = (
                            "/workdir/runs/" in raw or "/reference_files/" in raw
                        )
                        category = "unresolved" if managed else "external"
                        resolved_text = ""
                cached = (category, resolved_text)
                resolution_cache[raw] = cached
            category, resolved_text = cached
            counts[f"{category}_paths"] += 1
            if category == "reference":
                counts["current_paths"] += 1
            if category in {"current", "reference"}:
                continue
            bucket = category
            if len(examples[bucket]) < 20:
                examples[bucket].append(
                    {
                        "file": json_path.relative_to(run_root).as_posix(),
                        "key": key,
                        "recorded": raw,
                        "resolved": resolved_text,
                    }
                )

    return {
        "schema_version": 1,
        "read_only": True,
        "runs_root": str(run_root),
        "counts": dict(sorted(counts.items())),
        "absolute_path_keys": dict(key_counts.most_common()),
        "files_by_name": dict(file_counts.most_common()),
        "examples": examples,
    }


def export_portable_runtime(
    destination: Path,
    *,
    source_runs: Path | None = None,
    source_references: Path | None = None,
    source_libraries: Path | None = None,
    source_app_home: Path | None = None,
    include_references: bool = True,
    include_libraries: bool = True,
) -> dict[str, Any]:
    """Create a validated portable copy without modifying the source archive."""
    target = destination.expanduser().resolve()
    runs = Path(source_runs or runs_root(create=False)).expanduser().resolve()
    references = Path(
        source_references or reference_root(create=False)
    ).expanduser().resolve()
    libraries = Path(
        source_libraries or library_root(create=False)
    ).expanduser().resolve()
    source_app = (
        Path(source_app_home).expanduser().resolve()
        if source_app_home is not None
        else None
    )
    if target.exists():
        raise PortabilityError(f"Destination already exists: {target}")
    if not runs.is_dir():
        raise PortabilityError(f"Runs directory does not exist: {runs}")
    active_jobs = _active_job_paths(runs)
    if active_jobs:
        examples = ", ".join(
            path.parent.name for path in active_jobs[:5]
        )
        raise PortabilityError(
            f"Refusing to export while {len(active_jobs)} job(s) are active: {examples}. "
            "Stop workers or let jobs reach a terminal state first."
        )
    source_candidates = [runs]
    if include_references:
        if not references.is_dir():
            raise PortabilityError(f"Reference directory does not exist: {references}")
        source_candidates.append(references)
    if include_libraries:
        if not libraries.is_dir():
            raise PortabilityError(f"Library directory does not exist: {libraries}")
        source_candidates.append(libraries)
    if source_app is not None:
        if not source_app.is_dir():
            raise PortabilityError(f"App home does not exist: {source_app}")
        source_candidates.append(source_app)
    if any(_is_relative_to(target, source) for source in source_candidates):
        raise PortabilityError("Destination cannot be inside a copied source directory")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    destination_runs = staging / "workdir" / "runs"
    roots: list[_ExportRoot] = [
        _ExportRoot("runs", runs, destination_runs),
    ]
    if include_references:
        roots.append(
            _ExportRoot("reference", references, staging / "reference_files")
        )
    if include_libraries:
        roots.append(_ExportRoot("library", libraries, staging / "libraries"))
    if source_app is not None:
        roots.append(_ExportRoot("app", source_app, staging))
    export_roots = tuple(roots)
    source_json_hashes: dict[str, str] = {}
    report: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "source_unchanged": False,
        "rewritten": [],
        "unresolved": [],
        "json_files": [],
    }
    try:
        for source in source_candidates:
            for path in source.rglob("*.json"):
                if path.is_file():
                    source_json_hashes[str(path)] = _sha256_file(path)

        for root in sorted(export_roots, key=lambda item: item.scheme != "app"):
            if root.scheme != "app":
                shutil.copytree(root.source, root.destination, copy_function=shutil.copy2)
                continue
            excluded_top_level = {"config", "tmp"}
            for other in export_roots:
                if other.scheme == "app" or not _is_relative_to(other.source, root.source):
                    continue
                relative = other.source.relative_to(root.source)
                if relative.parts:
                    excluded_top_level.add(relative.parts[0])

            def ignore_app(directory: str, names: list[str]) -> set[str]:
                if Path(directory).resolve() == root.source:
                    return set(names).intersection(excluded_top_level)
                return set()

            shutil.copytree(
                root.source,
                root.destination,
                copy_function=shutil.copy2,
                ignore=ignore_app,
            )

        for json_path in destination_runs.rglob("*.json"):
            if json_path.name not in PATH_METADATA_FILENAMES:
                continue
            relative_file = json_path.relative_to(staging).as_posix()
            try:
                payload = json.loads(json_path.read_text())
            except (OSError, TypeError, ValueError) as exc:
                report["unresolved"].append(
                    {
                        "file": relative_file,
                        "key": "",
                        "recorded": "",
                        "reason": f"unreadable JSON: {exc}",
                    }
                )
                continue
            original_hash = _sha256_file(json_path)
            if json_path.name not in PROVENANCE_FILENAMES:
                payload = _rewrite_export_payload(
                    payload,
                    roots=export_roots,
                    report=report,
                    relative_file=relative_file,
                )
                _atomic_json(json_path, payload)
            report["json_files"].append(
                {
                    "file": relative_file,
                    "source_sha256": original_hash,
                    "export_sha256": _sha256_file(json_path),
                }
            )

        changed_sources = [
            path
            for path, digest in source_json_hashes.items()
            if not Path(path).is_file() or _sha256_file(Path(path)) != digest
        ]
        if changed_sources:
            raise PortabilityError(
                "Source JSON changed during export: " + ", ".join(changed_sources[:5])
            )
        report["source_unchanged"] = True
        if report["unresolved"]:
            raise PortabilityError(
                f"Portable export found {len(report['unresolved'])} unresolved "
                "operational path(s)"
            )
        manifest = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "layout": {
                "runs": "workdir/runs",
                "references": "reference_files" if include_references else "",
                "libraries": "libraries" if include_libraries else "",
                "app": "." if source_app is not None else "",
            },
            "source": {
                "runs": str(runs),
                "references": str(references) if include_references else "",
                "libraries": str(libraries) if include_libraries else "",
                "app": str(source_app) if source_app is not None else "",
            },
            "counts": {
                "source_json_files": len(source_json_hashes),
                "inspected_json_files": len(report["json_files"]),
                "rewritten_paths": len(report["rewritten"]),
                "unresolved_paths": 0,
            },
        }
        _atomic_json(staging / EXPORT_REPORT_NAME, report)
        _atomic_json(staging / EXPORT_MANIFEST_NAME, manifest)
        verification = verify_portable_export(staging)
        if not verification["valid"]:
            raise PortabilityError(
                f"Staged export verification failed with "
                f"{len(verification['errors'])} error(s)"
            )
        staging.rename(target)
        return {**manifest, "destination": str(target), "verification": verification}
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _portable_bundle_candidate(raw: str, bundle: Path, layout: dict[str, str]) -> Path | None:
    for scheme, key in (
        ("runs:///", "runs"),
        ("reference:///", "references"),
        ("library:///", "libraries"),
        ("app:///", "app"),
    ):
        if not raw.startswith(scheme):
            continue
        root_value = str(layout.get(key) or "")
        if not root_value:
            return None
        relative = Path(raw.removeprefix(scheme))
        if relative.is_absolute() or ".." in relative.parts:
            return None
        root = (bundle / root_value).resolve()
        candidate = (root / relative).resolve()
        return candidate if _is_relative_to(candidate, root) else None
    return None


def verify_portable_export(destination: Path) -> dict[str, Any]:
    """Validate a previously exported bundle without changing it."""
    bundle = destination.expanduser().resolve()
    errors: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    manifest_path = bundle / EXPORT_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, TypeError, ValueError) as exc:
        return {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "destination": str(bundle),
            "valid": False,
            "counts": {},
            "errors": [{"file": EXPORT_MANIFEST_NAME, "error": str(exc)}],
        }
    if int(manifest.get("schema_version") or 0) != EXPORT_SCHEMA_VERSION:
        errors.append(
            {"file": EXPORT_MANIFEST_NAME, "error": "unsupported schema version"}
        )
    layout = dict(manifest.get("layout") or {})
    runs_value = str(layout.get("runs") or "")
    exported_runs = (bundle / runs_value).resolve() if runs_value else bundle / "missing"
    if not exported_runs.is_dir() or not _is_relative_to(exported_runs, bundle):
        errors.append({"file": runs_value, "error": "runs directory is unavailable"})
    else:
        for json_path in exported_runs.rglob("*.json"):
            if json_path.name not in PATH_METADATA_FILENAMES:
                continue
            counts["json_files"] += 1
            try:
                payload = json.loads(json_path.read_text())
            except (OSError, TypeError, ValueError) as exc:
                errors.append(
                    {
                        "file": json_path.relative_to(bundle).as_posix(),
                        "error": f"unreadable JSON: {exc}",
                    }
                )
                continue
            for key, raw in _strings(payload):
                candidate = _portable_bundle_candidate(raw, bundle, layout)
                if raw.startswith(
                    ("runs:///", "reference:///", "library:///", "app:///")
                ):
                    counts["portable_paths"] += 1
                    if candidate is None:
                        errors.append(
                            {
                                "file": json_path.relative_to(bundle).as_posix(),
                                "error": f"portable path has no safe bundle root: {raw}",
                            }
                        )
                    elif not candidate.exists():
                        errors.append(
                            {
                                "file": json_path.relative_to(bundle).as_posix(),
                                "error": f"portable path is missing: {raw}",
                            }
                        )
                elif (
                    raw.startswith("/")
                    and not raw.startswith(CONTAINER_PREFIXES)
                    and json_path.name not in PROVENANCE_FILENAMES
                    and _operational_key(key)
                    and not _provenance_key(key)
                ):
                    errors.append(
                        {
                            "file": json_path.relative_to(bundle).as_posix(),
                            "error": f"absolute operational path remains: {raw}",
                        }
                    )

        for artifact_path in exported_runs.rglob("artifacts.json"):
            counts["artifact_manifests"] += 1
            run_dir = artifact_path.parent
            try:
                artifact_manifest = load_artifact_manifest(
                    run_dir, infer_legacy=False
                )
                for artifact in artifact_manifest.artifacts:
                    counts["artifacts"] += 1
                    resolved = artifact.resolve(run_dir, must_exist=True)
                    if resolved is None:
                        errors.append(
                            {
                                "file": artifact_path.relative_to(bundle).as_posix(),
                                "error": f"missing artifact: {artifact.path}",
                            }
                        )
                    elif artifact.sha256 and _sha256_file(resolved) != artifact.sha256:
                        errors.append(
                            {
                                "file": artifact_path.relative_to(bundle).as_posix(),
                                "error": f"artifact checksum mismatch: {artifact.path}",
                            }
                        )
            except (OSError, TypeError, ValueError) as exc:
                errors.append(
                    {
                        "file": artifact_path.relative_to(bundle).as_posix(),
                        "error": f"artifact manifest error: {exc}",
                    }
                )
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "destination": str(bundle),
        "valid": not errors,
        "counts": dict(sorted(counts.items())),
        "errors": errors,
    }
