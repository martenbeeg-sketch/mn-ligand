"""Tracked CPU precomputation for campaign pose-similarity tables."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.runtime import NATIVE_THREAD_ENVIRONMENT, adaptive_cpu_workers, runs_root


POSE_SIMILARITY_TASK_GROUP = "pose-similarity"
POSE_SIMILARITY_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def _normalise(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _signature(contexts: list[dict[str, Any]]) -> str:
    """Selection plus input-file state; avoids duplicate queued calculations."""
    inputs: list[dict[str, Any]] = []
    for context in contexts:
        paths = [context.get("reference_path"), context.get("reference_ligand_path")]
        paths.extend(item.get("source_path") for item in context.get("rendered", []))
        states = []
        for raw in paths:
            path = Path(str(raw or ""))
            try:
                stat = path.stat()
                states.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
            except OSError:
                states.append((str(path), None, None))
        inputs.append({"key": context.get("key"), "files": states})
    encoded = json.dumps({"version": POSE_SIMILARITY_SCHEMA_VERSION, "inputs": inputs}, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def find_pose_similarity_job(contexts: list[dict[str, Any]]) -> JobRecord | None:
    """Return the reusable job for this exact immutable input selection."""
    if not contexts:
        return None
    signature = _signature(contexts)
    group_dir = runs_root() / POSE_SIMILARITY_TASK_GROUP
    for metadata_path in group_dir.glob("*/metadata.json") if group_dir.is_dir() else ():
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError):
            continue
        if (
            metadata.get("input_signature") == signature
            and str(metadata.get("status") or "") in {"queued", "running", "completed"}
        ):
            return JobRecord.load(metadata_path.parent, task_group=POSE_SIMILARITY_TASK_GROUP)
    return None


def queue_pose_similarity_job(contexts: list[dict[str, Any]]) -> JobRecord:
    if not contexts:
        raise ValueError("No comparable pose contexts were selected")
    existing = find_pose_similarity_job(contexts)
    if existing is not None:
        return existing
    signature = _signature(contexts)
    max_workers = adaptive_cpu_workers(len(contexts), hard_cap=16)
    group_dir = runs_root() / POSE_SIMILARITY_TASK_GROUP
    run_id = str(uuid4())
    run_dir = group_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    now = _now()
    _write_json(run_dir / "input.json", {"schema_version": POSE_SIMILARITY_SCHEMA_VERSION, "contexts": _normalise(contexts)})
    _write_json(run_dir / "metadata.json", {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "pose_similarity",
        "workflow": "campaign_pose_similarity",
        "operation": "pose_similarity_precompute",
        "tool": "RDKit",
        "status": "queued",
        "input_signature": signature,
        "context_count": len(contexts),
        "created_at": now, "updated_at": now, "queued_at": now,
        "queued_command": [sys.executable, "-m", "mn_ligand.workflows.pose_similarity", "--run-dir", str(run_dir)],
        "gpu_queued": False,
        "resources": {"cpu_threads": max_workers, "gpu": False},
        "max_workers": max_workers,
        "cpu_worker_policy": "adaptive-global-limit",
        "progress": {"completed": 0, "total": len(contexts), "label": "Preparing pose comparisons"},
        "worker_finalizer": "pose_similarity",
    })
    write_artifact_manifest(run_dir, [])
    return JobRecord.load(run_dir, task_group=POSE_SIMILARITY_TASK_GROUP)


def cancel_queued_pose_similarity_job(job: JobRecord) -> None:
    """Cancel a not-yet-started calculation without touching its artifacts."""
    if job.task_group != POSE_SIMILARITY_TASK_GROUP or job.status != "queued":
        return
    path = job.run_dir / "metadata.json"
    try:
        metadata = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    metadata.update({"status": "cancelled", "cancelled_at": _now(), "updated_at": _now(), "completed_at": _now()})
    _write_json(path, metadata)


def _result_frame(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    matrix_data = payload.get("matrix") or {}
    matrix = pd.DataFrame(matrix_data.get("data") or [], index=matrix_data.get("index") or [], columns=matrix_data.get("columns") or [], dtype=float)
    return matrix, pd.DataFrame(payload.get("pairs") or []), [str(item) for item in payload.get("warnings") or []]


def load_pose_similarity_results(job: JobRecord) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    path = job.run_dir / "pose_similarity_results.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {str(key): _result_frame(value) for key, value in (payload.get("contexts") or {}).items() if isinstance(value, dict)}


def _calculate_context(context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """One independent target–compound comparison, safe for a CPU process."""
    os.environ["MN_LIGAND_POSE_SIMILARITY_WORKER"] = "1"
    from mn_ligand.app.pages.campaign_comparison import (  # noqa: PLC0415
        _model_text, _pose_similarity_tables, _preferred_viewer_structure_path,
    )
    from mn_ligand.app.viewers import aligned_structure_data  # noqa: PLC0415

    reference_path = Path(str(context["reference_path"]))
    rendered: list[dict[str, object]] = []
    for item in context.get("rendered") or []:
        row = pd.Series(item.get("row") or {})
        source_path = _preferred_viewer_structure_path(Path(str(item["source_path"])), structure_kind=str(item.get("kind") or ""))
        kind = str(item.get("kind") or "")
        if kind == "complex":
            data, _rmsd, _matched = aligned_structure_data(str(reference_path), reference_path.stat().st_mtime_ns, str(source_path), source_path.stat().st_mtime_ns)
        else:
            data, _format = _model_text(source_path, int(row.get("_viewer_pose_index") or 1))
        rendered.append({"row": row, "data": data, "kind": kind, "source_path": source_path})
    reference_data, _ = _model_text(reference_path)
    matrix, pairs, warnings = _pose_similarity_tables(rendered, reference_structure_data=reference_data, reference_ligand_path=Path(str(context.get("reference_ligand_path") or "")) if context.get("reference_ligand_path") else None)
    return str(context["key"]), {
        "matrix": {"index": matrix.index.tolist(), "columns": matrix.columns.tolist(), "data": matrix.where(pd.notna(matrix), None).values.tolist()},
        "pairs": pairs.where(pd.notna(pairs), None).to_dict(orient="records"), "warnings": warnings,
    }


def run_pose_similarity_job(run_dir: Path) -> None:
    """Calculate independent contexts in parallel within the global CPU limit."""
    for key, value in NATIVE_THREAD_ENVIRONMENT.items():
        os.environ.setdefault(key, value)
    input_payload = json.loads((run_dir / "input.json").read_text())
    contexts = list(input_payload.get("contexts") or [])
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    max_workers = adaptive_cpu_workers(
        len(contexts), requested=int(metadata.get("max_workers") or 1), hard_cap=16
    )
    output: dict[str, Any] = {"schema_version": POSE_SIMILARITY_SCHEMA_VERSION, "contexts": {}}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_calculate_context, context) for context in contexts]
        for index, future in enumerate(as_completed(futures), start=1):
            key, result = future.result()
            output["contexts"][key] = result
            _write_json(run_dir / "pose_similarity_results.json", output)
            metadata = json.loads(metadata_path.read_text())
            metadata["progress"] = {"completed": index, "total": len(contexts), "label": f"Calculated {index} of {len(contexts)} target-compound contexts in parallel"}
            metadata["updated_at"] = _now()
            _write_json(metadata_path, metadata)


def finalize_pose_similarity_job(run_dir: Path, *, returncode: int) -> JobRecord:
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    results = run_dir / "pose_similarity_results.json"
    success = returncode == 0 and results.is_file()
    metadata.update({"status": "completed" if success else "failed", "updated_at": _now(), "completed_at": _now()})
    if success:
        metadata.pop("error", None)
        metadata.pop("finalizer_error", None)
    else:
        metadata["error"] = "Pose-similarity precomputation did not complete."
    _write_json(metadata_path, metadata)
    artifacts = [ArtifactRef.from_path(run_dir, results, "pose_similarity_results", role="result", label="Precomputed pose similarity") ] if results.is_file() else []
    write_artifact_manifest(run_dir, artifacts)
    return JobRecord.load(run_dir, task_group=POSE_SIMILARITY_TASK_GROUP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_pose_similarity_job(Path(args.run_dir).resolve())
