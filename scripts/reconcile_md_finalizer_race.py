#!/usr/bin/env python3
"""Recover a successful MD production child misclassified by a finalizer race.

This repair is deliberately narrow: it requires the known temporary-manifest
rename error, a zero native return code, the native success marker, and a
complete typed trajectory/final-structure/checkpoint manifest.  It recomputes
trajectory analytics, preserves the failed bookkeeping files in an audit
directory, then lets the normal MD finalizer rebuild authoritative metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mn_ligand.core.jobs import JobRecord
from mn_ligand.runtime import resolve_run_dir
from mn_ligand.workflows.md_simulation import finalize_md_job

if __package__:
    from scripts.recompute_md_analysis import _compute_replica
else:
    from recompute_md_analysis import _compute_replica


KNOWN_ERROR = "Worker finalizer failed: [Errno 2] No such file or directory"
SUCCESS_MARKER = '"status": "MD production completed successfully"'
REQUIRED_ARTIFACT_TYPES = frozenset(
    {"md_trajectory", "md_final_structure", "md_checkpoint"}
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _replace_run_id(value: Any, source_run_id: str, target_run_id: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_run_id(item, source_run_id, target_run_id)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_run_id(item, source_run_id, target_run_id)
            for item in value
        ]
    if isinstance(value, str):
        return value.replace(source_run_id, target_run_id)
    return value


def reconcile(target_run_id: str, template_run_id: str) -> JobRecord:
    target_dir = resolve_run_dir("bound-ligand-md", target_run_id)
    template_dir = resolve_run_dir("bound-ligand-md", template_run_id)
    if target_dir is None or template_dir is None:
        raise FileNotFoundError("Target or successful template production run is missing")

    target = JobRecord.load(target_dir, task_group="bound-ligand-md")
    template = JobRecord.load(template_dir, task_group="bound-ligand-md")
    error = str(target.result.get("error") or target.metadata.get("error") or "")
    returncode = target.result.get("returncode")
    stderr = (target_dir / "stderr.log").read_text(errors="replace")
    artifact_types = {
        artifact.artifact_type
        for artifact in (target.artifact_manifest.artifacts if target.artifact_manifest else ())
    }
    missing_files = [
        artifact.path
        for artifact in (target.artifact_manifest.artifacts if target.artifact_manifest else ())
        if artifact.resolve(target_dir, must_exist=True) is None
    ]
    if target.status != "failed" or KNOWN_ERROR not in error:
        raise RuntimeError("Target does not have the known artifact-finalizer race")
    if returncode != 0 or SUCCESS_MARKER not in stderr:
        raise RuntimeError("Native MD success evidence is incomplete")
    if not REQUIRED_ARTIFACT_TYPES.issubset(artifact_types) or missing_files:
        raise RuntimeError(
            "Typed native artifact evidence is incomplete: "
            f"types={sorted(artifact_types)}, missing={missing_files}"
        )
    if template.status != "completed" or template.result.get("success") is not True:
        raise RuntimeError("Template must be a successful sibling continuation run")
    if target.metadata.get("workflow_id") != template.metadata.get("workflow_id"):
        raise RuntimeError("Template and target must belong to the same workflow")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_dir = target_dir / ".finalizer-reconciliation" / stamp
    audit_dir.mkdir(parents=True, exist_ok=False)
    for filename in ("result.json", "metadata.json", "artifacts.json"):
        shutil.copy2(target_dir / filename, audit_dir / filename)

    recovered = _replace_run_id(
        template.result,
        template.run_id,
        target.run_id,
    )
    recovered["success"] = True
    recovered["reconciliation"] = {
        "kind": "md_finalizer_race_recovery",
        "reconciled_at": datetime.now(timezone.utc).isoformat(),
        "template_run_id": template.run_id,
        "native_returncode": returncode,
        "native_success_marker": True,
        "prior_error": error,
        "audit_path": audit_dir.relative_to(target_dir).as_posix(),
    }

    output_files = recovered.setdefault("md_result", {}).setdefault(
        "output_files", {}
    )
    by_type = {
        artifact.artifact_type: artifact
        for artifact in target.artifact_manifest.artifacts
    }
    output_files.update(
        {
            "production_trajectory": f"/output/{by_type['md_trajectory'].path}",
            "production_pdb": f"/output/{by_type['md_final_structure'].path}",
            "production_checkpoint": f"/output/{by_type['md_checkpoint'].path}",
        }
    )
    production_logs = sorted(target_dir.rglob("predicted_lig_production.log"))
    if production_logs:
        output_files["production_log"] = (
            "/output/" + production_logs[0].relative_to(target_dir).as_posix()
        )

    # The failed worker overwrote the native result.  Install a provisional
    # path-complete result so the standard analytics reader can resolve the
    # exact target trajectory, then replace only its analytics with recomputed
    # values from that trajectory.
    _write_json(target_dir / "result.json", recovered)
    try:
        computed = _compute_replica(
            str(target_dir),
            str(audit_dir),
            target.metadata.get("residue_mapping"),
        )
        analytics = _read_json(Path(computed["staged_path"]))
        recovered["md_result"]["analytics"] = analytics
        _write_json(target_dir / "result.json", recovered)
        finalized = finalize_md_job(
            JobRecord.load(target_dir, task_group="bound-ligand-md")
        )
        metadata = _read_json(target_dir / "metadata.json")
        metadata.pop("error", None)
        metadata.pop("finalizer_error", None)
        metadata["reconciliation"] = recovered["reconciliation"]
        metadata["updated_at"] = recovered["reconciliation"]["reconciled_at"]
        _write_json(target_dir / "metadata.json", metadata)
        return JobRecord.load(target_dir, task_group="bound-ligand-md")
    except Exception:
        shutil.copy2(audit_dir / "result.json", target_dir / "result.json")
        shutil.copy2(audit_dir / "metadata.json", target_dir / "metadata.json")
        shutil.copy2(audit_dir / "artifacts.json", target_dir / "artifacts.json")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_run_id")
    parser.add_argument("template_run_id")
    args = parser.parse_args()
    job = reconcile(args.target_run_id, args.template_run_id)
    print(f"{job.run_id} {job.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
