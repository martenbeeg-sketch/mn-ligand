#!/usr/bin/env python3
"""Backfill selector-compatible target identities into MD metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.provenance import compact_target_identifier, target_key
from mn_ligand.runtime import resolve_run_dir, runs_root


TARGET_FIELDS = ("target_key", "target_run_id", "target_provenance_key")


def _read(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    temporary.replace(path)


def _source_job(parameters: dict) -> JobRecord | None:
    source = parameters.get("source")
    source = source if isinstance(source, dict) else {}
    source_id = str(source.get("run_id") or "")
    source_group = str(source.get("task_group") or "")
    if source_id and source_group:
        source_dir = resolve_run_dir(source_group, source_id)
        if source_dir is not None:
            return JobRecord.load(source_dir, task_group=source_group)

    prep_id = str(parameters.get("prepared_system_run_id") or "")
    prep_dir = resolve_run_dir("md-system-prep", prep_id) if prep_id else None
    if prep_dir is None:
        return None
    prep = JobRecord.load(prep_dir, task_group="md-system-prep")
    source_id = str(
        prep.metadata.get("target_run_id")
        or prep.metadata.get("structure_run_id")
        or ""
    )
    if not source_id:
        return None
    for candidate in runs_root().glob(f"*/{source_id}"):
        if candidate.is_dir():
            return JobRecord.load(candidate, task_group=candidate.parent.name)
    return None


def _identity(source: JobRecord) -> dict[str, str]:
    fallback = str(
        source.metadata.get("pdb_id")
        or source.metadata.get("source_pdb_id")
        or source.run_id
    )
    return {
        "target_key": target_key(
            run_id=source.run_id,
            metadata=source.metadata,
            fallback_origin=fallback,
        ),
        "target_run_id": source.run_id,
        "target_provenance_key": compact_target_identifier(
            run_id=source.run_id,
            metadata=source.metadata,
            fallback_origin=fallback,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes; otherwise only report what would change.",
    )
    args = parser.parse_args()
    changed_workflows = 0
    changed_children = 0
    for workflow_dir in sorted((runs_root() / "workflows").iterdir()):
        workflow_path = workflow_dir / "workflow.json"
        workflow = _read(workflow_path)
        if str(workflow.get("workflow_type") or "") != "md-simulation":
            continue
        parameters = workflow.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        source = _source_job(parameters)
        if source is None:
            print(f"SKIP {workflow_dir.name}: source target not found")
            continue
        identity = _identity(source)
        workflow_changed = any(parameters.get(key) != value for key, value in identity.items())
        parameters.update(identity)
        workflow["parameters"] = parameters

        metadata_path = workflow_dir / "metadata.json"
        metadata = _read(metadata_path)
        metadata_parameters = metadata.get("parameters")
        metadata_parameters = (
            metadata_parameters if isinstance(metadata_parameters, dict) else {}
        )
        metadata_parameters.update(identity)
        metadata["parameters"] = metadata_parameters
        metadata.update(identity)

        input_path = workflow_dir / "input.json"
        workflow_input = _read(input_path)
        input_parameters = workflow_input.get("parameters")
        input_parameters = input_parameters if isinstance(input_parameters, dict) else {}
        input_parameters.update(identity)
        workflow_input["parameters"] = input_parameters

        child_changes: list[tuple[Path, dict]] = []
        for child in workflow.get("children") or []:
            if not isinstance(child, dict):
                continue
            child_dir = resolve_run_dir(
                str(child.get("task_group") or ""),
                str(child.get("run_id") or ""),
            )
            if child_dir is None:
                continue
            child_path = child_dir / "metadata.json"
            child_metadata = _read(child_path)
            if any(child_metadata.get(key) != value for key, value in identity.items()):
                child_metadata.update(identity)
                child_changes.append((child_path, child_metadata))

        if workflow_changed or child_changes:
            print(
                f"{'UPDATE' if args.apply else 'WOULD UPDATE'} "
                f"{workflow_dir.name}: {identity['target_key']} "
                f"({len(child_changes)} children)"
            )
        if not args.apply:
            continue
        if workflow_changed:
            _write(workflow_path, workflow)
            _write(metadata_path, metadata)
            _write(input_path, workflow_input)
            changed_workflows += 1
        for child_path, child_metadata in child_changes:
            _write(child_path, child_metadata)
            changed_children += 1

    if args.apply:
        print(f"Updated {changed_workflows} workflows and {changed_children} child jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
