#!/usr/bin/env python3
"""Atomically publish a staged target repair with backup and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _digest(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _atomic_json(path: Path, payload: object) -> None:
    staged = path.with_name(f".{path.name}.target-repair.tmp")
    staged.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(staged, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    targets = [
        item
        for item in manifest.get("artifacts", [])
        if item.get("artifact_type") in {"prepared_target", "prepared_receptor"}
        and str(item.get("path", "")).lower().endswith(".pdb")
    ]
    if len(targets) != 1:
        raise RuntimeError(f"Expected one prepared-target artifact, found {len(targets)}")

    target_record = targets[0]
    target_path = run_dir / target_record["path"]
    staged_path = target_path.parent / ".target_repair_staged.pdb"
    report_path = target_path.parent / ".target_repair_report.json"
    report = json.loads(report_path.read_text())
    if not report.get("target_validation", {}).get("valid"):
        raise RuntimeError("Refusing to publish a target that failed validation")

    old_sha, old_size = _digest(target_path)
    new_sha, new_size = _digest(staged_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = target_path.with_name(f"{target_path.name}.pre-repair-{timestamp}.bak")
    shutil.copy2(target_path, backup_path)
    os.replace(staged_path, target_path)

    final_report_path = target_path.with_name(
        f"{target_path.stem}.target-repair-{timestamp}.json"
    )
    os.replace(report_path, final_report_path)
    report_sha, report_size = _digest(final_report_path)
    relative_backup = str(backup_path.relative_to(run_dir))
    relative_report = str(final_report_path.relative_to(run_dir))

    target_record["sha256"] = new_sha
    target_record["size_bytes"] = new_size
    target_record.setdefault("metadata", {})["target_repair"] = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "old_sha256": old_sha,
        "old_size_bytes": old_size,
        "backup_path": relative_backup,
        "report_path": relative_report,
        "engine": report.get("refinement", {}).get("engine"),
        "platform": report.get("refinement", {}).get("platform"),
        "validation": report.get("target_validation"),
        "displacement": report.get("refinement", {}).get("displacement"),
    }
    manifest.setdefault("artifacts", []).append(
        {
            "kind": "run_artifact",
            "artifact_id": f"artifact-{report_sha[:16]}",
            "run_id": str(manifest.get("run_id") or run_dir.name),
            "artifact_type": "target_validation_report",
            "path": relative_report,
            "role": "target_repair_audit",
            "label": final_report_path.name,
            "media_type": "application/json",
            "size_bytes": report_size,
            "sha256": report_sha,
            "metadata": {"repaired_artifact_id": target_record.get("artifact_id")},
        }
    )
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, manifest)

    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.setdefault("target_repair_history", []).append(
        target_record["metadata"]["target_repair"]
    )
    metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(metadata_path, metadata)

    print(
        json.dumps(
            {
                "run_id": run_dir.name,
                "target": str(target_path),
                "backup": str(backup_path),
                "report": str(final_report_path),
                "old_sha256": old_sha,
                "new_sha256": new_sha,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
