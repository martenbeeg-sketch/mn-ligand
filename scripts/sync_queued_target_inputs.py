#!/usr/bin/env python3
"""Refresh queued campaign receptor copies from their current target artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from mn_ligand.core.jobs import iter_job_records
from mn_ligand.runtime import runs_root


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.target-sync.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_id")
    parser.add_argument(
        "--statuses",
        default="queued",
        help="Comma-separated mutable states; active jobs are always refused.",
    )
    args = parser.parse_args()
    allowed_statuses = {
        value.strip() for value in args.statuses.split(",") if value.strip()
    }
    if allowed_statuses & {"running", "preparing"}:
        raise ValueError("Active job inputs cannot be synchronized")
    root = runs_root()
    records = iter_job_records(root)
    run_dirs = {job.run_id: job.run_dir for job in records}
    synced = 0
    skipped = 0
    for job in records:
        if str(job.metadata.get("launch_campaign_id") or "") != args.campaign_id:
            continue
        # Never alter input beneath an active process.
        current_metadata = json.loads((job.run_dir / "metadata.json").read_text())
        if current_metadata.get("status") not in allowed_statuses:
            continue
        input_path = job.run_dir / "input.json"
        receptor_path = job.run_dir / "input" / "receptor.pdb"
        if not input_path.is_file() or not receptor_path.is_file():
            continue
        payload = json.loads(input_path.read_text())
        target = payload.get("target_artifact") or payload.get("target") or {}
        target_run_dir = run_dirs.get(str(target.get("run_id") or ""))
        if target_run_dir is None:
            skipped += 1
            continue
        manifest_path = target_run_dir / "artifacts.json"
        if not manifest_path.is_file():
            skipped += 1
            continue
        artifacts = json.loads(manifest_path.read_text()).get("artifacts", [])
        source_record = next(
            (
                item
                for item in artifacts
                if item.get("artifact_id") == target.get("artifact_id")
                and str(item.get("path", "")).lower().endswith(".pdb")
            ),
            None,
        ) or next(
            (
                item
                for item in artifacts
                if item.get("artifact_type") in {"prepared_target", "prepared_receptor"}
                and str(item.get("path", "")).lower().endswith(".pdb")
            ),
            None,
        )
        if source_record is None:
            skipped += 1
            continue
        source_path = target_run_dir / source_record["path"]
        old_sha = _sha(receptor_path)
        new_sha = _sha(source_path)
        if old_sha == new_sha:
            continue
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = receptor_path.with_name(
            f"{receptor_path.name}.pre-target-sync-{timestamp}.bak"
        )
        shutil.copy2(receptor_path, backup)
        staged = receptor_path.with_name(f".{receptor_path.name}.target-sync.tmp")
        shutil.copy2(source_path, staged)
        os.replace(staged, receptor_path)
        payload["target_artifact"] = source_record
        _write_json(input_path, payload)
        current_metadata.setdefault("target_input_sync_history", []).append(
            {
                "synced_at": datetime.now(timezone.utc).isoformat(),
                "target_run_id": target_run_dir.name,
                "old_sha256": old_sha,
                "new_sha256": new_sha,
                "backup_path": str(backup.relative_to(job.run_dir)),
            }
        )
        current_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(job.run_dir / "metadata.json", current_metadata)
        synced += 1
        print(job.run_id, old_sha[:12], "->", new_sha[:12])
    print(json.dumps({"synced": synced, "skipped": skipped}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
