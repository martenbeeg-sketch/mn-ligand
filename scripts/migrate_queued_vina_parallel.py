"""Migrate queued serial Vina campaign jobs to bounded ligand parallelism.

The migration is intentionally limited to jobs that are still queued.  It
acquires the same per-job claim used by workers before changing metadata, so a
job that starts concurrently is skipped instead of being modified in flight.
Run without ``--apply`` for a dry run.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mn_ligand.core.resources import acquire_job_claim
from mn_ligand.runtime import runs_root


SERIAL_PREFIX = (
    'for replicate in $(seq 1 "$DOCKING_REPLICATES"); do '
    'seed=$((DOCKING_SEED_START + replicate - 1)); '
    'result_dir=$(printf "results/replicate_%03d" "$replicate"); '
    'mkdir -p "$result_dir"; while read -r ligand; do '
    'compound_id=$(basename "$ligand" .pdbqt); '
)
SERIAL_SUFFIX = "done < ligand_index.txt; done"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def parallelize_vina_shell(shell: str, *, workers: int) -> str:
    if 'xargs -P "$DOCKING_CPU_WORKERS"' in shell:
        return shell
    start = shell.find(SERIAL_PREFIX)
    if start < 0:
        raise ValueError("serial Vina loop prefix was not found")
    end = shell.find(SERIAL_SUFFIX, start)
    if end < 0:
        raise ValueError("serial Vina loop suffix was not found")
    vina_command = shell[start + len(SERIAL_PREFIX) : end]
    if not vina_command.startswith("vina ") or " --out " not in vina_command:
        raise ValueError("serial loop does not contain the expected Vina command")
    vina_command = vina_command.replace(" --out ", " --cpu 1 --out ", 1)
    parallel = (
        "dock_vina_task() { "
        'ligand="$1"; replicate="$2"; seed="$3"; result_dir="$4"; '
        'compound_id=$(basename "$ligand" .pdbqt); '
        + vina_command
        + "}; export -f dock_vina_task; : > vina_tasks.tsv; "
        + 'for replicate in $(seq 1 "$DOCKING_REPLICATES"); do '
        + 'seed=$((DOCKING_SEED_START + replicate - 1)); '
        + 'result_dir=$(printf "results/replicate_%03d" "$replicate"); '
        + 'mkdir -p "$result_dir"; while read -r ligand; do '
        + 'printf "%s\\t%s\\t%s\\t%s\\n" "$ligand" "$replicate" "$seed" "$result_dir"; '
        + "done < ligand_index.txt; done > vina_tasks.tsv; "
        + 'xargs -P "$DOCKING_CPU_WORKERS" -n 4 bash -c '
        + "'dock_vina_task \"$1\" \"$2\" \"$3\" \"$4\"' _ "
        + "< vina_tasks.tsv"
    )
    updated = shell[:start] + parallel + shell[end + len(SERIAL_SUFFIX) :]
    seed_assignment = "DOCKING_SEED_START="
    assignment_at = updated.find(seed_assignment)
    if assignment_at < 0:
        raise ValueError("seed assignment was not found")
    assignment_end = updated.find(";", assignment_at)
    if assignment_end < 0:
        raise ValueError("seed assignment terminator was not found")
    updated = (
        updated[: assignment_end + 1]
        + f" DOCKING_CPU_WORKERS={workers};"
        + updated[assignment_end + 1 :]
    )
    return updated


def _update_command_record(path: Path, old_shell: str, new_shell: str, workers: int) -> None:
    if not path.is_file():
        return
    record = json.loads(path.read_text())
    argv = record.get("argv")
    if isinstance(argv, list) and argv and argv[-1] == old_shell:
        argv[-1] = new_shell
    commands = record.get("commands")
    if isinstance(commands, list):
        for command in commands:
            if isinstance(command, list) and command and command[-1] == old_shell:
                command[-1] = new_shell
    resources = record.get("resources")
    if isinstance(resources, dict):
        resources["cpu_threads"] = workers
    _write_json_atomic(path, record)


def migrate(campaign_id: str, *, workers: int, apply: bool) -> tuple[int, int]:
    candidates = 0
    migrated = 0
    worker_id = f"vina-queue-migration-{os.getpid()}"
    for metadata_path in runs_root().glob("docking/*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("launch_campaign_id") != campaign_id
            or metadata.get("engine") != "vina"
            or metadata.get("status") != "queued"
        ):
            continue
        command = metadata.get("queued_command")
        if not isinstance(command, list) or not command:
            continue
        old_shell = str(command[-1])
        try:
            new_shell = parallelize_vina_shell(old_shell, workers=workers)
        except ValueError as exc:
            print(f"SKIP {metadata_path.parent.name}: {exc}")
            continue
        if new_shell == old_shell:
            continue
        candidates += 1
        print(f"{'MIGRATE' if apply else 'WOULD MIGRATE'} {metadata_path.parent.name}")
        if not apply:
            continue
        claim = acquire_job_claim(
            metadata_path.parent,
            run_id=str(metadata.get("run_id") or metadata_path.parent.name),
            worker_id=worker_id,
            stale_after_seconds=300.0,
        )
        if claim is None:
            print(f"SKIP {metadata_path.parent.name}: worker claimed it concurrently")
            continue
        try:
            current = json.loads(metadata_path.read_text())
            if current.get("status") != "queued":
                print(f"SKIP {metadata_path.parent.name}: status is now {current.get('status')}")
                continue
            current_command = current.get("queued_command")
            if not isinstance(current_command, list) or not current_command:
                print(f"SKIP {metadata_path.parent.name}: queued command disappeared")
                continue
            current_shell = str(current_command[-1])
            new_shell = parallelize_vina_shell(current_shell, workers=workers)
            backup = metadata_path.with_name("metadata.pre-vina-parallel.json")
            if not backup.exists():
                _write_json_atomic(backup, current)
            current_command[-1] = new_shell
            current["queued_command"] = current_command
            current["cpu_workers"] = workers
            resources = current.setdefault("resources", {})
            resources["cpu_threads"] = workers
            current["updated_at"] = _utc_now_iso()
            current["queue_migration"] = {
                "kind": "parallel_vina_ligands",
                "workers": workers,
                "migrated_at": current["updated_at"],
            }
            _update_command_record(
                metadata_path.parent / "command.json",
                current_shell,
                new_shell,
                workers,
            )
            _write_json_atomic(metadata_path, current)
            migrated += 1
        finally:
            claim.release()
    return candidates, migrated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_id")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 128:
        parser.error("--workers must be between 1 and 128")
    candidates, migrated = migrate(
        args.campaign_id,
        workers=args.workers,
        apply=args.apply,
    )
    print(f"candidates={candidates} migrated={migrated}")


if __name__ == "__main__":
    main()
