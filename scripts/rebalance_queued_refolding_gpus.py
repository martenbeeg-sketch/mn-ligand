"""Reassign a bounded set of queued refolding jobs to another GPU.

Only jobs that are still queued are changed.  Each candidate is protected by
the same atomic claim used by workers, so a concurrently started job is skipped.
Run without ``--apply`` to preview the exact immutable run IDs.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mn_ligand.core.resources import acquire_job_claim, select_gpu_in_command
from mn_ligand.runtime import runs_root


DEFAULT_TOOLS = ("AlphaFold 3", "Boltz-2", "Nesso-1")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _backup(path: Path, suffix: str, payload: dict[str, Any]) -> None:
    backup = path.with_name(f"{path.stem}.{suffix}{path.suffix}")
    if not backup.exists():
        _write_json_atomic(backup, payload)


def _command_on_gpu(command: Any, gpu_id: int) -> Any:
    if not isinstance(command, list):
        return command
    return select_gpu_in_command([str(value) for value in command], gpu_id)


def _candidate_paths(
    campaign_id: str,
    *,
    source_gpu: int,
    count_per_tool: int,
) -> list[Path]:
    by_tool: dict[str, list[tuple[str, Path]]] = {tool: [] for tool in DEFAULT_TOOLS}
    for metadata_path in runs_root().glob("refolding/*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError, TypeError):
            continue
        tool = str(metadata.get("tool") or "")
        resources = metadata.get("resources")
        configured = resources.get("gpu_ids") if isinstance(resources, dict) else None
        if (
            metadata.get("launch_campaign_id") != campaign_id
            or metadata.get("status") != "queued"
            or tool not in by_tool
            or configured != [source_gpu]
        ):
            continue
        queue_time = str(metadata.get("queued_at") or metadata.get("created_at") or "")
        by_tool[tool].append((queue_time, metadata_path))

    selected: list[Path] = []
    for tool in DEFAULT_TOOLS:
        candidates = sorted(by_tool[tool], key=lambda item: (item[0], item[1].parent.name))
        if len(candidates) < count_per_tool:
            raise RuntimeError(
                f"Requested {count_per_tool} queued {tool} jobs but found {len(candidates)}"
            )
        selected.extend(path for _, path in candidates[:count_per_tool])
    return selected


def _migrate_one(
    metadata_path: Path,
    *,
    campaign_id: str,
    source_gpu: int,
    target_gpu: int,
    worker_id: str,
) -> bool:
    claim = acquire_job_claim(
        metadata_path.parent,
        run_id=metadata_path.parent.name,
        worker_id=worker_id,
        stale_after_seconds=300.0,
    )
    if claim is None:
        print(f"SKIP {metadata_path.parent.name}: claimed by a worker")
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        resources = metadata.get("resources")
        configured = resources.get("gpu_ids") if isinstance(resources, dict) else None
        if (
            metadata.get("status") != "queued"
            or metadata.get("launch_campaign_id") != campaign_id
            or configured != [source_gpu]
        ):
            print(f"SKIP {metadata_path.parent.name}: queue state or GPU scope changed")
            return False

        input_path = metadata_path.parent / "input.json"
        command_path = metadata_path.parent / "command.json"
        input_record = json.loads(input_path.read_text())
        command_record = json.loads(command_path.read_text())

        _backup(metadata_path, "pre-gpu-rebalance", metadata)
        _backup(input_path, "pre-gpu-rebalance", input_record)
        _backup(command_path, "pre-gpu-rebalance", command_record)

        migrated_at = _utc_now_iso()
        resources["gpu_ids"] = [target_gpu]
        metadata["gpu_device"] = str(target_gpu)
        metadata["queued_command"] = _command_on_gpu(metadata.get("queued_command"), target_gpu)
        metadata.pop("admission", None)
        metadata["updated_at"] = migrated_at
        metadata["queue_migration"] = {
            "kind": "gpu_rebalance",
            "campaign_id": campaign_id,
            "source_gpu": source_gpu,
            "target_gpu": target_gpu,
            "migrated_at": migrated_at,
        }

        parameters = input_record.get("parameters")
        if isinstance(parameters, dict):
            parameters["gpu_device"] = str(target_gpu)

        command_record["selected_gpu_ids"] = [target_gpu]
        command_resources = command_record.get("resources")
        if isinstance(command_resources, dict):
            command_resources["gpu_ids"] = [target_gpu]
        commands = command_record.get("commands")
        if isinstance(commands, list):
            command_record["commands"] = [
                _command_on_gpu(command, target_gpu) for command in commands
            ]
        argv = command_record.get("argv")
        if isinstance(argv, list):
            command_record["argv"] = _command_on_gpu(argv, target_gpu)
        command_record["queue_migration"] = dict(metadata["queue_migration"])

        _write_json_atomic(input_path, input_record)
        _write_json_atomic(command_path, command_record)
        _write_json_atomic(metadata_path, metadata)
        print(
            f"MIGRATED {metadata_path.parent.name} "
            f"{metadata.get('tool')} GPU {source_gpu} -> GPU {target_gpu}"
        )
        return True
    finally:
        claim.release()


def rebalance(
    campaign_id: str,
    *,
    source_gpu: int,
    target_gpu: int,
    count_per_tool: int,
    apply: bool,
) -> tuple[int, int]:
    selected = _candidate_paths(
        campaign_id,
        source_gpu=source_gpu,
        count_per_tool=count_per_tool,
    )
    if not apply:
        for metadata_path in selected:
            metadata = json.loads(metadata_path.read_text())
            print(f"WOULD MIGRATE {metadata_path.parent.name} {metadata.get('tool')}")
        return len(selected), 0

    worker_id = f"gpu-rebalance-{os.getpid()}"
    migrated = sum(
        _migrate_one(
            metadata_path,
            campaign_id=campaign_id,
            source_gpu=source_gpu,
            target_gpu=target_gpu,
            worker_id=worker_id,
        )
        for metadata_path in selected
    )
    return len(selected), migrated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_id")
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=1)
    parser.add_argument("--count-per-tool", type=int, default=4)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.source_gpu < 0 or args.target_gpu < 0:
        parser.error("GPU IDs must be non-negative")
    if args.source_gpu == args.target_gpu:
        parser.error("source and target GPU must differ")
    if args.count_per_tool < 1:
        parser.error("count-per-tool must be positive")
    selected, migrated = rebalance(
        args.campaign_id,
        source_gpu=args.source_gpu,
        target_gpu=args.target_gpu,
        count_per_tool=args.count_per_tool,
        apply=args.apply,
    )
    print(f"selected={selected} migrated={migrated}")


if __name__ == "__main__":
    main()
