#!/usr/bin/env python3
"""Apply adaptive global CPU sizing to preserved queued analysis jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from mn_ligand.core.resources import acquire_job_claim
from mn_ligand.runtime import (
    NATIVE_THREAD_ENVIRONMENT,
    adaptive_cpu_workers,
    runs_root,
)


ADAPTIVE_WORKFLOWS = {
    "native_md_geometry_interactions",
    "pandamap_interactions",
    "plip_interactions",
    "posebusters_validation",
}

def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _resize_command(value: object, workers: int) -> object:
    if not isinstance(value, list):
        return value
    if value and all(not isinstance(item, list) for item in value):
        command = list(value)
        for index, item in enumerate(command[:-1]):
            if str(item) == "--max-workers":
                command[index + 1] = str(workers)
        return command
    return [_resize_command(item, workers) for item in value]


def _cap_posebusters_threads(value: object) -> object:
    if not isinstance(value, list):
        return value
    if value and all(not isinstance(item, list) for item in value):
        command = [str(item) for item in value]
        if len(command) < 2 or Path(command[0]).name != "docker" or command[1] != "run":
            return value
        configured = {
            item.split("=", 1)[0]
            for index, item in enumerate(command)
            if index > 0 and command[index - 1] in {"-e", "--env"} and "=" in item
        }
        additions: list[str] = []
        for key, setting in NATIVE_THREAD_ENVIRONMENT.items():
            if key not in configured:
                additions.extend(["-e", f"{key}={setting}"])
        command[2:2] = additions
        return command
    return [_cap_posebusters_threads(item) for item in value]


def _posebusters_threads_are_capped(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if any(isinstance(item, list) for item in value):
        return all(_posebusters_threads_are_capped(item) for item in value)
    command = [str(item) for item in value]
    configured = {
        item.split("=", 1)[0]
        for index, item in enumerate(command)
        if index > 0 and command[index - 1] in {"-e", "--env"} and "=" in item
    }
    return set(NATIVE_THREAD_ENVIRONMENT).issubset(configured)


def _resize_payload(
    payload: dict[str, object],
    *,
    workers: int,
    work_items: int,
    cap_posebusters_threads: bool = False,
) -> dict[str, object]:
    updated = dict(payload)
    updated["max_workers"] = workers
    updated["cpu_worker_policy"] = "adaptive-global-limit"
    updated["cpu_work_items"] = work_items
    resources = updated.get("resources")
    if isinstance(resources, dict):
        updated["resources"] = {**resources, "cpu_threads": workers}
    for key in ("queued_command", "queued_commands", "argv", "commands"):
        if key in updated:
            updated[key] = _resize_command(updated[key], workers)
            if cap_posebusters_threads:
                updated[key] = _cap_posebusters_threads(updated[key])
    parameters = updated.get("parameters")
    if isinstance(parameters, dict):
        updated["parameters"] = {
            **parameters,
            "max_workers": workers,
            "cpu_worker_policy": "adaptive-global-limit",
            "cpu_work_items": work_items,
        }
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write updates; otherwise only report the proposed changes.",
    )
    args = parser.parse_args()
    changed = skipped_claimed = 0
    for metadata_path in runs_root(create=False).glob("**/metadata.json"):
        metadata = _read_json(metadata_path)
        workflow = str(metadata.get("workflow") or "")
        if (
            str(metadata.get("status") or "") != "queued"
            or workflow not in ADAPTIVE_WORKFLOWS
        ):
            continue
        work_items = max(
            1,
            int(
                metadata.get("compound_count")
                or metadata.get("pose_count")
                or 1
            ),
        )
        workers = adaptive_cpu_workers(work_items)
        current = int(metadata.get("max_workers") or 0)
        threads_capped = workflow != "posebusters_validation" or _posebusters_threads_are_capped(
            metadata.get("queued_command") or metadata.get("queued_commands")
        )
        if (
            current == workers
            and metadata.get("cpu_worker_policy")
            == "adaptive-global-limit"
            and int(metadata.get("cpu_work_items") or 0) == work_items
            and threads_capped
        ):
            continue
        if not args.apply:
            print(
                f"{metadata.get('job_code', metadata_path.parent.name)}\t"
                f"{workflow}\t{current or '?'} -> {workers}"
            )
            changed += 1
            continue
        claim = acquire_job_claim(
            metadata_path.parent,
            run_id=str(metadata.get("run_id") or metadata_path.parent.name),
            worker_id="adaptive-cpu-resize",
        )
        if claim is None:
            skipped_claimed += 1
            continue
        try:
            metadata = _read_json(metadata_path)
            if str(metadata.get("status") or "") != "queued":
                continue
            work_items = max(
                1,
                int(
                    metadata.get("compound_count")
                    or metadata.get("pose_count")
                    or 1
                ),
            )
            workers = adaptive_cpu_workers(work_items)
            _write_json(
                metadata_path,
                _resize_payload(
                    metadata,
                    workers=workers,
                    work_items=work_items,
                    cap_posebusters_threads=workflow == "posebusters_validation",
                ),
            )
            for name in ("input.json", "command.json"):
                path = metadata_path.parent / name
                if path.is_file():
                    _write_json(
                        path,
                        _resize_payload(
                            _read_json(path),
                            workers=workers,
                            work_items=work_items,
                            cap_posebusters_threads=workflow == "posebusters_validation",
                        ),
                    )
            changed += 1
        finally:
            claim.release()
    action = "updated" if args.apply else "would update"
    print(f"{action}: {changed}; skipped claimed/running: {skipped_claimed}")


if __name__ == "__main__":
    main()
