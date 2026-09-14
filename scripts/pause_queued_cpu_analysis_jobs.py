#!/usr/bin/env python3
"""Pause or resume queued interaction-analysis and PoseBusters jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "mn-ligand-workdir" / "workdir" / "runs"
PAUSE_MARKER = "docking_campaign_priority_20260805"
TARGET_WORKFLOWS = frozenset(
    {
        "native_md_geometry_interactions",
        "pandamap_interactions",
        "plip_interactions",
        "posebusters_validation",
    }
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pause", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes; without this flag, only show the planned count.",
    )
    args = parser.parse_args()

    candidates: list[Path] = []
    counts: dict[str, int] = {}
    for metadata_path in RUNS_DIR.glob("*/*/metadata.json"):
        metadata = _read_json(metadata_path)
        workflow = str(metadata.get("workflow") or "")
        if workflow not in TARGET_WORKFLOWS:
            continue
        status = str(metadata.get("status") or "")
        if args.pause:
            selected = status == "queued"
        else:
            selected = (
                status == "paused"
                and metadata.get("pause_marker") == PAUSE_MARKER
            )
        if not selected:
            continue
        candidates.append(metadata_path)
        tool = str(metadata.get("tool") or workflow)
        counts[tool] = counts.get(tool, 0) + 1

    print(
        json.dumps(
            {
                "mode": "pause" if args.pause else "resume",
                "apply": args.apply,
                "jobs": len(candidates),
                "by_tool": counts,
            },
            indent=2,
        )
    )
    if not args.apply:
        return

    changed = 0
    skipped = 0
    now = datetime.now(timezone.utc).isoformat()
    for metadata_path in candidates:
        metadata = _read_json(metadata_path)
        if args.pause:
            if str(metadata.get("status") or "") != "queued":
                skipped += 1
                continue
            metadata.update(
                {
                    "status": "paused",
                    "paused_at": now,
                    "paused_by": "codex",
                    "pause_marker": PAUSE_MARKER,
                    "pause_reason": "Prioritize active docking/cofolding campaign",
                    "updated_at": now,
                }
            )
        else:
            if not (
                str(metadata.get("status") or "") == "paused"
                and metadata.get("pause_marker") == PAUSE_MARKER
            ):
                skipped += 1
                continue
            metadata.update(
                {
                    "status": "queued",
                    "queued_at": now,
                    "resumed_at": now,
                    "updated_at": now,
                }
            )
            metadata.pop("pause_marker", None)
            metadata.pop("pause_reason", None)
        _write_json(metadata_path, metadata)
        changed += 1

    print(json.dumps({"changed": changed, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
