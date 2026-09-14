from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from mn_ligand.runtime import runs_root


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def release(dependency_run_id: str) -> int:
    root = runs_root()
    statuses = {
        path.parent.name: _read(path).get("status")
        for path in root.glob("**/metadata.json")
    }
    released = 0
    for path in root.glob("**/metadata.json"):
        metadata = _read(path)
        dependencies = [str(value) for value in metadata.get("depends_on_run_ids") or ()]
        if dependency_run_id not in dependencies or metadata.get("status") != "blocked":
            continue
        if not dependencies or not all(statuses.get(value) == "completed" for value in dependencies):
            continue
        now = datetime.now(timezone.utc).isoformat()
        metadata["status"] = "queued"
        metadata["queued_at"] = now
        metadata["updated_at"] = now
        metadata.pop("error", None)
        metadata.pop("blocked_reason", None)
        metadata["released_after_msa_run_id"] = dependency_run_id
        _write(path, metadata)
        released += 1
    return released


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dependency_run_id")
    args = parser.parse_args()
    print(json.dumps({"released_jobs": release(args.dependency_run_id)}, indent=2))


if __name__ == "__main__":
    main()
