from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

from mn_ligand.core.artifacts import write_artifact_manifest
from mn_ligand.core.jobs import display_job_code, short_job_code


ANALYSIS_SET_TASK_GROUP = "campaign-comparison-collections"
ANALYSIS_SET_SCHEMA_VERSION = 2


def compound_smiles_signature(values: list[str]) -> str:
    """Return a stable chemical-identity signature for compatibility checks."""
    cleaned = {str(value).strip() for value in values if str(value).strip()}
    if not cleaned:
        return ""
    try:
        from rdkit import Chem

        canonical = {
            Chem.MolToSmiles(molecule, isomericSmiles=True)
            for value in cleaned
            for molecule in [Chem.MolFromSmiles(value)]
            if molecule is not None
        }
        if canonical:
            return "||".join(sorted(canonical))
    except (ImportError, RuntimeError, ValueError):
        pass
    return "||".join(sorted(cleaned))


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def list_analysis_sets(run_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    group_dir = run_root / ANALYSIS_SET_TASK_GROUP
    if not group_dir.is_dir():
        return records
    for run_dir in group_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        metadata = _read_json(run_dir / "metadata.json")
        selection = _read_json(run_dir / "selection.json")
        if not metadata or not selection:
            continue
        run_id = str(metadata.get("run_id") or run_dir.name)
        records.append(
            {
                "collection_id": run_id,
                "analysis_set_id": run_id,
                "job_code": display_job_code(
                    metadata.get("job_code"), run_id
                ),
                "name": str(metadata.get("name") or "Saved comparison"),
                "description": str(metadata.get("description") or ""),
                "dataset_run_id": str(selection.get("dataset_run_id") or ""),
                "dataset": str(selection.get("dataset") or ""),
                "target_run_ids": tuple(
                    str(value) for value in selection.get("target_run_ids") or []
                ),
                "launch_campaign_ids": tuple(
                    str(value)
                    for value in selection.get("launch_campaign_ids") or []
                ),
                "engines": tuple(
                    str(value) for value in selection.get("engines") or []
                ),
                "engine_run_ids": tuple(
                    str(value)
                    for value in selection.get("engine_run_ids") or []
                ),
                "rescoring_run_ids": tuple(
                    str(value)
                    for value in selection.get("rescoring_run_ids") or []
                ),
                "created_at": str(metadata.get("created_at") or ""),
                "selection": selection,
                "open": "./compound-campaign-comparison?"
                + urlencode(
                    {
                        "analysis_set_id": run_id,
                        **(
                            {
                                "campaign_purpose": str(
                                    selection.get("campaign_purpose") or ""
                                )
                            }
                            if selection.get("campaign_purpose")
                            else {}
                        ),
                    }
                ),
            }
        )
    return sorted(
        records,
        key=lambda row: str(row.get("created_at") or ""),
        reverse=True,
    )


def save_analysis_set(
    run_root: Path,
    *,
    name: str,
    description: str,
    selection: dict[str, object],
) -> dict[str, object]:
    run_id = str(uuid4())
    run_dir = run_root / ANALYSIS_SET_TASK_GROUP / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": ANALYSIS_SET_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "campaign_analysis_set",
        "workflow": "campaign_analysis_set",
        "operation": "analysis_set",
        "status": "completed",
        "name": str(name).strip() or "Saved comparison",
        "description": str(description).strip(),
        "parent_run_id": str(selection.get("dataset_run_id") or ""),
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {"success": True, "selection_file": "selection.json"},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_artifact_manifest(run_dir, [])
    return {
        "collection_id": run_id,
        "job_code": metadata["job_code"],
        "name": metadata["name"],
        "run_dir": run_dir,
    }
