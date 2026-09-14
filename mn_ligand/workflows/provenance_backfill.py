from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mn_ligand.core.jobs import iter_job_records
from mn_ligand.core.provenance import (
    inherited_target_metadata,
    modification_history,
    provenance_origin_label,
)
from mn_ligand.runtime import runs_root


DERIVED_TARGET_GROUPS = ("structure-jobs", "target-trimming", "terminal-repair")


def _protein_atom_signature(pdb_data: str) -> str:
    atom_records = "\n".join(
        line for line in pdb_data.splitlines() if line.startswith("ATOM  ")
    )
    return hashlib.sha256(atom_records.encode()).hexdigest() if atom_records else ""


def _legacy_preparation_records(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Index old prepared-structures runs by their exact prepared protein atoms."""
    records: dict[str, list[dict[str, Any]]] = {}
    group = root / "prepared-structures"
    if not group.is_dir():
        return records
    for run_dir in group.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            input_payload = json.loads((run_dir / "input.json").read_text())
            result_payload = json.loads((run_dir / "result.json").read_text())
        except (OSError, ValueError, TypeError):
            continue
        prepared_pdb = str(result_payload.get("prepared_pdb_data") or "")
        signature = _protein_atom_signature(prepared_pdb)
        if (
            not signature
            or not input_payload.get("clean_protein")
            or not result_payload.get("protein_cleaned")
        ):
            continue
        records.setdefault(signature, []).append({
            "schema_version": 1,
            "source_task_group": "prepared-structures",
            "source_run_id": run_dir.name,
            "match": "exact prepared-protein ATOM signature",
            "pdb_id": str(input_payload.get("pdb_id") or ""),
            "protocol": "legacy Ligand-X staged PDBFixer cleaning",
            "clean_protein": True,
            "protein_cleaned": True,
            "ph": 7.4,
            "operations": [
                "remove water",
                "repair missing heavy atoms",
                "add hydrogens",
            ],
            "components": result_payload.get("components") or {},
            "modified_residue_mapping": (
                result_payload.get("modified_residue_mapping") or {}
            ),
            "energy_minimized": False,
            "_result_mtime_epoch": (run_dir / "result.json").stat().st_mtime,
        })
    return records


def _legacy_preparation_evidence(
    job, records: dict[str, list[dict[str, Any]]]
):
    if (
        job.task_group != "structure-jobs"
        or str(job.metadata.get("source") or "").lower() != "pdb"
        or job.metadata.get("legacy_preparation_evidence")
    ):
        return None
    for protein_path in job.run_dir.glob("*_protein_refined.pdb"):
        try:
            signature = _protein_atom_signature(
                protein_path.read_text(errors="replace")
            )
        except OSError:
            continue
        candidates = records.get(signature) or []
        if not candidates:
            continue
        pdb_id = str(job.metadata.get("pdb_id") or "")
        candidates = [
            evidence
            for evidence in candidates
            if not evidence.get("pdb_id")
            or pdb_id.upper() == str(evidence["pdb_id"]).upper()
        ]
        if not candidates:
            continue
        try:
            created_epoch = datetime.fromisoformat(job.created_at).timestamp()
        except (TypeError, ValueError):
            created_epoch = protein_path.stat().st_mtime
        evidence = min(
            candidates,
            key=lambda item: abs(
                float(item.get("_result_mtime_epoch") or 0.0) - created_epoch
            ),
        )
        return {
            key: value
            for key, value in evidence.items()
            if not key.startswith("_")
        }
    return None


def backfill_derived_target_provenance(*, write: bool = False) -> dict[str, Any]:
    """Backfill identity and modification history without touching artifacts."""
    jobs = iter_job_records(runs_root())
    original_jobs_by_id = {job.run_id: job for job in jobs}
    legacy_records = _legacy_preparation_records(runs_root())
    evidence_by_id = {
        job.run_id: evidence
        for job in jobs
        if (evidence := _legacy_preparation_evidence(job, legacy_records)) is not None
    }
    effective_jobs = [
        replace(
            job,
            metadata={
                **job.metadata,
                **(
                    {"legacy_preparation_evidence": evidence_by_id[job.run_id]}
                    if job.run_id in evidence_by_id
                    else {}
                ),
            },
        )
        for job in jobs
    ]
    jobs_by_id = {job.run_id: job for job in effective_jobs}
    changed = []
    for job in effective_jobs:
        if job.task_group not in DERIVED_TARGET_GROUPS:
            continue
        patch = {
            **(
                {"legacy_preparation_evidence": evidence_by_id[job.run_id]}
                if job.run_id in evidence_by_id
                else {}
            ),
            **inherited_target_metadata(job, jobs_by_id),
            "modification_history": modification_history(job, jobs_by_id),
            "provenance_origin": provenance_origin_label(job, jobs_by_id),
        }
        differences = {
            key: value
            for key, value in patch.items()
            if value not in (None, "", [], {})
            and original_jobs_by_id[job.run_id].metadata.get(key) != value
        }
        if not differences:
            continue
        changed.append(
            {
                "task_group": job.task_group,
                "run_id": job.run_id,
                "status": job.status,
                "fields": sorted(differences),
            }
        )
        if write:
            metadata = {**job.metadata, **differences}
            metadata["provenance_backfilled_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            temporary = job.run_dir / ".metadata.json.provenance.tmp"
            temporary.write_text(json.dumps(metadata, indent=2) + "\n")
            temporary.replace(job.run_dir / "metadata.json")
    return {
        "scanned": sum(job.task_group in DERIVED_TARGET_GROUPS for job in jobs),
        "updated": len(changed),
        "written": write,
        "jobs": changed,
    }
