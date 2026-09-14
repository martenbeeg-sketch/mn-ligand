from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable
from uuid import uuid4

from rdkit import Chem

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import (
    JOB_SCHEMA_VERSION,
    JobRecord,
    display_job_code,
    iter_job_records,
    short_job_code,
)
from mn_ligand.runtime import runs_root


MOLECULE_DESIGN_SELECTION_TASK_GROUP = "molecule-design-selections"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _safe_id(value: object, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (text or fallback)[:160]


def _plain_value(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, Path):
        return str(value)
    return value


def _as_bool(value: object) -> bool:
    return value is True or str(value).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _latest_qualifications(
    jobs: Iterable[JobRecord],
) -> dict[str, JobRecord]:
    latest: dict[str, JobRecord] = {}
    for job in jobs:
        if job.workflow != "molecule_qualification" or not job.parent_run_id:
            continue
        previous = latest.get(job.parent_run_id)
        key = (
            int(job.metadata.get("qualification_policy_version") or 0),
            str(job.created_at or ""),
        )
        previous_key = (
            int(previous.metadata.get("qualification_policy_version") or 0),
            str(previous.created_at or ""),
        ) if previous is not None else (-1, "")
        if key > previous_key:
            latest[job.parent_run_id] = job
    return latest


def target_design_campaigns(
    target_run_id: str,
    *,
    jobs: Iterable[JobRecord] | None = None,
) -> list[JobRecord]:
    records = list(jobs or iter_job_records(runs_root()))
    return sorted(
        (
            job
            for job in records
            if job.workflow == "molecule_generation_campaign"
            and job.status == "completed"
            and str(
                job.metadata.get("target_run_id")
                or job.parent_run_id
                or ""
            )
            == str(target_run_id)
        ),
        key=lambda job: str(job.created_at or ""),
    )


def target_design_compounds(
    target_run_id: str,
    *,
    jobs: Iterable[JobRecord] | None = None,
) -> list[dict[str, Any]]:
    records = list(jobs or iter_job_records(runs_root()))
    by_id = {job.run_id: job for job in records}
    campaigns = target_design_campaigns(target_run_id, jobs=records)
    campaign_ids = {job.run_id for job in campaigns}
    generation_jobs = [
        job
        for job in records
        if job.workflow == "molecule_generation"
        and job.parent_run_id in campaign_ids
    ]
    latest_qualification = _latest_qualifications(records)
    rows: list[dict[str, Any]] = []
    for generation in generation_jobs:
        qualification = latest_qualification.get(generation.run_id)
        if qualification is None or qualification.status != "completed":
            continue
        campaign = by_id.get(str(generation.parent_run_id or ""))
        table_path = qualification.run_dir / "qualified" / "qualification.csv"
        if not table_path.is_file():
            continue
        with table_path.open(newline="") as handle:
            qualification_rows = list(csv.DictReader(handle))
        for row in qualification_rows:
            compound_id = str(row.get("compound_id") or "")
            candidate_path = (
                qualification.run_dir
                / "qualified"
                / "candidates"
                / f"{_safe_id(compound_id, 'compound')}.sdf"
            )
            status = str(row.get("qualification_status") or "").strip()
            if not status:
                status = (
                    "qualified"
                    if str(row.get("qualified_for_docking") or "").lower()
                    == "true"
                    else "rejected"
                )
            canonical_smiles = str(
                row.get("canonical_isomeric_smiles") or ""
            )
            rows.append(
                {
                    "row_id": f"{qualification.run_id}:{compound_id}",
                    "target_run_id": target_run_id,
                    "campaign_run_id": campaign.run_id if campaign else "",
                    "campaign": (
                        str(campaign.metadata.get("name") or "")
                        if campaign
                        else ""
                    ),
                    "campaign_job": (
                        display_job_code(
                            campaign.metadata.get("job_code"),
                            campaign.run_id,
                        )
                        if campaign
                        else ""
                    ),
                    "generation_run_id": generation.run_id,
                    "generation_job": display_job_code(
                        generation.metadata.get("job_code"),
                        generation.run_id,
                    ),
                    "qualification_run_id": qualification.run_id,
                    "qualification_job": display_job_code(
                        qualification.metadata.get("job_code"),
                        qualification.run_id,
                    ),
                    "qualification_policy_version": int(
                        qualification.metadata.get(
                            "qualification_policy_version"
                        )
                        or 0
                    ),
                    "engine": str(
                        qualification.metadata.get("source_engine")
                        or generation.tool
                        or ""
                    ),
                    "engine_id": str(
                        qualification.metadata.get("source_engine_id")
                        or generation.metadata.get("engine_id")
                        or ""
                    ),
                    "compound_id": compound_id,
                    "canonical_isomeric_smiles": canonical_smiles,
                    "qualification_status": status,
                    "accepted_for_docking": (
                        str(row.get("qualified_for_docking") or "").lower()
                        == "true"
                    ),
                    "strict_posebusters_pass": (
                        str(row.get("posebusters_pass") or "").lower()
                        == "true"
                    ),
                    "review_warnings": str(
                        row.get("review_warnings") or ""
                    ),
                    "chemical_failures": str(
                        row.get("chemical_failures") or ""
                    ),
                    "geometry_failures": str(
                        row.get("posebusters_hard_failures")
                        or row.get("posebusters_failed_checks")
                        or ""
                    ),
                    "qed": row.get("qed", ""),
                    "molecular_weight": row.get("molecular_weight", ""),
                    "sa_score": row.get("sa_score", ""),
                    "logp": row.get("logp", ""),
                    "hbond_donors": row.get("hbond_donors", ""),
                    "hbond_acceptors": row.get("hbond_acceptors", ""),
                    "rotatable_bonds": row.get("rotatable_bonds", ""),
                    "ring_count": row.get("ring_count", ""),
                    "heavy_atom_count": row.get("heavy_atom_count", ""),
                    "formal_charge": row.get("formal_charge", ""),
                    "force_field": row.get("force_field", ""),
                    "selected_energy": row.get("selected_energy", ""),
                    "candidate_available": candidate_path.is_file(),
                    "candidate_relative_path": (
                        candidate_path.relative_to(
                            qualification.run_dir
                        ).as_posix()
                        if candidate_path.is_file()
                        else ""
                    ),
                }
            )
    return rows


def create_molecule_design_selection(
    *,
    target_run_id: str,
    selected_rows: list[dict[str, Any]],
    name: str = "",
) -> JobRecord:
    if not target_run_id:
        raise ValueError("A prepared target is required")
    if not selected_rows:
        raise ValueError("Select at least one accepted compound")
    normalized_selected_rows = [
        {
            str(key): _plain_value(value)
            for key, value in row.items()
        }
        for row in selected_rows
    ]
    if any(
        not _as_bool(row.get("accepted_for_docking"))
        for row in normalized_selected_rows
    ):
        raise ValueError(
            "Only compounds accepted by the current qualification policy can "
            "enter a docking/cofolding dataset"
        )

    run_id = str(uuid4())
    run_dir = (
        runs_root() / MOLECULE_DESIGN_SELECTION_TASK_GROUP / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    selected_sdf = run_dir / "selected_compounds.sdf"
    selected_csv = run_dir / "selected_compounds.csv"
    selection_json = run_dir / "selection.json"

    writer = Chem.SDWriter(str(selected_sdf))
    output_rows: list[dict[str, Any]] = []
    seen_smiles: set[str] = set()
    used_ids: set[str] = set()
    duplicate_count = 0
    try:
        for index, source in enumerate(normalized_selected_rows, start=1):
            qualification_run_id = str(
                source.get("qualification_run_id") or ""
            )
            qualification_dir = (
                runs_root()
                / "molecule-qualification"
                / qualification_run_id
            )
            relative_path = str(
                source.get("candidate_relative_path") or ""
            )
            candidate_path = (
                qualification_dir / relative_path
            ).resolve()
            try:
                candidate_path.relative_to(qualification_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    "Candidate path escapes its qualification job"
                ) from exc
            if not candidate_path.is_file():
                raise FileNotFoundError(candidate_path)
            molecules = [
                molecule
                for molecule in Chem.SDMolSupplier(
                    str(candidate_path),
                    removeHs=False,
                    sanitize=True,
                )
                if molecule is not None
            ]
            if not molecules:
                raise ValueError(
                    f"No readable molecule for {source.get('compound_id')}"
                )
            molecule = molecules[0]
            smiles = str(
                source.get("canonical_isomeric_smiles") or ""
            ).strip()
            parsed = Chem.MolFromSmiles(smiles)
            if parsed is None:
                raise ValueError(
                    f"No valid canonical SMILES for {source.get('compound_id')}"
                )
            canonical = Chem.MolToSmiles(
                parsed,
                canonical=True,
                isomericSmiles=True,
            )
            if canonical in seen_smiles:
                duplicate_count += 1
                continue
            seen_smiles.add(canonical)
            base_id = _safe_id(
                source.get("compound_id"),
                f"design-{index:07d}",
            )
            compound_id = base_id
            suffix = 2
            while compound_id in used_ids:
                compound_id = f"{base_id}-{suffix}"
                suffix += 1
            used_ids.add(compound_id)
            output = {
                **dict(source),
                "compound_id": compound_id,
                "canonical_isomeric_smiles": canonical,
                "source_compound_id": str(
                    source.get("compound_id") or ""
                ),
            }
            output_rows.append(output)
            molecule.SetProp("_Name", compound_id)
            for key, value in output.items():
                if value not in ("", None):
                    molecule.SetProp(str(key), str(value))
            writer.write(molecule)
    finally:
        writer.close()
    if not output_rows:
        raise ValueError(
            "The selected rows contained no unique readable compounds"
        )

    fieldnames = list(
        dict.fromkeys(
            key
            for row in output_rows
            for key in row
        )
    )
    with selected_csv.open("w", newline="") as handle:
        table_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        table_writer.writeheader()
        table_writer.writerows(output_rows)
    payload = {
        "schema_version": 1,
        "name": str(name).strip() or "Molecule design selection",
        "target_run_id": target_run_id,
        "selected_compound_count": len(output_rows),
        "duplicate_smiles_excluded": duplicate_count,
        "source_qualification_run_ids": sorted(
            {
                str(row.get("qualification_run_id") or "")
                for row in output_rows
            }
        ),
        "compounds": output_rows,
    }
    _write_json(selection_json, payload)
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "molecule_design_selection",
        "workflow": "molecule_design_selection",
        "operation": "promote_filtered_design_compounds",
        "status": "completed",
        "name": payload["name"],
        "parent_run_id": target_run_id,
        "target_run_id": target_run_id,
        "compound_count": len(output_rows),
        "duplicate_smiles_excluded": duplicate_count,
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "target_run_id": target_run_id,
            "selected_rows": normalized_selected_rows,
        },
    )
    _write_json(
        run_dir / "result.json",
        {
            "success": True,
            "compound_count": len(output_rows),
            "duplicate_smiles_excluded": duplicate_count,
        },
    )
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(
                run_dir,
                selected_sdf,
                "compound_set",
                role="filtered_molecule_design_handoff",
                metadata={
                    "compound_count": len(output_rows),
                    "target_run_id": target_run_id,
                    "geometry_source": "qualified_rdkit_etkdgv3",
                },
            ),
            ArtifactRef.from_path(
                run_dir,
                selected_csv,
                "compound_selection_table",
                role="selected_design_compound_provenance",
            ),
            ArtifactRef.from_path(
                run_dir,
                selection_json,
                "molecule_design_selection",
                role="immutable_filter_and_selection",
            ),
        ],
    )
    return JobRecord.load(
        run_dir,
        task_group=MOLECULE_DESIGN_SELECTION_TASK_GROUP,
    )
