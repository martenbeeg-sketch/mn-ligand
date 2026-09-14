from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rdkit import Chem

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.protein_preparation import (
    create_protein_import_job,
    prepared_target,
    run_protein_cleaning_job,
)


TASK_GROUP = "structure-jobs"
POLYMER_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _protein_only(pdb_data: str) -> str:
    lines = [
        line
        for line in pdb_data.splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
        and line[17:20].strip().upper() in POLYMER_RESIDUES
    ]
    if not lines:
        raise ValueError("Predicted complex does not contain protein ATOM records")
    return "\n".join((*lines, "TER", "END", ""))


def _ligand_block(pdb_data: str) -> str:
    excluded = {"HOH", "WAT", "DOD"}
    lines = [
        "HETATM" + line[6:]
        for line in pdb_data.splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
        and line[17:20].strip().upper() not in POLYMER_RESIDUES | excluded
    ]
    return "\n".join((*lines, "END", "")) if lines else ""


def _normalize_complex_records(pdb_data: str) -> str:
    """Make predicted polymer/non-polymer record types explicit before repair."""
    excluded = {"HOH", "WAT", "DOD"}
    lines: list[str] = []
    for line in pdb_data.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 20:
            if not line.startswith("END"):
                lines.append(line)
            continue
        residue = line[17:20].strip().upper()
        if residue in POLYMER_RESIDUES:
            lines.append("ATOM  " + line[6:])
        elif residue not in excluded:
            lines.append("HETATM" + line[6:])
    return "\n".join((*lines, "END", ""))


def promote_predicted_complex(
    *,
    source_job: JobRecord,
    source_artifact: ArtifactRef,
    source_path: Path,
    clean_and_repair: bool = True,
) -> JobRecord:
    run_id = str(uuid4())
    run_dir = runs_root() / TASK_GROUP / run_id
    input_dir = run_dir / "input"
    artifact_dir = run_dir / "artifacts"
    input_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir.mkdir()
    started = _utc_now_iso()
    initial_metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "structure",
        "workflow": "predicted_complex_promotion",
        "operation": "preparation",
        "source": str(source_job.tool or source_job.metadata.get("engine") or "prediction"),
        "tool": str(source_job.tool or source_job.metadata.get("engine") or "prediction"),
        "status": "preparing",
        "parent_run_id": source_job.run_id,
        "source_prediction_run_id": source_job.run_id,
        "source_prediction_artifact_id": source_artifact.artifact_id,
        "created_at": started,
        "updated_at": started,
    }
    (run_dir / "metadata.json").write_text(json.dumps(initial_metadata, indent=2) + "\n")
    (run_dir / "input.json").write_text(
        json.dumps({"predicted_complex": source_artifact.to_dict()}, indent=2) + "\n"
    )
    write_artifact_manifest(run_dir, [])
    command: list[str] = []

    def fail(error: str) -> JobRecord:
        completed = _utc_now_iso()
        failed_metadata = {
            **initial_metadata,
            "status": "failed",
            "error": error,
            "updated_at": completed,
            "completed_at": completed,
        }
        (run_dir / "metadata.json").write_text(json.dumps(failed_metadata, indent=2) + "\n")
        (run_dir / "result.json").write_text(
            json.dumps({"success": False, "error": error}, indent=2) + "\n"
        )
        (run_dir / "command.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "mode": "python" if command and command[0] == "gemmi" else "internal",
                    "commands": [command] if command else [],
                },
                indent=2,
            )
            + "\n"
        )
        return JobRecord.load(run_dir, task_group=TASK_GROUP)

    suffix = source_path.suffix.lower() or ".cif"
    source_local = input_dir / f"predicted_complex{suffix}"
    shutil.copy2(source_path, source_local)
    candidate = source_artifact.role or source_path.stem
    safe_candidate = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in candidate
    ).strip("_") or "candidate"
    engine = str(source_job.tool or source_job.metadata.get("engine") or "prediction")
    safe_engine = "".join(character.lower() if character.isalnum() else "_" for character in engine)
    prefix = f"{safe_engine.strip('_')}_{safe_candidate}"
    complex_path = artifact_dir / f"{prefix}_complex_refined.pdb"
    if suffix in {".pdb", ".ent"}:
        shutil.copy2(source_local, complex_path)
        command = ["internal-copy", source_local.relative_to(run_dir).as_posix(), complex_path.relative_to(run_dir).as_posix()]
    else:
        command = [
            "gemmi",
            "convert",
            source_local.relative_to(run_dir).as_posix(),
            complex_path.relative_to(run_dir).as_posix(),
        ]
        try:
            import gemmi

            structure = gemmi.read_structure(str(source_local))
            if len(structure) == 0:
                return fail("Gemmi found no coordinate model in the predicted complex")
            structure.setup_entities()
            structure.assign_label_seq_id()
            structure.assign_het_flags()
            structure.write_pdb(str(complex_path))
        except Exception as exc:
            return fail(f"Gemmi could not convert predicted CIF to PDB: {exc}")
        if not complex_path.is_file() or not complex_path.stat().st_size:
            return fail("Gemmi produced no PDB coordinates from the predicted CIF")
    pdb_data = _normalize_complex_records(complex_path.read_text(errors="replace"))
    complex_path.write_text(pdb_data)
    try:
        _protein_only(pdb_data)
    except ValueError as exc:
        return fail(str(exc))
    import_job = None
    cleaning_job = None
    repair_report_source = None
    if clean_and_repair:
        try:
            import_job = create_protein_import_job(
                pdb_data,
                filename=complex_path.name,
                source="predicted_complex",
            )
            cleaning_job, cleaned_payload = run_protein_cleaning_job(import_job.run_id)
        except Exception as exc:
            return fail(f"Predicted-complex cleaning could not start: {exc}")
        if cleaning_job.status != "completed" or not cleaned_payload.get("success"):
            return fail(
                str(
                    cleaned_payload.get("error")
                    or cleaning_job.result.get("error")
                    or "Predicted-complex cleaning failed"
                )
            )
        repaired_complex = str(cleaned_payload.get("prepared_pdb_data") or "")
        if not repaired_complex.strip():
            return fail("Predicted-complex cleaning returned no repaired complex")
        pdb_data = _normalize_complex_records(repaired_complex)
        complex_path.write_text(pdb_data)
        if cleaning_job.artifact_manifest is not None:
            report_refs = cleaning_job.artifact_manifest.by_type("repair_report")
            if report_refs:
                repair_report_source = report_refs[0].resolve(
                    cleaning_job.run_dir, must_exist=True
                )
    protein_path = artifact_dir / f"{prefix}_protein_refined.pdb"
    try:
        protein_path.write_text(_protein_only(pdb_data))
    except ValueError as exc:
        return fail(str(exc))
    artifacts = [
        ArtifactRef.from_path(
            run_dir,
            complex_path,
            "prepared_complex",
            role="predicted_complex",
            metadata={"source_run_id": source_job.run_id, "source_artifact_id": source_artifact.artifact_id},
        ),
        ArtifactRef.from_path(
            run_dir,
            protein_path,
            "prepared_receptor",
            role="receptor",
            metadata={"source_run_id": source_job.run_id},
        ),
    ]
    repair_report_path = artifact_dir / f"{prefix}_repair_report.json"
    if repair_report_source is not None:
        shutil.copy2(repair_report_source, repair_report_path)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                repair_report_path,
                "repair_report",
                role="report",
                metadata={"cleaning_run_id": cleaning_job.run_id if cleaning_job else ""},
            )
        )
    ligand_block = _ligand_block(pdb_data)
    ligand_path = artifact_dir / f"{prefix}_ligand_refined.sdf"
    if ligand_block:
        molecule = Chem.MolFromPDBBlock(
            ligand_block, removeHs=False, sanitize=False, proximityBonding=True
        )
        if molecule is not None:
            try:
                Chem.SanitizeMol(molecule)
            except Exception:
                pass
            with Chem.SDWriter(str(ligand_path)) as writer:
                writer.write(molecule)
            if ligand_path.is_file() and ligand_path.stat().st_size:
                artifacts.append(
                    ArtifactRef.from_path(
                        run_dir,
                        ligand_path,
                        "prepared_ligand_set",
                        role="ligand",
                        metadata={"source_run_id": source_job.run_id, "candidate_id": candidate},
                    )
                )
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "structure",
        "workflow": "predicted_complex_promotion",
        "operation": "preparation",
        "source": engine,
        "tool": engine,
        "status": "completed",
        "parent_run_id": source_job.run_id,
        "source_prediction_run_id": source_job.run_id,
        "source_prediction_artifact_id": source_artifact.artifact_id,
        "import_run_id": import_job.run_id if import_job is not None else "",
        "cleaning_run_id": cleaning_job.run_id if cleaning_job is not None else "",
        "clean_and_repair": bool(clean_and_repair),
        "candidate_id": candidate,
        "ligand_count": 1 if any(item.artifact_type == "prepared_ligand_set" for item in artifacts) else 0,
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "command.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "python" if command[0] == "gemmi" else "internal",
                "commands": [command],
            },
            indent=2,
        )
        + "\n"
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "success": True,
                "prepared_complex": complex_path.relative_to(run_dir).as_posix(),
                "prepared_receptor": protein_path.relative_to(run_dir).as_posix(),
                "prepared_ligand": (
                    ligand_path.relative_to(run_dir).as_posix() if ligand_path.is_file() else ""
                ),
                "repair_report": (
                    repair_report_path.relative_to(run_dir).as_posix()
                    if repair_report_path.is_file()
                    else ""
                ),
                "import_run_id": import_job.run_id if import_job is not None else "",
                "cleaning_run_id": cleaning_job.run_id if cleaning_job is not None else "",
            },
            indent=2,
        )
        + "\n"
    )
    write_artifact_manifest(run_dir, artifacts)
    return JobRecord.load(run_dir, task_group=TASK_GROUP)
