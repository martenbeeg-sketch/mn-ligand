from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.docker_runner import write_registered_command_record
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.protein_preparation import (
    DEFAULT_PROTEIN_CLEANING_IMAGE,
    _cleaning_command,
)


TASK_GROUP = "structure-jobs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _protein_only_pdb(pdb_data: str) -> str:
    return "\n".join(
        [
            *(
                line
                for line in pdb_data.splitlines()
                if line.startswith("ATOM  ")
                or line.startswith(
                    ("TER", "MODEL", "ENDMDL", "CRYST1", "HEADER", "TITLE", "REMARK")
                )
            ),
            "END",
            "",
        ]
    )


def minimize_prepared_complex(
    *,
    source_job: JobRecord,
    source_artifact: ArtifactRef,
    source_path: Path,
    image: str = DEFAULT_PROTEIN_CLEANING_IMAGE,
    use_gpu: bool = False,
) -> JobRecord:
    """Create a minimized typed complex from an existing prepared complex."""
    if source_artifact.artifact_type != "prepared_complex":
        raise ValueError("Energy minimization requires a prepared_complex artifact")
    if source_job.artifact_manifest is None:
        raise ValueError("The source job has no artifact manifest")
    ligand_refs = source_job.artifact_manifest.by_type("prepared_ligand_set")
    if not ligand_refs:
        raise ValueError("The source complex has no typed prepared ligand artifact")
    ligand_path = ligand_refs[0].resolve(source_job.run_dir, must_exist=True)
    if ligand_path is None:
        raise FileNotFoundError(ligand_refs[0].path)

    run_id = str(uuid4())
    run_dir = runs_root() / TASK_GROUP / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "energy_minimization",
        "workflow": "structure_import",
        "status": "running",
        "tool": "OpenMM LocalEnergyMinimizer",
        "source": "OpenMM minimization",
        "parent_run_id": source_job.run_id,
        "source_structure_run_id": source_job.run_id,
        "pdb_id": source_job.metadata.get("pdb_id") or "",
        "ligand_key": source_job.metadata.get("ligand_key") or "",
        "ligands": list(source_job.metadata.get("ligands") or []),
        "receptor": dict(source_job.metadata.get("receptor") or {}),
        "protein_chains": list(source_job.metadata.get("protein_chains") or []),
        "docker_image": image,
        "use_gpu": bool(use_gpu),
        "created_at": now,
        "updated_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "source_task_group": source_job.task_group,
            "input_artifact": source_artifact.to_dict(),
            "ligand_artifact": ligand_refs[0].to_dict(),
            "parameters": {
                "clean_protein": True,
                "map_modified_residues": False,
                "ph": 7.4,
                "add_missing_residues": False,
                "skip_terminal_missing_residues": True,
                "max_internal_gap": 15,
                "refine_rebuilt_positions": True,
                "preserve_nonwater_heterogens": False,
                "backfill_existing_structure": True,
            },
        },
    )
    runner_input = run_dir / "runner_input.json"
    native_result = run_dir / "native_result.json"
    _write_json(
        runner_input,
        {
            "pdb_id": source_job.metadata.get("pdb_id") or "protein",
            "pdb_path": f"/input/{source_path.name}",
            "clean_protein": True,
            "map_modified_residues": False,
            "ph": 7.4,
            "add_missing_residues": False,
            "skip_terminal_missing_residues": True,
            "max_internal_gap": 15,
            "refine_rebuilt_positions": True,
            "preserve_nonwater_heterogens": False,
        },
    )
    _write_json(native_result, {})
    native_result.chmod(0o666)
    command = _cleaning_command(
        image=image,
        run_dir=run_dir,
        source_path=source_path,
        runner_input=runner_input,
        native_result=native_result,
        use_gpu=use_gpu,
    )
    write_registered_command_record(
        run_dir,
        tool_id="openmm_md" if use_gpu else "protein_cleaning",
        commands=(command,),
        image=image,
    )
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    native_result.chmod(0o644)
    (run_dir / "stdout.log").write_text(process.stdout or "")
    (run_dir / "stderr.log").write_text(process.stderr or "")
    try:
        native_payload = json.loads(native_result.read_text())
    except (OSError, ValueError):
        native_payload = {}
    if process.returncode != 0 or native_payload.get("success") is not True:
        error = str(
            native_payload.get("error")
            or (process.stderr or "")[-4000:]
            or "OpenMM minimization failed"
        )
        _write_json(
            run_dir / "result.json",
            {"success": False, "error": error, "returncode": process.returncode},
        )
        write_artifact_manifest(run_dir, [])
        completed = _utc_now_iso()
        metadata.update(
            {
                "status": "failed",
                "error": error,
                "updated_at": completed,
                "completed_at": completed,
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        return JobRecord.load(run_dir, task_group=TASK_GROUP)

    prepared = str(native_payload.get("prepared_pdb_data") or "")
    refinement = dict((native_payload.get("repair_report") or {}).get("refinement") or {})
    if not prepared.strip() or not refinement.get("engine"):
        raise RuntimeError("Native output did not contain a minimized structure and refinement report")
    artifacts_dir = run_dir / "artifacts"
    reports_dir = artifacts_dir / "reports"
    artifacts_dir.mkdir()
    reports_dir.mkdir()
    stem = str(source_job.metadata.get("pdb_id") or "target").lower()
    complex_path = artifacts_dir / f"{stem}_complex_minimized.pdb"
    receptor_path = artifacts_dir / f"{stem}_protein_minimized.pdb"
    ligand_output = artifacts_dir / f"{stem}_ligand_minimized_input{ligand_path.suffix}"
    report_path = reports_dir / "minimization_report.json"
    complex_path.write_text(prepared)
    receptor_path.write_text(_protein_only_pdb(prepared))
    shutil.copy2(ligand_path, ligand_output)
    report = {
        "source_run_id": source_job.run_id,
        "source_artifact": source_artifact.to_dict(),
        "engine": refinement.get("engine"),
        "forcefield": refinement.get("forcefield"),
        "ligand_context": refinement.get("ligand_context"),
        "ligand_parameterization": refinement.get("ligand_parameterization"),
        "movable_atom_count": refinement.get("movable_atom_count"),
        "frozen_atom_count": refinement.get("frozen_atom_count"),
        "potential_energy_before_kj_mol": refinement.get(
            "potential_energy_before_kj_mol"
        ),
        "potential_energy_after_kj_mol": refinement.get(
            "potential_energy_after_kj_mol"
        ),
        "max_iterations": refinement.get("max_iterations"),
        "tolerance_kj_mol_nm": refinement.get("tolerance_kj_mol_nm"),
        "native_repair_report": native_payload.get("repair_report") or {},
    }
    _write_json(report_path, report)
    refs = [
        ArtifactRef.from_path(
            run_dir,
            complex_path,
            "prepared_complex",
            role="complex",
            metadata={"source_run_id": source_job.run_id, "energy_minimized": True},
        ),
        ArtifactRef.from_path(
            run_dir,
            receptor_path,
            "prepared_receptor",
            role="receptor",
            metadata={"source_run_id": source_job.run_id, "energy_minimized": True},
        ),
        ArtifactRef.from_path(
            run_dir,
            ligand_output,
            "prepared_ligand_set",
            role="ligand",
            metadata={"source_run_id": source_job.run_id},
        ),
        ArtifactRef.from_path(
            run_dir, report_path, "minimization_report", role="report"
        ),
    ]
    write_artifact_manifest(run_dir, refs)
    result = {
        "success": True,
        "source_run_id": source_job.run_id,
        "prepared_complex": complex_path.relative_to(run_dir).as_posix(),
        "prepared_receptor": receptor_path.relative_to(run_dir).as_posix(),
        "prepared_ligand_set": ligand_output.relative_to(run_dir).as_posix(),
        "minimization_report": report_path.relative_to(run_dir).as_posix(),
        "refinement": report,
    }
    _write_json(run_dir / "result.json", result)
    completed = _utc_now_iso()
    metadata.update(
        {
            "status": "completed",
            "updated_at": completed,
            "completed_at": completed,
            "energy_minimized": True,
        }
    )
    _write_json(run_dir / "metadata.json", metadata)
    return JobRecord.load(run_dir, task_group=TASK_GROUP)
