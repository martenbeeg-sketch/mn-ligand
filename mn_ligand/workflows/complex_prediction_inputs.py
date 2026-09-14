from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from uuid import uuid4

from rdkit import Chem

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.runtime import runs_root


TASK_GROUP = "complex-prediction-inputs"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def normalize_sequence(value: str) -> tuple[str, str]:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    header = "protein"
    if lines and lines[0].startswith(">"):
        header = lines.pop(0)[1:].strip().split()[0] or "protein"
    sequence = "".join(lines).replace(" ", "").upper()
    if len(sequence) < 10 or any(character not in VALID_AA for character in sequence):
        raise ValueError("Protein sequence must contain at least 10 standard amino acids")
    safe_header = re.sub(r"[^A-Za-z0-9_.-]+", "-", header).strip("-._") or "protein"
    return safe_header, sequence


def normalize_ligand(value: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    if "," not in raw:
        raise ValueError("Ligand input must use `LIGAND_ID,SMILES`")
    ligand_id, smiles = (part.strip() for part in raw.split(",", 1))
    molecule = Chem.MolFromSmiles(smiles)
    if not ligand_id or molecule is None:
        raise ValueError("Ligand ID and a valid SMILES are required")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", ligand_id).strip("-._") or "LIG"
    return safe_id, Chem.MolToSmiles(molecule, isomericSmiles=True)


def create_complex_prediction_inputs(
    *,
    protein_input: str,
    ligand_input: str,
) -> tuple[JobRecord, ArtifactRef, ArtifactRef, tuple[tuple[str, str], ...]]:
    protein_id, sequence = normalize_sequence(protein_input)
    ligand_id, smiles = normalize_ligand(ligand_input)
    run_id = str(uuid4())
    run_dir = runs_root() / TASK_GROUP / run_id
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    fasta_path = artifact_dir / "target_sequence.fasta"
    fasta_path.write_text(f">{protein_id}\n{sequence}\n")
    compound_path = artifact_dir / "ligand.smi"
    compound_path.write_text(f"{smiles}\t{ligand_id}\n")
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "complex_prediction_inputs",
        "workflow": "complex_prediction_inputs",
        "operation": "preparation",
        "tool": "typed sequence-ligand input",
        "status": "completed",
        "protein_id": protein_id,
        "protein_sequence_length": len(sequence),
        "ligand_id": ligand_id,
        "ligand_smiles": smiles,
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "input.json").write_text(
        json.dumps(
            {
                "protein": {"id": protein_id, "sequence": sequence},
                "ligand": {"id": ligand_id, "smiles": smiles},
            },
            indent=2,
        )
        + "\n"
    )
    (run_dir / "result.json").write_text(
        json.dumps({"success": True, "protein_id": protein_id, "ligand_id": ligand_id}, indent=2)
        + "\n"
    )
    manifest = write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(
                run_dir,
                fasta_path,
                "target_sequence",
                role="protein",
                metadata={"protein_id": protein_id, "sequence_length": len(sequence)},
            ),
            ArtifactRef.from_path(
                run_dir,
                compound_path,
                "compound_set",
                role="single_ligand",
                metadata={"compound_count": 1, "ligand_id": ligand_id},
            ),
        ],
    )
    return (
        JobRecord.load(run_dir, task_group=TASK_GROUP),
        manifest.by_type("target_sequence")[0],
        manifest.by_type("compound_set")[0],
        (("A", sequence),),
    )
