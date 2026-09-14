from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import (
    JOB_SCHEMA_VERSION,
    JobRecord,
    iter_job_records,
    short_job_code,
)
from mn_ligand.core.provenance import (
    inherited_target_metadata,
    modification_history,
)
from mn_ligand.core.residue_mapping import (
    derive_residue_mapping,
    residue_mapping_artifact,
    subset_residue_mapping,
)
from mn_ligand.runtime import runs_root
from mn_ligand.ligandx.lib.chemistry.preparation.target_validation import (
    prepare_target_for_publication,
)


TASK_GROUP = "target-trimming"
POLYMER_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pdb_chain_ranges(pdb_data: str) -> tuple[dict[str, Any], ...]:
    residues: dict[str, set[int]] = {}
    for line in pdb_data.splitlines():
        if (
            not line.startswith(("ATOM  ", "HETATM"))
            or len(line) < 27
            or line[17:20].strip().upper() not in POLYMER_RESIDUES
        ):
            continue
        chain = line[21].strip() or "_"
        try:
            residue = int(line[22:26])
        except ValueError:
            continue
        residues.setdefault(chain, set()).add(residue)
    return tuple(
        {
            "chain": chain,
            "start": min(values),
            "end": max(values),
            "residue_count": len(values),
            "residues": tuple(sorted(values)),
        }
        for chain, values in sorted(residues.items())
        if values
    )


def trim_pdb_data(
    pdb_data: str,
    ranges: dict[str, tuple[int, int]],
) -> tuple[str, dict[str, Any]]:
    if not ranges:
        raise ValueError("Select at least one chain range")
    selected: list[str] = []
    retained_ligand_lines = [
        line
        for line in pdb_data.splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
        and line[17:20].strip().upper() not in POLYMER_RESIDUES | {"HOH", "WAT", "DOD"}
    ]
    kept_residues: dict[str, set[tuple[int, str]]] = {chain: set() for chain in ranges}
    kept_atoms = 0
    for line in pdb_data.splitlines():
        if (
            not line.startswith(("ATOM  ", "HETATM"))
            or len(line) < 27
            or line[17:20].strip().upper() not in POLYMER_RESIDUES
        ):
            continue
        chain = line[21].strip() or "_"
        if chain not in ranges:
            continue
        try:
            residue = int(line[22:26])
        except ValueError:
            continue
        start, end = ranges[chain]
        if not int(start) <= residue <= int(end):
            continue
        selected.append(line)
        kept_atoms += 1
        kept_residues[chain].add((residue, line[26].strip()))
    if not selected:
        raise ValueError("The selected terminal ranges contain no coordinate records")
    output: list[str] = []
    previous_chain = ""
    for line in selected:
        chain = line[21].strip() or "_"
        if previous_chain and chain != previous_chain:
            output.append("TER")
        output.append(line)
        previous_chain = chain
    output.append("TER")
    output.extend(retained_ligand_lines)
    output.append("END")
    output.append("")
    summary = {
        "ranges": {
            chain: {"start": int(bounds[0]), "end": int(bounds[1])}
            for chain, bounds in ranges.items()
        },
        "chain_count": len(ranges),
        "residue_count": sum(len(values) for values in kept_residues.values()),
        "atom_count": kept_atoms,
        "retained_ligand_atom_count": len(retained_ligand_lines),
    }
    return "\n".join(output), summary


def create_trimmed_target_job(
    *,
    source_job: JobRecord,
    source_artifact: ArtifactRef,
    source_path: Path,
    ranges: dict[str, tuple[int, int]],
) -> JobRecord:
    if source_path.suffix.lower() not in {".pdb", ".ent"}:
        raise ValueError("Target trimming currently requires a PDB prepared target")
    trimmed, summary = trim_pdb_data(source_path.read_text(errors="replace"), ranges)
    run_id = str(uuid4())
    run_dir = runs_root() / TASK_GROUP / run_id
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    complex_path = artifact_dir / "complex_trimmed.pdb"
    complex_path.write_text(trimmed)
    receptor_lines = [
        line
        for line in trimmed.splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
        and line[17:20].strip().upper() in POLYMER_RESIDUES
    ]
    target_path = artifact_dir / "target_trimmed.pdb"
    validated_target, target_validation = prepare_target_for_publication(
        "\n".join((*receptor_lines, "TER", "END", ""))
    )
    target_path.write_text(validated_target)
    source_mapping_path = residue_mapping_artifact(source_job)
    if source_mapping_path is not None:
        source_mapping = json.loads(source_mapping_path.read_text())
        residue_mapping = subset_residue_mapping(
            source_mapping,
            target_path.read_text(),
        )
    else:
        residue_mapping = derive_residue_mapping(
            target_path.read_text(),
            source_path.read_text(errors="replace"),
            source_run_id=source_job.run_id,
            source_label=str(source_job.metadata.get("pdb_id") or source_path.name),
        )
    residue_mapping_path = artifact_dir / "residue_mapping.json"
    residue_mapping_path.write_text(json.dumps(residue_mapping, indent=2) + "\n")
    copied_ligands: list[Path] = []
    if source_job.artifact_manifest is not None:
        for index, ligand in enumerate(
            source_job.artifact_manifest.by_type("prepared_ligand_set"), start=1
        ):
            ligand_source = ligand.resolve(source_job.run_dir, must_exist=True)
            if ligand_source is None:
                continue
            ligand_target = artifact_dir / (
                f"ligand_{index}_refined{ligand_source.suffix.lower()}"
            )
            ligand_target.write_bytes(ligand_source.read_bytes())
            copied_ligands.append(ligand_target)
    now = _utc_now_iso()
    jobs_by_id = {job.run_id: job for job in iter_job_records(runs_root())}
    jobs_by_id[source_job.run_id] = source_job
    target_identity = inherited_target_metadata(source_job, jobs_by_id)
    job_code = short_job_code(run_id)
    range_text = ", ".join(
        f"{chain}:{bounds['start']}-{bounds['end']}"
        for chain, bounds in summary["ranges"].items()
    )
    history = [
        *modification_history(source_job, jobs_by_id),
        {
            "run_id": run_id,
            "job_code": job_code,
            "kind": "target_trimming",
            "label": "Target trimming",
            "tool": "chain-aware terminal trimmer",
            "summary": f"Trimmed protein termini {range_text}",
        },
    ]
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": job_code,
        "job_type": "target_trimming",
        "workflow": "target_trimming",
        "operation": "preparation",
        "tool": "chain-aware terminal trimmer",
        "status": "completed",
        "parent_run_id": source_job.run_id,
        "source_target_run_id": source_job.run_id,
        "prepared_target_run_id": run_id,
        "ligand_key": source_job.metadata.get("ligand_key") or "",
        "ligand_count": int(source_job.metadata.get("ligand_count") or bool(copied_ligands)),
        "trim_ranges": summary["ranges"],
        "residue_numbering": "source_author",
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
        **target_identity,
        "modification_history": history,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "input.json").write_text(
        json.dumps(
            {
                "source_target": source_artifact.to_dict(),
                "parameters": {"trim_ranges": summary["ranges"]},
            },
            indent=2,
        )
        + "\n"
    )
    (run_dir / "command.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "internal",
                "operation": "chain-aware terminal trimming",
                "arguments": summary["ranges"],
            },
            indent=2,
        )
        + "\n"
    )
    result = {
        "success": True,
        "prepared_target": "artifacts/target_trimmed.pdb",
        "prepared_complex": "artifacts/complex_trimmed.pdb",
        "residue_mapping": "artifacts/residue_mapping.json",
        "target_validation": target_validation,
        **summary,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(
                run_dir,
                complex_path,
                "prepared_complex",
                role="trimmed_complex",
                metadata={
                    "source_run_id": source_job.run_id,
                    "trim_ranges": summary["ranges"],
                    "ligand_retained": True,
                },
            ),
            ArtifactRef.from_path(
                run_dir,
                target_path,
                "prepared_receptor",
                role="trimmed_receptor",
                metadata={
                    "source_run_id": source_job.run_id,
                    "trim_ranges": summary["ranges"],
                },
            ),
            ArtifactRef.from_path(
                run_dir,
                residue_mapping_path,
                "residue_mapping",
                role="author_numbering",
                metadata={
                    "source_run_id": source_job.run_id,
                    "trim_ranges": summary["ranges"],
                },
            ),
            *[
                ArtifactRef.from_path(
                    run_dir,
                    path,
                    "prepared_ligand_set",
                    role="retained_ligand",
                    metadata={"source_run_id": source_job.run_id},
                )
                for path in copied_ligands
            ],
        ],
    )
    return JobRecord.load(run_dir, task_group=TASK_GROUP)
