from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
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
from mn_ligand.runtime import runs_root
from mn_ligand.modeller_runtime import modeller_python as resolve_modeller_python
from mn_ligand.ligandx.lib.chemistry.preparation.target_validation import (
    prepare_target_for_publication,
)
from mn_ligand.workflows.target_trimming import POLYMER_RESIDUES


TASK_GROUP = "terminal-repair"
MODELLER_RUNNER = Path(__file__).resolve().parent / "modeller_terminal_extension_runner.py"
ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
VALID_SEQUENCE = frozenset("ACDEFGHIKLMNPQRSTVWY")
REFERENCE_ONE_LETTER = {
    **ONE_LETTER,
    "ASX": "B",
    "GLX": "Z",
    "SEC": "U",
    "PYL": "O",
    # Common cacodylated cysteines normalized by Structure Import.
    "CAS": "C",
    "CAF": "C",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def modeller_python() -> Path:
    return resolve_modeller_python()


def modeller_readiness() -> tuple[bool, str]:
    interpreter = modeller_python()
    if not interpreter.is_file():
        return False, f"MODELLER interpreter not found: {interpreter}"
    process = subprocess.run(
        [str(interpreter), "-c", "import modeller; print(modeller.__version__)"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if process.returncode:
        return False, (process.stderr or process.stdout or "MODELLER import failed").strip()
    version = (process.stdout or "").strip().splitlines()
    return True, f"MODELLER {version[-1] if version else 'available'} · {interpreter}"


def normalize_extension_sequence(value: str, *, maximum: int = 30) -> str:
    sequence = "".join(str(value).split()).upper()
    if not sequence:
        raise ValueError("Enter at least one amino acid to append")
    invalid = sorted(set(sequence) - VALID_SEQUENCE)
    if invalid:
        raise ValueError(
            "Use canonical one-letter amino-acid codes only; invalid: "
            + ", ".join(invalid)
        )
    if len(sequence) > maximum:
        raise ValueError(f"Terminal extensions are limited to {maximum} residues")
    return sequence


def pdb_chain_sequences(pdb_data: str) -> tuple[dict[str, Any], ...]:
    chains: dict[str, list[tuple[int, str, str]]] = {}
    seen: set[tuple[str, int, str]] = set()
    for line in pdb_data.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 27:
            continue
        residue_name = line[17:20].strip().upper()
        if residue_name not in ONE_LETTER:
            continue
        chain = line[21].strip() or "_"
        try:
            residue_number = int(line[22:26])
        except ValueError:
            continue
        insertion = line[26].strip()
        key = (chain, residue_number, insertion)
        if key in seen:
            continue
        seen.add(key)
        chains.setdefault(chain, []).append(
            (residue_number, insertion, ONE_LETTER[residue_name])
        )
    rows = []
    for chain, residues in sorted(chains.items()):
        residues.sort(key=lambda item: (item[0], item[1]))
        numbers = [item[0] for item in residues]
        gaps = [
            (left, right)
            for left, right in zip(numbers, numbers[1:])
            if right != left + 1
        ]
        rows.append(
            {
                "chain": chain,
                "start": numbers[0],
                "end": numbers[-1],
                "sequence": "".join(item[2] for item in residues),
                "residue_count": len(residues),
                "gaps": tuple(gaps),
                "has_insertions": any(item[1] for item in residues),
            }
        )
    return tuple(rows)


def pdb_seqres_sequences(pdb_data: str) -> dict[str, str]:
    """Return declared PDB SEQRES sequences, including supported modifications."""
    chains: dict[str, list[str]] = {}
    for line in pdb_data.splitlines():
        if not line.startswith("SEQRES") or len(line) < 20:
            continue
        chain = line[11].strip() or "_"
        for residue_name in line[19:].split():
            code = REFERENCE_ONE_LETTER.get(residue_name.upper(), "X")
            chains.setdefault(chain, []).append(code)
    return {chain: "".join(sequence) for chain, sequence in chains.items()}


def infer_terminal_sequence(
    reference_sequence: str,
    observed_sequence: str,
) -> dict[str, Any]:
    """Infer terminal coordinate omissions when the observed chain matches SEQRES."""
    reference = "".join(reference_sequence.split()).upper()
    observed = "".join(observed_sequence.split()).upper()
    if not reference or not observed:
        return {
            "matched": False,
            "reason": "Reference or observed sequence is unavailable",
        }
    positions = [
        index
        for index in range(len(reference))
        if reference.startswith(observed, index)
    ]
    if len(positions) != 1:
        return {
            "matched": False,
            "reference_length": len(reference),
            "observed_length": len(observed),
            "reason": (
                "Observed coordinates do not map uniquely to the declared sequence"
            ),
        }
    start = positions[0]
    end = start + len(observed)
    return {
        "matched": True,
        "match_method": "exact observed-sequence substring of PDB SEQRES",
        "reference_length": len(reference),
        "observed_length": len(observed),
        "reference_start": start + 1,
        "reference_end": end,
        "n_terminal_sequence": reference[:start],
        "c_terminal_sequence": reference[end:],
    }


def _selected_chain_pdb(pdb_data: str, chain: str) -> str:
    chain_id = " " if chain == "_" else chain
    lines = [
        line
        for line in pdb_data.splitlines()
        if line.startswith("ATOM  ")
        and len(line) >= 27
        and line[21] == chain_id
        and line[17:20].strip().upper() in POLYMER_RESIDUES
    ]
    if not lines:
        raise ValueError(f"Chain {chain} has no canonical protein coordinates")
    return "\n".join((*lines, "TER", "END", ""))


def _pir(
    *,
    template_name: str,
    chain: str,
    start: int,
    end: int,
    observed_sequence: str,
    extension: str,
) -> str:
    modeller_chain = " " if chain == "_" else chain
    return (
        f">P1;template\n"
        f"structureX:{template_name}:{start}:{modeller_chain}:{end}:{modeller_chain}::::\n"
        f"{observed_sequence}{'-' * len(extension)}*\n"
        f">P1;extended\n"
        f"sequence:extended:{start}:{modeller_chain}:{end + len(extension)}:{modeller_chain}::::\n"
        f"{observed_sequence}{extension}*\n"
    )


def _coordinates(line: str) -> tuple[float, float, float]:
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


def _distance(left: str, right: str) -> float:
    a = _coordinates(left)
    b = _coordinates(right)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def merge_extension(
    original_pdb: str,
    model_pdb: str,
    *,
    chain: str,
    original_end: int,
    extension_length: int,
) -> tuple[str, dict[str, Any]]:
    chain_id = " " if chain == "_" else chain
    model_chain_lines = []
    for line in model_pdb.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54 or line[21] != chain_id:
            continue
        model_chain_lines.append(line)
    residue_keys: list[tuple[str, str]] = []
    for line in model_chain_lines:
        key = (line[22:26], line[26])
        if key not in residue_keys:
            residue_keys.append(key)
    extension_keys = residue_keys[-int(extension_length):]
    extension_lines = [
        line
        for line in model_chain_lines
        if (line[22:26], line[26]) in set(extension_keys)
    ]
    if not extension_lines:
        raise ValueError("MODELLER produced no C-terminal extension coordinates")
    residue_number_by_key = {
        key: original_end + offset
        for offset, key in enumerate(extension_keys, start=1)
    }
    original_lines = original_pdb.splitlines()
    max_serial = max(
        (
            int(line[6:11])
            for line in original_lines
            if line.startswith(("ATOM  ", "HETATM")) and line[6:11].strip().isdigit()
        ),
        default=0,
    )
    rewritten = []
    for index, line in enumerate(extension_lines, start=max_serial + 1):
        residue_number = residue_number_by_key[(line[22:26], line[26])]
        renumbered = (
            f"{line[:6]}{index:5d}{line[11:21]}{chain_id}"
            f"{residue_number:4d} {line[27:]}"
        )
        rewritten.append(renumbered)
    insertion_index = max(
        index
        for index, line in enumerate(original_lines)
        if line.startswith("ATOM  ") and len(line) >= 27 and line[21] == chain_id
    ) + 1
    output = [
        *original_lines[:insertion_index],
        *rewritten,
        *original_lines[insertion_index:],
    ]
    last_carbon = next(
        (
            line
            for line in reversed(original_lines[:insertion_index])
            if line.startswith("ATOM  ")
            and line[21] == chain_id
            and line[12:16].strip() == "C"
            and int(line[22:26]) == original_end
        ),
        None,
    )
    first_nitrogen = next(
        (
            line
            for line in rewritten
            if line[12:16].strip() == "N"
            and int(line[22:26]) == original_end + 1
        ),
        None,
    )
    junction = (
        _distance(last_carbon, first_nitrogen)
        if last_carbon is not None and first_nitrogen is not None
        else None
    )
    original_heavy = [
        line
        for line in original_lines
        if line.startswith(("ATOM  ", "HETATM"))
        and len(line) >= 54
        and (line[76:78].strip() if len(line) >= 78 else line[12:14].strip()).upper() != "H"
    ]
    extension_heavy = [
        line
        for line in rewritten
        if (line[76:78].strip() if len(line) >= 78 else line[12:14].strip()).upper() != "H"
    ]
    clashes = 0
    for new_atom in extension_heavy:
        new_residue = int(new_atom[22:26])
        for old_atom in original_heavy:
            old_number = old_atom[22:26].strip()
            old_residue = int(old_number) if old_number.lstrip("-").isdigit() else -9999
            if old_atom[21] == chain_id and new_residue == original_end + 1 and old_residue == original_end:
                continue
            if _distance(new_atom, old_atom) < 1.5:
                clashes += 1
    original_ligands = sum(
        line.startswith("HETATM")
        and line[17:20].strip().upper() not in {"HOH", "WAT", "DOD"}
        for line in original_lines
    )
    return "\n".join(output).rstrip() + "\n", {
        "extension_atom_count": len(rewritten),
        "junction_cn_angstrom": junction,
        "heavy_atom_clashes_below_1_5_angstrom": clashes,
        "retained_ligand_atom_count": original_ligands,
    }


def _write_status(run_dir: Path, metadata: dict[str, Any], result: dict[str, Any]) -> None:
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def create_terminal_repair_job(
    *,
    source_job: JobRecord,
    source_artifact: ArtifactRef,
    source_path: Path,
    chain: str,
    extension_sequence: str,
    model_count: int = 10,
    sequence_origin: str = "user_entered",
    sequence_evidence: dict[str, Any] | None = None,
) -> JobRecord:
    extension = normalize_extension_sequence(extension_sequence)
    if not 1 <= int(model_count) <= 50:
        raise ValueError("Number of models must be between 1 and 50")
    if source_path.suffix.lower() not in {".pdb", ".ent"}:
        raise ValueError("C-terminal Repair currently requires a PDB prepared complex")
    original = source_path.read_text(errors="replace")
    chain_rows = {str(row["chain"]): row for row in pdb_chain_sequences(original)}
    if chain not in chain_rows:
        raise ValueError(f"Protein chain {chain} was not found")
    selected = chain_rows[chain]
    if selected["has_insertions"]:
        raise ValueError("Repair does not yet support terminal chains with insertion codes")
    if selected["gaps"]:
        raise ValueError(
            "The selected chain has unresolved internal coordinate gaps; repair those "
            "before extending its terminus"
        )
    start, end = int(selected["start"]), int(selected["end"])
    run_id = str(uuid4())
    run_dir = runs_root() / TASK_GROUP / run_id
    work_dir = run_dir / "work"
    artifact_dir = run_dir / "artifacts"
    models_dir = artifact_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=False)
    work_dir.mkdir(parents=True, exist_ok=True)
    template_path = work_dir / "template.pdb"
    alignment_path = work_dir / "alignment.ali"
    native_output = work_dir / "native_result.json"
    template_path.write_text(_selected_chain_pdb(original, chain))
    alignment_path.write_text(
        _pir(
            template_name=template_path.stem,
            chain=chain,
            start=start,
            end=end,
            observed_sequence=str(selected["sequence"]),
            extension=extension,
        )
    )
    interpreter = modeller_python()
    command = [
        str(interpreter),
        str(MODELLER_RUNNER),
        "--alignment", alignment_path.name,
        "--template", template_path.name,
        "--chain", " " if chain == "_" else chain,
        "--first-new", str(end + 1),
        "--last-new", str(end + len(extension)),
        "--models", str(int(model_count)),
        "--output", native_output.name,
    ]
    now = _utc_now_iso()
    jobs_by_id = {job.run_id: job for job in iter_job_records(runs_root())}
    jobs_by_id[source_job.run_id] = source_job
    target_identity = inherited_target_metadata(source_job, jobs_by_id)
    job_code = short_job_code(run_id)
    history = [
        *modification_history(source_job, jobs_by_id),
        {
            "run_id": run_id,
            "job_code": job_code,
            "kind": "terminal_repair",
            "label": "MODELLER repair",
            "tool": "MODELLER",
            "summary": (
                f"Extended chain {chain} C-terminus by {len(extension)} aa "
                f"({extension})"
            ),
        },
    ]
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": job_code,
        "job_type": "terminal_repair",
        "workflow": "terminal_repair",
        "operation": "preparation",
        "tool": "MODELLER",
        "status": "running",
        "parent_run_id": source_job.run_id,
        "source_target_run_id": source_job.run_id,
        "chain": chain,
        "extension_sequence": extension,
        "sequence_origin": sequence_origin,
        "model_count": int(model_count),
        "created_at": now,
        "updated_at": now,
        **target_identity,
        "modification_history": history,
    }
    (run_dir / "input.json").write_text(
        json.dumps(
            {
                "source_target": source_artifact.to_dict(),
                "parameters": {
                    "chain": chain,
                    "extension_sequence": extension,
                    "sequence_origin": sequence_origin,
                    "sequence_evidence": dict(sequence_evidence or {}),
                    "model_count": int(model_count),
                },
            },
            indent=2,
        ) + "\n"
    )
    (run_dir / "command.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "native",
                "executable": str(interpreter),
                "argv": command[1:],
                "cwd": "work",
            },
            indent=2,
        ) + "\n"
    )
    _write_status(run_dir, metadata, {"success": False, "status": "running"})
    try:
        process = subprocess.run(
            command,
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        (run_dir / "stdout.log").write_text(process.stdout or "")
        (run_dir / "stderr.log").write_text(process.stderr or "")
        if process.returncode:
            raise RuntimeError(
                (process.stderr or process.stdout or "MODELLER failed").strip()
            )
        native = json.loads(native_output.read_text())
        model_rows = []
        for index, native_model in enumerate(native.get("models") or [], start=1):
            if native_model.get("failure"):
                continue
            model_source = work_dir / str(native_model.get("name") or "")
            if not model_source.is_file():
                continue
            merged, validation = merge_extension(
                original,
                model_source.read_text(errors="replace"),
                chain=chain,
                original_end=end,
                extension_length=len(extension),
            )
            model_path = models_dir / f"terminal_extension_{index:03d}.pdb"
            model_path.write_text(merged)
            model_rows.append(
                {
                    "model": index,
                    "path": model_path.relative_to(run_dir).as_posix(),
                    "dope": native_model.get("dope"),
                    "ga341": native_model.get("ga341"),
                    **validation,
                }
            )
        if not model_rows:
            raise RuntimeError("MODELLER did not produce a usable terminal-extension model")
        model_rows.sort(
            key=lambda row: (
                int(row["heavy_atom_clashes_below_1_5_angstrom"]),
                float(row["dope"]) if row["dope"] is not None else float("inf"),
            )
        )
        best = model_rows[0]
        best_path = artifact_dir / "complex_repaired.pdb"
        shutil.copy2(run_dir / str(best["path"]), best_path)
        validated_complex, target_validation = prepare_target_for_publication(
            best_path.read_text()
        )
        best_path.write_text(validated_complex)
        receptor_path = artifact_dir / "target_repaired.pdb"
        receptor_path.write_text(
            "\n".join(
                [
                    *(line for line in best_path.read_text().splitlines() if line.startswith("ATOM  ")),
                    "TER", "END", "",
                ]
            )
        )
        summary_path = artifact_dir / "repair_summary.json"
        summary = {
            "chain": chain,
            "original_terminal_residue": end,
            "extension_sequence": extension,
            "sequence_origin": sequence_origin,
            "sequence_evidence": dict(sequence_evidence or {}),
            "new_terminal_residue": end + len(extension),
            "requested_models": int(model_count),
            "completed_models": len(model_rows),
            "ranking": "fewest severe clashes, then lowest DOPE score",
            "best_model": best,
            "models": model_rows,
            "target_validation": target_validation,
            "interpretation": (
                "The appended terminus is a modeled flexible conformation and must be "
                "reviewed/equilibrated before production molecular dynamics."
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        copied_ligands = []
        if source_job.artifact_manifest is not None:
            for index, ligand in enumerate(
                source_job.artifact_manifest.by_type("prepared_ligand_set"), start=1
            ):
                ligand_source = ligand.resolve(source_job.run_dir, must_exist=True)
                if ligand_source is None:
                    continue
                ligand_target = artifact_dir / f"ligand_{index}{ligand_source.suffix.lower()}"
                shutil.copy2(ligand_source, ligand_target)
                copied_ligands.append(ligand_target)
        artifacts = [
            ArtifactRef.from_path(
                run_dir, best_path, "prepared_complex", role="repaired_complex",
                metadata={
                    "source_run_id": source_job.run_id,
                    "chain": chain,
                    "extension_sequence": extension,
                    "modeled_flexible_terminus": True,
                    "ligand_retained": True,
                },
            ),
            ArtifactRef.from_path(
                run_dir, receptor_path, "prepared_receptor", role="repaired_receptor",
                metadata={"chain": chain, "extension_sequence": extension},
            ),
            ArtifactRef.from_path(
                run_dir, summary_path, "repair_report", role="validation_summary"
            ),
            *[
                ArtifactRef.from_path(
                    run_dir, run_dir / str(row["path"]), "structure_model",
                    role="terminal_extension_model",
                    metadata={"model": row["model"], "dope": row["dope"]},
                )
                for row in model_rows
            ],
            *[
                ArtifactRef.from_path(
                    run_dir, path, "prepared_ligand_set", role="retained_ligand",
                    metadata={"source_run_id": source_job.run_id},
                )
                for path in copied_ligands
            ],
        ]
        write_artifact_manifest(run_dir, artifacts)
        finished = _utc_now_iso()
        metadata.update(
            {
                "status": "completed",
                "updated_at": finished,
                "completed_at": finished,
                "prepared_target_run_id": run_id,
                "completed_models": len(model_rows),
            }
        )
        result = {
            "success": True,
            "prepared_complex": "artifacts/complex_repaired.pdb",
            "prepared_target": "artifacts/target_repaired.pdb",
            "repair_summary": "artifacts/repair_summary.json",
            **summary,
        }
    except Exception as exc:
        if not (run_dir / "stdout.log").is_file():
            (run_dir / "stdout.log").write_text("")
        if not (run_dir / "stderr.log").is_file():
            (run_dir / "stderr.log").write_text(str(exc) + "\n")
        failed = _utc_now_iso()
        metadata.update({"status": "failed", "updated_at": failed, "completed_at": failed})
        result = {"success": False, "error": str(exc)}
        write_artifact_manifest(run_dir, [])
    _write_status(run_dir, metadata, result)
    return JobRecord.load(run_dir, task_group=TASK_GROUP)
