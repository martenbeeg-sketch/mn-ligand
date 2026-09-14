from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable
from uuid import uuid4

import gemmi

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    registered_tool,
    write_registered_command_record,
)
from mn_ligand.core.jobs import (
    JOB_SCHEMA_VERSION,
    JobRecord,
    iter_job_records,
    short_job_code,
)
from mn_ligand.runtime import (
    NATIVE_THREAD_ENVIRONMENT,
    adaptive_cpu_workers,
    runs_root,
)
from mn_ligand.workflows.pose_validation import (
    POSE_VALIDATION_INVENTORY_SCHEMA_VERSION,
    POSE_VALIDATION_SELECTION_POLICY,
    _copy_sdf_record,
    _safe_id,
    _source_receptor,
)


INTERACTION_ANALYSIS_TASK_GROUP = "interaction-analysis"
INTERACTION_RESIDUE_NUMBERING_POLICY_VERSION = 2
TARGET_COMPLEX_INVENTORY_SCHEMA_VERSION = 1
TARGET_COMPLEX_SELECTION_POLICY = "prepared-target-complex-v1"
INTERACTION_ENGINES = {
    "Native MD geometry": {
        "tool_id": "native-md-geometry",
        "image": "",
        "workflow": "native_md_geometry_interactions",
    },
    "PLIP": {
        "tool_id": "plip",
        "image": "ovolig-plip:latest",
        "workflow": "plip_interactions",
    },
    "PandaMap": {
        "tool_id": "pandamap",
        "image": "ovolig-pandamap:latest",
        "workflow": "pandamap_interactions",
    },
}
TARGET_COMPLEX_TASK_GROUPS = (
    "selected-complexes",
    "structure-jobs",
    "target-trimming",
    "terminal-repair",
)


def _stage_complex_as_pdb(source: Path, target: Path) -> None:
    """Stage one complex as real fixed-column PDB, regardless of its source."""
    structure = gemmi.read_structure(str(source))
    if not structure or not structure[0]:
        raise ValueError(f"No coordinates in {source.name}")
    target.write_text(structure.make_pdb_string())


def compatible_target_complex_jobs() -> list[JobRecord]:
    """Return immutable prepared complexes with a coordinate-bearing ligand."""
    jobs: list[JobRecord] = []
    for job in iter_job_records(
        runs_root(),
        task_groups=TARGET_COMPLEX_TASK_GROUPS,
        validate_artifacts=False,
    ):
        if job.status != "completed" or job.artifact_manifest is None:
            continue
        complexes = job.artifact_manifest.by_type("prepared_complex")
        ligands = job.artifact_manifest.by_type("prepared_ligand_set")
        if not complexes or not ligands:
            continue
        if any(
            artifact.resolve(job.run_dir, must_exist=True) is not None
            for artifact in complexes
        ) and any(
            artifact.resolve(job.run_dir, must_exist=True) is not None
            for artifact in ligands
        ):
            jobs.append(job)
    return jobs


def _complex_ligands(path: Path) -> list[dict[str, Any]]:
    structure = gemmi.read_structure(str(path))
    if not structure or not structure[0]:
        return []
    ligands: list[dict[str, Any]] = []
    for chain in structure[0]:
        for residue in chain:
            name = residue.name.strip().upper()
            info = gemmi.find_tabulated_residue(name)
            atom_names = {atom.name.strip().upper() for atom in residue}
            polymer_like = {"N", "CA", "C", "O"}.issubset(atom_names)
            if (
                name in {"HOH", "WAT", "DOD"}
                or info.is_amino_acid()
                or info.is_nucleic_acid()
                or (polymer_like and str(residue.het_flag) != "H")
            ):
                continue
            heavy_atoms = sum(
                atom.element.name.upper() != "H" for atom in residue
            )
            if not heavy_atoms:
                continue
            insertion = str(residue.seqid.icode).strip()
            ligands.append(
                {
                    "ligand_chain": str(chain.name).strip(),
                    "ligand_residue_name": name,
                    "ligand_residue_number": int(residue.seqid.num),
                    "ligand_insertion_code": insertion,
                    "ligand_heavy_atom_count": heavy_atoms,
                }
            )
    return ligands


def target_complex_candidates(source_job: JobRecord) -> list[dict[str, Any]]:
    if source_job.status != "completed" or source_job.artifact_manifest is None:
        return []
    ligand_key = str(source_job.metadata.get("ligand_key") or "")
    ligand_metadata = [
        dict(item)
        for item in source_job.metadata.get("ligands") or ()
        if isinstance(item, dict)
    ]
    smiles = str(source_job.metadata.get("ligand_smiles") or "")
    ligand_paths = [
        path
        for artifact in source_job.artifact_manifest.by_type(
            "prepared_ligand_set"
        )
        if (
            path := artifact.resolve(source_job.run_dir, must_exist=True)
        ) is not None
        and path.suffix.lower() in {".sdf", ".mol"}
    ]
    candidates: list[dict[str, Any]] = []
    for complex_index, artifact in enumerate(
        source_job.artifact_manifest.by_type("prepared_complex"), start=1
    ):
        source_path = artifact.resolve(source_job.run_dir, must_exist=True)
        if source_path is None:
            continue
        complex_ligands = _complex_ligands(source_path)
        for ligand_index, ligand in enumerate(complex_ligands, start=1):
            coordinate_key = "|".join(
                (
                    str(ligand["ligand_residue_name"]),
                    str(ligand["ligand_chain"]),
                    str(ligand["ligand_residue_number"]),
                    str(ligand["ligand_insertion_code"] or "_"),
                )
            )
            identity = next(
                (
                    item
                    for item in ligand_metadata
                    if str(item.get("coordinate_resname") or item.get("resname") or "").upper()
                    == str(ligand["ligand_residue_name"]).upper()
                    and str(item.get("chain") or "") == str(ligand["ligand_chain"])
                    and str(item.get("resseq") or "")
                    == str(ligand["ligand_residue_number"])
                    and str(item.get("icode") or "_")
                    == str(ligand["ligand_insertion_code"] or "_")
                ),
                None,
            )
            chemical_resname = str(
                (identity or {}).get("ccd_id")
                or (identity or {}).get("reference_ligand_ccd_id")
                or ""
            ).strip().upper()
            chemical_key = (
                "|".join(
                    (
                        chemical_resname,
                        str(ligand["ligand_chain"]),
                        str(ligand["ligand_residue_number"]),
                        str(ligand["ligand_insertion_code"] or "_"),
                    )
                )
                if chemical_resname
                else str((identity or {}).get("key") or coordinate_key)
            )
            ligand_smiles = str((identity or {}).get("smiles") or "")
            if not ligand_smiles and chemical_key == ligand_key:
                ligand_smiles = smiles
            candidates.append(
                {
                    "selection_id": _safe_id(
                        f"{chemical_key}__{coordinate_key}__prepared_complex_{complex_index}",
                        f"prepared_complex_{complex_index:03d}_ligand_{ligand_index:03d}",
                    ),
                    "compound_id": chemical_key,
                    "replicate": 1,
                    "prediction": (
                        f"{artifact.label or artifact.role or source_path.name}"
                        f" · {chemical_key}"
                        + (
                            f" (coordinate residue {coordinate_key})"
                            if chemical_key != coordinate_key
                            else ""
                        )
                    ),
                    "source_engine": source_job.tool or "Prepared target",
                    "source_kind": "prepared target complex",
                    "representative": True,
                    "selection_criterion": (
                        "Exact non-polymer residue in immutable prepared target complex"
                    ),
                    "smiles": ligand_smiles,
                    **ligand,
                    "_ligand_path": (
                        ligand_paths[0]
                        if len(ligand_paths) == 1
                        and len(complex_ligands) == 1
                        else None
                    ),
                    "_source_path": source_path,
                }
            )
    return candidates


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _author_numbering_reference(
    source_job: JobRecord,
    prepared_target: Path,
) -> tuple[Path, str]:
    """Prefer the immutable imported structure that retains author residue IDs."""
    try:
        source_input = json.loads(
            (source_job.run_dir / "input.json").read_text()
        )
    except (OSError, TypeError, ValueError):
        source_input = {}
    target_payload = (
        source_input.get("target_artifact") or source_input.get("target") or {}
    )
    target_run_id = (
        str(target_payload.get("run_id") or "")
        if isinstance(target_payload, dict) else ""
    )
    jobs = list(iter_job_records(runs_root()))
    jobs_by_id = {job.run_id: job for job in jobs}
    target_job = next(
        (job for job in jobs if job.run_id == target_run_id), None
    )
    lineage_job = target_job
    visited: set[str] = set()
    while lineage_job is not None and lineage_job.run_id not in visited:
        visited.add(lineage_job.run_id)
        if lineage_job.task_group == "protein-import":
            try:
                manifest = json.loads(
                    (lineage_job.run_dir / "artifacts.json").read_text()
                )
            except (OSError, TypeError, ValueError):
                manifest = {}
            for artifact in manifest.get("artifacts") or []:
                if not isinstance(artifact, dict):
                    continue
                if artifact.get("artifact_type") != "imported_target":
                    continue
                candidate = lineage_job.run_dir / str(
                    artifact.get("path") or ""
                )
                if candidate.is_file():
                    label = (
                        str(lineage_job.metadata.get("pdb_id") or "").strip()
                        or Path(candidate).name
                    )
                    return candidate, (
                        f"immutable imported-source author numbering from "
                        f"{label} job {lineage_job.run_id}"
                    )
            break
        parent_id = str(
            lineage_job.metadata.get("import_run_id")
            or lineage_job.metadata.get("source_target_run_id")
            or lineage_job.parent_run_id
            or ""
        )
        lineage_job = jobs_by_id.get(parent_id)
    pdb_id = str(
        (target_job.metadata.get("pdb_id") if target_job else "")
        or source_job.metadata.get("pdb_id")
        or ""
    ).strip().upper()
    if not pdb_id:
        return prepared_target, "prepared target numbering"
    for job in jobs:
        if (
            job.task_group != "protein-import"
            or job.status != "completed"
            or str(job.metadata.get("pdb_id") or "").strip().upper() != pdb_id
        ):
            continue
        try:
            manifest = json.loads((job.run_dir / "artifacts.json").read_text())
        except (OSError, TypeError, ValueError):
            continue
        for artifact in manifest.get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            if artifact.get("artifact_type") != "imported_target":
                continue
            candidate = job.run_dir / str(artifact.get("path") or "")
            if candidate.is_file():
                return candidate, (
                    f"immutable imported-source author numbering from "
                    f"{pdb_id} job "
                    f"{job.run_id}"
                )
    return prepared_target, "prepared target numbering"


def finalize_interaction_analysis_job(
    run_dir: Path,
    *,
    returncode: int,
) -> JobRecord:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    engine_slug = str(metadata.get("interaction_engine") or "").lower()
    summary = run_dir / "interaction_summary.csv"
    interactions = run_dir / "interactions.csv"
    report_path = run_dir / "interaction_report.json"
    try:
        report = json.loads(report_path.read_text())
    except (OSError, TypeError, ValueError):
        report = {}
    expected = int(metadata.get("pose_count") or 0)
    analyzed = int(report.get("analyzed_count") or 0)
    success = (
        returncode == 0
        and summary.is_file()
        and interactions.is_file()
        and analyzed == expected
    )
    artifacts: list[ArtifactRef] = []
    for path, artifact_type, role in (
        (summary, "interaction_summary", "summary"),
        (interactions, "protein_ligand_interactions", "normalized_interactions"),
        (report_path, "interaction_analysis_report", "run_report"),
    ):
        if path.is_file() and path.stat().st_size:
            artifacts.append(
                ArtifactRef.from_path(run_dir, path, artifact_type, role=role)
            )
    for path in sorted((run_dir / "prepared").glob("*")):
        if path.is_file():
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir, path, "interaction_complex", role=path.stem
                )
            )
    native_dir = run_dir / "native"
    if native_dir.is_dir():
        for path in sorted(native_dir.rglob("*")):
            if path.is_file():
                artifacts.append(
                    ArtifactRef.from_path(
                        run_dir, path, "native_interaction_output",
                        role=path.relative_to(native_dir).as_posix(),
                        checksum=False,
                    )
                )
    for name, role in (("stdout.log", "stdout"), ("stderr.log", "stderr")):
        path = run_dir / name
        path.touch(exist_ok=True)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir, path, "job_log", role=role, checksum=False
            )
        )
    error = "" if success else (
        (run_dir / "stderr.log").read_text(errors="replace")[-4000:]
        or f"{metadata.get('tool')} analyzed {analyzed}/{expected} expected poses"
    )
    result = {**report, "success": success, "returncode": returncode, "error": error}
    _write_json(run_dir / "result.json", result)
    write_artifact_manifest(run_dir, artifacts)
    completed = _utc_now_iso()
    metadata.update({
        "status": "completed" if success else "failed",
        "updated_at": completed,
        "completed_at": completed,
    })
    if error:
        metadata["error"] = error
    _write_json(metadata_path, metadata)
    return JobRecord.load(run_dir, task_group=INTERACTION_ANALYSIS_TASK_GROUP)


def queue_interaction_analysis_job(
    source_job: JobRecord,
    *,
    engine: str,
    selected_rows: Iterable[dict[str, Any]],
    max_workers: int | None = None,
    image: str | None = None,
    selection_schema_version: int = POSE_VALIDATION_INVENTORY_SCHEMA_VERSION,
    selection_policy: str = POSE_VALIDATION_SELECTION_POLICY,
    source_inventory_kind: str = "prediction_poses",
) -> JobRecord:
    if source_job.status != "completed":
        raise ValueError("Interaction analysis requires a completed source result")
    if engine not in INTERACTION_ENGINES:
        raise ValueError(f"Unsupported interaction engine: {engine}")
    rows = list(selected_rows)
    if not rows:
        raise ValueError("Select at least one compatible complex or pose")
    config = INTERACTION_ENGINES[engine]
    compound_work_items = len(
        {
            str(row.get("compound_id") or "").strip()
            for row in rows
            if str(row.get("compound_id") or "").strip()
        }
    ) or len(rows)
    max_workers = adaptive_cpu_workers(
        compound_work_items,
        requested=max_workers,
    )
    selected_image = str(image or config["image"])
    run_id = str(uuid4())
    run_dir = runs_root() / INTERACTION_ANALYSIS_TASK_GROUP / run_id
    input_dir = run_dir / "input"
    pose_dir = input_dir / "poses"
    complex_dir = input_dir / "complexes"
    ligand_dir = input_dir / "ligands"
    native_dir = run_dir / "native"
    pose_dir.mkdir(parents=True, exist_ok=False)
    complex_dir.mkdir()
    ligand_dir.mkdir()
    native_dir.mkdir()
    receptor_relative = ""
    source_receptor = _source_receptor(source_job)
    numbering_reference, numbering_reference_origin = (
        _author_numbering_reference(source_job, source_receptor)
    )
    reference_target = input_dir / "reference_target.pdb"
    reference_target.write_bytes(numbering_reference.read_bytes())
    reference_target_relative = reference_target.relative_to(run_dir).as_posix()
    if source_job.workflow == "docking_campaign":
        receptor = input_dir / "receptor.pdb"
        receptor.write_bytes(source_receptor.read_bytes())
        receptor_relative = receptor.relative_to(run_dir).as_posix()
    input_rows: list[dict[str, Any]] = []
    for index, source in enumerate(rows, start=1):
        pose_id = _safe_id(source.get("selection_id"), f"pose_{index:07d}")
        source_path = Path(source["_source_path"])
        mol_pred = ""
        mol_cond = receptor_relative
        complex_file = ""
        ligand_topology = ""
        if source_job.workflow == "docking_campaign":
            target = pose_dir / f"{pose_id}.sdf"
            _copy_sdf_record(
                source_path, int(source.get("_sdf_index") or 0), target
            )
            mol_pred = target.relative_to(run_dir).as_posix()
        else:
            target = complex_dir / f"{pose_id}.pdb"
            _stage_complex_as_pdb(source_path, target)
            complex_file = target.relative_to(run_dir).as_posix()
            mol_cond = ""
            source_ligand = source.get("_ligand_path")
            if source_ligand:
                ligand_source_path = Path(source_ligand)
                ligand_target = ligand_dir / (
                    f"{pose_id}{ligand_source_path.suffix.lower()}"
                )
                shutil.copy2(ligand_source_path, ligand_target)
                ligand_topology = ligand_target.relative_to(
                    run_dir
                ).as_posix()
        input_rows.append({
            "pose_id": pose_id,
            "compound_id": source.get("compound_id", ""),
            "source_engine": source.get("source_engine", source_job.tool),
            "source_kind": source.get("source_kind", ""),
            "replicate": source.get("replicate", ""),
            "prediction": source.get("prediction", ""),
            "selection_criterion": source.get("selection_criterion", ""),
            "smiles": source.get("smiles", ""),
            "ligand_chain": source.get("ligand_chain", ""),
            "ligand_residue_name": source.get("ligand_residue_name", ""),
            "ligand_residue_number": source.get("ligand_residue_number", ""),
            "ligand_insertion_code": source.get("ligand_insertion_code", ""),
            "ligand_heavy_atom_count": source.get(
                "ligand_heavy_atom_count", ""
            ),
            "mol_pred": mol_pred,
            "mol_cond": mol_cond,
            "complex_file": complex_file,
            "ligand_topology": ligand_topology,
            "reference_target": reference_target_relative,
            "source_artifact_path": source_path.relative_to(
                source_job.run_dir
            ).as_posix(),
        })
    table = input_dir / "interaction_inputs.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(input_rows[0]))
        writer.writeheader()
        writer.writerows(input_rows)
    tool_id = str(config["tool_id"])
    if engine == "Native MD geometry":
        runner = Path(__file__).with_name("native_interactions.py").resolve()
        command = [
            sys.executable,
            str(runner),
            "--input", str(table),
            "--native-output", str(native_dir),
            "--summary", str(run_dir / "interaction_summary.csv"),
            "--interactions", str(run_dir / "interactions.csv"),
            "--report", str(run_dir / "interaction_report.json"),
            "--max-workers", str(max_workers),
        ]
        resources = {
            "gpu": False,
            "cpu_threads": max_workers,
            "memory_gb": 2.0,
        }
    else:
        command = build_docker_command(
            DockerRunSpec(
                tool=registered_tool(tool_id, image=selected_image),
                command=(
                    "--input", "/workspace/input/interaction_inputs.csv",
                    "--native-output", "/workspace/native",
                    "--summary", "/workspace/interaction_summary.csv",
                    "--interactions", "/workspace/interactions.csv",
                    "--report", "/workspace/interaction_report.json",
                    "--max-workers", str(max_workers),
                ),
                mounts=(DockerMount(run_dir, "/workspace"),),
                environment=NATIVE_THREAD_ENVIRONMENT,
                gpu_enabled=False,
                use_host_user=True,
            )
        )
        resources = {
            **registered_tool(
                tool_id, image=selected_image
            ).resources.to_dict(),
            "cpu_threads": max_workers,
        }
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "interaction_analysis",
        "workflow": config["workflow"],
        "operation": "interaction_analysis",
        "status": "queued",
        "tool": engine,
        "interaction_engine": engine,
        "parent_run_id": source_job.run_id,
        "source_task_group": source_job.task_group,
        "source_workflow": source_job.workflow,
        "source_engine": source_job.tool,
        "selection_schema_version": int(selection_schema_version),
        "selection_policy": str(selection_policy),
        "source_inventory_kind": str(source_inventory_kind),
        "selection_ids": [
            str(source.get("selection_id") or "")
            for source in rows
        ],
        "residue_numbering_policy_version": (
            INTERACTION_RESIDUE_NUMBERING_POLICY_VERSION
        ),
        "pose_count": len(input_rows),
        "compound_count": len({str(row["compound_id"]) for row in input_rows}),
        "docker_image": selected_image,
        "created_at": now,
        "updated_at": now,
        "queued_at": now,
        "queued_command": command,
        "gpu_queued": False,
        "max_workers": max_workers,
        "cpu_worker_policy": "adaptive-global-limit",
        "cpu_work_items": compound_work_items,
        "protein_residue_numbering": (
            "immutable imported-source author numbering"
        ),
        "numbering_reference_origin": numbering_reference_origin,
        "resources": resources,
        "worker_finalizer": "interaction_analysis",
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "input.json", {
        "source_job": {
            "run_id": source_job.run_id,
            "task_group": source_job.task_group,
            "workflow": source_job.workflow,
            "tool": source_job.tool,
        },
        "engine": engine,
        "selection": [
            {key: value for key, value in row.items()
             if key not in {
                 "mol_pred", "mol_cond", "complex_file", "reference_target"
             }}
            for row in input_rows
        ],
        "parameters": {
            "selection_schema_version": int(selection_schema_version),
            "selection_policy": str(selection_policy),
            "source_inventory_kind": str(source_inventory_kind),
            "gpu": False,
            "max_workers": max_workers,
            "cpu_worker_policy": "adaptive-global-limit",
            "cpu_work_items": compound_work_items,
            "protein_residue_numbering": (
                "immutable imported-source author numbering"
            ),
            "residue_numbering_policy_version": (
                INTERACTION_RESIDUE_NUMBERING_POLICY_VERSION
            ),
            "numbering_reference_origin": numbering_reference_origin,
        },
    })
    if engine == "Native MD geometry":
        _write_json(run_dir / "command.json", {
            "tool_id": tool_id,
            "image": "",
            "argv": command,
            "commands": [command],
        })
    else:
        write_registered_command_record(
            run_dir, tool_id=tool_id, commands=(command,), image=selected_image
        )
    write_artifact_manifest(run_dir, [])
    return JobRecord.load(run_dir, task_group=INTERACTION_ANALYSIS_TASK_GROUP)
