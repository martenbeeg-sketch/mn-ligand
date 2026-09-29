from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    registered_tool,
    write_registered_command_record,
)
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.runtime import reference_root, runs_root
from mn_ligand.workflows.lddm import LDDM_CHECKPOINTS, mean_coordinate_uncertainty
from mn_ligand.workflows.pose_validation import _source_receptor, _source_smiles
from mn_ligand.workflows.rescoring import source_pose_rows


LDDM_EVALUATION_TASK_GROUP = "lddm-evaluation"
LDDM_EVALUATION_WORKFLOW = "lddm_pose_agreement"
DEFAULT_LDDM_IMAGE = "mn-lddm:cu128-f254fb4"


def _engine_label(source_job: JobRecord) -> str:
    name = str(source_job.metadata.get("tool") or source_job.tool or "")
    return "RosettaLigand" if name.lower() == "openvs" else name


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def compatible_source_jobs() -> list[JobRecord]:
    from mn_ligand.core.jobs import iter_job_records

    return [
        job
        for job in iter_job_records(
            runs_root(), task_groups=("docking",), validate_artifacts=False
        )
        if job.status == "completed"
        and (
            (
                job.workflow == "docking_campaign"
                and str(job.tool or job.metadata.get("engine") or "").lower()
                in {"vina", "gnina", "udp", "unidock", "unidock_pro"}
            )
            or job.workflow == "openvs_docking"
        )
    ]


def source_pose_inventory(source_job: JobRecord) -> list[dict[str, Any]]:
    if source_job.workflow != "openvs_docking":
        return source_pose_rows(source_job)
    path = source_job.run_dir / "openvs_scores.csv"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(newline="", errors="replace") as handle:
            for row in csv.DictReader(handle):
                pose_file = str(row.get("pose_file") or "").strip()
                compound_id = str(row.get("compound_id") or "").strip()
                if not pose_file or not compound_id:
                    continue
                if not (source_job.run_dir / pose_file).is_file():
                    continue
                rows.append(
                    {
                        "compound_id": compound_id,
                        "replicate": int(row.get("replicate") or 1),
                        "source_pose_rank": int(row.get("pose_rank") or 1),
                        "source_pose_file": pose_file,
                        "source_score_kcal_mol": None,
                        "source_rosetta_total_score_reu": row.get("total_score_reu"),
                        "source_rosetta_dg_reu": row.get("estimated_dg_reu"),
                    }
                )
    except (OSError, TypeError, ValueError, csv.Error):
        return []
    return rows


def _reference_molecule(path: Path) -> Chem.Mol | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".mol":
            return Chem.MolFromMolFile(
                str(path), removeHs=False, sanitize=False, strictParsing=False
            )
        if suffix == ".sdf":
            supplier = Chem.SDMolSupplier(
                str(path), removeHs=False, sanitize=False, strictParsing=False
            )
            return next((molecule for molecule in supplier if molecule is not None), None)
        if suffix == ".mol2":
            return Chem.MolFromMol2File(
                str(path), removeHs=False, sanitize=False
            )
        if suffix in {".pdb", ".ent"}:
            return Chem.MolFromPDBFile(
                str(path), removeHs=False, sanitize=False, proximityBonding=True
            )
    except (OSError, ValueError, RuntimeError):
        return None
    return None


def _source_reference_ligand(
    source_job: JobRecord,
    rows: list[dict[str, Any]],
    smiles_by_id: dict[str, str],
) -> tuple[Path | None, Chem.Mol | None]:
    for reference in sorted((source_job.run_dir / "input").glob("reference_ligand.*")):
        if reference.suffix.lower() in {".pdb", ".ent"}:
            continue
        molecule = _reference_molecule(reference)
        if molecule is not None and molecule.GetNumConformers():
            return reference, None

    # LDDM uses this molecule to identify pocket residues. Its coordinates are
    # not supplied to the sampler as ligand coordinates.
    for row in rows:
        compound_id = str(row.get("compound_id") or "")
        molecule = _source_pose_molecule(
            source_job, row, smiles_by_id.get(compound_id, "")
        )
        if molecule is not None:
            return None, molecule
    raise FileNotFoundError(
        "The source campaign has no coordinate-bearing reference ligand or readable pose"
    )


def _canonical_smiles(molecule: Chem.Mol) -> str:
    return Chem.MolToSmiles(
        Chem.RemoveHs(molecule, sanitize=False),
        canonical=True,
        isomericSmiles=False,
    )


def _sdf_molecule(path: Path, index: int = 0) -> Chem.Mol | None:
    try:
        supplier = Chem.SDMolSupplier(
            str(path), removeHs=False, sanitize=False, strictParsing=False
        )
        if index < 0 or index >= len(supplier):
            return None
        molecule = supplier[index]
        if molecule is None or molecule.GetNumConformers() < 1:
            return None
        molecule = Chem.RemoveHs(Chem.Mol(molecule), sanitize=False)
        Chem.SanitizeMol(molecule)
        return molecule
    except (OSError, ValueError, RuntimeError):
        return None


def _pdb_pose_molecule(path: Path, smiles: str) -> Chem.Mol | None:
    """Rebuild Rosetta pose chemistry from one unambiguous HET residue."""
    template = Chem.MolFromSmiles(str(smiles or ""), sanitize=True)
    if template is None:
        return None
    residues: dict[tuple[str, str, str, str], list[str]] = {}
    try:
        for line in path.read_text(errors="replace").splitlines():
            if not line.startswith("HETATM") or len(line) < 78:
                continue
            element = line[76:78].strip().upper()
            residue_name = line[17:20].strip().upper()
            if element in {"H", "D"} or residue_name in {"HOH", "WAT", "DOD"}:
                continue
            key = (line[21:22], line[22:26], line[26:27], residue_name)
            residues.setdefault(key, []).append(line)
    except OSError:
        return None

    candidates: list[Chem.Mol] = []
    for atom_lines in residues.values():
        if len(atom_lines) != template.GetNumHeavyAtoms():
            continue
        coordinate_molecule = Chem.MolFromPDBBlock(
            "\n".join(atom_lines + ["END", ""]),
            sanitize=False,
            removeHs=True,
            proximityBonding=True,
        )
        if (
            coordinate_molecule is None
            or coordinate_molecule.GetNumAtoms() != template.GetNumAtoms()
        ):
            continue
        try:
            molecule = AllChem.AssignBondOrdersFromTemplate(
                Chem.Mol(template), coordinate_molecule
            )
            Chem.SanitizeMol(molecule)
        except (ValueError, RuntimeError):
            continue
        if _canonical_smiles(molecule) == _canonical_smiles(template):
            candidates.append(molecule)
    return candidates[0] if len(candidates) == 1 else None


def _source_pose_molecule(
    source_job: JobRecord, row: dict[str, Any], smiles: str
) -> Chem.Mol | None:
    source_path = source_job.run_dir / str(row.get("source_pose_file") or "")
    pose_index = max(0, int(row.get("source_pose_rank") or 1) - 1)
    if source_path.suffix.lower() in {".pdb", ".ent"}:
        return _pdb_pose_molecule(source_path, smiles)
    if source_path.suffix.lower() == ".sdf":
        return _sdf_molecule(source_path, pose_index)
    return _sdf_molecule(source_path.with_suffix(".sdf"), pose_index)


def _fixed_frame_rmsd(left: Chem.Mol, right: Chem.Mol) -> float | None:
    """Symmetry-aware heavy-atom RMSD without fitting either pose."""
    if _canonical_smiles(left) != _canonical_smiles(right):
        return None
    try:
        return float(rdMolAlign.CalcRMS(left, right, maxMatches=10000))
    except (RuntimeError, ValueError):
        return None


def _sampled_poses(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    poses: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((run_dir / "results").glob("**/*_out.sdf")):
        fallback_id = path.name.removesuffix("_out.sdf")
        try:
            supplier = Chem.SDMolSupplier(
                str(path), removeHs=False, sanitize=False, strictParsing=False
            )
            for index, raw_molecule in enumerate(supplier, start=1):
                if raw_molecule is None or raw_molecule.GetNumConformers() < 1:
                    continue
                molecule = Chem.RemoveHs(Chem.Mol(raw_molecule), sanitize=False)
                try:
                    Chem.SanitizeMol(molecule)
                except ValueError:
                    continue
                compound_id = (
                    raw_molecule.GetProp("compound_id")
                    if raw_molecule.HasProp("compound_id")
                    else fallback_id
                )
                uncertainty = mean_coordinate_uncertainty(raw_molecule)
                poses.setdefault(str(compound_id), []).append(
                    {
                        "molecule": molecule,
                        "pose_index": index,
                        "mean_uncertainty": uncertainty,
                        "pose_file": path.relative_to(run_dir).as_posix(),
                    }
                )
        except (OSError, ValueError):
            continue
    for values in poses.values():
        values.sort(
            key=lambda row: (
                row["mean_uncertainty"]
                if row["mean_uncertainty"] is not None
                else float("inf"),
                int(row["pose_index"]),
            )
        )
        for rank, row in enumerate(values, start=1):
            row["confidence_rank"] = rank
    return poses


def _comparison_rows(
    run_dir: Path, source_job: JobRecord
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sampled = _sampled_poses(run_dir)
    comparisons: list[dict[str, Any]] = []
    source_rows = source_pose_inventory(source_job)
    smiles_by_id = _source_smiles(source_job)
    for source in source_rows:
        compound_id = str(source.get("compound_id") or "")
        lddm_poses = sampled.get(compound_id, [])
        if not lddm_poses:
            continue
        source_molecule = _source_pose_molecule(
            source_job, source, smiles_by_id.get(compound_id, "")
        )
        if source_molecule is None:
            continue
        matched = [
            (lddm_pose, _fixed_frame_rmsd(source_molecule, lddm_pose["molecule"]))
            for lddm_pose in lddm_poses
        ]
        matched = [(pose, rmsd) for pose, rmsd in matched if rmsd is not None]
        if not matched:
            continue
        closest, closest_rmsd = min(matched, key=lambda item: item[1])
        confident = min(
            lddm_poses,
            key=lambda row: (
                row["mean_uncertainty"]
                if row["mean_uncertainty"] is not None
                else float("inf"),
                int(row["pose_index"]),
            ),
        )
        confident_rmsd = next(
            (rmsd for pose, rmsd in matched if pose is confident), None
        )
        comparisons.append(
            {
                "compound_id": compound_id,
                "source_engine": str(
                    _engine_label(source_job)
                ),
                "source_replicate": int(source.get("replicate") or 1),
                "source_pose_rank": int(source.get("source_pose_rank") or 1),
                "source_pose_file": source.get("source_pose_file"),
                "source_score_kcal_mol": source.get("source_score_kcal_mol"),
                "source_cnn_score": source.get("source_cnn_score"),
                "source_cnn_affinity": source.get("source_cnn_affinity"),
                "source_rosetta_total_score_reu": source.get(
                    "source_rosetta_total_score_reu"
                ),
                "source_rosetta_dg_reu": source.get("source_rosetta_dg_reu"),
                "closest_lddm_pose_index": int(closest["pose_index"]),
                "closest_lddm_confidence_rank": int(closest["confidence_rank"]),
                "closest_lddm_pose_file": closest["pose_file"],
                "closest_lddm_mean_uncertainty": closest["mean_uncertainty"],
                "closest_lddm_pose_rmsd_angstrom": closest_rmsd,
                "best_confidence_lddm_pose_index": int(confident["pose_index"]),
                "best_confidence_lddm_pose_file": confident["pose_file"],
                "best_confidence_lddm_mean_uncertainty": confident[
                    "mean_uncertainty"
                ],
                "best_confidence_pose_rmsd_angstrom": confident_rmsd,
                "lddm_sample_count": len(lddm_poses),
                "pose_agreement_within_2a": closest_rmsd <= 2.0,
            }
        )

    summaries: list[dict[str, Any]] = []
    for compound_id in sorted({row["compound_id"] for row in comparisons}):
        rows = [row for row in comparisons if row["compound_id"] == compound_id]
        closest_values = [
            float(row["closest_lddm_pose_rmsd_angstrom"]) for row in rows
        ]
        confidence_values = [
            float(row["best_confidence_pose_rmsd_angstrom"])
            for row in rows
            if row.get("best_confidence_pose_rmsd_angstrom") is not None
        ]
        summaries.append(
            {
                "compound_id": compound_id,
                "source_pose_count": len(rows),
                "lddm_sample_count": max(int(row["lddm_sample_count"]) for row in rows),
                "mean_closest_lddm_rmsd_angstrom": sum(closest_values)
                / len(closest_values),
                "closest_pose_agreement_within_2a_count": sum(
                    bool(row["pose_agreement_within_2a"]) for row in rows
                ),
                "mean_best_confidence_pose_rmsd_angstrom": (
                    sum(confidence_values) / len(confidence_values)
                    if confidence_values
                    else None
                ),
            }
        )
    return comparisons, summaries


def finalize_lddm_pose_agreement_job(
    run_dir: Path, *, returncode: int
) -> JobRecord:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    source_dir = runs_root() / str(
        metadata.get("source_task_group") or "docking"
    ) / str(metadata.get("source_run_id") or "")
    source_job = JobRecord.load(
        source_dir,
        task_group="docking",
        load_artifacts=False,
        validate_artifacts=False,
    )
    comparisons, summaries = _comparison_rows(run_dir, source_job)
    comparison_path = run_dir / "pose_agreement.csv"
    comparison_fields = [
        "compound_id",
        "source_engine",
        "source_replicate",
        "source_pose_rank",
        "source_pose_file",
        "source_score_kcal_mol",
        "source_cnn_score",
        "source_cnn_affinity",
        "source_rosetta_total_score_reu",
        "source_rosetta_dg_reu",
        "source_rosetta_total_score_reu",
        "source_rosetta_dg_reu",
        "closest_lddm_pose_index",
        "closest_lddm_confidence_rank",
        "closest_lddm_pose_file",
        "closest_lddm_mean_uncertainty",
        "closest_lddm_pose_rmsd_angstrom",
        "best_confidence_lddm_pose_index",
        "best_confidence_lddm_pose_file",
        "best_confidence_lddm_mean_uncertainty",
        "best_confidence_pose_rmsd_angstrom",
        "lddm_sample_count",
        "pose_agreement_within_2a",
    ]
    with comparison_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparisons)
    summary_path = run_dir / "pose_agreement_summary.csv"
    summary_fields = [
        "compound_id",
        "source_pose_count",
        "lddm_sample_count",
        "mean_closest_lddm_rmsd_angstrom",
        "closest_pose_agreement_within_2a_count",
        "mean_best_confidence_pose_rmsd_angstrom",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    completed = _utc_now_iso()
    success = bool(comparisons)
    warning = ""
    error = ""
    if returncode and success:
        warning = (
            "LDDM exited with a nonzero code, but readable pose comparisons were produced."
        )
    elif not success:
        error = (
            "No source poses could be compared with LDDM samples. Check the run logs, "
            "compound identity, and generated sample files."
        )
    result = {
        "success": success,
        "returncode": int(returncode),
        "source_run_id": source_job.run_id,
        "source_engine": _engine_label(source_job),
        "compared_pose_count": len(comparisons),
        "compared_compound_count": len(summaries),
        "lddm_sample_count": sum(
            len(values) for values in _sampled_poses(run_dir).values()
        ),
        "agreement_threshold_angstrom": 2.0,
        "warning": warning,
        "error": error,
    }
    _write_json(run_dir / "result.json", result)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_path.touch(exist_ok=True)
    stderr_path.touch(exist_ok=True)
    artifacts = [
        ArtifactRef.from_path(
            run_dir, comparison_path, "pose_agreement", role="per_pose_comparisons"
        ),
        ArtifactRef.from_path(
            run_dir, summary_path, "pose_agreement", role="compound_summary"
        ),
        ArtifactRef.from_path(
            run_dir, stdout_path, "job_log", role="stdout", checksum=False
        ),
        ArtifactRef.from_path(
            run_dir, stderr_path, "job_log", role="stderr", checksum=False
        ),
    ]
    generated_poses = run_dir / "results" / "poses.sdf"
    pose_parts: list[str] = []
    for path in sorted((run_dir / "results").glob("**/*_out.sdf")):
        if not path.is_file() or not path.stat().st_size:
            continue
        content = path.read_text(errors="replace").strip()
        if content:
            pose_parts.append(content)
    pose_text = "\n".join(pose_parts) + ("\n" if pose_parts else "")
    if pose_text:
        generated_poses.write_text(pose_text)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                generated_poses,
                "pose_set",
                role="lddm_sampled_poses",
                metadata={"model": "LDDM", "pose_count": result["lddm_sample_count"]},
            )
        )
    write_artifact_manifest(run_dir, artifacts)
    metadata.update(
        {
            "status": "completed" if success else "failed",
            "updated_at": completed,
            "completed_at": completed,
        }
    )
    if warning:
        metadata["warning"] = warning
    if error:
        metadata["error"] = error
    _write_json(metadata_path, metadata)
    return JobRecord.load(
        run_dir, task_group=LDDM_EVALUATION_TASK_GROUP, validate_artifacts=False
    )


def queue_lddm_pose_agreement_job(
    *,
    source_job: JobRecord,
    checkpoint_path: str,
    samples_per_compound: int = 20,
    maximum_compounds: int = 100,
    seed: int = 2026,
    n_steps: int = 100,
    sampler: str = "ForwardEuler",
    sampling_noise: float = 5.0,
    gpu_device: str = "all",
    image: str = DEFAULT_LDDM_IMAGE,
) -> JobRecord:
    if source_job.status != "completed" or source_job.workflow not in {
        "docking_campaign",
        "openvs_docking",
    }:
        raise ValueError("LDDM pose agreement requires a completed docking campaign")
    engine = str(source_job.tool or source_job.metadata.get("engine") or "").lower()
    if source_job.workflow == "docking_campaign" and engine not in {
        "vina",
        "gnina",
        "udp",
        "unidock",
        "unidock_pro",
    }:
        raise ValueError("This docking engine does not have supported source poses")
    if checkpoint_path not in LDDM_CHECKPOINTS.values():
        raise ValueError("Unsupported LDDM checkpoint selection")
    if not (reference_root() / checkpoint_path).is_file():
        raise FileNotFoundError(f"LDDM checkpoint is missing: {checkpoint_path}")
    samples_per_compound = int(samples_per_compound)
    maximum_compounds = max(0, int(maximum_compounds))
    if not 1 <= samples_per_compound <= 100:
        raise ValueError("LDDM samples per compound must be between 1 and 100")
    if int(n_steps) < 1:
        raise ValueError("LDDM integration steps must be positive")
    if not 0.0 <= float(sampling_noise) <= 20.0:
        raise ValueError("LDDM sampling noise must be between 0 and 20")
    if sampler not in {"ForwardEuler", "HeunSampler"}:
        raise ValueError(f"Unsupported LDDM sampler: {sampler}")

    poses = source_pose_inventory(source_job)
    if not poses:
        raise ValueError("The selected docking campaign has no readable ligand poses")
    smiles_by_id = _source_smiles(source_job)
    compound_ids = list(
        dict.fromkeys(str(row.get("compound_id") or "") for row in poses)
    )
    compound_ids = [
        compound_id
        for compound_id in compound_ids
        if compound_id and smiles_by_id.get(compound_id)
    ]
    if maximum_compounds:
        compound_ids = compound_ids[:maximum_compounds]
    if not compound_ids:
        raise ValueError("No source poses have matching compound SMILES")
    poses = [row for row in poses if str(row.get("compound_id") or "") in compound_ids]
    reference, reference_molecule = _source_reference_ligand(
        source_job, poses, smiles_by_id
    )
    receptor = _source_receptor(source_job)

    run_id = str(uuid4())
    run_dir = runs_root() / LDDM_EVALUATION_TASK_GROUP / run_id
    input_dir = run_dir / "input"
    results_dir = run_dir / "results"
    input_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True)
    receptor_local = input_dir / "receptor.pdb"
    receptor_local.write_bytes(receptor.read_bytes())
    reference_local = input_dir / "reference_ligand.sdf"
    if reference is not None:
        if reference.suffix.lower() == ".sdf":
            reference_local.write_bytes(reference.read_bytes())
        else:
            reference_molecule = _reference_molecule(reference)
            if reference_molecule is None:
                raise ValueError("Could not convert the source reference ligand to SDF")
            writer = Chem.SDWriter(str(reference_local))
            writer.write(reference_molecule)
            writer.close()
    else:
        if reference_molecule is None:
            raise ValueError("Could not resolve a pocket reference ligand")
        writer = Chem.SDWriter(str(reference_local))
        writer.write(reference_molecule)
        writer.close()
    compounds_path = input_dir / "compounds.tsv"
    with compounds_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["compound_id", "smiles"])
        writer.writerows((item, smiles_by_id[item]) for item in compound_ids)
    selection_path = input_dir / "source_pose_inventory.csv"
    fields = [
        "compound_id",
        "replicate",
        "source_pose_rank",
        "source_pose_file",
        "source_score_kcal_mol",
        "source_cnn_score",
        "source_cnn_affinity",
    ]
    with selection_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in poses)

    selected_device = str(gpu_device).strip().lower().removeprefix("device=")
    gpu_ids = (
        None
        if selected_device in {"", "all", "automatic"}
        else tuple(int(value) for value in selected_device.split(","))
    )
    image = str(image or DEFAULT_LDDM_IMAGE).strip()
    command = build_docker_command(
        DockerRunSpec(
            tool=registered_tool("lddm_generation", image=image),
            command=(
                "--operation",
                "lddm-docking-campaign",
                "--target",
                "/work/input/receptor.pdb",
                "--reference-ligand",
                "/work/input/reference_ligand.sdf",
                "--checkpoint",
                f"/references/{checkpoint_path}",
                "--docking-compounds",
                "/work/input/compounds.tsv",
                "--output",
                "/work/results",
                "--count",
                str(samples_per_compound),
                "--batch-size",
                str(samples_per_compound),
                "--replicates",
                "1",
                "--seed",
                str(int(seed)),
                "--n-steps",
                str(int(n_steps)),
                "--sampler",
                str(sampler),
                "--sampling-noise",
                str(float(sampling_noise)),
            ),
            mounts=(
                DockerMount(run_dir, "/work"),
                DockerMount(reference_root(), "/references", read_only=True),
            ),
            gpu_enabled=True,
            gpu_devices=gpu_ids,
            use_host_user=False,
        )
    )
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "lddm_pose_agreement",
        "workflow": LDDM_EVALUATION_WORKFLOW,
        "operation": "pose_agreement_evaluation",
        "status": "queued",
        "engine": "LDDM",
        "tool": "LDDM",
        "parent_run_id": source_job.run_id,
        "source_run_id": source_job.run_id,
        "source_engine": _engine_label(source_job),
        "source_task_group": source_job.task_group,
        "compound_count": len(compound_ids),
        "source_pose_count": len(poses),
        "samples_per_compound": samples_per_compound,
        "seed": int(seed),
        "lddm_checkpoint_path": checkpoint_path,
        "lddm_n_steps": int(n_steps),
        "lddm_sampler": str(sampler),
        "lddm_sampling_noise": float(sampling_noise),
        "pose_rmsd_frame": "fixed_receptor_frame_symmetry_aware_heavy_atom",
        "affinity_estimated": False,
        "confidence_semantics": "mean_positive_atom_sigma_x_lower_is_more_confident",
        "docker_image": image,
        "gpu_device": str(gpu_device),
        "created_at": now,
        "updated_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "source_run_id": source_job.run_id,
            "source_engine": _engine_label(source_job),
            "source_pose_count": len(poses),
            "compound_ids": compound_ids,
            "parameters": {
                key: metadata[key]
                for key in (
                    "samples_per_compound",
                    "seed",
                    "lddm_checkpoint_path",
                    "lddm_n_steps",
                    "lddm_sampler",
                    "lddm_sampling_noise",
                )
            },
            "command": [value.replace(str(run_dir), "<run-dir>") for value in command],
        },
    )
    tool = registered_tool("lddm_generation", image=image)
    resources = tool.resources.to_dict()
    if gpu_ids:
        resources["gpu_ids"] = list(gpu_ids)
    write_registered_command_record(
        run_dir,
        tool_id="lddm_generation",
        commands=(command,),
        image=image,
        selected_gpu_ids=gpu_ids or (),
    )
    metadata.update(
        {
            "queued_at": now,
            "queued_command": command,
            "gpu_queued": True,
            "resources": resources,
            "worker_finalizer": "lddm_pose_agreement",
        }
    )
    _write_json(run_dir / "metadata.json", metadata)
    write_artifact_manifest(run_dir, [])
    return JobRecord.load(
        run_dir, task_group=LDDM_EVALUATION_TASK_GROUP, validate_artifacts=False
    )
