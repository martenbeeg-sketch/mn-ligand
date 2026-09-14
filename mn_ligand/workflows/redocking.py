from __future__ import annotations

import csv
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from rdkit import Chem
from rdkit.Chem import rdFMCS

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.workflows import (
    TERMINAL_STATES,
    WorkflowInput,
    WorkflowRecord,
    attach_workflow_child,
    create_workflow,
    refresh_workflow,
)
from mn_ligand.runtime import resolve_run_dir, runs_root
from mn_ligand.workflows.docking import DEFAULT_DOCKING_IMAGE, queue_docking_campaign_job


REDOCKING_ENGINES = ("vina", "gnina", "udp")
_VINA_SCORE = re.compile(r"REMARK VINA RESULT:\s*(-?\d+(?:\.\d+)?)")
_GNINA_SCORE = re.compile(r"REMARK\s+minimizedAffinity\s+(-?\d+(?:\.\d+)?)")
_CNN_SCORE = re.compile(r"REMARK\s+CNNscore\s+(-?\d+(?:\.\d+)?)")
_CNN_AFFINITY = re.compile(r"REMARK\s+CNNaffinity\s+(-?\d+(?:\.\d+)?)")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except (OSError, TypeError, ValueError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _float_match(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    return float(match.group(1)) if match else None


def _pdbqt_models(path: Path) -> list[list[str]]:
    models: list[list[str]] = []
    current: list[str] = []
    saw_model = False
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("MODEL"):
            saw_model = True
            if current:
                models.append(current)
            current = [line]
        elif line.startswith("ENDMDL"):
            current.append(line)
            models.append(current)
            current = []
        else:
            current.append(line)
    if current and (not saw_model or any(line.startswith(("ATOM  ", "HETATM")) for line in current)):
        models.append(current)
    return models


def _reference_molecule(path: Path) -> Chem.Mol:
    if path.suffix.lower() != ".sdf":
        raise ValueError("Redocking requires a coordinate-bearing reference ligand SDF")
    molecules = [molecule for molecule in Chem.SDMolSupplier(str(path), removeHs=False) if molecule]
    if len(molecules) != 1:
        raise ValueError("Redocking requires exactly one valid reference ligand molecule")
    molecule = Chem.RemoveHs(molecules[0])
    if molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D():
        raise ValueError("The reference ligand must contain one three-dimensional conformer")
    return molecule


def _model_payload(lines: Sequence[str]) -> tuple[Chem.Mol, dict[int, tuple[float, float, float]], dict[int, int]]:
    smiles_line = next((line for line in lines if line.startswith("REMARK SMILES ")), "")
    if not smiles_line:
        raise ValueError("Docking pose has no Meeko SMILES atom map")
    docked = Chem.MolFromSmiles(smiles_line.split(" ", 2)[2])
    if docked is None:
        raise ValueError("Docking pose contains an unreadable mapped SMILES")
    mapping_numbers: list[int] = []
    for line in lines:
        if line.startswith("REMARK SMILES IDX "):
            mapping_numbers.extend(int(value) for value in line.split()[3:])
    if not mapping_numbers or len(mapping_numbers) % 2:
        raise ValueError("Docking pose has an invalid Meeko SMILES index map")
    serial_for_smiles = {
        mapping_numbers[index] - 1: mapping_numbers[index + 1]
        for index in range(0, len(mapping_numbers), 2)
    }
    coordinates: dict[int, tuple[float, float, float]] = {}
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        coordinates[int(line[6:11])] = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    return docked, coordinates, serial_for_smiles


def _symmetry_rmsd(
    reference: Chem.Mol,
    lines: Sequence[str],
) -> tuple[float, Chem.Mol]:
    docked, coordinates, serial_for_smiles = _model_payload(lines)
    mcs = rdFMCS.FindMCS(
        [reference, docked],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=15,
    )
    query = Chem.MolFromSmarts(mcs.smartsString)
    expected_atoms = reference.GetNumHeavyAtoms()
    if (
        query is None
        or query.GetNumAtoms() != expected_atoms
        or docked.GetNumHeavyAtoms() != expected_atoms
    ):
        matched = query.GetNumAtoms() if query is not None else 0
        raise ValueError(
            f"Reference/pose identity mismatch: mapped {matched} of {expected_atoms} heavy atoms"
        )
    reference_matches = reference.GetSubstructMatches(query, uniquify=False, maxMatches=10000)
    docked_matches = docked.GetSubstructMatches(query, uniquify=False, maxMatches=10000)
    reference_conf = reference.GetConformer()
    best_rmsd = math.inf
    best_coordinates: dict[int, tuple[float, float, float]] = {}
    for reference_match in reference_matches:
        for docked_match in docked_matches:
            square_distance = 0.0
            candidate_coordinates: dict[int, tuple[float, float, float]] = {}
            for reference_index, docked_index in zip(reference_match, docked_match):
                serial = serial_for_smiles.get(docked_index)
                if serial is None or serial not in coordinates:
                    raise ValueError("Docking pose atom map does not resolve every heavy atom")
                predicted = coordinates[serial]
                observed = reference_conf.GetAtomPosition(reference_index)
                square_distance += (
                    (observed.x - predicted[0]) ** 2
                    + (observed.y - predicted[1]) ** 2
                    + (observed.z - predicted[2]) ** 2
                )
                candidate_coordinates[reference_index] = predicted
            rmsd = math.sqrt(square_distance / len(reference_match))
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_coordinates = candidate_coordinates
    if not math.isfinite(best_rmsd):
        raise ValueError("No symmetry-aware atom mapping could be evaluated")
    predicted = Chem.Mol(reference)
    conformer = Chem.Conformer(predicted.GetNumAtoms())
    for atom_index in range(predicted.GetNumAtoms()):
        point = best_coordinates.get(atom_index)
        if point is None:
            observed = reference_conf.GetAtomPosition(atom_index)
            point = (observed.x, observed.y, observed.z)
        conformer.SetAtomPosition(atom_index, point)
    predicted.RemoveAllConformers()
    predicted.AddConformer(conformer, assignId=True)
    return best_rmsd, predicted


def redocking_pose_metrics(reference: Chem.Mol, pose_path: Path) -> tuple[list[dict[str, Any]], Chem.Mol]:
    rows: list[dict[str, Any]] = []
    top_pose: Chem.Mol | None = None
    for rank, lines in enumerate(_pdbqt_models(pose_path), start=1):
        text = "\n".join(lines)
        rmsd, predicted = _symmetry_rmsd(reference, lines)
        if top_pose is None:
            top_pose = predicted
        rows.append(
            {
                "pose_rank": rank,
                "score_kcal_mol": _float_match(_VINA_SCORE, text)
                if _VINA_SCORE.search(text)
                else _float_match(_GNINA_SCORE, text),
                "cnn_score": _float_match(_CNN_SCORE, text),
                "cnn_affinity": _float_match(_CNN_AFFINITY, text),
                "symmetry_rmsd_angstrom": rmsd,
            }
        )
    if not rows or top_pose is None:
        raise ValueError(f"No readable docking models found in {pose_path.name}")
    return rows, top_pose


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.mean(present) if present else None


def _sample_sd(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return statistics.stdev(present) if len(present) > 1 else 0.0


def _summary_rows(replicates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for engine in REDOCKING_ENGINES:
        selected = [row for row in replicates if row["engine"] == engine]
        if not selected:
            continue
        top_rmsd = [row["top_pose_rmsd_angstrom"] for row in selected]
        best_rmsd = [row["best_pose_rmsd_angstrom"] for row in selected]
        rows.append(
            {
                "engine": engine,
                "replicate_count": len(selected),
                "mean_top_score_kcal_mol": _mean(row["top_score_kcal_mol"] for row in selected),
                "sd_top_score_kcal_mol": _sample_sd(row["top_score_kcal_mol"] for row in selected),
                "mean_top_rmsd_angstrom": statistics.mean(top_rmsd),
                "sd_top_rmsd_angstrom": _sample_sd(top_rmsd),
                "minimum_top_rmsd_angstrom": min(top_rmsd),
                "maximum_top_rmsd_angstrom": max(top_rmsd),
                "mean_best_of_n_rmsd_angstrom": statistics.mean(best_rmsd),
                "sd_best_of_n_rmsd_angstrom": _sample_sd(best_rmsd),
                "top_pose_recovery_at_1a": sum(value <= 1.0 for value in top_rmsd) / len(top_rmsd),
                "top_pose_recovery_at_2a": sum(value <= 2.0 for value in top_rmsd) / len(top_rmsd),
                "mean_cnn_score": _mean(row["top_cnn_score"] for row in selected),
                "sd_cnn_score": _sample_sd(row["top_cnn_score"] for row in selected),
                "mean_cnn_affinity": _mean(row["top_cnn_affinity"] for row in selected),
                "sd_cnn_affinity": _sample_sd(row["top_cnn_affinity"] for row in selected),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _reference_input(workflow: WorkflowRecord) -> tuple[Path, Chem.Mol]:
    for item in workflow.inputs:
        if item.artifact.artifact_type not in {
            "prepared_ligand_set",
            "reference_ligand",
            "benchmark_reference_ligand",
        }:
            continue
        source_dir = resolve_run_dir(item.source_task_group, item.artifact.run_id)
        path = item.artifact.resolve(source_dir, must_exist=True) if source_dir else None
        if path is not None:
            return path, _reference_molecule(path)
    raise ValueError("The typed crystallographic reference ligand is unavailable")


def finalize_redocking_benchmark(workflow_id: str) -> JobRecord:
    workflow = refresh_workflow(workflow_id)
    parent = JobRecord.load(workflow.run_dir, task_group="workflows")
    if workflow.status not in TERMINAL_STATES:
        return parent
    if bool(parent.result.get("redocking_finalized")):
        return parent

    errors: list[str] = []
    pose_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    artifacts: list[ArtifactRef] = []
    try:
        _, reference = _reference_input(workflow)
    except (OSError, ValueError) as exc:
        reference = None
        errors.append(str(exc))

    overlay_dir = workflow.run_dir / "overlays"
    for child in workflow.children:
        child_dir = resolve_run_dir(child.task_group, child.run_id)
        child_job = JobRecord.load(child_dir, task_group=child.task_group) if child_dir else None
        if child_job is None or child_job.status != "completed":
            errors.append(f"{child.step_id}: {child_job.status if child_job else 'missing'}")
            continue
        engine = str(child_job.metadata.get("engine") or "")
        replicate = int(child_job.metadata.get("benchmark_replicate") or 1)
        pose_paths = sorted((child_job.run_dir / "results").rglob("*_out.pdbqt"))
        if reference is None or len(pose_paths) != 1:
            errors.append(f"{child.step_id}: expected exactly one native PDBQT pose file")
            continue
        try:
            metrics, predicted = redocking_pose_metrics(reference, pose_paths[0])
        except (OSError, ValueError) as exc:
            errors.append(f"{child.step_id}: {exc}")
            continue
        for row in metrics:
            pose_rows.append(
                {
                    "engine": engine,
                    "replicate": replicate,
                    "run_id": child.run_id,
                    **row,
                }
            )
        top = metrics[0]
        replicate_rows.append(
            {
                "engine": engine,
                "replicate": replicate,
                "run_id": child.run_id,
                "pose_count": len(metrics),
                "top_score_kcal_mol": top["score_kcal_mol"],
                "top_cnn_score": top["cnn_score"],
                "top_cnn_affinity": top["cnn_affinity"],
                "top_pose_rmsd_angstrom": top["symmetry_rmsd_angstrom"],
                "best_pose_rmsd_angstrom": min(row["symmetry_rmsd_angstrom"] for row in metrics),
            }
        )
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{engine}-replicate-{replicate}.sdf"
        observed = Chem.Mol(reference)
        observed.SetProp("_Name", "crystallographic_reference")
        predicted.SetProp("_Name", f"{engine}_replicate_{replicate}_top_pose")
        writer = Chem.SDWriter(str(overlay_path))
        writer.write(observed)
        writer.write(predicted)
        writer.close()
        artifacts.append(
            ArtifactRef.from_path(
                workflow.run_dir,
                overlay_path,
                "redocking_overlay",
                role=f"{engine}_replicate_{replicate}",
                metadata={"engine": engine, "replicate": replicate},
            )
        )

    pose_fields = (
        "engine", "replicate", "run_id", "pose_rank", "score_kcal_mol",
        "cnn_score", "cnn_affinity", "symmetry_rmsd_angstrom",
    )
    replicate_fields = (
        "engine", "replicate", "run_id", "pose_count", "top_score_kcal_mol",
        "top_cnn_score", "top_cnn_affinity", "top_pose_rmsd_angstrom",
        "best_pose_rmsd_angstrom",
    )
    summary_fields = (
        "engine", "replicate_count", "mean_top_score_kcal_mol", "sd_top_score_kcal_mol",
        "mean_top_rmsd_angstrom", "sd_top_rmsd_angstrom", "minimum_top_rmsd_angstrom",
        "maximum_top_rmsd_angstrom", "mean_best_of_n_rmsd_angstrom",
        "sd_best_of_n_rmsd_angstrom", "top_pose_recovery_at_1a",
        "top_pose_recovery_at_2a", "mean_cnn_score", "sd_cnn_score",
        "mean_cnn_affinity", "sd_cnn_affinity",
    )
    summary_rows = _summary_rows(replicate_rows)
    metrics_dir = workflow.run_dir / "metrics"
    pose_path = metrics_dir / "redocking_poses.csv"
    replicate_path = metrics_dir / "redocking_replicates.csv"
    summary_path = metrics_dir / "redocking_summary.csv"
    _write_csv(pose_path, pose_rows, pose_fields)
    _write_csv(replicate_path, replicate_rows, replicate_fields)
    _write_csv(summary_path, summary_rows, summary_fields)
    artifacts.extend(
        (
            ArtifactRef.from_path(workflow.run_dir, pose_path, "redocking_metrics", role="per_pose"),
            ArtifactRef.from_path(workflow.run_dir, replicate_path, "redocking_metrics", role="replicates"),
            ArtifactRef.from_path(workflow.run_dir, summary_path, "redocking_summary", role="summary"),
        )
    )
    write_artifact_manifest(workflow.run_dir, artifacts)

    success = workflow.status == "completed" and not errors and len(replicate_rows) == len(workflow.children)
    completed_at = _utc_now_iso()
    workflow_payload = _read_json(workflow.run_dir / "workflow.json")
    workflow_payload.update(
        {
            "status": "completed" if success else "failed",
            "updated_at": completed_at,
            "completed_at": completed_at,
        }
    )
    _write_json(workflow.run_dir / "workflow.json", workflow_payload)
    metadata = _read_json(workflow.run_dir / "metadata.json")
    metadata.update(
        {
            "status": "completed" if success else "failed",
            "operation": "redocking",
            "tool": "Redocking benchmark",
            "updated_at": completed_at,
            "completed_at": completed_at,
        }
    )
    if errors:
        metadata["error"] = "; ".join(errors)
    _write_json(workflow.run_dir / "metadata.json", metadata)
    generic_result = _read_json(workflow.run_dir / "result.json")
    _write_json(
        workflow.run_dir / "result.json",
        {
            **generic_result,
            "success": success,
            "redocking_finalized": True,
            "replicate_count": len(replicate_rows),
            "pose_count": len(pose_rows),
            "engine_count": len(summary_rows),
            "rmsd_method": "symmetry-aware heavy-atom direct RMSD in the shared receptor frame",
            "summary": summary_rows,
            "error": "" if success else "; ".join(errors) or "Redocking benchmark failed",
        },
    )
    return JobRecord.load(workflow.run_dir, task_group="workflows")


def queue_redocking_benchmark(
    *,
    receptor_path: Path,
    target_artifact: ArtifactRef,
    target_task_group: str,
    reference_ligand_path: Path,
    reference_ligand_artifact: ArtifactRef,
    reference_task_group: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    box_mode: str = "fixed",
    box_padding_angstrom: float | None = None,
    engines: Sequence[str] = REDOCKING_ENGINES,
    replicates: int = 3,
    seed_start: int = 1001,
    image: str = DEFAULT_DOCKING_IMAGE,
    gpu_device: str = "all",
    search_mode: str = "detail",
    exhaustiveness: int = 30,
    poses: int = 10,
    use_scrub: bool = True,
    scrub_ph: float = 7.4,
    scrub_skip_tautomer: bool = True,
    context_metadata: dict[str, Any] | None = None,
) -> JobRecord:
    safe_engines = tuple(dict.fromkeys(str(engine).strip().lower() for engine in engines))
    if not safe_engines or any(engine not in REDOCKING_ENGINES for engine in safe_engines):
        raise ValueError("Select at least one supported redocking engine")
    replicate_count = int(replicates)
    if not 1 <= replicate_count <= 20:
        raise ValueError("Redocking replicates must be between 1 and 20")
    seed_start = int(seed_start)
    if seed_start < 1 or seed_start + replicate_count - 1 >= 2_147_483_647:
        raise ValueError("Redocking seed range must contain positive 32-bit integers")
    _reference_molecule(Path(reference_ligand_path))
    expected_steps = tuple(
        f"{engine}_replicate_{replicate}"
        for engine in safe_engines
        for replicate in range(1, replicate_count + 1)
    )
    context = dict(context_metadata or {})
    workflow = create_workflow(
        "redocking_benchmark",
        name="Redocking benchmark",
        parameters={
            "operation": "redocking",
            "engines": list(safe_engines),
            "replicates": replicate_count,
            "seed_start": seed_start,
            "center": {axis: float(value) for axis, value in zip("xyz", center)},
            "size": {axis: float(value) for axis, value in zip("xyz", size)},
            "box_mode": str(box_mode),
            "box_padding_angstrom": (
                float(box_padding_angstrom) if box_padding_angstrom is not None else None
            ),
            "exhaustiveness": int(exhaustiveness),
            "poses_per_replicate": int(poses),
            "search_mode": str(search_mode),
            "rmsd_method": "symmetry-aware heavy-atom direct RMSD in the shared receptor frame",
            "context": context,
        },
        inputs=(
            WorkflowInput(source_task_group=target_task_group, artifact=target_artifact),
            WorkflowInput(source_task_group=reference_task_group, artifact=reference_ligand_artifact),
        ),
        expected_steps=expected_steps,
    )
    for engine in safe_engines:
        for replicate in range(1, replicate_count + 1):
            child = queue_docking_campaign_job(
                receptor_path=Path(receptor_path),
                target_artifact=target_artifact,
                compound_paths=(Path(reference_ligand_path),),
                compound_artifacts=(reference_ligand_artifact,),
                center=center,
                size=size,
                box_mode=box_mode,
                box_padding_angstrom=box_padding_angstrom,
                engine=engine,
                image=image,
                gpu_device=gpu_device,
                mode="classic",
                search_mode=search_mode,
                exhaustiveness=exhaustiveness,
                poses=poses,
                use_scrub=use_scrub,
                scrub_ph=scrub_ph,
                scrub_skip_tautomer=scrub_skip_tautomer,
                maximum_compounds=1,
                seed_start=seed_start + replicate - 1,
            )
            step_id = f"{engine}_replicate_{replicate}"
            attach_workflow_child(workflow.workflow_id, child, step_id=step_id)
            child_metadata = _read_json(child.run_dir / "metadata.json")
            child_metadata.update(
                {
                    "benchmark_replicate": replicate,
                    "redocking_reference_run_id": reference_ligand_artifact.run_id,
                    "redocking_reference_artifact_id": reference_ligand_artifact.artifact_id,
                    **context,
                }
            )
            _write_json(child.run_dir / "metadata.json", child_metadata)
    return JobRecord.load(workflow.run_dir, task_group="workflows")


def advance_redocking_benchmarks() -> None:
    workflow_root = runs_root() / "workflows"
    if not workflow_root.is_dir():
        return
    for run_dir in sorted(path for path in workflow_root.iterdir() if path.is_dir()):
        payload = _read_json(run_dir / "workflow.json")
        if payload.get("workflow_type") != "redocking_benchmark":
            continue
        result = _read_json(run_dir / "result.json")
        if result.get("redocking_finalized"):
            continue
        finalize_redocking_benchmark(run_dir.name)
