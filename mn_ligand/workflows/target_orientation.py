from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
from rdkit import Chem

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import (
    JOB_SCHEMA_VERSION,
    JobRecord,
    iter_job_records,
    short_job_code,
)
from mn_ligand.core.provenance import inherited_target_metadata, modification_history
from mn_ligand.runtime import runs_root
from mn_ligand.ligandx.lib.chemistry.preparation.target_validation import (
    audit_target_geometry,
    prepare_target_for_publication,
)


TASK_GROUP = "target-orientation"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_identity(value: ArtifactRef | dict[str, Any]) -> tuple[str, str, str]:
    payload = value.to_dict() if isinstance(value, ArtifactRef) else value
    return (
        str(payload.get("run_id") or ""),
        str(payload.get("artifact_id") or ""),
        str(payload.get("sha256") or ""),
    )


def reusable_axis_aligned_target_job(
    *,
    source_artifact: ArtifactRef,
    axis_ligand_artifact: ArtifactRef,
    additional_ligands: Sequence[tuple[Path, ArtifactRef]] = (),
) -> JobRecord | None:
    """Return an exact, still-valid orientation instead of rebuilding it."""
    expected_additional = tuple(
        _artifact_identity(artifact) for _, artifact in additional_ligands
    )
    candidates = sorted(
        (
            job
            for job in iter_job_records(runs_root())
            if job.workflow == "target_orientation"
            and job.status == "completed"
        ),
        key=lambda job: str(job.metadata.get("created_at") or ""),
        reverse=True,
    )
    for job in candidates:
        try:
            payload = json.loads((job.run_dir / "input.json").read_text())
        except (OSError, TypeError, ValueError):
            continue
        if _artifact_identity(payload.get("source_target") or {}) != (
            _artifact_identity(source_artifact)
        ):
            continue
        if _artifact_identity(payload.get("axis_ligand") or {}) != (
            _artifact_identity(axis_ligand_artifact)
        ):
            continue
        observed_additional = tuple(
            _artifact_identity(value)
            for value in (payload.get("additional_ligands") or [])
            if isinstance(value, dict)
        )
        if observed_additional != expected_additional:
            continue
        targets = (
            job.artifact_manifest.by_type("prepared_target")
            if job.artifact_manifest is not None
            else ()
        )
        if len(targets) != 1:
            continue
        target_path = targets[0].resolve(job.run_dir, must_exist=True)
        if target_path is None or target_path.suffix.lower() not in {".pdb", ".ent"}:
            continue
        if not audit_target_geometry(
            target_path.read_text(errors="replace")
        )["valid"]:
            continue
        return job
    return None


def _ligand_molecules(path: Path) -> list[Chem.Mol]:
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        molecules = [
            molecule
            for molecule in Chem.SDMolSupplier(
                str(path), removeHs=False, sanitize=False
            )
            if molecule is not None and molecule.GetNumConformers()
        ]
    elif suffix == ".mol":
        molecule = Chem.MolFromMolFile(
            str(path), removeHs=False, sanitize=False
        )
        molecules = [molecule] if molecule is not None else []
    elif suffix == ".mol2":
        molecule = Chem.MolFromMol2File(
            str(path), removeHs=False, sanitize=False
        )
        molecules = [molecule] if molecule is not None else []
    elif suffix in {".pdb", ".ent"}:
        molecule = Chem.MolFromPDBFile(
            str(path), removeHs=False, sanitize=False
        )
        molecules = [molecule] if molecule is not None else []
    else:
        raise ValueError(
            "Ligand-axis alignment requires SDF, MOL, MOL2, or PDB coordinates"
        )
    if not molecules:
        raise ValueError(f"No coordinate-bearing ligand found in {path.name}")
    return molecules


def _rotation_from_vectors(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    destination = destination / np.linalg.norm(destination)
    cross = np.cross(source, destination)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(source, destination))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        perpendicular = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(source, perpendicular))) > 0.9:
            perpendicular = np.array([0.0, 0.0, 1.0])
        axis = np.cross(source, perpendicular)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / (sine * sine))


def apply_coordinate_transform(
    coordinates: np.ndarray,
    transform: dict[str, Any],
) -> np.ndarray:
    pivot = np.asarray(transform["pivot"], dtype=float)
    rotation = np.asarray(transform["rotation"], dtype=float)
    return (np.asarray(coordinates, dtype=float) - pivot) @ rotation.T + pivot


def ligand_longest_axis_transform(path: Path) -> dict[str, Any]:
    molecule = _ligand_molecules(path)[0]
    coordinates = np.asarray(
        molecule.GetConformer().GetPositions(), dtype=float
    )
    heavy_indices = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if heavy_indices:
        coordinates = coordinates[heavy_indices]
    if len(coordinates) < 2:
        raise ValueError("At least two ligand heavy atoms are required for axis alignment")
    pivot = coordinates.mean(axis=0)
    centered = coordinates - pivot
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
    rotation = _rotation_from_vectors(axis, np.array([1.0, 0.0, 0.0]))
    transformed = apply_coordinate_transform(
        coordinates,
        {"pivot": pivot.tolist(), "rotation": rotation.tolist()},
    )
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    return {
        "pivot": [float(value) for value in pivot],
        "rotation": [
            [float(value) for value in row]
            for row in rotation
        ],
        "source_longest_axis": [float(value) for value in axis],
        "aligned_longest_axis": [1.0, 0.0, 0.0],
        "ligand_box": {
            "center": [
                float(value) for value in (minimum + maximum) / 2.0
            ],
            "size": [
                float(value) for value in np.maximum(maximum - minimum, 1.0)
            ],
        },
    }


def transform_axis_aligned_box(
    box: dict[str, tuple[float, float, float]],
    transform: dict[str, Any],
) -> dict[str, tuple[float, float, float]]:
    center = np.asarray(box["center"], dtype=float)
    half_size = np.asarray(box["size"], dtype=float) / 2.0
    corners = np.asarray(
        [
            center + np.asarray((x, y, z), dtype=float) * half_size
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )
    transformed = apply_coordinate_transform(corners, transform)
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    return {
        "center": tuple(float(value) for value in (minimum + maximum) / 2.0),
        "size": tuple(float(value) for value in maximum - minimum),
    }


def transform_pdb_data(pdb_data: str, transform: dict[str, Any]) -> str:
    output: list[str] = []
    transformed_count = 0
    for line in pdb_data.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 54:
            try:
                coordinate = np.asarray(
                    [[
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]]
                )
            except ValueError:
                output.append(line)
                continue
            x, y, z = apply_coordinate_transform(coordinate, transform)[0]
            line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            transformed_count += 1
        output.append(line)
    if transformed_count == 0:
        raise ValueError("The prepared target contains no PDB coordinate records")
    return "\n".join(output) + "\n"


def transformed_ligand_sdf(path: Path, transform: dict[str, Any]) -> str:
    molecules = _ligand_molecules(path)
    for molecule in molecules:
        conformer = molecule.GetConformer()
        coordinates = np.asarray(conformer.GetPositions(), dtype=float)
        transformed = apply_coordinate_transform(coordinates, transform)
        for index, (x, y, z) in enumerate(transformed):
            conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
    blocks = [Chem.MolToMolBlock(molecule) for molecule in molecules]
    records: list[str] = []
    for molecule, block in zip(molecules, blocks):
        properties = []
        for name in molecule.GetPropNames(includePrivate=False, includeComputed=False):
            properties.append(f">  <{name}>\n{molecule.GetProp(name)}\n")
        records.append(block + "\n" + "\n".join(properties) + "\n$$$$\n")
    return "".join(records)


def create_axis_aligned_target_job(
    *,
    source_job: JobRecord,
    source_artifact: ArtifactRef,
    source_path: Path,
    axis_ligand_path: Path,
    axis_ligand_artifact: ArtifactRef,
    additional_ligands: Sequence[tuple[Path, ArtifactRef]] = (),
) -> JobRecord:
    if source_path.suffix.lower() not in {".pdb", ".ent"}:
        raise ValueError("Ligand-axis target orientation currently requires a PDB target")
    reusable = reusable_axis_aligned_target_job(
        source_artifact=source_artifact,
        axis_ligand_artifact=axis_ligand_artifact,
        additional_ligands=additional_ligands,
    )
    if reusable is not None:
        return reusable
    transform = ligand_longest_axis_transform(axis_ligand_path)
    run_id = str(uuid4())
    run_dir = runs_root() / TASK_GROUP / run_id
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    target_path = artifact_dir / "target_longest_axis_x.pdb"
    validated_target, target_validation = prepare_target_for_publication(
        transform_pdb_data(source_path.read_text(errors="replace"), transform)
    )
    target_path.write_text(validated_target)
    ligand_sources = [
        (axis_ligand_path, axis_ligand_artifact, "axis_ligand"),
        *[
            (path, artifact, f"coordinate_ligand_{index}")
            for index, (path, artifact) in enumerate(additional_ligands, start=1)
        ],
    ]
    ligand_outputs: list[tuple[Path, ArtifactRef, str]] = []
    for index, (path, artifact, role) in enumerate(ligand_sources, start=1):
        output = artifact_dir / f"ligand_{index}_longest_axis_x.sdf"
        output.write_text(transformed_ligand_sdf(path, transform))
        ligand_outputs.append((output, artifact, role))

    now = _utc_now_iso()
    jobs_by_id = {job.run_id: job for job in iter_job_records(runs_root())}
    jobs_by_id[source_job.run_id] = source_job
    job_code = short_job_code(run_id)
    history = [
        *modification_history(source_job, jobs_by_id),
        {
            "run_id": run_id,
            "job_code": job_code,
            "kind": "target_orientation",
            "label": "Ligand-axis target orientation",
            "tool": "principal-axis rigid transform",
            "summary": (
                "Rigidly rotated the protein and coordinate ligands together so "
                "the associated ligand's longest principal axis lies on global X"
            ),
        },
    ]
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": job_code,
        "job_type": "target_orientation",
        "workflow": "target_orientation",
        "operation": "preparation",
        "tool": "principal-axis rigid transform",
        "status": "completed",
        "parent_run_id": source_job.run_id,
        "source_target_run_id": source_job.run_id,
        "prepared_target_run_id": run_id,
        "coordinate_transform": transform,
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
        **inherited_target_metadata(source_job, jobs_by_id),
        "modification_history": history,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "input.json").write_text(
        json.dumps(
            {
                "source_target": source_artifact.to_dict(),
                "axis_ligand": axis_ligand_artifact.to_dict(),
                "additional_ligands": [
                    artifact.to_dict() for _, artifact in additional_ligands
                ],
                "parameters": {
                    "align_ligand_longest_axis_to": "x",
                    "shared_rigid_transform": True,
                },
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
                "operation": "shared principal-axis rigid transform",
                "transform": transform,
            },
            indent=2,
        )
        + "\n"
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "success": True,
                "prepared_target": "artifacts/target_longest_axis_x.pdb",
                "coordinate_transform": transform,
                "target_validation": target_validation,
            },
            indent=2,
        )
        + "\n"
    )
    artifacts = [
        ArtifactRef.from_path(
            run_dir,
            target_path,
            "prepared_target",
            role="axis_aligned_receptor",
            metadata={
                "source_run_id": source_job.run_id,
                "source_artifact_id": source_artifact.artifact_id,
                "coordinate_transform": transform,
            },
        )
    ]
    artifacts.extend(
        ArtifactRef.from_path(
            run_dir,
            output,
            "prepared_ligand_set",
            role=role,
            metadata={
                "source_run_id": source_artifact_ref.run_id,
                "source_artifact_id": source_artifact_ref.artifact_id,
                "coordinate_transform": transform,
            },
        )
        for output, source_artifact_ref, role in ligand_outputs
    )
    write_artifact_manifest(run_dir, artifacts)
    return JobRecord.load(run_dir, task_group=TASK_GROUP)
