from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JobRecord
from mn_ligand.workflows.target_orientation import (
    apply_coordinate_transform,
    create_axis_aligned_target_job,
    ligand_longest_axis_transform,
    transform_pdb_data,
)


LIGAND_SDF = """axis-ligand
  test

  3  2  0  0  0  0  0  0  0  0999 V2000
    5.0000   10.0000    2.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.0000   14.0000    2.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0000   12.0000    2.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
M  END
$$$$
"""

TARGET_PDB = """ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C
ATOM      2  CA  GLY A   2       4.000   6.000   3.000  1.00 20.00           C
TER
END
"""


def _pdb_coordinates(data: str) -> np.ndarray:
    return np.asarray(
        [
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            for line in data.splitlines()
            if line.startswith(("ATOM  ", "HETATM"))
        ]
    )


def test_longest_axis_transform_is_shared_and_keeps_ligand_centroid(
    tmp_path: Path,
) -> None:
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text(LIGAND_SDF)
    transform = ligand_longest_axis_transform(ligand)

    molecule = next(
        item
        for item in Chem.SDMolSupplier(
            str(ligand), removeHs=False, sanitize=False
        )
        if item is not None
    )
    ligand_coordinates = np.asarray(
        molecule.GetConformer().GetPositions(), dtype=float
    )
    transformed_ligand = apply_coordinate_transform(
        ligand_coordinates, transform
    )
    centered = transformed_ligand - transformed_ligand.mean(axis=0)
    _, axes = np.linalg.eigh(centered.T @ centered)
    longest_axis = axes[:, -1]

    assert abs(float(longest_axis[0])) > 0.999
    assert np.allclose(
        transformed_ligand.mean(axis=0),
        ligand_coordinates.mean(axis=0),
        atol=1e-10,
    )

    transformed_pdb = transform_pdb_data(TARGET_PDB, transform)
    before = _pdb_coordinates(TARGET_PDB)
    after = _pdb_coordinates(transformed_pdb)
    expected = apply_coordinate_transform(before, transform)
    assert np.allclose(after, expected, atol=0.001)
    assert np.isclose(
        np.linalg.norm(after[1] - after[0]),
        np.linalg.norm(before[1] - before[0]),
        atol=0.002,
    )


def test_axis_alignment_creates_typed_immutable_target_and_ligand(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    source_dir = runs / "structure-jobs" / "source-target"
    source_dir.mkdir(parents=True)
    target = source_dir / "receptor.pdb"
    ligand = source_dir / "ligand.sdf"
    target.write_text(TARGET_PDB)
    ligand.write_text(LIGAND_SDF)
    (source_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": source_dir.name,
                "job_code": "SRC01",
                "status": "completed",
                "workflow": "structure_preparation",
                "operation": "preparation",
                "created_at": "2026-07-25T00:00:00+00:00",
            }
        )
    )
    target_artifact = ArtifactRef.from_path(
        source_dir, target, "prepared_target", role="receptor"
    )
    ligand_artifact = ArtifactRef.from_path(
        source_dir, ligand, "prepared_ligand_set", role="ligand"
    )
    write_artifact_manifest(
        source_dir, [target_artifact, ligand_artifact]
    )
    source_job = JobRecord.load(source_dir, task_group="structure-jobs")

    aligned = create_axis_aligned_target_job(
        source_job=source_job,
        source_artifact=target_artifact,
        source_path=target,
        axis_ligand_path=ligand,
        axis_ligand_artifact=ligand_artifact,
    )

    assert aligned.status == "completed"
    assert aligned.task_group == "target-orientation"
    assert aligned.metadata["coordinate_transform"]["aligned_longest_axis"] == [
        1.0,
        0.0,
        0.0,
    ]
    assert aligned.artifact_manifest is not None
    assert len(aligned.artifact_manifest.by_type("prepared_target")) == 1
    assert len(aligned.artifact_manifest.by_type("prepared_ligand_set")) == 1
    assert (
        aligned.artifact_manifest.by_type("prepared_target")[0].path
        == "artifacts/target_longest_axis_x.pdb"
    )

    reused = create_axis_aligned_target_job(
        source_job=source_job,
        source_artifact=target_artifact,
        source_path=target,
        axis_ligand_path=ligand,
        axis_ligand_artifact=ligand_artifact,
    )
    assert reused.run_id == aligned.run_id
