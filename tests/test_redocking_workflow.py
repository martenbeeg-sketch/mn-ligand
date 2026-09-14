from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from rdkit import Chem

from mn_ligand.core.artifacts import ArtifactRef
from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.workflows import WorkflowRecord
from mn_ligand.workflows.redocking import (
    finalize_redocking_benchmark,
    queue_redocking_benchmark,
    redocking_pose_metrics,
)


def _reference_sdf(path: Path) -> Chem.Mol:
    molecule = Chem.MolFromSmiles("CCO")
    assert molecule is not None
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, (1.5, 0.0, 0.0))
    conformer.SetAtomPosition(2, (2.5, 0.0, 0.0))
    molecule.AddConformer(conformer)
    molecule.SetProp("_Name", "ethanol")
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()
    return molecule


def _pose_model(rank: int, score: float, offset: float) -> str:
    return "\n".join(
        (
            f"MODEL {rank}",
            f"REMARK VINA RESULT: {score:.3f} 0.000 0.000",
            "REMARK SMILES CCO",
            "REMARK SMILES IDX 1 1 2 2 3 3",
            f"ATOM      1  C   UNL     1       0.000   {offset:5.3f}   0.000  1.00  0.00     0.000 C ",
            f"ATOM      2  C   UNL     1       1.500   {offset:5.3f}   0.000  1.00  0.00     0.000 C ",
            f"ATOM      3  O   UNL     1       2.500   {offset:5.3f}   0.000  1.00  0.00     0.000 OA",
            "ENDMDL",
        )
    ) + "\n"


def _typed_inputs(runs_dir: Path) -> tuple[Path, ArtifactRef, Path, ArtifactRef]:
    target_dir = runs_dir / "protein-cleaning" / "target-1"
    target_dir.mkdir(parents=True)
    receptor = target_dir / "prepared.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"
    )
    target = ArtifactRef.from_path(target_dir, receptor, "prepared_target", role="receptor")
    reference_dir = runs_dir / "structure-jobs" / "reference-1"
    reference_dir.mkdir(parents=True)
    reference_path = reference_dir / "reference.sdf"
    _reference_sdf(reference_path)
    reference = ArtifactRef.from_path(
        reference_dir, reference_path, "prepared_ligand_set", role="ligand"
    )
    return receptor, target, reference_path, reference


def test_symmetry_aware_pose_metrics_keep_top_and_best_of_n(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.sdf"
    reference = _reference_sdf(reference_path)
    pose_path = tmp_path / "poses.pdbqt"
    pose_path.write_text(_pose_model(1, -7.0, 0.2) + _pose_model(2, -6.5, 0.1))

    rows, predicted = redocking_pose_metrics(reference, pose_path)

    assert [row["pose_rank"] for row in rows] == [1, 2]
    assert rows[0]["symmetry_rmsd_angstrom"] == pytest.approx(0.2)
    assert rows[1]["symmetry_rmsd_angstrom"] == pytest.approx(0.1)
    assert rows[0]["score_kcal_mol"] == -7.0
    assert predicted.GetNumConformers() == 1


def test_redocking_queues_typed_engine_replicates(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    receptor, target, reference_path, reference = _typed_inputs(runs_dir)

    parent = queue_redocking_benchmark(
        receptor_path=receptor,
        target_artifact=target,
        target_task_group="protein-cleaning",
        reference_ligand_path=reference_path,
        reference_ligand_artifact=reference,
        reference_task_group="structure-jobs",
        center=(1.0, 2.0, 3.0),
        size=(34.0, 35.0, 36.0),
        box_mode="padding",
        box_padding_angstrom=15.0,
        engines=("vina", "gnina"),
        replicates=2,
        seed_start=7001,
        context_metadata={
            "benchmark_dataset_run_id": "dataset-1",
            "benchmark_case_id": "case-1",
        },
    )

    workflow = WorkflowRecord.load(parent.run_id)
    assert parent.task_group == "workflows"
    assert workflow.workflow_type == "redocking_benchmark"
    assert workflow.parameters["box_mode"] == "padding"
    assert workflow.parameters["box_padding_angstrom"] == 15.0
    assert workflow.parameters["context"]["benchmark_dataset_run_id"] == "dataset-1"
    assert len(workflow.children) == 4
    assert len(workflow.inputs) == 2
    assert set(workflow.expected_steps) == {
        "vina_replicate_1", "vina_replicate_2", "gnina_replicate_1", "gnina_replicate_2"
    }
    for child in workflow.children:
        job = JobRecord.load(runs_dir / child.task_group / child.run_id, task_group=child.task_group)
        assert job.status == "queued"
        assert job.workflow_parent_run_id == parent.run_id
        assert job.metadata["redocking_reference_run_id"] == reference.run_id
        assert job.metadata["benchmark_case_id"] == "case-1"
        assert job.metadata["compound_count"] == 1
        assert job.metadata["box_mode"] == "padding"
        assert job.metadata["box_padding_angstrom"] == 15.0
        assert job.metadata["seed_start"] == 7000 + int(
            job.metadata["benchmark_replicate"]
        )


def test_redocking_finalizer_publishes_replicate_summary_and_overlays(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    receptor, target, reference_path, reference = _typed_inputs(runs_dir)
    parent = queue_redocking_benchmark(
        receptor_path=receptor,
        target_artifact=target,
        target_task_group="protein-cleaning",
        reference_ligand_path=reference_path,
        reference_ligand_artifact=reference,
        reference_task_group="structure-jobs",
        center=(1.0, 2.0, 3.0),
        size=(20.0, 20.0, 20.0),
        engines=("vina",),
        replicates=2,
        poses=2,
    )
    workflow = WorkflowRecord.load(parent.run_id)
    for index, child in enumerate(workflow.children, start=1):
        child_dir = runs_dir / child.task_group / child.run_id
        replicate_dir = child_dir / "results" / "replicate_001"
        replicate_dir.mkdir()
        (replicate_dir / "ethanol_out.pdbqt").write_text(
            _pose_model(1, -6.0 - index, 0.1 * index)
            + _pose_model(2, -5.0 - index, 0.05 * index)
        )
        metadata = json.loads((child_dir / "metadata.json").read_text())
        metadata["status"] = "completed"
        (child_dir / "metadata.json").write_text(json.dumps(metadata))

    completed = finalize_redocking_benchmark(parent.run_id)

    assert completed.status == "completed"
    assert completed.result["redocking_finalized"] is True
    assert completed.result["replicate_count"] == 2
    assert completed.result["pose_count"] == 4
    assert completed.artifact_manifest is not None
    assert len(completed.artifact_manifest.by_type("redocking_overlay")) == 2
    summary_artifact = completed.artifact_manifest.by_type("redocking_summary")[0]
    summary_path = summary_artifact.resolve(completed.run_dir, must_exist=True)
    assert summary_path is not None
    with summary_path.open(newline="") as handle:
        summary = next(csv.DictReader(handle))
    assert float(summary["mean_top_rmsd_angstrom"]) == pytest.approx(0.15)
    assert float(summary["sd_top_rmsd_angstrom"]) == pytest.approx(0.070710678)
    assert float(summary["top_pose_recovery_at_1a"]) == 1.0
    assert float(summary["mean_top_score_kcal_mol"]) == -7.5


def test_redocking_finalizer_marks_failed_child_without_claiming_success(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    receptor, target, reference_path, reference = _typed_inputs(runs_dir)
    parent = queue_redocking_benchmark(
        receptor_path=receptor,
        target_artifact=target,
        target_task_group="protein-cleaning",
        reference_ligand_path=reference_path,
        reference_ligand_artifact=reference,
        reference_task_group="structure-jobs",
        center=(1.0, 2.0, 3.0),
        size=(20.0, 20.0, 20.0),
        engines=("vina",),
        replicates=1,
    )
    child = WorkflowRecord.load(parent.run_id).children[0]
    child_dir = runs_dir / child.task_group / child.run_id
    metadata = json.loads((child_dir / "metadata.json").read_text())
    metadata.update({"status": "failed", "error": "native docking failed"})
    (child_dir / "metadata.json").write_text(json.dumps(metadata))

    failed = finalize_redocking_benchmark(parent.run_id)

    assert failed.status == "failed"
    assert failed.result["success"] is False
    assert failed.result["redocking_finalized"] is True
    assert "vina_replicate_1: failed" in failed.result["error"]
