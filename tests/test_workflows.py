from __future__ import annotations

import json
from pathlib import Path

from mn_ligand.core.artifacts import ArtifactRef
from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.workflows import (
    WorkflowRecord,
    add_workflow_input,
    attach_workflow_child,
    create_workflow,
    refresh_workflow,
)


def _child(runs_dir: Path, task_group: str, run_id: str, status: str, *, parent_run_id: str = "") -> JobRecord:
    run_dir = runs_dir / task_group / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "status": status,
                "parent_run_id": parent_run_id,
                "created_at": "2026-07-20T00:00:00+00:00",
            }
        )
    )
    return JobRecord.load(run_dir, task_group=task_group)


def test_workflow_tracks_declared_steps_and_preserves_direct_parent(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    workflow = create_workflow(
        "protein-complex-preparation",
        name="4LNW preparation",
        expected_steps=("import", "clean", "complex"),
    )
    imported = _child(runs_dir, "protein-import", "import-1", "completed")
    cleaned = _child(
        runs_dir,
        "protein-cleaning",
        "clean-1",
        "running",
        parent_run_id=imported.run_id,
    )

    after_import = attach_workflow_child(workflow.workflow_id, imported, step_id="import")
    assert after_import.status == "queued"
    after_clean = attach_workflow_child(
        workflow.workflow_id,
        cleaned,
        step_id="clean",
        depends_on=(imported.run_id,),
    )
    assert after_clean.status == "running"

    child_metadata = json.loads((cleaned.run_dir / "metadata.json").read_text())
    assert child_metadata["parent_run_id"] == imported.run_id
    assert child_metadata["workflow_parent_run_id"] == workflow.workflow_id
    assert child_metadata["workflow_step_id"] == "clean"

    complex_job = _child(
        runs_dir,
        "structure-jobs",
        "complex-1",
        "completed",
        parent_run_id=cleaned.run_id,
    )
    attach_workflow_child(
        workflow.workflow_id,
        complex_job,
        step_id="complex",
        depends_on=(cleaned.run_id,),
    )
    child_metadata["status"] = "completed"
    (cleaned.run_dir / "metadata.json").write_text(json.dumps(child_metadata))
    completed = refresh_workflow(workflow.workflow_id)

    assert completed.status == "completed"
    parent_job = JobRecord.load(completed.run_dir, task_group="workflows")
    assert parent_job.status == "completed"
    assert parent_job.result["progress"] == {"completed": 3, "total": 3, "failed": 0, "percent": 100}


def test_replacing_queued_child_cancels_superseded_work(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    workflow = create_workflow(
        "md-simulation",
        name="endpoint replacement",
        expected_steps=("endpoint_energy_replica_1",),
    )
    original = _child(runs_dir, "md-mmgbsa", "endpoint-old", "queued")
    replacement = _child(runs_dir, "md-mmgbsa", "endpoint-new", "queued")
    attach_workflow_child(
        workflow.workflow_id,
        original,
        step_id="endpoint_energy_replica_1",
    )

    updated = attach_workflow_child(
        workflow.workflow_id,
        replacement,
        step_id="endpoint_energy_replica_1",
        replace_step=True,
    )

    old_metadata = json.loads((original.run_dir / "metadata.json").read_text())
    assert old_metadata["status"] == "cancelled"
    assert old_metadata["superseded_by_run_id"] == replacement.run_id
    assert old_metadata["cancellation_requested_by"] == "workflow-replacement"
    assert next(
        child for child in updated.children if child.run_id == original.run_id
    ).required is False
    assert next(
        child for child in updated.children if child.run_id == replacement.run_id
    ).required is True


def test_workflow_inputs_are_portable_and_idempotent(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    workflow = create_workflow("screening", name="Screen target")
    source_dir = runs_dir / "protein-cleaning" / "source-1"
    artifact_path = source_dir / "artifacts" / "prepared.pdb"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("ATOM\nEND\n")
    artifact = ArtifactRef.from_path(source_dir, artifact_path, "prepared_target", role="receptor")

    add_workflow_input(workflow.workflow_id, "protein-cleaning", artifact)
    add_workflow_input(workflow.workflow_id, "protein-cleaning", artifact)
    loaded = WorkflowRecord.load(workflow.workflow_id)

    assert len(loaded.inputs) == 1
    assert loaded.inputs[0].artifact.path == "artifacts/prepared.pdb"
    assert not Path(loaded.inputs[0].artifact.path).is_absolute()
    input_payload = json.loads((loaded.run_dir / "input.json").read_text())
    assert input_payload["inputs"][0]["artifact"]["run_id"] == "source-1"


def test_required_failed_or_missing_child_fails_or_blocks_workflow(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    workflow = create_workflow("docking", name="Dock", expected_steps=("dock",))
    failed = _child(runs_dir, "structure-docking", "dock-1", "failed")

    assert attach_workflow_child(workflow.workflow_id, failed, step_id="dock").status == "failed"
    failed.run_dir.rename(failed.run_dir.with_name("removed"))
    assert refresh_workflow(workflow.workflow_id).status == "blocked"


def test_reference_child_preserves_original_workflow_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    original = create_workflow("md-simulation", name="Original preparation")
    continuation = create_workflow(
        "md-simulation",
        name="Continuation",
        expected_steps=("preparation_equilibration",),
    )
    preparation = _child(runs_dir, "md-system-prep", "prep-1", "completed")
    metadata_path = preparation.run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "workflow_id": original.workflow_id,
            "workflow_parent_run_id": original.workflow_id,
            "workflow_step_id": "preparation_equilibration",
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))

    attached = attach_workflow_child(
        continuation.workflow_id,
        preparation,
        step_id="preparation_equilibration",
        update_child_metadata=False,
    )

    assert attached.status == "completed"
    assert attached.children[0].run_id == preparation.run_id
    preserved = json.loads(metadata_path.read_text())
    assert preserved["workflow_id"] == original.workflow_id
    assert preserved["workflow_parent_run_id"] == original.workflow_id


def test_replacement_child_supersedes_failed_step_without_inflating_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    workflow = create_workflow(
        "md-simulation",
        name="Endpoint replacement",
        expected_steps=("production_replica_1", "endpoint_energy_replica_1"),
    )
    production = _child(
        runs_dir,
        "bound-ligand-md",
        "production-1",
        "completed",
    )
    failed = _child(runs_dir, "md-mmgbsa", "endpoint-failed", "failed")
    replacement = _child(
        runs_dir,
        "md-mmgbsa",
        "endpoint-replacement",
        "completed",
    )
    attach_workflow_child(
        workflow.workflow_id,
        production,
        step_id="production_replica_1",
    )
    attach_workflow_child(
        workflow.workflow_id,
        failed,
        step_id="endpoint_energy_replica_1",
        depends_on=(production.run_id,),
    )
    updated = attach_workflow_child(
        workflow.workflow_id,
        replacement,
        step_id="endpoint_energy_replica_1",
        depends_on=(production.run_id,),
        replace_step=True,
    )

    assert updated.status == "completed"
    endpoint_refs = [
        child
        for child in updated.children
        if child.step_id == "endpoint_energy_replica_1"
    ]
    assert [child.required for child in endpoint_refs] == [False, True]
    failed_metadata = json.loads(
        (failed.run_dir / "metadata.json").read_text()
    )
    assert failed_metadata["superseded_by_run_id"] == replacement.run_id
    result = json.loads((updated.run_dir / "result.json").read_text())
    assert result["progress"] == {
        "completed": 2,
        "total": 2,
        "failed": 0,
        "percent": 100,
    }


def test_failed_upstream_is_not_masked_by_queued_downstream(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    workflow = create_workflow(
        "md-simulation",
        name="Failed preparation",
        expected_steps=("preparation", "production", "analysis"),
    )
    failed = _child(runs_dir, "md-system-prep", "prep-1", "failed")
    blocked = _child(runs_dir, "bound-ligand-md", "production-1", "blocked")
    queued = _child(runs_dir, "md-analysis", "analysis-1", "queued")

    attach_workflow_child(workflow.workflow_id, failed, step_id="preparation")
    attach_workflow_child(workflow.workflow_id, blocked, step_id="production")
    updated = attach_workflow_child(
        workflow.workflow_id, queued, step_id="analysis"
    )

    assert updated.status == "failed"
