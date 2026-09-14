from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mn_ligand.app.pages.unified_jobs import (
    _apply_history_visibility,
    _job_row,
)
from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records, short_job_code


def test_short_job_code_uses_first_five_characters_of_uuid_suffix() -> None:
    run_id = "537818f6-4b68-426b-9577-f5479ba717d3"

    assert short_job_code(run_id) == "F5479"
    assert display_job_code("KOA", run_id) == "F5479"
    assert display_job_code("A1B2C", run_id) == "A1B2C"


def test_job_record_normalizes_legacy_structure_job(tmp_path: Path) -> None:
    run_dir = tmp_path / "structure-jobs" / "legacy-run"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "legacy-run",
                "status": "completed",
                "job_type": "structure",
                "source": "pdb",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    (run_dir / "target_protein_refined.pdb").write_text("ATOM\nEND\n")

    job = JobRecord.load(run_dir)

    assert job.run_id == "legacy-run"
    assert job.task_group == "structure-jobs"
    assert job.status == "completed"
    assert job.schema_version == 0
    assert job.tool == "pdb"
    assert job.artifact_manifest is not None
    assert job.artifact_manifest.by_type("prepared_receptor")


def test_job_record_maps_unknown_legacy_state_to_unknown(tmp_path: Path) -> None:
    run_dir = tmp_path / "qc" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"status": "mystery-state"}))

    assert JobRecord.load(run_dir).status == "unknown"


def test_job_record_reads_current_schema_version(tmp_path: Path) -> None:
    run_dir = tmp_path / "structure-jobs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"schema_version": 1, "status": "preparing"}))

    job = JobRecord.load(run_dir)

    assert job.schema_version == 1
    assert job.status == "preparing"


def test_job_record_can_skip_large_result_and_artifact_payloads(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "bound-ligand-md" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"schema_version": 1, "status": "completed"})
    )
    (run_dir / "result.json").write_text(
        json.dumps({"trajectory_analysis": [1, 2, 3]})
    )
    (run_dir / "artifacts.json").write_text("{broken")

    job = JobRecord.load(
        run_dir,
        load_result=False,
        load_artifacts=False,
        validate_artifacts=False,
    )

    assert job.status == "completed"
    assert job.result == {}
    assert job.artifact_manifest is None
    assert job.warnings == ()


def test_job_record_reads_workflow_relationships(tmp_path: Path) -> None:
    run_dir = tmp_path / "protein-cleaning" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "workflow_id": "workflow-1",
                "workflow_parent_run_id": "workflow-1",
            }
        )
    )

    job = JobRecord.load(run_dir)

    assert job.workflow_id == "workflow-1"
    assert job.workflow_parent_run_id == "workflow-1"


def test_job_record_reports_broken_artifact_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "structure-jobs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"status": "completed"}))
    (run_dir / "artifacts.json").write_text("{broken")

    job = JobRecord.load(run_dir)

    assert job.artifact_manifest is not None
    assert job.artifact_manifest.source == "unavailable"
    assert any("Artifact manifest error" in warning for warning in job.warnings)


def test_job_record_and_unified_row_report_partial_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "docking" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"status": "completed", "partial_success": True})
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "success": True,
                "partial_success": True,
                "warning": "Partial result: 715/715 eligible predictions completed; "
                "2 compounds were excluded by preflight.",
            }
        )
    )

    job = JobRecord.load(run_dir)
    row = _job_row(job)

    assert job.status == "completed"
    assert job.warnings == (
        "Partial result: 715/715 eligible predictions completed; "
        "2 compounds were excluded by preflight.",
    )
    assert row["status"] == "completed"
    assert "Completed with partial results." in row["detail"]
    assert "2 compounds were excluded by preflight." in row["detail"]


def test_iter_job_records_indexes_all_task_groups(tmp_path: Path) -> None:
    older = tmp_path / "admet" / "run-old"
    newer = tmp_path / "qc" / "run-new"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "metadata.json").write_text(
        json.dumps({"status": "completed", "created_at": "2026-01-01T00:00:00+00:00"})
    )
    (newer / "metadata.json").write_text(
        json.dumps({"status": "running", "created_at": "2026-02-01T00:00:00+00:00"})
    )

    jobs = iter_job_records(tmp_path)

    assert [(job.task_group, job.run_id) for job in jobs] == [("qc", "run-new"), ("admet", "run-old")]


def test_unified_job_row_distinguishes_dependency_wait_from_ready_queue(
    tmp_path: Path,
) -> None:
    waiting = JobRecord(
        run_id="waiting-run",
        task_group="md-analysis",
        run_dir=tmp_path,
        status="queued",
        metadata={"awaiting_parent": True},
    )
    ready = JobRecord(
        run_id="ready-run",
        task_group="bound-ligand-md",
        run_dir=tmp_path,
        status="queued",
        metadata={"resources": {"gpu": True}, "md_engine": "gromacs"},
    )

    waiting_row = _job_row(waiting)
    ready_row = _job_row(ready)

    assert waiting_row["status"] == "waiting"
    assert "upstream" in waiting_row["detail"]
    assert ready_row["status"] == "queued"
    assert "compatible worker" in ready_row["detail"]
    assert ready_row["gpu"] == "GPU requested"
    assert ready_row["tool"] == "gromacs"


def test_unified_job_row_shows_blocking_error_and_assigned_gpu(
    tmp_path: Path,
) -> None:
    blocked = JobRecord(
        run_id="blocked-run",
        task_group="bound-ligand-md",
        run_dir=tmp_path,
        status="blocked",
        metadata={
            "error": "Preparation job failed",
            "worker_id": "mn-ligand-gpu-1",
        },
    )

    row = _job_row(blocked)

    assert row["status"] == "blocked"
    assert "Preparation job failed" in row["detail"]
    assert row["gpu"] == "GPU 1"


def test_failed_and_superseded_jobs_are_hidden_by_default() -> None:
    frame = pd.DataFrame(
        [
            {
                "run_id": "completed",
                "raw_status": "completed",
                "superseded": False,
            },
            {
                "run_id": "failed",
                "raw_status": "failed",
                "superseded": False,
            },
            {
                "run_id": "superseded",
                "raw_status": "completed",
                "superseded": True,
            },
        ]
    )

    visible = _apply_history_visibility(
        frame,
        show_failed_history=False,
    )
    all_rows = _apply_history_visibility(
        frame,
        show_failed_history=True,
    )

    assert visible["run_id"].tolist() == ["completed"]
    assert all_rows["run_id"].tolist() == [
        "completed",
        "failed",
        "superseded",
    ]
