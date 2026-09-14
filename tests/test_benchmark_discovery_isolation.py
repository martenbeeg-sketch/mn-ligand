import json
from pathlib import Path

from mn_ligand.app.pages.discover_inputs import target_inventory
from mn_ligand.app.pages.docking import _job_rows as docking_job_rows
from mn_ligand.app.pages.docking_cofolding import (
    _job_rows as docking_cofolding_job_rows,
)
from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest


PDB = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
    "  1.00 20.00           C\nEND\n"
)


def _write_job(
    runs_dir: Path,
    task_group: str,
    run_id: str,
    *,
    job_code: str,
    metadata: dict[str, object],
    prepared_target: bool = False,
) -> None:
    run_dir = runs_dir / task_group / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "job_code": job_code,
                "status": "completed",
                "created_at": "2026-07-30T00:00:00+00:00",
                **metadata,
            }
        )
    )
    artifacts = []
    if prepared_target:
        target = run_dir / "artifacts" / "receptor.pdb"
        target.parent.mkdir()
        target.write_text(PDB)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                target,
                "prepared_target",
                role="receptor",
            )
        )
    write_artifact_manifest(run_dir, artifacts)


def test_discovery_excludes_benchmark_targets_and_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    _write_job(
        runs_dir,
        "structure-jobs",
        "ordinary-target",
        job_code="ORD01",
        metadata={"workflow": "structure_preparation", "pdb_id": "1ABC"},
        prepared_target=True,
    )
    _write_job(
        runs_dir,
        "rescoring",
        "benchmark-target",
        job_code="FA62C",
        metadata={
            "workflow": "benchmark_bound_chain_selection",
            "benchmark_dataset_run_id": "dataset-1",
            "benchmark_campaign_id": "campaign-1",
            "benchmark_case_id": "case-1",
        },
        prepared_target=True,
    )
    _write_job(
        runs_dir,
        "docking",
        "ordinary-docking",
        job_code="DOCK1",
        metadata={
            "workflow": "docking_campaign",
            "operation": "docking",
            "tool": "vina",
        },
    )
    _write_job(
        runs_dir,
        "docking",
        "benchmark-docking",
        job_code="D83CF",
        metadata={
            "workflow": "docking_campaign",
            "operation": "docking",
            "tool": "vina",
            "parameters": {
                "context": {
                    "benchmark_dataset_run_id": "dataset-1",
                    "benchmark_campaign_id": "campaign-1",
                    "benchmark_case_id": "case-1",
                }
            },
        },
    )

    discovery_targets = target_inventory(("prepared_target",))
    all_targets = target_inventory(
        ("prepared_target",),
        include_benchmarks=True,
    )

    assert [entry.choice.job.run_id for entry in discovery_targets] == [
        "ordinary-target"
    ]
    assert {entry.choice.job.run_id for entry in all_targets} == {
        "ordinary-target",
        "benchmark-target",
    }
    for rows in (
        docking_job_rows({"completed"}),
        docking_cofolding_job_rows({"completed"}),
    ):
        assert [row["job"] for row in rows] == ["DOCK1"]
