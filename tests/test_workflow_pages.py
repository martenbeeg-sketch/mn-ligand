from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.workflows import attach_workflow_child, create_workflow
from mn_ligand.app.pages.bound_ligand_md import _pymol_trajectory_script
from mn_ligand.workflows.compound_preparation import (
    create_compound_import_job,
)
from mn_ligand.workflows.protein_preparation import create_protein_import_job


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_pymol_trajectory_script_loads_topology_and_dcd_without_copying(
    tmp_path: Path,
) -> None:
    topology = tmp_path / "system topology.pdb"
    trajectory = tmp_path / "production trajectory.dcd"
    script = _pymol_trajectory_script(topology, trajectory, "LIG")

    assert f'load "{topology.resolve()}", md' in script
    assert f'load_traj "{trajectory.resolve()}", md' in script
    assert "show sticks, md and (resn LIG)" in script
    assert "intra_fit md and polymer and name CA" in script
    assert "mset" not in script


def _completed_child(runs_dir: Path) -> JobRecord:
    run_dir = runs_dir / "protein-import" / "import-ui-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"schema_version": 1, "run_id": run_dir.name, "status": "completed"})
    )
    return JobRecord.load(run_dir, task_group="protein-import")


def _controllable_job(runs_dir: Path, run_id: str, status: str) -> Path:
    run_dir = runs_dir / "md-mmgbsa" / run_id
    run_dir.mkdir(parents=True)
    command = ["fixture", str(run_dir)]
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "status": status,
                "worker_finalizer": "md_mmgbsa",
                "queued_command": command,
                "resources": {"gpu": False},
                "created_at": "2026-07-22T00:00:00+00:00",
            }
        )
    )
    for name, payload in (
        ("input.json", {"source_production_run_id": "production-1"}),
        ("source_input.json", {"trajectory": "/source/production.dcd"}),
        ("source_result.json", {"success": True}),
    ):
        (run_dir / name).write_text(json.dumps(payload))
    (run_dir / "command.json").write_text(
        json.dumps({"schema_version": 1, "argv": command, "commands": [command]})
    )
    return run_dir


def test_workflow_parent_renders_in_jobs_and_results(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    workflow = create_workflow(
        "protein-complex-preparation",
        name="UI workflow",
        expected_steps=("protein_import",),
    )
    attach_workflow_child(workflow.workflow_id, _completed_child(runs_dir), step_id="protein_import")

    jobs = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/unified_jobs.py").run(timeout=20)
    assert not jobs.exception
    assert jobs.dataframe

    results = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    results.query_params["task_group"] = "workflows"
    results.query_params["run_id"] = workflow.workflow_id
    results.run(timeout=20)
    assert not results.exception
    assert len(results.dataframe) >= 2
    workflow_stage_tables = [
        table.value
        for table in results.dataframe
        if {"results", "step", "status", "task", "required", "job", "depends_on"}
        == set(table.value.columns)
    ]
    assert len(workflow_stage_tables) == 1
    assert workflow_stage_tables[0].iloc[0]["results"].startswith(
        "./job-results?"
    )


def test_md_simulation_page_renders_without_sources(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/md_simulation.py").run(timeout=20)

    assert not page.exception
    assert any("No prepared complexes" in item.value for item in page.info)
    assert [tab.label for tab in page.tabs] == [
        "Target / Input", "Tool / Engine", "Run", "Results"
    ]
    assert next(
        control
        for control in page.segmented_control
        if control.label == "Production protocol"
    ).value == "Standard"
    assert next(
        control
        for control in page.selectbox
        if control.label == "System-preparation protocol"
    ).value == "Roe–Brooks 2020 inspired OpenMM"
    assert next(
        control
        for control in page.checkbox
        if control.label
        == "Automatically run endpoint MM/GBSA after production"
    ).value is False
    assert not any(
        control.label == "Endpoint engine" for control in page.selectbox
    )
    assert not any(
        control.label == "Legacy preparation preset"
        for control in page.segmented_control
    )
    assert {metric.label for metric in page.metric} >= {
        "CPU threads", "Free GPUs", "Queued jobs", "Active GPU leases"
    }
    assert next(
        button for button in page.button if button.label == "Submit MD workflow"
    ).disabled is True


def test_md_results_groups_workflow_children_and_separates_legacy_runs(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    workflow = create_workflow(
        "md-simulation",
        name="Grouped MD campaign",
        parameters={"replicas": 1},
        expected_steps=(
            "preparation_equilibration",
            "production_replica_1",
            "endpoint_energy_replica_1",
            "replicate_analysis",
        ),
    )

    child_dir = runs_dir / "bound-ligand-md" / "workflow-production"
    child_dir.mkdir(parents=True)
    (child_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": child_dir.name,
                "status": "completed",
                "created_at": "2026-07-25T12:00:00+00:00",
                "repeat_index": 1,
            }
        )
    )
    child = JobRecord.load(child_dir, task_group="bound-ligand-md")
    attach_workflow_child(
        workflow.workflow_id,
        child,
        step_id="production_replica_1",
    )

    legacy_dir = runs_dir / "bound-ligand-md" / "legacy-production"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": legacy_dir.name,
                "status": "completed",
                "created_at": "2026-05-15T08:37:20+00:00",
                "repeat_index": 2,
                "pdb_id": "4LNW",
                "ligand_label": "LIG",
            }
        )
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/md_simulation.py"
    ).run(timeout=20)

    assert not page.exception
    workflow_tables = [
        table.value
        for table in page.dataframe
        if "stages" in table.value.columns
    ]
    legacy_tables = [
        table.value
        for table in page.dataframe
        if {"results", "job", "description", "replicas", "status", "created"}
        == set(table.value.columns)
    ]
    stage_tables = [
        table.value
        for table in page.dataframe
        if {"results", "stage", "job", "status", "detail"}
        == set(table.value.columns)
    ]
    assert len(workflow_tables) == 1
    assert len(workflow_tables[0]) == 1
    assert workflow_tables[0].iloc[0]["description"] == "Grouped MD campaign"
    assert len(stage_tables) == 1
    assert len(stage_tables[0]) == 1
    assert stage_tables[0].iloc[0]["stage"] == "Production replica 1"
    assert "./md-results?" in stage_tables[0].iloc[0]["results"]
    assert len(legacy_tables) == 1
    assert len(legacy_tables[0]) == 1
    assert legacy_tables[0].iloc[0]["description"] == "4LNW · LIG"


def test_shared_discover_and_direct_pages_use_run_stage_resources(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))
    pages = (
        ("virtual_screening.py", "Run Vina campaign"),
        ("admet.py", "Run ADMET"),
        ("qc.py", "Run QC"),
    )

    for filename, run_label in pages:
        page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages" / filename).run(
            timeout=20
        )
        assert not page.exception
        assert [tab.label for tab in page.tabs] == [
            "Target / Input", "Tool / Engine", "Run", "Results"
        ]
        assert {metric.label for metric in page.metric} >= {
            "CPU threads", "Free GPUs", "Queued jobs", "Active GPU leases"
        }
        assert next(button for button in page.button if button.label == run_label)


def test_molecule_design_page_uses_campaign_tabs_and_queue_action(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv(
        "MN_LIGAND_REFERENCE_DIR", str(tmp_path / "references")
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/generative_design.py"
    ).run(timeout=20)

    assert not page.exception
    assert [tab.label for tab in page.tabs] == [
        "Target",
        "Conditioning",
        "Engines",
        "Run",
        "Results",
    ]
    assert {metric.label for metric in page.metric} >= {
        "CPU threads",
        "Free GPUs",
        "Queued jobs",
        "Active GPU leases",
    }
    queue = next(
        button
        for button in page.button
        if button.label == "Queue generation campaign"
    )
    assert queue.disabled is True


def test_md_results_page_renders_post_run_mmgbsa_controls(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = runs_dir / "bound-ligand-md" / "production-ui-1"
    output_dir = run_dir / run_dir.name
    output_dir.mkdir(parents=True)
    structure = output_dir / "production.pdb"
    structure.write_text("MODEL\nENDMDL\n")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "status": "completed",
                "use_gpu": False,
            }
        )
    )
    (run_dir / "input.json").write_text(
        json.dumps({"mmgbsa_backend": "openmm_gbsa", "mmgbsa_stride": 1})
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "success": True,
                "md_result": {
                    "status": "completed",
                    "output_files": {
                        "production_pdb": str(structure),
                    },
                },
            }
        )
    )
    analysis_dir = runs_dir / "md-mmgbsa" / "analysis-ui-1"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": analysis_dir.name,
                "status": "completed",
                "source_production_run_id": run_dir.name,
                "parameters": {
                    "start_pct": 50,
                    "end_pct": 100,
                    "stride": 2,
                    "backend": "openmm_gbsa",
                },
                "created_at": "2026-07-22T00:00:00+00:00",
            }
        )
    )
    (analysis_dir / "result.json").write_text(
        json.dumps(
            {
                "success": True,
                "mmgbsa": {
                    "status": "success",
                    "method": "OpenMM GBSA fixture",
                    "delta": {
                        "delta_g_bind_total_kj_mol": -5.0,
                        "delta_g_bind_total_kcal_mol": -1.195,
                    },
                },
            }
        )
    )

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/md_results.py")
    page.query_params["run_id"] = run_dir.name
    page.query_params["run_type"] = "bound-ligand-md"
    page.run(timeout=20)

    assert not page.exception
    assert any(button.label == "Queue MM/GBSA analysis" for button in page.button)
    assert any(select.label == "Endpoint energy engine" for select in page.selectbox)
    assert any(metric.label == "ΔG_bind total" and "-5.000" in metric.value for metric in page.metric)


def test_job_results_retries_without_mutating_failed_run(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = _controllable_job(runs_dir, "failed-ui-1", "failed")
    (run_dir / "result.json").write_text(json.dumps({"success": False, "error": "fixture"}))
    original = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    page.query_params["task_group"] = "md-mmgbsa"
    page.query_params["run_id"] = run_dir.name
    page.run(timeout=20)
    retry = next(button for button in page.button if button.label == "Retry as new job")
    retry.click().run(timeout=20)

    assert not page.exception
    retries = [path for path in run_dir.parent.iterdir() if path.is_dir() and path != run_dir]
    assert len(retries) == 1
    retry_metadata = json.loads((retries[0] / "metadata.json").read_text())
    assert retry_metadata["retry_of_run_id"] == run_dir.name
    assert {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    } == original


def test_job_results_exposes_cancel_control_for_queued_job(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = _controllable_job(runs_dir, "queued-results-ui-1", "queued")

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    page.query_params["task_group"] = "md-mmgbsa"
    page.query_params["run_id"] = run_dir.name
    page.run(timeout=20)

    assert not page.exception
    assert any(button.label == "Cancel job" for button in page.button)
    assert any(box.label == "Confirm cancellation" for box in page.checkbox)


def test_unified_jobs_cancels_queued_job_and_refresh_is_read_only(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = _controllable_job(runs_dir, "queued-ui-1", "queued")

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/unified_jobs.py").run(timeout=20)
    confirm = next(box for box in page.checkbox if box.label == "Confirm cancellation")
    confirm.set_value(True).run(timeout=20)
    cancel = next(button for button in page.button if button.label == "Cancel selected job")
    cancel.click().run(timeout=20)

    assert not page.exception
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "cancelled"
    assert metadata["cancellation_requested"] is True
    stable = (run_dir / "metadata.json").read_bytes()
    page.run(timeout=20)
    assert (run_dir / "metadata.json").read_bytes() == stable


def test_job_results_displays_resource_admission_reason(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = _controllable_job(runs_dir, "waiting-results-ui-1", "queued")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    metadata["admission"] = {
        "status": "waiting",
        "reasons": ["requires 16 GiB RAM; 8.00 GiB is available"],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    page.query_params["task_group"] = "md-mmgbsa"
    page.query_params["run_id"] = run_dir.name
    page.run(timeout=20)

    assert not page.exception
    assert any("Resource admission waiting" in warning.value for warning in page.warning)
    assert any("16 GiB RAM" in warning.value for warning in page.warning)

    jobs_page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/unified_jobs.py"
    ).run(timeout=20)
    assert not jobs_page.exception
    assert any(
        "16 GiB RAM" in str(frame.value.get("warning", "").to_string())
        for frame in jobs_page.dataframe
        if hasattr(frame.value, "get")
    )


def test_pocket_detection_page_renders_without_sources(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/pocket_detection.py"
    ).run(timeout=20)

    assert not page.exception
    assert any("Prepare a target" in item.value for item in page.info)
    assert [tab.label for tab in page.tabs] == ["Target", "Tool", "Run", "Results"]
    assert {metric.label for metric in page.metric} >= {
        "CPU threads", "Free GPUs", "Queued jobs", "Active GPU leases"
    }


def test_pocket_detection_page_groups_engines_and_queues_selected_cpu_tools(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = runs_dir / "protein-cleaning" / "target-p2rank-ui"
    artifact = run_dir / "artifacts" / "prepared_target.pdb"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "job_code": "P2RUI1",
                "status": "completed",
                "pdb_id": "P2R",
                "created_at": "2026-07-22T00:00:00+00:00",
            }
        )
    )
    write_artifact_manifest(
        run_dir,
        [ArtifactRef.from_path(run_dir, artifact, "prepared_target")],
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/pocket_detection.py"
    ).run(timeout=20)
    p2rank = next(control for control in page.checkbox if control.label == "P2Rank")
    p2rank.set_value(True).run(timeout=20)

    assert not page.exception
    assert any(select.label == "Structure profile" for select in page.selectbox)
    assert any("CPU-only" in caption.value for caption in page.caption)
    run_button = next(
        button for button in page.button if button.label == "Run selected engines"
    )
    assert run_button.disabled is False
    run_button.click().run(timeout=20)

    assert not page.exception
    queued_methods = sorted(
        json.loads(path.read_text())["method"]
        for path in (runs_dir / "pocket-detection").glob("*/input.json")
    )
    assert queued_methods == ["fpocket", "p2rank"]
    assert any("Queued 2 pocket-detection jobs" in item.value for item in page.success)
    target_table = next(
        frame.value
        for frame in page.dataframe
        if "Pocket Detection" in frame.value.columns
    )
    detection_summary = str(target_table.iloc[0]["Pocket Detection"])
    assert "fpocket: queued" in detection_summary
    assert "P2Rank: queued" in detection_summary


def test_pocket_detection_defaults_to_bound_ligand_generation_pocket(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = runs_dir / "structure-jobs" / "target-bound-pocket-ui"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    receptor = artifact_dir / "prepared_receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "END\n"
    )
    complex_path = artifact_dir / "prepared_complex.pdb"
    complex_path.write_text(
        receptor.read_text().removesuffix("END\n")
        + "HETATM    2  C1  LIG A 101       1.000   1.000   0.000  1.00 20.00           C\n"
        + "HETATM    3  C2  LIG A 101       2.000   1.000   0.000  1.00 20.00           C\n"
        + "HETATM    4  O1  LIG A 101       3.000   1.000   0.000  1.00 20.00           O\n"
        + "END\n"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "job_code": "BNDUI1",
                "status": "completed",
                "pdb_id": "BND",
                "ligand_key": "T3|A|101|_",
                "ligands": [
                    {
                        "key": "LIG|A|101|_",
                        "resname": "LIG",
                        "chain": "A",
                        "resseq": "101",
                        "icode": "_",
                        "ccd_id": "T3",
                        "name": "TRIIODOTHYRONINE",
                    }
                ],
                "created_at": "2026-07-28T00:00:00+00:00",
            }
        )
    )
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(
                run_dir,
                receptor,
                "prepared_receptor",
            ),
            ArtifactRef.from_path(
                run_dir,
                complex_path,
                "prepared_complex",
            ),
        ],
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/pocket_detection.py"
    ).run(timeout=20)

    bound = next(
        control
        for control in page.checkbox
        if control.label == "Bound ligand — generation pocket"
    )
    fpocket = next(
        control for control in page.checkbox if control.label == "fpocket"
    )
    assert bound.value is True
    assert fpocket.value is False
    ligand = next(
        control
        for control in page.selectbox
        if control.label == "Bound ligand defining the generation pocket"
    )
    assert ligand.value == "LIG|A|101|_"
    assert not page.exception

    run_button = next(
        button
        for button in page.button
        if button.label == "Run selected engines"
    )
    run_button.click().run(timeout=20)

    queued_inputs = [
        json.loads(path.read_text())
        for path in (runs_dir / "pocket-detection").glob("*/input.json")
    ]
    assert len(queued_inputs) == 1
    assert queued_inputs[0]["method"] == "bound_ligand"
    assert queued_inputs[0]["parameters"]["bound_ligand_key"] == (
        "LIG|A|101|_"
    )
    assert queued_inputs[0]["parameters"]["box_padding_angstrom"] == 4.0
    assert queued_inputs[0]["parameters"]["lining_cutoff_angstrom"] == 5.0


def test_pocket_detection_target_table_includes_descendant_target_runs(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    for run_id, pdb_id, parent_run_id in (
        ("base-target", "BASE", ""),
        ("derived-target", "DERIVED", "base-target"),
    ):
        run_dir = runs_dir / "structure-jobs" / run_id
        target = run_dir / "prepared_target.pdb"
        target.parent.mkdir(parents=True)
        target.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"
        )
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "job_code": "BASE1" if run_id == "base-target" else "DER01",
                    "status": "completed",
                    "pdb_id": pdb_id,
                    "parent_run_id": parent_run_id,
                    "created_at": "2026-07-23T00:00:00+00:00",
                }
            )
        )
        write_artifact_manifest(
            run_dir,
            [ArtifactRef.from_path(run_dir, target, "prepared_target")],
        )

    pocket_dir = runs_dir / "pocket-detection" / "derived-pocket"
    pocket_dir.mkdir(parents=True)
    (pocket_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": pocket_dir.name,
                "job_code": "POCK1",
                "status": "completed",
                "workflow": "pocket_detection",
                "tool": "fpocket",
                "prepared_target_run_id": "derived-target",
                "parent_run_id": "derived-target",
                "created_at": "2026-07-23T00:01:00+00:00",
            }
        )
    )
    (pocket_dir / "input.json").write_text(
        json.dumps(
            {
                "method": "fpocket",
                "parameters": {"max_pockets": 10, "box_padding_angstrom": 4.0},
            }
        )
    )
    (pocket_dir / "result.json").write_text(
        json.dumps(
            {
                "success": True,
                "method": "fpocket",
                "pocket_count": 4,
            }
        )
    )
    write_artifact_manifest(pocket_dir, [])

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/pocket_detection.py"
    ).run(timeout=20)

    assert not page.exception
    target_table = next(
        frame.value
        for frame in page.dataframe
        if "Pocket Detection" in frame.value.columns
    )
    summaries = dict(
        zip(target_table["Target"], target_table["Pocket Detection"])
    )
    assert "fpocket: completed, 4 pockets" in summaries["DERIVED"]
    assert "target job DER01" in summaries["DERIVED"]
    assert "fpocket: completed, 4 pockets" in summaries["BASE"]
    assert "target job DER01" in summaries["BASE"]


def test_job_results_shows_pocket_in_full_target_context(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    target_dir = runs_dir / "protein-cleaning" / "pocket-view-target"
    target_path = target_dir / "prepared_target.pdb"
    target_path.parent.mkdir(parents=True)
    target_path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  TYR A   2       4.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  CA  LEU A   3       8.000   0.000   0.000  1.00 20.00           C\nEND\n"
    )
    (target_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": target_dir.name,
                "status": "completed",
                "created_at": "2026-07-23T00:00:00+00:00",
            }
        )
    )
    target_ref = ArtifactRef.from_path(target_dir, target_path, "prepared_target")
    write_artifact_manifest(target_dir, [target_ref])

    run_dir = runs_dir / "pocket-detection" / "pocket-view-job"
    pocket_structure = run_dir / "artifacts" / "pockets" / "pocket_001.pdb"
    pocket_structure.parent.mkdir(parents=True)
    pocket_structure.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  TYR A   2       4.000   0.000   0.000  1.00 20.00           C\nEND\n"
    )
    pocket_set = run_dir / "artifacts" / "pockets" / "pocket_set.json"
    pocket_set.write_text(
        json.dumps(
            {
                "kind": "pocket_set",
                "schema_version": 1,
                "method": "fpocket",
                "source_target": target_ref.to_dict(),
                "source_complex": {},
                "parameters": {},
                "pockets": [
                    {
                        "pocket_id": "fpocket-1",
                        "rank": 1,
                        "method": "fpocket",
                        "score": 0.75,
                        "druggability_score": 0.8,
                        "center_angstrom": [4.0, 0.0, 0.0],
                        "size_angstrom": [12.0, 8.0, 8.0],
                        "residues": [
                            {
                                "chain_id": "A",
                                "residue_name": "ALA",
                                "residue_number": "1",
                                "insertion_code": "",
                            },
                            {
                                "chain_id": "A",
                                "residue_name": "TYR",
                                "residue_number": "2",
                                "insertion_code": "",
                            },
                        ],
                        "descriptors": {},
                        "metadata": {},
                        "structure_path": "artifacts/pockets/pocket_001.pdb",
                        "points_path": "artifacts/pockets/pocket_001.pdb",
                    }
                ],
            }
        )
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "status": "completed",
                "job_type": "pocket_detection",
                "workflow": "pocket_detection",
                "tool": "fpocket",
                "parent_run_id": target_dir.name,
                "prepared_target_run_id": target_dir.name,
                "created_at": "2026-07-23T00:01:00+00:00",
            }
        )
    )
    (run_dir / "input.json").write_text(
        json.dumps(
            {
                "source_task_group": "protein-cleaning",
                "input_artifact": target_ref.to_dict(),
                "method": "fpocket",
                "parameters": {},
            }
        )
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "success": True,
                "method": "fpocket",
                "pocket_count": 1,
                "pocket_set": "artifacts/pockets/pocket_set.json",
            }
        )
    )
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(
                run_dir, pocket_structure, "pocket", role="fpocket-1"
            ),
            ArtifactRef.from_path(
                run_dir, pocket_set, "pocket_set", role="ranked_pockets"
            ),
        ],
    )

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    page.query_params["task_group"] = "pocket-detection"
    page.query_params["run_id"] = run_dir.name
    page.run(timeout=20)

    assert not page.exception
    assert any(
        "Full prepared target: light grey" in caption.value
        for caption in page.caption
    )
    assert next(select for select in page.selectbox if select.label == "Pocket").value == 1


def test_docking_page_renders_without_sources(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/docking.py").run(timeout=20)

    assert not page.exception
    assert [tab.label for tab in page.tabs[:5]] == [
        "Target", "Compounds", "Engine", "Run", "Results"
    ]
    assert {metric.label for metric in page.metric} >= {
        "CPU threads", "Free GPUs", "Queued jobs", "Active GPU leases"
    }
    assert any("No matching jobs" in item.value for item in page.info)


def test_docking_cofolding_page_groups_multi_engine_controls(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MN_LIGAND_REFERENCE_DIR", str(tmp_path / "references"))
    target_dir = tmp_path / "runs" / "target-trimming" / "target-1"
    target_dir.mkdir(parents=True)
    target_path = target_dir / "target.pdb"
    target_path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
        "  1.00 20.00           C\nEND\n"
    )
    (target_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": target_dir.name,
                "status": "completed",
                "workflow": "target_trimming",
            }
        )
    )
    write_artifact_manifest(
        target_dir,
        [ArtifactRef.from_path(target_dir, target_path, "prepared_target")],
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/docking_cofolding.py"
    ).run(timeout=20)

    assert not page.exception
    assert [tab.label for tab in page.tabs[:5]] == [
        "Target", "Compounds", "Engines", "Run", "Results"
    ]
    checkbox_labels = {control.label for control in page.checkbox}
    assert checkbox_labels >= {
        "Uni-Dock Pro", "AutoDock Vina", "GNINA", "RosettaLigand",
        "Boltz-2", "AlphaFold 3", "Nesso-1",
    }
    assert next(
        control for control in page.checkbox if control.label == "AutoDock Vina"
    ).value is True
    repetition_controls = [
        control
        for control in page.number_input
        if control.label == "Independent runs per structure"
    ]
    assert len(repetition_controls) == 1
    assert repetition_controls[0].value == 1
    repetition_controls[0].set_value(3).run(timeout=20)
    assert next(
        control
        for control in page.number_input
        if control.label == "Independent runs per structure"
    ).value == 3
    assert not any(control.key == "boltz_replicates" for control in page.number_input)
    assert not any(control.key == "af3_model_seeds" for control in page.number_input)
    assert not any(control.key == "nesso_replicates" for control in page.number_input)
    box_mode = next(
        control for control in page.segmented_control if control.label == "Box sizing"
    )
    assert box_mode.value == "Fixed box"
    shared_fixed_sizes = [
        control.value
        for control in page.number_input
        if str(control.key).startswith("binding_shared_box_size_")
    ]
    assert shared_fixed_sizes == [20.0, 20.0, 20.0]
    assert len(
        [
            control
            for control in page.number_input
            if str(control.key).startswith("binding_global_center_")
        ]
    ) == 3
    box_mode.set_value("Padding").run(timeout=20)
    assert not page.exception
    assert next(
        control
        for control in page.number_input
        if control.label == "Padding on each side (Å)"
    ).value == 15.0
    assert not any(
        str(control.key).startswith("binding_shared_box_size_")
        for control in page.number_input
    )
    padding = next(
        control
        for control in page.number_input
        if control.label == "Padding on each side (Å)"
    )
    padding.set_value(10.0).run(timeout=20)
    assert padding.value == 10.0
    assert next(
        button
        for button in page.button
        if button.label == "Queue selected docking / cofolding engines"
    ).disabled is True
    assert {metric.label for metric in page.metric} >= {
        "CPU threads", "Free GPUs", "Queued jobs", "Active GPU leases"
    }
    assert any(
        "not a binding affinity" in caption.value for caption in page.caption
    )


def test_docking_cofolding_selects_unique_parents_from_imports(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    monkeypatch.setenv(
        "MN_LIGAND_REFERENCE_DIR", str(tmp_path / "references")
    )
    create_compound_import_job(
        (
            b"id,smiles,name\n"
            b"free,CN,Parent\n"
            b"salt,C[NH3+].[Cl-],Parent hydrochloride\n"
            b"other,CCO,Other\n"
        ),
        filename="parents.csv",
        dataset_name="Imported parents",
        id_column="id",
        smiles_column="smiles",
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/docking_cofolding.py"
    ).run(timeout=20)

    assert not page.exception
    assert any(
        control.label == "Imported compound dataset"
        for control in page.selectbox
    )
    selection = next(
        control
        for control in page.radio
        if control.label == "Compound selection"
    )
    assert selection.value == "Manual selection"
    metrics = {metric.label: metric.value for metric in page.metric}
    assert metrics["Available unique parents"] == "2"
    assert metrics["Selected compounds"] == "1"


def test_docking_cofolding_inherits_target_associated_reference(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    monkeypatch.setenv("MN_LIGAND_REFERENCE_DIR", str(tmp_path / "references"))
    target_dir = runs_dir / "target-trimming" / "target-with-t3"
    artifact_dir = target_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    receptor = artifact_dir / "target_longest_axis_x.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
        "  1.00 20.00           C\nEND\n"
    )
    reference = artifact_dir / "ligand_1_longest_axis_x.sdf"
    reference.write_text(
        "T3\n  test\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    1.0000    2.0000    3.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n$$$$\n"
    )
    (target_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": target_dir.name,
                "job_code": "T3REF",
                "status": "completed",
                "workflow": "target_trimming",
                "job_type": "target_trimming",
                "tool": "principal-axis rigid transform",
                "ligand_key": "T3|A|501|_",
                "created_at": "2026-07-25T00:00:00+00:00",
            }
        )
    )
    receptor_ref = ArtifactRef.from_path(
        target_dir,
        receptor,
        "prepared_target",
        role="trimmed_receptor",
    )
    reference_ref = ArtifactRef.from_path(
        target_dir,
        reference,
        "prepared_ligand_set",
        role="axis_ligand",
    )
    write_artifact_manifest(target_dir, [receptor_ref, reference_ref])
    create_compound_import_job(
        b"id,smiles\ncandidate-1,CCO\n",
        filename="candidate.csv",
        dataset_name="Candidate",
        id_column="id",
        smiles_column="smiles",
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/docking_cofolding.py"
    ).run(timeout=20)

    assert not page.exception
    assert any(
        "Target-associated reference ligand" in frame.value["input"].tolist()
        for frame in page.dataframe
        if "input" in frame.value.columns
    )
    next(
        control
        for control in page.radio
        if control.label == "Compound selection"
    ).set_value("All unique parents")
    page.run(timeout=20)
    for checkbox in page.checkbox:
        if (
            str(checkbox.key).startswith("binding_engine_")
            and checkbox.label != "AutoDock Vina"
        ):
            checkbox.set_value(False)
    next(
        control
        for control in page.text_input
        if control.label == "Launch campaign name"
    ).set_value("Inherited-reference campaign")
    page.run(timeout=20)
    launch = next(
        button
        for button in page.button
        if button.label == "Queue selected docking / cofolding engines"
    )
    assert launch.disabled is False
    launch.click().run(timeout=20)
    assert not page.exception
    docking_inputs = [
        json.loads(path.read_text())
        for path in (runs_dir / "docking").glob("*/input.json")
    ]
    assert len(docking_inputs) == 1
    inherited = docking_inputs[0]["reference_ligand_artifact"]
    assert inherited["artifact_id"] == reference_ref.artifact_id
    assert inherited["run_id"] == target_dir.name


def test_docking_cofolding_queues_target_specific_engine_matrix(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    monkeypatch.setenv(
        "MN_LIGAND_REFERENCE_DIR", str(tmp_path / "references")
    )
    target_refs: list[ArtifactRef] = []
    for index in (1, 2):
        run_dir = runs_dir / "target-trimming" / f"target-{index}"
        run_dir.mkdir(parents=True)
        target_path = run_dir / f"target_{index}.pdb"
        target_path.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
            "  1.00 20.00           C\nEND\n"
        )
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_dir.name,
                    "job_code": f"TGT0{index}",
                    "pdb_id": f"TARGET-{index}",
                    "status": "completed",
                    "workflow": "target_trimming",
                    "created_at": f"2026-08-01T00:00:0{index}+00:00",
                }
            )
        )
        target_ref = ArtifactRef.from_path(
            run_dir,
            target_path,
            "prepared_target",
            role="trimmed_receptor",
        )
        write_artifact_manifest(run_dir, [target_ref])
        target_refs.append(target_ref)
    create_compound_import_job(
        b"id,smiles\ncandidate-1,CCO\n",
        filename="candidate.csv",
        dataset_name="Candidate",
        id_column="id",
        smiles_column="smiles",
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/docking_cofolding.py"
    )
    page.session_state["binding_targets_selected_ids"] = [
        f"target-{index}:{target_refs[index - 1].artifact_id}"
        for index in (1, 2)
    ]
    page.session_state["binding_target_mode"] = "Multi-target ensemble"
    for engine_key in (
        "binding_engine_uni_dock_pro",
        "binding_engine_rosettaligand",
        "binding_engine_boltz_2",
        "binding_engine_alphafold_3",
        "binding_engine_nesso_1",
    ):
        page.session_state[engine_key] = False
    page.session_state["binding_engine_autodock_vina"] = True
    page.session_state["binding_engine_gnina"] = True
    page.session_state["binding_compound_selection_mode_v2"] = (
        "All unique parents"
    )
    page.session_state[
        "binding_target_engine_target-1_autodock_vina"
    ] = True
    page.session_state["binding_target_engine_target-1_gnina"] = False
    page.session_state[
        "binding_target_engine_target-2_autodock_vina"
    ] = False
    page.session_state["binding_target_engine_target-2_gnina"] = True
    page.session_state["binding_launch_campaign_name"] = (
        "Two targets, selected engines"
    )
    page.run(timeout=20)

    assert not page.exception
    assert any(
        control.label == "Displayed target model"
        for control in page.selectbox
    )
    launch = next(
        button
        for button in page.button
        if button.label == "Queue selected docking / cofolding engines"
    )
    assert launch.disabled is False
    launch.click().run(timeout=20)
    assert not page.exception

    docking_inputs = [
        json.loads(path.read_text())
        for path in (runs_dir / "docking").glob("*/input.json")
    ]
    assert len(docking_inputs) == 2
    assert {row["parameters"]["engine"] for row in docking_inputs} == {
        "vina",
        "gnina",
    }
    assert {
        row["target_artifact"]["run_id"] for row in docking_inputs
    } == {"target-1", "target-2"}
    assert {
        row["parameters"]["launch_campaign_label"]
        for row in docking_inputs
    } == {"Two targets, selected engines"}
    assert len(
        {row["parameters"]["launch_campaign_id"] for row in docking_inputs}
    ) == 1


def test_structure_import_exposes_af3_promotion_and_trimming_workflow(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))

    structure_import = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/structure_preparation.py"
    ).run(timeout=20)
    assert not structure_import.exception
    assert {
        tab.label for tab in structure_import.tabs
    } >= {
        "From PDB",
        "From docking",
        "From Boltz-2 prediction",
        "From AlphaFold 3 prediction",
        "From custom files",
        "Results",
    }
    assert any(
        "No prepared Structure Import results" in item.value
        for item in structure_import.info
    )
    assert any(
        "No completed AlphaFold 3 Structure Import predictions" in item.value
        for item in structure_import.info
    )
    assert not any(control.label == "Use MSA server" for control in structure_import.checkbox)
    assert {
        control.label for control in structure_import.number_input
    } >= {
        "Recycles",
        "Number of model seeds",
        "First model seed",
        "MSA batch size",
    }
    assert any(
        "AlphaFast/MMseqs" in item.value and "no MSA server" in item.value
        for item in structure_import.info
    )
    assert {
        button.label for button in structure_import.button
    } >= {
        "Run Boltz-2 complex preparation",
        "Run AlphaFold 3 complex preparation",
    }
    assert next(
        button
        for button in structure_import.button
        if button.label == "Run Boltz-2 complex preparation"
    ).disabled is True
    assert next(
        button
        for button in structure_import.button
        if button.label == "Run AlphaFold 3 complex preparation"
    ).disabled is True
    assert sum(tab.label == "Completed predictions" for tab in structure_import.tabs) == 2
    assert any(
        "strict Ligand-X/PDBFixer cleaning and repair" in caption.value
        for caption in structure_import.caption
    )
    assert any(
        "Uploaded structures pass through the same strict Ligand-X/PDBFixer" in caption.value
        for caption in structure_import.caption
    )
    assert any(
        "Noncanonical chemistry must not be silently guessed" in caption.value
        for caption in structure_import.caption
    )
    assert not any(item.label == "Protein source" for item in structure_import.radio)
    assert {
        item.label for item in structure_import.file_uploader
    } >= {
        "Protein file (PDB/mmCIF)",
        "Ligand file (SDF/MOL2/SMILES TXT)",
        "Optional complete complex PDB",
    }
    assert any(
        "only for new local files" in caption.value
        for caption in structure_import.caption
    )

    trimming = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/target_trimming.py"
    ).run(timeout=20)
    assert not trimming.exception
    assert [tab.label for tab in trimming.tabs] == ["Target", "Trim", "Run", "Results"]
    assert next(
        button for button in trimming.button if button.label == "Create trimmed target"
    ).disabled is True
    assert any(
        "retaining its ligand unchanged" in caption.value for caption in trimming.caption
    )


def test_sequence_modification_combines_trimming_and_repair_pages() -> None:
    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/sequence_modification.py"
    ).run(timeout=20)

    assert not page.exception
    assert [title.value for title in page.title] == ["Target Sequence Modification"]
    assert [tab.label for tab in page.tabs] == [
        "Trimming",
        "Target",
        "Trim",
        "Run",
        "Results",
        "C-terminal Repair",
        "Target",
        "Repair",
        "Run",
        "Results",
    ]
    assert {heading.value for heading in page.subheader} == {
        "Target Trimming",
        "C-terminal Repair",
    }


def test_protein_cleaning_exposes_hiqbind_safety_and_assembly_controls(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))
    create_protein_import_job(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n",
        filename="target.pdb",
        source="upload",
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/protein_cleaning.py"
    ).run(timeout=20)

    assert not page.exception
    assert {
        item.label for item in page.checkbox
    } >= {
        "Map supported modified residues",
        "Skip missing terminal residues",
        "Refine rebuilt coordinates",
        "Retain non-water cofactors and metals",
        "Use GPU for local refinement",
        "Build a biological assembly",
    }
    assert next(
        item
        for item in page.number_input
        if item.label == "Maximum internal gap to rebuild"
    ).value == 15
    assert next(
        item
        for item in page.text_input
        if item.label == "Biological assembly ID"
    ).value == "1"
    assert any(
        "MODELLER builds ensembles" in item.value
        for item in page.caption
    )


def test_structure_import_results_lists_and_previews_typed_imports(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = runs_dir / "structure-jobs" / "import-result-1"
    run_dir.mkdir(parents=True)
    receptor = run_dir / "target_protein_refined.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"
    )
    ligand = run_dir / "target_ligand_refined.sdf"
    ligand.write_text(
        "LIG\n  mn-ligand\n\n"
        "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    2.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n$$$$\n"
    )
    complex_path = run_dir / "target_complex_refined.pdb"
    complex_path.write_text(
        receptor.read_text().replace(
            "END\n",
            "HETATM    2  C1  LIG B 101       2.000   0.000   0.000  1.00 20.00           C\nEND\n",
        )
    )
    repair = run_dir / "repair_report.json"
    repair.write_text(
        json.dumps(
            {
                "pdbfixer": {
                    "strict": True,
                    "sequence_records_available": True,
                    "missing_residue_segments_added": [
                        {
                            "chain": "A",
                            "insertion_index": 1,
                            "residues": ["GLY"],
                            "terminal": False,
                        }
                    ],
                    "missing_residue_segments_skipped": [],
                }
            }
        )
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "job_code": "IMP01",
                "status": "completed",
                "source": "custom",
                "pdb_id": "TEST",
                "ligand_key": "LIG|B|101|_",
                "created_at": "2026-07-23T00:00:00+00:00",
            }
        )
    )
    (run_dir / "result.json").write_text(json.dumps({"success": True}))
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(run_dir, receptor, "prepared_receptor", role="receptor"),
            ArtifactRef.from_path(run_dir, ligand, "prepared_ligand_set", role="ligand"),
            ArtifactRef.from_path(run_dir, complex_path, "prepared_complex", role="complex"),
            ArtifactRef.from_path(run_dir, repair, "repair_report", role="report"),
        ],
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/structure_preparation.py"
    ).run(timeout=20)

    assert not page.exception
    assert not any(
        item.label == "Structure Import result"
        for item in page.selectbox
    )
    assert any(
        item.label == "Job" and item.value == "IMP01"
        for item in page.metric
    )
    results_frame = next(
        frame.value
        for frame in page.dataframe
        if {
            "job",
            "Last step",
            "status",
            "source",
            "target",
            "receptor",
            "organism",
            "UniProt",
            "chains",
            "residues",
            "ligand",
            "compound",
            "formula",
            "MW (Da)",
            "preparation",
            "Origin / history",
            "MD readiness",
            "tool",
            "complex",
            "created",
        }.issubset(
            frame.value.columns
        )
    )
    assert "result" not in results_frame.columns
    assert list(results_frame.columns)[:2] == ["job", "Last step"]
    assert results_frame.iloc[0]["Last step"] == "Custom file"
    assert results_frame.iloc[0]["Origin / history"] == "Custom file"
    assert "label=IMP01" in str(results_frame.iloc[0]["job"])
    assert results_frame.iloc[0]["chains"] == "A"
    assert results_frame.iloc[0]["residues"] == 1
    assert results_frame.iloc[0]["formula"] == "CH4"
    assert float(results_frame.iloc[0]["MW (Da)"]) > 16
    assert results_frame.iloc[0]["MD readiness"] == "Review modeled gap"
    assert any(
        "missing residues were modeled" in item.value
        for item in page.warning
    )
    assert any(
        "Full imported structure" in item.value
        for item in page.markdown
    )
    assert any(
        set(["type", "role", "label", "relative path"]).issubset(frame.value.columns)
        for frame in page.dataframe
    )


def test_structure_import_predictions_exclude_discover_runs_and_show_structure(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    input_dir = runs_dir / "complex-prediction-inputs" / "import-input"
    input_dir.mkdir(parents=True)
    (input_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": input_dir.name,
                "workflow": "complex_prediction_inputs",
                "status": "completed",
            }
        )
    )
    structure = (
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "HETATM    2  C1  LIG B   1       2.000   0.000   0.000  1.00 20.00           C\n"
        "END\n"
    )
    for run_id, parent_run_id, code in (
        ("imported-af3", input_dir.name, "IMPORT"),
        ("discover-af3", "prepared-target", "DISCOVER"),
    ):
        run_dir = runs_dir / "refolding" / run_id
        output_dir = run_dir / "output"
        output_dir.mkdir(parents=True)
        predicted = output_dir / f"{run_id}.pdb"
        predicted.write_text(structure)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "job_code": code,
                    "workflow": "alphafold3_refolding",
                    "status": "completed",
                    "parent_run_id": parent_run_id,
                }
            )
        )
        write_artifact_manifest(
            run_dir,
            [ArtifactRef.from_path(run_dir, predicted, "predicted_complex", role="ligand")],
        )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/structure_preparation.py"
    ).run(timeout=20)

    assert not page.exception
    prediction_select = next(
        item for item in page.selectbox if item.label == "AlphaFold 3 predicted complex"
    )
    assert any("imported-af3.pdb" in option for option in prediction_select.options)
    assert not any("discover-af3.pdb" in option for option in prediction_select.options)
    assert any("Predicted complex" in heading.value for heading in page.markdown)
    assert any("Protein is shown" in caption.value for caption in page.caption)


def test_redocking_benchmark_is_a_separate_workflow_page(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/redocking_benchmark.py"
    ).run(timeout=20)

    assert not page.exception
    assert [tab.label for tab in page.tabs[:5]] == [
        "Target", "Reference ligand", "Engines", "Run", "Results"
    ]
    assert {
        control.label for control in page.checkbox
    } >= {"Uni-Dock Pro", "AutoDock Vina", "GNINA"}
    assert next(
        control for control in page.segmented_control if control.label == "Box sizing"
    ).value == "Fixed box"
    assert [
        control.value
        for control in page.number_input
        if control.label in {"size_x", "size_y", "size_z"}
    ] == [20.0, 20.0, 20.0]
    assert next(
        button for button in page.button if button.label == "Run redocking benchmark"
    ).disabled is True
    assert any(
        "symmetry-aware heavy-atom RMSD" in caption.value for caption in page.caption
    )


def test_docking_page_exposes_typed_redocking_benchmark_controls(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/docking.py").run(timeout=20)
    operation = next(control for control in page.segmented_control if control.label == "Operation")
    operation.set_value("Redocking benchmark").run(timeout=20)

    assert not page.exception
    assert next(control for control in page.multiselect if control.label == "Engines").value == [
        "Uni-Dock Pro", "AutoDock Vina", "GNINA"
    ]
    assert next(
        control for control in page.number_input if control.label == "Independent runs"
    ).value == 3
    assert any("coordinate-bearing crystallographic reference ligand" in item.value for item in page.info)
    assert next(button for button in page.button if button.label == "Run redocking benchmark").disabled is True


def test_docking_page_exposes_general_independent_runs_and_seed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/docking.py").run(timeout=20)

    assert not page.exception
    repeats = next(
        control for control in page.number_input if control.label == "Independent runs"
    )
    assert repeats.value == 1
    assert repeats.disabled is False
    assert next(
        control for control in page.number_input if control.label == "First docking seed"
    ).value == 1001

    operation = next(control for control in page.segmented_control if control.label == "Operation")
    operation.set_value("Refolding").run(timeout=20)
    repeats = next(
        control for control in page.number_input if control.label == "Independent runs"
    )
    assert repeats.disabled is False
    assert repeats.value == 1
    assert not any(
        getattr(link, "label", "") == "Open Pocket Detection"
        for link in page.get("link_button")
    )
    assert not any(
        control.label.startswith(("center_", "size_")) for control in page.number_input
    )
    assert not any(control.label == "Docker image" for control in page.text_input)
    assert next(
        control for control in page.number_input if control.label == "First prediction seed"
    ).value == 1001

    engine = next(control for control in page.segmented_control if control.label == "Engine")
    engine.set_value("AlphaFold 3").run(timeout=20)
    repeats = next(
        control for control in page.number_input if control.label == "Independent runs"
    )
    assert repeats.disabled is True


def test_docking_page_renders_nesso_as_affinity_only_engine(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MN_LIGAND_REFERENCE_DIR", str(tmp_path / "references"))

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/docking.py").run(timeout=20)
    operation = next(control for control in page.segmented_control if control.label == "Operation")
    operation.set_value("Refolding").run(timeout=20)
    engine = next(control for control in page.segmented_control if control.label == "Engine")
    engine.set_value("Nesso-1").run(timeout=20)

    assert not page.exception
    assert any("does not emit a predicted complex or pose" in item.value for item in page.info)
    assert next(
        control for control in page.number_input if control.label == "Independent runs"
    ).disabled is False
    assert next(
        control for control in page.number_input if control.label == "First prediction seed"
    ).value == 42
    assert any("Nesso requires its v1.0.0 checkpoint" in item.value for item in page.warning)
    assert next(button for button in page.button if "Nesso-1" in button.label).disabled is True


def test_docking_page_renders_openvs_cpu_protocol_controls(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/docking.py").run(timeout=20)
    engine = next(control for control in page.segmented_control if control.label == "Engine")
    engine.set_value("RosettaLigand").run(timeout=20)

    assert not page.exception
    assert next(control for control in page.selectbox if control.label == "Protocol").value == (
        "VSH — high precision"
    )
    assert next(control for control in page.selectbox if control.label == "Placement").value == (
        "Reference-guided"
    )
    assert next(control for control in page.number_input if control.label == "Parallel CPU workers")
    gpu = next(control for control in page.selectbox if control.label == "GPU")
    assert gpu.value == "Not used"
    assert gpu.disabled is True
    assert not any(control.label == "Docker image" for control in page.text_input)
    assert any("PDBQT is not used" in caption.value for caption in page.caption)
    assert not any("Reference-guided RosettaLigand requires" in item.value for item in page.info)


def test_docking_page_renders_openvs_convergence_controls(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/docking.py").run(timeout=20)
    engine = next(control for control in page.segmented_control if control.label == "Engine")
    engine.set_value("RosettaLigand").run(timeout=20)
    protocol = next(control for control in page.selectbox if control.label == "Protocol")
    protocol.set_value("Convergence — exhaustive multi-seed VSH").run(timeout=20)

    assert not page.exception
    assert next(
        control for control in page.number_input if control.label == "Independent runs"
    ).value == 1
    assert next(
        control for control in page.number_input if control.label == "First docking seed"
    ).value == 1001
    assert next(
        control
        for control in page.number_input
        if control.label == "Pose-cluster threshold (Å)"
    ).value == 2.0
    assert next(
        control for control in page.number_input if control.label == "Parallel CPU workers"
    ).value == max(1, __import__("os").cpu_count() or 1)
    assert any(
        "prepares inputs once" in item.value and "SD ≤2 REU" in item.value
        for item in page.info
    )


def test_job_results_renders_redocking_summary_and_rmsd_method(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = runs_dir / "workflows" / "redocking-ui-1"
    metrics = run_dir / "metrics" / "redocking_summary.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        "engine,replicate_count,mean_top_rmsd_angstrom,sd_top_rmsd_angstrom\n"
        "vina,3,0.624,0.184\n"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "status": "completed",
                "job_type": "workflow",
                "workflow": "redocking_benchmark",
                "operation": "redocking",
                "created_at": "2026-07-22T00:00:00+00:00",
            }
        )
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "success": True,
                "redocking_finalized": True,
                "rmsd_method": "symmetry-aware heavy-atom direct RMSD in the shared receptor frame",
            }
        )
    )
    write_artifact_manifest(
        run_dir,
        [ArtifactRef.from_path(run_dir, metrics, "redocking_summary", role="summary")],
    )

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    page.query_params["task_group"] = "workflows"
    page.query_params["run_id"] = run_dir.name
    page.run(timeout=20)

    assert not page.exception
    assert any("RMSD is symmetry-aware" in caption.value for caption in page.caption)
    assert any("vina" in frame.value.to_string() for frame in page.dataframe)


def test_job_results_renders_general_docking_replicate_summary(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = runs_dir / "docking" / "docking-repeats-ui-1"
    run_dir.mkdir(parents=True)
    scores = run_dir / "scores.csv"
    scores.write_text(
        "compound_id,engine,replicate,seed,best_score_kcal_mol,pose_file\n"
        "candidate-1,vina,1,1001,-7.0,results/replicate_001/candidate-1_out.pdbqt\n"
        "candidate-1,vina,2,1002,-8.0,results/replicate_002/candidate-1_out.pdbqt\n"
    )
    summary = run_dir / "docking_replicate_summary.csv"
    summary.write_text(
        "compound_id,replicate_count,mean_score_kcal_mol,sample_sd_score_kcal_mol,"
        "representative_replicate\ncandidate-1,2,-7.5,0.7071,1\n"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "status": "completed",
                "job_type": "docking_campaign",
                "workflow": "docking_campaign",
                "operation": "docking",
                "tool": "vina",
                "replicates": 2,
                "created_at": "2026-07-23T00:00:00+00:00",
            }
        )
    )
    (run_dir / "result.json").write_text(
        json.dumps({"success": True, "replicates": 2})
    )
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(
                run_dir, scores, "docking_scores", role="ranked_scores"
            ),
            ArtifactRef.from_path(
                run_dir, summary, "docking_scores", role="replicate_summary"
            ),
        ],
    )

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    page.query_params["task_group"] = "docking"
    page.query_params["run_id"] = run_dir.name
    page.run(timeout=20)

    assert not page.exception
    assert any("sample standard deviations" in item.value for item in page.caption)
    assert any("candidate-1" in frame.value.to_string() for frame in page.dataframe)


def test_job_results_renders_docked_pose_in_receptor_with_reference(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    target_dir = runs_dir / "structure-jobs" / "target-ui-1"
    target_dir.mkdir(parents=True)
    receptor = target_dir / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
        "  1.00 20.00           C\nEND\n"
    )
    reference = target_dir / "reference.sdf"
    reference.write_text(
        "T3\n  test\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    1.0000    1.0000    1.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n$$$$\n"
    )
    (target_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": target_dir.name,
                "status": "completed",
            }
        )
    )
    receptor_ref = ArtifactRef.from_path(
        target_dir, receptor, "prepared_receptor", role="receptor"
    )
    reference_ref = ArtifactRef.from_path(
        target_dir, reference, "prepared_ligand_set", role="ligand"
    )
    write_artifact_manifest(target_dir, [receptor_ref, reference_ref])

    run_dir = runs_dir / "docking" / "docking-viewer-ui-1"
    pose_dir = run_dir / "results" / "replicate_001"
    pose_dir.mkdir(parents=True)
    pose = pose_dir / "candidate-1_out.sdf"
    pose.write_text(
        "candidate-1\n  test\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    2.0000    2.0000    2.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n$$$$\n"
    )
    scores = run_dir / "scores.csv"
    scores.write_text(
        "compound_id,engine,replicate,seed,best_score_kcal_mol,pose_file\n"
        "candidate-1,vina,1,1001,-8.25,"
        "results/replicate_001/candidate-1_out.pdbqt\n"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "status": "completed",
                "workflow": "docking_campaign",
                "job_type": "docking_campaign",
            }
        )
    )
    (run_dir / "result.json").write_text(json.dumps({"success": True}))
    (run_dir / "input.json").write_text(
        json.dumps(
            {
                "target_artifact": receptor_ref.to_dict(),
            }
        )
    )
    write_artifact_manifest(
        run_dir,
        [ArtifactRef.from_path(run_dir, scores, "docking_scores", role="ranked_scores")],
    )

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    page.query_params["task_group"] = "docking"
    page.query_params["run_id"] = run_dir.name
    page.run(timeout=20)

    assert not page.exception
    assert any(item.label == "Docked compound" for item in page.selectbox)
    assert any(
        item.label == "Docking score" and item.value == "-8.250 kcal/mol"
        for item in page.metric
    )
    assert any(
        "receptor cartoon is grey" in item.value
        and "T3 template/reference is green" in item.value
        for item in page.caption
    )
    show_all = next(
        item
        for item in page.checkbox
        if item.label == "Show all replicates together"
    )
    show_reference = next(
        item for item in page.checkbox if item.label == "Show T3 reference"
    )
    assert show_all.value is False
    assert show_reference.value is True
    show_all.set_value(True).run(timeout=20)
    assert not page.exception
    assert any(
        item.label == "Mean docking score" for item in page.metric
    )
    assert any(
        item.label == "Replicates" and item.value == "1"
        for item in page.metric
    )
    show_reference = next(
        item for item in page.checkbox if item.label == "Show T3 reference"
    )
    show_reference.set_value(False).run(timeout=20)
    assert not page.exception
    assert not any(
        "template/reference is green" in item.value for item in page.caption
    )


def test_job_results_renders_nesso_repetition_summary_with_micromolar_units(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = runs_dir / "refolding" / "nesso-repeats-ui-1"
    run_dir.mkdir(parents=True)
    per_run = run_dir / "nesso_affinity.csv"
    per_run.write_text(
        "candidate_id,replicate,seed,affinity_log10_ic50_uM,ic50_uM\n"
        "T3,1,42,0.5,3.1623\nT3,2,43,0.6,3.9811\n"
    )
    summary = run_dir / "nesso_replicate_summary.csv"
    summary.write_text(
        "candidate_id,replicate_count,mean_affinity_log10_ic50_uM,"
        "sample_sd_affinity_log10_ic50_uM,geometric_mean_ic50_uM,"
        "arithmetic_mean_ic50_uM,sample_sd_ic50_uM\n"
        "T3,2,0.55,0.07071,3.5481,3.5717,0.57898\n"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1, "run_id": run_dir.name, "status": "completed",
                "job_type": "learned_binding_affinity", "workflow": "nesso_affinity",
                "operation": "refolding", "tool": "Nesso-1", "replicates": 2,
                "created_at": "2026-07-23T00:00:00+00:00",
            }
        )
    )
    (run_dir / "result.json").write_text(
        json.dumps({"success": True, "replicates": 2, "structure_output": False})
    )
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(
                run_dir, per_run, "prediction_metrics", role="affinity_campaign"
            ),
            ArtifactRef.from_path(
                run_dir, summary, "prediction_metrics", role="replicate_summary"
            ),
        ],
    )

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    page.query_params["task_group"] = "refolding"
    page.query_params["run_id"] = run_dir.name
    page.run(timeout=20)

    assert not page.exception
    assert any("arithmetic and geometric IC50 in µM" in item.value for item in page.caption)
    assert any("T3" in frame.value.to_string() for frame in page.dataframe)
    assert len(page.get("vega_lite_chart")) >= 4
    assert any(
        "Nesso mean affinity with run-to-run uncertainty" in item.value
        for item in page.markdown
    )


def test_job_results_renders_alphafold3_interface_confidence_without_affinity_claim(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = runs_dir / "refolding" / "af3-confidence-ui-1"
    run_dir.mkdir(parents=True)
    metrics = run_dir / "alphafold3_metrics.csv"
    metrics.write_text(
        "candidate_id,ranking_score,iptm,ptm,fraction_disordered,has_clash\n"
        "T3,0.8,0.7,0.6,0.05,false\n"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "status": "completed",
                "workflow": "alphafold3_refolding",
                "operation": "refolding",
                "tool": "AlphaFold 3",
                "created_at": "2026-07-23T00:00:00+00:00",
            }
        )
    )
    (run_dir / "result.json").write_text(json.dumps({"success": True}))
    write_artifact_manifest(
        run_dir,
        [ArtifactRef.from_path(run_dir, metrics, "prediction_metrics", role="summary")],
    )

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    page.query_params["task_group"] = "refolding"
    page.query_params["run_id"] = run_dir.name
    page.run(timeout=20)

    assert not page.exception
    assert any("T3" in frame.value.to_string() for frame in page.dataframe)
    assert any(
        "They are not binding affinities" in item.value for item in page.caption
    )
    assert len(page.get("vega_lite_chart")) >= 3
    assert any(
        "Predicted disorder fraction" in item.value for item in page.markdown
    )


def test_campaign_comparison_filters_targets_and_combines_native_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    target_structure = (
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  GLY A   2       3.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  CA  SER A   3       3.000   3.000   0.000  1.00 20.00           C\n"
        "HETATM    4  C1  LIG B   1       1.000   1.000   1.000  1.00 20.00           C\n"
        "END\n"
    )
    for target_id in ("target-a", "target-b"):
        target_dir = runs_dir / "structure-jobs" / target_id
        target_dir.mkdir(parents=True)
        (target_dir / "input.pdb").write_text(target_structure)
    fixtures = (
        (
            "af3-compare-1",
            "alphafold3_refolding",
            "AlphaFold 3",
            "target-a",
            "Target A",
            "candidate_id,model_seed,sample,ranking_score,iptm,ptm,structure_file\n"
            "cmp-1,1,0,0.80,0.70,0.75,prediction-1.pdb\n"
            "cmp-1,1,1,0.99,0.99,0.99,prediction-ignore.pdb\n"
            "cmp-1,2,0,0.84,0.72,0.77,prediction-2.pdb\n"
            "cmp-2,1,0,0.60,0.55,0.65,prediction-3.pdb\n"
            "cmp-3,1,0,0.70,0.62,0.68,prediction-4.pdb\n"
            "cmp-4,1,0,0.50,0.48,0.58,prediction-5.pdb\n"
            "cmp-5,1,0,0.90,0.82,0.86,prediction-6.pdb\n",
        ),
        (
            "boltz-compare-1",
            "boltz2_refolding",
            "Boltz-2",
            "target-a",
            "Target A",
            "candidate_id,compound_id,model_id,replicate,"
            "affinity_probability_binary,confidence_score,structure_file\n"
            "cmp-1_model_0,cmp-1,cmp-1_model_0,1,0.90,0.80,boltz-1.pdb\n"
            "cmp-1_model_1,cmp-1,cmp-1_model_1,1,0.10,0.20,boltz-2.pdb\n"
            "cmp-2_model_0,cmp-2,cmp-2_model_0,1,0.50,0.70,boltz-3.pdb\n"
            "cmp-3_model_0,cmp-3,cmp-3_model_0,1,0.65,0.72,boltz-4.pdb\n"
            "cmp-4_model_0,cmp-4,cmp-4_model_0,1,0.35,0.60,boltz-5.pdb\n"
            "cmp-5_model_0,cmp-5,cmp-5_model_0,1,0.95,0.88,boltz-6.pdb\n",
        ),
        (
            "nesso-compare-1",
            "nesso_affinity",
            "Nesso-1",
            "target-b",
            "Target B",
            "candidate_id,replicate,pIC50,binder_probability\n"
            "cmp-1,1,7.0,0.90\ncmp-1,2,7.2,0.92\n"
            "cmp-2,1,5.0,0.40\n",
        ),
        (
            "vina-compare-1",
            "docking_campaign",
            "vina",
            "target-a",
            "Target A",
            "compound_id,replicate,seed,best_score_kcal_mol,pose_file\n"
            "cmp-1,1,1001,-10.0,pose-1.pdbqt\n"
            "cmp-1,2,1002,-9.8,pose-2.pdbqt\n"
            "cmp-2,1,1001,-8.0,pose-3.pdbqt\n"
            "cmp-3,1,1001,-8.8,pose-4.pdbqt\n"
            "cmp-4,1,1001,-7.2,pose-5.pdbqt\n"
            "cmp-5,1,1001,-11.0,pose-6.pdbqt\n",
        ),
        (
            "gnina-compare-1",
            "docking_campaign",
            "gnina",
            "target-a",
            "Target A",
            "compound_id,replicate,seed,best_score_kcal_mol,"
            "cnn_score,cnn_affinity,pose_file\n"
            "cmp-1,1,1001,-9.5,0.8,7.1,pose-1.pdbqt\n"
            "cmp-2,1,1001,-7.8,0.7,6.2,pose-3.pdbqt\n"
            "cmp-3,1,1001,-8.4,0.6,5.8,pose-4.pdbqt\n"
            "cmp-4,1,1001,-7.0,0.5,5.1,pose-5.pdbqt\n"
            "cmp-5,1,1001,-10.5,0.9,8.0,pose-6.pdbqt\n",
        ),
    )
    for (
        run_id,
        workflow,
        tool,
        target_id,
        target_label,
        csv_text,
    ) in fixtures:
        task_group = (
            "docking" if workflow == "docking_campaign" else "refolding"
        )
        run_dir = runs_dir / task_group / run_id
        run_dir.mkdir(parents=True)
        metric_path = run_dir / "metrics.csv"
        metric_path.write_text(csv_text)
        if workflow == "alphafold3_refolding":
            for filename in (
                "prediction-1.pdb",
                "prediction-ignore.pdb",
                "prediction-2.pdb",
                "prediction-3.pdb",
                "prediction-4.pdb",
                "prediction-5.pdb",
                "prediction-6.pdb",
            ):
                (run_dir / filename).write_text(target_structure)
        if workflow == "boltz2_refolding":
            for filename in (
                "boltz-1.pdb",
                "boltz-2.pdb",
                "boltz-3.pdb",
                "boltz-4.pdb",
                "boltz-5.pdb",
                "boltz-6.pdb",
            ):
                (run_dir / filename).write_text(target_structure)
        if workflow == "docking_campaign":
            sdf = (
                "pose\n  mn-ligand\n\n"
                "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
                "    1.0000    1.0000    1.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
                "M  END\n$$$$\n"
            )
            for filename in (
                "pose-1.sdf",
                "pose-2.sdf",
                "pose-3.sdf",
                "pose-4.sdf",
                "pose-5.sdf",
                "pose-6.sdf",
            ):
                (run_dir / filename).write_text(sdf)
                (run_dir / Path(filename).with_suffix(".pdbqt")).write_text(
                    "MODEL 1\n"
                    "HETATM    1  C1  LIG A   1       1.000   1.000   1.000  0.00  0.00    C\n"
                    "ENDMDL\n"
                )
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "job_code": run_id[:5].upper(),
                    "status": "completed",
                    "workflow": workflow,
                    "operation": "refolding",
                    "tool": tool,
                    "compound_count": 2,
                    "created_at": "2026-07-25T00:00:00+00:00",
                }
            )
        )
        (run_dir / "result.json").write_text(
            json.dumps({"success": True})
        )
        (run_dir / "input.json").write_text(
            json.dumps(
                {
                    "target": {
                        "run_id": target_id,
                        "label": target_label,
                        "path": "input.pdb",
                    },
                    "compound_sets": [
                        {
                            "run_id": "selection-1",
                            "label": "Shared compounds",
                            "metadata": {
                                "source_compound_run_id": "dataset-1"
                            },
                        }
                    ],
                }
            )
        )
        if workflow == "docking_campaign":
            (run_dir / "input.json").write_text(
                json.dumps(
                    {
                        "target_artifact": {
                            "run_id": target_id,
                            "label": target_label,
                            "path": "input.pdb",
                        },
                        "compound_artifacts": [
                            {
                                "run_id": "selection-1",
                                "label": "Shared compounds",
                                "metadata": {
                                    "source_compound_run_id": "dataset-1"
                                },
                            }
                        ],
                    }
                )
            )
        write_artifact_manifest(
            run_dir,
            [
                ArtifactRef.from_path(
                    run_dir,
                    metric_path,
                    "docking_scores"
                    if workflow == "docking_campaign"
                    else "prediction_metrics",
                    role=(
                        "affinity_campaign"
                        if workflow == "nesso_affinity"
                        else "summary"
                    ),
                )
            ],
        )

    validation_dir = runs_dir / "pose-validation" / "validation-compare-1"
    validation_dir.mkdir(parents=True)
    (validation_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "validation-compare-1",
                "job_code": "VAL01",
                "status": "completed",
                "workflow": "posebusters_validation",
                "operation": "pose_validation",
                "tool": "posebusters",
                "parent_run_id": "vina-compare-1",
                "selection_schema_version": 2,
                "selection_policy": "best-scientific-poses-v1",
                "created_at": "2026-07-26T00:00:00+00:00",
            }
        )
    )
    (validation_dir / "result.json").write_text(
        json.dumps({"success": True})
    )
    (validation_dir / "input.json").write_text("{}")
    (validation_dir / "posebusters_report.json").write_text(
        json.dumps(
            {
                "validated_count": 2,
                "applicable_checks": [
                    "bond_lengths",
                    "minimum_distance_to_protein",
                ],
            }
        )
    )
    (validation_dir / "posebusters_summary.csv").write_text(
        "pose_id,compound_id,replicate,prediction,passed_all,"
        "failed_checks\n"
        "pose-1,cmp-1,1,pose-1,True,\n"
        "pose-3,cmp-2,1,pose-3,False,bond_lengths\n"
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/campaign_comparison.py"
    ).run(timeout=20)

    assert not page.exception
    tab_labels = [tab.label for tab in page.tabs]
    assert {
        "Overview",
        "Scores & ranking",
        "Structural evidence",
        "Target × compound explorer",
        "Data & Analysis Sets",
        "Native metrics",
        "Correlations",
        "Rescoring",
        "Consensus",
        "Pose validity",
        "Interactions",
        "Data",
        "Analysis Sets",
    }.issubset(tab_labels)
    assert any(
        item.value == "#### Compound physical-validity map"
        for item in page.markdown
    )
    assert any(
        item.label == "Compounds with ≥1 PASS" and item.value == "1"
        for item in page.metric
    )
    assert {
        control.label for control in page.multiselect
    } >= {"Targets", "Launch campaigns", "Engines", "Engine runs"}
    assert any(
        control.label == "Compound dataset" for control in page.selectbox
    )
    targets = next(
        control for control in page.multiselect if control.label == "Targets"
    )
    assert len(targets.options) == 2
    engines = next(
        control for control in page.multiselect if control.label == "Engines"
    )
    assert "AutoDock Vina" in engines.options
    targets.set_value(["target-b"]).run(timeout=20)
    assert not page.exception
    engines = next(
        control for control in page.multiselect if control.label == "Engines"
    )
    assert engines.options == ["Nesso-1"]
    assert engines.value == ["Nesso-1"]
    targets = next(
        control for control in page.multiselect if control.label == "Targets"
    )
    targets.set_value(["target-a", "target-b"]).run(timeout=20)
    assert not page.exception
    engines = next(
        control for control in page.multiselect if control.label == "Engines"
    )
    assert set(engines.value) == set(engines.options)
    assert "AutoDock Vina" in engines.value
    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/campaign_comparison.py"
    ).run(timeout=20)
    assert not page.exception
    native_scores = {
        control.label: control.value
        for control in page.selectbox
        if control.label.endswith(" score")
    }
    repetition_mode = next(
        control
        for control in page.segmented_control
        if control.label == "Repetitions included"
    )
    assert repetition_mode.value == "Best repetitions"
    assert next(
        control
        for control in page.number_input
        if control.label == "Best repetitions per compound and campaign"
    ).value == 1
    assert native_scores["AlphaFold 3 score"] == "iptm"
    assert (
        native_scores["Boltz-2 score"]
        == "affinity_probability_binary"
    )
    assert native_scores["GNINA score"] == "cnn_ranked_cnn_affinity"
    assert native_scores["AutoDock Vina score"] == "best_score_kcal_mol"
    explorer_view = next(
        control
        for control in page.segmented_control
        if control.label == "Explore selected results"
    )
    explorer_view.set_value("RMSD & pose agreement").run(timeout=20)
    assert not page.exception
    assert any(
        control.label == "Poses to compare"
        for control in page.segmented_control
    )
    assert any(
        "Pose similarity" in item.value for item in page.markdown
    )
    assert any(
        "RMSD pose coverage" in expander.label for expander in page.expander
    )
    rmsd_coverage = next(
        frame.value
        for frame in page.dataframe
        if {"Engine", "Included in RMSD", "Exclusion reason"}.issubset(
            frame.value.columns
        )
    )
    assert {
        "AlphaFold 3",
        "Boltz-2",
        "GNINA",
        "AutoDock Vina",
    }.issubset(set(rmsd_coverage["Engine"]))
    cofolded = rmsd_coverage.loc[
        rmsd_coverage["Engine"].isin(["AlphaFold 3", "Boltz-2"])
    ]
    assert not cofolded.empty
    assert cofolded["Included in RMSD"].all()
    excluded = rmsd_coverage.loc[~rmsd_coverage["Included in RMSD"]]
    assert excluded["Exclusion reason"].astype(str).str.strip().ne("").all()
    rmsd_mode = next(
        control
        for control in page.segmented_control
        if control.label == "RMSD comparison"
    )
    assert rmsd_mode.options == [
        "Recovery from input pose",
        "Prediction agreement",
    ]
    assert rmsd_mode.value == "Recovery from input pose"
    assert any(
        "Input-pose RMSD (Å)" in frame.value.columns
        for frame in page.dataframe
    )
    rmsd_mode.set_value("Prediction agreement").run(timeout=20)
    assert not page.exception
    assert any(
        "matrix lists all" in caption.value for caption in page.caption
    )
    explorer_view = next(
        control
        for control in page.segmented_control
        if control.label == "Explore selected results"
    )
    explorer_view.set_value("3D structures").run(timeout=20)
    assert not page.exception
    assert any(
        control.label == "Predictions to display"
        for control in page.segmented_control
    )
    assert not any(
        "Pose similarity" in item.value for item in page.markdown
    )
    assert {
        control.label for control in page.checkbox
    } >= {
        "Show target protein",
        "Show input/reference ligand",
        "Show predicted proteins",
    }
    viewer_legends = [
        frame.value
        for frame in page.dataframe
        if "structure_file" in frame.value.columns
    ]
    assert viewer_legends
    expected_engine_order = [
        "AlphaFold 3",
        "Boltz-2",
        "GNINA",
        "Uni-Dock Pro",
        "AutoDock Vina",
        "RosettaLigand",
    ]
    viewer_engines = list(
        dict.fromkeys(viewer_legends[0]["engine"].tolist())
    )
    assert viewer_engines == [
        engine
        for engine in expected_engine_order
        if engine in viewer_engines
    ]
    vina_rows = viewer_legends[0].loc[
        viewer_legends[0]["engine"].eq("AutoDock Vina")
    ]
    assert not vina_rows.empty
    assert vina_rows["structure_file"].str.endswith(".sdf").all()
    viewer_layout = next(
        control
        for control in page.segmented_control
        if control.label == "Viewer layout"
    )
    assert viewer_layout.value == "Single compound"
    viewer_layout.set_value("Compound matrix").run(timeout=20)
    assert not page.exception
    assert any(
        control.label == "Focus compounds"
        for control in page.multiselect
    )
    assert not any(
        control.label == "Matrix structure-producing campaigns"
        for control in page.multiselect
    )
    assert any(
        "linked py3Dmol camera" in caption.value
        for caption in page.caption
    )
    assert any(
        "One square RMSD matrix is shown per 3D compound panel"
        in caption.value
        for caption in page.caption
    )
    matrix_legends = [
        frame.value
        for frame in page.dataframe
        if {"compound", "structure_file"}.issubset(frame.value.columns)
    ]
    assert matrix_legends
    for _, compound_rows in matrix_legends[0].groupby(
        "compound",
        sort=False,
    ):
        compound_engines = list(
            dict.fromkeys(compound_rows["engine"].tolist())
        )
        assert compound_engines == [
            engine
            for engine in expected_engine_order
            if engine in compound_engines
        ]
    assert len(page.get("vega_lite_chart")) >= 2
    assert any(
        "not pooled" in caption.value for caption in page.caption
    )
    correlation_view = next(
        control
        for control in page.segmented_control
        if control.label == "View"
    )
    correlation_view.set_value("Scatterplot matrix").run(timeout=20)
    assert not page.exception
    assert any(
        control.label == "Scatterplot-matrix metrics"
        for control in page.multiselect
    )
    collection_name = next(
        control
        for control in page.text_input
        if control.label == "Analysis Set name"
    )
    collection_name.set_value("Fixture comparison")
    save_button = next(
        button
        for button in page.button
        if button.label == "Save Analysis Set"
    )
    save_button.click().run(timeout=20)
    assert not page.exception
    collection_dirs = list(
        (runs_dir / "campaign-comparison-collections").iterdir()
    )
    assert len(collection_dirs) == 1
    saved_selection = json.loads(
        (collection_dirs[0] / "selection.json").read_text()
    )
    assert saved_selection["dataset_run_id"] == "dataset-1"
    assert set(saved_selection["engines"]) >= {
        "AlphaFold 3",
        "AutoDock Vina",
    }
    assert any(
        "Fixture comparison" in frame.value.to_string()
        for frame in page.dataframe
    )
    reopened = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/campaign_comparison.py"
    )
    reopened.query_params["analysis_set_id"] = collection_dirs[0].name
    reopened.run(timeout=20)
    assert not reopened.exception
    assert next(
        control
        for control in reopened.selectbox
        if control.label == "Compound dataset"
    ).value == "dataset-1"
    assert set(
        next(
            control
            for control in reopened.multiselect
            if control.label == "Engines"
        ).value
    ) == set(saved_selection["engines"])


def test_job_results_renders_openvs_reu_scores_and_md_handoff(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = runs_dir / "docking" / "openvs-ui-1"
    run_dir.mkdir(parents=True)
    scores = run_dir / "openvs_scores.csv"
    scores.write_text(
        "compound_id,engine,protocol,estimated_dg_reu,pose_file\n"
        "candidate-1,openvs,VSH,-18.25,results/candidate-1.complex.pdb\n"
    )
    complex_path = run_dir / "best_openvs_complex.pdb"
    complex_path.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "HETATM    2  C1  LG1 X   2       1.000   1.000   1.000  1.00 20.00           C\nEND\n"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "status": "completed",
                "job_type": "openvs_docking",
                "workflow": "openvs_docking",
                "operation": "docking",
                "tool": "openvs",
                "created_at": "2026-07-23T00:00:00+00:00",
            }
        )
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "success": True,
                "best_compound_id": "candidate-1",
                "best_estimated_dg_reu": -18.25,
                "score_units": "Rosetta energy units (REU; relative ranking only)",
            }
        )
    )
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(
                run_dir, scores, "docking_scores", role="ranked_scores"
            ),
            ArtifactRef.from_path(
                run_dir, complex_path, "docked_complex", role="best_ranked_complex"
            ),
        ],
    )

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    page.query_params["task_group"] = "docking"
    page.query_params["run_id"] = run_dir.name
    page.run(timeout=20)

    assert not page.exception
    assert any("candidate-1" in frame.value.to_string() for frame in page.dataframe)
    assert any("not kcal/mol" in caption.value for caption in page.caption)
    assert any(
        getattr(button, "label", "") == "Prepare for MD"
        for button in page.get("link_button")
    )
def test_sdf_record_selection_removes_separator_newline(tmp_path: Path) -> None:
    from mn_ligand.app.pages.job_results import _sdf_record

    first = (
        "first\n  test\n\n"
        "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n$$$$\n"
    )
    second = first.replace("first", "second", 1)
    path = tmp_path / "poses.sdf"
    path.write_text(first + second)

    selected = _sdf_record(path, 2)

    assert selected.startswith("second\n")
    assert not selected.startswith("\n")
    assert selected.endswith("$$$$\n")


def test_first_sdf_record_preserves_a_legal_blank_title(tmp_path: Path) -> None:
    from mn_ligand.app.pages.job_results import _first_sdf_record

    path = tmp_path / "blank-title.sdf"
    path.write_text(
        "\n"
        "  RDKit          3D\n\n"
        "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0"
        "  0  0  0  0  0\n"
        "M  END\n$$$$\n"
    )

    record = _first_sdf_record(path)

    assert record.startswith("\n  RDKit")
    assert record.endswith("$$$$\n")


def test_ligand_redesign_page_is_separate_and_starts_without_contact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    monkeypatch.setenv("MN_LIGAND_REFERENCE_DIR", str(tmp_path / "references"))

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/ligand_redesign.py"
    )
    page.run(timeout=20)

    assert not page.exception
    assert page.title[0].value == "Ligand Redesign"
    assert any(
        "atom-masked native inference" in item.value
        for item in page.caption
    )
    visible_text = " ".join(
        str(item.value)
        for collection in (page.caption, page.markdown, page.info, page.warning)
        for item in collection
    )
    assert "SER277" not in visible_text


def test_task_specific_ligand_design_pages_start_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MN_LIGAND_REFERENCE_DIR", str(tmp_path / "references"))
    expected = {
        "fragment_growing.py": "Fragment Growing",
        "scaffold_hopping.py": "Scaffold Hopping",
        "ligand_optimization.py": "Ligand Optimization",
    }
    for filename, title in expected.items():
        page = AppTest.from_file(
            PROJECT_DIR / "mn_ligand/app/pages" / filename
        ).run(timeout=20)
        assert not page.exception
        assert page.title[0].value == title


def test_generation_results_navigate_normalized_and_native_sdf_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "molecule-generation" / "generation-viewer-1"
    normalized_dir = run_dir / "normalized"
    native_dir = run_dir / "native" / "batch-1"
    input_dir = run_dir / "input"
    normalized_dir.mkdir(parents=True)
    native_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)

    def sdf_record(name: str, properties: dict[str, str] | None = None) -> str:
        property_text = "".join(
            f">  <{key}>\n{value}\n\n"
            for key, value in (properties or {}).items()
        )
        return (
            f"{name}\n"
            "  mn-ligand          3D\n\n"
            "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "M  END\n"
            f"{property_text}"
            "$$$$\n"
        )

    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "job_type": "molecule_generation",
                "workflow": "molecule_generation",
                "status": "completed",
                "tool": "Fixture generator",
            }
        )
    )
    (run_dir / "input.json").write_text(
        json.dumps(
            {
                "pocket_artifact": {
                    "run_id": "pocket-source-1",
                    "metadata": {
                        "method": "bound_ligand",
                        "descriptors": {"lining_residue_count": 3},
                    },
                }
            }
        )
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "requested_count": 3,
                "valid_output_count": 2,
                "unique_valid_compound_count": 2,
            }
        )
    )
    (normalized_dir / "generation_report.json").write_text(
        json.dumps(
            {
                "requested_count": 3,
                "valid_output_count": 2,
                "unique_valid_compound_count": 2,
            }
        )
    )
    (normalized_dir / "generated_compounds.csv").write_text(
        "compound_id,canonical_isomeric_smiles,generation_engine,"
        "native_source,native_index,coordinate_dimension\n"
        "fixture-1,C,fixture,batch-1/gen.sdf,0,3\n"
        "fixture-2,N,fixture,batch-1/gen.sdf,2,3\n"
    )
    (normalized_dir / "generated_compounds.sdf").write_text(
        sdf_record(
            "fixture-1",
            {
                "compound_id": "fixture-1",
                "canonical_isomeric_smiles": "C",
            },
        )
        + sdf_record(
            "fixture-2",
            {
                "compound_id": "fixture-2",
                "canonical_isomeric_smiles": "N",
            },
        )
    )
    (native_dir / "gen.sdf").write_text(
        sdf_record("RAW-1")
        + sdf_record("RAW-EXCLUDED")
        + sdf_record("RAW-3")
    )
    context_pdb = (
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
        "  1.00 20.00           C\n"
        "ATOM      2  CA  GLY A   2       2.000   0.000   0.000"
        "  1.00 20.00           C\n"
        "ATOM      3  CA  SER A   3       0.000   2.000   1.000"
        "  1.00 20.00           C\nEND\n"
    )
    (input_dir / "target.pdb").write_text(context_pdb)
    (input_dir / "pocket.pdb").write_text(context_pdb)
    (input_dir / "reference_ligand.sdf").write_text(
        sdf_record("reference-ligand")
    )
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    page.query_params["task_group"] = "molecule-generation"
    page.query_params["run_id"] = run_dir.name
    page.run(timeout=20)

    assert not page.exception
    source = next(item for item in page.selectbox if item.label == "Compound set")
    compound = next(item for item in page.selectbox if item.label == "Compound")
    assert source.options == [
        (
            "Normalized generated compounds — 2 RDKit-valid records "
            "(pre-qualification)"
        ),
        "Native engine output — native/batch-1/gen.sdf — 3 raw records",
    ]
    assert compound.options == ["1. fixture-1", "2. fixture-2"]
    assert next(
        item for item in page.metric if item.label == "Normalization"
    ).value == "RDKit-valid"
    checkbox_labels = {item.label for item in page.checkbox}
    assert {
        "Show generated compound",
        "Show pocket context",
        "Show reference ligand",
        "Show full receptor overlay",
    }.issubset(checkbox_labels)
    assert next(
        item for item in page.checkbox if item.label == "Show reference ligand"
    ).value
    next(
        item
        for item in page.checkbox
        if item.label == "Show full receptor overlay"
    ).set_value(True).run(timeout=20)
    assert not page.exception
    assert any(
        "Receptor overlay alignment" in item.value
        for item in page.caption
    )
    assert any(
        "not reconstructed from the pharmacophore" in item.value
        for item in page.caption
    )

    source = next(item for item in page.selectbox if item.label == "Compound set")
    source.select_index(1)
    page.run(timeout=20)
    compound = next(item for item in page.selectbox if item.label == "Compound")
    assert compound.options == [
        "1. RAW-1 — normalized as fixture-1",
        "2. RAW-EXCLUDED — raw only (not in normalized set)",
        "3. RAW-3 — normalized as fixture-2",
    ]
    compound.select_index(1)
    page.run(timeout=20)

    assert not page.exception
    assert next(
        item for item in page.metric if item.label == "Normalization"
    ).value == "Raw only"
    assert any(
        "absent from the normalized compound set" in item.value
        for item in page.warning
    )


def test_molecule_qualification_results_show_gates_and_qualified_3d(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "molecule-qualification" / "qualification-viewer-1"
    qualified_dir = run_dir / "qualified"
    qualified_dir.mkdir(parents=True)
    candidate_dir = qualified_dir / "candidates"
    candidate_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "job_type": "molecule_qualification",
                "workflow": "molecule_qualification",
                "status": "completed",
                "tool": "RDKit + PoseBusters",
                "parent_run_id": "generation-source-1",
                "qualified_compound_count": 1,
            }
        )
    )
    (run_dir / "input.json").write_text("{}")
    report = {
        "success": True,
        "input_count": 3,
        "chemical_pass_count": 2,
        "conformer_pass_count": 2,
        "posebusters_pass_count": 1,
        "posebusters_hard_pass_count": 1,
        "qualified_with_warning_count": 0,
        "qualified_compound_count": 1,
    }
    (run_dir / "result.json").write_text(json.dumps(report))
    (qualified_dir / "qualification_report.json").write_text(json.dumps(report))
    (qualified_dir / "qualification.csv").write_text(
        "compound_id,canonical_isomeric_smiles,chemical_pass,"
        "conformer_generation_pass,posebusters_pass,qualified_for_docking,"
        "qualification_status,review_warnings,force_field,qed,sa_score\n"
        "qualified-1,CCO,True,True,True,True,qualified,,MMFF94s,0.4,2.1\n"
        "rejected-1,OO,False,False,False,False,rejected,,,,4.0\n"
        "rejected-2,C1CC1,True,True,False,False,rejected,,UFF,0.3,3.0\n"
    )
    candidate_sdf = (
        "qualified-1\n"
        "  mn-ligand          3D\n\n"
        "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n"
        ">  <compound_id>\nqualified-1\n\n"
        ">  <canonical_isomeric_smiles>\nCCO\n\n"
        "$$$$\n"
    )
    (qualified_dir / "qualified_compounds.sdf").write_text(candidate_sdf)
    (candidate_dir / "qualified-1.sdf").write_text(candidate_sdf)
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/job_results.py")
    page.query_params["task_group"] = "molecule-qualification"
    page.query_params["run_id"] = run_dir.name
    page.run(timeout=20)

    assert not page.exception
    metrics = {item.label: item.value for item in page.metric}
    assert metrics["Input"] == "3"
    assert metrics["Chemical pass"] == "2"
    assert metrics["Valid 3D"] == "2"
    assert metrics["PB strict pass"] == "1"
    assert metrics["Review warnings"] == "0"
    assert metrics["Accepted for docking"] == "1"
    compound = next(
        item for item in page.selectbox if item.label == "3D candidate"
    )
    assert compound.options == ["1. qualified-1 — qualified"]
    assert metrics["Qualification"] == "Qualified"


def test_campaign_metric_summary_combines_supplemental_attempt_jobs() -> None:
    import pandas as pd

    from mn_ligand.app.pages.campaign_comparison import summarize_metric

    rows = pd.DataFrame(
        [
            {
                "candidate_id": "compound-1",
                "campaign_id": "af3-original",
                "campaign": "AlphaFold 3 original",
                "launch_campaign_id": "launch-1",
                "launch_campaign": "Combined launch",
                "engine": "AlphaFold 3",
                "target_run_id": "target-1",
                "target": "Target 1",
                "dataset": "Dataset",
                "ranking_score": 0.9,
            },
            {
                "candidate_id": "compound-1",
                "campaign_id": "af3-supplemental",
                "campaign": "AlphaFold 3 supplemental",
                "launch_campaign_id": "launch-1",
                "launch_campaign": "Combined launch",
                "engine": "AlphaFold 3",
                "target_run_id": "target-1",
                "target": "Target 1",
                "dataset": "Dataset",
                "ranking_score": 0.7,
            },
        ]
    )

    summary = summarize_metric(rows, "ranking_score")

    assert len(summary) == 1
    assert summary.iloc[0]["Attempts"] == 2
    assert summary.iloc[0]["Mean"] == 0.8
    assert summary.iloc[0]["campaign"] == "Combined launch · Target 1"


def test_pose_validation_display_splits_both_gnina_criteria() -> None:
    import pandas as pd

    from mn_ligand.app.pages.campaign_comparison import (
        _pose_validation_display_rows,
    )

    rows = pd.DataFrame(
        [
            {
                "engine": "GNINA",
                "pose_id": "shared",
                "selection_criterion": (
                    "CNN pose score + Vina/empirical score"
                ),
                "passed_all": True,
            },
            {
                "engine": "GNINA",
                "pose_id": "cnn-only",
                "selection_criterion": "CNN pose score",
                "passed_all": False,
            },
            {
                "engine": "AutoDock Vina",
                "pose_id": "vina",
                "selection_criterion": "Best emitted pose",
                "passed_all": True,
            },
        ]
    )

    displayed = _pose_validation_display_rows(rows)

    assert displayed["validation_group"].tolist() == [
        "GNINA (CNN score)",
        "GNINA (Vina score)",
        "GNINA (CNN score)",
        "AutoDock Vina",
    ]
    assert displayed.loc[
        displayed["pose_id"].eq("shared"), "passed_all"
    ].tolist() == [True, True]


def test_campaign_metric_chart_orders_best_compound_first() -> None:
    import pandas as pd

    from mn_ligand.app.pages.campaign_comparison import _metric_chart

    summary = pd.DataFrame(
        {
            "candidate_id": ["middle", "best", "worst"],
            "campaign": ["campaign"] * 3,
            "target": ["target"] * 3,
            "dataset": ["dataset"] * 3,
            "Mean": [-8.0, -12.0, -5.0],
            "Sample SD": [0.1, 0.2, 0.3],
            "Attempts": [3, 3, 3],
            "Lower": [-8.1, -12.2, -5.3],
            "Upper": [-7.9, -11.8, -4.7],
        }
    )

    lower_first = _metric_chart(
        summary,
        "Docking score",
        higher_is_better=False,
    ).to_dict()
    higher_first = _metric_chart(
        summary,
        "Confidence",
        higher_is_better=True,
    ).to_dict()

    assert lower_first["layer"][0]["encoding"]["x"]["sort"] == [
        "best",
        "middle",
        "worst",
    ]
    assert higher_first["layer"][0]["encoding"]["x"]["sort"] == [
        "worst",
        "middle",
        "best",
    ]
    assert lower_first["layer"][0]["encoding"]["xOffset"][
        "bandPosition"
    ] == 0.5
    assert lower_first["layer"][1]["encoding"]["xOffset"][
        "bandPosition"
    ] == 0.5


def test_campaign_native_metric_attempt_selection_respects_direction() -> None:
    import pandas as pd

    from mn_ligand.app.pages.campaign_comparison import (
        select_metric_attempts,
    )

    rows = pd.DataFrame(
        {
            "candidate_id": ["compound"] * 3,
            "campaign_id": ["run-1"] * 3,
            "campaign": ["campaign"] * 3,
            "launch_campaign_id": ["launch"] * 3,
            "launch_campaign": ["campaign"] * 3,
            "engine": ["engine"] * 3,
            "target_run_id": ["target"] * 3,
            "target": ["target"] * 3,
            "dataset": ["dataset"] * 3,
            "score": [-5.0, -12.0, -8.0],
            "replicate": [1, 2, 3],
        }
    )

    lower_best = select_metric_attempts(
        rows,
        "score",
        mode="Best repetitions",
        best_count=2,
        higher_is_better=False,
    )
    higher_best = select_metric_attempts(
        rows,
        "score",
        mode="Best repetitions",
        best_count=1,
        higher_is_better=True,
    )
    representative = select_metric_attempts(
        rows,
        "score",
        mode="Representative repetition",
        best_count=1,
        higher_is_better=False,
    )

    assert lower_best["replicate"].tolist() == [2, 3]
    assert higher_best["replicate"].tolist() == [1]
    assert representative["replicate"].tolist() == [3]


def test_md_selection_identifies_pi_specific_calls_for_normalized_display() -> None:
    import pandas as pd

    from mn_ligand.app.pages.campaign_comparison import (
        _md_selection_pi_specific_mask,
    )

    interactions = pd.DataFrame(
        {
            "interaction_type": [
                "hydrogen bond",
                "hydrophobic contact",
                "salt bridge",
                "alkyl–π",
                "carbon-π",
                "π–π stacking",
                "cation–π",
            ]
        }
    )

    assert _md_selection_pi_specific_mask(interactions).tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]


def test_md_selection_reference_report_uses_reviewed_residue_set() -> None:
    import pandas as pd

    from mn_ligand.app.pages.campaign_comparison import (
        _md_selection_filtered_reference_report,
    )

    report = {
        "contact_consensus": [
            {"residue": "ALA225 · chain A", "mean_contact_occupancy": 0.9},
            {"residue": "PHE218 · chain A", "mean_contact_occupancy": 0.2},
            {"residue": "LEU222 · chain B", "mean_contact_occupancy": 0.8},
        ],
        "ligand_depiction": {"atoms": [], "bonds": []},
    }
    reviewed = pd.DataFrame(
        [
            {
                "protein_chain": "A",
                "protein_residue_name": "ALA",
                "protein_residue_number": "225",
            },
            {
                "protein_chain": "A",
                "protein_residue_name": "PHE",
                "protein_residue_number": "218",
            },
        ]
    )

    filtered = _md_selection_filtered_reference_report(report, reviewed)

    assert [row["residue"] for row in filtered["contact_consensus"]] == [
        "ALA225 · chain A",
        "PHE218 · chain A",
    ]
    assert len(report["contact_consensus"]) == 3


def test_scatter_matrix_labels_break_engine_metric_and_unit() -> None:
    from mn_ligand.app.pages.campaign_comparison import (
        _scatter_matrix_display_label,
    )

    assert _scatter_matrix_display_label(
        "GNINA · Best Vina pose · minimized Vina score (kcal/mol) "
        "(favorable ↑)"
    ) == (
        "GNINA\nBest Vina pose\nminimized Vina score\n(kcal/mol)"
    )
