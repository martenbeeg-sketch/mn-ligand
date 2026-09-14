from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

from rdkit import Chem
from streamlit.testing.v1 import AppTest

from mn_ligand.core.artifacts import write_artifact_manifest
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord
from mn_ligand.workflows.rescoring import (
    _pdbqt_models,
    create_pose_selection_job,
    finalize_gnina_rescoring_job,
    queue_boltzina_rescoring_job,
    queue_gnina_rescoring_job,
    source_pose_rows,
)


POSE = """REMARK VINA RESULT: {score} 0.0 0.0
ROOT
ATOM      1  C   UNL     1       {x:5.3f}   0.000   0.000  1.00  0.00     0.000 C
ENDROOT
TORSDOF 0
"""


def _source_job(tmp_path: Path) -> JobRecord:
    run_dir = tmp_path / "source"
    (run_dir / "input").mkdir(parents=True)
    results = run_dir / "results" / "replicate_001"
    results.mkdir(parents=True)
    (run_dir / "input" / "receptor.pdb").write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"
    )
    (run_dir / "input" / "compounds.tsv").write_text(
        "compound_id\tsmiles\nethanol\tCCO\n"
    )
    first = POSE.format(score=-7.0, x=1.0)
    second = POSE.format(score=-6.0, x=2.0)
    (results / "ethanol_out.pdbqt").write_text(
        first + "ENDMDL\nMODEL 2\n" + second + "ENDMDL\n"
    )
    writer = Chem.SDWriter(str(results / "ethanol_out.sdf"))
    for rank, x in enumerate((1.0, 2.0), start=1):
        molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        for atom_index in range(molecule.GetNumAtoms()):
            conformer.SetAtomPosition(atom_index, (x + atom_index * 0.1, 0.0, 0.0))
        molecule.RemoveAllConformers()
        molecule.AddConformer(conformer)
        molecule.SetProp("_Name", f"ethanol-{rank}")
        writer.write(molecule)
    writer.close()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "job_code": "SRC01",
        "job_type": "docking_campaign",
        "workflow": "docking_campaign",
        "operation": "docking",
        "status": "completed",
        "tool": "vina",
        "engine": "vina",
        "compound_count": 1,
        "replicates": 1,
        "prepared_target_run_id": "target",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    (run_dir / "result.json").write_text(json.dumps({"success": True}))
    write_artifact_manifest(run_dir, [])
    return JobRecord.load(run_dir, task_group="docking")


def test_pdbqt_models_split_vina_output_without_moving_coordinates() -> None:
    text = (
        POSE.format(score=-7.0, x=1.0)
        + "ENDMDL\nMODEL 2\n"
        + POSE.format(score=-6.0, x=2.0)
        + "ENDMDL\n"
    )
    models = _pdbqt_models(text)
    assert len(models) == 2
    assert "MODEL" not in "".join(models)
    assert "-7.0" in models[0]
    assert "-6.0" in models[1]
    assert "   2.000" in models[1]


def test_pose_selection_is_immutable_and_preserves_all_selected_ranks(
    tmp_path: Path,
) -> None:
    source = _source_job(tmp_path)
    rows = source_pose_rows(source)
    assert [row["source_pose_rank"] for row in rows] == [1, 2]

    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        selection = create_pose_selection_job(
            source,
            compound_ids=("ethanol",),
            replicates=(1,),
            pose_ranks=(1, 2),
        )

    assert selection.status == "completed"
    assert selection.workflow == "pose_selection"
    assert selection.parent_run_id == source.run_id
    assert selection.result["pose_count"] == 2
    table = list(
        csv.DictReader((selection.run_dir / "pose_selection.csv").open())
    )
    assert [int(row["source_pose_rank"]) for row in table] == [1, 2]
    assert all((selection.run_dir / row["pose_file"]).is_file() for row in table)
    assert len(
        [
            molecule
            for molecule in Chem.SDMolSupplier(
                str(selection.run_dir / "selected_poses.sdf"),
                removeHs=False,
                sanitize=False,
            )
            if molecule is not None
        ]
    ) == 2


def test_gnina_rescoring_queue_uses_score_only_and_shared_selection(
    tmp_path: Path,
) -> None:
    source = _source_job(tmp_path)
    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        selection = create_pose_selection_job(source, pose_ranks=(1,))
        queued = queue_gnina_rescoring_job(
            selection_job=selection,
            gpu_device="1",
        )

    assert queued.status == "queued"
    assert queued.metadata["worker_finalizer"] == "gnina_rescoring"
    assert queued.parent_run_id == selection.run_id
    command = queued.metadata["queued_command"]
    assert command[command.index("--gpus") + 1] == "device=1"
    shell = command[-1]
    assert "--score_only" in shell
    assert "--minimize" not in shell


def test_gnina_finalizer_verifies_coordinate_preservation(tmp_path: Path) -> None:
    source = _source_job(tmp_path)
    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        selection = create_pose_selection_job(source, pose_ranks=(1,))
        queued = queue_gnina_rescoring_job(selection_job=selection)
    selected = next(
        csv.DictReader((selection.run_dir / "pose_selection.csv").open())
    )
    output = queued.run_dir / "results" / f"{selected['pose_id']}.pdbqt"
    output.write_text(
        (selection.run_dir / selected["pose_file"]).read_text()
        + "REMARK minimizedAffinity -7.25\n"
        + "REMARK CNNscore 0.75\n"
        + "REMARK CNNaffinity 6.5\n"
    )
    (queued.run_dir / "stdout.log").write_text("")
    (queued.run_dir / "stderr.log").write_text("")

    completed = finalize_gnina_rescoring_job(queued.run_dir, returncode=0)

    assert completed.status == "completed"
    assert completed.result["maximum_coordinate_displacement_angstrom"] == 0.0
    row = next(csv.DictReader((completed.run_dir / "rescoring_scores.csv").open()))
    assert float(row["gnina_cnn_score"]) == 0.75


def test_boltzina_queue_uses_pose_scoring_runner_and_processed_context(
    tmp_path: Path,
) -> None:
    source = _source_job(tmp_path)
    work_dir = tmp_path / "boltz-work"
    (work_dir / "processed").mkdir(parents=True)
    (work_dir / "processed" / "manifest.json").write_text(
        json.dumps({"records": [{"id": "context"}]})
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        selection = create_pose_selection_job(source, pose_ranks=(1,))
        queued = queue_boltzina_rescoring_job(
            selection_job=selection,
            boltz_work_dir=work_dir,
            cache_dir=cache,
            gpu_device="0",
        )

    assert queued.status == "queued"
    assert queued.metadata["worker_finalizer"] == "boltzina_rescoring"
    assert queued.parent_run_id == selection.run_id
    command = queued.metadata["queued_command"]
    assert command[command.index("--gpus") + 1] == "device=0"
    assert "ovolig-boltzina-cu128:latest" in command
    assert "/boltz-context" in command
    assert "/workspace/boltz_work" in command
    assert "--poses" in command


def test_rescoring_page_renders_multi_engine_workflow_without_sources(
    tmp_path: Path,
) -> None:
    project_dir = Path(__file__).parents[1]
    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        page = AppTest.from_file(
            project_dir / "mn_ligand" / "app" / "pages" / "rescoring.py"
        ).run(timeout=20)

    assert not page.exception
    assert [tab.label for tab in page.tabs] == [
        "Source results",
        "Poses",
        "Engines",
        "Run",
        "Results",
    ]
    assert {box.label for box in page.checkbox} >= {
        "GNINA score-only",
        "Boltzina",
    }
    assert next(
        button
        for button in page.button
        if button.label == "Queue selected rescoring engines"
    ).disabled
