from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from streamlit.testing.v1 import AppTest

from mn_ligand.core.jobs import JobRecord
from mn_ligand.workflows.pose_validation import (
    _source_smiles,
    cached_pose_validation_candidates,
    pose_validation_candidates,
    pose_validation_inventory_path,
    pose_validation_inventory_summary,
    queue_pose_validation_job,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _docking_fixture(
    runs_dir: Path,
    *,
    run_id: str = "docking-posebusters-fixture",
    job_code: str = "PB001",
) -> JobRecord:
    run_dir = runs_dir / "docking" / run_id
    result_dir = run_dir / "results" / "replicate_001"
    input_dir = run_dir / "input"
    result_dir.mkdir(parents=True)
    input_dir.mkdir()
    (input_dir / "receptor.pdb").write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"
    )
    pose_text = (
        "MODEL 1\n"
        "REMARK VINA RESULT: -8.0 0.0 0.0\n"
        "HETATM    1  C1  LIG L   1       1.000   0.000   0.000  1.00  0.00      0.000 C\n"
        "ENDMDL\n"
        "MODEL 2\n"
        "REMARK VINA RESULT: -7.0 0.0 0.0\n"
        "HETATM    1  C1  LIG L   1       2.000   0.000   0.000  1.00  0.00      0.000 C\n"
        "ENDMDL\n"
    )
    (result_dir / "cmp-1_out.pdbqt").write_text(pose_text)
    molecule = Chem.AddHs(Chem.MolFromSmiles("C"))
    AllChem.EmbedMolecule(molecule, randomSeed=7)
    writer = Chem.SDWriter(str(result_dir / "cmp-1_out.sdf"))
    writer.write(molecule)
    writer.write(molecule)
    writer.close()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "job_code": job_code,
                "status": "completed",
                "workflow": "docking_campaign",
                "tool": "vina",
                "engine": "vina",
                "created_at": "2026-07-25T00:00:00+00:00",
            }
        )
    )
    (run_dir / "input.json").write_text("{}")
    (run_dir / "result.json").write_text(json.dumps({"success": True}))
    return JobRecord.load(run_dir, task_group="docking")


def test_posebusters_selection_and_queue_are_cpu_native(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source = _docking_fixture(runs_dir)
    rows = pose_validation_candidates(source)
    assert len(rows) == 1
    assert rows[0]["representative"] is True
    assert rows[0]["selection_criterion"] == "Best emitted pose"

    job = queue_pose_validation_job(
        source,
        selected_rows=rows[:1],
        max_workers=3,
    )
    assert job.status == "queued"
    assert job.metadata["resources"]["gpu"] is False
    assert job.metadata["resources"]["cpu_threads"] == 1
    assert job.metadata["cpu_worker_policy"] == "adaptive-global-limit"
    assert "--gpus" not in job.metadata["queued_command"]
    command = job.metadata["queued_command"]
    assert "OMP_NUM_THREADS=1" in command
    assert "OPENBLAS_NUM_THREADS=1" in command
    assert "RAYON_NUM_THREADS=1" in command
    assert (job.run_dir / "input" / "poses" / "cmp-1__replicate_001__pose_001.sdf").is_file()


def test_pose_validation_candidate_inventory_is_persisted_and_reused(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source = _docking_fixture(runs_dir)

    first = cached_pose_validation_candidates(source)
    cache_path = pose_validation_inventory_path(source)
    payload = json.loads(cache_path.read_text())

    assert len(first) == 1
    assert payload["candidate_count"] == 1
    assert payload["rows"][0]["_source_relative_path"].startswith(
        "results/"
    )
    assert "_source_path" not in payload["rows"][0]
    summary = pose_validation_inventory_summary(source)
    assert summary == {
        "cached": True,
        "candidate_count": 1,
        "compound_ids": ("cmp-1",),
    }

    def fail_rescan(_source: JobRecord) -> list[dict[str, object]]:
        raise AssertionError("cached inventory should avoid rescanning")

    monkeypatch.setattr(
        "mn_ligand.workflows.pose_validation.pose_validation_candidates",
        fail_rescan,
    )
    second = cached_pose_validation_candidates(source)

    assert [row["selection_id"] for row in second] == [
        row["selection_id"] for row in first
    ]
    assert all(Path(row["_source_path"]).is_absolute() for row in second)


def test_gnina_pose_validation_keeps_both_scientific_rankings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source = _docking_fixture(runs_dir)
    metadata = json.loads((source.run_dir / "metadata.json").read_text())
    metadata["tool"] = "gnina"
    metadata["engine"] = "gnina"
    (source.run_dir / "metadata.json").write_text(json.dumps(metadata))
    pose_path = (
        source.run_dir / "results" / "replicate_001" / "cmp-1_out.pdbqt"
    )
    pose_path.write_text(
        "MODEL 1\n"
        "REMARK minimizedAffinity -9.0\n"
        "REMARK CNNscore 0.90\n"
        "REMARK CNNaffinity 7.0\n"
        "HETATM    1  C1  LIG L   1       1.000   0.000   0.000  1.00  0.00      0.000 C\n"
        "ENDMDL\n"
        "MODEL 2\n"
        "REMARK minimizedAffinity -11.0\n"
        "REMARK CNNscore 0.60\n"
        "REMARK CNNaffinity 8.0\n"
        "HETATM    1  C1  LIG L   1       2.000   0.000   0.000  1.00  0.00      0.000 C\n"
        "ENDMDL\n"
    )
    source = JobRecord.load(source.run_dir, task_group="docking")
    rows = pose_validation_candidates(source)

    assert len(rows) == 2
    assert [row["pose_rank"] for row in rows] == [1, 2]
    assert [row["selection_criterion"] for row in rows] == [
        "CNN pose score",
        "Vina/empirical score",
    ]


def test_gnina_pose_validation_deduplicates_identical_best_pose(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source = _docking_fixture(runs_dir)
    metadata = json.loads((source.run_dir / "metadata.json").read_text())
    metadata["tool"] = "gnina"
    metadata["engine"] = "gnina"
    (source.run_dir / "metadata.json").write_text(json.dumps(metadata))
    pose_path = (
        source.run_dir / "results" / "replicate_001" / "cmp-1_out.pdbqt"
    )
    pose_path.write_text(
        "MODEL 1\n"
        "REMARK minimizedAffinity -11.0\n"
        "REMARK CNNscore 0.90\n"
        "REMARK CNNaffinity 8.0\n"
        "HETATM    1  C1  LIG L   1       1.000   0.000   0.000  1.00  0.00      0.000 C\n"
        "ENDMDL\n"
        "MODEL 2\n"
        "REMARK minimizedAffinity -9.0\n"
        "REMARK CNNscore 0.60\n"
        "REMARK CNNaffinity 7.0\n"
        "HETATM    1  C1  LIG L   1       2.000   0.000   0.000  1.00  0.00      0.000 C\n"
        "ENDMDL\n"
    )
    source = JobRecord.load(source.run_dir, task_group="docking")
    rows = pose_validation_candidates(source)

    assert len(rows) == 1
    assert rows[0]["pose_rank"] == 1
    assert (
        rows[0]["selection_criterion"]
        == "CNN pose score + Vina/empirical score"
    )


def test_pose_validation_page_renders_without_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))
    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/pose_validation.py"
    ).run(timeout=20)
    assert not page.exception
    assert [tab.label for tab in page.tabs] == [
        "Target / Input",
        "Tool / Engine",
        "Run",
        "Results",
    ]
    assert next(
        button
        for button in page.button
        if button.label.startswith("Queue ")
    ).disabled is True


def test_pose_validation_page_defaults_to_all_missing_and_batches_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    covered = _docking_fixture(
        runs_dir,
        run_id="covered-source",
        job_code="COVER",
    )
    _docking_fixture(
        runs_dir,
        run_id="missing-source",
        job_code="MISS",
    )
    queue_pose_validation_job(
        covered,
        selected_rows=pose_validation_candidates(covered)[:1],
        max_workers=1,
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/pose_validation.py"
    ).run(timeout=20)

    assert not page.exception
    assert not any(
        control.label == "Predictions to validate"
        for control in page.radio
    )
    coverage = next(
        frame.value
        for frame in page.dataframe
        if "validation status" in frame.value.columns
    )
    assert {
        control.label for control in page.multiselect
    } >= {"Engines", "Validation status"}
    assert set(coverage["validation status"]) == {"Queued", "Not run"}
    assert bool(
        coverage.loc[
            coverage["validation status"].eq("Not run"),
            "selected",
        ].iloc[0]
    )
    assert not bool(
        coverage.loc[
            coverage["validation status"].eq("Queued"),
            "selected",
        ].iloc[0]
    )
    next(
        button for button in page.button if button.label == "Clear selection"
    ).click().run(timeout=20)
    assert not page.exception
    coverage = next(
        frame.value
        for frame in page.dataframe
        if "validation status" in frame.value.columns
    )
    assert not coverage["selected"].fillna(False).astype(bool).any()
    next(
        button
        for button in page.button
        if button.label == "Select all missing"
    ).click().run(timeout=20)
    assert not page.exception
    coverage = next(
        frame.value
        for frame in page.dataframe
        if "validation status" in frame.value.columns
    )
    assert bool(
        coverage.loc[
            coverage["validation status"].eq("Not run"),
            "selected",
        ].iloc[0]
    )
    submit = next(
        button
        for button in page.button
        if button.label.startswith("Queue ")
    )
    assert submit.label == "Queue 1 PoseBusters job"
    submit.click().run(timeout=20)
    assert not page.exception
    assert len(
        list((runs_dir / "pose-validation").iterdir())
    ) == 2


def test_cofolding_uses_exact_native_input_smiles(tmp_path: Path) -> None:
    run_dir = tmp_path / "refolding" / "boltz-native-input"
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "inputs" / "compound_0000001.yaml").write_text(
        "version: 1\n"
        "sequences:\n"
        "  - ligand:\n"
        '      id: "L"\n'
        '      smiles: "N[C@@H](C)C(=O)O"\n'
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "status": "completed",
                "workflow": "boltz2_refolding",
                "tool": "Boltz-2",
            }
        )
    )
    (run_dir / "input.json").write_text("{}")
    (run_dir / "result.json").write_text("{}")
    job = JobRecord.load(run_dir, task_group="refolding")

    assert _source_smiles(job) == {
        "compound_0000001": "N[C@@H](C)C(=O)O"
    }
