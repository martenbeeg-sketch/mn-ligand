from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JobRecord
from mn_ligand.workflows.predicted_complex_promotion import promote_predicted_complex
from mn_ligand.workflows.target_trimming import (
    create_trimmed_target_job,
    pdb_chain_ranges,
    trim_pdb_data,
)


PDB_COMPLEX = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  N   GLY A   2       2.000   0.000   0.000  1.00 20.00           N
ATOM      4  CA  GLY A   2       3.000   0.000   0.000  1.00 20.00           C
ATOM      5  N   SER A   3       4.000   0.000   0.000  1.00 20.00           N
ATOM      6  CA  SER A   3       5.000   0.000   0.000  1.00 20.00           C
HETATM    7  C1  LIG X 101       3.000   2.000   0.000  1.00 20.00           C
END
"""
PROJECT_DIR = Path(__file__).resolve().parents[1]


def _source_complex(runs_dir: Path) -> tuple[JobRecord, ArtifactRef, Path]:
    run_dir = runs_dir / "structure-jobs" / "source-complex"
    run_dir.mkdir(parents=True)
    complex_path = run_dir / "imported_complex_refined.pdb"
    complex_path.write_text(PDB_COMPLEX)
    ligand_path = run_dir / "imported_ligand_refined.sdf"
    ligand_path.write_text(
        "LIG\n  fixture\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    3.0000    2.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n$$$$\n"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "status": "completed",
                "source": "pdb",
                "ligand_key": "LIG|X|101|_",
                "ligand_count": 1,
                "created_at": "2026-07-23T00:00:00+00:00",
            }
        )
    )
    manifest = write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(run_dir, complex_path, "prepared_complex", role="complex"),
            ArtifactRef.from_path(
                run_dir, ligand_path, "prepared_ligand_set", role="ligand"
            ),
        ],
    )
    return (
        JobRecord.load(run_dir, task_group="structure-jobs"),
        manifest.by_type("prepared_complex")[0],
        complex_path,
    )


def test_chain_trimming_changes_only_protein_and_keeps_ligand() -> None:
    assert pdb_chain_ranges(PDB_COMPLEX)[0]["residues"] == (1, 2, 3)

    trimmed, summary = trim_pdb_data(PDB_COMPLEX, {"A": (2, 2)})

    assert "ALA A   1" not in trimmed
    assert "SER A   3" not in trimmed
    assert "GLY A   2" in trimmed
    assert "HETATM    7  C1  LIG X 101" in trimmed
    assert summary["retained_ligand_atom_count"] == 1


def test_trimmed_complex_job_publishes_complex_receptor_and_ligand(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source_job, source_artifact, source_path = _source_complex(runs_dir)

    job = create_trimmed_target_job(
        source_job=source_job,
        source_artifact=source_artifact,
        source_path=source_path,
        ranges={"A": (2, 3)},
    )

    assert job.status == "completed"
    assert job.artifact_manifest is not None
    assert len(job.artifact_manifest.by_type("prepared_complex")) == 1
    assert len(job.artifact_manifest.by_type("prepared_receptor")) == 1
    assert len(job.artifact_manifest.by_type("prepared_ligand_set")) == 1
    complex_path = job.artifact_manifest.by_type("prepared_complex")[0].resolve(
        job.run_dir, must_exist=True
    )
    assert complex_path is not None
    assert "HETATM    7  C1  LIG X 101" in complex_path.read_text()


def test_target_trimming_page_accepts_registered_complex_and_keeps_ligand(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source_job, source_artifact, source_path = _source_complex(runs_dir)
    trimmed = create_trimmed_target_job(
        source_job=source_job,
        source_artifact=source_artifact,
        source_path=source_path,
        ranges={"A": (1, 3)},
    )
    repaired_dir = runs_dir / "terminal-repair" / "repaired-complex"
    repaired_dir.mkdir(parents=True)
    repaired_path = repaired_dir / "complex_repaired.pdb"
    repaired_path.write_text(PDB_COMPLEX)
    (repaired_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": repaired_dir.name,
                "job_type": "terminal_repair",
                "status": "completed",
                "source": "MODELLER",
                "parent_run_id": trimmed.run_id,
                "chain": "A",
                "extension_sequence": "G",
                "created_at": "2026-07-24T12:00:00+00:00",
            }
        )
    )
    write_artifact_manifest(
        repaired_dir,
        [
            ArtifactRef.from_path(
                repaired_dir,
                repaired_path,
                "prepared_complex",
                role="complex",
            )
        ],
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/target_trimming.py"
    ).run(timeout=20)

    assert not page.exception
    target_frame = next(
        frame.value
        for frame in page.dataframe
        if {"Target", "Last step", "Origin"}.issubset(frame.value.columns)
    )
    assert {"PDB", "Target trimming", "MODELLER repair"}.issubset(
        set(target_frame["Last step"])
    )
    show_previous = next(
        item
        for item in page.checkbox
        if item.label == "Show previous target versions"
    )
    show_previous.set_value(True).run(timeout=20)
    assert not page.exception
    target_frame = next(
        frame.value
        for frame in page.dataframe
        if {"Target", "Last step", "Origin"}.issubset(frame.value.columns)
    )
    assert {"PDB", "Target trimming", "MODELLER repair"}.issubset(
        set(target_frame["Last step"])
    )
    assert next(
        button for button in page.button if button.label == "Create trimmed target"
    ).disabled is False
    assert any(
        metric.label == "Retained ligand atoms" and metric.value == "1"
        for metric in page.metric
    )
    preview_mode = next(
        item for item in page.radio if item.label == "Preview structure"
    )
    assert preview_mode.value == "Trimmed complex"
    assert preview_mode.options == [
        "Trimmed complex",
        "Original with retained-region overlay",
    ]
    assert any(
        "Prospective output" in caption.value and "unchanged" in caption.value
        for caption in page.caption
    )
    assert any(
        "Complex preview" in heading.value
        for heading in page.markdown
    )
    preview_mode.set_value("Original with retained-region overlay").run(timeout=20)
    assert not page.exception
    assert any(
        "retained protein and sequence residues are blue" in caption.value
        and "trimmed away are grey" in caption.value
        and "not trimmed" in caption.value
        for caption in page.caption
    )


def test_target_trimming_uses_cartoon_molstar_with_sequence_highlights() -> None:
    source = (
        PROJECT_DIR / "mn_ligand/app/pages/target_trimming.py"
    ).read_text()

    assert "molstar_custom_component(" in source
    assert "representation_type=\"cartoon\"" in source
    assert "highlighted_selections=retained_selections" in source
    assert "show_controls=True" in source
    assert "selection_mode=True" in source
    assert "py3Dmol" not in source


def test_predicted_pdb_complex_promotes_to_typed_downstream_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source_dir = runs_dir / "refolding" / "af3-source"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "candidate.pdb"
    source_path.write_text(PDB_COMPLEX.replace("HETATM", "ATOM  "))
    (source_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": source_dir.name,
                "status": "completed",
                "workflow": "alphafold3_refolding",
                "tool": "AlphaFold 3",
                "created_at": "2026-07-23T00:00:00+00:00",
            }
        )
    )
    manifest = write_artifact_manifest(
        source_dir,
        [ArtifactRef.from_path(source_dir, source_path, "predicted_complex", role="T3")],
    )
    source_job = JobRecord.load(source_dir, task_group="refolding")

    promoted = promote_predicted_complex(
        source_job=source_job,
        source_artifact=manifest.by_type("predicted_complex")[0],
        source_path=source_path,
        clean_and_repair=False,
    )

    assert promoted.status == "completed"
    assert promoted.parent_run_id == source_job.run_id
    assert promoted.artifact_manifest is not None
    assert promoted.artifact_manifest.by_type("prepared_complex")
    assert promoted.artifact_manifest.by_type("prepared_receptor")
    assert promoted.artifact_manifest.by_type("prepared_ligand_set")


def test_predicted_cif_uses_gemmi_for_macromolecular_conversion(
    tmp_path: Path, monkeypatch
) -> None:
    import gemmi

    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source_dir = runs_dir / "refolding" / "af3-cif-source"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "candidate.cif"
    structure = gemmi.read_pdb_string(PDB_COMPLEX)
    structure.setup_entities()
    structure.assign_label_seq_id()
    structure.make_mmcif_document().write_file(str(source_path))
    (source_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": source_dir.name,
                "status": "completed",
                "workflow": "alphafold3_refolding",
                "tool": "AlphaFold 3",
            }
        )
    )
    manifest = write_artifact_manifest(
        source_dir,
        [ArtifactRef.from_path(source_dir, source_path, "predicted_complex", role="LIG")],
    )
    source_job = JobRecord.load(source_dir, task_group="refolding")

    promoted = promote_predicted_complex(
        source_job=source_job,
        source_artifact=manifest.by_type("predicted_complex")[0],
        source_path=source_path,
        clean_and_repair=False,
    )

    assert promoted.status == "completed"
    command = json.loads((promoted.run_dir / "command.json").read_text())
    assert command["mode"] == "python"
    assert command["commands"][0][0:2] == ["gemmi", "convert"]
    assert "obabel" not in json.dumps(command)
    assert promoted.artifact_manifest is not None
    assert promoted.artifact_manifest.by_type("prepared_complex")
    assert promoted.artifact_manifest.by_type("prepared_ligand_set")


def test_predicted_complex_promotion_runs_cleaning_and_keeps_repair_report(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source_dir = runs_dir / "refolding" / "boltz-import-source"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "candidate.pdb"
    source_path.write_text(PDB_COMPLEX.replace("HETATM", "ATOM  "))
    (source_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": source_dir.name,
                "status": "completed",
                "workflow": "boltz2_refolding",
                "tool": "Boltz-2",
            }
        )
    )
    source_manifest = write_artifact_manifest(
        source_dir,
        [ArtifactRef.from_path(source_dir, source_path, "predicted_complex", role="LIG")],
    )
    source_job = JobRecord.load(source_dir, task_group="refolding")

    def fake_cleaning(import_run_id: str):
        clean_dir = runs_dir / "protein-cleaning" / "cleaned-prediction"
        artifact_dir = clean_dir / "artifacts"
        artifact_dir.mkdir(parents=True)
        receptor = artifact_dir / "prepared_target.pdb"
        receptor.write_text(
            "\n".join(
                line for line in PDB_COMPLEX.splitlines() if line.startswith("ATOM  ")
            )
            + "\nEND\n"
        )
        report = artifact_dir / "repair_report.json"
        report.write_text(
            json.dumps(
                {
                    "protein_cleaned": True,
                    "pdbfixer": {"missing_residue_segments": [], "missing_atoms": []},
                }
            )
        )
        (clean_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": clean_dir.name,
                    "status": "completed",
                    "parent_run_id": import_run_id,
                }
            )
        )
        write_artifact_manifest(
            clean_dir,
            [
                ArtifactRef.from_path(
                    clean_dir, receptor, "prepared_target", role="receptor"
                ),
                ArtifactRef.from_path(clean_dir, report, "repair_report", role="report"),
            ],
        )
        return (
            JobRecord.load(clean_dir, task_group="protein-cleaning"),
            {
                "success": True,
                "protein_cleaned": True,
                "prepared_pdb_data": PDB_COMPLEX,
            },
        )

    monkeypatch.setattr(
        "mn_ligand.workflows.predicted_complex_promotion.run_protein_cleaning_job",
        fake_cleaning,
    )
    promoted = promote_predicted_complex(
        source_job=source_job,
        source_artifact=source_manifest.by_type("predicted_complex")[0],
        source_path=source_path,
    )

    assert promoted.status == "completed"
    assert promoted.metadata["clean_and_repair"] is True
    assert promoted.metadata["import_run_id"]
    assert promoted.metadata["cleaning_run_id"] == "cleaned-prediction"
    assert promoted.artifact_manifest is not None
    assert promoted.artifact_manifest.by_type("repair_report")
    promoted_complex = promoted.artifact_manifest.by_type("prepared_complex")[0]
    promoted_path = promoted_complex.resolve(promoted.run_dir, must_exist=True)
    assert promoted_path is not None and "HETATM    7  C1  LIG" in promoted_path.read_text()


def test_predicted_complex_promotion_records_normalized_failure(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source_dir = runs_dir / "refolding" / "invalid-source"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "candidate.pdb"
    source_path.write_text("HETATM    1  C1  LIG X   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n")
    (source_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": source_dir.name,
                "status": "completed",
                "workflow": "alphafold3_refolding",
                "tool": "AlphaFold 3",
            }
        )
    )
    manifest = write_artifact_manifest(
        source_dir,
        [ArtifactRef.from_path(source_dir, source_path, "predicted_complex", role="bad")],
    )
    source_job = JobRecord.load(source_dir, task_group="refolding")

    promoted = promote_predicted_complex(
        source_job=source_job,
        source_artifact=manifest.by_type("predicted_complex")[0],
        source_path=source_path,
    )

    assert promoted.status == "failed"
    assert promoted.result["success"] is False
    assert "protein ATOM records" in promoted.result["error"]
    assert (promoted.run_dir / "input.json").is_file()
