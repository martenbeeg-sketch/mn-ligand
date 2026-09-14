from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JobRecord
from mn_ligand.workflows import terminal_repair
from mn_ligand.workflows.terminal_repair import (
    create_terminal_repair_job,
    merge_extension,
    normalize_extension_sequence,
    pdb_chain_sequences,
    pdb_seqres_sequences,
    infer_terminal_sequence,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
PDB_COMPLEX = """\
SEQRES   1 A    5  ALA GLY SER GLY GLY
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.200   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.400   0.000   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       3.000   1.000   0.000  1.00 20.00           O
ATOM      5  N   GLY A   2       3.730   0.000   0.000  1.00 20.00           N
ATOM      6  CA  GLY A   2       4.930   0.000   0.000  1.00 20.00           C
ATOM      7  C   GLY A   2       6.130   0.000   0.000  1.00 20.00           C
ATOM      8  O   GLY A   2       6.730   1.000   0.000  1.00 20.00           O
HETATM    9  C1  LIG X 101       4.000   4.000   0.000  1.00 20.00           C
END
"""
MODEL = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.200   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.400   0.000   0.000  1.00 20.00           C
ATOM      4  N   GLY A   2       3.730   0.000   0.000  1.00 20.00           N
ATOM      5  CA  GLY A   2       4.930   0.000   0.000  1.00 20.00           C
ATOM      6  C   GLY A   2       6.130   0.000   0.000  1.00 20.00           C
ATOM      7  N   SER A   3       7.460   0.000   0.000  1.00 20.00           N
ATOM      8  CA  SER A   3       8.660   0.000   0.000  1.00 20.00           C
ATOM      9  C   SER A   3       9.860   0.000   0.000  1.00 20.00           C
ATOM     10  O   SER A   3      10.460   1.000   0.000  1.00 20.00           O
END
"""


def _source(runs_dir: Path) -> tuple[JobRecord, ArtifactRef, Path]:
    run_dir = runs_dir / "structure-jobs" / "source"
    run_dir.mkdir(parents=True)
    complex_path = run_dir / "complex.pdb"
    complex_path.write_text(PDB_COMPLEX)
    ligand_path = run_dir / "ligand.sdf"
    ligand_path.write_text("ligand\n")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "source",
                "status": "completed",
                "source": "pdb",
                "pdb_id": "1ABC",
                "created_at": "2026-07-24T00:00:00+00:00",
            }
        )
    )
    manifest = write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(run_dir, complex_path, "prepared_complex"),
            ArtifactRef.from_path(run_dir, ligand_path, "prepared_ligand_set"),
        ],
    )
    return (
        JobRecord.load(run_dir, task_group="structure-jobs"),
        manifest.by_type("prepared_complex")[0],
        complex_path,
    )


def test_sequence_validation_and_chain_inventory() -> None:
    assert normalize_extension_sequence(" gs g\n") == "GSG"
    assert pdb_chain_sequences(PDB_COMPLEX) == (
        {
            "chain": "A",
            "start": 1,
            "end": 2,
            "sequence": "AG",
            "residue_count": 2,
            "gaps": (),
            "has_insertions": False,
        },
    )


def test_seqres_comparison_proposes_missing_c_terminal_sequence() -> None:
    declared = """\
SEQRES   1 A    7  ALA GLY CAS SER GLU VAL GLY
"""
    reference = pdb_seqres_sequences(declared)["A"]
    evidence = infer_terminal_sequence(reference, "AGCS")

    assert reference == "AGCSEVG"
    assert evidence["matched"] is True
    assert evidence["c_terminal_sequence"] == "EVG"
    assert evidence["n_terminal_sequence"] == ""


def test_merge_extension_keeps_source_ligand_and_reports_junction() -> None:
    merged, validation = merge_extension(
        PDB_COMPLEX, MODEL, chain="A", original_end=2, extension_length=1
    )

    assert "SER A   3" in merged
    assert "HETATM    9  C1  LIG X 101" in merged
    assert validation["retained_ligand_atom_count"] == 1
    assert 1.2 < validation["junction_cn_angstrom"] < 1.5


def test_merge_extension_restores_author_numbering_after_modeller_renumbering() -> None:
    original = (
        PDB_COMPLEX
        .replace(" A   1", " A 261")
        .replace(" A   2", " A 262")
    )
    merged, validation = merge_extension(
        original,
        MODEL,
        chain="A",
        original_end=262,
        extension_length=1,
    )

    assert "SER A 263" in merged
    assert validation["junction_cn_angstrom"] is not None


def test_repair_job_publishes_typed_complex_ensemble_and_report(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source_job, source_artifact, source_path = _source(runs_dir)

    def fake_run(command, *, cwd, capture_output, text, check):
        output_path = Path(cwd) / command[command.index("--output") + 1]
        model_path = Path(cwd) / "extended.B99990001.pdb"
        model_path.write_text(MODEL)
        output_path.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "name": model_path.name,
                            "failure": "",
                            "dope": -123.4,
                            "ga341": [1.0, 0.0, 0.0],
                        }
                    ]
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout="native output", stderr="")

    monkeypatch.setattr(terminal_repair.subprocess, "run", fake_run)
    job = create_terminal_repair_job(
        source_job=source_job,
        source_artifact=source_artifact,
        source_path=source_path,
        chain="A",
        extension_sequence="S",
        model_count=1,
    )

    assert job.status == "completed"
    assert job.artifact_manifest is not None
    assert len(job.artifact_manifest.by_type("prepared_complex")) == 1
    assert len(job.artifact_manifest.by_type("prepared_receptor")) == 1
    assert len(job.artifact_manifest.by_type("prepared_ligand_set")) == 1
    assert len(job.artifact_manifest.by_type("structure_model")) == 1
    assert len(job.artifact_manifest.by_type("repair_report")) == 1
    assert job.result["extension_sequence"] == "S"
    assert job.result["best_model"]["junction_cn_angstrom"] is not None


def test_repair_job_records_native_launch_failure(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source_job, source_artifact, source_path = _source(runs_dir)

    def failed_run(*args, **kwargs):
        raise FileNotFoundError("MODELLER interpreter disappeared")

    monkeypatch.setattr(terminal_repair.subprocess, "run", failed_run)
    job = create_terminal_repair_job(
        source_job=source_job,
        source_artifact=source_artifact,
        source_path=source_path,
        chain="A",
        extension_sequence="S",
        model_count=1,
    )

    assert job.status == "failed"
    assert "interpreter disappeared" in job.result["error"]
    assert job.artifact_manifest is not None
    assert job.artifact_manifest.artifacts == ()
    assert "interpreter disappeared" in (job.run_dir / "stderr.log").read_text()


def test_repair_page_has_shared_stages_and_launch_only_in_run_tab(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    _source(runs_dir)
    monkeypatch.setattr(
        terminal_repair,
        "modeller_readiness",
        lambda: (True, "MODELLER 10.8 · test interpreter"),
    )

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/repair.py"
    ).run(timeout=30)

    assert not page.exception
    assert [tab.label for tab in page.tabs] == ["Target", "Repair", "Run", "Results"]
    assert next(
        button for button in page.button if button.label == "Run C-terminal repair"
    ).disabled is False
    sequence_input = next(
        item for item in page.text_input
        if item.label == "C-terminal sequence to append"
    )
    assert sequence_input.value == "SGG"
    assert any(
        "probable missing C-terminal residue(s) (`SGG`)" in item.value
        for item in page.info
    )
    assert any(item.label == "Independent terminal conformations" for item in page.number_input)
    sequence_input.set_value("AAA").run(timeout=30)
    assert not page.exception
    assert sequence_input.value == "AAA"
    assert any(
        "user-edited extension" in item.value for item in page.caption
    )


def test_repair_page_uses_sequence_aware_cartoon_molstar() -> None:
    source = (PROJECT_DIR / "mn_ligand/app/pages/repair.py").read_text()

    assert "molstar_custom_component(" in source
    assert 'representation_type="cartoon"' in source
    assert "show_controls=True" in source
    assert "selection_mode=True" in source
    assert "PDB SEQRES" not in source or "pdb_seqres_sequences" in source
