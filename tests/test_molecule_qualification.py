from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pandas as pd
from rdkit import Chem

from mn_ligand.app.pages.discover_inputs import artifact_options
from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JobRecord
from mn_ligand.workflows.molecule_qualification import (
    finalize_molecule_qualification_job,
    queue_molecule_qualification_job,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _generation_job(runs_dir: Path) -> JobRecord:
    run_dir = runs_dir / "molecule-generation" / "generation-source-1"
    normalized = run_dir / "normalized"
    normalized.mkdir(parents=True)
    molecule = Chem.MolFromSmiles("CCO")
    molecule.SetProp("_Name", "source-1")
    molecule.SetProp("compound_id", "source-1")
    molecule.SetProp("canonical_isomeric_smiles", "CCO")
    writer = Chem.SDWriter(str(normalized / "generated_compounds.sdf"))
    writer.write(molecule)
    writer.close()
    (normalized / "generated_compounds.csv").write_text(
        "compound_id,canonical_isomeric_smiles,generation_engine,"
        "native_source,native_index,coordinate_dimension\n"
        "source-1,CCO,fixture,generated.sdf,0,2\n"
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
                "engine_id": "fixture",
                "seed": 77,
            }
        )
    )
    return JobRecord.load(run_dir, task_group="molecule-generation")


def test_qualification_job_scales_cpu_parallelism_and_is_smiles_first(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source = _generation_job(runs_dir)

    job = queue_molecule_qualification_job(source)

    assert job.task_group == "molecule-qualification"
    assert job.parent_run_id == source.run_id
    assert job.metadata["resources"]["gpu"] is False
    assert job.metadata["resources"]["cpu_threads"] == 1
    assert job.metadata["qualification_policy"]["max_workers"] == 16
    assert job.metadata["effective_max_workers"] == 1
    command = job.metadata["queued_command"]
    assert command[command.index("--mode") + 1] == "qualify-molecules"
    assert command[command.index("--max-workers") + 1] == "1"
    payload = json.loads((job.run_dir / "input.json").read_text())
    assert payload["identity_source"] == (
        "canonical stereochemistry-aware SMILES"
    )
    assert payload["geometry_source"] == "deterministic RDKit ETKDGv3"
    assert (
        queue_molecule_qualification_job(source).run_id
        == job.run_id
    )


def test_qualification_finalizer_publishes_only_qualified_compound_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source = _generation_job(runs_dir)
    job = queue_molecule_qualification_job(source)
    qualified = job.run_dir / "qualified"
    molecule = Chem.MolFromSmiles("CCO")
    molecule.SetProp("_Name", "source-1")
    writer = Chem.SDWriter(str(qualified / "qualified_compounds.sdf"))
    writer.write(molecule)
    writer.close()
    (qualified / "qualification.csv").write_text(
        "compound_id,chemical_pass,conformer_generation_pass,"
        "posebusters_pass,qualified_for_docking\n"
        "source-1,True,True,True,True\n"
    )
    (qualified / "posebusters_full.csv").write_text(
        "file,sanitization,bond_lengths\nsource-1,True,True\n"
    )
    (qualified / "qualification_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "input_count": 1,
                "chemical_pass_count": 1,
                "conformer_pass_count": 1,
                "posebusters_pass_count": 1,
                "qualified_compound_count": 1,
            }
        )
    )

    completed = finalize_molecule_qualification_job(
        job.run_dir,
        returncode=0,
    )

    assert completed.status == "completed"
    assert completed.artifact_manifest is not None
    handoff = completed.artifact_manifest.by_type("compound_set")
    assert len(handoff) == 1
    assert handoff[0].path == "qualified/qualified_compounds.sdf"
    assert handoff[0].metadata["qualification_required"] is True


def test_artifact_discovery_hides_legacy_generation_handoff_after_qualification(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source = _generation_job(runs_dir)
    legacy_sdf = source.run_dir / "normalized" / "generated_compounds.sdf"
    write_artifact_manifest(
        source.run_dir,
        [
            ArtifactRef.from_path(
                source.run_dir,
                legacy_sdf,
                "compound_set",
                role="legacy_prequalification_handoff",
            )
        ],
    )

    queue_molecule_qualification_job(source)

    options = artifact_options(("compound_set",))
    assert all(choice.job.run_id != source.run_id for choice in options.values())


def test_chemical_gate_rejects_pfm_cage_and_peroxide_and_builds_3d() -> None:
    source = PROJECT_DIR / "tools_to_implement" / "posebusters"
    sys.path.insert(0, str(source))
    try:
        qualification = importlib.import_module("mn_ligand_qualification")
        cage = (
            "CO[C@H]1CC[C@@H]2[C@H]3[C@@H]1C[C@@]3(C)C1(C)"
            "C[C@H]3C[C@@]4(CC[C@@H](C)C4)C[C@H](C)[C@@]21C3"
        )
        peroxide = "CCC([C@H]1C[C@@H]2COO[C@H]2C1)[C@H](O)C=C=O"
        _, cage_metrics, cage_failures = qualification._chemical_assessment(
            cage,
            min_heavy_atoms=5,
            max_heavy_atoms=80,
            max_absolute_charge=2,
            max_sa_score=6.0,
        )
        _, peroxide_metrics, peroxide_failures = (
            qualification._chemical_assessment(
                peroxide,
                min_heavy_atoms=5,
                max_heavy_atoms=80,
                max_absolute_charge=2,
                max_sa_score=6.0,
            )
        )
        molecule, _, ethanol_failures = qualification._chemical_assessment(
            "CCO",
            min_heavy_atoms=2,
            max_heavy_atoms=80,
            max_absolute_charge=2,
            max_sa_score=6.0,
        )
        conformer, geometry, error = qualification._standardized_conformer(
            molecule,
            seed=7,
            conformer_count=3,
        )
    finally:
        sys.path.remove(str(source))

    assert cage_metrics["sa_score"] > 6.0
    assert any("synthetic accessibility" in item for item in cage_failures)
    assert "peroxide" in peroxide_metrics["reactive_alerts"]
    assert any("peroxide" in item for item in peroxide_failures)
    assert ethanol_failures == []
    assert error == ""
    assert conformer is not None
    assert conformer.GetConformer().Is3D()
    assert geometry["geometry_source"] == "rdkit_etkdgv3"


def test_isolated_nonaromatic_ring_flatness_is_reviewable(
    tmp_path: Path,
) -> None:
    source = PROJECT_DIR / "tools_to_implement" / "posebusters"
    sys.path.insert(0, str(source))
    try:
        qualification = importlib.import_module("mn_ligand_qualification")
        smiles = (
            "CO[C@@]1(C(F)(F)F)C=CCC("
            "c2cc(Nc3cccc(NC(C)=O)c3)c(F)cc2O)=C1"
        )
        molecule = Chem.MolFromSmiles(smiles)
        molecule.SetProp("_Name", "flat-ring-review")
        molecule.SetProp("compound_id", "flat-ring-review")
        molecule.SetProp("canonical_isomeric_smiles", smiles)
        input_sdf = tmp_path / "generated.sdf"
        writer = Chem.SDWriter(str(input_sdf))
        writer.write(molecule)
        writer.close()
        input_table = tmp_path / "generated.csv"
        input_table.write_text(
            "compound_id,canonical_isomeric_smiles,generation_engine\n"
            f"flat-ring-review,{smiles},paopt\n"
        )
        report = qualification.qualify_molecules(
            input_sdf,
            input_table,
            tmp_path / "qualified",
            seed=20260727,
            conformer_count=20,
            max_workers=1,
            min_heavy_atoms=5,
            max_heavy_atoms=80,
            max_absolute_charge=2,
            max_sa_score=6.0,
        )
    finally:
        sys.path.remove(str(source))

    table = pd.read_csv(
        tmp_path / "qualified" / "qualification.csv"
    ).fillna("")
    row = table.iloc[0]
    assert row["posebusters_pass"] in (False, 0)
    assert row["posebusters_hard_pass"] in (True, 1)
    assert row["review_warnings"] == "non-aromatic_ring_non-flatness"
    assert row["qualification_status"] == "qualified_with_warning"
    assert row["qualified_for_docking"] in (True, 1)
    assert report["qualified_with_warning_count"] == 1
    assert report["qualified_compound_count"] == 1
