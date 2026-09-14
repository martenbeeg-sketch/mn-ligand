from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from streamlit.testing.v1 import AppTest

from mn_ligand.app.pages.docking_cofolding import (
    _imported_compound_options,
)
from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JobRecord
from mn_ligand.workflows.molecule_design_results import (
    create_molecule_design_selection,
    target_design_campaigns,
    target_design_compounds,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _write_job(
    run_dir: Path,
    *,
    workflow: str,
    status: str = "completed",
    parent_run_id: str = "",
    **metadata: object,
) -> JobRecord:
    run_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "job_code": run_dir.name[:5].upper(),
        "workflow": workflow,
        "status": status,
        "parent_run_id": parent_run_id,
        "created_at": metadata.pop("created_at", "2026-07-28T08:00:00+00:00"),
        **metadata,
    }
    (run_dir / "metadata.json").write_text(json.dumps(payload))
    (run_dir / "input.json").write_text("{}")
    (run_dir / "result.json").write_text("{}")
    write_artifact_manifest(run_dir, [])
    return JobRecord.load(run_dir, task_group=run_dir.parent.name)


def _fixture_jobs(runs_dir: Path) -> tuple[str, list[JobRecord]]:
    target = _write_job(
        runs_dir / "structure-jobs" / "target-1",
        workflow="structure_preparation",
        pdb_id="4LNW",
        tool="OpenMM minimization",
    )
    campaign = _write_job(
        runs_dir / "generation-campaigns" / "campaign-1",
        workflow="molecule_generation_campaign",
        parent_run_id=target.run_id,
        target_run_id=target.run_id,
        name="4LNW design",
    )
    generation = _write_job(
        runs_dir / "molecule-generation" / "generation-1",
        workflow="molecule_generation",
        parent_run_id=campaign.run_id,
        engine_id="paopt",
        tool="paOPT",
    )
    qualification = _write_job(
        runs_dir / "molecule-qualification" / "qualification-1",
        workflow="molecule_qualification",
        parent_run_id=generation.run_id,
        qualification_policy_version=2,
        source_engine="paOPT",
        source_engine_id="paopt",
        created_at="2026-07-28T09:00:00+00:00",
    )
    candidate_dir = qualification.run_dir / "qualified" / "candidates"
    candidate_dir.mkdir(parents=True)
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(molecule, randomSeed=7)
    molecule = Chem.RemoveHs(molecule)
    molecule.SetProp("_Name", "paopt-1")
    molecule.SetProp("compound_id", "paopt-1")
    molecule.SetProp("canonical_isomeric_smiles", "CCO")
    writer = Chem.SDWriter(str(candidate_dir / "paopt-1.sdf"))
    writer.write(molecule)
    writer.close()
    qualified_dir = qualification.run_dir / "qualified"
    (qualified_dir / "qualification.csv").write_text(
        "compound_id,canonical_isomeric_smiles,qualified_for_docking,"
        "posebusters_pass,qualification_status,review_warnings,qed,"
        "molecular_weight,sa_score,logp,hbond_donors,hbond_acceptors,"
        "rotatable_bonds,ring_count,heavy_atom_count,formal_charge,"
        "force_field,selected_energy\n"
        "paopt-1,CCO,True,False,qualified_with_warning,"
        "non-aromatic_ring_non-flatness,0.407,46.069,1.98,-0.001,1,1,"
        "0,0,3,0,MMFF94s,-1.3\n"
    )
    return target.run_id, [target, campaign, generation, qualification]


def test_target_design_results_join_latest_qualified_compounds(
    tmp_path: Path,
) -> None:
    target_run_id, jobs = _fixture_jobs(tmp_path / "runs")

    campaigns = target_design_campaigns(target_run_id, jobs=jobs)
    rows = target_design_compounds(target_run_id, jobs=jobs)

    assert [job.run_id for job in campaigns] == ["campaign-1"]
    assert len(rows) == 1
    assert rows[0]["engine"] == "paOPT"
    assert rows[0]["qualification_status"] == "qualified_with_warning"
    assert rows[0]["accepted_for_docking"] is True
    assert rows[0]["candidate_available"] is True


def test_design_selection_publishes_3d_compound_set_and_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    target_run_id, jobs = _fixture_jobs(runs_dir)
    rows = target_design_compounds(target_run_id, jobs=jobs)

    selection = create_molecule_design_selection(
        target_run_id=target_run_id,
        selected_rows=rows,
        name="QED and MW filtered",
    )

    assert selection.status == "completed"
    assert selection.result["compound_count"] == 1
    assert selection.artifact_manifest is not None
    compound_sets = selection.artifact_manifest.by_type("compound_set")
    assert len(compound_sets) == 1
    assert compound_sets[0].role == "filtered_molecule_design_handoff"
    selected = [
        molecule
        for molecule in Chem.SDMolSupplier(
            str(selection.run_dir / compound_sets[0].path),
            removeHs=False,
        )
        if molecule is not None
    ]
    assert len(selected) == 1
    assert selected[0].GetConformer().Is3D()
    assert selected[0].GetProp("source_compound_id") == "paopt-1"
    docking_options = _imported_compound_options()
    assert any(
        choice.job.run_id == selection.run_id
        for choice in docking_options.values()
    )


def test_target_design_summary_page_filters_and_previews_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    target_run_id, _jobs = _fixture_jobs(runs_dir)

    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/molecule_design_results.py"
    )
    page.query_params["target_run_id"] = target_run_id
    page.run(timeout=20)

    assert not page.exception
    metrics = {item.label: item.value for item in page.metric}
    assert metrics["Prepared target"] == "TARGE"
    assert metrics["Design campaigns"] == "1"
    assert metrics["Generation engines"] == "1"
    assert metrics["Generated records"] == "1"
    assert metrics["Accepted"] == "1"
    assert metrics["Accepted with warning"] == "1"
    preview = next(
        item for item in page.selectbox if item.label == "Preview compound"
    )
    assert preview.options == [
        "paopt-1 | paOPT | qualified with warning"
    ]
    assert any(
        item.label == "Create dataset for Docking / Cofolding"
        and item.disabled
        for item in page.button
    )
