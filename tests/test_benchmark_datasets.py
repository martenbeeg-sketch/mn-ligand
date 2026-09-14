from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from streamlit.testing.v1 import AppTest

from mn_ligand.workflows.benchmark_datasets import (
    analyze_benchmark_archive,
    benchmark_cases,
    create_benchmark_bound_chain_target,
    create_benchmark_dataset_job,
    load_benchmark_dataset,
)
from mn_ligand.workflows.refolding import ligand_bound_protein_sequence


def _receptor() -> bytes:
    return (
        b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
        b"ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 20.00           C  \n"
        b"ATOM      3  CA  SER A   3       7.600   0.000   0.000  1.00 20.00           C  \n"
        b"END\n"
    )


def _ligand() -> bytes:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=7) == 0
    molecule.SetProp("_Name", "reference")
    return (Chem.MolToMolBlock(molecule) + "\n$$$$\n").encode()


def _zip(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return output.getvalue()


def test_posebench_layout_is_auto_detected() -> None:
    source = _zip(
        {
            "posebusters_benchmark_set/1abc/1abc_protein.pdb": _receptor(),
            "posebusters_benchmark_set/1abc/1abc_ligand.sdf": _ligand(),
        }
    )
    result = analyze_benchmark_archive(source, "posebusters.zip")
    assert result["profile"] == "posebench:posebusters_benchmark"
    assert result["case_count"] == 1
    assert result["cases"][0]["case_id"] == "1abc"
    assert result["cases"][0]["coordinate_dimension"] == 3


def test_generic_manifest_preserves_case_identity_and_split() -> None:
    manifest = (
        b"case_id,target_id,receptor,ligand,split,cohort\n"
        b"case-a,target-a,inputs/protein.pdb,inputs/ligand.sdf,test,kinase\n"
    )
    source = _zip(
        {
            "inputs/protein.pdb": _receptor(),
            "inputs/ligand.sdf": _ligand(),
        }
    )
    result = analyze_benchmark_archive(
        source,
        "generic.zip",
        manifest_data=manifest,
        manifest_filename="benchmark_manifest.csv",
    )
    case = result["cases"][0]
    assert result["profile"] == "generic-manifest"
    assert case["case_id"] == "case-a"
    assert case["target_id"] == "target-a"
    assert case["split"] == "test"
    assert case["metadata"]["cohort"] == "kinase"


def test_benchmark_import_publishes_typed_case_artifacts(tmp_path: Path) -> None:
    source = _zip(
        {
            "astex_diverse_set/1xyz/1xyz_protein.pdb": _receptor(),
            "astex_diverse_set/1xyz/1xyz_ligand.sdf": _ligand(),
        }
    )
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        job = create_benchmark_dataset_job(
            dataset_name="Astex smoke",
            archive_data=source,
            archive_filename="astex.zip",
            source_provenance={
                "url": "https://example.org/astex",
                "citation": "Example benchmark v1",
            },
        )
    dataset = load_benchmark_dataset(job)
    cases = benchmark_cases(job)
    assert job.status == "completed"
    assert dataset["dataset_name"] == "Astex smoke"
    assert dataset["case_count"] == 1
    assert cases[0].case_id == "1xyz"
    assert cases[0].receptor_path.is_file()
    assert cases[0].ligand_path.is_file()
    assert job.artifact_manifest is not None
    assert len(job.artifact_manifest.by_type("benchmark_receptor")) == 1
    assert len(job.artifact_manifest.by_type("benchmark_reference_ligand")) == 1
    assert len(job.artifact_manifest.by_type("benchmark_dataset")) == 1
    assert len(job.artifact_manifest.by_type("benchmark_source_provenance")) == 1
    assert dataset["source_provenance"]["citation"] == "Example benchmark v1"
    assert all(not Path(item.path).is_absolute() for item in job.artifact_manifest.artifacts)


def test_benchmark_bound_chain_target_is_typed_and_single_chain(tmp_path: Path) -> None:
    receptor = (
        b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
        b"ATOM      2  CA  GLY B   1      30.000   0.000   0.000  1.00 20.00           C  \n"
        b"END\n"
    )
    source = _zip(
        {
            "posebusters_benchmark_set/1abc/1abc_protein.pdb": receptor,
            "posebusters_benchmark_set/1abc/1abc_ligand.sdf": _ligand(),
        }
    )
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        dataset_job = create_benchmark_dataset_job(
            dataset_name="Bound-chain smoke",
            archive_data=source,
            archive_filename="posebusters.zip",
        )
        case = benchmark_cases(dataset_job)[0]
        selection = ligand_bound_protein_sequence(case.receptor_path, case.ligand_path)
        target_job = create_benchmark_bound_chain_target(
            case,
            selection,
            campaign_id="campaign-test",
        )
    artifact = target_job.artifact_manifest.by_type("benchmark_bound_receptor")[0]
    receptor_text = artifact.resolve(target_job.run_dir, must_exist=True).read_text()
    assert selection["chain"] == "A"
    assert " A   1" in receptor_text
    assert " B   1" not in receptor_text
    assert target_job.metadata["benchmark_dataset_run_id"] == dataset_job.run_id
    assert target_job.metadata["benchmark_campaign_id"] == "campaign-test"


def test_archive_path_traversal_is_rejected() -> None:
    source = _zip(
        {
            "../protein.pdb": _receptor(),
            "ligand.sdf": _ligand(),
        }
    )
    with pytest.raises(ValueError, match="Unsafe archive member"):
        analyze_benchmark_archive(source, "unsafe.zip")


def test_tar_posebench_layout_is_supported() -> None:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, data in {
            "dockgen_set/abcd/abcd_processed.pdb": _receptor(),
            "dockgen_set/abcd/abcd_ligand.sdf": _ligand(),
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    result = analyze_benchmark_archive(output.getvalue(), "dockgen.tar.gz")
    assert result["profile"] == "posebench:dockgen"
    assert result["case_count"] == 1


def test_benchmark_pages_render_without_imported_datasets(tmp_path: Path) -> None:
    page_root = Path(__file__).parents[1] / "mn_ligand" / "app" / "pages"
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        for filename in (
            "benchmark_datasets.py",
            "benchmark_redocking.py",
            "benchmark_refolding.py",
            "benchmark_rescoring.py",
            "benchmark_results.py",
            "benchmark_dataset_results.py",
            "benchmark_campaigns.py",
        ):
            app = AppTest.from_file(str(page_root / filename)).run(timeout=20)
            assert not app.exception, filename


def test_benchmark_prepare_and_combined_result_navigation_are_separate() -> None:
    page_root = Path(__file__).parents[1] / "mn_ligand" / "app" / "pages"
    prepare = (page_root / "benchmark_datasets.py").read_text()
    explorer = (page_root / "benchmark_dataset_results.py").read_text()
    combined = (page_root / "benchmark_results.py").read_text()
    assert '["Input", "Format", "Validation", "Run", "Results"]' in prepare
    assert "Import or register another dataset" not in prepare
    assert '"Combined results"' in prepare
    assert "benchmark-dataset-results" in prepare
    assert '["Overview", "Cases / Viewer", "Artifacts", "Lineage"]' in explorer
    assert 'selection_mode="single-row-required"' in explorer
    assert '["Overview", "Artifacts", "Metrics", "Viewer", "Lineage", "Logs"]' in combined
    campaigns = (page_root / "benchmark_campaigns.py").read_text()
    assert "create_benchmark_bound_chain_target" in campaigns
    assert '"benchmark_bound_receptor"' in campaigns


def test_benchmark_launchers_cover_installed_compatible_engines() -> None:
    page_root = Path(__file__).parents[1] / "mn_ligand" / "app" / "pages"
    redocking = (page_root / "benchmark_redocking.py").read_text()
    refolding = (page_root / "benchmark_refolding.py").read_text()
    rescoring = (page_root / "benchmark_rescoring.py").read_text()
    assert all(
        engine in redocking
        for engine in ("AutoDock Vina", "GNINA", "Uni-Dock Pro", "RosettaLigand")
    )
    assert all(engine in refolding for engine in ("Boltz-2", "AlphaFold 3", "Nesso-1"))
    assert all(engine in rescoring for engine in ("GNINA score-only", "Boltzina"))
    assert "ligand RMSD" in (
        page_root / "benchmark_results.py"
    ).read_text()


def test_navigation_exposes_prepare_and_benchmark_groups() -> None:
    source = (
        Path(__file__).parents[1] / "mn_ligand" / "run_app.py"
    ).read_text()
    assert '"Benchmark Datasets"' in source
    assert '"Benchmark": benchmark_pages' in source
    assert '"Redocking / Refolding"' in source
    assert '"benchmark-campaigns"' in source
