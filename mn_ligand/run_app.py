from __future__ import annotations

from pathlib import Path

import streamlit as st

import mn_ligand
from mn_ligand.runtime import runs_root


def _make_page(
    package_root: Path,
    module_path: str,
    title: str,
    url_slug: str,
    *,
    visibility: str = "visible",
) -> st.Page:
    relative_module = module_path.removeprefix("mn_ligand.").replace(".", "/")
    page_path = package_root / f"{relative_module}.py"
    return st.Page(page=str(page_path), title=title, url_path=url_slug, visibility=visibility)


def main() -> None:
    package_root = Path(mn_ligand.__file__).resolve().parent
    st.set_page_config(
        page_title="mn-ligand",
        page_icon=str(package_root / "app/assets/mn_ligand_tab_icon.png"),
        layout="wide",
    )
    st.sidebar.title("mn-ligand")
    st.sidebar.caption("Docker-backed ligand workflows")

    jobs_page = st.Page(
        page=str(package_root / "app/pages/unified_jobs.py"),
        title="Jobs",
        url_path="jobs",
        default=True,
    )
    jobs_md_page = st.Page(
        page=str(package_root / "app/pages/jobs_md.py"),
        title="MD Production",
        url_path="jobs-md",
        visibility="hidden",
    )
    jobs_md_system_page = st.Page(
        page=str(package_root / "app/pages/jobs_md_system.py"),
        title="MD System Preparation",
        url_path="jobs-md-system-prep",
        visibility="hidden",
    )
    jobs_structure_page = st.Page(
        page=str(package_root / "app/pages/jobs_structure.py"),
        title="Structure",
        url_path="jobs-structure",
        visibility="hidden",
    )
    jobs_openfe_page = st.Page(
        page=str(package_root / "app/pages/jobs_openfe.py"),
        title="OpenFE",
        url_path="jobs-openfe",
        visibility="hidden",
    )
    jobs_admet_page = st.Page(
        page=str(package_root / "app/pages/jobs_admet.py"),
        title="ADMET",
        url_path="jobs-admet",
        visibility="hidden",
    )
    jobs_qc_page = st.Page(
        page=str(package_root / "app/pages/jobs_qc.py"),
        title="Quantum Chemistry",
        url_path="jobs-qc",
        visibility="hidden",
    )
    hidden_results_pages = [
        st.Page(
            page=str(package_root / "app/pages/job_results.py"),
            title="Job Results",
            url_path="job-results",
            visibility="hidden",
        ),
        st.Page(
            page=str(package_root / "app/pages/structure_results.py"),
            title="Structure Results",
            url_path="structure-results",
            visibility="hidden",
        ),
        st.Page(
            page=str(package_root / "app/pages/md_results.py"),
            title="MD Results",
            url_path="md-results",
            visibility="hidden",
        ),
        st.Page(
            page=str(package_root / "app/pages/openfe_results.py"),
            title="OpenFE Results",
            url_path="openfe-results",
            visibility="hidden",
        ),
        st.Page(
            page=str(package_root / "app/pages/admet_results.py"),
            title="ADMET Results",
            url_path="admet-results",
            visibility="hidden",
        ),
        st.Page(
            page=str(package_root / "app/pages/qc_results.py"),
            title="QC Results",
            url_path="qc-results",
            visibility="hidden",
        ),
        st.Page(
            page=str(package_root / "app/pages/campaign_comparison.py"),
            title="Compound Campaign Comparison",
            url_path="compound-campaign-comparison",
            visibility="hidden",
        ),
        st.Page(
            page=str(package_root / "app/pages/molecule_design_results.py"),
            title="Molecule Design Results",
            url_path="molecule-design-results",
            visibility="hidden",
        ),
        st.Page(
            page=str(package_root / "app/pages/target_ensemble_viewer.py"),
            title="Target Ensemble 3D Viewer",
            url_path="target-ensemble-viewer",
            visibility="hidden",
        ),
    ]

    md_simulation_page = _make_page(
        package_root,
        "mn_ligand.app.pages.md_simulation",
        "MD Simulation",
        "workspace-md-simulation",
    )
    structure_preparation_page = _make_page(
        package_root,
        "mn_ligand.app.pages.structure_preparation",
        "Structure Import",
        "workspace-structure-preparation",
    )
    sequence_modification_page = _make_page(
        package_root,
        "mn_ligand.app.pages.sequence_modification",
        "Sequence Modification",
        "prepare-sequence-modification",
    )
    target_trimming_page = _make_page(
        package_root,
        "mn_ligand.app.pages.target_trimming",
        "Target Trimming",
        "prepare-target-trimming",
        visibility="hidden",
    )
    repair_page = _make_page(
        package_root,
        "mn_ligand.app.pages.repair",
        "Repair",
        "prepare-repair",
        visibility="hidden",
    )
    compound_datasets_page = _make_page(
        package_root,
        "mn_ligand.app.pages.compound_datasets",
        "Compound Datasets",
        "prepare-compound-datasets",
    )
    complex_datasets_page = _make_page(
        package_root,
        "mn_ligand.app.pages.complex_datasets",
        "Complex Datasets",
        "prepare-complex-datasets",
    )
    benchmark_datasets_page = _make_page(
        package_root,
        "mn_ligand.app.pages.benchmark_datasets",
        "Benchmark Datasets",
        "prepare-benchmark-datasets",
    )
    settings_page = _make_page(
        package_root,
        "mn_ligand.app.pages.settings",
        "Settings",
        "settings",
    )
    evaluation_pages = [
        _make_page(
            package_root,
            "mn_ligand.app.pages.pose_validation",
            "Pose Validation",
            "evaluate-pose-validation",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.interaction_analysis",
            "Interaction Analysis",
            "evaluate-interaction-analysis",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.abfe",
            "Free Energy",
            "workspace-free-energy",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.admet",
            "ADMET",
            "workspace-admet",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.qc",
            "Quantum Chemistry",
            "workspace-qc",
        ),
    ]
    legacy_task_pages = [
        _make_page(
            package_root,
            "mn_ligand.app.pages.structure_prediction",
            "Structure Prediction",
            "discover-prediction",
            visibility="hidden",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.protein_import",
            "Protein Import",
            "workspace-protein-import",
            visibility="hidden",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.protein_cleaning",
            "Protein Cleaning / Repair",
            "workspace-protein-cleaning",
            visibility="hidden",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.md_system_preparation",
            "MMxPSA System Preparation",
            "workspace-md-system-preparation",
            visibility="hidden",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.md_production",
            "MMxPSA Production",
            "workspace-md-production",
            visibility="hidden",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.redocking_benchmark",
            "Legacy Redocking Benchmark",
            "discover-redocking-benchmark",
            visibility="hidden",
        ),
    ]
    discover_page_specs = (
        ("mn_ligand.app.pages.pocket_detection", "Pocket Detection", "discover-pocket-detection"),
        ("mn_ligand.app.pages.docking_cofolding", "Docking / Cofolding", "discover-docking"),
        (
            "mn_ligand.app.pages.refolding",
            "Redocking / Refolding",
            "discover-refolding",
        ),
        ("mn_ligand.app.pages.rescoring", "Rescoring", "discover-rescoring"),
        ("mn_ligand.app.pages.virtual_screening", "Virtual Screening", "discover-screening"),
    )
    discover_pages = [
        _make_page(package_root, module, title, slug)
        for module, title, slug in discover_page_specs
    ]
    pharmacophore_hypotheses_page = _make_page(
        package_root,
        "mn_ligand.app.pages.pharmacophore_hypotheses",
        "Pharmacophore Hypotheses",
        "generate-pharmacophore-hypotheses",
    )
    generate_pages = [
        _make_page(
            package_root,
            "mn_ligand.app.pages.generative_design",
            "De Novo Molecule Design",
            "generate-molecule-design",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.ligand_redesign",
            "Ligand Redesign",
            "generate-ligand-redesign",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.fragment_growing",
            "Fragment Growing",
            "generate-fragment-growing",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.scaffold_hopping",
            "Scaffold Hopping",
            "generate-scaffold-hopping",
        ),
        _make_page(
            package_root,
            "mn_ligand.app.pages.ligand_optimization",
            "Ligand Optimization",
            "generate-ligand-optimization",
        ),
    ]
    results_explorer_page = _make_page(
        package_root,
        "mn_ligand.app.pages.results_explorer",
        "Results Explorer",
        "results-explorer",
    )
    campaign_results_page = _make_page(
        package_root,
        "mn_ligand.app.pages.campaign_results",
        "Campaign Results",
        "campaign-results",
    )
    benchmark_pages = [
        _make_page(
            package_root,
            "mn_ligand.app.pages.benchmark_campaigns",
            "Redocking / Refolding",
            "benchmark-campaigns",
        ),
    ]
    hidden_benchmark_pages = [
        _make_page(package_root, module, title, slug, visibility="hidden")
        for module, title, slug in (
            ("mn_ligand.app.pages.benchmark_redocking", "Redocking", "benchmark-redocking"),
            ("mn_ligand.app.pages.benchmark_refolding", "Refolding", "benchmark-refolding"),
            ("mn_ligand.app.pages.benchmark_rescoring", "Rescoring", "benchmark-rescoring"),
            (
                "mn_ligand.app.pages.benchmark_dataset_results",
                "Dataset",
                "benchmark-dataset-results",
            ),
            ("mn_ligand.app.pages.benchmark_results", "Results", "benchmark-results"),
        )
    ]
    pg = st.navigation(
        {
            "Jobs": [jobs_page],
            "Results": [results_explorer_page, campaign_results_page],
            "Prepare": [
                structure_preparation_page,
                sequence_modification_page,
                compound_datasets_page,
                complex_datasets_page,
                benchmark_datasets_page,
                pharmacophore_hypotheses_page,
            ],
            "Discover": discover_pages,
            "Generate": generate_pages,
            "Benchmark": benchmark_pages,
            "Simulate": [md_simulation_page],
            "Evaluate": evaluation_pages,
            "System": [settings_page],
            "": [
                jobs_structure_page,
                jobs_md_system_page,
                jobs_md_page,
                jobs_openfe_page,
                jobs_admet_page,
                jobs_qc_page,
                repair_page,
                target_trimming_page,
                *legacy_task_pages,
                *hidden_results_pages,
                *hidden_benchmark_pages,
            ],
        },
        position="sidebar",
    )
    st.sidebar.divider()
    st.sidebar.caption(f"Runs: {runs_root()}")
    pg.run()


if __name__ == "__main__":
    main()
