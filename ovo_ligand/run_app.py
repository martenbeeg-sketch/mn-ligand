from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

import ovo_ligand
from ovo_ligand import plugin


def _workflow_pages():
    pages = []
    package_root = Path(ovo_ligand.__file__).resolve().parent
    workflows = plugin["extension_points"]["ovo.workflow_page"]
    workflows = sorted(workflows, key=lambda item: item["title"] != "Bound ligand MD")
    for workflow in workflows:
        module_name = workflow["path"]
        relative_module = module_name.removeprefix("ovo_ligand.").replace(".", "/")
        page_path = package_root / f"{relative_module}.py"
        pages.append(
            st.Page(
                page=str(page_path),
                title=workflow["title"],
                url_path=module_name.split(".")[-1],
            )
        )
    return pages


def _make_page(package_root: Path, module_path: str, title: str, url_slug: str) -> st.Page:
    relative_module = module_path.removeprefix("ovo_ligand.").replace(".", "/")
    page_path = package_root / f"{relative_module}.py"
    return st.Page(page=str(page_path), title=title, url_path=url_slug)


def main() -> None:
    st.set_page_config(page_title="ovo-ligand", layout="wide")
    st.sidebar.title("ovo-ligand")
    st.sidebar.caption("Docker-backed ligand workflows")

    package_root = Path(ovo_ligand.__file__).resolve().parent

    jobs_md_page = st.Page(
        page=str(package_root / "app/pages/jobs_md.py"),
        title="Jobs – MD",
        url_path="jobs-md",
        default=True,
    )
    jobs_md_legacy_page = st.Page(
        page=str(package_root / "app/pages/legacy_jobs_redirect.py"),
        title="Jobs (Legacy)",
        url_path="jobs",
        visibility="hidden",
    )
    jobs_structure_page = st.Page(
        page=str(package_root / "app/pages/jobs_structure.py"),
        title="Jobs – Structure",
        url_path="jobs-structure",
    )
    jobs_openfe_page = st.Page(
        page=str(package_root / "app/pages/jobs_openfe.py"),
        title="Jobs – OpenFE",
        url_path="jobs-openfe",
    )
    md_results_page = st.Page(
        page=str(package_root / "app/pages/md_results.py"),
        title="MD Results",
        url_path="md-results",
        visibility="hidden",
    )
    structure_results_page = st.Page(
        page=str(package_root / "app/pages/structure_results.py"),
        title="Structure Results",
        url_path="structure-results",
        visibility="hidden",
    )

    pipeline_pages = [
        _make_page(
            package_root,
            "ovo_ligand.app.pages.bound_ligand_md",
            "PDB -> Prepare -> MD -> Analysis",
            "pipeline-pdb-prepare-md-analysis",
        ),
        _make_page(
            package_root,
            "ovo_ligand.app.pages.docking",
            "PDB -> Prepare -> Docking -> MD -> Analysis",
            "pipeline-pdb-prepare-docking-md-analysis",
        ),
        _make_page(
            package_root,
            "ovo_ligand.app.pages.abfe",
            "Prepare -> ABFE",
            "pipeline-prepare-abfe",
        ),
        _make_page(
            package_root,
            "ovo_ligand.app.pages.rbfe",
            "Prepare series -> RBFE",
            "pipeline-prepare-series-rbfe",
        ),
    ]

    task_pages = [
        _make_page(
            package_root,
            "ovo_ligand.app.pages.structure_preparation",
            "Structure Preparation",
            "workspace-structure-preparation",
        ),
        _make_page(
            package_root,
            "ovo_ligand.app.pages.docking",
            "Docking",
            "workspace-docking",
        ),
        _make_page(
            package_root,
            "ovo_ligand.app.pages.md",
            "MD Run",
            "workspace-md-run",
        ),
        _make_page(
            package_root,
            "ovo_ligand.app.pages.bound_ligand_md",
            "Analysis (RMSD, energies, MM/GBSA, MM/PBSA)",
            "workspace-analysis",
        ),
        _make_page(
            package_root,
            "ovo_ligand.app.pages.admet",
            "Ligand Properties (ADMET/QC)",
            "workspace-ligand-properties",
        ),
        _make_page(
            package_root,
            "ovo_ligand.app.pages.abfe",
            "Free Energy (ABFE/RBFE)",
            "workspace-free-energy",
        ),
    ]
    structure_preparation_legacy_pages = [
        st.Page(
            page=str(package_root / "app/pages/legacy_structure_preparation_redirect.py"),
            title="Structure Preparation (Legacy)",
            url_path="structure-preparation",
            visibility="hidden",
        ),
        st.Page(
            page=str(package_root / "app/pages/legacy_structure_preparation_redirect.py"),
            title="Structure Preparation (Legacy 2)",
            url_path="structure_preparation",
            visibility="hidden",
        ),
    ]

    pg = st.navigation(
        {
            "Jobs": [jobs_md_page, jobs_structure_page, jobs_openfe_page],
            "Pipelines": pipeline_pages,
            "Task": task_pages,
            "Internal": [md_results_page, structure_results_page, jobs_md_legacy_page, *structure_preparation_legacy_pages],
        },
        position="sidebar",
    )
    st.sidebar.divider()
    st.sidebar.caption(f"Runs: {os.getenv('OVO_LIGAND_RUN_DIR', '/tmp/ovo-ligand-runs')}")
    pg.run()


if __name__ == "__main__":
    main()
