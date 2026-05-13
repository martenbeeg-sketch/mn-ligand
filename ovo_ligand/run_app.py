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
        title="Jobs – MD Production",
        url_path="jobs-md",
        default=True,
    )
    jobs_md_system_page = st.Page(
        page=str(package_root / "app/pages/jobs_md_system.py"),
        title="Jobs – MMxPSA System Prep",
        url_path="jobs-md-system-prep",
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
    jobs_admet_page = st.Page(
        page=str(package_root / "app/pages/jobs_admet.py"),
        title="Jobs – ADMET",
        url_path="jobs-admet",
    )
    jobs_qc_page = st.Page(
        page=str(package_root / "app/pages/jobs_qc.py"),
        title="Jobs – QC",
        url_path="jobs-qc",
    )
    hidden_results_pages = [
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
            "ovo_ligand.app.pages.md_system_preparation",
            "MMxPSA System Preparation",
            "workspace-md-system-preparation",
        ),
        _make_page(
            package_root,
            "ovo_ligand.app.pages.md_production",
            "MMxPSA Production",
            "workspace-md-production",
        ),
        _make_page(
            package_root,
            "ovo_ligand.app.pages.abfe",
            "Free Energy (ABFE/RBFE)",
            "workspace-free-energy",
        ),
        _make_page(
            package_root,
            "ovo_ligand.app.pages.admet",
            "Ligand Properties (ADMET)",
            "workspace-admet",
        ),
        _make_page(
            package_root,
            "ovo_ligand.app.pages.qc",
            "Quantum Chemistry (QC)",
            "workspace-qc",
        ),
    ]
    pg = st.navigation(
        {
            "Jobs": [jobs_structure_page, jobs_md_system_page, jobs_md_page, jobs_openfe_page, jobs_admet_page, jobs_qc_page],
            "Task": task_pages,
            "": hidden_results_pages,
        },
        position="sidebar",
    )
    st.sidebar.divider()
    st.sidebar.caption(f"Runs: {os.getenv('OVO_LIGAND_RUN_DIR', '/tmp/ovo-ligand-runs')}")
    pg.run()


if __name__ == "__main__":
    main()
