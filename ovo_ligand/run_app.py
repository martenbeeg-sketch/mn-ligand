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


def main() -> None:
    st.set_page_config(page_title="ovo-ligand", layout="wide")
    st.sidebar.title("ovo-ligand")
    st.sidebar.caption("Docker-backed ligand workflows")

    workflow_pages = _workflow_pages()
    package_root = Path(ovo_ligand.__file__).resolve().parent
    jobs_page = st.Page(
        page=str(package_root / "app/pages/jobs.py"),
        title="Jobs",
        url_path="jobs",
    )
    for page in workflow_pages:
        st.sidebar.page_link(page)
    st.sidebar.page_link(jobs_page)

    st.sidebar.divider()
    st.sidebar.caption(f"Runs: {os.getenv('OVO_LIGAND_RUN_DIR', '/tmp/ovo-ligand-runs')}")

    pg = st.navigation({"Ligand-X": workflow_pages, "Runs": [jobs_page]}, position="hidden")
    pg.run()


if __name__ == "__main__":
    main()
