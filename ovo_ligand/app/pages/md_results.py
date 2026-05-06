from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ovo_ligand.app.pages.bound_ligand_md import _render_md_results, _rewrite_output_paths, _run_root


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def render() -> None:
    st.title("MD Results")
    qp = st.query_params
    run_id = str(qp.get("run_id", "")).strip()
    if not run_id:
        st.info("No run selected. Open this page from the Jobs list.")
        if st.button("Back to Jobs"):
            st.switch_page("app/pages/jobs_md.py")
        return

    run_dir = _run_root() / "bound-ligand-md" / run_id
    if not run_dir.exists():
        st.error(f"Run not found: {run_id}")
        if st.button("Back to Jobs"):
            st.switch_page("app/pages/jobs_md.py")
        return

    metadata = _read_json(run_dir / "metadata.json")
    result = _read_json(run_dir / "result.json")

    top = st.columns([0.85, 0.15])
    with top[0]:
        st.caption(f"Run: {run_id}")
        if metadata.get("pdb_id"):
            st.caption(f"PDB: {metadata.get('pdb_id')}")
    with top[1]:
        if st.button("Back to Jobs"):
            st.switch_page("app/pages/jobs_md.py")

    if not result:
        st.warning("No result.json available for this run yet.")
        return

    rewritten = _rewrite_output_paths(result, run_dir)
    _render_md_results(rewritten, run_dir)


render()
