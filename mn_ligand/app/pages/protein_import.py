from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.core.jobs import display_job_code, iter_job_records
from mn_ligand.core.provenance import target_lineage_summary
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.bound_ligand_md import download_pdb
from mn_ligand.workflows.protein_preparation import create_protein_import_job, list_protein_import_jobs


def _job_url(run_id: str, code: str) -> str:
    query = urlencode({"task_group": "protein-import", "run_id": run_id, "label": code})
    return f"./job-results?{query}"


def _render_recent_imports() -> None:
    jobs = list_protein_import_jobs()
    if not jobs:
        st.info("No imported proteins yet.")
        return
    all_jobs = iter_job_records(runs_root())
    jobs_by_id = {job.run_id: job for job in all_jobs}
    rows = []
    for job in jobs:
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        context = target_lineage_summary(job, jobs_by_id)
        rows.append(
            {
                "job": _job_url(job.run_id, code),
                "Last step": context["last_step"],
                "source": job.metadata.get("source") or "",
                "target": context["target"],
                "receptor": context["receptor"],
                "origin / history": context["origin"],
                "format": job.metadata.get("structure_format") or "",
                "chains": ", ".join(job.result.get("chains") or []),
                "residues": job.result.get("protein_residues"),
                "created": job.created_at,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"job": st.column_config.LinkColumn("Job", display_text=r"label=([^&]+)")},
    )

    options = {
        f"{display_job_code(job.metadata.get('job_code'), job.run_id)} | "
        f"{job.metadata.get('pdb_id') or job.result.get('format') or 'target'}": job
        for job in jobs
    }
    selected_label = st.selectbox("Imported target", list(options), key="protein_import_clean_target")
    selected = options[selected_label]
    st.link_button(
        "Open in Protein Cleaning",
        f"./workspace-protein-cleaning?{urlencode({'source_run_id': selected.run_id})}",
    )


def render() -> None:
    st.title("Protein Import")
    pdb_tab, upload_tab = st.tabs(["PDB ID", "Upload structure"])

    with pdb_tab:
        pdb_id = st.text_input("PDB ID", value="4LNW", max_chars=4, key="protein_import_pdb_id").strip().upper()
        if st.button("Import from RCSB", type="primary", key="protein_import_rcsb"):
            try:
                with st.spinner(f"Importing {pdb_id} from RCSB..."):
                    data = download_pdb(pdb_id)
                    job = create_protein_import_job(
                        data,
                        filename=f"{pdb_id.lower()}.pdb",
                        source="pdb",
                        pdb_id=pdb_id,
                    )
                st.success(f"Imported protein {display_job_code(job.metadata.get('job_code'), job.run_id)}")
                st.session_state["last_protein_import_run_id"] = job.run_id
            except Exception as exc:
                st.error(f"Protein import failed: {exc}")

    with upload_tab:
        uploaded = st.file_uploader(
            "Protein structure",
            type=["pdb", "ent", "cif", "mmcif"],
            key="protein_import_upload",
        )
        if st.button("Register uploaded structure", type="primary", key="protein_import_register"):
            if uploaded is None:
                st.warning("Select a PDB or mmCIF file.")
            else:
                try:
                    data = uploaded.getvalue().decode("utf-8")
                    job = create_protein_import_job(
                        data,
                        filename=Path(uploaded.name).name,
                        source="upload",
                    )
                    st.success(f"Imported protein {display_job_code(job.metadata.get('job_code'), job.run_id)}")
                    st.session_state["last_protein_import_run_id"] = job.run_id
                except Exception as exc:
                    st.error(f"Protein import failed: {exc}")

    st.markdown("#### Imported targets")
    _render_recent_imports()


render()
