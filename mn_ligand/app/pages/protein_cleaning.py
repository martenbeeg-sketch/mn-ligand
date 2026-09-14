from __future__ import annotations

import shlex
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.core.jobs import display_job_code, iter_job_records
from mn_ligand.core.provenance import target_lineage_summary
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.protein_preparation import (
    DEFAULT_PROTEIN_CLEANING_IMAGE,
    imported_target,
    list_protein_cleaning_jobs,
    list_protein_import_jobs,
    run_protein_cleaning_job,
)


def _job_url(task_group: str, run_id: str, code: str) -> str:
    query = urlencode({"task_group": task_group, "run_id": run_id, "label": code})
    return f"./job-results?{query}"


def _render_cleaned_targets() -> None:
    jobs = list_protein_cleaning_jobs()
    if not jobs:
        st.info("No cleaned proteins yet.")
        return
    all_jobs = iter_job_records(runs_root())
    jobs_by_id = {job.run_id: job for job in all_jobs}
    rows = []
    for job in jobs:
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        source_code = display_job_code(None, job.parent_run_id) if job.parent_run_id else ""
        context = target_lineage_summary(job, jobs_by_id)
        rows.append(
            {
                "job": _job_url("protein-cleaning", job.run_id, code),
                "Last step": context["last_step"],
                "status": job.status,
                "target": context["target"],
                "receptor": context["receptor"],
                "source job": source_code,
                "origin / history": context["origin"],
                "tool": job.tool,
                "cleaned": job.result.get("protein_cleaned"),
                "created": job.created_at,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"job": st.column_config.LinkColumn("Job", display_text=r"label=([^&]+)")},
    )
    reusable = [job for job in jobs if job.status == "completed" and job.artifact_manifest and job.artifact_manifest.by_type("prepared_target")]
    if not reusable:
        return
    options = {
        f"{display_job_code(job.metadata.get('job_code'), job.run_id)} | "
        f"{job.metadata.get('pdb_id') or 'target'}": job
        for job in reusable
    }
    selected_label = st.selectbox("Prepared target", list(options), key="protein_cleaning_downstream_target")
    selected = options[selected_label]
    actions = st.columns(2)
    actions[0].link_button(
        "Use in Structure Import",
        f"./workspace-structure-preparation?{urlencode({'prepared_target_run_id': selected.run_id})}",
        width="stretch",
    )
    actions[1].link_button(
        "Detect Pockets",
        f"./discover-pocket-detection?{urlencode({'prepared_target_run_id': selected.run_id})}",
        width="stretch",
    )


def render() -> None:
    st.title("Protein Cleaning / Repair")
    imports = list_protein_import_jobs()
    compatible = [job for job in imports if job.status == "completed" and job.artifact_manifest and job.artifact_manifest.by_type("imported_target")]
    if not compatible:
        st.info("Import a protein structure before starting cleaning.")
        st.link_button("Open Protein Import", "./workspace-protein-import")
        st.markdown("#### Prepared targets")
        _render_cleaned_targets()
        return

    requested_run_id = str(st.query_params.get("source_run_id", "")).strip()
    options = {}
    default_index = 0
    for index, job in enumerate(compatible):
        artifact = imported_target(job)
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        label = f"{code} | {job.metadata.get('pdb_id') or artifact.label} | {job.metadata.get('source') or 'import'}"
        options[label] = job
        if job.run_id == requested_run_id:
            default_index = index
    selected_label = st.selectbox("Imported target", list(options), index=default_index, key="protein_cleaning_source")
    selected = options[selected_label]

    st.caption(
        "MODELLER builds ensembles for internal gaps up to 15 residues. Missing termini "
        "and longer gaps remain unresolved. PDBFixer/OpenMM then repair missing atoms, "
        "add hydrogens, and validate the modeled structure."
    )
    parameters = st.columns(3)
    map_modified = parameters[0].checkbox(
        "Map supported modified residues",
        value=True,
        key="protein_cleaning_map_modified",
    )
    skip_terminals = parameters[1].checkbox(
        "Skip missing terminal residues",
        value=True,
        key="protein_cleaning_skip_terminals",
    )
    refine_rebuilt = parameters[2].checkbox(
        "Refine rebuilt coordinates",
        value=True,
        key="protein_cleaning_refine_rebuilt",
    )
    execution = st.columns(3)
    max_internal_gap = int(
        execution[0].number_input(
            "Maximum internal gap to rebuild",
            min_value=0,
            max_value=100,
            value=15,
            step=1,
            key="protein_cleaning_max_internal_gap",
        )
    )
    preserve_heterogens = execution[1].checkbox(
        "Retain non-water cofactors and metals",
        value=False,
        key="protein_cleaning_preserve_heterogens",
    )
    use_gpu = execution[2].checkbox(
        "Use GPU for local refinement",
        value=False,
        key="protein_cleaning_gpu",
    )
    assembly = st.columns(2)
    use_assembly = assembly[0].checkbox(
        "Build a biological assembly",
        value=False,
        key="protein_cleaning_use_assembly",
    )
    assembly_id = assembly[1].text_input(
        "Biological assembly ID",
        value="1",
        disabled=not use_assembly,
        key="protein_cleaning_assembly_id",
    )
    image = DEFAULT_PROTEIN_CLEANING_IMAGE

    if st.button("Clean and repair protein", type="primary", key="protein_cleaning_run"):
        try:
            with st.spinner("Cleaning and repairing protein..."):
                job, native_payload = run_protein_cleaning_job(
                    selected.run_id,
                    image=image,
                    use_gpu=use_gpu,
                    map_modified_residues=map_modified,
                    skip_terminal_missing_residues=skip_terminals,
                    max_internal_gap=max_internal_gap,
                    refine_rebuilt_positions=refine_rebuilt,
                    biological_assembly_id=assembly_id if use_assembly else "",
                    preserve_nonwater_heterogens=preserve_heterogens,
                )
            code = display_job_code(job.metadata.get("job_code"), job.run_id)
            if job.status == "completed":
                st.success(f"Prepared target {code}")
                st.session_state["last_protein_cleaning_run_id"] = job.run_id
            else:
                st.error(f"Protein cleaning failed: {job.result.get('error') or 'unknown error'}")
                with st.expander("Docker command"):
                    st.code(shlex.join(native_payload.get("command") or []))
        except Exception as exc:
            st.error(f"Protein cleaning failed: {exc}")

    st.markdown("#### Prepared targets")
    _render_cleaned_targets()


render()
