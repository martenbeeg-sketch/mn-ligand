from __future__ import annotations

from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.components.molstar_viewer import molstar_custom_component
from mn_ligand.app.components.molstar_viewer.dataclasses import (
    ChainVisualization,
    StructureVisualization,
)
from mn_ligand.app.pages.discover_inputs import (
    render_selected_artifacts,
    render_target_viewer,
    select_target_artifact,
    target_viewer_path,
)
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import display_job_code, iter_job_records
from mn_ligand.core.provenance import target_lineage_summary
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.target_trimming import (
    TASK_GROUP,
    create_trimmed_target_job,
    pdb_chain_ranges,
    trim_pdb_data,
)


def _render_trimming_preview(
    *,
    original_pdb: str,
    trimmed_pdb: str,
    ranges: dict[str, tuple[int, int]],
    key: str,
) -> None:
    """Render the prospective trimmed complex before creating a job."""
    st.markdown("#### Complex preview")
    mode = st.radio(
        "Preview structure",
        ("Trimmed complex", "Original with retained-region overlay"),
        horizontal=True,
        key=f"{key}_mode",
    )
    pdb_data = trimmed_pdb if mode == "Trimmed complex" else original_pdb
    protein_pdb = "\n".join(
        [
            *(line for line in pdb_data.splitlines() if line.startswith("ATOM  ")),
            "TER",
            "END",
            "",
        ]
    )
    heterogen_pdb = "\n".join(
        [
            *(
                line
                for line in pdb_data.splitlines()
                if line.startswith("HETATM")
                and line[17:20].strip().upper() not in {"HOH", "WAT", "DOD"}
            ),
            "END",
            "",
        ]
    )
    structures: list[StructureVisualization] = []
    retained_selections = [
        f"{'' if chain == '_' else chain}{int(start)}-{int(end)}"
        for chain, (start, end) in ranges.items()
    ]
    if mode == "Trimmed complex":
        structures.append(
            StructureVisualization(
                pdb=protein_pdb,
                color="uniform",
                color_params={"value": "0x2563eb"},
                representation_type="cartoon",
                highlighted_selections=retained_selections,
            )
        )
    else:
        structures.append(
            StructureVisualization(
                pdb=protein_pdb,
                color="uniform",
                color_params={"value": "0xcbd5e1"},
                representation_type="cartoon",
                highlighted_selections=retained_selections,
                chains=[
                    ChainVisualization(
                        chain_id="" if chain == "_" else chain,
                        color="uniform",
                        color_params={"value": "0x2563eb"},
                        representation_type="cartoon",
                        residues=list(range(int(start), int(end) + 1)),
                        label=f"Retained {chain}:{start}-{end}",
                    )
                    for chain, (start, end) in ranges.items()
                ],
            )
        )
    if any(line.startswith("HETATM") for line in heterogen_pdb.splitlines()):
        structures.append(
            StructureVisualization(
                pdb=heterogen_pdb,
                color="uniform",
                color_params={"value": "0xd946ef"},
                representation_type="ball-and-stick",
            )
        )
    molstar_custom_component(
        structures=structures,
        key=f"{key}_{mode}_{sorted(ranges.items())}",
        height=650,
        width="100%",
        show_controls=True,
        selection_mode=True,
        force_reload=True,
    )
    if mode == "Trimmed complex":
        st.caption(
            "Prospective output in Mol*: retained protein and its sequence are blue; the "
            "unchanged ligand and other retained non-water heterogens are magenta. "
            "Use the sequence panel or click the structure to inspect residue numbers."
        )
    else:
        st.caption(
            "Original-complex context in Mol*: retained protein and sequence residues "
            "are blue; residues that would be trimmed away are grey. The ligand remains "
            "magenta and is not trimmed. Use the sequence panel or click residues to "
            "inspect numbering."
        )


def _result_rows() -> list[dict[str, object]]:
    all_jobs = iter_job_records(runs_root())
    jobs_by_id = {job.run_id: job for job in all_jobs}
    rows = []
    for job in (item for item in all_jobs if item.task_group == TASK_GROUP):
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        query = urlencode({"task_group": TASK_GROUP, "run_id": job.run_id, "label": code})
        context = target_lineage_summary(job, jobs_by_id)
        rows.append(
            {
                "result": f"./job-results?{query}",
                "Last step": context["last_step"],
                "job": code,
                "status": job.status,
                "target": context["target"],
                "receptor": context["receptor"],
                "ligand": context["ligand"],
                "source job": display_job_code(None, job.parent_run_id),
                "origin / history": context["origin"],
                "ranges": ", ".join(
                    f"{chain}:{bounds.get('start')}-{bounds.get('end')}"
                    for chain, bounds in (job.metadata.get("trim_ranges") or {}).items()
                ),
                "created": job.created_at,
            }
        )
    return rows


def render(*, embedded: bool = False) -> None:
    if embedded:
        st.subheader("Target Trimming")
    else:
        st.title("Target Trimming")
    st.caption(
        "Trim protein N- and C-termini in any typed prepared complex while "
        "retaining its ligand unchanged. The source complex is never modified."
    )
    target_tab, trim_tab, run_tab, results_tab = st.tabs(
        ["Target", "Trim", "Run", "Results"]
    )
    with target_tab:
        target = select_target_artifact(
            "prepared complex",
            ("prepared_complex",),
            key="target_trimming_source",
            show_viewer=False,
        )
        if target is not None:
            render_selected_artifacts({"Source target": target})
            render_target_viewer(
                target,
                viewer_path=target_viewer_path(target),
                key="target_trimming_viewer",
            )
        else:
            st.info("Import, prepare, trim, or repair a protein-ligand complex first.")

    source_path = (
        target.artifact.resolve(target.job.run_dir, must_exist=True)
        if target is not None
        else None
    )
    chain_rows = (
        pdb_chain_ranges(source_path.read_text(errors="replace"))
        if source_path is not None and source_path.suffix.lower() in {".pdb", ".ent"}
        else ()
    )
    ranges: dict[str, tuple[int, int]] = {}
    with trim_tab:
        if target is not None and not chain_rows:
            st.warning("The selected target is not a PDB structure with readable residue ranges.")
        for row in chain_rows:
            chain = str(row["chain"])
            st.markdown(f"#### Chain {chain}")
            columns = st.columns(2)
            start = int(
                columns[0].number_input(
                    "N-terminal residue to keep",
                    min_value=int(row["start"]),
                    max_value=int(row["end"]),
                    value=int(row["start"]),
                    step=1,
                    key=f"trim_{target.job.run_id}_{chain}_start" if target else f"trim_{chain}_start",
                )
            )
            end = int(
                columns[1].number_input(
                    "C-terminal residue to keep",
                    min_value=int(row["start"]),
                    max_value=int(row["end"]),
                    value=int(row["end"]),
                    step=1,
                    key=f"trim_{target.job.run_id}_{chain}_end" if target else f"trim_{chain}_end",
                )
            )
            if end < start:
                st.error(f"Chain {chain}: the C-terminal residue must be at or after the N-terminal residue.")
            else:
                ranges[chain] = (start, end)
        if source_path is not None and ranges:
            try:
                original_pdb = source_path.read_text(errors="replace")
                trimmed_pdb, preview = trim_pdb_data(original_pdb, ranges)
                metrics = st.columns(3)
                metrics[0].metric("Selected chains", preview["chain_count"])
                metrics[1].metric("Retained residues", preview["residue_count"])
                metrics[2].metric(
                    "Retained ligand atoms", preview["retained_ligand_atom_count"]
                )
                _render_trimming_preview(
                    original_pdb=original_pdb,
                    trimmed_pdb=trimmed_pdb,
                    ranges=ranges,
                    key=f"target_trimming_preview_{target.job.run_id}",
                )
            except ValueError as exc:
                st.warning(str(exc))

    with run_tab:
        render_run_resources(requires_gpu=False, selected_gpu="Not used", key="target_trimming")
        can_run = target is not None and source_path is not None and bool(ranges)
        if target is None:
            st.link_button("Open Structure Import", "./workspace-structure-preparation")
        if st.button(
            "Create trimmed target",
            type="primary",
            disabled=not can_run,
            key="create_trimmed_target",
        ) and target is not None and source_path is not None:
            try:
                job = create_trimmed_target_job(
                    source_job=target.job,
                    source_artifact=target.artifact,
                    source_path=source_path,
                    ranges=ranges,
                )
                st.success(
                    "Created trimmed prepared target "
                    f"{display_job_code(job.metadata.get('job_code'), job.run_id)}."
                )
            except Exception as exc:
                st.error(str(exc))

    with results_tab:
        rows = _result_rows()
        if not rows:
            st.info("No trimmed targets yet.")
        else:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config={"result": st.column_config.LinkColumn("Result", display_text="Open")},
            )

if __name__ == "__main__":
    render()
