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
    select_target_artifact,
)
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import display_job_code, iter_job_records
from mn_ligand.core.provenance import target_lineage_summary
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.terminal_repair import (
    TASK_GROUP,
    create_terminal_repair_job,
    modeller_readiness,
    infer_terminal_sequence,
    normalize_extension_sequence,
    pdb_chain_sequences,
    pdb_seqres_sequences,
)


def _jobs():
    return iter_job_records(runs_root(), task_groups=(TASK_GROUP,))


def _protein_and_heterogen_pdb(pdb_data: str) -> tuple[str, str]:
    protein = "\n".join(
        [
            *(line for line in pdb_data.splitlines() if line.startswith("ATOM  ")),
            "TER",
            "END",
            "",
        ]
    )
    heterogens = "\n".join(
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
    return protein, heterogens


def _render_repair_molstar(
    *,
    pdb_data: str,
    chain: str,
    observed_start: int,
    observed_end: int,
    extension_end: int | None = None,
    key: str,
) -> None:
    protein, heterogens = _protein_and_heterogen_pdb(pdb_data)
    molstar_chain = "" if chain == "_" else chain
    structures = [
        StructureVisualization(
            pdb=protein,
            color="uniform",
            color_params={"value": "0xcbd5e1"},
            representation_type="cartoon",
            highlighted_selections=[
                f"{molstar_chain}{observed_start}-{extension_end or observed_end}"
            ],
            chains=[
                ChainVisualization(
                    chain_id=molstar_chain,
                    color="uniform",
                    color_params={"value": "0x2563eb"},
                    representation_type="cartoon",
                    residues=list(range(observed_start, observed_end + 1)),
                    label=f"Observed chain {chain}",
                ),
                ChainVisualization(
                    chain_id=molstar_chain,
                    color="uniform",
                    color_params={"value": "0xf97316"},
                    representation_type="ball-and-stick",
                    residues=list(
                        range(
                            observed_end + 1 if extension_end else observed_end,
                            (extension_end or observed_end) + 1,
                        )
                    ),
                    label=(
                        "Modeled C-terminal extension"
                        if extension_end
                        else "Current C-terminus"
                    ),
                ),
            ],
        )
    ]
    if any(line.startswith("HETATM") for line in heterogens.splitlines()):
        structures.append(
            StructureVisualization(
                pdb=heterogens,
                color="uniform",
                color_params={"value": "0xd946ef"},
                representation_type="ball-and-stick",
            )
        )
    molstar_custom_component(
        structures=structures,
        key=key,
        height=650,
        width="100%",
        show_controls=True,
        selection_mode=True,
        force_reload=True,
    )
    st.caption(
        "Mol* shows the coordinate sequence and residue numbers. The observed selected "
        "chain is blue, the current or modeled C-terminal boundary is orange, other "
        "protein chains are grey, and retained ligand/cofactors are magenta. Use the "
        "sequence panel or click a residue to inspect it."
    )


def _ancestry_pdb_id(job) -> str:
    jobs = iter_job_records(runs_root())
    jobs_by_id = {item.run_id: item for item in jobs}
    current = job
    visited: set[str] = set()
    while current is not None and current.run_id not in visited:
        visited.add(current.run_id)
        pdb_id = str(current.metadata.get("pdb_id") or "").strip().upper()
        if pdb_id:
            return pdb_id
        current = jobs_by_id.get(current.parent_run_id)
    return ""


def _reference_sequence(job, chain: str) -> tuple[str, str]:
    """Resolve SEQRES from the selected artifact lineage or matching PDB import."""
    pdb_id = _ancestry_pdb_id(job)
    if not pdb_id:
        return "", ""
    candidates = []
    for candidate in iter_job_records(runs_root()):
        if candidate.status != "completed" or candidate.artifact_manifest is None:
            continue
        candidate_pdb = str(candidate.metadata.get("pdb_id") or "").strip().upper()
        if pdb_id and candidate_pdb != pdb_id:
            continue
        for artifact_type in ("imported_target", "prepared_complex"):
            for artifact in candidate.artifact_manifest.by_type(artifact_type):
                path = artifact.resolve(candidate.run_dir, must_exist=True)
                if path is None or path.suffix.lower() not in {".pdb", ".ent"}:
                    continue
                sequence = pdb_seqres_sequences(path.read_text(errors="replace")).get(chain)
                if sequence:
                    candidates.append((sequence, candidate.run_id, path.name))
    if not candidates:
        return "", ""
    sequence, run_id, name = max(candidates, key=lambda item: len(item[0]))
    return sequence, f"{name} · source job {display_job_code(None, run_id)}"


def render(*, embedded: bool = False) -> None:
    if embedded:
        st.subheader("C-terminal Repair")
    else:
        st.title("Repair")
    st.caption(
        "Append a short C-terminal amino-acid sequence with MODELLER while preserving "
        "the selected imported complex and ligand. The source structure is never modified."
    )
    target_tab, repair_tab, run_tab, results_tab = st.tabs(
        ["Target", "Repair", "Run", "Results"]
    )

    with target_tab:
        target = select_target_artifact(
            "prepared complex",
            ("prepared_complex",),
            key="terminal_repair_target",
            show_viewer=False,
        )
        if target is not None:
            render_selected_artifacts({"Source complex": target})
            target_path = target.artifact.resolve(target.job.run_dir, must_exist=True)
            if (
                target_path is not None
                and target_path.suffix.lower() in {".pdb", ".ent"}
            ):
                target_data = target_path.read_text(errors="replace")
                target_chains = pdb_chain_sequences(target_data)
                if target_chains:
                    first_chain = target_chains[0]
                    _render_repair_molstar(
                        pdb_data=target_data,
                        chain=str(first_chain["chain"]),
                        observed_start=int(first_chain["start"]),
                        observed_end=int(first_chain["end"]),
                        key="terminal_repair_source_viewer",
                    )
        else:
            st.info("Import or prepare a protein-ligand complex before repairing it.")

    source_path = (
        target.artifact.resolve(target.job.run_dir, must_exist=True)
        if target is not None
        else None
    )
    chain_rows = (
        pdb_chain_sequences(source_path.read_text(errors="replace"))
        if source_path is not None and source_path.suffix.lower() in {".pdb", ".ent"}
        else ()
    )
    chain = ""
    extension = ""
    model_count = 10
    valid_parameters = False
    with repair_tab:
        if target is not None and not chain_rows:
            st.warning("The selected complex has no readable canonical protein chain.")
        if chain_rows:
            chain = st.selectbox(
                "Protein chain to extend",
                [str(row["chain"]) for row in chain_rows],
                key="terminal_repair_chain",
            )
            selected = next(row for row in chain_rows if str(row["chain"]) == chain)
            reference_sequence, reference_source = _reference_sequence(target.job, chain)
            sequence_evidence = infer_terminal_sequence(
                reference_sequence,
                str(selected["sequence"]),
            )
            probable_extension = (
                str(sequence_evidence.get("c_terminal_sequence") or "")
                if sequence_evidence.get("matched")
                else ""
            )
            extension_input = st.text_input(
                "C-terminal sequence to append",
                value=probable_extension,
                placeholder="e.g. GSG",
                help="Canonical one-letter amino-acid codes; maximum 30 residues.",
                key=f"terminal_repair_sequence_{target.job.run_id}_{chain}",
            )
            sequence_origin = (
                "pdb_seqres_proposal"
                if probable_extension and extension_input.strip().upper() == probable_extension
                else "user_edited"
            )
            if probable_extension:
                if sequence_origin == "pdb_seqres_proposal":
                    st.caption(
                        "The editable field currently matches the reference-derived "
                        f"proposal `{probable_extension}`."
                    )
                else:
                    st.caption(
                        "You changed the reference-derived proposal. The submitted "
                        "sequence will be recorded as a user-edited extension."
                    )
            model_count = int(
                st.number_input(
                    "Independent terminal conformations",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                    key="terminal_repair_models",
                    help="The best model is selected by severe clashes, then DOPE score.",
                )
            )
            try:
                extension = normalize_extension_sequence(extension_input)
                valid_parameters = True
            except ValueError as exc:
                if extension_input:
                    st.warning(str(exc))
            metrics = st.columns(4)
            metrics[0].metric("Chain", chain)
            metrics[1].metric("Observed residues", selected["residue_count"])
            metrics[2].metric("Current C-terminus", selected["end"])
            metrics[3].metric(
                "New C-terminus",
                int(selected["end"]) + len(extension) if extension else "—",
            )
            if sequence_evidence.get("matched"):
                missing_n = str(sequence_evidence.get("n_terminal_sequence") or "")
                missing_c = str(sequence_evidence.get("c_terminal_sequence") or "")
                st.markdown(
                    "#### C-terminal sequence context\n"
                    f"Observed coordinate tail: `{str(selected['sequence'])[-30:]}`  \n"
                    f"Probable missing continuation: "
                    f"`{missing_c or 'none declared'}`"
                )
                st.info(
                    "Declared-sequence comparison: "
                    f"{len(missing_n)} probable missing N-terminal residue(s)"
                    + (f" (`{missing_n}`)" if missing_n else "")
                    + " and "
                    f"{len(missing_c)} probable missing C-terminal residue(s)"
                    + (f" (`{missing_c}`)" if missing_c else "")
                    + f". Reference: {reference_source}."
                )
                if not missing_c:
                    st.warning(
                        "The declared sequence has no missing C-terminal residues. "
                        "Anything entered below is a user-defined extension."
                    )
            elif reference_sequence:
                st.warning(
                    "A declared reference sequence was found, but the observed chain "
                    "does not map uniquely to it. Repair will not guess missing residues."
                )
            else:
                st.warning(
                    "No declared reference sequence is available for this imported "
                    "chain. Mol* shows the observed coordinate sequence, but missing "
                    "terminal identities cannot be inferred automatically."
                )
            if selected["gaps"]:
                st.error(
                    "This chain has unresolved internal coordinate gaps: "
                    + ", ".join(f"{left}–{right}" for left, right in selected["gaps"])
                    + ". Internal gaps must be repaired before terminal extension."
                )
                valid_parameters = False
            if selected["has_insertions"]:
                st.error("Chains with insertion codes are not currently supported.")
                valid_parameters = False
            st.caption(
                f"Observed sequence ends …{str(selected['sequence'])[-20:]}. "
                "The added residues are modeled as a flexible terminus; they are not "
                "experimental coordinates and require review/equilibration before MD. "
                "Probable missing residues cannot appear in the 3D viewer until they "
                "have coordinates; after Repair they are highlighted in orange."
            )
            _render_repair_molstar(
                pdb_data=source_path.read_text(errors="replace"),
                chain=chain,
                observed_start=int(selected["start"]),
                observed_end=int(selected["end"]),
                key=f"terminal_repair_preview_{target.job.run_id}_{chain}",
            )

    with run_tab:
        ready, readiness_message = modeller_readiness()
        if ready:
            st.success(readiness_message)
        else:
            st.error(readiness_message)
        render_run_resources(
            requires_gpu=False,
            selected_gpu="Not used",
            key="terminal_repair_resources",
        )
        st.caption(
            "MODELLER builds an ensemble for the appended residues and junction. "
            "The original complex, other chains, and ligand coordinates are retained."
        )
        can_run = (
            target is not None
            and source_path is not None
            and bool(chain)
            and valid_parameters
            and ready
        )
        if target is None:
            st.link_button("Open Structure Import", "./workspace-structure-preparation")
        if st.button(
            "Run C-terminal repair",
            type="primary",
            disabled=not can_run,
            key="run_terminal_repair",
        ) and target is not None and source_path is not None:
            with st.spinner("Building and validating terminal conformations…"):
                job = create_terminal_repair_job(
                    source_job=target.job,
                    source_artifact=target.artifact,
                    source_path=source_path,
                    chain=chain,
                    extension_sequence=extension,
                    model_count=model_count,
                    sequence_origin=sequence_origin,
                    sequence_evidence=sequence_evidence,
                )
            code = display_job_code(job.metadata.get("job_code"), job.run_id)
            if job.status == "completed":
                st.success(f"Repair {code} completed.")
            else:
                st.error(str(job.result.get("error") or f"Repair {code} failed."))

    with results_tab:
        jobs = _jobs()
        if not jobs:
            st.info("No C-terminal Repair jobs yet.")
        else:
            all_jobs = iter_job_records(runs_root())
            jobs_by_id = {job.run_id: job for job in all_jobs}
            rows = []
            for job in jobs:
                code = display_job_code(job.metadata.get("job_code"), job.run_id)
                query = urlencode(
                    {"task_group": TASK_GROUP, "run_id": job.run_id, "label": code}
                )
                context = target_lineage_summary(job, jobs_by_id)
                rows.append(
                    {
                        "Job": f"./job-results?{query}",
                        "Last step": context["last_step"],
                        "Status": job.status,
                        "Target": context["target"],
                        "Receptor": context["receptor"],
                        "Ligand": context["ligand"],
                        "Source job": display_job_code(None, job.parent_run_id),
                        "Origin / history": context["origin"],
                        "Chain": job.metadata.get("chain"),
                        "Extension": job.metadata.get("extension_sequence"),
                        "Models": job.metadata.get("completed_models")
                        or job.metadata.get("model_count"),
                        "Created": job.created_at,
                    }
                )
            event = st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                on_select="rerun",
                selection_mode="single-row",
                key="terminal_repair_results",
                column_config={
                    "Job": st.column_config.LinkColumn(
                        "Job", display_text=r"label=([^&]+)"
                    ),
                    "Created": st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD HH:mm"
                    ),
                },
            )
            selected_rows = list(
                getattr(getattr(event, "selection", None), "rows", []) or []
            )
            selected_index = int(selected_rows[0]) if selected_rows else 0
            selected_job = jobs[selected_index]
            if (
                selected_job.status == "completed"
                and selected_job.artifact_manifest is not None
                and selected_job.artifact_manifest.by_type("prepared_complex")
            ):
                artifact = selected_job.artifact_manifest.by_type("prepared_complex")[0]
                best = selected_job.result.get("best_model") or {}
                result_path = artifact.resolve(selected_job.run_dir, must_exist=True)
                if result_path is not None:
                    original_end = int(
                        selected_job.result.get("original_terminal_residue") or 0
                    )
                    _render_repair_molstar(
                        pdb_data=result_path.read_text(errors="replace"),
                        chain=str(selected_job.result.get("chain") or "_"),
                        observed_start=min(
                            (
                                int(row["start"])
                                for row in pdb_chain_sequences(
                                    result_path.read_text(errors="replace")
                                )
                                if str(row["chain"])
                                == str(selected_job.result.get("chain") or "_")
                            ),
                            default=1,
                        ),
                        observed_end=original_end,
                        extension_end=int(
                            selected_job.result.get("new_terminal_residue")
                            or original_end
                        ),
                        key=f"terminal_repair_result_{selected_job.run_id}",
                    )
                metrics = st.columns(4)
                metrics[0].metric("Chain", selected_job.result.get("chain") or "—")
                metrics[1].metric(
                    "Appended", selected_job.result.get("extension_sequence") or "—"
                )
                metrics[2].metric(
                    "Junction C–N (Å)",
                    (
                        f"{float(best['junction_cn_angstrom']):.2f}"
                        if best.get("junction_cn_angstrom") is not None
                        else "—"
                    ),
                )
                metrics[3].metric(
                    "Severe clashes",
                    best.get("heavy_atom_clashes_below_1_5_angstrom", "—"),
                )
                st.warning(str(selected_job.result.get("interpretation") or ""))

if __name__ == "__main__":
    render()
