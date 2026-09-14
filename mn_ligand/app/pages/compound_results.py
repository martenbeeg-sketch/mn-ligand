from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Draw,
    Lipinski,
    QED,
    rdMolDescriptors,
)

from mn_ligand.core.jobs import JobRecord, display_job_code
from mn_ligand.workflows.compound_preparation import (
    annotation_stripped_smiles_candidate,
    annotate_component_relationships,
    compound_component_records,
    normalize_compound_smiles,
    parent_duplicate_report,
)
from mn_ligand.workflows.compound_pubchem import (
    accepted_reviewed_compound_rows,
    compare_molecular_formulas,
    create_compound_review_job,
    latest_compound_review_map,
    pubchem_component_options,
    search_pubchem_candidates,
    selected_parent_candidate,
)


def _artifact_path(job: JobRecord, artifact_type: str) -> Path | None:
    if job.artifact_manifest is None:
        return None
    refs = job.artifact_manifest.by_type(artifact_type)
    return refs[0].resolve(job.run_dir, must_exist=True) if refs else None


def _read_json(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _download(
    column,
    job: JobRecord,
    artifact_type: str,
    label: str,
    mime: str,
) -> None:
    path = _artifact_path(job, artifact_type)
    if path is None:
        return
    column.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        key=f"{artifact_type}_{job.run_id}",
    )


def _source_compound_name_columns(frame: pd.DataFrame) -> list[str]:
    """Return imported human-readable compound-name columns, in file order.

    ``compound_id`` is the stable workflow key, while vendor files commonly
    keep the useful display name in a separate ``Name`` or ``Compound Name``
    column.  Preserve it rather than requiring a particular vendor schema.
    """
    recognised = {
        "name",
        "compound name",
        "compound_name",
        "chemical name",
        "chemical_name",
        "product name",
        "product_name",
        "ligand name",
        "ligand_name",
    }
    return [
        str(column)
        for column in frame.columns
        if str(column).strip().lower() in recognised
        and str(column).strip().lower() != "compound_id"
    ]


def _compound_table_columns(frame: pd.DataFrame) -> list[str]:
    preferred = (
        "compound_id",
        "structure_origin",
        "review_job",
        "Synonyms",
        "CAS Number",
        "smiles",
        "formula",
        "molecular_weight",
        "clogp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "ring_count",
        "formal_charge",
        "fragment_count",
        "qed",
        "validation_warning",
    )
    source_name_columns = _source_compound_name_columns(frame)
    selected = [
        column
        for column in ("compound_id", *source_name_columns, *preferred)
        if column in frame.columns
    ]
    selected = list(dict.fromkeys(selected))
    return selected or list(frame.columns)


def _reviewed_additions_frame(
    job: JobRecord,
    rejected_source: pd.DataFrame | None,
) -> pd.DataFrame:
    reviewed_rows = accepted_reviewed_compound_rows(job.run_id)
    if not reviewed_rows:
        return pd.DataFrame()
    source_records = (
        rejected_source.to_dict("records")
        if rejected_source is not None and not rejected_source.empty
        else []
    )
    combined: list[dict] = []
    for reviewed in reviewed_rows:
        source_row = str(reviewed.get("source_row") or "")
        compound_id = str(reviewed.get("compound_id") or "")
        source = next(
            (
                row
                for row in source_records
                if (
                    source_row
                    and str(row.get("source_row") or "") == source_row
                )
                or str(row.get("compound_id") or "") == compound_id
            ),
            {},
        )
        combined.append(
            {
                **source,
                **reviewed,
                "validation_warning": (
                    "PubChem-confirmed replacement for rejected vendor row"
                ),
            }
        )
    return pd.DataFrame(combined).fillna("")


def _component_context_table(
    components: pd.DataFrame,
    normalized: pd.DataFrame | None,
) -> pd.DataFrame:
    if normalized is not None and not normalized.empty:
        components = pd.DataFrame(
            annotate_component_relationships(
                normalized.to_dict("records"),
                components.to_dict("records"),
            )
        )
    display = components.rename(
        columns={
            "smiles": "component_smiles",
            "formula": "component_formula",
            "molecular_weight": "component_molecular_weight",
        }
    ).copy()
    heavy_atoms = pd.to_numeric(display["heavy_atoms"], errors="coerce")
    maximum = heavy_atoms.groupby(display["compound_id"]).transform("max")
    tied_maximum = heavy_atoms.eq(maximum).groupby(display["compound_id"]).transform(
        "sum"
    )
    contains_carbon = display["contains_carbon"].astype(str).str.lower().isin(
        {"true", "1"}
    )
    formula = display["component_formula"].astype(str).str.upper()
    display["component_interpretation"] = "organic co-component; review role"
    display.loc[~contains_carbon, "component_interpretation"] = (
        "non-carbon co-component; possible counterion or solvate"
    )
    display.loc[formula.isin({"H2O", "OH2"}), "component_interpretation"] = (
        "water / hydrate component"
    )
    display.loc[
        heavy_atoms.eq(maximum) & tied_maximum.eq(1),
        "component_interpretation",
    ] = "largest component; parent candidate only, review required"

    if normalized is not None and not normalized.empty:
        source_columns = [
            column
            for column in (
                "compound_id",
                "Product Name",
                "CAS Number",
                "Formula",
                "source_smiles",
            )
            if column in normalized.columns
        ]
        if len(source_columns) > 1:
            source = normalized[source_columns].rename(
                columns={
                    "Product Name": "source_product_name",
                    "CAS Number": "source_cas_number",
                    "Formula": "source_formula",
                    "source_smiles": "full_source_smiles",
                }
            )
            display = display.merge(source, on="compound_id", how="left")

    preferred = [
        "compound_id",
        "source_product_name",
        "source_cas_number",
        "source_formula",
        "component",
        "component_count",
        "component_smiles",
        "component_formula",
        "component_molecular_weight",
        "heavy_atoms",
        "formal_charge",
        "contains_carbon",
        "recognized_component",
        "component_category",
        "component_interpretation",
        "formulation_category",
        "parent_candidate",
        "parent_match_status",
        "unformulated_library_matches",
        "related_formulations",
        "full_source_smiles",
        "source_row",
    ]
    return display[[column for column in preferred if column in display.columns]]


def render_selected_compound(frame: pd.DataFrame, selected_index: int) -> None:
    selected = frame.iloc[selected_index].to_dict()
    smiles = str(
        selected.get("smiles")
        or selected.get("standardized_parent_smiles")
        or ""
    )
    compound_id = str(
        selected.get("compound_id")
        or selected.get("representative_compound_id")
        or selected.get("docking_parent_id")
        or "Compound"
    )
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        calculated = {
            "formula": rdMolDescriptors.CalcMolFormula(molecule),
            "molecular_weight": round(float(Descriptors.MolWt(molecule)), 4),
            "clogp": round(float(Crippen.MolLogP(molecule)), 4),
            "tpsa": round(float(rdMolDescriptors.CalcTPSA(molecule)), 4),
            "hbd": int(Lipinski.NumHDonors(molecule)),
            "hba": int(Lipinski.NumHAcceptors(molecule)),
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
            "ring_count": int(Lipinski.RingCount(molecule)),
            "formal_charge": int(Chem.GetFormalCharge(molecule)),
            "fragment_count": len(Chem.GetMolFrags(molecule)),
            "qed": round(float(QED.qed(molecule)), 4),
        }
        for property_name, value in calculated.items():
            if selected.get(property_name) in ("", None):
                selected[property_name] = value
    structure_column, detail_column = st.columns([1, 2])
    with structure_column:
        st.markdown("#### Selected compound")
        if molecule is None:
            st.warning("The normalized structure could not be rendered.")
        else:
            st.image(
                Draw.MolToImage(molecule, size=(560, 420)),
                caption=compound_id,
                width="stretch",
            )
    with detail_column:
        st.markdown("#### Source and calculated properties")
        omitted = {"source_smiles", "source_row"}
        source_name_columns = _source_compound_name_columns(frame)
        property_order = list(
            dict.fromkeys(
                ["compound_id", *source_name_columns, *selected.keys()]
            )
        )
        rows = [
            {"Property": key, "Value": str(selected.get(key, ""))}
            for key in property_order
            if key not in omitted and str(selected.get(key, "")).strip()
        ]
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            height=420,
        )


def render_compound_dataset_report(job: JobRecord) -> None:
    report = _read_json(_artifact_path(job, "import_report"))
    validation = _read_json(_artifact_path(job, "compound_validation_report"))
    summary = validation.get("summary") or report.get("validation") or {}
    job_code = display_job_code(job.metadata.get("job_code"), job.run_id)
    normalized_path = _artifact_path(job, "compound_set")
    rejected_path = _artifact_path(job, "rejected_compounds")
    normalized_frame = (
        pd.read_csv(normalized_path, keep_default_na=False)
        if normalized_path is not None
        and normalized_path.suffix.lower() == ".csv"
        else pd.DataFrame()
    )
    rejected_source = (
        pd.read_csv(rejected_path, keep_default_na=False)
        if rejected_path is not None
        else pd.DataFrame()
    )
    review_map = latest_compound_review_map(job.run_id)
    reviewed_additions = _reviewed_additions_frame(job, rejected_source)
    effective_frame = pd.concat(
        [normalized_frame, reviewed_additions],
        ignore_index=True,
        sort=False,
    ).fillna("")
    for numeric_column in (
        "molecular_weight",
        "exact_mass",
        "heavy_atoms",
        "hbd",
        "hba",
        "clogp",
        "tpsa",
        "rotatable_bonds",
        "ring_count",
        "formal_charge",
        "fraction_csp3",
        "qed",
        "fragment_count",
    ):
        if numeric_column in effective_frame.columns:
            effective_frame[numeric_column] = pd.to_numeric(
                effective_frame[numeric_column], errors="coerce"
            )
    resolved_ids = set(review_map)
    active_rejected_count = (
        int(
            (
            ~rejected_source["compound_id"]
            .astype(str)
            .isin(resolved_ids)
            ).sum()
        )
        if not rejected_source.empty
        else 0
    )
    effective_fragment_counts = pd.to_numeric(
        effective_frame.get("fragment_count"), errors="coerce"
    )
    effective_multi_count = int(effective_fragment_counts.gt(1).sum())
    duplicate_analysis = parent_duplicate_report(
        effective_frame.to_dict("records")
    )
    duplicate_summary = duplicate_analysis["summary"]
    parent_duplicate_frame = pd.DataFrame(duplicate_analysis["rows"])
    docking_parent_frame = pd.DataFrame(
        duplicate_analysis["docking_parent_rows"]
    )

    st.markdown("### Compound dataset")
    st.caption(
        f"{job.metadata.get('dataset_name') or report.get('dataset_name') or 'Dataset'} "
        f"· job {job_code} · worksheet "
        f"{report.get('sheet_name') or 'not applicable'} · "
        f"ID: {report.get('id_column') or 'generated'} · "
        f"SMILES: {report.get('smiles_column') or 'native'}"
    )
    st.link_button(
        "Compare target campaigns for this dataset",
        (
            "./compound-campaign-comparison?"
            + urlencode({"dataset_run_id": job.run_id})
        ),
        help=(
            "Compare this compound dataset across selected targets, engines "
            "and completed docking, cofolding and rescoring campaigns."
        ),
    )

    first_metrics = st.columns(7)
    first_metrics[0].metric("Source rows", summary.get("source_row_count", "—"))
    first_metrics[1].metric("Usable", len(effective_frame))
    first_metrics[2].metric("Needs review", active_rejected_count)
    first_metrics[3].metric(
        "Unique parent SMILES",
        duplicate_summary["unique_parent_count"],
        help=(
            "Distinct stereochemistry-aware standardized parent structures. "
            "Counterions and removable charge are ignored for this identity."
        ),
    )
    first_metrics[4].metric(
        "Redundant sources",
        duplicate_summary["redundant_entry_count"],
        help=(
            "Additional catalog or formulation records that resolve to an "
            "already counted unique parent."
        ),
    )
    first_metrics[5].metric("Multi-component", effective_multi_count)
    first_metrics[6].metric(
        "Mean MW",
        (
            f"{float(summary['molecular_weight_mean']):.2f}"
            if summary.get("molecular_weight_mean") is not None
            else "—"
        ),
    )

    download_columns = st.columns(4)
    _download(
        download_columns[0],
        job,
        "source_compound_dataset",
        "Original workbook",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    _download(
        download_columns[1],
        job,
        "compound_set",
        "Validated compound set",
        "text/csv",
    )
    _download(
        download_columns[2],
        job,
        "compound_components",
        "Component report",
        "text/csv",
    )
    _download(
        download_columns[3],
        job,
        "rejected_compounds",
        "Rejected rows",
        "text/csv",
    )

    (
        compounds_tab,
        components_tab,
        docking_parents_tab,
        duplicates_tab,
        rejected_tab,
        profiles_tab,
    ) = st.tabs(
        [
            "Usable compounds",
            "Multi-fragment compounds",
            "Docking-ready parents",
            "Parent duplicates",
            "Rejected / invalid",
            "Profiles",
        ]
    )
    with compounds_tab:
        if normalized_path is None or normalized_path.suffix.lower() != ".csv":
            st.info("Per-compound inspection is available for CSV/Excel imports.")
        else:
            frame = effective_frame
            if frame.empty:
                st.info("The normalized compound set is empty.")
            else:
                search = st.text_input(
                    "Search compounds",
                    key=f"compound_report_search_{job.run_id}",
                    placeholder="ID, product name, CAS, SMILES, or source metadata",
                )
                filtered = frame
                if search.strip():
                    mask = frame.astype(str).apply(
                        lambda column: column.str.contains(
                            search.strip(), case=False, regex=False
                        )
                    ).any(axis=1)
                    filtered = frame.loc[mask].reset_index(drop=True)
                st.caption(
                    f"Showing {len(filtered):,} of {len(frame):,} usable compounds"
                    f", including {len(reviewed_additions):,} "
                    "PubChem-confirmed review additions."
                )
                if filtered.empty:
                    st.info("No compounds match the search.")
                else:
                    event = st.dataframe(
                        filtered[_compound_table_columns(filtered)],
                        hide_index=True,
                        width="stretch",
                        height=min(500, 38 + 35 * min(len(filtered), 13)),
                        on_select="rerun",
                        selection_mode="single-row",
                        key=f"compound_report_table_{job.run_id}",
                        column_config={
                            "molecular_weight": st.column_config.NumberColumn(
                                format="%.2f"
                            ),
                            "clogp": st.column_config.NumberColumn(format="%.2f"),
                            "tpsa": st.column_config.NumberColumn(format="%.2f"),
                            "qed": st.column_config.NumberColumn(format="%.3f"),
                        },
                    )
                    selected_rows = list(
                        getattr(getattr(event, "selection", None), "rows", [])
                        or []
                    )
                    selected_index = (
                        int(selected_rows[0]) if selected_rows else 0
                    )
                    render_selected_compound(filtered, selected_index)

    with components_tab:
        components_path = _artifact_path(job, "compound_components")
        reviewed_component_rows: list[dict] = []
        for reviewed in reviewed_additions.to_dict("records"):
            if int(float(reviewed.get("fragment_count") or 1)) <= 1:
                continue
            try:
                reviewed_component_rows.extend(
                    compound_component_records(
                        str(reviewed.get("smiles") or ""),
                        compound_id=str(reviewed.get("compound_id") or ""),
                        source_row=int(reviewed.get("source_row") or 0),
                    )
                )
            except (TypeError, ValueError):
                continue
        if components_path is None and not reviewed_component_rows:
            st.success("No multi-component structures were reported.")
        else:
            component_frames = []
            if components_path is not None:
                component_frames.append(
                    pd.read_csv(components_path, keep_default_na=False)
                )
            if reviewed_component_rows:
                component_frames.append(pd.DataFrame(reviewed_component_rows))
            components = pd.concat(
                component_frames, ignore_index=True, sort=False
            ).fillna("")
            st.info(
                "These are the disconnected components of the supplied structures. "
                "Confirmed PubChem additions are included. No parent, salt, "
                "counterion, solvent, or co-component was removed."
            )
            normalized: pd.DataFrame | None = None
            component_context: pd.DataFrame | None = None
            if not effective_frame.empty:
                normalized = effective_frame
                component_context = _component_context_table(
                    components, normalized
                )
                fragment_counts = pd.to_numeric(
                    normalized.get("fragment_count"), errors="coerce"
                )
                multi_fragment = normalized.loc[
                    fragment_counts.gt(1)
                ].reset_index(drop=True)
                if not multi_fragment.empty:
                    relationships = (
                        component_context.groupby("compound_id", as_index=False)
                        .agg(
                            recognized_formulation_components=(
                                "recognized_component",
                                lambda values: ", ".join(
                                    sorted(
                                        {
                                            str(value)
                                            for value in values
                                            if str(value).strip()
                                        }
                                    )
                                ),
                            ),
                            formulation_category=(
                                "formulation_category",
                                "first",
                            ),
                            parent_match_status=(
                                "parent_match_status",
                                "first",
                            ),
                            unformulated_library_matches=(
                                "unformulated_library_matches",
                                "first",
                            ),
                            related_formulations=(
                                "related_formulations",
                                "first",
                            ),
                        )
                    )
                    fragment_smiles = (
                        component_context.groupby("compound_id", as_index=False)
                        .agg(
                            all_fragment_smiles=(
                                "component_smiles",
                                lambda values: " | ".join(
                                    str(value) for value in values
                                ),
                            )
                        )
                    )
                    parent_smiles = (
                        component_context.loc[
                            component_context["parent_candidate"].astype(bool)
                        ]
                        .groupby("compound_id", as_index=False)[
                            "component_smiles"
                        ]
                        .first()
                        .rename(
                            columns={
                                "component_smiles": "parent_candidate_smiles"
                            }
                        )
                    )
                    other_smiles = (
                        component_context.loc[
                            ~component_context["parent_candidate"].astype(bool)
                        ]
                        .groupby("compound_id", as_index=False)
                        .agg(
                            other_fragment_smiles=(
                                "component_smiles",
                                lambda values: " | ".join(
                                    str(value) for value in values
                                ),
                            )
                        )
                    )
                    relationships = (
                        relationships.merge(
                            fragment_smiles, on="compound_id", how="left"
                        )
                        .merge(parent_smiles, on="compound_id", how="left")
                        .merge(other_smiles, on="compound_id", how="left")
                    )
                    multi_fragment = multi_fragment.merge(
                        relationships, on="compound_id", how="left"
                    )
                    st.markdown("#### Multi-fragment source compounds")
                    st.dataframe(
                        multi_fragment[
                            [
                                *_compound_table_columns(multi_fragment),
                                *[
                                    column
                                    for column in (
                                        "recognized_formulation_components",
                                        "formulation_category",
                                        "parent_candidate_smiles",
                                        "other_fragment_smiles",
                                        "all_fragment_smiles",
                                        "parent_match_status",
                                        "unformulated_library_matches",
                                        "related_formulations",
                                    )
                                    if column in multi_fragment.columns
                                ],
                            ]
                        ],
                        hide_index=True,
                        width="stretch",
                        height=min(
                            430,
                            38 + 35 * min(len(multi_fragment), 11),
                        ),
                    )
            st.markdown("#### Individual disconnected components")
            st.caption(
                "Contains carbon is a chemical observation only. It does not identify "
                "the active parent: citrate, acetate, trifluoroacetate, piperazine and "
                "other formulation partners also contain carbon. Interpretations below "
                "are review aids and do not change or strip any structure."
            )
            if component_context is None:
                component_context = _component_context_table(
                    components, normalized
                )
            component_metrics = st.columns(3)
            component_metrics[0].metric(
                "Recognized components",
                int(
                    component_context["recognized_component"]
                    .astype(bool)
                    .sum()
                ),
            )
            component_metrics[1].metric(
                "Compounds with unsalted match",
                component_context.loc[
                    component_context["unformulated_library_matches"]
                    .astype(bool),
                    "compound_id",
                ].nunique(),
            )
            component_metrics[2].metric(
                "Ambiguous parent compounds",
                component_context.loc[
                    component_context["parent_match_status"].eq(
                        "ambiguous largest components"
                    ),
                    "compound_id",
                ].nunique(),
            )
            st.dataframe(
                component_context,
                hide_index=True,
                width="stretch",
                height=500,
            )

    with docking_parents_tab:
        st.info(
            "One row represents one unique stereochemistry-aware parent "
            "structure. Salt and formulation records are consolidated, while "
            "all source IDs, names, CAS numbers, and origins remain attached. "
            "These identities are ready for downstream ligand preparation; "
            "3D generation and protonation still occur before docking."
        )
        parent_metrics = st.columns(3)
        parent_metrics[0].metric(
            "Unique docking parents",
            duplicate_summary["unique_parent_count"],
        )
        parent_metrics[1].metric(
            "Consolidated source records",
            duplicate_summary["redundant_entry_count"],
        )
        parent_metrics[2].metric(
            "Ambiguous / excluded",
            duplicate_summary["ambiguous_parent_count"],
        )
        if docking_parent_frame.empty:
            st.info("No unambiguous docking parents are available.")
        else:
            parent_search = st.text_input(
                "Search docking-ready parents",
                key=f"docking_parent_search_{job.run_id}",
                placeholder="Parent ID, compound ID, product, CAS, or SMILES",
            )
            displayed_parents = docking_parent_frame
            if parent_search.strip():
                parent_mask = docking_parent_frame.astype(str).apply(
                    lambda column: column.str.contains(
                        parent_search.strip(),
                        case=False,
                        regex=False,
                    )
                ).any(axis=1)
                displayed_parents = docking_parent_frame.loc[
                    parent_mask
                ].reset_index(drop=True)
            st.caption(
                f"Showing {len(displayed_parents):,} of "
                f"{len(docking_parent_frame):,} unique parents."
            )
            st.dataframe(
                displayed_parents,
                hide_index=True,
                width="stretch",
                height=min(
                    560,
                    38 + 35 * min(len(displayed_parents), 15),
                ),
            )
            st.download_button(
                "Download docking-ready parent table",
                data=docking_parent_frame.to_csv(index=False).encode("utf-8"),
                file_name=f"{job_code}_docking_ready_parents.csv",
                mime="text/csv",
                key=f"download_docking_parents_{job.run_id}",
            )

    with duplicates_tab:
        st.info(
            "This is a comparison-only docking-parent analysis. Original "
            "SMILES and imported artifacts are unchanged. A unique largest "
            "component is selected, then removable charge is neutralized only "
            "for the duplicate key. Stereochemistry is preserved and "
            "tautomers are not merged."
        )
        duplicate_metrics = st.columns(4)
        duplicate_metrics[0].metric(
            "Duplicate groups",
            duplicate_summary["duplicate_group_count"],
        )
        duplicate_metrics[1].metric(
            "Entries in groups",
            duplicate_summary["duplicate_entry_count"],
        )
        duplicate_metrics[2].metric(
            "Redundant entries",
            duplicate_summary["redundant_entry_count"],
        )
        duplicate_metrics[3].metric(
            "Ambiguous parents",
            duplicate_summary["ambiguous_parent_count"],
        )
        if parent_duplicate_frame.empty:
            st.success("No parent-equivalent duplicate groups were found.")
        else:
            duplicate_search = st.text_input(
                "Search parent duplicates",
                key=f"parent_duplicate_search_{job.run_id}",
                placeholder="Group, compound ID, product, CAS, or SMILES",
            )
            displayed_duplicates = parent_duplicate_frame
            if duplicate_search.strip():
                duplicate_mask = parent_duplicate_frame.astype(str).apply(
                    lambda column: column.str.contains(
                        duplicate_search.strip(),
                        case=False,
                        regex=False,
                    )
                ).any(axis=1)
                displayed_duplicates = parent_duplicate_frame.loc[
                    duplicate_mask
                ].reset_index(drop=True)
            duplicate_columns = [
                column
                for column in (
                    "duplicate_group",
                    "group_size",
                    "match_type",
                    "compound_id",
                    "structure_origin",
                    "review_job",
                    "Product Name",
                    "Synonyms",
                    "CAS Number",
                    "formula",
                    "source_fragment_count",
                    "parent_status",
                    "parent_occurrences",
                    "parent_formula",
                    "standardized_parent_formula",
                    "parent_formal_charge",
                    "docking_parent_smiles",
                    "standardized_parent_smiles",
                    "smiles",
                    "validation_warning",
                )
                if column in displayed_duplicates.columns
            ]
            st.dataframe(
                displayed_duplicates[duplicate_columns],
                hide_index=True,
                width="stretch",
                height=min(
                    560,
                    38 + 35 * min(len(displayed_duplicates), 15),
                ),
            )
            st.download_button(
                "Download parent duplicate report",
                data=parent_duplicate_frame.to_csv(index=False).encode(
                    "utf-8"
                ),
                file_name=f"{job_code}_parent_duplicates.csv",
                mime="text/csv",
                key=f"download_parent_duplicates_{job.run_id}",
            )
        ambiguous_parent_frame = pd.DataFrame(
            [
                row
                for row in duplicate_analysis["all_parent_rows"]
                if str(row.get("parent_status") or "").startswith(
                    "ambiguous"
                )
            ]
        )
        if not ambiguous_parent_frame.empty:
            with st.expander(
                "Ambiguous multi-component parents requiring review"
            ):
                st.dataframe(
                    ambiguous_parent_frame,
                    hide_index=True,
                    width="stretch",
                )

    with rejected_tab:
        if rejected_path is None:
            st.success("No rows were rejected.")
        else:
            rejected = rejected_source.copy()
            rejected.insert(
                2,
                "review_decision",
                [
                    (
                        str(
                            review_map[str(compound_id)].metadata.get(
                                "decision"
                            )
                            or ""
                        )
                        if str(compound_id) in review_map
                        else ""
                    )
                    for compound_id in rejected["compound_id"]
                ],
            )
            rejected.insert(
                3,
                "review_job",
                [
                    (
                        display_job_code(
                            review_map[str(compound_id)].metadata.get(
                                "job_code"
                            ),
                            review_map[str(compound_id)].run_id,
                        )
                        if str(compound_id) in review_map
                        else ""
                    )
                    for compound_id in rejected["compound_id"]
                ],
            )
            rejected.insert(
                4,
                "review_origin",
                [
                    (
                        str(
                            review_map[str(compound_id)].metadata.get(
                                "structure_origin"
                            )
                            or ""
                        )
                        if str(compound_id) in review_map
                        else ""
                    )
                    for compound_id in rejected["compound_id"]
                ],
            )
            st.warning(
                "Rejected rows remain unchanged in the original workbook. "
                "Annotation-stripped structures are review candidates only."
            )
            rejected_view = st.radio(
                "Rejected-row view",
                ["Needs review", "Review history"],
                horizontal=True,
                key=f"rejected_view_{job.run_id}",
            )
            if rejected_view == "Needs review":
                displayed_rejected = rejected.loc[
                    rejected["review_decision"].eq("")
                ].reset_index(drop=True)
                st.caption(
                    "Accepted and skipped rows leave this active queue. "
                    "Use Review history to inspect their immutable source records."
                )
            else:
                displayed_rejected = rejected.loc[
                    rejected["review_decision"].ne("")
                ].reset_index(drop=True)
                st.caption(
                    "Resolved vendor rows are retained here for provenance."
                )
            selection_token = hashlib.sha1(
                "\x1f".join(
                    displayed_rejected.get(
                        "compound_id", pd.Series(dtype=str)
                    ).astype(str)
                ).encode("utf-8")
            ).hexdigest()[:10]
            event = st.dataframe(
                displayed_rejected,
                hide_index=True,
                width="stretch",
                height=min(
                    500, 38 + 35 * min(len(displayed_rejected), 13)
                ),
                on_select="rerun",
                selection_mode="single-row",
                key=(
                    f"rejected_compound_table_{job.run_id}_"
                    f"{rejected_view}_{selection_token}"
                ),
            )
            selected_rows = [
                int(index)
                for index in (
                    getattr(
                        getattr(event, "selection", None), "rows", []
                    )
                    or []
                )
                if 0 <= int(index) < len(displayed_rejected)
            ]
            if not selected_rows:
                if displayed_rejected.empty:
                    st.success(
                        "There are no compounds in this rejected-row view."
                    )
                else:
                    st.info(
                        "Select one rejected compound to search PubChem, "
                        "confirm a candidate, or skip it."
                    )
            else:
                selected = displayed_rejected.iloc[
                    int(selected_rows[0])
                ].to_dict()
                compound_id = str(selected.get("compound_id") or "")
                product_name = str(selected.get("Product Name") or "").strip()
                cas_number = str(selected.get("CAS Number") or "").strip()
                st.markdown(f"#### Review {compound_id}")
                identity_columns = st.columns(4)
                identity_columns[0].metric(
                    "Product", product_name or "—"
                )
                identity_columns[1].metric("CAS", cas_number or "—")
                identity_columns[2].metric(
                    "Source formula",
                    str(selected.get("Formula") or "—"),
                )
                identity_columns[3].metric(
                    "Current decision",
                    str(selected.get("review_decision") or "unreviewed"),
                )

                search_state_key = (
                    f"pubchem_candidates_{job.run_id}_{compound_id}"
                )
                search_columns = st.columns(3)
                if search_columns[0].button(
                    "Search PubChem by CAS",
                    disabled=not bool(cas_number),
                    key=f"pubchem_cas_{job.run_id}_{compound_id}",
                ):
                    try:
                        with st.spinner("Searching PubChem by CAS..."):
                            st.session_state[search_state_key] = (
                                search_pubchem_candidates(
                                    cas_number, query_type="cas"
                                )
                            )
                    except ValueError as exc:
                        st.error(str(exc))
                if search_columns[1].button(
                    "Search PubChem by name",
                    disabled=not bool(product_name),
                    key=f"pubchem_name_{job.run_id}_{compound_id}",
                ):
                    try:
                        with st.spinner("Searching PubChem by product name..."):
                            st.session_state[search_state_key] = (
                                search_pubchem_candidates(
                                    product_name, query_type="name"
                                )
                            )
                    except ValueError as exc:
                        st.error(str(exc))
                if search_columns[2].button(
                    "Skip / keep rejected",
                    key=f"pubchem_skip_{job.run_id}_{compound_id}",
                ):
                    try:
                        review_job = create_compound_review_job(
                            job,
                            source_row=int(selected.get("source_row") or 0),
                            compound_id=compound_id,
                            decision="rejected",
                        )
                    except (OSError, ValueError) as exc:
                        st.error(f"Could not record skipped review: {exc}")
                    else:
                        st.toast(
                            "Recorded skip decision as review job "
                            f"{display_job_code(review_job.metadata.get('job_code'), review_job.run_id)}."
                        )
                        st.rerun()

                candidates = list(
                    st.session_state.get(search_state_key) or []
                )
                if search_state_key in st.session_state and not candidates:
                    st.info("PubChem returned no candidates for this query.")
                if candidates:
                    st.caption(
                        f"PubChem returned {len(candidates)} candidate(s). "
                        "Name searches include the full vendor name and, when "
                        "present, the base name without a trailing formulation "
                        "qualifier. Confirming one records a typed review; it "
                        "does not overwrite the source workbook."
                    )
                    candidate_index = st.selectbox(
                        "PubChem candidate",
                        range(len(candidates)),
                        format_func=lambda index: (
                            f"CID {candidates[index].get('cid')} · "
                            f"{candidates[index].get('molecular_formula') or 'formula unavailable'} · "
                            f"{candidates[index].get('inchi_key') or 'InChIKey unavailable'} · "
                            f"{candidates[index].get('match_scope') or 'identifier match'}"
                        ),
                        key=f"pubchem_candidate_{job.run_id}_{compound_id}",
                    )
                    candidate = candidates[int(candidate_index)]
                    candidate_analysis = normalize_compound_smiles(
                        str(candidate.get("smiles") or "")
                    )
                    formula_comparison = compare_molecular_formulas(
                        str(selected.get("Formula") or ""),
                        str(candidate.get("molecular_formula") or ""),
                    )
                    candidate = {
                        **candidate,
                        "vendor_formula": str(
                            selected.get("Formula") or ""
                        ),
                        "formula_comparison": formula_comparison,
                        "fragment_count": candidate_analysis[
                            "fragment_count"
                        ],
                    }
                    component_options = pubchem_component_options(
                        str(candidate.get("smiles") or "")
                    )
                    if len(component_options) > 1:
                        component_index = st.selectbox(
                            "Single component to import",
                            range(len(component_options)),
                            format_func=lambda index: (
                                f"{component_options[index]['formula']} · "
                                f"{component_options[index]['heavy_atoms']} "
                                "heavy atoms · "
                                f"{component_options[index]['occurrences']}× "
                                "in PubChem record"
                                + (
                                    " · automatic parent candidate"
                                    if component_options[index].get(
                                        "automatic_parent_candidate"
                                    )
                                    else ""
                                )
                            ),
                            key=(
                                f"pubchem_component_{job.run_id}_"
                                f"{compound_id}_{candidate.get('cid')}"
                            ),
                        )
                    else:
                        component_index = 0
                    selected_component = component_options[
                        int(component_index)
                    ]
                    review_candidate = selected_parent_candidate(
                        candidate,
                        selected_component,
                    )
                    review_candidate.update(
                        {
                            "vendor_formula": str(
                                selected.get("Formula") or ""
                            ),
                            "formula_comparison": formula_comparison,
                        }
                    )
                    st.dataframe(
                        pd.DataFrame([candidate]),
                        hide_index=True,
                        width="stretch",
                    )
                    molecule = Chem.MolFromSmiles(
                        str(candidate.get("smiles") or "")
                    )
                    (
                        source_structure,
                        candidate_structure,
                        parent_structure,
                        candidate_action,
                    ) = (
                        st.columns([1, 1, 1, 2])
                    )
                    source_preview_smiles = str(
                        selected.get("annotation_stripped_candidate")
                        or annotation_stripped_smiles_candidate(
                            str(selected.get("smiles") or "")
                        )
                        or selected.get("smiles")
                        or ""
                    )
                    source_molecule = Chem.MolFromSmiles(
                        source_preview_smiles
                    )
                    with source_structure:
                        st.markdown("#### Vendor-side structure")
                        if source_molecule is not None:
                            st.image(
                                Draw.MolToImage(
                                    source_molecule, size=(500, 340)
                                ),
                                caption=(
                                    "Review-only annotation-stripped candidate"
                                    if selected.get(
                                        "annotation_stripped_candidate"
                                    )
                                    else "Vendor SMILES"
                                ),
                                width="stretch",
                            )
                        else:
                            st.info(
                                "The vendor SMILES is not renderable and no "
                                "annotation-stripped candidate is available."
                            )
                    with candidate_structure:
                        st.markdown("#### Full PubChem record")
                        if molecule is not None:
                            st.image(
                                Draw.MolToImage(molecule, size=(500, 340)),
                                caption=f"PubChem CID {candidate.get('cid')}",
                                width="stretch",
                            )
                    with parent_structure:
                        st.markdown("#### Parent to import")
                        parent_molecule = Chem.MolFromSmiles(
                            str(selected_component.get("smiles") or "")
                        )
                        if parent_molecule is not None:
                            st.image(
                                Draw.MolToImage(
                                    parent_molecule, size=(500, 340)
                                ),
                                caption=(
                                    f"{selected_component.get('formula')} · "
                                    "single component"
                                ),
                                width="stretch",
                            )
                    with candidate_action:
                        st.markdown("#### Candidate comparison")
                        st.warning(
                            "Confirmation imports only the selected single "
                            "component. The complete PubChem formulation and "
                            "vendor SMILES remain immutable provenance."
                        )
                        if formula_comparison["compatible"]:
                            st.success(formula_comparison["message"])
                        else:
                            st.warning(formula_comparison["message"])
                        if int(candidate_analysis["fragment_count"]) > 1:
                            st.info(
                                "PubChem returned a multi-fragment formulation "
                                f"with {candidate_analysis['fragment_count']} "
                                "disconnected components. Repeated parent "
                                "molecules and formulation partners are not "
                                "placed in the usable compound structure."
                            )
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "Field": "Vendor formula",
                                        "Value": str(
                                            selected.get("Formula") or ""
                                        ),
                                    },
                                    {
                                        "Field": "PubChem formula",
                                        "Value": str(
                                            candidate.get(
                                                "molecular_formula"
                                            )
                                            or ""
                                        ),
                                    },
                                    {
                                        "Field": "Formula relationship",
                                        "Value": str(
                                            formula_comparison["status"]
                                        ),
                                    },
                                    {
                                        "Field": "PubChem fragments",
                                        "Value": str(
                                            candidate_analysis[
                                                "fragment_count"
                                            ]
                                        ),
                                    },
                                    {
                                        "Field": "Selected parent formula",
                                        "Value": str(
                                            selected_component.get("formula")
                                            or ""
                                        ),
                                    },
                                    {
                                        "Field": "Parent occurrences in record",
                                        "Value": str(
                                            selected_component.get(
                                                "occurrences"
                                            )
                                            or 1
                                        ),
                                    },
                                    {
                                        "Field": "Selected parent SMILES",
                                        "Value": str(
                                            selected_component.get("smiles")
                                            or ""
                                        ),
                                    },
                                    {
                                        "Field": "Vendor SMILES",
                                        "Value": str(
                                            selected.get("smiles") or ""
                                        ),
                                    },
                                    {
                                        "Field": "PubChem SMILES",
                                        "Value": str(
                                            candidate.get("smiles") or ""
                                        ),
                                    },
                                ]
                            ),
                            hide_index=True,
                            width="stretch",
                        )
                        if candidate.get("pubchem_url"):
                            st.link_button(
                                "Open candidate in PubChem",
                                str(candidate["pubchem_url"]),
                            )
                        if st.button(
                            "Confirm selected parent compound",
                            type="primary",
                            key=(
                                f"pubchem_confirm_{job.run_id}_"
                                f"{compound_id}_{candidate.get('cid')}"
                            ),
                        ):
                            try:
                                review_job = create_compound_review_job(
                                    job,
                                    source_row=int(
                                        selected.get("source_row") or 0
                                    ),
                                    compound_id=compound_id,
                                    decision="accepted",
                                    query=str(candidate.get("query") or ""),
                                    query_type=str(
                                        candidate.get("query_type") or ""
                                    ),
                                    candidate=review_candidate,
                                )
                            except (OSError, ValueError) as exc:
                                st.error(
                                    f"Could not record PubChem review: {exc}"
                                )
                            else:
                                st.toast(
                                    "Confirmed candidate as review job "
                                    f"{display_job_code(review_job.metadata.get('job_code'), review_job.run_id)}."
                                )
                                st.rerun()

    with profiles_tab:
        numeric_columns = list(validation.get("numeric_columns") or [])
        if numeric_columns:
            st.markdown("#### Source numeric columns")
            st.dataframe(
                pd.DataFrame(numeric_columns),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No numeric source columns were profiled.")
        if summary:
            st.markdown("#### Validation summary")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Metric": key, "Value": value}
                        for key, value in summary.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
