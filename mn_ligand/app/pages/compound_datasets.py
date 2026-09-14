from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import display_job_code
from mn_ligand.workflows.compound_preparation import (
    TABULAR_COMPOUND_SUFFIXES,
    analyze_tabular_compound_dataset,
    compound_dataset_columns,
    compound_dataset_sheets,
    create_compound_import_job,
    list_compound_import_jobs,
)


def _job_url(job) -> str:
    code = display_job_code(job.metadata.get("job_code"), job.run_id)
    query = urlencode(
        {"task_group": job.task_group, "run_id": job.run_id, "label": code}
    )
    return f"./job-results?{query}"


def _suggest_column(columns: tuple[str, ...], candidates: tuple[str, ...]) -> int:
    lowered = {column.lower(): index for index, column in enumerate(columns)}
    return next((lowered[name] for name in candidates if name in lowered), -1)


@st.cache_data(show_spinner=False)
def _analyze_uploaded(
    data: bytes,
    filename: str,
    smiles_column: str,
    id_column: str,
    sheet_name: str,
) -> dict:
    return analyze_tabular_compound_dataset(
        data,
        filename,
        smiles_column=smiles_column,
        id_column=id_column,
        sheet_name=sheet_name,
    )


def render() -> None:
    st.title("Compound Datasets")
    st.caption(
        "Import reusable compound libraries, map tabular identifiers and SMILES, "
        "validate chemistry with RDKit, and inspect normalized structures and descriptors."
    )
    input_tab, columns_tab, validation_tab, run_tab, results_tab = st.tabs(
        ["Input", "Columns", "Validation", "Run", "Results"]
    )

    with input_tab:
        name_column, file_column = st.columns([1, 2], vertical_alignment="bottom")
        dataset_name = name_column.text_input(
            "Dataset name", key="compound_dataset_name"
        )
        dataset_file = file_column.file_uploader(
            "Compound dataset",
            type=["xlsx", "xlsm", "csv", "sdf", "smi", "smiles", "txt"],
            key="compound_dataset_file",
        )
        st.caption(
            "Excel workbooks are preserved in full, including every worksheet and "
            "column. Column mapping only controls the normalized downstream compound set."
        )

    file_data = dataset_file.getvalue() if dataset_file is not None else b""
    filename = dataset_file.name if dataset_file is not None else ""
    suffix = Path(filename).suffix.lower() if filename else ""
    is_tabular = suffix in TABULAR_COMPOUND_SUFFIXES
    sheet_name = ""
    id_column = ""
    smiles_column = ""
    analysis: dict | None = None
    analysis_error = ""

    with columns_tab:
        if dataset_file is None:
            st.info("Upload a compound dataset in the Input tab.")
        elif not is_tabular:
            st.info(
                "Column mapping applies to CSV and Excel datasets. SDF and SMILES "
                "files use their native identifiers and structures."
            )
        else:
            sheets = compound_dataset_sheets(file_data, filename)
            if sheets:
                sheet_name = st.selectbox(
                    "Worksheet containing compounds",
                    sheets,
                    key="compound_dataset_sheet",
                )
                st.caption(
                    f"The complete workbook and all {len(sheets)} worksheet(s) will be retained."
                )
            columns = compound_dataset_columns(
                file_data,
                filename,
                sheet_name=sheet_name,
            )
            if columns:
                mapping_columns = st.columns(2)
                smiles_column = mapping_columns[0].selectbox(
                    "SMILES column",
                    columns,
                    index=max(
                        0,
                        _suggest_column(
                            columns,
                            (
                                "smiles",
                                "canonical_smiles",
                                "isomeric_smiles",
                                "structure",
                            ),
                        ),
                    ),
                    key="compound_dataset_smiles_column",
                )
                id_options = ("Generate IDs", *columns)
                suggested_id = _suggest_column(
                    columns,
                    (
                        "compound_id",
                        "id",
                        "name",
                        "zincid",
                        "catalog_id",
                        "catalog number",
                        "catalog_number",
                    ),
                )
                id_selection = mapping_columns[1].selectbox(
                    "Compound ID column",
                    id_options,
                    index=suggested_id + 1 if suggested_id >= 0 else 0,
                    key="compound_dataset_id_column",
                )
                id_column = "" if id_selection == "Generate IDs" else id_selection
                st.dataframe(
                    pd.DataFrame({"Column": columns}),
                    hide_index=True,
                    width="stretch",
                )
                try:
                    analysis = _analyze_uploaded(
                        file_data,
                        filename,
                        smiles_column,
                        id_column,
                        sheet_name,
                    )
                except Exception as exc:
                    analysis_error = str(exc)
                    st.error(f"Dataset analysis failed: {exc}")

    with validation_tab:
        if dataset_file is None:
            st.info("Upload a compound dataset in the Input tab.")
        elif not is_tabular:
            st.info(
                "Detailed RDKit row validation and descriptor inspection currently "
                "apply to CSV and Excel imports."
            )
        elif analysis_error:
            st.error(analysis_error)
        elif analysis is not None:
            summary = analysis["summary"]
            metrics = st.columns(6)
            metrics[0].metric("Source rows", summary["source_row_count"])
            metrics[1].metric("Usable", summary["valid_count"])
            metrics[2].metric("Rejected", summary["invalid_count"])
            metrics[3].metric(
                "Duplicate structures", summary["duplicate_structure_count"]
            )
            metrics[4].metric("Charged", summary["charged_count"])
            metrics[5].metric("Multi-fragment", summary["multi_fragment_count"])
            if summary["multi_fragment_count"]:
                st.info(
                    "Multi-component formulations are retained exactly as supplied. "
                    "Components are reported separately; no salt, solvate, counterion, "
                    "or other fragment is removed automatically."
                )
                with st.expander(
                    "Multi-component structures and components", expanded=False
                ):
                    st.dataframe(
                        pd.DataFrame(analysis["component_rows"]),
                        hide_index=True,
                        width="stretch",
                        height=430,
                    )
            st.markdown("#### RDKit-validated compounds")
            st.dataframe(
                pd.DataFrame(analysis["valid_rows"]).head(500),
                hide_index=True,
                width="stretch",
                height=430,
            )
            if analysis["invalid_rows"]:
                st.markdown("#### Rejected rows")
                st.dataframe(
                    pd.DataFrame(analysis["invalid_rows"]).head(500),
                    hide_index=True,
                    width="stretch",
                    height=min(430, 38 + 35 * len(analysis["invalid_rows"])),
                )
                st.warning(
                    "Rejected rows remain in the original workbook and are recorded "
                    "in a separate report, but are excluded from downstream workflows. "
                    "Any annotation-stripped SMILES shown is a review candidate only "
                    "and is not substituted automatically."
                )
            else:
                st.success("Every mapped SMILES passed RDKit parsing and sanitization.")
            if analysis["numeric_columns"]:
                st.markdown("#### Original numeric-column profile")
                st.dataframe(
                    pd.DataFrame(analysis["numeric_columns"]),
                    hide_index=True,
                    width="stretch",
                )

    with run_tab:
        render_run_resources(
            requires_gpu=False,
            selected_gpu="Not used",
            key="compound_dataset_import",
        )
        if dataset_file is None:
            st.info("Upload a compound dataset in the Input tab.")
        elif is_tabular and analysis_error:
            st.error("Resolve the column mapping or invalid dataset before importing.")
        elif is_tabular and analysis is not None:
            st.write(
                f"{analysis['summary']['valid_count']} usable compounds will be registered; "
                f"{analysis['summary']['invalid_count']} rejected rows will be retained "
                "only in the source and validation report."
            )
        if st.button(
            "Register validated compound dataset",
            type="primary",
            disabled=(
                dataset_file is None
                or bool(analysis_error)
                or (is_tabular and analysis is None)
            ),
            key="compound_dataset_register",
        ):
            try:
                job = create_compound_import_job(
                    file_data,
                    filename=filename,
                    dataset_name=dataset_name,
                    id_column=id_column,
                    smiles_column=smiles_column,
                    sheet_name=sheet_name,
                )
                st.success(
                    f"Registered {job.metadata.get('compound_count')} usable compounds as "
                    f"{display_job_code(job.metadata.get('job_code'), job.run_id)}."
                )
            except Exception as exc:
                st.error(f"Compound dataset import failed: {exc}")

    with results_tab:
        st.link_button(
            "Compare target-campaign results",
            "./compound-campaign-comparison",
            help=(
                "Compare docking, cofolding and rescoring campaigns for this "
                "compound dataset by target and engine."
            ),
        )
        jobs = list_compound_import_jobs()
        if not jobs:
            st.info("No compound datasets imported yet.")
        else:
            rows = [
                {
                    "Job": _job_url(job),
                    "Compare": (
                        "./compound-campaign-comparison?"
                        + urlencode({"dataset_run_id": job.run_id})
                    ),
                    "Dataset": job.metadata.get("dataset_name"),
                    "Format": job.metadata.get("compound_format"),
                    "Usable": job.metadata.get("compound_count"),
                    "Rejected": job.metadata.get("invalid_compound_count", 0),
                    "Duplicates": job.metadata.get("duplicate_structure_count", 0),
                    "Created": job.created_at,
                    "_run_id": job.run_id,
                }
                for job in jobs
            ]
            st.dataframe(
                pd.DataFrame(rows).drop(columns=["_run_id"]),
                hide_index=True,
                width="stretch",
                key="compound_dataset_results",
                column_config={
                    "Job": st.column_config.LinkColumn(
                        "Job",
                        display_text=r"label=([^&]+)",
                    ),
                    "Compare": st.column_config.LinkColumn(
                        "Campaigns", display_text="Compare"
                    ),
                    "Created": st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD HH:mm"
                    ),
                },
            )
            st.caption(
                "Open a job ID to inspect compounds, components and salts, "
                "rejected rows, profiles, provenance, and downloads."
            )


render()
