from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mn_ligand.core.jobs import display_job_code
from mn_ligand.workflows.complex_datasets import (
    complex_dataset_jobs,
    complex_dataset_sheets,
    create_complex_dataset,
    read_complex_dataset,
    selected_complex_jobs,
    validate_complex_dataset_rows,
)


def _dataset_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Dataset": str(job.metadata.get("dataset_name") or ""),
            "Job": display_job_code(job.metadata.get("job_code"), job.run_id),
            "Complexes": int(job.metadata.get("complex_count") or 0),
            "Source file": str(job.metadata.get("source_filename") or ""),
            "Worksheet": str(job.metadata.get("source_sheet") or ""),
            "Created": job.created_at,
            "Run ID": job.run_id,
        }
        for job in complex_dataset_jobs()
    ])


def _complex_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Dataset": str(
                job.metadata.get("complex_dataset_name") or ""
            ),
            "Rank": job.metadata.get("selected_rank", ""),
            "Compound": str(job.metadata.get("compound_id") or ""),
            "Compound name": str(job.metadata.get("compound_name") or ""),
            "Selection status": str(
                job.metadata.get("selection_status") or ""
            ),
            "Prediction engine": str(job.metadata.get("engine") or ""),
            "Source job": str(
                job.metadata.get("source_prediction_job_code") or ""
            ),
            "Replicate": job.metadata.get("replicate", ""),
            "Prediction": str(job.metadata.get("prediction") or ""),
            "GNINA pose selection": str(
                job.metadata.get("gnina_pose_selection") or ""
            ),
            "Selected-complex job": display_job_code(
                job.metadata.get("job_code"), job.run_id
            ),
            "Run ID": job.run_id,
        }
        for job in selected_complex_jobs()
    ])


def render() -> None:
    st.title("Complex Datasets")
    st.caption(
        "Import a curated list of docking or cofolding poses for downstream "
        "MD. The interaction-analysis path identifies the reviewed pose, but "
        "the MD coordinates are always recovered from the immutable source "
        "prediction. Every selection retains its prediction, interaction "
        "analysis, pose, target, campaign, and spreadsheet evidence."
    )
    import_tab, datasets_tab = st.tabs(["Import selection", "Datasets"])
    with import_tab:
        upload = st.file_uploader(
            "Complex-selection workbook",
            type=["xlsx", "xlsm"],
            help=(
                "Required identity columns are Compound and Source job. The pose "
                "may be supplied by Interaction evidence path, or by Source pose "
                "path/MD complex path together with Replicate and Prediction "
                "('pose N'). A multi-pose docking SDF is treated as a container; "
                "Prediction selects its exact record. Older workbooks using "
                "Predicted complex path remain supported. Additional scientific "
                "selection columns are preserved."
            ),
        )
        if upload is not None:
            source_bytes = upload.getvalue()
            try:
                sheets = complex_dataset_sheets(source_bytes)
            except (ImportError, OSError, ValueError) as exc:
                st.error(f"Could not read workbook: {exc}")
                sheets = ()
            if sheets:
                sheet_name = st.selectbox("Worksheet", sheets)
                dataset_name = st.text_input(
                    "Complex dataset name",
                    value=Path(upload.name).stem,
                ).strip()
                signature = (
                    upload.name,
                    len(source_bytes),
                    sheet_name,
                    hash(source_bytes),
                )
                state_key = "complex_dataset_validation"
                if st.button(
                    "Validate selection provenance",
                    type="primary",
                ):
                    try:
                        with st.spinner(
                            "Resolving source jobs, poses, and complex files..."
                        ):
                            source = read_complex_dataset(
                                source_bytes, sheet_name
                            )
                            validated = validate_complex_dataset_rows(source)
                    except (OSError, TypeError, ValueError) as exc:
                        st.error(f"Could not validate complex dataset: {exc}")
                    else:
                        st.session_state[state_key] = {
                            "signature": signature,
                            "frame": validated,
                        }
                payload = st.session_state.get(state_key)
                if (
                    isinstance(payload, dict)
                    and payload.get("signature") == signature
                    and isinstance(payload.get("frame"), pd.DataFrame)
                ):
                    validated = payload["frame"]
                    valid_count = int(validated["Validation"].eq("Valid").sum())
                    invalid_count = len(validated) - valid_count
                    metrics = st.columns(3)
                    metrics[0].metric("Rows", len(validated))
                    metrics[1].metric("Valid", valid_count)
                    metrics[2].metric("Needs attention", invalid_count)
                    st.caption(
                        "Use Import to decide which valid rows enter this "
                        "dataset. Invalid rows cannot be imported."
                    )
                    editor = validated.copy()
                    editor.loc[
                        editor["Validation"].ne("Valid"), "Import"
                    ] = False
                    edited = st.data_editor(
                        editor,
                        hide_index=True,
                        width="stretch",
                        disabled=[
                            column for column in editor.columns
                            if column != "Import"
                        ],
                        column_config={
                            "Import": st.column_config.CheckboxColumn(
                                "Import", default=True
                            ),
                            "Predicted complex path": (
                                st.column_config.TextColumn(width="large")
                            ),
                            "Interaction evidence path": (
                                st.column_config.TextColumn(width="large")
                            ),
                            "Validation": st.column_config.TextColumn(
                                width="large"
                            ),
                            "Resolved source run ID": (
                                st.column_config.TextColumn(width="large")
                            ),
                            "Resolved analysis run ID": (
                                st.column_config.TextColumn(width="large")
                            ),
                            "Resolved source pose path": (
                                st.column_config.TextColumn(width="large")
                            ),
                            "Resolved source receptor path": (
                                st.column_config.TextColumn(width="large")
                            ),
                            "Analysis evidence path": (
                                st.column_config.TextColumn(width="large")
                            ),
                        },
                        key="complex_dataset_import_editor",
                    )
                    selected = edited.loc[
                        edited["Import"].astype(bool)
                        & edited["Validation"].eq("Valid")
                    ].copy()
                    st.caption(
                        f"{len(selected)} valid complex(es) selected for import."
                    )
                    if st.button(
                        "Create immutable complex dataset",
                        type="primary",
                        disabled=not dataset_name or selected.empty,
                    ):
                        try:
                            with st.spinner(
                                "Recovering source poses and writing lineage..."
                            ):
                                dataset = create_complex_dataset(
                                    name=dataset_name,
                                    source_filename=upload.name,
                                    source_bytes=source_bytes,
                                    sheet_name=sheet_name,
                                    selected_rows=selected,
                                )
                        except (OSError, TypeError, ValueError) as exc:
                            st.error(f"Could not create complex dataset: {exc}")
                        else:
                            st.success(
                                f"Created {dataset_name} with "
                                f"{dataset.metadata.get('complex_count')} "
                                "MD-selectable complexes."
                            )
                            st.cache_data.clear()
    with datasets_tab:
        datasets = _dataset_rows()
        if datasets.empty:
            st.info("No curated complex datasets have been imported yet.")
        else:
            st.markdown("#### Imported complex datasets")
            st.dataframe(
                datasets,
                hide_index=True,
                width="stretch",
                column_config={
                    "Created": st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD HH:mm"
                    ),
                    "Run ID": st.column_config.TextColumn(width="large"),
                },
            )
            st.markdown("#### MD-selectable complexes")
            st.dataframe(
                _complex_rows(),
                hide_index=True,
                width="stretch",
                column_config={
                    "Selection status": st.column_config.TextColumn(
                        width="large"
                    ),
                    "Run ID": st.column_config.TextColumn(width="large"),
                },
            )


render()
