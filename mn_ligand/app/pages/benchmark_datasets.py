from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.app.pages.benchmark_common import campaign_table, visible_dataset_jobs
from mn_ligand.core.jobs import display_job_code
from mn_ligand.workflows.benchmark_datasets import (
    analyze_benchmark_archive,
    create_benchmark_dataset_job,
    load_benchmark_dataset,
)


PROFILES = (
    "Auto-detect",
    "PoseBench — PoseBusters Benchmark",
    "PoseBench — Astex Diverse",
    "PoseBench — DockGen",
    "PoseBench — CASP15",
    "Generic manifest/archive",
)


@st.cache_data(show_spinner=False)
def _preview(
    archive_data: bytes,
    archive_filename: str,
    manifest_data: bytes | None,
    manifest_filename: str,
    profile: str,
) -> dict:
    return analyze_benchmark_archive(
        archive_data,
        archive_filename,
        manifest_data=manifest_data,
        manifest_filename=manifest_filename,
        requested_profile=profile,
    )


def _result_url(job) -> str:
    return "./benchmark-results?" + urlencode({"dataset_run_id": job.run_id})


def _dataset_url(job) -> str:
    code = display_job_code(job.metadata.get("job_code"), job.run_id)
    return (
        "./benchmark-dataset-results?"
        + urlencode({"dataset_run_id": job.run_id})
        + f"#{code}"
    )


def _render_results() -> None:
    jobs = visible_dataset_jobs()
    if not jobs:
        st.info("No benchmark datasets have been imported.")
        return
    rows = []
    for job in jobs:
        dataset = load_benchmark_dataset(job)
        campaigns = campaign_table(job.run_id)
        rows.append(
            {
                "Job": _dataset_url(job),
                "Dataset": dataset["dataset_name"],
                "Profile": dataset["profile"],
                "Cases": dataset["case_count"],
                "Rejected": dataset.get("rejected_case_count", 0),
                "Redocking / refolding jobs": len(campaigns),
                "Combined results": _result_url(job),
                "Created": job.created_at,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Job": st.column_config.LinkColumn(
                "Job",
                display_text=r".*#([A-Z0-9]+)$",
                help="Open the imported dataset for case and reference-complex exploration.",
            ),
            "Combined results": st.column_config.LinkColumn(
                "Combined results",
                display_text="Open",
                help="Open the combined redocking/refolding metrics, viewer, and lineage.",
            ),
            "Created": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )
    st.caption(
        "Select the Job ID to explore the immutable reference dataset. Select "
        "Combined results to review all redocking, refolding, and rescoring jobs "
        "associated with that dataset."
    )


def _render_import() -> None:
    st.subheader("Import benchmark dataset")
    input_tab, mapping_tab, validation_tab, run_tab, results_tab = st.tabs(
        ["Input", "Format", "Validation", "Run", "Results"]
    )

    with input_tab:
        dataset_name = st.text_input("Dataset name", key="benchmark_dataset_name")
        source_mode = st.radio(
            "Source",
            ("Upload archive", "Server-local directory"),
            horizontal=True,
            key="benchmark_dataset_source_mode",
        )
        archive = None
        source_directory = ""
        if source_mode == "Upload archive":
            archive = st.file_uploader(
                "Benchmark archive",
                type=["zip", "tar", "gz", "tgz"],
                key="benchmark_dataset_archive",
                help=(
                    "ZIP, TAR, TAR.GZ, and TGZ are accepted. Large PoseBench downloads "
                    "can instead be extracted once and registered by server path."
                ),
            )
        else:
            source_directory = st.text_input(
                "Dataset directory on this machine",
                key="benchmark_dataset_directory",
                placeholder="/mnt/data/benchmarks/posebusters_benchmark_set",
            )
            st.caption(
                "Files are copied into the immutable import job. The original directory "
                "is never modified and is not required by later campaigns."
            )

    with mapping_tab:
        profile = st.selectbox(
            "Import profile",
            PROFILES,
            key="benchmark_dataset_profile",
        )
        manifest = st.file_uploader(
            "Optional manifest",
            type=["csv", "json", "yaml", "yml"],
            key="benchmark_dataset_manifest",
            help=(
                "Columns/keys: case_id, receptor, ligand; optional target_id, complex, "
                "sequence, ligand_smiles, split, and arbitrary metadata. Without a "
                "manifest, standard PoseBench and per-case receptor/ligand names are "
                "detected automatically."
            ),
        )
        source_url = st.text_input(
            "Source DOI or URL (recommended)",
            key="benchmark_dataset_source_url",
            placeholder="https://doi.org/…",
        )
        source_citation = st.text_input(
            "Source citation, version, or license note (recommended)",
            key="benchmark_dataset_source_citation",
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Field": "case_id",
                        "Required": True,
                        "Meaning": "Stable benchmark case identity",
                    },
                    {
                        "Field": "receptor",
                        "Required": True,
                        "Meaning": "Reference receptor PDB path inside the archive",
                    },
                    {
                        "Field": "ligand",
                        "Required": True,
                        "Meaning": "Coordinate-bearing reference ligand SDF",
                    },
                    {
                        "Field": "complex",
                        "Required": False,
                        "Meaning": "Optional complete reference complex PDB",
                    },
                    {
                        "Field": "sequence",
                        "Required": False,
                        "Meaning": "Optional FASTA/sequence input for refolding",
                    },
                    {
                        "Field": "split",
                        "Required": False,
                        "Meaning": "Dataset split or cohort label",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.info(
            "PoseBench is an import profile, not a runtime dependency. mn-ligand "
            "keeps one generic benchmark case schema and prepares each engine's "
            "native input only when a campaign is launched."
        )

    archive_data = archive.getvalue() if archive is not None else None
    archive_name = archive.name if archive is not None else ""
    manifest_data = manifest.getvalue() if manifest is not None else None
    manifest_name = manifest.name if manifest is not None else ""
    preview: dict | None = None
    preview_error = ""
    with validation_tab:
        if archive_data is not None:
            try:
                preview = _preview(
                    archive_data,
                    archive_name,
                    manifest_data,
                    manifest_name,
                    profile,
                )
            except Exception as exc:
                preview_error = str(exc)
                st.error(preview_error)
        elif source_mode == "Server-local directory" and source_directory:
            st.info(
                "Server-local directories are inspected during registration to avoid "
                "loading a potentially multi-gigabyte dataset into the Streamlit process."
            )
        else:
            st.info("Provide a benchmark archive or directory in the Input tab.")
        if preview is not None:
            metrics = st.columns(4)
            metrics[0].metric("Detected cases", preview["case_count"])
            metrics[1].metric("Rejected", preview["rejected_case_count"])
            metrics[2].metric("Source files", preview["source_file_count"])
            metrics[3].metric("Profile", preview["profile"])
            rows = [
                {
                    "Case": case["case_id"],
                    "Target": case["target_id"],
                    "Split": case["split"],
                    "Receptor": case["receptor_member"],
                    "Reference ligand": case["ligand_member"],
                    "Ligand records": case["record_count"],
                    "3D dimension": case["coordinate_dimension"],
                }
                for case in preview["cases"]
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=480)
            if preview["errors"]:
                st.warning("Rejected cases are retained in the import report.")
                st.dataframe(
                    pd.DataFrame(preview["errors"]), hide_index=True, width="stretch"
                )

    with run_tab:
        render_run_resources(
            requires_gpu=False,
            selected_gpu="Not used",
            key="benchmark_dataset_import",
        )
        has_source = archive_data is not None or bool(source_directory.strip())
        blockers: list[str] = []
        if not has_source:
            blockers.append("Provide an archive or server-local directory.")
        if preview_error:
            blockers.append("Resolve the archive or manifest validation error.")
        if preview is not None and not preview["case_count"]:
            blockers.append("No valid cases were detected.")
        for blocker in blockers:
            st.info(blocker)
        if st.button(
            "Register benchmark dataset",
            type="primary",
            disabled=bool(blockers),
            key="benchmark_dataset_register",
        ):
            try:
                job = create_benchmark_dataset_job(
                    dataset_name=dataset_name,
                    archive_data=archive_data,
                    archive_filename=archive_name,
                    source_directory=(
                        Path(source_directory) if source_directory.strip() else None
                    ),
                    manifest_data=manifest_data,
                    manifest_filename=manifest_name,
                    requested_profile=profile,
                    source_provenance={
                        "url": source_url.strip(),
                        "citation": source_citation.strip(),
                    },
                )
                st.success(
                    f"Registered {job.metadata.get('case_count')} benchmark cases as "
                    f"{display_job_code(job.metadata.get('job_code'), job.run_id)}."
                )
            except Exception as exc:
                st.error(f"Benchmark dataset import failed: {exc}")

    with results_tab:
        st.subheader("Imported benchmark datasets")
        _render_results()


def render() -> None:
    st.title("Benchmark Datasets")
    st.caption(
        "Import immutable reference collections for redocking, refolding, "
        "rescoring, and cross-engine comparison."
    )
    _render_import()


if __name__ == "__main__":
    render()
