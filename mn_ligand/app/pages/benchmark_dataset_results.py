from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.benchmark_common import (
    campaign_table,
    render_case_viewer,
    visible_dataset_jobs,
)
from mn_ligand.app.pages.benchmark_results import _dataset_statistics
from mn_ligand.core.jobs import display_job_code
from mn_ligand.workflows.benchmark_datasets import (
    benchmark_cases,
    load_benchmark_dataset,
)


def render() -> None:
    st.title("Benchmark Dataset")
    datasets = visible_dataset_jobs()
    requested = str(st.query_params.get("dataset_run_id") or "")
    if not datasets:
        st.info("No benchmark datasets have been imported.")
        return
    labels = {
        (
            f"{load_benchmark_dataset(job)['dataset_name']} — "
            f"{display_job_code(job.metadata.get('job_code'), job.run_id)}"
        ): job
        for job in datasets
    }
    default = next(
        (index for index, job in enumerate(labels.values()) if job.run_id == requested),
        0,
    )
    selected = st.selectbox(
        "Imported benchmark dataset",
        list(labels),
        index=default,
        key="benchmark_dataset_explorer_selection",
    )
    job = labels[selected]
    dataset = load_benchmark_dataset(job)
    cases = benchmark_cases(job)
    statistics = _dataset_statistics(
        str(job.run_dir),
        (job.run_dir / "benchmark_dataset.json").stat().st_mtime_ns,
    )
    campaigns = campaign_table(job.run_id)

    overview_tab, cases_tab, artifacts_tab, lineage_tab = st.tabs(
        ["Overview", "Cases / Viewer", "Artifacts", "Lineage"]
    )
    with overview_tab:
        summary = st.columns(6)
        summary[0].metric("Cases", dataset["case_count"])
        summary[1].metric("Rejected", dataset.get("rejected_case_count", 0))
        summary[2].metric("Profile", dataset["profile"])
        summary[3].metric(
            "Median ligand MW",
            f"{statistics['molecular_weight'].median():.1f}"
            if not statistics.empty
            else "—",
        )
        summary[4].metric(
            "Median residues",
            f"{statistics['receptor_residues'].median():.0f}"
            if not statistics.empty
            else "—",
        )
        summary[5].metric("Derived jobs", len(campaigns))
        if not statistics.empty:
            plots = st.columns(2)
            plots[0].caption("Ligand molecular weight")
            plots[0].bar_chart(
                statistics[["case_id", "molecular_weight"]].set_index("case_id")
            )
            plots[1].caption("Receptor residues")
            plots[1].bar_chart(
                statistics[["case_id", "receptor_residues"]].set_index("case_id")
            )
        provenance = dataset.get("source_provenance") or {}
        if provenance:
            st.markdown("#### Source and provenance")
            st.json(provenance, expanded=False)

    with cases_tab:
        case_by_id = {case.case_id: case for case in cases}
        table = statistics.copy().reset_index(drop=True)
        table.insert(
            3,
            "ligand_smiles",
            table["case_id"].map(
                lambda case_id: str(
                    case_by_id.get(str(case_id)).payload.get("ligand_smiles", "")
                )
                if str(case_id) in case_by_id
                else ""
            ),
        )
        event = st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height=560,
            key=f"benchmark_dataset_case_table_{job.run_id}",
            on_select="rerun",
            selection_mode="single-row-required",
            selection_default={"selection": {"rows": [0]}},
            column_config={
                "molecular_weight": st.column_config.NumberColumn(
                    "MW (Da)", format="%.2f"
                ),
                "tpsa": st.column_config.NumberColumn("TPSA (Å²)", format="%.2f"),
                "logp": st.column_config.NumberColumn("cLogP", format="%.2f"),
                "qed": st.column_config.NumberColumn("QED", format="%.3f"),
                "ligand_smiles": st.column_config.TextColumn(
                    "Ligand SMILES", width="large"
                ),
            },
        )
        selected_rows = list(event.selection.rows)
        selected_index = selected_rows[0] if selected_rows else 0
        selected_row = table.iloc[selected_index]
        selected_case_id = str(selected_row["case_id"])
        selected_case = case_by_id.get(selected_case_id)

        st.markdown(f"#### {selected_case_id}")
        identity = st.columns(4)
        identity[0].metric("Target", str(selected_row["target_id"]) or "—")
        identity[1].metric("Split", str(selected_row["split"]) or "—")
        identity[2].metric("Formula", str(selected_row["formula"]) or "—")
        identity[3].metric("Formal charge", int(selected_row["formal_charge"]))
        compound = st.columns(6)
        compound[0].metric("MW", f"{selected_row['molecular_weight']:.2f} Da")
        compound[1].metric("Heavy atoms", int(selected_row["ligand_heavy_atoms"]))
        compound[2].metric("HBD / HBA", f"{int(selected_row['hbond_donors'])} / {int(selected_row['hbond_acceptors'])}")
        compound[3].metric("TPSA", f"{selected_row['tpsa']:.2f} Å²")
        compound[4].metric("cLogP", f"{selected_row['logp']:.2f}")
        compound[5].metric("QED", f"{selected_row['qed']:.3f}")
        geometry = st.columns(5)
        geometry[0].metric("Fragments", int(selected_row["ligand_fragments"]))
        geometry[1].metric("Rings", int(selected_row["ring_count"]))
        geometry[2].metric("Rotatable bonds", int(selected_row["rotatable_bonds"]))
        geometry[3].metric("Receptor chains", int(selected_row["receptor_chains"]))
        geometry[4].metric("Receptor residues", int(selected_row["receptor_residues"]))
        st.caption("Canonical reference-ligand SMILES")
        st.code(str(selected_row["ligand_smiles"]) or "Unavailable", language=None)
        if selected_case is not None:
            render_case_viewer(
                selected_case,
                key=f"benchmark-dataset-explorer-{job.run_id}-{selected_case_id}",
            )
        st.download_button(
            "Download dataset composition table",
            table.to_csv(index=False).encode(),
            file_name=f"{dataset['dataset_name']}-composition.csv",
            mime="text/csv",
        )

    with artifacts_tab:
        artifacts = list(job.artifact_manifest.artifacts) if job.artifact_manifest else []
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Type": artifact.artifact_type,
                        "Role": artifact.role,
                        "Path": artifact.path,
                        "Size": artifact.size_bytes,
                        "SHA-256": artifact.sha256,
                    }
                    for artifact in artifacts
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        report_path = job.run_dir / "artifacts" / "reports" / "import_report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            errors = report.get("errors") or []
            if errors:
                st.markdown("#### Rejected cases")
                st.dataframe(pd.DataFrame(errors), hide_index=True, width="stretch")

    with lineage_tab:
        if campaigns.empty:
            st.info("No redocking, refolding, or rescoring jobs use this dataset yet.")
        else:
            st.dataframe(
                campaigns,
                hide_index=True,
                width="stretch",
                column_config={
                    "Result": st.column_config.LinkColumn(
                        "Result", display_text="Open"
                    )
                },
            )
        st.link_button(
            "Open Redocking / Refolding",
            f"./benchmark-campaigns?dataset_run_id={job.run_id}",
        )
        st.link_button(
            "Open combined results",
            f"./benchmark-results?dataset_run_id={job.run_id}",
        )


if __name__ == "__main__":
    render()
