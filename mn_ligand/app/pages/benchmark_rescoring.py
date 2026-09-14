from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.benchmark_datasets import list_benchmark_dataset_jobs
from mn_ligand.workflows.rescoring import (
    create_pose_selection_job,
    queue_boltzina_rescoring_job,
    queue_gnina_rescoring_job,
    source_pose_rows,
)


def _write_metadata(job: JobRecord, values: dict) -> None:
    path = job.run_dir / "metadata.json"
    payload = json.loads(path.read_text())
    payload.update(values)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _sources(dataset_run_id: str) -> list[JobRecord]:
    return [
        job
        for job in iter_job_records(runs_root(), task_groups=("docking",))
        if job.status == "completed"
        and str(job.metadata.get("benchmark_dataset_run_id") or "") == dataset_run_id
        and job.workflow == "docking_campaign"
        and source_pose_rows(job)
    ]


def _boltz_context(dataset_run_id: str, case_id: str):
    for job in iter_job_records(runs_root(), task_groups=("refolding",)):
        if (
            job.status == "completed"
            and job.workflow == "boltz2_refolding"
            and str(job.metadata.get("benchmark_dataset_run_id") or "")
            == dataset_run_id
            and str(job.metadata.get("benchmark_case_id") or "") == case_id
        ):
            manifests = sorted(job.run_dir.glob("output/**/processed/manifest.json"))
            if manifests:
                return manifests[0].parent.parent, job
    return None, None


def render(*, embedded: bool = False) -> None:
    if embedded:
        st.subheader("Rescoring")
    else:
        st.title("Benchmark Rescoring")
    st.caption(
        "Apply coordinate-preserving scoring models to top-ranked poses from a "
        "benchmark docking campaign. Rescoring never changes ligand RMSD; it tests "
        "whether each score ranks near-native poses above incorrect poses."
    )
    dataset_tab, sources_tab, engines_tab, run_tab = st.tabs(
        ["Dataset", "Source poses", "Engines", "Run"]
    )
    datasets = list_benchmark_dataset_jobs()
    with dataset_tab:
        if datasets:
            labels = {
                (
                    f"{job.metadata.get('dataset_name')} — "
                    f"{display_job_code(job.metadata.get('job_code'), job.run_id)}"
                ): job
                for job in datasets
            }
            label = st.selectbox(
                "Imported benchmark dataset",
                list(labels),
                key="benchmark_rescore_dataset",
            )
            dataset_job = labels[label]
        else:
            dataset_job = None
            st.info("Import and dock a benchmark dataset first.")

    sources = _sources(dataset_job.run_id) if dataset_job is not None else []
    with sources_tab:
        source_labels = {
            (
                f"{job.metadata.get('benchmark_case_id')} · "
                f"{job.metadata.get('engine')} · "
                f"{display_job_code(job.metadata.get('job_code'), job.run_id)}"
            ): job
            for job in sources
        }
        selected_labels = st.multiselect(
            "Completed benchmark docking jobs",
            list(source_labels),
            default=list(source_labels),
            key="benchmark_rescore_sources",
        )
        selected_sources = [source_labels[value] for value in selected_labels]
        if sources:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Case": job.metadata.get("benchmark_case_id"),
                            "Engine": job.metadata.get("engine"),
                            "Job": display_job_code(
                                job.metadata.get("job_code"), job.run_id
                            ),
                            "Available poses": len(source_pose_rows(job)),
                            "Boltzina context": bool(
                                _boltz_context(
                                    dataset_job.run_id,
                                    str(job.metadata.get("benchmark_case_id") or ""),
                                )[0]
                            ),
                        }
                        for job in sources
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No completed coordinate-bearing benchmark docking jobs are available.")

    with engines_tab:
        columns = st.columns(2)
        use_gnina = columns[0].checkbox(
            "GNINA score-only", value=True, key="benchmark_rescore_gnina"
        )
        use_boltzina = columns[1].checkbox(
            "Boltzina",
            value=False,
            key="benchmark_rescore_boltzina",
            help=(
                "Each case needs a completed Boltz-2 benchmark context. Cases without "
                "one are skipped and reported."
            ),
        )
        pose_scope = st.radio(
            "Poses to rescore",
            ("Top-ranked pose", "All stored poses"),
            horizontal=True,
            key="benchmark_rescore_scope",
        )
        cnn_rotation = int(
            st.number_input(
                "GNINA CNN rotations",
                0,
                24,
                0,
                key="benchmark_rescore_rotation",
            )
        )
        st.info(
            "Ligand RMSD is inherited from the exact source pose. The aggregate page "
            "joins score and RMSD by case, engine, replicate, and pose rank to calculate "
            "ranking success and score–RMSD association."
        )

    with run_tab:
        gpu = st.selectbox(
            "GPU", ("Automatic", "GPU 0", "GPU 1"), key="benchmark_rescore_gpu"
        )
        render_run_resources(
            requires_gpu=use_gnina or use_boltzina,
            selected_gpu=gpu,
            key="benchmark_rescore",
        )
        blockers = []
        if dataset_job is None:
            blockers.append("Select a benchmark dataset.")
        if not selected_sources:
            blockers.append("Select at least one completed benchmark docking job.")
        if not (use_gnina or use_boltzina):
            blockers.append("Select at least one rescoring engine.")
        for blocker in blockers:
            st.info(blocker)
        if st.button(
            "Queue benchmark rescoring",
            type="primary",
            disabled=bool(blockers),
            key="benchmark_rescore_queue",
        ) and dataset_job is not None:
            gpu_device = "all" if gpu == "Automatic" else gpu.replace("GPU ", "")
            queued = []
            skipped = []
            try:
                for source in selected_sources:
                    available = source_pose_rows(source)
                    ranks = sorted({int(row["source_pose_rank"]) for row in available})
                    selection = create_pose_selection_job(
                        source,
                        pose_ranks=([1] if pose_scope == "Top-ranked pose" else ranks),
                    )
                    metadata = {
                        "benchmark_dataset_run_id": dataset_job.run_id,
                        "benchmark_campaign_id": source.metadata.get(
                            "benchmark_campaign_id", ""
                        ),
                        "benchmark_case_id": source.metadata.get(
                            "benchmark_case_id", ""
                        ),
                        "benchmark_mode": "rescoring",
                        "benchmark_source_docking_run_id": source.run_id,
                    }
                    _write_metadata(selection, metadata)
                    if use_gnina:
                        job = queue_gnina_rescoring_job(
                            selection_job=selection,
                            gpu_device=gpu_device,
                            cnn_rotation=cnn_rotation,
                        )
                        _write_metadata(job, metadata)
                        queued.append(job)
                    if use_boltzina:
                        context, _ = _boltz_context(
                            dataset_job.run_id,
                            str(source.metadata.get("benchmark_case_id") or ""),
                        )
                        if context is None:
                            skipped.append(
                                f"{source.metadata.get('benchmark_case_id')}: no Boltz-2 context"
                            )
                        else:
                            job = queue_boltzina_rescoring_job(
                                selection_job=selection,
                                boltz_work_dir=context,
                                gpu_device=gpu_device,
                            )
                            _write_metadata(job, metadata)
                            queued.append(job)
                st.success(f"Queued {len(queued)} benchmark rescoring job(s).")
                if skipped:
                    st.warning("; ".join(skipped))
                st.link_button(
                    "Open aggregate dataset results",
                    f"./benchmark-results?dataset_run_id={dataset_job.run_id}",
                )
            except Exception as exc:
                st.error(f"Could not queue benchmark rescoring: {exc}")


if __name__ == "__main__":
    render()
