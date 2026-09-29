from __future__ import annotations

from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records
from mn_ligand.runtime import reference_root, runs_root
from mn_ligand.workflows.lddm import LDDM_CHECKPOINTS
from mn_ligand.workflows.lddm_evaluation import (
    LDDM_EVALUATION_TASK_GROUP,
    compatible_source_jobs,
    queue_lddm_pose_agreement_job,
    source_pose_inventory,
)


def _source_label(job: JobRecord) -> str:
    code = display_job_code(job.metadata.get("job_code"), job.run_id)
    engine = str(job.metadata.get("tool") or job.tool or "Docking")
    if engine.lower() == "openvs":
        engine = "RosettaLigand"
    label = str(job.metadata.get("launch_campaign_label") or "").strip()
    suffix = f" · {label}" if label else ""
    return f"{code} · {engine}{suffix}"


def _history_rows(jobs: list[JobRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for job in jobs:
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        query = urlencode(
            {
                "task_group": job.task_group,
                "run_id": job.run_id,
                "label": code,
            }
        )
        rows.append(
            {
                "results": f"./job-results?{query}",
                "job": code,
                "source engine": job.metadata.get("source_engine", ""),
                "compounds": job.metadata.get("compared_compound_count", ""),
                "poses compared": job.metadata.get("compared_pose_count", ""),
                "status": job.status,
                "created": job.created_at,
            }
        )
    return rows


def render() -> None:
    st.title("LDDM Pose Agreement")
    st.caption(
        "Independently sample poses with LDDM for compounds from a completed "
        "docking campaign, then compare its confidence-ranked poses with every "
        "available source pose."
    )
    st.info(
        "LDDM uses the ligand identity and binding pocket, then generates new "
        "coordinates. Its mean atom coordinate uncertainty is a model-confidence "
        "estimate; it is not a binding energy or affinity. RMSD is measured in "
        "the fixed receptor frame with symmetry-aware heavy-atom mapping. Low "
        "uncertainty is not a calibrated probability, and agreement between "
        "methods does not by itself prove that a pose is correct."
    )

    source_tab, settings_tab, run_tab, results_tab = st.tabs(
        ["Source Campaign", "LDDM Settings", "Run", "Results"]
    )
    source_jobs = compatible_source_jobs()
    requested_run_id = str(st.query_params.get("source_run_id", "") or "")
    requested_source = next(
        (job for job in source_jobs if job.run_id == requested_run_id), None
    )
    source_by_label = {_source_label(job): job for job in source_jobs}
    with source_tab:
        if not source_jobs:
            st.info(
                "Complete a Vina, GNINA, Uni-Dock Pro, or RosettaLigand campaign first. Its "
                "receptor and ligand poses will be used as the evaluation input."
            )
            selected_source = None
        else:
            labels = list(source_by_label)
            selected_label = st.selectbox(
                "Completed docking campaign",
                labels,
                index=(labels.index(_source_label(requested_source))
                       if requested_source is not None
                       and _source_label(requested_source) in labels
                       else 0),
                key="lddm_eval_source_job",
            )
            selected_source = source_by_label[selected_label]
            rows = source_pose_inventory(selected_source)
            engine_label = str(
                selected_source.metadata.get("tool") or selected_source.tool
            )
            if engine_label.lower() == "openvs":
                engine_label = "RosettaLigand"
            compound_count = len(
                {str(row.get("compound_id") or "") for row in rows}
            )
            columns = st.columns(3)
            columns[0].metric("Engine", engine_label)
            columns[1].metric("Compounds with poses", compound_count)
            columns[2].metric("Source poses", len(rows))
            st.caption(
                "The stored reference ligand defines the pocket when available. "
                "Otherwise LDDM uses a source pose only to locate pocket residues; "
                "the source pose coordinates are not fixed during LDDM sampling."
            )

    with settings_tab:
        checkpoint_label = st.selectbox(
            "LDDM checkpoint",
            tuple(LDDM_CHECKPOINTS),
            key="lddm_eval_checkpoint",
            help=(
                "CD+BB (MIT) permits commercial use. CD+BB+BN is CC-BY-NC 4.0 "
                "and is limited to non-commercial use."
            ),
        )
        checkpoint_path = LDDM_CHECKPOINTS[checkpoint_label]
        if not (reference_root() / checkpoint_path).is_file():
            st.warning(f"Checkpoint file is missing: {checkpoint_path}")
        columns = st.columns(3)
        samples_per_compound = int(
            columns[0].number_input(
                "LDDM samples per compound",
                min_value=1,
                max_value=100,
                value=20,
                step=1,
                key="lddm_eval_sample_count",
            )
        )
        maximum_compounds = int(
            columns[1].number_input(
                "Maximum compounds (0 = all)",
                min_value=0,
                max_value=100000,
                value=100,
                step=10,
                key="lddm_eval_max_compounds",
            )
        )
        seed = int(
            columns[2].number_input(
                "Random seed",
                min_value=1,
                max_value=2147483646,
                value=2026,
                step=1,
                key="lddm_eval_seed",
            )
        )
        advanced = st.expander("Advanced sampling", expanded=False)
        with advanced:
            advanced_columns = st.columns(3)
            n_steps = int(
                advanced_columns[0].number_input(
                    "Integration steps",
                    min_value=1,
                    max_value=1000,
                    value=100,
                    step=10,
                    key="lddm_eval_steps",
                )
            )
            sampler = advanced_columns[1].selectbox(
                "Sampler",
                ("ForwardEuler", "HeunSampler"),
                key="lddm_eval_sampler",
            )
            sampling_noise = float(
                advanced_columns[2].number_input(
                    "Sampling noise",
                    min_value=0.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    key="lddm_eval_noise",
                )
            )
        st.caption(
            "The report includes both the lowest-RMSD sampled pose and the "
            "lowest-uncertainty sampled pose for each source pose."
        )

    with run_tab:
        render_run_resources(
            requires_gpu=True,
            selected_gpu="Automatic",
            key="lddm_pose_agreement",
        )
        if selected_source is None:
            st.info("Select a completed docking campaign to continue.")
        elif not (reference_root() / checkpoint_path).is_file():
            st.warning("Install the selected LDDM checkpoint before queueing a run.")
        else:
            if st.button(
                "Queue LDDM pose agreement",
                type="primary",
                key="lddm_eval_queue",
            ):
                try:
                    job = queue_lddm_pose_agreement_job(
                        source_job=selected_source,
                        checkpoint_path=checkpoint_path,
                        samples_per_compound=samples_per_compound,
                        maximum_compounds=maximum_compounds,
                        seed=seed,
                        n_steps=n_steps,
                        sampler=sampler,
                        sampling_noise=sampling_noise,
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    st.success(
                        "Queued LDDM pose agreement job "
                        f"{display_job_code(job.metadata.get('job_code'), job.run_id)}."
                    )
                    st.rerun()

    with results_tab:
        jobs = iter_job_records(
            runs_root(),
            task_groups=(LDDM_EVALUATION_TASK_GROUP,),
            validate_artifacts=False,
        )
        jobs = sorted(jobs, key=lambda job: job.created_at, reverse=True)
        if jobs:
            st.dataframe(
                pd.DataFrame(_history_rows(jobs)),
                hide_index=True,
                width="stretch",
                column_config={
                    "results": st.column_config.LinkColumn("Results", display_text="Open"),
                },
            )
            for job in jobs:
                if job.status != "completed":
                    continue
                report = job.run_dir / "pose_agreement.csv"
                if not report.is_file():
                    continue
                with st.expander(
                    f"{display_job_code(job.metadata.get('job_code'), job.run_id)} "
                    f"· {job.metadata.get('source_engine', 'Docking')} results"
                ):
                    summary_path = job.run_dir / "pose_agreement_summary.csv"
                    if summary_path.is_file():
                        try:
                            summary = pd.read_csv(summary_path)
                        except (OSError, ValueError) as exc:
                            st.warning(f"Could not read compound summary: {exc}")
                        else:
                            st.markdown("#### Compound summary")
                            st.dataframe(summary, hide_index=True, width="stretch")
                    st.markdown("#### Per-pose comparisons")
                    try:
                        frame = pd.read_csv(report)
                    except (OSError, ValueError) as exc:
                        st.warning(f"Could not read pose agreement report: {exc}")
                    else:
                        st.dataframe(frame, hide_index=True, width="stretch")
        else:
            st.info("No LDDM pose agreement jobs have been run yet.")
