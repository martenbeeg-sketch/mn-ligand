from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records
from mn_ligand.runtime import cpu_process_limit, runs_root
from mn_ligand.workflows.pose_validation import (
    POSE_VALIDATION_INVENTORY_SCHEMA_VERSION,
    POSE_VALIDATION_SELECTION_POLICY,
    POSE_VALIDATION_TASK_GROUP,
    cached_pose_validation_candidates,
    compatible_source_jobs,
    pose_validation_inventory_summary,
    queue_pose_validation_job,
)


def _source_label(job: JobRecord) -> str:
    code = display_job_code(job.metadata.get("job_code"), job.run_id)
    engine = str(job.tool or job.metadata.get("engine") or job.workflow)
    kind = (
        "cofolding"
        if job.workflow in {"alphafold3_refolding", "boltz2_refolding"}
        else "docking"
    )
    return f"{code} · {engine} · {kind}"


def _result_rows(
    jobs: Iterable[JobRecord] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source = jobs if jobs is not None else iter_job_records(
        runs_root(), task_groups=(POSE_VALIDATION_TASK_GROUP,)
    )
    for job in source:
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
                "poses": job.metadata.get("pose_count", ""),
                "passed": job.result.get("passed_count", ""),
                "failed": job.result.get("failed_count", ""),
                "status": job.status,
                "created": job.created_at,
            }
        )
    return rows


def _source_key(job: JobRecord) -> str:
    return f"{job.task_group}::{job.run_id}"


def _validation_jobs_by_source(
    jobs: Iterable[JobRecord] | None = None,
) -> dict[str, list[JobRecord]]:
    grouped: dict[str, list[JobRecord]] = {}
    source = jobs if jobs is not None else iter_job_records(
        runs_root(), task_groups=(POSE_VALIDATION_TASK_GROUP,)
    )
    for job in source:
        source_run_id = str(job.parent_run_id or "").strip()
        if not source_run_id:
            continue
        grouped.setdefault(source_run_id, []).append(job)
    return grouped


def _source_validation_status(jobs: list[JobRecord]) -> str:
    current = [
        job
        for job in jobs
        if int(job.metadata.get("selection_schema_version") or 0)
        == POSE_VALIDATION_INVENTORY_SCHEMA_VERSION
        and str(job.metadata.get("selection_policy") or "")
        == POSE_VALIDATION_SELECTION_POLICY
    ]
    statuses = {str(job.status) for job in current}
    if "running" in statuses:
        return "Running"
    if "queued" in statuses:
        return "Queued"
    if "completed" in statuses:
        return "Completed"
    if jobs:
        if any(job.status == "completed" for job in jobs):
            return "Completed — selection refresh required"
        return "Failed — retry available"
    return "Not run"


def _is_missing_validation(jobs: list[JobRecord]) -> bool:
    return not any(
        job.status in {"queued", "running", "completed"}
        and int(job.metadata.get("selection_schema_version") or 0)
        == POSE_VALIDATION_INVENTORY_SCHEMA_VERSION
        and str(job.metadata.get("selection_policy") or "")
        == POSE_VALIDATION_SELECTION_POLICY
        for job in jobs
    )


def render() -> None:
    st.title("Pose Validation")
    st.caption(
        "Test whether docked and cofolded ligand poses are chemically and "
        "geometrically plausible inside their protein pocket using PoseBusters."
    )
    input_tab, engine_tab, run_tab, results_tab = st.tabs(
        ["Target / Input", "Tool / Engine", "Run", "Results"]
    )
    sources = compatible_source_jobs()
    requested_run_id = str(st.query_params.get("source_run_id", "") or "")
    requested_group = str(
        st.query_params.get("source_task_group", "") or ""
    )
    requested = next(
        (
            job
            for job in sources
            if job.run_id == requested_run_id
            and (not requested_group or job.task_group == requested_group)
        ),
        None,
    )
    source_jobs = {_source_key(job): job for job in sources}
    pose_validation_jobs = iter_job_records(
        runs_root(),
        task_groups=(POSE_VALIDATION_TASK_GROUP,),
        validate_artifacts=False,
    )
    validation_jobs = _validation_jobs_by_source(pose_validation_jobs)
    inventory_summaries: dict[str, dict[str, object]] = {}

    def load_summary(
        item: tuple[str, JobRecord],
    ) -> tuple[str, dict[str, object]]:
        source_key, source = item
        try:
            return source_key, pose_validation_inventory_summary(source)
        except (OSError, TypeError, ValueError):
            return source_key, {
                "cached": False,
                "candidate_count": None,
                "compound_ids": (),
            }

    source_items = list(source_jobs.items())
    if source_items:
        with ThreadPoolExecutor(
            max_workers=min(8, len(source_items)),
            thread_name_prefix="pose-inventory-summary",
        ) as executor:
            for source_key, summary in executor.map(
                load_summary,
                source_items,
            ):
                inventory_summaries[source_key] = summary
    missing_source_keys = [
        source_key
        for source_key in source_jobs
        if _is_missing_validation(
            validation_jobs.get(source_jobs[source_key].run_id, [])
        )
    ]
    requested_source_key = _source_key(requested) if requested else ""
    if (
        requested_source_key
        and st.session_state.get("_posebusters_requested_source")
        != requested_source_key
    ):
        st.session_state["_posebusters_requested_source"] = (
            requested_source_key
        )
        st.session_state["posebusters_selected_sources"] = [
            requested_source_key
        ]
        st.session_state["_posebusters_editor_revision"] = (
            int(st.session_state.get("_posebusters_editor_revision", 0)) + 1
        )
    with input_tab:
        if not source_jobs:
            st.info(
                "Complete an AutoDock Vina, GNINA, Uni-Dock Pro, "
                "RosettaLigand, AlphaFold 3, or Boltz-2 result first."
            )
            selected_source_keys: list[str] = []
        else:
            selection_context = (
                tuple(source_jobs),
                tuple(missing_source_keys),
            )
            if (
                st.session_state.get("_posebusters_source_context")
                != selection_context
            ):
                st.session_state["_posebusters_source_context"] = (
                    selection_context
                )
                st.session_state["posebusters_selected_sources"] = (
                    [requested_source_key]
                    if requested_source_key
                    else list(missing_source_keys)
                )
                st.session_state["_posebusters_editor_revision"] = (
                    int(
                        st.session_state.get(
                            "_posebusters_editor_revision", 0
                        )
                    )
                    + 1
                )
            action_columns = st.columns(3)
            if action_columns[0].button(
                "Select all missing",
                key="posebusters_select_missing",
                help=(
                    "Select sources with no completed or active validation. "
                    "Failed-only jobs are selected for retry."
                ),
            ):
                st.session_state["posebusters_selected_sources"] = list(
                    missing_source_keys
                )
                st.session_state["_posebusters_editor_revision"] = (
                    int(
                        st.session_state.get(
                            "_posebusters_editor_revision", 0
                        )
                    )
                    + 1
                )
            if action_columns[1].button(
                "Select all eligible",
                key="posebusters_select_all",
            ):
                st.session_state["posebusters_selected_sources"] = list(
                    source_jobs
                )
                st.session_state["_posebusters_editor_revision"] = (
                    int(
                        st.session_state.get(
                            "_posebusters_editor_revision", 0
                        )
                    )
                    + 1
                )
            if action_columns[2].button(
                "Clear selection",
                key="posebusters_select_none",
            ):
                st.session_state["posebusters_selected_sources"] = []
                st.session_state["_posebusters_editor_revision"] = (
                    int(
                        st.session_state.get(
                            "_posebusters_editor_revision", 0
                        )
                    )
                    + 1
                )
            selected_source_keys = [
                value
                for value in st.session_state[
                    "posebusters_selected_sources"
                ]
                if value in source_jobs
            ]
            coverage_rows = []
            for source_key, source in source_jobs.items():
                children = validation_jobs.get(source.run_id, [])
                current_children = [
                    job
                    for job in children
                    if int(
                        job.metadata.get("selection_schema_version") or 0
                    )
                    == POSE_VALIDATION_INVENTORY_SCHEMA_VERSION
                    and str(job.metadata.get("selection_policy") or "")
                    == POSE_VALIDATION_SELECTION_POLICY
                ]
                latest = max(
                    current_children or children,
                    key=lambda job: str(job.created_at or ""),
                    default=None,
                )
                coverage_rows.append(
                    {
                        "_source_key": source_key,
                        "selected": source_key in selected_source_keys,
                        "source job": display_job_code(
                            source.metadata.get("job_code"),
                            source.run_id,
                        ),
                        "engine": source.tool,
                        "workflow": source.workflow,
                        "selected poses": inventory_summaries[
                            source_key
                        ].get("candidate_count"),
                        "validation status": _source_validation_status(
                            children
                        ),
                        "validation jobs": len(children),
                        "latest validation": (
                            display_job_code(
                                latest.metadata.get("job_code"),
                                latest.run_id,
                            )
                            if latest is not None
                            else ""
                        ),
                        "latest selected poses": (
                            int(latest.metadata.get("pose_count") or 0)
                            if latest is not None
                            else 0
                        ),
                    }
                )
            coverage_frame = pd.DataFrame(coverage_rows)
            filter_columns = st.columns(3)
            source_search = filter_columns[0].text_input(
                "Filter jobs",
                key="posebusters_source_search",
                placeholder="Job, engine, workflow, or status",
            )
            engine_options = sorted(
                coverage_frame["engine"].astype(str).unique()
            )
            if "posebusters_engine_filter" not in st.session_state:
                st.session_state["posebusters_engine_filter"] = list(
                    engine_options
                )
            else:
                st.session_state["posebusters_engine_filter"] = [
                    value
                    for value in st.session_state[
                        "posebusters_engine_filter"
                    ]
                    if value in engine_options
                ]
            selected_engines = filter_columns[1].multiselect(
                "Engines",
                engine_options,
                key="posebusters_engine_filter",
            )
            status_options = list(
                dict.fromkeys(
                    coverage_frame["validation status"].astype(str)
                )
            )
            if "posebusters_status_filter" not in st.session_state:
                st.session_state["posebusters_status_filter"] = list(
                    status_options
                )
            else:
                st.session_state["posebusters_status_filter"] = [
                    value
                    for value in st.session_state[
                        "posebusters_status_filter"
                    ]
                    if value in status_options
                ]
            selected_statuses = filter_columns[2].multiselect(
                "Validation status",
                status_options,
                key="posebusters_status_filter",
            )
            visible_coverage = coverage_frame.loc[
                coverage_frame["engine"].astype(str).isin(selected_engines)
                & coverage_frame["validation status"]
                .astype(str)
                .isin(selected_statuses)
            ].copy()
            search_text = str(source_search or "").strip()
            if search_text:
                searchable = visible_coverage[
                    [
                        "source job",
                        "engine",
                        "workflow",
                        "validation status",
                    ]
                ].astype(str).agg(" ".join, axis=1)
                visible_coverage = visible_coverage.loc[
                    searchable.str.contains(
                        search_text,
                        case=False,
                        regex=False,
                    )
                ].copy()
            editor_context = tuple(
                visible_coverage["_source_key"].astype(str)
            )
            if (
                st.session_state.get("_posebusters_editor_context")
                != editor_context
            ):
                st.session_state["_posebusters_editor_context"] = (
                    editor_context
                )
                st.session_state["_posebusters_editor_revision"] = (
                    int(
                        st.session_state.get(
                            "_posebusters_editor_revision", 0
                        )
                    )
                    + 1
                )
            if visible_coverage.empty:
                st.info("No source jobs match the current filters.")
            else:
                edited_coverage = st.data_editor(
                    visible_coverage,
                    hide_index=True,
                    width="stretch",
                    disabled=[
                        column
                        for column in visible_coverage.columns
                        if column != "selected"
                    ],
                    column_config={
                        "_source_key": None,
                        "selected": st.column_config.CheckboxColumn(
                            "Validate",
                            help=(
                                "The focused scientific pose selection from "
                                "each checked source job will be validated."
                            ),
                        ),
                    },
                    key=(
                        "posebusters_source_editor_"
                        + str(
                            st.session_state.get(
                                "_posebusters_editor_revision", 0
                            )
                        )
                    ),
                )
                visible_keys = set(
                    visible_coverage["_source_key"].astype(str)
                )
                checked_visible = set(
                    edited_coverage.loc[
                        edited_coverage["selected"]
                        .fillna(False)
                        .astype(bool),
                        "_source_key",
                    ].astype(str)
                )
                selected_source_keys = [
                    value
                    for value in selected_source_keys
                    if value not in visible_keys
                ] + [
                    value
                    for value in source_jobs
                    if value in checked_visible
                ]
            st.session_state["posebusters_selected_sources"] = list(
                selected_source_keys
            )
            status_metrics = st.columns(4)
            status_metrics[0].metric("Eligible jobs", len(source_jobs))
            status_metrics[1].metric(
                "Already covered",
                len(source_jobs) - len(missing_source_keys),
            )
            status_metrics[2].metric(
                "Missing / retry",
                len(missing_source_keys),
            )
            status_metrics[3].metric(
                "Selected jobs",
                len(selected_source_keys),
            )
        selected_summaries = [
            inventory_summaries.get(source_key, {})
            for source_key in selected_source_keys
        ]
        pending_inventory_count = sum(
            summary.get("candidate_count") is None
            for summary in selected_summaries
        )
        selected_pose_count = sum(
            int(summary.get("candidate_count") or 0)
            for summary in selected_summaries
        )
        selected_compound_ids = {
            str(compound_id)
            for summary in selected_summaries
            for compound_id in summary.get("compound_ids", ())
            if str(compound_id)
        }
        metrics = st.columns(3)
        metrics[0].metric("Selected jobs", len(selected_source_keys))
        metrics[1].metric("Selected poses", selected_pose_count)
        metrics[2].metric("Compounds", len(selected_compound_ids))
        if pending_inventory_count:
            st.caption(
                f"{pending_inventory_count} selected source inventory(s) "
                "will be generated when validation is submitted."
            )
        st.caption(
            "Selection is per source job. Counts contain only the focused "
            "scientific poses: best classical/Rosetta pose per repetition, "
            "AF3 sample 0, Boltz-2 model 0, and GNINA's CNN- and "
            "Vina-ranked poses (deduplicated when identical)."
        )
        if (
            selected_source_keys
            and not selected_pose_count
            and not pending_inventory_count
        ):
            st.warning("The selected sources contain no compatible poses.")

    with engine_tab:
        st.markdown("#### PoseBusters 0.6.5")
        st.write(
            "The docking configuration checks ligand loading, sanitization, "
            "connectivity, radicals, bond lengths and angles, internal clashes, "
            "ring geometry, internal energy, protein distance, protein clashes, "
            "and volume overlap."
        )
        st.info(
            "PoseBusters is CPU-only. It already parallelizes independent files "
            "with multiprocessing, so this workflow deliberately does not request "
            "a GPU or block an RTX 5090 lease."
        )
        st.metric("Global CPU limit", cpu_process_limit())
        st.caption(
            "PoseBusters workers are assigned automatically per validation "
            "job: the smaller of the global CPU limit and the number of "
            "selected compounds. One compound therefore uses one worker. The "
            "shared CPU lease pool prevents all running jobs from exceeding "
            "the global limit."
        )
        st.caption(
            "No global reference files or model weights are required. Source "
            "receptors, ligand topology, and optional complex coordinates are "
            "copied into the immutable validation job."
        )

    with run_tab:
        render_run_resources(
            requires_gpu=False,
            key="posebusters",
        )
        st.markdown("#### Submit validation")
        st.caption(
            f"{selected_pose_count} cached pose(s) from "
            f"{len(selected_source_keys)} source job(s) will be checked with "
            "an automatically sized CPU worker pool per validation job."
        )
        submit = st.button(
            (
                f"Queue {len(selected_source_keys)} PoseBusters "
                f"job{'s' if len(selected_source_keys) != 1 else ''}"
            ),
            type="primary",
            disabled=(
                not selected_source_keys
                or (
                    not selected_pose_count
                    and not pending_inventory_count
                )
            ),
            key="posebusters_submit",
        )
        if submit:
            queued: list[JobRecord] = []
            errors: list[str] = []
            for source_key in selected_source_keys:
                source = source_jobs[source_key]
                try:
                    source_rows = cached_pose_validation_candidates(source)
                except (OSError, TypeError, ValueError) as exc:
                    errors.append(f"{_source_label(source)}: {exc}")
                    continue
                if not source_rows:
                    continue
                try:
                    queued.append(
                        queue_pose_validation_job(
                            source,
                            selected_rows=source_rows,
                        )
                    )
                except (OSError, TypeError, ValueError) as exc:
                    errors.append(f"{_source_label(source)}: {exc}")
            if queued:
                st.success(
                    f"Queued {len(queued)} PoseBusters validation "
                    f"job{'s' if len(queued) != 1 else ''}."
                )
                queued_rows = []
                for job in queued:
                    code = display_job_code(
                        job.metadata.get("job_code"),
                        job.run_id,
                    )
                    queued_rows.append(
                        {
                            "job": code,
                            "source engine": job.metadata.get(
                                "source_engine", ""
                            ),
                            "poses": job.metadata.get("pose_count", ""),
                            "result": (
                                "./job-results?"
                                + urlencode(
                                    {
                                        "task_group": job.task_group,
                                        "run_id": job.run_id,
                                        "label": code,
                                    }
                                )
                            ),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(queued_rows),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "result": st.column_config.LinkColumn(
                            "Result",
                            display_text="Open",
                        )
                    },
                )
            for error in errors:
                st.error(error)

    with results_tab:
        rows = _result_rows(pose_validation_jobs)
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "results": st.column_config.LinkColumn(
                        "Results",
                        display_text="Open",
                    )
                },
            )
        else:
            st.info("No PoseBusters validation jobs are available yet.")


render()
