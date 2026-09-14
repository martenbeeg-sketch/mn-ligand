from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records
from mn_ligand.core.job_control import (
    cancellation_eligibility,
    create_job_retry,
    request_job_cancellation,
    retry_eligibility,
)
from mn_ligand.runtime import runs_root


TASK_LABELS = {
    "workflows": "Workflow",
    "protein-import": "Protein Import",
    "protein-cleaning": "Protein Cleaning / Repair",
    "compound-import": "Compound Dataset Import",
    "pocket-detection": "Pocket Detection",
    "structure-jobs": "Structure Import",
    "structure-docking": "Docking",
    "docking": "Docking",
    "batch-docking": "Virtual Screening",
    "prepared-structures": "Legacy Structure Import",
    "boltz2": "Structure Prediction",
    "md-system-prep": "MD System Preparation",
    "bound-ligand-md": "MD Production",
    "md-analysis": "MD Analysis",
    "md-mmgbsa": "MD Endpoint Energy",
    "openfe": "OpenFE",
    "abfe": "ABFE",
    "rbfe": "RBFE",
    "admet": "ADMET",
    "qc": "Quantum Chemistry",
}


def _first_value(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _gpu_label(job: JobRecord) -> str:
    metadata = job.metadata
    resources = metadata.get("resources") if isinstance(metadata.get("resources"), dict) else {}
    value = _first_value(metadata, "selected_gpu", "gpu_id", "gpu", "device") or _first_value(
        resources, "selected_gpu", "gpu_id", "gpu", "device"
    )
    if isinstance(metadata.get("gpu"), bool) or isinstance(
        resources.get("gpu"),
        bool,
    ):
        if value in {"True", "False"}:
            value = ""
    if value:
        return value
    worker_id = str(metadata.get("worker_id") or "")
    if worker_id.startswith("mn-ligand-gpu-"):
        return "GPU " + worker_id.rsplit("-", 1)[-1]
    use_gpu = metadata.get("use_gpu")
    if use_gpu is False:
        return "CPU"
    if isinstance(resources, dict) and resources.get("gpu") is True:
        return "GPU requested"
    return "Not recorded"


def _job_code(job: JobRecord) -> str:
    stored_code = _first_value(job.metadata, "openfe_job_code", "job_code") or _first_value(job.result, "job_code")
    return display_job_code(stored_code, job.run_id)


def _display_status(job: JobRecord) -> str:
    if job.status == "queued" and bool(job.metadata.get("awaiting_parent")):
        return "waiting"
    return job.status


def _status_detail(job: JobRecord, display_status: str) -> str:
    error = _first_value(job.result, "error", "failure_reason") or _first_value(
        job.metadata,
        "error",
        "failure_reason",
    )
    if error:
        return error
    if display_status == "waiting":
        return "Waiting for required upstream job(s); not eligible for a worker yet."
    if display_status == "queued":
        admission = job.metadata.get("admission")
        if isinstance(admission, dict) and admission.get("status") == "waiting":
            reasons = admission.get("reasons")
            if isinstance(reasons, list) and reasons:
                return "Waiting for resources: " + "; ".join(
                    str(reason) for reason in reasons
                )
        return "Ready to run; waiting for a compatible worker."
    if display_status in {"preparing", "running"}:
        worker = str(job.metadata.get("worker_id") or "").strip()
        return f"Active on {worker}." if worker else "Active worker job."
    if display_status == "blocked":
        return "Cannot run because a required upstream job failed or was blocked."
    if display_status == "completed":
        if bool(job.result.get("partial_success") or job.metadata.get("partial_success")):
            return "Completed with partial results."
        return "Completed successfully."
    if display_status == "cancelled":
        return "Cancelled; no further work will run."
    return ""


def _job_row(job: JobRecord) -> dict[str, Any]:
    task = TASK_LABELS.get(job.task_group, job.task_group.replace("-", " ").title())
    workflow = job.workflow or _first_value(job.metadata, "workflow_key", "protocol_preset", "preset")
    warnings = list(job.warnings)
    admission = job.metadata.get("admission")
    if isinstance(admission, dict) and admission.get("status") in {"waiting", "rejected"}:
        reasons = admission.get("reasons")
        if isinstance(reasons, list) and reasons:
            warnings.append("Resource admission: " + "; ".join(str(reason) for reason in reasons))
    display_status = _display_status(job)
    status_detail = _status_detail(job, display_status)
    warning = "; ".join(warnings)
    detail = "; ".join(
        value for value in (status_detail, warning) if value
    )
    query = urlencode({"task_group": job.task_group, "run_id": job.run_id, "label": _job_code(job)})
    progress = job.result.get("progress") if isinstance(job.result.get("progress"), dict) else {}
    progress_label = ""
    if progress:
        progress_label = f"{int(progress.get('completed') or 0)}/{int(progress.get('total') or 0)}"
    return {
        "job": f"./job-results?{query}",
        "job_code": _job_code(job),
        "status": display_status,
        "raw_status": job.status,
        "superseded": bool(job.metadata.get("superseded_by_run_id")),
        "task": task,
        "tool": (
            "RosettaLigand"
            if str(job.tool or "").strip().lower() == "openvs"
            else (
                job.tool
                or _first_value(job.metadata, "md_engine", "tool_id")
                or "Not recorded"
            )
        ),
        "workflow": workflow or "Unspecified",
        "gpu": _gpu_label(job),
        "artifacts": len(job.artifact_manifest.artifacts) if job.artifact_manifest else 0,
        "progress": progress_label,
        "warning": warning,
        "detail": detail,
        "created_at": job.created_at,
        "run_id": job.run_id,
        "task_group": job.task_group,
    }


def collect_unified_job_rows() -> list[dict[str, Any]]:
    return [_job_row(job) for job in iter_job_records(runs_root())]


def _parse_created_at(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _apply_date_filter(frame: pd.DataFrame, selected: str) -> pd.DataFrame:
    if selected == "Any time":
        return frame
    days = {"Today": 1, "Last 7 days": 7, "Last 30 days": 30}[selected]
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    keep = frame["created_at"].map(lambda value: (parsed := _parse_created_at(str(value))) is not None and parsed >= cutoff)
    return frame[keep]


def _apply_history_visibility(
    frame: pd.DataFrame,
    *,
    show_failed_history: bool,
) -> pd.DataFrame:
    if show_failed_history:
        return frame
    superseded = (
        frame["superseded"].fillna(False).astype(bool)
        if "superseded" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    return frame[(frame["raw_status"] != "failed") & ~superseded]


def render() -> None:
    st.title("Jobs")
    jobs = iter_job_records(runs_root())
    rows = [_job_row(job) for job in jobs]
    if not rows:
        st.info("No jobs found yet.")
        return

    frame = pd.DataFrame(rows)
    counts = st.columns(6)
    counts[0].metric("Jobs", len(frame))
    counts[1].metric(
        "Active",
        int(frame["status"].isin(["preparing", "running"]).sum()),
    )
    counts[2].metric("Ready queue", int((frame["status"] == "queued").sum()))
    counts[3].metric(
        "Dependency wait",
        int(frame["status"].isin(["waiting", "blocked"]).sum()),
    )
    counts[4].metric("Failed", int((frame["status"] == "failed").sum()))
    counts[5].metric("Completed", int((frame["status"] == "completed").sum()))
    st.caption(
        "**Active** is executing now · **Queued** is runnable and waiting for a "
        "worker · **Waiting** has unfinished dependencies · **Blocked** cannot "
        "run because an upstream job failed."
    )
    show_failed_history = st.checkbox(
        "Show failed and superseded history",
        value=False,
        key="unified_jobs_show_failed_history",
        help=(
            "Failed and superseded jobs remain stored for provenance and "
            "diagnostics but are hidden from the table by default."
        ),
    )

    filter_row = st.columns([1.5, 1, 1, 1])
    search = filter_row[0].text_input("Search", placeholder="Job code, run ID, task, tool", key="unified_jobs_search").strip()
    selected_statuses = filter_row[1].multiselect(
        "Status", sorted(frame["status"].unique()), key="unified_jobs_status"
    )
    selected_tasks = filter_row[2].multiselect("Task", sorted(frame["task"].unique()), key="unified_jobs_task")
    date_filter = filter_row[3].selectbox(
        "Created",
        ["Any time", "Today", "Last 7 days", "Last 30 days"],
        index=1,
        key="unified_jobs_date",
    )

    detail_row = st.columns(4)
    selected_tools = detail_row[0].multiselect("Tool", sorted(frame["tool"].unique()), key="unified_jobs_tool")
    selected_workflows = detail_row[1].multiselect(
        "Workflow", sorted(frame["workflow"].unique()), key="unified_jobs_workflow"
    )
    selected_gpus = detail_row[2].multiselect("GPU", sorted(frame["gpu"].unique()), key="unified_jobs_gpu")
    warning_filter = detail_row[3].selectbox(
        "Warnings", ["All", "Warnings only", "Without warnings"], key="unified_jobs_warnings"
    )

    visible_frame = _apply_history_visibility(
        frame,
        show_failed_history=show_failed_history,
    )
    hidden_history_count = len(frame) - len(visible_frame)
    filtered = visible_frame.copy()
    for column, selected in (
        ("status", selected_statuses),
        ("task", selected_tasks),
        ("tool", selected_tools),
        ("workflow", selected_workflows),
        ("gpu", selected_gpus),
    ):
        if selected:
            filtered = filtered[filtered[column].isin(selected)]
    filtered = _apply_date_filter(filtered, date_filter)
    if warning_filter == "Warnings only":
        filtered = filtered[filtered["warning"].astype(bool)]
    elif warning_filter == "Without warnings":
        filtered = filtered[~filtered["warning"].astype(bool)]
    if search:
        haystack = filtered[["job_code", "run_id", "task", "tool", "workflow"]].fillna("").astype(str).agg(" ".join, axis=1)
        filtered = filtered[haystack.str.contains(search, case=False, regex=False)]

    history_note = (
        f" · {hidden_history_count} failed/superseded hidden"
        if hidden_history_count
        else ""
    )
    st.caption(
        f"Showing {len(filtered)} of {len(frame)} jobs{history_note}"
    )
    if filtered.empty:
        st.info("No jobs match the current filters.")
        return

    st.dataframe(
        filtered[["job", "status", "task", "tool", "workflow", "progress", "gpu", "artifacts", "detail", "created_at"]],
        hide_index=True,
        width="stretch",
        column_config={
            "job": st.column_config.LinkColumn("Job", display_text=r"label=([^&]+)"),
            "artifacts": st.column_config.NumberColumn("Artifacts", format="%d"),
            "progress": st.column_config.TextColumn("Progress"),
            "detail": st.column_config.TextColumn(
                "State detail / warning",
                width="large",
            ),
            "created_at": st.column_config.TextColumn("Created", width="medium"),
        },
    )

    st.caption("Open a job for results, artifacts, lineage, logs, and supported actions.")
    visible_run_ids = set(visible_frame["run_id"].astype(str))
    actionable = [
        job
        for job in jobs
        if job.run_id in visible_run_ids
        and (
            cancellation_eligibility(job).allowed
            or retry_eligibility(job).allowed
        )
    ]
    st.subheader("Job actions")
    if not actionable:
        st.info("No visible job is currently cancellable or safely retryable.")
        return
    selected_job = st.selectbox(
        "Action target",
        actionable,
        format_func=lambda job: (
            f"{_job_code(job)} | {TASK_LABELS.get(job.task_group, job.task_group)} | {job.status}"
        ),
        key="unified_jobs_action_target",
    )
    cancel_state = cancellation_eligibility(selected_job)
    retry_state = retry_eligibility(selected_job)
    confirm_cancel = st.checkbox(
        "Confirm cancellation",
        key=f"unified_jobs_confirm_cancel_{selected_job.run_id}",
        disabled=not cancel_state.allowed,
    )
    action_columns = st.columns(2)
    if action_columns[0].button(
        "Cancel selected job",
        disabled=not cancel_state.allowed or not confirm_cancel,
        key=f"unified_jobs_cancel_{selected_job.run_id}",
    ):
        try:
            cancelled = request_job_cancellation(selected_job)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success(f"Cancellation requested for {_job_code(cancelled)}.")
    if action_columns[1].button(
        "Retry selected job",
        disabled=not retry_state.allowed,
        key=f"unified_jobs_retry_{selected_job.run_id}",
    ):
        try:
            retried = create_job_retry(selected_job)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.success(
                f"Queued immutable retry {_job_code(retried)} from {_job_code(selected_job)}."
            )
    if not cancel_state.allowed:
        st.caption(f"Cancel unavailable: {cancel_state.reason}")
    if not retry_state.allowed:
        st.caption(f"Retry unavailable: {retry_state.reason}")


render()
