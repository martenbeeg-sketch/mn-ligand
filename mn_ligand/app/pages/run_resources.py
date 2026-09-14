from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from mn_ligand.core.worker_health import WorkerHealthSnapshot, inspect_worker_health
from mn_ligand.runtime import runs_root


def resource_rows(snapshot: WorkerHealthSnapshot) -> list[dict[str, Any]]:
    """Normalize worker health into user-facing GPU availability rows."""
    rows: list[dict[str, Any]] = []
    for service in snapshot.services:
        current_run = service.current_run_id or service.lease_run_id
        state = service.worker_state.strip().lower()
        if not service.active:
            availability = "Offline"
        elif service.heartbeat_stale:
            availability = "Unknown (stale heartbeat)"
        elif service.lease_run_id:
            availability = "Leased"
        else:
            availability = "Free"
        if not service.active:
            worker_slot = "Offline"
        elif service.heartbeat_stale:
            worker_slot = "Unknown"
        elif current_run or state not in {"", "idle", "waiting"}:
            worker_slot = "Busy"
        else:
            worker_slot = "Idle"
        rows.append(
            {
                "GPU": f"GPU {service.gpu_id}",
                "GPU availability": availability,
                "worker": service.worker_id,
                "worker slot": worker_slot,
                "worker state": service.worker_state,
                "current run": current_run or "—",
                "heartbeat age (s)": (
                    round(service.heartbeat_age_seconds, 1)
                    if service.heartbeat_age_seconds is not None
                    else None
                ),
            }
        )
    return rows


def render_run_resources(
    *,
    requires_gpu: bool,
    selected_gpu: str = "Automatic",
    key: str,
) -> WorkerHealthSnapshot:
    """Render queue, CPU, worker, and GPU availability in a workflow Run tab."""
    st.markdown("#### Available resources")
    snapshot = inspect_worker_health(run_dir=runs_root())
    rows = resource_rows(snapshot)
    free_gpus = sum(row["GPU availability"] == "Free" for row in rows)
    summary = st.columns(4)
    summary[0].metric("CPU threads", max(1, os.cpu_count() or 1))
    summary[1].metric("Free GPUs", free_gpus)
    summary[2].metric("Queued jobs", snapshot.queued_jobs)
    summary[3].metric("Active GPU leases", snapshot.active_leases)
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "GPU availability": st.column_config.TextColumn("GPU availability"),
            "heartbeat age (s)": st.column_config.NumberColumn(
                "Heartbeat age (s)", format="%.1f"
            ),
        },
        key=f"{key}_resource_table",
    )
    cpu_rows = [
        {
            "worker": service.worker_id,
            "service": service.service_state,
            "worker state": service.worker_state,
            "current run": service.current_run_id or "—",
            "heartbeat age (s)": (
                round(service.heartbeat_age_seconds, 1)
                if service.heartbeat_age_seconds is not None
                else None
            ),
        }
        for service in snapshot.cpu_services
    ]
    if cpu_rows:
        st.caption("CPU-only worker")
        st.dataframe(
            pd.DataFrame(cpu_rows),
            hide_index=True,
            width="stretch",
            key=f"{key}_cpu_resource_table",
        )
    if requires_gpu:
        st.caption(
            f"Requested GPU: {selected_gpu}. Automatic scheduling uses the first "
            "compatible lease-free GPU worker; CPU-only jobs use the separate CPU worker."
        )
    else:
        st.caption(
            "This workflow is CPU-only. GPU availability is shown for awareness "
            "because other queued jobs share the same worker pool."
        )
    if not any(service.active for service in snapshot.services):
        st.warning("No configured GPU worker service is active.")
    elif any(service.active and service.heartbeat_stale for service in snapshot.services):
        st.warning("At least one active worker has a stale heartbeat.")
    if not any(service.active for service in snapshot.cpu_services):
        st.warning("The CPU-only worker service is not active.")
    return snapshot
