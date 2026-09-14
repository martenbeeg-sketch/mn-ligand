from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from mn_ligand.core.diagnostics import run_diagnostics
from mn_ligand.core.formulations import (
    FORMULATION_REGISTRY_ENV,
    load_formulation_registry,
)
from mn_ligand.core.manifests import load_tool_registry
from mn_ligand.core.worker_health import inspect_worker_health
from mn_ligand.runtime import (
    app_home,
    cpu_process_limit,
    cpu_process_limit_setting,
    reference_root,
    runs_root,
    runtime_settings_path,
    save_runtime_settings,
    unidock_pro_max_compounds,
    vina_compound_timeout_minutes,
)


def _path_status(label: str, path: Path) -> dict[str, object]:
    exists = path.is_dir()
    free_gb: float | None = None
    try:
        probe = path if exists else next(parent for parent in path.parents if parent.exists())
        free_gb = shutil.disk_usage(probe).free / (1024**3)
    except (OSError, StopIteration):
        pass
    return {
        "location": label,
        "path": str(path),
        "exists": exists,
        "readable": exists and os.access(path, os.R_OK),
        "writable": exists and os.access(path, os.W_OK),
        "free_GB": round(free_gb, 1) if free_gb is not None else None,
    }


def render() -> None:
    st.title("Settings")

    current_runs = runs_root(create=False)
    current_references = reference_root(create=False)
    st.dataframe(
        pd.DataFrame(
            [
                _path_status("Results and jobs", current_runs),
                _path_status("Reference files", current_references),
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "exists": st.column_config.CheckboxColumn("Exists"),
            "readable": st.column_config.CheckboxColumn("Readable"),
            "writable": st.column_config.CheckboxColumn("Writable"),
            "free_GB": st.column_config.NumberColumn("Free (GB)", format="%.1f"),
        },
    )

    with st.form("runtime_paths"):
        runs_value = st.text_input("Results and jobs directory", value=str(current_runs))
        references_value = st.text_input("Reference files directory", value=str(current_references))
        st.markdown("#### Compute resources")
        process_limit_value = int(
            st.number_input(
                "Default CPU process/worker limit",
                min_value=0,
                max_value=1024,
                value=cpu_process_limit_setting(),
                step=1,
                help=(
                    "Set 0 for automatic host CPU discovery. This limit is used "
                    "by parallel CPU analyses and preparation tasks, including "
                    "AmberTools MMPBSA.py MPI. Effective values are captured in "
                    "immutable job provenance."
                ),
            )
        )
        st.caption(
            f"Current effective limit: {cpu_process_limit()} "
            f"({'automatic' if cpu_process_limit_setting() == 0 else 'configured'})."
        )
        unidock_limit_value = int(
            st.number_input(
                "Uni-Dock Pro maximum compounds per batch",
                min_value=1,
                max_value=1_000_000,
                value=unidock_pro_max_compounds(),
                step=1000,
                help=(
                    "Hard safety limit for one Uni-Dock Pro ligand-index batch. "
                    "Docking submissions above this value are rejected without "
                    "silently dropping compounds."
                ),
            )
        )
        vina_timeout_value = int(
            st.number_input(
                "AutoDock Vina / GNINA timeout per compound (minutes)",
                min_value=1,
                max_value=1440,
                value=vina_compound_timeout_minutes(),
                step=1,
                help=(
                    "A Vina or GNINA compound exceeding this wall-clock limit is recorded "
                    "as excluded. Later replicas in the same job skip it and the "
                    "campaign can finish with partial success."
                ),
            )
        )
        submitted = st.form_submit_button("Save settings", type="primary")

    if submitted:
        try:
            target = save_runtime_settings(
                runs_dir=runs_value,
                reference_dir=references_value,
                cpu_process_limit=process_limit_value,
                unidock_pro_batch_limit=unidock_limit_value,
                vina_compound_timeout=vina_timeout_value,
            )
        except (OSError, ValueError) as exc:
            st.error(f"Could not save runtime paths: {exc}")
        else:
            try:
                config_label = target.relative_to(app_home()).as_posix()
            except ValueError:
                config_label = str(target)
            st.success(f"Runtime paths saved in {config_label}")
            st.warning("Existing jobs are not moved when the results directory changes.")

    st.subheader("Reference readiness")
    expected = (
        ("Boltz models", "boltz_models"),
        ("Shared MSA repository", "boltz_models/msa_repository"),
        ("AlphaFast databases", "alignment"),
        ("AlphaFold 3 weights", "alphafold3"),
    )
    rows = []
    for label, relative_path in expected:
        path = reference_root(create=False) / relative_path
        rows.append(
            {
                "resource": label,
                "relative_path": relative_path,
                "available": path.is_dir(),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"available": st.column_config.CheckboxColumn("Available")},
    )
    st.caption(f"Configuration: {runtime_settings_path()}")

    st.subheader("Compound formulation registry")
    st.caption(
        "Read-only installation registry used to recognize salts, counterions, "
        "solvates and formulation partners in imported compound datasets."
    )
    try:
        formulation_registry = load_formulation_registry()
    except (OSError, TypeError, ValueError) as exc:
        st.error(f"Formulation registry could not be loaded: {exc}")
    else:
        st.caption(
            f"Schema {formulation_registry.schema_version} · "
            f"{len(formulation_registry.components)} components · "
            f"source: {formulation_registry.source}"
        )
        with st.expander("Recognized formulation components", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "ID": component.component_id,
                            "Label": component.label,
                            "Category": component.category,
                            "SMILES variants": len(component.smiles),
                            "SMILES": " | ".join(component.smiles),
                        }
                        for component in formulation_registry.components
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        st.code(
            f"{FORMULATION_REGISTRY_ENV}=/path/to/formulations.yaml",
            language="bash",
        )
        st.caption(
            "Copy the packaged YAML, add a component ID, label, category and "
            "one or more valid SMILES variants, then set the environment variable "
            "and restart the app and workers."
        )

    st.subheader("Worker services")
    st.caption(
        "Streamlit submits typed jobs; independent systemd workers own Docker execution, "
        "logs, cancellation, and GPU leases. Closing this page does not stop a job."
    )
    worker_health = inspect_worker_health(run_dir=current_runs)
    all_worker_services = (*worker_health.services, *worker_health.cpu_services)
    summary_columns = st.columns(4)
    summary_columns[0].metric(
        "Active services", sum(service.active for service in all_worker_services)
    )
    summary_columns[1].metric("Queued jobs", worker_health.queued_jobs)
    summary_columns[2].metric(
        "CPU slots leased",
        f"{worker_health.cpu_slots_leased}/{worker_health.cpu_pool_capacity}",
    )
    summary_columns[3].metric("Active GPU leases", worker_health.active_leases)
    st.dataframe(
        pd.DataFrame([service.to_row() for service in all_worker_services]),
        hide_index=True,
        width="stretch",
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "heartbeat_stale": st.column_config.CheckboxColumn("Stale"),
            "heartbeat_age_s": st.column_config.NumberColumn("Heartbeat age (s)", format="%.1f"),
        },
    )
    unavailable = [service for service in all_worker_services if service.error]
    stale = [
        service
        for service in all_worker_services
        if service.active and service.heartbeat_stale
    ]
    if unavailable:
        st.info(
            "The Streamlit process cannot query the systemd user manager. "
            "Worker queue and lease information is still shown from the run store."
        )
    elif stale:
        st.warning(
            "Active service with a stale worker heartbeat: "
            + ", ".join(service.unit for service in stale)
        )
    elif worker_health.queued_jobs and not any(
        service.active for service in all_worker_services
    ):
        st.warning("Jobs are queued but no configured worker service is active.")
    st.code(
        "mn-ligand worker-service status --gpu-ids 0,1\n"
        "journalctl --user -u 'mn-ligand-worker@*.service' "
        "-u mn-ligand-cpu-worker.service -f",
        language="bash",
    )
    st.button("Refresh worker status", help="Rerun this page to read current service state.")

    st.subheader("Installation diagnostics")
    registry = load_tool_registry()
    gpu_tools = sum(tool.resources.gpu for tool in registry.tools)
    status_counts = Counter(tool.integration_status for tool in registry.tools)
    status_summary = ", ".join(
        f"{status} {count}" for status, count in sorted(status_counts.items())
    )
    st.caption(
        f"Registry schema {registry.schema_version}: {len(registry.tools)} tools "
        f"({gpu_tools} GPU, {len(registry.tools) - gpu_tools} CPU). {status_summary}."
    )
    if st.button("Run diagnostics", help="Checks local paths, Docker, GPUs, images, and reference files."):
        with st.spinner("Checking installation..."):
            diagnostics = run_diagnostics(registry=registry)
        status_icons = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "status": f"{status_icons[item.status]} {item.status}",
                        "check": item.label,
                        "message": item.message,
                    }
                    for item in diagnostics
                ]
            ),
            hide_index=True,
            width="stretch",
        )


render()
