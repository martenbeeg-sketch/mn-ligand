from __future__ import annotations

from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.discover_inputs import (
    bound_ligand_box,
    render_selected_artifacts,
    render_target_viewer,
    select_artifact,
    select_target_artifact,
    target_viewer_path,
)
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.app.pages.search_region import render_search_region
from mn_ligand.core.jobs import display_job_code, iter_job_records
from mn_ligand.core.provenance import target_lineage_summary
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.docking import DEFAULT_DOCKING_IMAGE
from mn_ligand.workflows.redocking import queue_redocking_benchmark


ENGINES = ("Uni-Dock Pro", "AutoDock Vina", "GNINA")
ENGINE_IDS = {"Uni-Dock Pro": "udp", "AutoDock Vina": "vina", "GNINA": "gnina"}


def _initialize_box(target):
    metadata_box = None
    if target is not None:
        center = target.job.metadata.get("center")
        size = target.job.metadata.get("size")
        if isinstance(center, dict) and isinstance(size, dict):
            try:
                metadata_box = {
                    "center": tuple(float(center[axis]) for axis in "xyz"),
                    "size": tuple(float(size[axis]) for axis in "xyz"),
                }
            except (KeyError, TypeError, ValueError):
                metadata_box = None
    ligand_box = None
    if target is not None:
        ligand_key = str(target.job.metadata.get("ligand_key") or "")
        if ligand_key:
            ligand_box = bound_ligand_box(target, ligand_key)
    default_box = metadata_box or ligand_box or {
        "center": (0.0, 0.0, 0.0),
        "size": (22.0, 22.0, 22.0),
    }
    source = "Stored docking box" if metadata_box else "Bound ligand" if ligand_box else "Manual"
    signature = target.job.run_id if target is not None else "none"
    return (
        tuple(float(value) for value in default_box["center"]),
        tuple(float(value) for value in default_box["size"]),
        source,
        signature,
    )


def _rows(statuses: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_jobs = iter_job_records(runs_root())
    jobs_by_id = {job.run_id: job for job in all_jobs}
    for job in all_jobs:
        if job.workflow != "redocking_benchmark" or job.status not in statuses:
            continue
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        query = urlencode({"task_group": job.task_group, "run_id": job.run_id, "label": code})
        context = target_lineage_summary(job, jobs_by_id)
        rows.append(
            {
                "result": f"./job-results?{query}",
                "Last step": context["last_step"],
                "job": code,
                "target": context["target"],
                "receptor": context["receptor"],
                "ligand": context["ligand"],
                "origin / history": context["origin"],
                "status": job.status,
                "engines": ", ".join(job.metadata.get("engines") or ()),
                "replicates": job.metadata.get("replicates"),
                "created": job.created_at,
            }
        )
    return rows


def _render_rows(statuses: set[str]) -> None:
    rows = _rows(statuses)
    if not rows:
        st.info("No matching redocking benchmarks.")
        return
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"result": st.column_config.LinkColumn("Result", display_text="Open")},
    )


def render() -> None:
    st.title("Redocking Benchmark")
    st.caption(
        "Validate pose recovery against a coordinate-bearing experimental ligand. "
        "This is separate from ordinary docking/cofolding campaigns."
    )
    target_tab, reference_tab, engines_tab, run_tab, results_tab = st.tabs(
        ["Target", "Reference ligand", "Engines", "Run", "Results"]
    )
    with target_tab:
        target = select_target_artifact(
            "Prepared experimental target",
            ("prepared_target", "prepared_receptor"),
            key="redocking_target",
            show_viewer=False,
        )
        center, size, source, signature = _initialize_box(target)
        st.markdown("#### Benchmark search box")
        center, size, box_mode, box_padding = render_search_region(
            center=center,
            source_size=size,
            source_label=source,
            key=f"redocking_region_{signature}",
        )
        if target is not None:
            render_selected_artifacts({"Prepared target": target})
            render_target_viewer(
                target,
                viewer_path=target_viewer_path(target),
                box={"center": center, "size": size},
                selected_ligand_key=str(target.job.metadata.get("ligand_key") or ""),
                cartoon_color="#94a3b8",
                ligand_color="cyanCarbon",
                box_color="#0891b2",
                show_box_center=True,
                key="redocking_target_viewer",
            )

    with reference_tab:
        reference = select_artifact(
            "Experimental reference ligand",
            ("prepared_ligand_set",),
            key="redocking_reference",
        )
        if reference is not None:
            render_selected_artifacts({"Reference ligand": reference})
        st.caption(
            "The ligand must retain crystallographic coordinates in the target coordinate frame."
        )

    with engines_tab:
        columns = st.columns(len(ENGINES))
        selected_engines = [
            engine
            for column, engine in zip(columns, ENGINES)
            if column.checkbox(engine, value=True, key=f"redocking_engine_{ENGINE_IDS[engine]}")
        ]
        settings = st.columns(3)
        exhaustiveness = int(
            settings[0].number_input("Exhaustiveness", 1, 1024, 30, key="redocking_exhaustiveness")
        )
        search_mode = settings[1].selectbox(
            "Uni-Dock search mode", ("fast", "balance", "detail"), index=2,
            key="redocking_search_mode",
        )
        poses = int(
            settings[2].number_input("Poses per replicate", 1, 100, 10, key="redocking_poses")
        )
        prep = st.columns(3)
        use_scrub = prep[0].checkbox("Use scrub.py", value=True, key="redocking_use_scrub")
        scrub_ph = float(
            prep[1].number_input(
                "Scrub pH", min_value=0.0, max_value=14.0, value=7.4, step=0.1,
                key="redocking_scrub_ph",
            )
        )
        skip_tautomer = prep[2].checkbox(
            "Skip tautomer enumeration", value=True, key="redocking_skip_tautomer"
        )
        st.caption(
            "The benchmark currently compares the validated Vina, GNINA and Uni-Dock Pro "
            "pose-recovery adapters using symmetry-aware heavy-atom RMSD."
        )

    with run_tab:
        execution = st.columns(3)
        replicates = int(
            execution[0].number_input("Independent runs", 1, 20, 3, key="redocking_replicates")
        )
        seed = int(
            execution[1].number_input(
                "First docking seed", 1, 2_147_483_000, 1001, key="redocking_seed_start"
            )
        )
        gpu = execution[2].selectbox("GPU", ("Automatic", "GPU 0", "GPU 1"), key="redocking_gpu")
        render_run_resources(requires_gpu=True, selected_gpu=gpu, key="redocking")
        blockers = []
        if target is None:
            blockers.append("Select a prepared experimental target.")
            st.link_button("Open Structure Import", "./workspace-structure-preparation")
        if reference is None:
            blockers.append("Select its coordinate-bearing experimental ligand.")
        if not selected_engines:
            blockers.append("Select at least one benchmark engine.")
        for message in blockers:
            st.info(message)
        if st.button(
            "Run redocking benchmark",
            type="primary",
            disabled=bool(blockers),
            key="redocking_queue",
        ) and target is not None and reference is not None:
            receptor_path = target.artifact.resolve(target.job.run_dir, must_exist=True)
            reference_path = reference.artifact.resolve(reference.job.run_dir, must_exist=True)
            if receptor_path is None or reference_path is None:
                st.error("The selected typed artifacts are no longer available.")
            else:
                gpu_device = str(gpu).removeprefix("GPU ") if gpu != "Automatic" else "all"
                try:
                    job = queue_redocking_benchmark(
                        receptor_path=receptor_path,
                        target_artifact=target.artifact,
                        target_task_group=target.job.task_group,
                        reference_ligand_path=reference_path,
                        reference_ligand_artifact=reference.artifact,
                        reference_task_group=reference.job.task_group,
                        center=center,
                        size=size,
                        box_mode=box_mode,
                        box_padding_angstrom=box_padding,
                        engines=[ENGINE_IDS[engine] for engine in selected_engines],
                        replicates=replicates,
                        seed_start=seed,
                        image=DEFAULT_DOCKING_IMAGE,
                        gpu_device=gpu_device,
                        search_mode=search_mode,
                        exhaustiveness=exhaustiveness,
                        poses=poses,
                        use_scrub=use_scrub,
                        scrub_ph=scrub_ph,
                        scrub_skip_tautomer=skip_tautomer,
                    )
                    st.success(
                        "Queued redocking benchmark "
                        f"{display_job_code(job.metadata.get('job_code'), job.run_id)}."
                    )
                except Exception as exc:
                    st.error(str(exc))
        st.markdown("#### Active benchmarks")
        if st.button("Refresh", key="redocking_running_refresh"):
            st.rerun()
        _render_rows({"queued", "preparing", "running"})

    with results_tab:
        st.caption("All redocking benchmark jobs are shown, independent of current inputs.")
        if st.button("Refresh", key="redocking_results_refresh"):
            st.rerun()
        _render_rows({"completed", "failed", "cancelled"})


render()
