from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.rescoring import (
    create_pose_selection_job,
    queue_boltzina_rescoring_job,
    queue_gnina_rescoring_job,
    source_pose_rows,
)


def _source_jobs() -> dict[str, JobRecord]:
    options: dict[str, JobRecord] = {}
    for job in iter_job_records(runs_root()):
        if job.status != "completed":
            continue
        if job.workflow not in {"docking_campaign", "openvs_docking"}:
            continue
        if not any((job.run_dir / "results").glob("**/*_out.pdbqt")):
            continue
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        engine = str(job.tool or job.metadata.get("engine") or "Docking")
        compounds = int(job.metadata.get("compound_count") or 0)
        label = f"{code} · {engine} · {compounds} compounds"
        options[label] = job
    return options


def _boltzina_contexts(source_job: JobRecord | None) -> dict[str, Path]:
    if source_job is None:
        return {}
    target_run_id = str(
        source_job.metadata.get("prepared_target_run_id")
        or source_job.metadata.get("parent_run_id")
        or ""
    )
    options: dict[str, Path] = {}
    for job in iter_job_records(runs_root(), task_groups=("refolding",)):
        if job.status != "completed" or job.workflow != "boltz2_refolding":
            continue
        if target_run_id and str(job.parent_run_id) != target_run_id:
            continue
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        for manifest in sorted(job.run_dir.glob("output/**/processed/manifest.json")):
            work_dir = manifest.parent.parent
            replicate = next(
                (
                    part.replace("replicate_", "replicate ")
                    for part in work_dir.parts
                    if part.startswith("replicate_")
                ),
                "context",
            )
            options[f"{code} · {replicate}"] = work_dir
    return options


def _result_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for job in iter_job_records(runs_root(), task_groups=("rescoring",)):
        if job.workflow not in {
            "pose_selection",
            "gnina_rescoring",
            "boltzina_rescoring",
        }:
            continue
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        query = urlencode(
            {"task_group": job.task_group, "run_id": job.run_id, "label": code}
        )
        rows.append(
            {
                "result": f"./job-results?{query}",
                "job": code,
                "kind": (
                    "selection" if job.workflow == "pose_selection" else "rescoring"
                ),
                "engine": job.tool,
                "source engine": job.metadata.get("source_engine", ""),
                "poses": job.metadata.get("pose_count", ""),
                "status": job.status,
                "created": job.created_at,
            }
        )
    return rows


def render() -> None:
    st.title("Rescoring")
    st.caption(
        "Apply one or several scoring models to immutable poses from a completed "
        "docking job. Pose coordinates, source scores, ranks, replicates, and "
        "provenance remain linked."
    )
    source_tab, poses_tab, engines_tab, run_tab, results_tab = st.tabs(
        ["Source results", "Poses", "Engines", "Run", "Results"]
    )

    sources = _source_jobs()
    with source_tab:
        if sources:
            source_label = st.selectbox(
                "Completed pose-producing job",
                list(sources),
                key="rescoring_source_job",
            )
            source_job = sources[source_label]
            source_code = display_job_code(
                source_job.metadata.get("job_code"), source_job.run_id
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "job": source_code,
                            "engine": source_job.tool,
                            "compounds": source_job.metadata.get(
                                "compound_count", ""
                            ),
                            "replicates": source_job.metadata.get("replicates", 1),
                            "target job": source_job.metadata.get(
                                "prepared_target_run_id",
                                source_job.parent_run_id,
                            ),
                        }
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.selectbox(
                "Completed pose-producing job",
                ["No compatible completed docking jobs"],
                disabled=True,
                key="rescoring_source_job_missing",
            )
            st.info("Complete a docking job with stored native PDBQT poses first.")
            source_job = None

    available = source_pose_rows(source_job) if source_job is not None else []
    compound_ids = sorted({str(row["compound_id"]) for row in available})
    replicate_ids = sorted({int(row["replicate"]) for row in available})
    rank_ids = sorted({int(row["source_pose_rank"]) for row in available})
    with poses_tab:
        selection_mode = st.radio(
            "Compound selection",
            ("All compounds", "Manual subset"),
            horizontal=True,
            key="rescoring_compound_mode",
        )
        if selection_mode == "Manual subset":
            selected_compounds = st.multiselect(
                "Compounds",
                compound_ids,
                default=compound_ids[: min(10, len(compound_ids))],
                key="rescoring_compounds",
            )
        else:
            selected_compounds = compound_ids
            st.caption(f"All {len(compound_ids)} compounds are selected.")
        selected_replicates = st.multiselect(
            "Independent replicates",
            replicate_ids,
            default=replicate_ids,
            key="rescoring_replicates",
        )
        pose_scope = st.radio(
            "Stored poses per compound and replicate",
            ("Top-ranked pose", "All stored poses", "Selected ranks"),
            horizontal=True,
            key="rescoring_pose_scope",
        )
        if pose_scope == "Top-ranked pose":
            selected_ranks = [1] if rank_ids else []
        elif pose_scope == "All stored poses":
            selected_ranks = rank_ids
        else:
            selected_ranks = st.multiselect(
                "Pose ranks",
                rank_ids,
                default=rank_ids[:1],
                key="rescoring_pose_ranks",
            )
        selected_count = sum(
            str(row["compound_id"]) in selected_compounds
            and int(row["replicate"]) in selected_replicates
            and int(row["source_pose_rank"]) in selected_ranks
            for row in available
        )
        metrics = st.columns(4)
        metrics[0].metric("Selected poses", selected_count)
        metrics[1].metric("Compounds", len(selected_compounds))
        metrics[2].metric("Replicates", len(selected_replicates))
        metrics[3].metric("Pose ranks", len(selected_ranks))
        if available:
            preview = pd.DataFrame(
                [
                    {key: value for key, value in row.items() if not key.startswith("_")}
                    for row in available
                    if str(row["compound_id"]) in selected_compounds
                    and int(row["replicate"]) in selected_replicates
                    and int(row["source_pose_rank"]) in selected_ranks
                ]
            )
            st.dataframe(preview, hide_index=True, width="stretch")

    contexts = _boltzina_contexts(source_job)
    with engines_tab:
        st.markdown("#### Coordinate-preserving scoring engines")
        engine_columns = st.columns(2)
        use_gnina = engine_columns[0].checkbox(
            "GNINA score-only",
            value=True,
            key="rescoring_engine_gnina",
            help=(
                "Evaluates the exact stored coordinates with GNINA empirical and "
                "CNN scores. No docking search or minimization is performed."
            ),
        )
        use_boltzina = engine_columns[1].checkbox(
            "Boltzina",
            value=False,
            disabled=not contexts,
            key="rescoring_engine_boltzina",
            help=(
                "Scores the exact docked pose with Boltz-2's affinity machinery "
                "while omitting Boltz-2 structure generation."
            ),
        )
        st.caption(
            "Both engines consume the same immutable pose selection. Their values "
            "have different definitions and should be compared as a consensus panel, "
            "not averaged as if they shared units."
        )
        if not contexts:
            st.info(
                "Boltzina additionally requires a completed Boltz-2 context for the "
                "same prepared target. GNINA remains available."
            )
        gnina_rotation = int(
            st.number_input(
                "GNINA CNN rotations",
                min_value=0,
                max_value=24,
                value=0,
                key="rescoring_gnina_rotation",
                help="Zero uses the validated GNINA default.",
            )
        )
        if contexts:
            context_label = st.selectbox(
                "Boltz-2 context for Boltzina",
                list(contexts),
                key="rescoring_boltzina_context",
            )
            boltz_work_dir = contexts[context_label]
        else:
            boltz_work_dir = None
        affinity_mw_correction = st.checkbox(
            "Boltzina molecular-weight correction",
            value=False,
            disabled=not contexts,
            key="rescoring_boltzina_mw",
        )

    with run_tab:
        selected_gpu = st.selectbox(
            "GPU", ("Automatic", "GPU 0", "GPU 1"), key="rescoring_gpu"
        )
        render_run_resources(
            requires_gpu=use_gnina or use_boltzina,
            selected_gpu=selected_gpu,
            key="rescoring",
        )
        selected_engines = [
            name
            for name, enabled in (
                ("GNINA score-only", use_gnina),
                ("Boltzina", use_boltzina),
            )
            if enabled
        ]
        st.write(
            ", ".join(selected_engines) if selected_engines else "No engines selected."
        )
        blockers: list[str] = []
        if source_job is None:
            blockers.append("Select a completed docking job.")
        if not selected_count:
            blockers.append("Select at least one stored pose.")
        if not selected_engines:
            blockers.append("Select at least one rescoring engine.")
        if use_boltzina and boltz_work_dir is None:
            blockers.append("Select a compatible Boltz-2 context for Boltzina.")
        for blocker in blockers:
            st.info(blocker)
        queue_clicked = st.button(
            "Queue selected rescoring engines",
            type="primary",
            disabled=bool(blockers),
            key="rescoring_queue",
        )
        if queue_clicked and source_job is not None:
            try:
                selection_job = create_pose_selection_job(
                    source_job,
                    compound_ids=selected_compounds,
                    replicates=selected_replicates,
                    pose_ranks=selected_ranks,
                )
                gpu_device = (
                    "all"
                    if selected_gpu == "Automatic"
                    else selected_gpu.replace("GPU ", "")
                )
                queued: list[JobRecord] = []
                if use_gnina:
                    queued.append(
                        queue_gnina_rescoring_job(
                            selection_job=selection_job,
                            gpu_device=gpu_device,
                            cnn_rotation=gnina_rotation,
                        )
                    )
                if use_boltzina and boltz_work_dir is not None:
                    queued.append(
                        queue_boltzina_rescoring_job(
                            selection_job=selection_job,
                            boltz_work_dir=boltz_work_dir,
                            gpu_device=gpu_device,
                            affinity_mw_correction=affinity_mw_correction,
                        )
                    )
                codes = [
                    display_job_code(item.metadata.get("job_code"), item.run_id)
                    for item in queued
                ]
                st.success(
                    f"Created immutable pose selection "
                    f"{display_job_code(selection_job.metadata.get('job_code'), selection_job.run_id)} "
                    f"and queued {len(queued)} engine job(s): {', '.join(codes)}."
                )
            except Exception as exc:
                st.error(f"Could not queue rescoring: {exc}")

    with results_tab:
        rows = _result_rows()
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "result": st.column_config.LinkColumn(
                        "Result", display_text="Open"
                    )
                },
            )
        else:
            st.info("No rescoring jobs yet.")


render()
