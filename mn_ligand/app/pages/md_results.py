from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.bound_ligand_md import (
    _render_md_results,
    _render_static_line_plot,
    _rewrite_output_paths,
    _run_root,
)
from mn_ligand.core.jobs import display_job_code
from mn_ligand.workflows.md_simulation import (
    create_mmgbsa_analysis_job,
    list_mmgbsa_analysis_jobs,
)
from mn_ligand.workflows.md_engines import (
    GROMACS_ENGINE,
    OPENMM_ENGINE,
    md_engine_spec,
    normalize_md_engine,
)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_total_frames_from_dcd(path: Path) -> int | None:
    try:
        import mdtraj as md
        with md.open(str(path)) as handle:
            return int(len(handle))
    except Exception:
        return None


def _collect_repeat_group_runs(run_dir: Path, metadata: dict) -> list[tuple[Path, dict, dict]]:
    group_id = str(metadata.get("repeat_group_id") or "").strip()
    if not group_id:
        return [(run_dir, metadata, _read_json(run_dir / "result.json"))]
    runs_root = run_dir.parent
    grouped: list[tuple[Path, dict, dict]] = []
    for d in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        md = _read_json(d / "metadata.json")
        if str(md.get("repeat_group_id") or "").strip() != group_id:
            continue
        grouped.append((d, md, _read_json(d / "result.json")))
    grouped.sort(key=lambda item: (int(item[1].get("repeat_index", 0) or 0), str(item[0].name)))
    return grouped or [(run_dir, metadata, _read_json(run_dir / "result.json"))]


def _render_workflow_replica_files(
    result_payload: dict,
    metadata: dict,
) -> None:
    """Keep production replicas as lightweight file handoffs.

    Stability, interaction, and endpoint-energy calculations belong to the
    workflow's immutable ``md-analysis`` child.  Repeating them here made a
    replica route slow and presented a second, potentially confusing analysis
    surface.
    """
    md_result = result_payload.get("md_result") or {}
    output_files = md_result.get("output_files") or {}
    status = str(md_result.get("status") or metadata.get("status") or "unknown")
    st.subheader("Production replica files")
    if result_payload.get("success"):
        st.success(f"Production replica status: {status}")
    else:
        st.warning(f"Production replica status: {status}")
    if output_files:
        rows = []
        for name, path in output_files.items():
            rows.append(
                {
                    "file": name,
                    "path": str(path),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("This replica did not declare output files.")


def _render_mmgbsa_convergence(mmgbsa: dict, analysis_dir: Path) -> None:
    artifacts = (
        mmgbsa.get("artifacts")
        if isinstance(mmgbsa.get("artifacts"), dict)
        else {}
    )
    raw_path = str(artifacts.get("per_frame_csv") or "").strip()
    if not raw_path:
        return
    candidates = [Path(raw_path), analysis_dir / raw_path]
    per_frame_path = next(
        (candidate for candidate in candidates if candidate.is_file()),
        None,
    )
    if per_frame_path is None:
        return
    try:
        frame = pd.read_csv(per_frame_path)
    except Exception as exc:
        st.caption(f"Per-frame endpoint energies could not be read: {exc}")
        return
    total_column = next(
        (
            column
            for column in (
                "delta_G_mmgbsa_kcalmol",
                "DELTA_TOTAL",
                "delta_g_bind_total_kcal_mol",
            )
            if column in frame.columns
        ),
        None,
    )
    if total_column is None or frame.empty:
        return
    values = pd.to_numeric(frame[total_column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return
    frame = frame.loc[valid].copy()
    values = values.loc[valid].astype(float)
    if "time_ps" in frame.columns:
        x_values = (
            pd.to_numeric(frame["time_ps"], errors="coerce")
            .ffill()
            .fillna(0.0)
            / 1000.0
        )
        x_label = "Analyzed trajectory time (ns)"
    else:
        x_values = pd.Series(range(1, len(frame) + 1), index=frame.index)
        x_label = "Analyzed frame"
    convergence = pd.DataFrame(
        {
            "x": x_values.to_numpy(),
            "per_frame_delta_g_kcal_mol": values.to_numpy(),
            "cumulative_mean_delta_g_kcal_mol": values.expanding().mean().to_numpy(),
        }
    ).set_index("x")
    st.markdown("##### Endpoint-energy convergence")
    _render_static_line_plot(
        convergence,
        [
            "per_frame_delta_g_kcal_mol",
            "cumulative_mean_delta_g_kcal_mol",
        ],
        "MM/GBSA estimate and cumulative mean",
        "ΔG estimate (kcal/mol)",
        x_label=x_label,
    )

    block_count = min(10, max(1, len(values) // 5))
    if block_count >= 2:
        block_ids = pd.cut(
            range(len(values)),
            bins=block_count,
            labels=False,
            include_lowest=True,
        )
        block_table = (
            pd.DataFrame(
                {
                    "block": block_ids,
                    "delta_g_kcal_mol": values.to_numpy(),
                }
            )
            .groupby("block", as_index=False)
            .agg(
                mean_delta_g_kcal_mol=("delta_g_kcal_mol", "mean"),
                sample_sd_kcal_mol=("delta_g_kcal_mol", "std"),
                frames=("delta_g_kcal_mol", "size"),
            )
        )
        block_plot = block_table.set_index("block")[
            ["mean_delta_g_kcal_mol"]
        ]
        _render_static_line_plot(
            block_plot,
            ["mean_delta_g_kcal_mol"],
            "Contiguous block means",
            "Mean ΔG estimate (kcal/mol)",
            x_label="Trajectory block",
        )
        with st.expander("MM/GBSA convergence table"):
            st.dataframe(block_table, hide_index=True, width="stretch")
    st.caption(
        "A stable cumulative mean and mutually consistent late block means "
        "support numerical stability of the endpoint estimate; they do not "
        "establish rigorous free-energy convergence."
    )


def _render_repeat_mmgbsa_aggregate(grouped_runs: list[tuple[Path, dict, dict]]) -> None:
    successful: list[dict] = []
    replicate_rows: list[dict] = []
    convergence_jobs = []
    for run_dir, metadata, result_payload in grouped_runs:
        endpoint_jobs = list_mmgbsa_analysis_jobs(run_dir.name)
        latest_completed = next(
            (
                job
                for job in endpoint_jobs
                if job.status == "completed"
                and str((job.result.get("mmgbsa") or {}).get("status") or "")
                == "success"
            ),
            None,
        )
        mm = (
            latest_completed.result.get("mmgbsa") or {}
            if latest_completed is not None
            else result_payload.get("mmgbsa") or {}
        )
        delta = mm.get("delta") or {}
        if str(mm.get("status")) == "success":
            successful.append(delta)
            if latest_completed is not None:
                convergence_jobs.append((metadata, latest_completed, mm))
        replicate_rows.append(
            {
                "replica": metadata.get("repeat_index") or "-",
                "production_results": "./md-results?"
                + urlencode(
                    {
                        "run_type": "bound-ligand-md",
                        "run_id": run_dir.name,
                    }
                ),
                "production_job": display_job_code(
                    metadata.get("job_code"), run_dir.name
                ),
                "endpoint_job": (
                    latest_completed.run_id if latest_completed is not None else ""
                ),
                "status": str(mm.get("status") or "not computed"),
                "frames": (
                    (mm.get("metadata") or {}).get("n_frames_analyzed")
                    if isinstance(mm.get("metadata"), dict)
                    else ""
                ),
                "delta_g_bind_kcal_mol": delta.get(
                    "delta_g_bind_total_kcal_mol"
                ),
            }
        )
    total = len(grouped_runs)
    ok = len(successful)
    st.subheader("Replicated endpoint-energy estimate")
    st.caption(
        f"Successful independent replicas: {ok}/{total}. These MM/GBSA-style "
        "endpoint estimates are trajectory summaries, not rigorous absolute "
        "binding free energies."
    )
    st.dataframe(
        replicate_rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            "production_results": st.column_config.LinkColumn(
                "Replica files", display_text="Open files"
            ),
            "production_job": st.column_config.TextColumn("Production job"),
        },
    )
    if ok == 0:
        st.info("No successful endpoint-energy results are available yet.")
        return

    def _stats(values: list[float]) -> tuple[float, float]:
        if not values:
            return (0.0, float("nan"))
        if len(values) == 1:
            return (float(values[0]), float("nan"))
        return (float(mean(values)), float(stdev(values)))

    metrics = [
        ("delta_g_bind_total_kj_mol", "delta_g_bind_total_kcal_mol", "ΔG_bind total"),
        ("delta_mm_kj_mol", "delta_mm_kcal_mol", "ΔMM"),
        ("delta_gbsa_kj_mol", "delta_gbsa_kcal_mol", "ΔGBSA"),
        ("delta_nonpolar_kj_mol", "delta_nonpolar_kcal_mol", "ΔNonpolar"),
    ]
    rows = []
    for k_kj, k_kcal, label in metrics:
        vals_kj = [float(d[k_kj]) for d in successful if k_kj in d]
        vals_kcal = [float(d[k_kcal]) for d in successful if k_kcal in d]
        if not vals_kj and not vals_kcal:
            continue
        m_kj, s_kj = _stats(vals_kj) if vals_kj else (0.0, float("nan"))
        m_kcal, s_kcal = _stats(vals_kcal) if vals_kcal else (0.0, float("nan"))
        rows.append(
            {
                "term": label,
                "mean_kJ_mol": round(m_kj, 3),
                "sd_kJ_mol": None if math.isnan(s_kj) else round(s_kj, 3),
                "mean_kcal_mol": round(m_kcal, 3),
                "sd_kcal_mol": None if math.isnan(s_kcal) else round(s_kcal, 3),
            }
        )
    if rows:
        st.table(rows)
    if ok < 2:
        st.caption("SD is not available with fewer than 2 successful repeats.")
    if convergence_jobs:
        with st.expander("Per-replica MM/GBSA convergence"):
            for metadata, endpoint_job, mmgbsa in convergence_jobs:
                st.markdown(
                    f"**Replica {metadata.get('repeat_index') or '-'} · "
                    f"{display_job_code(endpoint_job.metadata.get('job_code'), endpoint_job.run_id)}**"
                )
                _render_mmgbsa_convergence(
                    mmgbsa, endpoint_job.run_dir
                )


def _render_mmgbsa_summary_at_end(result_payload: dict, run_dir: Path, metadata: dict, input_payload: dict) -> dict:
    st.subheader("MM/GBSA Analysis (End Summary)")
    analysis_jobs = list_mmgbsa_analysis_jobs(run_dir.name)
    latest_success_job = None
    if analysis_jobs:
        st.markdown("**Post-run evaluations**")
        st.table(
            [
                {
                    "job": display_job_code(job.metadata.get("job_code"), job.run_id),
                    "status": job.status,
                    "engine": ((job.result.get("mmgbsa") or {}).get("method") or
                               ((job.result.get("mmgbsa") or {}).get("backend")) or
                               ((job.metadata.get("resources") or {}).get("tool_id")) or "openmm_md"),
                    "window": (
                        f"{((job.result.get('mmgbsa') or {}).get('start_pct') or (job.metadata.get('parameters') or {}).get('start_pct') or '-')}-"
                        f"{((job.result.get('mmgbsa') or {}).get('end_pct') or (job.metadata.get('parameters') or {}).get('end_pct') or '-')}%"
                    ),
                }
                for job in analysis_jobs
            ]
        )
        latest_success = next(
            (
                job
                for job in analysis_jobs
                if job.status == "completed"
                and str((job.result.get("mmgbsa") or {}).get("status") or "") == "success"
            ),
            None,
        )
        if latest_success is not None:
            latest_success_job = latest_success
            result_payload = dict(result_payload)
            result_payload["mmgbsa"] = latest_success.result.get("mmgbsa") or {}

    default_start_pct = int(input_payload.get("mmgbsa_start_pct", 20))
    default_end_pct = int(input_payload.get("mmgbsa_end_pct", 100))
    default_stride = int(input_payload.get("mmgbsa_stride", 1))
    sel_cols = st.columns(3)
    with sel_cols[0]:
        start_pct = float(st.number_input("Start (%)", min_value=0, max_value=100, value=default_start_pct, step=1, key=f"mmgbsa_start_pct_{run_dir.name}"))
    with sel_cols[1]:
        end_pct = float(st.number_input("End (%)", min_value=0, max_value=100, value=default_end_pct, step=1, key=f"mmgbsa_end_pct_{run_dir.name}"))
    output_files = ((result_payload.get("md_result") or {}).get("output_files") or {})
    traj_candidates = [
        output_files.get("production_trajectory"),
        output_files.get("npt_trajectory"),
    ]
    total_frames = None
    for cand in traj_candidates:
        if cand and Path(str(cand)).exists():
            total_frames = _read_total_frames_from_dcd(Path(str(cand)))
            if total_frames:
                break
    start_pct_clamped = max(0.0, min(100.0, float(start_pct)))
    end_pct_clamped = max(start_pct_clamped, min(100.0, float(end_pct)))
    suggested_stride = default_stride
    analyzed_frames = None
    if total_frames and total_frames > 0:
        start_idx = int((start_pct_clamped / 100.0) * total_frames)
        end_idx = int((end_pct_clamped / 100.0) * total_frames)
        analyzed_frames = max(1, end_idx - start_idx)
        suggested_stride = 1 if analyzed_frames <= 600 else int(math.ceil(analyzed_frames / 600.0))
    stride_key = f"mmgbsa_stride_{run_dir.name}"
    if stride_key not in st.session_state:
        st.session_state[stride_key] = int(suggested_stride)
    with sel_cols[2]:
        stride = int(st.number_input("Sampling stride", min_value=1, value=int(st.session_state[stride_key]), step=1, key=stride_key))
    if analyzed_frames is not None:
        if analyzed_frames <= 600:
            st.info(f"MM/GBSA analysis window has only ~{analyzed_frames} frame(s); using stride 1 is recommended.")
        else:
            st.caption(f"Auto-suggested stride for ~600 analyzed frames: {suggested_stride} (window ~{analyzed_frames} frames)")

    st.markdown("**Tool / Engine**")
    engine_columns = st.columns(3)
    source_engine = normalize_md_engine(
        metadata.get("md_engine")
        or result_payload.get("engine")
        or (result_payload.get("md_result") or {}).get("engine")
        or input_payload.get("md_engine")
        or OPENMM_ENGINE
    )
    default_backend = str(
        input_payload.get("mmgbsa_backend")
        or (
            "g_mmpbsa"
            if source_engine == GROMACS_ENGINE
            else "openmm_gbsa"
        )
    )
    backend_options = list(md_engine_spec(source_engine).endpoint_backends)
    if source_engine == GROMACS_ENGINE:
        backend_options = ["g_mmpbsa"]
    with engine_columns[0]:
        backend = st.selectbox(
            "Endpoint energy engine",
            backend_options,
            index=backend_options.index(default_backend) if default_backend in backend_options else 0,
            key=f"mmgbsa_backend_{run_dir.name}",
        )
    with engine_columns[1]:
        use_gpu = st.checkbox(
            "Use GPU",
            value=bool(metadata.get("use_gpu", True)),
            key=f"mmgbsa_gpu_{run_dir.name}",
        )
    with engine_columns[2]:
        gpu_device = st.selectbox(
            "GPU device",
            ["0", "1", "all"],
            index=0,
            disabled=not use_gpu,
            key=f"mmgbsa_gpu_device_{run_dir.name}",
        )
    st.caption(
        f"Engine: {md_engine_spec(source_engine).label} · Container: "
        f"{metadata.get('docker_image') or md_engine_spec(source_engine).default_image}"
    )
    if st.button("Queue MM/GBSA analysis", type="primary", key=f"queue_mmgbsa_end_{run_dir.name}"):
        try:
            queued = create_mmgbsa_analysis_job(
                run_dir.name,
                start_pct=start_pct,
                end_pct=end_pct,
                stride=stride,
                backend=backend,
                image=str(metadata.get("docker_image") or ""),
                use_gpu=use_gpu,
                gpu_device=gpu_device if use_gpu else "all",
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            st.error(f"Could not queue MM/GBSA analysis: {exc}")
        else:
            code = display_job_code(queued.metadata.get("job_code"), queued.run_id)
            st.success(f"Queued MM/GBSA job {code}. The source MD run is unchanged.")

    def _render_delta_metrics(delta_payload: dict, title: str) -> None:
        st.markdown(f"**{title}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ΔG_bind total", f"{float(delta_payload.get('delta_g_bind_total_kj_mol', 0.0)):.3f} kJ/mol")
        c1.caption(f"{float(delta_payload.get('delta_g_bind_total_kcal_mol', 0.0)):.3f} kcal/mol")
        c2.metric("ΔMM", f"{float(delta_payload.get('delta_mm_kj_mol', 0.0)):.3f} kJ/mol")
        c2.caption(f"{float(delta_payload.get('delta_mm_kcal_mol', 0.0)):.3f} kcal/mol")
        c3.metric("ΔGBSA/PBSA", f"{float(delta_payload.get('delta_gbsa_kj_mol', 0.0)):.3f} kJ/mol")
        c3.caption(f"{float(delta_payload.get('delta_gbsa_kcal_mol', 0.0)):.3f} kcal/mol")
        c4.metric("ΔNonpolar", f"{float(delta_payload.get('delta_nonpolar_kj_mol', 0.0)):.3f} kJ/mol")
        c4.caption(f"{float(delta_payload.get('delta_nonpolar_kcal_mol', 0.0)):.3f} kcal/mol")

    mmgbsa = result_payload.get("mmgbsa") or {}
    status = str(mmgbsa.get("status") or "unknown")
    if status == "success":
        gb_block = mmgbsa.get("gb") or {}
        pb_block = mmgbsa.get("pb") or {}
        if gb_block.get("delta"):
            _render_delta_metrics(gb_block.get("delta") or {}, "MM/GBSA (GB)")
        if pb_block.get("delta"):
            _render_delta_metrics(pb_block.get("delta") or {}, "MM/PBSA (PB)")
        if not gb_block.get("delta") and not pb_block.get("delta"):
            delta = mmgbsa.get("delta") or {}
            _render_delta_metrics(delta, "MM/GBSA")
        st.caption(f"Method: {mmgbsa.get('method', '-')}")
        st.caption(f"Trajectory: {mmgbsa.get('trajectory_path', '-')}")
        st.caption(f"Topology: {mmgbsa.get('topology_path', '-')}")
        artifacts = mmgbsa.get("artifacts") or {}
        if artifacts:
            st.markdown("**Generated Amber files**")
            st.table([{"name": k, "path": v} for k, v in artifacts.items()])
        _render_mmgbsa_convergence(
            mmgbsa,
            latest_success_job.run_dir
            if latest_success_job is not None
            else run_dir,
        )
    elif status == "failed":
        st.error(f"MM/GBSA failed: {mmgbsa.get('error', 'Unknown error')}")
    elif status == "skipped":
        st.warning(f"MM/GBSA skipped: {mmgbsa.get('reason', 'No reason provided')}")
    else:
        st.info("MM/GBSA has not been computed yet for this run.")
    return result_payload


def render() -> None:
    st.title("MD Results")
    qp = st.query_params
    run_id = str(qp.get("run_id", "")).strip()
    run_type = str(qp.get("run_type", "bound-ligand-md")).strip() or "bound-ligand-md"
    run_subdir = "md-system-prep" if run_type == "md-system-prep" else "bound-ligand-md"
    back_page = "app/pages/jobs_md_system.py" if run_subdir == "md-system-prep" else "app/pages/jobs_md.py"
    title_label = "MD System Preparation Results" if run_subdir == "md-system-prep" else "MD Production Results"
    st.title(title_label)
    if not run_id:
        st.info("No run selected. Open this page from the Jobs list.")
        if st.button("Back to Jobs"):
            st.switch_page(back_page)
        return

    run_dir = _run_root() / run_subdir / run_id
    if not run_dir.exists():
        st.error(f"Run not found: {run_id}")
        if st.button("Back to Jobs"):
            st.switch_page(back_page)
        return

    metadata = _read_json(run_dir / "metadata.json")
    result = _read_json(run_dir / "result.json")
    input_payload = _read_json(run_dir / "input.json")

    top = st.columns([0.85, 0.15])
    with top[0]:
        st.caption(f"Run: {run_id}")
        if metadata.get("pdb_id"):
            st.caption(f"PDB: {metadata.get('pdb_id')}")
    with top[1]:
        if st.button("Back to Jobs"):
            st.switch_page(back_page)

    if not result:
        st.warning("No result.json available for this run yet.")
        return

    rewritten = _rewrite_output_paths(result, run_dir)
    is_workflow_replica = bool(
        metadata.get("workflow_id") or metadata.get("workflow_parent_run_id")
    )
    if run_subdir == "bound-ligand-md" and is_workflow_replica:
        _render_workflow_replica_files(rewritten, metadata)
    elif run_subdir == "bound-ligand-md":
        grouped_runs = _collect_repeat_group_runs(run_dir, metadata)
        if len(grouped_runs) > 1:
            st.caption(f"Repeat group: {metadata.get('repeat_group_id')} ({len(grouped_runs)} runs)")
            _render_repeat_mmgbsa_aggregate(grouped_runs)
        tab_labels = []
        for rd, md, _ in grouped_runs:
            idx = md.get("repeat_index")
            label = f"Repeat {idx}" if idx else rd.name[:8]
            tab_labels.append(label)
        tabs = st.tabs(tab_labels)
        for tab, (rd, md, raw_result) in zip(tabs, grouped_runs):
            with tab:
                if not raw_result:
                    st.warning(f"No result.json available for run {rd.name}")
                    continue
                rw = _rewrite_output_paths(raw_result, rd)
                in_payload = _read_json(rd / "input.json")
                _render_md_results(rw, rd)
                _render_mmgbsa_summary_at_end(rw, rd, md, in_payload)
    else:
        _render_md_results(rewritten, run_dir)


render()
