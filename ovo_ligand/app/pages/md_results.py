from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from statistics import mean, stdev

import streamlit as st

from ovo_ligand.app.pages.bound_ligand_md import (
    DEFAULT_MD_IMAGE,
    _render_md_results,
    _rewrite_output_paths,
    _run_root,
    _repo_root,
)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _build_mmgbsa_recompute_command(
    image: str,
    run_dir: Path,
    use_gpu: bool,
    start_pct: float,
    end_pct: float,
    stride: int,
    backend: str,
) -> list[str]:
    command = ["docker", "run", "--rm"]
    shm_size = os.getenv("OVO_MD_DOCKER_SHM_SIZE", "64g").strip()
    if shm_size:
        command += ["--shm-size", shm_size]
    if use_gpu:
        command += ["--gpus", "all"]
    command += [
        "-v",
        f"{_repo_root()}:/ovo-ligand:ro",
        "-v",
        f"{run_dir}:/output",
        "-e",
        "PYTHONPATH=/ovo-ligand",
        image,
        "python",
        "-m",
        "ovo_ligand.workflows.bound_ligand_md",
        "mmgbsa",
        "--input",
        "/output/input.json",
        "--result",
        "/output/result.json",
        "--output",
        "/output/result.json",
        "--start-pct",
        str(start_pct),
        "--end-pct",
        str(end_pct),
        "--stride",
        str(stride),
        "--backend",
        str(backend),
    ]
    return command


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


def _render_repeat_mmgbsa_aggregate(grouped_runs: list[tuple[Path, dict, dict]]) -> None:
    successful = []
    for _, _, result_payload in grouped_runs:
        mm = (result_payload.get("mmgbsa") or {})
        if str(mm.get("status")) == "success":
            successful.append(mm.get("delta") or {})
    total = len(grouped_runs)
    ok = len(successful)
    st.subheader("MM/GBSA Repeat Aggregate")
    st.caption(f"Successful repeats: {ok}/{total}")
    if ok == 0:
        st.info("No successful MM/GBSA repeat results available for aggregation yet.")
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


def _render_mmgbsa_summary_at_end(result_payload: dict, run_dir: Path, metadata: dict, input_payload: dict) -> dict:
    st.subheader("MM/GBSA Analysis (End Summary)")
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

    input_json = run_dir / "input.json"
    result_json = run_dir / "result.json"
    if st.button("Recompute MM/GBSA", type="primary", key=f"recompute_mmgbsa_end_{run_dir.name}"):
        if not input_json.exists() or not result_json.exists():
            st.error("This run is missing input.json or result.json; cannot recompute MM/GBSA.")
        else:
            image = str(metadata.get("docker_image") or DEFAULT_MD_IMAGE)
            use_gpu = bool(metadata.get("use_gpu", True))
            cmd = _build_mmgbsa_recompute_command(
                image=image,
                run_dir=run_dir,
                use_gpu=use_gpu,
                start_pct=start_pct,
                end_pct=end_pct,
                stride=stride,
                backend=str(input_payload.get("mmgbsa_backend", "openmm_gbsa")),
            )
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                st.success("MM/GBSA recompute finished.")
                result_payload = _read_json(result_json)
            else:
                st.error(f"MM/GBSA recompute failed with exit code {proc.returncode}")
            if proc.stdout:
                with st.expander("stdout"):
                    st.code(proc.stdout)
            if proc.stderr:
                with st.expander("stderr"):
                    st.code(proc.stderr)

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
    if run_subdir == "bound-ligand-md":
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
