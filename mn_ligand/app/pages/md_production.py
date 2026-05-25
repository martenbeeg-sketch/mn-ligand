from __future__ import annotations

import json
import math
import os
from pathlib import Path
from uuid import uuid4

import streamlit as st

from mn_ligand.app.pages.bound_ligand_md import (
    DEFAULT_MD_IMAGE,
    PROTOCOL_PRESETS,
    _build_command,
    _build_md_input_payload,
    _ligand_label,
    _render_md_results,
    _render_simulation_input_view,
    _rewrite_output_paths,
    _run_root,
    _utc_now_iso,
    _write_run_metadata,
    parse_bound_ligands,
    _parse_protein_chains,
)
from mn_ligand.app.pages.common import (
    queue_gpu_job,
    resolve_run_artifact_path,
    try_dispatch_next_queued_gpu_job,
)
from mn_ligand.app.pages.md import _collect_md_system_prep_jobs


def _steps_to_ns(steps: int | float) -> float:
    return float(steps) * 0.000004


def _ensure_ligand_resname_lig(pdb_text: str) -> str:
    out: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 20:
            fixed = line[:17] + f"{'LIG':>3}" + line[20:]
            if fixed.startswith("ATOM  "):
                fixed = "HETATM" + fixed[6:]
            out.append(fixed)
        else:
            out.append(line)
    return "\n".join(out).rstrip() + "\n"


def _try_rebuild_complex_with_ligand(run_dir: Path, protein_pdb_data: str) -> str | None:
    ligand_raw_pdb = next(iter(sorted(run_dir.glob("*_ligand_raw.pdb"))), None)
    if ligand_raw_pdb is None or not ligand_raw_pdb.exists():
        return None
    try:
        lig_text = _ensure_ligand_resname_lig(ligand_raw_pdb.read_text())
        lig_lines = [ln for ln in lig_text.splitlines() if ln.startswith(("HETATM", "ATOM", "CONECT"))]
        if not lig_lines:
            return None
        merged = protein_pdb_data.rstrip() + "\n" + "\n".join(lig_lines) + "\nEND\n"
        if parse_bound_ligands(merged):
            return merged
    except Exception:
        return None
    return None


PRODUCTION_REPORT_INTERVAL_PRESETS = {
    "Preview": 500,
    "Short MD": 1000,
    "Longer MD": 2500,
    "Custom": 2500,  # Custom starts from Longer MD defaults
}


def _host_path_to_container_path(path: Path | None) -> str:
    """Map a host repo path to the container mount path (/mn-ligand/...)."""
    if path is None:
        return ""
    try:
        resolved = path.resolve()
        repo_root = _run_root().parents[2].resolve()
        if str(resolved).startswith(str(repo_root)):
            rel = resolved.relative_to(repo_root)
            return str(Path("/mn-ligand") / rel)
    except Exception:
        pass
    return str(path)


def render() -> None:
    try_dispatch_next_queued_gpu_job()
    st.title("MD Production")
    st.caption("MD Production UI revision: 2026-05-07-preset-fix-r4")
    st.caption("Select a prepared MD system, review the carried-over system parameters, inspect the exact structure, then run production.")

    prep_jobs = _collect_md_system_prep_jobs()
    if not prep_jobs:
        st.info("No MD system preparation jobs found. Run MD System Preparation first.")
        if st.button("Go to MD System Preparation"):
            st.switch_page("app/pages/md_system_preparation.py")
        return

    st.subheader("1. Select prepared MD system")
    selected_idx = st.selectbox(
        "MD system preparation job",
        options=[None] + list(range(len(prep_jobs))),
        format_func=lambda i: (
            "Select a prepared MD system job..."
            if i is None
            else
            f"{prep_jobs[i]['job_code']} | "
            f"{prep_jobs[i]['pdb_id'] or '-'} | "
            f"{prep_jobs[i]['ligand_key'] or '-'} | "
            f"{prep_jobs[i]['source']}"
        ),
        index=0,
    )
    if selected_idx is None:
        st.info("Select a prepared MD system job to continue.")
        return
    selected_job = prep_jobs[int(selected_idx)]

    md_system_prep_dir = _run_root() / "md-system-prep" / selected_job["run_id"]
    prep_input_json = md_system_prep_dir / "input.json"
    prep_result_json = md_system_prep_dir / "result.json"
    prep_input = {}
    prep_result = {}
    try:
        if prep_input_json.exists():
            prep_input = json.loads(prep_input_json.read_text())
        if prep_result_json.exists():
            prep_result = json.loads(prep_result_json.read_text())
    except Exception:
        prep_input = {}
        prep_result = {}
    # Normalize any container-style output paths (/output/...) to host-visible run paths.
    prep_result = _rewrite_output_paths(prep_result, md_system_prep_dir)

    complex_path = resolve_run_artifact_path(selected_job["complex_path"], must_exist=True)
    if complex_path is None:
        st.error("Could not resolve selected prepared complex path from metadata.")
        return
    protein_refined_path = next(iter(sorted(complex_path.parent.glob("*_protein_refined.pdb"))), None)
    # Require the NPT-final system snapshot from MD system preparation as production start coordinates.
    output_files = ((prep_result.get("md_result") or {}).get("output_files") or {})
    prep_system_pdb_str = str(output_files.get("system_pdb") or "").strip()
    prep_system_pdb_path = resolve_run_artifact_path(prep_system_pdb_str, must_exist=True) if prep_system_pdb_str else None
    npt_final_path_str = str(output_files.get("npt_pdb") or "").strip()
    npt_final_path = resolve_run_artifact_path(npt_final_path_str, must_exist=True) if npt_final_path_str else None
    npt_checkpoint_str = str(output_files.get("npt_checkpoint") or "").strip()
    npt_checkpoint_path = resolve_run_artifact_path(npt_checkpoint_str, must_exist=True) if npt_checkpoint_str else None
    npt_state_xml_str = str(output_files.get("npt_state_xml") or "").strip()
    npt_state_xml_path = resolve_run_artifact_path(npt_state_xml_str, must_exist=True) if npt_state_xml_str else None
    npt_system_xml_str = str(output_files.get("npt_system_xml") or "").strip()
    npt_system_xml_path = resolve_run_artifact_path(npt_system_xml_str, must_exist=True) if npt_system_xml_str else None
    npt_integrator_xml_str = str(output_files.get("npt_integrator_xml") or "").strip()
    npt_integrator_xml_path = resolve_run_artifact_path(npt_integrator_xml_str, must_exist=True) if npt_integrator_xml_str else None
    restart_mode = st.selectbox(
        "Restart mode",
        options=["Checkpoint (exact continuation)", "NPT-final PDB (coordinate restart)"],
        index=0,
        help=(
            "Checkpoint is exact continuation and recommended. "
            "NPT-final PDB rebuilds the system and reinitializes state."
        ),
    )
    use_checkpoint_restart = restart_mode.startswith("Checkpoint")

    if npt_final_path is None:
        st.error(
            "Selected MD system preparation job is missing a valid `md_result.output_files.npt_pdb` artifact. "
            "Production cannot start."
        )
        st.info(
            "Re-run MD system preparation and ensure `result.json` contains host-visible output files via "
            "`md_result.output_files`."
        )
        st.code(f"Prep result file: {prep_result_json}")
        return
    if use_checkpoint_restart and npt_checkpoint_path is None:
        st.error(
            "Checkpoint restart selected, but no valid `md_result.output_files.npt_checkpoint` artifact was found."
        )
        st.info("Switch restart mode to `NPT-final PDB (coordinate restart)` or regenerate MD system prep.")
        st.code(f"Prep result file: {prep_result_json}")
        return
    if use_checkpoint_restart and prep_system_pdb_path is None:
        st.error(
            "Checkpoint restart selected, but no valid `md_result.output_files.system_pdb` artifact was found."
        )
        st.info("Regenerate MD system prep so production can rebuild the exact checkpoint-compatible system.")
        st.code(f"Prep result file: {prep_result_json}")
        return
    production_start_path = npt_final_path
    try:
        complex_pdb_data = complex_path.read_text()
    except Exception as exc:
        st.error(f"Could not read refined complex: {exc}")
        return

    try:
        protein_pdb_data = production_start_path.read_text() if production_start_path else complex_pdb_data
    except Exception:
        protein_pdb_data = complex_pdb_data

    # For UI preview, try to parse ligands directly from production-start structure first.
    ligands = parse_bound_ligands(protein_pdb_data)
    if not ligands:
        ligands = parse_bound_ligands(complex_pdb_data)
    if not ligands:
        rebuilt = _try_rebuild_complex_with_ligand(complex_path.parent, protein_pdb_data)
        if rebuilt:
            complex_pdb_data = rebuilt
            ligands = parse_bound_ligands(complex_pdb_data)
    if not ligands:
        st.error("No ligand found in selected prepared complex.")
        return

    selected_ligand = ligands[0]
    if selected_job.get("ligand_key"):
        for lig in ligands:
            if lig.get("key") == selected_job["ligand_key"]:
                selected_ligand = lig
                break

    selected_protein_chains = prep_input.get("selected_protein_chains") or selected_job.get("protein_chains") or []
    if not selected_protein_chains:
        parsed = _parse_protein_chains(protein_pdb_data) or _parse_protein_chains(complex_pdb_data)
        selected_protein_chains = [parsed[0]["chain"]] if parsed else []
    refined_sdf_path = next(iter(sorted(complex_path.parent.glob("*_ligand_refined.sdf"))), None)
    reference_smi_path = next(iter(sorted(complex_path.parent.glob("*_ligand_ref.smi"))), None)

    st.caption(f"MD system prep run: `{selected_job['run_id']}`")
    st.caption(f"Source structure job complex: `{complex_path}`")
    st.caption(f"Production start coordinates (NPT-final PDB): `{production_start_path}`")
    st.caption(
        f"Production start state (checkpoint): `{npt_checkpoint_path if npt_checkpoint_path else 'not used / not found'}`"
    )
    if not use_checkpoint_restart:
        st.warning(
            "Coordinate restart mode selected: production will rebuild from NPT-final PDB. "
            "This is less exact than checkpoint continuation."
        )

    st.subheader("2. Prepared system summary")
    summary_cols = st.columns(4)
    summary_cols[0].metric("PDB", str(selected_job.get("pdb_id") or "UNKNOWN"))
    summary_cols[1].metric("Ligand", str(selected_ligand.get("resname") or "-"))
    summary_cols[2].metric("Chain(s)", ", ".join(selected_protein_chains) if selected_protein_chains else "-")
    summary_cols[3].metric("Prep preset", str(prep_input.get("protocol_preset") or "custom"))
    st.table(
        [
            {"parameter": "Protein force field", "value": "amber14-all + tip3p"},
            {"parameter": "Ligand force field", "value": str(prep_input.get("forcefield_method", "openff-2.2.0"))},
            {"parameter": "Ligand charge method", "value": str(prep_input.get("charge_method", "gasteiger"))},
            {"parameter": "Solvent box shape", "value": str(prep_input.get("box_shape", "dodecahedron"))},
            {"parameter": "Padding (nm)", "value": str(prep_input.get("padding_nm", 1.0))},
            {"parameter": "Ionic strength (M)", "value": str(prep_input.get("ionic_strength", 0.15))},
            {"parameter": "Temperature (K)", "value": str(prep_input.get("temperature_k", 300.0))},
            {"parameter": "Pressure (bar)", "value": str(prep_input.get("pressure", 1.0))},
        ]
    )
    mm_backend = str(prep_input.get("mmgbsa_backend") or "openmm_gbsa")
    amber_complex_prmtop = str(output_files.get("amber_complex_prmtop") or "").strip()
    amber_complex_inpcrd = str(output_files.get("amber_complex_inpcrd") or "").strip()
    amber_runtime_ready = bool(
        mm_backend == "ambertools_mmpbsa"
        and amber_complex_prmtop
        and amber_complex_inpcrd
        and (resolve_run_artifact_path(amber_complex_prmtop, must_exist=True) is not None)
        and (resolve_run_artifact_path(amber_complex_inpcrd, must_exist=True) is not None)
    )
    st.markdown("**Runtime Topology Mode**")
    if amber_runtime_ready:
        st.success("Amber-native runtime selected: production will use prepared Amber topology files.")
        st.caption(f"Amber `complex.prmtop`: `{amber_complex_prmtop}`")
        st.caption(f"Amber `complex.inpcrd`: `{amber_complex_inpcrd}`")
    elif mm_backend == "ambertools_mmpbsa":
        st.warning(
            "Amber MM/GBSA backend is selected, but Amber runtime topology files are missing. "
            "Production will fall back to OpenMM/PDB runtime."
        )
    else:
        st.info("OpenMM runtime selected: production will use the OpenMM/PDB workflow.")
    st.caption(f"Restart mode: `{restart_mode}`")

    st.subheader("3. Structure used for production")
    st.caption("This preview is built from the production-start coordinates file above (selected chain scope + selected ligand).")
    try:
        _render_simulation_input_view(protein_pdb_data, selected_ligand, selected_protein_chains)
    except Exception as exc:
        st.warning(f"Interactive structure viewer could not be rendered: {exc}")
        st.info(
            "Production will still use the selected protein coordinates and refined ligand chemistry shown above."
        )

    st.subheader("4. Production run settings")
    c1, c2 = st.columns(2)
    with c1:
        preset = st.selectbox("Preset", ["Preview", "Short MD", "Longer MD", "Custom"], index=2, key="md_prod_preset")
        # Custom intentionally starts from Longer MD defaults.
        preset_source = "Longer MD" if preset == "Custom" else preset
        default_steps = int((PROTOCOL_PRESETS.get(preset_source) or {}).get("production_steps", 500000))
        default_report = int(PRODUCTION_REPORT_INTERVAL_PRESETS.get(preset, 2500))
        steps_state_key = "md_prod_production_steps"
        report_state_key = "md_prod_report_interval"
        preset_applied_key = "md_prod_preset_applied"
        bootstrap_key = "md_prod_bootstrap_done"
        if not st.session_state.get(bootstrap_key, False):
            st.session_state[steps_state_key] = default_steps
            st.session_state[report_state_key] = default_report
            st.session_state[preset_applied_key] = preset
            st.session_state[bootstrap_key] = True
            st.rerun()
        if st.session_state.get(preset_applied_key) != preset:
            st.session_state[steps_state_key] = default_steps
            st.session_state[report_state_key] = default_report
            st.session_state[preset_applied_key] = preset
            st.rerun()
        production_steps = st.number_input(
            "Production steps",
            min_value=0,
            value=default_steps,
            step=1000,
            key=steps_state_key,
        )
        if st.button("Reset Production Defaults", key="md_prod_reset_defaults"):
            st.session_state[steps_state_key] = default_steps
            st.session_state[report_state_key] = default_report
            st.session_state[preset_applied_key] = preset
            st.rerun()
        st.caption(f"Estimated production time: ~{_steps_to_ns(int(production_steps)):.3f} ns")
    with c2:
        production_report_interval = st.number_input(
            "Production report interval",
            min_value=100,
            value=default_report,
            step=100,
            key=report_state_key,
        )
        allow_restrained_production = st.checkbox("Allow restrained production", value=False)
        repeat_count = int(st.number_input("Repetitions", min_value=1, value=1, step=1, key="md_prod_repeat_count"))
    st.caption(
        f"Temperature and pressure are inherited from system preparation: "
        f"{prep_input.get('temperature_k', 300.0)} K, {prep_input.get('pressure', 1.0)} bar."
    )
    st.caption(f"MM/GBSA backend from prepared system: `{mm_backend}`")
    st.markdown("**MM/GBSA at end of run**")
    mmgbsa_enabled = st.checkbox("Run MM/GBSA after MD production", value=True)
    mmc1, mmc2, mmc3 = st.columns(3)
    with mmc1:
        mmgbsa_start_pct = int(st.number_input("Start (%)", min_value=0, max_value=100, value=20, step=1))
    with mmc2:
        mmgbsa_end_pct = int(st.number_input("End (%)", min_value=0, max_value=100, value=100, step=1))
    expected_total_frames = max(1, int(production_steps) // max(1, int(production_report_interval)))
    start_pct_clamped = max(0, min(100, int(mmgbsa_start_pct)))
    end_pct_clamped = max(start_pct_clamped, min(100, int(mmgbsa_end_pct)))
    start_idx = int((start_pct_clamped / 100.0) * expected_total_frames)
    end_idx = int((end_pct_clamped / 100.0) * expected_total_frames)
    analyzed_frames = max(1, end_idx - start_idx)
    suggested_stride = 1 if analyzed_frames <= 600 else int(math.ceil(analyzed_frames / 600.0))
    stride_key = "md_prod_mmgbsa_stride"
    if stride_key not in st.session_state:
        st.session_state[stride_key] = suggested_stride
    with mmc3:
        mmgbsa_stride = int(st.number_input("Sampling stride", min_value=1, value=int(st.session_state[stride_key]), step=1, key=stride_key))
    if analyzed_frames <= 600:
        st.info(f"MM/GBSA analysis window has only ~{analyzed_frames} frame(s); using stride 1 is recommended.")
    else:
        st.caption(f"Auto-suggested stride for ~600 analyzed frames: {suggested_stride} (window ~{analyzed_frames} frames)")

    with st.expander("Docker/runtime settings"):
        image = st.text_input("MD Docker image", value=DEFAULT_MD_IMAGE)
        use_gpu = st.checkbox("Use GPU", value=True)

    refined_sdf_data = ""
    reference_smiles = ""
    try:
        if refined_sdf_path:
            refined_sdf_data = refined_sdf_path.read_text()
        if reference_smi_path:
            reference_smiles = reference_smi_path.read_text().strip()
    except Exception:
        pass

    st.subheader("5. Run MD production")
    if st.button("Run MD production", type="primary"):
        repeat_group_id = str(uuid4())
        queued_runs: list[dict[str, str]] = []
        for repeat_idx in range(repeat_count):
            run_id = str(uuid4())
            output_dir = _run_root() / "bound-ligand-md" / run_id
            output_dir.mkdir(parents=True, exist_ok=False)

            _write_run_metadata(
                output_dir,
                {
                    "created_at": _utc_now_iso(),
                    "workflow": "bound-ligand-md",
                    "status": "queued",
                    "ligand_source": selected_job.get("source") or "unknown",
                    "structure_run_id": selected_job.get("structure_run_id") or "",
                    "md_system_prep_run_id": selected_job["run_id"],
                    "pdb_id": selected_job.get("pdb_id"),
                    "ligand_key": selected_ligand.get("key"),
                    "ligand_label": _ligand_label(selected_ligand),
                    "docker_image": image,
                    "use_gpu": bool(use_gpu),
                    "run_dir": str(output_dir),
                    "preset": preset,
                    "restart_mode": restart_mode,
                    "repeat_group_id": repeat_group_id,
                    "repeat_index": repeat_idx + 1,
                    "repeat_total": repeat_count,
                },
            )

            input_payload = _build_md_input_payload(
                selected_job.get("pdb_id") or "UNKNOWN",
                protein_pdb_data,
                selected_ligand,
                run_id,
                prep_input.get("charge_method", "gasteiger"),
                prep_input.get("forcefield_method", "openff-2.2.0"),
                0,
                0,
                0,
                int(production_steps),
                float(prep_input.get("temperature_k", 300.0)),
                float(prep_input.get("padding_nm", 1.0)),
                False,
            )
            input_payload["protocol_preset"] = preset
            input_payload["box_shape"] = prep_input.get("box_shape", "dodecahedron")
            input_payload["ionic_strength"] = float(prep_input.get("ionic_strength", 0.15))
            input_payload["pressure"] = float(prep_input.get("pressure", 1.0))
            input_payload["production_report_interval"] = int(production_report_interval)
            input_payload["allow_restrained_production"] = bool(allow_restrained_production)
            input_payload["force_unrestrained_production"] = not bool(allow_restrained_production)
            input_payload["prepared_complex_path"] = str(complex_path)
            input_payload["source_md_system_prep_run_id"] = selected_job["run_id"]
            input_payload["source_md_system_prep_result_json"] = str(prep_result_json)
            # Keep all workflow outputs inside this run folder mounted at /output.
            input_payload["output_dir"] = "/output"
            input_payload["restart_mode"] = restart_mode
            input_payload["mmgbsa_enabled"] = bool(mmgbsa_enabled)
            input_payload["mmgbsa_backend"] = str(mm_backend)
            input_payload["mmgbsa_start_pct"] = int(mmgbsa_start_pct)
            input_payload["mmgbsa_end_pct"] = int(mmgbsa_end_pct)
            input_payload["mmgbsa_stride"] = int(mmgbsa_stride)
            input_payload["mmpbsa_use_mpi"] = bool(mm_backend == "ambertools_mmpbsa")
            cpu_default = max(1, min(32, int(os.cpu_count() or 8)))
            input_payload["mmpbsa_mpi_cores"] = int(cpu_default)
            input_payload["repeat_group_id"] = repeat_group_id
            input_payload["repeat_index"] = repeat_idx + 1
            input_payload["repeat_total"] = repeat_count
            input_payload["restart_contract"] = {
                "npt_pdb": str(npt_final_path) if npt_final_path else None,
                "npt_checkpoint": str(npt_checkpoint_path) if npt_checkpoint_path else None,
                "npt_state_xml": str(npt_state_xml_path) if npt_state_xml_path else None,
                "npt_system_xml": str(npt_system_xml_path) if npt_system_xml_path else None,
                "npt_integrator_xml": str(npt_integrator_xml_path) if npt_integrator_xml_path else None,
                "system_pdb": str(prep_system_pdb_path) if prep_system_pdb_path else None,
            }
            if use_checkpoint_restart and npt_checkpoint_path:
                input_payload["resume_from_checkpoint_path"] = _host_path_to_container_path(npt_checkpoint_path)
                input_payload["resume_system_pdb_path"] = _host_path_to_container_path(prep_system_pdb_path)
                if npt_state_xml_path:
                    input_payload["resume_state_xml_path"] = _host_path_to_container_path(npt_state_xml_path)
                if npt_system_xml_path:
                    input_payload["resume_system_xml_path"] = _host_path_to_container_path(npt_system_xml_path)
                if npt_integrator_xml_path:
                    input_payload["resume_integrator_xml_path"] = _host_path_to_container_path(npt_integrator_xml_path)
            else:
                input_payload.pop("resume_from_checkpoint_path", None)
                input_payload.pop("resume_system_pdb_path", None)
                input_payload.pop("resume_state_xml_path", None)
                input_payload.pop("resume_system_xml_path", None)
                input_payload.pop("resume_integrator_xml_path", None)

            if refined_sdf_data:
                input_payload["ligand_refined_sdf_data"] = refined_sdf_data
                input_payload["strict_refined_ligand"] = True
                input_payload["ligand_refined_sdf_path"] = str(refined_sdf_path)
            if reference_smiles:
                input_payload["reference_smiles"] = reference_smiles
            if reference_smi_path:
                input_payload["reference_smiles_path"] = str(reference_smi_path)

            final_input_protein_host = output_dir / "final_input_protein_refined.pdb"
            final_input_protein_host.write_text(protein_pdb_data if protein_pdb_data.endswith("\n") else protein_pdb_data + "\n")
            input_payload["input_complex_pdb_path"] = f"/output/{final_input_protein_host.name}"
            input_payload.pop("pdb_data", None)

            input_json = output_dir / "input.json"
            result_json = output_dir / "result.json"
            input_json.write_text(json.dumps(input_payload, indent=2))

            command = _build_command(image, output_dir, input_json, result_json, use_gpu)
            queue_gpu_job(output_dir, "bound-ligand-md", run_id, command)
            queued_runs.append(
                {
                    "run_id": run_id,
                    "run_dir": str(output_dir),
                    "repeat_label": f"{repeat_idx + 1}/{repeat_count}",
                }
            )

        st.success(f"Submitted {len(queued_runs)} repeat run(s) in group `{repeat_group_id}`.")
        for item in queued_runs:
            st.caption(f"Repeat {item['repeat_label']}: `{item['run_id']}`")

        # Try to start the first queued job immediately if GPU is currently free.
        dispatched = try_dispatch_next_queued_gpu_job()
        if dispatched:
            count = int(dispatched.get("count", 1) or 1)
            last = dispatched.get("last") or {}
            st.info(
                f"Dispatched {count} queued run(s). "
                f"Last: {last.get('run_id', 'unknown')} "
                f"({last.get('workflow', 'bound-ligand-md')})."
            )
        st.switch_page("app/pages/jobs_md.py")
        return


render()
