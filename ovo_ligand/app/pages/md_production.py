from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from uuid import uuid4

import streamlit as st

from ovo_ligand.app.pages.bound_ligand_md import (
    DEFAULT_MD_IMAGE,
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
from ovo_ligand.app.pages.md import _collect_md_system_prep_jobs


def _host_path_to_container_path(path: Path | None) -> str:
    """Map a host repo path to the container mount path (/ovo-ligand/...)."""
    if path is None:
        return ""
    try:
        resolved = path.resolve()
        repo_root = _run_root().parents[2].resolve()
        if str(resolved).startswith(str(repo_root)):
            rel = resolved.relative_to(repo_root)
            return str(Path("/ovo-ligand") / rel)
    except Exception:
        pass
    return str(path)


def render() -> None:
    st.title("MD Production")
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
        options=list(range(len(prep_jobs))),
        format_func=lambda i: (
            f"{prep_jobs[i]['job_code']} | "
            f"{prep_jobs[i]['pdb_id'] or '-'} | "
            f"{prep_jobs[i]['ligand_key'] or '-'} | "
            f"{prep_jobs[i]['source']}"
        ),
        index=0,
    )
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

    complex_path = Path(selected_job["complex_path"])
    protein_refined_path = next(iter(sorted(complex_path.parent.glob("*_protein_refined.pdb"))), None)
    # Require the NPT-final system snapshot from MD system preparation as production start coordinates.
    output_files = ((prep_result.get("md_result") or {}).get("output_files") or {})
    prep_system_pdb_str = str(output_files.get("system_pdb") or "").strip()
    prep_system_pdb_path = Path(prep_system_pdb_str) if prep_system_pdb_str else None
    if prep_system_pdb_path and not prep_system_pdb_path.exists():
        prep_system_pdb_path = None
    if prep_system_pdb_path is None:
        system_candidates = sorted(md_system_prep_dir.rglob("*_system.pdb"))
        if system_candidates:
            prep_system_pdb_path = system_candidates[0]
    npt_final_path_str = str(output_files.get("npt_pdb") or "").strip()
    npt_final_path = Path(npt_final_path_str) if npt_final_path_str else None
    if npt_final_path and not npt_final_path.exists():
        npt_final_path = None
    if npt_final_path is None:
        npt_candidates = sorted(md_system_prep_dir.rglob("*_npt_final.pdb"))
        if npt_candidates:
            npt_final_path = npt_candidates[0]
    npt_checkpoint_str = str(output_files.get("npt_checkpoint") or "").strip()
    npt_checkpoint_path = Path(npt_checkpoint_str) if npt_checkpoint_str else None
    if npt_checkpoint_path and not npt_checkpoint_path.exists():
        npt_checkpoint_path = None
    if npt_checkpoint_path is None:
        chk_candidates = sorted(md_system_prep_dir.rglob("*_npt_final.chk"))
        if chk_candidates:
            npt_checkpoint_path = chk_candidates[0]
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
            "Selected MD system preparation job has no NPT-final structure (`*_npt_final.pdb`). "
            "Production cannot start."
        )
        st.info(f"Expected in prep run folder: `{md_system_prep_dir}`")
        return
    if use_checkpoint_restart and npt_checkpoint_path is None:
        st.error(
            "Checkpoint restart selected, but no NPT checkpoint (`*_npt_final.chk`) was found."
        )
        st.info("Switch restart mode to `NPT-final PDB (coordinate restart)` or regenerate MD system prep.")
        return
    if use_checkpoint_restart and prep_system_pdb_path is None:
        st.error(
            "Checkpoint restart selected, but no MD system PDB (`*_system.pdb`) was found."
        )
        st.info("Regenerate MD system prep so production can rebuild the exact checkpoint-compatible system.")
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
    st.caption("Production input mapping")
    st.table(
        [
            {
                "role": "Production start coordinates (required)",
                "file": str(production_start_path),
            },
            {
                "role": "Production restart checkpoint",
                "file": str(npt_checkpoint_path) if npt_checkpoint_path else "not available",
            },
            {
                "role": "Checkpoint rebuild system PDB",
                "file": str(prep_system_pdb_path) if prep_system_pdb_path else "not available",
            },
            {"role": "Restart mode", "file": restart_mode},
            {"role": "Production ligand chemistry", "file": str(refined_sdf_path) if refined_sdf_path else "not available"},
            {"role": "Reference SMILES (optional)", "file": str(reference_smi_path) if reference_smi_path else "not available"},
            {"role": "Combined runtime protein file in run dir", "file": "final_input_protein_refined.pdb -> /output/final_input_protein_refined.pdb"},
        ]
    )

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
        preset = st.selectbox("Preset", ["Preview", "Short MD", "Longer MD", "Custom"], index=2)
        production_steps = st.number_input("Production steps", min_value=0, value=50000, step=1000)
    with c2:
        production_report_interval = st.number_input("Production report interval", min_value=100, value=2500, step=100)
        allow_restrained_production = st.checkbox("Allow restrained production", value=False)
    st.caption(
        f"Temperature and pressure are inherited from system preparation: "
        f"{prep_input.get('temperature_k', 300.0)} K, {prep_input.get('pressure', 1.0)} bar."
    )

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
        run_id = str(uuid4())
        output_dir = _run_root() / "bound-ligand-md" / run_id
        output_dir.mkdir(parents=True, exist_ok=False)

        _write_run_metadata(
            output_dir,
            {
                "created_at": _utc_now_iso(),
                "workflow": "bound-ligand-md",
                "status": "running",
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
        if use_checkpoint_restart and npt_checkpoint_path:
            input_payload["resume_from_checkpoint_path"] = _host_path_to_container_path(npt_checkpoint_path)
            input_payload["resume_system_pdb_path"] = _host_path_to_container_path(prep_system_pdb_path)
        else:
            input_payload.pop("resume_from_checkpoint_path", None)
            input_payload.pop("resume_system_pdb_path", None)

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
        with st.expander("Docker command", expanded=True):
            st.code(shlex.join(command))

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            st.success(f"MD production completed: {run_id}")
        else:
            st.error(f"Workflow failed with exit code {result.returncode}")
        st.code(str(output_dir))

        if result_json.exists():
            result_payload = _rewrite_output_paths(json.loads(result_json.read_text()), output_dir)
            metadata = _write_run_metadata(
                output_dir,
                {
                    "status": "completed" if result_payload.get("success") else "failed",
                    "completed_at": _utc_now_iso(),
                    "result_json": str(result_json),
                    "host_run_dir": str(output_dir),
                    "structure_run_id": selected_job.get("structure_run_id") or "",
                    "md_system_prep_run_id": selected_job["run_id"],
                },
            )
            result_payload["metadata"] = metadata
            result_json.write_text(json.dumps(result_payload, indent=2))
            _render_md_results(result_payload, output_dir)
        else:
            _write_run_metadata(
                output_dir,
                {
                    "status": "failed",
                    "completed_at": _utc_now_iso(),
                    "failure_reason": f"process_exit_{result.returncode}",
                    "host_run_dir": str(output_dir),
                    "md_system_prep_run_id": selected_job["run_id"],
                },
            )

        if result.stdout:
            with st.expander("stdout"):
                st.code(result.stdout)
        if result.stderr:
            with st.expander("stderr"):
                st.code(result.stderr)


render()
