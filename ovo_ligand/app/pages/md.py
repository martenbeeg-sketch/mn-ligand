from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from uuid import uuid4

import streamlit as st

from ovo_ligand.app.pages.bound_ligand_md import (
    DEFAULT_MD_IMAGE,
    PROTOCOL_PRESETS,
    _build_command,
    _build_md_input_payload,
    _ligand_label,
    _parse_protein_chains,
    _read_run_metadata,
    _render_md_results,
    _render_protocol_timeline,
    _render_simulation_input_view,
    _rewrite_output_paths,
    _run_root,
    _short_job_code,
    _utc_now_iso,
    _write_run_metadata,
    parse_bound_ligands,
)


def _steps_to_ns(steps: int | float) -> float:
    return float(steps) * 0.000004


def _collect_structure_jobs() -> list[dict]:
    runs_root = _run_root() / "structure-jobs"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_dir in sorted(
        [p for p in runs_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        metadata = _read_run_metadata(run_dir)
        if not metadata:
            continue
        pdb_id = str(metadata.get("pdb_id", "")).strip()
        complex_candidates = sorted(run_dir.glob(f"{pdb_id.lower()}*_complex_refined.pdb")) if pdb_id else []
        if not complex_candidates:
            complex_candidates = sorted(run_dir.glob("*_complex_refined.pdb"))
        if not complex_candidates:
            continue
        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": metadata.get("job_code") or _short_job_code(run_dir.name),
                "source": metadata.get("source") or "unknown",
                "pdb_id": pdb_id,
                "ligand_key": metadata.get("ligand_key") or "",
                "protein_chains": metadata.get("protein_chains") or [],
                "complex_path": str(complex_candidates[0]),
                "created_at": metadata.get("created_at") or "",
            }
        )
    return rows


def _collect_md_system_prep_jobs() -> list[dict]:
    runs_root = _run_root() / "md-system-prep"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_dir in sorted(
        [p for p in runs_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        metadata = _read_run_metadata(run_dir)
        result_path = run_dir / "result.json"
        input_path = run_dir / "input.json"
        if not metadata or not result_path.exists() or not input_path.exists():
            continue
        try:
            result_payload = json.loads(result_path.read_text())
            input_payload = json.loads(input_path.read_text())
        except Exception:
            continue
        if not bool(result_payload.get("success")):
            continue

        complex_path = str(input_payload.get("prepared_complex_path") or "").strip()
        if not complex_path:
            continue
        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": metadata.get("job_code") or _short_job_code(run_dir.name),
                "source": metadata.get("ligand_source") or "unknown",
                "pdb_id": metadata.get("pdb_id") or input_payload.get("pdb_id") or "",
                "ligand_key": metadata.get("ligand_key") or "",
                "protein_chains": input_payload.get("selected_protein_chains") or [],
                "complex_path": complex_path,
                "created_at": metadata.get("created_at") or "",
                "structure_run_id": metadata.get("structure_run_id") or "",
            }
        )
    return rows


def render() -> None:
    mode = str(st.session_state.get("md_task_mode", "system_prep") or "system_prep")
    is_production_mode = mode == "production"
    st.title("MD Production" if is_production_mode else "MD System Preparation")
    st.caption(
        "Import an MD system preparation job and run production."
        if is_production_mode
        else "Import a prepared structure job, tune equilibration/system parameters, and build reusable MD-ready systems (no production run)."
    )

    structure_jobs = _collect_md_system_prep_jobs() if is_production_mode else _collect_structure_jobs()
    if not structure_jobs:
        st.info(
            "No MD system preparation jobs found. Run MD System Preparation first."
            if is_production_mode
            else "No prepared structure jobs found. Create one first in Structure Preparation."
        )
        if st.button("Go to MD System Preparation" if is_production_mode else "Go to Structure Preparation"):
            st.switch_page("app/pages/md_system_preparation.py" if is_production_mode else "app/pages/structure_preparation.py")
        return

    st.subheader("1. Import prepared MD system" if is_production_mode else "1. Import prepared structure")
    selected_idx = st.selectbox(
        "Prepared MD system job" if is_production_mode else "Prepared structure job",
        options=[None] + list(range(len(structure_jobs))),
        format_func=lambda i: (
            "Select a job..."
            if i is None
            else
            f"{structure_jobs[i]['job_code']} | "
            f"{structure_jobs[i]['pdb_id'] or '-'} | "
            f"{structure_jobs[i]['ligand_key'] or '-'} | "
            f"{structure_jobs[i]['source']}"
        ),
        index=0,
    )
    if selected_idx is None:
        st.info("Select a prepared job to continue.")
        return
    selected_job = structure_jobs[int(selected_idx)]
    complex_path = Path(selected_job["complex_path"])
    protein_refined_path = next(iter(sorted(complex_path.parent.glob("*_protein_refined.pdb"))), None)
    structure_job_dir = complex_path.parent
    try:
        pdb_data = complex_path.read_text()
    except Exception as exc:
        st.error(f"Could not read refined complex: {exc}")
        return
    try:
        protein_pdb_data = protein_refined_path.read_text() if protein_refined_path else pdb_data
    except Exception:
        protein_pdb_data = pdb_data

    ligands = parse_bound_ligands(pdb_data)
    if not ligands:
        st.error("No ligand found in selected prepared complex.")
        return

    default_ligand_idx = 0
    if selected_job["ligand_key"]:
        for i, lig in enumerate(ligands):
            if lig.get("key") == selected_job["ligand_key"]:
                default_ligand_idx = i
                break

    protein_chains = _parse_protein_chains(pdb_data)
    chain_options = [item["chain"] for item in protein_chains]
    chain_defaults = [c for c in selected_job.get("protein_chains", []) if c in chain_options]
    if not chain_defaults and chain_options:
        chain_defaults = chain_options[:1]
    st.caption(f"Imported from: `{complex_path}`")
    st.caption(f"Structure run: `{selected_job['run_id']}`")

    selected_ligand = ligands[int(default_ligand_idx)]
    selected_protein_chains = chain_defaults

    st.subheader("2. Simulation input preview")
    _render_simulation_input_view(pdb_data, selected_ligand, selected_protein_chains)

    st.subheader("3. MD production setup" if is_production_mode else "3. MD system setup")
    preset_name = st.segmented_control("Protocol preset", list(PROTOCOL_PRESETS), default="Preview")
    preset = PROTOCOL_PRESETS[preset_name]
    st.caption(preset["description"])
    pipeline_placeholder = st.container()
    st.markdown("**3.1 Parameterize & System creation**")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        protein_forcefield = st.selectbox("Protein force field", ["amber14-all + tip3p"], index=0, disabled=True)
        forcefield_method = st.selectbox("Ligand force field", ["openff-2.2.0", "openff-2.1.0", "openff-2.0.0"], index=0)
        charge_method = st.selectbox("Ligand charge method", ["gasteiger", "mmff94", "am1bcc"], index=0)
    with col_b:
        box_shape = st.selectbox("Solvent box shape", ["dodecahedron", "cube", "octahedron"], index=0)
        padding_nm = st.number_input("Solvent padding (nm)", min_value=0.1, value=1.0, step=0.1)
    with col_c:
        ionic_strength = st.number_input("Ionic strength (M)", min_value=0.0, value=0.15, step=0.05)
        temperature = st.number_input("Target temperature (K)", min_value=1.0, value=300.0, step=1.0)
        pressure = st.number_input("Target pressure (bar)", min_value=0.1, value=1.0, step=0.1)

    st.markdown("**3.2 Minimize**")
    minimization_only = st.toggle("Stop after minimization", value=False)
    min_col1, min_col2 = st.columns(2)
    with min_col1:
        minimization_max_iterations = st.number_input("Minimization max iterations", min_value=100, value=5000, step=100)
    with min_col2:
        minimization_tolerance_kjmol_nm = st.number_input(
            "Minimization tolerance (kJ/mol/nm)",
            min_value=0.1,
            value=10.0,
            step=0.5,
        )

    st.markdown("**3.3 Equilibration restraints (minimization/heating/NVT)**")
    eq_col1, eq_col2 = st.columns(2)
    with eq_col1:
        apply_protein_restraints_during_heating_nvt = st.checkbox(
            "Apply protein restraints during equilibration",
            value=True,
            help="Active during minimization, heating, and NVT.",
        )
        protein_restraint_selection = st.selectbox(
            "Protein restraint selection",
            ["backbone", "heavy", "none"],
            index=0,
            disabled=not apply_protein_restraints_during_heating_nvt,
        )
        protein_restraint_k = st.number_input(
            "Protein restraint k (kJ/mol/nm²)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            disabled=not apply_protein_restraints_during_heating_nvt,
        )
    with eq_col2:
        ligand_restraints_enabled = st.checkbox(
            "Apply ligand positional restraints during equilibration",
            value=True,
            help="Active during minimization, heating, and NVT.",
        )
        ligand_lock_k = st.number_input(
            "Ligand positional restraint k (kJ/mol/nm²)",
            min_value=0.0,
            value=2500.0,
            step=100.0,
            disabled=not ligand_restraints_enabled,
        )
        enable_ligand_planarity_restraints = st.checkbox(
            "Enable ligand planarity/geometry restraints",
            value=False,
            help="Optional equilibration aid; disabled by default.",
        )
        ligand_planarity_k = st.number_input(
            "Ligand planarity restraint k (kJ/mol/nm²)",
            min_value=0.0,
            value=1500.0,
            step=100.0,
            disabled=not enable_ligand_planarity_restraints,
        )

    st.markdown("**3.4 Heat**")
    heat_col1, heat_col2 = st.columns(2)
    with heat_col1:
        heating_start_temperature = st.number_input("Heating start temperature (K)", min_value=0.0, value=50.0, step=10.0)
    with heat_col2:
        heating_stages = st.number_input("Heating stages", min_value=1, value=6, step=1)
    heating_steps = st.number_input(
        "Heating steps per stage",
        min_value=0,
        value=int(preset["heating_steps_per_stage"]),
        step=250,
    )
    st.caption(f"Estimated heating time (all stages): ~{_steps_to_ns(int(heating_steps) * int(heating_stages)):.3f} ns")

    st.markdown("**3.5 NVT equilibration**")
    nvt_steps = st.number_input("NVT steps", min_value=0, value=int(preset["nvt_steps"]), step=500)
    st.caption(f"Estimated NVT time: ~{_steps_to_ns(int(nvt_steps)):.3f} ns")

    st.markdown("**3.6 NPT equilibration**")
    npt_steps = st.number_input("NPT steps", min_value=0, value=int(preset["npt_steps"]), step=500)
    st.caption(f"Estimated NPT time: ~{_steps_to_ns(int(npt_steps)):.3f} ns")
    npt_release_enabled = st.checkbox(
        "Release restraints during NPT",
        value=True,
        help="Gradually release enabled protein/ligand restraints during NPT.",
        disabled=not (apply_protein_restraints_during_heating_nvt or ligand_restraints_enabled or enable_ligand_planarity_restraints),
    )
    npt_restraint_release_scales = "1.0"
    if apply_protein_restraints_during_heating_nvt or ligand_restraints_enabled or enable_ligand_planarity_restraints:
        npt_restraint_release_scales = st.text_input(
            "NPT ligand restraint release scales",
            value="1.0,0.5,0.2,0.05,0.0",
            help="Comma-separated scale factors (applied to ligand positional restraints).",
            disabled=not npt_release_enabled,
        )
        protein_npt_release_scales = st.text_input(
            "NPT protein restraint release scales",
            value="1.0,0.5,0.1,0.01,0.0",
            help="Comma-separated scale factors (applied to protein restraints).",
            disabled=not npt_release_enabled,
        )
        planarity_npt_release_scales = st.text_input(
            "NPT planarity restraint release scales",
            value="1.0,0.5,0.2,0.05,0.0",
            help="Comma-separated scale factors (applied to planarity restraints).",
            disabled=not npt_release_enabled or not enable_ligand_planarity_restraints,
        )
    else:
        protein_npt_release_scales = "1.0,0.5,0.1,0.01,0.0"
        planarity_npt_release_scales = "1.0,0.5,0.2,0.05,0.0"

    if is_production_mode:
        st.markdown("**3.7 Production policy**")
        allow_restrained_production = st.checkbox(
            "Allow restrained production",
            value=False,
            help="If off (recommended), production is forced unrestrained even if release settings are unsafe.",
        )
    else:
        allow_restrained_production = False

    if is_production_mode:
        production_steps = st.number_input("Production steps", min_value=0, value=int(preset["production_steps"]), step=1000)
        st.caption(f"Estimated production time: ~{_steps_to_ns(int(production_steps)):.3f} ns")
        production_report_interval = st.number_input("Production report interval", min_value=100, value=2500, step=100)
    else:
        production_steps = 0
        production_report_interval = 2500

    with pipeline_placeholder:
        _render_protocol_timeline(
            heating_steps,
            nvt_steps,
            npt_steps,
            production_steps,
            minimization_only,
            include_prepare=False,
            include_energy=False,
            include_production=is_production_mode,
        )

    image = DEFAULT_MD_IMAGE
    use_gpu = True
    if is_production_mode:
        with st.expander("Docker/runtime settings"):
            image = st.text_input("MD Docker image", value=DEFAULT_MD_IMAGE)
            use_gpu = st.checkbox("Use GPU", value=True)

    refined_sdf_path = next(iter(sorted(structure_job_dir.glob("*_ligand_refined.sdf"))), None)
    reference_smi_path = next(iter(sorted(structure_job_dir.glob("*_ligand_ref.smi"))), None)
    refined_sdf_data = ""
    reference_smiles = ""
    try:
        if refined_sdf_path:
            refined_sdf_data = refined_sdf_path.read_text()
        if reference_smi_path:
            reference_smiles = reference_smi_path.read_text().strip()
    except Exception:
        pass

    if is_production_mode:
        with st.expander("Input files used for this MD run", expanded=True):
            st.table(
                [
                    {"role": "Refined complex PDB (structure coordinates)", "file": str(complex_path)},
                    {
                        "role": "Refined protein PDB (MD protein input)",
                        "file": str(protein_refined_path) if protein_refined_path else "not found (fallback to refined complex PDB)",
                    },
                    {
                        "role": "Refined ligand SDF (parameterization input)",
                        "file": str(refined_sdf_path) if refined_sdf_path else "not found (fallback to extracted ligand PDB)",
                    },
                    {
                        "role": "Reference ligand SMILES",
                        "file": str(reference_smi_path) if reference_smi_path else "not found (optional)",
                    },
                    {"role": "Structure job folder", "file": str(structure_job_dir)},
                ]
            )

    preview_payload = _build_md_input_payload(
        selected_job["pdb_id"] or "UNKNOWN",
        protein_pdb_data,
        selected_ligand,
        "preview",
        charge_method,
        forcefield_method,
        heating_steps,
        nvt_steps,
        npt_steps,
        production_steps,
        temperature,
        padding_nm,
        minimization_only,
    )
    preview_payload["box_shape"] = box_shape
    preview_payload["ionic_strength"] = float(ionic_strength)
    preview_payload["pressure"] = float(pressure)
    preview_payload["production_report_interval"] = int(production_report_interval)
    preview_payload["minimization_max_iterations"] = int(minimization_max_iterations)
    preview_payload["minimization_tolerance_kjmol_nm"] = float(minimization_tolerance_kjmol_nm)
    preview_payload["heating_start_temperature"] = float(heating_start_temperature)
    preview_payload["heating_stages"] = int(heating_stages)
    preview_payload["npt_restraint_release_scales"] = str(npt_restraint_release_scales)
    preview_payload["protein_npt_release_scales"] = str(protein_npt_release_scales)
    preview_payload["planarity_npt_release_scales"] = str(planarity_npt_release_scales)
    preview_payload["apply_protein_restraints_during_heating_nvt"] = bool(apply_protein_restraints_during_heating_nvt)
    preview_payload["protein_restraint_selection"] = str(protein_restraint_selection)
    preview_payload["protein_restraint_k"] = float(protein_restraint_k)
    preview_payload["enable_ligand_planarity_restraints"] = bool(enable_ligand_planarity_restraints)
    preview_payload["allow_restrained_production"] = bool(allow_restrained_production)
    preview_payload["force_unrestrained_production"] = not bool(allow_restrained_production)
    preview_payload["ligand_restraints_enabled"] = bool(ligand_restraints_enabled)
    preview_payload["npt_release_enabled"] = bool(npt_release_enabled)
    preview_payload["ligand_lock_k_kjmol_nm2"] = float(ligand_lock_k)
    preview_payload["ligand_planarity_k_kjmol_nm2"] = float(ligand_planarity_k)
    if refined_sdf_data:
        preview_payload["ligand_refined_sdf_data"] = refined_sdf_data
        preview_payload["strict_refined_ligand"] = True
        preview_payload["ligand_refined_sdf_path"] = str(refined_sdf_path)
    if reference_smiles:
        preview_payload["reference_smiles"] = reference_smiles
    if reference_smi_path:
        preview_payload["reference_smiles_path"] = str(reference_smi_path)
    preview_payload["prepared_complex_path"] = str(complex_path)
    st.subheader("4. Run MD production" if is_production_mode else "4. Run MD system preparation")
    if st.button("Run MD production" if is_production_mode else "Run MD system preparation", type="primary"):
        run_id = str(uuid4())
        output_dir = (_run_root() / "bound-ligand-md" / run_id) if is_production_mode else (_run_root() / "md-system-prep" / run_id)
        output_dir.mkdir(parents=True, exist_ok=False)

        _write_run_metadata(
            output_dir,
            {
                "created_at": _utc_now_iso(),
                "workflow": "bound-ligand-md" if is_production_mode else "md-system-prep",
                "status": "running",
                "ligand_source": selected_job.get("source") or "unknown",
                "structure_run_id": selected_job["run_id"],
                "pdb_id": selected_job.get("pdb_id"),
                "ligand_key": selected_ligand.get("key"),
                "ligand_label": _ligand_label(selected_ligand),
                "docker_image": image,
                "use_gpu": bool(use_gpu),
                "run_dir": str(output_dir),
            },
        )

        input_json = output_dir / "input.json"
        result_json = output_dir / "result.json"
        input_payload = _build_md_input_payload(
            selected_job["pdb_id"] or "UNKNOWN",
            protein_pdb_data,
            selected_ligand,
            run_id,
            charge_method,
            forcefield_method,
            heating_steps,
            nvt_steps,
            npt_steps,
            production_steps,
            temperature,
            padding_nm,
            minimization_only,
        )
        input_payload["box_shape"] = box_shape
        input_payload["ionic_strength"] = float(ionic_strength)
        input_payload["pressure"] = float(pressure)
        input_payload["production_report_interval"] = int(production_report_interval)
        input_payload["minimization_max_iterations"] = int(minimization_max_iterations)
        input_payload["minimization_tolerance_kjmol_nm"] = float(minimization_tolerance_kjmol_nm)
        input_payload["heating_start_temperature"] = float(heating_start_temperature)
        input_payload["heating_stages"] = int(heating_stages)
        input_payload["npt_restraint_release_scales"] = str(npt_restraint_release_scales)
        input_payload["protein_npt_release_scales"] = str(protein_npt_release_scales)
        input_payload["planarity_npt_release_scales"] = str(planarity_npt_release_scales)
        input_payload["apply_protein_restraints_during_heating_nvt"] = bool(apply_protein_restraints_during_heating_nvt)
        input_payload["protein_restraint_selection"] = str(protein_restraint_selection)
        input_payload["protein_restraint_k"] = float(protein_restraint_k)
        input_payload["enable_ligand_planarity_restraints"] = bool(enable_ligand_planarity_restraints)
        input_payload["allow_restrained_production"] = bool(allow_restrained_production)
        input_payload["force_unrestrained_production"] = not bool(allow_restrained_production)
        input_payload["ligand_restraints_enabled"] = bool(ligand_restraints_enabled)
        input_payload["npt_release_enabled"] = bool(npt_release_enabled)
        input_payload["ligand_lock_k_kjmol_nm2"] = float(ligand_lock_k)
        input_payload["ligand_planarity_k_kjmol_nm2"] = float(ligand_planarity_k)
        if refined_sdf_data:
            input_payload["ligand_refined_sdf_data"] = refined_sdf_data
            input_payload["strict_refined_ligand"] = True
            input_payload["ligand_refined_sdf_path"] = str(refined_sdf_path)
        if reference_smiles:
            input_payload["reference_smiles"] = reference_smiles
        if reference_smi_path:
            input_payload["reference_smiles_path"] = str(reference_smi_path)
        input_payload["prepared_complex_path"] = str(complex_path)
        # Persist MD input structure as a file in this run folder and keep input.json lightweight.
        final_input_protein_host = output_dir / "final_input_protein_refined.pdb"
        final_input_protein_host.write_text(protein_pdb_data if protein_pdb_data.endswith("\n") else protein_pdb_data + "\n")
        input_payload["input_complex_pdb_path"] = f"/output/{final_input_protein_host.name}"
        input_payload.pop("pdb_data", None)
        input_json.write_text(json.dumps(input_payload, indent=2))
        command = _build_command(image, output_dir, input_json, result_json, use_gpu)
        with st.expander("Docker command", expanded=True):
            st.code(shlex.join(command))

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            st.success(f"MD production completed: {run_id}" if is_production_mode else f"MD system preparation completed: {run_id}")
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
                    "structure_run_id": selected_job["run_id"],
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
                },
            )

        if result.stdout:
            with st.expander("stdout"):
                st.code(result.stdout)
        if result.stderr:
            with st.expander("stderr"):
                st.code(result.stderr)
        st.switch_page("app/pages/jobs_md.py" if is_production_mode else "app/pages/jobs_md_system.py")
        return
