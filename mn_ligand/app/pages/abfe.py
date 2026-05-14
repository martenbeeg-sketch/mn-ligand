from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from mn_ligand.app.pages.bound_ligand_md import _render_structure_view, _run_root
from mn_ligand.app.pages.common import _run_docker_workflow, WORKFLOWS, try_dispatch_next_queued_gpu_job
from mn_ligand.workflows.bound_ligand_md import parse_bound_ligands

ABFE_PRESETS = {
    "fast": {
        "production_length_ns": 0.5,
        "equilibration_length_ns": 0.1,
        "protocol_repeats": 1,
        "n_replicas_complex": 30,
        "n_replicas_solvent": 14,
    },
    "balanced": {
        "production_length_ns": 5.0,
        "equilibration_length_ns": 0.5,
        "protocol_repeats": 3,
        "n_replicas_complex": 30,
        "n_replicas_solvent": 14,
    },
    "production": {
        "production_length_ns": 10.0,
        "equilibration_length_ns": 1.0,
        "protocol_repeats": 3,
        "n_replicas_complex": 30,
        "n_replicas_solvent": 14,
    },
}

RBFE_PRESETS = {
    "fast": {
        "lambda_windows": 11,
        "production_length_ns": 0.5,
        "equilibration_length_ns": 0.1,
        "protocol_repeats": 1,
    },
    "balanced": {
        "lambda_windows": 11,
        "production_length_ns": 2.0,
        "equilibration_length_ns": 0.5,
        "protocol_repeats": 3,
    },
    "production": {
        "lambda_windows": 11,
        "production_length_ns": 5.0,
        "equilibration_length_ns": 1.0,
        "protocol_repeats": 3,
    },
}

ABFE_PRESET_HELP = {
    "fast": "Quick smoke test. Lowest runtime, lowest statistical confidence.",
    "balanced": "Good default for iterative work. Better precision at moderate runtime.",
    "production": "Best confidence for reporting. Longest runtime.",
}

RBFE_PRESET_HELP = {
    "fast": "Quick network check and sanity run.",
    "balanced": "Recommended default for most RBFE campaigns.",
    "production": "Higher confidence for final ranking/reporting.",
}

def _estimate_abfe_timeout_hours(settings: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    fast_mode = bool(settings.get("fast_mode", True))
    prod_ns = float(settings.get("production_length_ns", 0.5 if fast_mode else 10.0))
    equil_ns = float(settings.get("equilibration_length_ns", 0.1 if fast_mode else 1.0))
    repeats = max(1, int(settings.get("protocol_repeats", 1 if fast_mode else 3)))
    replicas_complex = max(1, int(settings.get("n_replicas_complex", 30)))
    replicas_solvent = max(1, int(settings.get("n_replicas_solvent", 14)))
    user_timeout = settings.get("dag_timeout_hours")
    if user_timeout is not None:
        timeout_h = max(0.25, float(user_timeout))
        return timeout_h, {"source": "user_override"}

    base_timeout_h = 6.0 if fast_mode else 48.0
    ref_prod_ns, ref_equil_ns, ref_repeats = (0.5, 0.1, 1) if fast_mode else (10.0, 1.0, 3)
    ref_repl_c, ref_repl_s = 30.0, 14.0

    requested_units = repeats * (2.0 * prod_ns + 0.5 * equil_ns)
    reference_units = max(0.1, ref_repeats * (2.0 * ref_prod_ns + 0.5 * ref_equil_ns))
    base_scale = max(1.0, requested_units / reference_units)
    replica_scale = ((float(replicas_complex) / ref_repl_c) + (float(replicas_solvent) / ref_repl_s)) / 2.0
    replica_scale = max(0.5, replica_scale)
    scale_factor = max(1.0, base_scale * replica_scale)
    timeout_h = min(336.0, max(base_timeout_h, base_timeout_h * scale_factor * 1.75))
    return timeout_h, {"source": "auto_scaled", "scale_factor": scale_factor, "replica_scale": replica_scale}

def _estimate_abfe_runtime_hours(settings: dict[str, Any], timeout_hours: float) -> tuple[float, float]:
    """Heuristic walltime estimate range for UI guidance (not a hard guarantee)."""
    fast_mode = bool(settings.get("fast_mode", True))
    # ABFE jobs typically complete well before timeout; keep a broad range.
    if fast_mode:
        low_frac, high_frac = 0.25, 0.60
    else:
        low_frac, high_frac = 0.35, 0.80
    low = max(0.1, timeout_hours * low_frac)
    high = max(low, timeout_hours * high_frac)
    return low, high


def _apply_abfe_preset_to_state(preset: str) -> None:
    base = ABFE_PRESETS.get(preset, ABFE_PRESETS["fast"])
    st.session_state["openfe_abfe_prod_ns"] = float(base["production_length_ns"])
    st.session_state["openfe_abfe_eq_ns"] = float(base["equilibration_length_ns"])
    st.session_state["openfe_abfe_repeats"] = int(base["protocol_repeats"])
    st.session_state["openfe_abfe_rep_complex"] = int(base["n_replicas_complex"])
    st.session_state["openfe_abfe_rep_solvent"] = int(base["n_replicas_solvent"])
    st.session_state.setdefault("openfe_abfe_hmr", True)


def _apply_rbfe_preset_to_state(preset: str) -> None:
    base = RBFE_PRESETS.get(preset, RBFE_PRESETS["fast"])
    st.session_state["openfe_rbfe_lambda"] = int(base["lambda_windows"])
    st.session_state["openfe_rbfe_prod_ns"] = float(base["production_length_ns"])
    st.session_state["openfe_rbfe_eq_ns"] = float(base["equilibration_length_ns"])
    st.session_state["openfe_rbfe_repeats"] = int(base["protocol_repeats"])
    st.session_state.setdefault("openfe_rbfe_hmr", True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _collect_structure_jobs() -> list[dict[str, Any]]:
    runs_root = _run_root() / "structure-jobs"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        meta = _read_json(run_dir / "metadata.json")
        if str(meta.get("status", "")).lower() != "completed":
            continue
        pdb_id = str(meta.get("pdb_id") or "").lower()
        protein = next(iter(sorted(run_dir.glob(f"{pdb_id}*_protein_refined.pdb"))), None) if pdb_id else None
        if protein is None:
            protein = next(iter(sorted(run_dir.glob("*_protein_refined.pdb"))), None)
        ligand = next(iter(sorted(run_dir.glob("*_ligand_refined.sdf"))), None)
        complex_pdb = next(iter(sorted(run_dir.glob(f"{pdb_id}*_complex_refined.pdb"))), None) if pdb_id else None
        if complex_pdb is None:
            complex_pdb = next(iter(sorted(run_dir.glob("*_complex_refined.pdb"))), None)
        if not protein or not ligand:
            continue
        job_code = str(meta.get("job_code") or run_dir.name[:3]).upper()
        label = f"{job_code} | {meta.get('pdb_id', '-') } | {meta.get('ligand_key', '-')}"
        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": job_code,
                "label": label,
                "pdb_id": str(meta.get("pdb_id") or ""),
                "ligand_key": str(meta.get("ligand_key") or ""),
                "protein_refined": str(protein),
                "ligand_refined_sdf": str(ligand),
                "complex_refined": str(complex_pdb) if complex_pdb else "",
            }
        )
    return rows


def _load_text(path: str) -> str:
    try:
        return Path(path).read_text()
    except Exception:
        return ""


def _ensure_streamlit_help_icons_visible() -> None:
    st.markdown(
        """
<style>
/* Force Streamlit help tooltip icon visibility next to labels */
[data-testid="stTooltipIcon"] {
  display: inline-flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  color: currentColor !important;
}
[data-testid="stTooltipIcon"] svg {
  display: inline-block !important;
  visibility: visible !important;
  opacity: 1 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_selected_structure_preview(key_prefix: str, selected: dict[str, Any]) -> None:
    complex_pdb = _load_text(str(selected.get("complex_refined") or ""))
    if not complex_pdb:
        st.info("No refined complex PDB found for this job.")
        return
    ligands = parse_bound_ligands(complex_pdb)
    lig_key = str(selected.get("ligand_key") or "")
    chosen = None
    for lig in ligands:
        if str(lig.get("key") or "") == lig_key:
            chosen = lig
            break
    if chosen is None and ligands:
        chosen = ligands[0]
    if chosen is None:
        st.info("No ligand records found in refined complex.")
        return
    st.markdown("#### Selected structure preview")
    _render_structure_view(
        complex_pdb,
        ligands,
        chosen,
        [],
        show_molstar_tools=True,
        key_suffix=f"openfe_{key_prefix}_{selected.get('run_id')}",
        title="Refined complex (from structure job)",
        caption="Selected structure job output reused directly for OpenFE input.",
    )


def _run_abfe(selected: dict[str, Any]) -> None:
    settings = st.session_state.get("openfe_abfe_settings", {})
    job_code = str(selected.get("job_code") or "ABF")
    timeout_hours, _ = _estimate_abfe_timeout_hours(settings)
    params = {
        "abfe_container": WORKFLOWS["abfe"]["defaults"]["abfe_container"],
        "protein_pdb": str(selected["protein_refined"]),
        "ligand_sdf": str(selected["ligand_refined_sdf"]),
        "protocol": "quick-test",
        "protocol_settings": settings,
        "metadata": {
            "job_code": job_code,
            "structure_run_id": str(selected.get("run_id") or ""),
            "pdb_id": str(selected.get("pdb_id") or ""),
            "ligand_key": str(selected.get("ligand_key") or ""),
            "run_inputs": {
                "preset": str(st.session_state.get("openfe_abfe_preset", "fast")),
                "production_length_ns": float(settings.get("production_length_ns", 0.0)),
                "equilibration_length_ns": float(settings.get("equilibration_length_ns", 0.0)),
                "protocol_repeats": int(settings.get("protocol_repeats", 1)),
                "charge_method": str(settings.get("charge_method", "")),
                "ligand_forcefield": str(settings.get("ligand_forcefield", "")),
                "solvent_model": str(settings.get("solvent_model", "")),
                "temperature_k": float(settings.get("temperature", 0.0)),
                "pressure_bar": float(settings.get("pressure", 0.0)),
                "hmr": bool(float(settings.get("hydrogen_mass", 1.0)) > 1.5),
                "timestep_fs": float(settings.get("timestep_fs", 0.0)),
                "timeout_budget_hours_est": float(timeout_hours),
            },
        },
    }
    run = _run_docker_workflow("abfe", WORKFLOWS["abfe"], params)
    if run.get("queued"):
        st.success(f"OpenFE ABFE queued: {run['run_id']}")
    else:
        if run["returncode"] == 0:
            st.success(f"OpenFE ABFE completed: {run['run_id']}")
        else:
            st.error(f"OpenFE ABFE failed with exit code {run['returncode']}")
        if run.get("stderr"):
            with st.expander("stderr"):
                st.code(run["stderr"])
    st.switch_page("app/pages/jobs_openfe.py")


def _run_rbfe(selected_jobs: list[dict[str, Any]]) -> None:
    import tempfile

    first = selected_jobs[0]
    protein_path = str(first["protein_refined"])
    ligands = []
    for item in selected_jobs:
        lig_path = str(item["ligand_refined_sdf"])
        lig_text = _load_text(lig_path).strip()
        if lig_text:
            ligands.append(lig_text)
    if len(ligands) < 2:
        st.error("RBFE needs at least 2 ligands with readable refined SDF files.")
        return
    combined = "\n$$$$\n".join([x.split("$$$$")[0].strip() for x in ligands]) + "\n$$$$\n"
    tmp_dir = _run_root() / "openfe" / "rbfe-inputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".sdf", prefix="rbfe_ligands_", dir=str(tmp_dir), delete=False) as handle:
        handle.write(combined)
        ligands_path = handle.name

    settings = st.session_state.get("openfe_rbfe_settings", {})
    mapping = st.session_state.get("openfe_rbfe_mapping", {})
    network = st.session_state.get("openfe_rbfe_network", {})
    params = {
        "rbfe_container": WORKFLOWS["rbfe"]["defaults"]["rbfe_container"],
        "protein_pdb": protein_path,
        "ligands_sdf": ligands_path,
        "mapper": str(mapping.get("atom_mapper", "kartograf")),
        "protocol": "quick-test",
        "protocol_settings": settings,
        "network_topology": str(network.get("network_topology", "mst")),
        "central_ligand": str(network.get("central_ligand", "")) or None,
        "atom_mapper": str(mapping.get("atom_mapper", "kartograf")),
        "atom_map_hydrogens": bool(mapping.get("atom_map_hydrogens", True)),
        "lomap_max3d": float(mapping.get("lomap_max3d", 1.0)),
        "metadata": {
            "job_code": "RBF",
            "structure_run_id": str(first.get("run_id") or ""),
            "pdb_id": str(first.get("pdb_id") or ""),
            "ligand_key": ",".join([str(item.get("ligand_key") or "") for item in selected_jobs]),
            "run_inputs": {
                "preset": str(st.session_state.get("openfe_rbfe_preset", "fast")),
                "n_ligands": int(len(selected_jobs)),
                "network_topology": str(network.get("network_topology", "mst")),
                "atom_mapper": str(mapping.get("atom_mapper", "kartograf")),
                "lambda_windows": int(settings.get("lambda_windows", 11)),
                "production_length_ns": float(settings.get("production_length_ns", 0.0)),
                "equilibration_length_ns": float(settings.get("equilibration_length_ns", 0.0)),
                "protocol_repeats": int(settings.get("protocol_repeats", 1)),
                "charge_method": str(settings.get("charge_method", "")),
                "ligand_forcefield": str(settings.get("ligand_forcefield", "")),
                "solvent_model": str(settings.get("solvent_model", "")),
                "temperature_k": float(settings.get("temperature", 0.0)),
                "pressure_bar": float(settings.get("pressure", 0.0)),
                "hmr": bool(float(settings.get("hydrogen_mass", 1.0)) > 1.5),
                "timestep_fs": float(settings.get("timestep_fs", 0.0)),
            },
        },
    }
    run = _run_docker_workflow("rbfe", WORKFLOWS["rbfe"], params)
    if run.get("queued"):
        st.success(f"OpenFE RBFE queued: {run['run_id']}")
    else:
        if run["returncode"] == 0:
            st.success(f"OpenFE RBFE completed: {run['run_id']}")
        else:
            st.error(f"OpenFE RBFE failed with exit code {run['returncode']}")
        if run.get("stderr"):
            with st.expander("stderr"):
                st.code(run["stderr"])
    st.switch_page("app/pages/jobs_openfe.py")


def render() -> None:
    try_dispatch_next_queued_gpu_job()
    _ensure_streamlit_help_icons_visible()
    st.title("Free Energy (OpenFE)")
    st.caption("Use only refined protein/ligand outputs from completed structure-preparation jobs.")

    jobs = _collect_structure_jobs()
    if not jobs:
        st.info("No completed structure jobs with refined protein+ligand files found.")
        return

    mode = st.selectbox(
        "OpenFE calculation mode",
        options=["ABFE", "RBFE"],
        index=0,
    )

    if mode == "ABFE":
        options = {row["label"]: row for row in jobs}
        selected_label = st.selectbox("Structure preparation job", list(options.keys()), key="openfe_abfe_job")
        selected = options[selected_label]
        st.caption(
            f"Using: `{Path(selected['protein_refined']).name}` + `{Path(selected['ligand_refined_sdf']).name}`"
        )
        _render_selected_structure_preview("abfe", selected)
        st.markdown("#### ABFE settings")
        if "openfe_abfe_preset" not in st.session_state:
            st.session_state["openfe_abfe_preset"] = "fast"
            _apply_abfe_preset_to_state("fast")
        preset = st.selectbox(
            "Preset",
            options=["fast", "balanced", "production"],
            key="openfe_abfe_preset",
            help="Controls default simulation lengths/repeats. You can still override fields below.",
        )
        prev_preset = st.session_state.get("openfe_abfe_preset_prev")
        if prev_preset != preset:
            _apply_abfe_preset_to_state(preset)
            st.session_state["openfe_abfe_preset_prev"] = preset
        base = ABFE_PRESETS[preset]
        st.caption(f"Preset guidance: {ABFE_PRESET_HELP[preset]}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Simulation**")
            production_length_ns = st.number_input(
                "Production length (ns)",
                min_value=0.1,
                step=0.1,
                key="openfe_abfe_prod_ns",
                help="Sampling time per lambda replica/window. Increasing this usually improves convergence.",
            )
            equilibration_length_ns = st.number_input(
                "Equilibration length (ns)",
                min_value=0.05,
                step=0.05,
                key="openfe_abfe_eq_ns",
                help="Pre-production relaxation time. Too small can increase noise/instability.",
            )
            protocol_repeats = st.number_input(
                "Protocol repeats",
                min_value=1,
                max_value=5,
                step=1,
                key="openfe_abfe_repeats",
                help="Independent repeats for uncertainty estimation. 1 = fastest; 3 often a good target.",
            )
        with c2:
            st.markdown("**Ligand and Environment**")
            charge_method = st.selectbox(
                "Partial charge method",
                options=["am1bcc", "am1bccelf10", "nagl", "espaloma"],
                index=0,
                key="openfe_abfe_charge",
                help="How ligand partial charges are assigned. AM1-BCC is the usual robust default.",
            )
            ligand_forcefield = st.selectbox(
                "Ligand forcefield",
                options=["openff-2.2.1", "openff-2.2.0", "openff-2.1.0", "openff-2.0.0"],
                index=0,
                key="openfe_abfe_lig_ff",
                help="Small-molecule force field. Prefer latest stable OpenFF unless you need strict comparability.",
            )
            solvent_model = st.selectbox(
                "Solvent model",
                options=["tip3p", "tip4pew", "spce"],
                index=0,
                key="openfe_abfe_solvent",
                help="Explicit water model. TIP3P is the common compatibility default.",
            )
            box_shape = st.selectbox(
                "Box shape",
                options=["dodecahedron", "cube"],
                index=0,
                key="openfe_abfe_box",
                help="Dodecahedron is typically more atom-efficient than cube at similar padding.",
            )
            ionic_strength = st.number_input(
                "Ionic strength (M)",
                min_value=0.0,
                step=0.01,
                value=0.15,
                key="openfe_abfe_ionic",
                help="Salt concentration. 0.15 M approximates physiological conditions.",
            )
            temperature = st.number_input(
                "Temperature (K)",
                min_value=250.0,
                max_value=400.0,
                step=0.5,
                value=298.15,
                key="openfe_abfe_temp",
                help="Thermodynamic target temperature. Keep consistent across compared systems.",
            )
            pressure = st.number_input(
                "Pressure (bar)",
                min_value=0.5,
                max_value=2.0,
                step=0.1,
                value=1.0,
                key="openfe_abfe_press",
                help="Pressure control target for NPT phases; 1.0 bar standard.",
            )
        with st.expander("Advanced"):
            use_hmr = st.checkbox(
                "Use hydrogen mass repartitioning (HMR)",
                value=bool(st.session_state.get("openfe_abfe_hmr", True)),
                key="openfe_abfe_hmr",
                help="When enabled, timestep is fixed to 4 fs. When disabled, timestep is fixed to 2 fs.",
            )
            n_replicas_complex = st.number_input(
                "Complex replicas",
                min_value=5,
                max_value=64,
                step=1,
                key="openfe_abfe_rep_complex",
                help="Number of alchemical replicas/windows in complex leg. More can improve overlap at higher cost.",
            )
            n_replicas_solvent = st.number_input(
                "Solvent replicas",
                min_value=5,
                max_value=64,
                step=1,
                key="openfe_abfe_rep_solvent",
                help="Replicas/windows in solvent leg. Often lower than complex leg.",
            )
            minimization_steps = st.number_input(
                "Minimization steps",
                min_value=100,
                step=100,
                value=5000,
                key="openfe_abfe_min_steps",
                help="Initial energy minimization iterations before dynamics.",
            )
            if use_hmr:
                st.session_state["openfe_abfe_timestep"] = 4.0
            else:
                st.session_state["openfe_abfe_timestep"] = 2.0
            timestep_fs = st.number_input(
                "Timestep (fs)",
                min_value=1.0,
                max_value=4.0,
                step=0.5,
                key="openfe_abfe_timestep",
                disabled=True,
                help="Integrator timestep. 4 fs assumes hydrogen-mass repartitioning/constraints in protocol.",
            )

        st.session_state["openfe_abfe_settings"] = {
            "fast_mode": preset == "fast",
            "production_length_ns": float(production_length_ns),
            "equilibration_length_ns": float(equilibration_length_ns),
            "protocol_repeats": int(protocol_repeats),
            "charge_method": str(charge_method),
            "ligand_forcefield": str(ligand_forcefield),
            "solvent_model": str(solvent_model),
            "box_shape": str(box_shape),
            "ionic_strength": float(ionic_strength),
            "temperature": float(temperature),
            "pressure": float(pressure),
            "n_replicas_complex": int(n_replicas_complex),
            "n_replicas_solvent": int(n_replicas_solvent),
            "minimization_steps": int(minimization_steps),
            "timestep_fs": float(timestep_fs),
            "hydrogen_mass": 3.0 if bool(use_hmr) else 1.0,
        }
        timeout_hours, timeout_meta = _estimate_abfe_timeout_hours(st.session_state["openfe_abfe_settings"])
        rt_low_h, rt_high_h = _estimate_abfe_runtime_hours(st.session_state["openfe_abfe_settings"], timeout_hours)
        st.info(
            "Estimated ABFE timeout budget: "
            f"~{timeout_hours:.2f} h "
            f"({timeout_meta.get('source')})."
        )
        st.caption(
            "Expected runtime (heuristic): "
            f"~{rt_low_h:.2f}–{rt_high_h:.2f} h "
            "(depends strongly on GPU load, convergence, and restraint-search behavior)."
        )
        if st.button("Run OpenFE ABFE", type="primary"):
            _run_abfe(selected)
        return

    selected_labels = st.multiselect(
        "Structure preparation jobs (ligand series)",
        [row["label"] for row in jobs],
        default=[jobs[0]["label"], jobs[1]["label"]] if len(jobs) > 1 else [],
        key="openfe_rbfe_jobs",
    )
    if not selected_labels:
        st.info("Select at least two structure jobs for RBFE.")
        return
    selected_jobs = [row for row in jobs if row["label"] in set(selected_labels)]
    st.caption(f"{len(selected_jobs)} selected ligands will be combined into one RBFE ligand set.")
    _render_selected_structure_preview("rbfe", selected_jobs[0])
    st.markdown("#### RBFE settings")
    if "openfe_rbfe_preset" not in st.session_state:
        st.session_state["openfe_rbfe_preset"] = "fast"
        _apply_rbfe_preset_to_state("fast")
    r_preset = st.selectbox(
        "Preset",
        options=["fast", "balanced", "production"],
        key="openfe_rbfe_preset",
        help="Controls default RBFE runtime/precision. You can override fields below.",
    )
    prev_r_preset = st.session_state.get("openfe_rbfe_preset_prev")
    if prev_r_preset != r_preset:
        _apply_rbfe_preset_to_state(r_preset)
        st.session_state["openfe_rbfe_preset_prev"] = r_preset
    r_base = RBFE_PRESETS[r_preset]
    st.caption(f"Preset guidance: {RBFE_PRESET_HELP[r_preset]}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Network**")
        network_topology = st.selectbox(
            "Network topology",
            options=["mst", "radial", "maximal"],
            index=0,
            key="openfe_rbfe_topology",
            help="MST minimizes number of transformations; radial compares all ligands to one center; maximal uses all pairs.",
        )
        central_ligand = st.text_input(
            "Central ligand (radial only, optional)",
            value="",
            key="openfe_rbfe_central",
            help="Identifier of reference ligand for radial topology.",
        )
        st.markdown("**Atom mapping**")
        atom_mapper = st.selectbox(
            "Atom mapper",
            options=["kartograf", "lomap", "lomap_relaxed"],
            index=0,
            key="openfe_rbfe_mapper",
            help="How atoms are matched between ligand pairs. Kartograf is usually robust for 3D-aware mapping.",
        )
        atom_map_hydrogens = st.checkbox(
            "Map hydrogens (kartograf)",
            value=True,
            key="openfe_rbfe_map_h",
            help="Include hydrogens in mapping. Can improve realism but may reduce map availability for difficult pairs.",
        )
        lomap_max3d = st.number_input(
            "LOMAP max3d (A)",
            min_value=0.1,
            step=0.1,
            value=1.0,
            key="openfe_rbfe_lomap_max3d",
            help="Maximum 3D atom distance threshold for LOMAP mapping.",
        )
    with c2:
        st.markdown("**Simulation**")
        lambda_windows = st.number_input(
            "Lambda windows",
            min_value=5,
            max_value=21,
            step=2,
            key="openfe_rbfe_lambda",
            help="Number of intermediate alchemical states. More windows improve overlap at higher runtime.",
        )
        production_length_ns = st.number_input(
            "Production length (ns)",
            min_value=0.1,
            step=0.1,
            key="openfe_rbfe_prod_ns",
            help="Sampling time per lambda window/replica.",
        )
        equilibration_length_ns = st.number_input(
            "Equilibration length (ns)",
            min_value=0.05,
            step=0.05,
            key="openfe_rbfe_eq_ns",
            help="Relaxation before production per window.",
        )
        protocol_repeats = st.number_input(
            "Protocol repeats",
            min_value=1,
            max_value=5,
            step=1,
            key="openfe_rbfe_repeats",
            help="Independent repeats for uncertainty/robustness.",
        )
        st.markdown("**Ligand and Environment**")
        charge_method = st.selectbox(
            "Partial charge method",
            options=["am1bcc", "am1bccelf10", "nagl", "espaloma"],
            index=0,
            key="openfe_rbfe_charge",
            help="Ligand charge assignment method. AM1-BCC is the common baseline.",
        )
        ligand_forcefield = st.selectbox(
            "Ligand forcefield",
            options=["openff-2.2.1", "openff-2.2.0", "openff-2.1.0", "openff-2.0.0"],
            index=0,
            key="openfe_rbfe_lig_ff",
            help="Small-molecule force field used for ligand parameterization.",
        )
        solvent_model = st.selectbox(
            "Solvent model",
            options=["tip3p", "tip4pew", "spce"],
            index=0,
            key="openfe_rbfe_solvent",
            help="Water model for both legs.",
        )
        box_shape = st.selectbox(
            "Box shape",
            options=["dodecahedron", "cube"],
            index=0,
            key="openfe_rbfe_box",
            help="Simulation box geometry; dodecahedron is often more efficient.",
        )
        ionic_strength = st.number_input(
            "Ionic strength (M)",
            min_value=0.0,
            step=0.01,
            value=0.15,
            key="openfe_rbfe_ionic",
            help="Salt concentration in mol/L. 0.15 M is a common default.",
        )
        temperature = st.number_input(
            "Temperature (K)",
            min_value=250.0,
            max_value=400.0,
            step=0.5,
            value=298.15,
            key="openfe_rbfe_temp",
            help="Thermodynamic temperature target.",
        )
        pressure = st.number_input(
            "Pressure (bar)",
            min_value=0.5,
            max_value=2.0,
            step=0.1,
            value=1.0,
            key="openfe_rbfe_press",
            help="Pressure target for NPT segments.",
        )
    with st.expander("Advanced"):
        use_hmr = st.checkbox(
            "Use hydrogen mass repartitioning (HMR)",
            value=bool(st.session_state.get("openfe_rbfe_hmr", True)),
            key="openfe_rbfe_hmr",
            help="When enabled, timestep is fixed to 4 fs. When disabled, timestep is fixed to 2 fs.",
        )
        minimization_steps = st.number_input(
            "Minimization steps",
            min_value=100,
            step=100,
            value=5000,
            key="openfe_rbfe_min_steps",
            help="Energy minimization before equilibration.",
        )
        if use_hmr:
            st.session_state["openfe_rbfe_timestep"] = 4.0
        else:
            st.session_state["openfe_rbfe_timestep"] = 2.0
        timestep_fs = st.number_input(
            "Timestep (fs)",
            min_value=1.0,
            max_value=4.0,
            step=0.5,
            key="openfe_rbfe_timestep",
            disabled=True,
            help="Integrator timestep. Keep protocol-consistent across campaigns.",
        )

    st.session_state["openfe_rbfe_settings"] = {
        "fast_mode": r_preset == "fast",
        "lambda_windows": int(lambda_windows),
        "production_length_ns": float(production_length_ns),
        "equilibration_length_ns": float(equilibration_length_ns),
        "protocol_repeats": int(protocol_repeats),
        "charge_method": str(charge_method),
        "ligand_forcefield": str(ligand_forcefield),
        "solvent_model": str(solvent_model),
        "box_shape": str(box_shape),
        "ionic_strength": float(ionic_strength),
        "temperature": float(temperature),
        "pressure": float(pressure),
        "minimization_steps": int(minimization_steps),
        "timestep_fs": float(timestep_fs),
        "hydrogen_mass": 3.0 if bool(use_hmr) else 1.0,
    }
    st.session_state["openfe_rbfe_mapping"] = {
        "atom_mapper": str(atom_mapper),
        "atom_map_hydrogens": bool(atom_map_hydrogens),
        "lomap_max3d": float(lomap_max3d),
    }
    st.session_state["openfe_rbfe_network"] = {
        "network_topology": str(network_topology),
        "central_ligand": str(central_ligand).strip(),
    }
    if st.button("Run OpenFE RBFE", type="primary"):
        _run_rbfe(selected_jobs)


render()
