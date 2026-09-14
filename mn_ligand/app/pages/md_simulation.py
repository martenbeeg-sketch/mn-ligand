from __future__ import annotations

import math
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.discover_inputs import (
    bound_ligand_box,
    render_target_viewer,
    select_target_artifact,
    target_viewer_path,
)
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.app.pages.bound_ligand_md import PROTOCOL_PRESETS, parse_bound_ligands
from mn_ligand.app.pages.common import try_dispatch_next_queued_gpu_job
from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records
from mn_ligand.core.workflows import WorkflowRecord
from mn_ligand.runtime import resolve_run_dir, runs_root
from mn_ligand.workflows.md_simulation import (
    DEFAULT_MD_IMAGE,
    EXACT_CONTINUATION,
    INDEPENDENT_REPLICA,
    create_md_simulation,
    create_md_simulation_from_prepared,
    visible_md_workflow_rows,
)
from mn_ligand.workflows.md_engines import (
    GROMACS_ENGINE,
    OPENMM_ENGINE,
    md_engine_spec,
    normalize_md_engine,
    roe_brooks_stage_specification,
)
from mn_ligand.workflows.complex_datasets import selected_complex_jobs

PRODUCTION_PRESETS = {
    "Smoke": {
        "description": (
            "Technical validation only: checks parameterization, execution and "
            "artifact generation. Do not interpret it scientifically."
        ),
        "production_ns": 0.2,
        "replicas": 1,
        "report_interval_ps": 5.0,
        "burn_in_ns": 0.0,
        "revalidation_max_ns": 1.0,
        "analysis_enabled": True,
        "endpoint_enabled": False,
    },
    "Ligand MM/GBSA": {
        "description": (
            "Endpoint-energy starting point: three independent replicas, each "
            "reusing a prepared system that passed the Roe density-fit gate, "
            "then reseeding velocities and repeating only the density-fit gate "
            "before 50 ns recorded production and automatic endpoint analysis. "
            "Inspect stability and convergence before interpreting rankings."
        ),
        "production_ns": 50.0,
        "replicas": 3,
        "report_interval_ps": 10.0,
        "burn_in_ns": 0.0,
        "revalidation_max_ns": 5.0,
        "analysis_enabled": True,
        "endpoint_enabled": True,
    },
    "Stability": {
        "description": (
            "Recommended stability starting point: three independent replicas, "
            "each reusing a prepared system that passed the Roe density-fit gate "
            "and repeating only that density-fit gate after velocity reseeding, "
            "before 100 ns recorded production. Many systems require longer sampling."
        ),
        "production_ns": 100.0,
        "replicas": 3,
        "report_interval_ps": 10.0,
        "burn_in_ns": 0.0,
        "revalidation_max_ns": 5.0,
        "analysis_enabled": True,
        "endpoint_enabled": False,
    },
    "Manual": {
        "description": (
            "Expert-controlled trajectory length, replica count, output interval "
            "and optional unrecorded replica NPT equilibration. Exact values are "
            "preserved in provenance."
        ),
        "production_ns": 50.0,
        "replicas": 3,
        "report_interval_ps": 10.0,
        "burn_in_ns": 0.0,
        "revalidation_max_ns": 5.0,
        "analysis_enabled": True,
        "endpoint_enabled": False,
    },
}
PRODUCTION_TIMESTEP_FS = 4.0
MD_ENGINE_KEYS = {
    OPENMM_ENGINE: "md_engine_openmm",
    GROMACS_ENGINE: "md_engine_gromacs",
}


def _set_md_engine_selection(value: bool) -> None:
    for key in MD_ENGINE_KEYS.values():
        st.session_state[key] = value


def _production_for_engine(
    production: dict,
    engine: str,
    engine_settings: dict | None = None,
) -> dict:
    selected = normalize_md_engine(engine)
    settings = engine_settings or {}
    timestep_fs = float(
        settings.get("production_timestep_fs")
        or md_engine_spec(selected).production_timestep_fs
    )
    adjusted = dict(production)
    if adjusted.get("endpoint_backend") == "engine_native":
        adjusted["endpoint_backend"] = (
            "g_mmpbsa"
            if selected == GROMACS_ENGINE
            else "openmm_gbsa"
        )
    if timestep_fs == float(production["production_timestep_fs"]):
        return adjusted
    production_ns = float(adjusted["production_length_ns"])
    report_ps = float(adjusted["production_report_interval_ps"])
    burn_in_ns = float(adjusted.get("replica_equilibration_ns") or 0.0)
    revalidation_max_ns = float(
        adjusted.get("replica_revalidation_max_ns") or burn_in_ns
    )
    revalidation_increment_ns = float(
        adjusted.get("replica_revalidation_increment_ns") or 1.0
    )
    density_sample_interval_ps = float(
        adjusted.get("replica_density_sample_interval_ps") or 4.0
    )
    adjusted.update(
        {
            "production_timestep_fs": timestep_fs,
            "production_steps": max(
                1,
                int(round(production_ns * 1_000_000.0 / timestep_fs)),
            ),
            "production_report_interval": max(
                1,
                int(round(report_ps * 1000.0 / timestep_fs)),
            ),
            "replica_equilibration_steps": (
                max(
                    1,
                    int(round(burn_in_ns * 1_000_000.0 / timestep_fs)),
                )
                if burn_in_ns > 0
                else 0
            ),
            "replica_revalidation_max_steps": max(
                1,
                int(round(revalidation_max_ns * 1_000_000.0 / timestep_fs)),
            ),
            "replica_revalidation_increment_steps": max(
                1,
                int(round(revalidation_increment_ns * 1_000_000.0 / timestep_fs)),
            ),
            "replica_density_sample_interval_steps": max(
                1,
                int(round(density_sample_interval_ps * 1000.0 / timestep_fs)),
            ),
        }
    )
    return adjusted


def _job_label(job: JobRecord) -> str:
    code = display_job_code(job.metadata.get("job_code"), job.run_id)
    pdb_id = str(job.metadata.get("pdb_id") or "-")
    ligand = str(job.metadata.get("ligand_label") or job.metadata.get("ligand_key") or "-")
    return f"{code} | {pdb_id} | {ligand}"


def _structure_sources() -> list[tuple[JobRecord, object]]:
    sources: list[tuple[JobRecord, object]] = []
    for job in iter_job_records(runs_root(), task_groups=("structure-jobs",)):
        if job.status != "completed" or job.artifact_manifest is None:
            continue
        artifacts = job.artifact_manifest.by_type("prepared_complex")
        if artifacts and artifacts[0].resolve(job.run_dir, must_exist=True) is not None:
            sources.append((job, artifacts[0]))
    return sources


def _prepared_systems() -> list[JobRecord]:
    return [
        job
        for job in iter_job_records(runs_root(), task_groups=("md-system-prep",))
        if job.status == "completed" and (job.run_dir / "result.json").is_file()
    ]


def _new_prep_input(job: JobRecord, artifact, settings: dict) -> dict:
    complex_path = artifact.resolve(job.run_dir, must_exist=True)
    if complex_path is None:
        raise FileNotFoundError(artifact.path)
    pdb_data = complex_path.read_text()
    ligands = parse_bound_ligands(pdb_data)
    ligand_key = str(job.metadata.get("ligand_key") or "")
    selected = next((item for item in ligands if item.get("key") == ligand_key), ligands[0] if ligands else None)
    if selected is None:
        raise ValueError("The prepared complex contains no selectable ligand")
    payload = {
        "pdb_id": str(job.metadata.get("pdb_id") or "UNKNOWN"),
        "pdb_data": pdb_data,
        "ligand_key": selected["key"],
        "charge_method": settings["charge_method"],
        "forcefield_method": settings["forcefield_method"],
        "protein_forcefield_method": settings[
            "protein_forcefield_method"
        ],
        "water_model": settings["water_model"],
        "heating_steps_per_stage": settings["heating_steps_per_stage"],
        "heating_stages": settings["heating_stages"],
        "nvt_steps": settings["nvt_steps"],
        "npt_steps": settings["npt_steps"],
        "production_steps": 0,
        "integration_profile": settings["integration_profile"],
        "production_timestep_fs": settings["production_timestep_fs"],
        "hydrogen_mass_amu": settings["hydrogen_mass_amu"],
        "mass_repartition_factor": settings["mass_repartition_factor"],
        "temperature": settings["temperature"],
        "padding_nm": settings["padding_nm"],
        "box_shape": settings["box_shape"],
        "pressure": settings["pressure"],
        "ionic_strength": settings["ionic_strength"],
        "constraints": "HBonds",
        "minimization_only": False,
        "output_dir": "/output",
        "ligand_data_format": "pdb",
        "preserve_ligand_pose": True,
        "generate_conformer": False,
        "preview_before_equilibration": False,
        "preview_acknowledged": False,
        "pause_at_minimized": False,
        "minimized_acknowledged": False,
        "production_report_interval": 2500,
        "apply_protein_restraints_during_heating_nvt": True,
        "protein_restraint_selection": "backbone",
        "protein_restraint_k": 1000.0,
        "ligand_restraints_enabled": True,
        "ligand_lock_k_kjmol_nm2": 2500.0,
        "npt_release_enabled": True,
        "npt_restraint_release_scales": "1.0,0.5,0.2,0.05,0.0",
        "protein_npt_release_scales": "1.0,0.5,0.1,0.01,0.0",
        "force_unrestrained_production": True,
        "prepared_complex_path": str(complex_path),
        "mmgbsa_backend": settings["mmgbsa_backend"],
        "preparation_protocol": settings["preparation_protocol"],
        "density_stabilization_min_ns": settings[
            "density_stabilization_min_ns"
        ],
        "density_stabilization_max_ns": settings[
            "density_stabilization_max_ns"
        ],
        "density_stabilization_increment_ns": settings[
            "density_stabilization_increment_ns"
        ],
        "density_sample_interval_ps": settings[
            "density_sample_interval_ps"
        ],
        "density_plateau_required": settings["density_plateau_required"],
    }
    refined_ligands = job.artifact_manifest.by_type("prepared_ligand_set") if job.artifact_manifest else ()
    if refined_ligands:
        ligand_path = refined_ligands[0].resolve(job.run_dir, must_exist=True)
        if ligand_path:
            payload["ligand_refined_sdf_data"] = ligand_path.read_text()
            payload["ligand_refined_sdf_path"] = str(ligand_path)
            payload["strict_refined_ligand"] = True
    return payload


def _render_roe_protocol_info() -> None:
    with st.expander("What the Roe–Brooks preparation protocol does"):
        st.markdown(
            """
            The protocol prepares an explicitly solvated system for stable MD;
            it is not itself proof that slow protein or ligand motions are
            equilibrated. Mobile solvent and ions relax first while protein and
            ligand heavy atoms are restrained. Restraints are then reduced
            through alternating minimization and short NVT/NPT stages and are
            fully removed before the final density-stabilization stage.

            Production starts only from the unrestrained state. The density gate
            fits an exponential relaxation curve and checks its final slope, the
            fitted-final versus late-trajectory mean, and fit residual. Ligand
            escape after restraint release is retained as a scientific result.
            """
        )
        rows = []
        for stage in roe_brooks_stage_specification():
            duration = (
                f"{float(stage['duration_ps']):g} ps"
                if stage.get("duration_ps") is not None
                else "until converged"
                if stage["kind"] == "minimization"
                else "density gate"
            )
            rows.append(
                {
                    "Stage": stage["stage"],
                    "Type": str(stage["kind"]).upper(),
                    "Duration": duration,
                    "Restraint": stage.get("restraint_selection", "none"),
                    "k (kcal mol⁻¹ Å⁻²)": stage.get(
                        "restraint_k_kcal_mol_a2", 0.0
                    ),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Roe et al. used ff14SB for proteins, GAFF/AM1-BCC for their "
            "parameterized ligand example, TIP3P water, and Joung–Cheatham ions. "
            "The preparation sequence can also be used with other reasonable "
            "force fields, but those runs represent different physical models."
        )


def _engine_preparation_controls(engine: str) -> dict:
    spec = md_engine_spec(engine)
    st.markdown(f"##### {spec.label} preparation")
    if engine == GROMACS_ENGINE:
        st.caption(
            "Native GROMACS 2026.3 dynamics with AmberTools-generated "
            "protein/GAFF topology and explicitly selected solvent model."
        )
        preparation_protocol = "roe_brooks_2020"
        preset_name = "Roe–Brooks fixed sequence"
    else:
        preparation_label = st.selectbox(
            "System-preparation protocol",
            [
                "Roe–Brooks 2020",
                "Current staged equilibration (compatibility)",
            ],
            key=f"md_preparation_protocol_{engine}",
            help=(
                "Roe–Brooks progressively relaxes solvent, protein and ligand, "
                "then requires an unrestrained density plateau."
            ),
        )
        preparation_protocol = (
            "roe_brooks_2020"
            if preparation_label.startswith("Roe")
            else "current_staged"
        )
        preset_name = "Roe–Brooks fixed sequence"

    heating_stages = 6
    heating_steps = 2500
    nvt_steps = 25000
    npt_steps = 175000
    density_min_ns = 1.0
    density_max_ns = 5.0
    density_increment_ns = 1.0
    density_sample_interval_ps = 4.0
    density_plateau_required = True
    if preparation_protocol == "roe_brooks_2020":
        st.info(
            "Roe–Brooks: restrained relaxation → complete restraint release → "
            "unrestrained NPT density gate."
        )
        show_advanced_roe = st.checkbox(
            "Show advanced Roe–Brooks controls",
            value=False,
            key=f"md_show_advanced_roe_{engine}",
        )
        if show_advanced_roe:
            density_columns = st.columns(4)
            density_min_ns = float(
                density_columns[0].number_input(
                    "Minimum stabilization (ns)",
                    min_value=0.1,
                    value=1.0,
                    step=0.5,
                    key=f"md_density_min_{engine}",
                )
            )
            density_max_ns = float(
                density_columns[1].number_input(
                    "Maximum stabilization (ns)",
                    min_value=density_min_ns,
                    value=max(5.0, density_min_ns),
                    step=1.0,
                    key=f"md_density_max_{engine}",
                )
            )
            density_increment_ns = float(
                density_columns[2].number_input(
                    "Extension increment (ns)",
                    min_value=0.1,
                    value=1.0,
                    step=0.5,
                    key=f"md_density_increment_{engine}",
                )
            )
            density_sample_interval_ps = float(
                density_columns[3].number_input(
                    "Density interval (ps)",
                    min_value=1.0,
                    value=4.0,
                    step=1.0,
                    key=f"md_density_interval_{engine}",
                )
            )
            density_plateau_required = st.checkbox(
                "Require all density-plateau criteria",
                value=True,
                key=f"md_density_required_{engine}",
            )
    else:
        st.warning(
            "Compatibility mode reproduces the previous OpenMM staged "
            "equilibration and is not available for GROMACS."
        )
        preset_name = st.segmented_control(
            "Legacy preparation preset",
            list(PROTOCOL_PRESETS),
            default="Longer MD",
            key=f"md_legacy_preset_{engine}",
        )
        preset = PROTOCOL_PRESETS[str(preset_name)]
        heating_steps = int(preset["heating_steps_per_stage"])
        nvt_steps = int(preset["nvt_steps"])
        npt_steps = int(preset["npt_steps"])
        show_advanced_legacy = st.checkbox(
            "Show advanced legacy equilibration controls",
            value=False,
            key=f"md_show_advanced_legacy_{engine}",
        )
        if show_advanced_legacy:
            heating_stages = int(
                st.number_input(
                    "Heating stages",
                    min_value=1,
                    value=6,
                    step=1,
                    key=f"md_heating_stages_{engine}",
                )
            )
            step_columns = st.columns(3)
            heating_steps = int(
                step_columns[0].number_input(
                    "Heating steps/stage",
                    min_value=0,
                    value=heating_steps,
                    step=250,
                    key=f"md_heating_steps_{engine}",
                )
            )
            nvt_steps = int(
                step_columns[1].number_input(
                    "NVT steps",
                    min_value=0,
                    value=nvt_steps,
                    step=500,
                    key=f"md_nvt_steps_{engine}",
                )
            )
            npt_steps = int(
                step_columns[2].number_input(
                    "NPT steps",
                    min_value=0,
                    value=npt_steps,
                    step=500,
                    key=f"md_npt_steps_{engine}",
                )
            )

    if engine == OPENMM_ENGINE:
        protein_options = {
            "Amber ff14SB family (recommended)": "amber14-all",
            "Amber99SB-ILDN": "amber99sbildn",
            "Amber FB15": "amberfb15",
            "Amber ff15ipq": "amber15ipq",
        }
        ligand_options = {
            "OpenFF 2.2.0 / Sage (SMIRNOFF; recommended)": (
                "openff-2.2.0",
                "openmm_gbsa",
            ),
            "OpenFF 2.1.0 / Sage (SMIRNOFF)": (
                "openff-2.1.0",
                "openmm_gbsa",
            ),
            "GAFF2": (
                "gaff2",
                "openmm_gbsa",
            ),
            "GAFF (Roe ligand model)": (
                "gaff",
                "openmm_gbsa",
            ),
        }
        water_options = {
            "TIP3P (recommended with ff14SB/Amber99SB-ILDN)": "tip3p",
            "TIP3P-FB (recommended with FB15)": "tip3pfb",
            "SPC/E (recommended with ff15ipq)": "spce",
            "TIP4P-Ew": "tip4pew",
        }
        recommended_water = {
            "amber14-all": "tip3p",
            "amber99sbildn": "tip3p",
            "amberfb15": "tip3pfb",
            "amber15ipq": "spce",
        }
    else:
        protein_options = {
            "Amber ff14SB (recommended; smoke-qualified)": "ff14SB",
            "Amber ff19SB": "ff19SB",
            "Amber ff15ipq": "ff15ipq",
            "Amber ff03.r1 (legacy)": "ff03.r1",
        }
        ligand_options = {
            "GAFF2 (recommended; smoke-qualified)": (
                "gaff2",
                "ambertools_mmpbsa",
            ),
            "GAFF (Roe ligand model)": (
                "gaff",
                "ambertools_mmpbsa",
            ),
        }
        water_options = {
            "TIP3P (recommended with ff14SB/ff03)": "tip3p",
            "OPC (recommended with ff19SB)": "opc",
            "SPC/E (recommended with ff15ipq)": "spce",
            "TIP4P-Ew": "tip4pew",
        }
        recommended_water = {
            "ff14SB": "tip3p",
            "ff19SB": "opc",
            "ff15ipq": "spce",
            "ff03.r1": "tip3p",
        }
    protein_label = st.selectbox(
        "Protein force field",
        list(protein_options),
        key=f"md_protein_forcefield_{engine}",
    )
    protein_forcefield_method = protein_options[str(protein_label)]
    ligand_label = st.selectbox(
        "Ligand force field",
        list(ligand_options),
        key=f"md_parameterization_{engine}",
    )
    forcefield_method, mmgbsa_backend = ligand_options[
        str(ligand_label)
    ]
    water_label = st.selectbox(
        "Water model",
        list(water_options),
        key=f"md_water_model_{engine}",
    )
    water_model = water_options[str(water_label)]
    charge_options = (
        ["am1bcc", "gasteiger", "mmff94"]
        if forcefield_method.startswith("openff")
        else ["am1bcc"]
    )
    charge_method = st.selectbox(
        "Ligand charges",
        charge_options,
        key=f"md_charge_method_{engine}",
    )
    if water_model != recommended_water[protein_forcefield_method]:
        st.warning(
            f"{water_label} is not the recommended pairing for "
            f"{protein_label}. This is allowed for expert comparison, but the "
            "combination should be justified and validated."
        )
    if forcefield_method.startswith("openff"):
        st.caption(
            "OpenFF uses a SMIRNOFF ligand model. The selected protein and "
            "water models are applied independently by the OpenMM builder."
        )
    elif forcefield_method == "gaff":
        st.caption(
            "GAFF matches the ligand model reported in Roe et al.; GAFF2 is the "
            "newer default and was used for the native GROMACS smoke."
        )

    accelerated_label = (
        "Accelerated 4 fs / native mass factor 3 "
        "(recommended for stability and MM/GBSA)"
        if engine == GROMACS_ENGINE
        else "Accelerated 4 fs / HMR 4 amu "
        "(recommended for stability and MM/GBSA)"
    )
    integration_options = {
        accelerated_label: (
            "hmr_4fs",
            4.0,
            None if engine == GROMACS_ENGINE else 4.0,
        ),
        "Standard 2 fs / normal hydrogen masses": (
            "standard_2fs",
            2.0,
            None,
        ),
    }
    integration_label = st.selectbox(
        "Integration profile",
        list(integration_options),
        key=f"md_integration_profile_{engine}",
        help=(
            "The mass model is applied during system construction and remains "
            "unchanged through Roe preparation, density stabilization, replica "
            "revalidation, and production. A 4 fs timestep is available only "
            "with hydrogen-mass repartitioning."
        ),
    )
    (
        integration_profile,
        production_timestep_fs,
        hydrogen_mass_amu,
    ) = integration_options[str(integration_label)]
    mass_repartition_factor = (
        3.0
        if engine == GROMACS_ENGINE and integration_profile == "hmr_4fs"
        else None
    )
    if engine == GROMACS_ENGINE and mass_repartition_factor is not None:
        st.caption(
            "GROMACS applies native grompp mass repartitioning with factor 3 "
            "and h-bond constraints, preserving GPU-resident updates. The "
            "source Amber/GAFF topology retains normal masses."
        )
    else:
        st.caption(
            "HMR is suitable for equilibrium ligand stability and "
            "endpoint-energy sampling, but not for interpreting kinetic "
            "rates. Use the same profile for every system in a comparison."
        )

    system_columns = st.columns(3)
    with system_columns[0]:
        box_shape = st.selectbox(
            "Solvent box",
            ["octahedron", "dodecahedron", "cube"],
            key=f"md_box_shape_{engine}",
            help="Roe et al. used a truncated octahedral box.",
        )
        padding_nm = float(
            st.number_input(
                "Padding (nm)",
                min_value=0.1,
                value=1.0,
                step=0.1,
                key=f"md_padding_nm_{engine}",
            )
        )
    with system_columns[1]:
        ionic_strength = float(
            st.number_input(
                "Ionic strength (M)",
                min_value=0.0,
                value=0.15,
                step=0.05,
                key=f"md_ionic_strength_{engine}",
            )
        )
        temperature = float(
            st.number_input(
                "Temperature (K)",
                min_value=1.0,
                value=300.0,
                step=1.0,
                key=f"md_temperature_{engine}",
            )
        )
    with system_columns[2]:
        pressure = float(
            st.number_input(
                "Pressure (bar)",
                min_value=0.1,
                value=1.0,
                step=0.1,
                key=f"md_pressure_{engine}",
            )
        )
        st.caption("Standard condition: 300 K, 1 bar, 0.15 M.")

    return {
        "protocol": preset_name,
        "mmgbsa_backend": mmgbsa_backend,
        "forcefield_method": forcefield_method,
        "protein_forcefield_method": protein_forcefield_method,
        "water_model": water_model,
        "charge_method": charge_method,
        "box_shape": box_shape,
        "padding_nm": padding_nm,
        "ionic_strength": ionic_strength,
        "temperature": temperature,
        "pressure": pressure,
        "integration_profile": integration_profile,
        "production_timestep_fs": production_timestep_fs,
        "hydrogen_mass_amu": hydrogen_mass_amu,
        "mass_repartition_factor": mass_repartition_factor,
        "heating_stages": heating_stages,
        "heating_steps_per_stage": heating_steps,
        "nvt_steps": nvt_steps,
        "npt_steps": npt_steps,
        "preparation_protocol": preparation_protocol,
        "density_stabilization_min_ns": density_min_ns,
        "density_stabilization_max_ns": density_max_ns,
        "density_stabilization_increment_ns": density_increment_ns,
        "density_sample_interval_ps": density_sample_interval_ps,
        "density_plateau_required": density_plateau_required,
    }


def _production_controls(
    engines: list[str] | None = None,
    engine_settings: dict[str, dict] | None = None,
) -> tuple[dict, int, bool]:
    engines = engines or [OPENMM_ENGINE]
    engine_settings = engine_settings or {}
    st.markdown("#### Production")
    preset_name = st.segmented_control(
        "Production protocol",
        list(PRODUCTION_PRESETS),
        default="Stability",
        key="md_production_protocol",
    )
    preset = PRODUCTION_PRESETS[str(preset_name)]
    st.caption(str(preset["description"]))
    columns = st.columns(3)
    with columns[0]:
        production_ns = float(
            st.number_input(
                "Length per trajectory (ns)",
                min_value=0.004,
                value=float(preset["production_ns"]),
                step=1.0,
                key=f"md_production_ns_{preset_name}",
            )
        )
        report_interval_ps = float(
            st.number_input(
                "Saved-frame interval (ps)",
                min_value=0.004,
                value=float(preset["report_interval_ps"]),
                step=1.0,
                key=f"md_report_ps_{preset_name}",
            )
        )
    with columns[1]:
        start_label = st.selectbox(
            "Production start",
            ["Start independent replicas", "Continue exact NPT checkpoint"],
            help=(
                "Exact continuation preserves the NPT state. Independent replicas "
                "receive new seeded velocities and undergo unrestrained Roe-style "
                "density revalidation."
            ),
        )
        start_mode = EXACT_CONTINUATION if start_label.startswith("Continue") else INDEPENDENT_REPLICA
        replicas = int(
            st.number_input(
                "Production trajectories",
                min_value=1,
                value=int(preset["replicas"]),
                step=1,
                disabled=start_mode == EXACT_CONTINUATION,
                key=f"md_replicas_{preset_name}",
            )
        )
    with columns[2]:
        burn_in_ns = float(
            st.number_input(
                "Minimum unrecorded replica NPT equilibration (ns)",
                min_value=0.0,
                value=float(preset["burn_in_ns"]),
                step=0.1,
                disabled=start_mode == EXACT_CONTINUATION,
                key=f"md_burn_in_ns_{preset_name}",
                help=(
                    "Runs continuously before recorded production without "
                    "restarting the simulation. This time does not count toward "
                    "the requested production length."
                ),
            )
        )
        revalidation_max_ns = float(
            st.number_input(
                "Maximum replica NPT revalidation (ns)",
                min_value=burn_in_ns,
                value=max(
                    burn_in_ns,
                    float(preset["revalidation_max_ns"]),
                ),
                step=0.5,
                disabled=start_mode == EXACT_CONTINUATION,
                key=f"md_revalidation_max_ns_{preset_name}",
            )
        )
        st.caption(
            "Independent replicas receive distinct recorded seeds. Unrecorded "
            "NPT equilibration and density revalidation continue directly into "
            "production without a restart. These frames do not count toward the "
            "requested production length and are excluded from plots, summaries, "
            "and endpoint analysis."
        )

    manual_analysis_selection = str(preset_name) == "Manual"
    analysis_state_key = f"md_analysis_enabled_{preset_name}"
    endpoint_state_key = f"md_endpoint_enabled_{preset_name}"
    if not manual_analysis_selection:
        st.session_state[analysis_state_key] = bool(
            preset["analysis_enabled"]
        )
        st.session_state[endpoint_state_key] = bool(
            preset["endpoint_enabled"]
        )
    st.markdown("##### Post-run analysis")
    optional_columns = st.columns(2)
    analysis_enabled = optional_columns[0].checkbox(
        "Aggregate replica stability analysis",
        value=bool(preset["analysis_enabled"]),
        key=analysis_state_key,
        disabled=not manual_analysis_selection,
        help=(
            "Combines per-replica stability metrics and reports mean and sample "
            "standard deviation."
        ),
    )
    endpoint_enabled = optional_columns[1].checkbox(
        "Automatically run endpoint MM/GBSA after production",
        value=bool(preset["endpoint_enabled"]),
        key=endpoint_state_key,
        disabled=not manual_analysis_selection,
        help=(
            "Creates a separate immutable endpoint-energy job for each completed "
            "trajectory. Leave this off when you want to inspect stability and "
            "choose the trajectory window later from post-evaluation."
        ),
    )
    if not manual_analysis_selection:
        st.caption(
            f"{preset_name} defines these analysis steps automatically. "
            "Select Manual to override them."
        )
    st.caption(
        "Recommended: inspect density, RMSD and ligand stability first, then run "
        "MM/GBSA from post-evaluation with a justified stable trajectory window. "
        "Enable automatic analysis for standardized screening campaigns."
    )
    endpoint_backend = (
        "g_mmpbsa"
        if engines == [GROMACS_ENGINE]
        else "engine_native"
        if len(engines) > 1
        else "openmm_gbsa"
    )
    endpoint_start_pct = 20
    endpoint_target_frames = 400
    if endpoint_enabled:
        with st.expander("Advanced endpoint-energy settings"):
            endpoint_columns = st.columns(3)
            endpoint_options = (
                ["g_mmpbsa"]
                if engines == [GROMACS_ENGINE]
                else ["engine_native"]
                if len(engines) > 1
                else ["openmm_gbsa", "ambertools_mmpbsa"]
            )
            endpoint_backend = endpoint_columns[0].selectbox(
                "Endpoint engine",
                endpoint_options,
                help=(
                    "Parallel workflows use OpenMM GBSA for OpenMM and "
                    "g_mmpbsa for GROMACS. Endpoint calculations remain "
                    "separate immutable jobs."
                ),
            )
            endpoint_start_pct = int(
                endpoint_columns[1].number_input(
                    "Analyze final trajectory window (%)",
                    min_value=1,
                    max_value=100,
                    value=80,
                    step=5,
                    help=(
                        "80 means analyze the final 80% and discard the first 20%."
                    ),
                )
            )
            endpoint_target_frames = int(
                endpoint_columns[2].number_input(
                    "Target analyzed frames per replica",
                    min_value=10,
                    max_value=5000,
                    value=400,
                    step=50,
                )
            )
            st.warning(
                "MM/GBSA is an endpoint estimate, not an alchemical binding free "
                "energy. Compare only consistently prepared systems and protocols."
            )
    production_steps = max(
        1, int(round(production_ns * 1_000_000.0 / PRODUCTION_TIMESTEP_FS))
    )
    report_interval = max(
        1, int(round(report_interval_ps * 1000.0 / PRODUCTION_TIMESTEP_FS))
    )
    burn_in_steps = max(
        0, int(round(burn_in_ns * 1_000_000.0 / PRODUCTION_TIMESTEP_FS))
    )
    revalidation_max_steps = max(
        burn_in_steps,
        int(round(revalidation_max_ns * 1_000_000.0 / PRODUCTION_TIMESTEP_FS)),
    )
    revalidation_increment_ns = min(1.0, revalidation_max_ns)
    revalidation_increment_steps = (
        max(
            1,
            int(
                round(
                    revalidation_increment_ns
                    * 1_000_000.0
                    / PRODUCTION_TIMESTEP_FS
                )
            ),
        )
        if revalidation_max_steps > 0
        else 0
    )
    density_sample_interval_ps = 4.0
    density_sample_interval_steps = max(
        1,
        int(
            round(
                density_sample_interval_ps
                * 1000.0
                / PRODUCTION_TIMESTEP_FS
            )
        ),
    )
    stored_frames = max(1, production_steps // report_interval)
    endpoint_window_frames = max(
        1, int(stored_frames * endpoint_start_pct / 100.0)
    )
    endpoint_stride = max(
        1,
        int(
            math.ceil(
                endpoint_window_frames / max(1, endpoint_target_frames)
            )
        ),
    )
    if start_mode == EXACT_CONTINUATION:
        replicas = 1
        st.info("Exact continuation performs no minimization, velocity reassignment, or additional equilibration.")
    elif burn_in_steps == 0 and revalidation_max_steps > 0:
        st.caption(
            "The reusable prepared system has passed the Roe density-fit gate. "
            "After assigning distinct seeded velocities, each replica repeats "
            "only the unrecorded density fit and starts recorded production as "
            "soon as the plateau criteria pass."
        )
    else:
        st.caption(
            f"Continuous replica Roe equilibration: minimum {burn_in_ns:.3f} ns, "
            f"maximum {revalidation_max_ns:.3f} ns; recorded production starts "
            "without restarting after the density plateau passes. Equilibration "
            "time is excluded from the requested production duration and analysis."
        )
    timestep_summary = ", ".join(
        f"{md_engine_spec(engine).label} "
        f"{float(engine_settings.get(engine, {}).get('production_timestep_fs') or md_engine_spec(engine).production_timestep_fs):.1f} fs"
        for engine in engines
    )
    st.caption(
        f"Production timesteps: {timestep_summary} · "
        f"physical length {production_ns:.3f} ns/trajectory · approximately "
        f"{stored_frames:,} stored frames/trajectory at the requested interval"
        + (
            f" · endpoint stride {endpoint_stride}"
            if endpoint_enabled
            else ""
        )
    )
    production = {
        "production_steps": production_steps,
        "production_length_ns": production_ns,
        "production_timestep_fs": PRODUCTION_TIMESTEP_FS,
        "production_report_interval": report_interval,
        "production_report_interval_ps": report_interval_ps,
        "continuation_mode": start_mode,
        "replica_equilibration_steps": burn_in_steps if start_mode == INDEPENDENT_REPLICA else 0,
        "replica_equilibration_ns": burn_in_ns if start_mode == INDEPENDENT_REPLICA else 0.0,
        "replica_density_revalidation": (
            start_mode == INDEPENDENT_REPLICA
            and revalidation_max_steps > 0
        ),
        "replica_revalidation_max_steps": (
            revalidation_max_steps
            if start_mode == INDEPENDENT_REPLICA
            and revalidation_max_steps > 0
            else 0
        ),
        "replica_revalidation_max_ns": (
            revalidation_max_ns
            if start_mode == INDEPENDENT_REPLICA
            and revalidation_max_steps > 0
            else 0.0
        ),
        "replica_revalidation_increment_steps": (
            revalidation_increment_steps
            if start_mode == INDEPENDENT_REPLICA
            and revalidation_max_steps > 0
            else 0
        ),
        "replica_revalidation_increment_ns": (
            revalidation_increment_ns
            if start_mode == INDEPENDENT_REPLICA
            and revalidation_max_steps > 0
            else 0.0
        ),
        "replica_density_sample_interval_steps": (
            density_sample_interval_steps
            if start_mode == INDEPENDENT_REPLICA
            and revalidation_max_steps > 0
            else 0
        ),
        "replica_density_sample_interval_ps": density_sample_interval_ps,
        "replica_density_plateau_required": True,
        "endpoint_enabled": endpoint_enabled,
        "endpoint_backend": endpoint_backend,
        "endpoint_start_pct": 100 - endpoint_start_pct,
        "endpoint_end_pct": 100,
        "endpoint_stride": endpoint_stride,
        "endpoint_target_frames": endpoint_target_frames,
        # Legacy aliases are retained in metadata but production itself disables
        # embedded endpoint calculation when the worker input is activated.
        "mmgbsa_enabled": endpoint_enabled,
        "mmgbsa_start_pct": 100 - endpoint_start_pct,
        "mmgbsa_end_pct": 100,
        "mmgbsa_stride": endpoint_stride,
    }
    return production, replicas, analysis_enabled


def _workflow_stage_summary(workflow: WorkflowRecord) -> str:
    step_ids = tuple(workflow.expected_steps) or tuple(
        child.step_id for child in workflow.children
    )
    production_count = sum(
        step_id.startswith("production_replica_") for step_id in step_ids
    )
    endpoint_count = sum(
        step_id.startswith("endpoint_energy_replica_") for step_id in step_ids
    )
    stages = ["Preparation"]
    if production_count:
        stages.append(f"Production ×{production_count}")
    if endpoint_count:
        stages.append(f"MM/GBSA ×{endpoint_count}")
    if "replicate_analysis" in step_ids:
        stages.append("Aggregate analysis")
    return " → ".join(stages)


def _workflow_child_result_url(job: JobRecord) -> str:
    if job.task_group == "bound-ligand-md":
        return "./md-results?" + urlencode(
            {"run_type": "bound-ligand-md", "run_id": job.run_id}
        )
    if job.task_group == "md-system-prep":
        return "./md-results?" + urlencode(
            {"run_type": "md-system-prep", "run_id": job.run_id}
        )
    return "./job-results?" + urlencode(
        {"task_group": job.task_group, "run_id": job.run_id}
    )


def _workflow_step_label(step_id: str) -> str:
    if step_id == "preparation_equilibration":
        return "Preparation and equilibration"
    if step_id == "replicate_analysis":
        return "Aggregate replica analysis"
    if step_id.startswith("production_replica_"):
        return "Production replica " + step_id.removeprefix(
            "production_replica_"
        )
    if step_id.startswith("endpoint_energy_replica_"):
        return "Endpoint MM/GBSA replica " + step_id.removeprefix(
            "endpoint_energy_replica_"
        )
    return step_id.replace("_", " ").capitalize()


def _workflow_child_rows(
    workflow: WorkflowRecord,
) -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    required_states: list[str] = []
    for child in workflow.children:
        child_dir = resolve_run_dir(child.task_group, child.run_id)
        job = (
            JobRecord.load(
                child_dir,
                task_group=child.task_group,
                load_result=False,
                load_artifacts=False,
                validate_artifacts=False,
            )
            if child_dir is not None
            else None
        )
        status = job.status if job is not None else "missing"
        if child.required:
            required_states.append(status)
        rows.append(
            {
                "results": (
                    _workflow_child_result_url(job) if job is not None else ""
                ),
                "stage": _workflow_step_label(child.step_id),
                "job": (
                    display_job_code(
                        job.metadata.get("job_code"), job.run_id
                    )
                    if job is not None
                    else child.run_id[:8]
                ),
                "status": status,
                "detail": (
                    str(
                        job.metadata.get("error")
                        or job.metadata.get("status_reason")
                        or ""
                    )
                    if job is not None
                    else "Run folder is missing"
                ),
            }
        )

    attached_steps = {child.step_id for child in workflow.children}
    missing_expected = bool(set(workflow.expected_steps) - attached_steps)
    if any(
        state in {"running", "preparing", "paused"}
        for state in required_states
    ):
        display_status = "running"
    elif any(state == "failed" for state in required_states):
        display_status = "failed"
    elif any(state in {"blocked", "missing"} for state in required_states):
        display_status = "blocked"
    elif any(state == "cancelled" for state in required_states):
        display_status = "cancelled"
    elif any(state == "queued" for state in required_states) or missing_expected:
        display_status = "queued"
    elif required_states and all(
        state == "completed" for state in required_states
    ):
        display_status = "completed"
    else:
        display_status = workflow.status
    return rows, display_status


def _build_md_result_rows() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, list[dict[str, object]]],
]:
    workflow_rows: list[dict[str, object]] = []
    legacy_rows: list[dict[str, object]] = []
    workflow_children: dict[str, list[dict[str, object]]] = {}
    workflow_root = runs_root() / "workflows"
    if workflow_root.is_dir():
        workflows: list[WorkflowRecord] = []
        for path in workflow_root.iterdir():
            if not path.is_dir():
                continue
            try:
                workflow = WorkflowRecord.load(path.name)
            except (FileNotFoundError, TypeError, ValueError):
                continue
            if workflow.workflow_type != "md-simulation":
                continue
            workflows.append(workflow)

        workflows_by_id = {
            workflow.workflow_id: workflow for workflow in workflows
        }
        superseded_workflow_ids = {
            str(predecessor_id)
            for workflow in workflows
            for predecessor_id in (
                workflow.parameters.get("corrected_from_workflow_id"),
                workflow.parameters.get("replacement_of_invalid_workflow_id"),
            )
            if predecessor_id
        }
        for workflow in workflows:
            child_rows, display_status = _workflow_child_rows(workflow)
            if workflow.workflow_id in superseded_workflow_ids:
                display_status = "superseded"
            workflow_children[workflow.workflow_id] = child_rows
            job_code = display_job_code(None, workflow.workflow_id)
            description = workflow.name
            corrected_from = str(
                workflow.parameters.get("corrected_from_workflow_id") or ""
            )
            predecessor = workflows_by_id.get(corrected_from)
            if predecessor is not None:
                description = f"{predecessor.name} [corrected continuation]"
            workflow_rows.append(
                {
                    "results": "./job-results?"
                    + urlencode(
                        {
                            "task_group": "workflows",
                            "run_id": workflow.workflow_id,
                            "label": job_code,
                        }
                    ),
                    "description": description,
                    "engine": md_engine_spec(
                        workflow.parameters.get("engine") or OPENMM_ENGINE
                    ).label,
                    "comparison": str(
                        workflow.parameters.get("comparison_group_id") or ""
                    )[:8],
                    "replicas": workflow.parameters.get("replicas", ""),
                    "stages": _workflow_stage_summary(workflow),
                    "status": display_status,
                    "created": workflow.created_at,
                    "_workflow_id": workflow.workflow_id,
                }
            )
    for job in iter_job_records(
        runs_root(),
        task_groups=("bound-ligand-md",),
        load_result=False,
        load_artifacts=False,
        validate_artifacts=False,
    ):
        # Modern production jobs are children of an MD workflow. Showing them
        # beside their parent makes one campaign look like multiple unrelated
        # simulations; the workflow result page already exposes every child.
        if job.metadata.get("workflow_id") or job.metadata.get(
            "workflow_parent_run_id"
        ):
            continue
        job_code = display_job_code(
            job.metadata.get("job_code"), job.run_id
        )
        legacy_rows.append(
            {
                "results": "./md-results?"
                + urlencode(
                    {
                        "run_type": "bound-ligand-md",
                        "run_id": job.run_id,
                        "label": job_code,
                    }
                ),
                "description": (
                    f"{job.metadata.get('pdb_id') or '-'} · "
                    f"{job.metadata.get('ligand_label') or job.metadata.get('ligand_key') or '-'}"
                ),
                "replicas": job.metadata.get("repeat_index", ""),
                "status": job.status,
                "created": job.created_at,
            }
        )
    return (
        sorted(
            workflow_rows,
            key=lambda row: str(row.get("created") or ""),
            reverse=True,
        ),
        sorted(
            legacy_rows,
            key=lambda row: str(row.get("created") or ""),
            reverse=True,
        ),
        workflow_children,
    )


@st.cache_data(ttl=10, show_spinner=False)
def _cached_md_result_rows(
    run_root: str,
    run_root_mtime_ns: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, list[dict[str, object]]],
]:
    """Cache the expensive run-directory inventory between Streamlit reruns.

    The Results tab is evaluated even while another tab is visible.  Building
    its table requires opening every MD workflow plus all of their child job
    metadata, so keep a short-lived snapshot instead of repeating that work on
    every widget interaction.  The root mtime immediately invalidates the
    snapshot when a new run directory is created; the TTL covers status-only
    updates to existing runs.
    """
    del run_root_mtime_ns
    return _build_md_result_rows()


def _md_result_rows() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, list[dict[str, object]]],
]:
    root = runs_root()
    try:
        root_mtime_ns = root.stat().st_mtime_ns
    except OSError:
        root_mtime_ns = 0
    return _cached_md_result_rows(str(root), root_mtime_ns)


def render() -> None:
    st.title("MD Simulation")
    st.caption("Prepare, equilibrate, run independent production replicas, and aggregate their results as one workflow.")
    active_tab = st.segmented_control(
        "MD Simulation section",
        ["Target / Input", "Tool / Engine", "Run", "Results"],
        default="Target / Input",
        key="md_simulation_active_tab",
        label_visibility="collapsed",
    )

    # st.tabs renders every tab body on every interaction.  These controls can
    # traverse thousands of historical job records, so retain the current setup
    # in session state and render only the section the user selected.
    mode = str(
        st.session_state.get("md_simulation_mode") or "New MD simulation"
    )
    selected_source = st.session_state.get("md_simulation_selected_source")
    prep_settings_by_engine = st.session_state.get(
        "md_simulation_prep_settings", {}
    )
    selected_engines = st.session_state.get(
        "md_simulation_selected_engines", [OPENMM_ENGINE]
    )
    production = st.session_state.get("md_simulation_production", {})
    replicas = int(st.session_state.get("md_simulation_replicas", 0) or 0)
    analysis_enabled = bool(
        st.session_state.get("md_simulation_analysis_enabled", False)
    )

    if active_tab == "Target / Input":
        mode = st.segmented_control(
            "Starting point",
            ["New MD simulation", "Reuse prepared system"],
            default="New MD simulation",
            key="md_simulation_mode",
        )
        if mode == "New MD simulation":
            complex_source_mode = st.segmented_control(
                "Complex source",
                ["Curated complex datasets", "All prepared/docked complexes"],
                default="Curated complex datasets",
                help=(
                    "Curated datasets contain only poses explicitly imported "
                    "for MD and retain their prediction and selection lineage."
                ),
            )
            curated_jobs = selected_complex_jobs()
            curated_ids = {job.run_id for job in curated_jobs}
            curated_annotations = {
                job.run_id: {
                    "Dataset": str(
                        job.metadata.get("complex_dataset_name") or ""
                    ),
                    "Compound": str(job.metadata.get("compound_id") or ""),
                    "Selection status": str(
                        job.metadata.get("selection_status") or ""
                    ),
                    "Source prediction": str(
                        job.metadata.get("source_prediction_job_code") or ""
                    ),
                    "Prediction engine": str(job.metadata.get("engine") or ""),
                    "Replicate": job.metadata.get("replicate", ""),
                    "Prediction": str(job.metadata.get("prediction") or ""),
                }
                for job in curated_jobs
            }
            curated_mode = complex_source_mode == "Curated complex datasets"
            complex_choice = select_target_artifact(
                (
                    "Curated MD complex"
                    if curated_mode
                    else "Prepared or docked complex"
                ),
                (
                    ("prepared_complex",)
                    if curated_mode
                    else ("prepared_complex", "docked_complex")
                ),
                key=(
                    "md_curated_complex"
                    if curated_mode
                    else "md_prepared_complex"
                ),
                requested_run_id=str(
                    st.query_params.get("source_run_id", "")
                ).strip(),
                show_viewer=False,
                allowed_run_ids=curated_ids if curated_mode else None,
                excluded_run_ids=None if curated_mode else curated_ids,
                row_annotations=(
                    curated_annotations if curated_mode else None
                ),
            )
            if complex_choice is None:
                if curated_mode:
                    st.info(
                        "No curated MD complexes are available. Import a "
                        "selection workbook in Prepare → Complex Datasets."
                    )
                    st.link_button(
                        "Open Complex Datasets", "./prepare-complex-datasets"
                    )
                else:
                    st.info(
                        "No prepared complexes or docked complexes are available."
                    )
            else:
                if complex_choice.artifact.artifact_type == "docked_complex":
                    st.info(
                        "This docked pose will pass through the normal ligand "
                        "parameterization, solvation, minimization, and equilibration workflow."
                    )
                elif complex_choice.job.task_group == "selected-complexes":
                    st.info(
                        "This curated pose will enter the standard MD "
                        "parameterization, solvation, minimization, and "
                        "equilibration workflow. Its source prediction and "
                        "selection evidence remain linked in provenance."
                    )
                selected_source = (complex_choice.job, complex_choice.artifact)
                complex_path = target_viewer_path(complex_choice)
                ligand_key = str(complex_choice.job.metadata.get("ligand_key") or "")
                if (
                    not ligand_key
                    and complex_path is not None
                    and complex_path.suffix.lower() == ".pdb"
                ):
                    ligands = parse_bound_ligands(
                        complex_path.read_text(errors="replace")
                    )
                    ligand_key = str(ligands[0].get("key") or "") if ligands else ""
                render_target_viewer(
                    complex_choice,
                    viewer_path=complex_path,
                    box=bound_ligand_box(complex_choice, ligand_key)
                    if ligand_key
                    else None,
                    selected_ligand_key=ligand_key,
                    key="md_prepared_complex_viewer",
                )
        else:
            systems = _prepared_systems()
            if not systems:
                st.info("No completed prepared MD systems are available.")
            else:
                selected_index = st.selectbox(
                    "Prepared MD system",
                    range(len(systems)),
                    format_func=lambda index: _job_label(systems[index]),
                )
                selected_source = systems[int(selected_index)]
                fingerprint = str(
                    selected_source.metadata.get("compatibility_fingerprint") or ""
                )
                st.caption(
                    f"Compatibility fingerprint: `{fingerprint}`"
                    if fingerprint
                    else "A compatibility fingerprint will be generated at submission."
                )
        st.session_state["md_simulation_selected_source"] = selected_source

    if active_tab == "Tool / Engine":
        if mode == "New MD simulation":
            st.markdown("#### MD engines")
            action_columns = st.columns(2)
            action_columns[0].button(
                "Select all engines",
                on_click=_set_md_engine_selection,
                args=(True,),
                key="md_select_all_engines",
            )
            action_columns[1].button(
                "Deselect all engines",
                on_click=_set_md_engine_selection,
                args=(False,),
                key="md_deselect_all_engines",
            )
            engine_columns = st.columns(2)
            openmm_enabled = engine_columns[0].checkbox(
                "OpenMM",
                value=True,
                key=MD_ENGINE_KEYS[OPENMM_ENGINE],
                help="Select OpenMM and configure its preparation independently.",
            )
            gromacs_enabled = engine_columns[1].checkbox(
                "GROMACS",
                value=False,
                key=MD_ENGINE_KEYS[GROMACS_ENGINE],
                help="Select GROMACS and configure its preparation independently.",
            )
            selected_engines = []
            if openmm_enabled:
                selected_engines.append(OPENMM_ENGINE)
            if gromacs_enabled:
                selected_engines.append(GROMACS_ENGINE)
            if not selected_engines:
                st.warning("Select at least one MD engine.")
            _render_roe_protocol_info()
            st.markdown("#### Engine-specific preparation")
            prep_settings_by_engine = {}
            for engine in (OPENMM_ENGINE, GROMACS_ENGINE):
                with st.expander(
                    f"{md_engine_spec(engine).label} settings",
                    expanded=bool(
                        st.session_state.get(MD_ENGINE_KEYS[engine], False)
                    ),
                ):
                    prep_settings_by_engine[engine] = (
                        _engine_preparation_controls(engine)
                    )
            if len(selected_engines) > 1:
                openmm_settings = prep_settings_by_engine.get(OPENMM_ENGINE, {})
                gromacs_settings = prep_settings_by_engine.get(
                    GROMACS_ENGINE, {}
                )
                matched_keys = (
                    "protein_forcefield_method",
                    "forcefield_method",
                    "water_model",
                    "charge_method",
                    "box_shape",
                    "padding_nm",
                    "ionic_strength",
                    "temperature",
                    "pressure",
                )
                if all(
                    openmm_settings.get(key) == gromacs_settings.get(key)
                    for key in matched_keys
                ):
                    st.success(
                        "The selected engines use matched preparation settings. "
                        "Their native trajectories remain independent."
                    )
                else:
                    st.warning(
                        "The selected engines use different physical models or "
                        "conditions. Compare them as separate protocols, not as "
                        "an engine-only benchmark or pooled replicas."
                    )
        else:
            st.info("System and equilibration settings are inherited.")
            inherited_contract = (
                selected_source.metadata.get("compatibility_contract") or {}
                if selected_source is not None
                else {}
            )
            inherited_engine = normalize_md_engine(
                (
                    selected_source.metadata.get("md_engine")
                    if selected_source is not None
                    else None
                )
                or inherited_contract.get("engine")
                or OPENMM_ENGINE
            )
            selected_engines = [inherited_engine]
            st.caption(
                f"Prepared-system engine: {md_engine_spec(inherited_engine).label}"
            )
        production, replicas, analysis_enabled = _production_controls(
            selected_engines,
            prep_settings_by_engine,
        )
        st.session_state["md_simulation_prep_settings"] = prep_settings_by_engine
        st.session_state["md_simulation_selected_engines"] = selected_engines
        st.session_state["md_simulation_production"] = production
        st.session_state["md_simulation_replicas"] = replicas
        st.session_state["md_simulation_analysis_enabled"] = analysis_enabled

    if active_tab == "Run":
        st.markdown("#### Runtime")
        use_gpu = st.checkbox("Use GPU", value=True)
        render_run_resources(
            requires_gpu=use_gpu,
            selected_gpu="Automatic" if use_gpu else "Not used",
            key="md_simulation",
        )
        if selected_source is None:
            st.info("Select a compatible input in Target / Input before submitting.")
            st.link_button(
                "Open Structure Import", "./workspace-structure-preparation"
            )
        if st.button(
            "Submit MD workflow",
            type="primary",
            disabled=selected_source is None or not selected_engines,
        ):
            try:
                workflows = []
                if mode == "New MD simulation":
                    source_job, source_artifact = selected_source
                    comparison_group_id = (
                        str(uuid4()) if len(selected_engines) > 1 else ""
                    )
                    for engine in selected_engines:
                        workflow = create_md_simulation(
                            source_job=source_job,
                            source_artifact=source_artifact,
                            prep_input=_new_prep_input(
                                source_job,
                                source_artifact,
                                prep_settings_by_engine[engine],
                            ),
                            production=_production_for_engine(
                                production,
                                engine,
                                prep_settings_by_engine[engine],
                            ),
                            replicas=replicas,
                            analysis_enabled=analysis_enabled,
                            image=md_engine_spec(engine).default_image,
                            use_gpu=use_gpu,
                            engine=engine,
                            comparison_group_id=comparison_group_id,
                        )
                        workflows.append(workflow)
                else:
                    workflow = create_md_simulation_from_prepared(
                        prep_job=selected_source,
                        production=_production_for_engine(
                            production,
                            selected_engines[0],
                        ),
                        replicas=replicas,
                        analysis_enabled=analysis_enabled,
                        image=md_engine_spec(
                            selected_engines[0]
                        ).default_image,
                        use_gpu=use_gpu,
                    )
                    workflows.append(workflow)
                if len(workflows) == 1:
                    st.success(
                        f"Created MD workflow `{workflows[0].workflow_id}`."
                    )
                else:
                    st.success(
                        "Created parallel OpenMM and GROMACS workflows: "
                        + ", ".join(
                            f"`{item.workflow_id}`" for item in workflows
                        )
                        + "."
                    )
                with st.spinner("Starting queued MD children..."):
                    try_dispatch_next_queued_gpu_job()
            except Exception as exc:
                st.error(f"Could not create MD workflow: {exc}")

    if active_tab == "Results":
        workflow_rows, legacy_rows, _workflow_children = _md_result_rows()
        if workflow_rows:
            st.subheader("MD workflows")
            st.caption(
                "Each row is one complete MD campaign. Open it to inspect its "
                "preparation, production replicas, endpoint calculations and "
                "aggregate analysis together."
            )
            show_workflow_history = st.checkbox(
                "Show failed, blocked, cancelled, and superseded MD campaign history",
                value=False,
                key="md_results_show_workflow_history",
                help=(
                    "Historical and replaced campaigns remain stored and can "
                    "be shown for diagnostics, but are hidden from the current "
                    "campaign view by default."
                ),
            )
            visible_workflow_rows = visible_md_workflow_rows(
                workflow_rows,
                show_history=show_workflow_history,
            )
            hidden_workflow_count = (
                len(workflow_rows) - len(visible_workflow_rows)
            )
            if hidden_workflow_count:
                st.caption(
                    f"{hidden_workflow_count} historical MD campaign(s) hidden."
                )
            if not visible_workflow_rows:
                st.info("No current MD workflows are available.")
                visible_workflow_rows = []
            st.dataframe(
                pd.DataFrame(visible_workflow_rows).drop(
                    columns=["_workflow_id"],
                    errors="ignore",
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "results": st.column_config.LinkColumn(
                        "Job", display_text=r"label=([^&]+)"
                    ),
                    "description": st.column_config.TextColumn(
                        "System", width="large"
                    ),
                    "replicas": st.column_config.NumberColumn("Replicas"),
                    "stages": st.column_config.TextColumn(
                        "Stages", width="large"
                    ),
                    "status": st.column_config.TextColumn("Status"),
                    "created": st.column_config.DatetimeColumn("Created"),
                },
            )
        else:
            st.info("No combined MD workflows are available.")

        if legacy_rows:
            with st.expander(
                f"Legacy standalone production trajectories ({len(legacy_rows)})"
            ):
                st.caption(
                    "These trajectories predate the combined workflow model and "
                    "have no recorded workflow parent. They cannot be grouped "
                    "reliably with preparation or analysis jobs."
                )
                st.dataframe(
                    pd.DataFrame(legacy_rows),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "results": st.column_config.LinkColumn(
                            "Job", display_text=r"label=([^&]+)"
                        ),
                        "description": st.column_config.TextColumn(
                            "System", width="large"
                        ),
                        "replicas": st.column_config.NumberColumn("Replica"),
                        "status": st.column_config.TextColumn("Status"),
                        "created": st.column_config.DatetimeColumn("Created"),
                    },
                )


render()
