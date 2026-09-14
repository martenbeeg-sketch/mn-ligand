from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OPENMM_ENGINE = "openmm"
GROMACS_ENGINE = "gromacs"
SUPPORTED_MD_ENGINES = (OPENMM_ENGINE, GROMACS_ENGINE)


@dataclass(frozen=True)
class MDEngineSpec:
    engine_id: str
    label: str
    tool_id: str
    default_image: str
    production_timestep_fs: float
    trajectory_format: str
    checkpoint_format: str
    endpoint_backends: tuple[str, ...]


_ENGINE_SPECS = {
    OPENMM_ENGINE: MDEngineSpec(
        engine_id=OPENMM_ENGINE,
        label="OpenMM",
        tool_id="openmm_md",
        default_image="ovolig-md-cu128:latest",
        production_timestep_fs=4.0,
        trajectory_format="dcd",
        checkpoint_format="openmm-checkpoint",
        endpoint_backends=("openmm_gbsa", "ambertools_mmpbsa"),
    ),
    GROMACS_ENGINE: MDEngineSpec(
        engine_id=GROMACS_ENGINE,
        label="GROMACS",
        tool_id="gromacs_md",
        default_image="ovolig-gromacs-cu128:latest",
        production_timestep_fs=4.0,
        trajectory_format="xtc",
        checkpoint_format="gromacs-cpt",
        endpoint_backends=("g_mmpbsa", "ambertools_mmpbsa"),
    ),
}


def normalize_md_engine(value: Any) -> str:
    engine = str(value or OPENMM_ENGINE).strip().lower()
    aliases = {
        "open mm": OPENMM_ENGINE,
        "gmx": GROMACS_ENGINE,
        "gromcas": GROMACS_ENGINE,
    }
    engine = aliases.get(engine, engine)
    if engine not in _ENGINE_SPECS:
        raise ValueError(f"Unsupported MD engine: {value}")
    return engine


def md_engine_spec(value: Any) -> MDEngineSpec:
    return _ENGINE_SPECS[normalize_md_engine(value)]


def endpoint_backend_supported(engine: Any, backend: Any) -> bool:
    selected = str(backend or "").strip().lower()
    return selected in md_engine_spec(engine).endpoint_backends


def engine_restart_artifact_types(engine: Any, *, exact: bool) -> set[str]:
    selected = normalize_md_engine(engine)
    if selected == GROMACS_ENGINE:
        return {
            "md_topology",
            "equilibrated_system",
            "md_checkpoint",
            "md_index",
        }
    if exact:
        return {
            "md_checkpoint",
            "openmm_system",
            "openmm_integrator",
        }
    return {
        "openmm_state",
        "openmm_system",
        "openmm_integrator",
    }


def roe_brooks_stage_specification(
    *,
    temperature_k: float = 300.0,
    pressure_bar: float = 1.0,
) -> tuple[dict[str, Any], ...]:
    return (
        {
            "stage": "1",
            "kind": "minimization",
            "steps": 1000,
            "restraint_k_kcal_mol_a2": 5.0,
            "restraint_selection": "large_molecule_heavy",
            "constraints": "none",
            "precision": "double",
        },
        {
            "stage": "2",
            "kind": "nvt",
            "steps": 15000,
            "timestep_fs": 1.0,
            "duration_ps": 15.0,
            "restraint_k_kcal_mol_a2": 5.0,
            "restraint_selection": "large_molecule_heavy",
            "generate_velocities": True,
            "temperature_k": float(temperature_k),
        },
        {
            "stage": "3",
            "kind": "minimization",
            "steps": 1000,
            "restraint_k_kcal_mol_a2": 2.0,
            "restraint_selection": "large_molecule_heavy",
            "constraints": "none",
            "precision": "double",
        },
        {
            "stage": "4",
            "kind": "minimization",
            "steps": 1000,
            "restraint_k_kcal_mol_a2": 0.1,
            "restraint_selection": "large_molecule_heavy",
            "constraints": "none",
            "precision": "double",
        },
        {
            "stage": "5",
            "kind": "minimization",
            "steps": 1000,
            "restraint_k_kcal_mol_a2": 0.0,
            "restraint_selection": "none",
            "constraints": "none",
            "precision": "double",
        },
        {
            "stage": "6",
            "kind": "npt",
            "steps": 5000,
            "timestep_fs": 1.0,
            "duration_ps": 5.0,
            "restraint_k_kcal_mol_a2": 1.0,
            "restraint_selection": "large_molecule_heavy",
            "generate_velocities": True,
            "temperature_k": float(temperature_k),
            "pressure_bar": float(pressure_bar),
        },
        {
            "stage": "7",
            "kind": "npt",
            "steps": 5000,
            "timestep_fs": 1.0,
            "duration_ps": 5.0,
            "restraint_k_kcal_mol_a2": 0.5,
            "restraint_selection": "large_molecule_heavy",
            "generate_velocities": False,
            "temperature_k": float(temperature_k),
            "pressure_bar": float(pressure_bar),
        },
        {
            "stage": "8",
            "kind": "npt",
            "steps": 10000,
            "timestep_fs": 1.0,
            "duration_ps": 10.0,
            "restraint_k_kcal_mol_a2": 0.5,
            "restraint_selection": "polymer_backbone_and_ligand_heavy",
            "generate_velocities": False,
            "temperature_k": float(temperature_k),
            "pressure_bar": float(pressure_bar),
        },
        {
            "stage": "9",
            "kind": "npt",
            "steps": 5000,
            "timestep_fs": 2.0,
            "duration_ps": 10.0,
            "restraint_k_kcal_mol_a2": 0.0,
            "restraint_selection": "none",
            "generate_velocities": False,
            "temperature_k": float(temperature_k),
            "pressure_bar": float(pressure_bar),
        },
        {
            "stage": "10",
            "kind": "density_stabilization_npt",
            "increment_ns": 1.0,
            "restraint_k_kcal_mol_a2": 0.0,
            "restraint_selection": "none",
            "generate_velocities": False,
            "temperature_k": float(temperature_k),
            "pressure_bar": float(pressure_bar),
        },
    )
