from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from mn_ligand.ligandx.services.md.workflow.analytics import (
    EquilibrationAnalytics,
    ligand_formal_charges_from_sdf_data,
)
from mn_ligand.ligandx.services.md.workflow.equilibration_runner import (
    fit_density_plateau,
)
from mn_ligand.workflows.bound_ligand_md import (
    _prepare_ambertools_topology_artifacts,
    parse_bound_ligands,
)
from mn_ligand.workflows.gromacs_protocol import (
    atom_groups,
    parse_xvg_series,
    render_gromacs_mdp,
    write_gromacs_index,
    write_molecule_type_restraints,
    write_roe_brooks_mdp_set,
)
from mn_ligand.workflows.md_engines import GROMACS_ENGINE


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def parse_gromacs_performance(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {
            "ns_per_day": None,
            "source": "unavailable",
            "sample_count": 0,
        }
    values = [
        float(match.group(1))
        for match in re.finditer(
            r"^Performance:\s+([0-9]+(?:\.[0-9]+)?)",
            log_path.read_text(errors="replace"),
            flags=re.MULTILINE,
        )
    ]
    return {
        "ns_per_day": values[-1] if values else None,
        "source": "GROMACS production.log",
        "sample_count": len(values),
    }


def _run(
    command: list[str],
    *,
    cwd: Path,
    stdin: str | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    if log_path is not None:
        with log_path.open("a") as handle:
            handle.write("$ " + shlex.join(command) + "\n")
            handle.write(completed.stdout)
            handle.write(completed.stderr)
            handle.write("\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"{completed.stderr[-4000:]}"
        )
    return completed


def _gmx_binary(*, double_precision: bool = False) -> str:
    variable = "GMX_DOUBLE_BIN" if double_precision else "GMX_BIN"
    default = "gmx_d" if double_precision else "gmx"
    binary = os.getenv(variable, default).strip() or default
    if shutil.which(binary) is None:
        raise RuntimeError(f"Required GROMACS executable is unavailable: {binary}")
    return binary


def _g_mmpbsa_gmx_binary() -> str:
    binary = os.getenv(
        "GMX_MMPBSA_BIN",
        "/opt/gromacs-mmpbsa/bin/gmx",
    ).strip()
    if not binary or shutil.which(binary) is None:
        raise RuntimeError(
            "The GROMACS 2025 compatibility executable required by "
            "g_mmpbsa is unavailable"
        )
    return binary


def _gpu_mdrun_flags() -> list[str]:
    return shlex.split(
        os.getenv(
            "MN_GROMACS_GPU_FLAGS",
            "-nb gpu -pme gpu -bonded gpu -update gpu",
        )
    )


def _stage_gpu_mdrun_flags(stage: dict[str, Any]) -> list[str]:
    flags = _gpu_mdrun_flags()
    if (
        float(stage.get("timestep_fs") or 0.0) <= 2.0
        or float(stage.get("mass_repartition_factor") or 1.0) > 1.0
    ):
        return flags
    filtered: list[str] = []
    index = 0
    while index < len(flags):
        if flags[index : index + 2] == ["-update", "gpu"]:
            index += 2
            continue
        filtered.append(flags[index])
        index += 1
    return filtered


def _run_stage(
    *,
    output_dir: Path,
    stage: dict[str, Any],
    start_coordinates: Path,
    reference_coordinates: Path,
    topology: Path,
    index: Path,
    previous_checkpoint: Path | None,
    use_gpu: bool,
    log_path: Path,
    name: str | None = None,
) -> tuple[Path, Path | None, Path]:
    stage_name = name or f"roe_{stage['stage']}"
    double_precision = str(stage.get("precision")) == "double"
    gmx = _gmx_binary(double_precision=double_precision)
    tpr = output_dir / f"{stage_name}.tpr"
    command = [
        gmx,
        "grompp",
        "-f",
        str(output_dir / str(stage["mdp"])),
        "-c",
        str(start_coordinates),
        "-r",
        str(reference_coordinates),
        "-p",
        str(topology),
        "-n",
        str(index),
        "-o",
        str(tpr),
        "-maxwarn",
        "1",
    ]
    if previous_checkpoint is not None and str(stage["kind"]) != "minimization":
        command.extend(["-t", str(previous_checkpoint)])
    _run(command, cwd=output_dir, log_path=log_path)
    mdrun = [gmx, "mdrun", "-deffnm", stage_name]
    if use_gpu and not double_precision:
        mdrun.extend(_stage_gpu_mdrun_flags(stage))
    _run(mdrun, cwd=output_dir, log_path=log_path)
    coordinates = output_dir / f"{stage_name}.gro"
    checkpoint = output_dir / f"{stage_name}.cpt"
    energy = output_dir / f"{stage_name}.edr"
    if not coordinates.is_file() or not energy.is_file():
        raise RuntimeError(f"GROMACS stage {stage_name} did not produce expected outputs")
    return coordinates, checkpoint if checkpoint.is_file() else None, energy


def _extract_density(
    energy_files: list[Path],
    output_dir: Path,
    log_path: Path,
) -> tuple[list[float], list[float]]:
    combined_times: list[float] = []
    combined_density: list[float] = []
    offset = 0.0
    for file_index, energy_file in enumerate(energy_files, start=1):
        xvg = output_dir / f"density_{file_index:03d}.xvg"
        _run(
            [
                _gmx_binary(),
                "energy",
                "-f",
                str(energy_file),
                "-o",
                str(xvg),
                "-xvg",
                "none",
            ],
            cwd=output_dir,
            stdin="Density\n0\n",
            log_path=log_path,
        )
        times, densities = parse_xvg_series(xvg)
        if not times:
            continue
        shifted = [offset + value - times[0] for value in times]
        combined_times.extend(shifted)
        combined_density.extend(
            density_kg_m3 / 1000.0
            for density_kg_m3 in densities
        )
        offset = shifted[-1]
    return combined_times, combined_density


def _selected_ligand(config: dict[str, Any], pdb_data: str) -> dict[str, Any]:
    selected_key = str(config.get("ligand_key") or "")
    ligands = parse_bound_ligands(pdb_data)
    selected = next(
        (ligand for ligand in ligands if ligand.get("key") == selected_key),
        ligands[0] if ligands else None,
    )
    if selected is None:
        raise ValueError("The GROMACS input contains no selectable ligand")
    return selected


def _export_amber_system(
    config: dict[str, Any],
    source_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    selected = _selected_ligand(config, source_path.read_text())
    amber = _prepare_ambertools_topology_artifacts(
        {**config, "mmgbsa_backend": "ambertools_mmpbsa"},
        selected,
        {"output_files": {"system_pdb": str(source_path)}},
        output_dir,
    )
    if amber.get("status") != "success":
        raise RuntimeError(f"Amber/GAFF2 preparation failed: {amber}")
    try:
        import parmed
    except ImportError as exc:
        raise RuntimeError("ParmEd is required for Amber-to-GROMACS export") from exc
    files = amber.get("files") or {}
    structure = parmed.load_file(
        str(files["complex_prmtop"]),
        str(files["complex_inpcrd"]),
    )
    hydrogen_mass_amu = config.get("hydrogen_mass_amu")
    mass_repartition_factor = config.get("mass_repartition_factor")
    if (
        mass_repartition_factor is None
        and hydrogen_mass_amu is not None
        and float(config.get("production_timestep_fs") or 2.0) > 2.0
    ):
        mass_repartition_factor = 3.0
    amber["integration_profile"] = {
        "name": str(config.get("integration_profile") or "standard_2fs"),
        "production_timestep_fs": float(
            config.get("production_timestep_fs") or 2.0
        ),
        "hydrogen_mass_amu": (
            float(hydrogen_mass_amu)
            if hydrogen_mass_amu is not None
            else None
        ),
        "mass_repartition_factor": (
            float(mass_repartition_factor)
            if mass_repartition_factor is not None
            else None
        ),
        "mass_repartition_method": (
            "gromacs_grompp"
            if mass_repartition_factor is not None
            else "none"
        ),
    }
    topology = output_dir / "system.top"
    coordinates = output_dir / "system.gro"
    system_pdb = output_dir / "system.pdb"
    structure.save(str(topology), overwrite=True)
    structure.save(str(coordinates), overwrite=True)
    structure.save(str(system_pdb), overwrite=True)
    groups = atom_groups(structure)
    index = output_dir / "index.ndx"
    write_gromacs_index(index, groups)
    write_molecule_type_restraints(topology)
    return topology, coordinates, system_pdb, index, amber


def prepare_gromacs_system(
    config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(
        str(
            config.get("input_complex_pdb_path")
            or config.get("prepared_complex_path")
            or ""
        )
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"Prepared complex is unavailable: {source_path}")
    topology, coordinates, system_pdb, index, amber = _export_amber_system(
        config,
        source_path,
        output_dir,
    )
    temperature = float(config.get("temperature") or 300.0)
    pressure = float(config.get("pressure") or 1.0)
    density_interval = float(config.get("density_sample_interval_ps") or 4.0)
    stages = write_roe_brooks_mdp_set(
        output_dir,
        temperature_k=temperature,
        pressure_bar=pressure,
        density_sample_interval_ps=density_interval,
        production_timestep_fs=float(
            config.get("production_timestep_fs") or 2.0
        ),
        mass_repartition_factor=(
            float(
                (amber.get("integration_profile") or {})[
                    "mass_repartition_factor"
                ]
            )
            if (amber.get("integration_profile") or {}).get(
                "mass_repartition_factor"
            )
            is not None
            else None
        ),
        random_seed=int(config.get("replica_seed") or -1),
    )
    protocol_path = output_dir / "roe_brooks_protocol.json"
    protocol_report: dict[str, Any] = {
        "protocol": "roe_brooks_2020",
        "engine": GROMACS_ENGINE,
        "paper": "Roe and Brooks, J. Chem. Phys. 153, 054123 (2020)",
        "stages": stages,
        "coordinate_wrapping_during_preparation": False,
        "parameterization": amber.get("method"),
        "integration_profile": amber.get("integration_profile"),
    }
    _write_json(protocol_path, protocol_report)
    log_path = output_dir / "gromacs_preparation.log"
    current = coordinates
    initial_reference = coordinates
    previous_checkpoint: Path | None = None
    step5_reference: Path | None = None
    completed_rows: list[dict[str, Any]] = []
    for stage in stages[:9]:
        reference = (
            step5_reference
            if str(stage["stage"]) in {"6", "7", "8"}
            and step5_reference is not None
            else initial_reference
        )
        current, previous_checkpoint, energy = _run_stage(
            output_dir=output_dir,
            stage=stage,
            start_coordinates=current,
            reference_coordinates=reference,
            topology=topology,
            index=index,
            previous_checkpoint=previous_checkpoint,
            use_gpu=bool(config.get("use_gpu", True)),
            log_path=log_path,
        )
        if str(stage["stage"]) == "5":
            step5_reference = current
        completed_rows.append(
            {
                **stage,
                "coordinates": current.name,
                "checkpoint": previous_checkpoint.name
                if previous_checkpoint is not None
                else "",
                "energy": energy.name,
            }
        )

    density_min_ns = float(config.get("density_stabilization_min_ns") or 1.0)
    density_max_ns = float(config.get("density_stabilization_max_ns") or 5.0)
    density_increment_ns = float(
        config.get("density_stabilization_increment_ns") or 1.0
    )
    if (
        density_min_ns <= 0
        or density_increment_ns <= 0
        or density_max_ns < density_min_ns
    ):
        raise ValueError("Invalid Roe-Brooks density stabilization window")
    density_stage = dict(stages[9])
    density_energy_files: list[Path] = []
    elapsed_ns = 0.0
    density_fit: dict[str, Any] = {
        "plateau": False,
        "reason": "Density stabilization has not been evaluated",
    }
    density_times: list[float] = []
    densities: list[float] = []
    segment = 0
    while elapsed_ns + 1.0e-12 < density_max_ns:
        segment += 1
        segment_ns = min(density_increment_ns, density_max_ns - elapsed_ns)
        density_stage["steps"] = max(
            1,
            int(round(segment_ns * 1_000_000.0 / 2.0)),
        )
        (output_dir / str(density_stage["mdp"])).write_text(
            render_gromacs_mdp(
                density_stage,
                density_sample_interval_ps=density_interval,
            )
        )
        current, previous_checkpoint, energy = _run_stage(
            output_dir=output_dir,
            stage=density_stage,
            start_coordinates=current,
            reference_coordinates=current,
            topology=topology,
            index=index,
            previous_checkpoint=previous_checkpoint,
            use_gpu=bool(config.get("use_gpu", True)),
            log_path=log_path,
            name=f"roe_10_{segment:03d}",
        )
        density_energy_files.append(energy)
        elapsed_ns += segment_ns
        density_times, densities = _extract_density(
            density_energy_files,
            output_dir,
            log_path,
        )
        if elapsed_ns + 1.0e-12 >= density_min_ns:
            density_fit = fit_density_plateau(density_times, densities)
            if density_fit.get("plateau") is True:
                break

    density_csv = output_dir / "density_stabilization.csv"
    with density_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_ps", "density_g_ml"))
        writer.writerows(zip(density_times, densities))
    density_report = {
        "protocol": "Roe-Brooks 2020 GROMACS",
        "plateau_required": bool(config.get("density_plateau_required", True)),
        "stabilization_ns": elapsed_ns,
        "minimum_ns": density_min_ns,
        "maximum_ns": density_max_ns,
        "increment_ns": density_increment_ns,
        "sample_interval_ps": density_interval,
        "fit": density_fit,
    }
    density_json = output_dir / "density_stabilization.json"
    _write_json(density_json, density_report)
    if (
        bool(config.get("density_plateau_required", True))
        and density_fit.get("plateau") is not True
    ):
        raise RuntimeError(
            "Roe-Brooks density plateau criteria were not satisfied within "
            f"{density_max_ns:.3f} ns"
        )
    if previous_checkpoint is None:
        raise RuntimeError("GROMACS preparation produced no final checkpoint")
    final_coordinates = output_dir / "equilibrated.gro"
    final_checkpoint = output_dir / "equilibrated.cpt"
    shutil.copy2(current, final_coordinates)
    shutil.copy2(previous_checkpoint, final_checkpoint)
    protocol_report["stages"] = completed_rows + [
        {
            **density_stage,
            "stage": "10",
            "segments": segment,
            "duration_ns": elapsed_ns,
            "density_plateau": density_fit.get("plateau") is True,
        }
    ]
    protocol_report["density_stabilization"] = density_report
    _write_json(protocol_path, protocol_report)
    result = {
        "success": True,
        "engine": GROMACS_ENGINE,
        "md_result": {
            "engine": GROMACS_ENGINE,
            "preparation_protocol": protocol_report,
            "output_files": {
                "system_pdb": str(system_pdb),
                "npt_pdb": str(system_pdb),
                "npt_checkpoint": str(final_checkpoint),
                "gromacs_topology": str(topology),
                "gromacs_coordinates": str(final_coordinates),
                "gromacs_checkpoint": str(final_checkpoint),
                "gromacs_index": str(index),
                "gromacs_tpr": str(output_dir / f"roe_10_{segment:03d}.tpr"),
                "amber_complex_prmtop": str(
                    (amber.get("files") or {}).get("complex_prmtop") or ""
                ),
                "amber_complex_inpcrd": str(
                    (amber.get("files") or {}).get("complex_inpcrd") or ""
                ),
                "amber_com_prmtop": str(
                    (amber.get("files") or {}).get("com_prmtop") or ""
                ),
                "amber_rec_prmtop": str(
                    (amber.get("files") or {}).get("rec_prmtop") or ""
                ),
                "amber_lig_prmtop": str(
                    (amber.get("files") or {}).get("lig_prmtop") or ""
                ),
                "density_report": str(density_json),
                "density_series": str(density_csv),
                "protocol_report": str(protocol_path),
            },
        },
    }
    _write_json(output_path, result)
    return result


def _copy_prepared_file(
    config: dict[str, Any],
    key: str,
    output_dir: Path,
) -> Path:
    source = Path(str(config.get(key) or ""))
    if not source.is_file():
        raise FileNotFoundError(
            f"Prepared GROMACS file is unavailable: {key}={source}"
        )
    target = output_dir / source.name
    shutil.copy2(source, target)
    return target


def _copy_continuation_file(
    config: dict[str, Any],
    key: str,
    output_dir: Path,
    name: str,
) -> Path:
    source = Path(str(config.get(key) or ""))
    if not source.is_file():
        raise FileNotFoundError(
            f"Strict GROMACS continuation is missing {key}: {source}"
        )
    target = output_dir / name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _run_gromacs_continuation(
    config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Extend one production run using the native GROMACS append contract."""
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    topology = _copy_continuation_file(
        config, "source_topology_path", output_dir, "system.top"
    )
    index = _copy_continuation_file(
        config, "source_index_path", output_dir, "index.ndx"
    )
    source_tpr = _copy_continuation_file(
        config, "source_tpr_path", output_dir, "production_source.tpr"
    )
    checkpoint = _copy_continuation_file(
        config, "source_checkpoint_path", output_dir, "production.cpt"
    )
    native_trajectory = _copy_continuation_file(
        config,
        "source_native_trajectory_path",
        output_dir,
        "production.xtc",
    )
    energy = _copy_continuation_file(
        config, "source_energy_path", output_dir, "production.edr"
    )
    native_log = _copy_continuation_file(
        config, "source_log_path", output_dir, "production.log"
    )
    source_topology_dir = Path(str(config.get("source_topology_path") or "")).parent
    for source_restraint in source_topology_dir.glob("*.itp"):
        shutil.copy2(source_restraint, output_dir / source_restraint.name)

    extension_steps = int(config.get("production_steps") or 0)
    prior_steps = int(config.get("production_prior_steps") or 0)
    timestep_fs = float(config.get("production_timestep_fs") or 2.0)
    if extension_steps <= 0 or prior_steps <= 0 or timestep_fs <= 0:
        raise ValueError(
            "Strict GROMACS continuation requires positive prior steps, "
            "extension steps and timestep"
        )
    extension_ps = extension_steps * timestep_fs / 1000.0
    cumulative_steps = prior_steps + extension_steps
    cumulative_ps = cumulative_steps * timestep_fs / 1000.0
    tpr = output_dir / "production.tpr"
    wrapper_log = output_dir / "gromacs_continuation.log"
    gmx = _gmx_binary()
    _run(
        [
            gmx,
            "convert-tpr",
            "-s",
            str(source_tpr),
            "-extend",
            str(extension_ps),
            "-o",
            str(tpr),
        ],
        cwd=output_dir,
        log_path=wrapper_log,
    )
    command = [
        gmx,
        "mdrun",
        "-s",
        str(tpr),
        "-deffnm",
        "production",
        "-cpi",
        str(checkpoint),
        "-append",
    ]
    if bool(config.get("use_gpu", True)):
        command.extend(_gpu_mdrun_flags())
    _run(command, cwd=output_dir, log_path=wrapper_log)

    final_gro = output_dir / "production.gro"
    final_checkpoint = output_dir / "production.cpt"
    if not all(
        path.is_file()
        for path in (
            native_trajectory,
            energy,
            native_log,
            final_gro,
            final_checkpoint,
            tpr,
        )
    ):
        raise RuntimeError(
            "GROMACS continuation did not produce its complete append artifact set"
        )
    trajectory = output_dir / "production_whole.xtc"
    final_pdb = output_dir / "production.pdb"
    _run(
        [
            gmx, "trjconv", "-s", str(tpr), "-f", str(native_trajectory),
            "-o", str(trajectory), "-n", str(index), "-pbc", "mol",
            "-center", "-ur", "compact",
        ],
        cwd=output_dir,
        stdin="Protein_Ligand\nSystem\n",
        log_path=wrapper_log,
    )
    _run(
        [
            gmx, "trjconv", "-s", str(tpr), "-f", str(trajectory),
            "-o", str(final_pdb), "-n", str(index), "-dump",
            str(cumulative_ps),
        ],
        cwd=output_dir,
        stdin="System\n",
        log_path=wrapper_log,
    )
    analytics = EquilibrationAnalytics().compute(
        output_dir=str(output_dir),
        system_id=str(config.get("system_id") or "gromacs"),
        topology_pdb=str(final_pdb),
        production_traj=str(trajectory),
        ligand_id="LIG",
        production_steps=cumulative_steps,
        production_report_interval=max(
            1, int(config.get("production_report_interval") or 5000)
        ),
        dt_ps=timestep_fs / 1000.0,
        residue_mapping=(
            config.get("residue_mapping")
            if isinstance(config.get("residue_mapping"), dict)
            else None
        ),
        ligand_formal_charges=ligand_formal_charges_from_sdf_data(
            str(config.get("ligand_refined_sdf_data") or "")
        ),
    )
    analytics["performance"] = parse_gromacs_performance(native_log)
    analytics["continuation"] = {
        "mode": "gromacs_cpt_append",
        "prior_steps": prior_steps,
        "extension_steps": extension_steps,
        "cumulative_steps": cumulative_steps,
        "cumulative_duration_ns": cumulative_ps / 1000.0,
    }
    analysis_report = output_dir / "trajectory_analysis.json"
    _write_json(analysis_report, analytics)
    result = {
        "success": True,
        "engine": GROMACS_ENGINE,
        "md_result": {
            "engine": GROMACS_ENGINE,
            "analytics": analytics,
            "continuation": analytics["continuation"],
            "output_files": {
                "production_trajectory": str(trajectory),
                "native_production_trajectory": str(native_trajectory),
                "production_pdb": str(final_pdb),
                "production_checkpoint": str(final_checkpoint),
                "production_topology": str(topology),
                "production_tpr": str(tpr),
                "production_index": str(index),
                "thermodynamic_series": str(energy),
                "production_log": str(native_log),
                "analysis_report": str(analysis_report),
            },
        },
    }
    _write_json(output_path, result)
    return result


def run_gromacs_production(
    config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    if bool(config.get("strict_checkpoint_resume", False)):
        return _run_gromacs_continuation(config, output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    topology = _copy_prepared_file(config, "source_topology_path", output_dir)
    coordinates = _copy_prepared_file(
        config,
        "source_coordinates_path",
        output_dir,
    )
    checkpoint = _copy_prepared_file(
        config,
        "source_checkpoint_path",
        output_dir,
    )
    index = _copy_prepared_file(config, "source_index_path", output_dir)
    source_topology_dir = Path(
        str(config.get("source_topology_path") or "")
    ).parent
    for source_restraint in source_topology_dir.glob("roe_posre_*.itp"):
        shutil.copy2(
            source_restraint,
            output_dir / source_restraint.name,
        )
    log_path = output_dir / "gromacs_production.log"
    temperature = float(config.get("temperature") or 300.0)
    pressure = float(config.get("pressure") or 1.0)
    timestep_fs = float(config.get("production_timestep_fs") or 2.0)
    mass_repartition_factor = config.get("mass_repartition_factor")
    if (
        mass_repartition_factor is None
        and config.get("hydrogen_mass_amu") is not None
        and timestep_fs > 2.0
    ):
        mass_repartition_factor = 3.0
    start_mode = str(
        config.get("continuation_mode") or "independent_replica"
    )
    current = coordinates
    current_checkpoint: Path | None = checkpoint
    burn_in_steps = int(config.get("replica_equilibration_steps") or 0)
    density_revalidation = bool(
        config.get("replica_density_revalidation", False)
    )
    replica_revalidation_report: Path | None = None
    replica_revalidation_series: Path | None = None
    if (
        start_mode != "exact_checkpoint"
        and (burn_in_steps > 0 or density_revalidation)
    ):
        max_steps = int(
            config.get("replica_revalidation_max_steps") or burn_in_steps
        )
        increment_steps = int(
            config.get("replica_revalidation_increment_steps") or burn_in_steps
        )
        sample_interval_steps = int(
            config.get("replica_density_sample_interval_steps") or 0
        )
        if density_revalidation and max_steps < burn_in_steps:
            raise ValueError(
                "Replica density revalidation maximum must be at least its minimum"
            )
        if density_revalidation and (
            increment_steps <= 0 or sample_interval_steps <= 0
        ):
            raise ValueError(
                "Replica density revalidation increment and sample interval "
                "must be positive"
            )
        completed_steps = 0
        density_energy_files: list[Path] = []
        density_times: list[float] = []
        densities: list[float] = []
        density_fit: dict[str, Any] = {
            "plateau": False,
            "reason": "Replica density revalidation was not requested",
        }
        segment = 0
        while completed_steps < max_steps:
            segment += 1
            stage_steps = min(
                increment_steps if density_revalidation else burn_in_steps,
                max_steps - completed_steps,
            )
            burn_stage = {
                "stage": "10" if density_revalidation else "replica_burn_in",
                "kind": (
                    "density_stabilization_npt"
                    if density_revalidation
                    else "npt"
                ),
                "steps": stage_steps,
                "timestep_fs": timestep_fs,
                "temperature_k": temperature,
                "pressure_bar": pressure,
                "generate_velocities": segment == 1,
                "mass_repartition_factor": mass_repartition_factor,
            }
            if density_revalidation:
                burn_stage["energy_interval_steps"] = sample_interval_steps
            stage_name = (
                f"replica_revalidation_{segment:03d}"
                if density_revalidation
                else "replica_burn_in"
            )
            burn_path = output_dir / f"{stage_name}.mdp"
            burn_path.write_text(
                render_gromacs_mdp(
                    burn_stage,
                    density_sample_interval_ps=(
                        sample_interval_steps * timestep_fs / 1000.0
                        if density_revalidation
                        else 4.0
                    ),
                    random_seed=int(config.get("replica_seed") or -1),
                )
            )
            burn_stage["mdp"] = burn_path.name
            current, current_checkpoint, energy = _run_stage(
                output_dir=output_dir,
                stage=burn_stage,
                start_coordinates=current,
                reference_coordinates=current,
                topology=topology,
                index=index,
                previous_checkpoint=(
                    current_checkpoint if segment > 1 else None
                ),
                use_gpu=bool(config.get("use_gpu", True)),
                log_path=log_path,
                name=stage_name,
            )
            completed_steps += stage_steps
            if not density_revalidation:
                break
            density_energy_files.append(energy)
            density_times, densities = _extract_density(
                density_energy_files,
                output_dir,
                log_path,
            )
            if completed_steps >= burn_in_steps:
                density_fit = fit_density_plateau(density_times, densities)
                if density_fit.get("plateau") is True:
                    break
        if density_revalidation:
            replica_revalidation_series = (
                output_dir / "replica_density_revalidation.csv"
            )
            with replica_revalidation_series.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("time_ps", "density_g_ml"))
                writer.writerows(zip(density_times, densities))
            report = {
                "protocol": "Roe-Brooks replica density revalidation",
                "seed": int(config.get("replica_seed") or -1),
                "minimum_steps": burn_in_steps,
                "maximum_steps": max_steps,
                "completed_steps": completed_steps,
                "duration_ns": completed_steps * timestep_fs / 1_000_000.0,
                "sample_interval_ps": (
                    sample_interval_steps * timestep_fs / 1000.0
                ),
                "plateau_required": bool(
                    config.get("replica_density_plateau_required", True)
                ),
                "fit": density_fit,
            }
            replica_revalidation_report = (
                output_dir / "replica_density_revalidation.json"
            )
            _write_json(replica_revalidation_report, report)
            if (
                report["plateau_required"]
                and density_fit.get("plateau") is not True
            ):
                raise RuntimeError(
                    "Replica Roe-Brooks density plateau criteria were not "
                    f"satisfied within {report['duration_ns']:.3f} ns"
                )

    production_stage = {
        "stage": "production",
        "kind": "npt",
        "steps": int(config.get("production_steps") or 0),
        "timestep_fs": timestep_fs,
        "temperature_k": temperature,
        "pressure_bar": pressure,
        "generate_velocities": False,
        "mass_repartition_factor": mass_repartition_factor,
    }
    production_mdp = output_dir / "production.mdp"
    production_mdp.write_text(render_gromacs_mdp(production_stage))
    production_stage["mdp"] = production_mdp.name
    final_gro, final_checkpoint, energy = _run_stage(
        output_dir=output_dir,
        stage=production_stage,
        start_coordinates=current,
        reference_coordinates=current,
        topology=topology,
        index=index,
        previous_checkpoint=current_checkpoint,
        use_gpu=bool(config.get("use_gpu", True)),
        log_path=log_path,
        name="production",
    )
    native_trajectory = output_dir / "production.xtc"
    trajectory = output_dir / "production_whole.xtc"
    tpr = output_dir / "production.tpr"
    final_pdb = output_dir / "production.pdb"
    if (
        not native_trajectory.is_file()
        or not tpr.is_file()
        or final_checkpoint is None
    ):
        raise RuntimeError("GROMACS production did not produce XTC/TPR/CPT artifacts")
    _run(
        [
            _gmx_binary(),
            "trjconv",
            "-s",
            str(tpr),
            "-f",
            str(native_trajectory),
            "-o",
            str(trajectory),
            "-n",
            str(index),
            "-pbc",
            "mol",
            "-center",
            "-ur",
            "compact",
        ],
        cwd=output_dir,
        stdin="Protein_Ligand\nSystem\n",
        log_path=log_path,
    )
    _run(
        [
            _gmx_binary(),
            "trjconv",
            "-s",
            str(tpr),
            "-f",
            str(trajectory),
            "-o",
            str(final_pdb),
            "-n",
            str(index),
            "-dump",
            "0",
        ],
        cwd=output_dir,
        stdin="System\n",
        log_path=log_path,
    )
    analytics = EquilibrationAnalytics().compute(
        output_dir=str(output_dir),
        system_id=str(config.get("system_id") or "gromacs"),
        topology_pdb=str(final_pdb),
        production_traj=str(trajectory),
        ligand_id="LIG",
        production_steps=int(production_stage["steps"]),
        production_report_interval=max(
            1,
            int(config.get("production_report_interval") or 5000),
        ),
        dt_ps=timestep_fs / 1000.0,
        residue_mapping=(
            config.get("residue_mapping")
            if isinstance(config.get("residue_mapping"), dict)
            else None
        ),
        ligand_formal_charges=ligand_formal_charges_from_sdf_data(
            str(config.get("ligand_refined_sdf_data") or "")
        ),
    )
    analytics["performance"] = parse_gromacs_performance(
        output_dir / "production.log"
    )
    analysis_report = output_dir / "trajectory_analysis.json"
    _write_json(analysis_report, analytics)
    result = {
        "success": True,
        "engine": GROMACS_ENGINE,
        "md_result": {
            "engine": GROMACS_ENGINE,
            "analytics": analytics,
            "output_files": {
                "production_trajectory": str(trajectory),
                "native_production_trajectory": str(native_trajectory),
                "production_pdb": str(final_pdb),
                "production_checkpoint": str(final_checkpoint),
                "production_topology": str(topology),
                "production_tpr": str(tpr),
                "production_index": str(index),
                "thermodynamic_series": str(energy),
                "analysis_report": str(analysis_report),
                **(
                    {
                        "replica_density_series": str(
                            replica_revalidation_series
                        ),
                        "replica_density_report": str(
                            replica_revalidation_report
                        ),
                    }
                    if (
                        replica_revalidation_series is not None
                        and replica_revalidation_report is not None
                    )
                    else {}
                ),
            },
        },
    }
    _write_json(output_path, result)
    return result


def _resolve_source_output(source_result: dict[str, Any], key: str) -> Path:
    output_files = (
        (source_result.get("md_result") or {}).get("output_files") or {}
    )
    path = Path(str(output_files.get(key) or ""))
    if not path.is_file():
        raise FileNotFoundError(
            f"GROMACS endpoint input is missing: {key}={path}"
        )
    return path


def _trajectory_window_ns(
    trajectory: Path,
    topology_pdb: Path,
    *,
    start_pct: float,
    end_pct: float,
    stride: int,
) -> tuple[float, float, float, int]:
    try:
        import mdtraj as md
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "MDTraj and NumPy are required to map endpoint percentages to time"
        ) from exc
    loaded = md.load(str(trajectory), top=str(topology_pdb))
    if loaded.n_frames < 1:
        raise ValueError("The endpoint trajectory contains no frames")
    times_ps = np.asarray(loaded.time, dtype=float)
    if len(times_ps) != loaded.n_frames or not np.all(np.isfinite(times_ps)):
        raise ValueError("The endpoint trajectory has invalid frame times")
    start_index = min(
        loaded.n_frames - 1,
        int((float(start_pct) / 100.0) * loaded.n_frames),
    )
    end_index = min(
        loaded.n_frames - 1,
        max(start_index, int((float(end_pct) / 100.0) * loaded.n_frames) - 1),
    )
    frame_spacing_ps = (
        float(np.median(np.diff(times_ps)))
        if loaded.n_frames > 1
        else 1.0
    )
    return (
        float(times_ps[start_index]) / 1000.0,
        float(times_ps[end_index]) / 1000.0,
        max(frame_spacing_ps * max(1, int(stride)) / 1000.0, 1.0e-9),
        max(1, ((end_index - start_index) // max(1, int(stride))) + 1),
    )


def run_g_mmpbsa(
    config: dict[str, Any],
    source_result: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    from statistics import mean

    output_dir = output_path.parent
    trajectory = _resolve_source_output(
        source_result,
        "production_trajectory",
    )
    topology_pdb = _resolve_source_output(
        source_result,
        "production_pdb",
    )
    production_tpr = _resolve_source_output(source_result, "production_tpr")
    topology = _resolve_source_output(source_result, "production_topology")
    index = _resolve_source_output(source_result, "production_index")
    executable = shutil.which("g_mmpbsa")
    if executable is None:
        raise RuntimeError("g_mmpbsa is not installed in the GROMACS image")
    binding_xvg = output_dir / "g_mmpbsa_binding_energy.xvg"
    summary_csv = output_dir / "g_mmpbsa_summary.csv"
    residue_csv = output_dir / "g_mmpbsa_residue_decomposition.csv"
    parameter_file = output_dir / "g_mmpbsa.mdp"
    parameter_file.write_text(
        "polar = yes\n"
        "apolar = yes\n"
        "cfac = 1.5\n"
        "gridspace = 0.5\n"
        "gmemceil = 4000\n"
        "fadd = 5\n"
        "pcharge = 1\n"
        "prad = 0.95\n"
        "pconc = 0.150\n"
        "ncharge = -1\n"
        "nrad = 1.81\n"
        "nconc = 0.150\n"
        "pdie = 2\n"
        "sdie = 80\n"
        "vdie = 1\n"
        "srad = 1.4\n"
        "swin = 0.3\n"
        "srfm = smol\n"
        "chgm = spl4\n"
        "sdens = 10\n"
        "temp = 300\n"
        "bcfl = mdh\n"
        "PBsolver = lpbe\n"
        "gamma = 0.02267\n"
        "sasaconst = 3.84928\n"
        "sasrad = 1.4\n"
    )
    compatibility_mdp = output_dir / "g_mmpbsa_compatibility.mdp"
    compatibility_mdp.write_text(
        "integrator = md\n"
        "nsteps = 0\n"
        "dt = 0.002\n"
        "continuation = yes\n"
        "constraints = h-bonds\n"
        "cutoff-scheme = Verlet\n"
        "coulombtype = PME\n"
        "rlist = 1.0\n"
        "rcoulomb = 1.0\n"
        "rvdw = 1.0\n"
        "tcoupl = no\n"
        "pcoupl = no\n"
        "gen-vel = no\n"
    )
    compatibility_tpr = output_dir / "g_mmpbsa_compatibility.tpr"
    compatibility_log = output_dir / "g_mmpbsa_compatibility.log"
    _run(
        [
            _g_mmpbsa_gmx_binary(),
            "grompp",
            "-f",
            str(compatibility_mdp),
            "-c",
            str(topology_pdb),
            "-p",
            str(topology),
            "-n",
            str(index),
            "-o",
            str(compatibility_tpr),
            "-maxwarn",
            "1",
        ],
        cwd=output_dir,
        log_path=compatibility_log,
    )
    start_pct = float(config.get("mmgbsa_start_pct") or 20.0)
    end_pct = float(config.get("mmgbsa_end_pct") or 100.0)
    stride = max(1, int(config.get("mmgbsa_stride") or 1))
    start_ns, end_ns, sampling_interval_ns, analyzed_frames = (
        _trajectory_window_ns(
            trajectory,
            topology_pdb,
            start_pct=start_pct,
            end_pct=end_pct,
            stride=stride,
        )
    )
    command = [
        executable,
        "run",
        "-f",
        str(trajectory),
        "-s",
        str(compatibility_tpr),
        "-n",
        str(index),
        "-i",
        str(parameter_file),
        "-unit1",
        "Protein",
        "-unit2",
        "Ligand",
        "-b",
        str(start_ns),
        "-e",
        str(end_ns),
        "-dt",
        str(sampling_interval_ns),
        "-tu",
        "ns",
        "-o",
        str(binding_xvg),
        "-os",
        str(summary_csv),
        "-ores",
        str(residue_csv),
        "-decomp",
        "-pbsa",
    ]
    log_path = output_dir / "g_mmpbsa.log"
    _run(command, cwd=output_dir, log_path=log_path)
    missing_outputs = [
        path
        for path in (binding_xvg, summary_csv, residue_csv)
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing_outputs:
        log_tail = log_path.read_text(errors="replace")[-4000:]
        raise RuntimeError(
            "g_mmpbsa did not produce the required MM/PBSA artifacts: "
            + ", ".join(path.name for path in missing_outputs)
            + "\n"
            + log_tail
        )
    _, totals = parse_xvg_series(binding_xvg)
    if not totals:
        raise RuntimeError(
            "g_mmpbsa produced no binding-energy samples; inspect "
            f"{log_path}"
        )
    total_mean = mean(totals) if totals else None
    mmgbsa = {
        "status": "success",
        "backend": "g_mmpbsa",
        "method": "g_mmpbsa_mm_pbsa",
        "start_pct": start_pct,
        "end_pct": end_pct,
        "stride": stride,
        "trajectory_path": str(trajectory),
        "topology_path": str(compatibility_tpr),
        "delta": {
            "delta_g_bind_total_kj_mol": total_mean,
            "delta_g_bind_total_kcal_mol": (
                total_mean / 4.184 if total_mean is not None else None
            ),
        },
        "metadata": {
            "n_frames_analyzed": len(totals) or analyzed_frames,
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "sampling_interval_ns": sampling_interval_ns,
            "pbc_corrected_trajectory": True,
            "production_tpr": str(production_tpr),
            "endpoint_tpr_gromacs_version": "2025.4",
        },
        "artifacts": {
            "per_frame_xvg": str(binding_xvg),
            "summary_csv": str(summary_csv),
            "residue_decomposition_csv": str(residue_csv),
            "parameter_file": str(parameter_file),
            "compatibility_mdp": str(compatibility_mdp),
            "compatibility_tpr": str(compatibility_tpr),
            "compatibility_log": str(compatibility_log),
            "log": str(log_path),
        },
    }
    result = {"success": True, "mmgbsa": mmgbsa}
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="mn-ligand GROMACS MD adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "production"):
        child = subparsers.add_parser(command)
        child.add_argument("--input", required=True)
        child.add_argument("--output", required=True)
    endpoint = subparsers.add_parser("endpoint")
    endpoint.add_argument("--input", required=True)
    endpoint.add_argument("--result", required=True)
    endpoint.add_argument("--output", required=True)
    endpoint.add_argument("--start-pct", type=float, default=20.0)
    endpoint.add_argument("--end-pct", type=float, default=100.0)
    endpoint.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    output_path = Path(args.output)
    try:
        config = _read_json(Path(args.input))
        if args.command == "prepare":
            prepare_gromacs_system(config, output_path)
        elif args.command == "production":
            run_gromacs_production(config, output_path)
        else:
            config.update(
                {
                    "mmgbsa_start_pct": float(args.start_pct),
                    "mmgbsa_end_pct": float(args.end_pct),
                    "mmgbsa_stride": int(args.stride),
                }
            )
            run_g_mmpbsa(
                config,
                _read_json(Path(args.result)),
                output_path,
            )
    except Exception as exc:
        _write_json(
            output_path,
            {
                "success": False,
                "engine": GROMACS_ENGINE,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
