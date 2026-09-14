from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable

from mn_ligand.workflows.md_engines import roe_brooks_stage_specification


KCAL_A2_TO_KJ_NM2 = 418.4
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYM", "CYX", "GLN", "GLU",
    "GLH", "GLY", "HID", "HIE", "HIP", "HIS", "ILE", "LEU", "LYS", "LYN",
    "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
WATER_RESIDUES = {"HOH", "WAT", "SOL", "TIP3", "TIP3P", "TIP4P"}
ION_RESIDUES = {
    "NA", "NA+", "CL", "CL-", "K", "K+", "MG", "MG2", "CA", "CA2", "ZN",
    "ZN2",
}
BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}


def _mdp_lines(values: dict[str, Any]) -> str:
    return "\n".join(
        f"{key:<28} = {value}"
        for key, value in values.items()
        if value is not None
    ) + "\n"


def render_gromacs_mdp(
    stage: dict[str, Any],
    *,
    density_sample_interval_ps: float = 4.0,
    random_seed: int = -1,
) -> str:
    kind = str(stage["kind"])
    timestep_fs = float(stage.get("timestep_fs") or 1.0)
    mass_repartition_factor = float(
        stage.get("mass_repartition_factor") or 1.0
    )
    dt_ps = timestep_fs / 1000.0
    stage_id = str(stage.get("stage") or "")
    values: dict[str, Any] = {
        "integrator": "steep" if kind == "minimization" else "md",
        "nsteps": int(stage.get("steps") or 0),
        "dt": f"{dt_ps:.6f}" if kind != "minimization" else None,
        "cutoff-scheme": "Verlet",
        "nstlist": 20,
        "rlist": 1.0,
        "coulombtype": "PME",
        "rcoulomb": 1.0,
        "vdwtype": "Cut-off",
        "rvdw": 1.0,
        "DispCorr": "EnerPres",
        "constraints": (
            "none"
            if kind == "minimization"
            else "h-bonds"
            if timestep_fs <= 2.0 or mass_repartition_factor > 1.0
            else "all-bonds"
        ),
        "mass-repartition-factor": (
            mass_repartition_factor
            if kind != "minimization" and mass_repartition_factor > 1.0
            else None
        ),
        "constraint-algorithm": "lincs",
        "continuation": (
            "no"
            if kind == "minimization" or stage.get("generate_velocities")
            else "yes"
        ),
        "nstxout": 0,
        "nstvout": 0,
        "nstfout": 0,
        "nstxout-compressed": (
            max(1, int(round(density_sample_interval_ps / dt_ps)))
            if kind == "density_stabilization_npt"
            else max(1, int(round(10.0 / dt_ps)))
            if kind != "minimization"
            else 0
        ),
        "nstenergy": (
            max(1, int(round(density_sample_interval_ps / dt_ps)))
            if kind == "density_stabilization_npt"
            else max(1, int(round(2.0 / dt_ps)))
            if kind != "minimization"
            else 10
        ),
        "nstlog": (
            max(1, int(round(density_sample_interval_ps / dt_ps)))
            if kind == "density_stabilization_npt"
            else 1000
            if kind != "minimization"
            else 10
        ),
        "compressed-x-grps": "System",
    }
    if kind == "minimization":
        values.update({"emtol": 10.0, "emstep": 0.01})
    else:
        production_like = stage_id in {"9", "production"} or (
            stage_id == "10" and not stage.get("generate_velocities")
        )
        nvt = kind == "nvt"
        values.update(
            {
                "tcoupl": "V-rescale",
                "tc-grps": "System",
                "tau-t": 0.5 if nvt else 1.0,
                "ref-t": float(stage.get("temperature_k") or 300.0),
                "pcoupl": (
                    "no"
                    if nvt
                    else "Parrinello-Rahman"
                    if production_like
                    else "C-rescale"
                ),
                "pcoupltype": None if nvt else "isotropic",
                "tau-p": None if nvt else 5.0 if production_like else 1.0,
                "ref-p": None
                if nvt
                else float(stage.get("pressure_bar") or 1.0),
                "compressibility": None if nvt else "4.5e-5",
                "gen-vel": "yes" if stage.get("generate_velocities") else "no",
                "gen-temp": (
                    float(stage.get("temperature_k") or 300.0)
                    if stage.get("generate_velocities")
                    else None
                ),
                "gen-seed": (
                    int(random_seed)
                    if stage.get("generate_velocities")
                    else None
                ),
            }
        )
    if float(stage.get("restraint_k_kcal_mol_a2") or 0.0) > 0:
        values["define"] = f"-DROE_STAGE_{stage_id}"
        if kind not in {"minimization", "nvt"}:
            values["refcoord-scaling"] = "com"
    return _mdp_lines(values)


def write_roe_brooks_mdp_set(
    output_dir: Path,
    *,
    temperature_k: float,
    pressure_bar: float,
    density_sample_interval_ps: float,
    production_timestep_fs: float = 2.0,
    mass_repartition_factor: float | None = None,
    random_seed: int = -1,
) -> list[dict[str, Any]]:
    stages = [
        dict(stage)
        for stage in roe_brooks_stage_specification(
            temperature_k=temperature_k,
            pressure_bar=pressure_bar,
        )
    ]
    for stage in stages:
        if stage["kind"] != "minimization" and mass_repartition_factor is not None:
            stage["mass_repartition_factor"] = float(
                mass_repartition_factor
            )
        if stage["stage"] == "10":
            stage["timestep_fs"] = float(production_timestep_fs)
            stage["steps"] = max(
                1,
                int(round(1_000_000.0 / float(production_timestep_fs))),
            )
        mdp_path = output_dir / f"roe_{stage['stage']}.mdp"
        mdp_path.write_text(
            render_gromacs_mdp(
                stage,
                density_sample_interval_ps=density_sample_interval_ps,
                random_seed=random_seed,
            )
        )
        stage["mdp"] = mdp_path.name
    return stages


def _atom_is_heavy(atom: Any) -> bool:
    atomic_number = int(getattr(atom, "atomic_number", 0) or 0)
    if atomic_number:
        return atomic_number > 1
    return not str(getattr(atom, "name", "")).upper().startswith("H")


def atom_groups(structure: Any, ligand_resname: str = "LIG") -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "System": [],
        "Protein": [],
        "Ligand": [],
        "Protein_Ligand": [],
        "Backbone": [],
        "Large_Molecule_Heavy": [],
        "Roe_Backbone_Ligand_Heavy": [],
        "Solvent": [],
        "Ions": [],
    }
    ligand_resname = ligand_resname.upper()
    for index, atom in enumerate(structure.atoms, start=1):
        resname = str(atom.residue.name).strip().upper()
        atom_name = str(atom.name).strip().upper()
        heavy = _atom_is_heavy(atom)
        groups["System"].append(index)
        if resname in PROTEIN_RESIDUES:
            groups["Protein"].append(index)
            groups["Protein_Ligand"].append(index)
            if atom_name in BACKBONE_NAMES:
                groups["Backbone"].append(index)
            if heavy:
                groups["Large_Molecule_Heavy"].append(index)
            if heavy and atom_name in BACKBONE_NAMES:
                groups["Roe_Backbone_Ligand_Heavy"].append(index)
        elif resname == ligand_resname:
            groups["Ligand"].append(index)
            groups["Protein_Ligand"].append(index)
            if heavy:
                groups["Large_Molecule_Heavy"].append(index)
                groups["Roe_Backbone_Ligand_Heavy"].append(index)
        elif resname in WATER_RESIDUES:
            groups["Solvent"].append(index)
        elif resname in ION_RESIDUES:
            groups["Ions"].append(index)
        elif heavy:
            groups["Large_Molecule_Heavy"].append(index)
    return groups


def write_gromacs_index(path: Path, groups: dict[str, Iterable[int]]) -> None:
    lines: list[str] = []
    for name, raw_indices in groups.items():
        indices = [int(value) for value in raw_indices]
        if not indices:
            continue
        lines.append(f"[ {name} ]")
        lines.extend(
            " ".join(str(value) for value in indices[offset: offset + 15])
            for offset in range(0, len(indices), 15)
        )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_roe_position_restraints(
    path: Path,
    groups: dict[str, list[int]],
) -> None:
    stage_groups = {
        "1": groups.get("Large_Molecule_Heavy", []),
        "2": groups.get("Large_Molecule_Heavy", []),
        "3": groups.get("Large_Molecule_Heavy", []),
        "4": groups.get("Large_Molecule_Heavy", []),
        "6": groups.get("Large_Molecule_Heavy", []),
        "7": groups.get("Large_Molecule_Heavy", []),
        "8": groups.get("Roe_Backbone_Ligand_Heavy", []),
    }
    force_constants = {
        "1": 5.0,
        "2": 5.0,
        "3": 2.0,
        "4": 0.1,
        "6": 1.0,
        "7": 0.5,
        "8": 0.5,
    }
    lines = [
        "; Generated Roe-Brooks positional restraints.",
        "; kcal mol^-1 A^-2 converted to kJ mol^-1 nm^-2.",
    ]
    for stage, indices in stage_groups.items():
        force = force_constants[stage] * KCAL_A2_TO_KJ_NM2
        lines.extend(
            [
                f"#ifdef ROE_STAGE_{stage}",
                "[ position_restraints ]",
                "; atom  type      fx          fy          fz",
            ]
        )
        lines.extend(
            f"{index:8d}     1 {force:11.4f} {force:11.4f} {force:11.4f}"
            for index in indices
        )
        lines.extend(["#endif", ""])
    path.write_text("\n".join(lines).rstrip() + "\n")


def inject_restraint_include(topology_path: Path, include_name: str) -> None:
    text = topology_path.read_text()
    marker = f'#include "{include_name}"'
    if marker in text:
        return
    lines = text.splitlines()
    molecule_sections = [
        index
        for index, line in enumerate(lines)
        if line.strip().lower() == "[ moleculetype ]"
    ]
    system_section = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().lower() == "[ system ]"
        ),
        len(lines),
    )
    insert_at = (
        molecule_sections[1]
        if len(molecule_sections) > 1
        else system_section
    )
    lines[insert_at:insert_at] = [
        "",
        "; mn-ligand Roe-Brooks restraints for the first solute molecule type",
        marker,
        "",
    ]
    topology_path.write_text("\n".join(lines) + "\n")


def write_molecule_type_restraints(topology_path: Path) -> list[Path]:
    lines = topology_path.read_text().splitlines()
    molecule_starts = [
        index
        for index, line in enumerate(lines)
        if line.strip().lower() == "[ moleculetype ]"
    ]
    generated: list[Path] = []
    insertions: list[tuple[int, list[str]]] = []
    for molecule_index, start in enumerate(molecule_starts):
        end = (
            molecule_starts[molecule_index + 1]
            if molecule_index + 1 < len(molecule_starts)
            else next(
                (
                    index
                    for index in range(start + 1, len(lines))
                    if lines[index].strip().lower() == "[ system ]"
                ),
                len(lines),
            )
        )
        molecule_name = ""
        for line in lines[start + 1:end]:
            stripped = line.strip()
            if stripped and not stripped.startswith((";", "#", "[")):
                molecule_name = stripped.split()[0]
                break
        atoms_start = next(
            (
                index
                for index in range(start + 1, end)
                if lines[index].strip().lower() == "[ atoms ]"
            ),
            None,
        )
        if atoms_start is None:
            continue
        atoms_end = next(
            (
                index
                for index in range(atoms_start + 1, end)
                if lines[index].strip().startswith("[")
            ),
            end,
        )
        large_heavy: list[int] = []
        backbone_ligand_heavy: list[int] = []
        for line in lines[atoms_start + 1:atoms_end]:
            stripped = line.strip()
            if not stripped or stripped.startswith((";", "#")):
                continue
            fields = stripped.split()
            if len(fields) < 5:
                continue
            try:
                atom_index = int(fields[0])
            except ValueError:
                continue
            resname = fields[3].upper()
            atom_name = fields[4].upper()
            heavy = not atom_name.startswith("H")
            if not heavy or resname in WATER_RESIDUES or resname in ION_RESIDUES:
                continue
            large_heavy.append(atom_index)
            if resname not in PROTEIN_RESIDUES or atom_name in BACKBONE_NAMES:
                backbone_ligand_heavy.append(atom_index)
        if not large_heavy:
            continue
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", molecule_name or str(molecule_index + 1))
        restraint_path = topology_path.with_name(
            f"roe_posre_{safe_name}.itp"
        )
        write_roe_position_restraints(
            restraint_path,
            {
                "Large_Molecule_Heavy": large_heavy,
                "Roe_Backbone_Ligand_Heavy": backbone_ligand_heavy,
            },
        )
        generated.append(restraint_path)
        insertions.append(
            (
                end,
                [
                    "",
                    f"; mn-ligand Roe-Brooks restraints for {molecule_name}",
                    f'#include "{restraint_path.name}"',
                    "",
                ],
            )
        )
    for insert_at, insertion in sorted(insertions, reverse=True):
        lines[insert_at:insert_at] = insertion
    topology_path.write_text("\n".join(lines) + "\n")
    return generated


def parse_xvg_series(path: Path) -> tuple[list[float], list[float]]:
    times: list[float] = []
    values: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "@")):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            continue
        try:
            times.append(float(fields[0]))
            values.append(float(fields[-1]))
        except ValueError:
            continue
    return times, values
