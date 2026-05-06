from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mdtraj as md
import openmm
from openmm import unit
from openmm.app import ForceField as OpenMMForceField, NoCutoff, HBonds
from openmmforcefields.generators import SMIRNOFFTemplateGenerator


def summarize_terms(df, columns):
    rows = []
    for col in columns:
        values = df[col].dropna()
        rows.append({
            "term": col,
            "mean": float(values.mean()),
            "std": float(values.std()),
            "sem": float(values.std() / np.sqrt(len(values))),
            "min": float(values.min()),
            "max": float(values.max()),
            "n_frames": int(len(values)),
        })
    return pd.DataFrame(rows)


def select_frames(traj, start=0, stop=None, stride=1):
    stop = traj.n_frames if stop is None else stop
    return traj[start:stop:stride]


def clone_system(system):
    xml = openmm.XmlSerializer.serialize(system)
    return openmm.XmlSerializer.deserialize(xml)


def find_nonbonded_force(system):
    for i, force in enumerate(system.getForces()):
        if isinstance(force, openmm.NonbondedForce):
            return i, force
    raise RuntimeError("No NonbondedForce found")


def make_nonbonded_component_system(system, component):
    """
    component='vdw'  -> LJ only, charges zeroed
    component='elec' -> Coulomb only, epsilons zeroed
    """
    new_system = clone_system(system)
    nb_index, nb = find_nonbonded_force(new_system)

    if component not in {"vdw", "elec"}:
        raise ValueError("component must be 'vdw' or 'elec'")

    for i in range(nb.getNumParticles()):
        q, sigma, epsilon = nb.getParticleParameters(i)
        if component == "vdw":
            nb.setParticleParameters(i, 0.0 * q.unit, sigma, epsilon)
        else:
            nb.setParticleParameters(i, q, sigma, 0.0 * epsilon.unit)

    for i in range(nb.getNumExceptions()):
        p1, p2, charge_prod, sigma, epsilon = nb.getExceptionParameters(i)
        if component == "vdw":
            nb.setExceptionParameters(i, p1, p2, 0.0 * charge_prod.unit, sigma, epsilon)
        else:
            nb.setExceptionParameters(i, p1, p2, charge_prod, sigma, 0.0 * epsilon.unit)

    for i, force in enumerate(new_system.getForces()):
        force.setForceGroup(i)

    return new_system, nb_index


def build_gbsa_forcefield(ligand=None, ligand_ff="openff-2.2.0"):
    ff = OpenMMForceField("amber14-all.xml", "implicit/obc2.xml")

    if ligand is not None:
        smirnoff = SMIRNOFFTemplateGenerator(forcefield=ligand_ff)
        smirnoff.add_molecules(ligand)
        ff.registerTemplateGenerator(smirnoff.generator)

    return ff


def create_gbsa_system(topology, ligand=None, ligand_ff="openff-2.2.0", label="system"):
    ff = build_gbsa_forcefield(ligand=ligand, ligand_ff=ligand_ff)

    system = ff.createSystem(
        topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
        rigidWater=False,
        removeCMMotion=False,
    )

    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)

    gbsa_force_indices = []
    for i, force in enumerate(system.getForces()):
        name = force.__class__.__name__
        if ("GBSA" in name) or ("GB" in name):
            gbsa_force_indices.append(i)

    if not gbsa_force_indices:
        raise RuntimeError(f"No GBSA-like force found in {label}")

    return system, gbsa_force_indices


def make_context(system, platform_name="CUDA"):
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
    except Exception:
        platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)
    return context, integrator, platform


def group_mask_from_indices(indices):
    mask = 0
    for i in indices:
        mask |= 1 << i
    return mask


def get_group_energy_kjmol(context, positions_nm, group_indices):
    context.setPositions(positions_nm * unit.nanometer)
    state = context.getState(getEnergy=True, groups=group_mask_from_indices(group_indices))
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def get_single_group_energy_kjmol(context, positions_nm, group_index):
    context.setPositions(positions_nm * unit.nanometer)
    state = context.getState(getEnergy=True, groups={group_index})
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def total_sasa_A2(traj):
    sasa_nm2 = md.shrake_rupley(traj, mode="atom")
    return sasa_nm2.sum(axis=1) * 100.0


def nonpolar_from_delta_sasa(delta_sasa_A2, gamma=0.00542, beta=0.92, beta_mode="binding"):
    """
    beta_mode:
      binding: gamma * ΔSASA - beta
      none:    gamma * ΔSASA
      plus:    gamma * ΔSASA + beta
    """
    if beta_mode == "binding":
        return gamma * delta_sasa_A2 - beta
    if beta_mode == "none":
        return gamma * delta_sasa_A2
    if beta_mode == "plus":
        return gamma * delta_sasa_A2 + beta
    raise ValueError(f"Unknown beta_mode: {beta_mode}")


def run_single_trajectory_mmgbsa_from_files(
    production_dcd,
    topology_pdb,
    ligand_sdf,
    ligand_resname="T3",
    ligand_ff="openff-2.2.0",
    output_prefix="mmgbsa",
    start_frame=0,
    stop_frame=None,
    stride=1,
    timestep_fs=4.0,
    production_report_interval=1000,
    sasa_gamma_kcal_per_A2=0.00542,
    sasa_beta_kcal=0.92,
    sasa_beta_mode="binding",
    platform_name="CUDA",
):
    """
    Same as run_single_trajectory_mmgbsa(), but does not require an active
    OpenMM Simulation object.

    Inputs:
      production_dcd: trajectory file
      topology_pdb: PDB with same atom order as DCD
      ligand_sdf: refined ligand SDF used for OpenFF template generation
    """
    from openff.toolkit import Molecule

    production_dcd = Path(production_dcd)
    topology_pdb = Path(topology_pdb)
    ligand_sdf = Path(ligand_sdf)

    if not production_dcd.exists():
        raise FileNotFoundError(production_dcd)
    if not topology_pdb.exists():
        raise FileNotFoundError(topology_pdb)
    if not ligand_sdf.exists():
        raise FileNotFoundError(ligand_sdf)

    ligand = Molecule.from_file(
        str(ligand_sdf),
        file_format="sdf",
        allow_undefined_stereo=True,
    )
    ligand.name = ligand_resname

    print("Ligand loaded from SDF")
    print(" atoms:", ligand.n_atoms)
    print(" bonds:", ligand.n_bonds)
    print(" total charge:", ligand.total_charge)
    print(" aromatic atoms:", sum(a.is_aromatic for a in ligand.atoms))
    print(" aromatic bonds:", sum(b.is_aromatic for b in ligand.bonds))

    metadata = {
        "method": "single-trajectory OpenMM/OpenFF MMGBSA-like",
        "production_dcd": str(production_dcd),
        "topology_pdb": str(topology_pdb),
        "ligand_sdf": str(ligand_sdf),
        "ligand_resname": ligand_resname,
        "ligand_forcefield": ligand_ff,
        "protein_forcefield": "amber14-all.xml",
        "gb_model": "implicit/obc2.xml",
        "sasa_gamma_kcal_per_A2": sasa_gamma_kcal_per_A2,
        "sasa_beta_kcal": sasa_beta_kcal,
        "sasa_beta_mode": sasa_beta_mode,
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "stride": stride,
        "timestep_fs": timestep_fs,
        "production_report_interval": production_report_interval,
        "frame_dt_ps_original": timestep_fs * production_report_interval / 1000.0,
        "openmm_version": openmm.version.version,
        "mdtraj_version": md.__version__,
    }

    print("Loading production trajectory from files...")
    traj_full_all = md.load(str(production_dcd), top=str(topology_pdb))
    traj_full = select_frames(traj_full_all, start=start_frame, stop=stop_frame, stride=stride)

    frame_dt_ps = timestep_fs * production_report_interval * stride / 1000.0

    print("Trajectory:")
    print(" all frames:", traj_full_all.n_frames)
    print(" selected frames:", traj_full.n_frames)
    print(" atoms:", traj_full.n_atoms)

    complex_indices = traj_full.topology.select(f"protein or resname {ligand_resname}")
    protein_indices = traj_full.topology.select("protein")
    ligand_indices = traj_full.topology.select(f"resname {ligand_resname}")

    if len(complex_indices) == 0:
        raise ValueError("No complex atoms selected")
    if len(protein_indices) == 0:
        raise ValueError("No protein atoms selected")
    if len(ligand_indices) == 0:
        raise ValueError(f"No ligand atoms selected for resname {ligand_resname}")

    print("complex atoms:", len(complex_indices))
    print("protein atoms:", len(protein_indices))
    print("ligand atoms:", len(ligand_indices))

    traj_complex = traj_full.atom_slice(complex_indices)
    traj_protein = traj_full.atom_slice(protein_indices)
    traj_ligand = traj_full.atom_slice(ligand_indices)

    top_complex = traj_complex.topology.to_openmm()
    top_protein = traj_protein.topology.to_openmm()
    top_ligand = traj_ligand.topology.to_openmm()

    traj_complex[0].save_pdb(f"{output_prefix}_complex_frame0.pdb")
    traj_protein[0].save_pdb(f"{output_prefix}_protein_frame0.pdb")
    traj_ligand[0].save_pdb(f"{output_prefix}_ligand_frame0.pdb")

    print("Building GBSA systems...")
    system_complex, gbsa_groups_complex = create_gbsa_system(
        top_complex, ligand=ligand, ligand_ff=ligand_ff, label="complex"
    )
    system_protein, gbsa_groups_protein = create_gbsa_system(
        top_protein, ligand=None, ligand_ff=ligand_ff, label="protein"
    )
    system_ligand, gbsa_groups_ligand = create_gbsa_system(
        top_ligand, ligand=ligand, ligand_ff=ligand_ff, label="ligand"
    )

    print("Building vdW/electrostatic component systems...")
    system_complex_vdw, group_complex_vdw = make_nonbonded_component_system(system_complex, "vdw")
    system_protein_vdw, group_protein_vdw = make_nonbonded_component_system(system_protein, "vdw")
    system_ligand_vdw, group_ligand_vdw = make_nonbonded_component_system(system_ligand, "vdw")

    system_complex_elec, group_complex_elec = make_nonbonded_component_system(system_complex, "elec")
    system_protein_elec, group_protein_elec = make_nonbonded_component_system(system_protein, "elec")
    system_ligand_elec, group_ligand_elec = make_nonbonded_component_system(system_ligand, "elec")

    print("Creating contexts...")
    ctx_complex_gb, int_complex_gb, platform = make_context(system_complex, platform_name)
    ctx_protein_gb, int_protein_gb, _ = make_context(system_protein, platform_name)
    ctx_ligand_gb, int_ligand_gb, _ = make_context(system_ligand, platform_name)

    ctx_complex_vdw, int_complex_vdw, _ = make_context(system_complex_vdw, platform_name)
    ctx_protein_vdw, int_protein_vdw, _ = make_context(system_protein_vdw, platform_name)
    ctx_ligand_vdw, int_ligand_vdw, _ = make_context(system_ligand_vdw, platform_name)

    ctx_complex_elec, int_complex_elec, _ = make_context(system_complex_elec, platform_name)
    ctx_protein_elec, int_protein_elec, _ = make_context(system_protein_elec, platform_name)
    ctx_ligand_elec, int_ligand_elec, _ = make_context(system_ligand_elec, platform_name)

    print("Using platform:", platform.getName())

    print("Computing SASA...")
    sasa_complex_A2 = total_sasa_A2(traj_complex)
    sasa_protein_A2 = total_sasa_A2(traj_protein)
    sasa_ligand_A2 = total_sasa_A2(traj_ligand)
    delta_sasa_A2 = sasa_complex_A2 - sasa_protein_A2 - sasa_ligand_A2
    delta_nonpolar_kcal = nonpolar_from_delta_sasa(
        delta_sasa_A2,
        gamma=sasa_gamma_kcal_per_A2,
        beta=sasa_beta_kcal,
        beta_mode=sasa_beta_mode,
    )

    print("Evaluating energies...")
    rows = []

    for frame_idx in range(traj_full.n_frames):
        g_complex_gb = get_group_energy_kjmol(ctx_complex_gb, traj_complex.xyz[frame_idx], gbsa_groups_complex)
        g_protein_gb = get_group_energy_kjmol(ctx_protein_gb, traj_protein.xyz[frame_idx], gbsa_groups_protein)
        g_ligand_gb = get_group_energy_kjmol(ctx_ligand_gb, traj_ligand.xyz[frame_idx], gbsa_groups_ligand)
        delta_gb = g_complex_gb - g_protein_gb - g_ligand_gb

        e_complex_vdw = get_single_group_energy_kjmol(ctx_complex_vdw, traj_complex.xyz[frame_idx], group_complex_vdw)
        e_protein_vdw = get_single_group_energy_kjmol(ctx_protein_vdw, traj_protein.xyz[frame_idx], group_protein_vdw)
        e_ligand_vdw = get_single_group_energy_kjmol(ctx_ligand_vdw, traj_ligand.xyz[frame_idx], group_ligand_vdw)
        delta_vdw = e_complex_vdw - e_protein_vdw - e_ligand_vdw

        e_complex_elec = get_single_group_energy_kjmol(ctx_complex_elec, traj_complex.xyz[frame_idx], group_complex_elec)
        e_protein_elec = get_single_group_energy_kjmol(ctx_protein_elec, traj_protein.xyz[frame_idx], group_protein_elec)
        e_ligand_elec = get_single_group_energy_kjmol(ctx_ligand_elec, traj_ligand.xyz[frame_idx], group_ligand_elec)
        delta_elec = e_complex_elec - e_protein_elec - e_ligand_elec

        delta_np_kcal = float(delta_nonpolar_kcal[frame_idx])
        delta_np_kj = delta_np_kcal * 4.184

        delta_total_kj = delta_vdw + delta_elec + delta_gb + delta_np_kj
        delta_total_kcal = delta_total_kj / 4.184

        rows.append({
            "frame": frame_idx,
            "source_frame": start_frame + frame_idx * stride,
            "time_ps": frame_idx * frame_dt_ps,

            "delta_E_vdw_kjmol": delta_vdw,
            "delta_E_elec_kjmol": delta_elec,
            "delta_G_gbsa_kjmol": delta_gb,
            "delta_G_nonpolar_kjmol": delta_np_kj,
            "delta_G_mmgbsa_kjmol": delta_total_kj,

            "delta_E_vdw_kcalmol": delta_vdw / 4.184,
            "delta_E_elec_kcalmol": delta_elec / 4.184,
            "delta_G_gbsa_kcalmol": delta_gb / 4.184,
            "delta_G_nonpolar_kcalmol": delta_np_kcal,
            "delta_G_mmgbsa_kcalmol": delta_total_kcal,

            "SASA_complex_A2": float(sasa_complex_A2[frame_idx]),
            "SASA_protein_A2": float(sasa_protein_A2[frame_idx]),
            "SASA_ligand_A2": float(sasa_ligand_A2[frame_idx]),
            "delta_SASA_A2": float(delta_sasa_A2[frame_idx]),
        })

        if frame_idx % 10 == 0 or frame_idx == traj_full.n_frames - 1:
            print(
                f"frame {frame_idx+1}/{traj_full.n_frames}: "
                f"vdW={delta_vdw/4.184:.2f}, "
                f"elec={delta_elec/4.184:.2f}, "
                f"GB={delta_gb/4.184:.2f}, "
                f"np={delta_np_kcal:.2f}, "
                f"total={delta_total_kcal:.2f} kcal/mol"
            )

    del ctx_complex_gb, ctx_protein_gb, ctx_ligand_gb
    del ctx_complex_vdw, ctx_protein_vdw, ctx_ligand_vdw
    del ctx_complex_elec, ctx_protein_elec, ctx_ligand_elec
    del int_complex_gb, int_protein_gb, int_ligand_gb
    del int_complex_vdw, int_protein_vdw, int_ligand_vdw
    del int_complex_elec, int_protein_elec, int_ligand_elec

    per_frame_df = pd.DataFrame(rows)

    summary_cols = [
        "delta_E_vdw_kcalmol",
        "delta_E_elec_kcalmol",
        "delta_G_gbsa_kcalmol",
        "delta_G_nonpolar_kcalmol",
        "delta_G_mmgbsa_kcalmol",
    ]
    summary_df = summarize_terms(per_frame_df, summary_cols)

    metadata["n_frames_total"] = int(traj_full_all.n_frames)
    metadata["n_frames_analyzed"] = int(traj_full.n_frames)
    metadata["complex_atoms"] = int(len(complex_indices))
    metadata["protein_atoms"] = int(len(protein_indices))
    metadata["ligand_atoms"] = int(len(ligand_indices))
    metadata["platform"] = platform.getName()

    per_frame_csv = f"{output_prefix}_mmgbsa_per_frame.csv"
    summary_csv = f"{output_prefix}_mmgbsa_summary.csv"
    metadata_json = f"{output_prefix}_mmgbsa_metadata.json"
    plot_png = f"{output_prefix}_mmgbsa_terms.png"

    metadata["outputs"] = {
        "per_frame_csv": per_frame_csv,
        "summary_csv": summary_csv,
        "metadata_json": metadata_json,
        "plot_png": plot_png,
    }

    per_frame_df.to_csv(per_frame_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)

    plt.figure(figsize=(9, 5))
    plt.plot(per_frame_df["time_ps"], per_frame_df["delta_E_vdw_kcalmol"], label="ΔE_vdW")
    plt.plot(per_frame_df["time_ps"], per_frame_df["delta_E_elec_kcalmol"], label="ΔE_elec")
    plt.plot(per_frame_df["time_ps"], per_frame_df["delta_G_gbsa_kcalmol"], label="ΔG_GBSA")
    plt.plot(per_frame_df["time_ps"], per_frame_df["delta_G_nonpolar_kcalmol"], label="ΔG_nonpolar")
    plt.plot(per_frame_df["time_ps"], per_frame_df["delta_G_mmgbsa_kcalmol"], label="Total", linewidth=2)
    plt.xlabel("Time (ps)")
    plt.ylabel("Energy (kcal/mol)")
    plt.title("Single-trajectory MM/GBSA-like terms")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_png, dpi=200)
    plt.show()

    print("\nWrote:")
    print(per_frame_csv)
    print(summary_csv)
    print(metadata_json)
    print(plot_png)

    return per_frame_df, summary_df, metadata
