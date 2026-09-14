"""
Equilibration analytics module.

Post-processes MD equilibration output files to produce quantitative KPIs
and time-series data for frontend visualization. Runs after all simulation
stages complete; never modifies the simulation itself.

Data flow:
  EquilibrationAnalytics.compute()
    ├─ _parse_log()       → thermodynamic time series from StateDataReporter TSV
    ├─ _compute_rmsd()    → backbone + ligand RMSD from NPT DCD trajectory
    ├─ _compute_structural_dynamics()
    │                      → RMSF, site retention and interaction occupancy
    └─ _evaluate_kpis()   → pass/warn/fail summary with absolute tolerances
"""

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── KPI thresholds (absolute, per scientific literature) ──────────────────────
# Plateau is defined as: std(last 20% of frames) < threshold
_ENERGY_STD_THRESHOLD_KJ = 500.0      # kJ/mol  — energy stable
_TEMP_STD_THRESHOLD_K = 5.0           # K       — thermostat converged
_DENSITY_STD_THRESHOLD_GCM3 = 0.05   # g/cm³   — barostat converged
_DENSITY_TARGET_GCM3 = 1.0           # g/cm³   — expected water density

# RMSD pass/warn/fail (final-20%-mean vs threshold)
_BACKBONE_RMSD_PASS_A = 2.5           # Å
_BACKBONE_RMSD_WARN_A = 3.5           # Å
_LIGAND_RMSD_PASS_A = 2.0             # Å
_LIGAND_RMSD_WARN_A = 5.0             # Å

# Max frames to load for RMSD (stride to cap compute time)
# Increased from 200 to 2500 to provide better resolution for long trajectories
_RMSD_MAX_FRAMES = 2500
_CONTACT_CUTOFF_NM = 0.45
_HYDROPHOBIC_CUTOFF_NM = 0.40
_SALT_BRIDGE_CUTOFF_NM = 0.40
INTERACTION_HOTSPOT_WEIGHTS = {
    "hydrogen_bond": 3.0,
    "salt_bridge": 2.5,
    "water_bridge": 1.5,
    "hydrophobic": 1.0,
}


def interaction_hotspot_score(
    *,
    hydrogen_bond: float,
    salt_bridge: float,
    water_bridge: float,
    hydrophobic: float,
) -> float:
    """Return the weighted interaction-hotspot score (not binding energy)."""

    return round(
        INTERACTION_HOTSPOT_WEIGHTS["hydrogen_bond"] * hydrogen_bond
        + INTERACTION_HOTSPOT_WEIGHTS["salt_bridge"] * salt_bridge
        + INTERACTION_HOTSPOT_WEIGHTS["water_bridge"] * water_bridge
        + INTERACTION_HOTSPOT_WEIGHTS["hydrophobic"] * hydrophobic,
        4,
    )
_SITE_DISPLACEMENT_CUTOFF_NM = 0.50
_PROTEIN_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})

_HYDROPHOBIC_PROTEIN_ATOMS = {
    "ALA": {"CB"},
    "VAL": {"CB", "CG1", "CG2"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "MET": {"CB", "CG", "SD", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"CB", "CG", "CD"},
    "CYS": {"CB", "SG"},
}

# Report interval × integration timestep (ps) used in equilibration_runner.py
# report_interval=1000, dt=0.004 ps → 4 ps per data point
_DEFAULT_REPORT_INTERVAL = 1000
_DT_PS = 0.004


def ligand_formal_charges_from_sdf_data(
    sdf_data: str | None,
) -> dict[str, int]:
    if not str(sdf_data or "").strip():
        return {}
    try:
        from rdkit import Chem

        molecule = Chem.MolFromMolBlock(
            str(sdf_data),
            removeHs=False,
            sanitize=True,
        )
        if molecule is None:
            return {}
        element_counts: Counter[str] = Counter()
        charges: dict[str, int] = {}
        for atom in molecule.GetAtoms():
            symbol = str(atom.GetSymbol())
            element_counts[symbol] += 1
            formal_charge = int(atom.GetFormalCharge())
            if formal_charge:
                charges[f"{symbol}{element_counts[symbol]}"] = formal_charge
        return charges
    except Exception:
        return {}


def _secondary_structure_fractions(dssp: Any) -> dict[str, list[float]]:
    import numpy as np

    assignments = np.asarray(dssp)
    fractions = {
        "helix_fraction": [],
        "sheet_fraction": [],
        "coil_fraction": [],
    }
    if assignments.ndim != 2 or not assignments.size:
        return fractions
    for frame in assignments:
        valid_count = int(np.count_nonzero(np.isin(frame, ("H", "E", "C"))))
        for field, code in (
            ("helix_fraction", "H"),
            ("sheet_fraction", "E"),
            ("coil_fraction", "C"),
        ):
            value = (
                float(np.count_nonzero(frame == code)) / valid_count
                if valid_count
                else 0.0
            )
            fractions[field].append(round(value, 4))
    return fractions


def _wernet_nilsson_presence(
    frame_xyz: Any,
    triplets: Any,
    box_vectors: Any | None = None,
) -> Any:
    import numpy as np

    candidates = np.asarray(triplets, dtype=int)
    if candidates.size == 0:
        return np.zeros(0, dtype=bool)
    donor_xyz = frame_xyz[candidates[:, 0]]
    hydrogen_xyz = frame_xyz[candidates[:, 1]]
    acceptor_xyz = frame_xyz[candidates[:, 2]]
    donor_acceptor = acceptor_xyz - donor_xyz
    donor_hydrogen = hydrogen_xyz - donor_xyz
    if box_vectors is not None:
        box = np.asarray(box_vectors, dtype=float)
        inverse_box = np.linalg.inv(box)
        donor_acceptor = (
            donor_acceptor @ inverse_box
            - np.round(donor_acceptor @ inverse_box)
        ) @ box
        donor_hydrogen = (
            donor_hydrogen @ inverse_box
            - np.round(donor_hydrogen @ inverse_box)
        ) @ box
    distances = np.linalg.norm(donor_acceptor, axis=1)
    denominators = distances * np.linalg.norm(donor_hydrogen, axis=1)
    valid = denominators > 0.0
    cosines = np.ones(len(candidates), dtype=float)
    cosines[valid] = np.sum(
        donor_acceptor[valid] * donor_hydrogen[valid],
        axis=1,
    ) / denominators[valid]
    angles_degrees = np.degrees(
        np.arccos(np.clip(cosines, -1.0, 1.0))
    )
    distance_cutoffs = 0.33 - 0.000044 * angles_degrees**2
    return (
        valid
        & (angles_degrees < 45.0)
        & (distances < distance_cutoffs)
    )


class EquilibrationAnalytics:
    """
    Computes quantitative KPIs from completed MD equilibration output files.

    Usage:
        result = EquilibrationAnalytics().compute(
            output_dir, system_id, topology_pdb, npt_traj, log_path, ligand_id
        )

    Returns a dict suitable for JSON serialization. On any internal failure
    returns {"error": "<message>"} rather than raising, so a completed
    simulation result is never lost due to an analytics bug.
    """

    def compute(
        self,
        output_dir: str,
        system_id: str,
        topology_pdb: str | None,
        nvt_traj: str | None = None,
        npt_traj: str | None = None,
        production_traj: str | None = None,
        log_path: str | None = None,
        ligand_id: str = "ligand",
        nvt_steps: int = 0,
        npt_steps: int = 0,
        production_steps: int = 0,
        nvt_report_interval: int = 1000,
        npt_report_interval: int = 1000,
        production_report_interval: int = 2500,
        report_interval: int | None = None,
        dt_ps: float | None = None,
        residue_mapping: dict[str, Any] | None = None,
        ligand_formal_charges: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """
        Run all analytics passes and return combined result dict.

        Args:
            output_dir:    MD output directory (unused directly; paths passed explicitly)
            system_id:     System identifier (used for logging only)
            topology_pdb:  Path to topology PDB (reference for RMSD)
            nvt_traj:      Path to NVT DCD trajectory (optional)
            npt_traj:      Path to NPT DCD trajectory (optional)
            production_traj: Path to production DCD trajectory (optional)
            log_path:      Path to StateDataReporter log (TSV)
            ligand_id:     Ligand identifier; residue name derived as ligand_id[:3].upper()
            nvt_steps:     Number of NVT steps (for time axis calculation)
            npt_steps:     Number of NPT steps (for time axis calculation)
            production_steps: Number of production steps (for time axis calculation)
            report_interval: MD report interval (steps) for fallback time calculation
            dt_ps:         MD timestep (ps) for time axis calculation

        Returns:
            {
                "thermodynamics": {...},
                "rmsd": {...},
                "kpi_summary": {...},
            }
            or {"error": "<message>"} on failure.
        """
        try:
            ligand_resname = (ligand_id[:3] if ligand_id else "LIG").upper()
            logger.info(
                f"[ANALYTICS] Starting analytics for system={system_id}, "
                f"ligand_resname={ligand_resname}"
            )

            thermo = self._parse_log(log_path)
            speed_values = [
                float(value)
                for value in thermo.get("speed_ns_per_day") or []
                if float(value) == float(value) and float(value) > 0.0
            ]
            
            # Use provided dt_ps or default
            if dt_ps is None:
                dt_ps = _DT_PS
            
            rmsd = self._compute_rmsd(
                topology_pdb, nvt_traj, npt_traj, production_traj,
                ligand_resname, nvt_steps, npt_steps, production_steps, dt_ps,
                nvt_report_interval, npt_report_interval, production_report_interval
            )
            structural_dynamics = self._compute_structural_dynamics(
                topology_pdb=topology_pdb,
                production_traj=production_traj,
                ligand_resname=ligand_resname,
                production_report_interval=production_report_interval,
                dt_ps=dt_ps,
                residue_mapping=residue_mapping,
                ligand_formal_charges=ligand_formal_charges,
            )
            kpi = self._evaluate_kpis(thermo, rmsd)

            logger.info(
                f"[ANALYTICS] Complete — overall_pass={kpi.get('overall_pass')}, "
                f"warnings={kpi.get('warnings')}"
            )
            return {
                "thermodynamics": thermo,
                "performance": {
                    "ns_per_day": speed_values[-1] if speed_values else None,
                    "source": (
                        "OpenMM StateDataReporter"
                        if speed_values
                        else "unavailable"
                    ),
                    "sample_count": len(speed_values),
                },
                "rmsd": rmsd,
                "structural_dynamics": structural_dynamics,
                "kpi_summary": kpi,
            }
        except Exception as e:
            logger.warning(f"[ANALYTICS] Analytics computation failed: {e}", exc_info=True)
            return {"error": str(e)}

    # ── Log parser ─────────────────────────────────────────────────────────────

    def _parse_log(self, log_path: str | None) -> dict[str, Any]:
        """
        Parse the OpenMM StateDataReporter TSV log file.

        The log is written with:
            separator='\t', step=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, volume=True, density=True, speed=True

        The reporter is re-attached (with clear()) for each stage (NVT, NPT), so
        step numbers restart. We use cumulative row index × report_interval × dt
        as the time axis to get a monotonic time series in picoseconds.

        Returns dict with arrays (empty on failure):
            step[], time_ps[], potential_energy_kjmol[],
            temperature_k[], density_gcm3[], volume_nm3[]
        """
        empty = {
            "step": [], "time_ps": [], "potential_energy_kjmol": [],
            "temperature_k": [], "density_gcm3": [], "volume_nm3": [],
            "speed_ns_per_day": [],
        }

        if not log_path or not os.path.exists(log_path):
            logger.debug(f"[ANALYTICS] Log file not found: {log_path}")
            return empty

        try:
            steps: list[int] = []
            times: list[float] = []
            energies: list[float] = []
            temperatures: list[float] = []
            densities: list[float] = []
            volumes: list[float] = []
            speeds: list[float] = []

            col_step = col_pe = col_temp = col_vol = col_den = col_speed = -1
            row_index = 0

            with open(log_path, "r") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue

                    # Header line starts with '#'
                    if line.startswith("#"):
                        # Parse column names from header
                        # Format: #"Step"\t"Potential Energy (kJ/mole)"\t...
                        header = line.lstrip("# ").replace('"', '')
                        cols = [c.strip() for c in header.split("\t")]
                        for i, c in enumerate(cols):
                            cl = c.lower()
                            if "step" in cl:
                                col_step = i
                            elif "potential" in cl:
                                col_pe = i
                            elif "temperature" in cl:
                                col_temp = i
                            elif "volume" in cl:
                                col_vol = i
                            elif "density" in cl:
                                col_den = i
                            elif "speed" in cl and "ns/day" in cl:
                                col_speed = i
                        continue

                    # Data row
                    parts = line.split("\t")
                    try:
                        def _safe(idx: int) -> float | None:
                            if idx < 0 or idx >= len(parts):
                                return None
                            try:
                                v = float(parts[idx])
                                return v if v == v else None  # NaN check
                            except (ValueError, IndexError):
                                return None

                        pe = _safe(col_pe)
                        temp = _safe(col_temp)
                        vol = _safe(col_vol)
                        den = _safe(col_den)
                        speed = _safe(col_speed)

                        # Skip rows where all values are None/NaN
                        if all(v is None for v in [pe, temp, vol, den, speed]):
                            continue

                        step_val = int(parts[col_step]) if col_step >= 0 else row_index
                        # Derive time from step number × integration timestep.
                        # This is correct for both equilibration (report_interval=1000)
                        # and production (report_interval=2500) logs.
                        time_ps = step_val * _DT_PS

                        steps.append(step_val)
                        times.append(round(time_ps, 3))
                        energies.append(pe if pe is not None else float("nan"))
                        temperatures.append(temp if temp is not None else float("nan"))
                        volumes.append(vol if vol is not None else float("nan"))
                        densities.append(den if den is not None else float("nan"))
                        speeds.append(
                            speed if speed is not None else float("nan")
                        )
                        row_index += 1

                    except (ValueError, IndexError):
                        # Malformed row — skip silently
                        continue

            logger.info(f"[ANALYTICS] Parsed {row_index} rows from log")
            return {
                "step": steps,
                "time_ps": times,
                "potential_energy_kjmol": energies,
                "temperature_k": temperatures,
                "density_gcm3": densities,
                "volume_nm3": volumes,
                "speed_ns_per_day": speeds,
            }

        except Exception as e:
            logger.warning(f"[ANALYTICS] Log parse failed: {e}")
            return empty

    # ── RMSD computation ───────────────────────────────────────────────────────

    def _compute_rmsd(
        self,
        topology_pdb: str | None,
        nvt_traj: str | None,
        npt_traj: str | None,
        production_traj: str | None,
        ligand_resname: str,
        nvt_steps: int = 0,
        npt_steps: int = 0,
        production_steps: int = 0,
        dt_ps: float = _DT_PS,
        nvt_report_interval: int = 1000,
        npt_report_interval: int = 1000,
        production_report_interval: int = 2500,
    ) -> dict[str, Any]:
        """
        Compute backbone and ligand RMSD from all available trajectories (NVT, NPT, Production).

        Processes all trajectory phases and combines them into a single continuous dataset
        with phase boundary markers for visualization.

        Method:
        1. Load each trajectory (strided to stay under _RMSD_MAX_FRAMES total).
        2. Apply robust PBC imaging (anchor to protein/largest molecule) to fix ligand jumps.
        3. Align (superpose) protein backbone to frame 0 of first trajectory.
        4. Compute RMSD for backbone and ligand.
        5. Combine phases with proper time offsets.

        Returns:
            {
                "time_ps": [...],
                "backbone_rmsd_angstrom": [...],
                "ligand_rmsd_angstrom": [...],
                "phase_boundaries": [{"phase": "nvt", "start_ps": 0, "end_ps": 100}, ...],
                "warnings": [...],
            }
        """
        empty = {
            "time_ps": [],
            "backbone_rmsd_angstrom": [],
            "ligand_rmsd_angstrom": [],
            "phase_boundaries": [],
            "per_phase_local": {},
            "warnings": [],
        }

        if not topology_pdb or not os.path.exists(topology_pdb):
            logger.debug(f"[ANALYTICS] Topology PDB not found: {topology_pdb} — skipping RMSD")
            return empty
        
        # Collect available trajectories with their report intervals
        trajectories = []
        if nvt_traj and os.path.exists(nvt_traj):
            trajectories.append(("nvt", nvt_traj, nvt_steps, nvt_report_interval))
        if npt_traj and os.path.exists(npt_traj):
            trajectories.append(("npt", npt_traj, npt_steps, npt_report_interval))
        if production_traj and os.path.exists(production_traj):
            trajectories.append(("production", production_traj, production_steps, production_report_interval))
        
        if not trajectories:
            logger.debug("[ANALYTICS] No trajectory files found — skipping RMSD")
            return empty

        try:
            import mdtraj as md
            import numpy as np

            warnings_list: list[str] = []
            
            # Calculate total frames and stride to stay under _RMSD_MAX_FRAMES
            total_frames = 0
            frame_counts = []
            for phase_name, traj_path, steps, report_interval in trajectories:
                temp_traj = md.load(traj_path, top=topology_pdb)
                n = temp_traj.n_frames
                frame_counts.append(n)
                total_frames += n
            
            if total_frames == 0:
                return {**empty, "warnings": ["All trajectories have 0 frames"]}
            
            # Calculate global stride to keep total analyzed frames under limit
            stride = max(1, total_frames // _RMSD_MAX_FRAMES)
            logger.info(
                f"[ANALYTICS] RMSD: {total_frames} total frames across {len(trajectories)} phases, "
                f"stride={stride} (analyzing ~{total_frames // stride} frames)"
            )
            
            # Process each trajectory phase
            all_time_ps = []
            all_backbone_rmsd = []
            all_ligand_rmsd = []
            phase_boundaries = []
            per_phase_local: dict[str, Any] = {}
            
            current_time_offset_ps = 0.0
            reference_traj = None
            backbone_sel = None
            ligand_sel = None
            has_ligand = False

            for phase_idx, (phase_name, traj_path, steps, report_interval) in enumerate(trajectories):
                logger.info(f"[ANALYTICS] Processing {phase_name.upper()} trajectory: {traj_path}")
                
                # Load trajectory with stride
                traj = md.load(traj_path, top=topology_pdb)
                if stride > 1:
                    traj = traj[::stride]
                
                n_frames = traj.n_frames
                if n_frames == 0:
                    logger.warning(f"[ANALYTICS] {phase_name.upper()} trajectory has 0 frames after striding")
                    continue
                
                # ── Robust PBC Imaging & Alignment ─────────────────────────────────
                if traj.unitcell_lengths is None:
                    if phase_idx == 0:
                        warnings_list.append("No unit cell info — skipping PBC correction (RMSD may be inflated)")
                else:
                    # 1. Identify anchors
                    anchor_molecules = []
                    protein_sel_temp = traj.topology.select('protein')
                    molecules = traj.topology.find_molecules()
                    
                    if len(protein_sel_temp) > 10:
                        protein_atom_set = set(protein_sel_temp)
                        anchor_molecules = [
                            sorted(list(mol), key=lambda a: a.index) 
                            for mol in molecules 
                            if any(atom.index in protein_atom_set for atom in mol)
                        ]
                    
                    # Fallback to largest molecule
                    if not anchor_molecules and len(molecules) > 0:
                        largest_mol = max(molecules, key=len)
                        anchor_molecules = [sorted(list(largest_mol), key=lambda a: a.index)]
                        if phase_idx == 0:
                            logger.info(f"[ANALYTICS] Fallback: Anchoring PBC to largest molecule ({len(largest_mol)} atoms)")
                    
                    # Apply imaging
                    if anchor_molecules:
                        traj.image_molecules(inplace=True, anchor_molecules=anchor_molecules)
                    else:
                        traj.image_molecules(inplace=True)
                
                # 2. Select Atoms for RMSD (only once, on first trajectory)
                if phase_idx == 0:
                    # Backbone/CA
                    backbone_sel = traj.topology.select("protein and name CA")
                    if len(backbone_sel) == 0:
                        backbone_sel = traj.topology.select("protein and backbone")
                    if len(backbone_sel) == 0:
                        backbone_sel = traj.topology.select("protein")
                    
                    if len(backbone_sel) == 0:
                        logger.warning("[ANALYTICS] No protein atoms found — skipping RMSD")
                        return {**empty, "warnings": ["No protein atoms found"]}
                    
                    # Ligand
                    ligand_sel = traj.topology.select(f"resname {ligand_resname} and not element H")
                    
                    if len(ligand_sel) == 0:
                        # Fallback: non-protein, non-water, non-ion
                        solvent_query = "(water or resname HOH or resname WAT or resname SOL or resname TIP3 or resname TIP4P)"
                        ion_query = "(resname NA or resname CL or resname MG or resname K or resname CA or resname ZN)"
                        ligand_query = f"not (protein or {solvent_query} or {ion_query}) and not element H"
                        
                        ligand_sel = traj.topology.select(ligand_query)
                        
                        if len(ligand_sel) > 0:
                            warnings_list.append(
                                f"Ligand resname '{ligand_resname}' not found — "
                                f"used auto-detected ligand ({len(ligand_sel)} atoms)"
                            )
                    
                    has_ligand = len(ligand_sel) > 0
                    
                    # Store reference trajectory (first frame of first phase)
                    reference_traj = traj
                
                # 3. Align (Superpose)
                # Align to frame 0 of the reference trajectory (first phase)
                if phase_idx == 0:
                    traj.superpose(traj, 0, atom_indices=backbone_sel)
                else:
                    # Align to frame 0 of reference trajectory
                    traj.superpose(reference_traj, 0, atom_indices=backbone_sel)

                # 4. Compute RMSD
                # mdtraj.rmsd returns result in nanometers, we need Angstroms (* 10)
                
                # Backbone RMSD
                if phase_idx == 0:
                    rmsd_bb_nm = md.rmsd(traj, traj, 0, atom_indices=backbone_sel)
                else:
                    rmsd_bb_nm = md.rmsd(traj, reference_traj, 0, atom_indices=backbone_sel)
                backbone_rmsd_phase = [round(float(r) * 10.0, 4) for r in rmsd_bb_nm]
                # Local per-phase backbone RMSD (reference = frame 0 of this phase)
                rmsd_bb_local_nm = md.rmsd(traj, traj, 0, atom_indices=backbone_sel)
                backbone_rmsd_local_phase = [round(float(r) * 10.0, 4) for r in rmsd_bb_local_nm]
                
                # Ligand RMSD
                ligand_rmsd_phase = []
                if has_ligand:
                    # Get ligand coordinates (n_frames, n_atoms, 3)
                    ligand_xyz = traj.xyz[:, ligand_sel, :]
                    ref_ligand_xyz = reference_traj.xyz[0, ligand_sel, :]
                    
                    # Calculate displacement
                    diff = ligand_xyz - ref_ligand_xyz
                    
                    # Sum of squares along spatial dimension (axis 2) -> (n_frames, n_atoms)
                    dist_sq = np.sum(diff**2, axis=2)
                    
                    # Mean over atoms (axis 1) -> (n_frames,)
                    mean_dist_sq = np.mean(dist_sq, axis=1)
                    
                    # Sqrt -> RMSD in nm
                    rmsd_lig_nm = np.sqrt(mean_dist_sq)
                    
                    ligand_rmsd_phase = [round(float(r) * 10.0, 4) for r in rmsd_lig_nm]
                    # Local per-phase ligand RMSD (reference = phase frame 0)
                    local_ref_ligand_xyz = traj.xyz[0, ligand_sel, :]
                    local_diff = ligand_xyz - local_ref_ligand_xyz
                    local_dist_sq = np.sum(local_diff**2, axis=2)
                    local_mean_dist_sq = np.mean(local_dist_sq, axis=1)
                    local_rmsd_lig_nm = np.sqrt(local_mean_dist_sq)
                    ligand_rmsd_local_phase = [round(float(r) * 10.0, 4) for r in local_rmsd_lig_nm]
                else:
                    ligand_rmsd_local_phase = []

                # 5. Calculate time axis for this phase
                # Each kept frame is spaced by stride * report_interval integration steps.
                phase_duration_ps = steps * dt_ps
                time_per_frame_ps = stride * report_interval * dt_ps
                time_ps_phase = [
                    round(current_time_offset_ps + (i * time_per_frame_ps), 3)
                    for i in range(n_frames)
                ]
                local_time_ps_phase = [round(i * time_per_frame_ps, 3) for i in range(n_frames)]

                # Record phase boundary (nominal span from configured step counts)
                phase_start_ps = current_time_offset_ps
                phase_end_ps = current_time_offset_ps + phase_duration_ps
                phase_boundaries.append({
                    "phase": phase_name,
                    "start_ps": round(phase_start_ps, 1),
                    "end_ps": round(phase_end_ps, 1)
                })
                per_phase_local[phase_name] = {
                    "time_ps": local_time_ps_phase,
                    "backbone_rmsd_angstrom": backbone_rmsd_local_phase,
                    "ligand_rmsd_angstrom": ligand_rmsd_local_phase,
                }

                # Append to combined arrays
                all_time_ps.extend(time_ps_phase)
                all_backbone_rmsd.extend(backbone_rmsd_phase)
                if has_ligand:
                    all_ligand_rmsd.extend(ligand_rmsd_phase)

                # Next phase must start strictly after the last plotted sample time.
                # If the DCD has more frames than steps/report_interval implies, or step
                # counts are underreported, sample times can extend past phase_end_ps;
                # starting the next phase at phase_end_ps alone makes time go backward
                # on the Plotly line plot.
                last_sample_ps = time_ps_phase[-1] if time_ps_phase else phase_start_ps
                _eps = max(1e-3, 0.01 * time_per_frame_ps) if time_per_frame_ps > 0 else 1e-3
                if last_sample_ps <= phase_end_ps + 1e-6:
                    current_time_offset_ps = phase_end_ps
                else:
                    current_time_offset_ps = last_sample_ps + _eps
                    warnings_list.append(
                        f"RMSD time axis: {phase_name} trajectory spans past nominal phase end "
                        f"({last_sample_ps:.1f} ps > {phase_end_ps:.1f} ps); next phase offset adjusted "
                        "to keep a monotonic plot."
                    )
                
                logger.info(
                    f"[ANALYTICS] {phase_name.upper()} RMSD: {n_frames} frames, "
                    f"time range: {phase_start_ps:.1f}-{phase_end_ps:.1f} ps"
                )

            logger.info(
                f"[ANALYTICS] RMSD computed: {len(all_backbone_rmsd)} total points across {len(phase_boundaries)} phases, "
                f"backbone max={max(all_backbone_rmsd) if all_backbone_rmsd else 0:.2f}Å"
            )

            return {
                "time_ps": all_time_ps,
                "backbone_rmsd_angstrom": all_backbone_rmsd,
                "ligand_rmsd_angstrom": all_ligand_rmsd,
                "phase_boundaries": phase_boundaries,
                "per_phase_local": per_phase_local,
                "warnings": warnings_list,
            }

        except ImportError:
            logger.warning("[ANALYTICS] MDTraj not available — skipping RMSD")
            return {**empty, "warnings": ["MDTraj not available"]}
        except Exception as e:
            logger.warning(f"[ANALYTICS] RMSD computation failed: {e}", exc_info=True)
            return {**empty, "warnings": [f"RMSD computation failed: {str(e)}"]}

    def _compute_structural_dynamics(
        self,
        *,
        topology_pdb: str | None,
        production_traj: str | None,
        ligand_resname: str,
        production_report_interval: int,
        dt_ps: float,
        residue_mapping: dict[str, Any] | None = None,
        analysis_start_ps: float = 0.0,
        ligand_formal_charges: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Compute production-only flexibility and protein–ligand stability.

        Coordinates are imaged through the periodic box and aligned on protein
        C-alpha atoms before analysis. "Reference-site retained" is deliberately
        explicit: the ligand heavy-atom centroid must stay within 5 Å of its
        first production-frame position and retain at least one protein contact
        within 4.5 Å. It is not presented as proof of binding.
        """
        empty = {
            "time_ps": [],
            "rmsf": {"residues": [], "ca_rmsf_angstrom": []},
            "ligand_rmsf": {"atoms": [], "rmsf_angstrom": []},
            "radius_of_gyration": {
                "protein_angstrom": [],
                "ligand_angstrom": [],
                "complex_angstrom": [],
            },
            "secondary_structure": {
                "helix_fraction": [],
                "sheet_fraction": [],
                "coil_fraction": [],
            },
            "pocket": {
                "ligand_centroid_displacement_angstrom": [],
                "minimum_protein_distance_angstrom": [],
                "reference_site_retained": [],
                "retained_fraction": None,
                "site_displacement_cutoff_angstrom": 5.0,
                "contact_cutoff_angstrom": 4.5,
            },
            "contacts": {"residues": [], "distance_series": {}},
            "water_bridges": {
                "residues": [],
                "method": "Wernet-Nilsson two-sided hydrogen-bond geometry",
            },
            "salt_bridges": {
                "applicable": False,
                "ligand_charged_atoms": [],
                "residues": [],
                "distance_cutoff_angstrom": 4.0,
            },
            "interaction_network": {"nodes": [], "edges": []},
            "interface_rin": {"applicable": False, "interfaces": []},
            "warnings": [],
        }
        if (
            not topology_pdb
            or not production_traj
            or not os.path.exists(topology_pdb)
            or not os.path.exists(production_traj)
        ):
            return empty

        try:
            import mdtraj as md
            import numpy as np

            with md.open(production_traj) as trajectory_file:
                source_frame_count = len(trajectory_file)
            if source_frame_count == 0:
                return {**empty, "warnings": ["Production trajectory has no frames"]}
            stride = max(1, source_frame_count // _RMSD_MAX_FRAMES)
            trajectory = md.load(
                production_traj,
                top=topology_pdb,
                stride=stride,
            )
            time_per_frame_ps = (
                max(1, int(production_report_interval))
                * float(dt_ps)
                * stride
            )
            discarded_frames = 0
            if analysis_start_ps > 0.0:
                discarded_frames = min(
                    trajectory.n_frames,
                    int(np.ceil(float(analysis_start_ps) / time_per_frame_ps)),
                )
                trajectory = trajectory[discarded_frames:]
                if trajectory.n_frames == 0:
                    return {
                        **empty,
                        "warnings": [
                            "No production frames remain after applying the "
                            f"{float(analysis_start_ps):.3f} ps analysis start."
                        ],
                        "analysis_window": {
                            "start_ps": float(analysis_start_ps),
                            "discarded_frames": discarded_frames,
                        },
                    }

            warnings_list: list[str] = []
            if trajectory.unitcell_lengths is not None:
                try:
                    protein_indices = trajectory.topology.select("protein")
                    protein_set = set(int(value) for value in protein_indices)
                    molecules = trajectory.topology.find_molecules()
                    anchors = [
                        sorted(list(molecule), key=lambda atom: atom.index)
                        for molecule in molecules
                        if any(atom.index in protein_set for atom in molecule)
                    ]
                    if anchors:
                        trajectory.image_molecules(
                            inplace=True, anchor_molecules=anchors
                        )
                    else:
                        trajectory.image_molecules(inplace=True)
                except Exception as exc:
                    warnings_list.append(f"PBC imaging failed: {exc}")

            ca_indices = trajectory.topology.select("protein and name CA")
            if len(ca_indices) == 0:
                ca_indices = trajectory.topology.select(
                    "protein and backbone"
                )
            protein_heavy = trajectory.topology.select(
                "protein and not element H"
            )
            ligand_heavy = trajectory.topology.select(
                f"resname {ligand_resname} and not element H"
            )
            if len(ligand_heavy) == 0:
                ligand_heavy = trajectory.topology.select(
                    "not (protein or water or resname HOH or resname WAT "
                    "or resname SOL or resname NA or resname CL or resname K "
                    "or resname MG or resname CA or resname ZN) "
                    "and not element H"
                )
                if len(ligand_heavy):
                    warnings_list.append(
                        f"Ligand resname '{ligand_resname}' was not found; "
                        "used the non-protein, non-solvent heavy-atom selection."
                    )
            if len(ca_indices) == 0 or len(protein_heavy) == 0:
                return {
                    **empty,
                    "warnings": ["Protein atoms were not available for analysis"],
                }
            if len(ligand_heavy) == 0:
                return {
                    **empty,
                    "warnings": ["Ligand heavy atoms were not available for analysis"],
                }

            trajectory.superpose(
                trajectory, 0, atom_indices=ca_indices
            )
            time_ps = [
                round(
                    float(analysis_start_ps) + index * time_per_frame_ps,
                    3,
                )
                for index in range(trajectory.n_frames)
            ]

            protein_residues = []
            seen_residue_indices: set[int] = set()
            for atom_index in protein_heavy:
                residue = trajectory.topology.atom(int(atom_index)).residue
                if residue.index in seen_residue_indices:
                    continue
                seen_residue_indices.add(residue.index)
                protein_residues.append(residue)
            mapping_rows = [
                row
                for row in (residue_mapping or {}).get("residues") or []
                if isinstance(row, dict)
            ]
            mapped_by_residue_index: dict[int, dict[str, Any]] = {}
            if mapping_rows:
                if len(mapping_rows) != len(protein_residues):
                    warnings_list.append(
                        "Native residue mapping was not applied because its "
                        f"{len(mapping_rows)} residues do not match the "
                        f"{len(protein_residues)} analyzed protein residues."
                    )
                else:
                    mismatches = [
                        (residue, row)
                        for residue, row in zip(protein_residues, mapping_rows)
                        if str(row.get("structure_residue_name") or "").upper()
                        not in {"", str(residue.name).upper()}
                    ]
                    if mismatches:
                        warnings_list.append(
                            "Native residue mapping was not applied because "
                            f"{len(mismatches)} residue names did not match."
                        )
                    else:
                        mapped_by_residue_index = {
                            residue.index: row
                            for residue, row in zip(
                                protein_residues,
                                mapping_rows,
                            )
                        }

            def residue_label(residue: Any) -> str:
                mapped = mapped_by_residue_index.get(residue.index)
                if mapped is not None:
                    insertion_code = str(
                        mapped.get("native_insertion_code") or ""
                    ).strip()
                    return (
                        f"{mapped.get('native_residue_name') or residue.name}"
                        f"{mapped.get('native_residue_number')}{insertion_code}"
                        f" · chain {mapped.get('native_chain') or '_'}"
                    )
                chain = residue.chain
                chain_id = (
                    str(getattr(chain, "chain_id", "") or "").strip()
                    or str(chain.index + 1)
                )
                return f"{residue.name}{residue.resSeq} · chain {chain_id}"

            rmsf_values: list[float] = []
            rmsf_labels: list[str] = []
            if trajectory.n_frames > 1:
                rmsf_nm = md.rmsf(
                    trajectory,
                    None,
                    0,
                    atom_indices=ca_indices,
                )
                rmsf_values = [
                    round(float(value) * 10.0, 4) for value in rmsf_nm
                ]
                rmsf_labels = [
                    residue_label(
                        trajectory.topology.atom(int(atom_index)).residue
                    )
                    for atom_index in ca_indices
                ]

            ligand_rmsf_values: list[float] = []
            ligand_rmsf_labels: list[str] = []
            if trajectory.n_frames > 1:
                ligand_rmsf_nm = md.rmsf(
                    trajectory,
                    None,
                    0,
                    atom_indices=ligand_heavy,
                )
                ligand_rmsf_values = [
                    round(float(value) * 10.0, 4)
                    for value in ligand_rmsf_nm
                ]
                ligand_rmsf_labels = [
                    str(trajectory.topology.atom(int(atom_index)).name)
                    for atom_index in ligand_heavy
                ]

            protein_rg = [
                round(float(value) * 10.0, 4)
                for value in md.compute_rg(
                    trajectory.atom_slice(protein_heavy)
                )
            ]
            ligand_rg = [
                round(float(value) * 10.0, 4)
                for value in md.compute_rg(
                    trajectory.atom_slice(ligand_heavy)
                )
            ]
            complex_indices = np.concatenate(
                (protein_heavy, ligand_heavy)
            )
            complex_rg = [
                round(float(value) * 10.0, 4)
                for value in md.compute_rg(
                    trajectory.atom_slice(complex_indices)
                )
            ]

            secondary_structure = {
                "helix_fraction": [],
                "sheet_fraction": [],
                "coil_fraction": [],
            }
            try:
                dssp = md.compute_dssp(trajectory, simplified=True)
                if dssp.size:
                    secondary_structure = _secondary_structure_fractions(dssp)
            except Exception as exc:
                warnings_list.append(
                    f"Secondary-structure analysis could not be calculated: {exc}"
                )

            residue_atom_positions: dict[str, list[int]] = {}
            for local_position, atom_index in enumerate(protein_heavy):
                residue = trajectory.topology.atom(int(atom_index)).residue
                label = residue_label(residue)
                residue_atom_positions.setdefault(label, []).append(
                    local_position
                )

            reference_centroid = np.mean(
                trajectory.xyz[0, ligand_heavy, :], axis=0
            )
            centroid_displacements: list[float] = []
            minimum_distances: list[float] = []
            retained: list[bool] = []
            per_residue_distances = {
                label: [] for label in residue_atom_positions
            }
            contact_counts = {
                label: 0 for label in residue_atom_positions
            }
            contact_backbone_counts = {
                label: 0 for label in residue_atom_positions
            }
            contact_sidechain_counts = {
                label: 0 for label in residue_atom_positions
            }
            closest_ligand_atoms: dict[str, Counter[str]] = {
                label: Counter() for label in residue_atom_positions
            }
            closest_protein_atoms: dict[str, Counter[str]] = {
                label: Counter() for label in residue_atom_positions
            }
            hydrophobic_counts = {
                label: 0 for label in residue_atom_positions
            }
            hydrophobic_backbone_counts = {
                label: 0 for label in residue_atom_positions
            }
            hydrophobic_sidechain_counts = {
                label: 0 for label in residue_atom_positions
            }
            hydrophobic_ligand_positions = {
                local_position
                for local_position, atom_index in enumerate(ligand_heavy)
                if (
                    trajectory.topology.atom(int(atom_index)).element
                    is not None
                    and trajectory.topology.atom(
                        int(atom_index)
                    ).element.symbol
                    in {"C", "S"}
                )
            }
            hydrophobic_protein_positions = {
                local_position
                for local_position, atom_index in enumerate(protein_heavy)
                if (
                    str(
                        trajectory.topology.atom(
                            int(atom_index)
                        ).name
                    ).upper()
                    in _HYDROPHOBIC_PROTEIN_ATOMS.get(
                        str(
                            trajectory.topology.atom(
                                int(atom_index)
                            ).residue.name
                        ).upper(),
                        set(),
                    )
                )
            }
            normalized_formal_charges = {
                str(name).strip(): int(charge)
                for name, charge in (ligand_formal_charges or {}).items()
                if int(charge)
            }
            charged_ligand_positions = {
                local_position: normalized_formal_charges[
                    str(
                        trajectory.topology.atom(int(atom_index)).name
                    ).strip()
                ]
                for local_position, atom_index in enumerate(ligand_heavy)
                if str(
                    trajectory.topology.atom(int(atom_index)).name
                ).strip()
                in normalized_formal_charges
            }
            charged_protein_positions: dict[int, int] = {}
            for local_position, atom_index in enumerate(protein_heavy):
                atom = trajectory.topology.atom(int(atom_index))
                residue_name = str(atom.residue.name).upper()
                atom_name = str(atom.name).upper()
                if (
                    residue_name == "ASP"
                    and atom_name in {"OD1", "OD2"}
                ) or (
                    residue_name == "GLU"
                    and atom_name in {"OE1", "OE2"}
                ):
                    charged_protein_positions[local_position] = -1
                elif (
                    residue_name == "LYS"
                    and atom_name == "NZ"
                ) or (
                    residue_name == "ARG"
                    and atom_name in {"NE", "NH1", "NH2"}
                ) or (
                    residue_name in {"HIP", "HSP"}
                    and atom_name in {"ND1", "NE2"}
                ):
                    charged_protein_positions[local_position] = 1
            salt_bridge_counts = {
                label: 0 for label in residue_atom_positions
            }
            salt_bridge_backbone_counts = {
                label: 0 for label in residue_atom_positions
            }
            salt_bridge_sidechain_counts = {
                label: 0 for label in residue_atom_positions
            }
            salt_bridge_ligand_atoms: dict[str, Counter[str]] = {
                label: Counter() for label in residue_atom_positions
            }
            for frame_xyz in trajectory.xyz:
                ligand_xyz = frame_xyz[ligand_heavy, :]
                protein_xyz = frame_xyz[protein_heavy, :]
                ligand_centroid = np.mean(ligand_xyz, axis=0)
                displacement_nm = float(
                    np.linalg.norm(ligand_centroid - reference_centroid)
                )
                differences = (
                    ligand_xyz[:, np.newaxis, :]
                    - protein_xyz[np.newaxis, :, :]
                )
                atom_min_nm = np.sqrt(
                    np.sum(differences * differences, axis=2)
                ).min(axis=0)
                minimum_nm = float(np.min(atom_min_nm))
                centroid_displacements.append(
                    round(displacement_nm * 10.0, 4)
                )
                minimum_distances.append(round(minimum_nm * 10.0, 4))
                retained.append(
                    displacement_nm <= _SITE_DISPLACEMENT_CUTOFF_NM
                    and minimum_nm <= _CONTACT_CUTOFF_NM
                )
                for label, positions in residue_atom_positions.items():
                    position_array = np.asarray(positions, dtype=int)
                    residue_distances = np.sqrt(
                        np.sum(
                            differences[:, position_array, :]
                            * differences[:, position_array, :],
                            axis=2,
                        )
                    )
                    ligand_local, residue_local = np.unravel_index(
                        int(np.argmin(residue_distances)),
                        residue_distances.shape,
                    )
                    residue_min_nm = float(
                        residue_distances[ligand_local, residue_local]
                    )
                    per_residue_distances[label].append(
                        round(residue_min_nm * 10.0, 4)
                    )
                    if residue_min_nm <= _CONTACT_CUTOFF_NM:
                        contact_counts[label] += 1
                        backbone_local_positions = [
                            local_index
                            for local_index, protein_position in enumerate(
                                position_array
                            )
                            if str(
                                trajectory.topology.atom(
                                    int(protein_heavy[protein_position])
                                ).name
                            ).upper()
                            in _PROTEIN_BACKBONE_ATOM_NAMES
                        ]
                        sidechain_local_positions = [
                            local_index
                            for local_index, protein_position in enumerate(
                                position_array
                            )
                            if str(
                                trajectory.topology.atom(
                                    int(protein_heavy[protein_position])
                                ).name
                            ).upper()
                            not in _PROTEIN_BACKBONE_ATOM_NAMES
                        ]
                        if backbone_local_positions and np.any(
                            residue_distances[:, backbone_local_positions]
                            <= _CONTACT_CUTOFF_NM
                        ):
                            contact_backbone_counts[label] += 1
                        if sidechain_local_positions and np.any(
                            residue_distances[:, sidechain_local_positions]
                            <= _CONTACT_CUTOFF_NM
                        ):
                            contact_sidechain_counts[label] += 1
                        ligand_atom = trajectory.topology.atom(
                            int(ligand_heavy[ligand_local])
                        )
                        protein_atom = trajectory.topology.atom(
                            int(protein_heavy[position_array[residue_local]])
                        )
                        closest_ligand_atoms[label][str(ligand_atom.name)] += 1
                        closest_protein_atoms[label][str(protein_atom.name)] += 1
                    residue_hydrophobic_positions = [
                        position
                        for position in positions
                        if position in hydrophobic_protein_positions
                    ]
                    if (
                        hydrophobic_ligand_positions
                        and residue_hydrophobic_positions
                    ):
                        hydrophobic_distances = differences[
                            np.ix_(
                                sorted(hydrophobic_ligand_positions),
                                residue_hydrophobic_positions,
                            )
                        ]
                        if np.any(
                            np.sqrt(
                                np.sum(
                                    hydrophobic_distances
                                    * hydrophobic_distances,
                                    axis=2,
                                )
                            )
                            <= _HYDROPHOBIC_CUTOFF_NM
                        ):
                            hydrophobic_counts[label] += 1
                            hydrophobic_sidechain_counts[label] += 1
                    charged_pairs = [
                        (ligand_position, protein_position)
                        for ligand_position, ligand_charge
                        in charged_ligand_positions.items()
                        for protein_position in positions
                        if protein_position in charged_protein_positions
                        and (
                            ligand_charge
                            * charged_protein_positions[protein_position]
                            < 0
                        )
                    ]
                    if charged_pairs:
                        pair_distances = np.asarray(
                            [
                                differences[
                                    ligand_position,
                                    protein_position,
                                ]
                                for ligand_position, protein_position
                                in charged_pairs
                            ],
                            dtype=float,
                        )
                        pair_norms = np.sqrt(
                            np.sum(pair_distances * pair_distances, axis=1)
                        )
                        closest_pair = int(np.argmin(pair_norms))
                        if (
                            float(pair_norms[closest_pair])
                            <= _SALT_BRIDGE_CUTOFF_NM
                        ):
                            salt_bridge_counts[label] += 1
                            ligand_position, protein_position = charged_pairs[
                                closest_pair
                            ]
                            protein_atom = trajectory.topology.atom(
                                int(protein_heavy[protein_position])
                            )
                            if (
                                str(protein_atom.name).upper()
                                in _PROTEIN_BACKBONE_ATOM_NAMES
                            ):
                                salt_bridge_backbone_counts[label] += 1
                            else:
                                salt_bridge_sidechain_counts[label] += 1
                            ligand_atom = trajectory.topology.atom(
                                int(ligand_heavy[ligand_position])
                            )
                            salt_bridge_ligand_atoms[label][
                                str(ligand_atom.name)
                            ] += 1

            hydrogen_bond_frames: dict[str, set[int]] = {
                label: set() for label in residue_atom_positions
            }
            hydrogen_bond_backbone_frames: dict[str, set[int]] = {
                label: set() for label in residue_atom_positions
            }
            hydrogen_bond_sidechain_frames: dict[str, set[int]] = {
                label: set() for label in residue_atom_positions
            }
            hydrogen_bond_ligand_atoms: dict[str, Counter[str]] = {
                label: Counter() for label in residue_atom_positions
            }
            water_bridge_frames: dict[str, set[int]] = {
                label: set() for label in residue_atom_positions
            }
            water_bridge_backbone_frames: dict[str, set[int]] = {
                label: set() for label in residue_atom_positions
            }
            water_bridge_sidechain_frames: dict[str, set[int]] = {
                label: set() for label in residue_atom_positions
            }
            try:
                ligand_set = set(int(value) for value in ligand_heavy)
                protein_set = set(int(value) for value in protein_heavy)
                for frame_index, triples in enumerate(
                    md.wernet_nilsson(trajectory, exclude_water=True)
                ):
                    labels_in_frame: set[str] = set()
                    backbone_labels_in_frame: set[str] = set()
                    sidechain_labels_in_frame: set[str] = set()
                    for donor, _hydrogen, acceptor in triples:
                        donor_i = int(donor)
                        acceptor_i = int(acceptor)
                        protein_atom = None
                        ligand_atom = None
                        if donor_i in ligand_set and acceptor_i in protein_set:
                            protein_atom = acceptor_i
                            ligand_atom = donor_i
                        elif (
                            acceptor_i in ligand_set
                            and donor_i in protein_set
                        ):
                            protein_atom = donor_i
                            ligand_atom = acceptor_i
                        if protein_atom is not None:
                            protein_topology_atom = trajectory.topology.atom(
                                protein_atom
                            )
                            label = residue_label(protein_topology_atom.residue)
                            labels_in_frame.add(label)
                            if (
                                str(protein_topology_atom.name).upper()
                                in _PROTEIN_BACKBONE_ATOM_NAMES
                            ):
                                backbone_labels_in_frame.add(label)
                            else:
                                sidechain_labels_in_frame.add(label)
                            hydrogen_bond_ligand_atoms[label][
                                str(trajectory.topology.atom(ligand_atom).name)
                            ] += 1
                    for label in labels_in_frame:
                        hydrogen_bond_frames.setdefault(label, set()).add(
                            frame_index
                        )
                    for label in backbone_labels_in_frame:
                        hydrogen_bond_backbone_frames.setdefault(
                            label, set()
                        ).add(frame_index)
                    for label in sidechain_labels_in_frame:
                        hydrogen_bond_sidechain_frames.setdefault(
                            label, set()
                        ).add(frame_index)
            except Exception as exc:
                warnings_list.append(
                    f"Hydrogen-bond occupancy could not be calculated: {exc}"
                )

            try:
                topology = trajectory.topology
                protein_all = {
                    int(value) for value in topology.select("protein")
                }
                ligand_residue_indices = {
                    int(topology.atom(index).residue.index)
                    for index in ligand_set
                }
                ligand_all = {
                    int(atom.index)
                    for atom in topology.atoms
                    if int(atom.residue.index) in ligand_residue_indices
                }
                donor_pairs: list[tuple[int, int]] = []
                for left, right in topology.bonds:
                    left_symbol = (
                        left.element.symbol if left.element is not None else ""
                    )
                    right_symbol = (
                        right.element.symbol if right.element is not None else ""
                    )
                    if left_symbol in {"N", "O"} and right_symbol == "H":
                        donor_pairs.append((int(left.index), int(right.index)))
                    elif (
                        right_symbol in {"N", "O"}
                        and left_symbol == "H"
                    ):
                        donor_pairs.append((int(right.index), int(left.index)))
                acceptors = {
                    int(atom.index)
                    for atom in topology.atoms
                    if atom.element is not None
                    and atom.element.symbol in {"N", "O"}
                }
                ligand_donors = [
                    pair for pair in donor_pairs if pair[0] in ligand_all
                ]
                protein_donors = [
                    pair for pair in donor_pairs if pair[0] in protein_all
                ]
                ligand_acceptors = sorted(acceptors & ligand_all)
                protein_acceptors = sorted(acceptors & protein_all)
                water_oxygen = {
                    int(atom.index)
                    for atom in topology.atoms
                    if atom.residue.is_water
                    and atom.element is not None
                    and atom.element.symbol == "O"
                }
                water_donors: dict[int, list[tuple[int, int]]] = {}
                for donor, hydrogen in donor_pairs:
                    if donor in water_oxygen:
                        water_donors.setdefault(donor, []).append(
                            (donor, hydrogen)
                        )
                ligand_polar_atoms = sorted(
                    set(ligand_acceptors)
                    | {donor for donor, _ in ligand_donors}
                )
                close_waters_by_frame = (
                    md.compute_neighbors(
                        trajectory,
                        0.35,
                        query_indices=ligand_polar_atoms,
                        haystack_indices=sorted(water_oxygen),
                        periodic=trajectory.unitcell_lengths is not None,
                    )
                    if ligand_polar_atoms and water_oxygen
                    else [np.asarray([], dtype=int)] * trajectory.n_frames
                )
                candidate_cache: dict[
                    int,
                    tuple[np.ndarray, np.ndarray, list[str], list[str]],
                ] = {}
                for frame_index, close_waters in enumerate(
                    close_waters_by_frame
                ):
                    box_vectors = (
                        trajectory.unitcell_vectors[frame_index]
                        if trajectory.unitcell_vectors is not None
                        else None
                    )
                    for water in (int(value) for value in close_waters):
                        if water not in candidate_cache:
                            water_donor_pairs = water_donors.get(water, [])
                            ligand_triplets = [
                                (donor, hydrogen, acceptor)
                                for donor, hydrogen in water_donor_pairs
                                for acceptor in ligand_acceptors
                            ] + [
                                (donor, hydrogen, water)
                                for donor, hydrogen in ligand_donors
                            ]
                            protein_triplets = [
                                (donor, hydrogen, acceptor)
                                for donor, hydrogen in water_donor_pairs
                                for acceptor in protein_acceptors
                            ] + [
                                (donor, hydrogen, water)
                                for donor, hydrogen in protein_donors
                            ]
                            protein_labels = [
                                residue_label(
                                    topology.atom(acceptor).residue
                                )
                                for _donor, _hydrogen, acceptor
                                in protein_triplets[
                                    : len(water_donor_pairs)
                                    * len(protein_acceptors)
                                ]
                            ] + [
                                residue_label(topology.atom(donor).residue)
                                for donor, _hydrogen in protein_donors
                            ]
                            protein_scopes = [
                                (
                                    "BB"
                                    if str(topology.atom(acceptor).name).upper()
                                    in _PROTEIN_BACKBONE_ATOM_NAMES
                                    else "SC"
                                )
                                for _donor, _hydrogen, acceptor
                                in protein_triplets[
                                    : len(water_donor_pairs)
                                    * len(protein_acceptors)
                                ]
                            ] + [
                                (
                                    "BB"
                                    if str(topology.atom(donor).name).upper()
                                    in _PROTEIN_BACKBONE_ATOM_NAMES
                                    else "SC"
                                )
                                for donor, _hydrogen in protein_donors
                            ]
                            candidate_cache[water] = (
                                np.asarray(ligand_triplets, dtype=int).reshape(
                                    (-1, 3)
                                ),
                                np.asarray(protein_triplets, dtype=int).reshape(
                                    (-1, 3)
                                ),
                                protein_labels,
                                protein_scopes,
                            )
                        (
                            ligand_triplets,
                            protein_triplets,
                            protein_labels,
                            protein_scopes,
                        ) = candidate_cache[water]
                        if not np.any(
                            _wernet_nilsson_presence(
                                trajectory.xyz[frame_index],
                                ligand_triplets,
                                box_vectors,
                            )
                        ):
                            continue
                        protein_presence = _wernet_nilsson_presence(
                            trajectory.xyz[frame_index],
                            protein_triplets,
                            box_vectors,
                        )
                        present_indices = np.flatnonzero(protein_presence)
                        for label in {
                            protein_labels[index] for index in present_indices
                        }:
                            water_bridge_frames.setdefault(
                                label, set()
                            ).add(frame_index)
                        for label in {
                            protein_labels[index]
                            for index in present_indices
                            if protein_scopes[index] == "BB"
                        }:
                            water_bridge_backbone_frames.setdefault(
                                label, set()
                            ).add(frame_index)
                        for label in {
                            protein_labels[index]
                            for index in present_indices
                            if protein_scopes[index] == "SC"
                        }:
                            water_bridge_sidechain_frames.setdefault(
                                label, set()
                            ).add(frame_index)
            except Exception as exc:
                warnings_list.append(
                    f"Water-bridge occupancy could not be calculated: {exc}"
                )

            frame_count = trajectory.n_frames
            contact_rows = []
            for label, count in contact_counts.items():
                contact_backbone_count = contact_backbone_counts.get(label, 0)
                contact_sidechain_count = contact_sidechain_counts.get(label, 0)
                hbond_count = len(hydrogen_bond_frames.get(label, set()))
                hbond_backbone_count = len(
                    hydrogen_bond_backbone_frames.get(label, set())
                )
                hbond_sidechain_count = len(
                    hydrogen_bond_sidechain_frames.get(label, set())
                )
                hydrophobic_count = hydrophobic_counts.get(label, 0)
                hydrophobic_backbone_count = hydrophobic_backbone_counts.get(
                    label, 0
                )
                hydrophobic_sidechain_count = hydrophobic_sidechain_counts.get(
                    label, 0
                )
                water_bridge_count = len(
                    water_bridge_frames.get(label, set())
                )
                water_bridge_backbone_count = len(
                    water_bridge_backbone_frames.get(label, set())
                )
                water_bridge_sidechain_count = len(
                    water_bridge_sidechain_frames.get(label, set())
                )
                salt_bridge_count = salt_bridge_counts.get(label, 0)
                salt_bridge_backbone_count = salt_bridge_backbone_counts.get(
                    label, 0
                )
                salt_bridge_sidechain_count = salt_bridge_sidechain_counts.get(
                    label, 0
                )
                if (
                    count == 0
                    and hbond_count == 0
                    and water_bridge_count == 0
                    and salt_bridge_count == 0
                ):
                    continue
                distances = per_residue_distances[label]
                ligand_atom_counts = closest_ligand_atoms[label].copy()
                ligand_atom_counts.update(
                    hydrogen_bond_ligand_atoms.get(label, Counter())
                )
                ligand_atom_counts.update(
                    salt_bridge_ligand_atoms.get(label, Counter())
                )
                hbond_occupancy = hbond_count / frame_count
                hydrophobic_occupancy = hydrophobic_count / frame_count
                contact_rows.append(
                    {
                        "residue": label,
                        "contact_occupancy": round(count / frame_count, 4),
                        "contact_backbone_occupancy": round(
                            contact_backbone_count / frame_count, 4
                        ),
                        "contact_sidechain_occupancy": round(
                            contact_sidechain_count / frame_count, 4
                        ),
                        "hydrogen_bond_occupancy": round(hbond_occupancy, 4),
                        "hydrogen_bond_backbone_occupancy": round(
                            hbond_backbone_count / frame_count, 4
                        ),
                        "hydrogen_bond_sidechain_occupancy": round(
                            hbond_sidechain_count / frame_count, 4
                        ),
                        "hydrophobic_occupancy": round(
                            hydrophobic_occupancy, 4
                        ),
                        "hydrophobic_backbone_occupancy": round(
                            hydrophobic_backbone_count / frame_count, 4
                        ),
                        "hydrophobic_sidechain_occupancy": round(
                            hydrophobic_sidechain_count / frame_count, 4
                        ),
                        "water_bridge_occupancy": round(
                            water_bridge_count / frame_count, 4
                        ),
                        "water_bridge_backbone_occupancy": round(
                            water_bridge_backbone_count / frame_count, 4
                        ),
                        "water_bridge_sidechain_occupancy": round(
                            water_bridge_sidechain_count / frame_count, 4
                        ),
                        "salt_bridge_occupancy": round(
                            salt_bridge_count / frame_count, 4
                        ),
                        "salt_bridge_backbone_occupancy": round(
                            salt_bridge_backbone_count / frame_count, 4
                        ),
                        "salt_bridge_sidechain_occupancy": round(
                            salt_bridge_sidechain_count / frame_count, 4
                        ),
                        "binding_importance_score": round(
                            interaction_hotspot_score(
                                hydrogen_bond=hbond_occupancy,
                                salt_bridge=(
                                    salt_bridge_count / frame_count
                                ),
                                water_bridge=(
                                    water_bridge_count / frame_count
                                ),
                                hydrophobic=hydrophobic_occupancy,
                            ),
                            4,
                        ),
                        "binding_importance_backbone_score": round(
                            interaction_hotspot_score(
                                hydrogen_bond=(
                                    hbond_backbone_count / frame_count
                                ),
                                salt_bridge=(
                                    salt_bridge_backbone_count / frame_count
                                ),
                                water_bridge=(
                                    water_bridge_backbone_count / frame_count
                                ),
                                hydrophobic=(
                                    hydrophobic_backbone_count / frame_count
                                ),
                            ),
                            4,
                        ),
                        "binding_importance_sidechain_score": round(
                            interaction_hotspot_score(
                                hydrogen_bond=(
                                    hbond_sidechain_count / frame_count
                                ),
                                salt_bridge=(
                                    salt_bridge_sidechain_count / frame_count
                                ),
                                water_bridge=(
                                    water_bridge_sidechain_count / frame_count
                                ),
                                hydrophobic=(
                                    hydrophobic_sidechain_count / frame_count
                                ),
                            ),
                            4,
                        ),
                        "top_ligand_atom": (
                            ligand_atom_counts.most_common(1)[0][0]
                            if ligand_atom_counts
                            else ""
                        ),
                        "top_protein_atom": (
                            closest_protein_atoms[label].most_common(1)[0][0]
                            if closest_protein_atoms[label]
                            else ""
                        ),
                        "mean_minimum_distance_angstrom": round(
                            float(np.mean(distances)), 4
                        ),
                    }
                )
            contact_rows.sort(
                key=lambda row: (
                    float(row["contact_occupancy"]),
                    float(row["hydrogen_bond_occupancy"]),
                ),
                reverse=True,
            )
            distance_series = {
                str(row["residue"]): per_residue_distances[
                    str(row["residue"])
                ]
                for row in contact_rows
            }
            network_nodes = [
                {
                    "id": str(row["residue"]),
                    "contact_occupancy": row["contact_occupancy"],
                    "hydrogen_bond_occupancy": row[
                        "hydrogen_bond_occupancy"
                    ],
                    "hydrophobic_occupancy": row[
                        "hydrophobic_occupancy"
                    ],
                    "water_bridge_occupancy": row[
                        "water_bridge_occupancy"
                    ],
                    "salt_bridge_occupancy": row[
                        "salt_bridge_occupancy"
                    ],
                    "binding_importance_score": row[
                        "binding_importance_score"
                    ],
                    "top_ligand_atom": row["top_ligand_atom"],
                }
                for row in contact_rows
            ]
            network_edges = []
            for row in contact_rows:
                for interaction, field in (
                    ("hydrogen_bond", "hydrogen_bond_occupancy"),
                    ("hydrophobic", "hydrophobic_occupancy"),
                    ("water_bridge", "water_bridge_occupancy"),
                    ("salt_bridge", "salt_bridge_occupancy"),
                ):
                    occupancy = float(row[field])
                    if occupancy <= 0:
                        continue
                    network_edges.append(
                        {
                            "source": row["top_ligand_atom"] or ligand_resname,
                            "target": str(row["residue"]),
                            "weight": occupancy,
                            "interaction": interaction,
                        }
                    )

            interface_rin = {
                "applicable": False,
                "contact_cutoff_angstrom": round(
                    _CONTACT_CUTOFF_NM * 10.0, 2
                ),
                "interfaces": [],
                "residue_pairs": [],
                "nodes": [],
            }
            chain_labels = {
                residue.chain.index: (
                    str(getattr(residue.chain, "chain_id", "") or "").strip()
                    or str(residue.chain.index + 1)
                )
                for residue in protein_residues
            }
            if len(chain_labels) > 1:
                interface_rin["applicable"] = True
                residue_pairs = np.asarray(
                    [
                        (left.index, right.index)
                        for left_index, left in enumerate(protein_residues)
                        for right in protein_residues[left_index + 1 :]
                        if left.chain.index != right.chain.index
                    ],
                    dtype=int,
                )
                if residue_pairs.size:
                    try:
                        pair_distances, returned_pairs = md.compute_contacts(
                            trajectory,
                            contacts=residue_pairs,
                            scheme="closest-heavy",
                            periodic=True,
                        )
                        pair_contacts = pair_distances <= _CONTACT_CUTOFF_NM
                        pair_rows = []
                        node_weights: Counter[str] = Counter()
                        interface_frames: dict[
                            tuple[str, str], np.ndarray
                        ] = {}
                        interface_contact_counts: dict[
                            tuple[str, str], np.ndarray
                        ] = {}
                        for column, pair in enumerate(returned_pairs):
                            left = trajectory.topology.residue(int(pair[0]))
                            right = trajectory.topology.residue(int(pair[1]))
                            left_label = residue_label(left)
                            right_label = residue_label(right)
                            occupancy = float(
                                np.mean(pair_contacts[:, column])
                            )
                            if occupancy <= 0:
                                continue
                            left_chain = chain_labels[left.chain.index]
                            right_chain = chain_labels[right.chain.index]
                            interface_key = tuple(
                                sorted((left_chain, right_chain))
                            )
                            pair_rows.append(
                                {
                                    "chain_a": left_chain,
                                    "chain_b": right_chain,
                                    "residue_a": left_label,
                                    "residue_b": right_label,
                                    "contact_occupancy": round(
                                        occupancy, 4
                                    ),
                                    "mean_minimum_distance_angstrom": round(
                                        float(
                                            np.mean(
                                                pair_distances[:, column]
                                            )
                                            * 10.0
                                        ),
                                        4,
                                    ),
                                    "minimum_distance_angstrom": round(
                                        float(
                                            np.min(
                                                pair_distances[:, column]
                                            )
                                            * 10.0
                                        ),
                                        4,
                                    ),
                                }
                            )
                            node_weights[left_label] += occupancy
                            node_weights[right_label] += occupancy
                            interface_frames.setdefault(
                                interface_key,
                                np.zeros(
                                    trajectory.n_frames, dtype=bool
                                ),
                            )
                            interface_frames[interface_key] |= pair_contacts[
                                :, column
                            ]
                            interface_contact_counts.setdefault(
                                interface_key,
                                np.zeros(
                                    trajectory.n_frames, dtype=int
                                ),
                            )
                            interface_contact_counts[
                                interface_key
                            ] += pair_contacts[:, column].astype(int)
                        pair_rows.sort(
                            key=lambda row: float(
                                row["contact_occupancy"]
                            ),
                            reverse=True,
                        )
                        interface_rin["residue_pairs"] = pair_rows[:100]
                        interface_rin["nodes"] = [
                            {
                                "residue": label,
                                "weighted_degree": round(weight, 4),
                            }
                            for label, weight in node_weights.most_common()
                        ]
                        interface_rin["interfaces"] = [
                            {
                                "chain_a": key[0],
                                "chain_b": key[1],
                                "contact_fraction": round(
                                    float(np.mean(frames)), 4
                                ),
                                "mean_contacts_per_frame": round(
                                    float(
                                        np.mean(
                                            interface_contact_counts[key]
                                        )
                                    ),
                                    4,
                                ),
                            }
                            for key, frames in sorted(
                                interface_frames.items()
                            )
                        ]
                    except Exception as exc:
                        warnings_list.append(
                            f"Interface RIN could not be calculated: {exc}"
                        )

            return {
                "time_ps": time_ps,
                "analysis_window": {
                    "start_ps": float(analysis_start_ps),
                    "discarded_frames": discarded_frames,
                    "analyzed_frames": trajectory.n_frames,
                    "trajectory_stride": stride,
                },
                "rmsf": {
                    "residues": rmsf_labels,
                    "ca_rmsf_angstrom": rmsf_values,
                },
                "ligand_rmsf": {
                    "atoms": ligand_rmsf_labels,
                    "rmsf_angstrom": ligand_rmsf_values,
                },
                "radius_of_gyration": {
                    "protein_angstrom": protein_rg,
                    "ligand_angstrom": ligand_rg,
                    "complex_angstrom": complex_rg,
                },
                "secondary_structure": secondary_structure,
                "pocket": {
                    "ligand_centroid_displacement_angstrom": centroid_displacements,
                    "minimum_protein_distance_angstrom": minimum_distances,
                    "reference_site_retained": retained,
                    "retained_fraction": round(
                        sum(retained) / len(retained), 4
                    )
                    if retained
                    else None,
                    "site_displacement_cutoff_angstrom": round(
                        _SITE_DISPLACEMENT_CUTOFF_NM * 10.0, 2
                    ),
                    "contact_cutoff_angstrom": round(
                        _CONTACT_CUTOFF_NM * 10.0, 2
                    ),
                },
                "contacts": {
                    "residues": contact_rows,
                    "distance_series": distance_series,
                    "interaction_hotspot_score": {
                        "description": (
                            "Weighted interaction occupancy; not a binding-"
                            "energy estimate"
                        ),
                        "weights": dict(INTERACTION_HOTSPOT_WEIGHTS),
                    },
                },
                "water_bridges": {
                    "residues": [
                        {
                            "residue": row["residue"],
                            "occupancy": row["water_bridge_occupancy"],
                        }
                        for row in contact_rows
                        if float(row["water_bridge_occupancy"]) > 0
                    ],
                    "method": (
                        "Wernet-Nilsson two-sided hydrogen-bond geometry"
                    ),
                },
                "salt_bridges": {
                    "applicable": bool(charged_ligand_positions),
                    "ligand_charged_atoms": [
                        {
                            "atom": str(
                                trajectory.topology.atom(
                                    int(ligand_heavy[position])
                                ).name
                            ),
                            "formal_charge": charge,
                        }
                        for position, charge
                        in charged_ligand_positions.items()
                    ],
                    "residues": [
                        {
                            "residue": row["residue"],
                            "occupancy": row["salt_bridge_occupancy"],
                        }
                        for row in contact_rows
                        if float(row["salt_bridge_occupancy"]) > 0
                    ],
                    "distance_cutoff_angstrom": round(
                        _SALT_BRIDGE_CUTOFF_NM * 10.0,
                        2,
                    ),
                },
                "interaction_network": {
                    "nodes": network_nodes,
                    "edges": network_edges,
                },
                "interface_rin": interface_rin,
                "residue_numbering": {
                    "scheme": (
                        "source_author"
                        if mapped_by_residue_index
                        else "trajectory_internal"
                    ),
                    "source_run_id": str(
                        (residue_mapping or {}).get("source_run_id") or ""
                    ),
                    "source_label": str(
                        (residue_mapping or {}).get("source_label") or ""
                    ),
                },
                "warnings": warnings_list,
            }
        except ImportError:
            return {
                **empty,
                "warnings": ["MDTraj or NumPy is unavailable"],
            }
        except Exception as exc:
            logger.warning(
                "[ANALYTICS] Structural-dynamics analysis failed: %s",
                exc,
                exc_info=True,
            )
            return {
                **empty,
                "warnings": [
                    f"Structural-dynamics analysis failed: {exc}"
                ],
            }

    def compute_production_rmsd_window(
        self,
        *,
        topology_pdb: str,
        production_traj: str,
        ligand_resname: str,
        production_report_interval: int,
        dt_ps: float,
        analysis_start_ps: float,
    ) -> dict[str, Any]:
        import mdtraj as md
        import numpy as np

        with md.open(production_traj) as trajectory_file:
            source_frame_count = len(trajectory_file)
        if source_frame_count == 0:
            raise ValueError("Production trajectory has no frames")
        stride = max(1, source_frame_count // _RMSD_MAX_FRAMES)
        trajectory = md.load(
            production_traj,
            top=topology_pdb,
            stride=stride,
        )
        time_per_frame_ps = (
            max(1, int(production_report_interval))
            * float(dt_ps)
            * stride
        )
        reference_index = int(
            np.ceil(float(analysis_start_ps) / time_per_frame_ps)
        )
        if reference_index >= trajectory.n_frames:
            raise ValueError(
                "No production frames remain after the requested RMSD start"
            )

        warnings_list: list[str] = []
        if trajectory.unitcell_lengths is not None:
            try:
                protein_indices = trajectory.topology.select("protein")
                protein_set = set(int(value) for value in protein_indices)
                molecules = trajectory.topology.find_molecules()
                anchors = [
                    sorted(list(molecule), key=lambda atom: atom.index)
                    for molecule in molecules
                    if any(atom.index in protein_set for atom in molecule)
                ]
                if anchors:
                    trajectory.image_molecules(
                        inplace=True,
                        anchor_molecules=anchors,
                    )
                else:
                    trajectory.image_molecules(inplace=True)
            except Exception as exc:
                warnings_list.append(f"PBC imaging failed: {exc}")

        backbone_indices = trajectory.topology.select(
            "protein and name CA"
        )
        if len(backbone_indices) == 0:
            backbone_indices = trajectory.topology.select(
                "protein and backbone"
            )
        if len(backbone_indices) == 0:
            raise ValueError("Protein backbone atoms were not available")
        ligand_indices = trajectory.topology.select(
            f"resname {ligand_resname} and not element H"
        )
        if len(ligand_indices) == 0:
            ligand_indices = trajectory.topology.select(
                "not (protein or water or resname HOH or resname WAT "
                "or resname SOL or resname NA or resname CL or resname K "
                "or resname MG or resname CA or resname ZN) "
                "and not element H"
            )
        if len(ligand_indices) == 0:
            raise ValueError("Ligand heavy atoms were not available")

        reference = trajectory[reference_index : reference_index + 1]
        analyzed = trajectory[reference_index:]
        analyzed.superpose(
            reference,
            0,
            atom_indices=backbone_indices,
        )
        backbone_rmsd = [
            round(float(value) * 10.0, 4)
            for value in md.rmsd(
                analyzed,
                reference,
                0,
                atom_indices=backbone_indices,
            )
        ]
        ligand_difference = (
            analyzed.xyz[:, ligand_indices, :]
            - reference.xyz[0, ligand_indices, :]
        )
        ligand_rmsd = [
            round(float(value) * 10.0, 4)
            for value in np.sqrt(
                np.mean(
                    np.sum(ligand_difference**2, axis=2),
                    axis=1,
                )
            )
        ]
        reference_time_ps = reference_index * time_per_frame_ps
        time_ps = [
            round(reference_time_ps + index * time_per_frame_ps, 3)
            for index in range(analyzed.n_frames)
        ]
        local = {
            "time_ps": time_ps,
            "backbone_rmsd_angstrom": backbone_rmsd,
            "ligand_rmsd_angstrom": ligand_rmsd,
        }
        return {
            **local,
            "phase_boundaries": [
                {
                    "phase": "production",
                    "start_ps": reference_time_ps,
                    "end_ps": time_ps[-1],
                }
            ],
            "per_phase_local": {"production": local},
            "analysis_window": {
                "start_ps": reference_time_ps,
                "end_ps": time_ps[-1],
                "discarded_frames": reference_index,
                "discarded_source_frames": reference_index * stride,
                "analyzed_frames": analyzed.n_frames,
                "trajectory_stride": stride,
                "reference": "first retained production frame",
                "reference_time_ps": reference_time_ps,
                "reference_loaded_frame_index": reference_index,
                "reference_source_frame_index": reference_index * stride,
                "reference_coordinates": "production trajectory frame",
                "topology_coordinates_used_as_reference": False,
            },
            "warnings": warnings_list,
        }

    # ── KPI evaluator ──────────────────────────────────────────────────────────

    def _evaluate_kpis(
        self,
        thermo: dict[str, Any],
        rmsd: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Evaluate pass/warn/fail status for each KPI using absolute tolerances.

        Plateau definition: std(last 20% of series) < threshold.
        RMSD threshold: mean(last 20% of series) < pass_threshold.

        Returns:
            {
                "energy_stable": bool | None,
                "density_converged": bool | None,
                "backbone_rmsd_status": "pass" | "warn" | "fail" | None,
                "ligand_rmsd_status": "pass" | "warn" | "fail" | None,
                "overall_pass": bool,
                "warnings": [...],
            }
        """
        import math

        def last20_std(series: list[float]) -> float | None:
            """Std of last 20% of series, ignoring NaN."""
            if not series:
                return None
            n = max(1, len(series) // 5)
            tail = [v for v in series[-n:] if not math.isnan(v)]
            if len(tail) < 2:
                return None
            mean = sum(tail) / len(tail)
            variance = sum((v - mean) ** 2 for v in tail) / len(tail)
            return variance ** 0.5

        def last20_mean(series: list[float]) -> float | None:
            """Mean of last 20% of series, ignoring NaN."""
            if not series:
                return None
            n = max(1, len(series) // 5)
            tail = [v for v in series[-n:] if not math.isnan(v)]
            if not tail:
                return None
            return sum(tail) / len(tail)

        warnings: list[str] = list(rmsd.get("warnings", []))
        all_none = True

        # Energy stability
        energy_stable = None
        energy_std = last20_std(thermo.get("potential_energy_kjmol", []))
        if energy_std is not None:
            all_none = False
            energy_stable = energy_std < _ENERGY_STD_THRESHOLD_KJ
            if not energy_stable:
                warnings.append(
                    f"Energy not stable: std={energy_std:.0f} kJ/mol "
                    f"(threshold {_ENERGY_STD_THRESHOLD_KJ:.0f})"
                )

        # Density convergence
        density_converged = None
        density_std = last20_std(thermo.get("density_gcm3", []))
        density_mean = last20_mean(thermo.get("density_gcm3", []))
        if density_std is not None and density_mean is not None:
            all_none = False
            density_converged = density_std < _DENSITY_STD_THRESHOLD_GCM3
            if not density_converged:
                warnings.append(
                    f"Density not converged: std={density_std:.4f} g/cm³ "
                    f"(threshold {_DENSITY_STD_THRESHOLD_GCM3:.2f})"
                )
            elif abs(density_mean - _DENSITY_TARGET_GCM3) > 0.1:
                warnings.append(
                    f"Density converged but far from target: "
                    f"mean={density_mean:.3f} g/cm³ (expected ~{_DENSITY_TARGET_GCM3:.1f})"
                )

        def phase_series(series_key: str, phase_name: str) -> list[float]:
            """Extract a phase-specific RMSD series from global RMSD arrays via phase boundaries."""
            values = rmsd.get(series_key, []) or []
            times = rmsd.get("time_ps", []) or []
            boundaries = rmsd.get("phase_boundaries", []) or []
            if not values or not times or not boundaries or len(values) != len(times):
                return []
            phase = next((p for p in boundaries if str(p.get("phase", "")).lower() == phase_name.lower()), None)
            if not phase:
                return []
            start = float(phase.get("start_ps", 0.0))
            end = float(phase.get("end_ps", start))
            selected = [
                v for t, v in zip(times, values)
                if (t >= start) and (t <= end)
            ]
            return selected

        # Prefer production-only RMSD for KPI status; fallback to full combined series.
        backbone_series_for_kpi = phase_series("backbone_rmsd_angstrom", "production") or rmsd.get("backbone_rmsd_angstrom", [])
        ligand_series_for_kpi = phase_series("ligand_rmsd_angstrom", "production") or rmsd.get("ligand_rmsd_angstrom", [])
        rmsd_kpi_source = "production" if phase_series("backbone_rmsd_angstrom", "production") else "combined"

        # Backbone RMSD
        backbone_status = None
        bb_mean = last20_mean(backbone_series_for_kpi)
        if bb_mean is not None:
            all_none = False
            if bb_mean < _BACKBONE_RMSD_PASS_A:
                backbone_status = "pass"
            elif bb_mean < _BACKBONE_RMSD_WARN_A:
                backbone_status = "warn"
                warnings.append(
                    f"Backbone RMSD elevated: {bb_mean:.2f}Å "
                    f"(pass <{_BACKBONE_RMSD_PASS_A}Å)"
                )
            else:
                backbone_status = "fail"
                warnings.append(
                    f"Backbone RMSD too high: {bb_mean:.2f}Å "
                    f"(fail >{_BACKBONE_RMSD_WARN_A}Å) — protein may be unstable"
                )

        # Ligand RMSD
        ligand_status = None
        lig_mean = last20_mean(ligand_series_for_kpi)
        if lig_mean is not None:
            all_none = False
            if lig_mean < _LIGAND_RMSD_PASS_A:
                ligand_status = "pass"
            elif lig_mean < _LIGAND_RMSD_WARN_A:
                ligand_status = "warn"
                warnings.append(
                    f"Ligand RMSD elevated: {lig_mean:.2f}Å "
                    f"(pass <{_LIGAND_RMSD_PASS_A}Å) — check binding pose"
                )
            else:
                ligand_status = "fail"
                warnings.append(
                    f"Ligand RMSD too high: {lig_mean:.2f}Å "
                    f"(fail >{_LIGAND_RMSD_WARN_A}Å) — ligand may have dissociated"
                )

        # Overall pass: all evaluated KPIs must pass
        statuses = [
            energy_stable,
            density_converged,
            backbone_status == "pass" if backbone_status else None,
            ligand_status == "pass" if ligand_status else None,
        ]
        evaluated = [s for s in statuses if s is not None]
        overall_pass = all(evaluated) if evaluated else False

        return {
            "energy_stable": energy_stable,
            "density_converged": density_converged,
            "backbone_rmsd_status": backbone_status,
            "ligand_rmsd_status": ligand_status,
            "overall_pass": overall_pass,
            "warnings": warnings,
            "backbone_rmsd_pass_a": _BACKBONE_RMSD_PASS_A,
            "ligand_rmsd_pass_a": _LIGAND_RMSD_PASS_A,
            "rmsd_kpi_source": rmsd_kpi_source,
        }
