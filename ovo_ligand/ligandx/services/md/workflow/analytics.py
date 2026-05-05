"""
Equilibration analytics module.

Post-processes MD equilibration output files to produce quantitative KPIs
and time-series data for frontend visualization. Runs after all simulation
stages complete; never modifies the simulation itself.

Data flow:
  EquilibrationAnalytics.compute()
    ├─ _parse_log()       → thermodynamic time series from StateDataReporter TSV
    ├─ _compute_rmsd()    → backbone + ligand RMSD from NPT DCD trajectory
    └─ _evaluate_kpis()   → pass/warn/fail summary with absolute tolerances
"""

import logging
import os
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

# Report interval × integration timestep (ps) used in equilibration_runner.py
# report_interval=1000, dt=0.004 ps → 4 ps per data point
_DEFAULT_REPORT_INTERVAL = 1000
_DT_PS = 0.004


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
            
            # Use provided dt_ps or default
            if dt_ps is None:
                dt_ps = _DT_PS
            
            rmsd = self._compute_rmsd(
                topology_pdb, nvt_traj, npt_traj, production_traj,
                ligand_resname, nvt_steps, npt_steps, production_steps, dt_ps,
                nvt_report_interval, npt_report_interval, production_report_interval
            )
            kpi = self._evaluate_kpis(thermo, rmsd)

            logger.info(
                f"[ANALYTICS] Complete — overall_pass={kpi.get('overall_pass')}, "
                f"warnings={kpi.get('warnings')}"
            )
            return {
                "thermodynamics": thermo,
                "rmsd": rmsd,
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

            col_step = col_pe = col_temp = col_vol = col_den = -1
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

                        # Skip rows where all values are None/NaN
                        if all(v is None for v in [pe, temp, vol, den]):
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

                # 5. Calculate time axis for this phase
                # Each kept frame is spaced by stride * report_interval integration steps.
                phase_duration_ps = steps * dt_ps
                time_per_frame_ps = stride * report_interval * dt_ps
                time_ps_phase = [
                    round(current_time_offset_ps + (i * time_per_frame_ps), 3)
                    for i in range(n_frames)
                ]

                # Record phase boundary (nominal span from configured step counts)
                phase_start_ps = current_time_offset_ps
                phase_end_ps = current_time_offset_ps + phase_duration_ps
                phase_boundaries.append({
                    "phase": phase_name,
                    "start_ps": round(phase_start_ps, 1),
                    "end_ps": round(phase_end_ps, 1)
                })

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
                "warnings": warnings_list,
            }

        except ImportError:
            logger.warning("[ANALYTICS] MDTraj not available — skipping RMSD")
            return {**empty, "warnings": ["MDTraj not available"]}
        except Exception as e:
            logger.warning(f"[ANALYTICS] RMSD computation failed: {e}", exc_info=True)
            return {**empty, "warnings": [f"RMSD computation failed: {str(e)}"]}

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

        # Backbone RMSD
        backbone_status = None
        bb_mean = last20_mean(rmsd.get("backbone_rmsd_angstrom", []))
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
        lig_mean = last20_mean(rmsd.get("ligand_rmsd_angstrom", []))
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
        }
