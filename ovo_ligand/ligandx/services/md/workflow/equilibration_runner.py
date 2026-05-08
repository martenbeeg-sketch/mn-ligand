"""
Equilibration runner module for MD optimization.

Handles the complete equilibration protocol: minimization, thermal heating,
NVT, NPT equilibration, and optional production MD.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List, Set, Tuple

logger = logging.getLogger(__name__)


def emit_progress(progress: int, status: str, completed_stages: List[str]) -> None:
    """
    Emit a progress update that can be parsed by the streaming endpoint.

    Writes a special JSON line to stderr that the router can parse.
    Format: MD_PROGRESS:{"progress": N, "status": "...", "completed_stages": [...]}
    """
    progress_data = {
        "progress": progress,
        "status": status,
        "completed_stages": completed_stages
    }
    # Write to stderr with a special prefix so the router can identify it
    sys.stderr.write(f"MD_PROGRESS:{json.dumps(progress_data)}\n")
    sys.stderr.flush()


class EquilibrationRunner:
    """Runs the complete MD equilibration protocol."""

    # Standard residue names for protein identification
    PROTEIN_RESIDUES = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
        'THR', 'TRP', 'TYR', 'VAL',
        # Non-standard
        'MSE', 'HYP', 'PCA', 'SEP', 'TPO', 'CSO', 'PTR', 'KCX',
        'HIE', 'HID', 'HIP', 'CYX', 'ACE', 'NME',
    }

    WATER_RESIDUES = {'HOH', 'WAT', 'H2O', 'TIP', 'TIP3', 'TIP4'}
    ION_RESIDUES = {'NA', 'CL', 'MG', 'K', 'CA', 'ZN', 'FE', 'MN'}
    BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}
    # Relative scaling factors for ligand restraints across NPT release stages.
    # Stage 1 keeps full restraint, last stage turns restraints off.
    NPT_RESTRAINT_RELEASE_SCALES = (1.0, 0.5, 0.2, 0.05, 0.0)

    @staticmethod
    def _parse_scales(value: str, default: List[float]) -> List[float]:
        try:
            vals = [float(x.strip()) for x in str(value).split(",") if x.strip()]
            return vals if vals else list(default)
        except Exception:
            return list(default)

    @staticmethod
    def _set_context_param_if_present(context, name: str, value: float) -> bool:
        try:
            params = set(context.getParameters().keys())
            if name in params:
                context.setParameter(name, float(value))
                return True
        except Exception:
            return False
        return False

    @staticmethod
    def _list_active_restraint_params(context) -> Dict[str, Optional[float]]:
        out = {"k_prot": None, "k_lig": None, "k_plan": None, "k": None, "kp": None}
        try:
            params = set(context.getParameters().keys())
            for key in list(out.keys()):
                if key in params:
                    out[key] = float(context.getParameter(key))
        except Exception:
            pass
        return out

    @staticmethod
    def _compute_stage_ranges(nvt_steps: int, npt_steps: int, production_steps: int) -> Dict[str, Tuple[int, int]]:
        """
        Compute progress percentage ranges for each stage based on step counts.

        Early fixed-cost stages (preparation, minimisation, heating, NVT) are
        allocated a small reserved band at the front of the bar.  The remaining
        percentage is split between NPT and production in proportion to their
        step counts, so that a 10 ns production run with a 1 ns NPT equilibration
        naturally occupies ~90 % of the progress bar rather than the old fixed 5 %.

        Returns a dict mapping stage name → (start_pct, end_pct).
        """
        # Fixed band for early (fast) stages: 0–15 %
        EARLY_END = 15
        # Allocation within the early band (must sum to EARLY_END)
        PREP_END  =  5   # preparation:    0 –  5 %
        MIN_END   =  9   # minimisation:   5 –  9 %
        HEAT_END  = 12   # thermal heating: 9 – 12 %
        NVT_END   = EARLY_END  # NVT:      12 – 15 %

        if production_steps > 0:
            # NPT and production share 15–100 % proportionally
            total_late = npt_steps + production_steps
            npt_share  = int((npt_steps / total_late) * (100 - EARLY_END))
            npt_end    = EARLY_END + npt_share
            return {
                'preparation': (0,        PREP_END),
                'minimization': (PREP_END, MIN_END),
                'heating':      (MIN_END,  HEAT_END),
                'nvt':          (HEAT_END, NVT_END),
                'npt':          (NVT_END,  npt_end),
                'production':   (npt_end,  100),
            }
        else:
            # No production: reserve first 35 % for early stages, NPT fills the rest
            return {
                'preparation': (0,   5),
                'minimization': (5,  10),
                'heating':      (10, 15),
                'nvt':          (15, 35),
                'npt':          (35, 100),
            }

    def __init__(self, output_dir: str = "data/md_outputs"):
        """
        Initialize equilibration runner.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _cleanup_file_handler(self, file_handler: Optional[logging.FileHandler]) -> None:
        """Remove and close the file handler from the root logger."""
        if file_handler:
            try:
                file_handler.flush()
                file_handler.close()
                root_logger = logging.getLogger()
                root_logger.removeHandler(file_handler)
            except Exception as e:
                logger.warning(f"Error cleaning up file handler: {e}")

    def _minimize_on_cpu(self, simulation, unit, maxIterations: int = 5000) -> None:
        """
        Minimize energy using CPU platform as fallback for CUDA NaN issues.

        Creates a temporary CPU context, runs minimization, and transfers
        the minimized positions back to the original context.
        Enforces PBC before transfer to handle non-orthogonal boxes.
        """
        import openmm

        # Get positions WITH PBC enforcement to ensure atoms are in primary cell
        state = simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True
        )
        positions = state.getPositions()
        box_vectors = state.getPeriodicBoxVectors()

        logger.info("Creating temporary CPU context for minimization...")

        # Create CPU context with the same system
        cpu_platform = openmm.Platform.getPlatformByName('CPU')
        cpu_integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds
        )
        cpu_context = openmm.Context(simulation.system, cpu_integrator, cpu_platform)
        cpu_context.setPeriodicBoxVectors(*box_vectors)
        cpu_context.setPositions(positions)

        # Minimize on CPU
        logger.info("Running minimization on CPU platform...")
        openmm.LocalEnergyMinimizer.minimize(
            cpu_context, 10 * unit.kilojoule_per_mole / unit.nanometer, maxIterations
        )

        # Get minimized positions with PBC enforcement
        min_state = cpu_context.getState(
            getPositions=True, getEnergy=True, enforcePeriodicBox=True
        )
        min_positions = min_state.getPositions()
        min_energy = min_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        logger.info(f"CPU minimization completed, energy: {min_energy:.2f} kJ/mol")

        # Transfer minimized positions back to main context
        simulation.context.setPositions(min_positions)

        # Clean up CPU context
        del cpu_context, cpu_integrator

        logger.info("Minimized positions transferred back to main context")

    def _resolve_clashes(
        self, simulation, unit,
        target_max_force: float = 10000.0,
        max_steps: int = 1000
    ) -> None:
        """
        Resolve steric clashes using capped steepest descent in-place.

        Solvation (especially with dodecahedron boxes) can place water
        molecules too close together, producing forces of ~10^5 kJ/mol/nm.
        OpenMM's L-BFGS minimizer has no step-size cap, so these huge forces
        cause the first step to diverge to NaN.

        This method runs manual capped steepest descent directly on the
        simulation's own context (CUDA or CPU): each step evaluates forces,
        moves atoms along force vectors with displacement capped at 0.002 nm,
        then applies constraints to maintain bond geometry.

        Operating on the original context avoids issues with creating a
        separate CPU context that can fail with dodecahedron periodic boxes
        due to PME/SHAKE incompatibilities when sharing the System object.

        Only physical forces drive the relaxation (no artificial restraints).

        Args:
            simulation: OpenMM Simulation
            unit: OpenMM unit module
            target_max_force: Stop when max force drops below this (kJ/mol/nm)
            max_steps: Maximum number of steepest descent steps
        """
        import numpy as np

        # Capped steepest descent: max 0.002 nm displacement per step
        max_disp_nm = 0.002

        for step in range(max_steps):
            state = simulation.context.getState(
                getPositions=True, getForces=True, getEnergy=True
            )
            pos_arr = state.getPositions(asNumpy=True).value_in_unit(
                unit.nanometer
            )
            force_arr = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            energy_val = state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            )

            if np.any(np.isnan(pos_arr)) or np.any(np.isnan(force_arr)):
                logger.warning(
                    f"NaN detected at clash resolution step {step}, stopping"
                )
                break

            force_mags = np.linalg.norm(force_arr, axis=1)
            max_force_val = float(np.max(force_mags))

            if step % 50 == 0:
                logger.info(
                    f"Clash resolution step {step}: "
                    f"energy={energy_val:.1f} kJ/mol, "
                    f"max_force={max_force_val:.0f} kJ/mol/nm"
                )

            if max_force_val < target_max_force:
                logger.info(
                    f"Clashes resolved at step {step}: "
                    f"max_force={max_force_val:.0f} < "
                    f"{target_max_force:.0f} kJ/mol/nm"
                )
                break

            # Move each atom along its force direction, capping displacement
            scale = np.minimum(
                max_disp_nm / (force_mags + 1e-10),
                max_disp_nm / target_max_force
            )
            displacement = force_arr * scale[:, np.newaxis]
            new_pos = pos_arr + displacement

            simulation.context.setPositions(new_pos * unit.nanometer)
            simulation.context.applyConstraints(1e-5)
            simulation.context.computeVirtualSites()

        logger.info(
            f"[COMPLETE] Clash resolution: energy={energy_val:.1f} kJ/mol "
            f"after {min(step + 1, max_steps)} steps"
        )

    def _gentle_minimization(
        self, simulation, unit, maxIterations: int = 5000
    ) -> None:
        """
        Last-resort minimization using capped steepest descent followed by
        L-BFGS.

        When L-BFGS fails with NaN (because steric clashes produce forces
        of ~10^5 kJ/mol/nm that cause the uncapped first step to diverge),
        this method runs manual steepest descent where the maximum per-atom
        displacement is capped at 0.01 nm per step. This is equivalent to
        GROMACS's steepest descent with emstep=0.01.

        After reducing the maximum force below a threshold, switches to
        L-BFGS for final convergence.
        """
        import openmm
        import numpy as np

        logger.info("Attempting capped steepest descent minimization...")

        # Get current positions with PBC enforcement
        state = simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True
        )
        positions = state.getPositions()
        box_vectors = state.getPeriodicBoxVectors()

        # Temporarily disable barostat during steepest descent
        original_baro_freq = None
        for i in range(simulation.system.getNumForces()):
            force = simulation.system.getForce(i)
            if isinstance(force, openmm.MonteCarloBarostat):
                original_baro_freq = force.getFrequency()
                force.setFrequency(0)
                break

        try:
            # Create CPU context for steepest descent
            cpu_platform = openmm.Platform.getPlatformByName('CPU')
            sd_integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
            sd_context = openmm.Context(
                simulation.system, sd_integrator, cpu_platform
            )
            sd_context.setPeriodicBoxVectors(*box_vectors)
            sd_context.setPositions(positions)

            # Capped steepest descent parameters
            max_disp_nm = 0.01  # Maximum per-atom displacement per step (nm)
            sd_steps = 1000
            lbfgs_threshold = 1000.0  # Switch to L-BFGS when max_force < this

            prev_energy = None
            for step in range(sd_steps):
                sd_state = sd_context.getState(
                    getPositions=True, getForces=True, getEnergy=True
                )
                pos_q = sd_state.getPositions(asNumpy=True)
                force_q = sd_state.getForces(asNumpy=True)
                energy_val = sd_state.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )

                pos_arr = pos_q.value_in_unit(unit.nanometer)
                force_arr = force_q.value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                )

                if np.any(np.isnan(pos_arr)) or np.any(np.isnan(force_arr)):
                    logger.warning(
                        f"NaN detected at steepest descent step {step}, stopping"
                    )
                    break

                force_mags = np.linalg.norm(force_arr, axis=1)
                max_force = float(np.max(force_mags))

                if step % 100 == 0:
                    logger.info(
                        f"Steepest descent step {step}: "
                        f"energy={energy_val:.2f} kJ/mol, "
                        f"max_force={max_force:.1f} kJ/mol/nm"
                    )

                if max_force < lbfgs_threshold:
                    logger.info(
                        f"Max force {max_force:.1f} below threshold "
                        f"{lbfgs_threshold}, switching to L-BFGS"
                    )
                    break

                # Move each atom along its force direction, capping displacement
                scale = np.minimum(
                    max_disp_nm / (force_mags + 1e-10),
                    max_disp_nm / lbfgs_threshold
                )
                displacement = force_arr * scale[:, np.newaxis]
                new_pos = pos_arr + displacement

                sd_context.setPositions(new_pos * unit.nanometer)

                # Apply bond constraints to maintain geometry
                sd_context.applyConstraints(1e-5)
                sd_context.computeVirtualSites()

                prev_energy = energy_val

            # Get energy after steepest descent
            sd_final = sd_context.getState(getEnergy=True)
            sd_energy = sd_final.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            )
            logger.info(
                f"Steepest descent finished: energy={sd_energy:.2f} kJ/mol "
                f"after {min(step + 1, sd_steps)} steps"
            )

            # Try L-BFGS on the relaxed context
            logger.info("Attempting L-BFGS on steepest-descent-relaxed system...")
            try:
                openmm.LocalEnergyMinimizer.minimize(
                    sd_context,
                    10 * unit.kilojoule_per_mole / unit.nanometer,
                    maxIterations
                )
                lbfgs_state = sd_context.getState(getEnergy=True)
                lbfgs_energy = lbfgs_state.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
                logger.info(
                    f"L-BFGS succeeded after steepest descent: "
                    f"energy={lbfgs_energy:.2f} kJ/mol"
                )
            except Exception as lbfgs_err:
                logger.warning(
                    f"L-BFGS failed after steepest descent ({lbfgs_err}), "
                    "using steepest-descent positions"
                )

            # Transfer final positions back to main context
            final_state = sd_context.getState(
                getPositions=True, getEnergy=True, enforcePeriodicBox=True
            )
            simulation.context.setPositions(final_state.getPositions())
            final_energy = final_state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            )
            logger.info(
                f"Gentle minimization completed: "
                f"final energy={final_energy:.2f} kJ/mol"
            )

            del sd_context, sd_integrator

        finally:
            # Re-enable barostat in the System
            if original_baro_freq is not None:
                for i in range(simulation.system.getNumForces()):
                    force = simulation.system.getForce(i)
                    if isinstance(force, openmm.MonteCarloBarostat):
                        force.setFrequency(original_baro_freq)
                        break

    # ── Main protocol ──────────────────────────────────────────────────

    def run_equilibration_protocol(
        self,
        simulation,
        system_id: str = "complex",
        pause_at_minimized: bool = False,
        minimization_only: bool = False,
        skip_minimization: bool = False,
        nvt_steps: int = 25000,
        npt_steps: int = 175000,
        heating_steps_per_stage: int = 2500,
        heating_start_temperature: float = 50.0,
        heating_stages: int = 6,
        report_interval: int = 1000,
        production_steps: int = 0,
        production_report_interval: int = 2500,
        minimization_max_iterations: int = 5000,
        minimization_tolerance_kjmol_nm: float = 10.0,
        npt_restraint_release_scales_csv: str = "1.0,0.5,0.2,0.05,0.0",
        npt_release_enabled: bool = True,
        protein_npt_release_scales_csv: str = "1.0,0.5,0.1,0.01,0.0",
        planarity_npt_release_scales_csv: str = "1.0,0.5,0.2,0.05,0.0",
        allow_restrained_production: bool = False,
        force_unrestrained_production: bool = True,
        resume_from_checkpoint_path: str | None = None,
        resume_state_xml_path: str | None = None,
        resume_system_xml_path: str | None = None,
        resume_integrator_xml_path: str | None = None,
        production_only_from_prepared: bool = False,
        temperature: float = 300.0,
        pressure: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run the complete staged equilibration protocol.

        Stages:
        1. Energy minimization
        2. Thermal heating (50K → 300K in 50K increments)
        3. NVT equilibration (constant volume, 300K)
        4. NPT equilibration (constant pressure, 1 bar)
        5. Production MD (optional, if production_steps > 0)

        Args:
            simulation: OpenMM Simulation object
            system_id: System identifier for output files
            pause_at_minimized: Whether to pause after minimization
            minimization_only: Whether to stop after minimization
            skip_minimization: Whether to skip minimization (e.g. when resuming)
            nvt_steps: Number of NVT steps
            npt_steps: Number of NPT steps
            heating_steps_per_stage: Number of heating MD steps per temperature stage
            report_interval: Reporting interval
            production_steps: Number of production MD steps (0 = skip)
            production_report_interval: DCD frame save interval for production
            temperature: Simulation temperature in Kelvin
            pressure: Simulation pressure in bar

        Returns:
            Dict with equilibration results and file paths
        """
        from openmm.app import StateDataReporter, DCDReporter
        from openmm import unit
        from ..utils.pdb_utils import write_pdb_file
        import numpy as np

        results = {}

        # Setup console log capture
        console_log_path = os.path.join(self.output_dir, f"{system_id}_console.log")
        file_handler = None

        try:
            # Add file handler to capture console output
            file_handler = logging.FileHandler(console_log_path, mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            # Add handler to root logger to capture all MD-related logs
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)

            logger.info("Setting up reporters for equilibration...")

            # Setup file paths
            log_path = os.path.join(self.output_dir, f"{system_id}_equilibration.log")
            nvt_traj_path = os.path.join(self.output_dir, f"{system_id}_nvt_equilibration.dcd")
            npt_traj_path = os.path.join(self.output_dir, f"{system_id}_npt_equilibration.dcd")
            minimized_pdb_path = os.path.join(self.output_dir, f"{system_id}_minimized.pdb")
            nvt_pdb_path = os.path.join(self.output_dir, f"{system_id}_nvt_final.pdb")
            npt_pdb_path = os.path.join(self.output_dir, f"{system_id}_npt_final.pdb")
            npt_checkpoint_path = os.path.join(self.output_dir, f"{system_id}_npt_final.chk")
            npt_state_xml_path = os.path.join(self.output_dir, f"{system_id}_npt_final_state.xml")
            npt_system_xml_path = os.path.join(self.output_dir, f"{system_id}_npt_system.xml")
            npt_integrator_xml_path = os.path.join(self.output_dir, f"{system_id}_npt_integrator.xml")
            production_traj_path = os.path.join(self.output_dir, f"{system_id}_production.dcd")
            production_pdb_path = os.path.join(self.output_dir, f"{system_id}_production_final.pdb")
            production_log_path = os.path.join(self.output_dir, f"{system_id}_production.log")

            # Clear any existing reporters
            simulation.reporters.clear()

            # Add state data reporter
            simulation.reporters.append(
                StateDataReporter(
                    log_path, report_interval,
                    step=True, potentialEnergy=True, kineticEnergy=True,
                    totalEnergy=True, temperature=True, volume=True,
                    density=True, speed=True, separator='\t'
                )
            )

            # Initialize energy tracking variables
            initial_energy_val = None
            final_energy_val = None

            # Compute dynamic progress ranges based on step counts
            ranges = self._compute_stage_ranges(nvt_steps, npt_steps, production_steps)

            # Emit preparation complete (system was built before this point)
            emit_progress(ranges['preparation'][1], "System preparation complete", ["preparation"])

            resume_from_checkpoint = bool(resume_from_checkpoint_path and str(resume_from_checkpoint_path).strip())
            if resume_from_checkpoint:
                bundle_system_xml = str(resume_system_xml_path or "").strip()
                bundle_integrator_xml = str(resume_integrator_xml_path or "").strip()
                bundle_used = False
                if bundle_system_xml and bundle_integrator_xml and os.path.exists(bundle_system_xml) and os.path.exists(bundle_integrator_xml):
                    try:
                        from openmm import XmlSerializer
                        from openmm.app import Simulation
                        with open(bundle_system_xml, "r") as sf:
                            restored_system = XmlSerializer.deserialize(sf.read())
                        with open(bundle_integrator_xml, "r") as inf:
                            restored_integrator = XmlSerializer.deserialize(inf.read())
                        platform = simulation.context.getPlatform()
                        simulation = Simulation(simulation.topology, restored_system, restored_integrator, platform)
                        bundle_used = True
                        logger.info("Rebuilt simulation context from serialized OpenMM bundle.")
                    except Exception as bundle_exc:
                        logger.warning("OpenMM bundle restore failed; using current simulation object. reason=%s", bundle_exc)

                checkpoint_path = str(resume_from_checkpoint_path).strip()
                if not os.path.exists(checkpoint_path):
                    raise RuntimeError(f"Checkpoint not found for production resume: {checkpoint_path}")
                logger.info("Resuming simulation from checkpoint: %s", checkpoint_path)
                resume_mode = "checkpoint"
                try:
                    simulation.loadCheckpoint(checkpoint_path)
                    logger.info("Checkpoint loaded successfully.")
                except Exception as exc:
                    logger.warning("Checkpoint load failed: %s", exc)
                    state_xml_path = str(resume_state_xml_path or "").strip()
                    if state_xml_path and os.path.exists(state_xml_path):
                        logger.info("Falling back to state XML resume: %s", state_xml_path)
                        try:
                            simulation.loadState(state_xml_path)
                            resume_mode = "state_xml"
                            logger.info("State XML loaded successfully.")
                        except Exception as state_exc:
                            if bool(production_only_from_prepared):
                                logger.warning(
                                    "State XML resume also failed (%s). Falling back to production-only coordinate continuation.",
                                    state_exc,
                                )
                                resume_from_checkpoint = False
                            else:
                                raise
                    elif bool(production_only_from_prepared):
                        logger.warning(
                            "No usable state XML provided. Falling back to production-only coordinate continuation."
                        )
                        resume_from_checkpoint = False
                    else:
                        raise

                if resume_from_checkpoint:
                    completed_stages = ["preparation", "minimization", "thermal_heating", "nvt", "npt"]
                    equilibration_stats = {
                        "energy_minimization": {
                            "initial_energy": None,
                            "final_energy": None,
                            "energy_change": None,
                        },
                        "thermal_heating": {"status": "skipped_resume_checkpoint"},
                        "nvt_equilibration": {"status": "skipped_resume_checkpoint"},
                        "npt_equilibration": {
                            "status": "loaded_from_checkpoint" if resume_mode == "checkpoint" else "loaded_from_state_xml",
                            "checkpoint_path": checkpoint_path if resume_mode == "checkpoint" else None,
                            "state_xml_path": (str(resume_state_xml_path).strip() if resume_mode == "state_xml" else None),
                        },
                    }
                    output_files = {
                        "equilibration_log": log_path,
                        "console_log": console_log_path,
                        "npt_checkpoint": checkpoint_path if resume_mode == "checkpoint" else None,
                        "npt_state_xml": (str(resume_state_xml_path).strip() if resume_mode == "state_xml" else None),
                        "npt_system_xml": bundle_system_xml or None,
                        "npt_integrator_xml": bundle_integrator_xml or None,
                    }

                    if production_steps > 0:
                        prod_progress_start = ranges['production'][0]
                        prod_progress_end = 100
                        start_msg = "Starting production MD from checkpoint..." if resume_mode == "checkpoint" else "Starting production MD from state XML..."
                        emit_progress(prod_progress_start, start_msg, completed_stages)
                        prod_result = self._run_production(
                            simulation, production_steps, production_report_interval,
                            production_traj_path, production_log_path,
                            production_pdb_path, unit,
                            prod_progress_start, prod_progress_end,
                            completed_stages,
                            allow_restrained_production=allow_restrained_production,
                            force_unrestrained_production=force_unrestrained_production,
                        )
                        equilibration_stats["production"] = prod_result
                        output_files["production_trajectory"] = production_traj_path
                        output_files["production_pdb"] = production_pdb_path
                        output_files["production_log"] = production_log_path
                        completed_stages.append("production")
                        emit_progress(100, "MD production completed successfully", completed_stages)

                    results.update({
                        "status": "success",
                        "output_files": output_files,
                        "equilibration_stats": equilibration_stats,
                        "restart_resume": {
                            "requested_checkpoint_path": checkpoint_path,
                            "requested_state_xml_path": (str(resume_state_xml_path).strip() if resume_state_xml_path else None),
                            "requested_system_xml_path": (bundle_system_xml or None),
                            "requested_integrator_xml_path": (bundle_integrator_xml or None),
                            "bundle_context_rebuild_used": bool(bundle_used),
                            "resume_mode": (
                                "bundle_checkpoint" if (bundle_used and resume_mode == "checkpoint")
                                else "bundle_state_xml" if (bundle_used and resume_mode == "state_xml")
                                else "checkpoint" if resume_mode == "checkpoint"
                                else "state_xml"
                            ),
                        },
                        "restraint_protocol": {
                            "production_unrestrained": bool(equilibration_stats.get("production", {}).get("production_unrestrained", True)),
                            "force_unrestrained_production": bool(force_unrestrained_production),
                            "allow_restrained_production": bool(allow_restrained_production),
                            "npt_release_stages": [],
                            "warnings": equilibration_stats.get("production", {}).get("warnings", []),
                            "resumed_from_checkpoint": bool(resume_mode == "checkpoint"),
                            "resumed_from_state_xml": bool(resume_mode == "state_xml"),
                        },
                    })
                    self._cleanup_file_handler(file_handler)
                    return results

            if bool(production_only_from_prepared):
                logger.info("Production-only mode enabled: skipping minimization/heating/NVT/NPT.")
                try:
                    simulation.minimizeEnergy(maxIterations=500)
                except Exception as exc:
                    logger.warning("Production-only pre-stabilization minimization failed: %s", exc)
                try:
                    simulation.context.setVelocitiesToTemperature(float(temperature) * unit.kelvin)
                except Exception:
                    logger.warning("Could not initialize velocities to target temperature before production-only run.")

                completed_stages = ["preparation", "npt"]
                equilibration_stats = {
                    "energy_minimization": {
                        "initial_energy": None,
                        "final_energy": None,
                        "energy_change": None,
                    },
                    "thermal_heating": {"status": "skipped_production_only"},
                    "nvt_equilibration": {"status": "skipped_production_only"},
                    "npt_equilibration": {"status": "skipped_production_only"},
                }
                output_files = {
                    "equilibration_log": log_path,
                    "console_log": console_log_path,
                }

                if production_steps > 0:
                    prod_progress_start = ranges['production'][0]
                    prod_progress_end = 100
                    emit_progress(prod_progress_start, "Starting production-only MD...", completed_stages)
                    prod_result = self._run_production(
                        simulation, production_steps, production_report_interval,
                        production_traj_path, production_log_path,
                        production_pdb_path, unit,
                        prod_progress_start, prod_progress_end,
                        completed_stages,
                        allow_restrained_production=allow_restrained_production,
                        force_unrestrained_production=force_unrestrained_production,
                    )
                    equilibration_stats["production"] = prod_result
                    output_files["production_trajectory"] = production_traj_path
                    output_files["production_pdb"] = production_pdb_path
                    output_files["production_log"] = production_log_path
                    completed_stages.append("production")
                    emit_progress(100, "Production-only MD completed successfully", completed_stages)

                results.update({
                    "status": "success",
                    "output_files": output_files,
                    "equilibration_stats": equilibration_stats,
                    "restart_resume": {
                        "requested_checkpoint_path": (str(resume_from_checkpoint_path).strip() if resume_from_checkpoint_path else None),
                        "requested_state_xml_path": (str(resume_state_xml_path).strip() if resume_state_xml_path else None),
                        "requested_system_xml_path": (str(resume_system_xml_path).strip() if resume_system_xml_path else None),
                        "requested_integrator_xml_path": (str(resume_integrator_xml_path).strip() if resume_integrator_xml_path else None),
                        "bundle_context_rebuild_used": False,
                        "resume_mode": "coordinate_continuation",
                    },
                    "restraint_protocol": {
                        "production_unrestrained": bool(equilibration_stats.get("production", {}).get("production_unrestrained", True)),
                        "force_unrestrained_production": bool(force_unrestrained_production),
                        "allow_restrained_production": bool(allow_restrained_production),
                        "npt_release_stages": [],
                        "warnings": equilibration_stats.get("production", {}).get("warnings", []),
                        "production_only_from_prepared": True,
                    },
                })
                self._cleanup_file_handler(file_handler)
                return results

            # Stage 1: Energy Minimization
            if not skip_minimization:
                min_result = self._run_minimization(
                    simulation, minimized_pdb_path, unit, temperature,
                    max_iterations=minimization_max_iterations,
                    tolerance_kjmol_nm=minimization_tolerance_kjmol_nm,
                    progress_start=ranges['minimization'][0],
                    progress_end=ranges['minimization'][1],
                )
                initial_energy_val = min_result.get('initial_energy')
                final_energy_val = min_result.get('final_energy')

                if minimization_only:
                    logger.info("Minimization only requested - stopping workflow.")
                    self._cleanup_file_handler(file_handler)
                    return {
                        "status": "success",
                        "message": "Minimization completed successfully",
                        "minimization_only": True,
                        "final_energy": final_energy_val,
                        "equilibration_stats": {
                            "minimized_energy": final_energy_val
                        },
                        "output_files": {
                            "minimized_pdb": minimized_pdb_path,
                            "equilibration_log": log_path,
                            "console_log": console_log_path
                        }
                    }

                if pause_at_minimized:
                    logger.info("Pause at minimized structure requested.")
                    self._cleanup_file_handler(file_handler)
                    return {
                        "status": "minimized_ready",
                        "message": "Minimization completed. Paused for inspection.",
                        "final_energy": final_energy_val,
                        "output_files": {
                            "minimized_pdb": minimized_pdb_path,
                            "equilibration_log": log_path,
                            "console_log": console_log_path
                        }
                    }
            else:
                logger.info("Skipping minimization (resuming from minimized state)")

            # Stage 2: Thermal Heating (gradual temperature ramping)
            heating_result = self._run_thermal_heating(
                simulation, unit,
                target_temperature=temperature,
                start_temperature=heating_start_temperature,
                n_stages=heating_stages,
                steps_per_stage=heating_steps_per_stage,
                progress_start=ranges['heating'][0],
                progress_end=ranges['heating'][1],
            )

            # Stage 3: NVT Equilibration
            nvt_result = self._run_nvt(
                simulation, nvt_steps, report_interval,
                nvt_traj_path, log_path, nvt_pdb_path, unit,
                temperature,
                progress_start=ranges['nvt'][0],
                progress_end=ranges['nvt'][1],
            )

            # Stage 4: NPT Equilibration
            npt_result = self._run_npt(
                simulation, npt_steps, report_interval,
                npt_traj_path, log_path, npt_pdb_path, unit,
                release_scales_csv=npt_restraint_release_scales_csv,
                release_enabled=bool(npt_release_enabled),
                protein_release_scales_csv=protein_npt_release_scales_csv,
                planarity_release_scales_csv=planarity_npt_release_scales_csv,
                progress_start=ranges['npt'][0],
                progress_end=ranges['npt'][1],
            )

            # Build completed stages list
            completed_stages = ["preparation", "minimization", "thermal_heating", "nvt", "npt"]

            # Build equilibration statistics
            equilibration_stats = {
                "energy_minimization": {
                    "initial_energy": initial_energy_val,
                    "final_energy": final_energy_val,
                    "energy_change": (final_energy_val - initial_energy_val)
                                    if (initial_energy_val and final_energy_val) else None
                },
                "thermal_heating": heating_result,
                "nvt_equilibration": nvt_result,
                "npt_equilibration": npt_result
            }

            output_files = {
                "equilibration_log": log_path,
                "console_log": console_log_path,
                "nvt_trajectory": nvt_traj_path,
                "npt_trajectory": npt_traj_path,
                "minimized_pdb": minimized_pdb_path,
                "nvt_pdb": nvt_pdb_path,
                "npt_pdb": npt_pdb_path
            }

            # Persist exact post-NPT restart artifacts for downstream production runs.
            simulation.saveCheckpoint(npt_checkpoint_path)
            simulation.saveState(npt_state_xml_path)
            try:
                from openmm import XmlSerializer
                with open(npt_system_xml_path, "w") as sf:
                    sf.write(XmlSerializer.serialize(simulation.system))
                with open(npt_integrator_xml_path, "w") as inf:
                    inf.write(XmlSerializer.serialize(simulation.integrator))
            except Exception as exc:
                logger.warning("Failed to write OpenMM serialized bundle (system/integrator): %s", exc)
            output_files["npt_checkpoint"] = npt_checkpoint_path
            output_files["npt_state_xml"] = npt_state_xml_path
            output_files["npt_system_xml"] = npt_system_xml_path
            output_files["npt_integrator_xml"] = npt_integrator_xml_path

            # Stage 4: Production MD (optional)
            if production_steps > 0:
                prod_progress_start = ranges['production'][0]
                prod_progress_end = 100
                emit_progress(prod_progress_start, "Starting production MD...", completed_stages)

                prod_result = self._run_production(
                    simulation, production_steps, production_report_interval,
                    production_traj_path, production_log_path,
                    production_pdb_path, unit,
                    prod_progress_start, prod_progress_end,
                    completed_stages,
                    allow_restrained_production=allow_restrained_production,
                    force_unrestrained_production=force_unrestrained_production,
                )

                equilibration_stats["production"] = prod_result
                output_files["production_trajectory"] = production_traj_path
                output_files["production_pdb"] = production_pdb_path
                output_files["production_log"] = production_log_path
                completed_stages.append("production")

            logger.info("[COMPLETE] Complete equilibration protocol finished successfully")
            emit_progress(100, "MD optimization completed successfully", completed_stages)

            results.update({
                "status": "success",
                "output_files": output_files,
                "equilibration_stats": equilibration_stats,
                "restart_resume": {
                    "requested_checkpoint_path": None,
                    "requested_state_xml_path": None,
                    "requested_system_xml_path": None,
                    "requested_integrator_xml_path": None,
                    "bundle_context_rebuild_used": False,
                    "resume_mode": "fresh_protocol",
                },
                "restraint_protocol": {
                    "production_unrestrained": bool(equilibration_stats.get("production", {}).get("production_unrestrained", True)),
                    "force_unrestrained_production": bool(force_unrestrained_production),
                    "allow_restrained_production": bool(allow_restrained_production),
                    "npt_release_stages": npt_result.get("release_stage_details", []),
                    "warnings": equilibration_stats.get("production", {}).get("warnings", []),
                },
            })

            self._cleanup_file_handler(file_handler)
            return results

        except Exception as e:
            import traceback
            logger.error(f"Equilibration protocol failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._cleanup_file_handler(file_handler)
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "output_files": {
                    "console_log": console_log_path
                }
            }

    def _run_minimization(
        self, simulation, output_path: str, unit, temperature: float = 300.0,
        max_iterations: int = 5000,
        tolerance_kjmol_nm: float = 10.0,
        progress_start: int = 5, progress_end: int = 9,
    ) -> Dict[str, Any]:
        """
        Run energy minimization.

        Args:
            simulation: OpenMM Simulation
            output_path: Path for minimized PDB
            unit: OpenMM unit module
            temperature: Target temperature in K

        Returns:
            Dict with initial and final energy values
        """
        from ..utils.pdb_utils import write_pdb_file
        import numpy as np

        logger.info("=== STAGE 1: ENERGY MINIMIZATION ===")
        emit_progress(progress_start, "Running energy minimization...", ["preparation"])

        # Get initial energy and check forces (diagnostic)
        initial_energy_val = None
        max_force = 0.0
        try:
            diag_state = simulation.context.getState(
                getPositions=True, getForces=True, getEnergy=True
            )
            initial_energy = diag_state.getPotentialEnergy()
            initial_energy_val = initial_energy.value_in_unit(unit.kilojoule_per_mole)
            diag_forces = diag_state.getForces(asNumpy=True)
            forces_kj = diag_forces.value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            force_mags = np.linalg.norm(forces_kj, axis=1)
            max_force = float(np.max(force_mags))
            max_force_idx = int(np.argmax(force_mags))

            if np.isnan(initial_energy_val) or np.isinf(initial_energy_val):
                logger.warning(f"Initial energy is NaN/Inf: {initial_energy_val}")
            else:
                logger.info(
                    f"Initial potential energy: {initial_energy_val:.2f} kJ/mol, "
                    f"max_force={max_force:.1f} kJ/mol/nm at atom {max_force_idx}"
                )

            if max_force > 10000:
                # Log top clashing atoms for debugging
                top5 = np.argsort(force_mags)[-5:][::-1]
                all_atoms = list(simulation.topology.atoms())
                pos_nm = diag_state.getPositions(asNumpy=True).value_in_unit(
                    unit.nanometer
                )
                for idx in top5:
                    a = all_atoms[idx]
                    logger.info(
                        f"  Clash: atom {idx} {a.residue.name}:{a.name} "
                        f"(res {a.residue.id}), "
                        f"|F|={force_mags[idx]:.0f} kJ/mol/nm"
                    )
        except Exception as e:
            logger.warning(f"Could not get initial energy/forces: {e}")

        # Pre-step: resolve steric clashes with capped steepest descent.
        # OpenMM's L-BFGS minimizer has no step-size limit. When solvation
        # places water molecules too close (forces ~10^5 kJ/mol/nm), the
        # uncapped first step diverges to NaN. We run capped steepest descent
        # to bring forces into a reasonable range before L-BFGS.
        clash_threshold = 10000.0  # kJ/mol/nm -- above this, L-BFGS diverges
        if max_force > clash_threshold:
            logger.info(
                f"Max force {max_force:.0f} exceeds {clash_threshold:.0f} "
                f"kJ/mol/nm -- running capped steepest descent to resolve "
                f"clashes before L-BFGS..."
            )
            self._resolve_clashes(simulation, unit, target_max_force=clash_threshold)

        # Run L-BFGS minimization
        logger.info("Performing L-BFGS energy minimization...")
        try:
            simulation.minimizeEnergy(
                maxIterations=int(max_iterations),
                tolerance=float(tolerance_kjmol_nm) * unit.kilojoule_per_mole / unit.nanometer
            )
            logger.info("L-BFGS minimization succeeded")
        except Exception as err:
            if "NaN" not in str(err):
                raise
            # Fallback: try on CPU
            logger.warning(f"L-BFGS failed ({err}), trying CPU fallback...")
            # Save positions before CPU attempt (minimizeEnergy corrupts on NaN)
            saved = simulation.context.getState(getPositions=True)
            simulation.context.setPositions(saved.getPositions())
            try:
                self._minimize_on_cpu(simulation, unit, maxIterations=int(max_iterations))
            except Exception as cpu_err:
                logger.warning(
                    f"CPU L-BFGS also failed ({cpu_err}), "
                    "using clash-resolved positions"
                )

        # Get final energy
        final_energy_val = None
        try:
            final_state = simulation.context.getState(getEnergy=True)
            final_energy = final_state.getPotentialEnergy()
            final_energy_val = final_energy.value_in_unit(unit.kilojoule_per_mole)
            logger.info(f"Minimized potential energy: {final_energy_val:.2f} kJ/mol")

            if initial_energy_val is not None and not np.isnan(initial_energy_val):
                energy_change = final_energy_val - initial_energy_val
                logger.info(f"Energy change: {energy_change:.2f} kJ/mol")
        except Exception as e:
            logger.warning(f"Could not get final energy: {e}")

        # Save minimized structure
        write_pdb_file(
            simulation.topology,
            simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
            output_path,
            keep_ids=True
        )

        logger.info("[COMPLETE] Energy minimization completed")
        emit_progress(progress_end, "Energy minimization completed", ["preparation", "minimization"])

        return {
            'initial_energy': initial_energy_val,
            'final_energy': final_energy_val
        }

    def _run_thermal_heating(
        self, simulation, unit,
        target_temperature: float = 300.0,
        start_temperature: float = 50.0,
        n_stages: int = 6,
        steps_per_stage: int = 2500,
        progress_start: int = 9,
        progress_end: int = 12,
    ) -> Dict[str, Any]:
        """
        Gradually heat the system from low temperature to target temperature.

        Uses a separate 1 fs timestep integrator for heating because the main
        4 fs HMR integrator is too aggressive for initial dynamics from
        minimized structures. After heating completes, positions are transferred
        back to the main simulation context.

        CRITICAL: Each temperature stage runs energy minimization BEFORE dynamics
        to ensure positions satisfy HBonds constraints. This prevents NaN
        coordinates from constraint violations when velocities are initialized.

        Heating stages: 50K → 100K → 150K → 200K → 250K → 300K
        Each stage: Minimization + 2,500 steps = 2.5 ps (at 1 fs timestep)
        Total: ~15 ps dynamics + ~30-60s minimization time

        Args:
            simulation: OpenMM Simulation (main, 4 fs HMR)
            unit: OpenMM unit module
            target_temperature: Final target temperature (K)
            steps_per_stage: Number of MD steps per heating stage
        """
        import openmm

        logger.info("=== STAGE 2: THERMAL HEATING ===")
        emit_progress(progress_start, "Starting thermal heating...", ["preparation", "minimization"])

        # Temperature stages (evenly spaced from start to target)
        if n_stages < 1:
            n_stages = 1
        if n_stages == 1:
            temp_stages = [float(target_temperature)]
        else:
            delta = (float(target_temperature) - float(start_temperature)) / float(n_stages - 1)
            temp_stages = [float(start_temperature) + i * delta for i in range(n_stages - 1)]
            temp_stages.append(float(target_temperature))
        num_stages = len(temp_stages)

        # Get current positions and box vectors from main simulation
        state = simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True
        )
        positions = state.getPositions()
        box_vectors = state.getPeriodicBoxVectors()

        # Create a conservative 1 fs integrator for heating
        # The main 4 fs HMR integrator is too aggressive for initial dynamics
        logger.info("Creating 1 fs heating integrator (4 fs HMR too aggressive for initial dynamics)")
        heating_integrator = openmm.LangevinMiddleIntegrator(
            temp_stages[0] * unit.kelvin,  # Start at first temperature
            1.0 / unit.picosecond,         # Friction
            0.001 * unit.picoseconds       # 1 fs timestep (conservative)
        )

        # Create a Simulation wrapper for heating (creates its own context)
        # This allows us to call minimizeEnergy() and have proper context management
        platform = simulation.context.getPlatform()
        logger.info(f"Creating heating simulation on {platform.getName()} platform")
        from openmm.app import Simulation as OpenMMSimulation
        heating_simulation = OpenMMSimulation(
            simulation.topology, simulation.system,
            heating_integrator, platform
        )

        # Set initial positions and box vectors
        heating_simulation.context.setPeriodicBoxVectors(*box_vectors)
        heating_simulation.context.setPositions(positions)

        # Apply constraints and compute virtual sites before dynamics
        logger.info("Applying constraints and computing virtual sites...")
        heating_simulation.context.applyConstraints(1e-5)
        heating_simulation.context.computeVirtualSites()

        total_steps = 0
        for i, temp in enumerate(temp_stages):
            stage_num = i + 1
            logger.info(f"Heating stage {stage_num}/{num_stages}: {temp:.0f} K")

            # Update integrator temperature
            heating_simulation.integrator.setTemperature(temp * unit.kelvin)

            # CRITICAL: Minimize at this temperature BEFORE initializing velocities
            # This ensures positions satisfy HBonds constraints before dynamics
            logger.info(f"Minimizing at {temp:.0f} K before dynamics...")
            try:
                start_energy = heating_simulation.context.getState(getEnergy=True).getPotentialEnergy()
                heating_simulation.minimizeEnergy(maxIterations=1000, tolerance=10.0)
                end_energy = heating_simulation.context.getState(getEnergy=True).getPotentialEnergy()
                energy_change = (end_energy - start_energy).value_in_unit(unit.kilojoule_per_mole)
                logger.info(f"Minimized at {temp:.0f} K: dE={energy_change:.1f} kJ/mol")

                # Check for NaN coordinates after minimization
                positions_check = heating_simulation.context.getState(getPositions=True).getPositions()
                if any(
                    pos.x != pos.x or pos.y != pos.y or pos.z != pos.z
                    for pos in positions_check
                ):
                    logger.error(f"NaN detected in positions after minimization at {temp:.0f} K!")
                    raise ValueError("NaN coordinates after minimization")

            except Exception as min_err:
                logger.warning(f"Minimization failed at {temp:.0f} K: {min_err}")
                logger.warning("Continuing anyway - system may still be stable")

            # Re-initialize velocities at this temperature (AFTER minimization)
            heating_simulation.context.setVelocitiesToTemperature(temp * unit.kelvin)

            # Run dynamics at this temperature in small chunks to detect NaN early
            chunk_size = 500
            for chunk_start in range(0, steps_per_stage, chunk_size):
                chunk_steps = min(chunk_size, steps_per_stage - chunk_start)
                heating_simulation.step(chunk_steps)
                total_steps += chunk_steps

                # Check for NaN after each chunk
                state_check = heating_simulation.context.getState(getPositions=True, getEnergy=True)
                positions_check = state_check.getPositions()
                if any(
                    pos.x != pos.x or pos.y != pos.y or pos.z != pos.z
                    for pos in positions_check
                ):
                    energy_check = state_check.getPotentialEnergy()
                    logger.error(
                        f"NaN detected during dynamics at {temp:.0f} K "
                        f"after {total_steps} steps! Energy: {energy_check}"
                    )
                    raise ValueError(f"NaN coordinates during heating at {temp:.0f} K")

            # Log successful completion of this temperature stage
            stage_energy = heating_simulation.context.getState(getEnergy=True).getPotentialEnergy()
            logger.info(
                f"Completed {temp:.0f} K stage: "
                f"{steps_per_stage} steps, E={stage_energy.value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol"
            )

            # Calculate progress within heating stage
            heating_progress = progress_start + int((stage_num / num_stages) * (progress_end - progress_start))
            emit_progress(
                heating_progress,
                f"Heating: {temp:.0f} K ({stage_num}/{num_stages})",
                ["preparation", "minimization"]
            )

        # Get final state from heating simulation
        final_state = heating_simulation.context.getState(
            getPositions=True, getEnergy=True,
            enforcePeriodicBox=True
        )
        final_positions = final_state.getPositions()
        final_box_vectors = final_state.getPeriodicBoxVectors()
        final_energy = final_state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

        # Transfer heated positions to main simulation.
        # CRITICAL ordering: box vectors first, then positions, then applyConstraints.
        #
        # setPositions() does NOT enforce constraints. LangevinMiddleIntegrator uses
        # the LFMiddle (leapfrog) discretisation: setVelocitiesToTemperature() internally
        # calls calcForcesAndEnergy() for a half-step velocity correction before assigning
        # velocities. If positions are not on the constraint manifold when that force
        # evaluation fires, the resulting forces are garbage and the first simulation.step()
        # immediately produces NaN coordinates.
        #
        # DO NOT transfer raw velocities from the 1 fs heating context. Those velocities
        # carry the leapfrog half-step offset for dt=0.001 ps and are physically
        # incompatible with the 4 fs main integrator. setVelocitiesToTemperature() in
        # _run_nvt will generate correctly calibrated velocities for the 4 fs context.
        logger.info("Transferring heated state to main simulation context")
        if final_box_vectors is not None:
            simulation.context.setPeriodicBoxVectors(*final_box_vectors)
        simulation.context.setPositions(final_positions)
        simulation.context.applyConstraints(1e-6)
        simulation.context.computeVirtualSites()

        # Update main integrator temperature to target
        simulation.integrator.setTemperature(target_temperature * unit.kelvin)

        # Clean up heating simulation (releases context and integrator)
        del heating_simulation

        logger.info(
            f"[COMPLETE] Thermal heating completed: {target_temperature:.0f} K, "
            f"energy={final_energy:.1f} kJ/mol"
        )
        emit_progress(progress_end, "Thermal heating completed", ["preparation", "minimization", "thermal_heating"])

        return {
            "stages": num_stages,
            "total_steps": total_steps,
            "duration_ps": total_steps * 0.001,  # 1 fs timestep
            "final_temperature_K": target_temperature,
            "final_energy_kJ": final_energy
        }

    def _run_nvt(
        self, simulation, steps: int, report_interval: int,
        traj_path: str, log_path: str, pdb_path: str, unit,
        temperature: float = 300.0,
        progress_start: int = 12,
        progress_end: int = 15,
    ) -> Dict[str, Any]:
        """
        Run NVT equilibration (constant volume, constant temperature).

        Equilibrates the system at the target temperature with fixed volume.
        """
        from openmm.app import StateDataReporter, DCDReporter
        from ..utils.pdb_utils import write_pdb_file

        logger.info("=== STAGE 3: NVT EQUILIBRATION ===")
        emit_progress(progress_start, "Starting NVT equilibration...", ["preparation", "minimization", "thermal_heating"])

        # Clear reporters and add NVT reporters
        simulation.reporters.clear()
        simulation.reporters.append(DCDReporter(traj_path, report_interval))
        simulation.reporters.append(
            StateDataReporter(
                log_path, report_interval,
                step=True, potentialEnergy=True, kineticEnergy=True,
                totalEnergy=True, temperature=True, volume=True,
                density=True, speed=True, separator='\t'
            )
        )

        # Set velocities to target temperature
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

        # Get initial temperature
        initial_temp = simulation.integrator.getTemperature().value_in_unit(unit.kelvin)
        logger.info(f"Initial temperature: {initial_temp:.1f} K")

        # Run NVT with progress reporting
        logger.info(f"Running NVT equilibration for {steps} steps...")
        chunk_size = max(steps // 10, 1000)  # Report progress ~10 times
        completed_steps = 0

        while completed_steps < steps:
            steps_to_run = min(chunk_size, steps - completed_steps)
            simulation.step(steps_to_run)
            completed_steps += steps_to_run
            nvt_progress = progress_start + int((completed_steps / steps) * (progress_end - progress_start))
            progress_pct = (completed_steps / steps) * 100
            emit_progress(nvt_progress, f"NVT equilibration: {progress_pct:.0f}%", ["preparation", "minimization", "thermal_heating"])
            logger.info(f"NVT progress: {completed_steps}/{steps} steps ({progress_pct:.1f}%)")

        # Get final temperature
        final_temp = simulation.integrator.getTemperature().value_in_unit(unit.kelvin)
        logger.info(f"Final temperature: {final_temp:.1f} K")

        # Save NVT structure
        write_pdb_file(
            simulation.topology,
            simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
            pdb_path,
            keep_ids=True
        )

        logger.info("[COMPLETE] NVT equilibration completed")
        emit_progress(progress_end, "NVT equilibration completed", ["preparation", "minimization", "thermal_heating", "nvt"])

        return {
            "steps": steps,
            "duration_ps": steps * 0.004,
            "initial_temperature_K": initial_temp,
            "final_temperature_K": final_temp,
        }

    def _run_npt(
        self, simulation, steps: int, report_interval: int,
        traj_path: str, log_path: str, pdb_path: str, unit,
        release_scales_csv: str = "1.0,0.5,0.2,0.05,0.0",
        protein_release_scales_csv: str = "1.0,0.5,0.1,0.01,0.0",
        planarity_release_scales_csv: str = "1.0,0.5,0.2,0.05,0.0",
        release_enabled: bool = True,
        progress_start: int = 15,
        progress_end: int = 95,
    ) -> Dict[str, Any]:
        """
        Run NPT equilibration (constant pressure, constant temperature).

        Equilibrates the system at target temperature and pressure,
        allowing the box volume to adjust to the correct density.
        """
        from openmm.app import StateDataReporter, DCDReporter
        from ..utils.pdb_utils import write_pdb_file

        logger.info("=== STAGE 4: NPT EQUILIBRATION ===")
        emit_progress(progress_start, "Starting NPT equilibration...", ["preparation", "minimization", "thermal_heating", "nvt"])

        # Clear reporters and add NPT reporters
        simulation.reporters.clear()
        simulation.reporters.append(DCDReporter(traj_path, report_interval))
        simulation.reporters.append(
            StateDataReporter(
                log_path, report_interval,
                step=True, potentialEnergy=True, kineticEnergy=True,
                totalEnergy=True, temperature=True, volume=True,
                density=True, speed=True, separator='\t'
            )
        )

        # Get initial volume
        initial_state = simulation.context.getState(getEnergy=True)
        initial_volume = initial_state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
        logger.info(f"Initial volume: {initial_volume:.2f} nm^3")

        # Run NPT equilibration with staged restraint release.
        logger.info(f"Running NPT equilibration for {steps} steps ({steps * 0.004 / 1000:.1f} ns)")
        context = simulation.context
        release_scales = list(self.NPT_RESTRAINT_RELEASE_SCALES)
        ligand_scales = self._parse_scales(release_scales_csv, list(self.NPT_RESTRAINT_RELEASE_SCALES))
        protein_scales = self._parse_scales(protein_release_scales_csv, [1.0, 0.5, 0.1, 0.01, 0.0])
        planarity_scales = self._parse_scales(planarity_release_scales_csv, list(self.NPT_RESTRAINT_RELEASE_SCALES))
        if not release_enabled:
            ligand_scales = [1.0]
            protein_scales = [1.0]
            planarity_scales = [1.0]
        release_scales = ligand_scales
        try:
            param_names = set(context.getParameters().keys())
        except Exception:
            param_names = set()
        has_ligand_k = ("k_lig" in param_names) or ("k" in param_names)
        has_planarity_k = ("k_plan" in param_names) or ("kp" in param_names)
        has_protein_k = "k_prot" in param_names
        initial_k_lig = context.getParameter("k_lig") if "k_lig" in param_names else (context.getParameter("k") if "k" in param_names else None)
        initial_kp = context.getParameter("k_plan") if "k_plan" in param_names else (context.getParameter("kp") if "kp" in param_names else None)
        initial_k_prot = context.getParameter("k_prot") if has_protein_k else None
        if has_ligand_k or has_planarity_k or has_protein_k:
            logger.info(
                "Applying staged NPT restraint release: protein=%s ligand=%s planarity=%s",
                protein_scales,
                ligand_scales,
                planarity_scales,
            )
        else:
            release_scales = [1.0]
            logger.info("No ligand restraint parameters found in context; NPT runs without staged release")

        # Run with staged release + progress reporting
        completed_steps = 0
        n_stages = max(len(ligand_scales), len(protein_scales), len(planarity_scales))
        stage_lengths = [steps // n_stages] * n_stages
        for i in range(steps % n_stages):
            stage_lengths[i] += 1

        release_stage_details = []
        for stage_idx, stage_steps in enumerate(stage_lengths, start=1):
            if stage_steps <= 0:
                continue
            lig_scale = ligand_scales[min(stage_idx - 1, len(ligand_scales) - 1)]
            prot_scale = protein_scales[min(stage_idx - 1, len(protein_scales) - 1)]
            plan_scale = planarity_scales[min(stage_idx - 1, len(planarity_scales) - 1)]
            if has_ligand_k and initial_k_lig is not None:
                self._set_context_param_if_present(context, "k_lig", float(initial_k_lig) * float(lig_scale))
                self._set_context_param_if_present(context, "k", float(initial_k_lig) * float(lig_scale))
            if has_planarity_k and initial_kp is not None:
                self._set_context_param_if_present(context, "k_plan", float(initial_kp) * float(plan_scale))
                self._set_context_param_if_present(context, "kp", float(initial_kp) * float(plan_scale))
            if has_protein_k and initial_k_prot is not None:
                self._set_context_param_if_present(context, "k_prot", float(initial_k_prot) * float(prot_scale))
            logger.info(
                "NPT restraint-release stage %d/%d: protein=%.3f ligand=%.3f planarity=%.3f steps=%d",
                stage_idx,
                n_stages,
                prot_scale,
                lig_scale,
                plan_scale,
                stage_steps,
            )
            release_stage_details.append(
                {
                    "stage": int(stage_idx),
                    "protein_scale": float(prot_scale) if has_protein_k else None,
                    "ligand_scale": float(lig_scale) if has_ligand_k else None,
                    "planarity_scale": float(plan_scale) if has_planarity_k else None,
                    "steps": int(stage_steps),
                }
            )

            chunk_size = max(stage_steps // 4, 500)
            stage_done = 0
            while stage_done < stage_steps:
                to_run = min(chunk_size, stage_steps - stage_done)
                simulation.step(to_run)
                stage_done += to_run
                completed_steps += to_run

                overall_frac = completed_steps / steps
                npt_progress = progress_start + int(overall_frac * (progress_end - progress_start))
                progress_pct = (completed_steps / steps) * 100
                emit_progress(
                    npt_progress,
                    f"NPT equilibration: {progress_pct:.0f}%",
                    ["preparation", "minimization", "thermal_heating", "nvt"]
                )
                logger.info(f"NPT progress: {completed_steps}/{steps} steps ({progress_pct:.1f}%)")

        # Get final volume
        final_state = simulation.context.getState(getEnergy=True)
        final_volume = final_state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
        volume_change = final_volume - initial_volume
        logger.info(
            f"Final volume: {final_volume:.2f} nm^3 "
            f"(change: {volume_change:+.2f} nm^3, {volume_change/initial_volume*100:+.1f}%)"
        )

        # Save NPT structure
        write_pdb_file(
            simulation.topology,
            simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
            pdb_path,
            keep_ids=True
        )

        logger.info("[COMPLETE] NPT equilibration completed")
        emit_progress(progress_end, "NPT equilibration completed", ["preparation", "minimization", "thermal_heating", "nvt", "npt"])

        return {
            "steps": steps,
            "duration_ps": steps * 0.004,
            "initial_volume_nm3": initial_volume,
            "final_volume_nm3": final_volume,
            "restraint_release_scales": release_scales,
            "restraint_release_applied": bool(has_ligand_k or has_planarity_k or has_protein_k),
            "release_stage_details": release_stage_details,
            "final_scales": {
                "protein": float(protein_scales[-1]) if protein_scales else None,
                "ligand": float(ligand_scales[-1]) if ligand_scales else None,
                "planarity": float(planarity_scales[-1]) if planarity_scales else None,
            },
        }

    def _run_production(
        self, simulation, steps: int, report_interval: int,
        traj_path: str, log_path: str, pdb_path: str, unit,
        progress_start: int = 95, progress_end: int = 100,
        completed_stages: Optional[List[str]] = None,
        checkpoint_interval: int = 25000,
        allow_restrained_production: bool = False,
        force_unrestrained_production: bool = True,
    ) -> Dict[str, Any]:
        """
        Run unrestrained production MD.

        Args:
            simulation: OpenMM Simulation
            steps: Number of production steps
            report_interval: DCD trajectory save interval
            traj_path: Path for DCD trajectory
            log_path: Path for state data log
            pdb_path: Path for final PDB structure
            unit: OpenMM unit module
            progress_start: Starting progress percentage
            progress_end: Ending progress percentage
            completed_stages: List of completed stages so far
            checkpoint_interval: Steps between checkpoint saves

        Returns:
            Dict with production MD statistics
        """
        from openmm.app import StateDataReporter, DCDReporter, CheckpointReporter
        from ..utils.pdb_utils import write_pdb_file

        if completed_stages is None:
            completed_stages = ["preparation", "minimization", "thermal_heating", "nvt", "npt"]

        duration_ns = steps * 0.004 / 1000  # 4 fs timestep
        logger.info(f"=== STAGE 5: PRODUCTION MD ({duration_ns:.1f} ns, {steps} steps) ===")
        warnings: List[str] = []
        active_before = self._list_active_restraint_params(simulation.context)
        any_active_before = any(v is not None and abs(float(v)) > 1e-12 for v in active_before.values())
        if force_unrestrained_production and (not allow_restrained_production) and any_active_before:
            for pname in ("k_prot", "k_lig", "k_plan", "k", "kp"):
                self._set_context_param_if_present(simulation.context, pname, 0.0)
            simulation.context.reinitialize(preserveState=True)
            logger.info("Production MD starting unrestrained: no protein, ligand, or planarity restraint forces active.")
        elif any_active_before:
            msg = "WARNING: Production MD is restrained."
            warnings.append(msg)
            logger.warning(msg)

        # Clear reporters and add production reporters
        simulation.reporters.clear()
        simulation.reporters.append(DCDReporter(traj_path, report_interval))
        simulation.reporters.append(
            StateDataReporter(
                log_path, 1000,
                step=True, potentialEnergy=True, kineticEnergy=True,
                totalEnergy=True, temperature=True, volume=True,
                density=True, speed=True, separator='\t'
            )
        )

        # Add checkpoint reporter
        checkpoint_path = os.path.join(self.output_dir, "production_checkpoint.chk")
        simulation.reporters.append(
            CheckpointReporter(checkpoint_path, checkpoint_interval)
        )

        # Run production with progress reporting
        chunk_size = max(steps // 20, 5000)  # Report ~20 times
        completed_steps = 0

        while completed_steps < steps:
            to_run = min(chunk_size, steps - completed_steps)
            simulation.step(to_run)
            completed_steps += to_run

            frac = completed_steps / steps
            prod_progress = progress_start + int(frac * (progress_end - progress_start))
            elapsed_ns = completed_steps * 0.004 / 1000
            emit_progress(
                prod_progress,
                f"Production MD: {elapsed_ns:.1f}/{duration_ns:.1f} ns",
                completed_stages
            )

            if completed_steps % (chunk_size * 5) == 0 or completed_steps == steps:
                logger.info(
                    f"Production: {completed_steps}/{steps} steps "
                    f"({elapsed_ns:.1f}/{duration_ns:.1f} ns, {frac*100:.0f}%)"
                )

        # Get final state info
        final_state = simulation.context.getState(
            getEnergy=True, getPositions=True, enforcePeriodicBox=True
        )
        final_temp_val = final_state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        final_volume = final_state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)

        # Save final structure
        write_pdb_file(
            simulation.topology,
            final_state.getPositions(),
            pdb_path,
            keep_ids=True
        )

        n_frames = steps // report_interval
        active_after = self._list_active_restraint_params(simulation.context)
        any_active_after = any(v is not None and abs(float(v)) > 1e-12 for v in active_after.values())
        logger.info(
            f"[COMPLETE] Production MD completed: {duration_ns:.1f} ns, "
            f"{n_frames} trajectory frames saved"
        )

        return {
            "steps": steps,
            "duration_ns": duration_ns,
            "duration_ps": steps * 0.004,
            "trajectory_frames": n_frames,
            "final_volume_nm3": final_volume,
            "production_unrestrained": not any_active_after,
            "active_restraints_in_production": active_after,
            "warnings": warnings,
        }
