"""
ABFE Service
Implements absolute binding free energy calculations using OpenFE ecosystem.
"""
from __future__ import annotations
import os
import json
import traceback
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import tempfile
import glob
import signal
import time
from collections import defaultdict
from datetime import datetime, timezone

# Initialize logger early
logger = logging.getLogger(__name__)

# NumPy for statistical calculations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Standard deviation calculations will use fallback.")

# OpenFE and dependencies
try:
    import openfe
    from openfe.protocols.openmm_afe import AbsoluteBindingProtocol
    from openfe.protocols.openmm_utils.omm_settings import OpenFFPartialChargeSettings
    from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges
    from gufe.protocols import execute_DAG
    from openff.units import unit
    OPENFE_AVAILABLE = True
except ImportError:
    OPENFE_AVAILABLE = False
    OPENFE_AVAILABLE = False
    openfe = None
    AbsoluteBindingProtocol = None
    logging.warning("OpenFE not available. ABFE calculations will not work.")

# RDKit for molecule handling
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

class ABFEService:
    """Service for absolute binding free energy calculations using OpenFE."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize ABFE service.
        
        Args:
            output_dir: Directory for storing ABFE calculation outputs.
                       If None, uses ABFE_OUTPUT_DIR env var or defaults to 'data/abfe_outputs'
        """
        if not OPENFE_AVAILABLE:
            raise ImportError("OpenFE is not available. Please install openfe package.")

        # Use provided output_dir, or fall back to environment variable, or use default
        if output_dir is None:
            output_dir = os.getenv("ABFE_OUTPUT_DIR", "data/abfe_outputs")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Job tracking directory (for file-based tracking across processes)
        self.jobs_dir = self.output_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache (for backward compatibility, but file-based is primary)
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
        # Initialize chemistry utilities
        from ovo_ligand.ligandx.lib.chemistry import get_ligand_preparer, get_protein_preparer
        self.ligand_preparer = get_ligand_preparer()
        self.protein_preparer = get_protein_preparer()
        
        logger.info(f"ABFE service initialized with output directory: {self.output_dir}")
    
    def delete_job(self, job_id: str) -> bool:
        """Delete job metadata and associated files."""
        file_path = self.jobs_dir / f"{job_id}.json"
        
        try:
            # Delete metadata file if it exists
            if file_path.exists():
                os.remove(file_path)
            
            # Delete output directory if it exists
            job_output_dir = self.output_dir / job_id
            if job_output_dir.exists() and job_output_dir.is_dir():
                import shutil
                shutil.rmtree(job_output_dir)
            
            # Remove from cache
            if job_id in self.jobs:
                del self.jobs[job_id]
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        For now, we just mark it as failed in the metadata.
        """
        job = self.get_job_status(job_id)
        if not job:
            return False
        
        if job.get('status') in ['running', 'submitted', 'preparing']:
            self._update_job_status(job_id, {
                'status': 'failed',
                'error': 'Job cancelled by user'
            })
            return True
        
        return False
    
    def prepare_ligand_from_structure(
        self,
        ligand_data: str,
        ligand_id: str = "ligand",
        data_format: str = "sdf",
        charge_method: str = "am1bcc"
    ) -> Optional[openfe.SmallMoleculeComponent]:
        """
        Prepare ligand from structure data and assign partial charges.
        
        Args:
            ligand_data: Structure data (SDF, MOL, PDB format)
            ligand_id: Identifier for the ligand
            data_format: Format of ligand data ('sdf', 'mol', 'pdb')
            charge_method: Partial charge method ('am1bcc', 'gasteiger', 'mmff94')
            
        Returns:
            OpenFE SmallMoleculeComponent with assigned charges, or None if failed
        """
        try:
            # Load ligand using RDKit
            if data_format.lower() in ['sdf', 'mol']:
                mol = Chem.MolFromMolBlock(ligand_data, removeHs=False)
            elif data_format.lower() == 'pdb':
                mol = Chem.MolFromPDBBlock(ligand_data, removeHs=False)
            else:
                logger.error(f"Unsupported ligand format: {data_format}")
                return None
            
            if mol is None:
                logger.error(f"Failed to parse ligand structure")
                return None
            
            # Prepare ligand (add Hs, generate 3D if needed)
            # We don't need optimization here as we'll do charge assignment
            mol = self.ligand_preparer.prepare(mol, add_hs=True, generate_3d=True, optimize=False)
            logger.info("Prepared ligand using LigandPreparer")
            
            # Convert to OpenFE SmallMoleculeComponent
            ligand = openfe.SmallMoleculeComponent.from_rdkit(mol, name=ligand_id)
            
            # Assign partial charges using OpenFE utilities
            logger.info(f"Assigning partial charges using {charge_method} method")
            charge_settings = OpenFFPartialChargeSettings(
                partial_charge_method=charge_method,
                off_toolkit_backend="ambertools"
            )
            
            charged_ligands = bulk_assign_partial_charges(
                molecules=[ligand],
                overwrite=False,
                method=charge_settings.partial_charge_method,
                toolkit_backend=charge_settings.off_toolkit_backend,
                generate_n_conformers=charge_settings.number_of_conformers,
                nagl_model=charge_settings.nagl_model,
                processors=1
            )
            
            if charged_ligands and len(charged_ligands) > 0:
                logger.info(f"Successfully prepared ligand: {ligand_id}")
                return charged_ligands[0]
            else:
                logger.error("Failed to assign charges to ligand")
                return None
                
        except Exception as e:
            logger.error(f"Error preparing ligand: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def load_protein(
        self,
        pdb_data: str,
        protein_id: str = "protein"
    ) -> Optional[openfe.ProteinComponent]:
        """
        Load protein from PDB data.
        Automatically cleans protein to remove ligands/heteroatoms.
        
        Args:
            pdb_data: PDB format data as string
            protein_id: Identifier for the protein
            
        Returns:
            OpenFE ProteinComponent or None if failed
        """
        try:
            # First, clean the protein to remove any ligands/heteroatoms
            # This is required for ABFE calculations - protein must be ligand-free
            logger.info("Cleaning protein structure to remove ligands/heteroatoms...")
            
            try:
                # Clean protein structure
                cleaning_result = self.protein_preparer.clean_structure_staged(
                    pdb_data,
                    remove_heterogens=True,
                    remove_water=True,
                    add_missing_residues=True,
                    add_missing_atoms=True,
                    add_missing_hydrogens=True,
                    keep_ligands=False
                )
                
                # Get the final stage (should be 'after_hydrogens' or similar)
                # We want the most processed version
                stages = cleaning_result.get('stages', {})
                if 'after_hydrogens' in stages:
                    cleaned_pdb_data = stages['after_hydrogens']
                elif 'after_missing_atoms' in stages:
                    cleaned_pdb_data = stages['after_missing_atoms']
                elif 'after_water' in stages:
                    cleaned_pdb_data = stages['after_water']
                elif 'after_heterogens' in stages:
                    cleaned_pdb_data = stages['after_heterogens']
                else:
                    # Fallback to original if cleaning didn't produce stages (unlikely)
                    cleaned_pdb_data = stages.get('original', pdb_data)
                
                logger.info("Successfully cleaned protein structure using ProteinPreparer")
                
            except Exception as e:
                logger.warning(f"Protein cleaning failed: {e}, attempting to load protein as-is")
                cleaned_pdb_data = pdb_data
            
            # Write cleaned PDB to temporary file for OpenFE
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_file:
                tmp_file.write(cleaned_pdb_data)
                tmp_path = tmp_file.name
            
            # Load protein using OpenFE
            protein = openfe.ProteinComponent.from_pdb_file(tmp_path, name=protein_id)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            logger.info(f"Successfully loaded protein: {protein_id}")
            return protein
            
        except Exception as e:
            logger.error(f"Error loading protein: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _get_runtime_config(self, simulation_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Normalize ABFE runtime settings used for timeout estimation."""
        cfg = simulation_settings or {}
        fast_mode = bool(cfg.get('fast_mode', True))

        if fast_mode:
            default_prod_ns, default_equil_ns, default_repeats = 0.5, 0.1, 1
        else:
            default_prod_ns, default_equil_ns, default_repeats = 10.0, 1.0, 3

        return {
            'fast_mode': fast_mode,
            'production_length_ns': float(cfg.get('production_length_ns', default_prod_ns)),
            'equilibration_length_ns': float(cfg.get('equilibration_length_ns', default_equil_ns)),
            'protocol_repeats': int(cfg.get('protocol_repeats', default_repeats)),
            'n_replicas_complex': int(cfg.get('n_replicas_complex', 30)),
            'n_replicas_solvent': int(cfg.get('n_replicas_solvent', 14)),
            'dag_timeout_hours': cfg.get('dag_timeout_hours'),
        }

    def _estimate_dag_timeout_seconds(self, simulation_settings: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Estimate DAG timeout from requested workload, with optional user override.

        Workload scales primarily with production length and repeats across both
        solvent and complex legs. Equilibration contributes less, but is included.
        """
        runtime_cfg = self._get_runtime_config(simulation_settings)
        fast_mode = runtime_cfg['fast_mode']
        prod_ns = runtime_cfg['production_length_ns']
        equil_ns = runtime_cfg['equilibration_length_ns']
        repeats = max(1, runtime_cfg['protocol_repeats'])
        replicas_complex = max(1, runtime_cfg.get('n_replicas_complex', 30))
        replicas_solvent = max(1, runtime_cfg.get('n_replicas_solvent', 14))
        user_timeout_hours = runtime_cfg['dag_timeout_hours']

        if user_timeout_hours is not None:
            timeout_hours = max(0.25, float(user_timeout_hours))
            timeout_seconds = int(timeout_hours * 3600)
            details = {
                'source': 'user_override',
                'timeout_hours': timeout_hours,
                'fast_mode': fast_mode,
                'production_length_ns': prod_ns,
                'equilibration_length_ns': equil_ns,
                'protocol_repeats': repeats,
            }
            return timeout_seconds, details

        # Historical fixed defaults were too low for production ABFE workloads on
        # realistic systems; use a larger base budget.
        base_timeout_hours = 6.0 if fast_mode else 48.0
        # Reference workloads that matched previous defaults.
        ref_prod_ns, ref_equil_ns, ref_repeats = (0.5, 0.1, 1) if fast_mode else (10.0, 1.0, 3)
        ref_replicas_complex, ref_replicas_solvent = 30.0, 14.0

        # Approximate relative compute load:
        # - two legs (solvent + complex)
        # - production dominates; equilibration weighted lower
        requested_units = repeats * (2.0 * prod_ns + 0.5 * equil_ns)
        reference_units = max(0.1, ref_repeats * (2.0 * ref_prod_ns + 0.5 * ref_equil_ns))
        base_scale = max(1.0, requested_units / reference_units)
        # Replica count directly affects per-iteration cost.
        replica_scale = (
            (float(replicas_complex) / ref_replicas_complex)
            + (float(replicas_solvent) / ref_replicas_solvent)
        ) / 2.0
        replica_scale = max(0.5, replica_scale)
        scale_factor = max(1.0, base_scale * replica_scale)

        # Add conservative buffer for setup overhead and runtime variance.
        buffered_hours = base_timeout_hours * scale_factor * 1.75
        timeout_hours = min(336.0, max(base_timeout_hours, buffered_hours))
        timeout_seconds = int(timeout_hours * 3600)

        details = {
            'source': 'auto_scaled',
            'timeout_hours': timeout_hours,
            'base_timeout_hours': base_timeout_hours,
            'scale_factor': scale_factor,
            'replica_scale': replica_scale,
            'fast_mode': fast_mode,
            'production_length_ns': prod_ns,
            'equilibration_length_ns': equil_ns,
            'protocol_repeats': repeats,
            'n_replicas_complex': replicas_complex,
            'n_replicas_solvent': replicas_solvent,
        }
        return timeout_seconds, details
    
    def create_chemical_systems(
        self,
        protein: openfe.ProteinComponent,
        ligand: openfe.SmallMoleculeComponent,
        solvent_nacl_concentration: float = 0.15
    ) -> tuple:
        """
        Create ChemicalSystems for ABFE calculation.
        
        Args:
            protein: Protein component
            ligand: Ligand component
            solvent_nacl_concentration: NaCl concentration in M (default 0.15 M)
            
        Returns:
            Tuple of (systemA, systemB) where:
            - systemA: Complex system (protein + ligand + solvent)
            - systemB: Apo system (protein + solvent, ligand decoupled)
        """
        try:
            # Create solvent component
            # ion_concentration must be a quantity with units (molar)
            solvent = openfe.SolventComponent(
                positive_ion='Na+',
                negative_ion='Cl-',
                neutralize=True,
                ion_concentration=solvent_nacl_concentration * unit.molar
            )
            
            # State A: ligand is fully interacting in the complex
            systemA = openfe.ChemicalSystem(
                {
                    'ligand': ligand,
                    'protein': protein,
                    'solvent': solvent,
                },
                name=ligand.name
            )
            
            # State B: ligand is fully decoupled in the complex
            # Only protein and solvent are defined
            systemB = openfe.ChemicalSystem(
                {
                    'protein': protein,
                    'solvent': solvent,
                }
            )
            
            logger.info("Created chemical systems for ABFE calculation")
            logger.info(f"System A (complex): {systemA.name}")
            logger.info(f"System B (apo): protein + solvent")
            
            return systemA, systemB
            
        except Exception as e:
            logger.error(f"Error creating chemical systems: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None
    
    def setup_abfe_protocol(
        self,
        simulation_settings: Optional[Dict[str, Any]] = None
    ) -> AbsoluteBindingProtocol:
        """
        Set up ABFE protocol with user-customizable settings.

        All settings are optional — OpenFE defaults are used when not specified.
        Settings are grouped into: core simulation, environment, advanced, and restraints.

        Args:
            simulation_settings: Optional dict with any of:
                Core: fast_mode, production_length_ns, equilibration_length_ns,
                      protocol_repeats, time_per_iteration_ps, dag_timeout_hours
                Checkpoints: production_n_checkpoints, production_checkpoint_interval_ns,
                             equilibration_n_checkpoints, equilibration_checkpoint_interval_ns
                Ligand: ligand_forcefield, charge_method
                Environment: temperature, pressure, solvent_model, solvent_padding_nm,
                             box_shape, ionic_strength
                Advanced: n_replicas_complex, n_replicas_solvent, minimization_steps,
                          timestep_fs
                Restraints: restraint_settings dict (host_selection, host_min_distance_nm,
                            host_max_distance_nm, rmsf_cutoff_nm, dssp_filter,
                            anchor_finding_strategy)

        Returns:
            Configured AbsoluteBindingProtocol
        """
        try:
            settings = AbsoluteBindingProtocol.default_settings()
            cfg = simulation_settings or {}

            # --- Mode selection ---
            fast_mode = cfg.get('fast_mode', True)
            if fast_mode:
                logger.info("Using FAST MODE (~15-30 min)")
                default_prod_ns, default_equil_ns, default_repeats = 0.5, 0.1, 1
            else:
                logger.info("Using STANDARD/PRODUCTION MODE (hours)")
                default_prod_ns, default_equil_ns, default_repeats = 10.0, 1.0, 3

            # --- Core simulation lengths ---
            prod_len_ns = float(cfg.get('production_length_ns', default_prod_ns))
            equil_len_ns = float(cfg.get('equilibration_length_ns', default_equil_ns))
            protocol_repeats = int(cfg.get('protocol_repeats', default_repeats))
            prod_len = prod_len_ns * unit.nanosecond
            equil_len = equil_len_ns * unit.nanosecond

            # --- Checkpoint intervals ---
            prod_n_cp = int(cfg.get('production_n_checkpoints', 10))
            equil_n_cp = int(cfg.get('equilibration_n_checkpoints', 5))
            prod_cp_interval = (
                float(cfg['production_checkpoint_interval_ns']) * unit.nanosecond
                if 'production_checkpoint_interval_ns' in cfg
                else prod_len / max(prod_n_cp, 1)
            )
            equil_cp_interval = (
                float(cfg['equilibration_checkpoint_interval_ns']) * unit.nanosecond
                if 'equilibration_checkpoint_interval_ns' in cfg
                else equil_len / max(equil_n_cp, 1)
            )

            # OpenFE/OpenMM AFE validates that checkpoint interval must evenly divide
            # the state MCMC move attempt time (commonly 2.5 ps). For very short smoke
            # settings, naive n-checkpoint division can produce invalid tiny intervals.
            # Quantize intervals to valid multiples and clamp to at least one move period.
            mcmc_move_ps = 2.5

            def _quantity_to_ps(q):
                # Support OpenMM-style and Pint/OpenFF-style quantities.
                if hasattr(q, "value_in_unit"):
                    return float(q.value_in_unit(unit.picosecond))
                if hasattr(q, "m_as"):
                    return float(q.m_as(unit.picosecond))
                if hasattr(q, "to"):
                    converted = q.to(unit.picosecond)
                    if hasattr(converted, "magnitude"):
                        return float(converted.magnitude)
                # Last-resort unitless fallback (assume already in ps scale)
                return float(q)

            def _quantize_checkpoint_interval(cp_interval):
                cp_ps = _quantity_to_ps(cp_interval)
                # round to nearest integer number of MCMC move periods
                n_moves = max(1, int(round(cp_ps / mcmc_move_ps)))
                quantized_ps = n_moves * mcmc_move_ps
                return quantized_ps * unit.picosecond

            raw_prod_cp_interval = prod_cp_interval
            raw_equil_cp_interval = equil_cp_interval
            prod_cp_interval = _quantize_checkpoint_interval(prod_cp_interval)
            equil_cp_interval = _quantize_checkpoint_interval(equil_cp_interval)

            if abs(
                _quantity_to_ps(raw_prod_cp_interval)
                - _quantity_to_ps(prod_cp_interval)
            ) > 1e-12:
                logger.warning(
                    "Adjusted production checkpoint interval from %.6f ps to %.6f ps "
                    "to satisfy OpenFE move-period divisibility.",
                    _quantity_to_ps(raw_prod_cp_interval),
                    _quantity_to_ps(prod_cp_interval),
                )
            if abs(
                _quantity_to_ps(raw_equil_cp_interval)
                - _quantity_to_ps(equil_cp_interval)
            ) > 1e-12:
                logger.warning(
                    "Adjusted equilibration checkpoint interval from %.6f ps to %.6f ps "
                    "to satisfy OpenFE move-period divisibility.",
                    _quantity_to_ps(raw_equil_cp_interval),
                    _quantity_to_ps(equil_cp_interval),
                )

            # --- Apply simulation lengths to both legs ---
            for phase_settings in [settings.solvent_simulation_settings, settings.complex_simulation_settings]:
                phase_settings.production_length = prod_len
                phase_settings.equilibration_length = equil_len

            # --- Checkpoint intervals ---
            settings.solvent_output_settings.checkpoint_interval = prod_cp_interval
            settings.complex_output_settings.checkpoint_interval = prod_cp_interval
            settings.solvent_equil_output_settings.checkpoint_interval = equil_cp_interval
            settings.complex_equil_output_settings.checkpoint_interval = equil_cp_interval

            # --- Protocol repeats ---
            try:
                settings.protocol_repeats = protocol_repeats
            except (AttributeError, TypeError) as e:
                logger.warning(f"Could not set protocol_repeats: {e}")

            # --- Ligand forcefield ---
            if 'ligand_forcefield' in cfg:
                settings.forcefield_settings.small_molecule_forcefield = cfg['ligand_forcefield']

            # --- Environment: temperature & pressure ---
            if 'temperature' in cfg:
                settings.thermo_settings.temperature = float(cfg['temperature']) * unit.kelvin
            if 'pressure' in cfg:
                settings.thermo_settings.pressure = float(cfg['pressure']) * unit.bar

            # --- Environment: solvent model & box shape ---
            solvent_model = cfg.get('solvent_model')
            box_shape = cfg.get('box_shape')
            solvent_padding_nm = cfg.get('solvent_padding_nm')
            for solvation in [settings.complex_solvation_settings, settings.solvent_solvation_settings]:
                if solvent_model:
                    solvation.solvent_model = solvent_model
                if box_shape:
                    solvation.box_shape = box_shape
                if solvent_padding_nm is not None:
                    solvation.solvent_padding = float(solvent_padding_nm) * unit.nanometer

            # --- Advanced: n_replicas ---
            if 'n_replicas_complex' in cfg:
                settings.complex_simulation_settings.n_replicas = int(cfg['n_replicas_complex'])
            if 'n_replicas_solvent' in cfg:
                settings.solvent_simulation_settings.n_replicas = int(cfg['n_replicas_solvent'])

            # --- Advanced: minimization steps ---
            if 'minimization_steps' in cfg:
                min_steps = int(cfg['minimization_steps'])
                settings.complex_simulation_settings.minimization_steps = min_steps
                settings.solvent_simulation_settings.minimization_steps = min_steps

            # --- Advanced: integrator timestep ---
            if 'timestep_fs' in cfg:
                settings.integrator_settings.timestep = float(cfg['timestep_fs']) * unit.femtosecond

            # --- Fast mode: reduce pre-equilibration to speed up Boresch restraint search ---
            if fast_mode:
                settings.complex_equil_simulation_settings.production_length = 0.5 * unit.nanosecond
                settings.complex_equil_simulation_settings.equilibration_length = 0.2 * unit.nanosecond
                settings.complex_equil_simulation_settings.equilibration_length_nvt = 0.1 * unit.nanosecond
                settings.solvent_equil_simulation_settings.production_length = 0.25 * unit.nanosecond
                settings.solvent_equil_simulation_settings.equilibration_length = 0.1 * unit.nanosecond
                settings.solvent_equil_simulation_settings.equilibration_length_nvt = 0.05 * unit.nanosecond
                logger.info("  - Reduced pre-equil lengths for fast mode")

            # --- Restraint settings (Boresch) ---
            # Only override OpenFE defaults when the user explicitly provides values.
            restraint_cfg = cfg.get('restraint_settings', {})
            if hasattr(settings, 'restraint_settings') and restraint_cfg:
                if 'host_min_distance_nm' in restraint_cfg:
                    settings.restraint_settings.host_min_distance = float(restraint_cfg['host_min_distance_nm']) * unit.nanometer
                if 'host_max_distance_nm' in restraint_cfg:
                    settings.restraint_settings.host_max_distance = float(restraint_cfg['host_max_distance_nm']) * unit.nanometer
                if 'dssp_filter' in restraint_cfg:
                    settings.restraint_settings.dssp_filter = restraint_cfg['dssp_filter']
                if 'host_selection' in restraint_cfg:
                    settings.restraint_settings.host_selection = restraint_cfg['host_selection']
                if 'rmsf_cutoff_nm' in restraint_cfg:
                    settings.restraint_settings.rmsf_cutoff = float(restraint_cfg['rmsf_cutoff_nm']) * unit.nanometer
                if 'anchor_finding_strategy' in restraint_cfg and hasattr(settings.restraint_settings, 'anchor_finding_strategy'):
                    settings.restraint_settings.anchor_finding_strategy = restraint_cfg['anchor_finding_strategy']

            # --- Log final configuration ---
            logger.info(f"Protocol settings:")
            logger.info(f"  Production: {prod_len}, Equilibration: {equil_len}")
            logger.info(f"  Prod checkpoints: {prod_cp_interval}, Equil checkpoints: {equil_cp_interval}")
            logger.info(f"  Repeats: {protocol_repeats}")
            logger.info(f"  Temperature: {settings.thermo_settings.temperature}, Pressure: {settings.thermo_settings.pressure}")
            logger.info(f"  Solvent: {settings.complex_solvation_settings.solvent_model}, Box: {settings.complex_solvation_settings.box_shape}")
            logger.info(f"  Complex replicas: {settings.complex_simulation_settings.n_replicas}, Solvent replicas: {settings.solvent_simulation_settings.n_replicas}")
            logger.info(f"  Ligand FF: {settings.forcefield_settings.small_molecule_forcefield}")
            logger.info(f"  Timestep: {settings.integrator_settings.timestep}")
            logger.info(f"  Restraints: host={settings.restraint_settings.host_selection}, "
                        f"dist={settings.restraint_settings.host_min_distance}-{settings.restraint_settings.host_max_distance}, "
                        f"rmsf={settings.restraint_settings.rmsf_cutoff}, dssp={settings.restraint_settings.dssp_filter}")

            protocol = AbsoluteBindingProtocol(settings=settings)
            logger.info("ABFE protocol configured successfully")
            return protocol

        except Exception as e:
            logger.error(f"Error setting up ABFE protocol: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def run_abfe_calculation(
        self,
        protein_pdb: str,
        ligand_sdf: str,
        job_id: str,
        simulation_settings: Optional[Dict[str, Any]] = None,
        ligand_id: str = "ligand",
        protein_id: str = "protein"
    ) -> Dict[str, Any]:
        """
        Run complete ABFE calculation workflow.
        
        Args:
            protein_pdb: PDB data for protein
            ligand_sdf: SDF data for ligand
            job_id: Unique identifier for this job
            simulation_settings: Optional custom simulation settings
            ligand_id: Identifier for ligand
            protein_id: Identifier for protein
            
        Returns:
            Dictionary with job status and results
        """
        try:
            logger.info(f"Starting ABFE calculation for job {job_id}")
            total_steps = 7

            def _stage_update(step: int, stage: str, note: str = "") -> None:
                safe_step = max(1, min(int(step), total_steps))
                progress = int(round((safe_step / total_steps) * 100))
                remaining = max(0, total_steps - safe_step)
                payload: Dict[str, Any] = {
                    "status": "running",
                    "current_step": safe_step,
                    "total_steps": total_steps,
                    "progress": progress,
                    "stage": stage,
                    "steps_remaining": remaining,
                    "status_message": note or stage,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                self._update_job_status(job_id, payload)
            
            # Create job directory
            job_dir = self.output_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize job tracking (file-based for cross-process compatibility)
            job_status = {
                'status': 'preparing',
                'job_dir': str(job_dir),
                'error': None,
                'results': None,
                'ligand_id': ligand_id,
                'protein_id': protein_id
            }
            self._save_job_status(job_id, job_status)
            self.jobs[job_id] = job_status
            
            # Step 1: Prepare ligand
            logger.info("Step 1: Preparing ligand...")
            _stage_update(1, "prepare_ligand", "Preparing ligand")
            # Extract charge method from simulation settings
            charge_method = simulation_settings.get('charge_method', 'am1bcc') if simulation_settings else 'am1bcc'
            logger.info(f"Using partial charge method: {charge_method}")
            
            ligand = self.prepare_ligand_from_structure(
                ligand_sdf,
                ligand_id=ligand_id,
                data_format='sdf',
                charge_method=charge_method
            )
            
            if ligand is None:
                raise ValueError("Failed to prepare ligand")
            
            # Step 2: Load protein
            logger.info("Step 2: Loading protein...")
            _stage_update(2, "load_protein", "Loading protein")
            protein = self.load_protein(pdb_data=protein_pdb, protein_id=protein_id)
            
            if protein is None:
                raise ValueError("Failed to load protein")
            
            # Step 3: Create chemical systems
            logger.info("Step 3: Creating chemical systems...")
            _stage_update(3, "create_chemical_systems", "Creating chemical systems")
            systemA, systemB = self.create_chemical_systems(protein, ligand)
            
            if systemA is None or systemB is None:
                raise ValueError("Failed to create chemical systems")
            
            # Step 4: Set up protocol
            logger.info("Step 4: Setting up ABFE protocol...")
            _stage_update(4, "setup_protocol", "Setting up ABFE protocol")
            protocol = self.setup_abfe_protocol(simulation_settings)

            # Step 5: Create ProtocolDAG
            logger.info("Step 5: Creating protocol DAG...")
            _stage_update(5, "create_dag", "Creating protocol DAG")
            dag = protocol.create(
                stateA=systemA,
                stateB=systemB,
                mapping=None  # No atom mapping for ABFE
            )
            
            # Update job status
            self._update_job_status(job_id, {'status': 'running', 'updated_at': datetime.now(timezone.utc).isoformat()})
            
            # Step 6: Execute DAG with timeout guard
            logger.info("Step 6: Executing simulation DAG...")
            _stage_update(6, "execute_dag", "Executing OpenFE DAG")

            timeout_seconds, timeout_details = self._estimate_dag_timeout_seconds(simulation_settings)
            timeout_hours = timeout_seconds / 3600.0
            self._update_job_status(
                job_id,
                {
                    "timeout_budget_hours": timeout_hours,
                    "timeout_source": timeout_details.get("source"),
                    "eta_remaining_hours": round(timeout_hours, 2),
                },
            )
            logger.info(
                "DAG execution timeout: %.2fh (source=%s, mode=%s, prod=%.3f ns, equil=%.3f ns, repeats=%d, scale=%.2f)",
                timeout_hours,
                timeout_details.get('source'),
                'fast' if timeout_details.get('fast_mode') else 'production',
                timeout_details.get('production_length_ns', 0.0),
                timeout_details.get('equilibration_length_ns', 0.0),
                timeout_details.get('protocol_repeats', 0),
                timeout_details.get('scale_factor', 1.0),
            )

            dag_start = time.monotonic()
            def _timeout_handler(signum, frame):
                elapsed_seconds = max(0.0, time.monotonic() - dag_start)
                elapsed_hours = elapsed_seconds / 3600.0
                early_phase = elapsed_seconds < 20 * 60
                if early_phase:
                    context_hint = (
                        "Timeout happened early in execution, so Boresch restraint "
                        "search/setup may be the bottleneck."
                    )
                else:
                    context_hint = (
                        "Timeout happened after substantial runtime, which usually means "
                        "the timeout budget is too low for the requested repeats and/or "
                        "simulation lengths."
                    )
                raise TimeoutError(
                    f"ABFE DAG execution exceeded timeout ({timeout_hours:.2f}h, elapsed {elapsed_hours:.2f}h). "
                    f"{context_hint} "
                    "Try one or more of: set fast_mode=false, increase simulation_settings.dag_timeout_hours, "
                    "reduce protocol_repeats/production_length_ns, or adjust restraint_settings "
                    "(e.g. wider host_max_distance_nm or different host_selection)."
                )
            
            prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                dag_results = execute_DAG(
                    dag,
                    scratch_basedir=job_dir,
                    shared_basedir=job_dir,
                    n_retries=3,
                    keep_shared=True
                )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev_handler)
            
            # Step 7: Extract results
            logger.info("Step 7: Extracting results...")
            _stage_update(7, "extract_results", "Extracting free-energy result")
            binding_free_energy = self._extract_free_energy(dag_results, job_id)
            
            # Try to save ProtocolResult to JSON if it exists and we have a valid result
            if hasattr(dag_results, 'to_dict') or hasattr(dag_results, 'protocol_unit_results'):
                try:
                    logger.info("Attempting to save ProtocolResult to JSON for future reference...")
                    
                    # Helper to make objects serializable
                    import numpy as np
                    def _make_serializable(obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, (np.integer,)):
                            return int(obj)
                        elif isinstance(obj, (np.floating,)):
                            return float(obj)
                        elif hasattr(obj, 'm'):  # OpenMM/OpenFF Quantity
                            return obj.m
                        elif hasattr(obj, 'to_dict'):
                            return _make_serializable(obj.to_dict())
                        elif isinstance(obj, dict):
                            return {k: _make_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [_make_serializable(item) for item in obj]
                        elif isinstance(obj, tuple):
                            return [_make_serializable(item) for item in obj]
                        elif isinstance(obj, (str, int, float, bool, type(None))):
                            return obj
                        else:
                            # Fallback for objects like AbsoluteBindingSettings
                            return str(obj)

                    if hasattr(dag_results, 'to_dict'):
                        result_dict = dag_results.to_dict()
                    else:
                        # Manually construct dict from ProtocolDAGResult
                        result_dict = {
                            'protocol_unit_results': [
                                _make_serializable(pur) for pur in dag_results.protocol_unit_results
                            ] if hasattr(dag_results, 'protocol_unit_results') else []
                        }

                    serializable_dict = _make_serializable(result_dict)
                    result_file = job_dir / "protocol_result.json"
                    
                    with open(result_file, 'w') as f:
                        json.dump(serializable_dict, f, indent=2)
                    logger.info(f"Saved ProtocolResult to {result_file}")
                except Exception as save_error:
                    logger.warning(f"Failed to save ProtocolResult: {save_error}")
            
            # Update job with results
            results = {
                'binding_free_energy_kcal_mol': binding_free_energy,
                'ligand_id': ligand_id,
                'protein_id': protein_id,
                'job_dir': str(job_dir)
            }
            self._update_job_status(job_id, {
                'status': 'completed',
                'results': results,
                'stage': 'completed',
                'steps_remaining': 0,
                'progress': 100,
                'updated_at': datetime.now(timezone.utc).isoformat(),
            })
            
            logger.info(f"ABFE calculation completed for job {job_id}")
            logger.info(f"Binding free energy: {binding_free_energy} kcal/mol")
            
            return self.get_job_status(job_id)
            
        except Exception as e:
            error_msg = f"Error in ABFE calculation: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Update job status to failed
            self._update_job_status(job_id, {
                'status': 'failed',
                'error': error_msg,
                'stage': 'failed',
                'updated_at': datetime.now(timezone.utc).isoformat(),
            })
            
            return {
                'status': 'failed',
                'error': error_msg
            }
    
    def _extract_free_energy(self, dag_results, job_id: str) -> Optional[float]:
        """
        Extract binding free energy from DAG results.
        
        First attempts to extract directly from dag_results (ProtocolResult objects).
        If that fails, falls back to parsing JSON result files from the job directory.
        
        Args:
            dag_results: Results from execute_DAG (ProtocolResult or dict)
            job_id: Job identifier for fallback parsing
            
        Returns:
            Binding free energy in kcal/mol, or None if extraction failed
        """
        try:
            # Method 1: Try to extract directly from dag_results
            try:
                logger.debug(f"Attempting to extract from dag_results. Type: {type(dag_results)}")
                
                # Check if dag_results has get_estimate method (ProtocolResult)
                if hasattr(dag_results, 'get_estimate'):
                    logger.debug("dag_results has get_estimate method, calling it...")
                    estimate = dag_results.get_estimate()
                    if estimate is not None:
                        # Handle unit quantities
                        if hasattr(estimate, 'm'):
                            try:
                                dg_kcal_mol = estimate.to(unit.kilocalorie_per_mole).m
                            except (AttributeError, ValueError, TypeError):
                                dg_kcal_mol = estimate.m
                            logger.info(f"Extracted binding free energy from dag_results: {dg_kcal_mol:.2f} kcal/mol")
                            return float(dg_kcal_mol)
                        elif isinstance(estimate, (int, float)):
                            logger.info(f"Extracted binding free energy from dag_results: {estimate:.2f} kcal/mol")
                            return float(estimate)

                # Check for ProtocolDAGResult structure (list of ProtocolUnitResult)
                if hasattr(dag_results, 'protocol_unit_results'):
                    logger.debug("dag_results has protocol_unit_results, searching for estimates...")
                    # Iterate through unit results to find the estimate
                    # Typically the last unit or a specific unit contains the final result
                    # But since we don't know the exact graph structure, we look for any 'estimate' or 'unit_estimate'
                    
                    estimates = []
                    for pur in dag_results.protocol_unit_results:
                        # Check outputs
                        if hasattr(pur, 'outputs'):
                            outputs = pur.outputs
                            # Check for 'unit_estimate' (common in OpenFE)
                            if 'unit_estimate' in outputs:
                                est = outputs['unit_estimate']
                                estimates.append(est)
                            # Check for 'estimate'
                            elif 'estimate' in outputs:
                                est = outputs['estimate']
                                estimates.append(est)
                    
                    if estimates:
                        # If multiple estimates, we might need to sum them or pick the right one
                        # For ABFE, usually we want the sum of complex and solvent legs if they are separate
                        # But often the ProtocolDAGResult aggregates them.
                        # If we have multiple, let's log them and try to sum them if they look like components
                        logger.info(f"Found {len(estimates)} estimates in protocol units")
                        
                        total_dg = 0.0
                        valid_sum = False
                        
                        for est in estimates:
                            val = 0.0
                            if hasattr(est, 'm'):
                                try:
                                    val = est.to(unit.kilocalorie_per_mole).m
                                    valid_sum = True
                                except:
                                    val = est.m
                                    valid_sum = True
                            elif isinstance(est, (int, float)):
                                val = est
                                valid_sum = True
                            
                            total_dg += val
                        
                        if valid_sum:
                            logger.info(f"Sum of estimates from protocol units: {total_dg:.2f} kcal/mol")
                            return float(total_dg)

                # Try to serialize ProtocolResult to get JSON representation
                if hasattr(dag_results, 'to_dict') or hasattr(dag_results, 'dict'):
                    logger.debug("Attempting to serialize ProtocolResult to dict...")
                    try:
                        if hasattr(dag_results, 'to_dict'):
                            result_dict = dag_results.to_dict()
                        else:
                            result_dict = dag_results.dict()
                        
                        # Look for estimate in serialized dict
                        estimate = result_dict.get('estimate')
                        if estimate is not None:
                            if hasattr(estimate, 'm'):
                                try:
                                    dg_kcal_mol = estimate.to(unit.kilocalorie_per_mole).m
                                except (AttributeError, ValueError, TypeError):
                                    dg_kcal_mol = estimate.m
                                logger.info(f"Extracted binding free energy from serialized ProtocolResult: {dg_kcal_mol:.2f} kcal/mol")
                                return float(dg_kcal_mol)
                            elif isinstance(estimate, (int, float)):
                                logger.info(f"Extracted binding free energy from serialized ProtocolResult: {estimate:.2f} kcal/mol")
                                return float(estimate)
                    except Exception as serialize_error:
                        logger.debug(f"Failed to serialize ProtocolResult: {serialize_error}")
                
                logger.debug(f"Could not extract estimate from dag_results. Available attributes: {[attr for attr in dir(dag_results) if not attr.startswith('_')]}")
                
            except Exception as e:
                logger.warning(f"Direct extraction from dag_results failed: {e}")
                logger.debug(f"dag_results type: {type(dag_results)}")
                logger.debug(traceback.format_exc())
            
            # Method 2: Fall back to parsing JSON files from job directory
            logger.info("Falling back to parsing results from JSON files in job directory...")
            try:
                parsed_results = self.parse_results_from_job(job_id)
                
                if parsed_results.get('error'):
                    logger.warning(f"Error parsing results from job directory: {parsed_results.get('error')}")
                    # Don't return None yet - try to find results in subdirectories
                else:
                    dg_results = parsed_results.get('dg_results', [])
                    if dg_results and len(dg_results) > 0:
                        # Extract the mean DG value from the first ligand result
                        dg_value = dg_results[0].get('dg_kcal_mol')
                        if dg_value is not None:
                            logger.info(f"Extracted binding free energy from JSON files: {dg_value:.2f} kcal/mol")
                            return float(dg_value)
                        else:
                            logger.warning("No dg_kcal_mol found in parsed results")
                    else:
                        logger.warning("No dg_results found in parsed results")
                
                # Method 3: Try to find results in shared/scratch subdirectories
                job_dir = self.output_dir / job_id
                if job_dir.exists():
                    logger.info(f"Searching for result files in job directory: {job_dir}")
                    # Look for result JSON files in all subdirectories recursively
                    result_files = list(job_dir.rglob("*.json"))
                    logger.debug(f"Found {len(result_files)} JSON files in job directory")
                    
                    # Try to load and parse each JSON file
                    for result_file in result_files:
                        try:
                            name, result = self._load_valid_result_json(result_file)
                            if name is not None and result is not None:
                                estimate = result.get('estimate')
                                if estimate is not None:
                                    # Handle unit quantities
                                    if hasattr(estimate, 'm'):
                                        try:
                                            dg_kcal_mol = estimate.to(unit.kilocalorie_per_mole).m
                                        except (AttributeError, ValueError, TypeError):
                                            dg_kcal_mol = estimate.m
                                    elif isinstance(estimate, (int, float)):
                                        dg_kcal_mol = estimate
                                    else:
                                        continue
                                    
                                    logger.info(f"Extracted binding free energy from {result_file}: {dg_kcal_mol:.2f} kcal/mol")
                                    return float(dg_kcal_mol)
                        except Exception as parse_error:
                            logger.debug(f"Failed to parse {result_file}: {parse_error}")
                            continue
                    
            except Exception as e:
                logger.error(f"Error parsing results from job directory: {e}")
                logger.error(traceback.format_exc())
            
            # All methods failed
            logger.warning("Free energy extraction failed - all extraction methods failed")
            logger.info("Please check the job directory for detailed results")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting free energy: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _save_job_status(self, job_id: str, status: Dict[str, Any]) -> None:
        """
        Save job status to file (for cross-process compatibility).
        
        Args:
            job_id: Job identifier
            status: Job status dictionary
        """
        job_file = self.jobs_dir / f"{job_id}.json"
        try:
            with open(job_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save job status to file: {e}")
    
    def _update_job_status(self, job_id: str, updates: Dict[str, Any]) -> None:
        """
        Update job status (both in-memory and file-based).
        
        Args:
            job_id: Job identifier
            updates: Dictionary with fields to update
        """
        # Update in-memory cache
        if job_id not in self.jobs:
            self.jobs[job_id] = {}
        self.jobs[job_id].update(updates)
        
        # Update file-based storage
        current_status = self.get_job_status(job_id)
        if current_status.get('status') != 'not_found':
            current_status.update(updates)
            self._save_job_status(job_id, current_status)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of an ABFE calculation job.
        Tries file-based storage first (for cross-process compatibility),
        then falls back to in-memory cache.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary
        """
        # Try to load from file first (for cross-process compatibility)
        job_file = self.jobs_dir / f"{job_id}.json"
        if job_file.exists():
            try:
                with open(job_file, 'r') as f:
                    status = json.load(f)
                    # Update in-memory cache
                    self.jobs[job_id] = status
                    return status
            except Exception as e:
                logger.warning(f"Failed to load job status from file: {e}")
        
        # Fall back to in-memory cache
        if job_id in self.jobs:
            return self.jobs[job_id]
        
        return {'status': 'not_found', 'error': f'Job {job_id} not found'}
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all ABFE calculation jobs.
        Loads from file-based storage for cross-process compatibility.
        Also verifies job completion status by checking for result files.
        
        Returns:
            List of job dictionaries
        """
        jobs = []
        
        # Load from file-based storage
        if self.jobs_dir.exists():
            for job_file in self.jobs_dir.glob("*.json"):
                try:
                    job_id = job_file.stem
                    with open(job_file, 'r') as f:
                        job_info = json.load(f)
                        
                    # Verify job status: if status says 'running' but results exist, update to 'completed'
                    if job_info.get('status') in ['running', 'preparing', 'submitted']:
                        # Check if job directory has completed results
                        job_dir = self.output_dir / job_id
                        if job_dir.exists():
                            # Check for result JSON files that indicate completion
                            result_fns = self._collect_result_jsons([job_dir])
                            if result_fns:
                                # Check if any result file has valid 'estimate' field (indicates completion)
                                has_valid_results = False
                                for result_fn in result_fns:
                                    try:
                                        name, result = self._load_valid_result_json(result_fn)
                                        if name is not None and result is not None and result.get('estimate') is not None:
                                            has_valid_results = True
                                            break
                                    except Exception:
                                        continue
                                
                                if has_valid_results:
                                    # Try to parse results to get binding free energy
                                    try:
                                        parsed_results = self.parse_results_from_job(job_id)
                                        if parsed_results and not parsed_results.get('error') and parsed_results.get('dg_results'):
                                            # Job has valid completed results, update status
                                            logger.info(f"Job {job_id} has completed results but status was '{job_info.get('status')}'. Updating to 'completed'.")
                                            job_info['status'] = 'completed'
                                            # Extract binding free energy if available
                                            dg_results = parsed_results.get('dg_results', [])
                                            if dg_results and len(dg_results) > 0:
                                                dg_value = dg_results[0].get('dg_kcal_mol')
                                                if dg_value is not None:
                                                    if 'results' not in job_info:
                                                        job_info['results'] = {}
                                                    job_info['results']['binding_free_energy_kcal_mol'] = dg_value
                                                    job_info['results']['job_dir'] = str(job_dir)
                                                    # Also update ligand/protein IDs if available
                                                    if 'ligand_id' not in job_info.get('results', {}):
                                                        job_info['results']['ligand_id'] = job_info.get('ligand_id', 'unknown')
                                                    if 'protein_id' not in job_info.get('results', {}):
                                                        job_info['results']['protein_id'] = job_info.get('protein_id', 'unknown')
                                            # Save updated status
                                            self._save_job_status(job_id, job_info)
                                    except Exception as e:
                                        # If parsing fails but we have result files, still mark as completed
                                        logger.warning(f"Could not fully parse results for job {job_id}, but result files exist: {e}")
                                        job_info['status'] = 'completed'
                                        if 'results' not in job_info:
                                            job_info['results'] = {}
                                        job_info['results']['job_dir'] = str(job_dir)
                                        self._save_job_status(job_id, job_info)
                    
                    jobs.append({'job_id': job_id, **job_info})
                except Exception as e:
                    logger.warning(f"Failed to load job file {job_file}: {e}")
        
        # Also include in-memory jobs that might not be saved yet
        for job_id, job_info in self.jobs.items():
            if not any(j.get('job_id') == job_id for j in jobs):
                jobs.append({'job_id': job_id, **job_info})
        
        return jobs
    
    def _collect_result_jsons(self, results_dirs: List[Path]) -> List[Path]:
        """
        Collect all result JSON files from given directories.
        
        Args:
            results_dirs: List of directories to search for result JSONs
            
        Returns:
            List of paths to result JSON files
        """
        result_fns = []
        for result_dir in results_dirs:
            if not result_dir.exists():
                continue
            # Look for JSON files recursively
            json_files = list(result_dir.rglob("*.json"))
            result_fns.extend(json_files)
        return result_fns
    
    def _load_json(self, fpath: Path) -> Optional[Dict[str, Any]]:
        """
        Load JSON file with error handling.
        
        Args:
            fpath: Path to JSON file
            
        Returns:
            Dictionary with JSON contents or None if failed
        """
        try:
            with open(fpath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load JSON from {fpath}: {e}")
            return None
    
    def _load_valid_result_json(
        self,
        fpath: Path,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Load the data from a results JSON into a dict.
        
        Supports multiple OpenFE result formats:
        1. Legacy format with 'unit_results' dict and 'estimate' at top level
        2. Newer format with 'protocol_unit_results' list (no top-level estimate)
        
        Parameters
        ----------
        fpath : Path
            The path to deserialized results.
            
        Returns
        -------
        tuple[str | None, dict | None]
            Tuple of (ligand_name, result_dict) or (None, None) if invalid.
        """
        if fpath.name == 'db.json':
            return None, None
            
        result = self._load_json(fpath)
        if result is None:
            return None, None
        
        # Check for protocol_unit_results format (newer format from protocol_result.json)
        protocol_unit_results = result.get('protocol_unit_results', [])
        if protocol_unit_results and isinstance(protocol_unit_results, list):
            # This is the newer protocol_result.json format
            try:
                name = self._get_name(result)
                # For this format, we don't require top-level estimate
                # Results are in protocol_unit_results
                return name, result
            except (ValueError, IndexError, KeyError) as e:
                logger.debug(f"{fpath}: Error extracting ligand name from protocol_unit_results format: {e}")
                return None, None
            
        # Try the legacy format with unit_results
        try:
            name = self._get_name(result)
        except (ValueError, IndexError, KeyError) as e:
            # logger.debug(f"{fpath}: Error extracting ligand name: {e}")
            # logger.debug(f"{fpath}: JSON structure - top-level keys: {list(result.keys()) if result else 'None'}")
            return None, None
            
        if result.get("estimate") is None:
            errormsg = f"{fpath}: No 'estimate' found, assuming to be a failed simulation."
            logger.warning(errormsg)
            return None, None
            
        return name, result
    
    def _get_name(self, result: Dict[str, Any]) -> str:
        """
        Get the ligand name from a unit's results data.
        
        Supports multiple formats:
        1. Legacy format with unit_results dict containing 'name' field
        2. Newer format with protocol_units list containing ligand molprops
        
        Parameters
        ----------
        result : dict
            A results dict.
            
        Returns
        -------
        str
            Ligand name corresponding to the results.
        """
        try:
            # First try the protocol_units format (newer protocol_result.json format)
            protocol_units = result.get('protocol_units', [])
            if protocol_units and isinstance(protocol_units, list) and len(protocol_units) > 0:
                # Extract name from protocol_units[0].inputs.stateA.components.ligand.molprops['ofe-name']
                try:
                    inputs = protocol_units[0].get('inputs', {})
                    stateA = inputs.get('stateA', {})
                    components = stateA.get('components', {})
                    ligand = components.get('ligand', {})
                    if isinstance(ligand, dict):
                        molprops = ligand.get('molprops', {})
                        if molprops and 'ofe-name' in molprops:
                            return molprops['ofe-name']
                except (KeyError, IndexError, TypeError):
                    pass  # Fall through to try other methods
            
            # Try unit_results format (legacy format)
            unit_results = result.get('unit_results', {})
            if not unit_results:
                # Log the structure for debugging
                # logger.debug(f"No unit_results at top level. Available keys: {list(result.keys())}")
                
                # Try alternative locations - sometimes unit_results is nested in protocol_result
                if 'protocol_result' in result:
                    unit_results = result['protocol_result'].get('unit_results', {})
                    logger.debug(f"Checking protocol_result for unit_results...")
                
                if not unit_results:
                    # Check if we have protocol_unit_results - might have ligand info there
                    protocol_unit_results = result.get('protocol_unit_results', [])
                    if protocol_unit_results and isinstance(protocol_unit_results, list):
                        # Try to extract from inputs in protocol_unit_results
                        for pur in protocol_unit_results:
                            inputs = pur.get('inputs', {})
                            stateA = inputs.get('stateA', {})
                            components = stateA.get('components', {})
                            ligand = components.get('ligand', {})
                            if isinstance(ligand, dict):
                                molprops = ligand.get('molprops', {})
                                if molprops and 'ofe-name' in molprops:
                                    return molprops['ofe-name']
                    
                    available_keys = list(result.keys())
                    raise ValueError(
                        f"No unit_results found in result JSON. "
                        f"Available top-level keys: {available_keys}. "
                        f"This may indicate the ABFE calculation hasn't completed yet or the result file structure is different than expected."
                    )
            
            # Get first unit result
            first_unit = list(unit_results.values())[0]
            nm = first_unit.get('name', '')
            
            if not nm:
                raise ValueError("No name found in unit results")
            
            # Parse name from format like "Binding, ligand_name complex" or "Binding, ligand_name solvent"
            toks = nm.split('Binding, ')
            if len(toks) < 2:
                raise ValueError(f"Unexpected name format: {nm}")
            
            name_part = toks[1]
            if ' solvent' in name_part:
                name = name_part.split(' solvent')[0]
            elif ' complex' in name_part:
                name = name_part.split(' complex')[0]
            else:
                name = name_part
            
            return name
            
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"Failed to extract ligand name: {e}")
    
    def _get_legs_from_result_jsons(
        self,
        result_fns: List[Path]
    ) -> Dict[str, Dict[str, List]]:
        """
        Iterate over a list of result JSONs and populate a dict with all data needed
        for results processing.
        
        Parameters
        ----------
        result_fns : list[Path]
            List of filepaths containing results formatted as JSON.
            
        Returns
        -------
        dict[str, dict[str, list]]
            Data extracted from the given result JSONs, organized by the ligand name
            and simulation type.
        """
        dgs = defaultdict(lambda: defaultdict(list))
        
        for result_fn in result_fns:
            name, result = self._load_valid_result_json(result_fn)
            
            if name is None:
                continue
            
            # Extract overall estimate
            estimate = result.get("estimate")
            uncertainty = result.get("uncertainty")
            if estimate is not None:
                # Handle unit objects - extract magnitude if it's a quantity
                if hasattr(estimate, 'm'):  # OpenFE unit quantity
                    estimate_value = estimate.m
                    uncertainty_value = uncertainty.m if uncertainty is not None and hasattr(uncertainty, 'm') else 0.0
                elif isinstance(estimate, (int, float)):
                    estimate_value = estimate
                    uncertainty_value = uncertainty if isinstance(uncertainty, (int, float)) else 0.0
                else:
                    # Try to convert to float
                    try:
                        estimate_value = float(estimate)
                        uncertainty_value = float(uncertainty) if uncertainty else 0.0
                    except (ValueError, TypeError):
                        continue
                
                dgs[name]['overall'].append([estimate_value, uncertainty_value])
            
            # Helper function to process outputs from any format
            def process_outputs(outputs: Dict, name: str):
                if 'unit_estimate' in outputs:
                    simtype = outputs.get('simtype', 'unknown')
                    dg = outputs['unit_estimate']
                    dg_error = outputs.get('unit_estimate_error', 0.0)
                    
                    # Handle unit objects
                    if hasattr(dg, 'm'):
                        dg_value = dg.m
                        dg_error_value = dg_error.m if hasattr(dg_error, 'm') else 0.0
                    elif isinstance(dg, (int, float)):
                        dg_value = dg
                        dg_error_value = dg_error if isinstance(dg_error, (int, float)) else 0.0
                    else:
                        try:
                            dg_value = float(dg)
                            dg_error_value = float(dg_error) if dg_error else 0.0
                        except (ValueError, TypeError):
                            return
                    
                    dgs[name][simtype].append([dg_value, dg_error_value])
                
                if 'standard_state_correction' in outputs:
                    corr = outputs['standard_state_correction']
                    if corr is not None:
                        # Handle unit objects
                        if hasattr(corr, 'm'):
                            corr_value = corr.m
                        elif isinstance(corr, (int, float)):
                            corr_value = corr
                        else:
                            try:
                                corr_value = float(corr)
                            except (ValueError, TypeError):
                                return
                        
                        dgs[name]['standard_state_correction'].append([corr_value, 0.0])
            
            # Try protocol_unit_results format (newer format - list based)
            protocol_unit_results = result.get('protocol_unit_results', [])
            if protocol_unit_results and isinstance(protocol_unit_results, list):
                for pur in protocol_unit_results:
                    outputs = pur.get('outputs', {})
                    process_outputs(outputs, name)
                continue  # Skip the legacy format processing
            
            # Extract unit results (legacy format - dict based)
            unit_results = result.get('unit_results', {})
            proto_key = [
                k for k in unit_results.keys()
                if k.startswith("ProtocolUnitResult")
            ]
            
            for p in proto_key:
                unit_result = unit_results[p]
                outputs = unit_result.get('outputs', {})
                process_outputs(outputs, name)
        
        return dgs
    
    def extract_results_dict(
        self,
        results_dirs: List[Path],
    ) -> Dict[str, Dict[str, List]]:
        """
        Get a dictionary of ABFE results from a list of directories.
        
        Parameters
        ----------
        results_dirs : list[Path]
            A list of directories with ABFE result files to process.
            
        Returns
        -------
        dict[str, dict[str, list]]
            Simulation results, organized by the leg's ligand names and simulation type.
        """
        # Find and filter result jsons
        result_fns = self._collect_result_jsons(results_dirs)
        
        if not result_fns:
            logger.warning(f"No result JSON files found in directories: {results_dirs}")
            return {}
        
        logger.info(f"Found {len(result_fns)} result JSON files")
        
        # Pair legs of simulations together into dict of dicts
        sim_results = self._get_legs_from_result_jsons(result_fns)
        
        return sim_results
    
    def generate_dg(self, results_dict: Dict[str, Dict[str, List]]) -> List[Dict[str, Any]]:
        """
        Compute DG values for the given results.
        
        Supports two scenarios:
        1. Overall estimate provided at top level (legacy format)
        2. Calculate from legs: DG_binding = DG_complex - DG_solvent + standard_state_correction
        
        Parameters
        ----------
        results_dict : dict[str, dict[str, list]]
            Dictionary of results created by extract_results_dict.
            
        Returns
        -------
        list[dict]
            A list of dictionaries with the dG results for each ligand.
        """
        data = []
        
        # Check the type of error which should be used based on the number of repeats
        repeats = {len(v.get("overall", [])) for v in results_dict.values()}
        use_mbar_error = 1 in repeats or 0 in repeats  # Use MBAR error if only single repeat or no overall
        
        for lig, results in sorted(results_dict.items()):
            overall_data = results.get("overall", [])
            
            # If no overall data, try to calculate from legs
            if not overall_data:
                # Calculate binding free energy from legs: DG_binding = DG_complex - DG_solvent
                complex_data = results.get("complex", [])
                solvent_data = results.get("solvent", [])
                standard_state_data = results.get("standard_state_correction", [])
                
                if complex_data and solvent_data:
                    # Get mean values for each leg
                    complex_dg = sum([v[0] for v in complex_data]) / len(complex_data)
                    complex_err = sum([v[1] for v in complex_data]) / len(complex_data)
                    solvent_dg = sum([v[0] for v in solvent_data]) / len(solvent_data)
                    solvent_err = sum([v[1] for v in solvent_data]) / len(solvent_data)
                    
                    # Standard state correction (from complex leg, if available)
                    std_corr = 0.0
                    if standard_state_data:
                        std_corr = sum([v[0] for v in standard_state_data]) / len(standard_state_data)
                    
                    # DG_binding = DG_complex - DG_solvent + standard_state_correction
                    # Note: Values are in kT units, need to convert to kcal/mol
                    # 1 kT at 298.15 K ≈ 0.593 kcal/mol
                    kT_to_kcal = 0.593
                    dg = (complex_dg - solvent_dg + std_corr) * kT_to_kcal
                    
                    # Error propagation: sqrt(err_complex^2 + err_solvent^2)
                    error = ((complex_err ** 2 + solvent_err ** 2) ** 0.5) * kT_to_kcal
                    
                    data.append({
                        "ligand": lig,
                        "dg_kcal_mol": round(dg, 2),
                        "uncertainty_kcal_mol": round(error, 2),
                    })
                continue
            
            # Calculate mean DG from overall data
            dg_values = [v[0] for v in overall_data]
            dg = sum(dg_values) / len(dg_values)
            
            # Calculate error
            if use_mbar_error:
                # Use average MBAR error from legs
                complex_errors = [x[1] for x in results.get("complex", [])]
                solvent_errors = [x[1] for x in results.get("solvent", [])]
                
                if complex_errors and solvent_errors:
                    mean_complex_error = sum(complex_errors) / len(complex_errors)
                    mean_solvent_error = sum(solvent_errors) / len(solvent_errors)
                    error = (mean_complex_error ** 2 + mean_solvent_error ** 2) ** 0.5
                elif overall_data:
                    # Fall back to overall uncertainty
                    error = sum([v[1] for v in overall_data]) / len(overall_data)
                else:
                    error = 0.0
            else:
                # Use standard deviation of repeats
                if len(dg_values) > 1:
                    if NUMPY_AVAILABLE:
                        error = float(np.std(dg_values))
                    else:
                        # Fallback calculation without numpy
                        mean_val = sum(dg_values) / len(dg_values)
                        variance = sum((x - mean_val) ** 2 for x in dg_values) / len(dg_values)
                        error = variance ** 0.5
                else:
                    error = overall_data[0][1] if overall_data else 0.0
            
            data.append({
                "ligand": lig,
                "dg_kcal_mol": round(dg, 2),
                "uncertainty_kcal_mol": round(error, 2),
            })
        
        return data
    
    def generate_dg_raw(self, results_dict: Dict[str, Dict[str, List]]) -> List[Dict[str, Any]]:
        """
        Get all the transformation cycle legs found and their DG values.
        
        Note: Values from protocol_unit_results are in kT units and need conversion.
        
        Parameters
        ----------
        results_dict : dict[str, dict[str, list]]
            Dictionary of results created by extract_results_dict.
            
        Returns
        -------
        list[dict]
            A list of dictionaries with the individual cycle leg dG results in kcal/mol.
        """
        data = []
        
        # 1 kT at 298.15 K ≈ 0.593 kcal/mol
        kT_to_kcal = 0.593
        
        for lig, results in sorted(results_dict.items()):
            for simtype, repeats in sorted(results.items()):
                if simtype != "overall":
                    for repeat in repeats:
                        dg_value = repeat[0]
                        dg_error = repeat[1]
                        
                        # Convert from kT to kcal/mol
                        dg_kcal = dg_value * kT_to_kcal
                        error_kcal = dg_error * kT_to_kcal
                        
                        data.append({
                            "leg": simtype,
                            "ligand": lig,
                            "dg_kcal_mol": round(dg_kcal, 2),
                            "uncertainty_kcal_mol": round(error_kcal, 2),
                        })
        
        return data
    
    def parse_results_from_job(
        self,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Parse ABFE results from a job directory.
        
        This method searches for OpenFE result JSON files in the job directory
        and extracts binding free energy data.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary with parsed results including:
            - dg_results: Overall DG values per ligand
            - dg_raw: Raw leg contributions
            - ligands: List of ligand names found
        """
        job_status = self.get_job_status(job_id)
        
        if job_status.get('status') == 'not_found':
            return {
                'error': f'Job {job_id} not found',
                'dg_results': [],
                'dg_raw': [],
                'ligands': []
            }
        
        job_dir = job_status.get('job_dir')
        if not job_dir:
            return {
                'error': f'No job directory found for job {job_id}',
                'dg_results': [],
                'dg_raw': [],
                'ligands': []
            }
        
        job_path = Path(job_dir)
        if not job_path.exists():
            return {
                'error': f'Job directory does not exist: {job_dir}',
                'dg_results': [],
                'dg_raw': [],
                'ligands': []
            }
        
        # Search for result JSON files in the job directory
        # OpenFE typically stores results in subdirectories
        results_dirs = [job_path]
        
        # Also check for common subdirectories
        for subdir in job_path.iterdir():
            if subdir.is_dir():
                results_dirs.append(subdir)
        
        try:
            # Extract results
            results_dict = self.extract_results_dict(results_dirs)
            
            if not results_dict:
                # List what files were found for debugging
                json_files = list(job_path.rglob("*.json"))
                logger.error(f"Found {len(json_files)} JSON files in job directory but none were valid result files")
                for jf in json_files[:5]:  # Log first 5 files
                    logger.error(f"  - {jf.relative_to(job_path)}")
                
                return {
                    'error': (
                        f'No valid ABFE result files found in job directory. '
                        f'Found {len(json_files)} JSON files but none contained the expected structure with unit_results. '
                        f'This usually means the ABFE calculation is still running or failed before producing results. '
                        f'Check the job directory for log files: {job_dir}'
                    ),
                    'dg_results': [],
                    'dg_raw': [],
                    'ligands': []
                }
            
            # Generate formatted results
            dg_results = self.generate_dg(results_dict)
            dg_raw = self.generate_dg_raw(results_dict)
            ligands = list(results_dict.keys())
            
            return {
                'dg_results': dg_results,
                'dg_raw': dg_raw,
                'ligands': ligands,
                'job_dir': str(job_dir)
            }
            
        except Exception as e:
            logger.error(f"Error parsing results from job {job_id}: {e}")
            logger.error(traceback.format_exc())
            return {
                'error': f'Error parsing results: {str(e)}',
                'dg_results': [],
                'dg_raw': [],
                'ligands': []
            }
    
    def get_detailed_analysis(self, job_id: str) -> Dict[str, Any]:
        """
        Get detailed analysis data for an ABFE job including:
        - Overlap matrices
        - Convergence data
        - Output files
        - Thermodynamic cycle breakdown
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary with detailed analysis data
        """
        try:
            job_status = self.get_job_status(job_id)
            
            if job_status.get('status') == 'not_found':
                return {'error': f'Job {job_id} not found', 'job_id': job_id}
            
            job_dir = self.output_dir / job_id
            if not job_dir.exists():
                return {'error': f'Job directory not found: {job_dir}', 'job_id': job_id}
            
            # Initialize result structure
            result = {
                'job_id': job_id,
                'legs': [],
                'convergence_data': None,
                'thermodynamic_cycle': None,
                'output_files': {
                    'logs': [],
                    'structures': [],
                    'trajectories': [],
                    'analysis_plots': []
                }
            }
            
            # Find all leg directories (complex and solvent)
            # OpenFE creates directories named like:
            # shared_AbsoluteBindingComplexUnit-UUID_attempt_0
            # shared_AbsoluteBindingSolventUnit-UUID_attempt_0
            # Multiple repeats create multiple directories with different UUIDs but same attempt number
            # We identify repeats by sorting directories and pairing them up
            complex_dirs = []
            solvent_dirs = []
            
            for subdir in job_dir.iterdir():
                if subdir.is_dir():
                    subdir_name = subdir.name.lower()
                    
                    # Check for AbsoluteBindingComplexUnit or AbsoluteBindingSolventUnit
                    if 'absolutebindingcomplexunit' in subdir_name or 'complexunit' in subdir_name:
                        complex_dirs.append(subdir)
                    elif 'absolutebindingsolventunit' in subdir_name or 'solventunit' in subdir_name:
                        solvent_dirs.append(subdir)
            
            # Sort directories by modification time to pair them as repeats
            complex_dirs.sort(key=lambda d: d.stat().st_mtime)
            solvent_dirs.sort(key=lambda d: d.stat().st_mtime)
            
            # Determine number of repeats (max of complex and solvent dirs)
            num_repeats = max(len(complex_dirs), len(solvent_dirs))
            
            # Process each repeat and leg
            convergence_checkpoints = []
            leg_results = {}
            
            for repeat_num in range(num_repeats):
                # Process solvent leg first (OpenFE runs solvent before complex)
                if repeat_num < len(solvent_dirs):
                    leg_dir = solvent_dirs[repeat_num]
                    leg_analysis = self._analyze_leg('solvent', leg_dir, job_id, repeat_num)
                    result['legs'].append(leg_analysis)
                    
                    if leg_analysis.get('convergence_checkpoints'):
                        convergence_checkpoints.extend(leg_analysis['convergence_checkpoints'])
                    
                    if repeat_num == 0 and leg_analysis.get('mbar_analysis'):
                        leg_results['solvent'] = leg_analysis['mbar_analysis']
                    
                    self._collect_output_files(leg_dir, 'solvent', result['output_files'], repeat_num)
                
                # Then complex leg
                if repeat_num < len(complex_dirs):
                    leg_dir = complex_dirs[repeat_num]
                    leg_analysis = self._analyze_leg('complex', leg_dir, job_id, repeat_num)
                    result['legs'].append(leg_analysis)
                    
                    if leg_analysis.get('convergence_checkpoints'):
                        convergence_checkpoints.extend(leg_analysis['convergence_checkpoints'])
                    
                    if repeat_num == 0 and leg_analysis.get('mbar_analysis'):
                        leg_results['complex'] = leg_analysis['mbar_analysis']
                    
                    self._collect_output_files(leg_dir, 'complex', result['output_files'], repeat_num)
            
            # Build convergence data
            if convergence_checkpoints:
                result['convergence_data'] = {
                    'forward_reverse_available': len(convergence_checkpoints) > 2,
                    'checkpoints': sorted(convergence_checkpoints, key=lambda x: (x['leg'], x['iteration']))
                }
            
            # Build thermodynamic cycle if both legs complete
            parsed = self.parse_results_from_job(job_id)
            if parsed.get('dg_results') and not parsed.get('error'):
                dg_result = parsed['dg_results'][0] if parsed['dg_results'] else None
                dg_raw = parsed.get('dg_raw', [])
                
                # Extract complex and solvent contributions
                complex_dg = next((r['dg_kcal_mol'] for r in dg_raw if r['leg'] == 'complex'), None)
                complex_err = next((r['uncertainty_kcal_mol'] for r in dg_raw if r['leg'] == 'complex'), 0)
                solvent_dg = next((r['dg_kcal_mol'] for r in dg_raw if r['leg'] == 'solvent'), None)
                solvent_err = next((r['uncertainty_kcal_mol'] for r in dg_raw if r['leg'] == 'solvent'), 0)
                restraint_corr = next((r['dg_kcal_mol'] for r in dg_raw if 'standard_state' in r['leg']), 0)
                
                if complex_dg is not None and solvent_dg is not None and dg_result:
                    result['thermodynamic_cycle'] = {
                        'dg_complex': complex_dg,
                        'dg_complex_error': complex_err,
                        'dg_solvent': solvent_dg,
                        'dg_solvent_error': solvent_err,
                        'dg_restraint_correction': restraint_corr,
                        'dg_binding': dg_result['dg_kcal_mol'],
                        'dg_binding_error': dg_result['uncertainty_kcal_mol']
                    }
            
            # Fallback: build thermodynamic cycle from leg MBAR data + stored job results
            if not result.get('thermodynamic_cycle'):
                complex_leg = next((l for l in result['legs'] if l['leg_type'] == 'complex' and l.get('free_energy_kcal_mol') is not None), None)
                solvent_leg = next((l for l in result['legs'] if l['leg_type'] == 'solvent' and l.get('free_energy_kcal_mol') is not None), None)
                
                if complex_leg and solvent_leg:
                    dg_complex = complex_leg['free_energy_kcal_mol']
                    dg_complex_err = complex_leg.get('uncertainty_kcal_mol', 0)
                    dg_solvent = solvent_leg['free_energy_kcal_mol']
                    dg_solvent_err = solvent_leg.get('uncertainty_kcal_mol', 0)
                    
                    # Try to get the overall binding free energy from stored job results
                    dg_binding = None
                    job_results = job_status.get('result', {}).get('results', {}) if isinstance(job_status.get('result'), dict) else {}
                    if not job_results:
                        job_results = job_status.get('results', {})
                    if isinstance(job_results, dict):
                        dg_binding = job_results.get('binding_free_energy_kcal_mol')
                    
                    if dg_binding is None:
                        # Calculate from legs: binding = complex - solvent
                        dg_binding = dg_complex - dg_solvent
                    
                    import math
                    dg_binding_err = math.sqrt(dg_complex_err**2 + dg_solvent_err**2)
                    
                    result['thermodynamic_cycle'] = {
                        'dg_complex': dg_complex,
                        'dg_complex_error': dg_complex_err,
                        'dg_solvent': dg_solvent,
                        'dg_solvent_error': dg_solvent_err,
                        'dg_restraint_correction': 0,
                        'dg_binding': dg_binding,
                        'dg_binding_error': dg_binding_err
                    }
                    logger.info(f"Built thermodynamic cycle from leg MBAR data for job {job_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting detailed analysis for job {job_id}: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e), 'job_id': job_id}
    
    def _analyze_leg(self, leg_type: str, leg_dir: Path, job_id: str, repeat_num: int = 0) -> Dict[str, Any]:
        """Analyze a single leg (complex or solvent) of the ABFE calculation."""
        leg_analysis = {
            'leg_name': leg_dir.name,
            'leg_type': leg_type,
            'repeat_num': repeat_num,
            'status': 'pending',
            'convergence_checkpoints': []
        }
        
        try:
            # Check for analysis plots
            overlap_matrix = leg_dir / 'mbar_overlap_matrix.png'
            if overlap_matrix.exists():
                leg_analysis['overlap_matrix_path'] = f'/api/abfe/file/{job_id}/{leg_dir.name}/mbar_overlap_matrix.png'
            
            replica_matrix = leg_dir / 'replica_exchange_matrix.png'
            if replica_matrix.exists():
                leg_analysis['replica_exchange_matrix_path'] = f'/api/abfe/file/{job_id}/{leg_dir.name}/replica_exchange_matrix.png'
            
            replica_timeseries = leg_dir / 'replica_state_timeseries.png'
            if replica_timeseries.exists():
                leg_analysis['replica_state_timeseries_path'] = f'/api/abfe/file/{job_id}/{leg_dir.name}/replica_state_timeseries.png'
            
            # Parse real-time analysis YAML if available
            yaml_files = list(leg_dir.glob('*_real_time_analysis.yaml'))
            if yaml_files:
                try:
                    import yaml
                    with open(yaml_files[0], 'r') as f:
                        analysis_data = yaml.safe_load(f)
                    
                    if analysis_data and isinstance(analysis_data, list):
                        # Get the latest checkpoint
                        latest = analysis_data[-1]
                        leg_analysis['status'] = 'completed' if latest.get('percent_complete', 0) >= 100 else 'running'
                        
                        if 'mbar_analysis' in latest:
                            mbar = latest['mbar_analysis']
                            # Convert kT to kcal/mol (1 kT at 298K ≈ 0.593 kcal/mol)
                            kT_to_kcal = 0.593
                            leg_analysis['mbar_analysis'] = {
                                'free_energy_in_kT': mbar.get('free_energy_in_kT', 0),
                                'standard_error_in_kT': mbar.get('standard_error_in_kT', 0),
                                'number_of_uncorrelated_samples': mbar.get('number_of_uncorrelated_samples', 0),
                                'n_equilibrium_iterations': mbar.get('n_equilibrium_iterations', 0),
                                'statistical_inefficiency': mbar.get('statistical_inefficiency', 0)
                            }
                            leg_analysis['free_energy_kT'] = mbar.get('free_energy_in_kT', 0)
                            leg_analysis['free_energy_kcal_mol'] = mbar.get('free_energy_in_kT', 0) * kT_to_kcal
                            leg_analysis['uncertainty_kcal_mol'] = mbar.get('standard_error_in_kT', 0) * kT_to_kcal
                        
                        if 'timing_data' in latest:
                            timing = latest['timing_data']
                            leg_analysis['timing_data'] = {
                                'iteration_seconds': timing.get('iteration_seconds', 0),
                                'average_seconds_per_iteration': timing.get('average_seconds_per_iteration', 0),
                                'estimated_time_remaining': timing.get('estimated_time_remaining', ''),
                                'estimated_total_time': timing.get('estimated_total_time', ''),
                                'ns_per_day': timing.get('ns_per_day', 0),
                                'percent_complete': latest.get('percent_complete', 0)
                            }
                        
                        # Extract all checkpoints for convergence analysis
                        for checkpoint in analysis_data:
                            if 'mbar_analysis' in checkpoint:
                                leg_analysis['convergence_checkpoints'].append({
                                    'iteration': checkpoint.get('iteration', 0),
                                    'percent_complete': checkpoint.get('percent_complete', 0),
                                    'leg': leg_type,
                                    'free_energy_kT': checkpoint['mbar_analysis'].get('free_energy_in_kT', 0),
                                    'standard_error_kT': checkpoint['mbar_analysis'].get('standard_error_in_kT', 0),
                                    'n_uncorrelated_samples': checkpoint['mbar_analysis'].get('number_of_uncorrelated_samples', 0)
                                })
                        
                        leg_analysis['n_iterations'] = latest.get('iteration', 0)
                        
                except Exception as yaml_error:
                    logger.warning(f"Failed to parse YAML for {leg_type}: {yaml_error}")
            
            # Check for .nc files to determine status if YAML not available
            if leg_analysis['status'] == 'pending':
                nc_files = list(leg_dir.glob('*.nc'))
                checkpoint_files = list(leg_dir.glob('*checkpoint*'))
                if nc_files or checkpoint_files:
                    leg_analysis['status'] = 'running'
                
                # Check if overlap matrix exists (indicates completion)
                if overlap_matrix.exists():
                    leg_analysis['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Error analyzing leg {leg_type}: {e}")
            leg_analysis['status'] = 'failed'
            leg_analysis['error'] = str(e)
        
        return leg_analysis
    
    def _collect_output_files(self, leg_dir: Path, leg_type: str, output_files: Dict[str, List], repeat_num: int = 0) -> None:
        """Collect and categorize output files from a leg directory."""
        try:
            for file_path in leg_dir.iterdir():
                if not file_path.is_file():
                    continue
                
                file_info = {
                    'filename': file_path.name,
                    'path': str(file_path),
                    'size_bytes': file_path.stat().st_size,
                    'leg': leg_type,
                    'repeat_num': repeat_num,
                    'leg_dir': leg_dir.name  # Store leg directory name for file retrieval
                }
                
                ext = file_path.suffix.lower()
                name = file_path.name.lower()
                
                if ext == '.log' or 'simulation.log' in name:
                    file_info['file_type'] = 'log'
                    file_info['description'] = f'{leg_type.capitalize()} simulation log'
                    output_files['logs'].append(file_info)
                
                elif ext == '.pdb':
                    file_info['file_type'] = 'structure'
                    if 'minimized' in name:
                        file_info['description'] = f'{leg_type.capitalize()} minimized structure'
                    elif 'equil' in name:
                        file_info['description'] = f'{leg_type.capitalize()} equilibrated structure'
                    elif 'alchemical' in name:
                        file_info['description'] = f'{leg_type.capitalize()} alchemical system'
                    else:
                        file_info['description'] = f'{leg_type.capitalize()} structure'
                    output_files['structures'].append(file_info)
                
                elif ext in ['.xtc', '.nc']:
                    file_info['file_type'] = 'trajectory'
                    file_info['description'] = f'{leg_type.capitalize()} trajectory'
                    output_files['trajectories'].append(file_info)
                
                elif ext == '.png':
                    file_info['file_type'] = 'plot'
                    if 'overlap' in name:
                        file_info['description'] = 'MBAR overlap matrix'
                    elif 'replica_exchange' in name:
                        file_info['description'] = 'Replica exchange matrix'
                    elif 'timeseries' in name:
                        file_info['description'] = 'Replica state timeseries'
                    else:
                        file_info['description'] = f'{leg_type.capitalize()} analysis plot'
                    output_files['analysis_plots'].append(file_info)
                    
        except Exception as e:
            logger.warning(f"Error collecting output files from {leg_dir}: {e}")
    
    def get_file_path(self, job_id: str, leg_name: str, filename: str) -> Optional[Path]:
        """
        Get the full path to a file in a job's leg directory.
        
        Args:
            job_id: Job identifier
            leg_name: Name of the leg directory
            filename: Name of the file
            
        Returns:
            Full path to the file if it exists, None otherwise
        """
        job_dir = self.output_dir / job_id
        
        # Try direct match first
        file_path = job_dir / leg_name / filename
        if file_path.exists():
            return file_path
        
        # Search in subdirectories that match the leg name
        for subdir in job_dir.iterdir():
            if subdir.is_dir() and leg_name in subdir.name:
                candidate = subdir / filename
                if candidate.exists():
                    return candidate
        
        return None
