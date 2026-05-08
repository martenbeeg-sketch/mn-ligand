"""
MD Optimization Service - Modular Architecture

This module provides a complete MD optimization service using modular components.
All heavy simulation logic has been extracted into separate modules.
"""

import os
import json
import logging
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from .config import MDOptimizationConfig
from .validation import (
    validate_system_result,
    validate_equilibration_result,
    validate_ligand_preparation,
    validate_protein_preparation,
)
from .preparation import ProteinPreparation, LigandPreparation, ChargeAssignment, SystemBuilder
from .simulation import EnergyMinimization, Equilibration, TrajectoryProcessor, SimulationRunner
from .utils import PDBWriter, EnvironmentValidator, clean_results_for_json
from .workflow.analytics import EquilibrationAnalytics
from .workflow import (
    SolvatedSystemBuilder,
    EquilibrationRunner,
    LigandProcessor,
    TrajectoryProcessorRunner
)

logger = logging.getLogger(__name__)


class MDOptimizationService:
    """MD optimization service"""
    
    def __init__(self, output_dir: str = "data/md_outputs", job_id: Optional[str] = None):
        """Initialize with basic path setup."""
        self.base_dir = output_dir
        if job_id:
            self.output_dir = os.path.join(output_dir, job_id)
        else:
            self.output_dir = output_dir
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        self._initialized = False
        self.environment_status = {}
        self.ligand_ff = None
        self.protein_ff = None
        self._last_ligand_error: Optional[str] = None
    
    def _lazy_init(self):
        """Perform heavy initialization only when needed."""
        if self._initialized:
            return
            
        logger.info("Initializing MD Optimization Service components...")
        
        # Validate environment
        from .utils import EnvironmentValidator
        self.environment_status = EnvironmentValidator.validate_environment()
        
        # Initialize chemistry utilities
        from ovo_ligand.ligandx.lib.chemistry import get_pdb_parser, get_component_analyzer, get_protein_preparer
        self.pdb_parser = get_pdb_parser()
        self.component_analyzer = get_component_analyzer()
        self.protein_preparer = get_protein_preparer()
        
        # Initialize modular components
        from .preparation import ProteinPreparation, LigandPreparation, ChargeAssignment, SystemBuilder
        from .simulation import EnergyMinimization, Equilibration, TrajectoryProcessor
        from .utils import PDBWriter
        
        self.protein_prep = ProteinPreparation()
        self.ligand_prep = LigandPreparation()
        self.charge_assignment = ChargeAssignment()
        self.system_builder_config = SystemBuilder()
        self.minimization = EnergyMinimization()
        self.equilibration = Equilibration()
        self.trajectory_processor = TrajectoryProcessor()
        self.pdb_writer = PDBWriter()
        
        # Initialize workflow components
        from .workflow import (
            SolvatedSystemBuilder,
            EquilibrationRunner,
            LigandProcessor,
            TrajectoryProcessorRunner
        )
        self.ligand_processor = LigandProcessor(self.environment_status)
        self.solvated_system_builder = SolvatedSystemBuilder(self.output_dir)
        self.equilibration_runner = EquilibrationRunner(self.output_dir)
        self.trajectory_runner = TrajectoryProcessorRunner(self.output_dir)
        
        # Initialize force fields
        self._initialize_force_fields()
        
        self._initialized = True
        logger.info("[COMPLETE] MD Optimization Service initialized")
    
    def _initialize_force_fields(self):
        """Initialize force fields."""
        self.ligand_ff = None
        self.protein_ff = None
        
        try:
            if self.environment_status.get('openff', False):
                from openff.toolkit import ForceField
                self.ligand_ff = ForceField('openff-2.2.0.offxml')
                logger.info("[COMPLETE] OpenFF Sage force field loaded")
            
            if self.environment_status.get('openmm', False):
                from openmm.app import ForceField as OpenMMForceField
                self.protein_ff = OpenMMForceField('amber14-all.xml', 'amber14/tip3p.xml')
                logger.info("[COMPLETE] AMBER force field loaded")
        except Exception as e:
            logger.error(f"Force field initialization failed: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        self._lazy_init()
        return {
            "service": "MDOptimizationService",
            "version": "2.0",
            "output_dir": self.output_dir,
            "environment": {
                "openff": self.environment_status.get('openff', False),
                "openmm": self.environment_status.get('openmm', False),
                "rdkit": self.environment_status.get('rdkit', False),
                "pdbfixer": self.environment_status.get('pdbfixer', False),
                "mdtraj": self.environment_status.get('mdtraj', False),
                "platforms": self.environment_status.get('openmm_platforms', [])
            },
            "force_fields": {
                "ligand_ff_loaded": self.ligand_ff is not None,
                "protein_ff_loaded": self.protein_ff is not None
            }
        }

    def _get_jobs_dir(self) -> Path:
        """Get the directory where job metadata is stored."""
        jobs_dir = Path(self.base_dir) / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        return jobs_dir

    def save_job(self, job_id: str, data: Dict[str, Any]):
        """Save job metadata to a JSON file."""
        file_path = self._get_jobs_dir() / f"{job_id}.json"
        
        # Ensure timestamp is present
        if 'updated_at' not in data:
            data['updated_at'] = datetime.datetime.utcnow().isoformat()
        if 'created_at' not in data and not file_path.exists():
            data['created_at'] = datetime.datetime.utcnow().isoformat()
            
        # If updating existing job, merge data
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
                existing_data.update(data)
                data = existing_data
            except Exception as e:
                logger.warning(f"Failed to read existing job data for {job_id}: {e}")

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job metadata by ID."""
        file_path = self._get_jobs_dir() / f"{job_id}.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read job {job_id}: {e}")
            return None

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all persisted jobs."""
        jobs = []
        jobs_dir = self._get_jobs_dir()
        for file_path in jobs_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    jobs.append(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to read job file {file_path}: {e}")
        
        # Sort by creation time (newest first)
        return sorted(jobs, key=lambda x: x.get('created_at', ''), reverse=True)

    def delete_job(self, job_id: str) -> bool:
        """Delete job metadata and associated files."""
        file_path = self._get_jobs_dir() / f"{job_id}.json"
        
        try:
            # Delete metadata file if it exists
            if file_path.exists():
                os.remove(file_path)
            
            # Delete output directory if it exists
            job_output_dir = Path(self.base_dir) / job_id
            if job_output_dir.exists() and job_output_dir.is_dir():
                import shutil
                shutil.rmtree(job_output_dir)
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        For now, we just mark it as failed in the metadata.
        In a more advanced implementation, we would kill the process.
        """
        job = self.get_job(job_id)
        if not job:
            return False
        
        if job.get('status') in ['running', 'submitted']:
            job['status'] = 'failed'
            job['error'] = 'Job cancelled by user'
            self.save_job(job_id, job)
            return True
        
        return False
    
    def validate_input(
        self,
        protein_pdb: Optional[str] = None,
        ligand_smiles: Optional[str] = None,
        ligand_structure: Optional[str] = None,
        ligand_format: str = "sdf"
    ) -> Dict[str, Any]:
        """Validate input data before optimization."""
        self._lazy_init()
        issues = []
        warnings = []
        
        # Validate protein
        if protein_pdb:
            pv = self.pdb_writer.validate_pdb_data(protein_pdb)
            if not pv['valid']:
                issues.extend(pv['issues'])
        else:
            issues.append("No protein PDB data provided")
        
        # Validate ligand
        if ligand_smiles and ligand_structure:
            issues.append("Provide either ligand_smiles OR ligand_structure, not both")
        elif not ligand_smiles and not ligand_structure:
            issues.append("No ligand data provided")
        
        # Validate environment
        if not self.environment_status.get('openff'):
            issues.append("OpenFF not available")
        if not self.environment_status.get('openmm'):
            issues.append("OpenMM not available")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
    
    def get_optimization_config(
        self,
        temperature: float = 300.0,
        pressure: float = 1.0,
        ionic_strength: float = 0.15,
        nvt_steps: int = 25000,
        npt_steps: int = 25000
    ) -> Dict[str, Any]:
        """Get optimization configuration."""
        self._lazy_init()
        return {
            "system": self.system_builder_config.get_system_config(ionic_strength, temperature, pressure),
            "minimization": self.minimization.get_minimization_config(),
            "equilibration": self.equilibration.get_equilibration_config(nvt_steps, npt_steps, temperature, pressure),
            "charge_assignment": self.charge_assignment.get_charge_config()
        }
    
    def estimate_runtime(
        self,
        protein_atoms: int,
        ligand_atoms: int,
        nvt_steps: int = 25000,
        npt_steps: int = 25000
    ) -> Dict[str, Any]:
        """Estimate runtime for optimization."""
        self._lazy_init()
        ss = self.system_builder_config.estimate_system_size(protein_atoms, ligand_atoms)
        et = self.equilibration.estimate_equilibration_time(nvt_steps, npt_steps)
        return {
            "system_size": ss,
            "equilibration": et,
            "total_estimated_hours": et["estimated_hours_cpu"] * 2
        }
    
    def prepare_ligand_from_smiles(
        self,
        smiles: str,
        ligand_id: str = "ligand",
        generate_conformer: bool = True,
        charge_method: str = "mmff94"
    ):
        """
        Prepare ligand from SMILES string.

        Returns OpenFF Molecule object or None if failed.
        """
        self._lazy_init()
        result = self.ligand_processor.prepare_ligand_from_smiles(
            smiles, ligand_id, generate_conformer, charge_method
        )
        if result['success']:
            return result['molecule']
        else:
            logger.error(f"Ligand preparation failed: {result['error']}")
            return None
    
    def prepare_ligand_from_structure(
        self,
        structure_data: str,
        ligand_id: str = "ligand",
        data_format: str = "sdf",
        preserve_pose: bool = True,
        charge_method: str = "mmff94"
    ):
        """
        Prepare ligand from structure data (SDF/MOL/PDB).

        Returns OpenFF Molecule object or None if failed.
        """
        self._lazy_init()
        result = self.ligand_processor.prepare_ligand_from_structure(
            structure_data, ligand_id, data_format, preserve_pose, charge_method
        )
        if result['success']:
            self._last_ligand_error = None
            return result['molecule']
        else:
            self._last_ligand_error = str(result.get("error") or "unknown ligand preparation error")
            logger.error(f"Ligand preparation failed: {self._last_ligand_error}")
            return None
    
    def prepare_protein(self, pdb_data: str, pdb_id: str = "protein") -> Optional[str]:
        """
        Prepare protein structure for MD simulation.
        
        Returns path to cleaned protein PDB file or None if failed.
        """
        self._lazy_init()
        logger.info(f"=== PREPARING PROTEIN {pdb_id} ===")
        
        try:
            # Parse structure to identify components
            structure = self.pdb_parser.parse_string(pdb_data, pdb_id)
            components = self.component_analyzer.identify_components(structure)
            
            if not components.get("protein"):
                logger.error("No protein residues found in structure")
                return None
            
            # Extract protein component
            protein_pdb = self.pdb_parser.extract_residues_as_string(structure, components["protein"])
            logger.info(f"Extracted {len(components['protein'])} protein residues")
            
            # Clean protein structure
            cleaning_result = self.protein_preparer.clean_structure_staged(
                protein_pdb,
                remove_heterogens=True,
                remove_water=True,
                add_missing_residues=True,
                add_missing_atoms=True,
                add_missing_hydrogens=True,
                ph=7.4,
                add_solvation=False,
                keep_ligands=False
            )
            
            # Get final cleaned stage
            stages = cleaning_result.get('stages', {})
            stage_info = cleaning_result.get('stage_info', {})
            
            if stages:
                final_stage = max(stage_info.items(), key=lambda x: x[1].get('step', 0))
                cleaned_pdb = stages[final_stage[0]]
            else:
                cleaned_pdb = protein_pdb
            
            # Save cleaned protein
            output_path = os.path.join(self.output_dir, f"{pdb_id}_cleaned.pdb")
            with open(output_path, 'w') as f:
                f.write(cleaned_pdb)
            
            logger.info(f"[COMPLETE] Protein preparation complete: {output_path}")
            return output_path
            
        except Exception as e:
            import traceback
            logger.error(f"Protein preparation failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def optimize(self, config: MDOptimizationConfig) -> Dict[str, Any]:
        """
        MD optimization workflow using config object.
        
        Args:
            config: MDOptimizationConfig with workflow parameters
        
        Returns:
            Dict with status, system info, and equilibration stats
        """
        self._lazy_init()
        logger.info("=== MD OPTIMIZATION WORKFLOW ===")
        logger.info(f"System ID: {config.system_id}")
        logger.info(f"Protein ID: {config.protein_id}")
        logger.info(f"Mode: {'protein-only' if config.is_protein_only else 'protein-ligand'}")
        if not config.is_protein_only:
            logger.info(f"Ligand ID: {config.ligand_id}")

        # Validate configuration
        valid, error = config.validate()
        if not valid:
            return {"status": "error", "error": error}

        try:
            # Step 1: Environment validation
            self._validate_environment()

            # Step 2: Ligand preparation (skipped in protein-only mode and amber-native mode)
            amber_native = str(getattr(config, "md_backend", "openmm_openff")).strip().lower() == "amber_native"
            if not config.is_protein_only and not amber_native:
                prepared_ligand = self._prepare_ligand(config)
                if not prepared_ligand:
                    return {"status": "error", "error": f"Ligand preparation failed for {config.ligand_id}"}
            else:
                logger.info("=== STEP 2: SKIPPED (protein-only or amber-native mode) ===")
            
            # Step 3 & 4: Protein preparation and System creation
            prepared_protein_path, system_result = self._prepare_and_create_system(config)
            if not prepared_protein_path or not system_result:
                return {"status": "error", "error": "Protein preparation or system creation failed"}
            
            # Check for preview pause
            if config.preview_before_equilibration and not config.preview_acknowledged:
                return self._handle_preview_pause(config, prepared_protein_path, system_result)
            
            # Step 5: Equilibration protocol
            equilibration_result = self._run_equilibration(config, system_result)
            if not equilibration_result:
                return {"status": "error", "error": "Equilibration failed"}
            
            # Handle minimization-only or paused states
            if equilibration_result.get("status") == "minimized_ready":
                return clean_results_for_json(equilibration_result)
            
            if equilibration_result.get("minimization_only"):
                return clean_results_for_json(equilibration_result)
            
            # Combine and return results
            return self._combine_results(config, prepared_protein_path, system_result, equilibration_result)
            
        except Exception as e:
            import traceback
            logger.error(f"MD optimization workflow failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return clean_results_for_json({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    def _validate_environment(self) -> None:
        """Validate environment has required dependencies."""
        logger.info("=== STEP 1: ENVIRONMENT VALIDATION ===")
        req_check = EnvironmentValidator.check_minimum_requirements()
        if not req_check['met']:
            raise RuntimeError(f"Missing dependencies: {req_check['missing']}")
    
    def _prepare_ligand(self, config: MDOptimizationConfig) -> Any:
        """Prepare ligand from SMILES or structure data. Result is cached for reuse."""
        if config.ligand_smiles:
            logger.info("=== STEP 2: LIGAND PREPARATION (SMILES-BASED) ===")
            logger.info(f"Using charge method: {config.charge_method}")
            prepared_ligand = self.prepare_ligand_from_smiles(
                config.ligand_smiles, config.ligand_id, config.generate_conformer,
                config.charge_method
            )
        else:
            logger.info("=== STEP 2: LIGAND PREPARATION (STRUCTURE-BASED) ===")
            logger.info(f"Using charge method: {config.charge_method}")
            prepared_ligand = self.prepare_ligand_from_structure(
                config.ligand_structure_data, config.ligand_id,
                config.ligand_data_format, config.preserve_ligand_pose,
                config.charge_method
            )

        valid, error = validate_ligand_preparation(prepared_ligand, config.ligand_id)
        if not valid:
            if self._last_ligand_error:
                raise RuntimeError(f"{error}: {self._last_ligand_error}")
            raise RuntimeError(error)

        self._cached_prepared_ligand = prepared_ligand
        return prepared_ligand
    
    def _prepare_and_create_system(self, config: MDOptimizationConfig) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Prepare protein and create solvated system."""
        if str(getattr(config, "md_backend", "openmm_openff")).strip().lower() == "amber_native":
            logger.info("=== AMBER-NATIVE MODE: BUILDING SIMULATION FROM PRMTOP/INPCRD ===")
            amber_result = self._create_system_from_amber_artifacts(config)
            valid, error = validate_system_result(amber_result)
            if not valid:
                raise RuntimeError(error)
            prepared_protein_path = str(
                getattr(config, "amber_system_pdb_path", None)
                or getattr(config, "resume_system_pdb_path", None)
                or ""
            )
            return prepared_protein_path, amber_result

        # Checkpoint-resume path: rebuild simulation directly from original system PDB,
        # bypassing protein cleanup on NPT/final snapshots.
        if getattr(config, "resume_from_checkpoint_path", None):
            resume_system_pdb_path = getattr(config, "resume_system_pdb_path", None)
            if not resume_system_pdb_path or not os.path.exists(str(resume_system_pdb_path)):
                raise RuntimeError(
                    "Checkpoint resume requested but resume_system_pdb_path is missing or unreadable."
                )
            logger.info("=== RESUME MODE: REBUILD SYSTEM FOR CHECKPOINT LOAD ===")
            logger.info(f"Using system PDB from prep run: {resume_system_pdb_path}")
            with open(str(resume_system_pdb_path), "r") as f:
                system_pdb_data = f.read()

            if config.is_protein_only:
                system_result = self.solvated_system_builder.recreate_system_from_pdb_protein_only(
                    system_pdb_data, config.system_id,
                    temperature=config.temperature,
                    pressure=config.pressure
                )
            else:
                system_result = self.solvated_system_builder.recreate_system_from_pdb(
                    system_pdb_data, self._get_prepared_ligand(config), config.system_id,
                    config.forcefield_method,
                    temperature=config.temperature,
                    pressure=config.pressure
                )
            # Keep a meaningful provenance pointer even though protein prep is bypassed.
            prepared_protein_path = str(resume_system_pdb_path)
            valid, error = validate_system_result(system_result)
            if not valid:
                raise RuntimeError(error)
            return prepared_protein_path, system_result

        if config.preview_acknowledged or config.minimized_acknowledged:
            # Resuming workflow
            logger.info("=== RESUMING WORKFLOW ===")
            logger.info(f"Using force field method: {config.forcefield_method}")
            prepared_protein_path = os.path.join(self.output_dir, f"{config.protein_id}_cleaned.pdb")

            # Load the solvated system PDB that was saved during preview
            system_pdb_path = os.path.join(self.output_dir, f"{config.system_id}_system.pdb")
            if not os.path.exists(system_pdb_path):
                raise RuntimeError(f"Solvated system PDB not found at {system_pdb_path}. Cannot resume workflow.")

            with open(system_pdb_path, 'r') as f:
                system_pdb_data = f.read()

            logger.info(f"Loading solvated system from {system_pdb_path}")
            if config.is_protein_only:
                system_result = self.solvated_system_builder.recreate_system_from_pdb_protein_only(
                    system_pdb_data, config.system_id,
                    temperature=config.temperature,
                    pressure=config.pressure
                )
            else:
                system_result = self.solvated_system_builder.recreate_system_from_pdb(
                    system_pdb_data, self._get_prepared_ligand(config), config.system_id,
                    config.forcefield_method,
                    temperature=config.temperature,
                    pressure=config.pressure
                )
        else:
            # Normal flow
            logger.info("=== STEP 3: PROTEIN PREPARATION ===")
            prepared_protein_path = self.prepare_protein(config.protein_pdb_data, config.protein_id)
            valid, error = validate_protein_preparation(prepared_protein_path, config.protein_id)
            if not valid:
                raise RuntimeError(error)

            logger.info("=== STEP 4: SYSTEM CREATION AND SOLVATION ===")
            logger.info(f"Box shape: {config.box_shape}, padding: {config.padding_nm} nm")
            if config.is_protein_only:
                logger.info("Using AMBER14 force field (protein-only mode)")
                system_result = self.solvated_system_builder.create_solvated_system_protein_only(
                    config.protein_pdb_data, config.protein_id, config.system_id,
                    ionic_strength_m=config.ionic_strength,
                    padding_nm=config.padding_nm,
                    box_shape=config.box_shape,
                    temperature=config.temperature,
                    pressure=config.pressure
                )
            else:
                logger.info(f"Using force field method: {config.forcefield_method}")
                system_result = self.solvated_system_builder.create_solvated_system(
                    config.protein_pdb_data, self._get_prepared_ligand(config),
                    config.protein_id, config.ligand_id, config.system_id,
                    padding_nm=config.padding_nm,
                    ionic_strength_m=config.ionic_strength,
                    forcefield_method=config.forcefield_method,
                    box_shape=config.box_shape,
                    temperature=config.temperature,
                    pressure=config.pressure
                )

        valid, error = validate_system_result(system_result)
        if not valid:
            raise RuntimeError(error)

        return prepared_protein_path, system_result

    def _create_system_from_amber_artifacts(self, config: MDOptimizationConfig) -> Dict[str, Any]:
        from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, unit
        from openmm.app import AmberPrmtopFile, AmberInpcrdFile, HBonds, PME, CutoffNonPeriodic, PDBFile

        def _resolve(path_value: Optional[str]) -> Optional[str]:
            p = str(path_value or "").strip()
            if not p:
                return None
            if os.path.exists(p):
                return p
            marker = "ovo-ligand/"
            if marker in p:
                mapped = os.path.join("/ovo-ligand", p.split(marker, 1)[1])
                if os.path.exists(mapped):
                    return mapped
            if p.startswith("/output/"):
                mapped = os.path.join(self.output_dir, p.removeprefix("/output/"))
                if os.path.exists(mapped):
                    return mapped
            return None

        prmtop_path = _resolve(getattr(config, "amber_complex_prmtop_path", None))
        inpcrd_path = _resolve(getattr(config, "amber_complex_inpcrd_path", None))
        if not prmtop_path or not inpcrd_path:
            return {
                "status": "error",
                "error": (
                    "Amber-native backend requires readable amber_complex_prmtop_path and "
                    "amber_complex_inpcrd_path."
                ),
            }

        prmtop = AmberPrmtopFile(prmtop_path)
        inpcrd = AmberInpcrdFile(inpcrd_path)
        periodic = inpcrd.boxVectors is not None
        openmm_system = prmtop.createSystem(
            nonbondedMethod=(PME if periodic else CutoffNonPeriodic),
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=HBonds,
            rigidWater=True,
        )
        if periodic:
            openmm_system.addForce(
                MonteCarloBarostat(
                    float(config.pressure) * unit.bar,
                    float(config.temperature) * unit.kelvin,
                    25,
                )
            )
        integrator = LangevinMiddleIntegrator(
            float(config.temperature) * unit.kelvin,
            1.0 / unit.picosecond,
            0.004 * unit.picoseconds,
        )
        simulation, platform_name = self.solvated_system_builder._create_simulation_with_fallback(
            prmtop.topology, openmm_system, integrator
        )
        simulation.context.setPositions(inpcrd.positions)
        if inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

        system_pdb_path = _resolve(getattr(config, "amber_system_pdb_path", None))
        if not system_pdb_path:
            system_pdb_path = os.path.join(self.output_dir, f"{config.system_id}_amber_system.pdb")
            with open(system_pdb_path, "w") as handle:
                PDBFile.writeFile(prmtop.topology, inpcrd.positions, handle, keepIds=True)

        return {
            "status": "success",
            "simulation": simulation,
            "platform": platform_name,
            "total_atoms": prmtop.topology.getNumAtoms(),
            "system_pdb_path": system_pdb_path,
            "system_info": {
                "total_atoms": prmtop.topology.getNumAtoms(),
                "residues": prmtop.topology.getNumResidues(),
                "chains": prmtop.topology.getNumChains(),
                "source": "amber_prmtop_inpcrd",
                "periodic": bool(periodic),
            },
        }
    
    def _get_prepared_ligand(self, config: MDOptimizationConfig) -> Any:
        """Get prepared ligand, using cached result from _prepare_ligand() if available."""
        if hasattr(self, '_cached_prepared_ligand') and self._cached_prepared_ligand is not None:
            logger.info("Using cached prepared ligand (skipping redundant preparation)")
            return self._cached_prepared_ligand
        # Fallback: prepare fresh (e.g. resume workflow where _prepare_ligand was called in a previous task)
        return self._prepare_ligand(config)
    
    def _handle_preview_pause(self, config: MDOptimizationConfig, 
                             prepared_protein_path: str, 
                             system_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle preview pause in workflow."""
        logger.info("Preview option requested - pausing workflow.")
        return clean_results_for_json({
            "status": "preview_ready",
            "workflow_stage": "system_prepared",
            "message": "System PDB is ready for inspection. Re-run with preview_acknowledged=True to continue.",
            "system_id": config.system_id,
            "protein_id": config.protein_id,
            **({"ligand_id": config.ligand_id} if not config.is_protein_only else {}),
            "total_atoms": system_result.get('total_atoms', 0),
            "system_info": system_result.get('system_info', {}),
            "output_files": {
                "protein_prepared": prepared_protein_path,
                "system_pdb": system_result.get('system_pdb_path')
            }
        })
    
    def _run_equilibration(self, config: MDOptimizationConfig, 
                          system_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run equilibration protocol."""
        logger.info("=== STEP 5: EQUILIBRATION PROTOCOL ===")
        
        if "simulation" not in system_result:
            raise RuntimeError("System result missing 'simulation' key")
        
        equilibration_result = self.equilibration_runner.run_equilibration_protocol(
            system_result["simulation"],
            config.system_id,
            nvt_steps=config.nvt_steps,
            npt_steps=config.npt_steps,
            heating_steps_per_stage=getattr(config, "heating_steps_per_stage", 2500),
            heating_start_temperature=getattr(config, "heating_start_temperature", 50.0),
            heating_stages=getattr(config, "heating_stages", 6),
            pause_at_minimized=config.pause_at_minimized,
            minimization_only=config.minimization_only,
            skip_minimization=config.minimized_acknowledged,
            production_steps=config.production_steps,
            production_report_interval=config.production_report_interval,
            minimization_max_iterations=getattr(config, "minimization_max_iterations", 5000),
            minimization_tolerance_kjmol_nm=getattr(config, "minimization_tolerance_kjmol_nm", 10.0),
            npt_restraint_release_scales_csv=getattr(config, "npt_restraint_release_scales", "1.0,0.5,0.2,0.05,0.0"),
            npt_release_enabled=getattr(config, "npt_release_enabled", True),
            protein_npt_release_scales_csv=getattr(config, "protein_npt_release_scales", "1.0,0.5,0.1,0.01,0.0"),
            planarity_npt_release_scales_csv=getattr(config, "planarity_npt_release_scales", "1.0,0.5,0.2,0.05,0.0"),
            allow_restrained_production=getattr(config, "allow_restrained_production", False),
            force_unrestrained_production=getattr(config, "force_unrestrained_production", True),
            resume_from_checkpoint_path=getattr(config, "resume_from_checkpoint_path", None),
            resume_state_xml_path=getattr(config, "resume_state_xml_path", None),
            resume_system_xml_path=getattr(config, "resume_system_xml_path", None),
            resume_integrator_xml_path=getattr(config, "resume_integrator_xml_path", None),
            production_only_from_prepared=bool(getattr(config, "production_only_from_prepared", False)),
            temperature=config.temperature,
            pressure=config.pressure
        )
        
        valid, error = validate_equilibration_result(equilibration_result)
        if not valid:
            raise RuntimeError(error)
        
        return equilibration_result
    
    def _combine_results(self, config: MDOptimizationConfig, 
                        prepared_protein_path: str,
                        system_result: Dict[str, Any],
                        equilibration_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all results into final output."""
        result = {
            "status": "success",
            "system_id": config.system_id,
            "protein_id": config.protein_id,
            **({"ligand_id": config.ligand_id} if not config.is_protein_only else {}),
            "total_atoms": system_result.get('total_atoms', 0),
            "system_info": system_result.get('system_info', {}),
            "equilibration_stats": equilibration_result.get('equilibration_stats', {}),
            "output_files": {
                "protein_prepared": prepared_protein_path,
                "system_pdb": system_result.get('system_pdb_path'),
            }
        }

        # Preserve ligand assembly/QC audit data produced during system construction.
        for key in (
            "ligand_coordinate_lock",
            "ligand_coordinate_lock_qc",
            "ligand_assembly_qc",
            "ligand_assembly_qc_after_enforcement",
            "ligand_final_enforcement",
            "protein_positional_restraints",
            "ligand_positional_restraints",
            "ligand_planarity_restraints",
        ):
            if key in system_result:
                result[key] = system_result[key]
        
        # Merge equilibration output files
        if "output_files" in equilibration_result:
            result["output_files"].update(equilibration_result["output_files"])

        # Post-hoc analytics: parse log + compute RMSD from all trajectory phases.
        # Wrapped in try/except so a completed simulation result is never lost
        # due to an analytics bug.
        try:
            output_files = result["output_files"]
            
            # Use production PDB if available, otherwise NPT PDB
            topology_pdb = output_files.get("production_pdb") or output_files.get("npt_pdb")
            
            # Use production log if available, otherwise equilibration log
            log_path = output_files.get("production_log") or output_files.get("equilibration_log")
            
            analytics = EquilibrationAnalytics().compute(
                output_dir=self.output_dir,
                system_id=config.system_id,
                topology_pdb=topology_pdb,
                nvt_traj=output_files.get("nvt_trajectory"),
                npt_traj=output_files.get("npt_trajectory"),
                production_traj=output_files.get("production_trajectory"),
                log_path=log_path,
                ligand_id=config.ligand_id if not config.is_protein_only else "",
                nvt_steps=config.nvt_steps,
                npt_steps=config.npt_steps,
                production_steps=config.production_steps,
                nvt_report_interval=1000,
                npt_report_interval=1000,
                production_report_interval=config.production_report_interval,
                dt_ps=0.004,
            )
            result["analytics"] = analytics
        except Exception as exc:
            logger.warning(f"[ANALYTICS] Post-hoc analytics failed: {exc}", exc_info=True)
            result["analytics"] = None

        logger.info("[COMPLETE] Complete MD optimization workflow finished successfully")
        return clean_results_for_json(result)
    
    def process_trajectory(
        self,
        dcd_path: str,
        pdb_path: str,
        stride: int = 20,
        align: bool = True,
        remove_solvent_flag: bool = True,
        include_unitcell: bool = True
    ) -> Dict[str, Any]:
        """
        Process trajectory for web delivery.
        
        Returns dict with 'pdb_data', 'unitcell_data', and 'error'.
        """
        self._lazy_init()
        return self.trajectory_runner.process_trajectory(
            dcd_path, pdb_path, stride, align, remove_solvent_flag, include_unitcell
        )
    
    def get_trajectory_files(self, system_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Get trajectory and topology file paths for a system."""
        self._lazy_init()
        return self.trajectory_runner.get_trajectory_files(system_id)


__all__ = ['MDOptimizationService']
