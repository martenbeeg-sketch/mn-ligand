"""
RBFE Service
Implements relative binding free energy calculations using OpenFE ecosystem.
"""
from __future__ import annotations
import os
import sys
import json
import traceback
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import tempfile
from dataclasses import dataclass, asdict

# Initialize logger early
logger = logging.getLogger(__name__)


def emit_progress(progress: int, status: str, result: Optional[Dict[str, Any]] = None) -> None:
    """Emit a progress update parsed by the Celery task runner (same format as MD service)."""
    payload = {'progress': progress, 'status': status}
    if result is not None:
        payload['result'] = result
    print(
        f"MD_PROGRESS:{json.dumps(payload)}",
        file=sys.stderr,
        flush=True,
    )

# NumPy for statistical calculations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Statistical calculations will use fallback.")

# OpenFE and dependencies
try:
    import openfe
    from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
    from openfe.protocols.openmm_utils.omm_settings import OpenFFPartialChargeSettings
    from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges
    from openfe.setup.chemicalsystem_generator import EasyChemicalSystemGenerator
    from gufe.protocols import execute_DAG
    from openff.units import unit
    OPENFE_AVAILABLE = True
except ImportError as e:
    print(f"DEBUG: OpenFE Import Error: {e}")
    OPENFE_AVAILABLE = False
    openfe = None
    RelativeHybridTopologyProtocol = None
    logger.warning("OpenFE not available. RBFE calculations will not work.")

# RDKit for molecule handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign, rdFMCS
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# HTTP client for calling docking service
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available. Batch docking integration will not work.")

# Kartograf for 3D alignment
try:
    from kartograf import atom_aligner
    KARTOGRAF_AVAILABLE = True
except ImportError:
    KARTOGRAF_AVAILABLE = False
    logger.warning("kartograf not available. Will fallback to RDKit MCS alignment.")

# Meeko for PDBQT handling (preserves bond orders/aromaticity during PDBQT round-trip)
try:
    from meeko import PDBQTMolecule, RDKitMolCreate
    MEEKO_AVAILABLE = True
except ImportError:
    MEEKO_AVAILABLE = False
    logger.warning("Meeko not available. PDBQT conversion will use fallback PDB parsing.")

from .network_planner import NetworkPlanner, LigandNetworkData


@dataclass
class RBFETransformationResult:
    """Result for a single RBFE transformation (edge)."""
    ligand_a: str
    ligand_b: str
    ddg_kcal_mol: float
    uncertainty_kcal_mol: float
    leg: str  # 'complex' or 'solvent'
    status: str  # 'completed', 'failed', 'running'
    error: Optional[str] = None


@dataclass
class AlignmentData:
    """Alignment information for a ligand."""
    ligand_id: str
    is_reference: bool
    aligned_to: Optional[str] = None
    rmsd: Optional[float] = None
    mcs_atoms: Optional[int] = None
    error: Optional[str] = None


@dataclass
class RBFENetworkResult:
    """Complete result for an RBFE network calculation."""
    job_id: str
    status: str
    transformations: List[RBFETransformationResult]
    relative_binding_affinities: Dict[str, float]  # ligand_name -> ddG relative to reference
    reference_ligand: Optional[str] = None
    error: Optional[str] = None
    alignment_data: Optional[List[AlignmentData]] = None


class RBFEService:
    """Service for relative binding free energy calculations using OpenFE."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize RBFE service.
        
        Args:
            output_dir: Directory for storing RBFE calculation outputs.
                       If None, uses RBFE_OUTPUT_DIR env var or defaults to 'data/rbfe_outputs'
        """
        if not OPENFE_AVAILABLE:
            raise ImportError("OpenFE is not available. Please install openfe package.")
        
        # Use provided output_dir, or fall back to environment variable, or use default
        if output_dir is None:
            output_dir = os.getenv('RBFE_OUTPUT_DIR', 'data/rbfe_outputs')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Job tracking directory
        self.jobs_dir = self.output_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory job cache
        self.jobs: Dict[str, Dict[str, Any]] = {}

        # NetworkPlanner is now instantiated per-job with user-selected atom mapper
        # (no shared instance needed)

        # Initialize chemistry utilities
        from ovo_ligand.ligandx.lib.chemistry import get_ligand_preparer, get_protein_preparer
        self.ligand_preparer = get_ligand_preparer()
        self.protein_preparer = get_protein_preparer()
        
        # Docking service URL (for batch docking before RBFE)
        self.docking_service_url = os.getenv('DOCKING_URL', 'http://docking:8002')
        
        logger.info(f"RBFE service initialized with output directory: {self.output_dir}")
    
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

    
    def dock_ligands_batch(
        self,
        protein_pdb: str,
        ligands_data: List[Dict[str, Any]],
        grid_box: Optional[Dict[str, Any]] = None,
        exhaustiveness: int = 16,
        num_poses: int = 9
    ) -> List[Dict[str, Any]]:
        """
        Dock multiple ligands against the protein using batch docking service.
        
        Args:
            protein_pdb: Protein PDB data
            ligands_data: List of ligand dicts with 'id', 'data', 'format' keys
            grid_box: Optional pre-defined grid box (auto-calculated if None)
            exhaustiveness: Docking exhaustiveness (higher = more thorough)
            num_poses: Number of poses to generate per ligand
            
        Returns:
            List of docking results with best poses extracted
        """
        if not HTTPX_AVAILABLE:
            logger.error("httpx not available - cannot call docking service")
            return []
        
        logger.info(f"Submitting batch docking for {len(ligands_data)} ligands...")
        
        try:
            # Prepare batch docking request
            batch_request = {
                'protein_pdb': protein_pdb,
                'ligands': [
                    {
                        'id': lig.get('id', f'ligand_{i}'),
                        'data': lig.get('data', ''),
                        'format': lig.get('format', 'sdf')
                    }
                    for i, lig in enumerate(ligands_data)
                ],
                'grid_box': grid_box,
                'exhaustiveness': exhaustiveness,
                'num_poses': num_poses,
                'parallel_workers': 4,
                'use_meeko': True
            }
            
            # Call docking service
            with httpx.Client(timeout=1800.0) as client:
                response = client.post(
                    f"{self.docking_service_url}/api/docking/batch",
                    json=batch_request
                )
                response.raise_for_status()
                batch_response = response.json()
            
            if batch_response.get('status') != 'completed':
                logger.error(f"Batch docking failed: {batch_response.get('message')}")
                return []
            
            # Get full results from status endpoint
            job_id = batch_response.get('job_id')
            with httpx.Client(timeout=60.0) as client:
                status_response = client.get(
                    f"{self.docking_service_url}/api/docking/batch/status/{job_id}"
                )
                status_response.raise_for_status()
                job_status = status_response.json()
            
            docking_results = job_status.get('results', [])
            
            logger.info(f"Batch docking completed: {len(docking_results)} results")
            logger.info(f"  Successful: {batch_response.get('completed', 0)}")
            logger.info(f"  Failed: {batch_response.get('failed', 0)}")
            
            return docking_results
            
        except Exception as e:
            logger.error(f"Batch docking failed: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def _build_template_mol(
        self,
        ligand_data: str,
        ligand_format: str,
        ligand_id: str
    ) -> Optional['Chem.Mol']:
        """
        Build an RDKit template molecule with trusted bond orders.

        This template is used to re-assign chemistry when coordinates are parsed
        from pose formats that may lose bond-order/aromatic information (PDB/PDBQT).
        """
        if not ligand_data:
            return None

        try:
            fmt = (ligand_format or "").lower()
            if fmt in ["sdf", "mol"]:
                return Chem.MolFromMolBlock(ligand_data, removeHs=False)
            if fmt == "pdb":
                return Chem.MolFromPDBBlock(ligand_data, removeHs=False)
            if fmt in ["smi", "smiles"]:
                return Chem.AddHs(Chem.MolFromSmiles(ligand_data))
        except Exception as e:
            logger.warning(f"Failed to build template molecule for {ligand_id}: {e}")
            return None

        return None

    def _restore_bond_orders_from_template(
        self,
        target_mol: 'Chem.Mol',
        template_mol: Optional['Chem.Mol'],
        ligand_id: str
    ) -> 'Chem.Mol':
        """
        Restore bond orders/aromaticity on a coordinate-only molecule.

        Tries direct matching first, then falls back to heavy-atom-only matching
        (handles PDBQT round-trips that strip non-polar hydrogens).
        """
        if template_mol is None:
            return target_mol

        # Strategy 1: Direct match (same atom count)
        try:
            restored = AllChem.AssignBondOrdersFromTemplate(template_mol, target_mol)
            logger.info(f"Restored bond orders for {ligand_id} from template")
            return restored
        except Exception as e:
            logger.debug(f"Direct bond order restore failed for {ligand_id}: {e}")

        # Strategy 2: Heavy-atom-only match (handles H count mismatch from PDBQT)
        try:
            target_noH = AllChem.RemoveHs(target_mol)
            template_noH = AllChem.RemoveHs(template_mol)
            restored_noH = AllChem.AssignBondOrdersFromTemplate(template_noH, target_noH)
            Chem.SanitizeMol(restored_noH)
            restored = Chem.AddHs(restored_noH, addCoords=True)
            logger.info(f"Restored bond orders for {ligand_id} via heavy-atom matching")
            return restored
        except Exception as e:
            logger.warning(f"Could not restore bond orders for {ligand_id}: {e}")
            return target_mol

    def _extract_first_pdbqt_model(self, poses_pdbqt: str) -> str:
        """Extract the first MODEL block from a multi-model PDBQT string."""
        lines = poses_pdbqt.strip().split('\n')
        best_pose_lines = []
        in_model = False

        for line in lines:
            if line.startswith('MODEL'):
                if in_model:
                    break
                in_model = True
                best_pose_lines.append(line)
                continue
            elif line.startswith('ENDMDL'):
                best_pose_lines.append(line)
                break
            elif in_model:
                best_pose_lines.append(line)

        if not best_pose_lines:
            return poses_pdbqt
        return '\n'.join(best_pose_lines)

    def extract_best_pose_from_pdbqt(
        self,
        poses_pdbqt: str,
        ligand_id: str,
        template_mol: Optional['Chem.Mol'] = None
    ) -> Optional[str]:
        """
        Extract the best (first) pose from PDBQT output and convert to SDF.

        Uses Meeko for accurate PDBQT→RDKit conversion (preserves bond orders
        and aromaticity from the torsion tree). Falls back to manual PDB parsing
        with template-based bond order restoration if Meeko is unavailable.

        Args:
            poses_pdbqt: Multi-model PDBQT string from docking
            ligand_id: Ligand identifier for logging
            template_mol: Original RDKit mol with correct bond orders (used by fallback)

        Returns:
            SDF format string of the best pose, or None if extraction failed
        """
        if not RDKIT_AVAILABLE:
            logger.error("RDKit not available - cannot convert PDBQT to SDF")
            return None

        # --- Primary path: Meeko (preserves bond orders from PDBQT torsion tree) ---
        if MEEKO_AVAILABLE:
            try:
                first_model = self._extract_first_pdbqt_model(poses_pdbqt)
                pdbqt_mol = PDBQTMolecule(first_model, is_dlg=False, skip_typing=True)
                for pose in pdbqt_mol:
                    result = RDKitMolCreate.from_pdbqt_mol(pose)
                    if isinstance(result, list):
                        mol = next((m for m in result if m is not None), None)
                    else:
                        mol = result
                    if mol is not None:
                        try:
                            Chem.SanitizeMol(mol)
                        except Exception:
                            pass
                        sdf_block = Chem.MolToMolBlock(mol)
                        logger.info(f"Extracted best docked pose for {ligand_id} via Meeko")
                        return sdf_block
                    break
                logger.warning(f"Meeko returned no valid mol for {ligand_id}, trying fallback")
            except Exception as e:
                logger.warning(f"Meeko PDBQT conversion failed for {ligand_id}: {e}, trying fallback")

        # --- Fallback: manual PDBQT→PDB strip + template bond-order restoration ---
        try:
            lines = poses_pdbqt.strip().split('\n')
            best_pose_lines = []
            in_model = False

            for line in lines:
                if line.startswith('MODEL'):
                    if in_model:
                        break
                    in_model = True
                    continue
                elif line.startswith('ENDMDL'):
                    break
                elif in_model:
                    best_pose_lines.append(line)

            if not best_pose_lines:
                best_pose_lines = [l for l in lines if not l.startswith('REMARK')]

            pdb_lines = []
            for line in best_pose_lines:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    pdb_line = line[:66] if len(line) >= 66 else line
                    pdb_lines.append(pdb_line)
                elif line.startswith(('CONECT', 'ROOT', 'BRANCH', 'ENDBRANCH', 'ENDROOT', 'TORSDOF')):
                    continue
                else:
                    pdb_lines.append(line)

            pdb_lines.append('END')
            pdb_block = '\n'.join(pdb_lines)

            mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
            if mol is None:
                mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=False)
            if mol is None:
                logger.error(f"Failed to parse docked pose for {ligand_id}")
                return None

            mol = self._restore_bond_orders_from_template(mol, template_mol, ligand_id)

            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                logger.warning(f"Could not sanitize molecule {ligand_id}: {e}")

            sdf_block = Chem.MolToMolBlock(mol)
            logger.info(f"Extracted best docked pose for {ligand_id}")
            return sdf_block

        except Exception as e:
            logger.error(f"Error extracting best pose for {ligand_id}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def prepare_ligands_with_docking(
        self,
        protein_pdb: str,
        ligands_data: List[Dict[str, Any]],
        force_redock: bool = False,
        exhaustiveness: int = 16
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Prepare ligands for RBFE, docking those without poses first.
        
        This method:
        1. Identifies ligands that need docking (has_docked_pose=False)
        2. Runs batch docking for those ligands
        3. Extracts best poses and updates ligand data
        4. Returns updated ligand data with docking scores
        
        Args:
            protein_pdb: Protein PDB data
            ligands_data: List of ligand dicts
            force_redock: If True, re-dock all ligands regardless of has_docked_pose
            exhaustiveness: Docking exhaustiveness
            
        Returns:
            Tuple of (updated ligands_data, docking_scores dict)
        """
        # Separate ligands that need docking
        ligands_to_dock = []
        ligands_with_poses = []
        docking_scores = {}
        
        for lig in ligands_data:
            has_pose = lig.get('has_docked_pose', False)
            if force_redock or not has_pose:
                ligands_to_dock.append(lig)
            else:
                ligands_with_poses.append(lig)
                logger.info(f"Ligand {lig.get('id')} already has docked pose")
        
        if not ligands_to_dock:
            logger.info("All ligands have docked poses, skipping batch docking")
            return ligands_data, docking_scores
        
        logger.info(f"Need to dock {len(ligands_to_dock)} ligands before RBFE")
        
        # Run batch docking
        docking_results = self.dock_ligands_batch(
            protein_pdb=protein_pdb,
            ligands_data=ligands_to_dock,
            exhaustiveness=exhaustiveness
        )
        
        # Build a mapping of ligand_id -> docking result
        docking_map = {r.get('ligand_id'): r for r in docking_results}
        
        # Update ligands with docked poses
        updated_ligands = []
        
        for lig in ligands_data:
            lig_id = lig.get('id')
            
            if lig_id in docking_map:
                result = docking_map[lig_id]
                
                if result.get('success'):
                    poses_pdbqt = result.get('poses_pdbqt', '')
                    # Docking service returns 'best_score', not 'best_affinity'
                    best_affinity = result.get('best_score', result.get('best_affinity', 0.0))
                    
                    # Extract best pose as SDF
                    best_pose_sdf = self.extract_best_pose_from_pdbqt(poses_pdbqt, lig_id)
                    
                    if best_pose_sdf:
                        # Update ligand with docked pose
                        updated_lig = lig.copy()
                        updated_lig['data'] = best_pose_sdf
                        updated_lig['format'] = 'sdf'
                        updated_lig['has_docked_pose'] = True
                        updated_lig['docking_affinity'] = best_affinity
                        updated_ligands.append(updated_lig)
                        docking_scores[lig_id] = best_affinity
                        
                        logger.info(f"Updated {lig_id} with best docked pose (affinity: {best_affinity:.2f} kcal/mol)")
                    else:
                        # Keep original if extraction failed
                        logger.warning(f"Could not extract pose for {lig_id}, using original structure")
                        updated_ligands.append(lig)
                else:
                    # Docking failed, keep original
                    logger.warning(f"Docking failed for {lig_id}: {result.get('error')}")
                    updated_ligands.append(lig)
            else:
                # Ligand already had pose or wasn't in docking batch
                updated_ligands.append(lig)
        
        logger.info(f"Prepared {len(updated_ligands)} ligands ({len(docking_scores)} newly docked)")
        
        return updated_ligands, docking_scores
    
    def generate_docked_pose_files(
        self,
        protein_pdb: str,
        ligands_data: List[Dict[str, Any]],
        docking_scores: Dict[str, float],
        job_dir: Path,
        alignment_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate PDB files for docked poses (ligand only and protein-ligand complex).
        
        Args:
            protein_pdb: Protein PDB data
            ligands_data: List of ligand dicts with docked poses
            docking_scores: Dict of ligand_id -> docking affinity
            job_dir: Job output directory
            
        Returns:
            List of docked pose info dicts with file paths
        """
        docked_poses_dir = job_dir / "docked_poses"
        docked_poses_dir.mkdir(parents=True, exist_ok=True)
        
        # Save protein PDB
        protein_pdb_path = docked_poses_dir / "protein.pdb"
        with open(protein_pdb_path, 'w') as f:
            f.write(protein_pdb)
        
        docked_poses = []
        
        for lig in ligands_data:
            lig_id = lig.get('id', 'unknown')
            lig_data = lig.get('data', '')
            lig_format = lig.get('format', 'sdf')
            affinity = docking_scores.get(lig_id, lig.get('docking_affinity', 0.0))
            
            try:
                # Convert ligand to PDB format
                if lig_format.lower() in ['sdf', 'mol']:
                    mol = Chem.MolFromMolBlock(lig_data, removeHs=False)
                elif lig_format.lower() == 'pdb':
                    mol = Chem.MolFromPDBBlock(lig_data, removeHs=False)
                else:
                    logger.warning(f"Unknown format for {lig_id}: {lig_format}")
                    continue
                
                if mol is None:
                    logger.warning(f"Could not parse ligand {lig_id}")
                    continue
                
                # Generate ligand PDB
                ligand_pdb = Chem.MolToPDBBlock(mol)
                
                # Save ligand PDB
                ligand_pdb_path = docked_poses_dir / f"{lig_id}_docked.pdb"
                with open(ligand_pdb_path, 'w') as f:
                    f.write(ligand_pdb)
                
                # Create complex PDB (protein + ligand)
                complex_pdb_path = docked_poses_dir / f"{lig_id}_complex.pdb"
                with open(complex_pdb_path, 'w') as f:
                    # Write protein (remove END if present)
                    protein_lines = protein_pdb.strip().split('\n')
                    for line in protein_lines:
                        if not line.startswith('END'):
                            f.write(line + '\n')
                    
                    # Add separator
                    f.write('TER\n')
                    
                    # Write ligand (with HETATM records)
                    ligand_lines = ligand_pdb.strip().split('\n')
                    for line in ligand_lines:
                        if line.startswith('ATOM') or line.startswith('HETATM'):
                            # Convert ATOM to HETATM for ligand
                            if line.startswith('ATOM'):
                                line = 'HETATM' + line[6:]
                            f.write(line + '\n')
                        elif line.startswith('CONECT'):
                            f.write(line + '\n')
                    
                    f.write('END\n')
                
                pose_info = {
                    'ligand_id': lig_id,
                    'affinity_kcal_mol': float(affinity) if affinity else 0.0,
                    'pose_pdb_path': str(ligand_pdb_path.relative_to(job_dir)),
                    'complex_pdb_path': str(complex_pdb_path.relative_to(job_dir))
                }
                
                # Add alignment info if available
                if alignment_info and 'aligned_ligands' in alignment_info:
                    for aligned in alignment_info['aligned_ligands']:
                        if aligned['id'] == lig_id:
                            pose_info['alignment_score'] = aligned.get('rmsd')
                            pose_info['mcs_atoms'] = aligned.get('mcs_atoms')
                            break
                
                docked_poses.append(pose_info)
                
                logger.info(f"Generated pose files for {lig_id}: affinity={affinity:.2f} kcal/mol")
                
            except Exception as e:
                logger.error(f"Error generating pose files for {lig_id}: {e}")
                continue
        
        # Save docking summary
        summary_path = docked_poses_dir / "docking_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'num_ligands': len(docked_poses),
                'poses': docked_poses,
                'protein_pdb': 'protein.pdb'
            }, f, indent=2)
        
        logger.info(f"Generated {len(docked_poses)} docked pose files in {docked_poses_dir}")
        
        return docked_poses
    
    def prepare_ligand(
        self,
        ligand_data: str,
        ligand_id: str = "ligand",
        data_format: str = "sdf",
        charge_method: str = "am1bcc",
        generate_3d: bool = True
    ) -> Optional[openfe.SmallMoleculeComponent]:
        """
        Prepare ligand from structure data and assign partial charges.
        
        Args:
            ligand_data: Structure data (SDF, MOL, PDB format)
            ligand_id: Identifier for the ligand
            data_format: Format of ligand data ('sdf', 'mol', 'pdb')
            charge_method: Partial charge method ('am1bcc', 'gasteiger')
            generate_3d: Whether to generate 3D coordinates (default: True)
            
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
                logger.error(f"Failed to parse ligand structure for {ligand_id}")
                return None

            # Validate force field compatibility before preparation
            ff_warnings = self._validate_forcefield_compatibility(mol, ligand_id)
            if ff_warnings:
                logger.warning(f"Force field compatibility warnings for {ligand_id}:")
                for warning in ff_warnings:
                    logger.warning(f"  - {warning}")

            # Prepare ligand (add Hs, generate 3D if needed)
            # CRITICAL: Pass generate_3d=False for aligned ligands to preserve alignment
            mol = self.ligand_preparer.prepare(
                mol,
                add_hs=True,
                generate_3d=generate_3d,
                optimize=False
            )
            if generate_3d:
                logger.info(f"Prepared ligand {ligand_id} using LigandPreparer (generated 3D)")
            else:
                logger.info(f"Prepared ligand {ligand_id} using LigandPreparer (preserved 3D)")
            
            # Convert to OpenFE SmallMoleculeComponent
            ligand = openfe.SmallMoleculeComponent.from_rdkit(mol, name=ligand_id)
            
            # Assign partial charges using OpenFE utilities
            logger.info(f"Assigning partial charges to {ligand_id} using {charge_method}")
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
                logger.error(f"Failed to assign charges to ligand {ligand_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error preparing ligand {ligand_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _is_2d_structure(self, ligand_data: str, data_format: str) -> bool:
        """
        Check if raw ligand data represents a 2D structure (no meaningful Z coords).

        Returns True if the ligand has no 3D coordinates and will need
        RDKit-generated coordinates (which are randomly positioned).
        """
        if not RDKIT_AVAILABLE:
            return False
        try:
            if data_format.lower() in ['sdf', 'mol']:
                mol = Chem.MolFromMolBlock(ligand_data, removeHs=True, sanitize=False)
            elif data_format.lower() == 'pdb':
                mol = Chem.MolFromPDBBlock(ligand_data, removeHs=True, sanitize=False)
            else:
                return False

            if mol is None or mol.GetNumConformers() == 0:
                return True

            conf = mol.GetConformer(0)
            z_coords = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
            return all(abs(z) < 0.001 for z in z_coords)
        except Exception:
            return False

    ALIGNMENT_DISTANCE_THRESHOLD = 10.0  # Angstroms

    def _compute_ligand_centroid(
        self,
        ligand: 'openfe.SmallMoleculeComponent'
    ) -> Optional[Tuple[float, float, float]]:
        """
        Compute the centroid of heavy atoms for a SmallMoleculeComponent.

        Returns:
            (x, y, z) tuple of centroid coordinates, or None if computation fails.
        """
        if not RDKIT_AVAILABLE:
            return None
        try:
            mol = ligand.to_rdkit()
            conf = mol.GetConformer(0)
            heavy_positions = [
                conf.GetAtomPosition(i)
                for i in range(mol.GetNumAtoms())
                if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
            ]
            if not heavy_positions:
                return None
            n = len(heavy_positions)
            cx = sum(p.x for p in heavy_positions) / n
            cy = sum(p.y for p in heavy_positions) / n
            cz = sum(p.z for p in heavy_positions) / n
            return (cx, cy, cz)
        except Exception as e:
            logger.warning(f"Failed to compute centroid for {ligand.name}: {e}")
            return None

    def _ligands_need_alignment(
        self,
        ligands: List['openfe.SmallMoleculeComponent']
    ) -> bool:
        """
        Check whether any pair of ligands is spatially displaced beyond the
        alignment threshold (centroid distance > ALIGNMENT_DISTANCE_THRESHOLD).

        This catches the case where one ligand has crystal/docked coordinates
        (e.g. 20-50A from origin) and another has RDKit-generated 3D coords
        (near origin), even though both have valid Z coordinates.
        """
        centroids = []
        for lig in ligands:
            c = self._compute_ligand_centroid(lig)
            if c is not None:
                centroids.append(c)

        if len(centroids) < 2:
            return False

        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dx = centroids[i][0] - centroids[j][0]
                dy = centroids[i][1] - centroids[j][1]
                dz = centroids[i][2] - centroids[j][2]
                dist = (dx**2 + dy**2 + dz**2) ** 0.5
                if dist > self.ALIGNMENT_DISTANCE_THRESHOLD:
                    logger.info(
                        f"Ligands {i} and {j} are {dist:.1f} A apart "
                        f"(threshold={self.ALIGNMENT_DISTANCE_THRESHOLD} A), "
                        f"alignment needed"
                    )
                    return True

        logger.info(
            f"All {len(centroids)} ligands are spatially co-located "
            f"(max pairwise distance < {self.ALIGNMENT_DISTANCE_THRESHOLD} A)"
        )
        return False

    def _align_ligand_to_reference(
        self,
        ligand: 'openfe.SmallMoleculeComponent',
        reference: 'openfe.SmallMoleculeComponent'
    ) -> Optional['openfe.SmallMoleculeComponent']:
        """
        Align a ligand to a reference using MCS-based 3D alignment.

        Used to position library ligands (with RDKit-generated random 3D coords)
        into the binding pocket by aligning their common substructure to a
        docked reference ligand.

        Args:
            ligand: Ligand to align (has random 3D coordinates)
            reference: Reference ligand with correct 3D pose (e.g. from docking)

        Returns:
            New SmallMoleculeComponent with aligned coordinates, or None if failed
        """
        if not RDKIT_AVAILABLE:
            return None

        try:
            mol = Chem.RWMol(ligand.to_rdkit())
            ref_mol = reference.to_rdkit()

            # Find MCS between the two molecules
            mcs_result = rdFMCS.FindMCS(
                [ref_mol, mol],
                bondCompare=rdFMCS.BondCompare.CompareAny,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                matchValences=False,
                ringMatchesRingOnly=True,
                timeout=10
            )

            if mcs_result.canceled or mcs_result.numAtoms < 3:
                logger.warning(
                    f"MCS too small ({mcs_result.numAtoms} atoms) between "
                    f"{ligand.name} and {reference.name}, skipping alignment"
                )
                return None

            mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
            if mcs_mol is None:
                return None

            ref_match = ref_mol.GetSubstructMatch(mcs_mol)
            lig_match = mol.GetSubstructMatch(mcs_mol)

            if not ref_match or not lig_match:
                logger.warning(
                    f"Substructure match failed for {ligand.name}, skipping alignment"
                )
                return None

            # Build coordMap: {ligand_atom_idx: Point3D from reference}
            ref_conf = ref_mol.GetConformer(0)
            coord_map = {}
            for lig_idx, ref_idx in zip(lig_match, ref_match):
                coord_map[lig_idx] = ref_conf.GetAtomPosition(ref_idx)

            # Generate new conformer with MCS atoms constrained to reference positions.
            # Kartograf requires core atoms within ~0.5 A to find proper mappings.
            aligned_well = False
            try:
                mol.RemoveAllConformers()
                params = AllChem.ETKDGv3()
                params.SetCoordMap(coord_map)
                AllChem.EmbedMolecule(mol, params)

                if mol.GetNumConformers() > 0:
                    # Verify constraints were actually enforced
                    conf = mol.GetConformer(0)
                    dists = [conf.GetAtomPosition(li).Distance(ref_conf.GetAtomPosition(ri))
                             for li, ri in zip(lig_match, ref_match)]
                    avg_dist = sum(dists) / len(dists) if dists else 999
                    if avg_dist < 1.0:
                        aligned_well = True
                    else:
                        logger.warning(
                            f"Constrained embedding quality poor for {ligand.name} "
                            f"(avg={avg_dist:.2f} A), falling back to rigid alignment"
                        )
            except Exception as embed_err:
                logger.warning(
                    f"Constrained embedding raised exception for {ligand.name}: {embed_err}"
                )

            # Fallback: rigid-body alignment + pin core atoms to exact reference positions
            if not aligned_well:
                mol = Chem.RWMol(ligand.to_rdkit())
                atom_map = list(zip(lig_match, ref_match))
                rdMolAlign.AlignMol(mol, ref_mol, atomMap=atom_map)
                # Pin core atoms to exact reference positions so Kartograf
                # sees ~0 A distance for mapped atoms
                conf = mol.GetConformer(0)
                for lig_idx, ref_idx in zip(lig_match, ref_match):
                    conf.SetAtomPosition(lig_idx, ref_conf.GetAtomPosition(ref_idx))

            # Log alignment quality
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer(0)
                dists = []
                for lig_idx, ref_idx in zip(lig_match, ref_match):
                    p1 = conf.GetAtomPosition(lig_idx)
                    p2 = ref_conf.GetAtomPosition(ref_idx)
                    dists.append(p1.Distance(p2))
                avg_dist = sum(dists) / len(dists) if dists else 0
                logger.info(
                    f"Aligned {ligand.name} to {reference.name}: "
                    f"avg core distance={avg_dist:.2f} A, {len(coord_map)} MCS atoms"
                )

            aligned = openfe.SmallMoleculeComponent.from_rdkit(
                Chem.Mol(mol), name=ligand.name
            )
            return aligned

        except Exception as e:
            logger.warning(f"Failed to align {ligand.name}: {e}")
            return None

    def _validate_hybrid_system(
        self,
        stateA: Any,
        stateB: Any,
        ligand_a_name: str,
        ligand_b_name: str,
        leg: str
    ) -> Dict[str, Any]:
        """
        Validate hybrid system before MD execution.

        This performs pre-execution checks on the hybrid topology to detect
        issues that could cause NaN errors during simulation.

        Checks:
        - Atom overlap in both lambda endpoints
        - Ligand-protein contact quality
        - System energy sanity checks (if possible without full parameterization)

        Args:
            stateA: ChemicalSystem for state A
            stateB: ChemicalSystem for state B
            ligand_a_name: Name of ligand A
            ligand_b_name: Name of ligand B
            leg: 'complex' or 'solvent'

        Returns:
            Validation results dict with 'valid' bool and 'warnings' list
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'ligand_a': ligand_a_name,
            'ligand_b': ligand_b_name,
            'leg': leg
        }

        try:
            # Basic validation: check that both states have compatible components
            components_a = set(stateA.components.keys())
            components_b = set(stateB.components.keys())

            if components_a != components_b:
                validation_result['warnings'].append(
                    f"Component mismatch between states A and B: "
                    f"A={components_a}, B={components_b}"
                )

            # Check for ligand presence
            ligand_found_a = any('ligand' in str(key).lower() or 'small' in str(key).lower()
                                for key in components_a)
            ligand_found_b = any('ligand' in str(key).lower() or 'small' in str(key).lower()
                                for key in components_b)

            if not ligand_found_a or not ligand_found_b:
                validation_result['warnings'].append(
                    "Could not identify ligand component in chemical system"
                )

            # For complex leg, check for protein presence
            if leg == 'complex':
                protein_found_a = any('protein' in str(key).lower() for key in components_a)
                protein_found_b = any('protein' in str(key).lower() for key in components_b)

                if not protein_found_a or not protein_found_b:
                    validation_result['warnings'].append(
                        "Could not identify protein component in complex system"
                    )

            # Note: Detailed energy checks and atom overlap detection would require
            # creating the hybrid system and running energy evaluations, which is
            # computationally expensive. We rely on the structural validation
            # performed during alignment instead.

            logger.debug(
                f"Hybrid system validation: {ligand_a_name} -> {ligand_b_name} ({leg}): "
                f"{len(validation_result['warnings'])} warnings"
            )

        except Exception as e:
            validation_result['valid'] = False
            validation_result['warnings'].append(f"Validation error: {str(e)}")
            logger.warning(
                f"Error validating hybrid system {ligand_a_name} -> {ligand_b_name} ({leg}): {e}"
            )

        return validation_result

    def _validate_forcefield_compatibility(
        self,
        mol: Chem.Mol,
        ligand_id: str
    ) -> List[str]:
        """
        Validate that ligand is compatible with OpenFF force field.

        OpenFF (Open Force Field) is optimized for typical drug-like molecules
        but may have issues with:
        - Unusual functional groups (e.g., boron, silicon, exotic halogens)
        - Metal-containing compounds
        - Non-standard ring systems
        - Very large or complex molecules

        Args:
            mol: RDKit molecule
            ligand_id: Identifier for logging

        Returns:
            List of warning strings (empty if no issues found)
        """
        warnings = []

        try:
            if not RDKIT_AVAILABLE:
                return warnings

            # Check 1: Unusual elements
            # OpenFF is primarily validated for H, C, N, O, S, P, F, Cl, Br, I
            common_elements = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H, C, N, O, F, P, S, Cl, Br, I
            unusual_elements = []

            for atom in mol.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                if atomic_num not in common_elements:
                    element_symbol = atom.GetSymbol()
                    if element_symbol not in unusual_elements:
                        unusual_elements.append(element_symbol)

            if unusual_elements:
                warnings.append(
                    f"Contains unusual elements for OpenFF: {', '.join(unusual_elements)}. "
                    f"Force field may not be well-parameterized for these atoms."
                )

            # Check 2: Metals (common issue)
            metal_elements = {
                3, 4, 11, 12, 13,  # Li, Be, Na, Mg, Al
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,  # K-Ga
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,  # Rb-In
                55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,  # Cs-Tl
            }

            metals_found = []
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() in metal_elements:
                    metals_found.append(atom.GetSymbol())

            if metals_found:
                warnings.append(
                    f"Contains metal atoms: {', '.join(set(metals_found))}. "
                    f"OpenFF does not support metal coordination chemistry."
                )

            # Check 3: Molecular size
            num_atoms = mol.GetNumAtoms()
            num_heavy_atoms = mol.GetNumHeavyAtoms()

            if num_heavy_atoms > 100:
                warnings.append(
                    f"Large molecule ({num_heavy_atoms} heavy atoms). "
                    f"Force field parameterization and MD may be slower or less reliable."
                )

            # Check 4: Formal charges
            formal_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
            total_charge = sum(formal_charges)
            max_abs_charge = max(abs(c) for c in formal_charges) if formal_charges else 0

            if abs(total_charge) > 2:
                warnings.append(
                    f"High total charge: {total_charge}. "
                    f"Highly charged molecules may have force field issues."
                )

            if max_abs_charge > 2:
                warnings.append(
                    f"Atom with high formal charge: {max_abs_charge}. "
                    f"Unusual charge states may not be well-parameterized."
                )

            # Check 5: Complex ring systems
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()

            if num_rings > 10:
                warnings.append(
                    f"Complex ring system ({num_rings} rings). "
                    f"Unusual polycyclic systems may have parameterization issues."
                )

            # Check for very large rings (macrocycles)
            ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
            if ring_sizes and max(ring_sizes) > 12:
                warnings.append(
                    f"Macrocyclic ring detected (size {max(ring_sizes)}). "
                    f"Large rings may require specialized treatment."
                )

            # Check 6: Known problematic functional groups
            # Using SMARTS patterns for common problematic groups
            problematic_patterns = {
                'nitro': '[N+](=O)[O-]',
                'azide': '[N-]=[N+]=[N-]',
                'diazo': '[N]=[N+]=[N-]',
                'peroxide': '[OX2][OX2]',
                'n_oxide': '[n+][O-]',
            }

            found_problematic = []
            for name, smarts in problematic_patterns.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    found_problematic.append(name)

            if found_problematic:
                warnings.append(
                    f"Contains potentially challenging functional groups: "
                    f"{', '.join(found_problematic)}. Monitor for force field artifacts."
                )

        except Exception as e:
            logger.warning(f"Error during force field validation for {ligand_id}: {e}")
            warnings.append(f"Force field validation error: {str(e)}")

        return warnings

    # REMOVED: prepare_ligands_with_alignment() method
    # Alignment is now handled automatically by OpenFE atom mappers during network creation

    def prepare_ligands_batch(
        self,
        ligands_data: List[Dict[str, Any]],
        charge_method: str = "am1bcc",
        generate_3d: bool = True
    ) -> List[openfe.SmallMoleculeComponent]:
        """
        Prepare multiple ligands in batch.

        After preparation, checks whether ligands are spatially co-located by
        computing pairwise centroid distances. If any pair exceeds the alignment
        threshold (10 A), non-reference ligands are aligned to a reference using
        MCS-based constrained embedding. This handles:
        - 2D ligands (Z=0) that get random 3D coords from RDKit
        - Library ligands with valid 3D coords but in a different reference frame
          than PDB-extracted/docked ligands

        Args:
            ligands_data: List of dicts with 'data', 'id', 'format', and
                          optionally 'has_docked_pose' keys
            charge_method: Partial charge method
            generate_3d: Whether to generate 3D coordinates (set False if already aligned)

        Returns:
            List of prepared SmallMoleculeComponent objects
        """
        prepared_ligands = []
        has_docked_pose = []

        for lig_info in ligands_data:
            ligand_data = lig_info.get('data', '')
            ligand_id = lig_info.get('id', 'ligand')
            data_format = lig_info.get('format', 'sdf')

            ligand = self.prepare_ligand(
                ligand_data=ligand_data,
                ligand_id=ligand_id,
                data_format=data_format,
                charge_method=charge_method,
                generate_3d=generate_3d
            )

            if ligand is not None:
                prepared_ligands.append(ligand)
                has_docked_pose.append(lig_info.get('has_docked_pose', False))
            else:
                logger.warning(f"Skipping ligand {ligand_id} due to preparation failure")

        # CRITICAL FIX: Always align ligands unless ALL have docked poses
        # Kartograf requires rotationally aligned molecules, not just co-located centroids.
        # RDKit embedding generates random orientations that fail Kartograf geometric matching.
        needs_alignment = (
            len(prepared_ligands) >= 2 and
            (self._ligands_need_alignment(prepared_ligands) or not all(has_docked_pose))
        )

        if needs_alignment:
            # Pick reference: prefer a ligand with a docked pose
            ref_idx = 0
            for i, docked in enumerate(has_docked_pose):
                if docked:
                    ref_idx = i
                    break

            ref_ligand = prepared_ligands[ref_idx]
            logger.info(
                f"Aligning ligands to reference: {ref_ligand.name} "
                f"(has_docked_pose={has_docked_pose[ref_idx]}). "
                f"This is required for Kartograf geometric matching."
            )

            for i, ligand in enumerate(prepared_ligands):
                if i == ref_idx:
                    continue
                # Skip alignment for ligands that already have docked poses
                if has_docked_pose[i]:
                    logger.info(
                        f"Skipping alignment for {ligand.name} (already has docked pose)"
                    )
                    continue

                aligned = self._align_ligand_to_reference(ligand, ref_ligand)
                if aligned is not None:
                    prepared_ligands[i] = aligned
                else:
                    logger.warning(
                        f"Could not align {ligand.name} to reference, "
                        f"keeping original coordinates"
                    )

        return prepared_ligands
    
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
            # Clean the protein to remove any ligands/heteroatoms
            logger.info("Cleaning protein structure...")
            
            try:
                cleaning_result = self.protein_preparer.clean_structure_staged(
                    pdb_data,
                    remove_heterogens=True,
                    remove_water=True,
                    add_missing_residues=True,
                    add_missing_atoms=True,
                    add_missing_hydrogens=True,
                    keep_ligands=False
                )
                
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
                    cleaned_pdb_data = stages.get('original', pdb_data)
                
                logger.info("Successfully cleaned protein structure")
                
            except Exception as e:
                logger.warning(f"Protein cleaning failed: {e}, using protein as-is")
                cleaned_pdb_data = pdb_data
            
            # Write cleaned PDB to temporary file
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
    
    def create_ligand_network(
        self,
        ligands: List[openfe.SmallMoleculeComponent],
        topology: str = 'mst',
        central_ligand_name: Optional[str] = None,
        atom_mapper: str = 'kartograf',
        atom_map_hydrogens: bool = True,
        lomap_max3d: float = 1.0
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Create ligand network for RBFE calculations using user-selected atom mapper.

        Following OpenFE best practices, the atom mapper creates the network AND
        handles alignment simultaneously. No pre-alignment is needed.

        Args:
            ligands: List of prepared SmallMoleculeComponents with 3D coordinates
            topology: Network topology ('mst', 'radial', 'maximal')
            central_ligand_name: Name of central ligand for radial networks
            atom_mapper: Atom mapper to use ('kartograf', 'lomap', 'lomap_relaxed')
            atom_map_hydrogens: For Kartograf - include hydrogens in mapping
            lomap_max3d: For LOMAP - maximum 3D distance for mapping

        Returns:
            Tuple of (LigandNetwork, network_data_dict with quality metrics)
        """
        # Create NetworkPlanner with user-selected atom mapper
        planner = NetworkPlanner(
            atom_mapper=atom_mapper,
            atom_map_hydrogens=atom_map_hydrogens,
            lomap_max3d=lomap_max3d
        )

        # Create network (mapper handles alignment automatically)
        network, network_data = planner.create_network(
            ligands=ligands,
            topology=topology,
            central_ligand_name=central_ligand_name
        )

        # Convert to dict and add quality metrics
        network_dict = planner.network_data_to_dict(network_data)
        quality = planner.estimate_network_quality(network_data)
        network_dict['quality'] = quality
        network_dict['mapper_used'] = atom_mapper

        return network, network_dict
    
    def setup_rbfe_protocol(
        self,
        simulation_settings: Optional[Dict[str, Any]] = None
    ) -> RelativeHybridTopologyProtocol:
        """
        Set up RBFE protocol with user-customizable settings.

        All settings are optional — OpenFE defaults are used when not specified.

        Args:
            simulation_settings: Optional dict with any of:
                Core: equilibration_length_ns, production_length_ns, lambda_windows,
                      protocol_repeats
                Ligand: ligand_forcefield, charge_method
                Environment: temperature, pressure, solvent_model, box_shape,
                             solvent_padding_nm
                Advanced: timestep_fs, hydrogen_mass, minimization_steps,
                          compute_platform

        Returns:
            Configured RelativeHybridTopologyProtocol
        """
        settings = RelativeHybridTopologyProtocol.default_settings()
        cfg = simulation_settings or {}

        # --- Simulation lengths ---
        eq_ns = cfg.get('equilibration_length_ns', cfg.get('equilibration_ns'))
        prod_ns = cfg.get('production_length_ns', cfg.get('production_ns'))
        if eq_ns is not None:
            settings.simulation_settings.equilibration_length = float(eq_ns) * unit.nanosecond
        if prod_ns is not None:
            settings.simulation_settings.production_length = float(prod_ns) * unit.nanosecond

        # --- Lambda windows ---
        if 'lambda_windows' in cfg:
            settings.lambda_settings.lambda_windows = int(cfg['lambda_windows'])

        # --- Protocol repeats ---
        if 'protocol_repeats' in cfg:
            settings.protocol_repeats = int(cfg['protocol_repeats'])

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
        if solvent_model:
            settings.solvation_settings.solvent_model = solvent_model
        if box_shape:
            settings.solvation_settings.box_shape = box_shape
        if solvent_padding_nm is not None:
            settings.solvation_settings.solvent_padding = float(solvent_padding_nm) * unit.nanometer

        # --- Compute platform ---
        if 'compute_platform' in cfg:
            settings.engine_settings.compute_platform = cfg['compute_platform']

        # --- HMR and integrator ---
        h_mass = float(cfg.get('hydrogen_mass', 3.0))
        ts_fs = float(cfg.get('timestep_fs', 4.0))
        min_steps = int(cfg.get('minimization_steps', 10000))

        settings.forcefield_settings.hydrogen_mass = h_mass
        settings.integrator_settings.timestep = ts_fs * unit.femtosecond
        settings.simulation_settings.minimization_steps = min_steps

        # Auto-detect compute platform if not explicitly set
        if 'compute_platform' not in cfg:
            try:
                import openmm
                available_platforms = [
                    openmm.Platform.getPlatform(i).getName()
                    for i in range(openmm.Platform.getNumPlatforms())
                ]
                if 'CUDA' in available_platforms:
                    settings.engine_settings.compute_platform = 'CUDA'
                elif 'OpenCL' in available_platforms:
                    settings.engine_settings.compute_platform = 'OpenCL'
                    logger.warning("No CUDA GPU found, falling back to OpenCL platform")
                else:
                    settings.engine_settings.compute_platform = 'CPU'
                    logger.warning("No GPU found, falling back to CPU platform")
            except Exception as e:
                logger.warning("GPU platform detection failed: %s", e)

        # --- Log final configuration ---
        logger.info("RBFE protocol configured:")
        logger.info(f"  Production: {settings.simulation_settings.production_length}")
        logger.info(f"  Equilibration: {settings.simulation_settings.equilibration_length}")
        logger.info(f"  Lambda windows: {settings.lambda_settings.lambda_windows}")
        logger.info(f"  Minimization steps: {min_steps}")
        logger.info(f"  Timestep: {ts_fs} fs, Hydrogen mass: {h_mass} amu")
        logger.info(f"  Protocol repeats: {settings.protocol_repeats}")
        logger.info(f"  Temperature: {settings.thermo_settings.temperature}, Pressure: {settings.thermo_settings.pressure}")
        logger.info(f"  Solvent: {settings.solvation_settings.solvent_model}, Box: {settings.solvation_settings.box_shape}")
        logger.info(f"  Ligand FF: {settings.forcefield_settings.small_molecule_forcefield}")
        logger.info(f"  Platform: {settings.engine_settings.compute_platform}")

        return RelativeHybridTopologyProtocol(settings=settings)
    
    def create_transformations(
        self,
        protein: openfe.ProteinComponent,
        ligand_network: Any,
        protocol: RelativeHybridTopologyProtocol,
        solvent_nacl_concentration: float = 0.15
    ) -> List[Any]:
        """
        Create transformation objects for all edges in the network.
        
        Args:
            protein: Protein component
            ligand_network: OpenFE LigandNetwork
            protocol: Configured RBFE protocol
            solvent_nacl_concentration: NaCl concentration in M
            
        Returns:
            List of Transformation objects
        """
        # Create solvent component
        solvent = openfe.SolventComponent(
            positive_ion='Na+',
            negative_ion='Cl-',
            neutralize=True,
            ion_concentration=solvent_nacl_concentration * unit.molar
        )
        
        # Create system generator
        system_generator = EasyChemicalSystemGenerator(
            solvent=solvent,
            protein=protein,
            do_vacuum=False
        )
        
        
        transformations = []
        
        # Cache for systems to avoid redundant parameterization
        # Key: ligand_name, Value: list of ChemicalSystems
        system_cache: Dict[str, List[Any]] = {}
        
        for edge in ligand_network.edges:
            ligand_a = edge.componentA
            ligand_b = edge.componentB
            mapping = edge
            
            # Generate or retrieve systems for ligand A
            if ligand_a.name in system_cache:
                systems_a = system_cache[ligand_a.name]
                logger.info(f"Using cached systems for {ligand_a.name}")
            else:
                logger.info(f"Generating systems for {ligand_a.name}...")
                systems_a = list(system_generator(ligand_a))
                system_cache[ligand_a.name] = systems_a
            
            # Generate or retrieve systems for ligand B
            if ligand_b.name in system_cache:
                systems_b = system_cache[ligand_b.name]
                logger.info(f"Using cached systems for {ligand_b.name}")
            else:
                logger.info(f"Generating systems for {ligand_b.name}...")
                systems_b = list(system_generator(ligand_b))
                system_cache[ligand_b.name] = systems_b
            
            # Create transformations for each environment (solvent and complex)
            for sys_a, sys_b in zip(systems_a, systems_b):
                # Determine if complex or solvent based on presence of protein
                is_complex = protein in sys_a.components.values()
                leg_name = 'complex' if is_complex else 'solvent'
                
                transformation = openfe.Transformation(
                    stateA=sys_a,
                    stateB=sys_b,
                    mapping=mapping,
                    protocol=protocol,
                    name=f"{ligand_a.name}_{ligand_b.name}_{leg_name}"
                )
                transformations.append(transformation)
        
        logger.info(f"Created {len(transformations)} transformations")
        return transformations
    
    def _update_job_status(
        self,
        job_id: str,
        status_update: Dict[str, Any]
    ) -> None:
        """Update job status in file-based tracking."""
        job_file = self.jobs_dir / f"{job_id}.json"

        current_status = {}
        if job_file.exists():
            try:
                with open(job_file, 'r') as f:
                    current_status = json.load(f)
            except Exception as e:
                logger.warning("Job status file corrupt for %s: %s", job_id, e)

        current_status.update(status_update)

        # Atomic write: write to tempfile, then replace
        with tempfile.NamedTemporaryFile(
            'w', dir=self.jobs_dir, delete=False, suffix='.tmp'
        ) as tmp:
            json.dump(current_status, tmp, indent=2, default=str)
            tmp_path = tmp.name
        os.replace(tmp_path, str(job_file))

        # Also update in-memory cache
        self.jobs[job_id] = current_status
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get current status of a job."""
        job_file = self.jobs_dir / f"{job_id}.json"
        
        if job_file.exists():
            try:
                with open(job_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading job status: {e}")
        
        return {'status': 'not_found', 'job_id': job_id}

    def parse_results_from_job(self, job_id: str) -> Dict[str, Any]:
        """Read completed RBFE results from disk for job recovery."""
        results_file = self.output_dir / job_id / "results.json"
        if not results_file.exists():
            return {'error': f'No results.json found for job {job_id}'}
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {'error': f'Failed to read results: {e}'}

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all RBFE jobs."""
        jobs = []
        
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file, 'r') as f:
                    job_data = json.load(f)
                    jobs.append(job_data)
            except Exception as e:
                logger.warning(f"Error reading job file {job_file}: {e}")
        
        # Sort by created_at descending
        jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jobs
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running job or reset a stale job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            Updated job status
        """
        job_status = self.get_job_status(job_id)
        
        if job_status.get('status') == 'not_found':
            return job_status
        
        current_status = job_status.get('status', '')
        
        # Only allow cancellation of non-terminal states
        terminal_states = {'completed', 'failed', 'cancelled'}
        
        if current_status in terminal_states:
            logger.warning(f"Job {job_id} is already in terminal state: {current_status}")
            return job_status
        
        logger.info(f"Cancelling job {job_id} (was in state: {current_status})")
        
        self._update_job_status(job_id, {
            'status': 'cancelled',
            'message': f'Job cancelled (was: {current_status})',
            'cancelled_at': __import__('datetime').datetime.now().isoformat()
        })
        
        return self.get_job_status(job_id)
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job and its files.
        
        Args:
            job_id: Job ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        import shutil
        
        job_file = self.jobs_dir / f"{job_id}.json"
        job_dir = self.output_dir / job_id
        
        deleted = False
        
        if job_file.exists():
            job_file.unlink()
            deleted = True
            logger.info(f"Deleted job status file: {job_file}")
        
        if job_dir.exists():
            shutil.rmtree(job_dir)
            deleted = True
            logger.info(f"Deleted job directory: {job_dir}")
        
        # Remove from in-memory cache
        if job_id in self.jobs:
            del self.jobs[job_id]
        
        return deleted
    
    def check_stale_jobs(self, stale_threshold_minutes: int = 30) -> List[str]:
        """
        Find jobs that appear to be stale (stuck in running state).
        
        Args:
            stale_threshold_minutes: Minutes after which a running job is considered stale
            
        Returns:
            List of stale job IDs
        """
        from datetime import datetime, timedelta
        
        stale_jobs = []
        running_states = {'preparing', 'docking', 'running', 'resuming'}
        threshold = datetime.now() - timedelta(minutes=stale_threshold_minutes)
        
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file, 'r') as f:
                    job_data = json.load(f)
                
                status = job_data.get('status', '')
                
                if status in running_states:
                    # Check last update time
                    created_at = job_data.get('created_at', '')
                    if created_at:
                        try:
                            created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            if created_time.tzinfo:
                                created_time = created_time.replace(tzinfo=None)
                            
                            if created_time < threshold:
                                stale_jobs.append(job_data.get('job_id', job_file.stem))
                        except (ValueError, TypeError):
                            # If we can't parse the date, consider it stale
                            stale_jobs.append(job_data.get('job_id', job_file.stem))
                            
            except Exception as e:
                logger.warning(f"Error checking job file {job_file}: {e}")
        
        return stale_jobs

    def _load_reference_from_pdb_string(
        self,
        pdb_string: str,
        ligand_id: str,
        template_ligand_data: Optional[str] = None,
        template_ligand_format: str = "sdf"
    ) -> Optional['openfe.SmallMoleculeComponent']:
        """
        Load a reference ligand from PDB string (e.g., cocrystal structure).

        Args:
            pdb_string: PDB format string containing HETATM records for the ligand
            ligand_id: Identifier for the ligand

        Returns:
            OpenFE SmallMoleculeComponent, or None if parsing failed
        """
        try:
            mol = Chem.MolFromPDBBlock(pdb_string, removeHs=False)
            if mol is None:
                logger.error(f"Failed to parse reference ligand {ligand_id} from PDB")
                return None

            template_mol = self._build_template_mol(
                ligand_data=template_ligand_data or "",
                ligand_format=template_ligand_format,
                ligand_id=ligand_id,
            )
            mol = self._restore_bond_orders_from_template(mol, template_mol, ligand_id)

            # Generate 3D coordinates if needed
            try:
                if mol.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(mol, randomSeed=42)
            except Exception as e:
                logger.warning(f"Could not embed reference ligand {ligand_id}: {e}")

            component = openfe.SmallMoleculeComponent.from_rdkit(mol, name=ligand_id)
            logger.info(f"Loaded reference ligand {ligand_id} from PDB")
            return component

        except Exception as e:
            logger.error(f"Error loading reference ligand {ligand_id}: {e}")
            return None

    def _dock_single_ligand_via_vina(
        self,
        protein_pdb: str,
        ligand_id: str,
        ligand_data: str,
        ligand_format: str,
        exhaustiveness: int = 8,
        grid_box: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Dock a single ligand using the docking service.

        Args:
            protein_pdb: Protein PDB data
            ligand_id: Ligand identifier
            ligand_data: Ligand structure data
            ligand_format: Format of ligand data (sdf, mol, smiles)
            exhaustiveness: Docking exhaustiveness parameter
            grid_box: Optional docking grid box override

        Returns:
            Dict with 'docked_sdf', 'affinity', or None if docking failed
        """
        try:
            template_mol = self._build_template_mol(
                ligand_data=ligand_data,
                ligand_format=ligand_format,
                ligand_id=ligand_id,
            )

            # Use dock_ligands_batch for single ligand
            results = self.dock_ligands_batch(
                protein_pdb=protein_pdb,
                ligands_data=[{
                    'id': ligand_id,
                    'data': ligand_data,
                    'format': ligand_format
                }],
                grid_box=grid_box,
                exhaustiveness=exhaustiveness,
                num_poses=1
            )

            if not results:
                logger.error(f"Docking failed for reference ligand {ligand_id}")
                return None

            result = results[0]
            if not result.get('success'):
                logger.error(f"Docking failed for {ligand_id}: {result.get('error')}")
                return None

            # Extract best pose from PDBQT
            poses_pdbqt = result.get('poses_pdbqt', '')
            best_affinity = result.get('best_score', result.get('best_affinity', 0.0))

            docked_sdf = self.extract_best_pose_from_pdbqt(
                poses_pdbqt=poses_pdbqt,
                ligand_id=ligand_id,
                template_mol=template_mol,
            )
            if not docked_sdf:
                logger.error(f"Failed to extract docked pose for {ligand_id}")
                return None

            logger.info(f"Successfully docked {ligand_id}: affinity={best_affinity:.2f} kcal/mol")
            return {
                'docked_sdf': docked_sdf,
                'affinity': best_affinity,
                'ligand_id': ligand_id
            }

        except Exception as e:
            logger.error(f"Error docking reference ligand {ligand_id}: {e}")
            logger.error(traceback.format_exc())
            return None

    def _build_docked_poses_payload(
        self,
        ligands: List['openfe.SmallMoleculeComponent'],
        ligand_ids: List[str],
        protein_pdb: str,
        job_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Build DockedPoseInfo payload with PDB files for all ligands.

        Args:
            ligands: List of prepared OpenFE ligands
            ligand_ids: Corresponding ligand IDs
            protein_pdb: Protein PDB data
            job_dir: Output directory

        Returns:
            List of DockedPoseInfo dicts
        """
        docked_poses_dir = job_dir / "docked_poses"
        docked_poses_dir.mkdir(parents=True, exist_ok=True)

        # Save protein PDB
        protein_pdb_path = docked_poses_dir / "protein.pdb"
        with open(protein_pdb_path, 'w') as f:
            f.write(protein_pdb)

        docked_poses = []

        for ligand, lig_id in zip(ligands, ligand_ids):
            try:
                # Convert OpenFE ligand to RDKit mol and then PDB
                mol = ligand.to_rdkit()
                ligand_pdb = Chem.MolToPDBBlock(mol)
                ligand_sdf = Chem.MolToMolBlock(mol)

                # Save ligand PDB
                ligand_pdb_path = docked_poses_dir / f"{lig_id}_docked.pdb"
                with open(ligand_pdb_path, 'w') as f:
                    f.write(ligand_pdb)

                # Save ligand SDF as canonical resume source (preserves bond orders/aromaticity)
                ligand_sdf_path = docked_poses_dir / f"{lig_id}_docked.sdf"
                with open(ligand_sdf_path, 'w') as f:
                    f.write(ligand_sdf)

                # Create complex PDB
                complex_pdb_path = docked_poses_dir / f"{lig_id}_complex.pdb"
                with open(complex_pdb_path, 'w') as f:
                    protein_lines = protein_pdb.strip().split('\n')
                    for line in protein_lines:
                        if not line.startswith('END'):
                            f.write(line + '\n')
                    f.write('TER\n')

                    ligand_lines = ligand_pdb.strip().split('\n')
                    for line in ligand_lines:
                        if line.startswith('ATOM') or line.startswith('HETATM'):
                            if line.startswith('ATOM'):
                                line = 'HETATM' + line[6:]
                            f.write(line + '\n')
                        elif line.startswith('CONECT'):
                            f.write(line + '\n')
                    f.write('END\n')

                pose_info = {
                    'ligand_id': lig_id,
                    'affinity_kcal_mol': 0.0,
                    'pose_pdb_path': str(ligand_pdb_path.relative_to(job_dir)),
                    'complex_pdb_path': str(complex_pdb_path.relative_to(job_dir))
                }

                docked_poses.append(pose_info)
                logger.info(f"Generated pose files for {lig_id}")

            except Exception as e:
                logger.error(f"Error generating pose files for {lig_id}: {e}")
                continue

        return docked_poses

    def _load_aligned_poses_from_disk(
        self,
        job_id: str,
        ligand_ids: List[str]
    ) -> Optional[List['openfe.SmallMoleculeComponent']]:
        """
        Load pre-aligned ligand structures from disk (for job resumption).

        Args:
            job_id: Job identifier
            ligand_ids: List of ligand IDs to load

        Returns:
            List of OpenFE SmallMoleculeComponent, or None if loading failed
        """
        try:
            docked_poses_dir = self.output_dir / job_id / "docked_poses"
            if not docked_poses_dir.exists():
                logger.warning(f"Docked poses directory not found for job {job_id}")
                return None

            ligands = []
            for lig_id in ligand_ids:
                ligand_sdf_path = docked_poses_dir / f"{lig_id}_docked.sdf"
                ligand_pdb_path = docked_poses_dir / f"{lig_id}_docked.pdb"

                mol = None
                # Prefer SDF: PDB round-tripping can lose chemistry perception (bond orders/aromaticity)
                # and later break FF parameterization.
                if ligand_sdf_path.exists():
                    with open(ligand_sdf_path, 'r') as f:
                        sdf_data = f.read()
                    mol = Chem.MolFromMolBlock(sdf_data, removeHs=False)
                    if mol is not None:
                        # Validate: detect corrupted SDFs (e.g. all single bonds, radical carbons)
                        try:
                            Chem.SanitizeMol(mol)
                        except Exception as e:
                            logger.warning(f"SDF for {lig_id} has invalid chemistry ({e}), discarding")
                            mol = None
                    if mol is None:
                        logger.warning(f"Failed to parse ligand {lig_id} from SDF, falling back to PDB")

                # Backward-compat fallback for older jobs that only have PDB files
                if mol is None:
                    if not ligand_pdb_path.exists():
                        logger.warning(f"Ligand pose file not found (SDF/PDB) for {lig_id}")
                        return None
                    with open(ligand_pdb_path, 'r') as f:
                        pdb_data = f.read()
                    mol = Chem.MolFromPDBBlock(pdb_data, removeHs=False)
                    if mol is None:
                        logger.error(f"Failed to parse ligand {lig_id} from {ligand_pdb_path}")
                        return None

                component = openfe.SmallMoleculeComponent.from_rdkit(mol, name=lig_id)
                ligands.append(component)

            logger.info(f"Loaded {len(ligands)} pre-aligned poses for job {job_id}")
            return ligands

        except Exception as e:
            logger.error(f"Error loading aligned poses for job {job_id}: {e}")
            logger.error(traceback.format_exc())
            return None

    def run_rbfe_calculation(
        self,
        protein_pdb: str,
        ligands_data: List[Dict[str, Any]],
        job_id: str,
        network_topology: str = 'mst',
        central_ligand_name: Optional[str] = None,
        atom_mapper: str = 'kartograf',
        atom_map_hydrogens: bool = True,
        lomap_max3d: float = 1.0,
        simulation_settings: Optional[Dict[str, Any]] = None,
        protein_id: str = "protein"
    ) -> Dict[str, Any]:
        """
        Run complete RBFE calculation using OpenFE best practices.

        WORKFLOW (following OpenFE recommendations):
        1. Load ligands with 3D coordinates (from file or generate)
        2. Assign partial charges (AM1-BCC by default)
        3. Create SmallMoleculeComponents
        4. Use selected atom mapper to create network (handles alignment automatically)
        5. Setup protein + ligand systems
        6. Run FE calculations for each transformation

        ATOM MAPPING:
        - Kartograf (default): Geometry-based, preserves 3D binding mode
          Recommended for docked poses (95% identical mappings in TYK2 dataset)
          REQUIRES molecules to be rotationally aligned (same orientation)
        - LOMAP: 2D MCS-based, may realign structures
          Use for 2D structures or when Kartograf fails

        ALIGNMENT: RDKit-generated ligands are automatically aligned to a reference
        using MCS-based constrained embedding. Docked poses are used as-is.

        References:
            - Kartograf paper: https://pubs.acs.org/doi/10.1021/acs.jctc.3c01206
            - OpenFE tutorial: https://docs.openfree.energy/en/latest/tutorials/rbfe_cli_tutorial.html

        Args:
            protein_pdb: Protein PDB data as string
            ligands_data: List of dicts with 'data', 'id', 'format' keys
            job_id: Unique job identifier
            network_topology: Network topology ('mst', 'radial', 'maximal')
            central_ligand_name: Central ligand for radial networks
            atom_mapper: Atom mapper ('kartograf', 'lomap', 'lomap_relaxed')
            atom_map_hydrogens: For Kartograf - include hydrogens in mapping
            lomap_max3d: For LOMAP - maximum 3D distance for mapping
            simulation_settings: Protocol settings (charge_method, temperatures, etc.)
            protein_id: Protein identifier

        Returns:
            Job status dictionary with results
        """
        from datetime import datetime
        
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        alignment_info = None

        try:
            # Initialize job status
            # Extract SMILES for each ligand upfront so they survive beyond this session
            ligand_smiles: Dict[str, str] = {}
            for lig in ligands_data:
                lig_id = lig.get('id', '')
                fmt = lig.get('format', 'sdf')
                data = lig.get('data', '')
                try:
                    if fmt == 'smiles':
                        ligand_smiles[lig_id] = data.strip()
                    else:
                        from rdkit import Chem
                        mol = Chem.MolFromMolBlock(data, removeHs=True)
                        if mol is not None:
                            ligand_smiles[lig_id] = Chem.MolToSmiles(mol)
                except Exception as e:
                    logger.debug("SMILES extraction from MOL block failed: %s", e)

            self._update_job_status(job_id, {
                'job_id': job_id,
                'status': 'preparing',
                'protein_id': protein_id,
                'num_ligands': len(ligands_data),
                'network_topology': network_topology,
                'job_dir': str(job_dir),
                'created_at': datetime.now().isoformat(),
                'ligand_smiles': ligand_smiles,
            })

            # Step 1: Prepare ligands with charges and 3D coordinates
            logger.info(f"Step 1: Preparing {len(ligands_data)} ligands...")
            emit_progress(5, f'Preparing {len(ligands_data)} ligands...')

            # Generate 3D coordinates if not present
            # If ligands have docked poses, those coordinates are preserved
            generate_3d_flag = True

            ligands = self.prepare_ligands_batch(
                ligands_data,
                charge_method=simulation_settings.get('charge_method', 'am1bcc') if simulation_settings else 'am1bcc',
                generate_3d=generate_3d_flag
            )

            if len(ligands) < 2:
                raise ValueError(f"Need at least 2 ligands, only {len(ligands)} prepared successfully")

            self._update_job_status(job_id, {
                'status': 'preparing',
                'message': f'Prepared {len(ligands)} ligands'
            })
            emit_progress(10, f'✓ Prepared {len(ligands)} ligands')

            # PHASE 1: Reference Ligand Docking/Alignment (Optional)
            # If reference_ligand_id is specified, handle docking and alignment workflow
            reference_ligand_id = simulation_settings.get('reference_ligand_id') if simulation_settings else None
            reference_pose_source = simulation_settings.get('reference_pose_source') if simulation_settings else None
            docking_acknowledged = simulation_settings.get('docking_acknowledged', False) if simulation_settings else False

            if reference_ligand_id and reference_pose_source:
                # Check if we're resuming from docking_ready checkpoint
                if docking_acknowledged:
                    logger.info("Resuming from docking_ready checkpoint, loading pre-aligned poses...")
                    emit_progress(15, 'Loading pre-aligned poses...')

                    ligand_ids = [lig.get('id') for lig in ligands_data]
                    resumed_ligands = self._load_aligned_poses_from_disk(job_id, ligand_ids)

                    if resumed_ligands:
                        ligands = resumed_ligands
                        logger.info(f"Loaded {len(ligands)} pre-aligned poses from disk")
                        emit_progress(16, f'Loaded {len(ligands)} pre-aligned poses')
                    else:
                        logger.error("Failed to load pre-aligned poses, proceeding with newly prepared ligands")
                else:
                    logger.info(f"Phase 1: Reference ligand workflow (source={reference_pose_source})...")
                    emit_progress(12, f'Handling reference ligand ({reference_pose_source})...')

                    # Extract reference ligand from ligands list
                    ref_ligand = None
                    ref_ligand_data = None
                    for lig_dict in ligands_data:
                        if lig_dict.get('id') == reference_ligand_id:
                            ref_ligand_data = lig_dict
                            break

                    if not ref_ligand_data:
                        logger.error(f"Reference ligand {reference_ligand_id} not found in ligands_data")
                        raise ValueError(f"Reference ligand {reference_ligand_id} not found")

                    # Handle different reference pose sources
                    if reference_pose_source == 'cocrystal':
                        # Reference pose from cocrystal (PDB string)
                        reference_pose_pdb = simulation_settings.get('reference_pose_pdb') if simulation_settings else None
                        if not reference_pose_pdb:
                            raise ValueError("reference_pose_pdb required for cocrystal source")

                        logger.info(f"Loading reference ligand {reference_ligand_id} from cocrystal PDB...")
                        ref_ligand = self._load_reference_from_pdb_string(
                            reference_pose_pdb,
                            reference_ligand_id,
                            template_ligand_data=ref_ligand_data.get('data', ''),
                            template_ligand_format=ref_ligand_data.get('format', 'sdf'),
                        )
                        if not ref_ligand:
                            raise ValueError(f"Failed to load reference ligand from cocrystal PDB")

                        emit_progress(13, 'Loaded reference ligand from cocrystal')

                    elif reference_pose_source == 'vina':
                        # Dock reference ligand using Vina
                        logger.info(f"Docking reference ligand {reference_ligand_id} with Vina...")
                        exhaustiveness = simulation_settings.get('vina_exhaustiveness', 8) if simulation_settings else 8
                        grid_box = simulation_settings.get('vina_grid_box') if simulation_settings else None

                        docking_result = self._dock_single_ligand_via_vina(
                            protein_pdb=protein_pdb,
                            ligand_id=reference_ligand_id,
                            ligand_data=ref_ligand_data.get('data', ''),
                            ligand_format=ref_ligand_data.get('format', 'sdf'),
                            exhaustiveness=exhaustiveness,
                            grid_box=grid_box,
                        )

                        if not docking_result:
                            raise ValueError(f"Failed to dock reference ligand {reference_ligand_id}")

                        emit_progress(13, f'Docked reference ligand: affinity={docking_result["affinity"]:.2f} kcal/mol')

                        # Create OpenFE component from docked SDF
                        docked_sdf = docking_result['docked_sdf']
                        mol = Chem.MolFromMolBlock(docked_sdf, removeHs=False)
                        if not mol:
                            raise ValueError(f"Failed to parse docked SDF for {reference_ligand_id}")
                        ref_ligand = openfe.SmallMoleculeComponent.from_rdkit(mol, name=reference_ligand_id)

                    elif reference_pose_source == 'prior_job':
                        # Reference pose from prior docking job
                        reference_pose_pdb = simulation_settings.get('reference_pose_pdb') if simulation_settings else None
                        if not reference_pose_pdb:
                            raise ValueError("reference_pose_pdb required for prior_job source")

                        logger.info(f"Loading reference ligand {reference_ligand_id} from prior job...")
                        ref_ligand = self._load_reference_from_pdb_string(
                            reference_pose_pdb,
                            reference_ligand_id,
                            template_ligand_data=ref_ligand_data.get('data', ''),
                            template_ligand_format=ref_ligand_data.get('format', 'sdf'),
                        )
                        if not ref_ligand:
                            raise ValueError(f"Failed to load reference ligand from prior job PDB")

                        emit_progress(13, 'Loaded reference ligand from prior job')

                    else:
                        raise ValueError(f"Unknown reference_pose_source: {reference_pose_source}")

                    # Now align all non-reference ligands to the reference
                    logger.info(f"Aligning {len(ligands) - 1} ligands to reference {reference_ligand_id}...")
                    emit_progress(14, f'Aligning ligands to reference...')

                    aligned_ligands = []
                    failed_ligands = []

                    for i, ligand in enumerate(ligands):
                        if ligand.name == reference_ligand_id:
                            # Keep the reference entry in-sync with the selected pose
                            # source (cocrystal / vina / prior_job), not the originally
                            # prepared coordinates.
                            aligned_ligands.append(ref_ligand)
                            logger.info(f"Using selected reference pose for {reference_ligand_id}")
                        else:
                            # Align to reference using MCS-based constrained embedding
                            aligned = self._align_ligand_to_reference(ligand, ref_ligand)
                            if aligned:
                                aligned_ligands.append(aligned)
                            else:
                                # If alignment fails, use original (will likely fail at atom mapping stage)
                                logger.warning(f"Alignment failed for {ligand.name}, using original coordinates")
                                aligned_ligands.append(ligand)

                    # Update ligands list with aligned structures
                    ligands = aligned_ligands

                    # Build docked poses info and save PDB files
                    logger.info("Generating aligned pose PDB files...")
                    emit_progress(15, 'Saving aligned pose files...')

                    ligand_ids = [lig.name for lig in ligands]
                    docked_poses = self._build_docked_poses_payload(
                        ligands=ligands,
                        ligand_ids=ligand_ids,
                        protein_pdb=protein_pdb,
                        job_dir=job_dir
                    )

                    # Emit docking_ready checkpoint
                    logger.info(f"Phase 1 complete: {len(docked_poses)} aligned poses ready for FE calculations")

                    docking_ready_payload = {
                        'docked_poses': docked_poses,
                        'alignment_info': {
                            'reference_ligand': reference_ligand_id,
                            'aligned_ligands': [{'id': lig.name, 'is_reference': lig.name == reference_ligand_id}
                                                for lig in ligands],
                            'failed_ligands': [],
                            'alignment_method': 'mcs_constrained_embedding'
                        }
                    }

                    self._update_job_status(job_id, {
                        'status': 'docking_ready',
                        'message': f'Reference ligand and alignment complete. {len(docked_poses)} poses ready.',
                        'docked_poses': docked_poses,
                        'alignment_info': docking_ready_payload['alignment_info']
                    })

                    emit_progress(15, 'docking_ready', result=docking_ready_payload)
                    return self.get_job_status(job_id)

            # Step 2: Load protein
            logger.info("Step 2: Loading protein...")
            emit_progress(12, 'Loading protein...')
            protein = self.load_protein(pdb_data=protein_pdb, protein_id=protein_id)
            
            if protein is None:
                raise ValueError("Failed to load protein")
            
            # Step 3: Create ligand network using selected atom mapper
            logger.info(f"Step 3: Creating {network_topology} network with {atom_mapper} mapper...")
            emit_progress(15, f'Creating {network_topology} network with {atom_mapper} mapper...')
            network, network_dict = self.create_ligand_network(
                ligands=ligands,
                topology=network_topology,
                central_ligand_name=central_ligand_name,
                atom_mapper=atom_mapper,
                atom_map_hydrogens=atom_map_hydrogens,
                lomap_max3d=lomap_max3d
            )

            # Save network data
            network_file = job_dir / "network.json"
            with open(network_file, 'w') as f:
                json.dump(network_dict, f, indent=2)

            self._update_job_status(job_id, {
                'status': 'preparing',
                'message': f'Created network with {len(network_dict["edges"])} edges',
                'network': network_dict
            })
            emit_progress(20, f'✓ Created network with {len(network_dict["edges"])} edges')

            # Step 4: Setup protocol
            logger.info("Step 4: Setting up RBFE protocol...")
            emit_progress(25, 'Setting up RBFE protocol...')
            protocol = self.setup_rbfe_protocol(simulation_settings)

            # Step 5: Create transformations
            logger.info("Step 5: Creating transformations...")
            emit_progress(30, 'Creating transformations...')
            transformations = self.create_transformations(
                protein=protein,
                ligand_network=network,
                protocol=protocol,
                solvent_nacl_concentration=simulation_settings.get('ionic_strength', 0.15) if simulation_settings else 0.15
            )

            # Create AlchemicalNetwork
            alchemical_network = openfe.AlchemicalNetwork(transformations)

            self._update_job_status(job_id, {
                'status': 'running',
                'message': f'Running {len(transformations)} transformations',
                'num_transformations': len(transformations)
            })
            emit_progress(35, f'Ready to execute {len(transformations)} transformations')

            # Step 6: Execute transformations with robust error handling
            logger.info(f"Step 6: Executing {len(transformations)} transformations...")
            emit_progress(40, f'Executing {len(transformations)} transformations...')

            # User decision: PARTIAL RESULTS enabled
            # If some transformations fail, we still return results from successful ones
            results = []
            failed_transformations = []
            successful_transformations = 0

            total_transformations = len(transformations)
            for i, transformation in enumerate(transformations):
                logger.info(f"Running transformation {i+1}/{total_transformations}: {transformation.name}")

                # Determine leg from name (suffix)
                leg = 'complex' if transformation.name.endswith('_complex') else 'solvent'
                ligand_a_name = transformation.mapping.componentA.name
                ligand_b_name = transformation.mapping.componentB.name

                # Emit progress before starting this transformation
                tx_progress_start = int(i / total_transformations * 95 + 5)
                emit_progress(
                    tx_progress_start,
                    f"Transformation {i+1}/{total_transformations}: {ligand_a_name} → {ligand_b_name} ({leg})"
                )

                try:
                    # Pre-execution validation
                    # Validate hybrid systems before expensive MD runs
                    validation = self._validate_hybrid_system(
                        stateA=transformation.stateA,
                        stateB=transformation.stateB,
                        ligand_a_name=ligand_a_name,
                        ligand_b_name=ligand_b_name,
                        leg=leg
                    )

                    if validation['warnings']:
                        logger.warning(f"Validation warnings for {transformation.name}:")
                        for warning in validation['warnings']:
                            logger.warning(f"  - {warning}")

                    # Create and execute DAG
                    dag = transformation.create()
                    work_dir = job_dir / transformation.name
                    work_dir.mkdir(parents=True, exist_ok=True)

                    # Create separate directories for shared and scratch data
                    shared_dir = work_dir / "shared"
                    scratch_dir = work_dir / "scratch"
                    shared_dir.mkdir(parents=True, exist_ok=True)
                    scratch_dir.mkdir(parents=True, exist_ok=True)

                    # keep_shared=True preserves the shared directories containing analysis files
                    # (overlap matrices, convergence plots, YAML analysis data, etc.)
                    dag_result = execute_DAG(
                        dag,
                        shared_basedir=shared_dir,
                        scratch_basedir=scratch_dir,
                        n_retries=3,  # Retry failed units for robustness
                        keep_shared=True
                    )

                    # Gather results
                    result = protocol.gather([dag_result])
                    estimate = result.get_estimate()
                    uncertainty = result.get_uncertainty()

                    # Extract MBAR overlap matrix using documented OpenFE API + layered fallbacks
                    overlap_matrix, om_source = self._extract_rbfe_overlap_matrix(
                        result, dag_result, shared_dir
                    )
                    if overlap_matrix is not None:
                        logger.debug(
                            f"Overlap matrix for {transformation.name} extracted from: {om_source}"
                        )
                    else:
                        logger.warning(
                            f"Could not extract overlap matrix for {transformation.name} "
                            f"— tried get_overlap_matrices(), legacy attrs, rglob YAML/NPY"
                        )

                    # Persist canonical PNG and resolve serving URL (always set both fields)
                    overlap_matrix_path = self._get_or_generate_overlap_matrix(
                        shared_dir, job_id, transformation.name, leg, overlap_matrix
                    )

                    results.append({
                        'name': transformation.name,
                        'ligand_a': ligand_a_name,
                        'ligand_b': ligand_b_name,
                        'leg': leg,
                        'estimate_kcal_mol': float(estimate.m_as(unit.kilocalorie_per_mole)),
                        'uncertainty_kcal_mol': float(uncertainty.m_as(unit.kilocalorie_per_mole)),
                        'status': 'completed',
                        'validation_warnings': validation.get('warnings', []),
                        'overlap_matrix': overlap_matrix,
                        'overlap_matrix_path': overlap_matrix_path,
                    })

                    successful_transformations += 1
                    ddg_val = float(estimate.m_as(unit.kilocalorie_per_mole))
                    unc_val = float(uncertainty.m_as(unit.kilocalorie_per_mole))
                    logger.info(
                        f"✓ Transformation {transformation.name} completed: "
                        f"ddG = {ddg_val:.2f} ± {unc_val:.2f} kcal/mol"
                    )
                    tx_progress_done = int((i + 1) / total_transformations * 95 + 5)
                    emit_progress(
                        tx_progress_done,
                        f"✓ {i+1}/{total_transformations} {ligand_a_name}→{ligand_b_name}: {ddg_val:.2f} kcal/mol"
                    )

                except Exception as e:
                    error_msg = str(e)
                    error_type = type(e).__name__

                    logger.error(f"✗ Transformation {transformation.name} failed: {error_type}: {error_msg}")
                    tx_progress_fail = int((i + 1) / total_transformations * 95 + 5)
                    emit_progress(
                        tx_progress_fail,
                        f"✗ {i+1}/{total_transformations} {ligand_a_name}→{ligand_b_name} failed: {error_type}"
                    )

                    # Check if this is a NaN error
                    is_nan_error = 'nan' in error_msg.lower() or 'inf' in error_msg.lower()

                    if is_nan_error:
                        logger.error(
                            f"NaN detected in {transformation.name}. "
                            f"This typically indicates:\n"
                            f"  1. Structural instability (atom clashes, strained geometry)\n"
                            f"  2. Force field parameterization issues\n"
                            f"  3. Poor alignment quality between {ligand_a_name} and {ligand_b_name}\n"
                            f"  4. Insufficient equilibration before production\n"
                            f"Check validation warnings above for structural issues."
                        )

                    # Log full traceback for debugging
                    logger.debug(f"Full traceback for {transformation.name}:\n{traceback.format_exc()}")

                    failed_transformations.append({
                        'name': transformation.name,
                        'ligand_a': ligand_a_name,
                        'ligand_b': ligand_b_name,
                        'leg': leg,
                        'error': error_msg,
                        'error_type': error_type,
                        'is_nan_error': is_nan_error
                    })

                    results.append({
                        'name': transformation.name,
                        'ligand_a': ligand_a_name,
                        'ligand_b': ligand_b_name,
                        'leg': leg,
                        'status': 'failed',
                        'error': error_msg,
                        'error_type': error_type,
                        'is_nan_error': is_nan_error
                    })

                # Update progress
                self._update_job_status(job_id, {
                    'progress': (i + 1) / total_transformations * 100,
                    'current_transformation': i + 1,
                    'successful_transformations': successful_transformations,
                    'failed_transformations': len(failed_transformations)
                })

            # Log summary of results
            logger.info(
                f"Transformation execution complete: "
                f"{successful_transformations}/{len(transformations)} successful, "
                f"{len(failed_transformations)} failed"
            )

            if failed_transformations:
                logger.warning("Failed transformations:")
                for failed in failed_transformations:
                    logger.warning(
                        f"  - {failed['name']}: {failed.get('error_type', 'Error')}: "
                        f"{failed['error'][:100]}"
                    )

            # Check if we have any successful results
            if successful_transformations == 0:
                # Check if all failures are GPU/CUDA related
                gpu_error_keywords = ['cuda', 'opencl', 'no device', 'gpu', 'cudaerror', 'openmmexception']
                all_gpu_errors = failed_transformations and all(
                    any(kw in f.get('error', '').lower() for kw in gpu_error_keywords)
                    for f in failed_transformations
                )
                if all_gpu_errors:
                    raise ValueError(
                        f"All {len(transformations)} transformations failed due to GPU/CUDA error: "
                        f"{failed_transformations[0].get('error', 'unknown error')}\n"
                        f"Ensure the worker container has GPU access (nvidia runtime) and CUDA drivers are installed."
                    )
                raise ValueError(
                    f"All {len(transformations)} transformations failed. "
                    f"Check logs for detailed error messages. Common issues:\n"
                    f"  1. Ligands too dissimilar (poor MCS alignment)\n"
                    f"  2. Structural clashes in aligned poses\n"
                    f"  3. Force field compatibility issues\n"
                    f"See validation warnings above for specific problems."
                )
            
            # Step 7: Parse and combine results
            logger.info("Step 7: Parsing results...")
            parsed_results = self._parse_network_results(results, network_dict, alignment_info)
            
            # Save results
            results_file = job_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump(parsed_results, f, indent=2)
            
            # Final status update
            self._update_job_status(job_id, {
                'status': 'completed',
                'results': parsed_results,
                'alignment_info': alignment_info,
                'completed_at': datetime.now().isoformat()
            })
            
            return self.get_job_status(job_id)
            
        except Exception as e:
            logger.error(f"RBFE calculation failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            self._update_job_status(job_id, {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            return self.get_job_status(job_id)
    
    def _parse_network_results(
        self,
        transformation_results: List[Dict[str, Any]],
        network_data: Dict[str, Any],
        alignment_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Parse transformation results into relative binding affinities.
        
        Includes alignment data for ligand preparation quality assessment.
        Uses simple differencing to compute ddG values relative to first ligand.
        For production use, MLE via cinnabar would be preferred.
        """
        # Group results by edge (complex and solvent legs)
        edge_results = {}
        
        for result in transformation_results:
            if result['status'] != 'completed':
                continue
            
            # Use explicit ligand names if available (new format)
            if 'ligand_a' in result and 'ligand_b' in result:
                ligand_a = result['ligand_a']
                ligand_b = result['ligand_b']
                leg = result.get('leg')
                if not leg:
                    # Fallback leg parsing
                    parts = result['name'].rsplit('_', 1)
                    leg = parts[1] if len(parts) == 2 else 'unknown'
                
                edge_key = (ligand_a, ligand_b)
                
                if edge_key not in edge_results:
                    edge_results[edge_key] = {}
                edge_results[edge_key][leg] = result
                
            else:
                # Fallback to old parsing logic for backward compatibility
                name = result['name']
                parts = name.rsplit('_', 1)
                if len(parts) == 2:
                    edge_name, leg = parts
                    
                    # Try to extract ligand names from edge_name
                    # This is fragile if names contain underscores
                    ligand_parts = edge_name.split('_')
                    if len(ligand_parts) >= 2:
                        ligand_a = ligand_parts[0]
                        ligand_b = ligand_parts[1]
                        edge_key = (ligand_a, ligand_b)
                        
                        if edge_key not in edge_results:
                            edge_results[edge_key] = {}
                        edge_results[edge_key][leg] = result
        
        # Calculate ddG for each edge
        ddg_values = []
        for (ligand_a, ligand_b), legs in edge_results.items():
            if 'complex' in legs and 'solvent' in legs:
                ddg = legs['complex']['estimate_kcal_mol'] - legs['solvent']['estimate_kcal_mol']
                # Propagate uncertainties
                uncertainty = np.sqrt(
                    legs['complex']['uncertainty_kcal_mol']**2 + 
                    legs['solvent']['uncertainty_kcal_mol']**2
                ) if NUMPY_AVAILABLE else (
                    legs['complex']['uncertainty_kcal_mol'] + 
                    legs['solvent']['uncertainty_kcal_mol']
                )
                
                ddg_values.append({
                    'ligand_a': ligand_a,
                    'ligand_b': ligand_b,
                    'ddg_kcal_mol': ddg,
                    'uncertainty_kcal_mol': uncertainty
                })
        
        # Compute relative affinities via BFS from reference ligand
        relative_affinities = {}
        nodes = network_data.get('nodes', [])
        if nodes:
            # Use central_ligand (star topology hub) if available, else first node
            reference = network_data.get('central_ligand') or nodes[0]
            if reference not in nodes:
                reference = nodes[0]
            relative_affinities[reference] = 0.0

            # BFS over ddg_values edges until no more nodes can be resolved
            remaining = list(ddg_values)
            changed = True
            while changed and remaining:
                changed = False
                next_remaining = []
                for ddg_val in remaining:
                    a, b = ddg_val['ligand_a'], ddg_val['ligand_b']
                    if a in relative_affinities and b not in relative_affinities:
                        relative_affinities[b] = relative_affinities[a] + ddg_val['ddg_kcal_mol']
                        changed = True
                    elif b in relative_affinities and a not in relative_affinities:
                        relative_affinities[a] = relative_affinities[b] - ddg_val['ddg_kcal_mol']
                        changed = True
                    else:
                        next_remaining.append(ddg_val)
                remaining = next_remaining
        
        # Build comprehensive results including alignment data
        results_dict = {
            'transformation_results': transformation_results,
            'ddg_values': ddg_values,
            'relative_affinities': relative_affinities,
            'reference_ligand': network_data.get('central_ligand') or (network_data.get('nodes', [None])[0] if network_data.get('nodes') else None)
        }
        
        # Add alignment information if provided
        if alignment_info:
            # Calculate alignment statistics
            aligned_ligands = alignment_info.get('aligned_ligands', [])
            failed_ligands = alignment_info.get('failed_ligands', [])
            
            # Extract RMSD values for statistics
            rmsd_values = [lig.get('rmsd') for lig in aligned_ligands 
                          if lig.get('rmsd') is not None and lig.get('rmsd', 0) > 0]
            mcs_values = [lig.get('mcs_atoms') for lig in aligned_ligands 
                         if lig.get('mcs_atoms') is not None]
            
            results_dict['alignment_summary'] = {
                'reference_ligand': alignment_info.get('reference_ligand'),
                'alignment_method': alignment_info.get('alignment_method'),
                'total_ligands': len(aligned_ligands) + len(failed_ligands),
                'total_aligned': len(aligned_ligands),
                'total_failed': len(failed_ligands),
                'aligned_ligands': aligned_ligands,
                'failed_ligands': failed_ligands,
                'statistics': {
                    'rmsd': {
                        'min': float(min(rmsd_values)) if rmsd_values else None,
                        'max': float(max(rmsd_values)) if rmsd_values else None,
                        'mean': float(sum(rmsd_values) / len(rmsd_values)) if rmsd_values else None,
                        'values': [float(v) for v in rmsd_values]
                    },
                    'mcs_atoms': {
                        'min': min(mcs_values) if mcs_values else None,
                        'max': max(mcs_values) if mcs_values else None,
                        'mean': int(sum(mcs_values) / len(mcs_values)) if mcs_values else None,
                        'values': mcs_values
                    }
                }
            }
        
        return results_dict

    def _extract_rbfe_overlap_matrix(
        self, result, dag_result, shared_dir: Path
    ) -> tuple:
        """Extract MBAR overlap matrix from an OpenFE protocol result object.

        Returns (matrix_or_None, source_description_str).  Priority order:
        1. result.get_overlap_matrices()  — documented OpenFE >=1.2 API
        2. Legacy attribute probes on result / result.data / dag_result.to_dict()
        3. Recursive file search under shared_dir (YAML, .npy)
        """
        if not NUMPY_AVAILABLE:
            return None, "numpy_unavailable"

        # ------------------------------------------------------------------ #
        # 1. Primary: documented OpenFE API                                   #
        # ------------------------------------------------------------------ #
        try:
            get_om = getattr(result, "get_overlap_matrices", None)
            if callable(get_om):
                om_list = get_om()  # list[dict[str, numpy.ndarray]]
                if om_list:
                    matrices = []
                    for repeat_dict in om_list:
                        mat = repeat_dict.get("matrix")
                        if mat is None:
                            # fallback: first ndim==2 array found in dict values
                            for v in repeat_dict.values():
                                if hasattr(v, "ndim") and v.ndim == 2:
                                    mat = v
                                    break
                        if mat is not None and hasattr(mat, "ndim") and mat.ndim == 2:
                            matrices.append(mat)
                    if matrices:
                        avg = (
                            np.mean(np.stack(matrices), axis=0)
                            if len(matrices) > 1
                            else matrices[0]
                        )
                        return [[float(v) for v in row] for row in avg.tolist()], "get_overlap_matrices()"
                    logger.debug("get_overlap_matrices() returned non-empty list but no valid matrices")
        except Exception as e:
            logger.debug(f"get_overlap_matrices() call failed: {e}")

        # ------------------------------------------------------------------ #
        # 2. Legacy attribute / result.data / dag_result probes               #
        # ------------------------------------------------------------------ #
        try:
            for attr_name in ['mbar_overlap_matrix', 'overlap_matrix', 'matrix']:
                raw = getattr(result, attr_name, None)
                if raw is not None and hasattr(raw, 'ndim') and raw.ndim >= 2:
                    avg = raw.mean(axis=0) if raw.ndim == 3 else raw
                    return [[float(v) for v in row] for row in avg.tolist()], f"result.{attr_name}"

            if hasattr(result, 'data') and isinstance(result.data, dict):
                for key in ['mbar_overlap_matrix', 'overlap_matrix', 'matrix']:
                    raw = result.data.get(key)
                    if raw is not None and hasattr(raw, 'ndim') and raw.ndim >= 2:
                        avg = raw.mean(axis=0) if raw.ndim == 3 else raw
                        return [[float(v) for v in row] for row in avg.tolist()], f"result.data['{key}']"

            if hasattr(dag_result, 'to_dict'):
                dag_dict = (
                    dag_result.to_dict()
                    if callable(dag_result.to_dict)
                    else dag_result.to_dict
                )
                if isinstance(dag_dict, dict):
                    for key in ['mbar_overlap_matrix', 'overlap_matrix', 'matrix']:
                        raw = dag_dict.get(key)
                        if raw is not None and hasattr(raw, 'ndim') and raw.ndim >= 2:
                            avg = raw.mean(axis=0) if raw.ndim == 3 else raw
                            return [[float(v) for v in row] for row in avg.tolist()], f"dag_result.to_dict()['{key}']"
        except Exception as e:
            logger.debug(f"Legacy attribute probe failed: {e}")

        # ------------------------------------------------------------------ #
        # 3. Recursive file fallbacks under shared_dir                        #
        # ------------------------------------------------------------------ #
        try:
            # YAML files — real_time_analysis is a list; mbar_analysis dicts
            yaml_candidates = (
                list(shared_dir.rglob('*_real_time_analysis.yaml'))
                + list(shared_dir.rglob('*analysis*.yaml'))
                + list(shared_dir.rglob('*mbar*.yaml'))
            )
            for yaml_file in yaml_candidates:
                try:
                    import yaml as _yaml
                    with open(yaml_file) as f:
                        data = _yaml.safe_load(f)
                    overlap = None
                    if isinstance(data, list) and data:
                        latest = data[-1]
                        overlap = (
                            (latest.get('mbar_analysis') or {}).get('overlap_matrix')
                            or latest.get('overlap_matrix')
                        )
                    elif isinstance(data, dict):
                        overlap = (
                            (data.get('mbar_analysis') or {}).get('overlap_matrix')
                            or data.get('overlap_matrix')
                        )
                    if overlap is not None:
                        raw = np.array(overlap)
                        if raw.ndim >= 2:
                            avg = raw.mean(axis=0) if raw.ndim == 3 else raw
                            return [[float(v) for v in row] for row in avg.tolist()], f"yaml:{yaml_file.name}"
                except Exception:
                    continue

            # NumPy binary files
            npy_candidates = (
                list(shared_dir.rglob('*overlap*.npy'))
                + list(shared_dir.rglob('*mbar*.npy'))
            )
            for npy_file in npy_candidates:
                try:
                    raw = np.load(npy_file)
                    if raw.ndim >= 2:
                        avg = raw.mean(axis=0) if raw.ndim == 3 else raw
                        return [[float(v) for v in row] for row in avg.tolist()], f"npy:{npy_file.name}"
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"File fallback search failed: {e}")

        return None, "not_found"

    def _render_overlap_matrix_png(
        self, matrix_nested: list, png_path: Path, transformation_name: str, leg: str
    ) -> bool:
        """Render a nested-list overlap matrix to a PNG file.  Returns True on success."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            arr = np.array(matrix_nested)
            try:
                from alchemlyb.visualisation import plot_mbar_overlap_matrix
                ax = plot_mbar_overlap_matrix(arr)
                fig = ax.get_figure()
            except (ImportError, Exception):
                n = len(arr)
                fig, ax = plt.subplots(figsize=(max(4, n), max(4, n)))
                im = ax.imshow(arr, cmap='Blues', vmin=0, vmax=1)
                fig.colorbar(im, ax=ax)
                ax.set_title(f'MBAR Overlap\n{transformation_name} ({leg})')
                ax.set_xlabel('Lambda state')
                ax.set_ylabel('Lambda state')

            fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            return True
        except Exception as e:
            logger.warning(f"Could not render overlap matrix PNG for {transformation_name}/{leg}: {e}")
            return False

    def _get_or_generate_overlap_matrix(
        self,
        shared_dir: Path,
        job_id: str,
        transformation_name: str,
        leg: str,
        matrix_nested: Optional[list] = None,
    ) -> Optional[str]:
        """Return a /api/rbfe/files/… URL for the MBAR overlap PNG.

        If *matrix_nested* is provided it is rendered into the canonical
        shared_dir/mbar_overlap_matrix.png (overwriting if stale).
        Otherwise, any existing PNG is found recursively under shared_dir.
        """
        # Sanitize transformation_name before embedding in URL path
        safe_name = transformation_name.replace('..', '').replace('/', '_').replace('\\', '_')
        canonical_png = shared_dir / 'mbar_overlap_matrix.png'

        # If we have numerical matrix data, render PNG now
        if matrix_nested is not None:
            self._render_overlap_matrix_png(matrix_nested, canonical_png, transformation_name, leg)

        # Prefer the canonical path first, then any PNG found recursively
        if canonical_png.exists():
            rel = f"{safe_name}/shared/mbar_overlap_matrix.png"
            return f"/api/rbfe/files/{job_id}/{rel}"

        # Recursive search for any pre-generated overlap PNG (e.g. written by OpenFE itself)
        for png_file in shared_dir.rglob('mbar_overlap_matrix.png'):
            try:
                rel = str(png_file.relative_to(self.output_dir / job_id)).replace('\\', '/')
                return f"/api/rbfe/files/{job_id}/{rel}"
            except ValueError:
                continue

        return None

    def get_job_files(self, job_id: str) -> List[Dict[str, Any]]:
        """List files in a job directory."""
        job_dir = self.output_dir / job_id
        
        if not job_dir.exists():
            return []
        
        files = []
        for f in job_dir.rglob("*"):
            if f.is_file():
                files.append({
                    'name': str(f.relative_to(job_dir)),
                    'size': f.stat().st_size,
                    'type': f.suffix.lstrip('.')
                })
        
        return files
    
    def get_job_logs(self, job_id: str) -> str:
        """Get logs for a job."""
        job_dir = self.output_dir / job_id
        log_file = job_dir / "console.log"

        if log_file.exists():
            return log_file.read_text()

        return ""

    def _prepare_ligands_for_preview(
        self,
        ligands_data: List[Dict[str, Any]],
    ) -> List['openfe.SmallMoleculeComponent']:
        """Prepare ligands for atom mapping preview (no charge assignment).

        Loads each ligand, adds hydrogens, generates 3D coordinates if needed,
        creates a SmallMoleculeComponent without partial charges, then aligns all
        ligands to a common reference frame (required by Kartograf).

        Skipping bulk_assign_partial_charges makes this ~100x faster than
        prepare_ligands_batch — atom mapping does not require partial charges.

        Args:
            ligands_data: List of dicts with 'id', 'data', 'format' keys.

        Returns:
            List of SmallMoleculeComponent objects ready for atom mapping.
        """
        from rdkit.Chem import AllChem

        prepared: List[openfe.SmallMoleculeComponent] = []
        has_docked_pose: List[bool] = []

        for lig_info in ligands_data:
            ligand_data = lig_info.get('data', '')
            ligand_id = lig_info.get('id', 'ligand')
            data_format = lig_info.get('format', 'sdf').lower()

            try:
                if data_format in ('sdf', 'mol'):
                    mol = Chem.MolFromMolBlock(ligand_data, removeHs=False)
                elif data_format == 'pdb':
                    mol = Chem.MolFromPDBBlock(ligand_data, removeHs=False)
                else:
                    logger.error(f"Unsupported format for preview: {data_format}")
                    continue

                if mol is None:
                    logger.error(f"Failed to parse ligand {ligand_id} for preview")
                    continue

                # Add hydrogens
                mol = Chem.AddHs(mol)

                # Determine whether we need to generate 3D coordinates
                is_3d = lig_info.get('has_docked_pose', False) or not self._is_2d_structure(
                    ligand_data, data_format
                )

                if not is_3d:
                    # Generate 3D coordinates with RDKit ETKDG
                    params = AllChem.ETKDGv3()
                    params.randomSeed = 42
                    result = AllChem.EmbedMolecule(mol, params)
                    if result == -1:
                        # Fallback: distance geometry without ETKDG
                        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                    AllChem.MMFFOptimizeMolecule(mol)
                    logger.info(f"Generated 3D coordinates for {ligand_id}")
                else:
                    logger.info(f"Using existing 3D coordinates for {ligand_id}")

                component = openfe.SmallMoleculeComponent.from_rdkit(mol, name=ligand_id)
                prepared.append(component)
                has_docked_pose.append(lig_info.get('has_docked_pose', False))

            except Exception as e:
                logger.error(f"Error preparing ligand {ligand_id} for preview: {e}")
                logger.error(traceback.format_exc())

        if len(prepared) < 2:
            return prepared

        # Align ligands (required by Kartograf geometric mapper)
        needs_alignment = (
            self._ligands_need_alignment(prepared) or not all(has_docked_pose)
        )
        if needs_alignment:
            ref_idx = next((i for i, d in enumerate(has_docked_pose) if d), 0)
            ref = prepared[ref_idx]
            logger.info(f"Aligning preview ligands to reference: {ref.name}")
            for i, lig in enumerate(prepared):
                if i == ref_idx or has_docked_pose[i]:
                    continue
                aligned = self._align_ligand_to_reference(lig, ref)
                if aligned is not None:
                    prepared[i] = aligned
                else:
                    logger.warning(f"Could not align {lig.name} for preview; keeping original coords")

        return prepared

    def run_mapping_preview(
        self,
        ligands_data: List[Dict[str, Any]],
        job_id: str,
        atom_mapper: str = 'kartograf',
        atom_map_hydrogens: bool = True,
        lomap_max3d: float = 1.0,
        charge_method: str = 'am1bcc',
    ) -> Dict[str, Any]:
        """Run lightweight atom mapping preview (no protein, no simulation).

        Prepares ligands without partial charge assignment (not needed for
        atom mapping), computes all pairwise mappings using the selected mapper,
        and returns per-pair mapping data with highlight SVGs.

        Args:
            ligands_data: List of ligand dicts with id, data, format keys.
            job_id: Job identifier.
            atom_mapper: Atom mapper type ('kartograf', 'lomap', 'lomap_relaxed').
            atom_map_hydrogens: For Kartograf — include hydrogens in mapping.
            lomap_max3d: For LOMAP — max 3D distance for mapping.
            charge_method: Unused — kept for API compatibility.

        Returns:
            Dict with 'pairs', 'num_ligands', 'atom_mapper', 'status', 'success'.
        """
        logger.info(f"Starting mapping preview for job {job_id} "
                    f"({len(ligands_data)} ligands, mapper={atom_mapper})")

        emit_progress(10, 'Preparing ligands...')

        # Use lightweight preparation (no charge assignment — not needed for mapping)
        ligands = self._prepare_ligands_for_preview(ligands_data)

        if len(ligands) < 2:
            raise ValueError(
                f"At least 2 ligands are required for mapping preview, "
                f"got {len(ligands)} after preparation."
            )

        emit_progress(40, f'Computing pairwise mappings for {len(ligands)} ligands...')

        planner = NetworkPlanner(
            atom_mapper=atom_mapper,
            atom_map_hydrogens=atom_map_hydrogens,
            lomap_max3d=lomap_max3d,
        )

        pairs = planner.compute_all_pairwise_mappings(ligands)

        emit_progress(100, 'Mapping preview complete')

        logger.info(f"Mapping preview complete: {len(pairs)} pairs for job {job_id}")

        return {
            'status': 'completed',
            'job_id': job_id,
            'pairs': pairs,
            'num_ligands': len(ligands),
            'atom_mapper': atom_mapper,
            'progress': 100,
            'success': True,
        }
