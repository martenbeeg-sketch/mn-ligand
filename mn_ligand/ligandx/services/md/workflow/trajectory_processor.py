"""
Trajectory processor module for MD optimization.

Handles trajectory loading, processing, and conversion for web delivery.
"""

import os
import logging
import tempfile
from typing import Dict, Any, Optional, Tuple

from ..utils.pdb_utils import normalize_nonpolymer_residue_ids_in_pdb_block

logger = logging.getLogger(__name__)


class TrajectoryProcessorRunner:
    """Processes MD trajectories for analysis and visualization."""
    
    def __init__(self, output_dir: str = "data/md_outputs"):
        """
        Initialize trajectory processor.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
    
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
        Load, subsample, clean, and convert a trajectory for web delivery.
        
        Transforms OpenMM simulation output (DCD + PDB topology) into
        a multi-model PDB string that can be animated in 3dmol.js.
        
        Args:
            dcd_path: Path to the trajectory DCD file
            pdb_path: Path to the topology PDB file
            stride: Interval at which to sample frames
            align: Whether to superpose the trajectory on the first frame
            remove_solvent_flag: Whether to remove solvent and ions
            include_unitcell: Whether to extract unit cell information
            
        Returns:
            Dict containing:
                - 'pdb_data': String with trajectory in multi-model PDB format
                - 'unitcell_data': List of unit cell vectors for each frame
                - 'error': Error message if processing fails
        """
        try:
            # Check if MDTraj is available
            try:
                import mdtraj as md
            except ImportError:
                logger.error("MDTraj not available - cannot process trajectory")
                return {'pdb_data': '', 'error': 'MDTraj not available', 'unitcell_data': None}
            
            logger.info(f"Processing trajectory: {dcd_path} with topology: {pdb_path}")
            logger.info(f"Settings: stride={stride}, align={align}, remove_solvent={remove_solvent_flag}")
            
            # Load the trajectory with temporal subsampling
            traj = md.load_dcd(dcd_path, top=pdb_path, stride=stride)
            logger.info(f"Loaded {len(traj)} frames, {traj.n_atoms} atoms")
            
            # Extract unit cell information if requested
            unitcell_vectors = None
            if include_unitcell and traj.unitcell_vectors is not None:
                unitcell_vectors = traj.unitcell_vectors.tolist()
                logger.info(f"Extracted unit cell vectors for {len(unitcell_vectors)} frames")
            
            # Apply solvent removal if requested
            if remove_solvent_flag:
                initial_atoms = traj.n_atoms
                traj.remove_solvent(inplace=True)
                logger.info(f"Removed solvent: {initial_atoms} -> {traj.n_atoms} atoms")
            
            # Apply periodic imaging first so molecules stay contiguous in view/export.
            # This should happen regardless of alignment preference.
            if traj.unitcell_lengths is not None:
                try:
                    anchor_molecules = []
                    protein_sel = traj.topology.select('protein')
                    molecules = traj.topology.find_molecules()
                    if len(protein_sel) > 10:
                        protein_atom_set = set(protein_sel)
                        anchor_molecules = [
                            sorted(list(mol), key=lambda a: a.index)
                            for mol in molecules
                            if any(atom.index in protein_atom_set for atom in mol)
                        ]
                        if anchor_molecules:
                            logger.info(f"Anchoring PBC imaging to {len(anchor_molecules)} protein molecules")
                    if not anchor_molecules and len(molecules) > 0:
                        largest_mol = max(molecules, key=len)
                        anchor_molecules = [sorted(list(largest_mol), key=lambda a: a.index)]
                        logger.info(f"Fallback: Anchoring PBC imaging to largest molecule ({len(largest_mol)} atoms)")

                    if anchor_molecules:
                        traj.image_molecules(inplace=True, anchor_molecules=anchor_molecules)
                    else:
                        traj.image_molecules(inplace=True)
                except Exception as e:
                    logger.warning(f"Could not apply periodic boundary imaging: {e}")
            else:
                logger.warning("No unit cell information found in trajectory. Skipping PBC imaging.")

            # Optional alignment (after imaging).
            if align:
                logger.info("Applying trajectory alignment...")
                try:
                    # Try to align on protein CA, then protein, then largest molecule
                    molecules = traj.topology.find_molecules()
                    protein_sel = traj.topology.select('protein')
                    anchor_molecules = []
                    if len(protein_sel) > 10:
                        protein_atom_set = set(protein_sel)
                        anchor_molecules = [
                            sorted(list(mol), key=lambda a: a.index)
                            for mol in molecules
                            if any(atom.index in protein_atom_set for atom in mol)
                        ]

                        align_indices = []
                        if len(protein_sel) > 0:
                            protein_ca = traj.topology.select('protein and name CA')
                            if len(protein_ca) > 0:
                                align_indices = protein_ca
                                logger.info("Aligning trajectory on protein alpha carbons")
                            else:
                                align_indices = protein_sel
                                logger.info("Aligning trajectory on protein atoms")
                        elif anchor_molecules:
                            # Flatten anchor molecules to get atom indices
                            anchor_atoms = [atom.index for mol in anchor_molecules for atom in mol]
                            align_indices = anchor_atoms
                            logger.info(f"Aligning trajectory on anchor molecule ({len(anchor_atoms)} atoms)")
                        
                        if len(align_indices) > 0:
                            traj.superpose(traj, 0, atom_indices=align_indices)
                        else:
                            traj.superpose(traj, 0)

                        logger.info("[COMPLETE] Trajectory alignment completed")
                except Exception as e:
                    logger.warning(f"Could not apply trajectory alignment: {e}")
            
            # Convert to multi-model PDB string
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_file:
                temp_pdb_path = temp_file.name
            
            try:
                traj.save_pdb(temp_pdb_path)
                
                with open(temp_pdb_path, 'r') as f:
                    pdb_string = f.read()
                pdb_string = normalize_nonpolymer_residue_ids_in_pdb_block(pdb_string)
                
                logger.info(f"[COMPLETE] Trajectory converted to multi-model PDB ({len(pdb_string)} characters)")
                
                return {
                    'pdb_data': pdb_string,
                    'unitcell_data': unitcell_vectors,
                    'error': None
                }
                
            finally:
                if os.path.exists(temp_pdb_path):
                    os.unlink(temp_pdb_path)
            
        except Exception as e:
            import traceback
            logger.error(f"Trajectory processing error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'pdb_data': '', 'error': str(e), 'unitcell_data': None}
    
    def get_trajectory_files(self, system_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the trajectory and topology file paths for a given system.
        
        Args:
            system_id: System identifier used in optimization
            
        Returns:
            Tuple of (dcd_path, pdb_path) or (None, None) if not found
        """
        try:
            dcd_path = os.path.join(self.output_dir, f"{system_id}_npt_equilibration.dcd")
            pdb_path = os.path.join(self.output_dir, f"{system_id}_npt_final.pdb")
            
            if os.path.exists(dcd_path) and os.path.exists(pdb_path):
                logger.info(f"Found trajectory files: DCD={dcd_path}, PDB={pdb_path}")
                return dcd_path, pdb_path
            else:
                logger.warning(f"Trajectory files not found for system {system_id}")
                logger.warning(f"  DCD exists: {os.path.exists(dcd_path)}")
                logger.warning(f"  PDB exists: {os.path.exists(pdb_path)}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error finding trajectory files for {system_id}: {e}")
            return None, None
    
    def get_trajectory_info(self, dcd_path: str, pdb_path: str) -> Dict[str, Any]:
        """
        Get information about trajectory files.
        
        Args:
            dcd_path: Path to DCD file
            pdb_path: Path to PDB file
            
        Returns:
            Dict with trajectory information
        """
        info = {
            'dcd_path': dcd_path,
            'pdb_path': pdb_path,
            'dcd_size_mb': 0,
            'pdb_size_mb': 0,
            'dcd_exists': False,
            'pdb_exists': False,
            'frame_count': None,
            'atom_count': None
        }
        
        try:
            if os.path.exists(dcd_path):
                info['dcd_size_mb'] = os.path.getsize(dcd_path) / (1024 * 1024)
                info['dcd_exists'] = True
            
            if os.path.exists(pdb_path):
                info['pdb_size_mb'] = os.path.getsize(pdb_path) / (1024 * 1024)
                info['pdb_exists'] = True
            
            # Try to get frame count using MDTraj
            if info['dcd_exists'] and info['pdb_exists']:
                try:
                    import mdtraj as md
                    traj = md.load_dcd(dcd_path, top=pdb_path)
                    info['frame_count'] = len(traj)
                    info['atom_count'] = traj.n_atoms
                except ImportError:
                    logger.warning("MDTraj not available for trajectory info")
                except Exception as e:
                    logger.warning(f"Could not load trajectory for info: {e}")
                    
        except Exception as e:
            logger.warning(f"Could not get file info: {e}")
        
        return info
