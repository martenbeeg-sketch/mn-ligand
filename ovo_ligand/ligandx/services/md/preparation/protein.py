"""
Protein preparation module for MD optimization.

All functions return JSON-serializable data (dicts/strings/lists).
No unserialized objects (Topology, Context, etc.) are passed between functions.
"""

import logging
from typing import Dict, Any, Optional
from ovo_ligand.ligandx.lib.chemistry import get_pdb_parser, get_protein_preparer

logger = logging.getLogger(__name__)


class ProteinPreparation:
    """Handles protein preparation for MD simulations with proper serialization."""
    
    def __init__(self):
        """Initialize protein preparation utilities."""
        self.pdb_parser = get_pdb_parser()
        self.protein_preparer = get_protein_preparer()
    
    def prepare_protein(self, pdb_data: str, pdb_id: str = "protein") -> Dict[str, Any]:
        """
        Prepare protein structure for MD simulation.
        
        Args:
            pdb_data: PDB format string
            pdb_id: Protein identifier
            
        Returns:
            Dict with keys:
                - 'success': bool
                - 'pdb_data': str (cleaned PDB)
                - 'error': str (if failed)
                - 'warnings': list of str
        """
        try:
            logger.info(f"Preparing protein: {pdb_id}")
            
            # Parse structure
            try:
                structure = self.pdb_parser.parse_string(pdb_data)
            except Exception as e:
                logger.error(f"Failed to parse PDB: {e}")
                return {
                    'success': False,
                    'error': f"PDB parsing failed: {str(e)}",
                    'pdb_data': None
                }
            
            # Clean protein structure
            try:
                cleaned_result = self.protein_preparer.clean_structure_staged(
                    pdb_data,
                    remove_heterogens=True,
                    remove_water=True,
                    add_missing_residues=True,
                    add_missing_atoms=True,
                    add_missing_hydrogens=True,
                    ph=7.4,
                    add_solvation=False,
                    keep_ligands=False
                )
                
                # Extract final cleaned stage
                stages = cleaned_result.get('stages', {})
                stage_info = cleaned_result.get('stage_info', {})
                
                if not stages:
                    logger.warning("No cleaned stages returned")
                    cleaned_pdb = pdb_data
                else:
                    # Get the final stage (highest step number)
                    final_stage = max(stage_info.items(), key=lambda x: x[1].get('step', 0))
                    cleaned_pdb = stages.get(final_stage[0], pdb_data)
                
                logger.info(f"[COMPLETE] Protein preparation completed for {pdb_id}")
                
                return {
                    'success': True,
                    'pdb_data': cleaned_pdb,
                    'error': None,
                    'warnings': []
                }
                
            except Exception as e:
                logger.warning(f"Protein cleaning failed: {e}, using original structure")
                return {
                    'success': True,
                    'pdb_data': pdb_data,
                    'error': None,
                    'warnings': [f"Protein cleaning failed: {str(e)}"]
                }
        
        except Exception as e:
            logger.error(f"Protein preparation failed: {e}")
            return {
                'success': False,
                'error': f"Protein preparation failed: {str(e)}",
                'pdb_data': None,
                'warnings': []
            }
    
    def validate_protein_structure(self, pdb_data: str) -> Dict[str, Any]:
        """
        Validate protein structure for MD simulation.
        
        Args:
            pdb_data: PDB format string
            
        Returns:
            Dict with keys:
                - 'valid': bool
                - 'atom_count': int
                - 'residue_count': int
                - 'issues': list of str
        """
        try:
            structure = self.pdb_parser.parse_string(pdb_data)
            
            # Count atoms and residues
            atom_count = 0
            residue_count = 0
            issues = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        residue_count += 1
                        for atom in residue:
                            atom_count += 1
            
            if atom_count == 0:
                issues.append("No atoms found in structure")
            if residue_count == 0:
                issues.append("No residues found in structure")
            
            return {
                'valid': len(issues) == 0,
                'atom_count': atom_count,
                'residue_count': residue_count,
                'issues': issues
            }
        
        except Exception as e:
            logger.error(f"Structure validation failed: {e}")
            return {
                'valid': False,
                'atom_count': 0,
                'residue_count': 0,
                'issues': [f"Validation error: {str(e)}"]
            }
