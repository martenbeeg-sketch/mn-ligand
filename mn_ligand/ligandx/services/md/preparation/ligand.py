"""
Ligand preparation module for MD optimization.

All functions return JSON-serializable data (dicts/strings/lists).
No unserialized objects (Molecule, ForceField, etc.) are passed between functions.
"""

import logging
from typing import Dict, Any, Optional
from mn_ligand.ligandx.lib.chemistry import get_ligand_preparer

logger = logging.getLogger(__name__)


class LigandPreparation:
    """Handles ligand preparation for MD simulations."""
    
    def __init__(self):
        """Initialize ligand preparation utilities."""
        self.ligand_preparer = get_ligand_preparer()
    
    def prepare_ligand_from_smiles(
        self,
        smiles: str,
        ligand_id: str = "ligand",
        generate_conformer: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare ligand from SMILES string.
        
        Args:
            smiles: SMILES string
            ligand_id: Ligand identifier
            generate_conformer: Whether to generate 3D conformer
            
        Returns:
            Dict with keys:
                - 'success': bool
                - 'sdf_data': str (SDF format)
                - 'pdb_data': str (PDB format, optional)
                - 'error': str (if failed)
        """
        try:
            logger.info(f"Preparing ligand from SMILES: {smiles}")
            
            # Convert SMILES to 3D structure using shared utilities
            try:
                result = self.utils.smiles_to_3d(
                    smiles,
                    add_hydrogens=True,
                    generate_conformer=generate_conformer
                )
                
                if not result or not result.get('success'):
                    error_msg = result.get('error', 'Unknown error') if result else 'No result'
                    logger.error(f"SMILES conversion failed: {error_msg}")
                    return {
                        'success': False,
                        'error': f"SMILES conversion failed: {error_msg}",
                        'sdf_data': None,
                        'pdb_data': None
                    }
                
                sdf_data = result.get('sdf_data')
                pdb_data = result.get('pdb_data')
                
                logger.info(f"[COMPLETE] Ligand preparation completed for {ligand_id}")
                
                return {
                    'success': True,
                    'sdf_data': sdf_data,
                    'pdb_data': pdb_data,
                    'error': None
                }
            
            except Exception as e:
                logger.error(f"SMILES conversion error: {e}")
                return {
                    'success': False,
                    'error': f"SMILES conversion failed: {str(e)}",
                    'sdf_data': None,
                    'pdb_data': None
                }
        
        except Exception as e:
            logger.error(f"Ligand preparation failed: {e}")
            return {
                'success': False,
                'error': f"Ligand preparation failed: {str(e)}",
                'sdf_data': None,
                'pdb_data': None
            }
    
    def prepare_ligand_from_structure(
        self,
        structure_data: str,
        ligand_id: str = "ligand",
        data_format: str = "sdf"
    ) -> Dict[str, Any]:
        """
        Prepare ligand from structure file (SDF, PDB, MOL).
        
        Args:
            structure_data: Structure data as string
            ligand_id: Ligand identifier
            data_format: Format ('sdf', 'pdb', 'mol')
            
        Returns:
            Dict with keys:
                - 'success': bool
                - 'sdf_data': str (SDF format)
                - 'pdb_data': str (PDB format, optional)
                - 'error': str (if failed)
        """
        try:
            logger.info(f"Preparing ligand from {data_format.upper()}: {ligand_id}")
            
            # Validate format
            if data_format.lower() not in ['sdf', 'pdb', 'mol']:
                return {
                    'success': False,
                    'error': f"Unsupported format: {data_format}",
                    'sdf_data': None,
                    'pdb_data': None
                }
            
            # For SDF, return as-is (already in correct format)
            if data_format.lower() == 'sdf':
                logger.info(f"[COMPLETE] Ligand preparation completed for {ligand_id}")
                return {
                    'success': True,
                    'sdf_data': structure_data,
                    'pdb_data': None,
                    'error': None
                }
            
            # For PDB or MOL, try to convert to SDF
            try:
                # Use RDKit to convert if available
                from rdkit import Chem
                
                if data_format.lower() == 'pdb':
                    mol = Chem.MolFromPDBBlock(structure_data, removeHs=False)
                else:  # mol
                    mol = Chem.MolFromMolBlock(structure_data, removeHs=False)
                
                if mol is None:
                    logger.error(f"Failed to parse {data_format.upper()}")
                    return {
                        'success': False,
                        'error': f"Failed to parse {data_format.upper()}",
                        'sdf_data': None,
                        'pdb_data': None
                    }
                
                # Convert to SDF
                sdf_data = Chem.MolToMolBlock(mol)
                
                logger.info(f"[COMPLETE] Ligand preparation completed for {ligand_id}")
                
                return {
                    'success': True,
                    'sdf_data': sdf_data,
                    'pdb_data': structure_data if data_format.lower() == 'pdb' else None,
                    'error': None
                }
            
            except ImportError:
                logger.warning("RDKit not available, returning structure as-is")
                return {
                    'success': True,
                    'sdf_data': structure_data if data_format.lower() == 'sdf' else None,
                    'pdb_data': structure_data if data_format.lower() == 'pdb' else None,
                    'error': None
                }
        
        except Exception as e:
            logger.error(f"Ligand preparation failed: {e}")
            return {
                'success': False,
                'error': f"Ligand preparation failed: {str(e)}",
                'sdf_data': None,
                'pdb_data': None
            }
    
    def validate_ligand_structure(self, structure_data: str, data_format: str = "sdf") -> Dict[str, Any]:
        """
        Validate ligand structure.
        
        Args:
            structure_data: Structure data as string
            data_format: Format ('sdf', 'pdb', 'mol')
            
        Returns:
            Dict with keys:
                - 'valid': bool
                - 'atom_count': int
                - 'issues': list of str
        """
        try:
            from rdkit import Chem
            
            # Parse structure
            if data_format.lower() == 'pdb':
                mol = Chem.MolFromPDBBlock(structure_data, removeHs=False)
            elif data_format.lower() == 'mol':
                mol = Chem.MolFromMolBlock(structure_data, removeHs=False)
            else:  # sdf
                mol = Chem.MolFromMolBlock(structure_data, removeHs=False)
            
            if mol is None:
                return {
                    'valid': False,
                    'atom_count': 0,
                    'issues': [f"Failed to parse {data_format.upper()}"]
                }
            
            atom_count = mol.GetNumAtoms()
            issues = []
            
            if atom_count == 0:
                issues.append("No atoms found in structure")
            
            return {
                'valid': len(issues) == 0,
                'atom_count': atom_count,
                'issues': issues
            }
        
        except ImportError:
            logger.warning("RDKit not available, skipping validation")
            return {
                'valid': True,
                'atom_count': -1,
                'issues': ["RDKit not available for validation"]
            }
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'valid': False,
                'atom_count': 0,
                'issues': [f"Validation error: {str(e)}"]
            }
