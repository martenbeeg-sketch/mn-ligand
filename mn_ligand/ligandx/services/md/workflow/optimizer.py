"""
MD Optimizer workflow module.

This module contains the heavy OpenMM/OpenFF simulation logic extracted from service.py.
All public methods return JSON-serializable dicts. Internal methods may use OpenMM objects
but they are not exposed outside this module.
"""

import os
import logging
import traceback
from typing import Dict, Any, Optional
from io import StringIO

logger = logging.getLogger(__name__)


class MDOptimizer:
    """
    MD optimization workflow handler.
    
    This class encapsulates the heavy simulation logic including:
    - Ligand preparation (SMILES and structure-based)
    - Protein preparation
    - System creation and solvation
    - Equilibration protocol
    - Trajectory processing
    
    All public methods return JSON-serializable dicts.
    """
    
    def __init__(self, output_dir: str, environment_status: Dict[str, Any], utils):
        """
        Initialize MD optimizer.
        
        Args:
            output_dir: Directory for output files
            environment_status: Environment validation status dict
            utils: Molecular utilities instance
        """
        self.output_dir = output_dir
        self.environment_status = environment_status
        self.utils = utils
        self.ligand_ff = None
        self.protein_ff = None
        
        # Initialize force fields
        self._initialize_force_fields()
    
    def _initialize_force_fields(self):
        """Initialize force fields with proper error handling."""
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
    
    def _assign_partial_charges_with_fallback(self, molecule):
        """
        Assign partial charges with MMFF94-first fallback handling.
        
        Args:
            molecule: OpenFF Molecule object
            
        Returns:
            OpenFF Molecule with assigned charges, or None if all methods fail
        """
        if not molecule:
            return None
            
        try:
            # Method 1: MMFF94 (primary method - reliable and fast)
            try:
                logger.info("Attempting MMFF94 charge assignment...")
                molecule.assign_partial_charges(partial_charge_method="mmff94")
                logger.info("[COMPLETE] MMFF94 charges assigned successfully")
                return molecule
            except Exception as e:
                logger.warning(f"MMFF94 charge assignment failed: {e}")
            
            # Method 2: Gasteiger (fallback only)
            try:
                logger.info("Attempting Gasteiger charge assignment...")
                molecule.assign_partial_charges(partial_charge_method="gasteiger")
                logger.info("[COMPLETE] Gasteiger charges assigned successfully")
                return molecule
            except Exception as e:
                logger.error(f"Gasteiger charge assignment failed: {e}")
            
            logger.error("All charge assignment methods failed")
            return None
            
        except Exception as e:
            logger.error(f"Charge assignment with fallback failed: {e}")
            return None
    
    def prepare_ligand_from_smiles(self, smiles: str, ligand_id: str = "ligand", 
                                   generate_conformer: bool = True) -> Dict[str, Any]:
        """
        Prepare ligand from SMILES string.
        
        Args:
            smiles: SMILES string of the ligand
            ligand_id: Identifier for the ligand
            generate_conformer: Whether to generate a new 3D conformer
        
        Returns:
            Dict with 'success', 'molecule' (internal), 'error'
        """
        logger.info(f"=== PREPARING LIGAND FROM SMILES ===")
        logger.info(f"SMILES: {smiles}")
        
        if not self.environment_status.get('rdkit', False):
            return {"success": False, "error": "RDKit not available", "molecule": None}
        
        if not self.environment_status.get('openff', False):
            return {"success": False, "error": "OpenFF Toolkit not available", "molecule": None}
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from openff.toolkit import Molecule
            
            # Parse SMILES
            rdkit_mol = Chem.MolFromSmiles(smiles)
            if rdkit_mol is None:
                return {"success": False, "error": f"Invalid SMILES: {smiles}", "molecule": None}
            
            # Add hydrogens
            rdkit_mol = Chem.AddHs(rdkit_mol)
            rdkit_mol.SetProp("_Name", ligand_id)
            
            # Generate 3D conformer
            if generate_conformer:
                params = AllChem.EmbedParameters()
                params.useRandomCoords = True
                result = AllChem.EmbedMolecule(rdkit_mol, params)
                if result != 0:
                    AllChem.EmbedMolecule(rdkit_mol, useRandomCoords=True)
                AllChem.MMFFOptimizeMolecule(rdkit_mol)
            
            # Create OpenFF Molecule
            try:
                molecule = Molecule.from_rdkit(rdkit_mol)
            except Exception as e:
                if "stereochemistry" in str(e).lower():
                    mol_copy = Chem.Mol(rdkit_mol)
                    Chem.AssignStereochemistry(mol_copy, cleanIt=True, force=True)
                    molecule = Molecule.from_rdkit(mol_copy)
                else:
                    raise
            
            # Assign charges
            charged_molecule = self._assign_partial_charges_with_fallback(molecule)
            if not charged_molecule:
                return {"success": False, "error": "Charge assignment failed", "molecule": None}
            
            logger.info("[COMPLETE] Ligand preparation from SMILES completed")
            return {"success": True, "molecule": charged_molecule, "error": None}
            
        except Exception as e:
            logger.error(f"Ligand preparation failed: {e}")
            return {"success": False, "error": str(e), "molecule": None}
    
    def prepare_ligand_from_structure(self, structure_data: str, ligand_id: str = "ligand",
                                      data_format: str = "sdf", preserve_pose: bool = True) -> Dict[str, Any]:
        """
        Prepare ligand from structure data (SDF/MOL/PDB).
        
        Args:
            structure_data: Structure data string
            ligand_id: Identifier for the ligand
            data_format: Format ('sdf', 'mol', 'pdb')
            preserve_pose: Whether to preserve original 3D pose
        
        Returns:
            Dict with 'success', 'molecule' (internal), 'error'
        """
        logger.info(f"=== PREPARING LIGAND FROM STRUCTURE ===")
        logger.info(f"Format: {data_format}, Preserve pose: {preserve_pose}")
        
        if not self.environment_status.get('rdkit', False):
            return {"success": False, "error": "RDKit not available", "molecule": None}
        
        if not self.environment_status.get('openff', False):
            return {"success": False, "error": "OpenFF Toolkit not available", "molecule": None}
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from openff.toolkit import Molecule
            
            # Parse structure based on format
            if data_format.lower() == 'sdf' or data_format.lower() == 'mol':
                rdkit_mol = Chem.MolFromMolBlock(structure_data, removeHs=False)
            elif data_format.lower() == 'pdb':
                rdkit_mol = Chem.MolFromPDBBlock(structure_data, removeHs=False)
            else:
                return {"success": False, "error": f"Unsupported format: {data_format}", "molecule": None}
            
            if rdkit_mol is None:
                return {"success": False, "error": f"Failed to parse {data_format}", "molecule": None}
            
            # Add hydrogens if needed
            rdkit_mol = Chem.AddHs(rdkit_mol, addCoords=True)
            rdkit_mol.SetProp("_Name", ligand_id)
            
            # Optimize if not preserving pose
            if not preserve_pose:
                AllChem.MMFFOptimizeMolecule(rdkit_mol)
            
            # Create OpenFF Molecule
            try:
                molecule = Molecule.from_rdkit(rdkit_mol)
            except Exception as e:
                if "stereochemistry" in str(e).lower():
                    mol_copy = Chem.Mol(rdkit_mol)
                    Chem.AssignStereochemistry(mol_copy, cleanIt=True, force=True)
                    molecule = Molecule.from_rdkit(mol_copy)
                else:
                    raise
            
            # Assign charges
            charged_molecule = self._assign_partial_charges_with_fallback(molecule)
            if not charged_molecule:
                return {"success": False, "error": "Charge assignment failed", "molecule": None}
            
            logger.info("[COMPLETE] Ligand preparation from structure completed")
            return {"success": True, "molecule": charged_molecule, "error": None}
            
        except Exception as e:
            logger.error(f"Ligand preparation failed: {e}")
            return {"success": False, "error": str(e), "molecule": None}
    
    def prepare_protein(self, pdb_data: str, pdb_id: str = "protein") -> Dict[str, Any]:
        """
        Prepare protein structure for MD simulation.
        
        Args:
            pdb_data: PDB format string
            pdb_id: Protein identifier
            
        Returns:
            Dict with 'success', 'pdb_path', 'pdb_data', 'error'
        """
        logger.info(f"=== PREPARING PROTEIN {pdb_id} ===")
        
        try:
            # Parse structure
            structure = self.utils.parse_pdb_string(pdb_data, pdb_id)
            components = self.utils.identify_structure_components(structure)
            
            if not components.get("protein"):
                return {"success": False, "error": "No protein residues found", "pdb_path": None, "pdb_data": None}
            
            # Extract protein only
            protein_pdb = self.utils.extract_residues_as_pdb(structure, components["protein"])
            
            # Clean protein structure
            cleaning_result = self.utils.clean_protein_structure_staged(
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
            
            return {
                "success": True,
                "pdb_path": output_path,
                "pdb_data": cleaned_pdb,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Protein preparation failed: {e}")
            return {"success": False, "error": str(e), "pdb_path": None, "pdb_data": None}
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get current workflow status and environment info.
        
        Returns:
            Dict with environment status (JSON-serializable)
        """
        return {
            "environment": {
                "openff": self.environment_status.get('openff', False),
                "openmm": self.environment_status.get('openmm', False),
                "rdkit": self.environment_status.get('rdkit', False),
                "pdbfixer": self.environment_status.get('pdbfixer', False),
                "platforms": self.environment_status.get('openmm_platforms', [])
            },
            "force_fields": {
                "ligand_ff_loaded": self.ligand_ff is not None,
                "protein_ff_loaded": self.protein_ff is not None
            },
            "output_dir": self.output_dir
        }
