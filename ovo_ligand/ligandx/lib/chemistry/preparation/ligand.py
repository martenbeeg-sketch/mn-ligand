"""
Ligand preparation utilities.

Provides functionality for preparing ligand molecules including
hydrogen addition, 3D coordinate generation, and optimization.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Optional RDKit import
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Ligand preparation will be disabled.")
    RDKIT_AVAILABLE = False


class LigandPreparer:
    """Utilities for preparing ligand molecules."""
    
    def __init__(self):
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - ligand preparation features limited")
    
    def prepare(self, mol, add_hs: bool = True, generate_3d: bool = True, 
                optimize: bool = True) -> Any:
        """
        Prepare a ligand molecule for simulation/docking.
        
        Args:
            mol: RDKit molecule object
            add_hs: Whether to add hydrogens
            generate_3d: Whether to generate 3D coordinates
            optimize: Whether to perform MMFF optimization
            
        Returns:
            Prepared RDKit molecule
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit not available for ligand preparation")
            
        try:
            # Add hydrogens
            if add_hs:
                mol = Chem.AddHs(mol, addCoords=True)
            
            # Generate 3D coordinates if needed
            if generate_3d:
                needs_3d = self._needs_3d_coordinates(mol)
                if needs_3d:
                    logger.info("Generating 3D coordinates for molecule (was 2D or had no conformers)")
                    # Clear existing conformers if they are 2D
                    if mol.GetNumConformers() > 0:
                        mol.RemoveAllConformers()
                    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                    if result == -1:
                        # Embedding failed, try with random coordinates
                        logger.warning("Standard embedding failed, trying with random coordinates")
                        params = AllChem.ETKDGv3()
                        params.useRandomCoords = True
                        result = AllChem.EmbedMolecule(mol, params)
                        if result == -1:
                            logger.error("Failed to generate 3D coordinates for molecule")
            
            # Optimize geometry
            if optimize and mol.GetNumConformers() > 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception as e:
                    logger.warning(f"MMFF optimization failed: {e}")
            
            return mol
        except Exception as e:
            logger.error(f"Error preparing ligand: {e}")
            raise
    
    def _needs_3d_coordinates(self, mol) -> bool:
        """
        Check if a molecule needs 3D coordinate generation.
        
        A molecule needs 3D coordinates if:
        - It has no conformers
        - It has a conformer but all Z coordinates are 0 (2D structure)
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            True if 3D coordinates need to be generated
        """
        if mol.GetNumConformers() == 0:
            return True
        
        # Check if the existing conformer is 2D (all Z coordinates are 0)
        conf = mol.GetConformer(0)
        z_coords = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
        
        # If all Z coordinates are effectively 0, it's a 2D structure
        if all(abs(z) < 0.001 for z in z_coords):
            logger.info("Detected 2D structure (all Z coordinates are ~0)")
            return True
        
        return False
    
    def calculate_properties(self, mol) -> Dict[str, Any]:
        """
        Calculate physicochemical properties for a molecule using RDKit.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with properties (MW, LogP, HBD, HBA, TPSA, etc.)
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit not available for property calculation")
            
        try:
            # Calculate properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            qed = Descriptors.qed(mol)
            exact_mass = Descriptors.ExactMolWt(mol)
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            num_rings = Descriptors.RingCount(mol)
            num_aromatic_rings = Descriptors.NumAromaticRings(mol)
            molecular_formula = rdMolDescriptors.CalcMolFormula(mol)
            
            # Count stereo centers
            stereo_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            
            # Check Lipinski violations
            lipinski_violations = 0
            if mw > 500: lipinski_violations += 1
            if logp > 5: lipinski_violations += 1
            if hbd > 5: lipinski_violations += 1
            if hba > 10: lipinski_violations += 1
            
            return {
                "molecular_weight": mw,
                "logp": logp,
                "hydrogen_bond_donors": hbd,
                "hydrogen_bond_acceptors": hba,
                "tpsa": tpsa,
                "qed": qed,
                "stereo_centers": stereo_centers,
                "lipinski_violations": lipinski_violations,
                "exact_mass": exact_mass,
                "num_rotatable_bonds": num_rotatable_bonds,
                "num_rings": num_rings,
                "num_aromatic_rings": num_aromatic_rings,
                "molecular_formula": molecular_formula,
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "num_heavy_atoms": mol.GetNumHeavyAtoms()
            }
        except Exception as e:
            logger.error(f"Error calculating properties: {e}")
            raise
    
    def process_sdf_data(self, sdf_data: str) -> Dict[str, Any]:
        """
        Process SDF data and extract basic information.
        
        Args:
            sdf_data: SDF format data as string
            
        Returns:
            Dictionary with SDF processing results
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit not available for SDF processing")
        
        try:
            # Create supplier from SDF data
            supplier = Chem.SDMolSupplier()
            supplier.SetData(sdf_data)
            
            # Convert to list to check molecules
            mols = [m for m in supplier if m is not None]
            
            if not mols:
                raise ValueError("No valid molecules found in SDF data")
            
            mol = mols[0]  # Get first molecule
            conf_count = mol.GetNumConformers()
            
            return {
                "molecule_count": len(mols),
                "conformer_count": conf_count,
                "atom_count": mol.GetNumAtoms(),
                "has_3d": conf_count > 0,
                "sdf_data": sdf_data
            }
            
        except Exception as e:
            logger.error(f"Error processing SDF data: {e}")
            raise


# Singleton instance
_ligand_preparer_instance = None


def get_ligand_preparer() -> LigandPreparer:
    """Get or create LigandPreparer singleton instance."""
    global _ligand_preparer_instance
    if _ligand_preparer_instance is None:
        _ligand_preparer_instance = LigandPreparer()
    return _ligand_preparer_instance
