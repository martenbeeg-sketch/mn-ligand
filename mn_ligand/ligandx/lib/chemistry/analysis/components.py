"""
Structure component identification and analysis.

Provides functionality for identifying different components in molecular structures
(proteins, nucleic acids, ligands, water, ions, etc.)
"""

import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


# Standard residue classifications
RESIDUE_CLASSIFICATIONS = {
    'protein_residues': {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        # Non-standard amino acids
        'MSE', 'HYP', 'PCA', 'SEP', 'TPO', 'CSO', 'PTR', 'KCX'
    },
    'nucleic_residues': {
        'DA', 'DT', 'DG', 'DC', 'A', 'U', 'G', 'C', 'DU', 'I', 'N'
    },
    'water_residues': {'HOH', 'WAT', 'H2O', 'TIP', 'TIP3', 'TIP4'},
    'common_ions': {
        'NA', 'MG', 'K', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN', 
        'CD', 'HG', 'CL', 'BR', 'I', 'F', 'LI', 'BE', 'B', 'AL', 
        'SI', 'P', 'S', 'TL', 'PB'
    }
}

# Standard amino acid one-letter codes
AMINO_ACID_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # Common variants
    'MSE': 'M', 'SEC': 'C', 'PYL': 'K'
}

# Simple atomic masses for center of mass calculations
ATOMIC_MASSES = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'P': 30.974, 'S': 32.065, 'F': 18.998, 'CL': 35.453,
    'BR': 79.904, 'I': 126.904
}


class ComponentAnalyzer:
    """Analyzer for identifying and categorizing structure components."""
    
    def __init__(self):
        self.protein_residues = RESIDUE_CLASSIFICATIONS['protein_residues']
        self.nucleic_residues = RESIDUE_CLASSIFICATIONS['nucleic_residues']
        self.water_residues = RESIDUE_CLASSIFICATIONS['water_residues']
        self.common_ions = RESIDUE_CLASSIFICATIONS['common_ions']
    
    def identify_components(self, structure) -> Dict[str, List]:
        """
        Identify different components in a structure.
        
        Args:
            structure: BioPython structure object
            
        Returns:
            Dictionary with component types as keys and lists of residues as values
        """
        components = {
            "protein": [],
            "nucleic": [],
            "ligands": [],
            "water": [],
            "ions": [],
            "other": []
        }
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    
                    if resname in self.protein_residues:
                        components["protein"].append(residue)
                    elif resname in self.nucleic_residues:
                        components["nucleic"].append(residue)
                    elif resname in self.water_residues:
                        components["water"].append(residue)
                    elif resname in self.common_ions:
                        components["ions"].append(residue)
                    else:
                        # Check if it's a ligand (HETATM with multiple atoms)
                        if residue.get_id()[0].startswith('H_') and len(list(residue.get_atoms())) > 1:
                            components["ligands"].append(residue)
                        else:
                            components["other"].append(residue)
        
        return components
    
    def calculate_center_of_mass(self, residue) -> Tuple[float, float, float]:
        """
        Calculate center of mass for a residue.
        
        Args:
            residue: BioPython residue object
            
        Returns:
            Tuple of (x, y, z) coordinates
        """
        atoms = list(residue.get_atoms())
        if not atoms:
            return (0.0, 0.0, 0.0)
        
        total_mass = 0.0
        weighted_coords = [0.0, 0.0, 0.0]
        
        for atom in atoms:
            element = atom.element.upper() if hasattr(atom, 'element') else atom.get_name()[0]
            mass = ATOMIC_MASSES.get(element, 12.011)  # Default to carbon mass
            coord = atom.get_coord()
            
            total_mass += mass
            for i in range(3):
                # Convert numpy types to native Python floats
                weighted_coords[i] += mass * float(coord[i])
        
        if total_mass > 0:
            # Convert to native Python floats to avoid numpy serialization issues
            return tuple(float(coord / total_mass) for coord in weighted_coords)
        else:
            return (0.0, 0.0, 0.0)
    
    def get_protein_sequence(self, structure) -> str:
        """
        Extract protein sequence from structure.
        
        Args:
            structure: BioPython structure object
            
        Returns:
            Protein sequence string (one letter codes)
        """
        sequence = ""
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_id = residue.get_id()
                    # Include standard amino acids (hetflag = ' ') and modified residues
                    if res_id[0] == ' ' or res_id[0] == 'H_MSE':
                        resname = residue.get_resname()
                        if resname in AMINO_ACID_MAP:
                            sequence += AMINO_ACID_MAP[resname]
                        else:
                            # Unknown amino acid - use X
                            sequence += 'X'
        
        return sequence
    
    def get_structure_info(self, structure, pdb_data: str = None) -> Dict[str, Any]:
        """
        Extract comprehensive information from a structure.
        
        Args:
            structure: BioPython structure object
            pdb_data: Optional PDB data string
            
        Returns:
            Dictionary with structure information
        """
        structure_id = structure.get_id()
        
        # Extract header information if available
        header = getattr(structure, "header", {})
        resolution = header.get("resolution", "Not available")
        exp_method = header.get("structure_method", "Not available")
        
        # Count chains, residues, and atoms
        chains = list(structure.get_chains())
        chain_count = len(chains)
        
        residue_count = 0
        atom_count = 0
        for chain in chains:
            residue_count += len(list(chain.get_residues()))
            atom_count += len(list(chain.get_atoms()))
        
        # Identify components
        components = self.identify_components(structure)
        
        info = {
            "structure_id": structure_id,
            "metadata": {
                "resolution": resolution,
                "experimental_method": exp_method,
                "chain_count": chain_count,
                "residue_count": residue_count,
                "atom_count": atom_count
            },
            "components": {
                component_type: len(residues) 
                for component_type, residues in components.items()
            }
        }
        
        if pdb_data:
            info["pdb_data"] = pdb_data
            
        return info
    
    def validate_protein_structure(self, structure, pdb_data: str = None) -> Dict[str, Any]:
        """
        Validate protein structure.
        
        Args:
            structure: BioPython structure object
            pdb_data: Optional PDB data string for re-parsing
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'residue_count': 0,
            'chain_count': 0,
            'has_protein': False
        }
        
        try:
            components = self.identify_components(structure)
            
            protein_residues = components.get('protein', [])
            result['residue_count'] = len(protein_residues)
            result['chain_count'] = len(list(structure.get_chains()))
            result['has_protein'] = len(protein_residues) > 0
            
            # Extract sequence
            result['sequence'] = self.get_protein_sequence(structure)
            
            if not result['has_protein']:
                result['errors'].append("No protein residues found")
            
            if result['residue_count'] < 10:
                result['warnings'].append("Very small protein (less than 10 residues)")
                
            # Check for missing atoms (basic check)
            for residue in protein_residues:
                if not residue.has_id('CA'):
                    result['warnings'].append(
                        f"Residue {residue.get_resname()} {residue.get_id()[1]} missing CA atom"
                    )
            
            if not result['errors']:
                result['valid'] = True
                
            return result
            
        except Exception as e:
            result['errors'].append(f"Failed to analyze structure: {str(e)}")
            return result


# Singleton instance
_component_analyzer_instance = None


def get_component_analyzer() -> ComponentAnalyzer:
    """Get or create ComponentAnalyzer singleton instance."""
    global _component_analyzer_instance
    if _component_analyzer_instance is None:
        _component_analyzer_instance = ComponentAnalyzer()
    return _component_analyzer_instance
