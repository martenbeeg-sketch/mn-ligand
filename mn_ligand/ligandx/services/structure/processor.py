# services/structure_processor.py
"""
Structure Processing Service

This module provides comprehensive protein structure processing capabilities including:
- Component identification (protein, ligands, water, ions)
- Structure cleaning and preparation using PDBFixer
- Ligand extraction and 2D image generation
- SDF file processing with conformer handling

Best Practices:
- Always check availability flags before using optional dependencies
- Use context managers for file operations
- Implement proper error handling for structure parsing
- Cache expensive operations when possible

Dependencies:
- BioPython: Core structure parsing and manipulation
- RDKit (optional): Ligand processing and 2D image generation
- PDBFixer (optional): Protein structure cleaning and preparation

Author: Molecular Structure Processing Team
Version: 1.0.0
Last Updated: August 2025
"""

import base64
from io import BytesIO
from mn_ligand.ligandx.lib.chemistry import (
    get_pdb_parser,
    get_component_analyzer,
    get_protein_preparer,
)
from mn_ligand.ligandx.lib.chemistry.parsers.mmcif import get_mmcif_parser

# Import RDKit for ligand processing and 2D image generation
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Ligand 2D image generation will be disabled.")
    RDKIT_AVAILABLE = False

# Valid element symbols recognized by RDKit (uppercase for fast lookup)
VALID_ELEMENTS_UPPER = {
    'H', 'HE', 'LI', 'BE', 'B', 'C', 'N', 'O', 'F', 'NE',
    'NA', 'MG', 'AL', 'SI', 'P', 'S', 'CL', 'AR',
    'K', 'CA', 'SC', 'TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
    'GA', 'GE', 'AS', 'SE', 'BR', 'KR',
    'RB', 'SR', 'Y', 'ZR', 'NB', 'MO', 'TC', 'RU', 'RH', 'PD', 'AG', 'CD',
    'IN', 'SN', 'SB', 'TE', 'I', 'XE',
    'CS', 'BA', 'LA', 'CE', 'PR', 'ND', 'PM', 'SM', 'EU', 'GD', 'TB', 'DY',
    'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA', 'W', 'RE', 'OS', 'IR', 'PT',
    'AU', 'HG', 'TL', 'PB', 'BI', 'PO', 'AT', 'RN',
    'FR', 'RA', 'AC', 'TH', 'PA', 'U', 'NP', 'PU', 'AM', 'CM', 'BK', 'CF',
    'ES', 'FM', 'MD', 'NO', 'LR'
}

# Capitalized version for RDKit compatibility
VALID_ELEMENTS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr'
}

# Two-letter elements that are commonly confused with single-letter ones
# These should take priority when inferring from atom name
TWO_LETTER_PRIORITY = {'BR', 'CL', 'FE', 'ZN', 'MG', 'CA', 'NA', 'MN', 'CO', 'CU', 'NI', 'SE', 'SI'}

# Cache for CCD SMILES lookups (None = known-missing, str = SMILES)
_CCD_SMILES_CACHE: dict = {}


def _get_ccd_smiles(res_name: str):
    """Fetch canonical SMILES from wwPDB CCD REST API for bond order assignment."""
    key = res_name.upper()
    if key in _CCD_SMILES_CACHE:
        return _CCD_SMILES_CACHE[key]
    try:
        import requests
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{key}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # Try rcsb_chem_comp_descriptor first (OpenEye SMILES preferred)
            for desc in data.get("rcsb_chem_comp_descriptor", {}).get("descriptors", []):
                if desc.get("type") == "SMILES" and desc.get("program") == "OpenEye OEToolkits":
                    _CCD_SMILES_CACHE[key] = desc["descriptor"]
                    return _CCD_SMILES_CACHE[key]
            # Fallback: pdbx_chem_comp_descriptor (non-stereo SMILES)
            for desc in data.get("pdbx_chem_comp_descriptor", []):
                dtype = desc.get("type", "")
                if "SMILES" in dtype and "stereo" not in dtype.lower():
                    smiles = desc.get("descriptor")
                    if smiles:
                        _CCD_SMILES_CACHE[key] = smiles
                        return smiles
    except Exception:
        pass
    _CCD_SMILES_CACHE[key] = None
    return None


def infer_element_from_atom_name(atom_name: str, current_element: str = '') -> str:
    """
    Infer element symbol from PDB atom name with priority for two-letter elements.
    
    This handles cases where element column has single-letter element (like "B")
    but atom name indicates two-letter element (like "BR" for Bromine).
    
    Args:
        atom_name: Atom name from PDB (columns 13-16)
        current_element: Current element from PDB element column (columns 77-78)
        
    Returns:
        Properly capitalized element symbol (e.g., 'Br', 'Cl', 'Fe')
    """
    name = (atom_name or '').strip()
    name_alpha = ''.join(ch for ch in name if ch.isalpha()).upper()
    candidate = (current_element or '').strip().upper()
    
    # First, check if atom name strongly suggests a two-letter element
    # This handles Br->B, Cl->C, Fe->F, etc. mismatches
    if len(name_alpha) >= 2:
        two_char = name_alpha[:2]
        if two_char in TWO_LETTER_PRIORITY:
            # Atom name suggests two-letter element
            # Only trust single-letter element if it's different from the first char
            if candidate and candidate != two_char[0] and candidate in VALID_ELEMENTS_UPPER:
                # Element field has a different, valid element - trust it
                return candidate.capitalize() if len(candidate) > 1 else candidate
            else:
                # Use the two-letter element from atom name
                return two_char.capitalize()  # e.g., 'BR' -> 'Br'
    
    # Use existing element field if valid
    if candidate and candidate in VALID_ELEMENTS_UPPER:
        return candidate.capitalize() if len(candidate) > 1 else candidate
    
    # Deduce from atom name
    for length in (2, 1):
        if len(name_alpha) >= length:
            guess = name_alpha[:length]
            if guess in VALID_ELEMENTS_UPPER:
                return guess.capitalize() if length > 1 else guess
    
    return candidate.capitalize() if candidate else ''


def fix_malformed_pdb_serials(pdb_data: str) -> str:
    """
    Fix malformed atom serial numbers in PDB files.
    
    Some MD simulations (e.g., GROMACS) output PDB files with malformed atom serial numbers
    like 'A000', 'B001', etc. instead of proper integers. This function fixes them by
    extracting just the numeric part or renumbering sequentially.
    
    Args:
        pdb_data: Raw PDB format string
        
    Returns:
        PDB string with corrected atom serial numbers
    """
    fixed_lines = []
    serial_counter = 1
    
    for line in pdb_data.split('\n'):
        if line.startswith(('ATOM', 'HETATM')):
            # Extract current serial from columns 7-11 (0-indexed: 6-11)
            try:
                serial_str = line[6:11].strip() if len(line) >= 11 else ''
                
                # Try to parse as integer first
                try:
                    serial = int(serial_str)
                except ValueError:
                    # If it fails, extract only digits or use counter
                    digits_only = ''.join(c for c in serial_str if c.isdigit())
                    if digits_only:
                        serial = int(digits_only)
                    else:
                        serial = serial_counter
                
                # Format serial number properly (right-justified in 5 characters)
                formatted_serial = str(serial).rjust(5)
                
                # Replace serial in line (columns 7-11, 0-indexed: 6-11)
                padded_line = line.ljust(80)  # Ensure line is at least 80 chars
                fixed_line = padded_line[:6] + formatted_serial + padded_line[11:]
                fixed_lines.append(fixed_line.rstrip())
                
                serial_counter = serial + 1
            except Exception as e:
                # If something goes wrong, keep original line
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def sanitize_pdb_for_rdkit(pdb_data: str) -> str:
    """
    Sanitize PDB data by fixing malformed atom serial numbers.

    RDKit correctly reads element symbols from columns 77-78 of PDB files per the PDB format
    specification. We trust the element column and let RDKit handle element parsing.

    This function only fixes malformed atom serial numbers (e.g., 'A000' -> '1') that some
    MD simulations (e.g., GROMACS) may produce.

    Previous versions of this function attempted to infer elements from atom names, which
    caused bugs like "O1S" (oxygen with positional suffix) being misidentified as "Os" (osmium).
    RDKit handles these cases correctly by reading the element column directly.

    Args:
        pdb_data: Raw PDB format string

    Returns:
        Sanitized PDB string with corrected atom serial numbers
    """
    # Only fix malformed atom serial numbers - trust element column for everything else
    return fix_malformed_pdb_serials(pdb_data)

class StructureProcessor:
    """
    Service for processing protein structures and identifying components.

    Attributes:
        pdb_parser: PDB parsing utilities
        mmcif_parser: mmCIF parsing utilities
        component_analyzer: Structure component analysis utilities
        protein_preparer: Protein preparation utilities

    Methods:
        process_structure(pdb_data): Process a protein structure to identify and separate components
        process_structure_with_ligands(pdb_data, clean_protein, include_2d_images): Process a protein structure with ligand extraction, protein cleaning, and ligand reinsertion
        get_ligand_info(structure, ligand_residues): Get detailed information about ligands in the structure
        extract_ligands(structure, ligand_residues): Extract ligands from a structure as separate entities
    """

    def __init__(self):
        """Initialize the StructureProcessor with modular chemistry utilities."""
        self.pdb_parser = get_pdb_parser()
        self.component_analyzer = get_component_analyzer()
        self.protein_preparer = get_protein_preparer()
        # Initialize mmCIF parser (may fail if BioPython not fully installed)
        try:
            self.mmcif_parser = get_mmcif_parser()
        except ImportError:
            self.mmcif_parser = None
            print("Warning: mmCIF parser not available")
    
    def detect_format(self, structure_data: str) -> str:
        """
        Detect whether structure data is in PDB or mmCIF format.
        
        Args:
            structure_data: Structure data as string
            
        Returns:
            'cif' or 'pdb'
        """
        data = structure_data.strip()
        # mmCIF files typically start with 'data_' or contain 'loop_' and '_atom_site'
        if data.startswith('data_') or ('loop_' in data and '_atom_site' in data):
            return 'cif'
        # PDB files start with HEADER, ATOM, HETATM, MODEL, etc.
        if data.startswith(('HEADER', 'ATOM', 'HETATM', 'MODEL', 'REMARK', 'TITLE', 'CRYST1')):
            return 'pdb'
        # Default to PDB if unclear
        return 'pdb'
    
    def parse_structure(self, structure_data: str, structure_id: str = "structure"):
        """
        Parse structure data, auto-detecting format (PDB or mmCIF).
        
        Args:
            structure_data: Structure data as string
            structure_id: Identifier for the structure
            
        Returns:
            BioPython structure object
        """
        detected_format = self.detect_format(structure_data)
        
        if detected_format == 'cif':
            if self.mmcif_parser is None:
                raise ImportError("mmCIF parser not available. Cannot parse CIF format.")
            print(f"[COMPLETE] Detected mmCIF format, using mmCIF parser")
            return self.mmcif_parser.parse_string(structure_data, structure_id)
        else:
            # For PDB format, sanitize first to fix element columns
            sanitized_data = sanitize_pdb_for_rdkit(structure_data)
            return self.pdb_parser.parse_string(sanitized_data, structure_id)
    
    def process_structure(self, structure_data):
        """
        Process a protein structure to identify and separate components.
        Supports both PDB and mmCIF formats (auto-detected).

        
        Args:
            structure_data (str): PDB or mmCIF format data as a string
            
        Returns:
            dict: A dictionary containing the processed structure components
        """
        # Parse the structure (auto-detects format)
        structure = self.parse_structure(structure_data)
        
        # Identify and separate components
        components = self.component_analyzer.identify_components(structure)
        
        # Extract individual components as PDB strings
        processed_data = {
            "full_structure": structure_data,
            "components": {}
        }
        
        for component_type, residues in components.items():
            if residues:
                pdb_string = self.pdb_parser.extract_residues_as_string(structure, residues)
                processed_data["components"][component_type] = {
                    "pdb_data": pdb_string,
                    "count": len(residues)
                }
        
        return processed_data
    
    def process_structure_with_ligands(self, structure_data, clean_protein=True, include_2d_images=True, 
                                       target_pdb_id=None, target_structure_id=None):
        """
        Process a protein structure with ligand extraction, protein cleaning, and ligand reinsertion.
        Supports both PDB and mmCIF formats (auto-detected).
        
        Args:
            structure_data (str): PDB or mmCIF format data as a string
            clean_protein (bool): Whether to clean the protein using PDBFixer
            include_2d_images (bool): Whether to include 2D images of ligands
            target_pdb_id (str, optional): Original PDB ID for ligand target information
            target_structure_id (str, optional): Structure identifier for ligand target information
            
        Returns:
            dict: A dictionary containing the processed structure with ligands
        """
        # Parse the structure (auto-detects format)
        structure = self.parse_structure(structure_data)
        
        # Identify components
        components = self.component_analyzer.identify_components(structure)
        
        # Extract ligands with target information
        ligands = self.extract_ligands(structure, components["ligands"], 
                                     target_pdb_id=target_pdb_id, 
                                     target_structure_id=target_structure_id)
        
        # Clean the protein if requested
        cleaned_protein_data = None
        if clean_protein and components["protein"]:
            try:
                # Extract protein component only for cleaning
                protein_pdb = self.pdb_parser.extract_residues_as_string(structure, components["protein"])
                
                # Clean protein structure without removing heterogens (since we extracted protein only)
                # We don't need to remove heterogens because we already extracted only protein residues
                # Use staged cleaning and get the final result
                cleaning_result = self.protein_preparer.clean_structure_staged(
                    protein_pdb,
                    remove_heterogens=False,  # Don't remove heterogens since we only have protein
                    remove_water=True,        # Remove any water that might be in protein selection
                    add_missing_residues=True,
                    add_missing_atoms=True,
                    add_missing_hydrogens=True,
                    ph=7.4,
                    add_solvation=False,
                    keep_ligands=False
                )
                # Get the final cleaned stage (highest step number)
                stages = cleaning_result['stages']
                stage_info = cleaning_result['stage_info']
                # Find the stage with the highest step number
                final_stage = max(stage_info.items(), key=lambda x: x[1].get('step', 0))
                cleaned_protein_data = stages[final_stage[0]]
                
                # Reinsert ligands into cleaned protein
                if ligands:
                    cleaned_protein_data = self.reinsert_ligands(cleaned_protein_data, ligands)
                    
            except Exception as e:
                print(f"Warning: Protein cleaning failed: {e}")
                # Fall back to original structure
                cleaned_protein_data = structure_data
        else:
            cleaned_protein_data = structure_data
        
        # Generate 2D images for ligands if requested
        if include_2d_images and ligands and RDKIT_AVAILABLE:
            try:
                ligands = self.generate_ligand_2d_images(ligands)
            except Exception as e:
                print(f"Warning: 2D image generation failed: {e}")
        
        return {
            "original_structure": structure_data,
            "processed_structure": cleaned_protein_data,
            "components": {
                component_type: len(residues) 
                for component_type, residues in components.items()
            },
            "ligands": ligands,
            "protein_cleaned": clean_protein and cleaned_protein_data != structure_data
        }
    
    
    def get_ligand_info(self, structure, ligand_residues):
        """
        Get detailed information about ligands in the structure.
        
        Args:
            structure: BioPython structure object
            ligand_residues: List of ligand residues
            
        Returns:
            list: List of dictionaries with ligand information
        """
        ligand_info = []
        
        for residue in ligand_residues:
            # Get basic information about the ligand
            chain_id = residue.get_parent().get_id()
            res_name = residue.get_resname()
            res_id = residue.get_id()[1]
            
            # Count atoms in the ligand
            atom_count = len(list(residue.get_atoms()))
            
            # Get the center of mass
            com = self.component_analyzer.calculate_center_of_mass(residue)
            
            ligand_info.append({
                "name": res_name,
                "chain": chain_id,
                "residue_number": res_id,
                "atom_count": atom_count,
                "center_of_mass": com
            })
        
        return ligand_info
    
    def _calculate_center_of_mass(self, residue):
        """Calculate the center of mass of a residue."""
        return self.component_analyzer.calculate_center_of_mass(residue)
    
    def extract_ligands(self, structure, ligand_residues, target_pdb_id=None, target_structure_id=None):
        """
        Extract ligands from a structure as separate entities with full coordinate preservation.
        
        Args:
            structure: BioPython structure object
            ligand_residues: List of ligand residues
            target_pdb_id (str, optional): Original PDB ID for ligand target information
            target_structure_id (str, optional): Structure identifier for ligand target information
            
        Returns:
            dict: Dictionary of extracted ligands with metadata including original coordinates
        """
        ligands = {}
        
        for i, residue in enumerate(ligand_residues):
            # Extract residue information
            chain_id = residue.get_parent().get_id()
            res_name = residue.get_resname()
            res_id = residue.get_id()[1]
            
            # Create a unique ID for the ligand
            ligand_id = f"{res_name}_{chain_id}_{res_id}"
            
            # Get center of mass
            center_of_mass = self.component_analyzer.calculate_center_of_mass(residue)
            
            # Extract full atom coordinates from original structure
            original_coordinates = []
            for atom in residue.get_atoms():
                coord = atom.get_coord()
                original_coordinates.append({
                    "atom_name": atom.get_name(),
                    "element": atom.element,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "z": float(coord[2])
                })
            
            # Convert residue to PDB string
            ligand_pdb = self.pdb_parser.extract_residues_as_string(structure, [residue])
            
            # Convert PDB to SDF format for MD optimization compatibility
            # IMPORTANT: Preserve original 3D coordinates if they exist
            ligand_sdf = None
            has_valid_3d_coords = False
            # Fetch CCD SMILES for bond order assignment and structure representation
            ccd_smiles = _get_ccd_smiles(res_name)
            try:
                if RDKIT_AVAILABLE:
                    from rdkit import Chem
                    from rdkit.Chem import AllChem
                    # Sanitize PDB to remove invalid elements (e.g., 'R' placeholders from Boltz2)
                    sanitized_pdb = sanitize_pdb_for_rdkit(ligand_pdb)
                    # Convert PDB to SDF using RDKit
                    # ALWAYS removeHs=True - PDB hydrogens often have invalid (0,0,0) coordinates
                    mol = Chem.MolFromPDBBlock(sanitized_pdb, removeHs=True)
                    if mol is not None:
                        # Assign correct bond orders using CCD template (fixes aromaticity)
                        # PDB files have no bond order info; without this, benzene = cyclohexane
                        if ccd_smiles:
                            try:
                                template = Chem.MolFromSmiles(ccd_smiles)
                                if template is not None:
                                    Chem.RemoveHs(template)
                                    mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
                                    print(f"[COMPLETE] Assigned bond orders from CCD for {ligand_id}")
                            except Exception as e:
                                # Retry with charge-neutral template.
                                # PDB mols have no formal charges; CCD SMILES with [N+]/[O-] (e.g. HEPES/EPE)
                                # cause the subgraph match to fail. Stripping charges allows topology-only
                                # matching while still transferring correct bond orders (e.g. S=O).
                                try:
                                    template2 = Chem.MolFromSmiles(ccd_smiles)
                                    if template2 is not None:
                                        Chem.RemoveHs(template2)
                                        rw = Chem.RWMol(template2)
                                        for atom in rw.GetAtoms():
                                            atom.SetFormalCharge(0)
                                        mol = AllChem.AssignBondOrdersFromTemplate(rw.GetMol(), mol)
                                        print(f"[COMPLETE] Assigned bond orders from neutral CCD template for {ligand_id}")
                                except Exception as e2:
                                    print(f"⚠ CCD bond order assignment failed for {ligand_id}: {e2} — using raw parse")
                        # Check if molecule has valid 3D coordinates
                        if mol.GetNumConformers() > 0:
                            conf = mol.GetConformer(0)
                            if conf.Is3D():
                                has_valid_3d_coords = True
                                # Add hydrogens while preserving 3D coordinates
                                mol = Chem.AddHs(mol, addCoords=True)
                                print(f"[COMPLETE] Preserved original 3D coordinates for ligand {ligand_id}")
                            else:
                                # Has conformer but not 3D, add hydrogens and preserve what we can
                                mol = Chem.AddHs(mol, addCoords=True)
                        else:
                            # No conformer, add hydrogens with coordinates
                            mol = Chem.AddHs(mol, addCoords=True)
                        
                        # Check if hydrogens have valid coordinates (not at origin)
                        # This is critical for QC calculations - hydrogens at (0,0,0) will fail
                        if mol.GetNumConformers() > 0:
                            conf = mol.GetConformer(0)
                            h_atoms_at_origin = 0
                            for atom in mol.GetAtoms():
                                if atom.GetAtomicNum() == 1:  # Hydrogen
                                    pos = conf.GetAtomPosition(atom.GetIdx())
                                    if abs(pos.x) < 0.001 and abs(pos.y) < 0.001 and abs(pos.z) < 0.001:
                                        h_atoms_at_origin += 1
                            
                            if h_atoms_at_origin > 0:
                                # Hydrogens at origin - need to re-embed
                                has_valid_3d_coords = False
                                print(f"⚠ {h_atoms_at_origin} hydrogens at origin for {ligand_id}, will re-embed")
                        
                        # Regenerate conformer if we don't have valid 3D coordinates
                        if not has_valid_3d_coords or mol.GetNumConformers() == 0:
                            try:
                                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                                AllChem.MMFFOptimizeMolecule(mol)
                                print(f"[COMPLETE] Generated new 3D conformer for ligand {ligand_id}")
                            except Exception as e:
                                AllChem.Compute2DCoords(mol)
                                print(f"⚠ Generated 2D coordinates for ligand {ligand_id} (3D embedding failed: {e})")
                        
                        # Convert to SDF format - use confId=0 to write only first conformer
                        ligand_sdf = Chem.MolToMolBlock(mol, confId=0)
                        print(f"[COMPLETE] Generated SDF data for ligand {ligand_id}")
                    else:
                        print(f"⚠ Could not convert {ligand_id} PDB to SDF format")
                else:
                    print("⚠ RDKit not available - SDF conversion skipped")
            except Exception as e:
                print(f"⚠ SDF conversion failed for {ligand_id}: {e}")
            
            # Count atoms in the ligand
            atom_count = len(list(residue.get_atoms()))
            
            # Convert center_of_mass to native Python floats (avoid numpy serialization issues)
            center_of_mass_native = tuple(float(x) for x in center_of_mass) if center_of_mass else (0.0, 0.0, 0.0)
            
            # Store ligand information with both PDB and SDF data, plus target information
            # Ensure all numeric values are native Python types for JSON serialization
            ligand_data = {
                "name": str(res_name),
                "het_id": str(res_name),  # 3-letter HET ID for QC naming
                "chain": str(chain_id),
                "residue_number": int(res_id),
                "pdb_data": ligand_pdb,
                "atom_count": int(atom_count),
                "center_of_mass": center_of_mass_native,
                "original_coordinates": original_coordinates,  # Full atom coordinates
                "has_valid_3d_coords": bool(has_valid_3d_coords),
                "smiles": ccd_smiles,  # CCD SMILES for proper structure representation
            }
            
            # Add target information if provided
            if target_pdb_id:
                ligand_data["target_pdb_id"] = str(target_pdb_id)
            if target_structure_id:
                ligand_data["target_structure_id"] = str(target_structure_id)
            
            # Add binding site information
            ligand_data["binding_site_info"] = {
                "chain_id": str(chain_id),
                "residue_number": int(res_id),
                "center_of_mass": center_of_mass_native
            }
            
            # Add SDF data if conversion was successful
            if ligand_sdf:
                ligand_data["sdf_data"] = ligand_sdf
                print(f"[COMPLETE] Ligand {ligand_id} has both PDB and SDF data")
            else:
                print(f"⚠ Ligand {ligand_id} has only PDB data")
            
            ligands[ligand_id] = ligand_data
        
        return ligands
    
    def reinsert_ligands(self, cleaned_protein_data, ligands):
        """
        Reinsert extracted ligands into the cleaned protein structure.
        Verifies that ligand coordinates are preserved and match original binding site.
        
        Args:
            cleaned_protein_data: Cleaned PDB data
            ligands: Dictionary of ligands to reinsert (with original_coordinates if available)
            
        Returns:
            str: PDB data with reinserted ligands
        """
        # Create a combined structure by appending ligand data
        combined_pdb_lines = []
        
        # Add protein lines (excluding END)
        protein_lines = cleaned_protein_data.strip().split('\n')
        for line in protein_lines:
            if not line.startswith('END'):
                combined_pdb_lines.append(line)
        
        # Add ligand lines, preserving original coordinates
        for ligand_id, ligand_info in ligands.items():
            ligand_pdb = ligand_info['pdb_data']
            ligand_lines = ligand_pdb.strip().split('\n')
            
            # Verify that ligand PDB contains coordinates
            has_coords = False
            for line in ligand_lines:
                if line.startswith(('HETATM', 'ATOM')):
                    # Check if line has valid coordinates (not all zeros)
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        if abs(x) > 0.001 or abs(y) > 0.001 or abs(z) > 0.001:
                            has_coords = True
                            break
                    except (ValueError, IndexError):
                        pass
            
            if not has_coords and 'original_coordinates' in ligand_info:
                print(f"⚠ Warning: Ligand {ligand_id} PDB lacks coordinates, but original_coordinates available")
            
            # Add ligand lines with preserved chain IDs and residue numbers
            for line in ligand_lines:
                if line.startswith(('HETATM', 'ATOM')) and not line.startswith('END'):
                    # Ensure chain ID and residue number match original if specified
                    if 'binding_site_info' in ligand_info:
                        binding_info = ligand_info['binding_site_info']
                        # Verify chain ID matches (if line format allows)
                        try:
                            line_chain = line[21:22] if len(line) > 21 else ' '
                            original_chain = binding_info.get('chain_id', '')
                            if original_chain and line_chain != original_chain:
                                # Update chain ID in line if needed
                                if len(line) >= 22:
                                    line = line[:21] + original_chain + line[22:]
                        except (IndexError, ValueError):
                            pass
                    combined_pdb_lines.append(line)
            
            print(f"[COMPLETE] Reinserted ligand {ligand_id} with preserved coordinates")

        # Add CONECT records for double/triple bonds so Mol* displays correct bond orders.
        # PDB CONECT encodes multiplicity by repeating the pair: twice = double, three times = triple.
        # Without these records Mol* falls back to distance-based inference and misses S=O bonds.
        if RDKIT_AVAILABLE:
            from rdkit import Chem as _Chem
            for ligand_id, ligand_info in ligands.items():
                sdf_data = ligand_info.get('sdf_data')
                if not sdf_data:
                    continue
                try:
                    sdf_mol = _Chem.MolFromMolBlock(sdf_data, removeHs=True)
                    if sdf_mol is None:
                        continue
                    # Collect heavy-atom serial numbers from HETATM/ATOM lines in order
                    heavy_serials = []
                    for line in ligand_info['pdb_data'].strip().split('\n'):
                        if line.startswith(('HETATM', 'ATOM')):
                            elem = line[76:78].strip() if len(line) >= 78 else ''
                            if elem and elem == 'H':
                                continue
                            try:
                                heavy_serials.append(int(line[6:11].strip()))
                            except (ValueError, IndexError):
                                pass
                    if sdf_mol.GetNumAtoms() != len(heavy_serials):
                        print(f"⚠ Atom count mismatch for {ligand_id} ({sdf_mol.GetNumAtoms()} vs {len(heavy_serials)}), skipping CONECT records")
                        continue
                    for bond in sdf_mol.GetBonds():
                        order = int(round(bond.GetBondTypeAsDouble()))
                        if order <= 1:
                            continue
                        s1 = heavy_serials[bond.GetBeginAtomIdx()]
                        s2 = heavy_serials[bond.GetEndAtomIdx()]
                        for _ in range(order):
                            combined_pdb_lines.append(f"CONECT{s1:5d}{s2:5d}")
                    print(f"[COMPLETE] Generated CONECT records for {ligand_id}")
                except Exception as e:
                    print(f"⚠ Failed to generate CONECT records for {ligand_id}: {e}")

        # Add END record
        combined_pdb_lines.append('END')
        
        return '\n'.join(combined_pdb_lines)
    
    def generate_ligand_2d_images(self, ligands):
        """
        Generate 2D images for ligands using RDKit.
        
        Args:
            ligands: Dictionary of ligands
            
        Returns:
            dict: Dictionary of ligands with added 2D image data
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is not available. Please install it to generate 2D ligand images.")
        
        for ligand_id, ligand_info in ligands.items():
            try:
                # Sanitize and convert PDB to RDKit molecule
                sanitized_pdb = sanitize_pdb_for_rdkit(ligand_info['pdb_data'])
                mol = Chem.MolFromPDBBlock(sanitized_pdb)
                
                if mol is not None:
                    # Clean up the molecule
                    mol = Chem.RemoveHs(mol)  # Remove hydrogens for cleaner 2D depiction
                    
                    # Generate 2D coordinates
                    AllChem.Compute2DCoords(mol)
                    
                    # Generate the image
                    img = Draw.MolToImage(mol, size=(200, 200))
                    
                    # Convert image to base64 for embedding in HTML
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Add the image to the ligand info
                    ligand_info['image_data'] = f"data:image/png;base64,{img_str}"
                else:
                    ligand_info['image_data'] = None
                    print(f"Warning: Could not convert ligand {ligand_id} to RDKit molecule")
            except Exception as e:
                print(f"Error generating 2D image for ligand {ligand_id}: {e}")
                ligand_info['image_data'] = None
        
        return ligands
        
    def process_sdf_with_conformers(self, sdf_data, output_path=None):
        """
        Process an SDF file with multiple conformers and optionally write to a file.
        
        Args:
            sdf_data (str): SDF format data as a string
            output_path (str, optional): Path to write the processed SDF file
            
        Returns:
            dict: A dictionary containing the processed SDF data and metadata
        """
        # Use shared utilities for SDF processing
        result = self.utils.process_sdf_data(sdf_data)
        
        # Handle file output if requested
        if output_path and result.get("conformer_count", 0) > 0:
            try:
                if not RDKIT_AVAILABLE:
                    raise ImportError("RDKit is not available for file output.")
                    
                supplier = Chem.SDMolSupplier()
                supplier.SetData(sdf_data)
                mols = [m for m in supplier if m is not None]
                
                if mols:
                    mol = mols[0]
                    conf_count = mol.GetNumConformers()
                    
                    writer = Chem.SDWriter(output_path)
                    for cid in range(conf_count):
                        mol.SetProp('ID', f'conformer_{cid}')
                        writer.write(mol, confId=cid)
                    writer.close()
            except Exception as e:
                print(f"Error writing SDF file: {e}")
        
        return result