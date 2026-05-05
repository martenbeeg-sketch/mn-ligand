"""
Centralized structure validation for all services.

This module provides validation functions to ensure that structures
match the requirements of specific services (e.g., QC requires small
molecules, docking requires proteins).
"""
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from io import StringIO

logger = logging.getLogger(__name__)

# Conditional imports
try:
    from Bio.PDB import PDBParser
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False
    PDBParser = None

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None


class StructureValidationError(ValueError):
    """Raised when structure validation fails."""
    pass


# Service requirements mapping
SERVICE_REQUIREMENTS = {
    'qc': {
        'structure_type': 'small_molecule',
        'formats': ['sdf', 'xyz', 'pdb', 'mol'],
        'description': 'small molecule structure (SDF, XYZ, MOL, or PDB format)'
    },
    'admet': {
        'structure_type': 'small_molecule',
        'formats': ['sdf', 'smiles'],
        'description': 'small molecule structure (SDF format or SMILES string)'
    },
    'docking': {
        'structure_type': 'protein',
        'formats': ['pdb', 'cif', 'mmcif'],
        'description': 'protein structure (PDB, CIF, or mmCIF format)'
    },
    'md': {
        'structure_type': 'protein',
        'formats': ['pdb', 'cif', 'mmcif'],
        'description': 'protein structure (PDB, CIF, or mmCIF format)'
    },
    'boltz2': {
        'structure_type': 'protein',
        'formats': ['pdb', 'cif', 'mmcif'],
        'description': 'protein structure (PDB, CIF, or mmCIF format)'
    },
    'protein_cleaning': {
        'structure_type': 'protein',
        'formats': ['pdb', 'cif', 'mmcif'],
        'description': 'protein structure (PDB, CIF, or mmCIF format)'
    },
    'boltz': {  # Alias for boltz2
        'structure_type': 'protein',
        'formats': ['pdb', 'cif', 'mmcif'],
        'description': 'protein structure (PDB, CIF, or mmCIF format)'
    }
}

# Standard protein residues
PROTEIN_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    # Non-standard amino acids
    'MSE', 'HYP', 'PCA', 'SEP', 'TPO', 'CSO', 'PTR', 'KCX'
}


def get_service_requirements(service_name: str) -> Dict[str, Any]:
    """
    Get the structure requirements for a specific service.
    
    Args:
        service_name: Name of the service (e.g., 'qc', 'docking', 'md')
        
    Returns:
        Dictionary with requirements including structure_type and formats
        
    Raises:
        ValueError: If service name is not recognized
    """
    service_name_lower = service_name.lower()
    
    if service_name_lower not in SERVICE_REQUIREMENTS:
        available = ', '.join(SERVICE_REQUIREMENTS.keys())
        raise ValueError(
            f"Unknown service: {service_name}. "
            f"Available services: {available}"
        )
    
    return SERVICE_REQUIREMENTS[service_name_lower].copy()


def detect_structure_type(structure_data: str, format_hint: Optional[str] = None) -> str:
    """
    Detect the type of structure (protein, small_molecule, or complex).
    
    Args:
        structure_data: Structure data as string (PDB, SDF, etc.)
        format_hint: Optional format hint ('pdb', 'sdf', 'xyz', etc.)
        
    Returns:
        Structure type: 'protein', 'small_molecule', or 'complex'
    """
    if not structure_data or not structure_data.strip():
        raise StructureValidationError("Structure data is empty")
    
    # IMPORTANT: Check for PDB patterns first, regardless of format hint
    # This catches cases where PDB data is converted to XYZ format
    has_pdb_atom_records = 'ATOM  ' in structure_data or 'HETATM' in structure_data
    
    # Check if it looks like PDB format (most common for proteins)
    lines = structure_data.split('\n')
    atom_count = 0
    hetatm_count = 0
    protein_residue_count = 0
    has_protein_residues = False
    total_residues_seen = set()
    
    for line in lines:
        if line.startswith('ATOM  '):
            atom_count += 1
            if len(line) >= 20:
                resname = line[17:20].strip()
                if resname in PROTEIN_RESIDUES:
                    # Track unique residues
                    try:
                        chain = line[21] if len(line) > 21 else 'A'
                        resnum = line[22:26].strip() if len(line) > 26 else ''
                        residue_key = f"{chain}_{resnum}_{resname}"
                        if residue_key not in total_residues_seen:
                            total_residues_seen.add(residue_key)
                            protein_residue_count += 1
                            has_protein_residues = True
                    except (IndexError, ValueError):
                        # If we can't parse, but it's a protein residue, count it
                        if resname in PROTEIN_RESIDUES:
                            has_protein_residues = True
                            protein_residue_count += 1
        elif line.startswith('HETATM'):
            hetatm_count += 1
    
    # IMPORTANT: Distinguish between real proteins and extracted ligands
    # Extracted ligands may appear in PDB format but should be treated as small molecules
    # Key distinction: ligands are HETATM-only with no or very few protein ATOM records
    if has_protein_residues and protein_residue_count >= 1:
        # If we have many HETATM records but very few protein residues,
        # this is likely an extracted ligand in PDB format (HETATM records)
        # that happens to have a residue name matching a protein (e.g., 'LIG' misdetected)
        if hetatm_count > 0 and protein_residue_count <= 3 and atom_count <= 5:
            logger.info(f"Detected HETATM-dominant structure with few protein residues ({protein_residue_count}) - treating as small molecule ligand")
            return 'small_molecule'

        # If we have significant HETATM (ligands) AND many protein residues, it's a complex
        if hetatm_count > 10 and protein_residue_count > 5:
            return 'complex'
        # Otherwise it's a protein
        return 'protein'
    
    # If we have PDB ATOM records but no protein residues detected, 
    # it might still be a protein (maybe residue names weren't recognized)
    # Be conservative: if we have many ATOM records (>50), it's likely a protein
    if has_pdb_atom_records and atom_count > 50 and not has_protein_residues:
        logger.warning(f"PDB format with {atom_count} ATOM records but no recognized protein residues - treating as protein")
        return 'protein'
    
    # Check for XYZ format (common for QC calculations - should be small molecules)
    # BUT: If the data contains PDB-like patterns (ATOM/HETATM), it might be a protein
    # XYZ format: first line is atom count, second is comment, then atom lines
    # IMPORTANT: Check for PDB patterns FIRST, even if format hint is XYZ
    # This catches cases where PDB data is converted to XYZ format
    
    # Check if data contains PDB-like patterns (ATOM/HETATM records)
    has_pdb_patterns = 'ATOM' in structure_data or 'HETATM' in structure_data
    
    if format_hint and format_hint.lower() == 'xyz' and not has_pdb_patterns:
        # Only treat as XYZ if it doesn't have PDB patterns
        try:
            xyz_lines = [l.strip() for l in lines if l.strip()]
            if len(xyz_lines) >= 3:
                # First line should be atom count
                try:
                    atom_count_xyz = int(xyz_lines[0])
                    # Check if any lines look like protein residues (unlikely in XYZ)
                    # But if we see many atoms (>100), it might be a protein
                    if atom_count_xyz > 100:
                        # Could be a protein - be conservative
                        logger.warning(f"Large XYZ file ({atom_count_xyz} atoms) - might be a protein")
                        # Don't return yet - let PDB parsing check it
                    # Small XYZ files are likely small molecules
                    elif atom_count_xyz <= 100:
                        return 'small_molecule'
                except ValueError:
                    pass
        except Exception:
            pass
    elif format_hint and format_hint.lower() == 'xyz' and has_pdb_patterns:
        # XYZ format hint but contains PDB patterns - likely a protein converted to XYZ
        logger.warning("XYZ format hint but data contains PDB patterns - treating as PDB")
        # Continue to PDB parsing below
    
    # If format hint is SDF or MOL, it's likely a small molecule
    if format_hint and format_hint.lower() in ['sdf', 'mol']:
        # Verify by trying to parse as molecule
        if RDKIT_AVAILABLE:
            try:
                if format_hint.lower() == 'sdf':
                    mol = Chem.MolFromMolBlock(structure_data)
                else:
                    mol = Chem.MolFromMolBlock(structure_data)
                if mol is not None:
                    # Check atom count - if very large, might be a protein
                    atom_count_mol = mol.GetNumAtoms()
                    if atom_count_mol > 500:
                        logger.warning(f"Large molecule in SDF ({atom_count_mol} atoms) - might be a protein")
                        # Could be a protein - be conservative
                        return 'unknown'
                    return 'small_molecule'
            except Exception:
                pass
    
    # Try to parse as PDB to check for protein (more reliable)
    if BIO_AVAILABLE and PDBParser is not None and (atom_count > 0 or 'ATOM' in structure_data or 'HETATM' in structure_data):
        try:
            parser = PDBParser(QUIET=True)
            structure_file = StringIO(structure_data)
            structure = parser.get_structure('temp', structure_file)
            
            # Count protein residues
            protein_count = 0
            total_atoms = 0
            for model in structure:
                for chain in model:
                    for residue in chain:
                        res_id = residue.get_id()
                        resname = residue.get_resname()
                        total_atoms += len(list(residue.get_atoms()))
                        if res_id[0] == ' ':  # Standard residue
                            if resname in PROTEIN_RESIDUES:
                                protein_count += 1
            
            # STRICT: If we find ANY protein residues, it's a protein
            if protein_count >= 1:
                # Check for ligands (HETATM residues)
                hetatm_residues = 0
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            res_id = residue.get_id()
                            if res_id[0] != ' ':  # HETATM residue
                                resname = residue.get_resname()
                                if resname not in ['HOH', 'WAT', 'H2O']:  # Exclude water
                                    hetatm_residues += 1
                
                if hetatm_residues > 0 and protein_count > 5:
                    return 'complex'
                return 'protein'
            
            # If no protein residues but many atoms, might be a large small molecule
            # But if it's in PDB format with no protein residues, it's likely a small molecule
            if total_atoms > 0 and protein_count == 0:
                return 'small_molecule'
        except Exception as e:
            logger.debug(f"Could not parse as PDB for type detection: {e}")
    
    
    # Try to detect XYZ format before attempting SMILES parsing
    # XYZ format: first line is atom count (integer), second is comment, then atom lines
    if not has_pdb_patterns:
        try:
            xyz_lines = [l.strip() for l in lines if l.strip()]
            if len(xyz_lines) >= 3:
                # Check if first line is an integer (atom count)
                try:
                    atom_count_xyz = int(xyz_lines[0])
                    # If first line is a valid atom count, this is likely XYZ format
                    # XYZ files for small molecules typically have < 500 atoms
                    if 1 <= atom_count_xyz <= 500:
                        logger.info(f"Detected XYZ format with {atom_count_xyz} atoms")
                        return 'small_molecule'
                except ValueError:
                    # First line is not an integer, not XYZ format
                    pass
        except Exception:
            pass
    
    # Try to parse as small molecule with RDKit
    if RDKIT_AVAILABLE:
        try:
            # Try SDF/MOL format
            mol = Chem.MolFromMolBlock(structure_data)
            if mol is not None:
                atom_count_mol = mol.GetNumAtoms()
                # If very large, might be a protein (but unlikely in SDF)
                if atom_count_mol > 1000:
                    logger.warning(f"Very large molecule in SDF ({atom_count_mol} atoms)")
                    return 'unknown'
                return 'small_molecule'
        except Exception:
            pass
        
        try:
            # Try SMILES - but only for single-line inputs
            # This prevents XYZ files from being incorrectly parsed as SMILES
            stripped_data = structure_data.strip()
            if '\n' not in stripped_data and len(stripped_data) < 500:
                mol = Chem.MolFromSmiles(stripped_data)
                if mol is not None:
                    return 'small_molecule'
        except Exception:
            pass
    
    # If we have ATOM records but no protein residues detected, it might be a small molecule
    # But be careful - if we have many ATOM records, it's suspicious
    if atom_count > 0 and not has_protein_residues:
        # If we have many atoms (>50), it's suspicious - might be a protein we didn't detect
        if atom_count > 50:
            logger.warning(f"Many ATOM records ({atom_count}) but no protein residues detected - might be a protein")
            return 'unknown'
        return 'small_molecule'
    
    # If we only have HETATM and no ATOM records, it's definitely a small molecule
    # This handles extracted ligands that are written as pure HETATM records
    if hetatm_count > 0 and atom_count == 0:
        logger.info(f"Detected HETATM-only structure ({hetatm_count} HETATM, 0 ATOM) - treating as small molecule")
        return 'small_molecule'
    
    # Default: if no structure detected, return unknown
    if atom_count == 0 and hetatm_count == 0:
        # Might be SMILES or other text format
        if len(structure_data.strip()) < 200 and '\n' not in structure_data.strip():
            return 'small_molecule'
        return 'unknown'
    
    # Last resort: if we detected some structure but can't classify, return unknown
    logger.warning(f"Could not definitively determine structure type (atoms: {atom_count}, hetatm: {hetatm_count}, protein_residues: {protein_residue_count})")
    return 'unknown'


def _extract_structure_data(structure_input: Union[str, Dict[str, Any]], 
                           format_hint: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Extract structure data and format from various input types.
    
    Args:
        structure_input: Can be:
            - String (raw structure data)
            - Dict with 'pdb_data', 'sdf_data', 'xyz_data', or 'format' keys
        format_hint: Optional format hint
        
    Returns:
        Tuple of (structure_data_string, detected_format)
    """
    if isinstance(structure_input, str):
        return structure_input, format_hint
    
    if isinstance(structure_input, dict):
        # Try to get format from dict
        format_from_dict = structure_input.get('format')
        if format_hint is None:
            format_hint = format_from_dict
        
        # Try to extract structure data in priority order
        if 'sdf_data' in structure_input and structure_input['sdf_data']:
            return structure_input['sdf_data'], format_hint or 'sdf'
        elif 'xyz_data' in structure_input and structure_input['xyz_data']:
            return structure_input['xyz_data'], format_hint or 'xyz'
        elif 'pdb_data' in structure_input and structure_input['pdb_data']:
            return structure_input['pdb_data'], format_hint or 'pdb'
        elif 'smiles' in structure_input and structure_input['smiles']:
            return structure_input['smiles'], format_hint or 'smiles'
        else:
            raise StructureValidationError(
                "Structure dictionary does not contain valid structure data. "
                "Expected one of: 'pdb_data', 'sdf_data', 'xyz_data', or 'smiles'"
            )
    
    raise StructureValidationError(
        f"Invalid structure input type: {type(structure_input)}. "
        "Expected string or dictionary."
    )


def validate_structure_for_service(
    service_name: str,
    structure_data: Union[str, Dict[str, Any]],
    format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate that a structure matches the requirements for a specific service.
    
    Args:
        service_name: Name of the service (e.g., 'qc', 'docking', 'md')
        structure_data: Structure data (string or dict with structure data)
        format: Optional format hint ('pdb', 'sdf', 'xyz', etc.)
        
    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'structure_type': str,
            'detected_format': str,
            'errors': List[str],
            'warnings': List[str]
        }
        
    Raises:
        StructureValidationError: If validation fails critically
    """
    result = {
        'valid': False,
        'structure_type': None,
        'detected_format': None,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Get service requirements
        requirements = get_service_requirements(service_name)
        required_type = requirements['structure_type']
        allowed_formats = requirements['formats']
        
        # Extract structure data
        try:
            structure_string, detected_format = _extract_structure_data(structure_data, format)
            result['detected_format'] = detected_format
        except StructureValidationError as e:
            result['errors'].append(str(e))
            return result
        
        # Detect structure type
        try:
            structure_type = detect_structure_type(structure_string, detected_format)
            result['structure_type'] = structure_type
        except Exception as e:
            result['errors'].append(f"Could not detect structure type: {str(e)}")
            return result
        
        # Validate structure type matches requirement
        if required_type == 'protein':
            if structure_type not in ['protein', 'complex']:
                # More specific error messages
                if structure_type == 'small_molecule':
                    result['errors'].append(
                        f"{service_name.upper()} service requires a {requirements['description']}. "
                        f"The current structure is a small molecule. "
                        f"Please load a protein structure (PDB, CIF, or mmCIF format) instead."
                    )
                elif structure_type == 'unknown':
                    result['errors'].append(
                        f"{service_name.upper()} service requires a {requirements['description']}. "
                        f"Could not determine the structure type. "
                        f"Please ensure you have loaded a valid protein structure (PDB, CIF, or mmCIF format)."
                    )
                else:
                    result['errors'].append(
                        f"{service_name.upper()} service requires a {requirements['description']}. "
                        f"Current structure type: {structure_type.replace('_', ' ')}. "
                        f"Please load a protein structure (PDB, CIF, or mmCIF format) instead."
                    )
            elif structure_type == 'complex':
                result['warnings'].append(
                    "Structure contains both protein and ligands. "
                    "The protein component will be used."
                )
        elif required_type == 'small_molecule':
            if structure_type not in ['small_molecule']:
                if structure_type in ['protein', 'complex']:
                    result['errors'].append(
                        f"{service_name.upper()} service requires a {requirements['description']}. "
                        f"The current structure is a {structure_type.replace('_', ' ')}. "
                        f"Please load a small molecule structure (SDF or XYZ format) instead. "
                        f"QC and ADMET calculations are only available for small molecules."
                    )
                elif structure_type == 'unknown':
                    result['errors'].append(
                        f"{service_name.upper()} service requires a {requirements['description']}. "
                        f"Could not determine the structure type. "
                        f"Please ensure you have loaded a valid small molecule structure (SDF or XYZ format)."
                    )
                else:
                    result['errors'].append(
                        f"{service_name.upper()} service requires a {requirements['description']}. "
                        f"Current structure type: {structure_type.replace('_', ' ')}. "
                        f"Please load a small molecule structure (SDF or XYZ format) instead."
                    )
        
        # Validate format if detected
        if detected_format:
            detected_format_lower = detected_format.lower()
            # Normalize format names
            format_mapping = {
                'cif': 'cif',
                'mmcif': 'cif',
                'pdb': 'pdb'
            }
            normalized_format = format_mapping.get(detected_format_lower, detected_format_lower)
            
            # Check if format is allowed (with some flexibility)
            format_allowed = False
            for allowed in allowed_formats:
                if normalized_format == allowed or detected_format_lower == allowed:
                    format_allowed = True
                    break
                # Special case: SMILES is allowed for ADMET even if not in formats list
                if service_name == 'admet' and detected_format_lower == 'smiles':
                    format_allowed = True
                    break
            
            if not format_allowed:
                result['warnings'].append(
                    f"Detected format '{detected_format}' may not be optimal for {service_name}. "
                    f"Recommended formats: {', '.join(allowed_formats)}"
                )
        
        # If no errors, validation passed
        if not result['errors']:
            result['valid'] = True
        
    except ValueError as e:
        result['errors'].append(str(e))
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}", exc_info=True)
        result['errors'].append(f"Validation error: {str(e)}")
    
    return result

