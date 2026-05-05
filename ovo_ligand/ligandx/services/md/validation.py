"""
Input validation module for MD optimization service.

Provides validation utilities for service inputs and results.
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_system_result(result: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate system creation result.
    
    Args:
        result: System creation result dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not result:
        return False, "System result is None"
    
    if not isinstance(result, dict):
        return False, "System result must be a dictionary"
    
    if result.get("status") != "success":
        error_msg = result.get("error", "Unknown error")
        return False, f"System creation failed: {error_msg}"
    
    if "simulation" not in result:
        return False, "System result missing 'simulation' key"
    
    if "total_atoms" not in result:
        return False, "System result missing 'total_atoms' key"
    
    return True, ""


def validate_equilibration_result(result: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate equilibration result.
    
    Args:
        result: Equilibration result dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not result:
        return False, "Equilibration result is None"
    
    if not isinstance(result, dict):
        return False, "Equilibration result must be a dictionary"
    
    status = result.get("status")
    valid_statuses = {"success", "minimized_ready"}
    
    if status not in valid_statuses:
        error_msg = result.get("error", "Unknown error")
        return False, f"Equilibration failed with status '{status}': {error_msg}"
    
    return True, ""


def validate_ligand_preparation(ligand: Any, ligand_id: str) -> Tuple[bool, str]:
    """
    Validate prepared ligand.
    
    Args:
        ligand: Prepared ligand object
        ligand_id: Ligand identifier
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if ligand is None:
        return False, f"Ligand preparation failed for {ligand_id}"
    
    return True, ""


def validate_protein_preparation(protein_path: str, protein_id: str) -> Tuple[bool, str]:
    """
    Validate prepared protein.
    
    Args:
        protein_path: Path to prepared protein PDB file
        protein_id: Protein identifier
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not protein_path:
        return False, f"Protein preparation failed for {protein_id}"
    
    return True, ""


__all__ = [
    'ValidationError',
    'validate_system_result',
    'validate_equilibration_result',
    'validate_ligand_preparation',
    'validate_protein_preparation',
]
