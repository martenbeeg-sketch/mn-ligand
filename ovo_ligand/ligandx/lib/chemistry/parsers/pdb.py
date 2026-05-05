"""
PDB format parsing utilities.

Provides functionality for parsing and writing PDB format molecular structures.
"""

import io
import logging
from typing import Optional, List, Any

logger = logging.getLogger(__name__)

# Conditional Bio imports
try:
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.PDB.PDBExceptions import PDBException
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False
    PDBParser = None
    PDBIO = None
    Select = None
    PDBException = Exception


class ResidueSelector:
    """BioPython Select class for extracting specific residues."""
    
    def __init__(self, residue_list: List):
        if not BIO_AVAILABLE:
            raise ImportError("BioPython is required for ResidueSelector")
        self.residue_list = residue_list
    
    def accept_residue(self, residue):
        return residue in self.residue_list


# Create proper inheritance if BioPython is available
if BIO_AVAILABLE and Select is not None:
    class ResidueSelector(Select):
        """BioPython Select class for extracting specific residues."""
        
        def __init__(self, residue_list: List):
            self.residue_list = residue_list
        
        def accept_residue(self, residue):
            return residue in self.residue_list


class PDBParserUtils:
    """Utilities for parsing and writing PDB format structures."""
    
    def __init__(self):
        if not BIO_AVAILABLE or PDBParser is None:
            raise ImportError("BioPython is required for PDBParserUtils")
        self._parser = PDBParser(QUIET=True)
        self._io = PDBIO() if PDBIO is not None else None
    
    def parse_string(self, pdb_data: str, structure_id: str = "structure"):
        """
        Parse PDB data from string.
        
        Args:
            pdb_data: PDB format data as string
            structure_id: Identifier for the structure
            
        Returns:
            BioPython structure object
        """
        try:
            structure_file = io.StringIO(pdb_data)
            return self._parser.get_structure(structure_id, structure_file)
        except PDBException as e:
            raise ValueError(f"Error parsing PDB data: {str(e)}")
    
    def structure_to_string(self, structure) -> str:
        """
        Convert BioPython structure to PDB format string.
        
        Args:
            structure: BioPython structure object
            
        Returns:
            PDB format string
        """
        if self._io is None:
            raise ImportError("BioPython PDBIO not available")
        output = io.StringIO()
        self._io.set_structure(structure)
        self._io.save(output)
        return output.getvalue()
    
    def extract_residues_as_string(self, structure, residues: List, 
                                    structure_id: str = "extracted") -> str:
        """
        Extract specific residues from structure as PDB string.
        
        Args:
            structure: BioPython structure object
            residues: List of residues to extract
            structure_id: Identifier for extracted structure
            
        Returns:
            PDB format string containing only specified residues
        """
        if self._io is None:
            raise ImportError("BioPython PDBIO not available")
        output = io.StringIO()
        self._io.set_structure(structure)
        selector = ResidueSelector(residues)
        self._io.save(output, selector)
        return output.getvalue()
    
    def write_to_file(self, structure, output_path: str, 
                      select_residues: Optional[List] = None):
        """
        Write structure to PDB file.
        
        Args:
            structure: BioPython structure object
            output_path: Path to output file
            select_residues: Optional list of residues to select
        """
        if self._io is None:
            raise ImportError("BioPython PDBIO not available")
        self._io.set_structure(structure)
        
        if select_residues:
            selector = ResidueSelector(select_residues)
            self._io.save(output_path, selector)
        else:
            self._io.save(output_path)
    
    def validate_pdb_id(self, pdb_id: str) -> bool:
        """
        Validate PDB ID format.
        
        Args:
            pdb_id: PDB identifier
            
        Returns:
            True if valid, False otherwise
        """
        return (isinstance(pdb_id, str) and
                len(pdb_id) >= 4 and
                pdb_id.isalnum())


# Singleton instance
_pdb_parser_instance = None


def get_pdb_parser() -> PDBParserUtils:
    """Get or create PDBParserUtils singleton instance."""
    global _pdb_parser_instance
    if _pdb_parser_instance is None:
        _pdb_parser_instance = PDBParserUtils()
    return _pdb_parser_instance
