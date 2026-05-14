"""
mmCIF format parsing utilities.

Provides functionality for parsing mmCIF format molecular structures.
"""

import io
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Conditional Bio imports
try:
    from Bio.PDB import MMCIFParser
    from Bio.PDB.PDBExceptions import PDBException
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False
    MMCIFParser = None
    PDBException = Exception


class MMCIFParserUtils:
    """Utilities for parsing mmCIF format structures."""
    
    def __init__(self):
        if not BIO_AVAILABLE or MMCIFParser is None:
            raise ImportError("BioPython is required for MMCIFParserUtils")
        self._parser = MMCIFParser(QUIET=True)
    
    def parse_string(self, mmcif_data: str, structure_id: str = "structure"):
        """
        Parse mmCIF data from string.
        
        Args:
            mmcif_data: mmCIF format data as string
            structure_id: Identifier for the structure
            
        Returns:
            BioPython structure object
        """
        try:
            structure_file = io.StringIO(mmcif_data)
            return self._parser.get_structure(structure_id, structure_file)
        except PDBException as e:
            raise ValueError(f"Error parsing mmCIF data: {str(e)}")


# Singleton instance
_mmcif_parser_instance = None


def get_mmcif_parser() -> MMCIFParserUtils:
    """Get or create MMCIFParserUtils singleton instance."""
    global _mmcif_parser_instance
    if _mmcif_parser_instance is None:
        _mmcif_parser_instance = MMCIFParserUtils()
    return _mmcif_parser_instance
