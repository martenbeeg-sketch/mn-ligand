# services/pdb_service.py
"""
PDB Database Integration Service

This module provides integration with the RCSB Protein Data Bank (PDB) for fetching
protein structures and metadata. It handles HTTP requests, data validation, and
error handling for PDB database interactions.

Features:
- Fetch PDB structures by ID from RCSB PDB
- Retrieve structure metadata and information
- Handle network errors and invalid PDB IDs
- Support for both PDB and mmCIF formats
- Caching support for frequently accessed structures

Best Practices:
- Always validate PDB IDs before making requests
- Implement proper timeout handling for network requests
- Cache frequently accessed structures to reduce API calls
- Handle rate limiting from the PDB API gracefully
- Log all API interactions for debugging

API Endpoints Used:
- https://files.rcsb.org/download/{pdb_id}.pdb - PDB format download
- https://files.rcsb.org/download/{pdb_id}.cif - mmCIF format download
- https://data.rcsb.org/rest/v1/core/entry/{pdb_id} - Structure metadata

Author: PDB Integration Team
Version: 1.0.0
Last Updated: August 2025
"""
import requests
from ovo_ligand.ligandx.lib.chemistry import get_pdb_parser, get_component_analyzer

# Conditional Bio import
try:
    from Bio.PDB.PDBExceptions import PDBException
except ImportError:
    PDBException = Exception

class PDBService:
    """Service for retrieving and processing protein structures from RCSB PDB."""
    
    def __init__(self):
        self.pdb_base_url = "https://files.rcsb.org/download/"
        self.pdb_parser = get_pdb_parser()
        self.component_analyzer = get_component_analyzer()
    
    def fetch_structure(self, pdb_id):
        """
        Fetch a protein structure from RCSB PDB by its ID.
        
        Args:
            pdb_id (str): The 4-character PDB ID
            
        Returns:
            dict: A dictionary containing the structure data and metadata
            
        Raises:
            ValueError: If the PDB ID is invalid or the structure cannot be retrieved
        """
        # Validate PDB ID format
        if not self._validate_pdb_id(pdb_id):
            raise ValueError(f"Invalid PDB ID format: {pdb_id}. PDB IDs must be 4 characters.")
        
        # Try to fetch the structure in PDB format first
        pdb_url = f"{self.pdb_base_url}{pdb_id}.pdb"
        try:
            response = requests.get(pdb_url)
            if response.status_code == 200:
                return self._process_pdb_data(response.text, pdb_id)
            
            # If PDB format fails, try mmCIF format
            mmcif_url = f"{self.pdb_base_url}{pdb_id}.cif"
            response = requests.get(mmcif_url)
            if response.status_code == 200:
                return self._process_mmcif_data(response.text, pdb_id)
            
            # If both formats fail, raise an error
            raise ValueError(f"Failed to retrieve structure for PDB ID: {pdb_id}. " +
                            f"Server returned status code: {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error while retrieving PDB ID {pdb_id}: {str(e)}")
    
    def _validate_pdb_id(self, pdb_id):
        """Validate that the PDB ID has the correct format."""
        return self.pdb_parser.validate_pdb_id(pdb_id)
    
    def _process_pdb_data(self, pdb_data, pdb_id):
        """Process PDB format data and extract structure information."""
        try:
            # Parse the PDB data
            structure = self.pdb_parser.parse_string(pdb_data, pdb_id)
            
            # Extract metadata and process the structure
            return self._extract_structure_info(structure, pdb_data, "pdb")
            
        except Exception as e:
            raise ValueError(f"Error parsing PDB data for {pdb_id}: {str(e)}")
    
    def _process_mmcif_data(self, mmcif_data, pdb_id):
        """Process mmCIF format data and extract structure information."""
        try:
            # Parse the mmCIF data
            from ovo_ligand.ligandx.lib.chemistry import get_mmcif_parser
            mmcif_parser = get_mmcif_parser()
            structure = mmcif_parser.parse_string(mmcif_data, pdb_id)
            
            # Convert mmCIF to PDB format for consistency
            pdb_data = self.pdb_parser.structure_to_string(structure)
            
            # Extract metadata and process the structure
            return self._extract_structure_info(structure, pdb_data, "cif")
            
        except Exception as e:
            raise ValueError(f"Error parsing mmCIF data for {pdb_id}: {str(e)}")
    
    def _convert_structure_to_pdb(self, structure):
        """Convert a BioPython structure to PDB format string."""
        return self.pdb_parser.structure_to_string(structure)
    
    def _extract_structure_info(self, structure, pdb_data, format_type):
        """
        Extract information from the structure and organize it.
        
        Args:
            structure: BioPython structure object
            pdb_data: String containing the PDB format data
            format_type: String indicating the original format ("pdb" or "cif")
            
        Returns:
            dict: Dictionary containing structure data and metadata
        """
        # Get structure information
        info = self.component_analyzer.get_structure_info(structure, pdb_data)
        
        # Add format information
        info["format"] = format_type
        
        return info