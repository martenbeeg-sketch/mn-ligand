"""
PDB file writing utilities for MD simulations.

All functions return JSON-serializable data (dicts/strings/lists).
No unserialized objects are passed between functions.
"""

import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class PDBWriter:
    """Handles PDB file writing with proper serialization."""
    
    @staticmethod
    def validate_pdb_data(pdb_data: str) -> Dict[str, Any]:
        """
        Validate PDB data format.
        
        Args:
            pdb_data: PDB format string
            
        Returns:
            Dict with keys:
                - 'valid': bool
                - 'line_count': int
                - 'has_atoms': bool
                - 'issues': list of str
        """
        if not pdb_data:
            return {
                'valid': False,
                'line_count': 0,
                'has_atoms': False,
                'issues': ['Empty PDB data']
            }
        
        lines = pdb_data.strip().split('\n')
        atom_lines = [l for l in lines if l.startswith(('ATOM', 'HETATM'))]
        issues = []
        
        if not atom_lines:
            issues.append('No ATOM or HETATM records found')
        
        if not any(l.startswith('END') for l in lines):
            issues.append('Missing END record')
        
        return {
            'valid': len(issues) == 0,
            'line_count': len(lines),
            'has_atoms': len(atom_lines) > 0,
            'atom_count': len(atom_lines),
            'issues': issues
        }
    
    @staticmethod
    def get_pdb_statistics(pdb_data: str) -> Dict[str, Any]:
        """
        Get statistics about PDB data.
        
        Args:
            pdb_data: PDB format string
            
        Returns:
            Dict with statistics (JSON-serializable)
        """
        if not pdb_data:
            return {
                'atom_count': 0,
                'residue_count': 0,
                'chain_count': 0,
                'has_hetatm': False
            }
        
        lines = pdb_data.strip().split('\n')
        atom_count = len([l for l in lines if l.startswith('ATOM')])
        hetatm_count = len([l for l in lines if l.startswith('HETATM')])
        
        # Extract unique residues and chains
        residues = set()
        chains = set()
        
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                if len(line) >= 27:
                    chain = line[21]
                    chains.add(chain)
                if len(line) >= 26:
                    residue_num = line[22:26].strip()
                    residues.add(residue_num)
        
        return {
            'atom_count': atom_count,
            'hetatm_count': hetatm_count,
            'total_atoms': atom_count + hetatm_count,
            'residue_count': len(residues),
            'chain_count': len(chains),
            'has_hetatm': hetatm_count > 0
        }
    
    @staticmethod
    def sanitize_pdb_data(pdb_data: str) -> Dict[str, Any]:
        """
        Sanitize PDB data by fixing common issues.
        
        Args:
            pdb_data: PDB format string
            
        Returns:
            Dict with keys:
                - 'success': bool
                - 'pdb_data': str (sanitized)
                - 'changes': list of str (what was fixed)
                - 'error': str (if failed)
        """
        try:
            if not pdb_data:
                return {
                    'success': False,
                    'pdb_data': None,
                    'changes': [],
                    'error': 'Empty PDB data'
                }
            
            lines = pdb_data.strip().split('\n')
            changes = []
            sanitized_lines = []
            
            for line in lines:
                # Ensure END record exists
                if line.startswith('END'):
                    sanitized_lines.append(line)
                    continue
                
                # Fix common formatting issues
                if line.startswith(('ATOM', 'HETATM')):
                    # Ensure line is long enough
                    if len(line) < 66:
                        line = line.ljust(66)
                        changes.append(f"Extended short line: {line[:20]}")
                    
                    sanitized_lines.append(line)
                else:
                    sanitized_lines.append(line)
            
            # Ensure END record
            if not any(l.startswith('END') for l in sanitized_lines):
                sanitized_lines.append('END')
                changes.append('Added missing END record')
            
            sanitized_pdb = '\n'.join(sanitized_lines)
            
            return {
                'success': True,
                'pdb_data': sanitized_pdb,
                'changes': changes,
                'error': None
            }
        
        except Exception as e:
            logger.error(f"PDB sanitization failed: {e}")
            return {
                'success': False,
                'pdb_data': None,
                'changes': [],
                'error': f"Sanitization failed: {str(e)}"
            }
