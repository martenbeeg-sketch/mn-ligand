"""
Charge assignment module for MD optimization.

All functions return JSON-serializable data (dicts/strings/lists).
No unserialized objects are passed between functions.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChargeAssignment:
    """Handles partial charge assignment for MD simulations with proper serialization."""
    
    @staticmethod
    def get_available_methods() -> Dict[str, Any]:
        """
        Get available charge assignment methods.
        
        Returns:
            Dict with available methods and their properties (JSON-serializable)
        """
        return {
            'methods': [
                {
                    'name': 'mmff94',
                    'description': 'MMFF94 force field charges',
                    'speed': 'fast',
                    'reliability': 'high',
                    'requires_rdkit': True,
                    'requires_ambertools': False
                },
                {
                    'name': 'gasteiger',
                    'description': 'Gasteiger partial charges',
                    'speed': 'very_fast',
                    'reliability': 'medium',
                    'requires_rdkit': True,
                    'requires_ambertools': False
                },
                {
                    'name': 'am1bcc',
                    'description': 'AM1-BCC charges (requires AmberTools)',
                    'speed': 'slow',
                    'reliability': 'very_high',
                    'requires_rdkit': True,
                    'requires_ambertools': True
                }
            ],
            'default_method': 'mmff94',
            'fallback_method': 'gasteiger'
        }
    
    @staticmethod
    def validate_charge_method(method: str) -> Dict[str, Any]:
        """
        Validate charge assignment method.
        
        Args:
            method: Charge method name
            
        Returns:
            Dict with keys:
                - 'valid': bool
                - 'method': str
                - 'issues': list of str
        """
        available = ChargeAssignment.get_available_methods()
        valid_methods = [m['name'] for m in available['methods']]
        
        issues = []
        if method not in valid_methods:
            issues.append(f"Unknown method: {method}. Available: {', '.join(valid_methods)}")
        
        return {
            'valid': len(issues) == 0,
            'method': method,
            'issues': issues
        }
    
    @staticmethod
    def get_charge_config(
        primary_method: str = 'mmff94',
        fallback_method: str = 'gasteiger'
    ) -> Dict[str, Any]:
        """
        Get charge assignment configuration.
        
        Args:
            primary_method: Primary charge method
            fallback_method: Fallback method if primary fails
            
        Returns:
            Dict with configuration (JSON-serializable)
        """
        return {
            'primary_method': primary_method,
            'fallback_method': fallback_method,
            'strategy': 'fallback',
            'description': f'Try {primary_method}, fallback to {fallback_method} if needed',
            'avoid_am1bcc': False,
            'reason': 'AM1-BCC available via OpenFE with explicit ambertools backend'
        }
    
    @staticmethod
    def estimate_charge_time(atom_count: int, method: str = 'mmff94') -> Dict[str, Any]:
        """
        Estimate charge assignment time.
        
        Args:
            atom_count: Number of atoms in molecule
            method: Charge method
            
        Returns:
            Dict with time estimates (JSON-serializable)
        """
        # Rough time estimates (ms per atom)
        time_per_atom = {
            'mmff94': 0.5,      # 0.5 ms per atom
            'gasteiger': 0.1,   # 0.1 ms per atom (very fast)
            'am1bcc': 10.0      # 10 ms per atom (slow)
        }
        
        ms_per_atom = time_per_atom.get(method, 1.0)
        total_ms = atom_count * ms_per_atom
        
        return {
            'method': method,
            'atom_count': atom_count,
            'ms_per_atom': ms_per_atom,
            'estimated_ms': total_ms,
            'estimated_seconds': total_ms / 1000,
            'note': 'Rough estimate for charge assignment'
        }
    
    @staticmethod
    def get_charge_assignment_status(
        method_used: str,
        fallback_used: bool = False
    ) -> Dict[str, Any]:
        """
        Get status of charge assignment.
        
        Args:
            method_used: Method that was used
            fallback_used: Whether fallback was used
            
        Returns:
            Dict with status information (JSON-serializable)
        """
        return {
            'method_used': method_used,
            'fallback_used': fallback_used,
            'status': 'success',
            'message': f'Charges assigned using {method_used}' + (
                ' (fallback)' if fallback_used else ''
            )
        }
