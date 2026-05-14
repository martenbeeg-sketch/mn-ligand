"""
System building module for MD optimization.

All functions return JSON-serializable data (dicts/strings/lists).
No unserialized objects are passed between functions.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SystemBuilder:
    """Handles system building for MD simulations with proper serialization."""
    
    @staticmethod
    def validate_system_parameters(
        ionic_strength: float = 0.15,
        temperature: float = 300.0,
        pressure: float = 1.0
    ) -> Dict[str, Any]:
        """
        Validate system building parameters.
        
        Args:
            ionic_strength: Ionic strength in M
            temperature: Temperature in K
            pressure: Pressure in bar
            
        Returns:
            Dict with keys:
                - 'valid': bool
                - 'parameters': dict
                - 'issues': list of str
        """
        issues = []
        
        if ionic_strength < 0:
            issues.append("ionic_strength must be >= 0")
        if ionic_strength > 5:
            issues.append("ionic_strength is very high (> 5 M)")
        
        if temperature < 0:
            issues.append("temperature must be >= 0 K")
        if temperature > 10000:
            issues.append("temperature is very high (> 10000 K)")
        
        if pressure < 0:
            issues.append("pressure must be >= 0 bar")
        if pressure > 10000:
            issues.append("pressure is very high (> 10000 bar)")
        
        return {
            'valid': len(issues) == 0,
            'parameters': {
                'ionic_strength': ionic_strength,
                'temperature': temperature,
                'pressure': pressure
            },
            'issues': issues
        }
    
    @staticmethod
    def get_system_config(
        ionic_strength: float = 0.15,
        temperature: float = 300.0,
        pressure: float = 1.0,
        water_model: str = 'tip3p'
    ) -> Dict[str, Any]:
        """
        Get system building configuration.
        
        Args:
            ionic_strength: Ionic strength in M
            temperature: Temperature in K
            pressure: Pressure in bar
            water_model: Water model ('tip3p', 'tip4p', 'tip5p')
            
        Returns:
            Dict with configuration (JSON-serializable)
        """
        return {
            'solvation': {
                'water_model': water_model,
                'ionic_strength_M': ionic_strength,
                'positive_ion': 'Na+',
                'negative_ion': 'Cl-',
                'neutralize': True
            },
            'conditions': {
                'temperature_K': temperature,
                'pressure_bar': pressure,
                'ph': 7.4
            },
            'force_fields': {
                'protein': 'AMBER14',
                'ligand': 'OpenFF-2.1.0',
                'water': water_model.upper()
            }
        }
    
    @staticmethod
    def estimate_system_size(
        protein_atoms: int,
        ligand_atoms: int,
        ionic_strength: float = 0.15,
        water_model: str = 'tip3p'
    ) -> Dict[str, Any]:
        """
        Estimate system size after solvation.
        
        Args:
            protein_atoms: Number of protein atoms
            ligand_atoms: Number of ligand atoms
            ionic_strength: Ionic strength in M
            water_model: Water model
            
        Returns:
            Dict with size estimates (JSON-serializable)
        """
        # Rough estimates
        solute_atoms = protein_atoms + ligand_atoms
        
        # Water molecules: ~30 Å³ per water, estimate box volume
        # Typical protein: ~1.4 g/cm³ density
        # Rough: 3-4 water molecules per solute atom
        water_molecules = solute_atoms * 3.5
        
        # Water atoms (3 per molecule)
        water_atoms = water_molecules * 3
        
        # Ions (rough estimate based on ionic strength)
        # ~0.15 M NaCl ~ 1 ion pair per 300 water molecules
        ion_pairs = max(1, int(water_molecules / 300))
        ion_atoms = ion_pairs * 2
        
        total_atoms = solute_atoms + water_atoms + ion_atoms
        
        return {
            'solute_atoms': solute_atoms,
            'protein_atoms': protein_atoms,
            'ligand_atoms': ligand_atoms,
            'water_molecules': int(water_molecules),
            'water_atoms': int(water_atoms),
            'ion_pairs': ion_pairs,
            'ion_atoms': ion_atoms,
            'total_atoms': int(total_atoms),
            'note': 'Rough estimates for system size'
        }
    
    @staticmethod
    def get_solvation_options() -> Dict[str, Any]:
        """
        Get available solvation options.
        
        Returns:
            Dict with solvation options (JSON-serializable)
        """
        return {
            'water_models': [
                {
                    'name': 'tip3p',
                    'description': 'TIP3P water model',
                    'atoms_per_molecule': 3,
                    'speed': 'fast',
                    'accuracy': 'medium'
                },
                {
                    'name': 'tip4p',
                    'description': 'TIP4P water model',
                    'atoms_per_molecule': 4,
                    'speed': 'medium',
                    'accuracy': 'high'
                },
                {
                    'name': 'tip5p',
                    'description': 'TIP5P water model',
                    'atoms_per_molecule': 5,
                    'speed': 'slow',
                    'accuracy': 'very_high'
                }
            ],
            'default_water_model': 'tip3p',
            'ions': {
                'positive': ['Na+', 'K+', 'Mg2+', 'Ca2+'],
                'negative': ['Cl-', 'Br-', 'I-']
            },
            'default_ions': {
                'positive': 'Na+',
                'negative': 'Cl-'
            }
        }
