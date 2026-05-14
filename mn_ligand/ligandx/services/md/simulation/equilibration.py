"""
Equilibration module for MD optimization.

All functions return JSON-serializable data (dicts/strings/lists).
No unserialized objects (Simulation, Context, etc.) are passed between functions.
"""

import logging
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class Equilibration:
    """Handles equilibration for MD simulations with proper serialization."""
    
    def __init__(self):
        """Initialize equilibration utilities."""
        pass
    
    def validate_equilibration_parameters(
        self,
        nvt_steps: int = 25000,
        npt_steps: int = 25000,
        temperature: float = 300.0,
        pressure: float = 1.0
    ) -> Dict[str, Any]:
        """
        Validate equilibration parameters.
        
        Args:
            nvt_steps: NVT equilibration steps
            npt_steps: NPT equilibration steps
            temperature: Temperature in Kelvin
            pressure: Pressure in bar
            
        Returns:
            Dict with keys:
                - 'valid': bool
                - 'parameters': dict (validated parameters)
                - 'issues': list of str
        """
        issues = []
        
        if nvt_steps < 1:
            issues.append("nvt_steps must be >= 1")
        if npt_steps < 1:
            issues.append("npt_steps must be >= 1")
        
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
                'nvt_steps': nvt_steps,
                'npt_steps': npt_steps,
                'temperature': temperature,
                'pressure': pressure
            },
            'issues': issues
        }
    
    def get_equilibration_config(
        self,
        nvt_steps: int = 25000,
        npt_steps: int = 25000,
        temperature: float = 300.0,
        pressure: float = 1.0
    ) -> Dict[str, Any]:
        """
        Get equilibration configuration.
        
        Args:
            nvt_steps: NVT equilibration steps
            npt_steps: NPT equilibration steps
            temperature: Temperature in Kelvin
            pressure: Pressure in bar
            
        Returns:
            Dict with configuration (JSON-serializable)
        """
        return {
            'nvt_equilibration': {
                'steps': nvt_steps,
                'duration_ps': nvt_steps * 0.004,  # 2 fs timestep
                'temperature_K': temperature,
                'ensemble': 'NVT',
                'thermostat': 'Langevin'
            },
            'npt_equilibration': {
                'steps': npt_steps,
                'duration_ps': npt_steps * 0.004,  # 2 fs timestep
                'temperature_K': temperature,
                'pressure_bar': pressure,
                'ensemble': 'NPT',
                'thermostat': 'Langevin',
                'barostat': 'MonteCarloBarostat'
            },
            'total_equilibration_ps': (nvt_steps + npt_steps) * 0.004
        }
    
    def estimate_equilibration_time(
        self,
        nvt_steps: int = 25000,
        npt_steps: int = 25000
    ) -> Dict[str, Any]:
        """
        Estimate equilibration time.
        
        Args:
            nvt_steps: NVT equilibration steps
            npt_steps: NPT equilibration steps
            
        Returns:
            Dict with time estimates (JSON-serializable)
        """
        timestep_fs = 2.0  # 2 fs per step
        total_steps = nvt_steps + npt_steps
        total_fs = total_steps * timestep_fs
        total_ps = total_fs / 1000
        total_ns = total_ps / 1000
        
        # Rough estimate: 1 ns per 1000 steps on modern hardware
        estimated_hours = total_ns / 10  # Very rough estimate
        
        return {
            'total_steps': total_steps,
            'total_femtoseconds': total_fs,
            'total_picoseconds': total_ps,
            'total_nanoseconds': total_ns,
            'estimated_hours_cpu': estimated_hours,
            'note': 'Actual time depends on hardware and system size'
        }
