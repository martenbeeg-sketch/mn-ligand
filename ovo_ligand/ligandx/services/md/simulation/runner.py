"""
Simulation runner module for MD optimization.

Handles OpenMM simulation creation, platform selection, and execution.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class SimulationRunner:
    """Handles OpenMM simulation setup and execution."""
    
    def __init__(self, output_dir: str = "data/md_outputs"):
        """
        Initialize simulation runner.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def create_simulation(
        self,
        topology,
        system,
        integrator,
        positions,
        platform_name: Optional[str] = None
    ) -> Tuple[Any, str]:
        """
        Create OpenMM Simulation with robust platform fallback.
        
        Args:
            topology: OpenMM Topology
            system: OpenMM System
            integrator: OpenMM Integrator
            positions: Initial positions
            platform_name: Preferred platform (None for auto-select)
            
        Returns:
            Tuple of (Simulation object, platform name used)
        """
        from openmm import Platform
        from openmm.app import Simulation
        
        simulation = None
        used_platform = None
        
        # Define platform priority
        platforms_to_try = []
        if platform_name:
            platforms_to_try.append(platform_name)
        platforms_to_try.extend(['CUDA', 'OpenCL', 'CPU'])
        
        for pname in platforms_to_try:
            try:
                logger.info(f"Attempting to initialize simulation with {pname} platform...")
                platform = Platform.getPlatformByName(pname)
                
                if pname == 'CUDA':
                    properties = {'Precision': 'mixed', 'CudaDeviceIndex': '0'}
                    simulation = Simulation(topology, system, integrator, platform, properties)
                elif pname == 'OpenCL':
                    properties = {'Precision': 'mixed'}
                    simulation = Simulation(topology, system, integrator, platform, properties)
                else:  # CPU
                    simulation = Simulation(topology, system, integrator, platform)
                
                used_platform = pname
                logger.info(f"[COMPLETE] Successfully initialized simulation with {pname}")
                break
                
            except Exception as e:
                logger.warning(f"Failed to initialize {pname} simulation: {e}")
                continue
        
        if simulation is None:
            raise RuntimeError("Could not initialize simulation on any platform (CUDA, OpenCL, CPU)")
        
        # Set initial positions
        simulation.context.setPositions(positions)
        
        return simulation, used_platform
    
    def create_integrator(
        self,
        temperature_k: float = 300.0,
        friction_ps: float = 1.0,
        timestep_fs: float = 2.0
    ):
        """
        Create Langevin integrator.
        
        Args:
            temperature_k: Temperature in Kelvin
            friction_ps: Friction coefficient in 1/ps
            timestep_fs: Timestep in femtoseconds
            
        Returns:
            LangevinMiddleIntegrator
        """
        from openmm import LangevinMiddleIntegrator, unit
        
        return LangevinMiddleIntegrator(
            temperature_k * unit.kelvin,
            friction_ps / unit.picosecond,
            timestep_fs * unit.femtoseconds
        )
    
    def add_barostat(
        self,
        system,
        pressure_bar: float = 1.0,
        temperature_k: float = 300.0,
        frequency: int = 25
    ):
        """
        Add Monte Carlo barostat to system.
        
        Args:
            system: OpenMM System
            pressure_bar: Pressure in bar
            temperature_k: Temperature in Kelvin
            frequency: Update frequency in steps
        """
        from openmm import MonteCarloBarostat, unit
        
        barostat = MonteCarloBarostat(
            pressure_bar * unit.bar,
            temperature_k * unit.kelvin,
            frequency
        )
        system.addForce(barostat)
        logger.info(f"[COMPLETE] Added barostat: {pressure_bar} bar, {temperature_k} K")
    
        
    def save_state(
        self,
        simulation,
        output_path: str,
        keep_ids: bool = True
    ):
        """
        Save current simulation state to PDB file.
        
        Args:
            simulation: OpenMM Simulation
            output_path: Path to save PDB
            keep_ids: Whether to keep original residue IDs
        """
        from ..utils.pdb_utils import write_pdb_file
        
        positions = simulation.context.getState(getPositions=True).getPositions()
        write_pdb_file(simulation.topology, positions, output_path, keep_ids)
        logger.info(f"[COMPLETE] Saved state to {output_path}")
