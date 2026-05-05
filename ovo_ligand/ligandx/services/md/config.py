"""
Configuration classes for MD optimization service.

Provides structured configuration objects for MD workflows.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MDOptimizationConfig:
    """Configuration for MD optimization workflow."""

    protein_pdb_data: str
    ligand_smiles: Optional[str] = None
    ligand_structure_data: Optional[str] = None
    ligand_data_format: str = "sdf"
    preserve_ligand_pose: bool = True
    generate_conformer: bool = True
    protein_id: str = "protein"
    ligand_id: str = "ligand"
    system_id: str = "system"
    preview_before_equilibration: bool = False
    preview_acknowledged: bool = False
    pause_at_minimized: bool = False
    minimization_only: bool = False
    minimized_acknowledged: bool = False
    job_id: Optional[str] = None
    charge_method: str = "am1bcc"
    forcefield_method: str = "openff-2.2.0"
    box_shape: str = "dodecahedron"
    nvt_steps: int = 25000
    npt_steps: int = 175000
    # Thermal heating protocol runs as 6 temperature stages (50K increments) with 1 fs timestep.
    # Total heating duration (ps) = 6 * heating_steps_per_stage * 0.001
    heating_steps_per_stage: int = 2500
    production_steps: int = 0
    production_report_interval: int = 2500
    temperature: float = 300.0
    pressure: float = 1.0
    ionic_strength: float = 0.15
    padding_nm: float = 1.0

    def validate(self) -> tuple[bool, str]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.protein_pdb_data:
            return False, "protein_pdb_data is required"
        
        if not self.protein_pdb_data.strip():
            return False, "protein_pdb_data cannot be empty"
        
        # Check ligand input
        has_smiles = bool(self.ligand_smiles and self.ligand_smiles.strip())
        has_structure = bool(self.ligand_structure_data and self.ligand_structure_data.strip())
        
        if has_smiles and has_structure:
            return False, "Provide either ligand_smiles OR ligand_structure_data, not both"

        # Validate format (only when ligand data is provided)
        if has_smiles or has_structure:
            valid_formats = {"sdf", "mol", "pdb"}
            if self.ligand_data_format.lower() not in valid_formats:
                return False, f"ligand_data_format must be one of {valid_formats}"

        # Validate IDs
        if not self.protein_id or not self.protein_id.strip():
            return False, "protein_id cannot be empty"

        if (has_smiles or has_structure) and (not self.ligand_id or not self.ligand_id.strip()):
            return False, "ligand_id cannot be empty"

        if not self.system_id or not self.system_id.strip():
            return False, "system_id cannot be empty"

        return True, ""

    @property
    def is_protein_only(self) -> bool:
        has_smiles = bool(self.ligand_smiles and self.ligand_smiles.strip())
        has_structure = bool(self.ligand_structure_data and self.ligand_structure_data.strip())
        return not has_smiles and not has_structure
    
    @classmethod
    def from_dict(cls, data: dict) -> "MDOptimizationConfig":
        """
        Create config from dictionary.

        Args:
            data: Dictionary with configuration parameters

        Returns:
            MDOptimizationConfig instance
        """
        return cls(
            protein_pdb_data=data.get('protein_pdb_data', ''),
            ligand_smiles=data.get('ligand_smiles'),
            ligand_structure_data=data.get('ligand_structure_data') or data.get('ligand_sdf_data'),
            ligand_data_format=data.get('ligand_data_format', 'sdf'),
            preserve_ligand_pose=data.get('preserve_ligand_pose', True),
            generate_conformer=data.get('generate_conformer', True),
            protein_id=data.get('protein_id', 'protein'),
            ligand_id=data.get('ligand_id', 'ligand'),
            system_id=data.get('system_id', 'system'),
            preview_before_equilibration=data.get('preview_before_equilibration', False),
            preview_acknowledged=data.get('preview_acknowledged', False),
            pause_at_minimized=data.get('pause_at_minimized', False),
            minimization_only=data.get('minimization_only', False),
            minimized_acknowledged=data.get('minimized_acknowledged', False),
            job_id=data.get('job_id'),
            charge_method=data.get('charge_method', 'am1bcc'),
            forcefield_method=data.get('forcefield_method', 'openff-2.2.0'),
            box_shape=data.get('box_shape', 'dodecahedron'),
            nvt_steps=data.get('nvt_steps', 25000),
            npt_steps=data.get('npt_steps', 175000),
            heating_steps_per_stage=data.get('heating_steps_per_stage', 2500),
            production_steps=data.get('production_steps', 0),
            production_report_interval=data.get('production_report_interval', 2500),
            temperature=data.get('temperature', 300.0),
            pressure=data.get('pressure', 1.0),
            ionic_strength=data.get('ionic_strength', 0.15),
            padding_nm=data.get('padding_nm', 1.0),
        )


__all__ = ['MDOptimizationConfig']
