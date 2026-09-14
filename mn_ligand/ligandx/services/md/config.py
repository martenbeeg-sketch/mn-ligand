"""
Configuration classes for MD optimization service.

Provides structured configuration objects for MD workflows.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MDOptimizationConfig:
    """Configuration for MD optimization workflow."""

    protein_pdb_data: str
    ligand_smiles: Optional[str] = None
    ligand_structure_data: Optional[str] = None
    ligand_refined_sdf_data: Optional[str] = None
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
    protein_forcefield_method: str = "amber14-all"
    water_model: str = "tip3p"
    box_shape: str = "dodecahedron"
    nvt_steps: int = 25000
    npt_steps: int = 175000
    # Thermal heating protocol runs as 6 temperature stages (50K increments) with 1 fs timestep.
    # Total heating duration (ps) = 6 * heating_steps_per_stage * 0.001
    heating_steps_per_stage: int = 2500
    production_steps: int = 0
    production_report_interval: int = 2500
    integration_profile: str = "hmr_4fs"
    production_timestep_fs: float = 4.0
    hydrogen_mass_amu: Optional[float] = 4.0
    temperature: float = 300.0
    pressure: float = 1.0
    ionic_strength: float = 0.15
    padding_nm: float = 1.0
    minimization_max_iterations: int = 5000
    minimization_tolerance_kjmol_nm: float = 10.0
    preparation_protocol: str = "current_staged"
    density_stabilization_min_ns: float = 1.0
    density_stabilization_max_ns: float = 5.0
    density_stabilization_increment_ns: float = 1.0
    density_sample_interval_ps: float = 4.0
    density_plateau_required: bool = True
    heating_start_temperature: float = 50.0
    heating_stages: int = 6
    npt_restraint_release_scales: str = "1.0,0.5,0.2,0.05,0.0"
    npt_release_enabled: bool = True
    protein_npt_release_scales: str = "1.0,0.5,0.1,0.01,0.0"
    planarity_npt_release_scales: str = "1.0,0.5,0.2,0.05,0.0"
    allow_restrained_production: bool = False
    force_unrestrained_production: bool = True
    resume_from_checkpoint_path: Optional[str] = None
    resume_system_pdb_path: Optional[str] = None
    resume_state_xml_path: Optional[str] = None
    resume_system_xml_path: Optional[str] = None
    resume_integrator_xml_path: Optional[str] = None
    production_only_from_prepared: bool = False
    strict_checkpoint_resume: bool = False
    append_production_outputs: bool = False
    production_prior_steps: int = 0
    coordinate_restart_policy: str = "legacy_minimize_rethermalize"
    replica_equilibration_steps: int = 0
    replica_density_revalidation: bool = False
    replica_revalidation_max_steps: int = 0
    replica_revalidation_increment_steps: int = 0
    replica_density_sample_interval_steps: int = 0
    replica_density_plateau_required: bool = True
    replica_seed: Optional[int] = None
    md_backend: str = "openmm_openff"
    amber_complex_prmtop_path: Optional[str] = None
    amber_complex_inpcrd_path: Optional[str] = None
    amber_system_pdb_path: Optional[str] = None
    residue_mapping: Optional[dict[str, Any]] = None

    def validate(self) -> tuple[bool, str]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.protein_pdb_data:
            return False, "protein_pdb_data is required"
        if float(self.production_timestep_fs) <= 0:
            return False, "production_timestep_fs must be > 0"
        if (
            float(self.production_timestep_fs) > 2.0
            and self.hydrogen_mass_amu is None
        ):
            return False, (
                "production timesteps above 2 fs require hydrogen-mass "
                "repartitioning"
            )
        
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

        valid_protein_forcefields = {
            "amber14-all",
            "amber99sbildn",
            "amberfb15",
            "amber15ipq",
            "ff14SB",
            "ff19SB",
            "ff15ipq",
            "ff03.r1",
        }
        if self.protein_forcefield_method not in valid_protein_forcefields:
            return (
                False,
                "protein_forcefield_method must be one of "
                + ", ".join(sorted(valid_protein_forcefields)),
            )
        valid_water_models = {
            "tip3p",
            "tip3pfb",
            "opc",
            "spce",
            "tip4pew",
        }
        if self.water_model not in valid_water_models:
            return (
                False,
                "water_model must be one of "
                + ", ".join(sorted(valid_water_models)),
            )

        valid_preparation_protocols = {
            "current_staged",
            "roe_brooks_2020",
        }
        preparation_protocol = str(
            self.preparation_protocol or "current_staged"
        ).strip().lower()
        if preparation_protocol not in valid_preparation_protocols:
            return (
                False,
                "preparation_protocol must be current_staged or "
                "roe_brooks_2020",
            )
        if preparation_protocol == "roe_brooks_2020":
            if float(self.density_stabilization_min_ns) <= 0:
                return False, "density_stabilization_min_ns must be > 0"
            if (
                float(self.density_stabilization_max_ns)
                < float(self.density_stabilization_min_ns)
            ):
                return (
                    False,
                    "density_stabilization_max_ns must be >= "
                    "density_stabilization_min_ns",
                )
            if float(self.density_stabilization_increment_ns) <= 0:
                return False, "density_stabilization_increment_ns must be > 0"
            if float(self.density_sample_interval_ps) <= 0:
                return False, "density_sample_interval_ps must be > 0"

        if (self.md_backend or "").strip().lower() == "amber_native":
            if not self.amber_complex_prmtop_path or not str(self.amber_complex_prmtop_path).strip():
                return False, "amber_complex_prmtop_path is required for amber_native backend"
            if not self.amber_complex_inpcrd_path or not str(self.amber_complex_inpcrd_path).strip():
                return False, "amber_complex_inpcrd_path is required for amber_native backend"

        if self.coordinate_restart_policy == "independent_replica":
            if self.replica_equilibration_steps < 0:
                return False, "replica_equilibration_steps must be >= 0"
            if self.replica_density_revalidation:
                if self.replica_revalidation_max_steps < self.replica_equilibration_steps:
                    return False, (
                        "replica_revalidation_max_steps must be >= "
                        "replica_equilibration_steps"
                    )
                if self.replica_revalidation_increment_steps <= 0:
                    return False, "replica_revalidation_increment_steps must be > 0"
                if self.replica_density_sample_interval_steps <= 0:
                    return False, "replica_density_sample_interval_steps must be > 0"
            if not self.resume_system_pdb_path:
                return False, "independent replica requires equilibrated coordinates"
            if not self.resume_state_xml_path:
                return False, "independent replica requires a serialized OpenMM State"
            if not self.resume_system_xml_path or not self.resume_integrator_xml_path:
                return False, "independent replica requires serialized OpenMM System and Integrator"

        if self.strict_checkpoint_resume:
            if not self.resume_from_checkpoint_path:
                return False, "strict continuation requires a checkpoint"
            if not self.resume_system_xml_path or not self.resume_integrator_xml_path:
                return False, "strict continuation requires serialized OpenMM System and Integrator"

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
            ligand_refined_sdf_data=data.get('ligand_refined_sdf_data'),
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
            protein_forcefield_method=data.get(
                'protein_forcefield_method', 'amber14-all'
            ),
            water_model=data.get('water_model', 'tip3p'),
            box_shape=data.get('box_shape', 'dodecahedron'),
            nvt_steps=data.get('nvt_steps', 25000),
            npt_steps=data.get('npt_steps', 175000),
            heating_steps_per_stage=data.get('heating_steps_per_stage', 2500),
            production_steps=data.get('production_steps', 0),
            production_report_interval=data.get('production_report_interval', 2500),
            integration_profile=data.get('integration_profile', 'hmr_4fs'),
            production_timestep_fs=data.get('production_timestep_fs', 4.0),
            hydrogen_mass_amu=data.get('hydrogen_mass_amu', 4.0),
            temperature=data.get('temperature', 300.0),
            pressure=data.get('pressure', 1.0),
            ionic_strength=data.get('ionic_strength', 0.15),
            padding_nm=data.get('padding_nm', 1.0),
            minimization_max_iterations=data.get('minimization_max_iterations', 5000),
            minimization_tolerance_kjmol_nm=data.get('minimization_tolerance_kjmol_nm', 10.0),
            preparation_protocol=data.get('preparation_protocol', 'current_staged'),
            density_stabilization_min_ns=data.get('density_stabilization_min_ns', 1.0),
            density_stabilization_max_ns=data.get('density_stabilization_max_ns', 5.0),
            density_stabilization_increment_ns=data.get('density_stabilization_increment_ns', 1.0),
            density_sample_interval_ps=data.get('density_sample_interval_ps', 4.0),
            density_plateau_required=data.get('density_plateau_required', True),
            heating_start_temperature=data.get('heating_start_temperature', 50.0),
            heating_stages=data.get('heating_stages', 6),
            npt_restraint_release_scales=data.get('npt_restraint_release_scales', "1.0,0.5,0.2,0.05,0.0"),
            npt_release_enabled=data.get('npt_release_enabled', True),
            protein_npt_release_scales=data.get('protein_npt_release_scales', "1.0,0.5,0.1,0.01,0.0"),
            planarity_npt_release_scales=data.get('planarity_npt_release_scales', "1.0,0.5,0.2,0.05,0.0"),
            allow_restrained_production=data.get('allow_restrained_production', False),
            force_unrestrained_production=data.get('force_unrestrained_production', True),
            resume_from_checkpoint_path=data.get('resume_from_checkpoint_path'),
            resume_system_pdb_path=data.get('resume_system_pdb_path'),
            resume_state_xml_path=data.get('resume_state_xml_path'),
            resume_system_xml_path=data.get('resume_system_xml_path'),
            resume_integrator_xml_path=data.get('resume_integrator_xml_path'),
            production_only_from_prepared=data.get('production_only_from_prepared', False),
            strict_checkpoint_resume=data.get('strict_checkpoint_resume', False),
            append_production_outputs=data.get('append_production_outputs', False),
            production_prior_steps=data.get('production_prior_steps', 0),
            coordinate_restart_policy=data.get('coordinate_restart_policy', 'legacy_minimize_rethermalize'),
            replica_equilibration_steps=data.get('replica_equilibration_steps', 0),
            replica_density_revalidation=data.get('replica_density_revalidation', False),
            replica_revalidation_max_steps=data.get('replica_revalidation_max_steps', 0),
            replica_revalidation_increment_steps=data.get('replica_revalidation_increment_steps', 0),
            replica_density_sample_interval_steps=data.get('replica_density_sample_interval_steps', 0),
            replica_density_plateau_required=data.get('replica_density_plateau_required', True),
            replica_seed=data.get('replica_seed'),
            md_backend=data.get('md_backend', 'openmm_openff'),
            amber_complex_prmtop_path=data.get('amber_complex_prmtop_path'),
            amber_complex_inpcrd_path=data.get('amber_complex_inpcrd_path'),
            amber_system_pdb_path=data.get('amber_system_pdb_path'),
            residue_mapping=data.get('residue_mapping'),
        )


__all__ = ['MDOptimizationConfig']
