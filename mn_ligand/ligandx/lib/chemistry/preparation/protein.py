"""
Protein structure preparation utilities.

Provides functionality for cleaning and preparing protein structures
using PDBFixer and OpenMM.
"""

import io
import logging
from typing import Dict, Any, Optional

from mn_ligand.ligandx.lib.chemistry.preparation.target_validation import (
    prepare_target_for_publication,
    remove_internal_oxt,
)

logger = logging.getLogger(__name__)

# Optional imports with availability flags
try:
    import pdbfixer
    import openmm as mm
    from openmm import unit
    from openmm.app import CutoffNonPeriodic, ForceField, Modeller, Simulation
    from openmm.app import PDBFile
    PDBFIXER_AVAILABLE = True
except ImportError:
    logger.warning("PDBFixer not available. Protein cleaning will be disabled.")
    PDBFIXER_AVAILABLE = False


class ProteinPreparer:
    """Utilities for preparing protein structures."""
    
    def __init__(self):
        if not PDBFIXER_AVAILABLE:
            logger.warning("PDBFixer not available - protein preparation features limited")
    
    def _remove_heterogens_stage(self, fixer, remove_water: bool) -> None:
        """Remove heterogens from protein structure."""
        fixer.removeHeterogens(keepWater=not remove_water)
    
    def _find_missing_residues_stage(self, fixer) -> None:
        """Find missing residues in protein structure."""
        fixer.findMissingResidues()
    
    def _add_missing_atoms_stage(self, fixer) -> None:
        """Find and add missing heavy atoms to protein structure."""
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
    
    def _add_missing_hydrogens_stage(self, fixer, ph: float) -> None:
        """Add missing hydrogens to protein structure."""
        fixer.addMissingHydrogens(ph)
    
    def _fixer_to_pdb_string(self, fixer) -> str:
        """Convert PDBFixer instance to PDB format string."""
        output = io.StringIO()
        # Preserve deposited chain IDs and residue numbers. OpenMM otherwise
        # rewrites them to A/B/... and 1..N, which breaks typed chain selection
        # and residue-level repair provenance downstream.
        PDBFile.writeFile(fixer.topology, fixer.positions, output, keepIds=True)
        return output.getvalue()

    @staticmethod
    def _residue_label(residue) -> dict[str, Any]:
        chain = residue.chain
        chain_id = str(getattr(chain, "id", "") or "_")
        return {
            "chain": chain_id,
            "residue": str(getattr(residue, "name", "") or ""),
            "index": int(getattr(residue, "index", -1)),
        }

    @staticmethod
    def _atom_name(atom) -> str:
        return str(getattr(atom, "name", atom))

    @staticmethod
    def _atom_key(atom) -> tuple[str, str, str, str]:
        residue = atom.residue
        return (
            str(residue.chain.id or "_"),
            str(residue.id),
            str(residue.name),
            str(atom.name),
        )

    @staticmethod
    def _missing_residue_policy(
        fixer,
        *,
        skip_terminal_missing_residues: bool,
        max_internal_gap: int | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Apply HiQBind's conservative terminal/long-gap policy."""
        chains = list(fixer.topology.chains())
        sequences = {
            str(sequence.chainId): list(sequence.residues)
            for sequence in getattr(fixer, "sequences", [])
        }
        added: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        remove: list[tuple[int, int]] = []
        prior_missing = {str(chain.id): 0 for chain in chains}
        for key, residues in sorted(fixer.missingResidues.items()):
            chain_index, insertion_index = key
            chain_id = str(chains[chain_index].id)
            sequence = sequences.get(chain_id, [])
            terminal = bool(
                insertion_index == 0
                or (
                    sequence
                    and prior_missing[chain_id] + insertion_index + len(residues)
                    == len(sequence)
                )
            )
            reason = ""
            if terminal and skip_terminal_missing_residues:
                reason = "terminal"
            elif max_internal_gap is not None and len(residues) > int(max_internal_gap):
                reason = f"gap_exceeds_{int(max_internal_gap)}"
            item = {
                "chain_index": int(chain_index),
                "chain": chain_id or "_",
                "insertion_index": int(insertion_index),
                "residues": [str(residue) for residue in residues],
                "terminal": terminal,
            }
            if reason:
                item["reason"] = reason
                skipped.append(item)
                remove.append(key)
            else:
                added.append(item)
            prior_missing[chain_id] += len(residues)
        for key in remove:
            fixer.missingResidues.pop(key, None)
        return added, skipped

    def _refine_rebuilt_positions(
        self,
        fixer,
        *,
        original_atom_keys: set[tuple[str, str, str, str]],
        ligand_sdf_data: str | None = None,
    ) -> dict[str, Any]:
        """Minimize rebuilt content while restraining the experimental scaffold.

        Original heavy atoms outside residues adjacent to rebuilt content are
        frozen by setting their particle masses to zero. Rebuilt atoms,
        hydrogens, and neighboring residues may relax.
        """
        topology = fixer.topology
        positions = fixer.positions
        protein_atom_count = topology.getNumAtoms()
        ligand_atom_indices: set[int] = set()
        ligand_forcefield = None
        ligand_parameterization = "not_provided"
        if ligand_sdf_data:
            try:
                from rdkit import Chem
                from openff.toolkit import Molecule, Topology as OFFTopology
                from openmmforcefields.generators import SMIRNOFFTemplateGenerator

                molecule = Chem.MolFromMolBlock(
                    ligand_sdf_data,
                    removeHs=False,
                    sanitize=True,
                )
                if molecule is None or molecule.GetNumConformers() == 0:
                    raise ValueError("RDKit could not read a 3D ligand from SDF")
                off_molecule = Molecule.from_rdkit(
                    molecule,
                    allow_undefined_stereo=True,
                    hydrogens_are_explicit=True,
                )
                ligand_topology = OFFTopology.from_molecules(off_molecule).to_openmm()
                conformer = molecule.GetConformer()
                ligand_positions = (
                    [
                        mm.Vec3(
                            conformer.GetAtomPosition(index).x,
                            conformer.GetAtomPosition(index).y,
                            conformer.GetAtomPosition(index).z,
                        )
                        for index in range(molecule.GetNumAtoms())
                    ]
                    * unit.angstrom
                )
                modeller = Modeller(topology, positions)
                modeller.add(ligand_topology, ligand_positions)
                topology = modeller.topology
                positions = modeller.positions
                ligand_atom_indices = set(range(protein_atom_count, topology.getNumAtoms()))
                ligand_forcefield = SMIRNOFFTemplateGenerator(
                    molecules=[off_molecule],
                ).generator
                ligand_parameterization = "SMIRNOFF"
            except Exception as exc:
                raise RuntimeError(f"Ligand-aware refinement setup failed: {exc}") from exc
        residues = list(topology.residues())
        rebuilt_residue_indices = {
            atom.residue.index
            for atom in topology.atoms()
            if atom.index not in ligand_atom_indices
            and self._atom_key(atom) not in original_atom_keys
            and getattr(atom.element, "symbol", "") != "H"
        }
        # MODELLER can complete atoms before PDBFixer sees the structure. Such
        # atoms then look "original" here, even when a newly built side chain
        # overlaps another residue. Detect severe non-bonded overlaps directly
        # and relax both residues instead of freezing an invalid conformation.
        bonded_pairs = {
            frozenset((left.index, right.index)) for left, right in topology.bonds()
        }
        protein_heavy_atoms = [
            atom
            for atom in topology.atoms()
            if atom.index not in ligand_atom_indices
            and getattr(atom.element, "symbol", "") != "H"
        ]
        sidechain_clashing_residue_indices: set[int] = set()
        backbone_clashing_residue_indices: set[int] = set()
        clash_examples: list[dict[str, Any]] = []
        backbone_names = {"N", "CA", "C", "O", "OXT"}
        for index, left in enumerate(protein_heavy_atoms):
            left_position = positions[left.index].value_in_unit(unit.angstrom)
            for right in protein_heavy_atoms[index + 1 :]:
                if left.residue.index == right.residue.index:
                    continue
                right_position = positions[right.index].value_in_unit(unit.angstrom)
                separation = float(
                    (
                        (left_position.x - right_position.x) ** 2
                        + (left_position.y - right_position.y) ** 2
                        + (left_position.z - right_position.z) ** 2
                    )
                    ** 0.5
                )
                bonded = frozenset((left.index, right.index)) in bonded_pairs
                malformed_peptide = bool(
                    bonded
                    and {left.name, right.name} == {"C", "N"}
                    and not 1.1 <= separation <= 1.6
                )
                if bonded and not malformed_peptide:
                    continue
                if separation >= 1.8 and not malformed_peptide:
                    continue
                affected = {left.residue.index, right.residue.index}
                if malformed_peptide or left.name in backbone_names or right.name in backbone_names:
                    backbone_clashing_residue_indices.update(affected)
                else:
                    sidechain_clashing_residue_indices.update(affected)
                if len(clash_examples) < 20:
                    clash_examples.append(
                        {
                            "distance_angstrom": separation,
                            "left": {
                                **self._residue_label(left.residue),
                                "atom": left.name,
                            },
                            "right": {
                                **self._residue_label(right.residue),
                                "atom": right.name,
                            },
                        }
                    )

        rebuilt_relax_residue_indices = set(rebuilt_residue_indices)
        for residue_index in tuple(rebuilt_residue_indices):
            residue = residues[residue_index]
            for neighbor_index in (residue_index - 1, residue_index + 1):
                if (
                    0 <= neighbor_index < len(residues)
                    and residues[neighbor_index].chain.index == residue.chain.index
                ):
                    rebuilt_relax_residue_indices.add(neighbor_index)

        forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")
        if ligand_forcefield is not None:
            forcefield.registerTemplateGenerator(ligand_forcefield)
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=CutoffNonPeriodic,
            constraints=None,
            rigidWater=False,
        )
        frozen_atoms = 0
        movable_atoms = 0
        frozen_atom_indices: set[int] = set()
        for atom in topology.atoms():
            is_hydrogen = getattr(atom.element, "symbol", "") == "H"
            sidechain_clash_atom = bool(
                atom.residue.index in sidechain_clashing_residue_indices
                and atom.name not in backbone_names
            )
            if atom.index in ligand_atom_indices:
                system.setParticleMass(atom.index, 0.0)
                frozen_atoms += 1
                frozen_atom_indices.add(atom.index)
            elif (
                is_hydrogen
                or atom.residue.index in rebuilt_relax_residue_indices
                or atom.residue.index in backbone_clashing_residue_indices
                or sidechain_clash_atom
            ):
                movable_atoms += 1
            else:
                system.setParticleMass(atom.index, 0.0)
                frozen_atoms += 1
                frozen_atom_indices.add(atom.index)

        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin,
            10 / unit.picosecond,
            1 * unit.femtosecond,
        )
        simulation = Simulation(topology, system, integrator)
        simulation.context.setPositions(positions)
        before = simulation.context.getState(getEnergy=True)
        mm.LocalEnergyMinimizer.minimize(
            simulation.context,
            tolerance=10 * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=1000,
        )
        after = simulation.context.getState(getEnergy=True, getPositions=True)
        refined_positions = after.getPositions()
        fixer.positions = refined_positions[:protein_atom_count]

        def displacement_summary(atom_indices: list[int]) -> dict[str, float | int]:
            displacements: list[float] = []
            for atom_index in atom_indices:
                before_position = positions[atom_index].value_in_unit(unit.angstrom)
                after_position = refined_positions[atom_index].value_in_unit(unit.angstrom)
                displacements.append(
                    float(
                        (
                            (before_position.x - after_position.x) ** 2
                            + (before_position.y - after_position.y) ** 2
                            + (before_position.z - after_position.z) ** 2
                        )
                        ** 0.5
                    )
                )
            if not displacements:
                return {"atom_count": 0, "rmsd_angstrom": 0.0, "max_angstrom": 0.0}
            return {
                "atom_count": len(displacements),
                "rmsd_angstrom": float(
                    (sum(value * value for value in displacements) / len(displacements))
                    ** 0.5
                ),
                "max_angstrom": max(displacements),
            }

        protein_atoms = list(topology.atoms())[:protein_atom_count]
        heavy_atom_indices = [
            atom.index
            for atom in protein_atoms
            if getattr(atom.element, "symbol", "") != "H"
        ]
        backbone_atom_indices = [
            atom.index for atom in protein_atoms if atom.name in backbone_names
        ]
        frozen_heavy_atom_indices = [
            atom.index
            for atom in protein_atoms
            if atom.index in frozen_atom_indices
            and getattr(atom.element, "symbol", "") != "H"
        ]
        displacement = {
            "protein_heavy_atoms": displacement_summary(heavy_atom_indices),
            "protein_backbone_atoms": displacement_summary(backbone_atom_indices),
            "frozen_scaffold_heavy_atoms": displacement_summary(
                frozen_heavy_atom_indices
            ),
        }
        if displacement["frozen_scaffold_heavy_atoms"]["max_angstrom"] > 0.01:
            raise RuntimeError(
                "Restrained target repair moved the frozen experimental scaffold by "
                f"{displacement['frozen_scaffold_heavy_atoms']['max_angstrom']:.3f} Å"
            )
        if displacement["protein_backbone_atoms"]["rmsd_angstrom"] > 0.25:
            raise RuntimeError(
                "Restrained target repair exceeded the protein-backbone RMSD limit: "
                f"{displacement['protein_backbone_atoms']['rmsd_angstrom']:.3f} Å"
            )
        return {
            "engine": "OpenMM LocalEnergyMinimizer",
            "platform": simulation.context.getPlatform().getName(),
            "forcefield": ["amber14-all.xml", "amber14/tip3p.xml"],
            "restraint_strategy": "freeze_original_heavy_atoms_except_rebuilt-neighbor_residues",
            "rebuilt_residue_count": len(rebuilt_residue_indices),
            "clashing_residue_count": len(
                sidechain_clashing_residue_indices
                | backbone_clashing_residue_indices
            ),
            "clashing_residue_indices": sorted(
                sidechain_clashing_residue_indices
                | backbone_clashing_residue_indices
            ),
            "sidechain_clashing_residue_indices": sorted(
                sidechain_clashing_residue_indices
            ),
            "backbone_clashing_residue_indices": sorted(
                backbone_clashing_residue_indices
            ),
            "pre_minimization_clashes": clash_examples,
            "movable_atom_count": movable_atoms,
            "frozen_atom_count": frozen_atoms,
            "displacement": displacement,
            "ligand_context": bool(ligand_atom_indices),
            "ligand_atom_count": len(ligand_atom_indices),
            "ligand_parameterization": ligand_parameterization,
            "potential_energy_before_kj_mol": float(
                before.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            ),
            "potential_energy_after_kj_mol": float(
                after.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            ),
            "max_iterations": 1000,
            "tolerance_kj_mol_nm": 10.0,
        }

    def repair_imported_structure(
        self,
        pdb_data: str,
        *,
        ph: float = 7.4,
        add_missing_residues: bool = True,
        skip_terminal_missing_residues: bool = True,
        max_internal_gap: int | None = 15,
        refine_rebuilt_positions: bool = True,
        ligand_sdf_data: str | None = None,
    ) -> Dict[str, Any]:
        """Run one strict PDBFixer pass while sequence records are still available."""
        if not PDBFIXER_AVAILABLE:
            raise ImportError("PDBFixer not available. Please install it for protein cleaning.")

        normalized_input, removed_internal_oxt = remove_internal_oxt(pdb_data)
        fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(normalized_input))
        original_atom_keys = {self._atom_key(atom) for atom in fixer.topology.atoms()}
        fixer.findMissingResidues()
        detected_missing_residues = [
            {
                "chain_index": int(chain_index),
                "insertion_index": int(insertion_index),
                "residues": [str(residue) for residue in residues],
            }
            for (chain_index, insertion_index), residues in sorted(fixer.missingResidues.items())
        ]
        if not add_missing_residues:
            fixer.missingResidues = {}
            added_missing_residues: list[dict[str, Any]] = []
            skipped_missing_residues = [
                {**segment, "reason": "missing_residue_repair_disabled"}
                for segment in detected_missing_residues
            ]
        else:
            added_missing_residues, skipped_missing_residues = self._missing_residue_policy(
                fixer,
                skip_terminal_missing_residues=skip_terminal_missing_residues,
                max_internal_gap=max_internal_gap,
            )

        fixer.findNonstandardResidues()
        nonstandard = [
            {
                **self._residue_label(residue),
                "replacement": str(replacement),
            }
            for residue, replacement in fixer.nonstandardResidues
        ]
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        missing_atoms = [
            {
                **self._residue_label(residue),
                "atoms": [self._atom_name(atom) for atom in atoms],
            }
            for residue, atoms in fixer.missingAtoms.items()
        ]
        missing_terminals = [
            {
                **self._residue_label(residue),
                "atoms": [self._atom_name(atom) for atom in atoms],
            }
            for residue, atoms in fixer.missingTerminals.items()
        ]
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(float(ph))
        refinement = (
            self._refine_rebuilt_positions(
                fixer,
                original_atom_keys=original_atom_keys,
                ligand_sdf_data=ligand_sdf_data,
            )
            if refine_rebuilt_positions
            else {"enabled": False}
        )
        repaired = self._fixer_to_pdb_string(fixer)
        repaired, target_validation = prepare_target_for_publication(repaired)
        return {
            "pdb_data": repaired,
            "report": {
                "engine": "PDBFixer",
                "strict": True,
                "ph": float(ph),
                "add_missing_residues": bool(add_missing_residues),
                "skip_terminal_missing_residues": bool(skip_terminal_missing_residues),
                "max_internal_gap": max_internal_gap,
                "refine_rebuilt_positions": bool(refine_rebuilt_positions),
                "nonstandard_residues": nonstandard,
                "missing_residue_segments": detected_missing_residues,
                "missing_residue_segments_added": added_missing_residues,
                "missing_residue_segments_skipped": skipped_missing_residues,
                "missing_atoms": missing_atoms,
                "missing_terminal_atoms": missing_terminals,
                "refinement": refinement,
                "target_validation": {
                    **target_validation,
                    "removed_pre_pdbfixer_internal_oxt_count": len(
                        removed_internal_oxt
                    ),
                    "removed_pre_pdbfixer_internal_oxt_atoms": removed_internal_oxt,
                },
            },
        }
    
    def _add_solvation_to_pdb(self, pdb_data: str, box_size: float = 10.0, box_shape: str = 'cubic') -> str:
        """
        Add solvation box to protein structure using OpenMM Modeller.

        Args:
            pdb_data: PDB format data as string
            box_size: Padding distance in Angstroms for the solvation box
            box_shape: Shape of the solvent box ('cubic' or 'octahedral')

        Returns:
            PDB format string with solvation added
        """
        try:
            from openmm.app import Modeller, PDBFile, forcefield
            from openmm import unit
            import tempfile
            import os

            # Map user-facing shape names to OpenMM boxShape values
            shape_map = {
                'cubic': 'cube',
                'octahedral': 'octahedron',
            }
            omm_box_shape = shape_map.get(box_shape, 'cube')

            # Write input PDB to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_input:
                tmp_input.write(pdb_data)
                tmp_input_path = tmp_input.name

            try:
                # Read PDB file using PDBFile
                pdb_file = PDBFile(tmp_input_path)

                # Create Modeller from the PDB file
                modeller = Modeller(pdb_file.topology, pdb_file.positions)

                # Add solvent (water box)
                padding_nm = box_size * 0.1  # Convert Angstroms to nanometers

                # Load a standard forcefield for solvation
                try:
                    ff = forcefield.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
                except:
                    try:
                        ff = forcefield.ForceField('charmm36.xml', 'charmm36/water.xml')
                    except:
                        ff = forcefield.ForceField('amber14-all.xml')

                # Add solvent with specified padding and box shape
                modeller.addSolvent(ff, padding=padding_nm * unit.nanometer, boxShape=omm_box_shape)
                
                logger.info(f"Added solvation box with {modeller.topology.getNumAtoms()} total atoms")
                
                # Write solvated structure to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_output:
                    tmp_output_path = tmp_output.name
                
                # Write PDB file
                with open(tmp_output_path, 'w') as f:
                    PDBFile.writeFile(modeller.topology, modeller.positions, f)
                
                # Read the solvated PDB back as string
                with open(tmp_output_path, 'r') as f:
                    solvated_pdb = f.read()
                
                # Clean up temporary files
                os.unlink(tmp_input_path)
                os.unlink(tmp_output_path)
                
                return solvated_pdb
                
            except Exception as e:
                # Clean up on error
                if os.path.exists(tmp_input_path):
                    os.unlink(tmp_input_path)
                if 'tmp_output_path' in locals() and os.path.exists(tmp_output_path):
                    os.unlink(tmp_output_path)
                raise
            
        except ImportError:
            raise ImportError("OpenMM not available. Please install it for solvation.")
        except Exception as e:
            logger.error(f"Error adding solvation: {str(e)}")
            raise
    
    def clean_structure_staged(self, pdb_data: str,
                               remove_heterogens: bool = True,
                               remove_water: bool = True,
                               add_missing_residues: bool = True,
                               add_missing_atoms: bool = True,
                               add_missing_hydrogens: bool = True,
                               ph: float = 7.4,
                               add_solvation: bool = False,
                               solvation_box_size: float = 10.0,
                               solvation_box_shape: str = 'cubic',
                               keep_ligands: bool = False) -> Dict[str, Any]:
        """
        Clean protein structure with step-by-step control, returning all intermediate stages.
        
        Args:
            pdb_data: PDB format data as string
            remove_heterogens: Whether to remove heterogens
            remove_water: Whether to remove water molecules
            add_missing_residues: Whether to find missing residues
            add_missing_atoms: Whether to add missing heavy atoms
            add_missing_hydrogens: Whether to add missing hydrogens
            ph: pH for protonation state
            add_solvation: Whether to add solvation box
            solvation_box_size: Padding distance in Angstroms for solvation box
            keep_ligands: Whether to extract and reinsert ligands after cleaning
            
        Returns:
            Dictionary with stages and metadata
        """
        if not PDBFIXER_AVAILABLE:
            raise ImportError("PDBFixer not available. Please install it for protein cleaning.")
        
        logger.info("Cleaning protein structure with PDBFixer (staged)")
        
        stages = {}
        stage_info = {}
        extracted_ligands = {}
        
        try:
            # Extract ligands before cleaning if keep_ligands is True
            if keep_ligands:
                try:
                    from mn_ligand.ligandx.lib.chemistry.parsers.pdb import get_pdb_parser
                    from mn_ligand.ligandx.lib.chemistry.analysis.components import get_component_analyzer
                    
                    parser = get_pdb_parser()
                    analyzer = get_component_analyzer()
                    
                    structure = parser.parse_string(pdb_data, "structure")
                    components = analyzer.identify_components(structure)
                    ligand_residues = components.get("ligands", [])
                    
                    if ligand_residues:
                        from mn_ligand.ligandx.services.structure.processor import StructureProcessor
                        processor = StructureProcessor()
                        extracted_ligands = processor.extract_ligands(structure, ligand_residues)
                        logger.info(f"Extracted {len(extracted_ligands)} ligand(s) for preservation")
                except Exception as e:
                    logger.warning(f"Failed to extract ligands: {e}. Continuing without ligand preservation.")
                    keep_ligands = False
            
            # Stage 0: Original
            stages['original'] = pdb_data
            stage_info['original'] = {'description': 'Original structure', 'step': 0}
            
            current_pdb = pdb_data
            
            # Stage 1: After removing heterogens
            if remove_heterogens:
                pdb_io = io.StringIO(current_pdb)
                fixer = pdbfixer.PDBFixer(pdbfile=pdb_io)
                self._remove_heterogens_stage(fixer, remove_water=False)
                current_pdb = self._fixer_to_pdb_string(fixer)
                stages['after_heterogens'] = current_pdb
                stage_info['after_heterogens'] = {'description': 'After removing heterogens', 'step': 1}
            
            # Stage 2: After removing water
            if remove_water:
                pdb_io = io.StringIO(current_pdb)
                fixer = pdbfixer.PDBFixer(pdbfile=pdb_io)
                fixer.removeHeterogens(keepWater=False)
                current_pdb = self._fixer_to_pdb_string(fixer)
                stages['after_water'] = current_pdb
                stage_info['after_water'] = {'description': 'After removing water', 'step': 2}
            
            # Stage 3: After finding missing residues and adding missing atoms
            if add_missing_atoms:
                pdb_io = io.StringIO(current_pdb)
                fixer = pdbfixer.PDBFixer(pdbfile=pdb_io)
                if add_missing_residues:
                    self._find_missing_residues_stage(fixer)
                self._add_missing_atoms_stage(fixer)
                current_pdb = self._fixer_to_pdb_string(fixer)
                stage_description = 'After finding missing residues and adding missing atoms' if add_missing_residues else 'After adding missing atoms'
                stages['after_missing_atoms'] = current_pdb
                stage_info['after_missing_atoms'] = {'description': stage_description, 'step': 3}
            
            # Stage 4: After adding hydrogens
            if add_missing_hydrogens:
                pdb_io = io.StringIO(current_pdb)
                fixer = pdbfixer.PDBFixer(pdbfile=pdb_io)
                if add_missing_residues:
                    self._find_missing_residues_stage(fixer)
                if add_missing_atoms:
                    self._add_missing_atoms_stage(fixer)
                self._add_missing_hydrogens_stage(fixer, ph)
                current_pdb = self._fixer_to_pdb_string(fixer)
                stages['after_hydrogens'] = current_pdb
                stage_info['after_hydrogens'] = {'description': 'After adding missing hydrogens', 'step': 4}
            
            # Stage 5: After adding solvation
            if add_solvation:
                current_pdb = self._add_solvation_to_pdb(current_pdb, solvation_box_size, solvation_box_shape)
                stages['after_solvation'] = current_pdb
                stage_info['after_solvation'] = {'description': 'After adding solvation', 'step': 5}
            
            # Reinsert ligands if they were extracted
            if keep_ligands and extracted_ligands:
                try:
                    from mn_ligand.ligandx.services.structure.processor import StructureProcessor
                    processor = StructureProcessor()
                    current_pdb = processor.reinsert_ligands(current_pdb, extracted_ligands)
                    stages['final_with_ligands'] = current_pdb
                    stage_info['final_with_ligands'] = {
                        'description': 'Final structure with reinserted ligands',
                        'step': 6
                    }
                    logger.info(f"Reinserted {len(extracted_ligands)} ligand(s)")
                except Exception as e:
                    logger.warning(f"Failed to reinsert ligands: {e}")
            
            logger.info(f"Protein cleaning completed successfully. Generated {len(stages)} stages.")
            result = {'stages': stages, 'stage_info': stage_info}
            if keep_ligands and extracted_ligands:
                result['ligands'] = extracted_ligands
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning protein structure (staged): {str(e)}")
            raise


# Singleton instance
_protein_preparer_instance = None


def get_protein_preparer() -> ProteinPreparer:
    """Get or create ProteinPreparer singleton instance."""
    global _protein_preparer_instance
    if _protein_preparer_instance is None:
        _protein_preparer_instance = ProteinPreparer()
    return _protein_preparer_instance
