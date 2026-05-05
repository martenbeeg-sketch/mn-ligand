"""
Protein structure preparation utilities.

Provides functionality for cleaning and preparing protein structures
using PDBFixer and OpenMM.
"""

import io
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Optional imports with availability flags
try:
    import pdbfixer
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
        PDBFile.writeFile(fixer.topology, fixer.positions, output)
        return output.getvalue()
    
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
                    from ovo_ligand.ligandx.lib.chemistry.parsers.pdb import get_pdb_parser
                    from ovo_ligand.ligandx.lib.chemistry.analysis.components import get_component_analyzer
                    
                    parser = get_pdb_parser()
                    analyzer = get_component_analyzer()
                    
                    structure = parser.parse_string(pdb_data, "structure")
                    components = analyzer.identify_components(structure)
                    ligand_residues = components.get("ligands", [])
                    
                    if ligand_residues:
                        from ovo_ligand.ligandx.services.structure.processor import StructureProcessor
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
                    from ovo_ligand.ligandx.services.structure.processor import StructureProcessor
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
