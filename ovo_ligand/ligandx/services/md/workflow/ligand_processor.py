"""
Ligand processor module for MD optimization.

Handles ligand preparation from SMILES and structure data with OpenFF.
"""

import logging
from typing import Dict, Any, Optional
import traceback
import urllib.request
import tempfile
import os

logger = logging.getLogger(__name__)


class LigandProcessor:
    """Processes ligands for MD simulation using OpenFF."""
    
    def __init__(self, environment_status: Dict[str, Any]):
        """
        Initialize ligand processor.
        
        Args:
            environment_status: Environment validation status dict
        """
        self.environment_status = environment_status
    
    def assign_partial_charges(self, molecule, method: str = "mmff94"):
        """
        Assign partial charges using the specified method.

        Args:
            molecule: OpenFF Molecule object
            method: Charge calculation method ('mmff94', 'gasteiger', 'am1bcc', 'orca')

        Returns:
            OpenFF Molecule with assigned charges, or None if method fails
        """
        if not molecule:
            return None

        method = method.lower()
        logger.info(f"Assigning partial charges using method: {method}")

        try:
            if method == "mmff94":
                return self._assign_charges_mmff94(molecule)
            elif method == "gasteiger":
                return self._assign_charges_gasteiger(molecule)
            elif method == "am1bcc":
                return self._assign_charges_am1bcc(molecule)
            elif method == "orca":
                return self._assign_charges_orca(molecule)
            else:
                logger.error(f"Unknown charge method: {method}")
                return None
        except Exception as e:
            logger.error(f"Charge assignment with method '{method}' failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _assign_charges_mmff94(self, molecule):
        """Assign charges using MMFF94 method."""
        logger.info("Attempting MMFF94 charge assignment...")
        molecule.assign_partial_charges(partial_charge_method="mmff94")
        logger.info("[COMPLETE] MMFF94 charges assigned successfully")
        return molecule

    def _assign_charges_gasteiger(self, molecule):
        """Assign charges using Gasteiger method."""
        logger.info("Attempting Gasteiger charge assignment...")
        molecule.assign_partial_charges(partial_charge_method="gasteiger")
        logger.info("[COMPLETE] Gasteiger charges assigned successfully")
        return molecule

    def _assign_charges_am1bcc(self, molecule):
        """
        Assign charges using AM1-BCC method via AmberTools antechamber.

        Uses OpenFE's bulk_assign_partial_charges for robust handling,
        which explicitly routes to AmberTools toolkit backend.

        bulk_assign_partial_charges expects OpenFE SmallMoleculeComponent objects
        (which have .to_openff()), not raw OpenFF Molecule objects. We convert
        to SmallMoleculeComponent first, then extract the charged OpenFF Molecule.

        This is the same proven pattern used in ABFE/RBFE services.
        """
        logger.info("Attempting AM1-BCC charge assignment via AmberTools...")

        try:
            import openfe
            from openfe.protocols.openmm_utils.omm_settings import OpenFFPartialChargeSettings
            from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges

            # Create charge settings with explicit ambertools backend
            charge_settings = OpenFFPartialChargeSettings(
                partial_charge_method="am1bcc",
                off_toolkit_backend="ambertools"
            )

            # Convert OpenFF Molecule → RDKit → OpenFE SmallMoleculeComponent
            # bulk_assign_partial_charges expects SmallMoleculeComponent (has .to_openff())
            rdkit_mol = molecule.to_rdkit()
            ligand_component = openfe.SmallMoleculeComponent.from_rdkit(rdkit_mol)

            # Use OpenFE's bulk charge assignment (same as ABFE/RBFE)
            charged_components = bulk_assign_partial_charges(
                molecules=[ligand_component],
                overwrite=False,
                method=charge_settings.partial_charge_method,
                toolkit_backend=charge_settings.off_toolkit_backend,
                generate_n_conformers=charge_settings.number_of_conformers,
                nagl_model=charge_settings.nagl_model,
                processors=1
            )

            if not charged_components or len(charged_components) == 0:
                raise ValueError("AM1-BCC charge assignment returned no molecules")

            # Extract charged OpenFF Molecule from the SmallMoleculeComponent
            charged_molecule = charged_components[0].to_openff()
            logger.info("[COMPLETE] AM1-BCC charges assigned successfully via AmberTools")
            return charged_molecule

        except ImportError as e:
            logger.error(f"OpenFE charge_generation module not available: {e}")
            logger.error("Make sure OpenFE is installed in the conda environment")
            raise
        except Exception as e:
            logger.error(f"AM1-BCC charge assignment failed: {e}")
            logger.error("Make sure AmberTools is installed and antechamber is available")
            raise

    def _assign_charges_orca(self, molecule):
        """
        Assign charges using ORCA quantum chemistry calculations.

        This method uses the QC service to calculate charges via DFT.
        Significantly slower but handles exotic chemistry.
        """
        logger.info("Attempting ORCA quantum chemical charge assignment...")
        logger.warning("ORCA charge calculation may take 5-10 minutes")

        try:
            # Convert molecule to XYZ format for ORCA
            from rdkit import Chem
            rdkit_mol = molecule.to_rdkit()

            # Get geometry in XYZ format
            xyz_block = self._rdkit_to_xyz(rdkit_mol)

            # Call QC service for charge calculation
            # TODO: Implement QC service integration
            # For now, raise NotImplementedError
            raise NotImplementedError(
                "ORCA charge calculation requires QC service integration. "
                "This feature is planned for future release. "
                "Try AM1-BCC or Gasteiger methods instead."
            )

        except NotImplementedError:
            raise
        except Exception as e:
            logger.error(f"ORCA charge assignment failed: {e}")
            raise

    def _rdkit_to_xyz(self, rdkit_mol):
        """Convert RDKit molecule to XYZ format string."""
        from rdkit import Chem

        conf = rdkit_mol.GetConformer()
        num_atoms = rdkit_mol.GetNumAtoms()

        xyz_lines = [str(num_atoms), ""]

        for i, atom in enumerate(rdkit_mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

        return "\n".join(xyz_lines)
    
    def prepare_ligand_from_smiles(
        self,
        smiles: str,
        ligand_id: str = "ligand",
        generate_conformer: bool = True,
        charge_method: str = "mmff94"
    ) -> Dict[str, Any]:
        """
        Prepare ligand from SMILES string.

        Args:
            smiles: SMILES string of the ligand
            ligand_id: Identifier for the ligand
            generate_conformer: Whether to generate a new 3D conformer
            charge_method: Charge calculation method ('mmff94', 'gasteiger', 'am1bcc', 'orca')

        Returns:
            Dict with 'success', 'molecule' (OpenFF Molecule), 'error'
        """
        logger.info(f"=== PREPARING LIGAND FROM SMILES ===")
        logger.info(f"SMILES: {smiles}")
        logger.info(f"Ligand ID: {ligand_id}")
        
        if not self.environment_status.get('rdkit', False):
            return {"success": False, "error": "RDKit not available", "molecule": None}
        
        if not self.environment_status.get('openff', False):
            return {"success": False, "error": "OpenFF Toolkit not available", "molecule": None}
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from openff.toolkit import Molecule
            
            # Step 1: Parse SMILES
            logger.info("Step 1: Parsing SMILES with RDKit")
            rdkit_mol = Chem.MolFromSmiles(smiles)
            if rdkit_mol is None:
                return {"success": False, "error": f"Invalid SMILES: {smiles}", "molecule": None}
            logger.info(f"[COMPLETE] Parsed: {rdkit_mol.GetNumAtoms()} heavy atoms")
            
            # Step 2: Add hydrogens
            logger.info("Step 2: Adding hydrogens")
            rdkit_mol = Chem.AddHs(rdkit_mol)
            rdkit_mol.SetProp("_Name", ligand_id)
            logger.info(f"[COMPLETE] Total atoms: {rdkit_mol.GetNumAtoms()}")
            
            # Step 3: Generate 3D conformer
            if generate_conformer:
                logger.info("Step 3: Generating 3D conformer (ETKDGv3)")
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                result = AllChem.EmbedMolecule(rdkit_mol, params)
                if result == 0:
                    logger.info("[COMPLETE] ETKDGv3 conformer generation successful")
                else:
                    logger.warning("ETKDGv3 failed, using basic embedding")
                    AllChem.EmbedMolecule(rdkit_mol, useRandomCoords=True)
                
                # Step 4: Optimize geometry
                logger.info("Step 4: Optimizing geometry")
                AllChem.MMFFOptimizeMolecule(rdkit_mol)
                logger.info("[COMPLETE] MMFF94 optimization complete")
            
            # Step 5: Create OpenFF Molecule with stereochemistry handling
            logger.info("Step 5: Creating OpenFF Molecule")
            molecule = self._create_openff_molecule(rdkit_mol)
            
            if molecule is None:
                return {"success": False, "error": "Failed to create OpenFF molecule", "molecule": None}
            
            # Step 6: Assign partial charges
            logger.info(f"Step 6: Assigning partial charges using {charge_method} method...")
            try:
                charged_molecule = self.assign_partial_charges(molecule, method=charge_method)
            except NotImplementedError as e:
                logger.error(f"Charge method '{charge_method}' not implemented: {e}")
                return {
                    "success": False,
                    "error": f"Charge calculation method '{charge_method}' not yet implemented. Use 'am1bcc', 'mmff94', or 'gasteiger' instead.",
                    "molecule": None
                }

            if not charged_molecule:
                return {"success": False, "error": f"Charge assignment failed with method: {charge_method}", "molecule": None}

            logger.info("[COMPLETE] Ligand preparation from SMILES completed successfully")
            return {"success": True, "molecule": charged_molecule, "error": None}

        except Exception as e:
            logger.error(f"Ligand preparation from SMILES failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e), "molecule": None}
    
    def prepare_ligand_from_structure(
        self,
        structure_data: str,
        ligand_id: str = "ligand",
        data_format: str = "sdf",
        preserve_pose: bool = True,
        charge_method: str = "mmff94"
    ) -> Dict[str, Any]:
        """
        Prepare ligand from structure data (SDF/MOL/PDB format).

        Args:
            structure_data: Structure data string
            ligand_id: Ligand identifier
            data_format: Format ('sdf', 'mol', 'pdb')
            preserve_pose: Whether to preserve original 3D pose
            charge_method: Charge calculation method ('mmff94', 'gasteiger', 'am1bcc', 'orca')

        Returns:
            Dict with 'success', 'molecule' (OpenFF Molecule), 'error'
        """
        logger.info(f"=== PREPARING LIGAND FROM STRUCTURE ===")
        logger.info(f"Ligand ID: {ligand_id}")
        logger.info(f"Data format: {data_format}")
        logger.info(f"Preserve pose: {preserve_pose}")
        
        if not self.environment_status.get('rdkit', False):
            return {"success": False, "error": "RDKit not available", "molecule": None}
        
        if not self.environment_status.get('openff', False):
            return {"success": False, "error": "OpenFF Toolkit not available", "molecule": None}
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from openff.toolkit import Molecule
            from ..utils.pdb_utils import sanitize_pdb_block

            # Notebook-parity path: for SDF input, first try OpenFF from_file() directly.
            if data_format.lower() == "sdf":
                direct_openff = self._load_openff_molecule_from_sdf(structure_data, ligand_id)
                if direct_openff is not None:
                    logger.info("Step 1: Parsed SDF via OpenFF Molecule.from_file()")
                    logger.info(f"Step 2: Assigning partial charges using {charge_method} method...")
                    try:
                        charged_molecule = self.assign_partial_charges(direct_openff, method=charge_method)
                    except NotImplementedError as e:
                        logger.error(f"Charge method '{charge_method}' not implemented: {e}")
                        return {
                            "success": False,
                            "error": f"Charge calculation method '{charge_method}' not yet implemented. Use 'am1bcc', 'mmff94', or 'gasteiger' instead.",
                            "molecule": None
                        }
                    if not charged_molecule:
                        return {"success": False, "error": f"Charge assignment failed with method: {charge_method}", "molecule": None}
                    logger.info("[COMPLETE] Ligand preparation from structure completed successfully (OpenFF SDF path)")
                    return {"success": True, "molecule": charged_molecule, "error": None}
            
            # Step 1: Parse structure data
            logger.info(f"Step 1: Parsing {data_format.upper()} structure data")
            
            if data_format.lower() == 'sdf' or data_format.lower() == 'mol':
                rdkit_mol = self._parse_ligand_molblock_with_fallbacks(
                    structure_data=structure_data,
                    data_format=data_format.lower(),
                )
            elif data_format.lower() == 'pdb':
                sanitized_pdb = sanitize_pdb_block(structure_data)
                rdkit_mol = Chem.MolFromPDBBlock(sanitized_pdb, removeHs=False)
                rdkit_mol = self._apply_ccd_template_bond_orders(rdkit_mol, sanitized_pdb)
            else:
                return {"success": False, "error": f"Unsupported format: {data_format}", "molecule": None}
            
            if rdkit_mol is None:
                return {"success": False, "error": f"Failed to parse {data_format}", "molecule": None}
            
            logger.info(f"[COMPLETE] Parsed: {rdkit_mol.GetNumAtoms()} atoms")
            
            # Step 2: Clean up hydrogens (remove orphans)
            rdkit_mol = self._clean_hydrogens(rdkit_mol)
            
            # Step 3: Check for valid 3D coordinates
            has_valid_3d_coords = self._check_3d_coords(rdkit_mol)
            
            # Step 4: Add hydrogens while preserving coordinates
            logger.info("Step 4: Adding hydrogens while preserving coordinates")
            rdkit_mol = Chem.AddHs(rdkit_mol, addCoords=True)
            rdkit_mol.SetProp("_Name", ligand_id)
            
            # Step 5: Handle 3D coordinates
            if preserve_pose and has_valid_3d_coords:
                logger.info("Step 5: Preserving original 3D pose")
            else:
                logger.info("Step 5: Generating new 3D conformer (ETKDGv3)")
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                result = AllChem.EmbedMolecule(rdkit_mol, params)
                if result != 0:
                    AllChem.EmbedMolecule(rdkit_mol, useRandomCoords=True)
                AllChem.MMFFOptimizeMolecule(rdkit_mol)
            
            # Step 6: Create OpenFF Molecule
            logger.info("Step 6: Creating OpenFF Molecule")
            molecule = self._create_openff_molecule(rdkit_mol, allow_undefined_stereo=True)
            
            if molecule is None:
                return {"success": False, "error": "Failed to create OpenFF molecule", "molecule": None}
            
            # Step 7: Assign partial charges
            logger.info(f"Step 7: Assigning partial charges using {charge_method} method...")
            try:
                charged_molecule = self.assign_partial_charges(molecule, method=charge_method)
            except NotImplementedError as e:
                logger.error(f"Charge method '{charge_method}' not implemented: {e}")
                return {
                    "success": False,
                    "error": f"Charge calculation method '{charge_method}' not yet implemented. Use 'am1bcc', 'mmff94', or 'gasteiger' instead.",
                    "molecule": None
                }

            if not charged_molecule:
                return {"success": False, "error": f"Charge assignment failed with method: {charge_method}", "molecule": None}

            logger.info("[COMPLETE] Ligand preparation from structure completed successfully")
            return {"success": True, "molecule": charged_molecule, "error": None}

        except Exception as e:
            logger.error(f"Ligand preparation from structure failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e), "molecule": None}

    def _load_openff_molecule_from_sdf(self, structure_data: str, ligand_id: str):
        """Load OpenFF molecule from SDF text via temporary file."""
        from openff.toolkit import Molecule

        text = structure_data or ""
        if not text.strip():
            return None

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".sdf")
            os.close(fd)
            with open(tmp_path, "w", encoding="utf-8") as handle:
                handle.write(text)
            molecule = Molecule.from_file(tmp_path, file_format="SDF", allow_undefined_stereo=True)
            if molecule is not None:
                molecule.name = ligand_id
            return molecule
        except Exception as exc:
            logger.warning("OpenFF SDF direct load failed: %s", exc)
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _parse_ligand_molblock_with_fallbacks(self, structure_data: str, data_format: str):
        """
        Parse ligand SDF/MOL text robustly.
        """
        from rdkit import Chem

        text = structure_data or ""
        if not text.strip():
            return None

        mol = Chem.MolFromMolBlock(text, removeHs=False)
        if mol is not None:
            return mol

        try:
            mol = Chem.MolFromMolBlock(text, removeHs=False, strictParsing=False)
            if mol is not None:
                logger.info("Recovered ligand parse using strictParsing=False")
                return mol
        except Exception:
            pass

        if "$$$$" in text:
            first_record = text.split("$$$$", 1)[0].rstrip() + "\n"
            try:
                mol = Chem.MolFromMolBlock(first_record, removeHs=False, strictParsing=False)
                if mol is not None:
                    logger.info("Recovered ligand parse from first SDF record")
                    return mol
            except Exception:
                pass

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".sdf")
            os.close(fd)
            with open(tmp_path, "w", encoding="utf-8") as handle:
                handle.write(text)
            supplier = Chem.SDMolSupplier(tmp_path, removeHs=False, sanitize=True)
            for candidate in supplier:
                if candidate is not None:
                    logger.info("Recovered ligand parse using SDMolSupplier fallback")
                    return candidate
        except Exception as exc:
            logger.warning("SDMolSupplier fallback failed: %s", exc)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        return None

    def _apply_ccd_template_bond_orders(self, rdkit_mol, pdb_block: str):
        """
        Recover ligand bond orders/aromaticity for PDB input by mapping against
        the RCSB CCD ideal SDF template for this residue name.
        """
        from rdkit.Chem import AllChem

        if rdkit_mol is None:
            return rdkit_mol

        resname = self._extract_resname_from_pdb_block(pdb_block)
        if not resname or len(resname) != 3:
            logger.info("No valid residue name found for CCD template lookup; using PDB-derived bonding")
            return rdkit_mol

        template = self._fetch_ccd_template_mol(resname)
        if template is None:
            logger.info("No CCD template available for %s; using PDB-derived bonding", resname)
            return rdkit_mol

        try:
            mapped = AllChem.AssignBondOrdersFromTemplate(template, rdkit_mol)
            self._log_template_mapping_qc(template, mapped, resname)
            logger.info("[COMPLETE] Applied CCD bond-order template for ligand residue %s", resname)
            return mapped
        except Exception as exc:
            logger.warning("CCD template mapping failed for %s: %s. Using PDB-derived bonding.", resname, exc)
            return rdkit_mol

    def _extract_resname_from_pdb_block(self, pdb_block: str) -> Optional[str]:
        for line in pdb_block.splitlines():
            if line.startswith(("HETATM", "ATOM")) and len(line) >= 20:
                resname = line[17:20].strip().upper()
                if resname:
                    return resname
        return None

    def _fetch_ccd_template_mol(self, resname: str):
        from rdkit import Chem

        urls = [
            f"https://files.rcsb.org/ligands/view/{resname}_ideal.sdf",
            f"https://files.rcsb.org/ligands/view/{resname}.sdf",
        ]
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=20) as response:
                    sdf = response.read().decode("utf-8", errors="ignore")
                mol = Chem.MolFromMolBlock(sdf, removeHs=False, sanitize=True)
                if mol is not None:
                    return mol
            except Exception:
                continue
        return None

    def _log_template_mapping_qc(self, template_mol, mapped_mol, resname: str) -> None:
        """Log chemistry-enforcement diagnostics after bond-order template transfer."""
        from rdkit import Chem

        try:
            th = Chem.RemoveHs(template_mol, sanitize=False)
            mh = Chem.RemoveHs(mapped_mol, sanitize=False)
            template_heavy = th.GetNumAtoms()
            mapped_heavy = mh.GetNumAtoms()
            template_aromatic_bonds = sum(1 for b in th.GetBonds() if b.GetIsAromatic())
            mapped_aromatic_bonds = sum(1 for b in mh.GetBonds() if b.GetIsAromatic())
            logger.info(
                "CCD mapping QC [%s]: heavy_atoms template=%d mapped=%d, aromatic_bonds template=%d mapped=%d",
                resname,
                template_heavy,
                mapped_heavy,
                template_aromatic_bonds,
                mapped_aromatic_bonds,
            )
            if template_heavy != mapped_heavy:
                logger.warning(
                    "CCD mapping QC [%s]: heavy atom count mismatch after template mapping (%d vs %d)",
                    resname,
                    template_heavy,
                    mapped_heavy,
                )
        except Exception as exc:
            logger.warning("CCD mapping QC logging failed for %s: %s", resname, exc)
    
    def _clean_hydrogens(self, rdkit_mol):
        """Remove orphan hydrogens from RDKit molecule."""
        from rdkit import Chem
        
        try:
            # First standard removal
            rdkit_mol = Chem.RemoveHs(rdkit_mol, implicitOnly=False)
            
            # Then manual removal of any remaining H atoms (orphans)
            rw_mol = Chem.RWMol(rdkit_mol)
            atoms_to_remove = []
            for atom in rw_mol.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    atoms_to_remove.append(atom.GetIdx())
            
            if atoms_to_remove:
                logger.info(f"Removing {len(atoms_to_remove)} orphan hydrogen atoms")
                for idx in sorted(atoms_to_remove, reverse=True):
                    rw_mol.RemoveAtom(idx)
                rdkit_mol = rw_mol.GetMol()
            
            logger.info(f"[COMPLETE] Cleaned structure: {rdkit_mol.GetNumAtoms()} heavy atoms")
            return rdkit_mol
            
        except Exception as e:
            logger.warning(f"Error during hydrogen removal: {e}")
            return rdkit_mol
    
    def _check_3d_coords(self, rdkit_mol) -> bool:
        """Check if molecule has valid 3D coordinates."""
        if rdkit_mol.GetNumConformers() > 0:
            conf = rdkit_mol.GetConformer(0)
            if conf.Is3D():
                logger.info("[COMPLETE] Found valid 3D coordinates in input structure")
                return True
        return False
    
    def _create_openff_molecule(self, rdkit_mol, allow_undefined_stereo: bool = False):
        """Create OpenFF Molecule from RDKit molecule with stereochemistry handling."""
        from rdkit import Chem
        from openff.toolkit import Molecule
        
        try:
            # Try with allow_undefined_stereo first
            if allow_undefined_stereo:
                try:
                    molecule = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)
                    logger.info(f"[COMPLETE] OpenFF molecule created (allowing undefined stereo): {molecule.n_atoms} atoms")
                    return molecule
                except Exception:
                    pass
            
            # Try standard creation
            try:
                molecule = Molecule.from_rdkit(rdkit_mol)
                logger.info(f"[COMPLETE] OpenFF molecule created: {molecule.n_atoms} atoms")
                return molecule
            except Exception as e:
                if "stereochemistry" not in str(e).lower():
                    raise
                
                # Handle stereochemistry issues
                logger.warning(f"Stereochemistry issue detected: {e}")
                logger.info("Attempting stereochemistry assignment...")
                
                # Try to assign stereochemistry
                mol_copy = Chem.Mol(rdkit_mol)
                Chem.AssignStereochemistry(mol_copy, cleanIt=True, force=True)
                
                try:
                    molecule = Molecule.from_rdkit(mol_copy, allow_undefined_stereo=True)
                    logger.info(f"[COMPLETE] OpenFF molecule created with assigned stereo: {molecule.n_atoms} atoms")
                    return molecule
                except Exception:
                    # Final fallback: remove all stereochemistry
                    Chem.RemoveStereochemistry(mol_copy)
                    molecule = Molecule.from_rdkit(mol_copy, allow_undefined_stereo=True)
                    logger.warning("⚠ Created OpenFF molecule with stereochemistry removed")
                    return molecule
                    
        except Exception as e:
            logger.error(f"Failed to create OpenFF molecule: {e}")
            return None
