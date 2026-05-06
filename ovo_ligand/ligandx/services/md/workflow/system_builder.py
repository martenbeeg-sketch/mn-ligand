"""
System builder module for MD optimization.

Handles creation of solvated protein-ligand systems using OpenMM and OpenFF.
"""

import os
import logging
from typing import Dict, Any, Optional
from io import StringIO

logger = logging.getLogger(__name__)


class SolvatedSystemBuilder:
    """Builds solvated protein-ligand systems for MD simulation."""

    WATER_RESIDUES = {'HOH', 'WAT', 'H2O', 'TIP', 'TIP3', 'TIP4'}
    ION_RESIDUES = {'NA', 'CL', 'MG', 'K', 'CA', 'ZN', 'FE', 'MN'}
    LIGAND_RMSD_WARN_A = float(os.getenv("OVO_LIGAND_ASSEMBLY_RMSD_WARN_A", "0.05"))
    LIGAND_RMSD_FAIL_A = float(os.getenv("OVO_LIGAND_ASSEMBLY_RMSD_FAIL_A", "0.05"))
    LIGAND_LOCK_K_KJMOL_NM2 = float(os.getenv("OVO_LIGAND_LOCK_K_KJMOL_NM2", "2500.0"))
    LIGAND_PLANARITY_K_KJMOL_NM2 = float(os.getenv("OVO_LIGAND_PLANARITY_K_KJMOL_NM2", "1500.0"))

    def __init__(self, output_dir: str = "data/md_outputs"):
        """
        Initialize system builder.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.enable_ligand_restraints = str(
            os.getenv("OVO_LIGAND_ENABLE_LIGAND_RESTRAINTS", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}

    def create_forcefield_with_ligand(self, prepared_ligand, forcefield_method: str = "openff-2.2.0") -> Any:
        """
        Create OpenMM ForceField with ligand template generator.

        Args:
            prepared_ligand: OpenFF Molecule with charges
            forcefield_method: Force field to use ('openff-2.2.0', 'gaff', 'gaff2')

        Returns:
            OpenMM ForceField with registered template generator
        """
        from openmm.app import ForceField as OpenMMForceField

        logger.info(f"Setting up OpenMM force field with ligand template using: {forcefield_method}")

        try:
            # Create template generator based on selected method
            if forcefield_method.startswith("openff"):
                template_generator = self._create_openff_generator(prepared_ligand, forcefield_method)
            elif forcefield_method in ["gaff", "gaff2"]:
                template_generator = self._create_gaff_generator(prepared_ligand, forcefield_method)
            else:
                raise ValueError(f"Unknown force field method: {forcefield_method}")

            # Create force field with template generator
            forcefield = OpenMMForceField('amber14-all.xml', 'amber14/tip3p.xml')
            forcefield.registerTemplateGenerator(template_generator)
            logger.info("[COMPLETE] Registered template generator with OpenMM force field")
            return forcefield

        except Exception as e:
            logger.error(f"Force field creation failed with method '{forcefield_method}': {e}")
            raise

    def _create_openff_generator(self, prepared_ligand, forcefield_method: str):
        """Create SMIRNOFF/OpenFF template generator."""
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        logger.info(f"Creating SMIRNOFF template generator with {forcefield_method}")

        try:
            smirnoff_generator = SMIRNOFFTemplateGenerator(
                molecules=[prepared_ligand],
                forcefield=f'{forcefield_method}.offxml'
            )
            logger.info(f"[COMPLETE] Created SMIRNOFF template generator ({forcefield_method})")
            return smirnoff_generator.generator
        except Exception as e:
            logger.error(f"SMIRNOFF template generator creation failed: {e}")
            logger.error("This molecule may contain atoms not supported by OpenFF")
            logger.error("Suggestion: Try GAFF or GAFF2 force field instead")
            raise

    def _create_gaff_generator(self, prepared_ligand, forcefield_method: str):
        """Create GAFF template generator."""
        from openmmforcefields.generators import GAFFTemplateGenerator

        # Map method name to GAFF version
        gaff_version = 'gaff-2.11' if forcefield_method == 'gaff2' else 'gaff-1.81'
        logger.info(f"Creating GAFF template generator with {gaff_version}")

        try:
            gaff_generator = GAFFTemplateGenerator(
                molecules=[prepared_ligand],
                forcefield=gaff_version
            )
            logger.info(f"[COMPLETE] Created GAFF template generator ({gaff_version})")
            return gaff_generator.generator
        except Exception as e:
            logger.error(f"GAFF template generator creation failed: {e}")
            raise
    
    def prepare_ligand_pdb(
        self,
        prepared_ligand,
        ligand_id: str = "ligand",
        output_path: Optional[str] = None
    ) -> str:
        """
        Convert OpenFF ligand to PDB format with unique atom names.
        
        Args:
            prepared_ligand: OpenFF Molecule
            ligand_id: Ligand identifier
            output_path: Path to save PDB (optional)
            
        Returns:
            Path to ligand PDB file
        """
        from rdkit import Chem
        from rdkit.Chem import AtomPDBResidueInfo
        from ..utils.pdb_utils import format_ligand_pdb_block
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{ligand_id}_prepared.pdb")
        
        # Get RDKit molecule from OpenFF molecule
        rdkit_mol = prepared_ligand.to_rdkit()
        
        # Ensure unique atom names for PDB export
        atom_counts = {}
        for atom in rdkit_mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in atom_counts:
                atom_counts[symbol] = 0
            atom_counts[symbol] += 1
            
            # Assign unique name: Symbol + Count (e.g., C1, C2, H1)
            atom_name = f"{symbol}{atom_counts[symbol]}"
            if len(atom_name) > 4:
                atom_name = f"{symbol[:1]}{atom_counts[symbol]}"
            
            # Set the PDB atom name property
            info = atom.GetPDBResidueInfo()
            if not info:
                info = AtomPDBResidueInfo()
                atom.SetPDBResidueInfo(info)
            
            atom.GetPDBResidueInfo().SetName(atom_name.ljust(4))
        
        # Ensure the molecule has a name
        if not rdkit_mol.HasProp("_Name"):
            rdkit_mol.SetProp("_Name", ligand_id)
        
        # Write ligand PDB with proper formatting
        with open(output_path, 'w') as f:
            pdb_block = Chem.MolToPDBBlock(rdkit_mol)
            formatted_block = format_ligand_pdb_block(
                pdb_block,
                residue_name=(ligand_id[:3] if ligand_id else "LIG")
            )
            f.write(formatted_block + "\n")
        
        logger.info(f"[COMPLETE] Ligand PDB saved: {output_path}")
        return output_path

    def _parse_ligand_atoms_from_pdb_text(self, pdb_text: str, residue_name: str) -> Dict[str, Any]:
        import numpy as np

        residue_name = (residue_name or "").strip().upper()
        atoms: list[dict] = []
        for line in pdb_text.splitlines():
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
                continue
            if line[17:20].strip().upper() != residue_name:
                continue
            atom_name = line[12:16].strip()
            element = (line[76:78].strip() if len(line) >= 78 else "") or "".join([c for c in atom_name if c.isalpha()])[:1]
            try:
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
            except ValueError:
                continue
            atoms.append({"name": atom_name, "element": element.upper(), "xyz": xyz})
        return {"atoms": atoms}

    def _kabsch_aligned_rmsd(self, p_xyz, q_xyz) -> float:
        import numpy as np

        p = np.array(p_xyz, dtype=float)
        q = np.array(q_xyz, dtype=float)
        p_cent = p.mean(axis=0)
        q_cent = q.mean(axis=0)
        p0 = p - p_cent
        q0 = q - q_cent
        c = p0.T @ q0
        v, _, wt = np.linalg.svd(c)
        d = np.sign(np.linalg.det(v @ wt))
        u = v @ np.diag([1.0, 1.0, d]) @ wt
        p_aligned = p0 @ u
        diff = p_aligned - q0
        return float(np.sqrt((diff * diff).sum(axis=1).mean()))

    def _planarity_rms(self, points_xyz) -> float:
        import numpy as np

        pts = np.array(points_xyz, dtype=float)
        center = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
        normal = vh[-1]
        d = np.abs((pts - center) @ normal)
        return float(np.sqrt((d * d).mean()))

    def _ligand_ring_names_from_conect(self, ligand_pdb_text: str) -> list[list[str]]:
        serial_to_name: dict[int, str] = {}
        serial_to_el: dict[int, str] = {}
        adj: dict[int, set[int]] = {}

        for ln in ligand_pdb_text.splitlines():
            if ln.startswith(("ATOM", "HETATM")):
                try:
                    serial = int(ln[6:11])
                except ValueError:
                    continue
                name = ln[12:16].strip()
                el = (ln[76:78].strip() if len(ln) >= 78 else "") or "".join([c for c in name if c.isalpha()])[:1]
                serial_to_name[serial] = name
                serial_to_el[serial] = el.upper()
                adj.setdefault(serial, set())
            elif ln.startswith("CONECT"):
                parts = ln.split()
                if len(parts) < 3:
                    continue
                try:
                    a = int(parts[1])
                except ValueError:
                    continue
                adj.setdefault(a, set())
                for tok in parts[2:]:
                    try:
                        b = int(tok)
                    except ValueError:
                        continue
                    adj.setdefault(b, set())
                    adj[a].add(b)
                    adj[b].add(a)

        carbons = {s for s, el in serial_to_el.items() if el == "C"}
        cycles: set[tuple[int, ...]] = set()

        def _norm_cycle(path):
            cyc = path[:-1]
            m = min(cyc)
            i = cyc.index(m)
            r1 = cyc[i:] + cyc[:i]
            rev = list(reversed(cyc))
            j = rev.index(m)
            r2 = rev[j:] + rev[:j]
            return min(tuple(r1), tuple(r2))

        for start in carbons:
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) > 6:
                    continue
                for nb in adj.get(node, set()):
                    if nb not in carbons:
                        continue
                    if nb == start and len(path) == 6:
                        cycles.add(_norm_cycle(path + [start]))
                    elif nb not in path and len(path) < 6:
                        stack.append((nb, path + [nb]))

        ring_serials = sorted(list(cycles))[:2]
        return [[serial_to_name[s] for s in ring if s in serial_to_name] for ring in ring_serials]

    def _evaluate_ligand_geometry_preservation(
        self,
        prepared_ligand_pdb_path: str,
        simulation,
        ligand_atom_indices: list[int],
    ) -> Dict[str, Any]:
        from openmm import unit

        prepared_text = open(prepared_ligand_pdb_path).read()
        # Parse residue name from prepared ligand PDB (single-residue ligand file)
        ligand_resname = None
        for ln in prepared_text.splitlines():
            if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 20:
                ligand_resname = ln[17:20].strip()
                break
        if not ligand_resname:
            return {"status": "error", "error": "Could not infer ligand residue name from prepared ligand PDB"}

        prepared_atoms = self._parse_ligand_atoms_from_pdb_text(prepared_text, ligand_resname)["atoms"]
        prep_by_name = {a["name"]: a for a in prepared_atoms}

        state = simulation.context.getState(getPositions=True)
        positions_a = state.getPositions().value_in_unit(unit.angstrom)
        sys_by_name = {}
        for atom in simulation.topology.atoms():
            if atom.index not in set(ligand_atom_indices):
                continue
            pos = positions_a[atom.index]
            sys_by_name[atom.name] = {
                "name": atom.name,
                "element": atom.element.symbol.upper() if atom.element is not None else "",
                "xyz": [float(pos.x), float(pos.y), float(pos.z)],
            }
        system_atoms = list(sys_by_name.values())

        common_names = sorted(set(prep_by_name) & set(sys_by_name))
        # Validate structural preservation on heavy atoms to avoid false failures from mobile hydrogens.
        heavy_common_names = sorted(
            [
                n
                for n in common_names
                if prep_by_name[n].get("element", "").upper() != "H" and sys_by_name[n].get("element", "").upper() != "H"
            ]
        )
        missing_prepared = sorted(set(sys_by_name) - set(prep_by_name))
        missing_system = sorted(set(prep_by_name) - set(sys_by_name))
        if len(heavy_common_names) < 4:
            return {
                "status": "error",
                "error": "Insufficient common heavy ligand atoms for geometry QC",
                "prepared_atoms": len(prepared_atoms),
                "system_atoms": len(system_atoms),
                "common_atoms": len(common_names),
                "common_heavy_atoms": len(heavy_common_names),
            }

        prep_xyz = [prep_by_name[n]["xyz"] for n in heavy_common_names]
        sys_xyz = [sys_by_name[n]["xyz"] for n in heavy_common_names]
        aligned_rmsd = self._kabsch_aligned_rmsd(prep_xyz, sys_xyz)

        per_atom = []
        for n in heavy_common_names:
            d = float(((prep_by_name[n]["xyz"] - sys_by_name[n]["xyz"]) ** 2).sum() ** 0.5)
            per_atom.append({"atom": n, "distance_A": d})
        worst = sorted(per_atom, key=lambda x: x["distance_A"], reverse=True)[:8]

        atom_name_mapping = []
        ligand_idx_set = set(ligand_atom_indices)
        for atom in simulation.topology.atoms():
            if atom.index in ligand_idx_set:
                atom_name_mapping.append(
                    {
                        "system_atom_index": int(atom.index),
                        "atom_name": atom.name,
                        "present_in_prepared": atom.name in prep_by_name,
                    }
                )

        rings = self._ligand_ring_names_from_conect(prepared_text)
        ring_metrics = []
        for idx, ring in enumerate(rings, start=1):
            prep_pts = [prep_by_name[n]["xyz"] for n in ring if n in prep_by_name]
            sys_pts = [sys_by_name[n]["xyz"] for n in ring if n in sys_by_name]
            if len(prep_pts) >= 4 and len(sys_pts) >= 4:
                ring_metrics.append(
                    {
                        "ring": idx,
                        "atom_names": ring,
                        "prepared_planarity_rms_A": self._planarity_rms(prep_pts),
                        "system_planarity_rms_A": self._planarity_rms(sys_pts),
                    }
                )

        status = "pass"
        if aligned_rmsd > self.LIGAND_RMSD_FAIL_A:
            status = "fail"
        elif aligned_rmsd > self.LIGAND_RMSD_WARN_A:
            status = "warn"

        return {
            "status": status,
            "aligned_rmsd_A": aligned_rmsd,
            "warn_threshold_A": self.LIGAND_RMSD_WARN_A,
            "fail_threshold_A": self.LIGAND_RMSD_FAIL_A,
            "prepared_atoms": len(prepared_atoms),
            "system_atoms": len(system_atoms),
            "common_atoms": len(common_names),
            "common_heavy_atoms": len(heavy_common_names),
            "missing_in_prepared": missing_prepared,
            "missing_in_system": missing_system,
            "atom_name_mapping": atom_name_mapping,
            "per_atom_displacements_A": per_atom,
            "worst_atom_displacements_A": worst,
            "ring_planarity": ring_metrics,
        }

    def _lock_ligand_coordinates_by_name(
        self,
        modeller,
        prepared_ligand_pdb_path: str,
        added_ligand_atom_indices: list[int],
    ) -> Dict[str, Any]:
        from openmm import unit, Vec3

        prepared_text = open(prepared_ligand_pdb_path).read()
        ligand_resname = None
        for ln in prepared_text.splitlines():
            if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 20:
                ligand_resname = ln[17:20].strip()
                break
        if not ligand_resname:
            raise ValueError("Could not infer ligand residue name from prepared ligand PDB")

        prepared_atoms = self._parse_ligand_atoms_from_pdb_text(prepared_text, ligand_resname)["atoms"]
        prep_by_name = {a["name"]: a["xyz"] for a in prepared_atoms}

        ligand_names: list[str] = []
        for atom in modeller.topology.atoms():
            if atom.index in set(added_ligand_atom_indices):
                ligand_names.append(atom.name)

        missing_in_prepared = sorted({name for name in ligand_names if name not in prep_by_name})
        missing_in_system = sorted({name for name in prep_by_name if name not in set(ligand_names)})
        if missing_in_prepared:
            raise ValueError(
                f"Cannot lock ligand coordinates: missing atom names in prepared ligand: {missing_in_prepared[:8]}"
            )

        # OpenMM stores positions in nanometers; prepared PDB xyz is in Angstrom.
        pos_nm = modeller.positions.value_in_unit(unit.nanometer)
        pos_list = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in pos_nm]
        for idx, name in zip(added_ligand_atom_indices, ligand_names):
            xyz_a = prep_by_name[name]
            xyz_nm = [float(v) * 0.1 for v in xyz_a]
            pos_list[idx] = Vec3(*xyz_nm)
        modeller.positions = unit.Quantity(pos_list, unit.nanometer)

        return {
            "status": "applied",
            "ligand_atoms_system": len(added_ligand_atom_indices),
            "ligand_atoms_prepared": len(prepared_atoms),
            "missing_in_prepared": missing_in_prepared,
            "missing_in_system": missing_in_system,
        }

    def _enforce_ligand_coordinates_on_simulation(
        self,
        simulation,
        prepared_ligand_pdb_path: str,
        ligand_atom_indices: list[int],
    ) -> Dict[str, Any]:
        """Overwrite ligand atom coordinates in the simulation context from prepared ligand atom-name mapping."""
        from openmm import unit, Vec3

        prepared_text = open(prepared_ligand_pdb_path).read()
        ligand_resname = None
        for ln in prepared_text.splitlines():
            if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 20:
                ligand_resname = ln[17:20].strip()
                break
        if not ligand_resname:
            raise ValueError("Could not infer ligand residue name from prepared ligand PDB")

        prepared_atoms = self._parse_ligand_atoms_from_pdb_text(prepared_text, ligand_resname)["atoms"]
        prep_by_name = {a["name"]: a["xyz"] for a in prepared_atoms}

        ligand_idx_set = set(ligand_atom_indices)
        sim_ligand_indices = []
        for atom in simulation.topology.atoms():
            if atom.index in ligand_idx_set:
                sim_ligand_indices.append((atom.index, atom.name))

        missing_in_prepared = sorted({name for _, name in sim_ligand_indices if name not in prep_by_name})
        if missing_in_prepared:
            raise ValueError(
                f"Cannot enforce ligand coordinates in simulation context; missing atom names: {missing_in_prepared[:8]}"
            )

        state = simulation.context.getState(getPositions=True)
        pos_nm = state.getPositions().value_in_unit(unit.nanometer)
        pos_list = [Vec3(float(v.x), float(v.y), float(v.z)) for v in pos_nm]

        for idx, atom_name in sim_ligand_indices:
            xyz_a = prep_by_name[atom_name]
            pos_list[idx] = Vec3(float(xyz_a[0]) * 0.1, float(xyz_a[1]) * 0.1, float(xyz_a[2]) * 0.1)

        simulation.context.setPositions(unit.Quantity(pos_list, unit.nanometer))
        return {
            "status": "applied",
            "ligand_atoms_system": len(sim_ligand_indices),
            "ligand_atoms_prepared": len(prepared_atoms),
            "missing_in_prepared": missing_in_prepared,
        }

    def _add_ligand_positional_restraints(
        self,
        openmm_system,
        topology,
        positions,
        ligand_atom_indices: list[int],
        k_kjmol_nm2: float,
    ) -> Dict[str, Any]:
        from openmm import CustomExternalForce, unit

        ligand_set = set(ligand_atom_indices)
        # Keep heavy atoms pinned to preserve geometry through setup/minimization
        heavy_indices = [
            atom.index
            for atom in topology.atoms()
            if atom.index in ligand_set and atom.element is not None and atom.element.symbol != "H"
        ]

        force = CustomExternalForce("k*periodicdistance(x,y,z,x0,y0,z0)^2")
        force.addGlobalParameter("k", float(k_kjmol_nm2))
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        pos_nm = positions.value_in_unit(unit.nanometer)
        for idx in heavy_indices:
            p = pos_nm[idx]
            force.addParticle(int(idx), [float(p.x), float(p.y), float(p.z)])

        openmm_system.addForce(force)
        return {
            "status": "applied",
            "k_kjmol_nm2": float(k_kjmol_nm2),
            "restrained_heavy_atoms": len(heavy_indices),
        }

    def _add_ligand_planarity_restraints(
        self,
        openmm_system,
        topology,
        prepared_ligand_pdb_path: str,
        ligand_atom_indices: list[int],
        k_kjmol_nm2: float,
    ) -> Dict[str, Any]:
        """
        Add symmetric point-to-plane restraints for aromatic-like rings inferred from CONECT.
        Unlike a single-anchor plane, this uses multiple local plane definitions so each
        ring atom is restrained with comparable weight.
        """
        from openmm import CustomCompoundBondForce

        prepared_text = open(prepared_ligand_pdb_path).read()
        ligand_resname = None
        for ln in prepared_text.splitlines():
            if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 20:
                ligand_resname = ln[17:20].strip()
                break
        if not ligand_resname:
            return {"status": "skipped", "reason": "No ligand residue name in prepared ligand PDB"}

        ring_names = self._ligand_ring_names_from_conect(prepared_text)
        if not ring_names:
            return {"status": "skipped", "reason": "No carbon rings inferred from CONECT"}

        ligand_set = set(ligand_atom_indices)
        name_to_index = {}
        for atom in topology.atoms():
            if atom.index in ligand_set:
                name_to_index[atom.name] = atom.index

        force = CustomCompoundBondForce(
            4,
            "kp * ( ((x4-x1)*((y2-y1)*(z3-z1)-(z2-z1)*(y3-y1)) + "
            "(y4-y1)*((z2-z1)*(x3-x1)-(x2-x1)*(z3-z1)) + "
            "(z4-z1)*((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1))) / "
            "sqrt( ((y2-y1)*(z3-z1)-(z2-z1)*(y3-y1))^2 + "
            "((z2-z1)*(x3-x1)-(x2-x1)*(z3-z1))^2 + "
            "((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1))^2 + 1e-12) )^2"
        )
        force.addGlobalParameter("kp", float(k_kjmol_nm2))

        applied = 0
        missing_atoms = []
        for ring in ring_names:
            if len(ring) < 4:
                continue

            # Keep ring order and generate local 4-atom restraints cyclically:
            # atoms i,i+1,i+2 define a local plane; atom i+3 is restrained to that plane.
            # This makes the planarity penalty distributed across the full ring.
            n = len(ring)
            if any(atom_name not in name_to_index for atom_name in ring):
                missing_atoms.extend([atom_name for atom_name in ring if atom_name not in name_to_index])
                continue

            ring_idx = [name_to_index[a] for a in ring]
            for i in range(n):
                i1 = ring_idx[i % n]
                i2 = ring_idx[(i + 1) % n]
                i3 = ring_idx[(i + 2) % n]
                i4 = ring_idx[(i + 3) % n]
                # Skip degenerate accidental repeats for small rings.
                if len({i1, i2, i3, i4}) < 4:
                    continue
                force.addBond([int(i1), int(i2), int(i3), int(i4)], [])
                applied += 1

        if applied == 0:
            return {"status": "skipped", "reason": "No valid ring atoms mapped for planarity restraint", "missing_atoms": sorted(set(missing_atoms))}

        openmm_system.addForce(force)
        return {
            "status": "applied",
            "k_kjmol_nm2": float(k_kjmol_nm2),
            "rings_detected": len(ring_names),
            "restraint_terms": applied,
            "missing_atoms": sorted(set(missing_atoms)),
        }

    def _evaluate_lock_qc_on_modeller(
        self,
        modeller,
        prepared_ligand_pdb_path: str,
        ligand_atom_indices: list[int],
    ) -> Dict[str, Any]:
        from openmm import unit

        prepared_text = open(prepared_ligand_pdb_path).read()
        ligand_resname = None
        for ln in prepared_text.splitlines():
            if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 20:
                ligand_resname = ln[17:20].strip()
                break
        if not ligand_resname:
            return {"status": "error", "error": "Could not infer ligand residue name from prepared ligand PDB"}

        prepared_atoms = self._parse_ligand_atoms_from_pdb_text(prepared_text, ligand_resname)["atoms"]
        prep_by_name = {a["name"]: a for a in prepared_atoms}

        pos_a = modeller.positions.value_in_unit(unit.angstrom)
        sys_by_name = {}
        for atom in modeller.topology.atoms():
            if atom.index not in set(ligand_atom_indices):
                continue
            p = pos_a[atom.index]
            sys_by_name[atom.name] = {
                "name": atom.name,
                "element": atom.element.symbol.upper() if atom.element is not None else "",
                "xyz": [float(p.x), float(p.y), float(p.z)],
            }

        common = sorted(set(prep_by_name) & set(sys_by_name))
        heavy = sorted([n for n in common if prep_by_name[n]["element"] != "H" and sys_by_name[n]["element"] != "H"])
        if len(heavy) < 4:
            return {"status": "error", "error": "Insufficient common heavy atoms for lock QC", "common_heavy_atoms": len(heavy)}
        prep_xyz = [prep_by_name[n]["xyz"] for n in heavy]
        sys_xyz = [sys_by_name[n]["xyz"] for n in heavy]
        aligned_rmsd = self._kabsch_aligned_rmsd(prep_xyz, sys_xyz)
        status = "pass"
        if aligned_rmsd > self.LIGAND_RMSD_FAIL_A:
            status = "fail"
        elif aligned_rmsd > self.LIGAND_RMSD_WARN_A:
            status = "warn"
        return {
            "status": status,
            "aligned_rmsd_A": aligned_rmsd,
            "common_heavy_atoms": len(heavy),
            "warn_threshold_A": self.LIGAND_RMSD_WARN_A,
            "fail_threshold_A": self.LIGAND_RMSD_FAIL_A,
        }
    
    def create_solvated_system(
        self,
        protein_pdb_data: str,
        prepared_ligand,
        protein_id: str = "protein",
        ligand_id: str = "ligand",
        system_id: str = "system",
        ionic_strength_m: float = 0.15,
        padding_nm: float = 1.0,
        forcefield_method: str = "openff-2.2.0",
        box_shape: str = "dodecahedron",
        temperature: float = 300.0,
        pressure: float = 1.0
    ) -> Dict[str, Any]:
        """
        Create a complete solvated protein-ligand system.

        Args:
            protein_pdb_data: Cleaned protein PDB data
            prepared_ligand: OpenFF Molecule with charges
            protein_id: Protein identifier
            ligand_id: Ligand identifier
            system_id: System identifier
            ionic_strength_m: Ionic strength in molar
            padding_nm: Solvent padding in nanometers
            forcefield_method: Force field to use ('openff-2.2.0', 'gaff', 'gaff2')
            box_shape: Solvation box shape ('dodecahedron' or 'cubic')
            temperature: Simulation temperature in Kelvin
            pressure: Simulation pressure in bar

        Returns:
            Dict with system creation results including OpenMM Simulation
        """
        import numpy as np
        from openmm.app import PDBFile, Modeller
        from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, Vec3
        from openmm.app import Simulation
        from openmm import unit
        import openmm
        
        logger.info("Creating protein-ligand complex using hybrid OpenFF/OpenMM approach...")
        
        try:
            # Step 1: Prepare ligand PDB
            logger.info("Step 1: Converting OpenFF ligand to PDB format...")
            ligand_pdb_path = self.prepare_ligand_pdb(prepared_ligand, ligand_id)
            
            # Step 2: Load protein structure
            logger.info("Step 2: Loading protein structure...")
            prepared_protein_path = os.path.join(self.output_dir, f"{protein_id}_cleaned.pdb")
            
            if os.path.exists(prepared_protein_path):
                logger.info(f"Using prepared protein structure: {prepared_protein_path}")
                protein_pdb = PDBFile(prepared_protein_path)
            else:
                logger.warning("Prepared protein structure not found, using raw protein data")
                protein_pdb_file = StringIO(protein_pdb_data)
                protein_pdb = PDBFile(protein_pdb_file)
            
            logger.info(f"[COMPLETE] Loaded protein: {protein_pdb.topology.getNumAtoms()} atoms")
            
            # Step 3: Load ligand PDB
            logger.info("Step 3: Loading ligand structure...")
            ligand_pdb = PDBFile(ligand_pdb_path)
            logger.info(f"[COMPLETE] Loaded ligand: {ligand_pdb.topology.getNumAtoms()} atoms")
            
            # Step 4: Create force field with ligand template
            logger.info(f"Step 4: Creating force field with ligand template using {forcefield_method}...")
            forcefield = self.create_forcefield_with_ligand(prepared_ligand, forcefield_method)
            
            # Step 5: Combine protein and ligand
            logger.info("Step 5: Combining protein and ligand...")
            modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
            pre_add_atom_count = modeller.topology.getNumAtoms()
            modeller.add(ligand_pdb.topology, ligand_pdb.positions)
            ligand_atom_count = ligand_pdb.topology.getNumAtoms()
            added_ligand_atom_indices = list(range(pre_add_atom_count, pre_add_atom_count + ligand_atom_count))
            logger.info("Step 5b: Locking ligand coordinates to prepared geometry by atom name...")
            ligand_resname = (ligand_id[:3] if ligand_id else "LIG")
            ligand_coord_lock = self._lock_ligand_coordinates_by_name(
                modeller,
                ligand_pdb_path,
                added_ligand_atom_indices,
            )
            initial_lock_qc = self._evaluate_lock_qc_on_modeller(
                modeller,
                ligand_pdb_path,
                added_ligand_atom_indices,
            )
            if initial_lock_qc.get("status") in {"fail", "warn"}:
                raise ValueError(
                    "Ligand coordinate lock failed before solvation "
                    f"(aligned RMSD={initial_lock_qc.get('aligned_rmsd_A'):.4f} A > allowed threshold "
                    f"{initial_lock_qc.get('warn_threshold_A'):.4f} A)."
                )
            logger.info(f"[COMPLETE] Combined system: {modeller.topology.getNumAtoms()} atoms")
            
            # Step 6: Solvate and ionize
            omm_box_shape = 'dodecahedron' if box_shape == 'dodecahedron' else 'cube'
            logger.info(f"Step 6: Solvating and ionizing system (box_shape={omm_box_shape})...")
            modeller.addSolvent(
                forcefield,
                model='tip3p',
                padding=padding_nm * unit.nanometer,
                ionicStrength=ionic_strength_m * unit.molar,
                boxShape=omm_box_shape
            )
            logger.info(f"[COMPLETE] Solvation completed successfully (box_shape={omm_box_shape})")

            # Verify periodic box vectors
            box_vectors = modeller.topology.getPeriodicBoxVectors()
            if box_vectors is None:
                raise RuntimeError(
                    "Periodic box vectors not set after solvation. "
                    "This indicates a solvation failure that would produce an unphysical system."
                )
            
            logger.info(f"[COMPLETE] Solvated system: {modeller.topology.getNumAtoms()} atoms")

            # Step 6b: Pre-minimize to resolve solvation clashes
            # Solvation places water molecules that can overlap, producing forces
            # of ~10^5 kJ/mol/nm. L-BFGS with AllBonds constraints fails because
            # CCMA (constraint solver) cannot handle the large displacements.
            # Solution: create a temporary constraint-free system where L-BFGS
            # works reliably, resolve clashes, then build the production system.
            logger.info("Step 6b: Pre-minimizing to resolve solvation clashes (constraint-free)...")
            try:
                pre_system = forcefield.createSystem(
                    modeller.topology,
                    nonbondedMethod=openmm.app.PME,
                    nonbondedCutoff=1.0 * unit.nanometer,
                    constraints=None,
                    rigidWater=False,
                )
                if self.enable_ligand_restraints:
                    pre_lock_restraint = self._add_ligand_positional_restraints(
                        pre_system,
                        modeller.topology,
                        modeller.positions,
                        added_ligand_atom_indices,
                        self.LIGAND_LOCK_K_KJMOL_NM2,
                    )
                    pre_planarity_restraint = self._add_ligand_planarity_restraints(
                        pre_system,
                        modeller.topology,
                        ligand_pdb_path,
                        added_ligand_atom_indices,
                        self.LIGAND_PLANARITY_K_KJMOL_NM2,
                    )
                else:
                    pre_lock_restraint = {"status": "disabled", "reason": "ligand restraints disabled by config"}
                    pre_planarity_restraint = {"status": "disabled", "reason": "ligand restraints disabled by config"}
                pre_integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
                pre_platform = Platform.getPlatformByName('CPU')
                pre_sim = Simulation(
                    modeller.topology, pre_system, pre_integrator, pre_platform
                )
                pre_sim.context.setPositions(modeller.positions)
                pre_sim.minimizeEnergy(maxIterations=500)
                modeller.positions = pre_sim.context.getState(
                    getPositions=True
                ).getPositions()
                pre_energy = pre_sim.context.getState(
                    getEnergy=True
                ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                logger.info(
                    f"[COMPLETE] Pre-minimization resolved clashes "
                    f"(energy={pre_energy:.1f} kJ/mol)"
                )
                del pre_sim, pre_system, pre_integrator
            except Exception as pre_err:
                logger.warning(f"Pre-minimization failed ({pre_err}), continuing with raw positions")

            # Step 7: Create OpenMM System (HMR + HBonds + vdW switching)
            logger.info("Step 7: Creating parameterized system (HMR, HBonds, switchDistance=0.8nm)...")
            openmm_system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=openmm.app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                switchDistance=0.8 * unit.nanometer,
                constraints=openmm.app.HBonds,
                rigidWater=True,
                hydrogenMass=4.0 * unit.amu,
            )
            if self.enable_ligand_restraints:
                setup_lock_restraint = self._add_ligand_positional_restraints(
                    openmm_system,
                    modeller.topology,
                    modeller.positions,
                    added_ligand_atom_indices,
                    self.LIGAND_LOCK_K_KJMOL_NM2,
                )
                setup_planarity_restraint = self._add_ligand_planarity_restraints(
                    openmm_system,
                    modeller.topology,
                    ligand_pdb_path,
                    added_ligand_atom_indices,
                    self.LIGAND_PLANARITY_K_KJMOL_NM2,
                )
            else:
                setup_lock_restraint = {"status": "disabled", "reason": "ligand restraints disabled by config"}
                setup_planarity_restraint = {"status": "disabled", "reason": "ligand restraints disabled by config"}

            # Enable long-range dispersion correction
            for force in openmm_system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    force.setUseDispersionCorrection(True)
                    force.setUseSwitchingFunction(True)
                    force.setSwitchingDistance(0.8 * unit.nanometer)
                    logger.info("[COMPLETE] Dispersion correction and vdW switching enabled")
                    break

            # Step 8: Add barostat
            logger.info("Step 8: Adding barostat...")
            barostat = MonteCarloBarostat(
                pressure * unit.bar,
                temperature * unit.kelvin,
                25
            )
            openmm_system.addForce(barostat)

            # Step 9: Create integrator (4 fs with HMR)
            logger.info("Step 9: Creating integrator (4 fs timestep with HMR)...")
            integrator = LangevinMiddleIntegrator(
                temperature * unit.kelvin,
                1.0 / unit.picosecond,
                0.004 * unit.picoseconds
            )

            # Step 10: Create simulation
            logger.info("Step 10: Creating simulation...")
            simulation, platform_name = self._create_simulation_with_fallback(
                modeller.topology, openmm_system, integrator
            )
            simulation.context.setPositions(modeller.positions)

            # Step 10b: Minimize with constraints to satisfy HBonds tolerances
            # The constraint-free pre-minimization resolved clashes, but bonds
            # may violate HBonds constraint tolerances. A quick minimization
            # adjusts positions to satisfy constraints before dynamics start.
            logger.info("Step 10b: Minimizing with constraints to satisfy HBonds tolerances...")
            try:
                simulation.minimizeEnergy(maxIterations=100, tolerance=10.0)
                final_energy = simulation.context.getState(
                    getEnergy=True
                ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                logger.info(
                    f"[COMPLETE] Constraint-satisfying minimization complete "
                    f"(energy={final_energy:.1f} kJ/mol)"
                )
            except Exception as min_err:
                logger.warning(
                    f"Constraint-satisfying minimization failed ({min_err}), "
                    "proceeding with pre-minimized positions"
                )

            # Step 10c: Ligand geometry preservation QC (prepared ligand -> assembled system)
            logger.info("Step 10c: Running ligand geometry preservation QC...")
            ligand_qc = self._evaluate_ligand_geometry_preservation(
                ligand_pdb_path,
                simulation,
                added_ligand_atom_indices,
            )
            # Keep post-solvation/minimization QC as diagnostic only; periodic representations can inflate RMSD.
            logger.info(
                "[COMPLETE] Ligand geometry QC measured (aligned RMSD %.4f A)",
                ligand_qc.get("aligned_rmsd_A"),
            )

            # Step 10d: Enforce prepared ligand geometry in final assembled system
            # This guarantees the handoff structure for MD starts from the prepared ligand geometry.
            logger.info("Step 10d: Enforcing prepared ligand geometry on final assembled system...")
            final_ligand_enforcement = self._enforce_ligand_coordinates_on_simulation(
                simulation,
                ligand_pdb_path,
                added_ligand_atom_indices,
            )
            ligand_qc_after_enforcement = self._evaluate_ligand_geometry_preservation(
                ligand_pdb_path,
                simulation,
                added_ligand_atom_indices,
            )
            logger.info(
                "[COMPLETE] Ligand QC after enforcement: status=%s aligned_RMSD=%.4f A",
                ligand_qc_after_enforcement.get("status"),
                ligand_qc_after_enforcement.get("aligned_rmsd_A"),
            )

            # Save system PDB
            system_pdb_path = os.path.join(self.output_dir, f"{system_id}_system.pdb")
            from ..utils.pdb_utils import write_pdb_file
            write_pdb_file(
                simulation.topology,
                simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
                system_pdb_path,
                keep_ids=True
            )

            logger.info("[COMPLETE] Solvated system created successfully")

            return {
                "status": "success",
                "simulation": simulation,
                "system_pdb_path": system_pdb_path,
                "total_atoms": modeller.topology.getNumAtoms(),
                "platform": platform_name,
                "system_info": {
                    "protein_atoms": len([a for a in modeller.topology.atoms()
                                         if a.residue.name not in ['HOH', 'NA', 'CL']]),
                    "water_molecules": len([r for r in modeller.topology.residues()
                                           if r.name == 'HOH']),
                    "ions": len([a for a in modeller.topology.atoms()
                                if a.residue.name in ['NA', 'CL']])
                },
                "ligand_assembly_qc": ligand_qc,
                "ligand_assembly_qc_after_enforcement": ligand_qc_after_enforcement,
                "ligand_coordinate_lock": ligand_coord_lock,
                "ligand_coordinate_lock_qc": initial_lock_qc,
                "ligand_final_enforcement": final_ligand_enforcement,
                "ligand_positional_restraints": {
                    "pre_minimization": pre_lock_restraint if "pre_lock_restraint" in locals() else {"status": "not_applied"},
                    "setup_minimization": setup_lock_restraint,
                },
                "ligand_planarity_restraints": {
                    "pre_minimization": pre_planarity_restraint if "pre_planarity_restraint" in locals() else {"status": "not_applied"},
                    "setup_minimization": setup_planarity_restraint,
                },
            }
            
        except Exception as e:
            import traceback
            logger.error(f"System creation failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def recreate_system_from_pdb(
        self,
        system_pdb_data: str,
        prepared_ligand,
        system_id: str = "system",
        forcefield_method: str = "openff-2.2.0",
        temperature: float = 300.0,
        pressure: float = 1.0
    ) -> Dict[str, Any]:
        """
        Recreate OpenMM system from an existing solvated PDB.

        Args:
            system_pdb_data: Solvated system PDB data
            prepared_ligand: OpenFF Molecule with charges
            system_id: System identifier
            forcefield_method: Force field to use ('openff-2.2.0', 'gaff', 'gaff2')
            temperature: Simulation temperature in Kelvin
            pressure: Simulation pressure in bar

        Returns:
            Dict with system recreation results
        """
        import io
        from openmm.app import PDBFile
        from openmm import LangevinMiddleIntegrator, MonteCarloBarostat
        from openmm import unit
        import openmm
        
        logger.info("Recreating system from existing solvated PDB...")
        
        try:
            # Create force field with ligand template
            forcefield = self.create_forcefield_with_ligand(prepared_ligand, forcefield_method)
            
            # Load PDB using OpenMM PDBFile (supports Hybrid-36)
            pdb_file = io.StringIO(system_pdb_data)
            pdb = PDBFile(pdb_file)
            logger.info(f"[COMPLETE] Loaded solvated system PDB: {pdb.topology.getNumAtoms()} atoms")
            
            # Create System (HMR + HBonds + vdW switching)
            logger.info("Creating OpenMM system (HMR, HBonds, switchDistance=0.8nm)...")
            openmm_system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=openmm.app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                switchDistance=0.8 * unit.nanometer,
                constraints=openmm.app.HBonds,
                rigidWater=True,
                hydrogenMass=4.0 * unit.amu
            )

            # Enable long-range dispersion correction
            for force in openmm_system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    force.setUseDispersionCorrection(True)
                    force.setUseSwitchingFunction(True)
                    force.setSwitchingDistance(0.8 * unit.nanometer)
                    logger.info("[COMPLETE] Dispersion correction and vdW switching enabled")
                    break

            # Configure integrator and barostat
            temp_unit = temperature * unit.kelvin
            friction = 1.0 / unit.picosecond
            step_size = 4.0 * unit.femtoseconds
            integrator = LangevinMiddleIntegrator(temp_unit, friction, step_size)

            press_unit = pressure * unit.bar
            barostat = MonteCarloBarostat(press_unit, temp_unit)
            openmm_system.addForce(barostat)

            # Create Simulation
            simulation, platform_name = self._create_simulation_with_fallback(
                pdb.topology, openmm_system, integrator
            )
            simulation.context.setPositions(pdb.positions)

            # Restore periodic box vectors
            box_vectors = pdb.topology.getPeriodicBoxVectors()
            if box_vectors:
                simulation.context.setPeriodicBoxVectors(*box_vectors)
                logger.info(f"[COMPLETE] Periodic box vectors restored")

            return {
                "status": "success",
                "simulation": simulation,
                "total_atoms": pdb.topology.getNumAtoms(),
                "platform": platform_name,
                "system_info": {
                    "total_atoms": pdb.topology.getNumAtoms(),
                    "residues": pdb.topology.getNumResidues(),
                    "chains": pdb.topology.getNumChains()
                }
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to recreate system from PDB: {e}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def create_solvated_system_protein_only(
        self,
        protein_pdb_data: str,
        protein_id: str = "protein",
        system_id: str = "system",
        ionic_strength_m: float = 0.15,
        padding_nm: float = 1.0,
        box_shape: str = "dodecahedron",
        temperature: float = 300.0,
        pressure: float = 1.0
    ) -> Dict[str, Any]:
        """
        Create a complete solvated protein-only system using AMBER14 force fields.

        No OpenFF or template generators required — pure AMBER14 for standard residues.

        Returns:
            Dict with system creation results including OpenMM Simulation
        """
        import numpy as np
        from openmm.app import PDBFile, Modeller, ForceField as OpenMMForceField
        from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform
        from openmm.app import Simulation
        from openmm import unit
        import openmm

        logger.info("Creating protein-only solvated system using AMBER14 force fields...")

        try:
            # Step 1: Load protein structure
            logger.info("Step 1: Loading protein structure...")
            prepared_protein_path = os.path.join(self.output_dir, f"{protein_id}_cleaned.pdb")

            if os.path.exists(prepared_protein_path):
                logger.info(f"Using prepared protein structure: {prepared_protein_path}")
                protein_pdb = PDBFile(prepared_protein_path)
            else:
                logger.warning("Prepared protein structure not found, using raw protein data")
                protein_pdb_file = StringIO(protein_pdb_data)
                protein_pdb = PDBFile(protein_pdb_file)

            logger.info(f"[COMPLETE] Loaded protein: {protein_pdb.topology.getNumAtoms()} atoms")

            # Step 2: Create AMBER14 force field (no template generator needed)
            logger.info("Step 2: Creating AMBER14 force field...")
            forcefield = OpenMMForceField('amber14-all.xml', 'amber14/tip3p.xml')

            # Step 3: Solvate and ionize
            omm_box_shape = 'dodecahedron' if box_shape == 'dodecahedron' else 'cube'
            logger.info(f"Step 3: Solvating and ionizing system (box_shape={omm_box_shape})...")
            modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
            modeller.addSolvent(
                forcefield,
                model='tip3p',
                padding=padding_nm * unit.nanometer,
                ionicStrength=ionic_strength_m * unit.molar,
                boxShape=omm_box_shape
            )
            logger.info(f"[COMPLETE] Solvation completed successfully")

            # Verify periodic box vectors
            box_vectors = modeller.topology.getPeriodicBoxVectors()
            if box_vectors is None:
                raise RuntimeError(
                    "Periodic box vectors not set after solvation. "
                    "This indicates a solvation failure that would produce an unphysical system."
                )

            logger.info(f"[COMPLETE] Solvated system: {modeller.topology.getNumAtoms()} atoms")

            # Step 3b: Pre-minimize to resolve solvation clashes (constraint-free)
            logger.info("Step 3b: Pre-minimizing to resolve solvation clashes (constraint-free)...")
            try:
                pre_system = forcefield.createSystem(
                    modeller.topology,
                    nonbondedMethod=openmm.app.PME,
                    nonbondedCutoff=1.0 * unit.nanometer,
                    constraints=None,
                    rigidWater=False,
                )
                pre_integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
                pre_platform = Platform.getPlatformByName('CPU')
                pre_sim = Simulation(
                    modeller.topology, pre_system, pre_integrator, pre_platform
                )
                pre_sim.context.setPositions(modeller.positions)
                pre_sim.minimizeEnergy(maxIterations=500)
                modeller.positions = pre_sim.context.getState(
                    getPositions=True
                ).getPositions()
                pre_energy = pre_sim.context.getState(
                    getEnergy=True
                ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                logger.info(
                    f"[COMPLETE] Pre-minimization resolved clashes "
                    f"(energy={pre_energy:.1f} kJ/mol)"
                )
                del pre_sim, pre_system, pre_integrator
            except Exception as pre_err:
                logger.warning(f"Pre-minimization failed ({pre_err}), continuing with raw positions")

            # Step 4: Create OpenMM System (HMR + HBonds + vdW switching)
            logger.info("Step 4: Creating parameterized system (HMR, HBonds, switchDistance=0.8nm)...")
            openmm_system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=openmm.app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                switchDistance=0.8 * unit.nanometer,
                constraints=openmm.app.HBonds,
                rigidWater=True,
                hydrogenMass=4.0 * unit.amu,
            )

            # Enable long-range dispersion correction
            for force in openmm_system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    force.setUseDispersionCorrection(True)
                    force.setUseSwitchingFunction(True)
                    force.setSwitchingDistance(0.8 * unit.nanometer)
                    logger.info("[COMPLETE] Dispersion correction and vdW switching enabled")
                    break

            # Step 5: Add barostat
            logger.info("Step 5: Adding barostat...")
            barostat = MonteCarloBarostat(
                pressure * unit.bar,
                temperature * unit.kelvin,
                25
            )
            openmm_system.addForce(barostat)

            # Step 6: Create integrator (4 fs with HMR)
            logger.info("Step 6: Creating integrator (4 fs timestep with HMR)...")
            integrator = LangevinMiddleIntegrator(
                temperature * unit.kelvin,
                1.0 / unit.picosecond,
                0.004 * unit.picoseconds
            )

            # Step 7: Create simulation
            logger.info("Step 7: Creating simulation...")
            simulation, platform_name = self._create_simulation_with_fallback(
                modeller.topology, openmm_system, integrator
            )
            simulation.context.setPositions(modeller.positions)

            # Step 7b: Minimize with constraints
            logger.info("Step 7b: Minimizing with constraints to satisfy HBonds tolerances...")
            try:
                simulation.minimizeEnergy(maxIterations=100, tolerance=10.0)
                final_energy = simulation.context.getState(
                    getEnergy=True
                ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                logger.info(
                    f"[COMPLETE] Constraint-satisfying minimization complete "
                    f"(energy={final_energy:.1f} kJ/mol)"
                )
            except Exception as min_err:
                logger.warning(
                    f"Constraint-satisfying minimization failed ({min_err}), "
                    "proceeding with pre-minimized positions"
                )

            # Save system PDB
            system_pdb_path = os.path.join(self.output_dir, f"{system_id}_system.pdb")
            from ..utils.pdb_utils import write_pdb_file
            write_pdb_file(
                simulation.topology,
                simulation.context.getState(getPositions=True).getPositions(),
                system_pdb_path,
                keep_ids=True
            )

            logger.info("[COMPLETE] Protein-only solvated system created successfully")

            return {
                "status": "success",
                "simulation": simulation,
                "system_pdb_path": system_pdb_path,
                "total_atoms": modeller.topology.getNumAtoms(),
                "platform": platform_name,
                "system_info": {
                    "protein_atoms": len([a for a in modeller.topology.atoms()
                                         if a.residue.name not in ['HOH', 'NA', 'CL']]),
                    "water_molecules": len([r for r in modeller.topology.residues()
                                           if r.name == 'HOH']),
                    "ions": len([a for a in modeller.topology.atoms()
                                if a.residue.name in ['NA', 'CL']])
                }
            }

        except Exception as e:
            import traceback
            logger.error(f"Protein-only system creation failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def recreate_system_from_pdb_protein_only(
        self,
        system_pdb_data: str,
        system_id: str = "system",
        temperature: float = 300.0,
        pressure: float = 1.0
    ) -> Dict[str, Any]:
        """
        Recreate protein-only OpenMM system from an existing solvated PDB.

        Uses pure AMBER14 force fields — no template generator needed.

        Returns:
            Dict with system recreation results
        """
        import io
        from openmm.app import PDBFile, ForceField as OpenMMForceField
        from openmm import LangevinMiddleIntegrator, MonteCarloBarostat
        from openmm import unit
        import openmm

        logger.info("Recreating protein-only system from existing solvated PDB...")

        try:
            # Create AMBER14 force field
            forcefield = OpenMMForceField('amber14-all.xml', 'amber14/tip3p.xml')

            # Load PDB
            pdb_file = io.StringIO(system_pdb_data)
            pdb = PDBFile(pdb_file)
            logger.info(f"[COMPLETE] Loaded solvated system PDB: {pdb.topology.getNumAtoms()} atoms")

            # Create System (HMR + HBonds + vdW switching)
            logger.info("Creating OpenMM system (HMR, HBonds, switchDistance=0.8nm)...")
            openmm_system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=openmm.app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                switchDistance=0.8 * unit.nanometer,
                constraints=openmm.app.HBonds,
                rigidWater=True,
                hydrogenMass=4.0 * unit.amu
            )

            # Enable long-range dispersion correction
            for force in openmm_system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    force.setUseDispersionCorrection(True)
                    force.setUseSwitchingFunction(True)
                    force.setSwitchingDistance(0.8 * unit.nanometer)
                    logger.info("[COMPLETE] Dispersion correction and vdW switching enabled")
                    break

            # Configure integrator and barostat
            temp_unit = temperature * unit.kelvin
            friction = 1.0 / unit.picosecond
            step_size = 4.0 * unit.femtoseconds
            integrator = LangevinMiddleIntegrator(temp_unit, friction, step_size)

            press_unit = pressure * unit.bar
            barostat = MonteCarloBarostat(press_unit, temp_unit)
            openmm_system.addForce(barostat)

            # Create Simulation
            simulation, platform_name = self._create_simulation_with_fallback(
                pdb.topology, openmm_system, integrator
            )
            simulation.context.setPositions(pdb.positions)

            # Restore periodic box vectors
            box_vectors = pdb.topology.getPeriodicBoxVectors()
            if box_vectors:
                simulation.context.setPeriodicBoxVectors(*box_vectors)
                logger.info(f"[COMPLETE] Periodic box vectors restored")

            return {
                "status": "success",
                "simulation": simulation,
                "total_atoms": pdb.topology.getNumAtoms(),
                "platform": platform_name,
                "system_info": {
                    "total_atoms": pdb.topology.getNumAtoms(),
                    "residues": pdb.topology.getNumResidues(),
                    "chains": pdb.topology.getNumChains()
                }
            }

        except Exception as e:
            import traceback
            logger.error(f"Failed to recreate protein-only system from PDB: {e}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def _create_simulation_with_fallback(self, topology, system, integrator):
        """Create simulation with platform fallback."""
        from openmm import Platform
        from openmm.app import Simulation
        
        simulation = None
        platform_name = None
        
        for pname in ['CUDA', 'OpenCL', 'CPU']:
            try:
                logger.info(f"Attempting {pname} platform...")
                platform = Platform.getPlatformByName(pname)
                
                if pname == 'CUDA':
                    properties = {'Precision': 'mixed', 'CudaDeviceIndex': '0'}
                    simulation = Simulation(topology, system, integrator, platform, properties)
                elif pname == 'OpenCL':
                    properties = {'Precision': 'mixed'}
                    simulation = Simulation(topology, system, integrator, platform, properties)
                else:
                    simulation = Simulation(topology, system, integrator, platform)
                
                platform_name = pname
                logger.info(f"[COMPLETE] Using {pname} platform")
                break
                
            except Exception as e:
                logger.warning(f"Failed to initialize {pname}: {e}")
                continue
        
        if simulation is None:
            raise RuntimeError("Could not initialize simulation on any platform")
        
        return simulation, platform_name
    
