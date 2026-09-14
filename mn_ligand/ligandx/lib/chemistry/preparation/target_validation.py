"""Geometry and topology checks for published prepared targets.

These checks deliberately live at the target-preparation boundary.  Docking,
refolding, MD, and analysis workflows must all consume the same validated
target instead of applying engine-specific receptor repairs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import dist


@dataclass(frozen=True)
class TargetAtom:
    serial: int
    chain: str
    residue_number: int
    insertion_code: str
    residue_name: str
    atom_name: str
    element: str
    xyz: tuple[float, float, float]

    @property
    def residue_key(self) -> tuple[str, int, str]:
        return (self.chain, self.residue_number, self.insertion_code)


def _target_atoms(pdb_data: str) -> list[TargetAtom]:
    atoms: list[TargetAtom] = []
    for line in pdb_data.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        try:
            atom_name = line[12:16].strip()
            element = (line[76:78].strip() if len(line) >= 78 else "")
            element = (element or atom_name[:1]).upper()
            atoms.append(
                TargetAtom(
                    serial=int(line[6:11]),
                    chain=line[21].strip() or "_",
                    residue_number=int(line[22:26]),
                    insertion_code=line[26].strip() or "_",
                    residue_name=line[17:20].strip(),
                    atom_name=atom_name,
                    element=element,
                    xyz=(
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ),
                )
            )
        except (TypeError, ValueError):
            continue
    return atoms


def remove_internal_oxt(pdb_data: str) -> tuple[str, list[dict[str, object]]]:
    """Remove OXT records that are not on the last polymer residue of a chain."""
    atoms = _target_atoms(pdb_data)
    residue_order: dict[str, list[tuple[str, int, str]]] = {}
    for atom in atoms:
        keys = residue_order.setdefault(atom.chain, [])
        if atom.residue_key not in keys:
            keys.append(atom.residue_key)
    terminal_keys = {keys[-1] for keys in residue_order.values() if keys}
    removed_atoms = [
        atom
        for atom in atoms
        if atom.atom_name == "OXT" and atom.residue_key not in terminal_keys
    ]
    if not removed_atoms:
        return (pdb_data if pdb_data.endswith("\n") else pdb_data + "\n"), []
    serials = {atom.serial for atom in removed_atoms}
    kept = [
        line
        for line in pdb_data.splitlines()
        if not (
            line.startswith("ATOM  ")
            and line[6:11].strip().isdigit()
            and int(line[6:11]) in serials
        )
    ]
    return "\n".join(kept) + "\n", [asdict(atom) for atom in removed_atoms]


def audit_target_geometry(
    pdb_data: str,
    *,
    clash_distance_angstrom: float = 1.8,
) -> dict[str, object]:
    """Return severe non-bonded heavy-atom overlaps in a prepared target."""
    atoms = [atom for atom in _target_atoms(pdb_data) if atom.element != "H"]
    residue_order: dict[str, list[tuple[str, int, str]]] = {}
    for atom in atoms:
        keys = residue_order.setdefault(atom.chain, [])
        if atom.residue_key not in keys:
            keys.append(atom.residue_key)
    residue_positions = {
        key: index
        for keys in residue_order.values()
        for index, key in enumerate(keys)
    }
    clashes: list[dict[str, object]] = []
    residue_atoms = {
        (atom.residue_key, atom.atom_name): atom for atom in atoms
    }
    for keys in residue_order.values():
        for left_key, right_key in zip(keys, keys[1:]):
            carbon = residue_atoms.get((left_key, "C"))
            nitrogen = residue_atoms.get((right_key, "N"))
            if carbon is None or nitrogen is None:
                continue
            separation = dist(carbon.xyz, nitrogen.xyz)
            if 1.1 <= separation <= 1.6:
                continue
            clashes.append(
                {
                    "distance_angstrom": round(separation, 4),
                    "kind": "malformed_peptide_bond",
                    "left": asdict(carbon),
                    "right": asdict(nitrogen),
                }
            )
    for index, left in enumerate(atoms):
        for right in atoms[index + 1 :]:
            if left.residue_key == right.residue_key:
                continue
            separation = dist(left.xyz, right.xyz)
            if separation >= float(clash_distance_angstrom):
                continue
            adjacent = bool(
                left.chain == right.chain
                and abs(
                    residue_positions[left.residue_key]
                    - residue_positions[right.residue_key]
                )
                == 1
            )
            if adjacent:
                # Peptide C-N geometry was validated above. Other adjacent
                # pairs are handled by PDBFixer/OpenMM stereochemistry.
                continue
            if left.atom_name == right.atom_name == "SG" and separation <= 2.3:
                continue
            clashes.append(
                {
                    "distance_angstrom": round(separation, 4),
                    "left": asdict(left),
                    "right": asdict(right),
                }
            )
    return {
        "valid": not clashes,
        "clash_distance_angstrom": float(clash_distance_angstrom),
        "severe_clash_count": len(clashes),
        "severe_clashes": clashes,
    }


def prepare_target_for_publication(pdb_data: str) -> tuple[str, dict[str, object]]:
    """Apply unambiguous topology cleanup and reject invalid target geometry."""
    repaired, removed_oxt = remove_internal_oxt(pdb_data)
    audit = audit_target_geometry(repaired)
    report: dict[str, object] = {
        "removed_internal_oxt_count": len(removed_oxt),
        "removed_internal_oxt_atoms": removed_oxt,
        **audit,
    }
    if not audit["valid"]:
        examples = []
        for clash in list(audit["severe_clashes"])[:3]:
            left = clash["left"]
            right = clash["right"]
            examples.append(
                f"{left['chain']}:{left['residue_name']}{left['residue_number']}:{left['atom_name']}"
                f"-{right['chain']}:{right['residue_name']}{right['residue_number']}:{right['atom_name']}"
                f" ({clash['distance_angstrom']:.3f} A)"
            )
        raise ValueError(
            "Prepared target has severe non-bonded heavy-atom overlaps: "
            + "; ".join(examples)
        )
    return repaired, report
