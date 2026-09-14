from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import re
from typing import Any

for _thread_environment_key in (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_environment_key] = "1"

import gemmi
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

from mn_ligand.core.residue_mapping import sequence_author_residue_mapping


CONTACT_CUTOFF_ANGSTROM = 4.5
HYDROPHOBIC_CUTOFF_ANGSTROM = 4.0
HBOND_CUTOFF_ANGSTROM = 3.5
SALT_BRIDGE_CUTOFF_ANGSTROM = 4.0
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})
WATER_NAMES = frozenset({"HOH", "WAT", "DOD"})
HYDROPHOBIC_PROTEIN_ATOMS = {
    "ALA": {"CB"},
    "VAL": {"CB", "CG1", "CG2"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "MET": {"CB", "CG", "SD", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "PRO": {"CB", "CG", "CD"},
}
PROTEIN_DONORS = {
    "ARG": {"NE", "NH1", "NH2"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"},
    "HIP": {"ND1", "NE2"},
    "HSP": {"ND1", "NE2"},
    "LYS": {"NZ"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TRP": {"NE1"},
    "TYR": {"OH"},
}
PROTEIN_ACCEPTORS = {
    "ASN": {"OD1"},
    "ASP": {"OD1", "OD2"},
    "GLN": {"OE1"},
    "GLU": {"OE1", "OE2"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
}


def protein_atom_scope(atom_name: object) -> str:
    return "BB" if str(atom_name or "").strip().upper() in BACKBONE_ATOMS else "SC"


def _safe_id(value: object, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (text or fallback)[:160]


def _atom_name(atom: Chem.Atom) -> str:
    info = atom.GetPDBResidueInfo()
    if info is not None and info.GetName().strip():
        return info.GetName().strip()
    for key in ("_TriposAtomName", "atomName"):
        if atom.HasProp(key) and atom.GetProp(key).strip():
            return atom.GetProp(key).strip()
    return f"{atom.GetSymbol()}{atom.GetIdx() + 1}"


def _rdkit_ligand(path: Path) -> tuple[Chem.Mol, list[dict[str, Any]]]:
    molecule = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"RDKit could not read ligand coordinates from {path.name}")
    conformer = molecule.GetConformer()
    atoms = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        position = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append({
            "index": atom.GetIdx(),
            "name": _atom_name(atom),
            "element": atom.GetSymbol(),
            "formal_charge": int(atom.GetFormalCharge()),
            "xyz": np.asarray([position.x, position.y, position.z], dtype=float),
        })
    return molecule, atoms


def _complex_ligand(
    path: Path,
    row: dict[str, object],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    structure = gemmi.read_structure(str(path))
    if not structure or not structure[0]:
        raise ValueError(f"No coordinates in {path.name}")
    selected_name = str(row.get("ligand_residue_name") or "").strip().upper()
    selected_chain = str(row.get("ligand_chain") or "").strip()
    selected_number = str(row.get("ligand_residue_number") or "").strip()
    protein: list[dict[str, Any]] = []
    ligand: list[dict[str, Any]] = []
    waters: list[dict[str, Any]] = []
    for chain in structure[0]:
        for residue in chain:
            name = residue.name.strip().upper()
            info = gemmi.find_tabulated_residue(name)
            is_protein = info.is_amino_acid()
            matches_ligand = (
                (not selected_name or name == selected_name)
                and (not selected_chain or str(chain.name).strip() == selected_chain)
                and (
                    not selected_number
                    or str(residue.seqid.num) == selected_number
                )
            )
            target = protein if is_protein else ligand if matches_ligand else None
            if name in WATER_NAMES:
                target = waters
            if target is None:
                continue
            for atom in residue:
                if atom.element.name.upper() == "H":
                    continue
                target.append({
                    "name": atom.name.strip(),
                    "element": atom.element.name.title(),
                    "formal_charge": int(atom.charge or 0),
                    "xyz": np.asarray(
                        [atom.pos.x, atom.pos.y, atom.pos.z], dtype=float
                    ),
                    "chain": str(chain.name).strip(),
                    "residue_name": name,
                    "residue_number": int(residue.seqid.num),
                    "insertion_code": str(residue.seqid.icode).strip(),
                })
    if not ligand:
        raise ValueError("Selected ligand residue is unavailable in the complex")
    return protein, ligand, waters


def _receptor_atoms(path: Path) -> list[dict[str, Any]]:
    structure = gemmi.read_structure(str(path))
    atoms: list[dict[str, Any]] = []
    for chain in structure[0]:
        for residue in chain:
            name = residue.name.strip().upper()
            if not gemmi.find_tabulated_residue(name).is_amino_acid():
                continue
            for atom in residue:
                if atom.element.name.upper() == "H":
                    continue
                atoms.append({
                    "name": atom.name.strip(),
                    "element": atom.element.name.title(),
                    "formal_charge": 0,
                    "xyz": np.asarray(
                        [atom.pos.x, atom.pos.y, atom.pos.z], dtype=float
                    ),
                    "chain": str(chain.name).strip(),
                    "residue_name": name,
                    "residue_number": int(residue.seqid.num),
                    "insertion_code": str(residue.seqid.icode).strip(),
                })
    return atoms


def _ligand_features(molecule: Chem.Mol) -> tuple[set[int], set[int]]:
    factory = ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )
    donors: set[int] = set()
    acceptors: set[int] = set()
    for feature in factory.GetFeaturesForMol(molecule):
        family = feature.GetFamily()
        if family == "Donor":
            donors.update(int(value) for value in feature.GetAtomIds())
        elif family == "Acceptor":
            acceptors.update(int(value) for value in feature.GetAtomIds())
    return donors, acceptors


def _protein_charge(atom: dict[str, Any]) -> int:
    residue = str(atom["residue_name"]).upper()
    name = str(atom["name"]).upper()
    if residue == "ASP" and name in {"OD1", "OD2"}:
        return -1
    if residue == "GLU" and name in {"OE1", "OE2"}:
        return -1
    if residue == "LYS" and name == "NZ":
        return 1
    if residue == "ARG" and name in {"NE", "NH1", "NH2"}:
        return 1
    if residue in {"HIP", "HSP"} and name in {"ND1", "NE2"}:
        return 1
    return 0


def _protein_donor(atom: dict[str, Any]) -> bool:
    residue = str(atom["residue_name"]).upper()
    name = str(atom["name"]).upper()
    return name == "N" or name in PROTEIN_DONORS.get(residue, set())


def _protein_acceptor(atom: dict[str, Any]) -> bool:
    residue = str(atom["residue_name"]).upper()
    name = str(atom["name"]).upper()
    return name in {"O", "OXT"} or name in PROTEIN_ACCEPTORS.get(residue, set())


def _residue_key(atom: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(atom.get("chain") or ""),
        str(atom.get("residue_name") or ""),
        int(atom.get("residue_number") or 0),
        str(atom.get("insertion_code") or ""),
    )


def analyze_static_pose(
    protein_atoms: list[dict[str, Any]],
    ligand_atoms: list[dict[str, Any]],
    *,
    ligand_molecule: Chem.Mol | None = None,
) -> list[dict[str, Any]]:
    if not protein_atoms or not ligand_atoms:
        return []
    ligand_donors: set[int] = set()
    ligand_acceptors: set[int] = set()
    if ligand_molecule is not None:
        ligand_donors, ligand_acceptors = _ligand_features(ligand_molecule)
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for atom in protein_atoms:
        grouped.setdefault(_residue_key(atom), []).append(atom)
    rows: list[dict[str, Any]] = []
    for residue_key, residue_atoms in grouped.items():
        candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for protein_atom in residue_atoms:
            for ligand_atom in ligand_atoms:
                distance = float(
                    np.linalg.norm(protein_atom["xyz"] - ligand_atom["xyz"])
                )
                candidates.append((distance, protein_atom, ligand_atom))
        nearest = min(candidates, key=lambda item: item[0])
        interaction_candidates: list[
            tuple[str, float, dict[str, Any], dict[str, Any]]
        ] = []
        if nearest[0] <= CONTACT_CUTOFF_ANGSTROM:
            interaction_candidates.append(("contact", *nearest))
        for distance, protein_atom, ligand_atom in candidates:
            protein_name = str(protein_atom["name"]).upper()
            ligand_index = int(ligand_atom.get("index", -1))
            if (
                distance <= HYDROPHOBIC_CUTOFF_ANGSTROM
                and str(ligand_atom["element"]) in {"C", "S"}
                and protein_name
                in HYDROPHOBIC_PROTEIN_ATOMS.get(
                    str(protein_atom["residue_name"]).upper(), set()
                )
            ):
                interaction_candidates.append(
                    ("hydrophobic", distance, protein_atom, ligand_atom)
                )
            if (
                distance <= HBOND_CUTOFF_ANGSTROM
                and (
                    (
                        ligand_index in ligand_donors
                        and _protein_acceptor(protein_atom)
                    )
                    or (
                        ligand_index in ligand_acceptors
                        and _protein_donor(protein_atom)
                    )
                )
            ):
                interaction_candidates.append(
                    ("hydrogen bond", distance, protein_atom, ligand_atom)
                )
            protein_charge = _protein_charge(protein_atom)
            ligand_charge = int(ligand_atom.get("formal_charge") or 0)
            if (
                distance <= SALT_BRIDGE_CUTOFF_ANGSTROM
                and protein_charge * ligand_charge < 0
            ):
                interaction_candidates.append(
                    ("salt bridge", distance, protein_atom, ligand_atom)
                )
        best_by_type: dict[
            str, tuple[float, dict[str, Any], dict[str, Any]]
        ] = {}
        for kind, distance, protein_atom, ligand_atom in interaction_candidates:
            current = best_by_type.get(kind)
            if current is None or distance < current[0]:
                best_by_type[kind] = (distance, protein_atom, ligand_atom)
        for kind, (distance, protein_atom, ligand_atom) in best_by_type.items():
            chain, residue_name, residue_number, insertion_code = residue_key
            rows.append({
                "interaction_type": kind,
                "protein_chain": chain,
                "protein_residue_name": residue_name,
                "protein_residue_number": residue_number,
                "protein_insertion_code": insertion_code,
                "protein_atom_name": protein_atom["name"],
                "protein_atom_scope": protein_atom_scope(protein_atom["name"]),
                "coordinate_protein_chain": protein_atom.get(
                    "coordinate_chain", chain
                ),
                "coordinate_protein_residue_number": protein_atom.get(
                    "coordinate_residue_number", residue_number
                ),
                "coordinate_protein_insertion_code": protein_atom.get(
                    "coordinate_insertion_code", insertion_code
                ),
                "ligand_atom_name": ligand_atom["name"],
                "distance_angstrom": round(distance, 3),
                "angle_degree": "",
                "present": True,
                "native_fields_json": json.dumps({
                    "method": "app-native static geometry",
                    "protein_atom": protein_atom["name"],
                    "protein_atom_scope": protein_atom_scope(
                        protein_atom["name"]
                    ),
                    "coordinate_protein_chain": protein_atom.get(
                        "coordinate_chain", chain
                    ),
                    "coordinate_protein_residue_number": protein_atom.get(
                        "coordinate_residue_number", residue_number
                    ),
                    "coordinate_protein_insertion_code": protein_atom.get(
                        "coordinate_insertion_code", insertion_code
                    ),
                    "ligand_atom": ligand_atom["name"],
                }, sort_keys=True),
            })
    return rows


def apply_author_residue_numbering(
    protein_atoms: list[dict[str, Any]],
    structure_pdb_data: str,
    author_pdb_data: str,
) -> int:
    mapping = sequence_author_residue_mapping(
        structure_pdb_data,
        author_pdb_data,
    )
    mapped_count = 0
    for atom in protein_atoms:
        coordinate_chain = str(atom.get("chain") or "")
        coordinate_number = int(atom.get("residue_number") or 0)
        coordinate_insertion = str(atom.get("insertion_code") or "")
        atom["coordinate_chain"] = coordinate_chain
        atom["coordinate_residue_number"] = coordinate_number
        atom["coordinate_insertion_code"] = coordinate_insertion
        author = mapping.get((
            coordinate_chain or "_",
            coordinate_number,
            coordinate_insertion,
        ))
        if author is None:
            continue
        atom["chain"] = author["chain"]
        atom["residue_number"] = author["residue_number"]
        atom["insertion_code"] = author["insertion_code"]
        atom["residue_name"] = author["residue_name"]
        mapped_count += 1
    return mapped_count


def _analyze(
    item: tuple[int, dict[str, object]],
    workspace: Path,
    prepared: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    index, row = item
    pose_id = _safe_id(row.get("pose_id"), f"pose_{index:07d}")
    metadata = {
        key: row.get(key, "")
        for key in (
            "pose_id", "compound_id", "source_engine", "source_kind",
            "replicate", "prediction", "selection_criterion",
            "ligand_chain", "ligand_residue_name",
            "ligand_residue_number", "ligand_insertion_code",
            "ligand_heavy_atom_count",
        )
    }
    metadata["pose_id"] = pose_id
    try:
        molecule: Chem.Mol | None = None
        complex_file = str(row.get("complex_file") or "").strip()
        if complex_file:
            source = workspace / complex_file
            protein, ligand, _waters = _complex_ligand(source, row)
            target = prepared / f"{pose_id}.complex.pdb"
            structure = gemmi.read_structure(str(source))
            target.write_text(structure.make_pdb_string())
            topology_path = str(row.get("ligand_topology") or "").strip()
            if topology_path:
                molecule, topology_atoms = _rdkit_ligand(
                    workspace / topology_path
                )
                if len(topology_atoms) == len(ligand):
                    for target_atom, topology_atom in zip(
                        ligand, topology_atoms, strict=True
                    ):
                        target_atom["index"] = topology_atom["index"]
                        target_atom["formal_charge"] = topology_atom[
                            "formal_charge"
                        ]
                        target_atom["name"] = topology_atom["name"]
        else:
            receptor = workspace / str(row.get("mol_cond") or "")
            ligand_path = workspace / str(row.get("mol_pred") or "")
            protein = _receptor_atoms(receptor)
            molecule, ligand = _rdkit_ligand(ligand_path)
            target = prepared / f"{pose_id}.complex.pdb"
            receptor_lines = [
                line
                for line in receptor.read_text(
                    errors="replace"
                ).splitlines()
                if line.startswith(("ATOM  ", "HETATM"))
            ]
            ligand_lines = []
            serial = len(receptor_lines) + 1
            for line in Chem.MolToPDBBlock(molecule).splitlines():
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                line = f"HETATM{serial:5d}" + line[11:]
                ligand_lines.append(
                    line[:17]
                    + "LIG"
                    + line[20:21]
                    + "Z"
                    + f"{1:4d}"
                    + line[26:]
                )
                serial += 1
            target.write_text(
                "\n".join(receptor_lines + ligand_lines) + "\nEND\n"
            )
            source = receptor
        reference_target = str(row.get("reference_target") or "").strip()
        if reference_target:
            apply_author_residue_numbering(
                protein,
                source.read_text(errors="replace"),
                (workspace / reference_target).read_text(errors="replace"),
            )
        pose_rows = [
            {**metadata, **detail}
            for detail in analyze_static_pose(
                protein, ligand, ligand_molecule=molecule
            )
        ]
        residues = {
            (
                row["protein_chain"],
                row["protein_residue_name"],
                row["protein_residue_number"],
            )
            for row in pose_rows
        }
        counts: dict[str, int] = {}
        for detail in pose_rows:
            kind = str(detail["interaction_type"])
            counts[kind] = counts.get(kind, 0) + 1
        return pose_rows, {
            **metadata,
            "success": True,
            "interaction_count": len(pose_rows),
            "contacted_residue_count": len(residues),
            "contacted_residues": "; ".join(
                f"{chain}:{name}{number}"
                for chain, name, number in sorted(residues)
            ),
            "interaction_counts_json": json.dumps(counts, sort_keys=True),
        }, None
    except Exception as exc:
        failure = {**metadata, "error": str(exc)}
        return [], {
            **metadata,
            "success": False,
            "interaction_count": 0,
            "contacted_residue_count": 0,
            "contacted_residues": "",
            "interaction_counts_json": "{}",
            "error": str(exc),
        }, failure


def run(
    input_path: Path,
    native_dir: Path,
    summary_path: Path,
    interactions_path: Path,
    report_path: Path,
    *,
    max_workers: int,
) -> int:
    workspace = input_path.resolve().parents[1]
    native_dir.mkdir(parents=True, exist_ok=True)
    prepared = workspace / "prepared"
    prepared.mkdir(exist_ok=True)
    with input_path.open(newline="") as handle:
        inputs = list(csv.DictReader(handle))
    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        for pose_rows, summary, failure in executor.map(
            lambda item: _analyze(item, workspace, prepared),
            enumerate(inputs, start=1),
        ):
            details.extend(pose_rows)
            summaries.append(summary)
            if failure is not None:
                failures.append(failure)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    columns = [
        "pose_id", "compound_id", "source_engine", "source_kind", "replicate",
        "prediction", "selection_criterion", "ligand_chain",
        "ligand_residue_name", "ligand_residue_number",
        "ligand_insertion_code", "ligand_heavy_atom_count",
        "interaction_type", "protein_chain", "protein_residue_name",
        "protein_residue_number", "protein_insertion_code",
        "protein_atom_name", "protein_atom_scope", "ligand_atom_name",
        "coordinate_protein_chain",
        "coordinate_protein_residue_number",
        "coordinate_protein_insertion_code",
        "distance_angstrom", "angle_degree", "present",
        "native_fields_json",
    ]
    pd.DataFrame(details, columns=columns).to_csv(interactions_path, index=False)
    report_path.write_text(json.dumps({
        "engine": "Native MD geometry",
        "expected_count": len(inputs),
        "analyzed_count": len(inputs) - len(failures),
        "failed_count": len(failures),
        "interaction_count": len(details),
        "failures": failures,
        "static_pose_analysis": True,
        "occupancy_semantics": False,
        "water_bridges_assessed": False,
        "protein_atom_scope": "BB/SC",
        "method": (
            "Single-frame geometric adaptation of the app MD contact analysis"
        ),
    }, indent=2) + "\n")
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--native-output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--interactions", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    return run(
        args.input,
        args.native_output,
        args.summary,
        args.interactions,
        args.report,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
