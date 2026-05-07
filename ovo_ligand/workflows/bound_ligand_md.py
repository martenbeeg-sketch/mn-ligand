#!/usr/bin/env python
"""Run the first bound-ligand MD workflow inside the MD Docker image.

This wrapper keeps the user-facing app small while using the vendored
Ligand-X-derived MD and structure modules that live inside ovo-ligand.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
import warnings
from pathlib import Path
from typing import Any


WATER = {"HOH", "WAT", "H2O", "TIP", "TIP3", "TIP4"}
COMMON_IONS = {
    "NA", "MG", "K", "CA", "MN", "FE", "CO", "NI", "CU", "ZN",
    "CD", "HG", "CL", "BR", "I", "F", "LI", "BE", "AL", "TL", "PB",
}

MODIFIED_RESIDUE_MAPPINGS = {
    "CAS": {
        "target": "CYS",
        "keep_atoms": {"N", "CA", "C", "O", "CB", "SG"},
        "description": "CAS mapped to CYS by keeping protein-compatible atoms and dropping arsenic substituent atoms.",
    }
}


def download_pdb(pdb_id: str) -> str:
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError("PDB ID must be exactly 4 alphanumeric characters")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read().decode("utf-8")
    if "ATOM" not in data and "HETATM" not in data:
        raise ValueError(f"Downloaded file for {pdb_id} does not look like a PDB")
    return data


def map_modified_residues_to_standard(pdb_data: str) -> tuple[str, dict[str, Any]]:
    """Map selected modified amino acids to standard residues before cleaning.

    This is intentionally conservative: only residue-specific atom names listed in
    MODIFIED_RESIDUE_MAPPINGS are kept, and alternate locations are collapsed to
    the highest-occupancy atom for each residue/atom name.
    """
    mapped_groups: dict[tuple[str, str, str, str, str], list[str]] = {}
    passthrough_lines = []
    dropped_atoms: dict[str, int] = {}

    for line in pdb_data.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
            passthrough_lines.append(line)
            continue

        resname = line[17:20].strip()
        mapping = MODIFIED_RESIDUE_MAPPINGS.get(resname)
        if not mapping:
            passthrough_lines.append(line)
            continue

        atom_name = line[12:16].strip()
        residue_key = "|".join([resname, line[21].strip() or "_", line[22:26].strip(), line[26].strip() or "_"])
        if atom_name not in mapping["keep_atoms"]:
            dropped_atoms[residue_key] = dropped_atoms.get(residue_key, 0) + 1
            continue

        group_key = (resname, line[21], line[22:26], line[26], atom_name)
        mapped_groups.setdefault(group_key, []).append(line)

    mapped_lines_by_original_order: list[tuple[int, str]] = []
    mapping_counts: dict[str, int] = {}
    for group_lines in mapped_groups.values():
        selected_line = max(group_lines, key=_atom_occupancy)
        residue_key = "|".join(
            [
                selected_line[17:20].strip(),
                selected_line[21].strip() or "_",
                selected_line[22:26].strip(),
                selected_line[26].strip() or "_",
            ]
        )
        target = MODIFIED_RESIDUE_MAPPINGS[selected_line[17:20].strip()]["target"]
        converted = "ATOM  " + selected_line[6:16] + " " + f"{target:>3}" + selected_line[20:]
        mapping_counts[residue_key] = mapping_counts.get(residue_key, 0) + 1
        mapped_lines_by_original_order.append((pdb_data.find(selected_line), converted))

    mapped_iter = iter(line for _, line in sorted(mapped_lines_by_original_order, key=lambda item: item[0]))
    output_lines = []
    emitted_groups = set()
    for line in pdb_data.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
            output_lines.append(line)
            continue
        resname = line[17:20].strip()
        mapping = MODIFIED_RESIDUE_MAPPINGS.get(resname)
        if not mapping:
            output_lines.append(line)
            continue
        atom_name = line[12:16].strip()
        if atom_name not in mapping["keep_atoms"]:
            continue
        group_key = (resname, line[21], line[22:26], line[26], atom_name)
        if group_key in emitted_groups:
            continue
        emitted_groups.add(group_key)
        output_lines.append(next(mapped_iter))

    report = {
        "enabled": True,
        "mappings": {
            key: {
                "target": MODIFIED_RESIDUE_MAPPINGS[key.split("|")[0]]["target"],
                "kept_atoms": kept,
                "dropped_atoms": dropped_atoms.get(key, 0),
            }
            for key, kept in sorted(mapping_counts.items())
        },
    }
    return "\n".join(output_lines) + "\n", report


def _atom_occupancy(line: str) -> tuple[float, int]:
    try:
        occupancy = float(line[54:60])
    except ValueError:
        occupancy = 0.0
    altloc = line[16].strip()
    altloc_rank = 2 if not altloc else 1 if altloc == "A" else 0
    return occupancy, altloc_rank


def ligand_key(resname: str, chain: str, resseq: str, icode: str) -> str:
    return "|".join([resname.strip(), chain.strip() or "_", resseq.strip(), icode.strip() or "_"])


def parse_bound_ligands(pdb_data: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for line in pdb_data.splitlines():
        if not line.startswith("HETATM"):
            continue
        resname = line[17:20].strip()
        if resname in WATER or resname in COMMON_IONS:
            continue
        chain = line[21].strip() or "_"
        resseq = line[22:26].strip()
        icode = line[26].strip() or "_"
        key = ligand_key(resname, chain, resseq, icode)
        atom_name = line[12:16].strip()
        element = line[76:78].strip() or atom_name[:1]
        item = grouped.setdefault(
            key,
            {
                "key": key,
                "resname": resname,
                "chain": chain,
                "resseq": resseq,
                "icode": icode,
                "atom_count": 0,
                "heavy_atom_count": 0,
                "center": [0.0, 0.0, 0.0],
            },
        )
        item["atom_count"] += 1
        if element.upper() != "H":
            item["heavy_atom_count"] += 1
        try:
            coords = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            for idx, value in enumerate(coords):
                item["center"][idx] += value
        except ValueError:
            pass

    ligands = []
    for item in grouped.values():
        if item["atom_count"]:
            item["center"] = [round(value / item["atom_count"], 3) for value in item["center"]]
        ligands.append(item)
    return sorted(ligands, key=lambda x: (-x["heavy_atom_count"], x["resname"], x["chain"], x["resseq"]))


def extract_ligand_pdb(pdb_data: str, selected_key: str) -> str:
    lines = []
    for line in pdb_data.splitlines():
        if not line.startswith("HETATM"):
            continue
        key = ligand_key(line[17:20], line[21], line[22:26], line[26])
        if key == selected_key:
            lines.append(line)
    if not lines:
        raise ValueError(f"Selected ligand was not found in PDB data: {selected_key}")
    return "\n".join(lines + ["END", ""])


def _fetch_ccd_smiles(resname: str) -> str:
    resname = (resname or "").upper().strip()
    if not resname:
        return ""
    try:
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{resname}"
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        for desc in data.get("rcsb_chem_comp_descriptor", {}).get("descriptors", []):
            if desc.get("type") == "SMILES" and desc.get("program") == "OpenEye OEToolkits":
                return desc.get("descriptor", "") or ""
        for desc in data.get("pdbx_chem_comp_descriptor", []):
            dtype = (desc.get("type") or "").upper()
            if "SMILES" in dtype and "STEREO" not in dtype:
                return desc.get("descriptor", "") or ""
    except Exception:
        return ""
    return ""


def _build_ligand_sdf_artifacts(
    ligand_pdb: str,
    ligand_resname: str,
    output_dir: Path,
    file_prefix: str,
    reference_smiles: str = "",
) -> dict[str, str]:
    """
    Create HiQBind-style ligand artifacts:
      - ref.smi
      - raw.sdf (from extracted ligand PDB)
      - refined.sdf (bond orders from reference SMILES when possible)
    """
    artifacts: dict[str, str] = {}
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
    except Exception:
        return artifacts

    # HiQBind-style precedence:
    # 1) explicit reference SMILES provided by user/workflow
    # 2) fallback to CCD/OpenEye SMILES lookup
    ref_smi = (reference_smiles or "").strip()
    if not ref_smi:
        ref_smi = _fetch_ccd_smiles(ligand_resname)
    if ref_smi:
        ref_smi_path = output_dir / f"{file_prefix}_ligand_ref.smi"
        ref_smi_path.write_text(ref_smi + "\n")
        artifacts["ligand_ref_smi"] = str(ref_smi_path)

    pdb_path = output_dir / f"{file_prefix}_ligand_raw.pdb"
    pdb_path.write_text(ligand_pdb if ligand_pdb.endswith("\n") else ligand_pdb + "\n")
    artifacts["ligand_raw_pdb"] = str(pdb_path)

    raw_mol = Chem.MolFromPDBBlock(ligand_pdb, removeHs=False)
    if raw_mol is None:
        return artifacts

    raw_sdf_path = output_dir / f"{file_prefix}_ligand_raw.sdf"
    with Chem.SDWriter(str(raw_sdf_path)) as writer:
        writer.write(raw_mol)
    artifacts["ligand_raw_sdf"] = str(raw_sdf_path)

    # HiQBind-style ligand fix logic (ported):
    # sanitize -> remove H -> reference normalization -> template mapping -> identity check
    fix_err = ""
    has_ref = False
    try:
        Chem.SanitizeMol(raw_mol)
    except Exception as exc:
        fix_err = f"Sanitize failed: {exc}"
    working = Chem.RemoveAllHs(raw_mol) if not fix_err else raw_mol

    ref_mol = None
    if ref_smi:
        try:
            ref_mol = Chem.MolFromSmiles(ref_smi, sanitize=False)
            _fix_valence(ref_mol)
            ref_smi = Chem.MolToSmiles(ref_mol, kekuleSmiles=True, isomericSmiles=False)
            ref_mol = Chem.MolFromSmiles(ref_smi)
            has_ref = ref_mol is not None
        except Exception:
            has_ref = False

    if has_ref and not fix_err:
        if working.GetNumHeavyAtoms() != ref_mol.GetNumHeavyAtoms():
            fix_err = "Number of atoms not match"
        else:
            try:
                working = _assign_bond_orders_from_template(ref_mol, working)
            except Exception:
                flat = _reconstruct_mol(working)
                if flat is not None:
                    try:
                        working = _assign_bond_orders_from_template(ref_mol, flat)
                    except Exception as exc:
                        fix_err = f"Fix failed: {exc}"
                else:
                    fix_err = "Fix failed: reconstruction failed"
        if not fix_err:
            status = _is_same_molecule(working, ref_mol)
            artifacts["ligand_fix_identity_status"] = int(status)
            if status == 3:
                fix_err = f"NOT same after fix. Error code: {status}"
            elif status == 2:
                artifacts["ligand_fix_warning"] = (
                    "Ligand fix identity status=2 (same formula/bond count, non-identical graph). Continuing with warning."
                )

    refined = Chem.AddHs(working, addCoords=True) if working is not None else None
    if refined is None:
        refined = raw_mol

    refined_sdf_path = output_dir / f"{file_prefix}_ligand_refined.sdf"
    with Chem.SDWriter(str(refined_sdf_path)) as writer:
        writer.write(refined)
    artifacts["ligand_refined_sdf"] = str(refined_sdf_path)
    artifacts["reference_smiles_source"] = "provided" if reference_smiles.strip() else ("ccd" if ref_smi else "none")
    artifacts["reference_smiles_found"] = bool(has_ref)
    artifacts["ligand_fix_error"] = fix_err

    if fix_err and has_ref:
        try:
            ref_mol_noh = Chem.RemoveHs(ref_mol)
            AllChem.Compute2DCoords(ref_mol_noh)
            mol_noh = Chem.RemoveHs(working if working is not None else raw_mol)
            AllChem.Compute2DCoords(mol_noh)
            img = Draw.MolsToGridImage(
                [ref_mol_noh, mol_noh],
                legends=[f"{file_prefix} Ref", f"{file_prefix} Fixed"],
                subImgSize=(500, 500),
                returnPNG=True,
            )
            png_path = output_dir / f"{file_prefix}_ligand_fix_compare.png"
            png_path.write_bytes(img)
            artifacts["ligand_fix_compare_png"] = str(png_path)
        except Exception:
            pass

    if not has_ref:
        raise ValueError("No reference found for ligand refinement")
    if fix_err:
        raise ValueError(fix_err)
    return artifacts


def _fix_valence(mol) -> None:
    from rdkit import Chem
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "B" and len(atom.GetNeighbors()) == 4:
            atom.SetFormalCharge(-1)
        if atom.GetSymbol() == "N" and len(atom.GetNeighbors()) == 4:
            atom.SetFormalCharge(1)
    Chem.SanitizeMol(mol)


def _get_num_bonds_noh(mol) -> int:
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetSymbol() != "H" and bond.GetEndAtom().GetSymbol() != "H":
            count += 1
    return count


def _get_formula_noh(mol) -> str:
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        s = atom.GetSymbol()
        if s == "H":
            continue
        counts[s] = counts.get(s, 0) + 1
    return "".join(f"{k}{counts[k]}" for k in sorted(counts))


def _is_same_molecule(mol, ref_mol) -> int:
    from rdkit import Chem
    key = Chem.MolToInchiKey(mol)
    ref_key = Chem.MolToInchiKey(ref_mol)
    smi = Chem.MolToSmiles(mol)
    ref_smi = Chem.MolToSmiles(ref_mol)
    if key == ref_key:
        return 0 if smi == ref_smi else 1
    if key[:-1] == ref_key[:-1]:
        return 1
    if _get_formula_noh(mol) == _get_formula_noh(ref_mol) and _get_num_bonds_noh(mol) == _get_num_bonds_noh(ref_mol):
        return 2
    return 3


def _reconstruct_mol(mol):
    from rdkit import Chem
    rw = Chem.RWMol()
    mapping = {}
    pos = []
    conf = mol.GetConformer()
    nxt = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        na = Chem.Atom(atom.GetAtomicNum())
        na.SetFormalCharge(atom.GetFormalCharge())
        rw.AddAtom(na)
        mapping[atom.GetIdx()] = nxt
        p = conf.GetAtomPosition(atom.GetIdx())
        pos.append([float(p.x), float(p.y), float(p.z)])
        nxt += 1
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in mapping and j in mapping:
            rw.AddBond(mapping[i], mapping[j], Chem.BondType.SINGLE)
    nm = rw.GetMol()
    c = Chem.Conformer(len(pos))
    for i, p in enumerate(pos):
        c.SetAtomPosition(i, p)
    nm.AddConformer(c)
    try:
        Chem.SanitizeMol(nm)
    except Exception:
        return None
    return nm


def _assign_bond_orders_from_template(refmol, mol):
    from rdkit import Chem
    refmol2 = Chem.Mol(refmol)
    mol2 = Chem.Mol(mol)
    matching = mol2.GetSubstructMatch(refmol2)
    if not matching:
        for b in mol2.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        for b in refmol2.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        for a in refmol2.GetAtoms():
            a.SetFormalCharge(0)
        for a in mol2.GetAtoms():
            a.SetFormalCharge(0)
    matches = mol2.GetSubstructMatches(refmol2, uniquify=False)
    if not matches:
        raise ValueError("No matching found")
    if len(matches) > 1:
        warnings.warn("More than one matching pattern found - picking one")
    m = matches[0]
    for b in refmol.GetBonds():
        a1 = m[b.GetBeginAtomIdx()]
        a2 = m[b.GetEndAtomIdx()]
        b2 = mol2.GetBondBetweenAtoms(a1, a2)
        b2.SetBondType(b.GetBondType())
        b2.SetIsAromatic(b.GetIsAromatic())
    for a in refmol.GetAtoms():
        a2 = mol2.GetAtomWithIdx(m[a.GetIdx()])
        a2.SetHybridization(a.GetHybridization())
        a2.SetIsAromatic(a.GetIsAromatic())
        num_hs = max(0, a.GetNumExplicitHs() + a.GetNumImplicitHs() - len([n for n in a2.GetNeighbors() if n.GetSymbol() == "H"]))
        a2.SetNumExplicitHs(num_hs)
        a2.SetFormalCharge(a.GetFormalCharge())
    Chem.SanitizeMol(mol2)
    for atom in mol2.GetAtoms():
        atom.SetNumRadicalElectrons(0)
    mol2.UpdatePropertyCache()
    return mol2


def write_discovery(pdb_id: str, output_path: Path) -> dict[str, Any]:
    pdb_data = download_pdb(pdb_id)
    ligands = parse_bound_ligands(pdb_data)
    output = {"pdb_id": pdb_id.upper(), "ligands": ligands, "pdb_data": pdb_data}
    output_path.write_text(json.dumps(output, indent=2))
    return output


def prepare_structure(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    from ovo_ligand.ligandx.services.structure.processor import StructureProcessor

    pdb_id = config.get("pdb_id", "protein").upper()
    raw_pdb_data = config.get("pdb_data") or download_pdb(pdb_id)
    if config.get("map_modified_residues", True):
        structure_input, mapping_report = map_modified_residues_to_standard(raw_pdb_data)
    else:
        structure_input = raw_pdb_data
        mapping_report = {"enabled": False, "mappings": {}}
    processor = StructureProcessor()
    processed = processor.process_structure_with_ligands(
        structure_input,
        clean_protein=bool(config.get("clean_protein", True)),
        include_2d_images=False,
        target_pdb_id=pdb_id,
    )
    prepared_pdb_data = processed.get("processed_structure") or raw_pdb_data
    output = {
        "success": True,
        "pdb_id": pdb_id,
        "raw_pdb_data": raw_pdb_data,
        "mapped_input_pdb_data": structure_input,
        "prepared_pdb_data": prepared_pdb_data,
        "ligands": parse_bound_ligands(prepared_pdb_data),
        "protein_cleaned": bool(processed.get("protein_cleaned", False)),
        "components": processed.get("components", {}),
        "modified_residue_mapping": mapping_report,
    }
    output_path.write_text(json.dumps(output, indent=2))
    return output


def run_ligandx_md(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    from ovo_ligand.ligandx.services.md.config import MDOptimizationConfig
    from ovo_ligand.ligandx.services.md.service import MDOptimizationService

    pdb_data = config.get("pdb_data")
    if not pdb_data:
        input_complex_pdb_path = str(config.get("input_complex_pdb_path", "")).strip()
        if input_complex_pdb_path:
            pdb_data = Path(input_complex_pdb_path).read_text()
    if not pdb_data:
        pdb_data = download_pdb(config["pdb_id"])
    selected_key = config["ligand_key"]
    selected = next((lig for lig in parse_bound_ligands(pdb_data) if lig["key"] == selected_key), None)
    provided_refined_sdf = str(config.get("ligand_refined_sdf_data", "") or "")
    strict_refined = bool(config.get("strict_refined_ligand", False))
    if not selected and not (strict_refined and provided_refined_sdf):
        raise ValueError(f"Could not find selected ligand {selected_key}")
    if not selected:
        parts = selected_key.split("|")
        selected = {
            "key": selected_key,
            "resname": parts[0] if len(parts) > 0 else "LIG",
            "chain": parts[1] if len(parts) > 1 else "_",
            "resseq": parts[2] if len(parts) > 2 else "?",
            "icode": parts[3] if len(parts) > 3 else "_",
            "atom_count": 0,
            "heavy_atom_count": 0,
            "center": [0.0, 0.0, 0.0],
        }

    ligand_pdb = ""
    try:
        ligand_pdb = extract_ligand_pdb(pdb_data, selected_key)
    except Exception:
        if not (strict_refined and provided_refined_sdf):
            raise
    output_dir = Path(config.get("output_dir", "/output/md_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = config.get("job_id") or f"{config.get('pdb_id', 'pdb').lower()}_{selected['resname'].lower()}"
    file_prefix = f"{config.get('pdb_id', 'pdb').lower()}_{selected['resname'].lower()}"

    strict_refined = bool(config.get("strict_refined_ligand", False))
    provided_refined_sdf = str(config.get("ligand_refined_sdf_data", "") or "")
    prep_artifacts: dict[str, str] = {}
    downloaded_pdb_path: Path | None = None
    # If strict refined ligand is provided by structure-prep, do not regenerate prep artifacts in MD run folder.
    if not (strict_refined and provided_refined_sdf):
        downloaded_pdb_path = output_dir / f"{file_prefix}_downloaded.pdb"
        downloaded_pdb_path.write_text(pdb_data if pdb_data.endswith("\n") else pdb_data + "\n")
        prep_artifacts = _build_ligand_sdf_artifacts(
            ligand_pdb,
            selected["resname"],
            output_dir,
            file_prefix,
            reference_smiles=str(config.get("reference_smiles", "") or config.get("ref_smi", "")),
        )

    ligand_structure_data = ligand_pdb
    ligand_data_format = "pdb"
    ligand_input_source = "extracted_ligand_pdb"
    # Prefer caller-provided refined SDF data from structure-preparation jobs.
    # This keeps MD input consistent with already prepared ligand artifacts.
    if provided_refined_sdf.strip():
        # Ligand-X structure parser for `sdf` currently consumes a single MOL block.
        # Normalize SDF payloads to first record to avoid parse failures on trailing "$$$$".
        ligand_structure_data = provided_refined_sdf
        if "$$$$" in ligand_structure_data:
            ligand_structure_data = ligand_structure_data.split("$$$$", 1)[0].rstrip() + "\n"
        ligand_data_format = "sdf"
        ligand_input_source = "provided_refined_sdf_data"
    else:
        refined_sdf_path = prep_artifacts.get("ligand_refined_sdf")
        if refined_sdf_path:
            try:
                ligand_structure_data = Path(refined_sdf_path).read_text()
                if "$$$$" in ligand_structure_data:
                    ligand_structure_data = ligand_structure_data.split("$$$$", 1)[0].rstrip() + "\n"
                ligand_data_format = "sdf"
                ligand_input_source = "generated_refined_sdf"
            except Exception:
                pass

    def _build_md_config(_ligand_structure_data: str, _ligand_data_format: str) -> MDOptimizationConfig:
        return MDOptimizationConfig.from_dict(
            {
                "protein_pdb_data": pdb_data,
                "ligand_structure_data": _ligand_structure_data,
                "ligand_data_format": _ligand_data_format,
                "preserve_ligand_pose": True,
                "generate_conformer": False,
                "protein_id": config.get("pdb_id", "protein").lower(),
                "ligand_id": selected["resname"],
                "system_id": config.get("system_id", f"{config.get('pdb_id', 'complex').lower()}_{selected['resname'].lower()}"),
                "job_id": job_id,
                "charge_method": config.get("charge_method", "gasteiger"),
                "forcefield_method": config.get("forcefield_method", "openff-2.2.0"),
                "box_shape": config.get("box_shape", "dodecahedron"),
                "nvt_steps": int(config.get("nvt_steps", 2500)),
                "npt_steps": int(config.get("npt_steps", 2500)),
                "heating_steps_per_stage": int(config.get("heating_steps_per_stage", 250)),
                "production_steps": int(config.get("production_steps", 0)),
                "production_report_interval": int(config.get("production_report_interval", 2500)),
                "temperature": float(config.get("temperature", 300.0)),
                "pressure": float(config.get("pressure", 1.0)),
                "ionic_strength": float(config.get("ionic_strength", 0.15)),
                "padding_nm": float(config.get("padding_nm", 1.0)),
                "minimization_max_iterations": int(config.get("minimization_max_iterations", 5000)),
                "minimization_tolerance_kjmol_nm": float(config.get("minimization_tolerance_kjmol_nm", 10.0)),
                "heating_start_temperature": float(config.get("heating_start_temperature", 50.0)),
                "heating_stages": int(config.get("heating_stages", 6)),
                "npt_restraint_release_scales": str(config.get("npt_restraint_release_scales", "1.0,0.5,0.2,0.05,0.0")),
                "npt_release_enabled": bool(config.get("npt_release_enabled", True)),
                "protein_npt_release_scales": str(config.get("protein_npt_release_scales", "1.0,0.5,0.1,0.01,0.0")),
                "planarity_npt_release_scales": str(config.get("planarity_npt_release_scales", "1.0,0.5,0.2,0.05,0.0")),
                "allow_restrained_production": bool(config.get("allow_restrained_production", False)),
                "force_unrestrained_production": bool(config.get("force_unrestrained_production", True)),
                "resume_from_checkpoint_path": config.get("resume_from_checkpoint_path"),
                "resume_system_pdb_path": config.get("resume_system_pdb_path"),
                "minimization_only": bool(config.get("minimization_only", False)),
            }
        )

    # Optional runtime overrides for ligand restraints in system construction.
    os.environ["OVO_LIGAND_ENABLE_LIGAND_RESTRAINTS"] = "1" if bool(config.get("ligand_restraints_enabled", config.get("apply_ligand_restraints_during_heating_nvt", True))) else "0"
    os.environ["OVO_LIGAND_ENABLE_PROTEIN_RESTRAINTS"] = "1" if bool(config.get("apply_protein_restraints_during_heating_nvt", True)) else "0"
    os.environ["OVO_LIGAND_PROTEIN_RESTRAINT_SELECTION"] = str(config.get("protein_restraint_selection", "backbone"))
    os.environ["OVO_LIGAND_PROTEIN_RESTRAINT_K_KJMOL_NM2"] = str(config.get("protein_restraint_k", 1000.0))
    os.environ["OVO_LIGAND_ENABLE_PLANARITY_RESTRAINTS"] = "1" if bool(config.get("enable_ligand_planarity_restraints", False)) else "0"
    if "ligand_lock_k_kjmol_nm2" in config:
        os.environ["OVO_LIGAND_LOCK_K_KJMOL_NM2"] = str(config.get("ligand_lock_k_kjmol_nm2"))
    if "ligand_planarity_k_kjmol_nm2" in config:
        os.environ["OVO_LIGAND_PLANARITY_K_KJMOL_NM2"] = str(config.get("ligand_planarity_k_kjmol_nm2"))

    service = MDOptimizationService(output_dir=str(output_dir), job_id=job_id)
    attempt_sources: list[tuple[str, str, str]] = []
    # Strict mode: do not fall back away from refined ligand input.
    if strict_refined and ligand_input_source == "provided_refined_sdf_data":
        attempt_sources.append(("provided_refined_sdf_data", ligand_structure_data, "sdf"))
    else:
        # Priority: caller-provided refined SDF -> generated refined SDF -> extracted ligand PDB
        if ligand_input_source == "provided_refined_sdf_data":
            attempt_sources.append(("provided_refined_sdf_data", ligand_structure_data, "sdf"))
        if prep_artifacts.get("ligand_refined_sdf"):
            try:
                attempt_sources.append(("generated_refined_sdf", Path(prep_artifacts["ligand_refined_sdf"]).read_text(), "sdf"))
            except Exception:
                pass
        attempt_sources.append(("extracted_ligand_pdb", ligand_pdb, "pdb"))

    result = {}
    used_source = ligand_input_source
    used_format = ligand_data_format
    attempt_log: list[dict[str, str]] = []
    for source_name, source_data, source_format in attempt_sources:
        md_config = _build_md_config(source_data, source_format)
        result = service.optimize(md_config)
        used_source = source_name
        used_format = source_format
        status = str(result.get("status", ""))
        error_msg = str(result.get("error", ""))
        attempt_log.append({"source": source_name, "format": source_format, "status": status, "error": error_msg})
        # Stop on success or on non-ligand-prep failures (those likely won't be fixed by ligand input fallback).
        if status != "error":
            break
        if "Ligand preparation failed" not in error_msg:
            break
    preparation_artifacts: dict[str, Any] = {
        **prep_artifacts,
        "ligand_input_used_for_md": used_source,
        "ligand_input_format_used_for_md": used_format,
        "ligand_input_attempts": attempt_log,
        "strict_refined_ligand": strict_refined,
    }
    if config.get("input_complex_pdb_path"):
        preparation_artifacts["input_complex_pdb_path"] = str(config.get("input_complex_pdb_path"))
    if downloaded_pdb_path is not None:
        preparation_artifacts["downloaded_pdb"] = str(downloaded_pdb_path)
    # Provenance: point back to structure-prep assets when reused.
    if config.get("prepared_complex_path"):
        preparation_artifacts["prepared_complex_path"] = str(config.get("prepared_complex_path"))
    if config.get("ligand_refined_sdf_path"):
        preparation_artifacts["ligand_refined_sdf_source_path"] = str(config.get("ligand_refined_sdf_path"))
    if config.get("reference_smiles_path"):
        preparation_artifacts["reference_smiles_source_path"] = str(config.get("reference_smiles_path"))

    wrapped = {
        "success": result.get("status") != "error",
        "pdb_id": config.get("pdb_id", "").upper(),
        "selected_ligand": selected,
        "ligand_pdb": ligand_pdb,
        "preparation_artifacts": preparation_artifacts,
        "md_result": result,
        "mmgbsa": {
            "status": "not_implemented",
            "message": "Ligand-X contains ABFE/RBFE free-energy workflows, but no MM/GBSA implementation was found in the original code.",
        },
    }
    output_path.write_text(json.dumps(wrapped, indent=2))
    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover")
    discover.add_argument("--pdb-id", required=True)
    discover.add_argument("--output", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)

    args = parser.parse_args()
    try:
        if args.command == "discover":
            write_discovery(args.pdb_id, Path(args.output))
        elif args.command == "prepare":
            config = json.loads(Path(args.input).read_text())
            prepare_structure(config, Path(args.output))
        elif args.command == "run":
            config = json.loads(Path(args.input).read_text())
            run_ligandx_md(config, Path(args.output))
    except Exception as exc:
        payload = {"success": False, "error": str(exc)}
        output_arg = getattr(args, "output", None)
        if output_arg:
            Path(output_arg).write_text(json.dumps(payload, indent=2))
        raise


if __name__ == "__main__":
    main()
