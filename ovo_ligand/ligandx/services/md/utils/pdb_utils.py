"""
PDB utility functions for MD optimization.

Handles PDB sanitization, element inference, and formatting.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASX", "GLX", "SEC", "PYL",
}

STANDARD_NA = {
    "A", "C", "G", "U", "DA", "DC", "DG", "DT", "DU", "I",
}

COMMON_SOLVENT_IONS = {
    "HOH", "WAT", "SOL", "NA", "CL", "K", "MG", "CA", "ZN",
}


# Valid element symbols
VALID_ELEMENTS = {
    "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG", "AL", "SI", "P", "S", "CL", "AR",
    "K", "CA", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR",
    "RB", "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "I", "XE",
    "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU",
    "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO", "AT", "RN",
    "FR", "RA", "AC", "TH", "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM", "MD", "NO", "LR",
    "RF", "DB", "SG", "BH", "HS", "MT", "DS", "RG", "CN", "FL", "LV", "TS", "OG",
    "D", "T"  # common hydrogen isotopes
}


def infer_element_symbol(element_field: str, atom_name: str) -> Optional[str]:
    """
    Infer the correct element symbol for a PDB record.
    
    Priority:
        1. Check if atom name suggests a two-letter element (BR, CL, FE, etc.)
        2. Use existing element field if valid and consistent with atom name
        3. Deduce from atom name
    
    This handles cases where element column has single-letter element (like "B")
    but atom name indicates two-letter element (like "BR" for Bromine).
    
    Args:
        element_field: Element field from PDB (columns 77-78)
        atom_name: Atom name from PDB (columns 13-16)
        
    Returns:
        Element symbol or None if cannot be inferred
    """
    candidate = (element_field or "").strip().upper()
    name = (atom_name or "").strip()
    name_alpha = ''.join(ch for ch in name if ch.isalpha()).upper()
    
    # Two-letter elements that are commonly confused with single-letter ones
    # These should take priority when inferring from atom name
    TWO_LETTER_PRIORITY = {"BR", "CL", "FE", "ZN", "MG", "CA", "NA", "MN", "CO", "CU", "NI", "SE", "SI"}
    
    # First, check if atom name strongly suggests a two-letter element
    # This handles Br->B, Cl->C, Fe->F, etc. mismatches
    if len(name_alpha) >= 2:
        two_char = name_alpha[:2]
        if two_char in TWO_LETTER_PRIORITY:
            # Atom name suggests two-letter element
            # Only use single-letter element if it's NOT a prefix of the two-letter one
            if candidate and candidate != two_char[0]:
                # Element field has different element - trust it
                pass
            else:
                # Element is empty, same as first char, or matches two-letter
                return two_char
    
    # Use existing element field if valid
    if candidate in VALID_ELEMENTS:
        return candidate

    # Deduce from atom name
    for length in (2, 1):
        if len(name_alpha) >= length:
            guess = name_alpha[:length]
            if guess in VALID_ELEMENTS:
                return guess

    # Handle cases like 'FE' where element field only holds 'E'
    if len(name_alpha) >= 2 and name_alpha[1] == 'E' and name_alpha[:2] in VALID_ELEMENTS:
        return name_alpha[:2]

    return candidate or None


def sanitize_pdb_block(pdb_block: str) -> str:
    """
    Ensure element columns in a PDB block are valid.

    OpenMM and RDKit correctly read element symbols from columns 77-78 per the PDB format
    specification. We trust the element column and let the molecular modeling tools handle
    element parsing.

    Previous versions attempted to infer elements from atom names, which caused bugs like
    "O1S" (oxygen with positional suffix) being misidentified as "Os" (osmium). Public
    packages like RDKit handle these cases correctly by reading the element column directly.

    Args:
        pdb_block: PDB format string

    Returns:
        Original PDB string (element columns trusted as-is)
    """
    if not pdb_block:
        return pdb_block

    # Trust element columns from source PDB files
    # RDKit and OpenMM handle element parsing correctly
    return pdb_block


def format_ligand_pdb_block(pdb_block: str, residue_name: str = "LIG", chain_id: str = "A") -> str:
    """
    Normalize PDB residue metadata so the entire ligand is treated as a single residue.
    
    This avoids template-generator mismatches when RDKit outputs multiple residue names.
    
    Args:
        pdb_block: PDB format string
        residue_name: Residue name to use (max 3 chars)
        chain_id: Chain ID to use
        
    Returns:
        Formatted PDB string
    """
    if not pdb_block:
        return pdb_block

    residue_name = (residue_name or "LIG").upper()[:3]
    residue_name = residue_name.rjust(3)
    chain_id = (chain_id or "A")[0]
    res_seq = 1

    formatted_lines = []

    for line in pdb_block.splitlines():
        record = line[0:6].strip()
        if record in ("ATOM", "HETATM"):
            line = list(line.ljust(80))
            line[17:20] = list(residue_name)
            line[21] = chain_id
            line[22:26] = list(f"{res_seq:4d}")
            formatted_lines.append("".join(line).rstrip())
        elif record == "CONECT":
            formatted_lines.append(line.rstrip())
        # Skip TER/END records; we'll append a single END later

    if formatted_lines and formatted_lines[-1] != "END":
        formatted_lines.append("END")

    return "\n".join(formatted_lines)


def write_pdb_file(topology, positions, output_path: str, keep_ids: bool = True):
    """
    Write PDB files while guarding against Quantity conversion issues.
    
    Args:
        topology: OpenMM Topology object
        positions: OpenMM positions
        output_path: Path to write PDB file
        keep_ids: Whether to keep original residue IDs
    """
    from contextlib import suppress
    from openmm.app import PDBFile
    from openmm import unit, Vec3

    def _coerced_box_vectors():
        """
        Convert periodic box vectors to raw floats (in nm) so OpenMM's PDB writer
        does not attempt to call float() on openmm.unit.Quantity objects when
        formatting CRYST1 records.
        """
        box_vectors = topology.getPeriodicBoxVectors()
        if not box_vectors:
            return None, False

        def _component_to_float(component):
            if isinstance(component, unit.Quantity):
                return float(component.value_in_unit(unit.nanometer))
            return float(component)

        coerced_vectors = []
        needs_coercion = False
        for vec in box_vectors:
            comps = (_component_to_float(vec[0]), _component_to_float(vec[1]), _component_to_float(vec[2]))
            needs_coercion = needs_coercion or any(isinstance(comp, unit.Quantity) for comp in (vec[0], vec[1], vec[2]))
            coerced_vectors.append(Vec3(*comps))

        if needs_coercion:
            topology.setPeriodicBoxVectors(tuple(coerced_vectors))

        return box_vectors if needs_coercion else None, needs_coercion

    original_vectors, coerced = _coerced_box_vectors()
    try:
        with open(output_path, 'w') as handle:
            PDBFile.writeFile(topology, positions, handle, keepIds=keep_ids)
    except TypeError as e:
        if "Quantity.__float__" in str(e):
            logger.error("PDB write still failed due to Quantity conversion issues: %s", e)
        raise
    finally:
        if coerced and original_vectors:
            with suppress(Exception):
                topology.setPeriodicBoxVectors(original_vectors)

    _normalize_nonpolymer_residue_ids(output_path)


def _normalize_nonpolymer_residue_ids(output_path: str) -> None:
    """
    Normalize likely ligand/non-polymer residues in written PDBs so they do not collide
    with protein chain+residue numbering in sequence-oriented viewers.

    Strategy:
      - Keep standard amino acids/nucleic acids unchanged.
      - Keep common waters/ions unchanged.
      - Move remaining residues to chain 'L' with unique residue numbers starting at 1.
    """
    try:
        path = output_path
        with open(path, "r") as handle:
            lines = handle.readlines()

        nonpolymer_map: Dict[tuple[str, str, str], tuple[str, int]] = {}
        next_resseq = 1
        updated: list[str] = []

        for line in lines:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 26:
                updated.append(line)
                continue

            resname = line[17:20].strip().upper()
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip() or "0"
            icode = line[26].strip() if len(line) > 26 else ""

            is_polymer = resname in STANDARD_AA or resname in STANDARD_NA
            is_solvent_or_ion = resname in COMMON_SOLVENT_IONS
            if is_polymer or is_solvent_or_ion:
                updated.append(line)
                continue

            key = (resname, chain_id, f"{resseq}{icode}")
            if key not in nonpolymer_map:
                nonpolymer_map[key] = ("L", next_resseq)
                next_resseq += 1
            new_chain, new_resseq = nonpolymer_map[key]

            line_chars = list(line.rstrip("\n").ljust(80))
            line_chars[21] = new_chain
            line_chars[22:26] = list(f"{new_resseq:4d}")
            updated.append("".join(line_chars).rstrip() + "\n")

        with open(path, "w") as handle:
            handle.writelines(updated)

        if nonpolymer_map:
            logger.info(
                "Normalized %d non-polymer residue group(s) to dedicated ligand chain in %s",
                len(nonpolymer_map),
                output_path,
            )
    except Exception as exc:
        logger.warning("Non-polymer residue normalization skipped for %s: %s", output_path, exc)


def normalize_nonpolymer_residue_ids_in_pdb_block(pdb_block: str) -> str:
    """
    In-memory variant of non-polymer residue normalization for PDB strings.
    Useful for trajectory/frame exports generated without writing through write_pdb_file.
    """
    if not pdb_block:
        return pdb_block

    try:
        lines = pdb_block.splitlines()
        nonpolymer_map: Dict[tuple[str, str, str], tuple[str, int]] = {}
        next_resseq = 1
        updated: list[str] = []

        for line in lines:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 26:
                updated.append(line)
                continue

            resname = line[17:20].strip().upper()
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip() or "0"
            icode = line[26].strip() if len(line) > 26 else ""

            is_polymer = resname in STANDARD_AA or resname in STANDARD_NA
            is_solvent_or_ion = resname in COMMON_SOLVENT_IONS
            if is_polymer or is_solvent_or_ion:
                updated.append(line)
                continue

            key = (resname, chain_id, f"{resseq}{icode}")
            if key not in nonpolymer_map:
                nonpolymer_map[key] = ("L", next_resseq)
                next_resseq += 1
            new_chain, new_resseq = nonpolymer_map[key]

            line_chars = list(line.ljust(80))
            line_chars[21] = new_chain
            line_chars[22:26] = list(f"{new_resseq:4d}")
            updated.append("".join(line_chars).rstrip())

        return "\n".join(updated) + ("\n" if pdb_block.endswith("\n") else "")
    except Exception as exc:
        logger.warning("Non-polymer normalization skipped for PDB block: %s", exc)
        return pdb_block


def clean_results_for_json(results: Any) -> Any:
    """
    Recursively clean results dictionary to remove non-JSON-serializable objects.
    
    Args:
        results: Data structure to clean
        
    Returns:
        JSON-serializable version of the data
    """
    import math
    
    if isinstance(results, dict):
        cleaned = {}
        for key, value in results.items():
            if key in ['simulation', 'final_simulation']:
                # Skip non-serializable simulation objects
                continue
            elif hasattr(value, '__module__') and 'openmm' in str(value.__module__):
                # Skip any OpenMM objects
                continue
            else:
                cleaned[key] = clean_results_for_json(value)
        return cleaned
    elif isinstance(results, list):
        return [clean_results_for_json(item) for item in results]
    elif isinstance(results, (int, float)):
        # Handle NaN and infinity values
        if math.isnan(results):
            return None
        elif math.isinf(results):
            return None
        else:
            return results
    else:
        return results
