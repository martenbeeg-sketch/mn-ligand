from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mn_ligand.core.artifacts import ArtifactRef, load_artifact_manifest, write_artifact_manifest
from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    registered_tool,
    write_registered_command_record,
)
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.core.residue_mapping import (
    derive_residue_mapping,
    identity_residue_mapping,
)
from mn_ligand.runtime import PROJECT_DIR, resolve_run_dir, runs_root
from mn_ligand.ligandx.lib.chemistry.preparation.target_validation import (
    audit_target_geometry,
    prepare_target_for_publication,
    remove_internal_oxt,
)
from mn_ligand.workflows.bound_ligand_md import (
    MODIFIED_RESIDUE_MAPPINGS,
    is_bound_ligand_atom,
    parse_bound_ligands,
)


DEFAULT_PROTEIN_CLEANING_IMAGE = "ovolig-md-cu128:latest"
DEFAULT_MODELLER_PYTHON = Path(
    os.environ.get("MN_LIGAND_MODELLER_PYTHON")
    or "/home/user/mambaforge/envs/mn-ligand-modeller/bin/python"
)

CANONICAL_AMINO_ACIDS = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
ONE_TO_THREE = {value: key for key, value in CANONICAL_AMINO_ACIDS.items()}
REFERENCE_AMINO_ACIDS = {
    **CANONICAL_AMINO_ACIDS,
    "CAS": "C",
    "CAF": "C",
    "MSE": "M",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _safe_filename(filename: str, structure_format: str) -> str:
    suffix = ".cif" if structure_format == "mmcif" else ".pdb"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename).stem).strip("-._") or "target"
    return f"{stem}{suffix}"


def detect_structure_format(data: str, filename: str = "") -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".cif", ".mmcif"} or "_atom_site." in data:
        if "_atom_site." not in data:
            raise ValueError("Uploaded mmCIF does not contain an atom_site table")
        return "mmcif"
    if suffix in {".pdb", ".ent", ""} or any(
        line.startswith(("ATOM  ", "HETATM")) for line in data.splitlines()
    ):
        if not any(line.startswith(("ATOM  ", "HETATM")) for line in data.splitlines()):
            raise ValueError("Uploaded PDB contains no ATOM or HETATM records")
        return "pdb"
    raise ValueError("Protein import supports PDB and mmCIF structures")


def pdb_chemical_components(data: str) -> dict[str, dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}

    def component(code: str) -> dict[str, Any] | None:
        normalized = code.strip().upper()
        if not normalized:
            return None
        return components.setdefault(
            normalized,
            {"ccd_id": normalized, "name": "", "formula": "", "synonyms": []},
        )

    name_parts: dict[str, list[str]] = {}
    synonym_parts: dict[str, list[str]] = {}
    formula_parts: dict[str, list[str]] = {}
    for line in data.splitlines():
        record = line[:6].strip()
        if record in {"HETNAM", "HETSYN"}:
            code = line[11:14].strip()
            text = line[15:70].strip()
            if component(code) is not None and text:
                target = name_parts if record == "HETNAM" else synonym_parts
                target.setdefault(code, []).append(text)
        elif record == "FORMUL":
            code = line[12:15].strip()
            text = line[18:70].strip()
            if component(code) is not None and text:
                formula_parts.setdefault(code, []).append(text)

    for code, item in components.items():
        item["name"] = " ".join(name_parts.get(code, [])).strip().rstrip(";")
        formula = " ".join(formula_parts.get(code, [])).strip()
        item["formula"] = re.sub(r"^\*\s*", "", formula)
        synonyms = " ".join(synonym_parts.get(code, []))
        item["synonyms"] = [value.strip() for value in synonyms.split(";") if value.strip()]
    return components


def structure_ligands(data: str, structure_format: str) -> list[dict[str, Any]]:
    if structure_format != "pdb":
        return []
    components = pdb_chemical_components(data)
    noncanonical_site_keys = {
        str(site["key"]) for site in detect_noncanonical_residues(data)
    }
    ligands: list[dict[str, Any]] = []
    for ligand in parse_bound_ligands(data):
        if str(ligand.get("key") or "") in noncanonical_site_keys:
            continue
        code = str(ligand.get("resname") or "").upper()
        details = components.get(code, {})
        ligands.append(
            {
                **ligand,
                "ccd_id": code if code not in {"LIG", "UNL"} else "",
                "name": str(details.get("name") or ""),
                "formula": str(details.get("formula") or ""),
                "synonyms": list(details.get("synonyms") or []),
            }
        )
    return ligands


def _pdb_key_value_records(data: str, record_name: str) -> list[dict[str, str]]:
    text = " ".join(
        line[10:80].strip() for line in data.splitlines() if line.startswith(record_name)
    )
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for segment in text.split(";"):
        if ":" not in segment:
            continue
        key, value = segment.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "mol_id" and current:
            records.append(current)
            current = {}
        current[key] = value
    if current:
        records.append(current)
    return records


def pdb_receptor_metadata(data: str) -> dict[str, Any]:
    lines = data.splitlines()
    title = " ".join(line[10:80].strip() for line in lines if line.startswith("TITLE ")).strip()
    methods = " ".join(line[10:80].strip() for line in lines if line.startswith("EXPDTA")).strip()
    resolution = None
    for line in lines:
        if line.startswith("REMARK   2 RESOLUTION."):
            match = re.search(r"RESOLUTION\.\s+([0-9.]+)\s+ANGSTROMS", line, flags=re.IGNORECASE)
            if match:
                resolution = float(match.group(1))
                break
    compounds = _pdb_key_value_records(data, "COMPND")
    sources = {
        item.get("mol_id", ""): item for item in _pdb_key_value_records(data, "SOURCE")
    }
    entities: list[dict[str, Any]] = []
    for compound in compounds:
        source = sources.get(compound.get("mol_id", ""), {})
        chains = [value.strip() for value in compound.get("chain", "").split(",") if value.strip()]
        entities.append(
            {
                "entity_id": compound.get("mol_id", ""),
                "name": compound.get("molecule", ""),
                "chains": chains,
                "ec": compound.get("ec", ""),
                "mutation": compound.get("mutation", ""),
                "source_organisms": [source["organism_scientific"]]
                if source.get("organism_scientific")
                else [],
                "expression_hosts": [source["expression_system"]]
                if source.get("expression_system")
                else [],
                "gene": source.get("gene", ""),
            }
        )
    header = next((line for line in lines if line.startswith("HEADER")), "")
    return {
        "title": title,
        "experimental_method": methods,
        "resolution_angstrom": resolution,
        "deposition_date": header[50:59].strip() if len(header) >= 59 else "",
        "entities": entities,
        "metadata_source": "pdb_header",
    }


def structure_summary(data: str, structure_format: str) -> dict[str, Any]:
    if structure_format == "pdb":
        atom_lines = [line for line in data.splitlines() if line.startswith(("ATOM  ", "HETATM"))]
        protein_lines = [
            line for line in atom_lines if line.startswith("ATOM  ") and not is_bound_ligand_atom(line)
        ]
        chains = sorted({line[21].strip() or "_" for line in protein_lines if len(line) > 21})
        residues = {
            (line[21].strip() or "_", line[22:26].strip(), line[26].strip() or "_")
            for line in protein_lines
            if len(line) >= 27
        }
        return {
            "format": structure_format,
            "atom_records": len(atom_lines),
            "protein_atom_records": len(protein_lines),
            "protein_residues": len(residues),
            "chains": chains,
            "ligands": structure_ligands(data, structure_format),
            "receptor": pdb_receptor_metadata(data),
            "noncanonical_residues": detect_noncanonical_residues(data),
        }
    return {
        "format": structure_format,
        "atom_records": sum(1 for line in data.splitlines() if line.startswith(("ATOM ", "HETATM "))),
        "protein_atom_records": None,
        "protein_residues": None,
        "chains": [],
        "ligands": [],
        "receptor": {},
        "noncanonical_residues": [],
    }


def _site_key(resname: str, chain: str, resseq: str, icode: str = "") -> str:
    return "|".join(
        [
            resname.strip().upper(),
            chain.strip() or "_",
            str(resseq).strip(),
            icode.strip() or "_",
        ]
    )


def detect_noncanonical_residues(pdb_data: str) -> list[dict[str, Any]]:
    """Return polymer-like noncanonical sites with residue-specific evidence.

    A MODRES record is treated as a suggested canonical parent, never as a
    mandatory global mapping. Identical component names may therefore receive
    different user-selected replacements at different residue positions.
    """
    modres: dict[tuple[str, str, str], dict[str, str]] = {}
    seqadv: dict[tuple[str, str, str], dict[str, str]] = {}
    for line in pdb_data.splitlines():
        if line.startswith("MODRES") and len(line) >= 28:
            chain = line[16].strip() or "_"
            resseq = line[18:22].strip()
            icode = line[22].strip() or "_"
            modres[(chain, resseq, icode)] = {
                "resname": line[12:15].strip().upper(),
                "suggested_target": line[24:27].strip().upper(),
                "comment": line[29:70].strip(),
            }
        elif line.startswith("SEQADV") and len(line) >= 42:
            chain = line[16].strip() or "_"
            resseq = line[18:22].strip()
            icode = line[22].strip() or "_"
            seqadv[(chain, resseq, icode)] = {
                "resname": line[12:15].strip().upper(),
                "suggested_target": line[39:42].strip().upper(),
                "comment": line[49:70].strip(),
            }

    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in pdb_data.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        resname = line[17:20].strip().upper()
        chain = line[21].strip() or "_"
        resseq = line[22:26].strip()
        icode = line[26].strip() or "_"
        annotation = modres.get((chain, resseq, icode))
        sequence_annotation = seqadv.get((chain, resseq, icode))
        if (
            resname in CANONICAL_AMINO_ACIDS
            or (
                annotation is None
                and sequence_annotation is None
                and resname not in MODIFIED_RESIDUE_MAPPINGS
            )
        ):
            continue
        site = observed.setdefault(
            (chain, resseq, icode),
            {
                "key": _site_key(resname, chain, resseq, icode),
                "resname": resname,
                "chain": chain,
                "resseq": resseq,
                "icode": icode,
                "atom_count": 0,
                "suggested_target": "",
                "evidence": "unresolved",
                "evidence_detail": "",
            },
        )
        site["atom_count"] += 1
        evidence: list[str] = []
        details: list[str] = []
        if annotation and annotation["resname"] == resname:
            target = annotation["suggested_target"]
            site["suggested_target"] = target if target in CANONICAL_AMINO_ACIDS else ""
            evidence.append("MODRES")
            details.append(
                f"MODRES: {target or 'unknown'}"
                + (f" ({annotation['comment']})" if annotation["comment"] else "")
            )
        if sequence_annotation and sequence_annotation["resname"] == resname:
            target = sequence_annotation["suggested_target"]
            if target in CANONICAL_AMINO_ACIDS:
                site["suggested_target"] = target
            evidence.append("SEQADV")
            details.append(
                f"SEQADV: {target or 'unknown'}"
                + (
                    f" ({sequence_annotation['comment']})"
                    if sequence_annotation["comment"]
                    else ""
                )
            )
        if evidence:
            site["evidence"] = "PDB " + " + ".join(evidence)
            site["evidence_detail"] = "; ".join(details)
    return sorted(
        observed.values(),
        key=lambda site: (
            str(site["chain"]),
            int(site["resseq"]) if str(site["resseq"]).lstrip("-").isdigit() else 0,
            str(site["icode"]),
        ),
    )


def repair_noncanonical_residues_with_modeller(
    pdb_data: str,
    replacements: list[dict[str, str]],
    *,
    work_dir: Path,
    modeller_python: Path = DEFAULT_MODELLER_PYTHON,
) -> tuple[str, dict[str, Any]]:
    """Apply explicit per-site amino-acid replacements with MODELLER."""
    if not replacements:
        return pdb_data, {"enabled": False, "engine": "MODELLER", "replacements": []}
    valid_targets = set(CANONICAL_AMINO_ACIDS)
    normalized: list[dict[str, str]] = []
    detected = {site["key"]: site for site in detect_noncanonical_residues(pdb_data)}
    for replacement in replacements:
        key = str(replacement.get("key") or "")
        target = str(replacement.get("target") or "").upper()
        if key not in detected:
            raise ValueError(f"Noncanonical residue site was not detected: {key}")
        if target not in valid_targets:
            raise ValueError(f"Replacement for {key} must be a canonical amino acid")
        site = detected[key]
        normalized.append(
            {
                "key": key,
                "resname": str(site["resname"]),
                "chain": str(site["chain"]),
                "resseq": str(site["resseq"]),
                "icode": "" if site["icode"] == "_" else str(site["icode"]),
                "target": target,
                "evidence": str(site["evidence"]),
                "suggested_target": str(site["suggested_target"]),
            }
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    source_path = work_dir / "modeller_input.pdb"
    replacements_path = work_dir / "modeller_replacements.json"
    modeled_path = work_dir / "modeller_repaired.pdb"
    report_path = work_dir / "modeller_report.json"
    source_path.write_text(pdb_data if pdb_data.endswith("\n") else pdb_data + "\n")
    _write_json(replacements_path, normalized)
    runner = PROJECT_DIR / "mn_ligand" / "workflows" / "modeller_residue_repair_runner.py"
    command = [
        str(modeller_python),
        str(runner),
        "--input",
        str(source_path),
        "--replacements",
        str(replacements_path),
        "--output",
        str(modeled_path),
        "--report",
        str(report_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False, cwd=work_dir)
    if process.returncode != 0 or not modeled_path.is_file():
        error = (process.stderr or process.stdout or "MODELLER residue repair failed")[-4000:]
        raise RuntimeError(error)
    report = json.loads(report_path.read_text())
    model_groups: list[list[str]] = []
    model_group_keys: list[tuple[str, str, str]] = []
    for line in modeled_path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        group_key = (line[21], line[22:26], line[26])
        if not model_group_keys or model_group_keys[-1] != group_key:
            model_group_keys.append(group_key)
            model_groups.append([])
        model_groups[-1].append(line)

    replacement_lines: dict[tuple[str, str, str], list[str]] = {}
    for item in report.get("replacements") or []:
        model_index = int(item["model_index"])
        # MODELLER residue indices are one-based.
        if model_index < 1 or model_index > len(model_groups):
            raise RuntimeError(f"MODELLER did not retain modeled residue {item['key']}")
        chain = " " if item["chain"] == "_" else str(item["chain"])
        resseq = f"{int(item['resseq']):4d}"
        icode = str(item.get("icode") or " ")[:1]
        target = str(item["target"]).upper()
        rewritten = []
        for line in model_groups[model_index - 1]:
            rewritten.append(
                "ATOM  "
                + line[6:16]
                + " "
                + f"{target:>3}"
                + line[20:21]
                + chain
                + resseq
                + icode
                + line[27:]
            )
        replacement_lines[(chain, resseq, icode)] = rewritten

    merged: list[str] = []
    emitted: set[tuple[str, str, str]] = set()
    for line in pdb_data.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27:
            key = (line[21], line[22:26], line[26])
            if key in replacement_lines:
                if key not in emitted:
                    merged.extend(replacement_lines[key])
                    emitted.add(key)
                continue
        merged.append(line)
    if set(replacement_lines) != emitted:
        raise RuntimeError("MODELLER replacements could not be merged into the source structure")
    repaired_data = "\n".join(merged) + "\n"
    report.update(
        {
            "command": command,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "merge": "MODELLER-selected residue coordinates merged into immutable source complex",
        }
    )
    return repaired_data, report


def synchronize_repaired_sequence_records(
    pdb_data: str,
    replacements: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    """Keep SEQRES/MODRES/SEQADV consistent with modeled canonical identities."""
    by_chain: dict[str, list[dict[str, str]]] = {}
    selected_modres: set[tuple[str, str, str]] = set()
    for item in replacements:
        chain = str(item.get("chain") or "_")
        by_chain.setdefault(chain, []).append(item)
        selected_modres.add(
            (
                chain,
                str(item.get("resseq") or ""),
                str(item.get("icode") or "_") or "_",
            )
        )
    for values in by_chain.values():
        values.sort(
            key=lambda item: (
                int(str(item["resseq"]))
                if str(item["resseq"]).lstrip("-").isdigit()
                else 0,
                str(item.get("icode") or ""),
            )
        )
    occurrence_targets: dict[str, dict[str, list[str]]] = {}
    for chain, values in by_chain.items():
        for item in values:
            occurrence_targets.setdefault(chain, {}).setdefault(
                str(item["resname"]).upper(), []
            ).append(str(item["target"]).upper())

    occurrence_index: dict[tuple[str, str], int] = {}
    output: list[str] = []
    removed_modres = 0
    seqres_replacements = 0
    for line in pdb_data.splitlines():
        if line.startswith("MODRES") and len(line) >= 27:
            key = (
                line[16].strip() or "_",
                line[18:22].strip(),
                line[22].strip() or "_",
            )
            if key in selected_modres:
                removed_modres += 1
                continue
        if line.startswith("SEQADV") and len(line) >= 42:
            key = (
                line[16].strip() or "_",
                line[18:22].strip(),
                line[22].strip() or "_",
            )
            replacement = next(
                (
                    item
                    for item in by_chain.get(key[0], [])
                    if str(item["resseq"]) == key[1]
                    and (str(item.get("icode") or "_") or "_") == key[2]
                ),
                None,
            )
            if replacement:
                target = str(replacement["target"]).upper()
                line = line[:12] + f"{target:>3}" + line[15:39] + f"{target:>3}" + line[42:]
        if line.startswith("SEQRES") and len(line) >= 20:
            chain = line[11].strip() or "_"
            tokens = line[19:].split()
            changed = False
            for index, token in enumerate(tokens):
                targets = occurrence_targets.get(chain, {}).get(token.upper(), [])
                counter_key = (chain, token.upper())
                occurrence = occurrence_index.get(counter_key, 0)
                if occurrence < len(targets):
                    tokens[index] = targets[occurrence]
                    occurrence_index[counter_key] = occurrence + 1
                    seqres_replacements += 1
                    changed = True
            if changed:
                line = line[:19] + " ".join(tokens).ljust(max(0, len(line) - 19))
        output.append(line)
    return "\n".join(output) + "\n", {
        "seqres_replacements": seqres_replacements,
        "removed_modres_records": removed_modres,
    }


def _global_sequence_alignment(reference: str, observed: str) -> tuple[str, str]:
    """Needleman-Wunsch alignment used only to locate missing coordinate segments."""
    rows, cols = len(reference) + 1, len(observed) + 1
    gap = -2
    scores = [[0] * cols for _ in range(rows)]
    trace = [[""] * cols for _ in range(rows)]
    for row in range(1, rows):
        scores[row][0] = row * gap
        trace[row][0] = "U"
    for col in range(1, cols):
        scores[0][col] = col * gap
        trace[0][col] = "L"
    for row in range(1, rows):
        for col in range(1, cols):
            diagonal = scores[row - 1][col - 1] + (
                2 if reference[row - 1] == observed[col - 1] else -1
            )
            up = scores[row - 1][col] + gap
            left = scores[row][col - 1] + gap
            best = max(diagonal, up, left)
            scores[row][col] = best
            trace[row][col] = "D" if best == diagonal else ("U" if best == up else "L")
    aligned_reference: list[str] = []
    aligned_observed: list[str] = []
    row, col = len(reference), len(observed)
    while row or col:
        direction = trace[row][col]
        if direction == "D":
            aligned_reference.append(reference[row - 1])
            aligned_observed.append(observed[col - 1])
            row -= 1
            col -= 1
        elif direction == "U":
            aligned_reference.append(reference[row - 1])
            aligned_observed.append("-")
            row -= 1
        else:
            aligned_reference.append("-")
            aligned_observed.append(observed[col - 1])
            col -= 1
    return "".join(reversed(aligned_reference)), "".join(reversed(aligned_observed))


def _chain_observed_sequence(pdb_data: str, chain: str) -> tuple[str, int, int]:
    residues: list[tuple[str, int, str]] = []
    seen: set[tuple[int, str]] = set()
    for line in pdb_data.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 27:
            continue
        line_chain = line[21].strip() or "_"
        if line_chain != chain:
            continue
        resname = line[17:20].strip().upper()
        if resname not in CANONICAL_AMINO_ACIDS:
            continue
        try:
            number = int(line[22:26])
        except ValueError:
            continue
        insertion = line[26].strip()
        key = (number, insertion)
        if key in seen:
            continue
        seen.add(key)
        residues.append((CANONICAL_AMINO_ACIDS[resname], number, insertion))
    if not residues:
        raise ValueError(f"Chain {chain} has no canonical protein coordinates")
    return "".join(item[0] for item in residues), residues[0][1], residues[-1][1]


def coordinate_gap_candidates(pdb_data: str) -> list[dict[str, Any]]:
    """Return internal author-number discontinuities with observed flanking residues."""
    chains: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for line in pdb_data.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        resname = line[17:20].strip().upper()
        code = REFERENCE_AMINO_ACIDS.get(resname)
        if not code:
            continue
        chain = line[21].strip() or "_"
        resseq = line[22:26].strip()
        icode = line[26].strip()
        key = (chain, resseq, icode)
        if key in seen or not resseq.lstrip("-").isdigit():
            continue
        seen.add(key)
        chains.setdefault(chain, []).append(
            {
                "chain": chain,
                "resseq": int(resseq),
                "icode": icode,
                "resname": resname,
                "code": code,
            }
        )
    candidates: list[dict[str, Any]] = []
    for chain, residues in sorted(chains.items()):
        residues.sort(key=lambda item: (int(item["resseq"]), str(item["icode"])))
        for index, (left, right) in enumerate(zip(residues, residues[1:])):
            if left["icode"] or right["icode"]:
                continue
            difference = int(right["resseq"]) - int(left["resseq"])
            if difference <= 1:
                continue
            candidates.append(
                {
                    "chain": chain,
                    "left_resseq": int(left["resseq"]),
                    "left_resname": str(left["resname"]),
                    "left_code": str(left["code"]),
                    "right_resseq": int(right["resseq"]),
                    "right_resname": str(right["resname"]),
                    "right_code": str(right["code"]),
                    "author_start": int(left["resseq"]) + 1,
                    "author_end": int(right["resseq"]) - 1,
                    "numbering_gap_length": difference - 1,
                    "insertion_index": index + 1,
                }
            )
    return candidates


def _pdb_reference_sequences(pdb_data: str) -> dict[str, str]:
    chains: dict[str, list[str]] = {}
    for line in pdb_data.splitlines():
        if not line.startswith("SEQRES") or len(line) < 20:
            continue
        chain = line[11].strip() or "_"
        chains.setdefault(chain, []).extend(
            REFERENCE_AMINO_ACIDS.get(token.upper(), "X")
            for token in line[19:].split()
        )
    return {chain: "".join(sequence) for chain, sequence in chains.items()}


def _pdb_declared_missing_residues(
    pdb_data: str,
) -> dict[str, list[list[dict[str, Any]]]]:
    sites: dict[str, list[dict[str, Any]]] = {}
    pattern = re.compile(
        r"^REMARK 465\s+(?:\d+\s+)?([A-Z0-9]{3})\s+(\S)\s+(-?\d+)([A-Za-z]?)\s*$"
    )
    for line in pdb_data.splitlines():
        match = pattern.match(line.rstrip())
        if not match:
            continue
        resname, chain, number, insertion = match.groups()
        if resname not in REFERENCE_AMINO_ACIDS:
            continue
        sites.setdefault(chain or "_", []).append(
            {
                "resname": resname,
                "code": REFERENCE_AMINO_ACIDS[resname],
                "number": int(number),
                "icode": insertion,
            }
        )
    segments: dict[str, list[list[dict[str, Any]]]] = {}
    for chain, values in sites.items():
        values.sort(key=lambda item: (int(item["number"]), str(item["icode"])))
        groups: list[list[dict[str, Any]]] = []
        for item in values:
            if (
                not groups
                or item["icode"]
                or groups[-1][-1]["icode"]
                or int(item["number"]) != int(groups[-1][-1]["number"]) + 1
            ):
                groups.append([item])
            else:
                groups[-1].append(item)
        segments[chain] = groups
    return segments


def detect_modeller_internal_gaps(
    pdb_data: str,
    *,
    max_internal_gap: int,
    user_gap_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    references = _pdb_reference_sequences(pdb_data)
    declared_missing = _pdb_declared_missing_residues(pdb_data)
    result: dict[str, dict[str, Any]] = {}
    chains = sorted(
        {
            line[21].strip() or "_"
            for line in pdb_data.splitlines()
            if line.startswith("ATOM  ") and len(line) >= 27
        }
    )
    for chain in chains:
        observed, start, end = _chain_observed_sequence(pdb_data, chain)
        confirmed = [
            item
            for item in list(user_gap_definitions or [])
            if str(item.get("chain") or "_") == chain
        ]
        if confirmed:
            confirmed.sort(key=lambda item: int(item["insertion_index"]))
            template_parts: list[str] = []
            target_parts: list[str] = []
            gaps = []
            cursor = 0
            inserted = 0
            for definition in confirmed:
                insertion_index = int(definition["insertion_index"])
                if insertion_index < cursor or insertion_index > len(observed):
                    raise ValueError(f"Invalid gap insertion index for chain {chain}")
                sequence = "".join(str(definition.get("sequence") or "").split()).upper()
                if not sequence or set(sequence) - set(ONE_TO_THREE):
                    raise ValueError(
                        f"Gap {chain}:{definition.get('author_start')}-"
                        f"{definition.get('author_end')} requires a canonical sequence"
                    )
                if (
                    str(definition.get("left_code") or "").upper()
                    != observed[insertion_index - 1]
                    or str(definition.get("right_code") or "").upper()
                    != observed[insertion_index]
                ):
                    raise ValueError(
                        f"Confirmed flanking residues do not match chain {chain} coordinates"
                    )
                template_parts.append(observed[cursor:insertion_index])
                target_parts.append(observed[cursor:insertion_index])
                template_parts.append("-" * len(sequence))
                target_parts.append(sequence)
                target_start = insertion_index + inserted + 1
                gap = {
                    **definition,
                    "chain": chain,
                    "target_start": target_start,
                    "target_end": target_start + len(sequence) - 1,
                    "sequence": sequence,
                    "residues": [ONE_TO_THREE[item] for item in sequence],
                    "length": len(sequence),
                    "terminal": False,
                    "eligible": len(sequence) <= int(max_internal_gap),
                    "reason": (
                        ""
                        if len(sequence) <= int(max_internal_gap)
                        else f"gap_exceeds_{int(max_internal_gap)}"
                    ),
                    "evidence": str(
                        definition.get("evidence") or "user-confirmed sequence and flanks"
                    ),
                }
                gaps.append(gap)
                cursor = insertion_index
                inserted += len(sequence)
            template_parts.append(observed[cursor:])
            target_parts.append(observed[cursor:])
            result[chain] = {
                "chain": chain,
                "start": start,
                "end": end,
                "observed_sequence": observed,
                "reference_sequence": "".join(target_parts),
                "template_alignment": "".join(template_parts),
                "target_alignment": "".join(target_parts),
                "gaps": gaps,
                "alignment_method": "user-confirmed missing sequence and coordinate flanks",
            }
            continue
        if chain not in references:
            continue
        explicit_segments = declared_missing.get(chain, [])
        if explicit_segments:
            template = list(references[chain])
            gaps = []
            valid_explicit = True
            for segment in explicit_segments:
                target_start = int(segment[0]["number"]) - start + 1
                target_end = target_start + len(segment) - 1
                sequence = "".join(str(item["code"]) for item in segment)
                if (
                    target_start < 1
                    or target_end > len(template)
                    or references[chain][target_start - 1 : target_end] != sequence
                ):
                    valid_explicit = False
                    break
                template[target_start - 1 : target_end] = ["-"] * len(segment)
                terminal = target_start == 1 or target_end == len(template)
                gaps.append(
                    {
                        "chain": chain,
                        "target_start": target_start,
                        "target_end": target_end,
                        "author_start": int(segment[0]["number"]),
                        "author_end": int(segment[-1]["number"]),
                        "insertion_index": target_start - 1,
                        "sequence": sequence,
                        "residues": [str(item["resname"]) for item in segment],
                        "length": len(segment),
                        "terminal": terminal,
                        "eligible": not terminal
                        and len(segment) <= int(max_internal_gap),
                        "reason": (
                            "terminal"
                            if terminal
                            else (
                                ""
                                if len(segment) <= int(max_internal_gap)
                                else f"gap_exceeds_{int(max_internal_gap)}"
                            )
                        ),
                        "evidence": "PDB REMARK 465 + SEQRES",
                    }
                )
            if valid_explicit:
                result[chain] = {
                    "chain": chain,
                    "start": start,
                    "end": end,
                    "observed_sequence": observed,
                    "reference_sequence": references[chain],
                    "template_alignment": "".join(template),
                    "target_alignment": references[chain],
                    "gaps": gaps,
                    "alignment_method": "authoritative PDB REMARK 465 residue numbers",
                }
                continue
        aligned_reference, aligned_observed = _global_sequence_alignment(
            references[chain], observed
        )
        template: list[str] = []
        target: list[str] = []
        gaps: list[dict[str, Any]] = []
        target_position = 0
        observed_before = 0
        index = 0
        while index < len(aligned_reference):
            ref, obs = aligned_reference[index], aligned_observed[index]
            if ref == "-":
                index += 1
                continue
            target.append(ref)
            target_position += 1
            template.append(obs if obs != "-" else "-")
            if obs != "-":
                observed_before += 1
                index += 1
                continue
            gap_start = target_position
            sequence = [ref]
            index += 1
            while index < len(aligned_reference):
                ref, obs = aligned_reference[index], aligned_observed[index]
                if ref == "-" or obs != "-":
                    break
                target.append(ref)
                template.append("-")
                target_position += 1
                sequence.append(ref)
                index += 1
            terminal = observed_before == 0 or observed_before == len(observed)
            gaps.append(
                {
                    "chain": chain,
                    "target_start": gap_start,
                    "target_end": gap_start + len(sequence) - 1,
                    "insertion_index": observed_before,
                    "sequence": "".join(sequence),
                    "residues": [ONE_TO_THREE[item] for item in sequence],
                    "length": len(sequence),
                    "terminal": terminal,
                    "eligible": not terminal and len(sequence) <= int(max_internal_gap),
                    "reason": (
                        "terminal"
                        if terminal
                        else (
                            ""
                            if len(sequence) <= int(max_internal_gap)
                            else f"gap_exceeds_{int(max_internal_gap)}"
                        )
                    ),
                }
            )
        result[chain] = {
            "chain": chain,
            "start": start,
            "end": end,
            "observed_sequence": observed,
            "reference_sequence": references[chain],
            "template_alignment": "".join(template),
            "target_alignment": "".join(target),
            "gaps": gaps,
            "alignment_method": "global SEQRES-to-coordinate alignment fallback",
        }
    return result


def _selected_chain_template(pdb_data: str, chain: str) -> str:
    chain_id = " " if chain == "_" else chain
    lines = [
        line
        for line in pdb_data.splitlines()
        if line.startswith("ATOM  ")
        and len(line) >= 27
        and line[21] == chain_id
        and line[17:20].strip().upper() in CANONICAL_AMINO_ACIDS
    ]
    if not lines:
        raise ValueError(f"Chain {chain} has no MODELLER template coordinates")
    return "\n".join([*lines, "TER", "END", ""])


def _internal_gap_pir(
    *,
    template_name: str,
    chain: str,
    start: int,
    end: int,
    template_alignment: str,
    target_alignment: str,
) -> str:
    modeller_chain = " " if chain == "_" else chain
    return (
        f">P1;template\n"
        f"structureX:{template_name}:{start}:{modeller_chain}:{end}:{modeller_chain}::::\n"
        f"{template_alignment}*\n"
        f">P1;repaired\n"
        # Keep deposited author numbering in the generated model. For an
        # internal gap, the target spans the same author-number interval as
        # the template; MODELLER then assigns the inserted residues their
        # missing deposited numbers instead of renumbering the chain from 1.
        f"sequence:repaired:{start}:{modeller_chain}:{end}:{modeller_chain}::::\n"
        f"{target_alignment}*\n"
    )


def _renumber_pdb_atom_serials(pdb_data: str) -> str:
    serial = 0
    output: list[str] = []
    for line in pdb_data.splitlines():
        # Replacing a polymer chain changes atom serials, so source CONECT
        # records would point at unrelated atoms after this merge. PDBFixer
        # reconstructs standard-polymer bonds and ligand preparation derives
        # ligand connectivity independently.
        if line.startswith("CONECT"):
            continue
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 11:
            serial += 1
            line = line[:6] + f"{serial:5d}" + line[11:]
        output.append(line)
    return "\n".join(output) + "\n"


def _restore_modeled_author_numbering(
    complex_pdb: str,
    model_pdb: str,
    *,
    chain: str,
    gaps: list[dict[str, Any]],
) -> str:
    """Map MODELLER's sequential residue IDs back to deposited author IDs."""
    chain_id = " " if chain == "_" else chain
    source_labels: list[tuple[int, str]] = []
    seen_source: set[tuple[int, str]] = set()
    for line in complex_pdb.splitlines():
        if not (
            line.startswith("ATOM  ")
            and len(line) >= 27
            and line[21] == chain_id
        ):
            continue
        label = (int(line[22:26]), line[26])
        if label not in seen_source:
            seen_source.add(label)
            source_labels.append(label)

    target_labels = list(source_labels)
    for gap in sorted(gaps, key=lambda item: int(item["insertion_index"]), reverse=True):
        length = int(gap["length"])
        author_start = gap.get("author_start")
        author_end = gap.get("author_end")
        if author_start is None or author_end is None:
            insertion_index = int(gap["insertion_index"])
            if not 0 < insertion_index < len(source_labels):
                raise ValueError(
                    f"Gap in chain {chain} needs confirmed deposited residue numbers"
                )
            left_number = source_labels[insertion_index - 1][0]
            right_number = source_labels[insertion_index][0]
            if right_number - left_number - 1 != length:
                raise ValueError(
                    f"Gap in chain {chain} needs confirmed deposited residue numbers"
                )
            author_start, author_end = left_number + 1, right_number - 1
        author_start = int(author_start)
        author_end = int(author_end)
        if author_end - author_start + 1 != length:
            raise ValueError(
                f"Gap {chain}:{author_start}-{author_end} numbering does not match "
                f"its {length}-residue sequence"
            )
        target_labels[int(gap["insertion_index"]) : int(gap["insertion_index"])] = [
            (number, " ") for number in range(author_start, author_end + 1)
        ]

    model_labels: list[tuple[str, int, str]] = []
    seen_model: set[tuple[str, int, str]] = set()
    for line in model_pdb.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 27:
            continue
        label = (line[21], int(line[22:26]), line[26])
        if label not in seen_model:
            seen_model.add(label)
            model_labels.append(label)
    if len(model_labels) != len(target_labels):
        raise RuntimeError(
            f"MODELLER chain {chain} has {len(model_labels)} residues; "
            f"expected {len(target_labels)} for author-number restoration"
        )
    label_map = dict(zip(model_labels, target_labels, strict=True))
    output: list[str] = []
    for line in model_pdb.splitlines():
        if line.startswith("ATOM  ") and len(line) >= 27:
            old = (line[21], int(line[22:26]), line[26])
            number, insertion = label_map[old]
            line = line[:22] + f"{number:4d}{insertion}" + line[27:]
        output.append(line)
    return "\n".join(output) + "\n"


def _merge_modeled_chain(
    complex_pdb: str,
    model_pdb: str,
    *,
    chain: str,
) -> tuple[str, dict[str, Any]]:
    chain_id = " " if chain == "_" else chain
    model_atom_lines = [
        line
        for line in model_pdb.splitlines()
        if line.startswith("ATOM  ") and len(line) >= 27
    ]
    model_chains = sorted({line[21] for line in model_atom_lines})
    source_model_chain = chain_id if chain_id in model_chains else (
        model_chains[0] if len(model_chains) == 1 else ""
    )
    model_lines = [
        line[:21] + chain_id + line[22:]
        for line in model_atom_lines
        if line[21] == source_model_chain
    ]
    if not model_lines:
        raise RuntimeError(f"MODELLER produced no coordinates for chain {chain}")
    original = complex_pdb.splitlines()
    first_index = next(
        (
            index
            for index, line in enumerate(original)
            if line.startswith("ATOM  ") and len(line) >= 27 and line[21] == chain_id
        ),
        None,
    )
    if first_index is None:
        raise RuntimeError(f"Could not locate source chain {chain} for MODELLER merge")
    retained = [
        line
        for line in original
        if not (
            line.startswith("ATOM  ")
            and len(line) >= 27
            and line[21] == chain_id
        )
    ]
    removed_before = sum(
        1
        for line in original[:first_index]
        if line.startswith("ATOM  ") and len(line) >= 27 and line[21] == chain_id
    )
    insertion_index = first_index - removed_before
    merged = retained[:insertion_index] + model_lines + retained[insertion_index:]
    original_ligands = [
        line for line in original if line.startswith("HETATM")
    ]
    merged_ligands = [line for line in merged if line.startswith("HETATM")]
    if original_ligands != merged_ligands:
        raise RuntimeError("MODELLER gap merge changed ligand or heterogen coordinate records")
    return _renumber_pdb_atom_serials("\n".join(merged) + "\n"), {
        "chain": chain,
        "modeled_atom_count": len(model_lines),
        "retained_heterogen_atom_count": len(merged_ligands),
    }


def model_internal_gaps_with_modeller(
    pdb_data: str,
    *,
    work_dir: Path,
    max_internal_gap: int = 15,
    model_count: int = 10,
    modeller_python: Path = DEFAULT_MODELLER_PYTHON,
    user_gap_definitions: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Model every eligible internal sequence gap as a MODELLER ensemble."""
    if not 1 <= int(model_count) <= 50:
        raise ValueError("MODELLER internal-gap model count must be between 1 and 50")
    detected = detect_modeller_internal_gaps(
        pdb_data,
        max_internal_gap=int(max_internal_gap),
        user_gap_definitions=user_gap_definitions,
    )
    report: dict[str, Any] = {
        "enabled": True,
        "engine": "MODELLER",
        "max_internal_gap": int(max_internal_gap),
        "model_count": int(model_count),
        "user_gap_definitions": list(user_gap_definitions or []),
        "chains": [],
        "modeled_gaps": [],
        "skipped_gaps": [],
    }
    current = pdb_data
    runner = PROJECT_DIR / "mn_ligand" / "workflows" / "modeller_internal_gap_runner.py"
    for chain, chain_report in detected.items():
        eligible = [gap for gap in chain_report["gaps"] if gap["eligible"]]
        report["skipped_gaps"].extend(
            gap for gap in chain_report["gaps"] if not gap["eligible"]
        )
        if not eligible:
            continue
        chain_dir = work_dir / f"chain_{chain}"
        chain_dir.mkdir(parents=True, exist_ok=True)
        template_path = chain_dir / "template.pdb"
        alignment_path = chain_dir / "alignment.ali"
        ranges_path = chain_dir / "ranges.json"
        native_result = chain_dir / "native_result.json"
        template_path.write_text(_selected_chain_template(current, chain))
        alignment_path.write_text(
            _internal_gap_pir(
                template_name=template_path.stem,
                chain=chain,
                start=int(chain_report["start"]),
                end=int(chain_report["end"]),
                template_alignment=str(chain_report["template_alignment"]),
                target_alignment=str(chain_report["target_alignment"]),
            )
        )
        _write_json(ranges_path, eligible)
        command = [
            str(modeller_python),
            str(runner),
            "--alignment",
            alignment_path.name,
            "--template",
            template_path.name,
            "--ranges",
            ranges_path.name,
            "--models",
            str(int(model_count)),
            "--output",
            native_result.name,
        ]
        process = subprocess.run(
            command,
            cwd=chain_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode or not native_result.is_file():
            error = (process.stderr or process.stdout or "MODELLER gap modeling failed")[-4000:]
            raise RuntimeError(error)
        native = json.loads(native_result.read_text())
        successful = [
            item
            for item in native.get("models") or []
            if not item.get("failure")
            and item.get("name")
            and (chain_dir / str(item["name"])).is_file()
        ]
        if not successful:
            raise RuntimeError(f"MODELLER produced no successful gap models for chain {chain}")
        # DOPE alone is not a publication criterion. Evaluate candidates in
        # DOPE order and prefer the first one without a severe target clash.
        # If every model needs relaxation, retain the least-clashing model and
        # require the downstream PDBFixer/OpenMM validation stage to clear it.
        evaluated_models: list[dict[str, Any]] = []
        merged_candidates: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        for candidate in sorted(
            successful,
            key=lambda item: (
                float(item["dope"])
                if item.get("dope") is not None
                else float("inf"),
                str(item["name"]),
            ),
        ):
            candidate_path = chain_dir / str(candidate["name"])
            numbered_model = _restore_modeled_author_numbering(
                current,
                candidate_path.read_text(),
                chain=chain,
                gaps=eligible,
            )
            merged_candidate, candidate_merge = _merge_modeled_chain(
                current,
                numbered_model,
                chain=chain,
            )
            merged_candidate, removed_oxt = remove_internal_oxt(merged_candidate)
            geometry = audit_target_geometry(merged_candidate)
            evaluation = {
                **candidate,
                "target_geometry_valid": bool(geometry["valid"]),
                "severe_clash_count": int(geometry["severe_clash_count"]),
                "severe_clashes": list(geometry["severe_clashes"])[:10],
                "removed_internal_oxt_count": len(removed_oxt),
            }
            evaluated_models.append(evaluation)
            merged_candidates.append((evaluation, merged_candidate, candidate_merge))
        selected_evaluation, current, merge_report = min(
            merged_candidates,
            key=lambda item: (
                0 if item[0]["target_geometry_valid"] else 1,
                int(item[0]["severe_clash_count"]),
                float(item[0]["dope"])
                if item[0].get("dope") is not None
                else float("inf"),
                str(item[0]["name"]),
            ),
        )
        best = next(
            item for item in successful if item["name"] == selected_evaluation["name"]
        )
        chain_entry = {
            "chain": chain,
            "command": command,
            "alignment": alignment_path.relative_to(work_dir).as_posix(),
            "models": successful,
            "model_geometry_evaluations": evaluated_models,
            "selected_model": best,
            "selected_model_requires_openmm_relaxation": not bool(
                selected_evaluation["target_geometry_valid"]
            ),
            "merge": merge_report,
            "gaps": eligible,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        report["chains"].append(chain_entry)
        report["modeled_gaps"].extend(eligible)
    report["success"] = True
    return current, report


def create_protein_import_job(
    data: str,
    *,
    filename: str,
    source: str,
    pdb_id: str = "",
) -> JobRecord:
    structure_format = detect_structure_format(data, filename)
    summary = structure_summary(data, structure_format)
    run_id = str(uuid4())
    run_dir = runs_root() / "protein-import" / run_id
    artifact_dir = run_dir / "artifacts" / "imported"
    report_dir = run_dir / "artifacts" / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True)

    target_path = artifact_dir / _safe_filename(filename, structure_format)
    target_path.write_text(data if data.endswith("\n") else data + "\n")
    residue_mapping_path = report_dir / "residue_mapping.json"
    residue_mapping = (
        identity_residue_mapping(
            data,
            source_run_id=run_id,
            source_label=pdb_id.upper() or Path(filename).name,
        )
        if structure_format == "pdb"
        else {
            "schema_version": 1,
            "source_run_id": run_id,
            "source_label": pdb_id.upper() or Path(filename).name,
            "mapping_basis": "mapping deferred until PDB conversion",
            "residues": [],
        }
    )
    _write_json(residue_mapping_path, residue_mapping)
    report_path = report_dir / "import_report.json"
    import_report = {
        "source": source,
        "pdb_id": pdb_id.upper(),
        "original_filename": Path(filename).name,
        **summary,
    }
    _write_json(report_path, import_report)
    input_payload = {
        "source": source,
        "pdb_id": pdb_id.upper(),
        "original_filename": Path(filename).name,
    }
    _write_json(run_dir / "input.json", input_payload)
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "protein_import",
        "status": "completed",
        "source": source,
        "pdb_id": pdb_id.upper(),
        "structure_format": structure_format,
        "residue_numbering": "source_author",
        "ligands": summary["ligands"],
        "receptor": summary["receptor"],
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "result.json",
        {
            "success": True,
            "imported_target": target_path.relative_to(run_dir).as_posix(),
            "import_report": report_path.relative_to(run_dir).as_posix(),
            "residue_mapping": residue_mapping_path.relative_to(run_dir).as_posix(),
            **summary,
        },
    )
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(
                run_dir,
                target_path,
                "imported_target",
                role="target",
                metadata={
                    "format": structure_format,
                    "source": source,
                    "pdb_id": pdb_id.upper(),
                    "ligands": summary["ligands"],
                    "receptor": summary["receptor"],
                },
            ),
            ArtifactRef.from_path(run_dir, report_path, "import_report", role="report"),
            ArtifactRef.from_path(
                run_dir,
                residue_mapping_path,
                "residue_mapping",
                role="author_numbering",
                metadata={"source_run_id": run_id, "pdb_id": pdb_id.upper()},
            ),
        ],
    )
    return JobRecord.load(run_dir, task_group="protein-import")


def imported_target(job: JobRecord) -> ArtifactRef:
    if job.task_group != "protein-import" or job.artifact_manifest is None:
        raise ValueError("Protein cleaning requires a protein-import job")
    targets = job.artifact_manifest.by_type("imported_target")
    if not targets:
        raise ValueError(f"Import job {job.run_id} has no imported_target artifact")
    return targets[0]


def load_protein_import_job(run_id: str) -> JobRecord:
    run_dir = resolve_run_dir("protein-import", run_id)
    if run_dir is None:
        raise FileNotFoundError(f"Protein import job not found: {run_id}")
    return JobRecord.load(run_dir, task_group="protein-import")


def list_protein_import_jobs() -> list[JobRecord]:
    root = runs_root() / "protein-import"
    if not root.is_dir():
        return []
    return sorted(
        (JobRecord.load(path, task_group="protein-import") for path in root.iterdir() if path.is_dir()),
        key=lambda job: (job.created_at, job.run_dir.stat().st_mtime),
        reverse=True,
    )


def list_protein_cleaning_jobs() -> list[JobRecord]:
    root = runs_root() / "protein-cleaning"
    if not root.is_dir():
        return []
    return sorted(
        (JobRecord.load(path, task_group="protein-cleaning") for path in root.iterdir() if path.is_dir()),
        key=lambda job: (job.created_at, job.run_dir.stat().st_mtime),
        reverse=True,
    )


def prepared_target(job: JobRecord) -> ArtifactRef:
    if job.task_group != "protein-cleaning" or job.artifact_manifest is None:
        raise ValueError("A prepared target requires a protein-cleaning job")
    targets = job.artifact_manifest.by_type("prepared_target")
    if not targets:
        raise ValueError(f"Cleaning job {job.run_id} has no prepared_target artifact")
    return targets[0]


def load_protein_cleaning_job(run_id: str) -> JobRecord:
    run_dir = resolve_run_dir("protein-cleaning", run_id)
    if run_dir is None:
        raise FileNotFoundError(f"Protein cleaning job not found: {run_id}")
    return JobRecord.load(run_dir, task_group="protein-cleaning")


def _protein_only_pdb(pdb_data: str) -> str:
    lines: list[str] = []
    for line in pdb_data.splitlines():
        if line.startswith("ATOM") and not is_bound_ligand_atom(line):
            lines.append(line)
        elif line.startswith(("TER", "MODEL", "ENDMDL", "CRYST1", "HEADER", "TITLE", "REMARK")):
            lines.append(line)
    return "\n".join(lines + ["END", ""])


def resolve_present_protein_chains(
    pdb_data: str,
    selected_chains: list[str],
) -> set[str]:
    """Resolve a single-chain selection across historical OpenMM ID rewrites."""
    selected = set(selected_chains)
    available = {
        line[21].strip() or "_"
        for line in pdb_data.splitlines()
        if line.startswith("ATOM") and len(line) >= 22
    }
    if (
        selected
        and not selected.intersection(available)
        and len(selected) == len(available) == 1
    ):
        return available
    return selected


def inline_structure_preview_data(complex_pdb: str, protein_pdb: str) -> str:
    """Use a complex preview only when it actually contains a protein."""
    if any(line.startswith("ATOM") for line in complex_pdb.splitlines()):
        return complex_pdb
    return protein_pdb


def _cleaning_command(
    *,
    image: str,
    run_dir: Path,
    source_path: Path,
    runner_input: Path,
    native_result: Path,
    use_gpu: bool,
) -> list[str]:
    structure_image = image.startswith("ovolig-structure:")
    shm_size = os.getenv("MN_MD_DOCKER_SHM_SIZE", "64g").strip()
    python_command = ["micromamba", "run", "-n", "base", "python"] if structure_image else ["python"]
    tool_id = "openmm_md" if use_gpu else "protein_cleaning"
    spec = DockerRunSpec(
        tool=registered_tool(tool_id, image=image),
        command=(
            *python_command,
            "-m",
            "mn_ligand.workflows.bound_ligand_md",
            "prepare",
            "--input",
            f"/output/{runner_input.name}",
            "--output",
            f"/output/{native_result.name}",
        ),
        mounts=(
            DockerMount(PROJECT_DIR, "/mn-ligand", read_only=True),
            DockerMount(run_dir, "/output"),
            DockerMount(source_path, f"/input/{source_path.name}", read_only=True),
        ),
        environment={"PYTHONPATH": "/mn-ligand"},
        gpu_enabled=use_gpu,
        shm_size=shm_size,
        use_host_user=False,
    )
    return build_docker_command(spec)


def _fail_cleaning_job(
    run_dir: Path,
    metadata: dict[str, Any],
    error: str,
    *,
    returncode: int | None,
) -> JobRecord:
    _write_json(run_dir / "result.json", {"success": False, "error": error, "returncode": returncode})
    write_artifact_manifest(run_dir, [])
    completed_at = _utc_now_iso()
    metadata.update({"status": "failed", "error": error, "updated_at": completed_at, "completed_at": completed_at})
    _write_json(run_dir / "metadata.json", metadata)
    return JobRecord.load(run_dir, task_group="protein-cleaning")


def run_protein_cleaning_job(
    import_run_id: str,
    *,
    image: str = DEFAULT_PROTEIN_CLEANING_IMAGE,
    use_gpu: bool = False,
    map_modified_residues: bool = True,
    skip_terminal_missing_residues: bool = True,
    max_internal_gap: int = 15,
    refine_rebuilt_positions: bool = True,
    biological_assembly_id: str = "",
    preserve_nonwater_heterogens: bool = False,
    noncanonical_replacements: list[dict[str, str]] | None = None,
    internal_gap_definitions: list[dict[str, Any]] | None = None,
) -> tuple[JobRecord, dict[str, Any]]:
    source_job = load_protein_import_job(import_run_id)
    source_ref = imported_target(source_job)
    source_path = source_ref.resolve(source_job.run_dir, must_exist=True)
    if source_path is None:
        raise FileNotFoundError(source_ref.path)

    run_id = str(uuid4())
    run_dir = runs_root() / "protein-cleaning" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    created_at = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "protein_cleaning",
        "status": "running",
        "tool": "MODELLER + ligandx-pdbfixer",
        "source": "MODELLER + ligandx-pdbfixer",
        "parent_run_id": source_job.run_id,
        "import_run_id": source_job.run_id,
        "pdb_id": source_job.metadata.get("pdb_id") or "",
        "ligands": list(source_job.metadata.get("ligands") or []),
        "receptor": dict(source_job.metadata.get("receptor") or {}),
        "docker_image": image,
        "use_gpu": use_gpu,
        "created_at": created_at,
        "updated_at": created_at,
    }
    _write_json(run_dir / "metadata.json", metadata)
    modeller_report: dict[str, Any] = {
        "enabled": False,
        "engine": "MODELLER",
        "replacements": [],
    }
    gap_report: dict[str, Any] = {
        "enabled": False,
        "engine": "MODELLER",
        "modeled_gaps": [],
        "skipped_gaps": [],
    }
    sequence_sync_report: dict[str, Any] = {}
    source_data = source_path.read_text()
    working_data = source_data
    if noncanonical_replacements:
        working_data, modeller_report = repair_noncanonical_residues_with_modeller(
            working_data,
            noncanonical_replacements,
            work_dir=run_dir / "artifacts" / "modeller" / "residue_repair",
        )
        working_data, sequence_sync_report = synchronize_repaired_sequence_records(
            working_data,
            list(modeller_report.get("replacements") or []),
        )
        map_modified_residues = False
    unresolved_noncanonical = detect_noncanonical_residues(working_data)
    if not unresolved_noncanonical:
        working_data, gap_report = model_internal_gaps_with_modeller(
            working_data,
            work_dir=run_dir / "artifacts" / "modeller" / "internal_gaps",
            max_internal_gap=int(max_internal_gap),
            model_count=10,
            user_gap_definitions=internal_gap_definitions,
        )
    else:
        gap_report["reason"] = (
            "Internal-gap modeling requires all noncanonical polymer sites to be "
            "resolved explicitly first"
        )
        gap_report["unresolved_noncanonical"] = unresolved_noncanonical
    cleaning_source_path = run_dir / "artifacts" / "modeller" / "repaired_source.pdb"
    cleaning_source_path.parent.mkdir(parents=True, exist_ok=True)
    cleaning_source_path.write_text(working_data)
    input_payload = {
        "source_task_group": "protein-import",
        "input_artifact": source_ref.to_dict(),
        "parameters": {
            "clean_protein": True,
            "map_modified_residues": map_modified_residues,
            "ph": 7.4,
            "add_missing_residues": False,
            "skip_terminal_missing_residues": skip_terminal_missing_residues,
            "max_internal_gap": int(max_internal_gap),
            "refine_rebuilt_positions": refine_rebuilt_positions,
            "biological_assembly_id": str(biological_assembly_id or ""),
            "preserve_nonwater_heterogens": bool(preserve_nonwater_heterogens),
            "noncanonical_replacements": list(noncanonical_replacements or []),
            "internal_gap_engine": "MODELLER",
            "internal_gap_model_count": 10,
            "internal_gap_definitions": list(internal_gap_definitions or []),
        },
    }
    _write_json(run_dir / "input.json", input_payload)
    runner_input = run_dir / "runner_input.json"
    native_result = run_dir / "native_result.json"
    _write_json(native_result, {})
    native_result.chmod(0o666)
    _write_json(
        runner_input,
        {
            "pdb_id": source_job.metadata.get("pdb_id") or "protein",
            "pdb_path": f"/input/{cleaning_source_path.name}",
            "clean_protein": True,
            "map_modified_residues": map_modified_residues,
            "ph": 7.4,
            "add_missing_residues": False,
            "skip_terminal_missing_residues": skip_terminal_missing_residues,
            "max_internal_gap": int(max_internal_gap),
            "refine_rebuilt_positions": refine_rebuilt_positions,
            "biological_assembly_id": str(biological_assembly_id or ""),
            "preserve_nonwater_heterogens": bool(preserve_nonwater_heterogens),
        },
    )
    command = _cleaning_command(
        image=image,
        run_dir=run_dir,
        source_path=cleaning_source_path,
        runner_input=runner_input,
        native_result=native_result,
        use_gpu=use_gpu,
    )
    write_registered_command_record(
        run_dir,
        tool_id="openmm_md" if use_gpu else "protein_cleaning",
        commands=(command,),
        image=image,
    )
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
    except Exception as exc:
        native_result.chmod(0o644)
        error = f"Could not start protein-cleaning container: {exc}"
        (run_dir / "stdout.log").write_text("")
        (run_dir / "stderr.log").write_text(error + "\n")
        return _fail_cleaning_job(run_dir, metadata, error, returncode=None), {"success": False, "error": error}
    native_result.chmod(0o644)
    (run_dir / "stdout.log").write_text(process.stdout or "")
    (run_dir / "stderr.log").write_text(process.stderr or "")
    try:
        native_payload = json.loads(native_result.read_text()) if native_result.is_file() else {}
    except (OSError, ValueError):
        native_payload = {}

    success = process.returncode == 0 and native_payload.get("success") is True
    if not success:
        error = str(native_payload.get("error") or (process.stderr or "")[-4000:] or "Protein cleaning failed")
        return _fail_cleaning_job(run_dir, metadata, error, returncode=process.returncode), native_payload

    prepared_data = str(native_payload.get("prepared_pdb_data") or "")
    if not prepared_data.strip():
        error = "Protein cleaner returned no prepared structure"
        return _fail_cleaning_job(run_dir, metadata, error, returncode=process.returncode), native_payload
    prepared_target = _protein_only_pdb(prepared_data)
    prepared_dir = run_dir / "artifacts" / "prepared"
    report_dir = run_dir / "artifacts" / "reports"
    prepared_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    prepared_path = prepared_dir / "prepared_target.pdb"
    prepared_path.write_text(prepared_target)
    residue_mapping_path = report_dir / "residue_mapping.json"
    residue_mapping = derive_residue_mapping(
        prepared_target,
        source_data,
        source_run_id=source_job.run_id,
        source_label=str(source_job.metadata.get("pdb_id") or source_path.name),
    )
    _write_json(residue_mapping_path, residue_mapping)
    report_path = report_dir / "repair_report.json"
    repair_report = {
        "protein_cleaned": bool(native_payload.get("protein_cleaned")),
        "components": native_payload.get("components") or {},
        "modified_residue_mapping": native_payload.get("modified_residue_mapping") or {},
        "pdbfixer": native_payload.get("repair_report") or {},
        "modeller_noncanonical_repair": modeller_report,
        "modeller_internal_gap_repair": gap_report,
        "sequence_record_synchronization": sequence_sync_report,
        "input_artifact": source_ref.to_dict(),
        "parameters": input_payload["parameters"],
    }
    _write_json(report_path, repair_report)
    result_payload = {
        "success": True,
        "prepared_target": prepared_path.relative_to(run_dir).as_posix(),
        "repair_report": report_path.relative_to(run_dir).as_posix(),
        "residue_mapping": residue_mapping_path.relative_to(run_dir).as_posix(),
        "protein_cleaned": repair_report["protein_cleaned"],
        "components": repair_report["components"],
        "ligand_count": len(native_payload.get("ligands") or []),
        "ligands": list(source_job.metadata.get("ligands") or []),
        "receptor": dict(source_job.metadata.get("receptor") or {}),
        "modeller_noncanonical_repair": modeller_report,
        "modeller_internal_gap_repair": gap_report,
    }
    _write_json(run_dir / "result.json", result_payload)
    manifest_items = [
        ArtifactRef.from_path(
            run_dir,
            prepared_path,
            "prepared_target",
            role="receptor",
            metadata={
                "source_run_id": source_job.run_id,
                "protonation_ph": 7.4,
                "ligands": list(source_job.metadata.get("ligands") or []),
                "receptor": dict(source_job.metadata.get("receptor") or {}),
                "noncanonical_replacements": list(noncanonical_replacements or []),
            },
        ),
        ArtifactRef.from_path(run_dir, report_path, "repair_report", role="report"),
        ArtifactRef.from_path(
            run_dir,
            residue_mapping_path,
            "residue_mapping",
            role="author_numbering",
            metadata={
                "source_run_id": source_job.run_id,
                "pdb_id": source_job.metadata.get("pdb_id") or "",
            },
        ),
    ]
    if noncanonical_replacements or gap_report.get("modeled_gaps"):
        manifest_items.append(
            ArtifactRef.from_path(
                run_dir,
                cleaning_source_path,
                "modeller_repaired_complex",
                role="intermediate",
                metadata={"noncanonical_replacements": list(noncanonical_replacements)},
            )
        )
    write_artifact_manifest(
        run_dir,
        manifest_items,
    )
    completed_at = _utc_now_iso()
    metadata.update(
        {
            "status": "completed",
            "updated_at": completed_at,
            "completed_at": completed_at,
            "residue_numbering": "source_author",
            "residue_mapping_artifact_id": next(
                item.artifact_id
                for item in manifest_items
                if item.artifact_type == "residue_mapping"
            ),
        }
    )
    _write_json(run_dir / "metadata.json", metadata)
    native_payload["modeller_noncanonical_repair"] = modeller_report
    native_payload["modeller_internal_gap_repair"] = gap_report
    return JobRecord.load(run_dir, task_group="protein-cleaning"), native_payload


def _fetch_rcsb_component(ccd_id: str) -> dict[str, Any]:
    code = ccd_id.strip().upper()
    if not code or code in {"LIG", "UNL"}:
        return {}
    request = urllib.request.Request(
        f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}",
        headers={"User-Agent": "mn-ligand/ligand-metadata-backfill"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}
    chemical = payload.get("chem_comp") if isinstance(payload.get("chem_comp"), dict) else {}
    descriptor = (
        payload.get("rcsb_chem_comp_descriptor")
        if isinstance(payload.get("rcsb_chem_comp_descriptor"), dict)
        else {}
    )
    return {
        "ccd_id": code,
        "name": str(chemical.get("name") or ""),
        "formula": str(chemical.get("formula") or ""),
        "molecular_weight": chemical.get("formula_weight"),
        "type": str(chemical.get("type") or ""),
        "smiles": str(descriptor.get("SMILES_stereo") or descriptor.get("SMILES") or ""),
        "inchi": str(descriptor.get("InChI") or ""),
        "inchikey": str(descriptor.get("InChIKey") or ""),
    }


def _fetch_rcsb_receptor(pdb_id: str) -> dict[str, Any]:
    code = pdb_id.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{4}", code):
        return {}

    def fetch(resource: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"https://data.rcsb.org/rest/v1/core/{resource}",
            headers={"User-Agent": "mn-ligand/receptor-metadata-backfill"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    try:
        entry = fetch(f"entry/{code}")
    except Exception:
        return {}
    identifiers = entry.get("rcsb_entry_container_identifiers") or {}
    entry_info = entry.get("rcsb_entry_info") or {}
    accession = entry.get("rcsb_accession_info") or {}
    entities: list[dict[str, Any]] = []
    for entity_id in identifiers.get("polymer_entity_ids") or []:
        try:
            payload = fetch(f"polymer_entity/{code}/{entity_id}")
        except Exception:
            continue
        polymer = payload.get("rcsb_polymer_entity") or {}
        entity_poly = payload.get("entity_poly") or {}
        container = payload.get("rcsb_polymer_entity_container_identifiers") or {}
        sources = payload.get("rcsb_entity_source_organism") or []
        hosts = payload.get("rcsb_entity_host_organism") or []
        genes = [
            str(gene.get("value") or "")
            for source in sources
            for gene in source.get("rcsb_gene_name") or []
            if gene.get("value")
        ]
        entities.append(
            {
                "entity_id": str(entity_id),
                "name": str(polymer.get("pdbx_description") or ""),
                "polymer_type": str(entity_poly.get("rcsb_entity_polymer_type") or entity_poly.get("type") or ""),
                "chains": list(container.get("auth_asym_ids") or container.get("asym_ids") or []),
                "sequence_length": entity_poly.get("rcsb_sample_sequence_length"),
                "uniprot_ids": list(container.get("uniprot_ids") or []),
                "source_organisms": list(
                    dict.fromkeys(
                        str(source.get("ncbi_scientific_name") or source.get("scientific_name") or "")
                        for source in sources
                        if source.get("ncbi_scientific_name") or source.get("scientific_name")
                    )
                ),
                "source_taxonomy_ids": list(
                    dict.fromkeys(source.get("ncbi_taxonomy_id") for source in sources if source.get("ncbi_taxonomy_id"))
                ),
                "expression_hosts": list(
                    dict.fromkeys(
                        str(host.get("ncbi_scientific_name") or host.get("scientific_name") or "")
                        for host in hosts
                        if host.get("ncbi_scientific_name") or host.get("scientific_name")
                    )
                ),
                "gene_names": list(dict.fromkeys(genes)),
            }
        )
    methods = [str(item.get("method") or "") for item in entry.get("exptl") or [] if item.get("method")]
    resolutions = entry_info.get("resolution_combined") or []
    return {
        "pdb_id": code,
        "title": str((entry.get("struct") or {}).get("title") or ""),
        "experimental_method": ", ".join(methods),
        "resolution_angstrom": min(resolutions) if resolutions else None,
        "deposition_date": str(accession.get("deposit_date") or ""),
        "release_date": str(accession.get("initial_release_date") or ""),
        "revision_date": str(accession.get("revision_date") or ""),
        "polymer_composition": str(entry_info.get("polymer_composition") or ""),
        "entities": entities,
        "metadata_source": "rcsb_data_api",
    }


def _reference_smiles_for_job(job: JobRecord) -> str:
    smiles = str(job.metadata.get("ligand_smiles") or "").strip()
    if smiles:
        return smiles
    if not job.artifact_manifest:
        return ""
    for artifact in job.artifact_manifest.artifacts:
        if artifact.role != "reference_smiles" and not artifact.path.lower().endswith((".smi", ".smiles")):
            continue
        path = artifact.resolve(job.run_dir, must_exist=True)
        if path is not None:
            return path.read_text(errors="replace").strip().split()[0]
    return ""


def _smiles_details(smiles: str) -> dict[str, Any]:
    if not smiles:
        return {}
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return {"smiles": smiles}
        return {
            "smiles": Chem.MolToSmiles(molecule, canonical=True),
            "formula": rdMolDescriptors.CalcMolFormula(molecule),
            "molecular_weight": round(float(Descriptors.MolWt(molecule)), 3),
        }
    except Exception:
        return {"smiles": smiles}


def _target_structure_path(job: JobRecord) -> Path | None:
    if not job.artifact_manifest:
        return None
    for artifact_type in ("prepared_complex", "imported_target", "prepared_target", "prepared_receptor"):
        artifacts = job.artifact_manifest.by_type(artifact_type)
        if artifacts:
            path = artifacts[0].resolve(job.run_dir, must_exist=True)
            if path is not None:
                return path
    return None


def backfill_ligand_metadata(*, fetch_rcsb: bool = True, write: bool = False) -> dict[str, Any]:
    """Recover ligand metadata for existing target jobs without changing artifacts."""
    jobs: list[JobRecord] = []
    for group in ("protein-import", "protein-cleaning", "structure-jobs"):
        root = runs_root() / group
        if not root.is_dir():
            continue
        jobs.extend(
            job
            for job in (JobRecord.load(path, task_group=group) for path in sorted(root.iterdir()) if path.is_dir())
            if job.status == "completed"
        )
    by_id = {job.run_id: job for job in jobs}
    component_cache: dict[str, dict[str, Any]] = {}
    receptor_cache: dict[str, dict[str, Any]] = {}
    changed: list[dict[str, Any]] = []
    for job in jobs:
        path = _target_structure_path(job)
        coordinate_ligands = (
            parse_bound_ligands(path.read_text(errors="replace"))
            if path is not None and path.suffix.lower() in {".pdb", ".ent"}
            else []
        )
        existing_ligands = job.metadata.get("ligands")
        source_ligands: list[dict[str, Any]] = (
            [dict(item) for item in existing_ligands if isinstance(item, dict)]
            if isinstance(existing_ligands, list) and existing_ligands
            else []
        )
        if not source_ligands and job.task_group == "protein-import" and path is not None:
            source_ligands = structure_ligands(path.read_text(errors="replace"), "pdb")
        elif not source_ligands and job.task_group == "protein-cleaning":
            parent = by_id.get(str(job.metadata.get("import_run_id") or job.parent_run_id))
            source_ligands = list(parent.metadata.get("ligands") or []) if parent else []

        selected_code = str(job.metadata.get("ligand_key") or "").partition("|")[0].strip().upper()
        ligand_id = str(job.metadata.get("ligand_id") or "").strip()
        smiles = _reference_smiles_for_job(job)
        if not source_ligands and job.task_group == "structure-jobs" and (selected_code or ligand_id or coordinate_ligands):
            source_is_pdb = str(job.metadata.get("source") or "").lower() == "pdb"
            source_ligands = [
                {
                    **(coordinate_ligands[0] if len(coordinate_ligands) == 1 else {}),
                    "ccd_id": selected_code
                    if source_is_pdb and selected_code not in {"LIG", "UNL"}
                    else "",
                    "reference_ligand_ccd_id": selected_code if not source_is_pdb else "",
                    "coordinate_resname": str(coordinate_ligands[0].get("resname") or "")
                    if len(coordinate_ligands) == 1
                    else "",
                    "name": ligand_id,
                    **_smiles_details(smiles),
                }
            ]

        enriched: list[dict[str, Any]] = []
        for ligand in source_ligands:
            item = dict(ligand)
            code = str(item.get("ccd_id") or item.get("resname") or "").upper()
            if fetch_rcsb and code not in {"", "LIG", "UNL"}:
                if code not in component_cache:
                    component_cache[code] = _fetch_rcsb_component(code)
                remote = component_cache[code]
                for key, value in remote.items():
                    if value not in ("", None, []) and item.get(key) in ("", None, []):
                        item[key] = value
            enriched.append(item)
        pdb_id = str(job.metadata.get("pdb_id") or "").strip().upper()
        receptor = dict(job.metadata.get("receptor") or {})
        if fetch_rcsb and pdb_id:
            if pdb_id not in receptor_cache:
                receptor_cache[pdb_id] = _fetch_rcsb_receptor(pdb_id)
            receptor = receptor_cache[pdb_id] or receptor
        if not receptor:
            for parent_id in (job.metadata.get("import_run_id"), job.metadata.get("source_structure_run_id")):
                parent = by_id.get(str(parent_id or ""))
                if parent and parent.metadata.get("receptor"):
                    receptor = dict(parent.metadata["receptor"])
                    break
        ligands_changed = bool(enriched) and enriched != job.metadata.get("ligands")
        receptor_changed = bool(receptor) and receptor != job.metadata.get("receptor")
        if not ligands_changed and not receptor_changed:
            continue
        changed.append(
            {
                "task_group": job.task_group,
                "run_id": job.run_id,
                "ligands": enriched or list(job.metadata.get("ligands") or []),
                "receptor": receptor,
            }
        )
        if write:
            metadata = dict(job.metadata)
            if enriched:
                metadata["ligands"] = enriched
            if receptor:
                metadata["receptor"] = receptor
            metadata["target_metadata_backfilled_at"] = _utc_now_iso()
            _write_json(job.run_dir / "metadata.json", metadata)
    return {"scanned": len(jobs), "updated": len(changed), "jobs": changed, "written": write}
