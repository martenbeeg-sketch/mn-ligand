from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JobRecord


RESIDUE_MAPPING_SCHEMA_VERSION = 1
POLYMER_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
POLYMER_RESIDUE_ALIASES = {
    "ASH": "ASP",
    "CYM": "CYS",
    "CYX": "CYS",
    "GLH": "GLU",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "LYN": "LYS",
}
ONE_LETTER_RESIDUES = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _pdb_residues(pdb_data: str) -> list[dict[str, Any]]:
    residues: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    ca_coordinates: dict[tuple[str, int, str], tuple[float, float, float]] = {}
    names: dict[tuple[str, int, str], str] = {}
    modified_residues: dict[tuple[str, int, str], str] = {}
    for line in pdb_data.splitlines():
        if not line.startswith("MODRES"):
            continue
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            modified_residues[(fields[3], int(fields[4]), "")] = fields[5]
        except ValueError:
            continue
    for line in pdb_data.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        chain = line[21].strip() or "_"
        try:
            number = int(line[22:26])
        except ValueError:
            continue
        insertion_code = line[26].strip()
        key = (chain, number, insertion_code)
        residue_name = line[17:20].strip().upper()
        residue_name = POLYMER_RESIDUE_ALIASES.get(residue_name, residue_name)
        if residue_name not in POLYMER_RESIDUES and key not in modified_residues:
            continue
        names[key] = modified_residues.get(key, residue_name)
        if line[12:16].strip() == "CA":
            try:
                ca_coordinates[key] = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError:
                pass
        if key not in seen:
            seen.add(key)
            residues.append(
                {
                    "chain": chain,
                    "residue_number": number,
                    "insertion_code": insertion_code,
                    "residue_name": names[key],
                }
            )
    for residue in residues:
        key = (
            residue["chain"],
            residue["residue_number"],
            residue["insertion_code"],
        )
        residue["ca_coordinates"] = ca_coordinates.get(key)
    return residues


def identity_residue_mapping(
    pdb_data: str,
    *,
    source_run_id: str = "",
    source_label: str = "",
) -> dict[str, Any]:
    residues = _pdb_residues(pdb_data)
    return {
        "schema_version": RESIDUE_MAPPING_SCHEMA_VERSION,
        "source_run_id": source_run_id,
        "source_label": source_label,
        "mapping_basis": "identity author numbering from imported structure",
        "residues": [
            {
                "structure_index": index,
                "structure_chain": residue["chain"],
                "structure_residue_number": residue["residue_number"],
                "structure_insertion_code": residue["insertion_code"],
                "structure_residue_name": residue["residue_name"],
                "native_chain": residue["chain"],
                "native_residue_number": residue["residue_number"],
                "native_insertion_code": residue["insertion_code"],
                "native_residue_name": residue["residue_name"],
                "mapping_method": "identity",
            }
            for index, residue in enumerate(residues, start=1)
        ],
    }


def sequence_author_residue_mapping(
    structure_pdb_data: str,
    author_pdb_data: str,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    structure_chains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    author_chains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for residue in _pdb_residues(structure_pdb_data):
        structure_chains[str(residue["chain"])].append(residue)
    for residue in _pdb_residues(author_pdb_data):
        author_chains[str(residue["chain"])].append(residue)

    candidates: list[
        tuple[int, int, float, str, str, list[tuple[int, int]]]
    ] = []
    for structure_chain, structure_residues in structure_chains.items():
        structure_sequence = "".join(
            ONE_LETTER_RESIDUES.get(
                str(residue["residue_name"]).upper(), "X"
            )
            for residue in structure_residues
        )
        for author_chain, author_residues in author_chains.items():
            author_sequence = "".join(
                ONE_LETTER_RESIDUES.get(
                    str(residue["residue_name"]).upper(), "X"
                )
                for residue in author_residues
            )
            matcher = SequenceMatcher(
                None,
                structure_sequence,
                author_sequence,
                autojunk=False,
            )
            pairs = [
                (block.a + offset, block.b + offset)
                for block in matcher.get_matching_blocks()
                for offset in range(block.size)
            ]
            candidates.append((
                len(pairs),
                int(structure_chain == author_chain),
                matcher.ratio(),
                structure_chain,
                author_chain,
                pairs,
            ))

    mapping: dict[tuple[str, int, str], dict[str, Any]] = {}
    used_structure_chains: set[str] = set()
    used_author_chains: set[str] = set()
    for (
        pair_count,
        _same_chain,
        _ratio,
        structure_chain,
        author_chain,
        pairs,
    ) in sorted(candidates, reverse=True):
        if (
            pair_count == 0
            or structure_chain in used_structure_chains
            or author_chain in used_author_chains
        ):
            continue
        used_structure_chains.add(structure_chain)
        used_author_chains.add(author_chain)
        structure_residues = structure_chains[structure_chain]
        author_residues = author_chains[author_chain]
        for structure_index, author_index in pairs:
            structure_residue = structure_residues[structure_index]
            author_residue = author_residues[author_index]
            mapping[(
                str(structure_residue["chain"]),
                int(structure_residue["residue_number"]),
                str(structure_residue["insertion_code"]),
            )] = {
                "chain": str(author_residue["chain"]),
                "residue_number": int(author_residue["residue_number"]),
                "insertion_code": str(author_residue["insertion_code"]),
                "residue_name": str(author_residue["residue_name"]),
                "mapping_method": "per-chain sequence alignment",
            }
    return mapping


def derive_residue_mapping(
    structure_pdb_data: str,
    native_pdb_data: str,
    *,
    source_run_id: str = "",
    source_label: str = "",
) -> dict[str, Any]:
    structure = _pdb_residues(structure_pdb_data)
    native = _pdb_residues(native_pdb_data)
    sequence_matches = sequence_author_residue_mapping(
        structure_pdb_data,
        native_pdb_data,
    )
    native_by_coordinate: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for residue in native:
        coordinates = residue.get("ca_coordinates")
        if coordinates is not None:
            native_by_coordinate[coordinates].append(residue)

    exact: dict[int, dict[str, Any]] = {}
    chain_offsets: dict[str, Counter[tuple[str, int]]] = defaultdict(Counter)
    for index, residue in enumerate(structure):
        coordinates = residue.get("ca_coordinates")
        matches = native_by_coordinate.get(coordinates, []) if coordinates is not None else []
        if len(matches) != 1:
            continue
        match = matches[0]
        exact[index] = match
        chain_offsets[residue["chain"]][
            (match["chain"], match["residue_number"] - residue["residue_number"])
        ] += 1
    for residue in structure:
        key = (
            str(residue["chain"]),
            int(residue["residue_number"]),
            str(residue["insertion_code"]),
        )
        match = sequence_matches.get(key)
        if match is None:
            continue
        chain_offsets[residue["chain"]][
            (match["chain"], match["residue_number"] - residue["residue_number"])
        ] += 1

    inferred = {
        chain: counts.most_common(1)[0][0]
        for chain, counts in chain_offsets.items()
        if counts
    }
    residues: list[dict[str, Any]] = []
    for index, residue in enumerate(structure, start=1):
        match = exact.get(index - 1)
        method = "coordinate"
        if match is None:
            match = sequence_matches.get((
                str(residue["chain"]),
                int(residue["residue_number"]),
                str(residue["insertion_code"]),
            ))
            method = "per-chain sequence alignment"
        if match is None and residue["chain"] in inferred:
            native_chain, offset = inferred[residue["chain"]]
            match = {
                "chain": native_chain,
                "residue_number": residue["residue_number"] + offset,
                "insertion_code": residue["insertion_code"],
                "residue_name": residue["residue_name"],
            }
            method = "chain-offset-inferred"
        if match is None:
            match = residue
            method = "unmapped-identity-fallback"
        residues.append(
            {
                "structure_index": index,
                "structure_chain": residue["chain"],
                "structure_residue_number": residue["residue_number"],
                "structure_insertion_code": residue["insertion_code"],
                "structure_residue_name": residue["residue_name"],
                "native_chain": match["chain"],
                "native_residue_number": match["residue_number"],
                "native_insertion_code": match["insertion_code"],
                "native_residue_name": match["residue_name"],
                "mapping_method": method,
            }
        )
    return {
        "schema_version": RESIDUE_MAPPING_SCHEMA_VERSION,
        "source_run_id": source_run_id,
        "source_label": source_label,
        "mapping_basis": (
            "C-alpha coordinate identity and per-chain sequence alignment "
            "with author-number offset inference"
        ),
        "residues": residues,
    }


def subset_residue_mapping(
    mapping: dict[str, Any],
    structure_pdb_data: str,
) -> dict[str, Any]:
    structure_residues = _pdb_residues(structure_pdb_data)
    mapping_rows = [
        row
        for row in mapping.get("residues") or []
        if isinstance(row, dict)
    ]
    source_rows = {
        (
            str(row.get("structure_chain") or "_"),
            int(row.get("structure_residue_number")),
            str(row.get("structure_insertion_code") or ""),
        ): row
        for row in mapping_rows
        if row.get("structure_residue_number") is not None
    }

    # A cofolding engine may emit the same protein sequence with residue
    # numbers reset to 1.  In that case an apparently valid numeric key can
    # point at a completely different residue in the pre-cofold mapping.  Use
    # exact keys only when they agree with the residue identity for most of the
    # structure; otherwise align each chain by sequence and transfer the
    # corresponding author-number row by position.
    exact_identity_count = sum(
        1
        for residue in structure_residues
        if (
            source := source_rows.get((
                str(residue["chain"]),
                int(residue["residue_number"]),
                str(residue["insertion_code"]),
            ))
        ) is not None
        and str(source.get("structure_residue_name") or "").upper()
        == str(residue["residue_name"]).upper()
    )
    exact_numbering_is_reliable = (
        not structure_residues
        or exact_identity_count / len(structure_residues) >= 0.8
    )

    structure_chains: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    source_chains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, residue in enumerate(structure_residues):
        structure_chains[str(residue["chain"])].append((index, residue))
    for row in mapping_rows:
        source_chains[str(row.get("structure_chain") or "_")].append(row)

    aligned_rows: dict[int, dict[str, Any]] = {}
    chain_candidates: list[
        tuple[int, int, float, str, str, list[tuple[int, int]]]
    ] = []
    for structure_chain, indexed_residues in structure_chains.items():
        structure_sequence = "".join(
            ONE_LETTER_RESIDUES.get(
                str(residue["residue_name"]).upper(), "X"
            )
            for _index, residue in indexed_residues
        )
        for source_chain, rows_for_chain in source_chains.items():
            source_sequence = "".join(
                ONE_LETTER_RESIDUES.get(
                    str(row.get("structure_residue_name") or "").upper(), "X"
                )
                for row in rows_for_chain
            )
            matcher = SequenceMatcher(
                None, structure_sequence, source_sequence, autojunk=False
            )
            pairs = [
                (block.a + offset, block.b + offset)
                for block in matcher.get_matching_blocks()
                for offset in range(block.size)
            ]
            chain_candidates.append((
                len(pairs),
                int(structure_chain == source_chain),
                matcher.ratio(),
                structure_chain,
                source_chain,
                pairs,
            ))
    used_structure_chains: set[str] = set()
    used_source_chains: set[str] = set()
    for (
        pair_count,
        _same_chain,
        _ratio,
        structure_chain,
        source_chain,
        pairs,
    ) in sorted(chain_candidates, reverse=True):
        if (
            pair_count == 0
            or structure_chain in used_structure_chains
            or source_chain in used_source_chains
        ):
            continue
        used_structure_chains.add(structure_chain)
        used_source_chains.add(source_chain)
        indexed_residues = structure_chains[structure_chain]
        rows_for_chain = source_chains[source_chain]
        for structure_position, source_position in pairs:
            structure_index, _residue = indexed_residues[structure_position]
            aligned_rows[structure_index] = rows_for_chain[source_position]

    rows: list[dict[str, Any]] = []
    for zero_index, residue in enumerate(structure_residues):
        index = zero_index + 1
        key = (
            residue["chain"],
            residue["residue_number"],
            residue["insertion_code"],
        )
        source = source_rows.get(key) if exact_numbering_is_reliable else None
        if source is not None and (
            str(source.get("structure_residue_name") or "").upper()
            != str(residue["residue_name"]).upper()
        ):
            source = None
        source = source or aligned_rows.get(zero_index)
        if source is None:
            source = {
                "native_chain": residue["chain"],
                "native_residue_number": residue["residue_number"],
                "native_insertion_code": residue["insertion_code"],
                "native_residue_name": residue["residue_name"],
                "mapping_method": "unmapped-identity-fallback",
            }
        rows.append(
            {
                **source,
                "structure_index": index,
                "structure_chain": residue["chain"],
                "structure_residue_number": residue["residue_number"],
                "structure_insertion_code": residue["insertion_code"],
                "structure_residue_name": residue["residue_name"],
                "mapping_method": (
                    source.get("mapping_method")
                    if exact_numbering_is_reliable and source_rows.get(key) is source
                    else (
                        "sequence-subset"
                        if zero_index in aligned_rows
                        else source.get("mapping_method")
                    )
                ),
            }
        )
    return {
        **mapping,
        "mapping_basis": f"{mapping.get('mapping_basis') or 'residue mapping'}; subset by retained residues",
        "residues": rows,
    }


def residue_mapping_artifact(job: JobRecord) -> Path | None:
    if job.artifact_manifest is None:
        return None
    artifacts = job.artifact_manifest.by_type("residue_mapping")
    if not artifacts:
        return None
    return artifacts[0].resolve(job.run_dir, must_exist=True)


def load_residue_mapping(job: JobRecord) -> dict[str, Any]:
    """Load author numbering from any supported job-level snapshot.

    New jobs expose the mapping as a typed artifact.  Older MD and analysis
    jobs may instead carry the immutable snapshot beside their inputs or
    embedded in ``input.json``.  Keeping these fallbacks here prevents views
    and workflows from independently guessing residue offsets.
    """
    candidates: list[Path] = []
    artifact = residue_mapping_artifact(job)
    if artifact is not None:
        candidates.append(artifact)
    candidates.extend(
        (
            job.run_dir / "source_residue_mapping.json",
            job.run_dir / "input" / "source_residue_mapping.json",
            job.run_dir / "artifacts" / "residue_mapping.json",
        )
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(
            payload.get("residues"), list
        ):
            return payload

    input_path = job.run_dir / "input.json"
    if input_path.is_file():
        try:
            input_payload = json.loads(input_path.read_text())
        except (OSError, ValueError, TypeError):
            input_payload = {}
        embedded = (
            input_payload.get("residue_mapping")
            if isinstance(input_payload, dict)
            else None
        )
        if isinstance(embedded, dict) and isinstance(
            embedded.get("residues"), list
        ):
            return embedded
    return {}


def store_residue_mapping(
    job: JobRecord,
    mapping: dict[str, Any],
    *,
    relative_path: str = "artifacts/residue_mapping.json",
) -> JobRecord:
    path = job.run_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2) + "\n")
    artifact = ArtifactRef.from_path(
        job.run_dir,
        path,
        "residue_mapping",
        role="author_numbering",
        metadata={
            "source_run_id": str(mapping.get("source_run_id") or ""),
            "source_label": str(mapping.get("source_label") or ""),
        },
    )
    existing = [
        item
        for item in (
            job.artifact_manifest.artifacts
            if job.artifact_manifest is not None
            else ()
        )
        if item.artifact_type != "residue_mapping"
    ]
    write_artifact_manifest(job.run_dir, [*existing, artifact])
    result_path = job.run_dir / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        result["residue_mapping"] = relative_path
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    metadata_path = job.run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["residue_mapping_artifact_id"] = artifact.artifact_id
    metadata["residue_numbering"] = "source_author"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return JobRecord.load(job.run_dir, task_group=job.task_group)


def relabel_structural_dynamics(
    structural_dynamics: dict[str, Any],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        row
        for row in mapping.get("residues") or []
        if isinstance(row, dict)
    ]
    exact = {
        (
            str(row.get("structure_residue_name") or "").upper(),
            int(row.get("structure_residue_number")),
            str(row.get("structure_chain") or "_"),
        ): row
        for row in rows
        if row.get("structure_residue_number") is not None
    }

    # Molecular-dynamics tools commonly replace a single protein chain with
    # either ``_``/``1`` while interaction reporters retain the original
    # chain identifier (usually ``A``).  The residue number and name are still
    # authoritative in that single-chain coordinate system, so keep a
    # chain-agnostic fallback only when it identifies exactly one mapping row.
    by_structure_residue: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        try:
            key = (
                str(row.get("structure_residue_name") or "").upper(),
                int(row.get("structure_residue_number")),
            )
        except (TypeError, ValueError):
            continue
        by_structure_residue.setdefault(key, []).append(row)

    def native_label(label: str) -> str:
        match = re.fullmatch(
            r"([A-Za-z0-9]+?)(-?\d+)([A-Za-z]?)\s*·\s*chain\s+(\S+)",
            str(label).strip(),
        )
        if match is None:
            return label
        residue_name, number_text, _insertion_code, chain = match.groups()
        number = int(number_text)
        row = exact.get((residue_name.upper(), number, chain))
        if row is None:
            candidates = by_structure_residue.get(
                (residue_name.upper(), number),
                [],
            )
            if len(candidates) == 1:
                row = candidates[0]
        if row is None and 1 <= number <= len(rows):
            positional = rows[number - 1]
            if str(positional.get("structure_residue_name") or "").upper() == (
                residue_name.upper()
            ):
                row = positional
        if row is None:
            return label
        native_insertion = str(
            row.get("native_insertion_code") or ""
        ).strip()
        return (
            f"{row.get('native_residue_name') or residue_name}"
            f"{row.get('native_residue_number')}{native_insertion}"
            f" · chain {row.get('native_chain') or '_'}"
        )

    rmsf = structural_dynamics.get("rmsf")
    if isinstance(rmsf, dict):
        rmsf["residues"] = [
            native_label(label) for label in rmsf.get("residues") or []
        ]
    contacts = structural_dynamics.get("contacts")
    if isinstance(contacts, dict):
        for row in contacts.get("residues") or []:
            if isinstance(row, dict) and row.get("residue"):
                row["residue"] = native_label(str(row["residue"]))
        distance_series = contacts.get("distance_series")
        if isinstance(distance_series, dict):
            contacts["distance_series"] = {
                native_label(str(label)): values
                for label, values in distance_series.items()
            }
    network = structural_dynamics.get("interaction_network")
    if isinstance(network, dict):
        for node in network.get("nodes") or []:
            if isinstance(node, dict) and node.get("id"):
                node["id"] = native_label(str(node["id"]))
        for edge in network.get("edges") or []:
            if isinstance(edge, dict) and edge.get("target"):
                edge["target"] = native_label(str(edge["target"]))
    structural_dynamics["residue_numbering"] = {
        "scheme": "source_author",
        "source_run_id": str(mapping.get("source_run_id") or ""),
        "source_label": str(mapping.get("source_label") or ""),
    }
    return structural_dynamics
