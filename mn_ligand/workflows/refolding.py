from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from rdkit import Chem

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    registered_tool,
    write_registered_command_record,
)
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.runtime import reference_root, runs_root


REFOLDING_TASK_GROUP = "refolding"
DEFAULT_ALPHAFOLD3_IMAGE = os.getenv("MN_AF3_IMAGE", "alphafast:latest")
DEFAULT_BOLTZ2_IMAGE = os.getenv("MN_BOLTZ2_IMAGE", "ovoex-boltz2:latest")
DEFAULT_NESSO_IMAGE = os.getenv("MN_NESSO_IMAGE", "ovolig-nesso-cu128:latest")
NESSO_ESM_MODEL = "models--facebook--esm2_t33_650M_UR50D"
DEFAULT_MSA_CPU_THREADS = max(1, int(os.getenv("MN_AF3_MSA_CPU_THREADS", "32")))
DEFAULT_MSA_CPU_RAM_GB = max(1, int(os.getenv("MN_AF3_MSA_CPU_RAM_GB", "400")))


def _reference_default(env_name: str, subdirectory: str) -> Path:
    configured = os.getenv(env_name)
    if configured:
        return Path(configured).expanduser()
    portable = reference_root() / subdirectory
    if os.getenv("MN_LIGAND_REFERENCE_DIR"):
        return portable
    legacy = Path("/mnt/db/reference_files") / subdirectory
    return portable if portable.exists() or not legacy.exists() else legacy


DEFAULT_ALPHAFOLD3_DB_DIR = _reference_default("MN_AF3_DATABASE_DIR", "alignment")
DEFAULT_ALPHAFOLD3_WEIGHTS_DIR = _reference_default("MN_AF3_WEIGHTS_DIR", "alphafold3")
DEFAULT_ALPHAFOLD3_MSA_REPOSITORY_DIR = Path(
    os.getenv(
        "MN_AF3_MSA_REPOSITORY_DIR",
        str(_reference_default("MN_AF3_MSA_REPOSITORY_DIR", "boltz_models/msa_repository")),
    )
).expanduser()


def configured_alphafold3_reference_paths() -> tuple[Path, Path, Path]:
    root = reference_root(create=False)
    database = Path(os.getenv("MN_AF3_DATABASE_DIR", str(root / "alignment"))).expanduser()
    weights = Path(os.getenv("MN_AF3_WEIGHTS_DIR", str(root / "alphafold3"))).expanduser()
    msa_repository = Path(
        os.getenv(
            "MN_AF3_MSA_REPOSITORY_DIR",
            str(root / "boltz_models" / "msa_repository"),
        )
    ).expanduser()
    return database, weights, msa_repository

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _safe_id(value: object, fallback: str = "candidate") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return text or fallback


def _clean_sequence(value: object) -> str:
    return "".join(str(value or "").split()).upper()


def msa_repository_path(sequence: str, repository_dir: Path) -> Path:
    digest = hashlib.sha256(_clean_sequence(sequence).encode("utf-8")).hexdigest()
    return Path(repository_dir) / f"{digest}.a3m"


def _a3m_records(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = ""
    sequence_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header:
                records.append((header, "".join(sequence_lines)))
            header = line
            sequence_lines = []
        elif header:
            sequence_lines.append(line)
    if header:
        records.append((header, "".join(sequence_lines)))
    return records


def _a3m_query(text: str) -> str:
    records = _a3m_records(text)
    if not records:
        return ""
    return "".join(character for character in records[0][1] if character.isupper()).replace("-", "")


def sanitize_af3_a3m(text: str, sequence: str) -> str:
    """Keep only A3M rows with the AF3-required query match-column count."""
    expected = _clean_sequence(sequence)
    valid: list[tuple[str, str]] = []
    for header, aligned in _a3m_records(text):
        match_columns = sum(1 for character in aligned if character.isupper() or character == "-")
        if match_columns == len(expected):
            valid.append((header, aligned))
    if not valid or _a3m_query("\n".join((valid[0][0], valid[0][1]))) != expected:
        valid.insert(0, (">query", expected))
    return "\n".join(f"{header}\n{aligned}" for header, aligned in valid) + "\n"


def find_cached_msa(sequence: str, repository_dir: Path) -> tuple[Path | None, str]:
    expected = _clean_sequence(sequence)
    canonical = msa_repository_path(expected, repository_dir)
    if canonical.is_file():
        text = canonical.read_text(errors="replace")
        if _a3m_query(text) == expected:
            return canonical, "sequence_hash"
    repository = Path(repository_dir)
    if repository.is_dir():
        for candidate in sorted(repository.glob("**/*.a3m")):
            try:
                if _a3m_query(candidate.read_text(errors="replace")) == expected:
                    return candidate, "sequence_scan"
            except OSError:
                continue
    return None, "missing"


def apply_cached_msas(input_dir: Path, repository_dir: Path) -> dict[str, int]:
    hits: set[str] = set()
    misses: set[str] = set()
    for input_path in sorted(Path(input_dir).glob("*.json")):
        payload = json.loads(input_path.read_text())
        changed = False
        for entity in payload.get("sequences", []):
            protein = entity.get("protein") if isinstance(entity, dict) else None
            if not isinstance(protein, dict):
                continue
            sequence = _clean_sequence(protein.get("sequence"))
            cached, _source = find_cached_msa(sequence, repository_dir)
            if cached is None:
                misses.add(sequence)
                continue
            protein["unpairedMsa"] = sanitize_af3_a3m(cached.read_text(errors="replace"), sequence)
            protein["pairedMsa"] = ""
            protein["templates"] = []
            hits.add(sequence)
            changed = True
        if changed:
            _write_json(input_path, payload)
    misses.difference_update(hits)
    return {"unique_hit_count": len(hits), "unique_miss_count": len(misses)}


def cache_generated_msas(data_dir: Path, repository_dir: Path) -> dict[str, int]:
    repository = Path(repository_dir)
    written = 0
    existing = 0
    for data_path in sorted(Path(data_dir).glob("**/*_data.json")):
        try:
            payload = json.loads(data_path.read_text())
        except (OSError, ValueError):
            continue
        for entity in payload.get("sequences", []):
            protein = entity.get("protein") if isinstance(entity, dict) else None
            if not isinstance(protein, dict):
                continue
            sequence = _clean_sequence(protein.get("sequence"))
            msa_text = str(protein.get("unpairedMsa") or "")
            if not sequence or len(_a3m_records(msa_text)) <= 1:
                continue
            target = msa_repository_path(sequence, repository)
            cached, _source = find_cached_msa(sequence, repository)
            if cached is not None:
                existing += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(sanitize_af3_a3m(msa_text, sequence))
            written += 1
    return {"written_count": written, "existing_count": existing}


def prepare_msa_dependent_commands(
    run_dir: Path,
    metadata: dict[str, Any],
    commands: Iterable[Iterable[str]],
) -> list[list[str]]:
    """Hydrate an AF3/Boltz run from the shared MSA repository after its barrier."""
    prepared = [list(command) for command in commands]
    if not metadata.get("msa_preparation_required"):
        return prepared
    repository = Path(str(metadata.get("msa_repository_dir") or "")).expanduser()
    if not repository.is_dir():
        raise RuntimeError(f"Required MSA repository is unavailable: {repository}")
    workflow = str(metadata.get("workflow") or "")
    if workflow == "alphafold3_refolding":
        input_dir = Path(run_dir) / "inputs"
        input_paths = sorted(input_dir.glob("*.json"))
        if not input_paths:
            raise RuntimeError(
                "AlphaFold input JSON is missing; refusing to launch an empty fold batch"
            )
        metrics = apply_cached_msas(input_dir, repository)
        if metrics["unique_miss_count"]:
            raise RuntimeError(
                f"MSA barrier completed but {metrics['unique_miss_count']} sequence(s) remain uncached"
            )
        data_dir = Path(run_dir) / "data"
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir()
        for input_path in input_paths:
            shutil.copy2(input_path, data_dir / input_path.name)
        return [
            command
            for command in prepared
            if not any("run_data_pipeline.py" in value for value in command)
        ]
    if workflow == "boltz2_refolding":
        sequences = [
            _clean_sequence(value)
            for value in (metadata.get("protein_sequences") or ())
            if _clean_sequence(value)
        ]
        cached_paths: list[Path] = []
        for sequence in sequences:
            cached, _source = find_cached_msa(sequence, repository)
            if cached is None:
                raise RuntimeError("MSA barrier completed but a Boltz protein MSA is missing")
            cached_paths.append(cached)
        msa_dir = Path(run_dir) / "msa"
        msa_dir.mkdir(exist_ok=True)
        staged: list[str] = []
        for index, cached in enumerate(cached_paths, start=1):
            target = msa_dir / f"protein_chain_{index}.a3m"
            target.write_text(sanitize_af3_a3m(cached.read_text(errors="replace"), sequences[index - 1]))
            staged.append(f"/work/msa/{target.name}")
        for yaml_path in (Path(run_dir) / "inputs").glob("*.yaml"):
            lines = [line for line in yaml_path.read_text().splitlines() if not line.strip().startswith("msa:")]
            rewritten: list[str] = []
            protein_index = 0
            in_protein = False
            for line in lines:
                if line.startswith("  - protein:"):
                    in_protein = True
                elif line.startswith("  - "):
                    in_protein = False
                rewritten.append(line)
                if in_protein and line.strip().startswith("sequence:"):
                    if protein_index >= len(staged):
                        raise RuntimeError("Boltz input has more protein chains than prepared MSAs")
                    rewritten.append(f"      msa: {json.dumps(staged[protein_index])}")
                    protein_index += 1
            if protein_index != len(staged):
                raise RuntimeError("Prepared MSA count does not match the Boltz protein-chain count")
            yaml_path.write_text("\n".join(rewritten) + "\n")
        return [
            [value for value in command if value != "--use_msa_server"]
            for command in prepared
        ]
    return prepared


def _canonical_af3_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles or "").strip())
    if molecule is None:
        raise ValueError(f"Invalid ligand SMILES for AlphaFold 3: {smiles}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def polymer_sequences_from_pdb(path: Path) -> list[tuple[str, str, str]]:
    """Extract protein, DNA, and RNA entities from first-model PDB records."""
    residue_codes = {
        "dna": {"DA": "A", "DC": "C", "DG": "G", "DI": "I", "DT": "T", "DU": "U"},
        "rna": {"A": "A", "C": "C", "G": "G", "I": "I", "U": "U"},
    }
    residues: dict[tuple[str, str], list[str]] = {}
    seen: set[tuple[str, str, str, str]] = set()
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        residue = line[17:20].strip().upper()
        amino_acid = AA3_TO_1.get(residue)
        if amino_acid is not None:
            kind, code = "protein", amino_acid
        elif residue in residue_codes["dna"]:
            kind, code = "dna", residue_codes["dna"][residue]
        elif residue in residue_codes["rna"]:
            kind, code = "rna", residue_codes["rna"][residue]
        else:
            continue
        chain = line[21].strip() or "A"
        key = (kind, chain, line[22:26].strip(), line[26].strip())
        if key in seen:
            continue
        seen.add(key)
        residues.setdefault((kind, chain), []).append(code)
    sequences = [
        (kind, chain, "".join(values))
        for (kind, chain), values in residues.items()
        if values
    ]
    if not sequences:
        raise ValueError(f"No polymer sequence could be extracted from PDB: {path}")
    return sequences


def protein_sequences_from_pdb(path: Path) -> list[tuple[str, str]]:
    """Backward-compatible protein-only view of the target polymers."""
    proteins = [
        (chain, sequence)
        for kind, chain, sequence in polymer_sequences_from_pdb(path)
        if kind == "protein"
    ]
    if not proteins:
        raise ValueError(f"No protein sequence could be extracted from PDB: {path}")
    return proteins


def ligand_bound_protein_sequence(
    receptor_path: Path,
    ligand_path: Path,
    *,
    contact_cutoff_angstrom: float = 6.0,
) -> dict[str, Any]:
    """Select one protein chain using the coordinate-bearing reference ligand."""
    cutoff = float(contact_cutoff_angstrom)
    if cutoff <= 0:
        raise ValueError("Ligand-contact cutoff must be positive")
    proteins = dict(protein_sequences_from_pdb(Path(receptor_path)))
    chain_coordinates: dict[str, list[tuple[float, float, float]]] = {
        chain: [] for chain in proteins
    }
    for line in Path(receptor_path).read_text(errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        chain = line[21].strip() or "A"
        residue = line[17:20].strip().upper()
        if chain not in proteins or residue not in AA3_TO_1:
            continue
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if element == "H" or (not element and line[12:16].strip().startswith("H")):
            continue
        try:
            coordinate = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        except ValueError:
            continue
        chain_coordinates[chain].append(coordinate)

    molecule = next(
        (
            item
            for item in Chem.SDMolSupplier(str(ligand_path), removeHs=False)
            if item is not None and item.GetNumConformers()
        ),
        None,
    )
    if molecule is None:
        raise ValueError("Reference ligand must be an RDKit-readable 3D SDF")
    conformer = molecule.GetConformer()
    ligand_coordinates = [
        (
            float(conformer.GetAtomPosition(index).x),
            float(conformer.GetAtomPosition(index).y),
            float(conformer.GetAtomPosition(index).z),
        )
        for index, atom in enumerate(molecule.GetAtoms())
        if atom.GetAtomicNum() > 1
    ]
    if not ligand_coordinates:
        raise ValueError("Reference ligand has no heavy atoms")

    cutoff_squared = cutoff * cutoff
    evidence = []
    for chain, coordinates in chain_coordinates.items():
        if not coordinates:
            continue
        minimum_squared = math.inf
        contacting_atoms = 0
        for x, y, z in coordinates:
            atom_minimum = min(
                (x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2
                for lx, ly, lz in ligand_coordinates
            )
            minimum_squared = min(minimum_squared, atom_minimum)
            if atom_minimum <= cutoff_squared:
                contacting_atoms += 1
        evidence.append(
            {
                "chain": chain,
                "sequence": proteins[chain],
                "contacting_protein_atoms": contacting_atoms,
                "minimum_distance_angstrom": math.sqrt(minimum_squared),
            }
        )
    if not evidence:
        raise ValueError("No coordinate-bearing protein chain is available")
    selected = min(
        evidence,
        key=lambda item: (
            -int(item["contacting_protein_atoms"]),
            float(item["minimum_distance_angstrom"]),
            str(item["chain"]),
        ),
    )
    return {
        **selected,
        "contact_cutoff_angstrom": cutoff,
        "selection_method": "maximum reference-ligand-contact atoms, then minimum distance",
        "available_protein_chains": [item["chain"] for item in evidence],
        "chain_evidence": evidence,
    }


def _polymer_entities(
    values: Iterable[tuple[str, str] | tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    entities = []
    for value in values:
        if len(value) == 2:
            chain, sequence = value
            kind = "protein"
        else:
            kind, chain, sequence = value
        kind = str(kind).strip().lower()
        if kind not in {"protein", "dna", "rna"}:
            raise ValueError(f"Unsupported polymer type: {kind}")
        entities.append((kind, str(chain), _clean_sequence(sequence)))
    return entities


def compounds_from_path(path: Path) -> list[tuple[str, str]]:
    path = Path(path)
    suffix = path.suffix.lower()
    records: list[tuple[str, str]] = []
    if suffix == ".csv":
        reader = csv.DictReader(path.open(newline=""))
        fields = {str(name).strip().lower(): str(name) for name in (reader.fieldnames or ())}
        smiles_key = next((fields[key] for key in ("smiles", "canonical_smiles", "isomeric_smiles") if key in fields), "")
        id_key = next((fields[key] for key in ("compound_id", "id", "name") if key in fields), "")
        if not smiles_key:
            raise ValueError(f"Compound CSV has no SMILES column: {path}")
        for index, row in enumerate(reader, start=1):
            smiles = str(row.get(smiles_key) or "").strip()
            if smiles:
                records.append((str(row.get(id_key) or f"compound_{index:07d}").strip(), _canonical_af3_smiles(smiles)))
    elif suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        for index, molecule in enumerate(supplier, start=1):
            if molecule is None:
                continue
            name = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
            records.append((name or f"compound_{index:07d}", Chem.MolToSmiles(molecule, isomericSmiles=True)))
    else:
        for index, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            if fields[0].lower() == "smiles":
                continue
            records.append((fields[1] if len(fields) > 1 else f"compound_{index:07d}", _canonical_af3_smiles(fields[0])))
    if not records:
        raise ValueError(f"No compounds could be read from: {path}")
    return records


def limit_compounds(
    compounds: Iterable[tuple[str, str]], maximum: int = 0
) -> list[tuple[str, str]]:
    """Return every compound when maximum is zero, otherwise apply a positive cap."""
    limit = int(maximum)
    if limit < 0:
        raise ValueError("Maximum compounds cannot be negative")
    records = list(compounds)
    return records if limit == 0 else records[:limit]


def alphafold3_input(
    name: str,
    protein_sequences: Iterable[tuple[str, str] | tuple[str, str, str]],
    ligand_smiles: str,
    *,
    model_seeds: Iterable[int] = (1,),
) -> dict[str, Any]:
    sequences: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, (kind, chain, sequence) in enumerate(
        _polymer_entities(protein_sequences)
    ):
        chain_id = (str(chain).strip() or chr(ord("A") + index))[:1]
        if chain_id in used_ids:
            chain_id = chr(ord("A") + index)
        used_ids.add(chain_id)
        sequences.append({kind: {"id": chain_id, "sequence": str(sequence)}})
    ligand_id = next((candidate for candidate in "LXYZUVW" if candidate not in used_ids), "Z")
    sequences.append({"ligand": {"id": ligand_id, "smiles": _canonical_af3_smiles(ligand_smiles)}})
    return {
        "name": _safe_id(name),
        "modelSeeds": [int(seed) for seed in model_seeds],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 1,
    }


def alphafast_readiness(
    db_dir: Path, weights_dir: Path, msa_repository_dir: Path = DEFAULT_ALPHAFOLD3_MSA_REPOSITORY_DIR
) -> dict[str, Any]:
    db_dir = Path(db_dir).expanduser().resolve()
    weights_dir = Path(weights_dir).expanduser().resolve()
    weights = sorted(weights_dir.glob("af3*.bin.zst")) if weights_dir.is_dir() else []
    return {
        "database_dir": str(db_dir),
        "weights_dir": str(weights_dir),
        "database_ready": db_dir.is_dir() and (db_dir / "mmseqs").is_dir(),
        "weights_ready": bool(weights),
        "weights_files": [path.name for path in weights],
        "msa_repository_ready": Path(msa_repository_dir).expanduser().is_dir(),
    }


def build_alphafast_commands(
    *,
    image: str,
    run_dir: Path,
    db_dir: Path,
    weights_dir: Path,
    gpu_device: str = "all",
    batch_size: int = 1,
    num_recycles: int = 10,
    run_data_pipeline: bool = True,
    use_mmseqs_gpu: bool = True,
    mmseqs_threads: int | None = None,
) -> list[list[str]]:
    run_dir = Path(run_dir).resolve()
    db_dir = Path(db_dir).expanduser().resolve()
    weights_dir = Path(weights_dir).expanduser().resolve()
    gpu = str(gpu_device).strip().lower().removeprefix("gpu ") or "0"
    gpu_ids = None if gpu in {"all", "auto", "automatic"} else (int(gpu.removeprefix("device=")),)
    tool = registered_tool("alphafold3", image=image)
    common_mounts = (
        DockerMount(run_dir, "/work"),
        DockerMount(db_dir, "/data/public_databases", read_only=True),
        DockerMount(db_dir / "mmseqs", "/data/mmseqs_databases", read_only=True),
    )
    pipeline_flags = [
        "python", "/app/alphafold/run_data_pipeline.py",
        "--input_dir=/work/inputs", "--output_dir=/work/data",
        "--db_dir=/data/public_databases", "--mmseqs_db_dir=/data/mmseqs_databases",
        "--use_mmseqs_gpu" if use_mmseqs_gpu else "--nouse_mmseqs_gpu",
        f"--batch_size={max(1, int(batch_size))}",
    ]
    if mmseqs_threads is not None:
        pipeline_flags.append(f"--mmseqs_n_threads={max(1, int(mmseqs_threads))}")
    pipeline = build_docker_command(
        DockerRunSpec(
            tool=tool,
            command=tuple(pipeline_flags),
            mounts=common_mounts,
            gpu_devices=gpu_ids if use_mmseqs_gpu else None,
            workdir="/app/alphafold",
            shm_size="32g",
            use_host_user=False,
        )
    )
    inference = build_docker_command(
        DockerRunSpec(
            tool=tool,
            command=(
                "python", "/app/alphafold/run_alphafold.py",
                "--input_dir=/work/data", "--model_dir=/data/models", "--norun_data_pipeline",
                "--output_dir=/work/output", "--force_output_dir",
                f"--num_recycles={max(1, int(num_recycles))}",
            ),
            mounts=(*common_mounts, DockerMount(weights_dir, "/data/models", read_only=True)),
            gpu_devices=gpu_ids,
            workdir="/app/alphafold",
            shm_size="32g",
            use_host_user=False,
        )
    )
    if not use_mmseqs_gpu and "--gpus" in pipeline:
        gpu_option = pipeline.index("--gpus")
        del pipeline[gpu_option : gpu_option + 2]
    return [pipeline, inference] if run_data_pipeline else [inference]


def portable_command_record(
    commands: Iterable[Iterable[str]],
    *,
    run_dir: Path,
    db_dir: Path,
    weights_dir: Path,
) -> list[list[str]]:
    replacements = {
        str(Path(run_dir).resolve()): "${RUN_DIR}",
        str(Path(db_dir).expanduser().resolve()): "${REFERENCE_DIR}/alignment",
        str(Path(weights_dir).expanduser().resolve()): "${REFERENCE_DIR}/alphafold3",
    }
    result: list[list[str]] = []
    for command in commands:
        portable: list[str] = []
        for argument in command:
            value = str(argument)
            for source, replacement in replacements.items():
                value = value.replace(source, replacement)
            portable.append(value)
        result.append(portable)
    return result


def finalize_alphafold3_msa_job(run_dir: Path, *, returncode: int) -> JobRecord:
    """Publish a sequence-addressed MSA prepared by the AlphaFast pipeline."""
    run_dir = Path(run_dir).resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    repository = Path(str(metadata.get("msa_repository_dir") or "")).expanduser()
    sequences = list(
        dict.fromkeys(
            _clean_sequence(value)
            for value in (
                metadata.get("protein_sequences")
                or [metadata.get("protein_sequence")]
            )
            if _clean_sequence(value)
        )
    )
    cache_metrics: dict[str, Any] = {}
    if returncode == 0:
        cache_metrics = cache_generated_msas(run_dir / "data", repository)
    cached_rows = [
        (sequence, *find_cached_msa(sequence, repository))
        for sequence in sequences
    ]
    success = returncode == 0 and bool(cached_rows) and all(
        cached is not None for _sequence, cached, _source in cached_rows
    )
    artifacts: list[ArtifactRef] = []
    if success:
        for index, (sequence, cached, _source) in enumerate(cached_rows, start=1):
            assert cached is not None
            sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
            published = run_dir / "artifacts" / f"prepared_msa_{index:03d}_{sequence_hash[:12]}.a3m"
            published.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, published)
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir,
                    published,
                    "protein_msa",
                    role="sequence_hash",
                    metadata={"sequence_sha256": sequence_hash},
                )
            )
    write_artifact_manifest(run_dir, artifacts)
    error = "" if success else (
        (run_dir / "stderr.log").read_text(errors="replace")[-4000:]
        if (run_dir / "stderr.log").exists()
        else "AlphaFast MSA preparation produced no reusable alignment"
    )
    fallback_run_id = ""
    if (
        not success
        and bool(metadata.get("use_gpu"))
        and "out of memory" in error.lower()
        and not metadata.get("cpu_fallback_run_id")
    ):
        input_payload = json.loads((run_dir / "input.json").read_text())
        replacement = queue_alphafold3_msa_job(
            protein_sequences=sequences,
            target_artifact=ArtifactRef.from_dict(input_payload["target"]),
            msa_repository_dir=repository,
            batch_size=max(1, len(sequences)),
            launch_campaign_id=str(metadata.get("launch_campaign_id") or ""),
            launch_campaign_label=str(metadata.get("launch_campaign_label") or ""),
            campaign_purpose=str(metadata.get("campaign_purpose") or ""),
            use_gpu=False,
        )
        fallback_run_id = replacement.run_id
        old_run_id = str(metadata.get("run_id") or run_dir.name)
        for dependent_path in runs_root().glob("**/metadata.json"):
            dependent = json.loads(dependent_path.read_text())
            dependencies = [
                str(value) for value in (dependent.get("depends_on_run_ids") or ())
            ]
            if old_run_id not in dependencies:
                continue
            dependent["depends_on_run_ids"] = [
                fallback_run_id if value == old_run_id else value
                for value in dependencies
            ]
            dependent.setdefault("msa_cpu_fallback_run_ids", []).append(
                fallback_run_id
            )
            if dependent.get("status") == "blocked":
                dependent["status"] = "queued"
                dependent.pop("completed_at", None)
                dependent.pop("blocked_by_run_ids", None)
                dependent.pop("error", None)
            _write_json(dependent_path, dependent)
        error = (
            error
            + "\nGPU MMseqs exhausted VRAM; queued automatic CPU/RAM MSA fallback "
            + fallback_run_id
        )
    _write_json(
        run_dir / "result.json",
        {
            "success": success,
            "returncode": returncode,
            "msa_cache": cache_metrics,
            "cache_sources": {
                hashlib.sha256(sequence.encode()).hexdigest(): source
                for sequence, _cached, source in cached_rows
            },
            "sequence_count": len(sequences),
            "error": error,
        },
    )
    now = _utc_now_iso()
    metadata.update(
        {
            "status": "completed" if success else "failed",
            "updated_at": now,
            "completed_at": now,
        }
    )
    if error:
        metadata["error"] = error
    else:
        metadata.pop("error", None)
    if fallback_run_id:
        metadata["cpu_fallback_run_id"] = fallback_run_id
    _write_json(metadata_path, metadata)
    return JobRecord.load(run_dir, task_group=REFOLDING_TASK_GROUP)


def queue_alphafold3_msa_job(
    *,
    protein_sequence: str = "",
    protein_sequences: Iterable[str] = (),
    target_artifact: ArtifactRef,
    image: str = DEFAULT_ALPHAFOLD3_IMAGE,
    db_dir: Path | None = None,
    weights_dir: Path | None = None,
    msa_repository_dir: Path | None = None,
    batch_size: int = 1,
    launch_campaign_id: str = "",
    launch_campaign_label: str = "",
    campaign_purpose: str = "",
    use_gpu: bool = True,
    gpu_device: str = "all",
    cpu_threads: int = DEFAULT_MSA_CPU_THREADS,
) -> JobRecord:
    """Queue one reusable MSA stage on any or one explicitly selected GPU."""
    sequences = list(
        dict.fromkeys(
            _clean_sequence(value)
            for value in (*tuple(protein_sequences), protein_sequence)
            if _clean_sequence(value)
        )
    )
    if not sequences:
        raise ValueError("A protein sequence is required for MSA preparation")
    configured_db, configured_weights, configured_msa = configured_alphafold3_reference_paths()
    db_dir = Path(db_dir) if db_dir is not None else configured_db
    weights_dir = Path(weights_dir) if weights_dir is not None else configured_weights
    repository = Path(msa_repository_dir) if msa_repository_dir is not None else configured_msa
    readiness = alphafast_readiness(db_dir, weights_dir, repository)
    if not readiness["database_ready"]:
        raise ValueError(f"AlphaFast database directory is incomplete: {readiness['database_dir']}")
    cached_rows = [find_cached_msa(sequence, repository) for sequence in sequences]
    run_id = str(uuid4())
    run_dir = runs_root() / REFOLDING_TASK_GROUP / run_id
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)
    for sequence in sequences:
        name = f"msa-{hashlib.sha256(sequence.encode()).hexdigest()[:12]}"
        _write_json(
            input_dir / f"{name}.json",
            {
                "name": name,
                "modelSeeds": [1],
                "sequences": [{"protein": {"id": "A", "sequence": sequence}}],
                "dialect": "alphafold3",
                "version": 1,
            },
        )
    if all(cached is not None for cached, _source in cached_rows):
        apply_cached_msas(input_dir, repository)
        data_dir = run_dir / "data"
        data_dir.mkdir()
        for input_path in input_dir.glob("*.json"):
            shutil.copy2(input_path, data_dir / input_path.name)
        commands = [["/usr/bin/true"]]
    else:
        commands = build_alphafast_commands(
            image=image,
            run_dir=run_dir,
            db_dir=db_dir,
            weights_dir=weights_dir,
            gpu_device=gpu_device,
            batch_size=max(int(batch_size), len(sequences)),
            run_data_pipeline=True,
            use_mmseqs_gpu=use_gpu,
            mmseqs_threads=None if use_gpu else cpu_threads,
        )[:1]
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "protein_msa_preparation",
        "workflow": "alphafold3_msa",
        "operation": "msa_preparation",
        "status": "queued",
        "tool": "AlphaFast MSA",
        "engine": "AlphaFast MSA",
        "docker_image": image,
        "parent_run_id": target_artifact.run_id,
        "use_gpu": bool(use_gpu),
        "gpu_device": str(gpu_device) if use_gpu else "cpu",
        "protein_sequence": sequences[0],
        "protein_sequences": sequences,
        "protein_sequence_count": len(sequences),
        "msa_repository_dir": str(repository.expanduser().resolve()),
        "launch_campaign_id": str(launch_campaign_id),
        "launch_campaign_label": str(launch_campaign_label),
        "campaign_purpose": str(campaign_purpose),
        "campaign_phase": "msa_preparation" if use_gpu else "msa_cpu_fallback",
        "queue_priority": 100,
        "created_at": now,
        "updated_at": now,
        "queued_at": now,
        "queued_command": commands[0],
        "queued_commands": commands,
        "gpu_queued": bool(use_gpu),
        "resources": {
            **registered_tool("alphafold3", image=image).resources.to_dict(),
            **(
                {
                    "gpu": True,
                    "min_vram_gb": 22,
                    **(
                        {}
                        if str(gpu_device).strip().lower()
                        in {"all", "auto", "automatic"}
                        else {
                            "gpu_ids": [
                                int(
                                    str(gpu_device)
                                    .strip()
                                    .lower()
                                    .removeprefix("device=")
                                    .removeprefix("gpu ")
                                )
                            ]
                        }
                    ),
                }
                if use_gpu
                else {
                    "gpu": False,
                    "gpu_ids": [],
                    "min_vram_gb": 0,
                    "exclusive_gpu": False,
                    "cpu_threads": max(1, int(cpu_threads)),
                    "ram_gb": DEFAULT_MSA_CPU_RAM_GB,
                }
            ),
        },
        "worker_finalizer": "alphafold3_msa",
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "target": target_artifact.to_dict(),
            "parameters": {
                "protein_sequence_sha256": hashlib.sha256(sequences[0].encode()).hexdigest(),
                "protein_sequence_sha256s": [
                    hashlib.sha256(sequence.encode()).hexdigest()
                    for sequence in sequences
                ],
                "msa_repository_dir": str(repository.expanduser().resolve()),
                "gpu_device": str(gpu_device) if use_gpu else "cpu",
                "use_mmseqs_gpu": bool(use_gpu),
                "mmseqs_cpu_threads": (
                    None if use_gpu else max(1, int(cpu_threads))
                ),
                "launch_campaign_id": str(launch_campaign_id),
                "launch_campaign_label": str(launch_campaign_label),
            },
        },
    )
    write_artifact_manifest(run_dir, [])
    return JobRecord.load(run_dir, task_group=REFOLDING_TASK_GROUP)


def _collect_alphafast_outputs(run_dir: Path) -> tuple[list[ArtifactRef], list[dict[str, Any]]]:
    artifacts: list[ArtifactRef] = []
    metrics: list[dict[str, Any]] = []
    for candidate_dir in sorted(
        path for path in (run_dir / "output").iterdir() if path.is_dir()
    ):
        sampled_summaries = sorted(
            candidate_dir.glob("seed-*_sample-*/*_summary_confidences.json")
        )
        summary_paths = sampled_summaries or sorted(
            candidate_dir.glob("*_summary_confidences.json")
        )
        for summary_path in summary_paths:
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, ValueError):
                summary = {}
            candidate_id = candidate_dir.name
            prediction_id = (
                summary_path.parent.name
                if summary_path.parent != candidate_dir
                else candidate_id
            )
            role = (
                f"{candidate_id}:{prediction_id}"
                if prediction_id != candidate_id
                else candidate_id
            )
            row = {
                "candidate_id": candidate_id,
                "prediction_id": prediction_id,
            }
            seed_sample = re.fullmatch(
                r"seed-(\d+)_sample-(\d+)", prediction_id
            )
            if seed_sample:
                row["model_seed"] = int(seed_sample.group(1))
                row["sample"] = int(seed_sample.group(2))
            for key in (
                "ranking_score",
                "iptm",
                "ptm",
                "fraction_disordered",
                "has_clash",
            ):
                if key in summary:
                    row[key] = summary[key]
            metrics.append(row)
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir,
                    summary_path,
                    "prediction_confidence",
                    role=role,
                )
            )
            structures = sorted(summary_path.parent.glob("*.cif")) + sorted(
                summary_path.parent.glob("*.pdb")
            )
            if structures:
                artifacts.append(
                    ArtifactRef.from_path(
                        run_dir,
                        structures[0],
                        "predicted_complex",
                        role=role,
                        metadata={
                            key: row[key]
                            for key in (
                                "ranking_score",
                                "iptm",
                                "ptm",
                                "model_seed",
                                "sample",
                            )
                            if key in row
                        },
                    )
                )
    if metrics:
        table = run_dir / "artifacts" / "alphafold3_metrics.csv"
        table.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted({key for row in metrics for key in row})
        with table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(metrics)
        artifacts.append(ArtifactRef.from_path(run_dir, table, "prediction_metrics", role="summary"))
    return artifacts, metrics


def finalize_alphafold3_refolding_job(run_dir: Path, *, returncode: int) -> JobRecord:
    """Validate AlphaFold 3 native output and publish typed prediction artifacts."""
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    input_payload = json.loads((run_dir / "input.json").read_text())
    parameters = dict(input_payload.get("parameters") or {})
    msa_metrics = dict(parameters.get("msa_cache") or {})
    repository_value = str(metadata.get("msa_repository_dir") or "")
    if returncode == 0 and repository_value and (run_dir / "data").is_dir():
        try:
            msa_metrics.update(cache_generated_msas(run_dir / "data", Path(repository_value)))
        except Exception as exc:
            msa_metrics["cache_error"] = str(exc)
    artifacts, metrics = _collect_alphafast_outputs(run_dir) if returncode == 0 else ([], [])
    success = returncode == 0 and any(
        artifact.artifact_type == "predicted_complex" for artifact in artifacts
    )
    stderr_path = run_dir / "stderr.log"
    stderr_path.touch(exist_ok=True)
    error = "" if success else (
        stderr_path.read_text(errors="replace")[-4000:]
        or "AlphaFold 3 produced no readable predicted complexes"
    )
    write_artifact_manifest(run_dir, artifacts)
    _write_json(
        run_dir / "result.json",
        {
            "success": success,
            "returncode": returncode,
            "prediction_count": len(metrics),
            "msa_cache": msa_metrics,
            "error": error,
        },
    )
    completed = _utc_now_iso()
    metadata.update(
        {
            "status": "completed" if success else "failed",
            "updated_at": completed,
            "completed_at": completed,
        }
    )
    if error:
        metadata["error"] = error
    _write_json(metadata_path, metadata)
    return JobRecord.load(run_dir, task_group=REFOLDING_TASK_GROUP)


def run_alphafold3_refolding_job(
    *,
    target_path: Path,
    target_artifact: ArtifactRef,
    compound_paths: Iterable[Path],
    compound_artifacts: Iterable[ArtifactRef],
    reference_ligand_artifact: ArtifactRef | None = None,
    image: str = DEFAULT_ALPHAFOLD3_IMAGE,
    db_dir: Path | None = None,
    weights_dir: Path | None = None,
    msa_repository_dir: Path | None = None,
    gpu_device: str = "all",
    max_compounds: int = 0,
    batch_size: int = 1,
    num_recycles: int = 10,
    model_seed_count: int = 1,
    model_seed_start: int = 1,
    protein_sequences: Iterable[tuple[str, str]] | None = None,
    launch_context: str = "",
    launch_campaign_id: str = "",
    launch_campaign_label: str = "",
    campaign_purpose: str = "",
    enqueue_only: bool = False,
) -> JobRecord:
    compound_paths = tuple(Path(path) for path in compound_paths)
    compound_artifacts = tuple(compound_artifacts)
    configured_db, configured_weights, configured_msa = configured_alphafold3_reference_paths()
    db_dir = Path(db_dir) if db_dir is not None else configured_db
    weights_dir = Path(weights_dir) if weights_dir is not None else configured_weights
    msa_repository_dir = Path(msa_repository_dir) if msa_repository_dir is not None else configured_msa
    readiness = alphafast_readiness(db_dir, weights_dir, msa_repository_dir)
    if not readiness["database_ready"]:
        raise ValueError(f"AlphaFast database directory is incomplete: {readiness['database_dir']}")
    if not readiness["weights_ready"]:
        raise ValueError(f"AlphaFast weights are missing: {readiness['weights_dir']}")
    polymers = (
        _polymer_entities(protein_sequences)
        if protein_sequences is not None
        else polymer_sequences_from_pdb(Path(target_path))
    )
    if not polymers or any(not sequence for _, _, sequence in polymers):
        raise ValueError("At least one protein, DNA, or RNA sequence is required")
    compounds: list[tuple[str, str]] = []
    for path in compound_paths:
        compounds.extend(compounds_from_path(path))
    compounds = limit_compounds(compounds, max_compounds)

    run_id = str(uuid4())
    run_dir = runs_root() / REFOLDING_TASK_GROUP / run_id
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)
    model_seed_count = max(1, int(model_seed_count))
    model_seed_start = int(model_seed_start)
    if model_seed_start < 1 or model_seed_start + model_seed_count - 1 >= 2_147_483_647:
        raise ValueError("AlphaFold 3 model-seed range must contain positive 32-bit integers")
    seeds = range(model_seed_start, model_seed_start + model_seed_count)
    used_names: set[str] = set()
    for index, (compound_id, smiles) in enumerate(compounds, start=1):
        name = _safe_id(compound_id, f"compound_{index:07d}")
        if name in used_names:
            name = f"{name}-{index}"
        used_names.add(name)
        _write_json(
            input_dir / f"{name}.json",
            alphafold3_input(name, polymers, smiles, model_seeds=seeds),
        )

    msa_metrics = apply_cached_msas(input_dir, msa_repository_dir)
    run_data_pipeline = msa_metrics["unique_miss_count"] > 0
    if not run_data_pipeline:
        data_dir = run_dir / "data"
        data_dir.mkdir()
        for input_path in input_dir.glob("*.json"):
            shutil.copy2(input_path, data_dir / input_path.name)

    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION, "run_id": run_id, "job_code": short_job_code(run_id),
        "job_type": "complex_refolding", "workflow": "alphafold3_refolding",
        "operation": "refolding", "status": "queued" if enqueue_only else "running",
        "tool": "AlphaFold 3", "engine": "AlphaFold 3", "docker_image": image,
        "parent_run_id": target_artifact.run_id, "use_gpu": True,
        "gpu_device": str(gpu_device), "compound_count": len(compounds),
        "msa_repository_dir": str(msa_repository_dir.expanduser().resolve()),
        "protein_input_mode": "sequence" if protein_sequences is not None else "prepared_target",
        "polymer_entity_counts": {
            kind: sum(1 for entity_kind, _, _ in polymers if entity_kind == kind)
            for kind in ("protein", "dna", "rna")
        },
        "launch_context": str(launch_context),
        "launch_campaign_id": str(launch_campaign_id),
        "launch_campaign_label": str(launch_campaign_label),
        "campaign_purpose": str(campaign_purpose),
        "reference_ligand_run_id": (
            reference_ligand_artifact.run_id if reference_ligand_artifact else ""
        ),
        "created_at": now, "updated_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "target": target_artifact.to_dict(),
            "compound_sets": [artifact.to_dict() for artifact in compound_artifacts],
            "reference_ligand_artifact": (
                reference_ligand_artifact.to_dict()
                if reference_ligand_artifact is not None
                else None
            ),
            "parameters": {
                "max_compounds": len(compounds), "batch_size": int(batch_size),
                "protein_input_mode": metadata["protein_input_mode"],
                "num_recycles": int(num_recycles), "model_seed_count": model_seed_count,
                "model_seed_start": model_seed_start,
                "gpu_device": str(gpu_device), "image": image,
                "launch_campaign_id": str(launch_campaign_id),
                "launch_campaign_label": str(launch_campaign_label),
                "campaign_purpose": str(campaign_purpose),
                "reference_layout": {
                    "database": "alignment",
                    "weights": "alphafold3",
                    "msa_repository": "boltz_models/msa_repository",
                },
                "msa_policy": "repository_then_alphafast_mmseqs_gpu",
                "msa_cache": msa_metrics,
                "readiness": {
                    "database_ready": readiness["database_ready"],
                    "weights_ready": readiness["weights_ready"],
                    "msa_repository_ready": readiness["msa_repository_ready"],
                },
            },
        },
    )
    commands = build_alphafast_commands(
        image=image, run_dir=run_dir, db_dir=db_dir, weights_dir=weights_dir,
        gpu_device=gpu_device, batch_size=batch_size, num_recycles=num_recycles,
        run_data_pipeline=run_data_pipeline,
    )
    manifest = registered_tool("alphafold3", image=image)
    selected_gpu_ids = (
        []
        if str(gpu_device).strip().lower() in {"all", "auto", "automatic"}
        else [int(str(gpu_device).strip().lower().removeprefix("device=").removeprefix("gpu "))]
    )
    _write_json(
        run_dir / "command.json",
        {
            "schema_version": 1,
            "mode": "docker",
            "tool_id": manifest.tool_id,
            "tool_version": manifest.version,
            "integration_status": manifest.integration_status,
            "image": manifest.image,
            "image_digest": manifest.image_digest,
            "resources": manifest.resources.to_dict(),
            "selected_gpu_ids": selected_gpu_ids,
            "path_base": {"RUN_DIR": ".", "REFERENCE_DIR": "configured_reference_root"},
            "commands": portable_command_record(
                commands, run_dir=run_dir, db_dir=db_dir, weights_dir=weights_dir
            ),
        },
    )
    if enqueue_only:
        resources = manifest.resources.to_dict()
        if selected_gpu_ids:
            resources["gpu_ids"] = selected_gpu_ids
        metadata.update(
            {
                "status": "queued",
                "queued_at": now,
                "queued_command": commands[0],
                "queued_commands": commands,
                "gpu_queued": True,
                "resources": resources,
                "worker_finalizer": "alphafold3_refolding",
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        write_artifact_manifest(run_dir, [])
        return JobRecord.load(run_dir, task_group=REFOLDING_TASK_GROUP)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    returncode = 0
    for index, command in enumerate(commands):
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        stdout_parts.append(process.stdout or "")
        stderr_parts.append(process.stderr or "")
        returncode = int(process.returncode)
        if returncode:
            break
    (run_dir / "stdout.log").write_text("\n".join(stdout_parts))
    (run_dir / "stderr.log").write_text("\n".join(stderr_parts))
    return finalize_alphafold3_refolding_job(run_dir, returncode=returncode)


def queue_alphafold3_refolding_job(**parameters: Any) -> JobRecord:
    """Create a queued AlphaFold 3 refolding campaign."""
    return run_alphafold3_refolding_job(enqueue_only=True, **parameters)


def configured_boltz2_cache_dir() -> Path:
    return Path(
        os.getenv("MN_BOLTZ_CACHE_DIR", str(reference_root(create=False) / "boltz_models"))
    ).expanduser()


def boltz2_readiness(cache_dir: Path | None = None) -> dict[str, Any]:
    cache = Path(cache_dir) if cache_dir is not None else configured_boltz2_cache_dir()
    cache = cache.expanduser().resolve()
    structure_checkpoint = cache / "boltz2_conf.ckpt"
    affinity_checkpoint = cache / "boltz2_aff.ckpt"
    return {
        "cache_dir": str(cache),
        "structure_checkpoint": str(structure_checkpoint),
        "affinity_checkpoint": str(affinity_checkpoint),
        "structure_ready": structure_checkpoint.is_file(),
        "affinity_ready": affinity_checkpoint.is_file(),
        "ready": structure_checkpoint.is_file() and affinity_checkpoint.is_file(),
    }


def boltz2_input_yaml(
    protein_sequences: Iterable[tuple[str, str] | tuple[str, str, str]],
    ligand_smiles: str,
    *,
    enable_affinity: bool = True,
    protein_msa_paths: Iterable[str] = (),
) -> str:
    lines = ["version: 1", "sequences:"]
    msa_paths = tuple(str(path) for path in protein_msa_paths)
    used_ids: set[str] = set()
    protein_index = 0
    for index, (kind, chain, sequence) in enumerate(
        _polymer_entities(protein_sequences)
    ):
        chain_id = (str(chain).strip() or chr(ord("A") + index))[:1]
        if chain_id in used_ids:
            chain_id = chr(ord("A") + index)
        used_ids.add(chain_id)
        lines.extend(
            [
                f"  - {kind}:",
                f"      id: {json.dumps(chain_id)}",
                f"      sequence: {_clean_sequence(sequence)}",
            ]
        )
        if kind == "protein":
            if protein_index < len(msa_paths) and msa_paths[protein_index]:
                lines.append(f"      msa: {json.dumps(msa_paths[protein_index])}")
            protein_index += 1
    ligand_id = next((candidate for candidate in "LXYZUVW" if candidate not in used_ids), "Z")
    lines.extend(
        [
            "  - ligand:",
            f"      id: {json.dumps(ligand_id)}",
            f"      smiles: {json.dumps(_canonical_af3_smiles(ligand_smiles))}",
        ]
    )
    if enable_affinity:
        lines.extend(
            [
                "properties:",
                "  - affinity:",
                f"      binder: {json.dumps(ligand_id)}",
            ]
        )
    return "\n".join(lines) + "\n"


def build_boltz2_command(
    *,
    image: str,
    run_dir: Path,
    cache_dir: Path,
    gpu_device: str = "all",
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 5,
    use_msa_server: bool = True,
    use_potentials: bool = True,
    input_dir: str = "/work/inputs",
    output_dir: str = "/work/output",
    seed: int | None = None,
) -> list[str]:
    selected = str(gpu_device).strip().lower().removeprefix("gpu ") or "0"
    gpu_ids = None if selected in {"all", "auto", "automatic"} else (
        int(selected.removeprefix("device=")),
    )
    native = [
        "predict",
        str(input_dir),
        "--out_dir",
        str(output_dir),
        "--model",
        "boltz2",
        "--accelerator",
        "gpu",
        "--sampling_steps",
        str(max(1, int(sampling_steps))),
        "--recycling_steps",
        str(max(1, int(recycling_steps))),
        "--diffusion_samples",
        str(max(1, int(diffusion_samples))),
        "--sampling_steps_affinity",
        str(max(1, int(sampling_steps_affinity))),
        "--diffusion_samples_affinity",
        str(max(1, int(diffusion_samples_affinity))),
        "--override",
    ]
    if seed is not None:
        native.extend(("--seed", str(int(seed))))
    if use_msa_server:
        native.append("--use_msa_server")
    if use_potentials:
        native.append("--use_potentials")
    return build_docker_command(
        DockerRunSpec(
            tool=registered_tool("boltz2", image=image),
            command=tuple(native),
            mounts=(
                DockerMount(Path(run_dir), "/work"),
                DockerMount(Path(cache_dir), "/cache"),
            ),
            environment={"BOLTZ_CACHE": "/cache"},
            gpu_devices=gpu_ids,
            shm_size="8g",
            use_host_user=False,
        )
    )


def _output_replicate(path: Path, output_root: Path) -> int:
    try:
        parts = path.relative_to(output_root).parts
    except ValueError:
        return 1
    for part in parts:
        match = re.fullmatch(r"replicate_(\d+)", part)
        if match:
            return int(match.group(1))
    return 1


def _write_metric_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _boltz2_replicate_summaries(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for compound_id in sorted({str(row.get("compound_id") or "") for row in metrics}):
        compound_rows = [
            row for row in metrics if str(row.get("compound_id") or "") == compound_id
        ]
        selected: list[dict[str, Any]] = []
        for replicate in sorted({int(row.get("replicate") or 1) for row in compound_rows}):
            candidates = [
                row for row in compound_rows if int(row.get("replicate") or 1) == replicate
            ]
            model_zero = [
                row
                for row in candidates
                if str(row.get("model_id") or "").endswith("_model_0")
            ]
            if model_zero:
                candidates = model_zero
            selected.append(
                max(
                    candidates,
                    key=lambda row: float(row.get("confidence_score") or -math.inf),
                )
            )
        affinity = [
            float(row["affinity_pred_value"])
            for row in selected if row.get("affinity_pred_value") is not None
        ]
        probability = [
            float(row["affinity_probability_binary"])
            for row in selected if row.get("affinity_probability_binary") is not None
        ]
        confidence = [
            float(row["confidence_score"])
            for row in selected if row.get("confidence_score") is not None
        ]
        if affinity:
            median = statistics.median(affinity)
            representative = min(
                (row for row in selected if row.get("affinity_pred_value") is not None),
                key=lambda row: (
                    round(abs(float(row["affinity_pred_value"]) - median), 12),
                    int(row.get("replicate") or 1),
                ),
            )
        else:
            representative = max(
                selected,
                key=lambda row: float(row.get("confidence_score") or -math.inf),
            )
        summaries.append(
            {
                "compound_id": compound_id,
                "replicate_count": len(selected),
                "mean_affinity_pred_value": statistics.mean(affinity) if affinity else None,
                "sample_sd_affinity_pred_value": (
                    statistics.stdev(affinity) if len(affinity) > 1 else None
                ),
                "mean_binder_probability": (
                    statistics.mean(probability) if probability else None
                ),
                "sample_sd_binder_probability": (
                    statistics.stdev(probability) if len(probability) > 1 else None
                ),
                "mean_confidence_score": (
                    statistics.mean(confidence) if confidence else None
                ),
                "sample_sd_confidence_score": (
                    statistics.stdev(confidence) if len(confidence) > 1 else None
                ),
                "representative_replicate": int(representative.get("replicate") or 1),
                "representative_model_id": str(representative.get("model_id") or ""),
                "representative_structure_file": str(
                    representative.get("structure_file") or ""
                ),
            }
        )
    return summaries


def _collect_boltz2_outputs(run_dir: Path) -> tuple[list[ArtifactRef], list[dict[str, Any]]]:
    artifacts: list[ArtifactRef] = []
    metrics: list[dict[str, Any]] = []
    recorded_affinity: set[Path] = set()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    seed_start = int(metadata.get("seed_start") or 1001)
    output_root = run_dir / "output"
    confidence_paths = sorted(
        output_root.glob("**/predictions/*/confidence_*_model_*.json")
    )
    for confidence_path in confidence_paths:
        try:
            confidence = json.loads(confidence_path.read_text())
        except (OSError, ValueError):
            confidence = {}
        candidate_dir = confidence_path.parent
        model_id = confidence_path.stem.removeprefix("confidence_")
        compound_id = candidate_dir.name
        replicate = _output_replicate(confidence_path, output_root)
        role = f"{compound_id}:replicate_{replicate:03d}:{model_id}"
        row: dict[str, Any] = {
            "candidate_id": compound_id,
            "compound_id": compound_id,
            "model_id": model_id,
            "replicate": replicate,
            "seed": seed_start + replicate - 1,
        }
        for key in (
            "confidence_score",
            "ptm",
            "iptm",
            "ligand_iptm",
            "complex_plddt",
            "complex_iplddt",
        ):
            if key in confidence:
                row[key] = confidence[key]
        structures = sorted(candidate_dir.glob(f"{model_id}.cif")) + sorted(
            candidate_dir.glob(f"{model_id}.pdb")
        )
        if not structures:
            structures = sorted(candidate_dir.glob("*_model_*.cif")) + sorted(
                candidate_dir.glob("*_model_*.pdb")
            )
        if structures:
            row["structure_file"] = structures[0].relative_to(run_dir).as_posix()
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir,
                    structures[0],
                    "predicted_complex",
                    role=role,
                    metadata={
                        key: row[key]
                        for key in ("confidence_score", "iptm", "ptm")
                        if key in row
                    },
                )
            )
        artifacts.append(
            ArtifactRef.from_path(
                run_dir, confidence_path, "prediction_confidence", role=role
            )
        )
        affinity_paths = sorted(candidate_dir.glob("affinity_*.json"))
        if affinity_paths:
            affinity_path = affinity_paths[0]
            try:
                affinity = json.loads(affinity_path.read_text())
            except (OSError, ValueError):
                affinity = {}
            for key in ("affinity_pred_value", "affinity_probability_binary"):
                if key in affinity:
                    row[key] = affinity[key]
            if affinity_path not in recorded_affinity:
                artifacts.append(
                    ArtifactRef.from_path(
                        run_dir, affinity_path, "affinity_result",
                        role=f"{compound_id}:replicate_{replicate:03d}"
                    )
                )
                recorded_affinity.add(affinity_path)
        metrics.append(row)
    if metrics:
        table = run_dir / "artifacts" / "boltz2_metrics.csv"
        _write_metric_table(table, metrics)
        artifacts.append(
            ArtifactRef.from_path(run_dir, table, "prediction_metrics", role="summary")
        )
    return artifacts, metrics


def finalize_boltz2_refolding_job(run_dir: Path, *, returncode: int) -> JobRecord:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    # Boltz can finish most inputs and still return non-zero because one input
    # failed. Always collect readable outputs so a single compound does not
    # erase the successful portion of a campaign.
    artifacts, metrics = _collect_boltz2_outputs(run_dir)
    compound_count = int(metadata.get("compound_count") or 0)
    replicates = int(metadata.get("replicates") or 1)
    completed_pairs = {
        (str(row.get("compound_id") or ""), int(row.get("replicate") or 1))
        for row in metrics
        if str(row.get("structure_file") or "").strip()
    }
    expected_pairs = compound_count * replicates
    expected_compounds = sorted(path.stem for path in (run_dir / "inputs").glob("*.yaml"))
    missing_pairs = [
        (compound_id, replicate)
        for compound_id in expected_compounds
        for replicate in range(1, replicates + 1)
        if (compound_id, replicate) not in completed_pairs
    ]
    failures_path = run_dir / "artifacts" / "boltz2_compound_failures.tsv"
    if missing_pairs:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with failures_path.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["compound_id", "replicate", "stage", "reason"])
            for compound_id, replicate in missing_pairs:
                writer.writerow(
                    [
                        compound_id,
                        replicate,
                        "prediction",
                        "Boltz-2 emitted no readable predicted complex",
                    ]
                )
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                failures_path,
                "compound_exclusions",
                role="failed_boltz2_compounds",
                metadata={"failed_compound_replicates": len(missing_pairs)},
            )
        )
    success = bool(completed_pairs) and any(
        artifact.artifact_type == "predicted_complex" for artifact in artifacts
    )
    partial_success = success and (
        returncode != 0 or len(completed_pairs) != expected_pairs
    )
    summaries = _boltz2_replicate_summaries(metrics) if replicates > 1 else []
    if summaries:
        summary_path = run_dir / "artifacts" / "boltz2_replicate_summary.csv"
        _write_metric_table(summary_path, summaries)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                summary_path,
                "prediction_metrics",
                role="replicate_summary",
                metadata={"representative": "affinity closest to replicate median"},
            )
        )
    stderr_path = run_dir / "stderr.log"
    stderr_path.touch(exist_ok=True)
    error = "" if success else (
        stderr_path.read_text(errors="replace")[-4000:]
        or (
            "Boltz-2 produced no readable predicted complexes; "
            f"{len(completed_pairs)}/{expected_pairs} expected "
            "compound-replicate predictions were complete"
        )
    )
    write_artifact_manifest(run_dir, artifacts)
    _write_json(
        run_dir / "result.json",
        {
            "success": success,
            "returncode": returncode,
            "prediction_count": len(metrics),
            "compound_count": compound_count,
            "replicates": replicates,
            "completed_replicate_pairs": len(completed_pairs),
            "completed_compounds": len({pair[0] for pair in completed_pairs}),
            "failed_compounds": len({pair[0] for pair in missing_pairs}),
            "failed_compound_ids": sorted({pair[0] for pair in missing_pairs}),
            "failed_compound_replicates": len(missing_pairs),
            "partial_success": partial_success,
            "affinity_count": sum(
                1 for artifact in artifacts if artifact.artifact_type == "affinity_result"
            ),
            "warning": (
                (
                    f"Partial result: {len(completed_pairs)}/{expected_pairs} "
                    "compound-replicate predictions produced readable complexes."
                )
                if partial_success
                else ""
            ),
            "error": error,
        },
    )
    completed = _utc_now_iso()
    metadata.update(
        {
            "status": "completed" if success else "failed",
            "partial_success": partial_success,
            "updated_at": completed,
            "completed_at": completed,
        }
    )
    if error:
        metadata["error"] = error
    else:
        metadata.pop("error", None)
    _write_json(metadata_path, metadata)
    return JobRecord.load(run_dir, task_group=REFOLDING_TASK_GROUP)


def run_boltz2_refolding_job(
    *,
    target_path: Path,
    target_artifact: ArtifactRef,
    compound_paths: Iterable[Path],
    compound_artifacts: Iterable[ArtifactRef],
    reference_ligand_artifact: ArtifactRef | None = None,
    image: str = DEFAULT_BOLTZ2_IMAGE,
    cache_dir: Path | None = None,
    gpu_device: str = "all",
    max_compounds: int = 0,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 5,
    use_msa_server: bool = True,
    use_potentials: bool = True,
    msa_paths: Iterable[Path] = (),
    replicates: int = 1,
    seed_start: int = 1001,
    protein_sequences: Iterable[tuple[str, str]] | None = None,
    launch_context: str = "",
    launch_campaign_id: str = "",
    launch_campaign_label: str = "",
    campaign_purpose: str = "",
    enqueue_only: bool = False,
) -> JobRecord:
    compound_paths = tuple(Path(path) for path in compound_paths)
    compound_artifacts = tuple(compound_artifacts)
    cache_dir = Path(cache_dir) if cache_dir is not None else configured_boltz2_cache_dir()
    readiness = boltz2_readiness(cache_dir)
    if not readiness["ready"]:
        raise ValueError(f"Boltz-2 checkpoints are incomplete: {readiness['cache_dir']}")
    polymers = (
        _polymer_entities(protein_sequences)
        if protein_sequences is not None
        else polymer_sequences_from_pdb(Path(target_path))
    )
    if not polymers or any(not sequence for _, _, sequence in polymers):
        raise ValueError("At least one protein, DNA, or RNA sequence is required")
    protein_count = sum(1 for kind, _, _ in polymers if kind == "protein")
    compounds: list[tuple[str, str]] = []
    for path in compound_paths:
        compounds.extend(compounds_from_path(path))
    compounds = limit_compounds(compounds, max_compounds)
    provided_msas = tuple(Path(path) for path in msa_paths)
    if provided_msas and len(provided_msas) != protein_count:
        raise ValueError("Boltz-2 requires one provided MSA per protein chain")
    if any(not path.is_file() for path in provided_msas):
        raise FileNotFoundError("One or more provided Boltz-2 MSA files are missing")
    replicates = int(replicates)
    seed_start = int(seed_start)
    if not 1 <= replicates <= 100:
        raise ValueError("Boltz-2 independent runs must be between 1 and 100")
    if seed_start < 1 or seed_start + replicates - 1 >= 2_147_483_647:
        raise ValueError("Boltz-2 seed range must contain positive 32-bit integers")
    run_id = str(uuid4())
    run_dir = runs_root() / REFOLDING_TASK_GROUP / run_id
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)
    staged_msa_paths: list[str] = []
    msa_dir = run_dir / "msa"
    if provided_msas:
        msa_dir.mkdir()
    for index, source_msa in enumerate(provided_msas):
        suffix = source_msa.suffix.lower() if source_msa.suffix.lower() in {".csv", ".a3m"} else ".a3m"
        target_msa = msa_dir / f"protein_chain_{index + 1}{suffix}"
        shutil.copy2(source_msa, target_msa)
        staged_msa_paths.append(f"/work/msa/{target_msa.name}")
    input_msa_paths = staged_msa_paths
    if not input_msa_paths and not use_msa_server:
        # Boltz requires an explicit MSA value when the remote MSA server is
        # disabled. ``empty`` is its documented single-sequence mode.
        input_msa_paths = ["empty"] * protein_count
    used_names: set[str] = set()
    for index, (compound_id, smiles) in enumerate(compounds, start=1):
        name = _safe_id(compound_id, f"compound_{index:07d}")
        if name in used_names:
            name = f"{name}-{index}"
        used_names.add(name)
        (input_dir / f"{name}.yaml").write_text(
            boltz2_input_yaml(
                polymers, smiles, enable_affinity=True,
                protein_msa_paths=input_msa_paths,
            )
        )
    if provided_msas:
        use_msa_server = False
    commands = [
        build_boltz2_command(
            image=image,
            run_dir=run_dir,
            cache_dir=cache_dir,
            gpu_device=gpu_device,
            sampling_steps=sampling_steps,
            recycling_steps=recycling_steps,
            diffusion_samples=diffusion_samples,
            sampling_steps_affinity=sampling_steps_affinity,
            diffusion_samples_affinity=diffusion_samples_affinity,
            use_msa_server=use_msa_server,
            use_potentials=use_potentials,
            output_dir=f"/work/output/replicate_{replicate:03d}",
            seed=seed_start + replicate - 1,
        )
        for replicate in range(1, replicates + 1)
    ]
    selected_gpu_ids = (
        []
        if str(gpu_device).strip().lower() in {"all", "auto", "automatic"}
        else [int(str(gpu_device).strip().lower().removeprefix("device=").removeprefix("gpu "))]
    )
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "complex_refolding",
        "workflow": "boltz2_refolding",
        "operation": "refolding",
        "status": "queued" if enqueue_only else "running",
        "tool": "Boltz-2",
        "engine": "Boltz-2",
        "docker_image": image,
        "parent_run_id": target_artifact.run_id,
        "use_gpu": True,
        "gpu_device": str(gpu_device),
        "compound_count": len(compounds),
        "replicates": replicates,
        "seed_start": seed_start,
        "protein_input_mode": "sequence" if protein_sequences is not None else "prepared_target",
        "polymer_entity_counts": {
            kind: sum(1 for entity_kind, _, _ in polymers if entity_kind == kind)
            for kind in ("protein", "dna", "rna")
        },
        "launch_context": str(launch_context),
        "launch_campaign_id": str(launch_campaign_id),
        "launch_campaign_label": str(launch_campaign_label),
        "campaign_purpose": str(campaign_purpose),
        "reference_ligand_run_id": (
            reference_ligand_artifact.run_id if reference_ligand_artifact else ""
        ),
        "created_at": now,
        "updated_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "target": target_artifact.to_dict(),
            "compound_sets": [artifact.to_dict() for artifact in compound_artifacts],
            "reference_ligand_artifact": (
                reference_ligand_artifact.to_dict()
                if reference_ligand_artifact is not None
                else None
            ),
            "parameters": {
                "max_compounds": len(compounds),
                "protein_input_mode": metadata["protein_input_mode"],
                "gpu_device": str(gpu_device),
                "sampling_steps": int(sampling_steps),
                "recycling_steps": int(recycling_steps),
                "diffusion_samples": int(diffusion_samples),
                "sampling_steps_affinity": int(sampling_steps_affinity),
                "diffusion_samples_affinity": int(diffusion_samples_affinity),
                "replicates": replicates,
                "seed_start": seed_start,
                "use_msa_server": bool(use_msa_server),
                "provided_msa_count": len(provided_msas),
                "use_potentials": bool(use_potentials),
                "launch_campaign_id": str(launch_campaign_id),
                "launch_campaign_label": str(launch_campaign_label),
                "campaign_purpose": str(campaign_purpose),
                "reference_layout": {"cache": "boltz_models"},
                "readiness": readiness,
            },
        },
    )
    write_registered_command_record(
        run_dir,
        tool_id="boltz2",
        commands=commands,
        image=image,
        selected_gpu_ids=selected_gpu_ids,
    )
    if enqueue_only:
        resources = registered_tool("boltz2", image=image).resources.to_dict()
        if selected_gpu_ids:
            resources["gpu_ids"] = selected_gpu_ids
        metadata.update(
            {
                "status": "queued",
                "queued_at": now,
                "queued_command": commands[0],
                "queued_commands": commands,
                "gpu_queued": True,
                "resources": resources,
                "worker_finalizer": "boltz2_refolding",
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        write_artifact_manifest(run_dir, [])
        return JobRecord.load(run_dir, task_group=REFOLDING_TASK_GROUP)
    try:
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        returncode = 0
        for command in commands:
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            stdout_parts.append(process.stdout or "")
            stderr_parts.append(process.stderr or "")
            returncode = int(process.returncode)
            if returncode:
                break
        stdout, stderr = "\n".join(stdout_parts), "\n".join(stderr_parts)
    except Exception as exc:
        returncode, stdout, stderr = -1, "", str(exc)
    (run_dir / "stdout.log").write_text(stdout)
    (run_dir / "stderr.log").write_text(stderr)
    return finalize_boltz2_refolding_job(run_dir, returncode=returncode)


def queue_boltz2_refolding_job(**parameters: Any) -> JobRecord:
    """Create a queued typed Boltz-2 refolding campaign."""
    return run_boltz2_refolding_job(enqueue_only=True, **parameters)


def configured_nesso_reference_paths() -> tuple[Path, Path, Path]:
    """Return the Nesso checkpoint, CCD, and reusable ESM Hugging Face cache."""
    root = reference_root(create=False)
    checkpoint = Path(
        os.getenv("MN_NESSO_CHECKPOINT_DIR", str(root / "nesso" / "v1.0.0"))
    ).expanduser()
    ccd = Path(os.getenv("MN_NESSO_CCD_PATH", str(root / "nesso" / "ccd.pkl"))).expanduser()
    configured_esm = os.getenv("MN_NESSO_ESM_CACHE_DIR")
    portable_esm = root / "nesso" / "huggingface"
    shared_esm = root / "proteina-complexa" / "hf-cache"
    esm_cache = Path(configured_esm).expanduser() if configured_esm else (
        portable_esm if portable_esm.exists() or not shared_esm.exists() else shared_esm
    )
    return checkpoint, ccd, esm_cache


def nesso_readiness(
    checkpoint_dir: Path | None = None,
    ccd_path: Path | None = None,
    esm_cache_dir: Path | None = None,
) -> dict[str, Any]:
    configured = configured_nesso_reference_paths()
    checkpoint = Path(checkpoint_dir or configured[0]).expanduser().resolve()
    ccd = Path(ccd_path or configured[1]).expanduser().resolve()
    esm_cache = Path(esm_cache_dir or configured[2]).expanduser().resolve()
    esm_root = esm_cache / NESSO_ESM_MODEL
    snapshots = tuple(sorted((esm_root / "snapshots").glob("*")))
    esm_ready = any(
        (snapshot / "model.safetensors").is_file()
        and (snapshot / "config.json").is_file()
        and (snapshot / "vocab.txt").is_file()
        for snapshot in snapshots
    )
    model_path = checkpoint / "model.safetensors"
    hparams_path = checkpoint / "hparams.json"
    return {
        "checkpoint_dir": str(checkpoint),
        "model_path": str(model_path),
        "hparams_path": str(hparams_path),
        "ccd_path": str(ccd),
        "esm_cache_dir": str(esm_cache),
        "checkpoint_ready": model_path.is_file() and hparams_path.is_file(),
        "ccd_ready": ccd.is_file(),
        "esm_ready": esm_ready,
        "ready": model_path.is_file() and hparams_path.is_file() and ccd.is_file() and esm_ready,
    }


def nesso_input_yaml(
    protein_sequences: Iterable[tuple[str, str]], ligand_smiles: str
) -> str:
    """Create one native Nesso protein-ligand affinity input."""
    return boltz2_input_yaml(protein_sequences, ligand_smiles, enable_affinity=True)


def build_nesso_command(
    *,
    image: str,
    run_dir: Path,
    checkpoint_dir: Path,
    ccd_path: Path,
    esm_cache_dir: Path,
    gpu_device: str = "all",
    recycling_steps: int = 5,
    num_workers: int = 2,
    refine_protein_inference: bool = True,
    refine_protein_cutoff: float = 22.0,
    refine_protein_tokens_budget: int = 256,
    affinity_protein_cutoff: float = 15.0,
    seed: int = 42,
    save_metadata: bool = False,
    input_dir: str = "/work/inputs",
    output_dir: str = "/work/output",
) -> list[str]:
    selected = str(gpu_device).strip().lower().removeprefix("gpu ") or "0"
    gpu_ids = None if selected in {"all", "auto", "automatic"} else (
        int(selected.removeprefix("device=")),
    )
    native = [
        "predict", str(input_dir), "--out_dir", str(output_dir),
        "--cache", "/cache", "--checkpoint", "/reference/checkpoint",
        "--ccd", "/reference/ccd.pkl", "--accelerator", "gpu", "--devices", "1",
        "--precision", "bf16-mixed", "--recycling_steps", str(max(1, int(recycling_steps))),
        "--num_workers", str(max(0, int(num_workers))), "--require_affinity", "--override",
        "--refine_protein_cutoff", str(float(refine_protein_cutoff)),
        "--refine_protein_tokens_budget", str(max(1, int(refine_protein_tokens_budget))),
        "--affinity_protein_cutoff", str(float(affinity_protein_cutoff)),
        "--seed", str(int(seed)), "--no_kernels",
    ]
    native.append("--refine_protein_inference" if refine_protein_inference else "--no_refine_protein_inference")
    if save_metadata:
        native.append("--save_metadata")
    return build_docker_command(
        DockerRunSpec(
            tool=registered_tool("nesso_affinity", image=image),
            command=tuple(native),
            mounts=(
                DockerMount(Path(run_dir), "/work"),
                DockerMount(Path(checkpoint_dir), "/reference/checkpoint", read_only=True),
                DockerMount(Path(ccd_path), "/reference/ccd.pkl", read_only=True),
                DockerMount(Path(esm_cache_dir), "/cache/huggingface", read_only=True),
            ),
            environment={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            gpu_devices=gpu_ids,
            shm_size="8g",
            use_host_user=False,
        )
    )


def _finite_float(payload: dict[str, Any], key: str) -> float | None:
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _nesso_replicate_summaries(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for candidate_id in sorted({str(row.get("candidate_id") or "") for row in metrics}):
        selected = [
            row for row in metrics if str(row.get("candidate_id") or "") == candidate_id
        ]
        selected.sort(key=lambda row: int(row.get("replicate") or 1))
        values = [float(row["affinity_log10_ic50_uM"]) for row in selected]
        micromolar = [10.0 ** value for value in values]
        probabilities = [
            float(row["binder_probability"])
            for row in selected if row.get("binder_probability") is not None
        ]
        spread = [
            float(row["ensemble_spread_log10_ic50_uM"])
            for row in selected
            if row.get("ensemble_spread_log10_ic50_uM") is not None
        ]
        median = statistics.median(values)
        representative = min(
            selected,
            key=lambda row: (
                round(abs(float(row["affinity_log10_ic50_uM"]) - median), 12),
                int(row.get("replicate") or 1),
            ),
        )
        summaries.append(
            {
                "candidate_id": candidate_id,
                "replicate_count": len(selected),
                "mean_affinity_log10_ic50_uM": statistics.mean(values),
                "sample_sd_affinity_log10_ic50_uM": (
                    statistics.stdev(values) if len(values) > 1 else None
                ),
                "derived_pIC50_from_mean": 6.0 - statistics.mean(values),
                "geometric_mean_ic50_uM": 10.0 ** statistics.mean(values),
                "arithmetic_mean_ic50_uM": statistics.mean(micromolar),
                "sample_sd_ic50_uM": (
                    statistics.stdev(micromolar) if len(micromolar) > 1 else None
                ),
                "mean_binder_probability": (
                    statistics.mean(probabilities) if probabilities else None
                ),
                "sample_sd_binder_probability": (
                    statistics.stdev(probabilities) if len(probabilities) > 1 else None
                ),
                "mean_ensemble_spread_log10_ic50_uM": (
                    statistics.mean(spread) if spread else None
                ),
                "representative_replicate": int(representative.get("replicate") or 1),
                "representative_native_file": str(
                    representative.get("native_file") or ""
                ),
            }
        )
    return summaries


def _collect_nesso_outputs(run_dir: Path) -> tuple[list[ArtifactRef], list[dict[str, Any]]]:
    artifacts: list[ArtifactRef] = []
    metrics: list[dict[str, Any]] = []
    metadata = json.loads((run_dir / "metadata.json").read_text())
    seed_start = int(metadata.get("seed_start") or metadata.get("seed") or 42)
    output_root = run_dir / "output"
    for affinity_path in sorted(output_root.glob("**/predictions/*/affinity.json")):
        try:
            native = json.loads(affinity_path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(native, dict):
            continue
        value = _finite_float(native, "affinity_pred_value")
        if value is None:
            continue
        candidate_id = affinity_path.parent.name
        replicate = _output_replicate(affinity_path, output_root)
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "replicate": replicate,
            "seed": seed_start + replicate - 1,
            "affinity_log10_ic50_uM": value,
            "pIC50": 6.0 - value,
            "ic50_uM": 10.0 ** value,
            "native_file": affinity_path.relative_to(run_dir).as_posix(),
        }
        key_map = {
            "affinity_pred_value1": "ensemble_log10_ic50_uM_1",
            "affinity_pred_value2": "ensemble_log10_ic50_uM_2",
            "affinity_probability_binary": "binder_probability",
            "entropy_crop_pl": "cropped_protein_ligand_entropy",
            "entropy_pp": "protein_protein_entropy",
            "entropy_pl": "protein_ligand_entropy",
            "entropy_ll": "ligand_ligand_entropy",
            "entropy_crop_pp": "cropped_protein_protein_entropy",
            "entropy_crop_ll": "cropped_ligand_ligand_entropy",
        }
        for source, normalized in key_map.items():
            parsed = _finite_float(native, source)
            if parsed is not None:
                row[normalized] = parsed
        first = row.get("ensemble_log10_ic50_uM_1")
        second = row.get("ensemble_log10_ic50_uM_2")
        if first is not None and second is not None:
            row["ensemble_spread_log10_ic50_uM"] = abs(float(first) - float(second))
        cropped_pl = row.get("cropped_protein_ligand_entropy")
        row["placement_confident"] = cropped_pl is not None and float(cropped_pl) > 0.0
        artifact_metadata = {
            key: row[key]
            for key in (
                "affinity_log10_ic50_uM", "pIC50", "binder_probability",
                "ensemble_spread_log10_ic50_uM", "placement_confident",
            )
            if key in row
        }
        artifacts.append(
            ArtifactRef.from_path(
                run_dir, affinity_path, "affinity_result",
                role=f"{candidate_id}:replicate_{replicate:03d}",
                metadata=artifact_metadata,
            )
        )
        artifacts.append(
            ArtifactRef.from_path(
                run_dir, affinity_path, "native_output",
                role=f"{candidate_id}:replicate_{replicate:03d}"
            )
        )
        metadata_path = affinity_path.with_name("predictions.safetensors")
        if metadata_path.is_file():
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir, metadata_path, "prediction_features",
                    role=f"{candidate_id}:replicate_{replicate:03d}"
                )
            )
        metrics.append(row)
    if metrics:
        table = run_dir / "artifacts" / "nesso_affinity.csv"
        _write_metric_table(table, metrics)
        artifacts.append(
            ArtifactRef.from_path(run_dir, table, "prediction_metrics", role="affinity_campaign")
        )
    return artifacts, metrics


def finalize_nesso_affinity_job(run_dir: Path, *, returncode: int) -> JobRecord:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    artifacts, metrics = _collect_nesso_outputs(run_dir) if returncode == 0 else ([], [])
    # The staged YAML files are the authoritative Nesso candidate inventory.
    # Campaign retries may inherit stale aggregate metadata from an earlier
    # multi-compound launch, while each target-associated refolding job stages
    # only its own ligand.  Using that stale count incorrectly marks complete
    # native output as a partial failure.
    staged_compound_count = sum(1 for path in (run_dir / "inputs").glob("*.yaml") if path.is_file())
    compound_count = staged_compound_count or int(metadata.get("compound_count") or 0)
    replicates = int(metadata.get("replicates") or 1)
    completed_pairs = {
        (str(row.get("candidate_id") or ""), int(row.get("replicate") or 1))
        for row in metrics
    }
    expected_pairs = compound_count * replicates
    success = returncode == 0 and len(completed_pairs) == expected_pairs
    summaries = _nesso_replicate_summaries(metrics) if replicates > 1 else []
    if summaries:
        summary_path = run_dir / "artifacts" / "nesso_replicate_summary.csv"
        _write_metric_table(summary_path, summaries)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                summary_path,
                "prediction_metrics",
                role="replicate_summary",
                metadata={
                    "log_aggregation": "mean log10(IC50/uM)",
                    "linear_aggregation": "arithmetic and geometric IC50 means",
                },
            )
        )
    stderr_path = run_dir / "stderr.log"
    stderr_path.touch(exist_ok=True)
    error = "" if success else (
        stderr_path.read_text(errors="replace")[-4000:]
        or (
            "Nesso produced no readable affinity.json outputs; "
            f"{len(completed_pairs)}/{expected_pairs} expected "
            "candidate-replicate predictions were complete"
        )
    )
    write_artifact_manifest(run_dir, artifacts)
    best = min(metrics, key=lambda row: float(row["affinity_log10_ic50_uM"])) if metrics else {}
    _write_json(
        run_dir / "result.json",
        {
            "success": success,
            "returncode": returncode,
            "affinity_count": len(metrics),
            "compound_count": compound_count,
            "replicates": replicates,
            "completed_replicate_pairs": len(completed_pairs),
            "best_candidate_id": best.get("candidate_id", ""),
            "best_affinity_log10_ic50_uM": best.get("affinity_log10_ic50_uM"),
            "best_pIC50": best.get("pIC50"),
            "untrusted_placement_count": sum(not bool(row["placement_confident"]) for row in metrics),
            "structure_output": False,
            "error": error,
        },
    )
    completed = _utc_now_iso()
    metadata.update(
        {
            "status": "completed" if success else "failed",
            "compound_count": compound_count,
            "updated_at": completed,
            "completed_at": completed,
        }
    )
    if error:
        metadata["error"] = error
    else:
        metadata.pop("error", None)
    _write_json(metadata_path, metadata)
    return JobRecord.load(run_dir, task_group=REFOLDING_TASK_GROUP)


def run_nesso_affinity_job(
    *,
    target_path: Path,
    target_artifact: ArtifactRef,
    compound_paths: Iterable[Path],
    compound_artifacts: Iterable[ArtifactRef],
    image: str = DEFAULT_NESSO_IMAGE,
    checkpoint_dir: Path | None = None,
    ccd_path: Path | None = None,
    esm_cache_dir: Path | None = None,
    gpu_device: str = "all",
    max_compounds: int = 0,
    recycling_steps: int = 5,
    num_workers: int = 2,
    refine_protein_inference: bool = True,
    refine_protein_cutoff: float = 22.0,
    refine_protein_tokens_budget: int = 256,
    affinity_protein_cutoff: float = 15.0,
    seed: int = 42,
    replicates: int = 1,
    save_metadata: bool = False,
    launch_campaign_id: str = "",
    launch_campaign_label: str = "",
    campaign_purpose: str = "",
    enqueue_only: bool = False,
) -> JobRecord:
    compound_paths = tuple(Path(path) for path in compound_paths)
    compound_artifacts = tuple(compound_artifacts)
    defaults = configured_nesso_reference_paths()
    checkpoint = Path(checkpoint_dir or defaults[0])
    ccd = Path(ccd_path or defaults[1])
    esm_cache = Path(esm_cache_dir or defaults[2])
    readiness = nesso_readiness(checkpoint, ccd, esm_cache)
    if not readiness["ready"]:
        raise ValueError("Nesso references are incomplete: " + json.dumps(readiness, sort_keys=True))
    proteins = protein_sequences_from_pdb(Path(target_path))
    compounds: list[tuple[str, str]] = []
    for path in compound_paths:
        compounds.extend(compounds_from_path(path))
    compounds = limit_compounds(compounds, max_compounds)
    replicates = int(replicates)
    seed_start = int(seed)
    if not 1 <= replicates <= 100:
        raise ValueError("Nesso independent runs must be between 1 and 100")
    if seed_start < 1 or seed_start + replicates - 1 >= 2_147_483_647:
        raise ValueError("Nesso seed range must contain positive 32-bit integers")
    run_id = str(uuid4())
    run_dir = runs_root() / REFOLDING_TASK_GROUP / run_id
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)
    used_names: set[str] = set()
    for index, (compound_id, smiles) in enumerate(compounds, start=1):
        name = _safe_id(compound_id, f"compound_{index:07d}")
        if name in used_names:
            name = f"{name}-{index}"
        used_names.add(name)
        (input_dir / f"{name}.yaml").write_text(nesso_input_yaml(proteins, smiles))
    commands = [
        build_nesso_command(
            image=image, run_dir=run_dir, checkpoint_dir=checkpoint, ccd_path=ccd,
            esm_cache_dir=esm_cache, gpu_device=gpu_device, recycling_steps=recycling_steps,
            num_workers=num_workers, refine_protein_inference=refine_protein_inference,
            refine_protein_cutoff=refine_protein_cutoff,
            refine_protein_tokens_budget=refine_protein_tokens_budget,
            affinity_protein_cutoff=affinity_protein_cutoff,
            seed=seed_start + replicate - 1,
            save_metadata=save_metadata,
            output_dir=f"/work/output/replicate_{replicate:03d}",
        )
        for replicate in range(1, replicates + 1)
    ]
    selected_gpu_ids = [] if str(gpu_device).strip().lower() in {"all", "auto", "automatic"} else [
        int(str(gpu_device).strip().lower().removeprefix("device=").removeprefix("gpu "))
    ]
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "learned_binding_affinity",
        "workflow": "nesso_affinity",
        "operation": "refolding",
        "status": "queued" if enqueue_only else "running",
        "tool": "Nesso-1",
        "engine": "Nesso-1",
        "docker_image": image,
        "parent_run_id": target_artifact.run_id,
        "use_gpu": True,
        "gpu_device": str(gpu_device),
        "compound_count": len(compounds),
        "replicates": replicates,
        "seed_start": seed_start,
        "structure_output": False,
        "launch_campaign_id": str(launch_campaign_id),
        "launch_campaign_label": str(launch_campaign_label),
        "campaign_purpose": str(campaign_purpose),
        "created_at": now,
        "updated_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "target": target_artifact.to_dict(),
            "compound_sets": [artifact.to_dict() for artifact in compound_artifacts],
            "parameters": {
                "max_compounds": len(compounds), "gpu_device": str(gpu_device),
                "recycling_steps": int(recycling_steps), "num_workers": int(num_workers),
                "refine_protein_inference": bool(refine_protein_inference),
                "refine_protein_cutoff": float(refine_protein_cutoff),
                "refine_protein_tokens_budget": int(refine_protein_tokens_budget),
                "affinity_protein_cutoff": float(affinity_protein_cutoff),
                "seed": seed_start, "seed_start": seed_start,
                "replicates": replicates,
                "save_metadata": bool(save_metadata), "no_kernels": True,
                "launch_campaign_id": str(launch_campaign_id),
                "launch_campaign_label": str(launch_campaign_label),
                "campaign_purpose": str(campaign_purpose),
                "reference_layout": {
                    "checkpoint": "nesso/v1.0.0", "ccd": "nesso/ccd.pkl",
                    "esm_model": "facebook/esm2_t33_650M_UR50D",
                },
                "readiness": readiness,
            },
        },
    )
    write_registered_command_record(
        run_dir, tool_id="nesso_affinity", commands=commands, image=image,
        selected_gpu_ids=selected_gpu_ids,
    )
    if enqueue_only:
        resources = registered_tool("nesso_affinity", image=image).resources.to_dict()
        if selected_gpu_ids:
            resources["gpu_ids"] = selected_gpu_ids
        metadata.update(
            {
                "queued_at": now, "queued_command": commands[0],
                "queued_commands": commands, "gpu_queued": True,
                "resources": resources, "worker_finalizer": "nesso_affinity",
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        write_artifact_manifest(run_dir, [])
        return JobRecord.load(run_dir, task_group=REFOLDING_TASK_GROUP)
    try:
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        returncode = 0
        for command in commands:
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            stdout_parts.append(process.stdout or "")
            stderr_parts.append(process.stderr or "")
            returncode = int(process.returncode)
            if returncode:
                break
        stdout, stderr = "\n".join(stdout_parts), "\n".join(stderr_parts)
    except Exception as exc:
        returncode, stdout, stderr = -1, "", str(exc)
    (run_dir / "stdout.log").write_text(stdout)
    (run_dir / "stderr.log").write_text(stderr)
    return finalize_nesso_affinity_job(run_dir, returncode=returncode)


def queue_nesso_affinity_job(**parameters: Any) -> JobRecord:
    """Create a queued typed Nesso affinity-only cofolding campaign."""
    return run_nesso_affinity_job(enqueue_only=True, **parameters)
