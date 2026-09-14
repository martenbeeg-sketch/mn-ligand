from __future__ import annotations

import csv
import io
import json
import re
import shutil
import tarfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator
from uuid import uuid4

import yaml
from rdkit import Chem

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.runtime import runs_root


BENCHMARK_DATASET_TASK_GROUP = "benchmark-datasets"
BENCHMARK_TARGET_TASK_GROUP = "benchmark-targets"
BENCHMARK_DATASET_SCHEMA_VERSION = 1
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 20 * 1024**3

_RECEPTOR_SUFFIXES = (
    "_protein_processed.pdb",
    "_protein.pdb",
    "_processed.pdb",
    "_receptor.pdb",
    "protein.pdb",
    "receptor.pdb",
)
_LIGAND_SUFFIXES = (
    "_ligand.sdf",
    "_ligands.sdf",
    "_ligand.pdb",
    "ligand.sdf",
    "reference_ligand.sdf",
)
_COMPLEX_SUFFIXES = ("_complex.pdb", "complex.pdb")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _safe_member_name(value: str) -> str:
    normalized = str(value).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
    ):
        raise ValueError(f"Unsafe archive member path: {value}")
    return path.as_posix()


def _safe_case_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-.")
    if not cleaned:
        raise ValueError("Every benchmark case requires a non-empty case_id")
    return cleaned[:160]


def _archive_members(data: bytes, filename: str) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    total = 0
    lower = filename.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            if len(infos) > MAX_ARCHIVE_MEMBERS:
                raise ValueError("Archive contains too many files")
            for info in infos:
                name = _safe_member_name(info.filename)
                total += int(info.file_size)
                if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValueError("Archive expands beyond the supported size limit")
                members[name] = archive.read(info)
    elif lower.endswith((".tar", ".tar.gz", ".tgz")):
        mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(fileobj=io.BytesIO(data), mode=mode) as archive:
            infos = [item for item in archive.getmembers() if item.isfile()]
            if len(infos) > MAX_ARCHIVE_MEMBERS:
                raise ValueError("Archive contains too many files")
            for info in infos:
                name = _safe_member_name(info.name)
                total += int(info.size)
                if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValueError("Archive expands beyond the supported size limit")
                handle = archive.extractfile(info)
                if handle is not None:
                    members[name] = handle.read()
    else:
        raise ValueError("Benchmark datasets must be ZIP, TAR, TAR.GZ, or TGZ archives")
    if not members:
        raise ValueError("The archive contains no regular files")
    return members


class _DirectoryMembers(Mapping[str, bytes]):
    """Lazy file mapping for benchmark trees that may contain several gigabytes."""

    def __init__(self, paths: dict[str, Path]) -> None:
        self.paths = paths

    def __getitem__(self, key: str) -> bytes:
        return self.paths[key].read_bytes()

    def __iter__(self) -> Iterator[str]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)


def _directory_members(root: Path) -> Mapping[str, bytes]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths: dict[str, Path] = {}
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            continue
        relative = _safe_member_name(path.relative_to(root).as_posix())
        total += path.stat().st_size
        if len(paths) >= MAX_ARCHIVE_MEMBERS:
            raise ValueError("Dataset directory contains too many files")
        if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError("Dataset directory exceeds the supported size limit")
        paths[relative] = path
    if not paths:
        raise ValueError("The dataset directory contains no regular files")
    return _DirectoryMembers(paths)


def _suffix_match(name: str, suffixes: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in suffixes)


def _case_id_from_path(path: str, suffixes: Iterable[str]) -> str:
    stem = PurePosixPath(path).name
    lowered = stem.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix):
            candidate = stem[: -len(suffix)]
            if candidate:
                return _safe_case_id(candidate)
    parent = PurePosixPath(path).parent.name
    return _safe_case_id(parent or PurePosixPath(path).stem)


def _parse_manifest(data: bytes, filename: str) -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    text = data.decode("utf-8-sig")
    if suffix == ".csv":
        return [dict(row) for row in csv.DictReader(io.StringIO(text))]
    payload = yaml.safe_load(text) if suffix in {".yaml", ".yml"} else json.loads(text)
    if isinstance(payload, dict):
        payload = payload.get("cases") or payload.get("entries") or payload.get("records")
    if not isinstance(payload, list):
        raise ValueError("Manifest must contain a list of benchmark cases")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _find_member(members: Mapping[str, bytes], requested: str) -> str:
    normalized = _safe_member_name(requested)
    if normalized in members:
        return normalized
    matches = [name for name in members if name.endswith(f"/{normalized}")]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Manifest path is missing or ambiguous: {requested}")


def _manifest_cases(
    members: Mapping[str, bytes],
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        case_id = _safe_case_id(
            row.get("case_id") or row.get("id") or row.get("target_id") or f"case-{index:05d}"
        )
        receptor_value = row.get("receptor") or row.get("protein") or row.get("receptor_path")
        ligand_value = (
            row.get("ligand")
            or row.get("reference_ligand")
            or row.get("ligand_path")
        )
        if not receptor_value or not ligand_value:
            raise ValueError(f"Case {case_id} requires receptor and ligand paths")
        case = {
            "case_id": case_id,
            "target_id": str(row.get("target_id") or case_id),
            "split": str(row.get("split") or ""),
            "receptor_member": _find_member(members, str(receptor_value)),
            "ligand_member": _find_member(members, str(ligand_value)),
            "complex_member": "",
            "sequence_member": "",
            "ligand_smiles": str(row.get("ligand_smiles") or row.get("smiles") or ""),
            "metadata": {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "case_id",
                    "id",
                    "target_id",
                    "split",
                    "receptor",
                    "protein",
                    "receptor_path",
                    "ligand",
                    "reference_ligand",
                    "ligand_path",
                    "complex",
                    "complex_path",
                    "sequence",
                    "sequence_path",
                    "ligand_smiles",
                    "smiles",
                }
            },
        }
        complex_value = row.get("complex") or row.get("complex_path")
        sequence_value = row.get("sequence") or row.get("sequence_path")
        if complex_value:
            case["complex_member"] = _find_member(members, str(complex_value))
        if sequence_value:
            case["sequence_member"] = _find_member(members, str(sequence_value))
        cases.append(case)
    return cases


def _auto_cases(members: Mapping[str, bytes]) -> tuple[list[dict[str, Any]], str]:
    names = " ".join(members).lower()
    if "casp15_set/" in names or "casp15_predicted_structures/" in names:
        cases: list[dict[str, Any]] = []
        member_names = set(members)
        for complex_member in members:
            path = PurePosixPath(complex_member)
            if path.parent.name != "targets" or not path.name.endswith("_lig.pdb"):
                continue
            case_id = _safe_case_id(path.name[: -len("_lig.pdb")])
            sequence = (path.parent / f"{case_id}.seq.txt").as_posix()
            smiles = (path.parent / f"{case_id}.smiles.txt").as_posix()
            cases.append(
                {
                    "case_id": case_id,
                    "target_id": case_id,
                    "split": "CASP15",
                    "receptor_member": complex_member,
                    "ligand_member": complex_member,
                    "complex_member": complex_member,
                    "sequence_member": sequence if sequence in member_names else "",
                    "ligand_smiles": (
                        members[smiles].decode(errors="replace").strip()
                        if smiles in member_names
                        else ""
                    ),
                    "metadata": {
                        "source_complex_requires_split": True,
                        "smiles_member": smiles if smiles in member_names else "",
                    },
                }
            )
        if not cases:
            raise ValueError("No CASP15 target reference complexes were detected")
        return cases, "posebench:casp15"

    receptors = [name for name in members if _suffix_match(name, _RECEPTOR_SUFFIXES)]
    ligands = [name for name in members if _suffix_match(name, _LIGAND_SUFFIXES)]
    complexes = [name for name in members if _suffix_match(name, _COMPLEX_SUFFIXES)]
    ligand_by_parent: dict[str, list[str]] = {}
    for path in ligands:
        ligand_by_parent.setdefault(str(PurePosixPath(path).parent), []).append(path)
    cases: list[dict[str, Any]] = []
    for receptor in receptors:
        parent = str(PurePosixPath(receptor).parent)
        case_id = _case_id_from_path(receptor, _RECEPTOR_SUFFIXES)
        candidates = ligand_by_parent.get(parent, [])
        if not candidates:
            candidates = [
                path
                for path in ligands
                if _case_id_from_path(path, _LIGAND_SUFFIXES).lower() == case_id.lower()
            ]
        if len(candidates) > 1:
            candidates = [
                path
                for path in candidates
                if path.lower().endswith(("_ligand.sdf", "_ligand.pdb"))
            ]
        if len(candidates) != 1:
            continue
        complex_candidates = [
            path
            for path in complexes
            if str(PurePosixPath(path).parent) == parent
            or _case_id_from_path(path, _COMPLEX_SUFFIXES).lower() == case_id.lower()
        ]
        cases.append(
            {
                "case_id": case_id,
                "target_id": case_id,
                "split": "",
                "receptor_member": receptor,
                "ligand_member": candidates[0],
                "complex_member": complex_candidates[0] if len(complex_candidates) == 1 else "",
                "sequence_member": "",
                "ligand_smiles": "",
                "metadata": {},
            }
        )
    unique: dict[str, dict[str, Any]] = {}
    for case in cases:
        if case["case_id"] in unique:
            raise ValueError(
                f"Auto-detection found duplicate case ID {case['case_id']}; provide a manifest"
            )
        unique[case["case_id"]] = case
    if not unique:
        raise ValueError(
            "No receptor/ligand pairs were detected. Add a CSV/JSON/YAML manifest "
            "or use per-case *_protein.pdb and *_ligand.sdf files."
        )
    profile = "generic"
    for value in ("posebusters_benchmark", "astex_diverse", "dockgen", "casp15"):
        if value in names:
            profile = f"posebench:{value}"
            break
    return list(unique.values()), profile


def _pdb_ligand_molecule(data: bytes, *, complex_source: bool = False):
    molecule = Chem.MolFromPDBBlock(
        data.decode(errors="replace"), removeHs=False, sanitize=False
    )
    if molecule is None:
        return None
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
    if complex_source:
        amino_acids = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL",
        }
        nucleic_acids = {
            "A", "C", "G", "I", "T", "U",
            "DA", "DC", "DG", "DI", "DT", "DU",
        }
        fragments = tuple(
            fragment
            for fragment in fragments
            if fragment.GetNumHeavyAtoms() > 1
            and not {
                atom.GetPDBResidueInfo().GetResidueName().strip().upper()
                for atom in fragment.GetAtoms()
                if atom.GetPDBResidueInfo() is not None
            }.intersection(amino_acids.union(nucleic_acids))
        )
        if not fragments:
            return None
        combined = fragments[0]
        for fragment in fragments[1:]:
            combined = Chem.CombineMols(combined, fragment)
        return combined
    return max(fragments, key=lambda item: item.GetNumHeavyAtoms(), default=None)


def _ligand_molecules(
    data: bytes, source_name: str, *, complex_source: bool = False
) -> list[Any]:
    if source_name.lower().endswith(".pdb"):
        molecule = _pdb_ligand_molecule(data, complex_source=complex_source)
        return [molecule] if molecule is not None else []
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(data), removeHs=False, sanitize=True)
    return [mol for mol in supplier if mol is not None]


def _ligand_summary(
    data: bytes, source_name: str, *, complex_source: bool = False
) -> dict[str, Any]:
    molecules = _ligand_molecules(data, source_name, complex_source=complex_source)
    if not molecules:
        raise ValueError(f"No RDKit-readable molecule in {source_name}")
    first = molecules[0]
    conformer = first.GetConformer() if first.GetNumConformers() else None
    smiles = Chem.MolToSmiles(first, canonical=True, isomericSmiles=True)
    return {
        "record_count": len(molecules),
        "smiles": smiles,
        "coordinate_dimension": (
            3 if conformer is not None and conformer.Is3D() else 2 if conformer else 0
        ),
    }


def analyze_benchmark_members(
    members: Mapping[str, bytes],
    *,
    manifest_data: bytes | None = None,
    manifest_filename: str = "",
    requested_profile: str = "Auto-detect",
) -> dict[str, Any]:
    manifest_name = manifest_filename
    if manifest_data is None:
        manifest_candidates = [
            name
            for name in members
            if PurePosixPath(name).name.lower()
            in {
                "benchmark_manifest.csv",
                "benchmark_manifest.json",
                "benchmark_manifest.yaml",
                "benchmark_manifest.yml",
                "manifest.csv",
                "manifest.json",
                "manifest.yaml",
                "manifest.yml",
            }
        ]
        if len(manifest_candidates) == 1:
            manifest_name = manifest_candidates[0]
            manifest_data = members[manifest_name]
    if manifest_data is not None:
        cases = _manifest_cases(
            members, _parse_manifest(manifest_data, manifest_name or "manifest.json")
        )
        profile = "generic-manifest"
    else:
        cases, profile = _auto_cases(members)
    requested = requested_profile.strip().lower()
    if requested.startswith("posebench") and not profile.startswith("posebench"):
        profile = requested.replace(" ", "-")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for case in cases:
        case_id = case["case_id"]
        if case_id in seen:
            raise ValueError(f"Duplicate benchmark case_id: {case_id}")
        seen.add(case_id)
        receptor_data = members[case["receptor_member"]]
        if not any(
            line.startswith((b"ATOM  ", b"HETATM"))
            for line in receptor_data.splitlines()
        ):
            errors.append({"case_id": case_id, "error": "Receptor has no PDB atoms"})
            continue
        try:
            ligand = _ligand_summary(
                members[case["ligand_member"]],
                case["ligand_member"],
                complex_source=bool(
                    case.get("metadata", {}).get("source_complex_requires_split")
                ),
            )
        except Exception as exc:
            errors.append({"case_id": case_id, "error": str(exc)})
            continue
        validated.append({**case, **ligand})
    return {
        "schema_version": BENCHMARK_DATASET_SCHEMA_VERSION,
        "profile": profile,
        "case_count": len(validated),
        "rejected_case_count": len(errors),
        "cases": validated,
        "errors": errors,
        "source_file_count": len(members),
        "manifest_name": manifest_name,
    }


def analyze_benchmark_archive(
    data: bytes,
    filename: str,
    *,
    manifest_data: bytes | None = None,
    manifest_filename: str = "",
    requested_profile: str = "Auto-detect",
) -> dict[str, Any]:
    return analyze_benchmark_members(
        _archive_members(data, filename),
        manifest_data=manifest_data,
        manifest_filename=manifest_filename,
        requested_profile=requested_profile,
    )


def _canonical_case_bytes(
    members: Mapping[str, bytes], case: dict[str, Any]
) -> tuple[bytes, bytes]:
    receptor_data = members[case["receptor_member"]]
    ligand_source = members[case["ligand_member"]]
    split_complex = bool(
        case.get("metadata", {}).get("source_complex_requires_split")
    )
    if split_complex:
        receptor_lines = [
            line
            for line in receptor_data.splitlines()
            if line.startswith(
                (b"HEADER", b"TITLE ", b"COMPND", b"SOURCE", b"ATOM  ", b"TER")
            )
        ]
        receptor_data = b"\n".join(receptor_lines) + b"\nEND\n"
    molecules = _ligand_molecules(
        ligand_source, case["ligand_member"], complex_source=split_complex
    )
    if not molecules:
        raise ValueError(f"No ligand could be canonicalized for {case['case_id']}")
    ligand_data = (Chem.MolToMolBlock(molecules[0]) + "\n$$$$\n").encode()
    return receptor_data, ligand_data


def create_benchmark_dataset_job(
    *,
    dataset_name: str,
    archive_data: bytes | None = None,
    archive_filename: str = "",
    source_directory: Path | None = None,
    manifest_data: bytes | None = None,
    manifest_filename: str = "",
    requested_profile: str = "Auto-detect",
    source_provenance: dict[str, Any] | None = None,
) -> JobRecord:
    if archive_data is not None:
        members = _archive_members(archive_data, archive_filename)
        source_kind = "uploaded_archive"
        source_label = archive_filename
    elif source_directory is not None:
        members = _directory_members(source_directory)
        source_kind = "server_directory"
        source_label = str(source_directory.expanduser().resolve())
    else:
        raise ValueError("Provide an archive or server-local dataset directory")
    analysis = analyze_benchmark_members(
        members,
        manifest_data=manifest_data,
        manifest_filename=manifest_filename,
        requested_profile=requested_profile,
    )
    if not analysis["cases"]:
        raise ValueError("The dataset contains no valid benchmark cases")

    run_id = str(uuid4())
    run_dir = runs_root() / BENCHMARK_DATASET_TASK_GROUP / run_id
    cases_root = run_dir / "artifacts" / "cases"
    reports_root = run_dir / "artifacts" / "reports"
    source_root = run_dir / "artifacts" / "source"
    cases_root.mkdir(parents=True, exist_ok=False)
    reports_root.mkdir(parents=True)
    source_root.mkdir(parents=True)
    artifacts: list[ArtifactRef] = []
    canonical_cases: list[dict[str, Any]] = []
    for case in analysis["cases"]:
        case_id = case["case_id"]
        case_root = cases_root / case_id
        case_root.mkdir()
        receptor = case_root / "receptor.pdb"
        ligand = case_root / "reference_ligand.sdf"
        receptor_data, ligand_data = _canonical_case_bytes(members, case)
        receptor.write_bytes(receptor_data)
        ligand.write_bytes(ligand_data)
        canonical = {
            "case_id": case_id,
            "target_id": case["target_id"],
            "split": case["split"],
            "receptor_path": receptor.relative_to(run_dir).as_posix(),
            "reference_ligand_path": ligand.relative_to(run_dir).as_posix(),
            "complex_path": "",
            "sequence_path": "",
            "ligand_smiles": case["ligand_smiles"] or case["smiles"],
            "ligand_record_count": case["record_count"],
            "coordinate_dimension": case["coordinate_dimension"],
            "source_paths": {
                "receptor": case["receptor_member"],
                "reference_ligand": case["ligand_member"],
                "complex": case["complex_member"],
                "sequence": case["sequence_member"],
            },
            "metadata": case["metadata"],
        }
        artifacts.extend(
            (
                ArtifactRef.from_path(
                    run_dir,
                    receptor,
                    "benchmark_receptor",
                    role="reference_receptor",
                    label=f"{case_id} receptor",
                    metadata={"case_id": case_id, "target_id": case["target_id"]},
                ),
                ArtifactRef.from_path(
                    run_dir,
                    ligand,
                    "benchmark_reference_ligand",
                    role="reference_ligand",
                    label=f"{case_id} reference ligand",
                    metadata={
                        "case_id": case_id,
                        "target_id": case["target_id"],
                        "smiles": canonical["ligand_smiles"],
                    },
                ),
            )
        )
        if case["complex_member"]:
            complex_path = case_root / "reference_complex.pdb"
            complex_path.write_bytes(members[case["complex_member"]])
            canonical["complex_path"] = complex_path.relative_to(run_dir).as_posix()
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir,
                    complex_path,
                    "benchmark_reference_complex",
                    role="reference_complex",
                    label=f"{case_id} reference complex",
                    metadata={"case_id": case_id, "target_id": case["target_id"]},
                )
            )
        if case["sequence_member"]:
            sequence_path = case_root / PurePosixPath(case["sequence_member"]).name
            sequence_path.write_bytes(members[case["sequence_member"]])
            canonical["sequence_path"] = sequence_path.relative_to(run_dir).as_posix()
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir,
                    sequence_path,
                    "benchmark_sequence",
                    role="reference_sequence",
                    label=f"{case_id} sequence",
                    metadata={"case_id": case_id},
                )
            )
        canonical_cases.append(canonical)

    name = dataset_name.strip() or Path(source_label).stem
    provenance = {
        str(key): value
        for key, value in dict(source_provenance or {}).items()
        if str(key).strip() and value is not None and str(value).strip()
    }
    manifest_path = run_dir / "benchmark_dataset.json"
    manifest_payload = {
        "kind": "benchmark_dataset",
        "schema_version": BENCHMARK_DATASET_SCHEMA_VERSION,
        "dataset_name": name,
        "profile": analysis["profile"],
        "source_kind": source_kind,
        "source_label": source_label,
        "source_provenance": provenance,
        "case_count": len(canonical_cases),
        "rejected_case_count": analysis["rejected_case_count"],
        "cases": canonical_cases,
    }
    _write_json(manifest_path, manifest_payload)
    table_path = run_dir / "benchmark_cases.csv"
    with table_path.open("w", newline="") as handle:
        fields = (
            "case_id",
            "target_id",
            "split",
            "receptor_path",
            "reference_ligand_path",
            "complex_path",
            "sequence_path",
            "ligand_smiles",
            "ligand_record_count",
            "coordinate_dimension",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(canonical_cases)
    report_path = reports_root / "import_report.json"
    _write_json(report_path, analysis)
    if archive_data is not None:
        source_archive = source_root / Path(archive_filename).name
        source_archive.write_bytes(archive_data)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                source_archive,
                "source_benchmark_dataset",
                role="original_upload",
                label=archive_filename,
            )
        )
    if manifest_data is not None:
        source_manifest = source_root / (Path(manifest_filename).name or "manifest.json")
        source_manifest.write_bytes(manifest_data)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                source_manifest,
                "source_benchmark_manifest",
                role="original_manifest",
            )
        )
    if provenance:
        provenance_path = source_root / "provenance.json"
        _write_json(provenance_path, provenance)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                provenance_path,
                "benchmark_source_provenance",
                role="source_citation",
                label=str(provenance.get("citation") or provenance.get("url") or name),
            )
        )
    artifacts.extend(
        (
            ArtifactRef.from_path(
                run_dir,
                manifest_path,
                "benchmark_dataset",
                role="canonical_manifest",
                label=name,
                metadata={
                    "case_count": len(canonical_cases),
                    "profile": analysis["profile"],
                },
            ),
            ArtifactRef.from_path(
                run_dir,
                table_path,
                "benchmark_cases",
                role="case_table",
                label=f"{name} cases",
                metadata={"case_count": len(canonical_cases)},
            ),
            ArtifactRef.from_path(
                run_dir, report_path, "benchmark_import_report", role="validation"
            ),
        )
    )
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "benchmark_dataset_import",
        "workflow": "benchmark_dataset_import",
        "operation": "benchmark_dataset_import",
        "status": "completed",
        "tool": "mn-ligand benchmark importer",
        "dataset_name": name,
        "benchmark_profile": analysis["profile"],
        "case_count": len(canonical_cases),
        "rejected_case_count": analysis["rejected_case_count"],
        "source_kind": source_kind,
        "source_provenance": provenance,
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "source_kind": source_kind,
            "source_label": source_label,
            "requested_profile": requested_profile,
            "manifest_filename": manifest_filename,
        },
    )
    _write_json(
        run_dir / "result.json",
        {
            "success": True,
            "case_count": len(canonical_cases),
            "rejected_case_count": analysis["rejected_case_count"],
            "profile": analysis["profile"],
        },
    )
    write_artifact_manifest(run_dir, artifacts)
    return JobRecord.load(run_dir, task_group=BENCHMARK_DATASET_TASK_GROUP)


def list_benchmark_dataset_jobs() -> list[JobRecord]:
    root = runs_root() / BENCHMARK_DATASET_TASK_GROUP
    if not root.is_dir():
        return []
    return sorted(
        (
            JobRecord.load(path, task_group=BENCHMARK_DATASET_TASK_GROUP)
            for path in root.iterdir()
            if path.is_dir()
        ),
        key=lambda job: (job.created_at, job.run_dir.stat().st_mtime),
        reverse=True,
    )


def load_benchmark_dataset(job: JobRecord) -> dict[str, Any]:
    path = job.run_dir / "benchmark_dataset.json"
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark dataset manifest missing for {job.run_id}")
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version") or 0) != BENCHMARK_DATASET_SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark dataset schema")
    return payload


@dataclass(frozen=True)
class BenchmarkCase:
    dataset_job: JobRecord
    payload: dict[str, Any]
    receptor_artifact: ArtifactRef
    ligand_artifact: ArtifactRef
    complex_artifact: ArtifactRef | None = None

    @property
    def case_id(self) -> str:
        return str(self.payload["case_id"])

    @property
    def receptor_path(self) -> Path:
        return self.dataset_job.run_dir / str(self.payload["receptor_path"])

    @property
    def ligand_path(self) -> Path:
        return self.dataset_job.run_dir / str(self.payload["reference_ligand_path"])


def create_benchmark_bound_chain_target(
    case: BenchmarkCase,
    selection: dict[str, Any],
    *,
    campaign_id: str,
) -> JobRecord:
    """Materialize one immutable ligand-bound receptor chain for a campaign case."""
    chain = str(selection.get("chain") or "").strip()
    if not chain:
        raise ValueError("A ligand-bound protein chain is required")
    source_lines = case.receptor_path.read_text(errors="replace").splitlines()
    retained = []
    atom_count = 0
    for line in source_lines:
        if line.startswith(("ATOM  ", "HETATM")):
            if (line[21].strip() or "A") != chain:
                continue
            atom_count += 1
            retained.append(line)
        elif line.startswith("TER"):
            if len(line) > 21 and (line[21].strip() or "A") == chain:
                retained.append(line)
        elif line.startswith(("HEADER", "TITLE ", "COMPND", "SOURCE", "REMARK")):
            retained.append(line)
    if not atom_count:
        raise ValueError(f"Selected chain {chain} contains no receptor atoms")
    retained.append("END")

    run_id = str(uuid4())
    run_dir = runs_root() / BENCHMARK_TARGET_TASK_GROUP / run_id
    input_dir = run_dir / "input"
    output_dir = run_dir / "artifacts"
    input_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir()
    source_copy = input_dir / "source_receptor.pdb"
    source_copy.write_bytes(case.receptor_path.read_bytes())
    receptor = output_dir / "ligand_bound_chain.pdb"
    receptor.write_text("\n".join(retained) + "\n")
    evidence = {
        key: value
        for key, value in selection.items()
        if key != "sequence"
    }
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "benchmark_bound_chain_selection",
        "workflow": "benchmark_bound_chain_selection",
        "operation": "benchmark_bound_chain_selection",
        "status": "completed",
        "tool": "mn-ligand bound-chain selector",
        "benchmark_dataset_run_id": case.dataset_job.run_id,
        "benchmark_campaign_id": campaign_id,
        "benchmark_case_id": case.case_id,
        "selected_chain": chain,
        "selection_evidence": evidence,
        "source_receptor_artifact_id": case.receptor_artifact.artifact_id,
        "reference_ligand_artifact_id": case.ligand_artifact.artifact_id,
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "source_receptor_run_id": case.receptor_artifact.run_id,
            "source_receptor_artifact_id": case.receptor_artifact.artifact_id,
            "reference_ligand_artifact_id": case.ligand_artifact.artifact_id,
            "selection_evidence": evidence,
        },
    )
    _write_json(
        run_dir / "result.json",
        {
            "success": True,
            "selected_chain": chain,
            "receptor_atom_count": atom_count,
            "selection_evidence": evidence,
        },
    )
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(
                run_dir,
                receptor,
                "benchmark_bound_receptor",
                role="ligand_bound_chain",
                label=f"{case.case_id} chain {chain}",
                metadata={
                    "case_id": case.case_id,
                    "chain": chain,
                    "source_artifact_id": case.receptor_artifact.artifact_id,
                    **evidence,
                },
            )
        ],
    )
    return JobRecord.load(run_dir, task_group=BENCHMARK_TARGET_TASK_GROUP)


def benchmark_cases(job: JobRecord) -> list[BenchmarkCase]:
    manifest = load_benchmark_dataset(job)
    artifacts = job.artifact_manifest.artifacts if job.artifact_manifest else ()
    by_case_type = {
        (str(item.metadata.get("case_id") or ""), item.artifact_type): item
        for item in artifacts
    }
    result: list[BenchmarkCase] = []
    for payload in manifest.get("cases") or []:
        case_id = str(payload.get("case_id") or "")
        receptor = by_case_type.get((case_id, "benchmark_receptor"))
        ligand = by_case_type.get((case_id, "benchmark_reference_ligand"))
        if receptor is None or ligand is None:
            continue
        result.append(
            BenchmarkCase(
                dataset_job=job,
                payload=dict(payload),
                receptor_artifact=receptor,
                ligand_artifact=ligand,
                complex_artifact=by_case_type.get(
                    (case_id, "benchmark_reference_complex")
                ),
            )
        )
    return result
