from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    docker_is_rootless,
    registered_tool,
    write_registered_command_record,
)
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, iter_job_records, short_job_code
from mn_ligand.core.pockets import PocketRecord, PocketSet, ResidueRef
from mn_ligand.runtime import PROJECT_DIR, reference_root, resolve_run_dir, runs_root
from mn_ligand.workflows.bound_ligand_md import extract_ligand_pdb, parse_bound_ligands
DEFAULT_FPOCKET_IMAGE = "ovolig-fpocket:latest"
DEFAULT_PESTO_IMAGE = "mnprot-pesto-cu128:latest"
DEFAULT_P2RANK_IMAGE = "ovolig-p2rank:latest"
POCKET_TASK_GROUP = "pocket-detection"
BOUND_LIGAND_METHOD = "bound_ligand"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def pesto_checkpoint_path() -> Path:
    return reference_root() / "pesto" / "i_v4_1" / "model_ckpt.pt"


def pesto_readiness() -> dict[str, Any]:
    checkpoint = pesto_checkpoint_path()
    return {
        "ready": checkpoint.is_file(),
        "checkpoint": checkpoint,
        "model_dir": checkpoint.parent.parent,
    }


def _coordinates(path: Path) -> list[tuple[float, float, float]]:
    values: list[tuple[float, float, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            values.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except (ValueError, IndexError):
            fields = line.split()
            try:
                values.append((float(fields[5]), float(fields[6]), float(fields[7])))
            except (ValueError, IndexError):
                continue
    return values


def _pdb_atom(line: str) -> dict[str, Any] | None:
    if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
        return None
    try:
        coordinates = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except ValueError:
        return None
    atom_name = line[12:16].strip()
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if not element:
        element = re.sub(r"[^A-Za-z]", "", atom_name)[:1].upper()
    return {
        "line": line,
        "record": line[:6].strip(),
        "atom_name": atom_name,
        "altloc": line[16].strip(),
        "residue_name": line[17:20].strip() or "UNK",
        "chain_id": line[21].strip() or "_",
        "residue_number": line[22:26].strip(),
        "insertion_code": line[26].strip(),
        "coordinates": coordinates,
        "element": element,
    }


def bound_ligands_from_pdb(path: Path, *, minimum_heavy_atoms: int = 3) -> list[dict[str, Any]]:
    if path.suffix.lower() not in {".pdb", ".ent"}:
        raise ValueError("Bound-ligand pocket selection currently requires an imported PDB complex")
    return [
        dict(ligand)
        for ligand in parse_bound_ligands(path.read_text(errors="replace"))
        if int(ligand.get("heavy_atom_count") or 0) >= minimum_heavy_atoms
    ]


def _prepared_target_source(prepared_target_run_id: str) -> tuple[JobRecord, ArtifactRef, Path]:
    source_job = next(
        (job for job in iter_job_records(runs_root()) if job.run_id == prepared_target_run_id),
        None,
    )
    if source_job is None or source_job.artifact_manifest is None:
        raise FileNotFoundError(f"Prepared target job not found: {prepared_target_run_id}")
    for artifact_type in ("prepared_target", "prepared_receptor"):
        for source_ref in source_job.artifact_manifest.by_type(artifact_type):
            source_path = source_ref.resolve(source_job.run_dir, must_exist=True)
            if source_path is not None:
                return source_job, source_ref, source_path
    raise FileNotFoundError(f"Prepared target artifact not found: {prepared_target_run_id}")


def _source_complex(job: JobRecord) -> tuple[ArtifactRef, Path] | None:
    jobs = iter_job_records(runs_root())
    jobs_by_id = {item.run_id: item for item in jobs}
    pending = [job]
    visited: set[str] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate.run_id in visited:
            continue
        visited.add(candidate.run_id)
        if candidate.artifact_manifest:
            for artifact_type in ("prepared_complex", "imported_target"):
                for artifact in candidate.artifact_manifest.by_type(artifact_type):
                    path = artifact.resolve(candidate.run_dir, must_exist=True)
                    if path is not None and path.suffix.lower() in {".pdb", ".ent"}:
                        return artifact, path
        for parent_id in (
            candidate.metadata.get("import_run_id"),
            candidate.metadata.get("source_structure_run_id"),
            candidate.parent_run_id,
        ):
            parent = jobs_by_id.get(str(parent_id or ""))
            if parent is not None and parent.run_id not in visited:
                pending.append(parent)
    return None


def bound_ligand_candidates(prepared_target_run_id: str) -> list[dict[str, Any]]:
    source_job, _, _ = _prepared_target_source(prepared_target_run_id)
    complex_source = _source_complex(source_job)
    return bound_ligands_from_pdb(complex_source[1]) if complex_source is not None else []


def _residues(path: Path) -> tuple[ResidueRef, ...]:
    unique: dict[tuple[str, str, str, str], ResidueRef] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        key = (
            line[21].strip() or "_",
            line[17:20].strip() or "UNK",
            line[22:26].strip(),
            line[26].strip(),
        )
        unique[key] = ResidueRef(
            chain_id=key[0], residue_name=key[1], residue_number=key[2], insertion_code=key[3]
        )
    def residue_order(item: tuple[str, str, str, str]) -> tuple[str, int, str, str]:
        try:
            number = int(item[2])
        except ValueError:
            number = 10**9
        return item[0], number, item[3], item[1]

    return tuple(unique[key] for key in sorted(unique, key=residue_order))


def _descriptor_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _descriptors(path: Path) -> dict[str, float]:
    descriptors: dict[str, float] = {}
    pattern = re.compile(r"-\s*(?P<label>[^:]+):\s*(?P<value>[-+]?\d+(?:\.\d+)?)")
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            descriptors[_descriptor_key(match.group("label"))] = float(match.group("value"))
    return descriptors


def _center_and_size(
    points: list[tuple[float, float, float]], *, box_padding_angstrom: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not points:
        raise ValueError("Pocket contains no coordinate points")
    axes = list(zip(*points))
    center = tuple((min(axis) + max(axis)) / 2.0 for axis in axes)
    size = tuple(max(4.0, max(axis) - min(axis) + 2.0 * box_padding_angstrom) for axis in axes)
    return center, size


def normalize_fpocket_output(
    fpocket_dir: Path,
    artifact_dir: Path,
    *,
    source_target: dict[str, Any],
    max_pockets: int,
    min_score: float | None,
    box_padding_angstrom: float,
) -> PocketSet:
    pockets_dir = fpocket_dir / "pockets"
    candidates: list[tuple[int, Path, Path]] = []
    for structure_path in pockets_dir.glob("pocket*_atm.pdb"):
        match = re.fullmatch(r"pocket(\d+)_atm\.pdb", structure_path.name)
        if not match:
            continue
        native_rank = int(match.group(1))
        points_path = pockets_dir / f"pocket{native_rank}_vert.pqr"
        if points_path.is_file():
            candidates.append((native_rank, structure_path, points_path))
    candidates.sort(key=lambda item: item[0])

    artifact_dir.mkdir(parents=True, exist_ok=True)
    records: list[PocketRecord] = []
    for native_rank, structure_path, points_path in candidates:
        descriptors = _descriptors(structure_path)
        score = descriptors.get("pocket_score")
        if min_score is not None and (score is None or score < min_score):
            continue
        rank = len(records) + 1
        if rank > max_pockets:
            break
        copied_structure = artifact_dir / f"pocket_{rank:03d}.pdb"
        copied_points = artifact_dir / f"pocket_{rank:03d}_points.pqr"
        shutil.copy2(structure_path, copied_structure)
        shutil.copy2(points_path, copied_points)
        points = _coordinates(points_path) or _coordinates(structure_path)
        center, size = _center_and_size(points, box_padding_angstrom=box_padding_angstrom)
        records.append(
            PocketRecord(
                pocket_id=f"fpocket-{native_rank}",
                rank=rank,
                method="fpocket",
                score=score,
                druggability_score=descriptors.get("drug_score"),
                center_angstrom=center,
                size_angstrom=size,
                residues=_residues(structure_path),
                descriptors=descriptors,
                structure_path=copied_structure.as_posix(),
                points_path=copied_points.as_posix(),
            )
        )
    if not records:
        raise ValueError("fpocket produced no pockets matching the selected score threshold")
    return PocketSet(
        method="fpocket",
        source_target=source_target,
        pockets=tuple(records),
        parameters={
            "max_pockets": max_pockets,
            "min_score": min_score,
            "box_padding_angstrom": box_padding_angstrom,
        },
    )


def _pesto_residue_key(row: dict[str, str]) -> tuple[str, str, str]:
    residue = str(row.get("residue") or "").strip()
    match = re.fullmatch(r"(-?\d+)(.*)", residue)
    if match is None:
        return str(row.get("chain") or "_"), residue, ""
    return str(row.get("chain") or "_"), match.group(1), match.group(2)


def normalize_pesto_output(
    score_csv: Path,
    target_path: Path,
    artifact_dir: Path,
    *,
    source_target: dict[str, Any],
    score_threshold: float,
    cluster_distance_angstrom: float,
    minimum_residues: int,
    max_pockets: int,
) -> PocketSet:
    with score_csv.open(newline="") as handle:
        score_rows = list(csv.DictReader(handle))
    scores = {
        _pesto_residue_key(row): float(row["score"])
        for row in score_rows
        if row.get("score") not in (None, "")
    }
    residue_atoms: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for line in target_path.read_text(errors="replace").splitlines():
        atom = _pdb_atom(line)
        if atom is None or atom["record"] != "ATOM" or atom["altloc"] not in {"", "A"}:
            continue
        key = (atom["chain_id"], atom["residue_number"], atom["insertion_code"])
        residue_atoms.setdefault(key, []).append(atom)
    selected = {
        key: atoms
        for key, atoms in residue_atoms.items()
        if scores.get(key, 0.0) >= score_threshold
    }
    representatives: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for key, atoms in selected.items():
        preferred = next((atom for atom in atoms if atom["atom_name"] == "CA"), None)
        if preferred is not None:
            representatives[key] = preferred["coordinates"]
            continue
        coordinates = [atom["coordinates"] for atom in atoms]
        representatives[key] = tuple(
            sum(point[axis] for point in coordinates) / len(coordinates) for axis in range(3)
        )
    cutoff_squared = cluster_distance_angstrom * cluster_distance_angstrom
    distances = {
        frozenset((left, right)): sum(
            (representatives[left][axis] - representatives[right][axis]) ** 2
            for axis in range(3)
        )
        for index, left in enumerate(sorted(selected))
        for right in sorted(selected)[index + 1 :]
    }
    candidate_clusters = [[key] for key in sorted(selected)]
    while True:
        best_pair: tuple[int, int] | None = None
        best_span = float("inf")
        for left_index, left_cluster in enumerate(candidate_clusters):
            for right_index in range(left_index + 1, len(candidate_clusters)):
                right_cluster = candidate_clusters[right_index]
                span = max(
                    distances[frozenset((left, right))]
                    for left in left_cluster
                    for right in right_cluster
                )
                if span <= cutoff_squared and span < best_span:
                    best_pair = (left_index, right_index)
                    best_span = span
        if best_pair is None:
            break
        left_index, right_index = best_pair
        candidate_clusters[left_index] = sorted(
            candidate_clusters[left_index] + candidate_clusters[right_index]
        )
        candidate_clusters.pop(right_index)
    clusters = [cluster for cluster in candidate_clusters if len(cluster) >= minimum_residues]
    clusters.sort(
        key=lambda cluster: (
            max(scores[key] for key in cluster),
            sum(scores[key] for key in cluster) / len(cluster),
            len(cluster),
        ),
        reverse=True,
    )
    if not clusters:
        raise ValueError(
            "PeSTo found no ligand-interface residue cluster matching the score and size thresholds"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pockets: list[PocketRecord] = []
    for rank, cluster in enumerate(clusters[:max_pockets], start=1):
        cluster_atoms = [atom for key in cluster for atom in selected[key]]
        coordinates = [atom["coordinates"] for atom in cluster_atoms]
        center, size = _center_and_size(coordinates, box_padding_angstrom=0.0)
        structure_path = artifact_dir / f"pocket_{rank:03d}.pdb"
        structure_path.write_text("\n".join([atom["line"] for atom in cluster_atoms] + ["END", ""]))
        points_path = artifact_dir / f"pocket_{rank:03d}_residues.pdb"
        representative_atoms = [
            next((atom for atom in selected[key] if atom["atom_name"] == "CA"), selected[key][0])
            for key in cluster
        ]
        points_path.write_text("\n".join([atom["line"] for atom in representative_atoms] + ["END", ""]))
        cluster_scores = [scores[key] for key in cluster]
        residues = tuple(
            ResidueRef(
                chain_id=key[0],
                residue_name=selected[key][0]["residue_name"],
                residue_number=key[1],
                insertion_code=key[2],
            )
            for key in sorted(cluster)
        )
        pockets.append(
            PocketRecord(
                pocket_id=f"pesto-ligand-{rank}",
                rank=rank,
                method="pesto",
                score=max(cluster_scores),
                center_angstrom=center,
                size_angstrom=size,
                residues=residues,
                descriptors={
                    "maximum_ligand_probability": max(cluster_scores),
                    "mean_ligand_probability": sum(cluster_scores) / len(cluster_scores),
                    "residue_count": float(len(cluster)),
                },
                metadata={"interface": "ligand", "score_threshold": score_threshold},
                structure_path=structure_path.relative_to(artifact_dir.parent.parent).as_posix(),
                points_path=points_path.relative_to(artifact_dir.parent.parent).as_posix(),
            )
        )
    return PocketSet(
        method="pesto",
        source_target=source_target,
        pockets=tuple(pockets),
        parameters={
            "interface": "ligand",
            "score_threshold": score_threshold,
            "cluster_distance_angstrom": cluster_distance_angstrom,
            "minimum_residues": minimum_residues,
            "max_pockets": max_pockets,
        },
    )


def _p2rank_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [
            {
                str(key or "").strip(): str(value or "").strip()
                for key, value in row.items()
            }
            for row in csv.DictReader(handle, skipinitialspace=True)
        ]


def _p2rank_residue_label(value: str) -> tuple[str, str]:
    match = re.fullmatch(r"(-?\d+)(.*)", str(value).strip())
    if match is None:
        return str(value).strip(), ""
    return match.group(1), match.group(2).strip()


def _p2rank_point_rank(line: str) -> int | None:
    if not line.startswith(("ATOM", "HETATM")):
        return None
    try:
        return int(line[22:26].strip())
    except (ValueError, IndexError):
        return None


def normalize_p2rank_output(
    predictions_csv: Path,
    residues_csv: Path,
    points_pdb_gz: Path,
    target_path: Path,
    artifact_dir: Path,
    *,
    source_target: dict[str, Any],
    profile: str,
    max_pockets: int,
    min_probability: float | None,
    box_padding_angstrom: float,
) -> PocketSet:
    """Normalize native P2Rank tables and SAS points into portable pocket artifacts."""
    prediction_rows = _p2rank_csv_rows(predictions_csv)
    residue_rows = _p2rank_csv_rows(residues_csv)
    required_prediction_columns = {
        "rank", "score", "probability", "center_x", "center_y", "center_z"
    }
    if prediction_rows and not required_prediction_columns.issubset(prediction_rows[0]):
        missing = sorted(required_prediction_columns - set(prediction_rows[0]))
        raise ValueError("P2Rank predictions are missing column(s): " + ", ".join(missing))

    residues_by_rank: dict[int, list[ResidueRef]] = {}
    residue_keys_by_rank: dict[int, set[tuple[str, str, str]]] = {}
    for row in residue_rows:
        try:
            rank = int(row.get("pocket") or 0)
        except ValueError:
            continue
        if rank < 1:
            continue
        number, insertion_code = _p2rank_residue_label(row.get("residue_label") or "")
        residue = ResidueRef(
            chain_id=row.get("chain") or "_",
            residue_name=row.get("residue_name") or "UNK",
            residue_number=number,
            insertion_code=insertion_code,
        )
        residues_by_rank.setdefault(rank, []).append(residue)
        residue_keys_by_rank.setdefault(rank, set()).add(
            (residue.chain_id, residue.residue_number, residue.insertion_code)
        )

    with gzip.open(points_pdb_gz, "rt", errors="replace") as handle:
        native_point_lines = [line.rstrip("\n") for line in handle]
    points_by_rank: dict[int, list[str]] = {}
    for line in native_point_lines:
        rank = _p2rank_point_rank(line)
        if rank is not None and rank > 0:
            points_by_rank.setdefault(rank, []).append(line)

    target_lines = target_path.read_text(errors="replace").splitlines()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records: list[PocketRecord] = []
    for row in sorted(prediction_rows, key=lambda item: int(item.get("rank") or 0)):
        native_rank = int(row.get("rank") or 0)
        probability = float(row.get("probability") or 0.0)
        if native_rank < 1 or (
            min_probability is not None and probability < min_probability
        ):
            continue
        rank = len(records) + 1
        if rank > max_pockets:
            break
        point_lines = points_by_rank.get(native_rank, [])
        if not point_lines:
            raise ValueError(f"P2Rank pocket {native_rank} has no assigned SAS points")
        point_path = artifact_dir / f"pocket_{rank:03d}_points.pdb"
        point_path.write_text("\n".join([*point_lines, "END", ""]))
        native_center = (
            float(row["center_x"]),
            float(row["center_y"]),
            float(row["center_z"]),
        )
        point_coordinates = _coordinates(point_path)
        if not point_coordinates:
            raise ValueError(f"P2Rank pocket {native_rank} has unreadable SAS-point coordinates")
        size = tuple(
            max(
                4.0,
                2.0 * max(abs(point[axis] - native_center[axis]) for point in point_coordinates)
                + 2.0 * box_padding_angstrom,
            )
            for axis in range(3)
        )

        residue_keys = residue_keys_by_rank.get(native_rank, set())
        structure_lines = []
        for line in target_lines:
            atom = _pdb_atom(line)
            if atom is None or atom["record"] != "ATOM":
                continue
            key = (atom["chain_id"], atom["residue_number"], atom["insertion_code"])
            if key in residue_keys:
                structure_lines.append(line)
        structure_path = artifact_dir / f"pocket_{rank:03d}.pdb"
        # Prepared targets are currently PDB artifacts. Keep point geometry as a
        # viewable fallback if a future mmCIF target cannot be sliced as PDB text.
        structure_path.write_text(
            "\n".join([*(structure_lines or point_lines), "END", ""])
        )
        relative_root = artifact_dir.parent.parent
        descriptors = {
            "probability": probability,
            "sas_point_count": float(row.get("sas_points") or len(point_lines)),
            "surface_atom_count": float(row.get("surf_atoms") or 0),
        }
        records.append(
            PocketRecord(
                pocket_id=f"p2rank-{native_rank}",
                rank=rank,
                method="p2rank",
                score=float(row.get("score") or 0.0),
                center_angstrom=native_center,
                size_angstrom=size,
                residues=tuple(residues_by_rank.get(native_rank, ())),
                descriptors=descriptors,
                metadata={
                    "native_rank": native_rank,
                    "profile": profile,
                    "probability_calibration": "profile-specific",
                },
                structure_path=structure_path.relative_to(relative_root).as_posix(),
                points_path=point_path.relative_to(relative_root).as_posix(),
            )
        )
    if not records:
        threshold = "" if min_probability is None else f" at probability >= {min_probability:g}"
        raise ValueError(f"P2Rank produced no pockets{threshold}")
    return PocketSet(
        method="p2rank",
        source_target=source_target,
        pockets=tuple(records),
        parameters={
            "profile": profile,
            "max_pockets": max_pockets,
            "min_probability": min_probability,
            "box_padding_angstrom": box_padding_angstrom,
        },
    )


def run_fpocket_native(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    target_path = Path(str(config.get("target_path") or ""))
    if not target_path.is_file():
        raise FileNotFoundError(f"Prepared target not found: {target_path}")
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    local_target = work_dir / f"target{target_path.suffix.lower() or '.pdb'}"
    shutil.copy2(target_path, local_target)
    command = [str(config.get("fpocket_executable") or "fpocket"), "-f", str(local_target)]
    process = subprocess.run(command, cwd=work_dir, capture_output=True, text=True, check=False)
    (output_dir / "fpocket.stdout.log").write_text(process.stdout or "")
    (output_dir / "fpocket.stderr.log").write_text(process.stderr or "")
    if process.returncode != 0:
        raise RuntimeError((process.stderr or process.stdout or "fpocket failed")[-4000:])
    native_dir = work_dir / f"{local_target.stem}_out"
    pocket_set_path = output_dir / "artifacts" / "pockets" / "pocket_set.json"
    pocket_set = normalize_fpocket_output(
        native_dir,
        pocket_set_path.parent,
        source_target=dict(config.get("source_target") or {}),
        max_pockets=int(config.get("max_pockets") or 10),
        min_score=(float(config["min_score"]) if config.get("min_score") is not None else None),
        box_padding_angstrom=float(config.get("box_padding_angstrom") or 4.0),
    )
    # Store portable run-relative paths rather than /output paths.
    relative_records = []
    for record in pocket_set.pockets:
        relative_records.append(
            PocketRecord(
                **{
                    **record.__dict__,
                    "structure_path": Path(record.structure_path).relative_to(output_dir).as_posix(),
                    "points_path": Path(record.points_path).relative_to(output_dir).as_posix(),
                }
            )
        )
    pocket_set = PocketSet(
        method=pocket_set.method,
        source_target=pocket_set.source_target,
        pockets=tuple(relative_records),
        parameters=pocket_set.parameters,
    )
    pocket_set.write(pocket_set_path)
    shutil.rmtree(work_dir)
    return {
        "success": True,
        "method": "fpocket",
        "pocket_count": len(pocket_set.pockets),
        "pocket_set": pocket_set_path.relative_to(output_dir).as_posix(),
        "command": command,
    }


def run_bound_ligand_native(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    target_path = Path(str(config.get("target_path") or ""))
    complex_path = Path(str(config.get("complex_path") or ""))
    ligand_key = str(config.get("bound_ligand_key") or "")
    for label, path in (("prepared target", target_path), ("source complex", complex_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label.title()} not found: {path}")
    if target_path.suffix.lower() not in {".pdb", ".ent"} or complex_path.suffix.lower() not in {".pdb", ".ent"}:
        raise ValueError("Bound-ligand pocket calculation currently requires PDB inputs")

    complex_data = complex_path.read_text(errors="replace")
    candidates = {str(item["key"]): item for item in parse_bound_ligands(complex_data)}
    selector = candidates.get(ligand_key)
    if selector is None:
        raise ValueError(f"Selected bound ligand was not found in source complex: {ligand_key}")
    ligand_data = extract_ligand_pdb(complex_data, ligand_key)
    ligand_atoms = [
        atom
        for line in ligand_data.splitlines()
        if (atom := _pdb_atom(line)) is not None and atom["altloc"] in {"", "A"}
    ]
    heavy_atoms = [atom for atom in ligand_atoms if atom["element"] != "H"]
    if len(heavy_atoms) < 3:
        raise ValueError(f"Selected bound ligand is missing or has fewer than three heavy atoms: {ligand_key}")

    padding = float(config.get("box_padding_angstrom") or 4.0)
    cutoff = float(config.get("lining_cutoff_angstrom") or 5.0)
    center, size = _center_and_size(
        [atom["coordinates"] for atom in heavy_atoms], box_padding_angstrom=padding
    )
    cutoff_squared = cutoff * cutoff
    protein_residues: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for line in target_path.read_text(errors="replace").splitlines():
        atom = _pdb_atom(line)
        if atom is None or atom["record"] != "ATOM" or atom["altloc"] not in {"", "A"}:
            continue
        key = (
            atom["chain_id"], atom["residue_name"], atom["residue_number"], atom["insertion_code"]
        )
        protein_residues.setdefault(key, []).append(atom)
    lining: list[list[dict[str, Any]]] = []
    for atoms in protein_residues.values():
        if any(
            sum((protein_atom["coordinates"][axis] - ligand_atom["coordinates"][axis]) ** 2 for axis in range(3))
            <= cutoff_squared
            for protein_atom in atoms
            for ligand_atom in heavy_atoms
        ):
            lining.append(atoms)

    artifact_dir = output_dir / "artifacts" / "pockets"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    structure_path = artifact_dir / "pocket_001.pdb"
    structure_lines = [atom["line"] for atoms in lining for atom in atoms]
    structure_path.write_text("\n".join(structure_lines + ["END", ""]))
    points_path = artifact_dir / "pocket_001_bound_ligand.pdb"
    points_path.write_text("\n".join([atom["line"] for atom in ligand_atoms] + ["END", ""]))

    selector = dict(selector)
    pocket = PocketRecord(
        pocket_id=f"bound-{ligand_key.replace('|', '-')}",
        rank=1,
        method=BOUND_LIGAND_METHOD,
        center_angstrom=center,
        size_angstrom=size,
        residues=_residues(structure_path),
        descriptors={
            "ligand_atom_count": float(len(ligand_atoms)),
            "ligand_heavy_atom_count": float(len(heavy_atoms)),
            "lining_residue_count": float(len(lining)),
        },
        metadata={"bound_ligand": selector, "lining_cutoff_angstrom": cutoff},
        structure_path=structure_path.relative_to(output_dir).as_posix(),
        points_path=points_path.relative_to(output_dir).as_posix(),
    )
    pocket_set = PocketSet(
        method=BOUND_LIGAND_METHOD,
        source_target=dict(config.get("source_target") or {}),
        source_complex=dict(config.get("source_complex") or {}),
        pockets=(pocket,),
        parameters={
            "bound_ligand_key": ligand_key,
            "box_padding_angstrom": padding,
            "lining_cutoff_angstrom": cutoff,
        },
    )
    pocket_set_path = artifact_dir / "pocket_set.json"
    pocket_set.write(pocket_set_path)
    return {
        "success": True,
        "method": BOUND_LIGAND_METHOD,
        "pocket_count": 1,
        "pocket_set": pocket_set_path.relative_to(output_dir).as_posix(),
        "bound_ligand": selector,
    }


def run_pocket_native(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    method = str(config.get("method") or "fpocket")
    if method == "fpocket":
        return run_fpocket_native(config, output_dir)
    if method == BOUND_LIGAND_METHOD:
        return run_bound_ligand_native(config, output_dir)
    raise ValueError(f"Unsupported pocket-detection method: {method}")


def _docker_is_rootless() -> bool:
    return docker_is_rootless()


def _docker_command(
    image: str, run_dir: Path, source_path: Path, complex_path: Path | None = None
) -> list[str]:
    shm_size = os.getenv("MN_DOCKER_SHM_SIZE", "4g").strip()
    mounts = [
        DockerMount(PROJECT_DIR, "/mn-ligand", read_only=True),
        DockerMount(run_dir, "/output"),
        DockerMount(source_path, f"/input/target/{source_path.name}", read_only=True),
    ]
    if complex_path is not None:
        mounts.append(
            DockerMount(complex_path, f"/input/complex/{complex_path.name}", read_only=True)
        )
    spec = DockerRunSpec(
        tool=registered_tool("fpocket", image=image),
        command=(
            "python",
            "-m",
            "mn_ligand.workflows.pocket_detection",
            "run-native",
            "--input",
            "/output/runner_input.json",
            "--output",
            "/output/native_result.json",
        ),
        mounts=tuple(mounts),
        environment={"PYTHONPATH": "/mn-ligand"},
        gpu_enabled=False,
        shm_size=shm_size,
    )
    return build_docker_command(spec, rootless=_docker_is_rootless())


def _pesto_docker_command(
    image: str,
    run_dir: Path,
    source_path: Path,
    model_dir: Path,
    gpu_device: str,
) -> list[str]:
    shm_size = os.getenv("MN_DOCKER_SHM_SIZE", "4g").strip()
    selected = str(gpu_device).strip().lower().removeprefix("device=")
    gpu_ids = None if selected == "all" else tuple(int(value) for value in selected.split(","))
    spec = DockerRunSpec(
        tool=registered_tool("pesto_ligand_interface", image=image),
        command=(
            "--input",
            "/input/target.pdb",
            "--output-dir",
            "/output/native/pesto",
            "--interface",
            "ligand",
            "--device",
            "cuda",
            "--checkpoint",
            "/models/i_v4_1/model_ckpt.pt",
        ),
        mounts=(
            DockerMount(run_dir, "/output"),
            DockerMount(source_path, "/input/target.pdb", read_only=True),
            DockerMount(model_dir, "/models", read_only=True),
        ),
        gpu_devices=gpu_ids,
        shm_size=shm_size,
    )
    return build_docker_command(spec, rootless=_docker_is_rootless())


def _p2rank_docker_command(
    image: str,
    run_dir: Path,
    source_path: Path,
    *,
    profile: str,
    cpu_threads: int,
) -> list[str]:
    command = [
        "predict",
        "-f",
        f"/input/target/{source_path.name}",
        "-o",
        "/output/native/p2rank",
        "-threads",
        str(cpu_threads),
        "-visualizations",
        "1",
    ]
    if profile == "alphafold":
        command.extend(["-c", "alphafold"])
    spec = DockerRunSpec(
        tool=registered_tool("p2rank", image=image),
        command=tuple(command),
        mounts=(
            DockerMount(run_dir, "/output"),
            DockerMount(source_path, f"/input/target/{source_path.name}", read_only=True),
        ),
        gpu_enabled=False,
        shm_size=os.getenv("MN_DOCKER_SHM_SIZE", "4g").strip(),
    )
    return build_docker_command(spec, rootless=_docker_is_rootless())


def _failed_job(
    run_dir: Path, metadata: dict[str, Any], error: str, returncode: int | None
) -> JobRecord:
    now = _utc_now_iso()
    metadata.update({"status": "failed", "error": error, "updated_at": now, "completed_at": now})
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "result.json", {"success": False, "error": error, "returncode": returncode})
    write_artifact_manifest(run_dir, [])
    return JobRecord.load(run_dir, task_group=POCKET_TASK_GROUP)


def _required_p2rank_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"P2Rank {label} is missing")
    if len(matches) > 1:
        raise ValueError(f"P2Rank produced multiple {label} files")
    return matches[0]


def finalize_pocket_detection_job(
    run_dir: Path,
    *,
    returncode: int,
) -> tuple[JobRecord, dict[str, Any]]:
    """Validate native output and publish normalized pocket artifacts."""
    run_dir = run_dir.resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    runner_input = json.loads((run_dir / "runner_input.json").read_text())
    method = str(metadata.get("tool") or runner_input.get("method") or "fpocket")
    _, source_ref, source_path = _prepared_target_source(
        str(metadata.get("prepared_target_run_id") or metadata.get("parent_run_id") or "")
    )
    native_result = run_dir / "native_result.json"
    native_result.chmod(0o644)
    if method == "p2rank" and returncode == 0:
        native_dir = run_dir / "native" / "p2rank"
        try:
            predictions = _required_p2rank_file(
                native_dir, "*_predictions.csv", "predictions CSV"
            )
            residues = _required_p2rank_file(
                native_dir, "*_residues.csv", "residue-score CSV"
            )
            points = _required_p2rank_file(
                native_dir / "visualizations" / "data",
                "*_points.pdb.gz",
                "SAS-point PDB",
            )
            profile = str(runner_input.get("p2rank_profile") or "default")
            pocket_set = normalize_p2rank_output(
                predictions,
                residues,
                points,
                source_path,
                run_dir / "artifacts" / "pockets",
                source_target=source_ref.to_dict(),
                profile=profile,
                max_pockets=int(runner_input.get("max_pockets", 10)),
                min_probability=(
                    float(runner_input["p2rank_min_probability"])
                    if runner_input.get("p2rank_min_probability") is not None
                    else None
                ),
                box_padding_angstrom=float(
                    runner_input.get("box_padding_angstrom", 4.0)
                ),
            )
            pocket_set_path = run_dir / "artifacts" / "pockets" / "pocket_set.json"
            pocket_set.write(pocket_set_path)
            native_payload = {
                "success": True,
                "method": "p2rank",
                "profile": profile,
                "pocket_count": len(pocket_set.pockets),
                "pocket_set": pocket_set_path.relative_to(run_dir).as_posix(),
                "predictions": predictions.relative_to(run_dir).as_posix(),
                "residue_scores": residues.relative_to(run_dir).as_posix(),
                "sas_points": points.relative_to(run_dir).as_posix(),
            }
            _write_json(native_result, native_payload)
        except Exception as exc:
            native_payload = {"success": False, "error": str(exc)}
            _write_json(native_result, native_payload)
    elif method == "pesto" and returncode == 0:
        score_csv = run_dir / "native" / "pesto" / "pesto_residue_scores.csv"
        scored_pdb = run_dir / "native" / "pesto" / "pesto_scored.pdb"
        try:
            pocket_set = normalize_pesto_output(
                score_csv,
                source_path,
                run_dir / "artifacts" / "pockets",
                source_target=source_ref.to_dict(),
                score_threshold=float(runner_input.get("pesto_score_threshold", 0.5)),
                cluster_distance_angstrom=float(
                    runner_input.get("pesto_cluster_distance_angstrom", 8.0)
                ),
                minimum_residues=int(runner_input.get("pesto_minimum_residues", 3)),
                max_pockets=int(runner_input.get("max_pockets", 10)),
            )
            pocket_set_path = run_dir / "artifacts" / "pockets" / "pocket_set.json"
            pocket_set.write(pocket_set_path)
            native_payload = {
                "success": True,
                "method": "pesto",
                "interface": "ligand",
                "pocket_count": len(pocket_set.pockets),
                "pocket_set": pocket_set_path.relative_to(run_dir).as_posix(),
                "residue_scores": score_csv.relative_to(run_dir).as_posix(),
                "scored_pdb": scored_pdb.relative_to(run_dir).as_posix(),
            }
            _write_json(native_result, native_payload)
        except Exception as exc:
            native_payload = {"success": False, "error": str(exc)}
            _write_json(native_result, native_payload)
    else:
        try:
            native_payload = json.loads(native_result.read_text())
        except (OSError, ValueError):
            native_payload = {}
    if returncode != 0 or native_payload.get("success") is not True:
        stderr = (run_dir / "stderr.log").read_text(errors="replace") if (run_dir / "stderr.log").is_file() else ""
        error = str(native_payload.get("error") or stderr[-4000:] or "Pocket detection failed")
        return _failed_job(run_dir, metadata, error, returncode), native_payload

    pocket_set_path = run_dir / str(native_payload.get("pocket_set") or "")
    try:
        pocket_set_path.resolve().relative_to(run_dir)
        pocket_set = PocketSet.read(pocket_set_path)
    except Exception as exc:
        return _failed_job(run_dir, metadata, f"Invalid pocket-set output: {exc}", returncode), native_payload
    if not pocket_set.pockets:
        return _failed_job(
            run_dir,
            metadata,
            "Invalid pocket-set output: no pockets were published",
            returncode,
        ), native_payload
    artifacts = [ArtifactRef.from_path(run_dir, pocket_set_path, "pocket_set", role="ranked_pockets")]
    if method == "p2rank":
        artifacts.extend(
            [
                ArtifactRef.from_path(
                    run_dir,
                    run_dir / native_payload["predictions"],
                    "pocket_score_table",
                    role="native_p2rank_predictions",
                    metadata={"profile": native_payload.get("profile", "default")},
                ),
                ArtifactRef.from_path(
                    run_dir,
                    run_dir / native_payload["residue_scores"],
                    "residue_score_table",
                    role="p2rank_residue_probabilities",
                    metadata={"profile": native_payload.get("profile", "default")},
                ),
                ArtifactRef.from_path(
                    run_dir,
                    run_dir / native_payload["sas_points"],
                    "native_output",
                    role="p2rank_sas_points",
                ),
            ]
        )
    elif method == "pesto":
        artifacts.extend(
            [
                ArtifactRef.from_path(
                    run_dir,
                    run_dir / native_payload["residue_scores"],
                    "residue_score_table",
                    role="ligand_interface_probabilities",
                    metadata={"interface": "ligand"},
                ),
                ArtifactRef.from_path(
                    run_dir,
                    run_dir / native_payload["scored_pdb"],
                    "scored_structure",
                    role="ligand_interface_probabilities",
                    metadata={"interface": "ligand", "score_field": "temperature_factor"},
                ),
            ]
        )
    for pocket in pocket_set.pockets:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                run_dir / pocket.structure_path,
                "pocket",
                role=pocket.pocket_id,
                metadata={
                    "rank": pocket.rank,
                    "score": pocket.score,
                    "method": pocket.method,
                    "descriptors": pocket.descriptors,
                    "center_angstrom": list(pocket.center_angstrom),
                    "size_angstrom": list(pocket.size_angstrom),
                },
            )
        )
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                run_dir / pocket.points_path,
                "pocket_points",
                role=pocket.pocket_id,
                metadata={"rank": pocket.rank, "method": pocket.method},
            )
        )
    write_artifact_manifest(run_dir, artifacts)
    result = {
        "success": True,
        "method": pocket_set.method,
        "pocket_count": len(pocket_set.pockets),
        "pocket_set": pocket_set_path.relative_to(run_dir).as_posix(),
        "top_pocket": pocket_set.pockets[0].to_dict(),
    }
    _write_json(run_dir / "result.json", result)
    completed_at = _utc_now_iso()
    metadata.update({"status": "completed", "updated_at": completed_at, "completed_at": completed_at})
    _write_json(run_dir / "metadata.json", metadata)
    return JobRecord.load(run_dir, task_group=POCKET_TASK_GROUP), native_payload


def run_pocket_detection_job(
    prepared_target_run_id: str,
    *,
    image: str = "",
    max_pockets: int = 10,
    min_score: float | None = None,
    box_padding_angstrom: float = 4.0,
    method: str = "fpocket",
    bound_ligand_key: str = "",
    lining_cutoff_angstrom: float = 5.0,
    pesto_score_threshold: float = 0.5,
    pesto_cluster_distance_angstrom: float = 8.0,
    pesto_minimum_residues: int = 3,
    p2rank_profile: str = "default",
    p2rank_min_probability: float | None = None,
    gpu_device: str = "all",
    enqueue_only: bool = False,
) -> tuple[JobRecord, dict[str, Any]]:
    method = str(method).strip().lower()
    if method not in {"fpocket", "p2rank", "pesto", BOUND_LIGAND_METHOD}:
        raise ValueError(f"Unsupported pocket-detection method: {method}")
    default_image = {
        "fpocket": DEFAULT_FPOCKET_IMAGE,
        "p2rank": DEFAULT_P2RANK_IMAGE,
        "pesto": DEFAULT_PESTO_IMAGE,
        BOUND_LIGAND_METHOD: DEFAULT_FPOCKET_IMAGE,
    }[method]
    image = str(image).strip() or default_image
    if max_pockets < 1:
        raise ValueError("max_pockets must be positive")
    if box_padding_angstrom < 0:
        raise ValueError("box_padding_angstrom cannot be negative")
    if lining_cutoff_angstrom <= 0:
        raise ValueError("lining_cutoff_angstrom must be positive")
    if not 0.0 <= pesto_score_threshold <= 1.0:
        raise ValueError("PeSTo score threshold must be between 0 and 1")
    if pesto_cluster_distance_angstrom <= 0 or pesto_minimum_residues < 1:
        raise ValueError("PeSTo clustering parameters must be positive")
    p2rank_profile = str(p2rank_profile).strip().lower()
    if p2rank_profile not in {"default", "alphafold"}:
        raise ValueError(f"Unsupported P2Rank profile: {p2rank_profile}")
    if p2rank_min_probability is not None and not 0.0 <= p2rank_min_probability <= 1.0:
        raise ValueError("P2Rank probability threshold must be between 0 and 1")
    gpu_device = str(gpu_device or "all").strip().lower().removeprefix("gpu ")
    if gpu_device not in {"all", "0", "1"}:
        raise ValueError(f"Unsupported GPU device: {gpu_device}")
    source_job, source_ref, source_path = _prepared_target_source(prepared_target_run_id)
    complex_ref: ArtifactRef | None = None
    complex_path: Path | None = None
    if method == BOUND_LIGAND_METHOD:
        if not bound_ligand_key.strip():
            raise ValueError("A bound ligand must be selected")
        complex_source = _source_complex(source_job)
        if complex_source is None:
            raise ValueError("Prepared target does not provide a source protein-ligand complex")
        complex_ref, complex_path = complex_source
        available_keys = {str(item["key"]) for item in bound_ligands_from_pdb(complex_path)}
        if bound_ligand_key not in available_keys:
            raise ValueError(f"Bound ligand is unavailable in the imported complex: {bound_ligand_key}")

    run_id = str(uuid4())
    run_dir = runs_root() / POCKET_TASK_GROUP / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "pocket_detection",
        "workflow": "pocket_detection",
        "status": "queued" if enqueue_only else "running",
        "tool": method,
        "source": method,
        "parent_run_id": source_job.run_id,
        "prepared_target_run_id": source_job.run_id,
        "pdb_id": source_job.metadata.get("pdb_id") or "",
        "docker_image": image,
        "use_gpu": method == "pesto",
        "gpu_device": gpu_device if method == "pesto" else "",
        "created_at": now,
        "updated_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    parameters = {
        "max_pockets": int(max_pockets),
        "min_score": min_score,
        "box_padding_angstrom": float(box_padding_angstrom),
        "bound_ligand_key": bound_ligand_key if method == BOUND_LIGAND_METHOD else "",
        "lining_cutoff_angstrom": float(lining_cutoff_angstrom),
        "pesto_score_threshold": float(pesto_score_threshold),
        "pesto_cluster_distance_angstrom": float(pesto_cluster_distance_angstrom),
        "pesto_minimum_residues": int(pesto_minimum_residues),
        "p2rank_profile": p2rank_profile,
        "p2rank_min_probability": p2rank_min_probability,
        "gpu_device": gpu_device if method == "pesto" else "",
    }
    input_artifacts = {"prepared_target": source_ref.to_dict()}
    if complex_ref is not None:
        input_artifacts["source_complex"] = complex_ref.to_dict()
    _write_json(
        run_dir / "input.json",
        {
            "source_task_group": source_job.task_group,
            "input_artifact": source_ref.to_dict(),
            "input_artifacts": input_artifacts,
            "method": method,
            "parameters": parameters,
        },
    )
    _write_json(
        run_dir / "runner_input.json",
        {
            "method": method,
            "target_path": f"/input/target/{source_path.name}",
            "source_target": source_ref.to_dict(),
            "complex_path": f"/input/complex/{complex_path.name}" if complex_path is not None else "",
            "source_complex": complex_ref.to_dict() if complex_ref is not None else {},
            **parameters,
        },
    )
    native_result = run_dir / "native_result.json"
    _write_json(native_result, {})
    native_result.chmod(0o666)
    pesto_model_dir = pesto_checkpoint_path().parent.parent
    if method == "pesto" and not pesto_checkpoint_path().is_file():
        return _failed_job(
            run_dir,
            metadata,
            f"PeSTo checkpoint is missing: {pesto_checkpoint_path()}",
            None,
        ), {"success": False, "error": "PeSTo checkpoint is missing"}
    if method == "pesto":
        command = _pesto_docker_command(
            image, run_dir, source_path, pesto_model_dir, gpu_device
        )
    elif method == "p2rank":
        command = _p2rank_docker_command(
            image,
            run_dir,
            source_path,
            profile=p2rank_profile,
            cpu_threads=registered_tool("p2rank", image=image).resources.cpu_threads,
        )
    else:
        command = _docker_command(image, run_dir, source_path, complex_path)
    metadata["command"] = command
    _write_json(run_dir / "metadata.json", metadata)
    selected_gpu_ids = (
        ()
        if method != "pesto" or str(gpu_device).lower() == "all"
        else tuple(int(value) for value in str(gpu_device).removeprefix("device=").split(","))
    )
    write_registered_command_record(
        run_dir,
        tool_id=(
            "pesto_ligand_interface"
            if method == "pesto"
            else "p2rank" if method == "p2rank" else "fpocket"
        ),
        commands=(command,),
        image=image,
        selected_gpu_ids=selected_gpu_ids,
    )
    if enqueue_only:
        manifest = registered_tool(
            "pesto_ligand_interface"
            if method == "pesto"
            else "p2rank" if method == "p2rank" else "fpocket",
            image=image,
        )
        resources = manifest.resources.to_dict()
        if selected_gpu_ids:
            resources["gpu_ids"] = list(selected_gpu_ids)
        metadata.update(
            {
                "status": "queued",
                "queued_at": now,
                "queued_command": command,
                "gpu_queued": method == "pesto",
                "resources": resources,
                "worker_finalizer": "pocket_detection",
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        write_artifact_manifest(run_dir, [])
        return JobRecord.load(run_dir, task_group=POCKET_TASK_GROUP), {
            "success": True,
            "queued": True,
            "command": command,
        }
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
    except Exception as exc:
        native_result.chmod(0o644)
        error = f"Could not start pocket-detection container: {exc}"
        (run_dir / "stdout.log").write_text("")
        (run_dir / "stderr.log").write_text(error + "\n")
        return _failed_job(run_dir, metadata, error, None), {"success": False, "error": error}
    (run_dir / "stdout.log").write_text(process.stdout or "")
    (run_dir / "stderr.log").write_text(process.stderr or "")
    return finalize_pocket_detection_job(run_dir, returncode=int(process.returncode))


def queue_pocket_detection_job(
    prepared_target_run_id: str,
    **parameters: Any,
) -> JobRecord:
    """Create a queued pocket job without running scientific work in the caller."""
    job, _ = run_pocket_detection_job(
        prepared_target_run_id,
        enqueue_only=True,
        **parameters,
    )
    return job


def load_pocket_detection_job(run_id: str) -> JobRecord:
    run_dir = resolve_run_dir(POCKET_TASK_GROUP, run_id)
    if run_dir is None:
        raise FileNotFoundError(f"Pocket-detection job not found: {run_id}")
    return JobRecord.load(run_dir, task_group=POCKET_TASK_GROUP)


def list_pocket_detection_jobs() -> list[JobRecord]:
    root = runs_root() / POCKET_TASK_GROUP
    if not root.is_dir():
        return []
    return sorted(
        (JobRecord.load(path, task_group=POCKET_TASK_GROUP) for path in root.iterdir() if path.is_dir()),
        key=lambda job: (job.created_at, job.run_dir.stat().st_mtime),
        reverse=True,
    )


def _main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    native = subparsers.add_parser("run-native")
    native.add_argument("--input", required=True)
    native.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run-native":
        output_path = Path(args.output)
        try:
            payload = run_pocket_native(json.loads(Path(args.input).read_text()), output_path.parent)
        except Exception as exc:
            payload = {"success": False, "error": str(exc)}
        _write_json(output_path, payload)
        if not payload.get("success"):
            raise SystemExit(1)


if __name__ == "__main__":
    _main()
