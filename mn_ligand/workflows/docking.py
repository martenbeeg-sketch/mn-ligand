from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import statistics
import subprocess
from typing import Any
from uuid import uuid4

from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolDescriptors

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    registered_tool,
    write_registered_command_record,
)
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.runtime import (
    DEFAULT_UNIDOCK_PRO_MAX_COMPOUNDS,
    adaptive_cpu_workers,
    app_home,
    cpu_process_limit,
    runs_root,
    unidock_pro_max_compounds,
    vina_compound_timeout_minutes,
)


UNIDOCK_PRO_MODES = ("classic", "hybrid", "ligand_based")
UNIDOCK_PRO_SEARCH_MODES = ("fast", "balance", "detail")
UNIDOCK_PRO_MAX_COMPOUNDS = DEFAULT_UNIDOCK_PRO_MAX_COMPOUNDS
UNIDOCK_PRO_MAX_TORSIONS = 48
DEFAULT_DOCKING_IMAGE = "avgu-docking-suite-cuda:latest"

_RDKIT_3D_SCRIPT = r"""from __future__ import annotations

import argparse
import math

from rdkit import Chem
from rdkit.Chem import AllChem


parser = argparse.ArgumentParser()
parser.add_argument("--smiles", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

molecule = Chem.MolFromSmiles(args.smiles, sanitize=True)
if molecule is None:
    raise SystemExit("RDKit could not sanitize the input SMILES")
molecule = Chem.AddHs(molecule)
parameters = AllChem.ETKDGv3()
parameters.randomSeed = 0x4D4E
parameters.useRandomCoords = False
status = AllChem.EmbedMolecule(molecule, parameters)
if status != 0:
    parameters.useRandomCoords = True
    status = AllChem.EmbedMolecule(molecule, parameters)
if status != 0 or molecule.GetNumConformers() != 1:
    raise SystemExit("RDKit ETKDGv3 could not generate a 3D conformer")
try:
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=300)
    else:
        AllChem.UFFOptimizeMolecule(molecule, maxIters=300)
except (RuntimeError, ValueError):
    pass
conformer = molecule.GetConformer()
if any(
    not math.isfinite(value)
    for atom_index in range(molecule.GetNumAtoms())
    for value in tuple(conformer.GetAtomPosition(atom_index))
):
    raise SystemExit("RDKit generated non-finite 3D coordinates")
molecule.SetProp("_Name", args.name)
writer = Chem.SDWriter(args.output)
writer.write(molecule)
writer.close()
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _safe_compound_id(value: str, index: int) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-._")
    return (normalized or f"compound_{index:07d}")[:100]


def _smiles_records(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    records: list[tuple[str, str]] = []
    if suffix == ".csv":
        with path.open(newline="", errors="replace") as handle:
            reader = csv.DictReader(handle)
            fields = {str(field).strip().lower(): str(field) for field in (reader.fieldnames or ())}
            smiles_field = next(
                (fields[key] for key in ("smiles", "canonical_smiles", "isomeric_smiles") if key in fields),
                "",
            )
            id_field = next(
                (fields[key] for key in ("compound_id", "id", "name", "zincid") if key in fields),
                "",
            )
            if not smiles_field:
                raise ValueError(f"Compound CSV has no SMILES column: {path.name}")
            for index, row in enumerate(reader, start=1):
                smiles = str(row.get(smiles_field) or "").strip()
                if smiles:
                    records.append((str(row.get(id_field) or "") if id_field else "", smiles))
    elif suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        for index, molecule in enumerate(supplier, start=1):
            if molecule is None:
                continue
            name = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
            records.append((name, Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True)))
    else:
        for line in path.read_text(errors="replace").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            parts = value.replace(",", " ", 1).split()
            if parts and parts[0].lower() in {"smiles", "canonical_smiles", "isomeric_smiles"}:
                continue
            if parts:
                records.append((parts[1] if len(parts) > 1 else "", parts[0]))
    return records


def load_compound_records(paths: Sequence[Path], *, maximum: int = 0) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for path in paths:
        for source_id, smiles in _smiles_records(path):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            base_id = _safe_compound_id(source_id, len(records) + 1)
            compound_id = base_id
            suffix = 2
            while compound_id in used_ids:
                compound_id = f"{base_id}-{suffix}"
                suffix += 1
            used_ids.add(compound_id)
            records.append({"compound_id": compound_id, "smiles": canonical})
            if maximum > 0 and len(records) >= maximum:
                return records
    if not records:
        raise ValueError("Selected compound datasets contain no valid molecules")
    return records


def gnina_pose_records(path: Path) -> list[dict[str, Any]]:
    """Return every GNINA model and its three native scores.

    GNINA normally orders output models by CNN pose score.  Parsing every MODEL
    block keeps that ordering separate from the empirical/Vina-like score and
    allows either criterion to select a pose without rerunning docking.
    """
    text = path.read_text(errors="replace")
    model_pattern = re.compile(
        r"(?ms)^MODEL\s+(\d+)\s*$.*?^ENDMDL\s*$"
    )
    vina_pattern = re.compile(r"REMARK VINA RESULT:\s*(-?\d+(?:\.\d+)?)")
    minimized_pattern = re.compile(
        r"REMARK\s+minimizedAffinity\s+(-?\d+(?:\.\d+)?)"
    )
    cnn_pattern = re.compile(r"REMARK\s+CNNscore\s+(-?\d+(?:\.\d+)?)")
    affinity_pattern = re.compile(
        r"REMARK\s+CNNaffinity\s+(-?\d+(?:\.\d+)?)"
    )
    matches = list(model_pattern.finditer(text))
    blocks = [
        (int(match.group(1)), match.group(0))
        for match in matches
    ] or [(1, text)]
    records: list[dict[str, Any]] = []
    for pose_index, block in blocks:
        vina = vina_pattern.search(block)
        minimized = minimized_pattern.search(block)
        cnn = cnn_pattern.search(block)
        affinity = affinity_pattern.search(block)
        empirical = minimized or vina
        records.append(
            {
                "pose_index": pose_index,
                "empirical_score_kcal_mol": (
                    float(empirical.group(1)) if empirical else None
                ),
                "cnn_score": float(cnn.group(1)) if cnn else None,
                "cnn_affinity": (
                    float(affinity.group(1)) if affinity else None
                ),
            }
        )
    return records


def select_gnina_pose(
    path: Path,
    criterion: str = "cnn_score",
) -> dict[str, Any] | None:
    records = gnina_pose_records(path)
    if not records:
        return None
    if criterion == "empirical_score":
        available = [
            row
            for row in records
            if row.get("empirical_score_kcal_mol") is not None
        ]
        return min(
            available or records,
            key=lambda row: float(
                row.get("empirical_score_kcal_mol")
                if row.get("empirical_score_kcal_mol") is not None
                else math.inf
            ),
        )
    available = [
        row for row in records if row.get("cnn_score") is not None
    ]
    return max(
        available or records,
        key=lambda row: float(
            row.get("cnn_score")
            if row.get("cnn_score") is not None
            else -math.inf
        ),
    )


def _score_rows(results_dir: Path, engine: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vina_pattern = re.compile(r"REMARK VINA RESULT:\s*(-?\d+(?:\.\d+)?)")
    minimized_pattern = re.compile(r"REMARK\s+minimizedAffinity\s+(-?\d+(?:\.\d+)?)")
    cnn_pattern = re.compile(r"REMARK\s+CNNscore\s+(-?\d+(?:\.\d+)?)")
    affinity_pattern = re.compile(r"REMARK\s+CNNaffinity\s+(-?\d+(?:\.\d+)?)")
    for path in sorted(results_dir.glob("*_out.pdbqt")):
        text = path.read_text(errors="replace")
        if not text.strip() or not re.search(r"^(?:ATOM|HETATM)", text, re.MULTILINE):
            continue
        if str(engine).strip().lower() == "gnina":
            cnn_pose = select_gnina_pose(path, "cnn_score") or {}
            if not cnn_pose:
                continue
            empirical_pose = (
                select_gnina_pose(path, "empirical_score") or cnn_pose
            )
            rows.append(
                {
                    "compound_id": path.name.removesuffix("_out.pdbqt"),
                    "engine": engine,
                    # Backward-compatible aliases describe the CNN-ranked pose.
                    "best_score_kcal_mol": cnn_pose.get(
                        "empirical_score_kcal_mol"
                    ),
                    "cnn_score": cnn_pose.get("cnn_score"),
                    "cnn_affinity": cnn_pose.get("cnn_affinity"),
                    "pose_file": path.name,
                    "pose_index": cnn_pose.get("pose_index", 1),
                    "pose_selection_criterion": "cnn_score",
                    "cnn_ranked_pose_index": cnn_pose.get("pose_index", 1),
                    "cnn_ranked_empirical_score_kcal_mol": cnn_pose.get(
                        "empirical_score_kcal_mol"
                    ),
                    "cnn_ranked_cnn_score": cnn_pose.get("cnn_score"),
                    "cnn_ranked_cnn_affinity": cnn_pose.get("cnn_affinity"),
                    "empirical_ranked_pose_index": empirical_pose.get(
                        "pose_index", 1
                    ),
                    "empirical_ranked_score_kcal_mol": empirical_pose.get(
                        "empirical_score_kcal_mol"
                    ),
                    "empirical_ranked_cnn_score": empirical_pose.get(
                        "cnn_score"
                    ),
                    "empirical_ranked_cnn_affinity": empirical_pose.get(
                        "cnn_affinity"
                    ),
                }
            )
            continue
        vina = vina_pattern.search(text)
        minimized = minimized_pattern.search(text)
        cnn = cnn_pattern.search(text)
        affinity = affinity_pattern.search(text)
        rows.append(
            {
                "compound_id": path.name.removesuffix("_out.pdbqt"),
                "engine": engine,
                "best_score_kcal_mol": float((vina or minimized).group(1)) if vina or minimized else None,
                "cnn_score": float(cnn.group(1)) if cnn else None,
                "cnn_affinity": float(affinity.group(1)) if affinity else None,
                "pose_file": path.name,
            }
        )
    return rows


def _replicate_score_rows(
    run_dir: Path,
    *,
    engine: str,
    replicates: int,
    seed_start: int,
) -> list[dict[str, Any]]:
    results_dir = run_dir / "results"
    rows: list[dict[str, Any]] = []
    replicate_dirs = sorted(
        path for path in results_dir.glob("replicate_*") if path.is_dir()
    )
    if not replicate_dirs:
        replicate_dirs = [results_dir]
    for fallback, replicate_dir in enumerate(replicate_dirs, start=1):
        match = re.fullmatch(r"replicate_(\d+)", replicate_dir.name)
        replicate = int(match.group(1)) if match else fallback
        for row in _score_rows(replicate_dir, engine):
            pose_name = str(row.get("pose_file") or "")
            row.update(
                {
                    "replicate": replicate,
                    "seed": seed_start + replicate - 1,
                    "pose_file": (
                        (replicate_dir / pose_name).relative_to(run_dir).as_posix()
                        if pose_name else ""
                    ),
                }
            )
            rows.append(row)
    return rows


def _top_pose_molecule(
    run_dir: Path,
    pose_file: object,
    pose_index: object = 1,
) -> Chem.Mol | None:
    relative = str(pose_file or "").strip()
    if not relative:
        return None
    path = run_dir / Path(relative).with_suffix(".sdf")
    if not path.is_file():
        return None
    try:
        molecules = [
            item
            for item in Chem.SDMolSupplier(
                str(path), removeHs=False, sanitize=False
            )
            if item is not None
        ]
        selected_index = max(0, int(pose_index or 1) - 1)
        molecule = (
            molecules[selected_index]
            if selected_index < len(molecules)
            else molecules[0]
            if molecules
            else None
        )
        return (
            Chem.RemoveHs(molecule, sanitize=False)
            if molecule is not None
            else None
        )
    except (OSError, ValueError):
        return None


def docking_pose_diagnostics(
    run_dir: Path,
    score_rows: list[dict[str, Any]],
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Measure fixed-frame pose placement and replicate RMSD.

    RMSD uses symmetry-aware atom mappings but never superposes the ligands:
    every pose remains in the shared receptor coordinate frame.
    """
    import numpy as np

    center_array = np.asarray(center, dtype=float)
    half_size = np.asarray(size, dtype=float) / 2.0
    enriched: list[dict[str, Any]] = []
    molecules: dict[tuple[str, int], Chem.Mol] = {}
    for source in score_rows:
        row = dict(source)
        compound_id = str(row.get("compound_id") or "")
        replicate = int(row.get("replicate") or 1)
        molecule = _top_pose_molecule(
            run_dir,
            row.get("pose_file"),
            row.get("pose_index", 1),
        )
        if molecule is not None and molecule.GetNumConformers():
            coordinates = np.asarray(
                molecule.GetConformer().GetPositions(), dtype=float
            )
            heavy_indices = [
                atom.GetIdx()
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() > 1
            ]
            heavy = coordinates[heavy_indices] if heavy_indices else coordinates
            if len(heavy):
                centroid = heavy.mean(axis=0)
                outside = np.any(
                    (heavy < center_array - half_size)
                    | (heavy > center_array + half_size),
                    axis=1,
                )
                row.update(
                    {
                        "pose_centroid_x": float(centroid[0]),
                        "pose_centroid_y": float(centroid[1]),
                        "pose_centroid_z": float(centroid[2]),
                        "box_center_distance_angstrom": float(
                            np.linalg.norm(centroid - center_array)
                        ),
                        "outside_box_atom_count": int(outside.sum()),
                    }
                )
            molecules[(compound_id, replicate)] = molecule
        enriched.append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for compound_id in sorted({key[0] for key in molecules}):
        selected = sorted(
            (
                (replicate, molecule)
                for (candidate, replicate), molecule in molecules.items()
                if candidate == compound_id
            ),
            key=lambda item: item[0],
        )
        pairwise: list[float] = []
        pair_labels: list[str] = []
        for left_index, (left_replicate, left) in enumerate(selected):
            for right_replicate, right in selected[left_index + 1 :]:
                if left.GetNumAtoms() != right.GetNumAtoms():
                    continue
                try:
                    rmsd = float(
                        rdMolAlign.CalcRMS(
                            left,
                            right,
                            maxMatches=100000,
                        )
                    )
                except (RuntimeError, ValueError):
                    continue
                pairwise.append(rmsd)
                pair_labels.append(
                    f"{left_replicate}-{right_replicate}:{rmsd:.3f}"
                )
        compound_rows = [
            row
            for row in enriched
            if str(row.get("compound_id") or "") == compound_id
        ]
        center_distances = [
            float(row["box_center_distance_angstrom"])
            for row in compound_rows
            if row.get("box_center_distance_angstrom") not in (None, "")
        ]
        summaries[compound_id] = {
            "pose_rmsd_pair_count": len(pairwise),
            "mean_pairwise_pose_rmsd_angstrom": (
                statistics.mean(pairwise) if pairwise else None
            ),
            "max_pairwise_pose_rmsd_angstrom": max(pairwise) if pairwise else None,
            "pairwise_pose_rmsd_angstrom": "; ".join(pair_labels),
            "mean_box_center_distance_angstrom": (
                statistics.mean(center_distances) if center_distances else None
            ),
            "max_box_center_distance_angstrom": (
                max(center_distances) if center_distances else None
            ),
            "replicates_with_atoms_outside_box": sum(
                int(row.get("outside_box_atom_count") or 0) > 0
                for row in compound_rows
            ),
        }
    return enriched, summaries


def _replicate_summary_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    compound_ids = sorted({str(row.get("compound_id") or "") for row in score_rows})
    for compound_id in compound_ids:
        selected = [
            row for row in score_rows
            if str(row.get("compound_id") or "") == compound_id
            and row.get("best_score_kcal_mol") is not None
        ]
        if not selected:
            continue
        selected.sort(key=lambda row: int(row.get("replicate") or 1))
        values = [float(row["best_score_kcal_mol"]) for row in selected]
        median = statistics.median(values)
        representative = min(
            selected,
            key=lambda row: (
                round(abs(float(row["best_score_kcal_mol"]) - median), 12),
                int(row.get("replicate") or 1),
            ),
        )
        cnn_values = [
            float(row["cnn_score"])
            for row in selected if row.get("cnn_score") is not None
        ]
        affinity_values = [
            float(row["cnn_affinity"])
            for row in selected if row.get("cnn_affinity") is not None
        ]
        summaries.append(
            {
                "compound_id": compound_id,
                "replicate_count": len(values),
                "mean_score_kcal_mol": statistics.mean(values),
                "sample_sd_score_kcal_mol": (
                    statistics.stdev(values) if len(values) > 1 else None
                ),
                "min_score_kcal_mol": min(values),
                "max_score_kcal_mol": max(values),
                "mean_cnn_score": statistics.mean(cnn_values) if cnn_values else None,
                "sample_sd_cnn_score": (
                    statistics.stdev(cnn_values) if len(cnn_values) > 1 else None
                ),
                "mean_cnn_affinity": (
                    statistics.mean(affinity_values) if affinity_values else None
                ),
                "sample_sd_cnn_affinity": (
                    statistics.stdev(affinity_values)
                    if len(affinity_values) > 1 else None
                ),
                "representative_replicate": int(representative["replicate"]),
                "representative_score_kcal_mol": float(
                    representative["best_score_kcal_mol"]
                ),
                "representative_pose_file": str(representative.get("pose_file") or ""),
            }
        )
    return summaries


def _excluded_compound_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "excluded_compounds.tsv"
    rows_by_id: dict[str, dict[str, str]] = {}
    for payload_path in (run_dir / "result.json", run_dir / "metadata.json"):
        try:
            payload = json.loads(payload_path.read_text())
        except (OSError, TypeError, ValueError):
            payload = {}
        for row in payload.get("exclusion_details", []):
            compound_id = str(row.get("compound_id") or "").strip()
            if compound_id:
                rows_by_id[compound_id] = {
                    str(key): str(value or "") for key, value in row.items()
                }
        for value in payload.get("excluded_compound_ids", []):
            compound_id = str(value or "").strip()
            if compound_id and compound_id not in rows_by_id:
                rows_by_id[compound_id] = {
                    "compound_id": compound_id,
                    "stage": "persisted_engine_exclusion",
                    "reason": "Explicitly excluded from this engine campaign",
                    "log_file": "",
                }
    if path.is_file():
        with path.open(newline="", errors="replace") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                compound_id = str(row.get("compound_id") or "").strip()
                if compound_id:
                    rows_by_id[compound_id] = {
                        str(key): str(value or "") for key, value in row.items()
                    }
    return list(rows_by_id.values())


def _engine_timeout_registry_path(
    launch_campaign_id: str, engine: str
) -> Path | None:
    campaign_id = str(launch_campaign_id or "").strip()
    safe_engine = str(engine or "").strip().lower()
    if (
        not campaign_id
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", campaign_id)
        or safe_engine not in {"vina", "gnina"}
    ):
        return None
    return (
        app_home()
        / "config"
        / "docking_exclusions"
        / f"{campaign_id}.{safe_engine}.json"
    )


def _load_engine_timeout_exclusions(
    launch_campaign_id: str, engine: str
) -> dict[str, dict[str, str]]:
    path = _engine_timeout_registry_path(launch_campaign_id, engine)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return {}
    rows = payload.get("compounds") if isinstance(payload, dict) else {}
    return {
        str(compound_id): {
            str(key): str(value or "") for key, value in dict(details).items()
        }
        for compound_id, details in dict(rows or {}).items()
        if str(compound_id).strip() and isinstance(details, dict)
    }


def _record_engine_timeout_exclusions(
    launch_campaign_id: str,
    *,
    engine: str,
    source_run_id: str,
    compound_ids: Sequence[str],
) -> None:
    safe_engine = str(engine or "").strip().lower()
    path = _engine_timeout_registry_path(launch_campaign_id, safe_engine)
    clean_ids = sorted({str(value).strip() for value in compound_ids if str(value).strip()})
    if path is None or not clean_ids:
        return
    compounds = _load_engine_timeout_exclusions(launch_campaign_id, safe_engine)
    recorded_at = _utc_now_iso()
    for compound_id in clean_ids:
        compounds[compound_id] = {
            "reason": (
                "GNINA compound timeout"
                if safe_engine == "gnina"
                else "AutoDock Vina compound timeout"
            ),
            "source_run_id": str(source_run_id),
            "recorded_at": recorded_at,
        }
    payload = {
        "schema_version": 1,
        "launch_campaign_id": str(launch_campaign_id),
        "engine": safe_engine,
        "compounds": compounds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _load_vina_timeout_exclusions(
    launch_campaign_id: str,
) -> dict[str, dict[str, str]]:
    """Backward-compatible wrapper for existing Vina callers and tests."""
    return _load_engine_timeout_exclusions(launch_campaign_id, "vina")


def _record_vina_timeout_exclusions(
    launch_campaign_id: str,
    *,
    source_run_id: str,
    compound_ids: Sequence[str],
) -> None:
    """Backward-compatible wrapper for the engine-scoped timeout registry."""
    _record_engine_timeout_exclusions(
        launch_campaign_id,
        engine="vina",
        source_run_id=source_run_id,
        compound_ids=compound_ids,
    )


def _failed_attempt_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "failed_attempts.tsv"
    if not path.is_file():
        return []
    with path.open(newline="", errors="replace") as handle:
        return [
            {str(key): str(value or "") for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
            if str(row.get("compound_id") or "").strip()
        ]


def _preparation_exception_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "preparation_exceptions.tsv"
    if not path.is_file():
        return []
    with path.open(newline="", errors="replace") as handle:
        return [
            {str(key): str(value or "") for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
            if str(row.get("compound_id") or "").strip()
        ]


def finalize_docking_campaign_job(run_dir: Path, *, returncode: int) -> JobRecord:
    """Normalize native docking outputs and publish the campaign artifacts."""
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    safe_engine = str(metadata.get("engine") or metadata.get("tool") or "vina")
    compound_count = int(metadata.get("compound_count") or 0)
    replicates = max(1, int(metadata.get("replicates") or 1))
    seed_start = int(metadata.get("seed_start") or 1001)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_path.touch(exist_ok=True)
    stderr_path.touch(exist_ok=True)
    excluded_rows = _excluded_compound_rows(run_dir)
    excluded_ids = {
        str(row.get("compound_id") or "").strip() for row in excluded_rows
    }
    exclusions_path = run_dir / "excluded_compounds.tsv"
    if excluded_rows:
        with exclusions_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["compound_id", "stage", "reason", "log_file"],
                delimiter="\t",
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(excluded_rows)
    timeout_ids = {
        str(row.get("compound_id") or "").strip()
        for row in excluded_rows
        if str(row.get("stage") or "").strip() == f"{safe_engine}_timeout"
    }
    if safe_engine in {"vina", "gnina"} and timeout_ids:
        _record_engine_timeout_exclusions(
            str(metadata.get("launch_campaign_id") or ""),
            engine=safe_engine,
            source_run_id=str(metadata.get("run_id") or run_dir.name),
            compound_ids=sorted(timeout_ids),
        )
    failed_attempt_rows = [
        row
        for row in _failed_attempt_rows(run_dir)
        if str(row.get("compound_id") or "").strip() not in excluded_ids
    ]
    exception_rows = _preparation_exception_rows(run_dir)
    exception_by_id = {
        str(row.get("compound_id") or "").strip(): row for row in exception_rows
    }

    score_rows = _replicate_score_rows(
        run_dir,
        engine=safe_engine,
        replicates=replicates,
        seed_start=seed_start,
    )
    center_payload = dict(metadata.get("center") or {})
    size_payload = dict(metadata.get("size") or {})
    center = tuple(float(center_payload[axis]) for axis in "xyz")
    size = tuple(float(size_payload[axis]) for axis in "xyz")
    score_rows, pose_diagnostics = docking_pose_diagnostics(
        run_dir,
        score_rows,
        center=center,
        size=size,
    )
    score_rows = [
        row
        for row in score_rows
        if str(row.get("compound_id") or "") not in excluded_ids
    ]
    for row in score_rows:
        exception = exception_by_id.get(str(row.get("compound_id") or ""))
        row["preparation_exception"] = str(exception.get("exception") or "") if exception else ""
        row["preparation_exception_reason"] = (
            str(exception.get("reason") or "") if exception else ""
        )
    scores_path = run_dir / "scores.csv"
    with scores_path.open("w", newline="") as handle:
        fields = [
            "compound_id",
            "engine",
            "replicate",
            "seed",
            "best_score_kcal_mol",
            "cnn_score",
            "cnn_affinity",
            "pose_file",
            "pose_index",
            "pose_selection_criterion",
            "cnn_ranked_pose_index",
            "cnn_ranked_empirical_score_kcal_mol",
            "cnn_ranked_cnn_score",
            "cnn_ranked_cnn_affinity",
            "empirical_ranked_pose_index",
            "empirical_ranked_score_kcal_mol",
            "empirical_ranked_cnn_score",
            "empirical_ranked_cnn_affinity",
            "pose_centroid_x",
            "pose_centroid_y",
            "pose_centroid_z",
            "box_center_distance_angstrom",
            "outside_box_atom_count",
            "preparation_exception",
            "preparation_exception_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(score_rows)
    pose_set_path = run_dir / "poses.sdf"
    with pose_set_path.open("w") as output:
        for pose in sorted(results_dir.glob("**/*_out.sdf")):
            text = pose.read_text(errors="replace").rstrip()
            output.write(text + ("\n" if text.endswith("$$$$") else "\n$$$$\n"))
    eligible_compound_count = max(0, compound_count - len(excluded_ids))
    expected_pairs = eligible_compound_count * replicates
    completed_pairs = {
        (str(row.get("compound_id") or ""), int(row.get("replicate") or 1))
        for row in score_rows
    }
    completed_compound_ids = {pair[0] for pair in completed_pairs}
    failed_attempt_rows = [
        row
        for row in failed_attempt_rows
        if (
            (
                str(row.get("replicate") or "").strip().isdigit()
                and (
                    str(row.get("compound_id") or "").strip(),
                    int(row.get("replicate") or 1),
                )
                not in completed_pairs
            )
            or (
                not str(row.get("replicate") or "").strip().isdigit()
                and str(row.get("compound_id") or "").strip()
                not in completed_compound_ids
            )
        )
    ]
    expected_ids: list[str] = []
    compounds_path = run_dir / "input" / "compounds.tsv"
    if compounds_path.is_file():
        with compounds_path.open(newline="", errors="replace") as handle:
            expected_ids = [
                str(row.get("compound_id") or "").strip()
                for row in csv.DictReader(handle, delimiter="\t")
                if str(row.get("compound_id") or "").strip()
            ]
    recorded_failed_pairs = {
        (
            str(row.get("compound_id") or "").strip(),
            int(row.get("replicate") or 1)
            if str(row.get("replicate") or "").strip().isdigit()
            else 1,
        )
        for row in failed_attempt_rows
    }
    missing_pairs = [
        (compound_id, replicate)
        for compound_id in expected_ids
        if compound_id not in excluded_ids
        for replicate in range(1, replicates + 1)
        if (compound_id, replicate) not in completed_pairs
        and (compound_id, replicate) not in recorded_failed_pairs
    ]
    if missing_pairs:
        failed_attempts_path = run_dir / "failed_attempts.tsv"
        needs_header = not failed_attempts_path.is_file() or not failed_attempts_path.stat().st_size
        with failed_attempts_path.open("a", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            if needs_header:
                writer.writerow(["compound_id", "replicate", "reason"])
            for compound_id, replicate in missing_pairs:
                writer.writerow(
                    [
                        compound_id,
                        replicate,
                        "Engine emitted no readable pose for this compound-replicate",
                    ]
                )
        failed_attempt_rows = [
            row
            for row in _failed_attempt_rows(run_dir)
            if str(row.get("compound_id") or "").strip() not in excluded_ids
        ]
    failed_attempts_path = run_dir / "failed_attempts.tsv"
    if failed_attempts_path.is_file() or failed_attempt_rows:
        with failed_attempts_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["compound_id", "replicate", "reason"],
                delimiter="\t",
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(failed_attempt_rows)
    complete_compounds = {
        compound_id
        for compound_id in {pair[0] for pair in completed_pairs}
        if sum(pair[0] == compound_id for pair in completed_pairs) == replicates
    }
    summary_rows = _replicate_summary_rows(score_rows)
    for row in summary_rows:
        row.update(pose_diagnostics.get(str(row.get("compound_id") or ""), {}))
    summary_path = run_dir / "docking_replicate_summary.csv"
    if replicates > 1:
        with summary_path.open("w", newline="") as handle:
            fields = [
                "compound_id",
                "replicate_count",
                "mean_score_kcal_mol",
                "sample_sd_score_kcal_mol",
                "min_score_kcal_mol",
                "max_score_kcal_mol",
                "mean_cnn_score",
                "sample_sd_cnn_score",
                "mean_cnn_affinity",
                "sample_sd_cnn_affinity",
                "representative_replicate",
                "representative_score_kcal_mol",
                "representative_pose_file",
                "pose_rmsd_pair_count",
                "mean_pairwise_pose_rmsd_angstrom",
                "max_pairwise_pose_rmsd_angstrom",
                "pairwise_pose_rmsd_angstrom",
                "mean_box_center_distance_angstrom",
                "max_box_center_distance_angstrom",
                "replicates_with_atoms_outside_box",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)
    success = eligible_compound_count > 0 and bool(completed_pairs)
    partial_success = success and (
        returncode != 0
        or len(completed_pairs) != expected_pairs
        or bool(excluded_ids)
        or bool(failed_attempt_rows)
    )
    partial_reasons: list[str] = []
    if excluded_ids:
        partial_reasons.append(
            f"{len(excluded_ids)} compound(s) excluded before docking"
        )
    if failed_attempt_rows:
        partial_reasons.append(
            f"{len(failed_attempt_rows)} compound-replicate attempt(s) failed "
            "or emitted no readable pose"
        )
    if returncode != 0:
        partial_reasons.append(f"engine return code {returncode}")
    stderr = stderr_path.read_text(errors="replace")
    completed = _utc_now_iso()
    result = {
        "success": success,
        "returncode": returncode,
        "compound_count": compound_count,
        "eligible_compound_count": eligible_compound_count,
        "excluded_compounds": len(excluded_ids),
        "excluded_compound_ids": sorted(excluded_ids),
        "exclusion_details": excluded_rows,
        "preparation_exceptions": len(exception_by_id),
        "preparation_exception_ids": sorted(exception_by_id),
        "preparation_exception_details": exception_rows,
        "replicates": replicates,
        "seed_start": seed_start,
        "completed_compounds": len(complete_compounds),
        "completed_replicate_pairs": len(completed_pairs),
        "partial_success": partial_success,
        "failed_compounds": max(
            0, eligible_compound_count - len(complete_compounds)
        ),
        "failed_attempts": len(failed_attempt_rows),
        "failed_attempt_details": failed_attempt_rows,
        "progress": {"completed": len(completed_pairs), "total": expected_pairs},
        "warning": (
            (
                f"Partial result: {len(completed_pairs)}/{expected_pairs} eligible "
                "compound-replicate predictions completed"
                + (f"; {'; '.join(partial_reasons)}." if partial_reasons else ".")
            )
            if partial_success
            else ""
        ),
        "error": "" if success else (
            stderr[-4000:]
            or (
                "Docking produced no readable poses"
                if not completed_pairs
                else (
                    f"Docking produced {len(completed_pairs)}/{expected_pairs} "
                    "expected compound-replicate score sets"
                )
            )
        ),
    }
    _write_json(run_dir / "result.json", result)
    artifacts = [
        ArtifactRef.from_path(run_dir, scores_path, "docking_scores", role="ranked_scores"),
        ArtifactRef.from_path(run_dir, stdout_path, "job_log", role="stdout", checksum=False),
        ArtifactRef.from_path(run_dir, stderr_path, "job_log", role="stderr", checksum=False),
    ]
    if exclusions_path.is_file():
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                exclusions_path,
                "compound_exclusions",
                role="excluded_compounds",
                metadata={"excluded_compound_count": len(excluded_ids)},
            )
        )
    if failed_attempts_path.is_file():
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                failed_attempts_path,
                "compound_exclusions",
                role="failed_docking_attempts",
            )
        )
    exceptions_path = run_dir / "preparation_exceptions.tsv"
    if exceptions_path.is_file():
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                exceptions_path,
                "compound_exclusions",
                role="preparation_exceptions",
                metadata={"exception_compound_count": len(exception_by_id)},
            )
        )
    if pose_set_path.stat().st_size:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                pose_set_path,
                "pose_set",
                role="docked_poses",
                metadata={"compound_count": len(score_rows), "engine": safe_engine},
            )
        )
    if replicates > 1:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                summary_path,
                "docking_scores",
                role="replicate_summary",
                metadata={
                    "replicates": replicates,
                    "seed_start": seed_start,
                    "representative": "score closest to replicate median",
                },
            )
        )
    write_artifact_manifest(run_dir, artifacts)
    metadata.update(
        {
            "status": "completed" if success else "failed",
            "partial_success": partial_success,
            "updated_at": completed,
            "completed_at": completed,
        }
    )
    if not success:
        metadata["error"] = result["error"]
    else:
        metadata.pop("error", None)
    _write_json(metadata_path, metadata)
    return JobRecord.load(run_dir, task_group="docking")


def run_docking_campaign_job(
    *,
    receptor_path: Path,
    target_artifact: ArtifactRef,
    compound_paths: Sequence[Path],
    compound_artifacts: Sequence[ArtifactRef],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    box_mode: str = "fixed",
    box_padding_angstrom: float | None = None,
    engine: str = "udp",
    image: str = DEFAULT_DOCKING_IMAGE,
    gpu_device: str = "all",
    mode: str = "classic",
    search_mode: str = "detail",
    exhaustiveness: int = 30,
    poses: int = 10,
    use_scrub: bool = True,
    scrub_ph: float = 7.4,
    scrub_skip_tautomer: bool = True,
    reference_ligand_path: Path | None = None,
    reference_ligand_artifact: ArtifactRef | None = None,
    replicates: int = 1,
    seed_start: int = 1001,
    maximum_compounds: int = 0,
    cpu_workers: int | None = None,
    compound_timeout_minutes: int | None = None,
    extra_args: Sequence[str] = (),
    launch_campaign_id: str = "",
    launch_campaign_label: str = "",
    campaign_purpose: str = "",
    enqueue_only: bool = False,
) -> JobRecord:
    safe_engine = str(engine).strip().lower()
    if safe_engine not in {"udp", "vina", "gnina"}:
        raise ValueError(f"Unsupported docking engine: {engine}")
    safe_mode = str(mode).strip().lower()
    if safe_engine == "udp" and safe_mode not in {"classic", "hybrid"}:
        raise ValueError(f"Unsupported receptor docking mode: {mode}")
    if safe_engine == "udp" and safe_mode == "hybrid" and reference_ligand_path is None:
        raise ValueError("Uni-Dock Pro hybrid mode requires a reference ligand")
    replicates = int(replicates)
    if not 1 <= replicates <= 100:
        raise ValueError("Docking replicates must be between 1 and 100")
    seed_start = int(seed_start)
    if seed_start < 1 or seed_start + replicates - 1 >= 2_147_483_647:
        raise ValueError("Docking seed range must contain positive 32-bit integers")
    records = load_compound_records(compound_paths, maximum=maximum_compounds)
    rdkit_torsion_exclusions: list[tuple[str, int]] = []
    if safe_engine == "udp":
        for record in records:
            molecule = Chem.MolFromSmiles(str(record["smiles"]), sanitize=True)
            if molecule is None:
                continue
            rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(molecule))
            if rotatable_bonds > UNIDOCK_PRO_MAX_TORSIONS:
                rdkit_torsion_exclusions.append(
                    (str(record["compound_id"]), rotatable_bonds)
                )
        excluded_by_rdkit = {compound_id for compound_id, _ in rdkit_torsion_exclusions}
        records_for_preparation = [
            record
            for record in records
            if str(record["compound_id"]) not in excluded_by_rdkit
        ]
    else:
        records_for_preparation = records
    prior_timeout_exclusions = (
        _load_engine_timeout_exclusions(launch_campaign_id, safe_engine)
        if safe_engine in {"vina", "gnina"}
        else {}
    )
    if prior_timeout_exclusions:
        records_for_preparation = [
            record
            for record in records_for_preparation
            if str(record["compound_id"]) not in prior_timeout_exclusions
        ]
    unidock_batch_limit = unidock_pro_max_compounds()
    if safe_engine == "udp" and len(records) > unidock_batch_limit:
        raise ValueError(
            "Uni-Dock Pro accepts at most "
            f"{unidock_batch_limit:,} compounds in one batch; "
            f"the selected dataset contains {len(records):,}. "
            f"Set Maximum compounds to {unidock_batch_limit:,} or less, "
            "or change the Uni-Dock Pro batch limit in Settings."
        )
    vina_cpu_workers = (
        adaptive_cpu_workers(
            len(records) * replicates,
            requested=cpu_workers,
            hard_cap=128,
        )
        if safe_engine == "vina"
        else 1
    )
    vina_timeout_minutes = (
        max(1, int(compound_timeout_minutes))
        if compound_timeout_minutes is not None
        else vina_compound_timeout_minutes()
    )
    run_id = str(uuid4())
    run_dir = runs_root() / "docking" / run_id
    input_dir = run_dir / "input"
    prepared_dir = run_dir / "prepared"
    results_dir = run_dir / "results"
    input_dir.mkdir(parents=True, exist_ok=False)
    prepared_dir.mkdir()
    results_dir.mkdir()
    receptor_local = input_dir / "receptor.pdb"
    receptor_local.write_bytes(receptor_path.read_bytes())
    reference_local = input_dir / "reference_ligand.sdf"
    if reference_ligand_path is not None:
        reference_local.write_bytes(reference_ligand_path.read_bytes())
    compounds_path = input_dir / "compounds.tsv"
    with compounds_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["compound_id", "smiles"])
        writer.writerows(
            (item["compound_id"], item["smiles"]) for item in records_for_preparation
        )
    exclusions_path = run_dir / "excluded_compounds.tsv"
    with exclusions_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["compound_id", "stage", "reason", "log_file"])
        for compound_id, rotatable_bonds in rdkit_torsion_exclusions:
            writer.writerow(
                [
                    compound_id,
                    "unidock_pro_preflight",
                    (
                        "RDKit reports "
                        f"{rotatable_bonds} rotatable bonds; Uni-Dock Pro supports "
                        f"at most {UNIDOCK_PRO_MAX_TORSIONS} ligand torsions"
                    ),
                    "",
                ]
            )
        for compound_id, details in sorted(prior_timeout_exclusions.items()):
            engine_label = "GNINA" if safe_engine == "gnina" else "AutoDock Vina"
            writer.writerow(
                [
                    compound_id,
                    f"prior_{safe_engine}_timeout",
                    (
                        f"Skipped because this compound timed out in {engine_label} "
                        "earlier in the same launch campaign"
                    ),
                    str(details.get("source_run_id") or ""),
                ]
            )
    config_path = input_dir / "docking.conf"
    config_path.write_text(
        "".join(
            f"{name} = {value:.3f}\n"
            for name, value in zip(
                ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"),
                (*center, *size),
            )
        )
    )
    converter_path = input_dir / "pdbqt_to_sdf.py"
    converter_path.write_text(
        Path(__file__).with_name("meeko_pose_conversion.py").read_text()
    )
    rdkit_3d_path = input_dir / "smiles_to_3d.py"
    rdkit_3d_path.write_text(_RDKIT_3D_SCRIPT)
    safe_extra_args = tuple(str(value) for value in extra_args)
    prep_source = "prepared_ligands/${compound_id}.sdf"
    if use_scrub:
        scrub_command = (
            f'scrub.py "$source_sdf" -o "{prep_source}" --ph {float(scrub_ph):.2f}'
            + (" --skip_tautomer" if scrub_skip_tautomer else "")
        )
        standardization_script = (
            "if ! "
            + scrub_command
            + ' >>"$ligand_log" 2>&1 || [[ ! -s "'
            + prep_source
            + '" ]]; then '
            + f'rm -f "{prep_source}"; cp "$source_sdf" "{prep_source}"; '
            + 'record_exception "$compound_id" "scrub_bypass" '
            + '"Scrub produced no valid molecule; Meeko used the RDKit SDF directly" '
            + '"$ligand_log"; fi; '
        )
    else:
        standardization_script = (
            f'if ! cp "$source_sdf" "{prep_source}"; then '
            'record_exclusion "$compound_id" "standardization" '
            '"Could not stage the RDKit SDF for Meeko" "$ligand_log"; continue; fi; '
        )
    unidock_torsion_gate = ""
    if safe_engine == "udp":
        unidock_torsion_gate = (
            'torsion_count=$(awk \'/^TORSDOF[[:space:]]+/ { value=$2 } '
            'END { print value+0 }\' "prepared_ligands/${compound_id}.pdbqt"); '
            f'if (( torsion_count > {UNIDOCK_PRO_MAX_TORSIONS} )); then '
            'record_exclusion "$compound_id" "unidock_pro_preflight" '
            f'"Prepared PDBQT has ${{torsion_count}} torsions; Uni-Dock Pro supports at most '
            f'{UNIDOCK_PRO_MAX_TORSIONS}" "$ligand_log"; continue; fi; '
        )
    gnina_macrocycle_gate = ""
    if safe_engine == "gnina":
        gnina_macrocycle_gate = (
            'if grep -Eq "[[:space:]](CG[0-9]+|G[0-9]+)[[:space:]]*$" '
            '"prepared_ligands/${compound_id}.pdbqt"; then '
            'rigid_pdbqt="prepared_ligands/${compound_id}.rigid-macrocycle.pdbqt"; '
            f'if ! mk_prepare_ligand.py -i "{prep_source}" --rigid_macrocycles '
            '-o "$rigid_pdbqt" >>"$ligand_log" 2>&1 '
            '|| [[ ! -s "$rigid_pdbqt" ]]; then '
            'record_exclusion "$compound_id" "gnina_macrocycle_preflight" '
            '"GNINA does not accept Meeko macrocycle closure atom types and rigid-macrocycle preparation failed" '
            '"$ligand_log"; continue; fi; '
            'mv "$rigid_pdbqt" "prepared_ligands/${compound_id}.pdbqt"; '
            'record_exception "$compound_id" "gnina_rigid_macrocycle" '
            '"Prepared as a rigid macrocycle because GNINA does not accept Meeko closure atom types" '
            '"$ligand_log"; fi; '
        )
    prepare_script = (
        "tail -n +2 input/compounds.tsv | while IFS=$'\\t' read -r compound_id smiles; do "
        '[[ -n "$compound_id" && -n "$smiles" ]] || continue; '
        'ligand_log="exclusion_logs/${compound_id}.preparation.log"; '
        'printf "%s\\t%s\\n" "$smiles" "$compound_id" > "prepared_ligands/${compound_id}.smi"; '
        'source_sdf="prepared_ligands/${compound_id}_source.sdf"; '
        'if ! python input/smiles_to_3d.py --smiles "$smiles" --name "$compound_id" '
        '--output "$source_sdf" >"$ligand_log" 2>&1 || [[ ! -s "$source_sdf" ]]; then '
        'record_exclusion "$compound_id" "3d_generation" '
        '"RDKit ETKDGv3 did not produce a valid non-empty SDF" "$ligand_log"; continue; fi; '
        + standardization_script
        + f'if ! mk_prepare_ligand.py -i "{prep_source}" '
        + '-o "prepared_ligands/${compound_id}.pdbqt" >>"$ligand_log" 2>&1 '
        + '|| [[ ! -s "prepared_ligands/${compound_id}.pdbqt" ]]; then '
        + 'record_exclusion "$compound_id" "pdbqt_preparation" '
        + '"Meeko did not produce a valid non-empty PDBQT" "$ligand_log"; continue; fi; '
        + unidock_torsion_gate
        + gnina_macrocycle_gate
        + 'printf "prepared_ligands/%s.pdbqt\\n" "$compound_id" >> ligand_index.txt; '
        + "done; "
        + '[[ -s ligand_index.txt ]] || { echo "No ligands were prepared successfully" >&2; exit 3; }'
    )
    reference_script = ""
    reference_argument = ""
    if safe_engine == "udp" and safe_mode == "hybrid":
        reference_script = (
            "mk_prepare_ligand.py -i input/reference_ligand.sdf "
            "-o prepared/reference_ligand.pdbqt; "
        )
        reference_argument = "--reference_ligand prepared/reference_ligand.pdbqt "
    if safe_engine == "udp":
        docking = (
            'for replicate in $(seq 1 "$DOCKING_REPLICATES"); do '
            'seed=$((DOCKING_SEED_START + replicate - 1)); '
            'result_dir=$(printf "results/replicate_%03d" "$replicate"); '
            'mkdir -p "$result_dir"; '
            "udp --receptor prepared/receptor.pdbqt "
            + reference_argument
            + '--ligand_index ligand_index.txt --config input/docking.conf --dir "$result_dir" '
            + f"--search_mode {shlex.quote(search_mode)} "
            + f"--num_modes {max(1, int(poses))} "
            + '--seed "$seed" '
            + shlex.join(safe_extra_args)
            + "; done"
        ).strip()
    elif safe_engine == "gnina":
        docking = (
            ': > timed_out_compounds.txt; '
            'for replicate in $(seq 1 "$DOCKING_REPLICATES"); do '
            'seed=$((DOCKING_SEED_START + replicate - 1)); '
            'result_dir=$(printf "results/replicate_%03d" "$replicate"); '
            'mkdir -p "$result_dir"; '
            "while read -r ligand; do compound_id=$(basename \"$ligand\" .pdbqt); "
            + 'if grep -Fxq "$compound_id" timed_out_compounds.txt; then continue; fi; '
            + 'set +e; timeout --signal=TERM --kill-after=30s '
            + '"${VINA_COMPOUND_TIMEOUT_MINUTES}m" '
            + "gnina --receptor prepared/receptor.pdbqt --ligand \"$ligand\" "
            + "--config input/docking.conf "
            + f"--exhaustiveness {max(1, int(exhaustiveness))} --num_modes {max(1, int(poses))} "
            + '--seed "$seed" '
            + shlex.join(safe_extra_args)
            + ' --out "${result_dir}/${compound_id}_out.pdbqt"; '
            + 'exit_code=$?; set -e; '
            + 'if (( exit_code == 124 || exit_code == 137 )); then '
            + 'rm -f "${result_dir}/${compound_id}_out.pdbqt"; '
            + 'printf "%s\\n" "$compound_id" >> timed_out_compounds.txt; '
            + 'printf "%s\\t%s\\t%s\\n" "$compound_id" "$replicate" '
            + '"GNINA exceeded ${VINA_COMPOUND_TIMEOUT_MINUTES}-minute timeout" '
            + '>> failed_attempts.tsv; '
            + 'record_exclusion "$compound_id" "gnina_timeout" '
            + '"GNINA exceeded ${VINA_COMPOUND_TIMEOUT_MINUTES}-minute timeout; later replicas skipped" ""; '
            + 'elif (( exit_code != 0 )) || '
            + '[[ ! -s "${result_dir}/${compound_id}_out.pdbqt" ]]; then '
            + 'rm -f "${result_dir}/${compound_id}_out.pdbqt"; '
            + 'printf "%s\\t%s\\t%s\\n" "$compound_id" "$replicate" '
            + '"GNINA docking failed or emitted no readable pose (exit ${exit_code})" '
            + '>> failed_attempts.tsv; '
            + 'fi; '
            + "done < ligand_index.txt; done"
        )
    else:
        docking = (
            "dock_vina_task() { "
            'ligand="$1"; '
            'compound_id=$(basename "$ligand" .pdbqt); '
            'for replicate in $(seq 1 "$DOCKING_REPLICATES"); do '
            + 'if grep -Fxq "$compound_id" timed_out_compounds.txt; then break; fi; '
            + 'seed=$((DOCKING_SEED_START + replicate - 1)); '
            + 'result_dir=$(printf "results/replicate_%03d" "$replicate"); '
            + 'mkdir -p "$result_dir"; '
            + 'set +e; timeout --signal=TERM --kill-after=30s '
            + '"${VINA_COMPOUND_TIMEOUT_MINUTES}m" '
            + "vina --receptor prepared/receptor.pdbqt --ligand \"$ligand\" "
            + "--config input/docking.conf "
            + f"--exhaustiveness {max(1, int(exhaustiveness))} "
            + f"--num_modes {max(1, int(poses))} "
            + '--seed "$seed" '
            + shlex.join(safe_extra_args)
            + ' --cpu 1 --out "${result_dir}/${compound_id}_out.pdbqt"; '
            + 'exit_code=$?; set -e; '
            + 'if (( exit_code == 124 || exit_code == 137 )); then '
            + 'printf "%s\\n" "$compound_id" >> timed_out_compounds.txt; '
            + 'printf "%s\\t%s\\t%s\\n" "$compound_id" "$replicate" '
            + '"AutoDock Vina exceeded ${VINA_COMPOUND_TIMEOUT_MINUTES}-minute timeout" '
            + '>> failed_attempts.tsv; '
            + 'record_exclusion "$compound_id" "vina_timeout" '
            + '"AutoDock Vina exceeded ${VINA_COMPOUND_TIMEOUT_MINUTES}-minute timeout; later replicas skipped" ""; '
            + 'break; '
            + 'elif (( exit_code != 0 )); then '
            + 'printf "%s\\t%s\\t%s\\n" "$compound_id" "$replicate" '
            + '"AutoDock Vina docking failed (exit ${exit_code})" >> failed_attempts.tsv; '
            + 'fi; done; '
            + "}; export -f dock_vina_task; "
            + ': > timed_out_compounds.txt; cp ligand_index.txt vina_tasks.tsv; '
            + 'xargs -P "$DOCKING_CPU_WORKERS" -n 1 bash -c '
            + "'dock_vina_task \"$1\"' _ "
            + "< vina_tasks.tsv"
        )
    shell_script = (
        "set -euo pipefail; cd /workspace; "
        + f"DOCKING_REPLICATES={replicates}; DOCKING_SEED_START={seed_start}; "
        + f"DOCKING_CPU_WORKERS={vina_cpu_workers}; "
        + f"VINA_COMPOUND_TIMEOUT_MINUTES={vina_timeout_minutes}; "
        + "export DOCKING_REPLICATES DOCKING_SEED_START DOCKING_CPU_WORKERS "
        + "VINA_COMPOUND_TIMEOUT_MINUTES; "
        + "mkdir -p prepared_ligands results exclusion_logs; : > ligand_index.txt; "
        + "if [[ ! -s excluded_compounds.tsv ]]; then "
        + "printf 'compound_id\\tstage\\treason\\tlog_file\\n' > excluded_compounds.tsv; fi; "
        + "printf 'compound_id\\texception\\treason\\tlog_file\\n' > preparation_exceptions.tsv; "
        + "printf 'compound_id\\treplicate\\treason\\n' > failed_attempts.tsv; "
        + "record_exclusion() { printf '%s\\t%s\\t%s\\t%s\\n' "
        + '"$1" "$2" "$3" "$4" >> excluded_compounds.tsv; }; export -f record_exclusion; '
        + "record_exception() { printf '%s\\t%s\\t%s\\t%s\\n' "
        + '"$1" "$2" "$3" "$4" >> preparation_exceptions.tsv; }; export -f record_exception; '
        # Prepared targets already carry PDBFixer/OpenMM-validated protonation.
        # Prefer preserving that topology.  Some otherwise valid prepared
        # targets contain a histidine protonation pattern that Meeko cannot
        # disambiguate, however, while its hydrogen-free template path works.
        # Fall back only for those receptors instead of imposing either policy
        # on every target.
        "if ! mk_prepare_receptor.py --read_pdb input/receptor.pdb "
        "-o prepared/receptor -p; then "
        "rm -f prepared/receptor.pdbqt prepared/receptor.json; "
        "awk 'substr($0,77,2) !~ /^[[:space:]]*H[[:space:]]*$/' "
        "input/receptor.pdb > prepared/receptor-noh.pdb; "
        "mk_prepare_receptor.py -i prepared/receptor-noh.pdb "
        "-o prepared/receptor -p; fi; "
        + reference_script
        + prepare_script
        + "; "
        + docking
        + "; while IFS= read -r -d '' pose; do "
        + 'compound_id=$(basename "$pose" _out.pdbqt); '
        + 'if ! python input/pdbqt_to_sdf.py --pdbqt "$pose" '
        + '--template "prepared_ligands/${compound_id}.sdf" '
        + '--output "${pose%.pdbqt}.sdf"; then '
        + 'printf "%s\\t%s\\t%s\\n" "$compound_id" "unknown" '
        + '"Pose conversion to SDF failed" >> failed_attempts.tsv; fi; done'
        + " < <(find results -type f -name '*_out.pdbqt' -print0)"
    )
    tool_id = {"udp": "unidock_pro", "gnina": "gnina", "vina": "vina"}[safe_engine]
    uses_gpu = safe_engine != "vina"
    gpu_ids: tuple[int, ...] | None = None
    if uses_gpu:
        selected = str(gpu_device).strip().lower().removeprefix("device=")
        gpu_ids = None if selected == "all" else tuple(int(value) for value in selected.split(","))
    command = build_docker_command(
        DockerRunSpec(
            tool=registered_tool(tool_id, image=image),
            command=("bash", "-lc", shell_script),
            mounts=(DockerMount(run_dir, "/workspace"),),
            gpu_enabled=uses_gpu,
            gpu_devices=gpu_ids if uses_gpu else None,
            use_host_user=False,
        )
    )
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "docking_campaign",
        "workflow": "docking_campaign",
        "operation": "docking",
        "status": "queued" if enqueue_only else "running",
        "engine": safe_engine,
        "tool": safe_engine,
        "parent_run_id": target_artifact.run_id,
        "prepared_target_run_id": target_artifact.run_id,
        "compound_run_ids": list(dict.fromkeys(item.run_id for item in compound_artifacts)),
        "compound_count": len(records),
        "launch_campaign_id": str(launch_campaign_id),
        "launch_campaign_label": str(launch_campaign_label),
        "campaign_purpose": str(campaign_purpose),
        "center": {axis: float(value) for axis, value in zip("xyz", center)},
        "size": {axis: float(value) for axis, value in zip("xyz", size)},
        "box_mode": str(box_mode),
        "box_padding_angstrom": (
            float(box_padding_angstrom) if box_padding_angstrom is not None else None
        ),
        "mode": safe_mode if safe_engine == "udp" else "classic",
        "search_mode": search_mode if safe_engine == "udp" else "",
        "exhaustiveness": int(exhaustiveness) if safe_engine != "udp" else None,
        "poses_per_compound": int(poses),
        "replicates": replicates,
        "seed_start": seed_start,
        "use_scrub": bool(use_scrub),
        "scrub_ph": float(scrub_ph),
        "scrub_skip_tautomer": bool(scrub_skip_tautomer),
        "docker_image": image,
        "gpu_device": str(gpu_device),
        "cpu_workers": vina_cpu_workers if safe_engine == "vina" else None,
        "compound_timeout_minutes": (
            vina_timeout_minutes if safe_engine in {"vina", "gnina"} else None
        ),
        "prior_timeout_excluded_compound_ids": sorted(prior_timeout_exclusions),
        "reference_ligand_run_id": reference_ligand_artifact.run_id if reference_ligand_artifact else "",
        "created_at": now,
        "updated_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "target_artifact": target_artifact.to_dict(),
            "compound_artifacts": [item.to_dict() for item in compound_artifacts],
            "reference_ligand_artifact": reference_ligand_artifact.to_dict() if reference_ligand_artifact else None,
            "parameters": {key: metadata[key] for key in (
                "engine", "compound_count", "center", "size", "box_mode",
                "box_padding_angstrom", "mode", "search_mode",
                "exhaustiveness", "poses_per_compound", "replicates", "seed_start",
                "use_scrub", "scrub_ph", "gpu_device", "cpu_workers",
                "compound_timeout_minutes",
                "launch_campaign_id",
                "launch_campaign_label",
                "campaign_purpose",
            )},
            "command": [
                value.replace(str(run_dir), "<run-dir>") for value in command[:-1]
            ] + ["<docker-script>"],
        },
    )
    write_registered_command_record(
        run_dir,
        tool_id=tool_id,
        commands=(command,),
        image=image,
        selected_gpu_ids=gpu_ids or (),
    )
    if enqueue_only:
        manifest = registered_tool(tool_id, image=image)
        resources = manifest.resources.to_dict()
        if safe_engine == "vina":
            resources["cpu_threads"] = vina_cpu_workers
        if gpu_ids:
            resources["gpu_ids"] = list(gpu_ids)
        metadata.update(
            {
                "status": "queued",
                "queued_at": now,
                "queued_command": command,
                "gpu_queued": uses_gpu,
                "resources": resources,
                "worker_finalizer": "docking_campaign",
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        write_artifact_manifest(run_dir, [])
        return JobRecord.load(run_dir, task_group="docking")
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        returncode = int(process.returncode)
        stdout = process.stdout or ""
        stderr = process.stderr or ""
    except Exception as exc:
        returncode, stdout, stderr = -1, "", str(exc)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    return finalize_docking_campaign_job(run_dir, returncode=returncode)


def queue_docking_campaign_job(**parameters: Any) -> JobRecord:
    """Create a queued docking campaign without running its container."""
    return run_docking_campaign_job(enqueue_only=True, **parameters)


def _container_path(value: str) -> str:
    path = PurePosixPath(str(value).strip())
    if not str(value).strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Container workspace path must be relative: {value}")
    return path.as_posix()


def build_unidock_pro_native_command(
    *,
    ligand_index: str,
    config: str,
    output_dir: str,
    mode: str = "classic",
    receptor: str = "",
    reference_ligand: str = "",
    search_mode: str = "balance",
    executable: str = "udp",
    extra_args: Sequence[str] = (),
) -> list[str]:
    mode = str(mode).strip().lower()
    if mode not in UNIDOCK_PRO_MODES:
        raise ValueError(f"Unsupported Uni-Dock Pro mode: {mode}")
    search_mode = str(search_mode).strip().lower()
    if search_mode not in UNIDOCK_PRO_SEARCH_MODES:
        raise ValueError(f"Unsupported Uni-Dock Pro search mode: {search_mode}")
    executable = str(executable).strip()
    if not executable or "/" in executable:
        raise ValueError("Uni-Dock Pro executable must be a command name")
    if mode in {"classic", "hybrid"} and not receptor:
        raise ValueError(f"{mode} mode requires a receptor")
    if mode in {"hybrid", "ligand_based"} and not reference_ligand:
        raise ValueError(f"{mode} mode requires a reference ligand")

    command = [executable]
    if mode in {"classic", "hybrid"}:
        command += ["--receptor", _container_path(receptor)]
    if mode in {"hybrid", "ligand_based"}:
        command += ["--reference_ligand", _container_path(reference_ligand)]
    command += [
        "--ligand_index",
        _container_path(ligand_index),
        "--config",
        _container_path(config),
        "--dir",
        _container_path(output_dir),
        "--search_mode",
        search_mode,
    ]
    command.extend(str(value) for value in extra_args)
    return command


def build_unidock_pro_docker_command(
    *,
    image: str,
    workspace: Path,
    gpu_devices: str = "all",
    **native_parameters: object,
) -> list[str]:
    image = str(image).strip()
    if not image:
        raise ValueError("Uni-Dock Pro Docker image is required")
    workspace = workspace.resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(workspace)
    devices = str(gpu_devices).strip().lower()
    gpu_ids: tuple[int, ...] | None = None
    if devices != "all":
        normalized = ",".join(part.strip() for part in devices.removeprefix("device=").split(",") if part.strip())
        if not normalized or any(not part.isdigit() for part in normalized.split(",")):
            raise ValueError(f"Invalid GPU device selection: {gpu_devices}")
        gpu_ids = tuple(int(part) for part in normalized.split(","))
    native = build_unidock_pro_native_command(**native_parameters)
    return build_docker_command(
        DockerRunSpec(
            tool=registered_tool("unidock_pro", image=image),
            command=tuple(native),
            mounts=(DockerMount(workspace, "/workspace"),),
            gpu_devices=gpu_ids,
            workdir="/workspace",
            use_host_user=False,
        )
    )
