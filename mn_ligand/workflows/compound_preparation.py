from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.formulations import load_formulation_registry
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.runtime import cpu_process_limit, resolve_run_dir, runs_root


COMPOUND_IMPORT_TASK_GROUP = "compound-import"
COMPOUND_SELECTION_TASK_GROUP = "compound-selection"
SIZE_ESTIMATE_METHOD = "etkdg-v3-mmff-uff-principal-axes-v1"
SUPPORTED_COMPOUND_SUFFIXES = {
    ".sdf",
    ".smi",
    ".smiles",
    ".txt",
    ".csv",
    ".xlsx",
    ".xlsm",
}
TABULAR_COMPOUND_SUFFIXES = {".csv", ".xlsx", ".xlsm"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _safe_filename(filename: str) -> str:
    source = Path(filename).name
    suffix = Path(source).suffix.lower()
    if suffix not in SUPPORTED_COMPOUND_SUFFIXES:
        raise ValueError(
            "Compound datasets support Excel XLSX/XLSM, CSV, SDF, "
            "SMI/SMILES, and TXT files"
        )
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(source).stem).strip("-._") or "compounds"
    return f"{stem}{suffix}"


def compound_dataset_sheets(data: bytes, filename: str) -> tuple[str, ...]:
    suffix = Path(_safe_filename(filename)).suffix.lower()
    if suffix not in {".xlsx", ".xlsm"}:
        return ()
    try:
        workbook = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"Excel workbook could not be read: {exc}") from exc
    return tuple(str(name) for name in workbook.sheet_names)


def _read_tabular_dataset(
    data: bytes,
    filename: str,
    *,
    sheet_name: str = "",
) -> pd.DataFrame:
    suffix = Path(_safe_filename(filename)).suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(
                io.BytesIO(data),
                dtype=str,
                keep_default_na=False,
            )
        elif suffix in {".xlsx", ".xlsm"}:
            frame = pd.read_excel(
                io.BytesIO(data),
                sheet_name=sheet_name or 0,
                dtype=str,
                keep_default_na=False,
                engine="openpyxl",
            )
        else:
            return pd.DataFrame()
    except Exception as exc:
        label = "CSV file" if suffix == ".csv" else "Excel worksheet"
        raise ValueError(f"{label} could not be read: {exc}") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.fillna("")


def compound_dataset_columns(
    data: bytes,
    filename: str,
    *,
    sheet_name: str = "",
) -> tuple[str, ...]:
    suffix = Path(_safe_filename(filename)).suffix.lower()
    if suffix not in TABULAR_COMPOUND_SUFFIXES:
        return ()
    frame = _read_tabular_dataset(data, filename, sheet_name=sheet_name)
    return tuple(column for column in frame.columns if column)


def _descriptor_record(molecule: Chem.Mol) -> dict[str, Any]:
    molecular_weight = float(Descriptors.MolWt(molecule))
    clogp = float(Crippen.MolLogP(molecule))
    hbd = int(Lipinski.NumHDonors(molecule))
    hba = int(Lipinski.NumHAcceptors(molecule))
    return {
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "molecular_weight": round(molecular_weight, 4),
        "exact_mass": round(float(rdMolDescriptors.CalcExactMolWt(molecule)), 4),
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "hbd": hbd,
        "hba": hba,
        "clogp": round(clogp, 4),
        "tpsa": round(float(rdMolDescriptors.CalcTPSA(molecule)), 4),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
        "ring_count": int(Lipinski.RingCount(molecule)),
        "formal_charge": int(
            sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
        ),
        "fraction_csp3": round(
            float(rdMolDescriptors.CalcFractionCSP3(molecule)), 4
        ),
        "qed": round(float(QED.qed(molecule)), 4),
        "fragment_count": len(Chem.GetMolFrags(molecule)),
    }


def normalize_compound_smiles(smiles: str) -> dict[str, Any]:
    """Sanitize one SMILES and return its canonical structure and descriptors."""
    source = str(smiles or "").strip()
    if not source:
        raise ValueError("SMILES cannot be empty")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(source, sanitize=True)
    if molecule is None:
        raise ValueError("RDKit could not parse and sanitize the SMILES")
    if molecule.GetNumHeavyAtoms() < 1:
        raise ValueError("SMILES contains no heavy atoms")
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        raise ValueError("SMILES contains dummy or query atoms")
    canonical = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    if not canonical:
        raise ValueError("RDKit could not generate canonical SMILES")
    return {
        "smiles": canonical,
        "source_smiles": source,
        **_descriptor_record(molecule),
    }


def _component_records(
    molecule: Chem.Mol,
    *,
    compound_id: str,
    source_row: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    for index, fragment in enumerate(fragments, start=1):
        descriptors = _descriptor_record(fragment)
        records.append(
            {
                "compound_id": compound_id,
                "source_row": source_row,
                "component": index,
                "component_count": len(fragments),
                "smiles": Chem.MolToSmiles(
                    fragment, canonical=True, isomericSmiles=True
                ),
                "formula": descriptors["formula"],
                "molecular_weight": descriptors["molecular_weight"],
                "heavy_atoms": descriptors["heavy_atoms"],
                "formal_charge": descriptors["formal_charge"],
                "contains_carbon": any(
                    atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms()
                ),
            }
        )
    return records


def compound_component_records(
    smiles: str,
    *,
    compound_id: str,
    source_row: int = 0,
) -> list[dict[str, Any]]:
    """Return descriptor-rich disconnected components for one valid SMILES."""
    source = str(smiles or "").strip()
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(source, sanitize=True)
    if molecule is None:
        raise ValueError("RDKit could not parse and sanitize the SMILES")
    return _component_records(
        molecule,
        compound_id=str(compound_id),
        source_row=int(source_row),
    )


def docking_parent_record(smiles: str) -> dict[str, Any]:
    """Derive one comparison-only docking parent without mutating the source."""
    source = str(smiles or "").strip()
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(source, sanitize=True)
    if molecule is None:
        raise ValueError("RDKit could not parse and sanitize the SMILES")
    # Docking engines generally do not model isotope effects, and several
    # downstream parameterizers reject explicit deuterium atoms.  Likewise,
    # vendor SMILES occasionally encode a metal-counterion bond (for example
    # ``[O][Na]``) rather than disconnected components.  Normalize these only
    # in the derived docking parent; the imported source structure remains
    # unchanged and auditable.
    docking_molecule = Chem.Mol(molecule)
    for atom in docking_molecule.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetIsotope():
            atom.SetIsotope(0)
    docking_molecule = Chem.RemoveHs(docking_molecule)
    docking_molecule = rdMolStandardize.MetalDisconnector().Disconnect(
        docking_molecule
    )
    Chem.SanitizeMol(docking_molecule)
    grouped: dict[str, dict[str, Any]] = {}
    for fragment in Chem.GetMolFrags(
        docking_molecule, asMols=True, sanitizeFrags=True
    ):
        canonical = Chem.MolToSmiles(
            fragment, canonical=True, isomericSmiles=True
        )
        if canonical not in grouped:
            descriptors = _descriptor_record(fragment)
            grouped[canonical] = {
                "molecule": fragment,
                "smiles": canonical,
                "formula": str(descriptors["formula"]),
                "heavy_atoms": int(descriptors["heavy_atoms"]),
                "formal_charge": int(descriptors["formal_charge"]),
                "occurrences": 0,
            }
        grouped[canonical]["occurrences"] += 1
    options = sorted(
        grouped.values(),
        key=lambda row: (
            -int(row["heavy_atoms"]),
            str(row["smiles"]),
        ),
    )
    if not options:
        raise ValueError("Compound contains no molecular components")
    maximum_heavy_atoms = int(options[0]["heavy_atoms"])
    largest = [
        option
        for option in options
        if int(option["heavy_atoms"]) == maximum_heavy_atoms
    ]
    if len(largest) != 1:
        return {
            "source_fragment_count": len(Chem.GetMolFrags(molecule)),
            "docking_fragment_count": len(
                Chem.GetMolFrags(docking_molecule)
            ),
            "component_type_count": len(options),
            "parent_status": "ambiguous equal-size largest components",
            "docking_parent_smiles": "",
            "standardized_parent_smiles": "",
            "parent_formula": "",
            "standardized_parent_formula": "",
            "parent_formal_charge": "",
            "parent_occurrences": "",
        }
    parent = largest[0]
    uncharged = rdMolStandardize.Uncharger().uncharge(
        Chem.Mol(parent["molecule"])
    )
    Chem.SanitizeMol(uncharged)
    standardized_smiles = Chem.MolToSmiles(
        uncharged, canonical=True, isomericSmiles=True
    )
    standardized_formula = rdMolDescriptors.CalcMolFormula(uncharged)
    return {
        "source_fragment_count": len(Chem.GetMolFrags(molecule)),
        "docking_fragment_count": len(Chem.GetMolFrags(docking_molecule)),
        "component_type_count": len(options),
        "parent_status": (
            "single-component source"
            if len(options) == 1
            else "unique largest component candidate"
        ),
        "docking_parent_smiles": str(parent["smiles"]),
        "standardized_parent_smiles": standardized_smiles,
        "parent_formula": str(parent["formula"]),
        "standardized_parent_formula": standardized_formula,
        "parent_formal_charge": int(parent["formal_charge"]),
        "parent_occurrences": int(parent["occurrences"]),
    }


def parent_duplicate_report(
    compound_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive unique parents and group counterion/protonation equivalents."""
    derived_rows: list[dict[str, Any]] = []
    for source in compound_rows:
        row = dict(source)
        try:
            parent = docking_parent_record(str(row.get("smiles") or ""))
        except ValueError as exc:
            parent = {
                "parent_status": f"unavailable: {exc}",
                "standardized_parent_smiles": "",
            }
        derived_rows.append({**row, **parent})

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in derived_rows:
        key = str(row.get("standardized_parent_smiles") or "")
        if key:
            groups.setdefault(key, []).append(row)

    duplicate_rows: list[dict[str, Any]] = []
    duplicate_groups = [
        (key, rows) for key, rows in groups.items() if len(rows) > 1
    ]
    duplicate_groups.sort(
        key=lambda item: (
            -len(item[1]),
            str(item[0]),
        )
    )
    redundant_count = 0
    for key, rows in duplicate_groups:
        redundant_count += len(rows) - 1
        group_id = "DP-" + hashlib.sha1(
            key.encode("utf-8")
        ).hexdigest()[:8].upper()
        exact_parent_count = len(
            {
                str(row.get("docking_parent_smiles") or "")
                for row in rows
            }
        )
        match_type = (
            "exact parent structure"
            if exact_parent_count == 1
            else "charge-standardized parent"
        )
        for row in rows:
            duplicate_rows.append(
                {
                    "duplicate_group": group_id,
                    "group_size": len(rows),
                    "redundant_entries": len(rows) - 1,
                    "match_type": match_type,
                    **row,
                }
            )

    def joined_values(
        rows: list[dict[str, Any]], *columns: str
    ) -> str:
        values: list[str] = []
        for row in rows:
            for column in columns:
                value = str(row.get(column) or "").strip()
                if value and value not in values:
                    values.append(value)
        return " | ".join(values)

    def representative_rank(row: dict[str, Any]) -> tuple[Any, ...]:
        product_name = str(
            row.get("Product Name")
            or row.get("product_name")
            or ""
        )
        explicitly_named_isomer = bool(
            re.search(
                r"(?:\b[EZRS]\b|[([][EZRS](?:[- )]|$)|isomer)",
                product_name,
                flags=re.IGNORECASE,
            )
        )
        warning = str(row.get("validation_warning") or "").strip()
        origin = str(row.get("structure_origin") or "").lower()
        try:
            source_row = int(row.get("source_row") or 10**9)
        except (TypeError, ValueError):
            source_row = 10**9
        return (
            int(int(row.get("source_fragment_count") or 1) != 1),
            int(not explicitly_named_isomer),
            int(bool(warning)),
            int("pubchem" in origin),
            source_row,
            str(row.get("compound_id") or ""),
        )

    docking_parent_rows: list[dict[str, Any]] = []
    for key, rows in sorted(
        groups.items(),
        key=lambda item: (
            representative_rank(min(item[1], key=representative_rank)),
            str(item[0]),
        ),
    ):
        representative = min(rows, key=representative_rank)
        parent_id = "DP-" + hashlib.sha1(
            key.encode("utf-8")
        ).hexdigest()[:8].upper()
        source_count = len(rows)
        docking_parent_rows.append(
            {
                "docking_parent_id": parent_id,
                "source_record_count": source_count,
                "redundant_source_records": max(source_count - 1, 0),
                "representative_compound_id": str(
                    representative.get("compound_id") or ""
                ),
                "representative_product_name": str(
                    representative.get("Product Name")
                    or representative.get("product_name")
                    or ""
                ),
                "compound_ids": joined_values(rows, "compound_id"),
                "product_names": joined_values(
                    rows, "Product Name", "product_name"
                ),
                "cas_numbers": joined_values(
                    rows, "CAS Number", "cas_number"
                ),
                "structure_origins": joined_values(
                    rows, "structure_origin"
                ),
                "source_formulations": joined_values(
                    rows, "parent_status"
                ),
                "standardized_parent_formula": str(
                    representative.get("standardized_parent_formula") or ""
                ),
                "standardized_parent_smiles": key,
                "representative_source_smiles": str(
                    representative.get("smiles") or ""
                ),
                "parent_formal_charge_before_standardization": str(
                    representative.get("parent_formal_charge") or "0"
                ),
                "status": (
                    "Unique parent; ligand preparation required"
                ),
            }
        )
    return {
        "rows": duplicate_rows,
        "all_parent_rows": derived_rows,
        "docking_parent_rows": docking_parent_rows,
        "summary": {
            "unique_parent_count": len(docking_parent_rows),
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_entry_count": len(duplicate_rows),
            "redundant_entry_count": redundant_count,
            "ambiguous_parent_count": sum(
                int(
                    str(row.get("parent_status") or "").startswith(
                        "ambiguous"
                    )
                )
                for row in derived_rows
            ),
        },
    }


@lru_cache(maxsize=20000)
def estimate_parent_3d_size(smiles: str) -> dict[str, float] | None:
    """Estimate rotationally invariant ligand dimensions from one fixed conformer."""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 0x4D4E
    parameters.useRandomCoords = False
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        parameters.useRandomCoords = True
        if AllChem.EmbedMolecule(molecule, parameters) != 0:
            return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(molecule):
            AllChem.MMFFOptimizeMolecule(molecule, maxIters=300)
        else:
            AllChem.UFFOptimizeMolecule(molecule, maxIters=300)
    except (RuntimeError, ValueError):
        pass
    conformer = molecule.GetConformer()
    heavy_indices = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    if not heavy_indices:
        return None
    coordinates = np.asarray(
        [list(conformer.GetAtomPosition(index)) for index in heavy_indices],
        dtype=float,
    )
    centered = coordinates - coordinates.mean(axis=0)
    if len(coordinates) == 1:
        extents = np.zeros(3, dtype=float)
        maximum_span = 0.0
    else:
        _, axes = np.linalg.eigh(centered.T @ centered)
        projected = centered @ axes
        extents = np.ptp(projected, axis=0)
        differences = coordinates[:, None, :] - coordinates[None, :, :]
        maximum_span = float(
            np.sqrt(np.max(np.sum(differences * differences, axis=2)))
        )
    length, width, thickness = sorted(
        (float(value) for value in extents), reverse=True
    )
    return {
        "estimated_3d_length_angstrom": length,
        "estimated_3d_width_angstrom": width,
        "estimated_3d_thickness_angstrom": thickness,
        "estimated_max_span_angstrom": maximum_span,
    }


def _size_estimate_cache_path() -> Path:
    return runs_root().parent / "cache" / "compound-size-estimates-v1.json"


def _size_estimate_key(smiles: str) -> str:
    return hashlib.sha256(
        f"{SIZE_ESTIMATE_METHOD}\0{smiles}".encode()
    ).hexdigest()


def _read_size_estimate_cache(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("method") != SIZE_ESTIMATE_METHOD
        or not isinstance(payload.get("entries"), dict)
    ):
        return {}
    return dict(payload["entries"])


def estimate_parent_3d_sizes(
    smiles_values: list[str],
    *,
    max_workers: int | None = None,
) -> dict[str, dict[str, float] | None]:
    """Load reusable estimates and calculate missing structures concurrently."""
    unique_smiles = list(
        dict.fromkeys(str(smiles).strip() for smiles in smiles_values)
    )
    unique_smiles = [smiles for smiles in unique_smiles if smiles]
    if not unique_smiles:
        return {}
    cache_path = _size_estimate_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        cached_entries = _read_size_estimate_cache(cache_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    estimates: dict[str, dict[str, float] | None] = {}
    missing: list[str] = []
    for smiles in unique_smiles:
        cached = cached_entries.get(_size_estimate_key(smiles))
        if (
            isinstance(cached, dict)
            and cached.get("smiles") == smiles
            and "dimensions" in cached
        ):
            dimensions = cached.get("dimensions")
            estimates[smiles] = (
                {
                    str(key): float(value)
                    for key, value in dimensions.items()
                }
                if isinstance(dimensions, dict)
                else None
            )
        else:
            missing.append(smiles)

    if missing:
        worker_count = max_workers
        if worker_count is None:
            worker_count = min(cpu_process_limit(), len(missing))
        worker_count = max(1, min(int(worker_count), len(missing)))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="compound-size",
        ) as executor:
            calculated = list(executor.map(estimate_parent_3d_size, missing))
        new_entries: dict[str, dict[str, Any]] = {}
        for smiles, dimensions in zip(missing, calculated):
            estimates[smiles] = dimensions
            new_entries[_size_estimate_key(smiles)] = {
                "smiles": smiles,
                "dimensions": dimensions,
            }
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            merged = _read_size_estimate_cache(cache_path)
            merged.update(new_entries)
            payload = {
                "schema_version": 1,
                "method": SIZE_ESTIMATE_METHOD,
                "updated_at": _utc_now_iso(),
                "entry_count": len(merged),
                "entries": merged,
            }
            temporary = cache_path.with_name(
                f".{cache_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(temporary, cache_path)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return estimates


def compound_box_fit_rows(
    parent_rows: list[dict[str, Any]],
    *,
    box_size: tuple[float, float, float],
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Compare reusable one-conformer principal dimensions with the full box."""
    available = sorted(
        (max(0.0, float(value)) for value in box_size),
        reverse=True,
    )
    smiles_values = [
        str(
            parent.get("standardized_parent_smiles")
            or parent.get("smiles")
            or ""
        ).strip()
        for parent in parent_rows
    ]
    estimates = estimate_parent_3d_sizes(
        smiles_values, max_workers=max_workers
    )
    annotated: list[dict[str, Any]] = []
    for parent, smiles in zip(parent_rows, smiles_values):
        row = dict(parent)
        dimensions = estimates.get(smiles)
        if dimensions is None:
            row["box_fit_status"] = "not estimated"
            annotated.append(row)
            continue
        row.update(
            {
                key: round(float(value), 3)
                for key, value in dimensions.items()
            }
        )
        ligand_extents = [
            float(dimensions["estimated_3d_length_angstrom"]),
            float(dimensions["estimated_3d_width_angstrom"]),
            float(dimensions["estimated_3d_thickness_angstrom"]),
        ]
        dimension_labels = ("length", "width", "thickness")
        failures = [
            (label, ligand, box_axis)
            for label, ligand, box_axis in zip(
                dimension_labels, ligand_extents, available
            )
            if ligand > box_axis
        ]
        if failures:
            row["box_fit_status"] = "likely too large"
            row["box_fit_reason"] = "; ".join(
                f"{label} {ligand:.3f} Å > box {box_axis:.3f} Å"
                for label, ligand, box_axis in failures
            )
            row["box_fit_max_excess_angstrom"] = round(
                max(ligand - box_axis for _, ligand, box_axis in failures),
                3,
            )
        else:
            row["box_fit_status"] = "fits estimated box"
            row["box_fit_reason"] = ""
            row["box_fit_max_excess_angstrom"] = 0.0
        annotated.append(row)
    return annotated


def create_docking_parent_selection_job(
    *,
    source_job: JobRecord,
    source_artifact: ArtifactRef,
    parent_rows: list[dict[str, Any]],
    selection_mode: str,
    excluded_parent_rows: list[dict[str, Any]] | None = None,
    box_size: tuple[float, float, float] | None = None,
) -> JobRecord:
    """Publish an immutable typed compound set and any explicit exclusions."""
    if (
        source_job.task_group != COMPOUND_IMPORT_TASK_GROUP
        and source_job.workflow != "molecule_design_selection"
    ):
        raise ValueError(
            "Docking-parent selections require a compound-import or "
            "molecule-design-selection job"
        )
    if source_artifact.run_id != source_job.run_id:
        raise ValueError("Source compound artifact does not belong to the import job")
    if not parent_rows:
        raise ValueError("Select at least one unique docking parent")

    modeling_state_prepared = all(
        bool(str(parent.get("modeling_smiles") or "").strip())
        for parent in parent_rows
    )
    records: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, parent in enumerate(parent_rows, start=1):
        identity_smiles = str(
            parent.get("standardized_parent_smiles") or ""
        ).strip()
        identity_molecule = Chem.MolFromSmiles(identity_smiles)
        if identity_molecule is None:
            raise ValueError(
                f"Selected parent {index} has no valid standardized SMILES"
            )
        canonical_identity = Chem.MolToSmiles(
            identity_molecule, canonical=True, isomericSmiles=True
        )
        modeling_smiles = str(
            parent.get("modeling_smiles") or canonical_identity
        ).strip()
        modeling_molecule = Chem.MolFromSmiles(modeling_smiles)
        if modeling_molecule is None:
            raise ValueError(
                f"Selected parent {index} has no valid modeling SMILES"
            )
        canonical_modeling = Chem.MolToSmiles(
            modeling_molecule, canonical=True, isomericSmiles=True
        )
        base_id = re.sub(
            r"[^A-Za-z0-9._-]+",
            "-",
            str(
                parent.get("representative_compound_id")
                or parent.get("docking_parent_id")
                or f"parent-{index:07d}"
            ).strip(),
        ).strip("-._") or f"parent-{index:07d}"
        compound_id = base_id
        suffix = 2
        while compound_id in used_ids:
            compound_id = f"{base_id}-{suffix}"
            suffix += 1
        used_ids.add(compound_id)
        records.append(
            {
                "compound_id": compound_id,
                # ``smiles`` is the cross-engine modeling contract.  Parent
                # identity remains separate so salts/protonation variants can
                # still be grouped without silently neutralizing inference.
                "smiles": canonical_modeling,
                "identity_parent_smiles": canonical_identity,
                "modeling_smiles": canonical_modeling,
                "modeling_formal_charge": int(
                    Chem.GetFormalCharge(modeling_molecule)
                ),
                "modeling_preparation": str(
                    parent.get("modeling_preparation") or "identity parent"
                ),
                "modeling_ph": parent.get("modeling_ph", ""),
                "docking_parent_id": str(
                    parent.get("docking_parent_id") or ""
                ),
                "representative_product_name": str(
                    parent.get("representative_product_name") or ""
                ),
                "source_record_count": int(
                    parent.get("source_record_count") or 1
                ),
                "source_compound_ids": str(
                    parent.get("compound_ids") or ""
                ),
                "source_product_names": str(
                    parent.get("product_names") or ""
                ),
                "source_cas_numbers": str(
                    parent.get("cas_numbers") or ""
                ),
                "source_formulations": str(
                    parent.get("source_formulations") or ""
                ),
                "source_structure_origins": str(
                    parent.get("structure_origins") or ""
                ),
                "formula": str(
                    parent.get("standardized_parent_formula") or ""
                ),
                "estimated_3d_length_angstrom": parent.get(
                    "estimated_3d_length_angstrom", ""
                ),
                "estimated_3d_width_angstrom": parent.get(
                    "estimated_3d_width_angstrom", ""
                ),
                "estimated_3d_thickness_angstrom": parent.get(
                    "estimated_3d_thickness_angstrom", ""
                ),
                "estimated_max_span_angstrom": parent.get(
                    "estimated_max_span_angstrom", ""
                ),
                "size_estimate_method": (
                    SIZE_ESTIMATE_METHOD
                    if parent.get("estimated_3d_length_angstrom") not in ("", None)
                    else ""
                ),
                "box_fit_status": str(parent.get("box_fit_status") or ""),
                "box_fit_reason": str(parent.get("box_fit_reason") or ""),
                "box_fit_max_excess_angstrom": parent.get(
                    "box_fit_max_excess_angstrom", ""
                ),
            }
        )

    exclusions: list[dict[str, Any]] = []
    for parent in excluded_parent_rows or []:
        exclusions.append(
            {
                "docking_parent_id": str(
                    parent.get("docking_parent_id") or ""
                ),
                "representative_compound_id": str(
                    parent.get("representative_compound_id") or ""
                ),
                "representative_product_name": str(
                    parent.get("representative_product_name") or ""
                ),
                "source_compound_ids": str(
                    parent.get("compound_ids") or ""
                ),
                "standardized_parent_smiles": str(
                    parent.get("standardized_parent_smiles") or ""
                ),
                "estimated_3d_length_angstrom": parent.get(
                    "estimated_3d_length_angstrom", ""
                ),
                "estimated_3d_width_angstrom": parent.get(
                    "estimated_3d_width_angstrom", ""
                ),
                "estimated_3d_thickness_angstrom": parent.get(
                    "estimated_3d_thickness_angstrom", ""
                ),
                "estimated_max_span_angstrom": parent.get(
                    "estimated_max_span_angstrom", ""
                ),
                "size_estimate_method": (
                    SIZE_ESTIMATE_METHOD
                    if parent.get("estimated_3d_length_angstrom")
                    not in ("", None)
                    else ""
                ),
                "exclusion_category": "docking_box_fit",
                "exclusion_reason": str(
                    parent.get("box_fit_reason")
                    or "Estimated ligand dimensions exceed the docking box"
                ),
                "box_fit_max_excess_angstrom": parent.get(
                    "box_fit_max_excess_angstrom", ""
                ),
                "box_size_x_angstrom": (
                    float(box_size[0]) if box_size is not None else ""
                ),
                "box_size_y_angstrom": (
                    float(box_size[1]) if box_size is not None else ""
                ),
                "box_size_z_angstrom": (
                    float(box_size[2]) if box_size is not None else ""
                ),
            }
        )

    run_id = str(uuid4())
    run_dir = runs_root() / COMPOUND_SELECTION_TASK_GROUP / run_id
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    selected_path = artifact_dir / "docking_parents.csv"
    pd.DataFrame(records).to_csv(selected_path, index=False)
    exclusion_path = artifact_dir / "excluded_docking_parents.csv"
    if exclusions:
        pd.DataFrame(exclusions).to_csv(exclusion_path, index=False)
    now = _utc_now_iso()
    dataset_name = str(
        source_job.metadata.get("dataset_name") or "Compound dataset"
    )
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "docking_parent_selection",
        "workflow": "compound_selection",
        "status": "completed",
        "parent_run_id": source_job.run_id,
        "source_compound_run_id": source_job.run_id,
        "dataset_name": dataset_name,
        "selection_mode": str(selection_mode),
        "compound_count": len(records),
        "excluded_compound_count": len(exclusions),
        "docking_box_size_angstrom": (
            [float(value) for value in box_size]
            if box_size is not None
            else None
        ),
        "modeling_state_contract": (
            "shared prepared modeling SMILES; identity parent retained separately"
        ),
        "modeling_state_prepared": modeling_state_prepared,
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "source_compound_artifact": source_artifact.to_dict(),
            "selection_mode": str(selection_mode),
            "selected_docking_parent_ids": [
                str(row.get("docking_parent_id") or "")
                for row in parent_rows
            ],
            "excluded_docking_parent_ids": [
                row["docking_parent_id"] for row in exclusions
            ],
            "exclusion_policy": (
                "exclude_estimated_box_mismatch" if exclusions else "none"
            ),
            "docking_box_size_angstrom": (
                [float(value) for value in box_size]
                if box_size is not None
                else None
            ),
            "modeling_state_contract": (
                "shared prepared modeling SMILES; identity parent retained separately"
            ),
            "modeling_state_prepared": modeling_state_prepared,
        },
    )
    _write_json(
        run_dir / "result.json",
        {
            "success": True,
            "compound_count": len(records),
            "excluded_compound_count": len(exclusions),
            "compound_set": "artifacts/docking_parents.csv",
            "exclusion_report": (
                "artifacts/excluded_docking_parents.csv"
                if exclusions
                else None
            ),
        },
    )
    artifact = ArtifactRef.from_path(
        run_dir,
        selected_path,
        "compound_set",
        role="selected_docking_parents",
        label=f"{dataset_name} — {len(records)} selected parents",
        metadata={
            "compound_count": len(records),
            "source_compound_run_id": source_job.run_id,
            "selection_mode": str(selection_mode),
            "parent_identity": "stereochemistry-aware standardized parent",
            "modeling_state": "shared prepared SMILES in the smiles column",
            "modeling_state_prepared": modeling_state_prepared,
        },
    )
    artifacts = [artifact]
    if exclusions:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                exclusion_path,
                "compound_exclusion_report",
                role="excluded_docking_parents",
                label=(
                    f"{dataset_name} — {len(exclusions)} excluded parents"
                ),
                metadata={
                    "excluded_compound_count": len(exclusions),
                    "exclusion_category": "docking_box_fit",
                    "source_compound_run_id": source_job.run_id,
                    "docking_box_size_angstrom": (
                        [float(value) for value in box_size]
                        if box_size is not None
                        else None
                    ),
                },
            )
        )
    write_artifact_manifest(run_dir, artifacts)
    return JobRecord.load(
        run_dir, task_group=COMPOUND_SELECTION_TASK_GROUP
    )


@lru_cache(maxsize=1)
def _common_component_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    registry = load_formulation_registry()
    for component in registry.components:
        for smiles in component.smiles:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(
                    f"Invalid SMILES in formulation registry "
                    f"{component.component_id!r}: {smiles}"
                )
            canonical = Chem.MolToSmiles(
                molecule, canonical=True, isomericSmiles=True
            )
            existing = lookup.get(canonical)
            if existing is not None and existing != component.label:
                raise ValueError(
                    f"Formulation SMILES {smiles!r} is assigned to both "
                    f"{existing!r} and {component.label!r}"
                )
            lookup[canonical] = component.label
    return lookup


def annotate_component_relationships(
    valid_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Label common components and exact parent matches without changing structures."""
    annotated = [dict(row) for row in component_rows]
    rows_by_compound: dict[str, list[dict[str, Any]]] = {}
    for row in annotated:
        rows_by_compound.setdefault(str(row["compound_id"]), []).append(row)

    parent_smiles: dict[str, str] = {}
    for compound_id, rows in rows_by_compound.items():
        maximum = max(int(row.get("heavy_atoms") or 0) for row in rows)
        largest = [
            row for row in rows if int(row.get("heavy_atoms") or 0) == maximum
        ]
        if len(largest) == 1:
            parent_smiles[compound_id] = str(largest[0].get("smiles") or "")

    single_component_ids: dict[str, list[str]] = {}
    for row in valid_rows:
        if int(row.get("fragment_count") or 1) != 1:
            continue
        single_component_ids.setdefault(str(row.get("smiles") or ""), []).append(
            str(row.get("compound_id") or "")
        )

    formulations_by_parent: dict[str, list[str]] = {}
    for compound_id, smiles in parent_smiles.items():
        formulations_by_parent.setdefault(smiles, []).append(compound_id)

    common_lookup = _common_component_lookup()
    recognized_by_compound: dict[str, set[str]] = {}
    for row in annotated:
        label = common_lookup.get(str(row.get("smiles") or ""), "")
        if label:
            recognized_by_compound.setdefault(
                str(row.get("compound_id") or ""), set()
            ).add(label)

    for row in annotated:
        compound_id = str(row.get("compound_id") or "")
        smiles = str(row.get("smiles") or "")
        parent = parent_smiles.get(compound_id, "")
        row["recognized_component"] = common_lookup.get(smiles, "")
        row["parent_candidate"] = bool(parent and smiles == parent)
        if row["recognized_component"]:
            row["component_category"] = row["recognized_component"]
        elif row["parent_candidate"]:
            row["component_category"] = "parent candidate"
        elif bool(row.get("contains_carbon")):
            row["component_category"] = (
                "unrecognized organic co-component; review"
            )
        else:
            row["component_category"] = (
                "unrecognized non-carbon co-component; review"
            )
        recognized = recognized_by_compound.get(compound_id, set())
        if recognized:
            row["formulation_category"] = "recognized: " + ", ".join(
                sorted(recognized)
            )
        elif all(
            bool(item.get("contains_carbon"))
            for item in rows_by_compound.get(compound_id, [])
        ):
            row["formulation_category"] = (
                "multiple organic components; mixture/co-drug/homolog review"
            )
        else:
            row["formulation_category"] = (
                "unrecognized formulation components; review"
            )
        row["unformulated_library_matches"] = ", ".join(
            item
            for item in single_component_ids.get(parent, [])
            if item and item != compound_id
        )
        row["related_formulations"] = ", ".join(
            item
            for item in formulations_by_parent.get(parent, [])
            if item and item != compound_id
        )
        if not parent:
            row["parent_match_status"] = "ambiguous largest components"
        elif row["unformulated_library_matches"]:
            row["parent_match_status"] = (
                "exact single-component structure present in library"
            )
        else:
            row["parent_match_status"] = (
                "no exact single-component structure found in library"
            )
    return annotated


def annotation_stripped_smiles_candidate(source_smiles: str) -> str:
    """Remove vendor prose-like dot annotations, without altering real fragments."""
    parts: list[str] = []
    current: list[str] = []
    bracket_depth = 0
    for character in source_smiles:
        if character == "[":
            bracket_depth += 1
        elif character == "]" and bracket_depth:
            bracket_depth -= 1
        if character == "." and bracket_depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(character)
    parts.append("".join(current))
    retained: list[str] = []
    for part in parts:
        token = part.strip()
        if re.fullmatch(
            r"\[(?:Z|E|Rotation|\(\+\)|\(-\)|\+|-|"
            r"(?:\d+(?:\.\d+)?|\d+/\d+)"
            r"(?:\s+[A-Za-z0-9+\-]+)?)\]",
            token,
            flags=re.IGNORECASE,
        ):
            continue
        retained.append(token)
    return ".".join(retained)


def analyze_tabular_compound_dataset(
    data: bytes,
    filename: str,
    *,
    smiles_column: str,
    id_column: str = "",
    sheet_name: str = "",
) -> dict[str, Any]:
    """Validate and normalize a CSV/Excel compound table with RDKit."""
    frame = _read_tabular_dataset(data, filename, sheet_name=sheet_name)
    if frame.empty:
        raise ValueError("Compound dataset contains no rows")
    if smiles_column not in frame.columns:
        raise ValueError(f"SMILES column was not found: {smiles_column}")
    if id_column and id_column not in frame.columns:
        raise ValueError(f"Compound ID column was not found: {id_column}")
    extra_fields = [
        column for column in frame.columns if column not in {smiles_column, id_column}
    ]
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    used_ids: dict[str, int] = {}
    canonical_counts: dict[str, int] = {}
    for offset, (_, source) in enumerate(frame.iterrows(), start=2):
        source_values = {
            field: str(source.get(field, "") or "").strip()
            for field in extra_fields
        }
        source_smiles = str(source.get(smiles_column, "") or "").strip()
        source_id = (
            str(source.get(id_column, "") or "").strip() if id_column else ""
        )
        base_id = source_id or f"compound_{offset - 1:07d}"
        used_ids[base_id] = used_ids.get(base_id, 0) + 1
        compound_id = (
            base_id
            if used_ids[base_id] == 1
            else f"{base_id}__{used_ids[base_id]}"
        )
        if not source_smiles:
            invalid_rows.append(
                {
                    "source_row": offset,
                    "compound_id": compound_id,
                    "smiles": "",
                    "error": "Missing SMILES",
                    **source_values,
                }
            )
            continue
        try:
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(source_smiles, sanitize=True)
            if molecule is None:
                raise ValueError("RDKit could not parse and sanitize the SMILES")
            if molecule.GetNumHeavyAtoms() < 1:
                raise ValueError("SMILES contains no heavy atoms")
            if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
                raise ValueError("SMILES contains dummy or query atoms")
            canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            if not canonical:
                raise ValueError("RDKit could not generate canonical SMILES")
            descriptors = _descriptor_record(molecule)
        except Exception as exc:
            candidate = annotation_stripped_smiles_candidate(source_smiles)
            candidate_valid = False
            if candidate and candidate != source_smiles:
                with rdBase.BlockLogs():
                    candidate_molecule = Chem.MolFromSmiles(
                        candidate, sanitize=True
                    )
                candidate_valid = (
                    candidate_molecule is not None
                    and candidate_molecule.GetNumHeavyAtoms() > 0
                    and not any(
                        atom.GetAtomicNum() == 0
                        for atom in candidate_molecule.GetAtoms()
                    )
                )
            invalid_rows.append(
                {
                    "source_row": offset,
                    "compound_id": compound_id,
                    "smiles": source_smiles,
                    "error": str(exc),
                    "annotation_stripped_candidate": (
                        candidate if candidate_valid else ""
                    ),
                    "candidate_status": (
                        "Review required; source vendor annotation removed"
                        if candidate_valid
                        else ""
                    ),
                    **source_values,
                }
            )
            continue
        canonical_counts[canonical] = canonical_counts.get(canonical, 0) + 1
        warnings = []
        if descriptors["fragment_count"] > 1:
            warnings.append("multiple disconnected fragments")
        if used_ids[base_id] > 1:
            warnings.append("duplicate compound ID renamed")
        valid_rows.append(
            {
                "compound_id": compound_id,
                "smiles": canonical,
                "source_smiles": source_smiles,
                "source_row": offset,
                **descriptors,
                "validation_warning": "; ".join(warnings),
                **source_values,
            }
        )
        if descriptors["fragment_count"] > 1:
            component_rows.extend(
                _component_records(
                    molecule,
                    compound_id=compound_id,
                    source_row=offset,
                )
            )
    component_rows = annotate_component_relationships(valid_rows, component_rows)
    if not valid_rows:
        raise ValueError("No downstream-usable SMILES were found")
    duplicate_structure_count = sum(
        count - 1 for count in canonical_counts.values() if count > 1
    )
    molecular_weights = [float(row["molecular_weight"]) for row in valid_rows]
    numeric_columns: list[dict[str, Any]] = []
    for field in extra_fields:
        source_values = frame[field].astype(str).str.strip()
        nonempty_count = int(source_values.ne("").sum())
        numeric_values = pd.to_numeric(
            source_values.where(source_values.ne("")),
            errors="coerce",
        ).dropna()
        if numeric_values.empty:
            continue
        numeric_columns.append(
            {
                "column": field,
                "numeric_count": int(len(numeric_values)),
                "nonempty_count": nonempty_count,
                "numeric_fraction": round(
                    len(numeric_values) / max(nonempty_count, 1), 4
                ),
                "missing_count": int(len(frame) - nonempty_count),
                "minimum": float(numeric_values.min()),
                "median": float(numeric_values.median()),
                "mean": float(numeric_values.mean()),
                "maximum": float(numeric_values.max()),
            }
        )
    summary = {
        "source_row_count": int(len(frame)),
        "valid_count": len(valid_rows),
        "invalid_count": len(invalid_rows),
        "duplicate_structure_count": duplicate_structure_count,
        "multi_fragment_count": sum(
            int(row["fragment_count"] > 1) for row in valid_rows
        ),
        "charged_count": sum(int(row["formal_charge"] != 0) for row in valid_rows),
        "charged_component_count": sum(
            int(row["formal_charge"] != 0) for row in component_rows
        ),
        "recognized_formulation_component_count": sum(
            int(bool(row.get("recognized_component"))) for row in component_rows
        ),
        "uncategorized_multi_fragment_count": len(
            {
                str(row["compound_id"])
                for row in component_rows
                if str(row.get("formulation_category") or "").startswith(
                    "unrecognized"
                )
            }
        ),
        "multi_fragment_with_unformulated_match_count": len(
            {
                str(row["compound_id"])
                for row in component_rows
                if row.get("parent_candidate")
                and row.get("unformulated_library_matches")
            }
        ),
        "molecular_weight_min": min(molecular_weights),
        "molecular_weight_mean": sum(molecular_weights) / len(molecular_weights),
        "molecular_weight_max": max(molecular_weights),
        "numeric_source_column_count": len(numeric_columns),
    }
    return {
        "columns": list(frame.columns),
        "extra_fields": extra_fields,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "component_rows": component_rows,
        "summary": summary,
        "numeric_columns": numeric_columns,
        "sheet_name": sheet_name,
        "smiles_column": smiles_column,
        "id_column": id_column,
    }


def summarize_compound_dataset(
    data: bytes,
    filename: str,
    *,
    smiles_column: str = "",
    sheet_name: str = "",
) -> dict[str, Any]:
    safe_name = _safe_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    text = data.decode("utf-8", errors="replace")
    if suffix == ".sdf":
        count = sum(1 for block in text.split("$$$$") if block.strip())
        data_format = "sdf"
    elif suffix in TABULAR_COMPOUND_SUFFIXES:
        frame = _read_tabular_dataset(data, filename, sheet_name=sheet_name)
        field_map = {
            str(name).strip().lower(): str(name) for name in frame.columns
        }
        smiles_field = str(smiles_column).strip() or next(
            (field_map[name] for name in ("smiles", "canonical_smiles", "isomeric_smiles") if name in field_map),
            "",
        )
        if smiles_field and smiles_field not in frame.columns:
            raise ValueError(f"SMILES column was not found: {smiles_field}")
        if not smiles_field:
            raise ValueError("Tabular compound datasets require a SMILES column")
        count = sum(
            1 for value in frame[smiles_field] if str(value or "").strip()
        )
        data_format = suffix.removeprefix(".")
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if lines and lines[0].lower().replace(" ", "_") in {"smiles", "smiles_name", "name_smiles"}:
            lines = lines[1:]
        count = len(lines)
        data_format = "smiles"
    if count < 1:
        raise ValueError("Compound dataset contains no readable compounds")
    return {
        "format": data_format,
        "compound_count": count,
        "original_filename": Path(filename).name,
        "size_bytes": len(data),
    }


def create_compound_import_job(
    data: bytes,
    *,
    filename: str,
    source: str = "upload",
    dataset_name: str = "",
    id_column: str = "",
    smiles_column: str = "",
    sheet_name: str = "",
) -> JobRecord:
    summary = summarize_compound_dataset(
        data,
        filename,
        smiles_column=smiles_column,
        sheet_name=sheet_name,
    )
    run_id = str(uuid4())
    run_dir = runs_root() / COMPOUND_IMPORT_TASK_GROUP / run_id
    artifact_dir = run_dir / "artifacts" / "compounds"
    report_dir = run_dir / "artifacts" / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True)

    dataset_path = artifact_dir / _safe_filename(filename)
    dataset_path.write_bytes(data)
    compound_set_path = dataset_path
    source_path: Path | None = None
    validation: dict[str, Any] | None = None
    rejected_path: Path | None = None
    components_path: Path | None = None
    validation_path: Path | None = None
    if summary["format"] in {"csv", "xlsx", "xlsm"}:
        columns = compound_dataset_columns(
            data,
            filename,
            sheet_name=sheet_name,
        )
        field_map = {str(name).strip().lower(): str(name) for name in columns}
        selected_smiles = str(smiles_column).strip() or next(
            field_map[name]
            for name in ("smiles", "canonical_smiles", "isomeric_smiles")
            if name in field_map
        )
        selected_id = str(id_column).strip() or next(
            (
                field_map[name]
                for name in ("compound_id", "id", "name", "zincid")
                if name in field_map
            ),
            "",
        )
        validation = analyze_tabular_compound_dataset(
            data,
            filename,
            smiles_column=selected_smiles,
            id_column=selected_id,
            sheet_name=sheet_name,
        )
        normalized_fields = [
            "compound_id",
            "smiles",
            "source_smiles",
            "source_row",
            "formula",
            "molecular_weight",
            "exact_mass",
            "heavy_atoms",
            "hbd",
            "hba",
            "clogp",
            "tpsa",
            "rotatable_bonds",
            "ring_count",
            "formal_charge",
            "fraction_csp3",
            "qed",
            "fragment_count",
            "validation_warning",
            *validation["extra_fields"],
        ]
        normalized = io.StringIO()
        writer = csv.DictWriter(
            normalized,
            fieldnames=list(dict.fromkeys(normalized_fields)),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(validation["valid_rows"])
        source_path = dataset_path
        normalized_dir = run_dir / "artifacts" / "normalized"
        normalized_dir.mkdir(parents=True)
        compound_set_path = normalized_dir / "compounds.csv"
        compound_set_path.write_text(normalized.getvalue())
        validation_path = report_dir / "validation_report.json"
        _write_json(
            validation_path,
            {
                "engine": "RDKit",
                "rdkit_version": rdBase.rdkitVersion,
                "sheet_name": sheet_name,
                "id_column": selected_id,
                "smiles_column": selected_smiles,
                "columns": validation["columns"],
                "summary": validation["summary"],
                "numeric_columns": validation["numeric_columns"],
                "component_rows": validation["component_rows"],
            },
        )
        if validation["component_rows"]:
            components_path = report_dir / "compound_components.csv"
            pd.DataFrame(validation["component_rows"]).to_csv(
                components_path, index=False
            )
        if validation["invalid_rows"]:
            rejected_path = report_dir / "rejected_compounds.csv"
            rejected_fields = list(
                dict.fromkeys(
                    [
                        "source_row",
                        "compound_id",
                        "smiles",
                        "error",
                        "annotation_stripped_candidate",
                        "candidate_status",
                        *validation["extra_fields"],
                    ]
                )
            )
            rejected_buffer = io.StringIO()
            rejected_writer = csv.DictWriter(
                rejected_buffer,
                fieldnames=rejected_fields,
                extrasaction="ignore",
            )
            rejected_writer.writeheader()
            rejected_writer.writerows(validation["invalid_rows"])
            rejected_path.write_text(rejected_buffer.getvalue())
        summary["compound_count"] = validation["summary"]["valid_count"]
    report_path = report_dir / "import_report.json"
    report = {
        **summary,
        "dataset_name": dataset_name.strip() or dataset_path.stem,
        "source": source,
        "id_column": id_column,
        "smiles_column": smiles_column,
        "sheet_name": sheet_name,
        "workbook_sheets": list(compound_dataset_sheets(data, filename)),
        "validation": validation["summary"] if validation else {},
    }
    _write_json(report_path, report)
    _write_json(run_dir / "input.json", {"source": source, **report})
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "compound_import",
        "status": "completed",
        "source": source,
        "dataset_name": report["dataset_name"],
        "compound_count": summary["compound_count"],
        "invalid_compound_count": (
            validation["summary"]["invalid_count"] if validation else 0
        ),
        "duplicate_structure_count": (
            validation["summary"]["duplicate_structure_count"] if validation else 0
        ),
        "compound_format": summary["format"],
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "result.json",
        {
            "success": True,
            "compound_count": summary["compound_count"],
            "invalid_compound_count": (
                validation["summary"]["invalid_count"] if validation else 0
            ),
            "format": summary["format"],
        },
    )
    artifacts = [
        ArtifactRef.from_path(
            run_dir,
            compound_set_path,
            "compound_set",
            role="normalized_compounds" if source_path is not None else "source_compounds",
            label=report["dataset_name"],
            metadata={"compound_count": summary["compound_count"], "format": summary["format"]},
        ),
        ArtifactRef.from_path(run_dir, report_path, "import_report", role="compound_import"),
    ]
    if source_path is not None:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                source_path,
                "source_compound_dataset",
                role="original_upload",
                label=Path(filename).name,
            )
        )
    if validation_path is not None:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                validation_path,
                "compound_validation_report",
                role="rdkit_validation",
            )
        )
    if rejected_path is not None:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                rejected_path,
                "rejected_compounds",
                role="invalid_rows",
                metadata={
                    "compound_count": validation["summary"]["invalid_count"]
                    if validation
                    else 0
                },
            )
        )
    if components_path is not None:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                components_path,
                "compound_components",
                role="multi_fragment_component_report",
                metadata={
                    "compound_count": validation["summary"][
                        "multi_fragment_count"
                    ]
                    if validation
                    else 0
                },
            )
        )
    write_artifact_manifest(run_dir, artifacts)
    return JobRecord.load(run_dir, task_group=COMPOUND_IMPORT_TASK_GROUP)


def load_compound_import_job(run_id: str) -> JobRecord:
    run_dir = resolve_run_dir(COMPOUND_IMPORT_TASK_GROUP, run_id)
    if run_dir is None:
        raise FileNotFoundError(f"Compound-import job not found: {run_id}")
    return JobRecord.load(run_dir, task_group=COMPOUND_IMPORT_TASK_GROUP)


def list_compound_import_jobs() -> list[JobRecord]:
    root = runs_root() / COMPOUND_IMPORT_TASK_GROUP
    if not root.is_dir():
        return []
    return sorted(
        (JobRecord.load(path, task_group=COMPOUND_IMPORT_TASK_GROUP) for path in root.iterdir() if path.is_dir()),
        key=lambda job: (job.created_at, job.run_dir.stat().st_mtime),
        reverse=True,
    )
