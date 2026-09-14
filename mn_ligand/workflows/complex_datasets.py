from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO, StringIO
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

import pandas as pd
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import (
    JOB_SCHEMA_VERSION,
    JobRecord,
    display_job_code,
    iter_job_records,
    short_job_code,
)
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.bound_ligand_md import parse_bound_ligands
from mn_ligand.workflows.pose_validation import _source_receptor


COMPLEX_DATASET_TASK_GROUP = "complex-datasets"
SELECTED_COMPLEX_TASK_GROUP = "selected-complexes"
COMPLEX_DATASET_SCHEMA_VERSION = 1
REQUIRED_COLUMNS = (
    "Compound",
    "Source job",
)
STEREOCHEMISTRY_POLICY_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda value: (
                value.item() if hasattr(value, "item") else str(value)
            ),
        )
        + "\n"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complex_dataset_sheets(data: bytes) -> tuple[str, ...]:
    workbook = pd.ExcelFile(BytesIO(data), engine="openpyxl")
    return tuple(str(value) for value in workbook.sheet_names)


def read_complex_dataset(data: bytes, sheet_name: str) -> pd.DataFrame:
    frame = pd.read_excel(
        BytesIO(data), sheet_name=sheet_name, engine="openpyxl"
    ).fillna("")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(
            "Complex-selection worksheet is missing required columns: "
            + ", ".join(missing)
        )
    if not any(
        column in frame
        for column in (
            "Interaction evidence path",
            "Predicted complex path",
            "Source pose path",
            "MD complex path",
        )
    ):
        raise ValueError(
            "Complex-selection worksheet is missing a source-pose or interaction-"
            "evidence path"
        )
    if frame.empty:
        raise ValueError("Complex-selection worksheet contains no rows")
    return frame


def _safe_name(value: object, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    return cleaned.strip("-._") or fallback


def _job_code_lookup(jobs: list[JobRecord]) -> dict[str, list[JobRecord]]:
    lookup: dict[str, list[JobRecord]] = {}
    for job in jobs:
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        lookup.setdefault(code.upper(), []).append(job)
    return lookup


def _analysis_job_for_path(
    path: Path,
    jobs_by_id: dict[str, JobRecord],
) -> JobRecord:
    root = runs_root(create=False).resolve()
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Interaction evidence path is outside the configured run store"
        ) from exc
    if (
        len(relative.parts) < 4
        or relative.parts[0] != "interaction-analysis"
        or relative.parts[2] != "prepared"
    ):
        raise ValueError(
            "Interaction evidence path must identify an immutable prepared "
            "interaction-analysis pose"
        )
    analysis_run_id = relative.parts[1]
    analysis_job = jobs_by_id.get(analysis_run_id)
    if analysis_job is None or analysis_job.task_group != "interaction-analysis":
        raise ValueError(
            f"Interaction-analysis job {analysis_run_id} was not found"
        )
    if analysis_job.status != "completed":
        raise ValueError(
            f"Interaction-analysis job {analysis_run_id} is not completed"
        )
    if not resolved.is_file():
        raise ValueError(f"Predicted complex does not exist: {resolved}")
    return analysis_job


def _pose_id(path: Path) -> str:
    name = path.name
    return (
        name[: -len(".complex.pdb")]
        if name.endswith(".complex.pdb")
        else path.stem
    )


def _prediction_rank(row: dict[str, Any]) -> int:
    prediction = str(row.get("Prediction") or "").strip()
    match = re.search(r"\bpose\s+(\d+)\b", prediction, re.IGNORECASE)
    if match is None:
        raise ValueError(
            "Docking selection requires Prediction in the form 'pose N'"
        )
    rank = int(match.group(1))
    if rank < 1:
        raise ValueError("Docking pose rank must be at least 1")
    return rank


def _source_job_for_artifact_path(
    path: Path,
    jobs_by_id: dict[str, JobRecord],
) -> JobRecord:
    root = runs_root(create=False).resolve()
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Source pose path is outside the configured run store") from exc
    if len(relative.parts) < 3:
        raise ValueError("Source pose path does not identify a prediction run")
    source_job = jobs_by_id.get(relative.parts[1])
    if source_job is None or source_job.run_dir.resolve() != (
        root / relative.parts[0] / relative.parts[1]
    ).resolve():
        raise ValueError("Source pose prediction job was not found")
    if source_job.status != "completed":
        raise ValueError("Source pose prediction job is not completed")
    if not resolved.is_file():
        raise ValueError(f"Source prediction pose does not exist: {resolved}")
    return source_job


def _direct_source_pose_reference(
    row: dict[str, Any],
    jobs_by_id: dict[str, JobRecord],
) -> tuple[JobRecord, Path, int, str]:
    """Resolve one selected spreadsheet row into one source-pose record."""
    path_text = str(
        row.get("Source pose path")
        or row.get("MD complex path")
        or row.get("Predicted complex path")
        or ""
    ).strip()
    if not path_text:
        raise ValueError("Selected row has no source pose path")
    source_path = Path(path_text).expanduser().resolve()
    source_job = _source_job_for_artifact_path(source_path, jobs_by_id)
    declared_code = str(row.get("Source job") or "").strip().upper()
    actual_code = display_job_code(
        source_job.metadata.get("job_code"), source_job.run_id
    )
    if declared_code and actual_code.upper() != declared_code:
        raise ValueError(f"Source job is {actual_code}, not {declared_code}")

    compound = str(row.get("Compound") or "").strip()
    sdf_index = 0
    pose_id = source_path.stem
    if source_job.task_group == "docking" and source_path.suffix.lower() == ".sdf":
        rank = _prediction_rank(row)
        sdf_index = rank - 1
        replicate_text = str(row.get("Replicate") or "").strip()
        try:
            replicate = int(float(replicate_text))
        except ValueError as exc:
            raise ValueError("Docking selection requires a numeric Replicate") from exc
        expected_directory = f"replicate_{replicate:03d}"
        if expected_directory not in source_path.parts:
            raise ValueError(
                f"Source pose path is not from selected replicate {replicate}"
            )
        if compound and compound not in source_path.name:
            raise ValueError("Source pose path does not match selected Compound")
        pose_id = f"{compound}__replicate_{replicate:03d}__pose_{rank:03d}"
    return source_job, source_path, sdf_index, pose_id


def _source_pose_reference(
    analysis_job: JobRecord,
    source_job: JobRecord,
    pose_id: str,
) -> tuple[Path, int, dict[str, Any]]:
    """Resolve an analysis pose back to its immutable prediction artifact."""
    inventory_path = analysis_job.run_dir / "input" / "interaction_inputs.csv"
    try:
        inventory = pd.read_csv(inventory_path).fillna("")
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Interaction analysis does not retain its source-pose inventory"
        ) from exc
    matches = inventory.loc[inventory["pose_id"].astype(str).eq(pose_id)]
    if len(matches) != 1:
        raise ValueError(
            f"Could not resolve {pose_id} to exactly one source prediction pose"
        )
    inventory_row = matches.iloc[0].to_dict()
    relative = str(inventory_row.get("source_artifact_path") or "").strip()
    if not relative:
        raise ValueError(f"{pose_id} has no source prediction artifact")
    source_path = (source_job.run_dir / relative).resolve()
    try:
        source_path.relative_to(source_job.run_dir.resolve())
    except ValueError as exc:
        raise ValueError("Source pose path escapes its immutable prediction run") from exc
    if not source_path.is_file():
        raise ValueError(f"Source prediction pose does not exist: {source_path}")

    sdf_index = 0
    if source_job.task_group == "docking" and source_path.suffix.lower() == ".sdf":
        prediction = str(inventory_row.get("prediction") or "")
        rank_match = re.search(r"\bpose\s+(\d+)\b", prediction, re.IGNORECASE)
        if rank_match is None:
            rank_match = re.search(r"__pose_(\d+)$", pose_id)
        if rank_match is None:
            raise ValueError(f"Could not recover the source pose rank for {pose_id}")
        sdf_index = int(rank_match.group(1)) - 1
        if sdf_index < 0:
            raise ValueError(f"Invalid source pose rank for {pose_id}")
    return source_path, sdf_index, inventory_row


def _source_pose_complex_pdb(
    source_job: JobRecord,
    source_path: Path,
    sdf_index: int,
) -> str:
    """Materialize an MD complex from the prediction, never an analysis copy."""
    if source_path.suffix.lower() != ".sdf":
        try:
            structure = gemmi.read_structure(str(source_path))
        except (OSError, RuntimeError) as exc:
            raise ValueError("Could not read the source prediction structure") from exc
        if not structure or not structure[0]:
            raise ValueError("Source prediction artifact has no coordinate model")
        pdb_data = structure.make_pdb_string()
        if not any(line.startswith("ATOM  ") for line in pdb_data.splitlines()):
            raise ValueError("Source prediction artifact has no protein coordinates")
        return pdb_data if pdb_data.endswith("\n") else pdb_data + "\n"

    supplier = Chem.SDMolSupplier(
        str(source_path), removeHs=False, sanitize=False
    )
    if sdf_index >= len(supplier) or supplier[sdf_index] is None:
        raise ValueError(
            f"Source docking pose {sdf_index + 1} is unavailable in {source_path.name}"
        )
    molecule = supplier[sdf_index]
    receptor_path = _source_receptor(source_job)
    receptor_lines = [
        line
        for line in receptor_path.read_text(errors="replace").splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    if not receptor_lines:
        raise ValueError("Source docking receptor contains no coordinates")
    ligand_lines: list[str] = []
    serial = len(receptor_lines) + 1
    for line in Chem.MolToPDBBlock(molecule).splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        line = f"HETATM{serial:5d}" + line[11:]
        ligand_lines.append(
            line[:17] + "LIG" + line[20:21] + "Z" + f"{1:4d}" + line[26:]
        )
        serial += 1
    if not ligand_lines:
        raise ValueError("Source docking pose contains no ligand coordinates")
    return "\n".join(receptor_lines + ligand_lines + ["END", ""])


def _compound_smiles(
    source_job: JobRecord,
    compound_id: str,
    jobs_by_id: dict[str, JobRecord],
) -> str:
    """Recover the immutable compound chemistry used by a prediction job."""
    def _smiles_from_table(table: pd.DataFrame) -> str:
        if "compound_id" not in table:
            return ""
        matches = table.loc[table["compound_id"].astype(str).eq(compound_id)]
        if len(matches) != 1:
            return ""
        for column in (
            "modeling_smiles",
            "smiles",
            "canonical_smiles",
            "isomeric_smiles",
        ):
            value = str(matches.iloc[0].get(column) or "").strip()
            if value:
                return value
        return ""

    # Docking jobs retain an immutable, per-run copy even when their original
    # compound-set job is no longer available.
    snapshot_path = source_job.run_dir / "input" / "compounds.tsv"
    if snapshot_path.exists():
        try:
            value = _smiles_from_table(
                pd.read_csv(snapshot_path, sep="\t").fillna("")
            )
        except (OSError, ValueError):
            value = ""
        if value:
            return value

    input_path = source_job.run_dir / "input.json"
    try:
        payload = json.loads(input_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("Source prediction does not retain its input record") from exc
    references = [
        *(payload.get("compound_sets") or ()),
        *(payload.get("compound_artifacts") or ()),
    ]
    for reference in references:
        run_id = str(reference.get("run_id") or "")
        relative = str(reference.get("path") or "")
        owner = jobs_by_id.get(run_id)
        if owner is None or not relative:
            continue
        table_path = (owner.run_dir / relative).resolve()
        try:
            table_path.relative_to(owner.run_dir.resolve())
            table = pd.read_csv(table_path).fillna("")
        except (OSError, ValueError):
            continue
        value = _smiles_from_table(table)
        if value:
            return value
    raise ValueError(
        f"Could not recover immutable source chemistry for {compound_id}"
    )


def source_stereochemistry_report(
    source_smiles: str,
    pose_molecule: Chem.Mol,
) -> dict[str, Any]:
    """Compare exact 3D tetrahedral geometry with immutable source identity."""
    source = Chem.MolFromSmiles(str(source_smiles or "").strip())
    if source is None:
        raise ValueError("The immutable source SMILES is invalid")
    source = Chem.RemoveHs(source)
    geometry = Chem.RemoveHs(Chem.Mol(pose_molecule), sanitize=False)
    try:
        Chem.SanitizeMol(geometry)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(
            "The selected pose cannot be sanitized for stereochemistry checking"
        ) from exc
    if geometry.GetNumConformers() != 1:
        raise ValueError(
            "The selected pose needs exactly one conformer for "
            "stereochemistry checking"
        )
    Chem.AssignStereochemistry(source, cleanIt=True, force=True)
    assigned_source_centres = Chem.FindMolChiralCenters(
        source,
        includeUnassigned=False,
        useLegacyImplementation=False,
    )
    source_stereo_smiles = Chem.MolToSmiles(
        source,
        isomericSmiles=True,
    )
    source_graph = Chem.MolToSmiles(source, isomericSmiles=False)
    geometry_graph = Chem.MolToSmiles(geometry, isomericSmiles=False)
    geometry_from_coordinates = Chem.Mol(geometry)
    if assigned_source_centres and source_graph != geometry_graph:
        # Docking formats can change aromatic/kekulized bond notation or a
        # protonation-state bond order without changing the heavy-atom
        # connectivity. Reapply only the immutable source bond orders while
        # retaining the exact predicted coordinates; absolute configuration is
        # still derived below exclusively from those coordinates.
        source_elements = [atom.GetSymbol() for atom in source.GetAtoms()]
        geometry_elements = [
            atom.GetSymbol() for atom in geometry.GetAtoms()
        ]
        if source_elements == geometry_elements:
            source_conformer = geometry.GetConformer()
            geometry_from_coordinates = Chem.Mol(source)
            conformer = Chem.Conformer(source.GetNumAtoms())
            for atom_index in range(source.GetNumAtoms()):
                conformer.SetAtomPosition(
                    atom_index,
                    source_conformer.GetAtomPosition(atom_index),
                )
            geometry_from_coordinates.RemoveAllConformers()
            geometry_from_coordinates.AddConformer(
                conformer, assignId=True
            )
        else:
            try:
                geometry_from_coordinates = (
                    AllChem.AssignBondOrdersFromTemplate(source, geometry)
                )
                Chem.SanitizeMol(geometry_from_coordinates)
            except (ValueError, RuntimeError) as exc:
                raise ValueError(
                    "Selected ligand graph does not match its immutable "
                    "source SMILES"
                ) from exc
        geometry_graph = Chem.MolToSmiles(
            geometry_from_coordinates,
            isomericSmiles=False,
        )
        if geometry_graph != source_graph:
            raise ValueError(
                "Selected ligand graph does not match its immutable source "
                "SMILES"
            )
    for atom in geometry_from_coordinates.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in geometry_from_coordinates.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    Chem.AssignAtomChiralTagsFromStructure(
        geometry_from_coordinates,
        confId=0,
        replaceExistingTags=True,
    )
    Chem.AssignStereochemistry(
        geometry_from_coordinates,
        cleanIt=True,
        force=True,
    )
    geometry_stereo_smiles = Chem.MolToSmiles(
        geometry_from_coordinates,
        isomericSmiles=True,
    )
    assigned_geometry_centres = Chem.FindMolChiralCenters(
        geometry_from_coordinates,
        includeUnassigned=True,
        useLegacyImplementation=False,
    )
    # With no source-defined tetrahedral centres there is no absolute
    # configuration to enforce. Otherwise canonical isomeric identity must be
    # identical after deriving the pose tags from its actual coordinates.
    matches = (
        True
        if not assigned_source_centres
        else geometry_stereo_smiles == source_stereo_smiles
    )
    return {
        "policy_version": STEREOCHEMISTRY_POLICY_VERSION,
        "matches": bool(matches),
        "source_smiles": source_stereo_smiles,
        "geometry_smiles": geometry_stereo_smiles,
        "source_centres": "; ".join(
            f"{index}:{label}" for index, label in assigned_source_centres
        ),
        "geometry_centres": "; ".join(
            f"{index}:{label}" for index, label in assigned_geometry_centres
        ),
        "defined_source_centre_count": len(assigned_source_centres),
    }


def _source_pose_ligand_sdf(
    source_job: JobRecord,
    source_path: Path,
    sdf_index: int,
    compound_id: str,
    complex_pdb: str,
    jobs_by_id: dict[str, JobRecord],
) -> tuple[str, str]:
    """Return the exact selected pose with explicit, source-backed chemistry."""
    immutable_smiles = _compound_smiles(
        source_job, compound_id, jobs_by_id
    )
    if source_job.task_group == "docking" and source_path.suffix.lower() == ".sdf":
        supplier = Chem.SDMolSupplier(
            str(source_path), removeHs=False, sanitize=False
        )
        if sdf_index >= len(supplier) or supplier[sdf_index] is None:
            raise ValueError(
                f"Source docking pose {sdf_index + 1} has no ligand topology"
            )
        molecule = Chem.Mol(supplier[sdf_index])
        try:
            Chem.SanitizeMol(molecule)
        except ValueError as exc:
            raise ValueError("Selected docking pose has invalid ligand chemistry") from exc
        smiles = immutable_smiles
    else:
        ligand_lines = [
            line
            for line in complex_pdb.splitlines()
            if line.startswith("HETATM")
            and line[76:78].strip().upper() != "H"
            and not (
                not line[76:78].strip()
                and re.match(
                    r"^\d*H(?:\d|$)",
                    line[12:16].strip().upper(),
                )
            )
        ]
        if not ligand_lines:
            raise ValueError("Selected predicted pose has no ligand coordinates")
        coordinate_molecule = Chem.MolFromPDBBlock(
            "\n".join(ligand_lines + ["END", ""]),
            sanitize=False,
            removeHs=True,
            proximityBonding=True,
        )
        smiles = immutable_smiles
        template = Chem.MolFromSmiles(smiles)
        if coordinate_molecule is None or template is None:
            raise ValueError("Could not construct the selected ligand chemistry")
        if coordinate_molecule.GetNumAtoms() != template.GetNumAtoms():
            raise ValueError(
                f"{compound_id}: predicted ligand atom count does not match "
                "its immutable source SMILES"
            )
        template_elements = [
            atom.GetSymbol() for atom in template.GetAtoms()
        ]
        coordinate_elements = [
            atom.GetSymbol() for atom in coordinate_molecule.GetAtoms()
        ]
        if template_elements == coordinate_elements:
            # AF3/Boltz preserve the input-SMILES atom order in their ligand
            # component. Build from that authoritative graph and transfer only
            # the exact predicted coordinates. Inferring a graph from a PDB
            # block can choose another symmetry match and falsely invert a
            # stereocentre even though the immutable input topology is known.
            molecule = Chem.Mol(template)
            source_conformer = coordinate_molecule.GetConformer()
            conformer = Chem.Conformer(template.GetNumAtoms())
            for atom_index in range(template.GetNumAtoms()):
                conformer.SetAtomPosition(
                    atom_index,
                    source_conformer.GetAtomPosition(atom_index),
                )
            molecule.RemoveAllConformers()
            molecule.AddConformer(conformer, assignId=True)
        else:
            try:
                molecule = AllChem.AssignBondOrdersFromTemplate(
                    template, coordinate_molecule
                )
                Chem.SanitizeMol(molecule)
            except (ValueError, RuntimeError) as exc:
                raise ValueError(
                    f"{compound_id}: predicted ligand coordinates cannot be "
                    "mapped to source chemistry"
                ) from exc
            expected_graph = Chem.MolToSmiles(
                template, isomericSmiles=False
            )
            observed_graph = Chem.MolToSmiles(
                molecule, isomericSmiles=False
            )
            if observed_graph != expected_graph:
                raise ValueError(
                    f"{compound_id}: predicted ligand connectivity does not "
                    "match its immutable source SMILES"
                )
        try:
            Chem.SanitizeMol(molecule)
            # Cofolding coordinate files normally contain heavy atoms only.
            # The selected-ligand SDF is the MD topology input, so materialize
            # the hydrogens dictated by the immutable source chemistry while
            # retaining the exact predicted heavy-atom coordinates.
            molecule = Chem.AddHs(molecule, addCoords=True)
            Chem.SanitizeMol(molecule)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(
                f"{compound_id}: could not add source-chemistry hydrogens to "
                "the selected predicted ligand"
            ) from exc
    if molecule.GetNumConformers() != 1:
        raise ValueError("Selected ligand does not contain exactly one coordinate set")
    stereo = source_stereochemistry_report(immutable_smiles, molecule)
    molecule.SetProp(
        "IMMUTABLE_SOURCE_SMILES",
        str(stereo["source_smiles"]),
    )
    molecule.SetProp(
        "PREDICTED_GEOMETRY_SMILES",
        str(stereo["geometry_smiles"]),
    )
    molecule.SetProp(
        "PREDICTED_GEOMETRY_STEREO_MATCH",
        str(bool(stereo["matches"])).lower(),
    )
    molecule.SetIntProp(
        "STEREOCHEMISTRY_POLICY_VERSION",
        int(stereo["policy_version"]),
    )
    molecule.SetProp(
        "SOURCE_STEREOCENTRES",
        str(stereo["source_centres"]),
    )
    molecule.SetProp(
        "PREDICTED_GEOMETRY_STEREOCENTRES",
        str(stereo["geometry_centres"]),
    )
    if not stereo["matches"]:
        raise ValueError(
            f"{compound_id}: selected pose inverts or loses immutable source "
            "stereochemistry"
        )
    molecule.SetProp("_Name", compound_id)
    sdf_output = StringIO()
    sdf_writer = Chem.SDWriter(sdf_output)
    sdf_writer.write(molecule)
    sdf_writer.flush()
    return sdf_output.getvalue(), smiles


def validate_complex_dataset_rows(frame: pd.DataFrame) -> pd.DataFrame:
    jobs = iter_job_records(runs_root(create=False))
    jobs_by_id = {job.run_id: job for job in jobs}
    jobs_by_code = _job_code_lookup(jobs)
    validated: list[dict[str, Any]] = []
    seen_paths: set[tuple[Path, int]] = set()
    interaction_tables: dict[str, pd.DataFrame] = {}
    for row_number, source in enumerate(frame.to_dict("records"), start=2):
        row = dict(source)
        compound = str(row.get("Compound") or "").strip()
        source_code = str(row.get("Source job") or "").strip().upper()
        evidence_text = str(
            row.get("Interaction evidence path")
            or ""
        ).strip()
        legacy_path = str(row.get("Predicted complex path") or "").strip()
        if not evidence_text and legacy_path.endswith(".complex.pdb"):
            evidence_text = legacy_path
        direct_path_text = str(
            row.get("Source pose path")
            or row.get("MD complex path")
            or (legacy_path if not evidence_text else "")
            or ""
        ).strip()
        errors: list[str] = []
        if not compound:
            errors.append("Compound is empty")
        if not source_code:
            errors.append("Source job is empty")
        if not evidence_text and not direct_path_text:
            errors.append("Source pose and interaction evidence paths are empty")
        complex_path = Path(evidence_text) if evidence_text else Path(".")
        analysis_job: JobRecord | None = None
        source_job: JobRecord | None = None
        source_pose_path: Path | None = None
        source_pose_sdf_index = 0
        source_pose_pdb = ""
        source_receptor_path: Path | None = None
        ligand_sdf_data = ""
        ligand_smiles = ""
        stereochemistry_match: bool | None = None
        pose_id = ""
        ligand_key = ""
        if evidence_text:
            try:
                analysis_job = _analysis_job_for_path(complex_path, jobs_by_id)
            except ValueError as exc:
                errors.append(str(exc))
        if analysis_job is not None:
            source_job = jobs_by_id.get(analysis_job.parent_run_id)
            if source_job is None:
                errors.append("The interaction result has no source prediction job")
            else:
                actual_code = display_job_code(
                    source_job.metadata.get("job_code"), source_job.run_id
                )
                if actual_code.upper() != source_code:
                    errors.append(
                        f"Source job is {actual_code}, not {source_code}"
                    )
            pose_id = _pose_id(complex_path)
            interactions_path = analysis_job.run_dir / "interactions.csv"
            if analysis_job.run_id not in interaction_tables:
                try:
                    interaction_tables[analysis_job.run_id] = (
                        pd.read_csv(interactions_path).fillna("")
                    )
                except (OSError, ValueError):
                    interaction_tables[analysis_job.run_id] = pd.DataFrame()
            interaction_rows = interaction_tables[analysis_job.run_id]
            matching = interaction_rows.loc[
                interaction_rows.get(
                    "pose_id", pd.Series(dtype=str)
                ).astype(str).eq(pose_id)
            ]
            if matching.empty:
                errors.append(
                    "Pose is not present in the interaction-analysis inventory"
                )
            elif not matching["compound_id"].astype(str).eq(compound).any():
                errors.append(
                    "Compound does not match the pose in interaction analysis"
                )
            if source_job is not None:
                try:
                    (
                        source_pose_path,
                        source_pose_sdf_index,
                        _source_inventory_row,
                    ) = _source_pose_reference(
                        analysis_job, source_job, pose_id
                    )
                    source_pose_pdb = _source_pose_complex_pdb(
                        source_job,
                        source_pose_path,
                        source_pose_sdf_index,
                    )
                    ligand_sdf_data, ligand_smiles = _source_pose_ligand_sdf(
                        source_job,
                        source_pose_path,
                        source_pose_sdf_index,
                        compound,
                        source_pose_pdb,
                        jobs_by_id,
                    )
                    stereochemistry_match = True
                    declared_source_path = str(
                        row.get("Source pose path") or ""
                    ).strip()
                    if (
                        declared_source_path
                        and Path(declared_source_path).expanduser().resolve()
                        != source_pose_path
                    ):
                        errors.append(
                            "Source pose path does not match the immutable "
                            "source-job pose"
                        )
                    if source_job.task_group == "docking":
                        source_receptor_path = _source_receptor(source_job).resolve()
                        declared_receptor_path = str(
                            row.get("Source receptor path") or ""
                        ).strip()
                        if (
                            declared_receptor_path
                            and Path(declared_receptor_path).expanduser().resolve()
                            != source_receptor_path
                        ):
                            errors.append(
                                "Source receptor path does not match the "
                                "immutable source-job receptor"
                            )
                except ValueError as exc:
                    if "stereochemistry" in str(exc).lower():
                        stereochemistry_match = False
                    errors.append(str(exc))
        elif direct_path_text:
            try:
                (
                    source_job,
                    source_pose_path,
                    source_pose_sdf_index,
                    pose_id,
                ) = _direct_source_pose_reference(row, jobs_by_id)
                source_pose_pdb = _source_pose_complex_pdb(
                    source_job, source_pose_path, source_pose_sdf_index
                )
                ligand_sdf_data, ligand_smiles = _source_pose_ligand_sdf(
                    source_job,
                    source_pose_path,
                    source_pose_sdf_index,
                    compound,
                    source_pose_pdb,
                    jobs_by_id,
                )
                stereochemistry_match = True
                if source_job.task_group == "docking":
                    source_receptor_path = _source_receptor(source_job).resolve()
                    declared_receptor_path = str(
                        row.get("Source receptor path") or ""
                    ).strip()
                    if (
                        declared_receptor_path
                        and Path(declared_receptor_path).expanduser().resolve()
                        != source_receptor_path
                    ):
                        errors.append(
                            "Source receptor path does not match the immutable "
                            "source-job receptor"
                        )
            except ValueError as exc:
                if "stereochemistry" in str(exc).lower():
                    stereochemistry_match = False
                errors.append(str(exc))
        resolved_path = (
            complex_path.expanduser().resolve() if evidence_text else None
        )
        if source_pose_pdb:
            try:
                ligands = parse_bound_ligands(source_pose_pdb)
            except (OSError, ValueError) as exc:
                errors.append(f"Could not read source prediction complex: {exc}")
                ligands = []
            if not ligands:
                errors.append("Source prediction contains no MD-selectable ligand")
            elif len(ligands) > 1:
                errors.append(
                    "Source prediction contains multiple ligands; explicit ligand "
                    "identity is required"
                )
            else:
                ligand_key = str(ligands[0].get("key") or "")
        uniqueness_path = resolved_path or source_pose_path
        uniqueness_key = (
            uniqueness_path,
            source_pose_sdf_index,
        ) if uniqueness_path is not None else None
        if uniqueness_key is not None and uniqueness_key in seen_paths:
            errors.append("Interaction evidence path is duplicated")
        if uniqueness_key is not None:
            seen_paths.add(uniqueness_key)
        code_matches = jobs_by_code.get(source_code, [])
        if source_code and len(code_matches) > 1 and source_job is None:
            errors.append("Source job code is ambiguous")
        validated.append({
            **row,
            "Row": row_number,
            "Resolved source run ID": source_job.run_id if source_job else "",
            "Resolved analysis run ID": (
                analysis_job.run_id if analysis_job else ""
            ),
            "Resolved pose ID": pose_id,
            "Resolved source pose path": str(source_pose_path or ""),
            "Resolved source pose SDF index": source_pose_sdf_index,
            "Resolved source receptor path": str(source_receptor_path or ""),
            "Analysis evidence path": str(resolved_path or ""),
            "Resolved ligand key": ligand_key,
            "Resolved ligand SMILES": ligand_smiles,
            "Source stereochemistry preserved": stereochemistry_match,
            "Stereochemistry policy version": (
                STEREOCHEMISTRY_POLICY_VERSION
            ),
            "Ligand SDF SHA256": (
                hashlib.sha256(ligand_sdf_data.encode()).hexdigest()
                if ligand_sdf_data
                else ""
            ),
            "Complex SHA256": (
                hashlib.sha256(source_pose_pdb.encode()).hexdigest()
                if source_pose_pdb
                else ""
            ),
            "Validation": "Valid" if not errors else "; ".join(errors),
            "Import": not errors,
        })
    return pd.DataFrame(validated)


def create_complex_dataset(
    *,
    name: str,
    source_filename: str,
    source_bytes: bytes,
    sheet_name: str,
    selected_rows: pd.DataFrame,
    selection_metadata: dict[str, Any] | None = None,
) -> JobRecord:
    if selected_rows.empty:
        raise ValueError("Select at least one valid complex")
    invalid = selected_rows.loc[selected_rows["Validation"].ne("Valid")]
    if not invalid.empty:
        raise ValueError("Every selected complex must pass provenance validation")
    root = runs_root()
    jobs = iter_job_records(root)
    jobs_by_id = {job.run_id: job for job in jobs}
    dataset_id = str(uuid4())
    dataset_dir = root / COMPLEX_DATASET_TASK_GROUP / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=False)
    dataset_name = str(name).strip() or Path(source_filename).stem
    now = _utc_now_iso()
    workbook_path = dataset_dir / _safe_name(
        source_filename, "complex-selection.xlsx"
    )
    workbook_path.write_bytes(source_bytes)
    normalized_path = dataset_dir / "selected_complexes.csv"
    selected_rows.to_csv(normalized_path, index=False)
    child_ids: list[str] = []
    created_children: list[Path] = []
    try:
        for row in selected_rows.to_dict("records"):
            source_run_id = str(row["Resolved source run ID"])
            analysis_run_id = str(row.get("Resolved analysis run ID") or "")
            source_job = jobs_by_id[source_run_id]
            analysis_job = jobs_by_id.get(analysis_run_id)
            source_path = Path(str(row["Resolved source pose path"])).resolve()
            source_sdf_index = int(row["Resolved source pose SDF index"])
            if analysis_job is not None:
                resolved_path, resolved_index, _ = _source_pose_reference(
                    analysis_job,
                    source_job,
                    str(row["Resolved pose ID"]),
                )
                if (
                    resolved_path != source_path
                    or resolved_index != source_sdf_index
                ):
                    raise ValueError(
                        "Validated source-pose selection changed before import"
                    )
            pdb_data = _source_pose_complex_pdb(
                source_job, source_path, source_sdf_index
            )
            ligand_sdf_data, ligand_smiles = _source_pose_ligand_sdf(
                source_job,
                source_path,
                source_sdf_index,
                str(row["Compound"]),
                pdb_data,
                jobs_by_id,
            )
            ligands = parse_bound_ligands(pdb_data)
            if not ligands:
                raise ValueError(
                    f"{row['Compound']} complex contains no selectable ligand"
                )
            ligand = ligands[0]
            child_id = str(uuid4())
            child_dir = root / SELECTED_COMPLEX_TASK_GROUP / child_id
            child_dir.mkdir(parents=True, exist_ok=False)
            created_children.append(child_dir)
            complex_path = child_dir / "selected_complex.pdb"
            complex_path.write_text(pdb_data)
            ligand_path = child_dir / "selected_ligand.sdf"
            ligand_path.write_text(ligand_sdf_data)
            selection_path = child_dir / "selection.json"
            selection_payload = {
                str(key): value
                for key, value in row.items()
                if str(key) != "Import"
            }
            _write_json(selection_path, selection_payload)
            source_target_id = str(
                source_job.metadata.get("prepared_target_run_id")
                or source_job.metadata.get("source_target_run_id")
                or source_job.parent_run_id
                or ""
            )
            pdb_id = str(
                source_job.metadata.get("pdb_id")
                or source_job.metadata.get("target_pdb_id")
                or "PREDICTED"
            )
            metadata = {
                "schema_version": JOB_SCHEMA_VERSION,
                "run_id": child_id,
                "job_code": short_job_code(child_id),
                "job_type": SELECTED_COMPLEX_TASK_GROUP,
                "workflow": "selected_complex",
                "status": "completed",
                "created_at": now,
                "updated_at": now,
                "completed_at": now,
                "parent_run_id": source_run_id,
                "source_structure_run_id": source_run_id,
                "source_target_run_id": source_target_id,
                "prepared_target_run_id": source_target_id,
                "source_analysis_run_id": analysis_run_id,
                "source_pose_id": str(row["Resolved pose ID"]),
                "source_pose_artifact_path": str(source_path),
                "source_pose_sdf_index": source_sdf_index,
                "source_receptor_path": str(
                    _source_receptor(source_job).resolve()
                    if source_job.task_group == "docking"
                    else ""
                ),
                "interaction_evidence_path": str(
                    row.get("Analysis evidence path")
                    or row.get("Predicted complex path")
                    or ""
                ),
                "source_prediction_job_code": str(row["Source job"]),
                "complex_dataset_run_id": dataset_id,
                "complex_dataset_name": dataset_name,
                "compound_id": str(row["Compound"]),
                "compound_name": str(row.get("Compound name") or ""),
                "ligand_id": str(row["Compound"]),
                "ligand_label": (
                    f"{row.get('Compound name')} ({row['Compound']})"
                    if str(row.get("Compound name") or "").strip()
                    else str(row["Compound"])
                ),
                "ligand_key": str(ligand.get("key") or ""),
                "ligand_smiles": ligand_smiles,
                "pdb_id": pdb_id,
                "engine": str(
                    row.get("Prediction engine") or source_job.tool or ""
                ),
                "replicate": row.get("Replicate", ""),
                "prediction": str(row.get("Prediction") or ""),
                "gnina_pose_selection": str(
                    row.get("GNINA pose selection") or ""
                ),
                "selection_status": str(row.get("Selection status") or ""),
                "selected_rank": row.get("Selected rank", ""),
                "selection_origin": str(row.get("Selection origin") or ""),
                "automatic_eligibility_checks_passed": row.get(
                    "Automatic eligibility checks passed", ""
                ),
                "manual_override_reasons": str(
                    row.get("Manual override reasons") or ""
                ),
                "reference_similarity_percent": row.get(
                    "Reference similarity (%)", ""
                ),
                "selection_score_percent": row.get(
                    "Selection score (%)", ""
                ),
                "required_interactions_met": row.get(
                    "Required interactions met", ""
                ),
                "matched_reference_interactions": str(
                    row.get("Matched reference interactions") or ""
                ),
                "missing_required_interactions": str(
                    row.get("Missing required interactions") or ""
                ),
                "reference_ligand_bend_index": row.get(
                    "Reference ligand bend index", ""
                ),
                "candidate_pose_bend_index": row.get(
                    "Candidate pose bend index", ""
                ),
                "bend_penalty_percentage_points": row.get(
                    "Bend penalty (percentage points)", ""
                ),
                "launch_campaign_id": str(
                    source_job.metadata.get("launch_campaign_id") or ""
                ),
                "launch_campaign_label": str(
                    source_job.metadata.get("launch_campaign_label") or ""
                ),
            }
            _write_json(child_dir / "metadata.json", metadata)
            _write_json(
                child_dir / "result.json",
                {
                    "success": True,
                    "prepared_complex": complex_path.name,
                    "prepared_ligand_set": ligand_path.name,
                    "compound_id": str(row["Compound"]),
                    "source_prediction_run_id": source_run_id,
                    "source_analysis_run_id": analysis_run_id,
                    "source_pose_id": str(row["Resolved pose ID"]),
                },
            )
            write_artifact_manifest(
                child_dir,
                [
                    ArtifactRef.from_path(
                        child_dir,
                        complex_path,
                        "prepared_complex",
                        role="md_selected_complex",
                        label=(
                            f"{row['Compound']} · "
                            f"{row.get('Prediction engine') or source_job.tool} · "
                            f"{row.get('Prediction') or row['Resolved pose ID']}"
                        ),
                        metadata={
                            "complex_dataset_run_id": dataset_id,
                            "source_prediction_run_id": source_run_id,
                            "source_analysis_run_id": analysis_run_id,
                            "source_pose_id": str(row["Resolved pose ID"]),
                            "source_pose_artifact_path": str(source_path),
                            "source_pose_sdf_index": source_sdf_index,
                            "compound_id": str(row["Compound"]),
                            "selection_status": str(
                                row.get("Selection status") or ""
                            ),
                        },
                    ),
                    ArtifactRef.from_path(
                        child_dir,
                        ligand_path,
                        "prepared_ligand_set",
                        role="md_selected_ligand",
                        label=f"{row['Compound']} selected-pose ligand topology",
                        metadata={
                            "complex_dataset_run_id": dataset_id,
                            "source_prediction_run_id": source_run_id,
                            "source_pose_artifact_path": str(source_path),
                            "source_pose_sdf_index": source_sdf_index,
                            "compound_id": str(row["Compound"]),
                            "smiles": ligand_smiles,
                            "coordinates": "exact selected prediction pose",
                        },
                    ),
                    ArtifactRef.from_path(
                        child_dir,
                        selection_path,
                        "complex_selection_record",
                        role="selection_provenance",
                        label=f"{row['Compound']} selection evidence",
                    ),
                ],
            )
            child_ids.append(child_id)
        _write_json(
            dataset_dir / "metadata.json",
            {
                "schema_version": JOB_SCHEMA_VERSION,
                "run_id": dataset_id,
                "job_code": short_job_code(dataset_id),
                "job_type": COMPLEX_DATASET_TASK_GROUP,
                "workflow": "complex_dataset_import",
                "status": "completed",
                "created_at": now,
                "updated_at": now,
                "completed_at": now,
                "dataset_name": dataset_name,
                "source_filename": source_filename,
                "source_sheet": sheet_name,
                "complex_count": len(child_ids),
                "selected_complex_run_ids": child_ids,
                "complex_dataset_schema_version": (
                    COMPLEX_DATASET_SCHEMA_VERSION
                ),
                "selection_provenance": dict(selection_metadata or {}),
            },
        )
        _write_json(
            dataset_dir / "result.json",
            {
                "success": True,
                "dataset_name": dataset_name,
                "complex_count": len(child_ids),
                "selected_complex_run_ids": child_ids,
            },
        )
        write_artifact_manifest(
            dataset_dir,
            [
                ArtifactRef.from_path(
                    dataset_dir,
                    workbook_path,
                    "complex_selection_workbook",
                    role="immutable_source",
                ),
                ArtifactRef.from_path(
                    dataset_dir,
                    normalized_path,
                    "complex_selection_table",
                    role="normalized_selection",
                ),
            ],
        )
    except Exception:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        for child_dir in created_children:
            shutil.rmtree(child_dir, ignore_errors=True)
        raise
    return JobRecord.load(dataset_dir, task_group=COMPLEX_DATASET_TASK_GROUP)


def complex_dataset_jobs() -> list[JobRecord]:
    return sorted(
        (
            job
            for job in iter_job_records(
                runs_root(create=False),
                task_groups=(COMPLEX_DATASET_TASK_GROUP,),
            )
            if job.status == "completed"
        ),
        key=lambda job: str(job.created_at or ""),
        reverse=True,
    )


def selected_complex_jobs() -> list[JobRecord]:
    return sorted(
        (
            job
            for job in iter_job_records(
                runs_root(create=False),
                task_groups=(SELECTED_COMPLEX_TASK_GROUP,),
            )
            if job.status == "completed"
        ),
        key=lambda job: (
            str(job.metadata.get("complex_dataset_name") or ""),
            int(job.metadata.get("selected_rank") or 0),
        ),
    )
