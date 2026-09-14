from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import (
    JOB_SCHEMA_VERSION,
    JobRecord,
    iter_job_records,
    short_job_code,
)
from mn_ligand.runtime import runs_root


PHARMACOPHORE_SCHEMA_VERSION = 1
PHARMACOPHORE_TASK_GROUP = "pharmacophore-hypotheses"
FEATURE_TYPES = (
    "Aromatic",
    "HydrogenDonor",
    "HydrogenAcceptor",
    "PositiveIon",
    "NegativeIon",
    "Hydrophobic",
    "Halogen",
    "ExcludedVolume",
)
OMTRA_XYZ_ELEMENTS = {
    "Aromatic": "P",
    "HydrogenDonor": "S",
    "HydrogenAcceptor": "F",
    "PositiveIon": "N",
    "NegativeIon": "O",
    "Hydrophobic": "C",
    "Halogen": "Cl",
}
PGMG_TYPES = {
    "Aromatic": "AROM",
    "HydrogenDonor": "HDON",
    "HydrogenAcceptor": "HACC",
    "PositiveIon": "POSC",
    # The canonical/UI type intentionally does not claim ring membership.
    # PGMG supports comma-separated multi-hot types, so preserve that
    # uncertainty instead of incorrectly forcing every hydrophobe to HYBL
    # (ring) or LHYBL (non-ring).
    "Hydrophobic": "HYBL,LHYBL",
}
RDKIT_FAMILY_MAP = {
    "Aromatic": "Aromatic",
    "Donor": "HydrogenDonor",
    "Acceptor": "HydrogenAcceptor",
    "PosIonizable": "PositiveIon",
    "NegIonizable": "NegativeIon",
    "Hydrophobe": "Hydrophobic",
    "LumpedHydrophobe": "Hydrophobic",
    "Halogen": "Halogen",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class PharmacophoreFeature:
    feature_id: str
    feature_type: str
    x: float
    y: float
    z: float
    radius: float = 1.0
    enabled: bool = True
    required: bool = True
    direction: tuple[float, float, float] | None = None
    source_atom_indices: tuple[int, ...] = ()
    source_residues: tuple[str, ...] = ()
    source: str = "manual"
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.feature_type not in FEATURE_TYPES:
            raise ValueError(f"Unsupported pharmacophore feature: {self.feature_type}")
        if float(self.radius) <= 0:
            raise ValueError("Pharmacophore feature radius must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, index: int = 1) -> PharmacophoreFeature:
        direction = payload.get("direction")
        return cls(
            feature_id=str(payload.get("feature_id") or f"feature-{index:03d}"),
            feature_type=str(payload.get("feature_type") or payload.get("name") or ""),
            x=float(payload.get("x")),
            y=float(payload.get("y")),
            z=float(payload.get("z")),
            radius=float(payload.get("radius") or 1.0),
            enabled=bool(payload.get("enabled", True)),
            required=bool(payload.get("required", True)),
            direction=(
                tuple(float(value) for value in direction)
                if isinstance(direction, (list, tuple)) and len(direction) == 3
                else None
            ),
            source_atom_indices=tuple(int(value) for value in payload.get("source_atom_indices") or ()),
            source_residues=tuple(str(value) for value in payload.get("source_residues") or ()),
            source=str(payload.get("source") or "manual"),
            notes=str(payload.get("notes") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["direction"] = list(self.direction) if self.direction is not None else None
        payload["source_atom_indices"] = list(self.source_atom_indices)
        payload["source_residues"] = list(self.source_residues)
        return payload


def load_pharmacophore(path: Path) -> list[PharmacophoreFeature]:
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version") or 0) != PHARMACOPHORE_SCHEMA_VERSION:
        raise ValueError("Unsupported pharmacophore hypothesis schema")
    return [
        PharmacophoreFeature.from_dict(item, index=index)
        for index, item in enumerate(payload.get("features") or (), start=1)
    ]


def _load_first_molecule(path: Path) -> Chem.Mol:
    suffix = path.suffix.lower()
    molecule: Chem.Mol | None = None
    if suffix == ".sdf":
        molecule = next((item for item in Chem.SDMolSupplier(str(path), removeHs=False) if item), None)
    elif suffix == ".mol":
        molecule = Chem.MolFromMolFile(str(path), removeHs=False)
    elif suffix in {".pdb", ".ent"}:
        molecule = Chem.MolFromPDBFile(str(path), removeHs=False)
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError("A coordinate-bearing SDF, MOL, or ligand-only PDB is required")
    return molecule


def extract_ligand_pharmacophore(path: Path) -> list[PharmacophoreFeature]:
    """Create an editable RDKit BaseFeatures hypothesis from one 3D ligand."""
    molecule = _load_first_molecule(path)
    factory = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    features: list[PharmacophoreFeature] = []
    for feature in factory.GetFeaturesForMol(molecule):
        feature_type = RDKIT_FAMILY_MAP.get(str(feature.GetFamily()))
        if feature_type is None:
            continue
        position = feature.GetPos()
        features.append(
            PharmacophoreFeature(
                feature_id=f"feature-{len(features) + 1:03d}",
                feature_type=feature_type,
                x=float(position.x),
                y=float(position.y),
                z=float(position.z),
                source_atom_indices=tuple(int(value) for value in feature.GetAtomIds()),
                source="rdkit-basefeatures",
                metadata={"rdkit_family": str(feature.GetFamily()), "rdkit_type": str(feature.GetType())},
            )
        )
    if not features:
        raise ValueError("No supported pharmacophore features were detected")
    return features


def extract_interaction_supported_pharmacophore(
    interaction_job: JobRecord,
    *,
    pose_id: str | None = None,
) -> tuple[list[PharmacophoreFeature], dict[str, Any]]:
    """Annotate ligand features with pose-level PLIP/PandaMap evidence.

    Current normalized interaction rows do not expose a stable cross-engine
    ligand-atom map. Consequently this method makes no atom-level claim: it
    marks feature *classes* supported by the observed interaction inventory and
    preserves that limitation in provenance for user review.
    """
    if interaction_job.status != "completed":
        raise ValueError("A completed interaction-analysis job is required")
    interactions_path = interaction_job.run_dir / "interactions.csv"
    if not interactions_path.is_file():
        raise FileNotFoundError("The interaction job has no normalized interaction table")
    jobs = {job.run_id: job for job in iter_job_records(runs_root())}
    source_job = jobs.get(str(interaction_job.parent_run_id or ""))
    if source_job is None:
        raise FileNotFoundError("The immutable interaction source job is unavailable")

    ligand_path: Path | None = None
    table_path = interaction_job.run_dir / "input" / "interaction_inputs.csv"
    input_rows: list[dict[str, str]] = []
    if table_path.is_file():
        with table_path.open(newline="", errors="replace") as handle:
            input_rows = list(csv.DictReader(handle))
    if pose_id:
        input_rows = [
            row for row in input_rows
            if str(row.get("pose_id") or "") == str(pose_id)
        ]
    if len(input_rows) != 1:
        raise ValueError(
            "Select exactly one analyzed complex or pose for a pharmacophore hypothesis"
        )
    selected_input = input_rows[0]
    topology_path = interaction_job.run_dir / str(
        selected_input.get("ligand_topology") or ""
    )
    if (
        topology_path.is_file()
        and topology_path.suffix.lower() in {".sdf", ".mol"}
    ):
        ligand_path = topology_path
    if ligand_path is None:
        pose_path = interaction_job.run_dir / str(
            selected_input.get("mol_pred") or ""
        )
        if pose_path.is_file() and pose_path.suffix.lower() in {".sdf", ".mol"}:
            ligand_path = pose_path
    if (
        ligand_path is None
        and str(selected_input.get("source_kind") or "")
        == "prepared target complex"
    ):
        raise FileNotFoundError(
            "This target complex has no unambiguous ligand-topology artifact "
            "for the selected residue; interaction analysis remains valid, but "
            "pharmacophore extraction requires an exact matching SDF or MOL"
        )
    if source_job.artifact_manifest is not None:
        for artifact_type in ("prepared_ligand_set", "compound_set"):
            if ligand_path is not None:
                break
            for artifact in source_job.artifact_manifest.by_type(artifact_type):
                candidate = artifact.resolve(source_job.run_dir, must_exist=True)
                if candidate is not None and candidate.suffix.lower() in {".sdf", ".mol"}:
                    ligand_path = candidate
                    break
            if ligand_path is not None:
                break
    if ligand_path is None:
        candidate = source_job.run_dir / str(
            selected_input.get("source_artifact_path") or ""
        )
        if candidate.is_file() and candidate.suffix.lower() in {".sdf", ".mol"}:
            ligand_path = candidate
    if ligand_path is None:
        raise FileNotFoundError(
            "No coordinate-bearing ligand with retained bond topology could be resolved"
        )

    with interactions_path.open(newline="", errors="replace") as handle:
        interaction_rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("pose_id") or "")
            == str(selected_input.get("pose_id") or "")
        ]
    observed = {
        str(row.get("interaction_type") or "").strip().lower()
        for row in interaction_rows
        if str(row.get("interaction_type") or "").strip()
    }
    supported_types: set[str] = set()
    for kind in observed:
        if "hydrogen" in kind or "hbond" in kind:
            supported_types.update({"HydrogenDonor", "HydrogenAcceptor"})
        if "hydrophob" in kind or "van der waals" in kind:
            supported_types.add("Hydrophobic")
        if "pi" in kind or "aromat" in kind:
            supported_types.add("Aromatic")
        if "salt" in kind or "cation" in kind or "ionic" in kind:
            supported_types.update({"PositiveIon", "NegativeIon"})
        if "halogen" in kind or "xb" in kind:
            supported_types.add("Halogen")
    residues = tuple(
        sorted(
            {
                (
                    f"{row.get('protein_chain', '')}:"
                    f"{row.get('protein_residue_name', '')}"
                    f"{row.get('protein_residue_number', '')}"
                )
                for row in interaction_rows
                if row.get("protein_residue_number")
            }
        )
    )
    engine = str(
        interaction_job.metadata.get("interaction_engine")
        or interaction_job.tool
        or ""
    )
    features = [
        replace(
            item,
            required=item.feature_type in supported_types,
            source=f"rdkit-basefeatures+{engine.lower()}-pose-evidence",
            source_residues=residues,
            metadata={
                **item.metadata,
                "interaction_job_run_id": interaction_job.run_id,
                "interaction_engine": engine,
                "interaction_types": sorted(observed),
                "association_scope": "pose-level feature-class evidence",
                "atom_level_interaction_mapping": False,
            },
        )
        for item in extract_ligand_pharmacophore(ligand_path)
    ]
    return features, {
        "interaction_job_run_id": interaction_job.run_id,
        "pose_id": str(selected_input.get("pose_id") or ""),
        "compound_id": str(selected_input.get("compound_id") or ""),
        "interaction_engine": engine,
        "interaction_count": len(interaction_rows),
        "interaction_types": sorted(observed),
        "contacted_residues": list(residues),
        "association_scope": "pose-level feature-class evidence",
        "atom_level_interaction_mapping": False,
        "ligand_source_job_run_id": source_job.run_id,
    }


def extract_plip_residue_directed_pharmacophore(
    interaction_job: JobRecord,
    *,
    pose_id: str,
    protein_chain: str,
    protein_residue_name: str,
    protein_residue_number: int,
    protein_atom_name: str,
    ligand_feature_type: str = "HydrogenAcceptor",
    interaction_kind: str = "hydrogen_bond_target_donor",
    target_distance_angstrom: float = 2.8,
    target_radius_angstrom: float = 1.0,
    retain_same_residue_backbone_contact: bool = False,
) -> tuple[list[PharmacophoreFeature], dict[str, Any]]:
    """Build an atom-resolved PLIP hypothesis with one required target contact.

    PLIP's observed contacts are retained as reference features.  The requested
    side-chain feature is placed along the vector from the selected protein
    atom toward the bound ligand centroid; it is a design constraint, not an
    assertion that the reference ligand already makes that contact.
    """
    if ligand_feature_type not in {
        "HydrogenAcceptor",
        "HydrogenDonor",
        "PositiveIon",
        "NegativeIon",
        "Hydrophobic",
        "Aromatic",
        "Halogen",
    }:
        raise ValueError(
            "Unsupported residue-directed ligand feature type: "
            f"{ligand_feature_type}"
        )
    distance = float(target_distance_angstrom)
    radius = float(target_radius_angstrom)
    if distance <= 0 or radius <= 0:
        raise ValueError("Target distance and radius must be positive")
    base_features, provenance = extract_interaction_supported_pharmacophore(
        interaction_job, pose_id=pose_id
    )
    if str(provenance.get("interaction_engine") or "").upper() != "PLIP":
        raise ValueError(
            "Atom-resolved residue-directed extraction currently requires PLIP"
        )

    interactions_path = interaction_job.run_dir / "interactions.csv"
    with interactions_path.open(newline="", errors="replace") as handle:
        interaction_rows = [
            row for row in csv.DictReader(handle)
            if str(row.get("pose_id") or "") == str(pose_id)
        ]
    native_dir = interaction_job.run_dir / "native" / str(pose_id)
    protonated = native_dir / f"{pose_id}_protonated.pdb"
    if not protonated.is_file():
        matches = sorted(native_dir.glob("*_protonated.pdb"))
        protonated = matches[0] if len(matches) == 1 else protonated
    if not protonated.is_file():
        raise FileNotFoundError("PLIP protonated complex is unavailable")

    atoms_by_serial: dict[int, tuple[float, float, float]] = {}
    ligand_coordinates: list[tuple[float, float, float]] = []
    target_atom: tuple[float, float, float] | None = None
    target_chain = str(protein_chain).strip()
    target_resname = str(protein_residue_name).strip().upper()
    target_atom_name = str(protein_atom_name).strip().upper()
    for line in protonated.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            serial = int(line[6:11])
            coordinates = (
                float(line[30:38]), float(line[38:46]), float(line[46:54])
            )
        except ValueError:
            continue
        atoms_by_serial[serial] = coordinates
        if line.startswith("HETATM") and line[21:22].strip() == "Z":
            ligand_coordinates.append(coordinates)
        if (
            line.startswith("ATOM  ")
            and line[21:22].strip() == target_chain
            and line[17:20].strip().upper() == target_resname
            and line[12:16].strip().upper() == target_atom_name
        ):
            try:
                matches_residue = int(line[22:26]) == int(protein_residue_number)
            except ValueError:
                matches_residue = False
            if matches_residue:
                target_atom = coordinates
    if target_atom is None:
        raise ValueError(
            "Required protein atom was not found in PLIP's author-numbered complex: "
            f"{target_chain}:{target_resname}{protein_residue_number}:{target_atom_name}"
        )
    if not ligand_coordinates:
        raise ValueError("PLIP's analyzed complex contains no ligand coordinates")
    prepared_atom: dict[str, Any] = {}
    mapping_path = native_dir / "residue_numbering.json"
    try:
        mapping_report = json.loads(mapping_path.read_text())
    except (OSError, TypeError, ValueError):
        mapping_report = {}
    mapping_rows = (
        mapping_report.get("predicted_to_reference")
        or mapping_report.get("prepared_to_reference")
        or ()
    )
    for mapping in mapping_rows:
        if (
            isinstance(mapping, dict)
            and str(mapping.get("reference_chain") or "") == target_chain
            and str(mapping.get("reference_residue_number") or "")
            == str(int(protein_residue_number))
            and str(mapping.get("reference_insertion_code") or "") == ""
        ):
            prepared_atom = {
                "chain": str(
                    mapping.get("predicted_chain")
                    or mapping.get("prepared_chain")
                    or ""
                ),
                "residue_name": target_resname,
                "residue_number": int(
                    mapping.get("predicted_residue_number")
                    or mapping.get("prepared_residue_number")
                ),
                "atom_name": target_atom_name,
                "coordinates": list(target_atom),
                "numbering": "prepared",
                "sidechain": True,
            }
            break

    observed_features: list[PharmacophoreFeature] = []
    excluded_reference_features: list[dict[str, Any]] = []
    seen: dict[tuple[str, float, float, float], int] = {}

    def add_observed(
        feature_type: str,
        coordinates: tuple[float, float, float],
        row: dict[str, str],
        *,
        source_atom_serial: int | None = None,
    ) -> None:
        key = (
            feature_type,
            round(coordinates[0], 3),
            round(coordinates[1], 3),
            round(coordinates[2], 3),
        )
        residue = (
            f"{row.get('protein_chain', '')}:"
            f"{row.get('protein_residue_name', '')}"
            f"{row.get('protein_residue_number', '')}"
        )
        evidence = {
            "protein_residue": residue,
            "interaction_type": row.get("interaction_type", ""),
            "distance_angstrom": row.get("distance_angstrom", ""),
            "angle_degree": row.get("angle_degree", ""),
        }
        if key in seen:
            index = seen[key]
            existing = observed_features[index]
            residues = tuple(sorted({*existing.source_residues, residue}))
            interaction_evidence = list(
                existing.metadata.get("interaction_evidence") or ()
            )
            interaction_evidence.append(evidence)
            observed_features[index] = replace(
                existing,
                source_residues=residues,
                notes=(
                    f"Observed {feature_type} evidence with "
                    f"{', '.join(residues)}; retained as a reference feature."
                ),
                metadata={
                    **existing.metadata,
                    "interaction_evidence": interaction_evidence,
                },
            )
            return
        seen[key] = len(observed_features)
        observed_features.append(
            PharmacophoreFeature(
                feature_id=f"plip-observed-{len(observed_features) + 1:03d}",
                feature_type=feature_type,
                x=coordinates[0],
                y=coordinates[1],
                z=coordinates[2],
                radius=1.0,
                required=False,
                source_atom_indices=(
                    (source_atom_serial,) if source_atom_serial is not None else ()
                ),
                source_residues=(residue,),
                source="plip-atom-resolved-observation",
                notes=(
                    f"Observed {row.get('interaction_type', '')} with {residue}; "
                    "retained as a reference interaction."
                ),
                metadata={
                    "interaction_job_run_id": interaction_job.run_id,
                    "interaction_type": row.get("interaction_type", ""),
                    "distance_angstrom": row.get("distance_angstrom", ""),
                    "angle_degree": row.get("angle_degree", ""),
                    "observed": True,
                    "required_target_contact": False,
                    "interaction_evidence": [evidence],
                },
            )
        )

    for row in interaction_rows:
        try:
            native = json.loads(str(row.get("native_fields_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            native = {}
        kind = str(row.get("interaction_type") or "").lower()
        same_target_residue = (
            str(row.get("protein_chain") or "") == target_chain
            and str(row.get("protein_residue_name") or "").upper()
            == target_resname
            and str(row.get("protein_residue_number") or "")
            == str(int(protein_residue_number))
        )
        is_backbone_hydrogen_bond = (
            ("hydrogen" in kind or "hbond" in kind)
            and str(native.get("sidechain") or "").lower() != "true"
        )
        if (
            same_target_residue
            and is_backbone_hydrogen_bond
            and not retain_same_residue_backbone_contact
        ):
            excluded_reference_features.append(
                {
                    "interaction_type": row.get("interaction_type", ""),
                    "protein_residue": (
                        f"{target_chain}:{target_resname}"
                        f"{int(protein_residue_number)}"
                    ),
                    "distance_angstrom": row.get("distance_angstrom", ""),
                    "angle_degree": row.get("angle_degree", ""),
                    "native_fields": native,
                    "reason": (
                        "Replaced by mandatory side-chain-directed design "
                        "constraint; retained in immutable PLIP evidence only"
                    ),
                }
            )
            continue
        feature_type = ""
        serial_value: Any = None
        if "hydrophob" in kind:
            feature_type, serial_value = "Hydrophobic", native.get("ligcarbonidx")
        elif "hydrogen" in kind or "hbond" in kind:
            protein_is_donor = str(native.get("protisdon") or "").lower() == "true"
            feature_type = "HydrogenAcceptor" if protein_is_donor else "HydrogenDonor"
            serial_value = (
                native.get("acceptoridx") if protein_is_donor
                else native.get("donoridx")
            )
        elif "halogen" in kind:
            feature_type, serial_value = "Halogen", native.get("don_idx")
        if feature_type and serial_value not in {None, ""}:
            try:
                serial = int(float(str(serial_value)))
            except ValueError:
                serial = -1
            coordinates = atoms_by_serial.get(serial)
            if coordinates is not None:
                add_observed(
                    feature_type, coordinates, row, source_atom_serial=serial
                )
        elif "salt" in kind:
            ligand_group = str(native.get("lig_group") or "").lower()
            desired_type = (
                "NegativeIon" if "carboxyl" in ligand_group else "PositiveIon"
            )
            representative = next(
                (
                    item for item in base_features
                    if item.feature_type == desired_type
                ),
                None,
            )
            if representative is not None:
                add_observed(
                    desired_type,
                    (representative.x, representative.y, representative.z),
                    row,
                )

    ligand_center = tuple(
        sum(point[axis] for point in ligand_coordinates)
        / len(ligand_coordinates)
        for axis in range(3)
    )
    vector = tuple(
        ligand_center[axis] - target_atom[axis] for axis in range(3)
    )
    vector_length = math.sqrt(sum(value * value for value in vector))
    if vector_length == 0:
        raise ValueError("Ligand centroid overlaps the required target atom")
    unit = tuple(value / vector_length for value in vector)
    feature_coordinates = tuple(
        target_atom[axis] + distance * unit[axis] for axis in range(3)
    )
    residue_label = (
        f"{target_chain}:{target_resname}{int(protein_residue_number)}:"
        f"{target_atom_name}"
    )
    sidechain_observed = False
    for row in interaction_rows:
        try:
            native = json.loads(str(row.get("native_fields_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            native = {}
        if (
            str(row.get("protein_chain") or "") == target_chain
            and str(row.get("protein_residue_name") or "").upper()
            == target_resname
            and str(row.get("protein_residue_number") or "")
            == str(int(protein_residue_number))
            and str(native.get("sidechain") or "").lower() == "true"
        ):
            sidechain_observed = True
            break
    required_feature = PharmacophoreFeature(
        feature_id="required-target-contact-001",
        feature_type=ligand_feature_type,
        x=feature_coordinates[0],
        y=feature_coordinates[1],
        z=feature_coordinates[2],
        radius=radius,
        required=True,
        direction=tuple(-value for value in unit),
        source_residues=(residue_label,),
        source="residue-directed-design-constraint",
        notes=(
            f"Mandatory {interaction_kind.replace('_', ' ')} using ligand "
            f"{ligand_feature_type}, directed to the "
            f"{residue_label} side chain; inferred from target geometry and "
            "requires downstream pose-level validation."
        ),
        metadata={
            "interaction_job_run_id": interaction_job.run_id,
            "observed": False,
            "required_target_contact": True,
            "protein_atom": {
                "chain": target_chain,
                "residue_name": target_resname,
                "residue_number": int(protein_residue_number),
                "atom_name": target_atom_name,
                "coordinates": list(target_atom),
                "numbering": "author",
                "sidechain": True,
            },
            "prepared_protein_atom": prepared_atom or None,
            "ligand_role": (
                {
                    "HydrogenAcceptor": "acceptor",
                    "HydrogenDonor": "donor",
                    "PositiveIon": "positive ion",
                    "NegativeIon": "negative ion",
                    "Hydrophobic": "hydrophobic",
                    "Aromatic": "aromatic",
                    "Halogen": "halogen donor",
                }[ligand_feature_type]
            ),
            "interaction_kind": str(interaction_kind),
            "placement_method": "protein-atom-to-bound-ligand-centroid-vector",
            "target_distance_angstrom": distance,
            "reference_sidechain_contact_observed": sidechain_observed,
        },
    )
    required_contact = dict(required_feature.metadata)
    creation_metadata = {
        **provenance,
        "association_scope": "PLIP atom-resolved observations plus residue-directed constraint",
        "atom_level_interaction_mapping": True,
        "observed_feature_count": len(observed_features),
        "excluded_reference_features": excluded_reference_features,
        "same_residue_backbone_contact_retained": bool(
            retain_same_residue_backbone_contact
        ),
        "required_target_contacts": [required_contact],
        "reference_sidechain_contact_observed": sidechain_observed,
    }
    return [required_feature, *observed_features], creation_metadata


def pharmit_payload(features: Iterable[PharmacophoreFeature]) -> dict[str, Any]:
    return {
        "points": [
            {
                "name": item.feature_type,
                "x": item.x,
                "y": item.y,
                "z": item.z,
                "radius": item.radius,
                "enabled": bool(item.enabled),
            }
            for item in features
            if item.feature_type != "ExcludedVolume"
        ]
    }


def omtra_xyz(features: Iterable[PharmacophoreFeature]) -> str:
    rows = [
        (OMTRA_XYZ_ELEMENTS[item.feature_type], item.x, item.y, item.z)
        for item in features
        if item.enabled and item.feature_type in OMTRA_XYZ_ELEMENTS
    ]
    return "\n".join(
        [str(len(rows)), "mn-ligand pharmacophore hypothesis"]
        + [f"{element} {x:.6f} {y:.6f} {z:.6f}" for element, x, y, z in rows]
    ) + "\n"


def pgmg_posp(features: Iterable[PharmacophoreFeature]) -> tuple[str, list[str]]:
    """Export the PGMG coordinate format and report unsupported feature IDs."""
    rows: list[str] = []
    omitted: list[str] = []
    for item in features:
        if not item.enabled:
            continue
        pgmg_type = PGMG_TYPES.get(item.feature_type)
        if pgmg_type is None:
            omitted.append(item.feature_id)
            continue
        rows.append(f"{pgmg_type} {item.x:.6f} {item.y:.6f} {item.z:.6f}")
    if len(rows) > 8:
        raise ValueError("PGMG supports at most eight enabled pharmacophore points")
    if not rows:
        raise ValueError("The hypothesis contains no PGMG-compatible enabled features")
    return "\n".join(rows) + "\n", omitted


def create_pharmacophore_hypothesis_job(
    *,
    name: str,
    features: Iterable[PharmacophoreFeature | dict[str, Any]],
    source_job: JobRecord | None = None,
    source_artifact: ArtifactRef | None = None,
    target_artifact: ArtifactRef | None = None,
    pocket_artifact: ArtifactRef | None = None,
    creation_method: str = "manual",
    creation_metadata: dict[str, Any] | None = None,
) -> JobRecord:
    normalized = tuple(
        item if isinstance(item, PharmacophoreFeature) else PharmacophoreFeature.from_dict(item, index=index)
        for index, item in enumerate(features, start=1)
    )
    if not normalized:
        raise ValueError("A pharmacophore hypothesis requires at least one feature")
    if source_artifact is not None and source_job is not None and source_artifact.run_id != source_job.run_id:
        raise ValueError("Source artifact does not belong to its source job")
    synchronized: list[PharmacophoreFeature] = []
    for item in normalized:
        metadata = dict(item.metadata)
        protein_atom = metadata.get("protein_atom") or {}
        atom_coordinates = (
            protein_atom.get("coordinates")
            if isinstance(protein_atom, dict)
            else None
        )
        if (
            bool(metadata.get("required_target_contact"))
            and isinstance(atom_coordinates, list)
            and len(atom_coordinates) == 3
        ):
            vector = tuple(
                float(atom_coordinates[axis])
                - (item.x, item.y, item.z)[axis]
                for axis in range(3)
            )
            distance = math.sqrt(sum(value * value for value in vector))
            direction = (
                tuple(value / distance for value in vector)
                if distance > 0
                else None
            )
            metadata["target_distance_angstrom"] = distance
            metadata["constraint_geometry_source"] = "saved-feature-row"
            metadata["interaction_kind"] = {
                "HydrogenAcceptor": "hydrogen_bond_target_donor",
                "HydrogenDonor": "hydrogen_bond_target_acceptor",
                "NegativeIon": "ionic_target_positive",
                "PositiveIon": "ionic_target_negative",
                "Hydrophobic": "hydrophobic_contact",
                "Aromatic": "aromatic_pi_interaction",
                "Halogen": "halogen_bond_ligand_donor",
            }.get(
                item.feature_type,
                str(metadata.get("interaction_kind") or "user_defined"),
            )
            item = replace(item, direction=direction, metadata=metadata)
        synchronized.append(item)
    normalized = tuple(synchronized)
    provenance = dict(creation_metadata or {})
    required_contacts: list[dict[str, Any]] = []
    for item in normalized:
        if (
            not item.enabled
            or not item.required
            or not bool(item.metadata.get("required_target_contact"))
        ):
            continue
        contact = dict(item.metadata)
        contact["ligand_role"] = {
            "HydrogenAcceptor": "acceptor",
            "HydrogenDonor": "donor",
        }.get(item.feature_type, item.feature_type)
        contact["ligand_feature"] = {
            "feature_id": item.feature_id,
            "feature_type": item.feature_type,
            "coordinates": [item.x, item.y, item.z],
            "radius_angstrom": item.radius,
            "direction": (
                list(item.direction) if item.direction is not None else None
            ),
            "enabled": item.enabled,
            "required": item.required,
        }
        required_contacts.append(contact)
    provenance["required_target_contacts"] = required_contacts

    run_id = str(uuid4())
    run_dir = runs_root() / PHARMACOPHORE_TASK_GROUP / run_id
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    canonical_path = artifact_dir / "pharmacophore.json"
    table_path = artifact_dir / "pharmacophore_features.csv"
    pharmit_path = artifact_dir / "pharmit.json"
    xyz_path = artifact_dir / "omtra_pharmacophore.xyz"
    pgmg_path = artifact_dir / "pgmg_pharmacophore.posp"
    contract_path = artifact_dir / "conditioning_contract.json"

    payload = {
        "kind": "pharmacophore_hypothesis",
        "schema_version": PHARMACOPHORE_SCHEMA_VERSION,
        "name": str(name).strip() or "Pharmacophore hypothesis",
        "creation_method": str(creation_method),
        "creation_metadata": provenance,
        "features": [item.to_dict() for item in normalized],
    }
    _write_json(canonical_path, payload)
    with table_path.open("w", newline="") as handle:
        fieldnames = [
            "feature_id", "feature_type", "x", "y", "z", "radius", "enabled",
            "required", "direction", "source_atom_indices", "source_residues", "source", "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in normalized:
            row = item.to_dict()
            row["direction"] = json.dumps(row["direction"])
            row["source_atom_indices"] = json.dumps(row["source_atom_indices"])
            row["source_residues"] = json.dumps(row["source_residues"])
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    _write_json(pharmit_path, pharmit_payload(normalized))
    xyz_path.write_text(omtra_xyz(normalized))
    pgmg_omitted: list[str] = []
    try:
        pgmg_text, pgmg_omitted = pgmg_posp(normalized)
        pgmg_path.write_text(pgmg_text)
    except ValueError:
        pgmg_path = Path()
    enabled_ids = [item.feature_id for item in normalized if item.enabled]
    required_ids = [
        item.feature_id
        for item in normalized
        if item.enabled and item.required
    ]
    optional_conditioned_ids = [
        item.feature_id
        for item in normalized
        if item.enabled and not item.required
    ]
    _write_json(
        contract_path,
        {
            "kind": "pharmacophore_conditioning_contract",
            "schema_version": 1,
            "enabled_native_conditioning_feature_ids": enabled_ids,
            "required_downstream_validation_feature_ids": required_ids,
            "optional_but_native_conditioned_feature_ids": (
                optional_conditioned_ids
            ),
            "semantics": {
                "enabled": (
                    "The feature is passed to compatible native engines."
                ),
                "required": (
                    "The feature is a campaign validation requirement. Native "
                    "OMTRA and PGMG formats do not encode priority or mandatory "
                    "flags, so required and non-required enabled points are "
                    "conditioned equally by those engines."
                ),
                "disabled": (
                    "The feature remains in provenance but is not exported as "
                    "native conditioning."
                ),
            },
            "engine_formats": {
                "omtra": {
                    "roles": ["pharmit_json", "omtra_xyz"],
                    "conditioned_feature_ids": enabled_ids,
                },
                "pgmg": {
                    "role": "pgmg_posp",
                    "conditioned_feature_ids": [
                        item.feature_id
                        for item in normalized
                        if item.enabled and item.feature_type in PGMG_TYPES
                    ],
                    "omitted_feature_ids": pgmg_omitted,
                    "maximum_features": 8,
                },
                "pocketflow": {
                    "role": None,
                    "conditioned_feature_ids": [],
                    "note": (
                        "PocketFlow consumes pocket geometry only; all "
                        "pharmacophore requirements are downstream filters."
                    ),
                },
            },
        },
    )

    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "pharmacophore_hypothesis",
        "workflow": "pharmacophore_hypothesis",
        "status": "completed",
        "name": payload["name"],
        "creation_method": creation_method,
        "creation_metadata": provenance,
        "feature_count": len(normalized),
        "enabled_feature_count": sum(item.enabled for item in normalized),
        "parent_run_id": source_job.run_id if source_job is not None else "",
        "prepared_target_run_id": target_artifact.run_id if target_artifact is not None else "",
        "pocket_run_id": pocket_artifact.run_id if pocket_artifact is not None else "",
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "source_artifact": source_artifact.to_dict() if source_artifact else None,
            "target_artifact": target_artifact.to_dict() if target_artifact else None,
            "pocket_artifact": pocket_artifact.to_dict() if pocket_artifact else None,
            "creation_method": creation_method,
            "creation_metadata": provenance,
        },
    )
    _write_json(
        run_dir / "result.json",
        {
            "success": True,
            "feature_count": len(normalized),
            "enabled_feature_count": sum(item.enabled for item in normalized),
            "pgmg_compatible": pgmg_path.is_file(),
            "pgmg_omitted_feature_ids": pgmg_omitted,
            "creation_metadata": provenance,
        },
    )
    artifacts = [
        ArtifactRef.from_path(run_dir, canonical_path, "pharmacophore_hypothesis", role="canonical"),
        ArtifactRef.from_path(run_dir, table_path, "pharmacophore_feature_table", role="editable_features"),
        ArtifactRef.from_path(run_dir, pharmit_path, "pharmacophore_exchange", role="pharmit_json"),
        ArtifactRef.from_path(run_dir, xyz_path, "pharmacophore_exchange", role="omtra_xyz"),
        ArtifactRef.from_path(
            run_dir,
            contract_path,
            "pharmacophore_conditioning_contract",
            role="native_conditioning_contract",
        ),
    ]
    if pgmg_path.is_file():
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                pgmg_path,
                "pharmacophore_exchange",
                role="pgmg_posp",
                metadata={"omitted_feature_ids": pgmg_omitted, "maximum_features": 8},
            )
        )
    write_artifact_manifest(run_dir, artifacts)
    return JobRecord.load(run_dir, task_group=PHARMACOPHORE_TASK_GROUP)
