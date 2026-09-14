from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from rdkit import Chem, rdBase


FORMULATION_REGISTRY_SCHEMA_VERSION = 1
FORMULATION_REGISTRY_ENV = "MN_LIGAND_FORMULATION_REGISTRY"


@dataclass(frozen=True)
class FormulationComponent:
    component_id: str
    label: str
    category: str
    smiles: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FormulationComponent:
        component_id = str(payload.get("id") or "").strip()
        label = str(payload.get("label") or "").strip()
        category = str(payload.get("category") or "").strip()
        smiles_value = payload.get("smiles")
        if not component_id:
            raise ValueError("Formulation component ID cannot be empty")
        if not label:
            raise ValueError(
                f"Formulation component {component_id!r} has no label"
            )
        if not category:
            raise ValueError(
                f"Formulation component {component_id!r} has no category"
            )
        if not isinstance(smiles_value, list) or not smiles_value:
            raise ValueError(
                f"Formulation component {component_id!r} requires SMILES variants"
            )
        smiles = tuple(
            str(value).strip() for value in smiles_value if str(value).strip()
        )
        if len(smiles) != len(smiles_value):
            raise ValueError(
                f"Formulation component {component_id!r} has an empty SMILES variant"
            )
        for value in smiles:
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(value, sanitize=True)
            if (
                molecule is None
                or molecule.GetNumHeavyAtoms() < 1
                or any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms())
            ):
                raise ValueError(
                    f"Formulation component {component_id!r} has invalid SMILES: "
                    f"{value}"
                )
        return cls(
            component_id=component_id,
            label=label,
            category=category,
            smiles=smiles,
        )


@dataclass(frozen=True)
class FormulationRegistry:
    components: tuple[FormulationComponent, ...]
    source: str
    schema_version: int = FORMULATION_REGISTRY_SCHEMA_VERSION

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        source: str,
    ) -> FormulationRegistry:
        version = int(payload.get("schema_version") or 0)
        if version != FORMULATION_REGISTRY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported formulation registry schema version: {version}"
            )
        raw_components = payload.get("components")
        if not isinstance(raw_components, list) or not raw_components:
            raise ValueError("Formulation registry contains no components")
        components = tuple(
            FormulationComponent.from_dict(dict(item))
            for item in raw_components
            if isinstance(item, dict)
        )
        if len(components) != len(raw_components):
            raise ValueError("Every formulation registry entry must be an object")
        ids = [component.component_id for component in components]
        if len(ids) != len(set(ids)):
            raise ValueError("Formulation registry contains duplicate component IDs")
        canonical_owners: dict[str, str] = {}
        for component in components:
            for smiles in component.smiles:
                molecule = Chem.MolFromSmiles(smiles)
                canonical = Chem.MolToSmiles(
                    molecule, canonical=True, isomericSmiles=True
                )
                owner = canonical_owners.get(canonical)
                if owner is not None and owner != component.component_id:
                    raise ValueError(
                        f"Formulation SMILES {smiles!r} is assigned to both "
                        f"{owner!r} and {component.component_id!r}"
                    )
                canonical_owners[canonical] = component.component_id
        return cls(
            components=components,
            source=source,
            schema_version=version,
        )


def load_formulation_registry(path: Path | None = None) -> FormulationRegistry:
    selected_path = path
    if selected_path is None:
        override = str(os.environ.get(FORMULATION_REGISTRY_ENV) or "").strip()
        selected_path = Path(override).expanduser() if override else None
    if selected_path is None:
        resource = resources.files("mn_ligand.manifests").joinpath(
            "formulations.yaml"
        )
        text = resource.read_text(encoding="utf-8")
        source = str(resource)
    else:
        selected_path = selected_path.resolve()
        text = selected_path.read_text(encoding="utf-8")
        source = str(selected_path)
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("Formulation registry root must be an object")
    return FormulationRegistry.from_dict(payload, source=source)
