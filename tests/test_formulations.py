from __future__ import annotations

from pathlib import Path

import pytest

from mn_ligand.core.formulations import (
    FORMULATION_REGISTRY_ENV,
    load_formulation_registry,
)


def test_bundled_formulation_registry_is_versioned_and_complete() -> None:
    registry = load_formulation_registry()

    assert registry.schema_version == 1
    assert len(registry.components) >= 20
    assert {"citrate", "tfa", "meglumine", "lysine", "tartrate"}.issubset(
        {component.component_id for component in registry.components}
    )
    assert registry.source.endswith("formulations.yaml")


def test_formulation_registry_can_be_extended_with_environment_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    custom = tmp_path / "formulations.yaml"
    custom.write_text(
        "schema_version: 1\n"
        "components:\n"
        "  - id: benzoate\n"
        "    label: benzoate\n"
        "    category: acid salt\n"
        "    smiles: ['O=C(O)c1ccccc1', 'O=C([O-])c1ccccc1']\n"
    )
    monkeypatch.setenv(FORMULATION_REGISTRY_ENV, str(custom))

    registry = load_formulation_registry()

    assert registry.source == str(custom.resolve())
    assert registry.components[0].component_id == "benzoate"
    assert len(registry.components[0].smiles) == 2


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            "schema_version: 2\ncomponents:\n  - id: x\n"
            "    label: x\n    category: salt\n    smiles: ['Cl']\n",
            "schema version",
        ),
        (
            "schema_version: 1\ncomponents:\n  - id: x\n"
            "    label: x\n    category: salt\n    smiles: ['not-smiles']\n",
            "invalid SMILES",
        ),
        (
            "schema_version: 1\ncomponents:\n"
            "  - id: x\n    label: x\n    category: salt\n    smiles: ['Cl']\n"
            "  - id: x\n    label: y\n    category: salt\n    smiles: ['Br']\n",
            "duplicate component IDs",
        ),
    ],
)
def test_formulation_registry_rejects_malformed_configuration(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    registry_path = tmp_path / "formulations.yaml"
    registry_path.write_text(payload)

    with pytest.raises(ValueError, match=message):
        load_formulation_registry(registry_path)
