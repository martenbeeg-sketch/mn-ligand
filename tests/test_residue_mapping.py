from __future__ import annotations

import json
from pathlib import Path

from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.residue_mapping import (
    derive_residue_mapping,
    identity_residue_mapping,
    load_residue_mapping,
    relabel_structural_dynamics,
    subset_residue_mapping,
)


def _ca(serial: int, residue: str, chain: str, number: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {residue:>3s} {chain}{number:4d}"
        f"    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n"
    )


def test_residue_mapping_restores_author_ids_and_fills_missing_sites() -> None:
    native = (
        _ca(1, "SER", "A", 145, 0.0)
        + _ca(2, "MET", "A", 147, 2.0)
        + "END\n"
    )
    prepared = (
        _ca(1, "SER", "A", 1, 0.0)
        + _ca(2, "HIS", "A", 2, 1.0)
        + _ca(3, "MET", "A", 3, 2.0)
        + "END\n"
    )

    mapping = derive_residue_mapping(
        prepared,
        native,
        source_run_id="imported-4lnw",
        source_label="4LNW",
    )

    assert [
        row["native_residue_number"] for row in mapping["residues"]
    ] == [145, 146, 147]
    assert mapping["residues"][1]["mapping_method"] == "chain-offset-inferred"


def test_subset_mapping_preserves_native_ids_after_terminal_trimming() -> None:
    source = (
        _ca(1, "ALA", "A", 1, 0.0)
        + _ca(2, "GLY", "A", 2, 1.0)
        + _ca(3, "SER", "A", 3, 2.0)
        + "END\n"
    )
    mapping = identity_residue_mapping(source)
    mapping["residues"][0]["native_residue_number"] = 145
    mapping["residues"][1]["native_residue_number"] = 146
    mapping["residues"][2]["native_residue_number"] = 147
    trimmed = _ca(1, "GLY", "A", 2, 1.0) + _ca(
        2, "SER", "A", 3, 2.0
    ) + "END\n"

    subset = subset_residue_mapping(mapping, trimmed)

    assert [
        row["native_residue_number"] for row in subset["residues"]
    ] == [146, 147]
    assert [
        row["structure_index"] for row in subset["residues"]
    ] == [1, 2]


def test_subset_mapping_uses_sequence_when_cofolding_renumbers_residues() -> None:
    source = (
        _ca(1, "ALA", "A", 17, 0.0)
        + _ca(2, "GLY", "A", 18, 1.0)
        + _ca(3, "SER", "A", 19, 2.0)
        + "END\n"
    )
    mapping = identity_residue_mapping(source)
    for row, native_number in zip(mapping["residues"], (161, 162, 163)):
        row["native_residue_number"] = native_number
    renumbered = (
        _ca(1, "ALA", "A", 1, 0.0)
        + _ca(2, "GLY", "A", 2, 1.0)
        + _ca(3, "SER", "A", 3, 2.0)
        + "END\n"
    )

    subset = subset_residue_mapping(mapping, renumbered)

    assert [
        row["native_residue_number"] for row in subset["residues"]
    ] == [161, 162, 163]
    assert all(
        row["mapping_method"] == "sequence-subset"
        for row in subset["residues"]
    )


def test_subset_mapping_normalizes_protonated_histidine_names() -> None:
    source = (
        _ca(1, "ALA", "A", 1, 0.0)
        + _ca(2, "HIS", "A", 2, 1.0)
        + _ca(3, "SER", "A", 3, 2.0)
        + "END\n"
    )
    mapping = identity_residue_mapping(source)
    for row, native_number in zip(mapping["residues"], (275, 276, 277)):
        row["native_residue_number"] = native_number
    gromacs = (
        _ca(1, "ALA", "A", 1, 0.0)
        + _ca(2, "HIE", "A", 2, 1.0)
        + _ca(3, "SER", "A", 3, 2.0)
        + "END\n"
    )

    subset = subset_residue_mapping(mapping, gromacs)

    assert len(subset["residues"]) == 3
    assert subset["residues"][1]["structure_residue_name"] == "HIS"
    assert [
        row["native_residue_number"] for row in subset["residues"]
    ] == [275, 276, 277]


def test_existing_md_analytics_can_be_relabelled_without_recalculation() -> None:
    mapping = {
        "source_run_id": "imported-4lnw",
        "source_label": "4LNW",
        "residues": [
            {
                "structure_residue_name": "SER",
                "structure_residue_number": 17,
                "structure_chain": "A",
                "native_residue_name": "SER",
                "native_residue_number": 161,
                "native_insertion_code": "",
                "native_chain": "A",
            }
        ],
    }
    structural = {
        "rmsf": {"residues": ["SER1 · chain 1"]},
        "contacts": {
            "residues": [{"residue": "SER1 · chain 1"}],
            "distance_series": {"SER1 · chain 1": [3.2]},
        },
        "interaction_network": {
            "nodes": [{"id": "SER1 · chain 1"}],
            "edges": [{"target": "SER1 · chain 1"}],
        },
    }

    relabelled = relabel_structural_dynamics(structural, mapping)

    assert relabelled["rmsf"]["residues"] == ["SER161 · chain A"]
    assert relabelled["contacts"]["residues"][0]["residue"] == (
        "SER161 · chain A"
    )
    assert relabelled["residue_numbering"]["scheme"] == "source_author"


def test_load_residue_mapping_supports_legacy_md_snapshot(
    tmp_path: Path,
) -> None:
    mapping = {
        "schema_version": 1,
        "source_label": "3GWS",
        "residues": [{
            "structure_chain": "A",
            "structure_residue_number": 130,
            "native_chain": "X",
            "native_residue_number": 331,
        }],
    }
    run_dir = tmp_path / "md-analysis"
    run_dir.mkdir()
    (run_dir / "source_residue_mapping.json").write_text(
        json.dumps(mapping)
    )
    job = JobRecord(
        run_id="legacy-md",
        task_group="md-analysis",
        run_dir=run_dir,
        status="completed",
    )

    assert load_residue_mapping(job) == mapping
