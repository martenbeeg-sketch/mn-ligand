from __future__ import annotations

import json
from pathlib import Path

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.workflows.provenance_backfill import (
    backfill_derived_target_provenance,
)


def _job(
    run_dir: Path,
    metadata: dict,
    *,
    artifact_name: str = "",
) -> Path | None:
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"schema_version": 1, "status": "completed", **metadata})
    )
    artifacts = []
    artifact_path = None
    if artifact_name:
        artifact_path = run_dir / artifact_name
        artifact_path.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"
        )
        artifacts.append(
            ArtifactRef.from_path(
                run_dir, artifact_path, "prepared_complex", role="complex"
            )
        )
    write_artifact_manifest(run_dir, artifacts)
    return artifact_path


def test_backfill_restores_identity_and_modification_history_without_artifact_changes(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    _job(
        runs / "structure-jobs" / "source",
        {
            "run_id": "source",
            "source": "pdb",
            "pdb_id": "1ABC",
            "ligand_key": "LIG|B|101|_",
            "receptor": {"title": "Example", "entities": []},
            "ligands": [{"ccd_id": "LIG", "molecular_weight": 42.0}],
        },
        artifact_name="source.pdb",
    )
    _job(
        runs / "target-trimming" / "trim",
        {
            "run_id": "trim",
            "job_type": "target_trimming",
            "parent_run_id": "source",
            "trim_ranges": {"A": {"start": 1, "end": 1}},
        },
    )
    repaired_artifact = _job(
        runs / "terminal-repair" / "repair",
        {
            "run_id": "repair",
            "job_type": "terminal_repair",
            "parent_run_id": "trim",
            "tool": "MODELLER",
            "chain": "A",
            "extension_sequence": "GG",
        },
        artifact_name="complex_repaired.pdb",
    )
    assert repaired_artifact is not None
    artifact_before = repaired_artifact.read_bytes()

    preview = backfill_derived_target_provenance(write=False)
    assert preview["updated"] == 3
    assert "pdb_id" not in json.loads(
        (runs / "terminal-repair" / "repair" / "metadata.json").read_text()
    )

    applied = backfill_derived_target_provenance(write=True)
    metadata = json.loads(
        (runs / "terminal-repair" / "repair" / "metadata.json").read_text()
    )

    assert applied["updated"] == 3
    assert metadata["pdb_id"] == "1ABC"
    assert metadata["source"] == "pdb"
    assert metadata["ligand_key"] == "LIG|B|101|_"
    assert metadata["receptor"]["title"] == "Example"
    assert metadata["provenance_origin"] == (
        "PDB → Target trimming → MODELLER repair"
    )
    assert [item["kind"] for item in metadata["modification_history"]] == [
        "target_trimming",
        "terminal_repair",
    ]
    assert metadata["provenance_backfilled_at"]
    assert repaired_artifact.read_bytes() == artifact_before


def test_backfill_recovers_legacy_pdbfixer_cleaning_from_exact_prepared_output(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    protein = _job(
        runs / "structure-jobs" / "legacy",
        {
            "run_id": "legacy",
            "job_type": "structure",
            "source": "pdb",
            "pdb_id": "1ABC",
            "created_at": "2026-05-22T10:40:24+00:00",
        },
        artifact_name="1abc_protein_refined.pdb",
    )
    assert protein is not None
    prepared = runs / "prepared-structures" / "preparation"
    prepared.mkdir(parents=True)
    (prepared / "input.json").write_text(
        json.dumps(
            {
                "pdb_id": "1ABC",
                "clean_protein": True,
                "map_modified_residues": True,
            }
        )
    )
    (prepared / "result.json").write_text(
        json.dumps(
            {
                "prepared_pdb_data": protein.read_text(),
                "protein_cleaned": True,
                "components": {"protein": 1, "water": 10},
                "modified_residue_mapping": {
                    "enabled": True,
                    "mappings": {
                        "CAS|A|1|_": {
                            "target": "CYS",
                            "kept_atoms": 6,
                            "dropped_atoms": 3,
                        }
                    },
                },
            }
        )
    )

    applied = backfill_derived_target_provenance(write=True)
    metadata = json.loads(
        (runs / "structure-jobs" / "legacy" / "metadata.json").read_text()
    )

    assert applied["updated"] == 1
    assert metadata["legacy_preparation_evidence"]["source_run_id"] == "preparation"
    assert metadata["legacy_preparation_evidence"]["energy_minimized"] is False
    assert [item["kind"] for item in metadata["modification_history"]] == [
        "modified_residue_mapping",
        "pdbfixer_cleaning",
    ]
    assert metadata["provenance_origin"] == (
        "PDB → Modified-residue mapping → PDBFixer cleaning"
    )
