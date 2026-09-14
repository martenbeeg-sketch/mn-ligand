from __future__ import annotations

import json
from pathlib import Path

import pytest

from mn_ligand.core.artifacts import (
    ArtifactManifest,
    ArtifactRef,
    load_artifact_manifest,
    write_structure_artifact_manifest,
)


def test_artifact_ref_requires_a_safe_relative_path() -> None:
    with pytest.raises(ValueError):
        ArtifactRef(run_id="run-1", artifact_type="prepared_receptor", path="/tmp/receptor.pdb")
    with pytest.raises(ValueError):
        ArtifactRef(run_id="run-1", artifact_type="prepared_receptor", path="../receptor.pdb")


def test_manifest_round_trip_preserves_portable_reference(tmp_path: Path) -> None:
    run_dir = tmp_path / "structure-jobs" / "run-1"
    artifact_path = run_dir / "artifacts" / "receptor.pdb"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("ATOM\nEND\n")

    artifact = ArtifactRef.from_path(run_dir, artifact_path, "prepared_receptor", role="receptor")
    manifest = ArtifactManifest(run_id=run_dir.name, artifacts=(artifact,))
    manifest.write(run_dir)
    loaded = load_artifact_manifest(run_dir, task_group="structure-jobs")

    assert loaded.source == "native"
    assert loaded.artifacts[0].path == "artifacts/receptor.pdb"
    assert loaded.artifacts[0].resolve(run_dir, must_exist=True) == artifact_path
    assert loaded.artifacts[0].sha256
    assert not Path(loaded.artifacts[0].path).is_absolute()


def test_legacy_structure_artifacts_are_inferred_without_writing(tmp_path: Path) -> None:
    run_dir = tmp_path / "structure-jobs" / "legacy-run"
    run_dir.mkdir(parents=True)
    (run_dir / "4lnw_protein_refined.pdb").write_text("ATOM\nEND\n")
    (run_dir / "4lnw_t3_ligand_refined.sdf").write_text("$$$$\n")
    (run_dir / "4lnw_t3_complex_refined.pdb").write_text("ATOM\nHETATM\nEND\n")

    manifest = load_artifact_manifest(run_dir, task_group="structure-jobs")

    assert manifest.source == "legacy_inferred"
    assert {item.artifact_type for item in manifest.artifacts} == {
        "prepared_receptor",
        "prepared_ligand_set",
        "prepared_complex",
    }
    assert not (run_dir / "artifacts.json").exists()


def test_new_structure_manifest_contains_relative_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "structure-jobs" / "new-run"
    run_dir.mkdir(parents=True)
    (run_dir / "target_protein_refined.pdb").write_text("ATOM\nEND\n")
    (run_dir / "target_ligand_refined.sdf").write_text("$$$$\n")
    (run_dir / "repair_report.json").write_text('{"engine": "MODELLER"}\n')

    manifest = write_structure_artifact_manifest(run_dir)
    payload = json.loads((run_dir / "artifacts.json").read_text())

    assert manifest.source == "native"
    assert payload["schema_version"] == 1
    assert all(not Path(item["path"]).is_absolute() for item in payload["artifacts"])
    assert all(item["run_id"] == "new-run" for item in payload["artifacts"])
    assert manifest.by_type("repair_report")[0].role == "report"
