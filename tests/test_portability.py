from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.portability import (
    PortabilityError,
    assert_job_portable,
    audit_runtime_portability,
    export_portable_runtime,
    validate_job_portability,
    verify_portable_export,
)


def test_portability_audit_is_read_only_and_classifies_paths(tmp_path: Path) -> None:
    run_root = tmp_path / "new-runs"
    job = run_root / "docking" / "run-1"
    result = job / "results" / "pose.sdf"
    result.parent.mkdir(parents=True)
    result.write_text("$$$$\n")
    metadata = job / "metadata.json"
    old_root = "/old/mn-ligand-workdir/workdir/runs"
    metadata.write_text(
        json.dumps(
            {
                "source_path": f"{old_root}/docking/run-1/results/pose.sdf",
                "container_path": "/workspace/results/pose.sdf",
            }
        )
        + "\n"
    )
    before = metadata.read_bytes()

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(run_root)}):
        report = audit_runtime_portability(run_root)

    assert report["read_only"] is True
    assert report["counts"]["relocatable_paths"] == 1
    assert report["counts"]["container_paths"] == 1
    assert metadata.read_bytes() == before


def _portable_source(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    runs = tmp_path / "source" / "runs"
    references = tmp_path / "source" / "reference_files"
    libraries = tmp_path / "source" / "libraries"
    run_dir = runs / "docking" / "run-1"
    result = run_dir / "results" / "pose.sdf"
    result.parent.mkdir(parents=True)
    result.write_text("pose\n$$$$\n")
    reference = references / "models" / "weights.bin"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"weights")
    libraries.mkdir(parents=True)
    (libraries / "compounds.csv").write_text("compound_id,smiles\na,CCO\n")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_path": str(result),
                "model_path": str(reference),
                "queued_command": ["tool", str(result), "/workspace/output.sdf"],
            },
            indent=2,
        )
        + "\n"
    )
    write_artifact_manifest(
        run_dir,
        [ArtifactRef.from_path(run_dir, result, "pose_set")],
    )
    return runs, references, libraries, run_dir


def test_portable_export_rewrites_only_copy_and_verifies(tmp_path: Path) -> None:
    runs, references, libraries, run_dir = _portable_source(tmp_path)
    source_metadata = run_dir / "metadata.json"
    before = source_metadata.read_bytes()
    destination = tmp_path / "portable"

    result = export_portable_runtime(
        destination,
        source_runs=runs,
        source_references=references,
        source_libraries=libraries,
    )

    assert source_metadata.read_bytes() == before
    exported = json.loads(
        (
            destination
            / "workdir"
            / "runs"
            / "docking"
            / "run-1"
            / "metadata.json"
        ).read_text()
    )
    assert exported["source_path"] == "runs:///docking/run-1/results/pose.sdf"
    assert exported["model_path"] == "reference:///models/weights.bin"
    assert exported["queued_command"][1] == str(
        run_dir / "results" / "pose.sdf"
    )
    assert result["verification"]["valid"] is True
    assert verify_portable_export(destination)["valid"] is True


def test_portable_export_refuses_existing_destination(tmp_path: Path) -> None:
    runs, references, libraries, _ = _portable_source(tmp_path)
    destination = tmp_path / "portable"
    destination.mkdir()

    with pytest.raises(PortabilityError, match="already exists"):
        export_portable_runtime(
            destination,
            source_runs=runs,
            source_references=references,
            source_libraries=libraries,
        )


def test_portable_export_aborts_on_unresolved_operational_path(tmp_path: Path) -> None:
    runs, references, libraries, run_dir = _portable_source(tmp_path)
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_path"] = "/unmanaged/missing/source.pdb"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    destination = tmp_path / "portable"

    with pytest.raises(PortabilityError, match="unresolved operational"):
        export_portable_runtime(
            destination,
            source_runs=runs,
            source_references=references,
            source_libraries=libraries,
        )

    assert not destination.exists()


def test_portable_verify_detects_artifact_checksum_change(tmp_path: Path) -> None:
    runs, references, libraries, _ = _portable_source(tmp_path)
    destination = tmp_path / "portable"
    export_portable_runtime(
        destination,
        source_runs=runs,
        source_references=references,
        source_libraries=libraries,
    )
    (
        destination
        / "workdir"
        / "runs"
        / "docking"
        / "run-1"
        / "results"
        / "pose.sdf"
    ).write_text(
        "changed\n"
    )

    report = verify_portable_export(destination)

    assert report["valid"] is False
    assert any("checksum mismatch" in item["error"] for item in report["errors"])


def test_portable_export_can_include_app_managed_files(tmp_path: Path) -> None:
    runs, references, libraries, run_dir = _portable_source(tmp_path)
    app_home = runs.parent
    stored = app_home / "storage" / "index.sqlite"
    stored.parent.mkdir()
    stored.write_bytes(b"index")
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["index_path"] = str(stored)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    destination = tmp_path / "portable"

    export_portable_runtime(
        destination,
        source_runs=runs,
        source_references=references,
        source_libraries=libraries,
        source_app_home=app_home,
    )

    exported = json.loads(
        (
            destination
            / "workdir"
            / "runs"
            / "docking"
            / "run-1"
            / "metadata.json"
        ).read_text()
    )
    assert exported["index_path"] == "app:///storage/index.sqlite"
    assert (destination / "storage" / "index.sqlite").read_bytes() == b"index"
    assert verify_portable_export(destination)["valid"] is True


def test_new_job_validator_accepts_relative_portable_and_container_paths(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "md-system-prep" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "source.pdb").write_text("END\n")
    (run_dir / "input.json").write_text(
        json.dumps(
            {
                "relative_path": "source.pdb",
                "managed_path": "runs:///md-system-prep/run-1/source.pdb",
                "container_path": "/output/source.pdb",
            }
        )
    )
    (run_dir / "metadata.json").write_text(
        json.dumps({"queued_command": ["tool", "/home/user/tool-config.json"]})
    )
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))

    report = validate_job_portability(run_dir)

    assert report["valid"] is True
    assert report["counts"]["relative_paths"] == 1
    assert report["counts"]["portable_paths"] == 1
    assert report["counts"]["container_paths"] == 1
    assert_job_portable(run_dir)


def test_new_job_validator_rejects_absolute_host_operational_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "fixture" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "input.json").write_text(
        json.dumps({"prepared_complex_path": "/home/old-machine/data/complex.pdb"})
    )

    report = validate_job_portability(run_dir)

    assert report["valid"] is False
    assert report["counts"]["absolute_host_paths"] == 1
    with pytest.raises(PortabilityError, match="absolute host path"):
        assert_job_portable(run_dir)


def test_new_job_validator_rejects_unsafe_portable_uri(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "fixture" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "input.json").write_text(
        json.dumps({"source_path": "runs:///../outside.pdb"})
    )

    report = validate_job_portability(run_dir)

    assert report["valid"] is False
    assert "invalid portable path" in report["errors"][0]["error"]
