from __future__ import annotations

import json
import gzip
import subprocess
from pathlib import Path
from unittest.mock import patch

from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.pockets import PocketRecord, PocketSet
from mn_ligand.core.worker import WorkerConfig, run_worker_once
from mn_ligand.workflows.pocket_detection import (
    BOUND_LIGAND_METHOD,
    _docker_command,
    bound_ligand_candidates,
    normalize_fpocket_output,
    normalize_p2rank_output,
    normalize_pesto_output,
    queue_pocket_detection_job,
    run_bound_ligand_native,
    run_pocket_detection_job,
)
from mn_ligand.workflows.protein_preparation import create_protein_import_job, run_protein_cleaning_job


PDB_DATA = """HEADER    TEST
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.400   0.000   0.000  1.00 20.00           C
ATOM      3  C   TYR A   2       6.000   0.000   0.000  1.00 20.00           C
HETATM    4  C1  LIG A 101       2.000   1.000   0.000  1.00 20.00           C
HETATM    5  C2  LIG A 101       3.000   1.000   0.000  1.00 20.00           C
HETATM    6  O1  LIG A 101       4.000   1.000   0.000  1.00 20.00           O
HETATM    7  O   HOH A 201       2.000   2.000   0.000  1.00 20.00           O
END
"""

PREPARED_PDB_DATA = "\n".join(
    line for line in PDB_DATA.splitlines() if not line.startswith("HETATM")
) + "\n"


def _pocket_structure(score: float = 0.65) -> str:
    return f"""HEADER 0  - Pocket Score                      : {score:.4f}
HEADER 1  - Drug Score                        : 0.8000
ATOM      1  CA  ALA A  10      10.000  20.000  30.000  1.00 20.00           C
ATOM      2  CA  TYR A  11      14.000  24.000  34.000  1.00 20.00           C
END
"""


POCKET_POINTS = """HETATM    1 APOL STP A   1      10.000  20.000  30.000  0.00  1.50
HETATM    2 APOL STP A   1      14.000  24.000  34.000  0.00  1.50
END
"""


def _p2rank_point(serial: int, rank: int, x: float, y: float, z: float) -> str:
    return (
        f"HETATM{serial:5d}  H   STP P{rank:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  0.50 0.800           H"
    )


def _write_fake_p2rank_outputs(output_dir: Path) -> None:
    native = output_dir / "native" / "p2rank"
    points_dir = native / "visualizations" / "data"
    points_dir.mkdir(parents=True)
    (native / "target.pdb_predictions.csv").write_text(
        "name, rank, score, probability, sas_points, surf_atoms, center_x, center_y, center_z, residue_ids, surf_atom_ids\n"
        "pocket1, 1, 9.77, 0.525, 2, 4, 12.0, 22.0, 32.0, A_1 A_2, 1 2 3 4\n"
    )
    (native / "target.pdb_residues.csv").write_text(
        "chain, residue_label, residue_name, score, zscore, probability, pocket\n"
        "A, 1, ALA, 0.8, 1.0, 0.7, 1\n"
        "A, 2, TYR, 0.7, 0.8, 0.6, 1\n"
    )
    with gzip.open(points_dir / "target.pdb_points.pdb.gz", "wt") as handle:
        handle.write(_p2rank_point(1, 1, 10.0, 20.0, 30.0) + "\n")
        handle.write(_p2rank_point(2, 1, 14.0, 24.0, 34.0) + "\n")
        handle.write("END\n")
    (native / "params.txt").write_text("model = default\n")
    (native / "run.log").write_text("P2Rank fixture completed\n")


def _fake_cleaning(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    output_mount = next(
        command[index + 1]
        for index, item in enumerate(command)
        if item == "-v" and command[index + 1].endswith(":/output")
    )
    output_dir = Path(output_mount.removesuffix(":/output"))
    (output_dir / "native_result.json").write_text(
        json.dumps(
            {
                "success": True,
                "prepared_pdb_data": PREPARED_PDB_DATA,
                "protein_cleaned": True,
                "components": {"protein": 1},
                "modified_residue_mapping": {},
                "ligands": [],
            }
        )
    )
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def _write_fake_fpocket_outputs(output_dir: Path) -> None:
    artifact_dir = output_dir / "artifacts" / "pockets"
    artifact_dir.mkdir(parents=True)
    structure_path = artifact_dir / "pocket_001.pdb"
    structure_path.write_text(_pocket_structure())
    points_path = artifact_dir / "pocket_001_points.pqr"
    points_path.write_text(POCKET_POINTS)
    runner_input = json.loads((output_dir / "runner_input.json").read_text())
    pocket_set = PocketSet(
        method="fpocket",
        source_target=runner_input["source_target"],
        parameters={"max_pockets": 10, "min_score": None, "box_padding_angstrom": 4.0},
        pockets=(
            PocketRecord(
                pocket_id="fpocket-1",
                rank=1,
                method="fpocket",
                score=0.65,
                center_angstrom=(12.0, 22.0, 32.0),
                size_angstrom=(12.0, 12.0, 12.0),
                structure_path="artifacts/pockets/pocket_001.pdb",
                points_path="artifacts/pockets/pocket_001_points.pqr",
            ),
        ),
    )
    pocket_set.write(artifact_dir / "pocket_set.json")
    (output_dir / "native_result.json").write_text(
        json.dumps(
            {
                "success": True,
                "method": "fpocket",
                "pocket_count": 1,
                "pocket_set": "artifacts/pockets/pocket_set.json",
            }
        )
    )


def test_fpocket_output_becomes_ranked_typed_pocket_set(tmp_path: Path) -> None:
    native = tmp_path / "target_out" / "pockets"
    native.mkdir(parents=True)
    (native / "pocket1_atm.pdb").write_text(_pocket_structure())
    (native / "pocket1_vert.pqr").write_text(POCKET_POINTS)
    (native / "pocket2_atm.pdb").write_text(_pocket_structure(-0.2))
    (native / "pocket2_vert.pqr").write_text(POCKET_POINTS)

    pocket_set = normalize_fpocket_output(
        native.parent,
        tmp_path / "artifacts" / "pockets",
        source_target={"artifact_id": "target-1", "sha256": "abc"},
        max_pockets=10,
        min_score=0.0,
        box_padding_angstrom=4.0,
    )

    assert len(pocket_set.pockets) == 1
    pocket = pocket_set.pockets[0]
    assert pocket.score == 0.65
    assert pocket.druggability_score == 0.8
    assert pocket.center_angstrom == (12.0, 22.0, 32.0)
    assert pocket.size_angstrom == (12.0, 12.0, 12.0)
    assert [residue.residue_number for residue in pocket.residues] == ["10", "11"]


def test_pocket_set_schema_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "pocket_set.json"
    pocket_set = PocketSet(
        method="fpocket",
        source_target={"artifact_id": "target"},
        source_complex={"artifact_id": "complex"},
        pockets=(
            PocketRecord(
                pocket_id="fpocket-1",
                rank=1,
                method="fpocket",
                center_angstrom=(1.0, 2.0, 3.0),
                size_angstrom=(10.0, 11.0, 12.0),
                metadata={"origin": "test"},
            ),
        ),
    )
    pocket_set.write(path)

    assert PocketSet.read(path) == pocket_set
    assert json.loads(path.read_text())["schema_version"] == 1


def test_pesto_ligand_scores_become_clustered_pocket_set(tmp_path: Path) -> None:
    target = tmp_path / "target.pdb"
    target.write_text(
        """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  TYR A   2       4.000   0.000   0.000  1.00 20.00           C
ATOM      3  CA  LEU A   3       8.000   0.000   0.000  1.00 20.00           C
ATOM      4  CA  GLY A  40      40.000   0.000   0.000  1.00 20.00           C
END
"""
    )
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "chain,residue,amino_acid,residue_name,score\n"
        "A,1,A,ALA,0.91\nA,2,Y,TYR,0.82\nA,3,L,LEU,0.76\nA,40,G,GLY,0.95\n"
    )
    artifact_dir = tmp_path / "artifacts" / "pockets"

    pocket_set = normalize_pesto_output(
        scores,
        target,
        artifact_dir,
        source_target={"artifact_id": "prepared-target"},
        score_threshold=0.5,
        cluster_distance_angstrom=8.0,
        minimum_residues=2,
        max_pockets=10,
    )

    assert pocket_set.method == "pesto"
    assert len(pocket_set.pockets) == 1
    pocket = pocket_set.pockets[0]
    assert pocket.score == 0.91
    assert [residue.residue_number for residue in pocket.residues] == ["1", "2", "3"]
    assert pocket.center_angstrom == (4.0, 0.0, 0.0)
    assert pocket.size_angstrom == (8.0, 4.0, 4.0)
    assert pocket.structure_path == "artifacts/pockets/pocket_001.pdb"
    assert pocket.points_path == "artifacts/pockets/pocket_001_residues.pdb"


def test_p2rank_native_tables_and_points_become_typed_pockets(tmp_path: Path) -> None:
    target = tmp_path / "target.pdb"
    target.write_text(PREPARED_PDB_DATA)
    _write_fake_p2rank_outputs(tmp_path)
    native = tmp_path / "native" / "p2rank"

    pocket_set = normalize_p2rank_output(
        native / "target.pdb_predictions.csv",
        native / "target.pdb_residues.csv",
        native / "visualizations" / "data" / "target.pdb_points.pdb.gz",
        target,
        tmp_path / "artifacts" / "pockets",
        source_target={"artifact_id": "prepared-target"},
        profile="default",
        max_pockets=10,
        min_probability=0.2,
        box_padding_angstrom=4.0,
    )

    assert pocket_set.method == "p2rank"
    assert len(pocket_set.pockets) == 1
    pocket = pocket_set.pockets[0]
    assert pocket.pocket_id == "p2rank-1"
    assert pocket.score == 9.77
    assert pocket.descriptors["probability"] == 0.525
    assert pocket.druggability_score is None
    assert pocket.center_angstrom == (12.0, 22.0, 32.0)
    assert pocket.size_angstrom == (12.0, 12.0, 12.0)
    assert [item.residue_number for item in pocket.residues] == ["1", "2"]
    assert pocket.structure_path == "artifacts/pockets/pocket_001.pdb"
    assert pocket.points_path == "artifacts/pockets/pocket_001_points.pdb"


def test_bound_ligand_native_reuses_structure_preparation_parser(tmp_path: Path) -> None:
    target = tmp_path / "prepared.pdb"
    target.write_text(PREPARED_PDB_DATA)
    source_complex = tmp_path / "imported.pdb"
    source_complex.write_text(PDB_DATA)
    output = tmp_path / "output"
    output.mkdir()

    payload = run_bound_ligand_native(
        {
            "target_path": str(target),
            "complex_path": str(source_complex),
            "bound_ligand_key": "LIG|A|101|_",
            "source_target": {"artifact_id": "prepared"},
            "source_complex": {"artifact_id": "imported"},
            "box_padding_angstrom": 4.0,
            "lining_cutoff_angstrom": 5.0,
        },
        output,
    )

    pocket_set = PocketSet.read(output / payload["pocket_set"])
    pocket = pocket_set.pockets[0]
    assert pocket_set.method == BOUND_LIGAND_METHOD
    assert pocket_set.source_complex["artifact_id"] == "imported"
    assert pocket.center_angstrom == (3.0, 1.0, 0.0)
    assert pocket.size_angstrom == (10.0, 8.0, 8.0)
    assert pocket.metadata["bound_ligand"]["key"] == "LIG|A|101|_"
    assert [residue.residue_number for residue in pocket.residues] == ["1", "2"]


def test_fpocket_docker_user_matches_daemon_mode(tmp_path: Path) -> None:
    source = tmp_path / "target.pdb"
    source.write_text(PDB_DATA)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with patch("mn_ligand.workflows.pocket_detection._docker_is_rootless", return_value=True):
        rootless = _docker_command("fpocket:test", run_dir, source)
    with patch("mn_ligand.workflows.pocket_detection._docker_is_rootless", return_value=False):
        rootful = _docker_command("fpocket:test", run_dir, source)

    assert "--user" not in rootless
    assert rootful[rootful.index("--user") + 1] == f"{__import__('os').getuid()}:{__import__('os').getgid()}"


def test_method_specific_default_images_are_declared() -> None:
    source = Path(__file__).parents[1] / "mn_ligand/workflows/pocket_detection.py"
    contents = source.read_text()

    assert '"pesto": DEFAULT_PESTO_IMAGE' in contents
    assert '"p2rank": DEFAULT_P2RANK_IMAGE' in contents


def test_pocket_job_consumes_prepared_target_and_publishes_artifacts(tmp_path: Path) -> None:
    def fake_fpocket(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output_mount = next(
            command[index + 1]
            for index, item in enumerate(command)
            if item == "-v" and command[index + 1].endswith(":/output")
        )
        output_dir = Path(output_mount.removesuffix(":/output"))
        artifact_dir = output_dir / "artifacts" / "pockets"
        artifact_dir.mkdir(parents=True)
        structure_path = artifact_dir / "pocket_001.pdb"
        structure_path.write_text(_pocket_structure())
        points_path = artifact_dir / "pocket_001_points.pqr"
        points_path.write_text(POCKET_POINTS)
        runner_input = json.loads((output_dir / "runner_input.json").read_text())
        pocket_set = PocketSet(
            method="fpocket",
            source_target=runner_input["source_target"],
            parameters={"max_pockets": 10, "min_score": None, "box_padding_angstrom": 4.0},
            pockets=(
                PocketRecord(
                    pocket_id="fpocket-1",
                    rank=1,
                    method="fpocket",
                    score=0.65,
                    center_angstrom=(12.0, 22.0, 32.0),
                    size_angstrom=(12.0, 12.0, 12.0),
                    structure_path="artifacts/pockets/pocket_001.pdb",
                    points_path="artifacts/pockets/pocket_001_points.pqr",
                ),
            ),
        )
        pocket_set.write(artifact_dir / "pocket_set.json")
        (output_dir / "native_result.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "method": "fpocket",
                    "pocket_count": 1,
                    "pocket_set": "artifacts/pockets/pocket_set.json",
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout="fpocket", stderr="")

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        imported = create_protein_import_job(PDB_DATA, filename="target.pdb", source="upload")
        with patch("mn_ligand.workflows.protein_preparation.subprocess.run", side_effect=_fake_cleaning):
            cleaned, _ = run_protein_cleaning_job(imported.run_id)
        with patch("mn_ligand.workflows.pocket_detection.subprocess.run", side_effect=fake_fpocket):
            job, native = run_pocket_detection_job(cleaned.run_id)

    assert native["success"] is True
    assert job.status == "completed"
    assert job.parent_run_id == cleaned.run_id
    assert job.result["pocket_count"] == 1
    assert job.artifact_manifest is not None
    assert job.artifact_manifest.by_type("pocket_set")
    assert job.artifact_manifest.by_type("pocket")
    assert job.artifact_manifest.by_type("pocket_points")
    input_payload = json.loads((job.run_dir / "input.json").read_text())
    assert input_payload["input_artifact"]["artifact_type"] == "prepared_target"
    assert not Path(input_payload["input_artifact"]["path"]).is_absolute()
    command_record = json.loads((job.run_dir / "command.json").read_text())
    assert command_record["tool_id"] == "fpocket"
    assert command_record["resources"]["gpu"] is False


def test_bound_ligand_job_uses_imported_complex_as_second_typed_input(tmp_path: Path) -> None:
    def fake_bound_ligand(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        mounts = {
            container: Path(host)
            for index, item in enumerate(command)
            if item == "-v"
            for host, container, *_ in [command[index + 1].split(":")]
        }
        output_dir = mounts["/output"]
        config = json.loads((output_dir / "runner_input.json").read_text())
        config["target_path"] = str(mounts[config["target_path"]])
        config["complex_path"] = str(mounts[config["complex_path"]])
        payload = run_bound_ligand_native(config, output_dir)
        (output_dir / "native_result.json").write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        imported = create_protein_import_job(PDB_DATA, filename="target.pdb", source="upload")
        with patch("mn_ligand.workflows.protein_preparation.subprocess.run", side_effect=_fake_cleaning):
            cleaned, _ = run_protein_cleaning_job(imported.run_id)
        assert [item["key"] for item in bound_ligand_candidates(cleaned.run_id)] == ["LIG|A|101|_"]
        with patch("mn_ligand.workflows.pocket_detection.subprocess.run", side_effect=fake_bound_ligand):
            job, native = run_pocket_detection_job(
                cleaned.run_id,
                method=BOUND_LIGAND_METHOD,
                bound_ligand_key="LIG|A|101|_",
            )

    assert native["success"] is True
    assert job.status == "completed"
    assert job.result["method"] == BOUND_LIGAND_METHOD
    input_payload = json.loads((job.run_dir / "input.json").read_text())
    assert input_payload["input_artifacts"]["prepared_target"]["artifact_type"] == "prepared_target"
    assert input_payload["input_artifacts"]["source_complex"]["artifact_type"] == "imported_target"


def test_pocket_job_records_container_start_failure(tmp_path: Path) -> None:
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        imported = create_protein_import_job(PDB_DATA, filename="target.pdb", source="upload")
        with patch("mn_ligand.workflows.protein_preparation.subprocess.run", side_effect=_fake_cleaning):
            cleaned, _ = run_protein_cleaning_job(imported.run_id)
        with patch(
            "mn_ligand.workflows.pocket_detection.subprocess.run",
            side_effect=OSError("docker unavailable"),
        ):
            job, payload = run_pocket_detection_job(cleaned.run_id)

    assert job.status == "failed"
    assert payload["success"] is False
    assert "docker unavailable" in job.result["error"]
    assert job.artifact_manifest is not None and job.artifact_manifest.artifacts == ()


def test_worker_executes_queued_fpocket_job_and_finalizes_artifacts(tmp_path: Path) -> None:
    class FakeProcess:
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        output_mount = next(
            command[index + 1]
            for index, item in enumerate(command)
            if item == "-v" and command[index + 1].endswith(":/output")
        )
        output_dir = Path(output_mount.removesuffix(":/output"))
        _write_fake_fpocket_outputs(output_dir)
        kwargs["stdout"].write("fpocket worker output\n")
        return FakeProcess()

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        imported = create_protein_import_job(PDB_DATA, filename="target.pdb", source="upload")
        with patch("mn_ligand.workflows.protein_preparation.subprocess.run", side_effect=_fake_cleaning):
            cleaned, _ = run_protein_cleaning_job(imported.run_id)
        with patch("mn_ligand.workflows.pocket_detection.subprocess.run") as direct_run:
            queued = queue_pocket_detection_job(cleaned.run_id)
        direct_run.assert_not_called()

        assert queued.status == "queued"
        assert queued.metadata["workflow"] == "pocket_detection"
        assert queued.metadata["worker_finalizer"] == "pocket_detection"
        assert queued.metadata["resources"]["gpu"] is False
        assert queued.artifact_manifest is not None
        assert queued.artifact_manifest.artifacts == ()

        worker_result = run_worker_once(
            WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05),
            popen=fake_popen,
            sleep=lambda _: None,
        )

    assert worker_result is not None and worker_result["status"] == "completed"
    completed = JobRecord.load(queued.run_dir, task_group="pocket-detection")
    assert completed.status == "completed"
    assert completed.result["pocket_count"] == 1
    assert completed.artifact_manifest is not None
    assert completed.artifact_manifest.by_type("pocket_set")
    assert completed.artifact_manifest.by_type("pocket")
    assert completed.artifact_manifest.by_type("pocket_points")
    assert "fpocket worker output" in (queued.run_dir / "stdout.log").read_text()
    assert not (queued.run_dir / ".worker-claim.json").exists()


def test_worker_records_invalid_queued_fpocket_output_as_failed(tmp_path: Path) -> None:
    class EmptyProcess:
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        imported = create_protein_import_job(PDB_DATA, filename="target.pdb", source="upload")
        with patch("mn_ligand.workflows.protein_preparation.subprocess.run", side_effect=_fake_cleaning):
            cleaned, _ = run_protein_cleaning_job(imported.run_id)
        queued = queue_pocket_detection_job(cleaned.run_id)
        worker_result = run_worker_once(
            WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05),
            popen=lambda *_args, **_kwargs: EmptyProcess(),
            sleep=lambda _: None,
        )

    assert worker_result is not None and worker_result["status"] == "failed"
    failed = JobRecord.load(queued.run_dir, task_group="pocket-detection")
    assert failed.status == "failed"
    assert failed.result["success"] is False
    assert "Pocket detection failed" in failed.result["error"]
    assert failed.artifact_manifest is not None and failed.artifact_manifest.artifacts == ()
    assert not (queued.run_dir / ".worker-claim.json").exists()


def test_worker_executes_cpu_p2rank_and_publishes_native_tables(tmp_path: Path) -> None:
    class FakeProcess:
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

    observed_command: list[str] = []

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        observed_command.extend(command)
        output_mount = next(
            command[index + 1]
            for index, item in enumerate(command)
            if item == "-v" and command[index + 1].endswith(":/output")
        )
        _write_fake_p2rank_outputs(Path(output_mount.removesuffix(":/output")))
        kwargs["stdout"].write("P2Rank worker output\n")
        return FakeProcess()

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        imported = create_protein_import_job(PDB_DATA, filename="target.pdb", source="upload")
        with patch("mn_ligand.workflows.protein_preparation.subprocess.run", side_effect=_fake_cleaning):
            cleaned, _ = run_protein_cleaning_job(imported.run_id)
        queued = queue_pocket_detection_job(
            cleaned.run_id,
            method="p2rank",
            p2rank_profile="alphafold",
            p2rank_min_probability=0.2,
        )
        worker_result = run_worker_once(
            WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05),
            popen=fake_popen,
            sleep=lambda _: None,
        )

    assert worker_result is not None and worker_result["status"] == "completed"
    assert "--gpus" not in observed_command
    assert observed_command[-2:] == ["-c", "alphafold"]
    completed = JobRecord.load(queued.run_dir, task_group="pocket-detection")
    assert completed.result["method"] == "p2rank"
    assert completed.artifact_manifest is not None
    assert completed.artifact_manifest.by_type("pocket_set")
    assert completed.artifact_manifest.by_type("pocket_score_table")
    assert completed.artifact_manifest.by_type("residue_score_table")
    assert completed.artifact_manifest.by_type("pocket")
    assert completed.artifact_manifest.by_type("pocket_points")
    command_record = json.loads((queued.run_dir / "command.json").read_text())
    assert command_record["tool_id"] == "p2rank"
    assert command_record["resources"]["gpu"] is False


def test_worker_rejects_exit_zero_p2rank_without_native_predictions(tmp_path: Path) -> None:
    class EmptyProcess:
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        imported = create_protein_import_job(PDB_DATA, filename="target.pdb", source="upload")
        with patch("mn_ligand.workflows.protein_preparation.subprocess.run", side_effect=_fake_cleaning):
            cleaned, _ = run_protein_cleaning_job(imported.run_id)
        queued = queue_pocket_detection_job(cleaned.run_id, method="p2rank")
        worker_result = run_worker_once(
            WorkerConfig.create(runs_dir=tmp_path, gpu_ids=(), heartbeat_seconds=0.05),
            popen=lambda *_args, **_kwargs: EmptyProcess(),
            sleep=lambda _: None,
        )

    assert worker_result is not None and worker_result["status"] == "failed"
    failed = JobRecord.load(queued.run_dir, task_group="pocket-detection")
    assert failed.result["error"] == "P2Rank predictions CSV is missing"
    assert failed.artifact_manifest is not None and failed.artifact_manifest.artifacts == ()
