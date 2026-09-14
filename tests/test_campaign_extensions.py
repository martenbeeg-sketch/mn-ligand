from __future__ import annotations

import json
from pathlib import Path

from mn_ligand.core.campaign_extensions import (
    _artifact_signature,
    _extension_commands,
    canonical_campaign_jobs,
    completed_repetitions,
)
from mn_ligand.core.jobs import JobRecord


def _job(tmp_path: Path, workflow: str, command: list[str], parameters: dict) -> JobRecord:
    run_dir = tmp_path / workflow
    run_dir.mkdir()
    (run_dir / "input").mkdir()
    (run_dir / "input.json").write_text(json.dumps({"parameters": parameters}))
    metadata = {
        "run_id": workflow,
        "status": "completed",
        "workflow": workflow,
        "queued_command": command,
        "queued_commands": [command],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    return JobRecord(
        run_id=workflow,
        task_group="test",
        run_dir=run_dir,
        status="completed",
        workflow=workflow,
        metadata=metadata,
    )


def test_artifact_signature_keeps_distinct_prepared_target_runs(tmp_path: Path) -> None:
    jobs = []
    for index, target_run_id in enumerate(("prepared-a", "prepared-b", "prepared-a")):
        run_dir = tmp_path / f"job-{index}"
        run_dir.mkdir()
        (run_dir / "input.json").write_text(
            json.dumps(
                {
                    "target": {
                        "run_id": target_run_id,
                        "artifact_id": "shared-artifact",
                        "sha256": "shared-target-sha",
                    },
                    "compound_artifacts": [{"sha256": "shared-compound-sha"}],
                }
            )
        )
        metadata = {
            "run_id": f"job-{index}",
            "status": "completed",
            "workflow": "boltz2_refolding",
            "engine": "Boltz-2",
        }
        jobs.append(
            JobRecord(
                run_id=f"job-{index}",
                task_group="refolding",
                run_dir=run_dir,
                status="completed",
                workflow="boltz2_refolding",
                metadata=metadata,
            )
        )

    assert _artifact_signature(jobs[0]) != _artifact_signature(jobs[1])
    assert _artifact_signature(jobs[0]) == _artifact_signature(jobs[2])


def test_docking_extension_starts_at_first_missing_repetition(tmp_path: Path) -> None:
    script = 'DOCKING_REPLICATES=1; for replicate in $(seq 1 "$DOCKING_REPLICATES"); do echo "$replicate"; done'
    job = _job(tmp_path, "docking_campaign", ["docker", "run", "image", "bash", "-lc", script], {"replicates": 1})
    result = job.run_dir / "results" / "replicate_001"
    result.mkdir(parents=True)
    (result / "compound_0000001_out.pdbqt").write_text("MODEL\n")
    command = _extension_commands(job, 1, 3)[0]
    assert "DOCKING_REPLICATES=3" in command[-1]
    assert 'DOCKING_REPLICATE_IDS="2 3"' in command[-1]
    assert "for replicate in $DOCKING_REPLICATE_IDS" in command[-1]


def test_docking_extension_reruns_an_internal_hole(tmp_path: Path) -> None:
    script = 'DOCKING_REPLICATES=3; for replicate in $(seq 1 "$DOCKING_REPLICATES"); do true; done'
    job = _job(tmp_path, "docking_campaign", ["docker", "run", "image", "bash", "-lc", script], {"replicates": 3})
    for replicate in (1, 3):
        result = job.run_dir / "results" / f"replicate_{replicate:03d}"
        result.mkdir(parents=True)
        (result / "compound_0000001_out.pdbqt").write_text("MODEL\n")
    command = _extension_commands(job, 2, 3)[0]
    assert 'DOCKING_REPLICATE_IDS="2"' in command[-1]


def test_indexed_refolding_extension_changes_directory_and_seed(tmp_path: Path) -> None:
    command = ["docker", "run", "image", "predict", "--out_dir", "/work/output/replicate_001", "--seed", "1001"]
    job = _job(tmp_path, "boltz2_refolding", command, {"replicates": 1, "seed_start": 1001})
    prediction = job.run_dir / "output/replicate_001/results/predictions/compound_0000001"
    prediction.mkdir(parents=True)
    (prediction / "confidence_compound_0000001_model_0.json").write_text("{}\n")
    (prediction / "compound_0000001_model_0.cif").write_text("data_model\n")
    commands = _extension_commands(job, 1, 3)
    assert [item[item.index("--out_dir") + 1] for item in commands] == [
        "/work/output/replicate_002",
        "/work/output/replicate_003",
    ]
    assert [item[item.index("--seed") + 1] for item in commands] == ["1002", "1003"]


def test_indexed_refolding_extension_reruns_only_missing_ids(tmp_path: Path) -> None:
    command = ["docker", "run", "image", "predict", "--out_dir", "/work/output/replicate_001", "--seed", "1001"]
    job = _job(tmp_path, "boltz2_refolding", command, {"replicates": 3, "seed_start": 1001})
    for replicate in (1, 3):
        prediction = job.run_dir / f"output/replicate_{replicate:03d}/results/predictions/compound_0000001"
        prediction.mkdir(parents=True)
        (prediction / "confidence_compound_0000001_model_0.json").write_text("{}\n")
        (prediction / "compound_0000001_model_0.cif").write_text("data_model\n")
    commands = _extension_commands(job, 2, 3)
    assert len(commands) == 1
    assert commands[0][commands[0].index("--out_dir") + 1] == "/work/output/replicate_002"
    assert commands[0][commands[0].index("--seed") + 1] == "1002"


def test_nesso_extension_reruns_only_validated_missing_id(tmp_path: Path) -> None:
    command = ["docker", "run", "image", "predict", "--out_dir", "/work/output/replicate_001", "--seed", "1001"]
    job = _job(tmp_path, "nesso_affinity", command, {"replicates": 3, "seed_start": 1001})
    inputs = job.run_dir / "inputs"
    inputs.mkdir()
    (inputs / "compound_0000001.yaml").write_text("sequences: []\n")
    for replicate in (1, 3):
        prediction = job.run_dir / f"output/replicate_{replicate:03d}/predictions/compound_0000001"
        prediction.mkdir(parents=True)
        (prediction / "affinity.json").write_text('{"affinity_pred_value": -2.0}\n')
    commands = _extension_commands(job, 2, 3)
    assert len(commands) == 1
    assert commands[0][commands[0].index("--out_dir") + 1] == "/work/output/replicate_002"
    assert commands[0][commands[0].index("--seed") + 1] == "1002"


def test_alphafold_extension_maps_missing_repeat_to_exact_seed(tmp_path: Path) -> None:
    inference = [
        "docker", "run", "image", "python", "run_alphafold.py",
        "--input_dir=/work/data_extension_legacy", "--num_seeds=3",
    ]
    job = _job(
        tmp_path,
        "alphafold3_refolding",
        inference,
        {"model_seed_count": 3, "model_seed_start": 1001},
    )
    inputs = job.run_dir / "inputs"
    inputs.mkdir()
    (inputs / "compound_0000001.json").write_text('{"modelSeeds": [1001]}\n')
    data = job.run_dir / "data"
    data.mkdir()
    (data / "compound_0000001.json").write_text('{"modelSeeds": [1001]}\n')
    for seed in (1001,):
        sample = job.run_dir / f"output/compound_0000001/seed-{seed}_sample-0"
        sample.mkdir(parents=True)
        (sample / f"compound_0000001_seed-{seed}_sample-0_model.cif").write_text("data_model\n")
        (sample / f"compound_0000001_seed-{seed}_sample-0_summary_confidences.json").write_text("{}\n")
    commands = _extension_commands(job, 1, 3)
    assert len(commands) == 1
    assert "--input_dir=/work/data_extension_002_003" in commands[0]
    assert not any(item.startswith("--num_seeds=") for item in commands[0])
    payload = json.loads(
        (job.run_dir / "data_extension_002_003/compound_0000001.json").read_text()
    )
    assert payload["modelSeeds"] == [1002, 1003]


def test_openvs_extension_uses_missing_repetition_ids(tmp_path: Path) -> None:
    runner = 'for replicate in $(seq 1 "$OPENVS_REPLICATES"); do\n  true\ndone\n'
    command = [
        "docker", "run", "--rm", "-e", "OPENVS_REPLICATES=1", "-e",
        "OPENVS_SEED_START=1001", "openvs:local", "bash", "/workspace/input/run_openvs.sh",
    ]
    job = _job(tmp_path, "openvs_docking", command, {"replicates": 1, "seed_start": 1001})
    (job.run_dir / "input" / "run_openvs.sh").write_text(runner)
    native = job.run_dir / "native" / "replicate_001" / "ligands_000"
    native.mkdir(parents=True)
    (native / "run.score.sc").write_text("SCORE\n")
    (native / "run.out").write_text("silent\n")
    poses = job.run_dir / "poses"
    poses.mkdir()
    (poses / "replicate_001_ligands_000_pose.pdb").write_text("ATOM\n")
    extended = _extension_commands(job, 1, 3)[0]
    assert "OPENVS_REPLICATES=3" in extended
    assert "OPENVS_REPLICATE_START=2" in extended
    assert "OPENVS_REPLICATE_IDS=2 3" in extended
    assert "OPENVS_REPLICATE_IDS" in (job.run_dir / "input" / "run_openvs.sh").read_text()


def test_openvs_completed_repetitions_use_native_directories(tmp_path: Path) -> None:
    job = _job(tmp_path, "openvs_docking", ["docker", "run", "openvs:local"], {"replicates": 3})
    for replicate in (1, 2, 3):
        output = job.run_dir / "native" / f"replicate_{replicate:03d}"
        chunk = output / "ligands_000"
        chunk.mkdir(parents=True)
        (chunk / "run.score.sc").write_text("SCORE\n")
        (chunk / "run.out").write_text("silent\n")
        poses = job.run_dir / "poses"
        poses.mkdir(exist_ok=True)
        (poses / f"replicate_{replicate:03d}_ligands_000_pose.pdb").write_text("ATOM\n")
    assert completed_repetitions(job) == 3


def test_openvs_incomplete_nonempty_directory_is_not_completed(tmp_path: Path) -> None:
    command = ["docker", "run", "openvs:local"]
    job = _job(tmp_path, "openvs_docking", command, {"replicates": 3})
    incomplete = job.run_dir / "native" / "replicate_001" / "ligands_000"
    incomplete.mkdir(parents=True)
    (incomplete / "rosetta.log").write_text("failed\n")
    for replicate in (2, 3):
        chunk = job.run_dir / "native" / f"replicate_{replicate:03d}" / "ligands_000"
        chunk.mkdir(parents=True)
        (chunk / "run.score.sc").write_text("SCORE\n")
        (chunk / "run.out").write_text("silent\n")
        poses = job.run_dir / "poses"
        poses.mkdir(exist_ok=True)
        (poses / f"replicate_{replicate:03d}_ligands_000_pose.pdb").write_text("ATOM\n")
    assert completed_repetitions(job) == 2


def test_openvs_recovery_marks_logical_repetition_complete(tmp_path: Path) -> None:
    runner = 'for replicate in $(seq 1 "$OPENVS_REPLICATES"); do\n  true\ndone\n'
    command = [
        "docker", "run", "--rm", "-e", "OPENVS_REPLICATES=1", "-e",
        "OPENVS_SEED_START=1001", "openvs:local", "bash", "/workspace/input/run_openvs.sh",
    ]
    job = _job(tmp_path, "openvs_docking", command, {"replicates": 1, "seed_start": 1001})
    job.metadata["recovered_repetition_indices"] = [1]
    (job.run_dir / "metadata.json").write_text(json.dumps(job.metadata))
    (job.run_dir / "input" / "run_openvs.sh").write_text(runner)

    assert completed_repetitions(job) == 1
    extended = _extension_commands(job, 1, 3)[0]
    assert "OPENVS_REPLICATE_IDS=2 3" in extended
    assert "validate_rosetta_params.py" in (
        job.run_dir / "input" / "run_openvs.sh"
    ).read_text()


def test_openvs_canonical_campaign_prefers_source_over_recovery(tmp_path: Path) -> None:
    jobs: list[JobRecord] = []
    for run_id, recovery_of in (("source-run", ""), ("recovery-run", "source-run")):
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        (run_dir / "input.json").write_text(
            json.dumps(
                {
                    "parameters": {"replicates": 1},
                    "target_artifact": {"run_id": "target-run"},
                    "compound_artifacts": [{"artifact_id": "compound-set"}],
                }
            )
        )
        metadata = {
            "run_id": run_id,
            "status": "completed",
            "workflow": "openvs_docking",
            "engine": "openvs",
            "launch_campaign_id": "campaign-1",
            "recovery_of_run_id": recovery_of,
            "created_at": "2026-08-07T00:00:00+00:00",
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata))
        jobs.append(
            JobRecord(
                run_id=run_id,
                task_group="docking",
                run_dir=run_dir,
                status="completed",
                workflow="openvs_docking",
                metadata=metadata,
            )
        )

    canonical = canonical_campaign_jobs("campaign-1", jobs)

    assert [job.run_id for job in canonical] == ["source-run"]
