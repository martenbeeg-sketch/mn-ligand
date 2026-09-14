from pathlib import Path
import csv
import json
import subprocess
from unittest.mock import patch

import pytest

from mn_ligand.core.artifacts import ArtifactRef
from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.worker import WorkerConfig, run_worker_once
from mn_ligand.workflows.docking import (
    UNIDOCK_PRO_MAX_COMPOUNDS,
    UNIDOCK_PRO_MAX_TORSIONS,
    build_unidock_pro_docker_command,
    build_unidock_pro_native_command,
    _record_engine_timeout_exclusions,
    _record_vina_timeout_exclusions,
    finalize_docking_campaign_job,
    load_compound_records,
    queue_docking_campaign_job,
    run_docking_campaign_job,
)


def test_vina_timeout_is_provenanced_and_reused_within_campaign(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    target_ref = ArtifactRef.from_path(target_dir, receptor, "prepared_target")
    compounds_dir = tmp_path / "compound-job"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text(
        "compound_id,smiles\n"
        "fast,CCO\n"
        "timed-out,CCCC\n"
    )
    compounds_ref = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")
    campaign_id = "campaign-timeout-test"

    with patch.dict(
        "os.environ",
        {
            "MN_LIGAND_APP_HOME": str(tmp_path / "app-home"),
            "MN_LIGAND_RUN_DIR": str(tmp_path / "runs"),
        },
        clear=False,
    ):
        _record_vina_timeout_exclusions(
            campaign_id,
            source_run_id="prior-run",
            compound_ids=["timed-out"],
        )
        queued = queue_docking_campaign_job(
            receptor_path=receptor,
            target_artifact=target_ref,
            compound_paths=[compounds],
            compound_artifacts=[compounds_ref],
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            engine="vina",
            replicates=3,
            compound_timeout_minutes=15,
            launch_campaign_id=campaign_id,
        )

    assert queued.metadata["compound_timeout_minutes"] == 15
    assert queued.metadata["prior_timeout_excluded_compound_ids"] == ["timed-out"]
    staged = (queued.run_dir / "input" / "compounds.tsv").read_text()
    exclusions = (queued.run_dir / "excluded_compounds.tsv").read_text()
    command = json.loads((queued.run_dir / "command.json").read_text())["argv"][-1]
    assert "fast" in staged
    assert "timed-out" not in staged
    assert "timed-out\tprior_vina_timeout" in exclusions
    assert 'timeout --signal=TERM --kill-after=30s "${VINA_COMPOUND_TIMEOUT_MINUTES}m"' in command
    assert 'grep -Fxq "$compound_id" timed_out_compounds.txt' in command


def test_gnina_timeout_is_provenanced_and_reused_within_campaign(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    target_ref = ArtifactRef.from_path(target_dir, receptor, "prepared_target")
    compounds_dir = tmp_path / "compound-job"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text(
        "compound_id,smiles\n"
        "fast,CCO\n"
        "timed-out,CCCC\n"
    )
    compounds_ref = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")
    campaign_id = "campaign-gnina-timeout-test"

    with patch.dict(
        "os.environ",
        {
            "MN_LIGAND_APP_HOME": str(tmp_path / "app-home"),
            "MN_LIGAND_RUN_DIR": str(tmp_path / "runs"),
        },
        clear=False,
    ):
        _record_engine_timeout_exclusions(
            campaign_id,
            engine="gnina",
            source_run_id="prior-run",
            compound_ids=["timed-out"],
        )
        queued = queue_docking_campaign_job(
            receptor_path=receptor,
            target_artifact=target_ref,
            compound_paths=[compounds],
            compound_artifacts=[compounds_ref],
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            engine="gnina",
            replicates=3,
            compound_timeout_minutes=15,
            launch_campaign_id=campaign_id,
        )

    assert queued.metadata["compound_timeout_minutes"] == 15
    assert queued.metadata["prior_timeout_excluded_compound_ids"] == ["timed-out"]
    staged = (queued.run_dir / "input" / "compounds.tsv").read_text()
    exclusions = (queued.run_dir / "excluded_compounds.tsv").read_text()
    command = json.loads((queued.run_dir / "command.json").read_text())["argv"][-1]
    assert "fast" in staged
    assert "timed-out" not in staged
    assert "timed-out\tprior_gnina_timeout" in exclusions
    assert 'timeout --signal=TERM --kill-after=30s "${VINA_COMPOUND_TIMEOUT_MINUTES}m"' in command
    assert 'record_exclusion "$compound_id" "gnina_timeout"' in command
    assert 'grep -Fxq "$compound_id" timed_out_compounds.txt' in command


def test_unidock_pro_classic_batch_command() -> None:
    command = build_unidock_pro_native_command(
        receptor="prepared/receptor.pdbqt",
        ligand_index="batches/001/ligand_index.txt",
        config="config/docking.txt",
        output_dir="results/001",
        mode="classic",
        search_mode="detail",
    )
    assert command[:3] == ["udp", "--receptor", "prepared/receptor.pdbqt"]
    assert command[command.index("--ligand_index") + 1] == "batches/001/ligand_index.txt"
    assert command[-2:] == ["--search_mode", "detail"]
    assert "--reference_ligand" not in command


def test_unidock_pro_rejects_more_than_ten_thousand_compounds(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    target_ref = ArtifactRef.from_path(target_dir, receptor, "prepared_target")
    compounds_dir = tmp_path / "compound-job"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text(
        "compound_id,smiles\n"
        + "".join(
            f"compound-{index},CCO\n"
            for index in range(UNIDOCK_PRO_MAX_COMPOUNDS + 1)
        )
    )
    compounds_ref = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")

    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        with pytest.raises(ValueError, match="at most 10,000 compounds"):
            queue_docking_campaign_job(
                receptor_path=receptor,
                target_artifact=target_ref,
                compound_paths=[compounds],
                compound_artifacts=[compounds_ref],
                center=(0.0, 0.0, 0.0),
                size=(20.0, 20.0, 20.0),
                engine="udp",
            )


def test_unidock_pro_filters_excessive_torsions_before_engine_launch(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    target_ref = ArtifactRef.from_path(target_dir, receptor, "prepared_target")
    compounds_dir = tmp_path / "compound-job"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text(
        "compound_id,smiles\n"
        "small,CCO\n"
        f"too-flexible,{'C' * (UNIDOCK_PRO_MAX_TORSIONS + 5)}\n"
    )
    compounds_ref = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")

    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        queued = queue_docking_campaign_job(
            receptor_path=receptor,
            target_artifact=target_ref,
            compound_paths=[compounds],
            compound_artifacts=[compounds_ref],
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            engine="udp",
        )

    staged = (queued.run_dir / "input" / "compounds.tsv").read_text()
    exclusions = (queued.run_dir / "excluded_compounds.tsv").read_text()
    command = json.loads((queued.run_dir / "command.json").read_text())["argv"]
    assert "small" in staged
    assert "too-flexible" not in staged
    assert "too-flexible\tunidock_pro_preflight" in exclusions
    assert f"at most {UNIDOCK_PRO_MAX_TORSIONS}" in exclusions
    assert "TORSDOF" in command[-1]


def test_unidock_pro_hybrid_requires_both_structures() -> None:
    with pytest.raises(ValueError, match="reference ligand"):
        build_unidock_pro_native_command(
            receptor="prepared/receptor.pdbqt",
            ligand_index="ligand_index.txt",
            config="config.txt",
            output_dir="results",
            mode="hybrid",
        )


def test_unidock_pro_docker_command_selects_gpu_without_shell(tmp_path: Path) -> None:
    command = build_unidock_pro_docker_command(
        image="unidock-pro:latest",
        workspace=tmp_path,
        gpu_devices="1",
        receptor="prepared/receptor.pdbqt",
        ligand_index="ligand_index.txt",
        config="config.txt",
        output_dir="results",
        mode="classic",
        extra_args=("--num_modes", "10"),
    )
    assert command[command.index("--gpus") + 1] == "device=1"
    assert command[command.index("-v") + 1] == f"{tmp_path.resolve()}:/workspace"
    assert "bash" not in command and "-lc" not in command
    assert command[-2:] == ["--num_modes", "10"]


def test_unidock_pro_rejects_host_and_parent_paths() -> None:
    for ligand_index in ("/tmp/ligand_index.txt", "../ligand_index.txt"):
        with pytest.raises(ValueError, match="relative"):
            build_unidock_pro_native_command(
                receptor="prepared/receptor.pdbqt",
                ligand_index=ligand_index,
                config="config.txt",
                output_dir="results",
            )


def test_compound_records_normalize_csv_and_duplicate_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.smi"
    first.write_text("compound_id,smiles\nA-1,CCO\nA-1,CCN\n")
    second.write_text("CCC A-1\n")

    records = load_compound_records([first, second])

    assert [item["compound_id"] for item in records] == ["A-1", "A-1-2", "A-1-3"]
    assert [item["smiles"] for item in records] == ["CCO", "CCN", "CCC"]


def test_docking_campaign_publishes_typed_batch_results(tmp_path: Path) -> None:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\nEND\n")
    target_ref = ArtifactRef.from_path(target_dir, receptor, "prepared_target")
    compounds_dir = tmp_path / "compound-job"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text("compound_id,smiles\nethanol,CCO\nethylamine,CCN\n")
    compounds_ref = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        mount = command[command.index("-v") + 1]
        run_dir = Path(mount.removesuffix(":/workspace"))
        results = run_dir / "results"
        (results / "ethanol_out.pdbqt").write_text("REMARK VINA RESULT: -7.5 0.0 0.0\n")
        (results / "ethanol_out.sdf").write_text("ethanol\n  test\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n")
        (results / "ethylamine_out.pdbqt").write_text("REMARK VINA RESULT: -6.5 0.0 0.0\n")
        (results / "ethylamine_out.sdf").write_text("ethylamine\n  test\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n")
        return subprocess.CompletedProcess(command, 0, stdout="completed", stderr="")

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False):
        with patch("mn_ligand.workflows.docking.subprocess.run", side_effect=fake_run):
            job = run_docking_campaign_job(
                receptor_path=receptor,
                target_artifact=target_ref,
                compound_paths=[compounds],
                compound_artifacts=[compounds_ref],
                center=(1.0, 2.0, 3.0),
                size=(20.0, 21.0, 22.0),
                engine="vina",
                gpu_device="1",
            )

    assert job.status == "completed"
    assert job.metadata["compound_count"] == 2
    assert job.metadata["prepared_target_run_id"] == target_ref.run_id
    assert job.metadata["compound_run_ids"] == [compounds_ref.run_id]
    assert job.result["completed_compounds"] == 2
    assert job.artifact_manifest is not None
    assert job.artifact_manifest.by_type("pose_set")
    assert job.artifact_manifest.by_type("docking_scores")
    command_record = json.loads((job.run_dir / "command.json").read_text())
    assert command_record["tool_id"] == "vina"
    assert "--gpus" not in command_record["argv"]


def test_docking_campaign_quarantines_one_bad_compound_and_completes(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    target_ref = ArtifactRef.from_path(target_dir, receptor, "prepared_target")
    compounds_dir = tmp_path / "compound-job"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text("compound_id,smiles\ngood,CCO\nbad,CCN\n")
    compounds_ref = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        mount = command[command.index("-v") + 1]
        run_dir = Path(mount.removesuffix(":/workspace"))
        results = run_dir / "results"
        (results / "good_out.pdbqt").write_text("REMARK VINA RESULT: -7.5 0.0 0.0\n")
        (results / "good_out.sdf").write_text(
            "good\n  test\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n"
        )
        (run_dir / "excluded_compounds.tsv").write_text(
            "compound_id\tstage\treason\tlog_file\n"
            "bad\tstandardization\tScrub produced no molecule\texclusion_logs/bad.log\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout="completed", stderr="")

    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        with patch("mn_ligand.workflows.docking.subprocess.run", side_effect=fake_run):
            job = run_docking_campaign_job(
                receptor_path=receptor,
                target_artifact=target_ref,
                compound_paths=[compounds],
                compound_artifacts=[compounds_ref],
                center=(1.0, 2.0, 3.0),
                size=(20.0, 20.0, 20.0),
                engine="vina",
            )

    assert job.status == "completed"
    assert job.result["compound_count"] == 2
    assert job.result["eligible_compound_count"] == 1
    assert job.result["excluded_compounds"] == 1
    assert job.result["excluded_compound_ids"] == ["bad"]
    assert job.result["completed_compounds"] == 1
    assert job.result["failed_compounds"] == 0
    assert job.result["progress"] == {"completed": 1, "total": 1}
    assert job.artifact_manifest is not None
    assert job.artifact_manifest.by_type("compound_exclusions")


def test_docking_partial_outputs_complete_with_failed_compound_manifest(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    target_ref = ArtifactRef.from_path(target_dir, receptor, "prepared_target")
    compounds_dir = tmp_path / "compound-job"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text("compound_id,smiles\ngood,CCO\nbad,CCN\n")
    compounds_ref = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")

    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        queued = queue_docking_campaign_job(
            receptor_path=receptor,
            target_artifact=target_ref,
            compound_paths=[compounds],
            compound_artifacts=[compounds_ref],
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            engine="vina",
        )
    (queued.run_dir / "results" / "good_out.pdbqt").write_text(
        "REMARK VINA RESULT: -7.5 0.0 0.0\n"
    )
    (queued.run_dir / "results" / "good_out.sdf").write_text(
        "good\n  test\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n"
    )
    (queued.run_dir / "failed_attempts.tsv").write_text(
        "compound_id\treplicate\treason\nbad\t1\tengine rejected ligand\n"
    )

    completed = finalize_docking_campaign_job(queued.run_dir, returncode=139)

    assert completed.status == "completed"
    assert completed.result["success"] is True
    assert completed.result["partial_success"] is True
    assert completed.result["completed_compounds"] == 1
    assert completed.result["failed_compounds"] == 1
    assert completed.result["failed_attempt_details"][0]["compound_id"] == "bad"


def test_docking_campaign_tags_scrub_bypass_without_excluding_compound(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    target_ref = ArtifactRef.from_path(target_dir, receptor, "prepared_target")
    compounds_dir = tmp_path / "compound-job"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text("compound_id,smiles\nexception,CCO\n")
    compounds_ref = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        mount = command[command.index("-v") + 1]
        run_dir = Path(mount.removesuffix(":/workspace"))
        results = run_dir / "results"
        (results / "exception_out.pdbqt").write_text(
            "REMARK VINA RESULT: -7.5 0.0 0.0\n"
        )
        (results / "exception_out.sdf").write_text(
            "exception\n  test\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n"
        )
        (run_dir / "preparation_exceptions.tsv").write_text(
            "compound_id\texception\treason\tlog_file\n"
            "exception\tscrub_bypass\tScrub empty; direct Meeko input\texception.log\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout="completed", stderr="")

    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        with patch("mn_ligand.workflows.docking.subprocess.run", side_effect=fake_run):
            job = run_docking_campaign_job(
                receptor_path=receptor,
                target_artifact=target_ref,
                compound_paths=[compounds],
                compound_artifacts=[compounds_ref],
                center=(1.0, 2.0, 3.0),
                size=(20.0, 20.0, 20.0),
                engine="vina",
            )

    assert job.status == "completed"
    assert job.result["excluded_compounds"] == 0
    assert job.result["preparation_exceptions"] == 1
    assert job.result["preparation_exception_ids"] == ["exception"]
    score = next(csv.DictReader((job.run_dir / "scores.csv").open()))
    assert score["preparation_exception"] == "scrub_bypass"
    assert score["preparation_exception_reason"] == "Scrub empty; direct Meeko input"


def test_docking_campaign_repeats_with_explicit_seeds_and_summary(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    target_ref = ArtifactRef.from_path(target_dir, receptor, "prepared_target")
    compounds_dir = tmp_path / "compound-job"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text("compound_id,smiles\nethanol,CCO\n")
    compounds_ref = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        mount = command[command.index("-v") + 1]
        run_dir = Path(mount.removesuffix(":/workspace"))
        for replicate, score in enumerate((-8.0, -7.0, -6.0), start=1):
            results = run_dir / "results" / f"replicate_{replicate:03d}"
            results.mkdir()
            (results / "ethanol_out.pdbqt").write_text(
                f"REMARK VINA RESULT: {score} 0.0 0.0\n"
            )
            (results / "ethanol_out.sdf").write_text(
                "ethanol\n  test\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\n"
                "M  END\n$$$$\n"
            )
        return subprocess.CompletedProcess(command, 0, stdout="completed", stderr="")

    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        with patch("mn_ligand.workflows.docking.subprocess.run", side_effect=fake_run):
            job = run_docking_campaign_job(
                receptor_path=receptor,
                target_artifact=target_ref,
                compound_paths=[compounds],
                compound_artifacts=[compounds_ref],
                center=(1.0, 2.0, 3.0),
                size=(20.0, 20.0, 20.0),
                engine="vina",
                replicates=3,
                seed_start=7001,
                cpu_workers=2,
            )

    assert job.status == "completed"
    assert job.metadata["replicates"] == 3
    assert job.metadata["seed_start"] == 7001
    assert job.metadata["cpu_workers"] == 2
    assert job.result["completed_replicate_pairs"] == 3
    score_rows = list(csv.DictReader((job.run_dir / "scores.csv").open()))
    assert [int(row["replicate"]) for row in score_rows] == [1, 2, 3]
    assert [int(row["seed"]) for row in score_rows] == [7001, 7002, 7003]
    summary = next(
        csv.DictReader((job.run_dir / "docking_replicate_summary.csv").open())
    )
    assert float(summary["mean_score_kcal_mol"]) == pytest.approx(-7.0)
    assert float(summary["sample_sd_score_kcal_mol"]) == pytest.approx(1.0)
    assert int(summary["representative_replicate"]) == 2
    assert job.artifact_manifest is not None
    assert any(
        item.role == "replicate_summary"
        for item in job.artifact_manifest.by_type("docking_scores")
    )
    command_record = json.loads((job.run_dir / "command.json").read_text())
    shell = command_record["argv"][-1]
    assert "DOCKING_REPLICATES=3" in shell
    assert "DOCKING_SEED_START=7001" in shell
    assert "DOCKING_CPU_WORKERS=2" in shell
    assert 'xargs -P "$DOCKING_CPU_WORKERS"' in shell
    assert "--cpu 1" in shell
    assert '--seed "$seed"' in shell
    syntax = subprocess.run(
        ["bash", "-n"],
        input=shell,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_gnina_minimized_affinity_is_normalized_as_docking_score(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "t3_out.pdbqt").write_text(
        "REMARK minimizedAffinity -9.67446518\n"
        "REMARK CNNscore 0.985242605\n"
        "REMARK CNNaffinity 8.57378483\n"
    )

    from mn_ligand.workflows.docking import _score_rows

    assert _score_rows(results, "gnina") == [
        {
            "compound_id": "t3",
            "engine": "gnina",
            "best_score_kcal_mol": -9.67446518,
            "cnn_score": 0.985242605,
            "cnn_affinity": 8.57378483,
            "pose_file": "t3_out.pdbqt",
            "pose_index": 1,
            "pose_selection_criterion": "cnn_score",
            "cnn_ranked_pose_index": 1,
            "cnn_ranked_empirical_score_kcal_mol": -9.67446518,
            "cnn_ranked_cnn_score": 0.985242605,
            "cnn_ranked_cnn_affinity": 8.57378483,
            "empirical_ranked_pose_index": 1,
            "empirical_ranked_score_kcal_mol": -9.67446518,
            "empirical_ranked_cnn_score": 0.985242605,
            "empirical_ranked_cnn_affinity": 8.57378483,
        }
    ]


def test_gnina_can_select_cnn_or_empirical_ranked_pose(tmp_path: Path) -> None:
    output = tmp_path / "compound_out.pdbqt"
    output.write_text(
        "MODEL 1\n"
        "REMARK minimizedAffinity -9.0\n"
        "REMARK CNNscore 0.90\n"
        "REMARK CNNaffinity 7.0\n"
        "ENDMDL\n"
        "MODEL 2\n"
        "REMARK minimizedAffinity -11.0\n"
        "REMARK CNNscore 0.60\n"
        "REMARK CNNaffinity 8.0\n"
        "ENDMDL\n"
    )

    from mn_ligand.workflows.docking import select_gnina_pose

    assert select_gnina_pose(output, "cnn_score") == {
        "pose_index": 1,
        "empirical_score_kcal_mol": -9.0,
        "cnn_score": 0.9,
        "cnn_affinity": 7.0,
    }
    assert select_gnina_pose(output, "empirical_score") == {
        "pose_index": 2,
        "empirical_score_kcal_mol": -11.0,
        "cnn_score": 0.6,
        "cnn_affinity": 8.0,
    }


def test_worker_executes_queued_vina_campaign_and_finalizes_artifacts(tmp_path: Path) -> None:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\nEND\n"
    )
    target_ref = ArtifactRef.from_path(target_dir, receptor, "prepared_target")
    compounds_dir = tmp_path / "compound-job"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text("compound_id,smiles\nethanol,CCO\n")
    compounds_ref = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")

    class FakeProcess:
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        mount = command[command.index("-v") + 1]
        run_dir = Path(mount.removesuffix(":/workspace"))
        (run_dir / "results" / "ethanol_out.pdbqt").write_text(
            "REMARK VINA RESULT: -7.5 0.0 0.0\n"
        )
        (run_dir / "results" / "ethanol_out.sdf").write_text(
            "ethanol\n  test\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n"
        )
        kwargs["stdout"].write("vina worker output\n")
        return FakeProcess()

    runs_dir = tmp_path / "runs"
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(runs_dir)}, clear=False):
        with patch("mn_ligand.workflows.docking.subprocess.run") as direct_run:
            queued = queue_docking_campaign_job(
                receptor_path=receptor,
                target_artifact=target_ref,
                compound_paths=[compounds],
                compound_artifacts=[compounds_ref],
                center=(1.0, 2.0, 3.0),
                size=(20.0, 21.0, 22.0),
                engine="vina",
            )
        direct_run.assert_not_called()
        assert queued.status == "queued"
        assert queued.metadata["workflow"] == "docking_campaign"
        assert queued.metadata["worker_finalizer"] == "docking_campaign"
        queued_command = " ".join(queued.metadata["queued_command"])
        assert "mk_prepare_receptor.py --read_pdb input/receptor.pdb" in queued_command
        assert "prepared/receptor-noh.pdb" in queued_command
        assert "if ! mk_prepare_receptor.py" in queued_command
        assert "python input/smiles_to_3d.py" in queued_command
        assert "excluded_compounds.tsv" in queued_command
        assert "record_exclusion" in queued_command
        assert "python input/pdbqt_to_sdf.py" in queued_command
        assert 'obabel "$pose" -osdf' not in queued_command
        shell_script = str(queued.metadata["queued_command"][-1])
        syntax_check = subprocess.run(
            ["bash", "-n"],
            input=shell_script,
            text=True,
            capture_output=True,
            check=False,
        )
        assert syntax_check.returncode == 0, syntax_check.stderr
        converter = queued.run_dir / "input" / "pdbqt_to_sdf.py"
        assert converter.is_file()
        converter_source = converter.read_text()
        assert "PDBQTMolecule" in converter_source
        assert "RDKitMolCreate" in converter_source
        assert "Chem.SanitizeMol(molecule)" in converter_source
        assert queued.metadata["resources"]["gpu"] is False
        assert queued.artifact_manifest is not None and queued.artifact_manifest.artifacts == ()

        worker_result = run_worker_once(
            WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(), heartbeat_seconds=0.05),
            popen=fake_popen,
            sleep=lambda _: None,
        )

    assert worker_result is not None and worker_result["status"] == "completed"
    completed = JobRecord.load(queued.run_dir, task_group="docking")
    assert completed.status == "completed"
    assert completed.result["completed_compounds"] == 1
    assert completed.artifact_manifest is not None
    assert completed.artifact_manifest.by_type("pose_set")
    assert completed.artifact_manifest.by_type("docking_scores")
    assert "vina worker output" in (queued.run_dir / "stdout.log").read_text()
    assert not (queued.run_dir / ".worker-claim.json").exists()


@pytest.mark.parametrize(("engine", "gpu_id"), (("gnina", 0), ("udp", 1)))
def test_gpu_docking_campaign_queues_explicit_worker_lease(
    tmp_path: Path, engine: str, gpu_id: int
) -> None:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    target_ref = ArtifactRef.from_path(target_dir, receptor, "prepared_target")
    compounds_dir = tmp_path / "compound-job"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text("compound_id,smiles\nethanol,CCO\n")
    compounds_ref = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")

    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        with patch("mn_ligand.workflows.docking.subprocess.run") as direct_run:
            queued = queue_docking_campaign_job(
                receptor_path=receptor,
                target_artifact=target_ref,
                compound_paths=[compounds],
                compound_artifacts=[compounds_ref],
                center=(0.0, 0.0, 0.0),
                size=(20.0, 20.0, 20.0),
                engine=engine,
                gpu_device=str(gpu_id),
                search_mode="fast",
            )

    direct_run.assert_not_called()
    assert queued.status == "queued"
    assert queued.metadata["gpu_queued"] is True
    assert queued.metadata["resources"]["gpu"] is True
    assert queued.metadata["resources"]["gpu_ids"] == [gpu_id]
    assert queued.metadata["queued_command"][
        queued.metadata["queued_command"].index("--gpus") + 1
    ] == f"device={gpu_id}"


def test_worker_rejects_empty_vina_native_output(tmp_path: Path) -> None:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    target_ref = ArtifactRef.from_path(target_dir, receptor, "prepared_target")
    compounds_dir = tmp_path / "compound-job"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text("compound_id,smiles\nethanol,CCO\n")
    compounds_ref = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")

    class EmptyProcess:
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

    runs_dir = tmp_path / "runs"
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(runs_dir)}, clear=False):
        queued = queue_docking_campaign_job(
            receptor_path=receptor,
            target_artifact=target_ref,
            compound_paths=[compounds],
            compound_artifacts=[compounds_ref],
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            engine="vina",
        )
        worker_result = run_worker_once(
            WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(), heartbeat_seconds=0.05),
            popen=lambda *_args, **_kwargs: EmptyProcess(),
            sleep=lambda _: None,
        )

    assert worker_result is not None and worker_result["status"] == "failed"
    failed = JobRecord.load(queued.run_dir, task_group="docking")
    assert failed.status == "failed"
    assert failed.result["success"] is False
    assert "no readable poses" in failed.result["error"]
    assert failed.artifact_manifest is not None
    assert not failed.artifact_manifest.by_type("pose_set")
    assert failed.artifact_manifest.by_type("docking_scores")
