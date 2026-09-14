from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from mn_ligand.core.artifacts import ArtifactRef
from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.worker import WorkerConfig, run_worker_once
from mn_ligand.workflows.openvs import (
    _symmetry_pose_rmsd,
    finalize_openvs_docking_job,
    queue_openvs_docking_job,
)


def _inputs(tmp_path: Path) -> tuple[Path, ArtifactRef, Path, ArtifactRef, Path, ArtifactRef]:
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    receptor = target_dir / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\nEND\n"
    )
    target = ArtifactRef.from_path(target_dir, receptor, "prepared_receptor")
    compounds_dir = tmp_path / "compounds"
    compounds_dir.mkdir()
    compounds = compounds_dir / "compounds.csv"
    compounds.write_text("compound_id,smiles\nethanol,CCO\nethylamine,CCN\n")
    compound_set = ArtifactRef.from_path(compounds_dir, compounds, "compound_set")
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference = reference_dir / "reference.sdf"
    reference.write_text("reference\n  test\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n")
    reference_set = ArtifactRef.from_path(
        reference_dir, reference, "prepared_ligand_set", role="ligand"
    )
    return receptor, target, compounds, compound_set, reference, reference_set


def _native_outputs(run_dir: Path, compound_ids: tuple[str, ...] = ("ethanol", "ethylamine")) -> None:
    native = run_dir / "native" / "ligands_000"
    native.mkdir(parents=True, exist_ok=True)
    header = (
        "SCORE: total_score -TdS dG dH lig_rms ligscore recscore time "
        "ligandname description\n"
    )
    rows = []
    poses = run_dir / "poses"
    poses.mkdir(exist_ok=True)
    for index, compound_id in enumerate(compound_ids, start=1):
        description = f"ligands_000_anchor_0001_{index:04d}"
        rows.append(
            f"SCORE: {-100 + index:.3f} 1.200 {-20 + index:.3f} {-21 + index:.3f} "
            f"{index:.3f} {-5 + index:.3f} {-90 + index:.3f} {10 + index:.3f} "
            f"{compound_id} {description}\n"
        )
        (poses / f"ligands_000_{description}.pdb").write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "HETATM    2  C1  LG1 X   2       1.000   1.000   1.000  1.00 20.00           C\nEND\n"
        )
    (native / "run.score.sc").write_text("SEQUENCE:\n" + header + "".join(rows))


def _convergence_outputs(run_dir: Path) -> None:
    values = (-10.0, -11.0, -9.5, -10.5, -10.2)
    poses = run_dir / "poses"
    poses.mkdir(exist_ok=True)
    for replicate, value in enumerate(values, start=1):
        native = run_dir / "native" / f"replicate_{replicate:03d}" / "ligands_000"
        native.mkdir(parents=True)
        description = f"replicate_{replicate:03d}_ligands_000_holo_0001"
        (native / "run.score.sc").write_text(
            "SEQUENCE:\n"
            "SCORE: total_score -TdS dG dH lig_rms ligscore recscore time ligandname description\n"
            f"SCORE: -100.0 2.0 {value} {value - 2.0} 0.0 -5.0 -90.0 100.0 "
            f"ethanol {description}\n"
        )
        shift = 0.05 * replicate
        (poses / f"{description}.pdb").write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            f"HETATM    2  C1  LG1 X   2       {1 + shift:5.3f}   1.000   1.000"
            "  1.00 20.00           C\n"
            f"HETATM    3  O1  LG1 X   2       {2 + shift:5.3f}   1.000   1.000"
            "  1.00 20.00           O\nEND\n"
        )


def test_openvs_queue_is_cpu_only_and_stages_rosetta_protocol(tmp_path: Path) -> None:
    receptor, target, compounds, compound_set, reference, reference_set = _inputs(tmp_path)
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False):
        queued = queue_openvs_docking_job(
            receptor_path=receptor,
            target_artifact=target,
            compound_paths=[compounds],
            compound_artifacts=[compound_set],
            center=(1.0, 2.0, 3.0),
            size=(20.0, 20.0, 20.0),
            protocol="vsh",
            reference_mode="reference_guided",
            reference_ligand_path=reference,
            reference_ligand_artifact=reference_set,
            cpu_workers=12,
            preserve_input_protonation=True,
            maximum_compounds=2,
        )

    assert queued.status == "queued"
    assert queued.workflow == "openvs_docking"
    assert queued.metadata["resources"]["gpu"] is False
    assert queued.metadata["resources"]["cpu_threads"] == 2
    assert queued.metadata["gpu_queued"] is False
    assert "--gpus" not in queued.metadata["queued_command"]
    assert "openvs:local" in queued.metadata["queued_command"]
    xml = (queued.run_dir / "input" / "dock.xml").read_text()
    assert 'runmode="VSH"' in xml
    assert 'final_optH_mode="1"' in xml
    runner = (queued.run_dir / "input" / "run_openvs.sh").read_text()
    assert "mol2genparams.py" in runner
    assert "input/smiles_to_3d.py" in runner
    assert "RDKit ETKDGv3 did not produce" in runner
    assert "excluded_compounds.tsv" in runner
    assert "validate_rosetta_params.py" in runner
    assert "rosetta_params_validation" in runner
    assert "continue" in runner
    assert "--partialcharge mmff94" in runner
    assert queued.metadata["preserve_input_protonation"] is True
    command = " ".join(queued.metadata["queued_command"])
    assert "OPENVS_PRESERVE_INPUT_PROTONATION=1" in command
    assert "extract_pdbs.linuxgccrelease" in runner
    assert '@"$flags" -gen_potential -beta_cart -no_autogen_cart_improper' in runner
    assert queued.artifact_manifest is not None and not queued.artifact_manifest.artifacts


def test_openvs_params_validator_rejects_four_character_atom_name_collisions(
    tmp_path: Path,
) -> None:
    receptor, target, compounds, compound_set, reference, reference_set = _inputs(tmp_path)
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False):
        queued = queue_openvs_docking_job(
            receptor_path=receptor,
            target_artifact=target,
            compound_paths=[compounds],
            compound_artifacts=[compound_set],
            center=(1.0, 2.0, 3.0),
            size=(20.0, 20.0, 20.0),
            reference_ligand_path=reference,
            reference_ligand_artifact=reference_set,
            maximum_compounds=1,
        )
    params = tmp_path / "HY-P2426.params"
    params.write_text(
        "NAME HY-P2426\n"
        "ATOM HC10 HC X 0.058\n"
        "ATOM HC100 HC X 0.058\n"
    )
    validator = queued.run_dir / "input" / "validate_rosetta_params.py"
    completed = subprocess.run(
        [sys.executable, str(validator), str(params)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "HC10 <- HC10, HC100" in completed.stderr


def test_openvs_params_validator_accepts_unique_rosetta_atom_names(
    tmp_path: Path,
) -> None:
    receptor, target, compounds, compound_set, reference, reference_set = _inputs(tmp_path)
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False):
        queued = queue_openvs_docking_job(
            receptor_path=receptor,
            target_artifact=target,
            compound_paths=[compounds],
            compound_artifacts=[compound_set],
            center=(1.0, 2.0, 3.0),
            size=(20.0, 20.0, 20.0),
            reference_ligand_path=reference,
            reference_ligand_artifact=reference_set,
            maximum_compounds=1,
        )
    params = tmp_path / "valid.params"
    params.write_text("NAME valid\nATOM HC10 HC X 0.058\nATOM HC11 HC X 0.058\n")
    validator = queued.run_dir / "input" / "validate_rosetta_params.py"

    completed = subprocess.run(
        [sys.executable, str(validator), str(params)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0


def test_openvs_workers_are_capped_by_global_cpu_limit(tmp_path: Path) -> None:
    receptor, target, compounds, compound_set, reference, reference_set = _inputs(tmp_path)
    with patch.dict(
        "os.environ",
        {
            "MN_LIGAND_RUN_DIR": str(tmp_path / "runs"),
            "MN_LIGAND_CPU_PROCESS_LIMIT": "4",
        },
        clear=False,
    ):
        queued = queue_openvs_docking_job(
            receptor_path=receptor,
            target_artifact=target,
            compound_paths=[compounds],
            compound_artifacts=[compound_set],
            center=(1.0, 2.0, 3.0),
            size=(20.0, 20.0, 20.0),
            reference_ligand_path=reference,
            reference_ligand_artifact=reference_set,
            cpu_workers=32,
            replicates=3,
        )

    assert queued.metadata["cpu_workers"] == 4
    assert queued.metadata["resources"]["cpu_threads"] == 4
    assert "OPENVS_CPU_WORKERS=4" in " ".join(
        queued.metadata["queued_command"]
    )


def test_openvs_pocket_center_mode_disables_reference_pharmacophore(tmp_path: Path) -> None:
    receptor, target, compounds, compound_set, _, _ = _inputs(tmp_path)
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False):
        queued = queue_openvs_docking_job(
            receptor_path=receptor,
            target_artifact=target,
            compound_paths=[compounds],
            compound_artifacts=[compound_set],
            center=(4.0, 5.0, 6.0),
            size=(22.0, 22.0, 22.0),
            protocol="vsx",
            reference_mode="pocket_center",
            maximum_compounds=1,
        )

    xml = (queued.run_dir / "input" / "dock.xml").read_text()
    assert 'runmode="VSX"' in xml
    assert 'reference_pool="none"' in xml
    assert 'use_pharmacophore="0"' in xml
    assert queued.metadata["cpu_workers"] == 1
    runner = (queued.run_dir / "input" / "run_openvs.sh").read_text()
    assert "generated_anchor_params=" in runner
    assert "generated_anchor_pdb=" in runner
    assert (
        'mv "$generated_anchor_params" prepared/anchor/reference_anchor.params'
        in runner
    )
    assert (
        'mv "$generated_anchor_pdb" prepared/anchor/reference_anchor_0001.pdb'
        in runner
    )
    assert (
        "python input/center_pdb.py prepared/anchor/reference_anchor_0001.pdb"
        in runner
    )


def test_openvs_convergence_stages_frozen_multi_seed_tasks(tmp_path: Path) -> None:
    receptor, target, compounds, compound_set, reference, reference_set = _inputs(tmp_path)
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False):
        queued = queue_openvs_docking_job(
            receptor_path=receptor,
            target_artifact=target,
            compound_paths=[compounds],
            compound_artifacts=[compound_set],
            center=(1.0, 2.0, 3.0),
            size=(20.0, 20.0, 20.0),
            protocol="convergence",
            reference_mode="reference_guided",
            reference_ligand_path=reference,
            reference_ligand_artifact=reference_set,
            replicates=5,
            seed_start=7001,
            cpu_workers=8,
            maximum_compounds=1,
        )

    xml = (queued.run_dir / "input" / "dock.xml").read_text()
    assert 'runmode="VSH"' in xml
    assert 'nrelax="30"' in xml
    assert 'repeats="8"' in xml
    assert 'npool="150"' in xml
    assert xml.count("<Stage ") == 2
    assert queued.metadata["replicates"] == 5
    assert queued.metadata["seed_start"] == 7001
    assert queued.metadata["cpu_workers"] == 5
    command = " ".join(queued.metadata["queued_command"])
    assert "OPENVS_REPLICATES=5" in command
    assert "OPENVS_SEED_START=7001" in command
    runner = (queued.run_dir / "input" / "run_openvs.sh").read_text()
    assert "-constant_seed -jran \"$seed\"" in runner
    assert "xargs -P \"$OPENVS_CPU_WORKERS\"" in runner


def test_openvs_convergence_finalizer_reports_running_sd_and_pose_clusters(
    tmp_path: Path,
) -> None:
    receptor, target, compounds, compound_set, reference, reference_set = _inputs(tmp_path)
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False):
        queued = queue_openvs_docking_job(
            receptor_path=receptor,
            target_artifact=target,
            compound_paths=[compounds],
            compound_artifacts=[compound_set],
            center=(1.0, 2.0, 3.0),
            size=(20.0, 20.0, 20.0),
            protocol="convergence",
            reference_mode="reference_guided",
            reference_ligand_path=reference,
            reference_ligand_artifact=reference_set,
            replicates=5,
            seed_start=7001,
            maximum_compounds=1,
        )
    _convergence_outputs(queued.run_dir)
    completed = finalize_openvs_docking_job(queued.run_dir, returncode=0)

    assert completed.status == "completed"
    assert completed.result["replicates"] == 5
    assert completed.result["converged_compounds"] == 1
    assert completed.result["pose_count"] == 5
    assert completed.artifact_manifest is not None
    convergence = completed.artifact_manifest.by_type("convergence_result")
    assert {item.role for item in convergence} == {
        "convergence_summary", "running_statistics"
    }
    summary = next(csv.DictReader((completed.run_dir / "openvs_convergence.csv").open()))
    assert summary["converged"] == "True"
    assert float(summary["sample_sd_dg_reu"]) < 1.0
    assert float(summary["dominant_cluster_fraction"]) == 1.0
    score_rows = list(csv.DictReader((completed.run_dir / "openvs_scores.csv").open()))
    assert [int(row["seed"]) for row in score_rows] == list(range(7001, 7006))
    assert {row["cluster_id"] for row in score_rows} == {"1"}


def test_openvs_standard_vsh_supports_replicates_and_summary(tmp_path: Path) -> None:
    receptor, target, compounds, compound_set, reference, reference_set = _inputs(tmp_path)
    with patch.dict(
        "os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False
    ):
        queued = queue_openvs_docking_job(
            receptor_path=receptor,
            target_artifact=target,
            compound_paths=[compounds],
            compound_artifacts=[compound_set],
            center=(1.0, 2.0, 3.0),
            size=(20.0, 20.0, 20.0),
            protocol="vsh",
            reference_mode="reference_guided",
            reference_ligand_path=reference,
            reference_ligand_artifact=reference_set,
            replicates=5,
            seed_start=8001,
            maximum_compounds=1,
        )
    _convergence_outputs(queued.run_dir)
    completed = finalize_openvs_docking_job(queued.run_dir, returncode=0)

    assert completed.status == "completed"
    assert completed.metadata["replicates"] == 5
    assert completed.result["replicate_summary_compounds"] == 1
    assert (completed.run_dir / "openvs_replicate_summary.csv").is_file()
    assert completed.artifact_manifest is not None
    summaries = completed.artifact_manifest.by_type("convergence_result")
    assert any(item.role == "replicate_summary" for item in summaries)


def test_openvs_pose_rmsd_accounts_for_molecular_symmetry(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared" / "mol2"
    prepared.mkdir(parents=True)
    (prepared / "symmetric.mol2").write_text(
        "@<TRIPOS>MOLECULE\nsymmetric\n 2 1 0 0 0\nSMALL\nNO_CHARGES\n\n"
        "@<TRIPOS>ATOM\n"
        "1 C1 0.0 0.0 0.0 C.3 1 LIG 0.0\n"
        "2 C2 4.0 0.0 0.0 C.3 1 LIG 0.0\n"
        "@<TRIPOS>BOND\n1 1 2 1\n"
    )
    first = tmp_path / "first.pdb"
    second = tmp_path / "second.pdb"
    first.write_text(
        "HETATM    1  C1  LG1 X   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "HETATM    2  C2  LG1 X   1       4.000   0.000   0.000  1.00  0.00           C  \n"
    )
    second.write_text(
        "HETATM    1  C1  LG1 X   1       4.000   0.000   0.000  1.00  0.00           C  \n"
        "HETATM    2  C2  LG1 X   1       0.000   0.000   0.000  1.00  0.00           C  \n"
    )

    assert _symmetry_pose_rmsd(
        tmp_path, "symmetric", first, second
    ) == pytest.approx(0.0)


def test_reference_guided_openvs_requires_coordinate_reference(tmp_path: Path) -> None:
    receptor, target, compounds, compound_set, _, _ = _inputs(tmp_path)
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False):
        with pytest.raises(ValueError, match="coordinate-bearing"):
            queue_openvs_docking_job(
                receptor_path=receptor,
                target_artifact=target,
                compound_paths=[compounds],
                compound_artifacts=[compound_set],
                center=(0.0, 0.0, 0.0),
                size=(20.0, 20.0, 20.0),
            )


def test_openvs_finalizer_normalizes_reu_scores_poses_and_best_complex(tmp_path: Path) -> None:
    receptor, target, compounds, compound_set, _, _ = _inputs(tmp_path)
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False):
        queued = queue_openvs_docking_job(
            receptor_path=receptor,
            target_artifact=target,
            compound_paths=[compounds],
            compound_artifacts=[compound_set],
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            protocol="vsx",
            reference_mode="pocket_center",
        )
    _native_outputs(queued.run_dir)
    completed = finalize_openvs_docking_job(queued.run_dir, returncode=0)

    assert completed.status == "completed"
    assert completed.result["score_units"].startswith("Rosetta energy units")
    assert completed.result["completed_compounds"] == 2
    assert completed.result["pose_count"] == 2
    assert completed.result["best_compound_id"] == "ethanol"
    assert completed.artifact_manifest is not None
    assert len(completed.artifact_manifest.by_type("pose_set")) == 2
    assert completed.artifact_manifest.by_type("docked_complex")
    assert completed.artifact_manifest.by_type("screening_result")
    score_text = (completed.run_dir / "openvs_scores.csv").read_text()
    assert "estimated_dg_reu" in score_text
    assert "kcal" not in score_text.lower()
    assert (completed.run_dir / "best_openvs_complex.pdb").is_file()


def test_openvs_finalizer_reports_preflight_exclusions_as_partial_success(
    tmp_path: Path,
) -> None:
    receptor, target, compounds, compound_set, _, _ = _inputs(tmp_path)
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False):
        queued = queue_openvs_docking_job(
            receptor_path=receptor,
            target_artifact=target,
            compound_paths=[compounds],
            compound_artifacts=[compound_set],
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            protocol="vsx",
            reference_mode="pocket_center",
        )
    _native_outputs(queued.run_dir, ("ethanol",))
    (queued.run_dir / "excluded_compounds.tsv").write_text(
        "compound_id\tstage\treason\tlog_file\n"
        "ethylamine\trosetta_params_validation\tatom-name collision\t"
        "exclusion_logs/ethylamine.preparation.log\n"
    )

    completed = finalize_openvs_docking_job(queued.run_dir, returncode=0)

    assert completed.status == "completed"
    assert completed.result["success"] is True
    assert completed.result["partial_success"] is True
    assert completed.metadata["partial_success"] is True
    assert completed.result["eligible_compound_count"] == 1
    assert completed.result["excluded_compound_ids"] == ["ethylamine"]


def test_openvs_finalizer_preserves_usable_chunks_as_partial_campaign_result(
    tmp_path: Path,
) -> None:
    receptor, target, compounds, compound_set, _, _ = _inputs(tmp_path)
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")}, clear=False):
        queued = queue_openvs_docking_job(
            receptor_path=receptor,
            target_artifact=target,
            compound_paths=[compounds],
            compound_artifacts=[compound_set],
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            protocol="vsx",
            reference_mode="pocket_center",
        )
    _native_outputs(queued.run_dir, ("ethanol",))
    (queued.run_dir / "stderr.log").write_text(
        "One RosettaLigand chunk failed; healthy chunks were retained\n"
    )

    completed = finalize_openvs_docking_job(queued.run_dir, returncode=4)

    assert completed.status == "completed"
    assert completed.result["success"] is True
    assert completed.result["partial_success"] is True
    assert completed.metadata["partial_success"] is True
    assert completed.result["completed_compounds"] == 1
    assert completed.result["failed_compounds"] == 1
    assert "chunk failed" in completed.result["warning"]
    assert completed.result["error"] == ""


def test_openvs_worker_rejects_missing_native_output(tmp_path: Path) -> None:
    receptor, target, compounds, compound_set, _, _ = _inputs(tmp_path)

    class EmptyProcess:
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    runs_dir = tmp_path / "runs"
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(runs_dir)}, clear=False):
        queued = queue_openvs_docking_job(
            receptor_path=receptor,
            target_artifact=target,
            compound_paths=[compounds],
            compound_artifacts=[compound_set],
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            reference_mode="pocket_center",
            maximum_compounds=1,
        )
        worker_result = run_worker_once(
            WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(), heartbeat_seconds=0.05),
            popen=lambda *_args, **_kwargs: EmptyProcess(),
            sleep=lambda _: None,
        )

    assert worker_result is not None and worker_result["status"] == "failed"
    failed = JobRecord.load(queued.run_dir, task_group="docking")
    assert failed.status == "failed"
    assert "0/1 replicate score sets" in failed.result["error"]
    command_record = json.loads((queued.run_dir / "command.json").read_text())
    assert command_record["resources"]["gpu"] is False
