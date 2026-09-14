import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mn_ligand.core.artifacts import ArtifactRef
from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.worker import WorkerConfig, run_worker_once
from mn_ligand.app.pages.discover_inputs import artifact_options
from mn_ligand.workflows.refolding import (
    apply_cached_msas,
    alphafast_readiness,
    alphafold3_input,
    boltz2_input_yaml,
    build_boltz2_command,
    build_nesso_command,
    build_alphafast_commands,
    limit_compounds,
    ligand_bound_protein_sequence,
    nesso_input_yaml,
    nesso_readiness,
    queue_alphafold3_msa_job,
    queue_alphafold3_refolding_job,
    queue_boltz2_refolding_job,
    queue_nesso_affinity_job,
    msa_repository_path,
    portable_command_record,
    protein_sequences_from_pdb,
    polymer_sequences_from_pdb,
)


def _typed_refolding_inputs(tmp_path: Path) -> tuple[Path, ArtifactRef, Path, ArtifactRef]:
    target_dir = tmp_path / "target-job"
    target_dir.mkdir()
    target = target_dir / "target.pdb"
    target.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1       1.400   0.000   0.000  1.00 20.00           C\nEND\n"
    )
    target_ref = ArtifactRef.from_path(target_dir, target, "prepared_target")
    compound_dir = tmp_path / "compound-job"
    compound_dir.mkdir()
    compounds = compound_dir / "compounds.csv"
    compounds.write_text("compound_id,smiles\nethanol,CCO\n")
    compound_ref = ArtifactRef.from_path(compound_dir, compounds, "compound_set")
    return target, target_ref, compounds, compound_ref


class _ImmediateProcess:
    returncode = 0

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode


def test_alphafold3_input_contains_protein_chains_and_ligand() -> None:
    payload = alphafold3_input("cmp 1", [("A", "ACD"), ("B", "GG")], "CCO", model_seeds=(1, 2))
    assert payload["dialect"] == "alphafold3"
    assert payload["name"] == "cmp-1"
    assert payload["modelSeeds"] == [1, 2]
    assert payload["sequences"][-1] == {"ligand": {"id": "L", "smiles": "CCO"}}


def test_alphafold3_input_rejects_invalid_ligand_smiles() -> None:
    with pytest.raises(ValueError, match="Invalid ligand SMILES"):
        alphafold3_input("bad", [("A", "ACD")], "not a molecule")


def test_boltz2_input_contains_typed_protein_ligand_and_affinity() -> None:
    payload = boltz2_input_yaml([("A", "ACD")], "CCO")

    assert "version: 1" in payload
    assert 'id: "A"' in payload
    assert 'smiles: "CCO"' in payload
    assert "affinity:" in payload


def test_boltz2_input_accepts_explicit_offline_msa() -> None:
    payload = boltz2_input_yaml(
        [("A", "ACD")], "CCO", protein_msa_paths=("/work/msa/protein_chain_1.csv",)
    )

    assert 'msa: "/work/msa/protein_chain_1.csv"' in payload


def test_boltz2_command_allocates_shared_memory_and_can_run_offline(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    run_dir.mkdir()
    cache_dir.mkdir()

    command = build_boltz2_command(
        image="ovoex-boltz2:latest", run_dir=run_dir, cache_dir=cache_dir,
        use_msa_server=False,
    )

    assert command[command.index("--shm-size") + 1] == "8g"
    assert "--use_msa_server" not in command

    repeated = build_boltz2_command(
        image="ovoex-boltz2:latest", run_dir=run_dir, cache_dir=cache_dir,
        output_dir="/work/output/replicate_002", seed=1002,
    )
    assert repeated[repeated.index("--out_dir") + 1] == "/work/output/replicate_002"
    assert repeated[repeated.index("--seed") + 1] == "1002"


def test_nesso_input_contains_sequence_smiles_and_affinity_binder() -> None:
    payload = nesso_input_yaml([("A", "ACD")], "CCO")

    assert 'id: "A"' in payload
    assert "sequence: ACD" in payload
    assert 'smiles: "CCO"' in payload
    assert "binder:" in payload


def _nesso_references(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / "references" / "nesso" / "v1.0.0"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "hparams.json").write_text("{}")
    ccd = checkpoint.parent / "ccd.pkl"
    ccd.write_bytes(b"trusted publisher fixture")
    esm_cache = tmp_path / "references" / "shared-hf-cache"
    snapshot = (
        esm_cache / "models--facebook--esm2_t33_650M_UR50D" / "snapshots" / "fixture"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"esm")
    (snapshot / "config.json").write_text("{}")
    (snapshot / "vocab.txt").write_text("<cls>\n")
    return checkpoint, ccd, esm_cache


def test_nesso_readiness_and_command_reuse_esm_cache_read_only(tmp_path: Path) -> None:
    checkpoint, ccd, esm_cache = _nesso_references(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    readiness = nesso_readiness(checkpoint, ccd, esm_cache)
    command = build_nesso_command(
        image="ovolig-nesso-cu128:latest", run_dir=run_dir,
        checkpoint_dir=checkpoint, ccd_path=ccd, esm_cache_dir=esm_cache,
        gpu_device="1",
    )

    assert readiness["ready"] is True
    assert command[command.index("--gpus") + 1] == "device=1"
    assert f"{esm_cache.resolve()}:/cache/huggingface:ro" in command
    assert "HF_HUB_OFFLINE=1" in command
    assert "--no_kernels" in command
    assert "--require_affinity" in command


def test_zero_compound_limit_keeps_the_complete_dataset() -> None:
    compounds = [("one", "CCO"), ("two", "CCN")]

    assert limit_compounds(compounds, 0) == compounds
    assert limit_compounds(compounds, 1) == compounds[:1]
    with pytest.raises(ValueError, match="cannot be negative"):
        limit_compounds(compounds, -1)


def test_protein_sequences_from_pdb_keeps_chains_and_residues_once(tmp_path: Path) -> None:
    pdb = tmp_path / "target.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      1.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  N   GLY B   2      0.000   1.000   0.000  1.00 20.00           N\n"
    )
    assert protein_sequences_from_pdb(pdb) == [("A", "A"), ("B", "G")]


def test_ligand_bound_protein_sequence_selects_contacting_chain(tmp_path: Path) -> None:
    receptor = tmp_path / "target.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      2  CA  GLY B   1      10.000   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      3  CB  GLY B   1      10.500   0.000   0.000  1.00 20.00           C  \n"
        "END\n"
    )
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=17) == 0
    conformer = molecule.GetConformer()
    for index in range(molecule.GetNumAtoms()):
        point = conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(index, (point.x + 10.0, point.y, point.z))
    ligand = tmp_path / "ligand.sdf"
    writer = Chem.SDWriter(str(ligand))
    writer.write(molecule)
    writer.close()

    selected = ligand_bound_protein_sequence(receptor, ligand)

    assert selected["chain"] == "B"
    assert selected["sequence"] == "G"
    assert selected["contacting_protein_atoms"] == 2
    assert selected["available_protein_chains"] == ["A", "B"]


def test_refolding_inputs_preserve_dna_and_rna_entities(tmp_path: Path) -> None:
    pdb = tmp_path / "mixed-polymers.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      2  P    DA B   1       1.000   0.000   0.000  1.00 20.00           P  \n"
        "ATOM      3  P     U C   1       2.000   0.000   0.000  1.00 20.00           P  \n"
        "END\n"
    )
    polymers = polymer_sequences_from_pdb(pdb)
    assert polymers == [
        ("protein", "A", "A"),
        ("dna", "B", "A"),
        ("rna", "C", "U"),
    ]
    af3 = alphafold3_input("mixed", polymers, "CCO")
    assert [next(iter(entity)) for entity in af3["sequences"][:-1]] == [
        "protein",
        "dna",
        "rna",
    ]
    boltz = boltz2_input_yaml(polymers, "CCO")
    assert "  - protein:" in boltz
    assert "  - dna:" in boltz
    assert "  - rna:" in boltz


def test_alphafast_commands_mount_configured_paths_and_gpu(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    db_dir = tmp_path / "db"
    weights_dir = tmp_path / "weights"
    commands = build_alphafast_commands(
        image="alphafast:latest", run_dir=run_dir, db_dir=db_dir,
        weights_dir=weights_dir, gpu_device="1", batch_size=4, num_recycles=7,
    )
    assert len(commands) == 2
    assert commands[0][commands[0].index("--gpus") + 1] == "device=1"
    assert f"{run_dir.resolve()}:/work" in commands[0]
    assert f"{weights_dir.resolve()}:/data/models:ro" in commands[1]
    assert "--num_recycles=7" in commands[1]
    assert "bash" not in commands[0] and "-lc" not in commands[0]


def test_cached_msa_is_inlined_using_shared_sequence_hash(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    repository = tmp_path / "msa_repository"
    input_dir.mkdir()
    repository.mkdir()
    payload = alphafold3_input("cmp", [("A", "ACD")], "CCO")
    (input_dir / "cmp.json").write_text(json.dumps(payload))
    msa_repository_path("ACD", repository).write_text(">query\nACD\n>hit\nACD\n")

    metrics = apply_cached_msas(input_dir, repository)

    staged = json.loads((input_dir / "cmp.json").read_text())
    protein = staged["sequences"][0]["protein"]
    assert metrics == {"unique_hit_count": 1, "unique_miss_count": 0}
    assert protein["unpairedMsa"].startswith(">query\nACD")
    assert protein["pairedMsa"] == ""
    assert protein["templates"] == []


def test_persisted_command_record_contains_no_absolute_run_or_reference_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "job"
    db_dir = tmp_path / "references" / "alignment"
    weights_dir = tmp_path / "references" / "alphafold3"
    commands = build_alphafast_commands(
        image="alphafast:latest", run_dir=run_dir, db_dir=db_dir, weights_dir=weights_dir
    )
    record = portable_command_record(
        commands, run_dir=run_dir, db_dir=db_dir, weights_dir=weights_dir
    )
    serialized = json.dumps(record)
    assert str(tmp_path) not in serialized
    assert "${RUN_DIR}" in serialized
    assert "${REFERENCE_DIR}" in serialized


def test_alphafast_readiness_requires_mmseqs_and_weights(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    weights_dir = tmp_path / "weights"
    (db_dir / "mmseqs").mkdir(parents=True)
    weights_dir.mkdir()
    (weights_dir / "af3.bin.zst").write_bytes(b"weights")
    status = alphafast_readiness(db_dir, weights_dir)
    assert status["database_ready"] is True
    assert status["weights_ready"] is True


def test_cofolding_queue_defaults_allow_any_available_gpu(tmp_path: Path) -> None:
    target, target_ref, compounds, compound_ref = _typed_refolding_inputs(tmp_path)
    db_dir = tmp_path / "references" / "alignment"
    (db_dir / "mmseqs").mkdir(parents=True)
    weights_dir = tmp_path / "references" / "alphafold3"
    weights_dir.mkdir(parents=True)
    (weights_dir / "af3.bin.zst").write_bytes(b"weights")
    msa_repository = tmp_path / "references" / "boltz_models" / "msa_repository"
    msa_repository.mkdir(parents=True)
    cache = tmp_path / "references" / "boltz_models"
    (cache / "boltz2_conf.ckpt").write_bytes(b"structure")
    (cache / "boltz2_aff.ckpt").write_bytes(b"affinity")
    checkpoint, ccd, esm_cache = _nesso_references(tmp_path)

    with patch.dict(
        "os.environ",
        {"MN_LIGAND_RUN_DIR": str(tmp_path / "runs")},
        clear=False,
    ):
        jobs = (
            queue_alphafold3_msa_job(
                protein_sequence="ACDEFGHIKLMNPQRSTVWY",
                target_artifact=target_ref,
                db_dir=db_dir,
                weights_dir=weights_dir,
                msa_repository_dir=msa_repository,
            ),
            queue_alphafold3_refolding_job(
                target_path=target,
                target_artifact=target_ref,
                compound_paths=[compounds],
                compound_artifacts=[compound_ref],
                db_dir=db_dir,
                weights_dir=weights_dir,
                msa_repository_dir=msa_repository,
            ),
            queue_boltz2_refolding_job(
                target_path=target,
                target_artifact=target_ref,
                compound_paths=[compounds],
                compound_artifacts=[compound_ref],
                cache_dir=cache,
            ),
            queue_nesso_affinity_job(
                target_path=target,
                target_artifact=target_ref,
                compound_paths=[compounds],
                compound_artifacts=[compound_ref],
                checkpoint_dir=checkpoint,
                ccd_path=ccd,
                esm_cache_dir=esm_cache,
            ),
        )

    for job in jobs:
        assert job.metadata["gpu_device"] == "all"
        assert job.metadata["resources"]["gpu"] is True
        assert "gpu_ids" not in job.metadata["resources"]
        command = job.metadata["queued_command"]
        assert command[command.index("--gpus") + 1] == "all"


def test_worker_runs_af3_pipeline_and_inference_then_finalizes(tmp_path: Path) -> None:
    target, target_ref, compounds, compound_ref = _typed_refolding_inputs(tmp_path)
    db_dir = tmp_path / "references" / "alignment"
    (db_dir / "mmseqs").mkdir(parents=True)
    weights_dir = tmp_path / "references" / "alphafold3"
    weights_dir.mkdir(parents=True)
    (weights_dir / "af3.bin.zst").write_bytes(b"weights")
    msa_repository = tmp_path / "references" / "boltz_models" / "msa_repository"
    msa_repository.mkdir(parents=True)
    runs_dir = tmp_path / "runs"
    calls = 0

    def fake_popen(_command: list[str], **kwargs: object) -> _ImmediateProcess:
        nonlocal calls
        calls += 1
        if calls == 1:
            (queued.run_dir / "data").mkdir(exist_ok=True)
        else:
            candidate = queued.run_dir / "output" / "ethanol"
            candidate.mkdir(parents=True)
            for sample in range(2):
                sample_dir = candidate / f"seed-41_sample-{sample}"
                sample_dir.mkdir()
                (sample_dir / f"ethanol_seed-41_sample-{sample}_summary_confidences.json").write_text(
                    json.dumps(
                        {
                            "ranking_score": 0.8 - sample * 0.1,
                            "iptm": 0.7,
                            "ptm": 0.6,
                        }
                    )
                )
                (sample_dir / f"ethanol_seed-41_sample-{sample}_model.cif").write_text(
                    "data_ethanol\n"
                )
        kwargs["stdout"].write(f"command {calls}\n")
        return _ImmediateProcess()

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(runs_dir)}, clear=False):
        queued = queue_alphafold3_refolding_job(
            target_path=target,
            target_artifact=target_ref,
            compound_paths=[compounds],
            compound_artifacts=[compound_ref],
            reference_ligand_artifact=target_ref,
            db_dir=db_dir,
            weights_dir=weights_dir,
            msa_repository_dir=msa_repository,
            gpu_device="1",
            protein_sequences=(("A", "ACDEFGHIKLMNPQRSTVWY"),),
            model_seed_count=2,
            model_seed_start=41,
            launch_context="structure_import",
        )
        assert len(queued.metadata["queued_commands"]) == 2
        assert queued.metadata["resources"]["gpu_ids"] == [1]
        assert queued.metadata["protein_input_mode"] == "sequence"
        assert queued.metadata["launch_context"] == "structure_import"
        queued_input = json.loads(next((queued.run_dir / "inputs").glob("*.json")).read_text())
        assert queued_input["modelSeeds"] == [41, 42]
        stored_input = json.loads((queued.run_dir / "input.json").read_text())
        assert (
            stored_input["reference_ligand_artifact"]["artifact_id"]
            == target_ref.artifact_id
        )
        stored_parameters = stored_input["parameters"]
        assert stored_parameters["model_seed_start"] == 41
        assert stored_parameters["model_seed_count"] == 2
        result = run_worker_once(
            WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(1,), heartbeat_seconds=0.05),
            popen=fake_popen,
            sleep=lambda _: None,
        )

    assert calls == 2
    assert result is not None and result["status"] == "completed" and result["gpu_id"] == 1
    completed = JobRecord.load(queued.run_dir, task_group="refolding")
    assert completed.artifact_manifest is not None
    complexes = completed.artifact_manifest.by_type("predicted_complex")
    confidences = completed.artifact_manifest.by_type("prediction_confidence")
    assert len(complexes) == 2
    assert len(confidences) == 2
    assert {item.role for item in complexes} == {
        "ethanol:seed-41_sample-0",
        "ethanol:seed-41_sample-1",
    }
    assert completed.artifact_manifest.by_type("prediction_metrics")


def test_worker_finalizes_boltz2_structure_confidence_and_affinity(tmp_path: Path) -> None:
    target, target_ref, compounds, compound_ref = _typed_refolding_inputs(tmp_path)
    cache = tmp_path / "references" / "boltz_models"
    cache.mkdir(parents=True)
    (cache / "boltz2_conf.ckpt").write_bytes(b"structure")
    (cache / "boltz2_aff.ckpt").write_bytes(b"affinity")
    runs_dir = tmp_path / "runs"

    def fake_popen(_command: list[str], **kwargs: object) -> _ImmediateProcess:
        candidate = (
            queued.run_dir
            / "output"
            / "boltz_results_ethanol"
            / "predictions"
            / "ethanol"
        )
        candidate.mkdir(parents=True)
        (candidate / "ethanol_model_0.cif").write_text("data_ethanol\n")
        (candidate / "confidence_ethanol_model_0.json").write_text(
            json.dumps({"confidence_score": 0.9, "iptm": 0.8, "ptm": 0.7})
        )
        (candidate / "affinity_ethanol.json").write_text(
            json.dumps({"affinity_pred_value": -2.5, "affinity_probability_binary": 0.95})
        )
        kwargs["stdout"].write("boltz worker output\n")
        return _ImmediateProcess()

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(runs_dir)}, clear=False):
        queued = queue_boltz2_refolding_job(
            target_path=target,
            target_artifact=target_ref,
            compound_paths=[compounds],
            compound_artifacts=[compound_ref],
            reference_ligand_artifact=target_ref,
            cache_dir=cache,
            gpu_device="0",
            use_msa_server=False,
            protein_sequences=(("A", "ACDEFGHIKLMNPQRSTVWY"),),
        )
        assert queued.metadata["resources"]["gpu_ids"] == [0]
        assert queued.metadata["protein_input_mode"] == "sequence"
        assert 'msa: "empty"' in next(
            (queued.run_dir / "inputs").glob("*.yaml")
        ).read_text()
        assert json.loads((queued.run_dir / "input.json").read_text())[
            "reference_ligand_artifact"
        ]["artifact_id"] == target_ref.artifact_id
        result = run_worker_once(
            WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(0,), heartbeat_seconds=0.05),
            popen=fake_popen,
            sleep=lambda _: None,
        )

    assert result is not None and result["status"] == "completed" and result["gpu_id"] == 0
    completed = JobRecord.load(queued.run_dir, task_group="refolding")
    assert completed.result["prediction_count"] == 1
    assert completed.result["affinity_count"] == 1
    assert completed.artifact_manifest is not None
    assert completed.artifact_manifest.by_type("predicted_complex")
    assert completed.artifact_manifest.by_type("prediction_confidence")
    assert completed.artifact_manifest.by_type("affinity_result")
    assert completed.artifact_manifest.by_type("prediction_metrics")


def test_worker_aggregates_independent_boltz2_runs(tmp_path: Path) -> None:
    target, target_ref, compounds, compound_ref = _typed_refolding_inputs(tmp_path)
    cache = tmp_path / "references" / "boltz_models"
    cache.mkdir(parents=True)
    (cache / "boltz2_conf.ckpt").write_bytes(b"structure")
    (cache / "boltz2_aff.ckpt").write_bytes(b"affinity")
    runs_dir = tmp_path / "runs"
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> _ImmediateProcess:
        calls.append(command)
        replicate = len(calls)
        candidate = (
            queued.run_dir / "output" / f"replicate_{replicate:03d}"
            / "boltz_results_ethanol" / "predictions" / "ethanol"
        )
        candidate.mkdir(parents=True)
        (candidate / "ethanol_model_0.cif").write_text("data_ethanol\n")
        (candidate / "confidence_ethanol_model_0.json").write_text(
            json.dumps({"confidence_score": 0.8 + 0.1 * replicate})
        )
        (candidate / "affinity_ethanol.json").write_text(
            json.dumps(
                {
                    "affinity_pred_value": -3.0 + replicate,
                    "affinity_probability_binary": 0.7 + 0.1 * replicate,
                }
            )
        )
        kwargs["stdout"].write(f"Boltz replicate {replicate}\n")
        return _ImmediateProcess()

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(runs_dir)}, clear=False):
        queued = queue_boltz2_refolding_job(
            target_path=target, target_artifact=target_ref,
            compound_paths=[compounds], compound_artifacts=[compound_ref],
            cache_dir=cache, gpu_device="0", replicates=2, seed_start=1001,
        )
        result = run_worker_once(
            WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(0,), heartbeat_seconds=0.05),
            popen=fake_popen, sleep=lambda _: None,
        )

    assert result is not None and result["status"] == "completed"
    assert len(calls) == 2
    assert [command[command.index("--seed") + 1] for command in calls] == ["1001", "1002"]
    completed = JobRecord.load(queued.run_dir, task_group="refolding")
    assert completed.result["completed_replicate_pairs"] == 2
    summary_ref = next(
        item for item in completed.artifact_manifest.by_type("prediction_metrics")
        if item.role == "replicate_summary"
    )
    row = next(csv.DictReader((queued.run_dir / summary_ref.path).open()))
    assert float(row["mean_affinity_pred_value"]) == pytest.approx(-1.5)
    assert float(row["sample_sd_affinity_pred_value"]) == pytest.approx(2 ** -0.5)


def test_boltz2_exit_zero_without_structure_is_failed(tmp_path: Path) -> None:
    target, target_ref, compounds, compound_ref = _typed_refolding_inputs(tmp_path)
    cache = tmp_path / "references" / "boltz_models"
    cache.mkdir(parents=True)
    (cache / "boltz2_conf.ckpt").write_bytes(b"structure")
    (cache / "boltz2_aff.ckpt").write_bytes(b"affinity")
    runs_dir = tmp_path / "runs"

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(runs_dir)}, clear=False):
        queued = queue_boltz2_refolding_job(
            target_path=target,
            target_artifact=target_ref,
            compound_paths=[compounds],
            compound_artifacts=[compound_ref],
            cache_dir=cache,
            gpu_device="0",
        )
        result = run_worker_once(
            WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(0,), heartbeat_seconds=0.05),
            popen=lambda *_args, **_kwargs: _ImmediateProcess(),
            sleep=lambda _: None,
        )

    assert result is not None and result["status"] == "failed"
    failed = JobRecord.load(queued.run_dir, task_group="refolding")
    assert failed.result["success"] is False
    assert "no readable predicted complexes" in failed.result["error"]
    assert failed.artifact_manifest is not None
    failures = failed.artifact_manifest.by_type("compound_exclusions")
    assert failures and failures[0].role == "failed_boltz2_compounds"


def test_boltz2_nonzero_exit_preserves_partial_compound_results(
    tmp_path: Path,
) -> None:
    target, target_ref, compounds, compound_ref = _typed_refolding_inputs(tmp_path)
    compounds.write_text("compound_id,smiles\nethanol,CCO\nmethanol,CO\n")
    cache = tmp_path / "references" / "boltz_models"
    cache.mkdir(parents=True)
    (cache / "boltz2_conf.ckpt").write_bytes(b"structure")
    (cache / "boltz2_aff.ckpt").write_bytes(b"affinity")
    runs_dir = tmp_path / "runs"

    def fake_popen(_command: list[str], **kwargs: object) -> _ImmediateProcess:
        candidate = (
            queued.run_dir
            / "output"
            / "boltz_results_ethanol"
            / "predictions"
            / "ethanol"
        )
        candidate.mkdir(parents=True)
        (candidate / "ethanol_model_0.cif").write_text("data_ethanol\n")
        (candidate / "confidence_ethanol_model_0.json").write_text(
            json.dumps({"confidence_score": 0.9})
        )
        kwargs["stderr"].write("methanol failed\n")
        process = _ImmediateProcess()
        process.returncode = 1
        return process

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(runs_dir)}, clear=False):
        queued = queue_boltz2_refolding_job(
            target_path=target,
            target_artifact=target_ref,
            compound_paths=[compounds],
            compound_artifacts=[compound_ref],
            cache_dir=cache,
            gpu_device="0",
            use_msa_server=False,
        )
        result = run_worker_once(
            WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(0,), heartbeat_seconds=0.05),
            popen=fake_popen,
            sleep=lambda _: None,
        )

    assert result is not None and result["status"] == "completed"
    completed = JobRecord.load(queued.run_dir, task_group="refolding")
    assert completed.result["success"] is True
    assert completed.result["partial_success"] is True
    assert completed.result["completed_compounds"] == 1
    assert completed.result["failed_compound_ids"] == ["methanol"]
    failure_artifacts = completed.artifact_manifest.by_type("compound_exclusions")
    assert failure_artifacts and failure_artifacts[0].role == "failed_boltz2_compounds"


def test_worker_finalizes_nesso_affinity_without_structure_artifact(tmp_path: Path) -> None:
    target, target_ref, compounds, compound_ref = _typed_refolding_inputs(tmp_path)
    checkpoint, ccd, esm_cache = _nesso_references(tmp_path)
    runs_dir = tmp_path / "runs"

    def fake_popen(_command: list[str], **kwargs: object) -> _ImmediateProcess:
        candidate = queued.run_dir / "output" / "predictions" / "ethanol"
        candidate.mkdir(parents=True)
        (candidate / "affinity.json").write_text(
            json.dumps(
                {
                    "affinity_pred_value": -2.5,
                    "affinity_pred_value1": -2.4,
                    "affinity_pred_value2": -2.6,
                    "affinity_probability_binary": 0.95,
                    "entropy_crop_pl": 1.25,
                }
            )
        )
        kwargs["stdout"].write("Nesso native worker output\n")
        return _ImmediateProcess()

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(runs_dir)}, clear=False):
        queued = queue_nesso_affinity_job(
            target_path=target, target_artifact=target_ref,
            compound_paths=[compounds], compound_artifacts=[compound_ref],
            checkpoint_dir=checkpoint, ccd_path=ccd, esm_cache_dir=esm_cache,
            gpu_device="1",
        )
        result = run_worker_once(
            WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(1,), heartbeat_seconds=0.05),
            popen=fake_popen, sleep=lambda _: None,
        )
        handoff = artifact_options(("affinity_result",))

    assert result is not None and result["status"] == "completed" and result["gpu_id"] == 1
    completed = JobRecord.load(queued.run_dir, task_group="refolding")
    assert completed.result["affinity_count"] == 1
    assert completed.result["best_pIC50"] == pytest.approx(8.5)
    assert completed.result["structure_output"] is False
    assert completed.artifact_manifest is not None
    assert completed.artifact_manifest.by_type("affinity_result")
    assert completed.artifact_manifest.by_type("prediction_metrics")
    assert completed.artifact_manifest.by_type("native_output")
    assert not completed.artifact_manifest.by_type("predicted_complex")
    assert any(choice.job.run_id == queued.run_id for choice in handoff.values())
    table = completed.artifact_manifest.by_type("prediction_metrics")[0]
    assert "ensemble_spread_log10_ic50_uM" in (queued.run_dir / table.path).read_text()


def test_worker_aggregates_independent_nesso_runs_in_log_and_micromolar_units(
    tmp_path: Path,
) -> None:
    target, target_ref, compounds, compound_ref = _typed_refolding_inputs(tmp_path)
    checkpoint, ccd, esm_cache = _nesso_references(tmp_path)
    runs_dir = tmp_path / "runs"
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> _ImmediateProcess:
        calls.append(command)
        replicate = len(calls)
        candidate = (
            queued.run_dir / "output" / f"replicate_{replicate:03d}"
            / "predictions" / "ethanol"
        )
        candidate.mkdir(parents=True)
        (candidate / "affinity.json").write_text(
            json.dumps(
                {
                    "affinity_pred_value": float(replicate - 1),
                    "affinity_probability_binary": 0.8 + 0.1 * replicate,
                    "entropy_crop_pl": 1.0,
                }
            )
        )
        kwargs["stdout"].write(f"Nesso replicate {replicate}\n")
        return _ImmediateProcess()

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(runs_dir)}, clear=False):
        queued = queue_nesso_affinity_job(
            target_path=target, target_artifact=target_ref,
            compound_paths=[compounds], compound_artifacts=[compound_ref],
            checkpoint_dir=checkpoint, ccd_path=ccd, esm_cache_dir=esm_cache,
            gpu_device="1", replicates=2, seed=42,
        )
        # A campaign-level aggregate must not override the one ligand actually
        # staged for this target-associated Nesso job.
        stale_metadata = json.loads((queued.run_dir / "metadata.json").read_text())
        stale_metadata["compound_count"] = 10
        (queued.run_dir / "metadata.json").write_text(json.dumps(stale_metadata))
        assert len(queued.metadata["queued_commands"]) == 2
        result = run_worker_once(
            WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(1,), heartbeat_seconds=0.05),
            popen=fake_popen, sleep=lambda _: None,
        )

    assert result is not None and result["status"] == "completed"
    assert [command[command.index("--seed") + 1] for command in calls] == ["42", "43"]
    completed = JobRecord.load(queued.run_dir, task_group="refolding")
    assert completed.metadata["compound_count"] == 1
    assert completed.result["compound_count"] == 1
    assert completed.result["structure_output"] is False
    summary_ref = next(
        item for item in completed.artifact_manifest.by_type("prediction_metrics")
        if item.role == "replicate_summary"
    )
    row = next(csv.DictReader((queued.run_dir / summary_ref.path).open()))
    assert float(row["mean_affinity_log10_ic50_uM"]) == pytest.approx(0.5)
    assert float(row["geometric_mean_ic50_uM"]) == pytest.approx(10 ** 0.5)
    assert float(row["arithmetic_mean_ic50_uM"]) == pytest.approx(5.5)
    assert float(row["sample_sd_ic50_uM"]) == pytest.approx(9 / (2 ** 0.5))
    assert not completed.artifact_manifest.by_type("predicted_complex")


def test_nesso_exit_zero_without_affinity_is_failed(tmp_path: Path) -> None:
    target, target_ref, compounds, compound_ref = _typed_refolding_inputs(tmp_path)
    checkpoint, ccd, esm_cache = _nesso_references(tmp_path)
    runs_dir = tmp_path / "runs"

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(runs_dir)}, clear=False):
        queued = queue_nesso_affinity_job(
            target_path=target, target_artifact=target_ref,
            compound_paths=[compounds], compound_artifacts=[compound_ref],
            checkpoint_dir=checkpoint, ccd_path=ccd, esm_cache_dir=esm_cache,
        )
        result = run_worker_once(
            WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(0,), heartbeat_seconds=0.05),
            popen=lambda *_args, **_kwargs: _ImmediateProcess(), sleep=lambda _: None,
        )

    assert result is not None and result["status"] == "failed"
    failed = JobRecord.load(queued.run_dir, task_group="refolding")
    assert failed.result["success"] is False
    assert "no readable affinity.json" in failed.result["error"]
    assert failed.artifact_manifest is not None and failed.artifact_manifest.artifacts == ()
