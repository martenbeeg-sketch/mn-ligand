from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JobRecord
from mn_ligand.workflows.generative_design import (
    GENERATOR_SPECS,
    create_generation_campaign_job,
    finalize_generation_job,
    queue_generation_job,
)


def _source_artifact(
    runs_dir: Path,
    *,
    run_id: str,
    filename: str,
    content: str,
    artifact_type: str,
    role: str,
    extra_artifacts: tuple[tuple[str, str, str, str], ...] = (),
) -> ArtifactRef:
    run_dir = runs_dir / "fixture-source" / run_id
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    path = artifact_dir / filename
    path.write_text(content)
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "created_at": "2026-07-27T00:00:00+00:00",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    artifacts = [
        ArtifactRef.from_path(
            run_dir, path, artifact_type, role=role
        )
    ]
    for extra_name, extra_content, extra_type, extra_role in extra_artifacts:
        extra_path = artifact_dir / extra_name
        extra_path.write_text(extra_content)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir, extra_path, extra_type, role=extra_role
            )
        )
    write_artifact_manifest(run_dir, artifacts)
    return artifacts[0]


def _generation_inputs(
    tmp_path: Path, monkeypatch
) -> tuple[Path, Path, ArtifactRef, ArtifactRef, ArtifactRef, ArtifactRef]:
    runs_dir = tmp_path / "runs"
    references = tmp_path / "references"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    monkeypatch.setenv("MN_LIGAND_REFERENCE_DIR", str(references))
    for spec in GENERATOR_SPECS:
        for relative in spec.reference_paths:
            path = references / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
    target = _source_artifact(
        runs_dir,
        run_id="target-1",
        filename="target.pdb",
        content="ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n",
        artifact_type="prepared_target",
        role="prepared_target",
    )
    pocket = _source_artifact(
        runs_dir,
        run_id="pocket-1",
        filename="pocket.pdb",
        content="ATOM      1  CA  ALA A   1       1.000   2.000   3.000\nEND\n",
        artifact_type="pocket",
        role="selected_pocket",
    )
    reference = _source_artifact(
        runs_dir,
        run_id="reference-1",
        filename="reference.sdf",
        content="\n  RDKit\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n",
        artifact_type="compound_set",
        role="reference_ligand",
    )
    hypothesis = _source_artifact(
        runs_dir,
        run_id="hypothesis-1",
        filename="hypothesis.json",
        content=json.dumps(
            {
                "kind": "pharmacophore_hypothesis",
                "features": [],
                "required_target_contacts": [],
            }
        ),
        artifact_type="pharmacophore_hypothesis",
        role="canonical_hypothesis",
        extra_artifacts=(
            (
                "pharmit.json",
                json.dumps({"points": []}),
                "pharmacophore_exchange",
                "pharmit_json",
            ),
            (
                "hypothesis.posp",
                "0\n",
                "pharmacophore_exchange",
                "pgmg_posp",
            ),
        ),
    )
    return runs_dir, references, target, pocket, reference, hypothesis


def test_campaign_queues_engine_specific_native_controls(
    tmp_path: Path, monkeypatch
) -> None:
    (
        _runs_dir,
        _references,
        target,
        pocket,
        reference,
        hypothesis,
    ) = _generation_inputs(tmp_path, monkeypatch)
    settings = {
        "omtra": {
            "integration_steps": 400,
            "stochastic_sampling": True,
            "noise_scale": 1.2,
            "epsilon": 0.02,
            "ligand_atoms_mean": 32,
            "ligand_atoms_std": 3,
        },
        "pocketflow": {
            "atom_temperature": 0.8,
            "bond_temperature": 1.1,
            "max_atoms": 45,
            "focus_strategy": "sample",
            "focus_threshold": 0.6,
            "min_protein_distance": 2.8,
        },
        "pocketxmol": {
            "diffusion_steps": 150,
            "ligand_atoms_mean": 34,
            "ligand_atoms_std": 4,
            "optimization_strength": 0.35,
        },
        "flowr_root": {
            "integration_steps": 160,
            "corrector_steps": 2,
            "solver": "midpoint",
            "use_sde_simulation": True,
            "sample_molecule_sizes": True,
            "filter_diversity": True,
            "diversity_threshold": 0.75,
        },
        "conditar": {"diffusion_steps": 1200, "pocket_radius": 12},
        "paopt": {
            "diffusion_steps": 900,
            "pocket_radius": 11,
            "optimize_properties": ["HIA", "hERG"],
            "minimize_properties": ["hERG"],
            "optimization_steps": 2,
            "gradient_estimate_pairs": 3,
            "perturbation_size": 0.025,
        },
        "drugrpg": {"max_atoms": 36},
        "pfm": {},
        "pgmg": {},
    }
    for value in settings.values():
        value.update(
            {
                "requested_count": 7,
                "batch_size": 3,
                "seed": 101,
                "max_runtime_seconds": 600,
            }
        )
    campaign = create_generation_campaign_job(
        name="Parameter handshake",
        engine_ids=settings,
        target_artifact=target,
        pocket_artifact=pocket,
        reference_artifact=reference,
        pharmacophore_artifact=hypothesis,
        engine_modes={
            key: (
                "Optimize while retaining similarity"
                if key == "pocketxmol"
                else "Use pocket geometry only"
            )
            for key in settings
        },
        objectives={"admet": True, "diversity": True},
        requested_count=7,
        batch_size=3,
        seed=101,
        engine_settings=settings,
    )

    jobs = {}
    for spec in GENERATOR_SPECS:
        jobs[spec.engine_id] = queue_generation_job(
            campaign,
            engine_id=spec.engine_id,
            target_artifact=target,
            pocket_artifact=pocket,
            reference_artifact=reference,
            pharmacophore_artifact=hypothesis,
            engine_mode=(
                "Optimize while retaining similarity"
                if spec.engine_id == "pocketxmol"
                else "Use pocket geometry only"
            ),
            objectives={"admet": True, "diversity": True},
            requested_count=7,
            batch_size=3,
            seed=101,
            engine_settings=settings[spec.engine_id],
        )

    assert set(jobs) == {spec.engine_id for spec in GENERATOR_SPECS}
    for engine_id, job in jobs.items():
        assert job.metadata["max_runtime_seconds"] == 600
        assert job.metadata["requested_count"] == 7
        assert job.metadata["batch_size"] == 3
        assert job.metadata["seed"] == 101
        payload = json.loads((job.run_dir / "input.json").read_text())
        assert payload["engine_settings"] == settings[engine_id]
        command = job.metadata["queued_command"]
        assert "--count" in command and command[command.index("--count") + 1] == "7"

    command = jobs["omtra"].metadata["queued_command"]
    assert command[command.index("--integration-steps") + 1] == "400"
    assert "--stochastic-sampling" in command
    command = jobs["pocketflow"].metadata["queued_command"]
    assert command[command.index("--atom-temperature") + 1] == "0.8"
    assert command[command.index("--focus-strategy") + 1] == "sample"
    command = jobs["pocketxmol"].metadata["queued_command"]
    assert command[command.index("--optimization-strength") + 1] == "0.35"
    command = jobs["flowr_root"].metadata["queued_command"]
    assert command[command.index("--solver") + 1] == "midpoint"
    assert "--use-sde-simulation" in command
    assert "--filter-diversity" in command
    command = jobs["conditar"].metadata["queued_command"]
    assert command[command.index("--diffusion-steps") + 1] == "1200"
    command = jobs["paopt"].metadata["queued_command"]
    assert command[command.index("--optimization-steps") + 1] == "2"
    assert command[command.index("--gradient-estimate-pairs") + 1] == "3"
    assert command[command.index("--perturbation-size") + 1] == "0.025"
    assert command[command.index("--optimize-properties") + 1] == "HIA"
    command = jobs["drugrpg"].metadata["queued_command"]
    assert command[command.index("--max-atoms") + 1] == "36"


def test_local_redesign_atom_mask_reaches_native_engine_commands(
    tmp_path: Path, monkeypatch
) -> None:
    (
        _runs_dir,
        _references,
        target,
        pocket,
        reference,
        hypothesis,
    ) = _generation_inputs(tmp_path, monkeypatch)
    shared_mask = {
        "redesign_mode": "replace_selected_atoms",
        "preserve_atom_indices": [0, 1, 2, 3],
        "redesign_atom_indices": [4, 5],
        "anchor_atom_indices": [3],
        "requested_count": 8,
        "batch_size": 4,
        "seed": 17,
    }
    settings = {
        "pocketxmol": {
            **shared_mask,
            "diffusion_steps": 50,
            "optimization_strength": 0.4,
        },
        "flowr_root": {
            **shared_mask,
            "integration_steps": 100,
            "filter_conditioned_substructure": True,
        },
    }
    modes = {
        engine_id: "Partial ligand redesign / atom-mask inpainting"
        for engine_id in settings
    }
    campaign = create_generation_campaign_job(
        name="Local redesign",
        engine_ids=settings,
        target_artifact=target,
        pocket_artifact=pocket,
        reference_artifact=reference,
        pharmacophore_artifact=hypothesis,
        engine_modes=modes,
        objectives={"local_redesign": True},
        requested_count=8,
        batch_size=4,
        seed=17,
        engine_settings=settings,
    )

    jobs = {
        engine_id: queue_generation_job(
            campaign,
            engine_id=engine_id,
            target_artifact=target,
            pocket_artifact=pocket,
            reference_artifact=reference,
            pharmacophore_artifact=hypothesis,
            engine_mode=modes[engine_id],
            objectives={"local_redesign": True},
            requested_count=8,
            batch_size=4,
            seed=17,
            engine_settings=engine_settings,
        )
        for engine_id, engine_settings in settings.items()
    }

    for job in jobs.values():
        command = job.metadata["queued_command"]
        assert command[command.index("--redesign-mode") + 1] == (
            "replace_selected_atoms"
        )
        assert [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--preserve-atom-index"
        ] == ["0", "1", "2", "3"]
        assert [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--redesign-atom-index"
        ] == ["4", "5"]
        assert command[command.index("--anchor-atom-index") + 1] == "3"
    assert "--filter-conditioned-substructure" in (
        jobs["flowr_root"].metadata["queued_command"]
    )


def test_task_specific_ligand_modes_reach_supported_native_commands(
    tmp_path: Path, monkeypatch
) -> None:
    (
        _runs_dir,
        _references,
        target,
        pocket,
        reference,
        _hypothesis,
    ) = _generation_inputs(tmp_path, monkeypatch)
    cases = (
        (
            "pocketxmol",
            "fragment_growing",
            {"preserve_atom_indices": [0, 1], "grow_size": 7},
            ("--grow-size", "7"),
        ),
        (
            "flowr_root",
            "fragment_growing",
            {"preserve_atom_indices": [0, 1], "grow_size": 7},
            ("--grow-size", "7"),
        ),
        (
            "flowr_root",
            "scaffold_hopping",
            {},
            ("--redesign-mode", "scaffold_hopping"),
        ),
        (
            "pocketxmol",
            "partial_optimization",
            {},
            ("--redesign-mode", "partial_optimization"),
        ),
    )
    for engine_id, mode, extra, expected in cases:
        settings = {
            "redesign_mode": mode,
            "requested_count": 3,
            "batch_size": 2,
            "seed": 9,
            **extra,
        }
        campaign = create_generation_campaign_job(
            name=mode,
            engine_ids=(engine_id,),
            target_artifact=target,
            pocket_artifact=pocket,
            reference_artifact=reference,
            pharmacophore_artifact=None,
            engine_modes={engine_id: mode},
            objectives={mode: True},
            requested_count=3,
            batch_size=2,
            seed=9,
            engine_settings={engine_id: settings},
        )
        job = queue_generation_job(
            campaign,
            engine_id=engine_id,
            target_artifact=target,
            pocket_artifact=pocket,
            reference_artifact=reference,
            pharmacophore_artifact=None,
            engine_mode=mode,
            objectives={mode: True},
            requested_count=3,
            batch_size=2,
            seed=9,
            engine_settings=settings,
        )
        command = job.metadata["queued_command"]
        assert expected[0] in command
        assert command[command.index(expected[0]) + 1] == expected[1]


def test_failed_generation_recovers_complete_native_molecules(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))
    run_dir = tmp_path / "runs" / "molecule-generation" / "partial-1"
    native_dir = run_dir / "native"
    normalized_dir = run_dir / "normalized"
    native_dir.mkdir(parents=True)
    normalized_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "status": "running",
                "engine_id": "conditar",
                "requested_count": 4,
            }
        )
    )
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(molecule, randomSeed=7)
    writer = Chem.SDWriter(str(native_dir / "partial.sdf"))
    writer.write(molecule)
    writer.close()
    (run_dir / "stderr.log").write_text("runtime budget exceeded\n")

    job = finalize_generation_job(run_dir, returncode=-15)

    assert job.status == "failed"
    result = json.loads((run_dir / "result.json").read_text())
    assert result["success"] is False
    assert result["valid_compound_count"] == 1
    assert result["partial_output_recovery_attempted"] is True
    assert result["partial_output_recovery_succeeded"] is True
    assert (normalized_dir / "generated_compounds.sdf").stat().st_size > 0
    assert job.artifact_manifest is not None
    assert job.artifact_manifest.by_type("generated_molecule_set")
