from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from mn_ligand.app.pages.md_simulation import (
    PRODUCTION_PRESETS,
    _production_for_engine,
)
from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JobRecord
from mn_ligand.workflows.gromacs_protocol import (
    atom_groups,
    inject_restraint_include,
    parse_xvg_series,
    render_gromacs_mdp,
    write_gromacs_index,
    write_molecule_type_restraints,
    write_roe_position_restraints,
    write_roe_brooks_mdp_set,
)
from mn_ligand.workflows.gromacs_md import _extract_density
from mn_ligand.workflows.gromacs_md import _stage_gpu_mdrun_flags
from mn_ligand.workflows.gromacs_md import _trajectory_window_ns
from mn_ligand.workflows.gromacs_md import parse_gromacs_performance
from mn_ligand.workflows.gromacs_md import run_g_mmpbsa
from mn_ligand.workflows.md_engines import (
    GROMACS_ENGINE,
    OPENMM_ENGINE,
    md_engine_spec,
    roe_brooks_stage_specification,
)
from mn_ligand.workflows.md_simulation import (
    INDEPENDENT_REPLICA,
    advance_md_workflow,
    create_md_simulation,
    create_mmgbsa_analysis_job,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _source_job(runs_dir: Path) -> tuple[JobRecord, ArtifactRef]:
    run_dir = runs_dir / "structure-jobs" / "structure-gromacs"
    run_dir.mkdir(parents=True)
    complex_path = run_dir / "complex.pdb"
    complex_path.write_text(
        "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C\n"
        "HETATM    2  C1  LIG A 501      12.000  10.000  10.000  1.00 20.00           C\n"
        "END\n"
    )
    _write_json(
        run_dir / "metadata.json",
        {
            "schema_version": 1,
            "run_id": run_dir.name,
            "status": "completed",
            "pdb_id": "TEST",
        },
    )
    artifact = ArtifactRef.from_path(
        run_dir,
        complex_path,
        "prepared_complex",
        role="complex",
    )
    write_artifact_manifest(run_dir, [artifact])
    return JobRecord.load(run_dir, task_group="structure-jobs"), artifact


def _prep_input(source_path: Path) -> dict:
    return {
        "pdb_id": "TEST",
        "ligand_key": "LIG|A|501|_",
        "prepared_complex_path": str(source_path),
        "forcefield_method": "gaff2",
        "charge_method": "am1bcc",
        "box_shape": "dodecahedron",
        "padding_nm": 1.0,
        "ionic_strength": 0.15,
        "constraints": "HBonds",
        "mmgbsa_backend": "ambertools_mmpbsa",
        "temperature": 300.0,
        "pressure": 1.0,
        "preparation_protocol": "roe_brooks_2020",
        "density_stabilization_min_ns": 1.0,
        "density_stabilization_max_ns": 5.0,
        "density_stabilization_increment_ns": 1.0,
        "density_sample_interval_ps": 4.0,
        "density_plateau_required": True,
    }


def test_engine_specs_and_roe_stage_contract_are_explicit() -> None:
    assert md_engine_spec(OPENMM_ENGINE).trajectory_format == "dcd"
    assert md_engine_spec(GROMACS_ENGINE).trajectory_format == "xtc"
    assert md_engine_spec(GROMACS_ENGINE).production_timestep_fs == 4.0
    stages = roe_brooks_stage_specification()
    assert [stage["stage"] for stage in stages] == [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
    ]
    assert sum(
        int(stage.get("steps") or 0)
        for stage in stages[:9]
        if stage["kind"] == "minimization"
    ) == 4000
    assert sum(
        int(stage.get("steps") or 0)
        for stage in stages[:9]
        if stage["kind"] != "minimization"
    ) == 40000
    assert stages[0]["precision"] == "double"
    assert stages[1]["duration_ps"] == 15.0
    assert stages[7]["restraint_selection"] == (
        "polymer_backbone_and_ligand_heavy"
    )
    assert stages[8]["timestep_fs"] == 2.0


def test_production_presets_are_task_oriented() -> None:
    assert list(PRODUCTION_PRESETS) == [
        "Smoke",
        "Ligand MM/GBSA",
        "Stability",
        "Manual",
    ]
    assert PRODUCTION_PRESETS["Smoke"]["production_ns"] == 0.2
    assert PRODUCTION_PRESETS["Ligand MM/GBSA"]["production_ns"] == 50.0
    assert PRODUCTION_PRESETS["Ligand MM/GBSA"]["replicas"] == 3
    assert PRODUCTION_PRESETS["Ligand MM/GBSA"]["burn_in_ns"] == 0.0
    assert PRODUCTION_PRESETS["Ligand MM/GBSA"]["revalidation_max_ns"] == 5.0
    assert PRODUCTION_PRESETS["Ligand MM/GBSA"]["analysis_enabled"] is True
    assert PRODUCTION_PRESETS["Ligand MM/GBSA"]["endpoint_enabled"] is True
    assert PRODUCTION_PRESETS["Stability"]["production_ns"] == 100.0
    assert PRODUCTION_PRESETS["Stability"]["replicas"] == 3
    assert PRODUCTION_PRESETS["Stability"]["burn_in_ns"] == 0.0
    assert PRODUCTION_PRESETS["Stability"]["revalidation_max_ns"] == 5.0
    assert PRODUCTION_PRESETS["Stability"]["analysis_enabled"] is True
    assert PRODUCTION_PRESETS["Stability"]["endpoint_enabled"] is False
    assert PRODUCTION_PRESETS["Manual"]["burn_in_ns"] == 0.0
    assert PRODUCTION_PRESETS["Manual"]["revalidation_max_ns"] == 5.0


def test_parse_gromacs_performance_uses_final_production_value(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "production.log"
    log_path.write_text(
        "Performance:      800.125        0.030\n"
        "Performance:      818.467        0.029\n"
    )

    assert parse_gromacs_performance(log_path) == {
        "ns_per_day": 818.467,
        "source": "GROMACS production.log",
        "sample_count": 2,
    }


def test_gromacs_mdp_translation_preserves_roe_dynamics() -> None:
    stages = roe_brooks_stage_specification()
    step1 = render_gromacs_mdp(dict(stages[0]))
    step2 = render_gromacs_mdp(dict(stages[1]), random_seed=123)
    step7 = render_gromacs_mdp(dict(stages[6]))
    step9 = render_gromacs_mdp(dict(stages[8]))
    step10_hmr = render_gromacs_mdp(
        {
            **dict(stages[9]),
            "timestep_fs": 4.0,
            "steps": 250_000,
            "mass_repartition_factor": 3.0,
        }
    )
    replica_revalidation = render_gromacs_mdp(
        {
            **dict(stages[9]),
            "timestep_fs": 4.0,
            "steps": 250_000,
            "mass_repartition_factor": 3.0,
            "generate_velocities": True,
        }
    )

    assert "integrator                   = steep" in step1
    assert "constraints                  = none" in step1
    assert "define                       = -DROE_STAGE_1" in step1
    assert "gen-vel                      = yes" in step2
    assert "gen-seed                     = 123" in step2
    assert "pcoupl                       = C-rescale" in step7
    assert "continuation                 = yes" in step7
    assert "refcoord-scaling             = com" in step7
    assert "pcoupl                       = Parrinello-Rahman" in step9
    assert "dt                           = 0.002000" in step9
    assert "constraints                  = h-bonds" in step9
    assert "dt                           = 0.004000" in step10_hmr
    assert "constraints                  = h-bonds" in step10_hmr
    assert "mass-repartition-factor      = 3.0" in step10_hmr
    assert "pcoupl                       = C-rescale" in replica_revalidation


def test_gromacs_native_hmr_keeps_gpu_resident_updates(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "MN_GROMACS_GPU_FLAGS",
        "-nb gpu -pme gpu -bonded gpu -update gpu",
    )

    assert _stage_gpu_mdrun_flags({"timestep_fs": 2.0}) == [
        "-nb",
        "gpu",
        "-pme",
        "gpu",
        "-bonded",
        "gpu",
        "-update",
        "gpu",
    ]
    assert _stage_gpu_mdrun_flags(
        {
            "timestep_fs": 4.0,
            "mass_repartition_factor": 3.0,
        }
    ) == [
        "-nb",
        "gpu",
        "-pme",
        "gpu",
        "-bonded",
        "gpu",
        "-update",
        "gpu",
    ]
    assert _stage_gpu_mdrun_flags({"timestep_fs": 4.0}) == [
        "-nb",
        "gpu",
        "-pme",
        "gpu",
        "-bonded",
        "gpu",
    ]


def test_gromacs_roe_set_applies_native_hmr_to_dynamic_stages(
    tmp_path: Path,
) -> None:
    stages = write_roe_brooks_mdp_set(
        tmp_path,
        temperature_k=300.0,
        pressure_bar=1.0,
        density_sample_interval_ps=4.0,
        production_timestep_fs=4.0,
        mass_repartition_factor=3.0,
    )

    assert "mass-repartition-factor" not in (
        tmp_path / "roe_1.mdp"
    ).read_text()
    for stage in stages:
        mdp = (tmp_path / str(stage["mdp"])).read_text()
        if stage["kind"] == "minimization":
            continue
        assert "constraints                  = h-bonds" in mdp
        assert "mass-repartition-factor      = 3.0" in mdp


def test_gromacs_component_groups_restraints_and_index_are_deterministic(
    tmp_path: Path,
) -> None:
    def atom(name: str, residue: str, atomic_number: int):
        return SimpleNamespace(
            name=name,
            atomic_number=atomic_number,
            residue=SimpleNamespace(name=residue),
        )

    structure = SimpleNamespace(
        atoms=[
            atom("N", "ALA", 7),
            atom("CA", "ALA", 6),
            atom("H", "ALA", 1),
            atom("C1", "LIG", 6),
            atom("H1", "LIG", 1),
            atom("O", "WAT", 8),
            atom("NA", "NA+", 11),
        ]
    )
    groups = atom_groups(structure)
    assert groups["Protein"] == [1, 2, 3]
    assert groups["Ligand"] == [4, 5]
    assert groups["Large_Molecule_Heavy"] == [1, 2, 4]
    assert groups["Roe_Backbone_Ligand_Heavy"] == [1, 2, 4]

    index = tmp_path / "index.ndx"
    restraints = tmp_path / "roe_position_restraints.itp"
    write_gromacs_index(index, groups)
    write_roe_position_restraints(restraints, groups)
    assert "[ Protein_Ligand ]" in index.read_text()
    restraint_text = restraints.read_text()
    assert "#ifdef ROE_STAGE_8" in restraint_text
    assert "209.2000" in restraint_text


def test_topology_include_and_xvg_parser_are_stable(tmp_path: Path) -> None:
    topology = tmp_path / "system.top"
    topology.write_text(
        "[ moleculetype ]\nSOLUTE 3\n"
        "[ atoms ]\n1 C 1 LIG C1 1 0 12\n"
        "[ moleculetype ]\nSOL 2\n"
        "[ system ]\nTest\n"
    )
    inject_restraint_include(topology, "roe_position_restraints.itp")
    inject_restraint_include(topology, "roe_position_restraints.itp")
    assert topology.read_text().count("roe_position_restraints.itp") == 1

    xvg = tmp_path / "density.xvg"
    xvg.write_text("@ title \"Density\"\n# comment\n0 0.98\n4 1.01\n")
    assert parse_xvg_series(xvg) == ([0.0, 4.0], [0.98, 1.01])


def test_gromacs_density_is_converted_from_kg_m3_to_g_ml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    energy = tmp_path / "density.edr"
    energy.write_text("energy")

    def fake_run(command, *, cwd, stdin="", log_path):
        del cwd, stdin, log_path
        output = Path(command[command.index("-o") + 1])
        output.write_text("0 1031.0\n4 1029.0\n")

    monkeypatch.setattr(
        "mn_ligand.workflows.gromacs_md._run",
        fake_run,
    )
    monkeypatch.setattr(
        "mn_ligand.workflows.gromacs_md._gmx_binary",
        lambda: "gmx",
    )

    times, densities = _extract_density(
        [energy],
        tmp_path,
        tmp_path / "log.txt",
    )

    assert times == [0.0, 4.0]
    assert densities == [1.031, 1.029]


def test_restraints_are_local_to_each_gromacs_molecule_type(
    tmp_path: Path,
) -> None:
    topology = tmp_path / "system.top"
    topology.write_text(
        "[ moleculetype ]\nProtein 3\n"
        "[ atoms ]\n"
        "1 N 1 ALA N 1 0 14\n"
        "2 CT 1 ALA CA 2 0 12\n"
        "3 H 1 ALA H 3 0 1\n"
        "[ moleculetype ]\nLigand 3\n"
        "[ atoms ]\n"
        "1 c3 1 LIG C1 1 0 12\n"
        "2 h1 1 LIG H1 2 0 1\n"
        "[ moleculetype ]\nSOL 2\n"
        "[ atoms ]\n"
        "1 OW 1 WAT O 1 0 16\n"
        "[ system ]\nTest\n"
    )
    generated = write_molecule_type_restraints(topology)
    assert [path.name for path in generated] == [
        "roe_posre_Protein.itp",
        "roe_posre_Ligand.itp",
    ]
    assert topology.read_text().count("roe_posre_") == 2
    protein_text = generated[0].read_text()
    ligand_text = generated[1].read_text()
    assert "#ifdef ROE_STAGE_8" in protein_text
    assert "#ifdef ROE_STAGE_8" in ligand_text
    assert "       2     1" in protein_text
    assert "       1     1" in ligand_text
    assert "       4     1" not in ligand_text


def test_endpoint_percent_window_is_converted_to_real_trajectory_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeTrajectory:
        n_frames = 6
        time = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    fake_mdtraj = SimpleNamespace(
        load=lambda trajectory, top: FakeTrajectory()
    )
    monkeypatch.setitem(sys.modules, "mdtraj", fake_mdtraj)
    trajectory = tmp_path / "trajectory.xtc"
    topology = tmp_path / "topology.pdb"
    trajectory.write_text("xtc")
    topology.write_text("pdb")
    assert _trajectory_window_ns(
        trajectory,
        topology,
        start_pct=50.0,
        end_pct=100.0,
        stride=2,
    ) == (0.03, 0.05, 0.02, 2)


def test_g_mmpbsa_builds_a_gromacs_2025_compatibility_tpr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from mn_ligand.workflows import gromacs_md

    class FakeTrajectory:
        n_frames = 2
        time = [0.0, 10.0]

    monkeypatch.setitem(
        sys.modules,
        "mdtraj",
        SimpleNamespace(load=lambda trajectory, top: FakeTrajectory()),
    )
    files = {
        "production_trajectory": tmp_path / "production.xtc",
        "production_pdb": tmp_path / "production.pdb",
        "production_tpr": tmp_path / "production.tpr",
        "production_topology": tmp_path / "system.top",
        "production_index": tmp_path / "index.ndx",
    }
    for path in files.values():
        path.write_text("artifact")
    commands: list[list[str]] = []

    def fake_which(binary: str) -> str | None:
        if binary == "g_mmpbsa":
            return "/fake/g_mmpbsa"
        if binary == "/fake/gmx-2025":
            return binary
        return None

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        stdin: str | None = None,
        log_path: Path | None = None,
    ) -> SimpleNamespace:
        commands.append(command)
        if "grompp" in command:
            Path(command[command.index("-o") + 1]).write_text("tpr")
        else:
            Path(command[command.index("-o") + 1]).write_text(
                "0 1.0\n10 3.0\n"
            )
            Path(command[command.index("-os") + 1]).write_text("summary\n")
            Path(command[command.index("-ores") + 1]).write_text("residue\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("GMX_MMPBSA_BIN", "/fake/gmx-2025")
    monkeypatch.setattr(gromacs_md.shutil, "which", fake_which)
    monkeypatch.setattr(gromacs_md, "_run", fake_run)
    result = run_g_mmpbsa(
        {
            "mmgbsa_start_pct": 0.0,
            "mmgbsa_end_pct": 100.0,
            "mmgbsa_stride": 1,
        },
        {
            "md_result": {
                "output_files": {
                    key: str(path) for key, path in files.items()
                }
            }
        },
        tmp_path / "result.json",
    )
    compatibility_tpr = tmp_path / "g_mmpbsa_compatibility.tpr"
    assert commands[0][0] == "/fake/gmx-2025"
    assert commands[0][1] == "grompp"
    assert commands[1][0] == "/fake/g_mmpbsa"
    assert commands[1][commands[1].index("-s") + 1] == str(
        compatibility_tpr
    )
    assert "-pbsa" in commands[1]
    parameter_text = (tmp_path / "g_mmpbsa.mdp").read_text()
    assert "polar = yes" in parameter_text
    assert "apolar = yes" in parameter_text
    assert "gmemceil = 4000" in parameter_text
    assert result["mmgbsa"]["metadata"][
        "endpoint_tpr_gromacs_version"
    ] == "2025.4"
    assert result["mmgbsa"]["delta"]["delta_g_bind_total_kj_mol"] == 2.0


def test_dual_engine_production_translation_preserves_physical_duration() -> None:
    production = {
        "production_length_ns": 10.0,
        "production_timestep_fs": 4.0,
        "production_steps": 2_500_000,
        "production_report_interval_ps": 10.0,
        "production_report_interval": 2500,
        "replica_equilibration_ns": 1.0,
        "replica_equilibration_steps": 250_000,
        "replica_revalidation_max_ns": 5.0,
        "replica_revalidation_max_steps": 1_250_000,
        "replica_revalidation_increment_ns": 1.0,
        "replica_revalidation_increment_steps": 250_000,
        "replica_density_sample_interval_ps": 4.0,
        "replica_density_sample_interval_steps": 1000,
        "endpoint_backend": "engine_native",
    }
    openmm = _production_for_engine(production, OPENMM_ENGINE)
    gromacs = _production_for_engine(production, GROMACS_ENGINE)
    gromacs_standard = _production_for_engine(
        production,
        GROMACS_ENGINE,
        {"production_timestep_fs": 2.0},
    )
    assert openmm["endpoint_backend"] == "openmm_gbsa"
    assert openmm["production_steps"] == 2_500_000
    assert gromacs["endpoint_backend"] == "g_mmpbsa"
    assert gromacs["production_timestep_fs"] == 4.0
    assert gromacs["production_steps"] == 2_500_000
    assert gromacs["replica_equilibration_steps"] == 250_000
    assert gromacs_standard["production_timestep_fs"] == 2.0
    assert gromacs_standard["production_steps"] == 5_000_000
    assert gromacs_standard["production_report_interval"] == 5000
    assert gromacs_standard["replica_equilibration_steps"] == 500_000
    assert gromacs_standard["replica_revalidation_max_steps"] == 2_500_000
    assert gromacs_standard["replica_revalidation_increment_steps"] == 500_000
    assert gromacs_standard["replica_density_sample_interval_steps"] == 2000


def test_gromacs_workflow_uses_native_tool_and_restart_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source_job, source_artifact = _source_job(runs_dir)
    source_path = source_artifact.resolve(source_job.run_dir, must_exist=True)
    workflow = create_md_simulation(
        source_job=source_job,
        source_artifact=source_artifact,
        prep_input=_prep_input(source_path),
        production={
            "production_steps": 1000,
            "production_timestep_fs": 2.0,
            "production_report_interval": 100,
            "continuation_mode": INDEPENDENT_REPLICA,
            "replica_equilibration_steps": 500,
            "replica_density_revalidation": True,
            "replica_revalidation_max_steps": 2500,
            "replica_revalidation_increment_steps": 500,
            "replica_density_sample_interval_steps": 20,
            "replica_density_plateau_required": True,
            "endpoint_enabled": False,
        },
        replicas=1,
        analysis_enabled=True,
        image="ovolig-gromacs-cu128:latest",
        use_gpu=False,
        engine=GROMACS_ENGINE,
    )
    prep_ref = next(
        child
        for child in workflow.children
        if child.step_id == "preparation_equilibration"
    )
    pending_production_ref = next(
        child
        for child in workflow.children
        if child.step_id == "production_replica_1"
    )
    pending_production_dir = (
        runs_dir
        / pending_production_ref.task_group
        / pending_production_ref.run_id
    )
    pending_metadata = json.loads(
        (pending_production_dir / "metadata.json").read_text()
    )
    pending_metadata["error"] = "stale activation error"
    _write_json(
        pending_production_dir / "metadata.json",
        pending_metadata,
    )
    prep_dir = runs_dir / prep_ref.task_group / prep_ref.run_id
    metadata = json.loads((prep_dir / "metadata.json").read_text())
    assert metadata["md_engine"] == GROMACS_ENGINE
    assert metadata["resources"]["gpu"] is False
    assert "mn_ligand.workflows.gromacs_md" in metadata["queued_command"]
    assert "prepare" in metadata["queued_command"]
    assert "OMP_NUM_THREADS=8" in metadata["queued_command"]
    assert json.loads((prep_dir / "command.json").read_text())["tool_id"] == (
        "gromacs_md"
    )

    prepared = {
        "system_pdb": prep_dir / "system.pdb",
        "gromacs_topology": prep_dir / "system.top",
        "gromacs_coordinates": prep_dir / "equilibrated.gro",
        "gromacs_checkpoint": prep_dir / "equilibrated.cpt",
        "gromacs_index": prep_dir / "index.ndx",
        "gromacs_tpr": prep_dir / "equilibrated.tpr",
    }
    for path in prepared.values():
        path.write_text("prepared")
    _write_json(
        prep_dir / "result.json",
        {
            "success": True,
            "engine": GROMACS_ENGINE,
            "md_result": {
                "engine": GROMACS_ENGINE,
                "preparation_protocol": {
                    "protocol": "roe_brooks_2020",
                    "density_stabilization": {
                        "fit": {"plateau": True}
                    },
                },
                "output_files": {
                    "npt_pdb": str(prepared["system_pdb"]),
                    "npt_checkpoint": str(prepared["gromacs_checkpoint"]),
                    **{
                        key: str(path)
                        for key, path in prepared.items()
                    },
                },
            },
        },
    )
    workflow = advance_md_workflow(workflow.workflow_id)
    production_ref = next(
        child
        for child in workflow.children
        if child.step_id == "production_replica_1"
    )
    production_dir = (
        runs_dir / production_ref.task_group / production_ref.run_id
    )
    production_metadata = json.loads(
        (production_dir / "metadata.json").read_text()
    )
    assert "error" not in production_metadata
    production_input = json.loads(
        (production_dir / "input.json").read_text()
    )
    assert production_metadata["md_engine"] == GROMACS_ENGINE
    assert "production" in production_metadata["queued_command"]
    assert production_input["source_topology_path"].startswith(
        "/prepared-system/"
    )
    assert production_input["source_checkpoint_path"].startswith(
        "/prepared-system/"
    )
    assert production_input["replica_density_revalidation"] is True
    assert production_input["replica_revalidation_max_steps"] == 2500
    assert production_input["replica_revalidation_increment_steps"] == 500
    assert production_input["replica_density_sample_interval_steps"] == 20
    prep_job = JobRecord.load(prep_dir, task_group="md-system-prep")
    artifact_types = {
        artifact.artifact_type
        for artifact in prep_job.artifact_manifest.artifacts
    }
    assert {"md_topology", "md_index", "md_checkpoint"}.issubset(
        artifact_types
    )


def test_gromacs_endpoint_job_uses_g_mmpbsa_and_native_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    run_dir = runs_dir / "bound-ligand-md" / "gromacs-production"
    run_dir.mkdir(parents=True)
    files = {
        "production_trajectory": run_dir / "production.xtc",
        "production_pdb": run_dir / "production.pdb",
        "production_checkpoint": run_dir / "production.cpt",
        "production_topology": run_dir / "system.top",
        "production_tpr": run_dir / "production.tpr",
        "production_index": run_dir / "index.ndx",
    }
    for path in files.values():
        path.write_text("artifact")
    _write_json(
        run_dir / "metadata.json",
        {
            "schema_version": 1,
            "run_id": run_dir.name,
            "status": "completed",
            "md_engine": GROMACS_ENGINE,
            "docker_image": "ovolig-gromacs-cu128:latest",
            "use_gpu": False,
        },
    )
    _write_json(
        run_dir / "input.json",
        {"md_engine": GROMACS_ENGINE},
    )
    _write_json(
        run_dir / "result.json",
        {
            "success": True,
            "engine": GROMACS_ENGINE,
            "md_result": {
                "engine": GROMACS_ENGINE,
                "output_files": {
                    key: str(path) for key, path in files.items()
                },
            },
        },
    )
    artifact_mapping = {
        "production_trajectory": ("md_trajectory", "production"),
        "production_pdb": ("md_final_structure", "coordinates"),
        "production_checkpoint": ("md_checkpoint", "checkpoint"),
        "production_topology": ("md_topology", "gromacs-topology"),
        "production_tpr": ("md_run_input", "gromacs-tpr"),
        "production_index": ("md_index", "gromacs-index"),
    }
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(
                run_dir,
                files[key],
                artifact_type,
                role=role,
            )
            for key, (artifact_type, role) in artifact_mapping.items()
        ],
    )
    queued = create_mmgbsa_analysis_job(
        run_dir.name,
        backend="g_mmpbsa",
        use_gpu=False,
    )
    assert queued.metadata["md_engine"] == GROMACS_ENGINE
    assert queued.metadata["resources"]["gpu"] is False
    assert json.loads((queued.run_dir / "command.json").read_text())[
        "tool_id"
    ] == "gromacs_md"
    assert "mn_ligand.workflows.gromacs_md" in queued.metadata[
        "queued_command"
    ]
    assert "endpoint" in queued.metadata["queued_command"]
    payload = json.loads((queued.run_dir / "input.json").read_text())
    assert {
        "md_topology",
        "md_run_input",
        "md_index",
    }.issubset(
        {
            artifact["artifact_type"]
            for artifact in payload["source_artifacts"]
        }
    )
