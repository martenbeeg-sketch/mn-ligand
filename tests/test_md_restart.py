from __future__ import annotations

from types import SimpleNamespace

from mn_ligand.ligandx.services.md.service import MDOptimizationService
from mn_ligand.ligandx.services.md.workflow.equilibration_runner import (
    EquilibrationRunner,
    _restart_platform_properties,
)


class _Quantity:
    def __init__(self, value: float):
        self.value = value

    def value_in_unit(self, _unit) -> float:
        return self.value


class _State:
    def __init__(self, energy: float, volume: float):
        self.energy = energy
        self.volume = volume

    def getPotentialEnergy(self) -> _Quantity:
        return _Quantity(self.energy)

    def getPeriodicBoxVolume(self) -> _Quantity:
        return _Quantity(self.volume)


class _Context:
    def __init__(self):
        self.parameters = {"k_prot": 1000.0, "k_lig": 2500.0, "k_plan": 100.0}
        self.velocity_request = None
        self.state_calls = 0

    def getParameters(self):
        return self.parameters

    def setParameter(self, name: str, value: float) -> None:
        self.parameters[name] = value

    def getState(self, **_kwargs) -> _State:
        self.state_calls += 1
        return _State(-1000.0 + self.state_calls, 500.0 + self.state_calls)

    def setVelocitiesToTemperature(self, temperature: float, seed: int) -> None:
        self.velocity_request = (temperature, seed)


class _Simulation:
    def __init__(self):
        self.context = _Context()
        self.steps: list[int] = []
        self.minimized = False

    def step(self, steps: int) -> None:
        self.steps.append(steps)

    def minimizeEnergy(self, **_kwargs) -> None:
        self.minimized = True


class _Unit:
    kelvin = 1.0
    kilojoule_per_mole = 1.0
    nanometer = 1.0


class _Platform:
    def __init__(self, name: str, precision: str = "mixed"):
        self.name = name
        self.precision = precision

    def getName(self) -> str:
        return self.name

    def getPropertyValue(self, _context, name: str) -> str:
        assert name == "Precision"
        return self.precision


class _RestartContext:
    def __init__(self, platform: _Platform):
        self.platform = platform

    def getPlatform(self) -> _Platform:
        return self.platform


class _RestartSimulation:
    def __init__(self, platform: _Platform):
        self.context = _RestartContext(platform)


def test_restart_context_preserves_gpu_precision() -> None:
    cuda = _RestartSimulation(_Platform("CUDA", "mixed"))
    cpu = _RestartSimulation(_Platform("CPU"))

    assert _restart_platform_properties(cuda) == {"Precision": "mixed"}
    assert _restart_platform_properties(cpu) == {}


def test_combined_result_preserves_restart_audit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "mn_ligand.ligandx.services.md.service.EquilibrationAnalytics.compute",
        lambda *_args, **_kwargs: {},
    )
    service = MDOptimizationService.__new__(MDOptimizationService)
    service.output_dir = str(tmp_path)
    config = SimpleNamespace(
        system_id="system",
        protein_id="protein",
        ligand_id="LIG",
        is_protein_only=False,
        nvt_steps=0,
        npt_steps=0,
        production_steps=100,
        production_report_interval=10,
    )
    restart_audit = {
        "resume_mode": "coordinate_continuation",
        "replica_initialization": {"seed": 12345, "burn_in_steps": 1000},
    }

    result = service._combine_results(
        config,
        "prepared.pdb",
        {"total_atoms": 2, "system_info": {}},
        {
            "equilibration_stats": {},
            "output_files": {},
            "restart_resume": restart_audit,
            "restraint_protocol": {"production_unrestrained": True},
        },
    )

    assert result["restart_resume"] == restart_audit
    assert result["restraint_protocol"]["production_unrestrained"] is True


def test_independent_replica_is_seeded_unrestrained_and_not_minimized(tmp_path) -> None:
    simulation = _Simulation()
    runner = EquilibrationRunner(str(tmp_path))

    report = runner._initialize_independent_replica(
        simulation,
        _Unit(),
        temperature=300.0,
        burn_in_steps=250000,
        seed=12345,
    )

    assert simulation.minimized is False
    assert simulation.steps == [250000]
    assert simulation.context.velocity_request == (300.0, 12345)
    assert all(value == 0.0 for value in simulation.context.parameters.values())
    assert report["burn_in_included_in_production"] is False
    assert report["burn_in_duration_ps"] == 1000.0


def test_independent_replica_extends_until_density_plateau(
    monkeypatch,
    tmp_path,
) -> None:
    simulation = _Simulation()
    runner = EquilibrationRunner(str(tmp_path))
    monkeypatch.setattr(
        runner,
        "_system_density_g_ml",
        lambda *_args: 1.03,
    )
    monkeypatch.setattr(
        "mn_ligand.ligandx.services.md.workflow.equilibration_runner.fit_density_plateau",
        lambda times, _densities: {
            "plateau": len(times) >= 4,
            "sample_count": len(times),
        },
    )

    report = runner._initialize_independent_replica(
        simulation,
        _Unit(),
        temperature=300.0,
        burn_in_steps=10,
        seed=12345,
        density_revalidation=True,
        revalidation_max_steps=30,
        revalidation_increment_steps=10,
        density_sample_interval_steps=5,
        timestep_fs=4.0,
    )

    assert simulation.steps == [5, 5, 5, 5]
    assert report["policy"] == "roe_density_revalidated_replica"
    assert report["burn_in_steps"] == 20
    assert report["density_fit"]["plateau"] is True
    assert report["burn_in_included_in_production"] is False
    assert (tmp_path / "replica_density_revalidation.csv").is_file()
    assert (tmp_path / "replica_density_revalidation.json").is_file()


def test_independent_replica_revalidation_fails_without_plateau(
    monkeypatch,
    tmp_path,
) -> None:
    simulation = _Simulation()
    runner = EquilibrationRunner(str(tmp_path))
    monkeypatch.setattr(
        runner,
        "_system_density_g_ml",
        lambda *_args: 1.03,
    )
    monkeypatch.setattr(
        "mn_ligand.ligandx.services.md.workflow.equilibration_runner.fit_density_plateau",
        lambda times, _densities: {
            "plateau": False,
            "sample_count": len(times),
        },
    )

    try:
        runner._initialize_independent_replica(
            simulation,
            _Unit(),
            temperature=300.0,
            burn_in_steps=10,
            seed=12345,
            density_revalidation=True,
            revalidation_max_steps=20,
            revalidation_increment_steps=10,
            density_sample_interval_steps=5,
            timestep_fs=4.0,
        )
    except RuntimeError as exc:
        assert "density plateau criteria were not satisfied" in str(exc)
        assert (tmp_path / "replica_density_revalidation.csv").is_file()
        assert (tmp_path / "replica_density_revalidation.json").is_file()
    else:
        raise AssertionError("Expected replica density revalidation to fail")
