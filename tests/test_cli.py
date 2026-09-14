from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from mn_ligand.cli import app
from mn_ligand.core.diagnostics import DiagnosticResult
from mn_ligand.core.worker_service import ServiceCommandResult


def test_init_creates_portable_runtime_layout() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as temp_dir:
        base = Path(temp_dir)
        home = base / "portable-home"
        tmpdir = base / "portable-tmp"
        result = runner.invoke(
            app,
            ["init", "--app-home", str(home), "--tmpdir", str(tmpdir)],
        )

        assert result.exit_code == 0, result.output
        assert str(home.resolve()) in result.output
        assert (home / "workdir" / "runs").is_dir()
        assert (home / "reference_files").is_dir()
        assert (home / "libraries").is_dir()
        assert tmpdir.is_dir()


def test_doctor_emits_json_and_success_exit_code() -> None:
    runner = CliRunner()
    diagnostics = [DiagnosticResult("registry", "Tool registry", "pass", "Loaded")]

    with patch("mn_ligand.core.diagnostics.run_diagnostics", return_value=diagnostics):
        result = runner.invoke(app, ["doctor", "--skip-images", "--json"])

    assert result.exit_code == 0, result.output
    assert '"check_id": "registry"' in result.output


def test_doctor_fails_when_required_check_fails() -> None:
    runner = CliRunner()
    diagnostics = [DiagnosticResult("docker", "Docker Engine", "fail", "Unavailable")]

    with patch("mn_ligand.core.diagnostics.run_diagnostics", return_value=diagnostics):
        result = runner.invoke(app, ["doctor", "--skip-images"])

    assert result.exit_code == 1
    assert "FAIL  Docker Engine: Unavailable" in result.output


def test_worker_once_reports_idle() -> None:
    runner = CliRunner()

    with patch("mn_ligand.core.worker.run_worker_once", return_value=None):
        result = runner.invoke(app, ["worker", "--once", "--gpu-ids", "1"])

    assert result.exit_code == 0, result.output
    assert "GPUs: [1]" in result.output
    assert '"status": "idle"' in result.output


def test_worker_accepts_stable_service_identity() -> None:
    runner = CliRunner()

    with patch("mn_ligand.core.worker.run_worker_once", return_value=None), patch(
        "mn_ligand.core.worker.WorkerConfig.create"
    ) as create:
        from mn_ligand.core.worker import WorkerConfig

        create.return_value = WorkerConfig(
            runs_dir=Path("/tmp/runs"), worker_id="mn-ligand-gpu-0", gpu_ids=(0,)
        )
        result = runner.invoke(
            app,
            ["worker", "--once", "--gpu-ids", "0", "--worker-id", "mn-ligand-gpu-0"],
        )

    assert result.exit_code == 0, result.output
    create.assert_called_once_with(
        gpu_ids=(0,), worker_id="mn-ligand-gpu-0", job_class="mixed"
    )


def test_worker_service_install_uses_selected_gpu_instances() -> None:
    runner = CliRunner()

    with patch("mn_ligand.core.worker_service.install_worker_service") as install:
        install.return_value = Path("/tmp/mn-ligand-worker@.service")
        result = runner.invoke(app, ["worker-service", "install", "--gpu-ids", "1"])

    assert result.exit_code == 0, result.output
    install.assert_called_once_with((1,), start=True)
    assert "Enabled CPU worker and GPU workers: [1] (started)" in result.output


def test_worker_service_status_propagates_systemd_status() -> None:
    runner = CliRunner()
    command_result = ServiceCommandResult(
        command=("systemctl",), returncode=0, stdout="active\n", stderr=""
    )

    with patch(
        "mn_ligand.core.worker_service.manage_worker_services", return_value=command_result
    ) as manage:
        result = runner.invoke(app, ["worker-service", "status", "--gpu-ids", "0,1"])

    assert result.exit_code == 0, result.output
    manage.assert_called_once_with("status", (0, 1), check=False)
    assert "active" in result.output
