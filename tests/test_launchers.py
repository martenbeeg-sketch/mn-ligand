from __future__ import annotations

import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mn_ligand.cli import app
from mn_ligand.launchers import install_user_launchers


def test_install_user_launchers_uses_current_environment_cli(
    tmp_path: Path, monkeypatch
) -> None:
    environment_cli = tmp_path / "environment" / "bin" / "mn-ligand"
    environment_cli.parent.mkdir(parents=True)
    environment_cli.write_text("#!/bin/sh\n")
    environment_cli.chmod(0o755)
    monkeypatch.setenv("MN_LIGAND_CLI", str(environment_cli))
    bin_dir = tmp_path / "bin"
    desktop_dir = tmp_path / "Desktop"
    worker_installs: list[tuple[tuple[int, ...], bool, bool]] = []

    def record_worker_install(gpu_ids, *, start=True, enable=True, **_kwargs):
        worker_installs.append((tuple(gpu_ids), start, enable))
        return tmp_path / "mn-ligand-worker@.service"

    monkeypatch.setattr(
        "mn_ligand.core.worker_service.install_worker_service",
        record_worker_install,
    )

    result = install_user_launchers(
        bin_dir=bin_dir,
        desktop_dir=desktop_dir,
        worker_gpu_ids=(0,),
    )

    assert Path(result["cli_wrapper"]).read_text().endswith(
        f'exec {environment_cli} "$@"\n'
    )
    app_wrapper = Path(result["app_wrapper"]).read_text()
    assert app_wrapper.endswith(
        f'exec {environment_cli} app "$@"\n'
    )
    assert f"{environment_cli} worker-service start --gpu-ids 0" in app_wrapper
    assert subprocess.run(
        ["sh", "-n", result["app_wrapper"]], check=False
    ).returncode == 0
    assert worker_installs == [((0,), False, False)]
    assert result["worker_services_installed"] is True
    desktop = Path(result["desktop_launcher"]).read_text()
    assert f"Exec={bin_dir / 'mn-ligand-app'}" in desktop
    assert "Terminal=true" in desktop


def test_install_user_launchers_can_skip_worker_services(
    tmp_path: Path, monkeypatch
) -> None:
    environment_cli = tmp_path / "environment" / "bin" / "mn-ligand"
    environment_cli.parent.mkdir(parents=True)
    environment_cli.write_text("#!/bin/sh\n")
    environment_cli.chmod(0o755)
    monkeypatch.setenv("MN_LIGAND_CLI", str(environment_cli))

    result = install_user_launchers(
        bin_dir=tmp_path / "bin",
        create_desktop=False,
        install_workers=False,
    )

    assert "worker-service start" not in Path(result["app_wrapper"]).read_text()
    assert result["worker_services_installed"] is False


def test_install_launchers_cli_can_skip_desktop(tmp_path: Path, monkeypatch) -> None:
    environment_cli = tmp_path / "environment" / "bin" / "mn-ligand"
    environment_cli.parent.mkdir(parents=True)
    environment_cli.write_text("#!/bin/sh\n")
    environment_cli.chmod(0o755)
    monkeypatch.setenv("MN_LIGAND_CLI", str(environment_cli))

    result = CliRunner().invoke(
        app,
        [
            "install-launchers",
            "--no-desktop",
            "--no-workers",
            "--bin-dir",
            str(tmp_path / "bin"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Start the app with: mn-ligand-app" in result.output
    assert not (tmp_path / "Desktop").exists()
