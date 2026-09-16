from __future__ import annotations

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

    result = install_user_launchers(bin_dir=bin_dir, desktop_dir=desktop_dir)

    assert Path(result["cli_wrapper"]).read_text().endswith(
        f'exec {environment_cli} "$@"\n'
    )
    assert Path(result["app_wrapper"]).read_text().endswith(
        f'exec {environment_cli} app "$@"\n'
    )
    desktop = Path(result["desktop_launcher"]).read_text()
    assert f"Exec={bin_dir / 'mn-ligand-app'}" in desktop
    assert "Terminal=true" in desktop


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
            "--bin-dir",
            str(tmp_path / "bin"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Start the app with: mn-ligand-app" in result.output
    assert not (tmp_path / "Desktop").exists()
