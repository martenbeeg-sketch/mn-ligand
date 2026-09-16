from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mn_ligand.core.worker_service import (
    CPU_SERVICE_NAME,
    install_worker_service,
    manage_worker_services,
    parse_gpu_ids,
    render_cpu_worker_service,
    render_worker_service,
    service_instance_name,
    uninstall_worker_service,
)


class RecordingRunner:
    def __init__(self, *, returncode: int = 0) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.returncode = returncode

    def __call__(self, command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(tuple(command))
        return subprocess.CompletedProcess(command, self.returncode, stdout="ok\n", stderr="")


def test_render_worker_service_uses_explicit_runtime_and_python_paths(tmp_path: Path) -> None:
    unit = render_worker_service(
        python_executable="/opt/mn/bin/python",
        project_dir=tmp_path / "project",
        runtime_home=tmp_path / "runtime",
        tmp_dir=tmp_path / "runtime" / "tmp",
        modeller_executable=tmp_path / "conda" / "envs" / "mn-ligand-modeller" / "bin" / "python",
    )

    assert "Requires=docker.service" in unit
    assert "After=docker.service" in unit
    assert f"WorkingDirectory={tmp_path / 'project'}" in unit
    assert f'Environment="MN_LIGAND_APP_HOME={tmp_path / "runtime"}"' in unit
    assert (
        f'Environment="MN_LIGAND_MODELLER_PYTHON='
        f'{tmp_path / "conda" / "envs" / "mn-ligand-modeller" / "bin" / "python"}"'
        in unit
    )
    assert 'ExecStart="/opt/mn/bin/python" -m mn_ligand.cli worker' in unit
    assert "--gpu-ids %i --worker-id mn-ligand-gpu-%i" in unit
    assert "--job-class gpu" in unit
    assert "Restart=on-failure" in unit
    assert "KillSignal=SIGINT" in unit
    assert "WantedBy=default.target" in unit


def test_render_cpu_worker_service_has_no_gpu_scope(tmp_path: Path) -> None:
    unit = render_cpu_worker_service(
        python_executable="/opt/mn/bin/python",
        project_dir=tmp_path / "project",
        runtime_home=tmp_path / "runtime",
        tmp_dir=tmp_path / "runtime" / "tmp",
    )

    assert "Description=mn-ligand durable CPU worker" in unit
    assert "--worker-id mn-ligand-cpu-0 --job-class cpu" in unit
    assert "--gpu-ids" not in unit


def test_worker_units_require_configured_data_mount(tmp_path: Path) -> None:
    mount = tmp_path / "data"
    unit = render_worker_service(
        runtime_home=mount / "mn-ligand",
        tmp_dir=mount / "mn-ligand" / "tmp",
        required_mount_path=mount,
    )

    assert f'RequiresMountsFor="{mount}"' in unit
    assert f'ConditionPathIsMountPoint="{mount}"' in unit
    assert f'Environment="MN_LIGAND_REQUIRED_MOUNT={mount}"' in unit


def test_install_writes_template_and_enables_selected_instances(tmp_path: Path) -> None:
    runner = RecordingRunner()

    destination = install_worker_service(
        (0, 1),
        unit_dir=tmp_path,
        python_executable="/opt/mn/bin/python",
        project_dir=tmp_path / "project",
        runtime_home=tmp_path / "runtime",
        tmp_dir=tmp_path / "runtime" / "tmp",
        runner=runner,
    )

    assert destination == tmp_path / "mn-ligand-worker@.service"
    assert destination.is_file()
    assert (tmp_path / CPU_SERVICE_NAME).is_file()
    assert runner.commands == [
        ("systemctl", "--user", "daemon-reload"),
        (
            "systemctl",
            "--user",
            "enable",
            "--now",
            "mn-ligand-worker@0.service",
            "mn-ligand-worker@1.service",
            CPU_SERVICE_NAME,
        ),
    ]


def test_manage_status_preserves_nonzero_status_for_cli(tmp_path: Path) -> None:
    runner = RecordingRunner(returncode=3)

    result = manage_worker_services("status", (1,), runner=runner, check=False)

    assert result.returncode == 3
    assert runner.commands == [
        (
            "systemctl",
            "--user",
            "status",
            "--no-pager",
            "--full",
            "mn-ligand-worker@1.service",
            CPU_SERVICE_NAME,
        )
    ]


def test_uninstall_disables_instances_and_removes_template(tmp_path: Path) -> None:
    destination = tmp_path / "mn-ligand-worker@.service"
    destination.write_text("unit")
    cpu_destination = tmp_path / CPU_SERVICE_NAME
    cpu_destination.write_text("unit")
    runner = RecordingRunner()

    removed = uninstall_worker_service((0,), unit_dir=tmp_path, runner=runner)

    assert removed == destination
    assert not destination.exists()
    assert not cpu_destination.exists()
    assert runner.commands == [
        (
            "systemctl",
            "--user",
            "disable",
            "--now",
            "mn-ligand-worker@0.service",
            CPU_SERVICE_NAME,
        ),
        ("systemctl", "--user", "daemon-reload"),
    ]


def test_gpu_id_validation_and_instance_names() -> None:
    assert parse_gpu_ids("1,0,1") == (1, 0)
    assert parse_gpu_ids("") == (0, 1)
    assert service_instance_name(0) == "mn-ligand-worker@0.service"
    with pytest.raises(ValueError, match="comma-separated"):
        parse_gpu_ids("0,no")
    with pytest.raises(ValueError, match="negative"):
        service_instance_name(-1)
