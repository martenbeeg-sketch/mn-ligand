from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    registered_tool,
    write_command_record,
    write_registered_command_record,
)
from mn_ligand.core.manifests import load_tool_registry


def test_cpu_command_uses_portable_mounts_and_rootful_user(tmp_path: Path) -> None:
    registry = load_tool_registry()
    source = tmp_path / "target.pdb"
    source.write_text("ATOM\n")
    output = tmp_path / "output"
    output.mkdir()
    spec = DockerRunSpec(
        tool=registry.get("fpocket"),
        command=("fpocket", "-f", "/work/input/target.pdb"),
        mounts=(
            DockerMount(source, "/work/input/target.pdb", read_only=True),
            DockerMount(output, "/work/output"),
        ),
        workdir="/work/output",
    )

    command = build_docker_command(spec, rootless=False)

    assert command[:3] == ["docker", "run", "--rm"]
    assert ["--user", f"{os.getuid()}:{os.getgid()}"] == command[3:5]
    assert f"{source.resolve()}:/work/input/target.pdb:ro" in command
    assert "--gpus" not in command
    assert command[-3:] == ["fpocket", "-f", "/work/input/target.pdb"]


def test_gpu_command_selects_gpu_one_and_omits_user_for_rootless(tmp_path: Path) -> None:
    registry = load_tool_registry()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = DockerRunSpec(
        tool=registry.get("unidock_pro"),
        command=("udp", "--config", "/work/input/config.json"),
        mounts=(DockerMount(workspace, "/work"),),
        gpu_devices=(1,),
        shm_size="8g",
        environment={"OMP_NUM_THREADS": "4"},
    )

    command = build_docker_command(spec, rootless=True)

    assert command[command.index("--gpus") + 1] == "device=1"
    assert "--user" not in command
    assert command[command.index("--shm-size") + 1] == "8g"
    assert "OMP_NUM_THREADS=4" in command


def test_cpu_tool_rejects_gpu_selection() -> None:
    tool = load_tool_registry().get("fpocket")

    with pytest.raises(ValueError, match="CPU tool"):
        DockerRunSpec(tool=tool, command=("fpocket",), gpu_devices=(1,))


def test_command_record_captures_tool_resources_and_gpu(tmp_path: Path) -> None:
    tool = load_tool_registry().get("openmm_md")
    spec = DockerRunSpec(tool=tool, command=("python", "run.py"), gpu_devices=(1,))
    command = build_docker_command(spec, rootless=True)

    target = write_command_record(tmp_path, spec, command, selected_gpu_ids=(1,))
    payload = json.loads(target.read_text())

    assert payload["tool_id"] == "openmm_md"
    assert payload["selected_gpu_ids"] == [1]
    assert payload["resources"]["cuda_min"] == "12.8"
    assert payload["argv"] == command


def test_registered_tool_supports_image_override_and_cpu_mode(tmp_path: Path) -> None:
    tool = registered_tool("openmm_md", image="custom-md:test")
    spec = DockerRunSpec(
        tool=tool,
        command=("python", "run.py"),
        gpu_enabled=False,
    )

    command = build_docker_command(spec, rootless=True)
    target = write_registered_command_record(
        tmp_path,
        tool_id="openmm_md",
        commands=(command,),
        image="custom-md:test",
        filename="command-secondary.json",
    )
    payload = json.loads(target.read_text())

    assert "--gpus" not in command
    assert "custom-md:test" in command
    assert payload["tool_id"] == "openmm_md"
    assert payload["image"] == "custom-md:test"
    assert payload["commands"] == [command]
