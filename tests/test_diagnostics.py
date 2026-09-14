from __future__ import annotations

import subprocess
from pathlib import Path

from mn_ligand.core.diagnostics import diagnostics_exit_code, run_diagnostics
from mn_ligand.core.manifests import ToolRegistry


def _registry() -> ToolRegistry:
    return ToolRegistry.from_dict(
        {
            "schema_version": 1,
            "tools": [
                {
                    "tool_id": "fixture",
                    "name": "Fixture",
                    "version": "1",
                    "image": "fixture:1",
                    "image_digest": "sha256:fixture",
                    "accepted_artifact_types": ["prepared_target"],
                    "produced_artifact_types": ["pocket_set"],
                    "resources": {"gpu": False},
                    "healthcheck": ["docker", "image", "inspect", "{image}"],
                }
            ],
        }
    )


def _configure_paths(tmp_path: Path, monkeypatch) -> None:
    for name in ("runs", "references", "libraries", "inputs", "temporary"):
        path = tmp_path / name
        path.mkdir()
        env_name = {
            "runs": "MN_LIGAND_RUN_DIR",
            "references": "MN_LIGAND_REFERENCE_DIR",
            "libraries": "MN_LIGAND_LIBRARY_DIR",
            "inputs": "MN_LIGAND_INPUT_DIR",
            "temporary": "MN_LIGAND_TMP_DIR",
        }[name]
        monkeypatch.setenv(env_name, str(path))


def test_diagnostics_pass_with_paths_docker_and_image(tmp_path: Path, monkeypatch) -> None:
    _configure_paths(tmp_path, monkeypatch)

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output = "27.5.1\n" if "version" in command else "[]\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    results = run_diagnostics(
        registry=_registry(), runner=runner, which=lambda name: f"/usr/bin/{name}" if name == "docker" else None
    )

    assert diagnostics_exit_code(results) == 0
    assert next(item for item in results if item.check_id == "docker").status == "pass"
    assert next(item for item in results if item.check_id == "image:fixture").status == "pass"


def test_diagnostics_fail_when_docker_is_missing(tmp_path: Path, monkeypatch) -> None:
    _configure_paths(tmp_path, monkeypatch)

    results = run_diagnostics(registry=_registry(), which=lambda _: None)

    assert diagnostics_exit_code(results) == 1
    assert next(item for item in results if item.check_id == "docker").status == "fail"
    assert next(item for item in results if item.check_id == "image:fixture").status == "warn"
