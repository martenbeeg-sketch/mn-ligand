from __future__ import annotations

import json
from pathlib import Path

import pytest

from mn_ligand.core.manifests import ToolRegistry, load_tool_registry


def _tool(tool_id: str = "fixture") -> dict[str, object]:
    return {
        "tool_id": tool_id,
        "name": "Fixture tool",
        "version": "1.0",
        "image": "fixture:1.0",
        "accepted_artifact_types": ["prepared_target"],
        "produced_artifact_types": ["pocket_set"],
        "resources": {"gpu": False},
        "healthcheck": ["docker", "image", "inspect", "{image}"],
    }


def test_bundled_registry_declares_current_migrated_tools() -> None:
    registry = load_tool_registry()

    assert registry.schema_version == 1
    assert {tool.image for tool in registry.tools} == {
        "ovolig-structure:latest",
        "ovolig-docking:latest",
        "ovolig-fpocket:latest",
        "ovolig-p2rank:latest",
        "mnprot-pesto-cu128:latest",
        "avgu-docking-suite-cuda:latest",
        "alphafast:latest",
        "ovoex-boltz2:latest",
        "ovolig-boltzina-cu128:latest",
        "ovolig-nesso-cu128:latest",
        "ovolig-posebusters:latest",
        "ovolig-plip:latest",
        "ovolig-pandamap:latest",
        "ovolig-md-cu128:latest",
        "ovolig-gromacs-cu128:latest",
        "ovolig-admet:latest",
        "ovolig-qc:latest",
        "openvs:local",
        "ovolig-omtra-cu128:latest",
        "ovolig-pocketxmol-cu128:latest",
        "ovolig-flowr-root-cu128:latest",
        "ovolig-conditar-cu128:latest",
        "ovolig-paopt-cu128:latest",
        "ovolig-drugrpg-cu128:latest",
        "ovolig-pfm-cu128:latest",
        "ovolig-pocketflow-cu128:latest",
        "ovolig-pgmg-cu128:latest",
    }
    assert registry.get("fpocket").resources.gpu is False
    assert registry.get("p2rank").resources.gpu is False
    assert registry.get("p2rank").code_license == "MIT"
    assert registry.get("pesto_ligand_interface").resources.cuda_min == "12.8"
    assert registry.get("unidock_pro").produced_artifact_types == ("pose_set", "docking_scores")
    assert registry.get("openmm_md").image == "ovolig-md-cu128:latest"
    assert registry.get("gromacs_md").image == "ovolig-gromacs-cu128:latest"
    assert registry.get("gromacs_md").integration_status == "validated"
    assert registry.get("plip").integration_status == "validated"
    assert registry.get("pandamap").integration_status == "validated"
    assert registry.get("gromacs_md").validation_evidence[0].job_ids[-1] == (
        "07715dda-e000-4d00-b6dd-ad9c7f1131c1"
    )
    assert registry.get("plip").validation_evidence[0].image_digest == (
        "sha256:9339862b17a5db844f9c5223f0f3dddf7f95030233411432a84e98c6f7054f5b"
    )
    assert registry.get("openvs").integration_status == "experimental"
    assert registry.get("openvs").resources.gpu is False
    assert registry.get("openvs").produced_artifact_types == (
        "screening_result", "pose_set", "docking_scores", "docked_complex",
        "convergence_result"
    )
    assert registry.get("quantum_chemistry").integration_status == "disabled"
    assert registry.get("legacy_docking").integration_status == "compatibility"
    generation_tools = (
        "omtra_generation",
        "pocketxmol_generation",
        "flowr_root_generation",
        "conditar_generation",
        "paopt_generation",
        "drugrpg_generation",
        "pfm_generation",
        "pocketflow_generation",
        "pgmg_generation",
    )
    assert all(
        registry.get(tool_id).integration_status == "experimental"
        for tool_id in generation_tools
    )
    assert all(
        registry.get(tool_id).resources.gpu is True
        for tool_id in generation_tools
    )


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "tools.json"
    path.write_text(json.dumps({"schema_version": 1, "tools": [_tool(), _tool()]}))

    with pytest.raises(ValueError, match="duplicate"):
        load_tool_registry(path)


def test_registry_rejects_absolute_reference_paths() -> None:
    tool = _tool()
    tool["references"] = [{"path": "/host/checkpoint.pt"}]

    with pytest.raises(ValueError, match="relative"):
        ToolRegistry.from_dict({"schema_version": 1, "tools": [tool]})


def test_registry_rejects_gpu_resources_for_cpu_tool() -> None:
    tool = _tool()
    tool["resources"] = {"gpu": False, "min_vram_gb": 4}

    with pytest.raises(ValueError, match="CPU tools"):
        ToolRegistry.from_dict({"schema_version": 1, "tools": [tool]})


def test_registry_rejects_unknown_integration_status() -> None:
    tool = _tool()
    tool["integration_status"] = "installed-means-ready"

    with pytest.raises(ValueError, match="integration status"):
        ToolRegistry.from_dict({"schema_version": 1, "tools": [tool]})


def test_registry_rejects_incomplete_validation_evidence() -> None:
    tool = _tool()
    tool["validation_evidence"] = [{"date": "2026-09-16"}]

    with pytest.raises(ValueError, match="Validation evidence requires"):
        ToolRegistry.from_dict({"schema_version": 1, "tools": [tool]})
