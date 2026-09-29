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
        "mn-structure:latest",
        "mn-docking:latest",
        "mn-fpocket:4bb0d84",
        "mn-p2rank:d8c8e0d",
        "mn-pesto:cu128",
        "mn-docking-suite:cu128",
        "mn-alphafast:cu128",
        "mn-boltz2:cu128",
        "mn-boltzina:cu128",
        "mn-nesso:1.0.0-cu128",
        "mn-posebusters:1a5f26a",
        "mn-plip:latest",
        "mn-pandamap:0007347",
        "mn-md:cu128",
        "mn-gromacs:2026.3-cu128",
        "mn-admet:latest",
        "mn-qc:latest",
        "mn-openvs:local",
        "mn-omtra:cu128-745127d",
        "mn-pocketxmol:cu128-65488cf",
        "mn-flowr-root:cu128-b2263e2",
        "mn-conditar:cu128-4294d286",
        "mn-paopt:cu128-4294d286",
        "mn-drugrpg:cu128-6fa0e41",
        "mn-pfm:cu128-33be6c1",
        "mn-pocketflow:cu128-a31a5a0",
        "mn-pgmg:cu128-85fb712",
        "mn-lddm:cu128-f254fb4",
    }
    assert registry.get("fpocket").resources.gpu is False
    assert registry.get("p2rank").resources.gpu is False
    assert registry.get("p2rank").code_license == "MIT"
    assert registry.get("pesto_ligand_interface").resources.cuda_min == "12.8"
    assert registry.get("unidock_pro").produced_artifact_types == ("pose_set", "docking_scores")
    assert registry.get("openmm_md").image == "mn-md:cu128"
    assert registry.get("gromacs_md").image == "mn-gromacs:2026.3-cu128"
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
