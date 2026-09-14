from __future__ import annotations

import ast
from pathlib import Path

from mn_ligand.core.manifests import load_tool_registry

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_raw_docker_run_construction_is_centralized() -> None:
    violations: list[str] = []
    source_root = PROJECT_DIR / "mn_ligand"
    allowed = source_root / "core" / "docker_runner.py"
    for path in source_root.rglob("*.py"):
        if path == allowed:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple)) or len(node.elts) < 2:
                continue
            first, second = node.elts[:2]
            if (
                isinstance(first, ast.Constant)
                and first.value == "docker"
                and isinstance(second, ast.Constant)
                and second.value == "run"
            ):
                violations.append(str(path.relative_to(PROJECT_DIR)))

    assert violations == []


def test_openvs_uses_centralized_cpu_docker_adapter() -> None:
    registry = load_tool_registry()
    workflow_sources = "\n".join(
        path.read_text()
        for path in (PROJECT_DIR / "mn_ligand" / "workflows").glob("*.py")
    ).lower()

    assert registry.get("openvs").integration_status == "experimental"
    assert registry.get("openvs").resources.gpu is False
    assert "run_openvs_docking_job" in workflow_sources
