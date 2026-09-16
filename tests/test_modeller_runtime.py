from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from mn_ligand import modeller_runtime


def _without_modeller_environment() -> dict[str, str]:
    excluded = {
        "CONDA_EXE",
        "CONDA_PREFIX",
        "MAMBA_ROOT_PREFIX",
        modeller_runtime.MODELLER_PYTHON_ENV,
    }
    return {key: value for key, value in os.environ.items() if key not in excluded}


def test_explicit_modeller_python_takes_priority(tmp_path: Path) -> None:
    interpreter = tmp_path / "custom" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    environment = _without_modeller_environment()
    environment[modeller_runtime.MODELLER_PYTHON_ENV] = str(interpreter)

    with patch.dict(os.environ, environment, clear=True):
        assert modeller_runtime.modeller_python() == interpreter.resolve()


def test_sibling_conda_environment_is_discovered(tmp_path: Path, monkeypatch) -> None:
    application_prefix = tmp_path / "conda" / "envs" / "mn-ligand"
    interpreter = (
        application_prefix.parent
        / modeller_runtime.MODELLER_ENV_NAME
        / "bin"
        / "python"
    )
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    monkeypatch.setattr(modeller_runtime.sys, "prefix", str(application_prefix))

    with patch.dict(os.environ, _without_modeller_environment(), clear=True):
        assert modeller_runtime.modeller_python() == interpreter.resolve()
