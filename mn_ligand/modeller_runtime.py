from __future__ import annotations

import os
import sys
from pathlib import Path


MODELLER_ENV_NAME = "mn-ligand-modeller"
MODELLER_PYTHON_ENV = "MN_LIGAND_MODELLER_PYTHON"


def _environment_python(prefix: Path) -> Path:
    return prefix / "bin" / "python"


def modeller_python_candidates() -> tuple[Path, ...]:
    """Return portable candidates for the isolated MODELLER interpreter."""
    candidates: list[Path] = []

    configured = os.getenv(MODELLER_PYTHON_ENV, "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())

    prefixes = [Path(sys.prefix)]
    active_prefix = os.getenv("CONDA_PREFIX", "").strip()
    if active_prefix:
        prefixes.append(Path(active_prefix).expanduser())
    for prefix in prefixes:
        if prefix.parent.name == "envs":
            candidates.append(_environment_python(prefix.parent / MODELLER_ENV_NAME))

    root_prefix = os.getenv("MAMBA_ROOT_PREFIX", "").strip()
    if root_prefix:
        candidates.append(
            _environment_python(Path(root_prefix).expanduser() / "envs" / MODELLER_ENV_NAME)
        )

    conda_executable = os.getenv("CONDA_EXE", "").strip()
    if conda_executable:
        conda_root = Path(conda_executable).expanduser().resolve().parent.parent
        candidates.append(_environment_python(conda_root / "envs" / MODELLER_ENV_NAME))

    home = Path.home()
    for root in ("mambaforge", "miniforge3", "miniconda3", "anaconda3", ".conda"):
        candidates.append(_environment_python(home / root / "envs" / MODELLER_ENV_NAME))

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return tuple(unique)


def modeller_python() -> Path:
    """Resolve MODELLER without assuming a username or Conda distribution path."""
    candidates = modeller_python_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if candidates:
        return candidates[0]
    return _environment_python(
        Path.home() / ".conda" / "envs" / MODELLER_ENV_NAME
    ).resolve(strict=False)
