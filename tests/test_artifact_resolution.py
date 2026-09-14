from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from mn_ligand.app.pages.common import resolve_run_artifact_path


def test_relative_artifact_resolves_under_configured_run_root() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        run_root = Path(temp_dir) / "runs"
        artifact = run_root / "structure-jobs" / "run-1" / "complex.pdb"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("END\n")
        with patch.dict(os.environ, {"MN_LIGAND_RUN_DIR": str(run_root)}):
            resolved = resolve_run_artifact_path(
                "structure-jobs/run-1/complex.pdb",
                must_exist=True,
            )
        assert resolved == artifact


def test_legacy_ovo_home_path_maps_to_configured_run_root() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        run_root = Path(temp_dir) / "runs"
        artifact = run_root / "structure-jobs" / "legacy-run" / "complex.pdb"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("END\n")
        legacy = "/old/ovo-ligand/.ovo-home/workdir/runs/structure-jobs/legacy-run/complex.pdb"
        with patch.dict(os.environ, {"MN_LIGAND_RUN_DIR": str(run_root)}):
            resolved = resolve_run_artifact_path(legacy, must_exist=True)
        assert resolved == artifact


def test_missing_absolute_path_is_preserved_when_existence_not_required() -> None:
    missing = Path("/does/not/exist/mn-ligand-artifact.pdb")
    assert resolve_run_artifact_path(missing) == missing
    assert resolve_run_artifact_path(missing, must_exist=True) is None
