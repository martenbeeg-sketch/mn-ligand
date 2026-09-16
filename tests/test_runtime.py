from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mn_ligand import runtime


RUNTIME_ENV_VARS = {
    "MN_LIGAND_APP_HOME",
    "MN_LIGAND_RUN_DIR",
    "MN_LIGAND_REFERENCE_DIR",
    "MN_LIGAND_LIBRARY_DIR",
    "MN_LIGAND_TMP_DIR",
    "MN_LIGAND_INPUT_DIR",
    "MN_LIGAND_CONFIG",
    "MN_LIGAND_INSTALL_CONFIG",
    "MN_LIGAND_REQUIRED_MOUNT",
    "MN_LIGAND_CPU_PROCESS_LIMIT",
    "MN_LIGAND_UNIDOCK_PRO_MAX_COMPOUNDS",
    "TMPDIR",
}


def _without_runtime_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in RUNTIME_ENV_VARS}


def test_defaults_preserve_legacy_storage_when_present() -> None:
    with patch.dict(os.environ, _without_runtime_environment(), clear=True):
        expected_home = runtime.DEFAULT_APP_HOME.resolve()
        assert runtime.app_home() == expected_home
        assert runtime.runs_root(create=False) == expected_home / "workdir" / "runs"
        expected_references = (
            runtime.SHARED_REFERENCE_DIR
            if runtime.SHARED_REFERENCE_DIR.is_dir()
            else expected_home / "reference_files"
        )
        assert runtime.reference_root() == expected_references
        assert runtime.library_root() == expected_home / "libraries"
        assert runtime.temporary_root() == expected_home / "tmp"
        assert runtime.input_root(create=False) == Path("/tmp/mn-ligand-inputs")


def test_fresh_install_uses_xdg_user_data_directory(tmp_path: Path) -> None:
    env = _without_runtime_environment()
    env["XDG_DATA_HOME"] = str(tmp_path / "xdg-data")
    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(runtime, "LEGACY_APP_HOME", tmp_path / "missing-legacy"),
    ):
        assert runtime.default_app_home() == tmp_path / "xdg-data" / "mn-ligand"


def test_app_home_derives_portable_subdirectories() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        env = _without_runtime_environment()
        env["MN_LIGAND_APP_HOME"] = temp_dir
        with patch.dict(os.environ, env, clear=True):
            home = Path(temp_dir).resolve()
            assert runtime.runs_root(create=False) == home / "workdir" / "runs"
            expected_references = (
                runtime.SHARED_REFERENCE_DIR
                if runtime.SHARED_REFERENCE_DIR.is_dir()
                else home / "reference_files"
            )
            assert runtime.reference_root() == expected_references
            assert runtime.library_root() == home / "libraries"


def test_explicit_paths_override_derived_paths() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        base = Path(temp_dir)
        env = _without_runtime_environment()
        env.update(
            {
                "MN_LIGAND_APP_HOME": str(base / "home"),
                "MN_LIGAND_RUN_DIR": str(base / "runs"),
                "MN_LIGAND_REFERENCE_DIR": str(base / "references"),
                "MN_LIGAND_LIBRARY_DIR": str(base / "libraries"),
                "MN_LIGAND_TMP_DIR": str(base / "tmp"),
                "MN_LIGAND_INPUT_DIR": str(base / "inputs"),
            }
        )
        with patch.dict(os.environ, env, clear=True):
            assert runtime.runs_root(create=False) == (base / "runs").resolve()
            assert runtime.reference_root() == (base / "references").resolve()
            assert runtime.library_root() == (base / "libraries").resolve()
            assert runtime.temporary_root() == (base / "tmp").resolve()
            assert runtime.input_root(create=False) == (base / "inputs").resolve()


def test_persisted_runtime_paths_are_loaded_and_created(tmp_path: Path) -> None:
    env = _without_runtime_environment()
    env["MN_LIGAND_APP_HOME"] = str(tmp_path / "home")
    env["MN_LIGAND_CONFIG"] = str(tmp_path / "config" / "runtime.json")
    with patch.dict(os.environ, env, clear=True):
        target = runtime.save_runtime_settings(
            runs_dir=tmp_path / "saved-runs",
            reference_dir=tmp_path / "saved-references",
            library_dir=tmp_path / "saved-libraries",
            input_dir=tmp_path / "saved-inputs",
            cpu_process_limit=12,
            unidock_pro_batch_limit=15_000,
            apply_to_process=False,
        )
        assert target == (tmp_path / "config" / "runtime.json").resolve()
        assert runtime.runs_root(create=False) == (tmp_path / "saved-runs").resolve()
        assert runtime.reference_root(create=False) == (tmp_path / "saved-references").resolve()
        assert runtime.load_runtime_settings()["schema_version"] == 1
        assert runtime.cpu_process_limit_setting() == 12
        assert runtime.cpu_process_limit() == 12
        assert runtime.unidock_pro_max_compounds() == 15_000
        assert runtime.temporary_root() == (tmp_path / "home" / "tmp").resolve()


def test_installation_record_bootstraps_app_home(tmp_path: Path) -> None:
    env = _without_runtime_environment()
    env["MN_LIGAND_INSTALL_CONFIG"] = str(tmp_path / "config" / "installation.json")
    installed_home = tmp_path / "data" / "mn-ligand"
    with patch.dict(os.environ, env, clear=True):
        target = runtime.save_installation_settings(runtime_home=installed_home)
        os.environ.pop("MN_LIGAND_APP_HOME", None)

        assert target == (tmp_path / "config" / "installation.json").resolve()
        assert runtime.installation_is_configured()
        assert runtime.app_home() == installed_home.resolve()


def test_missing_required_mount_is_rejected_before_directory_creation(
    tmp_path: Path,
) -> None:
    env = _without_runtime_environment()
    env["MN_LIGAND_REQUIRED_MOUNT"] = str(tmp_path / "unmounted-pool")
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="not mounted"):
            runtime.ensure_runtime_home(tmp_path / "unmounted-pool" / "app")

    assert not (tmp_path / "unmounted-pool" / "app").exists()


def test_cpu_process_limit_supports_automatic_and_environment_override(
    tmp_path: Path,
) -> None:
    env = _without_runtime_environment()
    env["MN_LIGAND_CONFIG"] = str(tmp_path / "missing-runtime.json")
    with patch.dict(os.environ, env, clear=True):
        with patch("mn_ligand.runtime.os.cpu_count", return_value=24):
            assert runtime.cpu_process_limit_setting() == 0
            assert runtime.cpu_process_limit() == 24
    env["MN_LIGAND_CPU_PROCESS_LIMIT"] = "6"
    with patch.dict(os.environ, env, clear=True):
        assert runtime.cpu_process_limit_setting() == 6
        assert runtime.cpu_process_limit() == 6


def test_unidock_batch_limit_defaults_and_supports_environment_override(
    tmp_path: Path,
) -> None:
    env = _without_runtime_environment()
    env["MN_LIGAND_CONFIG"] = str(tmp_path / "missing-runtime.json")
    with patch.dict(os.environ, env, clear=True):
        assert runtime.unidock_pro_max_compounds() == 10_000
    env["MN_LIGAND_UNIDOCK_PRO_MAX_COMPOUNDS"] = "25000"
    with patch.dict(os.environ, env, clear=True):
        assert runtime.unidock_pro_max_compounds() == 25_000


def test_ensure_runtime_home_creates_required_directories() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        base = Path(temp_dir)
        home = runtime.ensure_runtime_home(base / "home", base / "tmp")
        assert (home / "storage").is_dir()
        assert (home / "reference_files").is_dir()
        assert (home / "libraries").is_dir()
        assert (home / "workdir" / "runs").is_dir()
        assert (base / "tmp").is_dir()


def test_resolve_run_dir_rejects_path_traversal() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        env = _without_runtime_environment()
        env["MN_LIGAND_RUN_DIR"] = temp_dir
        run_dir = Path(temp_dir) / "protein-import" / "run-1"
        run_dir.mkdir(parents=True)
        with patch.dict(os.environ, env, clear=True):
            assert runtime.resolve_run_dir("protein-import", "run-1") == run_dir.resolve()
            assert runtime.resolve_run_dir("protein-import", "..") is None
            assert runtime.resolve_run_dir("../outside", "run-1") is None
            assert runtime.resolve_run_dir("protein-import", "nested/run") is None
