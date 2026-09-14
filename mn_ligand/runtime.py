from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_APP_HOME = Path("/mnt/data/RESULTS/mn-ligand-workdir")
DEFAULT_TMPDIR = DEFAULT_APP_HOME / "tmp"
DEFAULT_INPUT_DIR = Path("/tmp/mn-ligand-inputs")
SHARED_REFERENCE_DIR = Path("/mnt/db/reference_files")
RUNTIME_SETTINGS_SCHEMA_VERSION = 1
CPU_PROCESS_LIMIT_ENV = "MN_LIGAND_CPU_PROCESS_LIMIT"
UNIDOCK_PRO_MAX_COMPOUNDS_ENV = "MN_LIGAND_UNIDOCK_PRO_MAX_COMPOUNDS"
DEFAULT_UNIDOCK_PRO_MAX_COMPOUNDS = 10_000
VINA_COMPOUND_TIMEOUT_MINUTES_ENV = "MN_LIGAND_VINA_COMPOUND_TIMEOUT_MINUTES"
DEFAULT_VINA_COMPOUND_TIMEOUT_MINUTES = 15
NATIVE_THREAD_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def runtime_settings_path() -> Path:
    configured = os.getenv("MN_LIGAND_CONFIG")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else app_home() / "config" / "runtime.json"
    )


def load_runtime_settings() -> dict[str, Any]:
    path = runtime_settings_path()
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    version = int(payload.get("schema_version") or 0)
    return payload if version == RUNTIME_SETTINGS_SCHEMA_VERSION else {}


def cpu_process_limit_setting() -> int:
    configured = os.getenv(CPU_PROCESS_LIMIT_ENV)
    if configured is None:
        configured = load_runtime_settings().get("cpu_process_limit", 0)
    try:
        return max(0, int(configured or 0))
    except (TypeError, ValueError):
        return 0


def cpu_process_limit() -> int:
    configured = cpu_process_limit_setting()
    return configured if configured > 0 else max(1, int(os.cpu_count() or 1))


def unidock_pro_max_compounds() -> int:
    configured = os.getenv(UNIDOCK_PRO_MAX_COMPOUNDS_ENV)
    if configured is None:
        configured = load_runtime_settings().get(
            "unidock_pro_max_compounds",
            DEFAULT_UNIDOCK_PRO_MAX_COMPOUNDS,
        )
    try:
        return max(1, int(configured))
    except (TypeError, ValueError):
        return DEFAULT_UNIDOCK_PRO_MAX_COMPOUNDS


def vina_compound_timeout_minutes() -> int:
    configured = os.getenv(VINA_COMPOUND_TIMEOUT_MINUTES_ENV)
    if configured is None:
        configured = load_runtime_settings().get(
            "vina_compound_timeout_minutes",
            DEFAULT_VINA_COMPOUND_TIMEOUT_MINUTES,
        )
    try:
        return max(1, int(configured))
    except (TypeError, ValueError):
        return DEFAULT_VINA_COMPOUND_TIMEOUT_MINUTES


def adaptive_cpu_workers(
    work_items: int,
    *,
    requested: int | None = None,
    hard_cap: int | None = None,
) -> int:
    """Size independent CPU work without exceeding the global CPU limit."""
    available_work = max(1, int(work_items))
    limit = max(1, int(cpu_process_limit()))
    if requested is not None and int(requested) > 0:
        limit = min(limit, int(requested))
    if hard_cap is not None and int(hard_cap) > 0:
        limit = min(limit, int(hard_cap))
    return max(1, min(available_work, limit))


def _path_setting(key: str) -> str:
    value = load_runtime_settings().get(key)
    return str(value).strip() if value is not None else ""


def _configured_path(env_name: str, fallback: Path, *, setting_key: str = "") -> Path:
    value = os.getenv(env_name) or (_path_setting(setting_key) if setting_key else "")
    path = Path(value) if value else fallback
    return path.expanduser().resolve()


def app_home() -> Path:
    return _configured_path("MN_LIGAND_APP_HOME", DEFAULT_APP_HOME)


def runs_root(*, create: bool = True) -> Path:
    root = _configured_path(
        "MN_LIGAND_RUN_DIR", app_home() / "workdir" / "runs", setting_key="runs_dir"
    )
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def reference_root(*, create: bool = False) -> Path:
    fallback = SHARED_REFERENCE_DIR if SHARED_REFERENCE_DIR.is_dir() else app_home() / "reference_files"
    root = _configured_path(
        "MN_LIGAND_REFERENCE_DIR", fallback, setting_key="reference_dir"
    )
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def library_root(*, create: bool = False) -> Path:
    root = _configured_path(
        "MN_LIGAND_LIBRARY_DIR", app_home() / "libraries", setting_key="library_dir"
    )
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def temporary_root(*, create: bool = False) -> Path:
    configured = os.getenv("MN_LIGAND_TMP_DIR") or os.getenv("TMPDIR")
    root = Path(configured).expanduser().resolve() if configured else DEFAULT_TMPDIR.resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def input_root(*, create: bool = True) -> Path:
    root = _configured_path("MN_LIGAND_INPUT_DIR", DEFAULT_INPUT_DIR, setting_key="input_dir")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def save_runtime_settings(
    *,
    runs_dir: str | Path,
    reference_dir: str | Path,
    library_dir: str | Path | None = None,
    input_dir: str | Path | None = None,
    cpu_process_limit: int | None = None,
    unidock_pro_batch_limit: int | None = None,
    vina_compound_timeout: int | None = None,
    apply_to_process: bool = True,
) -> Path:
    resolved_runs = Path(runs_dir).expanduser().resolve()
    resolved_references = Path(reference_dir).expanduser().resolve()
    resolved_library = Path(library_dir).expanduser().resolve() if library_dir else library_root()
    resolved_input = Path(input_dir).expanduser().resolve() if input_dir else input_root(create=False)
    process_limit = (
        cpu_process_limit_setting()
        if cpu_process_limit is None
        else max(0, int(cpu_process_limit))
    )
    unidock_limit = (
        unidock_pro_max_compounds()
        if unidock_pro_batch_limit is None
        else max(1, int(unidock_pro_batch_limit))
    )
    vina_timeout = (
        vina_compound_timeout_minutes()
        if vina_compound_timeout is None
        else max(1, int(vina_compound_timeout))
    )
    for path in (resolved_runs, resolved_references, resolved_library, resolved_input):
        path.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RUNTIME_SETTINGS_SCHEMA_VERSION,
        "runs_dir": str(resolved_runs),
        "reference_dir": str(resolved_references),
        "library_dir": str(resolved_library),
        "input_dir": str(resolved_input),
        "cpu_process_limit": process_limit,
        "unidock_pro_max_compounds": unidock_limit,
        "vina_compound_timeout_minutes": vina_timeout,
    }
    target = runtime_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    if apply_to_process:
        os.environ["MN_LIGAND_RUN_DIR"] = str(resolved_runs)
        os.environ["MN_LIGAND_REFERENCE_DIR"] = str(resolved_references)
        os.environ["MN_LIGAND_LIBRARY_DIR"] = str(resolved_library)
        os.environ["MN_LIGAND_INPUT_DIR"] = str(resolved_input)
    return target


def resolve_run_dir(task_group: str, run_id: str, *, must_exist: bool = True) -> Path | None:
    for value in (task_group, run_id):
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            return None
    root = runs_root().resolve()
    group_dir = (root / task_group).resolve()
    candidate = (group_dir / run_id).resolve()
    if group_dir.parent != root or candidate.parent != group_dir:
        return None
    if must_exist and not candidate.is_dir():
        return None
    return candidate


def ensure_runtime_home(
    home: str | Path = DEFAULT_APP_HOME,
    tmpdir: str | Path = DEFAULT_TMPDIR,
) -> Path:
    home_path = Path(home).expanduser().resolve()
    tmp_path = Path(tmpdir).expanduser().resolve()

    home_path.mkdir(parents=True, exist_ok=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (home_path / "storage").mkdir(exist_ok=True)
    (home_path / "reference_files").mkdir(exist_ok=True)
    (home_path / "libraries").mkdir(exist_ok=True)
    (home_path / "workdir" / "runs").mkdir(parents=True, exist_ok=True)
    return home_path
