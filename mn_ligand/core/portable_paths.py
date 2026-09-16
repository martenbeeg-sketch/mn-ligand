from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Final
from urllib.parse import urlsplit

from mn_ligand.runtime import app_home, library_root, reference_root, runs_root


RUNS_SCHEME: Final = "runs"
REFERENCE_SCHEME: Final = "reference"
LIBRARY_SCHEME: Final = "library"
APP_SCHEME: Final = "app"
PORTABLE_SCHEMES: Final = frozenset(
    {RUNS_SCHEME, REFERENCE_SCHEME, LIBRARY_SCHEME, APP_SCHEME}
)


def _safe_relative(value: str) -> Path | None:
    relative = PurePosixPath(value.lstrip("/"))
    if not relative.parts or ".." in relative.parts:
        return None
    return Path(*relative.parts)


def _portable_uri_root(scheme: str) -> Path:
    roots = {
        RUNS_SCHEME: runs_root(create=False),
        REFERENCE_SCHEME: reference_root(create=False),
        LIBRARY_SCHEME: library_root(create=False),
        APP_SCHEME: app_home(),
    }
    return roots[scheme]


def _uri_candidate(raw: str, *, run_dir: Path | None = None) -> Path | None:
    parsed = urlsplit(raw)
    if parsed.scheme not in PORTABLE_SCHEMES:
        return None
    joined = "/".join(part for part in (parsed.netloc, parsed.path) if part)
    relative = _safe_relative(joined)
    if relative is None:
        return None
    root = _portable_uri_root(parsed.scheme)
    if parsed.scheme == RUNS_SCHEME and run_dir is not None:
        resolved_run = run_dir.resolve()
        if len(resolved_run.parents) >= 2:
            root = resolved_run.parents[1]
    return root / relative


def _legacy_candidate(raw: str) -> Path | None:
    normalized = raw.replace("\\", "/")
    mappings = (
        ("/workdir/runs/", runs_root(create=False)),
        ("/.ovo-home/workdir/runs/", runs_root(create=False)),
        ("/reference_files/", reference_root(create=False)),
        ("/libraries/", library_root(create=False)),
    )
    for marker, root in mappings:
        if marker not in normalized:
            continue
        relative = _safe_relative(normalized.split(marker, 1)[1])
        if relative is not None:
            return root / relative
    return None


def resolve_stored_path(
    value: str | Path | None,
    *,
    run_dir: Path | None = None,
    must_exist: bool = False,
) -> Path | None:
    """Resolve current and relocated paths without modifying stored metadata.

    Existing recorded paths always win. Relocation is attempted only when the
    original path is unavailable, preserving behavior on the source machine.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    portable = _uri_candidate(raw, run_dir=run_dir)
    if portable is not None:
        candidate = portable
    else:
        recorded = Path(raw).expanduser()
        if recorded.exists():
            return recorded
        if not recorded.is_absolute():
            base = run_dir.resolve() if run_dir is not None else runs_root(create=False)
            candidate = base / recorded
        else:
            candidate = _legacy_candidate(raw)
            if candidate is None:
                return None if must_exist else recorded

    if must_exist and not candidate.exists():
        return None
    return candidate


def portable_path(value: str | Path, *, run_dir: Path | None = None) -> str:
    """Encode an operational path relative to a managed runtime root."""
    path = Path(value).expanduser().resolve()
    roots: list[tuple[str, Path]] = []
    if run_dir is not None:
        roots.append(("", run_dir.resolve()))
    roots.extend(
        (
            (RUNS_SCHEME, runs_root(create=False).resolve()),
            (REFERENCE_SCHEME, reference_root(create=False).resolve()),
            (LIBRARY_SCHEME, library_root(create=False).resolve()),
            (APP_SCHEME, app_home().resolve()),
        )
    )
    for scheme, root in roots:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        return relative if not scheme else f"{scheme}:///{relative}"
    return str(path)
