from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def _atomic_text(path: Path, text: str, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.chmod(mode)
    temporary.replace(path)


def _environment_cli() -> Path:
    override = os.getenv("MN_LIGAND_CLI", "").strip()
    candidate = (
        Path(override).expanduser()
        if override
        else Path(sys.executable).with_name("mn-ligand")
    ).resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(
            f"mn-ligand executable was not found beside the active Python: {candidate}. "
            "Install the project with `python -m pip install -e .`."
        )
    return candidate


def install_user_launchers(
    *,
    bin_dir: Path | None = None,
    desktop_dir: Path | None = None,
    create_desktop: bool = True,
    worker_gpu_ids: Sequence[int] | None = None,
    install_workers: bool = True,
) -> dict[str, Any]:
    """Install launchers and, by default, shared user worker services."""
    cli = _environment_cli()
    selected_gpu_ids: tuple[int, ...] = ()
    if install_workers:
        if worker_gpu_ids is None:
            from mn_ligand.core.resources import discover_gpu_ids

            selected_gpu_ids = discover_gpu_ids()
        else:
            selected_gpu_ids = tuple(
                dict.fromkeys(int(value) for value in worker_gpu_ids)
            )
        if not selected_gpu_ids:
            raise RuntimeError(
                "No worker GPU IDs were selected or detected. Re-run with "
                "--worker-gpu-ids, or use --no-workers to install app-only launchers."
            )

        # Install without login autostart. The launcher starts these fixed
        # systemd units on app launch; repeated starts are idempotent, so two UI
        # windows do not create duplicate worker processes.
        from mn_ligand.core.worker_service import install_worker_service

        install_worker_service(selected_gpu_ids, start=False, enable=False)

    selected_bin = Path(
        bin_dir or Path.home() / ".local" / "bin"
    ).expanduser().resolve()
    cli_wrapper = selected_bin / "mn-ligand"
    app_wrapper = selected_bin / "mn-ligand-app"
    quoted_cli = shlex.quote(str(cli))
    _atomic_text(
        cli_wrapper,
        f"#!/bin/sh\nset -eu\nexec {quoted_cli} \"$@\"\n",
        mode=0o755,
    )
    _atomic_text(
        app_wrapper,
        "#!/bin/sh\nset -eu\n"
        + (
            f"if ! {quoted_cli} worker-service start --gpu-ids "
            f"{shlex.quote(','.join(str(value) for value in selected_gpu_ids))}\n"
            "then\n"
            '  echo "Warning: mn-ligand workers did not start; queued jobs may wait." >&2\n'
            "fi\n"
            if install_workers
            else ""
        )
        + f"exec {quoted_cli} app \"$@\"\n",
        mode=0o755,
    )

    desktop_path: Path | None = None
    if create_desktop:
        selected_desktop = Path(
            desktop_dir or Path.home() / "Desktop"
        ).expanduser().resolve()
        desktop_path = selected_desktop / "mn-ligand.desktop"
        _atomic_text(
            desktop_path,
            "\n".join(
                (
                    "[Desktop Entry]",
                    "Type=Application",
                    "Version=1.0",
                    "Name=MN Ligand",
                    "Comment=Start the MN Ligand modelling application",
                    f"Exec={app_wrapper}",
                    "Icon=applications-science",
                    "Terminal=true",
                    "Categories=Science;Education;",
                    "StartupNotify=true",
                    "",
                )
            ),
            mode=0o755,
        )
        gio = shutil.which("gio")
        if gio:
            subprocess.run(
                [gio, "set", str(desktop_path), "metadata::trusted", "true"],
                capture_output=True,
                check=False,
                text=True,
            )

    return {
        "environment_cli": str(cli),
        "cli_wrapper": str(cli_wrapper),
        "app_wrapper": str(app_wrapper),
        "desktop_launcher": str(desktop_path) if desktop_path else "",
        "worker_gpu_ids": list(selected_gpu_ids),
        "worker_services_installed": install_workers,
    }
