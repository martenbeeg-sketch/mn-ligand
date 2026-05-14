from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import typer


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_APP_HOME = PROJECT_DIR / "mn-ligand-workdir"
DEFAULT_TMPDIR = PROJECT_DIR / ".tmp"

app = typer.Typer(
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)


@app.callback()
def cli() -> None:
    """Command line helpers for the standalone mn-ligand app."""


@app.command(name="app", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run_app(
    ctx: typer.Context,
    app_home: str = typer.Option(
        str(DEFAULT_APP_HOME),
        "--app-home",
        help="Runtime directory for mn-ligand jobs and local app state.",
    ),
    tmpdir: str = typer.Option(
        str(DEFAULT_TMPDIR),
        "--tmpdir",
        help="Writable temporary directory for Streamlit startup.",
    ),
):
    """Run the ligand-only Streamlit app."""
    home_path = _ensure_runtime_home(app_home, tmpdir)
    os.environ.setdefault("TMPDIR", tmpdir)
    os.environ.setdefault("MN_LIGAND_APP_HOME", str(home_path))
    os.environ.setdefault("MN_LIGAND_RUN_DIR", str(home_path / "workdir" / "runs"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    streamlit_script_path = Path(__file__).resolve().parent / "run_app.py"
    sys.argv = ["streamlit", "run", str(streamlit_script_path)]
    sys.argv.extend(["--browser.gatherUsageStats", "0"])
    sys.argv.extend(["--server.showEmailPrompt", "0"])
    sys.argv.extend(["--logger.enableRich", "0"])
    sys.argv.extend(ctx.args)
    runpy.run_module("streamlit", run_name="__main__")


@app.command(name="init")
def init_home(
    app_home: str = typer.Option(
        str(DEFAULT_APP_HOME),
        "--app-home",
        help="Dedicated runtime directory for mn-ligand.",
    ),
    tmpdir: str = typer.Option(
        str(DEFAULT_TMPDIR),
        "--tmpdir",
        help="Writable temporary directory for Streamlit startup.",
    ),
):
    """Initialize a dedicated runtime directory for mn-ligand."""
    home_path = _ensure_runtime_home(app_home, tmpdir)
    typer.echo(f"Initialized mn-ligand runtime directory: {home_path}")


def _ensure_runtime_home(app_home: str, tmpdir: str) -> Path:
    home_path = Path(app_home).expanduser().resolve()
    tmp_path = Path(tmpdir).expanduser().resolve()

    home_path.mkdir(parents=True, exist_ok=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (home_path / "storage").mkdir(exist_ok=True)
    (home_path / "reference_files").mkdir(exist_ok=True)
    (home_path / "workdir" / "runs").mkdir(parents=True, exist_ok=True)
    return home_path


def main() -> None:
    app()


if __name__ == "__main__":
    main()
