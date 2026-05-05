from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from textwrap import dedent

import typer


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OVO_HOME = PROJECT_DIR / ".ovo-home"
DEFAULT_TMPDIR = PROJECT_DIR / ".tmp"

app = typer.Typer(
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)


@app.callback()
def cli() -> None:
    """Command line helpers for the ovo-ligand OVO plugin."""


@app.command(name="app", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run_app(
    ctx: typer.Context,
    ovo_home: str = typer.Option(
        str(DEFAULT_OVO_HOME),
        "--ovo-home",
        help="OVO home directory to use for the Streamlit app.",
    ),
    tmpdir: str = typer.Option(
        str(DEFAULT_TMPDIR),
        "--tmpdir",
        help="Writable temporary directory for OVO/Streamlit startup.",
    ),
):
    """Run the ligand-only Streamlit app."""
    os.environ.setdefault("OVO_HOME", ovo_home)
    os.environ.setdefault("TMPDIR", tmpdir)
    os.environ.setdefault("OVO_LIGAND_RUN_DIR", str(Path(ovo_home).expanduser().resolve() / "workdir" / "runs"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    if not Path(os.environ["OVO_HOME"]).joinpath("config.yml").exists():
        raise typer.BadParameter(
            f"OVO home is not initialized: {os.environ['OVO_HOME']}\n"
            "Run `ovo-ligand init` first, or pass --ovo-home to an existing OVO home."
        )

    streamlit_script_path = Path(__file__).resolve().parent / "run_app.py"
    sys.argv = ["streamlit", "run", str(streamlit_script_path)]
    sys.argv.extend(["--browser.gatherUsageStats", "0"])
    sys.argv.extend(["--server.showEmailPrompt", "0"])
    sys.argv.extend(["--logger.enableRich", "0"])
    sys.argv.extend(ctx.args)
    runpy.run_module("streamlit", run_name="__main__")


@app.command(name="init")
def init_home(
    ovo_home: str = typer.Option(
        str(DEFAULT_OVO_HOME),
        "--ovo-home",
        help="Dedicated OVO home directory for ovo-ligand.",
    ),
    tmpdir: str = typer.Option(
        str(DEFAULT_TMPDIR),
        "--tmpdir",
        help="Writable temporary directory for OVO/Streamlit startup.",
    ),
):
    """Initialize a dedicated OVO home for ovo-ligand."""
    home_path = Path(ovo_home).expanduser().resolve()
    tmp_path = Path(tmpdir).expanduser().resolve()
    config_path = home_path / "config.yml"

    if config_path.exists():
        typer.echo(f"OVO-ligand home already initialized: {home_path}")
        return

    home_path.mkdir(parents=True, exist_ok=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (home_path / "storage").mkdir(exist_ok=True)
    (home_path / "reference_files").mkdir(exist_ok=True)
    (home_path / "workdir").mkdir(exist_ok=True)
    (home_path / "nextflow").mkdir(exist_ok=True)

    config_path.write_text(_default_config(), encoding="utf-8")
    (home_path / "nextflow_local.config").write_text(_default_nextflow_config(), encoding="utf-8")

    typer.echo(f"Initialized OVO-ligand home: {home_path}")


def _default_config() -> str:
    return dedent(
        """
        db:
          url: sqlite:///ovo.db
          verbose: false
        reference_files_dir: ./reference_files
        nextflow_home: ./nextflow
        auth:
          admin_users: []
          allow_private_project_link_access: true
          hide_admin_warning: true
        storage:
          verbose: false
          path: ./storage
        props:
          check_new_version: false
          pyrosetta_license: false
          read_only: false
          rfdiffusion_backbones_limit: 1000
          rfdiffusion_backbones_limit_admin: 5000
          mpnn_sequences_limit: 100
          allowed_attachment_types:
            - image
            - text
            - docx
            - csv
            - xlsx
            - json
            - pdb
            - cif
            - mol
            - mol2
            - sdf
            - pdbqt
            - gro
            - xtc
            - zip
            - gz
            - tar
            - fa
            - fasta
            - tsv
            - html
        default_scheduler: local_docker
        local_scheduler: local_docker
        schedulers:
          local_docker:
            name: Local Docker
            type: NextflowScheduler
            submission_args:
              profile: docker,cpu_env
              max_memory: 8GB
              config: ./nextflow_local.config
            workdir: ./workdir
        plugins:
          ovo_ligand: {}
        """
    ).lstrip()


def _default_nextflow_config() -> str:
    return dedent(
        """
        process {
            executor = "local"
            beforeScript = ""
            queue = null
        }
        """
    ).lstrip()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
