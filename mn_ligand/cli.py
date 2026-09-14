from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

import typer

from mn_ligand.runtime import DEFAULT_APP_HOME, DEFAULT_TMPDIR, ensure_runtime_home

app = typer.Typer(
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)
worker_service_app = typer.Typer(
    help="Install and manage supervised per-GPU worker services.",
    no_args_is_help=True,
)
app.add_typer(worker_service_app, name="worker-service")


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
    home_path = ensure_runtime_home(app_home, tmpdir)
    os.environ.setdefault("TMPDIR", tmpdir)
    os.environ.setdefault("MN_LIGAND_APP_HOME", str(home_path))
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
    home_path = ensure_runtime_home(app_home, tmpdir)
    typer.echo(f"Initialized mn-ligand runtime directory: {home_path}")


@app.command(name="doctor")
def doctor(
    images: bool = typer.Option(
        True,
        "--images/--skip-images",
        help="Check availability of every image in the tool registry.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Check runtime paths, Docker, GPUs, images, and reference resources."""
    from mn_ligand.core.diagnostics import diagnostics_exit_code, run_diagnostics

    results = run_diagnostics(include_images=images)
    if json_output:
        typer.echo(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        for result in results:
            typer.echo(f"{result.status.upper():4}  {result.label}: {result.message}")
    raise typer.Exit(code=diagnostics_exit_code(results))


@app.command(name="worker")
def worker(
    once: bool = typer.Option(False, "--once", help="Process at most one runnable job and exit."),
    poll_seconds: float = typer.Option(2.0, "--poll-seconds", min=0.1, help="Idle polling interval."),
    gpu_ids: str = typer.Option(
        "",
        "--gpu-ids",
        help="Comma-separated GPU IDs available to this worker; defaults to local discovery.",
    ),
    worker_id: str = typer.Option("", "--worker-id", help="Stable worker identity for logs."),
    job_class: str = typer.Option(
        "mixed",
        "--job-class",
        help="Accepted jobs: cpu, gpu, or mixed.",
    ),
) -> None:
    """Run the durable local job worker."""
    from mn_ligand.core.worker import WorkerConfig, run_worker_once, serve_worker

    selected: tuple[int, ...] | None = None
    if gpu_ids.strip():
        values = tuple(part.strip() for part in gpu_ids.split(",") if part.strip())
        if not values or any(not value.isdigit() for value in values):
            raise typer.BadParameter("GPU IDs must be comma-separated non-negative integers", param_hint="--gpu-ids")
        selected = tuple(dict.fromkeys(int(value) for value in values))
    normalized_job_class = job_class.strip().lower()
    if normalized_job_class not in {"cpu", "gpu", "mixed"}:
        raise typer.BadParameter(
            "Job class must be cpu, gpu, or mixed", param_hint="--job-class"
        )
    config = WorkerConfig.create(
        gpu_ids=selected,
        worker_id=worker_id.strip(),
        job_class=normalized_job_class,
    )
    typer.echo(
        f"Worker {config.worker_id}; class: {config.job_class}; "
        f"GPUs: {list(config.gpu_ids) or 'none'}"
    )
    if once:
        result = run_worker_once(config)
        typer.echo(json.dumps(result or {"status": "idle"}, indent=2))
        return
    try:
        serve_worker(config, poll_seconds=poll_seconds)
    except KeyboardInterrupt:
        typer.echo("Worker stopped.")


def _service_gpu_ids(value: str) -> tuple[int, ...]:
    from mn_ligand.core.worker_service import parse_gpu_ids

    try:
        return parse_gpu_ids(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--gpu-ids") from exc


@worker_service_app.command(name="install")
def worker_service_install(
    gpu_ids: str = typer.Option("0,1", "--gpu-ids", help="GPU worker instances to install."),
    start: bool = typer.Option(True, "--start/--no-start", help="Start services after enabling them."),
) -> None:
    """Install and enable the systemd user-service template."""
    from mn_ligand.core.worker_service import WorkerServiceError, install_worker_service

    selected = _service_gpu_ids(gpu_ids)
    try:
        path = install_worker_service(selected, start=start)
    except WorkerServiceError as exc:
        typer.echo(f"Worker service installation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Installed {path}")
    typer.echo(
        f"Enabled CPU worker and GPU workers: {list(selected)}"
        + (" (started)" if start else "")
    )


def _manage_worker_service(action: str, gpu_ids: str) -> None:
    from mn_ligand.core.worker_service import WorkerServiceError, manage_worker_services

    selected = _service_gpu_ids(gpu_ids)
    try:
        result = manage_worker_services(action, selected, check=action != "status")
    except WorkerServiceError as exc:
        typer.echo(f"Worker service {action} failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if result.stdout.strip():
        typer.echo(result.stdout.rstrip())
    if result.stderr.strip():
        typer.echo(result.stderr.rstrip(), err=True)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
    if action != "status":
        completed_action = {"start": "Started", "stop": "Stopped", "restart": "Restarted"}[action]
        typer.echo(f"{completed_action} CPU worker and GPU workers: {list(selected)}")


@worker_service_app.command(name="start")
def worker_service_start(
    gpu_ids: str = typer.Option("0,1", "--gpu-ids", help="GPU worker instances to start."),
) -> None:
    """Start installed worker instances."""
    _manage_worker_service("start", gpu_ids)


@worker_service_app.command(name="stop")
def worker_service_stop(
    gpu_ids: str = typer.Option("0,1", "--gpu-ids", help="GPU worker instances to stop."),
) -> None:
    """Stop worker instances without disabling them."""
    _manage_worker_service("stop", gpu_ids)


@worker_service_app.command(name="restart")
def worker_service_restart(
    gpu_ids: str = typer.Option("0,1", "--gpu-ids", help="GPU worker instances to restart."),
) -> None:
    """Restart worker instances."""
    _manage_worker_service("restart", gpu_ids)


@worker_service_app.command(name="status")
def worker_service_status(
    gpu_ids: str = typer.Option("0,1", "--gpu-ids", help="GPU worker instances to inspect."),
) -> None:
    """Show full systemd status for worker instances."""
    _manage_worker_service("status", gpu_ids)


@worker_service_app.command(name="uninstall")
def worker_service_uninstall(
    gpu_ids: str = typer.Option("0,1", "--gpu-ids", help="GPU worker instances to remove."),
) -> None:
    """Stop, disable, and remove the worker service template."""
    from mn_ligand.core.worker_service import WorkerServiceError, uninstall_worker_service

    selected = _service_gpu_ids(gpu_ids)
    try:
        path = uninstall_worker_service(selected)
    except WorkerServiceError as exc:
        typer.echo(f"Worker service uninstall failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Removed {path}")


@app.command(name="backfill-target-metadata")
@app.command(name="backfill-ligands", hidden=True)
def backfill_ligands(
    apply: bool = typer.Option(False, "--apply", help="Write recovered ligand metadata to existing jobs."),
    fetch_rcsb: bool = typer.Option(
        True,
        "--fetch-rcsb/--no-fetch-rcsb",
        help="Enrich receptor and Chemical Component metadata with the RCSB Data API.",
    ),
) -> None:
    """Recover target identity and derived-job modification provenance."""
    from mn_ligand.workflows.protein_preparation import backfill_ligand_metadata
    from mn_ligand.workflows.provenance_backfill import (
        backfill_derived_target_provenance,
    )

    result = backfill_ligand_metadata(fetch_rcsb=fetch_rcsb, write=apply)
    derived = backfill_derived_target_provenance(write=apply)
    action = "Updated" if apply else "Would update"
    typer.echo(f"Scanned {result['scanned']} target jobs. {action} {result['updated']} jobs.")
    for item in result["jobs"]:
        identities = ", ".join(
            str(ligand.get("ccd_id") or ligand.get("name") or ligand.get("resname") or "unknown")
            for ligand in item["ligands"]
        )
        receptor_names = ", ".join(
            str(entity.get("name") or "")
            for entity in item.get("receptor", {}).get("entities", [])
            if entity.get("name")
        )
        summary = " | ".join(value for value in (receptor_names, identities) if value)
        typer.echo(f"{item['task_group']}/{item['run_id']}: {summary or 'metadata recovered'}")
    typer.echo(
        f"Scanned {derived['scanned']} derived target jobs. "
        f"{action} {derived['updated']} provenance records."
    )
    for item in derived["jobs"]:
        typer.echo(
            f"{item['task_group']}/{item['run_id']}: "
            + ", ".join(item["fields"])
        )
    if not apply and (result["updated"] or derived["updated"]):
        typer.echo("Run again with --apply to write these metadata updates.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
