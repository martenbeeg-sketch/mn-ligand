from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

import typer

from mn_ligand.runtime import (
    app_home as configured_app_home,
    ensure_runtime_home,
    library_root,
    reference_root,
    required_mount,
    runs_root,
    save_installation_settings,
    save_runtime_settings,
    temporary_root,
    validate_required_mount,
)

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
portability_app = typer.Typer(
    help="Audit, export, and verify runtime portability without modifying source data.",
    no_args_is_help=True,
)
app.add_typer(portability_app, name="portability")


@app.callback()
def cli() -> None:
    """Command line helpers for the standalone mn-ligand app."""


@app.command(name="app", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run_app(
    ctx: typer.Context,
    app_home: str = typer.Option(
        "",
        "--app-home",
        help="Runtime directory for mn-ligand jobs and local app state.",
    ),
    tmpdir: str = typer.Option(
        "",
        "--tmpdir",
        help="Writable temporary directory for Streamlit startup.",
    ),
):
    """Run the ligand-only Streamlit app."""
    if app_home.strip():
        os.environ["MN_LIGAND_APP_HOME"] = str(Path(app_home).expanduser().resolve())
    if tmpdir.strip():
        os.environ["MN_LIGAND_TMP_DIR"] = str(Path(tmpdir).expanduser().resolve())
    try:
        validate_required_mount()
    except RuntimeError as exc:
        typer.echo(f"App startup refused: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    selected_home = configured_app_home()
    selected_tmp = temporary_root()
    home_path = ensure_runtime_home(selected_home, selected_tmp)
    os.environ.setdefault("TMPDIR", str(selected_tmp))
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
        "",
        "--app-home",
        help="Dedicated runtime directory for mn-ligand.",
    ),
    tmpdir: str = typer.Option(
        "",
        "--tmpdir",
        help="Writable temporary directory for Streamlit startup.",
    ),
    runs_dir: str = typer.Option(
        "", "--runs-dir", help="Results and durable job directory."
    ),
    reference_dir: str = typer.Option(
        "", "--reference-dir", help="Reference data directory."
    ),
    library_dir: str = typer.Option(
        "", "--library-dir", help="Compound library directory."
    ),
    required_mount_path: str = typer.Option(
        "",
        "--require-mount",
        help="Refuse startup unless this data filesystem is mounted.",
    ),
    no_mount_guard: bool = typer.Option(
        False,
        "--no-mount-guard",
        help="Remove an existing required-mount guard during reconfiguration.",
    ),
):
    """Create and persist a complete machine-local mn-ligand installation."""
    selected_home = (
        Path(app_home).expanduser().resolve()
        if app_home.strip()
        else configured_app_home()
    )
    os.environ["MN_LIGAND_APP_HOME"] = str(selected_home)
    selected_tmp = (
        Path(tmpdir).expanduser().resolve()
        if tmpdir.strip()
        else temporary_root()
    )
    selected_runs = (
        Path(runs_dir).expanduser().resolve()
        if runs_dir.strip()
        else runs_root(create=False)
    )
    selected_references = (
        Path(reference_dir).expanduser().resolve()
        if reference_dir.strip()
        else reference_root(create=False)
    )
    selected_libraries = (
        Path(library_dir).expanduser().resolve()
        if library_dir.strip()
        else library_root(create=False)
    )
    if no_mount_guard and required_mount_path.strip():
        raise typer.BadParameter(
            "Use either --require-mount or --no-mount-guard, not both"
        )
    selected_mount = (
        None
        if no_mount_guard
        else (
            Path(required_mount_path).expanduser().resolve()
            if required_mount_path.strip()
            else required_mount()
        )
    )
    if selected_mount is not None:
        os.environ["MN_LIGAND_REQUIRED_MOUNT"] = str(selected_mount)
    else:
        os.environ["MN_LIGAND_REQUIRED_MOUNT"] = ""
    try:
        validate_required_mount()
        home_path = ensure_runtime_home(selected_home, selected_tmp)
        install_target = save_installation_settings(
            runtime_home=home_path,
            required_mount_path=selected_mount,
        )
        runtime_target = save_runtime_settings(
            runs_dir=selected_runs,
            reference_dir=selected_references,
            library_dir=selected_libraries,
            tmp_dir=selected_tmp,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Initialization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Installation configuration: {install_target}")
    typer.echo(f"Runtime configuration: {runtime_target}")
    typer.echo(f"App home: {home_path}")
    typer.echo(f"Results and jobs: {selected_runs}")
    typer.echo(f"Reference files: {selected_references}")
    typer.echo(f"Libraries: {selected_libraries}")
    typer.echo(f"Temporary files: {selected_tmp}")
    typer.echo(f"Required mount: {required_mount() or 'none'}")


@app.command(name="install-launchers")
def install_launchers(
    desktop: bool = typer.Option(
        True,
        "--desktop/--no-desktop",
        help="Also create a desktop launcher.",
    ),
    bin_dir: Path | None = typer.Option(
        None,
        "--bin-dir",
        help="Wrapper destination; defaults to ~/.local/bin.",
    ),
    desktop_dir: Path | None = typer.Option(
        None,
        "--desktop-dir",
        help="Desktop destination; defaults to ~/Desktop.",
    ),
    workers: bool = typer.Option(
        True,
        "--workers/--no-workers",
        help="Install shared CPU/GPU worker services and start them with the app launcher.",
    ),
    worker_gpu_ids: str = typer.Option(
        "",
        "--worker-gpu-ids",
        help="GPU IDs for worker services, comma-separated; default detects local GPUs.",
    ),
) -> None:
    """Install command/desktop launchers and optionally shared worker services."""
    from mn_ligand.launchers import install_user_launchers

    selected_gpu_ids: tuple[int, ...] | None = None
    if worker_gpu_ids.strip():
        try:
            parts = tuple(
                part.strip()
                for part in worker_gpu_ids.split(",")
                if part.strip()
            )
            if not parts or any(not part.isdigit() for part in parts):
                raise ValueError("GPU IDs must be comma-separated non-negative integers")
            selected_gpu_ids = tuple(dict.fromkeys(int(part) for part in parts))
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--worker-gpu-ids") from exc

    try:
        installed = install_user_launchers(
            bin_dir=bin_dir,
            desktop_dir=desktop_dir,
            create_desktop=desktop,
            worker_gpu_ids=selected_gpu_ids,
            install_workers=workers,
        )
    except (OSError, RuntimeError) as exc:
        typer.echo(f"Launcher installation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Command launcher: {installed['cli_wrapper']}")
    typer.echo(f"One-command app launcher: {installed['app_wrapper']}")
    if installed["desktop_launcher"]:
        typer.echo(f"Desktop launcher: {installed['desktop_launcher']}")
    if installed["worker_services_installed"]:
        typer.echo(
            "Worker services installed for GPU(s): "
            + ",".join(str(value) for value in installed["worker_gpu_ids"])
            + "; they start when the app launcher opens (not at user login)."
        )
    else:
        typer.echo("Worker services were not configured by this launcher install.")
    typer.echo(f"Start the app with: {installed['app_wrapper']}")


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


@portability_app.command(name="audit")
def portability_audit(
    runs_dir: Path | None = typer.Option(
        None,
        "--runs-dir",
        help="Runs directory to inspect; defaults to the configured runtime.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the full JSON report."),
) -> None:
    """Read JSON metadata and classify stored absolute paths without changing it."""
    from mn_ligand.core.portability import audit_runtime_portability

    report = audit_runtime_portability(runs_dir)
    if json_output:
        typer.echo(json.dumps(report, indent=2))
        return
    counts = report["counts"]
    typer.echo(f"Runs root: {report['runs_root']}")
    typer.echo("Mode: read-only (no runtime files changed)")
    for key in (
        "json_files",
        "unreadable_json_files",
        "absolute_values",
        "current_paths",
        "relocatable_paths",
        "reference_paths",
        "container_paths",
        "provenance_paths",
        "embedded_text_paths",
        "unresolved_paths",
        "external_paths",
    ):
        typer.echo(f"{key.replace('_', ' ').title()}: {counts.get(key, 0)}")


@portability_app.command(name="export")
def portability_export(
    destination: Path = typer.Argument(
        ...,
        help="New destination directory for the portable bundle.",
    ),
    runs_dir: Path | None = typer.Option(
        None,
        "--runs-dir",
        help="Source runs directory; defaults to the configured runtime.",
    ),
    reference_dir: Path | None = typer.Option(
        None,
        "--reference-dir",
        help="Source reference directory; defaults to the configured runtime.",
    ),
    library_dir: Path | None = typer.Option(
        None,
        "--library-dir",
        help="Source compound-library directory; defaults to the configured runtime.",
    ),
    app_home: Path | None = typer.Option(
        None,
        "--app-home",
        help="Source app home; defaults to the configured runtime.",
    ),
    include_references: bool = typer.Option(
        True,
        "--include-references/--skip-references",
        help="Copy reference data into the portable bundle.",
    ),
    include_libraries: bool = typer.Option(
        True,
        "--include-libraries/--skip-libraries",
        help="Copy compound libraries into the portable bundle.",
    ),
) -> None:
    """Create a validated portable copy; never modify the source archive."""
    from mn_ligand.core.portability import PortabilityError, export_portable_runtime

    try:
        result = export_portable_runtime(
            destination,
            source_runs=runs_dir,
            source_references=reference_dir,
            source_libraries=library_dir,
            source_app_home=app_home or configured_app_home(),
            include_references=include_references,
            include_libraries=include_libraries,
        )
    except (OSError, PortabilityError, ValueError) as exc:
        typer.echo(f"Portable export failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    counts = result["counts"]
    typer.echo(f"Portable export created: {result['destination']}")
    typer.echo(f"Source JSON files checked: {counts['source_json_files']}")
    typer.echo(f"Operational paths rewritten: {counts['rewritten_paths']}")
    typer.echo("Verification: passed")


@portability_app.command(name="verify")
def portability_verify(
    destination: Path = typer.Argument(..., help="Portable bundle to verify."),
    json_output: bool = typer.Option(False, "--json", help="Emit the full JSON report."),
) -> None:
    """Verify paths, declared artifacts, and checksums in a portable bundle."""
    from mn_ligand.core.portability import verify_portable_export

    report = verify_portable_export(destination)
    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        typer.echo(f"Portable bundle: {report['destination']}")
        typer.echo(f"Status: {'valid' if report['valid'] else 'invalid'}")
        for key, value in report["counts"].items():
            typer.echo(f"{key.replace('_', ' ').title()}: {value}")
        for error in report["errors"][:20]:
            typer.echo(f"ERROR {error['file']}: {error['error']}", err=True)
    if not report["valid"]:
        raise typer.Exit(code=1)


@portability_app.command(name="check-job")
def portability_check_job(
    run_dir: Path = typer.Argument(..., help="Job run directory to validate."),
    json_output: bool = typer.Option(False, "--json", help="Emit the full JSON report."),
) -> None:
    """Fail when canonical job metadata contains machine-bound paths."""
    from mn_ligand.core.portability import validate_job_portability

    report = validate_job_portability(run_dir)
    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        typer.echo(f"Job: {report['run_dir']}")
        typer.echo(f"Status: {'portable' if report['valid'] else 'not portable'}")
        for error in report["errors"][:20]:
            typer.echo(
                f"ERROR {error['file']} {error.get('key') or '<value>'}: {error['error']}",
                err=True,
            )
    if not report["valid"]:
        raise typer.Exit(code=1)


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

    try:
        validate_required_mount()
    except RuntimeError as exc:
        typer.echo(f"Worker startup refused: {exc}", err=True)
        raise typer.Exit(code=1) from exc

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
        validate_required_mount()
        path = install_worker_service(selected, start=start)
    except (RuntimeError, WorkerServiceError) as exc:
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
