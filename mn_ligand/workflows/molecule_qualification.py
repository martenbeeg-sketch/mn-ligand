from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    registered_tool,
    write_registered_command_record,
)
from mn_ligand.core.jobs import (
    JOB_SCHEMA_VERSION,
    JobRecord,
    iter_job_records,
    short_job_code,
)
from mn_ligand.runtime import cpu_process_limit, runs_root


MOLECULE_QUALIFICATION_TASK_GROUP = "molecule-qualification"
QUALIFICATION_POLICY_VERSION = 2
DEFAULT_IMAGE = "ovolig-posebusters:latest"
DEFAULT_POLICY: dict[str, int | float] = {
    "conformer_count": 20,
    "max_workers": 16,
    "min_heavy_atoms": 5,
    "max_heavy_atoms": 80,
    "max_absolute_charge": 2,
    "max_sa_score": 6.0,
}


def _record_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def generation_qualification_jobs(source_run_id: str) -> list[JobRecord]:
    return [
        job
        for job in iter_job_records(runs_root())
        if job.task_group == MOLECULE_QUALIFICATION_TASK_GROUP
        and job.workflow == "molecule_qualification"
        and job.parent_run_id == source_run_id
    ]


def queue_molecule_qualification_job(
    source_job: JobRecord,
    *,
    image: str = DEFAULT_IMAGE,
    policy: dict[str, int | float] | None = None,
) -> JobRecord:
    if source_job.status != "completed":
        raise ValueError("Molecule qualification requires a completed generation job")
    if source_job.workflow != "molecule_generation":
        raise ValueError("Molecule qualification requires a molecule-generation source")
    existing = generation_qualification_jobs(source_job.run_id)
    current = [
        job
        for job in existing
        if int(job.metadata.get("qualification_policy_version") or 0)
        == QUALIFICATION_POLICY_VERSION
    ]
    if current:
        return sorted(
            current,
            key=lambda job: str(job.created_at or ""),
        )[-1]

    source_sdf = source_job.run_dir / "normalized" / "generated_compounds.sdf"
    source_table = source_job.run_dir / "normalized" / "generated_compounds.csv"
    if not source_sdf.is_file() or not source_table.is_file():
        raise FileNotFoundError(
            "The generation job has no normalized SDF/CSV handshake"
        )

    parameters = {**DEFAULT_POLICY, **dict(policy or {})}
    input_count = max(1, _record_count(source_table))
    max_workers = max(
        1,
        min(cpu_process_limit(), input_count, int(parameters["max_workers"])),
    )
    run_id = str(uuid4())
    run_dir = runs_root() / MOLECULE_QUALIFICATION_TASK_GROUP / run_id
    input_dir = run_dir / "input"
    qualified_dir = run_dir / "qualified"
    native_dir = run_dir / "native"
    input_dir.mkdir(parents=True, exist_ok=False)
    qualified_dir.mkdir()
    native_dir.mkdir()
    shutil.copy2(source_sdf, input_dir / "generated_compounds.sdf")
    shutil.copy2(source_table, input_dir / "generated_compounds.csv")

    seed = int(source_job.metadata.get("seed") or 2026)
    command = build_docker_command(
        DockerRunSpec(
            tool=registered_tool("posebusters", image=image),
            command=(
                "--mode",
                "qualify-molecules",
                "--molecule-input",
                "/workspace/input/generated_compounds.sdf",
                "--molecule-table",
                "/workspace/input/generated_compounds.csv",
                "--qualification-output",
                "/workspace/qualified",
                "--seed",
                str(seed),
                "--conformer-count",
                str(max(1, int(parameters["conformer_count"]))),
                "--max-workers",
                str(max_workers),
                "--min-heavy-atoms",
                str(max(1, int(parameters["min_heavy_atoms"]))),
                "--max-heavy-atoms",
                str(max(1, int(parameters["max_heavy_atoms"]))),
                "--max-absolute-charge",
                str(max(0, int(parameters["max_absolute_charge"]))),
                "--max-sa-score",
                str(float(parameters["max_sa_score"])),
            ),
            mounts=(DockerMount(run_dir, "/workspace"),),
            gpu_enabled=False,
            use_host_user=True,
        )
    )
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "molecule_qualification",
        "workflow": "molecule_qualification",
        "operation": "chemical_and_3d_qualification",
        "status": "queued",
        "tool": "RDKit + PoseBusters",
        "parent_run_id": source_job.run_id,
        "source_task_group": source_job.task_group,
        "source_engine": source_job.tool,
        "source_engine_id": source_job.metadata.get("engine_id", ""),
        "seed": seed,
        "qualification_policy": parameters,
        "qualification_policy_version": QUALIFICATION_POLICY_VERSION,
        "input_compound_count": input_count,
        "effective_max_workers": max_workers,
        "docker_image": image,
        "created_at": now,
        "updated_at": now,
        "queued_at": now,
        "queued_command": command,
        "gpu_queued": False,
        "resources": {
            **registered_tool("posebusters", image=image).resources.to_dict(),
            "gpu": False,
            "cpu_threads": max_workers,
        },
        "worker_finalizer": "molecule_qualification",
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "source_job": {
                "run_id": source_job.run_id,
                "task_group": source_job.task_group,
                "workflow": source_job.workflow,
                "engine": source_job.tool,
            },
            "source_artifacts": {
                "sdf": "input/generated_compounds.sdf",
                "table": "input/generated_compounds.csv",
            },
            "identity_source": "canonical stereochemistry-aware SMILES",
            "geometry_source": "deterministic RDKit ETKDGv3",
            "parameters": {
                **parameters,
                "max_workers": max_workers,
                "seed": seed,
                "posebusters_config": "mol",
            },
        },
    )
    write_registered_command_record(
        run_dir,
        tool_id="posebusters",
        commands=(command,),
        image=image,
    )
    write_artifact_manifest(run_dir, [])
    return JobRecord.load(
        run_dir,
        task_group=MOLECULE_QUALIFICATION_TASK_GROUP,
    )


def finalize_molecule_qualification_job(
    run_dir: Path,
    *,
    returncode: int,
) -> JobRecord:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    qualified_dir = run_dir / "qualified"
    report_path = qualified_dir / "qualification_report.json"
    table_path = qualified_dir / "qualification.csv"
    qualified_path = qualified_dir / "qualified_compounds.sdf"
    try:
        report = json.loads(report_path.read_text())
    except (OSError, TypeError, ValueError):
        report = {}
    success = bool(
        returncode == 0
        and report.get("success")
        and table_path.is_file()
        and int(report.get("input_count") or 0) > 0
    )
    qualified_count = int(report.get("qualified_compound_count") or 0)
    warning_count = int(report.get("qualified_with_warning_count") or 0)
    artifacts: list[ArtifactRef] = []
    if qualified_count > 0 and qualified_path.is_file() and qualified_path.stat().st_size:
        artifacts.extend(
            [
                ArtifactRef.from_path(
                    run_dir,
                    qualified_path,
                    "chemically_qualified_molecule_set",
                    role="smiles_standardized_posebusters_validated_3d",
                    metadata={
                        "compound_count": qualified_count,
                        "geometry_source": "rdkit_etkdgv3",
                        "qualified_with_warning_count": warning_count,
                    },
                ),
                ArtifactRef.from_path(
                    run_dir,
                    qualified_path,
                    "compound_set",
                    role="downstream_compound_handoff",
                    metadata={
                        "compound_count": qualified_count,
                        "qualification_required": True,
                        "qualified_with_warning_count": warning_count,
                    },
                ),
            ]
        )
    for path, artifact_type, role in (
        (
            table_path,
            "generation_qualification_table",
            "per_compound_quality",
        ),
        (
            report_path,
            "generation_qualification_report",
            "quality_policy_and_counts",
        ),
        (
            qualified_dir / "posebusters_full.csv",
            "molecule_geometry_validation_metrics",
            "posebusters_mol_full_report",
        ),
    ):
        if path.is_file() and path.stat().st_size:
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir,
                    path,
                    artifact_type,
                    role=role,
                )
            )
    for name, role in (("stdout.log", "stdout"), ("stderr.log", "stderr")):
        path = run_dir / name
        path.touch(exist_ok=True)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                path,
                "job_log",
                role=role,
                checksum=False,
            )
        )
    error = "" if success else (
        (run_dir / "stderr.log").read_text(errors="replace")[-4000:]
        or "Molecule qualification did not produce a complete report"
    )
    result = {
        **report,
        "success": success,
        "returncode": int(returncode),
        "qualified_compound_count": qualified_count,
        "error": error,
    }
    _write_json(run_dir / "result.json", result)
    write_artifact_manifest(run_dir, artifacts)
    now = _utc_now_iso()
    metadata.update(
        {
            "status": "completed" if success else "failed",
            "updated_at": now,
            "completed_at": now,
            "qualified_compound_count": qualified_count,
            "qualified_with_warning_count": warning_count,
        }
    )
    if error:
        metadata["error"] = error
    _write_json(metadata_path, metadata)
    return JobRecord.load(
        run_dir,
        task_group=MOLECULE_QUALIFICATION_TASK_GROUP,
    )


def queue_missing_generation_qualifications() -> list[JobRecord]:
    queued: list[JobRecord] = []
    sources = [
        job
        for job in iter_job_records(runs_root())
        if job.task_group == "molecule-generation"
        and job.workflow == "molecule_generation"
        and job.status == "completed"
    ]
    for source in sources:
        if any(
            int(job.metadata.get("qualification_policy_version") or 0)
            == QUALIFICATION_POLICY_VERSION
            for job in generation_qualification_jobs(source.run_id)
        ):
            continue
        try:
            queued.append(queue_molecule_qualification_job(source))
        except (FileNotFoundError, ValueError):
            continue
    return queued
