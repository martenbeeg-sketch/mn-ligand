from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable
from uuid import uuid4

from rdkit import Chem
import yaml

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
from mn_ligand.runtime import (
    NATIVE_THREAD_ENVIRONMENT,
    adaptive_cpu_workers,
    runs_root,
)
from mn_ligand.workflows.refolding import compounds_from_path
from mn_ligand.workflows.rescoring import source_pose_rows


POSE_VALIDATION_TASK_GROUP = "pose-validation"
DEFAULT_POSEBUSTERS_IMAGE = "ovolig-posebusters:latest"
POSE_VALIDATION_INVENTORY_SCHEMA_VERSION = 2
POSE_VALIDATION_SELECTION_POLICY = "best-scientific-poses-v1"
COMPATIBLE_WORKFLOWS = frozenset(
    {
        "docking_campaign",
        "openvs_docking",
        "alphafold3_refolding",
        "boltz2_refolding",
    }
)
POSE_VALIDATION_SOURCE_TASK_GROUPS = ("docking", "refolding")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _safe_id(value: object, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (text or fallback)[:160]


def compatible_source_jobs(*, load_artifacts: bool = True) -> list[JobRecord]:
    return [
        job
        for job in iter_job_records(
            runs_root(),
            task_groups=POSE_VALIDATION_SOURCE_TASK_GROUPS,
            load_artifacts=load_artifacts,
            validate_artifacts=False,
        )
        if job.status == "completed" and job.workflow in COMPATIBLE_WORKFLOWS
        and not str(job.metadata.get("recovery_of_run_id") or "").strip()
    ]


def _excluded_compound_ids(source_job: JobRecord) -> set[str]:
    excluded: set[str] = set()
    for payload in (source_job.result, source_job.metadata):
        values = payload.get("excluded_compound_ids")
        if isinstance(values, list):
            excluded.update(
                str(value).strip() for value in values if str(value).strip()
            )
    path = source_job.run_dir / "excluded_compounds.tsv"
    if path.is_file():
        try:
            with path.open(newline="", errors="replace") as handle:
                excluded.update(
                    str(row.get("compound_id") or "").strip()
                    for row in csv.DictReader(handle, delimiter="\t")
                    if str(row.get("compound_id") or "").strip()
                )
        except (OSError, csv.Error):
            pass
    return excluded


def _source_receptor(source_job: JobRecord) -> Path:
    for candidate in (
        source_job.run_dir / "input" / "receptor.pdb",
        source_job.run_dir / "prepared" / "receptor.pdb",
    ):
        if candidate.is_file():
            return candidate
    try:
        payload = json.loads((source_job.run_dir / "input.json").read_text())
    except (OSError, TypeError, ValueError):
        payload = {}
    artifact_payload = payload.get("target_artifact") or payload.get("target")
    if isinstance(artifact_payload, dict):
        source = next(
            (
                job
                for job in iter_job_records(runs_root())
                if job.run_id == str(artifact_payload.get("run_id") or "")
            ),
            None,
        )
        if source is not None:
            try:
                artifact = ArtifactRef.from_dict(artifact_payload)
                resolved = artifact.resolve(source.run_dir, must_exist=True)
            except (TypeError, ValueError):
                resolved = None
            if resolved is not None:
                return resolved
    if source_job.artifact_manifest is not None:
        for artifact_type in (
            "prepared_receptor",
            "prepared_target",
            "prepared_complex",
        ):
            for artifact in source_job.artifact_manifest.by_type(artifact_type):
                resolved = artifact.resolve(source_job.run_dir, must_exist=True)
                if resolved is not None:
                    return resolved
    raise FileNotFoundError("The source result has no resolvable receptor")


def _source_smiles(source_job: JobRecord) -> dict[str, str]:
    native_inputs: dict[str, str] = {}
    inputs_dir = source_job.run_dir / "inputs"
    if source_job.workflow == "boltz2_refolding" and inputs_dir.is_dir():
        for path in sorted(inputs_dir.glob("*.yaml")):
            try:
                payload = yaml.safe_load(path.read_text()) or {}
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                continue
            for sequence in payload.get("sequences") or []:
                ligand = sequence.get("ligand") if isinstance(sequence, dict) else None
                if isinstance(ligand, dict) and ligand.get("smiles"):
                    native_inputs[path.stem] = str(ligand["smiles"])
                    break
    elif source_job.workflow == "alphafold3_refolding" and inputs_dir.is_dir():
        for path in sorted(inputs_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, TypeError, ValueError):
                continue
            candidate_id = str(payload.get("name") or path.stem)
            for sequence in payload.get("sequences") or []:
                ligand = sequence.get("ligand") if isinstance(sequence, dict) else None
                if isinstance(ligand, dict) and ligand.get("smiles"):
                    native_inputs[candidate_id] = str(ligand["smiles"])
                    break

    direct = source_job.run_dir / "input" / "compounds.tsv"
    if direct.is_file():
        with direct.open(newline="", errors="replace") as handle:
            fallback = {
                str(row.get("compound_id") or ""): str(row.get("smiles") or "")
                for row in csv.DictReader(handle, delimiter="\t")
                if row.get("compound_id") and row.get("smiles")
            }
        return {**fallback, **native_inputs}
    try:
        payload = json.loads((source_job.run_dir / "input.json").read_text())
    except (OSError, TypeError, ValueError):
        return native_inputs
    artifact_payloads = (
        payload.get("compound_sets")
        or payload.get("compound_artifacts")
        or []
    )
    if not isinstance(artifact_payloads, list):
        return native_inputs
    jobs = {job.run_id: job for job in iter_job_records(runs_root())}
    records: dict[str, str] = {}
    for item in artifact_payloads:
        if not isinstance(item, dict):
            continue
        source = jobs.get(str(item.get("run_id") or ""))
        if source is None:
            continue
        try:
            artifact = ArtifactRef.from_dict(item)
            path = artifact.resolve(source.run_dir, must_exist=True)
            if path is not None:
                records.update(dict(compounds_from_path(path)))
        except (OSError, TypeError, ValueError):
            continue
    return {**records, **native_inputs}


def _group_pose_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("compound_id") or ""),
            int(row.get("replicate") or 1),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def pose_validation_candidates(source_job: JobRecord) -> list[dict[str, Any]]:
    if source_job.workflow not in COMPATIBLE_WORKFLOWS:
        return []
    engine = str(source_job.tool or source_job.metadata.get("engine") or "")
    candidates: list[dict[str, Any]] = []
    if source_job.workflow == "docking_campaign":
        excluded_ids = _excluded_compound_ids(source_job)
        rows = [
            row
            for row in source_pose_rows(source_job)
            if str(row.get("compound_id") or "") not in excluded_ids
        ]
        selected: list[tuple[dict[str, Any], str]] = []
        for _, group_rows in _group_pose_rows(rows).items():
            if engine.strip().lower() == "gnina":
                cnn_rows = [
                    row
                    for row in group_rows
                    if row.get("source_cnn_score") is not None
                ]
                empirical_rows = [
                    row
                    for row in group_rows
                    if row.get("source_score_kcal_mol") is not None
                ]
                cnn = max(
                    cnn_rows or group_rows,
                    key=lambda row: float(
                        row.get("source_cnn_score")
                        if row.get("source_cnn_score") is not None
                        else float("-inf")
                    ),
                )
                empirical = min(
                    empirical_rows or group_rows,
                    key=lambda row: float(
                        row.get("source_score_kcal_mol")
                        if row.get("source_score_kcal_mol") is not None
                        else float("inf")
                    ),
                )
                criteria_by_rank: dict[int, list[str]] = {}
                rows_by_rank: dict[int, dict[str, Any]] = {}
                for row, criterion in (
                    (cnn, "CNN pose score"),
                    (empirical, "Vina/empirical score"),
                ):
                    rank = int(row["source_pose_rank"])
                    rows_by_rank[rank] = row
                    criteria_by_rank.setdefault(rank, []).append(criterion)
                selected.extend(
                    (
                        rows_by_rank[rank],
                        " + ".join(criteria_by_rank[rank]),
                    )
                    for rank in sorted(rows_by_rank)
                )
            else:
                selected.append(
                    (
                        min(
                            group_rows,
                            key=lambda row: int(row["source_pose_rank"]),
                        ),
                        "Best emitted pose",
                    )
                )
        for row, criterion in selected:
            source_sdf = (
                source_job.run_dir
                / Path(str(row["source_pose_file"])).with_suffix(".sdf")
            )
            if not source_sdf.is_file():
                continue
            rank = int(row["source_pose_rank"])
            replicate = int(row["replicate"])
            compound_id = str(row["compound_id"])
            candidates.append(
                {
                    "selection_id": _safe_id(
                        f"{compound_id}__replicate_{replicate:03d}__pose_{rank:03d}",
                        f"pose_{len(candidates) + 1:07d}",
                    ),
                    "compound_id": compound_id,
                    "replicate": replicate,
                    "prediction": f"pose {rank}",
                    "pose_rank": rank,
                    "source_engine": engine,
                    "source_kind": "docked pose",
                    "representative": True,
                    "selection_criterion": criterion,
                    "_source_path": source_sdf,
                    "_sdf_index": rank - 1,
                }
            )
        return candidates

    smiles = _source_smiles(source_job)
    if source_job.workflow == "openvs_docking":
        table_path = source_job.run_dir / "openvs_scores.csv"
        if not table_path.is_file():
            return []
        with table_path.open(newline="", errors="replace") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=1):
                relative = str(row.get("pose_file") or "")
                source_path = source_job.run_dir / relative
                if not relative or not source_path.is_file():
                    continue
                compound_id = str(row.get("compound_id") or f"compound_{index:07d}")
                replicate = int(row.get("replicate") or 1)
                rank = int(row.get("pose_rank") or 1)
                candidates.append(
                    {
                        "selection_id": _safe_id(
                            f"{compound_id}__replicate_{replicate:03d}__pose_{rank:03d}",
                            f"pose_{index:07d}",
                        ),
                        "compound_id": compound_id,
                        "replicate": replicate,
                        "prediction": f"pose {rank}",
                        "pose_rank": rank,
                        "source_engine": "RosettaLigand",
                        "source_kind": "docked complex",
                        "representative": True,
                        "selection_criterion": "Best emitted pose",
                        "smiles": smiles.get(compound_id, ""),
                        "_source_path": source_path,
                    }
                )
        return [
            row for row in candidates if int(row.get("pose_rank") or 1) == 1
        ]

    predictions = (
        source_job.artifact_manifest.by_type("predicted_complex")
        if source_job.artifact_manifest is not None
        else ()
    )
    roles = [str(artifact.role or artifact.label or "") for artifact in predictions]
    boltz_has_models = source_job.workflow == "boltz2_refolding" and any(
        re.search(r"_model_\d+$", role) for role in roles
    )
    af3_has_samples = source_job.workflow == "alphafold3_refolding" and any(
        re.search(r"(?:_|-)sample(?:_|-)\d+", role) for role in roles
    )
    input_payload: dict[str, Any] = {}
    try:
        input_payload = json.loads((source_job.run_dir / "input.json").read_text())
    except (OSError, TypeError, ValueError):
        pass
    input_parameters = input_payload.get("parameters")
    input_parameters = input_parameters if isinstance(input_parameters, dict) else {}
    af3_seed_start = int(input_parameters.get("model_seed_start") or 1001)
    for index, artifact in enumerate(predictions, start=1):
        source_path = artifact.resolve(source_job.run_dir, must_exist=True)
        if source_path is None:
            continue
        role = str(artifact.role or artifact.label or "")
        compound_id = role.split(":", 1)[0] or f"compound_{index:07d}"
        replicate_match = re.search(r"replicate_(\d+)", role)
        replicate = int(replicate_match.group(1)) if replicate_match else 1
        if source_job.workflow == "alphafold3_refolding":
            seed_match = re.search(r"seed-(\d+)", role)
            if seed_match:
                replicate = max(1, int(seed_match.group(1)) - af3_seed_start + 1)
        representative = True
        if boltz_has_models:
            representative = bool(re.search(r"_model_0$", role))
        elif af3_has_samples:
            representative = bool(
                re.search(r"(?:_|-)sample(?:_|-)0(?:$|:)", role)
            )
        if not representative:
            continue
        candidates.append(
            {
                "selection_id": _safe_id(
                    f"{compound_id}__{role}",
                    f"prediction_{index:07d}",
                ),
                "compound_id": compound_id,
                "replicate": replicate,
                "prediction": role or artifact.label,
                "pose_rank": "",
                "source_engine": (
                    "Boltz-2"
                    if source_job.workflow == "boltz2_refolding"
                    else "AlphaFold 3"
                ),
                "source_kind": "cofolded complex",
                "representative": representative,
                "selection_criterion": (
                    "Model 0"
                    if source_job.workflow == "boltz2_refolding"
                    else "Sample 0"
                ),
                "smiles": smiles.get(compound_id, ""),
                "_source_path": source_path,
            }
        )
    return candidates


def pose_validation_inventory_path(source_job: JobRecord) -> Path:
    return (
        runs_root().parent
        / "cache"
        / (
            "pose-validation-inventory-v"
            f"{POSE_VALIDATION_INVENTORY_SCHEMA_VERSION}"
        )
        / _safe_id(source_job.task_group, "source")
        / f"{_safe_id(source_job.run_id, 'run')}.json"
    )


def _pose_validation_source_revision(source_job: JobRecord) -> dict[str, Any]:
    return {
        "updated_at": source_job.updated_at,
        "completed_at": source_job.completed_at,
        "status": source_job.status,
        "replicates": int(
            source_job.metadata.get("replicates")
            or source_job.metadata.get("model_seed_count")
            or source_job.result.get("replicates")
            or 1
        ),
        "artifact_count": len(source_job.artifact_manifest.artifacts)
        if source_job.artifact_manifest is not None
        else 0,
        "progress": source_job.result.get("progress"),
    }


def pose_validation_inventory_summary(
    source_job: JobRecord,
) -> dict[str, Any]:
    """Read cached inventory counts without resolving every pose path."""
    try:
        payload = json.loads(
            pose_validation_inventory_path(source_job).read_text()
        )
    except (OSError, TypeError, ValueError):
        payload = {}
    cached_revision = payload.get("source_revision")
    current_revision = _pose_validation_source_revision(source_job)
    revision_matches = isinstance(cached_revision, dict) and all(
        cached_revision.get(key) == value
        for key, value in current_revision.items()
        if key != "artifact_count" or source_job.artifact_manifest is not None
    )
    valid = (
        isinstance(payload, dict)
        and payload.get("schema_version")
        == POSE_VALIDATION_INVENTORY_SCHEMA_VERSION
        and payload.get("source_run_id") == source_job.run_id
        and payload.get("source_task_group") == source_job.task_group
        and revision_matches
        and isinstance(payload.get("rows"), list)
    )
    if not valid:
        return {
            "cached": False,
            "candidate_count": None,
            "compound_ids": (),
        }
    rows = [item for item in payload["rows"] if isinstance(item, dict)]
    return {
        "cached": True,
        "candidate_count": int(
            payload.get("candidate_count")
            if payload.get("candidate_count") is not None
            else len(rows)
        ),
        "compound_ids": tuple(
            sorted(
                {
                    str(item.get("compound_id") or "")
                    for item in rows
                    if str(item.get("compound_id") or "")
                }
            )
        ),
    }


def cached_pose_validation_candidates(
    source_job: JobRecord,
    *,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Persist and reuse the immutable prediction inventory for one source."""
    cache_path = pose_validation_inventory_path(source_job)
    source_revision = _pose_validation_source_revision(source_job)
    if not refresh:
        try:
            payload = json.loads(cache_path.read_text())
        except (OSError, TypeError, ValueError):
            payload = {}
        if (
            isinstance(payload, dict)
            and payload.get("schema_version")
            == POSE_VALIDATION_INVENTORY_SCHEMA_VERSION
            and payload.get("source_run_id") == source_job.run_id
            and payload.get("source_task_group") == source_job.task_group
            and payload.get("source_revision") == source_revision
            and isinstance(payload.get("rows"), list)
        ):
            cached_rows: list[dict[str, Any]] = []
            for item in payload["rows"]:
                if not isinstance(item, dict):
                    continue
                relative = str(
                    item.get("_source_relative_path") or ""
                ).strip()
                if not relative:
                    continue
                source_path = (
                    source_job.run_dir / relative
                ).resolve()
                try:
                    source_path.relative_to(source_job.run_dir.resolve())
                except ValueError:
                    continue
                row = dict(item)
                row.pop("_source_relative_path", None)
                row["_source_path"] = source_path
                cached_rows.append(row)
            return cached_rows

    candidates = pose_validation_candidates(source_job)
    serialized: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        source_path = Path(row.pop("_source_path"))
        try:
            relative = source_path.resolve().relative_to(
                source_job.run_dir.resolve()
            )
        except ValueError:
            continue
        row["_source_relative_path"] = relative.as_posix()
        serialized.append(row)
    payload = {
        "schema_version": POSE_VALIDATION_INVENTORY_SCHEMA_VERSION,
        "source_run_id": source_job.run_id,
        "source_task_group": source_job.task_group,
        "source_workflow": source_job.workflow,
        "source_revision": source_revision,
        "created_at": _utc_now_iso(),
        "candidate_count": len(serialized),
        "compound_count": len(
            {
                str(row.get("compound_id") or "")
                for row in serialized
            }
        ),
        "rows": serialized,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(
        f".{cache_path.name}.{uuid4().hex}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(cache_path)
    return candidates


def _copy_sdf_record(source: Path, index: int, target: Path) -> None:
    supplier = Chem.SDMolSupplier(
        str(source),
        sanitize=False,
        removeHs=False,
        strictParsing=False,
    )
    molecule = supplier[index] if 0 <= index < len(supplier) else None
    if molecule is None:
        raise ValueError(f"Could not read pose {index + 1} from {source.name}")
    writer = Chem.SDWriter(str(target))
    writer.write(molecule)
    writer.close()


def finalize_pose_validation_job(
    run_dir: Path,
    *,
    returncode: int,
) -> JobRecord:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    summary = run_dir / "posebusters_summary.csv"
    report_path = run_dir / "posebusters_report.json"
    try:
        report = json.loads(report_path.read_text())
    except (OSError, TypeError, ValueError):
        report = {}
    expected = int(metadata.get("pose_count") or 0)
    success = (
        returncode == 0
        and summary.is_file()
        and int(report.get("validated_count") or 0) == expected
    )
    artifacts: list[ArtifactRef] = []
    for path, artifact_type, role in (
        (summary, "pose_validation_summary", "summary"),
        (
            run_dir / "native" / "posebusters_full.csv",
            "pose_validation_metrics",
            "full_report",
        ),
        (report_path, "pose_validation_report", "run_report"),
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
    for path in sorted((run_dir / "prepared").glob("*")):
        if path.is_file():
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir,
                    path,
                    (
                        "validated_pose"
                        if path.suffix.lower() == ".sdf"
                        else "validation_context"
                    ),
                    role=path.stem,
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
        or (
            f"PoseBusters produced {int(report.get('validated_count') or 0)}"
            f"/{expected} expected validation rows"
        )
    )
    result = {
        **report,
        "success": success,
        "returncode": returncode,
        "error": error,
    }
    _write_json(run_dir / "result.json", result)
    write_artifact_manifest(run_dir, artifacts)
    completed = _utc_now_iso()
    metadata.update(
        {
            "status": "completed" if success else "failed",
            "updated_at": completed,
            "completed_at": completed,
        }
    )
    if error:
        metadata["error"] = error
    _write_json(metadata_path, metadata)
    return JobRecord.load(run_dir, task_group=POSE_VALIDATION_TASK_GROUP)


def queue_pose_validation_job(
    source_job: JobRecord,
    *,
    selected_rows: Iterable[dict[str, Any]],
    max_workers: int | None = None,
    image: str = DEFAULT_POSEBUSTERS_IMAGE,
) -> JobRecord:
    if source_job.status != "completed":
        raise ValueError("PoseBusters requires a completed source result")
    rows = list(selected_rows)
    if not rows:
        raise ValueError("Select at least one docked or cofolded pose")
    compound_work_items = len(
        {
            str(row.get("compound_id") or "").strip()
            for row in rows
            if str(row.get("compound_id") or "").strip()
        }
    ) or len(rows)
    max_workers = adaptive_cpu_workers(
        compound_work_items,
        requested=max_workers,
        hard_cap=64,
    )
    run_id = str(uuid4())
    run_dir = runs_root() / POSE_VALIDATION_TASK_GROUP / run_id
    input_dir = run_dir / "input"
    pose_dir = input_dir / "poses"
    complex_dir = input_dir / "complexes"
    native_dir = run_dir / "native"
    pose_dir.mkdir(parents=True, exist_ok=False)
    complex_dir.mkdir()
    native_dir.mkdir()
    receptor_relative = ""
    if source_job.workflow == "docking_campaign":
        receptor = input_dir / "receptor.pdb"
        receptor.write_bytes(_source_receptor(source_job).read_bytes())
        receptor_relative = receptor.relative_to(run_dir).as_posix()

    input_rows: list[dict[str, Any]] = []
    for index, source in enumerate(rows, start=1):
        pose_id = _safe_id(source.get("selection_id"), f"pose_{index:07d}")
        source_path = Path(source["_source_path"])
        mol_pred = ""
        mol_cond = receptor_relative
        complex_file = ""
        if source_job.workflow == "docking_campaign":
            target = pose_dir / f"{pose_id}.sdf"
            _copy_sdf_record(
                source_path,
                int(source.get("_sdf_index") or 0),
                target,
            )
            mol_pred = target.relative_to(run_dir).as_posix()
        else:
            suffix = source_path.suffix.lower() or ".pdb"
            target = complex_dir / f"{pose_id}{suffix}"
            shutil.copy2(source_path, target)
            complex_file = target.relative_to(run_dir).as_posix()
            mol_cond = ""
        input_rows.append(
            {
                "pose_id": pose_id,
                "compound_id": source.get("compound_id", ""),
                "source_engine": source.get("source_engine", source_job.tool),
                "source_kind": source.get("source_kind", ""),
                "replicate": source.get("replicate", ""),
                "prediction": source.get("prediction", ""),
                "selection_criterion": source.get(
                    "selection_criterion", ""
                ),
                "smiles": source.get("smiles", ""),
                "mol_pred": mol_pred,
                "mol_cond": mol_cond,
                "complex_file": complex_file,
                "source_artifact_path": source_path.relative_to(
                    source_job.run_dir
                ).as_posix(),
            }
        )
    table = input_dir / "validation_inputs.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(input_rows[0]))
        writer.writeheader()
        writer.writerows(input_rows)

    command = build_docker_command(
        DockerRunSpec(
            tool=registered_tool("posebusters", image=image),
            command=(
                "--input",
                "/workspace/input/validation_inputs.csv",
                "--output",
                "/workspace/native/posebusters_full.csv",
                "--summary",
                "/workspace/posebusters_summary.csv",
                "--report",
                "/workspace/posebusters_report.json",
                "--max-workers",
                str(max_workers),
            ),
            mounts=(DockerMount(run_dir, "/workspace"),),
            environment=NATIVE_THREAD_ENVIRONMENT,
            gpu_enabled=False,
            use_host_user=True,
        )
    )
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "pose_validation",
        "workflow": "posebusters_validation",
        "operation": "pose_validation",
        "status": "queued",
        "tool": "PoseBusters",
        "parent_run_id": source_job.run_id,
        "source_task_group": source_job.task_group,
        "source_workflow": source_job.workflow,
        "source_engine": source_job.tool,
        "selection_schema_version": POSE_VALIDATION_INVENTORY_SCHEMA_VERSION,
        "selection_policy": POSE_VALIDATION_SELECTION_POLICY,
        "pose_count": len(input_rows),
        "compound_count": len(
            {str(row["compound_id"]) for row in input_rows}
        ),
        "max_workers": max_workers,
        "cpu_worker_policy": "adaptive-global-limit",
        "cpu_work_items": compound_work_items,
        "docker_image": image,
        "created_at": now,
        "updated_at": now,
        "queued_at": now,
        "queued_command": command,
        "gpu_queued": False,
        "resources": {
            **registered_tool("posebusters", image=image).resources.to_dict(),
            "cpu_threads": max_workers,
        },
        "worker_finalizer": "posebusters_validation",
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "source_job": {
                "run_id": source_job.run_id,
                "task_group": source_job.task_group,
                "workflow": source_job.workflow,
                "tool": source_job.tool,
            },
            "selection": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"mol_pred", "mol_cond", "complex_file"}
                }
                for row in input_rows
            ],
            "parameters": {
                "config": "dock",
                "selection_schema_version": (
                    POSE_VALIDATION_INVENTORY_SCHEMA_VERSION
                ),
                "selection_policy": POSE_VALIDATION_SELECTION_POLICY,
                "max_workers": max_workers,
                "cpu_worker_policy": "adaptive-global-limit",
                "cpu_work_items": compound_work_items,
                "gpu": False,
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
    return JobRecord.load(run_dir, task_group=POSE_VALIDATION_TASK_GROUP)
