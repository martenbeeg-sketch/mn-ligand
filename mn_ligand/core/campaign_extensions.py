from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from mn_ligand.core.jobs import JobRecord, iter_job_records
from mn_ligand.runtime import runs_root


SUPPORTED_WORKFLOWS = frozenset(
    {
        "docking_campaign",
        "openvs_docking",
        "alphafold3_refolding",
        "boltz2_refolding",
        "nesso_affinity",
    }
)
ACTIVE_STATES = frozenset({"queued", "preparing", "running", "paused"})


@dataclass(frozen=True)
class ExtensionResult:
    run_id: str
    workflow: str
    previous_repetitions: int
    target_repetitions: int
    status: str
    detail: str = ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _commands(metadata: dict[str, Any]) -> list[list[str]]:
    raw = metadata.get("queued_commands")
    if not isinstance(raw, list) or not raw:
        raw = [metadata.get("queued_command")]
    return [list(map(str, command)) for command in raw if isinstance(command, list) and command]


def _input_parameters(job: JobRecord) -> dict[str, Any]:
    payload = _read_json(job.run_dir / "input.json")
    return payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}


def configured_repetitions(job: JobRecord) -> int:
    parameters = _input_parameters(job)
    if job.workflow == "alphafold3_refolding":
        return max(1, int(parameters.get("model_seed_count") or job.metadata.get("model_seed_count") or 1))
    return max(1, int(parameters.get("replicates") or job.metadata.get("replicates") or 1))


def _excluded_entity_ids(job: JobRecord) -> set[str]:
    """Return compounds explicitly excluded from this engine's denominator."""
    excluded: set[str] = set()
    for payload in (job.result, job.metadata):
        values = payload.get("excluded_compound_ids")
        if isinstance(values, list):
            excluded.update(str(value).strip() for value in values if str(value).strip())
    path = job.run_dir / "excluded_compounds.tsv"
    if path.is_file():
        try:
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    compound_id = str(row.get("compound_id") or "").strip()
                    if compound_id:
                        excluded.add(compound_id)
        except (OSError, csv.Error):
            pass
    return excluded


def _expected_entity_count(job: JobRecord) -> int:
    staged = 0
    if job.workflow == "nesso_affinity":
        staged = sum(1 for path in (job.run_dir / "inputs").glob("*.yaml") if path.is_file())
    elif job.workflow == "alphafold3_refolding":
        staged = sum(1 for path in (job.run_dir / "inputs").glob("*.json") if path.is_file())
    payload = _read_json(job.run_dir / "input.json")
    compounds = payload.get("compound_artifacts") or payload.get("compound_sets") or []
    total = max(
        1,
        staged,
        int(job.metadata.get("compound_count") or 0),
        len(compounds) if isinstance(compounds, list) else 0,
    )
    return max(1, total - len(_excluded_entity_ids(job)))


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _completed_repetition_indices(job: JobRecord) -> set[int]:
    expected = _expected_entity_count(job)
    explicitly_eligible = {
        int(value)
        for value in job.metadata.get("extension_eligible_repetition_indices", [])
        if str(value).isdigit() and int(value) > 0
    }
    if job.workflow == "alphafold3_refolding":
        parameters = _input_parameters(job)
        seed_start = int(parameters.get("model_seed_start") or job.metadata.get("model_seed_start") or 1001)
        candidates_by_seed: dict[int, set[str]] = {}
        output = job.run_dir / "output"
        for path in output.glob("*/seed-*_sample-*") if output.is_dir() else ():
            match = re.search(r"seed-(\d+)_sample-", path.name)
            structures = list(path.glob("*_model.cif")) + list(path.glob("*_model.pdb"))
            summaries = list(path.glob("*_summary_confidences.json"))
            if match and any(_nonempty(item) for item in structures) and any(_nonempty(item) for item in summaries):
                candidates_by_seed.setdefault(int(match.group(1)), set()).add(path.parent.name)
        return explicitly_eligible | {
            seed - seed_start + 1
            for seed, candidates in candidates_by_seed.items()
            if seed >= seed_start and len(candidates) >= expected
        }
    if job.workflow == "openvs_docking":
        found = explicitly_eligible | {
            int(value)
            for value in job.metadata.get("recovered_repetition_indices", [])
            if str(value).isdigit() and int(value) > 0
        }
        native = job.run_dir / "native"
        poses = job.run_dir / "poses"
        expected_chunks = sum(1 for path in (job.run_dir / "chunks").glob("ligands_*") if path.is_file())
        for path in native.glob("replicate_*") if native.is_dir() else ():
            match = re.fullmatch(r"replicate_(\d+)", path.name)
            if not match or not path.is_dir():
                continue
            score_files = list(path.glob("*/run.score.sc"))
            silent_files = list(path.glob("*/run.out"))
            pose_files = list(poses.glob(f"{path.name}_*.pdb")) if poses.is_dir() else []
            if (
                score_files
                and len(score_files) == len(silent_files)
                and (not expected_chunks or len(score_files) == expected_chunks)
                and all(_nonempty(item) for item in (*score_files, *silent_files))
                and any(_nonempty(item) for item in pose_files)
            ):
                found.add(int(match.group(1)))
        return found
    if job.workflow == "docking_campaign":
        found: set[int] = set(explicitly_eligible)
        root = job.run_dir / "results"
        for path in root.glob("replicate_*") if root.is_dir() else ():
            match = re.fullmatch(r"replicate_(\d+)", path.name)
            poses = {
                item.name.removesuffix("_out.pdbqt")
                for item in path.glob("*_out.pdbqt")
                if _nonempty(item)
            }
            if match and len(poses) >= expected:
                found.add(int(match.group(1)))
        return found
    if job.workflow == "boltz2_refolding":
        found: set[int] = set(explicitly_eligible)
        root = job.run_dir / "output"
        for path in root.glob("replicate_*") if root.is_dir() else ():
            match = re.fullmatch(r"replicate_(\d+)", path.name)
            candidates: set[str] = set()
            for confidence in path.glob("**/predictions/*/confidence_*_model_*.json"):
                structures = list(confidence.parent.glob("*_model_*.cif")) + list(confidence.parent.glob("*_model_*.pdb"))
                if _nonempty(confidence) and any(_nonempty(item) for item in structures):
                    candidates.add(confidence.parent.name)
            if match and len(candidates) >= expected:
                found.add(int(match.group(1)))
        return found
    if job.workflow == "nesso_affinity":
        found: set[int] = set(explicitly_eligible)
        root = job.run_dir / "output"
        for path in root.glob("replicate_*") if root.is_dir() else ():
            match = re.fullmatch(r"replicate_(\d+)", path.name)
            candidates: set[str] = set()
            for affinity in path.glob("**/predictions/*/affinity.json"):
                payload = _read_json(affinity)
                if _nonempty(affinity) and payload.get("affinity_pred_value") is not None:
                    candidates.add(affinity.parent.name)
            if match and len(candidates) >= expected:
                found.add(int(match.group(1)))
        return found
    roots = (job.run_dir / "results", job.run_dir / "output")
    found: set[int] = set(explicitly_eligible)
    for root in roots:
        for path in root.glob("replicate_*") if root.is_dir() else ():
            match = re.fullmatch(r"replicate_(\d+)", path.name)
            if match and path.is_dir() and any(path.rglob("*")):
                found.add(int(match.group(1)))
    return found


def completed_repetitions(job: JobRecord) -> int:
    return len(_completed_repetition_indices(job))


def effective_repetitions(job: JobRecord) -> int:
    return max(configured_repetitions(job), completed_repetitions(job))


def _replace_flag(command: list[str], flag: str, value: str) -> list[str]:
    updated = list(command)
    if flag in updated:
        index = updated.index(flag)
        if index + 1 < len(updated):
            updated[index + 1] = value
            return updated
    updated.extend([flag, value])
    return updated


def _replace_env(command: list[str], name: str, value: str) -> list[str]:
    token = f"{name}="
    updated = list(command)
    for index, item in enumerate(updated):
        if item.startswith(token):
            updated[index] = token + value
            return updated
    image_index = max(1, updated.index("bash") - 1) if "bash" in updated else len(updated)
    updated[image_index:image_index] = ["-e", token + value]
    return updated


def _clone_indexed_command(base: list[str], replicate: int, seed: int) -> list[str]:
    command = list(base)
    for index, value in enumerate(command):
        command[index] = re.sub(r"/replicate_\d{3}\b", f"/replicate_{replicate:03d}", value)
    command = _replace_flag(command, "--seed", str(seed))
    return command


def _extension_commands(job: JobRecord, current: int, target: int) -> list[list[str]]:
    commands = _commands(job.metadata)
    if not commands:
        raise ValueError("The job has no reusable command")
    start = current + 1
    completed_indices = _completed_repetition_indices(job)
    missing_indices = [
        replicate for replicate in range(1, target + 1)
        if replicate not in completed_indices
    ]
    if not missing_indices:
        return []
    if job.workflow == "docking_campaign":
        command = list(commands[0])
        shell_index = command.index("-lc") + 1
        script = command[shell_index]
        script = re.sub(r"DOCKING_REPLICATES=\d+", f"DOCKING_REPLICATES={target}", script, count=1)
        script = re.sub(
            r"for replicate in \$\(seq [^;]+?\); do",
            "for replicate in $DOCKING_REPLICATE_IDS; do",
            script,
            count=1,
        )
        ids = " ".join(map(str, missing_indices))
        script = script.replace(
            f"DOCKING_REPLICATES={target};",
            f'DOCKING_REPLICATES={target}; DOCKING_REPLICATE_IDS="{ids}";',
            1,
        )
        command[shell_index] = script
        return [command]
    if job.workflow == "openvs_docking":
        # Historical/recovery Rosetta runs may have staged an older runner.
        # Extensions must use the current compound-level exclusion logic so a
        # single incompatible ligand cannot abort a repeated campaign.
        from mn_ligand.workflows.openvs import (
            _ROSETTA_PARAMS_VALIDATOR_SCRIPT,
            _RUNNER_SCRIPT,
        )

        runner = job.run_dir / "input" / "run_openvs.sh"
        runner.write_text(_RUNNER_SCRIPT)
        (job.run_dir / "input" / "validate_rosetta_params.py").write_text(
            _ROSETTA_PARAMS_VALIDATOR_SCRIPT
        )
        text = runner.read_text()
        text = text.replace(
            'for replicate in $(seq 1 "$OPENVS_REPLICATES"); do',
            'for replicate in ${OPENVS_REPLICATE_IDS:-$(seq "${OPENVS_REPLICATE_START:-1}" "$OPENVS_REPLICATES")}; do',
        )
        text = text.replace(
            'for replicate in $(seq "${OPENVS_REPLICATE_START:-1}" "$OPENVS_REPLICATES"); do',
            'for replicate in ${OPENVS_REPLICATE_IDS:-$(seq "${OPENVS_REPLICATE_START:-1}" "$OPENVS_REPLICATES")}; do',
        )
        runner.write_text(text)
        command = _replace_env(commands[0], "OPENVS_REPLICATES", str(target))
        command = _replace_env(command, "OPENVS_REPLICATE_START", str(start))
        command = _replace_env(
            command,
            "OPENVS_REPLICATE_IDS",
            " ".join(map(str, missing_indices)),
        )
        return [command]
    parameters = _input_parameters(job)
    seed_start = int(parameters.get("seed_start") or parameters.get("seed") or job.metadata.get("seed_start") or 1001)
    if job.workflow in {"boltz2_refolding", "nesso_affinity"}:
        base = commands[0]
        return [
            _clone_indexed_command(base, replicate, seed_start + replicate - 1)
            for replicate in missing_indices
        ]
    if job.workflow == "alphafold3_refolding":
        inference = next((command for command in reversed(commands) if any("run_alphafold.py" in item for item in command)), None)
        if inference is None:
            raise ValueError("AlphaFold inference command is missing")
        seed_start = int(parameters.get("model_seed_start") or 1001)
        source_dir = job.run_dir / "data"
        if current == 0 and not any(source_dir.glob("*.json")):
            requested_seeds = list(range(seed_start, seed_start + target))
            for source in (job.run_dir / "inputs").glob("*.json"):
                payload = _read_json(source)
                payload["modelSeeds"] = requested_seeds
                _write_json(source, payload)
            modified = [item for item in inference if not item.startswith("--num_seeds=")]
            return [*commands[:-1], modified]
        suffix = "_".join(f"{replicate:03d}" for replicate in missing_indices)
        extension_dir = job.run_dir / f"data_extension_{suffix}"
        if extension_dir.exists():
            shutil.rmtree(extension_dir)
        extension_dir.mkdir()
        missing_seeds = [
            seed_start + replicate - 1 for replicate in missing_indices
        ]
        for source in source_dir.glob("*.json"):
            payload = _read_json(source)
            payload["modelSeeds"] = missing_seeds
            _write_json(extension_dir / source.name, payload)
        if not any(extension_dir.glob("*.json")):
            raise ValueError("Processed AlphaFold input JSON is missing")
        command = [
            f"--input_dir=/work/{extension_dir.name}"
            if item.startswith("--input_dir=")
            else item
            for item in inference
            if not item.startswith("--num_seeds=")
        ]
        # AlphaFold reads the exact deterministic seed list from modelSeeds.
        # In particular, do not pass --num_seeds=1: the current AlphaFast
        # runner rejects it, and overriding the JSON would lose seed identity.
        return [command]
    raise ValueError(f"Unsupported repeatable workflow: {job.workflow}")


def _update_input(job: JobRecord, target: int) -> None:
    path = job.run_dir / "input.json"
    payload = _read_json(path)
    parameters = payload.setdefault("parameters", {})
    if job.workflow == "alphafold3_refolding":
        parameters["model_seed_count"] = target
    else:
        parameters["replicates"] = target
    _write_json(path, payload)


def _queue_extension(job: JobRecord, commands: list[list[str]], previous: int, target: int) -> None:
    path = job.run_dir / "metadata.json"
    metadata = _read_json(path)
    now = _utc_now_iso()
    history = list(metadata.get("extension_history") or [])
    history.append(
        {
            "requested_at": now,
            "previous_repetitions": previous,
            "target_repetitions": target,
            "previous_status": job.status,
        }
    )
    for key in (
        "error", "completed_at", "returncode", "worker_id", "worker_pid",
        "worker_heartbeat_at", "executed_command", "executed_commands",
        "command_index", "command_count", "stdout_tail", "stderr_tail",
        "cancellation_requested", "cancellation_requested_at", "cancelled_at",
        "finalizer_error", "selected_gpu",
    ):
        metadata.pop(key, None)
    metadata.update(
        {
            "status": "queued",
            "queued_at": now,
            "updated_at": now,
            "queued_command": commands[0],
            "queued_commands": commands,
            "replicates": target,
            "extension_history": history,
        }
    )
    if job.workflow == "alphafold3_refolding":
        metadata["model_seed_count"] = target
    metadata.pop("requested_repetitions", None)
    _write_json(path, metadata)
    _update_input(job, target)


def extend_job_repetitions(job: JobRecord, target: int) -> ExtensionResult:
    target = int(target)
    if target < 1 or target > 100:
        raise ValueError("Target repetitions must be between 1 and 100")
    configured = configured_repetitions(job)
    completed = completed_repetitions(job)
    current = max(completed, configured) if job.status in ACTIVE_STATES else completed
    if job.workflow not in SUPPORTED_WORKFLOWS:
        return ExtensionResult(job.run_id, job.workflow, current, target, "unsupported", "This workflow is not safely appendable yet")
    if target <= completed:
        return ExtensionResult(job.run_id, job.workflow, current, target, "unchanged", "Already at or above the requested count")
    if job.status in ACTIVE_STATES:
        metadata = _read_json(job.run_dir / "metadata.json")
        metadata["requested_repetitions"] = target
        metadata["updated_at"] = _utc_now_iso()
        _write_json(job.run_dir / "metadata.json", metadata)
        return ExtensionResult(job.run_id, job.workflow, current, target, "pending", "Will append after the currently queued or active work finishes")
    commands = _extension_commands(job, current, target)
    _queue_extension(job, commands, current, target)
    return ExtensionResult(job.run_id, job.workflow, current, target, "queued", f"Queued {target - current} missing repetition(s)")


def apply_pending_extension(run_dir: Path) -> ExtensionResult | None:
    job = JobRecord.load(run_dir, task_group=run_dir.parent.name)
    target = int(job.metadata.get("requested_repetitions") or 0)
    if not target:
        return None
    return extend_job_repetitions(job, target)


def campaign_jobs(campaign_id: str, jobs: Iterable[JobRecord] | None = None) -> list[JobRecord]:
    records = list(jobs) if jobs is not None else iter_job_records(runs_root())
    return [job for job in records if str(job.metadata.get("launch_campaign_id") or "") == str(campaign_id)]


def _artifact_signature(job: JobRecord) -> tuple[str, str, str, tuple[str, ...]]:
    payload = _read_json(job.run_dir / "input.json")
    target = payload.get("target_artifact") or payload.get("target") or {}
    compounds = payload.get("compound_artifacts") or payload.get("compound_sets") or []
    # A prepared-target run is a scientific input in its own right.  Two
    # preparation branches may intentionally produce byte-identical receptor
    # files, so their SHA-256 values must not merge their downstream jobs.
    # Retries retain the same target run ID and therefore still collapse.
    target_identity = str(
        target.get("run_id")
        or job.metadata.get("prepared_target_run_id")
        or target.get("artifact_id")
        or target.get("sha256")
        or job.parent_run_id
    )
    return (
        job.workflow,
        str(job.metadata.get("engine") or job.metadata.get("tool") or job.workflow).strip().lower(),
        target_identity,
        tuple(sorted(str(item.get("sha256") or item.get("artifact_id") or "") for item in compounds if isinstance(item, dict))),
    )


def canonical_campaign_jobs(campaign_id: str, jobs: Iterable[JobRecord] | None = None) -> list[JobRecord]:
    grouped: dict[tuple[str, str, str, tuple[str, ...]], list[JobRecord]] = {}
    for job in campaign_jobs(campaign_id, jobs):
        if job.workflow not in SUPPORTED_WORKFLOWS:
            continue
        grouped.setdefault(_artifact_signature(job), []).append(job)
    priority = {"running": 5, "preparing": 5, "queued": 4, "completed": 3, "paused": 2, "failed": 1, "cancelled": 0, "blocked": 0}
    return [
        max(
            group,
            key=lambda item: (
                not bool(item.metadata.get("recovery_of_run_id")),
                priority.get(item.status, -1),
                completed_repetitions(item),
                item.created_at,
            ),
        )
        for group in grouped.values()
    ]


def extend_campaign_repetitions(campaign_id: str, target: int) -> list[ExtensionResult]:
    return [extend_job_repetitions(job, target) for job in canonical_campaign_jobs(campaign_id)]
