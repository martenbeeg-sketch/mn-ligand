from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Iterable
from uuid import uuid4

from rdkit import Chem

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    registered_tool,
    write_registered_command_record,
)
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.docking import DEFAULT_DOCKING_IMAGE


RESCORING_TASK_GROUP = "rescoring"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _safe_id(value: object, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (text or fallback)[:160]


def _pdbqt_models(text: str) -> list[str]:
    """Split Vina/GNINA multi-model PDBQT without changing atom coordinates."""
    models: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("MODEL"):
            if current:
                models.append("\n".join(current).rstrip() + "\n")
                current = []
            continue
        if line.startswith("ENDMDL"):
            if current:
                models.append("\n".join(current).rstrip() + "\n")
                current = []
            continue
        current.append(line)
    if current and any(line.startswith(("ATOM", "HETATM")) for line in current):
        models.append("\n".join(current).rstrip() + "\n")
    return [
        model
        for model in models
        if any(line.startswith(("ATOM", "HETATM")) for line in model.splitlines())
    ]


def _remark_float(text: str, *names: str) -> float | None:
    for name in names:
        match = re.search(
            rf"^REMARK\s+{re.escape(name)}:?\s+(-?\d+(?:\.\d+)?)",
            text,
            flags=re.MULTILINE,
        )
        if match:
            return float(match.group(1))
    return None


def _pose_coordinates(text: str) -> list[tuple[float, float, float]]:
    coordinates: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
            continue
        try:
            coordinates.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
        except ValueError:
            continue
    return coordinates


def _maximum_coordinate_displacement(left: str, right: str) -> float | None:
    first = _pose_coordinates(left)
    second = _pose_coordinates(right)
    if not first or len(first) != len(second):
        return None
    return max(
        (
            (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
        )
        ** 0.5
        for (ax, ay, az), (bx, by, bz) in zip(first, second)
    )


def source_pose_rows(source_job: JobRecord) -> list[dict[str, Any]]:
    """Inventory every native pose model in a completed docking result."""
    rows: list[dict[str, Any]] = []
    results_root = source_job.run_dir / "results"
    for path in sorted(results_root.glob("**/*_out.pdbqt")):
        relative = path.relative_to(source_job.run_dir)
        replicate_match = next(
            (
                re.fullmatch(r"replicate_(\d+)", part)
                for part in relative.parts
                if part.startswith("replicate_")
            ),
            None,
        )
        replicate = int(replicate_match.group(1)) if replicate_match else 1
        compound_id = path.name.removesuffix("_out.pdbqt")
        for rank, model in enumerate(
            _pdbqt_models(path.read_text(errors="replace")), start=1
        ):
            rows.append(
                {
                    "compound_id": compound_id,
                    "replicate": replicate,
                    "source_pose_rank": rank,
                    "source_score_kcal_mol": _remark_float(
                        model, "minimizedAffinity", "VINA RESULT"
                    ),
                    "source_cnn_score": _remark_float(model, "CNNscore"),
                    "source_cnn_affinity": _remark_float(model, "CNNaffinity"),
                    "source_pose_file": relative.as_posix(),
                    "_model_text": model,
                }
            )
    return rows


def _source_receptor(source_job: JobRecord) -> Path:
    for candidate in (
        source_job.run_dir / "input" / "receptor.pdb",
        source_job.run_dir / "prepared" / "receptor.pdb",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("The source docking job has no stored receptor PDB")


def _source_smiles(source_job: JobRecord) -> dict[str, str]:
    path = source_job.run_dir / "input" / "compounds.tsv"
    if not path.is_file():
        return {}
    with path.open(newline="", errors="replace") as handle:
        return {
            str(row.get("compound_id") or ""): str(row.get("smiles") or "")
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("compound_id") and row.get("smiles")
        }


def create_pose_selection_job(
    source_job: JobRecord,
    *,
    compound_ids: Iterable[str] = (),
    replicates: Iterable[int] = (),
    pose_ranks: Iterable[int] = (1,),
) -> JobRecord:
    if source_job.status != "completed":
        raise ValueError("Only completed docking jobs can be selected for rescoring")
    available = source_pose_rows(source_job)
    selected_compounds = {str(value) for value in compound_ids}
    selected_replicates = {int(value) for value in replicates}
    selected_ranks = {int(value) for value in pose_ranks}
    selected = [
        row
        for row in available
        if (not selected_compounds or str(row["compound_id"]) in selected_compounds)
        and (not selected_replicates or int(row["replicate"]) in selected_replicates)
        and (not selected_ranks or int(row["source_pose_rank"]) in selected_ranks)
    ]
    if not selected:
        raise ValueError("The selected source job and filters contain no readable poses")

    run_id = str(uuid4())
    run_dir = runs_root() / RESCORING_TASK_GROUP / run_id
    pose_dir = run_dir / "poses"
    pose_dir.mkdir(parents=True, exist_ok=False)
    receptor = run_dir / "receptor.pdb"
    receptor.write_bytes(_source_receptor(source_job).read_bytes())
    smiles_by_id = _source_smiles(source_job)
    pose_rows: list[dict[str, Any]] = []
    selected_sdf = run_dir / "selected_poses.sdf"
    sdf_writer = Chem.SDWriter(str(selected_sdf))
    artifacts: list[ArtifactRef] = [
        ArtifactRef.from_path(
            run_dir, receptor, "prepared_target", role="rescoring_receptor"
        )
    ]
    for index, source in enumerate(selected, start=1):
        pose_id = _safe_id(
            (
                f"{source['compound_id']}__replicate_{int(source['replicate']):03d}"
                f"__pose_{int(source['source_pose_rank']):03d}"
            ),
            f"pose_{index:07d}",
        )
        pose_path = pose_dir / f"{pose_id}.pdbqt"
        pose_path.write_text(str(source["_model_text"]))
        source_sdf = (
            source_job.run_dir
            / Path(str(source["source_pose_file"])).with_suffix(".sdf")
        )
        sdf_pose_file = ""
        if source_sdf.is_file():
            try:
                molecule = next(
                    (
                        item
                        for item_index, item in enumerate(
                            Chem.SDMolSupplier(
                                str(source_sdf), removeHs=False, sanitize=False
                            ),
                            start=1,
                        )
                        if item_index == int(source["source_pose_rank"])
                        and item is not None
                    ),
                    None,
                )
            except (OSError, ValueError):
                molecule = None
            if molecule is not None:
                molecule.SetProp("_Name", pose_id)
                molecule.SetProp("source_compound_id", str(source["compound_id"]))
                molecule.SetIntProp("source_replicate", int(source["replicate"]))
                molecule.SetIntProp(
                    "source_pose_rank", int(source["source_pose_rank"])
                )
                sdf_writer.write(molecule)
                sdf_pose_file = selected_sdf.relative_to(run_dir).as_posix()
        row = {
            key: value for key, value in source.items() if not key.startswith("_")
        }
        row.update(
            {
                "pose_id": pose_id,
                "pose_file": pose_path.relative_to(run_dir).as_posix(),
                "sdf_pose_set": sdf_pose_file,
                "smiles": smiles_by_id.get(str(source["compound_id"]), ""),
            }
        )
        pose_rows.append(row)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                pose_path,
                "docked_pose",
                role=pose_id,
                metadata={
                    "compound_id": row["compound_id"],
                    "replicate": row["replicate"],
                    "source_pose_rank": row["source_pose_rank"],
                },
            )
        )
    sdf_writer.close()
    table = run_dir / "pose_selection.csv"
    fields = list(pose_rows[0])
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pose_rows)
    compounds = run_dir / "selected_compounds.csv"
    compound_rows = list(
        {
            (str(row["compound_id"]), str(row["smiles"]))
            for row in pose_rows
            if str(row.get("smiles") or "")
        }
    )
    with compounds.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("compound_id", "smiles"))
        writer.writerows(sorted(compound_rows))
    artifacts.extend(
        (
            ArtifactRef.from_path(
                run_dir, table, "pose_selection", role="immutable_selection"
            ),
            ArtifactRef.from_path(
                run_dir, compounds, "compound_set", role="selected_pose_compounds"
            ),
        )
    )
    if selected_sdf.is_file() and selected_sdf.stat().st_size:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                selected_sdf,
                "pose_set",
                role="coordinate_preserving_sdf_selection",
                metadata={"pose_count": len(pose_rows)},
            )
        )
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "pose_selection",
        "workflow": "pose_selection",
        "operation": "rescoring_selection",
        "status": "completed",
        "tool": "Pose selection",
        "parent_run_id": source_job.run_id,
        "source_task_group": source_job.task_group,
        "source_engine": source_job.tool,
        "pose_count": len(pose_rows),
        "compound_count": len({str(row["compound_id"]) for row in pose_rows}),
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "source_job": {
                "run_id": source_job.run_id,
                "task_group": source_job.task_group,
            },
            "selection": {
                "compound_ids": sorted(selected_compounds),
                "replicates": sorted(selected_replicates),
                "pose_ranks": sorted(selected_ranks),
            },
        },
    )
    _write_json(
        run_dir / "result.json",
        {
            "success": True,
            "pose_count": len(pose_rows),
            "compound_count": metadata["compound_count"],
        },
    )
    write_artifact_manifest(run_dir, artifacts)
    return JobRecord.load(run_dir, task_group=RESCORING_TASK_GROUP)


def _selection_rows(selection_job: JobRecord) -> list[dict[str, str]]:
    path = selection_job.run_dir / "pose_selection.csv"
    if not path.is_file():
        raise FileNotFoundError("Pose selection table is missing")
    return list(csv.DictReader(path.open(newline="", errors="replace")))


def finalize_gnina_rescoring_job(run_dir: Path, *, returncode: int) -> JobRecord:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    source_dir = Path(str(metadata["selection_run_dir"]))
    source_rows = {
        row["pose_id"]: row
        for row in csv.DictReader(
            (source_dir / "pose_selection.csv").open(newline="", errors="replace")
        )
    }
    result_rows: list[dict[str, Any]] = []
    for pose_id, source in source_rows.items():
        output_path = run_dir / "results" / f"{pose_id}.pdbqt"
        if not output_path.is_file():
            continue
        output_text = output_path.read_text(errors="replace")
        source_text = (source_dir / source["pose_file"]).read_text(errors="replace")
        result_rows.append(
            {
                "pose_id": pose_id,
                "compound_id": source["compound_id"],
                "replicate": int(source["replicate"]),
                "source_pose_rank": int(source["source_pose_rank"]),
                "source_engine": metadata.get("source_engine", ""),
                "source_score_kcal_mol": source.get("source_score_kcal_mol", ""),
                "gnina_empirical_score_kcal_mol": _remark_float(
                    output_text, "minimizedAffinity", "VINA RESULT"
                ),
                "gnina_cnn_score": _remark_float(output_text, "CNNscore"),
                "gnina_cnn_affinity": _remark_float(output_text, "CNNaffinity"),
                "maximum_coordinate_displacement_angstrom": (
                    _maximum_coordinate_displacement(source_text, output_text)
                ),
                "source_pose_file": source["pose_file"],
                "rescored_pose_file": output_path.relative_to(run_dir).as_posix(),
            }
        )
    table = run_dir / "rescoring_scores.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    if result_rows:
        with table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
            writer.writeheader()
            writer.writerows(result_rows)
    success = returncode == 0 and len(result_rows) == len(source_rows)
    artifacts: list[ArtifactRef] = []
    if result_rows:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir, table, "rescoring_scores", role="ranked_scores"
            )
        )
        for row in result_rows:
            output = run_dir / str(row["rescored_pose_file"])
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir,
                    output,
                    "rescored_pose",
                    role=str(row["pose_id"]),
                    metadata={"coordinates_preserved": True},
                )
            )
    for name, role in (("stdout.log", "stdout"), ("stderr.log", "stderr")):
        path = run_dir / name
        path.touch(exist_ok=True)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir, path, "job_log", role=role, checksum=False
            )
        )
    write_artifact_manifest(run_dir, artifacts)
    displacement = [
        float(row["maximum_coordinate_displacement_angstrom"])
        for row in result_rows
        if row["maximum_coordinate_displacement_angstrom"] is not None
    ]
    error = "" if success else (
        (run_dir / "stderr.log").read_text(errors="replace")[-4000:]
        or f"GNINA rescored {len(result_rows)}/{len(source_rows)} selected poses"
    )
    _write_json(
        run_dir / "result.json",
        {
            "success": success,
            "returncode": returncode,
            "pose_count": len(source_rows),
            "rescored_pose_count": len(result_rows),
            "maximum_coordinate_displacement_angstrom": max(displacement)
            if displacement
            else None,
            "error": error,
        },
    )
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
    return JobRecord.load(run_dir, task_group=RESCORING_TASK_GROUP)


def run_gnina_rescoring_job(
    *,
    selection_job: JobRecord,
    image: str = DEFAULT_DOCKING_IMAGE,
    gpu_device: str = "all",
    cnn_model: str = "",
    cnn_rotation: int = 0,
    enqueue_only: bool = False,
) -> JobRecord:
    if selection_job.workflow != "pose_selection":
        raise ValueError("GNINA rescoring requires an immutable pose-selection job")
    rows = _selection_rows(selection_job)
    if not rows:
        raise ValueError("The pose selection is empty")
    run_id = str(uuid4())
    run_dir = runs_root() / RESCORING_TASK_GROUP / run_id
    (run_dir / "prepared").mkdir(parents=True, exist_ok=False)
    (run_dir / "results").mkdir()
    index = run_dir / "pose_index.tsv"
    with index.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("pose_id", "pose_file"))
        writer.writerows((row["pose_id"], row["pose_file"]) for row in rows)
    optional = ""
    if str(cnn_model).strip():
        optional += " --cnn " + shlex.quote(str(cnn_model).strip())
    if int(cnn_rotation) > 0:
        optional += f" --cnn_rotation {min(24, int(cnn_rotation))}"
    shell_script = (
        "set -euo pipefail; cd /workspace; "
        "mk_prepare_receptor.py --read_pdb /source/receptor.pdb "
        "-o prepared/receptor -p; "
        "tail -n +2 pose_index.tsv | while IFS=$'\\t' read -r pose_id pose_file; do "
        "gnina --receptor prepared/receptor.pdbqt "
        '--ligand "/source/${pose_file}" --score_only '
        + optional
        + ' --out "results/${pose_id}.pdbqt" '
        + '--log "results/${pose_id}.log"; done'
    )
    selected = str(gpu_device).strip().lower().removeprefix("device=")
    gpu_ids = None if selected in {"", "all", "auto", "automatic"} else tuple(
        int(value) for value in selected.split(",")
    )
    command = build_docker_command(
        DockerRunSpec(
            tool=registered_tool("gnina", image=image),
            command=("bash", "-lc", shell_script),
            mounts=(
                DockerMount(run_dir, "/workspace"),
                DockerMount(selection_job.run_dir, "/source", read_only=True),
            ),
            gpu_enabled=True,
            gpu_devices=gpu_ids,
            use_host_user=False,
        )
    )
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "pose_rescoring",
        "workflow": "gnina_rescoring",
        "operation": "rescoring",
        "status": "queued" if enqueue_only else "running",
        "tool": "GNINA score-only",
        "engine": "gnina",
        "parent_run_id": selection_job.run_id,
        "selection_run_id": selection_job.run_id,
        "selection_run_dir": str(selection_job.run_dir),
        "source_run_id": selection_job.parent_run_id,
        "source_engine": selection_job.metadata.get("source_engine", ""),
        "pose_count": len(rows),
        "cnn_model": str(cnn_model).strip() or "default ensemble",
        "cnn_rotation": int(cnn_rotation),
        "coordinates_preserved": True,
        "docker_image": image,
        "gpu_device": str(gpu_device),
        "created_at": now,
        "updated_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "pose_selection": {
                "kind": "job",
                "run_id": selection_job.run_id,
                "task_group": selection_job.task_group,
            },
            "parameters": {
                "engine": "gnina",
                "mode": "score_only",
                "cnn_model": metadata["cnn_model"],
                "cnn_rotation": metadata["cnn_rotation"],
                "gpu_device": str(gpu_device),
            },
        },
    )
    write_registered_command_record(
        run_dir,
        tool_id="gnina",
        commands=(command,),
        image=image,
        selected_gpu_ids=gpu_ids or (),
    )
    if enqueue_only:
        resources = registered_tool("gnina", image=image).resources.to_dict()
        if gpu_ids:
            resources["gpu_ids"] = list(gpu_ids)
        metadata.update(
            {
                "queued_at": now,
                "queued_command": command,
                "gpu_queued": True,
                "resources": resources,
                "worker_finalizer": "gnina_rescoring",
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        write_artifact_manifest(run_dir, [])
        return JobRecord.load(run_dir, task_group=RESCORING_TASK_GROUP)
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    (run_dir / "stdout.log").write_text(process.stdout or "")
    (run_dir / "stderr.log").write_text(process.stderr or "")
    return finalize_gnina_rescoring_job(run_dir, returncode=int(process.returncode))


def queue_gnina_rescoring_job(**parameters: Any) -> JobRecord:
    return run_gnina_rescoring_job(enqueue_only=True, **parameters)


def finalize_boltzina_rescoring_job(
    run_dir: Path, *, returncode: int
) -> JobRecord:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    native = run_dir / "output" / "boltzina_results.csv"
    normalized = run_dir / "rescoring_scores.csv"
    selection_dir = Path(str(metadata["selection_run_dir"]))
    selection_rows = {
        row["pose_id"]: row
        for row in csv.DictReader(
            (selection_dir / "pose_selection.csv").open(
                newline="", errors="replace"
            )
        )
    }
    rows: list[dict[str, Any]] = []
    if native.is_file():
        for row in csv.DictReader(native.open(newline="", errors="replace")):
            pose_id = Path(str(row.get("ligand_name") or "")).stem
            source = selection_rows.get(pose_id, {})
            rows.append(
                {
                    "pose_id": pose_id,
                    "compound_id": source.get("compound_id", ""),
                    "replicate": source.get("replicate", ""),
                    "source_pose_rank": source.get(
                        "source_pose_rank", row.get("docking_rank", "")
                    ),
                    "source_engine": metadata.get("source_engine", ""),
                    "source_score_kcal_mol": source.get(
                        "source_score_kcal_mol", ""
                    ),
                    "boltzina_affinity_log10_ic50_uM": row.get(
                        "affinity_pred_value", ""
                    ),
                    "boltzina_binder_probability": row.get(
                        "affinity_probability_binary", ""
                    ),
                    "affinity_pred_value1": row.get("affinity_pred_value1", ""),
                    "affinity_pred_value2": row.get("affinity_pred_value2", ""),
                    "binder_probability1": row.get(
                        "affinity_probability_binary1", ""
                    ),
                    "binder_probability2": row.get(
                        "affinity_probability_binary2", ""
                    ),
                }
            )
    if rows:
        with normalized.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    expected = int(metadata.get("pose_count") or 0)
    success = returncode == 0 and len(rows) == expected
    artifacts: list[ArtifactRef] = []
    if rows:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir, normalized, "rescoring_scores", role="ranked_scores"
            )
        )
    if native.is_file():
        artifacts.append(
            ArtifactRef.from_path(
                run_dir, native, "native_scores", role="boltzina_native"
            )
        )
    for name, role in (("stdout.log", "stdout"), ("stderr.log", "stderr")):
        path = run_dir / name
        path.touch(exist_ok=True)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir, path, "job_log", role=role, checksum=False
            )
        )
    write_artifact_manifest(run_dir, artifacts)
    error = "" if success else (
        (run_dir / "stderr.log").read_text(errors="replace")[-4000:]
        or f"Boltzina rescored {len(rows)}/{expected} selected poses"
    )
    _write_json(
        run_dir / "result.json",
        {
            "success": success,
            "returncode": returncode,
            "pose_count": expected,
            "rescored_pose_count": len(rows),
            "error": error,
        },
    )
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
    return JobRecord.load(run_dir, task_group=RESCORING_TASK_GROUP)


def run_boltzina_rescoring_job(
    *,
    selection_job: JobRecord,
    boltz_work_dir: Path,
    image: str = "ovolig-boltzina-cu128:latest",
    cache_dir: Path | None = None,
    gpu_device: str = "all",
    batch_size: int = 1,
    seed: int = 1001,
    affinity_mw_correction: bool = False,
    enqueue_only: bool = False,
) -> JobRecord:
    from mn_ligand.workflows.refolding import configured_boltz2_cache_dir

    if selection_job.workflow != "pose_selection":
        raise ValueError("Boltzina rescoring requires an immutable pose selection")
    rows = _selection_rows(selection_job)
    selected_sdf = selection_job.run_dir / "selected_poses.sdf"
    if not selected_sdf.is_file():
        raise ValueError("The selected poses have no coordinate-bearing SDF records")
    boltz_work_dir = Path(boltz_work_dir).resolve()
    if not (boltz_work_dir / "processed" / "manifest.json").is_file():
        raise ValueError("Boltzina requires a compatible processed Boltz-2 work directory")
    cache_dir = Path(cache_dir or configured_boltz2_cache_dir()).resolve()
    run_id = str(uuid4())
    run_dir = runs_root() / RESCORING_TASK_GROUP / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    selected = str(gpu_device).strip().lower().removeprefix("device=")
    gpu_ids = None if selected in {"", "all", "auto", "automatic"} else tuple(
        int(value) for value in selected.split(",")
    )
    native = [
        "--poses",
        "/source/selected_poses.sdf",
        "--receptor",
        "/source/receptor.pdb",
        "--context-dir",
        "/boltz-context",
        "--work-dir",
        "/workspace/boltz_work",
        "--output-dir",
        "/workspace/output",
        "--batch-size",
        str(max(1, int(batch_size))),
        "--seed",
        str(int(seed)),
    ]
    if affinity_mw_correction:
        native.append("--affinity-mw-correction")
    command = build_docker_command(
        DockerRunSpec(
            tool=registered_tool("boltzina", image=image),
            command=tuple(native),
            mounts=(
                DockerMount(run_dir, "/workspace"),
                DockerMount(selection_job.run_dir, "/source", read_only=True),
                DockerMount(boltz_work_dir, "/boltz-context", read_only=True),
                DockerMount(
                    cache_dir / "boltz2_aff.ckpt",
                    "/root/.boltz/boltz2_aff.ckpt",
                    read_only=True,
                ),
                DockerMount(
                    cache_dir / "boltz2_conf.ckpt",
                    "/root/.boltz/boltz2_conf.ckpt",
                    read_only=True,
                ),
            ),
            gpu_enabled=True,
            gpu_devices=gpu_ids,
            shm_size="8g",
            use_host_user=False,
        )
    )
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "pose_rescoring",
        "workflow": "boltzina_rescoring",
        "operation": "rescoring",
        "status": "queued" if enqueue_only else "running",
        "tool": "Boltzina",
        "engine": "boltzina",
        "parent_run_id": selection_job.run_id,
        "selection_run_id": selection_job.run_id,
        "selection_run_dir": str(selection_job.run_dir),
        "source_run_id": selection_job.parent_run_id,
        "source_engine": selection_job.metadata.get("source_engine", ""),
        "pose_count": len(rows),
        "boltz_work_dir": str(boltz_work_dir),
        "batch_size": max(1, int(batch_size)),
        "seed": int(seed),
        "affinity_mw_correction": bool(affinity_mw_correction),
        "coordinates_preserved": True,
        "docker_image": image,
        "gpu_device": str(gpu_device),
        "created_at": now,
        "updated_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "pose_selection": {
                "kind": "job",
                "run_id": selection_job.run_id,
                "task_group": selection_job.task_group,
            },
            "parameters": {
                "engine": "boltzina",
                "mode": "pose_affinity_rescoring",
                "batch_size": metadata["batch_size"],
                "seed": metadata["seed"],
                "affinity_mw_correction": metadata["affinity_mw_correction"],
                "gpu_device": str(gpu_device),
            },
        },
    )
    write_registered_command_record(
        run_dir,
        tool_id="boltzina",
        commands=(command,),
        image=image,
        selected_gpu_ids=gpu_ids or (),
    )
    if enqueue_only:
        resources = registered_tool("boltzina", image=image).resources.to_dict()
        if gpu_ids:
            resources["gpu_ids"] = list(gpu_ids)
        metadata.update(
            {
                "queued_at": now,
                "queued_command": command,
                "gpu_queued": True,
                "resources": resources,
                "worker_finalizer": "boltzina_rescoring",
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        write_artifact_manifest(run_dir, [])
        return JobRecord.load(run_dir, task_group=RESCORING_TASK_GROUP)
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    (run_dir / "stdout.log").write_text(process.stdout or "")
    (run_dir / "stderr.log").write_text(process.stderr or "")
    return finalize_boltzina_rescoring_job(
        run_dir, returncode=int(process.returncode)
    )


def queue_boltzina_rescoring_job(**parameters: Any) -> JobRecord:
    return run_boltzina_rescoring_job(enqueue_only=True, **parameters)
