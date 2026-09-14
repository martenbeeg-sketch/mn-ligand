#!/usr/bin/env python3
"""Recompute MD analytics over every exact-continuation trajectory segment.

This is intended for historical OpenMM workflows whose continuation jobs contain
one DCD for the preceding segment at the job root and another DCD for the newly
computed segment below ``<run_id>/``.  The ordinary result points at the latter,
so analyzing it alone covers only the continuation delta.  This script follows
``continuation_source_run_id``, joins the distinct segments per replica in
chronological order, and commits one cumulative analysis per current replica.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mdtraj.formats import DCDTrajectoryFile

from mn_ligand.core.jobs import JobRecord
from mn_ligand.ligandx.services.md.workflow.analytics import (
    EquilibrationAnalytics,
    ligand_formal_charges_from_sdf_data,
)
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.md_simulation import _complete_analysis
if __package__:
    from scripts.recompute_md_analysis import (
        _copy_backup,
        _corrected_residue_mapping,
        _read_json,
        _replica_analysis_paths,
        _resolve_analysis_dir,
        _write_json_atomic,
    )
else:
    from recompute_md_analysis import (
        _copy_backup,
        _corrected_residue_mapping,
        _read_json,
        _replica_analysis_paths,
        _resolve_analysis_dir,
        _write_json_atomic,
    )


def _lineage(replica_dir: Path, root: Path) -> list[Path]:
    """Return completed production segments from oldest to newest."""
    chain: list[Path] = []
    seen: set[str] = set()
    current = replica_dir
    while True:
        if current.name in seen:
            raise RuntimeError(f"Continuation lineage cycle at {current.name}")
        seen.add(current.name)
        chain.append(current)
        payload = _read_json(current / "input.json")
        source_id = str(payload.get("continuation_source_run_id") or "").strip()
        if not source_id:
            break
        current = root / "bound-ligand-md" / source_id
        if not (current / "result.json").is_file():
            raise RuntimeError(f"Missing continuation source result: {current}")
    chain.reverse()
    return chain


def _topology_signature(topology_path: Path) -> tuple[tuple[str, str, int], ...]:
    import mdtraj as md

    topology = md.load_pdb(str(topology_path)).topology
    return tuple(
        (atom.name, atom.residue.name, atom.residue.index)
        for atom in topology.atoms
    )


def _join_dcds(paths: list[Path], output_path: Path) -> list[int]:
    frame_counts: list[int] = []
    with DCDTrajectoryFile(str(output_path), mode="w") as destination:
        for path in paths:
            count = 0
            with DCDTrajectoryFile(str(path), mode="r") as source:
                while True:
                    xyz, cell_lengths, cell_angles = source.read(n_frames=100)
                    if len(xyz) == 0:
                        break
                    destination.write(xyz, cell_lengths, cell_angles)
                    count += len(xyz)
            if count == 0:
                raise RuntimeError(f"Trajectory segment has no frames: {path}")
            frame_counts.append(count)
    return frame_counts


def _join_logs(paths: list[Path], output_path: Path) -> None:
    header: str | None = None
    with output_path.open("w") as destination:
        for path in paths:
            with path.open() as source:
                segment_header = source.readline()
                if not segment_header:
                    continue
                if header is None:
                    header = segment_header
                    destination.write(header)
                elif segment_header != header:
                    raise RuntimeError(f"Production log columns differ in {path}")
                shutil.copyfileobj(source, destination)


def _compute_cumulative_replica(
    replica_dir_text: str,
    root_text: str,
    stage_dir_text: str,
    residue_mapping: dict[str, Any] | None,
) -> dict[str, Any]:
    replica_dir = Path(replica_dir_text)
    root = Path(root_text)
    stage_dir = Path(stage_dir_text)
    segments = _lineage(replica_dir, root)
    if len(segments) < 2:
        raise RuntimeError(f"{replica_dir.name} has no continuation lineage to join")

    resolved = [_replica_analysis_paths(segment) for segment in segments]
    engines = {item[0] for item in resolved}
    if engines != {"openmm"}:
        raise RuntimeError(
            f"Cumulative DCD join currently requires OpenMM; found {sorted(engines)}"
        )
    signatures = [_topology_signature(item[1]) for item in resolved]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError(
            f"Topology atom ordering differs in lineage for {replica_dir.name}"
        )

    work_dir = Path(tempfile.mkdtemp(prefix=f"{replica_dir.name[:8]}-", dir=stage_dir))
    combined_dcd = work_dir / "production-cumulative.dcd"
    combined_log = work_dir / "production-cumulative.log"
    trajectory_paths = [item[2] for item in resolved]
    log_paths = [item[3] for item in resolved if item[3] is not None]
    try:
        frame_counts = _join_dcds(trajectory_paths, combined_dcd)
        log_path: Path | None = None
        if len(log_paths) == len(resolved):
            _join_logs([path for path in log_paths if path is not None], combined_log)
            log_path = combined_log

        inputs = _read_json(replica_dir / "input.json")
        ligand_sdf_data = str(inputs.get("ligand_refined_sdf_data") or "")
        if (
            not ligand_sdf_data
            and str(inputs.get("ligand_data_format") or "").lower() == "sdf"
        ):
            ligand_sdf_data = str(inputs.get("ligand_structure_data") or "")
        total_steps = sum(
            int(_read_json(segment / "input.json").get("production_steps") or 0)
            for segment in segments
        )
        cumulative_steps = int(inputs.get("cumulative_target_steps") or total_steps)
        if total_steps != cumulative_steps:
            raise RuntimeError(
                f"Lineage steps ({total_steps}) do not equal cumulative target "
                f"({cumulative_steps}) for {replica_dir.name}"
            )
        analytics = EquilibrationAnalytics().compute(
            output_dir=str(replica_dir),
            system_id=str(inputs.get("pdb_id") or replica_dir.name),
            topology_pdb=str(resolved[-1][1]),
            production_traj=str(combined_dcd),
            log_path=str(log_path) if log_path is not None else None,
            ligand_id="LIG",
            production_steps=cumulative_steps,
            production_report_interval=max(
                1, int(inputs.get("production_report_interval") or 2500)
            ),
            dt_ps=float(inputs.get("production_timestep_fs") or 4.0) / 1000.0,
            residue_mapping=residue_mapping,
            ligand_formal_charges=ligand_formal_charges_from_sdf_data(
                ligand_sdf_data
            ),
        )
        if analytics.get("error"):
            raise RuntimeError(str(analytics["error"]))
        analytics["cumulative_trajectory"] = {
            "mode": "exact_continuation_lineage_join",
            "segment_run_ids": [segment.name for segment in segments],
            "segment_frame_counts": frame_counts,
            "total_frame_count": sum(frame_counts),
            "total_steps": total_steps,
            "target_duration_ns": (
                total_steps
                * float(inputs.get("production_timestep_fs") or 4.0)
                / 1_000_000.0
            ),
        }
        staged_path = stage_dir / f"{replica_dir.name}.analytics.json"
        _write_json_atomic(staged_path, analytics)
        contacts = (
            ((analytics.get("structural_dynamics") or {}).get("contacts") or {}).get(
                "residues"
            )
            or []
        )
        return {
            "run_id": replica_dir.name,
            "staged_path": str(staged_path),
            "segments": [segment.name for segment in segments],
            "frame_counts": frame_counts,
            "contacts": len(contacts),
        }
    finally:
        shutil.rmtree(work_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_run_id")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    root = Path(runs_root())
    analysis_dir = _resolve_analysis_dir(root, args.analysis_run_id)
    analysis_inputs = _read_json(analysis_dir / "input.json")
    prior_result = _read_json(analysis_dir / "result.json")
    prior_replicas = [
        row for row in prior_result.get("replicas") or [] if isinstance(row, dict)
    ]
    production_ids = [
        str(row["run_id"]) for row in prior_replicas if row.get("run_id")
    ] or [str(value) for value in analysis_inputs.get("production_run_ids") or []]
    endpoint_ids = [
        str(row["endpoint_job_id"])
        for row in prior_replicas
        if row.get("endpoint_job_id")
    ] or [str(value) for value in analysis_inputs.get("endpoint_run_ids") or []]
    if not production_ids:
        raise RuntimeError("The analysis run has no production replicas")
    production_dirs = [root / "bound-ligand-md" / value for value in production_ids]
    corrected_mapping = _corrected_residue_mapping(analysis_dir)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage_dir = analysis_dir / ".cumulative-reanalysis" / stamp
    stage_dir.mkdir(parents=True, exist_ok=False)
    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(production_dirs))) as pool:
        futures = {
            pool.submit(
                _compute_cumulative_replica,
                str(path),
                str(root),
                str(stage_dir),
                corrected_mapping,
            ): path.name
            for path in production_dirs
        }
        for future in as_completed(futures):
            item = future.result()
            completed.append(item)
            print(
                f"{item['run_id']}: joined {item['frame_counts']} frames; "
                f"{item['contacts']} contact residues",
                flush=True,
            )

    backup_root = analysis_dir / "reanalysis-backups" / stamp
    for path in production_dirs:
        for name in ("result.json", "trajectory_analysis.json"):
            _copy_backup(path / name, backup_root, Path("replicas") / path.name / name)
    for name in (
        "result.json",
        "replicate_summary.json",
        "replicate_summary.csv",
        "metadata.json",
        "artifacts.json",
    ):
        _copy_backup(analysis_dir / name, backup_root, Path("analysis") / name)

    staged_by_id = {item["run_id"]: Path(item["staged_path"]) for item in completed}
    for replica_dir in production_dirs:
        analytics = _read_json(staged_by_id[replica_dir.name])
        result = _read_json(replica_dir / "result.json")
        result.setdefault("md_result", {})["analytics"] = analytics
        _write_json_atomic(replica_dir / "trajectory_analysis.json", analytics)
        _write_json_atomic(replica_dir / "result.json", result)

    analysis_job = JobRecord.load(analysis_dir, task_group="md-analysis")
    production_jobs = [
        JobRecord.load(path, task_group="bound-ligand-md") for path in production_dirs
    ]
    endpoint_jobs = [
        JobRecord.load(root / "md-mmgbsa" / value, task_group="md-mmgbsa")
        for value in endpoint_ids
    ]
    _complete_analysis(
        analysis_job,
        production_jobs,
        endpoint_jobs,
        residue_mapping_override=corrected_mapping,
    )
    report = _read_json(analysis_dir / "replicate_summary.json")
    ends = [
        (series.get("time_ns") or [None])[-1]
        for series in report.get("replica_series") or []
    ]
    if len(ends) != len(production_dirs) or any(
        end is None or float(end) < 99.0 for end in ends
    ):
        raise RuntimeError(f"Cumulative duration validation failed: {ends}")
    print(f"Updated {analysis_dir}", flush=True)
    print(f"Cumulative time-axis ends (ns): {ends}", flush=True)
    print(f"Backup {backup_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
