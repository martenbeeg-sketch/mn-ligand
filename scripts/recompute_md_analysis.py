#!/usr/bin/env python3
"""Recompute stored MD trajectory analytics and rebuild an analysis run in place."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.workflows import WorkflowRecord
from mn_ligand.ligandx.services.md.workflow.analytics import (
    EquilibrationAnalytics,
    INTERACTION_HOTSPOT_WEIGHTS,
    interaction_hotspot_score,
    ligand_formal_charges_from_sdf_data,
)
from mn_ligand.runtime import resolve_run_dir, runs_root
from mn_ligand.workflows.gromacs_md import parse_gromacs_performance
from mn_ligand.workflows.md_simulation import (
    _complete_analysis,
    source_author_residue_mapping,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".reanalysis.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _resolve_analysis_dir(root: Path, run_id: str) -> Path:
    direct = root / "md-analysis" / run_id
    if direct.is_dir():
        return direct
    workflow_dir = root / "workflows" / run_id
    if not workflow_dir.is_dir():
        raise RuntimeError(f"No MD analysis or workflow run found for {run_id}")
    workflow = _read_json(workflow_dir / "workflow.json")
    candidates = []
    for child in workflow.get("children") or []:
        if (
            not isinstance(child, dict)
            or child.get("task_group") != "md-analysis"
            or child.get("step_id") != "replicate_analysis"
        ):
            continue
        candidate = root / "md-analysis" / str(child.get("run_id") or "")
        if not (candidate / "result.json").is_file():
            continue
        metadata = _read_json(candidate / "metadata.json")
        if metadata.get("status") != "completed":
            continue
        candidates.append(
            (
                bool(child.get("required")),
                str(child.get("attached_at") or ""),
                candidate,
            )
        )
    if not candidates:
        raise RuntimeError(f"Workflow {run_id} has no completed MD analysis child")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _resolve_replica_file(
    replica_dir: Path,
    configured_path: object,
    fallback_names: tuple[str, ...],
) -> Path | None:
    configured_text = str(configured_path or "").strip()
    if configured_text:
        configured = Path(configured_text)
        if configured_text.startswith("/output/"):
            exact = replica_dir / configured_text.removeprefix("/output/")
        elif configured.is_absolute():
            exact = replica_dir / configured.name
        else:
            exact = replica_dir / configured
        if exact.is_file():
            return exact
    configured_name = Path(configured_text).name
    names = tuple(name for name in (configured_name, *fallback_names) if name)
    for name in names:
        for candidate in (
            replica_dir / name,
            replica_dir / replica_dir.name / name,
        ):
            if candidate.is_file():
                return candidate
    for name in names:
        matches = sorted(replica_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _replica_analysis_paths(
    replica_dir: Path,
) -> tuple[str, Path, Path, Path | None]:
    inputs = _read_json(replica_dir / "input.json")
    result = _read_json(replica_dir / "result.json")
    md_result = result.get("md_result") or {}
    output_files = (
        md_result.get("output_files")
        if isinstance(md_result, dict)
        and isinstance(md_result.get("output_files"), dict)
        else {}
    )
    engine = str(inputs.get("md_engine") or "gromacs").strip().lower()
    topology = _resolve_replica_file(
        replica_dir,
        output_files.get("production_pdb"),
        ("production.pdb",),
    )
    trajectory = _resolve_replica_file(
        replica_dir,
        output_files.get("production_trajectory"),
        ("production_whole.xtc", "production.xtc"),
    )
    log_path = _resolve_replica_file(
        replica_dir,
        output_files.get("production_log"),
        ("production.log",),
    )
    if topology is None or trajectory is None:
        raise RuntimeError(
            f"Could not resolve production topology/trajectory in {replica_dir}"
        )
    return engine, topology, trajectory, log_path


def _compute_replica(
    replica_dir_text: str,
    stage_dir_text: str,
    residue_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    replica_dir = Path(replica_dir_text)
    stage_dir = Path(stage_dir_text)
    inputs = _read_json(replica_dir / "input.json")
    prior_result = _read_json(replica_dir / "result.json")
    engine, topology, trajectory, log_path = _replica_analysis_paths(replica_dir)
    ligand_sdf_data = str(inputs.get("ligand_refined_sdf_data") or "")
    if not ligand_sdf_data and str(inputs.get("ligand_data_format") or "").lower() == "sdf":
        ligand_sdf_data = str(inputs.get("ligand_structure_data") or "")
    analytics = EquilibrationAnalytics().compute(
        output_dir=str(replica_dir),
        system_id=str(inputs.get("pdb_id") or replica_dir.name),
        topology_pdb=str(topology),
        production_traj=str(trajectory),
        log_path=str(log_path) if log_path is not None else None,
        ligand_id="LIG",
        production_steps=int(inputs.get("production_steps") or 0),
        production_report_interval=max(
            1, int(inputs.get("production_report_interval") or 2500)
        ),
        dt_ps=float(inputs.get("production_timestep_fs") or 4.0) / 1000.0,
        residue_mapping=(
            residue_mapping
            or (
                inputs.get("residue_mapping")
                if isinstance(inputs.get("residue_mapping"), dict)
                else None
            )
        ),
        ligand_formal_charges=ligand_formal_charges_from_sdf_data(
            ligand_sdf_data
        ),
    )
    if analytics.get("error"):
        raise RuntimeError(str(analytics["error"]))
    contacts = (
        ((analytics.get("structural_dynamics") or {}).get("contacts") or {}).get(
            "residues"
        )
        or []
    )
    required = {
        "contact_backbone_occupancy",
        "contact_sidechain_occupancy",
        "hydrogen_bond_backbone_occupancy",
        "hydrogen_bond_sidechain_occupancy",
    }
    if contacts and any(not required.issubset(row) for row in contacts):
        raise RuntimeError("BB/SC occupancy validation failed")
    prior_analytics = (
        (prior_result.get("md_result") or {}).get("analytics") or {}
        if isinstance(prior_result.get("md_result"), dict)
        else {}
    )
    if engine == "gromacs":
        analytics["performance"] = parse_gromacs_performance(
            log_path or replica_dir / "production.log"
        )
    elif isinstance(prior_analytics.get("performance"), dict):
        analytics["performance"] = prior_analytics["performance"]
    staged_path = stage_dir / f"{replica_dir.name}.analytics.json"
    _write_json_atomic(staged_path, analytics)
    return {
        "run_id": replica_dir.name,
        "contacts": len(contacts),
        "engine": engine,
        "staged_path": str(staged_path),
    }


def _rescore_analytics(analytics: dict[str, Any]) -> int:
    structural = analytics.get("structural_dynamics") or {}
    contacts_payload = structural.get("contacts") or {}
    contacts = contacts_payload.get("residues") or []
    for row in contacts:
        if not isinstance(row, dict):
            continue
        for region, output_field in (
            ("", "binding_importance_score"),
            ("backbone_", "binding_importance_backbone_score"),
            ("sidechain_", "binding_importance_sidechain_score"),
        ):
            row[output_field] = interaction_hotspot_score(
                hydrogen_bond=float(
                    row.get(f"hydrogen_bond_{region}occupancy") or 0.0
                ),
                salt_bridge=float(
                    row.get(f"salt_bridge_{region}occupancy") or 0.0
                ),
                water_bridge=float(
                    row.get(f"water_bridge_{region}occupancy") or 0.0
                ),
                hydrophobic=float(
                    row.get(f"hydrophobic_{region}occupancy") or 0.0
                ),
            )
    contacts_payload["interaction_hotspot_score"] = {
        "description": (
            "Weighted interaction occupancy; not a binding-energy estimate"
        ),
        "weights": dict(INTERACTION_HOTSPOT_WEIGHTS),
    }
    scores_by_residue = {
        str(row.get("residue") or ""): row.get("binding_importance_score")
        for row in contacts
        if isinstance(row, dict)
    }
    for node in (structural.get("interaction_network") or {}).get(
        "nodes", []
    ):
        if not isinstance(node, dict):
            continue
        residue = str(node.get("id") or "")
        if residue in scores_by_residue:
            node["binding_importance_score"] = scores_by_residue[residue]
    return len(contacts)


def _copy_backup(source: Path, backup_root: Path, relative: Path) -> None:
    if not source.is_file():
        return
    destination = backup_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _corrected_residue_mapping(
    analysis_dir: Path,
) -> dict[str, Any] | None:
    metadata = _read_json(analysis_dir / "metadata.json")
    workflow_id = str(metadata.get("workflow_id") or "")
    if not workflow_id:
        return None
    workflow = WorkflowRecord.load(workflow_id)
    workflow_metadata = _read_json(
        Path(runs_root()) / "workflows" / workflow_id / "metadata.json"
    )
    parameters = (
        workflow_metadata.get("parameters")
        if isinstance(workflow_metadata.get("parameters"), dict)
        else {}
    )
    target_run_id = str(
        parameters.get("target_run_id")
        or workflow_metadata.get("target_run_id")
        or ""
    )
    target_dir = resolve_run_dir("selected-complexes", target_run_id)
    if target_dir is not None:
        target_job = JobRecord.load(
            target_dir,
            task_group="selected-complexes",
        )
        prepared = (
            target_job.artifact_manifest.by_type("prepared_complex")
            if target_job.artifact_manifest is not None
            else ()
        )
        if prepared:
            target_path = prepared[0].resolve(
                target_job.run_dir,
                must_exist=True,
            )
            if target_path is not None:
                mapping = source_author_residue_mapping(
                    target_job,
                    prepared[0],
                    target_path.read_text(errors="replace"),
                )
                if mapping.get("residues"):
                    return mapping
    if not workflow.inputs:
        return None
    workflow_input = workflow.inputs[0]
    source_dir = resolve_run_dir(
        workflow_input.source_task_group,
        workflow_input.artifact.run_id,
    )
    if source_dir is None:
        return None
    source_job = JobRecord.load(
        source_dir,
        task_group=workflow_input.source_task_group,
    )
    source_path = workflow_input.artifact.resolve(
        source_job.run_dir,
        must_exist=True,
    )
    if source_path is None or source_path.suffix.lower() not in {".pdb", ".ent"}:
        return None
    mapping = source_author_residue_mapping(
        source_job,
        workflow_input.artifact,
        source_path.read_text(errors="replace"),
    )
    return mapping if mapping.get("residues") else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_run_id")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--scores-only",
        action="store_true",
        help="Recalculate hotspot scores from stored interaction occupancies",
    )
    args = parser.parse_args()

    root = Path(runs_root())
    analysis_dir = _resolve_analysis_dir(root, args.analysis_run_id)
    analysis_inputs = _read_json(analysis_dir / "input.json")
    corrected_mapping = _corrected_residue_mapping(analysis_dir)
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
        raise RuntimeError("The analysis run has no production_run_ids")
    production_dirs = [root / "bound-ligand-md" / value for value in production_ids]
    for path in production_dirs:
        if not (path / "result.json").is_file():
            raise RuntimeError(f"Missing production result: {path / 'result.json'}")
        _replica_analysis_paths(path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage_dir = analysis_dir / ".reanalysis" / stamp
    stage_dir.mkdir(parents=True, exist_ok=False)
    completed: list[dict[str, Any]] = []
    if args.scores_only:
        print(f"Rescoring {len(production_dirs)} stored replica analyses")
        for path in production_dirs:
            result = _read_json(path / "result.json")
            analytics = (result.get("md_result") or {}).get("analytics")
            if not isinstance(analytics, dict):
                raise RuntimeError(f"Missing stored analytics for {path.name}")
            count = _rescore_analytics(analytics)
            staged_path = stage_dir / f"{path.name}.analytics.json"
            _write_json_atomic(staged_path, analytics)
            completed.append(
                {
                    "run_id": path.name,
                    "contacts": count,
                    "engine": _replica_analysis_paths(path)[0],
                    "staged_path": str(staged_path),
                }
            )
    else:
        print(
            f"Recomputing {len(production_dirs)} replicas with "
            f"{args.workers} workers"
        )
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(production_dirs))
        ) as pool:
            futures = {
                pool.submit(
                    _compute_replica,
                    str(path),
                    str(stage_dir),
                    corrected_mapping,
                ): (
                    path.name
                )
                for path in production_dirs
            }
            for future in as_completed(futures):
                item = future.result()
                completed.append(item)
                print(
                    f"  {item['run_id']} ({item['engine']}): "
                    f"{item['contacts']} contact residues"
                )
    if args.scores_only:
        for item in completed:
            print(
                f"  {item['run_id']} ({item['engine']}): "
                f"{item['contacts']} contact residues"
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
    matrix = report.get("contact_matrix") or {}
    required_matrix = {
        "contact_backbone_occupancy",
        "contact_sidechain_occupancy",
        "hydrogen_bond_backbone_occupancy",
        "hydrogen_bond_sidechain_occupancy",
    }
    if not required_matrix.issubset(matrix):
        raise RuntimeError("Aggregate BB/SC matrix validation failed")
    print(f"Updated {analysis_dir}")
    print(f"Backup {backup_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
