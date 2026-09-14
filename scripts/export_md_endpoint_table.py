#!/usr/bin/env python3
"""Export per-replica GBSA/PBSA totals for completed 100 ns MD workflows."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


def _target_from_text(value: object) -> str:
    match = re.search(r"\b(4LNW|3GWS)\b", str(value or ""), re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _selected_complex_metadata(root: Path, run_id: object) -> dict[str, Any]:
    value = str(run_id or "").strip()
    if not value:
        return {}
    path = root / "selected-complexes" / value / "metadata.json"
    return _read_json(path) if path.is_file() else {}


def _native_ligand_metadata(root: Path, run_id: object) -> dict[str, Any]:
    value = str(run_id or "").strip()
    if not value:
        return {}
    path = root / "target-trimming" / value / "metadata.json"
    if not path.is_file():
        return {}
    metadata = _read_json(path)
    ligands = [row for row in metadata.get("ligands") or [] if isinstance(row, dict)]
    ligand = ligands[0] if ligands else {}
    return {
        "target": str(metadata.get("pdb_id") or ""),
        "compound": str(ligand.get("name") or ligand.get("ccd_id") or "T3"),
        "source": "PDB native ligand",
    }


def _workflow_identity(root: Path, workflow: dict[str, Any]) -> dict[str, str]:
    parameters = workflow.get("parameters") or {}
    selected = _selected_complex_metadata(root, parameters.get("target_run_id"))
    if selected:
        target = (
            _target_from_text(selected.get("complex_dataset_name"))
            or _target_from_text(parameters.get("complex_dataset_name"))
            or _target_from_text(workflow.get("name"))
        )
        compound_id = str(
            selected.get("compound_name")
            or selected.get("compound_id")
            or selected.get("ligand_label")
            or ""
        )
        return {
            "target": target,
            "compound": compound_id,
            "source": str(selected.get("engine") or ""),
        }
    native = _native_ligand_metadata(root, parameters.get("target_run_id"))
    if native:
        return native
    return {
        "target": _target_from_text(workflow.get("name")),
        "compound": "",
        "source": "",
    }


def _required_analysis(
    root: Path, workflow: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[str, Path, dict[str, Any]]] = []
    for child in workflow.get("children") or []:
        if not isinstance(child, dict):
            continue
        if not child.get("required") or child.get("task_group") != "md-analysis":
            continue
        directory = root / "md-analysis" / str(child.get("run_id") or "")
        result_path = directory / "result.json"
        if not result_path.is_file():
            continue
        candidates.append(
            (str(child.get("attached_at") or ""), directory, _read_json(result_path))
        )
    if not candidates:
        return None
    _, directory, result = max(candidates, key=lambda item: item[0])
    return directory, result


def _duration_ns(parameters: dict[str, Any]) -> float:
    production = parameters.get("production") or {}
    steps = int(production.get("production_steps") or 0)
    timestep_fs = float(production.get("production_timestep_fs") or 4.0)
    return steps * timestep_fs / 1_000_000.0


def collect_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    workflows_root = root / "workflows"
    for workflow_dir in workflows_root.iterdir():
        workflow_path = workflow_dir / "workflow.json"
        metadata_path = workflow_dir / "metadata.json"
        if not workflow_path.is_file() or not metadata_path.is_file():
            continue
        workflow = _read_json(workflow_path)
        if workflow.get("workflow_type") not in {"md_simulation", "md-simulation"}:
            continue
        parameters = workflow.get("parameters") or {}
        if workflow.get("status") != "completed" or _duration_ns(parameters) < 99.0:
            continue
        analysis = _required_analysis(root, workflow)
        if analysis is None:
            continue
        _, analysis_result = analysis
        replicas = [
            row for row in analysis_result.get("replicas") or [] if isinstance(row, dict)
        ]
        series = [
            row
            for row in analysis_result.get("replica_series") or []
            if isinstance(row, dict)
        ]
        if len(replicas) != 3 or len(series) != 3:
            continue
        identity = _workflow_identity(root, workflow)
        workflow_metadata = _read_json(metadata_path)
        md_job = str(workflow_metadata.get("job_code") or "")
        md_engine = str(parameters.get("engine") or "")
        duration_by_replica = {
            int(row.get("replica") or index): float((row.get("time_ns") or [0.0])[-1])
            for index, row in enumerate(series, start=1)
        }
        for index, replica in enumerate(replicas, start=1):
            replica_index = int(replica.get("replica") or index)
            endpoint_id = str(replica.get("endpoint_job_id") or "")
            endpoint_path = root / "md-mmgbsa" / endpoint_id / "result.json"
            if not endpoint_path.is_file():
                raise RuntimeError(f"Missing endpoint result for {md_job} replica {replica_index}")
            endpoint = _read_json(endpoint_path)
            mmgbsa = endpoint.get("mmgbsa") or {}
            gb_delta = (mmgbsa.get("gb") or {}).get("delta") or {}
            pb_delta = (mmgbsa.get("pb") or {}).get("delta") or {}
            execution = mmgbsa.get("execution") or {}
            gb_value = gb_delta.get("delta_g_bind_total_kcal_mol")
            pb_value = pb_delta.get("delta_g_bind_total_kcal_mol")
            if gb_value is None or pb_value is None:
                raise RuntimeError(
                    f"Missing GBSA/PBSA total for {md_job} replica {replica_index}"
                )
            rows.append(
                {
                    "md_job": md_job,
                    "target": identity["target"],
                    "compound": identity["compound"],
                    "source": identity["source"],
                    "md_engine": md_engine,
                    "replica": replica_index,
                    "analyzed_ns": duration_by_replica.get(replica_index),
                    "endpoint_frames": execution.get("mmpbsa_frame_count"),
                    "mm_gbsa_delta_g_bind_kcal_mol": gb_value,
                    "mm_pbsa_delta_g_bind_kcal_mol": pb_value,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            str(row["target"]),
            str(row["compound"]),
            str(row["source"]),
            str(row["md_job"]),
            int(row["replica"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--wide-output", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("mn-ligand-workdir/workdir/runs"),
    )
    args = parser.parse_args()
    rows = collect_rows(args.runs_root)
    if not rows:
        raise RuntimeError("No completed three-replica 100 ns MD endpoint results found")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    jobs = {str(row["md_job"]) for row in rows}
    print(f"Wrote {len(rows)} replica rows from {len(jobs)} MD jobs to {args.output_csv}")
    if args.wide_output is not None:
        grouped: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
        for row in rows:
            key = (
                str(row["md_job"]),
                str(row["target"]),
                str(row["compound"]),
                str(row["source"]),
                str(row["md_engine"]),
            )
            output = grouped.setdefault(
                key,
                {
                    "md_job": key[0],
                    "target": key[1],
                    "compound": key[2],
                    "source": key[3],
                    "md_engine": key[4],
                },
            )
            replica = int(row["replica"])
            output[f"mm_gbsa_repeat_{replica}_kcal_mol"] = row[
                "mm_gbsa_delta_g_bind_kcal_mol"
            ]
            output[f"mm_pbsa_repeat_{replica}_kcal_mol"] = row[
                "mm_pbsa_delta_g_bind_kcal_mol"
            ]
        wide_fields = [
            "md_job",
            "target",
            "compound",
            "source",
            "md_engine",
            "mm_gbsa_repeat_1_kcal_mol",
            "mm_gbsa_repeat_2_kcal_mol",
            "mm_gbsa_repeat_3_kcal_mol",
            "mm_pbsa_repeat_1_kcal_mol",
            "mm_pbsa_repeat_2_kcal_mol",
            "mm_pbsa_repeat_3_kcal_mol",
        ]
        wide_rows = sorted(
            grouped.values(),
            key=lambda row: (row["target"], row["compound"], row["md_job"]),
        )
        if any(set(wide_fields) - set(row) for row in wide_rows):
            raise RuntimeError("At least one MD job does not contain all three replicas")
        args.wide_output.parent.mkdir(parents=True, exist_ok=True)
        with args.wide_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=wide_fields)
            writer.writeheader()
            writer.writerows(wide_rows)
        print(f"Wrote {len(wide_rows)} compact job rows to {args.wide_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
