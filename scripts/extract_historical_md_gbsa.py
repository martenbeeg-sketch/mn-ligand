"""Extract historical MD MM/GBSA and MM/PBSA results from runtime records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


INVENTORY_FIELDS = (
    "record_origin",
    "gbsa_run_id",
    "status",
    "success",
    "active_revision",
    "superseded_by_run_id",
    "created_at",
    "completed_at",
    "workflow_id",
    "production_run_id",
    "prepared_system_run_id",
    "prepared_system_source_run_id",
    "system_id",
    "ligand_key",
    "md_engine",
    "replica",
    "production_duration_ns",
    "production_steps",
    "production_segment_steps",
    "production_prior_steps",
    "production_timestep_fs",
    "continuation_source_run_id",
    "gbsa_method",
    "gbsa_backend",
    "window_start_pct",
    "window_end_pct",
    "stride",
    "frames_analyzed",
    "delta_g_gb_kcal_mol",
    "delta_g_pb_kcal_mol",
    "delta_mm_kcal_mol",
    "delta_gbsa_kcal_mol",
    "delta_nonpolar_kcal_mol",
    "forcefield_method",
    "protein_forcefield_method",
    "water_model",
    "trajectory_path",
    "topology_path",
    "result_json",
    "summary_json",
    "detail_artifacts",
    "error",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _nested_delta(payload: dict[str, Any], section: str) -> dict[str, Any]:
    value = payload.get(section)
    if not isinstance(value, dict):
        return {}
    delta = value.get("delta")
    return delta if isinstance(delta, dict) else {}


def _duration_ns(
    production_input: dict[str, Any],
    workflow_metadata: dict[str, Any],
) -> float | None:
    for key in ("cumulative_target_duration_ns", "production_length_ns"):
        try:
            value = float(production_input.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    parameters = workflow_metadata.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    production = parameters.get("production")
    production = production if isinstance(production, dict) else {}
    try:
        workflow_duration = float(production.get("production_length_ns"))
    except (TypeError, ValueError):
        workflow_duration = 0.0
    if workflow_duration > 0:
        return workflow_duration
    try:
        steps = float(
            production_input.get("cumulative_target_steps")
            or production_input.get("production_steps")
        )
    except (TypeError, ValueError):
        return None
    timestep = production_input.get("production_timestep_fs")
    if timestep is None:
        # Historical OpenMM jobs predate the explicit timestep field and used
        # the then-standard 2 fs integration step.
        timestep = 2.0
    try:
        return steps * float(timestep) / 1_000_000.0
    except (TypeError, ValueError):
        return None


def _artifact_paths(run_dir: Path) -> str:
    names = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if any(token in lower for token in ("mmgbsa", "mmpbsa", "final_results")):
            names.append(path.relative_to(run_dir).as_posix())
    return ";".join(sorted(names))


def _base_row(
    *,
    runs_root: Path,
    production_run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    production_dir = runs_root / "bound-ligand-md" / production_run_id
    production_metadata = _read_json(production_dir / "metadata.json")
    production_input = _read_json(production_dir / "input.json")
    production_result = _read_json(production_dir / "result.json")
    prepared_system_run_id = str(
        production_metadata.get("md_system_prep_run_id")
        or production_metadata.get("parent_run_id")
        or production_input.get("source_md_system_prep_run_id")
        or ""
    )
    preparation_dir = runs_root / "md-system-prep" / prepared_system_run_id
    preparation_metadata = _read_json(preparation_dir / "metadata.json")
    preparation_input = _read_json(preparation_dir / "input.json")
    prepared_system_source_run_id = str(
        preparation_metadata.get("source_target_run_id")
        or preparation_metadata.get("structure_run_id")
        or preparation_metadata.get("parent_run_id")
        or ""
    )
    workflow_id = str(
        production_metadata.get("workflow_id")
        or production_metadata.get("repeat_group_id")
        or ""
    )
    workflow_metadata = _read_json(
        runs_root / "workflows" / workflow_id / "metadata.json"
    )
    total_steps = (
        production_input.get("cumulative_target_steps")
        or production_input.get("production_steps")
    )
    base = {
        "production_run_id": production_run_id,
        "prepared_system_run_id": prepared_system_run_id,
        "prepared_system_source_run_id": prepared_system_source_run_id,
        "system_id": str(
            production_input.get("pdb_id")
            or preparation_input.get("pdb_id")
            or production_metadata.get("pdb_id")
            or ""
        ),
        "ligand_key": str(
            production_input.get("ligand_key")
            or preparation_input.get("ligand_key")
            or production_metadata.get("ligand_key")
            or ""
        ),
        "md_engine": str(
            production_metadata.get("md_engine")
            or production_input.get("md_engine")
            or production_result.get("engine")
            or "openmm"
        ),
        "replica": (
            production_metadata.get("repeat_index")
            or production_input.get("repeat_index")
            or ""
        ),
        "production_duration_ns": _duration_ns(
            production_input,
            workflow_metadata,
        ),
        "production_steps": total_steps,
        "production_segment_steps": production_input.get("production_steps"),
        "production_prior_steps": production_input.get("production_prior_steps"),
        "production_timestep_fs": (
            production_input.get("production_timestep_fs") or 2.0
        ),
        "continuation_source_run_id": str(
            production_input.get("continuation_source_run_id")
            or production_metadata.get("continuation_of_run_id")
            or ""
        ),
        "forcefield_method": str(
            production_input.get("forcefield_method") or ""
        ),
        "protein_forcefield_method": str(
            production_input.get("protein_forcefield_method") or ""
        ),
        "water_model": str(production_input.get("water_model") or ""),
    }
    return base, production_metadata, production_input, workflow_id, production_result


def _energy_fields(mmgbsa: dict[str, Any]) -> dict[str, Any]:
    delta = mmgbsa.get("delta") if isinstance(mmgbsa.get("delta"), dict) else {}
    gb_delta = _nested_delta(mmgbsa, "gb") or delta
    pb_delta = _nested_delta(mmgbsa, "pb")
    return {
        "delta_g_gb_kcal_mol": gb_delta.get("delta_g_bind_total_kcal_mol"),
        "delta_g_pb_kcal_mol": pb_delta.get("delta_g_bind_total_kcal_mol"),
        "delta_mm_kcal_mol": gb_delta.get("delta_mm_kcal_mol"),
        "delta_gbsa_kcal_mol": gb_delta.get("delta_gbsa_kcal_mol"),
        "delta_nonpolar_kcal_mol": gb_delta.get("delta_nonpolar_kcal_mol"),
    }


def _standalone_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group = runs_root / "md-mmgbsa"
    if not group.is_dir():
        return rows
    for run_dir in sorted(path for path in group.iterdir() if path.is_dir()):
        metadata = _read_json(run_dir / "metadata.json")
        input_payload = _read_json(run_dir / "input.json")
        result = _read_json(run_dir / "result.json")
        mmgbsa = result.get("mmgbsa")
        if not isinstance(mmgbsa, dict):
            mmgbsa = _read_json(run_dir / "mmgbsa_summary.json")
        production_run_id = str(
            metadata.get("source_production_run_id")
            or input_payload.get("source_production_run_id")
            or result.get("source_production_run_id")
            or ""
        )
        base, _, _, workflow_id, _ = _base_row(
            runs_root=runs_root,
            production_run_id=production_run_id,
        )
        parameters = metadata.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        execution = mmgbsa.get("execution")
        execution = execution if isinstance(execution, dict) else {}
        method_metadata = mmgbsa.get("metadata")
        method_metadata = method_metadata if isinstance(method_metadata, dict) else {}
        error = str(result.get("error") or metadata.get("error") or "")
        row = {
            **base,
            "record_origin": "standalone md-mmgbsa",
            "gbsa_run_id": run_dir.name,
            "status": str(metadata.get("status") or "unknown"),
            "success": bool(result.get("success") is True),
            "active_revision": not bool(metadata.get("superseded_by_run_id")),
            "superseded_by_run_id": str(metadata.get("superseded_by_run_id") or ""),
            "created_at": str(metadata.get("created_at") or ""),
            "completed_at": str(metadata.get("completed_at") or ""),
            "workflow_id": str(metadata.get("workflow_id") or workflow_id),
            "gbsa_method": str(mmgbsa.get("method") or ""),
            "gbsa_backend": str(parameters.get("backend") or ""),
            "window_start_pct": parameters.get("start_pct"),
            "window_end_pct": parameters.get("end_pct"),
            "stride": parameters.get("stride"),
            "frames_analyzed": (
                execution.get("mmpbsa_frame_count")
                or method_metadata.get("n_frames_analyzed")
            ),
            "trajectory_path": str(mmgbsa.get("trajectory_path") or ""),
            "topology_path": str(mmgbsa.get("topology_path") or ""),
            "result_json": str(run_dir / "result.json"),
            "summary_json": str(run_dir / "mmgbsa_summary.json"),
            "detail_artifacts": _artifact_paths(run_dir),
            "error": error,
            **_energy_fields(mmgbsa),
        }
        rows.append(row)
    return rows


def _embedded_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group = runs_root / "bound-ligand-md"
    if not group.is_dir():
        return rows
    for production_dir in sorted(path for path in group.iterdir() if path.is_dir()):
        result = _read_json(production_dir / "result.json")
        mmgbsa = result.get("mmgbsa")
        if not isinstance(mmgbsa, dict) or mmgbsa.get("status") != "success":
            continue
        base, production_metadata, production_input, workflow_id, _ = _base_row(
            runs_root=runs_root,
            production_run_id=production_dir.name,
        )
        method_metadata = mmgbsa.get("metadata")
        method_metadata = method_metadata if isinstance(method_metadata, dict) else {}
        row = {
            **base,
            "record_origin": "embedded production result",
            "gbsa_run_id": production_dir.name,
            "status": str(production_metadata.get("status") or "unknown"),
            "success": True,
            "active_revision": not bool(
                production_metadata.get("superseded_by_run_id")
            ),
            "superseded_by_run_id": str(
                production_metadata.get("superseded_by_run_id") or ""
            ),
            "created_at": str(production_metadata.get("created_at") or ""),
            "completed_at": str(production_metadata.get("completed_at") or ""),
            "workflow_id": workflow_id,
            "gbsa_method": str(mmgbsa.get("method") or ""),
            "gbsa_backend": str(production_input.get("mmgbsa_backend") or ""),
            "window_start_pct": production_input.get("mmgbsa_start_pct"),
            "window_end_pct": production_input.get("mmgbsa_end_pct"),
            "stride": production_input.get("mmgbsa_stride"),
            "frames_analyzed": method_metadata.get("n_frames_analyzed"),
            "trajectory_path": str(mmgbsa.get("trajectory_path") or ""),
            "topology_path": str(mmgbsa.get("topology_path") or ""),
            "result_json": str(production_dir / "result.json"),
            "summary_json": "",
            "detail_artifacts": _artifact_paths(production_dir),
            "error": "",
            **_energy_fields(mmgbsa),
        }
        rows.append(row)
    return rows


def extract(runs_root: Path) -> list[dict[str, Any]]:
    rows = [*_standalone_rows(runs_root), *_embedded_rows(runs_root)]
    return sorted(
        rows,
        key=lambda row: (str(row.get("created_at") or ""), row["gbsa_run_id"]),
    )


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [
        row
        for row in rows
        if row["success"]
        and row["active_revision"]
        and row.get("delta_g_gb_kcal_mol") is not None
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = (
        "workflow_id",
        "system_id",
        "ligand_key",
        "md_engine",
        "production_duration_ns",
        "gbsa_method",
        "gbsa_backend",
        "window_start_pct",
        "window_end_pct",
        "stride",
    )
    for row in successful:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in grouped.items():
        gb_values = [float(row["delta_g_gb_kcal_mol"]) for row in group]
        pb_values = [
            float(row["delta_g_pb_kcal_mol"])
            for row in group
            if row.get("delta_g_pb_kcal_mol") is not None
        ]
        output = dict(zip(keys, key, strict=True))
        output.update(
            {
                "calculation_count": len(group),
                "replicas": ",".join(
                    str(value)
                    for value in sorted(
                        {row.get("replica") for row in group},
                        key=lambda value: str(value),
                    )
                ),
                "frames_analyzed_total": sum(
                    int(row.get("frames_analyzed") or 0) for row in group
                ),
                "delta_g_gb_mean_kcal_mol": mean(gb_values),
                "delta_g_gb_sample_sd_kcal_mol": (
                    stdev(gb_values) if len(gb_values) > 1 else None
                ),
                "delta_g_pb_mean_kcal_mol": mean(pb_values) if pb_values else None,
                "delta_g_pb_sample_sd_kcal_mol": (
                    stdev(pb_values) if len(pb_values) > 1 else None
                ),
                "gbsa_run_ids": ",".join(row["gbsa_run_id"] for row in group),
                "production_run_ids": ",".join(
                    row["production_run_id"] for row in group
                ),
                "prepared_system_run_ids": ",".join(
                    sorted({row["prepared_system_run_id"] for row in group})
                ),
                "prepared_system_source_run_ids": ",".join(
                    sorted(
                        {row["prepared_system_source_run_id"] for row in group}
                    )
                ),
            }
        )
        summaries.append(output)
    return sorted(
        summaries,
        key=lambda row: (
            str(row.get("system_id") or ""),
            float(row.get("production_duration_ns") or 0.0),
            str(row.get("md_engine") or ""),
            str(row.get("gbsa_method") or ""),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_summary(
    rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
) -> str:
    completed = sum(bool(row["success"]) for row in rows)
    failed = len(rows) - completed
    active = sum(bool(row["active_revision"]) for row in rows)
    lines = [
        "# Historical MD GBSA extraction",
        "",
        f"- Calculations found: {len(rows)}",
        f"- Successful calculations: {completed}",
        f"- Failed calculations retained in audit: {failed}",
        f"- Records not marked superseded: {active}",
        "",
        "The early `single-trajectory_openmm_mmgbsa_script` values are from the "
        "historical OpenMM/OpenFF MM/GBSA-like implementation. They must not be "
        "pooled numerically with AmberTools `ambertools_mmpbsa_mpi` GB/PB values.",
        "",
        "## Recovered simulation systems",
        "",
        "| System | MD engine | MD ns | Workflow/repeat group | Prepared system | Prepared target/source |",
        "|---|---:|---:|---|---|---|",
    ]
    system_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for group in groups:
        key = (
            group.get("system_id"),
            group.get("md_engine"),
            group.get("production_duration_ns"),
            group.get("workflow_id"),
            group.get("prepared_system_run_ids"),
            group.get("prepared_system_source_run_ids"),
        )
        system_rows[key] = group
    for key in sorted(
        system_rows,
        key=lambda value: (
            str(value[0] or ""),
            float(value[2] or 0.0),
            str(value[1] or ""),
        ),
    ):
        system, engine, duration, workflow, prepared, source = key
        lines.append(
            f"| {system or 'unknown'} | {engine or 'unknown'} | "
            f"{float(duration or 0.0):g} | `{workflow}` | `{prepared}` | `{source}` |"
        )
    lines.extend(
        [
            "",
        "## Active successful result groups",
        "",
        "| System | MD engine | MD ns | Method | Window | N | Frames | ΔG GB mean ± SD | ΔG PB mean ± SD |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group in groups:
        gb_mean = float(group["delta_g_gb_mean_kcal_mol"])
        gb_sd = group.get("delta_g_gb_sample_sd_kcal_mol")
        pb_mean = group.get("delta_g_pb_mean_kcal_mol")
        pb_sd = group.get("delta_g_pb_sample_sd_kcal_mol")
        gb_text = f"{gb_mean:.3f}" + (f" ± {float(gb_sd):.3f}" if gb_sd is not None else "")
        pb_text = "—" if pb_mean is None else f"{float(pb_mean):.3f}" + (
            f" ± {float(pb_sd):.3f}" if pb_sd is not None else ""
        )
        lines.append(
            "| {system} | {engine} | {duration:g} | `{method}` | {start}–{end}% / {stride} | "
            "{count} | {frames} | {gb} | {pb} |".format(
                system=group.get("system_id") or "unknown",
                engine=group.get("md_engine") or "unknown",
                duration=float(group.get("production_duration_ns") or 0.0),
                method=group.get("gbsa_method") or "unknown",
                start=group.get("window_start_pct"),
                end=group.get("window_end_pct"),
                stride=group.get("stride"),
                count=group.get("calculation_count"),
                frames=group.get("frames_analyzed_total"),
                gb=gb_text,
                pb=pb_text,
            )
        )
    lines.extend(
        [
            "",
            "Energies are in kcal/mol. `N` counts distinct active calculation "
            "records for the exact method/window configuration; the inventory "
            "CSV contains every run ID, production system, replica, component, "
            "failure, and raw artifact path.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = extract(args.runs_root.resolve())
    groups = _group_rows(rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "historical_md_gbsa_inventory.csv"
    active_path = output_dir / "historical_md_gbsa_active_successful.csv"
    groups_path = output_dir / "historical_md_gbsa_group_summary.csv"
    json_path = output_dir / "historical_md_gbsa_inventory.json"
    markdown_path = output_dir / "README.md"

    _write_csv(inventory_path, rows, INVENTORY_FIELDS)
    _write_csv(
        active_path,
        [row for row in rows if row["success"] and row["active_revision"]],
        INVENTORY_FIELDS,
    )
    group_fields = tuple(groups[0]) if groups else ()
    if group_fields:
        _write_csv(groups_path, groups, group_fields)
    else:
        groups_path.write_text("")
    json_path.write_text(json.dumps({"calculations": rows, "groups": groups}, indent=2) + "\n")
    markdown_path.write_text(_markdown_summary(rows, groups))

    print(f"calculations={len(rows)}")
    print(f"successful={sum(bool(row['success']) for row in rows)}")
    print(f"failed={sum(not bool(row['success']) for row in rows)}")
    print(f"active_successful={sum(bool(row['success']) and bool(row['active_revision']) for row in rows)}")
    print(f"groups={len(groups)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
