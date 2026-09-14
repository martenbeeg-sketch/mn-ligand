#!/usr/bin/env python3
"""Export per-replica mandatory-residue hydrogen-bond occupancies."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

if __package__:
    from scripts.export_md_endpoint_table import (
        _duration_ns,
        _read_json,
        _required_analysis,
        _workflow_identity,
    )
    from scripts.recompute_md_analysis import _corrected_residue_mapping
else:
    from export_md_endpoint_table import (
        _duration_ns,
        _read_json,
        _required_analysis,
        _workflow_identity,
    )
    from recompute_md_analysis import _corrected_residue_mapping


MANDATORY_RESIDUE = {"4LNW": ("SER", 277), "3GWS": ("ASN", 331)}


def _internal_residue(
    analysis_dir: Path,
    target: str,
) -> tuple[str, int]:
    residue_name, author_number = MANDATORY_RESIDUE[target]
    mapping = _corrected_residue_mapping(analysis_dir) or {}
    matches = [
        row
        for row in mapping.get("residues") or []
        if isinstance(row, dict)
        and str(row.get("native_residue_name") or "").upper() == residue_name
        and int(row.get("native_residue_number") or -1) == author_number
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one mapping for {residue_name}{author_number} in {analysis_dir}; "
            f"found {len(matches)}"
        )
    return residue_name, int(matches[0]["structure_residue_number"])


def _mandatory_contact(
    contacts: list[dict[str, Any]],
    *,
    target_name: str,
    author_number: int,
    internal_name: str,
    internal_number: int,
) -> dict[str, Any]:
    author_pattern = re.compile(rf"^{target_name}{author_number}\b")
    internal_pattern = re.compile(rf"^{internal_name}{internal_number}\b")
    author_hits = [
        row for row in contacts if author_pattern.search(str(row.get("residue") or ""))
    ]
    if len(author_hits) == 1:
        return author_hits[0]
    internal_hits = [
        row for row in contacts if internal_pattern.search(str(row.get("residue") or ""))
    ]
    if len(internal_hits) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve {target_name}{author_number}; "
            f"author hits={len(author_hits)}, internal hits={len(internal_hits)}"
        )
    return internal_hits[0]


def collect_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for workflow_dir in (root / "workflows").iterdir():
        workflow_path = workflow_dir / "workflow.json"
        metadata_path = workflow_dir / "metadata.json"
        if not workflow_path.is_file() or not metadata_path.is_file():
            continue
        workflow = _read_json(workflow_path)
        parameters = workflow.get("parameters") or {}
        if workflow.get("workflow_type") not in {"md_simulation", "md-simulation"}:
            continue
        if workflow.get("status") != "completed" or _duration_ns(parameters) < 99.0:
            continue
        analysis = _required_analysis(root, workflow)
        if analysis is None:
            continue
        analysis_dir, analysis_result = analysis
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
        target = identity["target"]
        if target not in MANDATORY_RESIDUE:
            continue
        residue_name, author_number = MANDATORY_RESIDUE[target]
        internal_name, internal_number = _internal_residue(analysis_dir, target)
        md_job = str(_read_json(metadata_path).get("job_code") or "")
        duration_by_replica = {
            int(row.get("replica") or index): float((row.get("time_ns") or [0.0])[-1])
            for index, row in enumerate(series, start=1)
        }
        for index, replica in enumerate(replicas, start=1):
            replica_index = int(replica.get("replica") or index)
            production_id = str(replica.get("run_id") or "")
            result = _read_json(root / "bound-ligand-md" / production_id / "result.json")
            contacts = (
                (((result.get("md_result") or {}).get("analytics") or {}).get(
                    "structural_dynamics"
                ) or {}).get("contacts")
                or {}
            ).get("residues") or []
            contact = _mandatory_contact(
                contacts,
                target_name=residue_name,
                author_number=author_number,
                internal_name=internal_name,
                internal_number=internal_number,
            )
            rows.append(
                {
                    "md_job": md_job,
                    "target": target,
                    "compound": identity["compound"],
                    "source": identity["source"],
                    "mandatory_hbond": f"{residue_name}{author_number} (BB+SC)",
                    "replica": replica_index,
                    "analyzed_ns": duration_by_replica[replica_index],
                    "hbond_bb_or_sc_occupancy": contact.get(
                        "hydrogen_bond_occupancy"
                    ),
                    "hbond_backbone_occupancy": contact.get(
                        "hydrogen_bond_backbone_occupancy"
                    ),
                    "hbond_sidechain_occupancy": contact.get(
                        "hydrogen_bond_sidechain_occupancy"
                    ),
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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
        raise RuntimeError("No mandatory-interaction rows found")
    _write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} replica rows to {args.output_csv}")

    if args.wide_output is not None:
        grouped: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
        for row in rows:
            key = tuple(
                str(row[field])
                for field in ("md_job", "target", "compound", "source", "mandatory_hbond")
            )
            output = grouped.setdefault(
                key,
                {
                    "md_job": key[0],
                    "target": key[1],
                    "compound": key[2],
                    "source": key[3],
                    "mandatory_hbond": key[4],
                },
            )
            replica = int(row["replica"])
            output[f"bb_or_sc_repeat_{replica}"] = row["hbond_bb_or_sc_occupancy"]
            output[f"backbone_repeat_{replica}"] = row["hbond_backbone_occupancy"]
            output[f"sidechain_repeat_{replica}"] = row["hbond_sidechain_occupancy"]
        fields = [
            "md_job",
            "target",
            "compound",
            "source",
            "mandatory_hbond",
            "bb_or_sc_repeat_1",
            "bb_or_sc_repeat_2",
            "bb_or_sc_repeat_3",
            "backbone_repeat_1",
            "backbone_repeat_2",
            "backbone_repeat_3",
            "sidechain_repeat_1",
            "sidechain_repeat_2",
            "sidechain_repeat_3",
        ]
        wide_rows = sorted(
            grouped.values(),
            key=lambda row: (row["target"], row["compound"], row["md_job"]),
        )
        if any(set(fields) - set(row) for row in wide_rows):
            raise RuntimeError("At least one MD job does not contain all three replicas")
        _write_csv(args.wide_output, [{field: row[field] for field in fields} for row in wide_rows])
        print(f"Wrote {len(wide_rows)} compact job rows to {args.wide_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
