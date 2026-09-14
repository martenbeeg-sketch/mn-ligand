#!/usr/bin/env python3
"""Queue focused-pose validation and interaction analysis for one campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from mn_ligand.core.jobs import JobRecord, iter_job_records
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.interaction_analysis import (
    INTERACTION_ENGINES,
    queue_interaction_analysis_job,
)
from mn_ligand.workflows.pose_validation import (
    COMPATIBLE_WORKFLOWS,
    POSE_VALIDATION_INVENTORY_SCHEMA_VERSION,
    POSE_VALIDATION_SELECTION_POLICY,
    cached_pose_validation_candidates,
    queue_pose_validation_job,
)


CAMPAIGN_ID = "0959d315-68f8-4ae7-a534-9726d919a187"
CAMPAIGN_LABEL = "4lnw_3gws_Giorgias_ten_compounds_20260805"
ANALYSIS_WORKFLOWS = {
    "PoseBusters": "posebusters_validation",
    "Native MD geometry": "native_md_geometry_interactions",
    "PLIP": "plip_interactions",
    "PandaMap": "pandamap_interactions",
}


def _campaign_sources() -> tuple[list[JobRecord], list[JobRecord]]:
    compatible: list[JobRecord] = []
    excluded: list[JobRecord] = []
    for job in iter_job_records(runs_root()):
        if job.metadata.get("launch_campaign_id") != CAMPAIGN_ID:
            continue
        if job.tool == "AlphaFast MSA":
            continue
        if job.status == "completed" and job.workflow in COMPATIBLE_WORKFLOWS:
            compatible.append(job)
        else:
            excluded.append(job)
    compatible.sort(key=lambda job: (job.parent_run_id, job.tool, job.run_id))
    excluded.sort(key=lambda job: (job.tool, job.run_id))
    return compatible, excluded


def _existing_children(source_ids: set[str]) -> set[tuple[str, str]]:
    combinations: set[tuple[str, str]] = set()
    workflows = set(ANALYSIS_WORKFLOWS.values())
    for job in iter_job_records(runs_root()):
        if job.parent_run_id in source_ids and job.workflow in workflows:
            combinations.add((job.parent_run_id, job.workflow))
    return combinations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    sources, excluded = _campaign_sources()
    # The campaign initially had 12 pose-bearing engine runs.  A targeted AF3
    # recovery adds a thirteenth source containing only the compound that the
    # original AF3 batch silently skipped.
    if len(sources) != 13:
        raise RuntimeError(
            f"Expected 13 pose-bearing campaign results after AF3 recovery; "
            f"found {len(sources)}"
        )
    source_ids = {job.run_id for job in sources}
    existing = _existing_children(source_ids)
    inventories: dict[str, list[dict[str, object]]] = {}
    for source in sources:
        rows = cached_pose_validation_candidates(source)
        if not rows:
            raise RuntimeError(
                f"Campaign source {source.run_id} ({source.tool}) has no poses"
            )
        inventories[source.run_id] = rows

    plan: list[dict[str, object]] = []
    for source in sources:
        target = (
            "3GWS"
            if source.parent_run_id
            == "3ece6397-b6a1-48a9-802a-39d6974ae1ba"
            else "4LNW"
        )
        missing = [
            label
            for label, workflow in ANALYSIS_WORKFLOWS.items()
            if (source.run_id, workflow) not in existing
        ]
        plan.append(
            {
                "target": target,
                "source_job": source.metadata.get("job_code"),
                "engine": source.tool,
                "poses": len(inventories[source.run_id]),
                "analyses_to_queue": missing,
            }
        )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "campaign_label": CAMPAIGN_LABEL,
                "pose_bearing_sources": len(sources),
                "focused_poses": sum(
                    len(rows) for rows in inventories.values()
                ),
                "excluded": [
                    {
                        "job": job.metadata.get("job_code"),
                        "tool": job.tool,
                        "workflow": job.workflow,
                        "reason": "no compatible predicted 3D pose inventory",
                    }
                    for job in excluded
                ],
                "plan": plan,
                "jobs_to_queue": sum(
                    len(item["analyses_to_queue"]) for item in plan
                ),
                "launch": args.launch,
            },
            indent=2,
        )
    )
    if not args.launch:
        return

    queued: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for source in sources:
        rows = inventories[source.run_id]
        target = (
            "3GWS"
            if source.parent_run_id
            == "3ece6397-b6a1-48a9-802a-39d6974ae1ba"
            else "4LNW"
        )
        if (
            source.run_id,
            ANALYSIS_WORKFLOWS["PoseBusters"],
        ) not in existing:
            try:
                job = queue_pose_validation_job(source, selected_rows=rows)
                queued.append(
                    {
                        "target": target,
                        "source": str(source.metadata.get("job_code") or ""),
                        "analysis": "PoseBusters",
                        "job": str(job.metadata.get("job_code") or ""),
                    }
                )
            except (OSError, TypeError, ValueError) as exc:
                failures.append(
                    {
                        "source": source.run_id,
                        "analysis": "PoseBusters",
                        "error": str(exc),
                    }
                )
        for engine in INTERACTION_ENGINES:
            workflow = ANALYSIS_WORKFLOWS[engine]
            if (source.run_id, workflow) in existing:
                continue
            try:
                job = queue_interaction_analysis_job(
                    source,
                    engine=engine,
                    selected_rows=rows,
                    selection_schema_version=(
                        POSE_VALIDATION_INVENTORY_SCHEMA_VERSION
                    ),
                    selection_policy=POSE_VALIDATION_SELECTION_POLICY,
                    source_inventory_kind="prediction_poses",
                )
                queued.append(
                    {
                        "target": target,
                        "source": str(source.metadata.get("job_code") or ""),
                        "analysis": engine,
                        "job": str(job.metadata.get("job_code") or ""),
                    }
                )
            except (OSError, TypeError, ValueError) as exc:
                failures.append(
                    {
                        "source": source.run_id,
                        "analysis": engine,
                        "error": str(exc),
                    }
                )
    print(json.dumps({"queued": queued, "failures": failures}, indent=2))
    if failures:
        raise RuntimeError(f"Some campaign analyses could not be queued: {failures}")


if __name__ == "__main__":
    main()
