#!/usr/bin/env python3
"""Queue the missing repaired-3GWS half of the active 4LNW/3GWS campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from mn_ligand.app.pages.discover_inputs import ArtifactChoice
from mn_ligand.app.pages.docking_cofolding import (
    ALL_ENGINES,
    _queue_target_engine_jobs,
)
from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.provenance import COMPOUND_DATASET_CAMPAIGN_PURPOSE
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.refolding import (
    configured_alphafold3_reference_paths,
    configured_nesso_reference_paths,
)
from mn_ligand.workflows.target_orientation import (
    ligand_longest_axis_transform,
    reusable_axis_aligned_target_job,
)


CAMPAIGN_ID = "0959d315-68f8-4ae7-a534-9726d919a187"
CAMPAIGN_LABEL = "4lnw_3gws_Giorgias_ten_compounds_20260805"
TARGET_RUN_ID = "6f5a529f-93f4-468f-a4c2-7c4a99abd4bf"
EXPECTED_ORIENTATION_ID = "3ece6397-b6a1-48a9-802a-39d6974ae1ba"
SELECTION_RUN_ID = "02292ba6-56bc-4592-a44f-9de8ea38de88"
MSA_RUN_ID = "6885ac8c-4199-4e1a-a586-db108ad15cd7"


def _job(run_id: str) -> JobRecord:
    matches = [
        path
        for group in runs_root().iterdir()
        if (path := group / run_id).is_dir()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one run directory for {run_id}; found {matches}"
        )
    return JobRecord.load(matches[0], task_group=matches[0].parent.name)


def _existing_campaign_target_jobs() -> list[tuple[str, str, str]]:
    existing: list[tuple[str, str, str]] = []
    for metadata_path in runs_root().glob("*/*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
            payload = json.loads((metadata_path.parent / "input.json").read_text())
        except (OSError, TypeError, ValueError):
            continue
        if metadata.get("launch_campaign_id") != CAMPAIGN_ID:
            continue
        target = payload.get("target_artifact") or payload.get("target") or {}
        if not isinstance(target, dict):
            continue
        if str(target.get("run_id") or "") != EXPECTED_ORIENTATION_ID:
            continue
        existing.append(
            (
                str(metadata.get("job_code") or ""),
                str(metadata.get("tool") or metadata.get("workflow") or ""),
                str(metadata.get("status") or ""),
            )
        )
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    target_job = _job(TARGET_RUN_ID)
    target_artifact = target_job.artifact_manifest.by_type(
        "prepared_receptor"
    )[0]
    ligand_artifact = target_job.artifact_manifest.by_type(
        "prepared_ligand_set"
    )[0]
    target_path = target_artifact.resolve(target_job.run_dir, must_exist=True)
    ligand_path = ligand_artifact.resolve(target_job.run_dir, must_exist=True)
    if target_path is None or ligand_path is None:
        raise RuntimeError("The selected 3GWS target or T3 ligand is unavailable")

    orientation = reusable_axis_aligned_target_job(
        source_artifact=target_artifact,
        axis_ligand_artifact=ligand_artifact,
    )
    if orientation is None or orientation.run_id != EXPECTED_ORIENTATION_ID:
        raise RuntimeError(
            "The validated repaired 3GWS orientation 39D69 was not resolved"
        )
    existing = _existing_campaign_target_jobs()
    if existing:
        raise RuntimeError(
            "The campaign already contains 3GWS engine jobs: " + repr(existing)
        )

    selection_job = _job(SELECTION_RUN_ID)
    selection_artifact = selection_job.artifact_manifest.by_type(
        "compound_set"
    )[0]
    selection_path = selection_artifact.resolve(
        selection_job.run_dir, must_exist=True
    )
    if selection_path is None:
        raise RuntimeError("The campaign compound selection is unavailable")
    msa_job = _job(MSA_RUN_ID)
    if msa_job.status != "completed":
        raise RuntimeError(f"Shared MSA job DB108 is {msa_job.status}")

    transform = ligand_longest_axis_transform(ligand_path)
    center = tuple(float(value) for value in transform["ligand_box"]["center"])
    context = {
        "target": ArtifactChoice(job=target_job, artifact=target_artifact),
        "pocket": None,
        "associated_ligand": ligand_path,
        "associated_ligand_artifact": ligand_artifact,
        "alignment_transform": transform,
        "center": center,
        "size": (35.0, 20.0, 20.0),
        "box_mode": "fixed",
        "box_padding": None,
    }
    af3_db, af3_weights, af3_msa = configured_alphafold3_reference_paths()
    nesso_checkpoint, nesso_ccd, nesso_esm = configured_nesso_reference_paths()
    config = {
        "gpu_device": "all",
        "docking_gpu_device": "all",
        "structure_gpu_device": "all",
        "docking_mode": "classic",
        "search_mode": "detail",
        "exhaustiveness": 30,
        "poses": 10,
        "use_scrub": True,
        "scrub_ph": 7.4,
        "scrub_skip_tautomer": True,
        "campaign_replicates": 3,
        "campaign_seed": 1001,
        "maximum_compounds": 0,
        "extra_args_text": "",
        "openvs_protocol": "vsh",
        "openvs_reference_mode": "reference_guided",
        "openvs_workers": 16,
        "openvs_ph": 7.4,
        "openvs_conformers": 20,
        "openvs_steps": 2000,
        "openvs_padding": 4.0,
        "openvs_cluster": 2.0,
        "boltz_max": 10,
        "boltz_recycles": 3,
        "boltz_steps": 200,
        "boltz_samples": 5,
        "af3_db_dir": af3_db,
        "af3_weights_dir": af3_weights,
        "af3_msa_dir": af3_msa,
        "af3_max": 10,
        "af3_batch": 1,
        "af3_recycles": 10,
        "nesso_checkpoint": nesso_checkpoint,
        "nesso_ccd": nesso_ccd,
        "nesso_esm_cache": nesso_esm,
        "nesso_max": 10,
        "nesso_recycles": 5,
        "nesso_refine": 22,
        "nesso_tokens": 256,
        "nesso_affinity": 15,
    }
    summary = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_label": CAMPAIGN_LABEL,
        "target": "3GWS · 7C4A9",
        "validated_orientation": "39D69",
        "selection": "9DE8E · 10 compounds",
        "shared_msa": "DB108 · completed",
        "engines": list(ALL_ENGINES),
        "center": center,
        "size": context["size"],
        "replicates": 3,
        "seed_start": 1001,
    }
    print(json.dumps(summary, indent=2))
    if not args.launch:
        return

    queued, failures, orientation_note = _queue_target_engine_jobs(
        context=context,
        engines=list(ALL_ENGINES),
        selected_compounds=(
            ArtifactChoice(job=selection_job, artifact=selection_artifact),
        ),
        compound_paths=[selection_path],
        reference=None,
        target_ligand_only=False,
        launch_campaign_id=CAMPAIGN_ID,
        launch_campaign_label=CAMPAIGN_LABEL,
        campaign_purpose=COMPOUND_DATASET_CAMPAIGN_PURPOSE,
        config=config,
        msa_dependency_ids=(MSA_RUN_ID,),
    )
    print(
        json.dumps(
            {
                "queued": queued,
                "failures": failures,
                "orientation": orientation_note,
            },
            indent=2,
        )
    )
    if failures or len(queued) != len(ALL_ENGINES):
        raise RuntimeError(
            f"Incomplete 3GWS launch: queued={queued}; failures={failures}"
        )


if __name__ == "__main__":
    main()
