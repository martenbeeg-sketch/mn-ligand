#!/usr/bin/env python3
"""Queue the one missing repaired-3GWS AlphaFold 3 compound on GPU 1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from mn_ligand.core.jobs import JobRecord, iter_job_records
from mn_ligand.core.provenance import COMPOUND_DATASET_CAMPAIGN_PURPOSE
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.compound_preparation import (
    create_docking_parent_selection_job,
)
from mn_ligand.workflows.refolding import (
    DEFAULT_ALPHAFOLD3_IMAGE,
    configured_alphafold3_reference_paths,
    queue_alphafold3_refolding_job,
)


CAMPAIGN_ID = "0959d315-68f8-4ae7-a534-9726d919a187"
CAMPAIGN_LABEL = "4lnw_3gws_Giorgias_ten_compounds_20260805"
FAILED_AF3_RUN_ID = "f7834203-b9cb-445c-8a9a-7a9640b6447f"
TARGET_RUN_ID = "3ece6397-b6a1-48a9-802a-39d6974ae1ba"
SOURCE_IMPORT_RUN_ID = "dc4d2c6e-f1af-4c82-9d4d-c47f361e639b"
SOURCE_SELECTION_RUN_ID = "02292ba6-56bc-4592-a44f-9de8ea38de88"
COMPOUND_ID = "ZINCsE000004Eaoz"
LAUNCH_CONTEXT = f"missing_prediction_recovery:{FAILED_AF3_RUN_ID}:{COMPOUND_ID}"


def _job(run_id: str) -> JobRecord:
    matches = [
        path
        for group in runs_root().iterdir()
        if (path := group / run_id).is_dir()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one job {run_id}; found {matches}")
    return JobRecord.load(matches[0], task_group=matches[0].parent.name)


def _existing_recovery() -> JobRecord | None:
    return next(
        (
            job
            for job in iter_job_records(runs_root(), task_groups=("refolding",))
            if job.workflow == "alphafold3_refolding"
            and job.metadata.get("launch_context") == LAUNCH_CONTEXT
        ),
        None,
    )


def _selected_source_row() -> dict[str, str]:
    selection = _job(SOURCE_SELECTION_RUN_ID)
    artifact = selection.artifact_manifest.by_type("compound_set")[0]
    path = artifact.resolve(selection.run_dir, must_exist=True)
    if path is None:
        raise RuntimeError("The original campaign selection is unavailable")
    with path.open(newline="") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if str(row.get("compound_id") or "") == COMPOUND_ID
        ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one selected row for {COMPOUND_ID}; found {len(rows)}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    existing = _existing_recovery()
    source_row = _selected_source_row()
    summary = {
        "campaign_id": CAMPAIGN_ID,
        "target": "repaired 3GWS · 39D69",
        "compound": COMPOUND_ID,
        "engine": "AlphaFold 3",
        "seeds": [1001, 1002, 1003],
        "samples_per_seed": 5,
        "gpu": 1,
        "existing_recovery": (
            {
                "job": existing.metadata.get("job_code"),
                "status": existing.status,
            }
            if existing is not None
            else None
        ),
        "launch": args.launch,
    }
    print(json.dumps(summary, indent=2))
    if not args.launch or existing is not None:
        return

    source_import = _job(SOURCE_IMPORT_RUN_ID)
    source_artifact = source_import.artifact_manifest.by_type("compound_set")[0]
    parent_row = {
        "docking_parent_id": source_row["docking_parent_id"],
        "representative_compound_id": source_row["compound_id"],
        "representative_product_name": source_row.get(
            "representative_product_name", ""
        ),
        "compound_ids": source_row.get("source_compound_ids", COMPOUND_ID),
        "product_names": source_row.get("source_product_names", ""),
        "cas_numbers": source_row.get("source_cas_numbers", ""),
        "structure_origins": source_row.get("source_structure_origins", ""),
        "source_formulations": source_row.get("source_formulations", ""),
        "standardized_parent_formula": source_row.get("formula", ""),
        "standardized_parent_smiles": source_row["identity_parent_smiles"],
        "representative_source_smiles": source_row["smiles"],
        "parent_formal_charge_before_standardization": source_row.get(
            "modeling_formal_charge", "0"
        ),
        "modeling_smiles": source_row["modeling_smiles"],
    }
    selection = create_docking_parent_selection_job(
        source_job=source_import,
        source_artifact=source_artifact,
        parent_rows=[parent_row],
        selection_mode=(
            f"Recovery of missing AF3 prediction from {FAILED_AF3_RUN_ID}"
        ),
    )
    selection_artifact = selection.artifact_manifest.by_type("compound_set")[0]
    selection_path = selection_artifact.resolve(selection.run_dir, must_exist=True)
    if selection_path is None:
        raise RuntimeError("The one-compound recovery selection was not published")

    target = _job(TARGET_RUN_ID)
    target_artifact = target.artifact_manifest.by_type("prepared_target")[0]
    target_path = target_artifact.resolve(target.run_dir, must_exist=True)
    reference_ligand = target.artifact_manifest.by_type("prepared_ligand_set")[0]
    if target_path is None:
        raise RuntimeError("The repaired 3GWS target is unavailable")
    db_dir, weights_dir, msa_repository_dir = (
        configured_alphafold3_reference_paths()
    )
    recovery = queue_alphafold3_refolding_job(
        target_path=target_path,
        target_artifact=target_artifact,
        compound_paths=[selection_path],
        compound_artifacts=[selection_artifact],
        reference_ligand_artifact=reference_ligand,
        image=DEFAULT_ALPHAFOLD3_IMAGE,
        db_dir=db_dir,
        weights_dir=weights_dir,
        msa_repository_dir=msa_repository_dir,
        gpu_device="1",
        max_compounds=1,
        batch_size=1,
        num_recycles=10,
        model_seed_count=3,
        model_seed_start=1001,
        launch_context=LAUNCH_CONTEXT,
        launch_campaign_id=CAMPAIGN_ID,
        launch_campaign_label=CAMPAIGN_LABEL,
        campaign_purpose=COMPOUND_DATASET_CAMPAIGN_PURPOSE,
    )
    input_names = sorted(path.stem for path in (recovery.run_dir / "inputs").glob("*.json"))
    if input_names != [COMPOUND_ID]:
        raise RuntimeError(f"Unexpected AF3 recovery inputs: {input_names}")
    print(
        json.dumps(
            {
                "queued_job": recovery.metadata.get("job_code"),
                "run_id": recovery.run_id,
                "selection_job": selection.metadata.get("job_code"),
                "input_names": input_names,
                "msa_cache": (
                    json.loads((recovery.run_dir / "input.json").read_text())
                    .get("parameters", {})
                    .get("msa_cache", {})
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
