from __future__ import annotations

import argparse
import json
from pathlib import Path

from mn_ligand.core.artifacts import ArtifactRef
from mn_ligand.core.job_control import request_job_cancellation
from mn_ligand.core.jobs import JobRecord
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.refolding import (
    configured_alphafold3_reference_paths,
    queue_alphafold3_msa_job,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def batch_pending(campaign_id: str) -> dict[str, object]:
    rows: list[tuple[Path, dict]] = []
    for path in runs_root().glob("**/metadata.json"):
        metadata = _read(path)
        if str(metadata.get("launch_campaign_id") or "") == campaign_id:
            rows.append((path, metadata))
    pending = [
        (path, metadata)
        for path, metadata in rows
        if metadata.get("campaign_phase") == "msa_cpu_fallback"
        and metadata.get("status") in {"queued", "running", "cancelled"}
        and metadata.get("cpu_fallback_of_run_id")
    ]
    if len(pending) < 2:
        raise ValueError("At least two queued CPU MSA fallbacks are required for batching")
    sequences = list(
        dict.fromkeys(
            str(sequence)
            for _path, metadata in pending
            for sequence in (
                metadata.get("protein_sequences")
                or [metadata.get("protein_sequence")]
            )
            if str(sequence)
        )
    )
    first_path, first_metadata = pending[0]
    target = ArtifactRef.from_dict(_read(first_path.parent / "input.json")["target"])
    db_dir, weights_dir, repository = configured_alphafold3_reference_paths()
    replacement = queue_alphafold3_msa_job(
        protein_sequences=sequences,
        target_artifact=target,
        db_dir=db_dir,
        weights_dir=weights_dir,
        msa_repository_dir=repository,
        batch_size=len(sequences),
        launch_campaign_id=campaign_id,
        launch_campaign_label=str(first_metadata.get("launch_campaign_label") or ""),
        campaign_purpose=str(first_metadata.get("campaign_purpose") or ""),
        use_gpu=False,
    )
    old_ids = [str(metadata.get("run_id") or path.parent.name) for path, metadata in pending]
    replacement_path = replacement.run_dir / "metadata.json"
    replacement_metadata = _read(replacement_path)
    replacement_metadata["batched_from_run_ids"] = old_ids
    _write(replacement_path, replacement_metadata)
    for path, metadata in pending:
        if metadata.get("status") in {"queued", "running"}:
            request_job_cancellation(
                JobRecord.load(path.parent, task_group=path.parent.parent.name),
                requested_by="cpu_msa_batch_consolidation",
            )
    rewired = 0
    for path, metadata in rows:
        dependencies = [str(value) for value in (metadata.get("depends_on_run_ids") or ())]
        if not any(value in old_ids for value in dependencies):
            continue
        metadata["depends_on_run_ids"] = list(
            dict.fromkeys(
                replacement.run_id if value in old_ids else value
                for value in dependencies
            )
        )
        metadata["msa_cpu_batch_run_id"] = replacement.run_id
        _write(path, metadata)
        rewired += 1
    return {
        "campaign_id": campaign_id,
        "cancelled_individual_jobs": len(old_ids),
        "batched_sequence_count": len(sequences),
        "batch_run_id": replacement.run_id,
        "rewired_structure_jobs": rewired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_id")
    args = parser.parse_args()
    print(json.dumps(batch_pending(args.campaign_id), indent=2))


if __name__ == "__main__":
    main()
