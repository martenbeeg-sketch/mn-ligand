from __future__ import annotations

import argparse
import json
from pathlib import Path

from mn_ligand.core.artifacts import ArtifactRef
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


def recover(campaign_id: str) -> dict[str, object]:
    rows: list[tuple[Path, dict]] = []
    for metadata_path in runs_root().glob("**/metadata.json"):
        metadata = _read(metadata_path)
        if str(metadata.get("launch_campaign_id") or "") == campaign_id:
            rows.append((metadata_path, metadata))
    failed = [
        (path, metadata)
        for path, metadata in rows
        if metadata.get("workflow") == "alphafold3_msa"
        and metadata.get("status") == "failed"
        and "out of memory" in str(metadata.get("error") or metadata.get("stderr_tail") or "").lower()
    ]
    if not failed:
        raise ValueError("No failed CUDA-OOM MSA jobs were found")
    db_dir, weights_dir, repository = configured_alphafold3_reference_paths()
    replacements: dict[str, str] = {}
    for metadata_path, metadata in failed:
        input_payload = _read(metadata_path.parent / "input.json")
        replacement = queue_alphafold3_msa_job(
            protein_sequence=str(metadata.get("protein_sequence") or ""),
            target_artifact=ArtifactRef.from_dict(input_payload["target"]),
            db_dir=db_dir,
            weights_dir=weights_dir,
            msa_repository_dir=repository,
            batch_size=1,
            launch_campaign_id=campaign_id,
            launch_campaign_label=str(metadata.get("launch_campaign_label") or ""),
            campaign_purpose=str(metadata.get("campaign_purpose") or ""),
            use_gpu=False,
        )
        replacement_path = replacement.run_dir / "metadata.json"
        replacement_metadata = _read(replacement_path)
        replacement_metadata["cpu_fallback_of_run_id"] = str(metadata.get("run_id") or metadata_path.parent.name)
        _write(replacement_path, replacement_metadata)
        replacements[str(metadata.get("run_id") or metadata_path.parent.name)] = replacement.run_id

    rewired = 0
    for metadata_path, metadata in rows:
        dependencies = [str(value) for value in (metadata.get("depends_on_run_ids") or ())]
        if not any(value in replacements for value in dependencies):
            continue
        metadata["depends_on_run_ids"] = [replacements.get(value, value) for value in dependencies]
        metadata["msa_cpu_fallback_run_ids"] = list(replacements.values())
        if metadata.get("status") == "blocked":
            metadata["status"] = "queued"
            metadata.pop("completed_at", None)
            metadata.pop("blocked_by_run_ids", None)
            metadata.pop("error", None)
        _write(metadata_path, metadata)
        rewired += 1
    return {
        "campaign_id": campaign_id,
        "cpu_fallback_jobs": len(replacements),
        "rewired_structure_jobs": rewired,
        "replacements": replacements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_id")
    args = parser.parse_args()
    print(json.dumps(recover(args.campaign_id), indent=2))


if __name__ == "__main__":
    main()
