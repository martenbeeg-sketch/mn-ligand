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


def retry_on_gpu(source_run_id: str) -> dict[str, object]:
    source_path = next(runs_root().glob(f"**/{source_run_id}/metadata.json"))
    source = _read(source_path)
    source_input = _read(source_path.parent / "input.json")
    target = ArtifactRef.from_dict(source_input["target"])
    sequences = source.get("protein_sequences") or [source["protein_sequence"]]
    db_dir, weights_dir, repository = configured_alphafold3_reference_paths()
    replacement = queue_alphafold3_msa_job(
        protein_sequences=sequences,
        target_artifact=target,
        db_dir=db_dir,
        weights_dir=weights_dir,
        msa_repository_dir=repository,
        batch_size=len(sequences),
        launch_campaign_id=str(source.get("launch_campaign_id") or ""),
        launch_campaign_label=str(source.get("launch_campaign_label") or ""),
        campaign_purpose=str(source.get("campaign_purpose") or ""),
        use_gpu=True,
    )
    replacement_path = replacement.run_dir / "metadata.json"
    replacement_metadata = _read(replacement_path)
    replacement_metadata["gpu_split_memory_limit"] = "16G"
    replacement_metadata["retry_of_run_id"] = source_run_id
    _write(replacement_path, replacement_metadata)

    rewired = 0
    for path in runs_root().glob("**/metadata.json"):
        metadata = _read(path)
        dependencies = [str(value) for value in metadata.get("depends_on_run_ids") or ()]
        if source_run_id not in dependencies:
            continue
        metadata["depends_on_run_ids"] = [
            replacement.run_id if value == source_run_id else value
            for value in dependencies
        ]
        metadata["msa_gpu_batch_run_id"] = replacement.run_id
        _write(path, metadata)
        rewired += 1
    return {
        "source_run_id": source_run_id,
        "batch_run_id": replacement.run_id,
        "sequence_count": len(sequences),
        "rewired_structure_jobs": rewired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run_id")
    args = parser.parse_args()
    print(json.dumps(retry_on_gpu(args.source_run_id), indent=2))


if __name__ == "__main__":
    main()
