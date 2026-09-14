from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from uuid import uuid4

from mn_ligand.core.artifacts import ArtifactRef
from mn_ligand.core.job_control import create_job_retry
from mn_ligand.core.jobs import JobRecord
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.refolding import (
    configured_alphafold3_reference_paths,
    queue_alphafold3_msa_job,
)


TERMINAL_STATES = {"completed", "failed", "cancelled", "blocked"}
STRUCTURE_WORKFLOWS = {
    "alphafold3_refolding",
    "boltz2_refolding",
    "nesso_affinity",
}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _campaign_jobs(label: str) -> list[JobRecord]:
    jobs: list[JobRecord] = []
    for metadata_path in runs_root().glob("**/metadata.json"):
        metadata = _read(metadata_path)
        if str(metadata.get("launch_campaign_label") or "") != label:
            continue
        jobs.append(
            JobRecord.load(
                metadata_path.parent,
                task_group=metadata_path.parent.parent.name,
            )
        )
    return jobs


def _af3_sequences_and_artifacts(
    jobs: list[JobRecord],
) -> dict[str, ArtifactRef]:
    sequences: dict[str, ArtifactRef] = {}
    for job in jobs:
        if job.workflow != "alphafold3_refolding":
            continue
        target = ArtifactRef.from_dict(_read(job.run_dir / "input.json")["target"])
        for input_path in sorted((job.run_dir / "inputs").glob("*.json")):
            payload = _read(input_path)
            for entity in payload.get("sequences") or ():
                protein = entity.get("protein") if isinstance(entity, dict) else None
                sequence = "".join(str((protein or {}).get("sequence") or "").split()).upper()
                if sequence:
                    sequences.setdefault(sequence, target)
    return sequences


def _boltz_protein_sequences(run_dir: Path) -> list[str]:
    result: list[str] = []
    for yaml_path in sorted((run_dir / "inputs").glob("*.yaml"))[:1]:
        in_protein = False
        for line in yaml_path.read_text().splitlines():
            if line.startswith("  - protein:"):
                in_protein = True
                continue
            if line.startswith("  - "):
                in_protein = False
            if in_protein:
                match = re.match(r"\s+sequence:\s+([A-Za-z]+)\s*$", line)
                if match:
                    result.append(match.group(1).upper())
    return result


def restart(label: str) -> dict[str, object]:
    originals = _campaign_jobs(label)
    if not originals:
        raise ValueError(f"No jobs found for campaign {label!r}")
    campaign_ids = {
        str(job.metadata.get("launch_campaign_id") or "") for job in originals
    }
    if len(campaign_ids) != 1:
        raise ValueError("The label resolves to more than one original campaign ID")
    original_campaign_id = next(iter(campaign_ids))
    active = [job for job in originals if job.status not in TERMINAL_STATES]
    if active:
        raise ValueError(f"{len(active)} original jobs are still active")
    prior_restarts: set[str] = set()
    active_restarts: set[str] = set()
    for metadata_path in runs_root().glob("**/metadata.json"):
        metadata = _read(metadata_path)
        if str(metadata.get("restart_of_campaign_id") or "") == original_campaign_id:
            prior_restarts.add(str(metadata.get("launch_campaign_id") or ""))
            if str(metadata.get("status") or "") not in TERMINAL_STATES:
                active_restarts.add(str(metadata.get("launch_campaign_id") or ""))
    if active_restarts:
        raise ValueError("This campaign already has an active MSA-first restart")

    new_campaign_id = str(uuid4())
    generation = len({value for value in prior_restarts if value}) + 1
    new_label = f"{label}_MSA_first_restart_v{generation}"
    db_dir, weights_dir, repository = configured_alphafold3_reference_paths()
    sequences = _af3_sequences_and_artifacts(originals)
    if not sequences:
        raise ValueError("No AlphaFold protein sequences were available for the MSA barrier")
    msa_jobs = [
        queue_alphafold3_msa_job(
            protein_sequences=list(sequences),
            target_artifact=next(iter(sequences.values())),
            db_dir=db_dir,
            weights_dir=weights_dir,
            msa_repository_dir=repository,
            batch_size=len(sequences),
            launch_campaign_id=new_campaign_id,
            launch_campaign_label=new_label,
            campaign_purpose=str(originals[0].metadata.get("campaign_purpose") or ""),
        )
    ]
    dependencies = [job.run_id for job in msa_jobs]
    replayed: list[JobRecord] = []
    try:
        for original in sorted(originals, key=lambda item: item.created_at):
            replay = create_job_retry(
                original,
                requested_by="msa_first_campaign_restart",
                allow_completed_replay=True,
            )
            metadata_path = replay.run_dir / "metadata.json"
            metadata = _read(metadata_path)
            metadata.update(
                {
                    "launch_campaign_id": new_campaign_id,
                    "launch_campaign_label": new_label,
                    "restart_of_campaign_id": original_campaign_id,
                    "restart_of_campaign_label": label,
                    "restart_generation": generation,
                    "campaign_restart_policy": "msa_gpu0_then_structure_gpu0_with_docking_gpu1",
                }
            )
            resources = dict(metadata.get("resources") or {})
            if replay.workflow in STRUCTURE_WORKFLOWS:
                metadata["depends_on_run_ids"] = dependencies
                metadata["campaign_phase"] = "structure_after_msa"
                metadata["gpu_device"] = "0"
                resources["gpu_ids"] = [0]
                if replay.workflow in {"alphafold3_refolding", "boltz2_refolding"}:
                    metadata["msa_preparation_required"] = True
                    metadata["msa_repository_dir"] = str(repository.resolve())
                if replay.workflow == "boltz2_refolding":
                    metadata["protein_sequences"] = _boltz_protein_sequences(replay.run_dir)
            elif replay.workflow == "docking_campaign" and str(
                metadata.get("engine") or ""
            ).lower() in {"udp", "gnina"}:
                metadata["campaign_phase"] = "gpu_docking_parallel_with_msa"
                metadata["gpu_device"] = "1"
                resources["gpu_ids"] = [1]
            else:
                metadata["campaign_phase"] = "cpu_parallel_with_msa"
            metadata["resources"] = resources
            _write(metadata_path, metadata)
            replayed.append(JobRecord.load(replay.run_dir, task_group=replay.task_group))
    except Exception:
        # Preserve already-created immutable records for diagnosis; dependencies
        # prevent structure jobs from escaping ahead of the MSA stage.
        raise
    return {
        "original_campaign_id": original_campaign_id,
        "new_campaign_id": new_campaign_id,
        "new_campaign_label": new_label,
        "msa_jobs": len(msa_jobs),
        "replayed_jobs": len(replayed),
        "msa_run_ids": dependencies,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_label")
    args = parser.parse_args()
    print(json.dumps(restart(args.campaign_label), indent=2))


if __name__ == "__main__":
    main()
