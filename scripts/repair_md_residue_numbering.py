"""Publish an immutable MD aggregate-analysis residue-numbering revision."""

from __future__ import annotations

import argparse

from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.workflows import WorkflowRecord
from mn_ligand.runtime import resolve_run_dir
from mn_ligand.workflows.md_simulation import (
    create_md_residue_numbering_revision,
    source_author_residue_mapping,
)


def preview(workflow_id: str) -> tuple[int, str, str]:
    workflow = WorkflowRecord.load(workflow_id)
    if not workflow.inputs:
        raise ValueError(f"Workflow {workflow_id} has no source artifact")
    workflow_input = workflow.inputs[0]
    source_dir = resolve_run_dir(
        workflow_input.source_task_group,
        workflow_input.artifact.run_id,
    )
    if source_dir is None:
        raise FileNotFoundError(workflow_input.artifact.run_id)
    source_job = JobRecord.load(
        source_dir,
        task_group=workflow_input.source_task_group,
    )
    source_path = workflow_input.artifact.resolve(source_job.run_dir, must_exist=True)
    if source_path is None:
        raise FileNotFoundError(workflow_input.artifact.path)
    mapping = source_author_residue_mapping(
        source_job,
        workflow_input.artifact,
        source_path.read_text(errors="replace"),
    )
    source = str(mapping.get("mapping_basis") or "source author mapping")
    rows = mapping.get("residues") or []
    if not rows:
        raise ValueError("The workflow source has no mappable protein residues")
    first = rows[0]
    last = rows[-1]
    first_label = (
        f"{first.get('native_chain')}:{first.get('native_residue_number')}"
    )
    last_label = f"{last.get('native_chain')}:{last.get('native_residue_number')}"
    print(f"workflow={workflow_id}")
    print(f"mapping_source={source}")
    print(f"protein_residues={len(rows)}")
    print(f"author_range={first_label}-{last_label}")
    return len(rows), first_label, last_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow_id")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    preview(args.workflow_id)
    if not args.apply:
        print("dry_run=true")
        return
    revision = create_md_residue_numbering_revision(args.workflow_id)
    print(f"revision_run_id={revision.run_id}")
    print(f"revision_status={revision.status}")


if __name__ == "__main__":
    main()
