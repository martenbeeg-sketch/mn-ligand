#!/usr/bin/env python
"""Queue or inspect a bounded all-engine benchmark smoke campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from rdkit import Chem

from mn_ligand.core.jobs import JobRecord, iter_job_records
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.benchmark_datasets import (
    benchmark_cases,
    list_benchmark_dataset_jobs,
)
from mn_ligand.workflows.docking import DEFAULT_DOCKING_IMAGE
from mn_ligand.workflows.openvs import DEFAULT_OPENVS_IMAGE, queue_openvs_docking_job
from mn_ligand.workflows.redocking import queue_redocking_benchmark
from mn_ligand.workflows.refolding import (
    DEFAULT_ALPHAFOLD3_IMAGE,
    DEFAULT_BOLTZ2_IMAGE,
    DEFAULT_NESSO_IMAGE,
    queue_alphafold3_refolding_job,
    queue_boltz2_refolding_job,
    queue_nesso_affinity_job,
)
from mn_ligand.workflows.rescoring import (
    create_pose_selection_job,
    queue_boltzina_rescoring_job,
    queue_gnina_rescoring_job,
    source_pose_rows,
)


def _write_metadata(job: JobRecord, values: dict[str, object]) -> None:
    path = job.run_dir / "metadata.json"
    payload = json.loads(path.read_text())
    payload.update(values)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _annotate_redocking_tree(job: JobRecord, values: dict[str, object]) -> None:
    _write_metadata(job, values)
    workflow_path = job.run_dir / "workflow.json"
    if not workflow_path.is_file():
        return
    workflow = json.loads(workflow_path.read_text())
    for child in workflow.get("children") or []:
        task_group = str(child.get("task_group") or "")
        run_id = str(child.get("run_id") or "")
        path = job.run_dir.parent.parent / task_group / run_id / "metadata.json"
        if path.is_file():
            payload = json.loads(path.read_text())
            payload.update(values)
            path.write_text(json.dumps(payload, indent=2) + "\n")


def _ligand_box(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    molecule = next(
        (item for item in Chem.SDMolSupplier(str(path), removeHs=False) if item),
        None,
    )
    if molecule is None or not molecule.GetNumConformers():
        raise ValueError("Reference ligand has no readable coordinates")
    conformer = molecule.GetConformer()
    points = [conformer.GetAtomPosition(index) for index in range(molecule.GetNumAtoms())]
    minimum = [min(getattr(point, axis) for point in points) for axis in "xyz"]
    maximum = [max(getattr(point, axis) for point in points) for axis in "xyz"]
    center = tuple((low + high) / 2.0 for low, high in zip(minimum, maximum))
    size = tuple(max(22.0, high - low + 10.0) for low, high in zip(minimum, maximum))
    return center, size


def _dataset(run_id: str) -> JobRecord:
    for job in list_benchmark_dataset_jobs():
        if job.run_id == run_id:
            return job
    raise ValueError(f"Benchmark dataset not found: {run_id}")


def _case(dataset: JobRecord, case_id: str):
    cases = benchmark_cases(dataset)
    if not case_id and len(cases) == 1:
        return cases[0]
    for case in cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Benchmark case not found: {case_id}")


def queue_primary(dataset: JobRecord, case, *, gpu_device: str) -> list[JobRecord]:
    campaign_id = str(uuid4())
    metadata = {
        "benchmark_dataset_run_id": dataset.run_id,
        "benchmark_campaign_id": campaign_id,
        "benchmark_case_id": case.case_id,
    }
    center, size = _ligand_box(case.ligand_path)
    queued: list[JobRecord] = []
    redocking = queue_redocking_benchmark(
        receptor_path=case.receptor_path,
        target_artifact=case.receptor_artifact,
        target_task_group=dataset.task_group,
        reference_ligand_path=case.ligand_path,
        reference_ligand_artifact=case.ligand_artifact,
        reference_task_group=dataset.task_group,
        center=center,
        size=size,
        engines=("vina", "gnina", "udp"),
        replicates=1,
        seed_start=1001,
        image=DEFAULT_DOCKING_IMAGE,
        gpu_device=gpu_device,
        search_mode="fast",
        exhaustiveness=8,
        poses=3,
        context_metadata={**metadata, "benchmark_mode": "redocking"},
    )
    _annotate_redocking_tree(
        redocking, {**metadata, "benchmark_mode": "redocking"}
    )
    queued.append(redocking)
    rosetta = queue_openvs_docking_job(
        receptor_path=case.receptor_path,
        target_artifact=case.receptor_artifact,
        compound_paths=(case.ligand_path,),
        compound_artifacts=(case.ligand_artifact,),
        center=center,
        size=size,
        protocol="vsh",
        reference_mode="reference_guided",
        reference_ligand_path=case.ligand_path,
        reference_ligand_artifact=case.ligand_artifact,
        image=DEFAULT_OPENVS_IMAGE,
        replicates=1,
        seed_start=1001,
        maximum_compounds=1,
        launch_campaign_id=campaign_id,
        launch_campaign_label="PoseBusters all-engine smoke",
    )
    _write_metadata(rosetta, {**metadata, "benchmark_mode": "redocking"})
    queued.append(rosetta)

    common = {
        "target_path": case.receptor_path,
        "target_artifact": case.receptor_artifact,
        "compound_paths": (case.ligand_path,),
        "compound_artifacts": (case.ligand_artifact,),
        "gpu_device": gpu_device,
        "max_compounds": 1,
        "launch_campaign_id": campaign_id,
        "launch_campaign_label": "PoseBusters all-engine smoke",
    }
    refold_metadata = {
        **metadata,
        "benchmark_mode": "refolding",
        "benchmark_reference_ligand_artifact_id": case.ligand_artifact.artifact_id,
    }
    boltz = queue_boltz2_refolding_job(
        **common,
        reference_ligand_artifact=case.ligand_artifact,
        image=DEFAULT_BOLTZ2_IMAGE,
        sampling_steps=20,
        recycling_steps=1,
        replicates=1,
        seed_start=1001,
        use_msa_server=False,
    )
    _write_metadata(boltz, refold_metadata)
    queued.append(boltz)
    alphafold = queue_alphafold3_refolding_job(
        **common,
        reference_ligand_artifact=case.ligand_artifact,
        image=DEFAULT_ALPHAFOLD3_IMAGE,
        num_recycles=1,
        model_seed_count=1,
        model_seed_start=1001,
    )
    _write_metadata(alphafold, refold_metadata)
    queued.append(alphafold)
    nesso_common = dict(common)
    nesso_common.pop("launch_campaign_id")
    nesso_common.pop("launch_campaign_label")
    nesso = queue_nesso_affinity_job(
        **nesso_common,
        image=DEFAULT_NESSO_IMAGE,
        recycling_steps=1,
        replicates=1,
        seed=1001,
        launch_campaign_id=campaign_id,
        launch_campaign_label="PoseBusters all-engine smoke",
    )
    _write_metadata(nesso, refold_metadata)
    queued.append(nesso)
    return queued


def queue_boltz_retry(
    dataset: JobRecord, case, *, gpu_device: str
) -> JobRecord:
    campaign_id = str(uuid4())
    job = queue_boltz2_refolding_job(
        target_path=case.receptor_path,
        target_artifact=case.receptor_artifact,
        compound_paths=(case.ligand_path,),
        compound_artifacts=(case.ligand_artifact,),
        reference_ligand_artifact=case.ligand_artifact,
        image=DEFAULT_BOLTZ2_IMAGE,
        gpu_device=gpu_device,
        max_compounds=1,
        launch_campaign_id=campaign_id,
        launch_campaign_label="PoseBusters all-engine smoke retry",
        sampling_steps=20,
        recycling_steps=1,
        replicates=1,
        seed_start=1001,
        use_msa_server=False,
    )
    _write_metadata(
        job,
        {
            "benchmark_dataset_run_id": dataset.run_id,
            "benchmark_campaign_id": campaign_id,
            "benchmark_case_id": case.case_id,
            "benchmark_mode": "refolding",
            "benchmark_reference_ligand_artifact_id": (
                case.ligand_artifact.artifact_id
            ),
        },
    )
    return job


def queue_redocking_retry(
    dataset: JobRecord, case, *, gpu_device: str
) -> JobRecord:
    campaign_id = str(uuid4())
    center, size = _ligand_box(case.ligand_path)
    job = queue_redocking_benchmark(
        receptor_path=case.receptor_path,
        target_artifact=case.receptor_artifact,
        target_task_group=dataset.task_group,
        reference_ligand_path=case.ligand_path,
        reference_ligand_artifact=case.ligand_artifact,
        reference_task_group=dataset.task_group,
        center=center,
        size=size,
        engines=("vina", "gnina", "udp"),
        replicates=1,
        seed_start=2001,
        image=DEFAULT_DOCKING_IMAGE,
        gpu_device=gpu_device,
        search_mode="fast",
        exhaustiveness=8,
        poses=3,
        context_metadata={
            "benchmark_dataset_run_id": dataset.run_id,
            "benchmark_campaign_id": campaign_id,
            "benchmark_case_id": case.case_id,
            "benchmark_mode": "redocking",
        },
    )
    _annotate_redocking_tree(
        job,
        {
            "benchmark_dataset_run_id": dataset.run_id,
            "benchmark_campaign_id": campaign_id,
            "benchmark_case_id": case.case_id,
            "benchmark_mode": "redocking",
        },
    )
    return job


def _boltz_context(dataset_run_id: str, case_id: str) -> Path | None:
    for job in iter_job_records(runs_root(), task_groups=("refolding",)):
        if (
            job.status == "completed"
            and job.workflow == "boltz2_refolding"
            and str(job.metadata.get("benchmark_dataset_run_id") or "") == dataset_run_id
            and str(job.metadata.get("benchmark_case_id") or "") == case_id
        ):
            manifests = sorted(job.run_dir.glob("output/**/processed/manifest.json"))
            if manifests:
                return manifests[0].parent.parent
    return None


def queue_rescoring(dataset: JobRecord, case_id: str, *, gpu_device: str) -> list[JobRecord]:
    candidates = [
        job
        for job in iter_job_records(runs_root(), task_groups=("docking",))
        if job.status == "completed"
        and job.workflow == "docking_campaign"
        and str(job.metadata.get("benchmark_dataset_run_id") or "") == dataset.run_id
        and str(job.metadata.get("benchmark_case_id") or "") == case_id
        and source_pose_rows(job)
    ]
    sources_by_engine: dict[str, JobRecord] = {}
    for job in candidates:
        engine = str(job.metadata.get("engine") or job.tool)
        sources_by_engine.setdefault(engine, job)
    sources = list(sources_by_engine.values())
    context = _boltz_context(dataset.run_id, case_id)
    if context is None:
        raise RuntimeError("Completed Boltz-2 context is required before rescoring")
    queued: list[JobRecord] = []
    for source in sources:
        selection = create_pose_selection_job(source, pose_ranks=(1,))
        metadata = {
            "benchmark_dataset_run_id": dataset.run_id,
            "benchmark_campaign_id": source.metadata.get("benchmark_campaign_id", ""),
            "benchmark_case_id": case_id,
            "benchmark_mode": "rescoring",
            "benchmark_source_docking_run_id": source.run_id,
        }
        _write_metadata(selection, metadata)
        gnina = queue_gnina_rescoring_job(
            selection_job=selection,
            gpu_device=gpu_device,
            cnn_rotation=0,
        )
        _write_metadata(gnina, metadata)
        queued.append(gnina)
        boltzina = queue_boltzina_rescoring_job(
            selection_job=selection,
            boltz_work_dir=context,
            gpu_device=gpu_device,
        )
        _write_metadata(boltzina, metadata)
        queued.append(boltzina)
    return queued


def status(dataset: JobRecord) -> list[dict[str, object]]:
    rows = []
    for job in iter_job_records(runs_root()):
        if str(job.metadata.get("benchmark_dataset_run_id") or "") != dataset.run_id:
            continue
        rows.append(
            {
                "task_group": job.task_group,
                "run_id": job.run_id,
                "workflow": job.workflow,
                "tool": job.tool,
                "status": job.status,
                "case_id": job.metadata.get("benchmark_case_id"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "primary",
            "boltz-retry",
            "redocking-retry",
            "rescoring",
            "status",
        ),
    )
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--gpu-device", default="all")
    args = parser.parse_args()
    dataset = _dataset(args.dataset_run_id)
    case = _case(dataset, args.case_id)
    if args.action == "primary":
        jobs = queue_primary(dataset, case, gpu_device=args.gpu_device)
        print(json.dumps({"queued": [job.run_id for job in jobs]}, indent=2))
    elif args.action == "boltz-retry":
        job = queue_boltz_retry(dataset, case, gpu_device=args.gpu_device)
        print(json.dumps({"queued": [job.run_id]}, indent=2))
    elif args.action == "redocking-retry":
        job = queue_redocking_retry(dataset, case, gpu_device=args.gpu_device)
        print(json.dumps({"queued": [job.run_id]}, indent=2))
    elif args.action == "rescoring":
        jobs = queue_rescoring(dataset, case.case_id, gpu_device=args.gpu_device)
        print(json.dumps({"queued": [job.run_id for job in jobs]}, indent=2))
    else:
        print(json.dumps(status(dataset), indent=2))


if __name__ == "__main__":
    main()
