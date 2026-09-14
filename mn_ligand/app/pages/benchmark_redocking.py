from __future__ import annotations

import json
from uuid import uuid4

import pandas as pd
import streamlit as st
from rdkit import Chem

from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import display_job_code
from mn_ligand.workflows.benchmark_datasets import (
    benchmark_cases,
    list_benchmark_dataset_jobs,
)
from mn_ligand.workflows.docking import DEFAULT_DOCKING_IMAGE
from mn_ligand.workflows.openvs import DEFAULT_OPENVS_IMAGE, queue_openvs_docking_job
from mn_ligand.workflows.redocking import queue_redocking_benchmark


ENGINE_IDS = {
    "AutoDock Vina": "vina",
    "GNINA": "gnina",
    "Uni-Dock Pro": "udp",
    "RosettaLigand": "openvs",
}


def _write_metadata(job, values: dict) -> None:
    path = job.run_dir / "metadata.json"
    payload = json.loads(path.read_text())
    payload.update(values)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _annotate_redocking_tree(job, values: dict) -> None:
    _write_metadata(job, values)
    workflow_path = job.run_dir / "workflow.json"
    if not workflow_path.is_file():
        return
    workflow = json.loads(workflow_path.read_text())
    for child in workflow.get("children") or []:
        task_group = str(child.get("task_group") or "")
        run_id = str(child.get("run_id") or "")
        child_path = job.run_dir.parent.parent / task_group / run_id / "metadata.json"
        if not child_path.is_file():
            continue
        payload = json.loads(child_path.read_text())
        payload.update(values)
        child_path.write_text(json.dumps(payload, indent=2) + "\n")


def _ligand_box(path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
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


def render(*, embedded: bool = False) -> None:
    if embedded:
        st.subheader("Redocking")
    else:
        st.title("Benchmark Redocking")
    st.caption(
        "Redock each crystallographic ligand into its own reference receptor. "
        "Symmetry-aware heavy-atom ligand RMSD is the primary endpoint; recovery "
        "at 1 Å and 2 Å is aggregated by dataset and engine."
    )
    dataset_tab, cases_tab, engines_tab, run_tab = st.tabs(
        ["Dataset", "Cases", "Engines", "Run"]
    )
    jobs = list_benchmark_dataset_jobs()
    with dataset_tab:
        if jobs:
            labels = {
                (
                    f"{job.metadata.get('dataset_name')} — "
                    f"{display_job_code(job.metadata.get('job_code'), job.run_id)} "
                    f"({job.metadata.get('case_count')} cases)"
                ): job
                for job in jobs
            }
            label = st.selectbox(
                "Imported benchmark dataset", list(labels), key="benchmark_redock_dataset"
            )
            dataset_job = labels[label]
        else:
            dataset_job = None
            st.info("Import a benchmark dataset under Prepare first.")
            st.link_button("Open Benchmark Datasets", "./prepare-benchmark-datasets")

    cases = benchmark_cases(dataset_job) if dataset_job is not None else []
    with cases_tab:
        case_ids = [case.case_id for case in cases]
        mode = st.radio(
            "Case selection",
            ("All cases", "Manual subset"),
            horizontal=True,
            key="benchmark_redock_case_mode",
        )
        if mode == "Manual subset":
            selected_ids = st.multiselect(
                "Benchmark cases",
                case_ids,
                default=case_ids[: min(10, len(case_ids))],
                key="benchmark_redock_cases",
            )
        else:
            selected_ids = case_ids
        selected_cases = [case for case in cases if case.case_id in selected_ids]
        if cases:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Case": case.case_id,
                            "Target": case.payload.get("target_id"),
                            "Split": case.payload.get("split"),
                            "Ligand records": case.payload.get("ligand_record_count"),
                            "3D": case.payload.get("coordinate_dimension"),
                            "Selected": case.case_id in selected_ids,
                        }
                        for case in cases
                    ]
                ),
                hide_index=True,
                width="stretch",
                height=450,
            )

    with engines_tab:
        columns = st.columns(4)
        enabled = {
            name: column.checkbox(
                name,
                value=True,
                key=f"benchmark_redock_engine_{engine}",
            )
            for column, (name, engine) in zip(columns, ENGINE_IDS.items())
        }
        settings = st.columns(4)
        replicates = int(
            settings[0].number_input(
                "Independent runs", 1, 20, 1, key="benchmark_redock_replicates"
            )
        )
        poses = int(
            settings[1].number_input(
                "Poses per run", 1, 100, 10, key="benchmark_redock_poses"
            )
        )
        exhaustiveness = int(
            settings[2].number_input(
                "Vina/GNINA exhaustiveness",
                1,
                1024,
                30,
                key="benchmark_redock_exhaustiveness",
            )
        )
        seed_start = int(
            settings[3].number_input(
                "First seed",
                1,
                2_147_483_000,
                1001,
                key="benchmark_redock_seed",
            )
        )
        st.caption(
            "Vina, GNINA, and Uni-Dock Pro share the validated docking-suite adapter. "
            "RosettaLigand runs through its CPU-native GALigandDock adapter and is "
            "joined by the dataset-level evaluator."
        )

    with run_tab:
        gpu = st.selectbox(
            "GPU", ("Automatic", "GPU 0", "GPU 1"), key="benchmark_redock_gpu"
        )
        render_run_resources(
            requires_gpu=any(
                enabled.get(name)
                for name in ("GNINA", "Uni-Dock Pro")
            ),
            selected_gpu=gpu,
            key="benchmark_redock",
        )
        blockers = []
        if dataset_job is None:
            blockers.append("Select an imported benchmark dataset.")
        if not selected_cases:
            blockers.append("Select at least one benchmark case.")
        if not any(enabled.values()):
            blockers.append("Select at least one engine.")
        for blocker in blockers:
            st.info(blocker)
        job_count = len(selected_cases) * (
            int(any(enabled[name] for name in ("AutoDock Vina", "GNINA", "Uni-Dock Pro")))
            + int(enabled["RosettaLigand"])
        )
        st.write(
            f"{len(selected_cases)} case(s) will create {job_count} benchmark parent "
            "campaign(s); engine runs remain immutable child jobs."
        )
        if st.button(
            "Queue benchmark redocking",
            type="primary",
            disabled=bool(blockers),
            key="benchmark_redock_queue",
        ) and dataset_job is not None:
            campaign_id = str(uuid4())
            queued = []
            gpu_device = "all" if gpu == "Automatic" else gpu.replace("GPU ", "")
            try:
                for case_index, case in enumerate(selected_cases):
                    center, size = _ligand_box(case.ligand_path)
                    common_metadata = {
                        "benchmark_dataset_run_id": dataset_job.run_id,
                        "benchmark_campaign_id": campaign_id,
                        "benchmark_case_id": case.case_id,
                        "benchmark_mode": "redocking",
                    }
                    fixed_engines = [
                        ENGINE_IDS[name]
                        for name in ("AutoDock Vina", "GNINA", "Uni-Dock Pro")
                        if enabled[name]
                    ]
                    if fixed_engines:
                        job = queue_redocking_benchmark(
                            receptor_path=case.receptor_path,
                            target_artifact=case.receptor_artifact,
                            target_task_group=dataset_job.task_group,
                            reference_ligand_path=case.ligand_path,
                            reference_ligand_artifact=case.ligand_artifact,
                            reference_task_group=dataset_job.task_group,
                            center=center,
                            size=size,
                            engines=fixed_engines,
                            replicates=replicates,
                            seed_start=seed_start + case_index * replicates,
                            image=DEFAULT_DOCKING_IMAGE,
                            gpu_device=gpu_device,
                            exhaustiveness=exhaustiveness,
                            poses=poses,
                            context_metadata=common_metadata,
                        )
                        _annotate_redocking_tree(job, common_metadata)
                        queued.append(job)
                    if enabled["RosettaLigand"]:
                        job = queue_openvs_docking_job(
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
                            replicates=replicates,
                            seed_start=seed_start + case_index * replicates,
                            maximum_compounds=1,
                            launch_campaign_id=campaign_id,
                            launch_campaign_label=f"{dataset_job.metadata.get('dataset_name')} redocking",
                        )
                        _write_metadata(job, common_metadata)
                        queued.append(job)
                st.success(
                    f"Queued benchmark campaign {campaign_id[:8]} with "
                    f"{len(queued)} case-level parent job(s)."
                )
                st.link_button(
                    "Open aggregate dataset results",
                    f"./benchmark-results?dataset_run_id={dataset_job.run_id}",
                )
            except Exception as exc:
                st.error(f"Could not queue benchmark campaign: {exc}")


if __name__ == "__main__":
    render()
