from __future__ import annotations

import json
from uuid import uuid4

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import display_job_code
from mn_ligand.workflows.benchmark_datasets import (
    benchmark_cases,
    list_benchmark_dataset_jobs,
)
from mn_ligand.workflows.refolding import (
    DEFAULT_ALPHAFOLD3_IMAGE,
    DEFAULT_BOLTZ2_IMAGE,
    DEFAULT_NESSO_IMAGE,
    queue_alphafold3_refolding_job,
    queue_boltz2_refolding_job,
    queue_nesso_affinity_job,
)


def _write_metadata(job, values: dict) -> None:
    path = job.run_dir / "metadata.json"
    payload = json.loads(path.read_text())
    payload.update(values)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def render(*, embedded: bool = False) -> None:
    if embedded:
        st.subheader("Refolding")
    else:
        st.title("Benchmark Refolding")
    st.caption(
        "Predict each reference protein–ligand complex from receptor sequence and "
        "ligand identity, then align the predicted protein to the reference receptor "
        "before measuring symmetry-aware ligand RMSD."
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
            selected = st.selectbox(
                "Imported benchmark dataset",
                list(labels),
                key="benchmark_refold_dataset",
            )
            dataset_job = labels[selected]
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
            key="benchmark_refold_case_mode",
        )
        selected_ids = (
            st.multiselect(
                "Benchmark cases",
                case_ids,
                default=case_ids[: min(10, len(case_ids))],
                key="benchmark_refold_cases",
            )
            if mode == "Manual subset"
            else case_ids
        )
        selected_cases = [case for case in cases if case.case_id in selected_ids]
        if cases:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Case": case.case_id,
                            "Target": case.payload.get("target_id"),
                            "Split": case.payload.get("split"),
                            "Reference SMILES": case.payload.get("ligand_smiles"),
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
        columns = st.columns(3)
        use_boltz = columns[0].checkbox(
            "Boltz-2", value=True, key="benchmark_refold_boltz"
        )
        use_af3 = columns[1].checkbox(
            "AlphaFold 3 / AlphaFast", value=True, key="benchmark_refold_af3"
        )
        use_nesso = columns[2].checkbox(
            "Nesso-1 affinity panel",
            value=False,
            key="benchmark_refold_nesso",
            help=(
                "Nesso-1 is installed but emits no structure. It contributes affinity "
                "coverage and runtime only, never ligand RMSD."
            ),
        )
        controls = st.columns(4)
        replicates = int(
            controls[0].number_input(
                "Independent runs", 1, 20, 1, key="benchmark_refold_replicates"
            )
        )
        seed = int(
            controls[1].number_input(
                "First seed", 1, 2_147_483_000, 1001, key="benchmark_refold_seed"
            )
        )
        boltz_steps = int(
            controls[2].number_input(
                "Boltz sampling steps",
                20,
                1000,
                200,
                key="benchmark_refold_boltz_steps",
            )
        )
        recycles = int(
            controls[3].number_input(
                "Recycling steps", 1, 20, 3, key="benchmark_refold_recycles"
            )
        )
        st.info(
            "Boltz-2 and AlphaFold 3 produce complexes and receive canonical "
            "protein-aligned ligand RMSD. Nesso-1 is retained as an explicitly "
            "structure-free affinity comparator."
        )

    with run_tab:
        gpu = st.selectbox(
            "GPU", ("Automatic", "GPU 0", "GPU 1"), key="benchmark_refold_gpu"
        )
        render_run_resources(
            requires_gpu=use_boltz or use_af3 or use_nesso,
            selected_gpu=gpu,
            key="benchmark_refold",
        )
        blockers = []
        if dataset_job is None:
            blockers.append("Select an imported benchmark dataset.")
        if not selected_cases:
            blockers.append("Select at least one benchmark case.")
        if not any((use_boltz, use_af3, use_nesso)):
            blockers.append("Select at least one engine.")
        for blocker in blockers:
            st.info(blocker)
        jobs_per_case = sum((use_boltz, use_af3, use_nesso))
        st.write(
            f"{len(selected_cases)} case(s) × {jobs_per_case} engine(s) = "
            f"{len(selected_cases) * jobs_per_case} queued jobs."
        )
        if st.button(
            "Queue benchmark refolding",
            type="primary",
            disabled=bool(blockers),
            key="benchmark_refold_queue",
        ) and dataset_job is not None:
            campaign_id = str(uuid4())
            gpu_device = "all" if gpu == "Automatic" else gpu.replace("GPU ", "")
            queued = []
            try:
                for case_index, case in enumerate(selected_cases):
                    common = {
                        "target_path": case.receptor_path,
                        "target_artifact": case.receptor_artifact,
                        "compound_paths": (case.ligand_path,),
                        "compound_artifacts": (case.ligand_artifact,),
                        "gpu_device": gpu_device,
                        "max_compounds": 1,
                        "launch_campaign_id": campaign_id,
                        "launch_campaign_label": (
                            f"{dataset_job.metadata.get('dataset_name')} refolding"
                        ),
                    }
                    metadata = {
                        "benchmark_dataset_run_id": dataset_job.run_id,
                        "benchmark_campaign_id": campaign_id,
                        "benchmark_case_id": case.case_id,
                        "benchmark_mode": "refolding",
                        "benchmark_reference_ligand_artifact_id": (
                            case.ligand_artifact.artifact_id
                        ),
                    }
                    if use_boltz:
                        job = queue_boltz2_refolding_job(
                            **common,
                            reference_ligand_artifact=case.ligand_artifact,
                            image=DEFAULT_BOLTZ2_IMAGE,
                            sampling_steps=boltz_steps,
                            recycling_steps=recycles,
                            replicates=replicates,
                            seed_start=seed + case_index * replicates,
                            use_msa_server=False,
                        )
                        _write_metadata(job, metadata)
                        queued.append(job)
                    if use_af3:
                        job = queue_alphafold3_refolding_job(
                            **common,
                            reference_ligand_artifact=case.ligand_artifact,
                            image=DEFAULT_ALPHAFOLD3_IMAGE,
                            num_recycles=max(1, recycles),
                            model_seed_count=replicates,
                            model_seed_start=seed + case_index * replicates,
                        )
                        _write_metadata(job, metadata)
                        queued.append(job)
                    if use_nesso:
                        nesso_common = dict(common)
                        nesso_common.pop("launch_campaign_id", None)
                        nesso_common.pop("launch_campaign_label", None)
                        job = queue_nesso_affinity_job(
                            **nesso_common,
                            image=DEFAULT_NESSO_IMAGE,
                            recycling_steps=max(1, recycles),
                            replicates=replicates,
                            seed=seed + case_index * replicates,
                            launch_campaign_id=campaign_id,
                            launch_campaign_label=(
                                f"{dataset_job.metadata.get('dataset_name')} refolding"
                            ),
                        )
                        _write_metadata(job, metadata)
                        queued.append(job)
                st.success(
                    f"Queued benchmark campaign {campaign_id[:8]} with "
                    f"{len(queued)} engine jobs."
                )
                st.link_button(
                    "Open aggregate dataset results",
                    f"./benchmark-results?dataset_run_id={dataset_job.run_id}",
                )
            except Exception as exc:
                st.error(f"Could not queue benchmark refolding: {exc}")


if __name__ == "__main__":
    render()
