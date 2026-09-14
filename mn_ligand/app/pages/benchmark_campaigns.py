from __future__ import annotations

from uuid import uuid4

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.benchmark_common import (
    campaign_table,
    render_case_viewer,
    visible_dataset_jobs,
)
from mn_ligand.app.pages.benchmark_redocking import (
    ENGINE_IDS,
    _annotate_redocking_tree,
    _ligand_box,
    _write_metadata,
)
from mn_ligand.app.pages.benchmark_rescoring import (
    _boltz_context,
    _sources,
)
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import display_job_code
from mn_ligand.workflows.benchmark_datasets import (
    benchmark_cases,
    create_benchmark_bound_chain_target,
    list_benchmark_dataset_jobs,
)
from mn_ligand.workflows.docking import DEFAULT_DOCKING_IMAGE
from mn_ligand.workflows.openvs import DEFAULT_OPENVS_IMAGE, queue_openvs_docking_job
from mn_ligand.workflows.redocking import queue_redocking_benchmark
from mn_ligand.workflows.refolding import (
    DEFAULT_ALPHAFOLD3_IMAGE,
    DEFAULT_BOLTZ2_IMAGE,
    DEFAULT_NESSO_IMAGE,
    ligand_bound_protein_sequence,
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

SELECTION_KEYS = (
    *(f"benchmark_campaign_redock_{engine}" for engine in ENGINE_IDS.values()),
    "benchmark_campaign_boltz",
    "benchmark_campaign_af3",
    "benchmark_campaign_nesso",
    "benchmark_campaign_gnina_score",
    "benchmark_campaign_boltzina",
)


def _set_engine_selection(value: bool) -> None:
    for key in SELECTION_KEYS:
        st.session_state[key] = value


def render() -> None:
    st.title("Redocking / Refolding")
    st.caption(
        "Select one immutable reference collection, choose engines, launch the "
        "case fan-out, and compare ligand-RMSD recovery in the final Results tab."
    )
    dataset_tab, engines_tab, run_tab, results_tab = st.tabs(
        ["Dataset", "Engines", "Run", "Results"]
    )
    jobs = visible_dataset_jobs()
    requested = str(st.query_params.get("dataset_run_id") or "")

    with dataset_tab:
        if not jobs:
            st.info("Import a benchmark dataset under Prepare first.")
            st.link_button("Open Benchmark Datasets", "./prepare-benchmark-datasets")
            dataset_job = None
        else:
            labels = {
                (
                    f"{job.metadata.get('dataset_name')} — "
                    f"{display_job_code(job.metadata.get('job_code'), job.run_id)} "
                    f"({job.metadata.get('case_count')} cases)"
                ): job
                for job in jobs
            }
            default = next(
                (i for i, job in enumerate(labels.values()) if job.run_id == requested),
                0,
            )
            label = st.selectbox(
                "Imported benchmark dataset",
                list(labels),
                index=default,
                key="benchmark_campaign_dataset",
            )
            dataset_job = labels[label]

    cases = benchmark_cases(dataset_job) if dataset_job is not None else []
    with dataset_tab:
        case_ids = [case.case_id for case in cases]
        selection_mode = st.radio(
            "Case selection",
            ("All cases", "Manual subset"),
            horizontal=True,
            key="benchmark_campaign_case_mode",
        )
        selected_ids = (
            st.multiselect(
                "Benchmark cases",
                case_ids,
                default=case_ids[: min(10, len(case_ids))],
                key="benchmark_campaign_cases",
            )
            if selection_mode == "Manual subset"
            else case_ids
        )
        selected_cases = [case for case in cases if case.case_id in selected_ids]
        if cases:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Case": case.case_id,
                            "Target": case.payload.get("target_id", ""),
                            "Split": case.payload.get("split", ""),
                            "Reference SMILES": case.payload.get("ligand_smiles", ""),
                            "Selected": case.case_id in selected_ids,
                        }
                        for case in cases
                    ]
                ),
                hide_index=True,
                width="stretch",
                height=420,
            )
            view_labels = {case.case_id: case for case in cases}
            viewed = st.selectbox(
                "Reference complex viewer",
                list(view_labels),
                key=f"benchmark_campaign_viewer_{dataset_job.run_id}",
            )
            render_case_viewer(
                view_labels[viewed],
                key=f"benchmark-campaign-{dataset_job.run_id}-{viewed}",
            )

    with engines_tab:
        actions = st.columns(2)
        actions[0].button(
            "Select all engines",
            on_click=_set_engine_selection,
            args=(True,),
            key="benchmark_campaign_select_all",
        )
        actions[1].button(
            "Deselect all engines",
            on_click=_set_engine_selection,
            args=(False,),
            key="benchmark_campaign_deselect_all",
        )
        st.markdown("#### Redocking")
        redock_columns = st.columns(4)
        redock_enabled = {
            name: column.checkbox(
                name, value=True, key=f"benchmark_campaign_redock_{engine}"
            )
            for column, (name, engine) in zip(redock_columns, ENGINE_IDS.items())
        }
        st.markdown("#### Refolding")
        refold_columns = st.columns(3)
        use_boltz = refold_columns[0].checkbox(
            "Boltz-2", value=True, key="benchmark_campaign_boltz"
        )
        use_af3 = refold_columns[1].checkbox(
            "AlphaFold 3 / AlphaFast", value=True, key="benchmark_campaign_af3"
        )
        use_nesso = refold_columns[2].checkbox(
            "Nesso-1 affinity panel",
            value=False,
            key="benchmark_campaign_nesso",
            help="Affinity-only comparator; it does not contribute ligand RMSD.",
        )
        st.markdown("#### Rescoring")
        rescore_columns = st.columns(2)
        use_gnina_score = rescore_columns[0].checkbox(
            "GNINA score-only", value=False, key="benchmark_campaign_gnina_score"
        )
        use_boltzina = rescore_columns[1].checkbox(
            "Boltzina", value=False, key="benchmark_campaign_boltzina"
        )
        st.caption(
            "Rescoring consumes completed redocking poses from this dataset. If "
            "redocking is queued now, return after it completes and launch rescoring."
        )
        with st.expander("Shared campaign repetitions", expanded=True):
            shared = st.columns(2)
            replicates = int(shared[0].number_input(
                "Independent runs per structure engine",
                1,
                20,
                1,
                key="benchmark_campaign_replicates",
            ))
            seed = int(shared[1].number_input(
                "First seed",
                1,
                2_147_483_000,
                1001,
                key="benchmark_campaign_seed",
            ))
        with st.expander(
            "AutoDock Vina, GNINA, and Uni-Dock Pro settings",
            expanded=any(
                redock_enabled[name]
                for name in ("AutoDock Vina", "GNINA", "Uni-Dock Pro")
            ),
        ):
            docking = st.columns(2)
            poses = int(docking[0].number_input(
                "Poses per run", 1, 100, 10, key="benchmark_campaign_poses"
            ))
            exhaustiveness = int(docking[1].number_input(
                "Exhaustiveness",
                1,
                1024,
                30,
                key="benchmark_campaign_exhaustiveness",
            ))
        with st.expander(
            "RosettaLigand settings",
            expanded=redock_enabled["RosettaLigand"],
        ):
            st.caption(
                "Reference-guided GALigandDock uses the crystallographic ligand and "
                "the same automatically derived search region."
            )
        with st.expander("Boltz-2 settings", expanded=use_boltz):
            boltz_steps = int(st.number_input(
                "Sampling steps",
                20,
                1000,
                200,
                key="benchmark_campaign_boltz_steps",
            ))
        with st.expander("AlphaFold 3 settings", expanded=use_af3):
            st.caption(
                "Independent runs map to model seeds; the shared recycling value is "
                "used for each selected benchmark case."
            )
        with st.expander(
            "Shared refolding settings",
            expanded=use_boltz or use_af3 or use_nesso,
        ):
            recycles = int(st.number_input(
                "Recycling steps",
                1,
                20,
                3,
                key="benchmark_campaign_recycles",
            ))
        with st.expander("Nesso-1 settings", expanded=use_nesso):
            st.caption(
                "Nesso-1 is an affinity-only comparator and does not contribute "
                "a predicted pose or ligand RMSD."
            )
        with st.expander(
            "GNINA score-only and Boltzina settings",
            expanded=use_gnina_score or use_boltzina,
        ):
            st.caption(
                "Every stored source pose is rescored without changing coordinates. "
                "Boltzina requires a completed Boltz-2 context for the same case."
            )

    with run_tab:
        gpu = st.selectbox(
            "GPU", ("Automatic", "GPU 0", "GPU 1"), key="benchmark_campaign_gpu"
        )
        requires_gpu = (
            redock_enabled.get("GNINA", False)
            or redock_enabled.get("Uni-Dock Pro", False)
            or use_boltz
            or use_af3
            or use_nesso
            or use_gnina_score
            or use_boltzina
        )
        render_run_resources(
            requires_gpu=requires_gpu,
            selected_gpu=gpu,
            key="benchmark_campaign",
        )
        selected_engine_count = sum(redock_enabled.values()) + sum(
            (use_boltz, use_af3, use_nesso)
        )
        sources = (
            [
                source
                for source in _sources(dataset_job.run_id)
                if str(source.metadata.get("benchmark_case_id") or "")
                in selected_ids
            ]
            if dataset_job is not None
            else []
        )
        blockers = []
        if dataset_job is None:
            blockers.append("Select an imported benchmark dataset.")
        if not selected_cases:
            blockers.append("Select at least one benchmark case.")
        if not selected_engine_count and not (use_gnina_score or use_boltzina):
            blockers.append("Select at least one engine.")
        if (use_gnina_score or use_boltzina) and not sources and not selected_engine_count:
            blockers.append("No completed redocking poses are available to rescore.")
        for blocker in blockers:
            st.info(blocker)
        st.write(
            f"{len(selected_cases)} selected case(s); "
            f"{selected_engine_count} redocking/refolding engine(s)."
        )
        if st.button(
            "Queue benchmark campaign",
            type="primary",
            disabled=bool(blockers),
            key="benchmark_campaign_queue",
        ) and dataset_job is not None:
            campaign_id = str(uuid4())
            gpu_device = "all" if gpu == "Automatic" else gpu.replace("GPU ", "")
            queued = []
            try:
                for case_index, case in enumerate(selected_cases):
                    metadata = {
                        "benchmark_dataset_run_id": dataset_job.run_id,
                        "benchmark_campaign_id": campaign_id,
                        "benchmark_case_id": case.case_id,
                    }
                    center, size = _ligand_box(case.ligand_path)
                    uses_bound_target = any(redock_enabled.values()) or use_boltz or use_af3
                    bound_chain = None
                    bound_target = None
                    bound_artifact = None
                    if uses_bound_target:
                        bound_chain = ligand_bound_protein_sequence(
                            case.receptor_path,
                            case.ligand_path,
                        )
                        bound_target = create_benchmark_bound_chain_target(
                            case,
                            bound_chain,
                            campaign_id=campaign_id,
                        )
                        bound_artifact = bound_target.artifact_manifest.by_type(
                            "benchmark_bound_receptor"
                        )[0]
                        metadata.update(
                            {
                                "benchmark_receptor_chain_policy": "single_reference_ligand_bound_protein_chain",
                                "benchmark_selected_receptor_chain": bound_chain["chain"],
                                "benchmark_bound_target_run_id": bound_target.run_id,
                                "benchmark_bound_target_artifact_id": bound_artifact.artifact_id,
                            }
                        )
                    fixed = [
                        ENGINE_IDS[name]
                        for name in ("AutoDock Vina", "GNINA", "Uni-Dock Pro")
                        if redock_enabled[name]
                    ]
                    if fixed:
                        job = queue_redocking_benchmark(
                            receptor_path=bound_artifact.resolve(
                                bound_target.run_dir, must_exist=True
                            ),
                            target_artifact=bound_artifact,
                            target_task_group=bound_target.task_group,
                            reference_ligand_path=case.ligand_path,
                            reference_ligand_artifact=case.ligand_artifact,
                            reference_task_group=dataset_job.task_group,
                            center=center,
                            size=size,
                            engines=fixed,
                            replicates=replicates,
                            seed_start=seed + case_index * replicates,
                            image=DEFAULT_DOCKING_IMAGE,
                            gpu_device=gpu_device,
                            exhaustiveness=exhaustiveness,
                            poses=poses,
                            context_metadata={**metadata, "benchmark_mode": "redocking"},
                        )
                        _annotate_redocking_tree(
                            job, {**metadata, "benchmark_mode": "redocking"}
                        )
                        queued.append(job)
                    if redock_enabled["RosettaLigand"]:
                        job = queue_openvs_docking_job(
                            receptor_path=bound_artifact.resolve(
                                bound_target.run_dir, must_exist=True
                            ),
                            target_artifact=bound_artifact,
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
                            seed_start=seed + case_index * replicates,
                            maximum_compounds=1,
                            launch_campaign_id=campaign_id,
                            launch_campaign_label=f"{dataset_job.metadata.get('dataset_name')} benchmark",
                        )
                        _write_metadata(job, {**metadata, "benchmark_mode": "redocking"})
                        queued.append(job)
                    common = {
                        "target_path": (
                            bound_artifact.resolve(bound_target.run_dir, must_exist=True)
                            if bound_artifact is not None
                            else case.receptor_path
                        ),
                        "target_artifact": bound_artifact or case.receptor_artifact,
                        "compound_paths": (case.ligand_path,),
                        "compound_artifacts": (case.ligand_artifact,),
                        "gpu_device": gpu_device,
                        "max_compounds": 1,
                        "launch_campaign_id": campaign_id,
                        "launch_campaign_label": f"{dataset_job.metadata.get('dataset_name')} benchmark",
                    }
                    refold_metadata = {
                        **metadata,
                        "benchmark_mode": "refolding",
                        "benchmark_reference_ligand_artifact_id": case.ligand_artifact.artifact_id,
                    }
                    if use_boltz or use_af3:
                        refold_metadata.update(
                            {
                                "benchmark_refolding_chain_policy": "single_reference_ligand_bound_protein_chain",
                                "benchmark_refolding_selected_chain": bound_chain["chain"],
                                "benchmark_refolding_chain_contact_cutoff_angstrom": bound_chain[
                                    "contact_cutoff_angstrom"
                                ],
                                "benchmark_refolding_chain_contacting_atoms": bound_chain[
                                    "contacting_protein_atoms"
                                ],
                                "benchmark_refolding_chain_minimum_distance_angstrom": bound_chain[
                                    "minimum_distance_angstrom"
                                ],
                                "benchmark_refolding_available_protein_chains": bound_chain[
                                    "available_protein_chains"
                                ],
                            }
                        )
                    if use_boltz:
                        job = queue_boltz2_refolding_job(
                            **common,
                            protein_sequences=(
                                (bound_chain["chain"], bound_chain["sequence"]),
                            ),
                            reference_ligand_artifact=case.ligand_artifact,
                            image=DEFAULT_BOLTZ2_IMAGE,
                            sampling_steps=boltz_steps,
                            recycling_steps=recycles,
                            replicates=replicates,
                            seed_start=seed + case_index * replicates,
                            use_msa_server=False,
                        )
                        _write_metadata(job, refold_metadata)
                        queued.append(job)
                    if use_af3:
                        job = queue_alphafold3_refolding_job(
                            **common,
                            protein_sequences=(
                                (bound_chain["chain"], bound_chain["sequence"]),
                            ),
                            reference_ligand_artifact=case.ligand_artifact,
                            image=DEFAULT_ALPHAFOLD3_IMAGE,
                            num_recycles=recycles,
                            model_seed_count=replicates,
                            model_seed_start=seed + case_index * replicates,
                        )
                        _write_metadata(job, refold_metadata)
                        queued.append(job)
                    if use_nesso:
                        nesso_common = dict(common)
                        nesso_common.pop("launch_campaign_id")
                        nesso_common.pop("launch_campaign_label")
                        job = queue_nesso_affinity_job(
                            **nesso_common,
                            image=DEFAULT_NESSO_IMAGE,
                            recycling_steps=recycles,
                            replicates=replicates,
                            seed=seed + case_index * replicates,
                            launch_campaign_id=campaign_id,
                            launch_campaign_label=f"{dataset_job.metadata.get('dataset_name')} benchmark",
                        )
                        _write_metadata(job, refold_metadata)
                        queued.append(job)
                if use_gnina_score or use_boltzina:
                    for source in sources:
                        ranks = sorted(
                            {
                                int(row["source_pose_rank"])
                                for row in source_pose_rows(source)
                            }
                        )
                        selection = create_pose_selection_job(source, pose_ranks=ranks)
                        rescore_metadata = {
                            "benchmark_dataset_run_id": dataset_job.run_id,
                            "benchmark_campaign_id": campaign_id,
                            "benchmark_case_id": source.metadata.get(
                                "benchmark_case_id", ""
                            ),
                            "benchmark_mode": "rescoring",
                            "benchmark_source_docking_run_id": source.run_id,
                        }
                        _write_metadata(selection, rescore_metadata)
                        if use_gnina_score:
                            job = queue_gnina_rescoring_job(
                                selection_job=selection, gpu_device=gpu_device
                            )
                            _write_metadata(job, rescore_metadata)
                            queued.append(job)
                        if use_boltzina:
                            context, _ = _boltz_context(
                                dataset_job.run_id,
                                str(source.metadata.get("benchmark_case_id") or ""),
                            )
                            if context is not None:
                                job = queue_boltzina_rescoring_job(
                                    selection_job=selection,
                                    boltz_work_dir=context,
                                    gpu_device=gpu_device,
                                )
                                _write_metadata(job, rescore_metadata)
                                queued.append(job)
                st.success(
                    f"Queued campaign {campaign_id[:8]} with {len(queued)} jobs."
                )
            except Exception as exc:
                st.error(f"Could not queue benchmark campaign: {exc}")

    with results_tab:
        st.subheader("Redocking / Refolding results")
        st.caption(
            "All jobs for the selected benchmark dataset are shown independently "
            "of the currently selected cases and engines."
        )
        if dataset_job is None:
            st.info("Select an imported benchmark dataset first.")
        else:
            if st.button("Refresh", key="benchmark_campaign_results_refresh"):
                st.rerun()
            result_frame = campaign_table(dataset_job.run_id)
            if result_frame.empty:
                st.info("No redocking, refolding, or rescoring jobs exist yet.")
            else:
                statuses = st.multiselect(
                    "Status",
                    sorted(result_frame["Status"].astype(str).unique()),
                    default=sorted(result_frame["Status"].astype(str).unique()),
                    key="benchmark_campaign_result_status",
                )
                displayed = result_frame.loc[
                    result_frame["Status"].astype(str).isin(statuses)
                ]
                st.dataframe(
                    displayed,
                    hide_index=True,
                    width="stretch",
                    height=560,
                    column_config={
                        "Result": st.column_config.LinkColumn(
                            "Result", display_text="Open"
                        ),
                        "Created": st.column_config.DatetimeColumn(
                            format="YYYY-MM-DD HH:mm"
                        ),
                    },
                )
                st.caption(
                    "Open any row for its dedicated Overview, Artifacts, Metrics, "
                    "Viewer, Lineage, and Logs tabs."
                )
            st.link_button(
                "Open aggregate dataset statistics and RMSD",
                f"./benchmark-results?dataset_run_id={dataset_job.run_id}",
            )


if __name__ == "__main__":
    render()
