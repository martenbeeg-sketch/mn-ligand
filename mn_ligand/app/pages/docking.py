from __future__ import annotations

import os
import shlex
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.discover_inputs import (
    artifact_box,
    bound_ligand_box,
    render_selected_artifacts,
    render_target_viewer,
    select_artifact,
    select_artifacts,
    select_target_artifact,
    target_viewer_path,
)
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import display_job_code, iter_job_records
from mn_ligand.core.provenance import is_benchmark_job, target_lineage_summary
from mn_ligand.runtime import (
    cpu_process_limit,
    runs_root,
    unidock_pro_max_compounds,
)
from mn_ligand.workflows.docking import (
    DEFAULT_DOCKING_IMAGE,
    queue_docking_campaign_job,
)
from mn_ligand.workflows.openvs import DEFAULT_OPENVS_IMAGE, queue_openvs_docking_job
from mn_ligand.workflows.redocking import queue_redocking_benchmark
from mn_ligand.workflows.refolding import (
    DEFAULT_ALPHAFOLD3_IMAGE,
    DEFAULT_BOLTZ2_IMAGE,
    DEFAULT_NESSO_IMAGE,
    alphafast_readiness,
    boltz2_readiness,
    configured_alphafold3_reference_paths,
    configured_boltz2_cache_dir,
    configured_nesso_reference_paths,
    nesso_readiness,
    queue_alphafold3_refolding_job,
    queue_boltz2_refolding_job,
    queue_nesso_affinity_job,
)


DOCKING_ENGINES = ("Uni-Dock Pro", "AutoDock Vina", "GNINA", "RosettaLigand")
REDOCKING_ENGINES = ("Uni-Dock Pro", "AutoDock Vina", "GNINA")
REFOLDING_ENGINES = ("Boltz-2", "AlphaFold 3", "Nesso-1")
DOCKING_TASK_GROUPS = {"docking", "batch-docking", "structure-docking"}


def _job_rows(statuses: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_jobs = iter_job_records(runs_root())
    jobs_by_id = {job.run_id: job for job in all_jobs}
    for job in all_jobs:
        if is_benchmark_job(job, jobs_by_id):
            continue
        operation = str(job.metadata.get("operation") or job.metadata.get("mode") or "").lower()
        if job.workflow == "redocking_benchmark":
            operation = "redocking"
        if job.task_group not in DOCKING_TASK_GROUPS and operation not in {"docking", "redocking", "refolding"}:
            continue
        if job.status not in statuses:
            continue
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        query = urlencode({"task_group": job.task_group, "run_id": job.run_id, "label": code})
        context = target_lineage_summary(job, jobs_by_id)
        rows.append(
            {
                "result": f"./job-results?{query}",
                "Last step": context["last_step"],
                "job": code,
                "target": context["target"],
                "receptor": context["receptor"],
                "ligand": context["ligand"],
                "origin / history": context["origin"],
                "operation": operation or "docking",
                "engine": job.tool or "Unspecified",
                "status": job.status,
                "progress": str((job.result.get("progress") or {}).get("completed", "")),
                "created": job.created_at,
            }
        )
    return rows


def _render_job_table(rows: list[dict[str, object]]) -> None:
    if not rows:
        st.info("No matching jobs.")
        return
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"result": st.column_config.LinkColumn("Result", display_text="Open")},
    )


def _metadata_box(choice) -> dict[str, tuple[float, float, float]] | None:
    if choice is None:
        return None
    center = choice.job.metadata.get("center")
    size = choice.job.metadata.get("size")
    if not isinstance(center, dict) or not isinstance(size, dict):
        return None
    try:
        return {
            "center": tuple(float(center[axis]) for axis in "xyz"),
            "size": tuple(float(size[axis]) for axis in "xyz"),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _initialize_box(
    target, pocket
) -> tuple[tuple[float, float, float], tuple[float, float, float], str, str]:
    pocket_box = artifact_box(pocket)
    metadata_box = _metadata_box(target)
    ligand_box = None
    if target is not None:
        ligand_key = str(target.job.metadata.get("ligand_key") or "")
        if ligand_key:
            ligand_box = bound_ligand_box(target, ligand_key)
    default_box = pocket_box or metadata_box or ligand_box or {
        "center": (0.0, 0.0, 0.0),
        "size": (22.0, 22.0, 22.0),
    }
    source = "Predicted pocket" if pocket_box else "Stored docking box" if metadata_box else "Bound ligand" if ligand_box else "Manual"
    signature = ":".join(
        (
            target.job.run_id if target is not None else "none",
            pocket.job.run_id if pocket is not None else "none",
            pocket.artifact.artifact_id if pocket is not None else "none",
        )
    )
    return (
        tuple(float(value) for value in default_box["center"]),
        tuple(float(value) for value in default_box["size"]),
        source,
        signature,
    )


def render(
    *, default_operation: str = "Docking", title: str = "Docking / Cofolding"
) -> None:
    st.title(title)
    operation_context = str(
        st.session_state.get("binding_operation") or default_operation
    )
    target_tab, compounds_tab, engine_tab, run_tab, results_tab = st.tabs(
        ["Target", "Compounds", "Engine", "Run", "Results"]
    )

    with target_tab:
        target = select_target_artifact(
            "Prepared target",
            ("prepared_target", "prepared_receptor"),
            key="binding_target",
            show_viewer=False,
        )
        pocket = None
        if operation_context != "Refolding":
            pocket = select_artifact(
                "Pocket",
                ("pocket",),
                key="binding_pocket",
                required=False,
                source_run_id=target.job.run_id if target is not None else "",
            )
        else:
            st.caption(
                "Refolding predicts from the complete target and compound inputs; "
                "it does not consume a docking pocket or docking box."
            )
        selected_target = {}
        if target is not None:
            selected_target["Prepared target"] = target
        if pocket is not None:
            selected_target["Pocket"] = pocket
        render_selected_artifacts(selected_target)
        center, size, box_source, box_signature = _initialize_box(target, pocket)
        if operation_context != "Refolding":
            st.markdown("#### Docking box")
            st.caption(f"Initial box source: {box_source}. Values remain editable for this campaign.")
            center_columns = st.columns(3)
            center_values = []
            for column, axis, value in zip(center_columns, "xyz", center):
                center_values.append(float(column.number_input(
                    f"center_{axis}", value=value, step=0.5, format="%.3f",
                    key=f"binding_center_{axis}_{box_signature}",
                )))
            size_columns = st.columns(3)
            size_values = []
            for column, axis, value in zip(size_columns, "xyz", size):
                size_values.append(float(column.number_input(
                    f"size_{axis}", value=value, min_value=1.0, step=1.0, format="%.2f",
                    key=f"binding_size_{axis}_{box_signature}",
                )))
            center = tuple(center_values)
            size = tuple(size_values)
        if target is not None:
            render_target_viewer(
                target,
                viewer_path=target_viewer_path(target),
                box=(
                    {"center": center, "size": size}
                    if operation_context != "Refolding"
                    else None
                ),
                selected_ligand_key=str(target.job.metadata.get("ligand_key") or ""),
                cartoon_color="#94a3b8",
                ligand_color="cyanCarbon",
                box_color="#0891b2",
                show_box_center=True,
                key="binding_target_viewer",
            )
        if target is None:
            st.info("A prepared target is required. Continue to Run for the preparation link.")
        if pocket is None and operation_context != "Refolding":
            st.caption("A pocket is optional for workflows that use a manual or bound-ligand box.")

    with compounds_tab:
        compounds = select_artifacts(
            "Prepared compound sets",
            ("compound_set", "prepared_ligand_set"),
            key="binding_compounds",
        )
        reference = select_artifact(
            "Reference ligand",
            ("prepared_ligand_set",),
            key="binding_reference",
            required=False,
        )
        selected_compounds = {}
        if compounds:
            selected_compounds["Prepared compound sets"] = compounds
        if reference is not None:
            selected_compounds["Reference ligand"] = reference
        render_selected_artifacts(selected_compounds)
        compound_count = sum(
            int(choice.artifact.metadata.get("compound_count") or choice.job.metadata.get("compound_count") or 1)
            for choice in compounds
        )
        if compounds:
            st.metric("Available compounds", f"{compound_count:,}")
        if not compounds:
            st.info("Prepared compounds are required. Continue to Run for the dataset link.")

    with engine_tab:
        operation = st.segmented_control(
            "Operation", ("Docking", "Redocking benchmark", "Refolding"),
            default=default_operation, key="binding_operation"
        ) or default_operation
        benchmark_engines: list[str] = []
        if operation == "Redocking benchmark":
            benchmark_engines = st.multiselect(
                "Engines",
                REDOCKING_ENGINES,
                default=REDOCKING_ENGINES,
                key="redocking_engines",
                help="Every selected engine is run independently for every replicate.",
            )
            engine = benchmark_engines[0] if benchmark_engines else DOCKING_ENGINES[0]
        else:
            engines = DOCKING_ENGINES if operation == "Docking" else REFOLDING_ENGINES
            engine = st.segmented_control(
                "Engine", engines, default=engines[0], key="binding_engine"
            ) or engines[0]
        search_tab = st.container()
        execution_tab = run_tab
        af3_ready = False
        boltz2_ready = False
        nesso_ready = False
        image = DEFAULT_DOCKING_IMAGE
        docking_mode = "classic"
        search_mode = "detail"
        exhaustiveness = 30
        poses = 10
        use_scrub = True
        scrub_ph = 7.4
        scrub_skip_tautomer = True
        maximum_compounds = 0
        extra_args_text = ""
        openvs_protocol = "vsh"
        openvs_reference_mode = "reference_guided"
        openvs_ph = 7.4
        openvs_padding = 4.0
        openvs_conformers = 20
        openvs_minimization_steps = 2000
        openvs_cpu_workers = min(
            cpu_process_limit(), max(1, os.cpu_count() or 1)
        )
        docking_replicates = 1
        docking_seed_start = 1001
        openvs_cluster_threshold = 2.0
        with search_tab:
            if operation in {"Docking", "Redocking benchmark"}:
                search_columns = st.columns(3)
                if operation == "Redocking benchmark":
                    exhaustiveness = int(search_columns[0].number_input(
                        "Exhaustiveness", min_value=1, max_value=1024, value=30, step=1,
                        key="redocking_exhaustiveness",
                    ))
                    search_mode = search_columns[1].selectbox(
                        "Uni-Dock search mode", ("fast", "balance", "detail"), index=2,
                        key="redocking_search_mode",
                    )
                    poses = int(search_columns[2].number_input(
                        "Poses per replicate", min_value=1, max_value=100, value=10, step=1,
                        key="redocking_poses",
                    ))
                    st.caption(
                        "RMSD uses a symmetry-aware heavy-atom mapping without alignment because "
                        "the crystal ligand and docked poses share the receptor coordinate frame."
                    )
                elif engine == "RosettaLigand":
                    protocol_label = search_columns[0].selectbox(
                        "Protocol",
                        (
                            "VSH — high precision",
                            "VSX — express",
                            "Convergence — exhaustive multi-seed VSH",
                        ),
                        key="openvs_protocol",
                        help="VSH permits flexible pocket side chains; VSX is the faster express preset.",
                    )
                    openvs_protocol = (
                        "vsh"
                        if str(protocol_label).startswith("VSH")
                        else "vsx"
                        if str(protocol_label).startswith("VSX")
                        else "convergence"
                    )
                    reference_label = search_columns[1].selectbox(
                        "Placement",
                        ("Reference-guided", "Pocket-center (unguided)"),
                        index=0,
                        key="openvs_reference_mode",
                        help=(
                            "Reference-guided reproduces the RosettaLigand holo workflow. "
                            "Unguided placement centers an anchor at the selected pocket center."
                        ),
                    )
                    openvs_reference_mode = (
                        "reference_guided"
                        if reference_label == "Reference-guided"
                        else "pocket_center"
                    )
                    openvs_padding = float(search_columns[2].number_input(
                        "Search padding (Å)", min_value=1.0, max_value=20.0,
                        value=4.0, step=0.5, key="openvs_padding",
                        help="GALigandDock uses padding around its ligand-centered grid, not an AutoDock box.",
                    ))
                    prep_columns = st.columns(3)
                    openvs_ph = float(prep_columns[0].number_input(
                        "Preparation pH", min_value=0.0, max_value=14.0,
                        value=7.4, step=0.1, key="openvs_ph",
                    ))
                    openvs_conformers = int(prep_columns[1].number_input(
                        "3D conformer trials", min_value=1, max_value=500,
                        value=20, step=1, key="openvs_conformers",
                    ))
                    openvs_minimization_steps = int(prep_columns[2].number_input(
                        "MMFF94 minimization steps", min_value=1, max_value=10000,
                        value=2000, step=100, key="openvs_minimization_steps",
                    ))
                    repeat_columns = st.columns(2)
                    docking_seed_start = int(repeat_columns[0].number_input(
                        "First docking seed", min_value=1, max_value=2_147_483_000,
                        value=1001, step=1, key="openvs_seed_start",
                        help="Each independent repetition uses the next explicit Rosetta seed.",
                    ))
                    openvs_cluster_threshold = float(
                        repeat_columns[1].number_input(
                            "Pose-cluster threshold (Å)",
                            min_value=0.1,
                            max_value=10.0,
                            value=2.0,
                            step=0.1,
                            key="openvs_cluster_threshold",
                            help=(
                                "Symmetry-aware ligand heavy-atom RMSD in the fixed "
                                "receptor coordinate frame."
                            ),
                        )
                    )
                    if openvs_protocol == "convergence":
                        st.info(
                            "Convergence mode prepares inputs once, runs an exhaustive two-stage "
                            "VSH search for every explicit seed, applies the same side-chain final "
                            "minimization, clusters replicate poses, and reports running mean/SD. "
                            "Its automatic convergence rule requires at least five replicates, "
                            "SD ≤2 REU, mean shift ≤1 REU, and ≥80% dominant-cluster occupancy."
                        )
                    st.caption(
                        "Molecules are prepared inside the RosettaLigand container as protonated, "
                        "MMFF94-charged MOL2 files and Rosetta generic-potential `.params`; "
                        "PDBQT is not used. Scores are reported in relative Rosetta energy units."
                    )
                    st.caption(
                        "Receptor preparation keeps protein ATOM/TER records from the selected "
                        "typed PDB, removes waters and other HETATM records, and appends either "
                        "the selected coordinate-bearing reference ligand or a pocket-centered "
                        "anchor. VSH additionally enables Rosetta pocket-side-chain and hydrogen optimization."
                    )
                elif engine == "Uni-Dock Pro":
                    docking_mode = search_columns[0].selectbox(
                        "Docking mode", ("classic", "hybrid"), key="unidock_mode",
                        help="Hybrid mode uses the selected prepared reference ligand.",
                    )
                    search_mode = search_columns[1].selectbox(
                        "Search mode", ("fast", "balance", "detail"), index=2,
                        key="binding_search_mode",
                    )
                    poses = int(search_columns[2].number_input(
                        "Poses per compound", min_value=1, max_value=100, value=10, step=1,
                        key="binding_poses",
                    ))
                else:
                    exhaustiveness = int(search_columns[0].number_input(
                        "Exhaustiveness", min_value=1, max_value=1024, value=30, step=1,
                        key="binding_exhaustiveness",
                    ))
                    poses = int(search_columns[1].number_input(
                        "Poses per compound", min_value=1, max_value=100, value=10, step=1,
                        key="binding_poses",
                    ))
                    search_columns[2].selectbox(
                        "Scoring", ("Default", "CNN scoring"), index=1 if engine == "GNINA" else 0,
                        disabled=engine != "GNINA", key="binding_scoring",
                    )
                if engine != "RosettaLigand" or operation == "Redocking benchmark":
                    scrub_columns = st.columns(3)
                    use_scrub = scrub_columns[0].checkbox(
                        "Use scrub.py", value=True, key="binding_use_scrub",
                        help="Normalize protonation and ligand chemistry before docking.",
                    )
                    scrub_ph = float(scrub_columns[1].number_input(
                        "Scrub pH", min_value=0.0, max_value=14.0, value=7.4, step=0.1,
                        key="binding_scrub_ph",
                    ))
                    scrub_skip_tautomer = scrub_columns[2].checkbox(
                        "Skip tautomer enumeration", value=True, key="binding_skip_tautomer"
                    )
                if operation == "Docking":
                    dataset_columns = st.columns(2)
                    maximum_compounds = int(dataset_columns[0].number_input(
                        "Maximum compounds",
                        min_value=0,
                        max_value=unidock_pro_max_compounds(),
                        value=0,
                        step=100,
                        help=(
                            "Use 0 to dock every compound when the dataset contains "
                            "no more compounds than the configured Uni-Dock Pro "
                            "batch limit. Change that limit on the Settings page."
                        ),
                        key="binding_maximum_compounds",
                    ))
                    if engine == "RosettaLigand":
                        dataset_columns[1].caption(
                            "RosettaLigand uses bounded protocol controls; arbitrary Rosetta arguments are disabled."
                        )
                    else:
                        extra_args_text = dataset_columns[1].text_input(
                            f"Additional {'UDP' if engine == 'Uni-Dock Pro' else engine} arguments",
                            value="", key="binding_extra_args",
                        )
            else:
                columns = st.columns(3)
                if engine == "Boltz-2":
                    columns[0].number_input(
                        "Recycles", min_value=1, max_value=12, value=3, step=1,
                        key="refolding_recycles",
                    )
                    columns[1].number_input(
                        "Diffusion samples", min_value=1, max_value=16, value=5, step=1,
                        key="refolding_samples",
                    )
                    columns[2].number_input(
                        "Sampling steps", min_value=10, max_value=400, value=200, step=10,
                        key="refolding_sampling_steps",
                    )
                    st.number_input(
                        "Maximum compounds",
                        min_value=0,
                        value=0,
                        step=1,
                        key="boltz2_max_compounds",
                        help="Use 0 to process every compound in the selected datasets.",
                    )
                    st.number_input(
                        "First prediction seed",
                        min_value=1,
                        max_value=2_147_483_000,
                        value=1001,
                        step=1,
                        key="boltz2_seed_start",
                        help=(
                            "Each independent Boltz-2 run uses the next explicit seed. "
                            "Diffusion samples remain within-run samples."
                        ),
                    )
                elif engine == "Nesso-1":
                    st.info(
                        "Nesso-1 predicts affinity from coarse-grained protein-ligand "
                        "cofolding features. It does not emit a predicted complex or pose."
                    )
                    columns[0].number_input(
                        "Recycles", min_value=1, max_value=12, value=5, step=1,
                        key="nesso_recycles",
                    )
                    columns[1].number_input(
                        "Pocket token budget", min_value=32, max_value=2048, value=256, step=32,
                        key="nesso_token_budget",
                    )
                    columns[2].number_input(
                        "Maximum compounds", min_value=0, max_value=10000, value=0, step=1,
                        key="nesso_max_compounds",
                        help="Use 0 to process every compound in the selected datasets.",
                    )
                    cutoff_columns = st.columns(3)
                    cutoff_columns[0].number_input(
                        "Refinement cutoff (Å)", min_value=1.0, max_value=50.0,
                        value=22.0, step=1.0, key="nesso_refine_cutoff",
                    )
                    cutoff_columns[1].number_input(
                        "Affinity cutoff (Å)", min_value=1.0, max_value=50.0,
                        value=15.0, step=1.0, key="nesso_affinity_cutoff",
                    )
                    cutoff_columns[2].number_input(
                        "First prediction seed", min_value=1, max_value=2147483000,
                        value=42, step=1,
                        key="nesso_seed",
                        help="Each independent Nesso run uses the next explicit seed.",
                    )
                else:
                    columns[0].number_input(
                        "Recycles", min_value=1, max_value=48, value=10, step=1,
                        key="af3_recycles",
                    )
                    columns[1].number_input(
                        "Model seeds", min_value=1, max_value=20, value=1, step=1,
                        key="af3_model_seeds",
                    )
                    columns[2].number_input(
                        "Maximum compounds", min_value=0, max_value=10000, value=0, step=1,
                        key="af3_max_compounds",
                        help="Use 0 to process every compound in the selected datasets.",
                    )
        with execution_tab:
            selected_gpu_label = str(st.session_state.get("binding_gpu") or "Automatic")
            render_run_resources(
                requires_gpu=not (operation == "Docking" and engine == "RosettaLigand"),
                selected_gpu=selected_gpu_label,
                key="docking",
            )
            st.markdown("#### Execution")
            columns = st.columns(3)
            if operation == "Docking" and engine == "RosettaLigand":
                openvs_cpu_workers = int(columns[0].number_input(
                    "Parallel CPU workers", min_value=1,
                    max_value=max(1, os.cpu_count() or 1),
                    value=min(
                        cpu_process_limit(), max(1, os.cpu_count() or 1)
                    ),
                    step=1, key="openvs_cpu_workers",
                    help=(
                        "Runs independent ligand/seed tasks concurrently. Each Rosetta process "
                        "is budgeted at approximately 4 GB RAM, so admission may cap large requests."
                    ),
                ))
            else:
                columns[0].number_input(
                    "Batch size", min_value=1, max_value=100000, value=1,
                    step=1, key="binding_batch_size",
                    disabled=operation in {"Docking", "Redocking benchmark"},
                    help="Docking engines consume the prepared compound index directly.",
                )
            if operation == "Docking":
                docking_replicates = int(columns[1].number_input(
                    "Independent runs", min_value=1, max_value=100, value=1, step=1,
                    key="docking_replicates",
                    help=(
                        "Repeats the same prepared campaign with consecutive explicit seeds. "
                        "Mean/SD analysis is emitted when this is greater than one."
                    ),
                ))
            elif operation == "Refolding":
                columns[1].number_input(
                    "Independent runs", min_value=1, max_value=100,
                    value=1, step=1, key="refolding_replicates",
                    disabled=engine == "AlphaFold 3",
                    help=(
                        "Runs the complete prediction repeatedly with consecutive seeds and "
                        "reports mean/sample SD. AlphaFold 3 uses its native Model seeds control."
                    ),
                )
            else:
                columns[1].number_input(
                    "Independent runs", min_value=1, max_value=20,
                    value=3, step=1, key="redocking_replicates",
                )
            columns[2].selectbox(
                "GPU", ("Not used",) if engine == "RosettaLigand" and operation == "Docking"
                else ("Automatic", "GPU 0", "GPU 1"),
                key="binding_gpu",
                disabled=engine == "RosettaLigand" and operation == "Docking",
            )
            if operation in {"Docking", "Redocking benchmark"}:
                default_image = (
                    DEFAULT_OPENVS_IMAGE
                    if operation == "Docking" and engine == "RosettaLigand"
                    else DEFAULT_DOCKING_IMAGE
                )
                image = default_image
                if operation == "Docking" and engine != "RosettaLigand":
                    docking_seed_start = int(st.number_input(
                        "First docking seed",
                        min_value=1,
                        max_value=2_147_483_000,
                        value=1001,
                        step=1,
                        key="binding_docking_seed_start",
                        help="Each independent run uses the next explicit native engine seed.",
                    ))
                elif operation == "Redocking benchmark":
                    docking_seed_start = int(st.number_input(
                        "First docking seed",
                        min_value=1,
                        max_value=2_147_483_000,
                        value=1001,
                        step=1,
                        key="redocking_seed_start",
                        help=(
                            "The same explicit seed sequence is used for every selected "
                            "engine to make benchmark repetitions reproducible."
                        ),
                    ))
            if operation == "Refolding" and engine == "Boltz-2":
                image = DEFAULT_BOLTZ2_IMAGE
                boltz_status = boltz2_readiness()
                boltz2_ready = bool(boltz_status["ready"])
                st.code(str(configured_boltz2_cache_dir()))
                if not boltz2_ready:
                    st.warning("The Boltz-2 structure and affinity checkpoints are missing.")
            elif operation == "Refolding" and engine == "AlphaFold 3":
                image = DEFAULT_ALPHAFOLD3_IMAGE
                af3_db_dir, af3_weights_dir, af3_msa_dir = configured_alphafold3_reference_paths()
                readiness = alphafast_readiness(
                    af3_db_dir,
                    af3_weights_dir,
                    af3_msa_dir,
                )
                af3_ready = bool(readiness["database_ready"] and readiness["weights_ready"])
                if not af3_ready:
                    st.warning("The shared reference directory needs `alignment/mmseqs/` and `alphafold3/` weights.")
                elif readiness["msa_repository_ready"]:
                    st.caption("Shared target MSA repository available; missing sequences use MMseqs-GPU.")
                else:
                    st.caption("No cached target MSAs found; MMseqs-GPU will generate and cache them.")
            elif operation == "Refolding" and engine == "Nesso-1":
                image = DEFAULT_NESSO_IMAGE
                nesso_checkpoint, nesso_ccd, nesso_esm_cache = configured_nesso_reference_paths()
                nesso_status = nesso_readiness(nesso_checkpoint, nesso_ccd, nesso_esm_cache)
                nesso_ready = bool(nesso_status["ready"])
                st.code(
                    "\n".join(
                        (str(nesso_checkpoint), str(nesso_ccd), str(nesso_esm_cache))
                    )
                )
                if not nesso_ready:
                    st.warning(
                        "Nesso requires its v1.0.0 checkpoint, publisher CCD, and the "
                        "ESM-2 650M cache."
                    )
                elif nesso_status["esm_ready"]:
                    st.caption("The existing ESM-2 650M cache will be reused read-only.")

        run_actions = run_tab.container()
        can_run_docking = operation == "Docking" and target is not None and bool(compounds)
        if operation == "Docking" and engine == "Uni-Dock Pro" and docking_mode == "hybrid" and reference is None:
            can_run_docking = False
            run_actions.info("Uni-Dock Pro hybrid mode requires a prepared reference ligand from the Compounds tab.")
        if (
            operation == "Docking"
            and engine == "RosettaLigand"
            and openvs_reference_mode == "reference_guided"
            and reference is None
        ):
            can_run_docking = False
            run_actions.info(
                "Reference-guided RosettaLigand requires a coordinate-bearing ligand in the target "
                "coordinate frame. Select it in the Compounds tab or use pocket-center placement."
            )
        can_run_af3 = (
            operation == "Refolding" and engine == "AlphaFold 3" and target is not None
            and bool(compounds) and af3_ready
        )
        can_run_boltz2 = (
            operation == "Refolding" and engine == "Boltz-2" and target is not None
            and bool(compounds) and boltz2_ready
        )
        can_run_nesso = (
            operation == "Refolding" and engine == "Nesso-1" and target is not None
            and bool(compounds) and nesso_ready
        )
        can_run_redocking = (
            operation == "Redocking benchmark"
            and target is not None
            and reference is not None
            and bool(benchmark_engines)
        )
        if operation == "Redocking benchmark" and reference is None:
            run_actions.info("Select one coordinate-bearing crystallographic reference ligand in the Compounds tab.")
        if target is None:
            run_actions.link_button(
                "Open Structure Import", "./workspace-structure-preparation"
            )
        if not compounds:
            run_actions.link_button(
                "Open Compound Datasets", "./prepare-compound-datasets"
            )
        run_label = (
            "Run redocking benchmark"
            if operation == "Redocking benchmark"
            else f"Run {operation.lower()} with {engine}"
        )
        run_clicked = run_actions.button(
            run_label,
            type="primary",
            disabled=not (
                can_run_docking or can_run_redocking or can_run_af3 or can_run_boltz2 or can_run_nesso
            ),
            help=(
                "Queues the selected typed complex-refolding campaign for the local GPU worker."
                if operation == "Refolding"
                else "Runs the selected prepared compound datasets through the docking campaign adapter."
                if operation == "Docking"
                else "Queues the selected scientific campaign."
            ),
            key="binding_queue",
        )
        if run_clicked and target is not None:
            target_path = target.artifact.resolve(target.job.run_dir, must_exist=True)
            compound_paths = [choice.artifact.resolve(choice.job.run_dir, must_exist=True) for choice in compounds]
            if target_path is None or any(path is None for path in compound_paths):
                run_actions.error("One or more selected artifacts are no longer available.")
            else:
                gpu_value = str(st.session_state.get("binding_gpu") or "Automatic")
                gpu_device = gpu_value.removeprefix("GPU ") if gpu_value != "Automatic" else "all"
                try:
                    if operation == "Docking":
                        reference_path = (
                            reference.artifact.resolve(reference.job.run_dir, must_exist=True)
                            if reference is not None else None
                        )
                        if engine == "RosettaLigand":
                            job = queue_openvs_docking_job(
                                receptor_path=target_path,
                                target_artifact=target.artifact,
                                compound_paths=[path for path in compound_paths if path is not None],
                                compound_artifacts=[choice.artifact for choice in compounds],
                                center=center,
                                size=size,
                                protocol=openvs_protocol,
                                reference_mode=openvs_reference_mode,
                                reference_ligand_path=reference_path,
                                reference_ligand_artifact=(
                                    reference.artifact if reference is not None else None
                                ),
                                image=image,
                                cpu_workers=openvs_cpu_workers,
                                ph=openvs_ph,
                                conformers=openvs_conformers,
                                minimization_steps=openvs_minimization_steps,
                                padding=openvs_padding,
                                replicates=docking_replicates,
                                seed_start=docking_seed_start,
                                cluster_threshold_angstrom=openvs_cluster_threshold,
                                maximum_compounds=maximum_compounds,
                            )
                            run_actions.success(
                                "Queued CPU-only RosettaLigand campaign "
                                f"{display_job_code(job.metadata.get('job_code'), job.run_id)} "
                                f"with {int(job.metadata.get('cpu_workers') or openvs_cpu_workers)} "
                                "parallel Rosetta workers and "
                                f"{docking_replicates} independent run"
                                f"{'s' if docking_replicates != 1 else ''}."
                            )
                        else:
                            engine_id = {
                                "AutoDock Vina": "vina",
                                "GNINA": "gnina",
                                "Uni-Dock Pro": "udp",
                            }[engine]
                            docking_parameters = {
                                "receptor_path": target_path,
                                "target_artifact": target.artifact,
                                "compound_paths": [path for path in compound_paths if path is not None],
                                "compound_artifacts": [choice.artifact for choice in compounds],
                                "center": center,
                                "size": size,
                                "engine": engine_id,
                                "image": image,
                                "gpu_device": gpu_device,
                                "mode": docking_mode,
                                "search_mode": search_mode,
                                "exhaustiveness": exhaustiveness,
                                "poses": poses,
                                "use_scrub": use_scrub,
                                "scrub_ph": scrub_ph,
                                "scrub_skip_tautomer": scrub_skip_tautomer,
                                "reference_ligand_path": reference_path,
                                "reference_ligand_artifact": (
                                    reference.artifact if reference is not None else None
                                ),
                                "replicates": docking_replicates,
                                "seed_start": docking_seed_start,
                                "maximum_compounds": maximum_compounds,
                                "extra_args": shlex.split(extra_args_text),
                            }
                            job = queue_docking_campaign_job(**docking_parameters)
                            run_actions.success(
                                f"Queued {engine} docking campaign "
                                f"{display_job_code(job.metadata.get('job_code'), job.run_id)} "
                                f"with {docking_replicates} independent run"
                                f"{'s' if docking_replicates != 1 else ''}. "
                                "The local worker owns execution."
                            )
                    elif operation == "Redocking benchmark":
                        reference_path = reference.artifact.resolve(
                            reference.job.run_dir, must_exist=True
                        ) if reference is not None else None
                        if reference is None or reference_path is None:
                            raise ValueError("The selected reference ligand is unavailable")
                        engine_ids = {
                            "AutoDock Vina": "vina",
                            "GNINA": "gnina",
                            "Uni-Dock Pro": "udp",
                        }
                        job = queue_redocking_benchmark(
                            receptor_path=target_path,
                            target_artifact=target.artifact,
                            target_task_group=target.job.task_group,
                            reference_ligand_path=reference_path,
                            reference_ligand_artifact=reference.artifact,
                            reference_task_group=reference.job.task_group,
                            center=center,
                            size=size,
                            engines=[engine_ids[value] for value in benchmark_engines],
                            replicates=int(st.session_state.get("redocking_replicates", 3)),
                            seed_start=docking_seed_start,
                            image=image,
                            gpu_device=gpu_device,
                            search_mode=search_mode,
                            exhaustiveness=exhaustiveness,
                            poses=poses,
                            use_scrub=use_scrub,
                            scrub_ph=scrub_ph,
                            scrub_skip_tautomer=scrub_skip_tautomer,
                        )
                        run_actions.success(
                            "Queued redocking benchmark "
                            f"{display_job_code(job.metadata.get('job_code'), job.run_id)} with "
                            f"{len(benchmark_engines)} engines and "
                            f"{int(st.session_state.get('redocking_replicates', 3))} replicates."
                        )
                    else:
                        if engine == "AlphaFold 3":
                            job = queue_alphafold3_refolding_job(
                                target_path=target_path,
                                target_artifact=target.artifact,
                                compound_paths=[path for path in compound_paths if path is not None],
                                compound_artifacts=[choice.artifact for choice in compounds],
                                image=image,
                                db_dir=af3_db_dir,
                                weights_dir=af3_weights_dir,
                                msa_repository_dir=af3_msa_dir,
                                gpu_device=gpu_device,
                                max_compounds=int(st.session_state.get("af3_max_compounds", 0)),
                                batch_size=int(st.session_state.get("binding_batch_size", 1)),
                                num_recycles=int(st.session_state.get("af3_recycles", 10)),
                                model_seed_count=int(st.session_state.get("af3_model_seeds", 1)),
                            )
                        elif engine == "Boltz-2":
                            job = queue_boltz2_refolding_job(
                                target_path=target_path,
                                target_artifact=target.artifact,
                                compound_paths=[path for path in compound_paths if path is not None],
                                compound_artifacts=[choice.artifact for choice in compounds],
                                image=image,
                                cache_dir=configured_boltz2_cache_dir(),
                                gpu_device=gpu_device,
                                max_compounds=int(st.session_state.get("boltz2_max_compounds", 0)),
                                recycling_steps=int(st.session_state.get("refolding_recycles", 3)),
                                sampling_steps=int(st.session_state.get("refolding_sampling_steps", 200)),
                                diffusion_samples=int(st.session_state.get("refolding_samples", 1)),
                                replicates=int(st.session_state.get("refolding_replicates", 1)),
                                seed_start=int(st.session_state.get("boltz2_seed_start", 1001)),
                            )
                        else:
                            job = queue_nesso_affinity_job(
                                target_path=target_path,
                                target_artifact=target.artifact,
                                compound_paths=[path for path in compound_paths if path is not None],
                                compound_artifacts=[choice.artifact for choice in compounds],
                                image=image,
                                checkpoint_dir=nesso_checkpoint,
                                ccd_path=nesso_ccd,
                                esm_cache_dir=nesso_esm_cache,
                                gpu_device=gpu_device,
                                max_compounds=int(st.session_state.get("nesso_max_compounds", 0)),
                                recycling_steps=int(st.session_state.get("nesso_recycles", 5)),
                                refine_protein_cutoff=float(st.session_state.get("nesso_refine_cutoff", 22.0)),
                                refine_protein_tokens_budget=int(st.session_state.get("nesso_token_budget", 256)),
                                affinity_protein_cutoff=float(st.session_state.get("nesso_affinity_cutoff", 15.0)),
                                seed=int(st.session_state.get("nesso_seed", 42)),
                                replicates=int(st.session_state.get("refolding_replicates", 1)),
                            )
                        run_actions.success(
                            f"Queued {engine} refolding campaign "
                            f"{display_job_code(job.metadata.get('job_code'), job.run_id)} with "
                            f"{int(job.metadata.get('replicates') or 1)} independent run(s). "
                            "The local worker owns execution."
                        )
                except Exception as exc:
                    run_actions.error(str(exc))

    with run_tab:
        st.markdown("#### Active runs")
        if st.button("Refresh", key="binding_running_refresh"):
            st.rerun()
        _render_job_table(_job_rows({"queued", "preparing", "running"}))

    with results_tab:
        if st.button("Refresh", key="binding_results_refresh"):
            st.rerun()
        _render_job_table(_job_rows({"completed", "failed", "cancelled"}))


if __name__ == "__main__":
    render()
