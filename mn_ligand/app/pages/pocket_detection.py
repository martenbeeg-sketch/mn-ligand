from __future__ import annotations

import json
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.discover_inputs import (
    render_target_viewer,
    select_target_artifact,
)
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import display_job_code, iter_job_records
from mn_ligand.core.provenance import target_lineage_summary
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.pocket_detection import (
    BOUND_LIGAND_METHOD,
    DEFAULT_FPOCKET_IMAGE,
    DEFAULT_P2RANK_IMAGE,
    DEFAULT_PESTO_IMAGE,
    bound_ligand_candidates,
    list_pocket_detection_jobs,
    pesto_readiness,
    queue_pocket_detection_job,
)


ENGINE_STATE_KEYS = {
    "Bound ligand": "pocket_detection_enable_bound_ligand",
    "fpocket": "pocket_detection_enable_fpocket",
    "P2Rank": "pocket_detection_enable_p2rank",
    "PeSTo": "pocket_detection_enable_pesto",
}


def _job_url(run_id: str, code: str) -> str:
    query = urlencode({"task_group": "pocket-detection", "run_id": run_id, "label": code})
    return f"./job-results?{query}"


def _job_target_run_id(job) -> str:
    return str(
        job.metadata.get("prepared_target_run_id")
        or job.metadata.get("parent_run_id")
        or job.parent_run_id
        or ""
    )


def _bound_ligand_label(
    candidate: dict[str, object],
    target_choice,
) -> str:
    key = str(candidate.get("key") or "")
    chemical: dict[str, object] = {}
    ligands = target_choice.job.metadata.get("ligands")
    if isinstance(ligands, list):
        chemical = next(
            (
                dict(item)
                for item in ligands
                if isinstance(item, dict)
                and str(item.get("key") or "") == key
            ),
            {},
        )
    ligand_id = str(
        chemical.get("ccd_id")
        or chemical.get("resname")
        or candidate.get("resname")
        or "LIG"
    )
    name = str(chemical.get("name") or "").strip()
    coordinate = (
        f"{candidate.get('resname', 'LIG')} "
        f"{candidate.get('chain', '_')}:{candidate.get('resseq', '')}"
    )
    identity = f"{ligand_id} — {name}" if name else ligand_id
    return f"{identity} ({coordinate})"


def _parameter_summary(job) -> str:
    try:
        payload = json.loads((job.run_dir / "input.json").read_text())
    except (OSError, ValueError, TypeError):
        return "—"
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        return "—"
    method = str(payload.get("method") or job.tool or "").lower()
    keys = {
        "fpocket": ("max_pockets", "min_score", "box_padding_angstrom"),
        "p2rank": (
            "p2rank_profile",
            "p2rank_min_probability",
            "max_pockets",
            "box_padding_angstrom",
        ),
        "pesto": (
            "pesto_score_threshold",
            "pesto_cluster_distance_angstrom",
            "pesto_minimum_residues",
            "max_pockets",
        ),
    }.get(method, ())
    values = [
        f"{key.removeprefix(method + '_').replace('_', ' ')}={parameters[key]}"
        for key in keys
        if parameters.get(key) is not None
    ]
    return ", ".join(values) or "defaults"


def _target_pocket_annotations() -> dict[str, dict[str, object]]:
    all_jobs = iter_job_records(runs_root())
    jobs_by_id = {job.run_id: job for job in all_jobs}
    latest_by_target: dict[str, dict[str, object]] = {}
    for job in list_pocket_detection_jobs():
        source_target_run_id = _job_target_run_id(job)
        if not source_target_run_id:
            continue
        method = str(job.result.get("method") or job.tool or "unknown").strip().lower()
        method_label = {
            "fpocket": "fpocket",
            "p2rank": "P2Rank",
            "pesto": "PeSTo",
        }.get(method, method or "unknown")
        pocket_count = job.result.get("pocket_count")
        count_label = (
            f", {int(pocket_count)} pockets"
            if isinstance(pocket_count, (int, float))
            else ""
        )
        source_job = jobs_by_id.get(source_target_run_id)
        source_code = (
            display_job_code(source_job.metadata.get("job_code"), source_job.run_id)
            if source_job is not None
            else source_target_run_id[:5].upper()
        )
        current_run_id = source_target_run_id
        visited: set[str] = set()
        while current_run_id and current_run_id not in visited:
            visited.add(current_run_id)
            target_methods = latest_by_target.setdefault(current_run_id, {})
            if method_label not in target_methods:
                target_methods[method_label] = (
                    f"{job.status}{count_label} · target job {source_code}"
                )
            current_job = jobs_by_id.get(current_run_id)
            if current_job is None:
                break
            current_run_id = str(
                current_job.parent_run_id
                or current_job.metadata.get("source_structure_run_id")
                or current_job.metadata.get("import_run_id")
                or ""
            )
    return {
        target_run_id: {
            "Pocket Detection": "; ".join(
                f"{method}: {status}"
                for method, status in sorted(methods.items())
            )
        }
        for target_run_id, methods in latest_by_target.items()
    }


def _job_rows(
    statuses: set[str],
    *,
    prepared_target_run_id: str = "",
) -> list[dict[str, object]]:
    all_jobs = iter_job_records(runs_root())
    jobs_by_id = {job.run_id: job for job in all_jobs}
    rows: list[dict[str, object]] = []
    for job in list_pocket_detection_jobs():
        if job.status not in statuses:
            continue
        if prepared_target_run_id and _job_target_run_id(job) != prepared_target_run_id:
            continue
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        context = target_lineage_summary(job, jobs_by_id)
        rows.append(
            {
                "job": _job_url(job.run_id, code),
                "Last step": context["last_step"],
                "status": job.status,
                "target": context["target"],
                "receptor": context["receptor"],
                "ligand": context["ligand"],
                "origin / history": context["origin"],
                "engine": job.result.get("method") or job.tool,
                "pockets": job.result.get("pocket_count"),
                "settings": _parameter_summary(job),
                "created": job.created_at,
            }
        )
    return rows


def _render_job_table(rows: list[dict[str, object]], empty_message: str) -> None:
    if not rows:
        st.info(empty_message)
        return
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"job": st.column_config.LinkColumn("Job", display_text=r"label=([^&]+)")},
    )


def render() -> None:
    st.title("Pocket Detection and Extraction")
    st.caption(
        "Extract a molecule-generation pocket from a known bound ligand or "
        "predict candidate pockets for apo and alternative-site workflows."
    )
    target_tab, tool_tab, run_tab, results_tab = st.tabs(["Target", "Tool", "Run", "Results"])

    with target_tab:
        requested_run_id = str(st.query_params.get("prepared_target_run_id", "")).strip()
        target_choice = select_target_artifact(
            "Prepared target",
            ("prepared_target", "prepared_receptor"),
            key="pocket_detection_target",
            requested_run_id=requested_run_id,
            show_viewer=False,
            row_annotations=_target_pocket_annotations(),
            row_annotation_defaults={"Pocket Detection": "Not run"},
        )
        if target_choice is not None:
            receptor_path = target_choice.artifact.resolve(target_choice.job.run_dir, must_exist=True)
            render_target_viewer(
                target_choice,
                viewer_path=receptor_path,
                show_ligand=False,
                show_box=False,
                key="pocket_detection_target_viewer",
            )
        else:
            st.info("Prepare a target before detecting pockets.")

    with tool_tab:
        pesto_status = pesto_readiness()
        try:
            bound_candidates = (
                bound_ligand_candidates(target_choice.job.run_id)
                if target_choice is not None
                else []
            )
        except (OSError, ValueError):
            bound_candidates = []
        st.markdown("#### Pocket construction methods")
        st.caption(
            "Select one or more methods. Bound-ligand extraction creates a "
            "deterministic pocket for molecule generation from an existing "
            "protein–ligand complex. fpocket, P2Rank, and PeSTo predict candidate "
            "sites when a bound ligand is absent or alternative sites are wanted."
        )
        action_columns = st.columns([0.16, 0.16, 0.68])
        if action_columns[0].button("Select all", key="pocket_detection_select_all"):
            for engine, state_key in ENGINE_STATE_KEYS.items():
                st.session_state[state_key] = (
                    (engine != "PeSTo" or bool(pesto_status["ready"]))
                    and (engine != "Bound ligand" or bool(bound_candidates))
                )
        if action_columns[1].button("Clear all", key="pocket_detection_clear_all"):
            for state_key in ENGINE_STATE_KEYS.values():
                st.session_state[state_key] = False
        engine_columns = st.columns(4)
        bound_ligand_enabled = engine_columns[0].checkbox(
            "Bound ligand — generation pocket",
            value=bool(bound_candidates),
            key=ENGINE_STATE_KEYS["Bound ligand"],
            disabled=not bool(bound_candidates),
            help=(
                "Extract the ligand envelope and nearby protein residues from "
                "the exact prepared complex. Recommended for ligand-centered "
                "molecule-generation campaigns."
            ),
        )
        fpocket_enabled = engine_columns[1].checkbox(
            "fpocket",
            value=not bool(bound_candidates),
            key=ENGINE_STATE_KEYS["fpocket"],
            help="CPU geometric pocket detection.",
        )
        p2rank_enabled = engine_columns[2].checkbox(
            "P2Rank",
            value=False,
            key=ENGINE_STATE_KEYS["P2Rank"],
            help="CPU machine-learning pocket detection.",
        )
        pesto_enabled = engine_columns[3].checkbox(
            "PeSTo",
            value=False,
            key=ENGINE_STATE_KEYS["PeSTo"],
            disabled=not bool(pesto_status["ready"]),
            help="GPU ligand-interface residue prediction.",
        )
        if not pesto_status["ready"]:
            st.warning(
                "PeSTo is unavailable because the i_v4_1 checkpoint is missing from "
                "the configured reference directory."
            )

        bound_ligand_key = ""
        bound_box_padding = 4.0
        bound_lining_cutoff = 5.0
        with st.expander(
            "Bound ligand — molecule-generation pocket settings",
            expanded=bound_ligand_enabled,
        ):
            if bound_candidates and target_choice is not None:
                candidate_by_key = {
                    str(candidate.get("key") or ""): candidate
                    for candidate in bound_candidates
                }
                preferred_chemical_id = str(
                    target_choice.job.metadata.get("ligand_key") or ""
                ).partition("|")[0]
                chemical_ligands = target_choice.job.metadata.get("ligands")
                preferred_coordinate_key = next(
                    (
                        str(item.get("key") or "")
                        for item in chemical_ligands
                        if isinstance(item, dict)
                        and preferred_chemical_id
                        in {
                            str(item.get("ccd_id") or ""),
                            str(item.get("resname") or ""),
                        }
                    ),
                    "",
                ) if isinstance(chemical_ligands, list) else ""
                candidate_keys = list(candidate_by_key)
                preferred_index = (
                    candidate_keys.index(preferred_coordinate_key)
                    if preferred_coordinate_key in candidate_keys
                    else 0
                )
                bound_ligand_key = st.selectbox(
                    "Bound ligand defining the generation pocket",
                    candidate_keys,
                    index=preferred_index,
                    format_func=lambda key: _bound_ligand_label(
                        candidate_by_key[key],
                        target_choice,
                    ),
                    disabled=not bound_ligand_enabled,
                    key=(
                        "pocket_detection_bound_ligand_"
                        f"{target_choice.job.run_id}"
                    ),
                )
                bound_columns = st.columns(2)
                bound_box_padding = float(
                    bound_columns[0].number_input(
                        "Ligand-envelope padding (Å)",
                        min_value=0.0,
                        max_value=20.0,
                        value=4.0,
                        step=0.5,
                        disabled=not bound_ligand_enabled,
                        key="pocket_detection_bound_padding",
                        help=(
                            "Added to both sides of the heavy-atom bounding "
                            "box used by downstream spatial workflows."
                        ),
                    )
                )
                bound_lining_cutoff = float(
                    bound_columns[1].number_input(
                        "Protein lining cutoff (Å)",
                        min_value=1.0,
                        max_value=15.0,
                        value=5.0,
                        step=0.5,
                        disabled=not bound_ligand_enabled,
                        key="pocket_detection_bound_cutoff",
                        help=(
                            "A residue is retained when any atom lies within "
                            "this distance of a ligand heavy atom."
                        ),
                    )
                )
                st.info(
                    "Purpose: molecule generation around a known ligand site. "
                    "This method is not pocket prediction. It writes the lining "
                    "protein atoms, ligand-coordinate points, box geometry, and "
                    "exact source-complex provenance as an immutable pocket job."
                )
            else:
                st.info(
                    "The selected target has no coordinate-bearing bound ligand. "
                    "Import or select a prepared protein–ligand complex, or use "
                    "fpocket, P2Rank, or PeSTo to predict candidate sites."
                )

        fpocket_max_pockets = 10
        fpocket_min_score = None
        fpocket_box_padding = 4.0
        with st.expander("fpocket settings", expanded=fpocket_enabled):
            columns = st.columns(3)
            fpocket_max_pockets = int(columns[0].number_input(
                "Maximum pockets",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                disabled=not fpocket_enabled,
                key="pocket_detection_fpocket_maximum",
            ))
            fpocket_filter_score = columns[1].checkbox(
                "Minimum score",
                value=False,
                disabled=not fpocket_enabled,
                key="pocket_detection_fpocket_filter_score",
            )
            fpocket_score_value = columns[1].number_input(
                "Score threshold",
                value=0.0,
                step=0.05,
                disabled=not fpocket_enabled or not fpocket_filter_score,
                key="pocket_detection_fpocket_score",
            )
            fpocket_min_score = (
                float(fpocket_score_value) if fpocket_filter_score else None
            )
            fpocket_box_padding = float(columns[2].number_input(
                "Docking-box padding (A)",
                min_value=0.0,
                max_value=20.0,
                value=4.0,
                step=0.5,
                disabled=not fpocket_enabled,
                key="pocket_detection_fpocket_padding",
            ))

        p2rank_profile = "default"
        p2rank_min_probability = None
        p2rank_max_pockets = 10
        p2rank_box_padding = 4.0
        with st.expander("P2Rank settings", expanded=p2rank_enabled):
            columns = st.columns(4)
            profile_label = columns[0].selectbox(
                "Structure profile",
                ("Experimental X-ray", "Predicted / NMR / cryo-EM"),
                disabled=not p2rank_enabled,
                key="pocket_detection_p2rank_profile",
            )
            p2rank_profile = (
                "alphafold"
                if profile_label == "Predicted / NMR / cryo-EM"
                else "default"
            )
            p2rank_filter_probability = columns[1].checkbox(
                "Minimum probability",
                value=False,
                disabled=not p2rank_enabled,
                key="pocket_detection_p2rank_filter_probability",
            )
            p2rank_probability_value = columns[1].slider(
                "Probability threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                disabled=not p2rank_enabled or not p2rank_filter_probability,
                key="pocket_detection_p2rank_probability",
            )
            p2rank_min_probability = (
                float(p2rank_probability_value)
                if p2rank_filter_probability
                else None
            )
            p2rank_max_pockets = int(columns[2].number_input(
                "Maximum P2Rank pockets",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                disabled=not p2rank_enabled,
                key="pocket_detection_p2rank_maximum",
            ))
            p2rank_box_padding = float(columns[3].number_input(
                "P2Rank box padding (A)",
                min_value=0.0,
                max_value=20.0,
                value=4.0,
                step=0.5,
                disabled=not p2rank_enabled,
                key="pocket_detection_p2rank_padding",
            ))
            st.caption(
                "P2Rank probabilities use the selected model profile and should not "
                "be compared across profiles. This adapter is CPU-only."
            )

        pesto_score_threshold = 0.5
        pesto_cluster_distance = 8.0
        pesto_minimum_residues = 3
        pesto_max_pockets = 10
        gpu_device = "all"
        with st.expander("PeSTo settings", expanded=pesto_enabled):
            columns = st.columns(4)
            pesto_score_threshold = float(columns[0].slider(
                "Ligand-interface score",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                disabled=not pesto_enabled,
                key="pocket_detection_pesto_threshold",
            ))
            pesto_cluster_distance = float(columns[1].number_input(
                "Maximum site span (A)", min_value=2.0, max_value=20.0, value=8.0,
                step=0.5, disabled=not pesto_enabled,
                key="pocket_detection_pesto_cluster_distance",
            ))
            pesto_minimum_residues = int(columns[2].number_input(
                "Minimum residues", min_value=1, max_value=100, value=3,
                step=1, disabled=not pesto_enabled,
                key="pocket_detection_pesto_minimum_residues",
            ))
            pesto_max_pockets = int(columns[3].number_input(
                "Maximum sites", min_value=1, max_value=100, value=10,
                step=1, disabled=not pesto_enabled,
                key="pocket_detection_pesto_maximum_sites",
            ))
            gpu_columns = st.columns(2)
            gpu_label = gpu_columns[0].selectbox(
                "GPU", ("Automatic", "GPU 0", "GPU 1"), index=0,
                disabled=not pesto_enabled,
                key="pocket_detection_pesto_gpu",
            )
            gpu_device = "all" if gpu_label == "Automatic" else gpu_label.removeprefix("GPU ")
            st.caption(
                "PeSTo reports ligand-interface scores. The app groups spatially "
                "adjacent high-scoring residues into pocket indicators."
            )

    with run_tab:
        selected_engines = [
            engine
            for engine, enabled in (
                ("Bound ligand", bound_ligand_enabled),
                ("fpocket", fpocket_enabled),
                ("P2Rank", p2rank_enabled),
                ("PeSTo", pesto_enabled),
            )
            if enabled
        ]
        render_run_resources(
            requires_gpu=pesto_enabled,
            selected_gpu=(
                str(st.session_state.get("pocket_detection_pesto_gpu") or "Automatic")
                if pesto_enabled
                else "Not used"
            ),
            key="pocket_detection",
        )
        queue_notice = str(
            st.session_state.pop("pocket_detection_queue_notice", "")
        )
        queue_errors = list(
            st.session_state.pop("pocket_detection_queue_errors", [])
        )
        if queue_notice:
            st.success(queue_notice)
        for queue_error in queue_errors:
            st.error(str(queue_error))
        if target_choice is not None:
            st.markdown("#### Submission")
            st.dataframe(
                pd.DataFrame([
                    {
                        "target": (
                            target_choice.job.metadata.get("pdb_id")
                            or target_choice.artifact.label
                        ),
                        "artifact": target_choice.artifact.artifact_type,
                        "target job": display_job_code(
                            target_choice.job.metadata.get("job_code"),
                            target_choice.job.run_id,
                        ),
                        "engines": ", ".join(selected_engines) or "None selected",
                    }
                ]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("A prepared target is required.")
            st.link_button("Open Structure Import", "./workspace-structure-preparation")
        if not selected_engines:
            st.info("Select at least one engine in the Tool tab.")
        can_run = target_choice is not None and bool(selected_engines)
        if st.button(
            "Run selected engines",
            type="primary",
            disabled=not can_run,
            key="pocket_detection_run",
        ):
            queued = []
            failures = []
            for engine in selected_engines:
                try:
                    if engine == "Bound ligand":
                        job = queue_pocket_detection_job(
                            target_choice.job.run_id,
                            image=DEFAULT_FPOCKET_IMAGE,
                            method=BOUND_LIGAND_METHOD,
                            max_pockets=1,
                            bound_ligand_key=bound_ligand_key,
                            box_padding_angstrom=bound_box_padding,
                            lining_cutoff_angstrom=bound_lining_cutoff,
                        )
                    elif engine == "fpocket":
                        job = queue_pocket_detection_job(
                            target_choice.job.run_id,
                            image=DEFAULT_FPOCKET_IMAGE,
                            method="fpocket",
                            max_pockets=fpocket_max_pockets,
                            min_score=fpocket_min_score,
                            box_padding_angstrom=fpocket_box_padding,
                        )
                    elif engine == "P2Rank":
                        job = queue_pocket_detection_job(
                            target_choice.job.run_id,
                            image=DEFAULT_P2RANK_IMAGE,
                            method="p2rank",
                            max_pockets=p2rank_max_pockets,
                            box_padding_angstrom=p2rank_box_padding,
                            p2rank_profile=p2rank_profile,
                            p2rank_min_probability=p2rank_min_probability,
                        )
                    else:
                        job = queue_pocket_detection_job(
                            target_choice.job.run_id,
                            image=DEFAULT_PESTO_IMAGE,
                            method="pesto",
                            max_pockets=pesto_max_pockets,
                            pesto_score_threshold=pesto_score_threshold,
                            pesto_cluster_distance_angstrom=pesto_cluster_distance,
                            pesto_minimum_residues=pesto_minimum_residues,
                            gpu_device=gpu_device,
                        )
                except (OSError, ValueError) as exc:
                    failures.append(f"{engine}: {exc}")
                    continue
                queued.append(
                    f"{engine} {display_job_code(job.metadata.get('job_code'), job.run_id)}"
                )
            if queued:
                st.session_state["pocket_detection_queue_notice"] = (
                    f"Queued {len(queued)} pocket-detection "
                    f"{'job' if len(queued) == 1 else 'jobs'}: {', '.join(queued)}. "
                    "The local workers own execution."
                )
            if failures:
                st.session_state["pocket_detection_queue_errors"] = [
                    f"Could not queue {failure}" for failure in failures
                ]
            if queued or failures:
                st.rerun()
        st.markdown("#### Active runs")
        _render_job_table(
            _job_rows({"queued", "preparing", "running"}),
            "No pocket-detection jobs are currently running.",
        )

    with results_tab:
        _render_job_table(
            _job_rows({"completed", "failed", "cancelled"}),
            "No pocket-detection results yet.",
        )


render()
