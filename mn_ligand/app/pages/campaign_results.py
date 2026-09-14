from __future__ import annotations

from collections import Counter
import json
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.analysis_sets import (
    ANALYSIS_SET_SCHEMA_VERSION,
    compound_smiles_signature,
    list_analysis_sets,
    save_analysis_set,
)
from mn_ligand.core.campaign_extensions import (
    SUPPORTED_WORKFLOWS,
    campaign_jobs,
    canonical_campaign_jobs,
    completed_repetitions,
    configured_repetitions,
    extend_campaign_repetitions,
)
from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records
from mn_ligand.core.provenance import target_lineage_summary
from mn_ligand.core.workflows import WorkflowRecord
from mn_ligand.runtime import resolve_run_dir, runs_root
from mn_ligand.workflows.md_simulation import (
    MD_WORKFLOW_TYPE,
    extend_md_simulation,
    extend_md_simulation_duration,
)


def _campaign_type(jobs: list[JobRecord]) -> str:
    purposes = [str(job.metadata.get("campaign_purpose") or "").strip() for job in jobs]
    purposes = [value for value in purposes if value]
    if purposes:
        return Counter(purposes).most_common(1)[0][0].replace("_", " ").title()
    operations = {str(job.metadata.get("operation") or "").strip() for job in jobs}
    operations.discard("")
    return " / ".join(sorted(value.replace("_", " ").title() for value in operations)) or "Workflow campaign"


def _target_key(job: JobRecord) -> str:
    try:
        payload = json.loads((job.run_dir / "input.json").read_text())
    except (OSError, ValueError, TypeError):
        payload = {}
    target = payload.get("target_artifact") or payload.get("target") or {}
    target_metadata = target.get("metadata") if isinstance(target.get("metadata"), dict) else {}
    return str(
        job.metadata.get("target_label")
        or job.metadata.get("pdb_id")
        or target_metadata.get("source_run_id")
        or target.get("run_id")
        or job.parent_run_id
        or "unknown"
    )


def _job_result_url(job: JobRecord) -> str:
    code = display_job_code(job.metadata.get("job_code"), job.run_id)
    return "./job-results?" + urlencode(
        {"task_group": job.task_group, "run_id": job.run_id, "label": code}
    )


def _target_job(job: JobRecord, jobs_by_id: dict[str, JobRecord]) -> JobRecord | None:
    try:
        payload = json.loads((job.run_dir / "input.json").read_text())
    except (OSError, ValueError, TypeError):
        payload = {}
    target = payload.get("target_artifact") or payload.get("target") or {}
    if isinstance(target, dict):
        target_run_id = str(target.get("run_id") or "").strip()
        if target_run_id and target_run_id in jobs_by_id:
            return jobs_by_id[target_run_id]
    for candidate_id in (
        job.metadata.get("prepared_target_run_id"),
        job.parent_run_id,
    ):
        candidate = jobs_by_id.get(str(candidate_id or ""))
        if candidate is not None:
            return candidate
    return None


def _label(campaign_id: str, jobs: list[JobRecord]) -> str:
    labels = [str(job.metadata.get("launch_campaign_label") or "").strip() for job in jobs]
    return next((value for value in labels if value), campaign_id)


def _status_summary(jobs: list[JobRecord]) -> str:
    counts = Counter(job.status for job in jobs)
    return ", ".join(f"{key}: {counts[key]}" for key in ("running", "queued", "completed", "failed", "blocked", "cancelled") if counts[key])


def _aggregate_result_url(campaign_id: str, jobs: list[JobRecord]) -> str:
    purpose = next(
        (
            str(job.metadata.get("campaign_purpose") or "").strip()
            for job in jobs
            if str(job.metadata.get("campaign_purpose") or "").strip()
        ),
        "",
    )
    parameters = {"launch_campaign_id": campaign_id}
    if purpose:
        parameters["campaign_purpose"] = purpose
    return "./compound-campaign-comparison?" + urlencode(parameters)


def _campaign_purpose(jobs: list[JobRecord]) -> str:
    return next(
        (
            str(job.metadata.get("campaign_purpose") or "").strip()
            for job in jobs
            if str(job.metadata.get("campaign_purpose") or "").strip()
        ),
        "",
    )


def _campaign_ligand_signatures(jobs: list[JobRecord]) -> frozenset[str]:
    """Return each canonical reference-ligand identity used by a campaign."""
    smiles_values: list[str] = []
    for job in jobs:
        paths = [
            *sorted((job.run_dir / "inputs").glob("*.json")),
            *sorted((job.run_dir / "data").glob("*.json")),
            *sorted((job.run_dir / "inputs").glob("*.yaml")),
            *sorted((job.run_dir / "inputs").glob("*.yml")),
        ]
        for path in paths:
            try:
                if path.suffix.lower() == ".json":
                    payload = json.loads(path.read_text())
                else:
                    import yaml

                    payload = yaml.safe_load(path.read_text())
            except (ImportError, OSError, TypeError, ValueError):
                continue
            payload = payload if isinstance(payload, dict) else {}
            sequences = payload.get("sequences")
            sequences = sequences if isinstance(sequences, list) else []
            smiles_values.extend(
                str(ligand.get("smiles") or "").strip()
                for sequence in sequences
                if isinstance(sequence, dict)
                and isinstance(sequence.get("ligand"), dict)
                for ligand in [sequence["ligand"]]
                if str(ligand.get("smiles") or "").strip()
            )
    return frozenset(
        signature
        for value in smiles_values
        if (signature := compound_smiles_signature([value]))
    )


def _campaign_target_ligand_labels(
    jobs: list[JobRecord],
    jobs_by_id: dict[str, JobRecord],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return stable, readable target and reference-ligand labels."""
    campaign_id = str(
        next(
            (
                job.metadata.get("launch_campaign_id")
                for job in jobs
                if job.metadata.get("launch_campaign_id")
            ),
            "",
        )
    )
    targets: set[str] = set()
    ligands: set[str] = set()
    for job in canonical_campaign_jobs(campaign_id, jobs):
        target_job = _target_job(job, jobs_by_id)
        context = target_lineage_summary(target_job or job, jobs_by_id)
        target = str(context.get("target") or "").strip()
        if target and target != "—":
            targets.add(target)
        ligand = str(context.get("ligand") or "").strip()
        if ligand and ligand != "—":
            # Provenance keys include chain/residue coordinates after the
            # chemical component identifier, for example T3|A|501|_.
            ligands.add(ligand.split("|", 1)[0])
    return tuple(sorted(targets)), tuple(sorted(ligands))


def _comparison_engine(job: JobRecord) -> str:
    normalized = str(job.tool or "").strip().lower()
    if job.workflow == "docking_campaign":
        return {
            "vina": "AutoDock Vina",
            "gnina": "GNINA",
            "udp": "Uni-Dock Pro",
            "unidock": "Uni-Dock Pro",
        }.get(normalized, str(job.tool or job.workflow))
    if job.workflow == "alphafold3_refolding":
        return "AlphaFold 3"
    if job.workflow == "boltz2_refolding":
        return "Boltz-2"
    if job.workflow == "nesso_affinity":
        return "Nesso-1"
    if job.workflow == "openvs_docking":
        return "RosettaLigand"
    return str(job.tool or job.workflow)


def _analysis_set_selection(
    campaign_ids: list[str],
    grouped: dict[str, list[JobRecord]],
    jobs_by_id: dict[str, JobRecord],
) -> dict[str, object]:
    selected_jobs = [
        job
        for campaign_id in campaign_ids
        for job in canonical_campaign_jobs(campaign_id, grouped[campaign_id])
    ]
    target_run_ids: list[str] = []
    target_launch_pairs: list[str] = []
    for job in selected_jobs:
        target_job = _target_job(job, jobs_by_id)
        target_run_id = str(target_job.run_id if target_job else "")
        launch_id = str(job.metadata.get("launch_campaign_id") or "")
        if target_run_id:
            target_run_ids.append(target_run_id)
            target_launch_pairs.append(f"{launch_id}::{target_run_id}")
    return {
        "schema_version": ANALYSIS_SET_SCHEMA_VERSION,
        "selection_type": "campaign_analysis_set",
        "campaign_purpose": _campaign_purpose(selected_jobs),
        "dataset_run_id": "",
        "dataset": "",
        "target_run_ids": list(dict.fromkeys(target_run_ids)),
        "launch_campaign_ids": list(dict.fromkeys(campaign_ids)),
        "target_launch_pairs": list(dict.fromkeys(target_launch_pairs)),
        "engines": list(
            dict.fromkeys(_comparison_engine(job) for job in selected_jobs)
        ),
        "engine_run_ids": [job.run_id for job in selected_jobs],
        "rescoring_run_ids": [],
    }


def _overview_rows(records: list[JobRecord]) -> tuple[list[dict[str, Any]], dict[str, list[JobRecord]]]:
    grouped: dict[str, list[JobRecord]] = {}
    for job in records:
        campaign_id = str(job.metadata.get("launch_campaign_id") or "").strip()
        if campaign_id:
            grouped.setdefault(campaign_id, []).append(job)
    lineage_root: dict[str, str] = {}
    for campaign_id, jobs in grouped.items():
        parents = [
            str(job.metadata.get("restart_of_campaign_id") or "").strip()
            for job in jobs
        ]
        parents = [value for value in parents if value]
        lineage_root[campaign_id] = (
            Counter(parents).most_common(1)[0][0] if parents else campaign_id
        )
    current_by_root: dict[str, str] = {}
    for root in set(lineage_root.values()):
        candidates = [
            campaign_id
            for campaign_id, candidate_root in lineage_root.items()
            if candidate_root == root
        ]
        current_by_root[root] = max(
            candidates,
            key=lambda campaign_id: max(
                (job.created_at for job in grouped[campaign_id] if job.created_at),
                default="",
            ),
        )

    rows: list[dict[str, Any]] = []
    for campaign_id, jobs in grouped.items():
        canonical = canonical_campaign_jobs(campaign_id, jobs)
        repeat_counts = [configured_repetitions(job) for job in canonical]
        rows.append(
            {
                "Campaign": _label(campaign_id, jobs),
                "Campaign ID": campaign_id[:8],
                "Type": _campaign_type(jobs),
                "Targets": len({_target_key(job) for job in canonical}),
                "Engines": ", ".join(sorted({job.tool or job.workflow for job in canonical})),
                "Repeat range": (f"{min(repeat_counts)}–{max(repeat_counts)}" if repeat_counts else "—"),
                "Jobs": len(canonical),
                "Status": _status_summary(canonical),
                "Created": min((job.created_at for job in jobs if job.created_at), default=""),
                "_campaign_id": campaign_id,
                "_superseded": current_by_root.get(lineage_root[campaign_id]) != campaign_id,
            }
        )
    for parent in records:
        if parent.task_group != "workflows" or parent.workflow != MD_WORKFLOW_TYPE:
            continue
        try:
            workflow = WorkflowRecord.load(parent.run_id)
        except (OSError, ValueError):
            continue
        children: list[JobRecord] = []
        for child in workflow.children:
            child_dir = resolve_run_dir(child.task_group, child.run_id)
            if child_dir is not None:
                children.append(JobRecord.load(child_dir, task_group=child.task_group))
        key = f"md:{workflow.workflow_id}"
        grouped[key] = children or [parent]
        rows.append(
            {
                "Campaign": workflow.name,
                "Campaign ID": workflow.workflow_id[:8],
                "Type": "MD simulation",
                "Targets": 1,
                "Engines": str(workflow.parameters.get("engine") or "OpenMM"),
                "Repeat range": str(workflow.parameters.get("replicas") or 1),
                "Jobs": len(children),
                "Status": workflow.status,
                "Created": workflow.created_at,
                "_campaign_id": key,
                "_superseded": False,
            }
        )
    rows.sort(key=lambda row: str(row["Created"]), reverse=True)
    return rows, grouped


def _detail_rows(
    jobs: list[JobRecord],
    jobs_by_id: dict[str, JobRecord],
    *,
    include_history: bool = False,
) -> list[dict[str, Any]]:
    canonical = (
        canonical_campaign_jobs(
            str(jobs[0].metadata.get("launch_campaign_id") or ""), jobs
        )
        if jobs
        else []
    )
    canonical_ids = {job.run_id for job in canonical}
    displayed_jobs = jobs if include_history else canonical
    rows = []
    for job in sorted(displayed_jobs, key=lambda item: item.created_at, reverse=True):
        target_job = _target_job(job, jobs_by_id)
        target_context = target_lineage_summary(target_job or job, jobs_by_id)
        error = str(
            job.metadata.get("error")
            or job.result.get("error")
            or job.result.get("message")
            or ""
        ).strip()
        if error:
            lines = [line.strip() for line in error.splitlines() if line.strip()]
            error = (lines[-1] if lines else error)[:240]
        rows.append(
            {
                **(
                    {
                        "Record": (
                            "Current"
                            if job.run_id in canonical_ids
                            else f"Superseded retry ({job.status})"
                        )
                    }
                    if include_history
                    else {}
                ),
                "Result job": _job_result_url(job),
                "Engine": job.tool or job.workflow,
                "Target PDB": target_context["target"],
                "Target job": _job_result_url(target_job) if target_job else "",
                "Target origin": target_context["origin"],
                "Status": job.status,
                "Failure reason": error if job.status == "failed" else "",
                "Configured repeats": configured_repetitions(job) if job.workflow in SUPPORTED_WORKFLOWS else None,
                "Completed repeats": completed_repetitions(job) if job.workflow in SUPPORTED_WORKFLOWS else None,
                "Run ID": job.run_id,
                "Created": job.created_at,
            }
        )
    return rows


st.title("Campaign Results")
st.caption(
    "Review complete launch campaigns and extend repeatable child jobs in place. "
    "Existing results and child run IDs are preserved; only missing repetitions are queued."
)

records = iter_job_records(runs_root())
jobs_by_id = {job.run_id: job for job in records}
overview, grouped = _overview_rows(records)
if not overview:
    st.info("No launch campaigns have been recorded yet.")
    st.stop()

show_superseded = st.checkbox(
    "Show superseded or stopped campaign attempts",
    value=False,
    help="Old restart generations are retained for provenance but hidden by default.",
)
display_overview = [
    row for row in overview if show_superseded or not row.get("_superseded", False)
]
visible = pd.DataFrame(display_overview).drop(
    columns=["_campaign_id", "_superseded"]
)
st.dataframe(visible, use_container_width=True, hide_index=True)

options = [row["_campaign_id"] for row in display_overview]
requested = str(st.query_params.get("campaign_id") or "")
default_index = options.index(requested) if requested in options else 0
campaign_selector_labels: dict[str, str] = {}
for row in display_overview:
    campaign_id = str(row["_campaign_id"])
    campaign_jobs_for_label = grouped[campaign_id]
    target_labels = (
        _campaign_target_ligand_labels(
            campaign_jobs_for_label, jobs_by_id
        )[0]
        if _campaign_purpose(campaign_jobs_for_label)
        == "target_ligand_redocking_refolding"
        else ()
    )
    target_text = ", ".join(target_labels) or f"{row['Targets']} target(s)"
    campaign_selector_labels[campaign_id] = (
        f"{row['Campaign']} · {row['Campaign ID']} · targets {target_text} · "
        f"repeats {row['Repeat range']}"
    )
selected_id = st.selectbox(
    "Open campaign",
    options,
    index=default_index,
    format_func=lambda value: campaign_selector_labels.get(value, value),
)
st.query_params["campaign_id"] = selected_id
selected_jobs = grouped[selected_id]
selected_row = next(row for row in overview if row["_campaign_id"] == selected_id)

selected_purpose = _campaign_purpose(selected_jobs)
results_view = st.segmented_control(
    "Campaign Results view",
    ("Campaign details", "Analysis Sets"),
    default="Campaign details",
    key="campaign_results_view",
    label_visibility="collapsed",
)

if (
    results_view == "Analysis Sets"
    and selected_purpose == "target_ligand_redocking_refolding"
):
    st.markdown("## Build an Analysis Set")
    anchor_signatures = _campaign_ligand_signatures(selected_jobs)
    anchor_targets, anchor_ligands = _campaign_target_ligand_labels(
        selected_jobs, jobs_by_id
    )
    compatible_ids = []
    for row in display_overview:
        campaign_id = str(row["_campaign_id"])
        candidate_jobs = grouped[campaign_id]
        if (
            not campaign_id.startswith("md:")
            and _campaign_purpose(candidate_jobs) == selected_purpose
            and anchor_signatures
            and not _campaign_ligand_signatures(candidate_jobs).isdisjoint(
                anchor_signatures
            )
        ):
            compatible_ids.append(campaign_id)
    if selected_id not in compatible_ids:
        compatible_ids = [selected_id]
    compatibility_rows: list[dict[str, object]] = []
    compatibility_rows_by_id: dict[str, dict[str, object]] = {}
    compatible_labels: dict[str, str] = {}
    for row in display_overview:
        campaign_id = str(row["_campaign_id"])
        if campaign_id not in compatible_ids:
            continue
        candidate_targets, candidate_ligands = (
            _campaign_target_ligand_labels(
                grouped[campaign_id], jobs_by_id
            )
        )
        shared_targets = sorted(set(anchor_targets) & set(candidate_targets))
        shared_ligands = sorted(set(anchor_ligands) & set(candidate_ligands))
        target_text = ", ".join(candidate_targets) or "Not recorded"
        ligand_text = ", ".join(candidate_ligands) or "Structure identified"
        compatible_labels[campaign_id] = (
            f"{row['Campaign']} · targets {target_text} · ligands {ligand_text}"
        )
        compatibility_row = {
            "Campaign": row["Campaign"],
            "Campaign ID": row["Campaign ID"],
            "Targets": target_text,
            "Reference ligands": ligand_text,
            "Shared target(s)": ", ".join(shared_targets) or "—",
            "Why compatible": (
                "Shared canonical ligand: "
                + (", ".join(shared_ligands) or "structure match")
            ),
            "Engines": row["Engines"],
        }
        compatibility_rows.append(compatibility_row)
        compatibility_rows_by_id[campaign_id] = compatibility_row
    with st.container(border=True):
        st.caption(
            "An Analysis Set is a saved comparison definition, not a new "
            "campaign. Only target-ligand redocking/refolding campaigns with "
            "at least one canonical reference ligand are offered. Targets and "
            "their physical campaign boundaries remain explicit; MD is kept "
            "separate. A shared target is not required: campaigns without one "
            "are added as separate target groups rather than compared as the "
            "same target."
        )
        st.markdown("#### Available compatible campaigns")
        st.dataframe(
            pd.DataFrame(compatibility_rows),
            hide_index=True,
            width="stretch",
        )
        selection_key = f"analysis_set_campaigns_v2_{selected_id}"
        choose_all, clear_all = st.columns(2)
        if choose_all.button(
            "Select all compatible",
            key=f"analysis_set_select_all_{selected_id}",
        ):
            st.session_state[selection_key] = list(compatible_ids)
        if clear_all.button(
            "Clear selection",
            key=f"analysis_set_clear_all_{selected_id}",
        ):
            st.session_state[selection_key] = []
        analysis_campaigns = st.multiselect(
            "Physical campaigns",
            compatible_ids,
            default=[],
            format_func=lambda value: compatible_labels.get(value, value),
            key=selection_key,
        )
        st.markdown(
            f"#### Analysis Set preview · {len(analysis_campaigns)} "
            "physical campaign(s)"
        )
        st.caption(
            "Only the campaigns in this preview will be saved in the Analysis "
            "Set. Review the campaign IDs and targets before saving."
        )
        selected_preview = [
            compatibility_rows_by_id[campaign_id]
            for campaign_id in analysis_campaigns
            if campaign_id in compatibility_rows_by_id
        ]
        if selected_preview:
            st.dataframe(
                pd.DataFrame(selected_preview),
                hide_index=True,
                width="stretch",
            )
        else:
            st.warning("Select at least one physical campaign.")
        form = st.form(f"campaign_results_analysis_set_{selected_id}")
        analysis_name = form.text_input(
            "Analysis Set name",
            placeholder="For example: T3 preparation comparison – July 2026",
        )
        analysis_notes = form.text_area(
            "Notes (optional)",
            placeholder="Scientific question, exclusions, or review status.",
        )
        save_set = form.form_submit_button(
            "Save Analysis Set",
            type="primary",
            disabled=not analysis_campaigns,
        )
        if save_set:
            if not analysis_name.strip():
                st.error("Enter an Analysis Set name.")
            else:
                saved_set = save_analysis_set(
                    runs_root(),
                    name=analysis_name,
                    description=analysis_notes,
                    selection=_analysis_set_selection(
                        list(analysis_campaigns),
                        grouped,
                        jobs_by_id,
                    ),
                )
                st.success(
                    f"Saved {saved_set['name']} · {saved_set['job_code']}."
                )
                st.link_button(
                    "Open Analysis Set",
                    "./compound-campaign-comparison?"
                    + urlencode(
                        {
                            "analysis_set_id": saved_set["collection_id"],
                            "campaign_purpose": selected_purpose,
                        }
                    ),
                    type="primary",
                )
        if len(compatible_ids) == 1:
            st.info(
                "No additional compatible historical campaign was found. "
                "Campaigns without a recoverable ligand identity are not "
                "assumed to be compatible."
            )

if results_view == "Analysis Sets":
    if selected_purpose != "target_ligand_redocking_refolding":
        st.info(
            "Analysis Sets for this campaign family are not available yet. "
            "MD campaigns remain separate from docking/refolding analyses."
        )
    st.markdown("## Saved Analysis Sets")
    saved_sets = [
        row
        for row in list_analysis_sets(runs_root())
        if not selected_purpose
        or str(row.get("selection", {}).get("campaign_purpose") or "")
        == selected_purpose
    ]
    if saved_sets:
        st.caption(
            "Open results at any time from this table. Saved Analysis Sets do "
            "not alter their physical campaigns."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Analysis Set": row["name"],
                    "Set ID": row["job_code"],
                    "Campaigns": ", ".join(
                        value[:8]
                        for value in row["launch_campaign_ids"]
                    ),
                    "Targets": len(row["target_run_ids"]),
                    "Engines": ", ".join(row["engines"]),
                    "Created": row["created_at"],
                    "Open results": row["open"],
                }
                for row in saved_sets
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Created": st.column_config.DatetimeColumn(
                    format="YYYY-MM-DD HH:mm"
                ),
                "Open results": st.column_config.LinkColumn(
                    "Open results",
                    display_text="Open scientific results",
                ),
            },
        )
    else:
        st.info("No Analysis Sets have been saved for this campaign family.")
    st.stop()

st.subheader(selected_row["Campaign"])
left, middle, right = st.columns(3)
left.metric("Campaign type", selected_row["Type"])
middle.metric("Targets", selected_row["Targets"])
right.metric("Canonical child jobs", selected_row["Jobs"])
st.caption(f"Campaign ID: {selected_id.removeprefix('md:')}")
campaign_repeat_target = 3
repeat_balance_rows: list[dict[str, object]] = []
if not selected_id.startswith("md:"):
    canonical_repeatable = canonical_campaign_jobs(selected_id, selected_jobs)
    if canonical_repeatable:
        campaign_repeat_target = max(
            configured_repetitions(job) for job in canonical_repeatable
        )
        for job in canonical_repeatable:
            completed = completed_repetitions(job)
            configured = configured_repetitions(job)
            if completed >= campaign_repeat_target:
                continue
            target_job = _target_job(job, jobs_by_id)
            target_context = target_lineage_summary(
                target_job or job,
                jobs_by_id,
            )
            repeat_balance_rows.append(
                {
                    "Engine": job.tool or job.workflow,
                    "Target": target_context["target"],
                    "Job": display_job_code(
                        job.metadata.get("job_code"), job.run_id
                    ),
                    "Execution": f"{completed}/{configured} completed",
                    "Campaign comparison target": campaign_repeat_target,
                    "Additional repetitions needed": (
                        campaign_repeat_target - completed
                    ),
                }
            )
if selected_id.startswith("md:"):
    st.dataframe(
        pd.DataFrame(
            {
                "Step": job.metadata.get("workflow_step_id") or job.metadata.get("workflow") or job.task_group,
                "Job": job.metadata.get("job_code") or job.run_id[:8],
                "Status": job.status,
                "Replica": job.metadata.get("repeat_index"),
                "Run ID": job.run_id,
            }
            for job in selected_jobs
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    if repeat_balance_rows:
        st.warning(
            "Execution is complete for every configured child job, but the "
            f"campaign is not repeat-balanced: {len(repeat_balance_rows)} "
            f"job(s) are below the comparison target of "
            f"{campaign_repeat_target}. This is not an execution failure."
        )
        with st.expander(
            "Jobs below the campaign comparison target",
            expanded=True,
        ):
            st.dataframe(
                pd.DataFrame(repeat_balance_rows),
                hide_index=True,
                width="stretch",
            )
    if any(
        job.task_group in {"docking", "refolding", "rescoring"}
        for job in selected_jobs
    ):
        st.link_button(
            "Open aggregate scientific results",
            _aggregate_result_url(selected_id, selected_jobs),
            help=(
                "Compare this launch by prepared target, ligand, engine, repetition, "
                "pose validation, interactions, and structure."
            ),
            type="primary",
        )
    show_retry_history = st.checkbox(
        "Show superseded retry history",
        value=False,
        key=f"show_campaign_retry_history_{selected_id}",
        help=(
            "Reveal old failed, cancelled, or replaced attempts. By default only the "
            "current canonical job for each target and engine is shown."
        ),
    )
    st.dataframe(
        pd.DataFrame(
            _detail_rows(
                selected_jobs,
                jobs_by_id,
                include_history=show_retry_history,
            )
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Result job": st.column_config.LinkColumn(
                "Docking / refolding job",
                display_text=r"label=([^&]+)",
                help="Open this prediction result page.",
            ),
            "Target job": st.column_config.LinkColumn(
                "Prepared-target job",
                display_text=r"label=([^&]+)",
                help="Open the result page for the exact prepared target used by this job.",
            ),
            "Target origin": st.column_config.TextColumn(width="large"),
            "Failure reason": st.column_config.TextColumn(width="large"),
            "Created": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )

st.subheader("Bring repeatable jobs to the same count")
st.caption(
    "For example, choosing 3 leaves jobs with three repeats unchanged, appends two to jobs "
    "with one, and appends one to jobs with two. Failed duplicate restart history is shown above "
    "but is not extended."
)
target = st.number_input(
    "Target repetitions",
    min_value=1,
    max_value=100,
    value=int(campaign_repeat_target),
    step=1,
)
confirmed = st.checkbox(
    "I understand that this queues additional compute for every canonical repeatable child job.",
    key=f"confirm_campaign_extension_{selected_id}",
)
if st.button(
    "Bring campaign to target repetitions",
    type="primary",
    disabled=not confirmed,
    key=f"extend_campaign_{selected_id}",
):
    try:
        if selected_id.startswith("md:"):
            workflow = extend_md_simulation(selected_id.removeprefix("md:"), int(target))
            results = []
            st.success(
                f"MD campaign now targets {workflow.parameters.get('replicas')} replicas; "
                "only missing production replicas were added."
            )
        else:
            results = extend_campaign_repetitions(selected_id, int(target))
    except Exception as exc:
        st.error(f"Campaign extension failed: {exc}")
    else:
        if results:
            summary = Counter(result.status for result in results)
            st.success(
                "Campaign extension recorded: "
                + ", ".join(f"{status} {count}" for status, count in sorted(summary.items()))
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "Job": result.run_id,
                        "Workflow": result.workflow,
                        "Before": result.previous_repetitions,
                        "Target": result.target_repetitions,
                        "Action": result.status,
                        "Detail": result.detail,
                    }
                    for result in results
                ),
                use_container_width=True,
                hide_index=True,
            )

if selected_id.startswith("md:"):
    st.subheader("Extend simulation duration")
    selected_workflow = WorkflowRecord.load(selected_id.removeprefix("md:"))
    production = dict(selected_workflow.parameters.get("production") or {})
    timestep_fs = float(production.get("production_timestep_fs") or 4.0)
    current_steps = int(production.get("production_steps") or 0)
    current_ns = current_steps * timestep_fs / 1_000_000.0
    engine = str(selected_workflow.parameters.get("engine") or "openmm").lower()
    if engine == "gromacs":
        st.caption(
            "Strict native GROMACS continuation: the final CPT, TPR, XTC, EDR and native log "
            "must all be present. The TPR is extended and mdrun resumes with -cpi and -append. "
            "Checksum or artifact mismatch is fatal; coordinate-only fallback is disabled."
        )
    else:
        st.caption(
            "Strict OpenMM checkpoint continuation: every replica must have an exact endpoint "
            "checkpoint plus the identical serialized System and Integrator. Positions, velocities, "
            "periodic box, time and stochastic state are preserved; coordinate-only fallback is disabled."
        )
    st.caption(
        "Existing trajectories and native logs are appended, and downstream endpoint-energy "
        "and aggregate-analysis children are superseded and recomputed."
    )
    st.metric("Current cumulative duration", f"{current_ns:g} ns")
    target_duration = st.number_input(
        "New cumulative duration (ns)",
        min_value=float(current_ns + max(timestep_fs / 1_000_000.0, 0.001)),
        value=float(max(current_ns * 2.0, current_ns + 1.0)),
        step=1.0,
        key=f"duration_target_{selected_id}",
    )
    duration_confirmed = st.checkbox(
        "Queue a strict continuation for every completed replica and recompute downstream energy/analysis results.",
        key=f"confirm_duration_extension_{selected_id}",
    )
    if st.button(
        "Extend MD duration",
        type="primary",
        disabled=not duration_confirmed,
        key=f"extend_duration_{selected_id}",
    ):
        try:
            extended = extend_md_simulation_duration(
                selected_id.removeprefix("md:"),
                float(target_duration),
            )
        except Exception as exc:
            st.error(f"Duration extension was refused: {exc}")
        else:
            st.success(
                f"Strict continuation queued to {target_duration:g} ns under the same MD campaign "
                f"({extended.workflow_id})."
            )
