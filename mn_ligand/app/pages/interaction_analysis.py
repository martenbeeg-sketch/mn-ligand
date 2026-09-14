from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records
from mn_ligand.core.residue_mapping import sequence_author_residue_mapping
from mn_ligand.runtime import cpu_process_limit, runs_root
from mn_ligand.workflows.interaction_analysis import (
    INTERACTION_ANALYSIS_TASK_GROUP,
    INTERACTION_ENGINES,
    INTERACTION_RESIDUE_NUMBERING_POLICY_VERSION,
    TARGET_COMPLEX_INVENTORY_SCHEMA_VERSION,
    TARGET_COMPLEX_SELECTION_POLICY,
    compatible_target_complex_jobs,
    queue_interaction_analysis_job,
    target_complex_candidates,
)
from mn_ligand.workflows.pose_validation import (
    POSE_VALIDATION_INVENTORY_SCHEMA_VERSION,
    POSE_VALIDATION_SELECTION_POLICY,
    cached_pose_validation_candidates,
    compatible_source_jobs,
    pose_validation_inventory_summary,
)


def _source_key(job: JobRecord) -> str:
    return f"{job.task_group}::{job.run_id}"


def _source_label(job: JobRecord) -> str:
    code = display_job_code(job.metadata.get("job_code"), job.run_id)
    return f"{code} · {job.tool or job.workflow}"


def _target_context(
    source: JobRecord,
    jobs_by_id: dict[str, JobRecord],
) -> tuple[str, str]:
    if (
        source.artifact_manifest is not None
        and source.artifact_manifest.by_type("prepared_complex")
        and source.workflow
        not in {
            "docking_campaign",
            "openvs_docking",
            "alphafold3_refolding",
            "boltz2_refolding",
        }
    ):
        code = display_job_code(source.metadata.get("job_code"), source.run_id)
        pdb_id = str(source.metadata.get("pdb_id") or "").strip()
        artifact_name = Path(
            source.artifact_manifest.by_type("prepared_complex")[0].path
        ).name
        return source.run_id, " · ".join(
            value for value in (pdb_id, artifact_name, code) if value
        )
    target_id = str(
        source.metadata.get("prepared_target_run_id")
        or source.parent_run_id
        or ""
    )
    target = jobs_by_id.get(target_id)
    if target is None:
        return target_id, target_id[:8] if target_id else "Unknown target"
    code = display_job_code(target.metadata.get("job_code"), target.run_id)
    pdb_id = str(
        target.metadata.get("pdb_id")
        or (target.metadata.get("receptor") or {}).get("pdb_id")
        or ""
    ).strip()
    artifact = str(
        target.result.get("prepared_target")
        or target.result.get("prepared_receptor")
        or target.result.get("prepared_complex")
        or ""
    )
    artifact_name = Path(artifact).name
    parts = [value for value in (pdb_id, artifact_name, code) if value]
    return target_id, " · ".join(parts) or code


def _source_context(
    source: JobRecord,
    jobs_by_id: dict[str, JobRecord],
) -> dict[str, str]:
    target_id, target = _target_context(source, jobs_by_id)
    campaign = str(
        source.metadata.get("launch_campaign_label")
        or source.metadata.get("launch_campaign_id")
        or "Standalone run"
    )
    return {
        "source_job": display_job_code(
            source.metadata.get("job_code"), source.run_id
        ),
        "prediction_engine": (
            "Prepared target complex"
            if source.artifact_manifest is not None
            and source.artifact_manifest.by_type("prepared_complex")
            and source.workflow
            not in {
                "docking_campaign",
                "openvs_docking",
                "alphafold3_refolding",
                "boltz2_refolding",
            }
            else str(source.tool or source.workflow or "")
        ),
        "campaign": (
            "Prepared target reference"
            if source.artifact_manifest is not None
            and source.artifact_manifest.by_type("prepared_complex")
            and source.workflow
            not in {
                "docking_campaign",
                "openvs_docking",
                "alphafold3_refolding",
                "boltz2_refolding",
            }
            else campaign
        ),
        "target_id": target_id,
        "target": target,
    }


def _count(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _job_records_by_id(
    run_ids: set[str],
    *,
    existing: dict[str, JobRecord] | None = None,
) -> dict[str, JobRecord]:
    """Resolve a small run-id set without constructing the global job index."""
    records = dict(existing or {})
    pending = {str(run_id) for run_id in run_ids if str(run_id)} - set(records)
    if not pending:
        return records
    root = runs_root()
    for group_dir in root.iterdir():
        if not group_dir.is_dir() or group_dir.name.startswith("."):
            continue
        for run_dir in group_dir.iterdir():
            if run_dir.name not in pending or not run_dir.is_dir():
                continue
            records[run_dir.name] = JobRecord.load(
                run_dir,
                task_group=group_dir.name,
                validate_artifacts=False,
            )
            pending.remove(run_dir.name)
            if not pending:
                return records
    return records


def _current_children(
    jobs: list[JobRecord] | None = None,
) -> dict[tuple[str, str], list[JobRecord]]:
    grouped: dict[tuple[str, str], list[JobRecord]] = {}
    source = jobs if jobs is not None else iter_job_records(
        runs_root(),
        task_groups=(INTERACTION_ANALYSIS_TASK_GROUP,),
        load_artifacts=False,
        validate_artifacts=False,
    )
    for job in source:
        policy = str(job.metadata.get("selection_policy") or "")
        schema_version = int(
            job.metadata.get("selection_schema_version") or 0
        )
        current_inventory = (
            (
                policy == POSE_VALIDATION_SELECTION_POLICY
                and schema_version
                == POSE_VALIDATION_INVENTORY_SCHEMA_VERSION
            )
            or (
                policy == TARGET_COMPLEX_SELECTION_POLICY
                and schema_version == TARGET_COMPLEX_INVENTORY_SCHEMA_VERSION
            )
        )
        if (
            not current_inventory
            or int(job.metadata.get("residue_numbering_policy_version") or 0)
            != INTERACTION_RESIDUE_NUMBERING_POLICY_VERSION
        ):
            continue
        grouped.setdefault(
            (
                str(job.parent_run_id or ""),
                str(job.metadata.get("interaction_engine") or job.tool),
            ),
            [],
        ).append(job)
    return grouped


def _covered(
    grouped: dict[tuple[str, str], list[JobRecord]],
    source: JobRecord,
    engine: str,
    selection_ids: set[str] | None = None,
) -> bool:
    active = [
        job
        for job in grouped.get((source.run_id, engine), [])
        if job.status in {"queued", "running", "completed"}
    ]
    if selection_ids is None:
        return bool(active)
    covered_ids = {
        str(value)
        for job in active
        for value in (job.metadata.get("selection_ids") or ())
    }
    return bool(selection_ids) and selection_ids.issubset(covered_ids)


def _result_rows(
    interaction_jobs: list[JobRecord] | None = None,
    jobs_by_id: dict[str, JobRecord] | None = None,
) -> list[dict[str, object]]:
    jobs = interaction_jobs if interaction_jobs is not None else list(
        iter_job_records(
            runs_root(),
            task_groups=(INTERACTION_ANALYSIS_TASK_GROUP,),
            load_artifacts=False,
            validate_artifacts=False,
        )
    )
    source_ids = {str(job.parent_run_id or "") for job in jobs}
    index = _job_records_by_id(source_ids, existing=jobs_by_id)
    target_ids = {
        str(
            source.metadata.get("prepared_target_run_id")
            or source.parent_run_id
            or ""
        )
        for source_id in source_ids
        if (source := index.get(source_id)) is not None
    }
    index = _job_records_by_id(target_ids, existing=index)
    rows: list[dict[str, object]] = []
    for job in jobs:
        source = index.get(str(job.parent_run_id or ""))
        context = (
            _source_context(source, index)
            if source is not None
            else {
                "source_job": str(job.parent_run_id or "")[:8],
                "prediction_engine": str(
                    job.metadata.get("source_engine") or ""
                ),
                "campaign": "Unknown source",
                "target_id": "",
                "target": "Unknown target",
            }
        )
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        query = urlencode(
            {"task_group": job.task_group, "run_id": job.run_id, "label": code}
        )
        current_policy = (
            int(job.metadata.get("residue_numbering_policy_version") or 0)
            == INTERACTION_RESIDUE_NUMBERING_POLICY_VERSION
        )
        rows.append({
            "result": f"./job-results?{query}",
            "analysis_run_id": job.run_id,
            "analysis_job": code,
            "analysis_engine": str(
                job.metadata.get("interaction_engine") or job.tool or ""
            ),
            **context,
            "poses": _count(job.metadata.get("pose_count")),
            "compounds": _count(job.metadata.get("compound_count")),
            "interactions": _count(job.result.get("interaction_count")),
            "status": (
                job.status if current_policy else "Legacy numbering"
            ),
            "created": job.created_at,
        })
    return sorted(rows, key=lambda row: str(row["created"]), reverse=True)


def _latest_result_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    current = frame.loc[frame["status"].ne("Legacy numbering")].copy()
    if current.empty:
        return current
    current["_created"] = pd.to_datetime(
        current["created"], errors="coerce", utc=True
    )
    return (
        current.sort_values("_created")
        .drop_duplicates(
            ["source_job", "analysis_engine"], keep="last"
        )
        .drop(columns="_created")
        .sort_values("created", ascending=False)
        .reset_index(drop=True)
    )


def _review_atom_scope(atom_name: object) -> str:
    normalized = str(atom_name or "").strip().upper()
    if not normalized or normalized.startswith("#"):
        return ""
    return "BB" if normalized in {"N", "CA", "C", "O", "OXT"} else "SC"


def _review_interaction_atoms(interactions: pd.DataFrame) -> pd.DataFrame:
    table = interactions.copy()
    for column, default in {
        "protein_atom_name": "",
        "protein_atom_scope": "",
        "ligand_atom_name": "",
        "native_fields_json": "{}",
    }.items():
        if column not in table:
            table[column] = default
    for index, row in table.iterrows():
        try:
            payload = json.loads(str(row.get("native_fields_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        interaction_type = str(
            row.get("interaction_type") or ""
        ).strip().lower().replace("_", " ")
        protein_atom = str(row.get("protein_atom_name") or "").strip()
        ligand_atom = str(row.get("ligand_atom_name") or "").strip()
        if not protein_atom:
            protein_atom = next(
                (
                    str(payload.get(key) or "").strip()
                    for key in (
                        "protein_atom",
                        "protatom",
                        "donor_atom",
                        "acceptor_atom",
                    )
                    if str(payload.get(key) or "").strip()
                ),
                "",
            )
        if not ligand_atom:
            ligand_atom = next(
                (
                    str(payload.get(key) or "").strip()
                    for key in (
                        "ligand_atom",
                        "ligatom",
                        "ligatom_orig_idx",
                        "lig_idx",
                    )
                    if str(payload.get(key) or "").strip()
                ),
                "",
            )
        protein_serial = ""
        ligand_serial = ""
        if "hydrophob" in interaction_type:
            protein_serial = str(payload.get("protcarbonidx") or "")
            ligand_serial = str(payload.get("ligcarbonidx") or "")
        elif "halogen" in interaction_type:
            protein_serial = str(payload.get("acc_idx") or "")
            ligand_serial = str(payload.get("don_idx") or "")
        elif "hydrogen" in interaction_type or "hbond" in interaction_type:
            protein_is_donor = str(
                payload.get("protisdon") or ""
            ).strip().lower() in {"true", "1", "yes"}
            protein_serial = str(payload.get(
                "donoridx" if protein_is_donor else "acceptoridx"
            ) or "")
            ligand_serial = str(payload.get(
                "acceptoridx" if protein_is_donor else "donoridx"
            ) or "")
        if not protein_serial:
            protein_serial = next(
                iter(re.findall(
                    r"\d+",
                    str(payload.get("prot_idx_list") or ""),
                )),
                "",
            )
        if not ligand_serial:
            ligand_serial = next(
                iter(re.findall(
                    r"\d+",
                    str(payload.get("lig_idx_list") or ""),
                )),
                "",
            )
        if not protein_atom and protein_serial:
            protein_atom = f"#{protein_serial}"
        if not ligand_atom and ligand_serial:
            ligand_atom = f"#{ligand_serial}"
        scope = str(row.get("protein_atom_scope") or "").strip().upper()
        if scope not in {"BB", "SC"}:
            native_sidechain = str(payload.get("sidechain") or "").lower()
            if native_sidechain in {"true", "1", "yes"}:
                scope = "SC"
            elif native_sidechain in {"false", "0", "no"}:
                scope = "BB"
            else:
                scope = _review_atom_scope(protein_atom)
        table.at[index, "protein_atom_name"] = protein_atom
        table.at[index, "ligand_atom_name"] = ligand_atom
        table.at[index, "protein_atom_scope"] = scope
    return table


def _review_author_numbering(
    job: JobRecord,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    table = interactions.copy()
    interaction_engine = str(
        job.metadata.get("interaction_engine") or job.tool or ""
    )
    if table.empty or interaction_engine != "Native MD geometry":
        return table
    reference = job.run_dir / "input" / "reference_target.pdb"
    if not reference.is_file():
        return table
    reference_text = reference.read_text(errors="replace")
    for pose_id in table["pose_id"].astype(str).unique():
        complex_path = job.run_dir / "prepared" / f"{pose_id}.complex.pdb"
        if not complex_path.is_file():
            continue
        try:
            mapping = sequence_author_residue_mapping(
                complex_path.read_text(errors="replace"),
                reference_text,
            )
        except (OSError, ValueError):
            continue
        pose_mask = table["pose_id"].astype(str).eq(pose_id)
        for index, row in table.loc[pose_mask].iterrows():
            try:
                residue_number = int(float(row["protein_residue_number"]))
            except (TypeError, ValueError):
                continue
            author = mapping.get((
                str(row.get("protein_chain") or "_").strip() or "_",
                residue_number,
                str(row.get("protein_insertion_code") or "").strip(),
            ))
            if author is None:
                continue
            table.at[index, "protein_chain"] = author["chain"]
            table.at[index, "protein_residue_number"] = author[
                "residue_number"
            ]
            table.at[index, "protein_insertion_code"] = author[
                "insertion_code"
            ]
            table.at[index, "protein_residue_name"] = author["residue_name"]
    return table


def _gnina_pose_selection(
    source_engine: object,
    selection_criterion: object,
) -> str:
    if str(source_engine or "").strip().lower() != "gnina":
        return ""
    criterion = str(selection_criterion or "").strip().lower()
    selected_by_cnn = "cnn" in criterion or "affinity" in criterion
    selected_by_vina = "vina" in criterion or "empirical" in criterion
    if selected_by_cnn and selected_by_vina:
        return "CNN + Vina"
    if selected_by_cnn:
        return "CNN"
    if selected_by_vina:
        return "Vina"
    return "Unspecified"


def _review_table_for_job(
    job: JobRecord,
    context: dict[str, object],
) -> pd.DataFrame:
    interactions_path = job.run_dir / "interactions.csv"
    if (
        not interactions_path.is_file()
        or not interactions_path.stat().st_size
    ):
        return pd.DataFrame()
    interactions = pd.read_csv(interactions_path).fillna("")
    if interactions.empty:
        return pd.DataFrame()
    interactions = _review_author_numbering(job, interactions)
    interactions = _review_interaction_atoms(interactions)
    for column, default in {
        "pose_id": "",
        "compound_id": "",
        "source_engine": "",
        "replicate": "",
        "protein_chain": "",
        "protein_residue_name": "",
        "protein_residue_number": "",
        "protein_insertion_code": "",
        "interaction_type": "",
        "distance_angstrom": "",
        "angle_degree": "",
        "prediction": "",
        "selection_criterion": "",
    }.items():
        if column not in interactions:
            interactions[column] = default
    input_complexes: dict[str, Path] = {}
    interaction_inputs = job.run_dir / "input" / "interaction_inputs.csv"
    if interaction_inputs.is_file():
        try:
            input_rows = pd.read_csv(interaction_inputs).fillna("")
            for _, input_row in input_rows.iterrows():
                pose_id = str(input_row.get("pose_id") or "").strip()
                complex_file = str(
                    input_row.get("complex_file") or ""
                ).strip()
                if pose_id and complex_file:
                    input_complexes[pose_id] = job.run_dir / complex_file
        except (OSError, ValueError):
            pass

    predicted_complex_paths: list[str] = []
    for pose_id_value in interactions["pose_id"]:
        pose_id = str(pose_id_value or "").strip()
        prepared_complex = (
            job.run_dir / "prepared" / f"{pose_id}.complex.pdb"
        )
        input_complex = input_complexes.get(pose_id)
        selected_path = (
            prepared_complex
            if pose_id and prepared_complex.is_file()
            else input_complex
            if input_complex is not None and input_complex.is_file()
            else None
        )
        predicted_complex_paths.append(
            str(selected_path.resolve()) if selected_path is not None else ""
        )
    protein_residue = (
        interactions["protein_chain"].astype(str)
        + ":"
        + interactions["protein_residue_name"].astype(str)
        + interactions["protein_residue_number"].astype(str)
        + interactions["protein_insertion_code"].astype(str)
    )
    pose = (
        interactions["compound_id"].astype(str)
        + " · "
        + interactions["source_engine"].astype(str)
        + " · attempt "
        + interactions["replicate"].astype(str)
    )
    review = pd.DataFrame({
        "Campaign": str(context.get("campaign") or ""),
        "Source job": str(context.get("source_job") or ""),
        "Prediction engine": str(context.get("prediction_engine") or ""),
        "Target": str(context.get("target") or ""),
        "Interaction detection engine": str(
            context.get("analysis_engine") or ""
        ),
        "Analysis job": str(context.get("analysis_job") or ""),
        "Pose": pose,
        "Compound": interactions["compound_id"].astype(str),
        "Replicate": interactions["replicate"],
        "Prediction": interactions["prediction"].astype(str),
        "Predicted complex path": predicted_complex_paths,
        "GNINA pose selection": [
            _gnina_pose_selection(source_engine, criterion)
            for source_engine, criterion in zip(
                interactions["source_engine"],
                interactions["selection_criterion"],
                strict=True,
            )
        ],
        "Interaction": interactions["interaction_type"].astype(str),
        "Protein residue": protein_residue,
        "Ligand atom": interactions["ligand_atom_name"].astype(str),
        "Protein atom": interactions["protein_atom_name"].astype(str),
        "Protein region": interactions["protein_atom_scope"].astype(str),
        "Distance (Å)": pd.to_numeric(
            interactions["distance_angstrom"], errors="coerce"
        ),
        "Angle (°)": pd.to_numeric(
            interactions["angle_degree"], errors="coerce"
        ),
    })
    return review


def _safe_excel_sheet_name(
    value: object,
    used_names: set[str],
) -> str:
    base = re.sub(r"[\[\]:*?/\\]+", "-", str(value or "Interactions"))
    base = base.strip()[:31] or "Interactions"
    candidate = base
    suffix = 2
    while candidate.lower() in used_names:
        ending = f"-{suffix}"
        candidate = f"{base[:31 - len(ending)]}{ending}"
        suffix += 1
    used_names.add(candidate.lower())
    return candidate


def _interaction_review_workbook(
    selection: pd.DataFrame,
    merged: pd.DataFrame,
    per_tool: dict[str, pd.DataFrame],
) -> bytes:
    output = BytesIO()
    used_names: set[str] = set()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        selection.to_excel(
            writer,
            sheet_name=_safe_excel_sheet_name("Selection", used_names),
            index=False,
        )
        merged.to_excel(
            writer,
            sheet_name=_safe_excel_sheet_name(
                "All interactions", used_names
            ),
            index=False,
        )
        for interaction_tool, table in per_tool.items():
            table.to_excel(
                writer,
                sheet_name=_safe_excel_sheet_name(
                    interaction_tool, used_names
                ),
                index=False,
            )
    return output.getvalue()


def _prepare_interaction_review_exports(
    visible: pd.DataFrame,
) -> tuple[dict[str, bytes], bytes, dict[str, int], list[str]]:
    jobs_by_id = {
        job.run_id: job for job in iter_job_records(runs_root())
    }
    grouped: dict[str, list[pd.DataFrame]] = {}
    failures: list[str] = []
    for context in visible.to_dict("records"):
        run_id = str(context.get("analysis_run_id") or "")
        job = jobs_by_id.get(run_id)
        if job is None:
            failures.append(
                f"{context.get('analysis_job') or run_id}: job not found"
            )
            continue
        try:
            table = _review_table_for_job(job, context)
        except (OSError, TypeError, ValueError) as exc:
            failures.append(
                f"{context.get('analysis_job') or run_id}: {exc}"
            )
            continue
        if table.empty:
            continue
        tool = str(context.get("analysis_engine") or "Unknown tool")
        grouped.setdefault(tool, []).append(table)
    per_tool = {
        tool: pd.concat(tables, ignore_index=True)
        for tool, tables in grouped.items()
        if tables
    }
    merged = (
        pd.concat(list(per_tool.values()), ignore_index=True)
        if per_tool
        else pd.DataFrame()
    )
    csv_files = (
        {
            "Combined": merged.to_csv(index=False).encode("utf-8"),
            **{
                tool: table.to_csv(index=False).encode("utf-8")
                for tool, table in per_tool.items()
            },
        }
        if per_tool
        else {}
    )
    selection_columns = [
        "campaign",
        "source_job",
        "prediction_engine",
        "target",
        "analysis_engine",
        "analysis_job",
        "compounds",
        "poses",
        "interactions",
        "status",
        "created",
    ]
    selection = visible[
        [column for column in selection_columns if column in visible]
    ].rename(columns={
        "campaign": "Campaign",
        "source_job": "Source job",
        "prediction_engine": "Prediction engine",
        "target": "Target",
        "analysis_engine": "Interaction tool",
        "analysis_job": "Analysis job",
        "compounds": "Compounds",
        "poses": "Poses",
        "interactions": "Interactions",
        "status": "Status",
        "created": "Created",
    })
    workbook = (
        _interaction_review_workbook(selection, merged, per_tool)
        if per_tool
        else b""
    )
    counts = {
        "Combined": len(merged),
        **{tool: len(table) for tool, table in per_tool.items()},
    }
    return csv_files, workbook, counts, failures


def _export_file_stem(value: object) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return stem.strip("-._").lower() or "interaction-tool"


def render() -> None:
    st.title("Interaction Analysis")
    st.caption(
        "Characterize protein–ligand contacts in prepared target complexes and "
        "focused docked/cofolded poses with app-native MD geometry, PLIP, "
        "PandaMap, or any combination."
    )
    input_tab, engine_tab, run_tab, results_tab = st.tabs(
        ["Target / Input", "Tool / Engine", "Run", "Results"],
        key="interaction_analysis_tab",
        on_change="rerun",
    )
    prediction_sources = compatible_source_jobs(load_artifacts=False)
    target_complex_sources = compatible_target_complex_jobs()
    sources = list(
        {
            (job.task_group, job.run_id): job
            for job in (*target_complex_sources, *prediction_sources)
        }.values()
    )
    target_complex_keys = {
        _source_key(job) for job in target_complex_sources
    }
    source_jobs = {_source_key(job): job for job in sources}
    interaction_jobs = list(
        iter_job_records(
            runs_root(),
            task_groups=(INTERACTION_ANALYSIS_TASK_GROUP,),
            load_artifacts=False,
            validate_artifacts=False,
        )
    )
    children = _current_children(interaction_jobs)
    all_jobs_by_id = {job.run_id: job for job in sources}
    all_jobs_by_id = _job_records_by_id(
        {
            str(source.metadata.get("prepared_target_run_id") or "")
            or str(source.parent_run_id or "")
            for source in sources
        }
        | {str(job.parent_run_id or "") for job in interaction_jobs},
        existing=all_jobs_by_id,
    )
    all_jobs_by_id = _job_records_by_id(
        {
            str(source.metadata.get("prepared_target_run_id") or "")
            or str(source.parent_run_id or "")
            for source in all_jobs_by_id.values()
        },
        existing=all_jobs_by_id,
    )

    def load_inventory_summary(item: tuple[str, JobRecord]):
        key, job = item
        try:
            if key in target_complex_keys:
                rows = target_complex_candidates(job)
                return key, {
                    "candidate_count": len(rows),
                    "selection_ids": frozenset(
                        str(row.get("selection_id") or "")
                        for row in rows
                        if str(row.get("selection_id") or "")
                    ),
                    "rows": rows,
                }
            summary = pose_validation_inventory_summary(job)
            return key, {
                "candidate_count": summary.get("candidate_count"),
                "selection_ids": None,
                "rows": None,
            }
        except (OSError, TypeError, ValueError):
            return key, {
                "candidate_count": None,
                "selection_ids": None,
                "rows": None,
            }

    inventory_summaries: dict[str, dict[str, object]] = {}
    if source_jobs:
        with ThreadPoolExecutor(
            max_workers=min(8, len(source_jobs)),
            thread_name_prefix="interaction-inventory-summary",
        ) as executor:
            for key, summary in executor.map(
                load_inventory_summary, source_jobs.items()
            ):
                inventory_summaries[key] = summary

    def candidate_count(key: str) -> int | None:
        value = inventory_summaries.get(key, {}).get("candidate_count")
        return int(value) if value is not None else None

    def has_candidates(key: str) -> bool:
        count = candidate_count(key)
        return count is None or count > 0

    def coverage_selection_ids(key: str) -> set[str] | None:
        value = inventory_summaries.get(key, {}).get("selection_ids")
        return set(value) if value is not None else None

    source_contexts = {
        key: _source_context(source, all_jobs_by_id)
        for key, source in source_jobs.items()
    }

    requested_run = str(st.query_params.get("source_run_id", "") or "")
    requested = next(
        (key for key, job in source_jobs.items() if job.run_id == requested_run),
        "",
    )
    engines = list(INTERACTION_ENGINES)
    if "interaction_selected_engines" not in st.session_state:
        st.session_state["interaction_selected_engines"] = engines
    selected_engines = [
        value for value in st.session_state["interaction_selected_engines"]
        if value in engines
    ]
    missing_sources = [
        key for key, source in source_jobs.items()
        if any(
            not _covered(
                children,
                source,
                engine,
                coverage_selection_ids(key),
            )
            for engine in selected_engines
        )
    ]
    selection_context = (
        tuple(source_jobs),
        tuple(selected_engines),
        tuple(missing_sources),
    )
    if (
        st.session_state.get("_interaction_source_context")
        != selection_context
    ):
        st.session_state["_interaction_source_context"] = selection_context
        st.session_state["interaction_selected_sources"] = list(missing_sources)
        st.session_state["_interaction_editor_revision"] = (
            int(st.session_state.get("_interaction_editor_revision", 0)) + 1
        )
    if (
        requested
        and st.session_state.get("_interaction_requested_source") != requested
    ):
        st.session_state["_interaction_requested_source"] = requested
        st.session_state["interaction_selected_sources"] = [requested]
        st.session_state["_interaction_editor_revision"] = (
            int(st.session_state.get("_interaction_editor_revision", 0)) + 1
        )

    if input_tab.open:
        if not source_jobs:
            st.info(
                "Complete a prepared target complex, docking result, or "
                "cofolding result first."
            )
        else:
            actions = st.columns(3)
            if actions[0].button("Select all missing", key="interaction_missing"):
                st.session_state["interaction_selected_sources"] = list(missing_sources)
                st.session_state["_interaction_editor_revision"] = (
                    int(
                        st.session_state.get(
                            "_interaction_editor_revision", 0
                        )
                    )
                    + 1
                )
            if actions[1].button("Select all eligible", key="interaction_all"):
                st.session_state["interaction_selected_sources"] = list(source_jobs)
                st.session_state["_interaction_editor_revision"] = (
                    int(
                        st.session_state.get(
                            "_interaction_editor_revision", 0
                        )
                    )
                    + 1
                )
            if actions[2].button("Clear selection", key="interaction_none"):
                st.session_state["interaction_selected_sources"] = []
                st.session_state["_interaction_editor_revision"] = (
                    int(
                        st.session_state.get(
                            "_interaction_editor_revision", 0
                        )
                    )
                    + 1
                )
            selected_keys = [
                key for key in st.session_state["interaction_selected_sources"]
                if key in source_jobs
            ]
            frame = pd.DataFrame([
                {
                    "_key": key,
                    "analyze": key in selected_keys,
                    "source job": display_job_code(
                        source.metadata.get("job_code"), source.run_id
                    ),
                    "source class": (
                        "Prepared target complex"
                        if key in target_complex_keys
                        else "Prediction poses"
                    ),
                    "target": source_contexts[key]["target"],
                    "campaign": source_contexts[key]["campaign"],
                    "engine": source.tool,
                    "workflow": source.workflow,
                    "focused poses": candidate_count(key),
                    "Native MD geometry": (
                        "Covered"
                        if _covered(
                            children,
                            source,
                            "Native MD geometry",
                            coverage_selection_ids(key),
                        )
                        else "Missing"
                    ),
                    "PLIP": (
                        "Covered"
                        if _covered(
                            children,
                            source,
                            "PLIP",
                            coverage_selection_ids(key),
                        )
                        else "Missing"
                    ),
                    "PandaMap": (
                        "Covered"
                        if _covered(
                            children,
                            source,
                            "PandaMap",
                            coverage_selection_ids(key),
                        )
                        else "Missing"
                    ),
                }
                for key, source in source_jobs.items()
            ])
            edited = st.data_editor(
                frame,
                hide_index=True,
                width="stretch",
                disabled=[column for column in frame if column != "analyze"],
                column_config={
                    "_key": None,
                    "analyze": st.column_config.CheckboxColumn("Analyze"),
                },
                key=(
                    "interaction_source_editor_"
                    + str(
                        st.session_state.get(
                            "_interaction_editor_revision", 0
                        )
                    )
                ),
            )
            selected_keys = edited.loc[
                edited["analyze"].fillna(False).astype(bool), "_key"
            ].astype(str).tolist()
            st.session_state["interaction_selected_sources"] = selected_keys
    selected_keys = [
        key for key in st.session_state.get("interaction_selected_sources", [])
        if key in source_jobs
    ]

    if engine_tab.open:
        selected_engines = st.multiselect(
            "Interaction engines",
            engines,
            key="interaction_selected_engines",
            help="Select both to queue independent PLIP and PandaMap jobs together.",
        )
        columns = st.columns(3)
        columns[0].markdown("#### Native MD geometry")
        columns[0].write(
            "App-native single-pose adaptation of the MD contact analysis. "
            "Reports static contacts, hydrogen-bond candidates, hydrophobic "
            "contacts and formal-charge salt bridges with backbone (BB) or "
            "side-chain (SC) protein-atom annotation."
        )
        columns[1].markdown("#### PLIP")
        columns[1].write(
            "Established rule-based detection of hydrogen bonds, hydrophobic "
            "contacts, salt bridges, π interactions, halogen bonds, water "
            "bridges, and metal complexes."
        )
        columns[2].markdown("#### PandaMap")
        columns[2].write(
            "Expanded interaction mapping with native 2D diagrams and an optional "
            "empirical ΔG estimate. The estimate is descriptive, not rigorous "
            "binding free energy."
        )
        st.info(
            "All engines are CPU-only. Native MD geometry runs in the app "
            "environment; PLIP and PandaMap retain their separate Docker images. "
            "Selecting several methods preserves independent outputs while the "
            "result viewer uses one common 2D rendering style."
        )
        st.metric("Global CPU limit", cpu_process_limit())
        st.caption(
            "CPU workers are assigned automatically per engine job: the "
            "smaller of the global CPU limit and the number of selected "
            "compounds. "
            "The shared CPU lease pool keeps the total across running jobs "
            "within the same global limit."
        )

    if run_tab.open:
        render_run_resources(requires_gpu=False, key="interaction-analysis")
        planned = [
            (key, engine)
            for key in selected_keys
            for engine in selected_engines
            if has_candidates(key)
            and not _covered(
                children,
                source_jobs[key],
                engine,
                coverage_selection_ids(key),
            )
        ]
        st.markdown("#### Submit interaction analysis")
        st.write(
            f"{len(planned)} missing engine job(s) will be queued for "
            f"{len(selected_keys)} selected source result(s)."
        )
        include_covered = st.checkbox(
            "Re-run already covered engine/source combinations",
            value=False,
            key="interaction_rerun_covered",
        )
        if include_covered:
            planned = [
                (key, engine)
                for key in selected_keys
                for engine in selected_engines
                if has_candidates(key)
            ]
        submit = st.button(
            f"Queue {len(planned)} interaction-analysis job(s)",
            type="primary",
            disabled=not planned,
            key="interaction_submit",
        )
        if submit:
            queued, errors = [], []
            submission_candidates: dict[
                str, list[dict[str, object]]
            ] = {}
            submission_sources: dict[str, JobRecord] = {}
            for key, engine in planned:
                if key not in submission_sources:
                    source = source_jobs[key]
                    submission_sources[key] = JobRecord.load(
                        source.run_dir,
                        task_group=source.task_group,
                        validate_artifacts=False,
                    )
                submission_source = submission_sources[key]
                if key not in submission_candidates:
                    cached_rows = inventory_summaries.get(key, {}).get("rows")
                    try:
                        submission_candidates[key] = (
                            list(cached_rows)
                            if isinstance(cached_rows, list)
                            else target_complex_candidates(submission_source)
                            if key in target_complex_keys
                            else cached_pose_validation_candidates(
                                submission_source
                            )
                        )
                    except (OSError, TypeError, ValueError) as exc:
                        errors.append(
                            f"{_source_label(source_jobs[key])}: {exc}"
                        )
                        submission_candidates[key] = []
                selected_rows = submission_candidates[key]
                if not selected_rows:
                    continue
                try:
                    queued.append(
                        queue_interaction_analysis_job(
                            submission_source,
                            engine=engine,
                            selected_rows=selected_rows,
                            selection_schema_version=(
                                TARGET_COMPLEX_INVENTORY_SCHEMA_VERSION
                                if key in target_complex_keys
                                else POSE_VALIDATION_INVENTORY_SCHEMA_VERSION
                            ),
                            selection_policy=(
                                TARGET_COMPLEX_SELECTION_POLICY
                                if key in target_complex_keys
                                else POSE_VALIDATION_SELECTION_POLICY
                            ),
                            source_inventory_kind=(
                                "prepared_target_complex"
                                if key in target_complex_keys
                                else "prediction_poses"
                            ),
                        )
                    )
                except (OSError, TypeError, ValueError) as exc:
                    errors.append(f"{_source_label(source_jobs[key])} · {engine}: {exc}")
            if queued:
                st.success(
                    f"Queued {len(queued)} independent interaction-analysis jobs. "
                    "Selected methods may run at the same time."
                )
            for error in errors:
                st.error(error)

    if results_tab.open:
        rows = _result_rows(interaction_jobs, all_jobs_by_id)
        if rows:
            latest = _latest_result_rows(rows)
            if latest.empty:
                st.warning(
                    "Only legacy interaction results are available. Re-run "
                    "them to use original imported-source residue numbering."
                )
            else:
                filter_columns = st.columns((1.2, 1.2, 1.4, 1))
                prediction_options = sorted(
                    latest["prediction_engine"].dropna().astype(str).unique()
                )
                analysis_options = sorted(
                    latest["analysis_engine"].dropna().astype(str).unique()
                )
                campaign_options = sorted(
                    latest["campaign"].dropna().astype(str).unique()
                )
                status_options = sorted(
                    latest["status"].dropna().astype(str).unique()
                )
                selected_campaigns = filter_columns[0].multiselect(
                    "Campaigns",
                    campaign_options,
                    default=campaign_options,
                    key="interaction_result_campaigns",
                )
                selected_analyses = filter_columns[1].multiselect(
                    "Interaction tools",
                    analysis_options,
                    default=analysis_options,
                    key="interaction_result_analysis_engines",
                )
                selected_predictions = filter_columns[2].multiselect(
                    "Prediction engines",
                    prediction_options,
                    default=prediction_options,
                    key="interaction_result_prediction_engines",
                )
                selected_statuses = filter_columns[3].multiselect(
                    "Status",
                    status_options,
                    default=(
                        ["completed"]
                        if "completed" in status_options
                        else status_options
                    ),
                    key="interaction_result_statuses",
                )
                search = st.text_input(
                    "Find campaign or job",
                    placeholder=(
                        "Campaign name, target, source job, or analysis job"
                    ),
                    key="interaction_result_search",
                ).strip()
                visible = latest.loc[
                    latest["prediction_engine"].isin(selected_predictions)
                    & latest["analysis_engine"].isin(selected_analyses)
                    & latest["campaign"].isin(selected_campaigns)
                    & latest["status"].isin(selected_statuses)
                ].copy()
                if search:
                    searchable = visible[
                        [
                            "campaign", "source_job", "analysis_job",
                            "prediction_engine", "target",
                        ]
                    ].fillna("").astype(str).agg(" ".join, axis=1)
                    visible = visible.loc[
                        searchable.str.contains(
                            search, case=False, regex=False
                        )
                    ]
                st.caption(
                    "One row is shown for the newest PLIP or PandaMap result "
                    "for each immutable source job. The source job identifies "
                    "the exact target-complex or prediction inventory."
                )
                st.dataframe(
                    visible[
                        [
                            "result", "source_job", "prediction_engine",
                            "target", "campaign", "analysis_engine",
                            "analysis_job", "compounds", "poses",
                            "interactions", "status", "created",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "result": st.column_config.LinkColumn(
                            "Result", display_text="Open"
                        ),
                        "source_job": "Source job",
                        "prediction_engine": "Source class / engine",
                        "analysis_engine": "Interaction engine",
                        "analysis_job": "Analysis job",
                    },
                )
                export_signature = (
                    "predicted-complex-path-v1",
                    *visible["analysis_run_id"].astype(str).tolist(),
                )
                if st.button(
                    "Prepare interaction-table downloads",
                    type="primary",
                    disabled=visible.empty,
                    key="prepare_interaction_review_exports",
                    help=(
                        "Uses every result currently visible after applying "
                        "the Campaign, Interaction tool, Prediction engine, "
                        "Status, and search filters."
                    ),
                ):
                    try:
                        with st.spinner(
                            "Collecting normalized interaction tables..."
                        ):
                            (
                                csv_files,
                                workbook,
                                counts,
                                failures,
                            ) = _prepare_interaction_review_exports(visible)
                    except (
                        ImportError,
                        OSError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        st.error(
                            f"Could not prepare interaction-table downloads: {exc}"
                        )
                    else:
                        st.session_state[
                            "interaction_review_export_payload"
                        ] = {
                            "signature": export_signature,
                            "csv_files": csv_files,
                            "workbook": workbook,
                            "counts": counts,
                            "failures": failures,
                        }
                export_payload = st.session_state.get(
                    "interaction_review_export_payload"
                )
                if (
                    isinstance(export_payload, dict)
                    and export_payload.get("signature") == export_signature
                ):
                    csv_files = export_payload.get("csv_files") or {}
                    counts = export_payload.get("counts") or {}
                    failures = export_payload.get("failures") or []
                    if csv_files:
                        st.caption(
                            "Prepared from the currently filtered result rows. "
                            "The combined CSV identifies every interaction "
                            "detection engine; additional CSV files and Excel "
                            "sheets keep the engines separate."
                        )
                        download_items = [
                            (
                                f"Download {tool} CSV",
                                data,
                                (
                                    "interaction-review-"
                                    f"{_export_file_stem(tool)}.csv"
                                ),
                                "text/csv",
                            )
                            for tool, data in csv_files.items()
                        ]
                        workbook = export_payload.get("workbook") or b""
                        if workbook:
                            download_items.append((
                                "Download Excel workbook",
                                workbook,
                                "interaction-review-by-tool.xlsx",
                                (
                                    "application/vnd.openxmlformats-officedocument"
                                    ".spreadsheetml.sheet"
                                ),
                            ))
                        download_columns = st.columns(
                            min(4, len(download_items))
                        )
                        for index, (
                            label,
                            data,
                            file_name,
                            mime,
                        ) in enumerate(download_items):
                            tool = label.removeprefix(
                                "Download "
                            ).removesuffix(" CSV")
                            row_count = counts.get(tool)
                            download_columns[
                                index % len(download_columns)
                            ].download_button(
                                (
                                    f"{label} ({row_count:,} rows)"
                                    if row_count is not None
                                    else label
                                ),
                                data=data,
                                file_name=file_name,
                                mime=mime,
                                key=(
                                    "interaction_review_download:"
                                    f"{file_name}"
                                ),
                            )
                    else:
                        st.warning(
                            "The filtered results contain no interaction rows "
                            "to export."
                        )
                    if failures:
                        with st.expander(
                            f"{len(failures)} result(s) could not be exported"
                        ):
                            for failure in failures:
                                st.write(failure)
            with st.expander("Complete interaction-analysis history"):
                history = pd.DataFrame(rows)
                st.dataframe(
                    history[
                        [
                            "result", "source_job", "prediction_engine",
                            "target", "campaign", "analysis_engine",
                            "analysis_job", "compounds", "poses",
                            "interactions", "status", "created",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "result": st.column_config.LinkColumn(
                            "Result", display_text="Open"
                        )
                    },
                )
        else:
            st.info("No PLIP or PandaMap interaction jobs are available yet.")


render()
