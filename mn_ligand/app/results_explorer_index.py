from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import pandas as pd

from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records


PREDICTION_GROUPS = frozenset({"docking", "refolding", "rescoring"})
EVALUATION_GROUPS = frozenset(
    {
        "pose-validation",
        "interaction-analysis",
        "bound-ligand-md",
        "md-analysis",
        "md-mmgbsa",
        "openfe",
        "abfe",
    }
)

ENGINE_LABELS = {
    "alphafold3": "AlphaFold 3",
    "alphafold3_refolding": "AlphaFold 3",
    "boltz2": "Boltz-2",
    "boltz2_refolding": "Boltz-2",
    "gnina": "GNINA",
    "nesso": "Nesso-1",
    "nesso_refolding": "Nesso-1",
    "openvs": "RosettaLigand",
    "openvs_docking": "RosettaLigand",
    "udp": "Uni-Dock Pro",
    "vina": "AutoDock Vina",
    "boltzina": "Boltzina",
}


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _job_url(job: JobRecord, *, compound_id: str = "") -> str:
    query: dict[str, str] = {
        "task_group": job.task_group,
        "run_id": job.run_id,
        "label": display_job_code(job.metadata.get("job_code"), job.run_id),
    }
    if compound_id:
        query["compound_id"] = compound_id
    return f"./job-results?{urlencode(query)}"


def _engine_label(job: JobRecord) -> str:
    values = (
        job.tool,
        job.workflow,
        job.metadata.get("engine"),
        job.metadata.get("tool"),
    )
    for value in values:
        normalized = str(value or "").strip()
        if normalized in ENGINE_LABELS:
            return ENGINE_LABELS[normalized]
        lowered = normalized.lower()
        if lowered in ENGINE_LABELS:
            return ENGINE_LABELS[lowered]
    return str(job.tool or job.workflow or job.task_group).replace("_", " ").title()


def _run_ids(payload: object) -> Iterable[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "run_id" or key.endswith("_run_id"):
                text = str(value or "").strip()
                if text:
                    yield text
            yield from _run_ids(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _run_ids(value)


def _resolve_dataset_id(
    job: JobRecord,
    jobs_by_id: dict[str, JobRecord],
    *,
    seen: set[str] | None = None,
) -> str:
    visited = set(seen or ())
    if job.run_id in visited:
        return ""
    visited.add(job.run_id)
    if job.task_group == "compound-import":
        return job.run_id
    payloads = (job.metadata, _json(job.run_dir / "input.json"))
    explicit: list[str] = []
    for payload in payloads:
        for key in ("source_compound_run_id", "compound_dataset_run_id", "dataset_run_id"):
            value = str(payload.get(key) or "").strip()
            if value:
                explicit.append(value)
    candidates = [*explicit]
    for payload in payloads:
        candidates.extend(_run_ids(payload))
    for run_id in candidates:
        candidate = jobs_by_id.get(run_id)
        if candidate is None:
            continue
        resolved = _resolve_dataset_id(candidate, jobs_by_id, seen=visited)
        if resolved:
            return resolved
    return ""


def _input_target(job: JobRecord) -> tuple[str, str]:
    payload = _json(job.run_dir / "input.json")
    for key in ("target_artifact", "target"):
        artifact = payload.get(key)
        if not isinstance(artifact, dict):
            continue
        return (
            str(artifact.get("run_id") or job.parent_run_id),
            str(artifact.get("label") or artifact.get("path") or "Prepared target"),
        )
    return str(job.parent_run_id or ""), "Prepared target"


def _target_context(
    job: JobRecord,
    jobs_by_id: dict[str, JobRecord],
) -> tuple[str, str, str, str, str, str, str, str]:
    target_run_id, artifact_label = _input_target(job)
    target = jobs_by_id.get(target_run_id)
    variant = artifact_label or "Prepared target"
    if target is not None:
        variant = str(
            target.metadata.get("target_label")
            or target.metadata.get("display_name")
            or artifact_label
            or target.workflow
            or "Prepared target"
        )
    cursor = target
    seen: set[str] = set()
    family_id = ""
    family = ""
    while cursor is not None and cursor.run_id not in seen:
        seen.add(cursor.run_id)
        pdb_id = str(cursor.metadata.get("pdb_id") or "").strip()
        receptor_value = cursor.metadata.get("receptor") or cursor.metadata.get(
            "protein_name"
        )
        if isinstance(receptor_value, dict):
            entities = receptor_value.get("entities")
            receptor_value = (
                entities[0].get("name")
                if isinstance(entities, list)
                and entities
                and isinstance(entities[0], dict)
                else receptor_value.get("name")
            )
        receptor = str(receptor_value or "").strip()
        if pdb_id:
            family_id = pdb_id.upper()
            family = pdb_id.upper()
            if receptor:
                family = f"{family_id} · {receptor}"
            break
        if receptor and not family:
            family_id = receptor
            family = receptor
        cursor = jobs_by_id.get(cursor.parent_run_id)
    if not family:
        family_id = target_run_id or "unresolved-target"
        family = "Unresolved target"
    variant_code = (
        display_job_code(target.metadata.get("job_code"), target.run_id)
        if target is not None
        else ""
    )
    origin = f"PDB {family_id}" if family_id and family_id != "unresolved-target" else family
    history_entries = (
        target.metadata.get("modification_history")
        if target is not None
        else []
    )
    history_labels: list[str] = []
    if isinstance(history_entries, list):
        for entry in history_entries:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label") or entry.get("kind") or "").strip()
            if label and (not history_labels or history_labels[-1] != label):
                history_labels.append(label)
    history = " → ".join(history_labels) or "No recorded preparation history"
    variant_label = (
        f"{origin} · prepared target {variant_code}"
        if variant_code
        else f"{origin} · prepared target"
    )
    target_result = _job_url(target) if target is not None else ""
    return (
        family_id,
        family,
        target_run_id,
        variant_label,
        origin,
        history,
        variant,
        target_result,
    )


def _dataset_label(dataset_id: str, jobs_by_id: dict[str, JobRecord]) -> str:
    job = jobs_by_id.get(dataset_id)
    if job is None:
        return "Unresolved compound dataset"
    return str(job.metadata.get("dataset_name") or job.metadata.get("original_filename") or dataset_id)


def _compound_ids_from_csv(path: Path) -> set[str]:
    if not path.is_file() or path.stat().st_size > 20_000_000:
        return set()
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        frame = pd.read_csv(path, sep=separator, nrows=200_000)
    except (OSError, ValueError, UnicodeDecodeError):
        return set()
    for column in (
        "candidate_id",
        "compound_id",
        "representative_compound_id",
        "selection_id",
    ):
        if column in frame:
            return {
                str(value).strip()
                for value in frame[column].dropna().tolist()
                if str(value).strip()
            }
    return set()


def compound_ids(job: JobRecord) -> tuple[str, ...]:
    if job.task_group in EVALUATION_GROUPS:
        payload = _json(job.run_dir / "input.json")
        selection = payload.get("selection")
        if isinstance(selection, list):
            values = {
                str(row.get("compound_id") or "").strip()
                for row in selection
                if isinstance(row, dict)
            }
            return tuple(sorted(value for value in values if value))
    values: set[str] = set()
    for path in (
        job.run_dir / "artifacts" / "alphafold3_metrics.csv",
        job.run_dir / "artifacts" / "boltz2_metrics.csv",
        job.run_dir / "artifacts" / "nesso_metrics.csv",
        job.run_dir / "docking_scores.csv",
        job.run_dir / "gnina_scores.csv",
        job.run_dir / "vina_scores.csv",
        job.run_dir / "udp_scores.csv",
        job.run_dir / "openvs_scores.csv",
        job.run_dir / "rescoring_scores.csv",
        job.run_dir / "input" / "compounds.tsv",
    ):
        values.update(_compound_ids_from_csv(path))
    if not values:
        for directory in ("inputs", "data"):
            source = job.run_dir / directory
            if source.is_dir():
                values.update(path.stem for path in source.glob("*.json"))
    return tuple(sorted(values))


def _evaluation_label(job: JobRecord) -> str:
    if job.task_group == "pose-validation":
        return "PoseBusters"
    if job.task_group == "interaction-analysis":
        return str(job.metadata.get("interaction_engine") or job.tool or "Interactions")
    if job.task_group in {"bound-ligand-md", "md-analysis"}:
        return "MD"
    if job.task_group == "md-mmgbsa":
        return "MM/GBSA"
    if job.task_group in {"openfe", "abfe"}:
        return "Free energy"
    return job.task_group.replace("-", " ").title()


def _assign_campaign_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Group legacy engine jobs launched together into one inferred campaign."""
    if frame.empty:
        return frame
    prepared = frame.copy()
    prepared["_created"] = pd.to_datetime(
        prepared["created"], errors="coerce", utc=True
    )
    assigned: dict[str, str] = {}
    labels: dict[str, str] = {}
    primary = prepared.loc[prepared["result_kind"].ne("Rescoring")].copy()
    for _, family in primary.groupby(
        ["dataset_id", "target_variant_id"], dropna=False, sort=False
    ):
        historical = family.loc[family["_explicit_campaign"].eq("")].sort_values(
            "_created"
        )
        batches: list[dict[str, Any]] = []
        for _, row in historical.iterrows():
            created = row["_created"]
            matching = next(
                (
                    batch
                    for batch in reversed(batches)
                    if pd.notna(created)
                    and pd.notna(batch["latest"])
                    and abs((created - batch["latest"]).total_seconds()) <= 900
                ),
                None,
            )
            if matching is None:
                matching = {
                    "id": f"historical:{row['prediction_run_id']}",
                    "latest": created,
                    "rows": [],
                }
                batches.append(matching)
            matching["latest"] = created
            matching["rows"].append(row)
            assigned[str(row["prediction_run_id"])] = str(matching["id"])
        for batch in batches:
            rows = batch["rows"]
            created = min(
                (row["_created"] for row in rows if pd.notna(row["_created"])),
                default=None,
            )
            timestamp = (
                created.strftime("%Y-%m-%d %H:%M UTC")
                if created is not None
                else "Historical launch"
            )
            engines = ", ".join(
                sorted({str(row["prediction_engine"]) for row in rows})
            )
            labels[str(batch["id"])] = f"{timestamp} · {engines}"

    explicit = primary.loc[primary["_explicit_campaign"].ne("")]
    for _, row in explicit.iterrows():
        campaign_id = str(row["_explicit_campaign"])
        assigned[str(row["prediction_run_id"])] = campaign_id
        labels.setdefault(campaign_id, str(row["campaign"]))

    rescoring = prepared.loc[prepared["result_kind"].eq("Rescoring")]
    for _, row in rescoring.iterrows():
        source_id = str(row["_source_prediction_id"] or "")
        campaign_id = assigned.get(source_id) or str(row["_explicit_campaign"] or "")
        if not campaign_id:
            campaign_id = f"historical:{row['prediction_run_id']}"
            labels[campaign_id] = str(row["campaign"])
        assigned[str(row["prediction_run_id"])] = campaign_id

    prepared["campaign_id"] = prepared["prediction_run_id"].map(assigned).fillna(
        prepared["campaign_id"]
    )
    prepared["campaign"] = prepared["campaign_id"].map(labels).fillna(
        prepared["campaign"]
    )
    return prepared.drop(
        columns=["_created", "_explicit_campaign", "_source_prediction_id"]
    )


def build_results_index(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = iter_job_records(run_root)
    jobs_by_id = {job.run_id: job for job in jobs}
    evaluations_by_parent: dict[str, list[JobRecord]] = defaultdict(list)
    for job in jobs:
        if job.task_group in EVALUATION_GROUPS and job.parent_run_id:
            evaluations_by_parent[job.parent_run_id].append(job)

    prediction_rows: list[dict[str, Any]] = []
    compound_rows: list[dict[str, Any]] = []
    for job in jobs:
        if job.task_group not in PREDICTION_GROUPS:
            continue
        source_job = job
        if job.task_group == "rescoring":
            source_id = str(job.metadata.get("source_run_id") or "")
            source_job = jobs_by_id.get(source_id, job)
        dataset_id = _resolve_dataset_id(source_job, jobs_by_id)
        (
            family_id,
            family,
            target_id,
            target_variant,
            target_origin,
            target_history,
            target_artifact,
            target_result,
        ) = _target_context(source_job, jobs_by_id)
        campaign_id = str(
            source_job.metadata.get("launch_campaign_id")
            or job.metadata.get("launch_campaign_id")
            or source_job.run_id
        )
        explicit_campaign = str(
            source_job.metadata.get("launch_campaign_id")
            or job.metadata.get("launch_campaign_id")
            or ""
        )
        campaign = str(
            source_job.metadata.get("launch_campaign_label")
            or job.metadata.get("launch_campaign_label")
            or f"{source_job.created_at[:16].replace('T', ' ')} · {_engine_label(source_job)}"
        )
        children = sorted(
            evaluations_by_parent.get(job.run_id, ()),
            key=lambda child: (
                child.status == "completed",
                child.created_at,
                child.updated_at,
            ),
            reverse=True,
        )
        latest: dict[str, JobRecord] = {}
        for child in children:
            latest.setdefault(_evaluation_label(child), child)
        row = {
            "dataset_id": dataset_id,
            "dataset": _dataset_label(dataset_id, jobs_by_id),
            "target_family_id": family_id,
            "target_family": family,
            "target_variant_id": target_id,
            "target_variant": target_variant,
            "target_origin": target_origin,
            "target_history": target_history,
            "target_artifact": target_artifact,
            "target_result": target_result,
            "campaign_id": campaign_id,
            "campaign": campaign,
            "_explicit_campaign": explicit_campaign,
            "_source_prediction_id": str(
                job.metadata.get("source_run_id") or ""
            ),
            "prediction_run_id": job.run_id,
            "prediction_job": display_job_code(job.metadata.get("job_code"), job.run_id),
            "prediction_engine": _engine_label(job),
            "result_kind": "Rescoring" if job.task_group == "rescoring" else (
                "Cofolding" if job.task_group == "refolding" else "Docking"
            ),
            "status": job.status,
            "created": job.created_at,
            "prediction_result": _job_url(job),
            "posebusters": _job_url(latest["PoseBusters"]) if "PoseBusters" in latest else "",
            "plip": _job_url(latest["PLIP"]) if "PLIP" in latest else "",
            "pandamap": _job_url(latest["PandaMap"]) if "PandaMap" in latest else "",
            "evaluation_count": len(children),
        }
        prediction_rows.append(row)
        for compound_id in compound_ids(job):
            compound_rows.append(
                {
                    **row,
                    "compound_id": compound_id,
                    "prediction_result": _job_url(job, compound_id=compound_id),
                    "posebusters": (
                        _job_url(latest["PoseBusters"], compound_id=compound_id)
                        if "PoseBusters" in latest
                        else ""
                    ),
                    "plip": (
                        _job_url(latest["PLIP"], compound_id=compound_id)
                        if "PLIP" in latest
                        else ""
                    ),
                    "pandamap": (
                        _job_url(latest["PandaMap"], compound_id=compound_id)
                        if "PandaMap" in latest
                        else ""
                    ),
                }
            )
    predictions = _assign_campaign_groups(pd.DataFrame(prediction_rows))
    compounds = pd.DataFrame(compound_rows)
    if not compounds.empty and not predictions.empty:
        campaign_lookup = predictions[
            ["prediction_run_id", "campaign_id", "campaign"]
        ].drop_duplicates("prediction_run_id")
        compounds = compounds.drop(
            columns=[
                "campaign_id",
                "campaign",
                "_explicit_campaign",
                "_source_prediction_id",
            ],
            errors="ignore",
        ).merge(campaign_lookup, on="prediction_run_id", how="left")
    return predictions, compounds
