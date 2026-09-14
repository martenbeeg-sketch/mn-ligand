from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.core.artifacts import ArtifactRef
from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records
from mn_ligand.core.provenance import (
    COMPOUND_DATASET_CAMPAIGN_PURPOSE,
    TARGET_LIGAND_CAMPAIGN_PURPOSE,
    binding_campaign_purpose,
    inherited_metadata_value,
    is_benchmark_job,
    iter_job_lineage,
    modification_history,
    provenance_last_step,
    provenance_origin_label,
)
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.bound_ligand_md import is_bound_ligand_atom, parse_bound_ligands
from mn_ligand.workflows.target_orientation import (
    transform_pdb_data,
    transformed_ligand_sdf,
)
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.app.viewers import render_persistent_3dmol


@dataclass(frozen=True)
class InputSpec:
    label: str
    artifact_types: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class ArtifactChoice:
    job: JobRecord
    artifact: ArtifactRef


@dataclass(frozen=True)
class TargetInventoryEntry:
    choice: ArtifactChoice
    viewer_path: Path
    row: dict[str, Any]


def hide_superseded_target_versions(
    entries: list[TargetInventoryEntry],
) -> list[TargetInventoryEntry]:
    """Hide internal preparation attempts once their target has been published.

    Explicit minimization, trimming, and terminal-repair jobs are scientifically
    meaningful target versions.  Keep them and their published parents visible;
    only collapse the protein-cleaning implementation jobs that feed the
    user-facing structure publication for the same target.
    """
    if len(entries) < 2:
        return entries

    published_targets = {
        str(entry.row.get("Target") or "").strip().upper()
        for entry in entries
        if entry.choice.job.task_group == "structure-jobs"
    }
    published_targets.discard("")

    visible: list[TargetInventoryEntry] = []
    for entry in entries:
        job = entry.choice.job
        target = str(entry.row.get("Target") or "").strip().upper()
        if job.task_group == "protein-cleaning" and target in published_targets:
            continue
        visible.append(entry)
    return visible


def is_discovery_target_artifact(
    job: JobRecord,
    artifact: ArtifactRef,
) -> bool:
    if artifact.artifact_type not in {"prepared_target", "prepared_receptor"}:
        return True
    workflow = str(job.workflow or "").lower()
    operation = str(job.metadata.get("operation") or "").lower()
    role = str(artifact.role or "").lower()
    if workflow in {
        "docking_redocking",
        "pose_selection",
        "target_orientation",
    }:
        return False
    if operation in {"docking", "rescoring_selection"}:
        return False
    return role not in {"axis_aligned_receptor", "rescoring_receptor"}


def target_binding_result_links(
    jobs: list[JobRecord],
    jobs_by_id: dict[str, JobRecord],
    visible_target_run_ids: set[str],
    *,
    campaign_purpose: str,
) -> dict[str, str]:
    target_run_ids: set[str] = set()
    for job in jobs:
        if job.status != "completed" or is_benchmark_job(job, jobs_by_id):
            continue
        if binding_campaign_purpose(job) != campaign_purpose:
            continue
        operation = str(
            job.metadata.get("operation")
            or job.metadata.get("mode")
            or ""
        ).lower()
        workflow = str(job.workflow or "").lower()
        if (
            operation not in {"docking", "redocking", "refolding"}
            and workflow != "docking_redocking"
        ):
            continue
        target_run_id = next(
            (
                candidate.run_id
                for candidate in iter_job_lineage(job, jobs_by_id)
                if candidate.run_id != job.run_id
                and candidate.run_id in visible_target_run_ids
            ),
            "",
        )
        if not target_run_id:
            continue
        target_run_ids.add(target_run_id)

    return {
        target_run_id: (
            "./compound-campaign-comparison?"
            + urlencode(
                {
                    "target_run_id": target_run_id,
                    "campaign_purpose": campaign_purpose,
                    "label": "Campaigns",
                }
            )
        )
        for target_run_id in target_run_ids
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


@st.cache_data(show_spinner=False)
def _structure_inventory(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    path = Path(path_text)
    if path.suffix.lower() not in {".pdb", ".ent"}:
        return {"chains": "", "residues": None, "ligands": [], "ligand_keys": []}
    text = path.read_text(errors="replace")
    chains: set[str] = set()
    residues: set[tuple[str, str, str]] = set()
    for line in text.splitlines():
        if not line.startswith("ATOM") or len(line) < 27 or is_bound_ligand_atom(line):
            continue
        chain = line[21].strip() or "_"
        chains.add(chain)
        residues.add((chain, line[22:26].strip(), line[26].strip()))
    ligands = parse_bound_ligands(text)
    return {
        "chains": ",".join(sorted(chains)),
        "residues": len(residues),
        "ligands": [str(item.get("resname") or "LIG") for item in ligands],
        "ligand_keys": [str(item.get("key") or "") for item in ligands],
        "ligand_records": ligands,
    }


def _reference_smiles(
    job: JobRecord,
    jobs_by_id: dict[str, JobRecord],
) -> str:
    for candidate in iter_job_lineage(job, jobs_by_id):
        value = str(candidate.metadata.get("ligand_smiles") or "").strip()
        if value:
            return value
        if not candidate.artifact_manifest:
            continue
        for artifact in candidate.artifact_manifest.artifacts:
            if artifact.role != "reference_smiles" and not artifact.path.lower().endswith((".smi", ".smiles")):
                continue
            path = artifact.resolve(candidate.run_dir, must_exist=True)
            if path is not None:
                return path.read_text(errors="replace").strip().split()[0]
    return ""


@st.cache_data(show_spinner=False)
def _smiles_properties(smiles: str) -> dict[str, Any]:
    if not smiles:
        return {"formula": "", "molecular_weight": None}
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return {"formula": "", "molecular_weight": None}
        return {
            "formula": rdMolDescriptors.CalcMolFormula(molecule),
            "molecular_weight": round(float(Descriptors.MolWt(molecule)), 2),
        }
    except Exception:
        return {"formula": "", "molecular_weight": None}


def _provenance_ligands(job: JobRecord, jobs_by_id: dict[str, JobRecord]) -> list[dict[str, Any]]:
    for candidate in iter_job_lineage(job, jobs_by_id):
        candidates = candidate.metadata.get("ligands")
        if isinstance(candidates, list) and candidates:
            return [dict(item) for item in candidates if isinstance(item, dict)]
    return []


def _provenance_receptor(job: JobRecord, jobs_by_id: dict[str, JobRecord]) -> dict[str, Any]:
    for candidate in iter_job_lineage(job, jobs_by_id):
        receptor = candidate.metadata.get("receptor")
        if isinstance(receptor, dict) and receptor:
            return dict(receptor)
    return {}


def _receptor_inventory(job: JobRecord, jobs_by_id: dict[str, JobRecord]) -> dict[str, Any]:
    receptor = _provenance_receptor(job, jobs_by_id)
    entities = [item for item in receptor.get("entities") or [] if isinstance(item, dict)]
    names = list(dict.fromkeys(str(item.get("name") or "").strip() for item in entities if item.get("name")))
    organisms = list(
        dict.fromkeys(
            str(value).strip()
            for item in entities
            for value in item.get("source_organisms") or []
            if str(value).strip()
        )
    )
    uniprot_ids = list(
        dict.fromkeys(
            str(value).strip()
            for item in entities
            for value in item.get("uniprot_ids") or []
            if str(value).strip()
        )
    )
    return {
        "name": "; ".join(names) or "-",
        "organism": ", ".join(organisms) or "-",
        "uniprot": ", ".join(uniprot_ids) or "-",
        "method": str(receptor.get("experimental_method") or "-") or "-",
        "resolution": receptor.get("resolution_angstrom"),
        "title": str(receptor.get("title") or "-") or "-",
    }


def _ligand_inventory(
    job: JobRecord,
    stats: dict[str, Any],
    jobs_by_id: dict[str, JobRecord],
) -> dict[str, Any]:
    coordinate_records = list(stats.get("ligand_records") or [])
    coordinate_ids = list(dict.fromkeys(str(item.get("resname") or "") for item in coordinate_records))
    provenance = _provenance_ligands(job, jobs_by_id)
    ligand_key = str(
        inherited_metadata_value(job, jobs_by_id, "ligand_key", default="") or ""
    )
    selected_code = ligand_key.partition("|")[0].strip()
    ligand_id = str(
        inherited_metadata_value(job, jobs_by_id, "ligand_id", default="") or ""
    ).strip()
    if selected_code and ligand_key:
        selected_records = [
            item for item in provenance if str(item.get("ccd_id") or item.get("resname") or "") == selected_code
        ]
        if selected_records:
            provenance = selected_records

    provenance_ids = [
        str(item.get("ccd_id") or item.get("resname") or "").strip()
        for item in provenance
        if str(item.get("ccd_id") or item.get("resname") or "").strip()
    ]
    identities = [ligand_id] if ligand_id else ([selected_code] if selected_code else provenance_ids)
    if not identities:
        identities = coordinate_ids
    identities = list(dict.fromkeys(value for value in identities if value))
    names = list(
        dict.fromkeys(str(item.get("name") or "").strip() for item in provenance if item.get("name"))
    )
    formulas = list(
        dict.fromkeys(str(item.get("formula") or "").strip() for item in provenance if item.get("formula"))
    )
    properties = _smiles_properties(_reference_smiles(job, jobs_by_id))
    if not formulas and properties["formula"]:
        formulas = [str(properties["formula"])]
    molecular_weight = properties["molecular_weight"]
    if molecular_weight is None:
        weights = [
            item.get("molecular_weight")
            for item in provenance
            if item.get("molecular_weight") not in (None, "")
        ]
        molecular_weight = weights[0] if weights else None
    return {
        "identities": ", ".join(identities) or "-",
        "coordinate_ids": ", ".join(coordinate_ids) or "-",
        "names": "; ".join(names) or "-",
        "formulas": ", ".join(formulas) or "-",
        "molecular_weight": molecular_weight,
    }


def _source_complex_path(job: JobRecord, jobs_by_id: dict[str, JobRecord]) -> Path | None:
    if job.artifact_manifest:
        for artifact_type in ("prepared_complex", "docked_complex"):
            complexes = job.artifact_manifest.by_type(artifact_type)
            if complexes:
                path = complexes[0].resolve(job.run_dir, must_exist=True)
                if path is not None:
                    return path
    import_run_id = str(job.metadata.get("import_run_id") or "")
    source_job = jobs_by_id.get(import_run_id)
    if source_job and source_job.artifact_manifest:
        for artifact_type in ("imported_target", "prepared_complex"):
            artifacts = source_job.artifact_manifest.by_type(artifact_type)
            if artifacts:
                path = artifacts[0].resolve(source_job.run_dir, must_exist=True)
                if path is not None:
                    return path
    return None


def _origin(job: JobRecord, jobs_by_id: dict[str, JobRecord]) -> str:
    return provenance_origin_label(job, jobs_by_id)


def _preparation(
    job: JobRecord,
    artifact: ArtifactRef,
    jobs_by_id: dict[str, JobRecord],
) -> str:
    steps: list[str] = []
    steps.extend(
        str(item.get("summary") or "")
        for item in modification_history(job, jobs_by_id)
        if item.get("summary")
    )
    input_payload = _read_json(job.run_dir / "input.json")
    parameters = input_payload.get("parameters") if isinstance(input_payload.get("parameters"), dict) else {}
    if parameters.get("clean_protein") or job.job_type == "protein_cleaning":
        steps.append("Cleaned/repaired")
    if parameters.get("map_modified_residues"):
        steps.append("Modified residues mapped")
    ph = parameters.get("ph") or artifact.metadata.get("protonation_ph")
    if ph not in (None, ""):
        steps.append(f"pH {ph}")
    if job.metadata.get("use_scrub"):
        steps.append("Ligand scrubbed")
    repair = job.metadata.get("receptor_meeko_repair")
    if isinstance(repair, dict) and repair.get("repair_applied"):
        steps.append("Receptor repaired")
    if artifact.artifact_type == "prepared_complex":
        steps.append("Complex prepared")
    elif artifact.artifact_type == "docked_complex":
        steps.append("Docked complex; MD preparation required")
    elif artifact.artifact_type in {"prepared_target", "prepared_receptor"} and not steps:
        steps.append("Receptor prepared")
    return "; ".join(dict.fromkeys(steps)) or "Imported"


@st.cache_data(show_spinner=False, ttl=15)
def _cached_job_records(runs_dir_text: str) -> list[JobRecord]:
    """Load the filesystem job inventory once for one burst of UI reruns."""
    return iter_job_records(Path(runs_dir_text))


def job_records_snapshot(runs_dir_text: str | None = None) -> list[JobRecord]:
    return _cached_job_records(
        runs_dir_text or str(runs_root().resolve())
    )


@st.cache_data(show_spinner=False, ttl=30)
def _cached_target_inventory(
    artifact_types: tuple[str, ...] = ("prepared_target", "prepared_receptor"),
    *,
    include_benchmarks: bool = False,
    include_transient_targets: bool = False,
    runs_dir_text: str,
) -> list[TargetInventoryEntry]:
    jobs = job_records_snapshot(runs_dir_text)
    jobs_by_id = {job.run_id: job for job in jobs}
    entries: list[TargetInventoryEntry] = []
    for job in jobs:
        if job.status != "completed" or not job.artifact_manifest:
            continue
        if not include_benchmarks and is_benchmark_job(job, jobs_by_id):
            continue
        for artifact in job.artifact_manifest.artifacts:
            if artifact.artifact_type not in artifact_types:
                continue
            if (
                not include_transient_targets
                and not is_discovery_target_artifact(job, artifact)
            ):
                continue
            path = artifact.resolve(job.run_dir, must_exist=True)
            if path is None:
                continue
            viewer_path = _source_complex_path(job, jobs_by_id) or path
            stats = _structure_inventory(str(viewer_path), viewer_path.stat().st_mtime_ns)
            ligand_info = _ligand_inventory(job, stats, jobs_by_id)
            receptor_info = _receptor_inventory(job, jobs_by_id)
            code = display_job_code(job.metadata.get("job_code"), job.run_id)
            job_url = "./job-results?" + urlencode(
                {
                    "task_group": job.task_group,
                    "run_id": job.run_id,
                    "label": code,
                }
            )
            target_name = str(
                inherited_metadata_value(job, jobs_by_id, "pdb_id", default="")
                or artifact.label
                or path.stem
            )
            kind = (
                "Complex"
                if artifact.artifact_type in {"prepared_complex", "docked_complex"}
                else "Receptor"
            )
            entries.append(
                TargetInventoryEntry(
                    choice=ArtifactChoice(job=job, artifact=artifact),
                    viewer_path=viewer_path,
                    row={
                        "Target key": f"{target_name} · {code}",
                        "Target": target_name,
                        "Target run ID": job.run_id,
                        "Job": job_url,
                        "Docking / cofolding": "",
                        "Redocking / refolding": "",
                        "Receptor": receptor_info["name"],
                        "Ligands": ligand_info["identities"],
                        "Tool": str(
                            job.tool or job.metadata.get("engine") or "-"
                        ).replace("_", " "),
                        "Origin": _origin(job, jobs_by_id),
                        "Last step": provenance_last_step(job, jobs_by_id),
                        "Residues": stats["residues"],
                        "Organism": receptor_info["organism"],
                        "UniProt": receptor_info["uniprot"],
                        "Kind": kind,
                        "Compound": ligand_info["names"],
                        "Coordinate ID": ligand_info["coordinate_ids"],
                        "Formula": ligand_info["formulas"],
                        "MW (Da)": ligand_info["molecular_weight"],
                        "Method": receptor_info["method"],
                        "Resolution (A)": receptor_info["resolution"],
                        "PDB Title": receptor_info["title"],
                        "Chains": stats["chains"] or "-",
                        "Preparation": _preparation(job, artifact, jobs_by_id),
                        "Created": job.created_at,
                    },
                )
            )
    docking_links = target_binding_result_links(
        jobs,
        jobs_by_id,
        {entry.choice.job.run_id for entry in entries},
        campaign_purpose=COMPOUND_DATASET_CAMPAIGN_PURPOSE,
    )
    redocking_links = target_binding_result_links(
        jobs,
        jobs_by_id,
        {entry.choice.job.run_id for entry in entries},
        campaign_purpose=TARGET_LIGAND_CAMPAIGN_PURPOSE,
    )
    for entry in entries:
        entry.row["Docking / cofolding"] = docking_links.get(
            entry.choice.job.run_id,
            "",
        )
        entry.row["Redocking / refolding"] = redocking_links.get(
            entry.choice.job.run_id,
            "",
        )
    entries.sort(
        key=lambda entry: (
            not bool(entry.choice.job.created_at),
            str(entry.choice.job.created_at or ""),
            entry.choice.job.run_id,
            entry.choice.artifact.artifact_id,
        )
    )
    return entries


def target_inventory(
    artifact_types: tuple[str, ...] = ("prepared_target", "prepared_receptor"),
    *,
    include_benchmarks: bool = False,
    include_transient_targets: bool = False,
) -> list[TargetInventoryEntry]:
    return _cached_target_inventory(
        artifact_types,
        include_benchmarks=include_benchmarks,
        include_transient_targets=include_transient_targets,
        runs_dir_text=str(runs_root().resolve()),
    )


def _matches_word_query(
    query: str, values: tuple[object, ...], *, require_all: bool
) -> bool:
    words = [word.casefold() for word in query.replace(",", " ").split()]
    if not words:
        return True
    searchable = " ".join(str(value) for value in values).casefold()
    checks = (word in searchable for word in words)
    return all(checks) if require_all else any(checks)


def _target_text_filters(
    entries: list[TargetInventoryEntry], *, key: str
) -> list[TargetInventoryEntry]:
    filter_columns = st.columns(4)
    target_query = filter_columns[0].text_input(
        "Target / PDB / ID",
        key=f"{key}_target_search",
        placeholder="PDB, target key, short code, or full run ID",
    )
    ligand_query = filter_columns[1].text_input(
        "Ligand",
        key=f"{key}_ligand_search",
        placeholder="Name, ID, or formula",
    )
    receptor_query = filter_columns[2].text_input(
        "Receptor",
        key=f"{key}_receptor_search",
        placeholder="Receptor words",
    )
    last_step_query = filter_columns[3].text_input(
        "Last step",
        key=f"{key}_last_step_search",
        placeholder="Preparation step words",
    )
    word_mode = st.segmented_control(
        "Words within each field",
        ("Any (OR)", "All (AND)"),
        default="Any (OR)",
        key=f"{key}_word_mode",
        help=(
            "Any matches a row when at least one entered word occurs in that "
            "field. All requires every entered word. Different populated "
            "fields are always combined with AND."
        ),
    ) or "Any (OR)"

    def matches(query: str, values: tuple[object, ...]) -> bool:
        return _matches_word_query(
            query,
            values,
            require_all=word_mode == "All (AND)",
        )

    return [
        entry
        for entry in entries
        if matches(
            target_query,
            (
                entry.row.get("Target key", ""),
                entry.row.get("Target", ""),
                entry.row.get("Target run ID", ""),
            ),
        )
        and matches(
            ligand_query,
            (
                entry.row.get("Ligands", ""),
                entry.row.get("Compound", ""),
                entry.row.get("Coordinate ID", ""),
                entry.row.get("Formula", ""),
            ),
        )
        and matches(receptor_query, (entry.row.get("Receptor", ""),))
        and matches(last_step_query, (entry.row.get("Last step", ""),))
    ]


def select_target_artifact(
    label: str,
    artifact_types: tuple[str, ...],
    *,
    key: str,
    requested_run_id: str = "",
    show_viewer: bool = True,
    allowed_run_ids: set[str] | None = None,
    excluded_run_ids: set[str] | None = None,
    row_annotations: dict[str, dict[str, Any]] | None = None,
    row_annotation_defaults: dict[str, Any] | None = None,
) -> ArtifactChoice | None:
    entries = target_inventory(artifact_types)
    if allowed_run_ids is not None:
        entries = [entry for entry in entries if entry.choice.job.run_id in allowed_run_ids]
    if excluded_run_ids:
        entries = [
            entry
            for entry in entries
            if entry.choice.job.run_id not in excluded_run_ids
        ]
    if row_annotations is not None or row_annotation_defaults is not None:
        annotations = row_annotations or {}
        defaults = row_annotation_defaults or {}
        entries = [
            TargetInventoryEntry(
                choice=entry.choice,
                viewer_path=entry.viewer_path,
                row={
                    **entry.row,
                    **defaults,
                    **annotations.get(entry.choice.job.run_id, {}),
                },
            )
            for entry in entries
        ]
    show_previous_versions = st.checkbox(
        "Show previous target versions",
        value=False,
        key=f"{key}_show_previous_versions",
        help=(
            "Reveal internal and historical preparation attempts. Published "
            "base, minimized, trimmed, and repaired target versions remain "
            "visible by default."
        ),
    )
    if not show_previous_versions:
        entries = hide_superseded_target_versions(entries)
    if not entries:
        st.info(f"No compatible {label.lower()} is available.")
        return None

    filtered = _target_text_filters(entries, key=key)
    if not filtered:
        st.info("No targets match the active filters.")
        return None

    st.caption(
        "Target key combines the PDB/target name with the short job code. "
        "Target run ID is the complete immutable ID used on result pages."
    )

    state_key = f"{key}_selected_id"
    valid_ids = {f"{entry.choice.job.run_id}:{entry.choice.artifact.artifact_id}" for entry in filtered}
    preferred = next(
        (
            f"{entry.choice.job.run_id}:{entry.choice.artifact.artifact_id}"
            for entry in filtered
            if entry.choice.job.run_id == requested_run_id
        ),
        "",
    )
    selected_id = str(st.session_state.get(state_key) or preferred or "")
    if selected_id not in valid_ids:
        selected_id = f"{filtered[0].choice.job.run_id}:{filtered[0].choice.artifact.artifact_id}"
    event = st.dataframe(
        pd.DataFrame([entry.row for entry in filtered]),
        hide_index=True,
        width="stretch",
        height=min(430, 38 + 35 * len(filtered)),
        on_select="rerun",
        selection_mode="single-row",
        key=f"{key}_table",
        column_config={
            "Job": st.column_config.LinkColumn(
                "Target job",
                display_text=r"label=([^&]+)",
            ),
            "Target key": st.column_config.TextColumn(
                "Target key",
                help="PDB/target name · short job code",
                width="medium",
            ),
            "Target run ID": st.column_config.TextColumn(
                "Target run ID",
                help="Complete immutable target identifier used by result pages",
                width="large",
            ),
            "Docking / cofolding": st.column_config.LinkColumn(
                "Docking / cofolding",
                display_text=r"label=([^&]+)",
            ),
            "Redocking / refolding": st.column_config.LinkColumn(
                "Redocking / refolding",
                display_text=r"label=([^&]+)",
            ),
            "Residues": st.column_config.NumberColumn(format="%d"),
            "MW (Da)": st.column_config.NumberColumn(format="%.2f"),
            "Resolution (A)": st.column_config.NumberColumn(format="%.2f"),
            "PDB Title": st.column_config.TextColumn(width="large"),
            "Pocket Detection": st.column_config.TextColumn(width="large"),
            "Created": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )
    selected_rows = list(getattr(getattr(event, "selection", None), "rows", []) or [])
    if selected_rows:
        selected_entry = filtered[int(selected_rows[0])]
        selected_id = f"{selected_entry.choice.job.run_id}:{selected_entry.choice.artifact.artifact_id}"
        st.session_state[state_key] = selected_id
    selected_entry = next(
        entry
        for entry in filtered
        if f"{entry.choice.job.run_id}:{entry.choice.artifact.artifact_id}" == selected_id
    )
    selected_code = display_job_code(
        selected_entry.choice.job.metadata.get("job_code"),
        selected_entry.choice.job.run_id,
    )
    st.caption(
        f"Selected: {selected_entry.row['Target']} · {selected_code} · "
        f"target run ID {selected_entry.choice.job.run_id}"
    )
    if show_viewer:
        render_target_viewer(selected_entry.choice, viewer_path=selected_entry.viewer_path, key=f"{key}_viewer")
    return selected_entry.choice


def select_target_artifacts(
    label: str,
    artifact_types: tuple[str, ...],
    *,
    key: str,
    minimum: int = 1,
    maximum: int | None = None,
    allowed_run_ids: set[str] | None = None,
    requested_run_ids: set[str] | None = None,
) -> tuple[ArtifactChoice, ...]:
    entries = target_inventory(artifact_types)
    if allowed_run_ids is not None:
        entries = [entry for entry in entries if entry.choice.job.run_id in allowed_run_ids]
    show_previous_versions = st.checkbox(
        "Show previous target versions",
        value=False,
        key=f"{key}_show_previous_versions",
        help=(
            "Reveal internal and historical preparation attempts. Published "
            "base, minimized, trimmed, and repaired target versions remain "
            "visible by default."
        ),
    )
    if not show_previous_versions:
        entries = hide_superseded_target_versions(entries)
    if not entries:
        st.info(f"No compatible {label.lower()} is available.")
        return ()
    filtered = _target_text_filters(entries, key=key)
    if not filtered:
        st.info("No targets match the active filters.")
        return ()
    st.caption(
        "Target key combines the PDB/target name with the short job code. "
        "Target run ID is the complete immutable ID used on result pages."
    )
    state_key = f"{key}_selected_ids"
    valid_ids = [f"{entry.choice.job.run_id}:{entry.choice.artifact.artifact_id}" for entry in filtered]
    selected_ids = [
        value for value in st.session_state.get(state_key, []) if value in set(valid_ids)
    ]
    if not selected_ids and requested_run_ids:
        selected_ids = [
            entry_id
            for entry, entry_id in zip(filtered, valid_ids, strict=True)
            if entry.choice.job.run_id in requested_run_ids
        ]
    if len(selected_ids) < minimum:
        selected_ids = valid_ids[: min(max(minimum, 1), len(valid_ids))]
    if maximum is not None:
        selected_ids = selected_ids[: max(0, int(maximum))]
        st.session_state[state_key] = selected_ids
    event = st.dataframe(
        pd.DataFrame([entry.row for entry in filtered]),
        hide_index=True,
        width="stretch",
        height=min(430, 38 + 35 * len(filtered)),
        on_select="rerun",
        selection_mode=(
            "single-row" if maximum == 1 else "multi-row"
        ),
        key=f"{key}_table",
        column_config={
            "Job": st.column_config.LinkColumn(
                "Target job",
                display_text=r"label=([^&]+)",
            ),
            "Target key": st.column_config.TextColumn(
                "Target key",
                help="PDB/target name · short job code",
                width="medium",
            ),
            "Target run ID": st.column_config.TextColumn(
                "Target run ID",
                help="Complete immutable target identifier used by result pages",
                width="large",
            ),
            "Docking / cofolding": st.column_config.LinkColumn(
                "Docking / cofolding",
                display_text=r"label=([^&]+)",
            ),
            "Redocking / refolding": st.column_config.LinkColumn(
                "Redocking / refolding",
                display_text=r"label=([^&]+)",
            ),
            "Residues": st.column_config.NumberColumn(format="%d"),
            "MW (Da)": st.column_config.NumberColumn(format="%.2f"),
            "Resolution (A)": st.column_config.NumberColumn(format="%.2f"),
            "PDB Title": st.column_config.TextColumn(width="large"),
            "Created": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )
    selected_rows = list(getattr(getattr(event, "selection", None), "rows", []) or [])
    if selected_rows:
        selected_ids = [valid_ids[int(index)] for index in selected_rows]
        if maximum is not None:
            selected_ids = selected_ids[: max(0, int(maximum))]
        st.session_state[state_key] = selected_ids
    selected = tuple(
        entry.choice
        for entry, entry_id in zip(filtered, valid_ids)
        if entry_id in set(selected_ids)
    )
    if selected:
        selected_id_set = set(selected_ids)
        selected_labels = [
            (
                f"{entry.row.get('Target key', entry.row.get('Target', '-'))} "
                f"· {entry.choice.job.run_id}"
            )
            for entry, entry_id in zip(filtered, valid_ids, strict=True)
            if entry_id in selected_id_set
        ]
        st.caption("Selected: " + "; ".join(selected_labels))
    else:
        st.caption(f"Selected 0 {label.lower()}.")
    return selected


def artifact_box(choice: ArtifactChoice | None) -> dict[str, tuple[float, float, float]] | None:
    if choice is None:
        return None
    metadata = choice.artifact.metadata
    center = metadata.get("center_angstrom") or metadata.get("center")
    size = metadata.get("size_angstrom") or metadata.get("size")
    if isinstance(center, dict):
        center = [center.get(axis) for axis in ("x", "y", "z")]
    if isinstance(size, dict):
        size = [size.get(axis) for axis in ("x", "y", "z")]
    try:
        center_values = tuple(float(value) for value in center)
        size_values = tuple(float(value) for value in size)
    except (TypeError, ValueError):
        return None
    if len(center_values) != 3 or len(size_values) != 3:
        return None
    return {"center": center_values, "size": size_values}


def target_viewer_path(choice: ArtifactChoice) -> Path | None:
    for entry in target_inventory(
        (choice.artifact.artifact_type,),
        include_benchmarks=True,
        include_transient_targets=True,
    ):
        if (
            entry.choice.job.run_id == choice.job.run_id
            and entry.choice.artifact.artifact_id == choice.artifact.artifact_id
        ):
            return entry.viewer_path
    return choice.artifact.resolve(choice.job.run_dir, must_exist=True)


def target_ligand_path(choice: ArtifactChoice) -> Path | None:
    if choice.job.artifact_manifest:
        candidates = [
            artifact
            for artifact in choice.job.artifact_manifest.artifacts
            if artifact.artifact_type
            in {
                "prepared_ligand_set",
                "pose_set",
                "docked_pose",
                "reference_ligand",
            }
        ]
        refined = [item for item in candidates if item.path.lower().endswith("_ligand_refined.sdf")]
        for artifact in (*refined, *candidates):
            path = artifact.resolve(choice.job.run_dir, must_exist=True)
            if path is not None and path.suffix.lower() in {
                ".sdf",
                ".mol",
                ".mol2",
                ".pdb",
            }:
                return path
    return next(iter(sorted(choice.job.run_dir.glob("*_ligand_refined.sdf"))), None)


def target_coordinate_ligand_box(
    choice: ArtifactChoice,
) -> dict[str, tuple[float, float, float]] | None:
    """Return the raw axis-aligned extent of the first associated ligand pose."""
    path = target_ligand_path(choice)
    if path is None:
        return None
    try:
        from rdkit import Chem
        import numpy as np

        suffix = path.suffix.lower()
        if suffix == ".sdf":
            molecule = next(
                (
                    item
                    for item in Chem.SDMolSupplier(
                        str(path), removeHs=False, sanitize=False
                    )
                    if item is not None
                ),
                None,
            )
        elif suffix == ".mol":
            molecule = Chem.MolFromMolFile(
                str(path), removeHs=False, sanitize=False
            )
        elif suffix == ".mol2":
            molecule = Chem.MolFromMol2File(
                str(path), removeHs=False, sanitize=False
            )
        else:
            molecule = Chem.MolFromPDBFile(
                str(path), removeHs=False, sanitize=False
            )
        if molecule is None or not molecule.GetNumConformers():
            return None
        coordinates = np.asarray(
            molecule.GetConformer().GetPositions(), dtype=float
        )
        heavy = [
            atom.GetIdx()
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() > 1
        ]
        if heavy:
            coordinates = coordinates[heavy]
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        center = (minimum + maximum) / 2.0
        size = np.maximum(maximum - minimum, 1.0)
        return {
            "center": tuple(float(value) for value in center),
            "size": tuple(float(value) for value in size),
        }
    except (OSError, RuntimeError, ValueError):
        return None


def bound_ligand_box(
    choice: ArtifactChoice,
    ligand_key: str,
    *,
    padding_angstrom: float = 4.0,
) -> dict[str, tuple[float, float, float]] | None:
    path = target_viewer_path(choice)
    if path is None or path.suffix.lower() not in {".pdb", ".ent"}:
        return None
    pdb_data = path.read_text(errors="replace")
    available = parse_bound_ligands(pdb_data)
    available_keys = {str(item.get("key") or "") for item in available}
    effective_key = ligand_key
    if effective_key not in available_keys and len(available) == 1:
        effective_key = str(available[0].get("key") or "")
    coordinates: list[tuple[float, float, float]] = []
    for line in pdb_data.splitlines():
        if not is_bound_ligand_atom(line):
            continue
        key = "|".join(
            (
                line[17:20].strip(),
                line[21].strip() or "_",
                line[22:26].strip(),
                line[26].strip() or "_",
            )
        )
        if key != effective_key:
            continue
        try:
            coordinates.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
    if not coordinates:
        return None
    axes = list(zip(*coordinates))
    center = tuple((min(axis) + max(axis)) / 2.0 for axis in axes)
    size = tuple(max(4.0, max(axis) - min(axis) + 2.0 * padding_angstrom) for axis in axes)
    return {"center": center, "size": size}


def render_target_viewer(
    choice: ArtifactChoice,
    *,
    viewer_path: Path | None = None,
    ligand_path: Path | None = None,
    box: dict[str, tuple[float, float, float]] | None = None,
    selected_ligand_key: str = "",
    show_ligand: bool = True,
    show_box: bool = True,
    cartoon_color: str = "spectrum",
    ligand_color: str = "redCarbon",
    box_color: str = "#ef4444",
    show_box_center: bool = False,
    coordinate_transform: dict[str, Any] | None = None,
    key: str = "target_viewer",
    height: int = 540,
) -> None:
    path = viewer_path or choice.artifact.resolve(choice.job.run_dir, must_exist=True)
    if path is None or path.suffix.lower() not in {".pdb", ".ent"}:
        st.info("A PDB preview is not available for the selected target.")
        return
    if path.stat().st_size > 15 * 1024 * 1024:
        st.info("The selected structure is larger than the 15 MB preview limit.")
        return
    import py3Dmol

    pdb_data = path.read_text(errors="replace")
    if coordinate_transform is not None:
        pdb_data = transform_pdb_data(pdb_data, coordinate_transform)
    viewer = py3Dmol.view(width="100%", height=max(240, height - 20))
    viewer.addModel(pdb_data, "pdb")
    viewer.setStyle({"hetflag": False}, {"cartoon": {"color": cartoon_color}})
    viewer.setStyle({"hetflag": True}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.2}})
    effective_ligand_key = selected_ligand_key
    available_ligands = parse_bound_ligands(pdb_data)
    available_keys = {str(item.get("key") or "") for item in available_ligands}
    if effective_ligand_key not in available_keys and len(available_ligands) == 1:
        effective_ligand_key = str(available_ligands[0].get("key") or "")
    if show_ligand and effective_ligand_key:
        parts = effective_ligand_key.split("|")
        if len(parts) >= 3:
            selector = {"resn": parts[0], "chain": "" if parts[1] == "_" else parts[1]}
            try:
                selector["resi"] = int(parts[2])
            except ValueError:
                pass
            viewer.setStyle(
                selector,
                {
                    "stick": {"colorscheme": ligand_color, "radius": 0.22},
                    "sphere": {"colorscheme": ligand_color, "scale": 0.18},
                },
            )
    explicit_ligand_path = ligand_path or target_ligand_path(choice)
    if show_ligand and explicit_ligand_path is not None and explicit_ligand_path.is_file():
        ligand_data = (
            transformed_ligand_sdf(
                explicit_ligand_path, coordinate_transform
            )
            if coordinate_transform is not None
            else explicit_ligand_path.read_text(errors="replace")
        )
        if ligand_data.strip():
            viewer.addModel(ligand_data, "sdf")
            viewer.setStyle(
                {"model": -1},
                {
                    "stick": {"colorscheme": ligand_color, "radius": 0.22},
                    "sphere": {"colorscheme": ligand_color, "scale": 0.18},
                },
            )
    effective_box = box if show_box else None
    if show_box and effective_box is None:
        metadata_center = choice.job.metadata.get("center")
        metadata_size = choice.job.metadata.get("size")
        if metadata_center and metadata_size:
            effective_box = artifact_box(
                ArtifactChoice(
                    job=choice.job,
                    artifact=ArtifactRef(
                        run_id=choice.job.run_id,
                        artifact_type="box",
                        path=choice.artifact.path,
                        metadata={"center": metadata_center, "size": metadata_size},
                    ),
                )
            )
    if effective_box:
        center = effective_box["center"]
        size = effective_box["size"]
        cx, cy, cz = center
        hx, hy, hz = (value / 2.0 for value in size)
        corners = (
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz),
            (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz),
            (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz),
            (cx - hx, cy + hy, cz + hz),
        )
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for x, y, z in corners:
            viewer.addSphere(
                {
                    "center": {"x": x, "y": y, "z": z},
                    "radius": 0.45,
                    "color": "#d9f2ff",
                    "opacity": 0.95,
                }
            )
        for start_index, end_index in edges:
            start = corners[start_index]
            end = corners[end_index]
            viewer.addCylinder(
                {
                    "start": {"x": start[0], "y": start[1], "z": start[2]},
                    "end": {"x": end[0], "y": end[1], "z": end[2]},
                    "radius": 0.10,
                    "color": box_color,
                    "fromCap": 1,
                    "toCap": 1,
                }
            )
        if show_box_center:
            viewer.addSphere(
                {
                    "center": {"x": center[0], "y": center[1], "z": center[2]},
                    "radius": 0.45,
                    "color": "#dc2626",
                }
            )
            viewer.addLabel(
                f"Box center: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}",
                {
                    "position": {"x": center[0], "y": center[1], "z": center[2]},
                    "fontColor": "#334155",
                    "backgroundColor": "white",
                    "backgroundOpacity": 0.72,
                    "fontSize": 11,
                },
            )
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    render_persistent_3dmol(
        viewer,
        key=(
            f"{key}:{choice.job.run_id}:"
            f"{choice.artifact.artifact_id}"
        ),
        height=height,
    )


@st.cache_data(show_spinner=False, ttl=30)
def _cached_artifact_options(
    artifact_types: tuple[str, ...],
    *,
    source_run_id: str = "",
    runs_dir_text: str,
) -> dict[str, ArtifactChoice]:
    options: dict[str, ArtifactChoice] = {}
    jobs = list(job_records_snapshot(runs_dir_text))
    generation_runs_with_qualification = {
        job.parent_run_id
        for job in jobs
        if job.workflow == "molecule_qualification"
        and job.parent_run_id
    }
    latest_qualification_by_parent: dict[str, JobRecord] = {}
    for candidate in jobs:
        if (
            candidate.workflow != "molecule_qualification"
            or not candidate.parent_run_id
        ):
            continue
        previous = latest_qualification_by_parent.get(candidate.parent_run_id)
        if previous is None or (
            int(candidate.metadata.get("qualification_policy_version") or 0),
            str(candidate.created_at or ""),
        ) > (
            int(previous.metadata.get("qualification_policy_version") or 0),
            str(previous.created_at or ""),
        ):
            latest_qualification_by_parent[candidate.parent_run_id] = candidate
    for job in jobs:
        if job.status != "completed" or job.artifact_manifest is None:
            continue
        if source_run_id and "pocket" in artifact_types:
            linked_target = str(
                job.metadata.get("prepared_target_run_id")
                or job.metadata.get("parent_run_id")
                or job.parent_run_id
                or ""
            )
            if linked_target != source_run_id:
                continue
        for artifact in job.artifact_manifest.artifacts:
            if artifact.artifact_type not in artifact_types:
                continue
            if (
                artifact.artifact_type == "compound_set"
                and job.workflow == "molecule_generation"
                and job.run_id in generation_runs_with_qualification
            ):
                continue
            if (
                artifact.artifact_type == "compound_set"
                and job.workflow == "molecule_qualification"
                and job.parent_run_id
                and latest_qualification_by_parent[job.parent_run_id].run_id
                != job.run_id
            ):
                continue
            if artifact.resolve(job.run_dir, must_exist=True) is None:
                continue
            code = display_job_code(job.metadata.get("job_code"), job.run_id)
            if artifact.artifact_type == "pharmacophore_hypothesis":
                saved_name = str(job.metadata.get("name") or "").strip()
                label = (
                    f"{saved_name or 'Unnamed pharmacophore hypothesis'} "
                    f"— {code}"
                )
            else:
                label = (
                    f"{code} | {artifact.artifact_type} | "
                    f"{artifact.label or artifact.role or artifact.artifact_id}"
                )
            options[label] = ArtifactChoice(job=job, artifact=artifact)
    return options


def artifact_options(
    artifact_types: tuple[str, ...], *, source_run_id: str = ""
) -> dict[str, ArtifactChoice]:
    return _cached_artifact_options(
        artifact_types,
        source_run_id=source_run_id,
        runs_dir_text=str(runs_root().resolve()),
    )


def select_artifact(
    label: str,
    artifact_types: tuple[str, ...],
    *,
    key: str,
    required: bool = True,
    source_run_id: str = "",
) -> ArtifactChoice | None:
    options = artifact_options(artifact_types, source_run_id=source_run_id)
    labels = list(options)
    if not required:
        labels = ["None", *labels]
    if not options and required:
        st.selectbox(label, ["No compatible prepared artifact"], disabled=True, key=f"{key}_missing")
        return None
    selected_label = st.selectbox(label, labels, key=key)
    return None if selected_label == "None" else options[selected_label]


def select_artifacts(
    label: str,
    artifact_types: tuple[str, ...],
    *,
    key: str,
    required: bool = True,
) -> tuple[ArtifactChoice, ...]:
    options = artifact_options(artifact_types)
    if not options:
        st.multiselect(label, ["No compatible prepared artifact"], disabled=True, key=f"{key}_missing")
        return ()
    defaults = [next(iter(options))] if required else []
    selected_labels = st.multiselect(label, list(options), default=defaults, key=key)
    return tuple(options[label] for label in selected_labels)


def render_selected_artifacts(selected: dict[str, ArtifactChoice | tuple[ArtifactChoice, ...]]) -> None:
    rows = []
    for label, value in selected.items():
        choices = value if isinstance(value, tuple) else (value,)
        for choice in choices:
            rows.append(
                {
                    "input": label,
                    "type": choice.artifact.artifact_type,
                    "job": display_job_code(choice.job.metadata.get("job_code"), choice.job.run_id),
                    "artifact": choice.artifact.label or choice.artifact.path,
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_discover_job(
    *,
    title: str,
    tools: tuple[str, ...],
    inputs: tuple[InputSpec, ...],
    task_key: str,
) -> None:
    st.title(title)
    input_tab, tool_tab, run_tab, results_tab = st.tabs(
        ["Target / Input", "Tool / Engine", "Run", "Results"]
    )
    selected: dict[str, ArtifactChoice] = {}
    missing: list[str] = []

    with tool_tab:
        tool = st.selectbox("Tool", tools, key=f"{task_key}_tool")
        st.caption(
            "Tool-specific protocol controls will appear here when the typed adapter is enabled."
        )

    with input_tab:
        pocket_choice: ArtifactChoice | None = None
        target_choice: ArtifactChoice | None = None
        for spec in inputs:
            if spec.artifact_types and all(
                artifact_type in {"prepared_target", "prepared_receptor", "prepared_complex"}
                for artifact_type in spec.artifact_types
            ):
                target_choice = select_target_artifact(
                    spec.label,
                    spec.artifact_types,
                    key=f"{task_key}_target_inventory",
                    show_viewer=False,
                )
                if target_choice is None and spec.required:
                    missing.append(spec.label)
                elif target_choice is not None:
                    selected[spec.label] = target_choice
                continue
            options = artifact_options(
                spec.artifact_types,
                source_run_id=target_choice.job.run_id
                if target_choice is not None and "pocket" in spec.artifact_types
                else "",
            )
            labels = list(options)
            if not spec.required:
                labels = ["None", *labels]
            if not options and spec.required:
                st.selectbox(spec.label, ["No compatible prepared artifact"], disabled=True)
                missing.append(spec.label)
                continue
            selected_label = st.selectbox(
                spec.label,
                labels,
                key=f"{task_key}_{spec.label.lower().replace(' ', '_')}",
            )
            if selected_label != "None":
                selected[spec.label] = options[selected_label]
                if "pocket" in spec.artifact_types:
                    pocket_choice = options[selected_label]

        if target_choice is not None:
            render_target_viewer(
                target_choice,
                viewer_path=target_viewer_path(target_choice),
                box=artifact_box(pocket_choice),
                key=f"{task_key}_target_viewer",
            )

    with run_tab:
        render_run_resources(
            requires_gpu=tool not in {"Vina campaign", "RosettaLigand campaign"},
            selected_gpu="Automatic",
            key=task_key,
        )
        if selected:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "input": label,
                            "type": choice.artifact.artifact_type,
                            "job": display_job_code(
                                choice.job.metadata.get("job_code"), choice.job.run_id
                            ),
                            "artifact": choice.artifact.label or choice.artifact.path,
                        }
                        for label, choice in selected.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        if missing:
            st.info("Prepare the required inputs before running this workflow.")
            st.link_button(
                "Open Structure Import", "./workspace-structure-preparation"
            )
        st.button(
            f"Run {tool}",
            type="primary",
            disabled=True,
            help=(
                "This route is artifact-only. Execution will be enabled when its "
                "typed Docker adapter is migrated."
            ),
            key=f"{task_key}_run",
        )

    with results_tab:
        st.info("No typed runs are available for this adapter yet.")
