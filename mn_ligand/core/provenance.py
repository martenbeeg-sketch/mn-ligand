from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

from mn_ligand.core.jobs import JobRecord, display_job_code


LINEAGE_METADATA_KEYS = (
    "import_run_id",
    "source_structure_run_id",
    "source_target_run_id",
    "structure_run_id",
    "parent_run_id",
    "workflow_parent_run_id",
)

BENCHMARK_METADATA_KEYS = (
    "benchmark_dataset_run_id",
    "benchmark_campaign_id",
    "benchmark_case_id",
    "benchmark_replicate",
)
TARGET_LIGAND_CAMPAIGN_PURPOSE = "target_ligand_redocking_refolding"
COMPOUND_DATASET_CAMPAIGN_PURPOSE = "compound_dataset_docking_cofolding"

TARGET_PREPARATION_STEP_CODES = {
    "modeller_residue_repair": "MRR",
    "modeller_gap_modeling": "MGP",
    "modified_residue_mapping": "MRM",
    "pdbfixer_cleaning": "PFX",
    "openmm_minimization": "OMM",
    "target_trimming": "TRM",
    "terminal_repair": "CTE",
    "prediction_promotion": "PRM",
    "target_orientation": "ORI",
    "docking": "DCK",
    "ligand_preparation": "LGP",
    "receptor_preparation": "RCP",
}


def _fallback_preparation_step_code(kind: object) -> str:
    words = [
        word
        for word in str(kind or "").strip().lower().replace("-", "_").split("_")
        if word
    ]
    if not words:
        return "UNK"
    if len(words) == 1:
        return words[0][:3].upper().ljust(3, "X")
    return (words[0][0] + words[-1][:2]).upper().ljust(3, "X")


def compact_target_identifier(
    *,
    run_id: str,
    metadata: Mapping[str, Any],
    fallback_origin: str = "",
) -> str:
    """Return a human-readable, provenance-derived prepared-target identifier."""
    origin = str(
        metadata.get("pdb_id")
        or metadata.get("target")
        or metadata.get("target_name")
        or fallback_origin
        or "TARGET"
    ).strip()
    if "." in origin or "/" in origin or "\\" in origin:
        origin = "TARGET"
    history = metadata.get("modification_history")
    history = history if isinstance(history, list) else []
    step_codes = [
        TARGET_PREPARATION_STEP_CODES.get(
            str(item.get("kind") or ""),
            _fallback_preparation_step_code(item.get("kind")),
        )
        for item in history
        if isinstance(item, Mapping) and str(item.get("kind") or "").strip()
    ]
    signature = "-".join(step_codes) or "RAW"
    job_code = display_job_code(metadata.get("job_code"), run_id)
    return f"{origin.upper()} · {signature} · {job_code}"


def target_key(
    *,
    run_id: str,
    metadata: Mapping[str, Any],
    fallback_origin: str = "",
) -> str:
    """Return the selector-compatible `target/PDB · short job code` key."""
    identifier = compact_target_identifier(
        run_id=run_id,
        metadata=metadata,
        fallback_origin=fallback_origin,
    )
    parts = [part.strip() for part in identifier.split("·") if part.strip()]
    return (
        f"{parts[0]} · {parts[-1]}"
        if len(parts) >= 2
        else identifier
    )


def binding_campaign_purpose(job: JobRecord) -> str:
    """Return the immutable campaign purpose, inferring legacy campaign jobs."""
    explicit = str(job.metadata.get("campaign_purpose") or "").strip()
    if explicit:
        return explicit
    workflow = str(job.workflow or "").lower()
    operation = str(job.metadata.get("operation") or "").lower()
    if workflow == "docking_redocking" or operation == "redocking":
        return TARGET_LIGAND_CAMPAIGN_PURPOSE
    if operation in {"docking", "refolding"} or workflow in {
        "docking_campaign",
        "openvs_docking",
        "alphafold3_refolding",
        "boltz2_refolding",
        "nesso_affinity",
    }:
        return COMPOUND_DATASET_CAMPAIGN_PURPOSE
    return ""


def _has_benchmark_marker(job: JobRecord) -> bool:
    if job.task_group in {"benchmark-datasets", "benchmark-targets"}:
        return True
    if job.workflow in {"benchmark_dataset_import", "redocking_benchmark"}:
        return True
    if str(job.metadata.get("job_type") or "").startswith("benchmark_"):
        return True
    if str(job.metadata.get("operation") or "").startswith("benchmark_"):
        return True
    if any(job.metadata.get(key) not in (None, "") for key in BENCHMARK_METADATA_KEYS):
        return True
    parameters = job.metadata.get("parameters")
    context = parameters.get("context") if isinstance(parameters, dict) else None
    return isinstance(context, dict) and any(
        context.get(key) not in (None, "") for key in BENCHMARK_METADATA_KEYS
    )


def is_benchmark_job(
    job: JobRecord,
    jobs_by_id: Mapping[str, JobRecord] | None = None,
) -> bool:
    """Return whether a job belongs to a benchmark dataset or campaign."""
    candidates = (
        iter_job_lineage(job, jobs_by_id)
        if jobs_by_id is not None
        else (job,)
    )
    return any(_has_benchmark_marker(candidate) for candidate in candidates)


def iter_job_lineage(
    job: JobRecord,
    jobs_by_id: Mapping[str, JobRecord],
) -> Iterator[JobRecord]:
    """Yield a job and every reachable ancestor once, nearest first."""
    pending = [job]
    visited: set[str] = set()
    while pending:
        current = pending.pop(0)
        if current.run_id in visited:
            continue
        visited.add(current.run_id)
        yield current
        candidate_ids = [
            str(current.metadata.get(key) or "")
            for key in LINEAGE_METADATA_KEYS
        ]
        candidate_ids.extend(
            [current.parent_run_id, current.workflow_parent_run_id]
        )
        for candidate_id in candidate_ids:
            ancestor = jobs_by_id.get(candidate_id)
            if ancestor is not None and ancestor.run_id not in visited:
                pending.append(ancestor)


def ordered_job_lineage(
    job: JobRecord,
    jobs_by_id: Mapping[str, JobRecord],
) -> list[JobRecord]:
    """Return reachable ancestors before their dependent descendants."""
    ordered: list[JobRecord] = []
    visited: set[str] = set()

    def visit(current: JobRecord) -> None:
        if current.run_id in visited:
            return
        visited.add(current.run_id)
        candidate_ids = [
            str(current.metadata.get(key) or "")
            for key in LINEAGE_METADATA_KEYS
        ]
        candidate_ids.extend([current.parent_run_id, current.workflow_parent_run_id])
        for candidate_id in candidate_ids:
            ancestor = jobs_by_id.get(candidate_id)
            if ancestor is not None:
                visit(ancestor)
        ordered.append(current)

    visit(job)
    return ordered


def inherited_metadata_value(
    job: JobRecord,
    jobs_by_id: Mapping[str, JobRecord],
    key: str,
    *,
    default: Any = None,
) -> Any:
    for candidate in iter_job_lineage(job, jobs_by_id):
        value = candidate.metadata.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def inherited_target_metadata(
    job: JobRecord,
    jobs_by_id: Mapping[str, JobRecord],
) -> dict[str, Any]:
    """Build a stable target-identity snapshot for a derived job."""
    snapshot: dict[str, Any] = {}
    for key in (
        "pdb_id",
        "source",
        "ligand_key",
        "ligand_id",
        "ligand_smiles",
        "ligand_count",
        "protein_chains",
        "receptor",
        "ligands",
    ):
        value = inherited_metadata_value(job, jobs_by_id, key)
        if value not in (None, "", [], {}):
            snapshot[key] = value
    return snapshot


def _source_label(value: object) -> str:
    source = str(value or "").strip().lower()
    if source.startswith("pdb") or source.startswith("rcsb"):
        return "PDB"
    if any(token in source for token in ("upload", "custom", "file")):
        return "Custom file"
    if any(token in source for token in ("vina", "gnina", "unidock", "openvs", "dock")):
        return "Docking"
    if any(token in source for token in ("boltz", "alphafold", "protenix", "predict")):
        return "Prediction"
    return str(value or "").strip().replace("_", " ").title()


def _job_json(job: JobRecord, relative_path: str) -> dict[str, Any]:
    try:
        payload = json.loads((job.run_dir / relative_path).read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _history_item(
    candidate: JobRecord,
    *,
    kind: str,
    label: str,
    summary: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": candidate.run_id,
        "job_code": display_job_code(
            candidate.metadata.get("job_code"), candidate.run_id
        ),
        "kind": kind,
        "label": label,
        "tool": candidate.tool,
        "summary": summary,
        **({"details": details} if details else {}),
    }


def _protein_cleaning_history(candidate: JobRecord) -> list[dict[str, Any]]:
    input_payload = _job_json(candidate, "input.json")
    parameters = (
        dict(input_payload.get("parameters") or {})
        if isinstance(input_payload.get("parameters"), dict)
        else {}
    )
    report = _job_json(candidate, "artifacts/reports/repair_report.json")
    native = _job_json(candidate, "native_result.json")
    noncanonical = list(
        parameters.get("noncanonical_replacements")
        or (report.get("modeller_noncanonical_repair") or {}).get("replacements")
        or []
    )
    gap_report = dict(report.get("modeller_internal_gap_repair") or {})
    modeled_gaps = list(
        gap_report.get("modeled_gaps")
        or parameters.get("internal_gap_definitions")
        or []
    )
    pdbfixer = dict(
        report.get("pdbfixer")
        or native.get("repair_report")
        or {}
    )
    history: list[dict[str, Any]] = []
    if noncanonical:
        replacements = [
            {
                "site": str(item.get("key") or ""),
                "target": str(item.get("target") or ""),
            }
            for item in noncanonical
            if isinstance(item, dict)
        ]
        history.append(
            _history_item(
                candidate,
                kind="modeller_residue_repair",
                label="MODELLER residue repair",
                summary=f"MODELLER repaired {len(replacements)} noncanonical residue site(s)",
                details={"replacements": replacements},
            )
        )
    if modeled_gaps:
        gaps = [
            {
                "chain": str(item.get("chain") or "_"),
                "start": item.get("author_start"),
                "end": item.get("author_end"),
                "sequence": str(item.get("sequence") or ""),
                "length": item.get("length") or len(str(item.get("sequence") or "")),
            }
            for item in modeled_gaps
            if isinstance(item, dict)
        ]
        history.append(
            _history_item(
                candidate,
                kind="modeller_gap_modeling",
                label="MODELLER gap modeling",
                summary=f"MODELLER modeled {len(gaps)} internal gap(s)",
                details={
                    "model_count": parameters.get("internal_gap_model_count"),
                    "gaps": gaps,
                },
            )
        )
    if parameters.get("map_modified_residues"):
        history.append(
            _history_item(
                candidate,
                kind="modified_residue_mapping",
                label="Modified-residue mapping",
                summary="Mapped declared modified residues to canonical identities",
            )
        )
    history.append(
        _history_item(
            candidate,
            kind="pdbfixer_cleaning",
            label="PDBFixer cleaning",
            summary=(
                "PDBFixer cleaned/repaired the protein"
                + (
                    f" and added hydrogens at pH {parameters.get('ph')}"
                    if parameters.get("ph") not in (None, "")
                    else ""
                )
            ),
            details={
                "missing_segments_added": list(
                    pdbfixer.get("missing_residue_segments_added") or []
                ),
                "missing_atoms": list(pdbfixer.get("missing_atoms") or []),
            },
        )
    )
    refinement = dict(pdbfixer.get("refinement") or {})
    if refinement:
        history.append(
            _history_item(
                candidate,
                kind="openmm_minimization",
                label="OpenMM minimization",
                summary="OpenMM locally minimized the prepared protein-ligand system",
                details={
                    "engine": refinement.get("engine"),
                    "forcefield": refinement.get("forcefield"),
                    "ligand_context": refinement.get("ligand_context"),
                    "potential_energy_before_kj_mol": refinement.get(
                        "potential_energy_before_kj_mol"
                    ),
                    "potential_energy_after_kj_mol": refinement.get(
                        "potential_energy_after_kj_mol"
                    ),
                },
            )
        )
    return history


def _structure_job_history(candidate: JobRecord) -> list[dict[str, Any]]:
    source = str(
        candidate.metadata.get("engine")
        or candidate.metadata.get("source")
        or candidate.tool
        or ""
    ).strip()
    normalized = source.lower()
    history: list[dict[str, Any]] = []
    legacy = candidate.metadata.get("legacy_preparation_evidence")
    if isinstance(legacy, dict):
        mapping_report = legacy.get("modified_residue_mapping")
        mappings = (
            mapping_report.get("mappings")
            if isinstance(mapping_report, dict)
            and isinstance(mapping_report.get("mappings"), dict)
            else {}
        )
        if mappings:
            replacements = [
                {
                    "site": str(site),
                    "target": str(details.get("target") or ""),
                }
                for site, details in mappings.items()
                if isinstance(details, dict)
            ]
            history.append(
                _history_item(
                    candidate,
                    kind="modified_residue_mapping",
                    label="Modified-residue mapping",
                    summary=(
                        f"Mapped {len(replacements)} modified residue site(s) "
                        "to canonical identities"
                    ),
                    details={"replacements": replacements},
                )
            )
        if legacy.get("clean_protein") and legacy.get("protein_cleaned"):
            history.append(
                _history_item(
                    candidate,
                    kind="pdbfixer_cleaning",
                    label="PDBFixer cleaning",
                    summary=(
                        "Legacy Ligand-X preparation removed water, repaired missing "
                        "atoms, and added hydrogens with PDBFixer"
                    ),
                    details={
                        "protocol": legacy.get("protocol"),
                        "ph": legacy.get("ph"),
                        "components": legacy.get("components") or {},
                        "evidence_run_id": legacy.get("source_run_id"),
                    },
                )
            )
    if any(
        token in normalized
        for token in ("vina", "gnina", "unidock", "uni-dock", "openvs", "dock")
    ):
        if "gnina" in normalized:
            engine = "GNINA"
        elif "unidock" in normalized or "uni-dock" in normalized:
            engine = "Uni-Dock Pro"
        elif "openvs" in normalized:
            engine = "OpenVS"
        elif "vina" in normalized:
            engine = "AutoDock Vina"
        else:
            engine = source or "Docking"
        history.append(
            _history_item(
                candidate,
                kind="docking",
                label=f"{engine} docking",
                summary=f"Generated the protein-ligand pose with {engine}",
            )
        )
    if candidate.metadata.get("use_scrub"):
        history.append(
            _history_item(
                candidate,
                kind="ligand_preparation",
                label="Ligand preparation",
                summary=(
                    "Prepared ligand protonation/tautomer and 3D coordinates"
                    + (
                        f" at pH {candidate.metadata.get('scrub_ph')}"
                        if candidate.metadata.get("scrub_ph") not in (None, "")
                        else ""
                    )
                ),
            )
        )
    repair = candidate.metadata.get("receptor_meeko_repair")
    if isinstance(repair, dict) and repair.get("repair_applied"):
        history.append(
            _history_item(
                candidate,
                kind="receptor_preparation",
                label="Receptor preparation",
                summary="Applied receptor compatibility repair before docking",
                details=repair,
            )
        )
    return history


def modification_history(
    job: JobRecord,
    jobs_by_id: Mapping[str, JobRecord],
) -> list[dict[str, Any]]:
    """Describe structure-changing jobs from oldest ancestor to current job."""
    history: list[dict[str, Any]] = []
    for candidate in ordered_job_lineage(job, jobs_by_id):
        kind = ""
        summary = ""
        label = ""
        if candidate.job_type == "protein_cleaning":
            history.extend(_protein_cleaning_history(candidate))
            continue
        if candidate.task_group == "structure-jobs":
            history.extend(_structure_job_history(candidate))
        if candidate.job_type == "target_trimming":
            kind = "target_trimming"
            label = "Target trimming"
            ranges = candidate.metadata.get("trim_ranges") or {}
            range_text = ", ".join(
                f"{chain}:{bounds.get('start')}-{bounds.get('end')}"
                for chain, bounds in ranges.items()
                if isinstance(bounds, dict)
            )
            summary = f"Trimmed protein termini {range_text}".strip()
        elif candidate.job_type == "terminal_repair":
            kind = "terminal_repair"
            label = "MODELLER repair"
            sequence = str(candidate.metadata.get("extension_sequence") or "")
            chain = str(candidate.metadata.get("chain") or "")
            summary = (
                f"Extended chain {chain} C-terminus by {len(sequence)} aa"
                + (f" ({sequence})" if sequence else "")
            )
        elif candidate.job_type == "energy_minimization":
            kind = "openmm_minimization"
            label = "OpenMM minimization"
            summary = "OpenMM minimized the existing prepared protein-ligand complex"
        elif candidate.job_type in {
            "predicted_complex_promotion",
            "prediction_promotion",
        }:
            kind = "prediction_promotion"
            label = "Prediction promotion"
            summary = "Promoted predicted complex"
        if not kind:
            continue
        history.append(
            {
                "run_id": candidate.run_id,
                "job_code": display_job_code(
                    candidate.metadata.get("job_code"), candidate.run_id
                ),
                "kind": kind,
                "label": label,
                "tool": candidate.tool,
                "summary": summary,
            }
        )
    return history


def provenance_origin_label(
    job: JobRecord,
    jobs_by_id: Mapping[str, JobRecord],
) -> str:
    lineage = ordered_job_lineage(job, jobs_by_id)
    source = ""
    root_sources = [
        candidate
        for candidate in lineage
        if candidate.job_type == "protein_import"
        or candidate.task_group == "protein-import"
    ]
    candidates = root_sources or [
        candidate
        for candidate in lineage
        if candidate.job_type
        not in {"protein_cleaning", "target_trimming", "terminal_repair"}
    ]
    for candidate in candidates:
        label = _source_label(candidate.metadata.get("source") or candidate.tool)
        if label:
            source = label
            break
    if not source:
        source = _source_label(job.tool) or "Unknown"
    labels = [item["label"] for item in modification_history(job, jobs_by_id)]
    return " → ".join(dict.fromkeys([source, *labels]))


def provenance_last_step(
    job: JobRecord,
    jobs_by_id: Mapping[str, JobRecord],
) -> str:
    """Return the most recent target-producing step in a job's lineage."""
    history = modification_history(job, jobs_by_id)
    if history:
        return str(history[-1].get("label") or "Unknown")
    origin = provenance_origin_label(job, jobs_by_id)
    return origin.rsplit(" → ", 1)[-1] if origin else "Unknown"


def target_lineage_summary(
    job: JobRecord,
    jobs_by_id: Mapping[str, JobRecord],
) -> dict[str, str]:
    """Return compact target identity and provenance fields for result tables."""
    receptor = inherited_metadata_value(job, jobs_by_id, "receptor", default={})
    receptor_name = inherited_metadata_value(
        job, jobs_by_id, "receptor_name", default=""
    )
    organism = inherited_metadata_value(job, jobs_by_id, "organism", default="")
    if isinstance(receptor, Mapping):
        entities = receptor.get("entities")
        first_entity = (
            entities[0]
            if isinstance(entities, list)
            and entities
            and isinstance(entities[0], Mapping)
            else {}
        )
        receptor_name = receptor_name or first_entity.get("name") or receptor.get("title")
        organisms = first_entity.get("source_organisms")
        if not organism and isinstance(organisms, list):
            organism = ", ".join(str(value) for value in organisms if value)
    elif receptor and not receptor_name:
        receptor_name = receptor

    target = (
        inherited_metadata_value(job, jobs_by_id, "pdb_id", default="")
        or inherited_metadata_value(job, jobs_by_id, "target", default="")
        or inherited_metadata_value(job, jobs_by_id, "target_name", default="")
    )
    ligand = (
        inherited_metadata_value(job, jobs_by_id, "ligand_key", default="")
        or inherited_metadata_value(job, jobs_by_id, "bound_ligand_key", default="")
        or inherited_metadata_value(job, jobs_by_id, "ligand_id", default="")
    )
    origin = provenance_origin_label(job, jobs_by_id)
    return {
        "last_step": provenance_last_step(job, jobs_by_id),
        "origin": origin,
        "target": str(target or "—"),
        "receptor": str(receptor_name or "—"),
        "organism": str(organism or "—"),
        "ligand": str(ligand or "—"),
    }
