from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any
from uuid import uuid4

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    registered_tool,
    write_registered_command_record,
)
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.core.portability import JOB_PORTABILITY_SCHEMA_VERSION, assert_job_portable
from mn_ligand.core.provenance import compact_target_identifier, target_key
from mn_ligand.core.residue_mapping import (
    derive_residue_mapping,
    relabel_structural_dynamics,
    residue_mapping_artifact,
    subset_residue_mapping,
)
from mn_ligand.core.workflows import (
    WorkflowRecord,
    add_workflow_input,
    attach_workflow_child,
    create_workflow,
    refresh_workflow,
    update_workflow_definition,
)
from mn_ligand.runtime import (
    PROJECT_DIR,
    cpu_process_limit,
    resolve_run_dir,
    runs_root,
)
from mn_ligand.workflows.md_engines import (
    GROMACS_ENGINE,
    OPENMM_ENGINE,
    endpoint_backend_supported,
    engine_restart_artifact_types,
    md_engine_spec,
    normalize_md_engine,
)
from mn_ligand.workflows.bound_ligand_md import parse_bound_ligands


MD_WORKFLOW_TYPE = "md-simulation"
DEFAULT_MD_IMAGE = "ovolig-md-cu128:latest"
DEFAULT_GROMACS_MD_IMAGE = "ovolig-gromacs-cu128:latest"
EXACT_CONTINUATION = "exact_checkpoint"
INDEPENDENT_REPLICA = "independent_replica"
HISTORICAL_MD_WORKFLOW_STATES = frozenset(
    {"failed", "blocked", "cancelled", "superseded"}
)
MD_TARGET_METADATA_KEYS = (
    "target_key",
    "target_run_id",
    "target_provenance_key",
)

_RESIDUE_LINEAGE_KEYS = (
    "import_run_id",
    "source_structure_run_id",
    "source_target_run_id",
    "structure_run_id",
    "parent_run_id",
)
SYSTEM_PARAMETER_KEYS = (
    "forcefield_method",
    "protein_forcefield_method",
    "water_model",
    "charge_method",
    "box_shape",
    "padding_nm",
    "ionic_strength",
    "constraints",
)
EQUILIBRATION_PARAMETER_KEYS = (
    "preparation_protocol",
    "temperature",
    "pressure",
    "heating_steps_per_stage",
    "heating_stages",
    "nvt_steps",
    "npt_steps",
    "apply_protein_restraints_during_heating_nvt",
    "protein_restraint_selection",
    "protein_restraint_k",
    "ligand_restraints_enabled",
    "ligand_lock_k_kjmol_nm2",
    "npt_restraint_release_scales",
    "density_stabilization_min_ns",
    "density_stabilization_max_ns",
    "density_stabilization_increment_ns",
    "density_sample_interval_ps",
    "density_plateau_required",
    "integration_profile",
    "production_timestep_fs",
    "hydrogen_mass_amu",
    "mass_repartition_factor",
)

_PREP_ARTIFACTS = {
    "system_pdb": ("md_system", "topology"),
    "npt_pdb": ("equilibrated_system", "coordinates"),
    "npt_checkpoint": ("md_checkpoint", "checkpoint"),
    "npt_state_xml": ("openmm_state", "state"),
    "npt_system_xml": ("openmm_system", "system"),
    "npt_integrator_xml": ("openmm_integrator", "integrator"),
    "nvt_trajectory": ("equilibration_trajectory", "nvt"),
    "npt_trajectory": ("equilibration_trajectory", "npt"),
    "density_report": ("md_equilibration_report", "density"),
    "density_series": ("md_equilibration_report", "density-series"),
    "gromacs_topology": ("md_topology", "gromacs-topology"),
    "gromacs_coordinates": ("equilibrated_system", "gromacs-coordinates"),
    "gromacs_checkpoint": ("md_checkpoint", "gromacs-checkpoint"),
    "gromacs_index": ("md_index", "gromacs-index"),
    "gromacs_tpr": ("md_run_input", "gromacs-tpr"),
    "protocol_report": ("md_protocol_report", "preparation"),
}


def visible_md_workflow_rows(
    rows: list[dict[str, Any]],
    *,
    show_history: bool,
) -> list[dict[str, Any]]:
    if show_history:
        return list(rows)
    return [
        row
        for row in rows
        if str(row.get("status") or "").lower()
        not in HISTORICAL_MD_WORKFLOW_STATES
    ]


def _validate_roe_preparation_result(prep_result: dict[str, Any]) -> None:
    md_result = (
        prep_result.get("md_result")
        if isinstance(prep_result.get("md_result"), dict)
        else {}
    )
    equilibration = (
        md_result.get("equilibration_stats")
        if isinstance(md_result.get("equilibration_stats"), dict)
        else {}
    )
    protocol = (
        md_result.get("preparation_protocol")
        if isinstance(md_result.get("preparation_protocol"), dict)
        else equilibration.get("preparation_protocol")
        if isinstance(equilibration.get("preparation_protocol"), dict)
        else {}
    )
    if str(protocol.get("protocol") or "").strip().lower() != "roe_brooks_2020":
        raise ValueError(
            "Prepared-system reuse requires a successful Roe-Brooks 2020 "
            "preparation result"
        )
    density = (
        protocol.get("density_stabilization")
        if isinstance(protocol.get("density_stabilization"), dict)
        else {}
    )
    density_fit = (
        density.get("fit")
        if isinstance(density.get("fit"), dict)
        else {}
    )
    if density_fit.get("plateau") is not True:
        raise ValueError(
            "Prepared-system reuse requires a passed Roe-Brooks density plateau"
        )
_PRODUCTION_ARTIFACTS = {
    "production_trajectory": ("md_trajectory", "production"),
    "native_production_trajectory": ("native_output", "gromacs-trajectory"),
    "production_pdb": ("md_final_structure", "coordinates"),
    "production_checkpoint": ("md_checkpoint", "checkpoint"),
    "production_topology": ("md_topology", "gromacs-topology"),
    "production_tpr": ("md_run_input", "gromacs-tpr"),
    "production_index": ("md_index", "gromacs-index"),
    "thermodynamic_series": ("md_thermodynamic_series", "production"),
    "analysis_report": ("md_trajectory_analysis", "production"),
    "replica_density_series": ("md_equilibration_report", "replica-density-series"),
    "replica_density_report": ("md_equilibration_report", "replica-density"),
}
MMGBSA_TASK_GROUP = "md-mmgbsa"

_TEMPLATE_SOURCE_PREP_KEYS = frozenset(
    {
        "job_id",
        "pdb_id",
        "pdb_data",
        "ligand_key",
        "prepared_complex_path",
        "input_complex_pdb_path",
        "ligand_refined_sdf_data",
        "ligand_refined_sdf_path",
        "residue_mapping",
        "residue_mapping_path",
        "source_md_system_prep_result_json",
        "use_gpu",
        "md_engine",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def _exclusive_file_lock(path: Path):
    """Serialize orchestration/finalization across durable worker processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _runtime_job_by_id(run_id: str) -> JobRecord | None:
    """Resolve a lineage job without assuming its task group."""
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        return None
    root = runs_root(create=False)
    if not root.is_dir():
        return None
    for group_dir in root.iterdir():
        candidate = group_dir / run_id
        if group_dir.is_dir() and candidate.is_dir():
            return JobRecord.load(candidate, task_group=group_dir.name)
    return None


def source_author_residue_mapping(
    source_job: JobRecord,
    source_artifact: ArtifactRef,
    structure_pdb_data: str,
) -> dict[str, Any]:
    """Map selected coordinates back to the earliest author-numbered source."""
    pending = [source_job]
    visited: set[str] = set()
    imported_job: JobRecord | None = None
    imported_path: Path | None = None
    while pending:
        current = pending.pop(0)
        if current.run_id in visited:
            continue
        visited.add(current.run_id)
        mapping_path = residue_mapping_artifact(current)
        if mapping_path is not None:
            candidate_mapping = subset_residue_mapping(
                _read_json(mapping_path),
                structure_pdb_data,
            )
            candidate_rows = [
                row
                for row in candidate_mapping.get("residues") or []
                if isinstance(row, dict)
            ]
            mapped_rows = [
                row
                for row in candidate_rows
                if row.get("mapping_method") != "unmapped-identity-fallback"
            ]
            identity_matches = sum(
                str(row.get("structure_residue_name") or "").upper()
                == str(row.get("native_residue_name") or "").upper()
                for row in mapped_rows
            )
            # Do not stop lineage traversal at a stale mapping that was
            # previously attached to a renumbered cofolded structure.  Such
            # mappings can have many numeric matches while assigning the
            # identities of unrelated residues.
            if (
                candidate_rows
                and len(mapped_rows) / len(candidate_rows) >= 0.8
                and identity_matches / max(1, len(mapped_rows)) >= 0.8
            ):
                return candidate_mapping
        if current.artifact_manifest is not None:
            imported = current.artifact_manifest.by_type("imported_target")
            if imported:
                candidate = imported[0].resolve(current.run_dir, must_exist=True)
                if candidate is not None and candidate.suffix.lower() in {".pdb", ".ent"}:
                    imported_job = current
                    imported_path = candidate
                    break
        ancestor_ids = [
            str(current.metadata.get(key) or "")
            for key in _RESIDUE_LINEAGE_KEYS
        ]
        ancestor_ids.extend([current.parent_run_id, current.workflow_parent_run_id])
        for ancestor_id in ancestor_ids:
            if ancestor_id and ancestor_id not in visited:
                ancestor = _runtime_job_by_id(ancestor_id)
                if ancestor is not None:
                    pending.append(ancestor)

    if imported_job is not None and imported_path is not None:
        return derive_residue_mapping(
            structure_pdb_data,
            imported_path.read_text(errors="replace"),
            source_run_id=imported_job.run_id,
            source_label=str(imported_job.metadata.get("pdb_id") or imported_path.name),
        )

    # With no earlier recorded lineage, this structure is itself the immutable
    # author source (for example, a promoted prediction or legacy upload).
    return derive_residue_mapping(
        structure_pdb_data,
        structure_pdb_data,
        source_run_id=source_job.run_id,
        source_label=str(
            source_job.metadata.get("pdb_id")
            or source_job.metadata.get("source_pdb_id")
            or source_artifact.label
            or source_job.run_id
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _docker_command(
    run_dir: Path,
    image: str,
    use_gpu: bool,
    *,
    prepared_system_dir: Path | None = None,
    engine: str = OPENMM_ENGINE,
) -> list[str]:
    selected_engine = normalize_md_engine(engine)
    spec = md_engine_spec(selected_engine)
    shm_size = os.getenv("MN_MD_DOCKER_SHM_SIZE", "64g").strip()
    environment = {"PYTHONPATH": "/mn-ligand"}
    if selected_engine == GROMACS_ENGINE:
        declared_threads = max(
            1,
            int(registered_tool(spec.tool_id, image=image).resources.cpu_threads),
        )
        environment["OMP_NUM_THREADS"] = str(
            min(declared_threads, cpu_process_limit())
        )
    mounts = []
    if prepared_system_dir is not None:
        mounts.append(DockerMount(prepared_system_dir, "/prepared-system", read_only=True))
    mounts.extend(
        (
            DockerMount(PROJECT_DIR, "/mn-ligand", read_only=True),
            DockerMount(run_dir, "/output"),
        )
    )
    command = (
        (
            "python",
            "-m",
            "mn_ligand.workflows.gromacs_md",
            "production" if prepared_system_dir is not None else "prepare",
            "--input",
            "/output/input.json",
            "--output",
            "/output/result.json",
        )
        if selected_engine == GROMACS_ENGINE
        else (
            "python",
            "-m",
            "mn_ligand.workflows.bound_ligand_md",
            "run",
            "--input",
            "/output/input.json",
            "--output",
            "/output/result.json",
        )
    )
    return build_docker_command(
        DockerRunSpec(
            tool=registered_tool(spec.tool_id, image=image),
            command=command,
            mounts=tuple(mounts),
            environment=environment,
            gpu_enabled=use_gpu,
            shm_size=shm_size,
            use_host_user=False,
        )
    )


def _worker_resources(
    image: str,
    use_gpu: bool,
    *,
    engine: str = OPENMM_ENGINE,
) -> dict[str, Any]:
    resources = registered_tool(
        md_engine_spec(engine).tool_id,
        image=image,
    ).resources.to_dict()
    if not use_gpu:
        resources.update({"gpu": False, "min_vram_gb": 0.0, "exclusive_gpu": False})
    return resources


def _new_child(
    task_group: str,
    *,
    status: str,
    metadata: dict[str, Any],
    input_payload: dict[str, Any],
) -> JobRecord:
    run_id = str(uuid4())
    run_dir = runs_root() / task_group / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    now = _utc_now_iso()
    _write_json(
        run_dir / "metadata.json",
        {
            "schema_version": JOB_SCHEMA_VERSION,
            "portability_schema_version": JOB_PORTABILITY_SCHEMA_VERSION,
            "run_id": run_id,
            "job_code": short_job_code(run_id),
            "job_type": task_group,
            "status": status,
            "created_at": now,
            "updated_at": now,
            **metadata,
        },
    )
    _write_json(run_dir / "input.json", input_payload)
    write_artifact_manifest(run_dir, [])
    return JobRecord.load(run_dir, task_group=task_group)


def _source_contract(source_job: JobRecord, source_artifact: ArtifactRef) -> dict[str, Any]:
    return {
        "task_group": source_job.task_group,
        "run_id": source_job.run_id,
        "artifact": source_artifact.to_dict(),
        "sha256": source_artifact.sha256,
    }


def _md_target_metadata(
    source_job: JobRecord,
    source_artifact: ArtifactRef | None = None,
) -> dict[str, str]:
    """Identify the exact structural target selected for an MD workflow."""
    fallback_origin = str(
        source_job.metadata.get("pdb_id")
        or source_job.metadata.get("source_pdb_id")
        or (source_artifact.label if source_artifact is not None else "")
        or source_job.run_id
    )
    return {
        "target_key": target_key(
            run_id=source_job.run_id,
            metadata=source_job.metadata,
            fallback_origin=fallback_origin,
        ),
        "target_run_id": source_job.run_id,
        "target_provenance_key": compact_target_identifier(
            run_id=source_job.run_id,
            metadata=source_job.metadata,
            fallback_origin=fallback_origin,
        ),
    }


def _inherited_md_target_metadata(*sources: dict[str, Any]) -> dict[str, str]:
    """Copy a complete target identity from workflow or parent metadata."""
    return {
        key: str(next((source.get(key) for source in sources if source.get(key)), ""))
        for key in MD_TARGET_METADATA_KEYS
    }


def compatibility_contract(
    source_contract: dict[str, Any], input_payload: dict[str, Any], *, engine: str = "openmm"
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "engine": engine,
        "source": source_contract,
        "system": {key: input_payload.get(key) for key in SYSTEM_PARAMETER_KEYS},
        "equilibration": {key: input_payload.get(key) for key in EQUILIBRATION_PARAMETER_KEYS},
    }


def compatibility_fingerprint(contract: dict[str, Any]) -> str:
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def compatibility_differences(existing: dict[str, Any], requested: dict[str, Any]) -> list[str]:
    differences: list[str] = []
    for section in ("engine", "source", "system", "equilibration"):
        if existing.get(section) != requested.get(section):
            differences.append(section)
    return differences


def continuation_mode(production: dict[str, Any]) -> str:
    explicit = str(production.get("continuation_mode") or "").strip()
    if explicit in {EXACT_CONTINUATION, INDEPENDENT_REPLICA}:
        return explicit
    restart_mode = str(production.get("restart_mode") or "")
    return EXACT_CONTINUATION if "checkpoint" in restart_mode.lower() else INDEPENDENT_REPLICA


def replica_seed(workflow_id: str, replica_index: int) -> int:
    digest = hashlib.sha256(f"{workflow_id}:{replica_index}".encode()).hexdigest()
    return int(digest[:8], 16) % 2147483646 + 1


def _endpoint_enabled(production: dict[str, Any]) -> bool:
    return bool(
        production.get(
            "endpoint_enabled",
            production.get("mmgbsa_enabled", False),
        )
    )


def md_prep_input_for_source(
    source_job: JobRecord,
    source_artifact: ArtifactRef,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Apply reusable MD preparation settings to a new structural source."""
    complex_path = source_artifact.resolve(source_job.run_dir, must_exist=True)
    if complex_path is None:
        raise FileNotFoundError(source_artifact.path)
    pdb_data = complex_path.read_text()
    ligands = parse_bound_ligands(pdb_data)
    preferred_key = str(source_job.metadata.get("ligand_key") or "")
    selected = next(
        (item for item in ligands if item.get("key") == preferred_key),
        ligands[0] if ligands else None,
    )
    if selected is None:
        raise ValueError("The prepared complex contains no selectable ligand")

    payload = {
        key: value
        for key, value in dict(settings).items()
        if key not in _TEMPLATE_SOURCE_PREP_KEYS
    }
    payload.update(
        {
            "pdb_id": str(source_job.metadata.get("pdb_id") or "UNKNOWN"),
            "pdb_data": pdb_data,
            "ligand_key": selected["key"],
            "production_steps": 0,
            "prepared_complex_path": str(complex_path),
        }
    )
    refined_ligands = (
        source_job.artifact_manifest.by_type("prepared_ligand_set")
        if source_job.artifact_manifest
        else ()
    )
    if refined_ligands:
        ligand_path = refined_ligands[0].resolve(
            source_job.run_dir,
            must_exist=True,
        )
        if ligand_path is not None:
            payload["ligand_refined_sdf_data"] = ligand_path.read_text()
            payload["ligand_refined_sdf_path"] = str(ligand_path)
            payload["strict_refined_ligand"] = True
    return payload


def _fresh_template_production(production: dict[str, Any]) -> dict[str, Any]:
    """Normalize an existing protocol to one fresh, independent production run."""
    fresh = dict(production)
    duration_ns = float(
        fresh.get("target_duration_ns")
        or fresh.get("production_length_ns")
        or 0.0
    )
    timestep_fs = float(fresh.get("production_timestep_fs") or 0.0)
    if duration_ns <= 0.0 or timestep_fs <= 0.0:
        raise ValueError("The template has no valid production duration or timestep")
    fresh["production_length_ns"] = duration_ns
    fresh["production_steps"] = max(
        1,
        int(round(duration_ns * 1_000_000.0 / timestep_fs)),
    )
    fresh["continuation_mode"] = INDEPENDENT_REPLICA
    fresh.pop("target_duration_ns", None)
    return fresh


def _scientific_invalid_reason(workflow: WorkflowRecord) -> str:
    """Return the recorded reason an MD workflow must never be reused."""
    parameters = workflow.parameters if isinstance(workflow.parameters, dict) else {}
    if not bool(parameters.get("scientifically_invalid")):
        return ""
    return str(
        parameters.get("scientific_invalid_reason")
        or "The prepared molecular system was marked scientifically invalid"
    ).strip()


def _require_scientifically_valid_workflow(workflow: WorkflowRecord) -> None:
    reason = _scientific_invalid_reason(workflow)
    if reason:
        raise ValueError(
            f"This MD workflow cannot be reused or extended: {reason.rstrip('.')}. "
            "Launch a fresh simulation from the immutable source complex."
        )


def create_md_simulation(
    *,
    source_job: JobRecord,
    source_artifact: ArtifactRef,
    prep_input: dict[str, Any],
    production: dict[str, Any],
    replicas: int,
    analysis_enabled: bool,
    image: str = DEFAULT_MD_IMAGE,
    use_gpu: bool = True,
    engine: str = OPENMM_ENGINE,
    comparison_group_id: str = "",
    name: str = "",
) -> WorkflowRecord:
    engine = normalize_md_engine(engine)
    spec = md_engine_spec(engine)
    image = str(image).strip() or spec.default_image
    prep_input = dict(prep_input)
    prep_input["md_engine"] = engine
    if replicas < 1:
        raise ValueError("MD simulation requires at least one production replica")
    if continuation_mode(production) == EXACT_CONTINUATION and replicas != 1:
        raise ValueError("Exact checkpoint continuation supports exactly one trajectory")
    source = _source_contract(source_job, source_artifact)
    target_metadata = _md_target_metadata(source_job, source_artifact)
    expected = ["preparation_equilibration", *(f"production_replica_{index}" for index in range(1, replicas + 1))]
    if _endpoint_enabled(production):
        expected.extend(
            f"endpoint_energy_replica_{index}"
            for index in range(1, replicas + 1)
        )
    if analysis_enabled:
        expected.append("replicate_analysis")
    workflow = create_workflow(
        MD_WORKFLOW_TYPE,
        name=(
            str(name).strip()
            or f"{source_job.metadata.get('pdb_id') or source_job.run_id} MD simulation"
        ),
        parameters={
            "mode": "new",
            "source": source,
            **target_metadata,
            "replicas": replicas,
            "analysis_enabled": analysis_enabled,
            "image": image,
            "use_gpu": use_gpu,
            "engine": engine,
            "comparison_group_id": str(comparison_group_id),
            "production": production,
        },
        expected_steps=expected,
    )
    add_workflow_input(workflow.workflow_id, source_job.task_group, source_artifact)
    prep_child = _new_child(
        "md-system-prep",
        status="preparing",
        metadata={
            "workflow": "md-system-prep",
            "structure_run_id": source_job.run_id,
            "parent_run_id": source_job.run_id,
            **target_metadata,
            "docker_image": image,
            "md_engine": engine,
            "comparison_group_id": str(comparison_group_id),
            "use_gpu": use_gpu,
            "gpu_queued": use_gpu,
            "resources": _worker_resources(image, use_gpu, engine=engine),
            "worker_finalizer": "md_job",
            "source_artifact_id": source_artifact.artifact_id,
            "source_artifact_sha256": source_artifact.sha256,
        },
        input_payload=prep_input,
    )
    command = _docker_command(
        prep_child.run_dir,
        image,
        use_gpu,
        engine=engine,
    )
    stored_input = _read_json(prep_child.run_dir / "input.json")
    stored_input["job_id"] = prep_child.run_id
    stored_input["use_gpu"] = bool(use_gpu)
    source_path = source_artifact.resolve(source_job.run_dir, must_exist=True)
    if source_path is None:
        raise FileNotFoundError(source_artifact.path)
    source_snapshot = prep_child.run_dir / "source_complex.pdb"
    source_snapshot.write_bytes(source_path.read_bytes())
    stored_input["input_complex_pdb_path"] = "/output/source_complex.pdb"
    stored_input["prepared_complex_path"] = "/output/source_complex.pdb"
    source_mapping_path = residue_mapping_artifact(source_job)
    mapping_snapshot = prep_child.run_dir / "source_residue_mapping.json"
    if source_mapping_path is not None:
        mapping_snapshot.write_bytes(source_mapping_path.read_bytes())
        residue_mapping = _read_json(mapping_snapshot)
    else:
        # Legacy prepared targets can already have been renumbered by cleaning
        # or trimming. Trace their immutable lineage instead of assuming the
        # selected PDB still contains deposited author IDs.
        residue_mapping = source_author_residue_mapping(
            source_job,
            source_artifact,
            source_snapshot.read_text(errors="replace"),
        )
        if residue_mapping.get("residues"):
            _write_json(mapping_snapshot, residue_mapping)
    if residue_mapping.get("residues"):
        stored_input["residue_mapping"] = residue_mapping
        stored_input["residue_mapping_path"] = "/output/source_residue_mapping.json"
    stored_input.pop("pdb_data", None)
    refined_ligand_data = str(stored_input.get("ligand_refined_sdf_data") or "")
    if refined_ligand_data:
        ligand_snapshot = prep_child.run_dir / "source_ligand_refined.sdf"
        ligand_snapshot.write_text(refined_ligand_data)
        stored_input["ligand_refined_sdf_path"] = "/output/source_ligand_refined.sdf"
    _write_json(prep_child.run_dir / "input.json", stored_input)
    prep_metadata = _read_json(prep_child.run_dir / "metadata.json")
    prep_metadata.update(
        {
            "status": "queued",
            "queued_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "queued_command": command,
        }
    )
    _write_json(prep_child.run_dir / "metadata.json", prep_metadata)
    write_registered_command_record(
        prep_child.run_dir,
        tool_id=spec.tool_id,
        commands=(command,),
        image=image,
    )
    assert_job_portable(prep_child.run_dir)
    attach_workflow_child(workflow.workflow_id, prep_child, step_id="preparation_equilibration")
    _create_pending_children(
        workflow,
        prep_child,
        production=production,
        replicas=replicas,
        analysis_enabled=analysis_enabled,
        image=image,
        use_gpu=use_gpu,
    )
    return refresh_workflow(workflow.workflow_id)


def create_md_simulation_from_template(
    *,
    template_workflow_id: str,
    source_job: JobRecord,
    source_artifact: ArtifactRef,
    name: str = "",
) -> WorkflowRecord:
    """Launch a fresh MD workflow using protocol settings from an old workflow.

    Only configuration is inherited. The new structural source is prepared from
    scratch and every production child is an independent replica; no prepared
    system, trajectory, checkpoint, or duration-extension child is reused.
    """
    template = WorkflowRecord.load(template_workflow_id)
    if template.workflow_type != MD_WORKFLOW_TYPE:
        raise ValueError("Template must be an MD simulation workflow")
    _require_scientifically_valid_workflow(template)
    prep_ref = next(
        (
            child
            for child in template.children
            if child.step_id == "preparation_equilibration"
        ),
        None,
    )
    if prep_ref is None:
        raise ValueError("Template has no system-preparation step")
    prep_dir = resolve_run_dir(prep_ref.task_group, prep_ref.run_id)
    if prep_dir is None:
        raise FileNotFoundError(f"Template preparation not found: {prep_ref.run_id}")
    template_prep = _read_json(prep_dir / "input.json")
    if not template_prep:
        raise ValueError("Template preparation input is unavailable")

    parameters = dict(template.parameters)
    production = _fresh_template_production(
        dict(parameters.get("production") or {})
    )
    replicas = int(parameters.get("replicas") or 0)
    if replicas < 1:
        raise ValueError("Template has no valid replica count")
    engine = normalize_md_engine(parameters.get("engine") or OPENMM_ENGINE)
    workflow = create_md_simulation(
        source_job=source_job,
        source_artifact=source_artifact,
        prep_input=md_prep_input_for_source(
            source_job,
            source_artifact,
            template_prep,
        ),
        production=production,
        replicas=replicas,
        analysis_enabled=bool(parameters.get("analysis_enabled", True)),
        image=str(parameters.get("image") or md_engine_spec(engine).default_image),
        use_gpu=bool(parameters.get("use_gpu", True)),
        engine=engine,
        comparison_group_id="",
        name=name,
    )
    provenance = dict(workflow.parameters)
    provenance["template_workflow_id"] = template.workflow_id
    provenance["template_settings_only"] = True
    return update_workflow_definition(
        workflow.workflow_id,
        parameters=provenance,
    )


def create_md_simulation_from_prepared(
    *,
    prep_job: JobRecord,
    production: dict[str, Any],
    replicas: int,
    analysis_enabled: bool,
    image: str = DEFAULT_MD_IMAGE,
    use_gpu: bool = True,
) -> WorkflowRecord:
    if prep_job.task_group != "md-system-prep" or prep_job.status != "completed":
        raise ValueError("Reuse requires a completed MD system-preparation job")
    if bool(prep_job.metadata.get("scientifically_invalid")):
        reason = str(
            prep_job.metadata.get("scientific_invalid_reason")
            or "The prepared molecular system was marked scientifically invalid"
        )
        raise ValueError(
            f"This prepared system cannot be reused: {reason}. "
            "Prepare a fresh system from the immutable source complex."
        )
    if replicas < 1:
        raise ValueError("MD simulation requires at least one production replica")
    if continuation_mode(production) == EXACT_CONTINUATION and replicas != 1:
        raise ValueError("Exact checkpoint continuation supports exactly one trajectory")
    finalize_md_job(prep_job)
    prep_job = JobRecord.load(prep_job.run_dir, task_group=prep_job.task_group)
    _validate_roe_preparation_result(
        _read_json(prep_job.run_dir / "result.json")
    )
    contract = prep_job.metadata.get("compatibility_contract") or {}
    engine = normalize_md_engine(
        prep_job.metadata.get("md_engine")
        or contract.get("engine")
        or OPENMM_ENGINE
    )
    spec = md_engine_spec(engine)
    image = str(image).strip()
    if not image or (
        engine == GROMACS_ENGINE and image == DEFAULT_MD_IMAGE
    ):
        image = spec.default_image
    if not prep_job.metadata.get("compatibility_fingerprint"):
        raise ValueError("Prepared system has no compatibility fingerprint")
    artifact_types = {
        artifact.artifact_type for artifact in (prep_job.artifact_manifest.artifacts if prep_job.artifact_manifest else ())
    }
    if not {"md_system", "equilibrated_system"}.issubset(artifact_types):
        raise ValueError("Prepared system is missing reusable system or equilibrated-coordinate artifacts")
    required_restart_types = engine_restart_artifact_types(
        engine,
        exact=continuation_mode(production) == EXACT_CONTINUATION,
    )
    if not required_restart_types.issubset(artifact_types):
        raise ValueError(
            f"{spec.label} restart is missing required artifacts: "
            + ", ".join(sorted(required_restart_types - artifact_types))
        )
    expected = [
        "preparation_equilibration",
        *(f"production_replica_{index}" for index in range(1, replicas + 1)),
    ]
    if _endpoint_enabled(production):
        expected.extend(
            f"endpoint_energy_replica_{index}"
            for index in range(1, replicas + 1)
        )
    if analysis_enabled:
        expected.append("replicate_analysis")
    target_metadata = _inherited_md_target_metadata(prep_job.metadata)
    workflow = create_workflow(
        MD_WORKFLOW_TYPE,
        name=f"{prep_job.metadata.get('pdb_id') or prep_job.run_id} MD continuation",
        parameters={
            "mode": "reuse",
            "prepared_system_run_id": prep_job.run_id,
            **target_metadata,
            "compatibility_fingerprint": prep_job.metadata["compatibility_fingerprint"],
            "replicas": replicas,
            "analysis_enabled": analysis_enabled,
            "image": image,
            "use_gpu": use_gpu,
            "engine": engine,
            "production": production,
        },
        expected_steps=expected,
    )
    for artifact in prep_job.artifact_manifest.artifacts if prep_job.artifact_manifest else ():
        if artifact.artifact_type in {
            "md_system",
            "md_topology",
            "md_index",
            "md_run_input",
            "equilibrated_system",
            "md_checkpoint",
            "openmm_system",
        }:
            add_workflow_input(workflow.workflow_id, prep_job.task_group, artifact)
    attach_workflow_child(
        workflow.workflow_id,
        prep_job,
        step_id="preparation_equilibration",
        update_child_metadata=False,
    )
    _create_pending_children(
        workflow,
        prep_job,
        production=production,
        replicas=replicas,
        analysis_enabled=analysis_enabled,
        image=image,
        use_gpu=use_gpu,
    )
    advance_md_workflow(workflow.workflow_id)
    return WorkflowRecord.load(workflow.workflow_id)


def _create_pending_children(
    workflow: WorkflowRecord,
    prep_job: JobRecord,
    *,
    production: dict[str, Any],
    replicas: int,
    analysis_enabled: bool,
    image: str,
    use_gpu: bool,
    start_index: int = 1,
    create_analysis: bool = True,
) -> None:
    engine = normalize_md_engine(workflow.parameters.get("engine"))
    target_metadata = _inherited_md_target_metadata(
        workflow.parameters,
        prep_job.metadata,
    )
    replica_ids: list[str] = []
    for index in range(start_index, replicas + 1):
        child = _new_child(
            "bound-ligand-md",
            status="queued",
            metadata={
                "workflow": "bound-ligand-md",
                "parent_run_id": prep_job.run_id,
                "md_system_prep_run_id": prep_job.run_id,
                **target_metadata,
                "repeat_group_id": workflow.workflow_id,
                "repeat_index": index,
                "repeat_total": replicas,
                "docker_image": image,
                "md_engine": engine,
                "comparison_group_id": workflow.parameters.get(
                    "comparison_group_id",
                    "",
                ),
                "use_gpu": use_gpu,
                "awaiting_parent": True,
            },
            input_payload={
                "md_engine": engine,
                "production_request": production,
                "source_md_system_prep_run_id": prep_job.run_id,
            },
        )
        replica_ids.append(child.run_id)
        attach_workflow_child(
            workflow.workflow_id,
            child,
            step_id=f"production_replica_{index}",
            depends_on=(prep_job.run_id,),
        )
    if analysis_enabled and create_analysis:
        analysis = _new_child(
            "md-analysis",
            status="queued",
            metadata={
                "workflow": "md-analysis",
                "awaiting_parent": True,
                **target_metadata,
            },
            input_payload={"production_run_ids": replica_ids},
        )
        attach_workflow_child(
            workflow.workflow_id,
            analysis,
            step_id="replicate_analysis",
            depends_on=replica_ids,
        )


def extend_md_simulation(workflow_id: str, target_replicas: int) -> WorkflowRecord:
    """Append independent replicas to an existing MD workflow in place."""
    workflow = WorkflowRecord.load(workflow_id)
    if workflow.workflow_type != MD_WORKFLOW_TYPE:
        raise ValueError("Only MD simulation workflows can be extended here")
    _require_scientifically_valid_workflow(workflow)
    target_replicas = int(target_replicas)
    current = int(workflow.parameters.get("replicas") or 1)
    if target_replicas <= current:
        return workflow
    if target_replicas > 100:
        raise ValueError("MD replicas must be between 1 and 100")
    production = dict(workflow.parameters.get("production") or {})
    if continuation_mode(production) == EXACT_CONTINUATION:
        raise ValueError("Exact checkpoint continuation cannot be extended as independent replicas")
    prep_ref = next(
        (item for item in workflow.children if item.step_id == "preparation_equilibration" and item.required),
        None,
    )
    if prep_ref is None:
        raise ValueError("The MD workflow has no preparation child")
    prep_dir = resolve_run_dir(prep_ref.task_group, prep_ref.run_id)
    if prep_dir is None:
        raise FileNotFoundError(prep_ref.run_id)
    prep_job = JobRecord.load(prep_dir, task_group=prep_ref.task_group)
    parameters = dict(workflow.parameters)
    parameters["replicas"] = target_replicas
    expected = [
        "preparation_equilibration",
        *(f"production_replica_{index}" for index in range(1, target_replicas + 1)),
    ]
    if _endpoint_enabled(production):
        expected.extend(f"endpoint_energy_replica_{index}" for index in range(1, target_replicas + 1))
    analysis_enabled = bool(parameters.get("analysis_enabled"))
    if analysis_enabled:
        expected.append("replicate_analysis")
    update_workflow_definition(
        workflow_id,
        parameters=parameters,
        expected_steps=expected,
    )
    workflow = WorkflowRecord.load(workflow_id)
    for child_ref in workflow.children:
        if not child_ref.step_id.startswith("production_replica_"):
            continue
        child_dir = resolve_run_dir(child_ref.task_group, child_ref.run_id)
        if child_dir is None:
            continue
        metadata = _read_json(child_dir / "metadata.json")
        metadata["repeat_total"] = target_replicas
        metadata["updated_at"] = _utc_now_iso()
        _write_json(child_dir / "metadata.json", metadata)
    _create_pending_children(
        workflow,
        prep_job,
        production=production,
        replicas=target_replicas,
        analysis_enabled=analysis_enabled,
        image=str(parameters.get("image") or DEFAULT_MD_IMAGE),
        use_gpu=bool(parameters.get("use_gpu", True)),
        start_index=current + 1,
        create_analysis=False,
    )
    workflow = WorkflowRecord.load(workflow_id)
    if analysis_enabled:
        production_ids = [
            child.run_id
            for child in workflow.children
            if child.required and child.step_id.startswith("production_replica_")
        ]
        endpoint_ids = [
            child.run_id
            for child in workflow.children
            if child.required and child.step_id.startswith("endpoint_energy_replica_")
        ]
        analysis = _new_child(
            "md-analysis",
            status="queued",
            metadata={
                "workflow": "md-analysis",
                "awaiting_parent": True,
                **_inherited_md_target_metadata(
                    workflow.parameters,
                    prep_job.metadata,
                ),
            },
            input_payload={
                "production_run_ids": production_ids,
                "endpoint_run_ids": endpoint_ids,
            },
        )
        attach_workflow_child(
            workflow_id,
            analysis,
            step_id="replicate_analysis",
            depends_on=(*production_ids, *endpoint_ids),
            replace_step=True,
        )
    advance_md_workflow(workflow_id)
    return WorkflowRecord.load(workflow_id)


def _mounted_preparation_path(prep_dir: Path, value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.startswith("/prepared-system/"):
        candidate = prep_dir / raw.removeprefix("/prepared-system/")
        return candidate if candidate.is_file() else None
    return _resolve_output_path(prep_dir, raw)


def extend_md_simulation_duration(
    workflow_id: str,
    target_duration_ns: float,
) -> WorkflowRecord:
    """Continue every replica with the strict native contract of its engine."""
    workflow = WorkflowRecord.load(workflow_id)
    _require_scientifically_valid_workflow(workflow)
    engine = normalize_md_engine(workflow.parameters.get("engine"))
    if engine == OPENMM_ENGINE:
        return _extend_openmm_md_simulation_duration(
            workflow_id, target_duration_ns
        )
    if engine == GROMACS_ENGINE:
        return _extend_gromacs_md_simulation_duration(
            workflow_id, target_duration_ns
        )
    raise ValueError(
        f"Duration extension has no strict continuation adapter for {engine}"
    )


def _extend_gromacs_md_simulation_duration(
    workflow_id: str,
    target_duration_ns: float,
) -> WorkflowRecord:
    """Continue GROMACS replicas through CPT + convert-tpr + mdrun -append."""
    workflow = WorkflowRecord.load(workflow_id)
    if workflow.workflow_type != MD_WORKFLOW_TYPE:
        raise ValueError("Only MD simulation workflows can be extended")
    if normalize_md_engine(workflow.parameters.get("engine")) != GROMACS_ENGINE:
        raise ValueError("The selected workflow is not a GROMACS workflow")
    if workflow.status not in {"completed", "failed"}:
        raise ValueError("Wait until every current production and endpoint job is terminal")
    production_request = dict(workflow.parameters.get("production") or {})
    timestep_fs = float(production_request.get("production_timestep_fs") or 2.0)
    current_steps = int(production_request.get("production_steps") or 0)
    if timestep_fs <= 0 or current_steps <= 0:
        raise ValueError("The existing GROMACS production step contract is invalid")
    target_steps = int(round(float(target_duration_ns) * 1_000_000.0 / timestep_fs))
    if target_steps <= current_steps:
        raise ValueError(
            f"Target duration must exceed the current "
            f"{current_steps * timestep_fs / 1_000_000.0:g} ns"
        )
    delta_steps = target_steps - current_steps
    prep_ref = next(
        (
            child for child in workflow.children
            if child.required and child.step_id == "preparation_equilibration"
        ),
        None,
    )
    if prep_ref is None:
        raise ValueError("The workflow has no required preparation child")
    prep_dir = resolve_run_dir(prep_ref.task_group, prep_ref.run_id)
    if prep_dir is None:
        raise FileNotFoundError(prep_ref.run_id)
    prep_job = JobRecord.load(prep_dir, task_group=prep_ref.task_group)
    production_refs = sorted(
        (
            child for child in workflow.children
            if child.required and child.step_id.startswith("production_replica_")
        ),
        key=lambda child: child.step_id,
    )
    required_outputs = {
        "production_checkpoint": "production.cpt",
        "native_production_trajectory": "production.xtc",
        "production_tpr": "production.tpr",
        "production_topology": "system.top",
        "production_index": "index.ndx",
        "thermodynamic_series": "production.edr",
    }
    sources: list[tuple[Any, JobRecord, dict[str, Path]]] = []
    for child_ref in production_refs:
        child_dir = resolve_run_dir(child_ref.task_group, child_ref.run_id)
        if child_dir is None:
            raise FileNotFoundError(child_ref.run_id)
        source = JobRecord.load(child_dir, task_group=child_ref.task_group)
        if source.status != "completed":
            raise ValueError(f"Replica {child_ref.step_id} is not completed")
        output_files = _result_output_files(source.result)
        resolved: dict[str, Path] = {}
        for key, fallback_name in required_outputs.items():
            path = _resolve_output_path(source.run_dir, output_files.get(key))
            if path is None:
                fallback = source.run_dir / fallback_name
                path = fallback if fallback.is_file() else None
            if path is None:
                raise ValueError(
                    f"Replica {child_ref.step_id} lacks strict GROMACS "
                    f"continuation artifact {key}"
                )
            resolved[key] = path
        production_log = _resolve_output_path(
            source.run_dir, output_files.get("production_log")
        )
        if production_log is None:
            fallback_log = source.run_dir / "production.log"
            production_log = fallback_log if fallback_log.is_file() else None
        if production_log is None:
            raise ValueError(
                f"Replica {child_ref.step_id} lacks production.log; "
                "GROMACS checksum-safe append is impossible"
            )
        resolved["production_log"] = production_log
        sources.append((child_ref, source, resolved))
    if not sources:
        raise ValueError("The workflow has no production replicas")

    parameters = dict(workflow.parameters)
    production_request["production_steps"] = target_steps
    production_request["target_duration_ns"] = float(target_duration_ns)
    parameters["production"] = production_request
    history = list(parameters.get("duration_extension_history") or [])
    history.append(
        {
            "requested_at": _utc_now_iso(),
            "previous_duration_ns": current_steps * timestep_fs / 1_000_000.0,
            "target_duration_ns": float(target_duration_ns),
            "extension_steps": delta_steps,
            "mode": "strict_gromacs_cpt_append",
        }
    )
    parameters["duration_extension_history"] = history
    update_workflow_definition(
        workflow_id,
        parameters=parameters,
        expected_steps=workflow.expected_steps,
    )

    image = str(parameters.get("image") or DEFAULT_GROMACS_MD_IMAGE)
    use_gpu = bool(parameters.get("use_gpu", True))
    new_production_ids: list[str] = []
    for child_ref, source, resolved in sources:
        source_input = _read_json(source.run_dir / "input.json")
        replica_index = int(source.metadata.get("repeat_index") or 1)
        child = _new_child(
            "bound-ligand-md",
            status="queued",
            metadata={
                "workflow": "bound-ligand-md",
                **_inherited_md_target_metadata(source.metadata),
                "parent_run_id": prep_job.run_id,
                "md_system_prep_run_id": prep_job.run_id,
                "md_engine": GROMACS_ENGINE,
                "repeat_group_id": workflow_id,
                "repeat_index": replica_index,
                "repeat_total": int(parameters.get("replicas") or len(sources)),
                "continuation_of_run_id": source.run_id,
                "duration_extension": True,
                "previous_duration_ns": current_steps * timestep_fs / 1_000_000.0,
                "target_duration_ns": float(target_duration_ns),
                "worker_finalizer": "md_job",
                "gpu_queued": use_gpu,
                "resources": _worker_resources(
                    image, use_gpu, engine=GROMACS_ENGINE
                ),
            },
            input_payload={},
        )
        continuation_dir = child.run_dir / "continuation_source"
        continuation_dir.mkdir()
        copied: dict[str, Path] = {}
        for key, path in resolved.items():
            target = continuation_dir / path.name
            shutil.copy2(path, target)
            copied[key] = target
        for include in source.run_dir.glob("*.itp"):
            shutil.copy2(include, continuation_dir / include.name)
        payload = dict(source_input)
        payload.update(
            {
                "job_id": child.run_id,
                "md_engine": GROMACS_ENGINE,
                "production_steps": delta_steps,
                "production_prior_steps": current_steps,
                "strict_checkpoint_resume": True,
                "continuation_mode": EXACT_CONTINUATION,
                "restart_mode": "GROMACS checkpoint append",
                "continuation_source_run_id": source.run_id,
                "cumulative_target_steps": target_steps,
                "cumulative_target_duration_ns": float(target_duration_ns),
                "source_checkpoint_path": (
                    f"/output/continuation_source/{copied['production_checkpoint'].name}"
                ),
                "source_native_trajectory_path": (
                    f"/output/continuation_source/{copied['native_production_trajectory'].name}"
                ),
                "source_tpr_path": (
                    f"/output/continuation_source/{copied['production_tpr'].name}"
                ),
                "source_topology_path": (
                    f"/output/continuation_source/{copied['production_topology'].name}"
                ),
                "source_index_path": (
                    f"/output/continuation_source/{copied['production_index'].name}"
                ),
                "source_energy_path": (
                    f"/output/continuation_source/{copied['thermodynamic_series'].name}"
                ),
                "source_log_path": (
                    f"/output/continuation_source/{copied['production_log'].name}"
                ),
                "mmgbsa_enabled": False,
                "use_gpu": use_gpu,
            }
        )
        _write_json(child.run_dir / "input.json", payload)
        command = _docker_command(
            child.run_dir,
            image,
            use_gpu,
            prepared_system_dir=prep_job.run_dir,
            engine=GROMACS_ENGINE,
        )
        metadata = _read_json(child.run_dir / "metadata.json")
        metadata.update(
            {
                "queued_command": command,
                "queued_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            }
        )
        _write_json(child.run_dir / "metadata.json", metadata)
        write_registered_command_record(
            child.run_dir,
            tool_id=md_engine_spec(GROMACS_ENGINE).tool_id,
            commands=(command,),
            image=image,
        )
        attach_workflow_child(
            workflow_id,
            child,
            step_id=child_ref.step_id,
            depends_on=(source.run_id,),
            replace_step=True,
        )
        new_production_ids.append(child.run_id)
    if bool(parameters.get("analysis_enabled")):
        analysis = _new_child(
            "md-analysis",
            status="queued",
            metadata={
                "workflow": "md-analysis",
                "awaiting_parent": True,
                **_inherited_md_target_metadata(
                    parameters,
                    prep_job.metadata,
                ),
            },
            input_payload={"production_run_ids": new_production_ids},
        )
        attach_workflow_child(
            workflow_id,
            analysis,
            step_id="replicate_analysis",
            depends_on=new_production_ids,
            replace_step=True,
        )
    return refresh_workflow(workflow_id)


def _extend_openmm_md_simulation_duration(
    workflow_id: str,
    target_duration_ns: float,
) -> WorkflowRecord:
    """Strictly continue every OpenMM replica to a longer cumulative duration."""
    workflow = WorkflowRecord.load(workflow_id)
    if workflow.workflow_type != MD_WORKFLOW_TYPE:
        raise ValueError("Only MD simulation workflows can be extended")
    if normalize_md_engine(workflow.parameters.get("engine")) != OPENMM_ENGINE:
        raise ValueError("The selected workflow is not an OpenMM workflow")
    if workflow.status not in {"completed", "failed"}:
        raise ValueError("Wait until every current production and endpoint job is terminal")
    production_request = dict(workflow.parameters.get("production") or {})
    timestep_fs = float(production_request.get("production_timestep_fs") or 4.0)
    if timestep_fs <= 0:
        raise ValueError("Production timestep must be positive")
    current_steps = int(production_request.get("production_steps") or 0)
    target_steps = int(round(float(target_duration_ns) * 1_000_000.0 / timestep_fs))
    if target_steps <= current_steps:
        raise ValueError(
            f"Target duration must exceed the current {current_steps * timestep_fs / 1_000_000.0:g} ns"
        )
    delta_steps = target_steps - current_steps
    prep_ref = next(
        (
            child
            for child in workflow.children
            if child.required and child.step_id == "preparation_equilibration"
        ),
        None,
    )
    if prep_ref is None:
        raise ValueError("The workflow has no required preparation child")
    prep_dir = resolve_run_dir(prep_ref.task_group, prep_ref.run_id)
    if prep_dir is None:
        raise FileNotFoundError(prep_ref.run_id)
    prep_job = JobRecord.load(prep_dir, task_group=prep_ref.task_group)
    production_refs = sorted(
        (
            child
            for child in workflow.children
            if child.required and child.step_id.startswith("production_replica_")
        ),
        key=lambda child: child.step_id,
    )
    if not production_refs:
        raise ValueError("The workflow has no production replicas")
    source_jobs: list[tuple[Any, JobRecord]] = []
    for child_ref in production_refs:
        child_dir = resolve_run_dir(child_ref.task_group, child_ref.run_id)
        if child_dir is None:
            raise FileNotFoundError(child_ref.run_id)
        source = JobRecord.load(child_dir, task_group=child_ref.task_group)
        if source.status != "completed":
            raise ValueError(f"Replica {child_ref.step_id} is not completed")
        output_files = _result_output_files(source.result)
        for key in ("production_checkpoint", "production_trajectory", "production_pdb"):
            resolved = (
                _openmm_production_checkpoint(
                    source.run_dir,
                    source.run_id,
                    output_files.get(key),
                )
                if key == "production_checkpoint"
                else _resolve_output_path(source.run_dir, output_files.get(key))
            )
            if resolved is None:
                raise ValueError(
                    f"Replica {child_ref.step_id} lacks required continuation artifact {key}"
                )
        source_input = _read_json(source.run_dir / "input.json")
        for key in ("resume_system_pdb_path", "resume_system_xml_path", "resume_integrator_xml_path"):
            if _mounted_preparation_path(prep_dir, source_input.get(key)) is None:
                raise ValueError(
                    f"Replica {child_ref.step_id} lacks the exact serialized restart bundle ({key})"
                )
        source_jobs.append((child_ref, source))

    parameters = dict(workflow.parameters)
    production_request["production_steps"] = target_steps
    production_request["target_duration_ns"] = float(target_duration_ns)
    parameters["production"] = production_request
    history = list(parameters.get("duration_extension_history") or [])
    history.append(
        {
            "requested_at": _utc_now_iso(),
            "previous_duration_ns": current_steps * timestep_fs / 1_000_000.0,
            "target_duration_ns": float(target_duration_ns),
            "extension_steps": delta_steps,
            "mode": "strict_openmm_checkpoint_append",
        }
    )
    parameters["duration_extension_history"] = history
    update_workflow_definition(
        workflow_id,
        parameters=parameters,
        expected_steps=workflow.expected_steps,
    )

    image = str(parameters.get("image") or DEFAULT_MD_IMAGE)
    use_gpu = bool(parameters.get("use_gpu", True))
    new_production_ids: list[str] = []
    for child_ref, source in source_jobs:
        source_input = _read_json(source.run_dir / "input.json")
        output_files = _result_output_files(source.result)
        checkpoint = _openmm_production_checkpoint(
            source.run_dir,
            source.run_id,
            output_files.get("production_checkpoint"),
        )
        trajectory = _resolve_output_path(
            source.run_dir, output_files.get("production_trajectory")
        )
        production_log = _resolve_output_path(
            source.run_dir, output_files.get("production_log")
        )
        assert checkpoint is not None and trajectory is not None
        replica_index = int(source.metadata.get("repeat_index") or 1)
        child = _new_child(
            "bound-ligand-md",
            status="queued",
            metadata={
                **{
                    key: value
                    for key, value in source.metadata.items()
                    if key
                    in {
                        "comparison_group_id",
                        "compatibility_fingerprint",
                        "docker_image",
                        "md_engine",
                        *MD_TARGET_METADATA_KEYS,
                        "use_gpu",
                    }
                },
                "workflow": "bound-ligand-md",
                "parent_run_id": prep_job.run_id,
                "md_system_prep_run_id": prep_job.run_id,
                "repeat_group_id": workflow_id,
                "repeat_index": replica_index,
                "repeat_total": int(parameters.get("replicas") or len(source_jobs)),
                "continuation_of_run_id": source.run_id,
                "duration_extension": True,
                "previous_duration_ns": current_steps * timestep_fs / 1_000_000.0,
                "target_duration_ns": float(target_duration_ns),
                "worker_finalizer": "md_job",
                "gpu_queued": use_gpu,
                "resources": _worker_resources(
                    image, use_gpu, engine=OPENMM_ENGINE
                ),
            },
            input_payload={},
        )
        continuation_dir = child.run_dir / "continuation_source"
        continuation_dir.mkdir()
        checkpoint_snapshot = continuation_dir / "production_checkpoint.chk"
        shutil.copy2(checkpoint, checkpoint_snapshot)
        # MDOptimizationService scopes its output below ``/output/<job_id>``.
        # Seed that exact directory so DCDReporter/StateDataReporter append to
        # the preceding segment instead of silently creating a delta-only file
        # beside an unused copy at the run root.
        continuation_output_dir = child.run_dir / child.run_id
        continuation_output_dir.mkdir()
        trajectory_target = continuation_output_dir / trajectory.name
        shutil.copy2(trajectory, trajectory_target)
        if production_log is not None:
            shutil.copy2(
                production_log,
                continuation_output_dir / production_log.name,
            )
        payload = dict(source_input)
        payload.update(
            {
                "job_id": child.run_id,
                "production_steps": delta_steps,
                "production_prior_steps": current_steps,
                "append_production_outputs": True,
                "production_only_from_prepared": True,
                "strict_checkpoint_resume": True,
                "continuation_mode": EXACT_CONTINUATION,
                "restart_mode": "Checkpoint (exact continuation)",
                "resume_from_checkpoint_path": "/output/continuation_source/production_checkpoint.chk",
                "continuation_source_run_id": source.run_id,
                "cumulative_target_steps": target_steps,
                "cumulative_target_duration_ns": float(target_duration_ns),
                "mmgbsa_enabled": False,
            }
        )
        _stage_openmm_continuation_inputs(
            source.run_dir,
            child.run_dir,
            payload,
        )
        _write_json(child.run_dir / "input.json", payload)
        command = _docker_command(
            child.run_dir,
            image,
            use_gpu,
            prepared_system_dir=prep_job.run_dir,
            engine=OPENMM_ENGINE,
        )
        metadata = _read_json(child.run_dir / "metadata.json")
        metadata.update(
            {
                "queued_command": command,
                "queued_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            }
        )
        _write_json(child.run_dir / "metadata.json", metadata)
        write_registered_command_record(
            child.run_dir,
            tool_id=md_engine_spec(OPENMM_ENGINE).tool_id,
            commands=(command,),
            image=image,
        )
        attach_workflow_child(
            workflow_id,
            child,
            step_id=child_ref.step_id,
            depends_on=(source.run_id,),
            replace_step=True,
        )
        new_production_ids.append(child.run_id)

    workflow = WorkflowRecord.load(workflow_id)
    if bool(parameters.get("analysis_enabled")):
        analysis = _new_child(
            "md-analysis",
            status="queued",
            metadata={
                "workflow": "md-analysis",
                "awaiting_parent": True,
                **_inherited_md_target_metadata(
                    parameters,
                    prep_job.metadata,
                ),
            },
            input_payload={"production_run_ids": new_production_ids},
        )
        attach_workflow_child(
            workflow_id,
            analysis,
            step_id="replicate_analysis",
            depends_on=new_production_ids,
            replace_step=True,
        )
    return refresh_workflow(workflow_id)


def _resolve_output_path(run_dir: Path, value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    candidates = [candidate]
    if raw.startswith("/output/"):
        candidates.insert(0, run_dir / raw.removeprefix("/output/"))
    candidates.append(run_dir / candidate.name)
    candidates.extend(run_dir.rglob(candidate.name))
    resolved_run = run_dir.resolve()
    for item in candidates:
        try:
            resolved = item.resolve()
            resolved.relative_to(resolved_run)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            return resolved
    return None


def _openmm_production_checkpoint(
    run_dir: Path,
    run_id: str,
    output_value: Any,
) -> Path | None:
    """Resolve current and legacy nested OpenMM production checkpoints."""
    checkpoint = _resolve_output_path(run_dir, output_value)
    if checkpoint is not None:
        return checkpoint
    for candidate in (
        run_dir / "production_checkpoint.chk",
        run_dir / run_id / "production_checkpoint.chk",
    ):
        if candidate.is_file():
            return candidate
    return None


def _stage_openmm_continuation_inputs(
    source_dir: Path,
    continuation_dir: Path,
    payload: dict[str, Any],
) -> None:
    """Snapshot legacy /output inputs required before checkpoint restore."""
    for key in (
        "input_complex_pdb_path",
        "prepared_complex_path",
        "ligand_refined_sdf_path",
        "residue_mapping_path",
    ):
        configured = payload.get(key)
        if not configured:
            continue
        source = _resolve_output_path(source_dir, configured)
        if source is None:
            if key in {"input_complex_pdb_path", "prepared_complex_path"}:
                raise ValueError(
                    f"OpenMM continuation source lacks required staged input {key}"
                )
            payload.pop(key, None)
            continue
        target = continuation_dir / source.name
        if not target.is_file():
            shutil.copy2(source, target)
        payload[key] = f"/output/{target.name}"


def _result_output_files(result: dict[str, Any]) -> dict[str, Any]:
    md_result = result.get("md_result") if isinstance(result.get("md_result"), dict) else {}
    return md_result.get("output_files") if isinstance(md_result.get("output_files"), dict) else {}


def _md_outputs_are_finalized(job: JobRecord) -> bool:
    """Return true when a successful MD result already has complete artifacts.

    Workers poll workflow state frequently.  Avoid re-hashing multi-gigabyte
    trajectories on every poll, and give a second worker a cheap idempotency
    check after it waits for the per-run finalization lock.
    """
    if job.status != "completed" or job.result.get("success") is not True:
        return False
    manifest = job.artifact_manifest
    if manifest is None or not manifest.artifacts:
        return False
    output_files = _result_output_files(job.result)
    mapping = (
        _PREP_ARTIFACTS
        if job.task_group == "md-system-prep"
        else _PRODUCTION_ARTIFACTS
    )
    expected_paths: set[str] = set()
    for key in mapping:
        resolved = _resolve_output_path(job.run_dir, output_files.get(key))
        if resolved is not None:
            expected_paths.add(
                resolved.resolve().relative_to(job.run_dir.resolve()).as_posix()
            )
    declared_paths = {artifact.path for artifact in manifest.artifacts}
    return bool(expected_paths) and expected_paths.issubset(declared_paths) and all(
        artifact.resolve(job.run_dir, must_exist=True) is not None
        for artifact in manifest.artifacts
    )


def finalize_md_job(job: JobRecord) -> JobRecord:
    """Finalize one MD child exactly once across all worker processes."""
    lock_path = job.run_dir / ".md-finalization.lock"
    with _exclusive_file_lock(lock_path):
        current = JobRecord.load(job.run_dir, task_group=job.task_group)
        if _md_outputs_are_finalized(current):
            return current
        return _finalize_md_job_unlocked(current)


def _finalize_md_job_unlocked(job: JobRecord) -> JobRecord:
    result = _read_json(job.run_dir / "result.json")
    metadata = _read_json(job.run_dir / "metadata.json")
    input_payload = _read_json(job.run_dir / "input.json")
    residue_mapping = input_payload.get("residue_mapping")
    if job.task_group == "bound-ligand-md" and isinstance(
        residue_mapping,
        dict,
    ):
        structural = (
            ((result.get("md_result") or {}).get("analytics") or {}).get(
                "structural_dynamics"
            )
            if isinstance(result.get("md_result"), dict)
            else None
        )
        if isinstance(structural, dict):
            relabel_structural_dynamics(structural, residue_mapping)
            _write_json(job.run_dir / "result.json", result)
        residue_mapping_path = job.run_dir / "source_residue_mapping.json"
        if not residue_mapping_path.is_file():
            _write_json(residue_mapping_path, residue_mapping)
    if result:
        metadata["status"] = "completed" if result.get("success") is True else "failed"
        if metadata["status"] in {"completed", "failed"}:
            metadata["completed_at"] = metadata.get("completed_at") or _utc_now_iso()
    output_files = _result_output_files(result)
    mapping = _PREP_ARTIFACTS if job.task_group == "md-system-prep" else _PRODUCTION_ARTIFACTS
    artifacts: list[ArtifactRef] = []
    seen: set[Path] = set()
    for key, (artifact_type, role) in mapping.items():
        path = _resolve_output_path(job.run_dir, output_files.get(key))
        if path is None or path in seen:
            continue
        seen.add(path)
        artifacts.append(ArtifactRef.from_path(job.run_dir, path, artifact_type, role=role))
    if (
        job.task_group == "md-system-prep"
        and normalize_md_engine(
            metadata.get("md_engine")
            or _read_json(job.run_dir / "input.json").get("md_engine")
            or OPENMM_ENGINE
        )
        == GROMACS_ENGINE
    ):
        for pattern, role in (
            ("roe_*.mdp", "gromacs-mdp"),
            ("roe_posre_*.itp", "gromacs-restraints"),
        ):
            for path in sorted(job.run_dir.glob(pattern)):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    artifacts.append(
                        ArtifactRef.from_path(
                            job.run_dir,
                            path,
                            "native_output",
                            role=role,
                        )
                    )
    if job.task_group == "bound-ligand-md":
        for path in sorted(job.run_dir.glob("mmgbsa_*")):
            if path.is_file() and path not in seen:
                artifacts.append(ArtifactRef.from_path(job.run_dir, path, "endpoint_energy", role="mmgbsa"))
    residue_mapping_path = job.run_dir / "source_residue_mapping.json"
    if residue_mapping_path.is_file() and residue_mapping_path not in seen:
        artifacts.append(
            ArtifactRef.from_path(
                job.run_dir,
                residue_mapping_path,
                "residue_mapping",
                role="author_numbering",
            )
        )
    write_artifact_manifest(job.run_dir, artifacts)
    if job.task_group == "md-system-prep" and metadata.get("status") == "completed":
        input_payload = _read_json(job.run_dir / "input.json")
        source_path = _resolve_output_path(job.run_dir, input_payload.get("prepared_complex_path"))
        source_artifact_id = str(metadata.get("source_artifact_id") or "")
        source_sha256 = str(metadata.get("source_artifact_sha256") or "")
        source_run_id = str(metadata.get("structure_run_id") or "")
        source_dir = resolve_run_dir("structure-jobs", source_run_id)
        if source_dir is not None and not source_sha256:
            source_job = JobRecord.load(source_dir, task_group="structure-jobs")
            prepared = source_job.artifact_manifest.by_type("prepared_complex") if source_job.artifact_manifest else ()
            if prepared:
                source_artifact_id = prepared[0].artifact_id
                source_sha256 = prepared[0].sha256
        source = {
            "task_group": "structure-jobs",
            "run_id": source_run_id,
            "artifact_id": source_artifact_id,
            "sha256": source_sha256 or (_sha256(source_path) if source_path else ""),
        }
        engine = normalize_md_engine(
            metadata.get("md_engine")
            or input_payload.get("md_engine")
            or OPENMM_ENGINE
        )
        contract = compatibility_contract(
            source,
            input_payload,
            engine=engine,
        )
        metadata["md_engine"] = engine
        metadata["compatibility_contract"] = contract
        metadata["compatibility_fingerprint"] = compatibility_fingerprint(contract)
    metadata["updated_at"] = _utc_now_iso()
    _write_json(job.run_dir / "metadata.json", metadata)
    return JobRecord.load(job.run_dir, task_group=job.task_group)


def _prepared_system_path(prep_dir: Path, path: Path) -> str:
    return str(Path("/prepared-system") / path.resolve().relative_to(prep_dir.resolve()))


def _activate_gromacs_production(
    child: JobRecord,
    prep_job: JobRecord,
    workflow: WorkflowRecord,
    *,
    prep_result: dict[str, Any],
    prep_input: dict[str, Any],
    request: dict[str, Any],
) -> None:
    output_files = _result_output_files(prep_result)
    required_keys = {
        "source_topology_path": "gromacs_topology",
        "source_coordinates_path": "gromacs_coordinates",
        "source_checkpoint_path": "gromacs_checkpoint",
        "source_index_path": "gromacs_index",
    }
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for target_key, source_key in required_keys.items():
        path = _resolve_output_path(prep_job.run_dir, output_files.get(source_key))
        if path is None:
            missing.append(source_key)
        else:
            resolved[target_key] = path
    if missing:
        raise ValueError(
            "Prepared GROMACS system is missing: " + ", ".join(sorted(missing))
        )

    repeat_index = int(child.metadata.get("repeat_index") or 1)
    start_mode = continuation_mode(request)
    payload = {
        **prep_input,
        **request,
        "job_id": child.run_id,
        "md_engine": GROMACS_ENGINE,
        "output_dir": "/output",
        "source_md_system_prep_run_id": prep_job.run_id,
        "source_md_system_prep_result_json": "/prepared-system/result.json",
        "continuation_mode": start_mode,
        "replica_seed": replica_seed(workflow.workflow_id, repeat_index),
        "repeat_group_id": workflow.workflow_id,
        "repeat_index": child.metadata.get("repeat_index"),
        "repeat_total": child.metadata.get("repeat_total"),
        "mmgbsa_enabled": False,
        "use_gpu": bool(workflow.parameters.get("use_gpu", True)),
    }
    for target_key, path in resolved.items():
        payload[target_key] = _prepared_system_path(prep_job.run_dir, path)
    _write_json(child.run_dir / "input.json", payload)

    image = str(workflow.parameters.get("image") or DEFAULT_GROMACS_MD_IMAGE)
    use_gpu = bool(workflow.parameters.get("use_gpu", True))
    command = _docker_command(
        child.run_dir,
        image,
        use_gpu,
        prepared_system_dir=prep_job.run_dir,
        engine=GROMACS_ENGINE,
    )
    metadata = _read_json(child.run_dir / "metadata.json")
    metadata.pop("error", None)
    metadata.pop("warning", None)
    metadata.update(
        {
            "status": "queued",
            "awaiting_parent": False,
            "md_engine": GROMACS_ENGINE,
            "gpu_queued": use_gpu,
            "resources": _worker_resources(
                image,
                use_gpu,
                engine=GROMACS_ENGINE,
            ),
            "worker_finalizer": "md_job",
            "queued_command": command,
            "compatibility_fingerprint": prep_job.metadata.get(
                "compatibility_fingerprint"
            ),
            "updated_at": _utc_now_iso(),
        }
    )
    _write_json(child.run_dir / "metadata.json", metadata)
    write_registered_command_record(
        child.run_dir,
        tool_id=md_engine_spec(GROMACS_ENGINE).tool_id,
        commands=(command,),
        image=image,
    )


def _activate_production(child: JobRecord, prep_job: JobRecord, workflow: WorkflowRecord) -> None:
    prep_result = _read_json(prep_job.run_dir / "result.json")
    _validate_roe_preparation_result(prep_result)
    prep_input = _read_json(prep_job.run_dir / "input.json")
    request = (_read_json(child.run_dir / "input.json").get("production_request") or {})
    engine = normalize_md_engine(
        workflow.parameters.get("engine")
        or prep_job.metadata.get("md_engine")
        or prep_input.get("md_engine")
        or OPENMM_ENGINE
    )
    if engine == GROMACS_ENGINE:
        _activate_gromacs_production(
            child,
            prep_job,
            workflow,
            prep_result=prep_result,
            prep_input=prep_input,
            request=request,
        )
        return
    output_files = _result_output_files(prep_result)
    npt_pdb = _resolve_output_path(prep_job.run_dir, output_files.get("npt_pdb"))
    system_pdb = _resolve_output_path(prep_job.run_dir, output_files.get("system_pdb"))
    checkpoint = _resolve_output_path(prep_job.run_dir, output_files.get("npt_checkpoint"))
    if npt_pdb is None or system_pdb is None:
        raise ValueError("Prepared system is missing NPT or system coordinates")
    start_mode = continuation_mode(request)
    repeat_index = int(child.metadata.get("repeat_index") or 1)
    payload = dict(prep_input)
    payload.update(
        {
            "job_id": child.run_id,
            "heating_steps_per_stage": 0,
            "nvt_steps": 0,
            "npt_steps": 0,
            "production_steps": int(request.get("production_steps") or 0),
            "production_report_interval": int(request.get("production_report_interval") or 2500),
            "output_dir": "/output",
            "source_md_system_prep_run_id": prep_job.run_id,
            "source_md_system_prep_result_json": "/prepared-system/result.json",
            "restart_mode": (
                "Checkpoint (exact continuation)"
                if start_mode == EXACT_CONTINUATION
                else "Independent replica from NPT coordinates"
            ),
            "continuation_mode": start_mode,
            "production_only_from_prepared": True,
            "strict_checkpoint_resume": start_mode == EXACT_CONTINUATION,
            "coordinate_restart_policy": (
                "independent_replica" if start_mode == INDEPENDENT_REPLICA else "legacy_minimize_rethermalize"
            ),
            "replica_equilibration_steps": (
                int(request.get("replica_equilibration_steps") or 0)
                if start_mode == INDEPENDENT_REPLICA
                else 0
            ),
            "replica_density_revalidation": bool(
                request.get("replica_density_revalidation", False)
            ) if start_mode == INDEPENDENT_REPLICA else False,
            "replica_revalidation_max_steps": int(
                request.get("replica_revalidation_max_steps") or 0
            ) if start_mode == INDEPENDENT_REPLICA else 0,
            "replica_revalidation_increment_steps": int(
                request.get("replica_revalidation_increment_steps") or 0
            ) if start_mode == INDEPENDENT_REPLICA else 0,
            "replica_density_sample_interval_steps": int(
                request.get("replica_density_sample_interval_steps") or 0
            ) if start_mode == INDEPENDENT_REPLICA else 0,
            "replica_density_plateau_required": bool(
                request.get("replica_density_plateau_required", True)
            ),
            "replica_seed": replica_seed(workflow.workflow_id, repeat_index),
            # Endpoint calculations are separate immutable children. Keeping this
            # false prevents expensive analysis from being coupled to production.
            "mmgbsa_enabled": False,
            "mmgbsa_start_pct": int(request.get("mmgbsa_start_pct") or 20),
            "mmgbsa_end_pct": int(request.get("mmgbsa_end_pct") or 100),
            "mmgbsa_stride": int(request.get("mmgbsa_stride") or 1),
            "repeat_group_id": workflow.workflow_id,
            "repeat_index": child.metadata.get("repeat_index"),
            "repeat_total": child.metadata.get("repeat_total"),
            "restart_contract": {
                "npt_pdb": _prepared_system_path(prep_job.run_dir, npt_pdb),
                "npt_checkpoint": _prepared_system_path(prep_job.run_dir, checkpoint) if checkpoint else None,
                "system_pdb": _prepared_system_path(prep_job.run_dir, system_pdb),
            },
        }
    )
    coordinate_restart = start_mode == INDEPENDENT_REPLICA
    serialized_paths: dict[str, Path] = {}
    for source_key in ("npt_state_xml", "npt_system_xml", "npt_integrator_xml"):
        path = _resolve_output_path(prep_job.run_dir, output_files.get(source_key))
        if path:
            serialized_paths[source_key] = path
    if checkpoint is not None and not coordinate_restart:
        payload["resume_from_checkpoint_path"] = _prepared_system_path(prep_job.run_dir, checkpoint)
        payload["resume_system_pdb_path"] = _prepared_system_path(prep_job.run_dir, system_pdb)
        for source_key, target_key in (
            ("npt_state_xml", "resume_state_xml_path"),
            ("npt_system_xml", "resume_system_xml_path"),
            ("npt_integrator_xml", "resume_integrator_xml_path"),
        ):
            path = serialized_paths.get(source_key)
            if path:
                payload[target_key] = _prepared_system_path(prep_job.run_dir, path)
    elif coordinate_restart:
        system_xml = serialized_paths.get("npt_system_xml")
        integrator_xml = serialized_paths.get("npt_integrator_xml")
        if system_xml is None or integrator_xml is None:
            raise ValueError("Independent replica requires serialized OpenMM System and Integrator")
        payload.pop("resume_from_checkpoint_path", None)
        payload["resume_system_pdb_path"] = _prepared_system_path(prep_job.run_dir, npt_pdb)
        state_xml = serialized_paths.get("npt_state_xml")
        payload["resume_system_xml_path"] = _prepared_system_path(prep_job.run_dir, system_xml)
        payload["resume_integrator_xml_path"] = _prepared_system_path(prep_job.run_dir, integrator_xml)
        if state_xml is None:
            raise ValueError("Independent replica requires a serialized OpenMM State")
        payload["resume_state_xml_path"] = _prepared_system_path(prep_job.run_dir, state_xml)
    else:
        for key in (
            "resume_from_checkpoint_path",
            "resume_system_pdb_path",
            "resume_state_xml_path",
            "resume_system_xml_path",
            "resume_integrator_xml_path",
        ):
            payload.pop(key, None)
    input_snapshot = child.run_dir / "final_input_protein_refined.pdb"
    input_snapshot.write_bytes(npt_pdb.read_bytes())
    payload["input_complex_pdb_path"] = "/output/final_input_protein_refined.pdb"
    payload["prepared_complex_path"] = "/output/final_input_protein_refined.pdb"
    refined_ligand_data = str(payload.get("ligand_refined_sdf_data") or "")
    if refined_ligand_data:
        ligand_snapshot = child.run_dir / "source_ligand_refined.sdf"
        ligand_snapshot.write_text(refined_ligand_data)
        payload["ligand_refined_sdf_path"] = "/output/source_ligand_refined.sdf"
    payload.pop("pdb_data", None)
    _write_json(child.run_dir / "input.json", payload)
    metadata = _read_json(child.run_dir / "metadata.json")
    metadata.pop("error", None)
    metadata.pop("warning", None)
    image = str(workflow.parameters.get("image") or DEFAULT_MD_IMAGE)
    use_gpu = bool(workflow.parameters.get("use_gpu", True))
    command = _docker_command(
        child.run_dir,
        image,
        use_gpu,
        prepared_system_dir=prep_job.run_dir,
        engine=OPENMM_ENGINE,
    )
    metadata.update(
        {
            "status": "queued",
            "awaiting_parent": False,
            "gpu_queued": use_gpu,
            "resources": _worker_resources(
                image,
                use_gpu,
                engine=OPENMM_ENGINE,
            ),
            "worker_finalizer": "md_job",
            "queued_command": command,
            "compatibility_fingerprint": prep_job.metadata.get("compatibility_fingerprint"),
            "updated_at": _utc_now_iso(),
        }
    )
    _write_json(child.run_dir / "metadata.json", metadata)
    write_registered_command_record(
        child.run_dir,
        tool_id=md_engine_spec(OPENMM_ENGINE).tool_id,
        commands=(command,),
        image=image,
    )


def finalize_md_worker_job(run_dir: Path, *, returncode: int) -> JobRecord:
    """Finalize a worker-owned MD preparation or production child."""
    run_dir = run_dir.resolve()
    task_group = run_dir.parent.name
    result_path = run_dir / "result.json"
    if returncode != 0 and not result_path.is_file():
        stderr = (run_dir / "stderr.log").read_text(errors="replace") if (
            run_dir / "stderr.log"
        ).is_file() else ""
        _write_json(
            result_path,
            {
                "success": False,
                "returncode": returncode,
                "error": stderr[-4000:] or "MD container execution failed",
            },
        )
    return finalize_md_job(JobRecord.load(run_dir, task_group=task_group))


def _rewrite_source_paths(value: Any, source_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_source_paths(item, source_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_source_paths(item, source_dir) for item in value]
    if not isinstance(value, str):
        return value
    rewritten = value.replace(str(source_dir.resolve()), "/source")
    if rewritten.startswith("/output/"):
        rewritten = "/source/" + rewritten.removeprefix("/output/")
    return rewritten


def _mmgbsa_command(
    run_dir: Path,
    source_dir: Path,
    *,
    prepared_system_dir: Path | None,
    image: str,
    use_gpu: bool,
    gpu_device: str,
    start_pct: float,
    end_pct: float,
    stride: int,
    backend: str,
    engine: str,
) -> list[str]:
    selected = str(gpu_device).strip().lower().removeprefix("gpu ") or "all"
    gpu_ids = None if selected in {"all", "auto", "automatic"} else (
        int(selected.removeprefix("device=")),
    )
    spec = md_engine_spec(engine)
    command = (
        (
            "python",
            "-m",
            "mn_ligand.workflows.gromacs_md",
            "endpoint",
            "--input",
            "/output/source_input.json",
            "--result",
            "/output/source_result.json",
            "--output",
            "/output/result.json",
            "--start-pct",
            str(float(start_pct)),
            "--end-pct",
            str(float(end_pct)),
            "--stride",
            str(int(stride)),
        )
        if backend == "g_mmpbsa"
        else (
            "python",
            "-m",
            "mn_ligand.workflows.bound_ligand_md",
            "mmgbsa",
            "--input",
            "/output/source_input.json",
            "--result",
            "/output/source_result.json",
            "--output",
            "/output/result.json",
            "--start-pct",
            str(float(start_pct)),
            "--end-pct",
            str(float(end_pct)),
            "--stride",
            str(int(stride)),
            "--backend",
            backend,
        )
    )
    mounts = [
        DockerMount(PROJECT_DIR, "/mn-ligand", read_only=True),
        DockerMount(source_dir, "/source", read_only=True),
        DockerMount(run_dir, "/output"),
    ]
    if prepared_system_dir is not None:
        mounts.append(
            DockerMount(
                prepared_system_dir,
                "/prepared-system",
                read_only=True,
            )
        )
    return build_docker_command(
        DockerRunSpec(
            tool=registered_tool(spec.tool_id, image=image),
            command=command,
            mounts=tuple(mounts),
            environment={"PYTHONPATH": "/mn-ligand"},
            gpu_enabled=use_gpu,
            gpu_devices=gpu_ids if use_gpu else None,
            shm_size=os.getenv("MN_MD_DOCKER_SHM_SIZE", "64g").strip(),
            use_host_user=False,
        )
    )


def create_mmgbsa_analysis_job(
    production_run_id: str,
    *,
    start_pct: float = 20.0,
    end_pct: float = 100.0,
    stride: int = 1,
    backend: str = "openmm_gbsa",
    image: str = "",
    use_gpu: bool | None = None,
    gpu_device: str = "all",
    attach_to_workflow: bool = True,
) -> JobRecord:
    source_dir = resolve_run_dir("bound-ligand-md", production_run_id)
    if source_dir is None:
        raise FileNotFoundError(f"MD production job not found: {production_run_id}")
    source_job = JobRecord.load(source_dir, task_group="bound-ligand-md")
    source_result = _read_json(source_dir / "result.json")
    if source_job.status != "completed" or source_result.get("success") is not True:
        raise ValueError("MM/GBSA requires a completed MD production job")
    if not 0.0 <= float(start_pct) <= float(end_pct) <= 100.0:
        raise ValueError("MM/GBSA percentages must satisfy 0 <= start <= end <= 100")
    if int(stride) < 1:
        raise ValueError("MM/GBSA stride must be positive")
    backend = str(backend).strip().lower()
    if backend not in {"openmm_gbsa", "ambertools_mmpbsa", "g_mmpbsa"}:
        raise ValueError(f"Unsupported MM/GBSA backend: {backend}")
    source_input = _read_json(source_dir / "input.json")
    if not source_input or not source_result:
        raise ValueError("MD production input or result metadata is missing")
    source_engine = normalize_md_engine(
        source_job.metadata.get("md_engine")
        or source_result.get("engine")
        or (source_result.get("md_result") or {}).get("engine")
        or source_input.get("md_engine")
        or OPENMM_ENGINE
    )
    if not endpoint_backend_supported(source_engine, backend):
        raise ValueError(
            f"Endpoint backend {backend} is incompatible with "
            f"{md_engine_spec(source_engine).label} artifacts"
        )
    source_artifacts: list[ArtifactRef] = [
        artifact
        for artifact in (source_job.artifact_manifest.artifacts if source_job.artifact_manifest else ())
        if artifact.artifact_type in {
            "md_trajectory",
            "md_final_structure",
            "md_checkpoint",
            "md_topology",
            "md_run_input",
            "md_index",
        }
        and artifact.resolve(source_dir, must_exist=True) is not None
    ]
    existing_types = {artifact.artifact_type for artifact in source_artifacts}
    output_files = _result_output_files(source_result)
    for key, (artifact_type, role) in _PRODUCTION_ARTIFACTS.items():
        if artifact_type in existing_types:
            continue
        path = _resolve_output_path(source_dir, output_files.get(key))
        if path is None:
            continue
        source_artifacts.append(
            ArtifactRef.from_path(source_dir, path, artifact_type, role=role)
        )
        existing_types.add(artifact_type)
    source_artifact_types = {artifact.artifact_type for artifact in source_artifacts}
    missing_types = {"md_trajectory", "md_final_structure"} - source_artifact_types
    if missing_types:
        raise ValueError(
            "MD production is missing required typed artifacts: "
            + ", ".join(sorted(missing_types))
        )
    run_id = str(uuid4())
    run_dir = runs_root() / MMGBSA_TASK_GROUP / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    image = (
        str(image).strip()
        or str(source_job.metadata.get("docker_image") or "")
        or md_engine_spec(source_engine).default_image
    )
    selected_use_gpu = bool(source_job.metadata.get("use_gpu", True)) if use_gpu is None else bool(use_gpu)
    if backend in {"ambertools_mmpbsa", "g_mmpbsa"}:
        selected_use_gpu = False
    prep_run_id = str(
        source_input.get("source_md_system_prep_run_id")
        or source_job.metadata.get("md_system_prep_run_id")
        or ""
    ).strip()
    prepared_system_dir = (
        resolve_run_dir("md-system-prep", prep_run_id)
        if prep_run_id
        else None
    )
    if (
        backend == "ambertools_mmpbsa"
        and prep_run_id
        and prepared_system_dir is None
    ):
        raise FileNotFoundError(
            f"MD system preparation job not found: {prep_run_id}"
        )
    staged_input = _rewrite_source_paths(source_input, source_dir)
    analysis_process_limit = min(
        max(
            1,
            int(
                source_input.get("mmpbsa_mpi_cores")
                or cpu_process_limit()
            ),
        ),
        cpu_process_limit(),
    )
    staged_input.update(
        {
            "mmgbsa_enabled": True,
            "mmgbsa_backend": backend,
            "mmgbsa_start_pct": float(start_pct),
            "mmgbsa_end_pct": float(end_pct),
            "mmgbsa_stride": int(stride),
            "mmpbsa_use_mpi": backend == "ambertools_mmpbsa",
            "mmpbsa_mpi_cores": analysis_process_limit,
        }
    )
    _write_json(run_dir / "source_input.json", staged_input)
    _write_json(run_dir / "source_result.json", _rewrite_source_paths(source_result, source_dir))
    command = _mmgbsa_command(
        run_dir,
        source_dir,
        prepared_system_dir=prepared_system_dir,
        image=image,
        use_gpu=selected_use_gpu,
        gpu_device=gpu_device,
        start_pct=float(start_pct),
        end_pct=float(end_pct),
        stride=int(stride),
        backend=backend,
        engine=source_engine,
    )
    selected_gpu_ids = (
        []
        if not selected_use_gpu
        or str(gpu_device).strip().lower() in {"all", "auto", "automatic"}
        else [int(str(gpu_device).strip().lower().removeprefix("device=").removeprefix("gpu "))]
    )
    resources = _worker_resources(
        image,
        selected_use_gpu,
        engine=source_engine,
    )
    resources["cpu_threads"] = analysis_process_limit
    if selected_gpu_ids:
        resources["gpu_ids"] = selected_gpu_ids
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "portability_schema_version": JOB_PORTABILITY_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "md_mmgbsa",
        "workflow": "md_mmgbsa",
        "status": "queued",
        "parent_run_id": source_job.run_id,
        "source_production_run_id": source_job.run_id,
        "repeat_group_id": source_job.metadata.get("repeat_group_id"),
        "repeat_index": source_job.metadata.get("repeat_index"),
        **_inherited_md_target_metadata(source_job.metadata),
        "docker_image": image,
        "md_engine": source_engine,
        "use_gpu": selected_use_gpu,
        "gpu_queued": selected_use_gpu,
        "gpu_device": str(gpu_device),
        "parameters": {
            "start_pct": float(start_pct),
            "end_pct": float(end_pct),
            "stride": int(stride),
            "backend": backend,
            "cpu_process_limit": staged_input["mmpbsa_mpi_cores"],
        },
        "resources": resources,
        "queued_at": now,
        "queued_command": command,
        "worker_finalizer": "md_mmgbsa",
        "created_at": now,
        "updated_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "source_production_run_id": source_job.run_id,
            "source_artifacts": [artifact.to_dict() for artifact in source_artifacts],
            "parameters": {
                "start_pct": float(start_pct),
                "end_pct": float(end_pct),
                "stride": int(stride),
                "backend": backend,
                "gpu_device": str(gpu_device),
                "cpu_process_limit": staged_input[
                    "mmpbsa_mpi_cores"
                ],
            },
        },
    )
    write_artifact_manifest(run_dir, [])
    write_registered_command_record(
        run_dir,
        tool_id=md_engine_spec(source_engine).tool_id,
        commands=(command,),
        image=image,
        selected_gpu_ids=selected_gpu_ids,
    )
    assert_job_portable(run_dir)
    job = JobRecord.load(run_dir, task_group=MMGBSA_TASK_GROUP)
    if attach_to_workflow and source_job.workflow_id:
        replica_index = int(source_job.metadata.get("repeat_index") or 1)
        attach_workflow_child(
            source_job.workflow_id,
            job,
            step_id=f"endpoint_energy_replica_{replica_index}",
            depends_on=(source_job.run_id,),
            replace_step=True,
        )
        job = JobRecord.load(run_dir, task_group=MMGBSA_TASK_GROUP)
    return job


def _normalize_analysis_paths(value: Any) -> Any:
    """Make container paths portable while preserving them in native_result.json."""
    if isinstance(value, dict):
        return {key: _normalize_analysis_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_analysis_paths(item) for item in value]
    if not isinstance(value, str):
        return value
    for prefix in ("/output/", "/source/", "/mn-ligand/"):
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return value


def finalize_mmgbsa_analysis_job(run_dir: Path, *, returncode: int) -> JobRecord:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    native_result = _read_json(run_dir / "result.json")
    native_result_path = run_dir / "native_result.json"
    _write_json(native_result_path, native_result)
    native_mmgbsa = (
        native_result.get("mmgbsa") if isinstance(native_result.get("mmgbsa"), dict) else {}
    )
    mmgbsa = _normalize_analysis_paths(native_mmgbsa)
    success = returncode == 0 and str(native_mmgbsa.get("status") or "") == "success"
    stderr = (run_dir / "stderr.log").read_text(errors="replace") if (
        run_dir / "stderr.log"
    ).is_file() else ""
    error = "" if success else str(
        mmgbsa.get("error") or native_result.get("error") or stderr[-4000:] or "MM/GBSA failed"
    )
    summary_path = run_dir / "mmgbsa_summary.json"
    _write_json(summary_path, mmgbsa or {"status": "failed", "error": error})
    artifacts = [
        ArtifactRef.from_path(run_dir, summary_path, "endpoint_energy", role="summary"),
        ArtifactRef.from_path(run_dir, native_result_path, "native_output", role="result"),
    ]
    endpoint_files = {
        *run_dir.glob("mmgbsa_*"),
        *run_dir.glob("g_mmpbsa*"),
    }
    for path in sorted(endpoint_files):
        if path.is_file() and path != summary_path:
            artifacts.append(
                ArtifactRef.from_path(run_dir, path, "endpoint_energy", role=path.stem)
            )
    write_artifact_manifest(run_dir, artifacts)
    result = {
        "success": success,
        "returncode": returncode,
        "source_production_run_id": metadata.get("source_production_run_id"),
        "mmgbsa": mmgbsa or {"status": "failed", "error": error},
        "error": error,
    }
    _write_json(run_dir / "result.json", result)
    completed = _utc_now_iso()
    metadata.update(
        {
            "status": "completed" if success else "failed",
            "completed_at": completed,
            "updated_at": completed,
        }
    )
    if error:
        metadata["error"] = error
    _write_json(metadata_path, metadata)
    return JobRecord.load(run_dir, task_group=MMGBSA_TASK_GROUP)


def list_mmgbsa_analysis_jobs(production_run_id: str = "") -> list[JobRecord]:
    root = runs_root() / MMGBSA_TASK_GROUP
    if not root.is_dir():
        return []
    jobs = [
        JobRecord.load(path, task_group=MMGBSA_TASK_GROUP)
        for path in root.iterdir()
        if path.is_dir()
    ]
    if production_run_id:
        jobs = [
            job
            for job in jobs
            if str(job.metadata.get("source_production_run_id") or "") == production_run_id
        ]
    return sorted(jobs, key=lambda job: (job.created_at, job.run_id), reverse=True)


def _series_summary(values: list[float]) -> dict[str, float | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "mean": mean(clean) if clean else None,
        "sample_sd": stdev(clean) if len(clean) > 1 else None,
        "count": len(clean),
    }


def _ligand_depiction(production_jobs: list[JobRecord]) -> dict[str, Any]:
    for production in production_jobs:
        candidate_roots = [production.run_dir]
        preparation_run_id = str(
            production.metadata.get("md_system_prep_run_id")
            or production.metadata.get("parent_run_id")
            or ""
        )
        if preparation_run_id:
            preparation_dir = (
                runs_root() / "md-system-prep" / preparation_run_id
            )
            if preparation_dir.is_dir():
                candidate_roots.append(preparation_dir)
        candidates = tuple(
            path
            for root in candidate_roots
            for path in (
                root / "source_ligand_refined.sdf",
                root / "ligand.sdf",
                root / "ambertools_topology" / "ligand.mol2",
            )
        )
        ligand_path = next((path for path in candidates if path.is_file()), None)
        if ligand_path is None:
            continue
        if ligand_path.suffix.lower() == ".mol2":
            molecule = Chem.MolFromMol2File(
                str(ligand_path), removeHs=True, sanitize=True
            )
        else:
            supplier = Chem.SDMolSupplier(
                str(ligand_path), removeHs=True, sanitize=True
            )
            molecule = next(
                (item for item in supplier if item is not None),
                None,
            )
        if molecule is None:
            continue
        rdDepictor.Compute2DCoords(molecule)
        conformer = molecule.GetConformer()
        coordinates = np.asarray(
            [
                [
                    float(conformer.GetAtomPosition(index).x),
                    float(conformer.GetAtomPosition(index).y),
                ]
                for index in range(molecule.GetNumAtoms())
            ],
            dtype=float,
        )
        coordinates -= np.mean(coordinates, axis=0)
        extent = float(np.max(np.ptp(coordinates, axis=0)))
        if extent > 0:
            coordinates /= extent
        element_counts: Counter[str] = Counter()
        atom_names = []
        for atom in molecule.GetAtoms():
            element_counts[atom.GetSymbol()] += 1
            atom_names.append(
                f"{atom.GetSymbol()}{element_counts[atom.GetSymbol()]}"
            )
        return {
            "atoms": [
                {
                    "name": atom_names[atom.GetIdx()],
                    "symbol": atom.GetSymbol(),
                    "formal_charge": int(atom.GetFormalCharge()),
                    "x": round(float(coordinates[atom.GetIdx(), 0]), 5),
                    "y": round(float(coordinates[atom.GetIdx(), 1]), 5),
                }
                for atom in molecule.GetAtoms()
            ],
            "bonds": [
                {
                    "begin": bond.GetBeginAtomIdx(),
                    "end": bond.GetEndAtomIdx(),
                    "order": float(bond.GetBondTypeAsDouble()),
                }
                for bond in molecule.GetBonds()
            ],
            "source": ligand_path.name,
        }
    return {}


def _normalized_secondary_structure(
    secondary_structure: dict[str, Any],
) -> dict[str, list[float]]:
    fields = ("helix_fraction", "sheet_fraction", "coil_fraction")
    values = {
        field: [float(value) for value in secondary_structure.get(field) or []]
        for field in fields
    }
    populated = [series for series in values.values() if series]
    if not populated:
        return values
    length = min(len(series) for series in populated)
    for index in range(length):
        total = sum(values[field][index] for field in fields)
        if total > 0.0:
            for field in fields:
                values[field][index] /= total
    return values


def _production_performance(
    production: JobRecord,
    result: dict[str, Any],
    analytics: dict[str, Any],
) -> dict[str, Any]:
    performance = (
        analytics.get("performance")
        if isinstance(analytics.get("performance"), dict)
        else {}
    )
    if performance.get("ns_per_day") is not None:
        return performance
    md_result = (
        result.get("md_result")
        if isinstance(result.get("md_result"), dict)
        else {}
    )
    engine = normalize_md_engine(
        md_result.get("engine")
        or result.get("engine")
        or production.metadata.get("engine")
    )
    if engine == GROMACS_ENGINE:
        from mn_ligand.workflows.gromacs_md import parse_gromacs_performance

        return parse_gromacs_performance(
            production.run_dir / "production.log"
        )
    return performance


def _required_md_hypothesis_interactions(
    analysis_job: JobRecord,
) -> list[dict[str, Any]]:
    """Load mandatory selection-hypothesis rows for an MD workflow."""

    workflow_id = str(
        analysis_job.workflow_parent_run_id
        or analysis_job.metadata.get("workflow_id")
        or ""
    )
    if not workflow_id:
        return []
    workflow_metadata_path = (
        runs_root(create=False)
        / "workflows"
        / workflow_id
        / "metadata.json"
    )
    workflow_metadata = _read_json(workflow_metadata_path)
    parameters = (
        workflow_metadata.get("parameters")
        if isinstance(workflow_metadata.get("parameters"), dict)
        else {}
    )
    dataset_id = str(parameters.get("complex_dataset_run_id") or "")
    if not dataset_id:
        return []
    dataset_metadata = _read_json(
        runs_root(create=False)
        / "complex-datasets"
        / dataset_id
        / "metadata.json"
    )
    provenance = (
        dataset_metadata.get("selection_provenance")
        if isinstance(dataset_metadata.get("selection_provenance"), dict)
        else {}
    )
    definition = provenance.get("hypothesis_definition") or []
    required = [
        row
        for row in definition
        if isinstance(row, dict)
        and bool(row.get("Enabled", True))
        and bool(row.get("Required"))
    ]
    if required:
        return required

    # Legacy complex datasets sometimes encoded the mandatory residue only in
    # their immutable dataset name (for example ``ASN331man``) while the copied
    # hypothesis table accidentally left Required=false.  Recover only an
    # unambiguous residue named immediately before ``man``/``mand``.
    dataset_name = str(
        dataset_metadata.get("dataset_name")
        or parameters.get("complex_dataset_name")
        or ""
    )
    legacy_residues = {
        match.group(1).upper()
        for match in re.finditer(
            r"\b([A-Z][A-Z0-9]{2}-?\d+)man",
            dataset_name,
        )
    }
    if len(legacy_residues) != 1:
        return []
    legacy_residue = next(iter(legacy_residues))
    recovered = []
    for row in definition:
        if not isinstance(row, dict) or not bool(row.get("Enabled", True)):
            continue
        residue_text = str(row.get("Protein residue") or "")
        if residue_text.rsplit(":", 1)[-1].upper() == legacy_residue:
            recovered_row = dict(row)
            recovered_row["Required"] = True
            recovered_row["Required source"] = "legacy dataset name"
            recovered.append(recovered_row)
    return recovered if len(recovered) == 1 else []


def _required_interaction_consensus(
    definitions: list[dict[str, Any]],
    contact_consensus: list[dict[str, Any]],
    contact_matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    """Match mandatory hypothesis rows to their MD occupancy measurements."""

    interaction_stems = {
        "contact": "contact",
        "direct contact": "contact",
        "hydrogen bond": "hydrogen_bond",
        "hydrophobic contact": "hydrophobic",
        "water bridge": "water_bridge",
        "salt bridge": "salt_bridge",
    }
    consensus_by_residue = {
        str(row.get("residue") or ""): row
        for row in contact_consensus
        if isinstance(row, dict)
    }
    matrix_residues = [str(value) for value in contact_matrix.get("residues") or []]
    output: list[dict[str, Any]] = []
    for definition in definitions:
        interaction = str(definition.get("Interaction") or "").strip().lower()
        stem = interaction_stems.get(interaction)
        residue_match = re.fullmatch(
            r"([^:]+):([A-Za-z]{3})(-?\d+)([A-Za-z]?)",
            str(definition.get("Protein residue") or "").strip(),
        )
        if stem is None or residue_match is None:
            continue
        chain, residue_name, residue_number, insertion_code = residue_match.groups()
        residue = (
            f"{residue_name.upper()}{residue_number}{insertion_code}"
            f" · chain {chain}"
        )
        region = str(definition.get("Protein region") or "").strip().upper()
        scope = "backbone" if region == "BB" else "sidechain" if region == "SC" else ""
        scoped_stem = f"{stem}_{scope}" if scope else stem
        consensus = consensus_by_residue.get(residue, {})
        replica_values: list[float] = []
        if residue in matrix_residues:
            residue_index = matrix_residues.index(residue)
            matrix_values = contact_matrix.get(f"{scoped_stem}_occupancy") or []
            if residue_index < len(matrix_values):
                replica_values = [
                    float(value) for value in matrix_values[residue_index]
                ]
        summary = _series_summary(replica_values)
        mean_value = consensus.get(f"mean_{scoped_stem}_occupancy")
        sd_value = consensus.get(f"sample_sd_{scoped_stem}_occupancy")
        output.append(
            {
                "interaction": interaction,
                "residue": residue,
                "protein_region": region or "BB+SC",
                "label": (
                    f"{interaction} · {residue_name.upper()}"
                    f"{residue_number}{insertion_code} ({region or 'BB+SC'})"
                ),
                "mean_occupancy": (
                    float(mean_value)
                    if mean_value is not None
                    else summary["mean"]
                ),
                "sample_sd_occupancy": (
                    float(sd_value)
                    if sd_value is not None
                    else summary["sample_sd"]
                ),
                "replica_occupancy": replica_values,
                "replica_count": len(replica_values),
                "reference_support": definition.get("Reference support"),
                "requirement_group": str(
                    definition.get("Requirement group") or ""
                ),
                "requirement_logic": str(
                    definition.get("Requirement logic") or "ALL"
                ),
            }
        )
    return output


def _complete_analysis(
    job: JobRecord,
    production_jobs: list[JobRecord],
    endpoint_jobs: list[JobRecord] | None = None,
    *,
    residue_mapping_override: dict[str, Any] | None = None,
) -> None:
    analysis_input = _read_json(job.run_dir / "input.json")
    analysis_mapping = (
        analysis_input.get("residue_mapping")
        if isinstance(analysis_input.get("residue_mapping"), dict)
        else None
    )
    aggregate_mapping: dict[str, Any] | None = (
        residue_mapping_override or analysis_mapping
    )
    endpoint_by_source = {
        str(endpoint.metadata.get("source_production_run_id") or ""): endpoint
        for endpoint in (endpoint_jobs or [])
    }
    rows: list[dict[str, Any]] = []
    replica_series: list[dict[str, Any]] = []
    contact_replica_maps: list[dict[str, float]] = []
    contact_backbone_replica_maps: list[dict[str, float]] = []
    contact_sidechain_replica_maps: list[dict[str, float]] = []
    hbond_replica_maps: list[dict[str, float]] = []
    hbond_backbone_replica_maps: list[dict[str, float]] = []
    hbond_sidechain_replica_maps: list[dict[str, float]] = []
    hydrophobic_replica_maps: list[dict[str, float]] = []
    hydrophobic_backbone_replica_maps: list[dict[str, float]] = []
    hydrophobic_sidechain_replica_maps: list[dict[str, float]] = []
    water_bridge_replica_maps: list[dict[str, float]] = []
    water_bridge_backbone_replica_maps: list[dict[str, float]] = []
    water_bridge_sidechain_replica_maps: list[dict[str, float]] = []
    salt_bridge_replica_maps: list[dict[str, float]] = []
    salt_bridge_backbone_replica_maps: list[dict[str, float]] = []
    salt_bridge_sidechain_replica_maps: list[dict[str, float]] = []
    importance_replica_maps: list[dict[str, float]] = []
    importance_backbone_replica_maps: list[dict[str, float]] = []
    importance_sidechain_replica_maps: list[dict[str, float]] = []
    distance_replica_maps: list[dict[str, float]] = []
    ligand_atom_replica_maps: list[dict[str, str]] = []
    salt_bridge_applicability: list[bool] = []
    interface_rin_replicas: list[dict[str, Any]] = []
    rmsf_by_residue: dict[str, list[float]] = {}
    ligand_rmsf_by_atom: dict[str, list[float]] = {}
    for production in production_jobs:
        result = _read_json(production.run_dir / "result.json")
        analytics = (
            (result.get("md_result") or {}).get("analytics") or {}
            if isinstance(result.get("md_result"), dict)
            else {}
        )
        rmsd = analytics.get("rmsd") if isinstance(analytics.get("rmsd"), dict) else {}
        performance = _production_performance(production, result, analytics)
        structural = (
            analytics.get("structural_dynamics")
            if isinstance(analytics.get("structural_dynamics"), dict)
            else {}
        )
        production_input = _read_json(production.run_dir / "input.json")
        production_mapping = residue_mapping_override or (
            production_input.get("residue_mapping")
            if isinstance(production_input.get("residue_mapping"), dict)
            else analysis_mapping
        )
        if isinstance(production_mapping, dict) and production_mapping.get(
            "residues"
        ):
            aggregate_mapping = aggregate_mapping or production_mapping
            numbering = structural.get("residue_numbering")
            if not (
                isinstance(numbering, dict)
                and numbering.get("scheme") == "source_author"
            ):
                # The aggregate is a new derived result. Relabel its in-memory
                # copy without rewriting immutable production results.
                structural = json.loads(json.dumps(structural))
                relabel_structural_dynamics(structural, production_mapping)
        pocket = (
            structural.get("pocket")
            if isinstance(structural.get("pocket"), dict)
            else {}
        )
        contacts = (
            structural.get("contacts")
            if isinstance(structural.get("contacts"), dict)
            else {}
        )
        rmsf = (
            structural.get("rmsf")
            if isinstance(structural.get("rmsf"), dict)
            else {}
        )
        ligand_rmsf = (
            structural.get("ligand_rmsf")
            if isinstance(structural.get("ligand_rmsf"), dict)
            else {}
        )
        radius_of_gyration = (
            structural.get("radius_of_gyration")
            if isinstance(structural.get("radius_of_gyration"), dict)
            else {}
        )
        secondary_structure = (
            _normalized_secondary_structure(structural.get("secondary_structure"))
            if isinstance(structural.get("secondary_structure"), dict)
            else {}
        )
        analysis_window = (
            structural.get("analysis_window")
            if isinstance(structural.get("analysis_window"), dict)
            else {}
        )
        interface_rin = (
            structural.get("interface_rin")
            if isinstance(structural.get("interface_rin"), dict)
            else {}
        )
        salt_bridges = (
            structural.get("salt_bridges")
            if isinstance(structural.get("salt_bridges"), dict)
            else {}
        )
        salt_bridge_applicability.append(
            bool(salt_bridges.get("applicable"))
        )
        rmsd_window = (
            rmsd.get("analysis_window")
            if isinstance(rmsd.get("analysis_window"), dict)
            else {}
        )
        backbone_values = [
            float(value)
            for value in rmsd.get("backbone_rmsd_angstrom") or []
        ]
        ligand_values = [
            float(value)
            for value in rmsd.get("ligand_rmsd_angstrom") or []
        ]
        endpoint = endpoint_by_source.get(production.run_id)
        endpoint_result = endpoint.result if endpoint is not None else {}
        mmgbsa = (
            endpoint_result.get("mmgbsa")
            if isinstance(endpoint_result.get("mmgbsa"), dict)
            else result.get("mmgbsa")
            if isinstance(result.get("mmgbsa"), dict)
            else {}
        )
        delta = (
            mmgbsa.get("delta")
            if isinstance(mmgbsa.get("delta"), dict)
            else {}
        )
        gb_delta = (
            (mmgbsa.get("gb") or {}).get("delta")
            if isinstance(mmgbsa.get("gb"), dict)
            and isinstance((mmgbsa.get("gb") or {}).get("delta"), dict)
            else delta
        )
        pb_delta = (
            (mmgbsa.get("pb") or {}).get("delta")
            if isinstance(mmgbsa.get("pb"), dict)
            and isinstance((mmgbsa.get("pb") or {}).get("delta"), dict)
            else {}
        )
        tail_backbone = backbone_values[-max(1, len(backbone_values) // 5):]
        tail_ligand = ligand_values[-max(1, len(ligand_values) // 5):]
        replica_index = int(
            production.metadata.get("repeat_index") or len(rows) + 1
        )
        rows.append(
            {
                "run_id": production.run_id,
                "replica": replica_index,
                "status": production.status,
                "success": result.get("success") is True,
                "analysis_start_ns": (
                    float(analysis_window["start_ps"]) / 1000.0
                    if analysis_window.get("start_ps") is not None
                    else 0.0
                ),
                "rmsd_reference_time_ns": (
                    float(rmsd_window["reference_time_ps"]) / 1000.0
                    if rmsd_window.get("reference_time_ps") is not None
                    else None
                ),
                "rmsd_reference_source_frame": rmsd_window.get(
                    "reference_source_frame_index"
                ),
                "rmsd_reference_coordinates": rmsd_window.get(
                    "reference_coordinates"
                ),
                "performance_ns_per_day": performance.get("ns_per_day"),
                "backbone_rmsd_tail_mean_angstrom": (
                    mean(tail_backbone) if tail_backbone else None
                ),
                "ligand_rmsd_tail_mean_angstrom": (
                    mean(tail_ligand) if tail_ligand else None
                ),
                "backbone_rmsd_max_angstrom": (
                    max(backbone_values) if backbone_values else None
                ),
                "ligand_rmsd_max_angstrom": (
                    max(ligand_values) if ligand_values else None
                ),
                "reference_site_retained_fraction": pocket.get(
                    "retained_fraction"
                ),
                "ligand_centroid_displacement_max_angstrom": (
                    max(
                        float(value)
                        for value in pocket.get(
                            "ligand_centroid_displacement_angstrom"
                        )
                        or []
                    )
                    if pocket.get("ligand_centroid_displacement_angstrom")
                    else None
                ),
                "endpoint_job_id": endpoint.run_id if endpoint is not None else "",
                "endpoint_status": endpoint.status if endpoint is not None else (
                    str(mmgbsa.get("status") or "not_requested")
                ),
                "endpoint_frames": (
                    (mmgbsa.get("metadata") or {}).get("n_frames_analyzed")
                    if isinstance(mmgbsa.get("metadata"), dict)
                    else None
                ),
                "delta_g_bind_kcal_mol": gb_delta.get(
                    "delta_g_bind_total_kcal_mol"
                ),
                "delta_mm_kcal_mol": gb_delta.get("delta_mm_kcal_mol"),
                "delta_gbsa_kcal_mol": gb_delta.get(
                    "delta_gbsa_kcal_mol"
                ),
                "delta_nonpolar_kcal_mol": gb_delta.get(
                    "delta_nonpolar_kcal_mol"
                ),
                "pb_delta_g_bind_kcal_mol": pb_delta.get(
                    "delta_g_bind_total_kcal_mol"
                ),
                "pb_delta_mm_kcal_mol": pb_delta.get(
                    "delta_mm_kcal_mol"
                ),
                "pb_delta_pbsa_kcal_mol": pb_delta.get(
                    "delta_gbsa_kcal_mol"
                ),
                "pb_delta_nonpolar_kcal_mol": pb_delta.get(
                    "delta_nonpolar_kcal_mol"
                ),
            }
        )
        structural_time = [
            float(value) / 1000.0
            for value in structural.get("time_ps") or []
        ]
        replica_series.append(
            {
                "replica": replica_index,
                "time_ns": structural_time,
                "backbone_rmsd_angstrom": backbone_values,
                "ligand_rmsd_angstrom": ligand_values,
                "ligand_centroid_displacement_angstrom": [
                    float(value)
                    for value in pocket.get(
                        "ligand_centroid_displacement_angstrom"
                    )
                    or []
                ],
                "minimum_protein_distance_angstrom": [
                    float(value)
                    for value in pocket.get(
                        "minimum_protein_distance_angstrom"
                    )
                    or []
                ],
                "contact_distance_series_angstrom": {
                    str(label): [
                        float(value) for value in values
                    ]
                    for label, values in (
                        contacts.get("distance_series") or {}
                    ).items()
                },
                "contact_cutoff_angstrom": pocket.get(
                    "contact_cutoff_angstrom"
                ),
                "protein_rg_angstrom": [
                    float(value)
                    for value in radius_of_gyration.get(
                        "protein_angstrom"
                    )
                    or []
                ],
                "ligand_rg_angstrom": [
                    float(value)
                    for value in radius_of_gyration.get(
                        "ligand_angstrom"
                    )
                    or []
                ],
                "complex_rg_angstrom": [
                    float(value)
                    for value in radius_of_gyration.get(
                        "complex_angstrom"
                    )
                    or []
                ],
                "helix_fraction": [
                    float(value)
                    for value in secondary_structure.get(
                        "helix_fraction"
                    )
                    or []
                ],
                "sheet_fraction": [
                    float(value)
                    for value in secondary_structure.get(
                        "sheet_fraction"
                    )
                    or []
                ],
                "coil_fraction": [
                    float(value)
                    for value in secondary_structure.get(
                        "coil_fraction"
                    )
                    or []
                ],
                "protein_rmsf_residues": [
                    str(value) for value in rmsf.get("residues") or []
                ],
                "protein_rmsf_angstrom": [
                    float(value)
                    for value in rmsf.get("ca_rmsf_angstrom") or []
                ],
                "ligand_rmsf_atoms": [
                    str(value) for value in ligand_rmsf.get("atoms") or []
                ],
                "ligand_rmsf_angstrom": [
                    float(value)
                    for value in ligand_rmsf.get("rmsf_angstrom") or []
                ],
            }
        )
        contact_map: dict[str, float] = {}
        contact_backbone_map: dict[str, float] = {}
        contact_sidechain_map: dict[str, float] = {}
        hbond_map: dict[str, float] = {}
        hbond_backbone_map: dict[str, float] = {}
        hbond_sidechain_map: dict[str, float] = {}
        hydrophobic_map: dict[str, float] = {}
        hydrophobic_backbone_map: dict[str, float] = {}
        hydrophobic_sidechain_map: dict[str, float] = {}
        water_bridge_map: dict[str, float] = {}
        water_bridge_backbone_map: dict[str, float] = {}
        water_bridge_sidechain_map: dict[str, float] = {}
        salt_bridge_map: dict[str, float] = {}
        salt_bridge_backbone_map: dict[str, float] = {}
        salt_bridge_sidechain_map: dict[str, float] = {}
        importance_map: dict[str, float] = {}
        importance_backbone_map: dict[str, float] = {}
        importance_sidechain_map: dict[str, float] = {}
        distance_map: dict[str, float] = {}
        ligand_atom_map: dict[str, str] = {}
        for contact in contacts.get("residues") or []:
            if not isinstance(contact, dict) or not contact.get("residue"):
                continue
            label = str(contact["residue"])
            contact_map[label] = float(
                contact.get("contact_occupancy") or 0.0
            )
            if contact.get("contact_backbone_occupancy") is not None:
                contact_backbone_map[label] = float(
                    contact["contact_backbone_occupancy"]
                )
            if contact.get("contact_sidechain_occupancy") is not None:
                contact_sidechain_map[label] = float(
                    contact["contact_sidechain_occupancy"]
                )
            hbond_map[label] = float(
                contact.get("hydrogen_bond_occupancy") or 0.0
            )
            if contact.get("hydrogen_bond_backbone_occupancy") is not None:
                hbond_backbone_map[label] = float(
                    contact["hydrogen_bond_backbone_occupancy"]
                )
            if contact.get("hydrogen_bond_sidechain_occupancy") is not None:
                hbond_sidechain_map[label] = float(
                    contact["hydrogen_bond_sidechain_occupancy"]
                )
            if contact.get("hydrophobic_occupancy") is not None:
                hydrophobic_map[label] = float(
                    contact["hydrophobic_occupancy"]
                )
            if contact.get("hydrophobic_backbone_occupancy") is not None:
                hydrophobic_backbone_map[label] = float(
                    contact["hydrophobic_backbone_occupancy"]
                )
            if contact.get("hydrophobic_sidechain_occupancy") is not None:
                hydrophobic_sidechain_map[label] = float(
                    contact["hydrophobic_sidechain_occupancy"]
                )
            if contact.get("water_bridge_occupancy") is not None:
                water_bridge_map[label] = float(
                    contact["water_bridge_occupancy"]
                )
            if contact.get("water_bridge_backbone_occupancy") is not None:
                water_bridge_backbone_map[label] = float(
                    contact["water_bridge_backbone_occupancy"]
                )
            if contact.get("water_bridge_sidechain_occupancy") is not None:
                water_bridge_sidechain_map[label] = float(
                    contact["water_bridge_sidechain_occupancy"]
                )
            if contact.get("salt_bridge_occupancy") is not None:
                salt_bridge_map[label] = float(
                    contact["salt_bridge_occupancy"]
                )
            if contact.get("salt_bridge_backbone_occupancy") is not None:
                salt_bridge_backbone_map[label] = float(
                    contact["salt_bridge_backbone_occupancy"]
                )
            if contact.get("salt_bridge_sidechain_occupancy") is not None:
                salt_bridge_sidechain_map[label] = float(
                    contact["salt_bridge_sidechain_occupancy"]
                )
            if contact.get("binding_importance_score") is not None:
                importance_map[label] = float(
                    contact["binding_importance_score"]
                )
            if contact.get("binding_importance_backbone_score") is not None:
                importance_backbone_map[label] = float(
                    contact["binding_importance_backbone_score"]
                )
            if contact.get("binding_importance_sidechain_score") is not None:
                importance_sidechain_map[label] = float(
                    contact["binding_importance_sidechain_score"]
                )
            distance_map[label] = float(
                contact.get("mean_minimum_distance_angstrom") or 0.0
            )
            ligand_atom_map[label] = str(
                contact.get("top_ligand_atom") or ""
            )
        contact_replica_maps.append(contact_map)
        contact_backbone_replica_maps.append(contact_backbone_map)
        contact_sidechain_replica_maps.append(contact_sidechain_map)
        hbond_replica_maps.append(hbond_map)
        hbond_backbone_replica_maps.append(hbond_backbone_map)
        hbond_sidechain_replica_maps.append(hbond_sidechain_map)
        hydrophobic_replica_maps.append(hydrophobic_map)
        hydrophobic_backbone_replica_maps.append(hydrophobic_backbone_map)
        hydrophobic_sidechain_replica_maps.append(hydrophobic_sidechain_map)
        water_bridge_replica_maps.append(water_bridge_map)
        water_bridge_backbone_replica_maps.append(water_bridge_backbone_map)
        water_bridge_sidechain_replica_maps.append(water_bridge_sidechain_map)
        salt_bridge_replica_maps.append(salt_bridge_map)
        salt_bridge_backbone_replica_maps.append(salt_bridge_backbone_map)
        salt_bridge_sidechain_replica_maps.append(salt_bridge_sidechain_map)
        importance_replica_maps.append(importance_map)
        importance_backbone_replica_maps.append(importance_backbone_map)
        importance_sidechain_replica_maps.append(importance_sidechain_map)
        distance_replica_maps.append(distance_map)
        ligand_atom_replica_maps.append(ligand_atom_map)
        if interface_rin.get("applicable"):
            interface_rin_replicas.append(
                {
                    "replica": int(
                        production.metadata.get("repeat_index") or 0
                    )
                    + 1,
                    **interface_rin,
                }
            )
        for label, value in zip(
            rmsf.get("residues") or [],
            rmsf.get("ca_rmsf_angstrom") or [],
        ):
            rmsf_by_residue.setdefault(str(label), []).append(float(value))
        for atom, value in zip(
            ligand_rmsf.get("atoms") or [],
            ligand_rmsf.get("rmsf_angstrom") or [],
        ):
            ligand_rmsf_by_atom.setdefault(str(atom), []).append(
                float(value)
            )
    aggregate = {
        key: _series_summary(
            [
                float(row[key])
                for row in rows
                if row.get(key) not in (None, "")
            ]
        )
        for key in (
            "backbone_rmsd_tail_mean_angstrom",
            "ligand_rmsd_tail_mean_angstrom",
            "reference_site_retained_fraction",
            "ligand_centroid_displacement_max_angstrom",
            "performance_ns_per_day",
            "delta_g_bind_kcal_mol",
            "delta_mm_kcal_mol",
            "delta_gbsa_kcal_mol",
            "delta_nonpolar_kcal_mol",
            "pb_delta_g_bind_kcal_mol",
            "pb_delta_mm_kcal_mol",
            "pb_delta_pbsa_kcal_mol",
            "pb_delta_nonpolar_kcal_mol",
        )
    }
    all_contact_labels = {
        label for mapping in contact_replica_maps for label in mapping
    }
    contact_by_residue = {
        label: [
            mapping.get(label, 0.0) for mapping in contact_replica_maps
        ]
        for label in all_contact_labels
    }
    contact_backbone_by_residue = {
        label: [
            mapping[label]
            for mapping in contact_backbone_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    contact_sidechain_by_residue = {
        label: [
            mapping[label]
            for mapping in contact_sidechain_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    hbond_by_residue = {
        label: [
            mapping.get(label, 0.0) for mapping in hbond_replica_maps
        ]
        for label in all_contact_labels
    }
    hbond_backbone_by_residue = {
        label: [
            mapping[label]
            for mapping in hbond_backbone_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    hbond_sidechain_by_residue = {
        label: [
            mapping[label]
            for mapping in hbond_sidechain_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    hydrophobic_by_residue = {
        label: [
            mapping[label]
            for mapping in hydrophobic_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    hydrophobic_backbone_by_residue = {
        label: [
            mapping[label]
            for mapping in hydrophobic_backbone_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    hydrophobic_sidechain_by_residue = {
        label: [
            mapping[label]
            for mapping in hydrophobic_sidechain_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    water_bridge_by_residue = {
        label: [
            mapping[label]
            for mapping in water_bridge_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    water_bridge_backbone_by_residue = {
        label: [
            mapping[label]
            for mapping in water_bridge_backbone_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    water_bridge_sidechain_by_residue = {
        label: [
            mapping[label]
            for mapping in water_bridge_sidechain_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    salt_bridge_by_residue = {
        label: [
            mapping[label]
            for mapping in salt_bridge_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    salt_bridge_backbone_by_residue = {
        label: [
            mapping[label]
            for mapping in salt_bridge_backbone_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    salt_bridge_sidechain_by_residue = {
        label: [
            mapping[label]
            for mapping in salt_bridge_sidechain_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    importance_by_residue = {
        label: [
            mapping[label]
            for mapping in importance_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    importance_backbone_by_residue = {
        label: [
            mapping[label]
            for mapping in importance_backbone_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    importance_sidechain_by_residue = {
        label: [
            mapping[label]
            for mapping in importance_sidechain_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    distance_by_residue = {
        label: [
            mapping[label]
            for mapping in distance_replica_maps
            if label in mapping
        ]
        for label in all_contact_labels
    }
    contact_consensus = []
    for label in sorted(
        contact_by_residue,
        key=lambda value: mean(contact_by_residue[value]),
        reverse=True,
    ):
        contact_summary = _series_summary(contact_by_residue[label])
        contact_backbone_summary = _series_summary(
            contact_backbone_by_residue.get(label, [])
        )
        contact_sidechain_summary = _series_summary(
            contact_sidechain_by_residue.get(label, [])
        )
        hbond_summary = _series_summary(hbond_by_residue.get(label, []))
        hbond_backbone_summary = _series_summary(
            hbond_backbone_by_residue.get(label, [])
        )
        hbond_sidechain_summary = _series_summary(
            hbond_sidechain_by_residue.get(label, [])
        )
        hydrophobic_summary = _series_summary(
            hydrophobic_by_residue.get(label, [])
        )
        hydrophobic_backbone_summary = _series_summary(
            hydrophobic_backbone_by_residue.get(label, [])
        )
        hydrophobic_sidechain_summary = _series_summary(
            hydrophobic_sidechain_by_residue.get(label, [])
        )
        water_bridge_summary = _series_summary(
            water_bridge_by_residue.get(label, [])
        )
        water_bridge_backbone_summary = _series_summary(
            water_bridge_backbone_by_residue.get(label, [])
        )
        water_bridge_sidechain_summary = _series_summary(
            water_bridge_sidechain_by_residue.get(label, [])
        )
        salt_bridge_summary = _series_summary(
            salt_bridge_by_residue.get(label, [])
        )
        salt_bridge_backbone_summary = _series_summary(
            salt_bridge_backbone_by_residue.get(label, [])
        )
        salt_bridge_sidechain_summary = _series_summary(
            salt_bridge_sidechain_by_residue.get(label, [])
        )
        importance_summary = _series_summary(
            importance_by_residue.get(label, [])
        )
        importance_backbone_summary = _series_summary(
            importance_backbone_by_residue.get(label, [])
        )
        importance_sidechain_summary = _series_summary(
            importance_sidechain_by_residue.get(label, [])
        )
        distance_summary = _series_summary(
            distance_by_residue.get(label, [])
        )
        ligand_atom_counts = Counter(
            mapping.get(label, "")
            for mapping in ligand_atom_replica_maps
            if mapping.get(label)
        )
        contact_consensus.append(
            {
                "residue": label,
                "mean_contact_occupancy": contact_summary["mean"],
                "sample_sd_contact_occupancy": contact_summary["sample_sd"],
                "mean_contact_backbone_occupancy": contact_backbone_summary[
                    "mean"
                ],
                "sample_sd_contact_backbone_occupancy": (
                    contact_backbone_summary["sample_sd"]
                ),
                "mean_contact_sidechain_occupancy": contact_sidechain_summary[
                    "mean"
                ],
                "sample_sd_contact_sidechain_occupancy": (
                    contact_sidechain_summary["sample_sd"]
                ),
                "mean_hydrogen_bond_occupancy": hbond_summary["mean"],
                "sample_sd_hydrogen_bond_occupancy": hbond_summary[
                    "sample_sd"
                ],
                "mean_hydrogen_bond_backbone_occupancy": (
                    hbond_backbone_summary["mean"]
                ),
                "sample_sd_hydrogen_bond_backbone_occupancy": (
                    hbond_backbone_summary["sample_sd"]
                ),
                "mean_hydrogen_bond_sidechain_occupancy": (
                    hbond_sidechain_summary["mean"]
                ),
                "sample_sd_hydrogen_bond_sidechain_occupancy": (
                    hbond_sidechain_summary["sample_sd"]
                ),
                "mean_hydrophobic_occupancy": hydrophobic_summary["mean"],
                "sample_sd_hydrophobic_occupancy": hydrophobic_summary[
                    "sample_sd"
                ],
                "mean_hydrophobic_backbone_occupancy": (
                    hydrophobic_backbone_summary["mean"]
                ),
                "sample_sd_hydrophobic_backbone_occupancy": (
                    hydrophobic_backbone_summary["sample_sd"]
                ),
                "mean_hydrophobic_sidechain_occupancy": (
                    hydrophobic_sidechain_summary["mean"]
                ),
                "sample_sd_hydrophobic_sidechain_occupancy": (
                    hydrophobic_sidechain_summary["sample_sd"]
                ),
                "mean_water_bridge_occupancy": water_bridge_summary["mean"],
                "sample_sd_water_bridge_occupancy": water_bridge_summary[
                    "sample_sd"
                ],
                "mean_water_bridge_backbone_occupancy": (
                    water_bridge_backbone_summary["mean"]
                ),
                "sample_sd_water_bridge_backbone_occupancy": (
                    water_bridge_backbone_summary["sample_sd"]
                ),
                "mean_water_bridge_sidechain_occupancy": (
                    water_bridge_sidechain_summary["mean"]
                ),
                "sample_sd_water_bridge_sidechain_occupancy": (
                    water_bridge_sidechain_summary["sample_sd"]
                ),
                "mean_salt_bridge_occupancy": salt_bridge_summary["mean"],
                "sample_sd_salt_bridge_occupancy": salt_bridge_summary[
                    "sample_sd"
                ],
                "mean_salt_bridge_backbone_occupancy": (
                    salt_bridge_backbone_summary["mean"]
                ),
                "sample_sd_salt_bridge_backbone_occupancy": (
                    salt_bridge_backbone_summary["sample_sd"]
                ),
                "mean_salt_bridge_sidechain_occupancy": (
                    salt_bridge_sidechain_summary["mean"]
                ),
                "sample_sd_salt_bridge_sidechain_occupancy": (
                    salt_bridge_sidechain_summary["sample_sd"]
                ),
                "mean_binding_importance_score": importance_summary["mean"],
                "sample_sd_binding_importance_score": importance_summary[
                    "sample_sd"
                ],
                "mean_binding_importance_backbone_score": (
                    importance_backbone_summary["mean"]
                ),
                "sample_sd_binding_importance_backbone_score": (
                    importance_backbone_summary["sample_sd"]
                ),
                "mean_binding_importance_sidechain_score": (
                    importance_sidechain_summary["mean"]
                ),
                "sample_sd_binding_importance_sidechain_score": (
                    importance_sidechain_summary["sample_sd"]
                ),
                "mean_minimum_distance_angstrom": distance_summary["mean"],
                "sample_sd_minimum_distance_angstrom": distance_summary[
                    "sample_sd"
                ],
                "top_ligand_atom": (
                    ligand_atom_counts.most_common(1)[0][0]
                    if ligand_atom_counts
                    else ""
                ),
                "replica_count": contact_summary["count"],
            }
        )
    rmsf_consensus = [
        {
            "residue": label,
            "mean_ca_rmsf_angstrom": summary["mean"],
            "sample_sd_ca_rmsf_angstrom": summary["sample_sd"],
            "replica_count": summary["count"],
        }
        for label in rmsf_by_residue
        if (summary := _series_summary(rmsf_by_residue[label]))["count"]
    ]
    ligand_rmsf_consensus = [
        {
            "atom": atom,
            "mean_rmsf_angstrom": summary["mean"],
            "sample_sd_rmsf_angstrom": summary["sample_sd"],
            "replica_count": summary["count"],
        }
        for atom in ligand_rmsf_by_atom
        if (summary := _series_summary(ligand_rmsf_by_atom[atom]))["count"]
    ]
    contact_matrix_labels = sorted(
        all_contact_labels,
        key=lambda label: mean(contact_by_residue[label]),
        reverse=True,
    )
    contact_matrix = {
        "residues": contact_matrix_labels,
        "replicas": [
            int(row.get("replica") or index + 1)
            for index, row in enumerate(rows)
        ],
        "contact_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in contact_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "contact_backbone_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in contact_backbone_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "contact_sidechain_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in contact_sidechain_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "hydrogen_bond_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in hbond_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "hydrogen_bond_backbone_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in hbond_backbone_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "hydrogen_bond_sidechain_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in hbond_sidechain_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "hydrophobic_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in hydrophobic_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "hydrophobic_backbone_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in hydrophobic_backbone_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "hydrophobic_sidechain_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in hydrophobic_sidechain_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "water_bridge_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in water_bridge_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "water_bridge_backbone_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in water_bridge_backbone_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "water_bridge_sidechain_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in water_bridge_sidechain_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "salt_bridge_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in salt_bridge_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "salt_bridge_backbone_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in salt_bridge_backbone_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "salt_bridge_sidechain_occupancy": [
            [
                float(mapping.get(label, 0.0))
                for mapping in salt_bridge_sidechain_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "binding_importance_score": [
            [
                float(mapping.get(label, 0.0))
                for mapping in importance_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "binding_importance_backbone_score": [
            [
                float(mapping.get(label, 0.0))
                for mapping in importance_backbone_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "binding_importance_sidechain_score": [
            [
                float(mapping.get(label, 0.0))
                for mapping in importance_sidechain_replica_maps
            ]
            for label in contact_matrix_labels
        ],
        "minimum_distance_angstrom": [
            [
                float(mapping.get(label, 0.0))
                for mapping in distance_replica_maps
            ]
            for label in contact_matrix_labels
        ],
    }
    report_path = job.run_dir / "replicate_summary.json"
    table_path = job.run_dir / "replicate_summary.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_id"])
        writer.writeheader()
        writer.writerows(rows)
    required_interactions = _required_md_hypothesis_interactions(job)
    required_interaction_consensus = _required_interaction_consensus(
        required_interactions,
        contact_consensus,
        contact_matrix,
    )
    report = {
        "replicas": rows,
        "replica_series": replica_series,
        "analysis_window": {
            "start_ns": min(
                (float(row["analysis_start_ns"]) for row in rows),
                default=0.0,
            ),
            "end_ns": max(
                (
                    max(series.get("time_ns") or [0.0])
                    for series in replica_series
                ),
                default=0.0,
            ),
        },
        "contact_consensus": contact_consensus,
        "contact_matrix": contact_matrix,
        "required_interaction_consensus": required_interaction_consensus,
        "salt_bridge_applicable": any(salt_bridge_applicability),
        "ligand_depiction": _ligand_depiction(production_jobs),
        "interface_rin_replicas": interface_rin_replicas,
        "rmsf_consensus": rmsf_consensus,
        "ligand_rmsf_consensus": ligand_rmsf_consensus,
        "completed": sum(item["status"] == "completed" for item in rows),
        "production_failed": sum(item["status"] != "completed" for item in rows),
        "endpoint_completed": sum(
            item["endpoint_status"] == "completed" for item in rows
        ),
        "endpoint_failed": sum(
            item["endpoint_status"] not in {"completed", "not_requested"}
            for item in rows
        ),
        "aggregate": aggregate,
    }
    if isinstance(aggregate_mapping, dict) and aggregate_mapping.get("residues"):
        report["residue_numbering"] = {
            "scheme": "source_author",
            "source_run_id": str(aggregate_mapping.get("source_run_id") or ""),
            "source_label": str(aggregate_mapping.get("source_label") or ""),
        }
    report["failed"] = report["production_failed"] + report["endpoint_failed"]
    _write_json(report_path, report)
    _write_json(job.run_dir / "result.json", {"success": report["failed"] == 0, **report})
    metadata = _read_json(job.run_dir / "metadata.json")
    target_metadata = (
        _inherited_md_target_metadata(production_jobs[0].metadata)
        if production_jobs
        else {}
    )
    metadata.update(
        {
            **target_metadata,
            "status": "completed" if report["failed"] == 0 else "failed",
            "awaiting_parent": False,
            "completed_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
    )
    _write_json(job.run_dir / "metadata.json", metadata)
    artifacts = [
            ArtifactRef.from_path(
                job.run_dir,
                report_path,
                "md_replicate_report",
                role="analysis",
            ),
            ArtifactRef.from_path(
                job.run_dir,
                table_path,
                "md_replicate_report",
                role="replicates",
            ),
        ]
    if isinstance(aggregate_mapping, dict) and aggregate_mapping.get("residues"):
        mapping_path = job.run_dir / "source_residue_mapping.json"
        _write_json(mapping_path, aggregate_mapping)
        artifacts.append(
            ArtifactRef.from_path(
                job.run_dir,
                mapping_path,
                "residue_mapping",
                role="author_numbering",
            )
        )
        metadata = _read_json(job.run_dir / "metadata.json")
        metadata["residue_numbering"] = "source_author"
        _write_json(job.run_dir / "metadata.json", metadata)
    write_artifact_manifest(job.run_dir, artifacts)


def create_md_residue_numbering_revision(workflow_id: str) -> JobRecord:
    """Publish a corrected aggregate analysis without rewriting MD replicas.

    This is intended for completed workflows launched from legacy targets that
    did not yet publish a typed residue mapping. Their preparation lineage is
    traced to the immutable imported structure, so intermediate renumbering is
    not mistaken for author numbering. The trajectories and their original
    analysis results remain immutable; a new aggregate-analysis child
    supersedes the prior display.
    """
    workflow = WorkflowRecord.load(workflow_id)
    if workflow.workflow_type != MD_WORKFLOW_TYPE:
        raise ValueError(f"Workflow {workflow_id} is not an MD simulation")
    if not workflow.inputs:
        raise ValueError(f"Workflow {workflow_id} has no source artifact")

    workflow_input = workflow.inputs[0]
    source_dir = resolve_run_dir(
        workflow_input.source_task_group,
        workflow_input.artifact.run_id,
    )
    if source_dir is None:
        raise FileNotFoundError(workflow_input.artifact.run_id)
    source_job = JobRecord.load(
        source_dir,
        task_group=workflow_input.source_task_group,
    )
    source_path = workflow_input.artifact.resolve(
        source_job.run_dir,
        must_exist=True,
    )
    if source_path is None:
        raise FileNotFoundError(workflow_input.artifact.path)
    mapping = source_author_residue_mapping(
        source_job,
        workflow_input.artifact,
        source_path.read_text(errors="replace"),
    )
    if not mapping.get("residues"):
        raise ValueError("The selected MD source has no mappable protein residues")

    required_children = [child for child in workflow.children if child.required]

    def load_children(prefix: str) -> list[JobRecord]:
        jobs: list[JobRecord] = []
        for child in required_children:
            if not child.step_id.startswith(prefix):
                continue
            child_dir = resolve_run_dir(child.task_group, child.run_id)
            if child_dir is None:
                raise FileNotFoundError(child.run_id)
            jobs.append(JobRecord.load(child_dir, task_group=child.task_group))
        return jobs

    production_jobs = load_children("production_replica_")
    endpoint_jobs = load_children("endpoint_energy_replica_")
    if not production_jobs or any(job.status != "completed" for job in production_jobs):
        raise ValueError("All required MD production replicas must be completed")
    if endpoint_jobs and any(job.status != "completed" for job in endpoint_jobs):
        raise ValueError("All required endpoint-energy jobs must be completed")

    prior_analysis = next(
        (
            child
            for child in required_children
            if child.step_id == "replicate_analysis"
        ),
        None,
    )
    revision = _new_child(
        "md-analysis",
        status="queued",
        metadata={
            "workflow": "md-analysis",
            "awaiting_parent": False,
            "residue_numbering_revision_of_run_id": (
                prior_analysis.run_id if prior_analysis is not None else ""
            ),
            "revision_reason": "restore source author residue numbering",
        },
        input_payload={
            "production_run_ids": [job.run_id for job in production_jobs],
            "endpoint_run_ids": [job.run_id for job in endpoint_jobs],
            "residue_mapping": mapping,
        },
    )
    _complete_analysis(
        revision,
        production_jobs,
        endpoint_jobs,
        residue_mapping_override=mapping,
    )
    revision = JobRecord.load(revision.run_dir, task_group="md-analysis")
    attach_workflow_child(
        workflow_id,
        revision,
        step_id="replicate_analysis",
        depends_on=tuple(
            [
                *(job.run_id for job in production_jobs),
                *(job.run_id for job in endpoint_jobs),
            ]
        ),
        replace_step=True,
    )
    return JobRecord.load(revision.run_dir, task_group="md-analysis")


def advance_md_workflow(workflow_id: str) -> WorkflowRecord:
    """Advance one workflow under an inter-process orchestration lock."""
    lock_path = runs_root() / "workflows" / workflow_id / ".md-orchestration.lock"
    with _exclusive_file_lock(lock_path):
        return _advance_md_workflow_unlocked(workflow_id)


def _advance_md_workflow_unlocked(workflow_id: str) -> WorkflowRecord:
    workflow = WorkflowRecord.load(workflow_id)
    if workflow.workflow_type != MD_WORKFLOW_TYPE:
        return workflow
    children: dict[str, list[JobRecord]] = {}
    required_child_ids = {
        child_ref.run_id
        for child_ref in workflow.children
        if child_ref.required
    }
    for child_ref in workflow.children:
        child_dir = resolve_run_dir(child_ref.task_group, child_ref.run_id)
        if child_dir is None:
            continue
        job = JobRecord.load(child_dir, task_group=child_ref.task_group)
        if job.task_group in {"md-system-prep", "bound-ligand-md"} and (job.run_dir / "result.json").is_file():
            job = finalize_md_job(job)
        children.setdefault(child_ref.step_id, []).append(job)

    def _preferred_job(jobs: list[JobRecord]) -> JobRecord:
        return next(
            (
                job
                for job in reversed(jobs)
                if job.run_id in required_child_ids
            ),
            jobs[-1],
        )

    prep_jobs = children.get("preparation_equilibration", [])
    if prep_jobs:
        prep_job = _preferred_job(prep_jobs)
    else:
        prep_id = str(workflow.parameters.get("prepared_system_run_id") or "")
        prep_dir = resolve_run_dir("md-system-prep", prep_id)
        prep_job = JobRecord.load(prep_dir, task_group="md-system-prep") if prep_dir else None
        if prep_job:
            prep_job = finalize_md_job(prep_job)
    production_jobs = [
        _preferred_job(jobs)
        for step, jobs in children.items()
        if step.startswith("production_replica_")
    ]
    if prep_job and prep_job.status == "completed":
        for child in production_jobs:
            if (
                child.status in {"queued", "blocked"}
                and child.metadata.get("awaiting_parent")
            ):
                try:
                    _activate_production(child, prep_job, workflow)
                except Exception as exc:
                    metadata = _read_json(child.run_dir / "metadata.json")
                    metadata.update({"status": "blocked", "error": str(exc), "updated_at": _utc_now_iso()})
                    _write_json(child.run_dir / "metadata.json", metadata)
    elif prep_job and prep_job.status in {"failed", "cancelled", "blocked"}:
        for child in production_jobs:
            if child.metadata.get("awaiting_parent"):
                metadata = _read_json(child.run_dir / "metadata.json")
                metadata.update(
                    {"status": "blocked", "error": f"Preparation job {prep_job.run_id} did not complete", "updated_at": _utc_now_iso()}
                )
                _write_json(child.run_dir / "metadata.json", metadata)

    production_jobs = [JobRecord.load(job.run_dir, task_group=job.task_group) for job in production_jobs]
    production_request = (
        workflow.parameters.get("production")
        if isinstance(workflow.parameters.get("production"), dict)
        else {}
    )
    endpoint_requested = _endpoint_enabled(production_request)
    endpoint_jobs = [
        _preferred_job(jobs)
        for step, jobs in children.items()
        if step.startswith("endpoint_energy_replica_")
    ]
    if endpoint_requested:
        existing_sources = {
            str(job.metadata.get("source_production_run_id") or "")
            for job in endpoint_jobs
        }
        for production in production_jobs:
            if (
                production.status != "completed"
                or production.run_id in existing_sources
            ):
                continue
            endpoint = create_mmgbsa_analysis_job(
                production.run_id,
                start_pct=float(
                    production_request.get("endpoint_start_pct")
                    or production_request.get("mmgbsa_start_pct")
                    or 20
                ),
                end_pct=float(
                    production_request.get("endpoint_end_pct")
                    or production_request.get("mmgbsa_end_pct")
                    or 100
                ),
                stride=int(
                    production_request.get("endpoint_stride")
                    or production_request.get("mmgbsa_stride")
                    or 1
                ),
                backend=str(
                    production_request.get("endpoint_backend")
                    or "openmm_gbsa"
                ),
                image=str(workflow.parameters.get("image") or DEFAULT_MD_IMAGE),
                use_gpu=bool(workflow.parameters.get("use_gpu", True)),
                gpu_device="automatic",
                attach_to_workflow=False,
            )
            replica_index = int(production.metadata.get("repeat_index") or 1)
            attach_workflow_child(
                workflow.workflow_id,
                endpoint,
                step_id=f"endpoint_energy_replica_{replica_index}",
                depends_on=(production.run_id,),
                replace_step=True,
            )
            endpoint_jobs.append(endpoint)
            existing_sources.add(production.run_id)
        endpoint_jobs = [
            JobRecord.load(job.run_dir, task_group=job.task_group)
            for job in endpoint_jobs
        ]
    analysis_jobs = children.get("replicate_analysis", [])
    production_terminal = production_jobs and all(
        job.status in {"completed", "failed", "cancelled", "blocked"}
        for job in production_jobs
    )
    endpoint_terminal = (
        not endpoint_requested
        or (
            len(endpoint_jobs) == len(production_jobs)
            and all(
                job.status in {"completed", "failed", "cancelled", "blocked"}
                for job in endpoint_jobs
            )
        )
    )
    if analysis_jobs and production_terminal and endpoint_terminal:
        analysis = JobRecord.load(
            _preferred_job(analysis_jobs).run_dir,
            task_group="md-analysis",
        )
        if analysis.status == "queued" and analysis.metadata.get("awaiting_parent"):
            _complete_analysis(analysis, production_jobs, endpoint_jobs)
        elif (
            analysis.status == "failed"
            and all(job.status == "completed" for job in production_jobs)
            and (
                not endpoint_requested
                or all(
                    job.status == "completed"
                    and str(
                        (job.result.get("mmgbsa") or {}).get("status") or ""
                    )
                    == "success"
                    for job in endpoint_jobs
                )
            )
        ):
            replacement = _new_child(
                "md-analysis",
                status="queued",
                metadata={
                    "workflow": "md-analysis",
                    "awaiting_parent": False,
                    "aggregate_retry_of_run_id": analysis.run_id,
                },
                input_payload={
                    "production_run_ids": [
                        job.run_id for job in production_jobs
                    ],
                    "endpoint_run_ids": [
                        job.run_id for job in endpoint_jobs
                    ],
                },
            )
            _complete_analysis(
                replacement,
                production_jobs,
                endpoint_jobs,
            )
            replacement = JobRecord.load(
                replacement.run_dir,
                task_group="md-analysis",
            )
            attach_workflow_child(
                workflow.workflow_id,
                replacement,
                step_id="replicate_analysis",
                depends_on=tuple(
                    [
                        *(job.run_id for job in production_jobs),
                        *(job.run_id for job in endpoint_jobs),
                    ]
                ),
                replace_step=True,
            )
    return refresh_workflow(workflow_id)


def advance_md_workflows() -> int:
    root = runs_root() / "workflows"
    if not root.is_dir():
        return 0
    advanced = 0
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        payload = _read_json(run_dir / "workflow.json")
        if payload.get("workflow_type") != MD_WORKFLOW_TYPE or payload.get("status") in {"completed", "failed", "cancelled"}:
            continue
        advance_md_workflow(run_dir.name)
        advanced += 1
    return advanced
