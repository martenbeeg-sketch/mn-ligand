from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any, Iterable
from uuid import uuid4

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    registered_tool,
    write_registered_command_record,
)
from mn_ligand.core.jobs import (
    JOB_SCHEMA_VERSION,
    JobRecord,
    iter_job_records,
    short_job_code,
)
from mn_ligand.runtime import reference_root, runs_root


GENERATION_TASK_GROUP = "molecule-generation"
GENERATION_CAMPAIGN_TASK_GROUP = "generation-campaigns"


@dataclass(frozen=True)
class GeneratorSpec:
    engine_id: str
    name: str
    theme: str
    tool_id: str
    image: str
    conditions: tuple[str, ...]
    reference_paths: tuple[str, ...]
    summary: str
    integration_status: str = "candidate"
    adapter_ready: bool = False

    def supports(self, condition: str) -> bool:
        return condition in self.conditions


GENERATOR_SPECS: tuple[GeneratorSpec, ...] = (
    GeneratorSpec(
        "omtra",
        "OMTRA",
        "Multi-condition foundation models",
        "omtra_generation",
        "ovolig-omtra-cu128:latest",
        ("pocket", "reference_ligand", "pharmacophore", "protein_pharmacophore"),
        ("generation/omtra/checkpoint.ckpt",),
        "Pocket-, ligand-, pharmacophore-, and combined protein/pharmacophore-conditioned 3D design.",
        integration_status="experimental",
        adapter_ready=True,
    ),
    GeneratorSpec(
        "flowr_root",
        "FLOWR.root",
        "Multi-condition foundation models",
        "flowr_root_generation",
        "ovolig-flowr-root-cu128:latest",
        ("pocket", "reference_ligand", "interaction_profile", "scaffold", "fragment"),
        ("generation/flowr_root/flowr_root_v2.2.ckpt",),
        "Pocket-aware generation with ProLIF interaction conditioning, scaffold hopping, growing, and inpainting.",
        integration_status="experimental",
        adapter_ready=True,
    ),
    GeneratorSpec(
        "pocketxmol",
        "PocketXMol",
        "Pocket and fragment design",
        "pocketxmol_generation",
        "ovolig-pocketxmol-cu128:latest",
        ("pocket", "reference_ligand", "scaffold", "fragment"),
        (
            "generation/pocketxmol/data/trained_models/pxm_use/checkpoints/pocketxmol.ckpt",
            "generation/pocketxmol/data/trained_models/pxm_use/train_config/train.yml",
            "generation/pocketxmol/data/trained_models/pxm/checkpoints/pocketxmol.ckpt",
            "generation/pocketxmol/data/trained_models/pxm/train_config/train.yml",
            "generation/pocketxmol/data/trained_models/tuned_ranker/checkpoints/tuned_ranker.ckpt",
            "generation/pocketxmol/data/trained_models/tuned_ranker/train_config/train_ranker.yml",
        ),
        "Pocket-conditioned de novo design, optimization, fragment growing/linking, and partial redesign.",
        integration_status="experimental",
        adapter_ready=True,
    ),
    GeneratorSpec(
        "conditar",
        "conDitar",
        "Pocket and property optimization",
        "conditar_generation",
        "ovolig-conditar-cu128:latest",
        ("pocket", "reference_ligand"),
        (
            "generation/conditar/Diff.pt",
            "generation/conditar/PocketAE.pt",
        ),
        "Pocket-conditioned diffusion for this group's permissioned local "
        "deployment; the image and source must not be redistributed.",
        integration_status="experimental",
        adapter_ready=True,
    ),
    GeneratorSpec(
        "paopt",
        "conDitar + paOPT",
        "Pocket and property optimization",
        "paopt_generation",
        "ovolig-paopt-cu128:latest",
        ("pocket", "reference_ligand", "property_objectives"),
        (
            "generation/conditar/Diff.pt",
            "generation/conditar/PocketAE.pt",
        ),
        "Property-steered conDitar generation with explicit ADMET endpoints "
        "and optimization direction for this group's permissioned deployment.",
        integration_status="experimental",
        adapter_ready=True,
    ),
    GeneratorSpec(
        "drugrpg",
        "DrugRPG",
        "Pocket and property optimization",
        "drugrpg_generation",
        "ovolig-drugrpg-cu128:latest",
        ("pocket", "reference_ligand"),
        ("generation/drugrpg/training.pt",),
        "Physics-guided pocket-conditioned diffusion for custom target pockets.",
        integration_status="experimental",
        adapter_ready=True,
    ),
    GeneratorSpec(
        "pfm",
        "PFM",
        "Pocket de novo generation",
        "pfm_generation",
        "ovolig-pfm-cu128:latest",
        ("pocket",),
        (
            "generation/pfm/pfm.pth",
            "generation/pfm/x_predictor.pth",
            "generation/pfm/h_predictor.pth",
            "generation/pfm/training.yml",
        ),
        "Perturbed flow matching for target-aware de novo 3D molecule generation.",
        integration_status="experimental",
        adapter_ready=True,
    ),
    GeneratorSpec(
        "pocketflow",
        "PocketFlow",
        "Pocket de novo generation",
        "pocketflow_generation",
        "ovolig-pocketflow-cu128:latest",
        ("pocket",),
        (
            "generation/pocketflow/ZINC-pretrained-255000.pt",
        ),
        "Autoregressive chemically constrained generation directly inside a pocket PDB.",
        integration_status="experimental",
        adapter_ready=True,
    ),
    GeneratorSpec(
        "pgmg",
        "PGMG",
        "Pharmacophore-first design",
        "pgmg_generation",
        "ovolig-pgmg-cu128:latest",
        ("pharmacophore",),
        (
            "generation/pgmg/model.pth",
            "generation/pgmg/tokenizer.pkl",
        ),
        "Fast pharmacophore-graph-conditioned SMILES generation; maximum eight supported points.",
        integration_status="experimental",
        adapter_ready=True,
    ),
)

GENERATOR_BY_ID = {item.engine_id: item for item in GENERATOR_SPECS}


def compatible_generators(
    *,
    has_pocket: bool,
    has_reference_ligand: bool,
    has_pharmacophore: bool,
) -> tuple[GeneratorSpec, ...]:
    available = {
        "pocket": has_pocket,
        "reference_ligand": has_reference_ligand,
        "pharmacophore": has_pharmacophore,
        "protein_pharmacophore": has_pocket and has_pharmacophore,
        "interaction_profile": has_pocket and has_reference_ligand,
        "scaffold": has_reference_ligand,
        "fragment": has_reference_ligand,
        "property_objectives": has_pocket,
    }
    return tuple(
        spec
        for spec in GENERATOR_SPECS
        if any(available.get(condition, False) for condition in spec.conditions)
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _recover_partial_generation_outputs(
    run_dir: Path,
    *,
    engine: str,
    requested_count: int,
) -> None:
    runner_path = (
        Path(__file__).resolve().parents[2]
        / "containers"
        / "generation"
        / "runner.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "_mn_ligand_generation_runner", runner_path
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(
            f"Could not load generation normalizer from {runner_path}"
        )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.normalize_outputs(
        run_dir / "native",
        run_dir / "normalized",
        engine=engine,
        requested_count=requested_count,
    )


def _artifact_job(artifact: ArtifactRef) -> JobRecord:
    job = next(
        (
            item
            for item in iter_job_records(runs_root())
            if item.run_id == artifact.run_id
        ),
        None,
    )
    if job is None:
        raise FileNotFoundError(
            f"Artifact source job {artifact.run_id} is unavailable"
        )
    return job


def _stage_artifact(
    artifact: ArtifactRef,
    destination: Path,
) -> Path:
    job = _artifact_job(artifact)
    source = artifact.resolve(job.run_dir, must_exist=True)
    if source is None:
        raise FileNotFoundError(
            f"Artifact {artifact.artifact_id} is unavailable"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.with_suffix(source.suffix.lower())
    shutil.copy2(source, destination)
    return destination


def _conditioning_target_artifact(artifact: ArtifactRef) -> ArtifactRef:
    if artifact.artifact_type != "prepared_complex":
        return artifact
    job = _artifact_job(artifact)
    if job.artifact_manifest is None:
        return artifact
    for artifact_type in ("prepared_receptor", "prepared_target"):
        candidates = job.artifact_manifest.by_type(artifact_type)
        if candidates:
            return candidates[0]
    return artifact


def _pharmacophore_exchange(
    artifact: ArtifactRef,
    *,
    role: str,
) -> ArtifactRef:
    job = _artifact_job(artifact)
    if job.artifact_manifest is None:
        raise FileNotFoundError("Pharmacophore artifact manifest is unavailable")
    candidate = next(
        (
            item
            for item in job.artifact_manifest.by_type(
                "pharmacophore_exchange"
            )
            if item.role == role
        ),
        None,
    )
    if candidate is None:
        raise ValueError(
            f"The selected hypothesis has no compatible {role} representation"
        )
    return candidate


def pharmacophore_required_contacts(
    artifact: ArtifactRef | None,
) -> list[dict[str, Any]]:
    """Return immutable residue-contact constraints embedded in a hypothesis."""
    if artifact is None:
        return []
    job = _artifact_job(artifact)
    path = artifact.resolve(job.run_dir, must_exist=True)
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return []
    creation_metadata = payload.get("creation_metadata") or {}
    if not isinstance(creation_metadata, dict):
        return []
    return [
        dict(item)
        for item in creation_metadata.get("required_target_contacts") or ()
        if isinstance(item, dict)
    ]


def _native_mode(
    spec: GeneratorSpec,
    *,
    has_pharmacophore: bool,
    redetect_pharmacophore: bool,
) -> str:
    if spec.engine_id != "omtra":
        return ""
    if has_pharmacophore or redetect_pharmacophore:
        return "fixed_protein_pharmacophore_ligand_denovo_condensed"
    return "fixed_protein_ligand_denovo_condensed"


def create_generation_campaign_job(
    *,
    name: str,
    engine_ids: Iterable[str],
    target_artifact: ArtifactRef,
    pocket_artifact: ArtifactRef | None,
    reference_artifact: ArtifactRef | None,
    pharmacophore_artifact: ArtifactRef | None,
    engine_modes: dict[str, str],
    objectives: dict[str, bool],
    requested_count: int,
    batch_size: int,
    seed: int,
    engine_settings: dict[str, dict[str, Any]] | None = None,
) -> JobRecord:
    selected_ids = tuple(dict.fromkeys(str(value) for value in engine_ids))
    if not selected_ids:
        raise ValueError("A generation campaign requires at least one engine")
    unknown = [value for value in selected_ids if value not in GENERATOR_BY_ID]
    if unknown:
        raise ValueError(f"Unknown generation engines: {', '.join(unknown)}")
    required_contacts = pharmacophore_required_contacts(
        pharmacophore_artifact
    )
    run_id = str(uuid4())
    run_dir = runs_root() / GENERATION_CAMPAIGN_TASK_GROUP / run_id
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    campaign_path = artifact_dir / "generation_campaign.json"
    payload = {
        "kind": "generation_campaign",
        "schema_version": 1,
        "name": str(name).strip() or "Molecule design campaign",
        "engines": list(selected_ids),
        "engine_modes": {
            key: str(engine_modes.get(key) or "") for key in selected_ids
        },
        "engine_settings": {
            key: dict((engine_settings or {}).get(key) or {})
            for key in selected_ids
        },
        "target_artifact": target_artifact.to_dict(),
        "pocket_artifact": (
            pocket_artifact.to_dict() if pocket_artifact else None
        ),
        "reference_artifact": (
            reference_artifact.to_dict() if reference_artifact else None
        ),
        "pharmacophore_artifact": (
            pharmacophore_artifact.to_dict()
            if pharmacophore_artifact
            else None
        ),
        "required_target_contacts": required_contacts,
        "objectives": {key: bool(value) for key, value in objectives.items()},
        "parameters": {
            "requested_count_per_engine": int(requested_count),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "per_engine_workloads": {
                key: {
                    field: int(value)
                    for field, value in dict(
                        (engine_settings or {}).get(key) or {}
                    ).items()
                    if field
                    in {
                        "requested_count",
                        "batch_size",
                        "seed",
                        "max_runtime_seconds",
                    }
                }
                for key in selected_ids
            },
        },
    }
    _write_json(campaign_path, payload)
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "generation_campaign",
        "workflow": "molecule_generation_campaign",
        "status": "completed",
        "name": payload["name"],
        "engine_ids": list(selected_ids),
        "engine_count": len(selected_ids),
        "required_target_contact_count": len(required_contacts),
        "parent_run_id": target_artifact.run_id,
        "target_run_id": target_artifact.run_id,
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "input.json", payload)
    _write_json(
        run_dir / "result.json",
        {
            "success": True,
            "engine_count": len(selected_ids),
            "requested_count_per_engine": int(requested_count),
        },
    )
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(
                run_dir,
                campaign_path,
                "generation_campaign",
                role="immutable_campaign",
            )
        ],
    )
    return JobRecord.load(
        run_dir, task_group=GENERATION_CAMPAIGN_TASK_GROUP
    )


def queue_generation_job(
    campaign_job: JobRecord,
    *,
    engine_id: str,
    target_artifact: ArtifactRef,
    pocket_artifact: ArtifactRef | None,
    reference_artifact: ArtifactRef | None,
    pharmacophore_artifact: ArtifactRef | None,
    engine_mode: str,
    objectives: dict[str, bool],
    requested_count: int,
    batch_size: int,
    seed: int,
    image: str | None = None,
    engine_settings: dict[str, Any] | None = None,
) -> JobRecord:
    spec = GENERATOR_BY_ID.get(engine_id)
    if spec is None:
        raise ValueError(f"Unknown generation engine: {engine_id}")
    if not spec.adapter_ready:
        raise ValueError(
            f"{spec.name} still requires a reviewed native configuration adapter"
        )
    if requested_count < 1:
        raise ValueError("Requested molecule count must be positive")
    missing_references = [
        value
        for value in spec.reference_paths
        if not (reference_root() / value).is_file()
    ]
    if missing_references:
        raise FileNotFoundError(
            f"{spec.name} is missing reference files: "
            + ", ".join(missing_references)
        )
    if spec.engine_id == "pgmg" and pharmacophore_artifact is None:
        raise ValueError("PGMG requires a compatible pharmacophore hypothesis")
    if spec.engine_id in {
        "pocketflow",
        "pfm",
        "drugrpg",
        "flowr_root",
        "conditar",
        "paopt",
    }:
        if pocket_artifact is None:
            raise ValueError(
                f"{spec.name} requires a coordinate pocket artifact"
            )
    if spec.engine_id == "flowr_root" and reference_artifact is None:
        raise ValueError(
            "FLOWR.root requires a reference ligand to define the pocket frame"
        )
    if spec.engine_id == "paopt" and reference_artifact is None:
        raise ValueError("paOPT requires a reference ligand")
    required_contacts = pharmacophore_required_contacts(
        pharmacophore_artifact
    )
    constraint_enforcement = (
        "native_pharmacophore_conditioning_plus_downstream_pose_validation"
        if required_contacts
        and (
            spec.supports("pharmacophore")
            or spec.supports("protein_pharmacophore")
        )
        else (
            "downstream_pose_validation"
            if required_contacts
            else "not_requested"
        )
    )

    run_id = str(uuid4())
    run_dir = runs_root() / GENERATION_TASK_GROUP / run_id
    input_dir = run_dir / "input"
    native_dir = run_dir / "native"
    normalized_dir = run_dir / "normalized"
    input_dir.mkdir(parents=True, exist_ok=False)
    native_dir.mkdir()
    normalized_dir.mkdir()
    staged: dict[str, str] = {}
    conditioning_target_artifact = _conditioning_target_artifact(
        target_artifact
    )
    target_path = _stage_artifact(
        conditioning_target_artifact, input_dir / "target"
    )
    staged["target"] = target_path.relative_to(run_dir).as_posix()
    pocket_path: Path | None = None
    if pocket_artifact is not None:
        pocket_path = _stage_artifact(
            pocket_artifact, input_dir / "pocket"
        )
        staged["pocket"] = pocket_path.relative_to(run_dir).as_posix()
    reference_path: Path | None = None
    if reference_artifact is not None:
        reference_path = _stage_artifact(
            reference_artifact, input_dir / "reference_ligand"
        )
        staged["reference_ligand"] = reference_path.relative_to(
            run_dir
        ).as_posix()
    pharmacophore_path: Path | None = None
    if pharmacophore_artifact is not None:
        role = "pgmg_posp" if spec.engine_id == "pgmg" else "pharmit_json"
        exchange = _pharmacophore_exchange(
            pharmacophore_artifact, role=role
        )
        pharmacophore_path = _stage_artifact(
            exchange, input_dir / "pharmacophore"
        )
        staged["pharmacophore"] = pharmacophore_path.relative_to(
            run_dir
        ).as_posix()
    redetect = (
        spec.engine_id == "omtra"
        and "Re-detect pharmacophore" in str(engine_mode)
    )
    if redetect:
        if reference_path is None:
            raise ValueError(
                "OMTRA pharmacophore re-detection requires a reference ligand"
            )
        pharmacophore_path = reference_path

    settings = dict(engine_settings or {})
    command_arguments = [
        "--output",
        "/work/native",
        "--normalized-output",
        "/work/normalized",
        "--count",
        str(int(requested_count)),
        "--batch-size",
        str(max(1, int(batch_size))),
        "--seed",
        str(int(seed)),
        "--run-name",
        f"{spec.engine_id}_{short_job_code(run_id).lower()}",
        "--checkpoint",
        f"/references/{spec.reference_paths[0]}",
        "--native-mode",
        str(engine_mode),
    ]
    if spec.engine_id == "omtra":
        command_arguments.extend(
            [
                "--integration-steps",
                str(max(25, int(settings.get("integration_steps") or 250))),
                "--noise-scale",
                str(float(settings.get("noise_scale") or 1.0)),
                "--epsilon",
                str(float(settings.get("epsilon") or 0.01)),
                "--ligand-atoms-mean",
                str(float(settings.get("ligand_atoms_mean") or 0.0)),
                "--ligand-atoms-std",
                str(float(settings.get("ligand_atoms_std") or 2.0)),
            ]
        )
        if bool(settings.get("stochastic_sampling")):
            command_arguments.append("--stochastic-sampling")
        command_arguments.extend(
            [
                "--target",
                f"/work/{target_path.relative_to(run_dir).as_posix()}",
                "--native-mode",
                _native_mode(
                    spec,
                    has_pharmacophore=pharmacophore_path is not None,
                    redetect_pharmacophore=redetect,
                ),
            ]
        )
        if pocket_path is not None:
            command_arguments.extend(
                [
                    "--pocket-structure",
                    f"/work/{pocket_path.relative_to(run_dir).as_posix()}",
                ]
            )
        if reference_path is not None:
            command_arguments.extend(
                [
                    "--reference-ligand",
                    f"/work/{reference_path.relative_to(run_dir).as_posix()}",
                ]
            )
        if pharmacophore_path is not None:
            command_arguments.extend(
                [
                    "--pharmacophore",
                    f"/work/{pharmacophore_path.relative_to(run_dir).as_posix()}",
                ]
            )
    elif spec.engine_id in {"pocketflow", "pfm", "drugrpg"}:
        if spec.engine_id == "pocketflow":
            command_arguments.extend(
                [
                    "--atom-temperature",
                    str(float(settings.get("atom_temperature") or 1.0)),
                    "--bond-temperature",
                    str(float(settings.get("bond_temperature") or 1.0)),
                    "--max-atoms",
                    str(max(5, int(settings.get("max_atoms") or 40))),
                    "--focus-strategy",
                    str(settings.get("focus_strategy") or "maximum"),
                    "--focus-threshold",
                    str(float(settings.get("focus_threshold") or 0.5)),
                    "--min-protein-distance",
                    str(
                        float(
                            settings.get("min_protein_distance") or 3.0
                        )
                    ),
                ]
            )
        elif spec.engine_id == "drugrpg":
            command_arguments.extend(
                [
                    "--max-atoms",
                    str(max(5, int(settings.get("max_atoms") or 30))),
                ]
            )
        command_arguments.extend(
            [
                "--pocket-structure",
                f"/work/{pocket_path.relative_to(run_dir).as_posix()}",
            ]
        )
    elif spec.engine_id == "conditar":
        command_arguments.extend(
            [
                "--pocket-structure",
                f"/work/{pocket_path.relative_to(run_dir).as_posix()}",
                "--secondary-reference",
                f"/references/{spec.reference_paths[1]}",
                "--diffusion-steps",
                str(
                    max(100, int(settings.get("diffusion_steps") or 1000))
                ),
                "--pocket-radius",
                str(float(settings.get("pocket_radius") or 10.0)),
            ]
        )
    elif spec.engine_id == "paopt":
        optimize_properties = [
            str(value) for value in settings.get("optimize_properties") or ()
        ]
        minimize_properties = [
            str(value) for value in settings.get("minimize_properties") or ()
        ]
        if not optimize_properties:
            raise ValueError("paOPT requires at least one ADMET endpoint")
        command_arguments.extend(
            [
                "--pocket-structure",
                f"/work/{pocket_path.relative_to(run_dir).as_posix()}",
                "--reference-ligand",
                f"/work/{reference_path.relative_to(run_dir).as_posix()}",
                "--secondary-reference",
                f"/references/{spec.reference_paths[1]}",
                "--optimization-steps",
                str(max(1, int(settings.get("optimization_steps") or 1))),
                "--gradient-estimate-pairs",
                str(
                    max(
                        1,
                        int(settings.get("gradient_estimate_pairs") or 4),
                    )
                ),
                "--diffusion-steps",
                str(
                    max(100, int(settings.get("diffusion_steps") or 1000))
                ),
                "--pocket-radius",
                str(float(settings.get("pocket_radius") or 10.0)),
                "--perturbation-size",
                str(float(settings.get("perturbation_size") or 0.03)),
            ]
        )
        for value in optimize_properties:
            command_arguments.extend(["--optimize-properties", value])
        for value in minimize_properties:
            command_arguments.extend(["--minimize-properties", value])
    elif spec.engine_id == "pocketxmol":
        redesign_mode = str(settings.get("redesign_mode") or "").strip()
        optimizing = (
            "optimize" in str(engine_mode).lower()
            or bool(redesign_mode)
        )
        command_arguments.extend(
            [
                "--diffusion-steps",
                str(
                    max(
                        10,
                        int(
                            settings.get("diffusion_steps")
                            or (50 if optimizing else 100)
                        ),
                    )
                ),
                "--ligand-atoms-mean",
                str(
                    float(
                        settings.get("ligand_atoms_mean")
                        or (38.0 if optimizing else 28.0)
                    )
                ),
                "--ligand-atoms-std",
                str(float(settings.get("ligand_atoms_std") or 2.0)),
                "--optimization-strength",
                str(
                    float(settings.get("optimization_strength") or 0.5)
                ),
            ]
        )
        command_arguments.extend(
            [
                "--target",
                f"/work/{target_path.relative_to(run_dir).as_posix()}",
                "--pocket-structure",
                f"/work/{pocket_path.relative_to(run_dir).as_posix()}",
            ]
        )
        if reference_path is not None:
            command_arguments.extend(
                [
                    "--reference-ligand",
                    f"/work/{reference_path.relative_to(run_dir).as_posix()}",
                ]
            )
        if redesign_mode:
            command_arguments.extend(
                ["--redesign-mode", redesign_mode]
            )
            if redesign_mode == "fragment_growing":
                command_arguments.extend(
                    [
                        "--grow-size",
                        str(max(1, int(settings.get("grow_size") or 10))),
                    ]
                )
            for value in settings.get("preserve_atom_indices") or ():
                command_arguments.extend(
                    ["--preserve-atom-index", str(int(value))]
                )
            for value in settings.get("redesign_atom_indices") or ():
                command_arguments.extend(
                    ["--redesign-atom-index", str(int(value))]
                )
            for value in settings.get("anchor_atom_indices") or ():
                command_arguments.extend(
                    ["--anchor-atom-index", str(int(value))]
                )
    elif spec.engine_id == "flowr_root":
        command_arguments.extend(
            [
                "--integration-steps",
                str(max(10, int(settings.get("integration_steps") or 100))),
                "--corrector-steps",
                str(max(0, int(settings.get("corrector_steps") or 0))),
                "--solver",
                str(settings.get("solver") or "euler"),
                "--diversity-threshold",
                str(float(settings.get("diversity_threshold") or 0.9)),
            ]
        )
        if bool(settings.get("use_sde_simulation")):
            command_arguments.append("--use-sde-simulation")
        if bool(settings.get("sample_molecule_sizes")):
            command_arguments.append("--sample-molecule-sizes")
        if bool(settings.get("filter_diversity")):
            command_arguments.append("--filter-diversity")
        command_arguments.extend(
            [
                "--pocket-structure",
                f"/work/{pocket_path.relative_to(run_dir).as_posix()}",
            ]
        )
        if reference_path is not None:
            command_arguments.extend(
                [
                    "--reference-ligand",
                    f"/work/{reference_path.relative_to(run_dir).as_posix()}",
                ]
            )
        redesign_mode = str(settings.get("redesign_mode") or "").strip()
        if redesign_mode:
            command_arguments.extend(
                ["--redesign-mode", redesign_mode]
            )
            if redesign_mode == "fragment_growing":
                command_arguments.extend(
                    [
                        "--grow-size",
                        str(max(1, int(settings.get("grow_size") or 10))),
                    ]
                )
            for value in settings.get("preserve_atom_indices") or ():
                command_arguments.extend(
                    ["--preserve-atom-index", str(int(value))]
                )
            for value in settings.get("redesign_atom_indices") or ():
                command_arguments.extend(
                    ["--redesign-atom-index", str(int(value))]
                )
            for value in settings.get("anchor_atom_indices") or ():
                command_arguments.extend(
                    ["--anchor-atom-index", str(int(value))]
                )
            if bool(settings.get("filter_conditioned_substructure", True)):
                command_arguments.append(
                    "--filter-conditioned-substructure"
                )
    elif spec.engine_id == "pgmg":
        command_arguments.extend(
            [
                "--pharmacophore",
                f"/work/{pharmacophore_path.relative_to(run_dir).as_posix()}",
                "--secondary-reference",
                f"/references/{spec.reference_paths[1]}",
            ]
        )
    selected_image = str(image or spec.image)
    tool = registered_tool(spec.tool_id, image=selected_image)
    command = build_docker_command(
        DockerRunSpec(
            tool=tool,
            command=tuple(command_arguments),
            mounts=(
                DockerMount(run_dir, "/work"),
                DockerMount(reference_root(), "/references", read_only=True),
            ),
            gpu_enabled=True,
            use_host_user=True,
            shm_size="8g",
        )
    )
    try:
        max_runtime_seconds = max(
            0, int(settings.get("max_runtime_seconds") or 0)
        )
    except (TypeError, ValueError):
        max_runtime_seconds = 0
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "molecule_generation",
        "workflow": "molecule_generation",
        "operation": "generation",
        "status": "queued",
        "tool": spec.name,
        "engine_id": spec.engine_id,
        "parent_run_id": campaign_job.run_id,
        "campaign_run_id": campaign_job.run_id,
        "target_run_id": target_artifact.run_id,
        "engine_mode": str(engine_mode),
        "requested_count": int(requested_count),
        "batch_size": max(1, int(batch_size)),
        "max_runtime_seconds": max_runtime_seconds,
        "seed": int(seed),
        "required_target_contact_count": len(required_contacts),
        "constraint_enforcement": constraint_enforcement,
        "docker_image": selected_image,
        "created_at": now,
        "updated_at": now,
        "queued_at": now,
        "queued_command": command,
        "gpu_queued": True,
        "resources": tool.resources.to_dict(),
        "worker_finalizer": "molecule_generation",
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "campaign_run_id": campaign_job.run_id,
            "engine_id": spec.engine_id,
            "engine_mode": str(engine_mode),
            "target_artifact": target_artifact.to_dict(),
            "conditioning_target_artifact": (
                conditioning_target_artifact.to_dict()
            ),
            "pocket_artifact": (
                pocket_artifact.to_dict() if pocket_artifact else None
            ),
            "reference_artifact": (
                reference_artifact.to_dict() if reference_artifact else None
            ),
            "pharmacophore_artifact": (
                pharmacophore_artifact.to_dict()
                if pharmacophore_artifact
                else None
            ),
            "required_target_contacts": required_contacts,
            "constraint_enforcement": constraint_enforcement,
            "staged_inputs": staged,
            "objectives": {
                key: bool(value) for key, value in objectives.items()
            },
            "engine_settings": dict(engine_settings or {}),
            "parameters": {
                "requested_count": int(requested_count),
                "batch_size": max(1, int(batch_size)),
                "seed": int(seed),
                "max_runtime_seconds": max_runtime_seconds,
            },
        },
    )
    write_registered_command_record(
        run_dir,
        tool_id=spec.tool_id,
        commands=(command,),
        image=selected_image,
    )
    write_artifact_manifest(run_dir, [])
    return JobRecord.load(run_dir, task_group=GENERATION_TASK_GROUP)


def finalize_generation_job(
    run_dir: Path,
    *,
    returncode: int,
) -> JobRecord:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    report_path = run_dir / "normalized" / "generation_report.json"
    table_path = run_dir / "normalized" / "generated_compounds.csv"
    compounds_path = run_dir / "normalized" / "generated_compounds.sdf"
    recovery_attempted = False
    recovery_error = ""
    if returncode != 0 or not report_path.is_file():
        recovery_attempted = True
        try:
            # The image normally performs this normalization after native
            # inference. If the worker stops inference at its runtime budget,
            # recover every complete molecule already flushed to native/.
            _recover_partial_generation_outputs(
                run_dir,
                engine=str(metadata.get("engine_id") or ""),
                requested_count=int(metadata.get("requested_count") or 0),
            )
        except Exception as exc:
            recovery_error = str(exc)
    try:
        report = json.loads(report_path.read_text())
    except (OSError, TypeError, ValueError):
        report = {}
    valid_count = int(report.get("unique_valid_compound_count") or 0)
    success = (
        returncode == 0
        and valid_count > 0
        and table_path.is_file()
        and compounds_path.is_file()
        and compounds_path.stat().st_size > 0
    )
    artifacts: list[ArtifactRef] = []
    if compounds_path.is_file() and compounds_path.stat().st_size:
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                compounds_path,
                "generated_molecule_set",
                role="rdkit_valid_generated_molecules",
                metadata={
                    "qualification_required_for_downstream": True,
                },
            )
        )
    for path, artifact_type, role in (
        (table_path, "generated_compound_table", "normalized_inventory"),
        (report_path, "generation_metrics", "normalization_report"),
    ):
        if path.is_file() and path.stat().st_size:
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir, path, artifact_type, role=role
                )
            )
    for path in sorted((run_dir / "native").rglob("*")):
        if path.is_file():
            artifacts.append(
                ArtifactRef.from_path(
                    run_dir,
                    path,
                    "native_generation_output",
                    role=path.relative_to(run_dir / "native").as_posix(),
                    checksum=False,
                )
            )
    for name, role in (("stdout.log", "stdout"), ("stderr.log", "stderr")):
        path = run_dir / name
        path.touch(exist_ok=True)
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                path,
                "job_log",
                role=role,
                checksum=False,
            )
        )
    error = "" if success else (
        (run_dir / "stderr.log").read_text(errors="replace")[-4000:]
        or "Generation produced no normalized valid compounds"
    )
    result = {
        **report,
        "success": success,
        "returncode": int(returncode),
        "valid_compound_count": valid_count,
        "partial_output_recovery_attempted": recovery_attempted,
        "partial_output_recovery_succeeded": (
            recovery_attempted and not recovery_error
        ),
        "error": error,
    }
    if recovery_error:
        result["partial_output_recovery_error"] = recovery_error
    _write_json(run_dir / "result.json", result)
    write_artifact_manifest(run_dir, artifacts)
    now = _utc_now_iso()
    metadata.update(
        {
            "status": "completed" if success else "failed",
            "updated_at": now,
            "completed_at": now,
            "valid_compound_count": valid_count,
            "partial_output_recovery_attempted": recovery_attempted,
            "partial_output_recovery_succeeded": (
                recovery_attempted and not recovery_error
            ),
        }
    )
    if recovery_error:
        metadata["partial_output_recovery_error"] = recovery_error
    if error:
        metadata["error"] = error
    _write_json(metadata_path, metadata)
    if success:
        try:
            from mn_ligand.workflows.molecule_qualification import (
                queue_molecule_qualification_job,
            )

            source_job = JobRecord.load(
                run_dir,
                task_group=GENERATION_TASK_GROUP,
            )
            qualification_job = queue_molecule_qualification_job(source_job)
            metadata["qualification_run_id"] = qualification_job.run_id
            result["qualification_run_id"] = qualification_job.run_id
            _write_json(metadata_path, metadata)
            _write_json(run_dir / "result.json", result)
        except Exception as exc:
            metadata["qualification_queue_error"] = str(exc)
            result["qualification_queue_error"] = str(exc)
            _write_json(metadata_path, metadata)
            _write_json(run_dir / "result.json", result)
    return JobRecord.load(run_dir, task_group=GENERATION_TASK_GROUP)
