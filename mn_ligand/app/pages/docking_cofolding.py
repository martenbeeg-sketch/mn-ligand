from __future__ import annotations

import json
import os
import shlex
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

import pandas as pd
import streamlit as st

from mn_ligand.core.artifacts import ArtifactRef
from mn_ligand.app.pages.compound_results import render_selected_compound
from mn_ligand.app.pages.discover_inputs import (
    ArtifactChoice,
    artifact_box,
    artifact_options,
    bound_ligand_box,
    job_records_snapshot,
    render_selected_artifacts,
    render_target_viewer,
    select_artifact,
    select_target_artifacts,
    target_coordinate_ligand_box,
    target_ligand_path,
    target_viewer_path,
)
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.app.pages.search_region import (
    FIXED_BOX,
    PADDING_BOX,
)
from mn_ligand.core.jobs import display_job_code
from mn_ligand.core.provenance import (
    COMPOUND_DATASET_CAMPAIGN_PURPOSE,
    TARGET_LIGAND_CAMPAIGN_PURPOSE,
    is_benchmark_job,
    target_lineage_summary,
)
from mn_ligand.runtime import (
    cpu_process_limit,
    runs_root,
    unidock_pro_max_compounds,
    vina_compound_timeout_minutes,
)
from mn_ligand.workflows.docking import DEFAULT_DOCKING_IMAGE, queue_docking_campaign_job
from mn_ligand.workflows.compound_preparation import (
    COMPOUND_IMPORT_TASK_GROUP,
    compound_box_fit_rows,
    create_docking_parent_selection_job,
    parent_duplicate_report,
)
from mn_ligand.workflows.compound_pubchem import (
    accepted_reviewed_compound_rows,
)
from mn_ligand.workflows.openvs import DEFAULT_OPENVS_IMAGE, queue_openvs_docking_job
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
    polymer_sequences_from_pdb,
    queue_alphafold3_msa_job,
    queue_alphafold3_refolding_job,
    queue_boltz2_refolding_job,
    queue_nesso_affinity_job,
)
from mn_ligand.workflows.target_orientation import (
    create_axis_aligned_target_job,
    ligand_longest_axis_transform,
    transform_axis_aligned_box,
)


CLASSICAL_ENGINES = ("Uni-Dock Pro", "AutoDock Vina", "GNINA", "RosettaLigand")
COFOLDING_ENGINES = ("Boltz-2", "AlphaFold 3", "Nesso-1")
ALL_ENGINES = CLASSICAL_ENGINES + COFOLDING_ENGINES
ENGINE_KEYS = {
    engine: f"binding_engine_{engine.lower().replace(' ', '_').replace('-', '_')}"
    for engine in ALL_ENGINES
}


def _imported_compound_options() -> dict[str, ArtifactChoice]:
    return {
        label: choice
        for label, choice in artifact_options(("compound_set",)).items()
        if (
            (
                choice.job.task_group == COMPOUND_IMPORT_TASK_GROUP
                and str(choice.job.metadata.get("job_type") or "")
                == "compound_import"
            )
            or choice.job.workflow == "molecule_design_selection"
        )
    }


def _docking_parent_frame(choice: ArtifactChoice) -> pd.DataFrame:
    path = choice.artifact.resolve(choice.job.run_dir, must_exist=True)
    if path is None:
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        source = pd.read_csv(path).fillna("")
        records = source.to_dict("records")
    else:
        from mn_ligand.workflows.docking import load_compound_records

        records = load_compound_records([path])
    reviewed = accepted_reviewed_compound_rows(choice.job.run_id)
    records.extend(reviewed)
    analysis = parent_duplicate_report(records)
    return pd.DataFrame(analysis["docking_parent_rows"])


def _parent_display_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in (
            "docking_parent_id",
            "representative_compound_id",
            "representative_product_name",
            "source_record_count",
            "compound_ids",
            "cas_numbers",
            "standardized_parent_formula",
            "standardized_parent_smiles",
            "source_formulations",
            "structure_origins",
            "estimated_3d_length_angstrom",
            "estimated_3d_width_angstrom",
            "estimated_3d_thickness_angstrom",
            "estimated_max_span_angstrom",
            "box_fit_status",
            "box_fit_reason",
            "box_fit_max_excess_angstrom",
        )
        if column in frame.columns
    ]


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


def _artifact_for_path(
    choice: ArtifactChoice, path
):
    if path is None:
        return None
    resolved_path = path.resolve()
    if choice.job.artifact_manifest is not None:
        for artifact in choice.job.artifact_manifest.artifacts:
            candidate = artifact.resolve(choice.job.run_dir, must_exist=True)
            if candidate is not None and candidate.resolve() == resolved_path:
                return artifact
    try:
        relative_path = resolved_path.relative_to(choice.job.run_dir.resolve())
    except ValueError:
        return None
    if resolved_path.is_file():
        return ArtifactRef(
            run_id=choice.job.run_id,
            artifact_type="prepared_ligand_set",
            path=relative_path.as_posix(),
            role="associated_ligand",
            label=resolved_path.name,
        )
    return None


def _initialize_box(
    target,
    pocket,
    *,
    ligand_box_override=None,
    pocket_box_override=None,
):
    pocket_box = pocket_box_override or artifact_box(pocket)
    metadata_box = _metadata_box(target)
    ligand_box = ligand_box_override
    if target is not None:
        if ligand_box is None:
            ligand_box = target_coordinate_ligand_box(target)
        ligand_key = str(target.job.metadata.get("ligand_key") or "")
        if ligand_key and ligand_box is None:
            ligand_box = bound_ligand_box(
                target, ligand_key, padding_angstrom=0.0
            )
    default_box = pocket_box or ligand_box or metadata_box or {
        "center": (0.0, 0.0, 0.0),
        "size": (22.0, 22.0, 22.0),
    }
    source = (
        "Predicted pocket" if pocket_box
        else "Target-associated ligand pose" if ligand_box
        else "Stored docking box" if metadata_box
        else "Manual"
    )
    signature = ":".join(
        (
            target.job.run_id if target is not None else "none",
            pocket.job.run_id if pocket is not None else "none",
            pocket.artifact.artifact_id if pocket is not None else "none",
            source.lower().replace(" ", "-"),
        )
    )
    return (
        tuple(float(value) for value in default_box["center"]),
        tuple(float(value) for value in default_box["size"]),
        source,
        signature,
    )


def _set_engine_selection(value: bool) -> None:
    for key in ENGINE_KEYS.values():
        st.session_state[key] = value


def _target_label(choice: ArtifactChoice) -> str:
    return str(
        choice.job.metadata.get("pdb_id")
        or choice.artifact.label
        or display_job_code(
            choice.job.metadata.get("job_code"), choice.job.run_id
        )
    )


def _render_shared_box_settings(
    *, selected_classical_context: bool, ensemble_mode: bool
) -> dict[str, Any]:
    if not selected_classical_context:
        return {"mode": "fixed", "size": (22.0, 22.0, 22.0), "padding": None}
    st.markdown(
        "#### Ensemble box size" if ensemble_mode else "#### Box size"
    )
    st.caption(
        (
            "One size applies to every selected, coordinate-aligned target."
            if ensemble_mode
            else "Set explicit dimensions or padding around the selected pocket or ligand."
        )
    )
    mode = st.segmented_control(
        "Box sizing",
        (FIXED_BOX, PADDING_BOX),
        default=FIXED_BOX,
        key="binding_shared_box_mode",
    ) or FIXED_BOX
    if mode == PADDING_BOX:
        padding = float(
            st.number_input(
                "Padding on each side (Å)",
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=0.5,
                key="binding_shared_box_padding",
            )
        )
        return {"mode": "padding", "size": None, "padding": padding}
    size_columns = st.columns(3)
    size = tuple(
        float(
            column.number_input(
                f"size_{axis}",
                value=20.0,
                min_value=1.0,
                step=1.0,
                format="%.2f",
                key=f"binding_shared_box_size_{axis}",
            )
        )
        for column, axis in zip(size_columns, "xyz")
    )
    return {"mode": "fixed", "size": size, "padding": None}


def _render_global_center(
    center: tuple[float, float, float], *, source_signature: str
) -> tuple[float, float, float]:
    signature_key = "binding_global_center_source_signature"
    center_keys = tuple(f"binding_global_center_{axis}" for axis in "xyz")
    if st.session_state.get(signature_key) != source_signature:
        for center_key, value in zip(center_keys, center):
            st.session_state[center_key] = float(value)
        st.session_state[signature_key] = source_signature
    columns = st.columns(3)
    return tuple(
        float(
            column.number_input(
                f"center_{axis}",
                step=0.5,
                format="%.3f",
                key=center_key,
            )
        )
        for column, axis, value, center_key in zip(
            columns, "xyz", center, center_keys
        )
    )


def _build_target_launch_context(
    target: ArtifactChoice,
    *,
    pocket: ArtifactChoice | None,
    axis_alignment: bool,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    box_mode: str,
    box_padding: float | None,
) -> dict[str, Any]:
    associated_ligand = target_ligand_path(target)
    associated_ligand_artifact = (
        _artifact_for_path(target, associated_ligand)
        if associated_ligand is not None
        else None
    )
    alignment_transform = None
    if axis_alignment and associated_ligand is not None:
        try:
            alignment_transform = ligand_longest_axis_transform(
                associated_ligand
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            st.warning(
                f"{_target_label(target)} axis alignment is unavailable: {exc}"
            )
            alignment_transform = None
    return {
        "target": target,
        "pocket": pocket,
        "associated_ligand": associated_ligand,
        "associated_ligand_artifact": associated_ligand_artifact,
        "alignment_transform": alignment_transform,
        "center": center,
        "size": size,
        "box_mode": box_mode,
        "box_padding": box_padding,
    }


def _render_target_contract(
    targets: tuple[ArtifactChoice, ...],
    *,
    selected_classical_context: bool,
    ensemble_mode: bool,
) -> list[dict[str, Any]]:
    target = targets[0]
    st.markdown(
        "### Multi-target ensemble"
        if ensemble_mode
        else "### Single-target docking"
    )
    if ensemble_mode and len(targets) == 1:
        st.info("Select another prepared target to build an ensemble.")
    with st.container():
        axis_alignment = st.checkbox(
            (
                "Align all target-associated ligand longest axes to X"
                if ensemble_mode
                else "Align target-associated ligand longest axis to X"
            ),
            value=False,
            key="binding_align_all_target_ligand_axes_x",
            help=(
                "For an ensemble, applies the equivalent ligand-derived rigid "
                "transform to every selected target."
            ),
        )
        pocket = (
            select_artifact(
                "Global pocket" if ensemble_mode else "Pocket",
                ("pocket",),
                key="binding_global_pocket",
                required=False,
            )
            if selected_classical_context
            else None
        )
        primary_ligand = target_ligand_path(target)
        primary_transform = None
        ligand_box_override = None
        pocket_box_override = None
        if axis_alignment and primary_ligand is not None:
            try:
                primary_transform = ligand_longest_axis_transform(
                    primary_ligand
                )
                aligned_box = primary_transform["ligand_box"]
                ligand_box_override = {
                    "center": tuple(aligned_box["center"]),
                    "size": tuple(aligned_box["size"]),
                }
                source_pocket_box = artifact_box(pocket)
                if source_pocket_box is not None:
                    pocket_box_override = transform_axis_aligned_box(
                        source_pocket_box, primary_transform
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                st.warning(f"Global axis alignment is unavailable: {exc}")
                primary_transform = None
        default_center, source_size, box_source, _ = _initialize_box(
            target,
            pocket,
            ligand_box_override=ligand_box_override,
            pocket_box_override=pocket_box_override,
        )
        if selected_classical_context:
            st.markdown(
                "#### Ensemble docking box"
                if ensemble_mode
                else "#### Docking box"
            )
            st.caption(
                f"Center and padding source: {box_source}. Selecting a global "
                "pocket replaces the associated-ligand source."
            )
            if len(targets) == 1:
                center_signature = ":".join(
                    (
                        target.job.run_id,
                        target.artifact.artifact_id,
                        pocket.artifact.artifact_id if pocket is not None else "ligand",
                        "aligned" if axis_alignment else "original",
                    )
                )
                center = _render_global_center(
                    default_center,
                    source_signature=center_signature,
                )
            else:
                center = default_center
                st.caption(
                    "Each target receives its own automatic box center from its "
                    "associated ligand (or the selected pocket)."
                )
        else:
            center = default_center
        shared_box = _render_shared_box_settings(
            selected_classical_context=selected_classical_context,
            ensemble_mode=ensemble_mode,
        )
        if shared_box["mode"] == "padding":
            box_padding = float(shared_box["padding"])
            size = tuple(
                max(1.0, float(value) + 2.0 * box_padding)
                for value in source_size
            )
            box_mode = "padding"
            st.caption(
                "Effective size: "
                + " × ".join(f"{value:.2f}" for value in size)
                + " Å."
            )
        else:
            size = tuple(float(value) for value in shared_box["size"])
            box_mode, box_padding = "fixed", None
        contexts: list[dict[str, Any]] = []
        for choice in targets:
            context = _build_target_launch_context(
                choice,
                pocket=pocket,
                axis_alignment=axis_alignment,
                center=center,
                size=size,
                box_mode=box_mode,
                box_padding=box_padding,
            )
            target_ligand_box = None
            target_pocket_box = None
            target_transform = context["alignment_transform"]
            if target_transform is not None:
                aligned_box = target_transform["ligand_box"]
                target_ligand_box = {
                    "center": tuple(aligned_box["center"]),
                    "size": tuple(aligned_box["size"]),
                }
                source_pocket_box = artifact_box(pocket)
                if source_pocket_box is not None:
                    target_pocket_box = transform_axis_aligned_box(
                        source_pocket_box, target_transform
                    )
            target_center, target_source_size, _, _ = _initialize_box(
                choice,
                pocket,
                ligand_box_override=target_ligand_box,
                pocket_box_override=target_pocket_box,
            )
            if len(targets) == 1:
                target_center = center
            target_size = (
                tuple(
                    max(1.0, float(value) + 2.0 * float(box_padding))
                    for value in target_source_size
                )
                if box_mode == "padding" and box_padding is not None
                else size
            )
            context["center"] = tuple(float(value) for value in target_center)
            context["size"] = tuple(float(value) for value in target_size)
            contexts.append(context)
        render_selected_artifacts(
            {
                "Prepared target": tuple(targets),
                **({"Global pocket": pocket} if pocket is not None else {}),
            }
        )
        st.markdown("#### 3D target and docking box")
        displayed_context = contexts[0]
        if len(contexts) > 1:
            context_labels = {
                (
                    f"{_target_label(context['target'])} · job "
                    f"{display_job_code(context['target'].job.metadata.get('job_code'), context['target'].job.run_id)}"
                ): context
                for context in contexts
            }
            displayed_label = st.selectbox(
                "Displayed target model",
                tuple(context_labels),
                key="binding_displayed_ensemble_target",
            )
            displayed_context = context_labels[displayed_label]
        displayed_target = displayed_context["target"]
        displayed_center = displayed_context["center"]
        displayed_size = displayed_context["size"]
        if len(contexts) > 1 and selected_classical_context:
            st.caption(
                "This model's automatic box: "
                + " × ".join(f"{value:.2f}" for value in displayed_size)
                + " Å at "
                + ", ".join(f"{value:.3f}" for value in displayed_center)
                + "."
            )
        render_target_viewer(
            displayed_target,
            viewer_path=target_viewer_path(displayed_target),
            ligand_path=displayed_context["associated_ligand"],
            box=(
                {"center": displayed_center, "size": displayed_size}
                if selected_classical_context
                else None
            ),
            selected_ligand_key=str(
                displayed_target.job.metadata.get("ligand_key") or ""
            ),
            cartoon_color="#94a3b8",
            ligand_color="cyanCarbon",
            box_color="#0891b2",
            show_box_center=selected_classical_context,
            coordinate_transform=displayed_context["alignment_transform"],
            key="binding_target_viewer",
            height=620,
        )
        return contexts


def _queue_target_engine_jobs(
    *,
    context: dict[str, Any],
    engines: list[str],
    selected_compounds: tuple[ArtifactChoice, ...],
    compound_paths: list[Any],
    reference: ArtifactChoice | None,
    target_ligand_only: bool,
    launch_campaign_id: str,
    launch_campaign_label: str,
    campaign_purpose: str,
    config: dict[str, Any],
    msa_dependency_ids: tuple[str, ...] = (),
) -> tuple[list[str], list[str], str]:
    target: ArtifactChoice = context["target"]
    associated_ligand = context["associated_ligand"]
    associated_ligand_artifact = context["associated_ligand_artifact"]
    target_path = target.artifact.resolve(
        target.job.run_dir, must_exist=True
    )
    if target_ligand_only:
        selected_compounds = (
            ArtifactChoice(job=target.job, artifact=associated_ligand_artifact),
        )
        compound_paths = [associated_ligand]
        reference_path = associated_ligand
    else:
        reference_path = (
            reference.artifact.resolve(
                reference.job.run_dir, must_exist=True
            )
            if reference is not None
            else None
        )
    inherited_reference_path = (
        associated_ligand
        if reference is None and associated_ligand_artifact is not None
        else None
    )
    if target_path is None or any(path is None for path in compound_paths):
        return [], ["One or more selected artifacts are unavailable"], ""
    try:
        launch_target_path = target_path
        launch_target_artifact = target.artifact
        launch_reference_path = reference_path or inherited_reference_path
        launch_reference_artifact = (
            reference.artifact
            if reference is not None
            else associated_ligand_artifact
        )
        orientation_job = None
        if context["alignment_transform"] is not None:
            axis_ligand_artifact = _artifact_for_path(
                target, associated_ligand
            )
            if associated_ligand is None or axis_ligand_artifact is None:
                raise ValueError(
                    "The associated coordinate ligand could not be resolved "
                    "as a typed source artifact"
                )
            additional_ligands = []
            if (
                reference_path is not None
                and reference is not None
                and (
                    reference.artifact.run_id,
                    reference.artifact.artifact_id,
                )
                != (
                    axis_ligand_artifact.run_id,
                    axis_ligand_artifact.artifact_id,
                )
            ):
                additional_ligands.append(
                    (reference_path, reference.artifact)
                )
            orientation_job = create_axis_aligned_target_job(
                source_job=target.job,
                source_artifact=target.artifact,
                source_path=target_path,
                axis_ligand_path=associated_ligand,
                axis_ligand_artifact=axis_ligand_artifact,
                additional_ligands=additional_ligands,
            )
            oriented_targets = (
                orientation_job.artifact_manifest.by_type("prepared_target")
                if orientation_job.artifact_manifest is not None
                else ()
            )
            if not oriented_targets:
                raise ValueError("Axis alignment produced no prepared target")
            launch_target_artifact = oriented_targets[0]
            launch_target_path = launch_target_artifact.resolve(
                orientation_job.run_dir, must_exist=True
            )
            if launch_target_path is None:
                raise ValueError(
                    "The axis-aligned prepared target is unavailable"
                )
            if launch_reference_artifact is not None:
                oriented_ligands = (
                    orientation_job.artifact_manifest.by_type(
                        "prepared_ligand_set"
                    )
                )
                launch_reference_artifact = next(
                    (
                        artifact
                        for artifact in oriented_ligands
                        if str(
                            artifact.metadata.get("source_artifact_id") or ""
                        )
                        == launch_reference_artifact.artifact_id
                    ),
                    None,
                )
                if launch_reference_artifact is None:
                    raise ValueError(
                        "The coordinate-matched reference ligand was not "
                        "emitted in the shared oriented frame"
                    )
                launch_reference_path = launch_reference_artifact.resolve(
                    orientation_job.run_dir, must_exist=True
                )
                if launch_reference_path is None:
                    raise ValueError(
                        "The oriented reference-ligand artifact is unavailable"
                    )
        common_paths = [path for path in compound_paths if path is not None]
        compound_artifacts = [
            choice.artifact for choice in selected_compounds
        ]
        if target_ligand_only:
            common_paths = (
                [launch_reference_path]
                if launch_reference_path is not None
                else []
            )
            compound_artifacts = (
                [launch_reference_artifact]
                if launch_reference_artifact is not None
                else []
            )
        queued: list[str] = []
        failures: list[str] = []
        for engine in engines:
            try:
                if engine in {
                    "AutoDock Vina",
                    "GNINA",
                    "Uni-Dock Pro",
                }:
                    engine_id = {
                        "AutoDock Vina": "vina",
                        "GNINA": "gnina",
                        "Uni-Dock Pro": "udp",
                    }[engine]
                    job = queue_docking_campaign_job(
                        receptor_path=launch_target_path,
                        target_artifact=launch_target_artifact,
                        compound_paths=common_paths,
                        compound_artifacts=compound_artifacts,
                        center=context["center"],
                        size=context["size"],
                        box_mode=context["box_mode"],
                        box_padding_angstrom=context["box_padding"],
                        engine=engine_id,
                        image=DEFAULT_DOCKING_IMAGE,
                        gpu_device=(
                            config["docking_gpu_device"]
                            if engine in {"GNINA", "Uni-Dock Pro"}
                            else config["gpu_device"]
                        ),
                        mode=(
                            config["docking_mode"]
                            if engine == "Uni-Dock Pro"
                            else "classic"
                        ),
                        search_mode=config["search_mode"],
                        exhaustiveness=config["exhaustiveness"],
                        poses=config["poses"],
                        use_scrub=config["use_scrub"],
                        scrub_ph=config["scrub_ph"],
                        scrub_skip_tautomer=config["scrub_skip_tautomer"],
                        reference_ligand_path=launch_reference_path,
                        reference_ligand_artifact=(
                            launch_reference_artifact
                        ),
                        replicates=config["campaign_replicates"],
                        seed_start=config["campaign_seed"],
                        maximum_compounds=config["maximum_compounds"],
                        cpu_workers=cpu_process_limit(),
                        compound_timeout_minutes=config["vina_compound_timeout_minutes"],
                        extra_args=shlex.split(config["extra_args_text"]),
                        launch_campaign_id=launch_campaign_id,
                        launch_campaign_label=launch_campaign_label,
                        campaign_purpose=campaign_purpose,
                    )
                elif engine == "RosettaLigand":
                    job = queue_openvs_docking_job(
                        receptor_path=launch_target_path,
                        target_artifact=launch_target_artifact,
                        compound_paths=common_paths,
                        compound_artifacts=compound_artifacts,
                        center=context["center"],
                        size=context["size"],
                        box_mode=context["box_mode"],
                        box_padding_angstrom=context["box_padding"],
                        protocol=config["openvs_protocol"],
                        reference_mode=config["openvs_reference_mode"],
                        reference_ligand_path=launch_reference_path,
                        reference_ligand_artifact=(
                            launch_reference_artifact
                        ),
                        image=DEFAULT_OPENVS_IMAGE,
                        cpu_workers=cpu_process_limit(),
                        ph=config["openvs_ph"],
                        preserve_input_protonation=(
                            bool(compound_artifacts)
                            and all(
                                bool(
                                    artifact.metadata.get(
                                        "modeling_state_prepared"
                                    )
                                )
                                for artifact in compound_artifacts
                            )
                        ),
                        conformers=config["openvs_conformers"],
                        minimization_steps=config["openvs_steps"],
                        padding=config["openvs_padding"],
                        replicates=config["campaign_replicates"],
                        seed_start=config["campaign_seed"],
                        cluster_threshold_angstrom=config["openvs_cluster"],
                        maximum_compounds=config["maximum_compounds"],
                        launch_campaign_id=launch_campaign_id,
                        launch_campaign_label=launch_campaign_label,
                        campaign_purpose=campaign_purpose,
                    )
                elif engine == "Boltz-2":
                    job = queue_boltz2_refolding_job(
                        target_path=launch_target_path,
                        target_artifact=launch_target_artifact,
                        compound_paths=common_paths,
                        compound_artifacts=compound_artifacts,
                        reference_ligand_artifact=(
                            launch_reference_artifact
                        ),
                        image=DEFAULT_BOLTZ2_IMAGE,
                        cache_dir=configured_boltz2_cache_dir(),
                        gpu_device=config["structure_gpu_device"],
                        max_compounds=config["boltz_max"],
                        recycling_steps=config["boltz_recycles"],
                        sampling_steps=config["boltz_steps"],
                        diffusion_samples=config["boltz_samples"],
                        replicates=config["campaign_replicates"],
                        seed_start=config["campaign_seed"],
                        launch_campaign_id=launch_campaign_id,
                        launch_campaign_label=launch_campaign_label,
                        campaign_purpose=campaign_purpose,
                    )
                elif engine == "AlphaFold 3":
                    job = queue_alphafold3_refolding_job(
                        target_path=launch_target_path,
                        target_artifact=launch_target_artifact,
                        compound_paths=common_paths,
                        compound_artifacts=compound_artifacts,
                        reference_ligand_artifact=(
                            launch_reference_artifact
                        ),
                        image=DEFAULT_ALPHAFOLD3_IMAGE,
                        db_dir=config["af3_db_dir"],
                        weights_dir=config["af3_weights_dir"],
                        msa_repository_dir=config["af3_msa_dir"],
                        gpu_device=config["structure_gpu_device"],
                        max_compounds=config["af3_max"],
                        batch_size=config["af3_batch"],
                        num_recycles=config["af3_recycles"],
                        model_seed_count=config["campaign_replicates"],
                        model_seed_start=config["campaign_seed"],
                        launch_campaign_id=launch_campaign_id,
                        launch_campaign_label=launch_campaign_label,
                        campaign_purpose=campaign_purpose,
                    )
                else:
                    job = queue_nesso_affinity_job(
                        target_path=launch_target_path,
                        target_artifact=launch_target_artifact,
                        compound_paths=common_paths,
                        compound_artifacts=compound_artifacts,
                        image=DEFAULT_NESSO_IMAGE,
                        checkpoint_dir=config["nesso_checkpoint"],
                        ccd_path=config["nesso_ccd"],
                        esm_cache_dir=config["nesso_esm_cache"],
                        gpu_device=config["structure_gpu_device"],
                        max_compounds=config["nesso_max"],
                        recycling_steps=config["nesso_recycles"],
                        refine_protein_cutoff=config["nesso_refine"],
                        refine_protein_tokens_budget=config["nesso_tokens"],
                        affinity_protein_cutoff=config["nesso_affinity"],
                        seed=config["campaign_seed"],
                        replicates=config["campaign_replicates"],
                        launch_campaign_id=launch_campaign_id,
                        launch_campaign_label=launch_campaign_label,
                        campaign_purpose=campaign_purpose,
                    )
                if engine in COFOLDING_ENGINES and msa_dependency_ids:
                    metadata_path = job.run_dir / "metadata.json"
                    job_metadata = json.loads(metadata_path.read_text())
                    job_metadata["depends_on_run_ids"] = list(msa_dependency_ids)
                    job_metadata["campaign_phase"] = "structure_after_msa"
                    if engine in {"AlphaFold 3", "Boltz-2"}:
                        job_metadata["msa_preparation_required"] = True
                        job_metadata["msa_repository_dir"] = str(config["af3_msa_dir"])
                    if engine == "Boltz-2":
                        job_metadata["protein_sequences"] = [
                            sequence
                            for kind, _chain, sequence in polymer_sequences_from_pdb(
                                launch_target_path
                            )
                            if kind == "protein"
                        ]
                    metadata_path.write_text(json.dumps(job_metadata, indent=2) + "\n")
                queued.append(
                    f"{engine} "
                    + display_job_code(
                        job.metadata.get("job_code"), job.run_id
                    )
                )
            except Exception as exc:
                failures.append(f"{engine}: {exc}")
        orientation_note = (
            "axis-aligned target "
            + display_job_code(
                orientation_job.metadata.get("job_code"),
                orientation_job.run_id,
            )
            if orientation_job is not None
            else ""
        )
        return queued, failures, orientation_note
    except Exception as exc:
        return [], [f"Could not prepare target frame: {exc}"], ""


@st.cache_data(show_spinner=False, ttl=30)
def _cached_job_rows(
    statuses: tuple[str, ...], runs_dir_text: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_jobs = job_records_snapshot(runs_dir_text)
    jobs_by_id = {job.run_id: job for job in all_jobs}
    for job in all_jobs:
        if is_benchmark_job(job, jobs_by_id):
            continue
        operation = str(job.metadata.get("operation") or job.metadata.get("mode") or "").lower()
        if job.workflow == "redocking_benchmark" or operation == "redocking":
            continue
        if operation not in {"docking", "refolding"} and job.task_group not in {
            "docking",
            "batch-docking",
            "structure-docking",
        }:
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
                "kind": "cofolding" if operation == "refolding" else "docking",
                "engine": job.tool or "Unspecified",
                "status": job.status,
                "created": job.created_at,
            }
        )
    return rows


def _job_rows(statuses: set[str]) -> list[dict[str, object]]:
    return _cached_job_rows(
        tuple(sorted(statuses)), str(runs_root().resolve())
    )


def _render_jobs(statuses: set[str]) -> None:
    rows = _job_rows(statuses)
    if not rows:
        st.info("No matching jobs.")
        return
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"result": st.column_config.LinkColumn("Result", display_text="Open")},
    )


def render(
    *,
    title: str = "Docking / Cofolding",
    caption: str = (
        "Run one prepared compound list against one or more targets with "
        "target-specific combinations of classical docking and structure-based "
        "cofolding engines."
    ),
    target_ligand_only: bool = False,
) -> None:
    st.title(title)
    st.caption(caption)

    selected_classical_context = any(
        bool(st.session_state.get(ENGINE_KEYS[engine], engine == "AutoDock Vina"))
        for engine in CLASSICAL_ENGINES
    )
    hidden_compounds = None
    if target_ligand_only:
        target_tab, engines_tab, run_tab, results_tab = st.tabs(
            ["Target", "Engines", "Run", "Results"]
        )
        hidden_compounds = st.empty()
        compounds_tab = hidden_compounds.container()
    else:
        target_tab, compounds_tab, engines_tab, run_tab, results_tab = st.tabs(
            ["Target", "Compounds", "Engines", "Run", "Results"]
        )

    with target_tab:
        requested_target_run_id = str(
            st.query_params.get("target_run_id", "") or ""
        ).strip()
        target_mode = st.segmented_control(
            "Target mode",
            ("Single target", "Multi-target ensemble"),
            default="Single target",
            key="binding_target_mode",
            help=(
                "Single target restricts the inventory to one selected row. "
                "Multi-target ensemble permits multiple similar, aligned targets."
            ),
        ) or "Single target"
        ensemble_mode = target_mode == "Multi-target ensemble"
        targets = select_target_artifacts(
            "Prepared targets",
            ("prepared_target", "prepared_receptor"),
            key="binding_targets",
            maximum=None if ensemble_mode else 1,
            requested_run_ids=(
                {requested_target_run_id}
                if requested_target_run_id
                else None
            ),
        )
        target_contexts: list[dict[str, Any]] = []
        if targets:
            target_contexts = _render_target_contract(
                targets,
                selected_classical_context=selected_classical_context,
                ensemble_mode=ensemble_mode,
            )
        target = targets[0] if targets else None
        if target_contexts:
            primary_context = target_contexts[0]
            pocket = primary_context["pocket"]
            associated_ligand = primary_context["associated_ligand"]
            associated_ligand_artifact = primary_context[
                "associated_ligand_artifact"
            ]
            alignment_transform = primary_context["alignment_transform"]
            center = primary_context["center"]
            size = primary_context["size"]
            box_mode = primary_context["box_mode"]
            box_padding = primary_context["box_padding"]
        else:
            pocket = None
            associated_ligand = None
            associated_ligand_artifact = None
            alignment_transform = None
            center, size = (0.0, 0.0, 0.0), (22.0, 22.0, 22.0)
            box_mode, box_padding = "fixed", None
        if not targets:
            st.info("A prepared target is required. Continue to Run for the preparation link.")

    ordered_target_box_sizes = [
        sorted(float(value) for value in context["size"])
        for context in target_contexts
    ]
    campaign_box_fit_size = (
        tuple(
            min(box_size[index] for box_size in ordered_target_box_sizes)
            for index in range(3)
        )
        if ordered_target_box_sizes
        else tuple(float(value) for value in size)
    )

    with compounds_tab:
        imported_options = (
            {} if target_ligand_only else _imported_compound_options()
        )
        if imported_options:
            requested_compound_run = str(
                st.query_params.get("compound_run_id", "") or ""
            ).strip()
            dataset_labels = list(imported_options)
            default_dataset_index = next(
                (
                    index
                    for index, label in enumerate(dataset_labels)
                    if imported_options[label].job.run_id
                    == requested_compound_run
                ),
                0,
            )
            selected_dataset_label = st.selectbox(
                "Imported compound dataset",
                dataset_labels,
                index=default_dataset_index,
                key="binding_imported_dataset",
            )
            dataset_choice = imported_options[selected_dataset_label]
            compounds = (dataset_choice,)
        else:
            st.selectbox(
                "Imported compound dataset",
                ["No compatible compound datasets"],
                disabled=True,
                key="binding_imported_dataset_missing",
            )
            dataset_choice = None
            compounds = ()
        reference = (
            None
            if target_ligand_only
            else select_artifact(
                "Optional reference-ligand override",
                ("prepared_ligand_set",),
                key="binding_reference",
                required=False,
            )
        )
        selected_inputs: dict[str, object] = {}
        if compounds:
            selected_inputs["Imported compound dataset"] = compounds
        if reference is not None:
            selected_inputs["Explicit reference override"] = reference
        elif (
            target is not None
            and associated_ligand_artifact is not None
        ):
            selected_inputs["Target-associated reference ligand"] = ArtifactChoice(
                job=target.job,
                artifact=associated_ligand_artifact,
            )
        render_selected_artifacts(selected_inputs)
        if reference is None and associated_ligand_artifact is not None:
            st.caption(
                "The coordinate-matched ligand associated with the selected target "
                "will be inherited automatically as the reference for provenance "
                "and result overlays."
            )
        parent_frame = (
            _docking_parent_frame(dataset_choice)
            if dataset_choice is not None
            else pd.DataFrame()
        )
        box_warning_frame = pd.DataFrame()
        exclude_oversized = False
        fit_box_size: tuple[float, float, float] | None = None
        if not parent_frame.empty and selected_classical_context:
            st.markdown("#### Docking-box fit preflight")
            fit_signature = (
                dataset_choice.job.run_id if dataset_choice is not None else "",
                tuple(round(float(value), 3) for value in campaign_box_fit_size),
            )
            if st.button(
                "Analyze dataset against current box",
                key="binding_analyze_box_fit",
                help=(
                    "Generate one deterministic 3D conformer per unique parent and "
                    "compare its principal-axis dimensions with the full box. Missing "
                    "estimates are calculated concurrently and stored for reuse."
                ),
            ):
                st.session_state["binding_box_fit_analysis"] = {
                    "signature": fit_signature,
                    "rows": compound_box_fit_rows(
                        parent_frame.to_dict("records"),
                        box_size=campaign_box_fit_size,
                    ),
                }
            fit_analysis = st.session_state.get("binding_box_fit_analysis")
            if (
                isinstance(fit_analysis, dict)
                and fit_analysis.get("signature") == fit_signature
                and isinstance(fit_analysis.get("rows"), list)
            ):
                parent_frame = pd.DataFrame(fit_analysis["rows"])
                fit_box_size = campaign_box_fit_size
                oversized_count = int(
                    parent_frame["box_fit_status"]
                    .astype(str)
                    .eq("likely too large")
                    .sum()
                )
                fit_metrics = st.columns(3)
                fit_metrics[0].metric(
                    "Smallest selected-target box",
                    " × ".join(
                        f"{value:.1f}" for value in campaign_box_fit_size
                    )
                    + " Å",
                )
                fit_metrics[1].metric("Likely too large", f"{oversized_count:,}")
                fit_metrics[2].metric(
                    "Not estimated",
                    f"{int(parent_frame['box_fit_status'].astype(str).eq('not estimated').sum()):,}",
                )
                box_warning_frame = parent_frame.loc[
                    parent_frame["box_fit_status"]
                    .astype(str)
                    .eq("likely too large")
                ].copy()
                if not box_warning_frame.empty:
                    box_warning_frame = box_warning_frame.sort_values(
                        "box_fit_max_excess_angstrom",
                        ascending=False,
                    ).reset_index(drop=True)
                handling = st.radio(
                    "Likely too-large compounds",
                    ("Keep and warn", "Exclude from all selected engines"),
                    horizontal=True,
                    key="binding_oversized_handling",
                    help=(
                        "Exclusion is explicit and is recorded in the immutable selection. "
                        "When classical and cofolding engines are launched together, the "
                        "same final membership is shared by every engine."
                    ),
                )
                exclude_oversized = handling.startswith("Exclude")
                st.caption(
                    "The full box dimensions are compared with sorted principal molecular "
                    "dimensions; no clearance is subtracted. Docking coordinates are not "
                    "rotated by this analysis. Raw estimates are persisted and reused for "
                    "other box sizes."
                )
            else:
                st.caption(
                    "Optional preflight: estimate prepared-ligand dimensions and identify "
                    "parents unlikely to fit the current docking box. No compound is removed "
                    "unless you explicitly choose exclusion after analysis."
                )
        selection_mode = st.radio(
            "Compound selection",
            ("Manual selection", "All unique parents"),
            horizontal=True,
            key="binding_compound_selection_mode_v2",
            help=(
                "No compound is selected by default. Select one or more table "
                "rows, or explicitly switch to All unique parents."
            ),
        )
        selected_parent_indices: list[int] = []
        selected_oversized = 0
        excluded_parent_rows: list[dict[str, object]] = []
        if not parent_frame.empty:
            display_frame = parent_frame[
                _parent_display_columns(parent_frame)
            ]
            if selection_mode == "All unique parents":
                st.dataframe(
                    display_frame,
                    hide_index=True,
                    width="stretch",
                    height=min(500, 38 + 35 * min(len(display_frame), 13)),
                    key="binding_parent_table_all",
                )
                selected_parent_indices = list(range(len(parent_frame)))
            else:
                selection_frame = display_frame.copy()
                selection_frame.insert(0, "Selected", False)
                if not selection_frame.empty:
                    selection_frame.loc[selection_frame.index[0], "Selected"] = True
                edited_selection = st.data_editor(
                    selection_frame,
                    hide_index=True,
                    width="stretch",
                    height=min(500, 38 + 35 * min(len(display_frame), 13)),
                    key="binding_parent_table_manual",
                    disabled=[
                        column
                        for column in selection_frame.columns
                        if column != "Selected"
                    ],
                    column_config={
                        "Selected": st.column_config.CheckboxColumn(
                            "Selected",
                            help="Compounds included in this campaign.",
                        )
                    },
                )
                selected_parent_indices = [
                    int(index)
                    for index, selected in edited_selection["Selected"].items()
                    if bool(selected) and 0 <= int(index) < len(parent_frame)
                ]
                st.caption(
                    "The first compound is visibly checked by default. Tick or "
                    "untick rows, or switch to All unique parents."
                )
            if not box_warning_frame.empty:
                warning_columns = [
                    column
                    for column in (
                        "representative_compound_id",
                        "representative_product_name",
                        "estimated_3d_length_angstrom",
                        "estimated_3d_width_angstrom",
                        "estimated_3d_thickness_angstrom",
                        "box_fit_reason",
                        "box_fit_max_excess_angstrom",
                    )
                    if column in box_warning_frame.columns
                ]
                st.markdown("##### Compounds exceeding the current box")
                st.caption(
                    "Select one row to inspect the same structure and complete "
                    "compound information available in Compound Dataset Results."
                )
                warning_event = st.dataframe(
                    box_warning_frame[warning_columns],
                    hide_index=True,
                    width="stretch",
                    height=min(
                        360,
                        38 + 35 * len(box_warning_frame),
                    ),
                    on_select="rerun",
                    selection_mode="single-row",
                    key="binding_box_warning_table",
                )
                warning_selection = [
                    int(index)
                    for index in warning_event.selection.rows
                    if 0 <= int(index) < len(box_warning_frame)
                ]
                if warning_selection:
                    render_selected_compound(
                        box_warning_frame,
                        warning_selection[0],
                    )
            selected_parent_rows = (
                parent_frame.iloc[selected_parent_indices].to_dict("records")
            )
            selected_oversized = sum(
                str(row.get("box_fit_status") or "") == "likely too large"
                for row in selected_parent_rows
            )
            if exclude_oversized:
                excluded_parent_rows = [
                    row
                    for row in selected_parent_rows
                    if str(row.get("box_fit_status") or "")
                    == "likely too large"
                ]
                selected_parent_rows = [
                    row
                    for row in selected_parent_rows
                    if str(row.get("box_fit_status") or "") != "likely too large"
                ]
            compound_metrics = st.columns(3)
            compound_metrics[0].metric(
                "Available unique parents", f"{len(parent_frame):,}"
            )
            compound_metrics[1].metric(
                "Selected compounds", f"{len(selected_parent_rows):,}"
            )
            compound_metrics[2].metric(
                "Fit warnings before exclusion", f"{selected_oversized:,}"
            )
            if selected_oversized and not exclude_oversized:
                st.warning(
                    f"{selected_oversized:,} selected compound(s) are likely too large "
                    "for at least one full box dimension. They will be retained."
                )
        else:
            selected_parent_rows = []
            if compounds:
                st.warning(
                    "The imported dataset contains no unambiguous unique parents."
                )
            else:
                st.info(
                    "An imported compound dataset is required. Continue to Run "
                    "for the dataset link."
                )
    if target_ligand_only:
        hidden_compounds.empty()
        dataset_choice = None
        compounds = ()
        reference = None
        selected_parent_rows = []
        selection_mode = "Target-associated ligand"
        selected_oversized = 0
        excluded_parent_rows = []
        exclude_oversized = False
        fit_box_size = None

    af3_db_dir, af3_weights_dir, af3_msa_dir = configured_alphafold3_reference_paths()
    af3_status = alphafast_readiness(af3_db_dir, af3_weights_dir, af3_msa_dir)
    af3_ready = bool(af3_status["database_ready"] and af3_status["weights_ready"])
    boltz_status = boltz2_readiness()
    boltz_ready = bool(boltz_status["ready"])
    nesso_checkpoint, nesso_ccd, nesso_esm_cache = configured_nesso_reference_paths()
    nesso_status = nesso_readiness(nesso_checkpoint, nesso_ccd, nesso_esm_cache)
    nesso_ready = bool(nesso_status["ready"])
    selected_engines_by_target: dict[str, list[str]] = {}

    with engines_tab:
        action_columns = st.columns(2)
        action_columns[0].button(
            "Select all engines",
            on_click=_set_engine_selection,
            args=(True,),
            key="binding_select_all_engines",
        )
        action_columns[1].button(
            "Deselect all engines",
            on_click=_set_engine_selection,
            args=(False,),
            key="binding_clear_all_engines",
        )
        st.markdown("#### Classical docking")
        classical_columns = st.columns(len(CLASSICAL_ENGINES))
        for column, engine in zip(classical_columns, CLASSICAL_ENGINES):
            column.checkbox(
                engine,
                value=True,
                key=ENGINE_KEYS[engine],
            )
        st.markdown("#### Cofolding / affinity")
        cofolding_columns = st.columns(len(COFOLDING_ENGINES))
        readiness = {"Boltz-2": boltz_ready, "AlphaFold 3": af3_ready, "Nesso-1": nesso_ready}
        for column, engine in zip(cofolding_columns, COFOLDING_ENGINES):
            column.checkbox(engine, value=True, key=ENGINE_KEYS[engine])
        campaign_engines = [
            engine
            for engine in ALL_ENGINES
            if bool(st.session_state.get(ENGINE_KEYS[engine], False))
        ]
        if target_contexts:
            st.markdown("#### Engines by target")
            st.caption(
                "Checked cells are queued. This lets one named campaign use "
                "different engine combinations for different targets."
            )
            for context in target_contexts:
                target_choice = context["target"]
                assignment_columns = st.columns(
                    [1.6] + [1.0] * len(campaign_engines)
                )
                assignment_columns[0].markdown(
                    f"**{_target_label(target_choice)}**  \n"
                    + display_job_code(
                        target_choice.job.metadata.get("job_code"),
                        target_choice.job.run_id,
                    )
                )
                assigned: list[str] = []
                for column, engine in zip(
                    assignment_columns[1:], campaign_engines, strict=True
                ):
                    engine_key = engine.lower().replace(" ", "_").replace(
                        "-", "_"
                    )
                    if column.checkbox(
                        engine,
                        value=True,
                        key=(
                            "binding_target_engine_"
                            f"{target_choice.job.run_id}_{engine_key}"
                        ),
                    ):
                        assigned.append(engine)
                selected_engines_by_target[target_choice.job.run_id] = assigned
        st.caption(
            "Boltz-2 emits complexes, confidence and affinity. AlphaFold 3 emits complexes and "
            "structural/interface confidence (ranking score, ipTM, pTM and clash status), not a "
            "binding affinity. Nesso-1 emits affinity only and no predicted structure."
        )
        if not boltz_ready:
            st.warning("Boltz-2 checkpoints are unavailable.")
        if not af3_ready:
            st.warning("AlphaFold 3 databases or weights are unavailable.")
        if not nesso_ready:
            st.warning("Nesso-1 checkpoint, CCD or ESM cache is unavailable.")

        with st.expander("Shared campaign repetitions", expanded=True):
            shared = st.columns(2)
            campaign_replicates = int(
                shared[0].number_input(
                    "Independent runs per structure",
                    min_value=1,
                    max_value=100,
                    value=1,
                    key="binding_campaign_replicates",
                    help=(
                        "Applied uniformly to Uni-Dock Pro, AutoDock Vina, GNINA, "
                        "RosettaLigand, Boltz-2, AlphaFold 3 and Nesso-1. For "
                        "AlphaFold 3, each independent run is a distinct model seed."
                    ),
                )
            )
            campaign_seed = int(
                shared[1].number_input(
                    "First seed per structure",
                    min_value=1,
                    max_value=2_147_483_000,
                    value=1001,
                    key="binding_campaign_seed_start",
                    help=(
                        "Applied uniformly to every selected engine. Independent "
                        "runs use consecutive seeds beginning with this value; "
                        "for AlphaFold 3 this is the first model seed."
                    ),
                )
            )

        with st.expander("Classical docking preparation", expanded=True):
            classical = st.columns(3)
            maximum_compounds = int(
                classical[0].number_input(
                    "Maximum compounds",
                    min_value=0,
                    max_value=unidock_pro_max_compounds(),
                    value=0,
                    step=100,
                    key="binding_maximum_compounds",
                    help=(
                        "Use 0 for every compound when the dataset does not exceed "
                        "the configured Uni-Dock Pro batch limit. Change that limit "
                        "on the Settings page."
                    ),
                )
            )
            use_scrub = classical[1].checkbox(
                "Use scrub.py", value=True, key="binding_use_scrub"
            )
            scrub_ph = float(
                classical[2].number_input(
                    "Scrub pH", min_value=0.0, max_value=14.0, value=7.4, step=0.1,
                    key="binding_scrub_ph",
                )
            )
            scrub_skip_tautomer = st.checkbox(
                "Skip tautomer enumeration", value=True, key="binding_skip_tautomer"
            )

        with st.expander(
            "AutoDock Vina and GNINA settings",
            expanded=bool(
                st.session_state.get(ENGINE_KEYS["AutoDock Vina"], False)
                or st.session_state.get(ENGINE_KEYS["GNINA"], False)
            ),
        ):
            vina = st.columns(4)
            exhaustiveness = int(
                vina[0].number_input(
                    "Exhaustiveness", min_value=1, max_value=1024, value=30,
                    key="binding_exhaustiveness",
                )
            )
            poses = int(
                vina[1].number_input(
                    "Poses per compound", min_value=1, max_value=100, value=10,
                    key="binding_poses",
                )
            )
            extra_args_text = vina[2].text_input(
                "Additional native arguments", value="", key="binding_extra_args"
            )
            vina_compound_timeout = int(
                vina[3].number_input(
                    "Vina / GNINA timeout per compound (minutes)",
                    min_value=1,
                    max_value=1440,
                    value=vina_compound_timeout_minutes(),
                    step=1,
                    key="binding_vina_compound_timeout",
                    help=(
                        "A Vina or GNINA compound exceeding this limit is reported "
                        "as an exclusion and skipped in later replicas of that job."
                    ),
                )
            )

        with st.expander(
            "Uni-Dock Pro settings",
            expanded=bool(
                st.session_state.get(ENGINE_KEYS["Uni-Dock Pro"], False)
            ),
        ):
            udp = st.columns(3)
            docking_mode = udp[0].selectbox(
                "Docking mode", ("classic", "hybrid"), key="unidock_mode"
            )
            search_mode = udp[1].selectbox(
                "Search mode", ("fast", "balance", "detail"), index=2,
                key="binding_search_mode",
            )
            udp[2].caption("Hybrid mode requires the optional reference ligand.")

        with st.expander(
            "RosettaLigand settings",
            expanded=bool(
                st.session_state.get(ENGINE_KEYS["RosettaLigand"], False)
            ),
        ):
            openvs = st.columns(3)
            protocol_label = openvs[0].selectbox(
                "Protocol",
                ("VSH — high precision", "VSX — express", "Convergence — exhaustive multi-seed VSH"),
                key="openvs_protocol",
            )
            openvs_protocol = (
                "vsh" if str(protocol_label).startswith("VSH")
                else "vsx" if str(protocol_label).startswith("VSX")
                else "convergence"
            )
            reference_label = openvs[1].selectbox(
                "Placement", ("Reference-guided", "Pocket-center (unguided)"),
                index=0,
                key="openvs_reference_mode",
                help=(
                    "Reference-guided is recommended when the target has a "
                    "coordinate-matched ligand. Choose pocket-center placement "
                    "for targets without a usable reference ligand."
                ),
            )
            openvs_reference_mode = (
                "reference_guided" if reference_label == "Reference-guided" else "pocket_center"
            )
            openvs_padding = float(
                openvs[2].number_input(
                    "Search padding (Å)", min_value=1.0, max_value=20.0, value=4.0, step=0.5,
                    key="openvs_padding",
                )
            )
            openvs_prep = st.columns(4)
            openvs_ph = float(
                openvs_prep[0].number_input(
                    "Preparation pH", min_value=0.0, max_value=14.0, value=7.4, step=0.1,
                    key="openvs_ph",
                )
            )
            openvs_conformers = int(
                openvs_prep[1].number_input(
                    "Conformer trials", min_value=1, max_value=500, value=20,
                    key="openvs_conformers",
                )
            )
            openvs_steps = int(
                openvs_prep[2].number_input(
                    "Minimization steps", min_value=1, max_value=10000, value=2000, step=100,
                    key="openvs_minimization_steps",
                )
            )
            openvs_workers = cpu_process_limit()
            openvs_prep[3].metric(
                "CPU workers",
                openvs_workers,
                help=(
                    "Inherited from the system-wide CPU process limit. "
                    "Rosetta ligand/replicate chunks run concurrently."
                ),
            )
            openvs_cluster = float(
                st.number_input(
                    "Pose-cluster threshold (Å)", min_value=0.1, max_value=10.0,
                    value=2.0, step=0.1, key="openvs_cluster_threshold",
                )
            )

        with st.expander(
            "Boltz-2 settings",
            expanded=bool(st.session_state.get(ENGINE_KEYS["Boltz-2"], False)),
        ):
            boltz = st.columns(4)
            boltz_recycles = int(boltz[0].number_input("Recycles", 1, 12, 3, key="boltz_recycles"))
            boltz_samples = int(
                boltz[1].number_input("Diffusion samples", 1, 16, 5, key="boltz_samples")
            )
            boltz_steps = int(
                boltz[2].number_input("Sampling steps", 10, 400, 200, 10, key="boltz_steps")
            )
            boltz_max = int(
                boltz[3].number_input("Maximum compounds", min_value=0, value=0, key="boltz_max")
            )

        with st.expander(
            "AlphaFold 3 settings",
            expanded=bool(
                st.session_state.get(ENGINE_KEYS["AlphaFold 3"], False)
            ),
        ):
            af3 = st.columns(3)
            af3_recycles = int(af3[0].number_input("Recycles", 1, 48, 10, key="af3_recycles"))
            af3_max = int(
                af3[1].number_input("Maximum compounds", min_value=0, value=0, key="af3_max")
            )
            af3_batch = int(af3[2].number_input("Batch size", 1, 100000, 1, key="af3_batch"))

        with st.expander(
            "Nesso-1 settings",
            expanded=bool(st.session_state.get(ENGINE_KEYS["Nesso-1"], False)),
        ):
            st.caption("Affinity-only coarse-grained cofolding; no pose or predicted complex.")
            nesso = st.columns(3)
            nesso_recycles = int(nesso[0].number_input("Recycles", 1, 12, 5, key="nesso_recycles"))
            nesso_tokens = int(
                nesso[1].number_input("Pocket token budget", 32, 2048, 256, 32, key="nesso_tokens")
            )
            nesso_max = int(
                nesso[2].number_input("Maximum compounds", min_value=0, value=0, key="nesso_max")
            )
            nesso_more = st.columns(2)
            nesso_refine = float(
                nesso_more[0].number_input(
                    "Refinement cutoff (Å)", 1.0, 50.0, 22.0, 1.0, key="nesso_refine"
                )
            )
            nesso_affinity = float(
                nesso_more[1].number_input(
                    "Affinity cutoff (Å)", 1.0, 50.0, 15.0, 1.0, key="nesso_affinity"
                )
            )

    selected_engines = sorted(
        {
            engine
            for engines in selected_engines_by_target.values()
            for engine in engines
        },
        key=ALL_ENGINES.index,
    )
    with run_tab:
        selected_gpu = st.selectbox(
            "GPU", ("Automatic", "GPU 0", "GPU 1"), key="binding_gpu"
        )
        requires_gpu = any(engine != "RosettaLigand" for engine in selected_engines)
        render_run_resources(
            requires_gpu=requires_gpu,
            selected_gpu=selected_gpu,
            key="docking_cofolding",
        )
        st.markdown("#### Selected campaign")
        campaign_rows = [
            {
                "Target": _target_label(context["target"]),
                "Job": display_job_code(
                    context["target"].job.metadata.get("job_code"),
                    context["target"].job.run_id,
                ),
                "Engines": ", ".join(
                    selected_engines_by_target.get(
                        context["target"].job.run_id, []
                    )
                ),
            }
            for context in target_contexts
        ]
        if campaign_rows:
            st.dataframe(
                pd.DataFrame(campaign_rows),
                hide_index=True,
                width="stretch",
            )
        else:
            st.write("No targets selected.")
        blockers: list[str] = []
        if not target_contexts:
            blockers.append("Select at least one prepared target.")
            st.link_button("Open Structure Import", "./workspace-structure-preparation")
        if target_ligand_only:
            for context in target_contexts:
                target_choice = context["target"]
                target_name = _target_label(target_choice)
                assigned = selected_engines_by_target.get(
                    target_choice.job.run_id, []
                )
                if not assigned:
                    blockers.append(
                        f"Assign at least one engine to {target_name}."
                    )
                if (
                    context["associated_ligand"] is None
                    or context["associated_ligand_artifact"] is None
                ):
                    blockers.append(
                        f"{target_name} has no coordinate-associated ligand "
                        "to redock or refold."
                    )
        else:
            if not compounds:
                blockers.append("Select a compound dataset.")
                st.link_button("Open Compound Datasets", "./prepare-compound-datasets")
            elif not selected_parent_rows:
                blockers.append("Select at least one unique compound parent.")
        if not selected_engines:
            blockers.append("Select at least one engine.")
        for engine, ready in (
            ("Boltz-2", boltz_ready),
            ("AlphaFold 3", af3_ready),
            ("Nesso-1", nesso_ready),
        ):
            if engine in selected_engines and not ready:
                blockers.append(f"{engine} is selected but its installation-managed references are unavailable.")
        for context in target_contexts:
            target_choice = context["target"]
            target_name = _target_label(target_choice)
            assigned = selected_engines_by_target.get(
                target_choice.job.run_id, []
            )
            reference_available = (
                reference is not None
                or context["associated_ligand_artifact"] is not None
            )
            if (
                "Uni-Dock Pro" in assigned
                and docking_mode == "hybrid"
                and not reference_available
            ):
                blockers.append(
                    f"Uni-Dock Pro hybrid mode requires a reference ligand "
                    f"for {target_name}."
                )
            if (
                "RosettaLigand" in assigned
                and openvs_reference_mode == "reference_guided"
                and not reference_available
            ):
                blockers.append(
                    "Reference-guided RosettaLigand requires a coordinate-"
                    f"bearing reference ligand for {target_name}."
                )
        campaign_name = st.text_input(
            "Launch campaign name",
            key="binding_launch_campaign_name",
            placeholder=(
                "For example: 4LNW · 25 Å box · standard multi-engine screen"
            ),
            help=(
                "Every engine queued by this button receives one shared, "
                "immutable launch-campaign ID across all selected targets. A "
                "name is required and can later be used in "
                "Compound Campaign Comparison."
            ),
        )
        if not campaign_name.strip():
            blockers.append("Enter a launch campaign name.")
        for message in blockers:
            st.info(message)

        run_clicked = st.button(
            (
                "Queue selected redocking / refolding engines"
                if target_ligand_only
                else "Queue selected docking / cofolding engines"
            ),
            type="primary",
            disabled=bool(blockers),
            key="binding_queue_selected",
        )
        if run_clicked and target_contexts:
            launch_campaign_id = str(uuid4())
            campaign_purpose = (
                TARGET_LIGAND_CAMPAIGN_PURPOSE
                if target_ligand_only
                else COMPOUND_DATASET_CAMPAIGN_PURPOSE
            )
            launch_campaign_label = campaign_name.strip()
            if target_ligand_only:
                selected_compounds: tuple[ArtifactChoice, ...] = ()
                compound_paths: list[Any] = []
            else:
                selection_job = create_docking_parent_selection_job(
                    source_job=dataset_choice.job,
                    source_artifact=dataset_choice.artifact,
                    parent_rows=selected_parent_rows,
                    selection_mode=(
                        f"{selection_mode}; excluded {selected_oversized} "
                        "likely too-large compounds"
                        if exclude_oversized and selected_oversized
                        else selection_mode
                    ),
                    excluded_parent_rows=excluded_parent_rows,
                    box_size=fit_box_size,
                )
                selection_artifact = (
                    selection_job.artifact_manifest.by_type("compound_set")[0]
                )
                selected_compounds = (
                    ArtifactChoice(
                        job=selection_job,
                        artifact=selection_artifact,
                    ),
                )
                compound_paths = [
                    selection_artifact.resolve(
                        selection_job.run_dir, must_exist=True
                    )
                ]
            gpu_device = (
                str(selected_gpu).removeprefix("GPU ")
                if selected_gpu != "Automatic"
                else "all"
            )
            needs_msa_stage = any(
                engine in COFOLDING_ENGINES for engine in selected_engines
            )
            msa_dependency_ids: tuple[str, ...] = ()
            msa_queue_failures: list[str] = []
            if needs_msa_stage:
                unique_sequences: dict[str, tuple[str, ArtifactRef]] = {}
                for context in target_contexts:
                    target_choice = context["target"]
                    assigned = selected_engines_by_target.get(
                        target_choice.job.run_id, []
                    )
                    if not any(engine in COFOLDING_ENGINES for engine in assigned):
                        continue
                    target_path = target_choice.artifact.resolve(
                        target_choice.job.run_dir, must_exist=True
                    )
                    if target_path is None:
                        msa_queue_failures.append(
                            f"{_target_label(target_choice)}: target artifact is unavailable"
                        )
                        continue
                    for kind, _chain, sequence in polymer_sequences_from_pdb(target_path):
                        if kind == "protein":
                            unique_sequences.setdefault(
                                sequence,
                                (sequence, target_choice.artifact),
                            )
                msa_jobs = []
                if unique_sequences:
                    try:
                        msa_jobs.append(
                            queue_alphafold3_msa_job(
                                protein_sequences=list(unique_sequences),
                                target_artifact=next(iter(unique_sequences.values()))[1],
                                image=DEFAULT_ALPHAFOLD3_IMAGE,
                                db_dir=af3_db_dir,
                                weights_dir=af3_weights_dir,
                                msa_repository_dir=af3_msa_dir,
                                batch_size=max(af3_batch, len(unique_sequences)),
                                gpu_device=gpu_device,
                                launch_campaign_id=launch_campaign_id,
                                launch_campaign_label=launch_campaign_label,
                                campaign_purpose=campaign_purpose,
                            )
                        )
                    except Exception as exc:
                        msa_queue_failures.append(str(exc))
                msa_dependency_ids = tuple(job.run_id for job in msa_jobs)
                if msa_jobs:
                    st.success(
                        f"Queued {len(msa_jobs)} unique MSA preparation job(s) "
                        + (
                            "on any available GPU."
                            if gpu_device == "all"
                            else f"on GPU {gpu_device}."
                        )
                    )
                if msa_queue_failures:
                    st.error(
                        "Could not establish the MSA barrier: "
                        + "; ".join(msa_queue_failures)
                    )
            queue_config = {
                "gpu_device": gpu_device,
                "docking_gpu_device": gpu_device,
                "structure_gpu_device": gpu_device,
                "docking_mode": docking_mode,
                "search_mode": search_mode,
                "exhaustiveness": exhaustiveness,
                "poses": poses,
                "use_scrub": use_scrub,
                "scrub_ph": scrub_ph,
                "scrub_skip_tautomer": scrub_skip_tautomer,
                "campaign_replicates": campaign_replicates,
                "campaign_seed": campaign_seed,
                "maximum_compounds": maximum_compounds,
                "extra_args_text": extra_args_text,
                "vina_compound_timeout_minutes": vina_compound_timeout,
                "openvs_protocol": openvs_protocol,
                "openvs_reference_mode": openvs_reference_mode,
                "openvs_workers": openvs_workers,
                "openvs_ph": openvs_ph,
                "openvs_conformers": openvs_conformers,
                "openvs_steps": openvs_steps,
                "openvs_padding": openvs_padding,
                "openvs_cluster": openvs_cluster,
                "boltz_max": boltz_max,
                "boltz_recycles": boltz_recycles,
                "boltz_steps": boltz_steps,
                "boltz_samples": boltz_samples,
                "af3_db_dir": af3_db_dir,
                "af3_weights_dir": af3_weights_dir,
                "af3_msa_dir": af3_msa_dir,
                "af3_max": af3_max,
                "af3_batch": af3_batch,
                "af3_recycles": af3_recycles,
                "nesso_checkpoint": nesso_checkpoint,
                "nesso_ccd": nesso_ccd,
                "nesso_esm_cache": nesso_esm_cache,
                "nesso_max": nesso_max,
                "nesso_recycles": nesso_recycles,
                "nesso_refine": nesso_refine,
                "nesso_tokens": nesso_tokens,
                "nesso_affinity": nesso_affinity,
            }
            queued_rows: list[str] = []
            failed_rows: list[str] = []
            orientation_rows: list[str] = []
            for context in ([] if msa_queue_failures else target_contexts):
                target_choice = context["target"]
                target_name = _target_label(target_choice)
                queued, failures, orientation_note = (
                    _queue_target_engine_jobs(
                        context=context,
                        engines=selected_engines_by_target.get(
                            target_choice.job.run_id, []
                        ),
                        selected_compounds=selected_compounds,
                        compound_paths=compound_paths,
                        reference=reference,
                        target_ligand_only=target_ligand_only,
                        launch_campaign_id=launch_campaign_id,
                        launch_campaign_label=launch_campaign_label,
                        campaign_purpose=campaign_purpose,
                        config=queue_config,
                        msa_dependency_ids=msa_dependency_ids,
                    )
                )
                queued_rows.extend(
                    f"{target_name}: {value}" for value in queued
                )
                failed_rows.extend(
                    f"{target_name}: {value}" for value in failures
                )
                if orientation_note:
                    orientation_rows.append(
                        f"{target_name}: {orientation_note}"
                    )
            if queued_rows:
                st.success("Queued: " + "; ".join(queued_rows))
            if orientation_rows:
                st.caption("Prepared: " + "; ".join(orientation_rows))
            if failed_rows:
                st.error("Could not queue: " + "; ".join(failed_rows))

        st.markdown("#### Active runs")
        if st.button("Refresh", key="binding_running_refresh"):
            _cached_job_rows.clear()
            st.rerun()
        _render_jobs({"queued", "preparing", "running"})

    with results_tab:
        st.caption("All ordinary docking and cofolding jobs are shown, independent of current inputs.")
        if st.button("Refresh", key="binding_results_refresh"):
            _cached_job_rows.clear()
            st.rerun()
        _render_jobs({"completed", "failed", "cancelled"})


if __name__ == "__main__":
    render()
