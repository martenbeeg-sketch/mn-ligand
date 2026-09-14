from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import math
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.pharmacophore_viewer import (
    hypothesis_structure_path,
    render_pharmacophore_viewer,
)
from mn_ligand.app.pages.discover_inputs import (
    ArtifactChoice,
    artifact_options,
    select_artifact,
    select_target_artifact,
)
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import display_job_code, iter_job_records
from mn_ligand.runtime import reference_root, runs_root
from mn_ligand.workflows.generative_design import (
    GENERATION_TASK_GROUP,
    GENERATOR_BY_ID,
    GENERATOR_SPECS,
    create_generation_campaign_job,
    pharmacophore_required_contacts,
    queue_generation_job,
)
from mn_ligand.workflows.pharmacophore import load_pharmacophore


def _conditioning_label(condition: str) -> str:
    return {
        "pocket": "Pocket",
        "reference_ligand": "Reference ligand",
        "pharmacophore": "Pharmacophore hypothesis",
        "protein_pharmacophore": "Protein + pharmacophore",
        "interaction_profile": "Detected protein–ligand interactions",
        "scaffold": "Fixed scaffold",
        "fragment": "Fragment growing/linking",
        "property_objectives": "Property objectives",
    }.get(condition, condition.replace("_", " ").title())


_SEED_SECONDS_PER_ATTEMPT = {
    "omtra": 45.0,
    "flowr_root": 8.0,
    "pocketxmol": 20.0,
    "conditar": 100.0,
    "drugrpg": 15.0,
    "pfm": 5.0,
    "pocketflow": 30.0,
    "pgmg": 25.0,
}
_PAOPT_SECONDS_PER_DIFFUSION_PASS = 75.0

_ENGINE_PARAMETER_WIDGET_PREFIXES = (
    "generation_engine_mode_",
    "generation_omtra_",
    "generation_pocketflow_",
    "generation_pocketxmol_",
    "generation_flowr_",
    "generation_conditar_",
    "generation_paopt_",
    "generation_drugrpg_",
    "generation_count_",
    "generation_batch_",
    "generation_seed_",
    "generation_runtime_",
)
_ENGINE_PARAMETER_STATE_KEY = "_generation_engine_parameter_values"

_ENGINE_PARAMETER_GUIDANCE = {
    "omtra": (
        "**Integration steps** trade runtime for numerical integration accuracy. "
        "**Stochastic sampling** increases exploration and run-to-run diversity; "
        "noise scale controls its magnitude and ε controls the stochastic solver "
        "endpoint. A custom **ligand atom prior** directs size; 0 preserves the "
        "checkpoint's learned size distribution."
    ),
    "pocketflow": (
        "**Atom/bond temperatures** below 1 favor high-probability choices; above "
        "1 explore more but can reduce validity. **Maximum atoms** caps growth. "
        "Deterministic focus selection is reproducible; sampled focus is more "
        "diverse. The **focus threshold** controls when growth may continue, while "
        "**minimum protein distance** rejects placements that are too close to "
        "protein atoms; overly large values can prevent pocket filling."
    ),
    "pocketxmol": (
        "**Diffusion steps** trade runtime for denoising/refinement. The atom-count "
        "mean and SD set the molecular-size prior. In optimization mode, smaller "
        "**redesign strength** keeps the result closer to the reference ligand; "
        "larger values permit broader chemical and geometric changes."
    ),
    "flowr_root": (
        "**Integration steps** and **corrector iterations** increase sampling work "
        "and may improve integration/refinement. Midpoint is more accurate per "
        "step but costs more than Euler. SDE sampling adds exploration. Variable "
        "sizes use the learned size distribution. The diversity filter removes "
        "near-duplicates above the selected similarity ceiling, so a lower ceiling "
        "enforces more diversity and may return fewer molecules."
    ),
    "conditar": (
        "**Diffusion steps** trade runtime for denoising quality; 1000 is the "
        "native configuration. **Pocket radius** controls how much protein context "
        "around the pocket/reference frame is encoded. Too small can omit relevant "
        "residues; too large can dilute local conditioning and increase memory use."
    ),
    "paopt": (
        "paOPT uses the conDitar diffusion controls, then estimates ADMET gradients. "
        "More **steering steps** repeat optimization; more **gradient pairs** reduce "
        "finite-difference noise but add two complete diffusion passes per pair and "
        "step. Perturbation size sets how locally gradients are estimated. Multiple "
        "objectives are balanced by the native MGDA solver rather than fixed weights."
    ),
    "drugrpg": (
        "**Generated ligand atoms** fixes the atom count used for every attempt. "
        "Larger molecules cost more and may fill a large pocket, but can increase "
        "clashes, synthetic complexity, and reconstruction failures."
    ),
    "pfm": (
        "PFM's validated custom-pocket adapter intentionally retains its learned "
        "atom-count prior and fixed 20-step ODE path. Attempts and seed change the "
        "sample population without silently altering the published sampler."
    ),
    "pgmg": (
        "PGMG is directed by the selected pharmacophore graph, not a temperature "
        "or guidance scale. Feature types, positions, tolerances, required/optional "
        "status, and the eight-point limit are therefore its meaningful design controls."
    ),
}


def _restore_engine_parameter_state() -> None:
    saved = st.session_state.get(_ENGINE_PARAMETER_STATE_KEY)
    if not isinstance(saved, dict):
        return
    for key, value in saved.items():
        if (
            isinstance(key, str)
            and key.startswith(_ENGINE_PARAMETER_WIDGET_PREFIXES)
            and key not in st.session_state
        ):
            st.session_state[key] = value


def _snapshot_engine_parameter_state() -> None:
    saved = st.session_state.get(_ENGINE_PARAMETER_STATE_KEY)
    snapshot = dict(saved) if isinstance(saved, dict) else {}
    for key, value in st.session_state.items():
        if (
            isinstance(key, str)
            and key.startswith(_ENGINE_PARAMETER_WIDGET_PREFIXES)
        ):
            snapshot[key] = value
    st.session_state[_ENGINE_PARAMETER_STATE_KEY] = snapshot


def _bound_ligand_choice(
    target: ArtifactChoice | None,
) -> ArtifactChoice | None:
    if target is None or target.job.artifact_manifest is None:
        return None
    for artifact in target.job.artifact_manifest.by_type(
        "prepared_ligand_set"
    ):
        if artifact.resolve(target.job.run_dir, must_exist=True) is not None:
            return ArtifactChoice(job=target.job, artifact=artifact)
    return None


def _bound_ligand_identity(target: ArtifactChoice | None) -> str:
    if target is None:
        return "Bound ligand"
    ligand_key = str(target.job.metadata.get("ligand_key") or "")
    requested_id = ligand_key.partition("|")[0].strip()
    ligands = target.job.metadata.get("ligands")
    candidates = (
        [dict(item) for item in ligands if isinstance(item, dict)]
        if isinstance(ligands, list)
        else []
    )
    selected = next(
        (
            item
            for item in candidates
            if requested_id
            and requested_id
            in {
                str(item.get("ccd_id") or ""),
                str(item.get("resname") or ""),
            }
        ),
        candidates[0] if candidates else {},
    )
    ligand_id = str(
        selected.get("ccd_id")
        or selected.get("resname")
        or requested_id
        or "Bound ligand"
    )
    ligand_name = str(selected.get("name") or "").strip()
    return f"{ligand_id} — {ligand_name}" if ligand_name else ligand_id


def _bound_ligand_pocket_choice(
    target: ArtifactChoice | None,
) -> ArtifactChoice | None:
    if target is None:
        return None
    options = artifact_options(
        ("pocket",),
        source_run_id=target.job.run_id,
    )
    candidates = [
        choice
        for choice in options.values()
        if str(
            choice.job.metadata.get("source")
            or choice.job.metadata.get("tool")
            or ""
        ).lower()
        == "bound_ligand"
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda choice: str(choice.job.created_at or ""),
    )


def _paopt_pass_multiplier(settings: dict[str, object]) -> int:
    steps = max(1, int(settings.get("optimization_steps") or 1))
    pairs = max(1, int(settings.get("gradient_estimate_pairs") or 4))
    return 1 + steps * (2 * pairs + 1)


def _engine_pass_multiplier(
    engine_id: str, settings: dict[str, object]
) -> float:
    multiplier = 1.0
    if engine_id == "paopt":
        multiplier = float(_paopt_pass_multiplier(settings))
    step_key_and_default = {
        "omtra": ("integration_steps", 250),
        "pocketxmol": ("diffusion_steps", 100),
        "flowr_root": ("integration_steps", 100),
        "conditar": ("diffusion_steps", 1000),
        "paopt": ("diffusion_steps", 1000),
    }.get(engine_id)
    if step_key_and_default:
        key, default = step_key_and_default
        multiplier *= max(
            0.1, float(settings.get(key) or default) / float(default)
        )
    if engine_id == "flowr_root":
        if str(settings.get("solver") or "euler") == "midpoint":
            multiplier *= 2.0
        multiplier *= 1.0 + max(
            0, int(settings.get("corrector_steps") or 0)
        )
    if engine_id == "drugrpg":
        multiplier *= max(
            0.25, float(settings.get("max_atoms") or 30) / 30.0
        )
    return multiplier


def _successful_generation_rates() -> dict[str, list[float]]:
    rates: dict[str, list[float]] = defaultdict(list)
    for job in iter_job_records(
        runs_root(), task_groups=(GENERATION_TASK_GROUP,)
    ):
        if str(job.metadata.get("status") or "") != "completed":
            continue
        try:
            started = datetime.fromisoformat(str(job.metadata["started_at"]))
            completed = datetime.fromisoformat(
                str(job.metadata["completed_at"])
            )
            elapsed_seconds = (completed - started).total_seconds()
            requested = max(
                1, int(job.metadata.get("requested_count") or 1)
            )
            input_payload = json.loads(
                (job.run_dir / "input.json").read_text()
            )
            settings = dict(input_payload.get("engine_settings") or {})
            multiplier = _engine_pass_multiplier(
                str(job.metadata.get("engine_id") or ""), settings
            )
            rate = elapsed_seconds / (requested * multiplier)
        except (KeyError, OSError, TypeError, ValueError):
            continue
        if rate > 0:
            rates[str(job.metadata.get("engine_id") or "")].append(rate)
    return rates


def _successful_generation_yields() -> dict[str, list[float]]:
    yields: dict[str, list[float]] = defaultdict(list)
    for job in iter_job_records(
        runs_root(), task_groups=(GENERATION_TASK_GROUP,)
    ):
        if str(job.metadata.get("status") or "") != "completed":
            continue
        try:
            requested = max(
                1, int(job.metadata.get("requested_count") or 1)
            )
            unique = max(
                0, int(job.result.get("valid_compound_count") or 0)
            )
        except (TypeError, ValueError):
            continue
        yields[str(job.metadata.get("engine_id") or "")].append(
            min(1.0, unique / requested)
        )
    return yields


def _conservative_generation_yield(
    engine_id: str,
    yields: dict[str, list[float]],
) -> tuple[float, str]:
    observed = sorted(yields.get(engine_id) or ())
    if observed:
        index = max(0, math.floor(0.25 * (len(observed) - 1)))
        return (
            observed[index],
            f"Historical P25 from {len(observed)} run(s)",
        )
    return 0.5, "Seed yield assumption; no successful typed run"


def _conservative_generation_rate(
    engine_id: str,
    rates: dict[str, list[float]],
) -> tuple[float, str, str]:
    observed = sorted(rates.get(engine_id) or ())
    if observed:
        index = max(0, math.ceil(0.75 * len(observed)) - 1)
        rate = observed[index]
        evidence = (
            f"{len(observed)} successful typed run(s); observed "
            f"{min(observed):.1f}–{max(observed):.1f} s/work unit"
        )
        return rate, "Historical P75", evidence
    if engine_id == "paopt":
        return (
            _PAOPT_SECONDS_PER_DIFFUSION_PASS,
            "Seed estimate",
            "No successful typed run yet; seconds per diffusion pass",
        )
    return (
        _SEED_SECONDS_PER_ATTEMPT.get(engine_id, 60.0),
        "Seed estimate",
        "No successful typed run yet; seconds per requested attempt",
    )


def _format_duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} sec"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} hr"


def render() -> None:
    st.title("De Novo Molecule Design")
    st.caption(
        "Build one immutable design campaign, then fan the same target and "
        "conditioning intent out to compatible generation engines."
    )
    target_tab, conditioning_tab, engine_tab, run_tab, results_tab = st.tabs(
        ["Target", "Conditioning", "Engines", "Run", "Results"]
    )
    required_contacts: list[dict[str, object]] = []

    with target_tab:
        target = select_target_artifact(
            "Prepared target",
            ("prepared_target", "prepared_receptor", "prepared_complex"),
            key="generation_target",
        )
        st.caption(
            "The selected target remains immutable. Pocket cropping and every "
            "engine-specific coordinate representation are staged in child jobs."
        )

    with conditioning_tab:
        st.markdown("#### Shared scientific intent")
        st.caption(
            "Select any meaningful combination. Engines that cannot represent a "
            "condition explicitly will show that limitation before launch."
        )
        default_ligand = _bound_ligand_choice(target)
        ligand_identity = _bound_ligand_identity(target)
        if default_ligand is not None and target is not None:
            reference_ligand = default_ligand
            st.selectbox(
                "Reference ligand from selected complex",
                [
                    (
                        f"{ligand_identity} — "
                        f"{display_job_code(target.job.metadata.get('job_code'), target.job.run_id)}"
                    )
                ],
                disabled=True,
                key=(
                    "generation_bound_reference_"
                    f"{target.job.run_id}"
                ),
                help=(
                    "The prepared ligand belonging to this exact immutable "
                    "protein–ligand complex is used automatically. Unrelated "
                    "docking poses and compound sets are not suggested."
                ),
            )
        else:
            reference_ligand = select_artifact(
                "Reference ligand, pose, scaffold, or fragment (optional)",
                (
                    "prepared_ligand_set",
                    "docked_pose",
                    "validated_pose",
                    "pose_set",
                    "compound_set",
                ),
                key="generation_reference_ligand",
                required=False,
            )

        automatic_pocket = _bound_ligand_pocket_choice(target)
        if automatic_pocket is not None and target is not None:
            pocket = automatic_pocket
            st.selectbox(
                "Pocket derived from the bound reference ligand",
                [
                    (
                        f"{ligand_identity} pocket — "
                        f"{display_job_code(automatic_pocket.job.metadata.get('job_code'), automatic_pocket.job.run_id)}"
                    )
                ],
                disabled=True,
                key=(
                    "generation_bound_pocket_"
                    f"{target.job.run_id}"
                ),
                help=(
                    "The latest completed bound-ligand pocket linked to the "
                    "selected complex is supplied automatically."
                ),
            )
            st.caption(
                "This coordinate pocket is automatically available to every "
                "selected engine that requires pocket conditioning."
            )
            try:
                pocket_input = json.loads(
                    (automatic_pocket.job.run_dir / "input.json").read_text()
                )
            except (OSError, TypeError, ValueError):
                pocket_input = {}
            pocket_parameters = pocket_input.get("parameters")
            pocket_parameters = (
                dict(pocket_parameters)
                if isinstance(pocket_parameters, dict)
                else {}
            )
            artifact_metadata = dict(automatic_pocket.artifact.metadata)
            descriptors = artifact_metadata.get("descriptors")
            descriptors = (
                dict(descriptors)
                if isinstance(descriptors, dict)
                else {}
            )
            with st.expander(
                "How this automatic pocket was produced",
                expanded=False,
            ):
                st.write(
                    "This is an immutable **bound-ligand pocket-detection "
                    "result**, not an on-the-fly or predicted pocket. The "
                    "workflow located the selected ligand in the exact prepared "
                    "complex, calculated a box around its heavy atoms, and "
                    "retained protein residues lining that ligand."
                )
                provenance_columns = st.columns(4)
                provenance_columns[0].metric(
                    "Method",
                    str(
                        automatic_pocket.job.metadata.get("source")
                        or automatic_pocket.job.metadata.get("tool")
                        or "bound_ligand"
                    ).replace("_", " "),
                )
                provenance_columns[1].metric(
                    "Box padding",
                    (
                        f"{float(pocket_parameters.get('box_padding_angstrom')):.1f} Å"
                        if pocket_parameters.get("box_padding_angstrom")
                        is not None
                        else "—"
                    ),
                )
                provenance_columns[2].metric(
                    "Lining cutoff",
                    (
                        f"{float(pocket_parameters.get('lining_cutoff_angstrom')):.1f} Å"
                        if pocket_parameters.get(
                            "lining_cutoff_angstrom"
                        )
                        is not None
                        else "—"
                    ),
                )
                provenance_columns[3].metric(
                    "Lining residues",
                    int(descriptors.get("lining_residue_count") or 0),
                )
                center = artifact_metadata.get("center_angstrom")
                size = artifact_metadata.get("size_angstrom")
                if (
                    isinstance(center, (list, tuple))
                    and len(center) == 3
                    and isinstance(size, (list, tuple))
                    and len(size) == 3
                ):
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Geometry": "Center (Å)",
                                    "X": round(float(center[0]), 3),
                                    "Y": round(float(center[1]), 3),
                                    "Z": round(float(center[2]), 3),
                                },
                                {
                                    "Geometry": "Box size (Å)",
                                    "X": round(float(size[0]), 3),
                                    "Y": round(float(size[1]), 3),
                                    "Z": round(float(size[2]), 3),
                                },
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                st.caption(
                    "The box is the axis-aligned extent of the bound ligand's "
                    "heavy atoms plus padding on both sides. A lining residue "
                    "is included when any of its atoms lies within the cutoff "
                    "of any ligand heavy atom."
                )
                pocket_links = st.columns(2)
                pocket_code = display_job_code(
                    automatic_pocket.job.metadata.get("job_code"),
                    automatic_pocket.job.run_id,
                )
                pocket_links[0].link_button(
                    f"Open pocket result {pocket_code}",
                    "./job-results?"
                    + urlencode(
                        {
                            "task_group": "pocket-detection",
                            "run_id": automatic_pocket.job.run_id,
                            "label": pocket_code,
                        }
                    ),
                    width="stretch",
                )
                pocket_links[1].link_button(
                    "Create or replace target pocket",
                    "./discover-pocket-detection?"
                    + urlencode(
                        {
                            "prepared_target_run_id": target.job.run_id,
                        }
                    ),
                    width="stretch",
                )
        else:
            pocket = select_artifact(
                "Pocket for pocket-conditioned engines",
                ("pocket",),
                key=(
                    "generation_pocket_"
                    f"{target.job.run_id if target is not None else 'none'}"
                ),
                required=False,
                source_run_id=(
                    target.job.run_id if target is not None else ""
                ),
            )
            if default_ligand is not None:
                st.warning(
                    "This complex has a bound reference ligand but no completed "
                    "bound-ligand pocket. Create that pocket before launching "
                    "engines that require coordinate-pocket conditioning."
                )
        hypothesis = select_artifact(
            "Saved pharmacophore hypothesis (optional)",
            ("pharmacophore_hypothesis",),
            key="generation_pharmacophore",
            required=False,
        )
        if hypothesis is None:
            st.link_button(
                "Create or edit a pharmacophore hypothesis",
                "./generate-pharmacophore-hypotheses",
            )
        else:
            try:
                required_contacts = pharmacophore_required_contacts(
                    hypothesis.artifact
                )
            except Exception as exc:
                st.warning(
                    f"Could not read target-contact constraints from the "
                    f"selected hypothesis: {exc}"
                )
            if required_contacts:
                contact_rows = []
                for contact in required_contacts:
                    atom = contact.get("protein_atom") or {}
                    contact_rows.append(
                        {
                            "Review status": "Review after pose prediction",
                            "Target atom": (
                                f"{atom.get('chain', '')}:"
                                f"{atom.get('residue_name', '')}"
                                f"{atom.get('residue_number', '')}:"
                                f"{atom.get('atom_name', '')}"
                            ),
                            "Ligand role": contact.get("ligand_role", ""),
                            "Distance (Å)": contact.get(
                                "target_distance_angstrom", ""
                            ),
                            "Reference observed": bool(
                                contact.get(
                                    "reference_sidechain_contact_observed"
                                )
                            ),
                        }
                    )
                st.markdown("##### Post-pose interaction review targets")
                st.dataframe(
                    pd.DataFrame(contact_rows),
                    hide_index=True,
                    width="stretch",
                )
                st.warning(
                    "This geometry can guide compatible generators, but it is "
                    "not guaranteed and does not automatically reject a design. "
                    "Review the contact after docking, cofolding, or refolding "
                    "with PLIP/PandaMap and scientific inspection."
                )
            try:
                hypothesis_path = hypothesis.artifact.resolve(
                    hypothesis.job.run_dir,
                    must_exist=True,
                )
                if hypothesis_path is None:
                    raise FileNotFoundError(
                        "The canonical pharmacophore artifact is unavailable."
                    )
                hypothesis_features = load_pharmacophore(hypothesis_path)
                enabled_features = [
                    feature
                    for feature in hypothesis_features
                    if feature.enabled
                ]
                required_features = [
                    feature
                    for feature in enabled_features
                    if feature.required
                ]
                observed_features = [
                    feature
                    for feature in enabled_features
                    if bool(feature.metadata.get("observed"))
                ]
                st.markdown("##### Selected hypothesis viewer")
                name_column, feature_column, required_column, method_column = (
                    st.columns(4)
                )
                name_column.metric(
                    "Saved name",
                    str(
                        hypothesis.job.metadata.get("name")
                        or "Pharmacophore hypothesis"
                    ),
                )
                feature_column.metric(
                    "Enabled features",
                    len(enabled_features),
                )
                required_column.metric(
                    "Post-pose review features",
                    len(required_features),
                )
                method_column.metric(
                    "Creation method",
                    str(
                        hypothesis.job.metadata.get("creation_method")
                        or "unspecified"
                    ).replace("-", " "),
                )

                control_columns = st.columns((2, 1, 1))
                focus_options = [
                    index
                    for index, feature in enumerate(hypothesis_features)
                    if feature.enabled
                ]
                focus_index = control_columns[0].selectbox(
                    "Focus feature — viewer only",
                    focus_options,
                    format_func=lambda index: (
                        f"{index + 1}. "
                        f"{hypothesis_features[index].feature_type}"
                        + (
                            " · post-pose contact review"
                            if hypothesis_features[index].metadata.get(
                                "required_target_contact"
                            )
                            else (
                                " · marked for post-pose review"
                                if hypothesis_features[index].required
                                else " · conditioning context"
                            )
                        )
                    ),
                    key=(
                        "generation_pharmacophore_focus_"
                        f"{hypothesis.job.run_id}"
                    ),
                )
                show_observed = control_columns[1].checkbox(
                    "Show observed features",
                    value=True,
                    key=(
                        "generation_pharmacophore_observed_"
                        f"{hypothesis.job.run_id}"
                    ),
                    help=(
                        "Observed features were derived from the reference "
                        "complex and remain part of the saved hypothesis."
                    ),
                )
                show_labels = control_columns[2].checkbox(
                    "Show all labels",
                    value=False,
                    key=(
                        "generation_pharmacophore_labels_"
                        f"{hypothesis.job.run_id}"
                    ),
                )
                if focus_index is not None:
                    render_pharmacophore_viewer(
                        hypothesis_structure_path(hypothesis.job),
                        hypothesis_features,
                        focus_index=int(focus_index),
                        show_observed=show_observed,
                        show_labels=show_labels,
                        viewer_key=(
                            "generation-pharmacophore:"
                            f"{hypothesis.job.run_id}"
                        ),
                    )

                with st.expander("Feature inventory"):
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Feature": index + 1,
                                    "Type": feature.feature_type,
                                    "Enabled": feature.enabled,
                                    "Post-pose review": feature.required,
                                    "Observed": bool(
                                        feature.metadata.get("observed")
                                    ),
                                    "Source": feature.source,
                                    "Residues": ", ".join(
                                        feature.source_residues
                                    ),
                                    "X": round(feature.x, 3),
                                    "Y": round(feature.y, 3),
                                    "Z": round(feature.z, 3),
                                    "Tolerance (Å)": round(
                                        feature.radius, 3
                                    ),
                                }
                                for index, feature in enumerate(
                                    hypothesis_features
                                )
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                st.link_button(
                    "Open hypothesis editor",
                    "./generate-pharmacophore-hypotheses",
                )
                if observed_features:
                    st.caption(
                        f"{len(observed_features)} enabled feature(s) were "
                        "observed in the reference complex."
                    )
            except Exception as exc:
                st.warning(
                    "Could not display the selected pharmacophore hypothesis: "
                    f"{exc}"
                )
        design_mode = st.radio(
            "Reference-ligand intent",
            (
                "Use as spatial reference only",
                "Preserve as scaffold",
                "Grow or link fragments",
                "Optimize while retaining similarity",
            ),
            horizontal=True,
            key="generation_reference_intent",
            disabled=reference_ligand is None,
        )
        st.markdown("##### Optional campaign objectives")
        objective_columns = st.columns(4)
        optimize_affinity = objective_columns[0].checkbox(
            "Affinity/interaction fit", value=True, key="generation_objective_affinity"
        )
        optimize_admet = objective_columns[1].checkbox(
            "ADMET", value=False, key="generation_objective_admet"
        )
        optimize_novelty = objective_columns[2].checkbox(
            "Novelty", value=True, key="generation_objective_novelty"
        )
        optimize_diversity = objective_columns[3].checkbox(
            "Diversity", value=True, key="generation_objective_diversity"
        )
        st.info(
            "Objectives are recorded separately from model conditioning. An engine "
            "may optimize an objective during generation or the campaign may apply "
            "it later as a transparent ranking/filtering stage."
        )

    selected_engine_ids: list[str] = []
    engine_modes: dict[str, str] = {}
    engine_settings: dict[str, dict[str, object]] = {}
    _restore_engine_parameter_state()
    with engine_tab:
        st.markdown("#### Select generation engines")
        selection_actions = st.columns((1, 1, 5))
        if selection_actions[0].button(
            "Select all",
            key="generation_engines_select_all",
            width="stretch",
        ):
            for spec in GENERATOR_SPECS:
                st.session_state[
                    f"generation_engine_{spec.engine_id}"
                ] = bool(spec.adapter_ready)
        if selection_actions[1].button(
            "Deselect all",
            key="generation_engines_deselect_all",
            width="stretch",
        ):
            for spec in GENERATOR_SPECS:
                st.session_state[
                    f"generation_engine_{spec.engine_id}"
                ] = False

        selector_columns = st.columns(3)
        for index, spec in enumerate(GENERATOR_SPECS):
            enabled = selector_columns[index % len(selector_columns)].checkbox(
                spec.name,
                value=False,
                key=f"generation_engine_{spec.engine_id}",
                disabled=not spec.adapter_ready,
                help=spec.summary,
            )
            if enabled:
                selected_engine_ids.append(spec.engine_id)

        if selected_engine_ids:
            st.caption(
                f"{len(selected_engine_ids)} engine(s) selected. Parameter "
                "panels for selected engines are expanded below. The "
                "checkboxes—not whether a panel is open—define campaign "
                "membership."
            )
        else:
            st.info(
                "Select one or more engines to include them in the campaign. "
                "Their collapsed panels can still be opened to inspect or "
                "prepare parameters."
            )

        st.markdown("#### Engine parameters")
        grouped: dict[str, list[object]] = defaultdict(list)
        for spec in GENERATOR_SPECS:
            grouped[spec.theme].append(spec)
        for theme, specs in grouped.items():
            st.markdown(f"##### {theme}")
            for spec in specs:
                selected = spec.engine_id in selected_engine_ids
                with st.expander(
                    f"{spec.name} — configure",
                    expanded=selected,
                ):
                    enabled = True
                    st.caption(spec.summary)
                    if spec.engine_id == "conditar":
                        st.warning(
                            "Local permission required. This group deployment is "
                            "authorized; do not redistribute its source, image, or weights."
                        )
                    if spec.adapter_ready:
                        st.caption(
                            "Adapter status: native container inference, normalized "
                            "artifacts, typed results, and compound-set handoff validated; "
                            "campaign-level scientific evaluation remains."
                        )
                    else:
                        st.caption(
                            "Adapter status: native configuration adapter still under review"
                        )
                    st.write(
                        "Supported conditioning: "
                        + ", ".join(_conditioning_label(item) for item in spec.conditions)
                    )
                    if enabled:
                        available_modes = []
                        if hypothesis is not None and (
                            spec.supports("pharmacophore")
                            or spec.supports("protein_pharmacophore")
                        ):
                            available_modes.append("Use shared editable hypothesis")
                        if reference_ligand is not None and spec.engine_id == "omtra":
                            available_modes.append("Re-detect pharmacophore from ligand with OMTRA")
                        if (
                            reference_ligand is not None
                            and spec.supports("interaction_profile")
                        ):
                            available_modes.append("Detect interactions from the reference complex")
                        if pocket is not None and spec.supports("pocket"):
                            available_modes.append("Use pocket geometry only")
                        if reference_ligand is not None and spec.supports("reference_ligand"):
                            available_modes.append(design_mode)
                        if not available_modes:
                            available_modes.append("No compatible conditioning is selected")
                        engine_modes[spec.engine_id] = st.selectbox(
                            f"{spec.name} conditioning path",
                            list(dict.fromkeys(available_modes)),
                            key=f"generation_engine_mode_{spec.engine_id}",
                        )
                        if spec.engine_id == "omtra":
                            st.markdown("##### OMTRA sampling controls")
                            integration_steps = st.number_input(
                                "Integration steps",
                                min_value=25,
                                max_value=2000,
                                value=250,
                                step=25,
                                key="generation_omtra_integration_steps",
                                help=(
                                    "More steps generally improve integration "
                                    "accuracy but increase runtime approximately "
                                    "linearly. Native default: 250."
                                ),
                            )
                            stochastic = st.checkbox(
                                "Stochastic sampling",
                                value=False,
                                key="generation_omtra_stochastic",
                                help=(
                                    "Adds sampling noise for exploration. Keep "
                                    "disabled for the most reproducible campaign."
                                ),
                            )
                            omtra_columns = st.columns(4)
                            noise_scale = omtra_columns[0].number_input(
                                "Noise scale",
                                min_value=0.0,
                                max_value=5.0,
                                value=1.0,
                                step=0.1,
                                disabled=not stochastic,
                                key="generation_omtra_noise_scale",
                            )
                            epsilon = omtra_columns[1].number_input(
                                "Stochastic ε",
                                min_value=0.001,
                                max_value=0.5,
                                value=0.01,
                                step=0.005,
                                format="%.3f",
                                disabled=not stochastic,
                                key="generation_omtra_epsilon",
                            )
                            atom_mean = omtra_columns[2].number_input(
                                "Mean ligand atoms",
                                min_value=0.0,
                                max_value=100.0,
                                value=0.0,
                                step=1.0,
                                key="generation_omtra_atom_mean",
                                help="0 uses OMTRA's learned size distribution.",
                            )
                            atom_std = omtra_columns[3].number_input(
                                "Atom-count SD",
                                min_value=0.1,
                                max_value=30.0,
                                value=2.0,
                                step=0.5,
                                disabled=float(atom_mean) == 0.0,
                                key="generation_omtra_atom_std",
                            )
                            engine_settings["omtra"] = {
                                "integration_steps": int(integration_steps),
                                "stochastic_sampling": bool(stochastic),
                                "noise_scale": float(noise_scale),
                                "epsilon": float(epsilon),
                                "ligand_atoms_mean": float(atom_mean),
                                "ligand_atoms_std": float(atom_std),
                            }
                        elif spec.engine_id == "pocketflow":
                            st.markdown("##### PocketFlow sampling controls")
                            pf_columns = st.columns(3)
                            atom_temperature = pf_columns[0].number_input(
                                "Atom temperature",
                                min_value=0.1,
                                max_value=3.0,
                                value=1.0,
                                step=0.1,
                                key="generation_pocketflow_atom_temp",
                                help="Lower is more conservative; higher explores more atom choices.",
                            )
                            bond_temperature = pf_columns[1].number_input(
                                "Bond temperature",
                                min_value=0.1,
                                max_value=3.0,
                                value=1.0,
                                step=0.1,
                                key="generation_pocketflow_bond_temp",
                            )
                            max_atoms = pf_columns[2].number_input(
                                "Maximum atoms",
                                min_value=5,
                                max_value=100,
                                value=40,
                                step=1,
                                key="generation_pocketflow_max_atoms",
                            )
                            pf_columns = st.columns(3)
                            focus_strategy = pf_columns[0].selectbox(
                                "Focus-atom choice",
                                ("maximum", "sample"),
                                format_func=lambda value: (
                                    "Highest probability"
                                    if value == "maximum"
                                    else "Sample distribution"
                                ),
                                key="generation_pocketflow_focus",
                            )
                            focus_threshold = pf_columns[1].number_input(
                                "Focus threshold",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.5,
                                step=0.05,
                                key="generation_pocketflow_focus_threshold",
                            )
                            min_distance = pf_columns[2].number_input(
                                "Min protein distance (Å)",
                                min_value=1.0,
                                max_value=6.0,
                                value=3.0,
                                step=0.1,
                                key="generation_pocketflow_min_distance",
                            )
                            engine_settings["pocketflow"] = {
                                "atom_temperature": float(atom_temperature),
                                "bond_temperature": float(bond_temperature),
                                "max_atoms": int(max_atoms),
                                "focus_strategy": focus_strategy,
                                "focus_threshold": float(focus_threshold),
                                "min_protein_distance": float(min_distance),
                            }
                        elif spec.engine_id == "pocketxmol":
                            st.markdown("##### PocketXMol sampling controls")
                            optimizing = "optimize" in engine_modes[
                                spec.engine_id
                            ].lower()
                            px_columns = st.columns(4)
                            px_steps = px_columns[0].number_input(
                                "Diffusion steps",
                                min_value=10,
                                max_value=500,
                                value=50 if optimizing else 100,
                                step=10,
                                key="generation_pocketxmol_steps",
                            )
                            px_mean = px_columns[1].number_input(
                                "Mean ligand atoms",
                                min_value=5.0,
                                max_value=100.0,
                                value=38.0 if optimizing else 28.0,
                                step=1.0,
                                key="generation_pocketxmol_atom_mean",
                            )
                            px_std = px_columns[2].number_input(
                                "Atom-count SD",
                                min_value=0.1,
                                max_value=30.0,
                                value=3.0 if optimizing else 2.0,
                                step=0.5,
                                key="generation_pocketxmol_atom_std",
                            )
                            px_strength = px_columns[3].number_input(
                                "Redesign strength",
                                min_value=0.05,
                                max_value=1.0,
                                value=0.5,
                                step=0.05,
                                disabled=not optimizing,
                                key="generation_pocketxmol_strength",
                                help=(
                                    "Optimization mode only. Smaller values "
                                    "retain more of the input molecule."
                                ),
                            )
                            engine_settings["pocketxmol"] = {
                                "diffusion_steps": int(px_steps),
                                "ligand_atoms_mean": float(px_mean),
                                "ligand_atoms_std": float(px_std),
                                "optimization_strength": float(px_strength),
                            }
                        elif spec.engine_id == "flowr_root":
                            st.markdown("##### FLOWR.root sampling controls")
                            flow_columns = st.columns(4)
                            flow_steps = flow_columns[0].number_input(
                                "Integration steps",
                                min_value=10,
                                max_value=1000,
                                value=100,
                                step=10,
                                key="generation_flowr_steps",
                            )
                            corrector_steps = flow_columns[1].number_input(
                                "Corrector iterations",
                                min_value=0,
                                max_value=20,
                                value=0,
                                step=1,
                                key="generation_flowr_corrector",
                            )
                            solver = flow_columns[2].selectbox(
                                "ODE solver",
                                ("euler", "midpoint"),
                                key="generation_flowr_solver",
                            )
                            use_sde = flow_columns[3].checkbox(
                                "Stochastic SDE",
                                value=False,
                                key="generation_flowr_sde",
                            )
                            flow_filter_columns = st.columns(3)
                            variable_sizes = flow_filter_columns[0].checkbox(
                                "Sample molecule sizes",
                                value=False,
                                key="generation_flowr_sizes",
                            )
                            filter_diversity = flow_filter_columns[1].checkbox(
                                "Native diversity filter",
                                value=False,
                                key="generation_flowr_diversity_filter",
                            )
                            diversity_threshold = flow_filter_columns[2].number_input(
                                "Similarity ceiling",
                                min_value=0.1,
                                max_value=1.0,
                                value=0.9,
                                step=0.05,
                                disabled=not filter_diversity,
                                key="generation_flowr_diversity_threshold",
                            )
                            engine_settings["flowr_root"] = {
                                "integration_steps": int(flow_steps),
                                "corrector_steps": int(corrector_steps),
                                "solver": solver,
                                "use_sde_simulation": bool(use_sde),
                                "sample_molecule_sizes": bool(variable_sizes),
                                "filter_diversity": bool(filter_diversity),
                                "diversity_threshold": float(
                                    diversity_threshold
                                ),
                            }
                        elif spec.engine_id in {"conditar", "paopt"}:
                            st.markdown(
                                f"##### {spec.name} diffusion controls"
                            )
                            diffusion_columns = st.columns(2)
                            diffusion_steps = diffusion_columns[0].number_input(
                                "Diffusion steps",
                                min_value=100,
                                max_value=2000,
                                value=1000,
                                step=100,
                                key=f"generation_{spec.engine_id}_diffusion_steps",
                                help=(
                                    "Native default: 1000. Reducing this saves "
                                    "time but may reduce reconstruction quality."
                                ),
                            )
                            pocket_radius = diffusion_columns[1].number_input(
                                "Pocket radius (Å)",
                                min_value=4,
                                max_value=20,
                                value=10,
                                step=1,
                                key=f"generation_{spec.engine_id}_pocket_radius",
                            )
                            engine_settings.setdefault(
                                spec.engine_id, {}
                            ).update(
                                {
                                    "diffusion_steps": int(diffusion_steps),
                                    "pocket_radius": int(pocket_radius),
                                }
                            )
                        elif spec.engine_id == "drugrpg":
                            max_atoms = st.number_input(
                                "Generated ligand atoms",
                                min_value=5,
                                max_value=100,
                                value=30,
                                step=1,
                                key="generation_drugrpg_atoms",
                                help=(
                                    "DrugRPG generates a fixed requested atom "
                                    "count in its custom-pocket inference path."
                                ),
                            )
                            engine_settings["drugrpg"] = {
                                "max_atoms": int(max_atoms)
                            }
                        elif spec.engine_id == "pfm":
                            st.caption(
                                "PFM's validated custom-pocket path uses the "
                                "official learned atom-count prior and fixed "
                                "20-step ODE sampler; attempts and seed are the "
                                "supported generation controls."
                            )
                        elif spec.engine_id == "pgmg":
                            st.caption(
                                "PGMG has no native temperature or guidance "
                                "parameter. Direct it by editing/selecting the "
                                "pharmacophore points; generation is filtered "
                                "to unique RDKit-valid molecules."
                            )
                        if spec.engine_id == "paopt":
                            endpoints = [
                                "BBBP",
                                "HIA",
                                "hERG",
                                "Ames",
                                "Carcinogenicity",
                                "CaCo2",
                            ]
                            st.dataframe(
                                pd.DataFrame(
                                    [
                                        {
                                            "Endpoint": "BBBP",
                                            "Meaning": "Blood–brain barrier penetration probability",
                                            "Typical direction": "Target-dependent",
                                        },
                                        {
                                            "Endpoint": "HIA",
                                            "Meaning": "Human intestinal absorption probability",
                                            "Typical direction": "Maximize",
                                        },
                                        {
                                            "Endpoint": "hERG",
                                            "Meaning": "Cardiac hERG liability",
                                            "Typical direction": "Minimize",
                                        },
                                        {
                                            "Endpoint": "Ames",
                                            "Meaning": "Mutagenicity liability",
                                            "Typical direction": "Minimize",
                                        },
                                        {
                                            "Endpoint": "Carcinogenicity",
                                            "Meaning": "Carcinogenicity liability",
                                            "Typical direction": "Minimize",
                                        },
                                        {
                                            "Endpoint": "CaCo2",
                                            "Meaning": "Caco-2 permeability",
                                            "Typical direction": "Maximize",
                                        },
                                    ]
                                ),
                                hide_index=True,
                                width="stretch",
                            )
                            optimize_properties = st.multiselect(
                                "ADMET endpoints to steer",
                                endpoints,
                                default=["Carcinogenicity"],
                                key="generation_paopt_properties",
                            )
                            minimize_properties = st.multiselect(
                                "Endpoints to minimize",
                                optimize_properties,
                                default=[
                                    value
                                    for value in optimize_properties
                                    if value
                                    in {"hERG", "Ames", "Carcinogenicity"}
                                ],
                                key="generation_paopt_minimize",
                                help=(
                                    "Selected endpoints are pushed lower; all other "
                                    "selected endpoints are pushed higher."
                                ),
                            )
                            optimization_steps = st.number_input(
                                "paOPT steering steps",
                                min_value=1,
                                max_value=10,
                                value=1,
                                step=1,
                                key="generation_paopt_steps",
                            )
                            gradient_estimate_pairs = st.number_input(
                                "paOPT gradient estimate pairs",
                                min_value=1,
                                max_value=16,
                                value=4,
                                step=1,
                                key="generation_paopt_estimate_pairs",
                                help=(
                                    "Each pair adds two full diffusion passes per "
                                    "steering step. Upstream default: 4 pairs."
                                ),
                            )
                            perturbation_size = st.number_input(
                                "Finite-difference perturbation size",
                                min_value=0.001,
                                max_value=0.2,
                                value=0.03,
                                step=0.005,
                                format="%.3f",
                                key="generation_paopt_perturbation",
                                help=(
                                    "Controls gradient-estimation perturbations; "
                                    "too small is noisy, too large is less local. "
                                    "Native default: 0.03."
                                ),
                            )
                            engine_settings.setdefault("paopt", {}).update(
                                {
                                    "optimize_properties": optimize_properties,
                                    "minimize_properties": minimize_properties,
                                    "optimization_steps": int(
                                        optimization_steps
                                    ),
                                    "gradient_estimate_pairs": int(
                                        gradient_estimate_pairs
                                    ),
                                    "perturbation_size": float(
                                        perturbation_size
                                    ),
                                }
                            )
                            st.caption(
                                "Multiple endpoints are combined by paOPT's "
                                "native MGDA solver, which computes a balanced "
                                "descent direction. Manual endpoint weights are "
                                "not supported by the upstream inference code. "
                                "All endpoints are ADMET-AI predictions and need "
                                "independent computational and experimental validation."
                            )
                        st.info(
                            _ENGINE_PARAMETER_GUIDANCE.get(
                                spec.engine_id, ""
                            )
                        )
                    if spec.engine_id == "pgmg":
                        st.warning(
                            "PGMG is CC BY-NC-SA 4.0 and limited to eight compatible "
                            "pharmacophore points. Commercial use needs separate permission."
                        )

    with run_tab:
        selected_specs = [GENERATOR_BY_ID[item] for item in selected_engine_ids]
        render_run_resources(
            requires_gpu=bool(selected_specs),
            selected_gpu="Automatic",
            key="molecule_generation",
        )
        blockers: list[str] = []
        if target is None:
            blockers.append("Select a prepared target.")
        if pocket is None and reference_ligand is None and hypothesis is None:
            blockers.append(
                "Select at least one pocket, reference ligand, or pharmacophore hypothesis."
            )
        if not selected_specs:
            blockers.append("Select at least one generation engine.")
        campaign_name = st.text_input(
            "Campaign name",
            value="Pocket-conditioned molecule design",
            key="generation_campaign_name",
        )
        parameter_columns = st.columns(3)
        requested_count = parameter_columns[0].number_input(
            "Molecules per engine",
            min_value=1,
            max_value=100000,
            value=100,
            step=10,
            key="generation_requested_count",
        )
        batch_size = parameter_columns[1].number_input(
            "Native batch size",
            min_value=1,
            max_value=4096,
            value=32,
            step=1,
            key="generation_batch_size",
        )
        seed = parameter_columns[2].number_input(
            "Campaign seed",
            min_value=0,
            max_value=2147483647,
            value=2026,
            step=1,
            key="generation_seed",
        )
        if selected_specs:
            with st.expander(
                "Per-engine workload and time controls", expanded=True
            ):
                st.caption(
                    "The hard limit applies to native inference. At the deadline "
                    "the worker stops and removes the container, then imports and "
                    "normalizes any complete native molecules already written. "
                    "Use 0 for no hard limit."
                )
                for spec in selected_specs:
                    workload_columns = st.columns((1.4, 1, 1, 1.1, 1.2))
                    workload_columns[0].markdown(f"**{spec.name}**")
                    engine_count = workload_columns[1].number_input(
                        "Attempts",
                        min_value=1,
                        max_value=100000,
                        value=int(requested_count),
                        step=1,
                        key=f"generation_count_{spec.engine_id}",
                    )
                    engine_batch = workload_columns[2].number_input(
                        "Batch size",
                        min_value=1,
                        max_value=4096,
                        value=int(batch_size),
                        step=1,
                        key=f"generation_batch_{spec.engine_id}",
                    )
                    engine_seed = workload_columns[3].number_input(
                        "Seed",
                        min_value=0,
                        max_value=2147483647,
                        value=int(seed),
                        step=1,
                        key=f"generation_seed_{spec.engine_id}",
                        help=(
                            "Use different seeds for independent sampling "
                            "replicates; retain seeds for reproducibility."
                        ),
                    )
                    max_runtime_minutes = workload_columns[4].number_input(
                        "Hard limit (min)",
                        min_value=0,
                        max_value=10080,
                        value=0,
                        step=10,
                        key=f"generation_runtime_{spec.engine_id}",
                        help=(
                            "0 means unlimited. Final artifact import and "
                            "normalization run after inference is stopped and "
                            "may take a short additional time."
                        ),
                    )
                    engine_settings.setdefault(spec.engine_id, {}).update(
                        {
                            "requested_count": int(engine_count),
                            "batch_size": int(engine_batch),
                            "seed": int(engine_seed),
                            "max_runtime_seconds": int(
                                max_runtime_minutes
                            )
                            * 60,
                        }
                    )
            with st.expander("Time-cost planner", expanded=True):
                planning_window_minutes = st.number_input(
                    "Available inference time per engine (minutes)",
                    min_value=1,
                    max_value=10080,
                    value=60,
                    step=10,
                    key="generation_planning_window",
                    help=(
                        "Used only for estimates and suggestions; it does not "
                        "change a job or impose a hard limit."
                    ),
                )
                historical_rates = _successful_generation_rates()
                historical_yields = _successful_generation_yields()
                estimate_rows = []
                for spec in selected_specs:
                    settings = engine_settings.get(spec.engine_id, {})
                    count = int(
                        settings.get("requested_count") or requested_count
                    )
                    multiplier = _engine_pass_multiplier(
                        spec.engine_id, settings
                    )
                    rate, basis, evidence = _conservative_generation_rate(
                        spec.engine_id, historical_rates
                    )
                    seconds_per_attempt = rate * multiplier
                    estimated_seconds = count * seconds_per_attempt
                    suggested_attempts = max(
                        1,
                        math.floor(
                            float(planning_window_minutes)
                            * 60
                            / seconds_per_attempt
                        ),
                    )
                    yield_fraction, yield_basis = (
                        _conservative_generation_yield(
                            spec.engine_id, historical_yields
                        )
                    )
                    estimate_rows.append(
                        {
                            "Engine": spec.name,
                            "Requested attempts": count,
                            "Estimated inference": _format_duration(
                                estimated_seconds
                            ),
                            "Expected unique": math.floor(
                                count * yield_fraction
                            ),
                            f"Attempts in {int(planning_window_minutes)} min": (
                                suggested_attempts
                            ),
                            "Expected unique in window": math.floor(
                                suggested_attempts * yield_fraction
                            ),
                            "Relative work / attempt": round(multiplier, 2),
                            "Estimate basis": basis,
                            "Evidence": f"{evidence}; {yield_basis}",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(estimate_rows),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Estimates include observed container startup and inference "
                    "time but exclude queue waiting and downstream docking, "
                    "cofolding, PLIP/PandaMap validation, and ranking. paOPT uses "
                    "1 + steering steps × (2 × gradient pairs + 1) diffusion "
                    "passes per requested attempt. Engines sharing one GPU run "
                    "sequentially, so campaign inference time is approximately "
                    "the sum of their estimates. Expected unique counts use a "
                    "conservative historical valid, stereochemistry-aware "
                    "deduplicated yield; they are estimates, not guarantees."
                )
        readiness_rows = []
        ref_root = reference_root()
        for spec in selected_specs:
            missing = [
                relative
                for relative in spec.reference_paths
                if not (ref_root / relative).is_file()
            ]
            if missing:
                blockers.append(
                    f"{spec.name} is missing {len(missing)} required reference file(s)."
                )
            if engine_modes.get(spec.engine_id) == "No compatible conditioning is selected":
                blockers.append(f"{spec.name} has no compatible selected condition.")
            if not spec.adapter_ready:
                blockers.append(
                    f"{spec.name} needs its reviewed native configuration adapter."
                )
            if spec.engine_id == "pgmg" and hypothesis is None:
                blockers.append("PGMG requires a saved compatible pharmacophore.")
            if spec.engine_id == "pocketflow" and pocket is None:
                blockers.append("PocketFlow requires a coordinate pocket.")
            if spec.engine_id == "paopt":
                if reference_ligand is None:
                    blockers.append(
                        "conDitar + paOPT requires a reference ligand."
                    )
                if not engine_settings.get("paopt", {}).get(
                    "optimize_properties"
                ):
                    blockers.append(
                        "conDitar + paOPT requires at least one ADMET endpoint."
                    )
            readiness_rows.append(
                {
                    "Engine": spec.name,
                    "Theme": spec.theme,
                    "Conditioning path": engine_modes.get(spec.engine_id, ""),
                    "Image": spec.image,
                    "References": "Ready" if not missing else "Missing: " + ", ".join(missing),
                    "Adapter": (
                        "Implemented; focused native validation completed"
                        if spec.adapter_ready
                        else "Configuration adapter pending"
                    ),
                    "Required-contact handling": (
                        "Native pharmacophore + downstream validation"
                        if required_contacts
                        and (
                            spec.supports("pharmacophore")
                            or spec.supports("protein_pharmacophore")
                        )
                        else (
                            "Downstream pose validation only"
                            if required_contacts
                            else "Not requested"
                        )
                    ),
                }
            )
        if readiness_rows:
            st.dataframe(pd.DataFrame(readiness_rows), hide_index=True, width="stretch")
        for message in dict.fromkeys(blockers):
            st.info(message)
        st.caption(
            f"Reference files are resolved below `{ref_root}`. Model weights are "
            "never copied into the source checkout or baked into distributable images."
        )
        if hypothesis is not None:
            st.caption(
                "All enabled engine-compatible hypothesis points are native "
                "conditioning. Required points are additionally carried into "
                "downstream pose validation. PocketFlow receives only the pocket "
                "and therefore always uses pharmacophore contacts downstream."
            )
        acknowledge_experimental = st.checkbox(
            "I understand focused native execution is validated, while "
            "target-specific scientific validation remains required",
            value=False,
            key="generation_acknowledge_experimental",
        )
        if not acknowledge_experimental:
            blockers.append("Acknowledge the experimental native status.")
        queue = st.button(
            "Queue generation campaign",
            type="primary",
            disabled=bool(blockers),
            help=(
                "Creates one immutable shared campaign and one independently "
                "queued GPU child for each selected adapter-ready engine."
            ),
            key="generation_queue_campaign",
        )
        if queue and target is not None:
            objectives = {
                "affinity_interaction_fit": optimize_affinity,
                "admet": optimize_admet,
                "novelty": optimize_novelty,
                "diversity": optimize_diversity,
            }
            try:
                campaign = create_generation_campaign_job(
                    name=campaign_name,
                    engine_ids=selected_engine_ids,
                    target_artifact=target.artifact,
                    pocket_artifact=pocket.artifact if pocket else None,
                    reference_artifact=(
                        reference_ligand.artifact
                        if reference_ligand
                        else None
                    ),
                    pharmacophore_artifact=(
                        hypothesis.artifact if hypothesis else None
                    ),
                    engine_modes=engine_modes,
                    objectives=objectives,
                    requested_count=int(requested_count),
                    batch_size=int(batch_size),
                    seed=int(seed),
                    engine_settings=engine_settings,
                )
                children, child_errors = [], []
                for spec in selected_specs:
                    try:
                        children.append(
                            queue_generation_job(
                                campaign,
                                engine_id=spec.engine_id,
                                target_artifact=target.artifact,
                                pocket_artifact=(
                                    pocket.artifact if pocket else None
                                ),
                                reference_artifact=(
                                    reference_ligand.artifact
                                    if reference_ligand
                                    else None
                                ),
                                pharmacophore_artifact=(
                                    hypothesis.artifact
                                    if hypothesis
                                    else None
                                ),
                                engine_mode=engine_modes.get(
                                    spec.engine_id, ""
                                ),
                                objectives=objectives,
                                requested_count=int(
                                    engine_settings.get(
                                        spec.engine_id, {}
                                    ).get("requested_count", requested_count)
                                ),
                                batch_size=int(
                                    engine_settings.get(
                                        spec.engine_id, {}
                                    ).get("batch_size", batch_size)
                                ),
                                seed=int(
                                    engine_settings.get(
                                        spec.engine_id, {}
                                    ).get("seed", seed)
                                ),
                                engine_settings=engine_settings.get(
                                    spec.engine_id, {}
                                ),
                            )
                        )
                    except Exception as exc:
                        child_errors.append(f"{spec.name}: {exc}")
                campaign_code = display_job_code(
                    campaign.metadata.get("job_code"), campaign.run_id
                )
                if children:
                    st.success(
                        f"Created immutable campaign {campaign_code} and queued "
                        f"{len(children)} generation engine job(s)."
                    )
                else:
                    st.warning(
                        f"Campaign {campaign_code} was saved, but no engine "
                        "child could be queued."
                    )
                for error in child_errors:
                    st.error(error)
            except Exception as exc:
                st.error(f"Could not queue generation campaign: {exc}")

    _snapshot_engine_parameter_state()

    with results_tab:
        rows = []
        for job in iter_job_records(
            runs_root(), task_groups=(GENERATION_TASK_GROUP,)
        ):
            code = display_job_code(
                job.metadata.get("job_code"), job.run_id
            )
            query = urlencode(
                {
                    "task_group": job.task_group,
                    "run_id": job.run_id,
                    "label": code,
                }
            )
            rows.append(
                {
                    "Result": f"./job-results?{query}",
                    "Job": code,
                    "Campaign": display_job_code(
                        None, str(job.parent_run_id or "")
                    ),
                    "Engine": job.tool,
                    "Status": job.status,
                    "Termination": (
                        "Runtime budget exceeded"
                        if job.metadata.get("timed_out")
                        else ""
                    ),
                    "Generated": job.result.get("valid_compound_count"),
                    "Requested": job.metadata.get("requested_count"),
                    "Runtime": (
                        _format_duration(
                            float(job.metadata.get("runtime_seconds") or 0)
                        )
                        if job.metadata.get("runtime_seconds")
                        else ""
                    ),
                    "Hard limit": (
                        _format_duration(
                            float(
                                job.metadata.get(
                                    "max_runtime_seconds"
                                )
                                or 0
                            )
                        )
                        if job.metadata.get("max_runtime_seconds")
                        else "Unlimited"
                    ),
                    "Created": job.created_at,
                }
            )
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Result": st.column_config.LinkColumn(
                        "Result", display_text="Open"
                    )
                },
            )
        else:
            st.info("No typed molecule-generation jobs are available yet.")


render()
