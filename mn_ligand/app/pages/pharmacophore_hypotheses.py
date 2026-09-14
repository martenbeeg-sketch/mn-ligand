from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from mn_ligand.app.pages.discover_inputs import (
    artifact_options,
    select_artifact,
    select_target_artifact,
    target_viewer_path,
)
from mn_ligand.app.viewers import render_persistent_3dmol
from mn_ligand.core.jobs import display_job_code, iter_job_records
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.pharmacophore import (
    FEATURE_TYPES,
    PHARMACOPHORE_TASK_GROUP,
    PharmacophoreFeature,
    create_pharmacophore_hypothesis_job,
    extract_interaction_supported_pharmacophore,
    extract_ligand_pharmacophore,
    extract_plip_residue_directed_pharmacophore,
    load_pharmacophore,
)


def _feature_frame(features: list[PharmacophoreFeature]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "_feature_id": item.feature_id,
                "enabled": item.enabled,
                "required": item.required,
                "feature_type": item.feature_type,
                "x": item.x,
                "y": item.y,
                "z": item.z,
                "radius": item.radius,
                "notes": item.notes,
                "_source": item.source,
                "_direction": json.dumps(
                    list(item.direction) if item.direction is not None else None
                ),
                "_source_atom_indices": json.dumps(list(item.source_atom_indices)),
                "_source_residues": json.dumps(list(item.source_residues)),
                "_metadata": json.dumps(item.metadata, sort_keys=True),
            }
            for item in features
        ]
    )


def _features_from_editor(frame: pd.DataFrame) -> list[PharmacophoreFeature]:
    features: list[PharmacophoreFeature] = []
    for index, row in frame.fillna("").iterrows():
        feature_type = str(row.get("feature_type") or "").strip()
        if feature_type not in FEATURE_TYPES:
            continue
        try:
            atom_indices = tuple(
                int(value)
                for value in json.loads(str(row.get("_source_atom_indices") or "[]"))
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            atom_indices = ()
        try:
            source_residues = tuple(
                str(value)
                for value in json.loads(str(row.get("_source_residues") or "[]"))
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            source_residues = ()
        try:
            metadata = dict(json.loads(str(row.get("_metadata") or "{}")))
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        try:
            raw_direction = json.loads(str(row.get("_direction") or "null"))
            direction = (
                tuple(float(value) for value in raw_direction)
                if isinstance(raw_direction, list) and len(raw_direction) == 3
                else None
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            direction = None
        features.append(
            PharmacophoreFeature(
                feature_id=str(
                    row.get("_feature_id")
                    or f"feature-{len(features) + 1:03d}"
                ),
                feature_type=feature_type,
                x=float(row.get("x")),
                y=float(row.get("y")),
                z=float(row.get("z")),
                radius=float(row.get("radius") or 1.0),
                enabled=bool(row.get("enabled", True)),
                required=bool(row.get("required", True)),
                direction=direction,
                source=str(row.get("_source") or "manual"),
                source_atom_indices=atom_indices,
                source_residues=source_residues,
                notes=str(row.get("notes") or ""),
                metadata=metadata,
            )
        )
    return features


def _coordinate_ligand_options() -> dict[str, object]:
    choices = artifact_options(
        ("prepared_ligand_set", "docked_pose", "pose_set", "validated_pose")
    )
    return {
        label: choice
        for label, choice in choices.items()
        if choice.artifact.path.lower().endswith((".sdf", ".mol", ".pdb", ".ent"))
    }


FEATURE_COLORS = {
    "HydrogenDonor": "#2563eb",
    "HydrogenAcceptor": "#ef4444",
    "Aromatic": "#8b5cf6",
    "PositiveIon": "#0ea5e9",
    "NegativeIon": "#f97316",
    "Hydrophobic": "#22c55e",
    "Halogen": "#06b6d4",
    "ExcludedVolume": "#64748b",
}

TARGET_INTERACTION_OPTIONS = {
    "Hydrogen bond · target donates, ligand accepts": {
        "kind": "hydrogen_bond_target_donor",
        "feature_type": "HydrogenAcceptor",
        "distance": 2.8,
        "guidance": (
            "Typical for Ser/Thr/Tyr hydroxyls and protonated Lys/Arg donors. "
            "For SER277:OG this is the recommended first hypothesis."
        ),
    },
    "Hydrogen bond · target accepts, ligand donates": {
        "kind": "hydrogen_bond_target_acceptor",
        "feature_type": "HydrogenDonor",
        "distance": 2.8,
        "guidance": (
            "Typical for Asp/Glu carboxylates, carbonyl oxygens, and an "
            "appropriately oriented neutral Ser/Thr/Tyr oxygen."
        ),
    },
    "Ionic · target positive, ligand negative": {
        "kind": "ionic_target_positive",
        "feature_type": "NegativeIon",
        "distance": 4.0,
        "guidance": (
            "Typical for protonated Arg/Lys sites. Confirm the expected "
            "protonation state before marking this for post-pose review."
        ),
    },
    "Ionic · target negative, ligand positive": {
        "kind": "ionic_target_negative",
        "feature_type": "PositiveIon",
        "distance": 4.0,
        "guidance": (
            "Typical for deprotonated Asp/Glu sites. Confirm the expected "
            "protonation state before marking this for post-pose review."
        ),
    },
    "Hydrophobic contact": {
        "kind": "hydrophobic_contact",
        "feature_type": "Hydrophobic",
        "distance": 3.8,
        "guidance": (
            "Use for a nonpolar pocket-facing atom or region. This is less "
            "directional than hydrogen bonding."
        ),
    },
    "Aromatic / π interaction": {
        "kind": "aromatic_pi_interaction",
        "feature_type": "Aromatic",
        "distance": 3.8,
        "guidance": (
            "Use near Phe/Tyr/Trp or an appropriate His ring. Ring-plane "
            "orientation still needs downstream pose validation."
        ),
    },
    "Halogen bond · ligand supplies halogen": {
        "kind": "halogen_bond_ligand_donor",
        "feature_type": "Halogen",
        "distance": 3.2,
        "guidance": (
            "Use when the selected target atom is a plausible halogen-bond "
            "acceptor. Directionality must be checked in the docked pose."
        ),
    },
}


def _interaction_complex_path(
    creation_metadata: dict[str, object],
) -> Path | None:
    run_id = str(creation_metadata.get("interaction_job_run_id") or "")
    pose_id = str(creation_metadata.get("pose_id") or "")
    if not run_id or not pose_id:
        return None
    path = (
        runs_root()
        / "interaction-analysis"
        / run_id
        / "prepared"
        / f"{pose_id}.complex.pdb"
    )
    return path if path.is_file() else None


def _row_metadata(row: pd.Series) -> dict[str, Any]:
    try:
        return dict(json.loads(str(row.get("_metadata") or "{}")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _numeric(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _render_pharmacophore_editor_viewer(
    structure_path: Path | None,
    frame: pd.DataFrame,
    *,
    focus_index: int,
    show_observed: bool,
    show_labels: bool,
    viewer_key: str,
) -> None:
    if structure_path is None or not structure_path.is_file():
        st.info(
            "Associate a coordinate target or PLIP-analyzed complex to enable "
            "the pharmacophore viewer."
        )
        return
    if structure_path.suffix.lower() not in {".pdb", ".ent"}:
        st.info("The pharmacophore viewer currently requires a PDB complex.")
        return
    try:
        import py3Dmol
    except ImportError:
        st.info("py3Dmol is unavailable in the current app environment.")
        return

    viewer = py3Dmol.view(width="100%", height=620)
    viewer.addModel(structure_path.read_text(errors="replace"), "pdb")
    viewer.setStyle(
        {"hetflag": False},
        {"cartoon": {"color": "#cbd5e1", "opacity": 0.82}},
    )
    viewer.setStyle(
        {"hetflag": True},
        {
            "stick": {"colorscheme": "cyanCarbon", "radius": 0.22},
            "sphere": {"colorscheme": "cyanCarbon", "scale": 0.16},
        },
    )
    author_numbered_structure = (
        "interaction-analysis" in structure_path.parts
        and "prepared" in structure_path.parts
    )

    visible_rows: list[tuple[int, pd.Series, dict[str, Any]]] = []
    for index, row in frame.fillna("").iterrows():
        if not bool(row.get("enabled", True)):
            continue
        metadata = _row_metadata(row)
        if bool(metadata.get("observed")) and not show_observed:
            continue
        visible_rows.append((int(index), row, metadata))

    highlighted_target_atoms: set[tuple[str, int, str]] = set()
    for index, row, metadata in visible_rows:
        feature_type = str(row.get("feature_type") or "")
        color = FEATURE_COLORS.get(feature_type, "#475569")
        center = {
            "x": _numeric(row.get("x")),
            "y": _numeric(row.get("y")),
            "z": _numeric(row.get("z")),
        }
        tolerance = max(0.2, _numeric(row.get("radius"), 1.0))
        is_required_contact = (
            bool(metadata.get("required_target_contact"))
            and bool(row.get("required", False))
        )
        is_focused = index == int(focus_index)
        viewer.addSphere(
            {
                "center": center,
                "radius": tolerance,
                "color": color,
                "opacity": (
                    0.55 if is_required_contact
                    else 0.32 if bool(metadata.get("observed"))
                    else 0.42
                ),
            }
        )
        viewer.addSphere(
            {
                "center": center,
                "radius": 0.22 if not is_focused else 0.34,
                "color": "#111827" if is_focused else color,
                "opacity": 1.0,
            }
        )
        protein_atom = metadata.get("protein_atom") or {}
        if is_required_contact and isinstance(protein_atom, dict):
            atom_coordinates = protein_atom.get("coordinates")
            if (
                isinstance(atom_coordinates, list)
                and len(atom_coordinates) == 3
            ):
                atom_center = {
                    "x": float(atom_coordinates[0]),
                    "y": float(atom_coordinates[1]),
                    "z": float(atom_coordinates[2]),
                }
                viewer.addCylinder(
                    {
                        "start": center,
                        "end": atom_center,
                        "radius": 0.06,
                        "color": "#be185d",
                        "opacity": 0.9,
                        "fromCap": 1,
                        "toCap": 1,
                    }
                )
                viewer.addSphere(
                    {
                        "center": atom_center,
                        "radius": 0.30,
                        "color": "#be185d",
                        "opacity": 1.0,
                    }
                )
            prepared_atom = metadata.get("prepared_protein_atom")
            display_atom = (
                protein_atom
                if author_numbered_structure
                else (
                    prepared_atom
                    if isinstance(prepared_atom, dict)
                    else protein_atom
                )
            )
            try:
                target_key = (
                    str(display_atom.get("chain") or ""),
                    int(display_atom.get("residue_number")),
                    str(display_atom.get("atom_name") or ""),
                )
            except (TypeError, ValueError):
                target_key = ("", 0, "")
            if target_key not in highlighted_target_atoms and target_key[1]:
                highlighted_target_atoms.add(target_key)
                viewer.addStyle(
                    {
                        "chain": target_key[0],
                        "resi": target_key[1],
                        "atom": target_key[2],
                    },
                    {
                        "stick": {"color": "#be185d", "radius": 0.20},
                        "sphere": {"color": "#be185d", "scale": 0.34},
                    },
                )
        if show_labels or is_focused or is_required_contact:
            label = (
                f"{index + 1}. {feature_type}"
                + (" · REVIEW TARGET" if is_required_contact else "")
            )
            viewer.addLabel(
                label,
                {
                    "position": center,
                    "fontColor": "#111827",
                    "backgroundColor": "white",
                    "backgroundOpacity": 0.78,
                    "fontSize": 10,
                },
            )

    viewer.setBackgroundColor("white")
    viewer.zoomTo({"hetflag": True})
    viewer.zoom(0.78)
    render_persistent_3dmol(
        viewer,
        key=viewer_key,
        height=640,
    )
    st.caption(
        "Protein: grey; bound ligand: cyan; tolerance volumes use feature "
        "colors. Magenta connects a post-pose review point to its target atom; "
        "it is not a guaranteed generated contact. The black center marks the "
        "focused table row."
    )


def render() -> None:
    st.title("Pharmacophore Hypotheses")
    st.caption(
        "Create an immutable, engine-neutral binding hypothesis, inspect and edit "
        "its features, then let each generation engine derive its private input format."
    )
    build_tab, results_tab = st.tabs(["Build / Edit", "Saved hypotheses"])

    with build_tab:
        use_target = st.checkbox(
            "Associate this hypothesis with a prepared target",
            value=True,
            key="pharmacophore_use_target",
        )
        target = (
            select_target_artifact(
                "Target context",
                ("prepared_target", "prepared_receptor", "prepared_complex"),
                key="pharmacophore_target",
                show_viewer=False,
            )
            if use_target
            else None
        )
        pocket = select_artifact(
            "Pocket context (optional)",
            ("pocket",),
            key="pharmacophore_pocket",
            required=False,
            source_run_id=target.job.run_id if target is not None else "",
        )
        mode = st.radio(
            "Starting point",
            (
                "Detect from coordinate ligand",
                "Derive from PLIP/PandaMap interactions",
                "Edit an existing hypothesis",
                "Start manually",
            ),
            horizontal=True,
            key="pharmacophore_start_mode",
        )
        source_choice = None
        initial: list[PharmacophoreFeature] = []
        creation_metadata: dict[str, object] = {}
        source_signature = mode
        error = ""
        if mode == "Detect from coordinate ligand":
            options = _coordinate_ligand_options()
            if options:
                label = st.selectbox(
                    "Reference or bound ligand",
                    list(options),
                    key="pharmacophore_source_ligand",
                )
                source_choice = options[label]
                source_signature = (
                    f"ligand:{source_choice.job.run_id}:"
                    f"{source_choice.artifact.artifact_id}"
                )
                path = source_choice.artifact.resolve(
                    source_choice.job.run_dir, must_exist=True
                )
                try:
                    if path is not None:
                        initial = extract_ligand_pharmacophore(path)
                except Exception as exc:
                    error = str(exc)
            else:
                st.info(
                    "No coordinate-bearing ligand artifact is available. Import a "
                    "complex, select a docked pose, or start manually."
                )
        elif mode == "Derive from PLIP/PandaMap interactions":
            options = artifact_options(("protein_ligand_interactions",))
            if options:
                label = st.selectbox(
                    "Completed interaction analysis",
                    list(options),
                    key="pharmacophore_interactions",
                )
                source_choice = options[label]
                input_table = (
                    source_choice.job.run_dir
                    / "input"
                    / "interaction_inputs.csv"
                )
                interaction_inputs: list[dict[str, str]] = []
                if input_table.is_file():
                    with input_table.open(
                        newline="", errors="replace"
                    ) as handle:
                        interaction_inputs = list(csv.DictReader(handle))
                pose_labels = {
                    (
                        f"{row.get('compound_id') or row.get('pose_id')} · "
                        f"{row.get('prediction') or row.get('pose_id')} · "
                        f"{row.get('pose_id')}"
                    ): row
                    for row in interaction_inputs
                }
                selected_pose: dict[str, str] | None = None
                if pose_labels:
                    pose_label = st.selectbox(
                        "Exact analyzed complex / pose",
                        list(pose_labels),
                        key="pharmacophore_interaction_pose",
                    )
                    selected_pose = pose_labels[pose_label]
                residue_directed = st.checkbox(
                    "Add a target side-chain interaction for post-pose review",
                    value=False,
                    key="pharmacophore_add_required_contact",
                    help=(
                        "PLIP observations remain reference features. A separate "
                        "ligand feature is placed toward the selected "
                        "author-numbered protein atom and flagged for later "
                        "scientific review."
                    ),
                )
                required_contact: dict[str, object] = {}
                if residue_directed:
                    st.caption(
                        "Residue numbering is the author numbering preserved by "
                        "PLIP, not a possibly renumbered prepared-receptor index."
                    )
                    contact_columns = st.columns(4)
                    required_contact["chain"] = contact_columns[0].text_input(
                        "Chain", value="A", key="pharmacophore_contact_chain"
                    )
                    required_contact["residue_name"] = contact_columns[1].text_input(
                        "Residue", value="SER", key="pharmacophore_contact_resname"
                    )
                    required_contact["residue_number"] = contact_columns[2].number_input(
                        "Author residue number",
                        min_value=1,
                        value=277,
                        step=1,
                        key="pharmacophore_contact_resnum",
                    )
                    required_contact["atom_name"] = contact_columns[3].text_input(
                        "Side-chain atom",
                        value="OG",
                        key="pharmacophore_contact_atom",
                    )
                    interaction_label = st.selectbox(
                        "Target interaction to review",
                        tuple(TARGET_INTERACTION_OPTIONS),
                        key="pharmacophore_contact_ligand_feature",
                        help=(
                            "Choose the protein-side chemistry. The app assigns "
                            "the complementary ligand pharmacophore feature."
                        ),
                    )
                    interaction_definition = TARGET_INTERACTION_OPTIONS[
                        interaction_label
                    ]
                    required_contact["interaction_kind"] = str(
                        interaction_definition["kind"]
                    )
                    required_contact["ligand_feature_type"] = str(
                        interaction_definition["feature_type"]
                    )
                    st.info(str(interaction_definition["guidance"]))
                    st.caption(
                        "Complementary ligand feature: "
                        f"`{required_contact['ligand_feature_type']}`. "
                        "OMTRA can export every listed choice. PGMG omits "
                        "negative-ion and halogen points; pocket-only engines "
                        "carry this only as a downstream pose-review target."
                    )
                    geometry_columns = st.columns(2)
                    required_contact["distance"] = geometry_columns[0].number_input(
                        "Target distance (Å)",
                        min_value=1.5,
                        max_value=6.0,
                        value=float(interaction_definition["distance"]),
                        step=0.1,
                        key=(
                            "pharmacophore_contact_distance_"
                            f"{interaction_definition['kind']}"
                        ),
                    )
                    required_contact["radius"] = geometry_columns[1].number_input(
                        "Tolerance (Å)",
                        min_value=0.2,
                        max_value=3.0,
                        value=1.0,
                        step=0.1,
                        key="pharmacophore_contact_radius",
                    )
                    required_contact["retain_backbone_contact"] = st.checkbox(
                        "Also retain an observed backbone hydrogen bond to this "
                        "same residue as an active feature",
                        value=False,
                        key="pharmacophore_retain_same_residue_backbone",
                        help=(
                            "Leave this off when the side-chain interaction is "
                            "intended to replace the reference backbone contact. "
                            "The original PLIP observation remains in provenance."
                        ),
                    )
                source_signature = (
                    f"interactions:{source_choice.job.run_id}:"
                    f"{source_choice.artifact.artifact_id}:"
                    f"{(selected_pose or {}).get('pose_id', '')}:"
                    f"{json.dumps(required_contact, sort_keys=True)}"
                )
                try:
                    if selected_pose is None:
                        raise ValueError(
                            "The interaction job has no immutable input inventory"
                        )
                    if residue_directed:
                        initial, creation_metadata = (
                            extract_plip_residue_directed_pharmacophore(
                                source_choice.job,
                                pose_id=str(selected_pose.get("pose_id") or ""),
                                protein_chain=str(required_contact["chain"]),
                                protein_residue_name=str(
                                    required_contact["residue_name"]
                                ),
                                protein_residue_number=int(
                                    required_contact["residue_number"]
                                ),
                                protein_atom_name=str(
                                    required_contact["atom_name"]
                                ),
                                ligand_feature_type=str(
                                    required_contact["ligand_feature_type"]
                                ),
                                interaction_kind=str(
                                    required_contact["interaction_kind"]
                                ),
                                target_distance_angstrom=float(
                                    required_contact["distance"]
                                ),
                                target_radius_angstrom=float(
                                    required_contact["radius"]
                                ),
                                retain_same_residue_backbone_contact=bool(
                                    required_contact[
                                        "retain_backbone_contact"
                                    ]
                                ),
                            )
                        )
                        st.info(
                            "PLIP ligand-atom observations are retained as "
                            "reference features. The target side-chain point is "
                            "a required design constraint and must be confirmed "
                            "by downstream docking and interaction analysis."
                        )
                        if not creation_metadata.get(
                            "reference_sidechain_contact_observed"
                        ):
                            st.warning(
                                "PLIP did not observe the requested side-chain "
                                "contact in the reference complex. This hypothesis "
                                "is deliberately proposing a new interaction."
                            )
                    else:
                        initial, creation_metadata = (
                            extract_interaction_supported_pharmacophore(
                                source_choice.job,
                                pose_id=selected_pose.get("pose_id"),
                            )
                        )
                        st.warning(
                            "Without a residue-directed PLIP constraint, the "
                            "normalized PLIP/PandaMap route supports pose-level "
                            "interaction classes and contacted residues, but not "
                            "an exact cross-engine ligand-atom contact map."
                        )
                    if creation_metadata.get("interaction_types"):
                        st.caption(
                            "Observed interaction classes: "
                            + ", ".join(
                                str(value)
                                for value in creation_metadata[
                                    "interaction_types"
                                ]
                            )
                        )
                except Exception as exc:
                    error = str(exc)
            else:
                st.info(
                    "No completed PLIP or PandaMap interaction artifact is "
                    "available. Analyze a prepared target complex or selected "
                    "prediction poses first."
                )
        elif mode == "Edit an existing hypothesis":
            options = artifact_options(("pharmacophore_hypothesis",))
            if options:
                label = st.selectbox(
                    "Existing immutable hypothesis",
                    list(options),
                    key="pharmacophore_existing",
                )
                source_choice = options[label]
                source_signature = (
                    f"hypothesis:{source_choice.job.run_id}:"
                    f"{source_choice.artifact.artifact_id}"
                )
                path = source_choice.artifact.resolve(
                    source_choice.job.run_dir, must_exist=True
                )
                try:
                    if path is not None:
                        initial = load_pharmacophore(path)
                        payload = json.loads(path.read_text())
                        creation_metadata = dict(
                            payload.get("creation_metadata") or {}
                        )
                        creation_metadata[
                            "edited_from_hypothesis_run_id"
                        ] = source_choice.job.run_id
                except Exception as exc:
                    error = str(exc)
            else:
                st.info("No saved hypothesis is available yet.")
        else:
            initial = [
                PharmacophoreFeature(
                    feature_id="feature-001",
                    feature_type="HydrogenAcceptor",
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    source="manual",
                )
            ]

        if error:
            st.error(error)
        state_key = "pharmacophore_editor_source"
        if st.session_state.get(state_key) != source_signature:
            st.session_state[state_key] = source_signature
            st.session_state["pharmacophore_feature_editor"] = _feature_frame(initial)
            st.session_state["pharmacophore_editor_revision"] = 0
        frame = st.session_state.get("pharmacophore_feature_editor", _feature_frame(initial))
        editor_revision = int(
            st.session_state.get("pharmacophore_editor_revision", 0)
        )
        editor_column, viewer_column = st.columns([1.08, 0.92])
        with editor_column:
            st.markdown("##### Feature table")
            st.caption(
                "Edit type, coordinates, tolerance, enabled state, or post-pose "
                "review flag. Add and remove rows directly; the viewer updates "
                "on rerun."
            )
            edited = st.data_editor(
                frame,
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                height=620,
                key=(
                    "pharmacophore_feature_editor_widget_"
                    f"{editor_revision}"
                ),
                column_config={
                    "feature_type": st.column_config.SelectboxColumn(
                        "Feature", options=list(FEATURE_TYPES), required=True
                    ),
                    "x": st.column_config.NumberColumn("X (Å)", format="%.3f"),
                    "y": st.column_config.NumberColumn("Y (Å)", format="%.3f"),
                    "z": st.column_config.NumberColumn("Z (Å)", format="%.3f"),
                    "radius": st.column_config.NumberColumn(
                        "Tolerance (Å)", min_value=0.1, format="%.2f"
                    ),
                    "required": st.column_config.CheckboxColumn(
                        "Post-pose review",
                        help=(
                            "Flag for later scientific inspection after docking, "
                            "cofolding, or refolding. This does not guarantee the "
                            "contact or automatically reject a design."
                        ),
                    ),
                    "_feature_id": None,
                    "_source": None,
                    "_direction": None,
                    "_source_atom_indices": None,
                    "_source_residues": None,
                    "_metadata": None,
                },
            ).reset_index(drop=True)
        with viewer_column:
            st.markdown("##### 3D pharmacophore editor")
            viewer_controls = st.columns(2)
            show_observed = viewer_controls[0].checkbox(
                "Observed PLIP features",
                value=True,
                key="pharmacophore_viewer_show_observed",
            )
            show_labels = viewer_controls[1].checkbox(
                "All labels",
                value=False,
                key="pharmacophore_viewer_show_labels",
            )
            focus_options = list(range(len(edited)))
            focus_index = (
                st.selectbox(
                    "Focus feature — viewer only",
                    focus_options,
                    format_func=lambda index: (
                        f"{index + 1}. "
                        f"{edited.iloc[index].get('feature_type', '')}"
                        + (
                            " · post-pose review"
                            if bool(edited.iloc[index].get("required", False))
                            else ""
                        )
                    ),
                    key="pharmacophore_viewer_focus",
                )
                if focus_options
                else 0
            )
            if focus_options:
                focused = edited.iloc[int(focus_index)]
                focused_type = str(focused.get("feature_type") or "")
                with st.expander("Fine-adjust focused feature", expanded=False):
                    with st.form(
                        f"pharmacophore_focus_form_{editor_revision}_{focus_index}"
                    ):
                        adjusted_type = st.selectbox(
                            "Feature type",
                            FEATURE_TYPES,
                            index=(
                                FEATURE_TYPES.index(focused_type)
                                if focused_type in FEATURE_TYPES
                                else 0
                            ),
                        )
                        coordinate_columns = st.columns(3)
                        adjusted_x = coordinate_columns[0].number_input(
                            "X (Å)",
                            value=_numeric(focused.get("x")),
                            format="%.3f",
                        )
                        adjusted_y = coordinate_columns[1].number_input(
                            "Y (Å)",
                            value=_numeric(focused.get("y")),
                            format="%.3f",
                        )
                        adjusted_z = coordinate_columns[2].number_input(
                            "Z (Å)",
                            value=_numeric(focused.get("z")),
                            format="%.3f",
                        )
                        adjusted_radius = st.number_input(
                            "Tolerance radius (Å)",
                            min_value=0.1,
                            value=_numeric(focused.get("radius"), 1.0),
                            step=0.1,
                            format="%.2f",
                        )
                        state_columns = st.columns(2)
                        adjusted_enabled = state_columns[0].checkbox(
                            "Enabled",
                            value=bool(focused.get("enabled", True)),
                        )
                        adjusted_required = state_columns[1].checkbox(
                            "Post-pose review",
                            value=bool(focused.get("required", False)),
                            help=(
                                "Records review intent; it is not a generation "
                                "guarantee or automatic acceptance rule."
                            ),
                        )
                        apply_adjustment = st.form_submit_button(
                            "Apply to feature"
                        )
                    if apply_adjustment:
                        updated = edited.copy()
                        updated.loc[int(focus_index), "feature_type"] = (
                            adjusted_type
                        )
                        updated.loc[int(focus_index), "x"] = adjusted_x
                        updated.loc[int(focus_index), "y"] = adjusted_y
                        updated.loc[int(focus_index), "z"] = adjusted_z
                        updated.loc[int(focus_index), "radius"] = (
                            adjusted_radius
                        )
                        updated.loc[int(focus_index), "enabled"] = (
                            adjusted_enabled
                        )
                        updated.loc[int(focus_index), "required"] = (
                            adjusted_required
                        )
                        st.session_state[
                            "pharmacophore_feature_editor"
                        ] = updated
                        st.session_state[
                            "pharmacophore_editor_revision"
                        ] = editor_revision + 1
                        st.rerun()
            structure_path = _interaction_complex_path(creation_metadata)
            if structure_path is None and target is not None:
                structure_path = target_viewer_path(target)
            _render_pharmacophore_editor_viewer(
                structure_path,
                edited,
                focus_index=int(focus_index),
                show_observed=show_observed,
                show_labels=show_labels,
                viewer_key=f"pharmacophore-editor:{source_signature}",
            )
            if st.button(
                "Reset all feature edits",
                key=f"pharmacophore_reset_{editor_revision}",
            ):
                st.session_state["pharmacophore_feature_editor"] = (
                    _feature_frame(initial)
                )
                st.session_state["pharmacophore_editor_revision"] = (
                    editor_revision + 1
                )
                st.rerun()
        name = st.text_input(
            "Hypothesis name",
            value="Ligand-derived pharmacophore"
            if mode
            in {
                "Detect from coordinate ligand",
                "Derive from PLIP/PandaMap interactions",
            }
            else "Pharmacophore hypothesis",
            key="pharmacophore_name",
        )
        st.caption(
            "Saving always creates a new immutable hypothesis. OMTRA/Pharmit JSON "
            "and XYZ exports are generated automatically. A PGMG `.posp` export is "
            "included only when the enabled feature set is compatible and has at most eight points."
        )
        st.info(
            "Native conditioning policy: every enabled compatible point is sent "
            "to OMTRA or PGMG. The post-pose review flag records what the "
            "scientist intends to inspect later; those native formats cannot "
            "encode that flag as feature priority. It does not guarantee a "
            "contact or automatically reject a molecule. Disable a reference "
            "point if it should remain visible in provenance but should not "
            "constrain generation."
        )
        if st.button(
            "Save immutable hypothesis",
            type="primary",
            disabled=edited.empty,
            key="pharmacophore_save",
        ):
            try:
                features = _features_from_editor(edited)
                job = create_pharmacophore_hypothesis_job(
                    name=name,
                    features=features,
                    source_job=source_choice.job if source_choice is not None else None,
                    source_artifact=(
                        source_choice.artifact if source_choice is not None else None
                    ),
                    target_artifact=target.artifact if target is not None else None,
                    pocket_artifact=pocket.artifact if pocket is not None else None,
                    creation_method={
                        "Detect from coordinate ligand": "rdkit-basefeatures-from-ligand",
                        "Derive from PLIP/PandaMap interactions": (
                            "plip-atom-resolved-residue-directed"
                            if creation_metadata.get("required_target_contacts")
                            else "interaction-supported-ligand-pharmacophore"
                        ),
                        "Edit an existing hypothesis": "edited-immutable-hypothesis",
                        "Start manually": "manual",
                    }[mode],
                    creation_metadata=creation_metadata,
                )
                st.success(
                    "Saved hypothesis "
                    f"{display_job_code(job.metadata.get('job_code'), job.run_id)}."
                )
            except Exception as exc:
                st.error(f"Could not save hypothesis: {exc}")

    with results_tab:
        rows = []
        for job in iter_job_records(
            runs_root(), task_groups=(PHARMACOPHORE_TASK_GROUP,)
        ):
            rows.append(
                {
                    "Job": display_job_code(job.metadata.get("job_code"), job.run_id),
                    "Name": job.metadata.get("name") or "Pharmacophore hypothesis",
                    "Method": job.metadata.get("creation_method") or "",
                    "Features": job.metadata.get("feature_count"),
                    "Enabled": job.metadata.get("enabled_feature_count"),
                    "Created": job.created_at,
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("No pharmacophore hypotheses have been saved.")


render()
