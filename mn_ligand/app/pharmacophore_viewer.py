from __future__ import annotations

from pathlib import Path
from typing import Sequence

import streamlit as st

from mn_ligand.app.viewers import render_persistent_3dmol
from mn_ligand.core.jobs import JobRecord, iter_job_records
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.pharmacophore import PharmacophoreFeature


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


def hypothesis_structure_path(job: JobRecord) -> Path | None:
    creation = job.metadata.get("creation_metadata")
    creation = dict(creation) if isinstance(creation, dict) else {}
    interaction_run_id = str(
        creation.get("interaction_job_run_id") or ""
    )
    pose_id = str(creation.get("pose_id") or "")
    if interaction_run_id and pose_id:
        candidate = (
            runs_root()
            / "interaction-analysis"
            / interaction_run_id
            / "prepared"
            / f"{pose_id}.complex.pdb"
        )
        if candidate.is_file():
            return candidate

    target_run_id = str(
        job.metadata.get("prepared_target_run_id") or ""
    )
    if not target_run_id:
        return None
    target = next(
        (
            candidate
            for candidate in iter_job_records(runs_root())
            if candidate.run_id == target_run_id
        ),
        None,
    )
    if target is None or target.artifact_manifest is None:
        return None
    for artifact_type in (
        "prepared_complex",
        "prepared_target",
        "prepared_receptor",
        "cleaned_structure",
    ):
        for artifact in target.artifact_manifest.by_type(artifact_type):
            path = artifact.resolve(target.run_dir, must_exist=True)
            if (
                path is not None
                and path.suffix.lower() in {".pdb", ".ent"}
            ):
                return path
    return None


def render_pharmacophore_viewer(
    structure_path: Path | None,
    features: Sequence[PharmacophoreFeature],
    *,
    focus_index: int = 0,
    show_observed: bool = True,
    show_labels: bool = False,
    viewer_key: str,
    height: int = 620,
) -> None:
    try:
        import py3Dmol
    except ImportError:
        st.info("py3Dmol is unavailable in the current app environment.")
        return
    if not features:
        st.info("The selected hypothesis contains no pharmacophore features.")
        return

    viewer = py3Dmol.view(width="100%", height=height)
    has_structure = bool(
        structure_path is not None
        and structure_path.is_file()
        and structure_path.suffix.lower() in {".pdb", ".ent"}
    )
    if has_structure and structure_path is not None:
        viewer.addModel(
            structure_path.read_text(errors="replace"),
            "pdb",
        )
        viewer.setStyle(
            {"hetflag": False},
            {"cartoon": {"color": "#cbd5e1", "opacity": 0.82}},
        )
        viewer.setStyle(
            {"hetflag": True},
            {
                "stick": {
                    "colorscheme": "cyanCarbon",
                    "radius": 0.22,
                },
                "sphere": {
                    "colorscheme": "cyanCarbon",
                    "scale": 0.16,
                },
            },
        )

    author_numbered_structure = bool(
        structure_path is not None
        and "interaction-analysis" in structure_path.parts
        and "prepared" in structure_path.parts
    )
    highlighted_target_atoms: set[tuple[str, int, str]] = set()
    visible_features: list[tuple[int, PharmacophoreFeature]] = []
    for index, feature in enumerate(features):
        if not feature.enabled:
            continue
        if bool(feature.metadata.get("observed")) and not show_observed:
            continue
        visible_features.append((index, feature))

    for index, feature in visible_features:
        metadata = dict(feature.metadata)
        color = FEATURE_COLORS.get(feature.feature_type, "#475569")
        center = {
            "x": float(feature.x),
            "y": float(feature.y),
            "z": float(feature.z),
        }
        tolerance = max(0.2, float(feature.radius))
        required_contact = bool(
            feature.required
            and metadata.get("required_target_contact")
        )
        focused = index == int(focus_index)
        viewer.addSphere(
            {
                "center": center,
                "radius": tolerance,
                "color": color,
                "opacity": (
                    0.55
                    if required_contact
                    else 0.32
                    if bool(metadata.get("observed"))
                    else 0.42
                ),
            }
        )
        viewer.addSphere(
            {
                "center": center,
                "radius": 0.34 if focused else 0.22,
                "color": "#111827" if focused else color,
                "opacity": 1.0,
            }
        )
        protein_atom = metadata.get("protein_atom")
        if required_contact and isinstance(protein_atom, dict):
            coordinates = protein_atom.get("coordinates")
            if (
                isinstance(coordinates, (list, tuple))
                and len(coordinates) == 3
            ):
                target_center = {
                    "x": float(coordinates[0]),
                    "y": float(coordinates[1]),
                    "z": float(coordinates[2]),
                }
                viewer.addCylinder(
                    {
                        "start": center,
                        "end": target_center,
                        "radius": 0.06,
                        "color": "#be185d",
                        "opacity": 0.9,
                        "fromCap": 1,
                        "toCap": 1,
                    }
                )
                viewer.addSphere(
                    {
                        "center": target_center,
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
            if (
                has_structure
                and target_key[1]
                and target_key not in highlighted_target_atoms
            ):
                highlighted_target_atoms.add(target_key)
                viewer.addStyle(
                    {
                        "chain": target_key[0],
                        "resi": target_key[1],
                        "atom": target_key[2],
                    },
                    {
                        "stick": {
                            "color": "#be185d",
                            "radius": 0.20,
                        },
                        "sphere": {
                            "color": "#be185d",
                            "scale": 0.34,
                        },
                    },
                )
        if show_labels or focused or required_contact:
            viewer.addLabel(
                (
                    f"{index + 1}. {feature.feature_type}"
                    + (" · REVIEW TARGET" if required_contact else "")
                ),
                {
                    "position": center,
                    "fontColor": "#111827",
                    "backgroundColor": "white",
                    "backgroundOpacity": 0.78,
                    "fontSize": 10,
                },
            )

    viewer.setBackgroundColor("white")
    viewer.zoomTo({"hetflag": True} if has_structure else {})
    viewer.zoom(0.78)
    render_persistent_3dmol(
        viewer,
        key=viewer_key,
        height=height + 20,
    )
    if not has_structure:
        st.info(
            "No associated coordinate target was available; only the saved "
            "pharmacophore geometry is shown."
        )
    st.caption(
        "Protein: grey; bound ligand: cyan; feature colors encode chemical "
        "roles. Magenta connects a post-pose review point to its target atom; "
        "it is not a guaranteed generated contact. The black center marks the "
        "focused feature."
    )
