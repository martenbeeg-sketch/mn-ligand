from __future__ import annotations

import streamlit as st

from mn_ligand.app.pages.discover_inputs import (
    ArtifactChoice,
    artifact_options,
    render_target_viewer,
    target_ligand_path,
    target_viewer_path,
)
from mn_ligand.core.jobs import display_job_code
from mn_ligand.workflows.target_orientation import ligand_longest_axis_transform


def _vector(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = str(st.query_params.get(name, "") or "")
    try:
        values = tuple(float(value) for value in raw.split(","))
    except ValueError:
        return default
    return values if len(values) == 3 else default


def _selected_targets() -> dict[str, ArtifactChoice]:
    requested = {
        value
        for value in str(st.query_params.get("targets", "") or "").split(",")
        if value
    }
    selected: dict[str, ArtifactChoice] = {}
    for _inventory_label, choice in artifact_options(
        ("prepared_target", "prepared_receptor")
    ).items():
        identity = f"{choice.job.run_id}:{choice.artifact.artifact_id}"
        if requested and identity not in requested:
            continue
        label = (
            str(choice.job.metadata.get("pdb_id") or choice.artifact.label)
            + " · job "
            + display_job_code(
                choice.job.metadata.get("job_code"), choice.job.run_id
            )
        )
        selected[label] = choice
    return selected


def render() -> None:
    st.title("Target ensemble 3D viewer")
    st.caption(
        "This page is isolated from Docking / Cofolding settings. Switching the "
        "model redraws only this viewer; campaign edits cannot make it disappear."
    )
    targets = _selected_targets()
    if not targets:
        st.error("No selected prepared targets could be resolved.")
        st.link_button("Back to Docking / Cofolding", "./discover-docking")
        return
    selected_label = st.selectbox(
        "Displayed target model",
        list(targets),
        key="target_ensemble_displayed_model",
    )
    choice = targets[selected_label]
    center = _vector("center", (0.0, 0.0, 0.0))
    size = _vector("size", (20.0, 20.0, 20.0))
    align = str(st.query_params.get("align", "0") or "0") == "1"
    show_box = str(st.query_params.get("show_box", "1") or "1") == "1"
    ligand_path = target_ligand_path(choice)
    coordinate_transform = None
    if align and ligand_path is not None:
        try:
            coordinate_transform = ligand_longest_axis_transform(ligand_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            st.warning(f"Axis alignment is unavailable for this model: {exc}")
    st.caption(
        "Global box: "
        + " × ".join(f"{value:.2f}" for value in size)
        + " Å at "
        + ", ".join(f"{value:.3f}" for value in center)
        + (" · ligand longest axis aligned to X" if align else "")
    )
    render_target_viewer(
        choice,
        viewer_path=target_viewer_path(choice),
        ligand_path=ligand_path,
        box={"center": center, "size": size} if show_box else None,
        selected_ligand_key=str(choice.job.metadata.get("ligand_key") or ""),
        show_box=show_box,
        cartoon_color="#94a3b8",
        ligand_color="cyanCarbon",
        box_color="#0891b2",
        show_box_center=show_box,
        coordinate_transform=coordinate_transform,
        key="target_ensemble_viewer",
        height=720,
    )
    st.link_button("Back to Docking / Cofolding", "./discover-docking")


if __name__ == "__main__":
    render()
