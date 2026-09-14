from __future__ import annotations

import streamlit as st

from mn_ligand.app.pages.discover_inputs import render_selected_artifacts, select_target_artifact


TOOLS = ("Boltz-2", "AlphaFold 3")


def render() -> None:
    st.title("Structure Prediction")
    target = select_target_artifact(
        "Prepared target",
        ("prepared_target", "prepared_receptor"),
        key="structure_prediction_target",
    )
    tool = st.segmented_control(
        "Prediction tool", TOOLS, default=TOOLS[0], key="structure_prediction_tool"
    ) or TOOLS[0]

    sampling, resources = st.tabs(["Sampling", "Resources"])
    with sampling:
        columns = st.columns(3)
        columns[0].number_input(
            "Models", min_value=1, max_value=100, value=5, step=1,
            key="structure_prediction_models",
        )
        columns[1].number_input(
            "Recycles", min_value=1, max_value=20, value=3, step=1,
            key="structure_prediction_recycles",
        )
        columns[2].number_input(
            "Diffusion samples", min_value=1, max_value=100, value=5, step=1,
            key="structure_prediction_samples",
        )
    with resources:
        columns = st.columns(3)
        columns[0].checkbox("Use MSA", value=True, key="structure_prediction_msa")
        columns[1].selectbox(
            "GPU", ("Automatic", "GPU 0", "GPU 1"), key="structure_prediction_gpu"
        )
        columns[2].selectbox(
            "Precision", ("Automatic", "BF16", "FP32"),
            key="structure_prediction_precision",
        )

    if target is not None:
        render_selected_artifacts({"Prepared target": target})
    else:
        st.info("Prepare a target first.")
        st.link_button("Open Structure Import", "./workspace-structure-preparation")
    st.button(
        f"Run structure prediction with {tool}",
        type="primary",
        disabled=True,
        help=f"The artifact contract is ready; execution awaits the typed {tool} adapter.",
        key="structure_prediction_run",
    )


render()
