from __future__ import annotations

import streamlit as st

from mn_ligand.app.pages import repair, target_trimming


def render() -> None:
    st.title("Target Sequence Modification")
    st.caption(
        "Modify the protein sequence and coordinates of an imported complex while "
        "retaining its ligand. Every operation creates a new typed target version; "
        "the source structure is never modified."
    )
    trimming_tab, repair_tab = st.tabs(["Trimming", "C-terminal Repair"])
    with trimming_tab:
        target_trimming.render(embedded=True)
    with repair_tab:
        repair.render(embedded=True)


render()
