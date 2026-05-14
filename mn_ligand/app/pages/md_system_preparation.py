import streamlit as st

st.session_state["md_task_mode"] = "system_prep"

from mn_ligand.app.pages.md import render


render()
