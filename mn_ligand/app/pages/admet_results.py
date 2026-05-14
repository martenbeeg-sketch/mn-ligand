from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from mn_ligand.app.pages.bound_ligand_md import _run_root


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def render() -> None:
    st.title("ADMET Results")
    run_id = st.query_params.get("run_id", "")
    if not run_id:
        st.info("No run selected. Open a run from Jobs – ADMET.")
        return
    run_dir = _run_root() / "admet" / str(run_id)
    if not run_dir.exists():
        st.error(f"Run directory not found: {run_dir}")
        return
    metadata = _read_json(run_dir / "metadata.json")
    result = _read_json(run_dir / "result.json")
    st.caption(f"Run: {run_id}")
    st.caption(f"Job code: {metadata.get('job_code', '-')}")
    st.caption(f"Status: {metadata.get('status', '-')}")
    if not result:
        st.info("No `result.json` found yet for this run.")
    else:
        smiles = str(result.get("smiles") or "").strip()
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    st.subheader("Molecule (2D)")
                    img = Draw.MolToImage(mol, size=(420, 300))
                    st.image(img)
            except Exception:
                pass
        st.subheader("ADMET Summary")
        for group in ["Physicochemical", "Absorption", "Distribution", "Metabolism", "Toxicity"]:
            values = result.get(group) or {}
            if not isinstance(values, dict) or not values:
                continue
            with st.expander(group, expanded=True):
                for k, v in values.items():
                    st.write(f"**{k}**: {v}")
        with st.expander("Raw Result JSON", expanded=False):
            st.json(result)
    with st.expander("Metadata"):
        st.json(metadata)


render()
