from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from mn_ligand.app.pages.bound_ligand_md import _run_root

try:
    import py3Dmol  # type: ignore
except Exception:
    py3Dmol = None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _find_input_molecule(run_dir: Path) -> Path | None:
    for name in ("input_molecule.sdf", "input_molecule.mol", "input_molecule.xyz"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def _render_py3dmol_molecule(path: Path) -> None:
    if py3Dmol is None:
        st.info("`py3Dmol` is not available in this environment.")
        return
    ext = path.suffix.lower().lstrip(".")
    if ext not in {"sdf", "mol", "xyz", "pdb"}:
        st.info(f"Preview not supported for `{path.name}`.")
        return
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
        viewer = py3Dmol.view(width=900, height=460)
        viewer.addModel(data, ext)
        viewer.setStyle({"stick": {"radius": 0.2}})
        viewer.setBackgroundColor("white")
        viewer.zoomTo()
        components.html(viewer._make_html(), height=470)
    except Exception as exc:
        st.warning(f"Could not render molecule preview: {exc}")


def _to_float(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _grade_gap(gap_ev: float | None) -> tuple[str, str]:
    if gap_ev is None:
        return ("unknown", "No HOMO-LUMO gap value available.")
    if gap_ev < 2.0:
        return ("high_reactivity_risk", "Small gap (<2.0 eV): potentially more reactive.")
    if gap_ev <= 5.0:
        return ("balanced", "Moderate gap (2.0-5.0 eV): often acceptable for screening.")
    return ("low_reactivity_risk", "Large gap (>5.0 eV): often less reactive/more electronically stable.")


def _grade_dipole(dipole_d: float | None) -> tuple[str, str]:
    if dipole_d is None:
        return ("unknown", "No dipole value available.")
    if dipole_d < 1.0:
        return ("low_polarity", "Low dipole (<1.0 D): can indicate lower polarity.")
    if dipole_d <= 6.0:
        return ("balanced", "Moderate dipole (1.0-6.0 D): often practical in lead-like space.")
    return ("high_polarity", "High dipole (>6.0 D): may challenge permeability (context-dependent).")


def _render_interpretation_panel(result: dict) -> None:
    st.subheader("Quick Interpretation")
    st.caption("Triage guidance only. Not a binding affinity prediction.")
    gap_ev = _to_float(result.get("gap_eV"))
    dipole_d = _to_float(result.get("dipole_debye"))
    gap_label, gap_msg = _grade_gap(gap_ev)
    dip_label, dip_msg = _grade_dipole(dipole_d)
    c1, c2 = st.columns(2)
    with c1:
        if gap_label == "high_reactivity_risk":
            st.error(f"HOMO-LUMO gap: {gap_msg}")
        elif gap_label == "balanced":
            st.success(f"HOMO-LUMO gap: {gap_msg}")
        elif gap_label == "low_reactivity_risk":
            st.info(f"HOMO-LUMO gap: {gap_msg}")
        else:
            st.warning(f"HOMO-LUMO gap: {gap_msg}")
    with c2:
        if dip_label in {"low_polarity", "high_polarity"}:
            st.warning(f"Dipole: {dip_msg}")
        elif dip_label == "balanced":
            st.success(f"Dipole: {dip_msg}")
        else:
            st.warning(f"Dipole: {dip_msg}")
    st.caption(
        "Use this together with docking, MD stability, MM/GBSA/FEP, and ADMET results before decisions."
    )


def render() -> None:
    st.title("QC Results")
    run_id = st.query_params.get("run_id", "")
    if not run_id:
        st.info("No run selected. Open a run from Jobs – QC.")
        return
    run_dir = _run_root() / "qc" / str(run_id)
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
        input_mol = _find_input_molecule(run_dir)
        if input_mol is not None:
            st.subheader("Input Molecule (3D)")
            _render_py3dmol_molecule(input_mol)
            st.caption(str(input_mol))
        st.subheader("QC Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Energy (Hartree)", f"{result.get('energy_hartree', '-')}")
        col2.metric("HOMO (eV)", f"{result.get('homo_eV', '-')}")
        col3.metric("LUMO (eV)", f"{result.get('lumo_eV', '-')}")
        col4, col5 = st.columns(2)
        col4.metric("Gap (eV)", f"{result.get('gap_eV', '-')}")
        col5.metric("Dipole (D)", f"{result.get('dipole_debye', '-')}")
        _render_interpretation_panel(result)
        st.caption(str(result.get("message") or ""))
        with st.expander("Raw Result JSON", expanded=False):
            st.json(result)
    with st.expander("Metadata"):
        st.json(metadata)


render()
