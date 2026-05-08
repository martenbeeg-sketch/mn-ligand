from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ovo_ligand.app.pages.bound_ligand_md import _render_ligand_summary, _render_structure_view, _run_root
from ovo_ligand.workflows.bound_ligand_md import parse_bound_ligands


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_text(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


def _pick_selected_ligand(ligands: list[dict], metadata: dict) -> dict | None:
    ligand_key = str(metadata.get("ligand_key", "")).strip()
    if ligand_key:
        for lig in ligands:
            if lig.get("key") == ligand_key:
                return lig
    return ligands[0] if ligands else None


def _render_ligand_2d_pair(raw_sdf_path: str, refined_sdf_path: str) -> None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, rdDepictor
    except Exception:
        st.info("RDKit not available in this environment; 2D ligand preview is disabled.")
        return

    raw_supplier = Chem.SDMolSupplier(raw_sdf_path, removeHs=False)
    refined_supplier = Chem.SDMolSupplier(refined_sdf_path, removeHs=False)
    raw_mol = raw_supplier[0] if raw_supplier and len(raw_supplier) else None
    refined_mol = refined_supplier[0] if refined_supplier and len(refined_supplier) else None
    if raw_mol is None or refined_mol is None:
        st.warning("Could not build RDKit molecule(s) for 2D preview.")
        return

    def _flat_2d(mol):
        m = Chem.Mol(mol)
        m = Chem.RemoveHs(m)
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(m)
        return m

    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"Raw ligand 2D (from SDF): {Path(raw_sdf_path).name}")
        st.image(Draw.MolToImage(_flat_2d(raw_mol), size=(520, 360)))
    with c2:
        st.caption(f"Refined ligand 2D (from SDF): {Path(refined_sdf_path).name}")
        st.image(Draw.MolToImage(_flat_2d(refined_mol), size=(520, 360)))


def render() -> None:
    st.title("Structure Results")
    qp = st.query_params
    run_id = str(qp.get("run_id", "")).strip()
    if not run_id:
        st.info("No structure run selected. Open this page from Jobs – Structure.")
        if st.button("Back to Structure Jobs"):
            st.switch_page("app/pages/jobs_structure.py")
        return

    run_dir = _run_root() / "structure-jobs" / run_id
    if not run_dir.exists():
        st.error(f"Structure run not found: {run_id}")
        if st.button("Back to Structure Jobs"):
            st.switch_page("app/pages/jobs_structure.py")
        return

    metadata = _read_json(run_dir / "metadata.json")
    pdb_id = str(metadata.get("pdb_id", "")).lower()

    protein_refined = next(iter(sorted(run_dir.glob(f"{pdb_id}*_protein_refined.pdb"))), None) if pdb_id else None
    if protein_refined is None:
        protein_refined = next(iter(sorted(run_dir.glob("*_protein_refined.pdb"))), None)
    complex_refined = next(iter(sorted(run_dir.glob(f"{pdb_id}*_complex_refined.pdb"))), None) if pdb_id else None
    if complex_refined is None:
        complex_refined = next(iter(sorted(run_dir.glob("*_complex_refined.pdb"))), None)

    protein_pdb_data = _load_text(protein_refined) if protein_refined else ""
    complex_pdb_data = _load_text(complex_refined) if complex_refined else ""
    ligands = parse_bound_ligands(complex_pdb_data) if complex_pdb_data else []
    selected_ligand = _pick_selected_ligand(ligands, metadata)
    selected_chains = metadata.get("protein_chains") or []

    top = st.columns([0.85, 0.15])
    with top[0]:
        st.caption(f"Run: {run_id}")
        if metadata.get("job_code"):
            st.caption(f"Job code: {metadata.get('job_code')}")
        if metadata.get("pdb_id"):
            st.caption(f"PDB: {metadata.get('pdb_id')}")
    with top[1]:
        if st.button("Back to Structure Jobs"):
            st.switch_page("app/pages/jobs_structure.py")

    st.markdown("#### Refined protein")
    if protein_pdb_data:
        _render_structure_view(
            protein_pdb_data,
            [],
            selected_ligand or {},
            selected_chains,
            show_molstar_tools=True,
            key_suffix=f"structure_result_protein_{run_id}",
            title="Refined protein view",
            caption="Refined protein produced by structure preparation.",
        )
    else:
        st.warning("Refined protein file was not found.")

    st.markdown("#### Ligand correction preview (2D)")
    raw_sdf = next(iter(sorted(run_dir.glob("*_ligand_raw.sdf"))), None)
    refined_sdf = next(iter(sorted(run_dir.glob("*_ligand_refined.sdf"))), None)
    if raw_sdf and refined_sdf:
        _render_ligand_2d_pair(str(raw_sdf), str(refined_sdf))
    else:
        st.info("No generated OpenMM files (raw/refined ligand SDF preview) found for this run.")

    st.markdown("#### Refined complex (final)")
    if complex_pdb_data and selected_ligand:
        _render_structure_view(
            complex_pdb_data,
            ligands,
            selected_ligand,
            selected_chains,
            show_molstar_tools=True,
            key_suffix=f"structure_result_complex_{run_id}",
            title="Final refined selected complex",
            caption="Prepared protein+ligand complex for downstream workflows.",
        )
        _render_ligand_summary(selected_ligand)
    elif complex_pdb_data:
        st.code(complex_pdb_data[:4000])
    else:
        st.warning("Refined complex file was not found.")


render()
