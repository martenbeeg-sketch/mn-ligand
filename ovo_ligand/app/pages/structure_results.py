from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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


def _compute_ligand_rmsd_no_align(input_sdf: Path, docked_sdf: Path) -> tuple[float | None, str]:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
    except Exception:
        return None, "RDKit unavailable"
    try:
        def _load_first_mol(path: Path):
            sup = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
            mol = sup[0] if sup and len(sup) else None
            if mol is not None:
                return mol, "sanitize=True"
            sup2 = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
            mol2 = sup2[0] if sup2 and len(sup2) else None
            if mol2 is not None:
                return mol2, "sanitize=False"
            return None, "none"

        m_in, in_mode = _load_first_mol(input_sdf)
        m_out, out_mode = _load_first_mol(docked_sdf)
        if m_in is None or m_out is None:
            return None, f"Could not read ligand SDF(s) (input={in_mode}, output={out_mode})"
        c_in = m_in.GetConformer()
        c_out = m_out.GetConformer()

        def _rms_from_pairs(pairs: list[tuple[int, int]]) -> float:
            if not pairs:
                raise ValueError("Empty atom mapping")
            s = 0.0
            for out_idx, in_idx in pairs:
                p_out = c_out.GetAtomPosition(int(out_idx))
                p_in = c_in.GetAtomPosition(int(in_idx))
                dx = p_out.x - p_in.x
                dy = p_out.y - p_in.y
                dz = p_out.z - p_in.z
                s += dx * dx + dy * dy + dz * dz
            return (s / float(len(pairs))) ** 0.5

        # Preferred: strict atom-index mapping, no fitting.
        if m_in.GetNumAtoms() == m_out.GetNumAtoms():
            pairs = [(i, i) for i in range(m_in.GetNumAtoms())]
            return float(_rms_from_pairs(pairs)), f"direct(index; input={in_mode}, output={out_mode})"

        # Fallback: map common scaffold but still no alignment.
        mcs = rdFMCS.FindMCS([m_in, m_out], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=10)
        if not mcs or not mcs.smartsString:
            return None, "No common substructure found"
        q = Chem.MolFromSmarts(mcs.smartsString)
        if q is None:
            return None, "MCS query generation failed"
        in_match = m_in.GetSubstructMatch(q)
        out_match = m_out.GetSubstructMatch(q)
        if not in_match or not out_match or len(in_match) != len(out_match):
            return None, "MCS atom mapping failed"
        pairs = list(zip(out_match, in_match))
        return float(_rms_from_pairs(pairs)), f"MCS-no-align({len(pairs)} atoms; input={in_mode}, output={out_mode})"
    except Exception as exc:
        return None, f"RMSD error: {exc}"


def _compute_pose_rmsd_in_protein_frame(
    input_sdf: Path,
    docked_sdf: Path,
    input_protein_pdb: Path | None = None,
    docked_protein_pdb: Path | None = None,
) -> tuple[float | None, str]:
    try:
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
    except Exception:
        return None, "RDKit/numpy unavailable"

    def _load_first_mol(path: Path):
        sup = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        mol = sup[0] if sup and len(sup) else None
        if mol is not None:
            return mol, "sanitize=True"
        sup2 = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        mol2 = sup2[0] if sup2 and len(sup2) else None
        if mol2 is not None:
            return mol2, "sanitize=False"
        return None, "none"

    def _kabsch(mobile_xyz, ref_xyz):
        mobile_center = mobile_xyz.mean(axis=0)
        ref_center = ref_xyz.mean(axis=0)
        x = mobile_xyz - mobile_center
        y = ref_xyz - ref_center
        c = x.T @ y
        v, _, wt = np.linalg.svd(c)
        d = np.sign(np.linalg.det(v @ wt))
        dmat = np.diag([1.0, 1.0, d])
        r = v @ dmat @ wt
        t = ref_center - (mobile_center @ r)
        return r, t

    def _protein_transform():
        if input_protein_pdb is None or docked_protein_pdb is None:
            return None, "no-protein-align-input"
        if not input_protein_pdb.exists() or not docked_protein_pdb.exists():
            return None, "missing-protein-pdb"
        try:
            import mdtraj as md
            ref = md.load_pdb(str(input_protein_pdb))
            mob = md.load_pdb(str(docked_protein_pdb))
            sel_ref = ref.topology.select("backbone and not element H")
            sel_mob = mob.topology.select("backbone and not element H")
            if len(sel_ref) == 0 or len(sel_mob) == 0 or len(sel_ref) != len(sel_mob):
                return None, "protein-backbone-selection-mismatch"
            ref_xyz = ref.xyz[0, sel_ref, :]
            mob_xyz = mob.xyz[0, sel_mob, :]
            r, t = _kabsch(mob_xyz, ref_xyz)
            return (r, t), f"protein-kabsch({len(sel_ref)} atoms)"
        except Exception:
            return None, "protein-align-failed"

    try:
        m_in, in_mode = _load_first_mol(input_sdf)
        m_out, out_mode = _load_first_mol(docked_sdf)
        if m_in is None or m_out is None:
            return None, f"Could not read ligand SDF(s) (input={in_mode}, output={out_mode})"

        transform, tf_mode = _protein_transform()
        conf_in = m_in.GetConformer()
        conf_out = m_out.GetConformer()

        def _coords(mol, conf, indices):
            import numpy as np
            arr = []
            for idx in indices:
                p = conf.GetAtomPosition(int(idx))
                arr.append([float(p.x), float(p.y), float(p.z)])
            return np.asarray(arr, dtype=float)

        def _apply_transform(xyz):
            if transform is None:
                return xyz
            r, t = transform
            return xyz @ r + t

        if m_in.GetNumAtoms() == m_out.GetNumAtoms():
            heavy = [a.GetIdx() for a in m_in.GetAtoms() if a.GetAtomicNum() > 1]
            if not heavy:
                return None, "No heavy atoms found"
            xyz_in = _coords(m_in, conf_in, heavy)
            xyz_out = _coords(m_out, conf_out, heavy)
            xyz_out = _apply_transform(xyz_out)
            rmsd = float(np.sqrt(np.mean(np.sum((xyz_out - xyz_in) ** 2, axis=1))))
            return rmsd, f"{tf_mode}; heavy-index({len(heavy)}); input={in_mode}, output={out_mode}"

        mcs = rdFMCS.FindMCS([m_in, m_out], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=10)
        if not mcs or not mcs.smartsString:
            return None, "No common substructure found"
        q = Chem.MolFromSmarts(mcs.smartsString)
        if q is None:
            return None, "MCS query generation failed"
        in_match = m_in.GetSubstructMatch(q)
        out_match = m_out.GetSubstructMatch(q)
        if not in_match or not out_match or len(in_match) != len(out_match):
            return None, "MCS atom mapping failed"
        in_heavy = []
        out_heavy = []
        for i_idx, o_idx in zip(in_match, out_match):
            if m_in.GetAtomWithIdx(int(i_idx)).GetAtomicNum() > 1 and m_out.GetAtomWithIdx(int(o_idx)).GetAtomicNum() > 1:
                in_heavy.append(int(i_idx))
                out_heavy.append(int(o_idx))
        if not in_heavy:
            return None, "No heavy-atom MCS mapping"
        xyz_in = _coords(m_in, conf_in, in_heavy)
        xyz_out = _coords(m_out, conf_out, out_heavy)
        xyz_out = _apply_transform(xyz_out)
        rmsd = float(np.sqrt(np.mean(np.sum((xyz_out - xyz_in) ** 2, axis=1))))
        return rmsd, f"{tf_mode}; heavy-MCS({len(in_heavy)}); input={in_mode}, output={out_mode}"
    except Exception as exc:
        return None, f"RMSD error: {exc}"


def _resolve_input_ligand_sdf_for_overlay(run_dir: Path, metadata: dict) -> Path | None:
    # 0) For docking-derived structure jobs, prefer docking input ligand in work/input.
    work_input_sdf = next(iter(sorted((run_dir / "work" / "input").glob("*.sdf"))), None)
    if work_input_sdf is not None and work_input_sdf.exists():
        return work_input_sdf

    # 1) Prefer local raw ligand for direct PDB-prepared structures.
    local_raw = next(iter(sorted(run_dir.glob("*_ligand_raw.sdf"))), None)
    if local_raw is not None and local_raw.exists():
        return local_raw

    # 2) Docking-derived structure jobs: use source structure refined ligand as input reference.
    src_run_id = str(metadata.get("source_structure_run_id") or "").strip()
    if src_run_id:
        src_dir = _run_root() / "structure-jobs" / src_run_id
        if src_dir.exists():
            src_refined = next(iter(sorted(src_dir.glob("*_ligand_refined.sdf"))), None)
            if src_refined is not None and src_refined.exists():
                return src_refined
            src_raw = next(iter(sorted(src_dir.glob("*_ligand_raw.sdf"))), None)
            if src_raw is not None and src_raw.exists():
                return src_raw

    # 3) Fallback to local refined ligand if nothing else is present.
    local_refined = next(iter(sorted(run_dir.glob("*_ligand_refined.sdf"))), None)
    if local_refined is not None and local_refined.exists():
        return local_refined
    return None


def _resolve_docking_io_sdf(dock_dir: Path) -> tuple[str, str]:
    input_sdf = next(iter(sorted((dock_dir / "work" / "input").glob("*.sdf"))), None)
    out_sdf = next(iter(sorted((dock_dir / "work" / "results").glob("*.sdf"))), None)
    return str(input_sdf) if input_sdf else "", str(out_sdf) if out_sdf else ""


def _collect_docking_runs_for_structure(run_dir: Path) -> list[dict]:
    # New location: top-level structure-docking/<docking_run_id>
    # Backward compatibility: legacy nested structure-jobs/<structure_run_id>/docking_runs/<docking_run_id>
    dock_root_new = _run_root() / "structure-docking"
    dock_root_legacy = run_dir / "docking_runs"
    rows: list[dict] = []
    structure_run_id = run_dir.name
    candidates: list[Path] = []
    if dock_root_new.exists():
        candidates.extend([p for p in dock_root_new.iterdir() if p.is_dir()])
    if dock_root_legacy.exists():
        candidates.extend([p for p in dock_root_legacy.iterdir() if p.is_dir()])

    # Self-contained docking-structure job: include its own result if present.
    self_meta = _read_json(run_dir / "metadata.json")
    self_res = _read_json(run_dir / "result.json")
    if str(self_meta.get("workflow") or "").strip().upper() == "UDP_REDOCKING" and self_res:
        in_sdf, out_sdf = _resolve_docking_io_sdf(run_dir)
        input_protein = next(iter(sorted((run_dir / "work" / "input").glob("*_protein_refined.pdb"))), None)
        if input_protein is None:
            input_protein = next(iter(sorted((run_dir / "work" / "input").glob("*.pdb"))), None)
        docked_protein = next(iter(sorted(run_dir.glob("*_protein_refined.pdb"))), None)
        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": self_meta.get("job_code") or run_dir.name[:3].upper(),
                "engine": self_meta.get("engine") or "udp",
                "status": self_meta.get("status") or "unknown",
                "best_score_kcal_mol": self_res.get("best_score_kcal_mol"),
                "poses": self_res.get("result_files_count"),
                "created_at": self_meta.get("created_at") or "",
                "completed_at": self_meta.get("completed_at") or "",
                "success": self_res.get("success"),
                "run_dir": str(run_dir),
                "best_pose_pdbqt": self_res.get("best_pose_pdbqt") or "",
                "ligand_out_sdf": out_sdf,
                "input_ligand_sdf": in_sdf,
                "input_protein_pdb": str(input_protein) if input_protein else "",
                "docked_protein_pdb": str(docked_protein) if docked_protein else str(input_protein) if input_protein else "",
            }
        )

    for d in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        meta = _read_json(d / "metadata.json")
        if str(meta.get("source_structure_run_id") or "").strip() not in {"", structure_run_id}:
            continue
        res = _read_json(d / "result.json")
        in_sdf, out_sdf = _resolve_docking_io_sdf(d)
        input_protein = next(iter(sorted((d / "work" / "input").glob("*_protein_refined.pdb"))), None)
        if input_protein is None:
            input_protein = next(iter(sorted((d / "work" / "input").glob("*.pdb"))), None)
        docked_protein = next(iter(sorted(d.glob("*_protein_refined.pdb"))), None)
        rows.append(
            {
                "run_id": d.name,
                "job_code": meta.get("job_code") or d.name[:3].upper(),
                "engine": meta.get("engine") or "udp",
                "status": meta.get("status") or "unknown",
                "best_score_kcal_mol": res.get("best_score_kcal_mol"),
                "poses": res.get("result_files_count"),
                "created_at": meta.get("created_at") or "",
                "completed_at": meta.get("completed_at") or "",
                "success": res.get("success"),
                "run_dir": str(d),
                "best_pose_pdbqt": res.get("best_pose_pdbqt") or "",
                "ligand_out_sdf": out_sdf,
                "input_ligand_sdf": in_sdf,
                "input_protein_pdb": str(input_protein) if input_protein else "",
                "docked_protein_pdb": str(docked_protein) if docked_protein else str(input_protein) if input_protein else "",
            }
        )
    return rows


def _attach_rmsd_to_docking_rows(rows: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for row in rows:
        r = dict(row)
        input_sdf = Path(str(r.get("input_ligand_sdf") or ""))
        docked_sdf = Path(str(r.get("ligand_out_sdf") or ""))
        if input_sdf.exists() and docked_sdf.exists():
            input_protein = Path(str(r.get("input_protein_pdb") or ""))
            docked_protein = Path(str(r.get("docked_protein_pdb") or ""))
            rmsd, mode = _compute_pose_rmsd_in_protein_frame(
                input_sdf,
                docked_sdf,
                input_protein if str(input_protein) else None,
                docked_protein if str(docked_protein) else None,
            )
            r["pose_rmsd_A"] = round(float(rmsd), 3) if rmsd is not None else None
            r["rmsd_mode"] = mode
        else:
            r["pose_rmsd_A"] = None
            r["rmsd_mode"] = "missing input/output SDF"
        enriched.append(r)
    return enriched


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


def _render_py3dmol_refined_complex(
    protein_pdb_data: str,
    ligand_sdf_data: str,
    overlay_input_ligand_sdf_data: str = "",
    show_input_overlay: bool = False,
    input_overlay_opacity: float = 0.35,
    title: str = "Final refined selected complex",
    caption: str = "Prepared protein+ligand complex for downstream workflows.",
) -> None:
    try:
        import py3Dmol
    except Exception:
        st.warning("py3Dmol is not available; cannot render refined complex.")
        return

    st.markdown(f"##### {title}")
    st.caption(caption)
    view = py3Dmol.view(width=1200, height=560)
    if protein_pdb_data.strip():
        view.addModel(protein_pdb_data, "pdb")
        view.setStyle(
            {"model": 0},
            {
                "cartoon": {"color": "#9ec9f5", "opacity": 0.92},
                "line": {"hidden": True},
                "stick": {"hidden": True},
                "sphere": {"hidden": True},
            },
        )
    if ligand_sdf_data.strip():
        view.addModel(ligand_sdf_data, "sdf")
        ligand_model_index = 1 if protein_pdb_data.strip() else 0
        view.setStyle(
            {"model": ligand_model_index},
            {
                "stick": {"colorscheme": "cyanCarbon", "radius": 0.18},
                "sphere": {"scale": 0.14, "colorscheme": "cyanCarbon"},
            },
        )
    if show_input_overlay and overlay_input_ligand_sdf_data.strip():
        view.addModel(overlay_input_ligand_sdf_data, "sdf")
        input_model_index = (1 if protein_pdb_data.strip() else 0) + (1 if ligand_sdf_data.strip() else 0)
        view.setStyle(
            {"model": input_model_index},
            {
                "stick": {"colorscheme": "grayCarbon", "radius": 0.12, "opacity": float(input_overlay_opacity)},
                "sphere": {"scale": 0.10, "colorscheme": "grayCarbon", "opacity": float(input_overlay_opacity)},
            },
        )
    view.zoomTo()
    components.html(view._make_html(), height=580, scrolling=False)


def render() -> None:
    st.title("Results")
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
    with top[1]:
        if st.button("Back to Structure Jobs"):
            st.switch_page("app/pages/jobs_structure.py")

    raw_sdf = next(iter(sorted(run_dir.glob("*_ligand_raw.sdf"))), None)
    refined_sdf = next(iter(sorted(run_dir.glob("*_ligand_refined.sdf"))), None)
    refined_sdf_data = _load_text(refined_sdf) if refined_sdf else ""
    input_overlay_sdf = _resolve_input_ligand_sdf_for_overlay(run_dir, metadata)
    input_overlay_sdf_data = _load_text(input_overlay_sdf) if input_overlay_sdf else ""

    show_input_overlay = st.toggle(
        "Show input ligand overlay",
        value=True,
        help="Overlays the input ligand pose used for this preparation (orange) against the refined ligand (cyan).",
        key=f"structure_overlay_input_{run_id}",
    )
    input_overlay_opacity = st.slider(
        "Input overlay opacity",
        min_value=0.05,
        max_value=0.90,
        value=0.35,
        step=0.05,
        key=f"structure_overlay_opacity_{run_id}",
    )

    st.markdown("#### Refined complex (final)")
    if protein_pdb_data and refined_sdf_data:
        _render_py3dmol_refined_complex(
            protein_pdb_data=protein_pdb_data,
            ligand_sdf_data=refined_sdf_data,
            overlay_input_ligand_sdf_data=input_overlay_sdf_data,
            show_input_overlay=show_input_overlay,
            input_overlay_opacity=float(input_overlay_opacity),
            title="Final refined selected complex",
            caption="Rendered from refined protein PDB + refined ligand SDF.",
        )
        if show_input_overlay:
            if input_overlay_sdf and input_overlay_sdf_data.strip():
                st.caption(
                    f"Overlay loaded: input ligand `{input_overlay_sdf.name}` in gray; refined ligand in cyan."
                )
            else:
                st.warning("Input ligand overlay requested, but no input ligand SDF could be resolved for this run.")
        if selected_ligand:
            _render_ligand_summary(selected_ligand)
    else:
        st.warning("Refined protein and/or refined ligand file missing; cannot render final complex view.")

    st.markdown("#### Ligand correction preview (2D)")
    if raw_sdf and refined_sdf:
        _render_ligand_2d_pair(str(raw_sdf), str(refined_sdf))
    else:
        st.info("No generated OpenMM files (raw/refined ligand SDF preview) found for this run.")


    st.markdown("#### Docking results (from this prepared structure)")
    docking_rows = _collect_docking_runs_for_structure(run_dir)
    if not docking_rows:
        st.info("No docking runs yet for this structure. Use the `From UDP docking` tab in Structure Preparation.")
        return
    docking_rows = _attach_rmsd_to_docking_rows(docking_rows)
    df = pd.DataFrame(docking_rows)
    st.data_editor(
        df[
            [
                "job_code",
                "engine",
                "status",
                "best_score_kcal_mol",
                "pose_rmsd_A",
                "rmsd_mode",
                "poses",
                "created_at",
                "completed_at",
                "success",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        disabled=True,
        key=f"structure_docking_runs_{run_id}",
    )
    selected = docking_rows[0]
    st.caption(
        f"Latest run: `{selected['job_code']}` ({selected['engine']}) | "
        f"best score: {selected.get('best_score_kcal_mol')} kcal/mol | poses: {selected.get('poses')}"
    )
    pose_path = Path(str(selected.get("best_pose_pdbqt") or ""))
    if pose_path.exists():
        st.caption(f"Best pose file: `{pose_path}`")
    input_sdf = Path(str(selected.get("input_ligand_sdf") or ""))
    docked_sdf = Path(str(selected.get("ligand_out_sdf") or ""))
    if input_sdf.exists() and docked_sdf.exists():
        input_protein = Path(str(selected.get("input_protein_pdb") or ""))
        docked_protein = Path(str(selected.get("docked_protein_pdb") or ""))
        rmsd, mode = _compute_pose_rmsd_in_protein_frame(
            input_sdf,
            docked_sdf,
            input_protein if str(input_protein) else None,
            docked_protein if str(docked_protein) else None,
        )
        if rmsd is not None:
            st.metric("Input vs docked pose RMSD (A, protein-frame)", f"{rmsd:.3f}")
            st.caption(f"RMSD mapping: {mode}")
        else:
            st.info(f"Could not compute RMSD: {mode}")
    else:
        st.info("RMSD not available: docking input/output SDF files are missing for this run.")


render()
