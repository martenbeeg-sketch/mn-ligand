from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import streamlit as st

from ovo_ligand.app.pages.common import _input_root
from ovo_ligand.app.pages.bound_ligand_md import (
    DEFAULT_MD_IMAGE,
    _parse_protein_chains,
    _prepare_structure_with_ligandx,
    _render_ligand_summary,
    _render_structure_view,
    _render_workflow_selection,
    _run_root,
    _short_job_code,
)
from ovo_ligand.workflows.bound_ligand_md import (
    MODIFIED_RESIDUE_MAPPINGS,
    _build_ligand_sdf_artifacts,
    _fetch_ccd_smiles,
    download_pdb,
    extract_ligand_pdb,
    parse_bound_ligands,
)


def _save_upload(scope: str, uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    safe_name = Path(uploaded_file.name).name
    target = _input_root() / "structure-preparation" / scope / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(uploaded_file.getvalue())
    return str(target)


def _switch_to(page: str) -> None:
    try:
        st.switch_page(page)
    except Exception:
        st.info(f"Continue in: `{page}`")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_structure_job(payload: dict) -> Path:
    run_id = str(uuid4())
    job_dir = _run_root() / "structure-jobs" / run_id
    job_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": run_id,
        "job_code": _short_job_code(run_id),
        "job_type": "structure",
        "status": "completed",
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        **payload,
    }
    (job_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return job_dir


def _atom_record_count(pdb_data: str) -> int:
    return sum(1 for line in pdb_data.splitlines() if line.startswith(("ATOM", "HETATM")))


def _residue_count(pdb_data: str) -> int:
    residues = set()
    for line in pdb_data.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
            continue
        residues.add((line[17:20].strip(), line[21].strip() or "_", line[22:26].strip(), line[26].strip() or "_"))
    return len(residues)


def _protein_only_pdb(pdb_data: str) -> str:
    lines: list[str] = []
    for line in pdb_data.splitlines():
        if line.startswith("ATOM"):
            lines.append(line)
        elif line.startswith(("TER", "MODEL", "ENDMDL", "CRYST1", "HEADER", "TITLE", "REMARK")):
            lines.append(line)
    return "\n".join(lines + ["END", ""])


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
        # Force a clean flat 2D depiction from graph connectivity (no mixed 3D-looking layout).
        m = Chem.Mol(mol)
        m = Chem.RemoveHs(m)
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(m)
        return m

    raw_2d = _flat_2d(raw_mol)
    refined_2d = _flat_2d(refined_mol)

    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"Raw ligand 2D (from SDF): {Path(raw_sdf_path).name}")
        st.image(Draw.MolToImage(raw_2d, size=(520, 360)))
    with c2:
        st.caption(f"Refined ligand 2D (from SDF): {Path(refined_sdf_path).name}")
        st.image(Draw.MolToImage(refined_2d, size=(520, 360)))


def _sdf_quick_summary(sdf_path: str) -> dict:
    try:
        from rdkit import Chem
        mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
        if mol is None:
            return {}
        return {
            "atoms": mol.GetNumAtoms(),
            "heavy_atoms": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1),
            "aromatic_atoms": sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
            "aromatic_bonds": sum(1 for b in mol.GetBonds() if b.GetIsAromatic()),
            "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True),
        }
    except Exception:
        return {}


def _extract_selected_complex_pdb(
    pdb_data: str,
    selected_protein_chains: list[str],
    selected_ligand_key: str,
) -> str:
    selected_chain_set = set(selected_protein_chains)
    lines: list[str] = []
    for line in pdb_data.splitlines():
        if line.startswith("ATOM"):
            chain = line[21].strip() or "_"
            if selected_chain_set and chain not in selected_chain_set:
                continue
            lines.append(line)
            continue
        if line.startswith("HETATM"):
            key = "|".join(
                [
                    line[17:20].strip(),
                    line[21].strip() or "_",
                    line[22:26].strip(),
                    line[26].strip() or "_",
                ]
            )
            if key == selected_ligand_key:
                lines.append(line)
            continue
        if line.startswith(("TER", "MODEL", "ENDMDL", "CRYST1", "HEADER", "TITLE", "REMARK")):
            lines.append(line)
    return "\n".join(lines + ["END", ""])


def render() -> None:
    st.title("Structure Preparation")
    st.caption("Prepare protein-ligand systems that can be reused by MD, free energy, and property workflows.")

    tabs = st.tabs(
        [
            "From PDB",
            "From Vina docking",
            "From Boltz prediction",
            "From custom files",
        ]
    )

    with tabs[0]:
        st.markdown("#### PDB -> Prepared complex")
        st.caption("MD-style wizard: download complex -> select chain/ligand -> prepare selected protein+ligand.")
        c1, c2 = st.columns([0.35, 0.65], vertical_alignment="bottom")
        with c1:
            pdb_id = st.text_input("PDB ID", value="4lnw", max_chars=4).strip().upper()
        with c2:
            run = st.button("Download and inspect complex", key="prep_from_pdb", type="primary")
        prepare_image = DEFAULT_MD_IMAGE
        prepare_use_gpu = False
        map_modified_residues = True
        st.caption("Preparation uses default containerized cleaning with modified-residue mapping enabled.")

        if run:
            try:
                raw_pdb = download_pdb(pdb_id)
                st.session_state["prep_wizard_pdb_id"] = pdb_id
                st.session_state["prep_wizard_raw_pdb_data"] = raw_pdb
                st.session_state["prep_wizard_ligands"] = parse_bound_ligands(raw_pdb)
                st.success(
                    f"Downloaded {pdb_id}. Inspect the whole complex, choose chains and ligand, then run preparation."
                )
            except Exception as exc:
                st.error(f"Preparation failed: {exc}")

        raw_pdb = st.session_state.get("prep_wizard_raw_pdb_data")
        ligands = st.session_state.get("prep_wizard_ligands", [])
        active_pdb_id = st.session_state.get("prep_wizard_pdb_id", pdb_id)
        if raw_pdb and ligands:
            if st.button("Start new structure preparation task", key="prep_start_new_task"):
                for key in [
                    "prep_wizard_pdb_id",
                    "prep_wizard_raw_pdb_data",
                    "prep_wizard_ligands",
                ]:
                    st.session_state.pop(key, None)
                st.rerun()

            excluded_resnames = set(MODIFIED_RESIDUE_MAPPINGS.keys())
            selectable_ligands = [lig for lig in ligands if lig.get("resname", "").upper() not in excluded_resnames]
            excluded_ligands = [lig for lig in ligands if lig.get("resname", "").upper() in excluded_resnames]
            if excluded_ligands:
                st.caption(
                    "Excluded from ligand selection (non-canonical residue mappings): "
                    + ", ".join(f"{lig['resname']} {lig['chain']}{lig['resseq']}" for lig in excluded_ligands)
                )
            if not selectable_ligands:
                st.warning("No selectable bound ligands found after excluding non-canonical residues.")
                return

            chain_entries = _parse_protein_chains(raw_pdb)
            chain_labels = {
                item["chain"]: f"Chain {item['chain']} ({item['residue_count']} residues, {item['start']}-{item['end']})"
                for item in chain_entries
            }
            chain_ids = [item["chain"] for item in chain_entries]
            selected_chains = st.multiselect(
                "Protein chain(s) to focus",
                options=chain_ids,
                default=chain_ids if chain_ids else [],
                format_func=lambda c: chain_labels.get(c, c),
                key=f"prep_chains_{active_pdb_id}",
            )
            ligand_labels = [f"{lig['resname']} chain {lig['chain']} residue {lig['resseq']}" for lig in selectable_ligands]
            selected_idx = st.radio(
                "Bound ligand",
                options=list(range(len(selectable_ligands))),
                index=0,
                format_func=lambda i: ligand_labels[i],
                key=f"prep_ligand_{active_pdb_id}",
            )
            selected = selectable_ligands[int(selected_idx)]
            _render_structure_view(
                raw_pdb,
                selectable_ligands,
                selected,
                selected_chains,
                show_molstar_tools=True,
                key_suffix=f"prep_raw_{active_pdb_id}",
                title="Complex view",
                caption="Full downloaded complex before preparation: selected ligand is red; other ligands are green.",
            )
            _render_ligand_summary(selected)
            _render_workflow_selection(selected_chains, selected)

            if st.button("Prepare selected protein and ligand", key=f"prep_selected_{active_pdb_id}", type="primary"):
                try:
                    with st.spinner("Running Ligand-X/HQBind-style protein preparation in container..."):
                        prepared_payload, command, proc = _prepare_structure_with_ligandx(
                            active_pdb_id,
                            raw_pdb,
                            prepare_image,
                            prepare_use_gpu,
                            map_modified_residues,
                        )
                    if proc.returncode != 0 or not prepared_payload.get("success"):
                        st.error("Containerized preparation failed; selected complex was not refined.")
                        with st.expander("Repair Docker command", expanded=True):
                            import shlex
                            st.code(shlex.join(command))
                        if proc.stdout:
                            with st.expander("repair stdout"):
                                st.code(proc.stdout)
                        if proc.stderr:
                            with st.expander("repair stderr"):
                                st.code(proc.stderr)
                        return

                    prepared = prepared_payload.get("prepared_pdb_data", raw_pdb)
                    mapping_report = prepared_payload.get("modified_residue_mapping", {})
                    prepared_ligands = parse_bound_ligands(prepared)
                    selected_prepared = next(
                        (lig for lig in prepared_ligands if lig.get("key") == selected.get("key")),
                        selected,
                    )
                    selected_complex_pdb = _extract_selected_complex_pdb(
                        prepared,
                        selected_protein_chains=selected_chains,
                        selected_ligand_key=selected_prepared["key"],
                    )
                    st.session_state["prepared_structure_last"] = {
                        "source": "pdb",
                        "pdb_id": active_pdb_id,
                        "prepared_pdb_data": prepared,
                        "prepared_selected_complex_pdb_data": selected_complex_pdb,
                        "ligands": prepared_ligands,
                        "selected_ligand": selected_prepared,
                        "selected_protein_chains": selected_chains,
                        "modified_mapping": mapping_report,
                    }
                    job_dir = _write_structure_job(
                        {
                            "source": "pdb",
                            "pdb_id": active_pdb_id,
                            "ligand_count": len(prepared_ligands),
                            "ligand_key": selected_prepared.get("key"),
                            "protein_chains": selected_chains,
                        }
                    )
                    st.success("Selected protein + ligand prepared.")
                    report_cols = st.columns(3)
                    report_cols[0].metric("Atom records (raw)", f"{_atom_record_count(raw_pdb):,}")
                    report_cols[1].metric("Atom records (prepared)", f"{_atom_record_count(prepared):,}")
                    report_cols[2].metric("Residues (prepared)", f"{_residue_count(prepared):,}")
                    st.markdown("#### Refined protein")
                    _render_structure_view(
                        prepared,
                        prepared_ligands,
                        selected_prepared,
                        selected_chains,
                        show_molstar_tools=True,
                        key_suffix=f"prep_refined_protein_{active_pdb_id}",
                        title="Refined protein view",
                        caption="Refined structure after protein cleanup and ligand reinsertion.",
                    )
                    with st.expander("Cleaning and mapping report", expanded=False):
                        st.markdown("What was done:")
                        st.markdown("- downloaded PDB structure from RCSB")
                        st.markdown("- selected protein chains + selected ligand")
                        st.markdown("- ran containerized Ligand-X protein cleaning/refinement")
                        st.markdown("- mapped supported modified residues to standard amino acids")
                        st.markdown("- removed dropped atoms, collapsed altloc variants, and reinserted ligands")
                        st.write(
                            {
                                "protein_cleaned": prepared_payload.get("protein_cleaned"),
                                "components": prepared_payload.get("components", {}),
                                "output_dir": prepared_payload.get("output_dir"),
                            }
                        )
                        st.json(mapping_report)
                    complex_refined_name = f"{active_pdb_id.lower()}_{selected_prepared['resname'].lower()}_complex_refined.pdb"
                    protein_refined_name = f"{active_pdb_id.lower()}_protein_refined.pdb"
                    protein_only_refined_pdb = _protein_only_pdb(prepared)
                    st.download_button(
                        "Download refined selected complex PDB",
                        data=selected_complex_pdb,
                        file_name=complex_refined_name,
                        mime="chemical/x-pdb",
                    )
                    (job_dir / complex_refined_name).write_text(selected_complex_pdb)
                    (job_dir / protein_refined_name).write_text(protein_only_refined_pdb)

                    ligand_pdb = extract_ligand_pdb(prepared, selected_prepared["key"])
                    file_prefix = f"{active_pdb_id.lower()}_{selected_prepared['resname'].lower()}"
                    reference_smiles = _fetch_ccd_smiles(selected_prepared["resname"])
                    if reference_smiles:
                        st.caption("Reference SMILES: downloaded from RCSB CCD")
                        ref_name = f"{active_pdb_id.lower()}_{selected_prepared['resname'].lower()}_ligand_ref.smi"
                        (job_dir / ref_name).write_text(reference_smiles + "\n")
                    else:
                        st.warning("Reference SMILES download failed for selected ligand (RCSB CCD).")
                    artifacts: dict = {}
                    ligand_artifact_error = ""
                    fallback_debug: list[str] = []
                    # Always persist raw ligand PDB in structure job folder.
                    ligand_raw_pdb_path = job_dir / f"{file_prefix}_ligand_raw.pdb"
                    ligand_raw_pdb_path.write_text(ligand_pdb if ligand_pdb.endswith("\n") else ligand_pdb + "\n")
                    try:
                        artifacts = _build_ligand_sdf_artifacts(
                            ligand_pdb=ligand_pdb,
                            ligand_resname=selected_prepared["resname"],
                            output_dir=job_dir,
                            file_prefix=file_prefix,
                            reference_smiles=reference_smiles,
                        )
                    except Exception as exc:
                        ligand_artifact_error = str(exc)
                        fallback_debug.append(f"builder_exception: {exc}")
                        # The builder is strict and may raise after writing artifacts.
                        # Reuse written files first before any fallback conversion.
                        for key, path in [
                            ("ligand_ref_smi", job_dir / f"{file_prefix}_ligand_ref.smi"),
                            ("ligand_raw_pdb", job_dir / f"{file_prefix}_ligand_raw.pdb"),
                            ("ligand_raw_sdf", job_dir / f"{file_prefix}_ligand_raw.sdf"),
                            ("ligand_refined_sdf", job_dir / f"{file_prefix}_ligand_refined.sdf"),
                        ]:
                            if path.exists():
                                artifacts[key] = str(path)
                                fallback_debug.append(f"reused_written_artifact: {path.name}")

                    # Fallback: if builder failed early, still try to emit raw/refined SDF locally.
                    if not (artifacts.get("ligand_raw_sdf") and Path(str(artifacts.get("ligand_raw_sdf"))).exists()):
                        try:
                            from rdkit import Chem
                            raw_mol = None
                            # Try multiple parse modes to maximize robustness.
                            try:
                                raw_mol = Chem.MolFromPDBBlock(
                                    ligand_pdb,
                                    removeHs=False,
                                    sanitize=False,
                                    proximityBonding=True,
                                )
                                fallback_debug.append(f"MolFromPDBBlock(sanitize=False) -> {'ok' if raw_mol is not None else 'none'}")
                            except Exception as parse_exc:
                                fallback_debug.append(f"MolFromPDBBlock error: {parse_exc}")
                            if raw_mol is None:
                                try:
                                    raw_mol = Chem.MolFromPDBFile(
                                        str(ligand_raw_pdb_path),
                                        removeHs=False,
                                        sanitize=False,
                                        proximityBonding=True,
                                    )
                                    fallback_debug.append(f"MolFromPDBFile(sanitize=False) -> {'ok' if raw_mol is not None else 'none'}")
                                except Exception as parse_exc:
                                    fallback_debug.append(f"MolFromPDBFile error: {parse_exc}")
                            if raw_mol is not None:
                                try:
                                    Chem.SanitizeMol(raw_mol)
                                    fallback_debug.append("SanitizeMol -> ok")
                                except Exception as san_exc:
                                    fallback_debug.append(f"SanitizeMol warning: {san_exc}")
                                raw_sdf_fallback = job_dir / f"{file_prefix}_ligand_raw.sdf"
                                with Chem.SDWriter(str(raw_sdf_fallback)) as writer:
                                    writer.write(raw_mol)
                                artifacts["ligand_raw_sdf"] = str(raw_sdf_fallback)
                                refined_sdf_fallback = job_dir / f"{file_prefix}_ligand_refined.sdf"
                                with Chem.SDWriter(str(refined_sdf_fallback)) as writer:
                                    writer.write(raw_mol)
                                artifacts["ligand_refined_sdf"] = str(refined_sdf_fallback)
                                fallback_debug.append("Fallback SDF write -> ok")
                            else:
                                fallback_debug.append("Fallback SDF write -> failed (no molecule parsed)")
                        except Exception as fallback_exc:
                            fallback_debug.append(f"Fallback SDF write -> exception: {fallback_exc}")
                    raw_sdf = artifacts.get("ligand_raw_sdf")
                    refined_sdf = artifacts.get("ligand_refined_sdf")
                    st.markdown("#### Ligand correction preview (2D)")
                    if raw_sdf and refined_sdf and Path(raw_sdf).exists() and Path(refined_sdf).exists():
                        _render_ligand_2d_pair(raw_sdf, refined_sdf)
                        st.caption(f"Reference SMILES source: {artifacts.get('reference_smiles_source', 'none')}")
                        with st.expander("Ligand correction report", expanded=False):
                            st.markdown("What was done:")
                            st.markdown("- extracted ligand from selected complex")
                            st.markdown("- downloaded/selected reference SMILES for template matching")
                            st.markdown("- assigned bond orders / aromaticity using the reference template")
                            st.markdown("- wrote raw and refined ligand SDF artifacts")
                            raw_stats = _sdf_quick_summary(raw_sdf)
                            refined_stats = _sdf_quick_summary(refined_sdf)
                            if raw_stats and refined_stats:
                                st.json(
                                    {
                                        "reference_smiles_source": artifacts.get("reference_smiles_source"),
                                        "reference_smiles_found": artifacts.get("reference_smiles_found"),
                                        "identity_status_code": artifacts.get("ligand_fix_identity_status"),
                                        "raw": raw_stats,
                                        "refined": refined_stats,
                                    }
                                )
                            if artifacts.get("ligand_fix_warning"):
                                st.warning(f"Stage-2 warning: {artifacts.get('ligand_fix_warning')}")
                            if artifacts.get("ligand_fix_error"):
                                st.error(f"Ligand correction error: {artifacts.get('ligand_fix_error')}")
                        if artifacts.get("ligand_fix_warning"):
                            st.warning(str(artifacts.get("ligand_fix_warning")))
                    else:
                        st.info("No 2D preview artifacts generated for this ligand.")
                    if ligand_artifact_error:
                        st.warning(f"Ligand refinement strict-check warning: {ligand_artifact_error}")
                    if fallback_debug:
                        (job_dir / f"{file_prefix}_ligand_artifact_debug.txt").write_text("\n".join(fallback_debug) + "\n")
                    if not (job_dir / f"{file_prefix}_ligand_raw.sdf").exists():
                        st.error(
                            "Ligand raw/refined SDF were not generated. "
                            f"Check debug file: {(job_dir / f'{file_prefix}_ligand_artifact_debug.txt').name}"
                        )

                    # Persist ligand correction artifacts directly in the structure job folder.
                    # Prefer explicit artifact paths, then fallback to any generated files in preview_dir.
                    explicit_paths = []
                    for key in ["ligand_ref_smi", "ligand_raw_pdb", "ligand_raw_sdf", "ligand_refined_sdf"]:
                        src = artifacts.get(key)
                        if src and Path(src).exists():
                            explicit_paths.append(Path(src))
                    if not explicit_paths:
                        explicit_paths = sorted(job_dir.glob(f"{file_prefix}_ligand_*.*"))

                    for src_path in explicit_paths:
                        suffix = src_path.suffix.lower()
                        if suffix not in {".pdb", ".sdf", ".smi"}:
                            continue
                        # Normalize ligand artifact naming: ..._ligand_raw.sdf / ..._ligand_refined.sdf
                        name = src_path.name
                        if "_ligand_raw" in name:
                            dst_name = f"{active_pdb_id.lower()}_{selected_prepared['resname'].lower()}_ligand_raw{suffix}"
                        elif "_ligand_refined" in name:
                            dst_name = f"{active_pdb_id.lower()}_{selected_prepared['resname'].lower()}_ligand_refined{suffix}"
                        elif "_ligand_ref" in name and suffix == ".smi":
                            dst_name = f"{active_pdb_id.lower()}_{selected_prepared['resname'].lower()}_ligand_ref{suffix}"
                        else:
                            dst_name = name
                        (job_dir / dst_name).write_bytes(src_path.read_bytes())
                    st.markdown("#### Refined complex (final)")
                    _render_structure_view(
                        selected_complex_pdb,
                        [selected_prepared],
                        selected_prepared,
                        selected_chains,
                        show_molstar_tools=True,
                        key_suffix=f"prep_refined_complex_{active_pdb_id}",
                        title="Final refined selected complex",
                        caption="Final selected protein+ligand complex used for downstream MD/free-energy workflows.",
                    )
                except Exception as exc:
                    st.error(f"Selected preparation failed: {exc}")

    with tabs[1]:
        st.markdown("#### Docking result -> Prepared complex")
        st.caption("Use your docking outputs as starting complexes for MD/analysis.")
        st.info("Run docking first, then come back here to standardize and register the complex.")
        if st.button("Open Ligand docking page", key="goto_docking"):
            _switch_to("app/pages/docking.py")
        if st.button("Open Batch ligand docking page", key="goto_batch_docking"):
            _switch_to("app/pages/batch_docking.py")

    with tabs[2]:
        st.markdown("#### Boltz output -> Prepared complex")
        st.caption("Use Boltz-2 predicted structures and normalize them for downstream workflows.")
        st.info("Generate or import a Boltz result, then map/select chain+ligand for preparation.")
        if st.button("Open Ligand Boltz-2 prediction page", key="goto_boltz"):
            _switch_to("app/pages/boltz2.py")

    with tabs[3]:
        st.markdown("#### Custom protein + ligand files")
        st.caption("Upload your own protein and ligand files and register them as prepared inputs.")
        protein_file = st.file_uploader("Protein file (PDB/mmCIF)", type=["pdb", "cif", "mmcif"], key="custom_protein")
        ligand_file = st.file_uploader("Ligand file (SDF/MOL2/SMILES TXT)", type=["sdf", "mol2", "smi", "txt"], key="custom_ligand")
        if st.button("Register custom prepared input", key="register_custom"):
            protein_path = _save_upload("custom/protein", protein_file)
            ligand_path = _save_upload("custom/ligand", ligand_file)
            if not protein_path or not ligand_path:
                st.warning("Upload both protein and ligand files first.")
            else:
                st.session_state["prepared_structure_last"] = {
                    "source": "custom",
                    "protein_path": protein_path,
                    "ligand_path": ligand_path,
                }
                _write_structure_job(
                    {
                        "source": "custom",
                        "protein_path": protein_path,
                        "ligand_path": ligand_path,
                    }
                )
                st.success("Custom input registered.")
                st.code(f"Protein: {protein_path}\nLigand:  {ligand_path}")

    st.divider()
    st.markdown("#### Next step")
    st.caption("Prepared structures can now be used by MD, ABFE/RBFE, and ligand property workflows.")
    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("Go to Bound ligand MD"):
            _switch_to("app/pages/bound_ligand_md.py")
    with n2:
        if st.button("Go to Ligand ABFE"):
            _switch_to("app/pages/abfe.py")
    with n3:
        if st.button("Go to Ligand RBFE"):
            _switch_to("app/pages/rbfe.py")


render()
