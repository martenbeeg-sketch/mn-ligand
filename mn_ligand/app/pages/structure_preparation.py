from __future__ import annotations

import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4
import streamlit as st

from mn_ligand.app.pages.common import _input_root
from mn_ligand.app.pages.bound_ligand_md import (
    DEFAULT_MD_IMAGE,
    _parse_protein_chains,
    _render_ligand_summary,
    _render_structure_view,
    _render_workflow_selection,
    _run_root,
    _short_job_code,
)
from mn_ligand.app.pages.discover_inputs import select_target_artifact
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.app.viewers import render_persistent_3dmol
from mn_ligand.core.artifacts import (
    ArtifactRef,
    load_artifact_manifest,
    write_artifact_manifest,
    write_structure_artifact_manifest,
)
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, display_job_code, iter_job_records
from mn_ligand.core.provenance import target_lineage_summary
from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    registered_tool,
    write_registered_command_record,
)
from mn_ligand.core.workflows import (
    add_workflow_input,
    attach_workflow_child,
    create_workflow,
    refresh_workflow,
)
from mn_ligand.workflows.protein_preparation import (
    CANONICAL_AMINO_ACIDS,
    coordinate_gap_candidates,
    create_protein_import_job,
    detect_noncanonical_residues,
    detect_modeller_internal_gaps,
    imported_target,
    inline_structure_preview_data,
    load_protein_import_job,
    prepared_target,
    resolve_present_protein_chains,
    run_protein_cleaning_job,
)
from mn_ligand.workflows.predicted_complex_promotion import promote_predicted_complex
from mn_ligand.workflows.complex_prediction_inputs import (
    create_complex_prediction_inputs,
    normalize_ligand,
    normalize_sequence,
)
from mn_ligand.workflows.refolding import (
    DEFAULT_ALPHAFOLD3_IMAGE,
    DEFAULT_BOLTZ2_IMAGE,
    alphafast_readiness,
    boltz2_readiness,
    configured_alphafold3_reference_paths,
    configured_boltz2_cache_dir,
    find_cached_msa,
    queue_alphafold3_refolding_job,
    queue_boltz2_refolding_job,
)

from mn_ligand.workflows.bound_ligand_md import (
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
        **payload,
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": _short_job_code(run_id),
        "job_type": "structure",
        "status": "preparing",
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }
    (job_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return job_dir


def _complete_structure_job(job_dir: Path) -> None:
    metadata_path = job_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["status"] = "completed"
    metadata["updated_at"] = _utc_now_iso()
    metadata["completed_at"] = _utc_now_iso()
    metadata_path.write_text(json.dumps(metadata, indent=2))


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


def _render_py3dmol_complex_preview(
    pdb_data: str,
    ligand_resname: str = "LIG",
    ligand_sdf_path: str | None = None,
    center: tuple[float, float, float] | None = None,
    size: tuple[float, float, float] | None = None,
    persist_key: str = "structure-preparation-complex",
) -> None:
    import py3Dmol

    ligand_resnames: set[str] = set()
    for line in pdb_data.splitlines():
        if not line.startswith("HETATM") or len(line) < 20:
            continue
        resn = line[17:20].strip().upper()
        if not resn or resn in {"HOH", "WAT"}:
            continue
        ligand_resnames.add(resn)
    if not ligand_resnames and ligand_resname:
        ligand_resnames.add(str(ligand_resname).upper())

    view = py3Dmol.view(width=1200, height=560)
    view.addModel(pdb_data, "pdb")
    # MN-docking style defaults.
    view.setStyle({"cartoon": {"color": "#9ec9f5", "opacity": 0.95}})
    view.setStyle({"resn": "HOH"}, {"line": {"hidden": True}})
    view.addStyle({"hetflag": True}, {"stick": {"colorscheme": "magentaCarbon", "radius": 0.16}})
    for resn in sorted(ligand_resnames):
        view.addStyle({"resn": resn}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.22}})
        view.addStyle({"resn": resn}, {"sphere": {"scale": 0.18, "colorscheme": "cyanCarbon"}})

    # Robust ligand visibility: add the prepared ligand SDF as its own model.
    if ligand_sdf_path:
        try:
            ligand_sdf = Path(str(ligand_sdf_path))
            if ligand_sdf.exists():
                sdf_block = ligand_sdf.read_text()
                if sdf_block.strip():
                    view.addModel(sdf_block, "sdf")
                    # Last model is the explicit ligand model.
                    view.setStyle(
                        {"model": -1},
                        {"stick": {"colorscheme": "cyanCarbon", "radius": 0.22},
                         "sphere": {"scale": 0.18, "colorscheme": "cyanCarbon"}},
                    )
        except Exception:
            pass

    if center is not None and size is not None:
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
        corners = [
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz),
            (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz),
            (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz),
            (cx - hx, cy + hy, cz + hz),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for x, y, z in corners:
            view.addSphere(
                {
                    "center": {"x": x, "y": y, "z": z},
                    "radius": 0.45,
                    "color": "#d9f2ff",
                    "opacity": 0.95,
                }
            )
        for i0, i1 in edges:
            p0 = corners[i0]
            p1 = corners[i1]
            view.addCylinder(
                {
                    "start": {"x": p0[0], "y": p0[1], "z": p0[2]},
                    "end": {"x": p1[0], "y": p1[1], "z": p1[2]},
                    "radius": 0.10,
                    "color": "#22d3ee",
                    "fromCap": 1,
                    "toCap": 1,
                }
            )
        view.addLabel(
            f"Box center: {cx:.2f}, {cy:.2f}, {cz:.2f}",
            {
                "position": {"x": cx, "y": cy, "z": cz},
                "fontSize": 11,
                "backgroundColor": "#ffffff",
                "backgroundOpacity": 0.6,
                "fontColor": "#111827",
            },
        )
    view.zoomTo()
    render_persistent_3dmol(
        view,
        key=persist_key,
        height=580,
    )


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
    selected_chain_set = resolve_present_protein_chains(
        pdb_data,
        selected_protein_chains,
    )
    # Some historical OpenMM outputs rewrote a single deposited chain (for
    # example X) to A. Never publish a ligand-only "complex" merely because
    # the stored selection uses the pre-cleaning chain ID.
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
                # Canonicalize selected ligand residue name to LIG across prepared structures.
                # Keep chain/resseq/icode unchanged for traceability.
                lines.append(line[:17] + f"{'LIG':>3}" + line[20:])
            continue
        if line.startswith(("TER", "MODEL", "ENDMDL", "CRYST1", "HEADER", "TITLE", "REMARK")):
            lines.append(line)
    return "\n".join(lines + ["END", ""])


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}


def _infer_center_from_sdf(sdf_path: Path) -> tuple[float, float, float] | None:
    if not sdf_path.exists():
        return None
    try:
        lines = sdf_path.read_text().splitlines()
        if len(lines) < 4:
            return None
        atom_count = int(lines[3][0:3].strip())
        if atom_count <= 0 or len(lines) < 4 + atom_count:
            return None
        coords: list[tuple[float, float, float]] = []
        for line in lines[4 : 4 + atom_count]:
            try:
                x = float(line[0:10].strip())
                y = float(line[10:20].strip())
                z = float(line[20:30].strip())
            except Exception:
                continue
            coords.append((x, y, z))
        if not coords:
            return None
        n = float(len(coords))
        return (
            sum(c[0] for c in coords) / n,
            sum(c[1] for c in coords) / n,
            sum(c[2] for c in coords) / n,
        )
    except Exception:
        return None


def _pdb_atom_identity(line: str) -> dict:
    """Return a compact identity record for a PDB ATOM/HETATM line."""
    return {
        "atom_name": line[12:16].strip(),
        "resname": line[17:20].strip(),
        "chain": line[21].strip() or "_",
        "resseq": line[22:26].strip(),
        "icode": line[26].strip() or "_",
        "serial": line[6:11].strip(),
    }


def _remove_terminal_oxt_atoms(pdb_data: str) -> tuple[str, list[str]]:
    """Remove terminal OXT atoms from a PDB block.

    PDBFixer/OpenMM may emit terminal OXT atoms. In some prepared receptors,
    RDKit/Meeko infers an impossible proximity bond to OXT, for example
    CA-OXT, producing valence errors such as `C, 5, is greater than permitted`.
    This repair is applied only after the original receptor fails RDKit
    validation in `_prepare_receptor_pdb_for_meeko`.
    """
    kept: list[str] = []
    removed: list[str] = []

    for line in pdb_data.splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "OXT":
            removed.append(line)
            continue
        kept.append(line)

    return "\n".join(kept) + "\n", removed


def _remove_orphan_atoms_for_meeko(pdb_data: str) -> tuple[str, list[dict[str, str]]]:
    """Remove atoms with zero neighbors in RDKit proximity bonding graph.

    Meeko template matching assumes hydrogen atoms have a bonded heavy neighbor.
    In some prepared receptors, isolated atom records can appear and trigger
    `atom.GetNeighbors()[0]` crashes in Meeko.
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromPDBBlock(
            pdb_data,
            sanitize=False,
            removeHs=False,
            proximityBonding=True,
        )
        if mol is None:
            return pdb_data, []

        orphan_serials: set[str] = set()
        orphan_atoms: list[dict[str, str]] = []
        for atom in mol.GetAtoms():
            if atom.GetDegree() != 0:
                continue
            info = atom.GetPDBResidueInfo()
            if info is None:
                continue
            serial = str(info.GetSerialNumber()).strip()
            if not serial:
                continue
            orphan_serials.add(serial)
            orphan_atoms.append(
                {
                    "resname": info.GetResidueName().strip(),
                    "chain": info.GetChainId().strip() or "_",
                    "resseq": str(info.GetResidueNumber()),
                    "icode": info.GetInsertionCode().strip() or "_",
                    "atom": info.GetName().strip(),
                    "element": atom.GetSymbol().strip() or "?",
                    "serial": serial,
                }
            )
        if not orphan_serials:
            return pdb_data, []

        kept: list[str] = []
        for line in pdb_data.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                serial = line[6:11].strip()
                if serial in orphan_serials:
                    continue
            kept.append(line)
        return "\n".join(kept) + "\n", orphan_atoms
    except Exception:
        return pdb_data, []


def _remove_all_hydrogens_for_meeko(pdb_data: str) -> tuple[str, list[dict[str, str]]]:
    """Remove all explicit hydrogen atom records from receptor PDB text."""
    kept: list[str] = []
    removed: list[dict[str, str]] = []
    for line in pdb_data.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            kept.append(line)
            continue
        atom_name = line[12:16].strip().upper()
        element = (line[76:78].strip().upper() if len(line) >= 78 else "")
        is_h = atom_name.startswith("H") or element == "H"
        if is_h:
            removed.append(_pdb_atom_identity(line))
            continue
        kept.append(line)
    return "\n".join(kept) + "\n", removed


def _rdkit_validate_pdb_for_meeko(pdb_data: str) -> tuple[bool, str]:
    """Validate a receptor PDB with RDKit in the same failure mode Meeko hits.

    Meeko converts ProDy atoms to RDKit molecules and sanitizes them. Using
    `proximityBonding=True` here catches distance-inferred overbonding before
    `mk_prepare_receptor.py` is called.
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromPDBBlock(
            pdb_data,
            sanitize=False,
            removeHs=False,
            proximityBonding=True,
        )
        if mol is None:
            return False, "RDKit could not parse receptor PDB"

        Chem.SanitizeMol(mol)
        return True, "OK"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _prepare_receptor_pdb_for_meeko(pdb_data: str) -> tuple[str, dict]:
    """Return a Meeko-safe receptor PDB plus a repair report.

    Policy:
    1. Validate the original receptor with RDKit.
    2. If validation fails, remove terminal OXT atoms.
    3. Validate again.
    4. Return the repaired PDB and a JSON-serializable report.

    This keeps normal receptors unchanged while automatically fixing the known
    PDBFixer/OpenMM terminal OXT case observed for 4LNW ASP A 263.
    """
    report = {
        "original_valid": False,
        "final_valid": False,
        "original_error": "",
        "final_error": "",
        "removed_oxt_count": 0,
        "removed_oxt_atoms": [],
        "removed_orphan_h_count": 0,
        "removed_orphan_h_atoms": [],
        "removed_all_h_count": 0,
        "removed_all_h_atoms": [],
        "repair_applied": False,
    }

    original = pdb_data if pdb_data.endswith("\n") else pdb_data + "\n"
    original, removed_orphan_h = _remove_orphan_atoms_for_meeko(original)
    report["removed_orphan_h_count"] = len(removed_orphan_h)
    report["removed_orphan_h_atoms"] = removed_orphan_h
    if removed_orphan_h:
        report["repair_applied"] = True

    ok, msg = _rdkit_validate_pdb_for_meeko(original)
    report["original_valid"] = bool(ok)
    report["original_error"] = "" if ok else msg

    if ok:
        report["final_valid"] = True
        return original, report

    cleaned, removed_oxt = _remove_terminal_oxt_atoms(original)
    report["removed_oxt_count"] = len(removed_oxt)
    report["removed_oxt_atoms"] = [_pdb_atom_identity(line) for line in removed_oxt]
    report["repair_applied"] = bool(removed_oxt)

    ok2, msg2 = _rdkit_validate_pdb_for_meeko(cleaned)
    if not ok2:
        no_h, removed_all_h = _remove_all_hydrogens_for_meeko(cleaned)
        report["removed_all_h_count"] = len(removed_all_h)
        report["removed_all_h_atoms"] = removed_all_h
        if removed_all_h:
            report["repair_applied"] = True
        ok3, msg3 = _rdkit_validate_pdb_for_meeko(no_h)
        report["final_valid"] = bool(ok3)
        report["final_error"] = "" if ok3 else msg3
        return no_h, report

    report["final_valid"] = bool(ok2)
    report["final_error"] = "" if ok2 else msg2

    return cleaned, report


def _collect_refined_structure_jobs() -> list[dict]:
    runs_root = _run_root() / "structure-jobs"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        metadata = {}
        try:
            metadata = json.loads((run_dir / "metadata.json").read_text())
        except Exception:
            metadata = {}
        manifest = load_artifact_manifest(run_dir, task_group="structure-jobs")
        protein_refs = manifest.by_type("prepared_receptor")
        ligand_refs = manifest.by_type("prepared_ligand_set")
        complex_refs = manifest.by_type("prepared_complex")
        smiles_refs = tuple(item for item in manifest.by_type("compound_set") if item.role == "reference_smiles")
        protein_path = protein_refs[0].resolve(run_dir, must_exist=True) if protein_refs else None
        ligand_path = ligand_refs[0].resolve(run_dir, must_exist=True) if ligand_refs else None
        complex_path = complex_refs[0].resolve(run_dir, must_exist=True) if complex_refs else None
        smiles_path = smiles_refs[0].resolve(run_dir, must_exist=True) if smiles_refs else None
        if protein_path is None or ligand_path is None:
            continue
        repair_refs = manifest.by_type("repair_report")
        repair_path = (
            repair_refs[0].resolve(run_dir, must_exist=True) if repair_refs else None
        )
        provenance = _structure_result_provenance(metadata)
        protein_data = protein_path.read_text(errors="replace")
        if not provenance["chains"]:
            provenance["chains"] = ", ".join(
                sorted(
                    {
                        line[21].strip() or "_"
                        for line in protein_data.splitlines()
                        if line.startswith("ATOM  ") and len(line) > 21
                    }
                )
            )
        if not provenance["residues"]:
            provenance["residues"] = _residue_count(protein_data)
        if not provenance["formula"] or not provenance["molecular_weight"]:
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors, rdMolDescriptors

                supplier = Chem.SDMolSupplier(str(ligand_path), removeHs=False)
                molecule = supplier[0] if supplier and len(supplier) else None
                if molecule is not None:
                    provenance["formula"] = (
                        provenance["formula"]
                        or rdMolDescriptors.CalcMolFormula(molecule)
                    )
                    provenance["molecular_weight"] = (
                        provenance["molecular_weight"]
                        or round(float(Descriptors.MolWt(molecule)), 3)
                    )
            except Exception:
                pass
        md_assessment = _structure_md_readiness(
            protein_data,
            _read_json(repair_path) if repair_path is not None else {},
        )
        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": metadata.get("job_code") or _short_job_code(run_dir.name),
                "status": metadata.get("status") or "",
                "pdb_id": provenance["pdb_id"],
                "ligand_key": provenance["ligand_key"],
                "protein_pdb": str(protein_path),
                "ligand_sdf": str(ligand_path),
                "complex_pdb": str(complex_path) if complex_path is not None else "",
                "ligand_ref_smi": str(smiles_path) if smiles_path is not None else "",
                "source": metadata.get("source") or "",
                "tool": metadata.get("tool") or metadata.get("engine") or metadata.get("source") or "",
                "receptor_name": provenance["receptor_name"],
                "organism": provenance["organism"],
                "uniprot": provenance["uniprot"],
                "chains": provenance["chains"],
                "residues": provenance["residues"],
                "compound": provenance["compound"],
                "formula": provenance["formula"],
                "molecular_weight": provenance["molecular_weight"],
                "preparation": provenance["preparation"],
                "md_readiness": md_assessment["label"],
                "md_assessment": md_assessment,
                "created_at": metadata.get("created_at") or "",
            }
        )
    return rows


def _structure_result_provenance(metadata: dict) -> dict[str, object]:
    """Resolve display metadata through import and prediction parent jobs."""
    sources = [metadata]
    import_run_id = str(metadata.get("import_run_id") or "")
    if import_run_id:
        sources.append(
            _read_json(_run_root() / "protein-import" / import_run_id / "metadata.json")
        )

    prediction_run_id = str(metadata.get("source_prediction_run_id") or "")
    if prediction_run_id:
        prediction_dir = _run_root() / "refolding" / prediction_run_id
        prediction_metadata = _read_json(prediction_dir / "metadata.json")
        prediction_input = _read_json(prediction_dir / "input.json")
        sources.append(prediction_metadata)
        target_input = prediction_input.get("target") or {}
        target_run_id = str(target_input.get("run_id") or "")
        if target_run_id:
            sources.append(
                _read_json(
                    _run_root() / "structure-jobs" / target_run_id / "metadata.json"
                )
            )

    def first_value(key: str, default=""):
        return next((source.get(key) for source in sources if source.get(key)), default)

    receptor = next(
        (
            dict(source.get("receptor") or {})
            for source in sources
            if source.get("receptor")
        ),
        {},
    )
    ligands = next(
        (
            list(source.get("ligands") or [])
            for source in sources
            if source.get("ligands")
        ),
        [],
    )
    ligand_key = str(first_value("ligand_key"))
    selected_resname = ligand_key.split("|", 1)[0] if ligand_key else ""
    ligand = next(
        (
            item
            for item in ligands
            if not selected_resname
            or str(item.get("resname") or item.get("ccd_id") or "") == selected_resname
        ),
        ligands[0] if ligands else {},
    )
    entities = list(receptor.get("entities") or [])
    entity = entities[0] if entities else {}
    organisms = list(
        entity.get("source_organisms")
        or receptor.get("source_organisms")
        or []
    )
    uniprot_ids = list(entity.get("uniprot_ids") or [])
    chains = list(first_value("protein_chains", []) or entity.get("chains") or [])
    sequence_length = entity.get("sequence_length")
    if not sequence_length:
        sequence_length = first_value("protein_residues", "")
    preparation = "Cleaned/repaired" if first_value("cleaning_run_id") else "Imported"
    if metadata.get("clean_and_repair") is False:
        preparation = "Imported without repair"
    return {
        "pdb_id": str(first_value("pdb_id")),
        "ligand_key": ligand_key or str(ligand.get("key") or ligand.get("resname") or ""),
        "receptor_name": str(entity.get("name") or receptor.get("title") or ""),
        "organism": ", ".join(str(item) for item in organisms),
        "uniprot": ", ".join(str(item) for item in uniprot_ids),
        "chains": ", ".join(str(item) for item in chains),
        "residues": sequence_length or "",
        "compound": str(ligand.get("name") or ligand.get("ccd_id") or ""),
        "formula": str(ligand.get("formula") or ""),
        "molecular_weight": ligand.get("molecular_weight") or "",
        "preparation": preparation,
    }


def _structure_md_readiness(
    protein_pdb: str,
    repair_report: dict,
    *,
    peptide_bond_limit_angstrom: float = 1.8,
) -> dict[str, object]:
    """Assess sequence/topology continuity without treating separate chains as breaks."""
    residues: list[tuple[tuple[str, str, str], dict[str, tuple[float, float, float]]]] = []
    current_key: tuple[str, str, str] | None = None
    current_atoms: dict[str, tuple[float, float, float]] = {}
    for line in protein_pdb.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        key = (
            line[21].strip() or "_",
            line[22:26].strip(),
            line[26].strip() or "_",
        )
        if key != current_key:
            current_key = key
            current_atoms = {}
            residues.append((key, current_atoms))
        try:
            current_atoms[line[12:16].strip()] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        except ValueError:
            continue

    chains = sorted({key[0] for key, _ in residues})
    breaks: list[dict[str, object]] = []
    for chain in chains:
        chain_residues = [item for item in residues if item[0][0] == chain]
        for (left_key, left_atoms), (right_key, right_atoms) in zip(
            chain_residues, chain_residues[1:]
        ):
            if "C" not in left_atoms or "N" not in right_atoms:
                continue
            distance = sum(
                (left_atoms["C"][axis] - right_atoms["N"][axis]) ** 2
                for axis in range(3)
            ) ** 0.5
            if distance > peptide_bond_limit_angstrom:
                breaks.append(
                    {
                        "chain": chain,
                        "after": left_key[1],
                        "before": right_key[1],
                        "c_n_distance_angstrom": round(distance, 3),
                    }
                )

    fixer = dict(repair_report.get("pdbfixer") or repair_report)
    modeller_gaps = dict(repair_report.get("modeller_internal_gap_repair") or {})
    modeled = [
        *list(modeller_gaps.get("modeled_gaps") or []),
        *list(fixer.get("missing_residue_segments_added") or []),
    ]
    unresolved = [
        *list(modeller_gaps.get("skipped_gaps") or []),
        *list(fixer.get("missing_residue_segments_skipped") or []),
    ]
    if breaks or unresolved:
        level = "blocked"
        label = "Not MD-ready"
        message = (
            "Protein topology has unresolved sequence gaps or implausible peptide-bond "
            "distances. Repair or intentionally terminate the affected chain before MD."
        )
    elif modeled:
        level = "review"
        label = "Review modeled gap"
        message = (
            "The protein is topologically continuous, but missing residues were modeled "
            "from sequence records. Review the loop and equilibrate it carefully before "
            "production MD."
        )
    else:
        level = "pass"
        label = "Topology continuous"
        message = (
            "No unresolved internal sequence gaps or peptide-bond discontinuities were "
            "detected. This check does not replace force-field and system-setup validation."
        )
    return {
        "level": level,
        "label": label,
        "message": message,
        "protein_chains": chains,
        "separate_chain_count": len(chains),
        "peptide_bond_breaks": breaks,
        "modeled_missing_segments": modeled,
        "unresolved_missing_segments": unresolved,
        "sequence_records_available": fixer.get("sequence_records_available"),
        "peptide_bond_limit_angstrom": peptide_bond_limit_angstrom,
    }


def _render_structure_import_results() -> None:
    st.markdown("#### Imported and prepared complexes")
    st.caption(
        "All Structure Import results are shown here, independent of the input "
        "currently selected in another tab."
    )
    rows = _collect_refined_structure_jobs()
    if not rows:
        st.info("No prepared Structure Import results are available yet.")
        return

    all_jobs = list(iter_job_records(_run_root()))
    jobs_by_id = {job.run_id: job for job in all_jobs}
    design_campaign_counts: dict[str, int] = {}
    for job in all_jobs:
        if job.workflow != "molecule_generation_campaign":
            continue
        target_run_id = str(
            job.metadata.get("target_run_id")
            or job.parent_run_id
            or ""
        )
        if target_run_id:
            design_campaign_counts[target_run_id] = (
                design_campaign_counts.get(target_run_id, 0) + 1
            )
    table_rows = []
    for row in rows:
        code = str(row["job_code"])
        job = jobs_by_id.get(str(row["run_id"]))
        context = (
            target_lineage_summary(job, jobs_by_id)
            if job is not None
            else {
                "last_step": str(row.get("source") or "Unknown"),
                "origin": str(row.get("source") or "Unknown"),
            }
        )
        table_rows.append(
            {
                "job": (
                    f"./structure-results?"
                    f"{urlencode({'run_id': row['run_id'], 'label': code})}"
                ),
                "Last step": context["last_step"],
                "Design results": (
                    "./molecule-design-results?"
                    + urlencode({"target_run_id": row["run_id"]})
                    if design_campaign_counts.get(str(row["run_id"]), 0)
                    else ""
                ),
                "Design campaigns": design_campaign_counts.get(
                    str(row["run_id"]), 0
                ),
                "status": row.get("status") or "",
                "source": row.get("source") or "",
                "target": row.get("pdb_id") or "",
                "receptor": row.get("receptor_name") or "",
                "organism": row.get("organism") or "",
                "UniProt": row.get("uniprot") or "",
                "chains": row.get("chains") or "",
                "residues": row.get("residues") or "",
                "ligand": row.get("ligand_key") or "",
                "compound": row.get("compound") or "",
                "formula": row.get("formula") or "",
                "MW (Da)": row.get("molecular_weight") or "",
                "preparation": row.get("preparation") or "",
                "Origin / history": context["origin"],
                "MD readiness": row.get("md_readiness") or "",
                "tool": row.get("tool") or "",
                "complex": bool(row.get("complex_pdb")),
                "created": row.get("created_at") or "",
            }
        )
    table_event = st.dataframe(
        table_rows,
        hide_index=True,
        width="stretch",
        key="structure_import_results_table",
        on_select="rerun",
        selection_mode="single-row-required",
        selection_default={"selection": {"rows": [0]}},
        column_config={
            "job": st.column_config.LinkColumn(
                "Job",
                display_text=r"label=([^&]+)",
            ),
            "Design results": st.column_config.LinkColumn(
                "Design results",
                display_text="Open",
            ),
        },
    )
    selected_rows = list(table_event.selection.rows)
    selected_index = selected_rows[0] if selected_rows else 0
    selected = rows[selected_index] if 0 <= selected_index < len(rows) else rows[0]
    run_dir = _run_root() / "structure-jobs" / str(selected["run_id"])
    metadata = _read_json(run_dir / "metadata.json")
    result = _read_json(run_dir / "result.json")
    manifest = load_artifact_manifest(run_dir, task_group="structure-jobs")

    summary = st.columns(4)
    summary[0].metric("Job", str(selected["job_code"]))
    summary[1].metric("Source", str(selected.get("source") or "import"))
    summary[2].metric("Target", str(selected.get("pdb_id") or "—"))
    summary[3].metric("Ligand", str(selected.get("ligand_key") or "—"))

    md_assessment = dict(selected.get("md_assessment") or {})
    if md_assessment.get("level") == "blocked":
        st.error(str(md_assessment.get("message") or "Structure is not MD-ready."))
    elif md_assessment.get("level") == "review":
        st.warning(str(md_assessment.get("message") or "Review before MD."))
    else:
        st.info(str(md_assessment.get("message") or "Topology continuity check passed."))
    if int(md_assessment.get("separate_chain_count") or 0) > 1:
        st.caption(
            f"This complex contains {md_assessment['separate_chain_count']} separate "
            "protein chains. Separate biological chains are not reported as sequence breaks."
        )
    with st.expander("MD-readiness details", expanded=False):
        st.json(md_assessment)

    complex_path = Path(str(selected["complex_pdb"])) if selected.get("complex_pdb") else None
    protein_path = Path(str(selected["protein_pdb"]))
    if protein_path.is_file():
        protein_preview = protein_path.read_text(errors="replace")
        complex_preview = (
            complex_path.read_text(errors="replace")
            if complex_path is not None and complex_path.is_file()
            else ""
        )
        st.markdown("#### Full imported structure")
        _render_py3dmol_complex_preview(
            inline_structure_preview_data(complex_preview, protein_preview),
            ligand_sdf_path=str(selected["ligand_sdf"]),
            persist_key=f"imported-structure:{selected['run_id']}",
        )

    artifact_rows = [
        {
            "type": artifact.artifact_type,
            "role": artifact.role,
            "label": artifact.label,
            "relative path": artifact.path,
        }
        for artifact in manifest.artifacts
    ]
    detail_tabs = st.tabs(["Artifacts", "Import information", "Repair report"])
    with detail_tabs[0]:
        st.dataframe(artifact_rows, hide_index=True, width="stretch")
    with detail_tabs[1]:
        st.json({"metadata": metadata, "result": result})
    with detail_tabs[2]:
        repair_refs = manifest.by_type("repair_report")
        if not repair_refs:
            st.info("No repair report was published for this import.")
        else:
            repair_path = repair_refs[0].resolve(run_dir, must_exist=True)
            st.json(_read_json(repair_path) if repair_path is not None else {})

    st.link_button(
        "Open detailed Structure Results",
        f"./structure-results?{urlencode({'run_id': selected['run_id'], 'label': selected['job_code']})}",
        type="primary",
    )
    if design_campaign_counts.get(str(selected["run_id"]), 0):
        st.link_button(
            "Open combined Molecule Design Results",
            (
                "./molecule-design-results?"
                + urlencode({"target_run_id": selected["run_id"]})
            ),
            type="primary",
        )


def _is_structure_import_prediction(job: JobRecord) -> bool:
    if str(job.metadata.get("launch_context") or "") == "structure_import":
        return True
    parent_run_id = str(job.metadata.get("parent_run_id") or "")
    if not parent_run_id:
        return False
    parent_metadata = _run_root() / "complex-prediction-inputs" / parent_run_id / "metadata.json"
    try:
        payload = json.loads(parent_metadata.read_text())
    except (OSError, ValueError):
        return False
    return payload.get("workflow") == "complex_prediction_inputs"


def _prediction_complex_options(workflow: str) -> dict[str, tuple[JobRecord, ArtifactRef, Path]]:
    options: dict[str, tuple[JobRecord, ArtifactRef, Path]] = {}
    for job in iter_job_records(_run_root(), task_groups=("refolding",)):
        if (
            job.status != "completed"
            or job.workflow != workflow
            or job.artifact_manifest is None
            or not _is_structure_import_prediction(job)
        ):
            continue
        for artifact in job.artifact_manifest.by_type("predicted_complex"):
            path = artifact.resolve(job.run_dir, must_exist=True)
            if path is None:
                continue
            code = display_job_code(job.metadata.get("job_code"), job.run_id)
            label = f"{code} | {artifact.role or artifact.label} | {artifact.label}"
            options[label] = (job, artifact, path)
    return options


def _render_predicted_complex_preview(path: Path, *, key: str) -> None:
    st.markdown("#### Predicted complex")
    if path.stat().st_size > 10 * 1024 * 1024:
        st.warning("The predicted structure exceeds the 10 MB inline-preview limit.")
        return
    try:
        import py3Dmol

        suffix = path.suffix.lower()
        structure_format = "cif" if suffix in {".cif", ".mmcif"} else "pdb"
        viewer = py3Dmol.view(width=1100, height=560)
        viewer.addModel(path.read_text(errors="replace"), structure_format)
        viewer.setStyle(
            {"hetflag": False},
            {"cartoon": {"color": "spectrum", "opacity": 0.9}},
        )
        viewer.setStyle(
            {"hetflag": True},
            {
                "stick": {"colorscheme": "redCarbon", "radius": 0.2},
                "sphere": {"colorscheme": "redCarbon", "scale": 0.16},
            },
        )
        viewer.zoomTo()
        render_persistent_3dmol(
            viewer,
            key=f"predicted-complex:{key}:{path.resolve()}",
            height=580,
        )
        st.caption(
            f"Protein is shown as a chain-coloured cartoon; ligand/non-polymer atoms are red. "
            f"Source: `{path.name}`."
        )
    except Exception as exc:
        st.warning(f"Could not render the selected predicted complex: {exc}")


def _render_prediction_complex_promotion(*, workflow: str, engine_label: str, key: str) -> None:
    st.markdown(f"#### {engine_label} output → Prepared complex")
    st.caption(
        "Select a completed typed predicted-complex artifact. Promotion preserves "
        "the prediction run, runs strict Ligand-X/PDBFixer cleaning and repair, "
        "retains the ligand, and creates a separate downstream structure job."
    )
    options = _prediction_complex_options(workflow)
    if not options:
        st.info(
            f"No completed {engine_label} Structure Import predictions are available. "
            f"Launch one from this page's Run tab."
        )
        return
    selected_label = st.selectbox(
        f"{engine_label} predicted complex",
        tuple(options),
        key=f"{key}_prediction",
    )
    source_job, source_artifact, source_path = options[selected_label]
    metadata = {
        "candidate": source_artifact.role or source_artifact.label,
        **source_artifact.metadata,
    }
    st.json(metadata)
    _render_predicted_complex_preview(source_path, key=f"{key}_{source_artifact.artifact_id}")
    query = urlencode(
        {
            "task_group": source_job.task_group,
            "run_id": source_job.run_id,
            "label": display_job_code(source_job.metadata.get("job_code"), source_job.run_id),
        }
    )
    st.link_button("Open prediction results", f"./job-results?{query}")
    if st.button(
        f"Create prepared complex from {engine_label}",
        type="primary",
        key=f"{key}_promote",
    ):
        try:
            promoted = promote_predicted_complex(
                source_job=source_job,
                source_artifact=source_artifact,
                source_path=source_path,
            )
            if promoted.status == "completed":
                st.success(
                    "Created prepared complex "
                    f"{display_job_code(promoted.metadata.get('job_code'), promoted.run_id)}."
                )
            else:
                st.error(str(promoted.result.get("error") or "Prediction promotion failed."))
        except Exception as exc:
            st.error(str(exc))


def _render_sequence_complex_prediction(*, engine_label: str, key: str) -> None:
    workflow = "boltz2_refolding" if engine_label == "Boltz-2" else "alphafold3_refolding"
    input_tab, engine_tab, run_tab, completed_tab = st.tabs(
        ["Input", "Engine", "Run", "Completed predictions"]
    )
    with input_tab:
        st.markdown("#### One protein + one ligand")
        protein_input = st.text_area(
            "Protein sequence (FASTA or raw)",
            height=220,
            key=f"{key}_protein",
        )
        ligand_input = st.text_input(
            "Ligand input (LIGAND_ID,SMILES)",
            placeholder="T3,CCO",
            key=f"{key}_ligand",
        )
        st.caption(
            "Sequence prediction requires the 20 canonical one-letter amino-acid "
            "codes. Noncanonical chemistry must not be silently guessed: provide an "
            "explicit canonical substitution before prediction. The predicted complex "
            "is cleaned and repaired when it is promoted to a prepared complex. A full "
            "sequence is predicted end-to-end, so coordinate gaps apply only to imported "
            "experimental structures."
        )
        protein_error = ""
        ligand_error = ""
        try:
            protein_id, sequence = normalize_sequence(protein_input)
            st.caption(f"Protein `{protein_id}` · {len(sequence)} residues")
        except ValueError as exc:
            protein_id, sequence = "", ""
            protein_error = str(exc)
        try:
            ligand_id, ligand_smiles = normalize_ligand(ligand_input)
            st.caption(f"Ligand `{ligand_id}` · canonical SMILES `{ligand_smiles}`")
        except ValueError as exc:
            ligand_id, ligand_smiles = "", ""
            ligand_error = str(exc)

    af3_db, af3_weights, af3_msa = configured_alphafold3_reference_paths()
    local_msa_path = None
    local_msa_source = "missing"
    if sequence:
        local_msa_path, local_msa_source = find_cached_msa(sequence, af3_msa)
    if engine_label == "Boltz-2":
        readiness = boltz2_readiness()
        ready = bool(readiness["ready"])
    else:
        readiness = alphafast_readiness(af3_db, af3_weights, af3_msa)
        ready = bool(readiness["database_ready"] and readiness["weights_ready"])

    with engine_tab:
        if engine_label == "Boltz-2":
            settings = st.columns(3)
            recycles = int(
                settings[0].number_input("Recycling steps", 1, 12, 3, key=f"{key}_recycles")
            )
            diffusion_samples = int(
                settings[1].number_input("Diffusion samples", 1, 16, 5, key=f"{key}_samples")
            )
            sampling_steps = int(
                settings[2].number_input(
                    "Sampling steps", 10, 400, 200, 10, key=f"{key}_sampling_steps"
                )
            )
            extras = st.columns(2)
            use_potentials = extras[0].checkbox(
                "Use potentials", value=True, key=f"{key}_potentials"
            )
            seed = int(
                extras[1].number_input(
                    "Prediction seed", 1, 2_147_483_000, 1001, key=f"{key}_seed"
                )
            )
            if local_msa_path is not None:
                st.success(
                    f"Local MSA found in the shared repository ({local_msa_source}): "
                    f"`{local_msa_path.name}`"
                )
            else:
                st.warning(
                    "No matching local MSA was found. Boltz-2 Structure Import will not "
                    "contact an MSA server; add/generate this sequence's MSA in the shared "
                    "local repository before launching."
                )
        else:
            settings = st.columns(4)
            recycles = int(
                settings[0].number_input("Recycles", 1, 48, 10, key=f"{key}_recycles")
            )
            model_seeds = int(
                settings[1].number_input(
                    "Number of model seeds", 1, 20, 1, key=f"{key}_model_seeds"
                )
            )
            model_seed_start = int(
                settings[2].number_input(
                    "First model seed", 1, 2_147_483_000, 1, key=f"{key}_model_seed_start"
                )
            )
            batch_size = int(
                settings[3].number_input("MSA batch size", 1, 100, 1, key=f"{key}_batch")
            )
            if local_msa_path is not None:
                st.success(
                    f"Local cached MSA found ({local_msa_source}). AlphaFold 3 will embed "
                    "it directly and skip MSA generation."
                )
            else:
                st.info(
                    "No cached MSA was found. AlphaFold 3 will generate it locally with "
                    "the installation-managed AlphaFast/MMseqs databases; no MSA server is used."
                )
            st.caption(
                "AF3 preparation controls: recycling count, explicit native model-seed "
                "range, and local MSA pipeline batch size."
            )
        if not ready:
            st.warning(f"{engine_label} installation-managed references are unavailable.")

    with run_tab:
        gpu = st.selectbox(
            "GPU", ("Automatic", "GPU 0", "GPU 1"), key=f"{key}_gpu"
        )
        render_run_resources(requires_gpu=True, selected_gpu=gpu, key=key)
        blockers = []
        if protein_error:
            blockers.append(protein_error)
        if ligand_error:
            blockers.append(ligand_error)
        if not ready:
            blockers.append(f"{engine_label} references are unavailable.")
        if engine_label == "Boltz-2" and sequence and local_msa_path is None:
            blockers.append("A matching local MSA is required for Boltz-2 Structure Import.")
        for blocker in blockers:
            st.info(blocker)
        if st.button(
            f"Run {engine_label} complex preparation",
            type="primary",
            disabled=bool(blockers),
            key=f"{key}_run",
        ):
            try:
                input_job, target_artifact, compound_artifact, proteins = (
                    create_complex_prediction_inputs(
                        protein_input=protein_input,
                        ligand_input=ligand_input,
                    )
                )
                target_path = target_artifact.resolve(input_job.run_dir, must_exist=True)
                compound_path = compound_artifact.resolve(input_job.run_dir, must_exist=True)
                if target_path is None or compound_path is None:
                    raise FileNotFoundError("Typed sequence-ligand inputs were not staged")
                gpu_device = str(gpu).removeprefix("GPU ") if gpu != "Automatic" else "all"
                if engine_label == "Boltz-2":
                    job = queue_boltz2_refolding_job(
                        target_path=target_path,
                        target_artifact=target_artifact,
                        compound_paths=(compound_path,),
                        compound_artifacts=(compound_artifact,),
                        protein_sequences=proteins,
                        image=DEFAULT_BOLTZ2_IMAGE,
                        cache_dir=configured_boltz2_cache_dir(),
                        gpu_device=gpu_device,
                        max_compounds=1,
                        recycling_steps=recycles,
                        sampling_steps=sampling_steps,
                        diffusion_samples=diffusion_samples,
                        use_msa_server=False,
                        use_potentials=use_potentials,
                        msa_paths=(local_msa_path,) if local_msa_path is not None else (),
                        replicates=1,
                        seed_start=seed,
                        launch_context="structure_import",
                    )
                else:
                    job = queue_alphafold3_refolding_job(
                        target_path=target_path,
                        target_artifact=target_artifact,
                        compound_paths=(compound_path,),
                        compound_artifacts=(compound_artifact,),
                        protein_sequences=proteins,
                        image=DEFAULT_ALPHAFOLD3_IMAGE,
                        db_dir=af3_db,
                        weights_dir=af3_weights,
                        msa_repository_dir=af3_msa,
                        gpu_device=gpu_device,
                        max_compounds=1,
                        batch_size=batch_size,
                        num_recycles=recycles,
                        model_seed_count=model_seeds,
                        model_seed_start=model_seed_start,
                        launch_context="structure_import",
                    )
                st.success(
                    f"Queued {engine_label} complex preparation "
                    f"{display_job_code(job.metadata.get('job_code'), job.run_id)}."
                )
            except Exception as exc:
                st.error(str(exc))

    with completed_tab:
        _render_prediction_complex_promotion(
            workflow=workflow,
            engine_label=engine_label,
            key=f"{key}_completed",
        )


def _split_ligand_id_and_smiles(value: str, fallback_ligand_id: str = "LIG") -> tuple[str, str]:
    raw = str(value or "").strip()
    if not raw:
        return fallback_ligand_id, ""
    if "," in raw:
        left, right = raw.split(",", 1)
        return (left.strip() or fallback_ligand_id), right.strip()
    return fallback_ligand_id, raw


RE_VINA = re.compile(r"REMARK VINA RESULT:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
RE_GNINA_MIN_AFF = re.compile(r"REMARK\s+minimizedAffinity\s+(-?\d+(?:\.\d+)?)")
RE_GNINA_CNNSCORE = re.compile(r"REMARK\s+CNNscore\s+(-?\d+(?:\.\d+)?)")
RE_GNINA_CNNAFF = re.compile(r"REMARK\s+CNNaffinity\s+(-?\d+(?:\.\d+)?)")


def _parse_best_vina_score(pdbqt_path: Path) -> float | None:
    if not pdbqt_path.exists():
        return None
    try:
        for line in pdbqt_path.read_text().splitlines():
            m = RE_VINA.search(line)
            if m:
                return float(m.group(1))
    except Exception:
        return None
    return None


def _parse_gnina_scores(pdbqt_path: Path) -> dict[str, float | None]:
    out = {
        "minimized_affinity_kcal_mol": None,
        "cnnscore": None,
        "cnnaffinity": None,
    }
    if not pdbqt_path.exists():
        return out
    try:
        for line in pdbqt_path.read_text().splitlines():
            m1 = RE_GNINA_MIN_AFF.search(line)
            if m1 and out["minimized_affinity_kcal_mol"] is None:
                out["minimized_affinity_kcal_mol"] = float(m1.group(1))
            m2 = RE_GNINA_CNNSCORE.search(line)
            if m2 and out["cnnscore"] is None:
                out["cnnscore"] = float(m2.group(1))
            m3 = RE_GNINA_CNNAFF.search(line)
            if m3 and out["cnnaffinity"] is None:
                out["cnnaffinity"] = float(m3.group(1))
            if all(v is not None for v in out.values()):
                break
    except Exception:
        return out
    return out


def _run_docking_from_prepared_structure(
    *,
    engine: str,
    structure_run_id: str,
    structure_job_code: str,
    pdb_id: str,
    ligand_key: str,
    ligand_id: str,
    protein_pdb: Path,
    ligand_sdf: Path,
    ligand_smiles: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    docking_mode: str,
    search_mode: str,
    exhaustiveness: int,
    use_scrub: bool,
    scrub_ph: float,
    scrub_skip_tautomer: bool,
    extra_udp_args: str = "",
    extra_vina_args: str = "",
    docker_image: str = "avgu-docking-suite-cuda:latest",
) -> dict:
    run_id = str(uuid4())
    # Docking run itself is a structure job folder so downstream MD/FEP can consume it directly.
    run_dir = _run_root() / "structure-jobs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    work_dir = run_dir / "work"
    input_dir = work_dir / "input"
    prep_dir = work_dir / "prepared"
    output_dir = work_dir / "results"
    input_dir.mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_local = input_dir / protein_pdb.name
    protein_local_noh = input_dir / (protein_pdb.stem + "_noh.pdb")
    ligand_local = input_dir / ligand_sdf.name

    # Validate/repair receptor before Meeko. PDBFixer/OpenMM can emit terminal
    # OXT atoms that RDKit/Meeko may overbond by proximity, causing valence
    # errors during `mk_prepare_receptor.py`.
    protein_text = protein_pdb.read_text()
    protein_text_for_meeko, receptor_repair_report = _prepare_receptor_pdb_for_meeko(protein_text)
    protein_local.write_text(protein_text_for_meeko)
    protein_text_noh, _removed_h_records = _remove_all_hydrogens_for_meeko(protein_text_for_meeko)
    protein_local_noh.write_text(protein_text_noh)
    ligand_local.write_bytes(ligand_sdf.read_bytes())

    receptor_repair_report_path = input_dir / "receptor_meeko_repair_report.json"
    receptor_repair_report_path.write_text(json.dumps(receptor_repair_report, indent=2))
    if not receptor_repair_report.get("final_valid"):
        raise RuntimeError(
            "Prepared receptor is still not RDKit/Meeko-safe after automatic repair: "
            + str(receptor_repair_report.get("final_error") or receptor_repair_report.get("original_error") or "unknown error")
        )

    ligand_smiles_path = input_dir / "ligand_input.smi"
    ligand_smiles_path.write_text((ligand_smiles.strip() + "\n") if ligand_smiles.strip() else "\n")

    receptor_pdbqt = prep_dir / "receptor.pdbqt"
    ligand_pdbqt = prep_dir / "ligand.pdbqt"
    config_txt = work_dir / "config.txt"
    ligand_index = work_dir / "ligand_index.txt"
    config_txt.write_text(
        (
            f"center_x = {float(center[0]):.3f}\n"
            f"center_y = {float(center[1]):.3f}\n"
            f"center_z = {float(center[2]):.3f}\n"
            f"size_x = {float(size[0]):.3f}\n"
            f"size_y = {float(size[1]):.3f}\n"
            f"size_z = {float(size[2]):.3f}\n"
        )
    )
    ligand_index.write_text("prepared/ligand.pdbqt\n")
    safe_engine = str(engine or "udp").strip().lower()
    if safe_engine not in {"udp", "vina", "gnina"}:
        safe_engine = "udp"

    safe_search_mode = str(search_mode or "detail").strip().lower()
    if safe_search_mode not in {"fast", "balance", "detail"}:
        safe_search_mode = "detail"
    safe_docking_mode = str(docking_mode or "classic").strip().lower()
    if safe_docking_mode not in {"classic", "hybrid"}:
        safe_docking_mode = "classic"
    safe_scrub_ph = float(scrub_ph)
    safe_scrub_skip_tautomer = bool(scrub_skip_tautomer)
    safe_exhaustiveness = int(exhaustiveness or 30)
    if safe_exhaustiveness < 1:
        safe_exhaustiveness = 1
    udp_extra_cli = str(extra_udp_args or "").strip()
    vina_extra_cli = str(extra_vina_args or "").strip()
    reference_arg = "--reference_ligand prepared/ligand.pdbqt " if (safe_engine == "udp" and safe_docking_mode == "hybrid") else ""
    ligand_source_for_prepare = "input/" + shlex.quote(ligand_local.name)
    ligand_source_mode = "structure_sdf"
    ligand_source_build_cmd = ""
    if ligand_smiles.strip():
        ligand_source_mode = "user_smiles"
        ligand_source_for_prepare = "prepared/ligand_from_smiles.sdf"
        ligand_source_build_cmd = (
            "obabel -ismi input/" + shlex.quote(ligand_smiles_path.name)
            + " -osdf -O prepared/ligand_from_smiles.sdf >/dev/null 2>&1; "
        )

    ligand_prepare_cmd = (
        ligand_source_build_cmd
        + "mk_prepare_ligand.py -i "
        + ligand_source_for_prepare
        + " -o prepared/ligand.pdbqt; "
    )
    if bool(use_scrub):
        scrub_flag = " --skip_tautomer" if safe_scrub_skip_tautomer else ""
        ligand_prepare_cmd = (
            ligand_source_build_cmd
            + "scrub.py " + ligand_source_for_prepare
            + " -o prepared/ligand_scrubbed.sdf"
            + f" --ph {safe_scrub_ph:.2f}"
            + scrub_flag
            + "; "
            + "mk_prepare_ligand.py -i prepared/ligand_scrubbed.sdf -o prepared/ligand.pdbqt; "
        )

    docking_cmd = (
        "udp --receptor prepared/receptor.pdbqt "
        + reference_arg
        + "--ligand_index ligand_index.txt "
        + "--config config.txt "
        + "--dir results "
        + f"--search_mode {safe_search_mode} "
        + (udp_extra_cli + " " if udp_extra_cli else "")
    )
    if safe_engine in {"vina", "gnina"}:
        engine_bin = "gnina" if safe_engine == "gnina" else "vina"
        docking_cmd = (
            f"{engine_bin} "
            + "--receptor prepared/receptor.pdbqt "
            + "--ligand prepared/ligand.pdbqt "
            + "--config config.txt "
            + f"--exhaustiveness {safe_exhaustiveness} "
            + (vina_extra_cli + " " if vina_extra_cli else "")
            + "--out results/ligand_out.pdbqt "
        )

    shell_cmd = (
        "set -euo pipefail; "
        + "cd /workspace/work; "
        + "if ! mk_prepare_receptor.py -i input/" + shlex.quote(protein_local.name) + " -o prepared/receptor -p; then "
        + "echo 'retry_meeko_with_noh' > prepared/receptor_prepare_retry.txt; "
        + "mk_prepare_receptor.py -i input/" + shlex.quote(protein_local_noh.name) + " -o prepared/receptor -p; "
        + "fi; "
        + ligand_prepare_cmd
        + docking_cmd
        + "; "
        + "if [[ -f results/ligand_out.pdbqt ]]; then "
        + "obabel results/ligand_out.pdbqt -O results/ligand_out.pdb >/dev/null 2>&1 || true; "
        + "obabel results/ligand_out.pdbqt -O results/ligand_out.sdf >/dev/null 2>&1 || true; "
        + "fi"
    )
    tool_id = {"udp": "unidock_pro", "gnina": "gnina", "vina": "vina"}[safe_engine]
    command = build_docker_command(
        DockerRunSpec(
            tool=registered_tool(tool_id, image=docker_image),
            command=("bash", "-lc", shell_cmd),
            mounts=(DockerMount(run_dir, "/workspace"),),
            gpu_enabled=safe_engine != "vina",
            use_host_user=False,
        )
    )
    metadata = {
        "run_id": run_id,
        "job_code": _short_job_code(run_id),
        "status": "running",
        "workflow": "DOCKING_REDOCKING",
        "job_type": "structure",
        "source": safe_engine,
        "engine": safe_engine,
        "source_structure_run_id": structure_run_id,
        "source_structure_job_code": structure_job_code,
        "pdb_id": pdb_id,
        "ligand_key": ligand_key,
        "ligand_id": ligand_id,
        "ligand_smiles": ligand_smiles,
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "size": {"x": float(size[0]), "y": float(size[1]), "z": float(size[2])},
        "docking_mode": (safe_docking_mode if safe_engine == "udp" else ""),
        "search_mode": (safe_search_mode if safe_engine == "udp" else ""),
        "exhaustiveness": (safe_exhaustiveness if safe_engine in {"vina", "gnina"} else None),
        "engine_call": {
            "engine": safe_engine,
            "receptor": "prepared/receptor.pdbqt",
            "ligand_input_mode": "ligand_index.txt" if safe_engine == "udp" else "prepared/ligand.pdbqt",
            "ligand_source_mode": ligand_source_mode,
            "ligand_source_input": ligand_source_for_prepare,
            "config": "config.txt",
            "output": "results/ligand_out.pdbqt" if safe_engine in {"vina", "gnina"} else "results/",
            "search_mode": (safe_search_mode if safe_engine == "udp" else None),
            "docking_mode": (safe_docking_mode if safe_engine == "udp" else None),
            "reference_ligand_used": bool(safe_engine == "udp" and safe_docking_mode == "hybrid"),
            "exhaustiveness": (safe_exhaustiveness if safe_engine in {"vina", "gnina"} else None),
            "extra_args": (udp_extra_cli if safe_engine == "udp" else vina_extra_cli),
        },
        "use_scrub": bool(use_scrub),
        "scrub_ph": safe_scrub_ph,
        "scrub_skip_tautomer": safe_scrub_skip_tautomer,
        "extra_udp_args": (udp_extra_cli if safe_engine == "udp" else ""),
        "extra_vina_args": (vina_extra_cli if safe_engine in {"vina", "gnina"} else ""),
        "docker_image": docker_image,
        "receptor_meeko_repair": receptor_repair_report,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    write_registered_command_record(
        run_dir,
        tool_id=tool_id,
        commands=(command,),
        image=docker_image,
    )
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    out_files = sorted(output_dir.rglob("*_out.pdbqt"))
    best_file = out_files[0] if out_files else None
    best_score = _parse_best_vina_score(best_file) if best_file is not None else None
    gnina_scores = _parse_gnina_scores(best_file) if (best_file is not None and safe_engine == "gnina") else {
        "minimized_affinity_kcal_mol": None,
        "cnnscore": None,
        "cnnaffinity": None,
    }
    result = {
        "success": proc.returncode == 0,
        "engine": safe_engine,
        "returncode": int(proc.returncode),
        "stdout_tail": (proc.stdout or "")[-12000:],
        "stderr_tail": (proc.stderr or "")[-12000:],
        "receptor_prepare_retry_used": (prep_dir / "receptor_prepare_retry.txt").exists(),
        "best_pose_pdbqt": str(best_file) if best_file is not None else "",
        "best_score_kcal_mol": best_score,
        "minimized_affinity_kcal_mol": gnina_scores.get("minimized_affinity_kcal_mol"),
        "cnnscore": gnina_scores.get("cnnscore"),
        "cnnaffinity": gnina_scores.get("cnnaffinity"),
        "result_files_count": len(out_files),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    metadata["status"] = "completed" if proc.returncode == 0 else "failed"
    metadata["completed_at"] = _utc_now_iso()
    metadata["updated_at"] = _utc_now_iso()
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"run_id": run_id, "run_dir": str(run_dir), "metadata": metadata, "result": result, "command": command}


def _materialize_docked_structure_outputs(
    *,
    source_structure: dict,
    docking_run: dict,
) -> str | None:
    result = docking_run.get("result", {}) or {}
    if not bool(result.get("success")):
        return None
    dock_run_dir = Path(str(docking_run.get("run_dir") or ""))
    dock_result_dir = dock_run_dir / "work" / "results"
    protein_src = Path(str(source_structure.get("protein_pdb") or ""))
    ligand_sdf_src = dock_result_dir / "ligand_out.sdf"
    ligand_pdb_src = dock_result_dir / "ligand_out.pdb"
    if not protein_src.exists() or not ligand_sdf_src.exists():
        return None

    try:
        protein_text = protein_src.read_text().rstrip() + "\n"
        ligand_text = ligand_pdb_src.read_text() if ligand_pdb_src.exists() else ""
        ligand_lines = [ln for ln in ligand_text.splitlines() if ln.startswith(("ATOM", "HETATM", "CONECT"))]
        merged_complex = protein_text + "\n".join(ligand_lines) + "\nEND\n"
    except Exception:
        merged_complex = protein_src.read_text().rstrip() + "\nEND\n"

    pdb_id = str(source_structure.get("pdb_id") or "dock").lower()
    job_dir = Path(str(docking_run.get("run_dir") or ""))
    if not job_dir.exists():
        return None
    meta_path = job_dir / "metadata.json"
    meta = {}
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        meta = {}
    meta.update(
        {
            "source": str((docking_run.get("metadata") or {}).get("engine") or (docking_run.get("metadata") or {}).get("source") or "udp"),
            "source_structure_run_id": source_structure.get("run_id"),
            "source_structure_job_code": source_structure.get("job_code"),
            "source_docking_run_id": docking_run.get("run_id"),
            "pdb_id": str(source_structure.get("pdb_id") or ""),
            "ligand_key": str(source_structure.get("ligand_key") or "LIG|A|1|_"),
            "status": "completed",
            "ligand_count": 1,
            "updated_at": _utc_now_iso(),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    job_code = str(meta.get("job_code") or _short_job_code(str(job_dir.name)))
    ligand_id = str(meta.get("ligand_id") or source_structure.get("ligand_id") or "lig").strip().lower()
    ligand_id = re.sub(r"[^a-z0-9]+", "-", ligand_id).strip("-") or "lig"
    file_prefix = f"docked_{job_code.lower()}_{ligand_id}"
    protein_dst = job_dir / f"{file_prefix}_protein_refined.pdb"
    complex_dst = job_dir / f"{file_prefix}_complex_refined.pdb"
    protein_dst.write_text(protein_src.read_text())
    complex_dst.write_text(merged_complex)

    # Run the same ligand correction/refinement helper used by PDB preparation,
    # using docked pose geometry (PDB) and the selected/reference SMILES template.
    ligand_pdb_text = ligand_pdb_src.read_text() if ligand_pdb_src.exists() else ""
    reference_smiles = str((docking_run.get("metadata") or {}).get("ligand_smiles") or "").strip()
    artifacts: dict = {}
    if ligand_pdb_text.strip():
        try:
            artifacts = _build_ligand_sdf_artifacts(
                ligand_pdb=ligand_pdb_text,
                ligand_resname="LIG",
                output_dir=job_dir,
                file_prefix=file_prefix,
                reference_smiles=reference_smiles if reference_smiles else None,
            )
        except Exception:
            artifacts = {}

    # Ensure required output files exist even if strict refinement path is unavailable.
    ligand_refined = Path(str(artifacts.get("ligand_refined_sdf") or job_dir / f"{file_prefix}_ligand_refined.sdf"))
    ligand_raw = Path(str(artifacts.get("ligand_raw_sdf") or job_dir / f"{file_prefix}_ligand_raw.sdf"))
    if not ligand_refined.exists():
        ligand_refined.write_text(ligand_sdf_src.read_text())
    if not ligand_raw.exists():
        ligand_raw.write_text(ligand_sdf_src.read_text())

    # Persist reference/template SMILES in result folder for traceability.
    smiles_dst = Path(str(artifacts.get("ligand_ref_smi") or job_dir / f"{file_prefix}_ligand_ref.smi"))
    if reference_smiles:
        smiles_dst.write_text(reference_smiles + "\n")
    write_structure_artifact_manifest(job_dir)
    return job_code


def render() -> None:
    st.title("Structure Import")
    st.caption("Import protein-ligand systems that can be reused by MD, free energy, and property workflows.")

    tabs = st.tabs(
        [
            "From PDB",
            "From docking",
            "From Boltz-2 prediction",
            "From AlphaFold 3 prediction",
            "From custom files",
            "Results",
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
        st.caption(
            "Preparation uses MODELLER for explicit residue-specific amino-acid repair, "
            "followed by containerized PDBFixer/OpenMM cleaning and validation."
        )

        if run:
            try:
                raw_pdb = download_pdb(pdb_id)
                import_job = create_protein_import_job(
                    raw_pdb,
                    filename=f"{pdb_id.lower()}.pdb",
                    source="pdb",
                    pdb_id=pdb_id,
                )
                workflow = create_workflow(
                    "protein-complex-preparation",
                    name=f"{pdb_id} protein-complex preparation",
                    parameters={"source": "pdb", "pdb_id": pdb_id},
                    expected_steps=("protein_import", "protein_cleaning", "complex_preparation"),
                )
                attach_workflow_child(workflow.workflow_id, import_job, step_id="protein_import")
                add_workflow_input(workflow.workflow_id, "protein-import", imported_target(import_job))
                st.session_state["prep_wizard_pdb_id"] = pdb_id
                st.session_state["prep_wizard_raw_pdb_data"] = raw_pdb
                st.session_state["prep_wizard_ligands"] = parse_bound_ligands(raw_pdb)
                st.session_state["prep_wizard_import_run_id"] = import_job.run_id
                st.session_state["prep_wizard_workflow_id"] = workflow.workflow_id
                st.success(
                    f"Downloaded {pdb_id}. Inspect the whole complex, choose chains and ligand, then run preparation."
                )
            except Exception as exc:
                st.error(f"Preparation failed: {exc}")

        raw_pdb = st.session_state.get("prep_wizard_raw_pdb_data")
        ligands = st.session_state.get("prep_wizard_ligands", [])
        active_pdb_id = st.session_state.get("prep_wizard_pdb_id", pdb_id)
        if raw_pdb and ligands:
            if st.button("Start new structure import task", key="prep_start_new_task"):
                for key in [
                    "prep_wizard_pdb_id",
                    "prep_wizard_raw_pdb_data",
                    "prep_wizard_ligands",
                    "prep_wizard_import_run_id",
                    "prep_wizard_workflow_id",
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
            ligand_labels = [f"{lig['resname']} chain {lig['chain']} residue {lig['resseq']}" for lig in selectable_ligands]
            selected_idx = st.radio(
                "Bound ligand",
                options=list(range(len(selectable_ligands))),
                index=0,
                format_func=lambda i: ligand_labels[i],
                key=f"prep_ligand_{active_pdb_id}",
            )
            selected = selectable_ligands[int(selected_idx)]
            ligand_chain = str(selected.get("chain") or "")
            default_chains = (
                [ligand_chain]
                if ligand_chain in chain_ids
                else (chain_ids if chain_ids else [])
            )
            selected_chains = st.multiselect(
                "Protein chain(s) to retain",
                options=chain_ids,
                default=default_chains,
                format_func=lambda c: chain_labels.get(c, c),
                key=f"prep_chains_{active_pdb_id}",
                help=(
                    "Defaults to the protein chain containing the selected ligand. "
                    "Select additional chains only when they are part of the intended "
                    "simulation system."
                ),
            )
            noncanonical_sites = [
                site
                for site in detect_noncanonical_residues(raw_pdb)
                if str(site["chain"]) in selected_chains
            ]
            noncanonical_replacements: list[dict[str, str]] = []
            unresolved_noncanonical: list[str] = []
            if noncanonical_sites:
                st.markdown("#### Noncanonical amino-acid repair")
                st.caption(
                    "Each site is independent. A PDB MODRES annotation supplies a suggested "
                    "default only; review it and override individual sites when the experimental "
                    "chemistry or intended sequence differs."
                )
                st.dataframe(
                    [
                        {
                            "site": site["key"],
                            "component": site["resname"],
                            "chain": site["chain"],
                            "residue": site["resseq"],
                            "suggested": site["suggested_target"] or "review required",
                            "evidence": site["evidence"],
                            "detail": site["evidence_detail"],
                        }
                        for site in noncanonical_sites
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                amino_acids = list(CANONICAL_AMINO_ACIDS)
                columns = st.columns(min(3, len(noncanonical_sites)))
                for index, site in enumerate(noncanonical_sites):
                    suggested = str(site.get("suggested_target") or "")
                    options = [""] + amino_acids
                    selected_target = columns[index % len(columns)].selectbox(
                        f"{site['resname']} {site['chain']}:{site['resseq']}"
                        + (str(site["icode"]) if site["icode"] != "_" else ""),
                        options=options,
                        index=options.index(suggested) if suggested in options else 0,
                        format_func=lambda value: (
                            "Choose canonical amino acid"
                            if not value
                            else f"{value} ({CANONICAL_AMINO_ACIDS[value]})"
                        ),
                        key=f"noncanonical_{active_pdb_id}_{site['key']}",
                        help=(
                            f"Evidence: {site['evidence']}. "
                            f"{site['evidence_detail'] or 'No canonical parent was declared.'}"
                        ),
                    )
                    if selected_target:
                        noncanonical_replacements.append(
                            {"key": str(site["key"]), "target": selected_target}
                        )
                    else:
                        unresolved_noncanonical.append(str(site["key"]))
                if unresolved_noncanonical:
                    st.warning(
                        "Choose a canonical amino acid for every retained noncanonical protein "
                        "site before preparation: " + ", ".join(unresolved_noncanonical)
                    )
            gap_candidates = [
                item
                for item in coordinate_gap_candidates(raw_pdb)
                if str(item["chain"]) in selected_chains
            ]
            internal_gap_definitions: list[dict[str, object]] = []
            unresolved_gaps: list[str] = []
            if gap_candidates:
                st.markdown("#### Internal gap reconstruction")
                st.caption(
                    "MODELLER requires the missing sequence and both observed flanking "
                    "residues. Deposited sequence records are prefilled when available; "
                    "otherwise enter them manually. Every gap must be explicitly confirmed."
                )
                deposited = detect_modeller_internal_gaps(
                    raw_pdb,
                    max_internal_gap=15,
                )
                inferred_by_range = {
                    (
                        str(gap["chain"]),
                        int(gap.get("author_start") or -1),
                        int(gap.get("author_end") or -1),
                    ): gap
                    for chain_data in deposited.values()
                    for gap in chain_data.get("gaps") or []
                }
                for candidate in gap_candidates:
                    gap_key = (
                        str(candidate["chain"]),
                        int(candidate["author_start"]),
                        int(candidate["author_end"]),
                    )
                    inferred = inferred_by_range.get(gap_key, {})
                    label = (
                        f"Chain {candidate['chain']} · missing "
                        f"{candidate['author_start']}–{candidate['author_end']}"
                    )
                    with st.expander(label, expanded=True):
                        cols = st.columns([0.2, 0.6, 0.2])
                        left_code = cols[0].text_input(
                            f"Left flank {candidate['left_resname']} "
                            f"{candidate['left_resseq']}",
                            value=str(candidate["left_code"]),
                            max_chars=1,
                            key=f"gap_left_{active_pdb_id}_{gap_key}",
                        ).strip().upper()
                        sequence = cols[1].text_input(
                            "Missing amino-acid sequence",
                            value=str(inferred.get("sequence") or ""),
                            key=f"gap_sequence_{active_pdb_id}_{gap_key}",
                            help="Canonical one-letter amino-acid sequence; maximum 15 residues.",
                        )
                        sequence = "".join(sequence.split()).upper()
                        right_code = cols[2].text_input(
                            f"Right flank {candidate['right_resname']} "
                            f"{candidate['right_resseq']}",
                            value=str(candidate["right_code"]),
                            max_chars=1,
                            key=f"gap_right_{active_pdb_id}_{gap_key}",
                        ).strip().upper()
                        evidence = str(
                            inferred.get("evidence")
                            or "User-provided sequence; coordinate numbering discontinuity"
                        )
                        st.caption(
                            f"Evidence: {evidence}. Observed coordinate flanks: "
                            f"{candidate['left_code']}{candidate['left_resseq']} / "
                            f"{candidate['right_code']}{candidate['right_resseq']}."
                        )
                        valid_sequence = bool(sequence) and not (
                            set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")
                        )
                        valid_flanks = (
                            left_code == str(candidate["left_code"])
                            and right_code == str(candidate["right_code"])
                        )
                        valid_length = len(sequence) <= 15
                        confirmed = st.checkbox(
                            "I confirm this missing sequence and both flanking residues",
                            value=False,
                            key=f"gap_confirm_{active_pdb_id}_{gap_key}",
                        )
                        if not valid_sequence:
                            st.warning("Enter a canonical missing amino-acid sequence.")
                        elif not valid_length:
                            st.warning(
                                "This gap exceeds the current 15-residue MODELLER limit."
                            )
                        elif not valid_flanks:
                            st.warning(
                                "The entered flanks do not match the observed coordinate residues."
                            )
                        if confirmed and valid_sequence and valid_flanks and valid_length:
                            internal_gap_definitions.append(
                                {
                                    **candidate,
                                    "sequence": sequence,
                                    "left_code": left_code,
                                    "right_code": right_code,
                                    "evidence": evidence + "; user confirmed",
                                    "confirmed": True,
                                }
                            )
                        else:
                            unresolved_gaps.append(label)
                if unresolved_gaps:
                    st.warning(
                        "Confirm every internal gap before preparation: "
                        + ", ".join(unresolved_gaps)
                    )
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

            if st.button(
                "Prepare selected protein and ligand",
                key=f"prep_selected_{active_pdb_id}",
                type="primary",
                disabled=bool(unresolved_noncanonical or unresolved_gaps),
            ):
                try:
                    with st.spinner("Running Ligand-X/HQBind-style protein preparation in container..."):
                        import_run_id = str(st.session_state.get("prep_wizard_import_run_id") or "")
                        if not import_run_id:
                            import_job = create_protein_import_job(
                                raw_pdb,
                                filename=f"{active_pdb_id.lower()}.pdb",
                                source="pdb",
                                pdb_id=active_pdb_id,
                            )
                            import_run_id = import_job.run_id
                            st.session_state["prep_wizard_import_run_id"] = import_run_id
                        else:
                            import_job = load_protein_import_job(import_run_id)
                        workflow_id = str(st.session_state.get("prep_wizard_workflow_id") or "")
                        if not workflow_id:
                            workflow = create_workflow(
                                "protein-complex-preparation",
                                name=f"{active_pdb_id} protein-complex preparation",
                                parameters={"source": "pdb", "pdb_id": active_pdb_id},
                                expected_steps=("protein_import", "protein_cleaning", "complex_preparation"),
                            )
                            workflow_id = workflow.workflow_id
                            st.session_state["prep_wizard_workflow_id"] = workflow_id
                            attach_workflow_child(workflow_id, import_job, step_id="protein_import")
                            add_workflow_input(workflow_id, "protein-import", imported_target(import_job))
                        cleaning_job, prepared_payload = run_protein_cleaning_job(
                            import_run_id,
                            image=prepare_image,
                            use_gpu=prepare_use_gpu,
                            map_modified_residues=False,
                            noncanonical_replacements=noncanonical_replacements,
                            internal_gap_definitions=internal_gap_definitions,
                        )
                        attach_workflow_child(
                            workflow_id,
                            cleaning_job,
                            step_id="protein_cleaning",
                            depends_on=(import_run_id,),
                        )
                    if cleaning_job.status != "completed" or not prepared_payload.get("success"):
                        st.error("Containerized preparation failed; selected complex was not refined.")
                        st.code(str(cleaning_job.run_dir))
                        return

                    prepared_payload["output_dir"] = str(cleaning_job.run_dir)

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
                            "parent_run_id": cleaning_job.run_id,
                            "import_run_id": import_run_id,
                            "cleaning_run_id": cleaning_job.run_id,
                            "workflow_id": workflow_id,
                            "workflow_parent_run_id": workflow_id,
                            "workflow_step_id": "complex_preparation",
                            "pdb_id": active_pdb_id,
                            "ligand_count": len(prepared_ligands),
                            "ligand_key": selected_prepared.get("key"),
                            "protein_chains": selected_chains,
                            "noncanonical_replacements": noncanonical_replacements,
                            "internal_gap_definitions": internal_gap_definitions,
                        }
                    )
                    attach_workflow_child(
                        workflow_id,
                        JobRecord.load(job_dir, task_group="structure-jobs"),
                        step_id="complex_preparation",
                        depends_on=(cleaning_job.run_id,),
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
                        st.markdown("- modeled each reviewed noncanonical amino-acid replacement with MODELLER")
                        st.markdown("- removed dropped atoms, collapsed altloc variants, and reinserted ligands")
                        st.write(
                            {
                                "protein_cleaned": prepared_payload.get("protein_cleaned"),
                                "components": prepared_payload.get("components", {}),
                                "output_dir": prepared_payload.get("output_dir"),
                            }
                        )
                        st.json(mapping_report)
                        st.json(
                            prepared_payload.get("modeller_noncanonical_repair")
                            or noncanonical_replacements
                        )
                        st.json(
                            prepared_payload.get("modeller_internal_gap_repair")
                            or {"modeled_gaps": []}
                        )
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
                            st.markdown("- wrote generated OpenMM files (raw and refined ligand SDF)")
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
                        st.info("No generated OpenMM files (2D preview) were produced for this ligand.")
                    if ligand_artifact_error:
                        st.warning(f"Ligand refinement strict-check warning: {ligand_artifact_error}")
                    if fallback_debug:
                        (job_dir / f"{file_prefix}_ligand_artifact_debug.txt").write_text("\n".join(fallback_debug) + "\n")
                    if not (job_dir / f"{file_prefix}_ligand_raw.sdf").exists():
                        st.error(
                            "Ligand raw/refined SDF were not generated. "
                            f"Check debug file: {(job_dir / f'{file_prefix}_ligand_artifact_debug.txt').name}"
                        )

                    cleaning_report = (
                        cleaning_job.run_dir
                        / "artifacts"
                        / "reports"
                        / "repair_report.json"
                    )
                    if cleaning_report.is_file():
                        (job_dir / "repair_report.json").write_bytes(
                            cleaning_report.read_bytes()
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
                    write_structure_artifact_manifest(job_dir)
                    _complete_structure_job(job_dir)
                    refresh_workflow(workflow_id)
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
        st.markdown("#### Prepared complex from docking")
        st.caption("Run re-/docking directly from an existing prepared structure job (refined protein + refined ligand).")
        prepared_rows = _collect_refined_structure_jobs()
        if not prepared_rows:
            st.info("No compatible prepared structure jobs found yet. Create one first in the `From PDB` tab.")
        else:
            rows_by_run_id = {str(row["run_id"]): row for row in prepared_rows}
            selected_choice = select_target_artifact(
                "Prepared target",
                ("prepared_target", "prepared_receptor"),
                key="prep_docking_source_target",
                show_viewer=False,
                allowed_run_ids=set(rows_by_run_id),
            )
            if selected_choice is None:
                st.info("Select a prepared target with a refined ligand to configure docking.")
                st.stop()
            selected = rows_by_run_id[selected_choice.job.run_id]
            protein_path = Path(str(selected["protein_pdb"]))
            ligand_path = Path(str(selected["ligand_sdf"]))
            complex_path = Path(str(selected["complex_pdb"])) if str(selected.get("complex_pdb") or "").strip() else None
            smi_path = Path(str(selected["ligand_ref_smi"])) if str(selected.get("ligand_ref_smi") or "").strip() else None

            st.caption(f"Protein: `{protein_path.name}`")
            st.caption(f"Ligand: `{ligand_path.name}`")
            inferred_center = _infer_center_from_sdf(ligand_path)
            if inferred_center is None:
                inferred_center = (0.0, 0.0, 0.0)

            ligand_key_value = str(selected.get("ligand_key") or "")
            ligand_id_default = ligand_key_value.split("|", 1)[0].strip() if ligand_key_value else "LIG"
            if smi_path is not None and smi_path.exists():
                default_smiles = _read_text(smi_path).strip().splitlines()[0] if _read_text(smi_path).strip() else ""
            else:
                default_smiles = _sdf_quick_summary(str(ligand_path)).get("smiles", "")

            st.markdown("#### Setup")
            docking_engine = st.selectbox(
                "Docking engine",
                options=["udp", "vina", "gnina"],
                index=0,
                key="prep_docking_engine_selector",
                help="Choose docking backend.",
            )
            smiles_value = st.text_input(
                "Ligand ID, SMILES",
                value=(f"{ligand_id_default}, {default_smiles}" if default_smiles else f"{ligand_id_default}, "),
                key=f"prep_docking_smiles_{selected['run_id']}",
                help="Format: LigandID, SMILES (example: T3, O=C...).",
            )
            ligand_id_value, smiles_only = _split_ligand_id_and_smiles(smiles_value, fallback_ligand_id=ligand_id_default)
            center_cols = st.columns(3)
            center_x = center_cols[0].number_input(
                "center_x",
                value=float(inferred_center[0]),
                step=0.5,
                format="%.3f",
                key=f"prep_docking_center_x_{selected['run_id']}",
            )
            center_y = center_cols[1].number_input(
                "center_y",
                value=float(inferred_center[1]),
                step=0.5,
                format="%.3f",
                key=f"prep_docking_center_y_{selected['run_id']}",
            )
            center_z = center_cols[2].number_input(
                "center_z",
                value=float(inferred_center[2]),
                step=0.5,
                format="%.3f",
                key=f"prep_docking_center_z_{selected['run_id']}",
            )
            size_cols = st.columns(4)
            size_x = size_cols[0].number_input("size_x", value=22.0, min_value=1.0, step=1.0, key=f"prep_docking_size_x_{selected['run_id']}")
            size_y = size_cols[1].number_input("size_y", value=22.0, min_value=1.0, step=1.0, key=f"prep_docking_size_y_{selected['run_id']}")
            size_z = size_cols[2].number_input("size_z", value=22.0, min_value=1.0, step=1.0, key=f"prep_docking_size_z_{selected['run_id']}")
            search_mode = "detail"
            exhaustiveness = 30
            if docking_engine == "udp":
                search_mode = size_cols[3].selectbox(
                    "search_mode",
                    options=["fast", "balance", "detail"],
                    index=2,
                    key=f"prep_docking_search_mode_{selected['run_id']}",
                    help="UDP search mode: fast (quick), balance, detail (most thorough; default).",
                )
            else:
                exhaustiveness = int(
                    size_cols[3].number_input(
                        "exhaustiveness",
                        min_value=1,
                        value=30,
                        step=1,
                        key=f"prep_docking_exhaustiveness_{selected['run_id']}",
                        help="Vina search exhaustiveness (higher = slower/more thorough).",
                    )
                )
            docking_mode = "classic"
            if docking_engine == "udp":
                docking_mode = st.selectbox(
                    "docking_mode",
                    options=["classic", "hybrid"],
                    index=0,
                    key=f"prep_docking_mode_{selected['run_id']}",
                    help="classic: receptor-only UDP command. hybrid: adds --reference_ligand.",
                )
            scrub_cols = st.columns(3)
            use_scrub = scrub_cols[0].checkbox(
                "Use scrub.py",
                value=True,
                key=f"prep_docking_use_scrub_{selected['run_id']}",
                help="Apply scrub.py ligand preprocessing before mk_prepare_ligand.py.",
            )
            scrub_ph = scrub_cols[1].number_input(
                "scrub pH",
                min_value=0.0,
                max_value=14.0,
                value=7.4,
                step=0.1,
                key=f"prep_docking_scrub_ph_{selected['run_id']}",
            )
            scrub_skip_tautomer = scrub_cols[2].checkbox(
                "skip tautomer",
                value=True,
                key=f"prep_docking_scrub_skip_taut_{selected['run_id']}",
            )
            extra_udp_args = ""
            extra_vina_args = ""
            if docking_engine == "udp":
                extra_udp_args = st.text_input(
                    "Additional UDP args (optional)",
                    value="",
                    key=f"prep_docking_extra_args_{selected['run_id']}",
                    help="Advanced: append extra arguments passed directly to `udp`.",
                )
            elif docking_engine in {"vina", "gnina"}:
                extra_vina_args = st.text_input(
                    f"Additional {str(docking_engine).capitalize()} args (optional)",
                    value="",
                    key=f"prep_docking_extra_vina_args_{selected['run_id']}",
                    help=f"Advanced: append extra arguments passed directly to `{docking_engine}`.",
                )

            st.caption(
                f"Inferred ligand COM center: ({inferred_center[0]:.3f}, {inferred_center[1]:.3f}, {inferred_center[2]:.3f})"
            )

            if complex_path is not None and complex_path.exists():
                try:
                    complex_pdb_data = _read_text(complex_path)
                    st.markdown("#### Prepared complex preview")
                    st.caption("Prepared complex used as docking source.")
                    _render_py3dmol_complex_preview(
                        complex_pdb_data,
                        ligand_resname="LIG",
                        ligand_sdf_path=str(ligand_path),
                        center=(float(center_x), float(center_y), float(center_z)),
                        size=(float(size_x), float(size_y), float(size_z)),
                        persist_key=f"prepared-docking:{selected['run_id']}",
                    )
                except Exception as exc:
                    st.warning(f"Preview unavailable: {exc}")

            st.info(
                "Docking launch is centralized in Docking / Cofolding. Select this "
                "prepared target and ligand there, configure the engine, and submit "
                "from its Run tab where live CPU/GPU availability is shown."
            )

    with tabs[2]:
        _render_sequence_complex_prediction(
            engine_label="Boltz-2",
            key="prepare_boltz2",
        )

    with tabs[3]:
        _render_sequence_complex_prediction(
            engine_label="AlphaFold 3",
            key="prepare_af3",
        )

    with tabs[4]:
        st.markdown("#### Custom protein + ligand files")
        st.caption(
            "This tab is only for new local files. Existing imported targets are "
            "selected directly in downstream workflow Target / Input tabs."
        )
        selected_cleaning_job = None
        protein_file = st.file_uploader(
            "Protein file (PDB/mmCIF)",
            type=["pdb", "cif", "mmcif"],
            key="custom_protein",
        )
        ligand_file = st.file_uploader("Ligand file (SDF/MOL2/SMILES TXT)", type=["sdf", "mol2", "smi", "txt"], key="custom_ligand")
        complex_file = st.file_uploader(
            "Optional complete complex PDB",
            type=["pdb"],
            key="custom_complex",
            help=(
                "Include the protein and ligand in one coordinate file to publish a "
                "prepared_complex that can be used by Target Trimming."
            ),
        )
        st.caption(
            "Uploaded structures pass through the same strict Ligand-X/PDBFixer "
            "cleaning used for PDB imports. Supported modified residues are normalized, "
            "missing residues/atoms are rebuilt when the source records permit it, and "
            "ligand coordinates are retained."
        )
        if st.button("Register custom prepared input", key="register_custom"):
            if selected_cleaning_job is not None:
                target_ref = prepared_target(selected_cleaning_job)
                resolved_target = target_ref.resolve(selected_cleaning_job.run_dir, must_exist=True)
                protein_path = str(resolved_target) if resolved_target is not None else None
            else:
                protein_path = _save_upload("custom/protein", protein_file)
            ligand_path = _save_upload("custom/ligand", ligand_file)
            complex_path = _save_upload("custom/complex", complex_file)
            if not protein_path or not ligand_path:
                st.warning("Upload both protein and ligand files first.")
            else:
                cleaning_job = selected_cleaning_job
                cleaned_complex_data = ""
                repair_report_source = None
                if complex_path or selected_cleaning_job is None:
                    cleaning_source = Path(complex_path or protein_path)
                    try:
                        import_job = create_protein_import_job(
                            cleaning_source.read_text(errors="replace"),
                            filename=cleaning_source.name,
                            source="manual_complex" if complex_path else "manual_protein",
                        )
                        cleaning_job, cleaned_payload = run_protein_cleaning_job(
                            import_job.run_id,
                            image=DEFAULT_MD_IMAGE,
                            use_gpu=False,
                            map_modified_residues=True,
                        )
                    except Exception as exc:
                        st.error(f"Structure cleaning could not start: {exc}")
                        return
                    if cleaning_job.status != "completed" or not cleaned_payload.get("success"):
                        st.error(
                            str(
                                cleaned_payload.get("error")
                                or cleaning_job.result.get("error")
                                or "Structure cleaning failed"
                            )
                        )
                        return
                    cleaned_complex_data = (
                        str(cleaned_payload.get("prepared_pdb_data") or "")
                        if complex_path
                        else ""
                    )
                    target_ref = prepared_target(cleaning_job)
                    resolved_target = target_ref.resolve(cleaning_job.run_dir, must_exist=True)
                    if resolved_target is None:
                        st.error("Structure cleaning did not publish a prepared target.")
                        return
                    protein_path = str(resolved_target)
                if cleaning_job is not None and cleaning_job.artifact_manifest is not None:
                    report_refs = cleaning_job.artifact_manifest.by_type("repair_report")
                    if report_refs:
                        repair_report_source = report_refs[0].resolve(
                            cleaning_job.run_dir, must_exist=True
                        )
                st.session_state["prepared_structure_last"] = {
                    "source": "custom",
                    "protein_path": protein_path,
                    "ligand_path": ligand_path,
                }
                protein_source = Path(protein_path)
                ligand_source = Path(ligand_path)
                protein_name = f"custom_target{protein_source.suffix.lower()}"
                ligand_name = f"custom_ligand{ligand_source.suffix.lower()}"
                job_dir = _write_structure_job(
                    {
                        "source": "custom",
                        "parent_run_id": cleaning_job.run_id if cleaning_job is not None else "",
                        "cleaning_run_id": cleaning_job.run_id if cleaning_job is not None else "",
                        "protein_path": protein_name,
                        "ligand_path": ligand_name,
                        "complex_path": "custom_complex_refined.pdb" if complex_path else "",
                        "clean_and_repair": True,
                    }
                )
                protein_copy = job_dir / protein_name
                ligand_copy = job_dir / ligand_name
                protein_copy.write_bytes(protein_source.read_bytes())
                ligand_copy.write_bytes(ligand_source.read_bytes())
                manifest_artifacts = [
                    ArtifactRef.from_path(
                        job_dir,
                        protein_copy,
                        "prepared_receptor",
                        role="receptor",
                    ),
                    ArtifactRef.from_path(
                        job_dir,
                        ligand_copy,
                        "prepared_ligand_set",
                        role="ligand",
                    ),
                ]
                if complex_path:
                    complex_copy = job_dir / "custom_complex_refined.pdb"
                    if cleaned_complex_data.strip():
                        complex_copy.write_text(cleaned_complex_data)
                    else:
                        complex_copy.write_bytes(Path(complex_path).read_bytes())
                    manifest_artifacts.append(
                        ArtifactRef.from_path(
                            job_dir,
                            complex_copy,
                            "prepared_complex",
                            role="complex",
                        )
                    )
                if repair_report_source is not None:
                    report_copy = job_dir / "repair_report.json"
                    report_copy.write_bytes(repair_report_source.read_bytes())
                    manifest_artifacts.append(
                        ArtifactRef.from_path(
                            job_dir,
                            report_copy,
                            "repair_report",
                            role="report",
                            metadata={
                                "cleaning_run_id": cleaning_job.run_id
                                if cleaning_job is not None
                                else ""
                            },
                        )
                    )
                write_artifact_manifest(
                    job_dir,
                    manifest_artifacts,
                )
                _complete_structure_job(job_dir)
                st.success("Custom input registered.")
                st.code(f"Protein: {protein_path}\nLigand:  {ligand_path}")

    with tabs[5]:
        _render_structure_import_results()

    st.divider()
    st.caption(
        "Prepared artifacts are selectable from the Target / Input tabs of MD, "
        "Docking / Cofolding, Free Energy, ADMET, and QC. Use the sidebar to open "
        "the desired workflow; launch actions are located in its Run tab."
    )


render()
