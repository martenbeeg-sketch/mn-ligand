from __future__ import annotations

import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import streamlit as st
import streamlit.components.v1 as components

from mn_ligand.app.pages.common import _input_root
from mn_ligand.app.pages.bound_ligand_md import (
    DEFAULT_MD_IMAGE,
    _parse_protein_chains,
    _prepare_structure_with_ligandx,
    _render_ligand_summary,
    _render_structure_view,
    _render_workflow_selection,
    _run_root,
    _short_job_code,
)
from mn_ligand.app.pages.boltz2_ui import render_boltz2_inline

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


def _render_py3dmol_complex_preview(
    pdb_data: str,
    ligand_resname: str = "LIG",
    ligand_sdf_path: str | None = None,
    center: tuple[float, float, float] | None = None,
    size: tuple[float, float, float] | None = None,
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
    components.html(view._make_html(), height=580, scrolling=False)


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
        "repair_applied": False,
    }

    original = pdb_data if pdb_data.endswith("\n") else pdb_data + "\n"
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
        protein_candidates = sorted(run_dir.glob("*_protein_refined.pdb"))
        ligand_candidates = sorted(run_dir.glob("*_ligand_refined.sdf"))
        complex_candidates = sorted(run_dir.glob("*_complex_refined.pdb"))
        smiles_candidates = sorted(run_dir.glob("*_ligand_ref.smi"))
        if not protein_candidates or not ligand_candidates:
            continue
        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": metadata.get("job_code") or _short_job_code(run_dir.name),
                "pdb_id": metadata.get("pdb_id") or "",
                "ligand_key": metadata.get("ligand_key") or "",
                "protein_pdb": str(protein_candidates[0]),
                "ligand_sdf": str(ligand_candidates[0]),
                "complex_pdb": str(complex_candidates[0]) if complex_candidates else "",
                "ligand_ref_smi": str(smiles_candidates[0]) if smiles_candidates else "",
                "source": metadata.get("source") or "",
                "created_at": metadata.get("created_at") or "",
            }
        )
    return rows


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
    ligand_local = input_dir / ligand_sdf.name

    # Validate/repair receptor before Meeko. PDBFixer/OpenMM can emit terminal
    # OXT atoms that RDKit/Meeko may overbond by proximity, causing valence
    # errors during `mk_prepare_receptor.py`.
    protein_text = protein_pdb.read_text()
    protein_text_for_meeko, receptor_repair_report = _prepare_receptor_pdb_for_meeko(protein_text)
    protein_local.write_text(protein_text_for_meeko)
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
    ligand_prepare_cmd = (
        "mk_prepare_ligand.py -i input/" + shlex.quote(ligand_local.name) + " -o prepared/ligand.pdbqt; "
    )
    if bool(use_scrub):
        scrub_flag = " --skip_tautomer" if safe_scrub_skip_tautomer else ""
        ligand_prepare_cmd = (
            "scrub.py input/" + shlex.quote(ligand_local.name)
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
        + "mk_prepare_receptor.py -i input/" + shlex.quote(protein_local.name) + " -o prepared/receptor -p; "
        + ligand_prepare_cmd
        + docking_cmd
        + "; "
        + "if [[ -f results/ligand_out.pdbqt ]]; then "
        + "obabel results/ligand_out.pdbqt -O results/ligand_out.pdb >/dev/null 2>&1 || true; "
        + "obabel results/ligand_out.pdbqt -O results/ligand_out.sdf >/dev/null 2>&1 || true; "
        + "fi"
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{run_dir}:/workspace",
        docker_image,
        "bash",
        "-lc",
        shell_cmd,
    ]
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
    return job_code


def render() -> None:
    st.title("Structure Preparation")
    st.caption("Prepare protein-ligand systems that can be reused by MD, free energy, and property workflows.")

    tabs = st.tabs(
        [
            "From PDB",
            "From docking",
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
        st.markdown("#### Prepared complex from docking")
        st.caption("Run re-/docking directly from an existing prepared structure job (refined protein + refined ligand).")
        prepared_rows = _collect_refined_structure_jobs()
        if not prepared_rows:
            st.info("No compatible prepared structure jobs found yet. Create one first in the `From PDB` tab.")
        else:
            options: dict[str, dict] = {}
            for row in prepared_rows:
                label = f"{row['job_code']} | {row['pdb_id'] or 'PDB?'} | {Path(row['ligand_sdf']).name}"
                options[label] = row
            selected_label = st.selectbox(
                "Prepared structure job",
                list(options.keys()),
                key="prep_docking_source_job",
            )
            selected = options[selected_label]
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
                    )
                except Exception as exc:
                    st.warning(f"Preview unavailable: {exc}")

            if st.button("Run docking from this prepared structure", type="primary", key=f"prep_docking_run_{selected['run_id']}"):
                try:
                    with st.spinner(f"Running {docking_engine.upper()} redocking from prepared structure..."):
                        run = _run_docking_from_prepared_structure(
                            engine=str(docking_engine),
                            structure_run_id=str(selected["run_id"]),
                            structure_job_code=str(selected["job_code"]),
                            pdb_id=str(selected.get("pdb_id") or ""),
                            ligand_key=str(selected.get("ligand_key") or ""),
                            ligand_id=str(ligand_id_value),
                            protein_pdb=protein_path,
                            ligand_sdf=ligand_path,
                            ligand_smiles=smiles_only,
                            center=(float(center_x), float(center_y), float(center_z)),
                            size=(float(size_x), float(size_y), float(size_z)),
                            docking_mode=str(docking_mode),
                            search_mode=str(search_mode),
                            exhaustiveness=int(exhaustiveness),
                            use_scrub=bool(use_scrub),
                            scrub_ph=float(scrub_ph),
                            scrub_skip_tautomer=bool(scrub_skip_tautomer),
                            extra_udp_args=str(extra_udp_args),
                            extra_vina_args=str(extra_vina_args),
                            docker_image="avgu-docking-suite-cuda:latest",
                        )
                    result = run.get("result", {})
                    if bool(result.get("success")):
                        st.success(f"{str(docking_engine).upper()} redocking completed: {run.get('metadata', {}).get('job_code', '')}")
                        registered_code = _materialize_docked_structure_outputs(
                            source_structure=selected,
                            docking_run=run,
                        )
                        if registered_code:
                            st.caption(
                                f"Docked pose saved in structure job `{registered_code}` "
                                "for downstream MD/FEP."
                            )
                        st.caption("Open this structure run from Jobs – Structure to inspect docking results.")
                    else:
                        st.error(f"{str(docking_engine).upper()} redocking failed.")
                    st.code(f"Docking run directory:\n{run.get('run_dir', '')}")
                    st.code(f"Docking pose outputs:\n{Path(str(run.get('run_dir', ''))) / 'work' / 'results'}")
                    st.caption(f"Ligand ID used: `{ligand_id_value}`")
                    with st.expander("Docking stderr tail", expanded=not bool(result.get("success"))):
                        st.code(str(result.get("stderr_tail") or ""))
                except Exception as exc:
                    st.error(f"Docking execution failed: {exc}")

    with tabs[2]:
        st.markdown("#### Boltz output -> Prepared complex")
        st.caption("Use Boltz-2 predicted structures and normalize them for downstream workflows.")
        render_boltz2_inline()

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
