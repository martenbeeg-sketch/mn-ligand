from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

from ovo_ligand.app.pages.bound_ligand_md import _short_job_code, _run_root
from ovo_ligand.app.pages.common import WORKFLOWS, _input_root, _run_docker_workflow, try_dispatch_next_queued_gpu_job
from ovo_ligand.workflows.bound_ligand_md import parse_bound_ligands, extract_ligand_pdb, _build_ligand_sdf_artifacts

EXAMPLE_PROTEIN = """>THRbeta_human
HKPEPTDEEWELIKTVTEAHVATNAQGSHWKQKRKFLPEDIGQAPIVNAPEGGKVDLEAFSHFTKIITPAITRVVDFAKKLPMFCELPCEDQIILLKGCCMEIMSLRAAVRYDPESETLTLNGEMAVTRGQLKNGGLGVVSDAIFDLGMSLSSFNLDDTEVALLQAVLLMSSDRPGLACVERIEKYQDSFLLAFEHYINYRKHHVTHFWPKLLMKVTDLRMIGACHASRFLHMKVECPTELFPPLFLEVFED"""
EXAMPLE_LIGAND = "T3,OC1=C(I)C=C(OC2=C(I)C=C(C[C@H](N)C(O)=O)C=C2I)C=C1"
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
BOLTZ2_UI_SCHEMA_VERSION = "2026-05-12-v2"
DEFAULT_BOLTZ_CACHE_DIR = "/mnt/db/reference_files/boltz_models"
DEFAULT_MSA_REPOSITORY_DIR = DEFAULT_BOLTZ_CACHE_DIR + "/msa_repository"
POLYMER_RESN = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","A","C","G","U","T","DA","DC","DG","DT","DU"}




def _switch_to(page: str) -> None:
    try:
        st.switch_page(page)
    except Exception:
        st.info(f"Continue in: `{page}`")

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> None:
    schema_key = "boltz2_ui_schema_version"
    if st.session_state.get(schema_key) != BOLTZ2_UI_SCHEMA_VERSION:
        # Force-reset persisted old defaults/format after UI contract change.
        st.session_state["boltz2_protein"] = ""
        st.session_state["boltz2_ligand"] = ""
        st.session_state[schema_key] = BOLTZ2_UI_SCHEMA_VERSION

    if "boltz2_protein" not in st.session_state:
        st.session_state["boltz2_protein"] = ""
    if "boltz2_ligand" not in st.session_state:
        st.session_state["boltz2_ligand"] = ""
    if "boltz2_enable_affinity" not in st.session_state:
        st.session_state["boltz2_enable_affinity"] = True
    if "boltz2_use_msa_server" not in st.session_state:
        st.session_state["boltz2_use_msa_server"] = True
    if "boltz2_use_potentials" not in st.session_state:
        st.session_state["boltz2_use_potentials"] = True
    if "boltz2_sampling_steps" not in st.session_state:
        st.session_state["boltz2_sampling_steps"] = 200
    if "boltz2_recycling_steps" not in st.session_state:
        st.session_state["boltz2_recycling_steps"] = 3
    if "boltz2_diffusion_samples" not in st.session_state:
        st.session_state["boltz2_diffusion_samples"] = 1
    if "boltz2_sampling_steps_affinity" not in st.session_state:
        st.session_state["boltz2_sampling_steps_affinity"] = 200
    if "boltz2_diffusion_samples_affinity" not in st.session_state:
        st.session_state["boltz2_diffusion_samples_affinity"] = 5
    if "boltz2_affinity_mw_correction" not in st.session_state:
        st.session_state["boltz2_affinity_mw_correction"] = False
    if "boltz2_cache_dir" not in st.session_state:
        st.session_state["boltz2_cache_dir"] = DEFAULT_BOLTZ_CACHE_DIR
    if "boltz2_msa_repository_dir" not in st.session_state:
        st.session_state["boltz2_msa_repository_dir"] = DEFAULT_MSA_REPOSITORY_DIR

    # Backward compatibility: convert legacy "LIG <smiles>" to comma style.
    raw_lig = str(st.session_state.get("boltz2_ligand", "") or "").strip()
    if raw_lig and "," not in raw_lig and " " in raw_lig:
        lid, rest = raw_lig.split(" ", 1)
        if lid.strip() and rest.strip():
            st.session_state["boltz2_ligand"] = f"{lid.strip()},{rest.strip()}"


def _load_example() -> None:
    st.session_state["boltz2_protein"] = EXAMPLE_PROTEIN
    st.session_state["boltz2_ligand"] = EXAMPLE_LIGAND


def _parse_fasta(text: str) -> tuple[str, str]:
    lines = text.strip().splitlines()
    header = ""
    seq_parts: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].strip()
        else:
            seq_parts.append("".join(ch for ch in line if ch.isalpha()).upper())
    sequence = "".join(seq_parts)
    if not header and not any(line.startswith(">") for line in lines):
        header = "protein"
        sequence = "".join(ch for ch in text if ch.isalpha()).upper()
    return header[:30] or "protein", sequence


def _validate_protein(sequence: str) -> tuple[bool, str]:
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    invalid = set(seq) - VALID_AMINO_ACIDS
    if invalid:
        return False, f"Invalid amino acids: {', '.join(sorted(invalid))}"
    if len(seq) < 10:
        return False, "Invalid FASTA/protein sequence: minimum length is 10 residues."
    if len(seq) > 2500:
        return False, f"Protein sequence too long ({len(seq)} aa). Maximum is 2500."
    return True, seq




def _safe_entity_id(value: str, fallback: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in (value or "").strip())
    token = token.strip("_")
    if not token:
        token = fallback
    if token[0].isdigit():
        token = f"{fallback}_{token}"
    return token[:32]


def _split_ligand_id_and_smiles(value: str, fallback_ligand_id: str = "LIG") -> tuple[str, str, str]:
    raw = (value or "").strip()
    if not raw:
        return fallback_ligand_id, "", "Ligand input is required in format: LIGAND_ID,SMILES"
    if "," not in raw:
        return fallback_ligand_id, "", "Use comma-separated format: LIGAND_ID,SMILES"
    ligand_raw, smiles_raw = raw.split(",", 1)
    ligand_id = _safe_entity_id(ligand_raw, fallback_ligand_id)
    smiles = smiles_raw.strip()
    if not smiles:
        return ligand_id, "", "SMILES part is empty. Use: LIGAND_ID,SMILES"
    return ligand_id, smiles, ""


def _validate_smiles(smiles: str) -> tuple[bool, str]:
    s = (smiles or "").strip()
    if not s:
        return False, "Ligand SMILES is required."
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz[]()=#@+-.0123456789\\/:,")
    invalid = set(s) - allowed
    if invalid:
        return False, f"Invalid SMILES characters: {', '.join(sorted(invalid))}"
    return True, s


def _build_input_yaml(protein_id: str, sequence: str, ligand_id: str, ligand_smiles: str, enable_affinity: bool) -> str:
    lines = [
        "version: 1",
        "sequences:",
        "  - protein:",
        '      id: "A"',
        f"      sequence: {sequence}",
        "  - ligand:",
        '      id: "B"',
        f'      smiles: "{ligand_smiles}"',
    ]
    if enable_affinity:
        lines.extend(
            [
                "properties:",
                "  - affinity:",
                '      binder: "B"',
            ]
        )
    return "\n".join(lines) + "\n"


def _find_structure_path(base_dir: Path) -> Path | None:
    patterns = [
        "boltz_results_*/predictions/**/*.pdb",
        "predictions/**/*.pdb",
        "boltz_results_*/predictions/**/*.cif",
        "predictions/**/*.cif",
    ]
    for pattern in patterns:
        matches = sorted(base_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _as_pdb_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return path.read_text()
    if suffix == ".cif":
        try:
            import mdtraj as md

            traj = md.load(str(path))
            tmp = path.with_suffix(".converted.pdb")
            traj.save(str(tmp))
            try:
                return tmp.read_text()
            finally:
                tmp.unlink(missing_ok=True)
        except Exception:
            return ""
    return ""


def _protein_only_pdb(pdb_data: str) -> str:
    lines: list[str] = []
    for line in pdb_data.splitlines():
        if line.startswith("ATOM") and len(line) >= 20:
            resn = line[17:20].strip().upper()
            if resn in POLYMER_RESN:
                lines.append(line)
        elif line.startswith(("TER", "MODEL", "ENDMDL", "CRYST1", "HEADER", "TITLE", "REMARK")):
            lines.append(line)
    return "\n".join(lines + ["END", ""])


def _pick_ligand_for_boltz(pdb_data: str, preferred_resname: str) -> tuple[str, str]:
    ligands = parse_bound_ligands(pdb_data)
    pref = (preferred_resname or "").strip().upper()
    if pref:
        for lig in ligands:
            if str(lig.get("resname") or "").strip().upper() == pref:
                return str(lig.get("key") or ""), str(lig.get("resname") or pref)
    if ligands:
        return str(ligands[0].get("key") or ""), str(ligands[0].get("resname") or "LIG")
    return "", pref or "LIG"


def _extract_selected_complex_pdb(pdb_data: str, selected_ligand_key: str) -> str:
    selected = set()
    if selected_ligand_key:
        parts = selected_ligand_key.split("|")
        if len(parts) >= 4:
            selected = {(parts[1], parts[2], parts[3])}
    lines: list[str] = []
    for line in pdb_data.splitlines():
        if line.startswith("ATOM"):
            lines.append(line)
            continue
        if line.startswith("HETATM") and len(line) >= 27:
            key = (
                line[17:20].strip(),
                line[21].strip() or "_",
                line[22:26].strip(),
            )
            if not selected or key in selected:
                lines.append(line)
            continue
        if line.startswith(("TER", "MODEL", "ENDMDL", "CRYST1", "HEADER", "TITLE", "REMARK")):
            lines.append(line)
    return "\n".join(lines + ["END", ""])




def _extract_first_nonpolymer_pdb(pdb_data: str) -> tuple[str, str, str]:
    # Fallback for Boltz outputs where ligand is encoded as ATOM records.
    groups: dict[tuple[str, str, str], list[str]] = {}
    for line in pdb_data.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
            continue
        resn = line[17:20].strip().upper()
        if not resn or resn in {"HOH", "WAT"}:
            continue
        chain = line[21].strip() or "_"
        resid = line[22:26].strip()
        # protein-like residues are all on chain A in these runs; keep explicit non-polymer preference
        if resn in {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","A","C","G","U","T","DA","DC","DG","DT","DU"}:
            continue
        key = (resn, chain, resid)
        groups.setdefault(key, []).append(line)
    if not groups:
        return "", "", ""
    (resn, chain, resid), lines = max(groups.items(), key=lambda kv: len(kv[1]))
    return "\n".join(lines + ["END", ""]), f"{resn}|{chain}|{resid}|_", resn


def _extract_ligand_pdb_any_record(pdb_data: str, ligand_key: str, ligand_resname: str) -> str:
    # Boltz outputs may encode ligands as ATOM instead of HETATM.
    resn = (ligand_resname or "").strip().upper()
    chain = ""
    resid = ""
    if ligand_key:
        parts = ligand_key.split("|")
        if len(parts) >= 3:
            resn = parts[0].strip().upper() or resn
            chain = parts[1].strip()
            resid = parts[2].strip()
    lines: list[str] = []
    for line in pdb_data.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
            continue
        l_resn = line[17:20].strip().upper()
        l_chain = line[21].strip() or "_"
        l_resid = line[22:26].strip()
        if resn and l_resn != resn:
            continue
        if chain and l_chain != chain:
            continue
        if resid and l_resid != resid:
            continue
        lines.append(line)
    if not lines:
        return ""
    return "\n".join(lines + ["END", ""])

def _extract_ligand_pdb_block(pdb_data: str) -> str:
    ligand_lines: list[str] = []
    for line in pdb_data.splitlines():
        if not line.startswith("HETATM") or len(line) < 20:
            continue
        resn = line[17:20].strip().upper()
        if resn in {"HOH", "WAT"}:
            continue
        ligand_lines.append(line)
    if not ligand_lines:
        return ""
    return "\n".join(ligand_lines + ["END", ""])


def _write_ligand_sdf(ligand_pdb: str, ligand_smiles: str, out_path: Path) -> bool:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        if ligand_pdb.strip():
            mol = Chem.MolFromPDBBlock(ligand_pdb, removeHs=False, sanitize=False, proximityBonding=True)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
                with Chem.SDWriter(str(out_path)) as writer:
                    writer.write(mol)
                return True

        mol_from_smiles = Chem.MolFromSmiles(ligand_smiles)
        if mol_from_smiles is None:
            return False
        mol_from_smiles = Chem.AddHs(mol_from_smiles)
        if AllChem.EmbedMolecule(mol_from_smiles, randomSeed=0xF00D) != 0:
            return False
        try:
            AllChem.UFFOptimizeMolecule(mol_from_smiles, maxIters=200)
        except Exception:
            pass
        with Chem.SDWriter(str(out_path)) as writer:
            writer.write(mol_from_smiles)
        return True
    except Exception:
        return False




def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _create_structure_job_stub(*, protein_id: str, protein_sequence: str, ligand_id: str, ligand_smiles: str) -> Path:
    structure_run_id = str(uuid4())
    structure_dir = _run_root() / "structure-jobs" / structure_run_id
    structure_dir.mkdir(parents=True, exist_ok=False)
    now_iso = _utc_now_iso()
    metadata = {
        "run_id": structure_run_id,
        "job_code": _short_job_code(structure_run_id),
        "job_type": "structure",
        "status": "running",
        "created_at": now_iso,
        "updated_at": now_iso,
        "source": "boltz",
        "source_workflow": "boltz2",
        "protein_id": protein_id,
        "ligand_id": ligand_id,
        "ligand_key": f"{ligand_id}|{protein_id}|1",
        "ligand_count": 1,
        "protein_sequence_length": len(protein_sequence),
        "ligand_smiles": ligand_smiles,
    }
    _write_json(structure_dir / "metadata.json", metadata)
    return structure_dir


def _update_structure_job_status(structure_dir: Path, *, status: str, boltz_run_id: str = "", boltz_job_code: str = "", message: str = "") -> None:
    meta_path = structure_dir / "metadata.json"
    meta = _read_json(meta_path)
    meta["status"] = status
    meta["updated_at"] = _utc_now_iso()
    if status in {"completed", "failed"}:
        meta["completed_at"] = meta.get("completed_at") or _utc_now_iso()
    if boltz_run_id:
        meta["boltz_run_id"] = boltz_run_id
    if boltz_job_code:
        meta["boltz_job_code"] = boltz_job_code
    if message:
        meta["status_message"] = message
    _write_json(meta_path, meta)


def _register_structure_job_from_boltz(
    structure_dir: Path,
    boltz_run_id: str,
    boltz_job_code: str,
    protein_id: str,
    protein_sequence: str,
    ligand_id: str,
    ligand_smiles: str,
    boltz_output_dir: str = "",
) -> tuple[Path | None, str]:
    boltz_run_dir = Path(str(boltz_output_dir)).resolve() if str(boltz_output_dir).strip() else (_run_root() / "structure-jobs" / structure_dir.name / "boltz2")
    if not boltz_run_dir.exists():
        boltz_run_dir = _run_root() / "boltz2" / boltz_run_id
    structure_path = _find_structure_path(boltz_run_dir)
    if structure_path is None:
        return None, "No predicted structure file found under Boltz output directory."

    pdb_data = _as_pdb_text(structure_path)
    if not pdb_data.strip():
        return None, "Could not parse predicted structure into PDB text (need PDB or readable CIF)."

    safe_protein_id = _safe_entity_id(protein_id, "PROTEIN").lower()
    safe_ligand_id = _safe_entity_id(ligand_id, "LIG").lower()
    file_prefix = f"{safe_protein_id}_{safe_ligand_id}"

    ligand_key, ligand_resname = _pick_ligand_for_boltz(pdb_data, ligand_id)
    selected_complex = _extract_selected_complex_pdb(pdb_data, ligand_key)

    complex_path = structure_dir / f"{file_prefix}_complex_refined.pdb"
    protein_path = structure_dir / f"{safe_protein_id}_protein_refined.pdb"
    complex_path.write_text(selected_complex if selected_complex.endswith("\n") else selected_complex + "\n")
    protein_path.write_text(_protein_only_pdb(selected_complex))

    ligand_pdb = extract_ligand_pdb(selected_complex, ligand_key) if ligand_key else ""
    if not ligand_pdb.strip():
        ligand_pdb = _extract_ligand_pdb_any_record(selected_complex, ligand_key, ligand_resname)
    if not ligand_pdb.strip():
        ligand_pdb, auto_key, auto_resn = _extract_first_nonpolymer_pdb(selected_complex)
        if ligand_pdb.strip():
            ligand_key = auto_key or ligand_key
            ligand_resname = auto_resn or ligand_resname
    if not ligand_pdb.strip():
        return None, "Could not extract ligand coordinates from Boltz predicted structure."

    artifacts = _build_ligand_sdf_artifacts(
        ligand_pdb=ligand_pdb,
        ligand_resname=ligand_resname or "LIG",
        output_dir=structure_dir,
        file_prefix=file_prefix,
        reference_smiles=ligand_smiles,
    )
    ligand_path = Path(str(artifacts.get("ligand_refined_sdf") or ""))
    if not ligand_path.exists():
        return None, "Failed to generate refined ligand SDF from Boltz predicted structure."

    meta_path = structure_dir / "metadata.json"
    metadata = _read_json(meta_path)
    metadata.update(
        {
            "status": "completed",
            "updated_at": _utc_now_iso(),
            "completed_at": metadata.get("completed_at") or _utc_now_iso(),
            "source": "boltz",
            "source_workflow": "boltz2",
            "boltz_run_id": boltz_run_id,
            "boltz_job_code": boltz_job_code,
            "protein_id": protein_id,
            "ligand_id": ligand_id,
            "ligand_key": ligand_key or f"{ligand_id}|{protein_id}|1",
            "ligand_resname": ligand_resname,
            "ligand_count": 1,
            "protein_sequence_length": len(protein_sequence),
            "ligand_smiles": ligand_smiles,
            "boltz_structure_path": str(structure_path),
            "ligand_artifact_source": "hiqbind_style",
        }
    )
    _write_json(meta_path, metadata)
    return structure_dir, ""


def render_boltz2_ui(*, show_page_title: bool = True) -> None:
    try_dispatch_next_queued_gpu_job()
    _default_state()

    if show_page_title:
        st.title("Ligand Boltz-2 prediction")
        st.caption("Two-entity mode by default: one protein + one ligand. Uses the same built-in example as boltz2-app-local.")

    st.markdown("#### Boltz Settings")
    with st.expander("Settings", expanded=False):
        st.text_input(
            "Cache directory",
            key="boltz2_cache_dir",
            help="Host path with Boltz reference/model cache. Mounted to /cache.",
        )
        st.text_input(
            "MSA repository directory",
            key="boltz2_msa_repository_dir",
            help="Host path for MSA repository. Mounted to /msa_repository.",
        )
        s1, s2, s3 = st.columns(3)
        with s1:
            st.checkbox("Use MSA server", key="boltz2_use_msa_server")
            st.checkbox("Use potentials", key="boltz2_use_potentials")
            st.checkbox("Enable affinity", key="boltz2_enable_affinity")
            st.checkbox("Affinity molecular-weight correction", key="boltz2_affinity_mw_correction")
        with s2:
            st.number_input("Sampling steps", min_value=10, max_value=400, step=10, key="boltz2_sampling_steps")
            st.number_input("Recycling steps", min_value=1, max_value=12, step=1, key="boltz2_recycling_steps")
        with s3:
            st.number_input("Diffusion samples", min_value=1, max_value=16, step=1, key="boltz2_diffusion_samples")
            st.number_input("Affinity sampling steps", min_value=10, max_value=400, step=10, key="boltz2_sampling_steps_affinity")
            st.number_input("Affinity diffusion samples", min_value=1, max_value=16, step=1, key="boltz2_diffusion_samples_affinity")

    st.markdown("#### Entities")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Entity 1: protein")
        st.text_area("Protein sequence (FASTA or raw)", key="boltz2_protein", height=220)
    with c2:
        st.caption("Entity 2: ligand")
        st.text_area("Ligand input (LIGAND_ID,SMILES)", key="boltz2_ligand", height=220)
        st.caption("Example format: T3,CCO")

    p1, p2 = st.columns([1, 3])
    with p1:
        st.button("Load example", on_click=_load_example, use_container_width=True)

    protein_header, protein_sequence = _parse_fasta(st.session_state.get("boltz2_protein", ""))
    protein_id = _safe_entity_id(protein_header, "PROTEIN")
    ligand_id, ligand_smiles_raw, ligand_input_msg = _split_ligand_id_and_smiles(st.session_state.get("boltz2_ligand", ""), "LIG")
    protein_ok, protein_msg = _validate_protein(protein_sequence)
    smiles_ok, smiles_or_msg = _validate_smiles(ligand_smiles_raw) if not ligand_input_msg else (False, "")

    if not protein_ok:
        st.warning(protein_msg)
    if ligand_input_msg:
        st.warning(ligand_input_msg)
    elif not smiles_ok:
        st.warning(smiles_or_msg)

    st.caption(f"Protein ID: `{protein_id}` | Ligand ID: `{ligand_id}` | Accelerator: `gpu` (fixed)")
    st.caption("Boltz YAML uses fixed chain IDs `A` (protein) and `B` (ligand) for compatibility.")

    yaml_preview = _build_input_yaml(
        protein_id=protein_id,
        sequence=protein_sequence if protein_ok else "",
        ligand_id=ligand_id,
        ligand_smiles=smiles_or_msg if smiles_ok else "",
        enable_affinity=bool(st.session_state.get("boltz2_enable_affinity", True)),
    )
    with st.expander("Input YAML preview", expanded=False):
        st.code(yaml_preview, language="yaml")

    run_disabled = not (protein_ok and smiles_ok)
    if st.button("Run Boltz-2", type="primary", disabled=run_disabled):
        structure_dir = _create_structure_job_stub(
            protein_id=protein_id,
            protein_sequence=protein_sequence,
            ligand_id=ligand_id,
            ligand_smiles=smiles_or_msg,
        )
        boltz_dir = structure_dir / "boltz2"
        boltz_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = boltz_dir / "input.yaml"
        yaml_path.write_text(yaml_preview)

        try:
            meta_path = structure_dir / "metadata.json"
            meta = _read_json(meta_path)
            meta["input_yaml_path"] = str(yaml_path)
            meta["boltz_output_dir"] = str(boltz_dir)
            _write_json(meta_path, meta)
        except Exception:
            pass

        params = {
            "boltz2_container": WORKFLOWS["boltz2"]["defaults"]["boltz2_container"],
            "input_yaml": str(yaml_path),
            "accelerator": "gpu",
            "boltz_cache_dir": str(st.session_state.get("boltz2_cache_dir", DEFAULT_BOLTZ_CACHE_DIR)),
            "boltz_msa_repository_dir": str(st.session_state.get("boltz2_msa_repository_dir", DEFAULT_MSA_REPOSITORY_DIR)),
            "use_msa_server": bool(st.session_state.get("boltz2_use_msa_server", True)),
            "use_potentials": bool(st.session_state.get("boltz2_use_potentials", True)),
            "sampling_steps": int(st.session_state.get("boltz2_sampling_steps", 200)),
            "recycling_steps": int(st.session_state.get("boltz2_recycling_steps", 3)),
            "diffusion_samples": int(st.session_state.get("boltz2_diffusion_samples", 1)),
            "sampling_steps_affinity": int(st.session_state.get("boltz2_sampling_steps_affinity", 200)),
            "diffusion_samples_affinity": int(st.session_state.get("boltz2_diffusion_samples_affinity", 5)),
            "affinity_mw_correction": bool(st.session_state.get("boltz2_affinity_mw_correction", False)),
            "metadata": {
                "source": "boltz",
                "status": "running",
                "entity_count": 2,
                "protein_sequence_length": len(protein_sequence),
                "protein_id": protein_id,
                "ligand_id": ligand_id,
                "ligand_smiles": smiles_or_msg,
                "structure_run_id": structure_dir.name,
                "structure_job_code": _read_json(structure_dir / "metadata.json").get("job_code", ""),
            },
        }

        with st.spinner("Running Boltz-2 prediction..."):
            run = _run_docker_workflow("boltz2", WORKFLOWS["boltz2"], params)

        if run.get("queued"):
            _update_structure_job_status(
                structure_dir,
                status="queued",
                boltz_run_id=str(run.get("run_id") or ""),
                boltz_job_code=_short_job_code(str(run.get("run_id") or "")) if str(run.get("run_id") or "") else "",
                message="Queued waiting for GPU lock.",
            )
            st.success(f"Boltz2 queued: {run['run_id']}")
            st.caption("Track this directly in Jobs – Structure (source=boltz, status=queued).")
            return

        if run.get("returncode") != 0:
            boltz_run_id = str(run.get("run_id") or "")
            boltz_job_code = _short_job_code(boltz_run_id) if boltz_run_id else ""
            _update_structure_job_status(
                structure_dir,
                status="failed",
                boltz_run_id=boltz_run_id,
                boltz_job_code=boltz_job_code,
                message=f"Boltz2 failed with exit code {run.get('returncode')}",
            )
            st.error(f"Boltz2 failed with exit code {run.get('returncode')}.")
            stderr_text = str(run.get("stderr") or "")
            if stderr_text:
                with st.expander("stderr", expanded=True):
                    st.code(stderr_text)
            return

        boltz_run_id = str(run.get("run_id") or "")
        boltz_job_code = _short_job_code(boltz_run_id) if boltz_run_id else ""
        registered_dir, error_message = _register_structure_job_from_boltz(
            structure_dir=structure_dir,
            boltz_run_id=boltz_run_id,
            boltz_job_code=boltz_job_code,
            protein_id=protein_id,
            protein_sequence=protein_sequence,
            ligand_id=ligand_id,
            ligand_smiles=smiles_or_msg,
        )
        if registered_dir is None:
            _update_structure_job_status(
                structure_dir,
                status="failed",
                boltz_run_id=boltz_run_id,
                boltz_job_code=boltz_job_code,
                message=f"Registration failed: {error_message}",
            )
            st.error(f"Boltz2 finished, but structure registration failed: {error_message}")
            st.code(f"Boltz run directory:\
{run.get('output_dir', '')}")
            return

        structure_dir = registered_dir
        structure_meta = json.loads((structure_dir / "metadata.json").read_text())
        st.success(f"Boltz2 completed and structure job registered: {structure_meta.get('job_code', '')}")
        st.code(
            "\n".join(
                [
                    f"Boltz run: {run.get('output_dir', '')}",
                    f"Structure job: {structure_dir}",
                    f"Complex PDB: {next(iter(sorted(structure_dir.glob('*_complex_refined.pdb'))), '')}",
                    f"Ligand SDF: {next(iter(sorted(structure_dir.glob('*_ligand_refined.sdf'))), '')}",
                ]
            )
        )
        if st.button("Open Structure Jobs"):
            _switch_to("app/pages/jobs_structure.py")



def render_boltz2_page() -> None:
    render_boltz2_ui(show_page_title=True)


def render_boltz2_inline() -> None:
    render_boltz2_ui(show_page_title=False)
