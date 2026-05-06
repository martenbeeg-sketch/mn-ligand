from __future__ import annotations

import json
import io
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import urllib.request
from hashlib import sha1
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from ovo_ligand.app.components.molstar_viewer import (
    ChainVisualization,
    StructureVisualization,
    molstar_custom_component,
)
from ovo_ligand.ligandx.services.md.utils.pdb_utils import normalize_nonpolymer_residue_ids_in_pdb_block

from ovo_ligand.workflows.bound_ligand_md import COMMON_IONS, WATER, extract_ligand_pdb, parse_bound_ligands

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU",
}


LIGAND_NAME_FALLBACKS = {
    "STI": "Imatinib",
}

DEFAULT_MD_IMAGE = "ovolig-md-cu128:latest"
UI_EDIT_STAMP_UTC = "2026-05-05 12:30 UTC"

FIXED_MD_DEFAULTS = {
    "output_dir": "/output/md_outputs",
    "ligand_data_format": "pdb",
    "preserve_ligand_pose": True,
    "generate_conformer": False,
    "box_shape": "dodecahedron",
    "pressure": 1.0,
    "ionic_strength": 0.15,
    "production_report_interval": 2500,
    "preview_before_equilibration": False,
    "preview_acknowledged": False,
    "pause_at_minimized": False,
    "minimized_acknowledged": False,
}

PROTOCOL_PRESETS = {
    "Preview": {
        "description": "Fast smoke test. Useful for checking preparation and container wiring.",
        "heating_steps_per_stage": 250,
        "nvt_steps": 2500,
        "npt_steps": 2500,
        "production_steps": 0,
    },
    "Short MD": {
        "description": "Small exploratory run with a short production segment.",
        "heating_steps_per_stage": 1000,
        "nvt_steps": 10000,
        "npt_steps": 10000,
        "production_steps": 50000,
    },
    "Longer MD": {
        "description": "More realistic production setup. Still review before spending GPU time.",
        "heating_steps_per_stage": 5000,
        "nvt_steps": 50000,
        "npt_steps": 50000,
        "production_steps": 500000,
    },
    "Custom": {
        "description": "Expose the core MD step counts and system settings.",
        "heating_steps_per_stage": 250,
        "nvt_steps": 2500,
        "npt_steps": 2500,
        "production_steps": 0,
    },
}


def _run_root() -> Path:
    root = Path(os.getenv("OVO_LIGAND_RUN_DIR", "/tmp/ovo-ligand-runs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_metadata_path(run_dir: Path) -> Path:
    return run_dir / "metadata.json"


def _read_run_metadata(run_dir: Path) -> dict:
    path = _run_metadata_path(run_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_run_metadata(run_dir: Path, update: dict) -> dict:
    current = _read_run_metadata(run_dir)
    merged = {**current, **update}
    merged["run_id"] = merged.get("run_id") or run_dir.name
    merged["job_code"] = merged.get("job_code") or _short_job_code(run_dir.name)
    merged["updated_at"] = _utc_now_iso()
    _run_metadata_path(run_dir).write_text(json.dumps(merged, indent=2))
    return merged


def _current_ligand_source() -> str:
    prepared = st.session_state.get("prepared_structure_last", {}) or {}
    source = str(prepared.get("source", "")).strip().lower()
    if source in {"pdb", "vina", "boltz", "custom"}:
        return source
    return "pdb"


def _download_pdb(pdb_id: str) -> str:
    pdb_id = pdb_id.strip().upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


@st.cache_data(show_spinner=False)
def _ligand_metadata(resname: str) -> dict[str, str]:
    resname = resname.upper()
    metadata = {
        "name": LIGAND_NAME_FALLBACKS.get(resname, resname),
        "formula": "",
        "type": "",
    }
    try:
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{resname}"
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        chem_comp = data.get("chem_comp", {})
        metadata["name"] = chem_comp.get("name") or metadata["name"]
        metadata["formula"] = chem_comp.get("formula") or ""
        metadata["type"] = chem_comp.get("type") or ""
    except Exception:
        pass
    return metadata


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_command(
    image: str,
    output_dir: Path,
    input_json: Path,
    result_json: Path,
    use_gpu: bool,
) -> list[str]:
    command = ["docker", "run", "--rm"]
    if use_gpu:
        command += ["--gpus", "all"]
    command += [
        "-v",
        f"{_repo_root()}:/ovo-ligand:ro",
        "-v",
        f"{output_dir}:/output",
        "-e",
        "PYTHONPATH=/ovo-ligand",
        image,
        "python",
        "-m",
        "ovo_ligand.workflows.bound_ligand_md",
        "run",
        "--input",
        f"/output/{input_json.name}",
        "--output",
        f"/output/{result_json.name}",
    ]
    return command


def _build_prepare_command(
    image: str,
    output_dir: Path,
    input_json: Path,
    result_json: Path,
    use_gpu: bool,
) -> list[str]:
    command = ["docker", "run", "--rm"]
    if use_gpu:
        command += ["--gpus", "all"]
    command += [
        "-v",
        f"{_repo_root()}:/ovo-ligand:ro",
        "-v",
        f"{output_dir}:/output",
        "-e",
        "PYTHONPATH=/ovo-ligand",
        image,
        "python",
        "-m",
        "ovo_ligand.workflows.bound_ligand_md",
        "prepare",
        "--input",
        f"/output/{input_json.name}",
        "--output",
        f"/output/{result_json.name}",
    ]
    return command


def _prepare_structure_with_ligandx(
    pdb_id: str,
    raw_pdb_data: str,
    image: str,
    use_gpu: bool,
    map_modified_residues: bool,
) -> tuple[dict, list[str], subprocess.CompletedProcess[str]]:
    run_id = str(uuid4())
    output_dir = _run_root() / "prepared-structures" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    input_json = output_dir / "input.json"
    result_json = output_dir / "result.json"
    input_json.write_text(
        json.dumps(
            {
                "pdb_id": pdb_id,
                "pdb_data": raw_pdb_data,
                "clean_protein": True,
                "map_modified_residues": map_modified_residues,
            },
            indent=2,
        )
    )
    command = _build_prepare_command(image, output_dir, input_json, result_json, use_gpu)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    payload = json.loads(result_json.read_text()) if result_json.exists() else {}
    payload["output_dir"] = str(output_dir)
    return payload, command, result


def _host_path_from_container(path_value: str, output_dir: Path) -> str:
    if not path_value.startswith("/output/"):
        return path_value
    return str(output_dir / path_value.removeprefix("/output/"))


def _rewrite_output_paths(payload: dict, output_dir: Path) -> dict:
    if not payload:
        return payload
    payload = json.loads(json.dumps(payload))
    md_result = payload.get("md_result", {})
    output_files = md_result.get("container_output_files") or md_result.get("output_files", {})
    corrected_output_files = {
        key: _host_path_from_container(value, output_dir)
        for key, value in output_files.items()
        if isinstance(value, str)
    }
    if corrected_output_files:
        md_result["container_output_files"] = output_files
        md_result["output_files"] = corrected_output_files
    payload["job_code"] = payload.get("job_code") or _short_job_code(output_dir.name)
    payload["host_run_dir"] = str(output_dir)
    return payload


def _latest_md_result() -> tuple[Path, dict] | tuple[None, None]:
    run_root = _run_root() / "bound-ligand-md"
    result_files = sorted(run_root.glob("*/result.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for result_file in result_files:
        try:
            payload = json.loads(result_file.read_text())
            return result_file.parent, _rewrite_output_paths(payload, result_file.parent)
        except Exception:
            continue
    return None, None


def _parse_protein_chains(pdb_data: str) -> list[dict[str, int]]:
    chains: dict[str, set[int]] = {}
    for line in pdb_data.splitlines():
        if not line.startswith("ATOM"):
            continue
        chain = line[21].strip()
        if not chain:
            continue
        try:
            residue_id = int(line[22:26])
        except ValueError:
            continue
        chains.setdefault(chain, set()).add(residue_id)
    return [
        {"chain": chain, "residue_count": len(residues), "start": min(residues), "end": max(residues)}
        for chain, residues in sorted(chains.items())
        if residues
    ]


def _selection_string(chain: str, residue: str | int) -> str | None:
    chain = str(chain).strip()
    if not chain.isalpha():
        return None
    return f"{chain.upper()}{residue}"


def _ligand_label(ligand: dict) -> str:
    meta = _ligand_metadata(ligand["resname"])
    location = f"{ligand['resname']} chain {ligand['chain']} residue {ligand['resseq']}"
    if meta["name"] and meta["name"] != ligand["resname"]:
        return f"{meta['name']} ({location})"
    return location


def _protein_only_pdb(pdb_data: str) -> str:
    lines = []
    for line in pdb_data.splitlines():
        if line.startswith("ATOM"):
            lines.append(line)
        elif line.startswith(("TER", "MODEL", "ENDMDL", "CRYST1", "HEADER", "TITLE", "REMARK")):
            lines.append(line)
    return "\n".join(lines + ["END", ""])


def _protein_only_without_selected_ligand(pdb_data: str, ligand_resname: str) -> str:
    """Protein-only export that also drops the selected ligand even if encoded as ATOM records."""
    ligand_resname = (ligand_resname or "").strip().upper()
    lines = []
    for line in pdb_data.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            residue = line[17:20].strip().upper() if len(line) >= 20 else ""
            if line.startswith("HETATM"):
                # Skip all non-protein records in protein-only export.
                continue
            if ligand_resname and residue == ligand_resname:
                # Some toolchains may output selected ligand atoms as ATOM; exclude them.
                continue
            lines.append(line)
        elif line.startswith(("TER", "MODEL", "ENDMDL", "CRYST1", "HEADER", "TITLE", "REMARK")):
            lines.append(line)
    return "\n".join(lines + ["END", ""])


def _protein_ligand_only_pdb(pdb_data: str) -> str:
    lines = []
    for line in pdb_data.splitlines():
        if line.startswith(("ATOM", "HETATM", "TER", "MODEL", "ENDMDL", "CRYST1", "HEADER", "TITLE", "REMARK")):
            residue = line[17:20].strip() if len(line) >= 20 else ""
            if line.startswith("HETATM") and residue in WATER.union(COMMON_IONS):
                continue
            lines.append(line)
    return "\n".join(lines + ["END", ""])


def _count_pdb_contents(pdb_data: str) -> dict[str, int]:
    counts = {"protein_atoms": 0, "ligand_atoms": 0, "water_atoms": 0, "ion_atoms": 0, "total_atoms": 0}
    for line in pdb_data.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        counts["total_atoms"] += 1
        residue = line[17:20].strip()
        if line.startswith("ATOM"):
            counts["protein_atoms"] += 1
        elif residue in WATER:
            counts["water_atoms"] += 1
        elif residue in COMMON_IONS:
            counts["ion_atoms"] += 1
        else:
            counts["ligand_atoms"] += 1
    return counts


@st.cache_data(show_spinner=False)
def _trajectory_to_multimodel_pdb(
    dcd_path: str,
    topology_pdb_path: str,
    stride: int,
    align: bool,
    include_solvent: bool,
) -> dict:
    from ovo_ligand.ligandx.services.md.workflow.trajectory_processor import TrajectoryProcessorRunner

    processor = TrajectoryProcessorRunner()
    return processor.process_trajectory(
        dcd_path=dcd_path,
        pdb_path=topology_pdb_path,
        stride=stride,
        align=align,
        remove_solvent_flag=not include_solvent,
        include_unitcell=True,
    )


def _render_py3dmol_view(
    pdb_data: str,
    ligand_resname: str | None,
    *,
    height: int,
    animate: bool = False,
    show_unit_cell: bool = False,
    persist_key: str | None = None,
    frame_index: int | None = None,
    frame_label: str | None = None,
    frame_count: int | None = None,
    playing: bool = False,
) -> None:
    import py3Dmol

    view = py3Dmol.view(width="100%", height=height)
    multi_frame_mode = animate or (frame_index is not None)
    if multi_frame_mode:
        view.addModelsAsFrames(pdb_data, "pdb")
    else:
        view.addModel(pdb_data, "pdb")

    view.setStyle({"hetflag": False}, {"cartoon": {"color": "spectrum"}})
    # Keep protein clean by default; avoid dense stick clutter on large systems.
    view.addStyle({"hetflag": False}, {"stick": {"radius": 0.06, "opacity": 0.18}})
    view.setStyle({"hetflag": True}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.18}})
    if ligand_resname:
        view.setStyle({"resn": ligand_resname}, {"stick": {"colorscheme": "redCarbon", "radius": 0.24}})
    view.setStyle({"resn": "HOH"}, {"sphere": {"radius": 0.18, "color": "#7dd3fc", "opacity": 0.35}})
    for ion in COMMON_IONS:
        view.setStyle({"resn": ion}, {"sphere": {"radius": 0.55, "color": "#f59e0b", "opacity": 0.75}})

    if show_unit_cell:
        try:
            view.addUnitCell({"box": {"color": "black", "linewidth": 1.5}})
        except Exception:
            pass
    # Prefer focusing on the macromolecule/ligand first for readability.
    if ligand_resname:
        view.zoomTo({"or": [{"hetflag": False}, {"resn": ligand_resname}]})
    else:
        view.zoomTo({"hetflag": False})
    if animate:
        view.animate({"loop": "forward", "reps": 0, "interval": 180})
    view.render()
    html = _with_3dmol_fallback_loader(view._make_html())
    html = _with_view_persistence(
        html,
        persist_key=persist_key,
        frame_index=frame_index,
        animate=animate,
        frame_count=frame_count,
        playing=playing,
    )
    html = _with_frame_overlay(html, frame_label=frame_label)
    components.html(html, height=height + 20, scrolling=False)


def _with_3dmol_fallback_loader(html: str) -> str:
    # Replace single-source 3Dmol script include with a resilient multi-CDN loader.
    # This reduces transient "3Dmol.js failed to load" errors in iframe rerenders.
    loader = """
<script>
(function() {
  if (window.$3Dmol) return;
  var sources = [
    "https://3Dmol.org/build/3Dmol-min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js",
    "https://unpkg.com/3dmol@2.0.4/build/3Dmol-min.js"
  ];
  function loadAt(i) {
    if (window.$3Dmol) return;
    if (i >= sources.length) return;
    var s = document.createElement("script");
    s.src = sources[i];
    s.async = false;
    s.onload = function() {};
    s.onerror = function() { loadAt(i + 1); };
    document.head.appendChild(s);
  }
  loadAt(0);
})();
</script>
"""
    patched = re.sub(
        r"<script[^>]+3Dmol[^>]*></script>",
        loader,
        html,
        count=1,
        flags=re.IGNORECASE,
    )
    return patched


def _with_view_persistence(
    html: str,
    *,
    persist_key: str | None,
    frame_index: int | None,
    animate: bool,
    frame_count: int | None,
    playing: bool,
) -> str:
    if not persist_key:
        return html
    viewer_match = re.search(r"var\s+(viewer_\d+)\s*=\s*null;", html)
    container_match = re.search(r'id="(3dmolviewer_\d+)"', html)
    if not viewer_match or not container_match:
        return html
    viewer_var = viewer_match.group(1)
    container_id = container_match.group(1)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", persist_key)
    frame_js = ""
    if (frame_index is not None) and (not animate):
        frame_js = f"""
    try {{
      v.setFrame({int(frame_index)});
      v.render();
    }} catch (e) {{}}
"""
    play_js = ""
    if playing and frame_count and frame_count > 1:
        start_idx = int(frame_index or 0) % int(frame_count)
        play_js = f"""
    (function() {{
      var n = {int(frame_count)};
      var i = {start_idx};
      var badge = document.createElement("div");
      badge.style.position = "absolute";
      badge.style.top = "10px";
      badge.style.left = "10px";
      badge.style.zIndex = "20";
      badge.style.background = "rgba(17,24,39,0.75)";
      badge.style.color = "#f3f4f6";
      badge.style.border = "1px solid rgba(156,163,175,0.45)";
      badge.style.borderRadius = "6px";
      badge.style.padding = "4px 8px";
      badge.style.font = "600 12px/1.2 system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
      var root = document.getElementById("{container_id}");
      if (root) root.appendChild(badge);
      function tick() {{
        try {{
          v.setFrame(i);
          v.render();
          badge.textContent = "Frame " + (i + 1) + " / " + n;
          i = (i + 1) % n;
        }} catch (e) {{}}
      }}
      tick();
      setInterval(tick, 180);
    }})();
"""
    script = f"""
<script>
$3Dmolpromise.then(function() {{
  var v = {viewer_var};
  if (!v) return;
  var storageKey = "ovolig_view_{safe_key}";
  try {{
    var saved = localStorage.getItem(storageKey);
    if (saved) {{
      v.setView(JSON.parse(saved));
      v.render();
    }}
  }} catch (e) {{}}
{frame_js}
  var root = document.getElementById("{container_id}");
  var saveView = function() {{
    try {{
      localStorage.setItem(storageKey, JSON.stringify(v.getView()));
    }} catch (e) {{}}
  }};
  if (root) {{
    root.addEventListener("mouseup", saveView);
    root.addEventListener("wheel", saveView, {{ passive: true }});
    root.addEventListener("touchend", saveView, {{ passive: true }});
  }}
{play_js}
}});
</script>
"""
    return html + script


def _with_frame_overlay(html: str, *, frame_label: str | None) -> str:
    if not frame_label:
        return html
    esc = (
        frame_label.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    badge = f"""
<style>
.ovolig-frame-badge {{
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 10;
  background: rgba(17, 24, 39, 0.75);
  color: #f3f4f6;
  border: 1px solid rgba(156, 163, 175, 0.45);
  border-radius: 6px;
  padding: 4px 8px;
  font: 600 12px/1.2 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}}
</style>
<div class="ovolig-frame-badge">{esc}</div>
"""
    return badge + html


def _render_static_line_plot(
    df: pd.DataFrame,
    y_columns: list[str],
    title: str,
    y_label: str,
    x_label: str = "Time (ps)",
) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 1.9), dpi=130)
    for column in y_columns:
        if column in df.columns:
            ax.plot(df.index, df[column], label=column.replace("_", " "))
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    if len(y_columns) > 1:
        ax.legend(loc="best", fontsize=7, frameon=False)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    fig.tight_layout(pad=0.6)
    st.pyplot(fig, clear_figure=True, use_container_width=False)


def _phase_time_windows(analytics: dict) -> list[tuple[str, float, float]]:
    rmsd = analytics.get("rmsd", {})
    boundaries = rmsd.get("phase_boundaries", []) or []
    windows: list[tuple[str, float, float]] = []
    for item in boundaries:
        try:
            name = str(item.get("phase", "")).strip().upper()
            start = float(item.get("start_ps"))
            end = float(item.get("end_ps"))
            if name and end >= start:
                windows.append((name, start, end))
        except Exception:
            continue
    return windows


def _slice_phase(df: pd.DataFrame, start_ps: float, end_ps: float) -> pd.DataFrame:
    if df.empty:
        return df
    phase_df = df[(df.index >= start_ps) & (df.index <= end_ps)]
    return phase_df


def _derive_local_phase_rmsd_from_global(
    rmsd_df: pd.DataFrame,
    phase_windows: list[tuple[str, float, float]],
) -> dict[str, dict[str, list[float]]]:
    derived: dict[str, dict[str, list[float]]] = {}
    for phase_name, start_ps, end_ps in phase_windows:
        phase_df = _slice_phase(rmsd_df, start_ps, end_ps)
        if phase_df.empty:
            continue
        local_time = (phase_df.index - float(phase_df.index[0])).tolist()
        bb = phase_df["protein_backbone_rmsd_A"].to_numpy()
        lig = phase_df["ligand_rmsd_A"].to_numpy()
        bb0 = float(bb[0]) if len(bb) else 0.0
        lig0 = float(lig[0]) if len(lig) else 0.0
        derived[phase_name.lower()] = {
            "time_ps": [round(float(x), 3) for x in local_time],
            # Phase-local normalized series from existing global RMSD data
            "backbone_rmsd_angstrom": [round(max(0.0, float(x) - bb0), 4) for x in bb],
            "ligand_rmsd_angstrom": [round(max(0.0, float(x) - lig0), 4) for x in lig],
        }
    return derived


def _count_multimodel_frames(pdb_data: str) -> int:
    model_frames = pdb_data.count("\nMODEL")
    if model_frames == 0:
        model_frames = 1 if pdb_data.strip() else 0
    return model_frames


@st.cache_data(show_spinner=False)
def _get_dcd_total_frames(dcd_path: str) -> int | None:
    try:
        import mdtraj as md

        with md.open(dcd_path) as handle:
            return int(len(handle))
    except Exception:
        return None


def _frame_from_multimodel_pdb(pdb_data: str, frame_index: int) -> str:
    if "MODEL" not in pdb_data:
        return pdb_data
    blocks: list[str] = []
    current: list[str] = []
    in_model = False
    for line in pdb_data.splitlines():
        if line.startswith("MODEL"):
            in_model = True
            current = [line]
            continue
        if in_model:
            current.append(line)
            if line.startswith("ENDMDL"):
                blocks.append("\n".join(current))
                in_model = False
            continue
    if not blocks:
        return pdb_data
    safe_index = max(0, min(frame_index, len(blocks) - 1))
    return blocks[safe_index] + "\n"


@st.cache_data(show_spinner=False)
def _frame_from_dcd_with_topology(
    dcd_path: str,
    topology_pdb_path: str,
    frame_index: int,
    stride: int,
    align: bool,
    include_solvent: bool,
) -> str | None:
    try:
        import mdtraj as md
    except Exception:
        return None

    try:
        stride = max(1, int(stride))
        sampled_idx = max(0, int(frame_index))
        raw_idx = sampled_idx * stride
        try:
            with md.open(dcd_path) as handle:
                total_frames = int(len(handle))
        except Exception:
            total_frames = None
        if total_frames is not None and total_frames > 0:
            raw_idx = min(raw_idx, total_frames - 1)

        # Load exact frame with full topology.
        frame = md.load_frame(dcd_path, raw_idx, top=topology_pdb_path)
        if frame.n_frames == 0:
            return None

        # Keep coordinates physically consistent in the periodic box.
        if frame.unitcell_lengths is not None:
            try:
                protein_sel = frame.topology.select("protein")
                molecules = frame.topology.find_molecules()
                anchor_molecules = []
                if len(protein_sel) > 10:
                    protein_atom_set = set(protein_sel)
                    anchor_molecules = [
                        sorted(list(mol), key=lambda a: a.index)
                        for mol in molecules
                        if any(atom.index in protein_atom_set for atom in mol)
                    ]
                if not anchor_molecules and molecules:
                    largest = max(molecules, key=len)
                    anchor_molecules = [sorted(list(largest), key=lambda a: a.index)]
                if anchor_molecules:
                    frame.image_molecules(inplace=True, anchor_molecules=anchor_molecules)
                else:
                    frame.image_molecules(inplace=True)
            except Exception:
                pass

        # Write frame through OpenMM PDB writer, same style as stage-final outputs.
        from openmm.app import PDBFile
        topology = PDBFile(topology_pdb_path).topology
        positions = frame.openmm_positions(0)
        sio = io.StringIO()
        PDBFile.writeFile(topology, positions, sio, keepIds=True)
        pdb_text = sio.getvalue()

        if not include_solvent:
            pdb_text = _protein_ligand_only_pdb(pdb_text)

        return normalize_nonpolymer_residue_ids_in_pdb_block(pdb_text)
    except Exception:
        return None


def _extract_ligand_pdb_from_frame(frame_pdb: str, ligand_resname: str) -> str:
    resname = (ligand_resname or "").strip().upper()
    if not resname:
        return ""
    ligand_lines: list[str] = []
    for line in frame_pdb.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 26:
            continue
        if line[17:20].strip().upper() != resname:
            continue
        chars = list(line.ljust(80))
        chars[0:6] = list("HETATM")
        ligand_lines.append("".join(chars).rstrip())
    if not ligand_lines:
        return ""
    ligand_lines.append("END")
    return "\n".join(ligand_lines) + "\n"


def _add_ligand_conect_to_frame(frame_pdb: str, ligand: dict | None) -> str:
    if not ligand:
        return frame_pdb
    resname = str(ligand.get("resname", "")).strip().upper()
    chain = str(ligand.get("chain", "")).strip()
    resseq = str(ligand.get("resseq", "")).strip()
    if not resname:
        return frame_pdb
    ligand_atom_lines: list[str] = []
    ligand_serials: list[int] = []
    for line in frame_pdb.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
            continue
        if line[17:20].strip().upper() != resname:
            continue
        if chain and line[21].strip() != chain:
            continue
        if resseq and line[22:26].strip() != resseq:
            continue
        try:
            ligand_serials.append(int(line[6:11]))
            ligand_atom_lines.append(line)
        except ValueError:
            continue
    if not ligand_atom_lines:
        return frame_pdb
    ligand_block = "\n".join(ligand_atom_lines) + "\nEND\n"
    try:
        from rdkit import Chem
    except Exception:
        return frame_pdb
    mol = Chem.MolFromPDBBlock(ligand_block, sanitize=True, removeHs=False, proximityBonding=True)
    if mol is None or mol.GetNumAtoms() != len(ligand_serials):
        return frame_pdb
    neighbors: dict[int, set[int]] = {serial: set() for serial in ligand_serials}
    for bond in mol.GetBonds():
        a = ligand_serials[bond.GetBeginAtomIdx()]
        b = ligand_serials[bond.GetEndAtomIdx()]
        neighbors[a].add(b)
        neighbors[b].add(a)
    conect_lines: list[str] = []
    for serial in ligand_serials:
        bonded = sorted(neighbors.get(serial, set()))
        if not bonded:
            continue
        fields = "".join(f"{s:5d}" for s in bonded[:4])
        conect_lines.append(f"CONECT{serial:5d}{fields}")
    if not conect_lines:
        return frame_pdb
    lines = frame_pdb.splitlines()
    insert_at = len(lines)
    for idx, line in enumerate(lines):
        if line.startswith("ENDMDL") or line == "END":
            insert_at = idx
            break
    out_lines = lines[:insert_at] + conect_lines + lines[insert_at:]
    return "\n".join(out_lines) + "\n"


def _ligand_pdb_to_sdf(ligand_pdb: str) -> str | None:
    if not ligand_pdb.strip():
        return None
    try:
        from rdkit import Chem
    except Exception:
        return None
    mol = Chem.MolFromPDBBlock(ligand_pdb, sanitize=True, removeHs=False, proximityBonding=True)
    if mol is None:
        return None
    return Chem.MolToMolBlock(mol)


def _ligand_pdb_to_mol2(ligand_pdb: str) -> str | None:
    if not ligand_pdb.strip():
        return None
    obabel = shutil.which("obabel")
    if not obabel:
        return None
    with tempfile.TemporaryDirectory(prefix="ovolig_mol2_") as tmpdir:
        in_path = Path(tmpdir) / "ligand.pdb"
        out_path = Path(tmpdir) / "ligand.mol2"
        in_path.write_text(ligand_pdb)
        proc = subprocess.run(
            [obabel, "-ipdb", str(in_path), "-omol2", "-O", str(out_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not out_path.exists():
            return None
        return out_path.read_text()


def _short_job_code(run_dir_name: str) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digest = sha1(run_dir_name.encode("utf-8")).digest()
    return "".join(letters[b % 26] for b in digest[:3])


def _phase_token_from_traj_label(traj_label: str) -> str:
    label = traj_label.upper()
    if "NVT" in label:
        return "NVT"
    if "NPT" in label:
        return "NPT"
    if "PROD" in label:
        return "PROD"
    return "TRAJ"


def _frame_time_ps_for_phase(
    analytics: dict,
    phase_token: str,
    frame_idx: int,
    frame_count: int,
) -> float | None:
    token_map = {"NVT": "NVT", "NPT": "NPT", "PROD": "PRODUCTION"}
    desired = token_map.get(phase_token)
    if not desired:
        return None
    for phase_name, start_ps, end_ps in _phase_time_windows(analytics):
        if phase_name == desired:
            if frame_count <= 1:
                return start_ps
            span = max(0.0, end_ps - start_ps)
            fraction = frame_idx / max(1, frame_count - 1)
            return start_ps + span * fraction
    return None


def _download_frame_filename(
    run_dir: Path,
    pdb_id: str | None,
    ligand: dict | None,
    traj_label: str,
    frame_idx: int,
    frame_count: int,
    analytics: dict,
) -> str:
    job = _short_job_code(run_dir.name)
    pdb = (pdb_id or "PDB").upper()
    if ligand:
        ligand_token = f"{ligand.get('resname', 'LIG')}{ligand.get('chain', '_')}{ligand.get('resseq', '_')}"
    else:
        ligand_token = "LIG"
    phase = _phase_token_from_traj_label(traj_label)
    t_ps = _frame_time_ps_for_phase(analytics, phase, frame_idx, frame_count)
    time_token = f"{t_ps:.2f}ps" if t_ps is not None else f"frame{frame_idx + 1}"
    return f"{job}_{pdb}_{ligand_token}_{phase}_{time_token}.pdb"


def _render_structure_view(
    pdb_data: str,
    ligands: list[dict],
    selected_ligand: dict | None,
    selected_protein_chains: list[str],
    show_molstar_tools: bool,
    key_suffix: str,
    title: str = "Complex view",
    caption: str | None = None,
) -> None:
    st.markdown(f"#### {title}")
    if caption:
        st.caption(caption)
    chain_visualizations = [
        ChainVisualization(
            chain_id=chain_id,
            color="uniform",
            color_params={"value": "0x3b82f6"},
            representation_type="cartoon+ball-and-stick",
            label=f"Selected protein chain {chain_id}",
        )
        for chain_id in selected_protein_chains
    ]
    structures = [
        StructureVisualization(
            pdb=_protein_only_pdb(pdb_data),
            color="chain-id",
            representation_type="cartoon",
            chains=chain_visualizations,
        )
    ]
    for ligand in ligands:
        try:
            is_selected = selected_ligand and ligand["key"] == selected_ligand["key"]
            structures.append(
                StructureVisualization(
                    pdb=extract_ligand_pdb(pdb_data, ligand["key"]),
                    color="uniform",
                    color_params={"value": "0xff4b4b" if is_selected else "0x2da44e"},
                    representation_type="ball-and-stick",
                )
            )
        except Exception:
            pass
    return molstar_custom_component(
        structures=structures,
        key=(
            f"bound_md_viewer_layers_{key_suffix}_"
            f"{selected_ligand['key'] if selected_ligand else 'none'}_{'-'.join(selected_protein_chains)}"
        ),
        height=520,
        width="100%",
        show_controls=show_molstar_tools,
        selection_mode=False,
        force_reload=True,
    )


def _render_ligand_summary(ligand: dict) -> None:
    meta = _ligand_metadata(ligand["resname"])
    st.markdown(f"#### {_ligand_label(ligand)}")
    cols = st.columns(5)
    cols[0].metric("Residue ID", ligand["resname"])
    cols[1].metric("Chain", ligand["chain"])
    cols[2].metric("Residue", ligand["resseq"])
    cols[3].metric("Heavy atoms", ligand["heavy_atom_count"])
    cols[4].metric("Atoms", ligand["atom_count"])
    if meta["formula"] or meta["type"]:
        st.caption(" | ".join(part for part in [meta["formula"], meta["type"]] if part))


def _render_workflow_selection(selected_protein_chains: list[str], ligand: dict) -> None:
    st.markdown("#### Workflow selection")
    chain_text = ", ".join(f"chain {chain}" for chain in selected_protein_chains) or "No protein chain focused"
    ligand_text = _ligand_label(ligand)
    st.markdown(
        f"""
        <div style="border:1px solid #d0d7de;border-radius:8px;padding:12px;background:#f6f8fa">
          <div style="font-size:0.78rem;color:#57606a;margin-bottom:6px">Shown in viewer</div>
          <div style="margin-bottom:8px"><span style="color:#0969da;font-weight:700">Blue</span>: {chain_text}</div>
          <div><span style="color:#cf222e;font-weight:700">Red</span>: {ligand_text}</div>
          <div style="margin-top:8px"><span style="color:#1a7f37;font-weight:700">Green</span>: other bound ligands</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("The red molecule is the ligand that will be sent to the MD workflow.")


def _render_simulation_input_view(
    pdb_data: str,
    selected_ligand: dict,
    selected_protein_chains: list[str],
) -> None:
    _render_structure_view(
        pdb_data=pdb_data,
        ligands=[selected_ligand],
        selected_ligand=selected_ligand,
        selected_protein_chains=selected_protein_chains,
        show_molstar_tools=False,
        key_suffix="simulation_input",
        title="Simulation input preview",
        caption=(
            "This preview shows the repaired protein and the final selected ligand only. "
            "Other crystal ligands are not included in the current MD input."
        ),
    )


def _render_protocol_timeline(
    heating_steps: int,
    nvt_steps: int,
    npt_steps: int,
    production_steps: int,
    minimization_only: bool,
    include_prepare: bool = True,
    include_energy: bool = True,
) -> None:
    stages = []
    if include_prepare:
        stages.append(("Prepare", "Protein cleanup and selected bound ligand extraction", "enabled"))
    stages.extend(
        [
        ("Parameterize", "OpenFF ligand force field and OpenMM system setup", "enabled"),
        ("Minimize", "Relax clashes before dynamics", "enabled"),
        ("Heat", f"Gradual thermalization, {heating_steps:,} steps per stage", "skipped" if minimization_only else "enabled"),
        ("NVT", f"Constant volume equilibration, {nvt_steps:,} steps", "skipped" if minimization_only else "enabled"),
        ("NPT", f"Constant pressure equilibration, {npt_steps:,} steps", "skipped" if minimization_only else "enabled"),
        ("Production", f"{production_steps:,} MD steps", "skipped" if minimization_only or production_steps == 0 else "enabled"),
        ]
    )
    if include_energy:
        stages.append(("Energy", "MM/GBSA not found in Ligand-X; result is marked pending", "pending"))
    cols = st.columns(len(stages))
    for col, (title, detail, state) in zip(cols, stages):
        color = {"enabled": "#1f883d", "skipped": "#8c959f", "pending": "#bf8700"}[state]
        bg = {"enabled": "#dafbe1", "skipped": "#f6f8fa", "pending": "#fff8c5"}[state]
        col.markdown(
            f"""
            <div style="border:1px solid #d0d7de;border-radius:8px;padding:10px;min-height:116px;background:{bg}">
              <div style="font-weight:700;color:{color};font-size:0.9rem">{title}</div>
              <div style="font-size:0.78rem;line-height:1.25;margin-top:6px">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_md_results(result_payload: dict, run_dir: Path) -> None:
    md_result = result_payload.get("md_result", {})
    output_files = md_result.get("output_files", {})
    analytics = md_result.get("analytics", {})
    ligand_resname = result_payload.get("selected_ligand", {}).get("resname")
    selected_ligand = result_payload.get("selected_ligand")
    pdb_id = result_payload.get("pdb_id")
    job_code = result_payload.get("job_code") or _short_job_code(run_dir.name)

    st.subheader("7. Results")
    status = md_result.get("status", "unknown")
    if result_payload.get("success"):
        st.success(f"MD workflow status: {status}")
    else:
        st.error(f"MD workflow status: {status}")
    st.caption(f"Host run directory: {run_dir}")
    st.caption(f"Job code: {job_code}")

    kpi = analytics.get("kpi_summary", {})
    system_info = md_result.get("system_info", {})
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total atoms", f"{md_result.get('total_atoms', 0):,}")
    metric_cols[1].metric("Water molecules", f"{system_info.get('water_molecules', 0):,}")
    metric_cols[2].metric("Ions", f"{system_info.get('ions', 0):,}")

    minimization = md_result.get("equilibration_stats", {}).get("energy_minimization", {})

    thermodynamics = analytics.get("thermodynamics", {})
    if thermodynamics.get("time_ps"):
        st.markdown("#### Thermodynamics")
        min_energy_df = pd.DataFrame(
            [
                {
                    "stage": "minimization",
                    "initial_energy_kjmol": minimization.get("initial_energy"),
                    "final_energy_kjmol": minimization.get("final_energy"),
                    "delta_energy_kjmol": minimization.get("energy_change"),
                }
            ]
        )
        thermo_df = pd.DataFrame(
            {
                "time_ps": thermodynamics.get("time_ps", []),
                "potential_energy_kjmol": thermodynamics.get("potential_energy_kjmol", []),
                "temperature_k": thermodynamics.get("temperature_k", []),
                "density_gcm3": thermodynamics.get("density_gcm3", []),
                "volume_nm3": thermodynamics.get("volume_nm3", []),
            }
        ).set_index("time_ps")
        phase_windows = _phase_time_windows(analytics)
        if phase_windows:
            thermo_tabs = st.tabs(["Energy"] + [name for name, _, _ in phase_windows])
            with thermo_tabs[0]:
                st.dataframe(min_energy_df, use_container_width=True, hide_index=True)
            for tab, (phase_name, start_ps, end_ps) in zip(thermo_tabs[1:], phase_windows):
                with tab:
                    phase_df = _slice_phase(thermo_df, start_ps, end_ps)
                    if phase_df.empty:
                        st.caption(f"No thermodynamics samples inside {phase_name} window ({start_ps:.2f}-{end_ps:.2f} ps).")
                    else:
                        _render_static_line_plot(
                            phase_df,
                            ["potential_energy_kjmol"],
                            f"{phase_name} Potential Energy",
                            "kJ/mol",
                        )
                        _render_static_line_plot(
                            phase_df,
                            ["temperature_k"],
                            f"{phase_name} Temperature",
                            "K",
                        )
                        _render_static_line_plot(
                            phase_df,
                            ["density_gcm3"],
                            f"{phase_name} Density",
                            "g/cm3",
                        )
                        _render_static_line_plot(
                            phase_df,
                            ["volume_nm3"],
                            f"{phase_name} Box Volume",
                            "nm3",
                        )
        else:
            st.caption("Minimization energy summary")
            st.dataframe(min_energy_df, use_container_width=True, hide_index=True)
            _render_static_line_plot(
                thermo_df,
                ["potential_energy_kjmol"],
                "Potential Energy",
                "kJ/mol",
            )
            _render_static_line_plot(
                thermo_df,
                ["temperature_k"],
                "Temperature",
                "K",
            )
            _render_static_line_plot(
                thermo_df,
                ["density_gcm3"],
                "Density",
                "g/cm3",
            )
            _render_static_line_plot(
                thermo_df,
                ["volume_nm3"],
                "Box Volume",
                "nm3",
            )
        with st.expander("Thermodynamics table"):
            st.dataframe(thermo_df, use_container_width=True)

    rmsd = analytics.get("rmsd", {})
    if rmsd.get("time_ps"):
        st.markdown("#### RMSD")
        rmsd_reference_mode = st.radio(
            "RMSD reference mode",
            ["Global reference (first phase frame 0)", "Per-phase local reference (phase frame 0)"],
            horizontal=True,
            key=f"rmsd_ref_mode_{run_dir.name}",
        )
        use_per_phase_local = rmsd_reference_mode.startswith("Per-phase")
        rmsd_df = pd.DataFrame(
            {
                "time_ps": rmsd.get("time_ps", []),
                "protein_backbone_rmsd_A": rmsd.get("backbone_rmsd_angstrom", []),
                "ligand_rmsd_A": rmsd.get("ligand_rmsd_angstrom", []),
            }
        ).set_index("time_ps")
        phase_windows = _phase_time_windows(analytics)
        per_phase_local = rmsd.get("per_phase_local", {}) or {}
        if (not per_phase_local) and phase_windows:
            per_phase_local = _derive_local_phase_rmsd_from_global(rmsd_df, phase_windows)
            if use_per_phase_local:
                st.caption(
                    "Local mode uses derived phase-normalized RMSD for this older run. "
                    "Re-run analytics/new MD runs to get native per-phase local RMSD."
                )
        if phase_windows:
            def _tail20_mean(series: pd.Series) -> float | None:
                clean = series.dropna()
                if clean.empty:
                    return None
                n = max(1, len(clean) // 5)
                return float(clean.iloc[-n:].mean())

            def _status(mean_value: float | None, pass_thr: float, warn_thr: float) -> str:
                if mean_value is None:
                    return "n/a"
                if mean_value < pass_thr:
                    return "pass"
                if mean_value < warn_thr:
                    return "warn"
                return "fail"

            phase_kpi_map: dict[str, dict[str, object]] = {}
            for phase_name, start_ps, end_ps in phase_windows:
                if use_per_phase_local and phase_name.lower() in per_phase_local:
                    blk = per_phase_local.get(phase_name.lower(), {})
                    phase_calc_df = pd.DataFrame(
                        {
                            "protein_backbone_rmsd_A": blk.get("backbone_rmsd_angstrom", []),
                            "ligand_rmsd_A": blk.get("ligand_rmsd_angstrom", []),
                        }
                    )
                else:
                    phase_calc_df = _slice_phase(rmsd_df, start_ps, end_ps)
                bb_mean = _tail20_mean(phase_calc_df["protein_backbone_rmsd_A"]) if "protein_backbone_rmsd_A" in phase_calc_df else None
                lig_mean = _tail20_mean(phase_calc_df["ligand_rmsd_A"]) if "ligand_rmsd_A" in phase_calc_df else None
                phase_kpi_map[phase_name] = {
                    "bb_mean": bb_mean,
                    "bb_status": _status(
                        bb_mean,
                        float(kpi.get("backbone_rmsd_pass_a", 2.5)),
                        3.5,
                    ),
                    "lig_mean": lig_mean,
                    "lig_status": _status(
                        lig_mean,
                        float(kpi.get("ligand_rmsd_pass_a", 2.0)),
                        5.0,
                    ),
                }

            phase_names = [name for name, _, _ in phase_windows]
            selected_phase = st.segmented_control(
                "RMSD phase",
                phase_names,
                default=phase_names[0],
                key=f"rmsd_phase_{run_dir.name}",
            )
            phase_name, start_ps, end_ps = next(
                (w for w in phase_windows if w[0] == selected_phase),
                phase_windows[0],
            )
            if use_per_phase_local and phase_name.lower() in per_phase_local:
                local_block = per_phase_local.get(phase_name.lower(), {})
                phase_df = pd.DataFrame(
                    {
                        "time_ps": local_block.get("time_ps", []),
                        "protein_backbone_rmsd_A": local_block.get("backbone_rmsd_angstrom", []),
                        "ligand_rmsd_A": local_block.get("ligand_rmsd_angstrom", []),
                    }
                ).set_index("time_ps")
                x_label = "Phase time (ps)"
            else:
                phase_df = _slice_phase(rmsd_df, start_ps, end_ps)
                x_label = "Time (ps)"
            if phase_df.empty:
                st.caption(f"No RMSD samples inside {phase_name} window ({start_ps:.2f}-{end_ps:.2f} ps).")
            else:
                if use_per_phase_local:
                    st.caption(
                        f"Reference frame for {phase_name}: phase-local frame 0 "
                        f"({phase_name} trajectory first frame)."
                    )
                else:
                    global_ref = phase_windows[0][0]
                    st.caption(
                        f"Reference frame for {phase_name}: global frame 0 "
                        f"(first frame of {global_ref})."
                    )
                _render_static_line_plot(
                    phase_df,
                    ["protein_backbone_rmsd_A", "ligand_rmsd_A"],
                    f"{phase_name} Protein/Ligand RMSD",
                    "Angstrom",
                    x_label=x_label,
                )
                summary_cols = st.columns(4)
                phase_kpi = phase_kpi_map.get(phase_name, {})
                bb_mean = phase_kpi.get("bb_mean")
                lig_mean = phase_kpi.get("lig_mean")
                summary_cols[0].metric("Backbone status", str(phase_kpi.get("bb_status", "n/a")))
                summary_cols[1].metric(
                    "Backbone tail20 mean",
                    "n/a" if bb_mean is None else f"{float(bb_mean):.3f} A",
                )
                summary_cols[2].metric("Ligand status", str(phase_kpi.get("lig_status", "n/a")))
                summary_cols[3].metric(
                    "Ligand tail20 mean",
                    "n/a" if lig_mean is None else f"{float(lig_mean):.3f} A",
                )
                bb_pass = float(kpi.get("backbone_rmsd_pass_a", 2.5))
                lig_pass = float(kpi.get("ligand_rmsd_pass_a", 2.0))
                st.caption(
                    f"Status thresholds — Backbone: pass < {bb_pass:.1f} A, warn {bb_pass:.1f}-3.5 A, fail >= 3.5 A. "
                    f"Ligand: pass < {lig_pass:.1f} A, warn {lig_pass:.1f}-5.0 A, fail >= 5.0 A."
                )
        else:
            st.caption("Reference frame: global frame 0 (first trajectory frame).")
            _render_static_line_plot(
                rmsd_df,
                ["protein_backbone_rmsd_A", "ligand_rmsd_A"],
                "Protein/Ligand RMSD",
                "Angstrom",
            )
            summary_cols = st.columns(4)
            summary_cols[0].metric("Backbone RMSD max", f"{rmsd_df['protein_backbone_rmsd_A'].max():.3f} A")
            summary_cols[1].metric("Backbone RMSD last", f"{rmsd_df['protein_backbone_rmsd_A'].iloc[-1]:.3f} A")
            summary_cols[2].metric("Ligand RMSD max", f"{rmsd_df['ligand_rmsd_A'].max():.3f} A")
            summary_cols[3].metric("Ligand RMSD last", f"{rmsd_df['ligand_rmsd_A'].iloc[-1]:.3f} A")
        with st.expander("RMSD table"):
            st.dataframe(rmsd_df, use_container_width=True)

    st.markdown("#### 3D snapshots")
    snapshot_options = {
        "Prepared protein": output_files.get("protein_prepared"),
        "Solvated system": output_files.get("system_pdb"),
        "After minimization": output_files.get("minimized_pdb"),
        "After NVT": output_files.get("nvt_pdb"),
        "After NPT": output_files.get("npt_pdb"),
        "After production": output_files.get("production_pdb"),
    }
    snapshot_options = {key: value for key, value in snapshot_options.items() if value and Path(value).exists()}
    if snapshot_options:
        snapshot_label = st.selectbox("Structure snapshot", list(snapshot_options), index=len(snapshot_options) - 1)
        snapshot_path = Path(snapshot_options[snapshot_label])
        snapshot_mode = st.segmented_control(
            "Snapshot contents",
            ["Protein + ligand", "Full solvent box"],
            default="Protein + ligand",
            help="Full solvent box keeps the CRYST1 unit-cell record plus waters and ions.",
        )
        show_box = st.checkbox("Show periodic box contour", value=False, key=f"snapshot_box_{run_dir.name}_{snapshot_label}")
        st.caption(str(snapshot_path))
        raw_snapshot_pdb = snapshot_path.read_text()
        snapshot_pdb = raw_snapshot_pdb if snapshot_mode == "Full solvent box" else _protein_ligand_only_pdb(raw_snapshot_pdb)
        counts = _count_pdb_contents(snapshot_pdb)
        st.caption(
            " | ".join(
                [
                    f"atoms {counts['total_atoms']:,}",
                    f"protein {counts['protein_atoms']:,}",
                    f"ligand {counts['ligand_atoms']:,}",
                    f"water {counts['water_atoms']:,}",
                    f"ions {counts['ion_atoms']:,}",
                ]
            )
        )
        _render_py3dmol_view(snapshot_pdb, ligand_resname, height=560, show_unit_cell=show_box)
    else:
        st.info("No PDB snapshots were found for this run.")

    trajectory_files = {
        "NVT trajectory": output_files.get("nvt_trajectory"),
        "NPT trajectory": output_files.get("npt_trajectory"),
        "Production trajectory": output_files.get("production_trajectory"),
    }
    trajectory_files = {key: value for key, value in trajectory_files.items() if value and Path(value).exists()}
    if trajectory_files:
        st.markdown("#### Trajectory playback")
        traj_label = st.selectbox("Trajectory", list(trajectory_files), index=len(trajectory_files) - 1)
        topology_candidates = [
            output_files.get("production_pdb"),
            output_files.get("npt_pdb"),
            output_files.get("system_pdb"),
        ]
        topology_path = next((Path(value) for value in topology_candidates if value and Path(value).exists()), None)
        traj_cols = st.columns(3)
        with traj_cols[0]:
            stride = st.number_input(
                "Frame stride",
                min_value=1,
                max_value=500,
                value=1,
                step=1,
                help="Use a larger stride for faster loading and smaller playback files.",
            )
        with traj_cols[1]:
            align_trajectory = st.checkbox("Align on protein", value=True)
        with traj_cols[2]:
            include_solvent_trajectory = st.checkbox(
                "Include waters/ions",
                value=False,
                help="Solvent playback can be very large. Start without solvent, then enable if needed.",
            )
        total_frames = _get_dcd_total_frames(trajectory_files[traj_label])
        sampled_frames = (total_frames + int(stride) - 1) // int(stride) if total_frames else None
        if total_frames:
            st.caption(
                f"Frame estimate before generation: total `{total_frames}` frame(s), "
                f"with stride `{int(stride)}` -> about `{sampled_frames}` frame(s) to render."
            )
        else:
            st.caption("Frame estimate unavailable (could not read DCD header); generation will still work.")
        show_box_trajectory = st.checkbox(
            "Show periodic box contour",
            value=False,
            key=f"trajectory_box_{run_dir.name}_{traj_label}",
        )
        if topology_path:
            if st.button("Generate trajectory viewer", key=f"trajectory_view_{run_dir.name}_{traj_label}"):
                with st.spinner("Converting DCD frames to a Py3Dmol trajectory..."):
                    try:
                        processed = _trajectory_to_multimodel_pdb(
                            trajectory_files[traj_label],
                            str(topology_path),
                            int(stride),
                            align_trajectory,
                            include_solvent_trajectory,
                        )
                    except Exception as exc:
                        processed = {"error": str(exc), "pdb_data": ""}
                if processed.get("error"):
                    st.error(f"Trajectory conversion failed: {processed['error']}")
                    st.info("Install MDTraj in the ovo-ligand environment to enable DCD playback: conda install -c conda-forge mdtraj")
                elif processed.get("pdb_data"):
                    st.session_state[f"trajectory_pdb_{run_dir.name}_{traj_label}"] = processed["pdb_data"]
                    st.session_state[f"trajectory_unitcell_{run_dir.name}_{traj_label}"] = processed.get("unitcell_data")
                    st.session_state[f"trajectory_frames_{run_dir.name}_{traj_label}"] = _count_multimodel_frames(
                        processed["pdb_data"]
                    )
            trajectory_pdb = st.session_state.get(f"trajectory_pdb_{run_dir.name}_{traj_label}")
            trajectory_frames = st.session_state.get(f"trajectory_frames_{run_dir.name}_{traj_label}", 0)
            if trajectory_pdb:
                st.caption("Trajectory export mode: template frame download from DCD + stage topology (authoritative path).")
                counts = _count_pdb_contents(trajectory_pdb)
                st.caption(
                    f"Multi-model PDB playback: {counts['total_atoms']:,} atom records, {trajectory_frames} frame(s). "
                    "Right-click/drag and scroll controls still work."
                )
                if trajectory_frames <= 1:
                    st.warning(
                        "Only one frame is available, so playback appears static. "
                        "Use a smaller stride (typically 1) and regenerate."
                    )
                play_state_key = f"trajectory_playing_{run_dir.name}_{traj_label}"
                frame_state_key = f"trajectory_frame_idx_{run_dir.name}_{traj_label}"
                frame_widget_key = f"{frame_state_key}_widget"
                if play_state_key not in st.session_state:
                    st.session_state[play_state_key] = trajectory_frames > 1
                if frame_state_key not in st.session_state:
                    st.session_state[frame_state_key] = 0
                if frame_widget_key not in st.session_state:
                    st.session_state[frame_widget_key] = int(st.session_state[frame_state_key])

                b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
                with b1:
                    if st.button("Play", key=f"trajectory_play_{run_dir.name}_{traj_label}", disabled=trajectory_frames <= 1):
                        st.session_state[play_state_key] = True
                with b2:
                    if st.button("Pause", key=f"trajectory_pause_{run_dir.name}_{traj_label}", disabled=trajectory_frames <= 1):
                        st.session_state[play_state_key] = False
                with b3:
                    prev_clicked = st.button(
                        "Prev", key=f"trajectory_prev_{run_dir.name}_{traj_label}", disabled=trajectory_frames <= 1
                    )
                    if prev_clicked and st.session_state[frame_state_key] > 0:
                        st.session_state[play_state_key] = False
                        st.session_state[frame_state_key] -= 1
                        st.session_state[frame_widget_key] = int(st.session_state[frame_state_key])
                with b4:
                    next_clicked = st.button(
                        "Next", key=f"trajectory_next_{run_dir.name}_{traj_label}", disabled=trajectory_frames <= 1
                    )
                    if next_clicked and st.session_state[frame_state_key] < max(0, trajectory_frames - 1):
                        st.session_state[play_state_key] = False
                        st.session_state[frame_state_key] += 1
                        st.session_state[frame_widget_key] = int(st.session_state[frame_state_key])

                slider_value = st.slider(
                    "Frame",
                    min_value=0,
                    max_value=max(0, trajectory_frames - 1),
                    value=min(st.session_state[frame_state_key], max(0, trajectory_frames - 1)),
                    step=1,
                    key=frame_widget_key,
                )
                st.session_state[frame_state_key] = int(slider_value)
                selected_frame_idx = int(slider_value)
                selected_frame_pdb = _frame_from_multimodel_pdb(trajectory_pdb, selected_frame_idx)
                selected_frame_pdb = normalize_nonpolymer_residue_ids_in_pdb_block(selected_frame_pdb)
                download_frame_pdb = _frame_from_dcd_with_topology(
                    dcd_path=trajectory_files[traj_label],
                    topology_pdb_path=str(topology_path),
                    frame_index=selected_frame_idx,
                    stride=int(stride),
                    align=align_trajectory,
                    include_solvent=include_solvent_trajectory,
                )
                if not download_frame_pdb:
                    download_frame_pdb = selected_frame_pdb
                download_frame_pdb = _add_ligand_conect_to_frame(download_frame_pdb, selected_ligand)
                frame_filename = _download_frame_filename(
                    run_dir=run_dir,
                    pdb_id=pdb_id,
                    ligand=selected_ligand,
                    traj_label=traj_label,
                    frame_idx=selected_frame_idx,
                    frame_count=trajectory_frames,
                    analytics=analytics,
                )
                st.caption(f"Frame {selected_frame_idx + 1} / {trajectory_frames}")
                st.caption(f"Filename: {frame_filename}")
                if st.session_state[play_state_key] and trajectory_frames > 1:
                    st.caption("Playback: running")
                st.download_button(
                    "Download current frame (.pdb)",
                    data=download_frame_pdb,
                    file_name=frame_filename,
                    mime="chemical/x-pdb",
                    key=f"trajectory_download_frame_{run_dir.name}_{traj_label}",
                )

                if st.session_state[play_state_key] and trajectory_frames > 1:
                    _render_py3dmol_view(
                        trajectory_pdb,
                        ligand_resname,
                        height=620,
                        animate=True,
                        show_unit_cell=show_box_trajectory,
                        persist_key=f"{run_dir.name}_{traj_label}",
                        frame_label=None,
                        frame_count=trajectory_frames,
                        frame_index=selected_frame_idx,
                        playing=True,
                    )
                else:
                    _render_py3dmol_view(
                        trajectory_pdb,
                        ligand_resname,
                        height=620,
                        animate=False,
                        show_unit_cell=show_box_trajectory,
                        persist_key=f"{run_dir.name}_{traj_label}",
                        frame_index=selected_frame_idx,
                        frame_label=f"Frame {selected_frame_idx + 1} / {trajectory_frames}",
                        frame_count=trajectory_frames,
                        playing=False,
                    )
        else:
            st.info("No topology PDB was found for trajectory playback.")

        with st.expander("Trajectory files"):
            st.caption("DCD trajectories are also available for VMD, PyMOL, ChimeraX, or MDTraj.")
            st.table([{"trajectory": key, "path": value} for key, value in trajectory_files.items()])

    with st.expander("Output files"):
        st.table([{"name": key, "host path": value} for key, value in output_files.items()])

    with st.expander("Raw result JSON"):
        st.json(result_payload)


def _build_md_input_payload(
    pdb_id: str,
    pdb_data: str,
    selected_ligand: dict,
    run_id: str,
    charge_method: str,
    forcefield_method: str,
    heating_steps: int,
    nvt_steps: int,
    npt_steps: int,
    production_steps: int,
    temperature: float,
    padding_nm: float,
    minimization_only: bool,
) -> dict:
    return {
        "pdb_id": pdb_id,
        "pdb_data": pdb_data,
        "ligand_key": selected_ligand["key"],
        "job_id": run_id,
        "charge_method": charge_method,
        "forcefield_method": forcefield_method,
        "heating_steps_per_stage": int(heating_steps),
        "nvt_steps": int(nvt_steps),
        "npt_steps": int(npt_steps),
        "production_steps": int(production_steps),
        "temperature": float(temperature),
        "padding_nm": float(padding_nm),
        "minimization_only": bool(minimization_only),
        **FIXED_MD_DEFAULTS,
    }


def _render_run_construction(
    input_payload: dict | None,
    image: str,
    use_gpu: bool,
) -> None:
    st.subheader("5. Run construction")
    st.caption("This is the recipe that will be written to input.json and passed to the Docker runner.")

    editable = [
        ("Selected ligand", input_payload.get("ligand_key") if input_payload else "not selected"),
        ("Charge method", input_payload.get("charge_method") if input_payload else ""),
        ("Ligand force field", input_payload.get("forcefield_method") if input_payload else ""),
        ("Heating steps per stage", input_payload.get("heating_steps_per_stage") if input_payload else ""),
        ("NVT steps", input_payload.get("nvt_steps") if input_payload else ""),
        ("NPT steps", input_payload.get("npt_steps") if input_payload else ""),
        ("Production steps", input_payload.get("production_steps") if input_payload else ""),
        ("Temperature K", input_payload.get("temperature") if input_payload else ""),
        ("Solvent padding nm", input_payload.get("padding_nm") if input_payload else ""),
        ("Minimization only", input_payload.get("minimization_only") if input_payload else ""),
        ("Docker image", image),
        ("Mounted source", str(_repo_root())),
        ("GPU", use_gpu),
    ]
    fixed = [
        ("Ligand pose", "preserved from bound PDB coordinates"),
        ("Ligand conformer generation", "disabled"),
        ("Ligand data format", FIXED_MD_DEFAULTS["ligand_data_format"]),
        ("Solvent box shape", FIXED_MD_DEFAULTS["box_shape"]),
        ("Pressure bar", FIXED_MD_DEFAULTS["pressure"]),
        ("Ionic strength M", FIXED_MD_DEFAULTS["ionic_strength"]),
        ("Production report interval", FIXED_MD_DEFAULTS["production_report_interval"]),
        ("Preview/pause flags", "disabled"),
        ("Container app path", "/ovo-ligand"),
        ("Container output path", FIXED_MD_DEFAULTS["output_dir"]),
    ]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### User-editable in this wizard")
        st.table([{"parameter": key, "value": value} for key, value in editable])
    with col2:
        st.markdown("##### Fixed for this first workflow")
        st.table([{"parameter": key, "value": value} for key, value in fixed])

    with st.expander("Ligand-X parameters exposed by MDOptimizationConfig"):
        st.caption("Source: ovo_ligand/ligandx/services/md/config.py")
        st.table(
            [
                {"parameter": "protein_pdb_data", "used": "repaired PDB from the wizard", "editable": "no"},
                {"parameter": "ligand_structure_data", "used": "selected bound ligand PDB block", "editable": "no"},
                {"parameter": "ligand_data_format", "used": "pdb", "editable": "no"},
                {"parameter": "preserve_ligand_pose", "used": True, "editable": "no"},
                {"parameter": "generate_conformer", "used": False, "editable": "no"},
                {"parameter": "charge_method", "used": input_payload.get("charge_method") if input_payload else "", "editable": "yes"},
                {"parameter": "forcefield_method", "used": input_payload.get("forcefield_method") if input_payload else "", "editable": "yes"},
                {"parameter": "box_shape", "used": FIXED_MD_DEFAULTS["box_shape"], "editable": "fixed"},
                {"parameter": "nvt_steps", "used": input_payload.get("nvt_steps") if input_payload else "", "editable": "yes"},
                {"parameter": "npt_steps", "used": input_payload.get("npt_steps") if input_payload else "", "editable": "yes"},
                {"parameter": "heating_steps_per_stage", "used": input_payload.get("heating_steps_per_stage") if input_payload else "", "editable": "yes"},
                {"parameter": "production_steps", "used": input_payload.get("production_steps") if input_payload else "", "editable": "yes"},
                {"parameter": "production_report_interval", "used": FIXED_MD_DEFAULTS["production_report_interval"], "editable": "fixed"},
                {"parameter": "temperature", "used": input_payload.get("temperature") if input_payload else "", "editable": "yes"},
                {"parameter": "pressure", "used": FIXED_MD_DEFAULTS["pressure"], "editable": "fixed"},
                {"parameter": "ionic_strength", "used": FIXED_MD_DEFAULTS["ionic_strength"], "editable": "fixed"},
                {"parameter": "padding_nm", "used": input_payload.get("padding_nm") if input_payload else "", "editable": "yes"},
                {"parameter": "minimization_only", "used": input_payload.get("minimization_only") if input_payload else "", "editable": "yes"},
            ]
        )

    with st.expander("input.json preview", expanded=False):
        if input_payload:
            preview_payload = {k: v for k, v in input_payload.items() if k != "pdb_data"}
            preview_payload["pdb_data"] = f"<repaired PDB omitted from preview, {len(input_payload['pdb_data'])} characters>"
            st.json(preview_payload)
        else:
            st.caption("Download a PDB and select a ligand to preview the run JSON.")


def render() -> None:
    now_milan = datetime.now(ZoneInfo("Europe/Rome")).strftime("%Y-%m-%d %H:%M %Z")
    with st.sidebar:
        st.markdown("### App Status")
        st.caption(f"Bound ligand MD edit stamp (Milan): {now_milan}")
        st.caption(f"Patch reference (UTC): {UI_EDIT_STAMP_UTC}")
        st.caption("If this stamp changed, you are on the latest patched page.")

    st.title("Bound ligand MD")
    st.caption("Download a PDB, inspect the complex, select a bound ligand, and run Ligand-X MD in Docker")

    st.info(
        "MM/GBSA was not found in the original Ligand-X code. This first workflow runs the implemented MD stages and records MM/GBSA as a pending analysis step."
    )

    st.subheader("1. Structure")
    input_col, status_col = st.columns([0.3, 0.7], vertical_alignment="bottom")
    with input_col:
        pdb_id = st.text_input("PDB ID", value="1iep", max_chars=4).strip().upper()
    with status_col:
        download = st.button("Download, repair, and inspect ligands", type="primary")

    with st.expander("Repair/runtime settings"):
        st.caption("By default the downloaded PDB is repaired with the vendored Ligand-X/PDBFixer workflow before ligand selection.")
        prepare_image = st.text_input("Repair Docker image", value=DEFAULT_MD_IMAGE)
        prepare_use_gpu = st.checkbox("Use GPU for repair container", value=False)
        map_modified_residues = st.checkbox("Map supported modified amino acids to standard residues", value=True)

    if download:
        try:
            raw_pdb_data = _download_pdb(pdb_id)
            with st.spinner("Repairing protein with Ligand-X staged cleaning and reinserting bound ligands..."):
                prepared, command, result = _prepare_structure_with_ligandx(
                    pdb_id,
                    raw_pdb_data,
                    prepare_image,
                    prepare_use_gpu,
                    map_modified_residues,
                )
            if result.returncode != 0 or not prepared.get("success"):
                st.error("Ligand-X repair failed. The raw PDB was not used for ligand selection.")
                with st.expander("Repair Docker command", expanded=True):
                    st.code(shlex.join(command))
                if result.stdout:
                    with st.expander("repair stdout"):
                        st.code(result.stdout)
                if result.stderr:
                    with st.expander("repair stderr"):
                        st.code(result.stderr)
                st.stop()
            pdb_data = prepared["prepared_pdb_data"]
            ligands = prepared["ligands"]
            st.session_state["bound_md_pdb_id"] = pdb_id
            st.session_state["bound_md_pdb_data"] = pdb_data
            st.session_state["bound_md_raw_pdb_data"] = raw_pdb_data
            st.session_state["bound_md_ligands"] = ligands
            st.session_state["bound_md_prepare_result"] = prepared
            if ligands:
                cleaned_text = "cleaned/repaired" if prepared.get("protein_cleaned") else "checked"
                st.success(f"Protein {cleaned_text}; found {len(ligands)} candidate bound ligand residue(s).")
            else:
                st.warning("No non-water, non-ion HETATM ligand residues were found.")
        except Exception as exc:
            st.error(f"Download or repair failed: {exc}")

    pdb_data = st.session_state.get("bound_md_pdb_data")
    ligands = st.session_state.get("bound_md_ligands", [])
    protein_chains = _parse_protein_chains(pdb_data) if pdb_data else []
    prepare_result = st.session_state.get("bound_md_prepare_result")
    if prepare_result:
        with st.expander("Structure repair summary"):
            st.write(
                {
                    "protein_cleaned": prepare_result.get("protein_cleaned"),
                    "components": prepare_result.get("components", {}),
                    "modified_residue_mapping": prepare_result.get("modified_residue_mapping", {}),
                    "output_dir": prepare_result.get("output_dir"),
                }
            )
    selected_ligand = None
    if ligands:
        st.subheader("2. Ligand")
        chain_options = [item["chain"] for item in protein_chains]
        if chain_options:
            default_chains = chain_options[:1]
            selected_protein_chains = st.multiselect(
                "Protein chain(s) to focus",
                chain_options,
                default=default_chains,
                format_func=lambda chain: next(
                    f"Chain {item['chain']} ({item['residue_count']} residues, {item['start']}-{item['end']})"
                    for item in protein_chains
                    if item["chain"] == chain
                ),
            )
        else:
            selected_protein_chains = []

        ligand_col, viewer_note_col = st.columns([0.55, 0.45], vertical_alignment="bottom")
        with ligand_col:
            selected_index = st.radio(
                "Molecule for simulation",
                range(len(ligands)),
                format_func=lambda i: _ligand_label(ligands[i]),
                horizontal=True,
            )
        with viewer_note_col:
            st.caption("Selected protein chains are blue. The simulation ligand is red; other bound ligands are green.")
        selected_ligand = ligands[selected_index]
        show_molstar_tools = st.toggle(
            "Show Mol* tools panel",
            value=False,
            help="Shows the native Mol* structure tools sidebar. Usually not needed for this workflow.",
        )
        _render_structure_view(
            pdb_data,
            ligands,
            selected_ligand,
            selected_protein_chains,
            show_molstar_tools,
            key_suffix="complex",
            title="Complex view",
            caption="Full repaired complex view: selected ligand is red; other bound ligands are green.",
        )

        summary_col, selection_col = st.columns([0.48, 0.52], gap="large")
        with summary_col:
            _render_ligand_summary(selected_ligand)
        with selection_col:
            _render_workflow_selection(selected_protein_chains, selected_ligand)
            with st.expander("Details"):
                st.json(selected_ligand)
        _render_simulation_input_view(pdb_data, selected_ligand, selected_protein_chains)
    elif pdb_data:
        _render_structure_view(pdb_data, [], None, [], False, key_suffix="complex_empty")

    st.subheader("3. MD protocol")
    preset_name = st.segmented_control("Protocol preset", list(PROTOCOL_PRESETS), default="Preview")
    preset = PROTOCOL_PRESETS[preset_name]
    st.caption(preset["description"])
    minimization_only = st.toggle("Stop after minimization", value=False)

    with st.expander("Simulation parameters", expanded=preset_name == "Custom"):
        col1, col2, col3 = st.columns(3)
        with col1:
            heating_steps = st.number_input(
                "Heating steps per stage",
                min_value=0,
                value=int(preset["heating_steps_per_stage"]),
                step=250,
                help="Steps used in each heating interval before equilibration.",
            )
            nvt_steps = st.number_input(
                "NVT steps",
                min_value=0,
                value=int(preset["nvt_steps"]),
                step=500,
                help="Constant-volume equilibration steps.",
            )
        with col2:
            npt_steps = st.number_input(
                "NPT steps",
                min_value=0,
                value=int(preset["npt_steps"]),
                step=500,
                help="Constant-pressure equilibration steps.",
            )
            production_steps = st.number_input(
                "Production steps",
                min_value=0,
                value=int(preset["production_steps"]),
                step=1000,
                help="Production MD steps. Preview mode leaves this at 0.",
            )
        with col3:
            temperature = st.number_input("Temperature K", min_value=1.0, value=300.0)
            padding_nm = st.number_input("Solvent padding nm", min_value=0.1, value=1.0)

        charge_method = st.selectbox("Ligand charge method", ["gasteiger", "mmff94", "am1bcc"], index=0)
        forcefield_method = st.selectbox(
            "Ligand force field", ["openff-2.2.0", "openff-2.1.0", "openff-2.0.0"], index=0
        )

    _render_protocol_timeline(heating_steps, nvt_steps, npt_steps, production_steps, minimization_only)

    with st.expander("Docker/runtime settings"):
        image = st.text_input("MD Docker image", value=prepare_image if "prepare_image" in locals() else DEFAULT_MD_IMAGE)
        use_gpu = st.checkbox("Use GPU", value=True)

    preview_run_id = "preview"
    preview_payload = None
    if pdb_data and selected_ligand:
        preview_payload = _build_md_input_payload(
            st.session_state.get("bound_md_pdb_id", pdb_id),
            pdb_data,
            selected_ligand,
            preview_run_id,
            charge_method,
            forcefield_method,
            heating_steps,
            nvt_steps,
            npt_steps,
            production_steps,
            temperature,
            padding_nm,
            minimization_only,
        )
    _render_run_construction(preview_payload, image, use_gpu)

    st.subheader("6. Run MD")
    disabled = not (pdb_data and selected_ligand)
    if disabled:
        st.caption("Download a PDB and select a ligand before running.")

    rendered_results = False
    if st.button("Run bound ligand MD", type="primary", disabled=disabled):
        run_id = str(uuid4())
        output_dir = _run_root() / "bound-ligand-md" / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        ligand_source = _current_ligand_source()
        _write_run_metadata(
            output_dir,
            {
                "created_at": _utc_now_iso(),
                "workflow": "bound-ligand-md",
                "status": "running",
                "ligand_source": ligand_source,
                "pdb_id": st.session_state.get("bound_md_pdb_id", pdb_id),
                "ligand_key": selected_ligand.get("key") if selected_ligand else None,
                "ligand_label": _ligand_label(selected_ligand) if selected_ligand else None,
                "docker_image": image,
                "use_gpu": bool(use_gpu),
                "run_dir": str(output_dir),
            },
        )
        input_json = output_dir / "input.json"
        result_json = output_dir / "result.json"
        input_payload = _build_md_input_payload(
            st.session_state.get("bound_md_pdb_id", pdb_id),
            pdb_data,
            selected_ligand,
            run_id,
            charge_method,
            forcefield_method,
            heating_steps,
            nvt_steps,
            npt_steps,
            production_steps,
            temperature,
            padding_nm,
            minimization_only,
        )
        input_json.write_text(json.dumps(input_payload, indent=2))
        command = _build_command(image, output_dir, input_json, result_json, use_gpu)

        with st.expander("Docker command", expanded=True):
            st.code(shlex.join(command))

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            st.success(f"Workflow completed: {run_id}")
        else:
            st.error(f"Workflow failed with exit code {result.returncode}")
        st.write("Output directory")
        st.code(str(output_dir))
        if result_json.exists():
            result_payload = _rewrite_output_paths(json.loads(result_json.read_text()), output_dir)
            metadata = _write_run_metadata(
                output_dir,
                {
                    "status": "completed" if result_payload.get("success") else "failed",
                    "completed_at": _utc_now_iso(),
                    "result_json": str(result_json),
                    "host_run_dir": str(output_dir),
                    "ligand_source": ligand_source,
                    "pdb_id": result_payload.get("pdb_id") or st.session_state.get("bound_md_pdb_id", pdb_id),
                    "ligand_key": (result_payload.get("selected_ligand") or {}).get("key"),
                    "ligand_label": _ligand_label(result_payload.get("selected_ligand")) if result_payload.get("selected_ligand") else None,
                },
            )
            result_payload["metadata"] = metadata
            result_json.write_text(json.dumps(result_payload, indent=2))
            st.session_state["bound_md_last_result_dir"] = str(output_dir)
            st.session_state["bound_md_last_result"] = result_payload
            _render_md_results(result_payload, output_dir)
            rendered_results = True
        else:
            _write_run_metadata(
                output_dir,
                {
                    "status": "failed",
                    "completed_at": _utc_now_iso(),
                    "failure_reason": f"process_exit_{result.returncode}",
                    "host_run_dir": str(output_dir),
                },
            )
        if result.stdout:
            with st.expander("stdout"):
                st.code(result.stdout)
        if result.stderr:
            with st.expander("stderr"):
                st.code(result.stderr)

    if not rendered_results and not st.session_state.get("bound_md_last_result"):
        latest_dir, latest_result = _latest_md_result()
        if latest_dir and latest_result:
            st.session_state["bound_md_last_result_dir"] = str(latest_dir)
            st.session_state["bound_md_last_result"] = latest_result

    last_result = st.session_state.get("bound_md_last_result")
    last_result_dir = st.session_state.get("bound_md_last_result_dir")
    if not rendered_results and last_result and last_result_dir:
        _render_md_results(last_result, Path(last_result_dir))
    elif not rendered_results:
        st.subheader("7. Results")
        st.caption("Run MD once to see energy metrics, RMSD plots, thermodynamic traces, and 3D snapshots.")

    st.subheader("8. Energy analysis")
    st.warning(
        "MM/GBSA is not wired yet because no MM/GBSA implementation was found in the original Ligand-X code. "
        "The current workflow can generate MD outputs; the result JSON records MM/GBSA as not implemented."
    )


if __name__ == "__main__":
    render()
