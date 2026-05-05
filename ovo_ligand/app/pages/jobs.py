from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ovo_ligand.app.pages.bound_ligand_md import _render_md_results, _rewrite_output_paths, _run_root, _short_job_code


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _collect_bound_md_runs() -> list[dict]:
    runs_root = _run_root() / "bound-ligand-md"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        metadata_path = run_dir / "metadata.json"
        result_path = run_dir / "result.json"
        metadata = _read_json(metadata_path) if metadata_path.exists() else {}
        result = _read_json(result_path) if result_path.exists() else {}

        selected_ligand = result.get("selected_ligand", {})
        md_result = result.get("md_result", {})
        output_files = md_result.get("output_files", {})
        has_nvt = bool(output_files.get("nvt_trajectory"))
        has_npt = bool(output_files.get("npt_trajectory"))
        has_prod = bool(output_files.get("production_trajectory"))

        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": metadata.get("job_code") or result.get("job_code") or _short_job_code(run_dir.name),
                "status": metadata.get("status") or ("completed" if result else "unknown"),
                "created_at": metadata.get("created_at") or "",
                "updated_at": metadata.get("updated_at") or "",
                "completed_at": metadata.get("completed_at") or "",
                "pdb_id": metadata.get("pdb_id") or result.get("pdb_id") or "",
                "ligand_key": metadata.get("ligand_key") or selected_ligand.get("key") or "",
                "ligand_resname": selected_ligand.get("resname") or "",
                "success": result.get("success"),
                "has_result": result_path.exists(),
                "has_nvt": has_nvt,
                "has_npt": has_npt,
                "has_production": has_prod,
                "run_dir": str(run_dir),
            }
        )
    return rows


def render() -> None:
    st.title("Jobs")
    st.caption("Browse and analyze historical bound-ligand MD jobs")

    rows = _collect_bound_md_runs()
    if not rows:
        st.info("No bound-ligand MD runs found yet.")
        return

    df = pd.DataFrame(rows)
    statuses = sorted([s for s in df["status"].dropna().unique().tolist() if s])
    selected_statuses = st.multiselect("Status filter", statuses, default=statuses)
    pdb_filter = st.text_input("PDB filter", value="").strip().upper()
    ligand_filter = st.text_input("Ligand filter", value="").strip().upper()

    filtered = df.copy()
    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    if pdb_filter:
        filtered = filtered[filtered["pdb_id"].astype(str).str.upper().str.contains(pdb_filter)]
    if ligand_filter:
        filtered = filtered[filtered["ligand_key"].astype(str).str.upper().str.contains(ligand_filter)]

    st.dataframe(
        filtered[
            [
                "job_code",
                "run_id",
                "status",
                "pdb_id",
                "ligand_key",
                "created_at",
                "completed_at",
                "success",
                "has_nvt",
                "has_npt",
                "has_production",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    if filtered.empty:
        st.caption("No jobs match current filters.")
        return

    options = filtered["run_id"].tolist()
    selected_run_id = st.selectbox("Select job", options, index=0)
    selected_run_dir = Path(filtered.loc[filtered["run_id"] == selected_run_id, "run_dir"].iloc[0])
    metadata = _read_json(selected_run_dir / "metadata.json")
    result = _read_json(selected_run_dir / "result.json")

    st.subheader("Selected job")
    st.json(metadata)

    if result:
        rewritten = _rewrite_output_paths(result, selected_run_dir)
        _render_md_results(rewritten, selected_run_dir)
    else:
        st.warning("No result.json for this run yet. Job may still be running or failed before result output.")


render()

