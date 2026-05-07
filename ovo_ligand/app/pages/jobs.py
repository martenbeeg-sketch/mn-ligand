from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import streamlit as st

from ovo_ligand.app.pages.bound_ligand_md import _run_root, _short_job_code


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _normalize_ligand_source(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in {"pdb", "vina", "boltz", "custom"}:
        return v
    if v == "costume":
        return "custom"
    return ""


def _infer_ligand_source(metadata: dict, result: dict) -> str:
    explicit = _normalize_ligand_source(
        metadata.get("ligand_source")
        or metadata.get("structure_source")
        or metadata.get("source")
        or (result.get("metadata") or {}).get("ligand_source")
        or result.get("ligand_source")
    )
    if explicit:
        return explicit
    if metadata.get("pdb_id") or result.get("pdb_id"):
        return "pdb"
    return "unknown"


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
                "ligand_source": _infer_ligand_source(metadata, result),
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


def _collect_md_system_prep_runs() -> list[dict]:
    runs_root = _run_root() / "md-system-prep"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        metadata_path = run_dir / "metadata.json"
        result_path = run_dir / "result.json"
        input_path = run_dir / "input.json"
        metadata = _read_json(metadata_path) if metadata_path.exists() else {}
        result = _read_json(result_path) if result_path.exists() else {}
        input_payload = _read_json(input_path) if input_path.exists() else {}

        selected_ligand = result.get("selected_ligand", {})
        md_result = result.get("md_result", {})
        output_files = md_result.get("output_files", {})
        has_nvt = bool(output_files.get("nvt_trajectory"))
        has_npt = bool(output_files.get("npt_trajectory"))
        has_prod = bool(output_files.get("production_trajectory"))

        effective_status = metadata.get("status") or ("completed" if result else "unknown")
        if result_path.exists():
            if result.get("success") is True:
                effective_status = "completed"
            elif result.get("success") is False:
                effective_status = "failed"

        structure_run_id = (
            metadata.get("structure_run_id")
            or (result.get("metadata") or {}).get("structure_run_id")
            or ""
        )
        structure_job_code = ""
        if structure_run_id:
            structure_meta = _read_json(_run_root() / "structure-jobs" / str(structure_run_id) / "metadata.json")
            structure_job_code = structure_meta.get("job_code") or _short_job_code(str(structure_run_id))

        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": metadata.get("job_code") or result.get("job_code") or _short_job_code(run_dir.name),
                "preset": metadata.get("preset") or input_payload.get("protocol_preset") or "custom",
                "status": effective_status,
                "ligand_source": _infer_ligand_source(metadata, result),
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
                "structure_run_id": structure_run_id,
                "structure_job_code": structure_job_code,
                "run_dir": str(run_dir),
            }
        )
    return rows


def _collect_structure_jobs() -> list[dict]:
    runs_root = _run_root() / "structure-jobs"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        metadata = _read_json(run_dir / "metadata.json")
        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": metadata.get("job_code") or _short_job_code(run_dir.name),
                "status": metadata.get("status") or "completed",
                "source": _normalize_ligand_source(metadata.get("source")) or "unknown",
                "pdb_id": metadata.get("pdb_id") or "",
                "ligand_count": metadata.get("ligand_count"),
                "created_at": metadata.get("created_at") or "",
                "protein_path": metadata.get("protein_path") or "",
                "ligand_path": metadata.get("ligand_path") or "",
            }
        )
    return rows


def _collect_openfe_jobs() -> list[dict]:
    roots = [
        _run_root() / "openfe",
        _run_root() / "abfe",
        _run_root() / "rbfe",
    ]
    rows: list[dict] = []
    for runs_root in roots:
        runs_root.mkdir(parents=True, exist_ok=True)
        workflow = runs_root.name.upper()
        for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
            metadata = _read_json(run_dir / "metadata.json")
            result = _read_json(run_dir / "result.json")
            rows.append(
                {
                    "run_id": run_dir.name,
                    "job_code": metadata.get("job_code") or result.get("job_code") or _short_job_code(run_dir.name),
                    "workflow": metadata.get("workflow") or workflow,
                    "status": metadata.get("status") or ("completed" if result else "unknown"),
                    "created_at": metadata.get("created_at") or "",
                    "completed_at": metadata.get("completed_at") or "",
                    "success": result.get("success"),
                }
            )
    return rows


def render_md_jobs() -> None:
    st.title("Jobs – MD Production")
    rows = _collect_bound_md_runs()
    if not rows:
        st.info("No bound-ligand MD runs found yet.")
        return
    df = pd.DataFrame(rows)
    statuses = sorted([s for s in df["status"].dropna().unique().tolist() if s])
    ligand_sources = sorted([s for s in df["ligand_source"].dropna().unique().tolist() if s])
    selected_statuses = st.multiselect("Status filter", statuses, default=statuses, key="md_status_filter")
    selected_sources = st.multiselect("Ligand source filter", ligand_sources, default=ligand_sources, key="md_source_filter")
    pdb_filter = st.text_input("PDB filter", value="", key="md_pdb_filter").strip().upper()
    ligand_filter = st.text_input("Ligand filter", value="", key="md_lig_filter").strip().upper()

    filtered = df.copy()
    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    if selected_sources:
        filtered = filtered[filtered["ligand_source"].isin(selected_sources)]
    if pdb_filter:
        filtered = filtered[filtered["pdb_id"].astype(str).str.upper().str.contains(pdb_filter)]
    if ligand_filter:
        filtered = filtered[filtered["ligand_key"].astype(str).str.upper().str.contains(ligand_filter)]
    if filtered.empty:
        st.caption("No MD jobs match current filters.")
        return

    table_rows = filtered[
        [
            "job_code", "status", "ligand_source", "pdb_id", "ligand_key",
            "created_at", "completed_at", "success", "has_nvt", "has_npt", "has_production", "run_id",
        ]
    ].copy()
    table_rows["open_run"] = table_rows["run_id"].apply(lambda rid: f"./md-results?run_id={rid}")
    table_rows["delete"] = False
    edited_rows = st.data_editor(
        table_rows[
            [
                "delete", "job_code", "open_run", "status", "ligand_source", "pdb_id",
                "ligand_key", "created_at", "completed_at", "success", "has_nvt", "has_npt", "has_production",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        disabled=[
            "job_code", "open_run", "status", "ligand_source", "pdb_id", "ligand_key",
            "created_at", "completed_at", "success", "has_nvt", "has_npt", "has_production",
        ],
        column_config={
            "delete": st.column_config.CheckboxColumn("delete", help="Select run for deletion"),
            "job_code": st.column_config.TextColumn("job_code"),
            "open_run": st.column_config.LinkColumn("open", display_text="Open"),
        },
        key="jobs_md_table_editor",
    )
    selected_indices = edited_rows.index[edited_rows["delete"] == True].tolist()  # noqa: E712
    selected_for_delete = table_rows.iloc[selected_indices]["run_id"].astype(str).tolist()
    st.caption("Use `Open` to view results. Select rows with `delete` to remove them.")
    confirm_delete = st.checkbox(
        "I understand this permanently deletes selected MD job folders",
        value=False,
        key="jobs_delete_md_confirm_checkbox",
    )
    if st.button(
        "Delete selected MD jobs",
        type="secondary",
        disabled=(not selected_for_delete) or (not confirm_delete),
        key="jobs_delete_md_selected_button",
    ):
        deleted: list[str] = []
        failed: list[str] = []
        for run_id in selected_for_delete:
            run_dir = _run_root() / "bound-ligand-md" / run_id
            try:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                deleted.append(run_id)
            except Exception as exc:
                failed.append(f"{run_id}: {exc}")
        if deleted:
            st.success(f"Deleted {len(deleted)} MD run(s).")
        if failed:
            st.error("Some runs could not be deleted:\n" + "\n".join(failed))
        st.rerun()


def render_md_system_prep_jobs() -> None:
    st.title("Jobs – MD System Preparation")
    rows = _collect_md_system_prep_runs()
    if not rows:
        st.info("No MD system preparation runs found yet.")
        return
    df = pd.DataFrame(rows)
    statuses = sorted([s for s in df["status"].dropna().unique().tolist() if s])
    ligand_sources = sorted([s for s in df["ligand_source"].dropna().unique().tolist() if s])
    selected_statuses = st.multiselect("Status filter", statuses, default=statuses, key="mdsp_status_filter")
    selected_sources = st.multiselect("Ligand source filter", ligand_sources, default=ligand_sources, key="mdsp_source_filter")
    pdb_filter = st.text_input("PDB filter", value="", key="mdsp_pdb_filter").strip().upper()
    ligand_filter = st.text_input("Ligand filter", value="", key="mdsp_lig_filter").strip().upper()

    filtered = df.copy()
    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    if selected_sources:
        filtered = filtered[filtered["ligand_source"].isin(selected_sources)]
    if pdb_filter:
        filtered = filtered[filtered["pdb_id"].astype(str).str.upper().str.contains(pdb_filter)]
    if ligand_filter:
        filtered = filtered[filtered["ligand_key"].astype(str).str.upper().str.contains(ligand_filter)]
    if filtered.empty:
        st.caption("No MD system preparation jobs match current filters.")
        return

    table_rows = filtered[
        [
            "job_code", "preset", "status", "ligand_source", "pdb_id", "ligand_key",
            "structure_job_code", "structure_run_id",
            "created_at", "completed_at", "success", "has_nvt", "has_npt", "run_id",
        ]
    ].copy()
    table_rows["job_code_link"] = table_rows.apply(
        lambda r: f"./md-results?run_id={r['run_id']}&run_type=md-system-prep&label={r['job_code']}",
        axis=1,
    )
    table_rows["structure_code_link"] = table_rows.apply(
        lambda r: (
            f"./structure-results?run_id={r['structure_run_id']}&label={r['structure_job_code']}"
            if str(r["structure_run_id"]).strip()
            else ""
        ),
        axis=1,
    )
    table_rows["delete"] = False
    edited_rows = st.data_editor(
        table_rows[
            [
                "delete", "job_code_link", "structure_code_link", "preset", "status", "ligand_source", "pdb_id",
                "ligand_key", "created_at", "completed_at", "success", "has_nvt", "has_npt",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        disabled=[
            "job_code_link", "structure_code_link", "preset", "status", "ligand_source", "pdb_id", "ligand_key",
            "created_at", "completed_at", "success", "has_nvt", "has_npt",
        ],
        column_config={
            "delete": st.column_config.CheckboxColumn("delete", help="Select run for deletion"),
            "job_code_link": st.column_config.LinkColumn("job_code", display_text=r"label=([^&]+)"),
            "structure_code_link": st.column_config.LinkColumn("structure_code", display_text=r"label=([^&]+)"),
        },
        key="jobs_mdsp_table_editor",
    )
    selected_indices = edited_rows.index[edited_rows["delete"] == True].tolist()  # noqa: E712
    selected_for_delete = table_rows.iloc[selected_indices]["run_id"].astype(str).tolist()
    st.caption("Use `Open` to inspect prepared MD system outputs. Select rows with `delete` to remove them.")
    confirm_delete = st.checkbox(
        "I understand this permanently deletes selected MD system preparation job folders",
        value=False,
        key="jobs_delete_mdsp_confirm_checkbox",
    )
    if st.button(
        "Delete selected MD system preparation jobs",
        type="secondary",
        disabled=(not selected_for_delete) or (not confirm_delete),
        key="jobs_delete_mdsp_selected_button",
    ):
        deleted: list[str] = []
        failed: list[str] = []
        for run_id in selected_for_delete:
            run_dir = _run_root() / "md-system-prep" / run_id
            try:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                deleted.append(run_id)
            except Exception as exc:
                failed.append(f"{run_id}: {exc}")
        if deleted:
            st.success(f"Deleted {len(deleted)} MD system preparation run(s).")
        if failed:
            st.error("Some runs could not be deleted:\n" + "\n".join(failed))
        st.rerun()


def render_structure_jobs() -> None:
    st.title("Jobs – Structure")
    rows = _collect_structure_jobs()
    if not rows:
        st.info("No structure preparation jobs found yet.")
        return
    df = pd.DataFrame(rows)
    statuses = sorted([s for s in df["status"].dropna().unique().tolist() if s])
    sources = sorted([s for s in df["source"].dropna().unique().tolist() if s])
    selected_statuses = st.multiselect("Status filter", statuses, default=statuses, key="structure_status_filter")
    selected_sources = st.multiselect("Source filter", sources, default=sources, key="structure_source_filter")
    pdb_filter = st.text_input("PDB filter", value="", key="structure_pdb_filter").strip().upper()

    filtered = df.copy()
    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    if selected_sources:
        filtered = filtered[filtered["source"].isin(selected_sources)]
    if pdb_filter:
        filtered = filtered[filtered["pdb_id"].astype(str).str.upper().str.contains(pdb_filter)]
    if filtered.empty:
        st.caption("No structure jobs match current filters.")
        return

    table_rows = filtered[
        [
            "job_code",
            "status",
            "source",
            "pdb_id",
            "ligand_count",
            "created_at",
            "protein_path",
            "ligand_path",
            "run_id",
        ]
    ].copy()
    table_rows["open_run"] = table_rows["run_id"].apply(lambda rid: f"./structure-results?run_id={rid}")
    table_rows["delete"] = False

    edited_rows = st.data_editor(
        table_rows[
            [
                "delete",
                "job_code",
                "open_run",
                "status",
                "source",
                "pdb_id",
                "ligand_count",
                "created_at",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        disabled=["job_code", "open_run", "status", "source", "pdb_id", "ligand_count", "created_at"],
        column_config={
            "delete": st.column_config.CheckboxColumn("delete", help="Select run for deletion"),
            "open_run": st.column_config.LinkColumn("open", display_text="Open"),
        },
        key="jobs_structure_table_editor",
    )
    selected_indices = edited_rows.index[edited_rows["delete"] == True].tolist()  # noqa: E712
    selected_for_delete = table_rows.iloc[selected_indices]["run_id"].astype(str).tolist()
    st.caption("Use `Open` to inspect prepared structure results. Select rows with `delete` to remove them.")
    confirm_delete = st.checkbox(
        "I understand this permanently deletes selected structure job folders",
        value=False,
        key="jobs_delete_structure_confirm_checkbox",
    )
    if st.button(
        "Delete selected structure jobs",
        type="secondary",
        disabled=(not selected_for_delete) or (not confirm_delete),
        key="jobs_delete_structure_selected_button",
    ):
        deleted: list[str] = []
        failed: list[str] = []
        for run_id in selected_for_delete:
            run_dir = _run_root() / "structure-jobs" / run_id
            try:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                deleted.append(run_id)
            except Exception as exc:
                failed.append(f"{run_id}: {exc}")
        if deleted:
            st.success(f"Deleted {len(deleted)} structure run(s).")
        if failed:
            st.error("Some runs could not be deleted:\n" + "\n".join(failed))
        st.rerun()


def render_openfe_jobs() -> None:
    st.title("Jobs – OpenFE")
    rows = _collect_openfe_jobs()
    if not rows:
        st.info("No OpenFE/ABFE/RBFE jobs found yet.")
        return
    st.dataframe(
        pd.DataFrame(rows)[
            ["job_code", "workflow", "status", "created_at", "completed_at", "success", "run_id"]
        ],
        hide_index=True,
        use_container_width=True,
    )
