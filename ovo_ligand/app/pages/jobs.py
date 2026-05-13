from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import streamlit as st

from ovo_ligand.app.pages.bound_ligand_md import _run_root, _short_job_code
from ovo_ligand.app.pages.common import reconcile_run_metadata_status


_ACTIVE_JOB_STATUSES = {"queued", "running", "preparing"}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _normalize_ligand_source(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in {"pdb", "vina", "gnina", "udp", "boltz", "custom"}:
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


def _infer_ligand_id(metadata: dict, selected_ligand: dict | None = None) -> str:
    selected_ligand = selected_ligand or {}
    label = str(metadata.get("ligand_label") or "").strip()
    if label:
        # Typical label format: "T3 chain A residue 501"
        token = label.split()[0].strip()
        if token:
            return token
    value = str(
        metadata.get("ligand_id")
        or metadata.get("ligand_resname")
        or selected_ligand.get("resname")
        or ""
    ).strip()
    if value:
        return value
    key = str(metadata.get("ligand_key") or selected_ligand.get("key") or "").strip()
    if key and "|" in key:
        return key.split("|", 1)[0].strip() or "LIG"
    return "LIG"


def _collect_bound_md_runs() -> list[dict]:
    runs_root = _run_root() / "bound-ligand-md"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        reconcile_run_metadata_status(run_dir)
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
        structure_run_id = str(metadata.get("structure_run_id") or "").strip()
        structure_ligand_id = ""
        if structure_run_id:
            structure_meta = _read_json(_run_root() / "structure-jobs" / structure_run_id / "metadata.json")
            structure_ligand_id = _infer_ligand_id(structure_meta)

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
                "ligand_id": (structure_ligand_id or _infer_ligand_id(metadata, selected_ligand)),
                "ligand_resname": selected_ligand.get("resname") or "",
                "has_result": result_path.exists(),
                "has_nvt": has_nvt,
                "has_npt": has_npt,
                "has_production": has_prod,
                "structure_run_id": structure_run_id,
                "run_dir": str(run_dir),
                "success": result.get("success"),            }
        )
    return rows


def _collect_md_system_prep_runs() -> list[dict]:
    runs_root = _run_root() / "md-system-prep"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        reconcile_run_metadata_status(run_dir)
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
        structure_source = ""
        structure_ligand_id = ""
        if structure_run_id:
            structure_meta = _read_json(_run_root() / "structure-jobs" / str(structure_run_id) / "metadata.json")
            structure_job_code = structure_meta.get("job_code") or _short_job_code(str(structure_run_id))
            structure_source = _normalize_ligand_source(structure_meta.get("source")) or ""
            structure_ligand_id = _infer_ligand_id(structure_meta)

        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": metadata.get("job_code") or result.get("job_code") or _short_job_code(run_dir.name),
                "preset": metadata.get("preset") or input_payload.get("protocol_preset") or "custom",
                "min_steps": input_payload.get("minimization_max_iterations"),
                "heating_scheme": f"{input_payload.get('heating_stages', '-') }x{input_payload.get('heating_steps_per_stage', '-')}",
                "restrained": bool(input_payload.get("apply_protein_restraints_during_heating_nvt") or input_payload.get("ligand_restraints_enabled") or input_payload.get("enable_ligand_planarity_restraints")),
                "nvt_steps": input_payload.get("nvt_steps"),
                "npt_steps": input_payload.get("npt_steps"),
                "release_scheme": (
                    "off"
                    if not bool(input_payload.get("npt_release_enabled"))
                    else f"P:{input_payload.get('protein_npt_release_scales', '-')}"
                ),
                "status": effective_status,
                "ligand_source": _infer_ligand_source(metadata, result),
                "created_at": metadata.get("created_at") or "",
                "updated_at": metadata.get("updated_at") or "",
                "completed_at": metadata.get("completed_at") or "",
                "pdb_id": metadata.get("pdb_id") or result.get("pdb_id") or "",
                "ligand_key": metadata.get("ligand_key") or selected_ligand.get("key") or "",
                "ligand_id": (structure_ligand_id or _infer_ligand_id(metadata, selected_ligand)),
                "ligand_resname": selected_ligand.get("resname") or "",
                "has_result": result_path.exists(),
                "has_nvt": has_nvt,
                "has_npt": has_npt,
                "has_production": has_prod,
                "structure_run_id": structure_run_id,
                "structure_job_code": structure_job_code,
                "structure_source": structure_source,
                "run_dir": str(run_dir),
                "success": result.get("success"),
            }
        )
    return rows


def _collect_structure_jobs() -> list[dict]:
    runs_root = _run_root() / "structure-jobs"
    structure_docking_root = _run_root() / "structure-docking"
    runs_root.mkdir(parents=True, exist_ok=True)
    structure_docking_root.mkdir(parents=True, exist_ok=True)
    docking_index: dict[str, list[Path]] = {}
    for d in [p for p in structure_docking_root.iterdir() if p.is_dir()]:
        meta = _read_json(d / "metadata.json")
        src = str(meta.get("source_structure_run_id") or "").strip()
        if not src:
            continue
        docking_index.setdefault(src, []).append(d)
    rows: list[dict] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        metadata = _read_json(run_dir / "metadata.json")
        docking_root_legacy = run_dir / "docking_runs"
        docking_runs = list(docking_index.get(run_dir.name, []))
        if docking_root_legacy.exists():
            docking_runs.extend([p for p in docking_root_legacy.iterdir() if p.is_dir()])
        docking_runs = sorted(docking_runs, key=lambda p: p.stat().st_mtime, reverse=True)
        docking_count = len(docking_runs)
        last_docking_status = ""
        if docking_runs:
            last_meta = _read_json(docking_runs[0] / "metadata.json")
            last_docking_status = str(last_meta.get("status") or "")
        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": metadata.get("job_code") or _short_job_code(run_dir.name),
                "status": metadata.get("status") or "completed",
                "source": _normalize_ligand_source(metadata.get("source")) or "unknown",
                "pdb_id": metadata.get("pdb_id") or "",
                "source_structure_job_code": metadata.get("source_structure_job_code") or "",
                "ligand_id": _infer_ligand_id(metadata),
                "ligand_count": metadata.get("ligand_count"),
                "docking_runs": docking_count,
                "last_docking_status": last_docking_status,
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
            reconcile_run_metadata_status(run_dir)
            metadata = _read_json(run_dir / "metadata.json")
            result = _read_json(run_dir / "result.json")
            live_status = _read_json(run_dir / "jobs" / f"{run_dir.name}.json")
            workflow_key = str(metadata.get("workflow_key") or runs_root.name).lower()
            # Always keep OpenFE run code independent from upstream structure code.
            # For older runs where metadata.job_code was overwritten with structure code,
            # fall back to run_id-derived short code.
            openfe_job_code = metadata.get("openfe_job_code") or _short_job_code(run_dir.name)

            source_job_code = metadata.get("source_job_code") or metadata.get("job_code_structure") or ""
            if not source_job_code:
                structure_run_id = str(metadata.get("structure_run_id") or "").strip()
                if structure_run_id:
                    structure_meta = _read_json(_run_root() / "structure-jobs" / structure_run_id / "metadata.json")
                    source_job_code = structure_meta.get("job_code") or _short_job_code(structure_run_id)

            status = live_status.get("status") or metadata.get("status") or ("completed" if result else "unknown")
            current_step = live_status.get("current_step")
            total_steps = live_status.get("total_steps")
            progress_pct = live_status.get("progress")
            steps_remaining = live_status.get("steps_remaining")
            stage = live_status.get("stage") or ""
            eta_remaining_hours = live_status.get("eta_remaining_hours")
            status_message = live_status.get("status_message") or ""

            rows.append(
                {
                    "run_id": run_dir.name,
                    "job_code": openfe_job_code,
                    "source_job_code": source_job_code,
                    "workflow": metadata.get("workflow") or workflow,
                    "workflow_key": workflow_key,
                    "pdb_id": metadata.get("pdb_id") or "",
                    "ligand_id": _infer_ligand_id(metadata),
                    "preset": ((metadata.get("run_inputs") or {}).get("preset") or ""),
                    "prod_ns": ((metadata.get("run_inputs") or {}).get("production_length_ns")),
                    "eq_ns": ((metadata.get("run_inputs") or {}).get("equilibration_length_ns")),
                    "repeats": ((metadata.get("run_inputs") or {}).get("protocol_repeats")),
                    "timeout_h": ((metadata.get("run_inputs") or {}).get("timeout_budget_hours_est")),
                    "status": status,
                    "stage": stage,
                    "progress_pct": progress_pct,
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "steps_remaining": steps_remaining,
                    "eta_h_remaining": eta_remaining_hours,
                    "status_message": status_message,
                    "created_at": metadata.get("created_at") or "",
                    "completed_at": metadata.get("completed_at") or "",
                    "success": result.get("success"),
                }
            )
    return rows


def _collect_admet_jobs() -> list[dict]:
    runs_root = _run_root() / "admet"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        reconcile_run_metadata_status(run_dir)
        metadata = _read_json(run_dir / "metadata.json")
        result = _read_json(run_dir / "result.json")
        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": metadata.get("job_code") or _short_job_code(run_dir.name),
                "status": metadata.get("status") or ("completed" if result else "unknown"),
                "input_smiles": (metadata.get("input_smiles") or "")[:120],
                "source_job_code": metadata.get("source_job_code") or "",
                "created_at": metadata.get("created_at") or "",
                "completed_at": metadata.get("completed_at") or "",
                "success": result.get("success"),
            }
        )
    return rows


def _collect_qc_jobs() -> list[dict]:
    runs_root = _run_root() / "qc"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        reconcile_run_metadata_status(run_dir)
        metadata = _read_json(run_dir / "metadata.json")
        result = _read_json(run_dir / "result.json")
        rows.append(
            {
                "run_id": run_dir.name,
                "job_code": metadata.get("job_code") or _short_job_code(run_dir.name),
                "status": metadata.get("status") or ("completed" if result else "unknown"),
                "source_job_code": metadata.get("source_job_code") or "",
                "method": metadata.get("qc_method") or "",
                "basis": metadata.get("qc_basis") or "",
                "charge": metadata.get("qc_charge"),
                "multiplicity": metadata.get("qc_multiplicity"),
                "created_at": metadata.get("created_at") or "",
                "completed_at": metadata.get("completed_at") or "",
                "success": result.get("success"),
            }
        )
    return rows


def _enable_jobs_auto_refresh(rows: list[dict], scope_key: str, interval_seconds: int = 10) -> None:
    """Auto-refresh jobs page while active jobs exist."""
    has_active = any(str(r.get("status") or "").strip().lower() in _ACTIVE_JOB_STATUSES for r in rows)
    if not has_active:
        return
    st.caption(f"Auto-refresh active ({scope_key}): every {interval_seconds}s while jobs are running/queued.")
    st.markdown(
        f"<meta http-equiv='refresh' content='{int(max(3, interval_seconds))}'>",
        unsafe_allow_html=True,
    )


def render_md_jobs() -> None:
    st.title("Jobs – MD Production")
    rows = _collect_bound_md_runs()
    _enable_jobs_auto_refresh(rows, "md-production")
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
        filtered = filtered[filtered["ligand_id"].astype(str).str.upper().str.contains(ligand_filter)]
    if filtered.empty:
        st.caption("No MD jobs match current filters.")
        return

    table_rows = filtered[
        [
            "job_code", "status", "ligand_source", "pdb_id", "ligand_id",
            "created_at", "completed_at", "success", "has_production", "run_id",
        ]
    ].copy()
    table_rows["job_code_link"] = table_rows.apply(
        lambda r: f"./md-results?run_id={r['run_id']}&label={r['job_code']}",
        axis=1,
    )
    table_rows["delete"] = False
    edited_rows = st.data_editor(
        table_rows[
            [
                "delete", "job_code_link", "status", "ligand_source", "pdb_id",
                "ligand_id", "created_at", "completed_at", "success", "has_production",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        disabled=[
            "job_code_link", "status", "ligand_source", "pdb_id", "ligand_id",
            "created_at", "completed_at", "success", "has_production",
        ],
        column_config={
            "delete": st.column_config.CheckboxColumn("delete", help="Select run for deletion"),
            "job_code_link": st.column_config.LinkColumn("job_code", display_text=r"label=([^&]+)"),
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
    _enable_jobs_auto_refresh(rows, "md-system-prep")
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
        filtered = filtered[filtered["ligand_id"].astype(str).str.upper().str.contains(ligand_filter)]
    if filtered.empty:
        st.caption("No MD system preparation jobs match current filters.")
        return

    table_rows = filtered[
        [
            "job_code", "min_steps", "heating_scheme", "restrained", "nvt_steps", "npt_steps", "release_scheme",
            "status", "ligand_source", "pdb_id", "ligand_id",
            "structure_job_code", "structure_run_id", "structure_source",
            "created_at", "completed_at", "success", "has_nvt", "has_npt", "run_id",
        ]
    ].copy()
    table_rows["input_ref"] = table_rows.apply(
        lambda r: (
            str(r["pdb_id"]).strip()
            if str(r["structure_source"]).strip().lower() == "pdb"
            else str(r["structure_job_code"]).strip()
        ),
        axis=1,
    )
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
                "delete", "job_code_link", "structure_code_link",
                "min_steps", "heating_scheme", "restrained", "nvt_steps", "npt_steps", "release_scheme",
                "status", "ligand_source", "input_ref",
                "ligand_id", "created_at", "completed_at", "success", "has_nvt", "has_npt",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        disabled=[
            "job_code_link", "structure_code_link",
            "min_steps", "heating_scheme", "restrained", "nvt_steps", "npt_steps", "release_scheme",
            "status", "ligand_source", "input_ref", "ligand_id",
            "created_at", "completed_at", "success", "has_nvt", "has_npt",
        ],
        column_config={
            "delete": st.column_config.CheckboxColumn("delete", help="Select run for deletion"),
            "job_code_link": st.column_config.LinkColumn("job_code", display_text=r"label=([^&]+)"),
            "structure_code_link": st.column_config.LinkColumn("structure_code", display_text=r"label=([^&]+)"),
            "input_ref": st.column_config.TextColumn("input"),
            "restrained": st.column_config.CheckboxColumn("restrained"),
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
    _enable_jobs_auto_refresh(rows, "structure")
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
            "source_structure_job_code",
            "ligand_id",
            "ligand_count",
            "created_at",
            "protein_path",
            "ligand_path",
            "run_id",
        ]
    ].copy()
    table_rows["input_ref"] = table_rows.apply(
        lambda r: (
            str(r["pdb_id"]).strip()
            if str(r["source"]).strip().lower() == "pdb"
            else str(r["source_structure_job_code"]).strip()
        ),
        axis=1,
    )
    table_rows["open_run"] = table_rows["run_id"].apply(lambda rid: f"./structure-results?run_id={rid}")
    table_rows["job_code_link"] = table_rows.apply(
        lambda r: f"./structure-results?run_id={r['run_id']}&label={r['job_code']}",
        axis=1,
    )
    table_rows["delete"] = False

    edited_rows = st.data_editor(
        table_rows[
            [
                "delete",
                "job_code_link",
                "status",
                "source",
                "input_ref",
                "ligand_id",
                "ligand_count",
                "created_at",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        disabled=["job_code_link", "status", "source", "input_ref", "ligand_id", "ligand_count", "created_at"],
        column_config={
            "delete": st.column_config.CheckboxColumn("delete", help="Select run for deletion"),
            "job_code_link": st.column_config.LinkColumn("job_code", display_text=r"label=([^&]+)"),
            "input_ref": st.column_config.TextColumn("input"),
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
    _enable_jobs_auto_refresh(rows, "openfe")
    if not rows:
        st.info("No OpenFE/ABFE/RBFE jobs found yet.")
        return
    df = pd.DataFrame(rows)
    table = df[
        [
            "job_code", "source_job_code", "workflow", "pdb_id", "ligand_id",
            "preset", "prod_ns", "eq_ns", "repeats", "timeout_h",
            "status", "stage", "progress_pct", "current_step", "total_steps", "steps_remaining", "eta_h_remaining", "status_message",
            "created_at", "completed_at", "success", "workflow_key", "run_id"
        ]
    ].copy()
    table["job_code_link"] = table.apply(
        lambda r: (
            f"./openfe-results?run_type={r['workflow_key']}&run_id={r['run_id']}&job_code={r['job_code']}"
        ),
        axis=1,
    )
    st.data_editor(
        table[
            [
                "job_code_link", "source_job_code", "workflow", "pdb_id", "ligand_id",
                "preset", "prod_ns", "eq_ns", "repeats", "timeout_h",
                "status", "stage", "progress_pct", "current_step", "total_steps", "steps_remaining", "eta_h_remaining", "status_message",
                "created_at", "completed_at", "success"
            ]
        ],
        hide_index=True,
        use_container_width=True,
        disabled=[
            "job_code_link", "source_job_code", "workflow", "pdb_id", "ligand_id",
            "preset", "prod_ns", "eq_ns", "repeats", "timeout_h",
            "status", "stage", "progress_pct", "current_step", "total_steps", "steps_remaining", "eta_h_remaining", "status_message",
            "created_at", "completed_at", "success"
        ],
        column_config={
            "job_code_link": st.column_config.LinkColumn("job_code", display_text=r".*job_code=([^&]+).*"),
            "source_job_code": st.column_config.TextColumn("source_structure_job"),
            "prod_ns": st.column_config.NumberColumn("prod_ns"),
            "eq_ns": st.column_config.NumberColumn("eq_ns"),
            "timeout_h": st.column_config.NumberColumn("timeout_h"),
            "progress_pct": st.column_config.ProgressColumn("progress_%", min_value=0, max_value=100),
            "current_step": st.column_config.NumberColumn("step"),
            "total_steps": st.column_config.NumberColumn("total_steps"),
            "steps_remaining": st.column_config.NumberColumn("steps_left"),
            "eta_h_remaining": st.column_config.NumberColumn("eta_h"),
            "status_message": st.column_config.TextColumn("message"),
        },
        key="jobs_openfe_table_editor",
    )
    st.caption("`job_code` is the OpenFE run code. `source_structure_job` shows the originating structure-preparation job.")


def render_admet_jobs() -> None:
    st.title("Jobs – ADMET")
    rows = _collect_admet_jobs()
    _enable_jobs_auto_refresh(rows, "admet")
    if not rows:
        st.info("No ADMET jobs found yet.")
        return
    df = pd.DataFrame(rows)
    table = df[["job_code", "status", "source_job_code", "input_smiles", "created_at", "completed_at", "success", "run_id"]].copy()
    table["job_code_link"] = table.apply(
        lambda r: f"./admet-results?run_id={r['run_id']}&label={r['job_code']}",
        axis=1,
    )
    table["delete"] = False
    edited_rows = st.data_editor(
        table[["delete", "job_code_link", "status", "source_job_code", "input_smiles", "created_at", "completed_at", "success"]],
        hide_index=True,
        use_container_width=True,
        disabled=["job_code_link", "status", "source_job_code", "input_smiles", "created_at", "completed_at", "success"],
        column_config={
            "delete": st.column_config.CheckboxColumn("delete", help="Select ADMET run for deletion"),
            "job_code_link": st.column_config.LinkColumn("job_code", display_text=r"label=([^&]+)"),
            "source_job_code": st.column_config.TextColumn("source_structure_job"),
        },
        key="jobs_admet_table_editor",
    )
    selected_indices = edited_rows.index[edited_rows["delete"] == True].tolist()  # noqa: E712
    selected_for_delete = table.iloc[selected_indices]["run_id"].astype(str).tolist()
    st.caption("Use `job_code` to inspect ADMET results. Select rows with `delete` to remove runs.")
    confirm_delete = st.checkbox(
        "I understand this permanently deletes selected ADMET job folders",
        value=False,
        key="jobs_delete_admet_confirm_checkbox",
    )
    if st.button(
        "Delete selected ADMET jobs",
        type="secondary",
        disabled=(not selected_for_delete) or (not confirm_delete),
        key="jobs_delete_admet_selected_button",
    ):
        deleted: list[str] = []
        failed: list[str] = []
        for run_id in selected_for_delete:
            run_dir = _run_root() / "admet" / run_id
            try:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                deleted.append(run_id)
            except Exception as exc:
                failed.append(f"{run_id}: {exc}")
        if deleted:
            st.success(f"Deleted {len(deleted)} ADMET run(s).")
        if failed:
            st.error("Some runs could not be deleted:\n" + "\n".join(failed))
        st.rerun()


def render_qc_jobs() -> None:
    st.title("Jobs – QC")
    rows = _collect_qc_jobs()
    _enable_jobs_auto_refresh(rows, "qc")
    if not rows:
        st.info("No QC jobs found yet.")
        return
    df = pd.DataFrame(rows)
    table = df[
        ["job_code", "status", "source_job_code", "method", "basis", "charge", "multiplicity", "created_at", "completed_at", "success", "run_id"]
    ].copy()
    table["job_code_link"] = table.apply(
        lambda r: f"./qc-results?run_id={r['run_id']}&label={r['job_code']}",
        axis=1,
    )
    st.data_editor(
        table[["job_code_link", "status", "source_job_code", "method", "basis", "charge", "multiplicity", "created_at", "completed_at", "success"]],
        hide_index=True,
        use_container_width=True,
        disabled=["job_code_link", "status", "source_job_code", "method", "basis", "charge", "multiplicity", "created_at", "completed_at", "success"],
        column_config={
            "job_code_link": st.column_config.LinkColumn("job_code", display_text=r"label=([^&]+)"),
            "source_job_code": st.column_config.TextColumn("source_structure_job"),
        },
        key="jobs_qc_table_editor",
    )
