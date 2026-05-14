from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from mn_ligand.app.pages.bound_ligand_md import _run_root


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _infer_status(metadata: dict[str, Any], result: dict[str, Any], service_job: dict[str, Any]) -> str:
    status = str(metadata.get("status") or "").strip().lower()
    if status:
        return status
    service_status = str(service_job.get("status") or "").strip().lower()
    if service_status:
        return service_status
    if result:
        if bool(result.get("success")):
            return "completed"
        return "failed"
    return "unknown"


def _render_abfe_summary(result: dict[str, Any]) -> None:
    payload = result.get("result") if isinstance(result.get("result"), dict) else {}
    final = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    dg = final.get("binding_free_energy_kcal_mol")
    c1, c2 = st.columns(2)
    with c1:
        if dg is None:
            st.metric("ΔG_bind", "n/a")
        else:
            st.metric("ΔG_bind", f"{float(dg):.3f} kcal/mol")
    with c2:
        st.markdown(
            "ABFE meaning: this is the absolute binding free energy estimate from solvent + complex legs. "
            "More negative usually indicates stronger binding."
        )


def render() -> None:
    st.title("OpenFE Results")
    qp = st.query_params
    run_id = str(qp.get("run_id", "")).strip()
    run_type = str(qp.get("run_type", "abfe")).strip().lower()
    run_subdir = "rbfe" if run_type == "rbfe" else "abfe"

    if not run_id:
        st.warning("Missing run_id")
        if st.button("Back to Jobs – OpenFE"):
            st.switch_page("app/pages/jobs_openfe.py")
        return

    run_dir = _run_root() / run_subdir / run_id
    if not run_dir.exists():
        st.error(f"Run not found: {run_dir}")
        if st.button("Back to Jobs – OpenFE"):
            st.switch_page("app/pages/jobs_openfe.py")
        return

    metadata = _read_json(run_dir / "metadata.json")
    result = _read_json(run_dir / "result.json")
    service_job = _read_json(run_dir / "jobs" / f"{run_id}.json")
    run_inputs = metadata.get("run_inputs") if isinstance(metadata.get("run_inputs"), dict) else {}
    status = _infer_status(metadata, result, service_job)

    st.caption(f"Run: {run_id}")
    st.caption(f"Workflow: {run_subdir.upper()}")
    st.caption(f"OpenFE job code: {metadata.get('job_code') or run_id[:3].upper()}")
    if metadata.get("source_job_code"):
        st.caption(f"Source structure job: {metadata.get('source_job_code')}")

    if status == "completed":
        st.success("Status: completed")
    elif status == "running":
        st.info("Status: running")
    elif status == "queued":
        st.info("Status: queued")
    elif status == "failed":
        st.error("Status: failed")
    else:
        st.warning(f"Status: {status}")

    st.markdown("### Key Results")
    if run_subdir == "abfe":
        _render_abfe_summary(result)
    else:
        rbfe_result = result.get("result") if isinstance(result.get("result"), dict) else {}
        st.json(rbfe_result if rbfe_result else result)

    if status != "completed":
        err = (result.get("error") if isinstance(result, dict) else None) or service_job.get("error")
        if err:
            st.markdown("### Error")
            st.code(str(err))

    st.markdown("### Run Setup")
    setup_rows = [
        {"parameter": "preset", "value": run_inputs.get("preset", "")},
        {"parameter": "pdb_id", "value": metadata.get("pdb_id", "")},
        {"parameter": "ligand_key", "value": metadata.get("ligand_key", "")},
        {"parameter": "production_length_ns", "value": run_inputs.get("production_length_ns", "")},
        {"parameter": "equilibration_length_ns", "value": run_inputs.get("equilibration_length_ns", "")},
        {"parameter": "protocol_repeats", "value": run_inputs.get("protocol_repeats", "")},
        {"parameter": "charge_method", "value": run_inputs.get("charge_method", "")},
        {"parameter": "ligand_forcefield", "value": run_inputs.get("ligand_forcefield", "")},
        {"parameter": "solvent_model", "value": run_inputs.get("solvent_model", "")},
        {"parameter": "temperature_k", "value": run_inputs.get("temperature_k", "")},
        {"parameter": "pressure_bar", "value": run_inputs.get("pressure_bar", "")},
        {"parameter": "hmr", "value": run_inputs.get("hmr", "")},
        {"parameter": "timestep_fs", "value": run_inputs.get("timestep_fs", "")},
        {"parameter": "timeout_budget_hours_est", "value": run_inputs.get("timeout_budget_hours_est", "")},
    ]
    if run_subdir == "rbfe":
        setup_rows.extend(
            [
                {"parameter": "n_ligands", "value": run_inputs.get("n_ligands", "")},
                {"parameter": "network_topology", "value": run_inputs.get("network_topology", "")},
                {"parameter": "atom_mapper", "value": run_inputs.get("atom_mapper", "")},
                {"parameter": "lambda_windows", "value": run_inputs.get("lambda_windows", "")},
            ]
        )
    st.dataframe(setup_rows, hide_index=True, use_container_width=True)

    st.markdown("### Artifacts")
    st.code(str(run_dir))
    files = sorted([p for p in run_dir.rglob("*") if p.is_file()], key=lambda p: str(p))
    if files:
        st.dataframe(
            [{"file": str(p.relative_to(run_dir)), "size_kb": round(p.stat().st_size / 1024.0, 2)} for p in files],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No files found yet.")

    if st.button("Back to Jobs – OpenFE"):
        st.switch_page("app/pages/jobs_openfe.py")


render()
