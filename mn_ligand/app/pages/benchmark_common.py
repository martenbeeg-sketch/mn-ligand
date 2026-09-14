from __future__ import annotations

from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.viewers import render_persistent_3dmol
from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records
from mn_ligand.core.provenance import target_lineage_summary
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.benchmark_datasets import BenchmarkCase
from mn_ligand.workflows.benchmark_datasets import (
    list_benchmark_dataset_jobs,
    load_benchmark_dataset,
)


def benchmark_value(job: JobRecord, key: str):
    value = job.metadata.get(key)
    if value not in {None, ""}:
        return value
    parameters = job.metadata.get("parameters")
    context = parameters.get("context") if isinstance(parameters, dict) else None
    return context.get(key) if isinstance(context, dict) else None


def campaign_jobs(dataset_run_id: str) -> list[JobRecord]:
    return [
        job
        for job in iter_job_records(runs_root())
        if str(benchmark_value(job, "benchmark_dataset_run_id") or "")
        == dataset_run_id
    ]


def visible_dataset_jobs() -> list[JobRecord]:
    """Hide scientifically superseded imports while preserving their run folders."""
    jobs = list_benchmark_dataset_jobs()
    superseded = {
        str(load_benchmark_dataset(job).get("source_provenance", {}).get(
            "supersedes_dataset_run_id", ""
        ))
        for job in jobs
    }
    return [job for job in jobs if job.run_id not in superseded]


def campaign_table(dataset_run_id: str) -> pd.DataFrame:
    rows = []
    all_jobs = list(iter_job_records(runs_root()))
    jobs_by_id = {job.run_id: job for job in all_jobs}
    selected = [
        job
        for job in all_jobs
        if str(benchmark_value(job, "benchmark_dataset_run_id") or "")
        == dataset_run_id
    ]
    for job in selected:
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        context = target_lineage_summary(job, jobs_by_id)
        rows.append(
            {
                "Result": "./job-results?"
                + urlencode(
                    {
                        "task_group": job.task_group,
                        "run_id": job.run_id,
                        "label": code,
                    }
                ),
                "Last step": context["last_step"],
                "Job": code,
                "Target": context["target"],
                "Receptor": context["receptor"],
                "Ligand": context["ligand"],
                "Origin / history": context["origin"],
                "Mode": benchmark_value(job, "benchmark_mode") or "",
                "Case": benchmark_value(job, "benchmark_case_id") or "",
                "Engine": job.tool or job.metadata.get("engine", ""),
                "Status": job.status,
                "Created": job.created_at,
            }
        )
    return pd.DataFrame(rows)


def render_case_viewer(case: BenchmarkCase, *, key: str) -> None:
    try:
        import py3Dmol
    except ImportError:
        st.warning("py3Dmol is unavailable; receptor and ligand cannot be displayed.")
        return
    viewer = py3Dmol.view(width=1100, height=620)
    viewer.addModel(case.receptor_path.read_text(errors="replace"), "pdb")
    viewer.setStyle(
        {"model": 0},
        {
            "cartoon": {"color": "lightblue", "opacity": 0.8},
            "line": {"color": "slategray", "opacity": 0.45},
        },
    )
    viewer.addModel(case.ligand_path.read_text(errors="replace"), "sdf")
    viewer.setStyle(
        {"model": 1},
        {"stick": {"colorscheme": "magentaCarbon", "radius": 0.24}},
    )
    viewer.addStyle(
        {"model": 1},
        {"sphere": {"colorscheme": "magentaCarbon", "scale": 0.28}},
    )
    viewer.zoomTo({"model": 1})
    viewer.zoom(0.75)
    render_persistent_3dmol(viewer, key=key, height=640)
    st.caption(
        "Reference receptor is blue/grey; the crystallographic reference ligand "
        "is magenta. The camera is centered on the ligand."
    )
