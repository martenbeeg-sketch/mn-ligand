from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.viewers import render_persistent_3dmol
from mn_ligand.core.jobs import display_job_code, iter_job_records
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.molecule_design_results import (
    create_molecule_design_selection,
    target_design_campaigns,
    target_design_compounds,
)


NUMERIC_COLUMNS = (
    "qed",
    "molecular_weight",
    "sa_score",
    "logp",
    "hbond_donors",
    "hbond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "heavy_atom_count",
    "formal_charge",
)


def _target_options() -> dict[str, str]:
    jobs = list(iter_job_records(runs_root()))
    targets = {
        str(
            job.metadata.get("target_run_id")
            or job.parent_run_id
            or ""
        )
        for job in jobs
        if job.workflow == "molecule_generation_campaign"
    }
    options: dict[str, str] = {}
    by_id = {job.run_id: job for job in jobs}
    for target_run_id in sorted(targets):
        target = by_id.get(target_run_id)
        if target is None:
            label = target_run_id
        else:
            code = display_job_code(
                target.metadata.get("job_code"),
                target.run_id,
            )
            target_name = str(
                target.metadata.get("pdb_id")
                or target.metadata.get("target_name")
                or target.tool
                or "prepared target"
            )
            label = f"{code} | {target_name}"
        options[label] = target_run_id
    return options


def _numeric_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _range_filter(
    frame: pd.DataFrame,
    column: str,
    label: str,
    *,
    key: str,
) -> pd.Series:
    values = frame[column].dropna()
    if values.empty:
        st.caption(f"{label}: unavailable")
        return pd.Series(True, index=frame.index)
    minimum = float(values.min())
    maximum = float(values.max())
    if abs(maximum - minimum) < 1e-12:
        st.caption(f"{label}: {minimum:.3g}")
        return pd.Series(True, index=frame.index)
    selected = st.slider(
        label,
        min_value=minimum,
        max_value=maximum,
        value=(minimum, maximum),
        key=key,
    )
    return frame[column].between(selected[0], selected[1], inclusive="both")


def _candidate_path(row: pd.Series) -> Path | None:
    qualification_run_id = str(row.get("qualification_run_id") or "")
    relative = str(row.get("candidate_relative_path") or "")
    if not qualification_run_id or not relative:
        return None
    run_dir = (
        runs_root()
        / "molecule-qualification"
        / qualification_run_id
    ).resolve()
    candidate = (run_dir / relative).resolve()
    try:
        candidate.relative_to(run_dir)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _render_candidate_viewer(frame: pd.DataFrame, target_run_id: str) -> None:
    candidates = frame.loc[frame["candidate_available"].astype(bool)].copy()
    if candidates.empty:
        st.info("No standardized 3D candidate is available in this filter scope.")
        return
    labels = [
        (
            f"{row['compound_id']} | {row['engine']} | "
            f"{str(row['qualification_status']).replace('_', ' ')}"
        )
        for _, row in candidates.iterrows()
    ]
    selected_label = st.selectbox(
        "Preview compound",
        labels,
        key=f"design_result_preview_{target_run_id}",
    )
    row = candidates.iloc[labels.index(selected_label)]
    path = _candidate_path(row)
    if path is None:
        st.warning("The selected standardized conformer is unavailable.")
        return
    metrics = st.columns(5)
    metrics[0].metric("Engine", str(row.get("engine") or "—"))
    metrics[1].metric("Status", str(row["qualification_status"]).replace("_", " ").title())
    metrics[2].metric(
        "QED",
        f"{float(row['qed']):.3f}" if pd.notna(row.get("qed")) else "—",
    )
    metrics[3].metric(
        "MW",
        f"{float(row['molecular_weight']):.1f} Da"
        if pd.notna(row.get("molecular_weight"))
        else "—",
    )
    metrics[4].metric(
        "SA",
        f"{float(row['sa_score']):.2f}"
        if pd.notna(row.get("sa_score"))
        else "—",
    )
    warning = str(row.get("review_warnings") or "")
    if warning:
        st.warning(f"Review warning: {warning}")
    failures = str(
        row.get("chemical_failures")
        or row.get("geometry_failures")
        or ""
    )
    if (
        str(row.get("qualification_status") or "") == "rejected"
        and failures
    ):
        st.error(f"Hard exclusion: {failures}")
    smiles = str(row.get("canonical_isomeric_smiles") or "")
    if smiles:
        st.caption("Canonical stereochemistry-aware SMILES")
        st.code(smiles, language=None)
    try:
        import py3Dmol

        viewer = py3Dmol.view(width=1100, height=620)
        viewer.addModel(path.read_text(errors="replace"), "sdf")
        viewer.setStyle(
            {"model": 0},
            {"stick": {"colorscheme": "cyanCarbon", "radius": 0.22}},
        )
        viewer.zoomTo({"model": 0})
        viewer.zoom(0.82)
        render_persistent_3dmol(
            viewer,
            key=f"design-summary:{target_run_id}:{row['row_id']}",
            height=640,
        )
    except Exception as exc:
        st.error(f"Compound preview failed: {exc}")
    st.caption(
        "The viewer shows the standardized molecule-only conformer. It does not "
        "represent a docked pose in the selected target."
    )


def render() -> None:
    st.title("Molecule Design Results")
    st.caption(
        "Compare every generation engine associated with one prepared target, "
        "filter qualified chemistry, inspect standardized 3D candidates, and "
        "promote an immutable compound dataset for Docking / Cofolding."
    )
    options = _target_options()
    requested_target = str(
        st.query_params.get("target_run_id", "")
        or st.query_params.get("run_id", "")
        or ""
    ).strip()
    if requested_target:
        target_run_id = requested_target
    elif options:
        selected = st.selectbox("Prepared target", list(options))
        target_run_id = options[selected]
    else:
        st.info("No prepared target has a molecule-design campaign yet.")
        return

    jobs = list(iter_job_records(runs_root()))
    by_id = {job.run_id: job for job in jobs}
    target = by_id.get(target_run_id)
    campaigns = target_design_campaigns(target_run_id, jobs=jobs)
    rows = target_design_compounds(target_run_id, jobs=jobs)
    target_code = (
        display_job_code(target.metadata.get("job_code"), target.run_id)
        if target is not None
        else target_run_id[:8]
    )
    header = st.columns(4)
    header[0].metric("Prepared target", target_code)
    header[1].metric("Design campaigns", len(campaigns))
    header[2].metric(
        "Generation engines",
        len({str(row.get("engine") or "") for row in rows}),
    )
    header[3].metric(
        "Latest policy",
        max(
            (int(row.get("qualification_policy_version") or 0) for row in rows),
            default=0,
        ),
    )
    if target is not None:
        st.link_button(
            "Open target structure",
            (
                "./structure-results?"
                + urlencode(
                    {
                        "run_id": target.run_id,
                        "label": target_code,
                    }
                )
            ),
        )
    if not rows:
        st.info(
            "No completed molecule-qualification results are linked to this "
            "prepared target."
        )
        return

    frame = _numeric_frame(rows)
    accepted = frame["accepted_for_docking"].astype(bool)
    summary = st.columns(5)
    summary[0].metric("Generated records", len(frame))
    summary[1].metric("Accepted", int(accepted.sum()))
    summary[2].metric(
        "Unique accepted",
        int(
            frame.loc[accepted, "canonical_isomeric_smiles"]
            .replace("", pd.NA)
            .nunique()
        ),
    )
    summary[3].metric(
        "Strict passes",
        int(frame["strict_posebusters_pass"].astype(bool).sum()),
    )
    summary[4].metric(
        "Accepted with warning",
        int(
            frame["qualification_status"]
            .astype(str)
            .eq("qualified_with_warning")
            .sum()
        ),
    )

    status_counts = (
        frame.groupby(["engine", "qualification_status"])
        .size()
        .rename("Compounds")
        .reset_index()
    )
    plot_columns = st.columns(2)
    with plot_columns[0]:
        st.markdown("#### Qualification by engine")
        pivot = status_counts.pivot(
            index="engine",
            columns="qualification_status",
            values="Compounds",
        ).fillna(0)
        st.bar_chart(pivot, height=330)
    with plot_columns[1]:
        st.markdown("#### QED versus molecular weight")
        plot_frame = frame.loc[
            frame["qed"].notna() & frame["molecular_weight"].notna(),
            ["qed", "molecular_weight", "engine", "heavy_atom_count"],
        ].rename(
            columns={
                "qed": "QED",
                "molecular_weight": "Molecular weight",
                "engine": "Engine",
                "heavy_atom_count": "Heavy atoms",
            }
        )
        if plot_frame.empty:
            st.info("QED/MW values are unavailable.")
        else:
            st.scatter_chart(
                plot_frame,
                x="Molecular weight",
                y="QED",
                color="Engine",
                size="Heavy atoms",
                height=330,
            )

    st.markdown("### Filter and select compounds")
    filter_columns = st.columns(2)
    with filter_columns[0]:
        engines = sorted(frame["engine"].astype(str).unique())
        selected_engines = st.multiselect(
            "Engines",
            engines,
            default=engines,
            key=f"design_engines_{target_run_id}",
        )
        statuses = sorted(
            frame["qualification_status"].astype(str).unique()
        )
        accepted_statuses = [
            status
            for status in statuses
            if status in {"qualified", "qualified_with_warning"}
        ]
        selected_statuses = st.multiselect(
            "Qualification status",
            statuses,
            default=accepted_statuses or statuses,
            key=f"design_statuses_{target_run_id}",
        )
    with filter_columns[1]:
        require_3d = st.checkbox(
            "Require standardized 3D",
            value=True,
            key=f"design_require_3d_{target_run_id}",
        )
        strict_only = st.checkbox(
            "Strict PoseBusters pass only",
            value=False,
            key=f"design_strict_only_{target_run_id}",
        )

    mask = (
        frame["engine"].astype(str).isin(selected_engines)
        & frame["qualification_status"].astype(str).isin(selected_statuses)
    )
    if require_3d:
        mask &= frame["candidate_available"].astype(bool)
    if strict_only:
        mask &= frame["strict_posebusters_pass"].astype(bool)
    numeric_columns = st.columns(4)
    with numeric_columns[0]:
        mask &= _range_filter(
            frame, "qed", "QED", key=f"design_qed_{target_run_id}"
        )
    with numeric_columns[1]:
        mask &= _range_filter(
            frame,
            "molecular_weight",
            "Molecular weight (Da)",
            key=f"design_mw_{target_run_id}",
        )
    with numeric_columns[2]:
        mask &= _range_filter(
            frame,
            "sa_score",
            "Synthetic accessibility",
            key=f"design_sa_{target_run_id}",
        )
    with numeric_columns[3]:
        mask &= _range_filter(
            frame,
            "logp",
            "cLogP",
            key=f"design_logp_{target_run_id}",
        )

    filtered = frame.loc[mask].reset_index(drop=True)
    st.caption(
        f"{len(filtered)} of {len(frame)} records match the current filters. "
        "Select one or more table rows to create a downstream dataset."
    )
    display_columns = [
        "compound_id",
        "engine",
        "campaign",
        "qualification_status",
        "qed",
        "molecular_weight",
        "sa_score",
        "logp",
        "hbond_donors",
        "hbond_acceptors",
        "rotatable_bonds",
        "ring_count",
        "heavy_atom_count",
        "formal_charge",
        "review_warnings",
        "canonical_isomeric_smiles",
    ]
    table_event = st.dataframe(
        filtered[display_columns],
        hide_index=True,
        width="stretch",
        height=min(680, 38 + max(1, len(filtered)) * 35),
        key=f"design_result_table_{target_run_id}",
        on_select="rerun",
        selection_mode="multi-row",
    )
    selected_indices = [
        int(index)
        for index in table_event.selection.rows
        if 0 <= int(index) < len(filtered)
    ]
    selected_rows = [
        filtered.iloc[index].to_dict()
        for index in selected_indices
    ]
    selected_accepted = [
        row for row in selected_rows if bool(row["accepted_for_docking"])
    ]
    if len(selected_accepted) != len(selected_rows):
        st.warning(
            "Rejected rows can be inspected but cannot enter a docking/cofolding "
            "dataset."
        )

    action_columns = st.columns([2, 1])
    with action_columns[0]:
        dataset_name = st.text_input(
            "Dataset name",
            value=f"{target_code} molecule-design selection",
            key=f"design_dataset_name_{target_run_id}",
        )
    with action_columns[1]:
        st.metric("Selected accepted rows", len(selected_accepted))
    if st.button(
        "Create dataset for Docking / Cofolding",
        type="primary",
        disabled=not selected_accepted,
        key=f"create_design_dataset_{target_run_id}",
    ):
        try:
            selection = create_molecule_design_selection(
                target_run_id=target_run_id,
                selected_rows=selected_accepted,
                name=dataset_name,
            )
        except Exception as exc:
            st.error(f"Could not create the compound dataset: {exc}")
        else:
            count = int(selection.result.get("compound_count") or 0)
            st.success(
                f"Created immutable dataset "
                f"{display_job_code(selection.metadata.get('job_code'), selection.run_id)} "
                f"with {count} unique compound(s)."
            )
            links = st.columns(2)
            with links[0]:
                st.link_button(
                    "Open dataset job",
                    (
                        "./job-results?"
                        + urlencode(
                            {
                                "task_group": selection.task_group,
                                "run_id": selection.run_id,
                            }
                        )
                    ),
                )
            with links[1]:
                st.link_button(
                    "Continue to Docking / Cofolding",
                    (
                        "./discover-docking?"
                        + urlencode(
                            {
                                "compound_run_id": selection.run_id,
                                "target_run_id": target_run_id,
                            }
                        )
                    ),
                    type="primary",
                )

    st.markdown("### Compound viewer")
    _render_candidate_viewer(filtered, target_run_id)


render()
