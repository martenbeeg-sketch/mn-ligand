from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.results_explorer_index import build_results_index
from mn_ligand.runtime import runs_root


LINK_COLUMNS = {
    "prediction_result": st.column_config.LinkColumn("Prediction", display_text="Open"),
    "posebusters": st.column_config.LinkColumn("PoseBusters", display_text="Open"),
    "plip": st.column_config.LinkColumn("PLIP", display_text="Open"),
    "pandamap": st.column_config.LinkColumn("PandaMap", display_text="Open"),
}

RESULTS_INDEX_SCHEMA_VERSION = 2


def _revision() -> tuple[int, int]:
    root = runs_root()
    metadata = list(root.glob("*/*/metadata.json"))
    return len(metadata), max((path.stat().st_mtime_ns for path in metadata), default=0)


@st.cache_data(show_spinner="Indexing result provenance…")
def _load_index(
    root: str, revision: tuple[int, int], schema_version: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del revision, schema_version
    return build_results_index(Path(root))


def _filter(
    frame: pd.DataFrame,
    *,
    datasets: list[str] | None = None,
    targets: list[str] | None = None,
    compounds: list[str] | None = None,
) -> pd.DataFrame:
    filtered = frame
    if datasets:
        filtered = filtered.loc[filtered["dataset_id"].isin(datasets)]
    if targets:
        filtered = filtered.loc[filtered["target_family_id"].isin(targets)]
    if compounds and "compound_id" in filtered:
        filtered = filtered.loc[filtered["compound_id"].isin(compounds)]
    return filtered.copy()


def _result_table(frame: pd.DataFrame, *, include_compound: bool = False) -> None:
    if frame.empty:
        st.info("No matching prediction or evaluation results.")
        return
    grouping = [
        *(["compound_id"] if include_compound else []),
        "prediction_engine",
        "result_kind",
    ]
    summary_rows: list[dict[str, object]] = []
    for keys, rows in frame.groupby(grouping, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ordered = rows.sort_values("created", ascending=False)
        latest = ordered.iloc[0]
        summary: dict[str, object] = dict(zip(grouping, keys))
        statuses = ordered["status"].astype(str).unique().tolist()
        summary["runs"] = len(ordered)
        summary["status"] = (
            "completed"
            if statuses == ["completed"]
            else ", ".join(statuses)
        )
        summary["prediction_result"] = latest["prediction_result"]
        for column in ("posebusters", "plip", "pandamap"):
            available = ordered.loc[ordered[column].astype(str).str.strip().ne("")]
            summary[column] = available.iloc[0][column] if not available.empty else ""
        summary["latest_created"] = latest["created"]
        summary_rows.append(summary)
    summarized = pd.DataFrame(summary_rows)
    columns = [
        *(["compound_id"] if include_compound else []),
        "prediction_engine",
        "result_kind",
        "runs",
        "status",
        "prediction_result",
        "posebusters",
        "plip",
        "pandamap",
        "latest_created",
    ]
    st.dataframe(
        summarized[columns].sort_values(
            ["prediction_engine", "latest_created"], ascending=[True, False]
        ),
        hide_index=True,
        width="stretch",
        column_config={
            **LINK_COLUMNS,
            "prediction_result": st.column_config.LinkColumn(
                "Latest prediction", display_text="Open"
            ),
            "runs": st.column_config.NumberColumn("Runs"),
            "latest_created": st.column_config.TextColumn("Latest run"),
        },
    )
    if len(summarized) < len(frame):
        with st.expander(f"Individual job runs ({len(frame)})"):
            st.dataframe(
                frame[
                    [
                        *(["compound_id"] if include_compound else []),
                        "prediction_engine",
                        "result_kind",
                        "prediction_job",
                        "status",
                        "prediction_result",
                        "posebusters",
                        "plip",
                        "pandamap",
                        "created",
                    ]
                ].sort_values(
                    ["prediction_engine", "created"], ascending=[True, False]
                ),
                hide_index=True,
                width="stretch",
                column_config=LINK_COLUMNS,
            )


def _campaign_link(dataset_id: str) -> str:
    return (
        "./compound-campaign-comparison?"
        + urlencode({"dataset_run_id": dataset_id})
    )


def _analysis_sets(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_dir = root / "campaign-comparison-collections"
    if not group_dir.is_dir():
        return pd.DataFrame()
    for run_dir in group_dir.iterdir():
        try:
            metadata = json.loads((run_dir / "metadata.json").read_text())
            selection = json.loads((run_dir / "selection.json").read_text())
        except (OSError, ValueError, TypeError):
            continue
        run_id = str(metadata.get("run_id") or run_dir.name)
        rows.append(
            {
                "open": "./compound-campaign-comparison?"
                + urlencode({"analysis_set_id": run_id}),
                "name": str(metadata.get("name") or "Analysis Set"),
                "description": str(metadata.get("description") or ""),
                "dataset": str(
                    selection.get("dataset")
                    or selection.get("dataset_run_id")
                    or ""
                ),
                "targets": len(selection.get("target_run_ids") or []),
                "engines": ", ".join(selection.get("engines") or []),
                "created": str(metadata.get("created_at") or ""),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("created", ascending=False)


def _analysis_set_view(root: Path) -> None:
    sets = _analysis_sets(root)
    if sets.empty:
        st.info(
            "No Analysis Sets yet. Create one from Data & Analysis Sets on a "
            "campaign comparison page."
        )
        return
    st.caption(
        "Analysis Sets reopen an exact multi-campaign comparison scope. They "
        "are analysis definitions and remain separate from physical campaigns."
    )
    st.dataframe(
        sets,
        hide_index=True,
        width="stretch",
        column_config={
            "open": st.column_config.LinkColumn("Open", display_text="Analyze"),
            "name": "Analysis Set",
            "created": st.column_config.DatetimeColumn(
                "Created", format="YYYY-MM-DD HH:mm"
            ),
        },
    )


def _target_provenance(rows: pd.DataFrame) -> None:
    row = rows.iloc[0]
    source = str(row.get("target_origin") or row.get("target_family") or "Unknown source")
    artifact = str(
        row.get("target_artifact")
        or row.get("target_variant")
        or "Prepared target"
    )
    history = str(
        row.get("target_history") or "No recorded preparation history"
    )
    st.caption(f"Source: {source}  ·  Artifact: {artifact}")
    st.caption(f"Preparation history: {history}")
    if str(row.get("target_result") or "").strip():
        st.link_button("Open prepared-target provenance", str(row["target_result"]))


def _overview(predictions: pd.DataFrame, compounds: pd.DataFrame) -> None:
    metrics = st.columns(4)
    metrics[0].metric("Datasets", predictions["dataset_id"].nunique())
    metrics[1].metric("Target families", predictions["target_family_id"].nunique())
    metrics[2].metric("Prediction jobs", len(predictions))
    metrics[3].metric(
        "Compounds indexed",
        compounds["compound_id"].nunique() if not compounds.empty else 0,
    )
    status = (
        predictions.groupby(["prediction_engine", "status"], dropna=False)
        .size()
        .rename("jobs")
        .reset_index()
    )
    st.markdown("#### Result coverage")
    st.dataframe(status, hide_index=True, width="stretch")


def _dataset_view(predictions: pd.DataFrame, compounds: pd.DataFrame) -> None:
    options = (
        predictions[["dataset_id", "dataset"]]
        .drop_duplicates()
        .sort_values("dataset")
    )
    selected = st.multiselect(
        "Compound datasets",
        options["dataset_id"].tolist(),
        default=options["dataset_id"].tolist()[:1],
        format_func=dict(zip(options["dataset_id"], options["dataset"])).get,
        key="results-explorer-datasets",
    )
    filtered = _filter(predictions, datasets=selected)
    for dataset_id, dataset_rows in filtered.groupby("dataset_id", sort=False):
        label = str(dataset_rows["dataset"].iloc[0])
        with st.expander(
            f"{label} · {dataset_rows['target_family_id'].nunique()} target(s) · "
            f"{len(dataset_rows)} prediction job(s)",
            expanded=len(selected) == 1,
        ):
            st.link_button(
                "Open campaign comparison",
                _campaign_link(str(dataset_id)),
            )
            for target, target_rows in dataset_rows.groupby("target_family", sort=False):
                st.markdown(f"##### {target}")
                for variant, variant_rows in target_rows.groupby(
                    "target_variant", sort=False
                ):
                    with st.expander(
                        f"{variant} · {variant_rows['campaign_id'].nunique()} campaign(s)"
                    ):
                        _target_provenance(variant_rows)
                        for campaign, campaign_rows in variant_rows.groupby(
                            "campaign", sort=False
                        ):
                            with st.expander(
                                f"{campaign} · {len(campaign_rows)} prediction job(s)",
                                expanded=variant_rows["campaign"].nunique() == 1,
                            ):
                                _result_table(campaign_rows)


def _target_view(predictions: pd.DataFrame, compounds: pd.DataFrame) -> None:
    del compounds
    options = (
        predictions[["target_family_id", "target_family"]]
        .drop_duplicates()
        .sort_values("target_family")
    )
    selected = st.multiselect(
        "Biological targets",
        options["target_family_id"].tolist(),
        default=options["target_family_id"].tolist()[:1],
        format_func=dict(zip(options["target_family_id"], options["target_family"])).get,
        key="results-explorer-targets",
    )
    filtered = _filter(predictions, targets=selected)
    for family, family_rows in filtered.groupby("target_family", sort=False):
        st.markdown(f"### {family}")
        coverage = (
            family_rows.groupby(
                [
                    "dataset",
                    "target_variant",
                    "campaign",
                    "prediction_engine",
                    "status",
                ],
                dropna=False,
            )
            .size()
            .rename("jobs")
            .reset_index()
        )
        st.dataframe(coverage, hide_index=True, width="stretch")
        with st.expander("Open target results", expanded=True):
            for variant, variant_rows in family_rows.groupby(
                "target_variant", sort=False
            ):
                st.markdown(f"#### {variant}")
                _target_provenance(variant_rows)
                for campaign, campaign_rows in variant_rows.groupby(
                    "campaign", sort=False
                ):
                    with st.expander(
                        f"{campaign} · {len(campaign_rows)} prediction job(s)",
                        expanded=variant_rows["campaign"].nunique() == 1,
                    ):
                        _result_table(campaign_rows)


def _compound_view(predictions: pd.DataFrame, compounds: pd.DataFrame) -> None:
    if compounds.empty:
        st.info("No compound-level results could be indexed.")
        return
    datasets = (
        compounds[["dataset_id", "dataset"]]
        .drop_duplicates()
        .sort_values("dataset")
    )
    dataset_id = st.selectbox(
        "Compound dataset",
        datasets["dataset_id"].tolist(),
        format_func=dict(zip(datasets["dataset_id"], datasets["dataset"])).get,
        key="results-explorer-compound-dataset",
    )
    available = compounds.loc[compounds["dataset_id"].eq(dataset_id)]
    selected = st.multiselect(
        "Compounds",
        sorted(available["compound_id"].unique().tolist()),
        default=sorted(available["compound_id"].unique().tolist())[:1],
        key="results-explorer-compounds",
    )
    filtered = _filter(compounds, datasets=[dataset_id], compounds=selected)
    for compound_id, compound_rows in filtered.groupby("compound_id", sort=False):
        with st.expander(
            f"{compound_id} · {compound_rows['target_family_id'].nunique()} target(s) · "
            f"{len(compound_rows)} result(s)",
            expanded=len(selected) == 1,
        ):
            for target, target_rows in compound_rows.groupby("target_family", sort=False):
                st.markdown(f"##### {target}")
                for variant, variant_rows in target_rows.groupby(
                    "target_variant", sort=False
                ):
                    with st.expander(
                        f"{variant} · {variant_rows['campaign_id'].nunique()} campaign(s)",
                        expanded=target_rows["target_variant"].nunique() == 1,
                    ):
                        _target_provenance(variant_rows)
                        for campaign, campaign_rows in variant_rows.groupby(
                            "campaign", sort=False
                        ):
                            with st.expander(
                                f"{campaign} · {len(campaign_rows)} prediction job(s)",
                                expanded=variant_rows["campaign"].nunique() == 1,
                            ):
                                _result_table(campaign_rows)


def render() -> None:
    st.title("Results Explorer")
    st.caption(
        "Navigate to the physical result or saved Analysis Set you want to "
        "inspect. Scientific comparison, ranking, pose validity, interactions, "
        "and 3D analysis live on the campaign result page."
    )
    root = runs_root()
    predictions, compounds = _load_index(
        str(root), _revision(), RESULTS_INDEX_SCHEMA_VERSION
    )
    if predictions.empty:
        if not _analysis_sets(root).empty:
            _analysis_set_view(root)
        else:
            st.info("No docking, cofolding, or rescoring results are available yet.")
        return
    show_incomplete = st.checkbox(
        "Show failed and incomplete jobs",
        value=False,
        help=(
            "Disabled by default because failed, queued, and running jobs do not "
            "represent usable scientific results. They remain available here for "
            "troubleshooting and in Jobs."
        ),
    )
    if not show_incomplete:
        predictions = predictions.loc[predictions["status"].eq("completed")].copy()
        compounds = compounds.loc[compounds["status"].eq("completed")].copy()
    browse_mode = st.segmented_control(
        "Browse by",
        ("Overview", "Datasets", "Targets", "Compounds", "Analysis Sets"),
        default="Overview",
        key="results_explorer_browse_mode",
    ) or "Overview"
    if browse_mode == "Overview":
        _overview(predictions, compounds)
    elif browse_mode == "Datasets":
        _dataset_view(predictions, compounds)
    elif browse_mode == "Targets":
        _target_view(predictions, compounds)
    elif browse_mode == "Compounds":
        _compound_view(predictions, compounds)
    else:
        _analysis_set_view(root)


render()
