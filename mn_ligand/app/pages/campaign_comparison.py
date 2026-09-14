from __future__ import annotations

import json
import hashlib
import math
import os
import re
from collections import Counter
from html import escape
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from zipfile import ZIP_DEFLATED, ZipFile

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from mn_ligand.app.pages.discover_inputs import target_inventory
from mn_ligand.app.viewers import (
    aligned_structure_data,
    closest_residue_ligand_atom_pair,
    is_mmcif_text,
    pdb_interaction_atom_coordinates,
    pdb_ligand_atom_aliases,
    render_persistent_3dmol,
)
from mn_ligand.app.analysis_sets import (
    ANALYSIS_SET_SCHEMA_VERSION,
    compound_smiles_signature,
    list_analysis_sets,
    save_analysis_set,
)
from mn_ligand.core.jobs import (
    JobRecord,
    display_job_code,
    iter_job_records,
)
from mn_ligand.core.residue_mapping import load_residue_mapping
from mn_ligand.core.campaign_extensions import (
    completed_repetitions,
    configured_repetitions,
)
from mn_ligand.core.provenance import (
    TARGET_PREPARATION_STEP_CODES,
    binding_campaign_purpose,
    compact_target_identifier,
)
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.interaction_analysis import (
    INTERACTION_RESIDUE_NUMBERING_POLICY_VERSION,
)
from mn_ligand.workflows.complex_datasets import (
    STEREOCHEMISTRY_POLICY_VERSION,
    create_complex_dataset,
    validate_complex_dataset_rows,
)
from mn_ligand.workflows.md_candidate_selection import (
    apply_reference_bend_penalty,
    infer_md_hypothesis,
    infer_static_hypothesis,
    interaction_residue,
    ligand_bend_index,
    normalize_interaction_type,
    score_candidate_poses,
    select_best_candidates,
)
from mn_ligand.workflows.pose_validation import (
    POSE_VALIDATION_INVENTORY_SCHEMA_VERSION,
    POSE_VALIDATION_SELECTION_POLICY,
)
from mn_ligand.workflows.pose_similarity import (
    POSE_SIMILARITY_TASK_GROUP,
    cancel_queued_pose_similarity_job,
    find_pose_similarity_job,
    load_pose_similarity_results,
    queue_pose_similarity_job,
)


ENGINE_METRICS: dict[str, tuple[tuple[str, str, bool], ...]] = {
    "AutoDock Vina": (
        ("best_score_kcal_mol", "Docking score (kcal/mol)", False),
    ),
    "GNINA": (
        (
            "cnn_ranked_cnn_affinity",
            "CNN-ranked pose · CNN affinity estimate",
            True,
        ),
        (
            "cnn_ranked_cnn_score",
            "CNN-ranked pose · CNN pose score",
            True,
        ),
        (
            "cnn_ranked_empirical_score_kcal_mol",
            "CNN-ranked pose · minimized Vina score (kcal/mol)",
            False,
        ),
        (
            "empirical_ranked_score_kcal_mol",
            "Best Vina pose · minimized Vina score (kcal/mol)",
            False,
        ),
        (
            "empirical_ranked_cnn_affinity",
            "Best Vina pose · CNN affinity estimate",
            True,
        ),
        (
            "empirical_ranked_cnn_score",
            "Best Vina pose · CNN pose score",
            True,
        ),
    ),
    "Uni-Dock Pro": (
        ("best_score_kcal_mol", "Docking score (kcal/mol)", False),
    ),
    "RosettaLigand": (
        ("estimated_dg_reu", "Estimated ΔG (REU)", False),
        ("total_score_reu", "Total score (REU)", False),
        ("ligand_score_reu", "Ligand score (REU)", False),
        (
            "ligand_rmsd_angstrom",
            "Reference-recovery ligand RMSD (Å)",
            False,
        ),
    ),
    "AlphaFold 3": (
        ("ranking_score", "Ranking score", True),
        ("iptm", "ipTM", True),
        ("ptm", "pTM", True),
        ("fraction_disordered", "Fraction disordered", False),
    ),
    "Boltz-2": (
        ("ic50_uM", "Predicted IC50 (µM)", False),
        ("affinity_probability_binary", "Binder probability", True),
        ("confidence_score", "Confidence score", True),
        ("iptm", "ipTM", True),
        ("ligand_iptm", "Ligand ipTM", True),
        ("complex_plddt", "Complex pLDDT", True),
        (
            "affinity_pred_value",
            "Native log10(IC50 / µM)",
            False,
        ),
    ),
    "Nesso-1": (
        ("ic50_uM", "Predicted IC50 (µM)", False),
        ("pIC50", "Predicted pIC50", True),
        ("binder_probability", "Binder probability", True),
        ("affinity_log10_ic50_uM", "log10(IC50 / µM)", False),
        (
            "ensemble_spread_log10_ic50_uM",
            "Affinity-head disagreement",
            False,
        ),
    ),
    "GNINA rescoring": (
        ("gnina_cnn_affinity", "CNN affinity estimate", True),
        (
            "gnina_empirical_score_kcal_mol",
            "Empirical score (kcal/mol)",
            False,
        ),
        ("gnina_cnn_score", "CNN pose score", True),
    ),
    "Boltzina rescoring": (
        ("boltzina_ic50_uM", "Predicted IC50 (µM)", False),
        (
            "boltzina_binder_probability",
            "Binder probability",
            True,
        ),
        (
            "boltzina_affinity_log10_ic50_uM",
            "Affinity log10(IC50 / µM)",
            False,
        ),
    ),
}
ENGINE_METRICS["GNINA · CNN-ranked"] = ENGINE_METRICS["GNINA"]
ENGINE_METRICS["GNINA · Vina-ranked"] = ENGINE_METRICS["GNINA"]

PRIMARY_METRIC = {
    "AutoDock Vina": "best_score_kcal_mol",
    "GNINA": "cnn_ranked_cnn_affinity",
    "Uni-Dock Pro": "best_score_kcal_mol",
    "RosettaLigand": "estimated_dg_reu",
    "AlphaFold 3": "iptm",
    "Boltz-2": "ic50_uM",
    "Nesso-1": "ic50_uM",
    "GNINA rescoring": "gnina_cnn_affinity",
    "Boltzina rescoring": "boltzina_ic50_uM",
    "GNINA · CNN-ranked": "cnn_ranked_cnn_affinity",
    "GNINA · Vina-ranked": "empirical_ranked_score_kcal_mol",
}

STRUCTURE_ENGINE_ORDER = (
    "AlphaFold 3",
    "Boltz-2",
    "GNINA",
    "Uni-Dock Pro",
    "AutoDock Vina",
    "RosettaLigand",
)

# Streamlit executes every tab body eagerly. Large screening campaigns can
# otherwise build all plots, pose-validation summaries and interaction tables
# before showing even the overview. Keep small comparisons unchanged while
# opening larger datasets in a lightweight, explicitly expandable overview.
LARGE_CAMPAIGN_EAGER_RENDER_THRESHOLD = 100


METRIC_REFERENCE_REGISTRY: dict[str, dict[str, str]] = {
    "ligand_rmsd_angstrom": {
        "family": "reference_recovery",
        "reference": "Input protein-bound ligand pose",
        "meaning": (
            "Recovery of the experimentally supplied ligand pose after "
            "redocking or refolding. Lower is better."
        ),
    },
    "fixed_frame_inter_engine_rmsd_angstrom": {
        "family": "inter_engine_agreement",
        "reference": "Another predicted pose in the shared receptor frame",
        "meaning": (
            "Agreement between predictions from engines or repetitions. It "
            "does not measure recovery of an experimental pose."
        ),
    },
}


def classify_campaign_shape(
    selected_jobs: pd.DataFrame,
    selected_metrics: pd.DataFrame,
) -> dict[str, object]:
    """Classify the selected physical campaigns by their entity dimensions."""
    target_count = int(
        selected_jobs.get("target_run_id", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .loc[lambda values: values.str.strip().ne("")]
        .nunique()
    )
    compound_count = int(
        selected_metrics.get("candidate_id", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .loc[lambda values: values.str.strip().ne("")]
        .nunique()
    )
    purpose_values = (
        selected_jobs.get("campaign_purpose", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
    )
    target_ligand = purpose_values.eq(
        "target_ligand_redocking_refolding"
    ).any()
    if target_count > 1 and compound_count > 1:
        shape = "multi_target_multi_compound"
        label = "Multi-target × multi-compound"
    elif target_count > 1:
        shape = "multi_target"
        label = "Multi-target"
    elif compound_count > 1:
        shape = "multi_compound"
        label = "Multi-compound"
    else:
        shape = "single_pair"
        label = "Single target–compound"
    return {
        "shape": shape,
        "label": label,
        "target_count": target_count,
        "compound_count": compound_count,
        "target_ligand_comparison": bool(target_ligand),
    }


def campaign_perspective_labels(shape: dict[str, object]) -> list[str]:
    entity_label = (
        "Target comparison"
        if shape.get("target_ligand_comparison")
        else "Target × compound explorer"
        if shape.get("shape") == "multi_target_multi_compound"
        else "Compound explorer"
        if int(shape.get("compound_count") or 0) > 1
        else "Result explorer"
    )
    return [
        "Overview",
        "Scores & ranking",
        "Structural evidence",
        entity_label,
        "Data & Analysis Sets",
    ]


_CAMPAIGN_EXPORT_IDENTITIES = (
    ("dataset", "Dataset"),
    ("dataset_run_id", "Dataset ID"),
    ("launch_campaign", "Launch campaign"),
    ("launch_campaign_id", "Launch campaign ID"),
    ("campaign", "Campaign"),
    ("campaign_id", "Campaign ID"),
    ("selected_target_key", "Selected target key"),
    ("selected_target_run_id", "Selected target ID"),
    ("coordinate_target_key", "Exact coordinate target key"),
    ("target_run_id", "Exact coordinate target ID"),
    ("selected_target", "Selected target provenance"),
    ("target", "Exact coordinate target provenance"),
    ("target_origin", "Target origin"),
    ("candidate_id", "Compound ID"),
    ("compound_name", "Compound name"),
    ("compound_name_source_column", "Compound name source column"),
    ("engine", "Engine"),
    ("engine_run", "Engine run"),
    ("job_code", "Job code"),
    ("campaign_purpose", "Campaign purpose"),
    ("status", "Status"),
    ("created_at", "Created at"),
    ("_ligand_smiles", "Ligand SMILES"),
    ("_structure_path", "Structure path"),
    ("_structure_kind", "Structure kind"),
)

COMPOUND_NAME_COLUMN_CANDIDATES = (
    "representative_product_name",
    "product_name",
    "compound_name",
    "preferred_name",
    "common_name",
    "name",
)

_CAMPAIGN_EXPORT_ATTEMPT_IDENTITIES = (
    ("replicate", "Recorded replicate"),
    ("repeat_index", "Recorded repeat index"),
    ("seed", "Seed"),
    ("model_seed", "Model seed"),
    ("sample", "Sample"),
    ("model_id", "Model ID"),
    ("prediction_id", "Prediction ID"),
    ("pose_index", "Pose index"),
    ("cnn_ranked_pose_index", "CNN-ranked pose index"),
    ("empirical_ranked_pose_index", "Vina-ranked pose index"),
    ("job_code", "Job code"),
    ("result", "Result link"),
)

_CAMPAIGN_NON_METRIC_COLUMNS = {
    "replicate",
    "repeat_index",
    "seed",
    "model_seed",
    "sample",
    "model_id",
    "prediction_id",
    "pose_index",
    "cnn_ranked_pose_index",
    "empirical_ranked_pose_index",
    "compound_count",
    "configured_repeats",
    "completed_repeats",
}


def _campaign_numeric_metric_definitions(
    engine_rows: pd.DataFrame,
    engine: str,
) -> list[tuple[str, str, bool | None]]:
    """Describe every numeric scientific output, including unregistered ones."""
    registered = {
        metric: (label, direction)
        for metric, label, direction in ENGINE_METRICS.get(engine, ())
    }
    definitions: list[tuple[str, str, bool | None]] = []
    for metric, (label, direction) in registered.items():
        if metric in engine_rows and pd.to_numeric(
            engine_rows[metric], errors="coerce"
        ).notna().any():
            definitions.append((metric, label, direction))
    identity_columns = {source for source, _ in _CAMPAIGN_EXPORT_IDENTITIES}
    for column in engine_rows:
        name = str(column)
        lowered = name.lower()
        if name in registered or name in identity_columns:
            continue
        if name in _CAMPAIGN_NON_METRIC_COLUMNS or name.startswith("_"):
            continue
        if lowered.endswith(("_id", "_index", "_file", "_path")):
            continue
        if lowered in {
            "status",
            "created_at",
            "result",
            "job_code",
            "campaign_purpose",
            "source_campaign_id",
            "selection_run_id",
            "target_family_run_id",
        }:
            continue
        if not pd.to_numeric(engine_rows[name], errors="coerce").notna().any():
            continue
        definitions.append(
            (
                name,
                name.replace("_", " ").strip().capitalize(),
                None,
            )
        )
    return definitions


def _select_boltz2_report_rows(engine_rows: pd.DataFrame) -> pd.DataFrame:
    """Select Boltz-2's native representative structure for every repeat."""
    required = {"candidate_id", "replicate", "model_id"}
    if not required.issubset(engine_rows.columns):
        return engine_rows
    grouping = [
        column
        for column in (
            "dataset_run_id",
            "campaign_id",
            "target_run_id",
            "candidate_id",
            "replicate",
        )
        if column in engine_rows
    ]
    if not grouping:
        return engine_rows
    prepared = engine_rows.copy()
    prepared["_boltz_model_zero"] = (
        prepared["model_id"].fillna("").astype(str).str.endswith("_model_0")
    )
    prepared["_boltz_confidence"] = pd.to_numeric(
        prepared.get(
            "confidence_score",
            pd.Series(float("nan"), index=prepared.index),
        ),
        errors="coerce",
    ).fillna(float("-inf"))
    prepared["_boltz_source_order"] = np.arange(len(prepared))
    prepared = prepared.sort_values(
        [*grouping, "_boltz_model_zero", "_boltz_confidence", "_boltz_source_order"],
        ascending=[*[True] * len(grouping), False, False, True],
        kind="stable",
    )
    return prepared.drop_duplicates(grouping, keep="first").drop(
        columns=[
            "_boltz_model_zero",
            "_boltz_confidence",
            "_boltz_source_order",
        ]
    )


def _select_alphafold3_report_rows(engine_rows: pd.DataFrame) -> pd.DataFrame:
    """Select the highest-ranked AF3 sample within every model-seed repeat."""
    required = {"candidate_id", "model_seed", "ranking_score"}
    if not required.issubset(engine_rows.columns):
        return engine_rows
    grouping = [
        column
        for column in (
            "dataset_run_id",
            "campaign_id",
            "target_run_id",
            "candidate_id",
            "model_seed",
        )
        if column in engine_rows
    ]
    repeat_grouping = [
        column for column in grouping if column != "model_seed"
    ]
    if not grouping or not repeat_grouping:
        return engine_rows
    prepared = engine_rows.copy()
    prepared["_af3_ranking_score"] = pd.to_numeric(
        prepared["ranking_score"], errors="coerce"
    ).fillna(float("-inf"))
    prepared["_af3_sample"] = pd.to_numeric(
        prepared.get(
            "sample",
            pd.Series(float("nan"), index=prepared.index),
        ),
        errors="coerce",
    ).fillna(float("inf"))
    prepared["_af3_source_order"] = np.arange(len(prepared))
    prepared = prepared.sort_values(
        [*grouping, "_af3_ranking_score", "_af3_sample", "_af3_source_order"],
        ascending=[*[True] * len(grouping), False, True, True],
        kind="stable",
    ).drop_duplicates(grouping, keep="first")
    prepared["replicate"] = (
        prepared.groupby(repeat_grouping, dropna=False, sort=False)[
            "model_seed"
        ]
        .rank(method="dense")
        .astype(int)
    )
    return prepared.drop(
        columns=[
            "_af3_ranking_score",
            "_af3_sample",
            "_af3_source_order",
        ]
    )


def _campaign_metric_export_tables(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return normalized and replicate-wide tables for campaign metrics."""
    identity_sources = [
        source for source, _ in _CAMPAIGN_EXPORT_IDENTITIES if source in metrics
    ]
    identity_labels = dict(_CAMPAIGN_EXPORT_IDENTITIES)
    attempt_identity_sources = [
        source
        for source, _ in _CAMPAIGN_EXPORT_ATTEMPT_IDENTITIES
        if source in metrics and source not in identity_sources
    ]
    attempt_identity_labels = dict(_CAMPAIGN_EXPORT_ATTEMPT_IDENTITIES)
    long_parts: list[pd.DataFrame] = []
    engines = (
        metrics.get("engine", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    for engine in engines:
        engine_rows = metrics.loc[
            metrics.get("engine", pd.Series("", index=metrics.index))
            .astype(str)
            .eq(engine)
        ].copy()
        if engine_rows.empty:
            continue
        if engine == "Boltz-2":
            engine_rows = _select_boltz2_report_rows(engine_rows)
        elif engine == "AlphaFold 3":
            engine_rows = _select_alphafold3_report_rows(engine_rows)
        definitions = _campaign_numeric_metric_definitions(engine_rows, engine)
        attempt_columns = [
            column
            for column, _ in _CAMPAIGN_EXPORT_ATTEMPT_IDENTITIES
            if column in engine_rows and column != "result"
        ]
        attempt = pd.Series("", index=engine_rows.index, dtype=str)
        for column in attempt_columns:
            values = engine_rows[column].fillna("").astype(str)
            usable = ~values.str.strip().isin({"", "nan", "None", "<NA>"})
            component = column + ":" + values
            has_prior_component = attempt.ne("") & usable
            attempt.loc[has_prior_component] = (
                attempt.loc[has_prior_component]
                + "::"
                + component.loc[has_prior_component]
            )
            first_component = attempt.eq("") & usable
            attempt.loc[first_component] = component.loc[first_component]
        if attempt.eq("").any():
            fallback_group = [
                column
                for column in identity_sources
                if column
                in {
                    "dataset_run_id",
                    "campaign_id",
                    "target_run_id",
                    "candidate_id",
                    "engine",
                }
            ]
            fallback = (
                engine_rows.groupby(fallback_group, dropna=False).cumcount() + 1
                if fallback_group
                else pd.Series(range(1, len(engine_rows) + 1), index=engine_rows.index)
            )
            attempt.loc[attempt.eq("")] = "row:" + fallback.astype(str)
        attempt = attempt + "::source_row:" + pd.Series(
            engine_rows.index + 1,
            index=engine_rows.index,
        ).astype(str)
        campaign_tokens = engine_rows.get(
            "campaign_id", pd.Series("", index=engine_rows.index)
        ).fillna("").astype(str)
        attempt = campaign_tokens + "::" + attempt
        engine_rows["_attempt_token"] = attempt

        for metric, label, higher_is_better in definitions:
            if metric not in engine_rows:
                continue
            values = pd.to_numeric(engine_rows[metric], errors="coerce")
            available = engine_rows.loc[values.notna()].copy()
            if available.empty:
                continue
            available["Value"] = values.loc[available.index].astype(float)
            grouping = [
                column
                for column in identity_sources
                if column
                in {
                    "dataset_run_id",
                    "campaign_id",
                    "target_run_id",
                    "candidate_id",
                    "engine",
                }
            ]
            recorded_repeat = pd.Series(
                float("nan"), index=available.index, dtype=float
            )
            for repeat_column in ("replicate", "repeat_index"):
                if repeat_column not in available:
                    continue
                candidate_repeat = pd.to_numeric(
                    available[repeat_column], errors="coerce"
                )
                if candidate_repeat.notna().any():
                    recorded_repeat = candidate_repeat
                    break
            if grouping:
                inferred_repeat = available.groupby(
                    grouping, dropna=False, sort=False
                )["_attempt_token"].transform(
                    lambda series: pd.factorize(series, sort=False)[0] + 1
                )
            else:
                inferred_repeat = pd.Series(
                    pd.factorize(available["_attempt_token"], sort=False)[0]
                    + 1,
                    index=available.index,
                )
            # A Boltz repeat contains several model predictions. Preserve each
            # model as an attempt, but never relabel the models as additional
            # independent repeats. Only infer repeat numbers for legacy rows
            # that did not record a repeat/repeat_index value.
            available["Replicate"] = recorded_repeat.fillna(inferred_repeat)
            part = available[
                identity_sources
                + attempt_identity_sources
                + ["_attempt_token", "Replicate", "Value"]
            ].copy()
            part = part.rename(
                columns={**identity_labels, **attempt_identity_labels}
            )
            part = part.rename(columns={"_attempt_token": "Attempt ID"})
            part["Parameter"] = metric
            part["Parameter label"] = label
            part["Higher is better"] = (
                "" if higher_is_better is None else bool(higher_is_better)
            )
            long_parts.append(part)

    if not long_parts:
        return pd.DataFrame(), pd.DataFrame()
    long = pd.concat(long_parts, ignore_index=True, sort=False)
    labelled_identities = [identity_labels[column] for column in identity_sources]
    for column in labelled_identities:
        long[column] = long[column].fillna("").astype(str)
    ordered_long = labelled_identities + [
        "Attempt ID",
        *[
            attempt_identity_labels[column]
            for column in attempt_identity_sources
        ],
        "Parameter",
        "Parameter label",
        "Higher is better",
        "Replicate",
        "Value",
    ]
    long = long[ordered_long].sort_values(
        labelled_identities + ["Parameter", "Replicate"],
        kind="stable",
    )
    wide_identity_labels = [
        identity_labels[column]
        for column in (
            "dataset",
            "dataset_run_id",
            "selected_target_key",
            "selected_target",
            "selected_target_run_id",
            "coordinate_target_key",
            "target",
            "target_run_id",
            "candidate_id",
            "engine",
        )
        if column in identity_sources
    ]
    wide_index = wide_identity_labels + [
        "Parameter",
        "Parameter label",
        "Higher is better",
    ]
    wide = long.pivot_table(
        index=wide_index,
        columns="Replicate",
        values="Value",
        aggfunc="mean",
        dropna=True,
    ).reset_index()
    replicate_columns = [
        column for column in wide.columns if isinstance(column, (int, float))
    ]
    wide = wide.rename(
        columns={column: f"Replicate {int(column)}" for column in replicate_columns}
    )
    value_columns = [f"Replicate {int(column)}" for column in replicate_columns]
    wide["Mean"] = wide[value_columns].mean(axis=1)
    wide["Sample SD"] = wide[value_columns].std(axis=1, ddof=1)
    wide["Replicate count"] = wide[value_columns].count(axis=1)
    return long.reset_index(drop=True), wide.reset_index(drop=True)


CAMPAIGN_DATABASE_EXPORT_SCHEMA_VERSION = 1
CAMPAIGN_DATABASE_METRIC_COLUMNS = (
    "schema_version",
    "observation_id",
    "dataset_id",
    "dataset_name",
    "launch_campaign_id",
    "launch_campaign_name",
    "campaign_id",
    "campaign_name",
    "campaign_purpose",
    "engine",
    "engine_run",
    "job_code",
    "selected_target_id",
    "selected_target_key",
    "selected_target_provenance",
    "coordinate_target_id",
    "coordinate_target_key",
    "coordinate_target_provenance",
    "target_origin",
    "compound_id",
    "compound_name",
    "compound_name_source_column",
    "ligand_smiles",
    "attempt_id",
    "replicate_number",
    "recorded_replicate",
    "recorded_repeat_index",
    "seed",
    "model_seed",
    "sample",
    "model_id",
    "prediction_id",
    "pose_index",
    "cnn_ranked_pose_index",
    "vina_ranked_pose_index",
    "selection_method",
    "score_type",
    "metric_name",
    "metric_label",
    "metric_unit",
    "higher_is_better",
    "value",
    "structure_path",
    "structure_kind",
    "status",
    "created_at",
    "result_link",
)


def _campaign_metric_unit(metric_name: object, metric_label: object) -> str:
    """Return a machine-friendly unit without inventing unknown semantics."""
    name = str(metric_name or "").lower()
    label = str(metric_label or "").lower()
    if "kcal_mol" in name or "kcal/mol" in label:
        return "kcal/mol"
    if name.endswith("_angstrom") or "(å)" in label:
        return "angstrom"
    if name.endswith("_um") or "(µm)" in label or "(um)" in label:
        return "micromolar"
    if name.endswith("_reu") or "(reu)" in label:
        return "REU"
    return ""


def _campaign_metric_selection(metric_name: object) -> tuple[str, str]:
    """Expose GNINA pose-selection semantics as explicit export dimensions."""
    name = str(metric_name or "")
    if name.startswith("cnn_ranked_"):
        return "cnn_ranked", name.removeprefix("cnn_ranked_")
    if name.startswith("empirical_ranked_"):
        score_type = name.removeprefix("empirical_ranked_")
        if score_type == "score_kcal_mol":
            score_type = "empirical_score_kcal_mol"
        return "vina_ranked", score_type
    return "", ""


def _campaign_database_metric_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Normalize heterogeneous engine metrics into one database-ready table."""
    long, _ = _campaign_metric_export_tables(metrics)
    if long.empty:
        return pd.DataFrame(columns=CAMPAIGN_DATABASE_METRIC_COLUMNS)
    # The general comparison workspace also discovers unregistered numeric
    # diagnostics such as pose centroids and box distances. They are useful
    # for debugging but make the database score export noisy and misleading.
    # Keep only the explicitly registered scientific metrics for each engine.
    registered_metrics = {
        engine: {metric for metric, _, _ in definitions}
        for engine, definitions in ENGINE_METRICS.items()
    }
    long = long.loc[
        [
            str(metric) in registered_metrics.get(str(engine), set())
            for engine, metric in zip(long["Engine"], long["Parameter"])
        ]
    ].copy()
    if long.empty:
        return pd.DataFrame(columns=CAMPAIGN_DATABASE_METRIC_COLUMNS)
    source_columns = {
        "Dataset ID": "dataset_id",
        "Dataset": "dataset_name",
        "Launch campaign ID": "launch_campaign_id",
        "Launch campaign": "launch_campaign_name",
        "Campaign ID": "campaign_id",
        "Campaign": "campaign_name",
        "Campaign purpose": "campaign_purpose",
        "Engine": "engine",
        "Engine run": "engine_run",
        "Job code": "job_code",
        "Selected target ID": "selected_target_id",
        "Selected target key": "selected_target_key",
        "Selected target provenance": "selected_target_provenance",
        "Exact coordinate target ID": "coordinate_target_id",
        "Exact coordinate target key": "coordinate_target_key",
        "Exact coordinate target provenance": "coordinate_target_provenance",
        "Target origin": "target_origin",
        "Compound ID": "compound_id",
        "Compound name": "compound_name",
        "Compound name source column": "compound_name_source_column",
        "Ligand SMILES": "ligand_smiles",
        "Attempt ID": "attempt_id",
        "Replicate": "replicate_number",
        "Recorded replicate": "recorded_replicate",
        "Recorded repeat index": "recorded_repeat_index",
        "Seed": "seed",
        "Model seed": "model_seed",
        "Sample": "sample",
        "Model ID": "model_id",
        "Prediction ID": "prediction_id",
        "Pose index": "pose_index",
        "CNN-ranked pose index": "cnn_ranked_pose_index",
        "Vina-ranked pose index": "vina_ranked_pose_index",
        "Parameter": "metric_name",
        "Parameter label": "metric_label",
        "Higher is better": "higher_is_better",
        "Value": "value",
        "Structure path": "structure_path",
        "Structure kind": "structure_kind",
        "Status": "status",
        "Created at": "created_at",
        "Result link": "result_link",
    }
    normalized = long.rename(columns=source_columns)
    for column in CAMPAIGN_DATABASE_METRIC_COLUMNS:
        if column not in normalized:
            normalized[column] = ""
    normalized["schema_version"] = CAMPAIGN_DATABASE_EXPORT_SCHEMA_VERSION
    normalized["metric_unit"] = [
        _campaign_metric_unit(metric, label)
        for metric, label in zip(
            normalized["metric_name"], normalized["metric_label"]
        )
    ]
    selections = [
        _campaign_metric_selection(metric)
        for metric in normalized["metric_name"]
    ]
    normalized["selection_method"] = [value[0] for value in selections]
    normalized["score_type"] = [value[1] for value in selections]
    normalized["observation_id"] = [
        "OBS-"
        + hashlib.sha256(
            f"{attempt_id}\0{metric_name}".encode("utf-8")
        ).hexdigest()[:20]
        for attempt_id, metric_name in zip(
            normalized["attempt_id"], normalized["metric_name"]
        )
    ]
    for column in CAMPAIGN_DATABASE_METRIC_COLUMNS:
        if (
            column not in {"value", "replicate_number"}
            and normalized[column].isna().any()
        ):
            normalized[column] = normalized[column].astype(object).where(
                normalized[column].notna(), ""
            )
    return normalized[list(CAMPAIGN_DATABASE_METRIC_COLUMNS)].reset_index(
        drop=True
    )


CAMPAIGN_DATABASE_REPEAT_IDENTITY_COLUMNS = (
    "schema_version",
    "dataset_id",
    "campaign_id",
    "engine",
    "job_code",
    "selected_target_id",
    "coordinate_target_id",
    "compound_id",
    "compound_name",
    "selection_method",
    "score_type",
    "metric_name",
    "metric_label",
    "metric_unit",
    "higher_is_better",
)


def _campaign_database_repeat_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return one calculation-ready row with repeat_N value columns."""
    normalized = _campaign_database_metric_rows(metrics)
    if normalized.empty:
        return pd.DataFrame(
            columns=[
                *CAMPAIGN_DATABASE_REPEAT_IDENTITY_COLUMNS,
            ]
        )
    identity_columns = list(CAMPAIGN_DATABASE_REPEAT_IDENTITY_COLUMNS)
    matrix = (
        normalized.pivot_table(
            index=identity_columns,
            columns="replicate_number",
            values="value",
            aggfunc="first",
            observed=True,
            dropna=True,
        )
        .reset_index()
    )
    numeric_repeat_columns = sorted(
        [
            column
            for column in matrix.columns
            if isinstance(column, (int, float, np.integer, np.floating))
        ],
        key=float,
    )
    rename_repeats = {
        column: f"repeat_{int(column)}" for column in numeric_repeat_columns
    }
    matrix = matrix.rename(columns=rename_repeats)
    repeat_columns = [rename_repeats[column] for column in numeric_repeat_columns]
    return matrix[
        [
            *identity_columns,
            *repeat_columns,
        ]
    ].reset_index(drop=True)


def _database_column_name(value: object) -> str:
    compact = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return compact.strip("_") or "field"


def _database_ready_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Prepare an auxiliary relation for portable CSV database import."""
    if frame is None:
        return pd.DataFrame()
    prepared = frame[
        [column for column in frame if not str(column).startswith("_")]
    ].copy()
    prepared.columns = [_database_column_name(column) for column in prepared]
    for column in prepared:
        if prepared[column].dtype == object:
            prepared[column] = prepared[column].map(
                lambda value: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple, set))
                else value
            )
    return prepared


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _campaign_raw_attempt_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return every source data row with stable, explicit attempt identity."""
    public = metrics[
        [column for column in metrics if not str(column).startswith("_")]
    ].copy().reset_index(drop=True)
    tokens: list[str] = []
    identity_columns = (
        "campaign_id",
        "engine_run",
        "target_run_id",
        "candidate_id",
        "replicate",
        "repeat_index",
        "seed",
        "model_seed",
        "sample",
        "model_id",
        "prediction_id",
        "pose_index",
        "cnn_ranked_pose_index",
        "empirical_ranked_pose_index",
        "job_code",
    )
    for row_index, row in public.iterrows():
        components = []
        for column in identity_columns:
            if column not in public:
                continue
            value = row.get(column)
            if pd.isna(value) or str(value).strip() in {"", "nan", "None", "<NA>"}:
                continue
            components.append(f"{column}:{value}")
        components.append(f"source_row:{row_index + 1}")
        tokens.append("::".join(components))
    public.insert(0, "Attempt ID", tokens)
    public.insert(
        0,
        "Data row ID",
        [f"DATA-{index:07d}" for index in range(1, len(public) + 1)],
    )
    public = public.rename(
        columns={
            "selected_target_key": "Selected target key",
            "selected_target_run_id": "Selected target ID",
            "coordinate_target_key": "Exact coordinate target key",
            "target_run_id": "Exact coordinate target ID",
            "selected_target": "Selected target provenance",
            "target": "Exact coordinate target provenance",
            "target_family_run_id": "Target family run ID",
            "candidate_id": "Compound ID",
            "compound_name": "Compound name",
            "compound_name_source_column": "Compound name source column",
        }
    )
    return public


def _excel_ready_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared = prepared[
        [column for column in prepared if not str(column).startswith("_")]
    ]
    for column in prepared:
        if isinstance(prepared[column].dtype, pd.DatetimeTZDtype):
            prepared[column] = prepared[column].dt.tz_localize(None)
        elif prepared[column].dtype == object:
            prepared[column] = prepared[column].map(
                lambda value: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple, set))
                else value
            )
    return prepared


def _campaign_results_workbook(
    selected_jobs: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    pose_rows: pd.DataFrame | None = None,
    pose_provenance: pd.DataFrame | None = None,
    interaction_summaries: pd.DataFrame | None = None,
    interactions: pd.DataFrame | None = None,
    interaction_provenance: pd.DataFrame | None = None,
) -> bytes:
    """Build one offline workbook from the currently selected campaign data."""
    long, wide = _campaign_metric_export_tables(metrics)
    raw = _campaign_raw_attempt_rows(metrics)
    sheets = {
        "All raw data": raw,
        "Metric values": long,
        "Metrics by repeat": wide,
        "Selected campaigns": selected_jobs,
        "Pose validity": pose_rows,
        "Pose provenance": pose_provenance,
        "Interaction summary": interaction_summaries,
        "Interactions": interactions,
        "Interaction provenance": interaction_provenance,
    }
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            if frame is None or frame.empty:
                continue
            _excel_ready_frame(frame).to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                freeze_panes=(1, 0),
            )
            worksheet = writer.sheets[sheet_name]
            worksheet.auto_filter.ref = worksheet.dimensions
            for cells in worksheet.iter_cols(
                min_row=1,
                max_row=min(200, worksheet.max_row),
            ):
                width = min(
                    48,
                    max(10, max(len(str(cell.value or "")) for cell in cells) + 2),
                )
                worksheet.column_dimensions[cells[0].column_letter].width = width
    return buffer.getvalue()


def _collect_campaign_results_workbook(
    run_root: Path,
    export_jobs: pd.DataFrame,
    data_metrics: pd.DataFrame,
) -> bytes:
    """Collect every selected score and its linked structural evidence."""
    structural_jobs = export_jobs.loc[
        export_jobs["engine"].isin(STRUCTURE_ENGINE_ORDER)
    ].copy()
    pose_rows, pose_provenance, _ = _pose_validation_rows(
        run_root,
        structural_jobs,
    )
    if not pose_rows.empty:
        pose_rows = _pose_validation_display_rows(pose_rows)
    (
        interaction_summaries,
        interactions,
        interaction_provenance,
    ) = _interaction_analysis_rows(run_root, structural_jobs)
    return _campaign_results_workbook(
        export_jobs,
        data_metrics,
        pose_rows=pose_rows,
        pose_provenance=pose_provenance,
        interaction_summaries=interaction_summaries,
        interactions=interactions,
        interaction_provenance=interaction_provenance,
    )


def _campaign_results_csv_bundle(
    selected_jobs: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    pose_rows: pd.DataFrame | None = None,
    pose_provenance: pd.DataFrame | None = None,
    interaction_summaries: pd.DataFrame | None = None,
    interactions: pd.DataFrame | None = None,
    interaction_provenance: pd.DataFrame | None = None,
) -> bytes:
    """Build a relational CSV bundle suitable for database ingestion."""
    tables = {
        "metric_observations.csv": _campaign_database_metric_rows(metrics),
        "metric_repeats_wide.csv": _campaign_database_repeat_matrix(metrics),
        "campaign_runs.csv": _database_ready_frame(selected_jobs),
        "pose_validation.csv": _database_ready_frame(pose_rows),
        "pose_validation_provenance.csv": _database_ready_frame(
            pose_provenance
        ),
        "interaction_summaries.csv": _database_ready_frame(
            interaction_summaries
        ),
        "interactions.csv": _database_ready_frame(interactions),
        "interaction_provenance.csv": _database_ready_frame(
            interaction_provenance
        ),
    }
    manifest = {
        "export": "campaign_database_csv_bundle",
        "schema_version": CAMPAIGN_DATABASE_EXPORT_SCHEMA_VERSION,
        "encoding": "UTF-8",
        "delimiter": ",",
        "primary_table": "metric_observations.csv",
        "tables": {
            name: {"rows": int(len(frame)), "columns": list(frame.columns)}
            for name, frame in tables.items()
            if not frame.empty
        },
    }
    readme = (
        "Campaign database export\n"
        "========================\n\n"
        "metric_observations.csv is the primary normalized table. Each row "
        "contains exactly one numeric metric observation from one exported "
        "repeat. For Boltz-2 and AlphaFold 3 this is the engine-ranked "
        "representative structure for that repeat. Engine-specific metrics are "
        "represented by metric_name, "
        "metric_label, metric_unit, and value rather than separate columns.\n\n"
        "Only registered scientific score, affinity, and confidence metrics "
        "are included. Geometry diagnostics such as pose centroids and box "
        "distances are intentionally omitted from this report. "
        "selection_method and score_type are populated only when an engine "
        "has distinct selection tracks, currently GNINA.\n\n"
        "metric_repeats_wide.csv is a calculation-ready companion view. It "
        "contains one row per campaign, compound, engine, and metric, with "
        "adjacent repeat_1, repeat_2, ... columns. For Boltz-2, the native "
        "representative structure model is selected once per independent "
        "repeat (model_0 when available, then highest structure confidence), "
        "and that model's binding and confidence values are reported. For "
        "AlphaFold 3, each model seed is one repeat and the sample with the "
        "highest ranking_score is reported.\n\n"
        "observation_id is the primary key for metric observations. "
        "campaign_id, dataset_id, compound_id, selected_target_id, "
        "coordinate_target_id, and attempt_id retain provenance and can be "
        "used as database join keys. Auxiliary CSV files represent separate "
        "relations and should not be concatenated with metric observations.\n\n"
        "Files use UTF-8, comma delimiters, one header row, and no index "
        "column. Empty strings represent unavailable optional values. See "
        "manifest.json for row counts and exact columns.\n"
    )
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, frame in tables.items():
            if not frame.empty:
                archive.writestr(name, _csv_bytes(frame))
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        archive.writestr("README.txt", readme)
    return output.getvalue()


def _collect_campaign_results_csv_bundle(
    run_root: Path,
    export_jobs: pd.DataFrame,
    data_metrics: pd.DataFrame,
) -> bytes:
    """Collect normalized scores and linked evidence as relational CSVs."""
    structural_jobs = export_jobs.loc[
        export_jobs["engine"].isin(STRUCTURE_ENGINE_ORDER)
    ].copy()
    pose_rows, pose_provenance, _ = _pose_validation_rows(
        run_root,
        structural_jobs,
    )
    if not pose_rows.empty:
        pose_rows = _pose_validation_display_rows(pose_rows)
    (
        interaction_summaries,
        interactions,
        interaction_provenance,
    ) = _interaction_analysis_rows(run_root, structural_jobs)
    return _campaign_results_csv_bundle(
        export_jobs,
        data_metrics,
        pose_rows=pose_rows,
        pose_provenance=pose_provenance,
        interaction_summaries=interaction_summaries,
        interactions=interactions,
        interaction_provenance=interaction_provenance,
    )


def _campaign_workbook_signature(
    export_jobs: pd.DataFrame,
    data_metrics: pd.DataFrame,
) -> str:
    """Fingerprint selected raw data so a stale workbook is never offered."""
    payload = {
        "export_schema_version": 2,
        "campaigns": sorted(
            export_jobs["campaign_id"].fillna("").astype(str).tolist()
        ),
        "metrics": data_metrics.to_json(
            orient="split",
            date_format="iso",
            default_handler=str,
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def _native_metric_plots_zip(
    plots: list[dict[str, object]],
    *,
    repetition_mode: str,
) -> bytes:
    """Render selected native-metric summaries as publication-ready PNGs."""
    archive_buffer = BytesIO()
    manifest: dict[str, object] = {
        "export": "native_engine_metric_plots",
        "repetition_mode": repetition_mode,
        "plots": [],
    }
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        for index, plot in enumerate(plots, start=1):
            engine = str(plot["engine"])
            metric = str(plot["metric"])
            metric_label = str(plot["metric_label"])
            higher_is_better = bool(plot["higher_is_better"])
            compare_targets = bool(plot.get("compare_targets"))
            compound_label_mode = str(
                plot.get("compound_label_mode") or "Compound IDs"
            )
            compound_label_layout = str(
                plot.get("compound_label_layout") or "Automatic"
            )
            summary = plot["summary"]
            if not isinstance(summary, pd.DataFrame) or summary.empty:
                continue
            prepared = _with_compound_plot_labels(
                summary,
                label_mode=compound_label_mode,
            )
            prepared["Mean"] = pd.to_numeric(
                prepared["Mean"], errors="coerce"
            )
            prepared["Sample SD"] = pd.to_numeric(
                prepared["Sample SD"], errors="coerce"
            ).fillna(0.0)
            prepared = prepared.dropna(subset=["Mean"])
            if prepared.empty:
                continue
            entity_column = (
                "campaign" if compare_targets else "_compound_plot_label"
            )
            labels = prepared[entity_column].fillna("Unknown").astype(str)
            if not compare_targets:
                if compound_label_mode == "Compound names":
                    labels = labels.str.split("|||", n=1).str[0]
                elif compound_label_mode == "Names + IDs":
                    labels = labels.str.replace("|||", "\n", regex=False)
            if not compare_targets and prepared[entity_column].duplicated(
                keep=False
            ).any():
                labels = (
                    labels
                    + " · "
                    + prepared["campaign"].fillna("Campaign").astype(str)
                )
            prepared = prepared.assign(_plot_label=labels).sort_values(
                "Mean",
                ascending=not higher_is_better,
                kind="stable",
            )
            height = max(4.2, 0.48 * len(prepared) + 1.9)
            figure, axis = plt.subplots(figsize=(12, height))
            positions = list(range(len(prepared)))
            axis.barh(
                positions,
                prepared["Mean"],
                xerr=prepared["Sample SD"],
                color="#2563eb",
                alpha=0.84,
                ecolor="#111827",
                capsize=4,
            )
            axis.set_yticks(positions, prepared["_plot_label"])
            axis.invert_yaxis()
            axis.set_xlabel(metric_label)
            axis.set_ylabel(
                "Prepared target" if compare_targets else "Compound / campaign"
            )
            axis.set_title(f"{engine} — {metric_label}", loc="left", weight="bold")
            axis.text(
                0,
                1.01,
                (
                    f"{repetition_mode}; mean ± sample SD across selected "
                    "independent attempts"
                ),
                transform=axis.transAxes,
                fontsize=9,
                color="#4b5563",
            )
            axis.grid(axis="x", alpha=0.22)
            axis.set_axisbelow(True)
            figure.tight_layout()
            slug = "-".join(
                "".join(
                    character.lower()
                    if character.isalnum()
                    else "-"
                    for character in value
                ).strip("-")
                for value in (engine, metric)
            )
            slug = f"{index:02d}-{slug}"
            png_buffer = BytesIO()
            figure.savefig(
                png_buffer,
                format="png",
                dpi=220,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(figure)
            png_path = f"plots/{slug}.png"
            archive.writestr(png_path, png_buffer.getvalue())
            manifest["plots"].append(
                {
                    "engine": engine,
                    "metric": metric,
                    "metric_label": metric_label,
                    "higher_is_better": higher_is_better,
                    "compound_label_mode": compound_label_mode,
                    "compound_label_layout": compound_label_layout,
                    "png": png_path,
                    "rows": len(prepared),
                }
            )
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True),
        )
        archive.writestr(
            "README.txt",
            "Each PNG reproduces one selected Native engine metric view.\n"
            "Use the separate normalized campaign CSV export for every raw "
            "repetition-level value.\n"
            "Error bars are sample standard deviations across the attempts "
            "selected by the recorded repetition mode.\n",
        )
    return archive_buffer.getvalue()


STRUCTURE_ENGINE_PALETTE = (
    ("cyanCarbon", "cyan"),
    ("magentaCarbon", "magenta"),
    ("orangeCarbon", "orange"),
    ("purpleCarbon", "purple"),
    ("yellowCarbon", "yellow"),
    ("blueCarbon", "blue"),
    ("whiteCarbon", "white"),
    ("greenCarbon", "green"),
)
STRUCTURE_ENGINE_STYLES = {
    engine: STRUCTURE_ENGINE_PALETTE[index]
    for index, engine in enumerate(STRUCTURE_ENGINE_ORDER)
}
STRUCTURE_ENGINE_STYLES.update(
    {
        "GNINA · CNN-ranked": ("orangeCarbon", "orange"),
        "GNINA · Vina-ranked": ("redCarbon", "red"),
    }
)

# Fixed across campaigns so color intensity has the same scientific meaning
# in every matrix. Values beyond the displayed maximum use the worst color
# rather than stretching the scale for that particular result set.
POSE_COMPARISON_SCALES = {
    "Atom-mapped RMSD (Å)": {
        "maximum": 4.0,
        "guidance": (
            "≤1 Å close agreement; 1–2 Å generally acceptable pose "
            "agreement; 2–3 Å different; ≥3 Å strongly different"
        ),
    },
    "Centroid displacement (Å)": {
        "maximum": 4.0,
        "guidance": (
            "≤0.5 Å nearly unchanged center; 0.5–1 Å close; 1–2 Å "
            "shifted; ≥2 Å substantially displaced"
        ),
    },
    "Shape distance": {
        "maximum": 1.0,
        "guidance": (
            "0 is identical volume; ≤0.1 very similar; 0.1–0.25 close; "
            "0.25–0.5 moderate; ≥0.5 weak overlap"
        ),
    },
}


def _pose_comparison_scale(metric_label: str) -> dict[str, object]:
    """Return the fixed cross-campaign display scale for a pose metric."""
    return POSE_COMPARISON_SCALES.get(
        metric_label,
        POSE_COMPARISON_SCALES["Atom-mapped RMSD (Å)"],
    )

def _reset_dataset_dependent_controls() -> None:
    for key in (
        "campaign_compare_target",
        "campaign_compare_launches",
        "campaign_compare_engines",
        "campaign_compare_campaigns",
        "campaign_compare_target_launch_ids",
        "campaign_compare_target_launch_table",
        "campaign_compare_scope_campaign",
        "campaign_compare_scope_campaigns",
        "_campaign_compare_scope_campaigns_context",
        "campaign_compare_target_launch_explicit_empty",
        "campaign_compare_target_launch_explicit_empty_v2",
        "campaign_compare_target_launch_table_revision",
        "campaign_compare_target_order_column",
        "campaign_compare_target_order_direction",
        "campaign_edit_analysis_set_scope",
        "campaign_compare_rescoring_engines",
        "campaign_compare_rescoring_runs",
        "campaign_native_engine",
        "campaign_native_metric",
        "campaign_native_repetition_mode",
        "campaign_native_best_repetitions",
        "campaign_native_compound_label_mode_v2",
        "campaign_native_compound_label_layout",
        "_campaign_targets_context",
        "_campaign_launches_context",
        "_campaign_engines_context",
        "_campaign_engine_runs_context",
        "campaign_correlation_target",
        "campaign_correlation_scope",
        "campaign_correlation_metrics",
        "campaign_correlation_method",
        "campaign_correlation_view",
        "campaign_correlation_pair",
        "campaign_correlation_matrix_metrics",
        "campaign_correlation_scatter_labels",
        "campaign_correlation_scatter_trend",
        "campaign_viewer_compound",
        "campaign_viewer_layout",
        "campaign_viewer_target",
        "campaign_viewer_campaigns",
        "campaign_viewer_mode",
        "campaign_viewer_predictions",
        "campaign_viewer_matrix_compounds",
        "campaign_viewer_matrix_target",
        "campaign_viewer_matrix_campaigns",
        "campaign_viewer_matrix_mode",
        "campaign_viewer_matrix_columns",
        "campaign_viewer_matrix_show_target",
        "campaign_viewer_matrix_show_reference_ligand",
        "campaign_viewer_matrix_show_predicted_proteins",
        "_campaign_viewer_matrix_target_context",
        "_campaign_viewer_matrix_campaign_context",
        "campaign_viewer_show_target",
        "campaign_viewer_show_reference_ligand",
        "campaign_viewer_show_predicted_proteins",
        "campaign_rmsd_result_view",
        "campaign_rmsd_mode",
        "campaign_rmsd_comparison_mode",
        "campaign_viewer_rmsd_matrix_mode",
        "campaign_viewer_rmsd_reference",
        "campaign_pose_validation_target",
    ):
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if str(key).startswith("campaign_native_metric_"):
            st.session_state.pop(key, None)


def _reset_prepared_target_scope() -> None:
    """Reset row selections when the physical launch campaign changes."""
    for key in (
        "campaign_compare_target_launch_ids",
        "campaign_compare_target_launch_explicit_empty_v2",
        "campaign_compare_target_launch_table_revision",
        "campaign_compare_scope_campaigns",
        "_campaign_compare_scope_campaigns_context",
    ):
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if str(key).startswith("campaign_compare_target_launch_table::"):
            st.session_state.pop(key, None)


def _compatible_target_ligand_launches(
    launch_rows: pd.DataFrame,
    anchor_launch_id: str,
) -> list[str]:
    """Limit combinations to one campaign family and ligand identity."""
    anchor = launch_rows.loc[
        launch_rows["_launch_campaign_id"].astype(str).eq(
            str(anchor_launch_id)
        )
    ]
    if anchor.empty:
        return []
    purpose = str(anchor.iloc[0].get("_campaign_purpose") or "")
    anchor_signatures = {
        member
        for value in anchor["_compound_signature"].tolist()
        for member in str(value).split("||")
        if member
    }
    if not anchor_signatures:
        # Missing chemical identity is not evidence of compatibility.
        return [str(anchor_launch_id)]
    compatible: list[str] = []
    same_family = launch_rows.loc[
        launch_rows["_campaign_purpose"].astype(str).eq(purpose)
    ]
    for launch_id, group in same_family.groupby(
        "_launch_campaign_id", sort=False
    ):
        candidate_signatures = {
            member
            for value in group["_compound_signature"].tolist()
            for member in str(value).split("||")
            if member
        }
        if not anchor_signatures.isdisjoint(candidate_signatures):
            compatible.append(str(launch_id))
    return compatible


def _reconcile_cascading_multiselect(
    *,
    key: str,
    context_key: str,
    context: tuple[object, ...],
    options: list[str],
    defaults: list[str],
) -> None:
    """Reset a downstream multiselect only when its upstream context changes."""
    normalized_context = tuple(str(value) for value in context) + tuple(
        str(value) for value in options
    )
    if st.session_state.get(context_key) != normalized_context:
        st.session_state[context_key] = normalized_context
        st.session_state[key] = [
            value for value in defaults if value in options
        ]
        return
    if key in st.session_state:
        current = list(st.session_state.get(key) or [])
        valid = [value for value in current if value in options]
        if valid != current:
            st.session_state[key] = valid


def _deep_link_campaign_scope(
    campaigns: pd.DataFrame,
    *,
    target_run_id: str,
    launch_campaign_id: str,
    campaign_purpose: str = "",
) -> pd.DataFrame:
    scoped = campaigns
    if target_run_id:
        scoped = scoped.loc[
            scoped["target_run_id"].eq(target_run_id)
        ]
    if launch_campaign_id:
        scoped = scoped.loc[
            scoped["launch_campaign_id"].eq(launch_campaign_id)
        ]
    if campaign_purpose:
        scoped = scoped.loc[
            scoped["campaign_purpose"].eq(campaign_purpose)
        ]
    return scoped.copy()


def _target_ligand_launch_rows(
    campaigns: pd.DataFrame,
) -> pd.DataFrame:
    inventory_rows = {
        entry.choice.job.run_id: entry.row
        for entry in target_inventory(include_transient_targets=True)
    }
    primary = campaigns.loc[
        ~campaigns["engine"].astype(str).str.endswith("rescoring")
    ].copy()
    rows: list[dict[str, object]] = []
    for (target_run_id, launch_campaign_id), group in primary.groupby(
        ["target_run_id", "launch_campaign_id"],
        sort=False,
        dropna=False,
    ):
        target_run_id = str(target_run_id)
        launch_campaign_id = str(launch_campaign_id)
        inventory = inventory_rows.get(target_run_id, {})
        created_values = pd.to_datetime(
            group["created_at"], errors="coerce", utc=True
        ).dropna()
        rows.append(
            {
                "Campaign": str(group.iloc[0]["launch_campaign"]),
                "Target job": str(inventory.get("Job") or ""),
                "Target": str(group.iloc[0]["target"]),
                "PDB / origin": str(inventory.get("Target") or "—"),
                "Tool": str(inventory.get("Tool") or "-"),
                "Origin": str(inventory.get("Origin") or "-"),
                "Last step": str(inventory.get("Last step") or "-"),
                "Residues": inventory.get("Residues"),
                "Engines": ", ".join(
                    sorted(group["engine"].astype(str).unique())
                ),
                "Created": (
                    created_values.min()
                    if not created_values.empty
                    else pd.NaT
                ),
                "_target_run_id": target_run_id,
                "_launch_campaign_id": launch_campaign_id,
                "_selection_id": (
                    f"{launch_campaign_id}::{target_run_id}"
                ),
                "_compound_signature": str(
                    group.iloc[0].get("compound_signature") or ""
                ),
                "_campaign_purpose": str(
                    group.iloc[0].get("campaign_purpose") or ""
                ),
            }
        )
    return pd.DataFrame(rows)


def target_engine_coverage(campaigns: pd.DataFrame) -> pd.DataFrame:
    """Summarize execution and repeat balance per prepared target/engine."""
    if campaigns.empty:
        return pd.DataFrame()
    prepared = campaigns.copy()
    configured = (
        prepared["configured_repeats"]
        if "configured_repeats" in prepared
        else pd.Series(1, index=prepared.index, dtype=int)
    )
    completed = (
        prepared["completed_repeats"]
        if "completed_repeats" in prepared
        else pd.Series(0, index=prepared.index, dtype=int)
    )
    prepared["configured_repeats"] = pd.to_numeric(
        configured, errors="coerce"
    ).fillna(1).clip(lower=1)
    prepared["completed_repeats"] = pd.to_numeric(
        completed, errors="coerce"
    ).fillna(0).clip(lower=0)
    prepared["_coverage_campaign"] = (
        prepared["launch_campaign_id"].fillna("").astype(str)
        if "launch_campaign_id" in prepared
        else "comparison scope"
    )
    prepared["campaign_target_repeats"] = prepared.groupby(
        "_coverage_campaign", dropna=False
    )["configured_repeats"].transform("max")
    if "target_origin" not in prepared:
        prepared["target_origin"] = ""
    if "target_artifact" not in prepared:
        prepared["target_artifact"] = ""
    prepared["target_preparation"] = prepared["target"].fillna(
        "Prepared target"
    ).astype(str)
    grouped = (
        prepared.groupby(
            ["target_run_id", "target_preparation", "engine"],
            as_index=False,
            dropna=False,
        )
        .agg(
            configured_repeats=("configured_repeats", "max"),
            completed_repeats=("completed_repeats", "max"),
            campaign_target_repeats=("campaign_target_repeats", "max"),
            engine_jobs=("campaign_id", "nunique"),
            target_origin=("target_origin", "first"),
            target_artifact=("target_artifact", "first"),
        )
    )
    grouped["completion_fraction"] = (
        grouped["completed_repeats"] / grouped["configured_repeats"]
    ).clip(lower=0, upper=1)
    grouped["repeat_coverage"] = (
        grouped["completed_repeats"].astype(int).astype(str)
        + "/"
        + grouped["configured_repeats"].astype(int).astype(str)
    )
    grouped["comparison_fraction"] = (
        grouped["completed_repeats"] / grouped["campaign_target_repeats"]
    ).clip(lower=0, upper=1)
    grouped["comparison_coverage"] = (
        grouped["completed_repeats"].astype(int).astype(str)
        + "/"
        + grouped["campaign_target_repeats"].astype(int).astype(str)
    )
    return grouped


def _render_target_engine_coverage(campaigns: pd.DataFrame) -> None:
    coverage = target_engine_coverage(campaigns)
    if coverage.empty:
        return
    engine_names = set(coverage["engine"].astype(str))
    engine_order = [
        engine for engine in STRUCTURE_ENGINE_ORDER if engine in engine_names
    ] + sorted(engine_names.difference(STRUCTURE_ENGINE_ORDER))
    target_order = list(
        dict.fromkeys(coverage["target_preparation"].astype(str).tolist())
    )
    st.markdown("#### Target × engine repeat coverage")
    st.caption(
        "Each cell reports completed repetitions against the highest configured "
        "repeat count in its physical campaign. The tooltip separately reports "
        "completed/configured execution. A successful 1/1 job therefore appears "
        "as 1/3 when the campaign comparison target is three."
    )
    rectangles = (
        alt.Chart(coverage)
        .mark_rect(stroke="white")
        .encode(
            x=alt.X(
                "engine:N",
                sort=engine_order,
                title="Engine",
                axis=alt.Axis(labelAngle=-25),
            ),
            y=alt.Y(
                "target_preparation:N",
                sort=target_order,
                title="Prepared target",
                axis=alt.Axis(labelLimit=420),
            ),
            color=alt.Color(
                "comparison_fraction:Q",
                scale=alt.Scale(
                    domain=[0, 1],
                    range=["#fee2e2", "#16a34a"],
                ),
                title="Comparison coverage",
            ),
            tooltip=[
                alt.Tooltip("target_preparation:N", title="Prepared target"),
                alt.Tooltip("target_origin:N", title="Full provenance"),
                alt.Tooltip("target_artifact:N", title="Artifact file"),
                alt.Tooltip("engine:N", title="Engine"),
                alt.Tooltip(
                    "comparison_coverage:N", title="Comparison coverage"
                ),
                alt.Tooltip(
                    "repeat_coverage:N", title="Completed/configured"
                ),
                alt.Tooltip("engine_jobs:Q", title="Engine jobs"),
            ],
        )
        .properties(height=max(320, 34 * len(target_order)))
    )
    labels = (
        alt.Chart(coverage)
        .mark_text(fontWeight="bold")
        .encode(
            x=alt.X("engine:N", sort=engine_order),
            y=alt.Y("target_preparation:N", sort=target_order),
            text="comparison_coverage:N",
            color=alt.condition(
                alt.datum.comparison_fraction >= 0.65,
                alt.value("white"),
                alt.value("#7f1d1d"),
            ),
        )
    )
    st.altair_chart(rectangles + labels, width="stretch")


def _render_target_ligand_launch_selector(
    campaigns: pd.DataFrame,
    *,
    requested_target_id: str,
    requested_launch_id: str,
    collection_selection: dict[str, object],
    scope_edit_enabled: bool | None = None,
) -> tuple[list[str], list[str], list[str]]:
    rows = _target_ligand_launch_rows(campaigns)
    if rows.empty:
        return [], [], []
    launch_rows = (
        rows.groupby("_launch_campaign_id", as_index=False)
        .agg(
            Campaign=("Campaign", "first"),
            Created=("Created", "min"),
            Targets=("_target_run_id", "nunique"),
            _compound_signature=(
                "_compound_signature",
                lambda values: "||".join(
                    sorted(
                        {
                            member
                            for value in values
                            for member in str(value).split("||")
                            if member
                        }
                    )
                ),
            ),
            _campaign_purpose=("_campaign_purpose", "first"),
        )
    )
    launch_rows = launch_rows.sort_values("Created", ascending=False)
    launch_options = launch_rows["_launch_campaign_id"].astype(str).tolist()
    launch_labels = {
        str(row["_launch_campaign_id"]): (
            f"{row['Campaign']} · {int(row['Targets'])} target(s) · "
            f"{pd.Timestamp(row['Created']).strftime('%Y-%m-%d %H:%M')}"
        )
        for _, row in launch_rows.iterrows()
    }
    requested_launches_from_collection = [
        str(value)
        for value in collection_selection.get("launch_campaign_ids") or []
        if str(value) in launch_options
    ]
    if requested_launches_from_collection:
        # An Analysis Set is the authoritative saved scope. Requiring one of
        # its physical campaigns to masquerade as an anchor is both redundant
        # and misleading, especially when the set intentionally has many.
        selected_scope_launches = requested_launches_from_collection
        st.info(
            f"Analysis Set scope · {len(selected_scope_launches)} physical "
            "campaign(s). Edit or create Analysis Sets from Campaign Results."
        )
    else:
        initial_launch = (
            requested_launch_id
            if requested_launch_id in launch_options
            else launch_options[0]
        )
        compatibility_anchor = st.selectbox(
            "Campaign",
            launch_options,
            index=launch_options.index(initial_launch),
            format_func=lambda value: launch_labels.get(str(value), str(value)),
            key="campaign_compare_scope_campaign",
            on_change=_reset_prepared_target_scope,
            help=(
                "The physical campaign shown by default. Historical campaigns "
                "can be combined into an Analysis Set from Campaign Results."
            ),
        )
        selected_scope_launches = [str(compatibility_anchor)]
    rows = rows.loc[
        rows["_launch_campaign_id"].astype(str).isin(
            [str(value) for value in selected_scope_launches]
        )
    ].reset_index(drop=True)
    edit_scope = (
        bool(scope_edit_enabled)
        if scope_edit_enabled is not None
        else st.toggle(
            "Edit prepared-target scope",
            value=False,
            key="campaign_edit_target_scope",
            help=(
                "The current Analysis Set or page link defines the initial "
                "target scope. Open this only when targets need to be added or "
                "removed."
            ),
        )
    )
    query = (
        st.text_input(
            "Search prepared targets",
            key="campaign_compare_target_search",
            placeholder="PDB, preparation, campaign, tool, or target job",
        ).strip().lower()
        if edit_scope
        else ""
    )
    visible = rows.loc[
        rows.apply(
            lambda row: not query
            or query
            in " ".join(
                str(value)
                for key, value in row.items()
                if not str(key).startswith("_")
            ).lower(),
            axis=1,
        )
    ].copy()
    if edit_scope:
        order_columns = {
            "Prepared target": "Target",
            "PDB / origin": "PDB / origin",
            "Target job": "Target job",
            "Preparation tool": "Tool",
            "Origin history": "Origin",
            "Last preparation step": "Last step",
            "Residue count": "Residues",
            "Created": "Created",
        }
        order_label = st.selectbox(
            "Order targets by",
            list(order_columns),
            key="campaign_compare_target_order_column",
            help=(
                "This server-side ordering preserves the selected target IDs "
                "and reapplies their row selectors after sorting."
            ),
        )
        order_direction = st.segmented_control(
            "Order direction",
            ("Ascending", "Descending"),
            default="Ascending",
            key="campaign_compare_target_order_direction",
        ) or "Ascending"
        visible = visible.sort_values(
            order_columns[str(order_label)],
            ascending=order_direction == "Ascending",
            na_position="last",
            kind="stable",
        )
    visible = visible.reset_index(drop=True)
    valid_selection_ids = set(rows["_selection_id"].astype(str))
    requested_selection_ids = {
        str(value)
        for value in collection_selection.get("target_launch_pairs") or []
    }
    requested_launches = {
        str(value) for value in selected_scope_launches
    }
    requested_targets = {
        str(value)
        for value in collection_selection.get("target_run_ids") or []
    }
    if requested_target_id:
        requested_targets = {requested_target_id}
    if not requested_selection_ids and (requested_launches or requested_targets):
        requested_selection_ids = set(
            rows.loc[
                (
                    rows["_launch_campaign_id"].astype(str).isin(
                        requested_launches
                    )
                    if requested_launches
                    else True
                )
                & (
                    rows["_target_run_id"].astype(str).isin(
                        requested_targets
                    )
                    if requested_targets
                    else True
                ),
                "_selection_id",
            ].astype(str)
        )
    if not requested_selection_ids:
        requested_selection_ids = valid_selection_ids
    state_key = "campaign_compare_target_launch_ids"
    if state_key not in st.session_state:
        st.session_state[state_key] = sorted(
            requested_selection_ids & valid_selection_ids
        )
    raw_current = {
        str(value) for value in st.session_state.get(state_key) or []
    }
    current = raw_current & valid_selection_ids
    if raw_current and not current and requested_selection_ids:
        current = requested_selection_ids & valid_selection_ids
        st.session_state[state_key] = sorted(current)
    explicit_empty_key = "campaign_compare_target_launch_explicit_empty_v2"
    if (
        not current
        and requested_selection_ids
        and not st.session_state.get(explicit_empty_key, False)
    ):
        # Recover scopes cleared by older table-selection behavior (including
        # sorting a dataframe, which emits an empty selection event).
        current = requested_selection_ids & valid_selection_ids
        st.session_state[state_key] = sorted(current)
    if edit_scope:
        st.caption(
            "Select prepared targets with the table's row selectors. "
            "Use Order targets by above instead of the column headers so row "
            "selectors are reapplied visibly. Selections outside the current "
            "search remain unchanged."
        )
        visible_selection_ids = set(visible["_selection_id"].astype(str))
        action_columns = st.columns([1, 1, 5])
        select_all_shown = action_columns[0].button(
            "Select all in table",
            key="campaign_compare_target_select_all_shown",
            help=(
                "Select every currently filtered target in the campaign "
                "chosen above."
            ),
        )
        clear_shown = action_columns[1].button(
            "Clear shown",
            key="campaign_compare_target_clear_shown",
        )
        revision_key = "campaign_compare_target_launch_table_revision"
        revision = int(st.session_state.get(revision_key, 0) or 0)
        if select_all_shown:
            current |= visible_selection_ids
            st.session_state[state_key] = sorted(current)
            st.session_state[explicit_empty_key] = False
            revision += 1
            st.session_state[revision_key] = revision
        elif clear_shown:
            current -= visible_selection_ids
            st.session_state[state_key] = sorted(current)
            st.session_state[explicit_empty_key] = not current
            revision += 1
            st.session_state[revision_key] = revision
        visible.insert(
            0,
            "In scope",
            visible["_selection_id"].astype(str).map(
                lambda selection_id: "✓" if selection_id in current else ""
            ),
        )
        default_rows = [
            index
            for index, selection_id in enumerate(
                visible["_selection_id"].astype(str)
            )
            if selection_id in current
        ]
        table_event = st.dataframe(
            visible,
            hide_index=True,
            width="stretch",
            height=min(430, 38 + 35 * len(visible)),
            key=(
                "campaign_compare_target_launch_table::"
                f"{hash(tuple(visible['_selection_id'].astype(str)))}::"
                f"{query or 'all'}::{revision}"
            ),
            on_select="rerun",
            selection_mode="multi-row",
            selection_default={"selection": {"rows": default_rows}},
            column_order=[
                column
                for column in visible.columns
                if not str(column).startswith("_")
            ],
            column_config={
                "Target job": st.column_config.LinkColumn(
                    "Target job",
                    display_text=r"label=([^&]+)",
                ),
                "Residues": st.column_config.NumberColumn(format="%d"),
                "Created": st.column_config.DatetimeColumn(
                    format="YYYY-MM-DD HH:mm"
                ),
                "_target_run_id": None,
                "_launch_campaign_id": None,
                "_selection_id": None,
            },
        )
        selected_visible = set(
            visible.iloc[
                [
                    int(index)
                    for index in table_event.selection.rows
                    if 0 <= int(index) < len(visible)
                ]
            ]["_selection_id"].astype(str)
        )
        if selected_visible:
            selected_selection_ids = (
                current - visible_selection_ids
            ) | selected_visible
            st.session_state[explicit_empty_key] = False
        else:
            # Streamlit reports an empty row selection when the user sorts the
            # dataframe. Empty events are therefore non-destructive; clearing
            # a scope is an explicit action through "Clear shown" above.
            selected_selection_ids = current
    else:
        selected_selection_ids = current
        st.caption(
            f"{len(current)} prepared-target campaign(s) selected. "
            "Use Edit prepared-target scope to change them."
        )
    st.session_state[state_key] = sorted(selected_selection_ids)
    selected_rows = rows.loc[
        rows["_selection_id"].astype(str).isin(selected_selection_ids)
    ]
    selected_targets = rows.loc[
        rows["_selection_id"].astype(str).isin(selected_selection_ids),
        "_target_run_id",
    ].astype(str).drop_duplicates().tolist()
    selected_launches = (
        selected_rows["_launch_campaign_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    return (
        selected_targets,
        selected_launches,
        sorted(selected_selection_ids),
    )


def _render_compound_campaign_selectors(
    campaign_scope: pd.DataFrame,
    *,
    requested_dataset: str,
    requested_target_id: str,
    requested_launch_id: str,
    requested_collection: dict[str, object] | None,
    collection_selection: dict[str, object],
) -> tuple[
    str,
    dict[str, str],
    list[str],
    list[str],
    list[str],
    list[str],
    pd.DataFrame,
    pd.DataFrame,
]:
    dataset_rows = campaign_scope[
        ["dataset_run_id", "dataset"]
    ].drop_duplicates()
    dataset_options = dataset_rows["dataset_run_id"].tolist()
    dataset_labels = dict(
        zip(dataset_rows["dataset_run_id"], dataset_rows["dataset"])
    )
    dataset_index = (
        dataset_options.index(requested_dataset)
        if requested_dataset in dataset_options
        else 0
    )
    filter_columns = st.columns(4)
    selected_dataset = filter_columns[0].selectbox(
        "Compound dataset",
        dataset_options,
        index=dataset_index,
        format_func=lambda value: dataset_labels.get(value, value),
        key="campaign_compare_dataset",
        on_change=_reset_dataset_dependent_controls,
        help=(
            "Comparisons are scoped to one immutable source compound dataset. "
            "Open this page from Compound Dataset Results to preselect it."
        ),
    )
    dataset_campaigns = campaign_scope.loc[
        campaign_scope["dataset_run_id"].eq(selected_dataset)
    ].copy()
    target_columns = [
        column
        for column in (
            "target_run_id",
            "selected_target_key",
            "selected_target",
            "selected_target_run_id",
            "coordinate_target_key",
            "target",
        )
        if column in dataset_campaigns
    ]
    targets = dataset_campaigns[target_columns].drop_duplicates(
        subset=["target_run_id"]
    )
    target_options = targets["target_run_id"].tolist()
    target_label_rows: list[tuple[str, str, str]] = []
    for row in targets.itertuples(index=False):
        row_payload = row._asdict()
        target_id = str(row_payload.get("target_run_id") or "")
        selected_key = str(
            row_payload.get("selected_target")
            or row_payload.get("selected_target_key")
            or row_payload.get("target")
            or target_id
        )
        coordinate_key = str(
            row_payload.get("coordinate_target_key")
            or _short_target_key(str(row_payload.get("target") or ""))
            or target_id
        )
        target_label_rows.append((target_id, selected_key, coordinate_key))
    selected_key_counts = Counter(
        selected_key for _, selected_key, _ in target_label_rows
    )
    target_labels = {
        target_id: (
            f"{selected_key} → coordinates {coordinate_key}"
            if selected_key_counts[selected_key] > 1
            and coordinate_key != selected_key
            else selected_key
        )
        for target_id, selected_key, coordinate_key in target_label_rows
    }
    requested_targets = {
        str(value)
        for value in collection_selection.get("target_run_ids") or []
    }
    if requested_target_id:
        requested_targets.add(requested_target_id)
    default_targets = (
        [
            value
            for value in target_options
            if value in requested_targets
        ]
        if requested_collection is not None or requested_target_id
        else target_options
    )
    _reconcile_cascading_multiselect(
        key="campaign_compare_target",
        context_key="_campaign_targets_context",
        context=("dataset", selected_dataset),
        options=target_options,
        defaults=default_targets,
    )
    selected_targets = filter_columns[1].multiselect(
        "Targets",
        target_options,
        format_func=lambda value: target_labels.get(value, value),
        key="campaign_compare_target",
        help=(
            "Labels use the target selected before campaign launch, including "
            "its ordered preparation-step abbreviations and original job code. "
            "When that target has multiple coordinate-oriented derivatives, "
            "the exact coordinate key is appended only to distinguish them."
        ),
    )
    target_campaigns = dataset_campaigns.loc[
        dataset_campaigns["target_run_id"].isin(selected_targets)
    ].copy()
    primary_target_campaigns = target_campaigns.loc[
        ~target_campaigns["engine"].astype(str).str.endswith("rescoring")
    ].copy()
    launch_rows = primary_target_campaigns[
        ["launch_campaign_id", "launch_campaign"]
    ].drop_duplicates()
    launch_options = launch_rows["launch_campaign_id"].tolist()
    launch_labels = dict(
        zip(
            launch_rows["launch_campaign_id"],
            launch_rows["launch_campaign"],
        )
    )
    requested_launches = {
        str(value)
        for value in collection_selection.get("launch_campaign_ids") or []
    }
    if requested_launch_id:
        requested_launches.add(requested_launch_id)
    default_launches = (
        [
            value
            for value in launch_options
            if value in requested_launches
        ]
        if requested_collection is not None or requested_launch_id
        else launch_options
    )
    _reconcile_cascading_multiselect(
        key="campaign_compare_launches",
        context_key="_campaign_launches_context",
        context=(
            "dataset",
            selected_dataset,
            "targets",
            "|".join(
                sorted(str(value) for value in selected_targets)
            ),
        ),
        options=launch_options,
        defaults=default_launches,
    )
    selected_launches = filter_columns[2].multiselect(
        "Launch campaigns",
        launch_options,
        format_func=lambda value: launch_labels.get(value, value),
        key="campaign_compare_launches",
        help=(
            "One launch campaign contains the engine jobs queued together from "
            "Docking / Cofolding. Historical campaigns are reconstructed from "
            "their immutable inputs and launch times."
        ),
    )
    launch_campaigns = primary_target_campaigns.loc[
        primary_target_campaigns["launch_campaign_id"].isin(
            selected_launches
        )
    ].copy()
    available_engines = sorted(launch_campaigns["engine"].unique())
    requested_engines = {
        str(value)
        for value in collection_selection.get("engines") or []
    }
    default_engines = (
        [
            value
            for value in available_engines
            if value in requested_engines
        ]
        if requested_collection is not None
        else available_engines
    )
    _reconcile_cascading_multiselect(
        key="campaign_compare_engines",
        context_key="_campaign_engines_context",
        context=(
            "dataset",
            selected_dataset,
            "targets",
            "|".join(
                sorted(str(value) for value in selected_targets)
            ),
            "launches",
            "|".join(
                sorted(str(value) for value in selected_launches)
            ),
        ),
        options=available_engines,
        defaults=default_engines,
    )
    selected_engines = filter_columns[3].multiselect(
        "Engines",
        available_engines,
        key="campaign_compare_engines",
    )
    selectable = launch_campaigns.loc[
        launch_campaigns["engine"].isin(selected_engines)
    ]
    campaign_options = selectable["campaign_id"].tolist()
    campaign_labels = dict(
        zip(selectable["campaign_id"], selectable["engine_run"])
    )
    with st.expander("Engine-run filtering", expanded=False):
        st.caption(
            "Normally keep every engine run in the selected launch campaigns. "
            "Use this only to distinguish repeated or alternative runs of the "
            "same engine."
        )
        requested_engine_runs = {
            str(value)
            for value in collection_selection.get("engine_run_ids") or []
        }
        default_engine_runs = (
            [
                value
                for value in campaign_options
                if value in requested_engine_runs
            ]
            if requested_collection is not None
            else campaign_options
        )
        _reconcile_cascading_multiselect(
            key="campaign_compare_campaigns",
            context_key="_campaign_engine_runs_context",
            context=(
                "dataset",
                selected_dataset,
                "targets",
                "|".join(
                    sorted(str(value) for value in selected_targets)
                ),
                "launches",
                "|".join(
                    sorted(str(value) for value in selected_launches)
                ),
                "engines",
                "|".join(
                    sorted(str(value) for value in selected_engines)
                ),
            ),
            options=campaign_options,
            defaults=default_engine_runs,
        )
        selected_campaigns = st.multiselect(
            "Engine runs",
            campaign_options,
            format_func=lambda value: campaign_labels.get(value, value),
            key="campaign_compare_campaigns",
        )
    return (
        selected_dataset,
        dataset_labels,
        list(selected_targets),
        list(selected_launches),
        list(selected_engines),
        list(selected_campaigns),
        target_campaigns,
        selectable,
    )


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _comparison_collections(run_root: Path) -> list[dict[str, object]]:
    return list_analysis_sets(run_root)


def _save_comparison_collection(
    run_root: Path,
    *,
    name: str,
    description: str,
    selection: dict[str, object],
) -> dict[str, object]:
    return save_analysis_set(
        run_root,
        name=name,
        description=description,
        selection=selection,
    )


def _source_run_dir(run_root: Path, run_id: str) -> Path | None:
    if not run_id:
        return None
    for task_group in run_root.iterdir():
        candidate = task_group / run_id
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _artifact_payload_path(
    run_root: Path,
    payload: dict,
) -> Path | None:
    run_id = str(payload.get("run_id") or "")
    source_dir = _source_run_dir(run_root, run_id)
    raw_path = str(payload.get("path") or "")
    if source_dir is None or not raw_path:
        return None
    path = Path(raw_path)
    candidate = (
        path.resolve()
        if path.is_absolute()
        else (source_dir / path).resolve()
    )
    return candidate if candidate.is_file() else None


def _sibling_artifact_path(
    run_root: Path,
    run_id: str,
    artifact_types: set[str],
) -> Path | None:
    source_dir = _source_run_dir(run_root, run_id)
    if source_dir is None:
        return None
    manifest = _read_json(source_dir / "artifacts.json")
    artifacts = manifest.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    for artifact in artifacts:
        if (
            isinstance(artifact, dict)
            and str(artifact.get("artifact_type") or "")
            in artifact_types
        ):
            path = _artifact_payload_path(run_root, artifact)
            if path is not None:
                return path
    return None


def _engine_name(workflow: str, tool: str) -> str:
    normalized = tool.strip().lower()
    if workflow == "docking_campaign":
        return {
            "vina": "AutoDock Vina",
            "gnina": "GNINA",
            "udp": "Uni-Dock Pro",
            "unidock": "Uni-Dock Pro",
        }.get(normalized, tool or workflow)
    if workflow == "openvs_docking":
        return "RosettaLigand"
    if workflow == "alphafold3_refolding":
        return "AlphaFold 3"
    if workflow == "boltz2_refolding":
        return "Boltz-2"
    if workflow == "nesso_affinity":
        return "Nesso-1"
    if workflow == "gnina_rescoring":
        return "GNINA rescoring"
    if workflow == "boltzina_rescoring":
        return "Boltzina rescoring"
    return tool or workflow or "Unknown"


def _source_campaign_payload(
    run_root: Path,
    *,
    task_group: str,
    workflow: str,
    payload: dict,
) -> dict:
    if task_group != "rescoring" or not workflow.endswith("_rescoring"):
        return payload
    pose_selection = payload.get("pose_selection")
    pose_selection = (
        pose_selection if isinstance(pose_selection, dict) else {}
    )
    selection_id = str(pose_selection.get("run_id") or "")
    if not selection_id:
        return payload
    selection_payload = _read_json(
        run_root / "rescoring" / selection_id / "input.json"
    )
    source_job = selection_payload.get("source_job")
    source_job = source_job if isinstance(source_job, dict) else {}
    source_id = str(source_job.get("run_id") or "")
    source_group = str(source_job.get("task_group") or "docking")
    if not source_id:
        return payload
    return _read_json(
        run_root / source_group / source_id / "input.json"
    ) or payload


def _comparison_target_artifact(
    run_root: Path,
    target: dict[str, object],
) -> dict[str, object]:
    """Use the source target identity for coordinate-only derivatives."""
    prepared = dict(target)
    seen: set[str] = set()
    while True:
        run_id = str(prepared.get("run_id") or "")
        metadata = prepared.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if (
            not run_id
            or run_id in seen
            or not isinstance(metadata.get("coordinate_transform"), dict)
        ):
            return prepared
        seen.add(run_id)
        orientation_input = _read_json(
            run_root / "target-orientation" / run_id / "input.json"
        )
        source_target = orientation_input.get("source_target")
        if not isinstance(source_target, dict):
            return prepared
        prepared = dict(source_target)


def _short_target_key(compact_identifier: str) -> str:
    """Match the target selector's `origin · short job code` identity."""
    parts = [part.strip() for part in compact_identifier.split("·")]
    parts = [part for part in parts if part]
    if len(parts) < 2:
        return compact_identifier
    return f"{parts[0]} · {parts[-1]}"


def _render_prepared_target_code_legend() -> None:
    st.caption(
        "Prepared-target names use `PDB · ordered preparation codes · "
        "target job`. The compact name is human-readable; the immutable "
        "target run ID remains the authoritative identity."
    )
    with st.expander("Prepared-target code legend"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Code": code,
                        "Preparation step": kind.replace("_", " ").title(),
                    }
                    for kind, code in TARGET_PREPARATION_STEP_CODES.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def _rescoring_source_run_id(payload: dict) -> str:
    pose_selection = payload.get("pose_selection")
    pose_selection = (
        pose_selection if isinstance(pose_selection, dict) else {}
    )
    return str(
        pose_selection.get("source_run_id")
        or payload.get("source_run_id")
        or ""
    )


def _coordinate_signature(parameters: dict) -> str:
    values: list[str] = []
    for field in ("center", "size"):
        coordinates = parameters.get(field)
        coordinates = coordinates if isinstance(coordinates, dict) else {}
        values.extend(
            f"{float(coordinates.get(axis) or 0.0):.3f}"
            for axis in "xyz"
        )
    return "|".join(values) if any(value != "0.000" for value in values) else ""


def _assign_launch_campaigns(campaigns: pd.DataFrame) -> pd.DataFrame:
    """Attach explicit launch IDs or conservatively infer historical batches."""
    if campaigns.empty:
        return campaigns
    prepared = campaigns.copy()
    prepared["_created"] = pd.to_datetime(
        prepared["created_at"], errors="coerce", utc=True
    )
    primary_mask = ~prepared["engine"].astype(str).str.endswith("rescoring")
    primary = prepared.loc[primary_mask].copy()
    assigned: dict[str, str] = {}

    explicit = primary.loc[
        primary["launch_campaign_id"].astype(str).str.strip().ne("")
    ]
    assigned.update(
        dict(zip(explicit["campaign_id"], explicit["launch_campaign_id"]))
    )
    historical = primary.loc[
        primary["launch_campaign_id"].astype(str).str.strip().eq("")
    ]
    classical_engines = {
        "AutoDock Vina",
        "GNINA",
        "Uni-Dock Pro",
        "RosettaLigand",
    }
    for _, family in historical.groupby(
        ["dataset_run_id", "target_run_id", "selection_run_id"],
        dropna=False,
        sort=False,
    ):
        family = family.sort_values("_created")
        anchors: list[dict[str, object]] = []
        classical = family.loc[
            family["engine"].astype(str).isin(classical_engines)
        ]
        for _, row in classical.iterrows():
            signature = str(row.get("_coordinate_signature") or "")
            created = row["_created"]
            matching = next(
                (
                    anchor
                    for anchor in reversed(anchors)
                    if anchor["signature"] == signature
                    and pd.notna(created)
                    and pd.notna(anchor["latest"])
                    and abs((created - anchor["latest"]).total_seconds())
                    <= 900
                ),
                None,
            )
            if matching is None:
                matching = {
                    "id": f"historical:{row['campaign_id']}",
                    "signature": signature,
                    "latest": created,
                }
                anchors.append(matching)
            else:
                matching["latest"] = created
            assigned[str(row["campaign_id"])] = str(matching["id"])

        nonclassical = family.loc[
            ~family["engine"].astype(str).isin(classical_engines)
        ]
        fallback_anchor: dict[str, object] | None = None
        for _, row in nonclassical.iterrows():
            created = row["_created"]
            eligible = [
                anchor
                for anchor in anchors
                if pd.notna(created)
                and pd.notna(anchor["latest"])
                and abs((created - anchor["latest"]).total_seconds())
                <= 900
            ]
            if eligible:
                matching = min(
                    eligible,
                    key=lambda anchor: abs(
                        (created - anchor["latest"]).total_seconds()
                    ),
                )
            elif (
                fallback_anchor is not None
                and pd.notna(created)
                and pd.notna(fallback_anchor["latest"])
                and abs(
                    (created - fallback_anchor["latest"]).total_seconds()
                )
                <= 900
            ):
                matching = fallback_anchor
            else:
                matching = {
                    "id": f"historical:{row['campaign_id']}",
                    "signature": "",
                    "latest": created,
                }
                anchors.append(matching)
                fallback_anchor = matching
            matching["latest"] = created
            assigned[str(row["campaign_id"])] = str(matching["id"])

    prepared["launch_campaign_id"] = prepared["campaign_id"].map(assigned).fillna(
        prepared["launch_campaign_id"]
    )
    source_launches = dict(
        zip(prepared["campaign_id"], prepared["launch_campaign_id"])
    )
    rescoring_mask = ~primary_mask
    prepared.loc[rescoring_mask, "launch_campaign_id"] = (
        prepared.loc[rescoring_mask, "source_campaign_id"]
        .map(source_launches)
        .fillna(
            prepared.loc[rescoring_mask, "launch_campaign_id"]
        )
    )

    labels: dict[str, str] = {}
    for launch_id, rows in prepared.loc[primary_mask].groupby(
        "launch_campaign_id", sort=False
    ):
        explicit_label = next(
            (
                str(value)
                for value in rows["launch_campaign_label"]
                if str(value).strip()
            ),
            "",
        )
        if explicit_label:
            labels[str(launch_id)] = explicit_label
            continue
        created = rows["_created"].min()
        timestamp = (
            created.strftime("%Y-%m-%d %H:%M UTC")
            if pd.notna(created)
            else "Historical launch"
        )
        engines = ", ".join(sorted(rows["engine"].astype(str).unique()))
        labels[str(launch_id)] = f"{timestamp} · {engines}"
    prepared["launch_campaign"] = prepared["launch_campaign_id"].map(labels).fillna(
        "Unlinked historical run"
    )
    return prepared.drop(columns=["_created"])


def _prepared_metric_rows(
    frame: pd.DataFrame,
    *,
    engine: str,
) -> pd.DataFrame:
    """Preserve every emitted prediction row and normalize its compound ID."""
    prepared = frame.copy()
    del engine  # Kept in the signature for explicit call-site provenance.
    candidate_source = next(
        (
            column
            for column in ("compound_id", "candidate_id")
            if column in prepared
        ),
        "",
    )
    if not candidate_source:
        return pd.DataFrame()
    prepared["candidate_id"] = prepared[candidate_source].astype(str)
    return prepared.loc[prepared["candidate_id"].str.strip().ne("")].copy()


def _normalized_compound_column(column: object) -> str:
    return "_".join(
        str(column).strip().lower().replace("-", " ").split()
    )


def _compound_name_mapping(
    compounds: pd.DataFrame,
    *,
    identifier_column: str,
) -> tuple[dict[str, str], str]:
    """Map IDs to genuine source names without treating IDs as names."""
    normalized_columns = {
        _normalized_compound_column(column): str(column)
        for column in compounds.columns
    }
    name_column = next(
        (
            normalized_columns[candidate]
            for candidate in COMPOUND_NAME_COLUMN_CANDIDATES
            if candidate in normalized_columns
        ),
        "",
    )
    if not name_column:
        return {}, ""
    mapping: dict[str, str] = {}
    for identifier, raw_name in zip(
        compounds[identifier_column],
        compounds[name_column],
    ):
        compound_id = str(identifier).strip()
        compound_name = str(raw_name).strip()
        if (
            not compound_id
            or not compound_name
            or compound_name.lower() in {"nan", "none", "<na>"}
            or compound_name.casefold() == compound_id.casefold()
        ):
            continue
        mapping.setdefault(compound_id, compound_name)
    return mapping, name_column


def _compound_set_csv_for_run(run_root: Path, run_id: str) -> Path | None:
    run_dir = _source_run_dir(run_root, run_id)
    if run_dir is None:
        return None
    manifest = _read_json(run_dir / "artifacts.json")
    artifacts = manifest.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    artifact = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict)
            and str(item.get("artifact_type") or "") == "compound_set"
            and str(item.get("path") or "").lower().endswith(".csv")
        ),
        None,
    )
    return (
        _artifact_payload_path(run_root, artifact)
        if artifact is not None
        else None
    )


def _campaign_runs_revision(run_root: Path) -> tuple[int, int, int]:
    """Return a cheap cache key that changes when campaign jobs change."""
    file_count = 0
    newest_mtime_ns = 0
    total_size = 0
    for task_group in ("docking", "refolding", "rescoring"):
        group_dir = run_root / task_group
        if not group_dir.is_dir():
            continue
        for run_dir in group_dir.iterdir():
            if not run_dir.is_dir():
                continue
            for name in (
                "metadata.json",
                "input.json",
                "artifact_manifest.json",
                "result.json",
            ):
                path = run_dir / name
                try:
                    stat = path.stat()
                except OSError:
                    continue
                file_count += 1
                newest_mtime_ns = max(newest_mtime_ns, stat.st_mtime_ns)
                total_size += stat.st_size
    return file_count, newest_mtime_ns, total_size


@st.cache_data(show_spinner=False)
def load_campaign_comparison_data(
    run_root: str,
    revision: tuple[int, int, int] | None = None,
    dataset_run_id: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load completed target-based campaigns for dataset-level comparison."""
    del revision  # Its value participates in the Streamlit cache key.
    root = Path(run_root)
    requested_dataset_run_id = str(dataset_run_id or "").strip()
    campaigns: list[dict[str, object]] = []
    metric_frames: list[pd.DataFrame] = []
    target_identity_cache: dict[str, tuple[str, str]] = {}

    def target_identity(
        run_id: str,
        *,
        fallback_origin: str,
    ) -> tuple[str, str]:
        if run_id in target_identity_cache:
            return target_identity_cache[run_id]
        target_dir = _source_run_dir(root, run_id)
        target_metadata = (
            _read_json(target_dir / "metadata.json")
            if target_dir is not None
            else {}
        )
        target_identifier = compact_target_identifier(
            run_id=run_id,
            metadata=target_metadata,
            fallback_origin=fallback_origin,
        )
        history = target_metadata.get("modification_history")
        history = history if isinstance(history, list) else []
        origin_head = (
            "PDB"
            if target_metadata.get("pdb_id")
            else str(target_metadata.get("source") or "Imported target")
            .replace("_", " ")
            .title()
        )
        target_origin = " → ".join(
            [
                origin_head,
                *[
                    str(item.get("label") or "").strip()
                    for item in history
                    if isinstance(item, dict)
                    and str(item.get("label") or "").strip()
                ],
            ]
        )
        target_identity_cache[run_id] = (
            target_identifier,
            target_origin,
        )
        return target_identity_cache[run_id]

    for job in iter_job_records(
        root,
        task_groups=("docking", "refolding", "rescoring"),
        validate_artifacts=False,
    ):
        if job.status != "completed":
            continue
        engine = _engine_name(job.workflow, job.tool)
        if engine not in ENGINE_METRICS:
            continue
        payload = _read_json(job.run_dir / "input.json")
        source_payload = _source_campaign_payload(
            root,
            task_group=job.task_group,
            workflow=job.workflow,
            payload=payload,
        )
        coordinate_target = (
            source_payload.get("target")
            or source_payload.get("target_artifact")
        )
        coordinate_target = (
            coordinate_target
            if isinstance(coordinate_target, dict)
            else {}
        )
        target = _comparison_target_artifact(root, coordinate_target)
        compound_sets = (
            source_payload.get("compound_sets")
            or source_payload.get("compound_artifacts")
        )
        compound_sets = (
            compound_sets if isinstance(compound_sets, list) else []
        )
        compound_set = next(
            (value for value in compound_sets if isinstance(value, dict)),
            {},
        )
        reference_ligand = source_payload.get("reference_ligand_artifact")
        reference_ligand = (
            reference_ligand
            if isinstance(reference_ligand, dict)
            else compound_set
        )
        parameters = source_payload.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        compound_metadata = compound_set.get("metadata")
        compound_metadata = (
            compound_metadata
            if isinstance(compound_metadata, dict)
            else {}
        )
        target_family_run_id = str(
            target.get("run_id")
            or job.metadata.get("parent_run_id")
            or job.parent_run_id
            or ""
        )
        coordinate_target_run_id = str(
            coordinate_target.get("run_id") or target_family_run_id
        )
        target_artifact_label = str(
            coordinate_target.get("label")
            or coordinate_target.get("path")
            or target.get("label")
            or target.get("path")
            or target_family_run_id
            or "Unknown target"
        )
        target_run_id = coordinate_target_run_id or target_family_run_id
        target_label, target_origin = target_identity(
            target_run_id,
            fallback_origin=target_artifact_label,
        )
        selected_target_label, _ = target_identity(
            target_family_run_id,
            fallback_origin=str(
                target.get("label")
                or target.get("path")
                or target_artifact_label
            ),
        )
        target_path = _artifact_payload_path(root, coordinate_target)
        reference_complex_path = _sibling_artifact_path(
            root,
            coordinate_target_run_id,
            {"prepared_complex"},
        )
        dataset_run_id = str(
            compound_metadata.get("source_compound_run_id")
            or compound_set.get("run_id")
            or ""
        )
        # Dataset deep links are the normal entry point for this page. Discard
        # unrelated jobs before opening compound tables, metric CSVs, pose
        # files, or cofolding inputs. Without this guard a HY216 page paid the
        # full I/O cost of every historical campaign in the workspace.
        if (
            requested_dataset_run_id
            and dataset_run_id != requested_dataset_run_id
        ):
            continue
        dataset_label = str(
            compound_set.get("label")
            or dataset_run_id
            or "Unknown compound set"
        )
        compound_smiles: dict[str, str] = {}
        compound_default_smiles = ""
        compound_names: dict[str, str] = {}
        compound_name_sources: dict[str, str] = {}
        selection_compound_names: dict[str, str] = {}
        selection_name_column = ""
        compound_set_path = _artifact_payload_path(root, compound_set)
        if (
            compound_set_path is not None
            and compound_set_path.suffix.lower() == ".csv"
        ):
            try:
                compound_frame = pd.read_csv(compound_set_path)
                identifier_column = next(
                    (
                        column
                        for column in (
                            "compound_id",
                            "candidate_id",
                            "representative_compound_id",
                        )
                        if column in compound_frame
                    ),
                    "",
                )
                smiles_column = next(
                    (
                        column
                        for column in (
                            "smiles",
                            "standardized_parent_smiles",
                            "canonical_smiles",
                        )
                        if column in compound_frame
                    ),
                    "",
                )
                if identifier_column and smiles_column:
                    compound_smiles = {
                        str(identifier): str(smiles)
                        for identifier, smiles in zip(
                            compound_frame[identifier_column],
                            compound_frame[smiles_column],
                        )
                        if str(identifier).strip()
                        and str(smiles).strip()
                        and str(smiles).lower() != "nan"
                    }
                if identifier_column:
                    (
                        selection_compound_names,
                        selection_name_column,
                    ) = _compound_name_mapping(
                        compound_frame,
                        identifier_column=identifier_column,
                    )
            except (OSError, ValueError, pd.errors.ParserError):
                compound_smiles = {}
        elif (
            compound_set_path is not None
            and compound_set_path.suffix.lower() in {".sdf", ".mol"}
        ):
            try:
                from rdkit import Chem

                supplier = Chem.SDMolSupplier(
                    str(compound_set_path), removeHs=True
                )
                molecules = [mol for mol in supplier if mol is not None]
                for index, molecule in enumerate(molecules, start=1):
                    smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
                    if not smiles:
                        continue
                    if len(molecules) == 1:
                        compound_default_smiles = smiles
                    identifiers = {
                        str(molecule.GetProp("_Name") or "").strip(),
                        f"compound_{index:07d}",
                    }
                    for property_name in molecule.GetPropNames():
                        if property_name.lower() in {
                            "compound_id",
                            "candidate_id",
                            "id",
                        }:
                            identifiers.add(
                                str(molecule.GetProp(property_name)).strip()
                            )
                    compound_smiles.update(
                        {
                            identifier: smiles
                            for identifier in identifiers
                            if identifier
                        }
                    )
            except (ImportError, OSError, RuntimeError, ValueError):
                compound_smiles = {}
                compound_default_smiles = ""
        source_compound_path = _compound_set_csv_for_run(
            root,
            dataset_run_id,
        )
        if source_compound_path is not None:
            try:
                source_compounds = pd.read_csv(source_compound_path)
                source_identifier_column = next(
                    (
                        column
                        for column in ("compound_id", "candidate_id")
                        if column in source_compounds
                    ),
                    "",
                )
                if source_identifier_column:
                    source_names, source_name_column = _compound_name_mapping(
                        source_compounds,
                        identifier_column=source_identifier_column,
                    )
                    compound_names.update(source_names)
                    compound_name_sources.update(
                        {
                            compound_id: source_name_column
                            for compound_id in source_names
                        }
                    )
            except (OSError, ValueError, pd.errors.ParserError):
                pass
        for compound_id, compound_name in selection_compound_names.items():
            if compound_id in compound_names:
                continue
            compound_names[compound_id] = compound_name
            compound_name_sources[compound_id] = selection_name_column
        # A CSV campaign artifact preserves the exact shared modeling SMILES
        # used to build the cofolding inputs. AlphaFold JSON inputs can embed a
        # multi-megabyte MSA in every compound file, so opening them merely to
        # recover the already indexed ligand SMILES makes a cold comparison
        # load scale with the duplicated MSA corpus. Only inspect engine inputs
        # for legacy/SDF campaigns that did not provide an identifier-to-SMILES
        # mapping in their compound artifact.
        cofold_input_paths: list[Path] = []
        if not compound_smiles:
            cofold_input_paths = [
                *sorted((job.run_dir / "inputs").glob("*.json")),
                *sorted((job.run_dir / "data").glob("*.json")),
                *sorted((job.run_dir / "inputs").glob("*.yaml")),
                *sorted((job.run_dir / "inputs").glob("*.yml")),
            ]
            for cofold_input_path in cofold_input_paths:
                try:
                    if cofold_input_path.suffix.lower() == ".json":
                        cofold_payload = _read_json(cofold_input_path)
                    else:
                        import yaml

                        loaded_payload = yaml.safe_load(
                            cofold_input_path.read_text()
                        )
                        cofold_payload = (
                            loaded_payload
                            if isinstance(loaded_payload, dict)
                            else {}
                        )
                except (ImportError, OSError, ValueError):
                    continue
                sequences = cofold_payload.get("sequences")
                sequences = sequences if isinstance(sequences, list) else []
                exact_smiles = next(
                    (
                        str(ligand.get("smiles") or "").strip()
                        for sequence in sequences
                        if isinstance(sequence, dict)
                        and isinstance(sequence.get("ligand"), dict)
                        for ligand in [sequence["ligand"]]
                        if str(ligand.get("smiles") or "").strip()
                    ),
                    "",
                )
                if not exact_smiles:
                    continue
                identifiers = {
                    str(cofold_payload.get("name") or "").strip(),
                    cofold_input_path.stem,
                }
                compound_smiles.update(
                    {
                        identifier: exact_smiles
                        for identifier in identifiers
                        if identifier
                    }
                )
                if len(cofold_input_paths) == 1:
                    compound_default_smiles = exact_smiles
        job_code = display_job_code(
            job.metadata.get("job_code"), job.run_id
        )
        campaign_label = f"{target_label} · {engine} · {job_code}"
        source_campaign_id = ""
        if job.task_group == "rescoring":
            pose_selection = payload.get("pose_selection")
            pose_selection = (
                pose_selection if isinstance(pose_selection, dict) else {}
            )
            selection_id = str(pose_selection.get("run_id") or "")
            if selection_id:
                selection_payload = _read_json(
                    root / "rescoring" / selection_id / "input.json"
                )
                source_job = selection_payload.get("source_job")
                source_job = (
                    source_job if isinstance(source_job, dict) else {}
                )
                source_campaign_id = str(source_job.get("run_id") or "")
        campaign = {
            "campaign_id": job.run_id,
            "campaign": campaign_label,
            "engine_run": f"{engine} · {job_code}",
            "job_code": job_code,
            "engine": engine,
            "selected_target_key": _short_target_key(
                selected_target_label
            ),
            "selected_target_run_id": target_family_run_id,
            "selected_target": selected_target_label,
            "coordinate_target_key": _short_target_key(target_label),
            "target_run_id": target_run_id,
            "target": target_label,
            "target_origin": target_origin,
            "target_artifact": target_artifact_label,
            "target_family_run_id": target_family_run_id,
            "_coordinate_target_run_id": coordinate_target_run_id,
            "_target_path": (
                str(target_path) if target_path is not None else ""
            ),
            "_reference_complex_path": (
                str(reference_complex_path)
                if reference_complex_path is not None
                else ""
            ),
            "_reference_ligand_path": (
                str(reference_ligand_path)
                if (
                    reference_ligand_path := _artifact_payload_path(
                        root, reference_ligand
                    )
                ) is not None
                else ""
            ),
            "dataset_run_id": dataset_run_id,
            "dataset": dataset_label,
            "selection_run_id": str(compound_set.get("run_id") or ""),
            "launch_campaign_id": str(
                job.metadata.get("launch_campaign_id") or ""
            ),
            "launch_campaign_label": str(
                job.metadata.get("launch_campaign_label") or ""
            ),
            "campaign_purpose": str(
                binding_campaign_purpose(job)
            ),
            "compound_signature": compound_smiles_signature(
                [*compound_smiles.values(), compound_default_smiles]
            ),
            "source_campaign_id": source_campaign_id,
            "_coordinate_signature": _coordinate_signature(parameters),
            "compound_count": int(job.metadata.get("compound_count") or 0),
            "status": job.status,
            "configured_repeats": configured_repetitions(job),
            "completed_repeats": completed_repetitions(job),
            "created_at": job.created_at,
            "result": (
                "./job-results?"
                + urlencode(
                    {
                        "task_group": job.task_group,
                        "run_id": job.run_id,
                        "label": job_code,
                    }
                )
            ),
        }
        metric_artifacts = [
            artifact
            for artifact in job.artifact_manifest.artifacts
            if artifact.artifact_type
            in {
                "prediction_metrics",
                "docking_scores",
                "rescoring_scores",
            }
            if artifact.role != "replicate_summary"
        ]
        loaded = False
        for artifact in metric_artifacts:
            metric_path = artifact.resolve(job.run_dir, must_exist=True)
            if metric_path is None or metric_path.suffix.lower() != ".csv":
                continue
            try:
                frame = pd.read_csv(metric_path)
            except (OSError, ValueError, pd.errors.ParserError):
                continue
            frame = _prepared_metric_rows(frame, engine=engine)
            if frame.empty:
                continue
            if engine == "Boltz-2" and "affinity_pred_value" in frame:
                log_ic50 = pd.to_numeric(
                    frame["affinity_pred_value"], errors="coerce"
                )
                frame["affinity_log10_ic50_uM"] = log_ic50
                frame["ic50_uM"] = log_ic50.map(
                    lambda value: (
                        10.0 ** max(-12.0, min(12.0, float(value)))
                        if pd.notna(value)
                        else float("nan")
                    )
                )
                frame["pIC50"] = 6.0 - log_ic50
            if (
                engine == "Nesso-1"
                and "ic50_uM" not in frame
                and "affinity_log10_ic50_uM" in frame
            ):
                log_ic50 = pd.to_numeric(
                    frame["affinity_log10_ic50_uM"],
                    errors="coerce",
                )
                frame["ic50_uM"] = log_ic50.map(
                    lambda value: (
                        10.0 ** max(-12.0, min(12.0, float(value)))
                        if pd.notna(value)
                        else float("nan")
                    )
                )
            if (
                engine == "Boltzina rescoring"
                and "boltzina_affinity_log10_ic50_uM" in frame
            ):
                log_ic50 = pd.to_numeric(
                    frame["boltzina_affinity_log10_ic50_uM"],
                    errors="coerce",
                )
                frame["boltzina_ic50_uM"] = log_ic50.map(
                    lambda value: (
                        10.0 ** max(-12.0, min(12.0, float(value)))
                        if pd.notna(value)
                        else float("nan")
                    )
                )
            predicted_complexes = [
                value
                for value in job.artifact_manifest.artifacts
                if value.artifact_type == "predicted_complex"
            ]
            predicted_complexes_by_candidate: dict[str, list[Any]] = {}
            for predicted_complex in predicted_complexes:
                candidate_key = str(predicted_complex.role or "").split(
                    ":", 1
                )[0]
                if candidate_key:
                    predicted_complexes_by_candidate.setdefault(
                        candidate_key, []
                    ).append(predicted_complex)
            structure_paths: list[str] = []
            structure_kinds: list[str] = []
            prediction_labels: list[str] = []
            for row_index, row in frame.iterrows():
                relative = next(
                    (
                        str(row.get(column) or "").strip()
                        for column in (
                            "structure_file",
                            "pose_file",
                            "rescored_pose_file",
                        )
                        if str(row.get(column) or "").strip()
                    ),
                    "",
                )
                structure_path = (
                    (job.run_dir / relative).resolve()
                    if relative and not Path(relative).is_absolute()
                    else Path(relative).resolve()
                    if relative
                    else None
                )
                structure_kind = (
                    "complex"
                    if engine
                    in {"AlphaFold 3", "Boltz-2", "RosettaLigand"}
                    else "pose"
                )
                if structure_path is None or not structure_path.is_file():
                    candidate_id = str(row.get("candidate_id") or "")
                    prediction_id = str(
                        row.get("prediction_id") or ""
                    )
                    matching = [
                        value
                        for value in predicted_complexes_by_candidate.get(
                            candidate_id, []
                        )
                        if (
                            not prediction_id
                            or prediction_id in str(value.role or "")
                        )
                    ]
                    artifact = matching[0] if matching else None
                    structure_path = (
                        artifact.resolve(job.run_dir, must_exist=True)
                        if artifact is not None
                        else None
                    )
                    if structure_path is not None:
                        structure_kind = "complex"
                structure_paths.append(
                    str(structure_path)
                    if structure_path is not None
                    and structure_path.is_file()
                    else ""
                )
                structure_kinds.append(structure_kind)
                attempt_parts = [
                    str(row.get(column))
                    for column in (
                        "replicate",
                        "seed",
                        "model_seed",
                        "prediction_id",
                        "model_id",
                    )
                    if str(row.get(column) or "").strip()
                    not in {"", "nan", "None"}
                ]
                prediction_labels.append(
                    f"{campaign_label} · "
                    + (
                        " · ".join(attempt_parts)
                        if attempt_parts
                        else f"result {row_index + 1}"
                    )
                )
            frame["_structure_path"] = structure_paths
            frame["_structure_kind"] = structure_kinds
            frame["_prediction_label"] = prediction_labels
            gnina_selection_columns = (
                "cnn_ranked_pose_index",
                "cnn_ranked_empirical_score_kcal_mol",
                "cnn_ranked_cnn_score",
                "cnn_ranked_cnn_affinity",
                "empirical_ranked_pose_index",
                "empirical_ranked_score_kcal_mol",
                "empirical_ranked_cnn_score",
                "empirical_ranked_cnn_affinity",
            )
            complete_gnina_selections = (
                engine == "GNINA"
                and all(column in frame for column in gnina_selection_columns)
                and frame[list(gnina_selection_columns)].notna().all().all()
            )
            if engine == "GNINA" and not complete_gnina_selections:
                from mn_ligand.workflows.docking import select_gnina_pose

                selections: list[dict[str, Any]] = []
                for (_, source_row), structure_path in zip(
                    frame.iterrows(),
                    structure_paths,
                ):
                    source = Path(structure_path) if structure_path else None
                    if source is not None and source.suffix.lower() == ".sdf":
                        source = source.with_suffix(".pdbqt")
                    fallback_pose = {
                        "pose_index": source_row.get("pose_index", 1),
                        "empirical_score_kcal_mol": source_row.get(
                            "best_score_kcal_mol"
                        ),
                        "cnn_score": source_row.get("cnn_score"),
                        "cnn_affinity": source_row.get("cnn_affinity"),
                    }
                    try:
                        parsed_cnn_pose = (
                            select_gnina_pose(source, "cnn_score")
                            if source is not None and source.is_file()
                            else None
                        ) or {}
                        cnn_pose = {
                            key: (
                                parsed_cnn_pose.get(key)
                                if parsed_cnn_pose.get(key) is not None
                                else value
                            )
                            for key, value in fallback_pose.items()
                        }
                        parsed_empirical_pose = (
                            select_gnina_pose(source, "empirical_score")
                            if source is not None and source.is_file()
                            else None
                        ) or {}
                        empirical_pose = {
                            key: (
                                parsed_empirical_pose.get(key)
                                if parsed_empirical_pose.get(key) is not None
                                else cnn_pose.get(key)
                            )
                            for key in fallback_pose
                        }
                    except (OSError, ValueError):
                        cnn_pose = fallback_pose
                        empirical_pose = cnn_pose
                    selections.append(
                        {
                            "cnn_ranked_pose_index": cnn_pose.get(
                                "pose_index", 1
                            ),
                            "cnn_ranked_empirical_score_kcal_mol": cnn_pose.get(
                                "empirical_score_kcal_mol"
                            ),
                            "cnn_ranked_cnn_score": cnn_pose.get("cnn_score"),
                            "cnn_ranked_cnn_affinity": cnn_pose.get(
                                "cnn_affinity"
                            ),
                            "empirical_ranked_pose_index": empirical_pose.get(
                                "pose_index", 1
                            ),
                            "empirical_ranked_score_kcal_mol": empirical_pose.get(
                                "empirical_score_kcal_mol"
                            ),
                            "empirical_ranked_cnn_score": empirical_pose.get(
                                "cnn_score"
                            ),
                            "empirical_ranked_cnn_affinity": empirical_pose.get(
                                "cnn_affinity"
                            ),
                        }
                    )
                selection_frame = pd.DataFrame(
                    selections,
                    index=frame.index,
                )
                for column in selection_frame.columns:
                    frame[column] = selection_frame[column]
            frame["_ligand_smiles"] = frame["candidate_id"].map(
                compound_smiles
            ).fillna(compound_default_smiles)
            frame["compound_name"] = frame["candidate_id"].map(
                compound_names
            ).fillna("")
            frame["compound_name_source_column"] = frame[
                "candidate_id"
            ].map(compound_name_sources).fillna("")
            for key, value in campaign.items():
                frame[key] = value
            metric_frames.append(frame)
            loaded = True
            break
        campaign["metrics_available"] = loaded
        campaigns.append(campaign)
    campaign_frame = pd.DataFrame(campaigns)
    campaign_frame = _assign_launch_campaigns(campaign_frame)
    metric_frame = (
        pd.concat(metric_frames, ignore_index=True, sort=False)
        if metric_frames
        else pd.DataFrame()
    )
    if not metric_frame.empty and not campaign_frame.empty:
        launch_columns = campaign_frame[
            [
                "campaign_id",
                "engine_run",
                "launch_campaign_id",
                "launch_campaign",
                "source_campaign_id",
            ]
        ]
        metric_frame = metric_frame.drop(
            columns=[
                column
                for column in launch_columns.columns
                if column != "campaign_id" and column in metric_frame
            ],
            errors="ignore",
        ).merge(launch_columns, on="campaign_id", how="left")
    return campaign_frame, metric_frame


def _pose_validation_rows(
    run_root: Path,
    selected_jobs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Load usable PoseBusters rows, with newer reruns superseding each pose."""
    source_lookup = {
        str(row["campaign_id"]): row
        for _, row in selected_jobs.iterrows()
    }
    frames: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    legacy_count = 0
    for job in iter_job_records(
        run_root, task_groups=("pose-validation",)
    ):
        source_id = str(job.metadata.get("parent_run_id") or "")
        if source_id not in source_lookup:
            continue
        source = source_lookup[source_id]
        report = _read_json(job.run_dir / "posebusters_report.json")
        usable = bool(
            report.get("applicable_checks")
            and int(job.metadata.get("selection_schema_version") or 0)
            == POSE_VALIDATION_INVENTORY_SCHEMA_VERSION
            and str(job.metadata.get("selection_policy") or "")
            == POSE_VALIDATION_SELECTION_POLICY
        )
        if job.status == "completed" and not usable:
            legacy_count += 1
        summary_path = job.run_dir / "posebusters_summary.csv"
        row_count = 0
        if job.status == "completed" and usable and summary_path.is_file():
            try:
                frame = pd.read_csv(summary_path)
            except (OSError, ValueError):
                frame = pd.DataFrame()
            if not frame.empty:
                row_count = len(frame)
                frame["source_run_id"] = source_id
                frame["validation_run_id"] = job.run_id
                frame["validation_created_at"] = str(job.created_at or "")
                frame["engine"] = str(source.get("engine") or "")
                frame["target_run_id"] = str(
                    source.get("target_run_id") or ""
                )
                frames.append(frame)
        provenance.append(
            {
                "source_job": display_job_code(
                    source_id, source.get("job_code")
                ),
                "engine": str(source.get("engine") or ""),
                "target": str(source.get("target") or ""),
                "validation_job": display_job_code(
                    job.run_id, job.metadata.get("job_code")
                ),
                "status": job.status,
                "validated_poses": row_count,
                "scientifically_usable": usable,
                "created_at": job.created_at,
                "result": (
                    "./job-results?"
                    + urlencode(
                        {
                            "task_group": "pose-validation",
                            "run_id": job.run_id,
                            "label": "PoseBusters",
                        }
                    )
                ),
            }
        )
    coverage = pd.DataFrame(provenance)
    if not frames:
        return pd.DataFrame(), coverage, legacy_count
    poses = pd.concat(frames, ignore_index=True, sort=False)
    poses["passed_all"] = (
        poses.get("passed_all", pd.Series(False, index=poses.index))
        .astype(str)
        .str.lower()
        .eq("true")
    )
    poses["compound_id"] = poses.get(
        "compound_id", pd.Series("", index=poses.index)
    ).fillna("").astype(str)
    fallback = (
        poses["compound_id"]
        + "|"
        + poses.get("replicate", pd.Series("", index=poses.index)).astype(str)
        + "|"
        + poses.get("prediction", pd.Series("", index=poses.index)).astype(str)
    )
    poses["_pose_key"] = poses.get(
        "pose_id", pd.Series("", index=poses.index)
    ).fillna("").astype(str)
    poses.loc[poses["_pose_key"].str.strip().eq(""), "_pose_key"] = fallback
    poses["_created"] = pd.to_datetime(
        poses["validation_created_at"], errors="coerce", utc=True
    )
    return (
        poses.sort_values("_created")
        .drop_duplicates(["source_run_id", "_pose_key"], keep="last")
        .drop(columns=["_created", "_pose_key"]),
        coverage,
        legacy_count,
    )


def _pose_validation_display_rows(poses: pd.DataFrame) -> pd.DataFrame:
    """Split GNINA's two scientific selections into distinct result groups."""
    rows: list[pd.Series] = []
    for _, source in poses.iterrows():
        engine = str(source.get("engine") or "")
        criterion = str(source.get("selection_criterion") or "").strip()
        if engine != "GNINA":
            row = source.copy()
            row["validation_group"] = engine
            rows.append(row)
            continue
        normalized = criterion.lower()
        groups: list[str] = []
        if "cnn" in normalized:
            groups.append("GNINA (CNN score)")
        if "vina" in normalized or "empirical" in normalized:
            groups.append("GNINA (Vina score)")
        if not groups:
            groups.append("GNINA (selection unspecified)")
        for group in groups:
            row = source.copy()
            row["validation_group"] = group
            rows.append(row)
    if not rows:
        return poses.assign(validation_group=pd.Series(dtype=str))
    return pd.DataFrame(rows).reset_index(drop=True)


@st.cache_data(ttl=30, show_spinner=False)
def _interaction_analysis_rows(
    run_root: Path,
    selected_jobs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the newest PLIP/PandaMap result for every selected source job."""
    source_lookup = {
        str(row["campaign_id"]): row for _, row in selected_jobs.iterrows()
    }
    summary_frames: list[pd.DataFrame] = []
    interaction_frames: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    for job in iter_job_records(run_root, task_groups=("interaction-analysis",)):
        source_id = str(job.metadata.get("parent_run_id") or "")
        if source_id not in source_lookup:
            continue
        source = source_lookup[source_id]
        analysis_engine = str(
            job.metadata.get("interaction_engine") or job.tool or ""
        )
        numbering_policy_version = int(
            job.metadata.get("residue_numbering_policy_version") or 0
        )
        current_numbering = (
            numbering_policy_version
            == INTERACTION_RESIDUE_NUMBERING_POLICY_VERSION
        )
        common = {
            "source_run_id": source_id,
            "analysis_run_id": job.run_id,
            "analysis_engine": analysis_engine,
            "analysis_created_at": str(job.created_at or ""),
            "source_engine": str(source.get("engine") or ""),
            "target_run_id": str(source.get("target_run_id") or ""),
            "target": str(source.get("target") or ""),
        }
        summary = pd.DataFrame()
        interactions = pd.DataFrame()
        summary_path = job.run_dir / "interaction_summary.csv"
        interactions_path = job.run_dir / "interactions.csv"
        if (
            current_numbering
            and job.status == "completed"
            and summary_path.is_file()
        ):
            try:
                summary = pd.read_csv(summary_path)
            except (OSError, ValueError):
                summary = pd.DataFrame()
        if (
            current_numbering
            and job.status == "completed"
            and interactions_path.is_file()
        ):
            try:
                interactions = pd.read_csv(interactions_path)
            except (OSError, ValueError):
                interactions = pd.DataFrame()
        if not summary.empty:
            for key, value in common.items():
                summary[key] = value
            summary_frames.append(summary)
        if not interactions.empty:
            for key, value in common.items():
                interactions[key] = value
            interaction_frames.append(interactions)
        provenance.append(
            {
                "source_job": display_job_code(
                    source_id, source.get("job_code")
                ),
                "source_engine": str(source.get("engine") or ""),
                "target": str(source.get("target") or ""),
                "analysis_engine": analysis_engine,
                "analysis_job": display_job_code(
                    job.run_id, job.metadata.get("job_code")
                ),
                "status": (
                    job.status
                    if current_numbering
                    else "Legacy numbering; rerun required"
                ),
                "residue_numbering_policy_version": numbering_policy_version,
                "analyzed_poses": len(summary),
                "interactions": len(interactions),
                "created_at": job.created_at,
                "result": (
                    "./job-results?"
                    + urlencode(
                        {
                            "task_group": "interaction-analysis",
                            "run_id": job.run_id,
                            "label": analysis_engine,
                        }
                    )
                ),
            }
        )
    provenance_frame = pd.DataFrame(provenance)
    if not summary_frames:
        return pd.DataFrame(), pd.DataFrame(), provenance_frame
    summaries = pd.concat(summary_frames, ignore_index=True, sort=False)
    interactions = (
        pd.concat(interaction_frames, ignore_index=True, sort=False)
        if interaction_frames
        else pd.DataFrame()
    )
    summaries["_created"] = pd.to_datetime(
        summaries["analysis_created_at"], errors="coerce", utc=True
    )
    newest_runs = (
        summaries.sort_values("_created")
        .drop_duplicates(["source_run_id", "analysis_engine"], keep="last")
        [["source_run_id", "analysis_engine", "analysis_run_id"]]
    )
    summaries = summaries.merge(
        newest_runs,
        on=["source_run_id", "analysis_engine", "analysis_run_id"],
        how="inner",
    ).drop(columns=["_created"])
    if not interactions.empty:
        interactions = interactions.merge(
            newest_runs,
            on=["source_run_id", "analysis_engine", "analysis_run_id"],
            how="inner",
        )
    return summaries, interactions, provenance_frame


def _campaign_interaction_review_atoms(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Fill the human-review atom columns exposed by native tool payloads."""
    table = interactions.copy().fillna("")
    for column, default in {
        "protein_atom_name": "",
        "protein_atom_scope": "",
        "ligand_atom_name": "",
        "native_fields_json": "{}",
    }.items():
        if column not in table:
            table[column] = default
    for index, row in table.iterrows():
        try:
            payload = json.loads(str(row.get("native_fields_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        interaction_type = str(
            row.get("interaction_type") or ""
        ).strip().lower().replace("_", " ")
        protein_atom = str(row.get("protein_atom_name") or "").strip()
        ligand_atom = str(row.get("ligand_atom_name") or "").strip()
        if not protein_atom:
            protein_atom = next(
                (
                    str(payload.get(key) or "").strip()
                    for key in (
                        "protein_atom", "protatom", "donor_atom",
                        "acceptor_atom",
                    )
                    if str(payload.get(key) or "").strip()
                ),
                "",
            )
        if not ligand_atom:
            ligand_atom = next(
                (
                    str(payload.get(key) or "").strip()
                    for key in (
                        "ligand_atom", "ligatom", "ligatom_orig_idx",
                        "lig_idx",
                    )
                    if str(payload.get(key) or "").strip()
                ),
                "",
            )
        protein_serial = ""
        ligand_serial = ""
        if "hydrophob" in interaction_type:
            protein_serial = str(payload.get("protcarbonidx") or "")
            ligand_serial = str(payload.get("ligcarbonidx") or "")
        elif "halogen" in interaction_type:
            protein_serial = str(payload.get("acc_idx") or "")
            ligand_serial = str(payload.get("don_idx") or "")
        elif "hydrogen" in interaction_type or "hbond" in interaction_type:
            protein_is_donor = str(
                payload.get("protisdon") or ""
            ).strip().lower() in {"true", "1", "yes"}
            protein_serial = str(payload.get(
                "donoridx" if protein_is_donor else "acceptoridx"
            ) or "")
            ligand_serial = str(payload.get(
                "acceptoridx" if protein_is_donor else "donoridx"
            ) or "")
        if not protein_serial:
            protein_serial = next(
                iter(re.findall(
                    r"\d+", str(payload.get("prot_idx_list") or "")
                )),
                "",
            )
        if not ligand_serial:
            ligand_serial = next(
                iter(re.findall(
                    r"\d+", str(payload.get("lig_idx_list") or "")
                )),
                "",
            )
        if not protein_atom and protein_serial:
            protein_atom = f"#{protein_serial}"
        if not ligand_atom and ligand_serial:
            ligand_atom = f"#{ligand_serial}"
        scope = str(row.get("protein_atom_scope") or "").strip().upper()
        if scope not in {"BB", "SC"}:
            native_sidechain = str(payload.get("sidechain") or "").lower()
            if native_sidechain in {"true", "1", "yes"}:
                scope = "SC"
            elif native_sidechain in {"false", "0", "no"}:
                scope = "BB"
            elif protein_atom and not protein_atom.startswith("#"):
                scope = (
                    "BB"
                    if protein_atom.upper() in {"N", "CA", "C", "O", "OXT"}
                    else "SC"
                )
        table.at[index, "protein_atom_name"] = protein_atom
        table.at[index, "ligand_atom_name"] = ligand_atom
        table.at[index, "protein_atom_scope"] = scope
    return table


def _campaign_interaction_gnina_selection(
    source_engine: object,
    selection_criterion: object,
) -> str:
    if str(source_engine or "").strip().lower() != "gnina":
        return ""
    criterion = str(selection_criterion or "").strip().lower()
    selected_by_cnn = "cnn" in criterion or "affinity" in criterion
    selected_by_vina = "vina" in criterion or "empirical" in criterion
    if selected_by_cnn and selected_by_vina:
        return "CNN + Vina"
    if selected_by_cnn:
        return "CNN"
    if selected_by_vina:
        return "Vina"
    return "Unspecified"


def _campaign_interaction_complex_paths(
    run_root: Path,
    interactions: pd.DataFrame,
) -> pd.Series:
    input_maps: dict[str, dict[str, Path]] = {}
    output: list[str] = []
    for _, row in interactions.iterrows():
        analysis_run_id = str(row.get("analysis_run_id") or "").strip()
        pose_id = str(row.get("pose_id") or "").strip()
        job_dir = run_root / "interaction-analysis" / analysis_run_id
        prepared = job_dir / "prepared" / f"{pose_id}.complex.pdb"
        selected: Path | None = prepared if prepared.is_file() else None
        if selected is None and analysis_run_id not in input_maps:
            input_maps[analysis_run_id] = {}
            inputs_path = job_dir / "input" / "interaction_inputs.csv"
            if inputs_path.is_file():
                try:
                    input_rows = pd.read_csv(inputs_path).fillna("")
                    input_maps[analysis_run_id] = {
                        str(item.get("pose_id") or "").strip(): (
                            job_dir / str(item.get("complex_file") or "").strip()
                        )
                        for item in input_rows.to_dict("records")
                        if str(item.get("pose_id") or "").strip()
                        and str(item.get("complex_file") or "").strip()
                    }
                except (OSError, ValueError):
                    pass
        if selected is None:
            candidate = input_maps.get(analysis_run_id, {}).get(pose_id)
            if candidate is not None and candidate.is_file():
                selected = candidate
        output.append(str(selected.resolve()) if selected is not None else "")
    return pd.Series(output, index=interactions.index, dtype=str)


def _campaign_interaction_source_paths(
    run_root: Path,
    interactions: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return immutable prediction-pose and receptor paths for each row."""
    input_maps: dict[str, dict[str, dict[str, object]]] = {}
    pose_paths: list[str] = []
    receptor_paths: list[str] = []
    for _, row in interactions.iterrows():
        analysis_run_id = str(row.get("analysis_run_id") or "").strip()
        source_run_id = str(row.get("source_run_id") or "").strip()
        pose_id = str(row.get("pose_id") or "").strip()
        analysis_dir = run_root / "interaction-analysis" / analysis_run_id
        if analysis_run_id not in input_maps:
            input_maps[analysis_run_id] = {}
            inputs_path = analysis_dir / "input" / "interaction_inputs.csv"
            if inputs_path.is_file():
                try:
                    input_rows = pd.read_csv(inputs_path).fillna("")
                    input_maps[analysis_run_id] = {
                        str(item.get("pose_id") or "").strip(): item
                        for item in input_rows.to_dict("records")
                        if str(item.get("pose_id") or "").strip()
                    }
                except (OSError, ValueError):
                    pass
        input_row = input_maps.get(analysis_run_id, {}).get(pose_id, {})
        source_dir = _source_run_dir(run_root, source_run_id)
        source_pose: Path | None = None
        source_receptor: Path | None = None
        relative = str(input_row.get("source_artifact_path") or "").strip()
        if source_dir is not None and relative:
            candidate = (source_dir / relative).resolve()
            try:
                candidate.relative_to(source_dir)
            except ValueError:
                candidate = Path()
            if candidate.is_file():
                source_pose = candidate
        if source_dir is not None:
            source_receptor = next(
                (
                    candidate.resolve()
                    for candidate in (
                        source_dir / "input" / "receptor.pdb",
                        source_dir / "prepared" / "receptor.pdb",
                    )
                    if candidate.is_file()
                ),
                None,
            )
        pose_paths.append(str(source_pose) if source_pose is not None else "")
        receptor_paths.append(
            str(source_receptor) if source_receptor is not None else ""
        )
    return (
        pd.Series(pose_paths, index=interactions.index, dtype=str),
        pd.Series(receptor_paths, index=interactions.index, dtype=str),
    )


def _safe_interaction_sheet_name(
    value: object,
    used_names: set[str],
) -> str:
    base = re.sub(r"[\[\]:*?/\\]+", "-", str(value or "Interactions"))
    base = base.strip()[:31] or "Interactions"
    candidate = base
    suffix = 2
    while candidate.lower() in used_names:
        ending = f"-{suffix}"
        candidate = f"{base[:31 - len(ending)]}{ending}"
        suffix += 1
    used_names.add(candidate.lower())
    return candidate


def _campaign_interaction_review_workbook(
    run_root: Path,
    selected_jobs: pd.DataFrame,
    summaries: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tuple[bytes, int]:
    """Create the campaign-scoped equivalent of interaction-review-by-tool."""
    if interactions.empty:
        return b"", 0
    source_context = {
        str(row.get("campaign_id") or ""): row
        for row in selected_jobs.to_dict("records")
    }
    table = _campaign_interaction_review_atoms(interactions.fillna(""))
    for column in (
        "pose_id", "compound_id", "source_engine", "replicate",
        "prediction", "selection_criterion", "interaction_type",
        "protein_chain", "protein_residue_name", "protein_residue_number",
        "protein_insertion_code", "ligand_atom_name", "protein_atom_name",
        "protein_atom_scope", "distance_angstrom", "angle_degree",
    ):
        if column not in table:
            table[column] = ""
    analysis_metadata: dict[str, dict[str, object]] = {}
    for analysis_run_id in table["analysis_run_id"].astype(str).unique():
        metadata_path = (
            run_root / "interaction-analysis" / analysis_run_id / "metadata.json"
        )
        try:
            payload = json.loads(metadata_path.read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        analysis_metadata[analysis_run_id] = (
            payload if isinstance(payload, dict) else {}
        )

    def source_value(source_id: object, key: str, default: str = "") -> str:
        row = source_context.get(str(source_id or ""), {})
        return str(row.get(key) or default)

    campaign_values = [
        source_value(source_id, "launch_campaign_label", "Campaign")
        for source_id in table["source_run_id"]
    ]
    source_jobs = [
        display_job_code(
            source_context.get(str(source_id or ""), {}).get("job_code"),
            str(source_id or ""),
        )
        for source_id in table["source_run_id"]
    ]
    analysis_jobs = [
        display_job_code(
            analysis_metadata.get(str(analysis_run_id), {}).get("job_code"),
            str(analysis_run_id),
        )
        for analysis_run_id in table["analysis_run_id"]
    ]
    insertion_codes = table["protein_insertion_code"].astype(str)
    protein_residue = (
        table["protein_chain"].astype(str)
        + ":"
        + table["protein_residue_name"].astype(str)
        + table["protein_residue_number"].astype(str)
        + insertion_codes
    )
    source_pose_paths, source_receptor_paths = (
        _campaign_interaction_source_paths(run_root, table)
    )
    interaction_evidence_paths = _campaign_interaction_complex_paths(
        run_root, table
    )
    review = pd.DataFrame({
        "Campaign": campaign_values,
        "Source job": source_jobs,
        "Prediction engine": [
            source_value(source_id, "engine", str(source_engine or ""))
            for source_id, source_engine in zip(
                table["source_run_id"], table["source_engine"], strict=True
            )
        ],
        "Target": [
            source_value(source_id, "target")
            for source_id in table["source_run_id"]
        ],
        "Interaction detection engine": table["analysis_engine"].astype(str),
        "Analysis job": analysis_jobs,
        "Pose": (
            table["compound_id"].astype(str)
            + " · "
            + table["source_engine"].astype(str)
            + " · attempt "
            + table["replicate"].astype(str)
        ),
        "Compound": table["compound_id"].astype(str),
        "Replicate": table["replicate"],
        "Prediction": table["prediction"].astype(str),
        "Source pose path": source_pose_paths,
        "Source receptor path": source_receptor_paths,
        "Interaction evidence path": interaction_evidence_paths,
        "GNINA pose selection": [
            _campaign_interaction_gnina_selection(engine, criterion)
            for engine, criterion in zip(
                table["source_engine"], table["selection_criterion"], strict=True
            )
        ],
        "Interaction": table["interaction_type"].astype(str),
        "Protein residue": protein_residue,
        "Ligand atom": table["ligand_atom_name"].astype(str),
        "Protein atom": table["protein_atom_name"].astype(str),
        "Protein region": table["protein_atom_scope"].astype(str),
        "Distance (Å)": pd.to_numeric(
            table["distance_angstrom"], errors="coerce"
        ),
        "Angle (°)": pd.to_numeric(table["angle_degree"], errors="coerce"),
    })
    summary_counts = (
        summaries.groupby("analysis_run_id", as_index=False)
        .agg(
            compounds=("compound_id", "nunique"),
            poses=("pose_id", "nunique"),
        )
        if not summaries.empty
        else pd.DataFrame(columns=["analysis_run_id", "compounds", "poses"])
    )
    interaction_counts = (
        table.groupby("analysis_run_id", as_index=False)
        .size()
        .rename(columns={"size": "interactions"})
    )
    selection = summary_counts.merge(
        interaction_counts, on="analysis_run_id", how="outer"
    ).fillna(0)
    selection_rows: list[dict[str, object]] = []
    for row in selection.to_dict("records"):
        analysis_run_id = str(row.get("analysis_run_id") or "")
        first = table.loc[
            table["analysis_run_id"].astype(str).eq(analysis_run_id)
        ].iloc[0]
        source_id = str(first.get("source_run_id") or "")
        metadata = analysis_metadata.get(analysis_run_id, {})
        selection_rows.append({
            "Campaign": source_value(
                source_id, "launch_campaign_label", "Campaign"
            ),
            "Source job": display_job_code(
                source_context.get(source_id, {}).get("job_code"), source_id
            ),
            "Prediction engine": source_value(
                source_id, "engine", str(first.get("source_engine") or "")
            ),
            "Target": source_value(source_id, "target"),
            "Interaction tool": str(first.get("analysis_engine") or ""),
            "Analysis job": display_job_code(
                metadata.get("job_code"), analysis_run_id
            ),
            "Compounds": int(row.get("compounds") or 0),
            "Poses": int(row.get("poses") or 0),
            "Interactions": int(row.get("interactions") or 0),
            "Status": str(metadata.get("status") or "completed"),
            "Created": str(metadata.get("created_at") or ""),
        })
    selection_frame = pd.DataFrame(selection_rows)
    per_tool = {
        tool: rows.reset_index(drop=True)
        for tool, rows in review.groupby(
            "Interaction detection engine", sort=True
        )
    }
    output = BytesIO()
    used_names: set[str] = set()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        selection_frame.to_excel(
            writer,
            sheet_name=_safe_interaction_sheet_name("Selection", used_names),
            index=False,
        )
        review.to_excel(
            writer,
            sheet_name=_safe_interaction_sheet_name(
                "All interactions", used_names
            ),
            index=False,
        )
        for tool, rows in per_tool.items():
            rows.to_excel(
                writer,
                sheet_name=_safe_interaction_sheet_name(tool, used_names),
                index=False,
            )
    return output.getvalue(), len(review)


def _render_interaction_analysis_summary(
    run_root: Path,
    selected_jobs: pd.DataFrame,
    selected_metrics: pd.DataFrame,
) -> None:
    structural = selected_jobs.loc[
        selected_jobs["engine"].isin(STRUCTURE_ENGINE_ORDER)
    ].copy()
    if structural.empty:
        st.info("No selected structure-producing campaigns can be analyzed.")
        return
    targets = structural[["target_run_id", "target"]].drop_duplicates()
    target_ids = targets["target_run_id"].astype(str).tolist()
    target_labels = dict(
        zip(targets["target_run_id"].astype(str), targets["target"].astype(str))
    )
    target_id = st.selectbox(
        "Interaction-analysis target",
        target_ids,
        format_func=lambda value: target_labels.get(value, value),
        key="campaign_interaction_analysis_target",
        help="Interaction fingerprints are summarized in one protein context at a time.",
    )
    target_jobs = structural.loc[
        structural["target_run_id"].astype(str).eq(str(target_id))
    ]
    summaries, interactions, provenance = _interaction_analysis_rows(
        run_root, target_jobs
    )
    if not provenance.empty and "status" in provenance:
        legacy_count = int(
            provenance["status"]
            .astype(str)
            .eq("Legacy numbering; rerun required")
            .sum()
        )
        if legacy_count:
            st.warning(
                f"{legacy_count} legacy interaction-analysis job(s) use "
                "processed residue indices and are excluded. Re-run them in "
                "Interaction Analysis to report immutable imported-source "
                "author residue numbering."
            )
    compounds = sorted(
        selected_metrics.loc[
            selected_metrics["target_run_id"].astype(str).eq(str(target_id)),
            "candidate_id",
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    if summaries.empty:
        st.info(
            "No completed PLIP or PandaMap results are linked to this target "
            "and campaign selection. Queue the missing jobs in Interaction Analysis."
        )
        if not provenance.empty:
            with st.expander("Linked interaction-analysis jobs"):
                st.dataframe(
                    provenance,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "result": st.column_config.LinkColumn(
                            "Result", display_text="Open"
                        )
                    },
                )
        return
    summaries["compound_id"] = summaries.get(
        "compound_id", pd.Series("", index=summaries.index)
    ).fillna("").astype(str)
    summaries = summaries.loc[summaries["compound_id"].isin(compounds)].copy()
    if summaries.empty:
        st.info("The linked analyses do not contain the selected compounds.")
        return
    summaries["success"] = (
        summaries.get("success", pd.Series(False, index=summaries.index))
        .astype(str)
        .str.lower()
        .eq("true")
    )
    for column in ("interaction_count", "contacted_residue_count"):
        summaries[column] = pd.to_numeric(
            summaries.get(column, 0), errors="coerce"
        ).fillna(0)
    if not interactions.empty:
        interactions["compound_id"] = interactions["compound_id"].astype(str)
        interactions = interactions.loc[
            interactions["compound_id"].isin(compounds)
        ].copy()
        interactions["interaction_type"] = (
            interactions["interaction_type"]
            .fillna("unspecified")
            .astype(str)
            .str.replace("_", " ", regex=False)
            .str.strip()
            .str.lower()
        )
        interactions["residue"] = (
            interactions["protein_chain"].fillna("").astype(str)
            + ":"
            + interactions["protein_residue_name"].fillna("").astype(str)
            + interactions["protein_residue_number"].fillna("").astype(str)
        )
    source_order = [
        engine
        for engine in STRUCTURE_ENGINE_ORDER
        if engine in set(summaries["source_engine"].astype(str))
    ]
    analysis_order = [
        engine
        for engine in ("PLIP", "PandaMap")
        if engine in set(summaries["analysis_engine"].astype(str))
    ]
    summaries["analysis_group"] = (
        summaries["source_engine"].astype(str)
        + " · "
        + summaries["analysis_engine"].astype(str)
    )
    group_order = [
        f"{source_engine} · {analysis_engine}"
        for source_engine in source_order
        for analysis_engine in analysis_order
        if (
            summaries["source_engine"].eq(source_engine)
            & summaries["analysis_engine"].eq(analysis_engine)
        ).any()
    ]
    metric_columns = st.columns(5)
    metric_columns[0].metric("Selected source jobs", len(target_jobs))
    metric_columns[1].metric(
        "Analyzed source jobs", summaries["source_run_id"].nunique()
    )
    metric_columns[2].metric(
        "Compounds analyzed", summaries["compound_id"].nunique()
    )
    metric_columns[3].metric(
        "Successful poses", int(summaries["success"].sum())
    )
    metric_columns[4].metric(
        "Detected interactions", int(summaries["interaction_count"].sum())
    )
    st.caption(
        "PLIP and PandaMap remain separate scientific measurements. The same "
        "focused pose policy is used for both, so their fingerprints can be "
        "compared without pooling different pose inventories."
    )
    assessed = (
        summaries.groupby(["compound_id", "analysis_group"], as_index=False)
        .agg(
            analyzed_poses=("success", "size"),
            successful_poses=("success", "sum"),
            mean_interactions=("interaction_count", "mean"),
            mean_contacted_residues=("contacted_residue_count", "mean"),
        )
    )
    grid = pd.MultiIndex.from_product(
        [compounds, group_order],
        names=["compound_id", "analysis_group"],
    ).to_frame(index=False)
    grid = grid.merge(
        assessed, on=["compound_id", "analysis_group"], how="left"
    )
    grid["coverage"] = "Not tested"
    grid.loc[grid["analyzed_poses"].fillna(0).gt(0), "coverage"] = "Analyzed"
    grid.loc[
        grid["analyzed_poses"].fillna(0).gt(0)
        & grid["successful_poses"].fillna(0).eq(0),
        "coverage",
    ] = "Failed"
    st.markdown("#### Compound interaction-analysis coverage")
    coverage_rect = (
        alt.Chart(grid)
        .mark_rect(stroke="white")
        .encode(
            x=alt.X(
                "analysis_group:N",
                sort=group_order,
                axis=alt.Axis(labelAngle=-25),
                title="Source engine · interaction engine",
            ),
            y=alt.Y("compound_id:N", sort=compounds, title="Compound"),
            color=alt.Color(
                "coverage:N",
                scale=alt.Scale(
                    domain=["Analyzed", "Failed", "Not tested"],
                    range=["#16a34a", "#dc2626", "#d1d5db"],
                ),
                title="Coverage",
            ),
            tooltip=[
                "compound_id:N",
                "analysis_group:N",
                "coverage:N",
                "analyzed_poses:Q",
                "successful_poses:Q",
                alt.Tooltip("mean_interactions:Q", format=".1f"),
                alt.Tooltip("mean_contacted_residues:Q", format=".1f"),
            ],
        )
        .properties(height=max(280, 31 * len(compounds)))
    )
    coverage_text = (
        alt.Chart(grid)
        .mark_text(fontWeight="bold")
        .encode(
            x=alt.X("analysis_group:N", sort=group_order),
            y=alt.Y("compound_id:N", sort=compounds),
            text=alt.Text("mean_interactions:Q", format=".1f"),
            color=alt.condition(
                alt.datum.coverage == "Analyzed",
                alt.value("white"),
                alt.value("#374151"),
            ),
        )
    )
    st.altair_chart(coverage_rect + coverage_text, width="stretch")
    st.caption(
        "Numbers are mean detected interactions per analyzed pose; gray cells "
        "have not been run for that source/analysis-engine combination."
    )
    if not interactions.empty:
        st.markdown("#### Interaction fingerprints")
        fingerprint = (
            interactions.groupby(
                ["source_engine", "analysis_engine", "interaction_type"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "detections"})
        )
        fingerprint["engine_pair"] = (
            fingerprint["source_engine"] + " · " + fingerprint["analysis_engine"]
        )
        st.altair_chart(
            alt.Chart(fingerprint)
            .mark_bar()
            .encode(
                x=alt.X("detections:Q", title="Detected interactions"),
                y=alt.Y("engine_pair:N", sort=group_order, title=None),
                color=alt.Color(
                    "interaction_type:N", title="Interaction type"
                ),
                tooltip=[
                    "source_engine:N",
                    "analysis_engine:N",
                    "interaction_type:N",
                    "detections:Q",
                ],
            )
            .properties(height=max(240, 38 * len(group_order))),
            width="stretch",
        )
        residue_summary = (
            interactions.groupby(["analysis_engine", "residue"], as_index=False)
            .agg(
                detections=("pose_id", "size"),
                compounds=("compound_id", "nunique"),
                source_engines=("source_engine", "nunique"),
            )
            .sort_values(
                ["analysis_engine", "compounds", "detections"],
                ascending=[True, False, False],
            )
        )
        st.markdown("#### Most recurrent contacted residues")
        st.dataframe(
            residue_summary.groupby("analysis_engine", group_keys=False)
            .head(15)
            .reset_index(drop=True),
            hide_index=True,
            width="stretch",
            column_config={
                "analysis_engine": "Analysis engine",
                "residue": "Protein residue",
                "detections": "Detections",
                "compounds": "Compounds",
                "source_engines": "Source engines",
            },
        )
        residue_sets = (
            interactions.groupby(
                ["source_run_id", "pose_id", "compound_id", "analysis_engine"]
            )["residue"]
            .agg(lambda values: frozenset(values))
            .reset_index()
        )
        paired = residue_sets.pivot_table(
            index=["source_run_id", "pose_id", "compound_id"],
            columns="analysis_engine",
            values="residue",
            aggfunc="first",
        ).reset_index()
        if {"PLIP", "PandaMap"}.issubset(paired.columns):
            paired = paired.dropna(subset=["PLIP", "PandaMap"]).copy()
            if not paired.empty:
                paired["contact_residue_jaccard"] = paired.apply(
                    lambda row: (
                        len(row["PLIP"] & row["PandaMap"])
                        / len(row["PLIP"] | row["PandaMap"])
                        if row["PLIP"] | row["PandaMap"]
                        else 1.0
                    ),
                    axis=1,
                )
                st.markdown("#### PLIP–PandaMap residue agreement")
                st.altair_chart(
                    alt.Chart(paired)
                    .mark_boxplot(color="#7c3aed")
                    .encode(
                        x=alt.X(
                            "compound_id:N",
                            sort=compounds,
                            title="Compound",
                            axis=alt.Axis(labelAngle=-45),
                        ),
                        y=alt.Y(
                            "contact_residue_jaccard:Q",
                            scale=alt.Scale(domain=[0, 1]),
                            title="Contact-residue Jaccard similarity",
                        ),
                        tooltip=[
                            "compound_id:N",
                            alt.Tooltip(
                                "contact_residue_jaccard:Q", format=".3f"
                            ),
                        ],
                    ),
                    width="stretch",
                )
                st.caption(
                    "Jaccard similarity is the overlap divided by the union of "
                    "protein residues contacted by the same pose. It compares "
                    "detectors without treating either engine as ground truth."
                )
    export_signature = hashlib.sha256(
        "\n".join([
            *sorted(structural["campaign_id"].astype(str).unique()),
            *sorted(
                (
                    selected_metrics["target_run_id"].astype(str)
                    + "::"
                    + selected_metrics["candidate_id"].astype(str)
                ).unique()
            ),
        ]).encode("utf-8")
    ).hexdigest()
    export_state_key = "campaign_interaction_review_workbooks"
    with st.expander("Download interaction review workbooks", expanded=True):
        st.caption(
            "Creates one separate Excel workbook for each prepared target in "
            "the campaign scope. Every file contains a Selection sheet, one "
            "row per detected interaction in All interactions, and separate "
            "Native MD geometry, PandaMap, and PLIP sheets when available."
        )
        if st.button(
            "Prepare one workbook per target",
            type="primary",
            key="prepare_campaign_interaction_review_workbooks",
        ):
            try:
                with st.spinner(
                    "Collecting campaign interactions by prepared target..."
                ):
                    all_summaries, all_interactions, _ = (
                        _interaction_analysis_rows(run_root, structural)
                    )
                    workbooks: list[dict[str, object]] = []
                    used_filenames: set[str] = set()
                    for export_target_id, export_target_label in zip(
                        targets["target_run_id"].astype(str),
                        targets["target"].astype(str),
                        strict=True,
                    ):
                        export_jobs = structural.loc[
                            structural["target_run_id"].astype(str).eq(
                                export_target_id
                            )
                        ].copy()
                        source_ids = set(
                            export_jobs["campaign_id"].astype(str)
                        )
                        target_summaries = all_summaries.loc[
                            all_summaries["source_run_id"]
                            .astype(str)
                            .isin(source_ids)
                        ].copy()
                        target_interactions = all_interactions.loc[
                            all_interactions["source_run_id"]
                            .astype(str)
                            .isin(source_ids)
                        ].copy()
                        selected_compounds = set(
                            selected_metrics.loc[
                                selected_metrics["target_run_id"]
                                .astype(str)
                                .eq(export_target_id),
                                "candidate_id",
                            ].dropna().astype(str)
                        )
                        target_summaries = target_summaries.loc[
                            target_summaries["compound_id"]
                            .astype(str)
                            .isin(selected_compounds)
                        ].copy()
                        target_interactions = target_interactions.loc[
                            target_interactions["compound_id"]
                            .astype(str)
                            .isin(selected_compounds)
                        ].copy()
                        workbook, row_count = (
                            _campaign_interaction_review_workbook(
                                run_root,
                                export_jobs,
                                target_summaries,
                                target_interactions,
                            )
                        )
                        if not workbook:
                            continue
                        stem = re.sub(
                            r"[^A-Za-z0-9._-]+",
                            "-",
                            export_target_label,
                        ).strip("-._").lower()[:80] or "target"
                        filename = (
                            f"{stem}-interaction-review-by-tool.xlsx"
                        )
                        suffix = 2
                        while filename in used_filenames:
                            filename = (
                                f"{stem}-{suffix}-interaction-review-by-tool.xlsx"
                            )
                            suffix += 1
                        used_filenames.add(filename)
                        workbooks.append({
                            "target": export_target_label,
                            "filename": filename,
                            "workbook": workbook,
                            "row_count": row_count,
                        })
            except (ImportError, OSError, TypeError, ValueError) as exc:
                st.error(f"Could not prepare interaction workbooks: {exc}")
            else:
                st.session_state[export_state_key] = {
                    "signature": export_signature,
                    "workbooks": workbooks,
                }
        export_payload = st.session_state.get(export_state_key)
        if (
            isinstance(export_payload, dict)
            and export_payload.get("signature") == export_signature
            and export_payload.get("workbooks")
        ):
            workbooks = export_payload["workbooks"]
            download_columns = st.columns(min(3, len(workbooks)))
            for index, item in enumerate(workbooks):
                row_count = int(item.get("row_count") or 0)
                target_label = str(item.get("target") or "Target")
                download_columns[index % len(download_columns)].download_button(
                    f"Download {target_label} ({row_count:,} rows)",
                    data=item["workbook"],
                    file_name=str(item["filename"]),
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key=f"download_campaign_interaction_review_{index}",
                )
    with st.expander("Linked interaction-analysis jobs"):
        st.dataframe(
            provenance,
            hide_index=True,
            width="stretch",
            column_config={
                "result": st.column_config.LinkColumn(
                    "Result", display_text="Open"
                )
            },
        )


def _render_pose_validation_summary(
    run_root: Path,
    selected_jobs: pd.DataFrame,
    selected_metrics: pd.DataFrame,
) -> None:
    structural = selected_jobs.loc[
        selected_jobs["engine"].isin(STRUCTURE_ENGINE_ORDER)
    ].copy()
    if structural.empty:
        st.info("No selected structure-producing campaigns can be validated.")
        return
    targets = structural[["target_run_id", "target"]].drop_duplicates()
    target_ids = targets["target_run_id"].astype(str).tolist()
    target_labels = dict(
        zip(
            targets["target_run_id"].astype(str),
            targets["target"].astype(str),
        )
    )
    target_id = st.selectbox(
        "Pose-validation target",
        target_ids,
        format_func=lambda value: target_labels.get(value, value),
        key="campaign_pose_validation_target",
        help="Pose validity is summarized in one protein target context at a time.",
    )
    target_jobs = structural.loc[
        structural["target_run_id"].astype(str).eq(str(target_id))
    ]
    poses, provenance, legacy_count = _pose_validation_rows(
        run_root, target_jobs
    )
    engines = [
        engine
        for engine in STRUCTURE_ENGINE_ORDER
        if engine in set(target_jobs["engine"].astype(str))
    ]
    compounds = sorted(
        selected_metrics.loc[
            selected_metrics["target_run_id"].astype(str).eq(str(target_id)),
            "candidate_id",
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    if legacy_count:
        st.warning(
            f"{legacy_count} legacy completed run(s) lack the current focused "
            "pose-selection policy or applicable-check manifest and are "
            "excluded. Re-run them with the current adapter."
        )
    if poses.empty:
        st.info(
            "No scientifically usable PoseBusters rows are linked to this "
            "target and selection. Run the missing jobs in Pose Validation."
        )
        if not provenance.empty:
            with st.expander("Linked validation jobs"):
                st.dataframe(
                    provenance,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "result": st.column_config.LinkColumn(
                            "Result", display_text="Open"
                        )
                    },
                )
        return
    poses = poses.loc[
        poses["compound_id"].isin(compounds)
        & poses["engine"].isin(engines)
    ].copy()
    poses = _pose_validation_display_rows(poses)
    validation_groups = []
    for engine in engines:
        if engine == "GNINA":
            validation_groups.extend(
                ["GNINA (CNN score)", "GNINA (Vina score)"]
            )
        else:
            validation_groups.append(engine)
    assessed = (
        poses.groupby(["compound_id", "validation_group"], as_index=False)
        .agg(
            assessed_poses=("passed_all", "size"),
            passing_poses=("passed_all", "sum"),
        )
    )
    assessed["status"] = assessed["passing_poses"].gt(0).map(
        {True: "PASS", False: "FAIL"}
    )
    grid = pd.MultiIndex.from_product(
        [compounds, validation_groups],
        names=["compound_id", "validation_group"],
    ).to_frame(index=False)
    grid = grid.merge(
        assessed,
        on=["compound_id", "validation_group"],
        how="left",
    )
    grid["status"] = grid["status"].fillna("Not tested")
    for column in ("assessed_poses", "passing_poses"):
        grid[column] = grid[column].fillna(0).astype(int)
    summary = (
        grid.groupby("compound_id", as_index=False)
        .agg(
            validation_groups_passed=(
                "status", lambda x: int((x == "PASS").sum())
            ),
            validation_groups_failed=(
                "status", lambda x: int((x == "FAIL").sum())
            ),
            validation_groups_not_tested=(
                "status", lambda x: int((x == "Not tested").sum())
            ),
            assessed_poses=("assessed_poses", "sum"),
            passing_poses=("passing_poses", "sum"),
        )
    )
    summary["tested_validation_groups"] = (
        summary["validation_groups_passed"]
        + summary["validation_groups_failed"]
    )
    summary["pose_pass_rate"] = (
        summary["passing_poses"]
        / summary["assessed_poses"].replace(0, pd.NA)
    )
    summary["made_it"] = summary["validation_groups_passed"].gt(0)
    summary["passed_every_tested_engine"] = (
        summary["tested_validation_groups"].gt(0)
        & summary["validation_groups_failed"].eq(0)
    )
    summary["complete_engine_coverage"] = summary[
        "validation_groups_not_tested"
    ].eq(0)
    summary = summary.sort_values(
        [
            "made_it",
            "validation_groups_passed",
            "validation_groups_failed",
            "compound_id",
        ],
        ascending=[False, False, True, True],
    )
    order = summary["compound_id"].tolist()
    metrics = st.columns(5)
    metrics[0].metric("Selected source jobs", len(target_jobs))
    metrics[1].metric(
        "Validated source jobs", poses["source_run_id"].nunique()
    )
    metrics[2].metric("Compounds assessed", poses["compound_id"].nunique())
    metrics[3].metric(
        "Compounds with ≥1 PASS", int(summary["made_it"].sum())
    )
    metrics[4].metric(
        "Passing poses", f"{int(poses['passed_all'].sum())} / {len(poses)}"
    )
    st.caption(
        "A compound “made it” when at least one stored pose passed every "
        "applicable PoseBusters check. Gray cells are untested and never count "
        "as passes; complete coverage is reported separately. GNINA's CNN and "
        "Vina-score selections are separate validation groups. If both select "
        "the same physical pose, its result is represented in both groups."
    )
    st.markdown("#### Compound physical-validity map")
    rect = (
        alt.Chart(grid)
        .mark_rect(stroke="white")
        .encode(
            x=alt.X(
                "validation_group:N",
                sort=validation_groups,
                axis=alt.Axis(labelAngle=-25),
                title="Engine / pose selection",
            ),
            y=alt.Y("compound_id:N", sort=order, title="Compound"),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=["PASS", "FAIL", "Not tested"],
                    range=["#16a34a", "#dc2626", "#d1d5db"],
                ),
                title="PoseBusters",
            ),
            tooltip=[
                "compound_id:N",
                "validation_group:N",
                "status:N",
                "passing_poses:Q",
                "assessed_poses:Q",
            ],
        )
        .properties(height=max(280, 31 * len(order)))
    )
    text = (
        alt.Chart(grid)
        .mark_text(fontWeight="bold")
        .encode(
            x=alt.X(
                "validation_group:N",
                sort=validation_groups,
            ),
            y=alt.Y("compound_id:N", sort=order),
            text="status:N",
            color=alt.condition(
                alt.datum.status == "Not tested",
                alt.value("#374151"),
                alt.value("white"),
            ),
        )
    )
    st.altair_chart(rect + text, width="stretch")

    st.markdown("#### Per-engine compound pass rate")
    engine_summary = (
        assessed.groupby("validation_group", as_index=False)
        .agg(
            compounds_assessed=("compound_id", "nunique"),
            compounds_passed=(
                "status", lambda x: int((x == "PASS").sum())
            ),
            assessed_poses=("assessed_poses", "sum"),
            passing_poses=("passing_poses", "sum"),
        )
    )
    engine_summary["compound_pass_rate"] = (
        engine_summary["compounds_passed"]
        / engine_summary["compounds_assessed"]
    )
    engine_summary["compounds_not_tested"] = (
        len(compounds) - engine_summary["compounds_assessed"]
    )
    engine_summary["failed_poses"] = (
        engine_summary["assessed_poses"] - engine_summary["passing_poses"]
    )
    engine_summary["pose_pass_rate"] = (
        engine_summary["passing_poses"]
        / engine_summary["assessed_poses"]
    )
    engine_rates = engine_summary.melt(
        id_vars=[
            "validation_group",
            "compounds_assessed",
            "compounds_passed",
            "compounds_not_tested",
            "assessed_poses",
            "passing_poses",
            "failed_poses",
        ],
        value_vars=["compound_pass_rate", "pose_pass_rate"],
        var_name="rate_kind",
        value_name="pass_rate",
    )
    engine_rates["rate_kind"] = engine_rates["rate_kind"].map(
        {
            "compound_pass_rate": "Compounds with ≥1 PASS",
            "pose_pass_rate": "All poses passing",
        }
    )
    st.altair_chart(
        alt.Chart(engine_rates)
        .mark_bar()
        .encode(
            x=alt.X(
                "pass_rate:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
                title="Pass rate",
            ),
            y=alt.Y(
                "validation_group:N",
                sort=validation_groups,
                title=None,
            ),
            yOffset="rate_kind:N",
            color=alt.Color(
                "rate_kind:N",
                title="Statistic",
                scale=alt.Scale(
                    domain=[
                        "Compounds with ≥1 PASS",
                        "All poses passing",
                    ],
                    range=["#16a34a", "#2563eb"],
                ),
            ),
            tooltip=[
                "validation_group:N",
                "rate_kind:N",
                "compounds_passed:Q",
                "compounds_assessed:Q",
                "compounds_not_tested:Q",
                alt.Tooltip("pass_rate:Q", format=".1%"),
                "passing_poses:Q",
                "failed_poses:Q",
                "assessed_poses:Q",
            ],
        )
        .properties(height=max(200, 58 * len(engine_summary))),
        width="stretch",
    )
    with st.expander("Engine statistics", expanded=True):
        st.dataframe(
            engine_summary.sort_values(
                "validation_group",
                key=lambda values: values.map(
                    {
                        engine: index
                        for index, engine in enumerate(validation_groups)
                    }
                ),
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "validation_group": "Engine / pose selection",
                "compound_pass_rate": st.column_config.NumberColumn(
                    "Compound pass rate", format="percent"
                ),
                "pose_pass_rate": st.column_config.NumberColumn(
                    "Pose pass rate", format="percent"
                ),
            },
        )

    failures: Counter[str] = Counter()
    failed_checks = poses.get(
        "failed_checks", pd.Series("", index=poses.index)
    )
    for value in failed_checks.loc[~poses["passed_all"]].fillna(""):
        for check in str(value).replace(" · ", ";").split(";"):
            if check.strip():
                failures[check.strip()] += 1
    if failures:
        st.markdown("#### Most common failed checks")
        failure_frame = pd.DataFrame(
            failures.most_common(15), columns=["check", "failed_poses"]
        )
        st.altair_chart(
            alt.Chart(failure_frame)
            .mark_bar(color="#dc2626")
            .encode(
                x=alt.X("failed_poses:Q", title="Failed poses"),
                y=alt.Y("check:N", sort="-x", title=None),
                tooltip=["check:N", "failed_poses:Q"],
            )
            .properties(height=max(220, 28 * len(failure_frame))),
            width="stretch",
        )
    st.markdown("#### Compound summary")
    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            "compound_id": "Compound",
            "made_it": st.column_config.CheckboxColumn("≥1 engine PASS"),
            "passed_every_tested_engine": st.column_config.CheckboxColumn(
                "Every tested engine PASS"
            ),
            "complete_engine_coverage": st.column_config.CheckboxColumn(
                "All selected engines tested"
            ),
            "pose_pass_rate": st.column_config.NumberColumn(
                "Pose pass rate", format="percent"
            ),
        },
    )
    with st.expander("Linked validation jobs and provenance"):
        st.dataframe(
            provenance,
            hide_index=True,
            width="stretch",
            column_config={
                "result": st.column_config.LinkColumn(
                    "Result", display_text="Open"
                )
            },
        )


def summarize_metric(
    frame: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    values = _with_analysis_campaigns(frame)
    values["metric_value"] = pd.to_numeric(values[metric], errors="coerce")
    values = values.dropna(subset=["metric_value"])
    if values.empty:
        return pd.DataFrame()
    group_columns = [
        "candidate_id",
        *(["compound_name"] if "compound_name" in values else []),
        "_analysis_campaign_id",
        "_analysis_campaign",
        "engine",
        *(["target_run_id"] if "target_run_id" in values else []),
        "target",
        "dataset",
    ]
    grouped = (
        values.groupby(
            group_columns,
            dropna=False,
        )["metric_value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "_analysis_campaign_id": "campaign_id",
                "_analysis_campaign": "campaign",
                "mean": "Mean",
                "std": "Sample SD",
                "count": "Attempts",
            }
        )
    )
    grouped["Sample SD"] = grouped["Sample SD"].fillna(0.0)
    grouped["Lower"] = grouped["Mean"] - grouped["Sample SD"]
    grouped["Upper"] = grouped["Mean"] + grouped["Sample SD"]
    return grouped


def _with_compound_plot_labels(
    summary: pd.DataFrame,
    *,
    label_mode: str,
) -> pd.DataFrame:
    """Add display labels while preserving compound IDs as row identity."""
    prepared = summary.copy()
    identifiers = prepared["candidate_id"].fillna("").astype(str)
    if label_mode == "Compound IDs" or "compound_name" not in prepared:
        prepared["_compound_plot_label"] = identifiers
        return prepared
    names = prepared["compound_name"].fillna("").astype(str).str.strip()
    usable = ~names.str.casefold().isin({"", "nan", "none", "<na>"})
    if label_mode == "Compound names":
        prepared["_compound_plot_label"] = [
            f"{name}|||{identifier}" if has_name else identifier
            for name, identifier, has_name in zip(names, identifiers, usable)
        ]
    else:
        prepared["_compound_plot_label"] = [
            f"{name}|||({identifier})" if has_name else identifier
            for name, identifier, has_name in zip(names, identifiers, usable)
        ]
    return prepared


def _compound_selector_labels(frame: pd.DataFrame) -> dict[str, str]:
    """Return stable ID-backed selector labels, adding names when available."""
    if frame.empty or "candidate_id" not in frame:
        return {}
    columns = ["candidate_id"]
    if "compound_name" in frame:
        columns.append("compound_name")
    labels: dict[str, str] = {}
    selector_rows = frame[columns].fillna("").astype(str).drop_duplicates()
    for _, row in selector_rows.iterrows():
        identifier = str(row.get("candidate_id", "")).strip()
        if not identifier or identifier.casefold() in {"nan", "none", "<na>"}:
            continue
        name = str(row.get("compound_name", "")).strip()
        has_name = (
            name.casefold() not in {"", "nan", "none", "<na>"}
            and name.casefold() != identifier.casefold()
        )
        if identifier not in labels or has_name:
            labels[identifier] = (
                f"{name} ({identifier})" if has_name else identifier
            )
    return labels


def _compound_axis_label_expression(label_mode: str) -> str | None:
    if label_mode == "Compound names":
        return "split(datum.label, '|||')[0]"
    if label_mode == "Names + IDs":
        return "split(datum.label, '|||')"
    return None


def select_metric_attempts(
    frame: pd.DataFrame,
    metric: str,
    *,
    mode: str,
    best_count: int,
    higher_is_better: bool,
) -> pd.DataFrame:
    """Select attempt rows per compound and logical engine campaign."""
    values = _with_analysis_campaigns(frame)
    values["_selection_metric"] = pd.to_numeric(
        values[metric],
        errors="coerce",
    )
    values = values.dropna(subset=["_selection_metric"])
    if values.empty or mode == "All repetitions":
        return values.drop(columns=["_selection_metric"], errors="ignore")
    group_columns = ["candidate_id", "_analysis_campaign_id"]
    if mode == "Representative repetition":
        medians = values.groupby(group_columns)["_selection_metric"].transform(
            "median"
        )
        values["_selection_distance"] = (
            values["_selection_metric"] - medians
        ).abs()
        values = (
            values.sort_values(
                [
                    *group_columns,
                    "_selection_distance",
                    "_selection_metric",
                ],
                ascending=[
                    True,
                    True,
                    True,
                    not higher_is_better,
                ],
                kind="stable",
            )
            .groupby(group_columns, sort=False, as_index=False)
            .head(1)
        )
    else:
        values = (
            values.sort_values(
                [*group_columns, "_selection_metric"],
                ascending=[True, True, not higher_is_better],
                kind="stable",
            )
            .groupby(group_columns, sort=False, as_index=False)
            .head(max(1, int(best_count)))
        )
    return values.drop(
        columns=["_selection_metric", "_selection_distance"],
        errors="ignore",
    )


def _with_analysis_campaigns(frame: pd.DataFrame) -> pd.DataFrame:
    """Group supplemental child jobs into their logical engine campaign.

    Jobs remain individually selectable by ``campaign_id`` elsewhere. Analysis
    combines only jobs sharing launch, engine, and target, so adding missing
    independent attempts later does not create a second bar or an extra
    consensus vote.
    """
    prepared = frame.copy()
    fallback_id = prepared.get(
        "campaign_id",
        pd.Series("", index=prepared.index, dtype=str),
    ).astype(str)
    launch_id = prepared.get(
        "launch_campaign_id",
        fallback_id,
    ).astype(str)
    launch_id = launch_id.where(launch_id.str.strip().ne(""), fallback_id)
    engine = prepared.get(
        "engine",
        pd.Series("", index=prepared.index, dtype=str),
    ).astype(str)
    target = prepared.get(
        "target_run_id",
        pd.Series("", index=prepared.index, dtype=str),
    ).astype(str)
    prepared["_analysis_campaign_id"] = (
        launch_id + "::" + engine + "::" + target
    )
    fallback_label = prepared.get(
        "campaign",
        fallback_id,
    ).astype(str)
    launch_label = prepared.get(
        "launch_campaign",
        fallback_label,
    ).astype(str)
    base_label = launch_label.where(
        launch_label.str.strip().ne(""),
        fallback_label,
    )
    target_label = prepared.get(
        "target",
        target,
    ).fillna("").astype(str)
    prepared["_analysis_campaign"] = (
        base_label
        + " · "
        + target_label
    )
    target_ligand_mask = prepared.get(
        "campaign_purpose",
        pd.Series("", index=prepared.index, dtype=str),
    ).astype(str).eq("target_ligand_redocking_refolding")
    prepared.loc[target_ligand_mask, "_analysis_campaign"] = target_label.loc[
        target_ligand_mask
    ]
    return prepared


def consensus_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _with_analysis_campaigns(frame)
    coverage = frame.groupby("_analysis_campaign_id")[
        "candidate_id"
    ].nunique()
    maximum_coverage = int(coverage.max()) if not coverage.empty else 0
    minimum_coverage = max(2, math.ceil(0.8 * maximum_coverage))
    eligible_campaigns = coverage.loc[
        coverage.ge(minimum_coverage)
    ].index
    frame = frame.loc[
        frame["_analysis_campaign_id"].isin(eligible_campaigns)
    ].copy()
    summaries: list[pd.DataFrame] = []
    for engine, primary in PRIMARY_METRIC.items():
        if engine.endswith("rescoring"):
            continue
        if primary not in frame:
            continue
        subset = frame.loc[frame["engine"].eq(engine)].copy()
        summary = summarize_metric(subset, primary)
        if summary.empty:
            continue
        higher_is_better = next(
            direction
            for metric, _, direction in ENGINE_METRICS[engine]
            if metric == primary
        )
        summary["Within-campaign percentile"] = summary.groupby(
            "campaign_id"
        )["Mean"].rank(
            method="average",
            pct=True,
            ascending=higher_is_better,
        )
        summary["Primary metric"] = primary
        summaries.append(summary)
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True, sort=False)


def correlation_feature_table(
    frame: pd.DataFrame,
    *,
    target_run_id: str,
    minimum_compounds: int = 5,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Build one direction-normalized compound value per engine metric."""
    target_frame = _with_analysis_campaigns(frame.loc[
        frame["target_run_id"].eq(target_run_id)
    ].copy())
    feature_series: list[pd.Series] = []
    labels: dict[str, str] = {}
    defaults: list[str] = []
    for engine, definitions in ENGINE_METRICS.items():
        if engine.endswith("rescoring"):
            continue
        engine_rows = target_frame.loc[
            target_frame["engine"].eq(engine)
        ]
        if engine_rows.empty:
            continue
        for metric, metric_label, higher_is_better in definitions:
            if metric not in engine_rows:
                continue
            values = engine_rows[
                ["candidate_id", "_analysis_campaign_id", metric]
            ].copy()
            values[metric] = pd.to_numeric(
                values[metric], errors="coerce"
            )
            values = values.dropna(subset=[metric])
            concentration_metric = metric.endswith("ic50_uM")
            if concentration_metric:
                values = values.loc[values[metric].gt(0)].copy()
                values["_correlation_value"] = values[metric].map(
                    math.log10
                )
            else:
                values["_correlation_value"] = values[metric]
            if values["candidate_id"].nunique() < minimum_compounds:
                continue
            campaign_means = (
                values.groupby(
                    ["candidate_id", "_analysis_campaign_id"],
                    dropna=False,
                )["_correlation_value"]
                .mean()
                .reset_index()
            )
            compound_means = campaign_means.groupby("candidate_id")[
                "_correlation_value"
            ].mean()
            if concentration_metric:
                compound_means = compound_means.map(
                    lambda value: 10.0 ** float(value)
                )
            feature_id = f"{engine}::{metric}"
            feature_series.append(
                (
                    compound_means
                    if higher_is_better
                    else -compound_means
                ).rename(feature_id)
            )
            labels[feature_id] = (
                f"{engine} · {metric_label} (favorable ↑)"
            )
            primary_default = PRIMARY_METRIC.get(engine) == metric
            if primary_default or (
                engine == "GNINA"
                and metric == "empirical_ranked_score_kcal_mol"
            ):
                defaults.append(feature_id)
    if not feature_series:
        return pd.DataFrame(), {}, []
    table = pd.concat(feature_series, axis=1)
    return table, labels, defaults


def correlation_matrices(
    table: pd.DataFrame,
    *,
    method: str,
    minimum_overlap: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation = table.corr(
        method=method.lower(),
        min_periods=minimum_overlap,
    )
    available = table.notna().astype(int)
    overlap = available.T @ available
    return correlation, overlap


def target_comparison_feature_table(
    frame: pd.DataFrame,
    *,
    minimum_targets: int = 3,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
    list[str],
]:
    """Aggregate one favorable-oriented feature value per prepared target."""
    prepared = _with_analysis_campaigns(frame)
    fallback_target = prepared.get(
        "target_run_id",
        pd.Series("", index=prepared.index, dtype=str),
    ).fillna("").astype(str)
    launch_id = prepared.get(
        "launch_campaign_id",
        fallback_target,
    ).fillna("").astype(str)
    prepared["_target_comparison_id"] = (
        launch_id.where(launch_id.str.strip().ne(""), fallback_target)
        + "::"
        + fallback_target
    )
    fallback_label = prepared.get(
        "target",
        fallback_target,
    ).fillna("").astype(str)
    launch_label = prepared.get(
        "launch_campaign",
        fallback_label,
    ).fillna("").astype(str)
    prepared["_target_comparison_label"] = fallback_label
    metadata_columns = [
        "_target_comparison_id",
        "_target_comparison_label",
        "target_run_id",
        "target",
    ]
    metadata = (
        prepared[
            [column for column in metadata_columns if column in prepared]
        ]
        .drop_duplicates("_target_comparison_id")
        .rename(
            columns={
                "_target_comparison_id": "target_comparison_id",
                "_target_comparison_label": "target_preparation",
            }
        )
    )
    feature_series: list[pd.Series] = []
    summary_rows: list[pd.DataFrame] = []
    labels: dict[str, str] = {}
    defaults: list[str] = []
    for engine, definitions in ENGINE_METRICS.items():
        if engine.endswith("rescoring"):
            continue
        engine_rows = prepared.loc[prepared["engine"].eq(engine)].copy()
        if engine_rows.empty:
            continue
        for metric, metric_label, higher_is_better in definitions:
            if metric not in engine_rows:
                continue
            values = engine_rows[
                [
                    "_target_comparison_id",
                    "_target_comparison_label",
                    metric,
                ]
            ].copy()
            values[metric] = pd.to_numeric(values[metric], errors="coerce")
            values = values.dropna(subset=[metric])
            if values["_target_comparison_id"].nunique() < minimum_targets:
                continue
            summary = (
                values.groupby(
                    [
                        "_target_comparison_id",
                        "_target_comparison_label",
                    ],
                    as_index=False,
                )[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
                .rename(
                    columns={
                        "_target_comparison_id": "target_comparison_id",
                        "_target_comparison_label": "target_preparation",
                        "mean": "Mean",
                        "std": "Sample SD",
                        "count": "Attempts",
                    }
                )
            )
            summary["Sample SD"] = summary["Sample SD"].fillna(0.0)
            feature_id = f"{engine}::{metric}"
            summary["feature_id"] = feature_id
            summary["engine"] = engine
            summary["metric"] = metric
            summary["metric_label"] = metric_label
            summary["higher_is_better"] = higher_is_better
            summary_rows.append(summary)
            favorable = summary.set_index("target_comparison_id")["Mean"]
            if not higher_is_better:
                favorable = -favorable
            feature_series.append(favorable.rename(feature_id))
            labels[feature_id] = f"{engine} · {metric_label} (favorable ↑)"
            if (
                PRIMARY_METRIC.get(engine) == metric
                or metric == "ligand_rmsd_angstrom"
            ):
                defaults.append(feature_id)
    table = (
        pd.concat(feature_series, axis=1)
        if feature_series
        else pd.DataFrame()
    )
    summaries = (
        pd.concat(summary_rows, ignore_index=True, sort=False)
        if summary_rows
        else pd.DataFrame()
    )
    return table, metadata, summaries, labels, list(dict.fromkeys(defaults))


def target_consensus_summary(
    feature_table: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank prepared targets within each direction-normalized feature."""
    available = [
        feature
        for feature in selected_features
        if feature in feature_table.columns
    ]
    if not available:
        return pd.DataFrame(), pd.DataFrame()
    percentiles = feature_table[available].rank(
        method="average",
        pct=True,
        ascending=True,
    )
    long = (
        percentiles.rename_axis("target_comparison_id")
        .reset_index()
        .melt(
            id_vars="target_comparison_id",
            var_name="feature_id",
            value_name="Within-feature percentile",
        )
        .dropna(subset=["Within-feature percentile"])
    )
    combined = (
        long.groupby("target_comparison_id")[
            "Within-feature percentile"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "Mean percentile",
                "std": "Sample SD",
                "count": "Contributing metrics",
            }
        )
        .sort_values("Mean percentile", ascending=False)
    )
    combined["Sample SD"] = combined["Sample SD"].fillna(0.0)
    return combined, long


def _scatter_matrix_display_label(label: str) -> str:
    """Use compact multiline labels for dense static pair plots."""
    compact = (
        str(label)
        .removesuffix(" (favorable ↑)")
        .rstrip()
        .replace(" · ", "\n")
    )
    prefix, separator, unit = compact.rpartition(" (")
    if separator and unit.endswith(")"):
        return f"{prefix}\n({unit}"
    return compact


def _metric_chart(
    summary: pd.DataFrame,
    label: str,
    *,
    higher_is_better: bool,
    compare_targets: bool = False,
    compound_label_mode: str = "Compound IDs",
    compound_label_layout: str = "Automatic",
) -> alt.Chart:
    summary = _with_compound_plot_labels(
        summary,
        label_mode=compound_label_mode,
    )
    entity_column = "campaign" if compare_targets else "_compound_plot_label"
    entity_title = "Prepared target" if compare_targets else "Compound"
    compound_order = (
        summary.groupby(entity_column, sort=False)["Mean"]
        .mean()
        .sort_values(
            ascending=not higher_is_better,
            kind="stable",
        )
        .index.astype(str)
        .tolist()
    )
    if compare_targets:
        base = alt.Chart(summary).encode(
            y=alt.Y(
                "campaign:N",
                title="Prepared target",
                sort=compound_order,
                axis=alt.Axis(labelLimit=420),
            ),
            tooltip=[
                alt.Tooltip("campaign:N", title="Prepared target"),
                alt.Tooltip("candidate_id:N", title="Ligand"),
                alt.Tooltip("compound_name:N", title="Ligand name"),
                alt.Tooltip("target:N", title="Target identity"),
                alt.Tooltip("Mean:Q", format=".4f"),
                alt.Tooltip("Sample SD:Q", format=".4f"),
                alt.Tooltip("Attempts:Q"),
            ],
        )
        bars = base.mark_bar(color="#2563eb", opacity=0.82).encode(
            x=alt.X("Mean:Q", title=label)
        )
        errors = base.mark_rule(color="#111827", strokeWidth=2).encode(
            x=alt.X("Lower:Q", title=label),
            x2="Upper:Q",
        )
        return (bars + errors).properties(
            height=max(390, 31 * summary["campaign"].nunique())
        )
    compound_count = int(summary[entity_column].nunique())
    use_horizontal_layout = (
        compound_label_layout == "Show all labels"
        or (
            compound_label_layout == "Automatic"
            and compound_count > 18
        )
    )
    compound_label_expression = _compound_axis_label_expression(
        compound_label_mode
    )
    if use_horizontal_layout:
        compound_axis_options: dict[str, object] = {
            "labelLimit": 420,
            "labelOverlap": False,
        }
        if compound_label_expression:
            compound_axis_options["labelExpr"] = compound_label_expression
        base = alt.Chart(summary).encode(
            y=alt.Y(
                f"{entity_column}:N",
                title=entity_title,
                sort=compound_order,
                axis=alt.Axis(**compound_axis_options),
            ),
            color=alt.Color("campaign:N", title="Campaign"),
            tooltip=[
                alt.Tooltip("candidate_id:N", title="Compound ID"),
                alt.Tooltip("compound_name:N", title="Compound name"),
                alt.Tooltip("campaign:N", title="Campaign"),
                alt.Tooltip("target:N", title="Target"),
                alt.Tooltip("dataset:N", title="Compound selection"),
                alt.Tooltip("Mean:Q", format=".4f"),
                alt.Tooltip("Sample SD:Q", format=".4f"),
                alt.Tooltip("Attempts:Q"),
            ],
        )
        bars = base.mark_bar(opacity=0.82).encode(
            x=alt.X("Mean:Q", title=label),
            yOffset=alt.YOffset("campaign:N", bandPosition=0.5),
        )
        errors = base.mark_rule(strokeWidth=2).encode(
            x=alt.X("Lower:Q", title=label),
            x2="Upper:Q",
            yOffset=alt.YOffset("campaign:N", bandPosition=0.5),
        )
        return (bars + errors).properties(
            height=max(390, 30 * compound_count)
        )
    compound_axis_options: dict[str, object] = {
        "labelAngle": -45,
        "labelLimit": 220,
        "labelOverlap": (
            "greedy" if compound_label_layout == "Compact" else False
        ),
    }
    if compound_label_expression:
        compound_axis_options["labelExpr"] = compound_label_expression
    base = alt.Chart(summary).encode(
        x=alt.X(
            f"{entity_column}:N",
            title=entity_title,
            sort=compound_order,
            axis=alt.Axis(**compound_axis_options),
        ),
        color=alt.Color("campaign:N", title="Campaign"),
        tooltip=[
            alt.Tooltip("candidate_id:N", title="Compound ID"),
            alt.Tooltip("compound_name:N", title="Compound name"),
            alt.Tooltip("campaign:N", title="Campaign"),
            alt.Tooltip("target:N", title="Target"),
            alt.Tooltip("dataset:N", title="Compound selection"),
            alt.Tooltip("Mean:Q", format=".4f"),
            alt.Tooltip("Sample SD:Q", format=".4f"),
            alt.Tooltip("Attempts:Q"),
        ],
    )
    bars = base.mark_bar(opacity=0.82).encode(
        y=alt.Y("Mean:Q", title=label),
        xOffset=alt.XOffset("campaign:N", bandPosition=0.5),
    )
    errors = base.mark_rule(strokeWidth=2).encode(
        y=alt.Y("Lower:Q", title=label),
        y2="Upper:Q",
        xOffset=alt.XOffset("campaign:N", bandPosition=0.5),
    )
    return (bars + errors).properties(height=390)


def _representative_structure_rows(frame: pd.DataFrame) -> pd.DataFrame:
    representatives: list[pd.Series] = []
    for _, campaign_rows in frame.groupby("campaign_id", sort=False):
        engine = str(campaign_rows.iloc[0]["engine"])
        primary = PRIMARY_METRIC.get(engine, "")
        values = (
            pd.to_numeric(campaign_rows.get(primary), errors="coerce")
            if primary in campaign_rows
            else pd.Series(dtype=float)
        )
        if values.notna().any():
            median = float(values.median())
            index = (values - median).abs().idxmin()
            representatives.append(campaign_rows.loc[index])
        else:
            representatives.append(campaign_rows.iloc[0])
    return (
        pd.DataFrame(representatives).reset_index(drop=True)
        if representatives
        else pd.DataFrame()
    )


def _best_structure_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the most favorable available prediction for each engine."""
    best_rows: list[pd.Series] = []
    for engine, engine_rows in frame.groupby("engine", sort=False):
        primary = PRIMARY_METRIC.get(str(engine), "")
        values = (
            pd.to_numeric(engine_rows.get(primary), errors="coerce")
            if primary in engine_rows
            else pd.Series(dtype=float)
        )
        if values.notna().any():
            higher_is_better = next(
                (
                    direction
                    for metric, _, direction in ENGINE_METRICS.get(
                        str(engine), ()
                    )
                    if metric == primary
                ),
                True,
            )
            index = (
                values.idxmax()
                if higher_is_better
                else values.idxmin()
            )
            best_rows.append(engine_rows.loc[index])
        else:
            best_rows.append(engine_rows.iloc[0])
    return (
        pd.DataFrame(best_rows).reset_index(drop=True)
        if best_rows
        else pd.DataFrame()
    )


def _native_repeat_structure_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one native representative per independent AF3/Boltz repeat.

    ``All repetitions`` means independent workflow repeats.  AF3 samples and
    Boltz model outputs are alternatives within one repeat, not extra
    repetitions to include in an all-vs-all pose matrix.
    """
    parts: list[pd.DataFrame] = []
    for engine, engine_rows in frame.groupby("engine", sort=False):
        if str(engine) == "AlphaFold 3":
            parts.append(_select_alphafold3_report_rows(engine_rows))
        elif str(engine) == "Boltz-2":
            parts.append(_select_boltz2_report_rows(engine_rows))
        else:
            parts.append(engine_rows)
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def _select_gnina_ranking_rows(
    frame: pd.DataFrame,
    criterion_label: str,
) -> pd.DataFrame:
    """Select or duplicate GNINA rows for its two emitted pose rankings."""
    selected = frame.copy()
    gnina_mask = selected["engine"].astype(str).eq("GNINA")
    if not gnina_mask.any():
        return selected
    criteria = (
        (
            "GNINA · CNN-ranked",
            "cnn_ranked_pose_index",
            "CNN-ranked",
        ),
        (
            "GNINA · Vina-ranked",
            "empirical_ranked_pose_index",
            "Vina-ranked",
        ),
    )
    if criterion_label == "Both rankings":
        parts = [selected.loc[~gnina_mask].copy()]
        for engine_label, index_column, ranking_label in criteria:
            ranked = selected.loc[gnina_mask].copy()
            ranked["engine"] = engine_label
            ranked["_viewer_pose_index"] = (
                pd.to_numeric(ranked.get(index_column), errors="coerce")
                .fillna(1)
                .astype(int)
            )
            ranked["campaign_id"] = (
                ranked["campaign_id"].astype(str)
                + f"::{ranking_label.lower()}"
            )
            ranked["_prediction_label"] = (
                ranked["_prediction_label"].astype(str)
                + f" · {ranking_label}"
            )
            parts.append(ranked)
        return pd.concat(parts, ignore_index=True)
    index_column = (
        "empirical_ranked_pose_index"
        if criterion_label == "Empirical / Vina score"
        else "cnn_ranked_pose_index"
    )
    engine_label = (
        "GNINA · Vina-ranked"
        if criterion_label == "Empirical / Vina score"
        else "GNINA · CNN-ranked"
    )
    ranking_label = (
        "Vina-ranked"
        if criterion_label == "Empirical / Vina score"
        else "CNN-ranked"
    )
    selected.loc[gnina_mask, "_viewer_pose_index"] = (
        pd.to_numeric(
            selected.loc[gnina_mask, index_column],
            errors="coerce",
        )
        .fillna(1)
        .astype(int)
    )
    selected.loc[gnina_mask, "engine"] = engine_label
    selected.loc[gnina_mask, "_prediction_label"] = (
        selected.loc[gnina_mask, "_prediction_label"].astype(str)
        + f" · {ranking_label}"
    )
    return selected


def _ordered_structure_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "engine" not in frame:
        return frame
    rank = {
        engine: index
        for index, engine in enumerate(STRUCTURE_ENGINE_ORDER)
    }
    gnina_rank = float(rank.get("GNINA", len(rank)))
    rank["GNINA · CNN-ranked"] = gnina_rank
    rank["GNINA · Vina-ranked"] = gnina_rank + 0.1
    ordered = frame.copy()
    ordered["_viewer_engine_order"] = ordered["engine"].astype(str).map(
        lambda engine: rank.get(engine, len(rank))
    )
    ordered = ordered.sort_values(
        ["_viewer_engine_order"],
        kind="stable",
    )
    return ordered.drop(columns=["_viewer_engine_order"])


def _model_text(path: Path, pose_index: int = 1) -> tuple[str, str]:
    suffix = path.suffix.lower()
    model_format = {
        ".cif": "cif",
        ".mmcif": "cif",
        ".pdb": "pdb",
        ".pdbqt": "pdbqt",
        ".sdf": "sdf",
        ".mol": "mol",
        ".mol2": "mol2",
    }.get(suffix, suffix.lstrip(".") or "pdb")
    text = path.read_text(errors="replace")
    if suffix == ".sdf":
        records = [
            record.strip("\r\n")
            for record in text.split("$$$$")
            if record.strip()
        ]
        selected = max(0, int(pose_index or 1) - 1)
        record = (
            records[selected]
            if selected < len(records)
            else records[0]
            if records
            else text.rstrip()
        )
        text = record + "\n$$$$\n"
    return text, model_format


def _preferred_viewer_structure_path(
    structure_path: Path,
    *,
    structure_kind: str,
) -> Path:
    """Use the same chemically complete pose source as single-job viewers."""
    if structure_kind == "pose" and structure_path.suffix.lower() == ".pdbqt":
        sdf_path = structure_path.with_suffix(".sdf")
        if sdf_path.is_file():
            return sdf_path
    return structure_path


def _pose_label(row: pd.Series, ordinal: int) -> str:
    attempt = next(
        (
            str(row.get(column))
            for column in (
                "replicate",
                "seed",
                "model_seed",
                "prediction_id",
                "model_id",
            )
            if str(row.get(column) or "").strip()
            not in {"", "nan", "None"}
        ),
        str(ordinal),
    )
    return (
        f"{row.get('engine', 'Unknown')}\n"
        f"{row.get('job_code', '')} · {attempt}"
    )


def _sdf_pose_molecule(path: Path, pose_index: int = 1):
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(
        str(path),
        removeHs=False,
        sanitize=False,
    )
    molecules = [value for value in supplier if value is not None]
    selected = max(0, int(pose_index or 1) - 1)
    molecule = (
        molecules[selected]
        if selected < len(molecules)
        else molecules[0]
        if molecules
        else None
    )
    if molecule is None:
        raise ValueError("RDKit could not read the ligand pose")
    molecule = Chem.RemoveHs(molecule, sanitize=False)
    if molecule.GetNumAtoms() < 1 or molecule.GetNumConformers() < 1:
        raise ValueError("The ligand pose contains no heavy-atom coordinates")
    Chem.GetSymmSSSR(molecule)
    return molecule


def _sdf_reference_molecule(path: Path, candidate_id: str):
    """Select a compound's input pose from its prepared ligand artifact."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(
        str(path),
        removeHs=False,
        sanitize=False,
    )
    molecules = [value for value in supplier if value is not None]
    if not molecules:
        raise ValueError("RDKit could not read the input/reference ligand")
    normalized_candidate = str(candidate_id or "").strip().lower()
    selected = None
    for molecule in molecules:
        identifiers = {str(molecule.GetProp("_Name") or "").strip().lower()}
        for property_name in molecule.GetPropNames():
            if property_name.lower() in {"compound_id", "candidate_id", "id"}:
                identifiers.add(
                    str(molecule.GetProp(property_name)).strip().lower()
                )
        if normalized_candidate and normalized_candidate in identifiers:
            selected = molecule
            break
    if selected is None and len(molecules) == 1:
        selected = molecules[0]
    if selected is None:
        match = re.search(r"(\d+)$", normalized_candidate)
        ordinal = int(match.group(1)) if match else 0
        if 1 <= ordinal <= len(molecules):
            selected = molecules[ordinal - 1]
    if selected is None:
        raise ValueError(
            "The selected compound could not be located in the input ligand set"
        )
    selected = Chem.RemoveHs(Chem.Mol(selected), sanitize=False)
    if selected.GetNumAtoms() < 1 or selected.GetNumConformers() < 1:
        raise ValueError("The input/reference ligand has no heavy-atom coordinates")
    Chem.GetSymmSSSR(selected)
    return selected


def _complex_pose_molecule(
    structure_data: str,
    ligand_smiles: str,
    ligand_template=None,
):
    """Recover the cofolded ligand on its input-SMILES atom topology."""
    import gemmi
    from rdkit import Chem

    template = Chem.MolFromSmiles(str(ligand_smiles))
    if (
        (template is None or template.GetNumAtoms() == 0)
        and ligand_template is not None
    ):
        template = Chem.Mol(ligand_template)
    if template is None or template.GetNumAtoms() == 0:
        raise ValueError(
            "No input SMILES or compatible docking-pose topology is available"
        )
    template = Chem.RemoveHs(template, sanitize=False)
    first_cif_token = next(
        (
            line.strip()
            for line in structure_data.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        "",
    )
    is_cif = first_cif_token.startswith(("data_", "loop_", "_entry."))
    if is_cif:
        document = gemmi.cif.read_string(structure_data)
        structure = gemmi.make_structure_from_block(document.sole_block())
    else:
        structure = gemmi.read_pdb_string(structure_data)
    expected_elements = [
        atom.GetSymbol().upper() for atom in template.GetAtoms()
    ]
    expected_counts = Counter(expected_elements)
    candidates: list[list[tuple[str, float, float, float]]] = []
    for chain in structure[0]:
        for residue in chain:
            residue_info = gemmi.find_tabulated_residue(residue.name)
            if residue_info.is_amino_acid() or residue_info.is_water():
                continue
            atoms = [
                atom
                for atom in residue
                if atom.element.name.upper() != "H"
            ]
            elements = [atom.element.name.upper() for atom in atoms]
            if (
                len(atoms) == template.GetNumAtoms()
                and Counter(elements) == expected_counts
            ):
                candidates.append(
                    [
                        (
                            atom.element.name.upper(),
                            float(atom.pos.x),
                            float(atom.pos.y),
                            float(atom.pos.z),
                        )
                        for atom in atoms
                    ]
                )
    if not candidates and not is_cif:
        pdb_groups: dict[
            tuple[str, str, str], list[tuple[str, float, float, float]]
        ] = {}
        for line in structure_data.splitlines():
            if line[:6].strip() != "HETATM" or len(line) < 54:
                continue
            residue_name = line[17:20].strip().upper()
            if residue_name in {"HOH", "WAT", "SOL"}:
                continue
            element = line[76:78].strip().upper()
            if not element:
                element = "".join(
                    character
                    for character in line[12:16]
                    if character.isalpha()
                )[:1].upper()
            try:
                coordinates = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError:
                continue
            key = (
                line[21:22].strip(),
                line[22:27].strip(),
                residue_name,
            )
            pdb_groups.setdefault(key, []).append((element, *coordinates))
        candidates = [
            atoms
            for atoms in pdb_groups.values()
            if len(atoms) == template.GetNumAtoms()
            and Counter(atom[0] for atom in atoms) == expected_counts
        ]
    if len(candidates) != 1:
        raise ValueError(
            "Could not identify exactly one non-polymer ligand matching "
            "the selected compound"
        )
    atoms = candidates[0]
    observed_elements = [atom[0] for atom in atoms]
    if observed_elements != expected_elements:
        raise ValueError(
            "The cofolded ligand atom order cannot be mapped safely to its "
            "input molecular graph"
        )
    conformer = Chem.Conformer(template.GetNumAtoms())
    for index, (_, x, y, z) in enumerate(atoms):
        conformer.SetAtomPosition(
            index,
            (x, y, z),
        )
    template.RemoveAllConformers()
    template.AddConformer(conformer, assignId=True)
    return template


def _symmetry_fixed_frame_rmsd(
    left,
    right,
    ligand_smiles: str = "",
    mapping_cache: dict[tuple[int, str], tuple[tuple[int, ...], ...]] | None = None,
) -> float:
    """Return a symmetry-aware heavy-atom RMSD without fitting either pose.

    When available, the input ligand defines the authoritative atom topology.
    This avoids treating engine-specific SDF bond-order or protonation inference
    as atom identity (for example, swapping equivalent iodines in T3).
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    if left.GetNumAtoms() != right.GetNumAtoms():
        raise ValueError("Heavy-atom counts differ")
    if left.GetNumAtoms() == 1:
        if left.GetAtomWithIdx(0).GetSymbol() != right.GetAtomWithIdx(0).GetSymbol():
            raise ValueError("Single-atom elements differ")
        left_point = left.GetConformer().GetAtomPosition(0)
        right_point = right.GetConformer().GetAtomPosition(0)
        return math.sqrt(
            (left_point.x - right_point.x) ** 2
            + (left_point.y - right_point.y) ** 2
            + (left_point.z - right_point.z) ** 2
        )
    # Some engine SDF conversions intentionally bypass sanitization to retain
    # native coordinates and charges. MCS ring constraints still require the
    # non-mutating ring cache to be initialized.
    Chem.GetSymmSSSR(left)
    Chem.GetSymmSSSR(right)
    query = None
    authoritative_query = False
    authoritative = Chem.MolFromSmiles(str(ligand_smiles or ""))
    if authoritative is not None:
        authoritative = Chem.RemoveHs(authoritative, sanitize=False)
        if authoritative.GetNumAtoms() == left.GetNumAtoms():
            query_parameters = Chem.AdjustQueryParameters()
            query_parameters.makeBondsGeneric = True
            query_parameters.adjustDegree = False
            query_parameters.adjustRingCount = False
            query_parameters.adjustRingChain = False
            query = Chem.AdjustQueryProperties(
                authoritative,
                query_parameters,
            )
            if (
                not left.HasSubstructMatch(query)
                or not right.HasSubstructMatch(query)
            ):
                query = None
            else:
                authoritative_query = True

    # Fall back to an engine-pair MCS for legacy results without an input
    # ligand definition. The authoritative query above is preferred because
    # converted SDF files can assign different bond orders or formal charges.
    if query is None:
        common = rdFMCS.FindMCS(
            [left, right],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=10,
        )
        query = Chem.MolFromSmarts(common.smartsString)
        if query is None or query.GetNumAtoms() != left.GetNumAtoms():
            mapped = query.GetNumAtoms() if query is not None else 0
            raise ValueError(
                f"Chemical identity mismatch ({mapped}/{left.GetNumAtoms()} "
                "heavy atoms mapped)"
            )
    # A pose participates in many pairwise comparisons.  Substructure matches
    # are topology-only (not coordinate-dependent), so reuse them for every
    # later pair involving the same RDKit molecule.  This is exactly the same
    # set of symmetry mappings as the previous calculation.
    query_key = Chem.MolToSmarts(query)

    def matches_for(molecule):
        key = (id(molecule), query_key)
        if mapping_cache is not None and key in mapping_cache:
            return mapping_cache[key]
        matches = molecule.GetSubstructMatches(
            query, uniquify=False, maxMatches=100_000
        )
        if mapping_cache is not None:
            mapping_cache[key] = matches
        return matches

    left_matches = matches_for(left)
    right_matches = matches_for(right)
    if not left_matches or not right_matches:
        raise ValueError("No symmetry-aware atom mapping is available")
    left_conf = left.GetConformer()
    right_conf = right.GetConformer()
    best = math.inf
    # With the authoritative input-ligand query, every left embedding differs
    # from any other only by a query automorphism.  Holding one left embedding
    # fixed while retaining *all* right embeddings enumerates precisely the
    # same relative atom permutations as the previous left×right Cartesian
    # product, without repeatedly evaluating equivalent combinations.
    #
    # The pair-specific MCS fallback does not have that guarantee: an MCS can
    # occur at distinct non-equivalent locations in either molecule.  Keep its
    # conservative exhaustive behaviour unchanged.
    left_embedding_sets = left_matches[:1] if authoritative_query else left_matches
    for left_match in left_embedding_sets:
        for right_match in right_matches:
            square_distance = 0.0
            for left_index, right_index in zip(left_match, right_match):
                left_point = left_conf.GetAtomPosition(left_index)
                right_point = right_conf.GetAtomPosition(right_index)
                square_distance += (
                    (left_point.x - right_point.x) ** 2
                    + (left_point.y - right_point.y) ** 2
                    + (left_point.z - right_point.z) ** 2
                )
            best = min(
                best,
                math.sqrt(square_distance / len(left_match)),
            )
    if not math.isfinite(best):
        raise ValueError("No symmetry-aware RMSD could be evaluated")
    return best


def _pose_centroid(molecule) -> tuple[float, float, float]:
    conformer = molecule.GetConformer()
    coordinates = [
        conformer.GetAtomPosition(index)
        for index in range(molecule.GetNumAtoms())
    ]
    return (
        sum(point.x for point in coordinates) / len(coordinates),
        sum(point.y for point in coordinates) / len(coordinates),
        sum(point.z for point in coordinates) / len(coordinates),
    )


def _fixed_frame_shape_distance(left, right) -> float:
    """Return RDKit shape-Tanimoto distance without moving either pose."""
    from rdkit.Chem import rdShapeHelpers

    return float(rdShapeHelpers.ShapeTanimotoDist(left, right))


_POSE_SIMILARITY_CACHE_VERSION = 1
_POSE_SIMILARITY_PAIR_CACHE_VERSION = 1


def _pose_similarity_pair_cache_file(
    left_fingerprint: str,
    right_fingerprint: str,
    ligand_smiles: str,
) -> Path:
    """Content-addressed exact metric cache for one unordered pose pair."""
    signature = {
        "version": _POSE_SIMILARITY_PAIR_CACHE_VERSION,
        "left": left_fingerprint,
        "right": right_fingerprint,
        "ligand_smiles": ligand_smiles,
    }
    # RMSD, centroid distance, and shape distance are symmetric, so cache one
    # canonical pair independently of presentation order.
    signature["left"], signature["right"] = sorted(
        (signature["left"], signature["right"])
    )
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return runs_root() / ".cache" / "pose-similarity" / "pairs" / f"{digest}.json"


def _read_pose_similarity_pair_cache(path: Path) -> tuple[float, float, float] | None:
    try:
        payload = json.loads(path.read_text())
        if payload.get("version") != _POSE_SIMILARITY_PAIR_CACHE_VERSION:
            return None
        return tuple(float(payload[key]) for key in ("rmsd", "centroid", "shape"))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _write_pose_similarity_pair_cache(
    path: Path,
    *,
    rmsd: float,
    centroid: float,
    shape: float,
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "version": _POSE_SIMILARITY_PAIR_CACHE_VERSION,
            "rmsd": rmsd,
            "centroid": centroid,
            "shape": shape,
        }, separators=(",", ":")) + "\n")
        temporary.replace(path)
    except (OSError, ValueError, TypeError):
        return


def _pose_input_fingerprint(item: dict[str, object]) -> str:
    """Coordinates and selected pose identity, independent of table labels."""
    source = Path(str(item.get("source_path") or ""))
    try:
        stat = source.stat()
        source_state: object = (str(source.resolve()), stat.st_size, stat.st_mtime_ns)
    except OSError:
        source_state = str(source)
    row = item["row"]
    payload = {
        "source": source_state,
        "pose_index": int(row.get("_viewer_pose_index") or 1),
        "kind": str(item.get("kind") or ""),
        # Aligned cofolded structures are represented by these coordinates,
        # rather than only their source file identity.
        "coordinates": hashlib.sha256(str(item.get("data") or "").encode()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _pose_similarity_cache_file(
    rendered: list[dict[str, object]],
    *,
    reference_structure_data: str,
    reference_ligand_path: Path | None,
) -> Path:
    """Return a content-addressed cache path for immutable pose comparisons."""
    pose_inputs = []
    for item in rendered:
        row = item["row"]
        source = Path(str(item["source_path"]))
        try:
            stat = source.stat()
            source_state = (str(source.resolve()), stat.st_size, stat.st_mtime_ns)
        except OSError:
            source_state = (str(source), None, None)
        pose_inputs.append(
            {
                "source": source_state,
                "pose_index": int(row.get("_viewer_pose_index") or 1),
                "kind": str(item.get("kind") or ""),
                "candidate": str(row.get("candidate_id") or ""),
                "engine": str(row.get("engine") or ""),
                "smiles": str(row.get("_ligand_smiles") or ""),
            }
        )
    if reference_ligand_path is not None:
        try:
            ref_stat = reference_ligand_path.stat()
            reference_ligand_state = (
                str(reference_ligand_path.resolve()),
                ref_stat.st_size,
                ref_stat.st_mtime_ns,
            )
        except OSError:
            reference_ligand_state = (str(reference_ligand_path), None, None)
    else:
        reference_ligand_state = None
    signature = {
        "version": _POSE_SIMILARITY_CACHE_VERSION,
        "poses": pose_inputs,
        "reference_ligand": reference_ligand_state,
        "reference_structure_sha256": hashlib.sha256(
            reference_structure_data.encode("utf-8")
        ).hexdigest(),
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return runs_root() / ".cache" / "pose-similarity" / f"{digest}.json"


def _read_pose_similarity_cache(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]] | None:
    try:
        payload = json.loads(path.read_text())
        if payload.get("version") != _POSE_SIMILARITY_CACHE_VERSION:
            return None
        matrix_data = payload["matrix"]
        matrix = pd.DataFrame(
            matrix_data["data"],
            index=matrix_data["index"],
            columns=matrix_data["columns"],
            dtype=float,
        )
        pairs = pd.DataFrame(payload.get("pairs") or [])
        warnings = [str(item) for item in payload.get("warnings") or []]
        return matrix, pairs, warnings
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _write_pose_similarity_cache(
    path: Path,
    matrix: pd.DataFrame,
    pairs: pd.DataFrame,
    warnings: list[str],
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _POSE_SIMILARITY_CACHE_VERSION,
            "matrix": {
                "index": matrix.index.tolist(),
                "columns": matrix.columns.tolist(),
                "data": matrix.astype(object).where(pd.notna(matrix), None).values.tolist(),
            },
            "pairs": pairs.where(pd.notna(pairs), None).to_dict(orient="records"),
            "warnings": warnings,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                payload,
                separators=(",", ":"),
                default=lambda value: value.item()
                if hasattr(value, "item")
                else str(value),
            )
            + "\n"
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError):
        # Cache availability must never prevent a scientifically valid view.
        return


def _pose_similarity_tables(
    rendered: list[dict[str, object]],
    *,
    reference_structure_data: str = "",
    reference_ligand_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cache_path = _pose_similarity_cache_file(
        rendered,
        reference_structure_data=reference_structure_data,
        reference_ligand_path=reference_ligand_path,
    )
    cached = _read_pose_similarity_cache(cache_path)
    if cached is not None:
        return cached
    poses: list[tuple[str, object, str]] = []
    warnings: list[str] = []
    prediction_labels = [
        _pose_label(item["row"], ordinal)
        for ordinal, item in enumerate(rendered, start=1)
    ]
    reference_label = "Input/reference ligand"
    reference_ligand_available = bool(
        reference_ligand_path is not None
        and reference_ligand_path.is_file()
        and reference_ligand_path.suffix.lower() in {".sdf", ".mol"}
    )
    has_reference = bool(reference_structure_data) or reference_ligand_available
    all_labels = [
        *([reference_label] if has_reference else []),
        *prediction_labels,
    ]
    pose_molecules: dict[int, object] = {}
    ligand_template = None
    for ordinal, item in enumerate(rendered, start=1):
        if item["kind"] == "complex":
            continue
        try:
            molecule = _sdf_pose_molecule(
                Path(item["source_path"]),
                int(item["row"].get("_viewer_pose_index") or 1),
            )
        except (OSError, RuntimeError, ValueError):
            continue
        pose_molecules[ordinal] = molecule
        if ligand_template is None:
            ligand_template = molecule
    shared_smiles = next(
        (
            str(item["row"].get("_ligand_smiles") or "").strip()
            for item in rendered
            if str(item["row"].get("_ligand_smiles") or "").strip()
        ),
        "",
    )
    for ordinal, item in enumerate(rendered, start=1):
        row = item["row"]
        label = _pose_label(row, ordinal)
        try:
            if item["kind"] == "complex":
                molecule = _complex_pose_molecule(
                    str(item["data"]),
                    str(row.get("_ligand_smiles") or shared_smiles),
                    ligand_template=ligand_template,
                )
            else:
                molecule = pose_molecules.get(ordinal)
                if molecule is None:
                    molecule = _sdf_pose_molecule(
                        Path(item["source_path"]),
                        int(row.get("_viewer_pose_index") or 1),
                    )
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.append(f"{label.replace(chr(10), ' ')}: {exc}")
            continue
        poses.append((label, molecule, _pose_input_fingerprint(item)))
    matrix = pd.DataFrame(
        float("nan"),
        index=all_labels,
        columns=all_labels,
        dtype=float,
    )
    for label, _, _ in poses:
        matrix.loc[label, label] = 0.0
    reference_molecule = None
    if reference_ligand_available:
        try:
            candidate_id = str(rendered[0]["row"].get("candidate_id") or "")
            reference_molecule = _sdf_reference_molecule(
                reference_ligand_path,
                candidate_id,
            )
            matrix.loc[reference_label, reference_label] = 0.0
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.append(f"{reference_label}: {exc}")
    elif reference_structure_data:
        try:
            reference_molecule = _complex_pose_molecule(
                reference_structure_data,
                shared_smiles,
                ligand_template=ligand_template,
            )
            matrix.loc[reference_label, reference_label] = 0.0
        except (RuntimeError, ValueError) as exc:
            warnings.append(f"{reference_label}: {exc}")
    pair_rows: list[dict[str, object]] = []
    mapping_cache: dict[tuple[int, str], tuple[tuple[int, ...], ...]] = {}
    if reference_molecule is not None:
        reference_centroid = _pose_centroid(reference_molecule)
        reference_fingerprint = hashlib.sha256(
            (reference_structure_data + "\0" + str(reference_ligand_path or "")).encode()
        ).hexdigest()
        for pose_label, pose, pose_fingerprint in poses:
            try:
                cache_file = _pose_similarity_pair_cache_file(
                    reference_fingerprint, pose_fingerprint, shared_smiles
                )
                cached_metrics = _read_pose_similarity_pair_cache(cache_file)
                if cached_metrics is None:
                    rmsd = _symmetry_fixed_frame_rmsd(
                        reference_molecule, pose, shared_smiles, mapping_cache
                    )
                    centroid_distance = math.dist(reference_centroid, _pose_centroid(pose))
                    shape_distance = _fixed_frame_shape_distance(reference_molecule, pose)
                    _write_pose_similarity_pair_cache(
                        cache_file, rmsd=rmsd, centroid=centroid_distance, shape=shape_distance
                    )
                else:
                    rmsd, centroid_distance, shape_distance = cached_metrics
            except (RuntimeError, ValueError) as exc:
                warnings.append(f"{reference_label} versus {pose_label}: {exc}")
                continue
            matrix.loc[reference_label, pose_label] = rmsd
            matrix.loc[pose_label, reference_label] = rmsd
            pair_rows.append(
                {
                    "Pose A": reference_label,
                    "Pose B": pose_label.replace("\n", " · "),
                    "Engine A": "Input reference",
                    "Engine B": pose_label.split("\n", 1)[0],
                    "Comparison type": "Input-pose recovery",
                    "Fixed-frame RMSD (Å)": rmsd,
                    "RMSD metric": "Reference recovery",
                    "Reference": "Ligand pose in input complex",
                    "Centroid distance (Å)": centroid_distance,
                    "Shape Tanimoto distance": shape_distance,
                }
            )
    for left_index, (left_label, left, left_fingerprint) in enumerate(poses):
        left_centroid = _pose_centroid(left)
        for right_label, right, right_fingerprint in poses[left_index + 1 :]:
            try:
                cache_file = _pose_similarity_pair_cache_file(
                    left_fingerprint, right_fingerprint, shared_smiles
                )
                cached_metrics = _read_pose_similarity_pair_cache(cache_file)
                if cached_metrics is None:
                    rmsd = _symmetry_fixed_frame_rmsd(
                        left, right, shared_smiles, mapping_cache
                    )
                    centroid_distance = math.dist(left_centroid, _pose_centroid(right))
                    shape_distance = _fixed_frame_shape_distance(left, right)
                    _write_pose_similarity_pair_cache(
                        cache_file, rmsd=rmsd, centroid=centroid_distance, shape=shape_distance
                    )
                else:
                    rmsd, centroid_distance, shape_distance = cached_metrics
            except (RuntimeError, ValueError) as exc:
                warnings.append(
                    f"{left_label.replace(chr(10), ' ')} versus "
                    f"{right_label.replace(chr(10), ' ')}: {exc}"
                )
                continue
            matrix.loc[left_label, right_label] = rmsd
            matrix.loc[right_label, left_label] = rmsd
            pair_rows.append(
                {
                    "Pose A": left_label.replace("\n", " · "),
                    "Pose B": right_label.replace("\n", " · "),
                    "Engine A": left_label.split("\n", 1)[0],
                    "Engine B": right_label.split("\n", 1)[0],
                    "Comparison type": "Prediction agreement",
                    "Fixed-frame RMSD (Å)": rmsd,
                    "RMSD metric": "Inter-engine / inter-replicate agreement",
                    "Reference": "Other predicted pose in shared receptor frame",
                    "Centroid distance (Å)": centroid_distance,
                    "Shape Tanimoto distance": shape_distance,
                }
            )
    pairs = pd.DataFrame(pair_rows)
    _write_pose_similarity_cache(cache_path, matrix, pairs, warnings)
    return matrix, pairs, warnings


def _engine_pose_rmsd_summary(
    pairs: pd.DataFrame,
    engine_order: list[str],
    value_column: str = "Fixed-frame RMSD (Å)",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means = pd.DataFrame(
        float("nan"),
        index=engine_order,
        columns=engine_order,
        dtype=float,
    )
    annotations = pd.DataFrame(
        "",
        index=engine_order,
        columns=engine_order,
        dtype=object,
    )
    rows: list[dict[str, object]] = []
    for left_index, left_engine in enumerate(engine_order):
        for right_engine in engine_order[left_index:]:
            if left_engine == right_engine:
                selected = pairs.loc[
                    pairs["Engine A"].eq(left_engine)
                    & pairs["Engine B"].eq(right_engine)
                ]
            else:
                selected = pairs.loc[
                    (
                        pairs["Engine A"].eq(left_engine)
                        & pairs["Engine B"].eq(right_engine)
                    )
                    | (
                        pairs["Engine A"].eq(right_engine)
                        & pairs["Engine B"].eq(left_engine)
                    )
                ]
            values = pd.to_numeric(
                selected[value_column],
                errors="coerce",
            ).dropna()
            if values.empty:
                continue
            mean = float(values.mean())
            sample_sd = (
                float(values.std(ddof=1))
                if len(values) > 1
                else float("nan")
            )
            annotation = (
                f"{mean:.2f}\n± {sample_sd:.2f}"
                if math.isfinite(sample_sd)
                else f"{mean:.2f}\n(n=1)"
            )
            means.loc[left_engine, right_engine] = mean
            means.loc[right_engine, left_engine] = mean
            annotations.loc[left_engine, right_engine] = annotation
            annotations.loc[right_engine, left_engine] = annotation
            rows.append(
                {
                    "Engine A": left_engine,
                    "Engine B": right_engine,
                    "Mean RMSD (Å)": mean,
                    "Sample SD (Å)": sample_sd,
                    "Pose pairs": int(len(values)),
                    "Metric": value_column,
                }
            )
    return means, annotations, pd.DataFrame(rows)


def _input_pose_recovery_matrix(
    recovery: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arrange one-reference recovery values as engine by attempt."""
    prepared = recovery.copy()
    prepared["Engine"] = prepared["Engine"].fillna("").astype(str)
    engine_order = list(dict.fromkeys(prepared["Engine"].tolist()))
    prepared["Attempt"] = prepared.groupby("Engine", sort=False).cumcount() + 1
    matrix = prepared.pivot(
        index="Engine",
        columns="Attempt",
        values="Input-pose RMSD (Å)",
    ).reindex(engine_order)
    matrix.columns = [f"Attempt {int(attempt)}" for attempt in matrix.columns]
    return prepared, matrix


def _target_engine_recovery_matrix(
    recovery: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize input recovery across prepared targets and engines."""
    summary = (
        recovery.groupby(["Prepared target", "Engine"], sort=False)[
            "Input-pose RMSD (Å)"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["Annotation"] = summary.apply(
        lambda row: (
            f"{row['mean']:.3f}\n± {row['std']:.3f}"
            if int(row["count"]) > 1 and pd.notna(row["std"])
            else f"{row['mean']:.3f}"
        ),
        axis=1,
    )
    target_order = list(
        dict.fromkeys(recovery["Prepared target"].astype(str).tolist())
    )
    engine_order = list(dict.fromkeys(recovery["Engine"].astype(str).tolist()))
    means = summary.pivot(
        index="Prepared target", columns="Engine", values="mean"
    ).reindex(index=target_order, columns=engine_order)
    annotations = summary.pivot(
        index="Prepared target", columns="Engine", values="Annotation"
    ).reindex(index=target_order, columns=engine_order).fillna("")
    return means, annotations, summary


def _render_pose_similarity(
    rendered: list[dict[str, object]],
    *,
    reference_structure_data: str = "",
    reference_ligand_path: Path | None = None,
    result_view: str = "",
) -> None:
    st.markdown(
        "#### Pose similarity — inter-engine and inter-replicate agreement"
    )
    st.caption(
        "Heavy-atom RMSD uses the input ligand's chemical topology to resolve "
        "equivalent atoms and is calculated directly in the shared receptor "
        "frame. Ligands are not independently superposed, so the value "
        "retains differences in pocket placement, orientation and internal "
        "conformation."
    )
    matrix, pairs, warnings = _pose_similarity_tables(
        rendered,
        reference_structure_data=reference_structure_data,
        reference_ligand_path=reference_ligand_path,
    )
    reference_label = "Input/reference ligand"
    included_labels = {
        str(label)
        for label in matrix.index
        if pd.notna(matrix.loc[label, label])
    }
    coverage_rows: list[dict[str, object]] = []
    for ordinal, item in enumerate(rendered, start=1):
        row = item["row"]
        label = _pose_label(row, ordinal)
        flat_label = label.replace("\n", " · ")
        warning_label = label.replace("\n", " ")
        reason = next(
            (
                warning.split(": ", 1)[1]
                for warning in warnings
                if warning.startswith(warning_label + ": ")
            ),
            "",
        )
        coverage_rows.append(
            {
                "Engine": str(row.get("engine") or ""),
                "Prediction": flat_label,
                "Structure type": str(item.get("kind") or ""),
                "Included in RMSD": label in included_labels,
                "Exclusion reason": reason,
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage_metrics = st.columns(3)
    coverage_metrics[0].metric("Selected predictions", len(coverage))
    comparable_predictions = included_labels - {reference_label}
    coverage_metrics[1].metric(
        "Comparable poses", len(comparable_predictions)
    )
    coverage_metrics[2].metric(
        "Excluded",
        int((~coverage["Included in RMSD"]).sum()) if not coverage.empty else 0,
    )
    with st.expander("RMSD pose coverage", expanded=bool(warnings)):
        st.dataframe(coverage, hide_index=True, width="stretch")
    comparison_options = ["Recovery from input pose", "Prediction agreement"]
    recovery_pairs = (
        pairs.loc[pairs["Comparison type"].eq("Input-pose recovery")].copy()
        if not pairs.empty and "Comparison type" in pairs
        else pd.DataFrame()
    )
    comparison_mode = (
        "Recovery from input pose"
        if result_view == "Recovery to input"
        else "Prediction agreement"
        if result_view
        else st.segmented_control(
            "RMSD comparison",
            comparison_options,
            default=comparison_options[0],
            key="campaign_rmsd_comparison_mode",
            help=(
                "Recovery from input pose uses the ligand stored in the input "
                "complex as the fixed reference. Prediction agreement compares "
                "engines and repetitions with each other."
            ),
        )
    )
    if comparison_mode == "Recovery from input pose":
        if recovery_pairs.empty:
            st.info(
                "The input ligand could not be mapped to the selected compound. "
                "Review RMSD pose coverage and mapping warnings, or use "
                "Prediction agreement."
            )
            if warnings:
                with st.expander("RMSD exclusions and mapping warnings"):
                    for warning in warnings:
                        st.warning(warning)
            return
        recovery = recovery_pairs.rename(
            columns={
                "Pose B": "Prediction",
                "Engine B": "Engine",
                "Fixed-frame RMSD (Å)": "Input-pose RMSD (Å)",
                "Centroid distance (Å)": "Input-centroid displacement (Å)",
            }
        )
        recovery, recovery_matrix = _input_pose_recovery_matrix(recovery)
        st.caption(
            "The receptor frame is fixed by protein alignment. The ligand is "
            "not independently superposed: RMSD retains orientation and "
            "conformation differences, while centroid displacement isolates "
            "movement of the pose center from the input ligand. Because every "
            "prediction has the same input reference, this is an engine × "
            "attempt matrix rather than a symmetric pose × pose matrix. "
            "Colors use the fixed 0–4 Å scale: ≤1 Å is close, 1–2 Å is "
            "generally acceptable pose recovery, and values above 2 Å "
            "indicate increasing disagreement."
        )
        annotations = recovery_matrix.map(
            lambda value: f"{value:.3f}" if pd.notna(value) else ""
        )
        figure_width = max(7.0, 1.35 * len(recovery_matrix.columns) + 4.0)
        figure_height = max(4.0, 0.62 * len(recovery_matrix.index) + 1.8)
        figure, axis = plt.subplots(
            figsize=(figure_width, figure_height),
            constrained_layout=True,
        )
        sns.heatmap(
            recovery_matrix,
            annot=annotations,
            fmt="",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=4,
            linewidths=0.5,
            cbar_kws={"label": "RMSD to input pose (Å)"},
            ax=axis,
        )
        axis.set_title("Input-pose recovery by engine and attempt")
        axis.set_xlabel("Independent attempt")
        axis.set_ylabel("Engine")
        st.pyplot(figure, width="content")
        plt.close(figure)
        with st.expander("Input-pose recovery values"):
            st.dataframe(
                recovery.sort_values("Input-pose RMSD (Å)")[
                    [
                        "Engine",
                        "Attempt",
                        "Prediction",
                        "Input-pose RMSD (Å)",
                        "Input-centroid displacement (Å)",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
        if warnings:
            with st.expander("RMSD exclusions and mapping warnings"):
                for warning in warnings:
                    st.warning(warning)
        return

    agreement_matrix = matrix.drop(
        index=[reference_label], columns=[reference_label], errors="ignore"
    )
    agreement_pairs = (
        pairs.loc[pairs["Comparison type"].eq("Prediction agreement")].copy()
        if not pairs.empty and "Comparison type" in pairs
        else pairs
    )
    matrix = agreement_matrix
    pairs = agreement_pairs
    if len(comparable_predictions) < 2 or pairs.empty:
        st.info(
            "At least two chemically compatible displayed poses are needed "
            "for a pairwise comparison."
        )
    else:
        st.caption(
            f"The matrix lists all {matrix.shape[0]} selected poses. Numeric "
            f"cells cover {len(included_labels)} chemically comparable poses; "
            "blank rows or cells are retained for excluded or incompatible poses."
        )
        matrix_mode = (
            "Individual repetitions"
            if result_view == "Individual-pose matrix"
            else "Engine mean ± SD"
            if result_view == "Engine agreement matrix"
            else st.segmented_control(
                "RMSD matrix",
                ("Individual repetitions", "Engine mean ± SD"),
                default="Individual repetitions",
                key="campaign_viewer_rmsd_matrix_mode",
            )
        )
        aggregate_table = pd.DataFrame()
        if matrix_mode == "Engine mean ± SD":
            engine_order = list(
                dict.fromkeys(
                    pairs["Engine A"].tolist()
                    + pairs["Engine B"].tolist()
                )
            )
            plotted, annotations, aggregate_table = (
                _engine_pose_rmsd_summary(pairs, engine_order)
            )
            title = "Engine-level pose RMSD (mean ± sample SD)"
        else:
            order = matrix.index.tolist()
            complete = matrix.notna().all(axis=0).all()
            if complete and len(order) >= 3:
                try:
                    from scipy.cluster.hierarchy import (
                        leaves_list,
                        linkage,
                    )
                    from scipy.spatial.distance import squareform

                    condensed = squareform(
                        matrix.to_numpy(), checks=False
                    )
                    clustered = leaves_list(
                        linkage(condensed, method="average")
                    )
                    order = [order[int(index)] for index in clustered]
                except (ImportError, ValueError):
                    pass
            plotted = matrix.loc[order, order]
            annotations = True
            title = "Pairwise pose RMSD"
        size = max(5.8, min(8.5, 0.55 * len(plotted) + 3.0))
        figure, axis = plt.subplots(figsize=(size, size))
        sns.heatmap(
            plotted,
            annot=annotations,
            fmt=("" if matrix_mode == "Engine mean ± SD" else ".2f"),
            cmap="RdYlGn_r",
            vmin=0,
            vmax=4,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Inter-prediction fixed-frame RMSD (Å)"},
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_title(title)
        figure.tight_layout()
        st.pyplot(figure, width="content")
        plt.close(figure)

        if matrix_mode == "Engine mean ± SD":
            st.caption(
                "Each off-diagonal cell summarizes every compatible pose pair "
                "between the two engines. Diagonal cells summarize variation "
                "among repetitions from the same engine. SD is the sample "
                "standard deviation (n−1)."
            )
            st.dataframe(
                aggregate_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Mean RMSD (Å)": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                    "Sample SD (Å)": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                },
            )
        else:
            reference = st.selectbox(
                "Reference pose",
                matrix.index.tolist(),
                format_func=lambda value: value.replace("\n", " · "),
                key="campaign_viewer_rmsd_reference",
                help=(
                    "RMSD-to-reference changes only the summary table; it "
                    "never changes molecular coordinates or viewer alignment."
                ),
            )
            summary_rows: list[dict[str, object]] = []
            for label in matrix.index:
                other_values = matrix.loc[label].drop(
                    labels=[label], errors="ignore"
                ).dropna()
                summary_rows.append(
                    {
                        "Pose": label.replace("\n", " · "),
                        "RMSD to reference (Å)": matrix.loc[
                            label, reference
                        ],
                        "Mean pairwise RMSD (Å)": (
                            float(other_values.mean())
                            if not other_values.empty
                            else float("nan")
                        ),
                        "Maximum pairwise RMSD (Å)": (
                            float(other_values.max())
                            if not other_values.empty
                            else float("nan")
                        ),
                        "Comparable poses": int(len(other_values)),
                    }
                )
            st.dataframe(
                pd.DataFrame(summary_rows).sort_values(
                    "RMSD to reference (Å)",
                    na_position="last",
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "RMSD to reference (Å)":
                        st.column_config.NumberColumn(format="%.3f"),
                    "Mean pairwise RMSD (Å)":
                        st.column_config.NumberColumn(format="%.3f"),
                    "Maximum pairwise RMSD (Å)":
                        st.column_config.NumberColumn(format="%.3f"),
                },
            )
        with st.expander("Inspect every pose pair"):
            st.dataframe(
                pairs.sort_values("Fixed-frame RMSD (Å)"),
                hide_index=True,
                width="stretch",
                column_config={
                    "Fixed-frame RMSD (Å)": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                    "Centroid distance (Å)": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                },
            )
        st.caption(
            "Practical guide: below 2 Å usually indicates a closely reproduced "
            "pose, 2–4 Å a related but shifted or reoriented pose, and above "
            "4 Å a substantially different pose. These are review thresholds, "
            "not universal physical cutoffs."
        )
    if warnings:
        with st.expander(
            f"Pose comparisons not calculated ({len(warnings)})"
        ):
            for warning in warnings:
                st.write(f"- {warning}")


def _render_saved_collections(
    run_root: Path,
    *,
    selection: dict[str, object],
) -> None:
    st.caption(
        "Save the current scope as a first-class Analysis Set. Analysis Sets "
        "are reusable analysis definitions: they preserve campaign, target, "
        "engine and rescoring identifiers without pretending to be physical "
        "simulation campaigns or copying their scientific results."
    )
    form = st.form("campaign_comparison_collection_form")
    name = form.text_input(
        "Analysis Set name",
        placeholder="For example: 4LNW standard seven-engine comparison",
    )
    description = form.text_area(
        "Notes (optional)",
        placeholder=(
            "Purpose, scientific question, exclusions, or review status."
        ),
    )
    submitted = form.form_submit_button(
        "Save Analysis Set",
        type="primary",
    )
    if submitted:
        if not name.strip():
            st.error("Enter an Analysis Set name.")
        else:
            saved = _save_comparison_collection(
                run_root,
                name=name,
                description=description,
                selection=selection,
            )
            st.success(
                f"Saved {saved['name']} · {saved['job_code']}."
            )

    collections = _comparison_collections(run_root)
    st.markdown("#### Saved Analysis Sets")
    if not collections:
        st.info("No Analysis Sets have been saved yet.")
        return
    rows = pd.DataFrame(
        [
            {
                "open": row["open"],
                "name": row["name"],
                "description": row["description"],
                "dataset": row["dataset"] or row["dataset_run_id"],
                "targets": len(row["target_run_ids"]),
                "launch campaigns": len(row["launch_campaign_ids"]),
                "engines": ", ".join(row["engines"]),
                "engine runs": len(row["engine_run_ids"]),
                "rescoring runs": len(row["rescoring_run_ids"]),
                "created_at": row["created_at"],
                "code": row["job_code"],
            }
            for row in collections
        ]
    )
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "open": st.column_config.LinkColumn(
                "Open", display_text="Inspect"
            ),
            "created_at": st.column_config.DatetimeColumn(
                "Saved", format="YYYY-MM-DD HH:mm"
            ),
            "name": "Analysis Set",
            "description": "Notes",
            "code": "ID",
        },
    )


def _render_correlations(frame: pd.DataFrame) -> None:
    target_rows = frame[
        ["target_run_id", "target"]
    ].drop_duplicates()
    target_ids = target_rows["target_run_id"].tolist()
    target_labels = dict(
        zip(target_rows["target_run_id"], target_rows["target"])
    )
    target_id = st.selectbox(
        "Correlation target",
        target_ids,
        format_func=lambda value: target_labels.get(value, value),
        key="campaign_correlation_target",
        help=(
            "Correlations are calculated within one target to avoid mixing "
            "target effects with compound effects."
        ),
    )
    feature_table, labels, primary_defaults = correlation_feature_table(
        frame,
        target_run_id=target_id,
    )
    if feature_table.shape[1] < 2:
        st.info(
            "At least two metrics with five overlapping compounds are needed."
        )
        return
    metric_scope = st.segmented_control(
        "Metric set",
        ("Focused summary", "All available metrics", "Custom"),
        default="Focused summary",
        key="campaign_correlation_scope",
    )
    focused_features = list(
        dict.fromkeys(
            value
            for value in primary_defaults
            if value in feature_table.columns
        )
    )
    if metric_scope == "All available metrics":
        selected_features = list(feature_table.columns)
        st.caption(
            "Showing every sufficiently covered score, affinity, probability, "
            "confidence, structural-quality, and disagreement metric."
        )
    elif metric_scope == "Custom":
        selected_features = st.multiselect(
            "Metrics",
            list(feature_table.columns),
            default=(
                focused_features
                if len(focused_features) >= 2
                else list(feature_table.columns)
            ),
            format_func=lambda value: labels.get(value, value),
            key="campaign_correlation_metrics",
        )
    else:
        selected_features = (
            focused_features
            if len(focused_features) >= 2
            else list(feature_table.columns)
        )
        st.caption(
            "Focused summary: primary docking/energy scores, predicted IC50, "
            "GNINA CNN affinity, and the primary ranking output of "
            "structure-only engines. GNINA's classical docking score and "
            "Boltz-2/Nesso binder probabilities remain available under All "
            "available metrics or Custom."
        )
    method = st.segmented_control(
        "Correlation",
        ("Spearman", "Pearson"),
        default="Spearman",
        key="campaign_correlation_method",
        help=(
            "Spearman is the recommended default for rankings and differently "
            "scaled model outputs. Pearson is appropriate only when a linear "
            "relationship between comparable continuous values is plausible."
        ),
    )
    if len(selected_features) < 2:
        st.info("Select at least two metrics.")
        return
    selected_table = feature_table[selected_features]
    view_mode = st.segmented_control(
        "View",
        (
            "Correlation matrix",
            "Scatterplot",
            "Scatterplot matrix",
        ),
        default="Correlation matrix",
        key="campaign_correlation_view",
    )
    if view_mode == "Scatterplot matrix":
        matrix_metrics = st.multiselect(
            "Scatterplot-matrix metrics",
            selected_features,
            default=selected_features[:10],
            format_func=lambda value: labels.get(value, value),
            key="campaign_correlation_matrix_metrics",
            help=(
                "Choose two to ten metrics. Every row/column combination is "
                "drawn as a static compound-level scatterplot."
            ),
        )
        if len(matrix_metrics) < 2:
            st.info("Select at least two scatterplot-matrix metrics.")
            return
        if len(matrix_metrics) > 10:
            st.warning(
                "Select no more than ten metrics to keep the static matrix "
                "readable and responsive."
            )
            return
        display_labels = {
            metric: _scatter_matrix_display_label(
                labels.get(metric, metric)
            )
            for metric in matrix_metrics
        }
        matrix_frame = selected_table[matrix_metrics].rename(
            columns=display_labels
        )
        correlation_method = str(method or "Spearman").lower()

        def correlation_panel(x, y, **_kwargs) -> None:
            paired = pd.DataFrame({"x": x, "y": y}).dropna()
            sample_count = len(paired)
            coefficient = (
                paired["x"].corr(
                    paired["y"],
                    method=correlation_method,
                )
                if sample_count >= 2
                else math.nan
            )
            axis = plt.gca()
            coefficient_text = (
                f"{coefficient:.2f}"
                if math.isfinite(coefficient)
                else "—"
            )
            axis.annotate(
                coefficient_text,
                xy=(0.5, 0.57),
                xycoords=axis.transAxes,
                ha="center",
                va="center",
                fontsize=15,
                fontweight="semibold",
                color="#1f2937",
            )
            axis.annotate(
                f"n = {sample_count}",
                xy=(0.5, 0.38),
                xycoords=axis.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#64748b",
            )
            axis.set_axis_off()

        with sns.axes_style("whitegrid"), sns.plotting_context(
            "notebook",
            font_scale=0.78,
        ):
            pair_grid = sns.PairGrid(
                matrix_frame,
                height=1.85,
                aspect=1,
                diag_sharey=False,
                dropna=False,
            )
            pair_grid.map_lower(
                sns.regplot,
                scatter_kws={
                    "color": "#2563eb",
                    "s": 22,
                    "alpha": 0.68,
                    "edgecolor": "none",
                },
                line_kws={
                    "color": "#dc2626",
                    "linewidth": 1.5,
                },
                ci=95,
                truncate=False,
            )
            pair_grid.map_diag(
                sns.histplot,
                color="#60a5fa",
                edgecolor="white",
                bins="auto",
            )
            pair_grid.map_upper(correlation_panel)
            pair_grid.figure.subplots_adjust(
                left=0.09,
                bottom=0.09,
                right=0.99,
                top=0.99,
                wspace=0.08,
                hspace=0.08,
            )
            for axis in pair_grid.axes.flat:
                axis.tick_params(labelsize=7)
                axis.xaxis.label.set_size(8)
                axis.yaxis.label.set_size(8)
        st.pyplot(
            pair_grid.figure,
            clear_figure=True,
            use_container_width=True,
        )
        plt.close(pair_grid.figure)
        st.caption(
            "Statistical pair plot: diagonal panels show distributions, lower "
            "panels show compound-level scatterplots with an OLS trend line "
            "and 95% confidence band, and upper panels report "
            f"{str(method or 'Spearman')} correlation with pairwise n. Metrics "
            "are direction-normalized; missing values are handled pairwise. "
            "When Spearman is selected, the line remains an OLS visual guide "
            "while the reported coefficient is the rank correlation. "
            "Use the single Scatterplot view for compound labels, regression, "
            "and the paired table."
        )
        return
    if view_mode == "Scatterplot":
        metric_pairs: list[tuple[str, str, float, int]] = []
        for left_index, left_metric in enumerate(selected_features):
            for right_metric in selected_features[left_index + 1 :]:
                pair_values = selected_table[
                    [left_metric, right_metric]
                ].dropna()
                if len(pair_values) < 2:
                    continue
                coefficient = pair_values.corr(
                    method=str(method).lower()
                ).iloc[0, 1]
                metric_pairs.append(
                    (
                        left_metric,
                        right_metric,
                        (
                            float(coefficient)
                            if pd.notna(coefficient)
                            else float("nan")
                        ),
                        len(pair_values),
                    )
                )
        metric_pairs.sort(
            key=lambda value: (
                -abs(value[2]) if pd.notna(value[2]) else float("inf"),
                -value[3],
            )
        )
        if not metric_pairs:
            st.info("No metric pair has at least two overlapping compounds.")
            return
        pair_index = st.selectbox(
            "Metric pair",
            list(range(len(metric_pairs))),
            format_func=lambda value: (
                f"{labels.get(metric_pairs[int(value)][0], metric_pairs[int(value)][0])}"
                " ↔ "
                f"{labels.get(metric_pairs[int(value)][1], metric_pairs[int(value)][1])}"
                f" · r={metric_pairs[int(value)][2]:.2f}"
                f" · n={metric_pairs[int(value)][3]}"
            ),
            key="campaign_correlation_pair",
            help=(
                "All valid metric combinations are generated automatically and "
                "ordered by absolute correlation."
            ),
        )
        x_metric, y_metric, _, _ = metric_pairs[int(pair_index)]
        paired = (
            selected_table[[x_metric, y_metric]]
            .dropna()
            .reset_index()
            .rename(
                columns={
                    "index": "candidate_id",
                    x_metric: "X value",
                    y_metric: "Y value",
                }
            )
        )
        if len(paired) < 2:
            st.info("The selected metrics have fewer than two paired compounds.")
            return
        scatter_controls = st.columns(2)
        show_labels = scatter_controls[0].checkbox(
            "Show compound labels",
            value=False,
            key="campaign_correlation_scatter_labels",
        )
        show_trend = scatter_controls[1].checkbox(
            "Show linear trend",
            value=True,
            disabled=len(paired) < 3,
            key="campaign_correlation_scatter_trend",
            help=(
                "The line is an ordinary linear fit for visual orientation. "
                "It is distinct from the selected correlation statistic."
            ),
        )
        base = alt.Chart(paired).encode(
            x=alt.X(
                "X value:Q",
                title=labels.get(x_metric, x_metric),
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "Y value:Q",
                title=labels.get(y_metric, y_metric),
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("candidate_id:N", title="Compound"),
                alt.Tooltip("X value:Q", format=".5g"),
                alt.Tooltip("Y value:Q", format=".5g"),
            ],
        )
        points = base.mark_circle(
            size=105,
            opacity=0.82,
            color="#2563eb",
        )
        chart = points
        if show_trend and len(paired) >= 3:
            chart += base.transform_regression(
                "X value", "Y value"
            ).mark_line(color="#475569", strokeWidth=2)
        if show_labels:
            chart += base.mark_text(
                align="left",
                baseline="middle",
                dx=7,
                fontSize=11,
            ).encode(text="candidate_id:N")
        st.altair_chart(
            chart.properties(height=480),
            width="stretch",
        )
        coefficient = paired[["X value", "Y value"]].corr(
            method=str(method).lower()
        ).iloc[0, 1]
        statistic_columns = st.columns(2)
        statistic_columns[0].metric(
            f"{method} correlation",
            (
                f"{float(coefficient):.3f}"
                if pd.notna(coefficient)
                else "—"
            ),
        )
        statistic_columns[1].metric(
            "Paired compounds",
            len(paired),
        )
        st.caption(
            "Lower-is-better metrics are sign-inverted before plotting, so both "
            "axes point toward more favorable predictions. IC50 is displayed in "
            "µM; repetition-level concentrations are aggregated geometrically."
        )
        st.dataframe(
            paired,
            hide_index=True,
            width="stretch",
        )
        return
    correlation, overlap = correlation_matrices(
        selected_table,
        method=str(method),
    )
    label_order = [labels[value] for value in selected_features]
    correlation = correlation.rename(
        index=labels, columns=labels
    ).reindex(index=label_order, columns=label_order)
    overlap = overlap.rename(
        index=labels, columns=labels
    ).reindex(index=label_order, columns=label_order)
    long_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for row_index, row_label in enumerate(label_order):
        for column_index, column_label in enumerate(label_order):
            value = correlation.loc[row_label, column_label]
            count = int(overlap.loc[row_label, column_label])
            long_rows.append(
                {
                    "Metric A": row_label,
                    "Metric B": column_label,
                    "Correlation": value,
                    "Compounds": count,
                    "Label": (
                        f"{float(value):.2f}"
                        if pd.notna(value)
                        else "—"
                    ),
                }
            )
            if column_index > row_index and pd.notna(value):
                pair_rows.append(
                    {
                        "Metric A": row_label,
                        "Metric B": column_label,
                        "Correlation": float(value),
                        "|Correlation|": abs(float(value)),
                        "Compounds": count,
                    }
                )
    long_frame = pd.DataFrame(long_rows)
    base = alt.Chart(long_frame).encode(
        x=alt.X(
            "Metric B:N",
            sort=label_order,
            title=None,
            axis=alt.Axis(labelAngle=-35, labelLimit=220),
        ),
        y=alt.Y(
            "Metric A:N",
            sort=label_order,
            title=None,
            axis=alt.Axis(labelLimit=260),
        ),
        tooltip=[
            "Metric A:N",
            "Metric B:N",
            alt.Tooltip("Correlation:Q", format=".3f"),
            "Compounds:Q",
        ],
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "Correlation:Q",
            scale=alt.Scale(
                domain=[-1, 0, 1],
                range=["#b91c1c", "#f8fafc", "#1d4ed8"],
            ),
            title=f"{method} r",
        )
    )
    text = base.mark_text(fontSize=11).encode(
        text="Label:N",
        color=alt.condition(
            "abs(datum.Correlation) > 0.55",
            alt.value("white"),
            alt.value("#111827"),
        ),
    )
    st.altair_chart(
        (heatmap + text).properties(
            height=max(390, 52 * len(label_order))
        ),
        width="stretch",
    )
    st.caption(
        "Every metric is first averaged across repetitions and campaigns for "
        "each compound. Metrics where lower is more favorable are sign-inverted, "
        "so positive correlation consistently means agreement about favorable "
        "compounds. Cells require at least five overlapping compounds."
    )
    if pair_rows:
        strongest = pd.DataFrame(pair_rows).sort_values(
            ["|Correlation|", "Compounds"],
            ascending=[False, False],
        )
        st.markdown("#### Strongest observed relationships")
        st.dataframe(
            strongest,
            hide_index=True,
            width="stretch",
        )
    with st.expander("Pairwise compound counts"):
        st.dataframe(overlap, width="stretch")


def _target_feature_controls(
    feature_table: pd.DataFrame,
    labels: dict[str, str],
    defaults: list[str],
    *,
    key: str,
) -> list[str]:
    options = list(feature_table.columns)
    initial = [value for value in defaults if value in options]
    if len(initial) < 2:
        initial = options
    return st.multiselect(
        "Engine metrics",
        options,
        default=initial,
        format_func=lambda value: labels.get(value, value),
        key=key,
        help=(
            "Primary native metrics are selected by default. Ligand RMSD is "
            "also included when emitted. Lower-is-better values are inverted "
            "internally so favorable values always point upward."
        ),
    )


def _render_target_correlations(frame: pd.DataFrame) -> None:
    feature_table, metadata, _, labels, defaults = (
        target_comparison_feature_table(frame)
    )
    if feature_table.shape[0] < 3 or feature_table.shape[1] < 2:
        st.info(
            "At least three prepared targets and two overlapping engine "
            "metrics are needed for target-preparation correlation."
        )
        return
    selected_features = _target_feature_controls(
        feature_table,
        labels,
        defaults,
        key="target_campaign_correlation_metrics",
    )
    if len(selected_features) < 2:
        st.info("Select at least two engine metrics.")
        return
    method = st.segmented_control(
        "Correlation",
        ("Spearman", "Pearson"),
        default="Spearman",
        key="target_campaign_correlation_method",
    )
    selected = feature_table[selected_features]
    correlation, overlap = correlation_matrices(
        selected,
        method=str(method),
        minimum_overlap=3,
    )
    display_labels = {
        feature: _scatter_matrix_display_label(labels.get(feature, feature))
        for feature in selected_features
    }
    correlation = correlation.rename(
        index=display_labels,
        columns=display_labels,
    )
    overlap = overlap.rename(index=display_labels, columns=display_labels)
    long = (
        correlation.rename_axis("Metric A")
        .reset_index()
        .melt(
            id_vars="Metric A",
            var_name="Metric B",
            value_name="Correlation",
        )
    )
    st.caption(
        "Each observation is one prepared-target campaign after averaging its "
        "independent attempts. Replicas quantify uncertainty and do not inflate "
        "the correlation sample size."
    )
    if len(feature_table) < 5:
        st.warning(
            f"Only {len(feature_table)} prepared targets overlap. Treat these "
            "correlations as exploratory; five or more are preferable."
        )
    heatmap = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X("Metric B:N", title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("Metric A:N", title=None),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(
                    domain=[-1, 0, 1],
                    range=["#b91c1c", "#f8fafc", "#1d4ed8"],
                ),
            ),
            tooltip=[
                "Metric A:N",
                "Metric B:N",
                alt.Tooltip("Correlation:Q", format=".3f"),
            ],
        )
        .properties(height=max(360, 48 * len(selected_features)))
    )
    st.altair_chart(heatmap, width="stretch")
    selectors = st.columns(2)
    x_feature = selectors[0].selectbox(
        "X metric",
        selected_features,
        format_func=lambda value: labels.get(value, value),
        key="target_campaign_correlation_x",
    )
    y_options = [
        feature for feature in selected_features if feature != x_feature
    ]
    y_feature = selectors[1].selectbox(
        "Y metric",
        y_options,
        format_func=lambda value: labels.get(value, value),
        key="target_campaign_correlation_y",
    )
    paired = (
        selected[[x_feature, y_feature]]
        .dropna()
        .rename(columns={x_feature: "X value", y_feature: "Y value"})
        .rename_axis("target_comparison_id")
        .reset_index()
        .merge(metadata, on="target_comparison_id", how="left")
    )
    base = alt.Chart(paired).encode(
        x=alt.X(
            "X value:Q",
            title=labels.get(x_feature, x_feature),
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y(
            "Y value:Q",
            title=labels.get(y_feature, y_feature),
            scale=alt.Scale(zero=False),
        ),
        tooltip=[
            alt.Tooltip("target_preparation:N", title="Target preparation"),
            alt.Tooltip("target:N", title="Target"),
            alt.Tooltip("X value:Q", format=".5g"),
            alt.Tooltip("Y value:Q", format=".5g"),
        ],
    )
    points = base.mark_circle(size=125, opacity=0.85, color="#7c3aed")
    point_labels = base.mark_text(
        align="left",
        baseline="middle",
        dx=8,
        fontSize=11,
    ).encode(text="target_preparation:N")
    st.altair_chart(
        (points + point_labels).properties(height=480),
        width="stretch",
    )
    coefficient = paired[["X value", "Y value"]].corr(
        method=str(method).lower()
    ).iloc[0, 1]
    statistics = st.columns(2)
    statistics[0].metric(
        f"{method} correlation",
        f"{float(coefficient):.3f}" if pd.notna(coefficient) else "—",
    )
    statistics[1].metric("Prepared targets", len(paired))
    with st.expander("Metric overlap"):
        st.dataframe(overlap, width="stretch")


def _render_target_consensus(frame: pd.DataFrame) -> None:
    feature_table, metadata, summaries, labels, defaults = (
        target_comparison_feature_table(frame)
    )
    if feature_table.empty:
        st.info("No sufficiently covered target-level metrics are available.")
        return
    selected_features = _target_feature_controls(
        feature_table,
        labels,
        defaults,
        key="target_campaign_consensus_metrics",
    )
    combined, long = target_consensus_summary(
        feature_table,
        selected_features,
    )
    if combined.empty:
        st.info("Select at least one target-level metric.")
        return
    combined = combined.merge(
        metadata,
        on="target_comparison_id",
        how="left",
    )
    combined["lower"] = (
        combined["Mean percentile"] - combined["Sample SD"]
    ).clip(lower=0)
    combined["upper"] = (
        combined["Mean percentile"] + combined["Sample SD"]
    ).clip(upper=1)
    order = combined["target_preparation"].astype(str).tolist()
    base = alt.Chart(combined).encode(
        x=alt.X(
            "target_preparation:N",
            sort=order,
            title="Prepared-target campaign",
            axis=alt.Axis(labelAngle=-35, labelLimit=220),
        ),
        tooltip=[
            alt.Tooltip("target_preparation:N", title="Target preparation"),
            alt.Tooltip("Mean percentile:Q", format=".3f"),
            alt.Tooltip("Sample SD:Q", format=".3f"),
            "Contributing metrics:Q",
        ],
    )
    bars = base.mark_bar(color="#7c3aed", opacity=0.82).encode(
        y=alt.Y(
            "Mean percentile:Q",
            scale=alt.Scale(domain=[0, 1]),
            title="Mean within-metric percentile",
        )
    )
    errors = base.mark_rule(strokeWidth=2, color="#111827").encode(
        y=alt.Y("lower:Q"),
        y2="upper:Q",
    )
    st.altair_chart((bars + errors).properties(height=420), width="stretch")
    profile = long.merge(
        metadata[["target_comparison_id", "target_preparation"]],
        on="target_comparison_id",
        how="left",
    )
    profile["Metric"] = profile["feature_id"].map(labels)
    heatmap = (
        alt.Chart(profile)
        .mark_rect()
        .encode(
            x=alt.X(
                "Metric:N",
                title="Engine metric",
                axis=alt.Axis(labelAngle=-35, labelLimit=220),
            ),
            y=alt.Y(
                "target_preparation:N",
                sort=order,
                title="Prepared-target campaign",
            ),
            color=alt.Color(
                "Within-feature percentile:Q",
                scale=alt.Scale(domain=[0, 1], scheme="viridis"),
                title="Percentile",
            ),
            tooltip=[
                "target_preparation:N",
                "Metric:N",
                alt.Tooltip("Within-feature percentile:Q", format=".3f"),
            ],
        )
        .properties(height=max(280, 38 * len(order)))
    )
    st.markdown("#### Target-by-engine profile")
    st.altair_chart(heatmap, width="stretch")
    st.caption(
        "Consensus combines direction-aware within-metric ranks, not raw units. "
        "It is a preparation-robustness aid, not a calibrated affinity."
    )
    with st.expander("Target-level means and replica variability"):
        st.dataframe(
            summaries.loc[summaries["feature_id"].isin(selected_features)],
            hide_index=True,
            width="stretch",
        )


def target_pose_validation_summary(
    poses: pd.DataFrame,
    selected_jobs: pd.DataFrame,
) -> pd.DataFrame:
    if poses.empty or selected_jobs.empty:
        return pd.DataFrame()
    columns = [
        "campaign_id",
        "launch_campaign_id",
        "launch_campaign",
        "target_run_id",
        "target",
    ]
    lookup = selected_jobs[
        [column for column in columns if column in selected_jobs]
    ].copy()
    lookup = lookup.rename(columns={"campaign_id": "source_run_id"})
    merged = poses.merge(
        lookup,
        on="source_run_id",
        how="left",
        suffixes=("", "_job"),
    )
    merged = _pose_validation_display_rows(merged)
    target_run_id = (
        merged["target_run_id_job"]
        if "target_run_id_job" in merged
        else merged["target_run_id"]
        if "target_run_id" in merged
        else pd.Series("", index=merged.index, dtype=str)
    ).fillna("").astype(str)
    launch_campaign_id = merged["launch_campaign_id"].fillna("").astype(str)
    merged["target_comparison_id"] = (
        launch_campaign_id.where(
            launch_campaign_id.str.strip().ne(""),
            target_run_id,
        )
        + "::"
        + target_run_id
    )
    target_label = (
        merged["target_job"]
        if "target_job" in merged
        else merged["target"]
        if "target" in merged
        else pd.Series("Prepared target", index=merged.index, dtype=str)
    ).fillna("Prepared target").astype(str)
    merged["target_preparation"] = target_label
    return (
        merged.groupby(
            [
                "target_comparison_id",
                "target_preparation",
                "validation_group",
            ],
            as_index=False,
        )["passed_all"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "pose_pass_rate",
                "sum": "passing_poses",
                "count": "assessed_poses",
            }
        )
    )


def _render_target_pose_validation(
    run_root: Path,
    selected_jobs: pd.DataFrame,
) -> None:
    structural = selected_jobs.loc[
        selected_jobs["engine"].isin(STRUCTURE_ENGINE_ORDER)
    ].copy()
    poses, provenance, legacy_count = _pose_validation_rows(
        run_root,
        structural,
    )
    if legacy_count:
        st.warning(
            f"{legacy_count} legacy validation run(s) are excluded because "
            "they do not use the current selection policy."
        )
    summary = target_pose_validation_summary(poses, structural)
    if summary.empty:
        st.info(
            "No scientifically usable PoseBusters rows are linked to the "
            "selected prepared-target campaigns."
        )
        if not provenance.empty:
            st.dataframe(provenance, hide_index=True, width="stretch")
        return
    target_order = list(
        dict.fromkeys(summary["target_preparation"].astype(str).tolist())
    )
    validation_order = list(
        dict.fromkeys(summary["validation_group"].astype(str).tolist())
    )
    heatmap = (
        alt.Chart(summary)
        .mark_rect(stroke="white")
        .encode(
            x=alt.X(
                "validation_group:N",
                sort=validation_order,
                title="Engine / pose selection",
                axis=alt.Axis(labelAngle=-30),
            ),
            y=alt.Y(
                "target_preparation:N",
                sort=target_order,
                title="Prepared-target campaign",
            ),
            color=alt.Color(
                "pose_pass_rate:Q",
                scale=alt.Scale(domain=[0, 1], scheme="redyellowgreen"),
                title="Pose pass rate",
            ),
            tooltip=[
                "target_preparation:N",
                "validation_group:N",
                alt.Tooltip("pose_pass_rate:Q", format=".1%"),
                "passing_poses:Q",
                "assessed_poses:Q",
            ],
        )
        .properties(height=max(300, 42 * len(target_order)))
    )
    st.caption(
        "Rows are prepared-target campaigns and columns are engine pose "
        "selections. Pass rates summarize attempts without treating them as "
        "independent target observations."
    )
    st.altair_chart(heatmap, width="stretch")
    st.dataframe(summary, hide_index=True, width="stretch")


def _render_linked_3d_focus(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Render the target and compound scope shared by RMSD and 3D."""
    structural_scope = frame.loc[
        frame["_structure_path"].astype(str).str.strip().ne("")
    ].copy()
    target_rows = structural_scope[
        ["target_run_id", "target", "launch_campaign"]
    ].drop_duplicates("target_run_id")
    target_ids = target_rows["target_run_id"].astype(str).tolist()
    target_labels = {
        str(row["target_run_id"]): (
            str(row.get("target") or "").strip()
            or str(row.get("launch_campaign") or "").strip()
            or str(row["target_run_id"])
        )
        for _, row in target_rows.iterrows()
    }
    compounds = list(
        dict.fromkeys(structural_scope["candidate_id"].astype(str).tolist())
    )
    compound_labels = _compound_selector_labels(structural_scope)
    target_key = "campaign_linked_3d_focus_targets"
    compound_key = "campaign_linked_3d_focus_compounds"
    scope_context = (tuple(target_ids), tuple(compounds))
    if st.session_state.get("_campaign_structural_scope_context") != scope_context:
        st.session_state["_campaign_structural_scope_context"] = scope_context
        st.session_state[target_key] = list(target_ids)
        st.session_state[compound_key] = list(compounds)
    st.markdown("#### Structures included in both views")
    st.caption(
        "This is the shared scope for the RMSD matrices and 3D structures. "
        "Remove a chip to hide that target or compound in both views. Engines "
        "can be hidden independently inside the 3D tab."
    )
    focus_columns = st.columns(2)
    selected_targets = focus_columns[0].multiselect(
        "Prepared targets",
        target_ids,
        format_func=lambda value: target_labels.get(str(value), str(value)),
        key=target_key,
    )
    selected_compounds = focus_columns[1].multiselect(
        "Compounds",
        compounds,
        format_func=lambda value: compound_labels.get(str(value), str(value)),
        key=compound_key,
    )
    focused = structural_scope
    focused = focused.loc[
        focused["target_run_id"].astype(str).isin(
            [str(value) for value in selected_targets]
        )
        & focused["candidate_id"].astype(str).isin(
            [str(value) for value in selected_compounds]
        )
    ]
    st.caption(
        f"Active structural scope: {len(selected_targets)} target(s) · "
        f"{len(selected_compounds)} compound(s)."
    )
    return focused


def _render_target_viewer_context(
    frame: pd.DataFrame,
    *,
    gnina_criterion_label: str = "",
    available_frame: pd.DataFrame | None = None,
) -> None:
    criterion_label = gnina_criterion_label or "Both rankings"
    available_structural = (
        available_frame
        if available_frame is not None
        else frame
    )
    available_structural = available_structural.loc[
        available_structural["_structure_path"].astype(str).str.strip().ne("")
    ]
    available_structural = _select_gnina_ranking_rows(
        available_structural,
        criterion_label,
    )
    available_engines = list(
        dict.fromkeys(available_structural["engine"].astype(str).tolist())
    )
    local_engine_key = "campaign_3d_engine_selector"
    engine_context_key = "_campaign_3d_engine_selector_options"
    engine_context = tuple(available_engines)
    if st.session_state.get(engine_context_key) != engine_context:
        previous_engines = list(
            st.session_state.get(local_engine_key, available_engines)
        )
        gnina_was_selected = any(
            str(engine) == "GNINA"
            or str(engine).startswith("GNINA · ")
            for engine in previous_engines
        )
        migrated_engines = [
            engine
            for engine in previous_engines
            if engine in available_engines
            and not str(engine).startswith("GNINA · ")
        ]
        if gnina_was_selected:
            migrated_engines.extend(
                engine
                for engine in available_engines
                if str(engine).startswith("GNINA · ")
            )
        st.session_state[local_engine_key] = list(
            dict.fromkeys(migrated_engines)
        )
        st.session_state[engine_context_key] = engine_context
    selected_3d_engines = [
        engine
        for engine in st.session_state.get(local_engine_key, available_engines)
        if engine in available_engines
    ]
    if st.session_state.get(local_engine_key) != selected_3d_engines:
        st.session_state[local_engine_key] = selected_3d_engines

    st.markdown("#### Engines displayed")
    selected_3d_engines = st.multiselect(
        "Engines shown in 3D",
        available_engines,
        key=local_engine_key,
        help=(
            "Hide structures from selected engines in this 3D view. "
            "The RMSD matrices remain unchanged."
        ),
    )
    structural = frame.loc[
        frame["_structure_path"].astype(str).str.strip().ne("")
    ].copy()
    structural = _select_gnina_ranking_rows(structural, criterion_label)
    structural = structural.loc[
        structural["engine"].astype(str).isin(
            [str(value) for value in selected_3d_engines]
        )
    ].copy()
    if structural.empty:
        st.info("Select at least one engine to display structures.")
        return
    if not structural.empty:
        coverage = (
            structural.groupby(
                ["target", "engine"],
                dropna=False,
            )
            .size()
            .reset_index(name="Stored predictions")
        )
        target_count = structural["target"].nunique()
        engine_count = structural["engine"].nunique()
        expected_cells = target_count * engine_count
        observed_cells = len(coverage)
        missing_cells = max(0, expected_cells - observed_cells)
        availability = (
            "complete"
            if missing_cells == 0
            else f"{missing_cells} target–engine combination(s) missing"
        )
        st.caption(
            f"Structure availability: {target_count} prepared target(s) · "
            f"{engine_count} engine(s) · {len(structural)} stored prediction(s) "
            f"· {availability}."
        )
        with st.expander(
            "Structure availability details",
            expanded=missing_cells > 0,
        ):
            coverage_matrix = coverage.pivot(
                index="target",
                columns="engine",
                values="Stored predictions",
            ).fillna(0).astype(int)
            st.dataframe(coverage_matrix, width="stretch")
    rmsd_presentation = str(
        st.session_state.get(
            "campaign_rmsd_target_presentation",
            "All targets (separate)",
        )
    )
    if rmsd_presentation != "Single-target drill-down":
        st.caption(
            "This is the structural counterpart of the all-target or combined "
            "RMSD view. Each panel is aligned to its own prepared input; camera "
            "rotation and zoom are linked. The combined statistical matrix "
            "does not imply a cross-target structural superposition."
        )
        _render_structure_comparison(
            structural,
            forced_layout="Target matrix",
            gnina_criterion_label=gnina_criterion_label,
        )
    else:
        st.caption(
            "Choose one prepared target for a coordinate-correct overlay of "
            "its engines and repetitions."
        )
        _render_structure_comparison(
            structural,
            forced_layout="Single compound",
            gnina_criterion_label=gnina_criterion_label,
        )


def _render_rescoring_comparison(frame: pd.DataFrame) -> None:
    rescoring = frame.loc[
        frame["engine"].astype(str).str.endswith("rescoring")
    ].copy()
    if rescoring.empty:
        st.info("No rescoring campaigns are selected.")
        return
    engines = sorted(rescoring["engine"].unique())
    engine = st.selectbox(
        "Rescoring engine",
        engines,
        key="campaign_rescoring_engine",
    )
    engine_rows = rescoring.loc[rescoring["engine"].eq(engine)].copy()
    run_rows = engine_rows[
        ["campaign_id", "engine_run"]
    ].drop_duplicates()
    run_options = run_rows["campaign_id"].tolist()
    run_labels = dict(
        zip(run_rows["campaign_id"], run_rows["engine_run"])
    )
    selected_runs = st.multiselect(
        "Rescoring runs",
        run_options,
        default=run_options,
        format_func=lambda value: run_labels.get(value, value),
        key="campaign_compare_rescoring_runs",
        help=(
            "Rescoring remains linked to the selected source launch campaign "
            "but is never pooled as another independent docking engine."
        ),
    )
    engine_rows = engine_rows.loc[
        engine_rows["campaign_id"].isin(selected_runs)
    ].copy()
    available = [
        definition
        for definition in ENGINE_METRICS[engine]
        if definition[0] in engine_rows
        and pd.to_numeric(
            engine_rows[definition[0]], errors="coerce"
        ).notna().any()
    ]
    if not available or "source_score_kcal_mol" not in engine_rows:
        st.info("This rescoring campaign has no paired source-score data.")
        return
    metric_labels = {
        metric: (label, higher)
        for metric, label, higher in available
    }
    metric = st.selectbox(
        "Rescored output",
        list(metric_labels),
        format_func=lambda value: metric_labels[value][0],
        key="campaign_rescoring_metric",
    )
    paired = engine_rows[
        [
            column
            for column in (
                "candidate_id",
                "pose_id",
                "replicate",
                "source_engine",
                "source_score_kcal_mol",
                metric,
                "campaign",
            )
            if column in engine_rows
        ]
    ].copy()
    paired["Original docking score"] = pd.to_numeric(
        paired["source_score_kcal_mol"], errors="coerce"
    )
    paired["Rescored value"] = pd.to_numeric(
        paired[metric], errors="coerce"
    )
    paired = paired.dropna(
        subset=["Original docking score", "Rescored value"]
    )
    if paired.empty:
        st.info("No complete original/rescored pairs are available.")
        return
    compound_pairs = (
        paired.groupby(
            ["candidate_id", "source_engine"], dropna=False
        )[["Original docking score", "Rescored value"]]
        .mean()
        .reset_index()
    )
    point_chart = (
        alt.Chart(compound_pairs)
        .mark_circle(size=95, opacity=0.8)
        .encode(
            x=alt.X(
                "Original docking score:Q",
                title="Original docking score (kcal/mol)",
            ),
            y=alt.Y(
                "Rescored value:Q",
                title=metric_labels[metric][0],
            ),
            color=alt.Color(
                "source_engine:N", title="Source engine"
            ),
            tooltip=[
                alt.Tooltip("candidate_id:N", title="Compound"),
                "source_engine:N",
                alt.Tooltip(
                    "Original docking score:Q", format=".4f"
                ),
                alt.Tooltip("Rescored value:Q", format=".4f"),
            ],
        )
        .properties(height=420)
    )
    if len(compound_pairs) >= 3:
        trend = point_chart.transform_regression(
            "Original docking score", "Rescored value"
        ).mark_line(color="#475569")
        chart = point_chart + trend
    else:
        chart = point_chart
    st.altair_chart(chart, width="stretch")
    if len(compound_pairs) >= 3:
        spearman = compound_pairs[
            ["Original docking score", "Rescored value"]
        ].corr(method="spearman").iloc[0, 1]
        st.metric(
            "Paired Spearman correlation",
            f"{float(spearman):.3f}",
            help=(
                "Association between the source docking score and this "
                "rescoring output for the same compounds."
            ),
        )
    else:
        st.caption(
            "At least three paired compounds are needed for a correlation."
        )
    st.caption(
        "Rescoring outputs are analyzed only within the selected rescoring "
        "engine and paired to their recorded source docking scores. They do "
        "not contribute an extra vote to the general cross-engine consensus."
    )
    st.dataframe(
        paired,
        hide_index=True,
        width="stretch",
    )


def _load_comparison_structures(
    rows: pd.DataFrame,
    reference_path: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    rendered: list[dict[str, object]] = []
    warnings: list[str] = []
    for _, row in rows.iterrows():
        original_structure_path = Path(str(row["_structure_path"]))
        kind = str(row["_structure_kind"])
        structure_path = _preferred_viewer_structure_path(
            original_structure_path,
            structure_kind=kind,
        )
        if not structure_path.is_file():
            warnings.append(
                f"{row['_prediction_label']}: structure file is unavailable"
            )
            continue
        if kind == "complex":
            try:
                structure_data, rmsd, matched = aligned_structure_data(
                    str(reference_path),
                    reference_path.stat().st_mtime_ns,
                    str(structure_path),
                    structure_path.stat().st_mtime_ns,
                )
            except Exception as exc:
                warnings.append(
                    f"{row['_prediction_label']}: alignment failed ({exc})"
                )
                continue
            rendered.append(
                {
                    "row": row,
                    "data": structure_data,
                    "format": "cif",
                    "kind": kind,
                    "alignment_rmsd": rmsd,
                    "matched_atoms": matched,
                    "source_path": structure_path,
                    "coordinate_frame": "protein Cα aligned",
                }
            )
        else:
            try:
                data, model_format = _model_text(
                    structure_path,
                    int(row.get("_viewer_pose_index") or 1),
                )
            except (OSError, ValueError) as exc:
                warnings.append(
                    f"{row['_prediction_label']}: loading failed ({exc})"
                )
                continue
            rendered.append(
                {
                    "row": row,
                    "data": data,
                    "format": model_format,
                    "kind": kind,
                    "alignment_rmsd": None,
                    "matched_atoms": None,
                    "source_path": structure_path,
                    "coordinate_frame": "native docking target frame",
                }
            )
    return rendered, warnings


def _render_target_matrix(structural: pd.DataFrame) -> None:
    candidates = sorted(structural["candidate_id"].astype(str).unique())
    compound_labels = _compound_selector_labels(structural)
    candidate = st.selectbox(
        "Focus compound",
        candidates,
        format_func=lambda value: compound_labels.get(str(value), str(value)),
        key="campaign_viewer_target_matrix_compound",
        help="The same compound is shown across every prepared-target panel.",
    )
    candidate_rows = structural.loc[
        structural["candidate_id"].astype(str).eq(str(candidate))
    ].copy()
    target_rows = (
        candidate_rows[
            [
                "target_run_id",
                "target",
                "launch_campaign",
            ]
        ]
        .drop_duplicates("target_run_id")
        .reset_index(drop=True)
    )
    target_ids = target_rows["target_run_id"].astype(str).tolist()
    target_labels = {
        str(row["target_run_id"]): (
            str(row.get("target") or "").strip()
            or str(row.get("launch_campaign") or "").strip()
            or str(row["target_run_id"])
        )
        for _, row in target_rows.iterrows()
    }
    selected_targets = target_ids
    matrix_rows = candidate_rows.loc[
        candidate_rows["target_run_id"].astype(str).isin(
            [str(value) for value in selected_targets]
        )
    ].copy()

    matrix_modes = (
        "Representative per campaign",
        "Best per engine",
        "All repetitions",
    )
    inherited_mode = str(
        st.session_state.get(
            "campaign_rmsd_mode", "Representative per campaign"
        )
    )
    default_matrix_mode = (
        inherited_mode
        if inherited_mode in matrix_modes
        else "Representative per campaign"
    )
    summary_mode = default_matrix_mode
    with st.expander("Panel display settings"):
        st.caption(f"Poses inherited from RMSD view: {summary_mode}.")
        settings = st.columns(3)
        grid_columns = int(
            settings[0].number_input(
                "Target grid columns",
                min_value=1,
                max_value=4,
                value=min(3, len(selected_targets)),
                key="campaign_viewer_target_matrix_columns",
            )
        )
        show_target = settings[1].checkbox(
            "Show target proteins",
            value=True,
            key="campaign_viewer_target_matrix_show_target",
        )
        show_reference_ligand = settings[2].checkbox(
            "Show input ligands",
            value=True,
            key="campaign_viewer_target_matrix_show_reference_ligand",
        )
        show_predicted_proteins = st.checkbox(
            "Show predicted proteins",
            value=False,
            key="campaign_viewer_target_matrix_show_predicted_proteins",
            help=(
                "Predicted proteins are hidden by default so ligand pose "
                "differences remain readable."
            ),
        )
    if summary_mode == "All repetitions" and len(matrix_rows) > 60:
        st.warning(
            f"This view will load {len(matrix_rows)} structure models across "
            f"{len(selected_targets)} linked panels. Use the shared 3D focus "
            "to restrict targets or an engine pair if interaction becomes slow."
        )

    available_engines = set(matrix_rows["engine"].astype(str))
    engine_order = [
        engine
        for engine in STRUCTURE_ENGINE_ORDER
        if engine in available_engines
    ] + sorted(available_engines - set(STRUCTURE_ENGINE_ORDER))
    engine_styles = {
        engine: STRUCTURE_ENGINE_STYLES.get(
            engine,
            STRUCTURE_ENGINE_PALETTE[
                (len(STRUCTURE_ENGINE_ORDER) + index)
                % len(STRUCTURE_ENGINE_PALETTE)
            ],
        )
        for index, engine in enumerate(engine_order)
    }

    panels: list[dict[str, object]] = []
    warnings: list[str] = []
    legend_rows: list[dict[str, object]] = []
    for target_id in [str(value) for value in selected_targets]:
        target_frame = matrix_rows.loc[
            matrix_rows["target_run_id"].astype(str).eq(target_id)
        ].copy()
        if target_frame.empty:
            continue
        displayed = (
            _best_structure_rows(target_frame)
            if summary_mode == "Best per engine"
            else target_frame
            if summary_mode == "All repetitions"
            else _representative_structure_rows(target_frame)
        )
        displayed = _ordered_structure_rows(displayed)
        target_path = Path(
            str(displayed.iloc[0].get("_target_path") or "")
        )
        reference_complex = str(
            displayed.iloc[0].get("_reference_complex_path") or ""
        )
        reference_path = (
            Path(reference_complex)
            if reference_complex and Path(reference_complex).is_file()
            else target_path
        )
        if not target_path.is_file() or not reference_path.is_file():
            warnings.append(
                f"{target_labels.get(target_id, target_id)}: prepared input "
                "target is unavailable"
            )
            continue
        rendered, target_warnings = _load_comparison_structures(
            displayed,
            reference_path,
        )
        warnings.extend(
            f"{target_labels.get(target_id, target_id)}: {message}"
            for message in target_warnings
        )
        if not rendered:
            continue
        reference_data, reference_format = _model_text(reference_path)
        panels.append(
            {
                "panel": len(panels) + 1,
                "target_id": target_id,
                "label": target_labels.get(target_id, target_id),
                "target": str(displayed.iloc[0].get("target") or ""),
                "reference_data": reference_data,
                "reference_format": reference_format,
                "rendered": rendered,
            }
        )
        for item in rendered:
            row = item["row"]
            _, engine_color = engine_styles[str(row["engine"])]
            legend_rows.append(
                {
                    "target_preparation": target_labels.get(
                        target_id, target_id
                    ),
                    "target_job": target_id,
                    "color": engine_color,
                    "engine": row["engine"],
                    "campaign": row["campaign"],
                    "prediction": row["_prediction_label"],
                    "protein_alignment_rmsd_angstrom": item[
                        "alignment_rmsd"
                    ],
                    "matched_ca_atoms": item["matched_atoms"],
                    "coordinate_frame": item["coordinate_frame"],
                    "structure_file": str(item["source_path"]),
                }
            )
    if not panels:
        st.info("No selected target structures could be loaded.")
        for message in dict.fromkeys(warnings):
            st.warning(message)
        return

    st.markdown("#### Structure color legend")
    st.caption(
        "Engine identity is encoded by ligand carbon color; heteroatoms keep "
        "their standard element colors."
    )
    legend_items = [
        ("Prepared target", "#cbd5e1"),
        ("Input/reference ligand", "green"),
        *[
            (engine, engine_styles[engine][1])
            for engine in engine_order
        ],
    ]
    legend_columns = st.columns(len(legend_items))
    for column, (label, color) in zip(
        legend_columns,
        legend_items,
        strict=True,
    ):
        column.markdown(
            (
                '<div style="display:flex;align-items:center;gap:0.45rem;'
                'min-height:2rem;font-size:0.86rem;">'
                f'<span style="display:inline-block;width:0.9rem;'
                f'height:0.9rem;border-radius:50%;background:{color};'
                'border:1px solid #64748b;flex:0 0 auto;"></span>'
                f"<span>{escape(label)}</span></div>"
            ),
            unsafe_allow_html=True,
        )

    st.markdown("#### Target-panel key")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Panel": panel["panel"],
                    "Campaign": panel["label"],
                    "Target job": panel["target_id"],
                    "Target": panel["target"],
                }
                for panel in panels
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    try:
        import py3Dmol

        columns = min(grid_columns, len(panels))
        rows = math.ceil(len(panels) / columns)
        viewer = py3Dmol.view(
            width=1650,
            height=max(460, 440 * rows),
            viewergrid=(rows, columns),
            linked=True,
        )
        for ordinal, panel in enumerate(panels):
            grid = (ordinal // columns, ordinal % columns)
            viewer.addModel(
                panel["reference_data"],
                panel["reference_format"],
                viewer=grid,
            )
            viewer.setStyle(
                {"model": 0, "hetflag": False},
                (
                    {"cartoon": {"color": "#cbd5e1", "opacity": 0.5}}
                    if show_target
                    else {}
                ),
                viewer=grid,
            )
            viewer.setStyle(
                {"model": 0, "hetflag": True},
                (
                    {
                        "stick": {
                            "colorscheme": "greenCarbon",
                            "radius": 0.16,
                        }
                    }
                    if show_reference_ligand
                    else {}
                ),
                viewer=grid,
            )
            for model_index, item in enumerate(
                panel["rendered"],
                start=1,
            ):
                engine = str(item["row"]["engine"])
                scheme, color = engine_styles[engine]
                viewer.addModel(
                    item["data"],
                    str(item["format"]),
                    viewer=grid,
                )
                if item["kind"] == "complex":
                    viewer.setStyle(
                        {"model": model_index, "hetflag": False},
                        (
                            {
                                "cartoon": {
                                    "color": color,
                                    "opacity": 0.18,
                                }
                            }
                            if show_predicted_proteins
                            else {}
                        ),
                        viewer=grid,
                    )
                    viewer.setStyle(
                        {"model": model_index, "hetflag": True},
                        {
                            "stick": {
                                "colorscheme": scheme,
                                "radius": 0.2,
                            }
                        },
                        viewer=grid,
                    )
                else:
                    viewer.setStyle(
                        {"model": model_index},
                        {
                            "stick": {
                                "colorscheme": scheme,
                                "radius": 0.2,
                            }
                        },
                        viewer=grid,
                    )
            viewer.zoomTo({"hetflag": True}, viewer=grid)
            viewer.zoom(0.8, viewer=grid)
        render_persistent_3dmol(
            viewer,
            key=(
                f"campaign-target-matrix:{candidate}:"
                + ",".join(
                    str(panel["target_id"]) for panel in panels
                )
            ),
            height=max(480, 450 * rows),
            panel_titles=[
                f"Panel {panel['panel']}"
                for panel in panels
            ],
            panel_columns=columns,
            panel_width=1650,
        )
    except Exception as exc:
        st.error(f"Target matrix viewer failed: {exc}")
        return

    st.caption(
        "Each panel is one prepared target with the same compound and linked "
        "camera controls. Docked poses remain in their own prepared-target "
        "frame; predicted complexes are protein-Cα aligned to that panel's "
        "prepared target."
    )
    st.dataframe(
        pd.DataFrame(legend_rows),
        hide_index=True,
        width="stretch",
    )
    for message in dict.fromkeys(warnings):
        st.warning(message)


def _render_compound_matrix(structural: pd.DataFrame) -> None:
    target_rows = structural[
        ["target_run_id", "target", "launch_campaign"]
    ].drop_duplicates("target_run_id")
    target_ids = target_rows["target_run_id"].tolist()
    target_labels = {
        row["target_run_id"]: (
            str(row.get("launch_campaign") or "").strip()
            or str(row.get("target") or "").strip()
            or str(row["target_run_id"])
        )
        for _, row in target_rows.iterrows()
    }
    target_id = st.selectbox(
        "Focus target (from comparison scope)",
        target_ids,
        format_func=lambda value: target_labels.get(value, value),
        key="campaign_viewer_matrix_target",
        help="Every panel uses one common prepared-target coordinate frame.",
    )
    target_frame = structural.loc[
        structural["target_run_id"].eq(target_id)
    ].copy()
    if (
        st.session_state.get("_campaign_viewer_matrix_target_context")
        != str(target_id)
    ):
        st.session_state["_campaign_viewer_matrix_target_context"] = str(
            target_id
        )
        st.session_state.pop("campaign_viewer_matrix_compounds", None)
        st.session_state.pop("campaign_viewer_matrix_campaigns", None)
    candidates = sorted(
        target_frame["candidate_id"].astype(str).unique()
    )
    compound_labels = _compound_selector_labels(target_frame)
    selected_candidates = st.multiselect(
        "Focus compounds",
        candidates,
        format_func=lambda value: compound_labels.get(str(value), str(value)),
        default=candidates[: min(4, len(candidates))],
        max_selections=12,
        key="campaign_viewer_matrix_compounds",
        help=(
            "All py3Dmol cameras are linked: rotating, translating or zooming "
            "one panel moves every panel."
        ),
    )
    if not selected_candidates:
        st.info("Select at least one compound for the 3D matrix.")
        return
    matrix_rows = target_frame.loc[
        target_frame["candidate_id"].astype(str).isin(
            [str(value) for value in selected_candidates]
        )
    ].copy()
    st.caption(
        "The matrix inherits the globally selected engines and engine runs."
    )

    settings = st.columns(4)
    summary_mode = settings[0].selectbox(
        "Predictions per panel",
        ("Best per engine", "Representative per campaign"),
        key="campaign_viewer_matrix_mode",
    )
    grid_columns = int(
        settings[1].number_input(
            "Grid columns",
            min_value=1,
            max_value=4,
            value=min(2, len(selected_candidates)),
            key="campaign_viewer_matrix_columns",
        )
    )
    show_target = settings[2].checkbox(
        "Show target protein",
        value=True,
        key="campaign_viewer_matrix_show_target",
    )
    show_reference_ligand = settings[3].checkbox(
        "Show input/reference ligand",
        value=True,
        key="campaign_viewer_matrix_show_reference_ligand",
    )
    show_predicted_proteins = st.checkbox(
        "Show predicted proteins in matrix",
        value=False,
        key="campaign_viewer_matrix_show_predicted_proteins",
        help=(
            "Predicted proteins are hidden by default so the ligand comparison "
            "remains readable."
        ),
    )

    target_path = Path(str(matrix_rows.iloc[0].get("_target_path") or ""))
    reference_complex = str(
        matrix_rows.iloc[0].get("_reference_complex_path") or ""
    )
    reference_path = (
        Path(reference_complex)
        if reference_complex and Path(reference_complex).is_file()
        else target_path
    )
    if not target_path.is_file() or not reference_path.is_file():
        st.warning("The prepared input target is unavailable for alignment.")
        return

    available_engines = set(matrix_rows["engine"].astype(str))
    engine_order = [
        engine
        for engine in STRUCTURE_ENGINE_ORDER
        if engine in available_engines
    ] + sorted(available_engines - set(STRUCTURE_ENGINE_ORDER))
    engine_styles = {
        engine: STRUCTURE_ENGINE_STYLES.get(
            engine,
            STRUCTURE_ENGINE_PALETTE[
                (len(STRUCTURE_ENGINE_ORDER) + index)
                % len(STRUCTURE_ENGINE_PALETTE)
            ],
        )
        for index, engine in enumerate(engine_order)
    }
    compound_rendered: dict[str, list[dict[str, object]]] = {}
    warnings: list[str] = []
    legend_rows: list[dict[str, object]] = []
    for candidate in selected_candidates:
        candidate_rows = matrix_rows.loc[
            matrix_rows["candidate_id"].astype(str).eq(str(candidate))
        ].copy()
        displayed = (
            _best_structure_rows(candidate_rows)
            if summary_mode == "Best per engine"
            else _representative_structure_rows(candidate_rows)
        )
        displayed = _ordered_structure_rows(displayed)
        rendered, candidate_warnings = _load_comparison_structures(
            displayed,
            reference_path,
        )
        warnings.extend(
            f"{candidate}: {message}" for message in candidate_warnings
        )
        compound_rendered[str(candidate)] = rendered
        for index, item in enumerate(rendered):
            row = item["row"]
            _, engine_color = engine_styles[str(row["engine"])]
            legend_rows.append(
                {
                    "compound": str(candidate),
                    "color": engine_color,
                    "engine": row["engine"],
                    "campaign": row["campaign"],
                    "prediction": row["_prediction_label"],
                    "protein_alignment_rmsd_angstrom": item[
                        "alignment_rmsd"
                    ],
                    "matched_ca_atoms": item["matched_atoms"],
                    "coordinate_frame": item["coordinate_frame"],
                    "structure_file": str(item["source_path"]),
                }
            )
    available_candidates = [
        str(candidate)
        for candidate in selected_candidates
        if compound_rendered.get(str(candidate))
    ]
    if not available_candidates:
        st.info("No selected compound structures could be loaded.")
        return

    try:
        import py3Dmol

        columns = min(grid_columns, len(available_candidates))
        rows = math.ceil(len(available_candidates) / columns)
        viewer = py3Dmol.view(
            width=1650,
            height=max(460, 440 * rows),
            viewergrid=(rows, columns),
            linked=True,
        )
        reference_data, reference_format = _model_text(reference_path)
        for ordinal, candidate in enumerate(available_candidates):
            grid = (ordinal // columns, ordinal % columns)
            viewer.addModel(reference_data, reference_format, viewer=grid)
            viewer.setStyle(
                {"model": 0, "hetflag": False},
                (
                    {"cartoon": {"color": "#cbd5e1", "opacity": 0.5}}
                    if show_target
                    else {}
                ),
                viewer=grid,
            )
            viewer.setStyle(
                {"model": 0, "hetflag": True},
                (
                    {
                        "stick": {
                            "colorscheme": "greenCarbon",
                            "radius": 0.16,
                        }
                    }
                    if show_reference_ligand
                    else {}
                ),
                viewer=grid,
            )
            for model_index, item in enumerate(
                compound_rendered[candidate],
                start=1,
            ):
                engine = str(item["row"]["engine"])
                scheme, color = engine_styles[engine]
                viewer.addModel(
                    item["data"],
                    str(item["format"]),
                    viewer=grid,
                )
                if item["kind"] == "complex":
                    viewer.setStyle(
                        {"model": model_index, "hetflag": False},
                        (
                            {
                                "cartoon": {
                                    "color": color,
                                    "opacity": 0.18,
                                }
                            }
                            if show_predicted_proteins
                            else {}
                        ),
                        viewer=grid,
                    )
                    viewer.setStyle(
                        {"model": model_index, "hetflag": True},
                        {
                            "stick": {
                                "colorscheme": scheme,
                                "radius": 0.2,
                            }
                        },
                        viewer=grid,
                    )
                else:
                    viewer.setStyle(
                        {"model": model_index},
                        {
                            "stick": {
                                "colorscheme": scheme,
                                "radius": 0.2,
                            }
                        },
                        viewer=grid,
                    )
            viewer.addLabel(
                candidate,
                {
                    "position": {"x": 0, "y": 0, "z": 0},
                    "fontColor": "#0f172a",
                    "backgroundColor": "white",
                    "backgroundOpacity": 0.8,
                    "fontSize": 16,
                    "inFront": True,
                },
                viewer=grid,
            )
            viewer.zoomTo({"hetflag": True}, viewer=grid)
            viewer.zoom(0.8, viewer=grid)
        render_persistent_3dmol(
            viewer,
            key=(
                f"campaign-comparison-matrix:{target_id}:"
                + ",".join(available_candidates)
            ),
            height=max(480, 450 * rows),
        )
    except Exception as exc:
        st.error(f"Compound matrix viewer failed: {exc}")
        return

    st.caption(
        "Panels share one linked py3Dmol camera: rotation, translation and "
        "zoom are synchronized. Every pose uses the prepared target frame; "
        "predicted complexes are protein-Cα aligned as rigid complexes."
    )
    st.dataframe(
        pd.DataFrame(legend_rows),
        hide_index=True,
        width="stretch",
    )

    all_pairs: list[pd.DataFrame] = []
    compound_pairs: dict[str, pd.DataFrame] = {}
    for candidate in available_candidates:
        _, candidate_pairs, candidate_warnings = _pose_similarity_tables(
            compound_rendered[candidate]
        )
        warnings.extend(
            f"{candidate}: {message}" for message in candidate_warnings
        )
        if not candidate_pairs.empty:
            candidate_pairs = candidate_pairs.copy()
            candidate_pairs.insert(0, "Compound", candidate)
            compound_pairs[candidate] = candidate_pairs
            all_pairs.append(candidate_pairs)
    if all_pairs:
        pairs = pd.concat(all_pairs, ignore_index=True)
        paired_engines = set(pairs["Engine A"]) | set(pairs["Engine B"])
        rmsd_engine_order = [
            engine for engine in engine_order if engine in paired_engines
        ]
        means, annotations, aggregate = _engine_pose_rmsd_summary(
            pairs,
            rmsd_engine_order,
        )
        st.markdown("#### Across-compound pose RMSD")
        st.caption(
            "Each cell aggregates chemically compatible pose comparisons "
            "within the selected compounds. Different compounds are never "
            "atom-mapped directly. Values are mean ± sample SD."
        )
        size = max(6.5, min(13.0, len(means) + 3.0))
        figure, axis = plt.subplots(figsize=(size, size))
        sns.heatmap(
            means,
            annot=annotations,
            fmt="",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=4,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Fixed-frame RMSD (Å)"},
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_title("Engine pose RMSD across selected compounds")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

        per_compound_matrices: dict[
            str, tuple[pd.DataFrame, pd.DataFrame]
        ] = {}
        for candidate in available_candidates:
            candidate_pair_table = compound_pairs.get(candidate)
            if candidate_pair_table is None:
                continue
            candidate_means, candidate_annotations, _ = (
                _engine_pose_rmsd_summary(
                    candidate_pair_table,
                    rmsd_engine_order,
                )
            )
            for engine in rmsd_engine_order:
                candidate_means.loc[engine, engine] = 0.0
                candidate_annotations.loc[engine, engine] = "0.00"
            per_compound_matrices[candidate] = (
                candidate_means,
                candidate_annotations,
            )
        matrix_candidates = [
            candidate
            for candidate in available_candidates
            if candidate in per_compound_matrices
        ]
        if matrix_candidates:
            compound_columns = min(grid_columns, len(matrix_candidates))
            compound_rows = math.ceil(
                len(matrix_candidates) / compound_columns
            )
            matrix_figure, matrix_axes = plt.subplots(
                compound_rows,
                compound_columns,
                squeeze=False,
                figsize=(
                    4.15 * compound_columns,
                    4.75 * compound_rows,
                ),
            )
            for ordinal, candidate in enumerate(matrix_candidates):
                axis = matrix_axes[
                    ordinal // compound_columns,
                    ordinal % compound_columns,
                ]
                candidate_means, _ = (
                    per_compound_matrices[candidate]
                )
                compact_annotations = candidate_means.apply(
                    lambda column: column.map(
                        lambda value: (
                            f"{float(value):.2f}"
                            if pd.notna(value)
                            else ""
                        )
                    )
                )
                sns.heatmap(
                    candidate_means,
                    annot=compact_annotations,
                    fmt="",
                    cmap="RdYlGn_r",
                    vmin=0,
                    vmax=4,
                    square=True,
                    linewidths=0.5,
                    cbar=False,
                    annot_kws={"fontsize": 7},
                    ax=axis,
                )
                axis.set_xlabel("")
                axis.set_ylabel("")
                axis.set_title(
                    candidate,
                    fontsize=10,
                    fontweight="semibold",
                    pad=10,
                )
                axis.tick_params(axis="both", labelsize=7, pad=2)
                axis.tick_params(axis="x", labelrotation=65)
                axis.tick_params(axis="y", labelrotation=0)
                for tick in (
                    list(axis.get_xticklabels())
                    + list(axis.get_yticklabels())
                ):
                    tick.set_color("black")
            for ordinal in range(
                len(matrix_candidates),
                compound_rows * compound_columns,
            ):
                matrix_axes[
                    ordinal // compound_columns,
                    ordinal % compound_columns,
                ].set_visible(False)
            matrix_figure.suptitle(
                "Per-compound engine pose RMSD",
                fontsize=13,
            )
            matrix_figure.subplots_adjust(
                left=0.07,
                right=0.985,
                bottom=0.07,
                top=0.94,
                wspace=0.42,
                hspace=0.82,
            )
            st.pyplot(matrix_figure, width="stretch")
            plt.close(matrix_figure)
            st.caption(
                "One square RMSD matrix is shown per 3D compound panel in "
                "the same order and compact grid layout. All matrices use the "
                "same numeric color range (dark = lower RMSD, yellow = higher "
                "RMSD); the redundant color bar is omitted."
            )
        st.dataframe(aggregate, hide_index=True, width="stretch")
    else:
        st.info(
            "At least two chemically compatible poses per compound are "
            "needed for the RMSD matrices."
        )
    for message in dict.fromkeys(warnings):
        st.warning(message)


def _render_analysis_set_input_recovery(candidate_rows: pd.DataFrame) -> None:
    """Compare every prepared target with its own input/reference ligand."""
    recovery_frames: list[pd.DataFrame] = []
    all_warnings: list[str] = []
    for _, target_group in candidate_rows.groupby("target_run_id", sort=False):
        displayed = _ordered_structure_rows(target_group)
        first = displayed.iloc[0]
        target_label = str(first.get("target") or first["target_run_id"])
        target_path = Path(str(first.get("_target_path") or ""))
        reference_complex_text = str(
            first.get("_reference_complex_path") or ""
        )
        reference_path = (
            Path(reference_complex_text)
            if reference_complex_text
            and Path(reference_complex_text).is_file()
            else target_path
        )
        reference_ligand_text = str(
            first.get("_reference_ligand_path") or ""
        )
        if not target_path.is_file() or not reference_path.is_file():
            all_warnings.append(
                f"{target_label}: prepared input target is unavailable."
            )
            continue
        rendered: list[dict[str, object]] = []
        for _, row in displayed.iterrows():
            original_path = Path(str(row.get("_structure_path") or ""))
            kind = str(row.get("_structure_kind") or "")
            structure_path = _preferred_viewer_structure_path(
                original_path,
                structure_kind=kind,
            )
            if not structure_path.is_file():
                continue
            if kind == "complex":
                try:
                    structure_data, alignment_rmsd, matched_atoms = (
                        aligned_structure_data(
                            str(reference_path),
                            reference_path.stat().st_mtime_ns,
                            str(structure_path),
                            structure_path.stat().st_mtime_ns,
                        )
                    )
                except Exception as exc:
                    all_warnings.append(
                        f"{target_label} · {row['_prediction_label']}: {exc}"
                    )
                    continue
                rendered.append(
                    {
                        "row": row,
                        "data": structure_data,
                        "format": "cif",
                        "kind": kind,
                        "alignment_rmsd": alignment_rmsd,
                        "matched_atoms": matched_atoms,
                        "source_path": structure_path,
                        "coordinate_frame": "protein Cα aligned",
                    }
                )
            else:
                data, model_format = _model_text(
                    structure_path,
                    int(row.get("_viewer_pose_index") or 1),
                )
                rendered.append(
                    {
                        "row": row,
                        "data": data,
                        "format": model_format,
                        "kind": kind,
                        "alignment_rmsd": None,
                        "matched_atoms": None,
                        "source_path": structure_path,
                        "coordinate_frame": "native docking target frame",
                    }
                )
        if not rendered:
            all_warnings.append(f"{target_label}: no structures could be loaded.")
            continue
        reference_data, _ = _model_text(reference_path)
        _, pairs, warnings = _pose_similarity_tables(
            rendered,
            reference_structure_data=reference_data,
            reference_ligand_path=(
                Path(reference_ligand_text) if reference_ligand_text else None
            ),
        )
        all_warnings.extend(f"{target_label}: {warning}" for warning in warnings)
        if pairs.empty or "Comparison type" not in pairs:
            continue
        recovery = pairs.loc[
            pairs["Comparison type"].eq("Input-pose recovery")
        ].rename(
            columns={
                "Pose B": "Prediction",
                "Engine B": "Engine",
                "Fixed-frame RMSD (Å)": "Input-pose RMSD (Å)",
                "Centroid distance (Å)": "Input-centroid displacement (Å)",
            }
        )
        if recovery.empty:
            continue
        recovery["Prepared target"] = target_label
        recovery["Attempt"] = (
            recovery.groupby("Engine", sort=False).cumcount() + 1
        )
        recovery_frames.append(recovery)

    if not recovery_frames:
        st.info(
            "None of the selected prepared targets has an input ligand that "
            "can be mapped to its predictions."
        )
        if all_warnings:
            with st.expander("RMSD exclusions and mapping warnings"):
                for warning in all_warnings:
                    st.warning(warning)
        return

    recovery = pd.concat(recovery_frames, ignore_index=True)
    means, annotations, summary = _target_engine_recovery_matrix(recovery)
    st.markdown("### Recovery to each prepared target's input pose")
    st.caption(
        "No focus target is required. Every row is a prepared target from the "
        "Analysis Set, and every prediction is compared with that target's own "
        "input ligand. Cells show mean RMSD across available independent "
        "attempts; ± values are sample SD. Colors use the fixed 0–4 Å "
        "scale across campaigns; ≤2 Å is the usual practical pose-recovery "
        "cutoff."
    )
    figure_width = min(10.0, max(7.0, 0.8 * len(means.columns) + 3.5))
    figure_height = min(7.0, max(4.5, 0.58 * len(means.index) + 2.0))
    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height), constrained_layout=True
    )
    sns.heatmap(
        means,
        annot=annotations,
        fmt="",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=4,
        linewidths=0.5,
        cbar_kws={"label": "Mean RMSD to own input pose (Å)"},
        ax=axis,
    )
    axis.set_title("Input-pose recovery across prepared targets")
    axis.set_xlabel("Engine")
    axis.set_ylabel("Prepared target")
    axis.tick_params(axis="x", labelrotation=35)
    st.pyplot(figure, width="content")
    plt.close(figure)
    with st.expander("Input-pose recovery values"):
        st.dataframe(
            recovery[
                [
                    "Prepared target",
                    "Engine",
                    "Attempt",
                    "Prediction",
                    "Input-pose RMSD (Å)",
                    "Input-centroid displacement (Å)",
                ]
            ].sort_values(["Prepared target", "Engine", "Attempt"]),
            hide_index=True,
            width="stretch",
        )
        st.dataframe(
            summary.rename(
                columns={
                    "mean": "Mean RMSD (Å)",
                    "std": "Sample SD (Å)",
                    "count": "Attempts",
                }
            ).drop(columns="Annotation"),
            hide_index=True,
            width="stretch",
        )
    if all_warnings:
        with st.expander("RMSD exclusions and mapping warnings"):
            for warning in all_warnings:
                st.warning(warning)


def _render_analysis_set_pose_agreement(
    candidate_rows: pd.DataFrame,
    *,
    result_view: str,
    pose_mode: str,
    combine_targets: bool = False,
    combine_compounds: bool = False,
    compound_labels: dict[str, str] | None = None,
) -> None:
    """Render every target independently without a target-focus selector."""
    metric_label = "Atom-mapped RMSD (Å)"
    if result_view == "Engine agreement matrix":
        metric_label = st.segmented_control(
            "Pose comparison metric",
            (
                "Atom-mapped RMSD (Å)",
                "Shape distance",
                "Centroid displacement (Å)",
            ),
            default="Atom-mapped RMSD (Å)",
            key="campaign_pose_agreement_metric",
            help=(
                "RMSD compares corresponding heavy atoms using the input "
                "ligand topology, including chemically equivalent atom "
                "mappings, and is the primary metric for the same ligand. "
                "Shape distance compares fixed-frame molecular volume (0 is "
                "identical), while centroid displacement measures only "
                "movement of the pose center."
            ),
        ) or "Atom-mapped RMSD (Å)"
    metric_columns = {
        "Atom-mapped RMSD (Å)": "Fixed-frame RMSD (Å)",
        "Shape distance": "Shape Tanimoto distance",
        "Centroid displacement (Å)": "Centroid distance (Å)",
    }
    value_column = metric_columns[metric_label]
    display_scale = _pose_comparison_scale(metric_label)
    st.caption(
        f"Fixed color scale across all campaigns: 0–"
        f"{float(display_scale['maximum']):g}. "
        f"Guide: {display_scale['guidance']}. Values above the maximum are "
        "shown with the highest-difference color. Thresholds are practical "
        "pose-comparison guides, not universal biological cutoffs."
    )
    panels: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    detail_frames: list[pd.DataFrame] = []
    all_warnings: list[str] = []
    grouping_columns = (
        ["candidate_id", "target_run_id"]
        if combine_compounds
        else ["target_run_id"]
    )
    # Build the job input from the same selected rows used below.  The worker
    # receives only source descriptors; it materializes alignments and RMSD
    # calculations outside this Streamlit request.
    precompute_contexts: list[dict[str, object]] = []
    for _, target_group in candidate_rows.groupby(grouping_columns, sort=False):
        if pose_mode == "Representative per campaign":
            displayed = _representative_structure_rows(target_group)
        elif pose_mode == "Best per engine":
            displayed = _best_structure_rows(target_group)
        else:
            displayed = target_group
        displayed = _ordered_structure_rows(displayed)
        if displayed.empty:
            continue
        first = displayed.iloc[0]
        target_path = Path(str(first.get("_target_path") or ""))
        reference_complex_text = str(first.get("_reference_complex_path") or "")
        reference_path = Path(reference_complex_text) if reference_complex_text and Path(reference_complex_text).is_file() else target_path
        rows: list[dict[str, object]] = []
        for _, row in displayed.iterrows():
            source_path = Path(str(row.get("_structure_path") or ""))
            if source_path.is_file():
                rows.append({
                    "row": {
                        "engine": str(row.get("engine") or ""),
                        "job_code": str(row.get("job_code") or ""),
                        "replicate": row.get("replicate"),
                        "seed": row.get("seed"),
                        "model_seed": row.get("model_seed"),
                        "prediction_id": row.get("prediction_id"),
                        "model_id": row.get("model_id"),
                        "candidate_id": str(row.get("candidate_id") or ""),
                        "_ligand_smiles": str(row.get("_ligand_smiles") or ""),
                        "_viewer_pose_index": int(row.get("_viewer_pose_index") or 1),
                    },
                    "source_path": str(source_path),
                    "kind": str(row.get("_structure_kind") or ""),
                })
        if len(rows) >= 2 and reference_path.is_file():
            candidate_id = str(first.get("candidate_id") or "")
            precompute_contexts.append({
                "key": f"{candidate_id}::{first.get('target_run_id')}",
                "reference_path": str(reference_path),
                "reference_ligand_path": str(first.get("_reference_ligand_path") or ""),
                "rendered": rows,
            })
    if not precompute_contexts:
        st.info("No selected target has enough comparable poses for RMSD.")
        return
    similarity_job = find_pose_similarity_job(precompute_contexts)
    if similarity_job is None:
        st.caption(
            f"{len(precompute_contexts)} target–compound contexts are ready. "
            "RMSD will not be calculated until you start the CPU job."
        )
        if not st.button(
            "Start pose-similarity calculation",
            key="campaign_start_pose_similarity",
            type="primary",
        ):
            return
        similarity_job = queue_pose_similarity_job(precompute_contexts)
        # Do not leave the previous "Start" controls above the newly queued
        # job during this same Streamlit render.
        st.rerun()
    if similarity_job.status != "completed":
        @st.fragment(run_every=5)
        def _poll_pose_similarity_status() -> None:
            """Poll job metadata without rerunning the expensive comparison page."""
            current_job = find_pose_similarity_job(precompute_contexts)
            if current_job is None:
                st.rerun()
            if current_job.status == "completed":
                # The charts are outside this fragment, so redraw the full page
                # exactly once when the worker publishes its completed result.
                st.rerun()
            progress = current_job.metadata.get("progress") if isinstance(current_job.metadata.get("progress"), dict) else {}
            completed = int(progress.get("completed") or 0)
            total = int(progress.get("total") or len(precompute_contexts))
            st.info("Pose-similarity calculation is running as a tracked CPU job. This page will become available when it completes.")
            st.progress(min(1.0, completed / max(1, total)), text=str(progress.get("label") or f"Queued: 0 of {total} contexts"))
            st.caption(f"Job {display_job_code(current_job.metadata.get('job_code'), current_job.run_id)} · {current_job.status} · {completed}/{total} contexts")
            if current_job.status == "queued" and st.button(
                "Cancel queued calculation",
                key="campaign_cancel_queued_pose_similarity",
            ):
                cancel_queued_pose_similarity_job(current_job)
                st.rerun()
            st.link_button("Open job", f"/job-results?{urlencode({'task_group': POSE_SIMILARITY_TASK_GROUP, 'run_id': current_job.run_id})}")

        _poll_pose_similarity_status()
        return
    precomputed_results = load_pose_similarity_results(similarity_job)
    for _, target_group in candidate_rows.groupby(grouping_columns, sort=False):
        if pose_mode == "Representative per campaign":
            displayed = _representative_structure_rows(target_group)
        elif pose_mode == "Best per engine":
            displayed = _best_structure_rows(target_group)
        else:
            displayed = target_group
        displayed = _ordered_structure_rows(displayed)
        first = displayed.iloc[0]
        target_label = str(first.get("target") or first["target_run_id"])
        candidate_id = str(first.get("candidate_id") or "")
        compound_label = (compound_labels or {}).get(candidate_id, candidate_id)
        panel_label = (
            f"{compound_label} · {target_label}"
            if combine_compounds
            else target_label
        )
        target_path = Path(str(first.get("_target_path") or ""))
        reference_complex_text = str(
            first.get("_reference_complex_path") or ""
        )
        reference_path = (
            Path(reference_complex_text)
            if reference_complex_text
            and Path(reference_complex_text).is_file()
            else target_path
        )
        reference_ligand_text = str(
            first.get("_reference_ligand_path") or ""
        )
        if not target_path.is_file() or not reference_path.is_file():
            all_warnings.append(
                f"{panel_label}: prepared input target is unavailable."
            )
            continue
        rendered: list[dict[str, object]] = []
        for _, row in displayed.iterrows():
            original_path = Path(str(row.get("_structure_path") or ""))
            kind = str(row.get("_structure_kind") or "")
            structure_path = _preferred_viewer_structure_path(
                original_path,
                structure_kind=kind,
            )
            if not structure_path.is_file():
                continue
            if kind == "complex":
                try:
                    structure_data, alignment_rmsd, matched_atoms = (
                        aligned_structure_data(
                            str(reference_path),
                            reference_path.stat().st_mtime_ns,
                            str(structure_path),
                            structure_path.stat().st_mtime_ns,
                        )
                    )
                except Exception as exc:
                    all_warnings.append(
                        f"{panel_label} · {row['_prediction_label']}: {exc}"
                    )
                    continue
                rendered.append(
                    {
                        "row": row,
                        "data": structure_data,
                        "format": "cif",
                        "kind": kind,
                        "alignment_rmsd": alignment_rmsd,
                        "matched_atoms": matched_atoms,
                        "source_path": structure_path,
                        "coordinate_frame": "protein Cα aligned",
                    }
                )
            else:
                data, model_format = _model_text(
                    structure_path,
                    int(row.get("_viewer_pose_index") or 1),
                )
                rendered.append(
                    {
                        "row": row,
                        "data": data,
                        "format": model_format,
                        "kind": kind,
                        "alignment_rmsd": None,
                        "matched_atoms": None,
                        "source_path": structure_path,
                        "coordinate_frame": "native docking target frame",
                    }
                )
        if len(rendered) < 2:
            all_warnings.append(
                f"{panel_label}: fewer than two structures could be loaded."
            )
            continue
        reference_data, _ = _model_text(reference_path)
        context_key = f"{candidate_id}::{first.get('target_run_id')}"
        calculated = precomputed_results.get(context_key)
        if calculated is None:
            all_warnings.append(f"{panel_label}: precomputed pose-similarity result is unavailable.")
            continue
        matrix, pairs, warnings = calculated
        all_warnings.extend(f"{panel_label}: {warning}" for warning in warnings)
        matrix = matrix.drop(
            index=["Input/reference ligand"],
            columns=["Input/reference ligand"],
            errors="ignore",
        )
        agreement_pairs = (
            pairs.loc[pairs["Comparison type"].eq("Prediction agreement")].copy()
            if "Comparison type" in pairs.columns
            else pd.DataFrame()
        )
        if matrix.empty or agreement_pairs.empty:
            continue
        agreement_pairs["Prepared target"] = target_label
        agreement_pairs["Candidate ID"] = candidate_id
        agreement_pairs["Compound"] = compound_label
        detail_frames.append(agreement_pairs)
        if result_view == "Engine agreement matrix":
            engine_order = list(
                dict.fromkeys(
                    displayed["engine"].astype(str).tolist()
                )
            )
            means, annotations, _ = _engine_pose_rmsd_summary(
                agreement_pairs,
                engine_order,
                value_column=value_column,
            )
            panels.append((panel_label, means, annotations))
        else:
            annotations = matrix.map(
                lambda value: f"{value:.2f}" if pd.notna(value) else ""
            )
            panels.append((panel_label, matrix, annotations))

    if not panels:
        st.info("No selected target has enough comparable poses for RMSD.")
        return
    if combine_compounds:
        if result_view != "Engine agreement matrix":
            st.info(
                "Multi-compound consensus is available for the engine-agreement "
                "matrix only. Select that RMSD result to continue."
            )
            return
        detail = pd.concat(detail_frames, ignore_index=True)
        context_rows: list[pd.DataFrame] = []
        for (candidate_id, target_label), context_pairs in detail.groupby(
            ["Candidate ID", "Prepared target"], sort=False
        ):
            engines = list(
                dict.fromkeys(
                    context_pairs["Engine A"].astype(str).tolist()
                    + context_pairs["Engine B"].astype(str).tolist()
                )
            )
            _, _, context_summary = _engine_pose_rmsd_summary(
                context_pairs,
                engines,
                value_column=value_column,
            )
            if context_summary.empty:
                continue
            context_summary["Candidate ID"] = str(candidate_id)
            context_summary["Compound"] = (compound_labels or {}).get(
                str(candidate_id), str(candidate_id)
            )
            context_summary["Prepared target"] = str(target_label)
            context_rows.append(context_summary)
        if not context_rows:
            st.info("No engine-pair summaries could be calculated for the selected compounds.")
            return
        context_summary = pd.concat(context_rows, ignore_index=True)
        inter_engine = context_summary.loc[
            context_summary["Engine A"].ne(context_summary["Engine B"])
        ].copy()
        if inter_engine.empty:
            st.info("At least two distinct engines are required for compound consensus.")
            return
        threshold = 1.0 if metric_label == "Atom-mapped RMSD (Å)" else 0.25
        consensus_rows = []
        for (target_label, candidate_id), values in inter_engine.groupby(
            ["Prepared target", "Candidate ID"], sort=False
        ):
            metric_values = pd.to_numeric(
                values["Mean RMSD (Å)"], errors="coerce"
            ).dropna()
            if metric_values.empty:
                continue
            label = (compound_labels or {}).get(str(candidate_id), str(candidate_id))
            consensus_rows.append(
                {
                    "Prepared target": str(target_label),
                    "Compound": label,
                    "Consensus pose distance": float(metric_values.median()),
                    f"Close agreement (≤{threshold:g})": float(
                        (metric_values <= threshold).mean()
                    ),
                    "Engine-pair contexts": int(len(metric_values)),
                }
            )
        if not consensus_rows:
            st.info("No comparable inter-engine pose distances are available for the selected compounds.")
            return
        consensus = pd.DataFrame(consensus_rows).sort_values(
            "Consensus pose distance", kind="stable"
        )
        st.markdown("### Compound pose-agreement consensus")
        st.caption(
            "Each target is reported independently. For every compound, the "
            "consensus is the median of its mean inter-engine distances within "
            "that target frame; lower values indicate stronger agreement."
        )
        st.dataframe(
            consensus,
            hide_index=True,
            width="stretch",
            column_config={
                "Consensus pose distance": st.column_config.NumberColumn(
                    metric_label, format="%.3f"
                ),
                f"Close agreement (≤{threshold:g})": st.column_config.ProgressColumn(
                    f"Close agreement (≤{threshold:g})", min_value=0.0, max_value=1.0,
                    format="%.0f%%",
                ),
            },
        )
        st.markdown("### Engine agreement by compound and prepared target")
        st.caption(
            "Columns are compounds and rows are engine pairs. Each target has "
            "its own receptor-coordinate frame and its own matrix; lower values "
            "mean those two engines predicted more similar poses for that compound."
        )
        for target_label, target_values in inter_engine.groupby(
            "Prepared target", sort=False
        ):
            matrix_rows = target_values.copy()
            matrix_rows["Engine pair"] = (
                matrix_rows["Engine A"].astype(str)
                + " ↔ "
                + matrix_rows["Engine B"].astype(str)
            )
            compound_matrix = matrix_rows.pivot_table(
                index="Engine pair",
                columns="Compound",
                values="Mean RMSD (Å)",
                aggfunc="first",
            )
            compound_summary = (
                matrix_rows.groupby("Compound", sort=False)["Mean RMSD (Å)"]
                .agg(["mean", "std", "count"])
                .sort_values("mean", kind="stable")
            )
            compound_matrix = compound_matrix.reindex(
                sorted(compound_matrix.index), axis=0
            ).reindex(compound_summary.index, axis=1)
            compact_compound_labels = {
                str(label): re.sub(r"^(.*) \(([^()]*)\)$", r"\1\n(\2)", str(label))
                for label in compound_matrix.columns
            }
            compound_matrix = compound_matrix.rename(
                columns=compact_compound_labels
            )
            annotations = compound_matrix.map(
                lambda value: f"{value:.2f}" if pd.notna(value) else ""
            )
            figure, axis = plt.subplots(
                figsize=(
                    min(21.0, max(8.5, 0.64 * len(compound_matrix.columns) + 4.0)),
                    min(14.0, max(6.0, 0.40 * len(compound_matrix.index) + 3.0)),
                ),
                constrained_layout=True,
            )
            sns.heatmap(
                compound_matrix,
                annot=annotations,
                fmt="",
                cmap="RdYlGn_r",
                vmin=0,
                vmax=float(display_scale["maximum"]),
                linewidths=0.5,
                cbar_kws={"label": metric_label},
                annot_kws={"fontsize": 7},
                ax=axis,
            )
            axis.set_title(str(target_label), fontsize=11, fontweight="semibold")
            axis.set_xlabel("Compound")
            axis.set_ylabel("Engine pair")
            axis.tick_params(axis="x", labelrotation=60, labelsize=7)
            axis.tick_params(axis="y", labelrotation=0, labelsize=7)
            st.pyplot(figure, width="stretch")
            plt.close(figure)
            figure, axis = plt.subplots(
                figsize=(
                    min(21.0, max(8.5, 0.64 * len(compound_summary) + 4.0)),
                    6.6,
                ),
                constrained_layout=True,
            )
            positions = np.arange(len(compound_summary))
            means = compound_summary["mean"].to_numpy(dtype=float)
            deviations = compound_summary["std"].fillna(0.0).to_numpy(dtype=float)
            axis.errorbar(
                positions,
                means,
                yerr=deviations,
                fmt="o",
                color="#2563eb",
                ecolor="#1e3a8a",
                elinewidth=1.2,
                capsize=3,
            )
            axis.set_xticks(
                positions,
                [
                    compact_compound_labels.get(str(label), str(label))
                    for label in compound_summary.index
                ],
                rotation=60,
                ha="right",
                fontsize=8,
            )
            axis.set_xlim(-0.6, max(0.6, len(compound_summary) - 0.4))
            axis.set_ylim(bottom=0)
            axis.set_title(
                f"{target_label} · mean ± SD across engine pairs",
                fontsize=10,
                fontweight="semibold",
            )
            axis.set_xlabel("Compound")
            axis.set_ylabel(metric_label)
            axis.grid(axis="y", alpha=0.25)
            st.pyplot(figure, width="stretch")
            plt.close(figure)
            summary_table = compound_summary.reset_index().rename(
                columns={
                    "mean": f"Mean {metric_label}",
                    "std": f"SD {metric_label}",
                    "count": "Engine-pair values",
                }
            )
            st.dataframe(
                summary_table,
                hide_index=True,
                width="stretch",
                column_config={
                    f"Mean {metric_label}": st.column_config.NumberColumn(
                        f"Mean {metric_label}", format="%.3f"
                    ),
                    f"SD {metric_label}": st.column_config.NumberColumn(
                        f"SD {metric_label}", format="%.3f"
                    ),
                },
            )
        with st.expander("Per-compound engine-pair values"):
            st.dataframe(
                inter_engine.sort_values(
                    ["Compound", "Prepared target", "Engine A", "Engine B"]
                ),
                hide_index=True,
                width="stretch",
            )
        if all_warnings:
            with st.expander("RMSD exclusions and mapping warnings"):
                for warning in all_warnings:
                    st.warning(warning)
        return
    if combine_targets:
        combined_pairs = pd.concat(detail_frames, ignore_index=True)
        engine_order = list(
            dict.fromkeys(candidate_rows["engine"].astype(str).tolist())
        )
        combined_means, combined_annotations, _ = _engine_pose_rmsd_summary(
            combined_pairs,
            engine_order,
            value_column=value_column,
        )
        panels = [
            (
                "All prepared targets combined",
                combined_means,
                combined_annotations,
            )
        ]
        st.markdown("### Combined pose agreement")
        st.caption(
            "One engine × engine matrix pools all valid within-target pose "
            f"pairs across the Analysis Set. Cells show mean {metric_label} "
            "± sample SD. "
            "Poses belonging to different prepared targets are never compared "
            "directly."
        )
    else:
        st.markdown("### Pose agreement for every prepared target")
        st.caption(
            "All prepared targets in the Analysis Set are shown. Each matrix "
            f"shows {metric_label} in its own receptor coordinate frame; "
            "poses from different prepared targets are never compared directly."
        )
    columns = (
        st.columns(2)
        if result_view == "Engine agreement matrix" and not combine_targets
        else None
    )
    for panel_index, (target_label, panel, annotations) in enumerate(panels):
        if result_view == "Engine agreement matrix":
            figure_size = (
                min(8.8, max(5.5, 0.62 * len(panel.columns) + 2.6)),
                min(7.2, max(4.8, 0.55 * len(panel.index) + 2.3)),
            )
        else:
            figure_size = (
                min(9.5, max(6.5, 0.38 * len(panel.columns) + 3.0)),
                min(8.2, max(6.0, 0.34 * len(panel.index) + 2.8)),
            )
        figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
        sns.heatmap(
            panel,
            annot=annotations,
            fmt="",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=float(display_scale["maximum"]),
            linewidths=0.5,
            cbar=result_view != "Engine agreement matrix",
            cbar_kws={"label": metric_label},
            annot_kws={"fontsize": 7},
            ax=axis,
        )
        axis.set_title(target_label, fontsize=10, fontweight="semibold")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", labelrotation=60, labelsize=7)
        axis.tick_params(axis="y", labelrotation=0, labelsize=7)
        if columns is None:
            st.pyplot(figure, width="content")
        else:
            with columns[panel_index % 2]:
                st.pyplot(figure, width="stretch")
        plt.close(figure)
    if detail_frames:
        with st.expander("Pairwise pose metrics"):
            st.dataframe(
                pd.concat(detail_frames, ignore_index=True),
                hide_index=True,
                width="stretch",
            )
    if all_warnings:
        with st.expander("RMSD exclusions and mapping warnings"):
            for warning in all_warnings:
                st.warning(warning)
def _render_structure_comparison(
    frame: pd.DataFrame,
    *,
    analysis_only: bool = False,
    forced_layout: str = "",
    gnina_criterion_label: str = "",
) -> None:
    structural = frame.loc[
        frame["_structure_path"].astype(str).str.strip().ne("")
    ].copy()
    if structural.empty:
        st.info(
            "The selected campaigns contain no structure-producing results. "
            "Nesso affinity campaigns intentionally do not emit structures."
        )
        return
    if "_viewer_pose_index" not in structural:
        structural["_viewer_pose_index"] = 1
    else:
        structural["_viewer_pose_index"] = (
            pd.to_numeric(structural["_viewer_pose_index"], errors="coerce")
            .fillna(1)
            .astype(int)
        )
    if structural["engine"].astype(str).eq("GNINA").any():
        criterion_label = gnina_criterion_label
        if not criterion_label:
            criterion_container = (
                st.expander("Advanced pose selection")
                if analysis_only
                else st.container()
            )
            with criterion_container:
                criterion_label = st.segmented_control(
                    "GNINA pose-selection criterion",
                    (
                        "Both rankings",
                        "CNN pose score",
                        "Empirical / Vina score",
                    ),
                    default=(
                        "Both rankings" if analysis_only else "CNN pose score"
                    ),
                    key="campaign_viewer_gnina_pose_criterion",
                    help=(
                        "Both rankings shows GNINA's CNN-ranked and Vina-ranked "
                        "poses as separate matrix entries."
                    ),
                )
        structural = _select_gnina_ranking_rows(structural, criterion_label)
    structural = _native_repeat_structure_rows(structural)
    layout = (
        "Single compound"
        if analysis_only
        else forced_layout
        if forced_layout
        else st.segmented_control(
            "Viewer layout",
            ("Single compound", "Target matrix", "Compound matrix"),
            default="Single compound",
            key="campaign_viewer_layout",
        )
    )
    if layout == "Target matrix":
        _render_target_matrix(structural)
        return
    if layout == "Compound matrix":
        _render_compound_matrix(structural)
        return
    candidates = sorted(structural["candidate_id"].astype(str).unique())
    compound_labels = _compound_selector_labels(structural)
    candidate_rows = pd.DataFrame()
    target_ids: list[str] = []
    target_labels: dict[str, str] = {}
    if not analysis_only:
        candidate = (
            candidates[0]
            if len(candidates) == 1
            else st.selectbox(
                "Compound",
                candidates,
                format_func=lambda value: compound_labels.get(
                    str(value), str(value)
                ),
                key="campaign_viewer_compound",
            )
        )
        candidate_rows = structural.loc[
            structural["candidate_id"].astype(str).eq(str(candidate))
        ].copy()
        target_rows = candidate_rows[
            ["target_run_id", "target", "launch_campaign"]
        ].drop_duplicates("target_run_id")
        target_ids = target_rows["target_run_id"].tolist()
        target_labels = {
            row["target_run_id"]: (
                str(row.get("target") or "").strip()
                or str(row.get("launch_campaign") or "").strip()
                or str(row["target_run_id"])
            )
            for _, row in target_rows.iterrows()
        }
    result_view = (
        st.segmented_control(
            "RMSD result",
            (
                "Engine agreement matrix",
                "Individual-pose matrix",
                "Recovery to input",
            ),
            default="Engine agreement matrix",
            key="campaign_rmsd_result_view",
            help=(
                "Engine agreement summarizes all selected pose pairs by "
                "engine. Individual-pose matrix retains every repetition. "
                "Recovery to input compares each prediction with the ligand "
                "stored in the prepared input complex."
            ),
        )
        if analysis_only
        else ""
    )
    analysis_pose_mode = ""
    target_presentation = "Single target"
    if analysis_only:
        analysis_pose_mode = st.segmented_control(
            "Poses to compare",
            (
                "Representative per campaign",
                "Best per engine",
                "All repetitions",
            ),
            default="All repetitions",
            key="campaign_rmsd_mode",
        )
        compound_scope = "Single compound"
        if result_view == "Engine agreement matrix" and len(candidates) > 1:
            compound_scope = st.segmented_control(
                "Compound scope",
                ("Single compound", "Selected compounds", "All compounds"),
                default="Single compound",
                key="campaign_rmsd_compound_scope",
                help=(
                    "Single compound preserves the detailed matrices below. "
                    "Selected and All summarize pose agreement per compound "
                    "without comparing different compounds directly."
                ),
            ) or "Single compound"
        if compound_scope != "Single compound":
            if compound_scope == "Selected compounds":
                selected_candidates = st.multiselect(
                    "Compounds included in consensus",
                    candidates,
                    default=candidates[:20],
                    format_func=lambda value: compound_labels.get(
                        str(value), str(value)
                    ),
                    key="campaign_rmsd_consensus_compounds",
                    help=(
                        "Twenty compounds are selected initially. Change this "
                        "list to compare any scientifically relevant subset."
                    ),
                )
            else:
                selected_candidates = candidates
                st.caption(
                    f"All {len(selected_candidates)} compounds in the active "
                    "scope will be summarized."
                )
            if not selected_candidates:
                st.info("Select at least one compound to calculate consensus.")
                return
            scoped_rows = structural.loc[
                structural["candidate_id"].astype(str).isin(
                    [str(value) for value in selected_candidates]
                )
            ].copy()
            _render_analysis_set_pose_agreement(
                scoped_rows,
                result_view=result_view,
                pose_mode=analysis_pose_mode,
                combine_compounds=True,
                compound_labels=compound_labels,
            )
            return
        candidate = (
            candidates[0]
            if len(candidates) == 1
            else st.selectbox(
                "Compound",
                candidates,
                format_func=lambda value: compound_labels.get(
                    str(value), str(value)
                ),
                key="campaign_viewer_compound",
            )
        )
        candidate_rows = structural.loc[
            structural["candidate_id"].astype(str).eq(str(candidate))
        ].copy()
        target_rows = candidate_rows[
            ["target_run_id", "target", "launch_campaign"]
        ].drop_duplicates("target_run_id")
        target_ids = target_rows["target_run_id"].tolist()
        target_labels = {
            row["target_run_id"]: (
                str(row.get("target") or "").strip()
                or str(row.get("launch_campaign") or "").strip()
                or str(row["target_run_id"])
            )
            for _, row in target_rows.iterrows()
        }
        presentation_options = ["All targets (separate)"]
        if (
            result_view == "Engine agreement matrix"
            and analysis_pose_mode == "All repetitions"
        ):
            presentation_options.append("Combined mean ± SD")
        presentation_options.append("Single-target drill-down")
        presentation_key = "campaign_rmsd_target_presentation"
        if st.session_state.get(presentation_key) not in presentation_options:
            st.session_state.pop(presentation_key, None)
        target_presentation = st.segmented_control(
            "Target presentation",
            presentation_options,
            default=presentation_options[0],
            key=presentation_key,
            help=(
                "Show every prepared target separately, pool valid "
                "within-target comparisons into one engine matrix, or inspect "
                "one target. Combined values never compare poses belonging to "
                "different target coordinate frames."
            ),
        )
        if target_presentation != "Single-target drill-down":
            if result_view == "Recovery to input":
                _render_analysis_set_input_recovery(candidate_rows)
            else:
                _render_analysis_set_pose_agreement(
                    candidate_rows,
                    result_view=result_view,
                    pose_mode=analysis_pose_mode,
                    combine_targets=(
                        target_presentation == "Combined mean ± SD"
                    ),
                )
            return
    if len(target_ids) == 1:
        target_id = target_ids[0]
    else:
        target_id = st.selectbox(
            (
                "Prepared target for drill-down"
                if analysis_only
                else "Prepared target to display"
            ),
            target_ids,
            format_func=lambda value: target_labels.get(value, value),
            key="campaign_viewer_target",
            help=(
                "Pairwise pose RMSD requires one shared receptor coordinate "
                "frame. Recovery to input does not use this selector: it "
                "automatically compares every prepared target with its own "
                "input ligand."
            ),
        )
    candidate_rows = candidate_rows.loc[
        candidate_rows["target_run_id"].eq(target_id)
    ].copy()
    if not analysis_only:
        st.caption(
            "Engines and engine runs are inherited from the comparison scope "
            "above; the viewer does not apply a second campaign filter."
        )
    pose_modes = (
        (
            "Representative per campaign",
            "Best per engine",
            "All repetitions",
            "Selected predictions",
        )
        if not analysis_only
        else ()
    )
    mode = analysis_pose_mode or st.segmented_control(
        "Predictions to display",
        pose_modes,
        default="Representative per campaign",
        key="campaign_viewer_mode",
    )
    if mode == "Representative per campaign":
        displayed = _representative_structure_rows(candidate_rows)
        st.caption(
            "Each campaign shows the attempt closest to its median primary "
            "metric across repetitions. GNINA uses CNN affinity as its primary "
            "compound-ranking metric; this avoids displaying only the most "
            "optimistic replicate."
        )
    elif mode == "Best per engine":
        displayed = _best_structure_rows(candidate_rows)
        st.caption(
            "One most-favorable prediction is shown per engine using its "
            "native primary metric: lowest docking/energy score or predicted "
            "IC50, and highest GNINA CNN affinity or AlphaFold 3 ipTM."
        )
    elif mode == "All repetitions":
        displayed = candidate_rows
    else:
        candidate_rows = candidate_rows.reset_index(drop=True)
        options = list(range(len(candidate_rows)))
        selected = st.multiselect(
            "Predictions",
            options,
            default=options[:1],
            format_func=lambda value: str(
                candidate_rows.iloc[int(value)]["_prediction_label"]
            ),
            key="campaign_viewer_predictions",
        )
        displayed = candidate_rows.iloc[
            [int(value) for value in selected]
        ]
    displayed = _ordered_structure_rows(displayed)
    if displayed.empty:
        st.info("Select at least one structure-producing prediction.")
        return

    target_path_text = str(displayed.iloc[0].get("_target_path") or "")
    reference_complex_text = str(
        displayed.iloc[0].get("_reference_complex_path") or ""
    )
    reference_ligand_text = str(
        displayed.iloc[0].get("_reference_ligand_path") or ""
    )
    target_path = Path(target_path_text)
    reference_path = (
        Path(reference_complex_text)
        if reference_complex_text
        and Path(reference_complex_text).is_file()
        else target_path
    )
    if not target_path.is_file() or not reference_path.is_file():
        st.warning("The prepared input target is unavailable for alignment.")
        return

    has_predicted_complex = bool(
        displayed["_structure_kind"].astype(str).eq("complex").any()
    )
    if analysis_only:
        show_target = False
        show_reference_ligand = False
        show_predicted_proteins = False
    else:
        viewer_controls = st.columns(3)
        show_target = viewer_controls[0].checkbox(
            "Show target protein",
            value=True,
            key="campaign_viewer_show_target",
            help=(
                "Toggle the prepared input target used as the common coordinate "
                "reference. This does not change any alignment."
            ),
        )
        show_reference_ligand = viewer_controls[1].checkbox(
            "Show input/reference ligand",
            value=True,
            key="campaign_viewer_show_reference_ligand",
            help=(
                "Show non-protein atoms stored with the prepared input complex."
            ),
        )
        show_predicted_proteins = viewer_controls[2].checkbox(
            "Show predicted proteins",
            value=False,
            disabled=not has_predicted_complex,
            key="campaign_viewer_show_predicted_proteins",
            help=(
                "Predicted proteins are hidden by default so ligand poses can be "
                "compared without several overlapping cartoons."
            ),
        )

    displayed_engines = list(
        dict.fromkeys(displayed["engine"].astype(str).tolist())
    )
    extra_engines = [
        engine
        for engine in displayed_engines
        if engine not in STRUCTURE_ENGINE_STYLES
    ]
    engine_styles = dict(STRUCTURE_ENGINE_STYLES)
    engine_styles.update(
        {
            engine: STRUCTURE_ENGINE_PALETTE[
                (len(STRUCTURE_ENGINE_ORDER) + index)
                % len(STRUCTURE_ENGINE_PALETTE)
            ]
            for index, engine in enumerate(extra_engines)
        }
    )
    rendered: list[dict[str, object]] = []
    for _, row in displayed.iterrows():
        original_structure_path = Path(str(row["_structure_path"]))
        kind = str(row["_structure_kind"])
        structure_path = _preferred_viewer_structure_path(
            original_structure_path,
            structure_kind=kind,
        )
        if not structure_path.is_file():
            continue
        if kind == "complex":
            try:
                structure_data, rmsd, matched = aligned_structure_data(
                    str(reference_path),
                    reference_path.stat().st_mtime_ns,
                    str(structure_path),
                    structure_path.stat().st_mtime_ns,
                )
            except Exception as exc:
                st.warning(
                    f"Could not align {row['_prediction_label']}: {exc}"
                )
                continue
            rendered.append(
                {
                    "row": row,
                    "data": structure_data,
                    "format": "cif",
                    "kind": kind,
                    "alignment_rmsd": rmsd,
                    "matched_atoms": matched,
                    "source_path": structure_path,
                    "coordinate_frame": "protein Cα aligned",
                }
            )
        else:
            data, model_format = _model_text(
                structure_path,
                int(row.get("_viewer_pose_index") or 1),
            )
            rendered.append(
                {
                    "row": row,
                    "data": data,
                    "format": model_format,
                    "kind": kind,
                    "alignment_rmsd": None,
                    "matched_atoms": None,
                    "source_path": structure_path,
                    "coordinate_frame": "native docking target frame",
                }
            )
    if not rendered:
        st.info("No selected structures could be loaded.")
        return
    if analysis_only:
        reference_structure_data, _ = _model_text(reference_path)
        _render_pose_similarity(
            rendered,
            reference_structure_data=reference_structure_data,
            reference_ligand_path=(
                Path(reference_ligand_text)
                if reference_ligand_text
                else None
            ),
            result_view=result_view,
        )
        return

    st.caption(
        "The prepared input complex is the reference frame. Predicted complexes "
        "are rigidly aligned through matched protein Cα atoms, transforming each "
        "protein and ligand together. Classical docking poses already use the "
        "prepared target coordinate frame and are never independently fitted "
        "to another ligand. Vina/GNINA poses are displayed from their stored "
        "SDF conversion so molecular connectivity is preserved."
    )
    try:
        import py3Dmol

        viewer = py3Dmol.view(width=1100, height=680)
        reference_data, reference_format = _model_text(reference_path)
        viewer.addModel(reference_data, reference_format)
        viewer.setStyle(
            {"model": 0, "hetflag": False},
            (
                {"cartoon": {"color": "#cbd5e1", "opacity": 0.55}}
                if show_target
                else {}
            ),
        )
        viewer.setStyle(
            {"model": 0, "hetflag": True},
            (
                {
                    "stick": {
                        "colorscheme": "greenCarbon",
                        "radius": 0.18,
                    }
                }
                if show_reference_ligand
                else {}
            ),
        )
        for index, item in enumerate(rendered, start=1):
            scheme, color = engine_styles[str(item["row"]["engine"])]
            viewer.addModel(item["data"], str(item["format"]))
            if item["kind"] == "complex":
                viewer.setStyle(
                    {"model": index, "hetflag": False},
                    (
                        {
                            "cartoon": {
                                "color": color,
                                "opacity": (
                                    0.18
                                    if len(rendered) > 1
                                    else 0.42
                                ),
                            }
                        }
                        if show_predicted_proteins
                        else {}
                    ),
                )
                viewer.setStyle(
                    {"model": index, "hetflag": True},
                    {
                        "stick": {
                            "colorscheme": scheme,
                            "radius": 0.22,
                        }
                    },
                )
            else:
                viewer.setStyle(
                    {"model": index},
                    {
                        "stick": {
                            "colorscheme": scheme,
                            "radius": 0.22,
                        }
                    },
                )
        viewer.zoomTo({"hetflag": True})
        viewer.zoom(0.78)
        render_persistent_3dmol(
            viewer,
            key=f"campaign-comparison:{target_id}:{candidate}",
            height=700,
        )
    except Exception as exc:
        st.error(f"Structure comparison viewer failed: {exc}")
        return

    legend_rows = []
    for index, item in enumerate(rendered):
        row = item["row"]
        _, engine_color = engine_styles[str(row["engine"])]
        legend_rows.append(
            {
                "color": engine_color,
                "engine": row["engine"],
                "campaign": row["campaign"],
                "prediction": row["_prediction_label"],
                "protein_alignment_rmsd_angstrom": item[
                    "alignment_rmsd"
                ],
                "matched_ca_atoms": item["matched_atoms"],
                "coordinate_frame": item["coordinate_frame"],
                "structure_file": str(item["source_path"]),
            }
        )
    st.dataframe(
        pd.DataFrame(legend_rows),
        hide_index=True,
        width="stretch",
    )


def _render_rmsd_analysis(
    frame: pd.DataFrame,
    *,
    gnina_criterion_label: str = "",
) -> None:
    """Render RMSD controls without rerunning the full comparison page."""
    _render_structure_comparison(
        frame,
        analysis_only=True,
        gnina_criterion_label=gnina_criterion_label,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _md_selection_evidence_index(run_root_text: str) -> list[dict]:
    """Index reference evidence once instead of rescanning on every widget."""
    run_root = Path(run_root_text)
    jobs = iter_job_records(run_root)
    jobs_by_id = {job.run_id: job for job in jobs}
    static_targets: dict[str, list] = {}
    output: list[dict] = []
    for job in jobs:
        if (
            job.task_group != "interaction-analysis"
            or job.status != "completed"
            or int(
                job.metadata.get("residue_numbering_policy_version") or 0
            )
            != INTERACTION_RESIDUE_NUMBERING_POLICY_VERSION
            or not (job.run_dir / "interactions.csv").is_file()
        ):
            continue
        source_id = str(job.parent_run_id or "")
        source = jobs_by_id.get(source_id)
        if source is None or source.task_group in {
            "docking",
            "cofolding",
            "selected-complexes",
        }:
            continue
        static_targets.setdefault(source_id, []).append(job)
    for source_id, analyses in static_targets.items():
        source = jobs_by_id[source_id]
        label = str(
            source.metadata.get("target_provenance_key")
            or source.metadata.get("target_key")
            or source.metadata.get("pdb_id")
            or source.metadata.get("dataset_name")
            or "Prepared target"
        )
        tools = sorted(
            {
                str(job.metadata.get("interaction_engine") or job.tool or "")
                for job in analyses
            }
        )
        display_tools = [
            (
                "Native geometry (static; MD-style cutoffs)"
                if tool == "Native MD geometry"
                else tool
            )
            for tool in tools
        ]
        output.append(
            {
                "key": f"static:{source_id}",
                "kind": "static",
                "target_run_id": source_id,
                "pdb_id": str(source.metadata.get("pdb_id") or ""),
                "tools": tools,
                "label": (
                    f"{label} · static direct interactions "
                    f"({', '.join(display_tools)}) · "
                    f"{display_job_code(source.metadata.get('job_code'), source_id)}"
                ),
            }
        )
    md_by_workflow: dict[str, object] = {}
    for job in jobs:
        report_path = job.run_dir / "replicate_summary.json"
        if (
            job.task_group != "md-analysis"
            or job.status != "completed"
            or not report_path.is_file()
        ):
            continue
        workflow_id = str(
            job.workflow_parent_run_id
            or job.metadata.get("workflow_parent_run_id")
            or job.metadata.get("workflow_id")
            or job.run_id
        )
        previous = md_by_workflow.get(workflow_id)
        if previous is None or str(job.updated_at or job.created_at or "") > str(
            previous.updated_at or previous.created_at or ""
        ):
            md_by_workflow[workflow_id] = job
    for workflow_id, job in md_by_workflow.items():
        target_id = str(job.metadata.get("target_run_id") or "")
        if not target_id:
            continue
        workflow_job = jobs_by_id.get(workflow_id)
        workflow_name = str(
            (workflow_job.metadata.get("name") if workflow_job else "")
            or "MD simulation"
        )
        parameters = (
            workflow_job.metadata.get("parameters")
            if workflow_job
            and isinstance(workflow_job.metadata.get("parameters"), dict)
            else {}
        )
        production = (
            parameters.get("production")
            if isinstance(parameters.get("production"), dict)
            else {}
        )
        duration = production.get("target_duration_ns") or production.get(
            "production_length_ns"
        )
        replicas = parameters.get("replicas")
        source = (
            parameters.get("source")
            if isinstance(parameters.get("source"), dict)
            else {}
        )
        source_artifact = (
            source.get("artifact")
            if isinstance(source.get("artifact"), dict)
            else {}
        )
        source_job = jobs_by_id.get(str(source.get("run_id") or ""))
        source_artifact_path = (
            source_job.run_dir / str(source_artifact.get("path") or "")
            if source_job is not None and source_artifact.get("path")
            else None
        )
        label = str(
            job.metadata.get("target_provenance_key")
            or job.metadata.get("target_key")
            or target_id
        )
        output.append(
            {
                "key": f"md:{job.run_id}",
                "kind": "md",
                "target_run_id": target_id,
                "analysis_run_id": job.run_id,
                "workflow_run_id": workflow_id,
                "source_complex_path": (
                    str(source_artifact_path.resolve())
                    if source_artifact_path is not None
                    and source_artifact_path.is_file()
                    else ""
                ),
                "pdb_id": str(job.metadata.get("pdb_id") or ""),
                "label": (
                    f"{workflow_name} · {label} · trajectory MD "
                    f"stability/occupancy"
                    + (f" · {duration:g} ns" if isinstance(duration, (int, float)) else "")
                    + (f" · {replicas} replicas" if replicas else "")
                    + " · "
                    f"{display_job_code(job.metadata.get('job_code'), job.run_id)}"
                ),
            }
        )
    return sorted(
        output,
        key=lambda item: (
            item["kind"] != "static",
            str(item["label"]),
        ),
    )


def _md_selection_related_target_ids(
    run_root: Path, target_row: dict
) -> set[str]:
    related = {
        str(target_row.get("target_run_id") or ""),
        str(target_row.get("selected_target_run_id") or ""),
        str(target_row.get("target_family_run_id") or ""),
        str(target_row.get("_coordinate_target_run_id") or ""),
    }
    pending = [value for value in related if value]
    visited: set[str] = set()
    while pending and len(visited) < 32:
        run_id = pending.pop()
        if run_id in visited:
            continue
        visited.add(run_id)
        run_dir = _source_run_dir(run_root, run_id)
        metadata = (
            _read_json(run_dir / "metadata.json")
            if run_dir is not None
            else {}
        )
        for key in (
            "parent_run_id",
            "source_target_run_id",
            "prepared_target_run_id",
            "import_run_id",
        ):
            value = str(metadata.get(key) or "")
            if value and value not in visited:
                related.add(value)
                pending.append(value)
    return {value for value in related if value}


def _md_selection_reference_sources(
    run_root: Path,
    target_row: dict,
    *,
    include_other_targets: bool,
) -> list[dict]:
    evidence = _md_selection_evidence_index(str(run_root.resolve()))
    if include_other_targets:
        return evidence
    related = _md_selection_related_target_ids(run_root, target_row)
    pdb_id = str(
        target_row.get("target")
        or target_row.get("selected_target")
        or ""
    ).split("·", 1)[0].strip().upper()
    exact = [
        item
        for item in evidence
        if str(item.get("target_run_id") or "") in related
    ]
    if exact:
        return exact
    return [
        item
        for item in evidence
        if pdb_id
        and str(item.get("pdb_id") or "").strip().upper() == pdb_id
    ]


@st.cache_data(ttl=300, show_spinner=False)
def _md_selection_workflow_source_complex(
    run_root_text: str,
    workflow_run_id: str,
    analysis_run_id: str,
) -> str:
    """Resolve the immutable complex recorded by an MD workflow."""
    run_root = Path(run_root_text)
    resolved_workflow_id = str(workflow_run_id or "")
    if not resolved_workflow_id and analysis_run_id:
        analysis_metadata = _read_json(
            run_root / "md-analysis" / analysis_run_id / "metadata.json"
        )
        resolved_workflow_id = str(
            analysis_metadata.get("workflow_parent_run_id")
            or analysis_metadata.get("workflow_id")
            or ""
        )
    metadata = _read_json(
        run_root
        / "workflows"
        / resolved_workflow_id
        / "metadata.json"
    )
    if not metadata and analysis_run_id:
        analysis_metadata = _read_json(
            run_root / "md-analysis" / analysis_run_id / "metadata.json"
        )
        resolved_workflow_id = str(
            analysis_metadata.get("workflow_parent_run_id")
            or analysis_metadata.get("workflow_id")
            or ""
        )
        metadata = _read_json(
            run_root
            / "workflows"
            / resolved_workflow_id
            / "metadata.json"
        )
    parameters = (
        metadata.get("parameters")
        if isinstance(metadata.get("parameters"), dict)
        else {}
    )
    # A continuation/reuse workflow intentionally has no preparation step of
    # its own.  Its authoritative input is the immutable system prepared by
    # the earlier workflow.  Resolve that explicit lineage first rather than
    # falling back to a target-level PDB with different residue numbering.
    prepared_system_run_id = str(
        parameters.get("prepared_system_run_id")
        or metadata.get("prepared_system_run_id")
        or ""
    )
    if prepared_system_run_id:
        prep_dir = run_root / "md-system-prep" / prepared_system_run_id
        for candidate in (
            prep_dir / "source_complex.pdb",
            *sorted(prep_dir.glob("*_input_complex.pdb")),
        ):
            if candidate.is_file():
                return str(candidate.resolve())
    # Prefer the exact, local PDB snapshot written by MD system preparation.
    # It remains usable even when the original source artifact was recorded
    # through a container mount path that is not visible to the UI process.
    system_prep_root = run_root / "md-system-prep"
    if system_prep_root.is_dir() and resolved_workflow_id:
        for prep_dir in system_prep_root.iterdir():
            if not prep_dir.is_dir():
                continue
            prep_metadata = _read_json(prep_dir / "metadata.json")
            prep_workflow_id = str(
                prep_metadata.get("workflow_parent_run_id")
                or prep_metadata.get("workflow_id")
                or ""
            )
            if prep_workflow_id != resolved_workflow_id:
                continue
            candidates = [prep_dir / "source_complex.pdb"]
            candidates.extend(
                sorted(prep_dir.glob("*_input_complex.pdb"))
            )
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate.resolve())
    source = (
        parameters.get("source")
        if isinstance(parameters.get("source"), dict)
        else {}
    )
    artifact = (
        source.get("artifact")
        if isinstance(source.get("artifact"), dict)
        else {}
    )
    source_run_id = str(source.get("run_id") or "")
    artifact_path = str(artifact.get("path") or "")
    if not source_run_id or not artifact_path:
        return ""
    source_dir = _source_run_dir(run_root, source_run_id)
    candidate = (
        source_dir / artifact_path
        if source_dir is not None
        else None
    )
    return (
        str(candidate.resolve())
        if candidate is not None and candidate.is_file()
        else ""
    )


@st.cache_data(ttl=300, show_spinner=False)
def _md_selection_static_interactions(
    run_root_text: str, target_run_id: str
) -> pd.DataFrame:
    run_root = Path(run_root_text)
    newest: dict[str, tuple[str, Path, str]] = {}
    for job in iter_job_records(
        run_root, task_groups=("interaction-analysis",)
    ):
        if (
            str(job.parent_run_id or "") != str(target_run_id)
            or job.status != "completed"
            or int(
                job.metadata.get("residue_numbering_policy_version") or 0
            )
            != INTERACTION_RESIDUE_NUMBERING_POLICY_VERSION
        ):
            continue
        path = job.run_dir / "interactions.csv"
        if not path.is_file():
            continue
        engine = str(
            job.metadata.get("interaction_engine") or job.tool or ""
        )
        created = str(job.created_at or "")
        if engine not in newest or created > newest[engine][0]:
            newest[engine] = (created, path, job.run_id)
    frames: list[pd.DataFrame] = []
    for engine, (_, path, run_id) in newest.items():
        try:
            frame = pd.read_csv(path).fillna("")
        except (OSError, ValueError):
            continue
        frame["analysis_engine"] = engine
        frame["analysis_run_id"] = run_id
        frames.append(frame)
    return (
        _campaign_interaction_review_atoms(
            pd.concat(frames, ignore_index=True, sort=False)
        )
        if frames
        else pd.DataFrame()
    )


@st.cache_data(ttl=300, show_spinner=False)
def _md_selection_candidate_interactions(
    run_root_text: str,
    target_selected_jobs: pd.DataFrame,
) -> pd.DataFrame:
    """Keep immutable campaign interaction rows hot during pose review."""
    _, interactions, _ = _interaction_analysis_rows(
        Path(run_root_text), target_selected_jobs
    )
    return interactions


def _md_selection_interaction_map(
    interactions: pd.DataFrame,
    complex_path: Path | None,
    *,
    maximum_contacts: int = 12,
    source_job: JobRecord | None = None,
):
    """Use the same atom-aware network renderer as interaction results."""
    if (
        interactions.empty
        or complex_path is None
        or not complex_path.is_file()
    ):
        return None
    from mn_ligand.app.pages.job_results import (
        _interaction_ligand_molecule,
        _static_interaction_network_figure,
    )

    table = interactions.fillna("").copy()
    pose_id = str(
        table.get("pose_id", pd.Series("", index=table.index)).iloc[0]
        or "reference"
    )
    if source_job is None:
        source_job = JobRecord(
            run_id="reference",
            task_group="interaction-analysis",
            run_dir=complex_path.parent,
            status="completed",
        )
    first = table.iloc[0]
    molecule = _interaction_ligand_molecule(
        source_job,
        pose_id,
        complex_path,
        ligand_chain=str(first.get("ligand_chain") or ""),
        ligand_residue_name=str(first.get("ligand_residue_name") or ""),
        ligand_residue_number=str(first.get("ligand_residue_number") or ""),
    )
    return _static_interaction_network_figure(
        table,
        molecule=molecule,
        complex_pdb_text=complex_path.read_text(errors="replace"),
        residue_limit=max(int(maximum_contacts), 1),
        include_proximity=True,
    )


def _md_selection_visual_interaction_rows(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Expand aggregate MD residue labels into viewer coordinate columns."""
    table = interactions.copy().fillna("")
    for column in (
        "protein_chain",
        "protein_residue_name",
        "protein_residue_number",
        "protein_insertion_code",
    ):
        if column not in table:
            table[column] = ""
    for index, row in table.iterrows():
        if (
            str(row.get("protein_residue_name") or "").strip()
            and str(row.get("protein_residue_number") or "").strip()
        ):
            continue
        residue = str(
            row.get("Protein residue")
            or row.get("residue")
            or ""
        ).strip()
        match = re.fullmatch(
            r"(?:(?P<chain>[^:]+):)?"
            r"(?P<name>[A-Za-z]{3})"
            r"(?P<number>-?\d+)"
            r"(?P<insertion>[A-Za-z]?)",
            residue,
        )
        if match is None:
            continue
        table.at[index, "protein_chain"] = str(
            match.group("chain") or ""
        )
        table.at[index, "protein_residue_name"] = match.group(
            "name"
        ).upper()
        table.at[index, "protein_residue_number"] = match.group("number")
        table.at[index, "protein_insertion_code"] = str(
            match.group("insertion") or ""
        )
    return table


def _md_selection_mapping_payload(run_dir: Path) -> dict:
    """Read a residue map from a run without assuming its task group."""
    candidates = [
        run_dir / "source_residue_mapping.json",
        run_dir / "input" / "source_residue_mapping.json",
        run_dir / "artifacts" / "residue_mapping.json",
    ]
    manifest = _read_json(run_dir / "artifacts.json")
    for artifact in manifest.get("artifacts", []):
        if isinstance(artifact, dict) and artifact.get("artifact_type") == "residue_mapping":
            candidates.append(run_dir / str(artifact.get("path") or ""))
    for candidate in candidates:
        mapping = _read_json(candidate)
        if isinstance(mapping.get("residues"), list):
            return mapping
    embedded = _read_json(run_dir / "input.json").get("residue_mapping")
    return embedded if isinstance(embedded, dict) else {}


@st.cache_data(ttl=300, show_spinner=False)
def _md_selection_lineage_residue_mapping(
    run_root_text: str,
    source_run_id: str,
    complex_path_text: str = "",
) -> dict:
    """Find the author-number map across analysis, target and MD lineage."""
    run_root = Path(run_root_text)
    complex_path = Path(complex_path_text) if complex_path_text else None
    if complex_path is not None:
        mapping = _md_selection_mapping_payload(complex_path.parent)
        if isinstance(mapping.get("residues"), list):
            return mapping
    pending = [str(source_run_id or "")]
    visited: set[str] = set()
    while pending and len(visited) < 32:
        run_id = pending.pop(0)
        if not run_id or run_id in visited:
            continue
        visited.add(run_id)
        run_dir = _source_run_dir(run_root, run_id)
        if run_dir is None:
            continue
        mapping = _md_selection_mapping_payload(run_dir)
        if isinstance(mapping.get("residues"), list):
            return mapping
        metadata = _read_json(run_dir / "metadata.json")
        for key in (
            "parent_run_id", "target_run_id", "structure_run_id",
            "source_target_run_id", "prepared_target_run_id",
            "import_run_id",
        ):
            related = str(metadata.get(key) or "")
            if related and related not in visited:
                pending.append(related)
    return {}


def _md_selection_reference_coordinate_rows(
    interactions: pd.DataFrame,
    complex_path: Path | None,
    *,
    source_job: JobRecord | None = None,
    run_root: Path | None = None,
) -> pd.DataFrame:
    """Use author labels while retaining coordinates from the displayed PDB."""
    table = _md_selection_visual_interaction_rows(interactions)
    pdb_residues: set[tuple[str, str]] = set()
    pdb_residue_names: dict[tuple[str, str], set[str]] = {}
    pdb_chains_by_number_name: dict[tuple[str, str], set[str]] = {}
    if complex_path is not None and complex_path.is_file():
        try:
            pdb_text = complex_path.read_text(errors="replace")
            protein_coordinates, _ = pdb_interaction_atom_coordinates(pdb_text)
            pdb_residues = {
                (str(chain).strip(), str(residue_number).strip())
                for chain, residue_number, atom_name in protein_coordinates
                if not str(atom_name).startswith("#")
            }
            # The coordinate parser intentionally returns atom locations only.
            # Retain residue names here too: a processed target can contain
            # both residue 71 (the actual detector coordinate) and residue
            # 215 (an author-numbered, sequence-modified residue).  Matching
            # by number alone would style the wrong side chain.
            if not is_mmcif_text(pdb_text):
                for line in pdb_text.splitlines():
                    if line[:6].strip() != "ATOM" or len(line) < 26:
                        continue
                    pdb_residue_names.setdefault(
                        (line[21:22].strip(), line[22:26].strip()),
                        set(),
                    ).add(line[17:20].strip().upper())
                    pdb_chains_by_number_name.setdefault(
                        (
                            line[22:26].strip(),
                            line[17:20].strip().upper(),
                        ),
                        set(),
                    ).add(line[21:22].strip())
        except OSError:
            pass
    source_run_id = str(source_job.run_id) if source_job is not None else ""
    resolved_root = run_root
    if resolved_root is None and source_job is not None:
        resolved_root = source_job.run_dir.parent.parent
    mapping = (
        _md_selection_lineage_residue_mapping(
            str(resolved_root.resolve()),
            source_run_id,
            str(complex_path.resolve()) if complex_path is not None else "",
        )
        if resolved_root is not None
        else (
            load_residue_mapping(source_job)
            if source_job is not None
            else {}
        )
    )
    residues = (
        mapping.get("residues")
        if isinstance(mapping.get("residues"), list)
        else []
    )
    coordinate_by_native: dict[tuple[str, str, str, str], dict] = {}
    native_by_coordinate: dict[tuple[str, str, str, str], dict] = {}
    coordinate_by_native_residue: dict[tuple[str, str, str], list[dict]] = {}
    native_by_coordinate_residue: dict[tuple[str, str, str], list[dict]] = {}
    for residue in residues:
        if not isinstance(residue, dict):
            continue
        key = (
            str(residue.get("native_chain") or "").strip(),
            str(residue.get("native_residue_number") or "").strip(),
            str(residue.get("native_insertion_code") or "").strip(),
            str(residue.get("native_residue_name") or "").strip().upper(),
        )
        coordinate_by_native[key] = residue
        coordinate_by_native_residue.setdefault(key[1:], []).append(residue)
        coordinate_key = (
            str(residue.get("structure_chain") or "").strip(),
            str(residue.get("structure_residue_number") or "").strip(),
            str(residue.get("structure_insertion_code") or "").strip(),
            str(residue.get("structure_residue_name") or "").strip().upper(),
        )
        native_by_coordinate[coordinate_key] = residue
        native_by_coordinate_residue.setdefault(
            coordinate_key[1:], []
        ).append(residue)
    if not coordinate_by_native:
        return table
    for index, row in table.iterrows():
        # Preserve coordinates already published by the exact interaction
        # analysis.  Those are more specific than target-lineage numbering:
        # PLIP/PandaMap and individual prediction engines may rewrite chains
        # or residue IDs while preparing their own complex.
        stored_coordinate = (
            str(row.get("coordinate_protein_chain") or "").strip(),
            str(
                row.get("coordinate_protein_residue_number") or ""
            ).strip(),
            str(row.get("protein_residue_name") or "").strip().upper(),
        )
        native_key = (
            str(row.get("protein_chain") or "").strip(),
            str(row.get("protein_residue_number") or "").strip(),
            str(row.get("protein_insertion_code") or "").strip(),
            str(row.get("protein_residue_name") or "").strip().upper(),
        )
        coordinate = coordinate_by_native.get(native_key)
        if coordinate is None:
            candidates = coordinate_by_native_residue.get(native_key[1:], [])
            if len(candidates) == 1:
                coordinate = candidates[0]
        if coordinate is None:
            # Older static and continuation-MD analyses contain preparation
            # coordinates. Convert their displayed/scoring row back to the
            # immutable author numbering while retaining those coordinates for
            # the PDB viewer.
            coordinate = native_by_coordinate.get(native_key)
            if coordinate is None:
                candidates = native_by_coordinate_residue.get(
                    native_key[1:], []
                )
                if len(candidates) == 1:
                    coordinate = candidates[0]
            if coordinate is None:
                continue
            table.at[index, "protein_chain"] = str(
                coordinate.get("native_chain") or ""
            )
            table.at[index, "protein_residue_number"] = str(
                coordinate.get("native_residue_number") or ""
            )
            table.at[index, "protein_insertion_code"] = str(
                coordinate.get("native_insertion_code") or ""
            )
            table.at[index, "protein_residue_name"] = str(
                coordinate.get("native_residue_name") or ""
            )
        # Engine poses usually preserve original author numbering, while an
        # MD-system PDB uses preparation numbering. Inspect the PDB that is
        # actually being drawn and choose the representation it contains,
        # including the residue name to avoid collisions after mutations.
        raw_coordinate = (
            str(row.get("protein_chain") or "").strip(),
            str(row.get("protein_residue_number") or "").strip(),
            str(row.get("protein_residue_name") or "").strip().upper(),
        )
        author_coordinate = (
            str(table.at[index, "protein_chain"] or "").strip(),
            str(table.at[index, "protein_residue_number"] or "").strip(),
            str(table.at[index, "protein_residue_name"] or "").strip().upper(),
        )
        structure_coordinate = (
            str(coordinate.get("structure_chain") or "").strip(),
            str(coordinate.get("structure_residue_number") or "").strip(),
            str(coordinate.get("structure_residue_name") or "").strip().upper(),
        )
        chosen_chain, chosen_number, _ = structure_coordinate

        def _present(candidate: tuple[str, str, str]) -> bool:
            chain, number, residue_name = candidate
            names = pdb_residue_names.get((chain, number), set())
            return (
                (chain, number) in pdb_residues
                and (not names or not residue_name or residue_name in names)
            )

        # Prefer the raw detector coordinate when it exactly exists in the
        # displayed prepared complex; otherwise use the author label or the
        # mapping's preparation coordinate as appropriate.
        for candidate in (
            stored_coordinate,
            raw_coordinate,
            author_coordinate,
            structure_coordinate,
        ):
            if _present(candidate):
                chosen_chain, chosen_number, _ = candidate
                break
        else:
            # Chain labels are not stable across all engine-preparation
            # formats (for example A, _, or 1).  A chain-agnostic fallback is
            # safe only when residue number + residue name identifies exactly
            # one chain in the complex actually being displayed.
            for _, number, residue_name in (
                stored_coordinate,
                raw_coordinate,
                author_coordinate,
                structure_coordinate,
            ):
                matching_chains = pdb_chains_by_number_name.get(
                    (number, residue_name), set()
                )
                if len(matching_chains) == 1:
                    chosen_chain = next(iter(matching_chains))
                    chosen_number = number
                    break
        table.at[index, "coordinate_protein_chain"] = chosen_chain
        table.at[index, "coordinate_protein_residue_number"] = chosen_number
        table.at[index, "coordinate_protein_insertion_code"] = str(
            coordinate.get("structure_insertion_code") or ""
        )
    return table


def _md_selection_author_numbered_candidate_rows(
    run_root: Path,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Canonicalise every engine's receptor contacts before scoring poses."""
    if interactions.empty or "source_run_id" not in interactions:
        return interactions
    working = interactions.copy()
    working["_mapping_order"] = np.arange(len(working))
    analysis_ids = working.get(
        "analysis_run_id", pd.Series("", index=working.index)
    ).fillna("").astype(str)
    source_ids = working["source_run_id"].fillna("").astype(str)
    working["_mapping_run_id"] = analysis_ids.where(
        analysis_ids.str.strip().ne(""), source_ids
    )
    frames: list[pd.DataFrame] = []
    for mapping_run_id, rows in working.groupby(
        "_mapping_run_id", sort=False
    ):
        is_analysis = rows.get(
            "analysis_run_id", pd.Series("", index=rows.index)
        ).fillna("").astype(str).str.strip().ne("").any()
        task_group = "interaction-analysis" if is_analysis else ""
        source_dir = (
            run_root / task_group / str(mapping_run_id)
            if task_group
            else _source_run_dir(run_root, str(mapping_run_id))
        )
        source_job = None
        if source_dir is not None and source_dir.is_dir():
            try:
                source_job = JobRecord.load(
                    source_dir,
                    task_group=(task_group or source_dir.parent.name),
                )
            except (OSError, TypeError, ValueError):
                source_job = JobRecord(
                    run_id=str(mapping_run_id),
                    task_group=(task_group or source_dir.parent.name),
                    run_dir=source_dir,
                    status="completed",
                )
        complex_path = None
        if is_analysis:
            complex_values = _campaign_interaction_complex_paths(
                run_root, rows.iloc[[0]]
            )
            candidate_path = Path(str(complex_values.iloc[0] or ""))
            if candidate_path.is_file():
                complex_path = candidate_path
        frames.append(
            _md_selection_reference_coordinate_rows(
                rows,
                complex_path,
                source_job=source_job,
                run_root=run_root,
            )
        )
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values("_mapping_order", kind="stable")
        .drop(columns=["_mapping_order", "_mapping_run_id"], errors="ignore")
        .reset_index(drop=True)
    )


def _md_selection_mark_interaction_usage(
    interactions: pd.DataFrame,
    hypothesis: pd.DataFrame,
    *,
    minimum_detector_support: int,
) -> pd.DataFrame:
    """Mark exact detector rows that can contribute to the pose score."""
    table = _md_selection_visual_interaction_rows(interactions)
    table["_selection_interaction"] = table.get(
        "interaction_type", pd.Series("", index=table.index)
    ).map(normalize_interaction_type)
    table["_selection_residue"] = table.apply(
        interaction_residue,
        axis=1,
    )
    scopes = table.get(
        "protein_atom_scope", pd.Series("", index=table.index)
    ).astype(str).str.upper()
    table["_selection_region"] = scopes.where(
        scopes.isin(["BB", "SC"]),
        "BB+SC",
    )
    table["Used in selection"] = False
    enabled = hypothesis.loc[
        hypothesis.get(
            "Enabled", pd.Series(False, index=hypothesis.index)
        ).astype(bool)
    ]
    for criterion in enabled.to_dict("records"):
        interaction = normalize_interaction_type(
            criterion.get("Interaction")
        )
        residue = str(criterion.get("Protein residue") or "").strip()
        region = str(
            criterion.get("Protein region") or "BB+SC"
        ).strip().upper()
        mask = (
            table["_selection_interaction"].eq(interaction)
            & table["_selection_residue"].eq(residue)
        )
        if region in {"BB", "SC"}:
            mask &= table["_selection_region"].eq(region)
        tool_count = (
            table.loc[mask, "analysis_engine"].astype(str).nunique()
            if "analysis_engine" in table
            else int(mask.any())
        )
        if tool_count >= max(int(minimum_detector_support), 1):
            table.loc[mask, "Used in selection"] = True
    return table


def _md_selection_pi_specific_mask(interactions: pd.DataFrame) -> pd.Series:
    """Identify π-derived calls normalized to MD hydrophobic contacts."""
    values = interactions.get(
        "interaction_type",
        pd.Series("", index=interactions.index),
    ).astype(str).str.lower().str.replace("π", "pi", regex=False)
    return values.str.contains(
        r"(?:^|[^a-z])pi(?:[^a-z]|$)|stack",
        regex=True,
        na=False,
    )


def _md_selection_candidate_detector_policy(
    *,
    reference_kind: str,
    evidence_mode: str,
    available_reference_tools: object = (),
) -> tuple[tuple[str, ...], int, str]:
    """Match candidate scoring to the detector used by the reference."""
    if str(reference_kind) == "md":
        return (
            ("Native MD geometry",),
            1,
            "Native MD geometry (MD-compatible direct contacts)",
        )
    if evidence_mode == "PLIP static contacts":
        return (("PLIP",), 1, "PLIP")
    if evidence_mode == "PandaMap static contacts (grouped)":
        return (("PandaMap",), 1, "PandaMap (grouped)")
    if evidence_mode == "Static native geometry":
        return (("Native MD geometry",), 1, "Native MD geometry")
    tools = tuple(
        sorted(
            {
                str(tool).strip()
                for tool in (available_reference_tools or ())
                if str(tool).strip()
            }
        )
    )
    if evidence_mode.startswith("Static consensus"):
        return (
            tools,
            2,
            "Static detector consensus (at least 2 matching detectors)",
        )
    return (tools, 1, ", ".join(tools) or "Matching reference detector")


def _md_selection_filtered_reference_report(
    report: dict,
    interactions: pd.DataFrame,
) -> dict:
    """Limit an MD interaction-network report to the reviewed residues.

    The editable hypothesis and 3D panel operate on ``interactions``.  The MD
    network renderer otherwise ranks the complete report independently, which
    can make a reviewed residue disappear from the 2D panel.  Return a shallow
    report copy with only the same residue set; never modify persisted analysis.
    """
    filtered = dict(report)
    consensus = [
        row
        for row in report.get("contact_consensus") or []
        if isinstance(row, dict)
    ]
    if interactions.empty:
        filtered["contact_consensus"] = []
        return filtered

    allowed = {
        interaction_residue(row)
        for _, row in interactions.iterrows()
        if interaction_residue(row)
    }

    def residue_parts(value: str) -> tuple[str, str, str] | None:
        match = re.fullmatch(
            r"(?:(?P<chain>[^:]+):)?(?P<name>[A-Za-z]{3})"
            r"(?P<number>-?\d+[A-Za-z]?)",
            str(value).strip(),
        )
        if match is None:
            return None
        return (
            str(match.group("chain") or ""),
            match.group("name").upper(),
            match.group("number"),
        )

    # Some legacy reports omit a chain that is present after lineage mapping.
    # A chainless fallback is safe only when name+number identifies one enabled
    # residue, avoiding accidental cross-chain matches in homomers.
    allowed_parts = [parts for value in allowed if (parts := residue_parts(value))]
    unambiguous_chainless = {
        (name, number)
        for _, name, number in allowed_parts
        if sum(
            1
            for _, other_name, other_number in allowed_parts
            if (other_name, other_number) == (name, number)
        )
        == 1
    }

    selected: list[dict] = []
    for row in consensus:
        residue = interaction_residue(row)
        if residue in allowed:
            selected.append(row)
            continue
        parts = residue_parts(residue)
        if parts is not None and (parts[1], parts[2]) in unambiguous_chainless:
            selected.append(row)
    filtered["contact_consensus"] = selected
    return filtered


def _render_md_selection_reference_visual(
    *,
    target_row: dict,
    interactions: pd.DataFrame,
    complex_path: Path | None,
    key: str,
    maximum_contacts: int = 12,
    heading: str = "Reference interaction guidance",
    complex_heading: str = "3D input complex",
    source_job: JobRecord | None = None,
    md_report: dict | None = None,
    normalized_for_md_hypothesis: bool = False,
) -> None:
    interactions = _md_selection_visual_interaction_rows(interactions)
    trajectory_md = (
        source_job is not None
        and source_job.task_group == "md-analysis"
    )
    st.markdown(f"#### {heading}")
    caption = (
        (
            "The 2D map uses the actual MD-source ligand and colors stable "
            "residues by interaction class. Aggregate trajectory occupancy "
            "does not retain one truthful atom-to-atom geometry. The 3D "
            "view maps original residue numbers onto the actual MD input PDB; "
            "its dashed connectors locate those stable contacts in the input "
            "geometry and are not claimed as one measured trajectory frame. "
            "Residue labels report the detected protein region for each "
            "direct class (HB = hydrogen bond, HP = hydrophobic, "
            "SB = salt bridge; BB = backbone, SC = side chain)."
            if trajectory_md
            else
            "The 2D map and 3D view use the same exact direct-contact rows, "
            "interaction colors and atom-to-atom connectors."
        )
        + " Selecting a different campaign target or reference evidence "
        "rebuilds both views."
    )
    if normalized_for_md_hypothesis:
        caption += (
            " For this MD-derived hypothesis, candidate contacts come from "
            "the matching Native MD geometry calculation applied to this "
            "exact static pose."
        )
    st.caption(caption)
    left, right = st.columns(2)
    with left:
        st.markdown("**2D interaction map**")
        if trajectory_md and isinstance(md_report, dict):
            from mn_ligand.app.pages.job_results import _md_plot_figure

            display_report = _md_selection_filtered_reference_report(
                md_report,
                interactions,
            )
            figure = _md_plot_figure(
                "interaction_network",
                display_report,
                interaction_limit=max(
                    int(maximum_contacts),
                    len(display_report.get("contact_consensus") or []),
                    1,
                ),
                include_water_bridges=False,
                include_proximity_interactions=False,
                show_interaction_scopes=True,
            )
        else:
            figure = _md_selection_interaction_map(
                interactions,
                complex_path,
                maximum_contacts=maximum_contacts,
                source_job=source_job,
            )
        if figure is None:
            st.info("No direct reference contacts are available for a 2D map.")
        else:
            st.pyplot(figure, width="stretch")
            plt.close(figure)
    with right:
        st.markdown(f"**{complex_heading}**")
        if complex_path is None or not complex_path.is_file():
            st.info("The selected input complex is unavailable.")
            return
        try:
            import py3Dmol

            from mn_ligand.app.pages.job_results import (
                INTERACTION_DISPLAY_STYLES,
                _add_dashed_3d_interaction,
                _interaction_atom_columns,
                _selected_interaction_rows,
            )

            pdb_text = complex_path.read_text(errors="replace")
            grid = (0, 0)
            viewer = py3Dmol.view(
                width=720,
                height=540,
                viewergrid=(1, 1),
            )
            viewer.addModel(
                pdb_text,
                "cif" if is_mmcif_text(pdb_text) else "pdb",
                viewer=grid,
            )
            viewer.setStyle(
                {"hetflag": False},
                {"cartoon": {"color": "#cbd5e1", "opacity": 0.72}},
                viewer=grid,
            )
            viewer.setStyle(
                {"hetflag": True},
                {
                    "stick": {
                        "color": "#06b6d4",
                        "radius": 0.22,
                    }
                },
                viewer=grid,
            )
            interaction_table = _interaction_atom_columns(
                interactions.fillna("")
            )
            interaction_table = _md_selection_reference_coordinate_rows(
                interaction_table,
                complex_path,
                source_job=source_job,
            )
            for coordinate_column, source_column in (
                ("coordinate_protein_chain", "protein_chain"),
                (
                    "coordinate_protein_residue_number",
                    "protein_residue_number",
                ),
            ):
                missing = interaction_table[coordinate_column].astype(
                    str
                ).str.strip().eq("")
                interaction_table.loc[
                    missing, coordinate_column
                ] = interaction_table.loc[missing, source_column]
            selected_rows, _ = _selected_interaction_rows(
                interaction_table,
                residue_limit=max(int(maximum_contacts), 1),
                include_proximity=True,
            )
            # Several detectors can report the same atom-to-atom contact.
            # Retaining every duplicate creates hundreds of overlapping WebGL
            # cylinders for a single pose, which can exhaust Chrome's element
            # buffer and repeatedly remount the Streamlit iframe.  This is a
            # display-only reduction: the full detector evidence remains in
            # the table below and continues to drive MD selection unchanged.
            if not selected_rows.empty:
                connector_columns = [
                    "coordinate_protein_chain",
                    "coordinate_protein_residue_number",
                    "protein_atom_name",
                    "ligand_atom_name",
                    "kind",
                ]
                selected_rows = selected_rows.copy()
                selected_rows["_visual_distance"] = pd.to_numeric(
                    selected_rows.get("distance_angstrom"),
                    errors="coerce",
                ).fillna(float("inf"))
                selected_rows = (
                    selected_rows
                    .sort_values("_visual_distance", kind="stable")
                    .drop_duplicates(connector_columns, keep="first")
                    .head(max(12, 2 * max(int(maximum_contacts), 1)))
                    .drop(columns="_visual_distance")
                )
            styles = INTERACTION_DISPLAY_STYLES
            residue_groups = selected_rows.groupby(
                [
                    "coordinate_protein_chain",
                    "coordinate_protein_residue_number",
                ],
                dropna=False,
            )
            protein_coordinates, ligand_coordinates = (
                pdb_interaction_atom_coordinates(pdb_text)
            )
            for (chain, residue_number), residue_rows in residue_groups:
                dominant_kind = str(
                    residue_rows["kind"].value_counts().index[0]
                )
                color = styles.get(
                    dominant_kind, styles["contact"]
                )[0]
                scopes = set(
                    residue_rows["protein_atom_scope"]
                    .astype(str)
                    .str.upper()
                )
                scope_label = (
                    "BB+SC"
                    if "BB+SC" in scopes or {"BB", "SC"}.issubset(scopes)
                    else "BB"
                    if scopes == {"BB"}
                    else "SC"
                    if scopes == {"SC"}
                    else "BB+SC"
                )
                residue_selector: dict[str, object] = {
                    "hetflag": False,
                    "resi": str(residue_number).strip(),
                }
                if str(chain).strip():
                    residue_selector["chain"] = str(chain).strip()
                viewer.addStyle(
                    residue_selector,
                    {"stick": {"color": color, "radius": 0.13}},
                    viewer=grid,
                )
                residue_coordinates = [
                    coordinates
                    for (
                        atom_chain,
                        atom_residue,
                        atom_name,
                    ), coordinates in protein_coordinates.items()
                    if atom_chain == str(chain).strip()
                    and atom_residue == str(residue_number).strip()
                    and not atom_name.startswith("#")
                ]
                if residue_coordinates:
                    label_position = np.mean(
                        residue_coordinates,
                        axis=0,
                    )
                    source_row = residue_rows.iloc[0]
                    kinds = " · ".join(
                        sorted(
                            {
                                str(kind).replace("_", " ")
                                for kind in residue_rows["kind"]
                            }
                        )
                    )
                    viewer.addLabel(
                        (
                            f"{source_row.get('protein_residue_name', '')}"
                            f"{source_row.get('protein_residue_number', '')}"
                            f" · {kinds} · {scope_label}"
                        ),
                        {
                            "position": {
                                "x": float(label_position[0]),
                                "y": float(label_position[1]),
                                "z": float(label_position[2]),
                            },
                            "fontSize": 10,
                            "fontColor": "#111827",
                            "backgroundColor": "#ffffff",
                            "backgroundOpacity": 0.82,
                            "borderColor": color,
                            "borderThickness": 1,
                        },
                        viewer=grid,
                    )
            for _, interaction in selected_rows.iterrows():
                chain = str(
                    interaction.get("coordinate_protein_chain") or ""
                ).strip()
                residue_number = str(
                    interaction.get(
                        "coordinate_protein_residue_number"
                    ) or ""
                ).strip()
                protein_atom = str(
                    interaction.get("protein_atom_name") or ""
                ).strip().upper()
                ligand_atom = str(
                    interaction.get("ligand_atom_name") or ""
                ).strip().upper()
                start = protein_coordinates.get(
                    (chain, residue_number, protein_atom)
                )
                end = ligand_coordinates.get(ligand_atom)
                if start is None or end is None:
                    scope = str(
                        interaction.get("protein_atom_scope") or ""
                    ).strip().upper()
                    backbone_atoms = {"N", "CA", "C", "O", "OXT"}
                    protein_candidates = [
                        coordinates
                        for (
                            atom_chain,
                            atom_residue,
                            atom_name,
                        ), coordinates in protein_coordinates.items()
                        if atom_chain == chain
                        and atom_residue == residue_number
                        and not atom_name.startswith("#")
                        and (
                            scope not in {"BB", "SC"}
                            or (
                                scope == "BB"
                                and atom_name.upper() in backbone_atoms
                            )
                            or (
                                scope == "SC"
                                and atom_name.upper() not in backbone_atoms
                            )
                        )
                    ]
                    ligand_candidates = [
                        coordinates
                        for atom_name, coordinates in ligand_coordinates.items()
                        if not atom_name.startswith("#")
                    ]
                    if start is not None:
                        protein_candidates = [start]
                    if end is not None:
                        ligand_candidates = [end]
                    fallback = (
                        min(
                            (
                                (
                                    float(
                                        np.linalg.norm(
                                            protein_coordinate
                                            - ligand_coordinate
                                        )
                                    ),
                                    protein_coordinate,
                                    ligand_coordinate,
                                )
                                for protein_coordinate in protein_candidates
                                for ligand_coordinate in ligand_candidates
                            ),
                            key=lambda item: item[0],
                        )[1:]
                        if protein_candidates and ligand_candidates
                        else closest_residue_ligand_atom_pair(
                            protein_coordinates,
                            ligand_coordinates,
                            chain=chain,
                            residue_number=residue_number,
                        )
                    )
                    if fallback is not None:
                        start, end = fallback
                if start is None or end is None:
                    continue
                kind = str(interaction.get("kind") or "contact")
                _add_dashed_3d_interaction(
                    viewer,
                    start,
                    end,
                    color=styles.get(kind, styles["contact"])[0],
                    grid=grid,
                )
            viewer.zoomTo({"hetflag": True}, viewer=grid)
            viewer.zoom(0.86, viewer=grid)
            # Do not use the camera-persistent iframe for the MD reference
            # panel. Chromium may repeatedly recreate that iframe after a
            # WebGL buffer error, turning one bad redraw into a visible loop.
            # This one-shot component owns a single canvas for the selected
            # immutable input complex; a normal Streamlit rerun still creates
            # a fresh view when the selected evidence actually changes.
            import streamlit.components.v1 as components

            components.html(viewer._make_html(), height=560, scrolling=False)
        except Exception as exc:
            st.warning(f"Could not render the input complex: {exc}")


def _render_md_candidate_pose_inspection(
    *,
    run_root: Path,
    target_row: dict,
    focus: pd.Series,
    candidate_interactions: pd.DataFrame,
    hypothesis: pd.DataFrame,
    detector_support: int,
    score_target: str,
    top_interactions: int,
    md_reference: bool,
) -> None:
    focus_rows = candidate_interactions.loc[
        candidate_interactions["source_run_id"].astype(str).eq(
            str(focus["source_run_id"])
        )
        & candidate_interactions["pose_id"].astype(str).eq(
            str(focus["pose_id"])
        )
    ].copy()
    if focus_rows.empty:
        st.info("This selected pose has no linked interaction rows.")
        return
    evidence_paths = _campaign_interaction_complex_paths(
        run_root, focus_rows.iloc[[0]]
    )
    evidence_path = Path(str(evidence_paths.iloc[0] or ""))
    preview_rows = _campaign_interaction_review_atoms(focus_rows)
    preview_rows["Detector interaction"] = preview_rows[
        "interaction_type"
    ].astype(str)
    pi_specific = _md_selection_pi_specific_mask(preview_rows)
    if md_reference:
        preview_rows.loc[
            pi_specific, "interaction_type"
        ] = "hydrophobic contact"
    contact_preview = _md_selection_mark_interaction_usage(
        preview_rows,
        hypothesis,
        minimum_detector_support=int(detector_support),
    )
    contact_preview["Selection interpretation"] = (
        contact_preview["interaction_type"].astype(str)
    )
    if md_reference:
        contact_preview.loc[
            pi_specific,
            "Selection interpretation",
        ] = (
            "Hydrophobic contact (normalized from "
            + contact_preview.loc[
                pi_specific, "Detector interaction"
            ].astype(str)
            + ")"
        )
    pose_key = (
        f"{score_target}:{focus['source_run_id']}:{focus['pose_id']}"
    )
    interaction_display = st.segmented_control(
        "Candidate interactions displayed",
        (
            "Selection hypothesis only",
            "All scoring-detector interactions",
        ),
        default="Selection hypothesis only",
        key=f"campaign_md_selection_candidate_interactions_v2:{pose_key}",
        help=(
            "Selection hypothesis only shows contacts that contribute to this "
            "pose's score. The other view remains restricted to the detector "
            "matched to the selected reference; detections from unrelated "
            "tools do not enter this primary scoring view."
        ),
    )
    displayed_contact_preview = (
        contact_preview.loc[
            contact_preview["Used in selection"].astype(bool)
        ].copy()
        if interaction_display == "Selection hypothesis only"
        else contact_preview
    )
    first_analysis_id = str(
        focus_rows.iloc[0].get("analysis_run_id") or ""
    )
    first_analysis_dir = (
        run_root / "interaction-analysis" / first_analysis_id
    )
    focus_source_job = (
        JobRecord.load(
            first_analysis_dir,
            task_group="interaction-analysis",
        )
        if first_analysis_dir.is_dir()
        else None
    )
    _render_md_selection_reference_visual(
        target_row=target_row,
        interactions=displayed_contact_preview,
        complex_path=evidence_path if evidence_path.is_file() else None,
        key=f"candidate:{focus['source_run_id']}:{focus['pose_id']}",
        maximum_contacts=int(top_interactions),
        heading="Exact candidate interaction guidance",
        complex_heading="3D interactions for this exact pose",
        source_job=focus_source_job,
        normalized_for_md_hypothesis=md_reference,
    )
    contact_preview["Protein residue"] = (
        contact_preview["protein_chain"].fillna("").astype(str)
        + ":"
        + contact_preview["protein_residue_name"].fillna("").astype(str)
        + contact_preview["protein_residue_number"].fillna("").astype(str)
    )
    st.dataframe(
        contact_preview[
            [
                "analysis_engine",
                "Used in selection",
                "Detector interaction",
                "Selection interpretation",
                "Protein residue",
                "protein_atom_scope",
                "protein_atom_name",
                "ligand_atom_name",
                "distance_angstrom",
            ]
        ].drop_duplicates(),
        hide_index=True,
        width="stretch",
    )
    result_links = []
    for analysis_run_id, engine in (
        focus_rows[["analysis_run_id", "analysis_engine"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        result_links.append(
            f"[{engine} 2D interaction result]"
            "(./job-results?"
            + urlencode(
                {
                    "task_group": "interaction-analysis",
                    "run_id": str(analysis_run_id),
                    "label": str(engine),
                }
            )
            + ")"
        )
    st.markdown(" · ".join(result_links))


def _md_selection_final_choice_record(
    focus: dict,
    automatic_group: pd.DataFrame,
    *,
    alternative_rank: int,
    alternative_count: int,
    selection_score_column: str,
    require_contacts: bool,
    require_posebusters: bool,
    use_bend_penalty: bool,
    reject_excess_bend: bool,
    minimum_similarity: float,
) -> dict:
    """Attach reproducible selection provenance to one exact pose."""
    focus_dict = dict(focus)
    automatic_row = (
        automatic_group.iloc[0].to_dict()
        if not automatic_group.empty
        else {}
    )
    focus_key = (
        str(focus_dict.get("source_run_id") or ""),
        str(focus_dict.get("pose_id") or ""),
    )
    automatic_key = (
        str(automatic_row.get("source_run_id") or ""),
        str(automatic_row.get("pose_id") or ""),
    )
    if automatic_row and focus_key == automatic_key:
        selection_origin = "Automatic selection retained"
        user_changed = False
    elif automatic_row:
        selection_origin = "User changed automatic selection"
        user_changed = True
    else:
        selection_origin = "Manual selection; no automatically eligible pose"
        user_changed = True
    override_reasons: list[str] = []
    if (
        require_contacts
        and str(focus_dict.get("Required interactions met", "")).lower()
        != "true"
    ):
        override_reasons.append("missing a Must match interaction")
    if (
        require_posebusters
        and str(focus_dict.get("PoseBusters passed", "")).lower() != "true"
    ):
        override_reasons.append("PoseBusters did not pass")
    if (
        use_bend_penalty
        and reject_excess_bend
        and str(focus_dict.get("Bend criterion met", "")).lower() != "true"
    ):
        override_reasons.append("hard bend maximum exceeded")
    if float(focus_dict.get(selection_score_column) or 0.0) < float(
        minimum_similarity
    ):
        override_reasons.append("below the minimum selection score")
    focus_dict.update(
        {
            "Selection origin": selection_origin,
            "User changed automatic selection": user_changed,
            "Automatic eligibility checks passed": not override_reasons,
            "Manual override reasons": "; ".join(override_reasons),
            "Automatic source run ID": automatic_row.get("source_run_id", ""),
            "Automatic source engine": automatic_row.get("source_engine", ""),
            "Automatic pose ID": automatic_row.get("pose_id", ""),
            "Automatic prediction": automatic_row.get("prediction", ""),
            "Final alternative rank": int(alternative_rank),
            "Alternatives offered": int(alternative_count),
        }
    )
    return focus_dict


def _md_selection_default_hypothesis_indices(
    hypothesis: pd.DataFrame,
    eligible: pd.Series,
    *,
    residue_limit: int,
) -> list[object]:
    """Choose one automatically scored criterion per unique residue."""
    eligible_rows = hypothesis.loc[eligible.astype(bool)].copy()
    if eligible_rows.empty:
        return []
    eligible_rows["_generic_contact"] = eligible_rows["Interaction"].map(
        normalize_interaction_type
    ).eq("contact")
    selected = (
        eligible_rows
        .sort_values(
            [
                "Protein residue",
                "_generic_contact",
                "Importance",
                "Reference support",
            ],
            ascending=[True, True, False, False],
            kind="stable",
        )
        .drop_duplicates("Protein residue", keep="first")
        .sort_values(
            [
                "_generic_contact",
                "Importance",
                "Reference support",
                "Protein residue",
            ],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .head(max(int(residue_limit), 1))
    )
    return selected.index.tolist()


@st.fragment
def _render_md_final_pose_selector(
    *,
    run_root: Path,
    target_row: dict,
    alternatives: pd.DataFrame,
    selected_group: dict,
    automatic_group: pd.DataFrame,
    candidate_interactions: pd.DataFrame,
    hypothesis: pd.DataFrame,
    selection_score_column: str,
    require_contacts: bool,
    require_posebusters: bool,
    use_bend_penalty: bool,
    reject_excess_bend: bool,
    minimum_similarity: float,
    detector_support: int,
    score_target: str,
    top_interactions: int,
    md_reference: bool,
    group_title: str,
    group_digest: str,
    choice_store_key: str,
) -> None:
    """Rerender only one compound's authoritative final-pose view."""
    st.markdown(f"### {group_title}")
    option_rows = {
        (
            str(row.get("source_run_id") or ""),
            str(row.get("pose_id") or ""),
        ): row
        for row in alternatives.to_dict("records")
    }
    options = list(option_rows)
    if not options:
        st.info("No pose alternatives are available for this selection.")
        return
    stored_choices = st.session_state.setdefault(choice_store_key, {})
    stored_row = stored_choices.get(group_digest, {})
    stored_key = (
        str(stored_row.get("source_run_id") or ""),
        str(stored_row.get("pose_id") or ""),
    )
    current_key = (
        stored_key
        if stored_key in options
        else (
            str(selected_group.get("source_run_id") or ""),
            str(selected_group.get("pose_id") or ""),
        )
    )
    default_index = options.index(current_key) if current_key in options else 0
    final_key = st.selectbox(
        f"Final pose for {group_title}",
        options,
        index=default_index,
        format_func=lambda value, rows=option_rows,
        option_order=options: (
            f"#{option_order.index(value) + 1} · "
            f"{rows[value].get('source_engine', '')} · "
            f"{rows[value].get('prediction', '')} · "
            f"selection {rows[value].get(selection_score_column, 0)}% · "
            f"hypothesis "
            f"{rows[value].get('Reference similarity (%)', 0)}% · "
            "stereochemistry ✓"
        ),
        key=(
            f"campaign_md_selection_final_pose:{score_target}:"
            f"{group_digest}"
        ),
    )
    focus_dict = _md_selection_final_choice_record(
        option_rows[final_key],
        automatic_group,
        alternative_rank=options.index(final_key) + 1,
        alternative_count=len(options),
        selection_score_column=selection_score_column,
        require_contacts=require_contacts,
        require_posebusters=require_posebusters,
        use_bend_penalty=use_bend_penalty,
        reject_excess_bend=reject_excess_bend,
        minimum_similarity=minimum_similarity,
    )
    selection_origin = str(focus_dict["Selection origin"])
    override_reasons = [
        reason
        for reason in str(focus_dict["Manual override reasons"]).split("; ")
        if reason
    ]
    stored_choices[group_digest] = focus_dict
    st.session_state[choice_store_key] = stored_choices
    st.caption(
        f"{selection_origin}. Selection score "
        f"{focus_dict.get(selection_score_column, 0)}%; raw hypothesis "
        f"correspondence "
        f"{focus_dict.get('Reference similarity (%)', 0)}%. "
        "Immutable-source stereochemistry: preserved."
    )
    if override_reasons:
        st.warning(
            "This manual choice overrides automatic eligibility: "
            + "; ".join(override_reasons)
            + ". The override and reasons will be exported."
        )
    _render_md_candidate_pose_inspection(
        run_root=run_root,
        target_row=target_row,
        focus=pd.Series(focus_dict),
        candidate_interactions=candidate_interactions,
        hypothesis=hypothesis,
        detector_support=int(detector_support),
        score_target=score_target,
        top_interactions=int(top_interactions),
        md_reference=md_reference,
    )
    st.divider()


def _md_selection_hypothesis_workbook(
    selection: pd.DataFrame,
    hypothesis: pd.DataFrame,
    provenance: pd.DataFrame,
    *,
    candidate_ranking: pd.DataFrame | None = None,
    reference_evidence: pd.DataFrame | None = None,
    selection_audit: pd.DataFrame | None = None,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        selection.to_excel(writer, sheet_name="Selection", index=False)
        hypothesis.to_excel(writer, sheet_name="Hypothesis", index=False)
        provenance.to_excel(
            writer, sheet_name="Selection provenance", index=False
        )
        if candidate_ranking is not None and not candidate_ranking.empty:
            candidate_ranking.to_excel(
                writer,
                sheet_name="Candidate ranking",
                index=False,
            )
        if reference_evidence is not None and not reference_evidence.empty:
            reference_evidence.to_excel(
                writer,
                sheet_name="Reference evidence",
                index=False,
            )
        if selection_audit is not None and not selection_audit.empty:
            selection_audit.to_excel(
                writer,
                sheet_name="Compound audit",
                index=False,
            )
    return output.getvalue()


@st.cache_data(show_spinner=False)
def _md_selection_ligand_bend_index(
    path_text: str,
    modified_ns: int,
    size_bytes: int,
) -> float | None:
    """Measure the ligand in an immutable exact-complex PDB."""
    del modified_ns, size_bytes
    path = Path(path_text)
    if not path.is_file():
        return None
    try:
        pdb_text = path.read_text(errors="replace")
    except OSError:
        return None
    residue_coordinates: dict[tuple[str, str, str], list[list[float]]] = {}
    for line in pdb_text.splitlines():
        if line[:6].strip() != "HETATM" or len(line) < 54:
            continue
        residue_name = line[17:20].strip().upper()
        if residue_name in {"HOH", "WAT", "SOL"}:
            continue
        atom_name = line[12:16].strip().upper()
        element = line[76:78].strip().upper()
        if element == "H" or (
            not element and re.match(r"^\d*H(?:\d|$)", atom_name)
        ):
            continue
        try:
            coordinates = [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ]
        except ValueError:
            continue
        residue_coordinates.setdefault(
            (
                line[21:22].strip(),
                line[22:26].strip(),
                residue_name,
            ),
            [],
        ).append(coordinates)
    heavy_coordinates = max(
        residue_coordinates.values(),
        key=len,
        default=[],
    )
    if not heavy_coordinates:
        _, ligand_coordinates = pdb_interaction_atom_coordinates(pdb_text)
        heavy_coordinates = [
            coordinates
            for atom_name, coordinates in ligand_coordinates.items()
            if not str(atom_name).startswith("#")
            and not re.match(r"^\d*H(?:\d|$)", str(atom_name).upper())
        ]
    return ligand_bend_index(heavy_coordinates)


def _md_selection_path_bend_index(path: Path | None) -> float | None:
    if path is None or not path.is_file():
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return _md_selection_ligand_bend_index(
        str(path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


def _md_selection_import_rows(
    run_root: Path,
    selected_jobs: pd.DataFrame,
    selected: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    source_context = {
        str(row.get("campaign_id") or ""): row
        for row in selected_jobs.to_dict("records")
    }
    rows: list[dict] = []
    for selected_row in selected.to_dict("records"):
        matches = interactions.loc[
            interactions["source_run_id"].astype(str).eq(
                str(selected_row.get("source_run_id") or "")
            )
            & interactions["pose_id"].astype(str).eq(
                str(selected_row.get("pose_id") or "")
            )
        ]
        if matches.empty:
            continue
        evidence = matches.sort_values(
            "analysis_engine", kind="stable"
        ).iloc[[0]].copy()
        source_pose, source_receptor = _campaign_interaction_source_paths(
            run_root, evidence
        )
        evidence_path = _campaign_interaction_complex_paths(
            run_root, evidence
        ).iloc[0]
        source_id = str(selected_row.get("source_run_id") or "")
        source = source_context.get(source_id, {})
        rows.append(
            {
                "Selected rank": selected_row.get("Selected rank", ""),
                "Compound": str(selected_row.get("compound_id") or ""),
                "Selection status": str(
                    selected_row.get("Selection status") or ""
                ),
                "Selection origin": str(
                    selected_row.get("Selection origin") or ""
                ),
                "User changed automatic selection": selected_row.get(
                    "User changed automatic selection", ""
                ),
                "Automatic eligibility checks passed": selected_row.get(
                    "Automatic eligibility checks passed", ""
                ),
                "Manual override reasons": str(
                    selected_row.get("Manual override reasons") or ""
                ),
                "Automatic source run ID": str(
                    selected_row.get("Automatic source run ID") or ""
                ),
                "Automatic source engine": str(
                    selected_row.get("Automatic source engine") or ""
                ),
                "Automatic pose ID": str(
                    selected_row.get("Automatic pose ID") or ""
                ),
                "Automatic prediction": str(
                    selected_row.get("Automatic prediction") or ""
                ),
                "Final alternative rank": selected_row.get(
                    "Final alternative rank", ""
                ),
                "Alternatives offered": selected_row.get(
                    "Alternatives offered", ""
                ),
                "Campaign": str(
                    source.get("launch_campaign_label")
                    or source.get("launch_campaign")
                    or source.get("campaign")
                    or ""
                ),
                "Source job": display_job_code(
                    source.get("job_code"), source_id
                ),
                "Prediction engine": str(
                    selected_row.get("source_engine")
                    or source.get("engine")
                    or ""
                ),
                "Target": str(source.get("target") or ""),
                "Replicate": selected_row.get("replicate", ""),
                "Prediction": str(
                    selected_row.get("prediction") or ""
                ),
                "GNINA pose selection": (
                    _campaign_interaction_gnina_selection(
                        selected_row.get("source_engine"),
                        selected_row.get("selection_criterion"),
                    )
                ),
                "Source pose path": source_pose.iloc[0],
                "Source receptor path": source_receptor.iloc[0],
                "Interaction evidence path": evidence_path,
                "Reference similarity (%)": selected_row.get(
                    "Reference similarity (%)", ""
                ),
                "Selection score (%)": selected_row.get(
                    "Selection score (%)", ""
                ),
                "Reference ligand bend index": selected_row.get(
                    "Reference ligand bend index", ""
                ),
                "Candidate pose bend index": selected_row.get(
                    "Candidate pose bend index", ""
                ),
                "Excess bend index": selected_row.get(
                    "Excess bend index", ""
                ),
                "Bend penalty (percentage points)": selected_row.get(
                    "Bend penalty (percentage points)", ""
                ),
                "Bend criterion met": selected_row.get(
                    "Bend criterion met", ""
                ),
                "Required interactions met": selected_row.get(
                    "Required interactions met", ""
                ),
                "Matched reference interactions": selected_row.get(
                    "Matched reference interactions", ""
                ),
                "Missing required interactions": selected_row.get(
                    "Missing required interactions", ""
                ),
                "Supporting interaction tools": selected_row.get(
                    "Supporting interaction tools", ""
                ),
                "Source stereochemistry preserved": selected_row.get(
                    "Source stereochemistry preserved", ""
                ),
                "Stereochemistry validation": str(
                    selected_row.get("Stereochemistry validation") or ""
                ),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def _md_selection_stereochemistry_status(
    selection_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Validate exact pose geometry against immutable chemistry for all engines."""
    if selection_rows.empty:
        return pd.DataFrame()
    validated = validate_complex_dataset_rows(selection_rows)
    columns = [
        "Resolved source run ID",
        "Resolved pose ID",
        "Source stereochemistry preserved",
        "Stereochemistry policy version",
        "Validation",
    ]
    return validated[
        [column for column in columns if column in validated]
    ].rename(
        columns={
            "Resolved source run ID": "source_run_id",
            "Resolved pose ID": "pose_id",
            "Validation": "Stereochemistry validation",
        }
    )


def _render_md_candidate_selection(
    run_root: Path,
    selected_jobs: pd.DataFrame,
    selected_metrics: pd.DataFrame,
) -> None:
    st.markdown("## Select predicted complexes for MD")
    st.caption(
        "Build a target-specific selection hypothesis from a reference "
        "complex—not from a hard-coded receptor preset—then rank every "
        "interaction-analyzed campaign pose. Water bridges are excluded from "
        "inference and scoring. The reviewed hypothesis and exact source-pose "
        "lineage are stored with the Complex Dataset."
    )
    if st.button(
        "Refresh completed reference and candidate analyses",
        key="campaign_md_selection_refresh_evidence",
        help=(
            "The selection workspace caches completed immutable analyses for "
            "five minutes. Use this after new MD or interaction jobs finish."
        ),
    ):
        _md_selection_evidence_index.clear()
        _md_selection_workflow_source_complex.clear()
        _md_selection_static_interactions.clear()
        _md_selection_candidate_interactions.clear()
        _md_selection_stereochemistry_status.clear()
        st.rerun()
    target_rows = (
        selected_jobs.sort_values("target_run_id", kind="stable")
        .drop_duplicates("target_run_id")
        .to_dict("records")
    )
    if not target_rows:
        st.info("No campaign target is available.")
        return
    target_by_id = {
        str(row["target_run_id"]): row for row in target_rows
    }
    score_target = st.selectbox(
        "Campaign target to work on",
        list(target_by_id),
        format_func=lambda value: (
            str(target_by_id[value].get("target") or value)
            + " · input "
            + str(
                target_by_id[value].get("selected_target_key")
                or target_by_id[value].get("selected_target")
                or ""
            )
        ),
        key="campaign_md_selection_score_target",
        help=(
            "Each campaign target has an independent hypothesis and selection "
            "process. Finish or save this target, then select the second target."
        ),
    )
    target_row = target_by_id[score_target]
    identity_columns = st.columns(3)
    identity_columns[0].metric(
        "Campaign target",
        str(target_row.get("coordinate_target_key") or target_row.get("target")),
    )
    identity_columns[1].metric(
        "Input target",
        str(
            target_row.get("selected_target_key")
            or target_row.get("selected_target")
            or "-"
        ),
    )
    identity_columns[2].metric(
        "Input target job",
        display_job_code(
            "",
            str(target_row.get("selected_target_run_id") or ""),
        ),
    )
    include_other_targets = st.checkbox(
        "Use reference evidence from another target",
        value=False,
        key="campaign_md_selection_other_reference",
        help=(
            "Off keeps the evidence selector restricted to this campaign "
            "target and its preparation lineage."
        ),
    )
    references = _md_selection_reference_sources(
        run_root,
        target_row,
        include_other_targets=include_other_targets,
    )
    if not references:
        reference_complex_text = str(
            target_row.get("_reference_complex_path") or ""
        )
        _render_md_selection_reference_visual(
            target_row=target_row,
            interactions=pd.DataFrame(),
            complex_path=(
                Path(reference_complex_text)
                if reference_complex_text
                else None
            ),
            key=f"{score_target}:no-evidence",
        )
        st.info(
            "The input target is shown above, but it has no completed static "
            "interaction analysis or trajectory-MD occupancy result."
        )
        return
    reference_by_key = {str(item["key"]): item for item in references}
    reference_key = st.selectbox(
        "Reference evidence for this target",
        list(reference_by_key),
        format_func=lambda value: reference_by_key[value]["label"],
        key=f"campaign_md_selection_reference:{score_target}",
        help=(
            "Static detectors describe one structure. Trajectory MD describes "
            "contact stability over frames and replicas. They are distinct "
            "evidence types."
        ),
    )
    reference = reference_by_key[reference_key]
    evidence_mode = ""
    if reference["kind"] == "static":
        available_tools = set(reference.get("tools") or ())
        mode_options = ["Static consensus (at least 2 detectors)"]
        if "PLIP" in available_tools:
            mode_options.append("PLIP static contacts")
        if "PandaMap" in available_tools:
            mode_options.append("PandaMap static contacts (grouped)")
        if "Native MD geometry" in available_tools:
            mode_options.append("Static native geometry")
        evidence_mode = st.segmented_control(
            "Static evidence model",
            mode_options,
            default=mode_options[0],
            key=f"campaign_md_selection_static_mode:{score_target}",
            help=(
                "PandaMap-specific alkyl–π, carbon–π and π–alkyl calls are "
                "lumped into hydrophobic contacts. Static native geometry is "
                "a single-structure detector, not trajectory MD."
            ),
        ) or mode_options[0]
    else:
        st.info(
            "This is an actual trajectory-MD result. It weights direct "
            "hydrogen bonds, hydrophobic contacts and salt bridges by their "
            "mean occupancy across frames and available replicas. This is "
            "different from Native geometry, which applies MD-like distance "
            "rules to only one static structure. Water bridges remain excluded."
        )
    (
        candidate_scoring_engines,
        detector_support,
        candidate_scoring_label,
    ) = _md_selection_candidate_detector_policy(
        reference_kind=str(reference["kind"]),
        evidence_mode=evidence_mode,
        available_reference_tools=reference.get("tools") or (),
    )
    hbond_region = "BB+SC"
    controls = st.columns(3)
    occupancy = controls[0].number_input(
        "Minimum MD occupancy",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.05,
        key=f"campaign_md_selection_occupancy:{score_target}",
        disabled=reference["kind"] != "md",
    )
    top_interactions = controls[1].number_input(
        "Residues enabled by default",
        min_value=1,
        max_value=50,
        value=12,
        step=1,
        key=f"campaign_md_selection_top_interactions:{score_target}",
        help=(
            "The complete evidence remains in the table. This only controls "
            "how many unique highest-occupancy or best-supported protein "
            "residues are enabled initially and emphasized in the linked "
            "views. Multiple interaction types on one residue consume one "
            "slot. Residues with a specific interaction are selected first; "
            "generic contact-only residues fill any remaining slots."
        ),
    )
    controls[2].text_input(
        "Candidate scoring evidence",
        value=candidate_scoring_label,
        disabled=True,
        key=(
            "campaign_md_selection_scoring_detector:"
            f"{score_target}:{reference_key}:{evidence_mode}"
        ),
        help=(
            "The primary candidate score uses the same detector and contact "
            "definitions as the selected reference. Trajectory-MD contacts "
            "use the Native MD geometry calculation on each static candidate "
            "pose. A static consensus requires at least two of the same "
            "reference detectors."
        ),
    )
    reference_pose_id = ""
    reference_rows = pd.DataFrame()
    visual_rows = pd.DataFrame()
    reference_source_job: JobRecord | None = None
    reference_md_report: dict | None = None
    reference_complex_text = str(
        target_row.get("_reference_complex_path") or ""
    )
    reference_complex_path = (
        Path(reference_complex_text) if reference_complex_text else None
    )
    if reference["kind"] == "static":
        reference_rows = _md_selection_static_interactions(
            str(run_root.resolve()), str(reference["target_run_id"])
        )
        if not reference_rows.empty and "pose_id" in reference_rows:
            reference_poses = (
                reference_rows[["pose_id", "compound_id"]]
                .fillna("")
                .drop_duplicates("pose_id")
            )
            reference_pose_ids = reference_poses["pose_id"].astype(str).tolist()
            if len(reference_pose_ids) == 1:
                reference_pose_id = reference_pose_ids[0]
            if len(reference_pose_ids) > 1:
                pose_compounds = dict(
                    zip(
                        reference_poses["pose_id"].astype(str),
                        reference_poses["compound_id"].astype(str),
                        strict=True,
                    )
                )
                reference_pose_id = st.selectbox(
                    "Reference bound ligand",
                    reference_pose_ids,
                    format_func=lambda value: (
                        f"{pose_compounds.get(value, '')} · {value}"
                    ),
                    key=(
                        "campaign_md_selection_reference_pose:"
                        f"{score_target}:{reference_key}"
                    ),
                    help=(
                        "Only interactions made by this exact ligand pose are "
                        "used to infer the hypothesis."
                    ),
                )
                reference_rows = reference_rows.loc[
                    reference_rows["pose_id"].astype(str).eq(
                        str(reference_pose_id)
                    )
                ]
        if evidence_mode == "PLIP static contacts":
            reference_rows = reference_rows.loc[
                reference_rows["analysis_engine"].astype(str).eq("PLIP")
            ]
        elif evidence_mode == "PandaMap static contacts (grouped)":
            reference_rows = reference_rows.loc[
                reference_rows["analysis_engine"].astype(str).eq("PandaMap")
            ]
        elif evidence_mode == "Static native geometry":
            reference_rows = reference_rows.loc[
                reference_rows["analysis_engine"]
                .astype(str)
                .eq("Native MD geometry")
            ]
        if not reference_rows.empty:
            first_reference = reference_rows.iloc[0]
            analysis_run_id = str(
                first_reference.get("analysis_run_id") or ""
            )
            analysis_dir = (
                run_root / "interaction-analysis" / analysis_run_id
            )
            if analysis_dir.is_dir():
                reference_source_job = JobRecord.load(
                    analysis_dir,
                    task_group="interaction-analysis",
                )
            prepared = (
                run_root
                / "interaction-analysis"
                / analysis_run_id
                / "prepared"
                / f"{first_reference.get('pose_id')}.complex.pdb"
            )
            if prepared.is_file():
                reference_complex_path = prepared
        reference_rows = _md_selection_reference_coordinate_rows(
            reference_rows,
            reference_complex_path,
            source_job=reference_source_job,
            run_root=run_root,
        )
        visual_rows = reference_rows
        inferred = infer_static_hypothesis(
            reference_rows, hydrogen_bond_region=hbond_region
        )
    else:
        resolved_md_source = _md_selection_workflow_source_complex(
            str(run_root.resolve()),
            str(reference.get("workflow_run_id") or ""),
            str(reference.get("analysis_run_id") or ""),
        )
        md_source_complex = Path(
            resolved_md_source
            or str(reference.get("source_complex_path") or "")
        )
        if md_source_complex.is_file():
            # Show the exact immutable complex that seeded this MD workflow,
            # including its real bound ligand, rather than a target-level
            # fallback structure.
            reference_complex_path = md_source_complex
        report = _read_json(
            run_root
            / "md-analysis"
            / str(reference["analysis_run_id"])
            / "replicate_summary.json"
        )
        reference_md_report = report
        inferred = infer_md_hypothesis(
            list(report.get("contact_consensus") or ()),
            minimum_occupancy=float(occupancy),
            hydrogen_bond_region=hbond_region,
        )
        visual_rows = inferred.rename(
            columns={
                "Interaction": "interaction_type",
                "Protein residue": "Protein residue",
                "Protein region": "protein_atom_scope",
            }
        )
        analysis_dir = (
            run_root / "md-analysis" / str(reference["analysis_run_id"])
        )
        if analysis_dir.is_dir():
            reference_source_job = JobRecord.load(
                analysis_dir,
                task_group="md-analysis",
            )
        visual_rows = _md_selection_reference_coordinate_rows(
            visual_rows,
            reference_complex_path,
            source_job=reference_source_job,
            run_root=run_root,
        )
        visual_rows["Protein residue"] = visual_rows.apply(
            interaction_residue,
            axis=1,
        )
        inferred = visual_rows.rename(
            columns={
                "interaction_type": "Interaction",
                "protein_atom_scope": "Protein region",
            }
        )
    inferred["_generic_contact"] = inferred["Interaction"].map(
        normalize_interaction_type
    ).eq("contact")
    inferred = (
        inferred.sort_values(
            [
                "Importance",
                "Reference support",
                "_generic_contact",
                "Protein residue",
            ],
            ascending=[False, False, True, True],
            kind="stable",
        )
        .drop(columns="_generic_contact")
        .reset_index(drop=True)
    )
    if inferred.empty:
        st.warning(
            "The selected reference produced no eligible direct contacts. "
            "Water-mediated contacts are intentionally not converted into "
            "requirements."
        )
        return
    eligible = pd.Series(True, index=inferred.index)
    if evidence_mode.startswith("Static consensus"):
        eligible = pd.to_numeric(
            inferred["Reference support"], errors="coerce"
        ).fillna(0).ge(2)
    inferred["Enabled"] = False
    # The limit is a residue limit, matching the linked 2D network. A generic
    # contact and a hydrophobic/hydrogen-bond call on the same residue must
    # not consume two of the available slots or double-weight that residue in
    # the initial score. The prior sort leaves the strongest supported,
    # chemically specific criterion first for each residue.
    default_indices = _md_selection_default_hypothesis_indices(
        inferred,
        eligible,
        residue_limit=int(top_interactions),
    )
    inferred.loc[
        default_indices, "Enabled"
    ] = True
    signature = (
        "specific-interactions-first-v2",
        score_target,
        reference_key,
        evidence_mode,
        hbond_region,
        round(float(occupancy), 4),
        int(top_interactions),
        tuple(
            inferred.fillna("").astype(str).itertuples(
                index=False, name=None
            )
        ),
    )
    state_key = f"campaign_md_selection_hypothesis:{score_target}"
    state = st.session_state.get(state_key)
    if not isinstance(state, dict) or state.get("signature") != signature:
        st.session_state[state_key] = {
            "signature": signature,
            "frame": inferred,
        }
    st.markdown("#### Review the inferred hypothesis")
    st.info(
        "**How filtering works:** Use in score includes a row in the weighted "
        "reference-similarity score. Must match makes it a hard filter when "
        "'Enforce rows marked Must match' is enabled below. Rows sharing a "
        "Requirement group can use ANY, so reproducing one or more alternatives "
        "passes that group; ALL requires every row in the group. An empty group "
        "keeps the row independently mandatory. Required protein region "
        "controls accepted atoms per row: BB = backbone, SC = side chain, "
        "BB+SC = either."
    )
    hypothesis_frame = st.session_state[state_key]["frame"]
    quick_options = list(range(len(hypothesis_frame)))
    with st.expander("Quick interaction requirement editor", expanded=True):
        with st.form(
            key=f"campaign_md_selection_quick_form:{score_target}",
            border=False,
        ):
            quick_columns = st.columns((2.0, 0.8, 1.3, 1.0, 0.8, 0.7))
            quick_row = quick_columns[0].selectbox(
                "Interaction to configure",
                quick_options,
                index=0,
                format_func=lambda value: (
                    f"{hypothesis_frame.iloc[int(value)]['Interaction']} · "
                    f"{hypothesis_frame.iloc[int(value)]['Protein residue']}"
                ),
                key=f"campaign_md_selection_quick_row:{score_target}",
            )
            selected_hypothesis_row = hypothesis_frame.iloc[int(quick_row)]
            current_region = str(
                selected_hypothesis_row.get("Protein region") or "BB+SC"
            )
            quick_region = quick_columns[1].selectbox(
                "Accepted region",
                ("BB", "SC", "BB+SC"),
                index=("BB", "SC", "BB+SC").index(current_region),
                key=f"campaign_md_selection_quick_region:{score_target}",
                help=(
                    "BB+SC means that either backbone or side chain is "
                    "accepted."
                ),
            )
            current_enforcement = (
                "Must match (hard filter)"
                if bool(selected_hypothesis_row.get("Required"))
                else "Weighted preference"
                if bool(selected_hypothesis_row.get("Enabled"))
                else "Ignore interaction"
            )
            enforcement_options = (
                "Weighted preference",
                "Must match (hard filter)",
                "Ignore interaction",
            )
            quick_enforcement = quick_columns[2].selectbox(
                "Selection behavior",
                enforcement_options,
                index=enforcement_options.index(current_enforcement),
                key=(
                    "campaign_md_selection_quick_enforcement:"
                    f"{score_target}"
                ),
            )
            quick_group = quick_columns[3].text_input(
                "Requirement group",
                value=str(
                    selected_hypothesis_row.get("Requirement group") or ""
                ),
                key=f"campaign_md_selection_quick_group:{score_target}",
                help=(
                    "Give alternative Must-match rows the same group name. "
                    "Leave empty for an independently mandatory row."
                ),
            )
            current_logic = str(
                selected_hypothesis_row.get("Requirement logic") or "ALL"
            ).upper()
            if current_logic not in {"ALL", "ANY"}:
                current_logic = "ALL"
            quick_logic = quick_columns[4].selectbox(
                "Group rule",
                ("ALL", "ANY"),
                index=("ALL", "ANY").index(current_logic),
                key=f"campaign_md_selection_quick_logic:{score_target}",
                help=(
                    "ANY passes when at least one Must-match row in the named "
                    "group is detected. ALL requires every row."
                ),
            )
            quick_submitted = quick_columns[5].form_submit_button(
                "Apply",
                type="primary",
                width="stretch",
            )
        if quick_submitted:
            row_label = hypothesis_frame.index[int(quick_row)]
            hypothesis_frame.at[row_label, "Protein region"] = quick_region
            hypothesis_frame.at[row_label, "Enabled"] = (
                quick_enforcement != "Ignore interaction"
            )
            hypothesis_frame.at[row_label, "Required"] = (
                quick_enforcement == "Must match (hard filter)"
            )
            hypothesis_frame.at[row_label, "Requirement group"] = (
                quick_group.strip()
            )
            hypothesis_frame.at[row_label, "Requirement logic"] = quick_logic
            st.session_state[state_key]["frame"] = hypothesis_frame
            st.rerun()
    st.caption(
        "Choose any inferred interaction and its accepted region, then use "
        "Weighted preference to reward matching poses, Must match to exclude "
        "non-matching poses, or Ignore interaction to remove it from scoring. "
        "For alternatives, assign the same Requirement group and choose ANY. "
        "The table below remains the complete reproducible hypothesis."
    )
    with st.form(
        key=f"campaign_md_selection_table_form:{score_target}",
        border=False,
    ):
        edited_hypothesis = st.data_editor(
            st.session_state[state_key]["frame"],
            hide_index=True,
            width="stretch",
            disabled=["Interaction", "Protein residue", "Reference support",
                      "Reference evidence"],
            column_config={
                "Enabled": st.column_config.CheckboxColumn(
                    "Use in score",
                    default=True,
                    help=(
                        "Include this interaction in the weighted reference-"
                        "similarity score."
                    ),
                ),
                "Required": st.column_config.CheckboxColumn(
                    "Must match",
                    default=False,
                    help=(
                        "Exclude a pose when it does not reproduce this "
                        "interaction and Enforce rows marked Must match is "
                        "enabled."
                    ),
                ),
                "Protein region": st.column_config.SelectboxColumn(
                    "Required protein region",
                    options=["BB", "SC", "BB+SC"],
                    help=(
                        "Set this independently for every interaction. BB "
                        "requires a backbone contact; SC requires a side-chain "
                        "contact; BB+SC means either region is acceptable."
                    ),
                ),
                "Requirement group": st.column_config.TextColumn(
                    help=(
                        "Rows with the same non-empty group name are evaluated "
                        "together. Empty means this Must-match row is independent."
                    )
                ),
                "Requirement logic": st.column_config.SelectboxColumn(
                    options=["ALL", "ANY"],
                    help=(
                        "ANY accepts one or more detected rows in the group; "
                        "ALL requires every row in the group."
                    ),
                ),
                "Importance": st.column_config.NumberColumn(
                    "Weight",
                    min_value=0.0,
                    format="%.3f",
                    help=(
                        "Relative contribution to reference similarity. This "
                        "does not make the row mandatory unless Must match is "
                        "selected."
                    ),
                ),
            },
            key=f"campaign_md_selection_hypothesis_editor:{score_target}",
        )
        table_submitted = st.form_submit_button(
            "Apply hypothesis table changes",
            type="primary",
        )
    if table_submitted:
        st.session_state[state_key]["frame"] = edited_hypothesis
        st.rerun()
    hypothesis = st.session_state[state_key]["frame"]
    enabled_residues = {
        str(value).strip()
        for value in hypothesis.loc[
            hypothesis["Enabled"].astype(bool), "Protein residue"
        ].tolist()
        if str(value).strip()
    }
    hypothesis_visual_rows = visual_rows.copy()
    if not hypothesis_visual_rows.empty:
        hypothesis_visual_rows["_hypothesis_residue"] = (
            hypothesis_visual_rows.apply(interaction_residue, axis=1)
        )
        hypothesis_visual_rows = hypothesis_visual_rows.loc[
            hypothesis_visual_rows["_hypothesis_residue"].isin(
                enabled_residues
            )
        ].drop(columns="_hypothesis_residue")
    mandatory_rows = hypothesis.loc[
        hypothesis["Required"].astype(bool)
    ].copy()
    if not mandatory_rows.empty:
        mandatory_terms: list[str] = []
        grouped_required = mandatory_rows.loc[
            mandatory_rows["Requirement group"].astype(str).str.strip().ne("")
        ]
        grouped_indices: set[object] = set()
        for group_name, members in grouped_required.groupby(
            "Requirement group",
            sort=False,
        ):
            grouped_indices.update(members.index)
            logic = (
                "ANY"
                if members["Requirement logic"]
                .astype(str)
                .str.upper()
                .eq("ANY")
                .any()
                else "ALL"
            )
            joiner = " OR " if logic == "ANY" else " AND "
            labels = [
                (
                    f"{row['Interaction']} · {row['Protein residue']} "
                    f"({row['Protein region']})"
                )
                for _, row in members.iterrows()
            ]
            mandatory_terms.append(
                f"{group_name}: (" + joiner.join(labels) + ")"
            )
        for index, row in mandatory_rows.iterrows():
            if index in grouped_indices:
                continue
            mandatory_terms.append(
                f"{row['Interaction']} · {row['Protein residue']} "
                f"({row['Protein region']})"
            )
        st.success(
            "Current mandatory rule: " + " AND ".join(mandatory_terms)
        )
    _render_md_selection_reference_visual(
        target_row=target_row,
        interactions=hypothesis_visual_rows,
        complex_path=reference_complex_path,
        key=f"{score_target}:{reference_key}:{reference_pose_id}:{evidence_mode}",
        maximum_contacts=int(top_interactions),
        source_job=reference_source_job,
        md_report=reference_md_report,
    )
    with st.expander(
        "All direct reference-interaction evidence",
        expanded=False,
    ):
        st.caption(
            "This table is not truncated by consensus support or by the "
            "default-enabled residue count. Repeated rows remain separate "
            "when different detectors or atoms reported them."
        )
        evidence_table = (
            reference_rows.copy()
            if reference["kind"] == "static"
            else visual_rows.copy()
        ).fillna("")
        if not evidence_table.empty:
            evidence_table["Protein residue"] = evidence_table.apply(
                interaction_residue,
                axis=1,
            )
            evidence_table = evidence_table.rename(
                columns={"protein_atom_scope": "Protein region"}
            )
            evidence_columns = [
                column
                for column in (
                    "analysis_engine",
                    "interaction_type",
                    "Protein residue",
                    "Protein region",
                    "protein_atom_name",
                    "ligand_atom_name",
                    "distance_angstrom",
                    "angle_degree",
                    "Importance",
                    "Reference support",
                    "Reference evidence",
                )
                if column in evidence_table
            ]
            st.dataframe(
                evidence_table[evidence_columns],
                hide_index=True,
                width="stretch",
            )

    target_selected_jobs = selected_jobs.loc[
        selected_jobs["target_run_id"].astype(str).eq(score_target)
    ].copy()
    target_selected_metrics = selected_metrics.loc[
        selected_metrics["target_run_id"].astype(str).eq(score_target)
    ].copy()
    compound_names, _ = _compound_name_mapping(
        target_selected_metrics,
        identifier_column="candidate_id",
    )
    compound_labels = _compound_selector_labels(target_selected_metrics)
    candidate_interactions = _md_selection_candidate_interactions(
        str(run_root.resolve()), target_selected_jobs
    )
    if candidate_interactions.empty:
        st.info(
            "Run interaction analysis on the campaign predictions before "
            "ranking MD candidates."
        )
        return
    candidate_interactions = _campaign_interaction_review_atoms(
        candidate_interactions
    )
    candidate_interactions = candidate_interactions.loc[
        candidate_interactions["target_run_id"].astype(str).eq(score_target)
    ]
    candidate_interactions = _md_selection_author_numbered_candidate_rows(
        run_root,
        candidate_interactions,
    )
    md_reference = reference["kind"] == "md"
    scoring_candidate_interactions = candidate_interactions.loc[
        candidate_interactions.get(
            "analysis_engine",
            pd.Series("", index=candidate_interactions.index),
        )
        .astype(str)
        .isin(candidate_scoring_engines)
    ].copy()
    available_candidate_scoring_engines = tuple(
        sorted(
            {
                str(value).strip()
                for value in scoring_candidate_interactions.get(
                    "analysis_engine", pd.Series(dtype=str)
                ).tolist()
                if str(value).strip()
            }
        )
    )
    if scoring_candidate_interactions.empty:
        st.warning(
            "No candidate-pose interaction analysis is available from the "
            f"required detector: {candidate_scoring_label}. Run that "
            "interaction detector for the campaign poses before scoring."
        )
        return
    if len(available_candidate_scoring_engines) < int(detector_support):
        st.warning(
            f"The selected reference requires {int(detector_support)} "
            "matching interaction detectors per contact, but only "
            f"{len(available_candidate_scoring_engines)} are available for "
            "the candidate poses. Scores remain detector-matched and may be "
            "zero until the missing analyses are completed."
        )
    st.caption(
        "Primary reference-similarity scoring is detector-matched: "
        + ", ".join(available_candidate_scoring_engines)
        + (
            f"; at least {int(detector_support)} detectors must reproduce "
            "each interaction."
            if int(detector_support) > 1
            else "."
        )
    )
    if md_reference:
        scoring_candidate_interactions.loc[
            _md_selection_pi_specific_mask(scoring_candidate_interactions),
            "interaction_type",
        ] = "hydrophobic contact"
    scored = score_candidate_poses(
        scoring_candidate_interactions,
        hypothesis,
        minimum_detector_support=int(detector_support),
    )
    if scored.empty:
        st.warning("No campaign poses could be scored against this hypothesis.")
        return
    stereo_input_rows = _md_selection_import_rows(
        run_root,
        target_selected_jobs,
        scored,
        candidate_interactions,
    )
    stereo_status = _md_selection_stereochemistry_status(stereo_input_rows)
    if not stereo_status.empty:
        stereo_status = stereo_status.drop_duplicates(
            ["source_run_id", "pose_id"],
            keep="last",
        )
        scored = scored.merge(
            stereo_status,
            on=["source_run_id", "pose_id"],
            how="left",
        )
    else:
        scored["Source stereochemistry preserved"] = pd.NA
        scored["Stereochemistry validation"] = (
            "Stereochemistry could not be verified"
        )
        scored["Stereochemistry policy version"] = pd.NA
    scored["Source stereochemistry preserved"] = (
        scored["Source stereochemistry preserved"].eq(True)
    )
    stereo_rejected_count = int(
        (~scored["Source stereochemistry preserved"]).sum()
    )
    st.info(
        "Immutable-source stereochemistry gate: "
        f"{int(scored['Source stereochemistry preserved'].sum())} of "
        f"{len(scored)} assessed poses preserve the imported compound "
        f"stereochemistry; {stereo_rejected_count} are excluded from "
        "automatic selection and final-pose alternatives. This hard gate "
        "applies to every docking and cofolding engine."
    )
    reference_bend_index = _md_selection_path_bend_index(
        reference_complex_path
    )
    pose_geometry_rows = (
        candidate_interactions.sort_values(
            ["source_run_id", "pose_id", "analysis_engine"],
            kind="stable",
        )
        .drop_duplicates(["source_run_id", "pose_id"], keep="first")
        .copy()
    )
    pose_geometry_rows["_complex_path"] = (
        _campaign_interaction_complex_paths(
            run_root,
            pose_geometry_rows,
        )
    )
    candidate_bend_indices: dict[tuple[str, str], float | None] = {}
    for geometry_row in pose_geometry_rows.to_dict("records"):
        complex_text = str(geometry_row.get("_complex_path") or "")
        candidate_bend_indices[
            (
                str(geometry_row.get("source_run_id") or ""),
                str(geometry_row.get("pose_id") or ""),
            )
        ] = _md_selection_path_bend_index(
            Path(complex_text) if complex_text else None
        )
    with st.expander(
        "Reference-ligand shape and bend penalty",
        expanded=True,
    ):
        st.caption(
            "The bend index is computed from the heavy-atom principal axes of "
            "the exact 3D ligand pose: 0 is line-like and larger values are "
            "more transversely spread or bent. It is invariant to translation, "
            "rotation and molecular size. This is a geometric selection "
            "heuristic, not an energetic proof that a conformation is "
            "accessible."
        )
        if reference_bend_index is None:
            st.warning(
                "The selected reference complex has no measurable ligand "
                "geometry, so bend-based ranking is unavailable."
            )
        else:
            st.metric(
                "Reference ligand bend index",
                f"{reference_bend_index:.3f}",
            )
        with st.form(
            key=f"campaign_md_selection_bend_form:{score_target}",
            border=False,
        ):
            bend_columns = st.columns((1.2, 1.0, 1.2, 1.0, 0.7))
            use_bend_penalty = bend_columns[0].checkbox(
                "Penalize excess bending",
                value=True,
                disabled=reference_bend_index is None,
                help=(
                    "Penalize only the part of a candidate's bend index that "
                    "exceeds the reference value plus the allowed tolerance."
                ),
            )
            bend_tolerance = bend_columns[1].number_input(
                "Allowed excess bend",
                min_value=0.0,
                max_value=1.0,
                value=0.10,
                step=0.02,
                disabled=not use_bend_penalty,
            )
            bend_penalty_strength = bend_columns[2].number_input(
                "Penalty points per 0.1",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=2.5,
                disabled=not use_bend_penalty,
                help=(
                    "Subtract this many percentage points from the interaction "
                    "similarity for each 0.1 of excess bend."
                ),
            )
            reject_excess_bend = bend_columns[3].checkbox(
                "Hard maximum",
                value=False,
                disabled=not use_bend_penalty,
                help=(
                    "Also exclude poses whose excess bend remains above the "
                    "maximum after applying the tolerance."
                ),
            )
            maximum_excess_bend = bend_columns[4].number_input(
                "Maximum",
                min_value=0.0,
                max_value=1.0,
                value=0.20,
                step=0.02,
                disabled=not use_bend_penalty or not reject_excess_bend,
            )
            st.form_submit_button(
                "Apply bend settings",
                type="primary",
            )
    scored = apply_reference_bend_penalty(
        scored,
        reference_bend_index=reference_bend_index,
        candidate_bend_indices=candidate_bend_indices,
        tolerance=float(bend_tolerance),
        penalty_points_per_0_1=(
            float(bend_penalty_strength) if use_bend_penalty else 0.0
        ),
        hard_maximum_excess=(
            float(maximum_excess_bend)
            if use_bend_penalty and reject_excess_bend
            else None
        ),
    )
    selection_score_column = (
        "Selection score (%)"
        if use_bend_penalty
        else "Reference similarity (%)"
    )
    validation_rows, _, _ = _pose_validation_rows(
        run_root, target_selected_jobs
    )
    if not validation_rows.empty:
        validation_status = (
            validation_rows[["source_run_id", "pose_id", "passed_all"]]
            .drop_duplicates(["source_run_id", "pose_id"], keep="last")
            .rename(columns={"passed_all": "PoseBusters passed"})
        )
        scored = scored.merge(
            validation_status,
            on=["source_run_id", "pose_id"],
            how="left",
        )
    else:
        scored["PoseBusters passed"] = pd.NA
    selection_controls = st.columns(4)
    selection_mode = selection_controls[0].selectbox(
        "Automatic selection",
        ("One per compound", "One per compound and engine"),
        key=f"campaign_md_selection_mode:{score_target}",
    )
    minimum_similarity = selection_controls[1].number_input(
        (
            "Minimum bend-adjusted selection score (%)"
            if use_bend_penalty
            else "Minimum reference similarity (%)"
        ),
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=5.0,
        key=f"campaign_md_selection_similarity:{score_target}",
        help=(
            "When bend penalization is enabled, this threshold applies after "
            "subtracting the geometric penalty. The unmodified interaction "
            "similarity remains visible and is exported separately."
        ),
    )
    require_contacts = selection_controls[2].checkbox(
        "Enforce rows marked Must match",
        value=True,
        key=f"campaign_md_selection_require_contacts:{score_target}",
        help=(
            "When enabled, a candidate missing any hypothesis row marked Must "
            "match is excluded from automatic selection."
        ),
    )
    require_posebusters = selection_controls[3].checkbox(
        "Require PoseBusters pass",
        value=False,
        key=f"campaign_md_selection_require_posebusters:{score_target}",
        disabled=validation_rows.empty,
        help=(
            "When enabled, only poses that passed every applicable PoseBusters "
            "check can be selected automatically."
        ),
    )
    allow_manual_hard_filter_override = st.checkbox(
        "Allow manual selection of poses that fail active hard filters",
        value=False,
        key=f"campaign_md_selection_allow_hard_filter_override:{score_target}",
        help=(
            "Off by default: poses missing a Must-match interaction, below "
            "the score threshold, failing a required PoseBusters check, or "
            "failing the hard bend limit cannot enter the final selectors. "
            "Enable only when you intentionally want an audited manual "
            "override. Immutable-source stereochemistry cannot be overridden."
        ),
    )
    stereo_eligible = scored["Source stereochemistry preserved"].eq(True)
    automatic_source = (
        scored.loc[
            scored["PoseBusters passed"].eq(True) & stereo_eligible
        ].copy()
        if require_posebusters
        else scored.loc[stereo_eligible].copy()
    )
    automatic = select_best_candidates(
        automatic_source,
        mode=selection_mode,
        minimum_similarity=float(minimum_similarity),
        require_required_interactions=require_contacts,
        score_column=selection_score_column,
        eligibility_column=(
            "Bend criterion met"
            if use_bend_penalty and reject_excess_bend
            else None
        ),
    )
    automatic_keys = set(
        zip(
            automatic["source_run_id"].astype(str),
            automatic["pose_id"].astype(str),
            strict=True,
        )
    )
    review = scored.sort_values(
        ["compound_id", selection_score_column],
        ascending=[True, False],
        kind="stable",
    ).copy()
    review_hard_filter_pass = pd.to_numeric(
        review[selection_score_column],
        errors="coerce",
    ).fillna(0.0).ge(float(minimum_similarity))
    review_hard_filter_pass &= (
        review["Source stereochemistry preserved"].eq(True)
    )
    if require_contacts:
        review_hard_filter_pass &= (
            review["Required interactions met"]
            .astype(str)
            .str.lower()
            .eq("true")
        )
    if require_posebusters:
        review_hard_filter_pass &= (
            review["PoseBusters passed"]
            .astype(str)
            .str.lower()
            .eq("true")
        )
    if use_bend_penalty and reject_excess_bend:
        review_hard_filter_pass &= (
            review["Bend criterion met"]
            .astype(str)
            .str.lower()
            .eq("true")
        )
    review.insert(
        0,
        "Passes active hard filters",
        review_hard_filter_pass.astype(bool),
    )
    review_keys = list(
        zip(
            review["source_run_id"].astype(str),
            review["pose_id"].astype(str),
            strict=True,
        )
    )
    review.insert(
        0,
        "Push to Complex Dataset",
        [key in automatic_keys for key in review_keys],
    )
    review["compound_name"] = (
        review["compound_id"]
        .astype(str)
        .map(compound_names)
        .fillna("")
    )
    review_leading_columns = [
        "Push to Complex Dataset",
        "Passes active hard filters",
        "compound_name",
        "compound_id",
    ]
    review = review[
        review_leading_columns
        + [
            column
            for column in review.columns
            if column not in review_leading_columns
        ]
    ]
    pose_editor_digest = hashlib.sha1(
        review[
            [
                "source_run_id",
                "pose_id",
                "Reference similarity (%)",
                "Selection score (%)",
                "Candidate pose bend index",
                "Required interactions met",
                "Source stereochemistry preserved",
                "Passes active hard filters",
            ]
        ]
        .astype(str)
        .to_csv(index=False)
        .encode("utf-8")
        + repr(sorted(automatic_keys)).encode("utf-8")
        + hypothesis.fillna("")
        .astype(str)
        .to_csv(index=False)
        .encode("utf-8")
        + repr(
            (
                reference_key,
                evidence_mode,
                tuple(candidate_scoring_engines),
                int(detector_support),
                selection_mode,
                float(minimum_similarity),
                bool(require_contacts),
                bool(require_posebusters),
                bool(use_bend_penalty),
                float(bend_tolerance),
                float(bend_penalty_strength),
                bool(reject_excess_bend),
                float(maximum_excess_bend),
            )
        ).encode("utf-8")
    ).hexdigest()[:12]
    st.markdown("#### Ranked poses and manual review")
    st.caption(
        "Automatic selection initializes exactly one immutable source pose per "
        "compound in One per compound mode. Adjust the checkboxes if needed, "
        "then apply once to rebuild the grouped inspection."
    )
    with st.form(
        key=(
            f"campaign_md_selection_pose_form:{score_target}:"
            f"{pose_editor_digest}"
        ),
        border=False,
    ):
        edited = st.data_editor(
            review,
            hide_index=True,
            width="stretch",
            disabled=[
                column for column in review
                if column != "Push to Complex Dataset"
            ],
            column_config={
                "Push to Complex Dataset": st.column_config.CheckboxColumn(),
                "compound_name": st.column_config.TextColumn(
                    "Compound name"
                ),
                "compound_id": st.column_config.TextColumn("Compound ID"),
                "Passes active hard filters": (
                    st.column_config.CheckboxColumn(
                        help=(
                            "True only when this exact pose passes every "
                            "currently active mandatory selection rule."
                        )
                    )
                ),
                "Source stereochemistry preserved": (
                    st.column_config.CheckboxColumn(
                        help=(
                            "Hard identity gate: the exact 3D pose must retain "
                            "the stereochemistry of the immutable imported "
                            "compound. This cannot be manually overridden."
                        )
                    )
                ),
                "Stereochemistry validation": st.column_config.TextColumn(
                    width="large"
                ),
                "Reference similarity (%)": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
                "Selection score (%)": st.column_config.ProgressColumn(
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f%%",
                    help=(
                        "Interaction similarity after the optional "
                        "reference-ligand bend penalty."
                    ),
                ),
                "Reference ligand bend index": st.column_config.NumberColumn(
                    format="%.3f"
                ),
                "Candidate pose bend index": st.column_config.NumberColumn(
                    format="%.3f"
                ),
                "Excess bend index": st.column_config.NumberColumn(
                    format="%.3f"
                ),
                "Matched reference interactions": st.column_config.TextColumn(
                    width="large"
                ),
                "Missing required interactions": st.column_config.TextColumn(
                    width="large"
                ),
            },
            key=(
                f"campaign_md_selection_pose_editor:{score_target}:"
                f"{pose_editor_digest}"
            ),
        )
        st.form_submit_button(
            "Apply pose selections",
            type="primary",
        )
    selected_review = edited.loc[
        edited["Push to Complex Dataset"].astype(bool)
    ].copy()
    stereo_blocked_rows = selected_review.loc[
        ~selected_review["Source stereochemistry preserved"].eq(True)
    ]
    if not stereo_blocked_rows.empty:
        st.error(
            f"{len(stereo_blocked_rows)} checked pose(s) were removed because "
            "their exact 3D stereochemistry does not match the immutable "
            "source compound. This identity gate cannot be overridden."
        )
        selected_review = selected_review.loc[
            selected_review["Source stereochemistry preserved"].eq(True)
        ].copy()
    blocked_manual_rows = selected_review.loc[
        ~selected_review["Passes active hard filters"].astype(bool)
    ]
    if (
        not allow_manual_hard_filter_override
        and not blocked_manual_rows.empty
    ):
        st.warning(
            f"{len(blocked_manual_rows)} checked pose(s) were not carried "
            "forward because they fail active hard filters. Enable the "
            "explicit manual-override option if this is intentional."
        )
        selected_review = selected_review.loc[
            selected_review["Passes active hard filters"].astype(bool)
        ].copy()
    shown_compounds = set(selected_review["compound_id"].astype(str))
    automatic_compounds = set(automatic["compound_id"].astype(str))
    compound_audit_rows: list[dict] = []
    for compound_id, compound_rows in review.groupby(
        "compound_id",
        sort=False,
    ):
        ranked_compound = compound_rows.sort_values(
            [
                selection_score_column,
                "Reference similarity (%)",
                "Interaction tool count",
            ],
            ascending=[False, False, False],
            kind="stable",
        )
        best_pose = ranked_compound.iloc[0]
        displayed_rows = selected_review.loc[
            selected_review["compound_id"].astype(str).eq(str(compound_id))
        ]
        score_pass = pd.to_numeric(
            compound_rows[selection_score_column],
            errors="coerce",
        ).fillna(0.0).ge(float(minimum_similarity))
        required_pass = (
            compound_rows["Required interactions met"]
            .astype(str)
            .str.lower()
            .eq("true")
        )
        posebusters_pass = (
            compound_rows["PoseBusters passed"]
            .astype(str)
            .str.lower()
            .eq("true")
        )
        bend_pass = (
            compound_rows["Bend criterion met"]
            .astype(str)
            .str.lower()
            .eq("true")
        )
        stereo_pass = (
            compound_rows["Source stereochemistry preserved"].eq(True)
        )
        all_filters = score_pass & stereo_pass
        if require_contacts:
            all_filters &= required_pass
        if require_posebusters:
            all_filters &= posebusters_pass
        if use_bend_penalty and reject_excess_bend:
            all_filters &= bend_pass
        reasons: list[str] = []
        if not stereo_pass.any():
            reasons.append(
                "no pose preserved the immutable source stereochemistry"
            )
        if not score_pass.any():
            reasons.append(
                f"no pose reached {float(minimum_similarity):g}% "
                "selection score"
            )
        if require_contacts and not required_pass.any():
            reasons.append("no pose reproduced every Must match interaction")
        if require_posebusters and not posebusters_pass.any():
            reasons.append("no pose passed PoseBusters")
        if (
            use_bend_penalty
            and reject_excess_bend
            and not bend_pass.any()
        ):
            reasons.append("all poses exceeded the hard bend maximum")
        if not all_filters.any() and not reasons:
            reasons.append(
                "no single pose passed all enabled filters simultaneously"
            )
        compound_text = str(compound_id)
        is_shown = compound_text in shown_compounds
        was_automatic = compound_text in automatic_compounds
        if is_shown and was_automatic:
            status = "Shown — automatic selection"
        elif is_shown:
            status = "Shown — manual inclusion"
        elif was_automatic:
            status = "Not shown — removed in manual review"
            reasons = ["no Push to Complex Dataset row remains checked"]
        else:
            status = "Not shown — no automatically eligible pose"
        compound_audit_rows.append(
            {
                "Compound name": compound_names.get(compound_text, ""),
                "Compound ID": compound_text,
                "Status": status,
                "Shown in 2D/3D": is_shown,
                "Automatically selected": was_automatic,
                "Poses assessed": len(compound_rows),
                "Poses passing all active filters": int(all_filters.sum()),
                "Selected pose engine": (
                    "; ".join(
                        displayed_rows["source_engine"]
                        .fillna("")
                        .astype(str)
                        .tolist()
                    )
                    if not displayed_rows.empty
                    else ""
                ),
                "Selected pose": (
                    "; ".join(
                        displayed_rows["prediction"]
                        .fillna("")
                        .astype(str)
                        .tolist()
                    )
                    if not displayed_rows.empty
                    else ""
                ),
                "Selected pose preserves stereochemistry": (
                    bool(
                        displayed_rows[
                            "Source stereochemistry preserved"
                        ].eq(True).all()
                    )
                    if not displayed_rows.empty
                    else False
                ),
                "Highest-scoring assessed pose engine": str(
                    best_pose.get("source_engine") or ""
                ),
                "Highest-scoring assessed pose": str(
                    best_pose.get("prediction") or ""
                ),
                "Highest-scoring pose excluded by stereochemistry": not bool(
                    best_pose.get("Source stereochemistry preserved")
                ),
                "Stereochemistry-preserving poses": int(stereo_pass.sum()),
                "Stereochemistry-rejected poses": int((~stereo_pass).sum()),
                "Best hypothesis correspondence (%)": best_pose.get(
                    "Reference similarity (%)", ""
                ),
                "Best bend-adjusted selection score (%)": best_pose.get(
                    "Selection score (%)", ""
                ),
                "Why not shown": "; ".join(reasons) if not is_shown else "",
                "Best-pose missing Must match interactions": str(
                    best_pose.get("Missing required interactions") or ""
                ),
            }
        )
    compound_selection_audit = pd.DataFrame(compound_audit_rows)
    with st.expander(
        "Compound selection audit — including compounds not shown below",
        expanded=len(shown_compounds) < review["compound_id"].nunique(),
    ):
        st.caption(
            "Every compound in the active campaign scope appears here. The "
            "2D/3D section below contains only compounds with a checked Push "
            "to Complex Dataset row."
        )
        st.dataframe(
            compound_selection_audit,
            hide_index=True,
            width="stretch",
            column_config={
                "Shown in 2D/3D": st.column_config.CheckboxColumn(),
                "Automatically selected": st.column_config.CheckboxColumn(),
                "Selected pose preserves stereochemistry": (
                    st.column_config.CheckboxColumn(
                        help=(
                            "The exact selected 3D pose retains the immutable "
                            "source-SMILES stereochemistry."
                        )
                    )
                ),
                "Highest-scoring pose excluded by stereochemistry": (
                    st.column_config.CheckboxColumn(
                        help=(
                            "The highest interaction/bend score belonged to "
                            "an invalid stereoisomer, so a lower-ranked valid "
                            "pose was selected instead."
                        )
                    )
                ),
                "Best hypothesis correspondence (%)": (
                    st.column_config.ProgressColumn(
                        min_value=0.0,
                        max_value=100.0,
                        format="%.1f%%",
                    )
                ),
                "Best bend-adjusted selection score (%)": (
                    st.column_config.ProgressColumn(
                        min_value=0.0,
                        max_value=100.0,
                        format="%.1f%%",
                    )
                ),
                "Why not shown": st.column_config.TextColumn(width="large"),
                "Best-pose missing Must match interactions": (
                    st.column_config.TextColumn(width="large")
                ),
            },
        )
    selection_group_columns = ["compound_id"]
    if selection_mode == "One per compound and engine":
        selection_group_columns.append("source_engine")
    alternative_count = st.number_input(
        "Best poses offered for each final selection",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        key=f"campaign_md_selection_alternative_count:{score_target}",
        help=(
            "Each final-pose selector is ranked by the active selection score. "
            "The pose displayed in its linked 2D/3D view is the exact pose "
            "that will be written to the Complex Dataset."
        ),
    )
    choice_store_key = (
        f"campaign_md_selection_final_choice_store:{score_target}:"
        f"{selection_mode}"
    )
    st.session_state.setdefault(choice_store_key, {})
    active_choice_digests: list[str] = []
    with st.expander(
        "Choose and inspect the final poses in 3D and 2D",
        expanded=not selected_review.empty,
    ):
        st.info(
            "The stored selectors are authoritative for every selected "
            "compound. Inspect one compound at a time below; changing its "
            "option redraws that exact pose and updates the exported choice. "
            "Changing the hypothesis or scoring configuration resets every "
            "selector to the newly ranked #1 pose."
        )
        if selected_review.empty:
            st.info(
                "Select at least one Push to Complex Dataset row and apply the "
                "table to create final-pose selectors."
            )
        else:
            selected_groups = selected_review.drop_duplicates(
                selection_group_columns,
                keep="first",
            )
            group_configs: list[dict] = []
            for _, selected_group in selected_groups.iterrows():
                group_mask = pd.Series(True, index=review.index)
                for column in selection_group_columns:
                    group_mask &= review[column].astype(str).eq(
                        str(selected_group[column])
                    )
                alternatives_pool = review.loc[group_mask].copy()
                if not allow_manual_hard_filter_override:
                    alternatives_pool = alternatives_pool.loc[
                        alternatives_pool[
                            "Passes active hard filters"
                        ].astype(bool)
                    ].copy()
                alternatives_pool = alternatives_pool.loc[
                    alternatives_pool[
                        "Source stereochemistry preserved"
                    ].eq(True)
                ].copy()
                alternatives = (
                    alternatives_pool
                    .sort_values(
                        [
                            selection_score_column,
                            "Reference similarity (%)",
                            "Interaction tool count",
                            "source_engine",
                            "source_run_id",
                            "pose_id",
                        ],
                        ascending=[False, False, False, True, True, True],
                        kind="stable",
                    )
                    .head(int(alternative_count))
                    .copy()
                )
                if alternatives.empty:
                    continue
                compound_id = str(selected_group["compound_id"])
                group_title = compound_labels.get(compound_id, compound_id)
                if "source_engine" in selection_group_columns:
                    group_title += f" · {selected_group['source_engine']}"
                group_digest = hashlib.sha1(
                    repr(
                        (
                            pose_editor_digest,
                            tuple(
                                str(selected_group[column])
                                for column in selection_group_columns
                            ),
                        )
                    ).encode("utf-8")
                ).hexdigest()[:10]
                active_choice_digests.append(group_digest)
                automatic_group = automatic.copy()
                for column in selection_group_columns:
                    automatic_group = automatic_group.loc[
                        automatic_group[column].astype(str).eq(
                            str(selected_group[column])
                        )
                    ]
                group_configs.append(
                    {
                        "alternatives": alternatives,
                        "selected_group": selected_group.to_dict(),
                        "automatic_group": automatic_group,
                        "group_title": group_title,
                        "group_digest": group_digest,
                    }
                )

            # Preserve an authoritative choice for every selected compound,
            # but mount the expensive py3Dmol inspection UI for only one at a
            # time. Mounting one WebGL canvas per compound exhausts Chromium's
            # graphics contexts and corrupts every 3D view on the page.
            stored_choices = st.session_state.setdefault(
                choice_store_key, {}
            )
            for config in group_configs:
                alternatives = config["alternatives"]
                option_rows = {
                    (
                        str(row.get("source_run_id") or ""),
                        str(row.get("pose_id") or ""),
                    ): row
                    for row in alternatives.to_dict("records")
                }
                options = list(option_rows)
                digest = str(config["group_digest"])
                selected_group = config["selected_group"]
                stored_row = stored_choices.get(digest, {})
                stored_key = (
                    str(stored_row.get("source_run_id") or ""),
                    str(stored_row.get("pose_id") or ""),
                )
                selected_key = (
                    str(selected_group.get("source_run_id") or ""),
                    str(selected_group.get("pose_id") or ""),
                )
                chosen_key = (
                    stored_key
                    if stored_key in option_rows
                    else selected_key
                    if selected_key in option_rows
                    else options[0]
                )
                stored_choices[digest] = _md_selection_final_choice_record(
                    option_rows[chosen_key],
                    config["automatic_group"],
                    alternative_rank=options.index(chosen_key) + 1,
                    alternative_count=len(options),
                    selection_score_column=selection_score_column,
                    require_contacts=require_contacts,
                    require_posebusters=require_posebusters,
                    use_bend_penalty=use_bend_penalty,
                    reject_excess_bend=reject_excess_bend,
                    minimum_similarity=float(minimum_similarity),
                )
            st.session_state[choice_store_key] = stored_choices

            if group_configs:
                config_by_digest = {
                    str(config["group_digest"]): config
                    for config in group_configs
                }
                inspect_key = (
                    "campaign_md_selection_inspected_group:"
                    f"{score_target}:{selection_mode}"
                )
                if st.session_state.get(inspect_key) not in config_by_digest:
                    st.session_state[inspect_key] = next(iter(config_by_digest))
                inspect_digest = st.selectbox(
                    "Compound/selection to inspect",
                    list(config_by_digest),
                    format_func=lambda digest: str(
                        config_by_digest[digest]["group_title"]
                    ),
                    key=inspect_key,
                    help=(
                        "Only this compound's 3D canvas is mounted. Final "
                        "choices for all other selected compounds remain "
                        "stored and are still exported."
                    ),
                )
                active_config = config_by_digest[inspect_digest]
                _render_md_final_pose_selector(
                    run_root=run_root,
                    target_row=target_row,
                    alternatives=active_config["alternatives"],
                    selected_group=active_config["selected_group"],
                    automatic_group=active_config["automatic_group"],
                    candidate_interactions=scoring_candidate_interactions,
                    hypothesis=hypothesis,
                    selection_score_column=selection_score_column,
                    require_contacts=require_contacts,
                    require_posebusters=require_posebusters,
                    use_bend_penalty=use_bend_penalty,
                    reject_excess_bend=reject_excess_bend,
                    minimum_similarity=float(minimum_similarity),
                    detector_support=int(detector_support),
                    score_target=score_target,
                    top_interactions=int(top_interactions),
                    md_reference=md_reference,
                    group_title=str(active_config["group_title"]),
                    group_digest=str(active_config["group_digest"]),
                    choice_store_key=choice_store_key,
                )
            else:
                st.info(
                    "No stereochemistry-preserving pose alternatives are "
                    "available for the selected rows."
                )
    stored_final_choices = st.session_state.get(choice_store_key, {})
    st.session_state[choice_store_key] = {
        digest: stored_final_choices[digest]
        for digest in active_choice_digests
        if digest in stored_final_choices
    }
    final_choice_rows = [
        st.session_state[choice_store_key][digest]
        for digest in active_choice_digests
        if digest in st.session_state[choice_store_key]
    ]
    chosen = pd.DataFrame(final_choice_rows)
    if not chosen.empty:
        chosen = chosen.drop(
            columns=["Push to Complex Dataset"],
            errors="ignore",
        )
        chosen["Selected rank"] = (
            chosen.sort_values(
                selection_score_column, ascending=False, kind="stable"
            )
            .groupby("compound_id")
            .cumcount()
            + 1
        )
        chosen["Selection status"] = chosen["Selection origin"]
    st.caption(
        f"{len(chosen)} exact final pose(s) selected. Every exported structure "
        "uses the stored immutable pose shown when that compound is inspected."
    )
    target_name = str(
        target_row.get("coordinate_target_key")
        or target_row.get("target")
        or "campaign"
    )
    dataset_name = st.text_input(
        "Complex Dataset name",
        value=f"{target_name}-reference-selected-for-MD",
        key=f"campaign_md_selection_dataset_name:{score_target}",
    ).strip()
    if st.button(
        "Create Complex Dataset from selected poses",
        type="primary",
        disabled=chosen.empty or not dataset_name,
        key=f"campaign_md_selection_create:{score_target}",
    ):
        try:
            selection_rows = _md_selection_import_rows(
                run_root,
                target_selected_jobs,
                chosen,
                candidate_interactions,
            )
            if compound_names:
                selection_rows.insert(
                    2,
                    "Compound name",
                    selection_rows["Compound"]
                    .astype(str)
                    .map(compound_names)
                    .fillna(""),
                )
            if len(selection_rows) != len(chosen):
                raise ValueError(
                    "At least one selected pose has no immutable interaction "
                    "evidence path"
                )
            provenance_row = {
                "Selection algorithm": (
                    "reference-hypothesis-exact-pose-selection-v5"
                ),
                "Hypothesis source": reference["label"],
                "Reference evidence kind": reference["kind"],
                "Static evidence model": evidence_mode,
                "Reference target run ID": reference["target_run_id"],
                "Reference analysis run ID": reference.get(
                    "analysis_run_id", ""
                ),
                "Reference MD workflow run ID": reference.get(
                    "workflow_run_id", ""
                ),
                "Reference pose ID": reference_pose_id,
                "Region selection": "Per interaction in hypothesis table",
                "Minimum MD occupancy": (
                    float(occupancy) if reference["kind"] == "md" else ""
                ),
                "Maximum automatically enabled protein residues": int(
                    top_interactions
                ),
                "Automatically enabled protein residues": len(
                    enabled_residues
                ),
                "Automatic residue criterion policy": (
                    "specific-interaction residues first; one criterion per "
                    "residue; generic contacts fill remaining slots"
                ),
                "Static native geometry is trajectory MD": False,
                "Water bridges included": False,
                "Candidate scoring detector policy": "Match reference detector",
                "Candidate scoring interaction engines": ", ".join(
                    candidate_scoring_engines
                ),
                "Minimum matching detectors": int(detector_support),
                "Minimum selection score (%)": float(minimum_similarity),
                "Score used for ranking": selection_score_column,
                "Reference ligand bend index": (
                    reference_bend_index
                    if reference_bend_index is not None
                    else ""
                ),
                "Excess bend penalty enabled": bool(use_bend_penalty),
                "Allowed excess bend": (
                    float(bend_tolerance) if use_bend_penalty else ""
                ),
                "Bend penalty points per 0.1": (
                    float(bend_penalty_strength)
                    if use_bend_penalty
                    else ""
                ),
                "Hard maximum excess bend enabled": bool(
                    use_bend_penalty and reject_excess_bend
                ),
                "Maximum excess bend": (
                    float(maximum_excess_bend)
                    if use_bend_penalty and reject_excess_bend
                    else ""
                ),
                "Automatic selection mode": selection_mode,
                "Require all required contacts": require_contacts,
                "Require PoseBusters pass": require_posebusters,
                "Require immutable source stereochemistry": True,
                "Stereochemistry policy version":
                    STEREOCHEMISTRY_POLICY_VERSION,
                "Alternatives offered per final selector": int(
                    alternative_count
                ),
                "Campaign target run ID": score_target,
            }
            candidate_ranking = review.drop(
                columns=["Push to Complex Dataset"],
                errors="ignore",
            ).copy()
            candidate_ranking_keys = list(
                zip(
                    candidate_ranking["source_run_id"].astype(str),
                    candidate_ranking["pose_id"].astype(str),
                    strict=True,
                )
            )
            final_rows_by_key = {
                (
                    str(row.get("source_run_id") or ""),
                    str(row.get("pose_id") or ""),
                ): row
                for row in chosen.to_dict("records")
            }
            candidate_ranking.insert(
                0,
                "Final selection",
                [key in final_rows_by_key for key in candidate_ranking_keys],
            )
            candidate_ranking.insert(
                1,
                "Automatic selection",
                [key in automatic_keys for key in candidate_ranking_keys],
            )
            candidate_ranking["Final selection origin"] = [
                str(final_rows_by_key.get(key, {}).get("Selection origin") or "")
                for key in candidate_ranking_keys
            ]
            reference_evidence_export = (
                reference_rows.copy()
                if reference["kind"] == "static"
                else visual_rows.copy()
            ).fillna("")
            if not reference_evidence_export.empty:
                reference_evidence_export["Protein residue"] = (
                    reference_evidence_export.apply(
                        interaction_residue,
                        axis=1,
                    )
                )
                reference_export_columns = [
                    column
                    for column in (
                        "analysis_engine",
                        "interaction_type",
                        "Protein residue",
                        "protein_atom_scope",
                        "protein_atom_name",
                        "ligand_atom_name",
                        "distance_angstrom",
                        "angle_degree",
                        "Importance",
                        "Reference support",
                        "Reference evidence",
                    )
                    if column in reference_evidence_export
                ]
                reference_evidence_export = reference_evidence_export[
                    reference_export_columns
                ]
            source_bytes = _md_selection_hypothesis_workbook(
                selection_rows,
                hypothesis,
                pd.DataFrame([provenance_row]),
                candidate_ranking=candidate_ranking,
                reference_evidence=reference_evidence_export,
                selection_audit=compound_selection_audit,
            )
            validated = validate_complex_dataset_rows(selection_rows)
            invalid = validated.loc[validated["Validation"].ne("Valid")]
            if not invalid.empty:
                raise ValueError(
                    "; ".join(
                        sorted(set(invalid["Validation"].astype(str)))
                    )
                )
            dataset = create_complex_dataset(
                name=dataset_name,
                source_filename="reference-derived-md-selection.xlsx",
                source_bytes=source_bytes,
                sheet_name="Selection",
                selected_rows=validated,
                selection_metadata={
                    "selection_algorithm": provenance_row[
                        "Selection algorithm"
                    ],
                    "configuration": provenance_row,
                    "hypothesis_definition": hypothesis.fillna("").to_dict(
                        "records"
                    ),
                    "final_pose_decisions": chosen.fillna("").to_dict(
                        "records"
                    ),
                    "compound_selection_audit": (
                        compound_selection_audit.fillna("").to_dict("records")
                    ),
                },
            )
        except (OSError, TypeError, ValueError) as exc:
            st.error(f"Could not create Complex Dataset: {exc}")
        else:
            st.success(
                f"Created {dataset_name} with "
                f"{dataset.metadata.get('complex_count')} exact MD-selectable "
                "complexes."
            )
            st.cache_data.clear()


@st.fragment
def _render_target_structural_explorer(frame: pd.DataFrame) -> None:
    """Keep matrix and 3D representations mounted for instant tab switching."""
    focused = _render_linked_3d_focus(frame)
    if focused.empty:
        st.info("Select at least one target and compound.")
        return
    gnina_criterion_label = ""
    if focused["engine"].astype(str).eq("GNINA").any():
        gnina_criterion_label = st.segmented_control(
            "GNINA pose-selection criterion",
            (
                "Both rankings",
                "CNN pose score",
                "Empirical / Vina score",
            ),
            default="Both rankings",
            key="campaign_viewer_gnina_pose_criterion",
            help=(
                "The same GNINA pose selection is used in the RMSD matrices "
                "and the 3D structures."
            ),
        )
    rmsd_tab, structures_tab = st.tabs(
        ("RMSD & pose agreement", "3D structures")
    )
    with rmsd_tab:
        st.markdown("## RMSD and pose agreement")
        st.caption(
            "Every selected pose is reported, including a reason when it "
            "cannot enter an RMSD matrix."
        )
        _render_rmsd_analysis(
            focused,
            gnina_criterion_label=gnina_criterion_label,
        )
    with structures_tab:
        st.markdown("## 3D structures")
        st.caption(
            "This view uses the same targets, compounds, repetitions, target "
            "presentation, and GNINA ranking as the RMSD tab. Engines may be "
            "hidden locally below without changing the RMSD matrices."
        )
        _render_target_viewer_context(
            focused,
            gnina_criterion_label=gnina_criterion_label,
            available_frame=frame,
        )


def target_compound_matrix(selected_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return cross-engine percentiles by physical target and compound."""
    consensus = consensus_percentiles(selected_metrics)
    if consensus.empty:
        return pd.DataFrame()
    return (
        consensus.groupby(["target_run_id", "target", "candidate_id"])[
            "Within-campaign percentile"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "Mean percentile",
                "std": "Sample SD",
                "count": "Contributing campaigns",
            }
        )
    )


def _render_target_compound_explorer(selected_metrics: pd.DataFrame) -> None:
    matrix = target_compound_matrix(selected_metrics)
    if matrix.empty:
        st.info("No compatible primary metrics are available for an entity matrix.")
        return
    matrix["Sample SD"] = matrix["Sample SD"].fillna(0.0)
    st.markdown("#### Target × compound result matrix")
    st.caption(
        "Cells are direction-aware within-campaign percentiles, not pooled raw "
        "scores. Select a target and compound below to inspect the exact engine "
        "and replicate rows contributing to that entity pair."
    )
    heatmap = (
        alt.Chart(matrix)
        .mark_rect()
        .encode(
            x=alt.X(
                "target:N",
                title="Prepared target",
                axis=alt.Axis(labelAngle=-25),
            ),
            y=alt.Y("candidate_id:N", title="Compound"),
            color=alt.Color(
                "Mean percentile:Q",
                scale=alt.Scale(domain=[0, 1], scheme="viridis"),
                title="Mean percentile",
            ),
            tooltip=[
                alt.Tooltip("target:N", title="Target"),
                alt.Tooltip("candidate_id:N", title="Compound"),
                alt.Tooltip("Mean percentile:Q", format=".3f"),
                alt.Tooltip("Sample SD:Q", format=".3f"),
                "Contributing campaigns:Q",
            ],
        )
        .properties(height=max(260, 28 * matrix["candidate_id"].nunique()))
    )
    st.altair_chart(heatmap, width="stretch")

    target_rows = matrix[["target_run_id", "target"]].drop_duplicates()
    target_ids = target_rows["target_run_id"].astype(str).tolist()
    target_labels = dict(
        zip(
            target_rows["target_run_id"].astype(str),
            target_rows["target"].astype(str),
        )
    )
    drill_columns = st.columns(2)
    selected_target = drill_columns[0].selectbox(
        "Drill down: target",
        target_ids,
        format_func=lambda value: target_labels.get(value, value),
        key="campaign_entity_drill_target",
    )
    compound_options = (
        matrix.loc[
            matrix["target_run_id"].astype(str).eq(str(selected_target)),
            "candidate_id",
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    compound_labels = _compound_selector_labels(selected_metrics)
    selected_compound = drill_columns[1].selectbox(
        "Drill down: compound",
        compound_options,
        format_func=lambda value: compound_labels.get(str(value), str(value)),
        key="campaign_entity_drill_compound",
    )
    detailed = selected_metrics.loc[
        selected_metrics["target_run_id"].astype(str).eq(str(selected_target))
        & selected_metrics["candidate_id"].astype(str).eq(str(selected_compound))
    ].copy()
    metric_columns = list(
        dict.fromkeys(
            metric
            for definitions in ENGINE_METRICS.values()
            for metric, _, _ in definitions
            if metric in detailed
        )
    )
    identity_columns = [
        column
        for column in (
            "result",
            "engine",
            "campaign",
            "target",
            "candidate_id",
            "replicate",
            "model_seed",
            "prediction_id",
        )
        if column in detailed
    ]
    st.dataframe(
        detailed[identity_columns + metric_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "result": st.column_config.LinkColumn("Result", display_text="Open")
        },
    )
    st.caption(
        "The existing 3D viewer below provides target and compound matrices, "
        "pose agreement, and per-engine structural inspection."
    )


def render() -> None:
    requested_purpose = str(
        st.query_params.get("campaign_purpose", "")
    ).strip()
    target_ligand_comparison = (
        requested_purpose == "target_ligand_redocking_refolding"
    )
    st.title(
        "Target-Ligand Redocking / Refolding Comparison"
        if target_ligand_comparison
        else "Compound Campaign Comparison"
    )
    st.caption(
        (
            "Compare the coordinate-matched ligand across one or more prepared "
            "versions of a target complex."
        )
        if target_ligand_comparison
        else (
            "Compare all completed target-based docking, cofolding and rescoring "
            "campaigns for one imported compound dataset while preserving each "
            "engine's native metric meaning."
        )
    )
    run_root = runs_root()
    requested_dataset_deep_link = str(
        st.query_params.get("dataset_run_id", "")
    ).strip()
    campaigns, metrics = load_campaign_comparison_data(
        str(run_root),
        _campaign_runs_revision(run_root),
        requested_dataset_deep_link,
    )
    if campaigns.empty:
        st.info("No completed target campaigns with normalized metrics.")
        st.page_link(
            "./discover-docking",
            label="Open Docking / Cofolding",
        )
        return

    requested_collection_id = str(
        st.query_params.get("analysis_set_id", "")
        or st.query_params.get("collection_id", "")
    ).strip()
    requested_target_id = str(
        st.query_params.get("target_run_id", "")
    ).strip()
    requested_launch_id = str(
        st.query_params.get("launch_campaign_id", "")
    ).strip()
    deep_link_signature = "|".join(
        (requested_target_id, requested_launch_id, requested_purpose)
    )
    if (
        deep_link_signature.strip("|")
        and st.session_state.get("_campaign_deep_link_applied")
        != deep_link_signature
    ):
        _reset_dataset_dependent_controls()
        st.session_state.pop("campaign_compare_dataset", None)
        st.session_state["_campaign_deep_link_applied"] = (
            deep_link_signature
        )
    campaign_scope = _deep_link_campaign_scope(
        campaigns,
        target_run_id=(
            "" if target_ligand_comparison else requested_target_id
        ),
        launch_campaign_id=(
            "" if target_ligand_comparison else requested_launch_id
        ),
        campaign_purpose=requested_purpose,
    )
    if campaign_scope.empty:
        st.warning(
            "The requested target campaign is unavailable; showing all "
            "completed campaigns instead."
        )
        campaign_scope = campaigns
    collection_records = _comparison_collections(runs_root())
    requested_collection = next(
        (
            row
            for row in collection_records
            if row["collection_id"] == requested_collection_id
        ),
        None,
    )
    collection_selection = (
        requested_collection["selection"]
        if requested_collection is not None
        else {}
    )
    if requested_collection_id and requested_collection is None:
        st.warning(
            "The requested Analysis Set is unavailable."
        )
    if (
        requested_collection is not None
        and st.session_state.get("_campaign_collection_applied")
        != requested_collection_id
    ):
        _reset_dataset_dependent_controls()
        st.session_state.pop("campaign_compare_dataset", None)
        st.session_state["_campaign_collection_applied"] = (
            requested_collection_id
        )
    requested_dataset = str(
        collection_selection.get("dataset_run_id")
        or st.query_params.get("dataset_run_id", "")
    ).strip()
    if not requested_dataset and (requested_target_id or requested_launch_id):
        if not campaign_scope.empty:
            requested_dataset = str(
                campaign_scope.iloc[0]["dataset_run_id"]
            )
    if requested_collection is not None:
        st.markdown(f"### Analysis Set · {requested_collection['name']}")
        if requested_collection.get("description"):
            st.caption(str(requested_collection["description"]))
    st.markdown("### Comparison scope")
    st.caption(
        "These filters are authoritative for every perspective below. Local "
        "controls only choose what to focus on or how to display repetitions."
    )
    analysis_scope_edit = (
        st.toggle(
            "Edit Analysis Set filters",
            value=False,
            key="campaign_edit_analysis_set_scope",
            help=(
                "The saved targets and engines are used automatically. Enable "
                "this only for a temporary narrower view; the saved Analysis "
                "Set is not changed."
            ),
        )
        if requested_collection is not None
        else None
    )
    if target_ligand_comparison:
        selected_dataset = ""
        dataset_labels: dict[str, str] = {}
        (
            selected_targets,
            selected_launches,
            selected_target_launch_pairs,
        ) = (
            _render_target_ligand_launch_selector(
                campaign_scope,
                requested_target_id=requested_target_id,
                requested_launch_id=requested_launch_id,
                collection_selection=collection_selection,
                scope_edit_enabled=analysis_scope_edit,
            )
        )
        target_campaigns = campaign_scope.loc[
            campaign_scope["target_run_id"].isin(selected_targets)
        ].copy()
        primary_target_campaigns = target_campaigns.loc[
            ~target_campaigns["engine"].astype(str).str.endswith(
                "rescoring"
            )
        ].copy()
        pair_ids = (
            primary_target_campaigns["launch_campaign_id"].astype(str)
            + "::"
            + primary_target_campaigns["target_run_id"].astype(str)
        )
        launch_campaigns = primary_target_campaigns.loc[
            pair_ids.isin(selected_target_launch_pairs)
        ].copy()
        available_engines = sorted(
            launch_campaigns["engine"].unique()
        )
        requested_engines = {
            str(value)
            for value in collection_selection.get("engines") or []
        }
        default_engines = (
            [
                value
                for value in available_engines
                if value in requested_engines
            ]
            if requested_collection is not None
            else available_engines
        )
        if requested_collection is not None and not analysis_scope_edit:
            selected_engines = default_engines
            st.caption(
                f"Saved scope: {len(selected_targets)} prepared target(s) · "
                f"{len(selected_engines)} engine(s)."
            )
        else:
            _reconcile_cascading_multiselect(
                key="campaign_compare_engines",
                context_key="_campaign_engines_context",
                context=(
                    "target-ligand-launches",
                    "|".join(
                        sorted(
                            str(value) for value in selected_launches
                        )
                    ),
                ),
                options=available_engines,
                defaults=default_engines,
            )
            selected_engines = st.multiselect(
                "Engines",
                available_engines,
                key="campaign_compare_engines",
            )
        selectable = launch_campaigns.loc[
            launch_campaigns["engine"].isin(selected_engines)
        ].copy()
        selected_campaigns = selectable["campaign_id"].tolist()
    else:
        (
            selected_dataset,
            dataset_labels,
            selected_targets,
            selected_launches,
            selected_engines,
            selected_campaigns,
            target_campaigns,
            selectable,
        ) = _render_compound_campaign_selectors(
            campaign_scope,
            requested_dataset=requested_dataset,
            requested_target_id=requested_target_id,
            requested_launch_id=requested_launch_id,
            requested_collection=requested_collection,
            collection_selection=collection_selection,
        )
    _render_prepared_target_code_legend()
    selected_jobs = selectable.loc[
        selectable["campaign_id"].isin(selected_campaigns)
    ].copy()
    selected_metrics = metrics.loc[
        metrics["campaign_id"].isin(selected_campaigns)
    ].copy()
    selected_compound_count = (
        selected_metrics["candidate_id"].astype(str).nunique()
        if "candidate_id" in selected_metrics
        else 0
    )
    st.caption(
        "Active scope · "
        f"{selected_jobs['target_run_id'].nunique()} target(s) · "
        f"{selected_compound_count} compound(s) · "
        f"{selected_jobs['engine'].nunique()} engine(s) · "
        f"{len(selected_jobs)} engine run(s)."
    )
    rescoring_mask = target_campaigns["engine"].astype(str).str.endswith(
        "rescoring"
    )
    if target_ligand_comparison:
        rescoring_pair_ids = (
            target_campaigns["launch_campaign_id"].astype(str)
            + "::"
            + target_campaigns["target_run_id"].astype(str)
        )
        rescoring_mask &= rescoring_pair_ids.isin(
            selected_target_launch_pairs
        )
    else:
        rescoring_mask &= target_campaigns["launch_campaign_id"].isin(
            selected_launches
        )
    linked_rescoring_jobs = target_campaigns.loc[rescoring_mask].copy()
    linked_rescoring_metrics = metrics.loc[
        metrics["campaign_id"].isin(
            linked_rescoring_jobs["campaign_id"]
        )
    ].copy()
    data_metrics = pd.concat(
        [selected_metrics, linked_rescoring_metrics],
        ignore_index=True,
        sort=False,
    )
    export_jobs = pd.concat(
        [selected_jobs, linked_rescoring_jobs],
        ignore_index=True,
        sort=False,
    ).drop_duplicates("campaign_id")
    if (
        requested_collection is not None
        and "campaign_compare_rescoring_runs" not in st.session_state
    ):
        requested_rescoring = {
            str(value)
            for value in collection_selection.get("rescoring_run_ids") or []
        }
        st.session_state["campaign_compare_rescoring_runs"] = [
            value
            for value in linked_rescoring_jobs["campaign_id"].tolist()
            if value in requested_rescoring
        ]
    if selected_jobs.empty or selected_metrics.empty:
        st.info("The selected filters contain no normalized result rows.")
        return

    campaign_shape = classify_campaign_shape(selected_jobs, selected_metrics)
    st.caption(
        f"Analysis shape: {campaign_shape['label']} · "
        f"{campaign_shape['target_count']} target(s) · "
        f"{campaign_shape['compound_count']} compound(s). The available "
        "perspectives adapt to these physical dimensions."
    )
    large_campaign = (
        int(campaign_shape["compound_count"])
        > LARGE_CAMPAIGN_EAGER_RENDER_THRESHOLD
    )
    if large_campaign and not st.toggle(
        "Load full analysis workspace",
        value=False,
        key="campaign_load_full_analysis_workspace",
        help=(
            "The full workspace constructs every Streamlit tab, including "
            "large score charts, pose-validation summaries and interaction "
            "tables. Leave this disabled for a fast campaign overview."
        ),
    ):
        st.info(
            "Fast overview mode is active for this large campaign. Enable "
            "Load full analysis workspace when you need plots, structural "
            "evidence, interaction analysis, exports, or MD selection."
        )
        summary_columns = st.columns(4)
        summary_columns[0].metric("Engine runs", len(selected_jobs))
        summary_columns[1].metric(
            "Engines", selected_jobs["engine"].nunique()
        )
        summary_columns[2].metric(
            "Compounds", selected_metrics["candidate_id"].nunique()
        )
        summary_columns[3].metric("Metric rows", len(selected_metrics))
        coverage = pd.crosstab(
            selected_metrics["candidate_id"],
            selected_metrics["engine"],
        ).reset_index()
        st.markdown("#### Result coverage")
        st.dataframe(coverage, hide_index=True, width="stretch")
        st.markdown("#### Included campaigns")
        st.dataframe(
            selected_jobs[
                [
                    "result",
                    "launch_campaign",
                    "campaign",
                    "target",
                    "dataset",
                    "compound_count",
                    "created_at",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "result": st.column_config.LinkColumn(
                    "Result", display_text="Open"
                ),
                "created_at": st.column_config.DatetimeColumn(
                    "Created", format="YYYY-MM-DD HH:mm"
                ),
            },
        )
        return
    explorer_options = (
        "Target × compound matrix",
        "RMSD & pose agreement",
        "3D structures",
        "MD candidate selection",
    )
    # Streamlit tabs execute every tab body eagerly. Once MD selection is
    # active, render it before creating the campaign tabs so hypothesis edits
    # and pose review do not rebuild unrelated charts, analyses and exports.
    if (
        not target_ligand_comparison
        and st.session_state.get("campaign_explorer_view")
        == "MD candidate selection"
    ):
        st.caption(
            "Fast workspace mode · unrelated campaign panels are not built "
            "while MD candidate selection is active."
        )
        fast_explorer_view = st.segmented_control(
            "Explore selected results",
            explorer_options,
            default="MD candidate selection",
            key="campaign_explorer_view",
        )
        if fast_explorer_view != "MD candidate selection":
            st.rerun()
        _render_md_candidate_selection(
            run_root,
            selected_jobs,
            selected_metrics,
        )
        return
    # A Streamlit ``st.tabs`` block executes every tab body on every rerun.
    # Keep the interactive structure viewer independent of the RMSD, score,
    # interaction and export panels: loading coordinates should not wait for
    # those unrelated analyses to be rebuilt.
    if (
        not target_ligand_comparison
        and st.session_state.get("campaign_explorer_view") == "3D structures"
    ):
        st.caption(
            "Fast 3D workspace mode · score charts, RMSD summaries, and "
            "other analysis panels are not rebuilt while viewing structures."
        )
        fast_explorer_view = st.segmented_control(
            "Explore selected results",
            explorer_options,
            default="3D structures",
            key="campaign_explorer_view",
        )
        if fast_explorer_view != "3D structures":
            st.rerun()
        st.markdown("## 3D structures")
        _render_structure_comparison(selected_metrics)
        return
    (
        overview_tab,
        scores_tab,
        structural_tab,
        explorer_tab,
        workspace_tab,
    ) = st.tabs(campaign_perspective_labels(campaign_shape))
    with scores_tab:
        (
            native_tab,
            correlation_tab,
            rescoring_tab,
            consensus_tab,
        ) = st.tabs(
            ["Native metrics", "Correlations", "Rescoring", "Consensus"]
        )
    with structural_tab:
        pose_validation_tab, interaction_analysis_tab = st.tabs(
            ["Pose validity", "Interactions"]
        )
    viewer_tab = explorer_tab
    with workspace_tab:
        data_tab, collections_tab = st.tabs(["Data", "Analysis Sets"])
    with overview_tab:
        summary_columns = st.columns(4)
        summary_columns[0].metric("Engine runs", len(selected_jobs))
        summary_columns[1].metric(
            "Engines", selected_jobs["engine"].nunique()
        )
        summary_columns[2].metric(
            (
                "Prepared targets"
                if target_ligand_comparison
                else "Compounds"
            ),
            (
                selected_jobs["target_run_id"].nunique()
                if target_ligand_comparison
                else selected_metrics["candidate_id"].nunique()
            ),
        )
        summary_columns[3].metric(
            "Metric rows", len(selected_metrics)
        )
        coverage_index = "target" if target_ligand_comparison else "candidate_id"
        coverage_source = selected_jobs if target_ligand_comparison else selected_metrics
        coverage = pd.crosstab(
            coverage_source[coverage_index],
            coverage_source["engine"],
        ).reset_index()
        st.markdown("#### Result coverage")
        st.dataframe(
            coverage,
            hide_index=True,
            width="stretch",
        )
        if target_ligand_comparison:
            _render_target_engine_coverage(selected_jobs)
        st.markdown("#### Included campaigns")
        st.dataframe(
            selected_jobs[
                [
                    "result",
                    "launch_campaign",
                    "campaign",
                    "target",
                    "target_origin",
                    "target_artifact",
                    "dataset",
                    "compound_count",
                    "created_at",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "result": st.column_config.LinkColumn(
                    "Result", display_text="Open"
                ),
                "created_at": st.column_config.DatetimeColumn(
                    "Created", format="YYYY-MM-DD HH:mm"
                ),
            },
        )

    with native_tab:
        st.markdown("## Native engine metrics")
        selected_engine_names = set(
            selected_metrics["engine"].astype(str).unique()
        )
        preferred_order = (
            *STRUCTURE_ENGINE_ORDER,
            "Nesso-1",
            "GNINA rescoring",
            "Boltzina rescoring",
        )
        engine_options = [
            engine
            for engine in preferred_order
            if engine in selected_engine_names
        ] + sorted(selected_engine_names.difference(preferred_order))
        st.caption(
            "All selected engines are shown below. Each plot defaults to the "
            "engine's primary scientific ranking output; use its selector to "
            "inspect any other emitted score without hiding the other engines."
        )
        compound_names = selected_metrics.get(
            "compound_name",
            pd.Series("", index=selected_metrics.index),
        ).fillna("").astype(str).str.strip()
        named_compound_ids = set(
            selected_metrics.loc[
                ~compound_names.str.casefold().isin(
                    {"", "nan", "none", "<na>"}
                ),
                "candidate_id",
            ].astype(str)
        )
        total_compound_ids = selected_metrics["candidate_id"].astype(
            str
        ).nunique()
        compound_label_mode = st.segmented_control(
            "Compound labels in plots",
            (
                ("Compound IDs", "Compound names", "Names + IDs")
                if named_compound_ids
                else ("Compound IDs",)
            ),
            default="Compound IDs",
            key="campaign_native_compound_label_mode_v2",
            help=(
                "Names come from preserved columns in the imported compound "
                "dataset. Names + IDs writes the ID on a second line in "
                "parentheses. Missing names display the ID alone."
            ),
        ) or "Compound IDs"
        if named_compound_ids:
            source_rows = selected_metrics.loc[
                selected_metrics["compound_name_source_column"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne(""),
                ["dataset", "compound_name_source_column"],
            ].drop_duplicates()
            mappings = "; ".join(
                f"{row.dataset} → {row.compound_name_source_column}"
                for row in source_rows.itertuples(index=False)
            )
            st.caption(
                f"Names available for {len(named_compound_ids)} of "
                f"{total_compound_ids} compounds"
                + (f". Source mapping: {mappings}." if mappings else ".")
            )
        else:
            st.caption(
                "The selected imported dataset has no separate compound-name "
                "column; bar plots therefore use compound IDs."
            )
        compound_label_layout = st.segmented_control(
            "Label visibility and plot layout",
            ("Automatic", "Show all labels", "Compact"),
            default="Automatic",
            key="campaign_native_compound_label_layout",
            help=(
                "Automatic uses a vertical plot for up to 18 compounds and a "
                "taller horizontal plot above that threshold. Show all labels "
                "always uses the horizontal layout. Compact retains a vertical "
                "plot and lets Altair thin overlapping labels."
            ),
        ) or "Automatic"
        repetition_mode = st.segmented_control(
            "Repetitions included",
            (
                "Best repetitions",
                "Representative repetition",
                "All repetitions",
            ),
            default="Best repetitions",
            key="campaign_native_repetition_mode",
            help=(
                "Best repetitions retains the requested number of most "
                "favorable independent attempts for each compound and logical "
                "campaign. Representative repetition retains the attempt "
                "closest to that group's median. All repetitions preserves "
                "the complete replicate distribution."
            ),
        )
        best_repetition_count = 1
        if repetition_mode == "Best repetitions":
            best_repetition_count = int(
                st.number_input(
                    "Best repetitions per compound and campaign",
                    min_value=1,
                    value=1,
                    step=1,
                    key="campaign_native_best_repetitions",
                    help=(
                        "The scientific direction of each selected metric is "
                        "used automatically. The default retains only its best "
                        "attempt."
                    ),
                )
            )
            st.caption(
                "Best-attempt summaries are useful for inspecting achievable "
                "poses but are optimistically selected. Use Representative "
                "repetition or All repetitions to assess typical behavior and "
                "replicate variability."
            )
        native_plot_exports: list[dict[str, object]] = []
        for engine_index, engine in enumerate(engine_options):
            definitions = ENGINE_METRICS.get(engine, ())
            engine_rows = selected_metrics.loc[
                selected_metrics["engine"].eq(engine)
            ]
            available_metrics = [
                definition
                for definition in definitions
                if definition[0] in selected_metrics
                and pd.to_numeric(
                    engine_rows[definition[0]],
                    errors="coerce",
                ).notna().any()
            ]
            if not available_metrics:
                continue
            if engine_index:
                st.divider()
            st.markdown(f"### {engine}")
            metric_groups: list[
                tuple[str, str, list[tuple[str, str, bool]]]
            ]
            if engine == "GNINA":
                metric_groups = [
                    (
                        "CNN-ranked pose",
                        "GNINA · CNN-ranked",
                        [
                            definition
                            for definition in available_metrics
                            if definition[0].startswith("cnn_ranked_")
                        ],
                    ),
                    (
                        "Vina-ranked pose",
                        "GNINA · Vina-ranked",
                        [
                            definition
                            for definition in available_metrics
                            if definition[0].startswith("empirical_ranked_")
                        ],
                    ),
                ]
            else:
                metric_groups = [("", engine, available_metrics)]

            for group_label, plot_engine, group_metrics in metric_groups:
                if not group_metrics:
                    continue
                if group_label:
                    st.markdown(f"#### {group_label}")
                metric_lookup = {
                    metric: (label, direction)
                    for metric, label, direction in group_metrics
                }
                metric_options = list(metric_lookup)
                primary_metric = PRIMARY_METRIC.get(plot_engine)
                default_index = (
                    metric_options.index(primary_metric)
                    if primary_metric in metric_options
                    else 0
                )
                metric = st.selectbox(
                    f"{group_label or engine} score",
                    metric_options,
                    index=default_index,
                    format_func=lambda value, lookup=metric_lookup: lookup[value][0],
                    key=(
                        "campaign_native_metric_"
                        + plot_engine.lower()
                        .replace(" ", "_")
                        .replace("-", "_")
                        .replace("·", "_")
                    ),
                )
                metric_label, higher_is_better = metric_lookup[metric]
                reference_semantics = METRIC_REFERENCE_REGISTRY.get(metric)
                st.caption(
                    f"{metric_label}: "
                    + (
                        "higher values rank better."
                        if higher_is_better
                        else "lower values rank better."
                    )
                    + " Error bars show mean ± sample SD across available "
                    "independent attempts."
                    + (
                        f" Reference: {reference_semantics['reference']}. "
                        f"{reference_semantics['meaning']}"
                        if reference_semantics
                        else ""
                    )
                )
                selected_attempt_rows = select_metric_attempts(
                    engine_rows,
                    metric,
                    mode=str(repetition_mode),
                    best_count=best_repetition_count,
                    higher_is_better=higher_is_better,
                )
                native_summary = summarize_metric(
                    selected_attempt_rows,
                    metric,
                )
                native_plot_exports.append(
                    {
                        "engine": plot_engine,
                        "metric": metric,
                        "metric_label": metric_label,
                        "higher_is_better": higher_is_better,
                        "compare_targets": target_ligand_comparison,
                        "compound_label_mode": compound_label_mode,
                        "compound_label_layout": compound_label_layout,
                        "summary": native_summary,
                    }
                )
                st.altair_chart(
                    _metric_chart(
                        native_summary,
                        metric_label,
                        higher_is_better=higher_is_better,
                        compare_targets=target_ligand_comparison,
                        compound_label_mode=compound_label_mode,
                        compound_label_layout=compound_label_layout,
                    ),
                    width="stretch",
                )
                with st.expander(f"{plot_engine} summarized values"):
                    st.dataframe(
                        native_summary.sort_values(
                            "Mean",
                            ascending=not higher_is_better,
                        ),
                        hide_index=True,
                        width="stretch",
                    )

        if native_plot_exports:
            with st.expander("Download selected graphs"):
                st.caption(
                    "Choose from the engine graphs currently visible above. "
                    "The ZIP contains one 220-DPI Matplotlib PNG per graph and "
                    "a manifest. Numerical data are exported separately in the "
                    "normalized CSV bundle below."
                )
                export_engines = [
                    str(plot["engine"]) for plot in native_plot_exports
                ]
                selected_plot_engines = st.multiselect(
                    "Graphs to include",
                    export_engines,
                    default=export_engines,
                    key="campaign_native_plot_exports",
                )
                selected_plot_exports = [
                    plot
                    for plot in native_plot_exports
                    if str(plot["engine"]) in selected_plot_engines
                ]
                plot_signature = json.dumps(
                    {
                        "repetition_mode": repetition_mode,
                        "best_count": best_repetition_count,
                        "compound_label_mode": compound_label_mode,
                        "compound_label_layout": compound_label_layout,
                        "plots": [
                            {
                                "engine": plot["engine"],
                                "metric": plot["metric"],
                                "summary": plot["summary"].to_json(
                                    orient="split", date_format="iso"
                                ),
                            }
                            for plot in selected_plot_exports
                        ],
                    },
                    sort_keys=True,
                )
                native_zip_key = "campaign_native_plot_zip"
                if st.button(
                    "Prepare PNG ZIP",
                    disabled=not selected_plot_exports,
                    key="prepare_campaign_native_plot_zip",
                ):
                    with st.spinner("Rendering selected graphs…"):
                        st.session_state[native_zip_key] = {
                            "signature": plot_signature,
                            "data": _native_metric_plots_zip(
                                selected_plot_exports,
                                repetition_mode=str(repetition_mode),
                            ),
                        }
                prepared_plot_zip = st.session_state.get(native_zip_key)
                if (
                    isinstance(prepared_plot_zip, dict)
                    and prepared_plot_zip.get("signature") == plot_signature
                    and isinstance(prepared_plot_zip.get("data"), bytes)
                ):
                    st.download_button(
                        "Download selected graphs (.zip)",
                        data=prepared_plot_zip["data"],
                        file_name="native-engine-metric-plots.zip",
                        mime="application/zip",
                        key="download_campaign_native_plot_zip",
                    )

        database_export_signature = _campaign_workbook_signature(
            export_jobs,
            data_metrics,
        )
        metric_csv_state_key = "campaign_comparison_metric_csv_export"
        database_export_state_key = "campaign_comparison_csv_bundle_export"
        with st.expander(
            "Download database-ready data (CSV)", expanded=True
        ):
            st.caption(
                "The ZIP uses a normalized relational layout suitable for "
                "PostgreSQL, SQLite, DuckDB, pandas, R, and similar systems. "
                "Its primary metric_observations.csv file has one numeric "
                "observation per row and the same fixed columns for every "
                "engine. Engine-specific outputs are values of metric_name, "
                "not separate columns. Additional CSV relations preserve "
                "campaign, pose-validation, and interaction evidence. Every "
                "selected repetition is exported, independent of the plot "
                "display mode. A calculation-ready wide view is also offered "
                "with adjacent repeat_1, repeat_2, repeat_3, … columns. "
                "Boltz-2 and AlphaFold 3 use one engine-ranked representative "
                "structure per independent repeat. Geometry diagnostics are "
                "excluded; selection fields are used only for engines such as "
                "GNINA that have multiple ranking tracks."
            )
            if st.button(
                "Prepare metric observations CSV",
                key="prepare_campaign_metric_csv",
            ):
                with st.spinner("Serializing normalized metric observations…"):
                    try:
                        metric_rows = _campaign_database_metric_rows(
                            data_metrics
                        )
                        repeat_matrix = _campaign_database_repeat_matrix(
                            data_metrics
                        )
                        st.session_state[metric_csv_state_key] = {
                            "signature": database_export_signature,
                            "data": _csv_bytes(metric_rows),
                            "repeat_data": _csv_bytes(repeat_matrix),
                        }
                    except Exception as exc:
                        st.session_state.pop(metric_csv_state_key, None)
                        st.error(f"Metric CSV export failed: {exc}")
            prepared_metric_csv = st.session_state.get(metric_csv_state_key)
            if (
                isinstance(prepared_metric_csv, dict)
                and prepared_metric_csv.get("signature")
                == database_export_signature
                and isinstance(prepared_metric_csv.get("data"), bytes)
            ):
                st.download_button(
                    "Download serial metric observations (.csv)",
                    data=prepared_metric_csv["data"],
                    file_name="campaign-metric-observations.csv",
                    mime="text/csv",
                    key="download_campaign_metric_csv",
                )
                if isinstance(
                    prepared_metric_csv.get("repeat_data"), bytes
                ):
                    st.download_button(
                        "Download repeat columns for calculations (.csv)",
                        data=prepared_metric_csv["repeat_data"],
                        file_name="campaign-metric-repeats-wide.csv",
                        mime="text/csv",
                        key="download_campaign_metric_repeats_csv",
                    )
            st.caption(
                "For linked pose-validation and interaction tables, prepare "
                "the complete multi-table archive below."
            )
            if st.button(
                "Prepare complete CSV bundle with linked evidence",
                key="prepare_campaign_comparison_csv_bundle",
            ):
                with st.spinner(
                    "Normalizing all repetitions and collecting linked pose "
                    "validity and interaction relations…"
                ):
                    try:
                        csv_bundle = _collect_campaign_results_csv_bundle(
                            run_root,
                            export_jobs,
                            data_metrics,
                        )
                        st.session_state[database_export_state_key] = {
                            "signature": database_export_signature,
                            "data": csv_bundle,
                        }
                    except Exception as exc:
                        st.session_state.pop(database_export_state_key, None)
                        st.error(f"CSV export failed: {exc}")
            prepared_database_export = st.session_state.get(
                database_export_state_key
            )
            if (
                isinstance(prepared_database_export, dict)
                and prepared_database_export.get("signature")
                == database_export_signature
                and isinstance(prepared_database_export.get("data"), bytes)
            ):
                st.download_button(
                    "Download normalized campaign data (.zip)",
                    data=prepared_database_export["data"],
                    file_name="campaign-database-csv-export.zip",
                    mime="application/zip",
                    key="download_campaign_comparison_csv",
                )

    with correlation_tab:
        st.markdown("## Cross-engine correlations")
        if target_ligand_comparison:
            _render_target_correlations(selected_metrics)
        else:
            st.caption(
                "Correlations compare compounds, not raw replicate rows. Use "
                "them to identify agreement or disagreement between scoring "
                "systems; they do not establish experimental validity."
            )
            _render_correlations(selected_metrics)

    with rescoring_tab:
        st.markdown("## Rescoring")
        _render_rescoring_comparison(linked_rescoring_metrics)

    with consensus_tab:
        st.markdown("## Consensus ranking")
        st.caption(
            (
                "Prepared targets are ranked independently within each selected "
                "native engine metric, then direction-aware percentiles are "
                "combined. Independent attempts contribute mean and sample SD, "
                "not additional target observations."
            )
            if target_ligand_comparison
            else (
                "Raw docking scores, Rosetta energy units, AF3 confidence and "
                "learned affinity values are not pooled. Each campaign is first "
                "converted to a within-campaign percentile (1 = best), then "
                "percentiles are summarized across selected engines and "
                "campaigns. To avoid inflating a compound through a tiny "
                "selected subset, consensus includes only campaigns covering at "
                "least 80% of the best-covered selected campaign and at least "
                "two compounds. Subset rescoring remains available in Native "
                "metrics and Data. This is a prioritization aid, not a "
                "calibrated binding score."
            )
        )
        if not target_ligand_comparison:
            st.caption(
                f"Compound labels: {compound_label_mode}. Layout: "
                f"{compound_label_layout}. Change these controls in Native "
                "metrics; the same settings apply to consensus plots."
            )
        if target_ligand_comparison:
            _render_target_consensus(selected_metrics)
        consensus = (
            pd.DataFrame()
            if target_ligand_comparison
            else consensus_percentiles(selected_metrics)
        )
        if consensus.empty:
            if not target_ligand_comparison:
                st.info(
                    "No primary metrics are available for consensus ranking."
                )
        else:
            combined = (
                consensus.groupby("candidate_id")[
                    "Within-campaign percentile"
                ]
                .agg(["mean", "std", "count"])
                .reset_index()
                .rename(
                    columns={
                        "mean": "Mean percentile",
                        "std": "Sample SD",
                        "count": "Contributing campaigns",
                    }
                )
                .sort_values("Mean percentile", ascending=False)
            )
            combined["Sample SD"] = combined["Sample SD"].fillna(0.0)
            compound_name_lookup = (
                consensus[["candidate_id", "compound_name"]]
                .drop_duplicates("candidate_id")
                .set_index("candidate_id")["compound_name"]
                if "compound_name" in consensus
                else pd.Series(dtype=str)
            )
            combined["compound_name"] = combined["candidate_id"].map(
                compound_name_lookup
            ).fillna("")
            combined_plot = _with_compound_plot_labels(
                combined,
                label_mode=compound_label_mode,
            )
            compound_order = combined_plot["_compound_plot_label"].tolist()
            consensus_horizontal = (
                compound_label_layout == "Show all labels"
                or (
                    compound_label_layout == "Automatic"
                    and len(combined_plot) > 18
                )
            )
            consensus_axis_options: dict[str, object] = {
                "labelLimit": 420 if consensus_horizontal else 220,
                "labelOverlap": (
                    "greedy"
                    if compound_label_layout == "Compact"
                    else False
                ),
            }
            consensus_label_expression = _compound_axis_label_expression(
                compound_label_mode
            )
            if consensus_label_expression:
                consensus_axis_options["labelExpr"] = (
                    consensus_label_expression
                )
            consensus_tooltips = [
                alt.Tooltip("candidate_id:N", title="Compound ID"),
                alt.Tooltip("compound_name:N", title="Compound name"),
                alt.Tooltip("Mean percentile:Q", format=".3f"),
                alt.Tooltip("Sample SD:Q", format=".3f"),
                "Contributing campaigns:Q",
            ]
            if consensus_horizontal:
                chart = (
                    alt.Chart(combined_plot)
                    .mark_bar(color="#7c3aed")
                    .encode(
                        y=alt.Y(
                            "_compound_plot_label:N",
                            sort=compound_order,
                            title="Compound",
                            axis=alt.Axis(**consensus_axis_options),
                        ),
                        x=alt.X(
                            "Mean percentile:Q",
                            scale=alt.Scale(domain=[0, 1]),
                            title="Mean within-campaign percentile",
                        ),
                        tooltip=consensus_tooltips,
                    )
                    .properties(height=max(390, 30 * len(combined_plot)))
                )
            else:
                consensus_axis_options["labelAngle"] = -45
                chart = (
                    alt.Chart(combined_plot)
                    .mark_bar(color="#7c3aed")
                    .encode(
                        x=alt.X(
                            "_compound_plot_label:N",
                            sort=compound_order,
                            title="Compound",
                            axis=alt.Axis(**consensus_axis_options),
                        ),
                        y=alt.Y(
                            "Mean percentile:Q",
                            scale=alt.Scale(domain=[0, 1]),
                            title="Mean within-campaign percentile",
                        ),
                        tooltip=consensus_tooltips,
                    )
                    .properties(height=390)
                )
            st.altair_chart(chart, width="stretch")
            by_target = (
                consensus.groupby(["candidate_id", "target"])[
                    "Within-campaign percentile"
                ]
                .agg(["mean", "count"])
                .reset_index()
                .rename(
                    columns={
                        "mean": "Mean percentile",
                        "count": "Contributing campaigns",
                    }
                )
            )
            by_target["compound_name"] = by_target["candidate_id"].map(
                compound_name_lookup
            ).fillna("")
            by_target = _with_compound_plot_labels(
                by_target,
                label_mode=compound_label_mode,
            )
            if by_target["target"].nunique() > 1:
                st.markdown("#### Target-by-compound profile")
                st.caption(
                    "This matrix keeps targets separate. A compound that ranks "
                    "high for one target and low for another may be a useful "
                    "selectivity signal, subject to experimental validation."
                )
                target_heatmap = (
                    alt.Chart(by_target)
                    .mark_rect()
                    .encode(
                        x=alt.X(
                            "target:N",
                            title="Target",
                            axis=alt.Axis(labelAngle=-25, labelLimit=240),
                        ),
                        y=alt.Y(
                            "_compound_plot_label:N",
                            title="Compound",
                            sort=compound_order,
                            axis=alt.Axis(
                                labelLimit=420,
                                labelOverlap=False,
                                **(
                                    {
                                        "labelExpr": consensus_label_expression
                                    }
                                    if consensus_label_expression
                                    else {}
                                ),
                            ),
                        ),
                        color=alt.Color(
                            "Mean percentile:Q",
                            title="Mean percentile",
                            scale=alt.Scale(
                                domain=[0, 1],
                                scheme="viridis",
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "candidate_id:N", title="Compound ID"
                            ),
                            alt.Tooltip(
                                "compound_name:N", title="Compound name"
                            ),
                            alt.Tooltip("target:N", title="Target"),
                            alt.Tooltip(
                                "Mean percentile:Q", format=".3f"
                            ),
                            "Contributing campaigns:Q",
                        ],
                    )
                    .properties(height=max(260, 28 * len(combined)))
                )
                st.altair_chart(target_heatmap, width="stretch")
                st.dataframe(
                    by_target.pivot(
                        index=["candidate_id", "compound_name"],
                        columns="target",
                        values="Mean percentile",
                    ).reset_index(),
                    hide_index=True,
                    width="stretch",
                )
            engine_matrix = (
                consensus.groupby(["candidate_id", "engine"])[
                    "Within-campaign percentile"
                ]
                .mean()
                .unstack("engine")
                .reset_index()
            )
            by_engine = (
                consensus.groupby(["candidate_id", "engine"])[
                    "Within-campaign percentile"
                ]
                .agg(["mean", "count"])
                .reset_index()
                .rename(
                    columns={
                        "mean": "Mean percentile",
                        "count": "Contributing campaigns",
                    }
                )
            )
            by_engine["compound_name"] = by_engine["candidate_id"].map(
                compound_name_lookup
            ).fillna("")
            by_engine = _with_compound_plot_labels(
                by_engine,
                label_mode=compound_label_mode,
            )
            if by_engine["engine"].nunique() > 1:
                st.markdown("#### Engine-by-compound profile")
                st.caption(
                    "Agreement across engines strengthens prioritization; large "
                    "differences identify model-dependent predictions worth "
                    "reviewing rather than averaging away."
                )
                engine_heatmap = (
                    alt.Chart(by_engine)
                    .mark_rect()
                    .encode(
                        x=alt.X("engine:N", title="Engine"),
                        y=alt.Y(
                            "_compound_plot_label:N",
                            title="Compound",
                            sort=compound_order,
                            axis=alt.Axis(
                                labelLimit=420,
                                labelOverlap=False,
                                **(
                                    {
                                        "labelExpr": consensus_label_expression
                                    }
                                    if consensus_label_expression
                                    else {}
                                ),
                            ),
                        ),
                        color=alt.Color(
                            "Mean percentile:Q",
                            title="Mean percentile",
                            scale=alt.Scale(
                                domain=[0, 1],
                                scheme="viridis",
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "candidate_id:N", title="Compound ID"
                            ),
                            alt.Tooltip(
                                "compound_name:N", title="Compound name"
                            ),
                            alt.Tooltip("engine:N", title="Engine"),
                            alt.Tooltip(
                                "Mean percentile:Q", format=".3f"
                            ),
                            "Contributing campaigns:Q",
                        ],
                    )
                    .properties(height=max(260, 28 * len(combined)))
                )
                st.altair_chart(engine_heatmap, width="stretch")
            st.dataframe(
                combined.merge(engine_matrix, on="candidate_id"),
                hide_index=True,
                width="stretch",
            )

    with pose_validation_tab:
        st.markdown("## Pose validity")
        if target_ligand_comparison:
            _render_target_pose_validation(run_root, selected_jobs)
        else:
            _render_pose_validation_summary(
                run_root,
                selected_jobs,
                selected_metrics,
            )

    with interaction_analysis_tab:
        st.markdown("## Protein–ligand interactions")
        _render_interaction_analysis_summary(
            run_root,
            selected_jobs,
            selected_metrics,
        )

    with viewer_tab:
        if target_ligand_comparison:
            _render_target_structural_explorer(selected_metrics)
        else:
            explorer_view = st.segmented_control(
                "Explore selected results",
                explorer_options,
                default="Target × compound matrix",
                key="campaign_explorer_view",
            )
            if explorer_view == "Target × compound matrix":
                _render_target_compound_explorer(selected_metrics)
            elif explorer_view == "RMSD & pose agreement":
                st.markdown("## RMSD and pose agreement")
                _render_rmsd_analysis(selected_metrics)
            elif explorer_view == "3D structures":
                st.markdown("## 3D structures")
                _render_structure_comparison(selected_metrics)
            else:
                _render_md_candidate_selection(
                    run_root,
                    selected_jobs,
                    selected_metrics,
                )

    with data_tab:
        st.markdown("## Result data")
        st.caption(
            "The tables below preview the same repetition-level values included "
            "in the normalized CSV export under Native metrics."
        )
        public_metrics = _campaign_database_metric_rows(data_metrics)
        _, metric_wide = _campaign_metric_export_tables(data_metrics)
        replicate_columns = [
            column
            for column in metric_wide
            if str(column).startswith("Replicate ")
        ]
        compact_columns = [
            column
            for column in (
                "Target",
                "Compound ID",
                "Compound name",
                "Engine",
                "Parameter label",
            )
            if column in metric_wide
        ] + replicate_columns
        if metric_wide.empty:
            st.info("No numeric scientific result parameters are available.")
        else:
            st.dataframe(
                metric_wide[compact_columns],
                hide_index=True,
                width="stretch",
            )
        with st.expander("Serial metric observations and provenance"):
            st.caption(
                "Every displayed value has a unique observation ID and an "
                "explicit repeat number. Boltz-2 and AlphaFold 3 contribute "
                "one engine-ranked representative structure per repeat."
            )
            st.dataframe(public_metrics, hide_index=True, width="stretch")
        st.info(
            "Use Download database-ready data (CSV) in Native metrics for the "
            "complete relational export. Plot PNG files are downloaded "
            "separately."
        )

    with collections_tab:
        st.markdown("## Analysis Sets")
        current_rescoring_runs = st.session_state.get(
            "campaign_compare_rescoring_runs",
            linked_rescoring_jobs["campaign_id"].tolist(),
        )
        _render_saved_collections(
            runs_root(),
            selection={
                "schema_version": ANALYSIS_SET_SCHEMA_VERSION,
                "selection_type": "campaign_analysis_set",
                "dataset_run_id": selected_dataset,
                "dataset": dataset_labels.get(
                    selected_dataset, selected_dataset
                ),
                "target_run_ids": list(selected_targets),
                "launch_campaign_ids": list(selected_launches),
                "target_launch_pairs": (
                    list(selected_target_launch_pairs)
                    if target_ligand_comparison
                    else []
                ),
                "engines": list(selected_engines),
                "engine_run_ids": list(selected_campaigns),
                "rescoring_run_ids": [
                    str(value) for value in current_rescoring_runs
                ],
            },
        )


if os.environ.get("MN_LIGAND_POSE_SIMILARITY_WORKER") != "1":
    render()
