from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


HYPOTHESIS_COLUMNS = (
    "Enabled",
    "Required",
    "Interaction",
    "Protein residue",
    "Protein region",
    "Requirement group",
    "Requirement logic",
    "Importance",
    "Reference support",
    "Reference evidence",
)

POSE_IDENTITY_COLUMNS = (
    "source_run_id",
    "source_engine",
    "target_run_id",
    "pose_id",
    "compound_id",
    "replicate",
    "prediction",
    "selection_criterion",
)


def ligand_bend_index(coordinates: object) -> float | None:
    """Return a size-independent 3D bend/spread index for one ligand pose.

    The value is derived from the heavy-atom coordinate principal axes.  A
    perfectly line-like point cloud approaches 0, while poses distributed
    across the two transverse axes receive progressively larger values.  The
    descriptor is invariant to translation, rotation and uniform scaling.
    """
    try:
        points = np.asarray(coordinates, dtype=float)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3:
        return None
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) < 3:
        return None
    centered = finite - finite.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= np.finfo(float).eps:
        return None
    transverse = float(eigenvalues[0] + eigenvalues[1])
    return float(np.sqrt(transverse / total))


def apply_reference_bend_penalty(
    scored: pd.DataFrame,
    *,
    reference_bend_index: float | None,
    candidate_bend_indices: dict[tuple[str, str], float | None],
    tolerance: float = 0.1,
    penalty_points_per_0_1: float = 10.0,
    hard_maximum_excess: float | None = None,
) -> pd.DataFrame:
    """Add reference-shape bend diagnostics and an adjusted selection score."""
    output = scored.copy()
    output["Reference ligand bend index"] = reference_bend_index
    candidate_values: list[float | None] = []
    excess_values: list[float | None] = []
    penalties: list[float] = []
    accepted: list[bool] = []
    adjusted: list[float] = []
    allowed = max(float(tolerance), 0.0)
    strength = max(float(penalty_points_per_0_1), 0.0)
    for row in output.to_dict("records"):
        key = (
            str(row.get("source_run_id") or ""),
            str(row.get("pose_id") or ""),
        )
        candidate = candidate_bend_indices.get(key)
        candidate_values.append(candidate)
        if reference_bend_index is None or candidate is None:
            excess = None
            penalty = 0.0
            is_accepted = hard_maximum_excess is None
        else:
            excess = max(
                0.0,
                float(candidate) - float(reference_bend_index) - allowed,
            )
            penalty = strength * excess / 0.1
            is_accepted = (
                hard_maximum_excess is None
                or excess <= max(float(hard_maximum_excess), 0.0)
            )
        raw_score = float(row.get("Reference similarity (%)") or 0.0)
        excess_values.append(excess)
        penalties.append(round(penalty, 2))
        accepted.append(is_accepted)
        adjusted.append(round(max(0.0, raw_score - penalty), 2))
    output["Candidate pose bend index"] = candidate_values
    output["Excess bend index"] = excess_values
    output["Bend penalty (percentage points)"] = penalties
    output["Bend criterion met"] = accepted
    output["Selection score (%)"] = adjusted
    return output


def normalize_interaction_type(value: object) -> str:
    text = re.sub(r"[_-]+", " ", str(value or "").strip().lower())
    if "water" in text and "bridge" in text:
        return "water bridge"
    if "hydrogen" in text or "hbond" in text:
        return "hydrogen bond"
    if (
        "hydrophob" in text
        or "alkyl pi" in text
        or "carbon pi" in text
        or "pi alkyl" in text
    ):
        return "hydrophobic contact"
    if "salt" in text:
        return "salt bridge"
    if (
        ("pi" in text and "stack" in text)
        or "aromatic face" in text
        or "aromatic edge" in text
    ):
        return "pi stacking"
    if "cation" in text and "pi" in text:
        return "pi cation"
    if "halogen" in text:
        return "halogen bond"
    return text


def interaction_residue(row: pd.Series | dict[str, Any]) -> str:
    chain = str(row.get("protein_chain") or "").strip()
    name = str(row.get("protein_residue_name") or "").strip().upper()
    number = str(row.get("protein_residue_number") or "").strip()
    if re.fullmatch(r"-?\d+\.0", number):
        number = number[:-2]
    insertion = str(row.get("protein_insertion_code") or "").strip()
    if name and number:
        return f"{chain + ':' if chain else ''}{name}{number}{insertion}"
    value = str(row.get("residue") or row.get("Protein residue") or "").strip()
    match = re.match(
        r"(?P<name>[A-Za-z]{3})(?P<number>-?\d+[A-Za-z]?)"
        r"(?:\s*·\s*chain\s+(?P<chain>\S+))?$",
        value,
        re.IGNORECASE,
    )
    if match:
        chain = str(match.group("chain") or "")
        return (
            f"{chain + ':' if chain else ''}"
            f"{match.group('name').upper()}{match.group('number')}"
        )
    return value


def _scope(value: object) -> str:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in {"BB", "SC"} else "BB+SC"


def infer_static_hypothesis(
    interactions: pd.DataFrame,
    *,
    hydrogen_bond_region: str = "BB+SC",
) -> pd.DataFrame:
    """Infer an editable hypothesis from direct reference-complex contacts."""
    if interactions.empty:
        return pd.DataFrame(columns=HYPOTHESIS_COLUMNS)
    table = interactions.copy().fillna("")
    table["_interaction"] = table.get(
        "interaction_type", pd.Series("", index=table.index)
    ).map(normalize_interaction_type)
    table = table.loc[table["_interaction"].ne("water bridge")].copy()
    if table.empty:
        return pd.DataFrame(columns=HYPOTHESIS_COLUMNS)
    table["_residue"] = table.apply(interaction_residue, axis=1)
    table["_region"] = table.get(
        "protein_atom_scope", pd.Series("", index=table.index)
    ).map(_scope)
    support_column = (
        "analysis_engine" if "analysis_engine" in table else "source_engine"
    )
    grouped_rows: list[dict[str, Any]] = []
    for (interaction, residue), group in table.groupby(
        ["_interaction", "_residue"], dropna=False, sort=False
    ):
        regions = set(group["_region"].astype(str))
        observed_region = (
            "BB+SC"
            if "BB+SC" in regions or {"BB", "SC"}.issubset(regions)
            else "BB"
            if regions == {"BB"}
            else "SC"
            if regions == {"SC"}
            else "BB+SC"
        )
        grouped_rows.append(
            {
                "_interaction": interaction,
                "_residue": residue,
                "_region": observed_region,
                "Requirement group": "",
                "Requirement logic": "ALL",
                "Reference support": int(
                    group[support_column].astype(str).nunique()
                ),
            }
        )
    grouped = pd.DataFrame(grouped_rows)
    maximum = max(int(grouped["Reference support"].max()), 1)
    grouped["Importance"] = (
        grouped["Reference support"].astype(float) / maximum
    ).round(3)
    grouped["Enabled"] = True
    grouped["Required"] = False
    grouped["Reference evidence"] = "Prepared target interaction analysis"
    return grouped.rename(
        columns={
            "_interaction": "Interaction",
            "_residue": "Protein residue",
            "_region": "Protein region",
        }
    )[list(HYPOTHESIS_COLUMNS)]


def infer_md_hypothesis(
    contact_consensus: list[dict[str, Any]],
    *,
    minimum_occupancy: float = 0.1,
    hydrogen_bond_region: str = "BB+SC",
) -> pd.DataFrame:
    """Infer an editable hypothesis from direct MD contacts.

    Water-bridge fields are deliberately ignored: only direct hydrogen bonds,
    hydrophobic contacts and salt bridges can become selection requirements.
    """
    rows: list[dict[str, Any]] = []
    region = _scope(hydrogen_bond_region)
    definitions = (
        ("hydrogen bond", "mean_hydrogen_bond"),
        ("hydrophobic contact", "mean_hydrophobic"),
        ("salt bridge", "mean_salt_bridge"),
    )
    for contact in contact_consensus:
        residue = interaction_residue(contact)
        for interaction, prefix in definitions:
            total = float(contact.get(f"{prefix}_occupancy") or 0.0)
            backbone = float(
                contact.get(f"{prefix}_backbone_occupancy") or 0.0
            )
            sidechain = float(
                contact.get(f"{prefix}_sidechain_occupancy") or 0.0
            )
            if interaction == "hydrogen bond" and region == "BB":
                total = backbone
            elif interaction == "hydrogen bond" and region == "SC":
                total = sidechain
            if total < minimum_occupancy:
                continue
            observed_region = (
                region
                if interaction == "hydrogen bond"
                and region in {"BB", "SC"}
                else (
                    "BB+SC"
                    if backbone > 0 and sidechain > 0
                    else "BB"
                    if backbone > 0
                    else "SC"
                    if sidechain > 0
                    else "BB+SC"
                )
            )
            rows.append(
                {
                    "Enabled": True,
                    "Required": False,
                    "Interaction": interaction,
                    "Protein residue": residue,
                    "Protein region": observed_region,
                    "Requirement group": "",
                    "Requirement logic": "ALL",
                    "Importance": round(total, 4),
                    "Reference support": total,
                    "Reference evidence": "MD direct-contact occupancy",
                }
            )
    return pd.DataFrame(rows, columns=HYPOTHESIS_COLUMNS)


def score_candidate_poses(
    interactions: pd.DataFrame,
    hypothesis: pd.DataFrame,
    *,
    minimum_detector_support: int = 1,
) -> pd.DataFrame:
    """Score every pose against a reviewed, reference-derived hypothesis."""
    enabled = hypothesis.loc[
        hypothesis.get("Enabled", pd.Series(False, index=hypothesis.index))
        .astype(bool)
    ].copy()
    if interactions.empty or enabled.empty:
        return pd.DataFrame()
    table = interactions.copy().fillna("")
    table["_interaction"] = table.get(
        "interaction_type", pd.Series("", index=table.index)
    ).map(normalize_interaction_type)
    table = table.loc[table["_interaction"].ne("water bridge")].copy()
    table["_residue"] = table.apply(interaction_residue, axis=1)
    table["_region"] = table.get(
        "protein_atom_scope", pd.Series("", index=table.index)
    ).map(_scope)
    identity = [
        column for column in POSE_IDENTITY_COLUMNS if column in table.columns
    ]
    if "pose_id" not in identity or "compound_id" not in identity:
        raise ValueError("Interaction rows do not retain pose identity")
    output: list[dict[str, Any]] = []
    for keys, pose in table.groupby(identity, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(identity, keys, strict=True))
        matched: list[str] = []
        required_groups: dict[str, list[dict[str, Any]]] = {}
        weighted_match = 0.0
        total_weight = 0.0
        supporting_tools: set[str] = set()
        for criterion_index, criterion in enumerate(
            enabled.to_dict("records")
        ):
            interaction = normalize_interaction_type(
                criterion.get("Interaction")
            )
            residue = str(criterion.get("Protein residue") or "").strip()
            region = _scope(criterion.get("Protein region"))
            weight = max(float(criterion.get("Importance") or 0.0), 0.0)
            total_weight += weight
            candidates = pose.loc[
                pose["_interaction"].eq(interaction)
                & pose["_residue"].eq(residue)
            ]
            if region in {"BB", "SC"}:
                candidates = candidates.loc[candidates["_region"].eq(region)]
            tool_column = (
                "analysis_engine"
                if "analysis_engine" in candidates
                else "source_engine"
            )
            tools = {
                str(value)
                for value in candidates.get(
                    tool_column, pd.Series(dtype=str)
                ).tolist()
                if str(value).strip()
            }
            detected = len(tools) >= max(int(minimum_detector_support), 1)
            label = f"{interaction}: {residue} ({region})"
            if detected:
                weighted_match += weight
                matched.append(label)
                supporting_tools.update(tools)
            if bool(criterion.get("Required")):
                group_name = str(
                    criterion.get("Requirement group") or ""
                ).strip()
                if not group_name:
                    group_name = f"independent:{criterion_index + 1}"
                logic = str(
                    criterion.get("Requirement logic") or "ALL"
                ).strip().upper()
                required_groups.setdefault(group_name, []).append(
                    {
                        "detected": detected,
                        "label": label,
                        "logic": "ANY" if logic == "ANY" else "ALL",
                    }
                )
        missing_required: list[str] = []
        matched_required_groups: list[str] = []
        for group_name, members in required_groups.items():
            group_logic = (
                "ANY"
                if any(member["logic"] == "ANY" for member in members)
                else "ALL"
            )
            detections = [bool(member["detected"]) for member in members]
            group_met = (
                any(detections) if group_logic == "ANY" else all(detections)
            )
            public_group = (
                group_name
                if not group_name.startswith("independent:")
                else members[0]["label"]
            )
            if group_met:
                matched_required_groups.append(
                    f"{public_group} ({group_logic})"
                )
            else:
                joiner = " OR " if group_logic == "ANY" else " AND "
                missing_required.append(
                    f"{public_group} ({group_logic}: "
                    + joiner.join(str(member["label"]) for member in members)
                    + ")"
                )
        row.update(
            {
                "Reference similarity (%)": round(
                    100.0 * weighted_match / total_weight, 2
                )
                if total_weight
                else 0.0,
                "Required interactions met": not missing_required,
                "Required interaction groups": "; ".join(
                    f"{name} ({'ANY' if any(item['logic'] == 'ANY' for item in items) else 'ALL'})"
                    for name, items in required_groups.items()
                ),
                "Matched required groups": "; ".join(
                    matched_required_groups
                ),
                "Matched reference interactions": "; ".join(matched),
                "Missing required interactions": "; ".join(missing_required),
                "Supporting interaction tools": "; ".join(
                    sorted(supporting_tools)
                ),
                "Interaction tool count": len(supporting_tools),
            }
        )
        output.append(row)
    return pd.DataFrame(output)


def select_best_candidates(
    scored: pd.DataFrame,
    *,
    mode: str = "One per compound",
    minimum_similarity: float = 0.0,
    require_required_interactions: bool = True,
    score_column: str = "Reference similarity (%)",
    eligibility_column: str | None = None,
) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    if score_column not in scored:
        raise ValueError(f"Selection score column is missing: {score_column}")
    eligible = scored.loc[
        pd.to_numeric(scored[score_column], errors="coerce")
        .fillna(0.0)
        .ge(float(minimum_similarity))
    ].copy()
    if require_required_interactions:
        eligible = eligible.loc[
            eligible["Required interactions met"].astype(bool)
        ]
    if eligibility_column is not None:
        if eligibility_column not in eligible:
            raise ValueError(
                f"Selection eligibility column is missing: "
                f"{eligibility_column}"
            )
        eligible = eligible.loc[eligible[eligibility_column].astype(bool)]
    if eligible.empty:
        return eligible
    grouping = ["compound_id"]
    if mode == "One per compound and engine":
        grouping.append("source_engine")
    eligible = eligible.sort_values(
        [
            score_column,
            "Interaction tool count",
            "pose_id",
            "source_engine",
            "source_run_id",
        ],
        ascending=[False, False, True, True, True],
        kind="stable",
    )
    selected = eligible.drop_duplicates(grouping, keep="first").copy()
    selected["Selected rank"] = (
        selected.sort_values(
            score_column, ascending=False, kind="stable"
        )
        .groupby("compound_id")
        .cumcount()
        + 1
    )
    selected["Selection status"] = "Selected by reference-derived hypothesis"
    return selected.sort_values(
        ["compound_id", "Selected rank"], kind="stable"
    ).reset_index(drop=True)
