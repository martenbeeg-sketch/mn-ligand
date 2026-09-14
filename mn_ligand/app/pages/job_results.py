from __future__ import annotations

from io import BytesIO
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyArrowPatch
from PIL import Image

from mn_ligand.app.pages.compound_results import render_compound_dataset_report
from mn_ligand.app.preferences import (
    load_network_appearance,
    save_network_appearance,
)
from mn_ligand.app.viewers import (
    aligned_structure_data,
    closest_residue_ligand_atom_pair,
    dashed_line_segments,
    distribute_2d_labels,
    horizontalize_2d_coordinates,
    is_mmcif_text,
    pdb_interaction_atom_coordinates,
    pdb_ligand_atom_aliases,
    render_persistent_3dmol,
    transform_2d_coordinates,
)
from mn_ligand.core.artifacts import ArtifactRef
from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records
from mn_ligand.core.job_control import (
    cancellation_eligibility,
    create_job_retry,
    request_job_cancellation,
    retry_eligibility,
)
from mn_ligand.core.pockets import PocketSet
from mn_ligand.core.residue_mapping import sequence_author_residue_mapping
from mn_ligand.runtime import resolve_run_dir, runs_root


VIEWABLE_SUFFIXES = {".pdb", ".ent", ".cif", ".mmcif", ".sdf", ".mol2"}


def _display_tool_name(value: object) -> str:
    text = str(value or "").strip()
    return "RosettaLigand" if text.lower() == "openvs" else (text or "-")


def _resolve_run_dir(task_group: str, run_id: str) -> Path | None:
    return resolve_run_dir(task_group, run_id)


def _latest_superseding_job(job: JobRecord) -> JobRecord:
    """Follow immutable revision links to the newest available result."""
    current = job
    visited = {job.run_id}
    while True:
        next_run_id = str(
            current.metadata.get("superseded_by_run_id") or ""
        ).strip()
        if not next_run_id or next_run_id in visited:
            return current
        next_dir = _resolve_run_dir(current.task_group, next_run_id)
        if next_dir is None:
            return current
        visited.add(next_run_id)
        current = JobRecord.load(next_dir, task_group=current.task_group)


def _requested_compound_index(compound_ids: list[str]) -> int:
    requested = str(st.query_params.get("compound_id", "") or "").strip()
    try:
        return compound_ids.index(requested)
    except ValueError:
        return 0


def _file_inventory(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                "modified": path.stat().st_mtime,
            }
        )
    return rows


def _detailed_result_url(job: JobRecord) -> str:
    routes: dict[str, tuple[str, dict[str, str]]] = {
        "structure-jobs": ("structure-results", {}),
        "bound-ligand-md": ("md-results", {}),
        "md-system-prep": ("md-results", {"run_type": "md-system-prep"}),
        "abfe": ("openfe-results", {"run_type": "abfe"}),
        "rbfe": ("openfe-results", {"run_type": "rbfe"}),
        "openfe": ("openfe-results", {"run_type": "openfe"}),
        "admet": ("admet-results", {}),
        "qc": ("qc-results", {}),
    }
    route = routes.get(job.task_group)
    if route is None:
        return ""
    slug, extra = route
    query = urlencode({"run_id": job.run_id, **extra})
    return f"./{slug}?{query}"


def _flatten_scalars(value: Any, *, prefix: str = "", depth: int = 0) -> list[dict[str, Any]]:
    if depth > 5:
        return []
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_scalars(child, prefix=child_prefix, depth=depth + 1))
        return rows
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and (len(value) > 300 or "\n" in value):
            return []
        return [{"metric": prefix, "value": "-" if value is None else str(value)}]
    return []


def _render_viewer(path: Path) -> None:
    if path.stat().st_size > 10 * 1024 * 1024:
        st.warning("This structure is larger than the 10 MB preview limit. Open its specialized result viewer instead.")
        return
    try:
        import py3Dmol

        suffix = path.suffix.lower()
        file_format = "cif" if suffix in {".cif", ".mmcif"} else suffix.lstrip(".")
        viewer = py3Dmol.view(width=1100, height=600)
        viewer.addModel(path.read_text(errors="replace"), file_format)
        if suffix in {".pdb", ".ent", ".cif", ".mmcif"}:
            viewer.setStyle({"hetflag": False}, {"cartoon": {"color": "spectrum"}})
            viewer.setStyle({"hetflag": True}, {"stick": {"colorscheme": "greenCarbon"}})
        else:
            viewer.setStyle({}, {"stick": {"colorscheme": "cyanCarbon"}})
        viewer.zoomTo()
        render_persistent_3dmol(
            viewer,
            key=f"artifact-preview:{path.resolve()}",
            height=620,
        )
    except Exception as exc:
        st.error(f"Structure preview failed: {exc}")


def _input_artifact_path(job: JobRecord, key: str) -> Path | None:
    try:
        payload = json.loads((job.run_dir / "input.json").read_text())
        artifact_payload = payload.get(key)
        if not isinstance(artifact_payload, dict):
            return None
        artifact = ArtifactRef.from_dict(artifact_payload)
    except (OSError, TypeError, ValueError):
        return None
    source_job = next(
        (
            candidate
            for candidate in iter_job_records(runs_root())
            if candidate.run_id == artifact.run_id
        ),
        None,
    )
    return (
        artifact.resolve(source_job.run_dir, must_exist=True)
        if source_job is not None
        else None
    )


def _input_artifact_paths(job: JobRecord, key: str) -> list[Path]:
    try:
        payload = json.loads((job.run_dir / "input.json").read_text())
        artifact_payloads = payload.get(key)
        if not isinstance(artifact_payloads, list):
            return []
    except (OSError, TypeError, ValueError):
        return []
    jobs_by_id = {
        candidate.run_id: candidate for candidate in iter_job_records(runs_root())
    }
    paths: list[Path] = []
    for artifact_payload in artifact_payloads:
        if not isinstance(artifact_payload, dict):
            continue
        try:
            artifact = ArtifactRef.from_dict(artifact_payload)
        except (TypeError, ValueError):
            continue
        source_job = jobs_by_id.get(artifact.run_id)
        path = (
            artifact.resolve(source_job.run_dir, must_exist=True)
            if source_job is not None
            else None
        )
        if path is not None:
            paths.append(path)
    return paths


def _sibling_input_artifact_path(
    job: JobRecord, key: str, artifact_types: set[str]
) -> Path | None:
    try:
        payload = json.loads((job.run_dir / "input.json").read_text())
        source_payload = payload.get(key)
        if not isinstance(source_payload, dict):
            return None
        source_run_id = str(source_payload.get("run_id") or "")
    except (OSError, TypeError, ValueError):
        return None
    source_job = next(
        (
            candidate
            for candidate in iter_job_records(runs_root())
            if candidate.run_id == source_run_id
        ),
        None,
    )
    if source_job is None or source_job.artifact_manifest is None:
        return None
    for artifact in source_job.artifact_manifest.artifacts:
        if artifact.artifact_type not in artifact_types:
            continue
        path = artifact.resolve(source_job.run_dir, must_exist=True)
        if path is not None:
            return path
    return None


def _aligned_structure_data(
    reference_path_text: str,
    reference_modified_ns: int,
    mobile_path_text: str,
    mobile_modified_ns: int,
) -> tuple[str, float, int]:
    return aligned_structure_data(
        reference_path_text,
        reference_modified_ns,
        mobile_path_text,
        mobile_modified_ns,
    )


def _sdf_record(path: Path, pose_index: int = 1) -> str:
    records = _sdf_records(path)
    selected = max(0, int(pose_index or 1) - 1)
    record = (
        records[selected]
        if selected < len(records)
        else records[0]
        if records
        else path.read_text(errors="replace").rstrip()
    )
    return record + "\n$$$$\n"


def _sdf_records(path: Path) -> list[str]:
    return [
        record.strip("\r\n")
        for record in path.read_text(errors="replace").split("$$$$")
        if record.strip()
    ]


def _sdf_properties(record: str) -> dict[str, str]:
    """Return scalar SD properties without requiring the molecule to sanitize."""
    properties: dict[str, str] = {}
    lines = record.splitlines()
    index = 0
    while index < len(lines):
        match = re.match(r"^>\s*<([^>]+)>", lines[index].strip())
        if match is None:
            index += 1
            continue
        key = match.group(1).strip()
        index += 1
        values: list[str] = []
        while index < len(lines) and lines[index].strip():
            values.append(lines[index].strip())
            index += 1
        properties[key] = "\n".join(values)
    return properties


def _first_sdf_record(path: Path) -> str:
    # A blank title is legal in an MDL molfile and is used by some prepared
    # reference ligands.  `_sdf_records` intentionally trims delimiter newlines
    # for record navigation, but doing that to a blank-title molfile shifts its
    # three-line header and makes py3Dmol silently reject the model.
    record, _, _ = path.read_text(errors="replace").partition("$$$$")
    return record.rstrip("\r\n") + "\n$$$$\n"


def _has_bound_input_ligand(path: Path) -> bool:
    try:
        import gemmi

        structure = gemmi.read_structure(str(path))
        for chain in structure[0]:
            for residue in chain:
                name = residue.name.strip().upper()
                if name in {"HOH", "WAT", "DOD"}:
                    continue
                if not gemmi.find_tabulated_residue(name).is_amino_acid():
                    return any(atom.element.name != "H" for atom in residue)
    except (IndexError, OSError, RuntimeError, ValueError):
        return False
    return False


def _representative_prediction_artifacts(
    job: JobRecord, predictions: list[ArtifactRef]
) -> list[ArtifactRef]:
    labels = [
        str(artifact.role or artifact.label or "") for artifact in predictions
    ]
    if job.workflow == "boltz2_refolding":
        has_models = any(re.search(r"_model_\d+$", label) for label in labels)
        if has_models:
            return [
                artifact
                for artifact, label in zip(predictions, labels)
                if re.search(r"_model_0$", label)
            ]
    if job.workflow == "alphafold3_refolding":
        has_samples = any(re.search(r"(?:_|-)sample(?:_|-)\d+$", label) for label in labels)
        if has_samples:
            return [
                artifact
                for artifact, label in zip(predictions, labels)
                if re.search(r"(?:_|-)sample(?:_|-)0$", label)
            ]
    return predictions


def _representative_prediction_metrics(
    job: JobRecord, table: pd.DataFrame
) -> pd.DataFrame:
    if job.workflow == "boltz2_refolding" and "model_id" in table.columns:
        model_ids = table["model_id"].astype(str)
        if model_ids.str.contains(r"_model_\d+$", regex=True).any():
            return table[model_ids.str.endswith("_model_0")].copy()
    if job.workflow == "alphafold3_refolding":
        if "sample" in table.columns:
            samples = pd.to_numeric(table["sample"], errors="coerce")
            if samples.notna().any():
                return table[samples.eq(0)].copy()
        if "prediction_id" in table.columns:
            prediction_ids = table["prediction_id"].astype(str)
            if prediction_ids.str.contains(
                r"(?:_|-)sample(?:_|-)\d+$", regex=True
            ).any():
                return table[
                    prediction_ids.str.contains(
                        r"(?:_|-)sample(?:_|-)0$", regex=True
                    )
                ].copy()
    return table


def _render_docking_complex_viewer(job: JobRecord) -> bool:
    if job.workflow != "docking_campaign":
        return False
    scores_path = job.run_dir / "scores.csv"
    receptor_path = _input_artifact_path(job, "target_artifact")
    if not scores_path.is_file() or receptor_path is None:
        return False
    try:
        scores = pd.read_csv(scores_path).fillna("")
    except (OSError, ValueError):
        return False
    if not {"compound_id", "replicate", "pose_file"}.issubset(scores.columns):
        return False
    is_gnina = str(
        job.metadata.get("engine") or job.metadata.get("tool") or ""
    ).strip().lower() == "gnina"
    gnina_criterion = "cnn_score"
    if is_gnina:
        criterion_label = st.segmented_control(
            "GNINA pose-selection criterion",
            ("CNN pose score", "Empirical / Vina score"),
            default="CNN pose score",
            key=f"gnina_viewer_pose_criterion_{job.run_id}",
            help=(
                "CNN pose score selects the model GNINA ranks first. Empirical / "
                "Vina score searches every emitted model and selects the most "
                "negative minimizedAffinity score."
            ),
        )
        gnina_criterion = (
            "empirical_score"
            if criterion_label == "Empirical / Vina score"
            else "cnn_score"
        )
        try:
            from mn_ligand.workflows.docking import select_gnina_pose

            updated_rows: list[dict[str, Any]] = []
            for row in scores.to_dict("records"):
                pose_file = str(row.get("pose_file") or "")
                selection = select_gnina_pose(
                    job.run_dir / pose_file,
                    gnina_criterion,
                )
                if selection:
                    row.update(
                        {
                            "pose_index": selection.get("pose_index", 1),
                            "best_score_kcal_mol": selection.get(
                                "empirical_score_kcal_mol"
                            ),
                            "cnn_score": selection.get("cnn_score"),
                            "cnn_affinity": selection.get("cnn_affinity"),
                        }
                    )
                updated_rows.append(row)
            scores = pd.DataFrame(updated_rows).fillna("")
        except (OSError, ValueError):
            pass
    try:
        from mn_ligand.workflows.docking import docking_pose_diagnostics

        center_payload = dict(job.metadata.get("center") or {})
        size_payload = dict(job.metadata.get("size") or {})
        diagnostic_rows, diagnostic_summaries = docking_pose_diagnostics(
            job.run_dir,
            scores.to_dict("records"),
            center=tuple(float(center_payload[axis]) for axis in "xyz"),
            size=tuple(float(size_payload[axis]) for axis in "xyz"),
        )
        scores = pd.DataFrame(diagnostic_rows).fillna("")
    except (KeyError, TypeError, ValueError):
        diagnostic_summaries = {}
    rows: list[dict[str, Any]] = []
    for record in scores.to_dict("records"):
        relative = str(record.get("pose_file") or "")
        pose_path = job.run_dir / str(Path(relative).with_suffix(".sdf"))
        if relative and pose_path.is_file():
            rows.append({**record, "_pose_path": pose_path})
    if not rows:
        return False

    compound_ids = sorted(
        {str(row["compound_id"]) for row in rows},
        key=lambda compound_id: min(
            float(row["best_score_kcal_mol"])
            for row in rows
            if str(row["compound_id"]) == compound_id
            and row.get("best_score_kcal_mol") not in ("", None)
        ),
    )
    compound_id = st.selectbox(
        "Docked compound",
        compound_ids,
        index=_requested_compound_index(compound_ids),
        key=f"docking_viewer_compound_{job.run_id}",
        help="Compounds are ordered by their best docking score; lower is more favorable.",
    )
    compound_rows = [
        row for row in rows if str(row["compound_id"]) == str(compound_id)
    ]
    reference_path = (
        _input_artifact_path(job, "reference_ligand_artifact")
        or _sibling_input_artifact_path(
            job,
            "target_artifact",
            {"prepared_ligand_set", "reference_ligand", "bound_ligand"},
        )
    )
    viewer_controls = st.columns(2)
    show_all_replicates = viewer_controls[0].checkbox(
        "Show all replicates together",
        value=False,
        key=f"docking_viewer_all_replicates_{job.run_id}_{compound_id}",
        help="Overlay every independent pose for this compound in the same receptor frame.",
    )
    show_reference = viewer_controls[1].checkbox(
        "Show T3 reference",
        value=reference_path is not None,
        disabled=reference_path is None,
        key=f"docking_viewer_reference_{job.run_id}_{compound_id}",
        help="Toggle the coordinate-bearing T3 template used to define or guide the docking site.",
    )
    if show_all_replicates:
        displayed_rows = sorted(
            compound_rows, key=lambda row: int(row["replicate"])
        )
        selected = displayed_rows[0]
    else:
        selected_index = st.selectbox(
            "Replicate and pose",
            list(range(len(compound_rows))),
            format_func=lambda index: (
                f"Replicate {int(compound_rows[index]['replicate'])} · "
                f"seed {int(compound_rows[index]['seed'])} · "
                f"{float(compound_rows[index]['best_score_kcal_mol']):.3f} kcal/mol"
            ),
            key=f"docking_viewer_pose_{job.run_id}_{compound_id}",
        )
        selected = compound_rows[int(selected_index)]
        displayed_rows = [selected]

    score_columns = st.columns(3)
    if show_all_replicates:
        scores = pd.Series(
            [float(row["best_score_kcal_mol"]) for row in displayed_rows]
        )
        pose_summary = diagnostic_summaries.get(str(compound_id), {})
        score_columns = st.columns(5)
        score_columns[0].metric(
            "Mean docking score", f"{scores.mean():.3f} kcal/mol"
        )
        score_columns[1].metric(
            "Sample SD",
            f"{scores.std(ddof=1):.3f} kcal/mol"
            if len(scores) > 1
            else "—",
        )
        score_columns[2].metric("Replicates", len(scores))
        mean_rmsd = pose_summary.get("mean_pairwise_pose_rmsd_angstrom")
        max_rmsd = pose_summary.get("max_pairwise_pose_rmsd_angstrom")
        score_columns[3].metric(
            "Mean pose RMSD",
            f"{float(mean_rmsd):.2f} Å" if mean_rmsd is not None else "—",
            help=(
                "Symmetry-aware heavy-atom RMSD in the fixed receptor frame; "
                "the poses are not superposed before measurement."
            ),
        )
        score_columns[4].metric(
            "Maximum pose RMSD",
            f"{float(max_rmsd):.2f} Å" if max_rmsd is not None else "—",
        )
        if max_rmsd is not None and float(max_rmsd) > 2.0:
            st.warning(
                f"Replicate placement is inconsistent: maximum fixed-frame ligand "
                f"RMSD is {float(max_rmsd):.2f} Å. Inspect individual replicates "
                "before using the score summary."
            )
    else:
        score_columns[0].metric(
            "Docking score",
            f"{float(selected['best_score_kcal_mol']):.3f} kcal/mol",
        )
        score_columns[1].metric("Replicate", int(selected["replicate"]))
        score_columns[2].metric("Seed", int(selected["seed"]))
        if is_gnina:
            score_columns = st.columns(3)
            score_columns[0].metric(
                "CNN pose score",
                (
                    f"{float(selected['cnn_score']):.3f}"
                    if selected.get("cnn_score") not in ("", None)
                    else "—"
                ),
            )
            score_columns[1].metric(
                "CNN affinity estimate",
                (
                    f"{float(selected['cnn_affinity']):.3f}"
                    if selected.get("cnn_affinity") not in ("", None)
                    else "—"
                ),
            )
            score_columns[2].metric(
                "Selected GNINA model",
                int(selected.get("pose_index") or 1),
            )
        center_distance = selected.get("box_center_distance_angstrom")
        if center_distance not in (None, ""):
            st.caption(
                f"Pose heavy-atom centroid is {float(center_distance):.2f} Å from "
                f"the configured docking-box center; "
                f"{int(selected.get('outside_box_atom_count') or 0)} heavy atoms "
                "are outside the box."
            )
    pose_colors = ("cyanCarbon", "magentaCarbon", "orangeCarbon", "purpleCarbon")
    pose_legend = ", ".join(
        f"replicate {int(row['replicate'])} "
        f"{('cyan', 'magenta', 'orange', 'purple')[index % 4]}"
        for index, row in enumerate(displayed_rows)
    )
    reference_legend = (
        ", and the T3 template/reference is green" if show_reference else ""
    )
    st.caption(
        f"Complex overlay: receptor cartoon is grey; {pose_legend}"
        f"{reference_legend}. Docking scores are ranking estimates, not experimentally "
        "measured binding free energies."
    )
    try:
        import py3Dmol

        viewer = py3Dmol.view(width=1100, height=650)
        receptor_format = (
            "cif"
            if receptor_path.suffix.lower() in {".cif", ".mmcif"}
            else "pdb"
        )
        viewer.addModel(receptor_path.read_text(errors="replace"), receptor_format)
        viewer.setStyle(
            {"model": 0, "hetflag": False},
            {"cartoon": {"color": "#cbd5e1", "opacity": 0.9}},
        )
        next_model = 1
        if show_reference and reference_path is not None:
            reference_format = (
                "sdf"
                if reference_path.suffix.lower() == ".sdf"
                else reference_path.suffix.lower().lstrip(".")
            )
            reference_data = (
                _first_sdf_record(reference_path)
                if reference_format == "sdf"
                else reference_path.read_text(errors="replace")
            )
            viewer.addModel(reference_data, reference_format)
            viewer.setStyle(
                {"model": next_model},
                {"stick": {"colorscheme": "greenCarbon", "radius": 0.18}},
            )
            next_model += 1
        first_pose_model = next_model
        for index, row in enumerate(displayed_rows):
            pose_path = Path(row["_pose_path"])
            viewer.addModel(
                _sdf_record(pose_path, int(row.get("pose_index") or 1)),
                "sdf",
            )
            viewer.setStyle(
                {"model": next_model},
                {
                    "stick": {
                        "colorscheme": pose_colors[index % len(pose_colors)],
                        "radius": 0.22,
                    }
                },
            )
            next_model += 1
        viewer.zoomTo({"model": first_pose_model})
        viewer.zoom(0.72)
        render_persistent_3dmol(
            viewer,
            key=f"docking-complex:{job.run_id}",
            height=670,
        )
    except Exception as exc:
        st.error(f"Docked-complex preview failed: {exc}")
    st.caption(
        "This is a coordinate-preserving visual complex assembled from the stored "
        "receptor and pose. The original receptor, pose, scores, and provenance remain "
        "separate typed artifacts."
    )
    return True


def _render_refolding_complex_viewer(job: JobRecord) -> bool:
    if job.workflow not in {"boltz2_refolding", "alphafold3_refolding"}:
        return False
    if job.artifact_manifest is None:
        return False
    target_path = _input_artifact_path(job, "target")
    predictions = [
        artifact
        for artifact in job.artifact_manifest.artifacts
        if artifact.artifact_type == "predicted_complex"
        and artifact.resolve(job.run_dir, must_exist=True) is not None
    ]
    predictions = _representative_prediction_artifacts(job, predictions)
    if target_path is None or not predictions:
        return False

    prediction_rows: list[dict[str, Any]] = []
    for index, artifact in enumerate(predictions, start=1):
        parts = str(artifact.role or artifact.label or f"prediction-{index}").split(":")
        replicate_match = next(
            (
                re.fullmatch(r"replicate_(\d+)", part)
                for part in parts
                if part.startswith("replicate_")
            ),
            None,
        )
        prediction_rows.append(
            {
                "artifact": artifact,
                "path": artifact.resolve(job.run_dir, must_exist=True),
                "compound_id": parts[0],
                "replicate": int(replicate_match.group(1)) if replicate_match else index,
                "label": str(artifact.role or artifact.label or f"Prediction {index}"),
            }
        )
    compounds = sorted({str(row["compound_id"]) for row in prediction_rows})
    compound = st.selectbox(
        "Predicted compound",
        compounds,
        index=_requested_compound_index(compounds),
        key=f"refolding_viewer_compound_{job.run_id}",
    )
    compound_rows = sorted(
        (
            row
            for row in prediction_rows
            if str(row["compound_id"]) == str(compound)
        ),
        key=lambda row: (int(row["replicate"]), str(row["label"])),
    )
    coordinate_inputs = [
        path
        for path in _input_artifact_paths(job, "compound_sets")
        if path.suffix.lower() in {".sdf", ".mol", ".mol2", ".pdb"}
    ]
    explicit_reference = _input_artifact_path(
        job, "reference_ligand_artifact"
    )
    target_has_ligand = _has_bound_input_ligand(target_path)
    sibling_ligand = _sibling_input_artifact_path(
        job,
        "target",
        {"prepared_ligand_set", "reference_ligand", "bound_ligand"},
    )
    reference_available = (
        target_has_ligand
        or explicit_reference is not None
        or sibling_ligand is not None
        or bool(coordinate_inputs)
    )
    controls = st.columns(3)
    show_all = controls[0].checkbox(
        "Show all predictions together",
        value=False,
        disabled=len(compound_rows) < 2,
        key=f"refolding_viewer_all_{job.run_id}_{compound}",
        help="Overlay all available seeded predictions after protein-structure alignment.",
    )
    show_input = controls[1].checkbox(
        "Show prepared input structure",
        value=True,
        key=f"refolding_viewer_input_{job.run_id}_{compound}",
    )
    show_reference = controls[2].checkbox(
        "Show input/reference ligand",
        value=reference_available,
        disabled=not reference_available,
        key=f"refolding_viewer_reference_{job.run_id}_{compound}",
        help=(
            "Show the coordinate-bearing ligand from the prepared input target. "
            "A separate coordinate compound file is used only when the target has no ligand."
        ),
    )
    if show_all:
        displayed = compound_rows
    else:
        selected_index = st.selectbox(
            "Prediction",
            list(range(len(compound_rows))),
            format_func=lambda index: compound_rows[index]["label"],
            key=f"refolding_viewer_prediction_{job.run_id}_{compound}",
        )
        displayed = [compound_rows[int(selected_index)]]

    palette = (
        ("cyanCarbon", "cyan"),
        ("magentaCarbon", "magenta"),
        ("orangeCarbon", "orange"),
        ("purpleCarbon", "purple"),
    )
    aligned: list[dict[str, Any]] = []
    for row in displayed:
        path = Path(row["path"])
        try:
            structure_data, rmsd, matched_atoms = _aligned_structure_data(
                str(target_path),
                target_path.stat().st_mtime_ns,
                str(path),
                path.stat().st_mtime_ns,
            )
        except Exception as exc:
            st.error(f"Could not align {row['label']}: {exc}")
            return True
        aligned.append(
            {
                **row,
                "structure_data": structure_data,
                "alignment_rmsd": rmsd,
                "matched_atoms": matched_atoms,
            }
        )
    alignment_rmsds = pd.Series(
        [float(row["alignment_rmsd"]) for row in aligned]
    )
    metrics = st.columns(3)
    metrics[0].metric("Displayed predictions", len(aligned))
    metrics[1].metric(
        "Mean protein alignment RMSD",
        f"{alignment_rmsds.mean():.2f} Å",
    )
    metrics[2].metric(
        "Matched Cα atoms",
        min(int(row["matched_atoms"]) for row in aligned),
    )
    prediction_legend = ", ".join(
        f"prediction {int(row['replicate'])} {palette[index % len(palette)][1]}"
        for index, row in enumerate(aligned)
    )
    optional_legend = []
    if show_input:
        optional_legend.append("prepared input protein grey")
    if show_reference and reference_available:
        optional_legend.append("input/reference compound green")
    st.caption(
        "Aligned complex overlay: "
        + ", ".join([*optional_legend, prediction_legend])
        + ". Every folded protein and its ligand were transformed together using "
        "a rigid least-squares fit of matched protein Cα atoms."
    )
    try:
        import py3Dmol

        viewer = py3Dmol.view(width=1100, height=650)
        next_model = 0
        target_model: int | None = None
        if show_input or (show_reference and target_has_ligand):
            target_format = (
                "cif"
                if target_path.suffix.lower() in {".cif", ".mmcif"}
                else "pdb"
            )
            viewer.addModel(
                target_path.read_text(errors="replace"), target_format
            )
            target_model = next_model
            if show_input:
                viewer.setStyle(
                    {"model": target_model, "hetflag": False},
                    {"cartoon": {"color": "#cbd5e1", "opacity": 0.65}},
                )
            if show_reference and target_has_ligand:
                viewer.setStyle(
                    {"model": target_model, "hetflag": True},
                    {"stick": {"colorscheme": "greenCarbon", "radius": 0.2}},
                )
            next_model += 1
        if show_reference and not target_has_ligand and (
            explicit_reference is not None
            or sibling_ligand is not None
            or coordinate_inputs
        ):
            reference_path = (
                explicit_reference or sibling_ligand or coordinate_inputs[0]
            )
            reference_format = reference_path.suffix.lower().lstrip(".")
            reference_data = (
                _first_sdf_record(reference_path)
                if reference_format == "sdf"
                else reference_path.read_text(errors="replace")
            )
            viewer.addModel(reference_data, reference_format)
            viewer.setStyle(
                {"model": next_model},
                {"stick": {"colorscheme": "greenCarbon", "radius": 0.18}},
            )
            next_model += 1
        first_prediction_model = next_model
        for index, row in enumerate(aligned):
            color_scheme = palette[index % len(palette)][0]
            viewer.addModel(row["structure_data"], "cif")
            viewer.setStyle(
                {"model": next_model, "hetflag": False},
                {
                    "cartoon": {
                        "color": palette[index % len(palette)][1],
                        "opacity": 0.28 if len(aligned) > 1 else 0.55,
                    }
                },
            )
            viewer.setStyle(
                {"model": next_model, "hetflag": True},
                {"stick": {"colorscheme": color_scheme, "radius": 0.22}},
            )
            next_model += 1
        viewer.zoomTo({"model": first_prediction_model, "hetflag": True})
        viewer.zoom(0.72)
        render_persistent_3dmol(
            viewer,
            key=f"refolding-complex:{job.run_id}",
            height=670,
        )
    except Exception as exc:
        st.error(f"Aligned cofolding preview failed: {exc}")
    return True


def _pocket_source_target(job: JobRecord) -> Path | None:
    try:
        payload = json.loads((job.run_dir / "input.json").read_text())
        artifact_payload = payload.get("input_artifact")
        if not isinstance(artifact_payload, dict):
            artifact_payload = (payload.get("input_artifacts") or {}).get("prepared_target")
        if not isinstance(artifact_payload, dict):
            return None
        artifact = ArtifactRef.from_dict(artifact_payload)
    except (OSError, TypeError, ValueError):
        return None
    source_task_group = str(payload.get("source_task_group") or "")
    source_dir = (
        resolve_run_dir(source_task_group, artifact.run_id)
        if source_task_group
        else None
    )
    if source_dir is None:
        source_job = next(
            (candidate for candidate in iter_job_records(runs_root()) if candidate.run_id == artifact.run_id),
            None,
        )
        source_dir = source_job.run_dir if source_job is not None else None
    return artifact.resolve(source_dir, must_exist=True) if source_dir is not None else None


def _draw_box(viewer: Any, center: tuple[float, float, float], size: tuple[float, float, float]) -> None:
    cx, cy, cz = center
    hx, hy, hz = (value / 2.0 for value in size)
    corners = (
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    )
    for start_index, end_index in (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ):
        start = corners[start_index]
        end = corners[end_index]
        viewer.addCylinder(
            {
                "start": {"x": start[0], "y": start[1], "z": start[2]},
                "end": {"x": end[0], "y": end[1], "z": end[2]},
                "radius": 0.12,
                "color": "#f97316",
                "fromCap": 1,
                "toCap": 1,
            }
        )


def _render_pocket_context(job: JobRecord) -> bool:
    if job.workflow != "pocket_detection":
        return False
    target_path = _pocket_source_target(job)
    pocket_set_value = str(job.result.get("pocket_set") or "")
    if target_path is None or not pocket_set_value:
        return False
    pocket_set_path = (job.run_dir / pocket_set_value).resolve()
    try:
        pocket_set_path.relative_to(job.run_dir.resolve())
        pocket_set = PocketSet.read(pocket_set_path)
    except (OSError, TypeError, ValueError):
        return False
    if not pocket_set.pockets:
        return False

    selected_rank = st.selectbox(
        "Pocket",
        [pocket.rank for pocket in pocket_set.pockets],
        format_func=lambda rank: next(
            (
                f"Rank {pocket.rank} · {pocket.method} · "
                f"score {pocket.score:.3f}"
                if pocket.score is not None
                else f"Rank {pocket.rank} · {pocket.method}"
            )
            for pocket in pocket_set.pockets
            if pocket.rank == rank
        ),
        key=f"pocket_context_rank_{job.run_id}",
    )
    pocket = next(item for item in pocket_set.pockets if item.rank == selected_rank)
    try:
        import py3Dmol

        target_format = (
            "cif"
            if target_path.suffix.lower() in {".cif", ".mmcif"}
            else "pdb"
        )
        viewer = py3Dmol.view(width=1100, height=650)
        viewer.addModel(target_path.read_text(errors="replace"), target_format)
        viewer.setStyle(
            {"model": 0, "hetflag": False},
            {"cartoon": {"color": "#cbd5e1", "opacity": 0.88}},
        )
        viewer.setStyle(
            {"model": 0, "hetflag": True},
            {"stick": {"colorscheme": "greenCarbon", "radius": 0.18}},
        )
        for residue in pocket.residues:
            selector: dict[str, Any] = {
                "model": 0,
                "chain": "" if residue.chain_id == "_" else residue.chain_id,
            }
            try:
                selector["resi"] = int(residue.residue_number)
            except ValueError:
                selector["resi"] = residue.residue_number
            viewer.addStyle(
                selector,
                {
                    "stick": {"color": "#d946ef", "radius": 0.24},
                    "sphere": {"color": "#d946ef", "scale": 0.2},
                },
            )
        pocket_structure = (job.run_dir / pocket.structure_path).resolve()
        if (
            pocket.structure_path
            and pocket_structure.is_file()
            and pocket_structure.suffix.lower() in {".pdb", ".ent"}
        ):
            pocket_structure.relative_to(job.run_dir.resolve())
            viewer.addModel(pocket_structure.read_text(errors="replace"), "pdb")
            viewer.setStyle(
                {"model": 1},
                {
                    "stick": {"color": "#d946ef", "radius": 0.24},
                    "sphere": {"color": "#d946ef", "scale": 0.2},
                },
            )
        _draw_box(viewer, pocket.center_angstrom, pocket.size_angstrom)
        viewer.addSphere(
            {
                "center": {
                    "x": pocket.center_angstrom[0],
                    "y": pocket.center_angstrom[1],
                    "z": pocket.center_angstrom[2],
                },
                "radius": 0.5,
                "color": "#f97316",
            }
        )
        viewer.setBackgroundColor("white")
        viewer.zoomTo({"model": 0})
        render_persistent_3dmol(
            viewer,
            key=f"pocket-context:{job.run_id}",
            height=670,
        )
        st.caption(
            "Full prepared target: light grey. Pocket-lining residues: magenta. "
            "Predicted docking box and center: orange."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "rank": pocket.rank,
                        "method": pocket.method,
                        "score": pocket.score,
                        "residues": len(pocket.residues),
                        "center (A)": ", ".join(f"{value:.2f}" for value in pocket.center_angstrom),
                        "box size (A)": ", ".join(f"{value:.2f}" for value in pocket.size_angstrom),
                    }
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        return True
    except Exception as exc:
        st.error(f"Pocket context preview failed: {exc}")
        return True


def _render_redocking_overlay(path: Path) -> None:
    try:
        import py3Dmol

        records = [record.strip() for record in path.read_text(errors="replace").split("$$$$") if record.strip()]
        if len(records) < 2:
            _render_viewer(path)
            return
        viewer = py3Dmol.view(width=1100, height=600)
        viewer.addModel(records[0] + "\n$$$$\n", "sdf")
        viewer.addModel(records[1] + "\n$$$$\n", "sdf")
        viewer.setStyle({"model": 0}, {"stick": {"colorscheme": "cyanCarbon"}})
        viewer.setStyle({"model": 1}, {"stick": {"colorscheme": "magentaCarbon"}})
        viewer.zoomTo()
        render_persistent_3dmol(
            viewer,
            key=f"redocking-overlay:{path.resolve()}",
            height=620,
        )
        st.caption("Crystal reference: cyan. Top-ranked docked pose: magenta.")
    except Exception as exc:
        st.error(f"Redocking overlay preview failed: {exc}")


def _render_redocking_metrics(job: JobRecord) -> bool:
    if job.workflow != "redocking_benchmark" or job.artifact_manifest is None:
        return False
    rendered = False
    labels = {
        "summary": "Engine summary",
        "replicates": "Replicate measurements",
        "per_pose": "Per-pose measurements",
    }
    for role in ("summary", "replicates", "per_pose"):
        artifact = next(
            (
                item
                for item in job.artifact_manifest.artifacts
                if item.role == role
                and item.artifact_type in {"redocking_summary", "redocking_metrics"}
            ),
            None,
        )
        path = artifact.resolve(job.run_dir, must_exist=True) if artifact else None
        if path is None:
            continue
        try:
            table = pd.read_csv(path)
        except (OSError, ValueError) as exc:
            st.warning(f"Could not read {labels[role].lower()}: {exc}")
            continue
        st.markdown(f"#### {labels[role]}")
        st.dataframe(table, hide_index=True, width="stretch")
        rendered = True
    if rendered:
        st.caption(
            "RMSD is symmetry-aware heavy-atom direct RMSD in the shared receptor frame. "
            "SD values are sample standard deviations (n−1)."
        )
    return rendered


def _render_openvs_scores(job: JobRecord) -> bool:
    if job.workflow != "openvs_docking" or job.artifact_manifest is None:
        return False
    artifact = next(
        (
            item
            for item in job.artifact_manifest.artifacts
            if item.artifact_type == "docking_scores" and item.role == "ranked_scores"
        ),
        None,
    )
    path = artifact.resolve(job.run_dir, must_exist=True) if artifact else None
    if path is None:
        return False
    try:
        table = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        st.warning(f"Could not read RosettaLigand scores: {exc}")
        return False
    st.markdown("#### RosettaLigand ranked scores")
    if {"compound_id", "estimated_dg_reu"}.issubset(table.columns):
        try:
            import altair as alt

            scores = table.copy()
            scores["estimated_dg_reu"] = pd.to_numeric(
                scores["estimated_dg_reu"], errors="coerce"
            )
            scores = scores.dropna(subset=["estimated_dg_reu"])
            tooltip = [
                alt.Tooltip("compound_id:N", title="Compound"),
                alt.Tooltip(
                    "estimated_dg_reu:Q",
                    title="Estimated ΔG (REU)",
                    format=".3f",
                ),
            ]
            encoding: dict[str, Any] = {
                "x": alt.X(
                    "compound_id:N",
                    title="Compound",
                    axis=alt.Axis(labelAngle=-45, labelLimit=180),
                ),
                "y": alt.Y(
                    "estimated_dg_reu:Q",
                    title="Estimated ΔG (REU)",
                    scale=alt.Scale(zero=False),
                ),
                "tooltip": tooltip,
            }
            if "replicate" in scores.columns:
                encoding["xOffset"] = alt.XOffset("replicate:N")
                encoding["color"] = alt.Color("replicate:N", title="Attempt")
                tooltip.insert(1, alt.Tooltip("replicate:O", title="Attempt"))
            st.markdown("##### RosettaLigand score distribution")
            st.altair_chart(
                alt.Chart(scores)
                .mark_point(filled=True, size=105, opacity=0.84)
                .encode(**encoding)
                .properties(height=360),
                width="stretch",
            )
        except (ImportError, TypeError, ValueError) as exc:
            st.warning(f"Could not render RosettaLigand score plot: {exc}")
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Estimated dG, dH, −TdS, and component scores are Rosetta energy units "
        "(REU) for relative ranking; they are not kcal/mol binding affinities."
    )
    convergence_artifacts = {
        item.role: item
        for item in job.artifact_manifest.artifacts
        if item.artifact_type == "convergence_result"
    }
    labels = {
        "convergence_summary": "RosettaLigand convergence summary",
        "replicate_summary": "RosettaLigand replicate summary",
        "running_statistics": "Running replicate statistics",
    }
    for role in ("convergence_summary", "replicate_summary", "running_statistics"):
        item = convergence_artifacts.get(role)
        table_path = item.resolve(job.run_dir, must_exist=True) if item else None
        if table_path is None:
            continue
        try:
            convergence_table = pd.read_csv(table_path)
        except (OSError, ValueError) as exc:
            st.warning(f"Could not read {labels[role].lower()}: {exc}")
            continue
        st.markdown(f"#### {labels[role]}")
        if {
            "compound_id",
            "mean_dg_reu",
            "sample_sd_dg_reu",
        }.issubset(convergence_table.columns):
            try:
                import altair as alt

                summary = convergence_table.copy()
                summary["mean_dg_reu"] = pd.to_numeric(
                    summary["mean_dg_reu"], errors="coerce"
                )
                summary["sample_sd_dg_reu"] = pd.to_numeric(
                    summary["sample_sd_dg_reu"], errors="coerce"
                ).fillna(0.0)
                summary = summary.dropna(subset=["mean_dg_reu"])
                summary["lower"] = (
                    summary["mean_dg_reu"] - summary["sample_sd_dg_reu"]
                )
                summary["upper"] = (
                    summary["mean_dg_reu"] + summary["sample_sd_dg_reu"]
                )
                base = alt.Chart(summary).encode(
                    x=alt.X(
                        "compound_id:N",
                        title="Compound",
                        axis=alt.Axis(labelAngle=-45, labelLimit=180),
                    )
                )
                bars = base.mark_rule(strokeWidth=3).encode(
                    y=alt.Y(
                        "lower:Q",
                        title="Mean estimated ΔG ± sample SD (REU)",
                        scale=alt.Scale(zero=False),
                    ),
                    y2="upper:Q",
                )
                means = base.mark_point(
                    filled=True, size=110, color="#7c3aed"
                ).encode(
                    y="mean_dg_reu:Q",
                    tooltip=[
                        alt.Tooltip("compound_id:N", title="Compound"),
                        alt.Tooltip("mean_dg_reu:Q", title="Mean", format=".3f"),
                        alt.Tooltip(
                            "sample_sd_dg_reu:Q",
                            title="Sample SD",
                            format=".3f",
                        ),
                    ],
                )
                st.altair_chart(
                    (bars + means).properties(height=350), width="stretch"
                )
            except (ImportError, TypeError, ValueError):
                pass
        st.dataframe(convergence_table, hide_index=True, width="stretch")
    if convergence_artifacts:
        st.caption(
            "Pose clusters use symmetry-aware ligand heavy-atom RMSD in the fixed receptor frame. "
            "The convergence flag requires ≥5 seeds, SD ≤2 REU, running-mean shift ≤1 REU, "
            "and ≥80% occupancy of the dominant pose cluster."
        )
    return True


def _render_docking_scores(job: JobRecord) -> bool:
    if job.workflow != "docking_campaign" or job.artifact_manifest is None:
        return False
    rendered = False
    is_gnina = str(
        job.metadata.get("engine") or job.metadata.get("tool") or ""
    ).strip().lower() == "gnina"
    if is_gnina:
        try:
            import altair as alt
            from mn_ligand.workflows.docking import gnina_pose_records

            pose_rows: list[dict[str, Any]] = []
            for path in sorted((job.run_dir / "results").glob("**/*_out.pdbqt")):
                replicate_match = re.search(r"replicate_(\d+)", path.as_posix())
                for pose in gnina_pose_records(path):
                    pose_rows.append(
                        {
                            "compound_id": path.name.removesuffix("_out.pdbqt"),
                            "replicate": (
                                int(replicate_match.group(1))
                                if replicate_match
                                else 1
                            ),
                            **pose,
                        }
                    )
            pose_table = pd.DataFrame(pose_rows)
            if not pose_table.empty:
                st.markdown("#### GNINA emitted-pose score comparison")
                st.altair_chart(
                    alt.Chart(pose_table)
                    .mark_circle(size=72, opacity=0.68)
                    .encode(
                        x=alt.X(
                            "empirical_score_kcal_mol:Q",
                            title="Empirical / Vina score (kcal/mol)",
                            scale=alt.Scale(zero=False),
                        ),
                        y=alt.Y(
                            "cnn_score:Q",
                            title="CNN pose score",
                            scale=alt.Scale(zero=False),
                        ),
                        color=alt.Color("replicate:N", title="Replicate"),
                        tooltip=[
                            alt.Tooltip("compound_id:N", title="Compound"),
                            alt.Tooltip("replicate:O", title="Replicate"),
                            alt.Tooltip("pose_index:O", title="GNINA model"),
                            alt.Tooltip(
                                "empirical_score_kcal_mol:Q",
                                title="Empirical score",
                                format=".3f",
                            ),
                            alt.Tooltip(
                                "cnn_score:Q",
                                title="CNN pose score",
                                format=".3f",
                            ),
                            alt.Tooltip(
                                "cnn_affinity:Q",
                                title="CNN affinity",
                                format=".3f",
                            ),
                        ],
                    )
                    .properties(height=420),
                    width="stretch",
                )
                st.caption(
                    "Every point is one pose emitted by GNINA. The CNN-ranked pose "
                    "maximizes CNN pose score; the empirical-ranked pose minimizes "
                    "the GNINA minimizedAffinity/Vina-like score. CNN affinity is a "
                    "property of a pose, not the criterion GNINA uses to order poses."
                )
        except (ImportError, OSError, ValueError) as exc:
            st.warning(f"Could not render GNINA emitted-pose scores: {exc}")
    labels = {
        "ranked_scores": "Per-run docking scores",
        "replicate_summary": "Docking replicate summary",
    }
    pose_diagnostics: dict[str, dict[str, Any]] = {}
    for role in ("ranked_scores", "replicate_summary"):
        artifact = next(
            (
                item
                for item in job.artifact_manifest.artifacts
                if item.artifact_type == "docking_scores" and item.role == role
            ),
            None,
        )
        path = artifact.resolve(job.run_dir, must_exist=True) if artifact else None
        if path is None:
            continue
        try:
            table = pd.read_csv(path)
        except (OSError, ValueError) as exc:
            st.warning(f"Could not read {labels[role].lower()}: {exc}")
            continue
        if role == "ranked_scores":
            try:
                from mn_ligand.workflows.docking import docking_pose_diagnostics

                center_payload = dict(job.metadata.get("center") or {})
                size_payload = dict(job.metadata.get("size") or {})
                diagnostic_rows, pose_diagnostics = docking_pose_diagnostics(
                    job.run_dir,
                    table.fillna("").to_dict("records"),
                    center=tuple(
                        float(center_payload[axis]) for axis in "xyz"
                    ),
                    size=tuple(float(size_payload[axis]) for axis in "xyz"),
                )
                table = pd.DataFrame(diagnostic_rows)
            except (KeyError, TypeError, ValueError):
                pose_diagnostics = {}
        elif role == "replicate_summary" and pose_diagnostics:
            diagnostics_frame = pd.DataFrame(
                [
                    {"compound_id": compound_id, **values}
                    for compound_id, values in pose_diagnostics.items()
                ]
            )
            table = table.drop(
                columns=[
                    column
                    for column in diagnostics_frame.columns
                    if column != "compound_id" and column in table.columns
                ],
                errors="ignore",
            ).merge(diagnostics_frame, on="compound_id", how="left")
        st.markdown(f"#### {labels[role]}")
        if role == "ranked_scores" and {
            "compound_id",
            "replicate",
            "best_score_kcal_mol",
        }.issubset(table.columns):
            try:
                import altair as alt

                chart_table = table[
                    ["compound_id", "replicate", "seed", "best_score_kcal_mol"]
                ].copy()
                chart_table["best_score_kcal_mol"] = pd.to_numeric(
                    chart_table["best_score_kcal_mol"], errors="coerce"
                )
                chart_table = chart_table.dropna(subset=["best_score_kcal_mol"])
                compound_order = (
                    chart_table.groupby("compound_id")["best_score_kcal_mol"]
                    .mean()
                    .sort_values()
                    .index.tolist()
                )
                replicate_points = (
                    alt.Chart(chart_table)
                    .mark_circle(size=105, opacity=0.82)
                    .encode(
                        x=alt.X(
                            "compound_id:N",
                            sort=compound_order,
                            title="Compound",
                            axis=alt.Axis(labelAngle=-45, labelLimit=180),
                        ),
                        xOffset=alt.XOffset("replicate:N"),
                        y=alt.Y(
                            "best_score_kcal_mol:Q",
                            title="Docking score (kcal/mol)",
                            scale=alt.Scale(zero=False),
                        ),
                        color=alt.Color(
                            "replicate:N",
                            title="Replicate",
                        ),
                        tooltip=[
                            alt.Tooltip("compound_id:N", title="Compound"),
                            alt.Tooltip("replicate:O", title="Replicate"),
                            alt.Tooltip("seed:O", title="Seed"),
                            alt.Tooltip(
                                "best_score_kcal_mol:Q",
                                title="Score (kcal/mol)",
                                format=".3f",
                            ),
                        ],
                    )
                    .properties(height=360)
                )
                st.markdown("##### Replicate score distribution")
                st.altair_chart(replicate_points, width="stretch")
                st.caption(
                    "Each point is one independent seeded run; points are not connected "
                    "because replicates have no sequential relationship. Lower scores are "
                    "more favorable within this engine and protocol."
                )
            except (ImportError, TypeError, ValueError) as exc:
                st.warning(f"Could not render replicate-score plot: {exc}")
        elif role == "replicate_summary" and {
            "compound_id",
            "mean_score_kcal_mol",
            "sample_sd_score_kcal_mol",
        }.issubset(table.columns):
            try:
                import altair as alt

                summary = table[
                    [
                        "compound_id",
                        "replicate_count",
                        "mean_score_kcal_mol",
                        "sample_sd_score_kcal_mol",
                    ]
                ].copy()
                for column in (
                    "mean_score_kcal_mol",
                    "sample_sd_score_kcal_mol",
                ):
                    summary[column] = pd.to_numeric(
                        summary[column], errors="coerce"
                    )
                summary = summary.dropna(subset=["mean_score_kcal_mol"])
                summary["sample_sd_score_kcal_mol"] = summary[
                    "sample_sd_score_kcal_mol"
                ].fillna(0.0)
                summary["sd_lower"] = (
                    summary["mean_score_kcal_mol"]
                    - summary["sample_sd_score_kcal_mol"]
                )
                summary["sd_upper"] = (
                    summary["mean_score_kcal_mol"]
                    + summary["sample_sd_score_kcal_mol"]
                )
                compound_order = summary.sort_values(
                    "mean_score_kcal_mol"
                )["compound_id"].tolist()
                base = alt.Chart(summary).encode(
                    x=alt.X(
                        "compound_id:N",
                        sort=compound_order,
                        title="Compound",
                        axis=alt.Axis(labelAngle=-45, labelLimit=180),
                    )
                )
                error_bars = base.mark_rule(strokeWidth=3).encode(
                    y=alt.Y(
                        "sd_lower:Q",
                        title="Mean docking score ± sample SD (kcal/mol)",
                        scale=alt.Scale(zero=False),
                    ),
                    y2="sd_upper:Q",
                )
                means = base.mark_point(
                    filled=True,
                    size=120,
                    color="#0f766e",
                ).encode(
                    y=alt.Y(
                        "mean_score_kcal_mol:Q",
                        scale=alt.Scale(zero=False),
                    ),
                    tooltip=[
                        alt.Tooltip("compound_id:N", title="Compound"),
                        alt.Tooltip(
                            "mean_score_kcal_mol:Q",
                            title="Mean score",
                            format=".3f",
                        ),
                        alt.Tooltip(
                            "sample_sd_score_kcal_mol:Q",
                            title="Sample SD",
                            format=".3f",
                        ),
                        alt.Tooltip(
                            "replicate_count:Q",
                            title="Replicates",
                            format=".0f",
                        ),
                    ],
                )
                st.markdown("##### Mean score and replicate variability")
                st.altair_chart(
                    (error_bars + means).properties(height=380),
                    width="stretch",
                )
                st.caption(
                    "Dots are mean docking scores and vertical bars are ±1 sample SD "
                    "(n−1) across independent runs. Smaller SD indicates more consistent "
                    "replicate scores; lower mean scores are more favorable."
                )
                if {
                    "mean_pairwise_pose_rmsd_angstrom",
                    "max_pairwise_pose_rmsd_angstrom",
                }.issubset(table.columns):
                    rmsd_table = table[
                        [
                            "compound_id",
                            "mean_pairwise_pose_rmsd_angstrom",
                            "max_pairwise_pose_rmsd_angstrom",
                        ]
                    ].melt(
                        id_vars="compound_id",
                        var_name="statistic",
                        value_name="pose_rmsd_angstrom",
                    )
                    rmsd_table["statistic"] = rmsd_table["statistic"].map(
                        {
                            "mean_pairwise_pose_rmsd_angstrom": "Mean pairwise RMSD",
                            "max_pairwise_pose_rmsd_angstrom": "Maximum pairwise RMSD",
                        }
                    )
                    pose_rmsd_chart = (
                        alt.Chart(rmsd_table.dropna())
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "compound_id:N",
                                sort=compound_order,
                                title="Compound",
                                axis=alt.Axis(labelAngle=-45, labelLimit=180),
                            ),
                            xOffset="statistic:N",
                            y=alt.Y(
                                "pose_rmsd_angstrom:Q",
                                title="Fixed-frame ligand RMSD (Å)",
                            ),
                            color=alt.Color("statistic:N", title="Statistic"),
                            tooltip=[
                                alt.Tooltip("compound_id:N", title="Compound"),
                                alt.Tooltip("statistic:N", title="Statistic"),
                                alt.Tooltip(
                                    "pose_rmsd_angstrom:Q",
                                    title="RMSD (Å)",
                                    format=".3f",
                                ),
                            ],
                        )
                        .properties(height=360)
                    )
                    threshold = (
                        alt.Chart(pd.DataFrame({"threshold": [2.0]}))
                        .mark_rule(color="#dc2626", strokeDash=[6, 4])
                        .encode(y="threshold:Q")
                    )
                    st.markdown("##### Replicate pose convergence")
                    st.altair_chart(
                        pose_rmsd_chart + threshold,
                        width="stretch",
                    )
                    st.caption(
                        "Symmetry-aware heavy-atom RMSD is measured directly in the "
                        "fixed receptor frame without superposing ligands. The dashed "
                        "2 Å line is a practical warning threshold for divergent placements."
                    )
            except (ImportError, TypeError, ValueError) as exc:
                st.warning(f"Could not render docking-statistics plot: {exc}")
        st.dataframe(table, hide_index=True, width="stretch")
        rendered = True
    if rendered and int(job.result.get("replicates") or 1) > 1:
        st.caption(
            "SD values are sample standard deviations (n−1). The representative "
            "pose is from the run whose score is closest to the replicate median."
        )
    return rendered


def _render_nesso_metric_plots(table: pd.DataFrame, *, role: str) -> None:
    try:
        import altair as alt
    except ImportError:
        return

    identifier = next(
        (
            column
            for column in ("candidate_id", "compound_id")
            if column in table.columns
        ),
        None,
    )
    if identifier is None:
        return

    if role == "affinity_campaign":
        attempt = next(
            (
                column
                for column in ("replicate", "seed")
                if column in table.columns
            ),
            None,
        )
        per_run_groups = (
            (
                "Nesso affinity across independent runs",
                ["affinity_log10_ic50_uM"],
                "Predicted log10(IC50 / µM)",
                False,
            ),
            (
                "Nesso IC50 across independent runs",
                ["ic50_uM"],
                "Predicted IC50 (µM, log scale)",
                True,
            ),
            (
                "Nesso binder probability across independent runs",
                ["binder_probability"],
                "Binder probability",
                False,
            ),
        )
        for title, candidates, y_title, logarithmic in per_run_groups:
            columns = [column for column in candidates if column in table.columns]
            if not columns:
                continue
            chart_columns = [identifier, *([attempt] if attempt else []), *columns]
            chart_table = table[chart_columns].copy()
            for column in columns:
                chart_table[column] = pd.to_numeric(
                    chart_table[column], errors="coerce"
                )
            chart_table = chart_table.melt(
                id_vars=[identifier, *([attempt] if attempt else [])],
                value_vars=columns,
                var_name="metric",
                value_name="value",
            ).dropna(subset=["value"])
            if logarithmic:
                chart_table = chart_table.loc[chart_table["value"] > 0]
            if chart_table.empty:
                continue
            encoding: dict[str, Any] = {
                "x": alt.X(
                    f"{identifier}:N",
                    title="Compound",
                    axis=alt.Axis(labelAngle=-45, labelLimit=180),
                ),
                "y": alt.Y(
                    "value:Q",
                    title=y_title,
                    scale=alt.Scale(type="log" if logarithmic else "linear", zero=False),
                ),
                "tooltip": [
                    alt.Tooltip(f"{identifier}:N", title="Compound"),
                    alt.Tooltip("value:Q", title="Value", format=".4f"),
                ],
            }
            if attempt:
                encoding["xOffset"] = alt.XOffset(f"{attempt}:N")
                encoding["color"] = alt.Color(f"{attempt}:N", title="Run")
                encoding["tooltip"].insert(
                    1, alt.Tooltip(f"{attempt}:N", title="Run")
                )
            st.markdown(f"##### {title}")
            st.altair_chart(
                alt.Chart(chart_table)
                .mark_point(filled=True, size=105, opacity=0.85)
                .encode(**encoding)
                .properties(height=350),
                width="stretch",
            )

        entropy_columns = [
            column
            for column in (
                "cropped_protein_ligand_entropy",
                "protein_protein_entropy",
                "protein_ligand_entropy",
                "ligand_ligand_entropy",
                "cropped_protein_protein_entropy",
            )
            if column in table.columns
        ]
        if entropy_columns:
            chart_columns = [
                identifier,
                *([attempt] if attempt else []),
                *entropy_columns,
            ]
            entropy = table[chart_columns].copy()
            for column in entropy_columns:
                entropy[column] = pd.to_numeric(entropy[column], errors="coerce")
            entropy = entropy.melt(
                id_vars=[identifier, *([attempt] if attempt else [])],
                value_vars=entropy_columns,
                var_name="component",
                value_name="value",
            ).dropna(subset=["value"])
            if not entropy.empty:
                encoding = {
                    "x": alt.X(
                        f"{identifier}:N",
                        title="Compound",
                        axis=alt.Axis(labelAngle=-45, labelLimit=180),
                    ),
                    "y": alt.Y(
                        "value:Q",
                        title="Nesso entropy feature",
                        scale=alt.Scale(zero=False),
                    ),
                    "color": alt.Color("component:N", title="Component"),
                    "tooltip": [
                        alt.Tooltip(f"{identifier}:N", title="Compound"),
                        alt.Tooltip("component:N", title="Component"),
                        alt.Tooltip("value:Q", title="Value", format=".4f"),
                    ],
                }
                if attempt:
                    encoding["shape"] = alt.Shape(f"{attempt}:N", title="Run")
                    encoding["tooltip"].insert(
                        1, alt.Tooltip(f"{attempt}:N", title="Run")
                    )
                st.markdown("##### Nesso model entropy diagnostics")
                st.altair_chart(
                    alt.Chart(entropy)
                    .mark_point(filled=True, size=82, opacity=0.75)
                    .encode(**encoding)
                    .properties(height=370),
                    width="stretch",
                )
        return

    if role != "replicate_summary":
        return

    summary_groups = (
        (
            "Nesso mean affinity with run-to-run uncertainty",
            "mean_affinity_log10_ic50_uM",
            "sample_sd_affinity_log10_ic50_uM",
            "Predicted log10(IC50 / µM)",
        ),
        (
            "Nesso mean binder probability with run-to-run uncertainty",
            "mean_binder_probability",
            "sample_sd_binder_probability",
            "Binder probability",
        ),
    )
    for title, mean_column, sd_column, y_title in summary_groups:
        if mean_column not in table.columns or sd_column not in table.columns:
            continue
        summary = table[[identifier, mean_column, sd_column]].copy()
        summary[mean_column] = pd.to_numeric(summary[mean_column], errors="coerce")
        summary[sd_column] = pd.to_numeric(
            summary[sd_column], errors="coerce"
        ).fillna(0.0)
        summary = summary.dropna(subset=[mean_column])
        if summary.empty:
            continue
        summary["lower"] = summary[mean_column] - summary[sd_column]
        summary["upper"] = summary[mean_column] + summary[sd_column]
        base = alt.Chart(summary).encode(
            x=alt.X(
                f"{identifier}:N",
                title="Compound",
                axis=alt.Axis(labelAngle=-45, labelLimit=180),
            )
        )
        error_bars = base.mark_rule(strokeWidth=3).encode(
            y=alt.Y(
                "lower:Q",
                title=f"{y_title} ± sample SD",
                scale=alt.Scale(zero=False),
            ),
            y2="upper:Q",
        )
        means = base.mark_point(filled=True, size=110, color="#0f766e").encode(
            y=alt.Y(f"{mean_column}:Q", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip(f"{identifier}:N", title="Compound"),
                alt.Tooltip(f"{mean_column}:Q", title="Mean", format=".4f"),
                alt.Tooltip(f"{sd_column}:Q", title="Sample SD", format=".4f"),
            ],
        )
        st.markdown(f"##### {title}")
        st.altair_chart(
            (error_bars + means).properties(height=350), width="stretch"
        )

    ic50_columns = [
        column
        for column in ("geometric_mean_ic50_uM", "arithmetic_mean_ic50_uM")
        if column in table.columns
    ]
    if ic50_columns:
        ic50 = table[[identifier, *ic50_columns]].copy()
        for column in ic50_columns:
            ic50[column] = pd.to_numeric(ic50[column], errors="coerce")
        ic50 = ic50.melt(
            id_vars=identifier,
            value_vars=ic50_columns,
            var_name="statistic",
            value_name="ic50_uM",
        ).dropna(subset=["ic50_uM"])
        ic50 = ic50.loc[ic50["ic50_uM"] > 0]
        if not ic50.empty:
            st.markdown("##### Nesso IC50 summary")
            st.altair_chart(
                alt.Chart(ic50)
                .mark_point(filled=True, size=110)
                .encode(
                    x=alt.X(
                        f"{identifier}:N",
                        title="Compound",
                        axis=alt.Axis(labelAngle=-45, labelLimit=180),
                    ),
                    xOffset=alt.XOffset("statistic:N"),
                    y=alt.Y(
                        "ic50_uM:Q",
                        title="Predicted IC50 (µM, log scale)",
                        scale=alt.Scale(type="log", zero=False),
                    ),
                    color=alt.Color("statistic:N", title="Statistic"),
                    tooltip=[
                        alt.Tooltip(f"{identifier}:N", title="Compound"),
                        alt.Tooltip("statistic:N", title="Statistic"),
                        alt.Tooltip("ic50_uM:Q", title="IC50 (µM)", format=".4f"),
                    ],
                )
                .properties(height=350),
                width="stretch",
            )

    spread_column = "mean_ensemble_spread_log10_ic50_uM"
    if spread_column in table.columns:
        spread = table[[identifier, spread_column]].copy()
        spread[spread_column] = pd.to_numeric(
            spread[spread_column], errors="coerce"
        )
        spread = spread.dropna(subset=[spread_column])
        if not spread.empty:
            st.markdown("##### Nesso ensemble disagreement")
            st.altair_chart(
                alt.Chart(spread)
                .mark_bar(color="#7c3aed")
                .encode(
                    x=alt.X(
                        f"{identifier}:N",
                        title="Compound",
                        axis=alt.Axis(labelAngle=-45, labelLimit=180),
                    ),
                    y=alt.Y(
                        f"{spread_column}:Q",
                        title="Mean ensemble spread (log10 IC50 / µM)",
                    ),
                    tooltip=[
                        alt.Tooltip(f"{identifier}:N", title="Compound"),
                        alt.Tooltip(
                            f"{spread_column}:Q",
                            title="Mean spread",
                            format=".4f",
                        ),
                    ],
                )
                .properties(height=350),
                width="stretch",
            )


def _render_prediction_metric_plots(
    job: JobRecord, table: pd.DataFrame, *, role: str
) -> None:
    if table.empty:
        return
    if job.workflow == "nesso_affinity":
        _render_nesso_metric_plots(table, role=role)
        return
    try:
        import altair as alt
    except ImportError:
        return

    identifier = next(
        (
            column
            for column in ("compound_id", "candidate_id")
            if column in table.columns
        ),
        None,
    )
    if identifier is None:
        return
    attempt = next(
        (
            column
            for column in ("replicate", "model_seed", "prediction_id")
            if column in table.columns
        ),
        None,
    )
    if role == "summary":
        groups = [
            (
                "Structural and interface confidence",
                [
                    "ranking_score",
                    "confidence_score",
                    "iptm",
                    "ptm",
                    "ligand_iptm",
                    "complex_plddt",
                    "complex_iplddt",
                ],
                "Confidence score",
            ),
            (
                "Affinity prediction",
                ["affinity_pred_value"],
                "Predicted log10(IC50 / µM)",
            ),
            (
                "Binder probability",
                ["affinity_probability_binary"],
                "Binder probability",
            ),
        ]
        if job.workflow == "alphafold3_refolding":
            groups.extend(
                [
                    (
                        "Predicted disorder fraction",
                        ["fraction_disordered"],
                        "Fraction disordered",
                    ),
                    (
                        "Clash diagnostic",
                        ["has_clash"],
                        "Clash indicator (0 = no, 1 = yes)",
                    ),
                ]
            )
        for title, candidates, y_title in groups:
            columns = [column for column in candidates if column in table.columns]
            if not columns:
                continue
            chart_columns = [identifier, *([attempt] if attempt else []), *columns]
            chart_table = table[chart_columns].copy()
            for column in columns:
                chart_table[column] = pd.to_numeric(
                    chart_table[column], errors="coerce"
                )
            chart_table = chart_table.melt(
                id_vars=[identifier, *([attempt] if attempt else [])],
                value_vars=columns,
                var_name="metric",
                value_name="value",
            ).dropna(subset=["value"])
            if chart_table.empty:
                continue
            encoding: dict[str, Any] = {
                "x": alt.X(
                    f"{identifier}:N",
                    title="Compound",
                    axis=alt.Axis(labelAngle=-45, labelLimit=180),
                ),
                "y": alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False)),
                "color": alt.Color("metric:N", title="Metric"),
                "tooltip": [
                    alt.Tooltip(f"{identifier}:N", title="Compound"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("value:Q", title="Value", format=".4f"),
                ],
            }
            if attempt:
                encoding["xOffset"] = alt.XOffset(f"{attempt}:N")
                encoding["shape"] = alt.Shape(f"{attempt}:N", title="Attempt")
                encoding["tooltip"].insert(
                    1, alt.Tooltip(f"{attempt}:N", title="Attempt")
                )
            st.markdown(f"##### {title}")
            st.altair_chart(
                alt.Chart(chart_table)
                .mark_point(filled=True, size=95, opacity=0.82)
                .encode(**encoding)
                .properties(height=350),
                width="stretch",
            )
        if (
            job.workflow == "alphafold3_refolding"
            and attempt
            and table[attempt].nunique() > 1
        ):
            confidence_columns = [
                column
                for column in ("ranking_score", "iptm", "ptm")
                if column in table.columns
            ]
            if confidence_columns:
                confidence = table[
                    [identifier, attempt, *confidence_columns]
                ].copy()
                for column in confidence_columns:
                    confidence[column] = pd.to_numeric(
                        confidence[column], errors="coerce"
                    )
                confidence = confidence.melt(
                    id_vars=[identifier, attempt],
                    value_vars=confidence_columns,
                    var_name="metric",
                    value_name="value",
                ).dropna(subset=["value"])
                summary = (
                    confidence.groupby([identifier, "metric"])["value"]
                    .agg(mean="mean", std="std")
                    .reset_index()
                )
                summary["std"] = summary["std"].fillna(0.0)
                summary["lower"] = summary["mean"] - summary["std"]
                summary["upper"] = summary["mean"] + summary["std"]
                base = alt.Chart(summary).encode(
                    x=alt.X(
                        f"{identifier}:N",
                        title="Compound",
                        axis=alt.Axis(labelAngle=-45, labelLimit=180),
                    ),
                    xOffset=alt.XOffset("metric:N"),
                    color=alt.Color("metric:N", title="Metric"),
                )
                bars = base.mark_rule(strokeWidth=3).encode(
                    y=alt.Y(
                        "lower:Q",
                        title="Mean confidence ± sample SD",
                        scale=alt.Scale(zero=False),
                    ),
                    y2="upper:Q",
                )
                means = base.mark_point(filled=True, size=105).encode(
                    y="mean:Q",
                    tooltip=[
                        alt.Tooltip(f"{identifier}:N", title="Compound"),
                        alt.Tooltip("metric:N", title="Metric"),
                        alt.Tooltip("mean:Q", title="Mean", format=".4f"),
                        alt.Tooltip("std:Q", title="Sample SD", format=".4f"),
                    ],
                )
                st.markdown("##### Confidence stability across model-seed attempts")
                st.altair_chart(
                    (bars + means).properties(height=360), width="stretch"
                )
    elif role == "replicate_summary":
        summary_groups = (
            (
                "Boltz-2 affinity across attempts",
                "mean_affinity_pred_value",
                "sample_sd_affinity_pred_value",
                "Predicted log10(IC50 / µM)",
            ),
            (
                "Boltz-2 confidence across attempts",
                "mean_confidence_score",
                "sample_sd_confidence_score",
                "Confidence score",
            ),
            (
                "Boltz-2 binder probability across attempts",
                "mean_binder_probability",
                "sample_sd_binder_probability",
                "Binder probability",
            ),
        )
        for title, mean_column, sd_column, y_title in summary_groups:
            if mean_column not in table.columns or sd_column not in table.columns:
                continue
            summary = table[[identifier, mean_column, sd_column]].copy()
            summary[mean_column] = pd.to_numeric(
                summary[mean_column], errors="coerce"
            )
            summary[sd_column] = pd.to_numeric(
                summary[sd_column], errors="coerce"
            ).fillna(0.0)
            summary = summary.dropna(subset=[mean_column])
            summary["lower"] = summary[mean_column] - summary[sd_column]
            summary["upper"] = summary[mean_column] + summary[sd_column]
            base = alt.Chart(summary).encode(
                x=alt.X(
                    f"{identifier}:N",
                    title="Compound",
                    axis=alt.Axis(labelAngle=-45, labelLimit=180),
                )
            )
            bars = base.mark_rule(strokeWidth=3).encode(
                y=alt.Y("lower:Q", title=f"{y_title} ± sample SD", scale=alt.Scale(zero=False)),
                y2="upper:Q",
            )
            points = base.mark_point(
                filled=True, size=110, color="#0f766e"
            ).encode(
                y=alt.Y(f"{mean_column}:Q", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip(f"{identifier}:N", title="Compound"),
                    alt.Tooltip(f"{mean_column}:Q", title="Mean", format=".4f"),
                    alt.Tooltip(f"{sd_column}:Q", title="Sample SD", format=".4f"),
                ],
            )
            st.markdown(f"##### {title}")
            st.altair_chart((bars + points).properties(height=350), width="stretch")


def _render_refolding_metrics(job: JobRecord) -> bool:
    if job.workflow not in {
        "alphafold3_refolding",
        "boltz2_refolding",
        "nesso_affinity",
    } or job.artifact_manifest is None:
        return False
    labels = {
        "summary": (
            "AlphaFold 3 sample-0 structural and interface confidence"
            if job.workflow == "alphafold3_refolding"
            else "Boltz-2 model-0 predictions per attempt"
        ),
        "affinity_campaign": "Nesso per-run affinity predictions",
        "replicate_summary": "Independent-run summary",
    }
    roles = (
        ("summary", "replicate_summary")
        if job.workflow == "boltz2_refolding"
        else ("summary",)
        if job.workflow == "alphafold3_refolding"
        else ("affinity_campaign", "replicate_summary")
    )
    rendered = False
    for role in roles:
        artifact = next(
            (
                item
                for item in job.artifact_manifest.artifacts
                if item.artifact_type == "prediction_metrics" and item.role == role
            ),
            None,
        )
        path = artifact.resolve(job.run_dir, must_exist=True) if artifact else None
        if path is None:
            continue
        try:
            table = pd.read_csv(path)
        except (OSError, ValueError) as exc:
            st.warning(f"Could not read {labels[role].lower()}: {exc}")
            continue
        if role == "summary":
            table = _representative_prediction_metrics(job, table)
        st.markdown(f"#### {labels[role]}")
        _render_prediction_metric_plots(job, table, role=role)
        st.dataframe(table, hide_index=True, width="stretch")
        rendered = True
    if rendered and job.workflow == "alphafold3_refolding":
        st.caption(
            "Plots and the displayed table use sample 0 from each model-seed attempt. "
            "ipTM and pTM are model-confidence measures for the predicted complex and "
            "interfaces; ranking score orders predictions. They are not binding affinities."
        )
    elif rendered and int(job.result.get("replicates") or 1) > 1:
        if job.workflow == "nesso_affinity":
            st.caption(
                "SD values are sample standard deviations (n−1). Nesso affinity is native "
                "log10(IC50 / µM); the summary also reports arithmetic and geometric IC50 in µM. "
                "Nesso does not emit a predicted structure."
            )
        else:
            st.caption(
                "SD values are sample standard deviations (n−1). Each Boltz-2 independent "
                "run is a separately seeded full prediction. Plots and the viewer use model 0 "
                "from each attempt; the other diffusion models remain available as artifacts."
            )
    return rendered


def _rescoring_table(job: JobRecord) -> pd.DataFrame | None:
    if job.workflow not in {"gnina_rescoring", "boltzina_rescoring"}:
        return None
    path = job.run_dir / "rescoring_scores.csv"
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path).fillna("")
    except (OSError, ValueError):
        return None


def _render_rescoring_metrics(job: JobRecord) -> bool:
    table = _rescoring_table(job)
    if table is None or table.empty:
        return False
    st.markdown("#### Pose rescoring")
    numeric_candidates = (
        (
            "gnina_cnn_score",
            "gnina_cnn_affinity",
            "gnina_empirical_score_kcal_mol",
        )
        if job.workflow == "gnina_rescoring"
        else (
            "boltzina_affinity_log10_ic50_uM",
            "boltzina_binder_probability",
        )
    )
    numeric = [column for column in numeric_candidates if column in table.columns]
    if numeric and "pose_id" in table.columns:
        try:
            import altair as alt

            long = table[["pose_id", *numeric]].copy()
            for column in numeric:
                long[column] = pd.to_numeric(long[column], errors="coerce")
            long = long.melt(
                id_vars=("pose_id",),
                value_vars=numeric,
                var_name="score",
                value_name="value",
            ).dropna(subset=["value"])
            if not long.empty:
                chart = (
                    alt.Chart(long)
                    .mark_circle(size=75, opacity=0.82)
                    .encode(
                        x=alt.X("pose_id:N", sort=None, title="Stored pose"),
                        y=alt.Y("value:Q", title="Native score value"),
                        color=alt.Color("score:N", title="Metric"),
                        tooltip=("pose_id:N", "score:N", "value:Q"),
                    )
                    .properties(height=360)
                )
                st.altair_chart(chart, width="stretch")
        except (ImportError, TypeError, ValueError) as exc:
            st.warning(f"Could not render rescoring plot: {exc}")
    st.dataframe(table, hide_index=True, width="stretch")
    if job.workflow == "gnina_rescoring":
        maximum = job.result.get("maximum_coordinate_displacement_angstrom")
        st.caption(
            "GNINA score-only evaluates each supplied pose without docking or "
            "minimization. "
            + (
                f"Maximum recorded atom displacement was {float(maximum):.4f} Å."
                if maximum is not None
                else "Coordinate displacement could not be verified for every pose."
            )
        )
    else:
        st.caption(
            "Boltzina omits Boltz-2 structure generation and applies its affinity "
            "machinery to the supplied docked complex. affinity_pred_value is "
            "log10(IC50 / µM); binder probability is a separate classifier output."
        )
    return True


def _render_pose_validation_metrics(job: JobRecord) -> bool:
    if job.workflow != "posebusters_validation":
        return False
    summary_path = job.run_dir / "posebusters_summary.csv"
    if not summary_path.is_file():
        return False
    try:
        summary = pd.read_csv(summary_path).fillna("")
    except (OSError, ValueError):
        return False
    if summary.empty:
        return False
    passed = summary["passed_all"].astype(str).str.lower().eq("true")
    metrics = st.columns(4)
    metrics[0].metric("Validated poses", len(summary))
    metrics[1].metric("Passed all checks", int(passed.sum()))
    metrics[2].metric("Failed ≥1 check", int((~passed).sum()))
    metrics[3].metric("Pass rate", f"{100.0 * passed.mean():.1f}%")
    full_path = job.run_dir / "native" / "posebusters_full.csv"
    full = pd.DataFrame()
    if full_path.is_file() and full_path.stat().st_size:
        try:
            full = pd.read_csv(full_path)
        except (OSError, ValueError):
            full = pd.DataFrame()
    try:
        validation_report = json.loads(
            (job.run_dir / "posebusters_report.json").read_text()
        )
    except (OSError, TypeError, ValueError):
        validation_report = {}
    configured_checks = {
        str(value) for value in validation_report.get("applicable_checks") or []
    }
    if not configured_checks:
        st.warning(
            "Legacy PoseBusters adapter output: this run predates applicable-check "
            "classification, so its overall PASS/FAIL summary may include optional "
            "diagnostics and must not be used for scientific decisions. Re-run the "
            "same source poses with the current adapter."
        )
    boolean_columns = []
    for column in full.columns:
        values = set(
            full[column].dropna().astype(str).str.lower().unique()
        )
        if (
            values
            and values.issubset({"true", "false"})
            and (not configured_checks or column in configured_checks)
        ):
            boolean_columns.append(column)
    if boolean_columns:
        check_labels = {
            "mol_pred_loaded": "Predicted ligand loaded",
            "mol_cond_loaded": "Protein context loaded",
            "sanitization": "RDKit sanitization",
            "inchi_convertible": "InChI convertible",
            "all_atoms_connected": "All ligand atoms connected",
            "no_radicals": "No radical atoms",
            "bond_lengths": "Bond lengths within bounds",
            "bond_angles": "Bond angles within bounds",
            "internal_steric_clash": "No internal steric clash",
            "aromatic_ring_flatness": "Aromatic-ring flatness",
            "non-aromatic_ring_non-flatness": "Non-aromatic-ring non-flatness",
            "double_bond_flatness": "Double-bond flatness",
            "internal_energy": "Internal energy within threshold",
            "protein-ligand_maximum_distance": "Ligand remains near protein",
            "minimum_distance_to_protein": "No protein–ligand distance clash",
            "minimum_distance_to_organic_cofactors": "No organic-cofactor distance clash",
            "minimum_distance_to_inorganic_cofactors": "No inorganic-cofactor distance clash",
            "minimum_distance_to_waters": "No water distance clash",
            "volume_overlap_with_protein": "Acceptable protein volume overlap",
            "volume_overlap_with_organic_cofactors": "Acceptable organic-cofactor volume overlap",
            "volume_overlap_with_inorganic_cofactors": "Acceptable inorganic-cofactor volume overlap",
            "volume_overlap_with_waters": "Acceptable water volume overlap",
        }
        rates = pd.DataFrame(
            [
                {
                    "Check": check_labels.get(
                        column, column.replace("_", " ").capitalize()
                    ),
                    "Passed": int(
                        full[column].astype(str).str.lower().eq("true").sum()
                    ),
                    "Failed": int(
                        full[column].astype(str).str.lower().eq("false").sum()
                    ),
                    "Pass rate": (
                        full[column].astype(str).str.lower().eq("true").mean()
                    ),
                }
                for column in boolean_columns
            ]
        ).sort_values(["Pass rate", "Check"])
        st.markdown("#### Applicable physical checks")
        st.dataframe(
            rates,
            hide_index=True,
            width="stretch",
            height=min(820, 38 + 35 * len(rates)),
            column_config={
                "Check": st.column_config.TextColumn("Physical check", width="large"),
                "Passed": st.column_config.NumberColumn("Passed", width="small"),
                "Failed": st.column_config.NumberColumn("Failed", width="small"),
                "Pass rate": st.column_config.ProgressColumn(
                    "Pass rate",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                    width="medium",
                ),
            },
        )
    st.markdown("#### Per-pose validation")
    compact = pd.DataFrame(
        {
            "Compound": summary.get("compound_id", ""),
            "Engine": summary.get("source_engine", ""),
            "Attempt": summary.get("replicate", ""),
            "Prediction": summary.get("prediction", "").astype(str).str.split(":").str[-1],
            "Selected by": summary.get("selection_criterion", ""),
            "Overall": passed.map({True: "PASS", False: "FAIL"}),
            "Passed": summary.get("passed_test_count", ""),
            "Failed": summary.get("failed_test_count", ""),
            "Failed checks": summary.get("failed_checks", "")
            .astype(str)
            .str.replace("_", " ", regex=False),
        }
    )
    st.dataframe(
        compact,
        hide_index=True,
        width="stretch",
        column_config={
            "Compound": st.column_config.TextColumn(width="medium"),
            "Engine": st.column_config.TextColumn(width="small"),
            "Attempt": st.column_config.NumberColumn(width="small"),
            "Prediction": st.column_config.TextColumn(width="medium"),
            "Selected by": st.column_config.TextColumn(width="medium"),
            "Overall": st.column_config.TextColumn(width="small"),
            "Passed": st.column_config.NumberColumn(width="small"),
            "Failed": st.column_config.NumberColumn(width="small"),
            "Failed checks": st.column_config.TextColumn(width="large"),
        },
    )
    if not full.empty:
        with st.expander("Inspect complete native PoseBusters metrics"):
            native_pose_ids = (
                full["pose_id"].astype(str).tolist()
                if "pose_id" in full.columns
                else [str(index) for index in full.index]
            )
            selected_native_pose = st.selectbox(
                "Pose",
                native_pose_ids,
                key=f"posebusters-native-metrics:{job.run_id}",
                format_func=lambda value: (
                    str(
                        full.loc[
                            full["pose_id"].astype(str).eq(value),
                            "compound_id",
                        ].iloc[0]
                    )
                    + " · "
                    + str(
                        full.loc[
                            full["pose_id"].astype(str).eq(value),
                            "source_engine",
                        ].iloc[0]
                    )
                    + " · attempt "
                    + str(
                        full.loc[
                            full["pose_id"].astype(str).eq(value),
                            "replicate",
                        ].iloc[0]
                    )
                    if "pose_id" in full.columns
                    else f"Row {int(value) + 1}"
                ),
            )
            native_row = (
                full.loc[full["pose_id"].astype(str).eq(selected_native_pose)].iloc[0]
                if "pose_id" in full.columns
                else full.iloc[int(selected_native_pose)]
            )
            native_parameters = pd.DataFrame(
                {
                    "Parameter": [
                        str(column).replace("_", " ")
                        for column in full.columns
                    ],
                    "Value": [
                        str(native_row.get(column, ""))
                        for column in full.columns
                    ],
                }
            )
            st.dataframe(
                native_parameters,
                hide_index=True,
                width="stretch",
                height=720,
                column_config={
                    "Parameter": st.column_config.TextColumn(
                        "Parameter", width="large"
                    ),
                    "Value": st.column_config.TextColumn(
                        "Value", width="large"
                    ),
                },
            )
    st.caption(
        "Passing means every applicable binary PoseBusters check passed. "
        "A failed validation is a scientific result, not a failed compute job."
    )
    return True


def _render_interaction_analysis_metrics(job: JobRecord) -> bool:
    if job.workflow not in {
        "native_md_geometry_interactions",
        "plip_interactions",
        "pandamap_interactions",
    }:
        return False
    summary_path = job.run_dir / "interaction_summary.csv"
    interactions_path = job.run_dir / "interactions.csv"
    if not summary_path.is_file():
        return False
    try:
        summary = pd.read_csv(summary_path).fillna("")
        interactions = (
            pd.read_csv(interactions_path).fillna("")
            if interactions_path.is_file() and interactions_path.stat().st_size
            else pd.DataFrame()
        )
    except (OSError, ValueError):
        return False
    if summary.empty:
        return False
    successful = summary["success"].astype(str).str.lower().eq("true")
    metrics = st.columns(4)
    metrics[0].metric("Analyzed complexes / poses", int(successful.sum()))
    metrics[1].metric("Failed complexes / poses", int((~successful).sum()))
    metrics[2].metric(
        "Detected interactions",
        int(pd.to_numeric(summary["interaction_count"], errors="coerce").fillna(0).sum()),
    )
    metrics[3].metric(
        "Compounds",
        int(summary.loc[successful, "compound_id"].astype(str).nunique()),
    )
    if not interactions.empty:
        st.markdown("#### Interaction fingerprint")
        counts = (
            interactions.groupby(["compound_id", "interaction_type"])
            .size()
            .rename("contacts")
            .reset_index()
        )
        import altair as alt

        chart = (
            alt.Chart(counts)
            .mark_bar()
            .encode(
                x=alt.X("compound_id:N", title="Compound", sort=None),
                y=alt.Y("contacts:Q", title="Detected contacts"),
                color=alt.Color("interaction_type:N", title="Interaction"),
                tooltip=("compound_id:N", "interaction_type:N", "contacts:Q"),
            )
            .properties(height=420)
        )
        st.altair_chart(chart, width="stretch")
        residue_counts = (
            interactions.assign(
                residue=(
                    interactions["protein_chain"].astype(str)
                    + ":"
                    + interactions["protein_residue_name"].astype(str)
                    + interactions["protein_residue_number"].astype(str)
                )
            )
            .groupby(["residue", "interaction_type"])
            .size()
            .rename("contacts")
            .reset_index()
            .sort_values("contacts", ascending=False)
        )
        st.markdown("#### Contacted protein residues")
        st.dataframe(
            residue_counts,
            hide_index=True,
            width="stretch",
            height=min(620, 38 + 35 * len(residue_counts)),
        )
    if job.workflow == "pandamap_interactions":
        values = pd.to_numeric(
            summary.get("empirical_delta_g_kcal_mol", pd.Series(dtype=float)),
            errors="coerce",
        )
        if values.notna().any():
            st.markdown("#### PandaMap empirical ΔG estimate")
            delta = summary.loc[values.notna(), ["compound_id", "replicate"]].copy()
            delta["Empirical ΔG (kcal/mol)"] = values[values.notna()].values
            st.dataframe(delta, hide_index=True, width="stretch")
            st.caption(
                "This is PandaMap's empirical interaction-based estimate. It is "
                "not an experimental affinity or a rigorous free-energy calculation."
            )
    st.markdown("#### Per-complex / pose summary")
    st.dataframe(summary, hide_index=True, width="stretch")
    if not interactions.empty:
        with st.expander("Inspect normalized atom/residue interaction rows"):
            normalized_rows = interactions.copy()
            for column in ("distance_angstrom", "angle_degree"):
                if column in normalized_rows:
                    normalized_rows[column] = pd.to_numeric(
                        normalized_rows[column],
                        errors="coerce",
                    )
            st.dataframe(
                normalized_rows,
                hide_index=True,
                width="stretch",
            )
    return True


def _render_generation_metrics(job: JobRecord) -> bool:
    if job.workflow != "molecule_generation":
        return False
    table_path = job.run_dir / "normalized" / "generated_compounds.csv"
    report_path = job.run_dir / "normalized" / "generation_report.json"
    try:
        table = (
            pd.read_csv(table_path).fillna("")
            if table_path.is_file()
            else pd.DataFrame()
        )
        report = (
            json.loads(report_path.read_text())
            if report_path.is_file()
            else {}
        )
    except (OSError, TypeError, ValueError):
        return False
    metrics = st.columns(3)
    metrics[0].metric(
        "Requested", int(report.get("requested_count") or 0)
    )
    metrics[1].metric(
        "Valid native outputs", int(report.get("valid_output_count") or 0)
    )
    metrics[2].metric(
        "Unique valid compounds",
        int(report.get("unique_valid_compound_count") or 0),
    )
    st.caption(
        "This inventory is RDKit-readable and deduplicated by canonical "
        "stereochemistry-aware SMILES. Chemical plausibility, standardized 3D, "
        "and PoseBusters molecule checks are performed by the derived "
        "qualification job before downstream handoff."
    )
    qualification_jobs = sorted(
        (
            candidate
            for candidate in iter_job_records(runs_root())
            if candidate.workflow == "molecule_qualification"
            and candidate.parent_run_id == job.run_id
        ),
        key=lambda candidate: str(candidate.created_at or ""),
    )
    if qualification_jobs:
        qualification = qualification_jobs[-1]
        count = int(
            qualification.result.get("qualified_compound_count")
            or qualification.metadata.get("qualified_compound_count")
            or 0
        )
        st.markdown("#### Downstream qualification")
        status_columns = st.columns(2)
        status_columns[0].metric(
            "Qualification", str(qualification.status).replace("_", " ").title()
        )
        status_columns[1].metric("Qualified compounds", count)
        st.link_button(
            "Open chemical and 3D qualification",
            (
                "./job-results?task_group=molecule-qualification"
                f"&run_id={qualification.run_id}"
            ),
        )
    else:
        st.warning(
            "No chemical and 3D qualification job is associated with this "
            "historical generation run yet."
        )
    if not table.empty:
        st.markdown("#### Generated compound inventory")
        st.dataframe(table, hide_index=True, width="stretch", height=620)
    return True


def _render_molecule_qualification_metrics(job: JobRecord) -> bool:
    if job.workflow != "molecule_qualification":
        return False
    report_path = job.run_dir / "qualified" / "qualification_report.json"
    table_path = job.run_dir / "qualified" / "qualification.csv"
    try:
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        table = (
            pd.read_csv(table_path).fillna("")
            if table_path.is_file()
            else pd.DataFrame()
        )
    except (OSError, TypeError, ValueError):
        return False
    values = (
        ("Input", "input_count"),
        ("Chemical pass", "chemical_pass_count"),
        ("Valid 3D", "conformer_pass_count"),
        ("PB strict pass", "posebusters_pass_count"),
        ("Review warnings", "qualified_with_warning_count"),
        ("Accepted for docking", "qualified_compound_count"),
    )
    for column, (label, key) in zip(st.columns(len(values)), values, strict=True):
        column.metric(label, int(report.get(key) or 0))
    st.caption(
        "Identity starts from canonical stereochemistry-aware SMILES. Passing "
        "molecules receive deterministic ETKDGv3 conformers, MMFF94s/UFF "
        "optimization, and PoseBusters molecule-only validation. An isolated "
        "non-aromatic-ring flatness result is retained as a review warning; core "
        "chemistry and geometry checks remain hard gates. Pocket placement is "
        "intentionally deferred to docking."
    )
    if not table.empty:
        st.markdown("#### Per-compound qualification")
        st.dataframe(table, hide_index=True, width="stretch", height=620)
    return True


def _render_molecule_qualification_viewer(job: JobRecord) -> bool:
    if job.workflow != "molecule_qualification":
        return False
    candidate_dir = job.run_dir / "qualified" / "candidates"
    table_path = job.run_dir / "qualified" / "qualification.csv"
    try:
        table = (
            pd.read_csv(table_path).fillna("")
            if table_path.is_file()
            else pd.DataFrame()
        )
    except (OSError, ValueError):
        table = pd.DataFrame()

    by_compound_id = {
        str(row.get("compound_id") or ""): row
        for _, row in table.iterrows()
    }
    candidates: list[tuple[str, pd.Series]] = []
    candidate_paths = (
        sorted(candidate_dir.glob("*.sdf"))
        if candidate_dir.is_dir()
        else []
    )
    for path in candidate_paths:
        try:
            records = _sdf_records(path)
        except OSError:
            records = []
        if not records:
            continue
        properties = _sdf_properties(records[0])
        compound_id = str(properties.get("compound_id") or path.stem)
        candidates.append(
            (
                records[0],
                by_compound_id.get(compound_id, pd.Series(dtype=object)),
            )
        )
    if not candidates:
        st.warning(
            "This qualification job produced no standardized 3D candidates. "
            "Chemical rejection reasons remain available in the Metrics tab."
        )
        return True

    def status_for(row: pd.Series) -> str:
        status = str(row.get("qualification_status") or "").strip()
        if status:
            return status
        accepted = str(row.get("qualified_for_docking") or "").lower() == "true"
        return "qualified" if accepted else "rejected"

    labels: list[str] = []
    for index, (record, row) in enumerate(candidates):
        properties = _sdf_properties(record)
        compound_id = str(
            row.get("compound_id")
            or properties.get("compound_id")
            or f"compound-{index + 1}"
        )
        labels.append(
            f"{index + 1}. {compound_id} — "
            f"{status_for(row).replace('_', ' ')}"
        )
    selected_label = st.selectbox(
        "3D candidate",
        labels,
        key=f"qualification_candidate_{job.run_id}",
    )
    selected_index = labels.index(selected_label)
    selected_record, row = candidates[selected_index]
    properties = _sdf_properties(selected_record)
    compound_id = str(
        row.get("compound_id")
        or properties.get("compound_id")
        or f"compound-{selected_index + 1}"
    )
    smiles = str(
        row.get("canonical_isomeric_smiles")
        or properties.get("canonical_isomeric_smiles")
        or ""
    )
    status = status_for(row)
    summary = st.columns(4)
    summary[0].metric("Record", f"{selected_index + 1} of {len(candidates)}")
    summary[1].metric("Qualification", status.replace("_", " ").title())
    summary[2].metric("Compound ID", compound_id)
    summary[3].metric(
        "3D method",
        str(row.get("force_field") or properties.get("force_field") or "ETKDGv3"),
    )
    warnings = str(row.get("review_warnings") or "").strip()
    hard_failures = str(
        row.get("chemical_failures")
        or row.get("conformer_generation_error")
        or row.get("posebusters_hard_failures")
        or ""
    ).strip()
    if status == "qualified_with_warning":
        st.warning(
            "Accepted for docking with a geometry review warning: "
            f"{warnings or 'review required'}. Inspect this conformer and confirm "
            "it with a stronger optimization if it is promoted."
        )
    elif status == "rejected":
        st.error(
            "Excluded from downstream handoff: "
            f"{hard_failures or row.get('posebusters_failed_checks') or 'hard gate failed'}"
        )
    else:
        st.success("Passed the chemical, standardized-3D, and hard geometry gates.")
    if smiles:
        st.caption("Canonical stereochemistry-aware SMILES")
        st.code(smiles, language=None)
    detail_columns = [
        column
        for column in (
            "qed",
            "sa_score",
            "molecular_weight",
            "logp",
            "heavy_atom_count",
            "selected_energy",
        )
        if column in row and row.get(column) != ""
    ]
    if detail_columns:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        column.replace("_", " ").title(): row.get(column)
                        for column in detail_columns
                    }
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    try:
        import py3Dmol

        viewer = py3Dmol.view(width=1100, height=650)
        viewer.addModel(selected_record + "\n$$$$\n", "sdf")
        viewer.setStyle(
            {"model": 0},
            {"stick": {"colorscheme": "cyanCarbon", "radius": 0.22}},
        )
        viewer.zoomTo({"model": 0})
        viewer.zoom(0.8)
        render_persistent_3dmol(
            viewer,
            key=f"qualified-compound:{job.run_id}",
            height=670,
        )
    except Exception as exc:
        st.error(f"Qualified-compound preview failed: {exc}")
    st.caption(
        "Cyan: a standardized, force-field-optimized 3D candidate. Warning and "
        "rejected conformers remain visible for diagnosis, but only accepted "
        "records enter the downstream compound set. No pocket pose is implied. "
        "Because molecule qualification rebuilds a free conformer from SMILES, a "
        "campaign receptor/reference overlay would falsely imply a bound pose; "
        "target-context overlays resume after docking or cofolding establishes a "
        "shared protein coordinate frame."
    )
    return True


def _render_generation_compound_viewer(job: JobRecord) -> bool:
    if job.workflow != "molecule_generation":
        return False

    normalized_path = job.run_dir / "normalized" / "generated_compounds.sdf"
    native_root = job.run_dir / "native"
    native_paths = (
        sorted(native_root.rglob("*.sdf"))
        if native_root.is_dir()
        else []
    )
    source_paths = [
        path
        for path in (normalized_path, *native_paths)
        if path.is_file()
    ]
    if not source_paths:
        return False

    table_path = job.run_dir / "normalized" / "generated_compounds.csv"
    try:
        table = (
            pd.read_csv(table_path).fillna("")
            if table_path.is_file()
            else pd.DataFrame()
        )
    except (OSError, ValueError):
        table = pd.DataFrame()

    record_sets: dict[Path, list[str]] = {}
    for path in source_paths:
        try:
            record_sets[path] = _sdf_records(path)
        except OSError:
            record_sets[path] = []

    st.info(
        "**Normalized generated compounds** are RDKit-readable and deduplicated "
        "by stereochemistry-aware canonical SMILES, but are not yet the qualified "
        "downstream set. **Native engine output** is preserved exactly for "
        "diagnostics and can contain invalid or duplicate records. Use the derived "
        "chemical and 3D qualification result for docking/cofolding handoff."
    )

    def source_label(path: Path) -> str:
        count = len(record_sets[path])
        noun = "record" if count == 1 else "records"
        if path == normalized_path:
            return (
                f"Normalized generated compounds — {count} RDKit-valid {noun} "
                "(pre-qualification)"
            )
        return (
            f"Native engine output — "
            f"{path.relative_to(job.run_dir).as_posix()} — {count} raw {noun}"
        )

    source_labels = [source_label(path) for path in source_paths]
    paths_by_label = dict(zip(source_labels, source_paths, strict=True))
    selected_source_label = st.selectbox(
        "Compound set",
        source_labels,
        key=f"generation_compound_set_{job.run_id}",
    )
    selected_path = paths_by_label[selected_source_label]
    records = record_sets[selected_path]
    if not records:
        st.warning("The selected SDF contains no readable records.")
        return True

    native_relative = (
        selected_path.relative_to(native_root).as_posix()
        if selected_path != normalized_path
        else ""
    )
    native_indices = (
        pd.to_numeric(table.get("native_index"), errors="coerce")
        if "native_index" in table
        else pd.Series(dtype=float)
    )

    def matching_row(record_index: int) -> pd.Series | None:
        if table.empty:
            return None
        if selected_path == normalized_path:
            return table.iloc[record_index] if record_index < len(table) else None
        if "native_source" not in table or native_indices.empty:
            return None
        matches = table.loc[
            table["native_source"].astype(str).eq(native_relative)
            & native_indices.eq(record_index)
        ]
        return matches.iloc[0] if not matches.empty else None

    def record_name(record_index: int) -> str:
        record = records[record_index]
        properties = _sdf_properties(record)
        row = matching_row(record_index)
        native_name = next(
            (
                line.strip()
                for line in record.splitlines()
                if line.strip()
            ),
            f"record-{record_index + 1}",
        )
        if selected_path == normalized_path:
            normalized_name = (
                str(row.get("compound_id") or "")
                if row is not None
                else properties.get("compound_id", "")
            )
            return f"{record_index + 1}. {normalized_name or native_name}"
        if row is not None:
            return (
                f"{record_index + 1}. {native_name} — normalized as "
                f"{row.get('compound_id')}"
            )
        return (
            f"{record_index + 1}. {native_name} — raw only "
            "(not in normalized set)"
        )

    compound_options = list(range(len(records)))
    default_index = 0
    requested_compound = str(st.query_params.get("compound_id", "") or "").strip()
    if requested_compound:
        for index in compound_options:
            row = matching_row(index)
            if row is not None and str(row.get("compound_id") or "") == requested_compound:
                default_index = index
                break
    compound_labels = [record_name(index) for index in compound_options]
    indices_by_label = dict(zip(compound_labels, compound_options, strict=True))
    selected_compound_label = st.selectbox(
        "Compound",
        compound_labels,
        index=default_index,
        key=f"generation_compound_{job.run_id}_{selected_path.relative_to(job.run_dir)}",
    )
    selected_index = indices_by_label[selected_compound_label]
    selected_record = records[selected_index] + "\n$$$$\n"
    properties = _sdf_properties(records[selected_index])
    selected_row = matching_row(selected_index)

    compound_id = (
        str(selected_row.get("compound_id") or "")
        if selected_row is not None
        else properties.get("compound_id", "")
    )
    molecule_name = next(
        (
            line.strip()
            for line in records[selected_index].splitlines()
            if line.strip()
        ),
        f"record-{selected_index + 1}",
    )
    smiles = (
        str(selected_row.get("canonical_isomeric_smiles") or "")
        if selected_row is not None
        else properties.get("canonical_isomeric_smiles", "")
    )
    engine = (
        str(selected_row.get("generation_engine") or "")
        if selected_row is not None
        else properties.get("generation_engine", str(job.tool or ""))
    )
    status = (
        "RDKit-valid"
        if selected_path == normalized_path or selected_row is not None
        else "Raw only"
    )
    summary = st.columns(4)
    summary[0].metric("Record", f"{selected_index + 1} of {len(records)}")
    summary[1].metric("Normalization", status)
    summary[2].metric("Compound ID", compound_id or molecule_name)
    summary[3].metric("Engine", engine or "—")
    if status == "Raw only":
        st.warning(
            "This native record is absent from the normalized compound set. It may "
            "be invalid, a stereochemical duplicate, or excluded during output "
            "normalization; do not use it as a downstream campaign input."
        )
    if smiles:
        st.caption("Canonical stereochemistry-aware SMILES")
        st.code(smiles, language=None)

    pocket_path = job.run_dir / "input" / "pocket.pdb"
    reference_path = job.run_dir / "input" / "reference_ligand.sdf"
    receptor_path = job.run_dir / "input" / "target.pdb"
    context_columns = st.columns(4)
    with context_columns[0]:
        show_generated = st.checkbox(
            "Show generated compound",
            value=True,
            key=f"generation_show_generated_{job.run_id}",
            help=(
                "Turn this off to inspect the reference ligand without overlap "
                "from the selected generated structure."
            ),
        )
    with context_columns[1]:
        show_pocket = (
            st.checkbox(
                "Show pocket context",
                value=True,
                key=f"generation_show_pocket_{job.run_id}",
            )
            if pocket_path.is_file()
            else False
        )
    with context_columns[2]:
        show_reference = (
            st.checkbox(
                "Show reference ligand",
                value=True,
                key=f"generation_show_reference_{job.run_id}",
                help=(
                    "Overlay the exact coordinate-bearing ligand recorded in the "
                    "immutable design campaign. It is not another generated result."
                ),
            )
            if reference_path.is_file()
            else False
        )
    with context_columns[3]:
        show_receptor = (
            st.checkbox(
                "Show full receptor overlay",
                value=False,
                key=f"generation_show_receptor_{job.run_id}",
                help=(
                    "Overlay the complete staged receptor. When a pocket is "
                    "available, the receptor is rigidly aligned to its matching "
                    "residue coordinates before display."
                ),
            )
            if receptor_path.is_file()
            else False
        )

    try:
        generation_input = json.loads((job.run_dir / "input.json").read_text())
    except (OSError, TypeError, ValueError):
        generation_input = {}
    pocket_artifact = generation_input.get("pocket_artifact")
    if pocket_path.is_file() and isinstance(pocket_artifact, dict):
        pocket_metadata = pocket_artifact.get("metadata")
        pocket_metadata = (
            pocket_metadata if isinstance(pocket_metadata, dict) else {}
        )
        pocket_method = str(pocket_metadata.get("method") or "recorded")
        descriptors = pocket_metadata.get("descriptors")
        descriptors = descriptors if isinstance(descriptors, dict) else {}
        lining_count = descriptors.get("lining_residue_count")
        source_code = display_job_code(
            "",
            str(pocket_artifact.get("run_id") or ""),
        )
        method_label = (
            "bound-reference-ligand extraction"
            if pocket_method == "bound_ligand"
            else pocket_method.replace("_", " ")
        )
        st.caption(
            "Pocket overlay provenance: selected immutable campaign pocket "
            f"{source_code or '—'} ({method_label}"
            + (
                f", {int(float(lining_count))} lining residues"
                if lining_count not in (None, "")
                else ""
            )
            + "). It is loaded from `input/pocket.pdb`; it is not reconstructed "
            "from the pharmacophore hypothesis."
        )

    receptor_data = ""
    receptor_format = "pdb"
    receptor_alignment: tuple[float, int] | None = None
    if show_receptor:
        if pocket_path.is_file():
            try:
                receptor_data, rmsd, matched_atoms = _aligned_structure_data(
                    str(pocket_path),
                    pocket_path.stat().st_mtime_ns,
                    str(receptor_path),
                    receptor_path.stat().st_mtime_ns,
                )
                receptor_format = "cif"
                receptor_alignment = (rmsd, matched_atoms)
            except Exception as exc:
                st.warning(
                    "The full receptor could not be aligned to the pocket and is "
                    f"therefore not shown: {exc}"
                )
                show_receptor = False
        else:
            receptor_data = receptor_path.read_text(errors="replace")
        if receptor_alignment is not None:
            rmsd, matched_atoms = receptor_alignment
            st.caption(
                "Receptor overlay alignment: "
                f"{matched_atoms} matching pocket/receptor Cα atoms, "
                f"{rmsd:.3f} Å RMSD. "
                + (
                    "The staged structures were already in the same campaign "
                    "coordinate frame; no material movement was introduced."
                    if rmsd <= 0.05
                    else "A rigid protein alignment was applied for display only."
                )
            )

    try:
        import py3Dmol

        viewer = py3Dmol.view(width=1100, height=650)
        next_model = 0
        if show_receptor:
            viewer.addModel(receptor_data, receptor_format)
            viewer.setStyle(
                {"model": next_model, "hetflag": False},
                {
                    "cartoon": {
                        "color": "#9ca3af",
                        "opacity": 0.52,
                    }
                },
            )
            next_model += 1
        if show_pocket:
            viewer.addModel(pocket_path.read_text(errors="replace"), "pdb")
            viewer.setStyle(
                {"model": next_model, "hetflag": False},
                {"cartoon": {"color": "#64748b", "opacity": 0.62}},
            )
            viewer.addStyle(
                {"model": next_model},
                {"line": {"color": "#94a3b8", "opacity": 0.32}},
            )
            next_model += 1
        generated_model: int | None = None
        if show_generated:
            generated_model = next_model
            viewer.addModel(selected_record, "sdf")
            viewer.setStyle(
                {"model": generated_model},
                {"stick": {"colorscheme": "cyanCarbon", "radius": 0.19}},
            )
            next_model += 1
        reference_model: int | None = None
        if show_reference:
            reference_model = next_model
            viewer.addModel(_first_sdf_record(reference_path), "sdf")
            viewer.setStyle(
                {"model": reference_model},
                {
                    "stick": {
                        "colorscheme": "magentaCarbon",
                        "radius": 0.30,
                        "opacity": 1.0,
                    },
                    "sphere": {
                        "colorscheme": "magentaCarbon",
                        "scale": 0.27,
                        "opacity": 0.58,
                    },
                },
            )
            next_model += 1
        focus_model = (
            reference_model
            if show_reference and not show_generated
            else generated_model
            if generated_model is not None
            else reference_model
        )
        if focus_model is not None:
            viewer.zoomTo({"model": focus_model})
        else:
            viewer.zoomTo()
        viewer.zoom(0.78)
        render_persistent_3dmol(
            viewer,
            key=f"generation-compound:{job.run_id}",
            height=670,
        )
    except Exception as exc:
        st.error(f"Generated-compound preview failed: {exc}")

    legend = []
    if show_generated:
        legend.append("selected generated compound is cyan")
    if show_reference:
        legend.append("reference ligand is magenta with translucent atom markers")
    if show_pocket:
        legend.append("pocket context is dark grey")
    if show_receptor:
        legend.append("full receptor is light grey")
    st.caption("Viewer: " + "; ".join(legend) + ".")
    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "-",
        compound_id or molecule_name or f"record-{selected_index + 1}",
    ).strip("-")
    st.download_button(
        "Download selected compound SDF",
        data=selected_record,
        file_name=f"{safe_name or 'generated-compound'}.sdf",
        mime="chemical/x-mdl-sdfile",
        key=f"download_generation_compound_{job.run_id}_{selected_path.name}_{selected_index}",
    )
    st.caption(
        "The per-compound download is created from the selected record in memory. "
        "The stored combined SDF and its relative artifact path are unchanged."
    )
    return True


def _interaction_native_value(value: object, *keys: str) -> str:
    try:
        payload = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    for key in keys:
        candidate = payload.get(key)
        if candidate not in (None, ""):
            return str(candidate)
    return ""


def _interaction_atom_scope(atom_name: object) -> str:
    normalized = str(atom_name or "").strip().upper()
    if not normalized or normalized.startswith("#"):
        return ""
    return "BB" if normalized in {"N", "CA", "C", "O", "OXT"} else "SC"


def _interaction_kind(value: object) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    if "water" in text and "bridge" in text:
        return "water bridge"
    if "salt" in text or "ionic" in text or "charge" in text:
        return "salt bridge"
    if "hydrogen" in text or "hbond" in text:
        return "hydrogen bond"
    if "hydrophob" in text:
        return "hydrophobic"
    if "halogen" in text:
        return "halogen bond"
    if _is_pi_stacking(text):
        return "π–π stacking"
    if "cation pi" in text:
        return "cation–π"
    if "pi cation" in text:
        return "π–cation"
    if "carbon pi" in text:
        return "carbon–π"
    if "donor pi" in text:
        return "donor–π"
    if "amide pi" in text:
        return "amide–π"
    if "alkyl pi" in text:
        return "alkyl–π"
    if "pi" in text or "π" in text:
        return "π interaction"
    if "metal" in text:
        return "metal coordination"
    return "contact"


INTERACTION_DISPLAY_STYLES = {
    "hydrogen bond": ("#800000", "--"),
    "hydrophobic": ("#90ee90", "--"),
    "water bridge": ("#1e90ff", "--"),
    "salt bridge": ("#cc33aa", "-."),
    "halogen bond": ("#e67e22", "-."),
    "π–π stacking": ("#7e22ce", ":"),
    "cation–π": ("#a21caf", ":"),
    "π–cation": ("#a21caf", ":"),
    "carbon–π": ("#9333ea", ":"),
    "donor–π": ("#c026d3", ":"),
    "amide–π": ("#c026d3", ":"),
    "alkyl–π": ("#6d28d9", ":"),
    "π interaction": ("#8e44ad", ":"),
    "metal coordination": ("#c49a00", "-"),
    "contact": ("#aaaaaa", ":"),
}


def _is_pi_stacking(value: object) -> bool:
    text = (
        str(value or "")
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("–", " ")
    )
    return "pi stack" in text or "pi pi" in text or "π π" in text


PROTEIN_AROMATIC_RING_ATOMS = {
    "PHE": (
        ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    ),
    "TYR": (
        ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    ),
    "HIS": (
        ("CG", "ND1", "CD2", "CE1", "NE2"),
    ),
    "TRP": (
        ("CG", "CD1", "NE1", "CE2", "CD2"),
        ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    ),
}


def _protein_aromatic_ring_centroid(
    protein_coordinates: dict[tuple[str, str, str], Any],
    *,
    chain: str,
    residue_number: str,
    residue_name: str,
    representative_atom: str,
):
    candidates = PROTEIN_AROMATIC_RING_ATOMS.get(
        str(residue_name or "").strip().upper(),
        (),
    )
    if representative_atom:
        containing = [
            ring for ring in candidates if representative_atom in ring
        ]
        if containing:
            candidates = tuple(containing)
    for ring in candidates:
        coordinates = [
            protein_coordinates.get((chain, residue_number, atom_name))
            for atom_name in ring
        ]
        if all(value is not None for value in coordinates):
            return np.mean(coordinates, axis=0)
    return None


def _ligand_atom_reference_indices(
    molecule: Any,
    pdb_text: str,
) -> dict[str, int]:
    if molecule is None:
        return {}
    trusted_generated = (
        molecule.HasProp("_interaction_atom_names_trusted")
        and molecule.GetBoolProp("_interaction_atom_names_trusted")
    )
    references: dict[str, int] = {}
    for atom in molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        name = (
            info.GetName().strip()
            if info is not None and info.GetName().strip()
            else atom.GetProp("_TriposAtomName").strip()
            if atom.HasProp("_TriposAtomName")
            else f"{atom.GetSymbol()}{atom.GetIdx() + 1}"
            if trusted_generated
            else ""
        )
        if name:
            references.setdefault(name.upper(), atom.GetIdx())
    coordinate_aliases = pdb_ligand_atom_aliases(pdb_text, molecule)
    references.update({
        str(alias).strip().upper(): int(atom_index)
        for alias, atom_index in coordinate_aliases.items()
    })
    for line in pdb_text.splitlines():
        if line[:6].strip() != "HETATM" or len(line) < 54:
            continue
        atom_name = line[12:16].strip().upper()
        atom_index = references.get(atom_name)
        atom_serial = line[6:11].strip()
        if atom_index is not None and atom_serial:
            references.setdefault(f"#{atom_serial}", atom_index)
    return references


def _ligand_aromatic_ring_indices(
    molecule: Any,
    atom_reference: str,
    reference_indices: dict[str, int],
) -> tuple[int, ...]:
    if molecule is None:
        return ()
    atom_index = reference_indices.get(
        str(atom_reference or "").strip().upper()
    )
    all_rings = [
        tuple(int(value) for value in ring)
        for ring in molecule.GetRingInfo().AtomRings()
    ]
    if atom_index is None:
        aromatic_rings = [
            ring
            for ring in all_rings
            if all(
                molecule.GetAtomWithIdx(index).GetIsAromatic()
                for index in ring
            )
        ]
        # Without PandaMap's representative atom, selecting one ring from a
        # multi-ring ligand would be an unsupported chemical assignment.
        return aromatic_rings[0] if len(aromatic_rings) == 1 else ()
    candidates = [
        ring for ring in all_rings if atom_index in ring
    ]
    aromatic = [
        ring
        for ring in candidates
        if all(molecule.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
    ]
    candidates = aromatic or candidates
    if not candidates:
        return ()
    return min(candidates, key=lambda ring: (len(ring), ring))


def _interaction_ligand_molecule(
    job: JobRecord,
    pose_id: str,
    complex_path: Path,
    *,
    ligand_chain: str = "",
    ligand_residue_name: str = "",
    ligand_residue_number: str = "",
):
    from rdkit import Chem

    def smiles_graph_with_coordinates(template, coordinate_molecule):
        """Keep immutable SMILES bonds while retaining pose coordinates.

        PDB CONECT records and distance-based bond perception are not a
        chemical graph authority.  In particular, inferred bonds can draw
        crossed chords through aromatic rings.  Cofolding outputs preserve
        input atom order, so use that order only when the elemental sequence
        agrees exactly; otherwise retain the sanitized SMILES depiction.
        """
        if template is None or coordinate_molecule is None:
            return None
        coordinate_heavy = Chem.RemoveHs(
            coordinate_molecule, sanitize=False
        )
        if template.GetNumAtoms() != coordinate_heavy.GetNumAtoms():
            return None
        template_elements = [
            atom.GetAtomicNum() for atom in template.GetAtoms()
        ]
        coordinate_elements = [
            atom.GetAtomicNum() for atom in coordinate_heavy.GetAtoms()
        ]
        if template_elements != coordinate_elements:
            return None
        result = Chem.Mol(template)
        if coordinate_heavy.GetNumConformers():
            source_conformer = coordinate_heavy.GetConformer()
            conformer = Chem.Conformer(result.GetNumAtoms())
            conformer.Set3D(True)
            for atom_index in range(result.GetNumAtoms()):
                conformer.SetAtomPosition(
                    atom_index,
                    source_conformer.GetAtomPosition(atom_index),
                )
                source_atom = coordinate_heavy.GetAtomWithIdx(atom_index)
                target_atom = result.GetAtomWithIdx(atom_index)
                info = source_atom.GetPDBResidueInfo()
                if info is not None:
                    target_atom.SetMonomerInfo(info)
            result.RemoveAllConformers()
            result.AddConformer(conformer, assignId=True)
        result.SetBoolProp("_interaction_atom_names_trusted", True)
        result.SetBoolProp("_interaction_smiles_graph_authoritative", True)
        return result

    smiles = ""
    interaction_inputs = job.run_dir / "input" / "interaction_inputs.csv"
    if interaction_inputs.is_file():
        try:
            input_rows = pd.read_csv(interaction_inputs).fillna("")
            matching = input_rows.loc[
                input_rows.get(
                    "pose_id",
                    pd.Series("", index=input_rows.index),
                ).astype(str).eq(pose_id)
            ]
            if not matching.empty:
                smiles = str(matching.iloc[0].get("smiles") or "").strip()
        except (OSError, ValueError):
            pass
    candidates = [
        job.run_dir / "input" / "poses" / f"{pose_id}.sdf",
        *sorted((job.run_dir / "input" / "ligands").glob(f"{pose_id}.*")),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        molecule = (
            Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
            if path.suffix.lower() in {".sdf", ".mol"}
            else None
        )
        if (
            molecule is not None
            and sum(
                atom.GetAtomicNum() != 1
                for atom in molecule.GetAtoms()
            ) <= 256
        ):
            molecule.SetBoolProp("_interaction_atom_names_trusted", True)
            return molecule
    if complex_path.is_file():
        complex_text = complex_path.read_text(errors="replace")
        if is_mmcif_text(complex_text):
            try:
                import gemmi

                structure = gemmi.make_structure_from_block(
                    gemmi.cif.read_string(complex_text).sole_block()
                )
                complex_text = structure.make_pdb_string()
            except (RuntimeError, ValueError):
                pass
        pdb_lines = complex_text.splitlines()
        residue_lines: dict[tuple[str, str, str, str], list[str]] = {}
        for line in pdb_lines:
            if line[:6].strip() != "HETATM" or len(line) < 54:
                continue
            residue_name = line[17:20].strip().upper()
            if residue_name in {"HOH", "WAT", "SOL"}:
                continue
            key = (
                line[21:22].strip(),
                line[22:26].strip(),
                line[26:27].strip(),
                residue_name,
            )
            residue_lines.setdefault(key, []).append(line)
        if residue_lines:
            requested_key = (
                str(ligand_chain).strip(),
                str(ligand_residue_number).strip(),
                "",
                str(ligand_residue_name).strip().upper(),
            )
            selected_lines = residue_lines.get(requested_key)
            if selected_lines is None:
                matching_keys = [
                    key
                    for key in residue_lines
                    if (
                        not requested_key[0]
                        or key[0] == requested_key[0]
                    )
                    and (
                        not requested_key[1]
                        or key[1] == requested_key[1]
                    )
                    and (
                        not requested_key[3]
                        or key[3] == requested_key[3]
                    )
                ]
                candidate_groups = (
                    [residue_lines[key] for key in matching_keys]
                    if matching_keys
                    else list(residue_lines.values())
                )
                selected_lines = max(
                    candidate_groups,
                    key=lambda lines: sum(
                        line[76:78].strip().upper() != "H"
                        for line in lines
                    ),
                )
            selected_serials = {
                line[6:11].strip() for line in selected_lines
            }
            conect_lines = []
            for line in pdb_lines:
                if not line.startswith("CONECT"):
                    continue
                serials = re.findall(r"\d+", line[6:])
                if not serials or serials[0] not in selected_serials:
                    continue
                retained = [
                    serial for serial in serials
                    if serial in selected_serials
                ]
                if retained:
                    conect_lines.append(
                        "CONECT" + "".join(
                            f"{int(serial):5d}" for serial in retained
                        )
                    )
            molecule = Chem.MolFromPDBBlock(
                "\n".join([*selected_lines, *conect_lines, "END"]),
                sanitize=False,
                removeHs=False,
            )
            if molecule is not None:
                template = Chem.MolFromSmiles(smiles) if smiles else None
                authoritative = smiles_graph_with_coordinates(
                    template,
                    molecule,
                )
                if authoritative is not None:
                    return authoritative
                if (
                    template is not None
                    and template.GetNumAtoms() == molecule.GetNumAtoms()
                ):
                    try:
                        from rdkit.Chem import AllChem

                        molecule = AllChem.AssignBondOrdersFromTemplate(
                            template,
                            molecule,
                        )
                    except (ValueError, RuntimeError):
                        pass
                if template is not None:
                    # A chemically valid imported graph is safer for 2D
                    # depiction than a coordinate-inferred graph whose
                    # topology could not be reconciled.
                    template.SetBoolProp(
                        "_interaction_atom_names_trusted",
                        True,
                    )
                    template.SetBoolProp(
                        "_interaction_smiles_graph_authoritative",
                        True,
                    )
                    return template
                molecule.SetBoolProp(
                    "_interaction_atom_names_trusted",
                    True,
                )
                return molecule
    return Chem.MolFromSmiles(smiles) if smiles else None


def _selected_interaction_rows(
    interactions: pd.DataFrame,
    *,
    residue_limit: int,
    include_proximity: bool,
    interaction_kinds: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if interactions.empty:
        return interactions.copy(), []
    table = interactions.copy()
    table["kind"] = table["interaction_type"].map(_interaction_kind)
    if not include_proximity:
        table = table.loc[table["kind"].ne("contact")].copy()
    if interaction_kinds is not None:
        table = table.loc[table["kind"].isin(interaction_kinds)].copy()
    if table.empty:
        return table, []
    table["residue"] = (
        table["protein_residue_name"].astype(str)
        + table["protein_residue_number"].astype(str)
    )
    table["_residue_key"] = (
        table["protein_chain"].astype(str)
        + ":"
        + table["residue"]
    )
    residue_order = (
        table.groupby("_residue_key", sort=False)
        .size()
        .sort_values(ascending=False, kind="stable")
        .head(max(1, int(residue_limit)))
        .index.tolist()
    )
    return (
        table.loc[table["_residue_key"].isin(residue_order)].copy(),
        residue_order,
    )


def _interaction_atom_columns(interactions: pd.DataFrame) -> pd.DataFrame:
    table = interactions.copy()
    required_defaults = {
        "pose_id": "",
        "interaction_type": "",
        "protein_chain": "",
        "protein_residue_name": "",
        "protein_residue_number": "",
        "protein_insertion_code": "",
        "protein_atom_name": "",
        "protein_atom_scope": "",
        "ligand_atom_name": "",
        "distance_angstrom": np.nan,
        "angle_degree": np.nan,
        "native_fields_json": "{}",
        "coordinate_protein_chain": "",
        "coordinate_protein_residue_number": "",
        "coordinate_protein_insertion_code": "",
    }
    for column, default in required_defaults.items():
        if column not in table:
            table[column] = default
    for index, row in table.iterrows():
        protein_atom = str(row.get("protein_atom_name") or "").strip()
        ligand_atom = str(row.get("ligand_atom_name") or "").strip()
        try:
            payload = json.loads(str(row.get("native_fields_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        if not protein_atom:
            protein_atom = next(
                (
                    str(payload.get(key) or "").strip()
                    for key in (
                        "protein_atom",
                        "protatom",
                        "donor_atom",
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
                        "ligand_atom",
                        "ligatom",
                        "ligatom_orig_idx",
                        "lig_idx",
                    )
                    if str(payload.get(key) or "").strip()
                ),
                "",
            )
        kind = _interaction_kind(row.get("interaction_type"))
        protein_serial = ""
        ligand_serial = ""
        if kind == "hydrophobic":
            protein_serial = str(payload.get("protcarbonidx") or "")
            ligand_serial = str(payload.get("ligcarbonidx") or "")
        elif kind == "halogen bond":
            protein_serial = str(payload.get("acc_idx") or "")
            ligand_serial = str(payload.get("don_idx") or "")
        elif kind == "hydrogen bond":
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
                iter(re.findall(r"\d+", str(payload.get("prot_idx_list") or ""))),
                "",
            )
        if not ligand_serial:
            ligand_serial = next(
                iter(re.findall(r"\d+", str(payload.get("lig_idx_list") or ""))),
                "",
            )
        if not protein_atom and protein_serial:
            protein_atom = f"#{protein_serial}"
        if not ligand_atom and ligand_serial:
            ligand_atom = f"#{ligand_serial}"
        table.at[index, "protein_atom_name"] = protein_atom
        table.at[index, "ligand_atom_name"] = ligand_atom
        scope = str(row.get("protein_atom_scope") or "").strip().upper()
        if scope not in {"BB", "SC"}:
            native_sidechain = str(payload.get("sidechain") or "").lower()
            if native_sidechain in {"true", "1", "yes"}:
                scope = "SC"
            elif native_sidechain in {"false", "0", "no"}:
                scope = "BB"
            else:
                scope = _interaction_atom_scope(protein_atom)
        table.at[index, "protein_atom_scope"] = scope
    return table


def _interaction_rows_with_author_numbering(
    job: JobRecord,
    pose_id: str,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    def normalized_text(value: object, *, default: str = "") -> str:
        if value is None or pd.isna(value):
            return default
        return str(value).strip() or default

    table = interactions.copy()
    if table.empty:
        return table
    coordinate_fields = (
        "coordinate_protein_chain",
        "coordinate_protein_residue_number",
        "coordinate_protein_insertion_code",
    )
    has_coordinate_numbering = all(
        field in table for field in coordinate_fields
    )
    for field, fallback in (
        ("coordinate_protein_chain", "protein_chain"),
        (
            "coordinate_protein_residue_number",
            "protein_residue_number",
        ),
        (
            "coordinate_protein_insertion_code",
            "protein_insertion_code",
        ),
    ):
        if field not in table:
            table[field] = table.get(fallback, "")
    reference = job.run_dir / "input" / "reference_target.pdb"
    complex_path = job.run_dir / "prepared" / f"{pose_id}.complex.pdb"
    if not reference.is_file() or not complex_path.is_file():
        return table
    try:
        mapping = sequence_author_residue_mapping(
            complex_path.read_text(errors="replace"),
            reference.read_text(errors="replace"),
        )
    except (OSError, ValueError):
        return table
    policy_version = int(
        job.metadata.get("residue_numbering_policy_version") or 0
    )
    interaction_engine = str(
        job.metadata.get("interaction_engine") or job.tool or ""
    )
    stored_author_numbering = (
        policy_version >= 2
        and interaction_engine != "Native MD geometry"
        and not has_coordinate_numbering
    )
    if stored_author_numbering:
        coordinate_by_author = {
            (
                str(author["chain"]),
                int(author["residue_number"]),
                str(author["insertion_code"]),
                str(author["residue_name"]).upper(),
            ): coordinate
            for coordinate, author in mapping.items()
        }
        for index, row in table.iterrows():
            try:
                author_number = int(
                    float(row["protein_residue_number"])
                )
            except (TypeError, ValueError):
                continue
            author_key = (
                normalized_text(
                    row.get("protein_chain"), default="_"
                ),
                author_number,
                normalized_text(
                    row.get("protein_insertion_code")
                ),
                normalized_text(
                    row.get("protein_residue_name")
                ).upper(),
            )
            coordinate = coordinate_by_author.get(author_key)
            if coordinate is None:
                continue
            table.at[index, "coordinate_protein_chain"] = coordinate[0]
            table.at[
                index, "coordinate_protein_residue_number"
            ] = coordinate[1]
            table.at[
                index, "coordinate_protein_insertion_code"
            ] = coordinate[2]
        return table
    if has_coordinate_numbering and policy_version >= 2:
        return table
    for index, row in table.iterrows():
        try:
            coordinate_number = int(
                float(row["coordinate_protein_residue_number"])
            )
        except (TypeError, ValueError):
            continue
        coordinate_chain = normalized_text(
            row["coordinate_protein_chain"], default="_"
        )
        coordinate_insertion = normalized_text(
            row["coordinate_protein_insertion_code"]
        )
        author = mapping.get((
            coordinate_chain,
            coordinate_number,
            coordinate_insertion,
        ))
        if author is None:
            continue
        table.at[index, "protein_chain"] = author["chain"]
        table.at[index, "protein_residue_number"] = author[
            "residue_number"
        ]
        table.at[index, "protein_insertion_code"] = author[
            "insertion_code"
        ]
        table.at[index, "protein_residue_name"] = author["residue_name"]
    return table


def _add_dashed_3d_interaction(
    viewer: Any,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: str,
    grid: tuple[int, int],
) -> None:
    for first, last in dashed_line_segments(start, end):
        viewer.addCylinder(
            {
                "start": {
                    "x": float(first[0]),
                    "y": float(first[1]),
                    "z": float(first[2]),
                },
                "end": {
                    "x": float(last[0]),
                    "y": float(last[1]),
                    "z": float(last[2]),
                },
                "radius": 0.075,
                "color": color,
                "opacity": 0.95,
                "fromCap": 1,
                "toCap": 1,
            },
            viewer=grid,
        )


def _static_interaction_network_figure(
    interactions: pd.DataFrame,
    *,
    molecule,
    complex_pdb_text: str = "",
    residue_limit: int,
    include_proximity: bool = False,
    interaction_kinds: set[str] | None = None,
    rotation_degrees: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    molecule_scale: float = 1.1,
    ellipse_width_scale: float = 1.1,
    ellipse_height_scale: float = 1.1,
    residue_circle_size: float = 380.0,
    residue_font_size: float = 3.25,
    legend_spacing: float = 0.06,
    canvas_scale: float = 1.0,
):
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    if interactions.empty or molecule is None:
        return None
    molecule = Chem.Mol(molecule)
    pdb_atom_aliases = pdb_ligand_atom_aliases(
        complex_pdb_text,
        molecule,
    )
    atom_reference_indices = _ligand_atom_reference_indices(
        molecule,
        complex_pdb_text,
    )
    rdDepictor.Compute2DCoords(molecule)
    conformer = molecule.GetConformer()
    heavy_atoms = [
        atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    if not heavy_atoms:
        return None
    atom_xy = np.asarray([
        [
            conformer.GetAtomPosition(atom.GetIdx()).x,
            conformer.GetAtomPosition(atom.GetIdx()).y,
        ]
        for atom in heavy_atoms
    ], dtype=float)
    heavy_position = {
        atom.GetIdx(): position for position, atom in enumerate(heavy_atoms)
    }
    ring_groups = [
        [
            heavy_position[atom_index]
            for atom_index in ring
            if atom_index in heavy_position
        ]
        for ring in molecule.GetRingInfo().AtomRings()
    ]
    atom_xy = horizontalize_2d_coordinates(
        atom_xy,
        anchor_groups=ring_groups,
    )
    atom_xy = transform_2d_coordinates(
        atom_xy,
        rotation_degrees=rotation_degrees,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
    )
    extent = np.ptp(atom_xy, axis=0)
    scale = (
        2.2 * max(0.1, float(molecule_scale))
        / max(float(np.max(extent)), 1.0)
    )
    atom_xy *= scale
    atom_names: dict[str, np.ndarray] = {}
    atom_indices: dict[str, np.ndarray] = {}
    trust_generated_names = (
        molecule.HasProp("_interaction_atom_names_trusted")
        and molecule.GetBoolProp("_interaction_atom_names_trusted")
    )
    for atom, coordinates in zip(heavy_atoms, atom_xy, strict=True):
        info = atom.GetPDBResidueInfo()
        name = (
            info.GetName().strip()
            if info is not None and info.GetName().strip()
            else atom.GetProp("_TriposAtomName").strip()
            if atom.HasProp("_TriposAtomName")
            else f"{atom.GetSymbol()}{atom.GetIdx() + 1}"
            if trust_generated_names
            else ""
        )
        if name:
            atom_names[name.upper()] = coordinates
        atom_indices[str(atom.GetIdx() + 1)] = coordinates
        atom_indices[str(atom.GetIdx())] = coordinates
    index_to_xy = {
        atom.GetIdx(): coordinates
        for atom, coordinates in zip(heavy_atoms, atom_xy, strict=True)
    }
    atom_aliases = {
        alias: index_to_xy[atom_index]
        for alias, atom_index in pdb_atom_aliases.items()
        if atom_index in index_to_xy
    }

    def ligand_target(interaction: pd.Series):
        ligand_name = str(
            interaction.get("ligand_atom_name") or ""
        ).strip()
        if _is_pi_stacking(interaction.get("interaction_type")):
            ring = _ligand_aromatic_ring_indices(
                molecule,
                ligand_name,
                atom_reference_indices,
            )
            ring_coordinates = [
                index_to_xy[index]
                for index in ring
                if index in index_to_xy
            ]
            if ring_coordinates:
                return np.mean(ring_coordinates, axis=0)
        target = atom_aliases.get(ligand_name.upper())
        if target is None:
            target = atom_names.get(ligand_name.upper())
        if target is None:
            target = atom_indices.get(ligand_name)
        return target

    fig, axis = plt.subplots(figsize=(5.2, 3.9), dpi=150)
    for bond in molecule.GetBonds():
        begin = index_to_xy.get(bond.GetBeginAtomIdx())
        end = index_to_xy.get(bond.GetEndAtomIdx())
        if begin is None or end is None:
            continue
        axis.plot(
            [begin[0], end[0]],
            [begin[1], end[1]],
            color="#252525",
            linewidth=1.0 + 0.25 * max(0.0, bond.GetBondTypeAsDouble() - 1.0),
            zorder=3,
        )
    atom_colors = {
        "C": "#151515", "N": "#2166c2", "O": "#d73027",
        "S": "#d9a300", "P": "#e67e22", "F": "#37a055",
        "Cl": "#37a055", "Br": "#8c510a", "I": "#7b3294",
    }
    for atom, coordinates in zip(heavy_atoms, atom_xy, strict=True):
        symbol = atom.GetSymbol()
        axis.scatter(
            [coordinates[0]], [coordinates[1]], s=48,
            color=atom_colors.get(symbol, "#666666"),
            edgecolor="white", linewidth=0.35, zorder=4,
        )
        if symbol != "C" or atom.GetFormalCharge():
            axis.text(
                coordinates[0], coordinates[1], symbol,
                color="white", fontsize=4.8, ha="center", va="center",
                zorder=5,
            )
    table = _interaction_atom_columns(interactions)
    table, residue_order = _selected_interaction_rows(
        table,
        residue_limit=residue_limit,
        include_proximity=include_proximity,
        interaction_kinds=interaction_kinds,
    )
    if table.empty:
        return None
    residue_targets: list[np.ndarray] = []
    for residue_key in residue_order:
        targets: list[np.ndarray] = []
        for _, interaction in table.loc[
            table["_residue_key"].eq(residue_key)
        ].iterrows():
            target = ligand_target(interaction)
            if target is not None:
                targets.append(target)
        residue_targets.append(
            np.mean(targets, axis=0) if targets else np.zeros(2)
        )
    residue_positions = distribute_2d_labels(
        residue_targets,
        x_radius=2.45 * max(0.1, float(ellipse_width_scale)),
        y_radius=1.62 * max(0.1, float(ellipse_height_scale)),
    )
    edge_styles = INTERACTION_DISPLAY_STYLES
    used_kinds: set[str] = set()
    for residue_index, (residue_key, residue_position) in enumerate(
        zip(residue_order, residue_positions, strict=True)
    ):
        residue_rows = table.loc[table["_residue_key"].eq(residue_key)]
        residue = str(residue_rows["residue"].iloc[0])
        x_value = float(residue_position[0])
        y_value = float(residue_position[1])
        scopes = sorted({
            str(value)
            for value in residue_rows["protein_atom_scope"]
            if str(value) in {"BB", "SC"}
        })
        axis.scatter(
            [x_value], [y_value], s=max(20.0, float(residue_circle_size)),
            color=_md_residue_node_color(residue),
            edgecolor="#4c738c", linewidth=0.75, zorder=4,
        )
        axis.text(
            x_value, y_value + 0.025, residue,
            fontsize=max(1.0, float(residue_font_size)),
            ha="center", va="center", zorder=5,
        )
        if scopes:
            axis.text(
                x_value, y_value - 0.085, "/".join(scopes),
                fontsize=max(1.0, float(residue_font_size) * 0.8),
                ha="center", va="center", color="#334155", zorder=5,
            )
        kinds = list(dict.fromkeys(residue_rows["kind"].astype(str)))
        for kind_index, kind in enumerate(kinds):
            kind_rows = residue_rows.loc[residue_rows["kind"].eq(kind)]
            stacking_rows = kind_rows.loc[
                kind_rows["interaction_type"].map(_is_pi_stacking)
            ]
            representative = (
                stacking_rows.iloc[0]
                if not stacking_rows.empty
                else kind_rows.iloc[0]
            )
            target = ligand_target(representative)
            if target is None:
                direction = np.asarray([x_value, y_value])
                target = atom_xy[
                    int(np.argmax(atom_xy @ direction))
                ]
            color, linestyle = edge_styles.get(
                kind, edge_styles["contact"]
            )
            axis.add_patch(FancyArrowPatch(
                (float(target[0]), float(target[1])),
                (x_value, y_value),
                connectionstyle=(
                    f"arc3,rad={0.07 * (kind_index + 1) * (-1 if residue_index % 2 else 1)}"
                ),
                arrowstyle="-",
                linestyle=linestyle,
                linewidth=1.25,
                color=color,
                alpha=0.85,
                zorder=1,
            ))
            used_kinds.add(kind)
    legend_handles = [
        plt.Line2D(
            [0], [0], color=edge_styles[kind][0],
            linestyle=edge_styles[kind][1], linewidth=1.5,
            label=kind.title(),
        )
        for kind in edge_styles
        if kind in used_kinds
    ]
    if legend_handles:
        axis.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -max(0.01, float(legend_spacing))),
            ncol=min(4, len(legend_handles)),
            fontsize=5,
            frameon=False,
        )
    axis.set_title("Static ligand interaction network", fontsize=9)
    axis.set_aspect("equal")
    fixed_canvas_scale = max(0.5, float(canvas_scale))
    axis.set_xlim(-2.9 * fixed_canvas_scale, 2.9 * fixed_canvas_scale)
    axis.set_ylim(-2.12 * fixed_canvas_scale, 2.12 * fixed_canvas_scale)
    axis.axis("off")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.91, bottom=0.20)
    return fig


def _render_interaction_analysis_viewer(job: JobRecord) -> bool:
    if job.workflow not in {
        "native_md_geometry_interactions",
        "plip_interactions",
        "pandamap_interactions",
    }:
        return False
    summary_path = job.run_dir / "interaction_summary.csv"
    interactions_path = job.run_dir / "interactions.csv"
    if not summary_path.is_file():
        return False
    try:
        summary = pd.read_csv(summary_path).fillna("")
        interactions = (
            pd.read_csv(interactions_path).fillna("")
            if interactions_path.is_file() and interactions_path.stat().st_size
            else pd.DataFrame()
        )
    except (OSError, ValueError):
        return False
    if summary.empty:
        return False
    summary = summary.copy()
    compound_series = summary.get(
        "compound_id",
        pd.Series("", index=summary.index),
    ).astype(str).str.strip()
    summary["_viewer_compound_id"] = compound_series.where(
        compound_series.ne(""),
        summary["pose_id"].astype(str),
    )
    compound_ids = list(
        dict.fromkeys(summary["_viewer_compound_id"].astype(str).tolist())
    )

    def compound_option_label(compound_id: str) -> str:
        compound_rows = summary.loc[
            summary["_viewer_compound_id"].eq(compound_id)
        ]
        replicate_count = compound_rows.get(
            "replicate",
            pd.Series(range(len(compound_rows)), index=compound_rows.index),
        ).astype(str).nunique()
        return (
            f"{compound_id} ({replicate_count} replicate"
            f"{'' if replicate_count == 1 else 's'})"
        )

    requested_compound = str(st.query_params.get("compound_id", "") or "").strip()
    selected_compound = st.selectbox(
        "Compound",
        compound_ids,
        index=(
            compound_ids.index(requested_compound)
            if requested_compound in compound_ids
            else 0
        ),
        format_func=compound_option_label,
        key=f"interaction-compound:{job.run_id}",
        help=(
            "All visualization panels and the interaction table are restricted "
            "to this compound. Every analyzed replicate remains visible."
        ),
    )
    selected_summary = summary.loc[
        summary["_viewer_compound_id"].eq(selected_compound)
    ].copy()
    selected_engines = {
        str(value).strip().lower()
        for value in selected_summary.get(
            "source_engine",
            pd.Series(dtype=str),
        )
        if str(value).strip()
    }
    if selected_engines == {"gnina"}:
        ranking_label = st.segmented_control(
            "GNINA poses shown",
            ("CNN-ranked poses", "Vina-ranked poses"),
            default="CNN-ranked poses",
            key=f"interaction-gnina-ranking:{job.run_id}",
            help=(
                "Shows at most one selected pose per replicate. A pose selected "
                "by both ranking methods is available in both views."
            ),
        )
        criteria = selected_summary.get(
            "selection_criterion",
            pd.Series("", index=selected_summary.index),
        ).astype(str).str.lower()
        ranking_mask = (
            criteria.str.contains("cnn", regex=False)
            if ranking_label == "CNN-ranked poses"
            else (
                criteria.str.contains("vina", regex=False)
                | criteria.str.contains("empirical", regex=False)
            )
        )
        selected_summary = selected_summary.loc[ranking_mask].copy()
        st.caption(
            f"Showing {len(selected_summary)} {ranking_label.lower()} across "
            f"{selected_summary['replicate'].astype(str).nunique()} replicate(s). "
            "Poses selected by both methods appear in either ranking view."
        )
    pose_ids = list(
        dict.fromkeys(selected_summary["pose_id"].astype(str).tolist())
    )

    def pose_summary(pose_id: str) -> pd.Series:
        return summary.loc[
            summary["pose_id"].astype(str).eq(pose_id)
        ].iloc[0]

    def pose_label(pose_id: str) -> str:
        row = pose_summary(pose_id)
        engine = str(row.get("source_engine") or "unknown engine")
        criterion = str(row.get("selection_criterion") or "").lower()
        ranking = ""
        if engine.strip().lower() == "gnina":
            is_cnn = "cnn" in criterion
            is_vina = "vina" in criterion or "empirical" in criterion
            ranking = (
                "CNN + Vina"
                if is_cnn and is_vina
                else "CNN"
                if is_cnn
                else "Vina"
                if is_vina
                else ""
            )
        return (
            f"{row.get('compound_id') or pose_id} · "
            f"{engine}"
            f"{' · ' + ranking if ranking else ''} · "
            f"attempt {row.get('replicate') or 1}"
        )

    def mapped_pose_rows(pose_id: str) -> pd.DataFrame:
        if interactions.empty or "pose_id" not in interactions:
            return pd.DataFrame()
        rows = interactions.loc[
            interactions["pose_id"].astype(str).eq(pose_id)
        ].copy()
        return _interaction_rows_with_author_numbering(
            job,
            pose_id,
            rows,
        )

    interactions_by_pose = {
        pose_id: mapped_pose_rows(pose_id)
        for pose_id in pose_ids
    }

    def pose_rows(pose_id: str) -> pd.DataFrame:
        return interactions_by_pose.get(pose_id, pd.DataFrame())

    selected_interactions = (
        interactions.loc[
            interactions["pose_id"].astype(str).isin(pose_ids)
        ].copy()
        if not interactions.empty and "pose_id" in interactions
        else pd.DataFrame()
    )
    residue_count = 1
    if not selected_interactions.empty:
        residue_count = max(
            1,
            int(
                selected_interactions[
                    [
                        "protein_chain",
                        "protein_residue_name",
                        "protein_residue_number",
                    ]
                ].drop_duplicates().shape[0]
            ),
        )
    st.markdown("#### Shared interaction display")
    display_controls = st.columns([2, 2, 1])
    residue_limit = int(display_controls[0].number_input(
        "Specific interaction residues shown",
        min_value=1,
        max_value=max(10, residue_count),
        value=min(10, max(1, residue_count)),
        step=1,
        key=f"interaction-network-limit:{job.run_id}",
        help=(
            "This shared residue ranking controls both the static networks and "
            "the highlighted residues and atom-to-atom connectors in 3D."
        ),
    ))
    show_proximity = display_controls[1].checkbox(
        "Show generic proximity contacts",
        value=False,
        key=f"interaction-network-proximity:{job.run_id}",
        help=(
            "Generic distance-cutoff contacts are stored but hidden by default "
            "because they duplicate more specific interactions."
        ),
    )
    matrix_columns = int(display_controls[2].number_input(
        "Matrix columns",
        min_value=3 if len(pose_ids) == 1 else 1,
        max_value=6,
        value=3,
        step=1,
        key=f"interaction-matrix-columns-v2:{job.run_id}",
        help=(
            "The layout keeps unused cells empty, so a single interaction "
            "map retains the same size as one panel in a three-column matrix."
        ),
    ))
    show_3d_residue_labels = st.checkbox(
        "Show residue names in 3D",
        value=True,
        key=f"interaction-3d-residue-labels:{job.run_id}",
        help=(
            "Labels use the imported structure's author residue names and "
            "numbers, while 3D selection continues to use prepared-coordinate "
            "identifiers internally."
        ),
    )
    present_kinds = set(
        selected_interactions.get(
            "interaction_type",
            pd.Series(dtype=str),
        ).map(_interaction_kind)
    )
    specific_kind_options = [
        kind
        for kind in INTERACTION_DISPLAY_STYLES
        if kind != "contact" and kind in present_kinds
    ]
    st.markdown("**Interaction types shown in 2D and 3D**")
    type_keys = {
        kind: f"interaction-type:{job.run_id}:{kind}"
        for kind in specific_kind_options
    }
    for type_key in type_keys.values():
        st.session_state.setdefault(type_key, True)
    all_types_key = f"interaction-type-all:{job.run_id}"
    st.session_state.setdefault(
        all_types_key,
        all(bool(st.session_state[type_key]) for type_key in type_keys.values()),
    )

    def set_all_interaction_types() -> None:
        show_all = bool(st.session_state[all_types_key])
        for type_key in type_keys.values():
            st.session_state[type_key] = show_all

    def update_all_interaction_types() -> None:
        st.session_state[all_types_key] = all(
            bool(st.session_state[type_key])
            for type_key in type_keys.values()
        )

    st.checkbox(
        "Show all interaction types",
        key=all_types_key,
        on_change=set_all_interaction_types,
    )
    type_columns = st.columns(max(1, min(4, len(specific_kind_options))))
    selected_interaction_kinds = {
        kind
        for index, kind in enumerate(specific_kind_options)
        if type_columns[index % len(type_columns)].checkbox(
            kind.title().replace("Π", "π"),
            key=type_keys[kind],
            on_change=update_all_interaction_types,
        )
    }
    if show_proximity and "contact" in present_kinds:
        selected_interaction_kinds.add("contact")
    orientation = st.columns(3)
    rotation_degrees = float(orientation[0].number_input(
        "Rotate ligand (°)",
        min_value=-180,
        max_value=180,
        value=0,
        step=5,
        key=f"interaction-network-rotation:{job.run_id}",
    ))
    flip_horizontal = orientation[1].checkbox(
        "Flip horizontally",
        value=False,
        key=f"interaction-network-flip-horizontal:{job.run_id}",
    )
    flip_vertical = orientation[2].checkbox(
        "Flip vertically",
        value=False,
        key=f"interaction-network-flip-vertical:{job.run_id}",
    )
    with st.expander("Network appearance"):
        appearance_defaults = load_network_appearance()
        sizing = st.columns(4)
        molecule_scale = float(sizing[0].number_input(
            "Molecule size", min_value=0.50, max_value=2.00,
            value=float(appearance_defaults["molecule_scale"]), step=0.05,
            key=f"interaction-network-molecule-size:{job.run_id}",
        ))
        ellipse_width_scale = float(sizing[1].number_input(
            "Ellipse width", min_value=0.50, max_value=2.00,
            value=float(appearance_defaults["ellipse_width_scale"]), step=0.05,
            key=f"interaction-network-ellipse-width:{job.run_id}",
        ))
        ellipse_height_scale = float(sizing[2].number_input(
            "Ellipse height", min_value=0.50, max_value=2.00,
            value=float(appearance_defaults["ellipse_height_scale"]), step=0.05,
            key=f"interaction-network-ellipse-height:{job.run_id}",
        ))
        canvas_scale = float(sizing[3].number_input(
            "Canvas scale", min_value=0.75, max_value=2.00,
            value=float(appearance_defaults["canvas_scale"]), step=0.05,
            key=f"interaction-network-canvas-scale:{job.run_id}",
        ))
        labels = st.columns(3)
        residue_circle_size = float(labels[0].number_input(
            "Residue circle size", min_value=100, max_value=600,
            value=int(appearance_defaults["residue_circle_size"]), step=10,
            key=f"interaction-network-residue-size:{job.run_id}",
        ))
        residue_font_size = float(labels[1].number_input(
            "Residue font size", min_value=2.0, max_value=8.0,
            value=float(appearance_defaults["residue_font_size"]), step=0.25,
            key=f"interaction-network-residue-font:{job.run_id}",
        ))
        legend_spacing = float(labels[2].number_input(
            "Legend separation", min_value=0.02, max_value=0.30,
            value=float(appearance_defaults["legend_spacing"]), step=0.02,
            key=f"interaction-network-legend-spacing:{job.run_id}",
        ))
        if st.button(
            "Save as global network default",
            type="primary",
            key=f"interaction-network-save-default:{job.run_id}",
        ):
            try:
                target = save_network_appearance({
                    "molecule_scale": molecule_scale,
                    "ellipse_width_scale": ellipse_width_scale,
                    "ellipse_height_scale": ellipse_height_scale,
                    "canvas_scale": canvas_scale,
                    "residue_circle_size": residue_circle_size,
                    "residue_font_size": residue_font_size,
                    "legend_spacing": legend_spacing,
                })
            except OSError as exc:
                st.error(f"Could not save global network defaults: {exc}")
            else:
                st.success(
                    "Saved globally for all interaction result pages "
                    f"({target.name})."
                )

    displayed_interaction_count = 0
    displayed_residues: set[tuple[str, str]] = set()
    for pose_id in pose_ids:
        displayed_rows, _ = _selected_interaction_rows(
            pose_rows(pose_id),
            residue_limit=residue_limit,
            include_proximity=show_proximity,
            interaction_kinds=selected_interaction_kinds,
        )
        displayed_interaction_count += len(displayed_rows)
        displayed_residues.update(
            (
                str(row["protein_chain"]),
                str(row["protein_residue_number"]),
            )
            for _, row in displayed_rows.iterrows()
        )
    status = st.columns(3)
    status[0].metric("Displayed interactions", displayed_interaction_count)
    status[1].metric("Displayed residues", len(displayed_residues))
    status[2].metric(
        "Analyses passed",
        f"{sum(str(pose_summary(value).get('success')).lower() == 'true' for value in pose_ids)}/{len(pose_ids)}",
    )

    st.markdown("#### Static interaction networks")
    static_columns = st.columns(matrix_columns)
    for pose_index, pose_id in enumerate(pose_ids):
        complex_path = job.run_dir / "prepared" / f"{pose_id}.complex.pdb"
        current_pose_interactions = pose_rows(pose_id)
        complex_pdb_text = (
            complex_path.read_text(errors="replace")
            if complex_path.is_file()
            else ""
        )
        with static_columns[pose_index % matrix_columns]:
            st.caption(pose_label(pose_id))
            figure = _static_interaction_network_figure(
                current_pose_interactions,
                molecule=_interaction_ligand_molecule(
                    job,
                    pose_id,
                    complex_path,
                    ligand_chain=str(
                        pose_summary(pose_id).get("ligand_chain") or ""
                    ),
                    ligand_residue_name=str(
                        pose_summary(pose_id).get(
                            "ligand_residue_name"
                        ) or ""
                    ),
                    ligand_residue_number=str(
                        pose_summary(pose_id).get(
                            "ligand_residue_number"
                        ) or ""
                    ),
                ),
                complex_pdb_text=complex_pdb_text,
                residue_limit=residue_limit,
                include_proximity=show_proximity,
                interaction_kinds=selected_interaction_kinds,
                rotation_degrees=rotation_degrees,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical,
                molecule_scale=molecule_scale,
                ellipse_width_scale=ellipse_width_scale,
                ellipse_height_scale=ellipse_height_scale,
                residue_circle_size=residue_circle_size,
                residue_font_size=residue_font_size,
                legend_spacing=legend_spacing,
                canvas_scale=canvas_scale,
            )
            if figure is None:
                st.info("No interactions match the current display settings.")
            else:
                st.pyplot(figure, width="stretch")
                plt.close(figure)
    st.caption(
        "Static panels share the same residue limit, proximity setting, "
        "orientation and appearance. BB and SC denote backbone and side chain."
    )

    complex_pose_ids = [
        pose_id
        for pose_id in pose_ids
        if (job.run_dir / "prepared" / f"{pose_id}.complex.pdb").is_file()
    ]
    if complex_pose_ids:
        st.markdown("#### Linked 3D interaction views")
        try:
            import py3Dmol

            columns = matrix_columns
            rows = math.ceil(len(complex_pose_ids) / columns)
            viewer = py3Dmol.view(
                width=1650,
                height=max(420, 390 * rows),
                viewergrid=(rows, columns),
                linked=True,
            )
            interaction_colors = {
                kind: style[0]
                for kind, style in INTERACTION_DISPLAY_STYLES.items()
            }
            alignment_reference = (
                job.run_dir / "input" / "reference_target.pdb"
            )
            alignment_results: list[tuple[float, int]] = []
            for pose_index, pose_id in enumerate(complex_pose_ids):
                grid = (pose_index // columns, pose_index % columns)
                complex_path = (
                    job.run_dir / "prepared" / f"{pose_id}.complex.pdb"
                )
                pdb_text = complex_path.read_text(errors="replace")
                ligand_molecule = _interaction_ligand_molecule(
                    job,
                    pose_id,
                    complex_path,
                    ligand_chain=str(
                        pose_summary(pose_id).get("ligand_chain") or ""
                    ),
                    ligand_residue_name=str(
                        pose_summary(pose_id).get(
                            "ligand_residue_name"
                        ) or ""
                    ),
                    ligand_residue_number=str(
                        pose_summary(pose_id).get(
                            "ligand_residue_number"
                        ) or ""
                    ),
                )
                ligand_reference_indices = _ligand_atom_reference_indices(
                    ligand_molecule,
                    pdb_text,
                )
                ligand_pdb_alias_indices = pdb_ligand_atom_aliases(
                    pdb_text,
                    ligand_molecule,
                )
                if alignment_reference.is_file():
                    try:
                        pdb_text, alignment_rmsd, matched_atoms = (
                            aligned_structure_data(
                                str(alignment_reference),
                                alignment_reference.stat().st_mtime_ns,
                                str(complex_path),
                                complex_path.stat().st_mtime_ns,
                            )
                        )
                    except (OSError, RuntimeError, ValueError):
                        pass
                    else:
                        alignment_results.append(
                            (alignment_rmsd, matched_atoms)
                        )
                structure_format = (
                    "cif"
                    if is_mmcif_text(pdb_text)
                    else "pdb"
                )
                viewer.addModel(pdb_text, structure_format, viewer=grid)
                viewer.setStyle(
                    {"hetflag": False},
                    {"cartoon": {"color": "#d1d5db", "opacity": 0.85}},
                    viewer=grid,
                )
                viewer.setStyle(
                    {"hetflag": True},
                    {"stick": {"colorscheme": "cyanCarbon", "radius": 0.22}},
                    viewer=grid,
                )
                protein_coordinates, ligand_coordinates = (
                    pdb_interaction_atom_coordinates(pdb_text)
                )
                ligand_coordinates_by_index = {}
                for alias, atom_index in ligand_pdb_alias_indices.items():
                    coordinates = ligand_coordinates.get(
                        str(alias).strip().upper()
                    )
                    if coordinates is not None:
                        ligand_coordinates_by_index.setdefault(
                            int(atom_index),
                            coordinates,
                        )
                selected_rows, _ = _selected_interaction_rows(
                    _interaction_atom_columns(pose_rows(pose_id)),
                    residue_limit=residue_limit,
                    include_proximity=show_proximity,
                    interaction_kinds=selected_interaction_kinds,
                )
                contacted_columns = [
                    "coordinate_protein_chain",
                    "coordinate_protein_residue_number",
                    "protein_residue_name",
                    "protein_residue_number",
                ]
                contacted = (
                    selected_rows[contacted_columns].drop_duplicates()
                    if not selected_rows.empty
                    else pd.DataFrame(columns=contacted_columns)
                )
                for _, residue in contacted.iterrows():
                    residue_number = str(
                        residue["coordinate_protein_residue_number"]
                    ).strip()
                    if not residue_number:
                        continue
                    selector: dict[str, object] = {
                        "hetflag": False,
                        "resi": residue_number,
                    }
                    chain = str(
                        residue["coordinate_protein_chain"]
                    ).strip()
                    if chain:
                        selector["chain"] = chain
                    viewer.addStyle(
                        selector,
                        {"stick": {"color": "#f59e0b", "radius": 0.12}},
                        viewer=grid,
                    )
                    if show_3d_residue_labels:
                        residue_atoms = [
                            coordinates
                            for (
                                atom_chain,
                                atom_residue,
                                atom_name,
                            ), coordinates in protein_coordinates.items()
                            if atom_chain == chain
                            and atom_residue == residue_number
                            and not atom_name.startswith("#")
                        ]
                        if residue_atoms:
                            label_position = np.mean(
                                residue_atoms,
                                axis=0,
                            )
                            viewer.addLabel(
                                (
                                    f"{residue['protein_residue_name']}"
                                    f"{residue['protein_residue_number']}"
                                ),
                                {
                                    "position": {
                                        "x": float(label_position[0]),
                                        "y": float(label_position[1]),
                                        "z": float(label_position[2]),
                                    },
                                    "fontSize": 11,
                                    "fontColor": "#111827",
                                    "backgroundColor": "#ffffff",
                                    "backgroundOpacity": 0.78,
                                    "borderColor": "#f59e0b",
                                    "borderThickness": 1,
                                },
                                viewer=grid,
                            )
                for _, interaction in selected_rows.iterrows():
                    chain = str(
                        interaction["coordinate_protein_chain"]
                    ).strip()
                    residue_number = str(
                        interaction[
                            "coordinate_protein_residue_number"
                        ]
                    ).strip()
                    protein_atom = str(
                        interaction.get("protein_atom_name") or ""
                    ).strip().upper()
                    ligand_atom = str(
                        interaction.get("ligand_atom_name") or ""
                    ).strip().upper()
                    kind = str(interaction["kind"])
                    start = protein_coordinates.get(
                        (chain, residue_number, protein_atom)
                    )
                    ligand_atom_index = ligand_reference_indices.get(
                        ligand_atom
                    )
                    end = (
                        ligand_coordinates_by_index.get(ligand_atom_index)
                        if ligand_atom_index is not None
                        else None
                    )
                    if end is None:
                        end = ligand_coordinates.get(ligand_atom)
                    if _is_pi_stacking(
                        interaction.get("interaction_type")
                    ):
                        protein_ring_center = (
                            _protein_aromatic_ring_centroid(
                                protein_coordinates,
                                chain=chain,
                                residue_number=residue_number,
                                residue_name=str(
                                    interaction.get(
                                        "protein_residue_name"
                                    ) or ""
                                ),
                                representative_atom=protein_atom,
                            )
                        )
                        ligand_ring = _ligand_aromatic_ring_indices(
                            ligand_molecule,
                            ligand_atom,
                            ligand_reference_indices,
                        )
                        ligand_ring_coordinates = []
                        for atom_index in ligand_ring:
                            coordinates = ligand_coordinates_by_index.get(
                                atom_index
                            )
                            if coordinates is not None:
                                ligand_ring_coordinates.append(
                                    coordinates
                                )
                        if protein_ring_center is not None:
                            start = protein_ring_center
                        if ligand_ring_coordinates:
                            end = np.mean(
                                ligand_ring_coordinates,
                                axis=0,
                            )
                    if (
                        (start is None or end is None)
                        and (not protein_atom or not ligand_atom)
                    ):
                        fallback = closest_residue_ligand_atom_pair(
                            protein_coordinates,
                            ligand_coordinates,
                            chain=chain,
                            residue_number=residue_number,
                        )
                        if fallback is not None:
                            start, end = fallback
                    if start is None or end is None:
                        continue
                    _add_dashed_3d_interaction(
                        viewer,
                        start,
                        end,
                        color=interaction_colors.get(
                            kind, interaction_colors["contact"]
                        ),
                        grid=grid,
                    )
                viewer.zoomTo({"hetflag": True}, viewer=grid)
                viewer.zoom(0.82, viewer=grid)
            render_persistent_3dmol(
                viewer,
                key=f"interaction-complex-matrix:{job.run_id}",
                height=max(440, 400 * rows),
            )
            alignment_caption = ""
            if alignment_results:
                alignment_caption = (
                    " Complexes are rigidly aligned to the immutable imported "
                    "target protein "
                    f"({min(value[1] for value in alignment_results)} matched "
                    "Cα atoms; maximum post-alignment RMSD "
                    f"{max(value[0] for value in alignment_results):.3f} Å)."
                )
            st.caption(
                "All 3D cameras are linked. Ligands are cyan, proteins grey, "
                "displayed residues orange, and dashed atom-to-atom connectors "
                "use the same interaction colors as the static networks."
                + alignment_caption
            )
        except Exception as exc:
            st.warning(f"Interaction structure matrix failed: {exc}")

    native_images = [
        (
            pose_id,
            job.run_dir / "native" / pose_id / "pandamap.png",
        )
        for pose_id in pose_ids
        if (job.run_dir / "native" / pose_id / "pandamap.png").is_file()
    ]
    if native_images:
        st.markdown("#### Native PandaMap diagrams")
        image_columns = st.columns(matrix_columns)
        for image_index, (pose_id, image_path) in enumerate(native_images):
            with image_columns[image_index % matrix_columns]:
                st.caption(pose_label(pose_id))
                st.image(str(image_path), width="stretch")

    if not selected_interactions.empty:
        st.markdown("#### Interactions for selected compound")
        pose_tables = [
            pose_rows(pose_id)
            for pose_id in pose_ids
            if not pose_rows(pose_id).empty
        ]
        table = _interaction_atom_columns(
            pd.concat(pose_tables, ignore_index=True)
            if pose_tables
            else pd.DataFrame()
        )
        if table.empty:
            st.info("No interaction rows are available for these poses.")
            return True
        table["Pose"] = table["pose_id"].astype(str).map(pose_label)
        table["Protein residue"] = (
            table["protein_chain"].astype(str)
            + ":"
            + table["protein_residue_name"].astype(str)
            + table["protein_residue_number"].astype(str)
        )

        table["Ligand atom"] = table["ligand_atom_name"].astype(str)
        table["Protein atom"] = table["protein_atom_name"].astype(str)
        table["Protein region"] = table["protein_atom_scope"].astype(str)
        display_columns = [
            "Pose",
            "interaction_type",
            "Protein residue",
            "Ligand atom",
            "Protein atom",
            "Protein region",
            "distance_angstrom",
            "angle_degree",
        ]
        display_table = table[
            [column for column in display_columns if column in table]
        ].copy()
        for numeric_column in ("distance_angstrom", "angle_degree"):
            if numeric_column in display_table:
                display_table[numeric_column] = pd.to_numeric(
                    display_table[numeric_column],
                    errors="coerce",
                )
        st.dataframe(
            display_table.rename(
                columns={
                    "interaction_type": "Interaction",
                    "distance_angstrom": "Distance (Å)",
                    "angle_degree": "Angle (°)",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No interactions were reported for the selected compound.")
    return True


def _render_pose_validation_viewer(job: JobRecord) -> bool:
    if job.workflow != "posebusters_validation":
        return False
    input_path = job.run_dir / "input" / "validation_inputs.csv"
    summary_path = job.run_dir / "posebusters_summary.csv"
    if not input_path.is_file() or not summary_path.is_file():
        return False
    try:
        inputs = pd.read_csv(input_path).fillna("")
        summary = pd.read_csv(summary_path).fillna("")
    except (OSError, ValueError):
        return False
    if inputs.empty or summary.empty or "pose_id" not in inputs.columns:
        return False
    table = inputs.merge(
        summary,
        on=[
            column
            for column in (
                "pose_id",
                "compound_id",
                "source_engine",
                "source_kind",
                "replicate",
                "prediction",
            )
            if column in inputs.columns and column in summary.columns
        ],
        how="left",
        suffixes=("", "_result"),
    )
    pose_ids = table["pose_id"].astype(str).tolist()
    requested_compound = str(st.query_params.get("compound_id", "") or "").strip()
    pose_index = next(
        (
            index
            for index, value in enumerate(table["compound_id"].astype(str))
            if value == requested_compound
        ),
        0,
    )
    pose_id = st.selectbox(
        "Validated pose",
        pose_ids,
        index=pose_index,
        format_func=lambda value: (
            f"{table.loc[table['pose_id'].astype(str).eq(value), 'compound_id'].iloc[0]}"
            f" · {table.loc[table['pose_id'].astype(str).eq(value), 'source_engine'].iloc[0]}"
            f" · replicate "
            f"{table.loc[table['pose_id'].astype(str).eq(value), 'replicate'].iloc[0]}"
            + (
                " · "
                + str(
                    table.loc[
                        table["pose_id"].astype(str).eq(value),
                        "selection_criterion",
                    ].iloc[0]
                )
                if "selection_criterion" in table
                and str(
                    table.loc[
                        table["pose_id"].astype(str).eq(value),
                        "selection_criterion",
                    ].iloc[0]
                )
                else ""
            )
        ),
        key=f"pose-validation-viewer:{job.run_id}",
    )
    selected = table.loc[table["pose_id"].astype(str).eq(pose_id)].iloc[0]
    try:
        validation_report = json.loads(
            (job.run_dir / "posebusters_report.json").read_text()
        )
    except (OSError, TypeError, ValueError):
        validation_report = {}
    if not validation_report.get("applicable_checks"):
        st.warning(
            "This historical run predates the applicable-check fix. Its displayed "
            "overall PASS/FAIL status is not scientifically valid; use a current "
            "re-run for interpretation."
        )
    passed = str(selected.get("passed_all") or "").lower() == "true"
    status_columns = st.columns(4)
    status_columns[0].metric("Overall", "PASS" if passed else "FAIL")
    status_columns[1].metric(
        "Checks passed", int(selected.get("passed_test_count") or 0)
    )
    status_columns[2].metric(
        "Checks failed", int(selected.get("failed_test_count") or 0)
    )
    status_columns[3].metric(
        "Replicate", selected.get("replicate") or "—"
    )
    preparation_error = str(selected.get("input_preparation_error") or "")
    failed_checks = str(selected.get("failed_checks") or "")
    if preparation_error:
        st.error(f"Input preparation failed: {preparation_error}")
    elif failed_checks:
        st.warning("Failed checks: " + failed_checks.replace(";", " ·"))
    else:
        st.success("All applicable PoseBusters checks passed.")

    pose_path = job.run_dir / str(selected.get("mol_pred") or "")
    receptor_path = job.run_dir / str(selected.get("mol_cond") or "")
    complex_path = job.run_dir / str(selected.get("complex_file") or "")
    prepared_ligand = job.run_dir / "prepared" / f"{pose_id}.ligand.sdf"
    prepared_protein = job.run_dir / "prepared" / f"{pose_id}.protein.pdb"
    if prepared_ligand.is_file() and prepared_protein.is_file():
        pose_path, receptor_path = prepared_ligand, prepared_protein
    elif complex_path.is_file():
        try:
            import py3Dmol

            viewer = py3Dmol.view(width=1100, height=650)
            file_format = (
                "cif"
                if complex_path.suffix.lower() in {".cif", ".mmcif"}
                else "pdb"
            )
            viewer.addModel(complex_path.read_text(errors="replace"), file_format)
            viewer.setStyle(
                {"model": 0, "hetflag": False},
                {"cartoon": {"color": "#cbd5e1", "opacity": 0.9}},
            )
            viewer.setStyle(
                {"model": 0, "hetflag": True},
                {"stick": {"colorscheme": "cyanCarbon", "radius": 0.22}},
            )
            viewer.zoomTo({"model": 0, "hetflag": True})
            viewer.zoom(0.72)
            render_persistent_3dmol(
                viewer,
                key=f"pose-validation-complex:{job.run_id}:{pose_id}",
                height=670,
            )
            return True
        except Exception as exc:
            st.error(f"Validated-complex preview failed: {exc}")
            return True
    if not pose_path.is_file() or not receptor_path.is_file():
        st.info("No complete receptor–ligand visualization is available for this row.")
        return True
    st.caption(
        "Validated complex: the exact receptor context supplied to PoseBusters is "
        "grey and the tested ligand pose is cyan. The view is focused on the ligand."
    )
    try:
        import py3Dmol

        viewer = py3Dmol.view(width=1100, height=650)
        receptor_format = (
            "cif"
            if receptor_path.suffix.lower() in {".cif", ".mmcif"}
            else "pdb"
        )
        viewer.addModel(receptor_path.read_text(errors="replace"), receptor_format)
        viewer.setStyle(
            {"model": 0, "hetflag": False},
            {"cartoon": {"color": "#cbd5e1", "opacity": 0.9}},
        )
        viewer.addModel(_first_sdf_record(pose_path), "sdf")
        viewer.setStyle(
            {"model": 1},
            {"stick": {"colorscheme": "cyanCarbon", "radius": 0.22}},
        )
        viewer.zoomTo({"model": 1})
        viewer.zoom(0.72)
        render_persistent_3dmol(
            viewer,
            key=f"pose-validation-complex:{job.run_id}:{pose_id}",
            height=670,
        )
    except Exception as exc:
        st.error(f"Validated-complex preview failed: {exc}")
    return True


def _sdf_record_by_name(path: Path, name: str) -> str:
    for record in path.read_text(errors="replace").split("$$$$"):
        lines = record.strip().splitlines()
        if lines and lines[0].strip() == name:
            return record.strip() + "\n$$$$\n"
    return ""


def _render_rescoring_viewer(job: JobRecord) -> bool:
    table = _rescoring_table(job)
    if table is None or table.empty or "pose_id" not in table.columns:
        return False
    selection_id = str(job.metadata.get("selection_run_id") or job.parent_run_id)
    selection_dir = resolve_run_dir("rescoring", selection_id)
    if selection_dir is None:
        return False
    receptor = selection_dir / "receptor.pdb"
    poses = selection_dir / "selected_poses.sdf"
    if not receptor.is_file() or not poses.is_file():
        return False
    pose_ids = [str(value) for value in table["pose_id"].tolist()]
    pose_id = st.selectbox(
        "Rescored pose",
        pose_ids,
        key=f"rescoring_viewer_pose_{job.run_id}",
    )
    pose_data = _sdf_record_by_name(poses, pose_id)
    if not pose_data:
        return False
    selected = table.loc[table["pose_id"].astype(str) == pose_id].iloc[0]
    score_columns = [
        column
        for column in (
            "source_score_kcal_mol",
            "gnina_empirical_score_kcal_mol",
            "gnina_cnn_score",
            "gnina_cnn_affinity",
            "boltzina_affinity_log10_ic50_uM",
            "boltzina_binder_probability",
        )
        if column in table.columns
    ]
    if score_columns:
        columns = st.columns(min(4, len(score_columns)))
        labels = {
            "source_score_kcal_mol": "Source docking score",
            "gnina_empirical_score_kcal_mol": "GNINA empirical",
            "gnina_cnn_score": "GNINA CNN score",
            "gnina_cnn_affinity": "GNINA CNN affinity",
            "boltzina_affinity_log10_ic50_uM": "Boltzina log10(IC50 µM)",
            "boltzina_binder_probability": "Boltzina binder probability",
        }
        for index, column in enumerate(score_columns):
            value = selected[column]
            columns[index % len(columns)].metric(
                labels[column],
                f"{float(value):.4f}" if value not in ("", None) else "—",
            )
    st.caption(
        "Coordinate-preserving rescoring view: the prepared receptor is grey and "
        "the exact selected docking pose is cyan. Rescoring changes annotations, "
        "not the displayed input coordinates."
    )
    try:
        import py3Dmol

        viewer = py3Dmol.view(width=1100, height=650)
        viewer.addModel(receptor.read_text(errors="replace"), "pdb")
        viewer.setStyle(
            {"model": 0, "hetflag": False},
            {"cartoon": {"color": "#cbd5e1", "opacity": 0.9}},
        )
        viewer.addModel(pose_data, "sdf")
        viewer.setStyle(
            {"model": 1},
            {"stick": {"colorscheme": "cyanCarbon", "radius": 0.22}},
        )
        viewer.zoomTo({"model": 1})
        viewer.zoom(0.72)
        render_persistent_3dmol(
            viewer,
            key=f"rescoring-pose:{job.run_id}",
            height=670,
        )
    except Exception as exc:
        st.error(f"Rescored-pose preview failed: {exc}")
    return True


def _lineage_rows(job: JobRecord, all_jobs: list[JobRecord]) -> list[dict[str, str]]:
    rows = [{"relationship": "current", "task": job.task_group, "run_id": job.run_id}]
    references: set[str] = set()
    if job.parent_run_id:
        references.add(job.parent_run_id)
    for key, value in job.metadata.items():
        if (
            key.endswith("_run_id")
            and key not in {"workflow_parent_run_id"}
            and isinstance(value, str)
            and value.strip()
            and value != job.run_id
        ):
            references.add(value.strip())
    by_id = {item.run_id: item for item in all_jobs}
    workflow_parent_id = job.workflow_parent_run_id
    if workflow_parent_id and workflow_parent_id != job.run_id:
        parent = by_id.get(workflow_parent_id)
        rows.append(
            {
                "relationship": "workflow parent",
                "task": parent.task_group if parent else "workflows",
                "run_id": workflow_parent_id,
            }
        )
    for run_id in sorted(references):
        parent = by_id.get(run_id)
        rows.append(
            {
                "relationship": "input/parent",
                "task": parent.task_group if parent else "unresolved",
                "run_id": run_id,
            }
        )
    known_children: set[str] = set()
    for child in all_jobs:
        if str(child.metadata.get("retry_of_run_id") or "") == job.run_id:
            rows.append({"relationship": "retry", "task": child.task_group, "run_id": child.run_id})
            known_children.add(child.run_id)
        if child.parent_run_id == job.run_id:
            rows.append({"relationship": "child", "task": child.task_group, "run_id": child.run_id})
            known_children.add(child.run_id)
        if job.task_group == "workflows" and child.workflow_parent_run_id == job.run_id and child.run_id not in known_children:
            rows.append({"relationship": "workflow child", "task": child.task_group, "run_id": child.run_id})
            known_children.add(child.run_id)
    return rows


def _render_mmgbsa_metrics(job: JobRecord) -> bool:
    if job.task_group != "md-mmgbsa":
        return False
    mmgbsa = (
        job.result.get("mmgbsa")
        if isinstance(job.result.get("mmgbsa"), dict)
        else {}
    )
    status = str(mmgbsa.get("status") or job.status or "unknown")
    st.markdown("#### Endpoint binding-energy analysis")
    if status != "success":
        error = str(
            mmgbsa.get("error")
            or job.result.get("error")
            or job.metadata.get("error")
            or "No endpoint-energy result is available."
        )
        st.error(error)
        return True

    def _energy_text(delta: dict, key: str, unit: str) -> str:
        value = delta.get(key)
        return "—" if value is None else f"{float(value):.3f} {unit}"

    def _render_energy_block(title: str, delta: dict) -> None:
        st.markdown(f"**{title}**")
        columns = st.columns(4)
        specifications = (
            ("ΔG bind", "delta_g_bind_total"),
            ("ΔMM", "delta_mm"),
            ("ΔPolar solvation", "delta_gbsa"),
            ("ΔNonpolar", "delta_nonpolar"),
        )
        for column, (label, key) in zip(columns, specifications):
            column.metric(
                label,
                _energy_text(delta, f"{key}_kcal_mol", "kcal/mol"),
            )
            column.caption(
                _energy_text(delta, f"{key}_kj_mol", "kJ/mol")
            )

    gb_delta = (
        (mmgbsa.get("gb") or {}).get("delta")
        if isinstance(mmgbsa.get("gb"), dict)
        else {}
    ) or {}
    pb_delta = (
        (mmgbsa.get("pb") or {}).get("delta")
        if isinstance(mmgbsa.get("pb"), dict)
        else {}
    ) or {}
    if gb_delta:
        _render_energy_block("MM/GBSA (GB)", gb_delta)
    if pb_delta:
        _render_energy_block("MM/PBSA (PB)", pb_delta)
    if not gb_delta and not pb_delta:
        _render_energy_block("MM/GBSA", mmgbsa.get("delta") or {})

    metadata = (
        mmgbsa.get("metadata")
        if isinstance(mmgbsa.get("metadata"), dict)
        else {}
    )
    parameters = (
        job.metadata.get("parameters")
        if isinstance(job.metadata.get("parameters"), dict)
        else {}
    )
    st.markdown("**Calculation details**")
    st.dataframe(
        [
            {"field": "Method", "value": mmgbsa.get("method") or mmgbsa.get("backend") or "—"},
            {"field": "Force field", "value": mmgbsa.get("forcefield_method") or "—"},
            {
                "field": "Trajectory window",
                "value": (
                    f"{mmgbsa.get('start_pct', parameters.get('start_pct', '—'))}"
                    f"–{mmgbsa.get('end_pct', parameters.get('end_pct', '—'))}%"
                ),
            },
            {"field": "Stride", "value": mmgbsa.get("stride", parameters.get("stride", "—"))},
            {
                "field": "Frames analyzed",
                "value": metadata.get(
                    "n_frames_analyzed",
                    metadata.get("mmpbsa_frame_count", "—"),
                ),
            },
            {
                "field": "MPI processes",
                "value": metadata.get(
                    "mmpbsa_mpi_effective_cores",
                    parameters.get("cpu_process_limit", "—"),
                ),
            },
            {"field": "Trajectory", "value": mmgbsa.get("trajectory_path") or "—"},
            {"field": "Topology", "value": mmgbsa.get("topology_path") or "—"},
        ],
        hide_index=True,
        width="stretch",
    )

    artifact_rows: list[dict[str, object]] = []
    artifacts = (
        mmgbsa.get("artifacts")
        if isinstance(mmgbsa.get("artifacts"), dict)
        else {}
    )
    for name, raw_path in artifacts.items():
        path = Path(str(raw_path))
        resolved = path if path.is_absolute() else job.run_dir / path
        artifact_rows.append(
            {
                "file": name,
                "path": str(raw_path),
                "exists": resolved.is_file(),
                "size_kb": (
                    round(resolved.stat().st_size / 1024.0, 2)
                    if resolved.is_file()
                    else None
                ),
            }
        )
    for name in ("mmgbsa_summary.json", "native_result.json"):
        path = job.run_dir / name
        if path.is_file() and not any(
            row["path"] == name for row in artifact_rows
        ):
            artifact_rows.append(
                {
                    "file": name.removesuffix(".json"),
                    "path": name,
                    "exists": True,
                    "size_kb": round(path.stat().st_size / 1024.0, 2),
                }
            )
    st.markdown("**Energy files**")
    st.caption(f"Run directory: {job.run_dir}")
    if artifact_rows:
        st.dataframe(artifact_rows, hide_index=True, width="stretch")
    else:
        st.info("No energy artifacts were declared by this result.")
    st.caption(
        "Endpoint GB/PB energies are approximate comparative estimates; "
        "interpret replica consistency and trajectory stability alongside them."
    )
    return True


def _render_md_production_metrics(job: JobRecord) -> bool:
    if job.task_group != "bound-ligand-md":
        return False
    md_result = (
        job.result.get("md_result")
        if isinstance(job.result.get("md_result"), dict)
        else {}
    )
    analytics = (
        md_result.get("analytics")
        if isinstance(md_result.get("analytics"), dict)
        else {}
    )
    performance = (
        analytics.get("performance")
        if isinstance(analytics.get("performance"), dict)
        else {}
    )
    throughput = performance.get("ns_per_day")
    if throughput is None:
        return False
    try:
        input_payload = json.loads((job.run_dir / "input.json").read_text())
    except (OSError, ValueError, TypeError):
        input_payload = {}
    production_ns = (
        float(input_payload.get("production_steps") or 0)
        * float(input_payload.get("production_timestep_fs") or 0)
        / 1_000_000.0
    )
    estimated_hours = (
        production_ns / float(throughput) * 24.0
        if production_ns > 0.0 and float(throughput) > 0.0
        else None
    )
    st.markdown("#### Production performance")
    columns = st.columns(3)
    columns[0].metric(
        "MD engine",
        str(job.result.get("engine") or md_result.get("engine") or "OpenMM"),
    )
    columns[1].metric("Engine throughput", f"{float(throughput):,.1f} ns/day")
    columns[2].metric(
        "Reported production time",
        f"{estimated_hours:.2f} h" if estimated_hours is not None else "—",
    )
    st.caption(
        f"Source: {performance.get('source') or 'engine log'}"
        + (
            f" · configured trajectory length: {production_ns:.3f} ns"
            if production_ns > 0.0
            else ""
        )
    )
    return True


_MD_PLOT_LABELS = {
    "backbone_rmsd": "Protein backbone RMSD",
    "ligand_rmsd": "Ligand RMSD",
    "radius_of_gyration": "Radius of gyration",
    "secondary_structure": "Secondary structure",
    "ligand_displacement": "Ligand reference-site displacement",
    "minimum_distance": "Minimum protein–ligand distance",
    "site_retention": "Binding-site retention (geometric)",
    "required_interaction_retention": "Mandatory hypothesis interactions",
    "protein_rmsf": "Protein Cα RMSF",
    "ligand_rmsf": "Ligand heavy-atom RMSF",
    "contact_occupancy": "Binding hotspots",
    "contact_scope_fraction": "Contact BB/SC fractions",
    "hbond_occupancy": "Protein–ligand hydrogen bonds",
    "hbond_scope_fraction": "Hydrogen-bond BB/SC fractions",
    "hydrophobic_occupancy": "Protein–ligand hydrophobic contacts",
    "hydrophobic_scope_fraction": "Hydrophobic-contact BB/SC fractions",
    "water_bridge_occupancy": "Protein–ligand water bridges",
    "water_bridge_scope_fraction": "Water-bridge BB/SC fractions",
    "salt_bridge_occupancy": "Protein–ligand salt bridges",
    "salt_bridge_scope_fraction": "Salt-bridge BB/SC fractions",
    "contact_matrix": "Contact persistence matrix",
    "interaction_composition": "Interaction-type composition",
    "interaction_network": "Ligand interaction network",
    "persistence_distance": "Persistence versus distance",
    "interface_rin": "Protein interface RIN",
    "endpoint_energy": "MM/GBSA endpoint-energy components",
    "endpoint_energy_pb": "MM/PBSA endpoint-energy components",
    "endpoint_binding_summary": "Final MM/GBSA versus MM/PBSA",
    "throughput": "Replica throughput",
}
def _md_residue_node_color(label: str) -> str:
    residue = str(label)[:3].upper()
    if residue in {"ASP", "GLU"}:
        return "#f4b6a6"
    if residue in {"ARG", "LYS", "HIS"}:
        return "#b9d8f0"
    if residue in {"ASN", "GLN", "SER", "THR", "TYR", "CYS"}:
        return "#e7e7e7"
    if residue in {
        "ALA",
        "VAL",
        "ILE",
        "LEU",
        "MET",
        "PHE",
        "TRP",
        "PRO",
    }:
        return "#c9f7c5"
    return "#e7e7e7"


def _md_display_residue_labels(labels: list[str]) -> list[str]:
    parsed = [
        re.match(r"^(.*?)\s*·\s*chain\s+(.+?)\s*$", str(label))
        for label in labels
    ]
    chains = {
        match.group(2)
        for match in parsed
        if match is not None
    }
    if len(chains) == 1 and all(match is not None for match in parsed):
        return [match.group(1).strip() for match in parsed if match is not None]
    return labels


def _available_md_plot_ids(report: dict[str, Any]) -> list[str]:
    series = [
        row
        for row in report.get("replica_series") or []
        if isinstance(row, dict)
    ]
    replicas = [
        row for row in report.get("replicas") or [] if isinstance(row, dict)
    ]
    field_requirements = {
        "backbone_rmsd": ("backbone_rmsd_angstrom",),
        "ligand_rmsd": ("ligand_rmsd_angstrom",),
        "radius_of_gyration": (
            "protein_rg_angstrom",
            "ligand_rg_angstrom",
            "complex_rg_angstrom",
        ),
        "secondary_structure": (
            "helix_fraction",
            "sheet_fraction",
            "coil_fraction",
        ),
        "ligand_displacement": (
            "ligand_centroid_displacement_angstrom",
        ),
        "minimum_distance": ("minimum_protein_distance_angstrom",),
    }
    available = [
        plot_id
        for plot_id, fields in field_requirements.items()
        if any(
            any(row.get(field) for field in fields)
            for row in series
        )
    ]
    if any(
        row.get("reference_site_retained_fraction") is not None
        for row in replicas
    ):
        available.append("site_retention")
    if report.get("required_interaction_consensus"):
        available.append("required_interaction_retention")
    if report.get("rmsf_consensus"):
        available.append("protein_rmsf")
    if report.get("ligand_rmsf_consensus"):
        available.append("ligand_rmsf")
    if report.get("contact_consensus"):
        available.extend(
            (
                "contact_occupancy",
                "hbond_occupancy",
                "interaction_network",
            )
        )
        contact_rows = [
            row
            for row in report.get("contact_consensus") or []
            if isinstance(row, dict)
        ]
        for scope_plot_id, field_stem in (
            ("contact_scope_fraction", "contact"),
            ("hbond_scope_fraction", "hydrogen_bond"),
            ("hydrophobic_scope_fraction", "hydrophobic"),
            ("water_bridge_scope_fraction", "water_bridge"),
            ("salt_bridge_scope_fraction", "salt_bridge"),
        ):
            if (
                scope_plot_id == "salt_bridge_scope_fraction"
                and not report.get("salt_bridge_applicable")
            ):
                continue
            if any(
                row.get(
                    f"mean_{field_stem}_backbone_occupancy"
                ) is not None
                and row.get(
                    f"mean_{field_stem}_sidechain_occupancy"
                ) is not None
                for row in contact_rows
            ):
                available.append(scope_plot_id)
        if any(
            row.get("mean_hydrophobic_occupancy") is not None
            for row in contact_rows
        ):
            available.append("hydrophobic_occupancy")
        if any(
            row.get("mean_water_bridge_occupancy") is not None
            for row in contact_rows
        ):
            available.append("water_bridge_occupancy")
        if report.get("salt_bridge_applicable"):
            available.append("salt_bridge_occupancy")
        if any(
            row.get(field) is not None
            for row in contact_rows
            for field in (
                "mean_hydrophobic_occupancy",
                "mean_water_bridge_occupancy",
                "mean_salt_bridge_occupancy",
            )
        ):
            available.append("interaction_composition")
        if any(
            row.get("mean_minimum_distance_angstrom") is not None
            for row in contact_rows
        ):
            available.append("persistence_distance")
    contact_matrix = report.get("contact_matrix")
    if (
        isinstance(contact_matrix, dict)
        and contact_matrix.get("residues")
        and contact_matrix.get("contact_occupancy")
    ):
        available.append("contact_matrix")
    if any(
        row.get("delta_g_bind_kcal_mol") is not None for row in replicas
    ):
        available.append("endpoint_energy")
    if any(
        row.get("pb_delta_g_bind_kcal_mol") is not None
        for row in replicas
    ):
        available.append("endpoint_energy_pb")
    if all(
        any(row.get(field) is not None for row in replicas)
        for field in (
            "delta_g_bind_kcal_mol",
            "pb_delta_g_bind_kcal_mol",
        )
    ):
        available.append("endpoint_binding_summary")
    if any(row.get("performance_ns_per_day") is not None for row in replicas):
        available.append("throughput")
    if report.get("interface_rin_replicas"):
        available.append("interface_rin")
    return [
        plot_id for plot_id in _MD_PLOT_LABELS if plot_id in available
    ]


def _plot_md_replica_series(
    axis,
    series: list[dict[str, Any]],
    field: str,
    y_label: str,
) -> bool:
    plotted = False
    x_label = "Production time (ns)"
    for row in series:
        values = [float(value) for value in row.get(field) or []]
        times = [float(value) for value in row.get("time_ns") or []]
        if not values:
            continue
        if len(times) != len(values):
            times = list(np.arange(len(values), dtype=float))
            x_label = "Analyzed frame"
        axis.plot(
            times,
            values,
            label=f"Replica {row.get('replica', '-')}",
            linewidth=0.9,
        )
        plotted = True
    axis.set_xlabel(x_label, fontsize=7)
    axis.set_ylabel(y_label, fontsize=7)
    if plotted:
        axis.legend(loc="best", fontsize=6, frameon=False)
    return plotted


def _plot_md_mean_series(
    axis,
    series: list[dict[str, Any]],
    fields: tuple[tuple[str, str], ...],
    y_label: str,
) -> bool:
    plotted = False
    for field, label in fields:
        rows = [
            np.asarray(row.get(field) or [], dtype=float)
            for row in series
            if row.get(field)
        ]
        if not rows:
            continue
        length = min(len(values) for values in rows)
        values = np.vstack([row[:length] for row in rows])
        times = next(
            (
                np.asarray(row.get("time_ns") or [], dtype=float)[:length]
                for row in series
                if len(row.get(field) or []) >= length
                and len(row.get("time_ns") or []) >= length
            ),
            np.arange(length, dtype=float),
        )
        mean_values = np.mean(values, axis=0)
        axis.plot(times, mean_values, label=label, linewidth=1.0)
        if values.shape[0] > 1:
            sample_sd = np.std(values, axis=0, ddof=1)
            axis.fill_between(
                times,
                mean_values - sample_sd,
                mean_values + sample_sd,
                alpha=0.12,
            )
        plotted = True
    axis.set_xlabel("Production time (ns)", fontsize=7)
    axis.set_ylabel(y_label, fontsize=7)
    if plotted:
        axis.legend(loc="best", fontsize=6, frameon=False)
    return plotted


def _md_hotspot_rows(
    report: dict[str, Any],
    *,
    limit: int = 12,
) -> tuple[list[dict[str, Any]], str, str, bool]:
    rows = [
        row
        for row in report.get("contact_consensus") or []
        if isinstance(row, dict)
    ]
    weighted_score_available = any(
        row.get("mean_binding_importance_score") is not None for row in rows
    )
    field = (
        "mean_binding_importance_score"
        if weighted_score_available
        else "mean_contact_occupancy"
    )
    error_field = (
        "sample_sd_binding_importance_score"
        if weighted_score_available
        else "sample_sd_contact_occupancy"
    )
    rows = sorted(
        rows,
        key=lambda row: float(row.get(field) or 0.0),
        reverse=True,
    )[:limit]
    return rows, field, error_field, weighted_score_available


def _plot_md_consensus_bars(
    axis,
    labels: list[str],
    replica_values: list[list[float]],
    *,
    colors: list[str] | None = None,
) -> bool:
    populated = [values for values in replica_values if values]
    if not populated:
        return False
    means = [
        float(np.mean(values)) if values else 0.0
        for values in replica_values
    ]
    sample_sd = [
        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for values in replica_values
    ]
    positions = np.arange(len(labels), dtype=float)
    axis.bar(
        positions,
        means,
        yerr=sample_sd,
        capsize=3,
        width=0.32,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        label="Mean ± sample SD",
    )
    for position, values in zip(positions, replica_values, strict=True):
        offsets = (
            np.linspace(-0.09, 0.09, len(values))
            if len(values) > 1
            else np.asarray([0.0])
        )
        axis.scatter(
            position + offsets,
            values,
            s=24,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
    axis.scatter(
        [],
        [],
        s=24,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        label="Replica value",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    if len(labels) == 1:
        axis.set_xlim(-0.65, 0.65)
    axis.legend(fontsize=5, frameon=False)
    return True


def _md_contact_replica_values(
    report: dict[str, Any],
    residue: str,
    field: str,
) -> list[float]:
    matrix = (
        report.get("contact_matrix")
        if isinstance(report.get("contact_matrix"), dict)
        else {}
    )
    residues = [str(value) for value in matrix.get("residues") or []]
    if residue not in residues:
        return []
    values = matrix.get(field) or []
    index = residues.index(residue)
    if index >= len(values):
        return []
    return [float(value) for value in values[index]]


def _md_replica_interaction_rows(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    replica: int | None,
) -> list[dict[str, Any]]:
    if replica is None:
        return rows
    matrix = (
        report.get("contact_matrix")
        if isinstance(report.get("contact_matrix"), dict)
        else {}
    )
    matrix_replicas = [
        int(value) for value in matrix.get("replicas") or []
    ]
    if replica not in matrix_replicas:
        return []
    replica_index = matrix_replicas.index(replica)
    field_map = {
        "mean_contact_occupancy": "contact_occupancy",
        "mean_contact_backbone_occupancy": "contact_backbone_occupancy",
        "mean_contact_sidechain_occupancy": "contact_sidechain_occupancy",
        "mean_hydrogen_bond_occupancy": "hydrogen_bond_occupancy",
        "mean_hydrogen_bond_backbone_occupancy": (
            "hydrogen_bond_backbone_occupancy"
        ),
        "mean_hydrogen_bond_sidechain_occupancy": (
            "hydrogen_bond_sidechain_occupancy"
        ),
        "mean_hydrophobic_occupancy": "hydrophobic_occupancy",
        "mean_hydrophobic_backbone_occupancy": (
            "hydrophobic_backbone_occupancy"
        ),
        "mean_hydrophobic_sidechain_occupancy": (
            "hydrophobic_sidechain_occupancy"
        ),
        "mean_water_bridge_occupancy": "water_bridge_occupancy",
        "mean_water_bridge_backbone_occupancy": (
            "water_bridge_backbone_occupancy"
        ),
        "mean_water_bridge_sidechain_occupancy": (
            "water_bridge_sidechain_occupancy"
        ),
        "mean_salt_bridge_occupancy": "salt_bridge_occupancy",
        "mean_salt_bridge_backbone_occupancy": (
            "salt_bridge_backbone_occupancy"
        ),
        "mean_salt_bridge_sidechain_occupancy": (
            "salt_bridge_sidechain_occupancy"
        ),
        "mean_binding_importance_score": "binding_importance_score",
        "mean_binding_importance_backbone_score": (
            "binding_importance_backbone_score"
        ),
        "mean_binding_importance_sidechain_score": (
            "binding_importance_sidechain_score"
        ),
        "mean_minimum_distance_angstrom": "minimum_distance_angstrom",
    }
    replica_rows: list[dict[str, Any]] = []
    for row in rows:
        replica_row = dict(row)
        residue = str(row.get("residue") or "")
        for target_field, matrix_field in field_map.items():
            values = _md_contact_replica_values(
                report,
                residue,
                matrix_field,
            )
            replica_row[target_field] = (
                values[replica_index]
                if replica_index < len(values)
                else 0.0
            )
        replica_rows.append(replica_row)
    return replica_rows


_MD_REPLICA_SPLIT_PLOTS = frozenset(
    {
        "backbone_rmsd",
        "ligand_rmsd",
        "ligand_displacement",
        "minimum_distance",
        "radius_of_gyration",
        "secondary_structure",
        "contact_occupancy",
        "hbond_occupancy",
        "hydrophobic_occupancy",
        "water_bridge_occupancy",
        "salt_bridge_occupancy",
        "contact_scope_fraction",
        "hbond_scope_fraction",
        "hydrophobic_scope_fraction",
        "water_bridge_scope_fraction",
        "salt_bridge_scope_fraction",
        "interaction_composition",
        "contact_matrix",
        "protein_rmsf",
        "ligand_rmsf",
        "persistence_distance",
        "interaction_network",
    }
)
_MD_REPLICA_BAR_PLOTS = frozenset(
    {
        "site_retention",
        "endpoint_energy",
        "endpoint_energy_pb",
        "endpoint_binding_summary",
        "throughput",
    }
)


def _md_ordered_plot_ids(
    selected: list[str],
    replica_display: str,
) -> list[str]:
    if replica_display != "Separate":
        return selected
    return [
        plot_id
        for plot_id in selected
        if plot_id in _MD_REPLICA_SPLIT_PLOTS
    ] + [
        plot_id
        for plot_id in selected
        if plot_id not in _MD_REPLICA_SPLIT_PLOTS
    ]


def _md_separate_plot_specs(
    selected: list[str],
    replica_ids: list[int],
    arrangement: str,
) -> list[tuple[str, int, bool]]:
    split_plot_ids = [
        plot_id for plot_id in selected if plot_id in _MD_REPLICA_SPLIT_PLOTS
    ]
    if arrangement == "Serial":
        return [
            (plot_id, replica_id, False)
            for replica_id in replica_ids
            for plot_id in split_plot_ids
        ]
    return [
        (plot_id, replica_id, False)
        for plot_id in split_plot_ids
        for replica_id in replica_ids
    ]


def _md_plot_figure(
    plot_id: str,
    report: dict[str, Any],
    *,
    replica: int | None = None,
    replica_bars: bool = False,
    selected_replicas: list[int] | None = None,
    interaction_limit: int = 12,
    include_water_bridges: bool = True,
    include_proximity_interactions: bool = True,
    show_interaction_scopes: bool = False,
) -> Any | None:
    series = [
        row
        for row in report.get("replica_series") or []
        if isinstance(row, dict)
        and (
            replica is None
            or int(row.get("replica") or 0) == replica
        )
        and (
            selected_replicas is None
            or int(row.get("replica") or 0) in selected_replicas
        )
    ]
    replicas = [
        row
        for row in report.get("replicas") or []
        if isinstance(row, dict)
        and (
            replica is None
            or int(row.get("replica") or 0) == replica
        )
        and (
            selected_replicas is None
            or int(row.get("replica") or 0) in selected_replicas
        )
    ]
    fig, axis = plt.subplots(figsize=(4.2, 3.0), dpi=130)
    plotted = False
    line_specs = {
        "backbone_rmsd": ("backbone_rmsd_angstrom", "RMSD (Å)"),
        "ligand_rmsd": ("ligand_rmsd_angstrom", "RMSD (Å)"),
        "ligand_displacement": (
            "ligand_centroid_displacement_angstrom",
            "Displacement (Å)",
        ),
        "minimum_distance": (
            "minimum_protein_distance_angstrom",
            "Distance (Å)",
        ),
    }
    if plot_id in line_specs:
        field, y_label = line_specs[plot_id]
        plotted = _plot_md_replica_series(
            axis, series, field, y_label
        )
        if replica is not None and axis.get_legend() is not None:
            axis.get_legend().remove()
    elif plot_id == "radius_of_gyration":
        plotted = _plot_md_mean_series(
            axis,
            series,
            (
                ("protein_rg_angstrom", "Protein"),
                ("ligand_rg_angstrom", "Ligand"),
                ("complex_rg_angstrom", "Complex"),
            ),
            "Rg (Å)",
        )
    elif plot_id == "secondary_structure":
        plotted = _plot_md_mean_series(
            axis,
            series,
            (
                ("helix_fraction", "Helix"),
                ("sheet_fraction", "Sheet"),
                ("coil_fraction", "Coil"),
            ),
            "Residue fraction",
        )
        axis.set_ylim(0, 1)
    elif plot_id == "site_retention":
        values = [
            float(row["reference_site_retained_fraction"])
            for row in replicas
            if row.get("reference_site_retained_fraction") is not None
        ]
        if replica_bars:
            axis.bar(
                [f"R{row.get('replica', '-')}" for row in replicas],
                values,
                width=0.55,
                color="#4c78a8",
            )
            plotted = bool(values)
        else:
            plotted = _plot_md_consensus_bars(
                axis,
                ["Retention"],
                [values],
                colors=["#4c78a8"],
            )
        axis.set_ylim(0, 1.08)
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.set_ylabel("Retained fraction", fontsize=7)
    elif plot_id == "required_interaction_retention":
        rows = [
            row
            for row in report.get("required_interaction_consensus") or []
            if isinstance(row, dict)
        ]
        positions = np.arange(len(rows), dtype=float)
        means = [float(row.get("mean_occupancy") or 0.0) for row in rows]
        errors = [
            float(row.get("sample_sd_occupancy") or 0.0) for row in rows
        ]
        axis.barh(
            positions,
            means,
            xerr=errors,
            capsize=2,
            color="#8b1a1a",
            edgecolor="white",
            linewidth=0.35,
        )
        for position, row in zip(positions, rows, strict=True):
            values = [
                float(value) for value in row.get("replica_occupancy") or []
            ]
            offsets = (
                np.linspace(-0.12, 0.12, len(values))
                if len(values) > 1
                else np.asarray([0.0])
            )
            if values:
                axis.scatter(
                    values,
                    position + offsets,
                    s=17,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=4,
                )
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [str(row.get("label") or row.get("residue") or "") for row in rows],
            fontsize=6,
        )
        axis.set_xlim(0.0, 1.05)
        axis.set_xlabel("MD occupancy", fontsize=7)
        plotted = bool(rows)
    elif plot_id in {
        "contact_occupancy",
        "hbond_occupancy",
        "hydrophobic_occupancy",
        "water_bridge_occupancy",
        "salt_bridge_occupancy",
    }:
        rows, field, error_field, weighted_score_available = (
            _md_hotspot_rows(report, limit=interaction_limit)
        )
        rows = _md_replica_interaction_rows(report, rows, replica)
        if plot_id == "hbond_occupancy":
            field = "mean_hydrogen_bond_occupancy"
            error_field = "sample_sd_hydrogen_bond_occupancy"
        elif plot_id == "hydrophobic_occupancy":
            field = "mean_hydrophobic_occupancy"
            error_field = "sample_sd_hydrophobic_occupancy"
        elif plot_id == "water_bridge_occupancy":
            field = "mean_water_bridge_occupancy"
            error_field = "sample_sd_water_bridge_occupancy"
        elif plot_id == "salt_bridge_occupancy":
            field = "mean_salt_bridge_occupancy"
            error_field = "sample_sd_salt_bridge_occupancy"
        positions = np.arange(len(rows))
        axis.barh(
            positions,
            [float(row.get(field) or 0.0) for row in rows],
            xerr=(
                None
                if replica is not None
                else [float(row.get(error_field) or 0.0) for row in rows]
            ),
            capsize=2,
            color={
                "contact_occupancy": "#6f52c8",
                "hbond_occupancy": "#800000",
                "hydrophobic_occupancy": "#90ee90",
                "water_bridge_occupancy": "#1e90ff",
                "salt_bridge_occupancy": "#cc33aa",
            }[plot_id],
            edgecolor="white",
            linewidth=0.35,
        )
        matrix_field = {
            "contact_occupancy": (
                "binding_importance_score"
                if weighted_score_available
                else "contact_occupancy"
            ),
            "hbond_occupancy": "hydrogen_bond_occupancy",
            "hydrophobic_occupancy": "hydrophobic_occupancy",
            "water_bridge_occupancy": "water_bridge_occupancy",
            "salt_bridge_occupancy": "salt_bridge_occupancy",
        }[plot_id]
        replica_point_values: list[float] = []
        if replica is None:
            for position, row in zip(positions, rows, strict=True):
                replica_values = _md_contact_replica_values(
                    report,
                    str(row.get("residue") or ""),
                    matrix_field,
                )
                if not replica_values:
                    continue
                replica_point_values.extend(replica_values)
                offsets = (
                    np.linspace(-0.12, 0.12, len(replica_values))
                    if len(replica_values) > 1
                    else np.asarray([0.0])
                )
                axis.scatter(
                    replica_values,
                    position + offsets,
                    s=17,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=4,
                )
            axis.scatter(
                [],
                [],
                s=17,
                facecolors="none",
                edgecolors="black",
                linewidths=0.7,
                label="Replica value",
            )
            axis.legend(fontsize=5, frameon=False, loc="lower right")
        axis.set_yticks(positions)
        residue_labels = [
            str(row.get("residue") or "") for row in rows
        ]
        axis.set_yticklabels(
            _md_display_residue_labels(residue_labels),
            fontsize=5.5,
        )
        if rows:
            axis.set_ylim(len(rows) - 0.5, -0.5)
        values = [float(row.get(field) or 0.0) for row in rows]
        axis.set_xlim(
            0,
            max(
                1.0,
                max(values, default=0.0) * 1.08,
                max(replica_point_values, default=0.0) * 1.08,
            ),
        )
        axis.set_xlabel(
            (
                (
                    (
                        "Interaction-hotspot score "
                        "(3H + 2.5SB + 1.5WB + Hyd)"
                        if replica is not None
                        else (
                            "Interaction-hotspot score "
                            "(3H + 2.5SB + 1.5WB + Hyd) ± SD"
                        )
                    )
                    if weighted_score_available
                    else (
                        "Contact occupancy"
                        if replica is not None
                        else "Mean contact occupancy ± sample SD"
                    )
                )
                if plot_id == "contact_occupancy"
                else (
                    (
                        "H-bond occupancy"
                        if replica is not None
                        else "Mean H-bond occupancy ± sample SD"
                    )
                    if plot_id == "hbond_occupancy"
                    else (
                        (
                            "Hydrophobic-contact occupancy"
                            if replica is not None
                            else (
                                "Mean hydrophobic-contact occupancy ± sample SD"
                            )
                        )
                        if plot_id == "hydrophobic_occupancy"
                        else (
                            (
                                "Water-bridge occupancy"
                                if replica is not None
                                else (
                                    "Mean water-bridge occupancy ± sample SD"
                                )
                            )
                            if plot_id == "water_bridge_occupancy"
                            else (
                                "Salt-bridge occupancy"
                                if replica is not None
                                else (
                                    "Mean salt-bridge occupancy ± sample SD"
                                )
                            )
                        )
                    )
                )
            ),
            fontsize=7,
        )
        plotted = bool(rows)
    elif plot_id in {
        "contact_scope_fraction",
        "hbond_scope_fraction",
        "hydrophobic_scope_fraction",
        "water_bridge_scope_fraction",
        "salt_bridge_scope_fraction",
    }:
        scope_stem, colors = {
            "contact_scope_fraction": ("contact", ("#4b32a8", "#9d8be0")),
            "hbond_scope_fraction": (
                "hydrogen_bond",
                ("#800000", "#d95f5f"),
            ),
            "hydrophobic_scope_fraction": (
                "hydrophobic",
                ("#3a9d50", "#90ee90"),
            ),
            "water_bridge_scope_fraction": (
                "water_bridge",
                ("#0878c9", "#78c5ff"),
            ),
            "salt_bridge_scope_fraction": (
                "salt_bridge",
                ("#8f207e", "#df7ed2"),
            ),
        }[plot_id]
        rows, _, _, _ = _md_hotspot_rows(
            report,
            limit=interaction_limit,
        )
        rows = _md_replica_interaction_rows(report, rows, replica)
        backbone_field = f"mean_{scope_stem}_backbone_occupancy"
        sidechain_field = f"mean_{scope_stem}_sidechain_occupancy"
        scoped_rows = []
        for row in rows:
            backbone = float(row.get(backbone_field) or 0.0)
            sidechain = float(row.get(sidechain_field) or 0.0)
            total = backbone + sidechain
            scoped_rows.append(
                (
                    row,
                    backbone / total if total > 0.0 else 0.0,
                    sidechain / total if total > 0.0 else 0.0,
                )
            )
        positions = np.arange(len(scoped_rows))
        backbone_fractions = [values[1] for values in scoped_rows]
        sidechain_fractions = [values[2] for values in scoped_rows]
        axis.barh(
            positions,
            backbone_fractions,
            color=colors[0],
            edgecolor="white",
            linewidth=0.35,
            label="BB",
        )
        axis.barh(
            positions,
            sidechain_fractions,
            left=backbone_fractions,
            color=colors[1],
            edgecolor="white",
            linewidth=0.35,
            label="SC",
        )
        for position, backbone, sidechain in zip(
            positions,
            backbone_fractions,
            sidechain_fractions,
            strict=True,
        ):
            if backbone >= 0.08:
                axis.text(
                    backbone / 2,
                    position,
                    f"{backbone:.0%}",
                    color="white",
                    fontsize=5,
                    ha="center",
                    va="center",
                )
            if sidechain >= 0.08:
                axis.text(
                    backbone + sidechain / 2,
                    position,
                    f"{sidechain:.0%}",
                    color="#222222",
                    fontsize=5,
                    ha="center",
                    va="center",
                )
        axis.set_yticks(positions)
        axis.set_yticklabels(
            _md_display_residue_labels(
                [str(values[0].get("residue") or "") for values in scoped_rows]
            ),
            fontsize=5.5,
        )
        if scoped_rows:
            axis.set_ylim(len(scoped_rows) - 0.5, -0.5)
        axis.set_xlim(0.0, 1.0)
        fraction_ticks = np.linspace(0.0, 1.0, 6)
        axis.set_xticks(fraction_ticks)
        axis.set_xticklabels([f"{value:.0%}" for value in fraction_ticks])
        axis.set_xlabel("Fraction of scoped interactions", fontsize=7)
        fig.legend(
            *axis.get_legend_handles_labels(),
            fontsize=4.5,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=2,
            handlelength=1.1,
            handletextpad=0.35,
            columnspacing=0.8,
            borderaxespad=0.0,
        )
        plotted = bool(scoped_rows)
    elif plot_id == "interaction_composition":
        rows, _, _, _ = _md_hotspot_rows(
            report,
            limit=interaction_limit,
        )
        rows = _md_replica_interaction_rows(report, rows, replica)
        positions = np.arange(len(rows))
        interaction_specs = [
            (
                "mean_hydrogen_bond_occupancy",
                "sample_sd_hydrogen_bond_occupancy",
                "H-bond",
                "#800000",
            ),
            (
                "mean_hydrophobic_occupancy",
                "sample_sd_hydrophobic_occupancy",
                "Hydrophobic",
                "#90ee90",
            ),
            (
                "mean_water_bridge_occupancy",
                "sample_sd_water_bridge_occupancy",
                "Water bridge",
                "#1e90ff",
            ),
        ]
        if report.get("salt_bridge_applicable"):
            interaction_specs.append(
                (
                    "mean_salt_bridge_occupancy",
                    "sample_sd_salt_bridge_occupancy",
                    "Salt bridge",
                    "#cc33aa",
                )
            )
        width = 0.72 / max(1, len(interaction_specs))
        for index, (
            field,
            error_field,
            label,
            color,
        ) in enumerate(interaction_specs):
            values = np.asarray(
                [float(row.get(field) or 0.0) for row in rows]
            )
            axis.bar(
                positions
                + (
                    index - (len(interaction_specs) - 1) / 2
                )
                * width,
                values,
                yerr=(
                    None
                    if replica is not None
                    else [
                        float(row.get(error_field) or 0.0)
                        for row in rows
                    ]
                ),
                capsize=2 if replica is None else 0,
                width=width,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                label=label,
            )
        axis.set_xticks(positions)
        residue_labels = [
            str(row.get("residue") or "") for row in rows
        ]
        axis.set_xticklabels(
            _md_display_residue_labels(residue_labels),
            rotation=55,
            ha="right",
            fontsize=5,
        )
        axis.set_ylabel("Fraction of analyzed frames", fontsize=7)
        axis.set_ylim(0, 1.05)
        axis.legend(
            fontsize=5,
            frameon=False,
            ncol=len(interaction_specs),
        )
        plotted = bool(rows)
    elif plot_id == "contact_matrix":
        hotspot_rows, _, _, _ = _md_hotspot_rows(
            report,
            limit=interaction_limit,
        )
        residue_order = [
            str(row.get("residue") or "") for row in hotspot_rows
        ]
        replica_blocks: list[np.ndarray] = []
        replica_labels: list[str] = []
        for replica_series in series:
            times = np.asarray(
                replica_series.get("time_ns") or [], dtype=float
            )
            cutoff = float(
                replica_series.get("contact_cutoff_angstrom") or 4.5
            )
            distance_series = (
                replica_series.get("contact_distance_series_angstrom") or {}
            )
            available_lengths = [
                min(len(times), len(values))
                for values in distance_series.values()
                if values
            ]
            length = max(available_lengths, default=0)
            if length == 0 or not residue_order:
                continue
            bin_size = max(1, int(np.ceil(length / 200)))
            block_rows: list[np.ndarray] = []
            for residue in residue_order:
                distances = distance_series.get(residue)
                contact_values = np.full(length, np.nan, dtype=float)
                if distances:
                    distance_values = np.asarray(distances, dtype=float)
                    row_length = min(length, len(distance_values))
                    contact_values[:row_length] = (
                        distance_values[:row_length] <= cutoff
                    ).astype(float)
                block_rows.append(
                    np.asarray(
                        [
                            float(
                                np.mean(
                                    contact_values[index:index + bin_size]
                                )
                            )
                            for index in range(0, length, bin_size)
                        ]
                    )
                )
            width = min(len(values) for values in block_rows)
            replica_blocks.append(
                np.vstack([values[:width] for values in block_rows])
            )
            replica_labels.append(
                f"Replica {replica_series.get('replica', '-')}\n"
                f"{times[0]:.0f}–{times[length - 1]:.0f} ns"
            )
        if replica_blocks:
            separator = np.full((len(residue_order), 3), np.nan)
            combined_parts: list[np.ndarray] = []
            tick_positions: list[float] = []
            offset = 0
            for index, block in enumerate(replica_blocks):
                combined_parts.append(block)
                tick_positions.append(offset + (block.shape[1] - 1) / 2)
                offset += block.shape[1]
                if index < len(replica_blocks) - 1:
                    combined_parts.append(separator)
                    offset += separator.shape[1]
            values = np.hstack(combined_parts)
            cmap = plt.get_cmap("coolwarm").copy()
            cmap.set_bad("#d9d9d9")
            image = axis.imshow(
                values,
                aspect="auto",
                vmin=0,
                vmax=1,
                cmap=cmap,
                interpolation="nearest",
            )
            axis.set_yticks(np.arange(len(residue_order)))
            axis.set_yticklabels(
                _md_display_residue_labels(residue_order),
                fontsize=4.8,
            )
            axis.set_xticks(tick_positions)
            axis.set_xticklabels(replica_labels, fontsize=5.5)
            axis.set_xlabel(
                "Production time within each replica · gray = not stored",
                fontsize=7,
            )
            colorbar_axis = axis.inset_axes((0.965, 0.70, 0.014, 0.24))
            colorbar = fig.colorbar(
                image,
                cax=colorbar_axis,
                ticks=(0.0, 1.0),
            )
            colorbar.ax.yaxis.set_ticks_position("left")
            colorbar.ax.tick_params(
                labelsize=4,
                length=1.5,
                width=0.4,
                pad=1,
            )
            plotted = True
        else:
            matrix = report.get("contact_matrix") or {}
            residues = [
                str(value) for value in matrix.get("residues") or []
            ][:20]
            values = np.asarray(
                (matrix.get("contact_occupancy") or [])[: len(residues)],
                dtype=float,
            )
            if values.size:
                image = axis.imshow(
                    values,
                    aspect="auto",
                    vmin=0,
                    vmax=1,
                    cmap="coolwarm",
                )
                axis.set_yticks(np.arange(len(residues)))
                axis.set_yticklabels(
                    _md_display_residue_labels(residues),
                    fontsize=5,
                )
                axis.set_xticks(np.arange(values.shape[1]))
                axis.set_xticklabels(
                    [
                        f"R{value}"
                        for value in matrix.get("replicas") or []
                    ],
                    fontsize=7,
                )
                colorbar_axis = axis.inset_axes(
                    (0.965, 0.70, 0.014, 0.24)
                )
                colorbar = fig.colorbar(
                    image,
                    cax=colorbar_axis,
                    ticks=(0.0, 1.0),
                )
                colorbar.ax.yaxis.set_ticks_position("left")
                colorbar.ax.tick_params(
                    labelsize=4,
                    length=1.5,
                    width=0.4,
                    pad=1,
                )
                plotted = True
    elif plot_id in {"protein_rmsf", "ligand_rmsf"}:
        labels: list[str]
        if replica is not None and series:
            field_prefix = (
                "protein_rmsf" if plot_id == "protein_rmsf" else "ligand_rmsf"
            )
            labels = [
                str(value)
                for value in series[0].get(
                    f"{field_prefix}_residues"
                    if plot_id == "protein_rmsf"
                    else f"{field_prefix}_atoms"
                )
                or []
            ]
            means = np.asarray(
                series[0].get(f"{field_prefix}_angstrom") or [],
                dtype=float,
            )
            errors = np.zeros(len(means), dtype=float)
        else:
            rows = (
                report.get("rmsf_consensus")
                if plot_id == "protein_rmsf"
                else report.get("ligand_rmsf_consensus")
            ) or []
            mean_field = (
                "mean_ca_rmsf_angstrom"
                if plot_id == "protein_rmsf"
                else "mean_rmsf_angstrom"
            )
            error_field = (
                "sample_sd_ca_rmsf_angstrom"
                if plot_id == "protein_rmsf"
                else "sample_sd_rmsf_angstrom"
            )
            labels = [
                str(
                    row.get(
                        "residue" if plot_id == "protein_rmsf" else "atom"
                    )
                    or ""
                )
                for row in rows
            ]
            means = np.asarray(
                [float(row.get(mean_field) or 0.0) for row in rows]
            )
            errors = np.asarray(
                [float(row.get(error_field) or 0.0) for row in rows]
            )
        positions = np.arange(len(means), dtype=float)
        protein_uses_native_numbers = False
        if plot_id == "protein_rmsf":
            residue_matches = [
                re.match(
                    r"^[A-Za-z]{3}(-?\d+)[A-Za-z]?"
                    r"(?:\s*·\s*chain\s+.+?)?$",
                    label,
                )
                for label in labels
            ]
            chain_matches = [
                re.search(r"\s*·\s*chain\s+(.+?)\s*$", label)
                for label in labels
            ]
            chains = {
                match.group(1)
                for match in chain_matches
                if match is not None
            }
            if (
                labels
                and all(match is not None for match in residue_matches)
                and len(chains) <= 1
            ):
                positions = np.asarray(
                    [
                        float(match.group(1))
                        for match in residue_matches
                        if match is not None
                    ],
                    dtype=float,
                )
                protein_uses_native_numbers = True
                native_order = np.argsort(positions, kind="stable")
                positions = positions[native_order]
                means = means[native_order]
                errors = errors[native_order]
                labels = [labels[index] for index in native_order]
                if len(positions) > 1:
                    # Never draw a continuous RMSF trace through absent author
                    # residue numbers.  NaN separators make true numbering
                    # gaps visible instead of inventing a smooth segment.
                    gap_indices = np.flatnonzero(np.diff(positions) > 1.0)
                    for gap_index in reversed(gap_indices.tolist()):
                        insert_at = gap_index + 1
                        positions = np.insert(
                            positions,
                            insert_at,
                            (positions[gap_index] + positions[insert_at]) / 2.0,
                        )
                        means = np.insert(means, insert_at, np.nan)
                        errors = np.insert(errors, insert_at, np.nan)
                        labels.insert(insert_at, "")
        axis.plot(positions, means, linewidth=0.9)
        if replica is None:
            axis.fill_between(
                positions,
                np.maximum(0.0, means - errors),
                means + errors,
                alpha=0.18,
            )
        if plot_id == "ligand_rmsf":
            axis.set_xticks(positions)
            axis.set_xticklabels(
                labels,
                rotation=90,
                fontsize=5,
            )
        axis.set_xlabel(
            (
                "Original residue number"
                if protein_uses_native_numbers
                else "Residue position"
            )
            if plot_id == "protein_rmsf"
            else "Ligand atom",
            fontsize=7,
        )
        axis.set_ylabel("RMSF (Å)", fontsize=7)
        plotted = bool(len(means))
    elif plot_id in {"endpoint_energy", "endpoint_energy_pb"}:
        components = (
            (
                (
                    "delta_g_bind_kcal_mol",
                    "delta_mm_kcal_mol",
                    "delta_gbsa_kcal_mol",
                    "delta_nonpolar_kcal_mol",
                )
                if plot_id == "endpoint_energy"
                else (
                    "pb_delta_g_bind_kcal_mol",
                    "pb_delta_mm_kcal_mol",
                    "pb_delta_pbsa_kcal_mol",
                    "pb_delta_nonpolar_kcal_mol",
                )
            )
        )
        labels = [
            "ΔG total",
            "ΔMM",
            "ΔGB" if plot_id == "endpoint_energy" else "ΔPB",
            "Δnonpolar",
        ]
        replica_values = [
            [
                float(row[field])
                for row in replicas
                if row.get(field) is not None
            ]
            for field in components
        ]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        if replica_bars:
            positions = np.arange(len(replicas), dtype=float)
            width = 0.18
            for index, (label, values, color) in enumerate(
                zip(labels, replica_values, colors, strict=True)
            ):
                axis.bar(
                    positions + (index - 1.5) * width,
                    values,
                    width=width,
                    color=color,
                    label=label,
                )
            axis.set_xticks(positions)
            axis.set_xticklabels(
                [f"R{row.get('replica', '-')}" for row in replicas]
            )
            axis.legend(
                fontsize=5,
                frameon=False,
                loc="upper left",
            )
            plotted = bool(replicas)
        else:
            plotted = _plot_md_consensus_bars(
                axis,
                labels,
                replica_values,
                colors=colors,
            )
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_ylabel("Energy (kcal/mol)", fontsize=7)
    elif plot_id == "endpoint_binding_summary":
        replica_values = [
            [
                float(row[field])
                for row in replicas
                if row.get(field) is not None
            ]
            for field in (
                "delta_g_bind_kcal_mol",
                "pb_delta_g_bind_kcal_mol",
            )
        ]
        if replica_bars:
            positions = np.arange(len(replicas), dtype=float)
            width = 0.28
            for offset, (label, values, color) in enumerate(
                zip(
                    ("MM/GBSA", "MM/PBSA"),
                    replica_values,
                    ("#4c78a8", "#e45756"),
                    strict=True,
                )
            ):
                axis.bar(
                    positions + (offset - 0.5) * width,
                    values,
                    width=width,
                    color=color,
                    label=label,
                )
            axis.set_xticks(positions)
            axis.set_xticklabels(
                [f"R{row.get('replica', '-')}" for row in replicas]
            )
            axis.legend(fontsize=5, frameon=False)
            plotted = bool(replicas)
        else:
            plotted = _plot_md_consensus_bars(
                axis,
                ["MM/GBSA", "MM/PBSA"],
                replica_values,
                colors=["#4c78a8", "#e45756"],
            )
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_ylabel("Final ΔG bind (kcal/mol)", fontsize=7)
        axis.margins(x=0.22)
    elif plot_id == "throughput":
        values = [
            float(row["performance_ns_per_day"])
            for row in replicas
            if row.get("performance_ns_per_day") is not None
        ]
        if replica_bars:
            axis.bar(
                [f"R{row.get('replica', '-')}" for row in replicas],
                values,
                width=0.55,
                color="#1f77b4",
            )
            plotted = bool(values)
        else:
            plotted = _plot_md_consensus_bars(
                axis,
                ["Throughput"],
                [values],
                colors=["#1f77b4"],
            )
        axis.set_ylabel("ns/day", fontsize=7)
    elif plot_id == "interaction_network":
        rows, _, _, _ = _md_hotspot_rows(
            report,
            limit=interaction_limit,
        )
        rows = _md_replica_interaction_rows(report, rows, replica)
        depiction = report.get("ligand_depiction") or {}
        atoms = [
            atom
            for atom in depiction.get("atoms") or []
            if isinstance(atom, dict)
        ]
        bonds = [
            bond
            for bond in depiction.get("bonds") or []
            if isinstance(bond, dict)
        ]
        if atoms:
            atom_xy = np.asarray(
                [
                    [float(atom.get("x") or 0.0), float(atom.get("y") or 0.0)]
                    for atom in atoms
                ],
                dtype=float,
            )
            atom_xy *= 1.68
            atom_xy[:, 1] += 0.16
        else:
            angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            atom_xy = np.column_stack((np.cos(angles), np.sin(angles))) * 0.55
            atoms = [{"symbol": "C"} for _ in range(6)]
            bonds = [
                {"begin": index, "end": (index + 1) % 6, "order": 1.0}
                for index in range(6)
            ]
        atom_positions = {
            str(atom.get("name") or ""): atom_xy[index]
            for index, atom in enumerate(atoms)
            if atom.get("name")
        }
        atom_colors = {
            "C": "#151515",
            "N": "#2166c2",
            "O": "#d73027",
            "S": "#d9a300",
            "P": "#e67e22",
            "F": "#37a055",
            "Cl": "#37a055",
            "Br": "#8c510a",
            "I": "#7b3294",
        }
        for bond in bonds:
            begin = int(bond.get("begin") or 0)
            end = int(bond.get("end") or 0)
            if begin >= len(atom_xy) or end >= len(atom_xy):
                continue
            start = atom_xy[begin]
            finish = atom_xy[end]
            order = float(bond.get("order") or 1.0)
            axis.plot(
                [start[0], finish[0]],
                [start[1], finish[1]],
                color="#252525",
                linewidth=1.0 + 0.35 * max(0.0, order - 1.0),
                zorder=3,
            )
        for atom, coordinates in zip(atoms, atom_xy):
            symbol = str(atom.get("symbol") or "C")
            formal_charge = int(atom.get("formal_charge") or 0)
            axis.scatter(
                [coordinates[0]],
                [coordinates[1]],
                s=54,
                color=atom_colors.get(symbol, "#666666"),
                edgecolor="white",
                linewidth=0.35,
                zorder=4,
            )
            if symbol != "C" or formal_charge:
                charge_label = (
                    "+" if formal_charge == 1
                    else "−" if formal_charge == -1
                    else f"{formal_charge:+d}" if formal_charge
                    else ""
                )
                axis.text(
                    coordinates[0],
                    coordinates[1],
                    f"{symbol}{charge_label}",
                    color="white",
                    fontsize=5.0,
                    ha="center",
                    va="center",
                    zorder=5,
                )
        node_angles = np.linspace(
            np.pi * 0.9, np.pi * 0.9 - 2 * np.pi, len(rows), endpoint=False
        )
        network_residue_labels = _md_display_residue_labels(
            [str(row.get("residue") or "") for row in rows]
        )
        edge_styles = [
            (
                "mean_hydrogen_bond_occupancy",
                "Hydrogen bond",
                "#800000",
            ),
            (
                "mean_hydrophobic_occupancy",
                "Hydrophobic",
                "#90ee90",
            ),
        ]
        if include_water_bridges:
            edge_styles.append(
                (
                    "mean_water_bridge_occupancy",
                    "Water bridge",
                    "#1e90ff",
                )
            )
        if report.get("salt_bridge_applicable"):
            edge_styles.append(
                (
                    "mean_salt_bridge_occupancy",
                    "Salt bridge",
                    "#cc33aa",
                )
            )
        scope_specs = {
            "mean_hydrogen_bond_occupancy": (
                "HB",
                "mean_hydrogen_bond_backbone_occupancy",
                "mean_hydrogen_bond_sidechain_occupancy",
            ),
            "mean_hydrophobic_occupancy": (
                "HP",
                "mean_hydrophobic_backbone_occupancy",
                "mean_hydrophobic_sidechain_occupancy",
            ),
            "mean_salt_bridge_occupancy": (
                "SB",
                "mean_salt_bridge_backbone_occupancy",
                "mean_salt_bridge_sidechain_occupancy",
            ),
        }

        def _interaction_scope(row: dict[str, Any], field: str) -> str:
            spec = scope_specs.get(field)
            if spec is None:
                return ""
            _short_name, backbone_field, sidechain_field = spec
            backbone = float(row.get(backbone_field) or 0.0) > 0
            sidechain = float(row.get(sidechain_field) or 0.0) > 0
            if backbone and sidechain:
                return "BB+SC"
            if backbone:
                return "BB"
            if sidechain:
                return "SC"
            return ""

        for row_index, (angle, row, residue_label) in enumerate(
            zip(node_angles, rows, network_residue_labels, strict=True)
        ):
            x_value = 2.18 * np.cos(angle)
            y_value = 1.40 * np.sin(angle) + 0.18
            contact = float(row.get("mean_contact_occupancy") or 0.0)
            direction = np.asarray([x_value, y_value], dtype=float)
            target = atom_positions.get(
                str(row.get("top_ligand_atom") or "")
            )
            if target is None:
                target = atom_xy[
                    int(
                        np.argmax(
                            atom_xy
                            @ (direction / np.linalg.norm(direction))
                        )
                    )
                ]
            drawn_edge = False
            for style_index, (field, _label, color) in enumerate(
                edge_styles
            ):
                occupancy = float(row.get(field) or 0.0)
                if occupancy <= 0:
                    continue
                curvature = (
                    0.10 * (-1 if (row_index + style_index) % 2 else 1)
                    * (style_index + 1)
                )
                axis.add_patch(
                    FancyArrowPatch(
                        (float(target[0]), float(target[1])),
                        (x_value, y_value),
                        connectionstyle=f"arc3,rad={curvature}",
                        arrowstyle="-",
                        linestyle="--",
                        linewidth=0.55 + 3.0 * occupancy,
                        color=color,
                        alpha=0.45 + 0.5 * occupancy,
                        zorder=1,
                    )
                )
                label_fraction = 0.52 + 0.11 * style_index
                label_x = float(target[0]) + label_fraction * (
                    x_value - float(target[0])
                )
                label_y = float(target[1]) + label_fraction * (
                    y_value - float(target[1])
                )
                axis.text(
                    label_x,
                    label_y,
                    (
                        "<1%"
                        if occupancy < 0.01
                        else f"{occupancy:.0%}"
                    )
                    + (
                        f" · {_interaction_scope(row, field)}"
                        if show_interaction_scopes
                        and _interaction_scope(row, field)
                        else ""
                    ),
                    fontsize=3.8,
                    color=color,
                    ha="center",
                    va="center",
                    zorder=5,
                )
                drawn_edge = True
            if not drawn_edge and include_proximity_interactions:
                axis.add_patch(
                    FancyArrowPatch(
                        (float(target[0]), float(target[1])),
                        (x_value, y_value),
                        connectionstyle="arc3,rad=0.08",
                        arrowstyle="-",
                        linestyle=":",
                        linewidth=0.7 + contact,
                        color="#999999",
                        alpha=0.65,
                        zorder=1,
                    )
                )
                axis.text(
                    float(target[0]) + 0.72 * (
                        x_value - float(target[0])
                    ),
                    float(target[1]) + 0.72 * (
                        y_value - float(target[1])
                    ),
                    f"{contact:.0%}",
                    fontsize=3.8,
                    color="#777777",
                    ha="center",
                    va="center",
                    zorder=5,
                )
            if not drawn_edge and not include_proximity_interactions:
                continue
            axis.scatter(
                [x_value],
                [y_value],
                s=115 + 70 * contact,
                color=_md_residue_node_color(
                    str(row.get("residue") or "")
                ),
                edgecolor="#4c738c",
                linewidth=0.75,
                zorder=2,
            )
            axis.text(
                x_value,
                y_value,
                (
                    residue_label
                    + (
                        "\n"
                        + " · ".join(
                            f"{short_name}:{_interaction_scope(row, field)}"
                            for field, _label, _color in edge_styles
                            for (
                                short_name,
                                backbone_field,
                                sidechain_field,
                            ) in [scope_specs.get(field, ("", "", ""))]
                            if field in scope_specs
                            and float(row.get(field) or 0.0) > 0
                            and _interaction_scope(row, field)
                        )
                        if show_interaction_scopes
                        else ""
                    )
                ),
                ha="center",
                va="center",
                fontsize=4.1 if show_interaction_scopes else 4.5,
                linespacing=1.15,
                zorder=3,
            )
        for _field, label, color in edge_styles:
            axis.plot(
                [],
                [],
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=label,
            )
        if include_proximity_interactions:
            axis.plot(
                [],
                [],
                color="#999999",
                linestyle=":",
                linewidth=1.0,
                label="Proximity only",
            )
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            fontsize=5,
            frameon=False,
        )
        axis.set_xlim(-2.42, 2.42)
        axis.set_ylim(-1.72, 1.72)
        axis.set_aspect("equal")
        axis.axis("off")
        plotted = bool(rows)
    elif plot_id == "persistence_distance":
        rows, _, _, _ = _md_hotspot_rows(
            report,
            limit=interaction_limit,
        )
        rows = [
            row
            for row in rows
            if row.get("mean_minimum_distance_angstrom") is not None
        ]
        if replica is None:
            mean_distances = [
                float(row.get("mean_minimum_distance_angstrom") or 0.0)
                for row in rows
            ]
            mean_occupancies = [
                float(row.get("mean_contact_occupancy") or 0.0)
                for row in rows
            ]
            axis.errorbar(
                mean_distances,
                mean_occupancies,
                xerr=[
                    float(
                        row.get(
                            "sample_sd_minimum_distance_angstrom"
                        )
                        or 0.0
                    )
                    for row in rows
                ],
                yerr=[
                    float(row.get("sample_sd_contact_occupancy") or 0.0)
                    for row in rows
                ],
                fmt="o",
                markersize=4,
                color="#6f52c8",
                ecolor="#6f52c8",
                elinewidth=0.7,
                capsize=2,
                label="Mean ± sample SD",
            )
            occupancy_errors = [
                float(row.get("sample_sd_contact_occupancy") or 0.0)
                for row in rows
            ]
        else:
            matrix = (
                report.get("contact_matrix")
                if isinstance(report.get("contact_matrix"), dict)
                else {}
            )
            matrix_replicas = [
                int(value) for value in matrix.get("replicas") or []
            ]
            replica_index = (
                matrix_replicas.index(replica)
                if replica in matrix_replicas
                else None
            )
            replica_rows: list[tuple[dict[str, Any], float, float]] = []
            if replica_index is not None:
                replica_series = next(
                    (
                        row
                        for row in series
                        if int(row.get("replica") or 0) == replica
                    ),
                    {},
                )
                temporal_distances = (
                    replica_series.get(
                        "contact_distance_series_angstrom"
                    )
                    or {}
                )
                for row in rows:
                    residue = str(row.get("residue") or "")
                    distances = _md_contact_replica_values(
                        report,
                        residue,
                        "minimum_distance_angstrom",
                    )
                    occupancies = _md_contact_replica_values(
                        report,
                        residue,
                        "contact_occupancy",
                    )
                    if replica_index >= len(distances):
                        residue_distances = temporal_distances.get(residue) or []
                        if residue_distances:
                            distances = [float("nan")] * len(matrix_replicas)
                            distances[replica_index] = float(
                                np.mean(
                                    np.asarray(
                                        residue_distances,
                                        dtype=float,
                                    )
                                )
                            )
                    if (
                        replica_index < len(distances)
                        and replica_index < len(occupancies)
                        and np.isfinite(distances[replica_index])
                    ):
                        replica_rows.append(
                            (
                                row,
                                distances[replica_index],
                                occupancies[replica_index],
                            )
                        )
            axis.scatter(
                [values[1] for values in replica_rows],
                [values[2] for values in replica_rows],
                s=24,
                color="#6f52c8",
            )
            rows = [values[0] for values in replica_rows]
            mean_distances = [values[1] for values in replica_rows]
            mean_occupancies = [values[2] for values in replica_rows]
        persistence_rows = rows
        persistence_labels = _md_display_residue_labels(
            [
                str(row.get("residue") or "")
                for row in persistence_rows
            ]
        )
        label_offsets = (
            (3, 5),
            (3, -9),
            (-30, 5),
            (-30, -9),
        )
        for index, (row, residue_label) in enumerate(
            zip(
                persistence_rows,
                persistence_labels,
                strict=True,
            )
        ):
            axis.annotate(
                residue_label,
                (
                    mean_distances[index],
                    mean_occupancies[index],
                ),
                fontsize=4,
                xytext=label_offsets[index % len(label_offsets)],
                textcoords="offset points",
            )
        axis.set_xlabel("Mean minimum distance (Å)", fontsize=7)
        axis.set_ylabel("Mean contact occupancy", fontsize=7)
        if replica is None:
            lower_values = [
                mean - error
                for mean, error in zip(
                    mean_occupancies,
                    occupancy_errors,
                    strict=True,
                )
            ]
            upper_values = [
                mean + error
                for mean, error in zip(
                    mean_occupancies,
                    occupancy_errors,
                    strict=True,
                )
            ]
            lower = max(0.0, min(lower_values, default=0.0))
            upper = min(1.0, max(upper_values, default=1.0))
            span = max(0.02, upper - lower)
            axis.set_ylim(
                max(0.0, lower - 0.12 * span),
                min(1.02, upper + 0.12 * span),
            )
            axis.legend(fontsize=5, frameon=False, loc="center right")
        else:
            lower = min(mean_occupancies, default=0.0)
            upper = max(mean_occupancies, default=1.0)
            span = max(0.02, upper - lower)
            axis.set_ylim(
                max(0.0, lower - 0.12 * span),
                min(1.02, upper + 0.12 * span),
            )
        plotted = bool(rows)
    elif plot_id == "interface_rin":
        replicas_with_interface = [
            row
            for row in report.get("interface_rin_replicas") or []
            if isinstance(row, dict)
        ]
        pair_rows = (
            replicas_with_interface[0].get("residue_pairs") or []
            if replicas_with_interface
            else []
        )[:20]
        residues = sorted(
            {
                str(row.get(key) or "")
                for row in pair_rows
                for key in ("residue_a", "residue_b")
                if row.get(key)
            }
        )
        angles = np.linspace(0, 2 * np.pi, len(residues), endpoint=False)
        positions = {
            residue: (np.cos(angle), np.sin(angle))
            for residue, angle in zip(residues, angles)
        }
        interface_labels = dict(
            zip(
                residues,
                _md_display_residue_labels(residues),
                strict=True,
            )
        )
        for row in pair_rows:
            left = positions.get(str(row.get("residue_a") or ""))
            right = positions.get(str(row.get("residue_b") or ""))
            if left is None or right is None:
                continue
            occupancy = float(row.get("contact_occupancy") or 0.0)
            axis.plot(
                [left[0], right[0]],
                [left[1], right[1]],
                color="#6f52c8",
                linewidth=0.4 + 2.0 * occupancy,
                alpha=0.65,
                zorder=1,
            )
        for residue, (x_value, y_value) in positions.items():
            axis.scatter(
                [x_value],
                [y_value],
                s=75,
                color=_md_residue_node_color(residue),
                edgecolor="#4c738c",
                linewidth=0.6,
                zorder=2,
            )
            axis.text(
                x_value * 1.18,
                y_value * 1.18,
                interface_labels[residue],
                fontsize=4,
                ha="center",
                va="center",
            )
        axis.set_aspect("equal")
        axis.axis("off")
        plotted = bool(pair_rows)
    if not plotted:
        plt.close(fig)
        return None
    axis.set_title(
        _MD_PLOT_LABELS[plot_id]
        + (f" — Replica {replica}" if replica is not None else ""),
        fontsize=9,
    )
    axis.tick_params(axis="both", labelsize=6.5)
    if plot_id not in {"contact_matrix", "interaction_network"}:
        axis.grid(True, alpha=0.18, linewidth=0.5)
    interaction_bar_plots = {
        "contact_occupancy",
        "hbond_occupancy",
        "hydrophobic_occupancy",
        "water_bridge_occupancy",
        "salt_bridge_occupancy",
        "contact_scope_fraction",
        "hbond_scope_fraction",
        "hydrophobic_scope_fraction",
        "water_bridge_scope_fraction",
        "salt_bridge_scope_fraction",
    }
    if plot_id in interaction_bar_plots:
        # Identical axis geometry keeps each occupancy panel aligned with its
        # BB/SC companion.  The scope legend lives in the reserved footer and
        # therefore cannot compress the plotting area.
        fig.subplots_adjust(
            left=0.20,
            right=0.97,
            top=0.86,
            bottom=0.18,
        )
    elif plot_id == "contact_matrix":
        # The compact color key is inset, so the matrix uses exactly the same
        # frame as the other residue-interaction panels.
        fig.subplots_adjust(
            left=0.20,
            right=0.97,
            top=0.86,
            bottom=0.18,
        )
    elif plot_id == "interaction_composition":
        fig.subplots_adjust(
            left=0.18,
            right=0.92,
            top=0.86,
            bottom=0.28,
        )
    else:
        fig.tight_layout(pad=0.7)
    return fig


def _md_plot_export_bundle(
    report: dict[str, Any],
    render_specs: list[tuple[str, int | None, bool]],
    *,
    column_count: int,
    interaction_limit: int,
    selected_replicas: list[int],
    panel_dpi: int = 300,
    combined_dpi: int = 600,
) -> tuple[bytes, dict[str, Any]]:
    panel_width_inches = 4.2
    panel_height_inches = 3.0
    panel_dpi = max(100, min(600, int(panel_dpi)))
    combined_dpi = max(100, min(600, int(combined_dpi)))
    rendered_panels: list[
        tuple[str, bytes, tuple[str, int | None, bool]]
    ] = []
    used_names: dict[str, int] = {}
    for index, (plot_id, replica_id, replica_bars) in enumerate(
        render_specs,
        start=1,
    ):
        figure = _md_plot_figure(
            plot_id,
            report,
            replica=replica_id,
            replica_bars=replica_bars,
            selected_replicas=(
                selected_replicas if replica_bars else None
            ),
            interaction_limit=interaction_limit,
        )
        if figure is None:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", plot_id.lower()).strip("-")
        if replica_id is not None:
            slug += f"-replica-{replica_id}"
        elif replica_bars:
            slug += "-selected-replicas"
        duplicate_index = used_names.get(slug, 0) + 1
        used_names[slug] = duplicate_index
        if duplicate_index > 1:
            slug += f"-{duplicate_index}"
        filename = f"{index:03d}_{slug}.png"
        buffer = BytesIO()
        try:
            figure.savefig(
                buffer,
                format="png",
                dpi=panel_dpi,
                facecolor="white",
                bbox_inches=None,
            )
        finally:
            plt.close(figure)
        rendered_panels.append(
            (filename, buffer.getvalue(), (plot_id, replica_id, replica_bars))
        )

    panel_count = len(rendered_panels)
    if not panel_count:
        raise RuntimeError("No visible MD plots were available for export")
    columns = max(1, min(int(column_count), panel_count))
    rows = int(math.ceil(panel_count / columns))
    maximum_composite_pixels = 120_000_000
    pixels_per_dpi_squared = (
        columns
        * rows
        * panel_width_inches
        * panel_height_inches
    )
    effective_combined_dpi = max(
        72,
        min(
            combined_dpi,
            int(
                math.sqrt(
                    maximum_composite_pixels / pixels_per_dpi_squared
                )
            ),
        ),
    )
    cell_width = round(panel_width_inches * effective_combined_dpi)
    cell_height = round(panel_height_inches * effective_combined_dpi)
    composite = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )
    for index, (_, panel_png, spec) in enumerate(rendered_panels):
        if effective_combined_dpi == panel_dpi:
            combined_panel_png = panel_png
        else:
            plot_id, replica_id, replica_bars = spec
            figure = _md_plot_figure(
                plot_id,
                report,
                replica=replica_id,
                replica_bars=replica_bars,
                selected_replicas=(
                    selected_replicas if replica_bars else None
                ),
                interaction_limit=interaction_limit,
            )
            if figure is None:
                continue
            combined_panel_buffer = BytesIO()
            try:
                figure.savefig(
                    combined_panel_buffer,
                    format="png",
                    dpi=effective_combined_dpi,
                    facecolor="white",
                    bbox_inches=None,
                )
            finally:
                plt.close(figure)
            combined_panel_png = combined_panel_buffer.getvalue()
        with Image.open(BytesIO(combined_panel_png)) as panel:
            composite.paste(
                panel.convert("RGB"),
                (
                    (index % columns) * cell_width,
                    (index // columns) * cell_height,
                ),
            )
    composite_buffer = BytesIO()
    composite.save(
        composite_buffer,
        format="PNG",
        dpi=(effective_combined_dpi, effective_combined_dpi),
        optimize=True,
    )
    composite.close()
    manifest = {
        "panel_count": panel_count,
        "panel_size_inches": [panel_width_inches, panel_height_inches],
        "panel_dpi": panel_dpi,
        "matrix_columns": columns,
        "matrix_rows": rows,
        "requested_composite_dpi": combined_dpi,
        "composite_dpi": effective_combined_dpi,
        "composite_pixel_size": [
            columns * cell_width,
            rows * cell_height,
        ],
        "selected_replicas": selected_replicas,
        "interaction_residue_limit": interaction_limit,
        "panels": [filename for filename, _, _ in rendered_panels],
    }
    archive_buffer = BytesIO()
    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for filename, png_data, _ in rendered_panels:
            archive.writestr(f"panels/{filename}", png_data)
        archive.writestr(
            "md-plots-combined.png",
            composite_buffer.getvalue(),
        )
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    return archive_buffer.getvalue(), manifest


def _md_plot_data_tables(
    report: dict[str, Any],
    *,
    selected_replicas: list[int] | None = None,
    interaction_limit: int = 12,
    selected_plots: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Flatten the data behind the MD plot matrix into analysis-ready tables."""
    available_replicas = sorted(
        {
            int(row.get("replica") or 0)
            for row in report.get("replica_series") or report.get("replicas") or []
            if isinstance(row, dict) and int(row.get("replica") or 0) > 0
        }
    )
    selected = set(
        available_replicas if selected_replicas is None else selected_replicas
    )
    settings = pd.DataFrame(
        [
            {"Setting": "Selected replicas", "Value": ", ".join(map(str, sorted(selected)))},
            {"Setting": "Interaction residue limit", "Value": int(interaction_limit)},
            {
                "Setting": "Selected plots",
                "Value": ", ".join(
                    _MD_PLOT_LABELS.get(value, value)
                    for value in (selected_plots or [])
                ),
            },
        ]
    )
    tables: dict[str, pd.DataFrame] = {"Export settings": settings}

    replica_rows = pd.DataFrame(
        [
            row
            for row in report.get("replicas") or []
            if isinstance(row, dict)
            and int(row.get("replica") or 0) in selected
        ]
    )
    if not replica_rows.empty:
        tables["Replica summary"] = replica_rows
        summary_long: list[dict[str, Any]] = []
        for _, row in replica_rows.iterrows():
            replica = int(row.get("replica") or 0)
            for parameter, value in row.items():
                if parameter == "replica" or isinstance(value, (dict, list, tuple)):
                    continue
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(numeric):
                    summary_long.append(
                        {
                            "Parameter": str(parameter),
                            "Replica": replica,
                            "Value": float(numeric),
                        }
                    )
        if summary_long:
            summary = pd.DataFrame(summary_long)
            wide = summary.pivot_table(
                index="Parameter", columns="Replica", values="Value", aggfunc="mean"
            ).reset_index()
            replica_columns = [
                column for column in wide if isinstance(column, (int, float))
            ]
            wide = wide.rename(
                columns={
                    column: f"Replica {int(column)}" for column in replica_columns
                }
            )
            value_columns = [
                f"Replica {int(column)}" for column in replica_columns
            ]
            wide["Mean"] = wide[value_columns].mean(axis=1)
            wide["Sample SD"] = wide[value_columns].std(axis=1, ddof=1)
            wide["Replica count"] = wide[value_columns].count(axis=1)
            tables["Summary by replica"] = wide

    time_series_fields = (
        "backbone_rmsd_angstrom",
        "ligand_rmsd_angstrom",
        "ligand_centroid_displacement_angstrom",
        "minimum_protein_distance_angstrom",
        "protein_rg_angstrom",
        "ligand_rg_angstrom",
        "complex_rg_angstrom",
        "helix_fraction",
        "sheet_fraction",
        "coil_fraction",
    )
    series_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    protein_rmsf_rows: list[dict[str, Any]] = []
    ligand_rmsf_rows: list[dict[str, Any]] = []
    consensus = [
        row for row in report.get("contact_consensus") or [] if isinstance(row, dict)
    ]
    ranked_residues = [str(row.get("residue") or "") for row in consensus]
    ranked_residues = [value for value in ranked_residues if value][
        : max(1, int(interaction_limit))
    ]
    for replica_series in report.get("replica_series") or []:
        if not isinstance(replica_series, dict):
            continue
        replica = int(replica_series.get("replica") or 0)
        if replica not in selected:
            continue
        times = list(replica_series.get("time_ns") or [])
        for index, time_ns in enumerate(times):
            output: dict[str, Any] = {"Replica": replica, "Time (ns)": time_ns}
            for field in time_series_fields:
                values = replica_series.get(field) or []
                output[field] = values[index] if index < len(values) else None
            series_rows.append(output)
        contact_distances = replica_series.get("contact_distance_series_angstrom")
        if isinstance(contact_distances, dict):
            residues = ranked_residues or list(contact_distances)
            for residue in residues:
                values = contact_distances.get(residue) or []
                for index, value in enumerate(values[: len(times)]):
                    distance_rows.append(
                        {
                            "Replica": replica,
                            "Time (ns)": times[index] if index < len(times) else None,
                            "Residue": residue,
                            "Minimum distance (angstrom)": value,
                        }
                    )
        for residue, value in zip(
            replica_series.get("protein_rmsf_residues") or [],
            replica_series.get("protein_rmsf_angstrom") or [],
        ):
            protein_rmsf_rows.append(
                {"Replica": replica, "Residue": residue, "RMSF (angstrom)": value}
            )
        for atom, value in zip(
            replica_series.get("ligand_rmsf_atoms") or [],
            replica_series.get("ligand_rmsf_angstrom") or [],
        ):
            ligand_rmsf_rows.append(
                {"Replica": replica, "Atom": atom, "RMSF (angstrom)": value}
            )
    for name, rows in (
        ("Time series", series_rows),
        ("Contact distances", distance_rows),
        ("Protein RMSF", protein_rmsf_rows),
        ("Ligand RMSF", ligand_rmsf_rows),
    ):
        if rows:
            tables[name] = pd.DataFrame(rows)

    if consensus:
        consensus_frame = pd.DataFrame(consensus)
        if ranked_residues and "residue" in consensus_frame:
            consensus_frame = consensus_frame.loc[
                consensus_frame["residue"].astype(str).isin(ranked_residues)
            ]
        tables["Contact consensus"] = consensus_frame
    matrix = report.get("contact_matrix")
    if isinstance(matrix, dict):
        residues = [str(value) for value in matrix.get("residues") or []]
        if ranked_residues:
            residues = [value for value in ranked_residues if value in residues]
        else:
            residues = residues[: max(1, int(interaction_limit))]
        matrix_replicas = [int(value) for value in matrix.get("replicas") or []]
        metric_names = [
            key
            for key, value in matrix.items()
            if key not in {"residues", "replicas"} and isinstance(value, list)
        ]
        matrix_rows: list[dict[str, Any]] = []
        source_residues = [str(value) for value in matrix.get("residues") or []]
        for residue in residues:
            residue_index = source_residues.index(residue)
            for replica_index, replica in enumerate(matrix_replicas):
                if replica not in selected:
                    continue
                output = {"Residue": residue, "Replica": replica}
                for metric in metric_names:
                    values = matrix.get(metric) or []
                    row_values = values[residue_index] if residue_index < len(values) else []
                    output[metric] = (
                        row_values[replica_index]
                        if isinstance(row_values, list) and replica_index < len(row_values)
                        else None
                    )
                matrix_rows.append(output)
        if matrix_rows:
            tables["Contact matrix"] = pd.DataFrame(matrix_rows)

    for sheet_name, report_key in (
        ("Protein RMSF consensus", "rmsf_consensus"),
        ("Ligand RMSF consensus", "ligand_rmsf_consensus"),
        ("Interface RIN", "interface_rin_replicas"),
    ):
        rows = [row for row in report.get(report_key) or [] if isinstance(row, dict)]
        if rows:
            frame = pd.DataFrame(rows)
            if "replica" in frame:
                frame = frame.loc[
                    pd.to_numeric(frame["replica"], errors="coerce").isin(selected)
                ]
            if not frame.empty:
                tables[sheet_name] = frame

    aggregate_rows: list[dict[str, Any]] = []
    for parameter, value in (report.get("aggregate") or {}).items():
        if isinstance(value, dict):
            aggregate_rows.append({"Parameter": parameter, **value})
        else:
            aggregate_rows.append({"Parameter": parameter, "Value": value})
    if aggregate_rows:
        tables["Aggregate"] = pd.DataFrame(aggregate_rows)
    return tables


def _md_plot_data_workbook(
    report: dict[str, Any],
    *,
    selected_replicas: list[int] | None = None,
    interaction_limit: int = 12,
    selected_plots: list[str] | None = None,
) -> bytes:
    tables = _md_plot_data_tables(
        report,
        selected_replicas=selected_replicas,
        interaction_limit=interaction_limit,
        selected_plots=selected_plots,
    )
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in tables.items():
            prepared = frame.copy()
            for column in prepared:
                if prepared[column].dtype == object:
                    prepared[column] = prepared[column].map(
                        lambda value: json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple, set))
                        else value
                    )
            prepared.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buffer.getvalue()


def _render_md_plot_matrix(
    report: dict[str, Any],
    *,
    run_id: str,
) -> None:
    available = _available_md_plot_ids(report)
    if not available:
        return
    defaults = list(available)
    replica_display = (
        st.segmented_control(
            "Replica display",
            ("Consensus", "Separate", "Both"),
            default="Consensus",
            key=f"md_plot_replica_display_{run_id}",
            help=(
                "Consensus shows replica means with sample SD where applicable. "
                "Separate creates one panel per replica from stored analysis data."
            ),
        )
        or "Consensus"
    )
    all_replica_ids = sorted(
        {
            int(row.get("replica") or 0)
            for row in report.get("replica_series") or []
            if isinstance(row, dict) and int(row.get("replica") or 0) > 0
        }
    )
    replica_arrangement = "Parallel"
    selected_replica_ids = list(all_replica_ids)
    if replica_display in {"Separate", "Both"} and all_replica_ids:
        st.markdown("##### Separate-replica view")
        replica_arrangement = st.radio(
            "Arrangement",
            ("Parallel", "Serial"),
            horizontal=True,
            key=f"md_replica_arrangement_{run_id}",
            help=(
                "Parallel groups each plot across the selected repeats before "
                "the next plot. Serial completes all selected plots for one "
                "repeat before the next. Matrix columns controls both layouts."
            ),
        )
        st.caption("Repeats shown")
        repeat_columns = st.columns(min(4, len(all_replica_ids)))
        selected_replica_ids = [
            replica_id
            for index, replica_id in enumerate(all_replica_ids)
            if repeat_columns[index % len(repeat_columns)].checkbox(
                f"Repeat {replica_id}",
                value=True,
                key=f"md_show_replica_{replica_id}_{run_id}",
            )
        ]
        if not selected_replica_ids:
            st.warning("Select at least one replica for the separate view.")
    available_interactions = len(
        [
            row
            for row in report.get("contact_consensus") or []
            if isinstance(row, dict)
        ]
    )
    interaction_limit = int(
        st.number_input(
            "Residues shown in interaction plots",
            min_value=1,
            max_value=max(12, available_interactions),
            value=min(12, max(1, available_interactions)),
            step=1,
            key=f"md_plot_interaction_limit_{run_id}",
            help=(
                "Uses the binding-hotspot ranking as the shared residue set "
                "and order for all compatible interaction panels."
            ),
        )
    )
    with st.expander("Plot matrix layout"):
        column_count = int(
            st.number_input(
                "Matrix columns",
                min_value=1,
                max_value=4,
                value=3,
                step=1,
                key=f"md_plot_matrix_columns_{run_id}",
            )
        )
        layout = pd.DataFrame(
            [
                {
                    "show": plot_id in defaults,
                    "order": (
                        defaults.index(plot_id) + 1
                        if plot_id in defaults
                        else len(defaults) + index + 1
                    ),
                    "plot": _MD_PLOT_LABELS[plot_id],
                    "plot_id": plot_id,
                }
                for index, plot_id in enumerate(available)
            ]
        )
        edited = st.data_editor(
            layout,
            hide_index=True,
            width="stretch",
            disabled=("plot", "plot_id"),
            column_config={
                "show": st.column_config.CheckboxColumn("Show"),
                "order": st.column_config.NumberColumn(
                    "Order", min_value=1, step=1
                ),
                "plot": st.column_config.TextColumn("Plot", width="large"),
                "plot_id": None,
            },
            key=f"md_plot_matrix_layout_v3_{run_id}",
        )
        st.caption(
            "All available figures are selected by default. Enable panels and "
            "assign order numbers. All panels use the same figure size; layout "
            "changes use stored data and do not reread trajectories. BB/SC "
            "panels normalize each residue's backbone and side-chain scoped "
            "occupancies to 100%; a frame containing both contributes to both "
            "scopes before normalization."
        )
    selected = (
        edited.loc[edited["show"].astype(bool)]
        .sort_values(["order", "plot"], kind="stable")
        ["plot_id"]
        .astype(str)
        .tolist()
    )
    if not selected:
        st.info("Select at least one plot in Plot matrix layout.")
        return
    selected = _md_ordered_plot_ids(selected, replica_display)
    render_specs: list[tuple[str, int | None, bool]] = []
    for plot_id in selected:
        if plot_id in _MD_REPLICA_BAR_PLOTS:
            if replica_display in {"Consensus", "Both"}:
                render_specs.append((plot_id, None, False))
            if replica_display in {"Separate", "Both"}:
                render_specs.append((plot_id, None, True))
            continue
        can_split = (
            plot_id in _MD_REPLICA_SPLIT_PLOTS
            and selected_replica_ids
        )
        if replica_display in {"Consensus", "Both"} or not can_split:
            render_specs.append((plot_id, None, False))
    if replica_display in {"Separate", "Both"}:
        separate_specs = _md_separate_plot_specs(
            selected,
            selected_replica_ids,
            replica_arrangement,
        )
        if replica_display == "Separate":
            summary_specs = [
                spec
                for spec in render_specs
                if spec[0] in _MD_REPLICA_BAR_PLOTS
            ]
            render_specs = separate_specs + summary_specs
        else:
            render_specs.extend(separate_specs)
    columns = st.columns(column_count)
    for index, (plot_id, replica_id, replica_bars) in enumerate(
        render_specs
    ):
        figure = _md_plot_figure(
            plot_id,
            report,
            replica=replica_id,
            replica_bars=replica_bars,
            selected_replicas=(
                selected_replica_ids if replica_bars else None
            ),
            interaction_limit=interaction_limit,
        )
        if figure is None:
            continue
        with columns[index % column_count]:
            st.pyplot(
                figure,
                clear_figure=True,
                use_container_width=True,
                bbox_inches=None,
            )

    st.markdown("##### Export current plot view")
    export_resolution_columns = st.columns(2)
    export_dpi = int(
        export_resolution_columns[0].number_input(
            "Individual PNG resolution (DPI)",
            min_value=100,
            max_value=600,
            value=300,
            step=50,
            key=f"md_plot_export_dpi_{run_id}",
        )
    )
    combined_export_dpi = int(
        export_resolution_columns[1].number_input(
            "Combined PNG resolution (DPI)",
            min_value=100,
            max_value=600,
            value=600,
            step=50,
            key=f"md_plot_combined_export_dpi_{run_id}",
        )
    )
    export_signature = json.dumps(
        {
            "render_specs": render_specs,
            "columns": column_count,
            "interaction_limit": interaction_limit,
            "selected_replicas": selected_replica_ids,
            "replica_display": replica_display,
            "replica_arrangement": replica_arrangement,
            "panel_dpi": export_dpi,
            "combined_dpi": combined_export_dpi,
        },
        sort_keys=True,
    )
    export_state_key = f"md_plot_export_bundle_{run_id}"
    if st.button(
        "Prepare PNG ZIP",
        key=f"md_prepare_plot_export_{run_id}",
        help=(
            "Exports every currently visible panel at the selected DPI and "
            "one large combined matrix PNG using the current filters and layout."
        ),
    ):
        with st.spinner("Rendering high-resolution plot bundle…"):
            try:
                archive_data, manifest = _md_plot_export_bundle(
                    report,
                    render_specs,
                    column_count=column_count,
                    interaction_limit=interaction_limit,
                    selected_replicas=selected_replica_ids,
                    panel_dpi=export_dpi,
                    combined_dpi=combined_export_dpi,
                )
                st.session_state[export_state_key] = {
                    "signature": export_signature,
                    "archive": archive_data,
                    "manifest": manifest,
                }
            except Exception as exc:
                st.session_state.pop(export_state_key, None)
                st.error(f"Plot export failed: {exc}")
    prepared_export = st.session_state.get(export_state_key)
    if (
        isinstance(prepared_export, dict)
        and prepared_export.get("signature") == export_signature
        and isinstance(prepared_export.get("archive"), bytes)
    ):
        manifest = prepared_export.get("manifest") or {}
        st.download_button(
            "Download plot PNG ZIP",
            data=prepared_export["archive"],
            file_name=f"md-plots-{run_id}.zip",
            mime="application/zip",
            key=f"md_download_plot_export_{run_id}",
        )
        effective_combined_dpi = int(
            manifest.get("composite_dpi") or 0
        )
        requested_combined_dpi = int(
            manifest.get("requested_composite_dpi") or 0
        )
        combined_resolution_note = (
            f"{effective_combined_dpi} DPI"
            + (
                f" (requested {requested_combined_dpi}; safety-capped)"
                if requested_combined_dpi > effective_combined_dpi
                else ""
            )
        )
        st.caption(
            f"{int(manifest.get('panel_count') or 0)} individual "
            f"{int(manifest.get('panel_dpi') or 0)}-DPI "
            f"PNGs · combined {int(manifest.get('matrix_columns') or 0)}-column "
            f"PNG at {combined_resolution_note} · "
            "manifest.json included."
        )

    data_signature = json.dumps(
        {
            "selected_replicas": selected_replica_ids,
            "interaction_limit": interaction_limit,
            "selected_plots": selected,
        },
        sort_keys=True,
    )
    data_state_key = f"md_plot_data_workbook_{run_id}"
    if st.button(
        "Prepare plot data workbook",
        key=f"md_prepare_plot_data_{run_id}",
        help=(
            "Creates an Excel workbook containing the selected replicas and "
            "the data behind the MD plot matrix, including time series, RMSF, "
            "contacts, consensus values, and replicate summaries."
        ),
    ):
        with st.spinner("Preparing MD plot data workbook…"):
            try:
                workbook = _md_plot_data_workbook(
                    report,
                    selected_replicas=selected_replica_ids,
                    interaction_limit=interaction_limit,
                    selected_plots=selected,
                )
                st.session_state[data_state_key] = {
                    "signature": data_signature,
                    "workbook": workbook,
                }
            except Exception as exc:
                st.session_state.pop(data_state_key, None)
                st.error(f"MD data export failed: {exc}")
    prepared_data = st.session_state.get(data_state_key)
    if (
        isinstance(prepared_data, dict)
        and prepared_data.get("signature") == data_signature
        and isinstance(prepared_data.get("workbook"), bytes)
    ):
        st.download_button(
            "Download MD plot data workbook",
            data=prepared_data["workbook"],
            file_name=f"md-plot-data-{run_id}.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key=f"md_download_plot_data_{run_id}",
        )


def _render_md_replicate_metrics(job: JobRecord) -> bool:
    if job.task_group != "md-analysis":
        return False
    report = job.result if isinstance(job.result, dict) else {}
    replicas = [
        row for row in report.get("replicas") or [] if isinstance(row, dict)
    ]
    if not replicas:
        st.info("No completed MD replica summaries are available.")
        return True

    st.markdown("#### Independent-replica comparison")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Replicas", len(replicas))
    summary_columns[1].metric(
        "Production completed", int(report.get("completed") or 0)
    )
    summary_columns[2].metric(
        "Endpoint completed", int(report.get("endpoint_completed") or 0)
    )
    summary_columns[3].metric("Failed stages", int(report.get("failed") or 0))
    performance = (
        (report.get("aggregate") or {}).get("performance_ns_per_day")
        if isinstance(report.get("aggregate"), dict)
        else {}
    )
    if isinstance(performance, dict) and performance.get("count"):
        performance_columns = st.columns(3)
        performance_columns[0].metric(
            "Mean throughput",
            f"{float(performance['mean']):,.1f} ns/day",
        )
        performance_columns[1].metric(
            "Replica SD",
            (
                f"{float(performance['sample_sd']):,.1f} ns/day"
                if performance.get("sample_sd") is not None
                else "—"
            ),
        )
        performance_columns[2].metric(
            "Measured replicas",
            int(performance["count"]),
        )
    st.dataframe(pd.DataFrame(replicas), hide_index=True, width="stretch")
    _render_md_plot_matrix(report, run_id=job.run_id)
    return True


def render() -> None:
    task_group = str(st.query_params.get("task_group", "")).strip()
    run_id = str(st.query_params.get("run_id", "")).strip()
    run_dir = _resolve_run_dir(task_group, run_id)

    st.title("Job Results")
    if run_dir is None:
        st.error("The requested job could not be found.")
        if st.button("Back to Jobs"):
            st.switch_page("app/pages/unified_jobs.py")
        return

    requested_job = JobRecord.load(run_dir, task_group=task_group)
    job = (
        _latest_superseding_job(requested_job)
        if requested_job.task_group == "md-analysis"
        else requested_job
    )
    if job.run_id != requested_job.run_id:
        run_dir = job.run_dir
        run_id = job.run_id
        st.info(
            "This MD analysis has been superseded. Showing the current "
            f"revision {display_job_code(job.metadata.get('job_code'), job.run_id)} "
            f"instead of historical revision "
            f"{display_job_code(requested_job.metadata.get('job_code'), requested_job.run_id)}."
        )
    top = st.columns([0.82, 0.18])
    with top[0]:
        st.caption(f"{task_group} / {run_id}")
    with top[1]:
        if st.button("Back to Jobs", width="stretch"):
            st.switch_page("app/pages/unified_jobs.py")

    admission = job.metadata.get("admission")
    if isinstance(admission, dict) and admission.get("status") in {"waiting", "rejected"}:
        reasons = admission.get("reasons")
        message = "; ".join(str(reason) for reason in reasons) if isinstance(reasons, list) else ""
        st.warning(f"Resource admission {admission.get('status')}: {message or 'capacity unavailable'}")

    cancel_state = cancellation_eligibility(job)
    retry_state = retry_eligibility(job)
    if cancel_state.allowed or retry_state.allowed:
        st.markdown("#### Job actions")
        action_columns = st.columns(2)
        with action_columns[0]:
            confirm_cancel = st.checkbox(
                "Confirm cancellation",
                key=f"job_results_confirm_cancel_{job.run_id}",
                disabled=not cancel_state.allowed,
            )
            if st.button(
                "Cancel job",
                disabled=not cancel_state.allowed or not confirm_cancel,
                key=f"job_results_cancel_{job.run_id}",
            ):
                try:
                    request_job_cancellation(job)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success("Cancellation requested. Refresh to follow the terminal state.")
        with action_columns[1]:
            if st.button(
                "Retry as new job",
                disabled=not retry_state.allowed,
                key=f"job_results_retry_{job.run_id}",
            ):
                try:
                    retried = create_job_retry(job)
                except (OSError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    retry_query = urlencode(
                        {
                            "task_group": retried.task_group,
                            "run_id": retried.run_id,
                        }
                    )
                    st.success(f"Queued retry {retried.run_id}.")
                    st.link_button("Open retry", f"./job-results?{retry_query}")

    detailed_url = _detailed_result_url(job)
    if detailed_url:
        st.link_button("Open detailed results", detailed_url)
    artifact_types = {
        artifact.artifact_type
        for artifact in (job.artifact_manifest.artifacts if job.artifact_manifest else ())
    }
    if "imported_target" in artifact_types:
        st.link_button(
            "Clean protein",
            f"./workspace-protein-cleaning?{urlencode({'source_run_id': job.run_id})}",
        )
    if "prepared_target" in artifact_types:
        st.link_button(
            "Use in Structure Import",
            f"./workspace-structure-preparation?{urlencode({'prepared_target_run_id': job.run_id})}",
        )
    if "docked_complex" in artifact_types:
        st.link_button(
            "Prepare for MD",
            f"./workspace-md-simulation?{urlencode({'source_run_id': job.run_id})}",
        )
    if job.workflow in {
        "docking_campaign",
        "openvs_docking",
        "alphafold3_refolding",
        "boltz2_refolding",
    }:
        st.link_button(
            "Validate poses with PoseBusters",
            (
                "./evaluate-pose-validation?"
                + urlencode(
                    {
                        "source_task_group": job.task_group,
                        "source_run_id": job.run_id,
                    }
                )
            ),
        )

    result_tab_label = "Dataset" if job.task_group == "compound-import" else "Viewer"
    overview_tab, artifacts_tab, metrics_tab, viewer_tab, lineage_tab, logs_tab = st.tabs(
        ["Overview", "Artifacts", "Metrics", result_tab_label, "Lineage", "Logs"]
    )
    inventory = _file_inventory(run_dir)
    rendered_mmgbsa = False

    with overview_tab:
        summary = [
            {"field": "status", "value": job.status},
            {"field": "task", "value": job.task_group},
            {"field": "job type", "value": job.job_type},
            {"field": "tool", "value": _display_tool_name(job.tool)},
            {"field": "workflow", "value": job.workflow or "-"},
            {"field": "workflow parent", "value": job.workflow_parent_run_id or "-"},
            {"field": "created", "value": job.created_at or "-"},
            {"field": "updated", "value": job.updated_at or "-"},
            {"field": "completed", "value": job.completed_at or "-"},
            {"field": "schema version", "value": str(job.schema_version)},
        ]
        st.dataframe(summary, hide_index=True, width="stretch")
        for warning in job.warnings:
            st.warning(warning)
        rendered_mmgbsa = _render_mmgbsa_metrics(job)
        workflow_children = job.result.get("children") if job.task_group == "workflows" else None
        if isinstance(workflow_children, list):
            progress = job.result.get("progress") if isinstance(job.result.get("progress"), dict) else {}
            st.markdown("#### Workflow progress")
            if progress:
                st.progress(int(progress.get("percent") or 0) / 100)
                st.caption(
                    f"{int(progress.get('completed') or 0)} of {int(progress.get('total') or 0)} steps completed"
                )
            if workflow_children:
                show_workflow_history = st.checkbox(
                    "Show failed and superseded workflow history",
                    value=False,
                    key=f"workflow_show_failed_history_{job.run_id}",
                    help=(
                        "Historical attempts remain on disk and in lineage, "
                        "but are hidden from this progress table by default."
                    ),
                )
                child_rows = []
                hidden_history_count = 0
                for item in workflow_children:
                    if not isinstance(item, dict):
                        continue
                    task_group = str(item.get("task_group") or "")
                    run_id = str(item.get("run_id") or "")
                    child_dir = _resolve_run_dir(task_group, run_id)
                    child_job = (
                        JobRecord.load(child_dir, task_group=task_group)
                        if child_dir is not None
                        else None
                    )
                    superseded = bool(
                        child_job is not None
                        and child_job.metadata.get("superseded_by_run_id")
                    )
                    optional_failure = bool(
                        not item.get("required", True)
                        and child_job is not None
                        and child_job.status == "failed"
                    )
                    if (
                        not show_workflow_history
                        and (superseded or optional_failure)
                    ):
                        hidden_history_count += 1
                        continue
                    detailed_url = (
                        _detailed_result_url(child_job)
                        if child_job is not None
                        else ""
                    )
                    if not detailed_url and child_job is not None:
                        detailed_url = "./job-results?" + urlencode(
                            {
                                "task_group": task_group,
                                "run_id": run_id,
                            }
                        )
                    child_rows.append(
                        {
                            "results": detailed_url,
                            "step": item.get("step_id", ""),
                            "status": (
                                "superseded"
                                if (
                                    child_job is not None
                                    and child_job.metadata.get(
                                        "superseded_by_run_id"
                                    )
                                )
                                else child_job.status
                                if child_job is not None
                                else item.get("status", "missing")
                            ),
                            "task": task_group,
                            "required": bool(item.get("required", True)),
                            "job": (
                                display_job_code(
                                    child_job.metadata.get("job_code"),
                                    child_job.run_id,
                                )
                                if child_job is not None
                                else run_id[:8]
                            ),
                            "depends_on": ", ".join(
                                item.get("depends_on") or []
                            ),
                        }
                    )
                if hidden_history_count:
                    st.caption(
                        f"{hidden_history_count} failed/superseded historical "
                        "attempt(s) hidden."
                    )
                st.dataframe(
                    child_rows,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "results": st.column_config.LinkColumn(
                            "Results", display_text="Open"
                        ),
                        "step": st.column_config.TextColumn(
                            "Workflow stage", width="large"
                        ),
                        "status": st.column_config.TextColumn("Status"),
                        "task": st.column_config.TextColumn("Task"),
                        "required": st.column_config.CheckboxColumn(
                            "Required"
                        ),
                        "job": st.column_config.TextColumn("Job"),
                        "depends_on": st.column_config.TextColumn(
                            "Depends on", width="large"
                        ),
                    },
                )
        st.markdown("#### Inputs and parameters")
        input_path = run_dir / "input.json"
        if input_path.is_file():
            try:
                st.json(json.loads(input_path.read_text()))
            except (OSError, ValueError):
                st.code(input_path.read_text(errors="replace")[-50000:])
        else:
            run_inputs = job.metadata.get("run_inputs") or job.metadata.get("parameters")
            st.json(run_inputs if isinstance(run_inputs, dict) else job.metadata)

    with artifacts_tab:
        artifacts = job.artifact_manifest.artifacts if job.artifact_manifest else ()
        if artifacts:
            st.dataframe(
                [
                    {
                        "type": item.artifact_type,
                        "role": item.role,
                        "path": item.path,
                        "size_mb": round((item.size_bytes or 0) / (1024 * 1024), 3),
                        "sha256": item.sha256,
                    }
                    for item in artifacts
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("This legacy job has no typed artifacts yet.")
        st.markdown("#### Run files")
        st.dataframe(inventory, hide_index=True, width="stretch")

    with metrics_tab:
        rendered_md_production = _render_md_production_metrics(job)
        rendered_redocking = _render_redocking_metrics(job)
        rendered_openvs = _render_openvs_scores(job)
        rendered_docking = _render_docking_scores(job)
        rendered_refolding = _render_refolding_metrics(job)
        rendered_rescoring = _render_rescoring_metrics(job)
        rendered_pose_validation = _render_pose_validation_metrics(job)
        rendered_interactions = _render_interaction_analysis_metrics(job)
        rendered_generation = _render_generation_metrics(job)
        rendered_molecule_qualification = _render_molecule_qualification_metrics(job)
        rendered_md_analysis = _render_md_replicate_metrics(job)
        metrics = _flatten_scalars(job.result)
        if metrics:
            if (
                rendered_redocking
                or rendered_openvs
                or rendered_docking
                or rendered_refolding
                or rendered_rescoring
                or rendered_pose_validation
                or rendered_interactions
                or rendered_generation
                or rendered_molecule_qualification
                or rendered_mmgbsa
                or rendered_md_production
                or rendered_md_analysis
            ):
                st.markdown("#### Result metadata")
            st.dataframe(pd.DataFrame(metrics), hide_index=True, width="stretch")
        elif not any(
            (
                rendered_redocking,
                rendered_openvs,
                rendered_docking,
                rendered_refolding,
                rendered_rescoring,
                rendered_pose_validation,
                rendered_interactions,
                rendered_generation,
                rendered_molecule_qualification,
                rendered_mmgbsa,
                rendered_md_analysis,
            )
        ):
            st.info("No scalar result metrics were found.")

    with viewer_tab:
        candidates = [run_dir / row["path"] for row in inventory if Path(row["path"]).suffix.lower() in VIEWABLE_SUFFIXES]
        rendered_pocket_context = _render_pocket_context(job)
        rendered_docking_context = _render_docking_complex_viewer(job)
        rendered_refolding_context = _render_refolding_complex_viewer(job)
        rendered_rescoring_context = _render_rescoring_viewer(job)
        rendered_pose_validation_context = _render_pose_validation_viewer(job)
        rendered_interaction_context = _render_interaction_analysis_viewer(job)
        rendered_generation_context = _render_generation_compound_viewer(job)
        rendered_qualification_context = _render_molecule_qualification_viewer(job)
        if job.task_group == "compound-import":
            render_compound_dataset_report(job)
        elif rendered_pocket_context:
            pass
        elif rendered_docking_context:
            pass
        elif rendered_refolding_context:
            pass
        elif rendered_rescoring_context:
            pass
        elif rendered_pose_validation_context:
            pass
        elif rendered_interaction_context:
            pass
        elif rendered_generation_context:
            pass
        elif rendered_qualification_context:
            pass
        elif candidates:
            if job.workflow == "redocking_benchmark":
                st.caption(
                    "Redocking overlay SDF files contain the crystallographic reference followed "
                    "by the top-ranked predicted pose for that engine replicate."
                )
            selected = st.selectbox(
                "Structure file",
                candidates,
                format_func=lambda path: path.relative_to(run_dir).as_posix(),
                key=f"generic_viewer_{task_group}_{run_id}",
            )
            if job.workflow == "redocking_benchmark" and selected.parent.name == "overlays":
                _render_redocking_overlay(selected)
            else:
                _render_viewer(selected)
        else:
            st.info("No supported structure file is available for preview.")

    with lineage_tab:
        all_jobs = iter_job_records(runs_root())
        st.dataframe(_lineage_rows(job, all_jobs), hide_index=True, width="stretch")

    with logs_tab:
        log_paths = sorted(path for path in run_dir.rglob("*.log") if path.is_file())
        if log_paths:
            selected_log = st.selectbox(
                "Log file",
                log_paths,
                format_func=lambda path: path.relative_to(run_dir).as_posix(),
                key=f"generic_log_{task_group}_{run_id}",
            )
            st.code(selected_log.read_text(errors="replace")[-100000:])
        else:
            tails = {
                key: value
                for payload in (job.metadata, job.result)
                for key, value in payload.items()
                if key in {"stdout", "stderr", "stdout_tail", "stderr_tail", "error"} and value
            }
            if tails:
                for key, value in tails.items():
                    st.markdown(f"#### {key}")
                    st.code(str(value)[-100000:])
            else:
                st.info("No logs were recorded for this job.")


if __name__ == "__main__":
    render()
