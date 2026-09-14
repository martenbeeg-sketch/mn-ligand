from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFMCS, rdMolDescriptors

from mn_ligand.app.pages.benchmark_common import (
    render_case_viewer,
    visible_dataset_jobs,
)
from mn_ligand.app.viewers import aligned_structure_data
from mn_ligand.core.jobs import JobRecord, display_job_code, iter_job_records
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.benchmark_datasets import (
    benchmark_cases,
    list_benchmark_dataset_jobs,
    load_benchmark_dataset,
)


def _job_url(job: JobRecord) -> str:
    return "./job-results?" + urlencode(
        {
            "task_group": job.task_group,
            "run_id": job.run_id,
            "label": display_job_code(job.metadata.get("job_code"), job.run_id),
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", errors="replace") as handle:
        return list(csv.DictReader(handle))


def _benchmark_value(job: JobRecord, key: str):
    value = job.metadata.get(key)
    if value not in {None, ""}:
        return value
    parameters = job.metadata.get("parameters")
    if not isinstance(parameters, dict):
        return None
    context = parameters.get("context")
    return context.get(key) if isinstance(context, dict) else None


@st.cache_data(show_spinner=False)
def _dataset_statistics(run_dir_text: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    run_dir = Path(run_dir_text)
    dataset = json.loads((run_dir / "benchmark_dataset.json").read_text())
    rows = []
    for case in dataset.get("cases") or []:
        ligand_path = run_dir / str(case.get("reference_ligand_path") or "")
        receptor_path = run_dir / str(case.get("receptor_path") or "")
        molecule = next(
            (
                molecule
                for molecule in Chem.SDMolSupplier(
                    str(ligand_path), removeHs=False
                )
                if molecule is not None
            ),
            None,
        )
        residues = set()
        chains = set()
        atom_count = 0
        if receptor_path.is_file():
            for line in receptor_path.read_text(errors="replace").splitlines():
                if not line.startswith("ATOM  ") or len(line) < 27:
                    continue
                atom_count += 1
                chain = line[21].strip() or "_"
                chains.add(chain)
                residues.add((chain, line[22:26].strip(), line[26].strip()))
        rows.append(
            {
                "case_id": case.get("case_id", ""),
                "target_id": case.get("target_id", ""),
                "split": case.get("split", ""),
                "receptor_chains": len(chains),
                "receptor_residues": len(residues),
                "receptor_atoms": atom_count,
                "ligand_fragments": (
                    len(Chem.GetMolFrags(molecule)) if molecule is not None else None
                ),
                "ligand_heavy_atoms": (
                    molecule.GetNumHeavyAtoms() if molecule is not None else None
                ),
                "formula": (
                    rdMolDescriptors.CalcMolFormula(molecule)
                    if molecule is not None
                    else ""
                ),
                "molecular_weight": (
                    Descriptors.MolWt(molecule) if molecule is not None else None
                ),
                "hbond_donors": (
                    Lipinski.NumHDonors(molecule) if molecule is not None else None
                ),
                "hbond_acceptors": (
                    Lipinski.NumHAcceptors(molecule) if molecule is not None else None
                ),
                "tpsa": (
                    rdMolDescriptors.CalcTPSA(molecule)
                    if molecule is not None
                    else None
                ),
                "rotatable_bonds": (
                    Lipinski.NumRotatableBonds(molecule)
                    if molecule is not None
                    else None
                ),
                "ring_count": (
                    Lipinski.RingCount(molecule) if molecule is not None else None
                ),
                "formal_charge": (
                    Chem.GetFormalCharge(molecule) if molecule is not None else None
                ),
                "logp": Crippen.MolLogP(molecule) if molecule is not None else None,
                "qed": QED.qed(molecule) if molecule is not None else None,
            }
        )
    return pd.DataFrame(rows)


def _reference_molecule(path: Path):
    molecule = next(
        (item for item in Chem.SDMolSupplier(str(path), removeHs=False) if item),
        None,
    )
    if molecule is None:
        raise ValueError("Reference ligand is not RDKit-readable")
    molecule = Chem.RemoveHs(molecule, sanitize=False)
    Chem.GetSymmSSSR(molecule)
    return molecule


def _complex_ligand(structure_data: str, ligand_smiles: str):
    import gemmi

    template = Chem.MolFromSmiles(ligand_smiles)
    if template is None:
        raise ValueError("Reference SMILES is unreadable")
    template = Chem.RemoveHs(template, sanitize=False)
    structure = gemmi.make_structure_from_block(
        gemmi.cif.read_string(structure_data).sole_block()
    )
    expected_elements = [atom.GetSymbol().upper() for atom in template.GetAtoms()]
    expected_counts = Counter(expected_elements)
    candidates = []
    for chain in structure[0]:
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info.is_amino_acid() or info.is_water():
                continue
            atoms = [atom for atom in residue if atom.element.name.upper() != "H"]
            if (
                len(atoms) == template.GetNumAtoms()
                and Counter(atom.element.name.upper() for atom in atoms)
                == expected_counts
            ):
                candidates.append(atoms)
    if len(candidates) != 1:
        raise ValueError("Could not identify exactly one matching predicted ligand")
    atoms = candidates[0]
    if [atom.element.name.upper() for atom in atoms] != expected_elements:
        raise ValueError("Predicted ligand atom order does not match its input graph")
    conformer = Chem.Conformer(template.GetNumAtoms())
    for index, atom in enumerate(atoms):
        conformer.SetAtomPosition(
            index, (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
        )
    template.RemoveAllConformers()
    template.AddConformer(conformer, assignId=True)
    Chem.GetSymmSSSR(template)
    return template


def _fixed_frame_rmsd(reference, predicted) -> float:
    if reference.GetNumAtoms() != predicted.GetNumAtoms():
        raise ValueError("Reference and prediction heavy-atom counts differ")
    common = rdFMCS.FindMCS(
        [reference, predicted],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=10,
    )
    query = Chem.MolFromSmarts(common.smartsString)
    if query is None or query.GetNumAtoms() != reference.GetNumAtoms():
        raise ValueError("The complete ligand graph could not be atom-mapped")
    reference_matches = reference.GetSubstructMatches(
        query, uniquify=False, maxMatches=100_000
    )
    predicted_matches = predicted.GetSubstructMatches(
        query, uniquify=False, maxMatches=100_000
    )
    reference_conf = reference.GetConformer()
    predicted_conf = predicted.GetConformer()
    best = math.inf
    for left in reference_matches:
        for right in predicted_matches:
            square = 0.0
            for left_index, right_index in zip(left, right):
                left_point = reference_conf.GetAtomPosition(left_index)
                right_point = predicted_conf.GetAtomPosition(right_index)
                square += (
                    (left_point.x - right_point.x) ** 2
                    + (left_point.y - right_point.y) ** 2
                    + (left_point.z - right_point.z) ** 2
                )
            best = min(best, math.sqrt(square / len(left)))
    if not math.isfinite(best):
        raise ValueError("No symmetry mapping could be evaluated")
    return best


@st.cache_data(show_spinner=False)
def _refolding_rmsd(
    receptor_path: str,
    receptor_mtime: int,
    ligand_path: str,
    ligand_mtime: int,
    predicted_path: str,
    predicted_mtime: int,
    ligand_smiles: str,
) -> tuple[float, float, int]:
    aligned, protein_rmsd, matched = aligned_structure_data(
        receptor_path,
        receptor_mtime,
        predicted_path,
        predicted_mtime,
    )
    reference = _reference_molecule(Path(ligand_path))
    predicted = _complex_ligand(aligned, ligand_smiles)
    return _fixed_frame_rmsd(reference, predicted), protein_rmsd, matched


def _redocking_rows(job: JobRecord) -> list[dict]:
    case_id = str(_benchmark_value(job, "benchmark_case_id") or "")
    rows = []
    for row in _read_csv(job.run_dir / "metrics" / "redocking_replicates.csv"):
        rows.append(
            {
                "case_id": case_id,
                "mode": "Redocking",
                "engine": str(row.get("engine") or "").upper(),
                "run_id": job.run_id,
                "source_docking_run_id": str(row.get("run_id") or ""),
                "replicate": int(row.get("replicate") or 1),
                "pose_rank": 1,
                "ligand_rmsd_angstrom": float(
                    row.get("top_pose_rmsd_angstrom") or "nan"
                ),
                "best_of_n_rmsd_angstrom": float(
                    row.get("best_pose_rmsd_angstrom") or "nan"
                ),
                "protein_alignment_rmsd_angstrom": 0.0,
                "matched_ca_atoms": None,
                "rmsd_frame": "shared fixed receptor frame",
                "job": _job_url(job),
            }
        )
    return rows


def _openvs_rows(job: JobRecord) -> list[dict]:
    case_id = str(job.metadata.get("benchmark_case_id") or "")
    rows = []
    for row in _read_csv(job.run_dir / "openvs_scores.csv"):
        value = row.get("ligand_rmsd_angstrom")
        if not value:
            continue
        rows.append(
            {
                "case_id": case_id,
                "mode": "Redocking",
                "engine": "RosettaLigand",
                "run_id": job.run_id,
                "source_docking_run_id": job.run_id,
                "replicate": int(row.get("replicate") or 1),
                "pose_rank": int(row.get("pose_rank") or 1),
                "ligand_rmsd_angstrom": float(value),
                "best_of_n_rmsd_angstrom": float(value),
                "protein_alignment_rmsd_angstrom": 0.0,
                "matched_ca_atoms": None,
                "rmsd_frame": "Rosetta reference-guided fixed receptor frame",
                "job": _job_url(job),
            }
        )
    return rows


def _refolding_rows(job: JobRecord, case) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []
    artifacts = (
        job.artifact_manifest.by_type("predicted_complex")
        if job.artifact_manifest
        else ()
    )
    for index, artifact in enumerate(artifacts, start=1):
        predicted = artifact.resolve(job.run_dir, must_exist=True)
        if predicted is None:
            continue
        try:
            rmsd, protein_rmsd, matched = _refolding_rmsd(
                str(case.receptor_path),
                case.receptor_path.stat().st_mtime_ns,
                str(case.ligand_path),
                case.ligand_path.stat().st_mtime_ns,
                str(predicted),
                predicted.stat().st_mtime_ns,
                str(case.payload.get("ligand_smiles") or ""),
            )
        except Exception as exc:
            warnings.append(
                f"{case.case_id} · {job.tool} · {artifact.role}: {exc}"
            )
            continue
        replicate_value = artifact.metadata.get("model_seed")
        if not replicate_value and "replicate_" in artifact.role:
            replicate_value = artifact.role.split("replicate_")[-1].split(":")[0]
        replicate = int(replicate_value or index)
        rows.append(
            {
                "case_id": case.case_id,
                "mode": "Refolding",
                "engine": job.tool,
                "run_id": job.run_id,
                "source_docking_run_id": "",
                "replicate": replicate,
                "pose_rank": 1,
                "ligand_rmsd_angstrom": rmsd,
                "best_of_n_rmsd_angstrom": rmsd,
                "protein_alignment_rmsd_angstrom": protein_rmsd,
                "matched_ca_atoms": matched,
                "rmsd_frame": "predicted complex rigidly protein-Cα aligned",
                "job": _job_url(job),
            }
        )
    return rows, warnings


def _rescoring_rows(
    jobs: list[JobRecord],
    pose_rmsd: dict[tuple[str, int], float],
) -> list[dict]:
    rows: list[dict] = []
    for job in jobs:
        if job.status != "completed" or job.workflow not in {
            "gnina_rescoring",
            "boltzina_rescoring",
        }:
            continue
        source_run_id = str(
            job.metadata.get("benchmark_source_docking_run_id") or ""
        )
        for row in _read_csv(job.run_dir / "rescoring_scores.csv"):
            rank = int(row.get("source_pose_rank") or 1)
            rows.append(
                {
                    "case_id": job.metadata.get("benchmark_case_id", ""),
                    "engine": job.tool,
                    "source_engine": row.get("source_engine", ""),
                    "source_docking_run_id": source_run_id,
                    "replicate": int(row.get("replicate") or 1),
                    "pose_rank": rank,
                    "ligand_rmsd_angstrom": pose_rmsd.get(
                        (source_run_id, rank)
                    ),
                    "source_score_kcal_mol": row.get(
                        "source_score_kcal_mol", ""
                    ),
                    "gnina_empirical_score_kcal_mol": row.get(
                        "gnina_empirical_score_kcal_mol", ""
                    ),
                    "gnina_cnn_score": row.get("gnina_cnn_score", ""),
                    "gnina_cnn_affinity": row.get("gnina_cnn_affinity", ""),
                    "boltzina_affinity_log10_ic50_uM": row.get(
                        "boltzina_affinity_log10_ic50_uM", ""
                    ),
                    "boltzina_binder_probability": row.get(
                        "boltzina_binder_probability", ""
                    ),
                    "maximum_coordinate_displacement_angstrom": row.get(
                        "maximum_coordinate_displacement_angstrom", ""
                    ),
                    "job": _job_url(job),
                }
            )
    return rows


def _collect(
    dataset_job,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    cases = {case.case_id: case for case in benchmark_cases(dataset_job)}
    jobs = [
        job
        for job in iter_job_records(runs_root())
        if str(_benchmark_value(job, "benchmark_dataset_run_id") or "")
        == dataset_job.run_id
    ]
    metric_rows: list[dict] = []
    pose_rmsd: dict[tuple[str, int], float] = {}
    warnings: list[str] = []
    for job in jobs:
        if job.status != "completed":
            continue
        if job.task_group == "workflows" and job.workflow == "redocking_benchmark":
            metric_rows.extend(_redocking_rows(job))
            for row in _read_csv(
                job.run_dir / "metrics" / "redocking_poses.csv"
            ):
                try:
                    pose_rmsd[
                        (
                            str(row.get("run_id") or ""),
                            int(row.get("pose_rank") or 1),
                        )
                    ] = float(row["symmetry_rmsd_angstrom"])
                except (KeyError, TypeError, ValueError):
                    continue
        elif job.workflow == "openvs_docking":
            metric_rows.extend(_openvs_rows(job))
        elif job.workflow in {"alphafold3_refolding", "boltz2_refolding"}:
            case = cases.get(str(job.metadata.get("benchmark_case_id") or ""))
            if case is not None:
                rows, errors = _refolding_rows(job, case)
                metric_rows.extend(rows)
                warnings.extend(errors)
    campaign_rows = [
        {
            "job": _job_url(job),
            "job_code": display_job_code(job.metadata.get("job_code"), job.run_id),
            "case_id": _benchmark_value(job, "benchmark_case_id") or "",
            "campaign_id": _benchmark_value(job, "benchmark_campaign_id") or "",
            "mode": _benchmark_value(job, "benchmark_mode") or "",
            "engine": job.tool or job.metadata.get("engine", ""),
            "status": job.status,
            "created": job.created_at,
        }
        for job in jobs
        if _benchmark_value(job, "benchmark_mode")
    ]
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(campaign_rows),
        pd.DataFrame(_rescoring_rows(jobs, pose_rmsd)),
        warnings,
    )


def render(*, embedded: bool = False) -> None:
    if embedded:
        st.subheader("Combined redocking / refolding results")
    else:
        st.title("Benchmark Redocking / Refolding Results")
    datasets = visible_dataset_jobs()
    requested = str(st.query_params.get("dataset_run_id") or "")
    if not datasets:
        st.info("No benchmark datasets have been imported.")
        return
    labels = {
        (
            f"{job.metadata.get('dataset_name')} — "
            f"{display_job_code(job.metadata.get('job_code'), job.run_id)}"
        ): job
        for job in datasets
    }
    default_index = next(
        (
            index
            for index, job in enumerate(labels.values())
            if job.run_id == requested
        ),
        0,
    )
    selected_label = st.selectbox(
        "Benchmark dataset",
        list(labels),
        index=default_index,
        key="benchmark_results_dataset",
    )
    dataset_job = labels[selected_label]
    dataset = load_benchmark_dataset(dataset_job)
    metrics, campaigns, rescoring, warnings = _collect(dataset_job)
    statistics = _dataset_statistics(
        str(dataset_job.run_dir),
        (dataset_job.run_dir / "benchmark_dataset.json").stat().st_mtime_ns,
    )

    overview_tab, artifacts_tab, metrics_tab, viewer_tab, lineage_tab, logs_tab = st.tabs(
        ["Overview", "Artifacts", "Metrics", "Viewer", "Lineage", "Logs"]
    )
    with overview_tab:
        summary = st.columns(7)
        summary[0].metric("Reference cases", dataset["case_count"])
        summary[1].metric("Rejected", dataset.get("rejected_case_count", 0))
        summary[2].metric(
            "Median ligand MW",
            (
                f"{statistics['molecular_weight'].median():.1f}"
                if not statistics.empty
                else "—"
            ),
        )
        summary[3].metric(
            "Median residues",
            (
                f"{statistics['receptor_residues'].median():.0f}"
                if not statistics.empty
                else "—"
            ),
        )
        summary[4].metric("Campaign jobs", len(campaigns))
        summary[5].metric(
            "Completed",
            int(campaigns["status"].eq("completed").sum())
            if not campaigns.empty
            else 0,
        )
        summary[6].metric("RMSD observations", len(metrics))
        st.write(
            f"**{dataset['dataset_name']}** · {dataset['profile']} · "
            f"{dataset['case_count']} canonical reference cases"
        )
        provenance = dataset.get("source_provenance") or {}
        if provenance:
            st.markdown("#### Source and provenance")
            st.json(provenance, expanded=False)
        st.markdown("#### Dataset composition")
        if statistics.empty:
            st.info("No canonical case statistics are available.")
        else:
            plots = st.columns(2)
            mw_bins = pd.cut(
                statistics["molecular_weight"],
                bins=[0, 200, 300, 400, 500, 700, 1000, float("inf")],
                right=False,
            ).value_counts().sort_index().rename_axis("range").reset_index(name="cases")
            mw_bins["range"] = mw_bins["range"].astype(str)
            plots[0].caption("Ligand molecular-weight distribution")
            plots[0].bar_chart(mw_bins, x="range", y="cases")
            residue_bins = pd.cut(
                statistics["receptor_residues"],
                bins=[0, 100, 200, 300, 500, 800, 1200, float("inf")],
                right=False,
            ).value_counts().sort_index().rename_axis("range").reset_index(name="cases")
            residue_bins["range"] = residue_bins["range"].astype(str)
            plots[1].caption("Receptor-residue distribution")
            plots[1].bar_chart(residue_bins, x="range", y="cases")
            st.dataframe(
                statistics,
                hide_index=True,
                width="stretch",
                height=420,
                column_config={
                    "molecular_weight": st.column_config.NumberColumn(
                        "MW (Da)", format="%.2f"
                    ),
                    "logp": st.column_config.NumberColumn("cLogP", format="%.2f"),
                    "qed": st.column_config.NumberColumn("QED", format="%.3f"),
                },
            )
            st.download_button(
                "Download dataset composition table",
                statistics.to_csv(index=False).encode(),
                file_name=f"{dataset['dataset_name']}-composition.csv",
                mime="text/csv",
            )
        report_path = dataset_job.run_dir / "artifacts" / "reports" / "import_report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            errors = report.get("errors") or []
            if errors:
                with st.expander(f"Rejected import cases ({len(errors)})"):
                    st.dataframe(pd.DataFrame(errors), hide_index=True, width="stretch")
        if metrics.empty:
            st.info(
                "No completed pose-producing benchmark results are available yet. "
                "Launch redocking or refolding from the Benchmark section."
            )
        else:
            aggregate = (
                metrics.groupby(["mode", "engine"], dropna=False)
                .agg(
                    observations=("ligand_rmsd_angstrom", "size"),
                    cases=("case_id", "nunique"),
                    median_rmsd=("ligand_rmsd_angstrom", "median"),
                    mean_rmsd=("ligand_rmsd_angstrom", "mean"),
                    recovery_1a=("ligand_rmsd_angstrom", lambda x: x.le(1.0).mean()),
                    recovery_2a=("ligand_rmsd_angstrom", lambda x: x.le(2.0).mean()),
                )
                .reset_index()
            )
            st.dataframe(aggregate, hide_index=True, width="stretch")
            chart = aggregate.pivot(
                index="engine", columns="mode", values="recovery_2a"
            ).fillna(0)
            st.markdown("#### Recovery at 2 Å")
            st.bar_chart(chart, horizontal=True)
        if warnings:
            with st.expander(
                f"RMSD evaluation warnings ({len(warnings)})", expanded=False
            ):
                st.dataframe(
                    pd.DataFrame({"Warning": warnings}),
                    hide_index=True,
                    width="stretch",
                )

    with metrics_tab:
        st.subheader("Ligand RMSD")
        if metrics.empty:
            st.info("No ligand RMSD measurements are available.")
        else:
            filters = st.columns(3)
            modes = filters[0].multiselect(
                "Modes",
                sorted(metrics["mode"].unique()),
                default=sorted(metrics["mode"].unique()),
                key="benchmark_results_modes",
            )
            engines = filters[1].multiselect(
                "Engines",
                sorted(metrics["engine"].unique()),
                default=sorted(metrics["engine"].unique()),
                key="benchmark_results_engines",
            )
            maximum = float(
                filters[2].number_input(
                    "Maximum displayed RMSD (Å)",
                    min_value=0.5,
                    max_value=100.0,
                    value=20.0,
                    step=0.5,
                    key="benchmark_results_max_rmsd",
                )
            )
            filtered = metrics.loc[
                metrics["mode"].isin(modes)
                & metrics["engine"].isin(engines)
                & metrics["ligand_rmsd_angstrom"].le(maximum)
            ].copy()
            if not filtered.empty:
                st.scatter_chart(
                    filtered,
                    x="case_id",
                    y="ligand_rmsd_angstrom",
                    color="engine",
                    size=None,
                )
            st.dataframe(
                filtered,
                hide_index=True,
                width="stretch",
                column_config={
                    "job": st.column_config.LinkColumn(
                        "Job result", display_text="Open"
                    ),
                    "ligand_rmsd_angstrom": st.column_config.NumberColumn(
                        "Ligand RMSD (Å)", format="%.3f"
                    ),
                    "protein_alignment_rmsd_angstrom": st.column_config.NumberColumn(
                        "Protein alignment RMSD (Å)", format="%.3f"
                    ),
                },
            )
            st.download_button(
                "Download ligand RMSD table",
                metrics.to_csv(index=False).encode(),
                file_name=f"{dataset['dataset_name']}-ligand-rmsd.csv",
                mime="text/csv",
            )
            st.caption(
                "Redocking is measured without ligand fitting in the fixed receptor "
                "frame. Refolding predictions are first rigidly aligned on matched "
                "protein Cα atoms; the ligand is never independently superposed."
            )

    with artifacts_tab:
        st.subheader("Combined result artifacts")
        result_jobs = [
            job
            for job in iter_job_records(runs_root())
            if str(_benchmark_value(job, "benchmark_dataset_run_id") or "")
            == dataset_job.run_id
        ]
        artifact_rows = []
        for job in result_jobs:
            for artifact in (
                job.artifact_manifest.artifacts if job.artifact_manifest else ()
            ):
                artifact_rows.append(
                    {
                        "Job": _job_url(job),
                        "Engine": job.tool or job.metadata.get("engine", ""),
                        "Type": artifact.artifact_type,
                        "Role": artifact.role,
                        "Path": artifact.path,
                        "Size": artifact.size_bytes,
                    }
                )
        if artifact_rows:
            st.dataframe(
                pd.DataFrame(artifact_rows),
                hide_index=True,
                width="stretch",
                height=420,
                column_config={
                    "Job": st.column_config.LinkColumn(
                        "Job result", display_text="Open"
                    )
                },
            )
        else:
            st.info("No result artifacts are available.")
        st.markdown("#### Rescoring outputs")
        if rescoring.empty:
            st.info("No completed benchmark rescoring outputs are available.")
        else:
            engine_filter = st.multiselect(
                "Rescoring engines",
                sorted(rescoring["engine"].unique()),
                default=sorted(rescoring["engine"].unique()),
                key="benchmark_results_rescoring_engines",
            )
            displayed = rescoring.loc[
                rescoring["engine"].isin(engine_filter)
            ].copy()
            st.dataframe(
                displayed,
                hide_index=True,
                width="stretch",
                column_config={
                    "job": st.column_config.LinkColumn(
                        "Job result", display_text="Open"
                    ),
                    "ligand_rmsd_angstrom": st.column_config.NumberColumn(
                        "Inherited ligand RMSD (Å)", format="%.3f"
                    ),
                    "maximum_coordinate_displacement_angstrom": (
                        st.column_config.NumberColumn(
                            "Coordinate displacement (Å)", format="%.6f"
                        )
                    ),
                },
            )
            st.download_button(
                "Download rescoring/RMSD table",
                rescoring.to_csv(index=False).encode(),
                file_name=f"{dataset['dataset_name']}-rescoring.csv",
                mime="text/csv",
            )
            st.caption(
                "RMSD is inherited by exact source docking run and pose rank. "
                "A missing value means the source pose was not part of a finalized "
                "canonical redocking metric table; it is never guessed."
            )

    with lineage_tab:
        if campaigns.empty:
            st.info("No benchmark campaigns have been launched for this dataset.")
        else:
            statuses = st.multiselect(
                "Status",
                sorted(campaigns["status"].unique()),
                default=sorted(campaigns["status"].unique()),
                key="benchmark_results_status",
            )
            st.dataframe(
                campaigns.loc[campaigns["status"].isin(statuses)],
                hide_index=True,
                width="stretch",
                column_config={
                    "job": st.column_config.LinkColumn(
                        "Job result", display_text="Open"
                    )
                },
            )

    with viewer_tab:
        case_frame = pd.DataFrame(dataset["cases"])
        if not metrics.empty:
            coverage = (
                metrics.groupby("case_id")
                .agg(
                    rmsd_observations=("ligand_rmsd_angstrom", "size"),
                    engines=("engine", "nunique"),
                    best_rmsd_angstrom=("ligand_rmsd_angstrom", "min"),
                )
                .reset_index()
            )
            case_frame = case_frame.merge(coverage, how="left", on="case_id")
        st.dataframe(case_frame, hide_index=True, width="stretch", height=560)
        cases = benchmark_cases(dataset_job)
        if cases:
            labels = {case.case_id: case for case in cases}
            selected_case = st.selectbox(
                "Reference complex viewer",
                list(labels),
                key=f"benchmark_results_case_viewer_{dataset_job.run_id}",
            )
            render_case_viewer(
                labels[selected_case],
                key=f"benchmark-results-{dataset_job.run_id}-{selected_case}",
            )

    with logs_tab:
        if warnings:
            st.warning(
                f"{len(warnings)} combined RMSD observation(s) could not be evaluated."
            )
            st.dataframe(
                pd.DataFrame({"Warning": warnings}),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No combined-results evaluation warnings were recorded.")


if __name__ == "__main__":
    render()
