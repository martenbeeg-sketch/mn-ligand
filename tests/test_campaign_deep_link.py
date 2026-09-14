from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd
import pytest

from mn_ligand.app.pages import campaign_comparison
from mn_ligand.app.pages.campaign_comparison import (
    CAMPAIGN_DATABASE_METRIC_COLUMNS,
    _campaign_database_metric_rows,
    _campaign_database_repeat_matrix,
    _campaign_metric_export_tables,
    _campaign_results_workbook,
    _campaign_results_csv_bundle,
    _input_pose_recovery_matrix,
    _native_metric_plots_zip,
    _select_gnina_ranking_rows,
    _target_engine_recovery_matrix,
    campaign_perspective_labels,
    classify_campaign_shape,
    _deep_link_campaign_scope,
    _target_ligand_launch_rows,
    target_engine_coverage,
    target_comparison_feature_table,
    target_consensus_summary,
    target_compound_matrix,
    target_pose_validation_summary,
)
from mn_ligand.core.provenance import compact_target_identifier


def test_input_pose_recovery_is_arranged_by_engine_and_attempt() -> None:
    recovery = pd.DataFrame(
        [
            {"Engine": "AlphaFold 3", "Input-pose RMSD (Å)": 0.21},
            {"Engine": "Boltz-2", "Input-pose RMSD (Å)": 0.34},
            {"Engine": "AlphaFold 3", "Input-pose RMSD (Å)": 0.58},
            {"Engine": "Boltz-2", "Input-pose RMSD (Å)": 0.37},
            {"Engine": "Boltz-2", "Input-pose RMSD (Å)": 0.69},
        ]
    )

    prepared, matrix = _input_pose_recovery_matrix(recovery)

    assert prepared["Attempt"].tolist() == [1, 1, 2, 2, 3]
    assert matrix.index.tolist() == ["AlphaFold 3", "Boltz-2"]
    assert matrix.columns.tolist() == ["Attempt 1", "Attempt 2", "Attempt 3"]
    assert matrix.loc["AlphaFold 3", "Attempt 1"] == pytest.approx(0.21)
    assert pd.isna(matrix.loc["AlphaFold 3", "Attempt 3"])
    assert matrix.loc["Boltz-2", "Attempt 3"] == pytest.approx(0.69)


def test_input_recovery_compares_each_target_across_engines() -> None:
    recovery = pd.DataFrame(
        [
            {
                "Prepared target": "4LNW · original",
                "Engine": "AlphaFold 3",
                "Input-pose RMSD (Å)": 0.2,
            },
            {
                "Prepared target": "4LNW · original",
                "Engine": "AlphaFold 3",
                "Input-pose RMSD (Å)": 0.4,
            },
            {
                "Prepared target": "4LNW · trimmed",
                "Engine": "AlphaFold 3",
                "Input-pose RMSD (Å)": 0.5,
            },
            {
                "Prepared target": "4LNW · trimmed",
                "Engine": "GNINA",
                "Input-pose RMSD (Å)": 0.7,
            },
        ]
    )

    means, annotations, summary = _target_engine_recovery_matrix(recovery)

    assert means.index.tolist() == ["4LNW · original", "4LNW · trimmed"]
    assert means.columns.tolist() == ["AlphaFold 3", "GNINA"]
    assert means.loc["4LNW · original", "AlphaFold 3"] == pytest.approx(0.3)
    assert "±" in annotations.loc["4LNW · original", "AlphaFold 3"]
    assert pd.isna(means.loc["4LNW · original", "GNINA"])
    assert int(summary["count"].sum()) == 4


def test_gnina_both_rankings_become_separate_pose_entries() -> None:
    frame = pd.DataFrame(
        [
            {
                "engine": "GNINA",
                "campaign_id": "campaign-1",
                "_prediction_label": "GNINA prediction",
                "_viewer_pose_index": 1,
                "cnn_ranked_pose_index": 2,
                "empirical_ranked_pose_index": 7,
            },
            {
                "engine": "AutoDock Vina",
                "campaign_id": "campaign-2",
                "_prediction_label": "Vina prediction",
                "_viewer_pose_index": 1,
            },
        ]
    )

    selected = _select_gnina_ranking_rows(frame, "Both rankings")

    assert selected["engine"].tolist() == [
        "AutoDock Vina",
        "GNINA · CNN-ranked",
        "GNINA · Vina-ranked",
    ]
    assert selected.loc[
        selected["engine"].eq("GNINA · CNN-ranked"),
        "_viewer_pose_index",
    ].item() == 2
    assert selected.loc[
        selected["engine"].eq("GNINA · Vina-ranked"),
        "_viewer_pose_index",
    ].item() == 7
    assert selected["campaign_id"].nunique() == 3

    cnn_only = _select_gnina_ranking_rows(frame, "CNN pose score")
    vina_only = _select_gnina_ranking_rows(
        frame,
        "Empirical / Vina score",
    )
    assert "GNINA · CNN-ranked" in cnn_only["engine"].tolist()
    assert "GNINA · Vina-ranked" in vina_only["engine"].tolist()
    assert cnn_only.loc[
        cnn_only["engine"].eq("GNINA · CNN-ranked"),
        "_viewer_pose_index",
    ].item() == 2
    assert vina_only.loc[
        vina_only["engine"].eq("GNINA · Vina-ranked"),
        "_viewer_pose_index",
    ].item() == 7


def test_structure_legend_distinguishes_vina_and_gnina_rankings() -> None:
    styles = campaign_comparison.STRUCTURE_ENGINE_STYLES

    assert styles["AutoDock Vina"][1] == "yellow"
    assert styles["GNINA · CNN-ranked"][1] == "orange"
    assert styles["GNINA · Vina-ranked"][1] == "red"
    assert len(
        {
            styles["AutoDock Vina"][1],
            styles["GNINA · CNN-ranked"][1],
            styles["GNINA · Vina-ranked"][1],
        }
    ) == 3


def test_engine_pose_summary_can_use_shape_or_centroid_metric() -> None:
    pairs = pd.DataFrame(
        [
            {
                "Engine A": "GNINA · Vina-ranked",
                "Engine B": "AutoDock Vina",
                "Fixed-frame RMSD (Å)": 2.2,
                "Shape Tanimoto distance": 0.18,
                "Centroid distance (Å)": 0.13,
            }
        ]
    )
    engines = ["GNINA · Vina-ranked", "AutoDock Vina"]

    shape, _, _ = campaign_comparison._engine_pose_rmsd_summary(
        pairs,
        engines,
        value_column="Shape Tanimoto distance",
    )
    centroid, _, _ = campaign_comparison._engine_pose_rmsd_summary(
        pairs,
        engines,
        value_column="Centroid distance (Å)",
    )

    assert shape.loc[
        "GNINA · Vina-ranked", "AutoDock Vina"
    ] == pytest.approx(0.18)
    assert centroid.loc[
        "GNINA · Vina-ranked", "AutoDock Vina"
    ] == pytest.approx(0.13)


def test_pose_comparison_color_scales_are_fixed_across_campaigns() -> None:
    rmsd = campaign_comparison._pose_comparison_scale(
        "Atom-mapped RMSD (Å)"
    )
    centroid = campaign_comparison._pose_comparison_scale(
        "Centroid displacement (Å)"
    )
    shape = campaign_comparison._pose_comparison_scale("Shape distance")

    assert rmsd["maximum"] == 4.0
    assert centroid["maximum"] == 4.0
    assert shape["maximum"] == 1.0
    assert "1–2 Å generally acceptable" in rmsd["guidance"]


def test_fixed_frame_rmsd_uses_input_ligand_symmetry() -> None:
    from rdkit import Chem

    ligand_smiles = (
        "N[C@@H](Cc1cc(I)c(Oc2ccc(O)c(I)c2)c(I)c1)C(=O)O"
    )
    converted_smiles = (
        "[NH3+][C@@H](CC1=CC(I)=C(OC2=CC=C(O)C(I)=C2)"
        "C(I)=C1)C(=O)[O-]"
    )
    left = Chem.MolFromSmiles(converted_smiles)
    right = Chem.Mol(left)
    left_conformer = Chem.Conformer(left.GetNumAtoms())
    right_conformer = Chem.Conformer(right.GetNumAtoms())
    for atom_index in range(left.GetNumAtoms()):
        point = (float(atom_index), 0.0, 0.0)
        left_conformer.SetAtomPosition(atom_index, point)
    authoritative = Chem.MolFromSmiles(ligand_smiles)
    parameters = Chem.AdjustQueryParameters()
    parameters.makeBondsGeneric = True
    parameters.adjustDegree = False
    parameters.adjustRingCount = False
    parameters.adjustRingChain = False
    query = Chem.AdjustQueryProperties(authoritative, parameters)
    mappings = left.GetSubstructMatches(
        query,
        uniquify=False,
        maxMatches=1000,
    )
    base_mapping = mappings[0]
    swapped_mapping = max(
        mappings[1:],
        key=lambda mapping: sum(
            left_index != right_index
            for left_index, right_index in zip(base_mapping, mapping)
        ),
    )
    for query_index, right_atom_index in enumerate(swapped_mapping):
        left_atom_index = base_mapping[query_index]
        right_conformer.SetAtomPosition(
            right_atom_index,
            (float(left_atom_index), 0.0, 0.0),
        )
    left.AddConformer(left_conformer)
    right.AddConformer(right_conformer)

    converted_graph_rmsd = campaign_comparison._symmetry_fixed_frame_rmsd(
        left,
        right,
    )
    authoritative_rmsd = campaign_comparison._symmetry_fixed_frame_rmsd(
        left,
        right,
        ligand_smiles,
    )

    assert converted_graph_rmsd > 0.2
    assert authoritative_rmsd == pytest.approx(0.0)


def test_campaign_shape_drives_adaptive_perspectives() -> None:
    jobs = pd.DataFrame(
        [
            {"target_run_id": "target-1", "campaign_purpose": "compound_dataset_docking_cofolding"},
            {"target_run_id": "target-2", "campaign_purpose": "compound_dataset_docking_cofolding"},
        ]
    )
    metrics = pd.DataFrame(
        [
            {"candidate_id": "CMP-1"},
            {"candidate_id": "CMP-2"},
        ]
    )

    shape = classify_campaign_shape(jobs, metrics)

    assert shape["shape"] == "multi_target_multi_compound"
    assert campaign_perspective_labels(shape) == [
        "Overview",
        "Scores & ranking",
        "Structural evidence",
        "Target × compound explorer",
        "Data & Analysis Sets",
    ]


def test_target_compound_matrix_preserves_both_physical_dimensions() -> None:
    metrics = pd.DataFrame(
        [
            {
                "campaign_id": f"campaign-{target}",
                "campaign": f"Campaign {target}",
                "launch_campaign_id": f"launch-{target}",
                "launch_campaign": f"Launch {target}",
                "target_run_id": target,
                "target": target.upper(),
                "candidate_id": compound,
                "engine": "GNINA",
                "dataset": "Dataset",
                "cnn_ranked_cnn_affinity": value,
            }
            for target in ("target-1", "target-2")
            for compound, value in (("CMP-1", 8.0), ("CMP-2", 6.0))
        ]
    )

    matrix = target_compound_matrix(metrics)

    assert len(matrix) == 4
    assert matrix["target_run_id"].nunique() == 2
    assert matrix["candidate_id"].nunique() == 2
    assert set(matrix.loc[matrix["candidate_id"].eq("CMP-1"), "Mean percentile"]) == {1.0}


def test_campaign_export_places_four_replicates_in_dedicated_columns() -> None:
    metrics = pd.DataFrame(
        [
            {
                "dataset": "Library A",
                "dataset_run_id": "dataset-1",
                "campaign": "Target 1 · GNINA",
                "campaign_id": "campaign-1",
                "target": "1ABC · MRR-PFX",
                "target_run_id": "target-1",
                "target_origin": "PDB → MODELLER repair → PDBFixer cleaning",
                "candidate_id": "CMP-1",
                "engine": "GNINA",
                "engine_run": "GNINA · ABC12",
                "replicate": replicate,
                "empirical_ranked_score_kcal_mol": value,
                "protein_ligand_entropy": 0.1 * replicate,
            }
            for replicate, value in enumerate((-8.0, -8.5, -9.0, -9.5), start=1)
        ]
    )

    long, wide = _campaign_metric_export_tables(metrics)

    row = wide.loc[
        wide["Parameter"].eq("empirical_ranked_score_kcal_mol")
    ].iloc[0]
    assert [row[f"Replicate {index}"] for index in range(1, 5)] == [
        -8.0,
        -8.5,
        -9.0,
        -9.5,
    ]
    assert row["Replicate count"] == 4
    assert row["Mean"] == -8.75
    assert set(long["Replicate"]) == {1, 2, 3, 4}
    assert "replicate" not in set(wide["Parameter"])
    entropy = wide.loc[wide["Parameter"].eq("protein_ligand_entropy")].iloc[0]
    assert [
        round(entropy[f"Replicate {index}"], 3) for index in range(1, 5)
    ] == [
        0.1,
        0.2,
        0.3,
        0.4,
    ]


def test_campaign_workbook_includes_pose_and_interaction_results() -> None:
    jobs = pd.DataFrame(
        [{"campaign_id": "campaign-1", "engine": "GNINA", "target": "1ABC"}]
    )
    metrics = pd.DataFrame(
        [
            {
                "campaign_id": "campaign-1",
                "target": "1ABC",
                "candidate_id": "CMP-1",
                "engine": "GNINA",
                "replicate": 1,
                "empirical_ranked_score_kcal_mol": -8.0,
            }
        ]
    )
    workbook = _campaign_results_workbook(
        jobs,
        metrics,
        pose_rows=pd.DataFrame([{"compound_id": "CMP-1", "passed_all": True}]),
        interaction_summaries=pd.DataFrame(
            [{"compound_id": "CMP-1", "hydrogen_bonds": 2}]
        ),
        interactions=pd.DataFrame(
            [{"compound_id": "CMP-1", "interaction_type": "hydrogen bond"}]
        ),
    )

    with pd.ExcelFile(BytesIO(workbook)) as excel_file:
        assert {
            "Metrics by replicate",
            "Metrics long",
            "Selected campaigns",
            "Pose validity",
            "Interaction summary",
            "Interactions",
        }.issubset(excel_file.sheet_names)


def test_campaign_csv_export_is_normalized_and_database_ready() -> None:
    jobs = pd.DataFrame(
        [{"campaign_id": "campaign-1", "engine": "GNINA", "target": "1ABC"}]
    )
    metrics = pd.DataFrame(
        [
            {
                "dataset": "Library A",
                "dataset_run_id": "dataset-1",
                "campaign": "Target 1 · GNINA",
                "campaign_id": "campaign-1",
                "target": "1ABC",
                "target_run_id": "target-1",
                "candidate_id": "CMP-1",
                "engine": "GNINA",
                "engine_run": "GNINA · ABC12",
                "replicate": 1,
                "empirical_ranked_score_kcal_mol": -8.0,
                "cnn_ranked_cnn_score": 0.72,
            }
        ]
    )

    normalized = _campaign_database_metric_rows(metrics)

    assert tuple(normalized.columns) == CAMPAIGN_DATABASE_METRIC_COLUMNS
    assert set(normalized["metric_name"]) == {
        "empirical_ranked_score_kcal_mol",
        "cnn_ranked_cnn_score",
    }
    assert normalized["observation_id"].is_unique
    assert set(normalized["selection_method"]) == {
        "cnn_ranked",
        "vina_ranked",
    }
    assert set(normalized["score_type"]) == {
        "cnn_score",
        "empirical_score_kcal_mol",
    }
    assert normalized.loc[
        normalized["metric_name"].eq("empirical_ranked_score_kcal_mol"),
        "metric_unit",
    ].item() == "kcal/mol"
    assert "empirical_ranked_score_kcal_mol" not in normalized.columns
    assert "cnn_ranked_cnn_score" not in normalized.columns

    bundle = _campaign_results_csv_bundle(jobs, metrics)
    with ZipFile(BytesIO(bundle)) as archive:
        assert {
            "metric_observations.csv",
            "metric_repeats_wide.csv",
            "campaign_runs.csv",
            "manifest.json",
            "README.txt",
        }.issubset(archive.namelist())
        exported = pd.read_csv(
            BytesIO(archive.read("metric_observations.csv"))
        )
    assert list(exported.columns) == list(CAMPAIGN_DATABASE_METRIC_COLUMNS)
    assert len(exported) == 2


def test_campaign_csv_export_preserves_triplicates_as_rows_and_columns() -> None:
    metrics = pd.DataFrame(
        [
            {
                "dataset": "Library A",
                "dataset_run_id": "dataset-1",
                "campaign": "Target 1 · Vina",
                "campaign_id": "campaign-1",
                "target": "1ABC",
                "target_run_id": "target-1",
                "candidate_id": "CMP-1",
                "engine": "AutoDock Vina",
                "engine_run": "AutoDock Vina · ABC12",
                "replicate": repeat,
                "best_score_kcal_mol": value,
                "pose_centroid_x": float(repeat),
                "box_center_distance_angstrom": float(repeat) / 10.0,
            }
            for repeat, value in enumerate((-8.0, -8.5, -9.0), start=1)
        ]
    )

    serial = _campaign_database_metric_rows(metrics)
    wide = _campaign_database_repeat_matrix(metrics)

    assert serial["replicate_number"].tolist() == [1, 2, 3]
    assert serial["value"].tolist() == [-8.0, -8.5, -9.0]
    assert set(serial["metric_name"]) == {"best_score_kcal_mol"}
    assert set(serial["selection_method"]) == {""}
    assert set(serial["score_type"]) == {""}
    assert len(wide) == 1
    assert wide.loc[0, ["repeat_1", "repeat_2", "repeat_3"]].tolist() == [
        -8.0,
        -8.5,
        -9.0,
    ]


def test_boltz_models_are_not_mislabeled_as_independent_repeats() -> None:
    metrics = pd.DataFrame(
        [
            {
                "dataset": "HY216",
                "dataset_run_id": "dataset-1",
                "campaign": "Target · Boltz-2",
                "campaign_id": "campaign-1",
                "target": "4LNW",
                "target_run_id": "target-1",
                "candidate_id": "HY-1",
                "engine": "Boltz-2",
                "engine_run": "Boltz-2 · TEST1",
                "replicate": repeat,
                "seed": 1000 + repeat,
                "model_id": f"HY-1_model_{model}",
                "confidence_score": repeat * 10.0 + model,
            }
            for repeat in (1, 2, 3)
            for model in range(5)
        ]
    )

    serial = _campaign_database_metric_rows(metrics)
    wide = _campaign_database_repeat_matrix(metrics)

    assert serial.groupby("replicate_number").size().to_dict() == {
        1: 1,
        2: 1,
        3: 1,
    }
    assert wide.loc[
        0, ["repeat_1", "repeat_2", "repeat_3"]
    ].tolist() == [10.0, 20.0, 30.0]
    assert list(wide.columns[-3:]) == ["repeat_1", "repeat_2", "repeat_3"]
    assert set(serial["model_id"]) == {"HY-1_model_0"}


def test_alphafold3_exports_best_sample_for_each_model_seed_repeat() -> None:
    metrics = pd.DataFrame(
        [
            {
                "dataset": "HY216",
                "dataset_run_id": "dataset-1",
                "campaign": "Target · AlphaFold 3",
                "campaign_id": "campaign-1",
                "target": "4LNW",
                "target_run_id": "target-1",
                "candidate_id": "HY-1",
                "engine": "AlphaFold 3",
                "engine_run": "AlphaFold 3 · TEST1",
                "model_seed": seed,
                "sample": sample,
                "prediction_id": f"seed-{seed}_sample-{sample}",
                "ranking_score": float(sample) + repeat / 10.0,
                "iptm": float(sample) / 10.0,
            }
            for repeat, seed in enumerate((1001, 1002, 1003), start=1)
            for sample in range(5)
        ]
    )

    serial = _campaign_database_metric_rows(metrics)
    wide = _campaign_database_repeat_matrix(metrics)
    ranking_serial = serial.loc[serial["metric_name"].eq("ranking_score")]
    ranking_wide = wide.loc[wide["metric_name"].eq("ranking_score")].iloc[0]

    assert ranking_serial["replicate_number"].tolist() == [1, 2, 3]
    assert set(ranking_serial["sample"]) == {4}
    assert ranking_wide[["repeat_1", "repeat_2", "repeat_3"]].tolist() == [
        4.1,
        4.2,
        4.3,
    ]
    assert list(wide.columns[-3:]) == ["repeat_1", "repeat_2", "repeat_3"]


def test_native_metric_plot_export_contains_png_data_and_manifest() -> None:
    summary = pd.DataFrame(
        {
            "campaign": ["4LNW · preparation A", "4LNW · preparation B"],
            "candidate_id": ["T3", "T3"],
            "Mean": [0.95, 0.91],
            "Sample SD": [0.01, 0.02],
            "Attempts": [3, 3],
        }
    )

    payload = _native_metric_plots_zip(
        [
            {
                "engine": "AlphaFold 3",
                "metric": "iptm",
                "metric_label": "ipTM",
                "higher_is_better": True,
                "compare_targets": True,
                "summary": summary,
            }
        ],
        repetition_mode="All repetitions",
    )

    with ZipFile(BytesIO(payload)) as archive:
        names = archive.namelist()
        png_name = next(name for name in names if name.endswith(".png"))
        assert archive.read(png_name).startswith(b"\x89PNG\r\n\x1a\n")
        assert any(name.endswith(".csv") for name in names)
        assert "manifest.json" in names
        assert "README.txt" in names


def test_campaign_deep_link_hard_scopes_target_and_launch() -> None:
    campaigns = pd.DataFrame(
        [
            {
                "campaign_id": "run-1",
                "target_run_id": "target-1",
                "launch_campaign_id": "launch-1",
                "engine": "GNINA",
            },
            {
                "campaign_id": "run-2",
                "target_run_id": "target-1",
                "launch_campaign_id": "launch-1",
                "engine": "Boltz-2",
            },
            {
                "campaign_id": "run-3",
                "target_run_id": "target-1",
                "launch_campaign_id": "launch-2",
                "engine": "AlphaFold 3",
            },
            {
                "campaign_id": "run-4",
                "target_run_id": "target-2",
                "launch_campaign_id": "launch-1",
                "engine": "GNINA",
            },
        ]
    )

    scoped = _deep_link_campaign_scope(
        campaigns,
        target_run_id="target-1",
        launch_campaign_id="launch-1",
    )

    assert scoped["campaign_id"].tolist() == ["run-1", "run-2"]
    assert scoped["engine"].tolist() == ["GNINA", "Boltz-2"]


def test_campaign_deep_link_hard_scopes_campaign_purpose() -> None:
    campaigns = pd.DataFrame(
        [
            {
                "campaign_id": "target-ligand",
                "target_run_id": "target-1",
                "launch_campaign_id": "launch-1",
                "campaign_purpose": "target_ligand_redocking_refolding",
            },
            {
                "campaign_id": "imported-library",
                "target_run_id": "target-1",
                "launch_campaign_id": "launch-1",
                "campaign_purpose": "compound_dataset_docking_cofolding",
            },
        ]
    )

    scoped = _deep_link_campaign_scope(
        campaigns,
        target_run_id="target-1",
        launch_campaign_id="launch-1",
        campaign_purpose="target_ligand_redocking_refolding",
    )

    assert scoped["campaign_id"].tolist() == ["target-ligand"]


def test_target_ligand_launch_rows_group_engines_by_prepared_target(
    monkeypatch,
) -> None:
    inventory = [
        SimpleNamespace(
            choice=SimpleNamespace(
                job=SimpleNamespace(run_id="target-1")
            ),
            row={
                "Job": "./job-results?label=TGT01",
                "Target": "4LNW",
                "Tool": "chain-aware terminal trimmer",
                "Origin": "PDB → Target trimming",
                "Last step": "Target trimming",
                "Residues": 246,
            },
        ),
        SimpleNamespace(
            choice=SimpleNamespace(
                job=SimpleNamespace(run_id="target-2")
            ),
            row={
                "Job": "./job-results?label=TGT02",
                "Target": "4LNW",
                "Tool": "MODELLER",
                "Origin": "PDB → Target trimming → MODELLER repair",
                "Last step": "MODELLER repair",
                "Residues": 250,
            },
        ),
    ]
    monkeypatch.setattr(
        campaign_comparison,
        "target_inventory",
        lambda **_kwargs: inventory,
    )
    campaigns = pd.DataFrame(
        [
            {
                "campaign_id": "run-1",
                "target_run_id": "target-1",
                "target": "4LNW · PFX-OMM-TRM · TGT01",
                "launch_campaign_id": "launch-1",
                "launch_campaign": "4lnw_trimmed",
                "engine": "GNINA",
                "created_at": "2026-07-30T09:00:00+00:00",
            },
            {
                "campaign_id": "run-2",
                "target_run_id": "target-1",
                "target": "4LNW · PFX-OMM-TRM · TGT01",
                "launch_campaign_id": "launch-1",
                "launch_campaign": "4lnw_trimmed",
                "engine": "Boltz-2",
                "created_at": "2026-07-30T09:01:00+00:00",
            },
            {
                "campaign_id": "run-3",
                "target_run_id": "target-2",
                "target": "4LNW · PFX-OMM-CTE · TGT02",
                "launch_campaign_id": "launch-2",
                "launch_campaign": "4lnw_repaired",
                "engine": "AlphaFold 3",
                "created_at": "2026-07-30T10:00:00+00:00",
            },
        ]
    )

    rows = _target_ligand_launch_rows(campaigns)

    assert rows["Campaign"].tolist() == [
        "4lnw_trimmed",
        "4lnw_repaired",
    ]
    assert rows["Target"].tolist() == [
        "4LNW · PFX-OMM-TRM · TGT01",
        "4LNW · PFX-OMM-CTE · TGT02",
    ]
    assert rows["Engines"].tolist() == [
        "Boltz-2, GNINA",
        "AlphaFold 3",
    ]
    assert rows["Last step"].tolist() == [
        "Target trimming",
        "MODELLER repair",
    ]
    assert rows["_selection_id"].tolist() == [
        "launch-1::target-1",
        "launch-2::target-2",
    ]


def test_target_launch_rows_keep_targets_separate_with_shared_launch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        campaign_comparison,
        "target_inventory",
        lambda **_kwargs: [],
    )
    campaigns = pd.DataFrame(
        [
            {
                "campaign_id": "run-1",
                "target_run_id": "target-1",
                "target": "target-1.pdb",
                "launch_campaign_id": "shared-launch",
                "launch_campaign": "Multi-target launch",
                "engine": "GNINA",
                "created_at": "2026-08-01T09:00:00+00:00",
            },
            {
                "campaign_id": "run-2",
                "target_run_id": "target-2",
                "target": "target-2.pdb",
                "launch_campaign_id": "shared-launch",
                "launch_campaign": "Multi-target launch",
                "engine": "GNINA",
                "created_at": "2026-08-01T09:01:00+00:00",
            },
        ]
    )

    rows = _target_ligand_launch_rows(campaigns)

    assert rows["_selection_id"].tolist() == [
        "shared-launch::target-1",
        "shared-launch::target-2",
    ]


def test_target_ligand_render_uses_table_and_engine_only_controls() -> None:
    source = Path(campaign_comparison.__file__).read_text()

    assert "_render_target_ligand_launch_selector(" in source
    assert '"Search prepared targets",' in source
    assert '"Edit prepared-target scope",' in source
    assert '"Campaign",' in source
    assert "Edit or create Analysis Sets from" in source
    assert '"Order targets by",' in source
    assert '"Order direction",' in source
    assert '"campaign_compare_target_launch_table::"' in source
    assert 'on_select="rerun"' in source
    assert 'selection_mode="multi-row"' in source
    assert 'selection_default={"selection": {"rows": default_rows}}' in source
    assert 'table_event.selection.rows' in source
    assert 'st.data_editor(\n            visible' not in source
    assert "selected_engines = st.multiselect(" in source
    assert '"Edit Analysis Set filters",' in source
    assert "selected_campaigns = selectable[\"campaign_id\"].tolist()" in source


def test_target_scope_uses_native_row_selection(monkeypatch) -> None:
    rows = pd.DataFrame(
        [
            {
                "Campaign": "Campaign",
                "Target job": "",
                "Target": f"Target {index}",
                "PDB / origin": "4LNW",
                "Tool": "Preparation",
                "Origin": "PDB",
                "Last step": "Minimization",
                "Residues": 250,
                "Engines": "GNINA",
                "Created": pd.Timestamp("2026-08-01", tz="UTC"),
                "_target_run_id": f"target-{index}",
                "_launch_campaign_id": "launch-1",
                "_selection_id": f"launch-1::target-{index}",
                "_compound_signature": "canonical-t3",
                "_campaign_purpose": "target_ligand_redocking_refolding",
            }
            for index in range(1, 4)
        ]
    )
    # Simulate the empty scope left behind by the former sort/selection bug.
    session_state: dict[str, object] = {
        "campaign_compare_target_launch_ids": []
    }
    reported_rows = [0, 1]
    expected_default_rows = [[0, 1, 2], [1, 2]]
    order_directions = ["Ascending", "Descending"]

    def dataframe(_frame, **kwargs):
        assert kwargs["on_select"] == "rerun"
        assert kwargs["selection_mode"] == "multi-row"
        assert kwargs["selection_default"] == {
            "selection": {"rows": expected_default_rows.pop(0)}
        }
        return SimpleNamespace(
            selection=SimpleNamespace(rows=list(reported_rows))
        )

    monkeypatch.setattr(
        campaign_comparison,
        "_target_ligand_launch_rows",
        lambda _campaigns: rows,
    )
    monkeypatch.setattr(
        campaign_comparison,
        "st",
        SimpleNamespace(
            session_state=session_state,
            selectbox=lambda _label, options, **_kwargs: options[0],
            segmented_control=(
                lambda *_args, **_kwargs: order_directions.pop(0)
            ),
            toggle=lambda *_args, **_kwargs: True,
            text_input=lambda *_args, **_kwargs: "",
            caption=lambda *_args, **_kwargs: None,
            columns=lambda *_args, **_kwargs: [
                SimpleNamespace(button=lambda *_args, **_kwargs: False)
                for _ in range(3)
            ],
            dataframe=dataframe,
            column_config=SimpleNamespace(
                LinkColumn=lambda *_args, **_kwargs: None,
                NumberColumn=lambda *_args, **_kwargs: None,
                DatetimeColumn=lambda *_args, **_kwargs: None,
            ),
        ),
    )

    targets, launches, selection_ids = (
        campaign_comparison._render_target_ligand_launch_selector(
            pd.DataFrame(),
            requested_target_id="",
            requested_launch_id="",
            collection_selection={},
        )
    )

    assert targets == ["target-1", "target-2"]
    assert launches == ["launch-1"]
    assert selection_ids == ["launch-1::target-1", "launch-1::target-2"]

    # Sorting currently emits an empty selection from Streamlit. It must not
    # erase the persisted comparison scope.
    reported_rows.clear()
    targets, launches, selection_ids = (
        campaign_comparison._render_target_ligand_launch_selector(
            pd.DataFrame(),
            requested_target_id="",
            requested_launch_id="",
            collection_selection={},
        )
    )
    assert targets == ["target-1", "target-2"]
    assert launches == ["launch-1"]
    assert selection_ids == ["launch-1::target-1", "launch-1::target-2"]


def test_analysis_set_scope_does_not_render_campaign_selector(monkeypatch) -> None:
    rows = pd.DataFrame(
        [
            {
                "Campaign": "Campaign A",
                "Target job": "",
                "Target": "4LNW · preparation A",
                "PDB / origin": "4LNW",
                "Tool": "Preparation",
                "Origin": "PDB",
                "Last step": "Minimization",
                "Residues": 250,
                "Engines": "GNINA",
                "Created": pd.Timestamp("2026-08-01", tz="UTC"),
                "_target_run_id": "target-1",
                "_launch_campaign_id": "launch-1",
                "_selection_id": "launch-1::target-1",
                "_compound_signature": "canonical-t3",
                "_campaign_purpose": "target_ligand_redocking_refolding",
            }
        ]
    )
    monkeypatch.setattr(
        campaign_comparison,
        "_target_ligand_launch_rows",
        lambda _campaigns: rows,
    )

    def unexpected_selectbox(*_args, **_kwargs):
        raise AssertionError("Analysis Set scope must not show Campaign")

    monkeypatch.setattr(
        campaign_comparison,
        "st",
        SimpleNamespace(
            session_state={},
            selectbox=unexpected_selectbox,
            info=lambda *_args, **_kwargs: None,
            toggle=lambda *_args, **_kwargs: False,
            caption=lambda *_args, **_kwargs: None,
        ),
    )

    targets, launches, selection_ids = (
        campaign_comparison._render_target_ligand_launch_selector(
            pd.DataFrame(),
            requested_target_id="",
            requested_launch_id="",
            collection_selection={
                "launch_campaign_ids": ["launch-1"],
                "target_run_ids": ["target-1"],
                "target_launch_pairs": ["launch-1::target-1"],
            },
        )
    )

    assert targets == ["target-1"]
    assert launches == ["launch-1"]
    assert selection_ids == ["launch-1::target-1"]


def test_analysis_set_only_offers_compatible_campaigns() -> None:
    launch_rows = pd.DataFrame(
        [
            {
                "_launch_campaign_id": "redock-a",
                "_compound_signature": "canonical-t3",
                "_campaign_purpose": "target_ligand_redocking_refolding",
            },
            {
                "_launch_campaign_id": "redock-a",
                "_compound_signature": "canonical-other",
                "_campaign_purpose": "target_ligand_redocking_refolding",
            },
            {
                "_launch_campaign_id": "redock-b",
                "_compound_signature": "canonical-t3",
                "_campaign_purpose": "target_ligand_redocking_refolding",
            },
            {
                "_launch_campaign_id": "md-a",
                "_compound_signature": "canonical-t3",
                "_campaign_purpose": "molecular_dynamics",
            },
            {
                "_launch_campaign_id": "redock-other-ligand",
                "_compound_signature": "canonical-unrelated",
                "_campaign_purpose": "target_ligand_redocking_refolding",
            },
        ]
    )

    assert campaign_comparison._compatible_target_ligand_launches(
        launch_rows,
        "redock-a",
    ) == ["redock-a", "redock-b"]


def test_viewer_offers_prepared_target_matrix() -> None:
    source = Path(campaign_comparison.__file__).read_text()

    assert (
        '("Single compound", "Target matrix", "Compound matrix")'
        in source
    )
    assert 'if layout == "Target matrix":' in source
    assert "_render_target_matrix(structural)" in source
    assert "Structures included in both views" in source
    assert '"Prepared targets"' in source
    assert '"Engines"' in source
    assert '"Compounds"' in source
    assert '"Engines shown in 3D"' in source
    assert '"Engines shown in RMSD and 3D"' not in source
    assert 'campaign_linked_3d_focus_engines' not in source
    assert "The RMSD matrices remain unchanged." in source
    assert '"All repetitions"' in source
    assert 'forced_layout="Target matrix"' in source
    assert 'st.tabs(' in source
    assert '"Target grid columns"' in source
    assert '"#### Target-panel key"' in source
    assert '"#### Structure color legend"' in source
    assert "Engine identity is encoded by ligand carbon color" in source
    assert "panel_titles=[" in source
    assert "panel_width=1650" in source
    assert '["target_run_id", "target", "launch_campaign"]' in source
    assert '"Structure availability details"' in source
    assert 'expanded=missing_cells > 0' in source
    assert 'coverage.pivot(' in source
    assert (
        "Each panel is one prepared target with the same compound"
        in source
    )


def test_target_matrix_includes_all_focused_targets_without_second_selector() -> None:
    source = Path(campaign_comparison.__file__).read_text()
    start = source.index("def _render_target_matrix")
    end = source.index("def _render_compound_matrix", start)
    target_matrix_source = source[start:end]

    assert "selected_targets = target_ids" in target_matrix_source
    assert '"Focus targets (from comparison scope)"' not in target_matrix_source
    assert '"Panel display settings"' in target_matrix_source


def test_3d_engine_filter_is_applied_to_structure_viewer() -> None:
    source = Path(campaign_comparison.__file__).read_text()
    start = source.index("def _render_target_viewer_context")
    end = source.index("def _render_rescoring_comparison", start)
    viewer_context_source = source[start:end]

    assert viewer_context_source.count(
        "_render_structure_comparison(\n            structural,"
    ) == 2
    assert "Engines shown in 3D" in viewer_context_source
    assert "The RMSD matrices remain unchanged." in viewer_context_source
    assert viewer_context_source.count("_select_gnina_ranking_rows(") == 2
    assert "_campaign_3d_engine_selector_options" in viewer_context_source


def test_target_feature_table_aggregates_replicates_by_prepared_target() -> None:
    rows = []
    for target_index, score in enumerate((-8.0, -7.0, -6.0), start=1):
        for replicate, offset in enumerate((-0.2, 0.2), start=1):
            rows.append(
                {
                    "campaign_id": f"vina-{target_index}-{replicate}",
                    "launch_campaign_id": f"launch-{target_index}",
                    "launch_campaign": f"prep-{target_index}",
                    "target_run_id": f"target-{target_index}",
                    "target": f"4LNW · PFX-OMM-TRM · TGT0{target_index}",
                    "engine": "AutoDock Vina",
                    "candidate_id": "T3",
                    "best_score_kcal_mol": score + offset,
                }
            )
    table, metadata, summaries, labels, defaults = (
        target_comparison_feature_table(pd.DataFrame(rows))
    )

    feature = "AutoDock Vina::best_score_kcal_mol"
    assert table.index.tolist() == [
        "launch-1::target-1",
        "launch-2::target-2",
        "launch-3::target-3",
    ]
    assert table[feature].tolist() == [8.0, 7.0, 6.0]
    assert summaries["Attempts"].tolist() == [2, 2, 2]
    assert metadata["target_preparation"].tolist() == [
        "4LNW · PFX-OMM-TRM · TGT01",
        "4LNW · PFX-OMM-TRM · TGT02",
        "4LNW · PFX-OMM-TRM · TGT03",
    ]
    assert "favorable ↑" in labels[feature]
    assert defaults == [feature]


def test_target_feature_table_keeps_shared_launch_targets_separate() -> None:
    frame = pd.DataFrame(
        [
            {
                "campaign_id": "vina-1",
                "launch_campaign_id": "shared-launch",
                "launch_campaign": "Multi-target launch",
                "target_run_id": "target-1",
                "target": "2H79 · MRR-PFX-OMM-TRM-ORI · TGT01",
                "engine": "AutoDock Vina",
                "candidate_id": "LIG-1",
                "best_score_kcal_mol": -8.0,
            },
            {
                "campaign_id": "vina-2",
                "launch_campaign_id": "shared-launch",
                "launch_campaign": "Multi-target launch",
                "target_run_id": "target-2",
                "target": "4LNW · MRM-PFX-OMM-ORI · TGT02",
                "engine": "AutoDock Vina",
                "candidate_id": "LIG-2",
                "best_score_kcal_mol": -7.0,
            },
            {
                "campaign_id": "vina-3",
                "launch_campaign_id": "shared-launch",
                "launch_campaign": "Multi-target launch",
                "target_run_id": "target-3",
                "target": "1N46 · PFX-OMM-ORI · TGT03",
                "engine": "AutoDock Vina",
                "candidate_id": "LIG-3",
                "best_score_kcal_mol": -6.0,
            },
        ]
    )

    table, metadata, _, _, _ = target_comparison_feature_table(frame)

    assert table.index.tolist() == [
        "shared-launch::target-1",
        "shared-launch::target-2",
        "shared-launch::target-3",
    ]
    assert metadata["target_preparation"].tolist() == [
        "2H79 · MRR-PFX-OMM-TRM-ORI · TGT01",
        "4LNW · MRM-PFX-OMM-ORI · TGT02",
        "1N46 · PFX-OMM-ORI · TGT03",
    ]


def test_target_engine_coverage_reports_repetitions_per_target() -> None:
    coverage = target_engine_coverage(
        pd.DataFrame(
            [
                {
                    "campaign_id": "run-1",
                    "target_run_id": "target-1",
                    "target": "2H79 · MRR-PFX-OMM · TGT01",
                    "engine": "GNINA",
                    "configured_repeats": 3,
                    "completed_repeats": 3,
                },
                {
                    "campaign_id": "run-2",
                    "target_run_id": "target-2",
                    "target": "4LNW · MRM-PFX-OMM · TGT02",
                    "engine": "GNINA",
                    "configured_repeats": 3,
                    "completed_repeats": 2,
                },
                {
                    "campaign_id": "run-3",
                    "target_run_id": "target-3",
                    "target": "6KKB · PFX-OMM · TGT03",
                    "engine": "AlphaFold 3",
                    "configured_repeats": 1,
                    "completed_repeats": 1,
                },
            ]
        )
    )

    assert coverage["repeat_coverage"].tolist() == ["3/3", "2/3", "1/1"]
    assert coverage["comparison_coverage"].tolist() == ["3/3", "2/3", "1/3"]
    assert coverage["target_preparation"].tolist() == [
        "2H79 · MRR-PFX-OMM · TGT01",
        "4LNW · MRM-PFX-OMM · TGT02",
        "6KKB · PFX-OMM · TGT03",
    ]


def test_compact_target_identifier_uses_origin_steps_and_job_code() -> None:
    identifier = compact_target_identifier(
        run_id="target-run",
        metadata={
            "pdb_id": "2h79",
            "job_code": "776ED",
            "modification_history": [
                {"kind": "modeller_residue_repair"},
                {"kind": "pdbfixer_cleaning"},
                {"kind": "openmm_minimization"},
                {"kind": "target_trimming"},
                {"kind": "target_orientation"},
            ],
        },
    )

    assert identifier == "2H79 · MRR-PFX-OMM-TRM-ORI · 776ED"


def test_target_consensus_ranks_preparations_without_counting_replicates() -> None:
    features = pd.DataFrame(
        {
            "engine-a": [3.0, 2.0, 1.0],
            "engine-b": [30.0, 20.0, 10.0],
        },
        index=["prep-a", "prep-b", "prep-c"],
    )

    combined, long = target_consensus_summary(
        features,
        ["engine-a", "engine-b"],
    )

    assert combined["target_comparison_id"].tolist() == [
        "prep-a",
        "prep-b",
        "prep-c",
    ]
    assert combined["Mean percentile"].tolist() == [1.0, 2 / 3, 1 / 3]
    assert long.groupby("target_comparison_id").size().tolist() == [2, 2, 2]


def test_target_pose_validation_summary_keeps_targets_and_engines_separate() -> None:
    jobs = pd.DataFrame(
        [
            {
                "campaign_id": "source-a",
                "launch_campaign_id": "shared-launch",
                "launch_campaign": "Multi-target launch",
                "target_run_id": "target-a",
                "target": "4LNW · PFX-OMM · TGT01",
            },
            {
                "campaign_id": "source-b",
                "launch_campaign_id": "shared-launch",
                "launch_campaign": "Multi-target launch",
                "target_run_id": "target-b",
                "target": "4LNW · PFX-OMM-TRM · TGT02",
            },
        ]
    )
    poses = pd.DataFrame(
        [
            {
                "source_run_id": "source-a",
                "engine": "AutoDock Vina",
                "passed_all": True,
            },
            {
                "source_run_id": "source-a",
                "engine": "AutoDock Vina",
                "passed_all": False,
            },
            {
                "source_run_id": "source-b",
                "engine": "AutoDock Vina",
                "passed_all": True,
            },
        ]
    )

    summary = target_pose_validation_summary(poses, jobs)

    assert summary["target_preparation"].tolist() == [
        "4LNW · PFX-OMM · TGT01",
        "4LNW · PFX-OMM-TRM · TGT02",
    ]
    assert summary["target_comparison_id"].tolist() == [
        "shared-launch::target-a",
        "shared-launch::target-b",
    ]
    assert summary["pose_pass_rate"].tolist() == [0.5, 1.0]
    assert summary["assessed_poses"].tolist() == [2, 1]
