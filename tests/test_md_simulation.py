from __future__ import annotations

from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mn_ligand.app.pages.job_results import (
    _MD_PLOT_LABELS,
    _MD_REPLICA_SPLIT_PLOTS,
    _available_md_plot_ids,
    _md_display_residue_labels,
    _md_ordered_plot_ids,
    _md_plot_figure,
    _md_plot_data_tables,
    _md_plot_data_workbook,
    _md_plot_export_bundle,
    _md_separate_plot_specs,
)
from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.worker import WorkerConfig, run_worker_once
from mn_ligand.core.workflows import WorkflowRecord
from mn_ligand.ligandx.services.md.config import MDOptimizationConfig
from mn_ligand.ligandx.services.md.workflow.equilibration_runner import (
    fit_density_plateau,
)
from mn_ligand.ligandx.services.md.workflow.system_builder import (
    SolvatedSystemBuilder,
)
from mn_ligand.workflows.bound_ligand_md import (
    _amber_partition_atom_counts,
    _parse_amber_final_results,
    _source_ligand_key,
    recompute_mmgbsa,
)
from mn_ligand.workflows.md_simulation import (
    EXACT_CONTINUATION,
    INDEPENDENT_REPLICA,
    SYSTEM_PARAMETER_KEYS,
    advance_md_workflow,
    compatibility_contract,
    compatibility_differences,
    compatibility_fingerprint,
    continuation_mode,
    create_mmgbsa_analysis_job,
    create_md_simulation,
    create_md_simulation_from_template,
    finalize_md_job,
    finalize_mmgbsa_analysis_job,
    replica_seed,
    _require_scientifically_valid_workflow,
    _normalized_secondary_structure,
    _openmm_production_checkpoint,
    _production_performance,
    _required_interaction_consensus,
    _stage_openmm_continuation_inputs,
    _validate_roe_preparation_result,
    visible_md_workflow_rows,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_minimal_amber_prmtop(path: Path, natom: int) -> None:
    path.write_text(
        "%VERSION  VERSION_STAMP = V0001.000\n"
        "%FLAG POINTERS\n"
        "%FORMAT(10I8)\n"
        f"{natom:8d}\n"
    )


def test_amber_topology_uses_immutable_source_ligand_key() -> None:
    selected = {
        "key": "LIG|X|1|_",
        "resname": "LIG",
        "original_key": "LG1|X|1|_",
        "original_resname": "LG1",
    }

    assert _source_ligand_key(selected) == "LG1|X|1|_"


def test_amber_partition_requires_complex_to_equal_receptor_plus_ligand(
    tmp_path: Path,
) -> None:
    complex_prmtop = tmp_path / "com.prmtop"
    receptor_prmtop = tmp_path / "rec.prmtop"
    ligand_prmtop = tmp_path / "lig.prmtop"
    _write_minimal_amber_prmtop(complex_prmtop, 3950)
    _write_minimal_amber_prmtop(receptor_prmtop, 3903)
    _write_minimal_amber_prmtop(ligand_prmtop, 47)

    balanced = _amber_partition_atom_counts(
        complex_prmtop,
        receptor_prmtop,
        ligand_prmtop,
    )
    assert balanced == {
        "complex_atoms": 3950,
        "receptor_atoms": 3903,
        "ligand_atoms": 47,
        "balanced": True,
    }

    _write_minimal_amber_prmtop(complex_prmtop, 3903)
    assert _amber_partition_atom_counts(
        complex_prmtop,
        receptor_prmtop,
        ligand_prmtop,
    )["balanced"] is False


def test_scientifically_invalid_md_workflow_cannot_be_reused_or_extended() -> None:
    workflow = SimpleNamespace(
        parameters={
            "scientifically_invalid": True,
            "scientific_invalid_reason": "Ligand was omitted from the topology",
        }
    )

    with pytest.raises(ValueError, match="cannot be reused or extended"):
        _require_scientifically_valid_workflow(workflow)


def test_md_workflow_history_is_hidden_by_default() -> None:
    rows = [
        {"job": "current", "status": "running"},
        {"job": "done", "status": "completed"},
        {"job": "failed", "status": "failed"},
        {"job": "blocked", "status": "blocked"},
        {"job": "cancelled", "status": "cancelled"},
        {"job": "superseded", "status": "superseded"},
    ]

    assert [
        row["job"]
        for row in visible_md_workflow_rows(rows, show_history=False)
    ] == ["current", "done"]
    assert visible_md_workflow_rows(rows, show_history=True) == rows


def test_openmm_checkpoint_resolves_legacy_nested_run_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "legacy-openmm-run"
    checkpoint = run_dir / run_dir.name / "production_checkpoint.chk"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert (
        _openmm_production_checkpoint(run_dir, run_dir.name, None)
        == checkpoint
    )


def test_openmm_continuation_snapshots_legacy_output_inputs(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    child_dir = tmp_path / "child"
    source_dir.mkdir()
    child_dir.mkdir()
    complex_pdb = source_dir / "final_input_protein_refined.pdb"
    ligand_sdf = source_dir / "source_ligand_refined.sdf"
    complex_pdb.write_text("ATOM\n")
    ligand_sdf.write_text("ligand\n")
    payload = {
        "input_complex_pdb_path": "/output/final_input_protein_refined.pdb",
        "prepared_complex_path": "/output/final_input_protein_refined.pdb",
        "ligand_refined_sdf_path": "/output/source_ligand_refined.sdf",
    }

    _stage_openmm_continuation_inputs(source_dir, child_dir, payload)

    assert (child_dir / complex_pdb.name).read_text() == "ATOM\n"
    assert (child_dir / ligand_sdf.name).read_text() == "ligand\n"
    assert payload["input_complex_pdb_path"] == "/output/final_input_protein_refined.pdb"
    assert payload["prepared_complex_path"] == "/output/final_input_protein_refined.pdb"
    assert payload["ligand_refined_sdf_path"] == "/output/source_ligand_refined.sdf"


def test_single_chain_plot_labels_hide_redundant_chain_suffix() -> None:
    assert _md_display_residue_labels(
        ["SER277 · chain A", "HIS381 · chain A"]
    ) == ["SER277", "HIS381"]
    assert _md_display_residue_labels(
        ["SER277 · chain A", "SER277 · chain B"]
    ) == ["SER277 · chain A", "SER277 · chain B"]


def test_separate_plot_order_places_replica_panels_before_summaries() -> None:
    selected = [
        "site_retention",
        "backbone_rmsd",
        "endpoint_energy",
        "hbond_occupancy",
    ]

    assert _md_ordered_plot_ids(selected, "Separate") == [
        "backbone_rmsd",
        "hbond_occupancy",
        "site_retention",
        "endpoint_energy",
    ]
    assert _md_ordered_plot_ids(selected, "Consensus") == selected


def test_separate_plot_specs_support_parallel_and_serial_repeats() -> None:
    selected = ["backbone_rmsd", "ligand_rmsd", "site_retention"]

    assert _md_separate_plot_specs(selected, [1, 3], "Parallel") == [
        ("backbone_rmsd", 1, False),
        ("backbone_rmsd", 3, False),
        ("ligand_rmsd", 1, False),
        ("ligand_rmsd", 3, False),
    ]
    assert _md_separate_plot_specs(selected, [1, 3], "Serial") == [
        ("backbone_rmsd", 1, False),
        ("ligand_rmsd", 1, False),
        ("backbone_rmsd", 3, False),
        ("ligand_rmsd", 3, False),
    ]


def test_md_plot_export_bundle_uses_independent_panel_and_combined_dpi() -> None:
    report = {
        "replicas": [
            {"replica": 1, "reference_site_retained_fraction": 0.8},
            {"replica": 2, "reference_site_retained_fraction": 0.9},
        ]
    }

    archive_data, manifest = _md_plot_export_bundle(
        report,
        [("site_retention", None, True)],
        column_count=1,
        interaction_limit=12,
        selected_replicas=[2],
        panel_dpi=100,
        combined_dpi=150,
    )

    assert manifest["panel_dpi"] == 100
    assert manifest["composite_dpi"] == 150
    with zipfile.ZipFile(BytesIO(archive_data)) as archive:
        panel_name = f"panels/{manifest['panels'][0]}"
        assert {panel_name, "md-plots-combined.png", "manifest.json"} <= set(
            archive.namelist()
        )
        with Image.open(BytesIO(archive.read(panel_name))) as panel:
            assert panel.size == (420, 300)
        with Image.open(
            BytesIO(archive.read("md-plots-combined.png"))
        ) as combined:
            assert combined.size == (630, 450)


def test_md_plot_data_export_respects_replica_and_residue_filters() -> None:
    report = {
        "replicas": [
            {"replica": 1, "performance_ns_per_day": 100.0},
            {"replica": 2, "performance_ns_per_day": 120.0},
        ],
        "replica_series": [
            {
                "replica": 1,
                "time_ns": [0.0, 1.0],
                "backbone_rmsd_angstrom": [0.0, 1.0],
                "protein_rmsf_residues": ["ALA1"],
                "protein_rmsf_angstrom": [0.5],
            },
            {
                "replica": 2,
                "time_ns": [0.0, 1.0],
                "backbone_rmsd_angstrom": [0.0, 1.2],
                "contact_distance_series_angstrom": {
                    "ASP2": [3.0, 3.1],
                    "GLU3": [4.0, 4.1],
                },
                "protein_rmsf_residues": ["ALA1"],
                "protein_rmsf_angstrom": [0.6],
            },
        ],
        "contact_consensus": [
            {"residue": "ASP2", "mean_contact_occupancy": 0.8},
            {"residue": "GLU3", "mean_contact_occupancy": 0.4},
        ],
        "contact_matrix": {
            "residues": ["ASP2", "GLU3"],
            "replicas": [1, 2],
            "contact_occupancy": [[0.7, 0.8], [0.3, 0.4]],
        },
    }

    tables = _md_plot_data_tables(
        report, selected_replicas=[2], interaction_limit=1
    )

    assert set(tables["Time series"]["Replica"]) == {2}
    assert set(tables["Contact distances"]["Residue"]) == {"ASP2"}
    assert set(tables["Contact matrix"]["Replica"]) == {2}
    assert set(tables["Contact matrix"]["Residue"]) == {"ASP2"}
    assert "Replica 2" in tables["Summary by replica"]
    assert "Replica 1" not in tables["Summary by replica"]

    workbook = _md_plot_data_workbook(
        report, selected_replicas=[2], interaction_limit=1
    )
    with pd.ExcelFile(BytesIO(workbook)) as excel_file:
        assert {
            "Export settings",
            "Summary by replica",
            "Time series",
            "Contact distances",
            "Contact matrix",
        }.issubset(excel_file.sheet_names)
def test_legacy_secondary_structure_series_are_normalized() -> None:
    normalized = _normalized_secondary_structure(
        {
            "helix_fraction": [0.016, 0.012],
            "sheet_fraction": [0.002, 0.004],
            "coil_fraction": [0.007, 0.009],
        }
    )

    assert normalized["helix_fraction"][0] == pytest.approx(0.64)
    assert normalized["sheet_fraction"][0] == pytest.approx(0.08)
    assert normalized["coil_fraction"][0] == pytest.approx(0.28)
    assert all(
        sum(frame) == pytest.approx(1.0)
        for frame in zip(
            normalized["helix_fraction"],
            normalized["sheet_fraction"],
            normalized["coil_fraction"],
            strict=True,
        )
    )


def test_gromacs_performance_is_recovered_from_production_log(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "bound-ligand-md" / "replica-3"
    run_dir.mkdir(parents=True)
    (run_dir / "production.log").write_text(
        "Performance:     1374.970        0.017        0.251\n"
    )
    production = JobRecord(
        run_id="replica-3",
        task_group="bound-ligand-md",
        run_dir=run_dir,
        status="completed",
    )

    performance = _production_performance(
        production,
        {"md_result": {"engine": "gromacs"}},
        {},
    )

    assert performance == {
        "ns_per_day": 1374.97,
        "source": "GROMACS production.log",
        "sample_count": 1,
    }


def test_md_plot_matrix_exposes_uniform_interaction_panels() -> None:
    report = {
        "replicas": [
            {
                "replica": 1,
                "reference_site_retained_fraction": 0.9,
                "performance_ns_per_day": 1000.0,
                "delta_g_bind_kcal_mol": -12.0,
            }
        ],
        "replica_series": [
            {
                "replica": 1,
                "time_ns": [0.0, 1.0],
                "backbone_rmsd_angstrom": [0.0, 1.0],
                "ligand_rmsd_angstrom": [0.0, 1.2],
                "protein_rg_angstrom": [20.0, 20.1],
                "ligand_rg_angstrom": [3.0, 3.1],
                "complex_rg_angstrom": [20.2, 20.3],
                "helix_fraction": [0.5, 0.5],
                "sheet_fraction": [0.2, 0.2],
                "coil_fraction": [0.3, 0.3],
                "ligand_centroid_displacement_angstrom": [0.0, 0.5],
                "minimum_protein_distance_angstrom": [2.0, 2.1],
            }
        ],
        "contact_consensus": [
            {
                "residue": "ASP145 · chain A",
                "mean_contact_occupancy": 0.8,
                "sample_sd_contact_occupancy": 0.1,
                "mean_hydrogen_bond_occupancy": 0.4,
                "mean_hydrogen_bond_backbone_occupancy": 0.1,
                "mean_hydrogen_bond_sidechain_occupancy": 0.3,
                "sample_sd_hydrogen_bond_occupancy": 0.05,
                "mean_hydrophobic_occupancy": 0.8,
                "mean_hydrophobic_backbone_occupancy": 0.0,
                "mean_hydrophobic_sidechain_occupancy": 0.8,
                "sample_sd_hydrophobic_occupancy": 0.1,
                "mean_water_bridge_occupancy": 0.2,
                "sample_sd_water_bridge_occupancy": 0.02,
                "mean_salt_bridge_occupancy": 0.3,
                "sample_sd_salt_bridge_occupancy": 0.03,
                "mean_binding_importance_score": 2.0,
                "sample_sd_binding_importance_score": 0.1,
                "mean_minimum_distance_angstrom": 3.1,
                "top_ligand_atom": "C1",
            }
        ],
        "contact_matrix": {
            "residues": ["ASP145 · chain A"],
            "replicas": [1],
            "contact_occupancy": [[0.8]],
            "hydrogen_bond_occupancy": [[0.25]],
            "hydrophobic_occupancy": [[0.6]],
            "water_bridge_occupancy": [[0.1]],
            "salt_bridge_occupancy": [[0.3]],
            "binding_importance_score": [[1.35]],
        },
        "required_interaction_consensus": [
            {
                "label": "hydrogen bond · ASP145 (BB+SC)",
                "residue": "ASP145 · chain A",
                "mean_occupancy": 0.4,
                "sample_sd_occupancy": 0.05,
                "replica_occupancy": [0.25],
            }
        ],
        "salt_bridge_applicable": True,
        "ligand_depiction": {
            "atoms": [
                {"name": "C1", "symbol": "C", "x": -0.5, "y": 0.0},
                {
                    "name": "N1",
                    "symbol": "N",
                    "formal_charge": 1,
                    "x": 0.5,
                    "y": 0.0,
                },
            ],
            "bonds": [{"begin": 0, "end": 1, "order": 1.0}],
        },
        "rmsf_consensus": [
            {
                "residue": "ASP145 · chain A",
                "mean_ca_rmsf_angstrom": 0.8,
                "sample_sd_ca_rmsf_angstrom": 0.1,
            }
        ],
        "ligand_rmsf_consensus": [
            {
                "atom": "C1",
                "mean_rmsf_angstrom": 0.5,
                "sample_sd_rmsf_angstrom": 0.05,
            }
        ],
    }

    available = _available_md_plot_ids(report)

    assert "radius_of_gyration" in available
    assert "contact_matrix" in available
    assert "interaction_network" in available
    assert "hydrophobic_occupancy" in available
    assert "water_bridge_occupancy" in available
    assert "salt_bridge_occupancy" in available
    assert "interaction_composition" in available
    assert "required_interaction_retention" in available
    figures = [
        _md_plot_figure(plot_id, report)
        for plot_id in (
            "backbone_rmsd",
            "required_interaction_retention",
            "interaction_network",
            "interaction_composition",
            "contact_occupancy",
            "hydrophobic_occupancy",
        )
    ]
    assert all(figure is not None for figure in figures)
    assert all(
        tuple(float(value) for value in figure.get_size_inches())
        == pytest.approx((4.2, 3.0))
        for figure in figures
        if figure is not None
    )
    for figure in figures:
        if figure is not None:
            figure.clear()
    network = _md_plot_figure("interaction_network", report)
    assert network is not None
    network_axis = network.axes[0]
    assert {
        text.get_text()
        for text in network_axis.texts
        if text.get_text().endswith("%")
    } >= {"40%", "80%", "20%", "30%"}
    assert len(
        {
            round(patch.get_linewidth(), 3)
            for patch in network_axis.patches
        }
    ) > 1
    direct_network = _md_plot_figure(
        "interaction_network",
        report,
        include_water_bridges=False,
        include_proximity_interactions=False,
        show_interaction_scopes=True,
    )
    assert direct_network is not None
    direct_labels = direct_network.axes[0].get_legend_handles_labels()[1]
    assert "Water bridge" not in direct_labels
    assert "Proximity only" not in direct_labels
    direct_text = {
        text.get_text() for text in direct_network.axes[0].texts
    }
    assert any("HB:BB+SC" in text for text in direct_text)
    assert any("HP:SC" in text for text in direct_text)
    replica_network = _md_plot_figure(
        "interaction_network",
        report,
        replica=1,
    )
    assert replica_network is not None
    assert replica_network.axes[0].get_title().endswith("Replica 1")
    assert {
        text.get_text()
        for text in replica_network.axes[0].texts
        if text.get_text().endswith("%")
    } >= {"25%", "60%", "10%", "30%"}
    composition = _md_plot_figure("interaction_composition", report)
    assert composition is not None
    assert composition.axes[0].get_ylim() == pytest.approx((0.0, 1.05))
    assert composition.axes[0].collections
    assert all(
        patch.get_y() == pytest.approx(0.0)
        for patch in composition.axes[0].patches
    )
    hotspot = _md_plot_figure("contact_occupancy", report)
    assert hotspot is not None
    assert "3H + 2.5SB + 1.5WB + Hyd" in hotspot.axes[0].get_xlabel()
    assert {
        "backbone_rmsd",
        "ligand_rmsd",
        "ligand_displacement",
        "minimum_distance",
        "contact_occupancy",
        "hbond_occupancy",
        "water_bridge_occupancy",
        "salt_bridge_occupancy",
        "interaction_composition",
    } <= _MD_REPLICA_SPLIT_PLOTS
    replica_hotspot = _md_plot_figure(
        "contact_occupancy",
        report,
        replica=1,
    )
    assert replica_hotspot is not None
    assert replica_hotspot.axes[0].get_title().endswith("Replica 1")
    assert "±" not in replica_hotspot.axes[0].get_xlabel()
    assert replica_hotspot.axes[0].patches[0].get_width() == pytest.approx(
        1.35
    )
    replica_composition = _md_plot_figure(
        "interaction_composition",
        report,
        replica=1,
    )
    assert replica_composition is not None
    assert replica_composition.axes[0].get_title().endswith("Replica 1")
    assert len(replica_composition.axes[0].collections) == 0
    assert sorted(
        patch.get_height()
        for patch in replica_composition.axes[0].patches
    ) == pytest.approx(sorted([0.25, 0.6, 0.1, 0.3]))


def test_contact_matrix_aligns_replicas_to_hotspot_residue_order() -> None:
    report = {
        "contact_consensus": [
            {
                "residue": "SER277 · chain A",
                "mean_contact_occupancy": 0.9,
                "sample_sd_contact_occupancy": 0.1,
                "mean_hydrogen_bond_occupancy": 0.1,
                "sample_sd_hydrogen_bond_occupancy": 0.02,
                "mean_minimum_distance_angstrom": 3.1,
                "sample_sd_minimum_distance_angstrom": 0.1,
            },
            {
                "residue": "HIS381 · chain A",
                "mean_contact_occupancy": 0.7,
                "sample_sd_contact_occupancy": 0.2,
                "mean_hydrogen_bond_occupancy": 0.9,
                "sample_sd_hydrogen_bond_occupancy": 0.03,
                "mean_minimum_distance_angstrom": 3.8,
                "sample_sd_minimum_distance_angstrom": 0.2,
            },
        ],
        "contact_matrix": {
            "residues": [
                "SER277 · chain A",
                "HIS381 · chain A",
            ],
            "replicas": [1, 2],
            "contact_occupancy": [
                [1.0, 0.8],
                [0.5, 0.9],
            ],
            "hydrogen_bond_occupancy": [
                [0.08, 0.12],
                [0.88, 0.92],
            ],
            "minimum_distance_angstrom": [
                [3.0, 3.2],
                [3.6, 4.0],
            ],
        },
        "replica_series": [
            {
                "replica": 1,
                "time_ns": [0.0, 1.0, 2.0],
                "contact_cutoff_angstrom": 4.5,
                "contact_distance_series_angstrom": {
                    "SER277 · chain A": [3.0, 3.1, 3.2],
                    "HIS381 · chain A": [5.0, 4.0, 5.0],
                },
            },
            {
                "replica": 2,
                "time_ns": [0.0, 1.0, 2.0],
                "contact_cutoff_angstrom": 4.5,
                "contact_distance_series_angstrom": {
                    "SER277 · chain A": [3.0, 5.0, 3.0],
                },
            },
        ],
    }

    figure = _md_plot_figure("contact_matrix", report)
    composition = _md_plot_figure("interaction_composition", report)
    interaction_bar = _md_plot_figure("hbond_occupancy", report)

    assert figure is not None
    assert composition is not None
    assert interaction_bar is not None
    assert figure.axes[0].get_position().bounds == pytest.approx(
        interaction_bar.axes[0].get_position().bounds
    )
    assert figure.axes[0].get_ylim() == pytest.approx(
        interaction_bar.axes[0].get_ylim()
    )
    assert [tick for tick in figure.axes[0].child_axes[0].get_yticks()] == [
        0.0,
        1.0,
    ]
    axis = figure.axes[0]
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        "SER277",
        "HIS381",
    ]
    assert [label.get_text() for label in axis.get_xticklabels()] == [
        "Replica 1\n0–2 ns",
        "Replica 2\n0–2 ns",
    ]
    missing_mask = np.ma.getmaskarray(axis.images[0].get_array())
    assert missing_mask[1, -3:].all()
    persistence = _md_plot_figure(
        "persistence_distance",
        report,
        interaction_limit=1,
    )
    assert persistence is not None
    assert persistence.axes[0].get_ylim()[0] > 0.0
    assert persistence.axes[0].get_ylim()[1] <= 1.02
    assert [text.get_text() for text in persistence.axes[0].get_legend().texts] == [
        "Mean ± sample SD"
    ]
    assert {text.get_text() for text in persistence.axes[0].texts} == {
        "SER277",
    }
    replica_persistence = _md_plot_figure(
        "persistence_distance",
        report,
        replica=2,
    )
    assert replica_persistence is not None
    hbond = _md_plot_figure("hbond_occupancy", report)
    assert hbond is not None
    assert [label.get_text() for label in hbond.axes[0].get_yticklabels()] == [
        "SER277",
        "HIS381",
    ]


def test_md_interaction_plots_split_backbone_and_sidechain_occupancy() -> None:
    report = {
        "contact_consensus": [
            {
                "residue": "SER277 · chain A",
                "mean_contact_occupancy": 0.9,
                "sample_sd_contact_occupancy": 0.1,
                "mean_hydrogen_bond_occupancy": 0.75,
                "sample_sd_hydrogen_bond_occupancy": 0.05,
                "mean_hydrogen_bond_backbone_occupancy": 0.25,
                "sample_sd_hydrogen_bond_backbone_occupancy": 0.02,
                "mean_hydrogen_bond_sidechain_occupancy": 0.5,
                "sample_sd_hydrogen_bond_sidechain_occupancy": 0.03,
                "mean_hydrophobic_occupancy": 0.4,
                "sample_sd_hydrophobic_occupancy": 0.04,
                "mean_hydrophobic_backbone_occupancy": 0.0,
                "sample_sd_hydrophobic_backbone_occupancy": 0.0,
                "mean_hydrophobic_sidechain_occupancy": 0.4,
                "sample_sd_hydrophobic_sidechain_occupancy": 0.04,
                "mean_water_bridge_occupancy": 0.2,
                "sample_sd_water_bridge_occupancy": 0.02,
                "mean_water_bridge_backbone_occupancy": 0.15,
                "sample_sd_water_bridge_backbone_occupancy": 0.01,
                "mean_water_bridge_sidechain_occupancy": 0.05,
                "sample_sd_water_bridge_sidechain_occupancy": 0.01,
            }
        ],
        "contact_matrix": {
            "residues": ["SER277 · chain A"],
            "replicas": [1, 2],
            "hydrogen_bond_occupancy": [[0.7, 0.8]],
            "hydrogen_bond_backbone_occupancy": [[0.2, 0.3]],
            "hydrogen_bond_sidechain_occupancy": [[0.45, 0.55]],
            "hydrophobic_occupancy": [[0.35, 0.45]],
            "hydrophobic_backbone_occupancy": [[0.0, 0.0]],
            "hydrophobic_sidechain_occupancy": [[0.35, 0.45]],
            "water_bridge_occupancy": [[0.18, 0.22]],
            "water_bridge_backbone_occupancy": [[0.14, 0.16]],
            "water_bridge_sidechain_occupancy": [[0.04, 0.06]],
        },
    }

    available = _available_md_plot_ids(report)
    assert "hbond_scope_fraction" in available
    assert "hydrophobic_occupancy" in available
    assert "hydrophobic_scope_fraction" in available
    assert "water_bridge_scope_fraction" in available
    hbond = _md_plot_figure("hbond_occupancy", report)
    assert hbond is not None
    assert [patch.get_width() for patch in hbond.axes[0].patches] == (
        pytest.approx([0.75])
    )

    scope = _md_plot_figure("hbond_scope_fraction", report)
    assert scope is not None
    assert sorted(patch.get_width() for patch in scope.axes[0].patches) == (
        pytest.approx([1 / 3, 2 / 3])
    )
    assert {text.get_text() for text in scope.legends[0].texts} == {
        "BB",
        "SC",
    }
    assert scope.axes[0].get_xlim() == pytest.approx((0.0, 1.0))
    assert scope.axes[0].get_xlabel() == "Fraction of scoped interactions"

    composition = _md_plot_figure("interaction_composition", report)
    assert composition is not None
    assert len(composition.axes[0].patches) == 3
    assert {text.get_text() for text in composition.axes[0].get_legend().texts} == {
        "H-bond",
        "Hydrophobic",
        "Water bridge",
    }


def test_md_interaction_bars_share_hotspot_residues_and_order() -> None:
    report = {
        "contact_consensus": [
            {
                "residue": "SER277 · chain A",
                "mean_contact_occupancy": 0.9,
                "mean_hydrogen_bond_occupancy": 0.7,
                "mean_hydrogen_bond_backbone_occupancy": 0.2,
                "mean_hydrogen_bond_sidechain_occupancy": 0.5,
                "mean_hydrophobic_occupancy": 0.0,
                "mean_hydrophobic_backbone_occupancy": 0.0,
                "mean_hydrophobic_sidechain_occupancy": 0.0,
            },
            {
                "residue": "LEU276 · chain A",
                "mean_contact_occupancy": 0.8,
                "mean_hydrogen_bond_occupancy": 0.0,
                "mean_hydrogen_bond_backbone_occupancy": 0.0,
                "mean_hydrogen_bond_sidechain_occupancy": 0.0,
                "mean_hydrophobic_occupancy": 0.6,
                "mean_hydrophobic_backbone_occupancy": 0.0,
                "mean_hydrophobic_sidechain_occupancy": 0.6,
            },
            {
                "residue": "ALA225 · chain A",
                "mean_contact_occupancy": 0.7,
                "mean_hydrogen_bond_occupancy": 0.0,
                "mean_hydrogen_bond_backbone_occupancy": 0.0,
                "mean_hydrogen_bond_sidechain_occupancy": 0.0,
                "mean_hydrophobic_occupancy": 0.4,
                "mean_hydrophobic_backbone_occupancy": 0.0,
                "mean_hydrophobic_sidechain_occupancy": 0.4,
            },
        ]
    }

    plot_ids = (
        "contact_occupancy",
        "hbond_occupancy",
        "hbond_scope_fraction",
        "hydrophobic_occupancy",
        "hydrophobic_scope_fraction",
    )
    figures = [
        _md_plot_figure(plot_id, report, interaction_limit=2)
        for plot_id in plot_ids
    ]
    assert all(figure is not None for figure in figures)
    assert [
        [label.get_text() for label in figure.axes[0].get_yticklabels()]
        for figure in figures
        if figure is not None
    ] == [["SER277", "LEU276"]] * len(plot_ids)
    assert [
        figure.axes[0].get_position().bounds
        for figure in figures
        if figure is not None
    ] == pytest.approx(
        [figures[0].axes[0].get_position().bounds] * len(plot_ids)
    )
    assert all(
        figure.axes[0].get_ylim() == pytest.approx((1.5, -0.5))
        for figure in figures
        if figure is not None
    )
    assert [
        plot_id for plot_id in _MD_PLOT_LABELS
        if plot_id in {
            "contact_occupancy",
            "contact_scope_fraction",
            "hbond_occupancy",
            "hbond_scope_fraction",
            "hydrophobic_occupancy",
            "hydrophobic_scope_fraction",
        }
    ] == [
        "contact_occupancy",
        "contact_scope_fraction",
        "hbond_occupancy",
        "hbond_scope_fraction",
        "hydrophobic_occupancy",
        "hydrophobic_scope_fraction",
    ]
    assert list(_MD_PLOT_LABELS).index("contact_matrix") < list(
        _MD_PLOT_LABELS
    ).index("interaction_composition")


def test_replica_specific_rg_secondary_structure_and_rmsf_plots() -> None:
    report = {
        "replica_series": [
            {
                "replica": 2,
                "time_ns": [0.0, 1.0],
                "protein_rg_angstrom": [18.0, 18.1],
                "ligand_rg_angstrom": [4.0, 4.1],
                "complex_rg_angstrom": [18.2, 18.3],
                "helix_fraction": [0.6, 0.61],
                "sheet_fraction": [0.1, 0.09],
                "coil_fraction": [0.3, 0.3],
                "protein_rmsf_residues": [
                    "HIS381 · chain A",
                    "SER277 · chain A",
                ],
                "protein_rmsf_angstrom": [1.1, 0.8],
                "ligand_rmsf_atoms": ["C1", "N1"],
                "ligand_rmsf_angstrom": [0.4, 0.7],
            }
        ]
    }

    figures = {
        plot_id: _md_plot_figure(plot_id, report, replica=2)
        for plot_id in (
            "radius_of_gyration",
            "secondary_structure",
            "protein_rmsf",
            "ligand_rmsf",
        )
    }

    assert all(figure is not None for figure in figures.values())
    assert all(
        figure.axes[0].get_title().endswith("Replica 2")
        for figure in figures.values()
        if figure is not None
    )
    assert len(figures["radius_of_gyration"].axes[0].lines) == 3
    assert len(figures["secondary_structure"].axes[0].lines) == 3
    protein_rmsf_y = figures["protein_rmsf"].axes[0].lines[0].get_ydata()
    protein_rmsf_x = figures["protein_rmsf"].axes[0].lines[0].get_xdata()
    assert list(protein_rmsf_x) == [277.0, 329.0, 381.0]
    assert protein_rmsf_y[0] == 0.8
    assert np.isnan(protein_rmsf_y[1])
    assert protein_rmsf_y[2] == 1.1
    assert (
        figures["protein_rmsf"].axes[0].get_xlabel()
        == "Original residue number"
    )
    assert [
        label.get_text()
        for label in figures["ligand_rmsf"].axes[0].get_xticklabels()
    ] == ["C1", "N1"]


def test_required_interaction_consensus_uses_requested_scope() -> None:
    rows = _required_interaction_consensus(
        [
            {
                "Enabled": True,
                "Required": True,
                "Interaction": "hydrogen bond",
                "Protein residue": "A:SER277",
                "Protein region": "BB",
                "Reference support": 0.76,
            }
        ],
        [
            {
                "residue": "SER277 · chain A",
                "mean_hydrogen_bond_backbone_occupancy": 0.55,
                "sample_sd_hydrogen_bond_backbone_occupancy": 0.1,
            }
        ],
        {
            "residues": ["SER277 · chain A"],
            "hydrogen_bond_backbone_occupancy": [[0.4, 0.5, 0.75]],
        },
    )

    assert rows[0]["residue"] == "SER277 · chain A"
    assert rows[0]["mean_occupancy"] == 0.55
    assert rows[0]["replica_occupancy"] == [0.4, 0.5, 0.75]


def test_consensus_energy_and_throughput_bars_show_mean_and_replicas() -> None:
    report = {
        "replicas": [
            {
                "replica": 1,
                "performance_ns_per_day": 800.0,
                "reference_site_retained_fraction": 1.0,
                "delta_g_bind_kcal_mol": -40.0,
                "delta_mm_kcal_mol": -60.0,
                "delta_gbsa_kcal_mol": 26.0,
                "delta_nonpolar_kcal_mol": -6.0,
                "pb_delta_g_bind_kcal_mol": 1.0,
                "pb_delta_mm_kcal_mol": -60.0,
                "pb_delta_pbsa_kcal_mol": 35.0,
                "pb_delta_nonpolar_kcal_mol": 26.0,
            },
            {
                "replica": 2,
                "performance_ns_per_day": 1000.0,
                "reference_site_retained_fraction": 1.0,
                "delta_g_bind_kcal_mol": -44.0,
                "delta_mm_kcal_mol": -64.0,
                "delta_gbsa_kcal_mol": 27.0,
                "delta_nonpolar_kcal_mol": -7.0,
                "pb_delta_g_bind_kcal_mol": 3.0,
                "pb_delta_mm_kcal_mol": -64.0,
                "pb_delta_pbsa_kcal_mol": 40.0,
                "pb_delta_nonpolar_kcal_mol": 27.0,
            },
            {
                "replica": 3,
                "performance_ns_per_day": 1200.0,
                "reference_site_retained_fraction": 1.0,
                "delta_g_bind_kcal_mol": -42.0,
                "delta_mm_kcal_mol": -62.0,
                "delta_gbsa_kcal_mol": 28.0,
                "delta_nonpolar_kcal_mol": -6.5,
                "pb_delta_g_bind_kcal_mol": 5.0,
                "pb_delta_mm_kcal_mol": -62.0,
                "pb_delta_pbsa_kcal_mol": 45.0,
                "pb_delta_nonpolar_kcal_mol": 28.0,
            },
        ]
    }

    available = _available_md_plot_ids(report)
    throughput = _md_plot_figure("throughput", report)
    retention = _md_plot_figure("site_retention", report)
    pb_energy = _md_plot_figure("endpoint_energy_pb", report)
    binding_summary = _md_plot_figure(
        "endpoint_binding_summary",
        report,
    )
    separate_throughput = _md_plot_figure(
        "throughput",
        report,
        replica_bars=True,
    )
    separate_binding = _md_plot_figure(
        "endpoint_binding_summary",
        report,
        replica_bars=True,
    )
    selected_repeat_figures = {
        plot_id: _md_plot_figure(
            plot_id,
            report,
            replica_bars=True,
            selected_replicas=[1, 3],
        )
        for plot_id in (
            "site_retention",
            "endpoint_energy",
            "endpoint_binding_summary",
            "throughput",
        )
    }

    assert "endpoint_energy" in available
    assert "endpoint_energy_pb" in available
    assert "endpoint_binding_summary" in available
    assert throughput is not None
    assert retention is not None
    assert pb_energy is not None
    assert binding_summary is not None
    assert separate_throughput is not None
    assert separate_binding is not None
    assert throughput.axes[0].patches[0].get_height() == pytest.approx(
        1000.0
    )
    assert throughput.axes[0].patches[0].get_width() == pytest.approx(0.32)
    assert retention.axes[0].patches[0].get_width() == pytest.approx(0.32)
    assert retention.axes[0].get_ylim() == pytest.approx((0.0, 1.08))
    assert all(
        figure is not None for figure in selected_repeat_figures.values()
    )
    assert {
        plot_id: [
            label.get_text() for label in figure.axes[0].get_xticklabels()
        ]
        for plot_id, figure in selected_repeat_figures.items()
        if figure is not None
    } == {
        "site_retention": ["R1", "R3"],
        "endpoint_energy": ["R1", "R3"],
        "endpoint_binding_summary": ["R1", "R3"],
        "throughput": ["R1", "R3"],
    }
    assert retention.axes[0].get_xlim() == pytest.approx((-0.65, 0.65))
    assert any(
        len(collection.get_offsets()) == 3
        for collection in throughput.axes[0].collections
        if hasattr(collection, "get_offsets")
    )
    assert [patch.get_height() for patch in pb_energy.axes[0].patches] == (
        pytest.approx([3.0, -62.0, 40.0, 27.0])
    )
    assert [
        patch.get_height()
        for patch in binding_summary.axes[0].patches
    ] == pytest.approx([-42.0, 3.0])
    assert [
        patch.get_height()
        for patch in separate_throughput.axes[0].patches
    ] == pytest.approx([800.0, 1000.0, 1200.0])
    assert len(separate_binding.axes[0].patches) == 6


def test_amber_pb_parser_combines_nonpolar_and_dispersion_terms(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "FINAL_RESULTS_MMPBSA_PB.dat"
    result_file.write_text(
        "Differences (Complex - Receptor - Ligand):\n"
        "VDWAALS -55.9046 2.5 0.1\n"
        "EEL -9.0097 4.2 0.2\n"
        "EPB 39.5975 5.5 0.3\n"
        "ENPOLAR -35.8805 0.5 0.03\n"
        "EDISPER 62.2864 0.9 0.05\n"
        "DELTA TOTAL 1.0891 6.0 0.3\n"
    )

    parsed = _parse_amber_final_results(result_file)

    assert parsed is not None
    assert parsed["delta_total_kcal"] == pytest.approx(1.0891)
    assert parsed["delta_pol_kcal"] == pytest.approx(39.5975)
    assert parsed["delta_np_kcal"] == pytest.approx(26.4059)


def test_recompute_mmgbsa_derives_gromacs_ligand_from_common_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_ambertools(
        input_config: dict[str, Any],
        selected: dict[str, Any],
        md_result: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        captured.update(selected)
        return {"status": "success"}

    monkeypatch.setattr(
        "mn_ligand.workflows.bound_ligand_md._compute_mmgbsa_ambertools",
        fake_ambertools,
    )
    result = recompute_mmgbsa(
        {
            "ligand_key": "LIG|A|501|_",
            "mmgbsa_backend": "ambertools_mmpbsa",
        },
        {"md_result": {"engine": "gromacs"}},
        tmp_path / "result.json",
    )

    assert result["mmgbsa"]["status"] == "success"
    assert captured == {
        "key": "LIG|A|501|_",
        "resname": "LIG",
        "chain": "A",
        "resseq": "501",
        "icode": "",
    }


def _source_job(runs_dir: Path) -> tuple[JobRecord, ArtifactRef]:
    run_dir = runs_dir / "structure-jobs" / "structure-1"
    run_dir.mkdir(parents=True)
    complex_path = run_dir / "target_complex_refined.pdb"
    complex_path.write_text(
        "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C\n"
        "HETATM    2  C1  LIG A 501      12.000  10.000  10.000  1.00 20.00           C\nEND\n"
    )
    _write_json(
        run_dir / "metadata.json",
        {"schema_version": 1, "run_id": run_dir.name, "status": "completed", "pdb_id": "TEST"},
    )
    artifact = ArtifactRef.from_path(run_dir, complex_path, "prepared_complex", role="complex")
    write_artifact_manifest(run_dir, [artifact])
    return JobRecord.load(run_dir, task_group="structure-jobs"), artifact


def _prep_input(source_path: Path) -> dict:
    return {
        "pdb_id": "TEST",
        "pdb_data": source_path.read_text(),
        "ligand_key": "LIG|A|501|_",
        "forcefield_method": "openff-2.2.0",
        "charge_method": "gasteiger",
        "box_shape": "dodecahedron",
        "padding_nm": 1.0,
        "ionic_strength": 0.15,
        "constraints": "HBonds",
        "mmgbsa_backend": "openmm_gbsa",
        "temperature": 300.0,
        "pressure": 1.0,
        "heating_steps_per_stage": 250,
        "heating_stages": 6,
        "nvt_steps": 2500,
        "npt_steps": 2500,
        "prepared_complex_path": str(source_path),
    }


def test_md_template_launch_is_fresh_independent_full_duration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    old_job, old_artifact = _source_job(runs_dir)
    template = create_md_simulation(
        source_job=old_job,
        source_artifact=old_artifact,
        prep_input=_prep_input(
            old_artifact.resolve(old_job.run_dir, must_exist=True)
        ),
        production={
            "production_steps": 12_500_000,
            "production_length_ns": 50.0,
            "production_timestep_fs": 4.0,
            "production_report_interval": 2500,
            "continuation_mode": INDEPENDENT_REPLICA,
            "endpoint_enabled": True,
            "endpoint_backend": "ambertools_mmpbsa",
            "target_duration_ns": 100.0,
        },
        replicas=3,
        analysis_enabled=True,
        use_gpu=False,
    )

    new_dir = runs_dir / "target-trimming" / "new-target"
    new_dir.mkdir(parents=True)
    new_complex = new_dir / "new_complex.pdb"
    new_complex.write_text(
        "ATOM      1  CA  ALA A   1      20.000  20.000  20.000  1.00 20.00           C\n"
        "HETATM    2  C1  T3  A 601      22.000  20.000  20.000  1.00 20.00           C\nEND\n"
    )
    _write_json(
        new_dir / "metadata.json",
        {
            "schema_version": 1,
            "run_id": new_dir.name,
            "status": "completed",
            "pdb_id": "3GWS",
            "ligand_key": "T3|A|601|_",
        },
    )
    new_artifact = ArtifactRef.from_path(
        new_dir,
        new_complex,
        "prepared_complex",
        role="complex",
    )
    write_artifact_manifest(new_dir, [new_artifact])
    new_job = JobRecord.load(new_dir, task_group="target-trimming")

    workflow = create_md_simulation_from_template(
        template_workflow_id=template.workflow_id,
        source_job=new_job,
        source_artifact=new_artifact,
        name="3GWS 7C4A9 OpenMM 100 ns x3 with endpoint analysis",
    )

    assert workflow.name == "3GWS 7C4A9 OpenMM 100 ns x3 with endpoint analysis"
    assert workflow.parameters["template_workflow_id"] == template.workflow_id
    assert workflow.parameters["template_settings_only"] is True
    assert workflow.parameters["source"]["run_id"] == new_job.run_id
    assert workflow.parameters["replicas"] == 3
    assert workflow.parameters["analysis_enabled"] is True
    assert workflow.parameters["production"]["production_length_ns"] == 100.0
    assert workflow.parameters["production"]["production_steps"] == 25_000_000
    assert "target_duration_ns" not in workflow.parameters["production"]
    assert workflow.parameters["production"]["continuation_mode"] == INDEPENDENT_REPLICA
    assert len(
        [child for child in workflow.children if child.step_id.startswith("production_replica_")]
    ) == 3
    assert len(
        [step for step in workflow.expected_steps if step.startswith("endpoint_energy_replica_")]
    ) == 3

    prep_ref = next(
        child
        for child in workflow.children
        if child.step_id == "preparation_equilibration"
    )
    prep_payload = json.loads(
        (runs_dir / prep_ref.task_group / prep_ref.run_id / "input.json").read_text()
    )
    assert prep_payload["pdb_id"] == "3GWS"
    assert prep_payload["ligand_key"] == "T3|A|601|_"
    assert prep_payload["prepared_complex_path"] == "/output/source_complex.pdb"
    assert "target_duration_ns" not in prep_payload


def _completed_production(runs_dir: Path) -> JobRecord:
    run_dir = runs_dir / "bound-ligand-md" / "production-1"
    output_dir = run_dir / "production-1"
    output_dir.mkdir(parents=True)
    trajectory = output_dir / "production.dcd"
    final_pdb = output_dir / "production.pdb"
    trajectory.write_text("trajectory")
    final_pdb.write_text("MODEL\nENDMDL\n")
    _write_json(
        run_dir / "metadata.json",
        {
            "schema_version": 1,
            "run_id": run_dir.name,
            "status": "completed",
            "docker_image": "ovolig-md-cu128:latest",
            "use_gpu": False,
        },
    )
    _write_json(
        run_dir / "input.json",
        {
            "prepared_complex_path": f"/output/{run_dir.name}/{final_pdb.name}",
            "mmgbsa_backend": "openmm_gbsa",
        },
    )
    _write_json(
        run_dir / "result.json",
        {
            "success": True,
            "md_result": {
                "equilibration_stats": {
                    "preparation_protocol": {
                        "protocol": "roe_brooks_2020",
                        "density_stabilization": {
                            "fit": {"plateau": True}
                        },
                    }
                },
                "output_files": {
                    "production_trajectory": f"/output/{run_dir.name}/{trajectory.name}",
                    "production_pdb": f"/output/{run_dir.name}/{final_pdb.name}",
                }
            },
        },
    )
    write_artifact_manifest(
        run_dir,
        [
            ArtifactRef.from_path(run_dir, trajectory, "md_trajectory", role="production"),
            ArtifactRef.from_path(run_dir, final_pdb, "md_final_structure", role="coordinates"),
        ],
    )
    return JobRecord.load(run_dir, task_group="bound-ligand-md")


def test_completed_md_finalization_is_idempotent(tmp_path: Path) -> None:
    source = _completed_production(tmp_path / "runs")
    manifest_path = source.run_dir / "artifacts.json"
    before = manifest_path.stat().st_mtime_ns

    finalized = finalize_md_job(source)

    assert finalized.status == "completed"
    assert manifest_path.stat().st_mtime_ns == before


def test_compatibility_fingerprint_is_stable_and_reports_changed_section() -> None:
    source = {"run_id": "one", "sha256": "abc"}
    first = compatibility_contract(source, {"temperature": 300.0, "padding_nm": 1.0})
    same = compatibility_contract(dict(source), {"padding_nm": 1.0, "temperature": 300.0})
    changed = compatibility_contract(source, {"temperature": 310.0, "padding_nm": 1.0})

    assert compatibility_fingerprint(first) == compatibility_fingerprint(same)
    assert compatibility_fingerprint(first) != compatibility_fingerprint(changed)
    assert compatibility_differences(first, changed) == ["equilibration"]


def test_continuation_modes_and_replica_seeds_are_explicit() -> None:
    assert continuation_mode({"continuation_mode": EXACT_CONTINUATION}) == EXACT_CONTINUATION
    assert continuation_mode({"continuation_mode": INDEPENDENT_REPLICA}) == INDEPENDENT_REPLICA
    assert replica_seed("workflow", 1) == replica_seed("workflow", 1)
    assert replica_seed("workflow", 1) != replica_seed("workflow", 2)


def test_roe_brooks_density_fit_accepts_plateau_and_rejects_drift() -> None:
    times = [float(index * 10) for index in range(201)]
    plateau = [
        0.98 + (1.02 - 0.98) * (1.0 - math.exp(-0.01 * time))
        for time in times
    ]
    drifting = [0.98 + 0.0001 * time for time in times]

    accepted = fit_density_plateau(times, plateau)
    rejected = fit_density_plateau(times, drifting)

    assert accepted["plateau"] is True
    assert accepted["criteria"] == {
        "slope_below_1e-6": True,
        "final_difference_below_0_02": True,
        "chi_squared_below_0_5": True,
    }
    assert rejected["plateau"] is False
    assert rejected["criteria"]["slope_below_1e-6"] is False


def test_roe_brooks_configuration_rejects_invalid_density_window() -> None:
    config = MDOptimizationConfig(
        protein_pdb_data="ATOM\n",
        preparation_protocol="roe_brooks_2020",
        density_stabilization_min_ns=2.0,
        density_stabilization_max_ns=1.0,
    )

    valid, message = config.validate()

    assert valid is False
    assert "density_stabilization_max_ns" in message


def test_md_config_preserves_selectable_physical_models() -> None:
    openmm = MDOptimizationConfig.from_dict(
        {
            "protein_pdb_data": "ATOM\n",
            "protein_forcefield_method": "amberfb15",
            "forcefield_method": "openff-2.2.0",
            "water_model": "tip3pfb",
        }
    )
    gromacs = MDOptimizationConfig.from_dict(
        {
            "protein_pdb_data": "ATOM\n",
            "protein_forcefield_method": "ff19SB",
            "forcefield_method": "gaff2",
            "water_model": "opc",
        }
    )

    assert openmm.validate()[0] is True
    assert openmm.protein_forcefield_method == "amberfb15"
    assert openmm.water_model == "tip3pfb"
    assert gromacs.validate()[0] is True
    assert gromacs.protein_forcefield_method == "ff19SB"
    assert gromacs.water_model == "opc"
    assert "protein_forcefield_method" in SYSTEM_PARAMETER_KEYS
    assert "water_model" in SYSTEM_PARAMETER_KEYS


def test_md_config_rejects_unknown_physical_models() -> None:
    unknown_protein = MDOptimizationConfig(
        protein_pdb_data="ATOM\n",
        protein_forcefield_method="unknown",
    )
    unknown_water = MDOptimizationConfig(
        protein_pdb_data="ATOM\n",
        water_model="unknown",
    )

    assert "protein_forcefield_method" in unknown_protein.validate()[1]
    assert "water_model" in unknown_water.validate()[1]


def test_ambertools_endpoint_backend_is_available_to_both_engines() -> None:
    from mn_ligand.workflows.md_engines import endpoint_backend_supported

    assert endpoint_backend_supported("openmm", "ambertools_mmpbsa") is True
    assert endpoint_backend_supported("gromacs", "ambertools_mmpbsa") is True


def test_openmm_model_registry_includes_roe_octahedron() -> None:
    assert SolvatedSystemBuilder.PROTEIN_FORCEFIELD_FILES["amber14-all"] == (
        "amber14-all.xml"
    )
    assert SolvatedSystemBuilder.WATER_FORCEFIELD_FILES["tip3p"] == (
        "amber14/tip3p.xml"
    )
    assert SolvatedSystemBuilder.SOLVENT_BOX_MODELS["tip3pfb"] == "tip3p"
    assert SolvatedSystemBuilder.OPENMM_BOX_SHAPES["octahedron"] == (
        "octahedron"
    )


def test_md_config_rejects_incomplete_restart_contracts() -> None:
    independent = MDOptimizationConfig(
        protein_pdb_data="ATOM\n",
        coordinate_restart_policy="independent_replica",
        replica_equilibration_steps=250000,
    )
    strict = MDOptimizationConfig(protein_pdb_data="ATOM\n", strict_checkpoint_resume=True)

    assert independent.validate()[0] is False
    assert "equilibrated coordinates" in independent.validate()[1]
    assert strict.validate()[0] is False
    assert "checkpoint" in strict.validate()[1]


def test_md_config_validates_replica_density_revalidation_window() -> None:
    config = MDOptimizationConfig(
        protein_pdb_data="ATOM\n",
        coordinate_restart_policy="independent_replica",
        replica_equilibration_steps=250000,
        replica_density_revalidation=True,
        replica_revalidation_max_steps=100000,
        replica_revalidation_increment_steps=250000,
        replica_density_sample_interval_steps=1000,
    )

    assert config.validate() == (
        False,
        "replica_revalidation_max_steps must be >= replica_equilibration_steps",
    )


def test_md_config_allows_density_only_revalidation_after_reseeding() -> None:
    config = MDOptimizationConfig(
        protein_pdb_data="ATOM\n",
        coordinate_restart_policy="independent_replica",
        replica_equilibration_steps=0,
        replica_density_revalidation=True,
        replica_revalidation_max_steps=1_250_000,
        replica_revalidation_increment_steps=250_000,
        replica_density_sample_interval_steps=1000,
        resume_system_pdb_path="system.pdb",
        resume_state_xml_path="state.xml",
        resume_system_xml_path="system.xml",
        resume_integrator_xml_path="integrator.xml",
    )

    assert config.validate() == (True, "")


def test_prepared_reuse_requires_passed_roe_density_fit() -> None:
    with pytest.raises(ValueError, match="Roe-Brooks 2020"):
        _validate_roe_preparation_result({"success": True, "md_result": {}})
    with pytest.raises(ValueError, match="density plateau"):
        _validate_roe_preparation_result(
            {
                "success": True,
                "md_result": {
                    "preparation_protocol": {
                        "protocol": "roe_brooks_2020",
                        "density_stabilization": {
                            "fit": {"plateau": False}
                        },
                    }
                },
            }
        )
    _validate_roe_preparation_result(
        {
            "success": True,
            "md_result": {
                "preparation_protocol": {
                    "protocol": "roe_brooks_2020",
                    "density_stabilization": {
                        "fit": {"plateau": True}
                    },
                }
            },
        }
    )


def test_md_config_requires_hmr_above_two_fs() -> None:
    config = MDOptimizationConfig(
        protein_pdb_data="ATOM\n",
        production_timestep_fs=4.0,
        hydrogen_mass_amu=None,
    )

    assert config.validate() == (
        False,
        "production timesteps above 2 fs require hydrogen-mass repartitioning",
    )


def test_duplicate_ligand_guard_checks_prepared_protein_component() -> None:
    protein_only = (
        "ATOM      1  CA  ALA A   1      10.000  10.000  10.000"
        "  1.00 20.00           C\nEND\n"
    )
    SolvatedSystemBuilder._assert_ligand_absent_from_protein_component(
        protein_only,
        "LIG",
    )

    for record in ("ATOM  ", "HETATM"):
        duplicate = (
            f"{record}    2  C1  LIG A 501      12.000  10.000  10.000"
            "  1.00 20.00           C\nEND\n"
        )
        with pytest.raises(ValueError, match="prepared protein component"):
            SolvatedSystemBuilder._assert_ligand_absent_from_protein_component(
                duplicate,
                "LIG",
            )


def test_md_workflow_unlocks_replicas_and_completes_analysis(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source_job, source_artifact = _source_job(runs_dir)
    workflow = create_md_simulation(
        source_job=source_job,
        source_artifact=source_artifact,
        prep_input=_prep_input(source_artifact.resolve(source_job.run_dir, must_exist=True)),
        production={
            "production_steps": 1000,
            "production_report_interval": 100,
            "restart_mode": "NPT-final PDB (coordinate restart)",
            "continuation_mode": INDEPENDENT_REPLICA,
            "replica_equilibration_steps": 250000,
            "replica_density_revalidation": True,
            "replica_revalidation_max_steps": 1250000,
            "replica_revalidation_increment_steps": 250000,
            "replica_density_sample_interval_steps": 1000,
            "replica_density_plateau_required": True,
            "mmgbsa_enabled": False,
        },
        replicas=2,
        analysis_enabled=True,
        use_gpu=False,
    )
    assert workflow.status == "running"
    assert len(workflow.children) == 4

    prep_ref = next(item for item in workflow.children if item.step_id == "preparation_equilibration")
    prep_dir = runs_dir / prep_ref.task_group / prep_ref.run_id
    prep_metadata = json.loads((prep_dir / "metadata.json").read_text())
    assert prep_metadata["worker_finalizer"] == "md_job"
    assert prep_metadata["gpu_queued"] is False
    assert prep_metadata["resources"]["gpu"] is False
    stored_prep_input = json.loads((prep_dir / "input.json").read_text())
    assert stored_prep_input["prepared_complex_path"] == "/output/source_complex.pdb"
    assert "pdb_data" not in stored_prep_input
    output_dir = prep_dir / prep_ref.run_id
    output_dir.mkdir()
    system_pdb = output_dir / "test_system.pdb"
    npt_pdb = output_dir / "test_npt.pdb"
    checkpoint = output_dir / "test.chk"
    state_xml = output_dir / "state.xml"
    system_xml = output_dir / "system.xml"
    integrator_xml = output_dir / "integrator.xml"
    for path in (system_pdb, npt_pdb, checkpoint, state_xml, system_xml, integrator_xml):
        path.write_text("prepared")
    _write_json(
        prep_dir / "result.json",
        {
            "success": True,
            "md_result": {
                "equilibration_stats": {
                    "preparation_protocol": {
                        "protocol": "roe_brooks_2020",
                        "density_stabilization": {
                            "fit": {"plateau": True}
                        },
                    }
                },
                "output_files": {
                    "system_pdb": f"/output/{prep_ref.run_id}/{system_pdb.name}",
                    "npt_pdb": f"/output/{prep_ref.run_id}/{npt_pdb.name}",
                    "npt_checkpoint": f"/output/{prep_ref.run_id}/{checkpoint.name}",
                    "npt_state_xml": f"/output/{prep_ref.run_id}/{state_xml.name}",
                    "npt_system_xml": f"/output/{prep_ref.run_id}/{system_xml.name}",
                    "npt_integrator_xml": f"/output/{prep_ref.run_id}/{integrator_xml.name}",
                }
            },
        },
    )

    workflow = advance_md_workflow(workflow.workflow_id)
    assert workflow.status == "running"
    prep_job = JobRecord.load(prep_dir, task_group="md-system-prep")
    assert prep_job.metadata["compatibility_fingerprint"]
    assert prep_job.artifact_manifest.by_type("equilibrated_system")

    production_refs = [item for item in workflow.children if item.step_id.startswith("production_replica_")]
    for child_ref in production_refs:
        run_dir = runs_dir / child_ref.task_group / child_ref.run_id
        metadata = json.loads((run_dir / "metadata.json").read_text())
        assert metadata["awaiting_parent"] is False
        assert metadata["queued_command"]
        assert metadata["worker_finalizer"] == "md_job"
        assert metadata["gpu_queued"] is False
        assert metadata["resources"]["gpu"] is False
        command_record = json.loads((run_dir / "command.json").read_text())
        assert command_record["tool_id"] == "openmm_md"
        production_input = json.loads((run_dir / "input.json").read_text())
        assert production_input["prepared_complex_path"] == "/output/final_input_protein_refined.pdb"
        assert production_input["source_md_system_prep_result_json"] == "/prepared-system/result.json"
        assert production_input["coordinate_restart_policy"] == "independent_replica"
        assert production_input["replica_equilibration_steps"] == 250000
        assert production_input["replica_density_revalidation"] is True
        assert production_input["replica_revalidation_max_steps"] == 1250000
        assert production_input["replica_revalidation_increment_steps"] == 250000
        assert production_input["replica_density_sample_interval_steps"] == 1000
        assert production_input["replica_density_plateau_required"] is True
        assert production_input["resume_system_pdb_path"].startswith("/prepared-system/")
        assert production_input["resume_state_xml_path"].startswith("/prepared-system/")
        assert production_input["resume_system_xml_path"].startswith("/prepared-system/")
        command = metadata["queued_command"]
        assert f"{prep_dir}:/prepared-system:ro" in command
        trajectory_dir = run_dir / child_ref.run_id
        trajectory_dir.mkdir()
        trajectory = trajectory_dir / "production.dcd"
        final_pdb = trajectory_dir / "production.pdb"
        trajectory.write_text("trajectory")
        final_pdb.write_text("structure")
        _write_json(
            run_dir / "result.json",
            {
                "success": True,
                "md_result": {
                    "output_files": {
                        "production_trajectory": f"/output/{child_ref.run_id}/{trajectory.name}",
                        "production_pdb": f"/output/{child_ref.run_id}/{final_pdb.name}",
                    }
                },
            },
        )

    completed = advance_md_workflow(workflow.workflow_id)
    assert completed.status == "completed"
    analysis_ref = next(item for item in completed.children if item.step_id == "replicate_analysis")
    analysis_job = JobRecord.load(runs_dir / analysis_ref.task_group / analysis_ref.run_id, task_group="md-analysis")
    assert analysis_job.status == "completed"
    assert analysis_job.artifact_manifest.by_type("md_replicate_report")
    assert WorkflowRecord.load(workflow.workflow_id).status == "completed"


def test_mmgbsa_analysis_is_an_immutable_typed_child_job(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source = _completed_production(runs_dir)
    prep_dir = runs_dir / "md-system-prep" / "prepared-system"
    prep_dir.mkdir(parents=True)
    _write_json(
        prep_dir / "metadata.json",
        {
            "schema_version": 1,
            "run_id": prep_dir.name,
            "status": "completed",
        },
    )
    source_input = json.loads((source.run_dir / "input.json").read_text())
    source_input["source_md_system_prep_run_id"] = prep_dir.name
    source_input["source_md_system_prep_result_json"] = (
        "/prepared-system/result.json"
    )
    _write_json(source.run_dir / "input.json", source_input)
    original_files = {
        path.relative_to(source.run_dir): path.read_bytes()
        for path in source.run_dir.rglob("*")
        if path.is_file()
    }

    queued = create_mmgbsa_analysis_job(
        source.run_id,
        start_pct=35,
        end_pct=90,
        stride=4,
        backend="ambertools_mmpbsa",
        use_gpu=False,
    )

    assert queued.task_group == "md-mmgbsa"
    assert queued.status == "queued"
    assert queued.parent_run_id == source.run_id
    assert queued.metadata["worker_finalizer"] == "md_mmgbsa"
    assert queued.metadata["resources"]["gpu"] is False
    assert queued.metadata["parameters"] == {
        "start_pct": 35.0,
        "end_pct": 90.0,
        "stride": 4,
        "backend": "ambertools_mmpbsa",
        "cpu_process_limit": queued.metadata["parameters"][
            "cpu_process_limit"
        ],
    }
    assert queued.metadata["parameters"]["cpu_process_limit"] >= 1
    assert queued.metadata["resources"]["cpu_threads"] == queued.metadata[
        "parameters"
    ]["cpu_process_limit"]
    payload = json.loads((queued.run_dir / "input.json").read_text())
    assert {item["artifact_type"] for item in payload["source_artifacts"]} == {
        "md_trajectory",
        "md_final_structure",
    }
    assert all(not Path(item["path"]).is_absolute() for item in payload["source_artifacts"])
    staged_result = json.loads((queued.run_dir / "source_result.json").read_text())
    assert staged_result["md_result"]["output_files"]["production_trajectory"].startswith(
        "/source/"
    )
    command = queued.metadata["queued_command"]
    assert f"{source.run_dir}:/source:ro" in command
    assert f"{prep_dir}:/prepared-system:ro" in command
    assert f"{queued.run_dir}:/output" in command
    assert {
        path.relative_to(source.run_dir): path.read_bytes()
        for path in source.run_dir.rglob("*")
        if path.is_file()
    } == original_files


def test_md_workflow_creates_and_aggregates_endpoint_jobs_per_replica(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source_job, source_artifact = _source_job(runs_dir)
    workflow = create_md_simulation(
        source_job=source_job,
        source_artifact=source_artifact,
        prep_input=_prep_input(
            source_artifact.resolve(source_job.run_dir, must_exist=True)
        ),
        production={
            "production_steps": 1000,
            "production_report_interval": 100,
            "continuation_mode": INDEPENDENT_REPLICA,
            "replica_equilibration_steps": 250000,
            "endpoint_enabled": True,
            "endpoint_backend": "openmm_gbsa",
            "endpoint_start_pct": 25,
            "endpoint_end_pct": 100,
            "endpoint_stride": 2,
        },
        replicas=2,
        analysis_enabled=True,
        use_gpu=False,
    )
    prep_ref = next(
        item
        for item in workflow.children
        if item.step_id == "preparation_equilibration"
    )
    prep_dir = runs_dir / prep_ref.task_group / prep_ref.run_id
    prep_output = prep_dir / prep_ref.run_id
    prep_output.mkdir()
    prep_files = {
        "system_pdb": prep_output / "system.pdb",
        "npt_pdb": prep_output / "npt.pdb",
        "npt_checkpoint": prep_output / "npt.chk",
        "npt_state_xml": prep_output / "state.xml",
        "npt_system_xml": prep_output / "system.xml",
        "npt_integrator_xml": prep_output / "integrator.xml",
    }
    for path in prep_files.values():
        path.write_text("prepared")
    _write_json(
        prep_dir / "result.json",
        {
            "success": True,
            "md_result": {
                "equilibration_stats": {
                    "preparation_protocol": {
                        "protocol": "roe_brooks_2020",
                        "density_stabilization": {
                            "fit": {"plateau": True}
                        },
                    }
                },
                "output_files": {
                    key: f"/output/{prep_ref.run_id}/{path.name}"
                    for key, path in prep_files.items()
                }
            },
        },
    )
    workflow = advance_md_workflow(workflow.workflow_id)
    production_refs = [
        item
        for item in workflow.children
        if item.step_id.startswith("production_replica_")
    ]
    for production_offset, child_ref in enumerate(production_refs):
        run_dir = runs_dir / child_ref.task_group / child_ref.run_id
        production_input = json.loads((run_dir / "input.json").read_text())
        assert production_input["mmgbsa_enabled"] is False
        output_dir = run_dir / child_ref.run_id
        output_dir.mkdir()
        trajectory = output_dir / "production.dcd"
        final_pdb = output_dir / "production.pdb"
        trajectory.write_text("trajectory")
        final_pdb.write_text("structure")
        _write_json(
            run_dir / "result.json",
            {
                "success": True,
                "md_result": {
                    "output_files": {
                        "production_trajectory": (
                            f"/output/{child_ref.run_id}/{trajectory.name}"
                        ),
                        "production_pdb": (
                            f"/output/{child_ref.run_id}/{final_pdb.name}"
                        ),
                    },
                    "analytics": {
                        "performance": {
                            "ns_per_day": 1000.0 + 200.0 * production_offset,
                            "source": "OpenMM StateDataReporter",
                            "sample_count": 2,
                        },
                        "rmsd": {
                            "backbone_rmsd_angstrom": [1.0, 1.5],
                            "ligand_rmsd_angstrom": [2.0, 2.5],
                        },
                        "structural_dynamics": {
                            "time_ps": [0.0, 10.0],
                            "pocket": {
                                "retained_fraction": 0.75,
                                "ligand_centroid_displacement_angstrom": [
                                    0.0,
                                    3.0,
                                ],
                                "minimum_protein_distance_angstrom": [
                                    2.0,
                                    2.5,
                                ],
                            },
                            "rmsf": {
                                "residues": ["ALA1 · chain A"],
                                "ca_rmsf_angstrom": [1.2],
                            },
                            "contacts": {
                                "residues": (
                                    [
                                        {
                                            "residue": "ALA1 · chain A",
                                            "contact_occupancy": 0.8,
                                            "hydrogen_bond_occupancy": 0.4,
                                        }
                                    ]
                                    if production_offset == 0
                                    else []
                                )
                            },
                        },
                    },
                },
            },
        )

    workflow = advance_md_workflow(workflow.workflow_id)
    endpoint_refs = [
        item
        for item in workflow.children
        if item.step_id.startswith("endpoint_energy_replica_")
    ]
    assert len(endpoint_refs) == 2
    assert {
        JobRecord.load(
            runs_dir / item.task_group / item.run_id,
            task_group=item.task_group,
        ).metadata["source_production_run_id"]
        for item in endpoint_refs
    } == {item.run_id for item in production_refs}

    expected_energies = [-12.0, -18.0]
    for child_ref, energy in zip(
        sorted(endpoint_refs, key=lambda item: item.step_id),
        expected_energies,
    ):
        run_dir = runs_dir / child_ref.task_group / child_ref.run_id
        _write_json(
            run_dir / "result.json",
            {
                "success": True,
                "mmgbsa": {
                    "status": "success",
                    "method": "OpenMM GBSA endpoint estimate",
                    "metadata": {"n_frames_analyzed": 50},
                    "delta": {
                        "delta_g_bind_total_kcal_mol": energy,
                        "delta_mm_kcal_mol": energy - 1.0,
                        "delta_gbsa_kcal_mol": 1.0,
                        "delta_nonpolar_kcal_mol": -0.5,
                    },
                },
            },
        )
        finalize_mmgbsa_analysis_job(run_dir, returncode=0)

    completed = advance_md_workflow(workflow.workflow_id)
    assert completed.status == "completed"
    analysis_ref = next(
        item
        for item in completed.children
        if item.step_id == "replicate_analysis"
    )
    report = json.loads(
        (
            runs_dir
            / analysis_ref.task_group
            / analysis_ref.run_id
            / "replicate_summary.json"
        ).read_text()
    )
    binding = report["aggregate"]["delta_g_bind_kcal_mol"]
    assert binding["count"] == 2
    assert binding["mean"] == -15.0
    assert round(binding["sample_sd"], 6) == round(3 * 2**0.5, 6)
    assert report["endpoint_completed"] == 2
    assert report["endpoint_failed"] == 0
    assert report["aggregate"]["reference_site_retained_fraction"][
        "count"
    ] == 2
    assert len(report["replica_series"]) == 2
    assert report["contact_consensus"][0]["mean_contact_occupancy"] == 0.4
    assert report["contact_consensus"][0]["replica_count"] == 2
    assert report["rmsf_consensus"][0]["mean_ca_rmsf_angstrom"] == 1.2
    performance = report["aggregate"]["performance_ns_per_day"]
    assert performance["count"] == 2
    assert performance["mean"] == 1100.0
    assert round(performance["sample_sd"], 6) == round(100 * 2**0.5, 6)


def test_worker_finalizes_reusable_mmgbsa_outputs_without_mutating_source(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source = _completed_production(runs_dir)
    original_result = (source.run_dir / "result.json").read_bytes()
    queued = create_mmgbsa_analysis_job(source.run_id, use_gpu=False)

    class ImmediateProcess:
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    def fake_popen(_command: list[str], **kwargs: Any) -> ImmediateProcess:
        _write_json(
            queued.run_dir / "result.json",
            {
                "success": True,
                "mmgbsa": {
                    "status": "success",
                    "method": "OpenMM GBSA endpoint estimate",
                    "trajectory_path": "/source/production-1/production.dcd",
                    "delta": {"delta_g_bind_total_kj_mol": -12.5},
                    "start_pct": 20.0,
                    "end_pct": 100.0,
                },
            },
        )
        (queued.run_dir / "mmgbsa_frames.csv").write_text("frame,total\n1,-12.5\n")
        kwargs["stdout"].write("native MM/GBSA complete\n")
        return ImmediateProcess()

    worker_result = run_worker_once(
        WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(), heartbeat_seconds=0.05),
        popen=fake_popen,
        sleep=lambda _: None,
    )

    assert worker_result is not None and worker_result["status"] == "completed"
    completed = JobRecord.load(queued.run_dir, task_group="md-mmgbsa")
    assert completed.result["mmgbsa"]["status"] == "success"
    assert completed.result["mmgbsa"]["trajectory_path"] == "production-1/production.dcd"
    assert completed.artifact_manifest is not None
    assert len(completed.artifact_manifest.by_type("endpoint_energy")) == 2
    assert completed.artifact_manifest.by_type("native_output")
    assert (source.run_dir / "result.json").read_bytes() == original_result


def test_worker_marks_native_mmgbsa_failure_failed(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs_dir))
    source = _completed_production(runs_dir)
    queued = create_mmgbsa_analysis_job(source.run_id, use_gpu=False)

    class ImmediateProcess:
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    def fake_popen(_command: list[str], **_kwargs: Any) -> ImmediateProcess:
        _write_json(
            queued.run_dir / "result.json",
            {"success": False, "mmgbsa": {"status": "failed", "error": "native failure"}},
        )
        return ImmediateProcess()

    worker_result = run_worker_once(
        WorkerConfig.create(runs_dir=runs_dir, gpu_ids=(), heartbeat_seconds=0.05),
        popen=fake_popen,
        sleep=lambda _: None,
    )

    assert worker_result is not None and worker_result["status"] == "failed"
    failed = JobRecord.load(queued.run_dir, task_group="md-mmgbsa")
    assert failed.status == "failed"
    assert failed.result["error"] == "native failure"
