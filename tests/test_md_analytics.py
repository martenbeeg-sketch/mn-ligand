from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mn_ligand.ligandx.services.md.workflow.analytics import (
    EquilibrationAnalytics,
    _secondary_structure_fractions,
    ligand_formal_charges_from_sdf_data,
)


def _small_protein_ligand_trajectory(tmp_path: Path) -> tuple[Path, Path]:
    md = pytest.importorskip("mdtraj")
    topology = md.Topology()
    chain = topology.add_chain("A")
    alanine = topology.add_residue("ALA", chain, resSeq=1)
    n_atom = topology.add_atom("N", md.element.nitrogen, alanine)
    ca_atom = topology.add_atom("CA", md.element.carbon, alanine)
    c_atom = topology.add_atom("C", md.element.carbon, alanine)
    o_atom = topology.add_atom("O", md.element.oxygen, alanine)
    cb_atom = topology.add_atom("CB", md.element.carbon, alanine)
    topology.add_bond(n_atom, ca_atom)
    topology.add_bond(ca_atom, c_atom)
    topology.add_bond(c_atom, o_atom)
    topology.add_bond(ca_atom, cb_atom)
    ligand = topology.add_residue("LIG", chain, resSeq=501)
    ligand_c = topology.add_atom("C1", md.element.carbon, ligand)
    ligand_o = topology.add_atom("O1", md.element.oxygen, ligand)
    topology.add_bond(ligand_c, ligand_o)

    xyz = np.asarray(
        [
            [
                [0.00, 0.00, 0.00],
                [0.10, 0.00, 0.00],
                [0.20, 0.00, 0.00],
                [0.25, 0.05, 0.00],
                [0.25, 0.00, 0.00],
                [0.35, 0.00, 0.00],
                [0.40, 0.00, 0.00],
            ],
            [
                [0.00, 0.00, 0.00],
                [0.10, 0.00, 0.00],
                [0.20, 0.00, 0.00],
                [0.25, 0.05, 0.00],
                [0.25, 0.00, 0.00],
                [0.45, 0.00, 0.00],
                [0.50, 0.00, 0.00],
            ],
            [
                [0.00, 0.00, 0.00],
                [0.10, 0.00, 0.00],
                [0.20, 0.00, 0.00],
                [0.25, 0.05, 0.00],
                [0.25, 0.00, 0.00],
                [1.05, 0.00, 0.00],
                [1.10, 0.00, 0.00],
            ],
        ],
        dtype=np.float32,
    )
    trajectory = md.Trajectory(xyz=xyz, topology=topology)
    topology_path = tmp_path / "system.pdb"
    trajectory_path = tmp_path / "production.dcd"
    trajectory[0].save_pdb(str(topology_path))
    trajectory.save_dcd(str(trajectory_path))
    return topology_path, trajectory_path


def _protein_ligand_water_trajectory(
    tmp_path: Path,
) -> tuple[Path, Path]:
    md = pytest.importorskip("mdtraj")
    topology = md.Topology()
    chain = topology.add_chain("A")
    serine = topology.add_residue("SER", chain, resSeq=10)
    protein_n = topology.add_atom("N", md.element.nitrogen, serine)
    protein_h = topology.add_atom("H", md.element.hydrogen, serine)
    protein_ca = topology.add_atom("CA", md.element.carbon, serine)
    topology.add_bond(protein_n, protein_h)
    topology.add_bond(protein_n, protein_ca)
    ligand = topology.add_residue("LIG", chain, resSeq=501)
    ligand_o = topology.add_atom("O1", md.element.oxygen, ligand)
    water = topology.add_residue("HOH", chain, resSeq=600)
    water_o = topology.add_atom("O", md.element.oxygen, water)
    water_h1 = topology.add_atom("H1", md.element.hydrogen, water)
    water_h2 = topology.add_atom("H2", md.element.hydrogen, water)
    topology.add_bond(water_o, water_h1)
    topology.add_bond(water_o, water_h2)

    frame = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.30, 0.00, 0.00],
            [0.20, 0.00, 0.00],
            [0.25, 0.00, 0.00],
            [0.20, 0.05, 0.00],
        ],
        dtype=np.float32,
    )
    trajectory = md.Trajectory(
        xyz=np.stack((frame.copy(), frame.copy())),
        topology=topology,
    )
    trajectory.xyz[1, :3, 0] -= 1.0
    topology_path = tmp_path / "water-system.pdb"
    trajectory_path = tmp_path / "water-production.dcd"
    trajectory[0].save_pdb(str(topology_path))
    trajectory.save_dcd(str(trajectory_path))
    return topology_path, trajectory_path


def _charged_protein_ligand_trajectory(
    tmp_path: Path,
) -> tuple[Path, Path]:
    md = pytest.importorskip("mdtraj")
    topology = md.Topology()
    chain = topology.add_chain("A")
    aspartate = topology.add_residue("ASP", chain, resSeq=10)
    protein_atoms = [
        topology.add_atom(name, element, aspartate)
        for name, element in (
            ("N", md.element.nitrogen),
            ("CA", md.element.carbon),
            ("C", md.element.carbon),
            ("O", md.element.oxygen),
            ("CB", md.element.carbon),
            ("CG", md.element.carbon),
            ("OD1", md.element.oxygen),
            ("OD2", md.element.oxygen),
        )
    ]
    for left, right in zip(protein_atoms, protein_atoms[1:]):
        topology.add_bond(left, right)
    ligand = topology.add_residue("LIG", chain, resSeq=501)
    topology.add_atom("N1", md.element.nitrogen, ligand)
    base = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.20, 0.00, 0.00],
            [0.25, 0.05, 0.00],
            [0.15, 0.10, 0.00],
            [0.20, 0.15, 0.00],
            [0.25, 0.20, 0.00],
            [0.20, 0.25, 0.00],
            [0.25, 0.50, 0.00],
        ],
        dtype=np.float32,
    )
    xyz = np.stack((base.copy(), base.copy()))
    xyz[0, -1] = [0.25, 0.50, 0.00]
    xyz[1, -1] = [0.25, 0.70, 0.00]
    trajectory = md.Trajectory(xyz=xyz, topology=topology)
    topology_path = tmp_path / "charged-system.pdb"
    trajectory_path = tmp_path / "charged-production.dcd"
    trajectory[0].save_pdb(str(topology_path))
    trajectory.save_dcd(str(trajectory_path))
    return topology_path, trajectory_path


def test_ligand_formal_charges_are_named_by_element_order() -> None:
    chemistry = pytest.importorskip("rdkit.Chem")
    molecule = chemistry.MolFromSmiles("[NH4+].[O-]C=O")

    charges = ligand_formal_charges_from_sdf_data(
        chemistry.MolToMolBlock(molecule)
    )

    assert charges == {"N1": 1, "O1": -1}


def test_structural_dynamics_reports_strict_salt_bridge_occupancy(
    tmp_path: Path,
) -> None:
    topology_path, trajectory_path = _charged_protein_ligand_trajectory(
        tmp_path
    )

    result = EquilibrationAnalytics()._compute_structural_dynamics(
        topology_pdb=str(topology_path),
        production_traj=str(trajectory_path),
        ligand_resname="LIG",
        production_report_interval=2500,
        dt_ps=0.004,
        ligand_formal_charges={"N1": 1},
    )

    contact = result["contacts"]["residues"][0]
    assert contact["residue"] == "ASP10 · chain A"
    assert contact["salt_bridge_occupancy"] == pytest.approx(0.5)
    assert contact["salt_bridge_backbone_occupancy"] == 0.0
    assert contact["salt_bridge_sidechain_occupancy"] == pytest.approx(0.5)
    assert contact["binding_importance_score"] == pytest.approx(1.25)
    assert contact["binding_importance_backbone_score"] == 0.0
    assert contact["binding_importance_sidechain_score"] == pytest.approx(
        1.25
    )
    assert result["contacts"]["interaction_hotspot_score"] == {
        "description": (
            "Weighted interaction occupancy; not a binding-energy estimate"
        ),
        "weights": {
            "hydrogen_bond": 3.0,
            "salt_bridge": 2.5,
            "water_bridge": 1.5,
            "hydrophobic": 1.0,
        },
    }
    assert result["salt_bridges"] == {
        "applicable": True,
        "ligand_charged_atoms": [
            {"atom": "N1", "formal_charge": 1}
        ],
        "residues": [
            {"residue": "ASP10 · chain A", "occupancy": 0.5}
        ],
        "distance_cutoff_angstrom": 4.0,
    }


def test_water_bridge_requires_two_hydrogen_bonds_to_same_water(
    tmp_path: Path,
) -> None:
    topology_path, trajectory_path = _protein_ligand_water_trajectory(
        tmp_path
    )

    result = EquilibrationAnalytics()._compute_structural_dynamics(
        topology_pdb=str(topology_path),
        production_traj=str(trajectory_path),
        ligand_resname="LIG",
        production_report_interval=2500,
        dt_ps=0.004,
    )

    contact = result["contacts"]["residues"][0]
    assert contact["water_bridge_occupancy"] == pytest.approx(0.5)
    assert contact["water_bridge_backbone_occupancy"] == pytest.approx(0.5)
    assert contact["water_bridge_sidechain_occupancy"] == 0.0
    assert contact["hydrogen_bond_backbone_occupancy"] == pytest.approx(0.5)
    assert contact["hydrogen_bond_sidechain_occupancy"] == 0.0
    assert contact["binding_importance_score"] == pytest.approx(2.25)
    assert contact["binding_importance_backbone_score"] == pytest.approx(
        2.25
    )
    assert contact["binding_importance_sidechain_score"] == 0.0
    assert result["water_bridges"] == {
        "residues": [{"residue": "SER10 · chain A", "occupancy": 0.5}],
        "method": "Wernet-Nilsson two-sided hydrogen-bond geometry",
    }


def test_secondary_structure_fractions_exclude_nonprotein_assignments() -> None:
    fractions = _secondary_structure_fractions(
        np.asarray(
            [
                ["H", "H", "E", "C", "NA", "NA"],
                ["H", "E", "E", "C", "NA", "NA"],
            ]
        )
    )

    assert fractions["helix_fraction"] == [0.5, 0.25]
    assert fractions["sheet_fraction"] == [0.25, 0.5]
    assert fractions["coil_fraction"] == [0.25, 0.25]
    assert [
        sum(values)
        for values in zip(
            fractions["helix_fraction"],
            fractions["sheet_fraction"],
            fractions["coil_fraction"],
            strict=True,
        )
    ] == [1.0, 1.0]


def test_structural_dynamics_reports_rmsf_retention_and_contacts(
    tmp_path: Path,
) -> None:
    topology_path, trajectory_path = _small_protein_ligand_trajectory(
        tmp_path
    )

    result = EquilibrationAnalytics()._compute_structural_dynamics(
        topology_pdb=str(topology_path),
        production_traj=str(trajectory_path),
        ligand_resname="LIG",
        production_report_interval=2500,
        dt_ps=0.004,
    )

    assert result["warnings"] == []
    assert result["time_ps"] == [0.0, 10.0, 20.0]
    assert result["rmsf"]["residues"] == ["ALA1 · chain A"]
    assert len(result["rmsf"]["ca_rmsf_angstrom"]) == 1
    assert result["ligand_rmsf"]["rmsf_angstrom"] == pytest.approx(
        [3.0912, 3.0912], abs=1e-3
    )
    assert result["pocket"]["reference_site_retained"] == [
        True,
        True,
        False,
    ]
    assert result["pocket"]["retained_fraction"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["contacts"]["residues"][0]["residue"] == "ALA1 · chain A"
    assert result["contacts"]["residues"][0][
        "contact_occupancy"
    ] == pytest.approx(2 / 3, abs=1e-4)
    assert result["contacts"]["residues"][0][
        "contact_sidechain_occupancy"
    ] == pytest.approx(2 / 3, abs=1e-4)
    assert result["contacts"]["residues"][0][
        "contact_backbone_occupancy"
    ] == pytest.approx(2 / 3, abs=1e-4)
    assert result["contacts"]["residues"][0]["top_ligand_atom"] == "C1"
    assert result["contacts"]["residues"][0][
        "hydrophobic_occupancy"
    ] == pytest.approx(2 / 3, abs=1e-4)
    assert result["contacts"]["residues"][0][
        "hydrophobic_sidechain_occupancy"
    ] == pytest.approx(2 / 3, abs=1e-4)
    assert result["contacts"]["residues"][0][
        "hydrophobic_backbone_occupancy"
    ] == 0.0
    assert result["interaction_network"]["edges"][0]["source"] == "C1"
    assert result["interface_rin"]["applicable"] is False


def test_structural_dynamics_excludes_requested_analysis_prefix(
    tmp_path: Path,
) -> None:
    topology_path, trajectory_path = _small_protein_ligand_trajectory(
        tmp_path
    )

    result = EquilibrationAnalytics()._compute_structural_dynamics(
        topology_pdb=str(topology_path),
        production_traj=str(trajectory_path),
        ligand_resname="LIG",
        production_report_interval=2500,
        dt_ps=0.004,
        analysis_start_ps=10.0,
    )

    assert result["time_ps"] == [10.0, 20.0]
    assert result["analysis_window"] == {
        "start_ps": 10.0,
        "discarded_frames": 1,
        "analyzed_frames": 2,
        "trajectory_stride": 1,
    }
    assert result["pocket"]["reference_site_retained"] == [True, False]
    assert result["contacts"]["residues"][0][
        "contact_occupancy"
    ] == pytest.approx(0.5)


def test_production_rmsd_window_references_first_retained_frame(
    tmp_path: Path,
) -> None:
    topology_path, trajectory_path = _small_protein_ligand_trajectory(
        tmp_path
    )

    result = EquilibrationAnalytics().compute_production_rmsd_window(
        topology_pdb=str(topology_path),
        production_traj=str(trajectory_path),
        ligand_resname="LIG",
        production_report_interval=2500,
        dt_ps=0.004,
        analysis_start_ps=10.0,
    )

    assert result["time_ps"] == [10.0, 20.0]
    assert result["backbone_rmsd_angstrom"][0] == pytest.approx(0.0)
    assert result["ligand_rmsd_angstrom"][0] == pytest.approx(0.0)
    assert result["analysis_window"]["reference"] == (
        "first retained production frame"
    )
    assert result["analysis_window"]["reference_time_ps"] == 10.0
    assert result["analysis_window"]["reference_source_frame_index"] == 1
    assert result["analysis_window"]["reference_coordinates"] == (
        "production trajectory frame"
    )
    assert (
        result["analysis_window"]["topology_coordinates_used_as_reference"]
        is False
    )


def test_structural_dynamics_uses_source_author_residue_mapping(
    tmp_path: Path,
) -> None:
    topology_path, trajectory_path = _small_protein_ligand_trajectory(
        tmp_path
    )
    mapping = {
        "source_run_id": "imported-4lnw",
        "source_label": "4LNW",
        "residues": [
            {
                "structure_residue_name": "ALA",
                "native_residue_name": "ALA",
                "native_chain": "A",
                "native_residue_number": 277,
                "native_insertion_code": "",
            }
        ],
    }

    result = EquilibrationAnalytics()._compute_structural_dynamics(
        topology_pdb=str(topology_path),
        production_traj=str(trajectory_path),
        ligand_resname="LIG",
        production_report_interval=2500,
        dt_ps=0.004,
        residue_mapping=mapping,
    )

    assert result["rmsf"]["residues"] == ["ALA277 · chain A"]
    assert result["contacts"]["residues"][0]["residue"] == (
        "ALA277 · chain A"
    )
    assert result["residue_numbering"] == {
        "scheme": "source_author",
        "source_run_id": "imported-4lnw",
        "source_label": "4LNW",
    }


def test_openmm_log_parser_preserves_engine_reported_speed(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "production.log"
    log_path.write_text(
        '#"Step"\t"Potential Energy (kJ/mole)"\t"Temperature (K)"'
        '\t"Density (g/mL)"\t"Speed (ns/day)"\n'
        "1000\t-100.0\t300.0\t1.0\t1.18e+03\n"
        "2000\t-101.0\t301.0\t1.01\t1.22e+03\n"
    )

    result = EquilibrationAnalytics()._parse_log(str(log_path))

    assert result["speed_ns_per_day"] == [1180.0, 1220.0]
