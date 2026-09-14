from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from mn_ligand.app.viewers import (
    closest_residue_ligand_atom_pair,
    dashed_line_segments,
    distribute_2d_labels,
    horizontalize_2d_coordinates,
    pdb_interaction_atom_coordinates,
    pdb_ligand_atom_aliases,
    transform_2d_coordinates,
)
from mn_ligand.app.preferences import (
    load_network_appearance,
    save_network_appearance,
)
from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.residue_mapping import sequence_author_residue_mapping
from mn_ligand.workflows.interaction_analysis import (
    INTERACTION_ENGINES,
    _author_numbering_reference,
)
from mn_ligand.workflows.native_interactions import (
    apply_author_residue_numbering,
    analyze_static_pose,
    protein_atom_scope,
)


def _write_job(
    run_dir: Path,
    *,
    workflow: str,
    task_group: str,
    **metadata: object,
) -> JobRecord:
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({
            "schema_version": 1,
            "run_id": run_dir.name,
            "job_code": run_dir.name[:5],
            "status": "completed",
            "workflow": workflow,
            "created_at": "2026-07-27T00:00:00+00:00",
            **metadata,
        })
    )
    (run_dir / "input.json").write_text("{}")
    return JobRecord.load(run_dir, task_group=task_group)


def test_cofolding_numbering_reference_prefers_imported_author_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    target = _write_job(
        runs / "structure-jobs" / "prepared-target",
        workflow="structure_import",
        task_group="structure-jobs",
        pdb_id="4LNW",
    )
    prepared = target.run_dir / "prepared.pdb"
    prepared.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
        "  1.00 20.00           C\nEND\n"
    )
    source = _write_job(
        runs / "refolding" / "cofolding-job",
        workflow="boltz2_refolding",
        task_group="refolding",
    )
    (source.run_dir / "input.json").write_text(
        json.dumps({"target": {"run_id": target.run_id}})
    )
    imported = _write_job(
        runs / "protein-import" / "imported-target",
        workflow="",
        task_group="protein-import",
        pdb_id="4LNW",
    )
    author_target = imported.run_dir / "artifacts" / "imported" / "4lnw.pdb"
    author_target.parent.mkdir(parents=True)
    author_target.write_text(
        "ATOM      1  CA  ALA A 145       0.000   0.000   0.000"
        "  1.00 20.00           C\nEND\n"
    )
    (imported.run_dir / "artifacts.json").write_text(json.dumps({
        "artifacts": [{
            "artifact_type": "imported_target",
            "path": "artifacts/imported/4lnw.pdb",
        }]
    }))

    resolved, origin = _author_numbering_reference(source, prepared)

    assert resolved == author_target
    assert "immutable imported-source author numbering" in origin


def test_numbering_reference_follows_import_lineage_for_uploaded_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    imported = _write_job(
        runs / "protein-import" / "uploaded-target",
        workflow="",
        task_group="protein-import",
        source="upload",
    )
    author_target = imported.run_dir / "artifacts" / "imported" / "target.pdb"
    author_target.parent.mkdir(parents=True)
    author_target.write_text(
        "ATOM      1  CA  ALA B 912       0.000   0.000   0.000"
        "  1.00 20.00           C\nEND\n"
    )
    (imported.run_dir / "artifacts.json").write_text(json.dumps({
        "artifacts": [{
            "artifact_type": "imported_target",
            "path": "artifacts/imported/target.pdb",
        }]
    }))
    prepared_job = _write_job(
        runs / "structure-jobs" / "prepared-upload",
        workflow="structure_import",
        task_group="structure-jobs",
        parent_run_id=imported.run_id,
    )
    prepared = prepared_job.run_dir / "prepared.pdb"
    prepared.write_text("END\n")
    source = _write_job(
        runs / "docking" / "docking-job",
        workflow="docking_campaign",
        task_group="docking",
    )
    (source.run_dir / "input.json").write_text(json.dumps({
        "target_artifact": {"run_id": prepared_job.run_id}
    }))

    resolved, origin = _author_numbering_reference(source, prepared)

    assert resolved == author_target
    assert "target.pdb job uploaded-target" in origin


def test_numbering_reference_falls_back_for_non_pdb_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    source = _write_job(
        runs / "refolding" / "custom-target",
        workflow="alphafold3_refolding",
        task_group="refolding",
    )
    prepared = source.run_dir / "custom.pdb"
    prepared.write_text("END\n")

    resolved, origin = _author_numbering_reference(source, prepared)

    assert resolved == prepared
    assert origin == "prepared target numbering"


def test_sequence_mapping_restores_author_numbers_after_trimming() -> None:
    prepared = "\n".join([
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
        "ATOM      2  CA  GLY A   2       1.000   0.000   0.000  1.00 20.00           C",
        "ATOM      3  CA  SER A   3       2.000   0.000   0.000  1.00 20.00           C",
        "END",
    ])
    author = "\n".join([
        "ATOM      1  CA  MET A 144       9.000   0.000   0.000  1.00 20.00           C",
        "ATOM      2  CA  ALA A 145      10.000   0.000   0.000  1.00 20.00           C",
        "ATOM      3  CA  GLY A 146      11.000   0.000   0.000  1.00 20.00           C",
        "ATOM      4  CA  SER A 147      12.000   0.000   0.000  1.00 20.00           C",
        "END",
    ])

    mapping = sequence_author_residue_mapping(prepared, author)

    assert mapping[("A", 1, "")]["residue_number"] == 145
    assert mapping[("A", 3, "")]["residue_number"] == 147


def test_native_atoms_keep_coordinate_ids_when_author_ids_are_applied() -> None:
    prepared = (
        "ATOM      1  OG  SER A   1       0.000   0.000   0.000"
        "  1.00 20.00           O\nEND\n"
    )
    author = (
        "ATOM      1  OG  SER A 277       5.000   0.000   0.000"
        "  1.00 20.00           O\nEND\n"
    )
    atoms = [{
        "name": "OG",
        "chain": "A",
        "residue_name": "SER",
        "residue_number": 1,
        "insertion_code": "",
    }]

    mapped = apply_author_residue_numbering(atoms, prepared, author)

    assert mapped == 1
    assert atoms[0]["residue_number"] == 277
    assert atoms[0]["coordinate_residue_number"] == 1


def test_native_md_geometry_is_available_as_interaction_engine() -> None:
    assert INTERACTION_ENGINES["Native MD geometry"]["workflow"] == (
        "native_md_geometry_interactions"
    )
    assert INTERACTION_ENGINES["Native MD geometry"]["image"] == ""


def test_ligand_coordinates_are_oriented_along_horizontal_axis() -> None:
    diagonal = np.asarray([
        [-2.0, -4.0],
        [-1.0, -2.0],
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    oriented = horizontalize_2d_coordinates(diagonal)

    extent = np.ptp(oriented, axis=0)
    distances = np.linalg.norm(
        oriented[:, np.newaxis, :] - oriented[np.newaxis, :, :],
        axis=2,
    )
    start, end = np.unravel_index(int(np.argmax(distances)), distances.shape)
    assert extent[0] > extent[1]
    assert np.allclose(oriented.mean(axis=0), 0.0)
    assert np.isclose(oriented[end, 1] - oriented[start, 1], 0.0)


def test_multiring_ligand_uses_ring_centroids_as_horizontal_axis() -> None:
    coordinates = np.asarray([
        [-3.0, -1.0],
        [-2.0, 0.0],
        [-3.0, 1.0],
        [2.0, 1.0],
        [3.0, 2.0],
        [2.0, 3.0],
        [5.0, -4.0],
    ])

    oriented = horizontalize_2d_coordinates(
        coordinates,
        anchor_groups=((0, 1, 2), (3, 4, 5)),
    )

    first_center = oriented[[0, 1, 2]].mean(axis=0)
    second_center = oriented[[3, 4, 5]].mean(axis=0)
    assert np.isclose(second_center[1] - first_center[1], 0.0)


def test_ligand_orientation_controls_rotate_and_flip_coordinates() -> None:
    coordinates = np.asarray([[1.0, 2.0], [-1.0, -2.0]])

    transformed = transform_2d_coordinates(
        coordinates,
        rotation_degrees=90,
        flip_horizontal=True,
        flip_vertical=True,
    )

    assert np.allclose(transformed, [[2.0, -1.0], [-2.0, 1.0]])


def test_interaction_labels_preserve_attachment_order_on_each_side() -> None:
    anchors = np.asarray([
        [-2.0, 1.0],
        [-2.0, -1.0],
        [2.0, 0.8],
        [2.0, -0.8],
    ])

    positions = distribute_2d_labels(
        anchors,
        x_radius=3.0,
        y_radius=2.0,
    )

    assert np.all(positions[:2, 0] < 0.0)
    assert np.all(positions[2:, 0] > 0.0)
    assert positions[0, 1] > positions[1, 1]
    assert positions[2, 1] > positions[3, 1]
    assert np.allclose(
        (positions[:, 0] / 3.0) ** 2 + (positions[:, 1] / 2.0) ** 2,
        1.0,
    )


def test_3d_interaction_atom_mapping_excludes_solvent() -> None:
    pdb_text = "\n".join([
        "ATOM      1  OG  SER A 277      10.000  11.000  12.000  1.00 20.00           O",
        "HETATM    2  O1  LIG A 501      13.000  14.000  15.000  1.00 20.00           O",
        "HETATM    3  O   HOH A 900      16.000  17.000  18.000  1.00 20.00           O",
    ])

    protein, ligand = pdb_interaction_atom_coordinates(pdb_text)

    assert np.allclose(protein[("A", "277", "OG")], [10.0, 11.0, 12.0])
    assert np.allclose(protein[("A", "277", "#1")], [10.0, 11.0, 12.0])
    assert np.allclose(ligand["O1"], [13.0, 14.0, 15.0])
    assert np.allclose(ligand["#2"], [13.0, 14.0, 15.0])
    assert "O" not in ligand


def test_pdb_ligand_serial_maps_to_matching_rdkit_atom() -> None:
    molecule = Chem.MolFromSmiles("CI")
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.SetAtomPosition(0, (1.0, 2.0, 3.0))
    conformer.SetAtomPosition(1, (2.0, 2.0, 3.0))
    molecule.AddConformer(conformer)
    pdb_text = "\n".join([
        "HETATM 3953  C1  LIG A 501       1.000   2.000   3.000  1.00 20.00           C",
        "HETATM 3954  I1  LIG A 501       2.000   2.000   3.000  1.00 20.00           I",
    ])

    aliases = pdb_ligand_atom_aliases(pdb_text, molecule)

    assert aliases["#3954"] == 1
    assert aliases["I1"] == 1


def test_3d_interaction_connector_falls_back_to_nearest_atom_pair() -> None:
    protein = {
        ("A", "277", "OG"): np.asarray([0.0, 0.0, 0.0]),
        ("A", "277", "CB"): np.asarray([3.0, 0.0, 0.0]),
        ("A", "278", "CA"): np.asarray([9.0, 0.0, 0.0]),
    }
    ligand = {
        "C1": np.asarray([4.0, 0.0, 0.0]),
        "O1": np.asarray([8.0, 0.0, 0.0]),
    }

    pair = closest_residue_ligand_atom_pair(
        protein,
        ligand,
        chain="A",
        residue_number="277",
    )

    assert pair is not None
    assert np.allclose(pair[0], [3.0, 0.0, 0.0])
    assert np.allclose(pair[1], [4.0, 0.0, 0.0])


def test_3d_interaction_connectors_are_broken_segments() -> None:
    segments = dashed_line_segments(
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([11.0, 0.0, 0.0]),
    )

    assert len(segments) == 6
    assert np.allclose(segments[0][0], [0.0, 0.0, 0.0])
    assert np.allclose(segments[0][1], [1.0, 0.0, 0.0])
    assert np.allclose(segments[1][0], [2.0, 0.0, 0.0])
    assert np.allclose(segments[-1][1], [11.0, 0.0, 0.0])


def test_network_appearance_is_saved_as_global_ui_preference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MN_LIGAND_APP_HOME", str(tmp_path))

    target = save_network_appearance({
        "molecule_scale": 1.25,
        "ellipse_width_scale": 0.85,
        "residue_circle_size": 420,
    })
    loaded = load_network_appearance()

    assert target == tmp_path / "config" / "ui_preferences.json"
    assert loaded["molecule_scale"] == 1.25
    assert loaded["ellipse_width_scale"] == 0.85
    assert loaded["residue_circle_size"] == 420
    assert loaded["legend_spacing"] == 0.06


def test_static_geometry_reports_backbone_and_side_chain_without_occupancy() -> None:
    ligand = Chem.MolFromSmiles("[NH4+]")
    protein_atoms = [
        {
            "name": "OD1",
            "element": "O",
            "xyz": np.asarray([0.0, 0.0, 0.0]),
            "chain": "A",
            "residue_name": "ASP",
            "residue_number": 145,
            "insertion_code": "",
        },
        {
            "name": "O",
            "element": "O",
            "xyz": np.asarray([0.0, 3.2, 0.0]),
            "chain": "A",
            "residue_name": "ALA",
            "residue_number": 146,
            "insertion_code": "",
        },
    ]
    ligand_atoms = [{
        "index": 0,
        "name": "N1",
        "element": "N",
        "formal_charge": 1,
        "xyz": np.asarray([0.0, 0.0, 3.0]),
    }]

    rows = analyze_static_pose(
        protein_atoms,
        ligand_atoms,
        ligand_molecule=ligand,
    )

    assert protein_atom_scope("OD1") == "SC"
    assert protein_atom_scope("O") == "BB"
    assert any(
        row["interaction_type"] == "salt bridge"
        and row["protein_atom_scope"] == "SC"
        for row in rows
    )
    assert any(
        row["interaction_type"] == "hydrogen bond"
        and row["protein_atom_scope"] == "SC"
        for row in rows
    )
    assert all(row["present"] is True for row in rows)
    assert all("occupancy" not in row for row in rows)
    assert not any(
        row["interaction_type"] == "water bridge" for row in rows
    )
