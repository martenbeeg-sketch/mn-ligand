from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from mn_ligand.workflows.complex_datasets import (
    source_stereochemistry_report,
)
from mn_ligand.workflows.md_candidate_selection import (
    apply_reference_bend_penalty,
    infer_md_hypothesis,
    infer_static_hypothesis,
    ligand_bend_index,
    normalize_interaction_type,
    score_candidate_poses,
    select_best_candidates,
)


def _embedded_pose(smiles: str) -> Chem.Mol:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=20260806) == 0
    return Chem.RemoveHs(molecule)


def test_source_stereochemistry_gate_accepts_matching_pose() -> None:
    smiles = "C[C@H](O)F"
    report = source_stereochemistry_report(
        smiles,
        _embedded_pose(smiles),
    )

    assert report["matches"] is True
    assert report["defined_source_centre_count"] == 1


def test_source_stereochemistry_gate_rejects_mirrored_pose() -> None:
    smiles = "C[C@H](O)F"
    mirrored = _embedded_pose(smiles)
    conformer = mirrored.GetConformer()
    for atom_index in range(mirrored.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        conformer.SetAtomPosition(atom_index, (-point.x, point.y, point.z))

    report = source_stereochemistry_report(smiles, mirrored)

    assert report["matches"] is False
    assert report["source_smiles"] != report["geometry_smiles"]


def test_source_stereochemistry_gate_allows_achiral_source() -> None:
    smiles = "CCO"
    report = source_stereochemistry_report(
        smiles,
        _embedded_pose(smiles),
    )

    assert report["matches"] is True
    assert report["defined_source_centre_count"] == 0


def test_source_stereochemistry_gate_tolerates_docking_protonation_change() -> None:
    report = source_stereochemistry_report(
        "C[C@H](N)C(=O)O",
        _embedded_pose("C[C@H](N)C(=O)[O-]"),
    )

    assert report["matches"] is True
    assert report["defined_source_centre_count"] == 1


def test_pandamap_specific_pi_hydrophobics_are_grouped() -> None:
    assert normalize_interaction_type("alkyl-pi") == "hydrophobic contact"
    assert normalize_interaction_type("carbon_pi") == "hydrophobic contact"
    assert normalize_interaction_type("pi alkyl") == "hydrophobic contact"
    assert normalize_interaction_type("aromatic-face") == "pi stacking"


def test_static_hypothesis_is_reference_derived_and_omits_water_bridges() -> None:
    rows = pd.DataFrame(
        [
            {
                "interaction_type": "hydrogen bond",
                "protein_chain": "A",
                "protein_residue_name": "SER",
                "protein_residue_number": 10,
                "protein_atom_scope": "SC",
                "analysis_engine": "PLIP",
            },
            {
                "interaction_type": "hydrogen bond",
                "protein_chain": "A",
                "protein_residue_name": "SER",
                "protein_residue_number": 10,
                "protein_atom_scope": "SC",
                "analysis_engine": "PandaMap",
            },
            {
                "interaction_type": "water bridge",
                "protein_chain": "A",
                "protein_residue_name": "ASN",
                "protein_residue_number": 11,
                "protein_atom_scope": "SC",
                "analysis_engine": "PLIP",
            },
        ]
    )
    hypothesis = infer_static_hypothesis(
        rows, hydrogen_bond_region="BB+SC"
    )
    assert hypothesis["Protein residue"].tolist() == ["A:SER10"]
    assert hypothesis["Protein region"].tolist() == ["SC"]
    assert hypothesis["Reference support"].tolist() == [2]


def test_md_hypothesis_uses_direct_occupancy_not_water_bridge_occupancy() -> None:
    hypothesis = infer_md_hypothesis(
        [
            {
                "residue": "ASN20 · chain B",
                "mean_hydrogen_bond_occupancy": 0.6,
                "mean_hydrogen_bond_backbone_occupancy": 0.0,
                "mean_hydrogen_bond_sidechain_occupancy": 0.6,
                "mean_hydrophobic_occupancy": 0.0,
                "mean_salt_bridge_occupancy": 0.0,
                "mean_water_bridge_occupancy": 0.9,
            },
            {
                "residue": "SER21 · chain B",
                "mean_hydrogen_bond_occupancy": 0.0,
                "mean_hydrophobic_occupancy": 0.0,
                "mean_salt_bridge_occupancy": 0.0,
                "mean_water_bridge_occupancy": 0.8,
            },
            {
                "residue": "LEU22 · chain B",
                "mean_hydrogen_bond_occupancy": 0.0,
                "mean_hydrophobic_occupancy": 0.7,
                "mean_hydrophobic_backbone_occupancy": 0.2,
                "mean_hydrophobic_sidechain_occupancy": 0.5,
                "mean_salt_bridge_occupancy": 0.0,
                "mean_water_bridge_occupancy": 0.0,
            },
        ],
        minimum_occupancy=0.1,
    )
    assert hypothesis["Protein residue"].tolist() == [
        "B:ASN20",
        "B:LEU22",
    ]
    assert hypothesis["Interaction"].tolist() == [
        "hydrogen bond",
        "hydrophobic contact",
    ]
    assert hypothesis["Protein region"].tolist() == ["SC", "BB+SC"]
    assert "water bridge" not in set(hypothesis["Interaction"])


def test_pose_scoring_honors_region_required_contacts_and_tool_support() -> None:
    hypothesis = pd.DataFrame(
        [
            {
                "Enabled": True,
                "Required": True,
                "Interaction": "hydrogen bond",
                "Protein residue": "A:SER10",
                "Protein region": "SC",
                "Importance": 2.0,
            },
            {
                "Enabled": True,
                "Required": False,
                "Interaction": "hydrophobic contact",
                "Protein residue": "A:LEU12",
                "Protein region": "BB+SC",
                "Importance": 1.0,
            },
        ]
    )
    identity = {
        "source_run_id": "run",
        "source_engine": "GNINA",
        "target_run_id": "target",
        "compound_id": "CMP",
        "replicate": 1,
        "prediction": "pose 1",
        "selection_criterion": "CNN",
    }
    rows = pd.DataFrame(
        [
            {
                **identity,
                "pose_id": "pose-1",
                "interaction_type": "hydrogen bond",
                "protein_chain": "A",
                "protein_residue_name": "SER",
                "protein_residue_number": 10,
                "protein_atom_scope": "SC",
                "analysis_engine": tool,
            }
            for tool in ("PLIP", "PandaMap")
        ]
        + [
            {
                **identity,
                "pose_id": "pose-1",
                "interaction_type": "hydrophobic contact",
                "protein_chain": "A",
                "protein_residue_name": "LEU",
                "protein_residue_number": 12,
                "protein_atom_scope": "SC",
                "analysis_engine": "PLIP",
            },
            {
                **identity,
                "pose_id": "pose-2",
                "interaction_type": "hydrogen bond",
                "protein_chain": "A",
                "protein_residue_name": "SER",
                "protein_residue_number": 10,
                "protein_atom_scope": "BB",
                "analysis_engine": "PLIP",
            },
            *[
                {
                    **identity,
                    "pose_id": "pose-3",
                    "interaction_type": "pi stacking",
                    "protein_chain": "A",
                    "protein_residue_name": "PHE",
                    "protein_residue_number": 14,
                    "protein_atom_scope": "SC",
                    "analysis_engine": tool,
                }
                for tool in ("PLIP", "PandaMap")
            ],
        ]
    )
    scored = score_candidate_poses(
        rows, hypothesis, minimum_detector_support=2
    ).set_index("pose_id")
    assert scored.loc["pose-1", "Reference similarity (%)"] == 66.67
    assert bool(scored.loc["pose-1", "Required interactions met"])
    assert not bool(scored.loc["pose-2", "Required interactions met"])
    assert scored.loc["pose-3", "Reference similarity (%)"] == 0.0
    assert not bool(scored.loc["pose-3", "Required interactions met"])
    selected = select_best_candidates(scored.reset_index())
    assert selected["pose_id"].tolist() == ["pose-1"]


def test_one_per_compound_keeps_only_one_source_when_pose_ids_repeat() -> None:
    scored = pd.DataFrame(
        [
            {
                "compound_id": "CMP",
                "source_run_id": "run-b",
                "source_engine": "Vina",
                "pose_id": "CMP__replicate_001__pose_001",
                "Reference similarity (%)": 83.3,
                "Required interactions met": True,
                "Interaction tool count": 2,
            },
            {
                "compound_id": "CMP",
                "source_run_id": "run-a",
                "source_engine": "GNINA",
                "pose_id": "CMP__replicate_001__pose_001",
                "Reference similarity (%)": 83.3,
                "Required interactions met": True,
                "Interaction tool count": 2,
            },
        ]
    )

    selected = select_best_candidates(scored, mode="One per compound")

    assert len(selected) == 1
    assert selected.iloc[0]["source_run_id"] == "run-a"


def test_ligand_bend_index_is_pose_frame_and_size_invariant() -> None:
    line = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    translated_scaled = [
        [10, -4, 2],
        [10, -1, 2],
        [10, 2, 2],
        [10, 5, 2],
    ]
    bent = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]

    assert ligand_bend_index(line) == 0.0
    assert ligand_bend_index(translated_scaled) == 0.0
    assert ligand_bend_index(bent) > 0.6


def test_reference_bend_penalty_changes_automatic_pose_ranking() -> None:
    scored = pd.DataFrame(
        [
            {
                "compound_id": "CMP",
                "source_run_id": "run-straight",
                "source_engine": "Vina",
                "pose_id": "straight",
                "Reference similarity (%)": 80.0,
                "Required interactions met": True,
                "Interaction tool count": 2,
            },
            {
                "compound_id": "CMP",
                "source_run_id": "run-bent",
                "source_engine": "GNINA",
                "pose_id": "bent",
                "Reference similarity (%)": 85.0,
                "Required interactions met": True,
                "Interaction tool count": 2,
            },
        ]
    )
    penalized = apply_reference_bend_penalty(
        scored,
        reference_bend_index=0.05,
        candidate_bend_indices={
            ("run-straight", "straight"): 0.08,
            ("run-bent", "bent"): 0.35,
        },
        tolerance=0.05,
        penalty_points_per_0_1=10.0,
    )

    selected = select_best_candidates(
        penalized,
        score_column="Selection score (%)",
    )

    assert penalized.loc[1, "Bend penalty (percentage points)"] == 25.0
    assert selected["pose_id"].tolist() == ["straight"]


def test_any_requirement_group_accepts_either_required_interaction() -> None:
    hypothesis = pd.DataFrame(
        [
            {
                "Enabled": True,
                "Required": True,
                "Interaction": "hydrogen bond",
                "Protein residue": residue,
                "Protein region": "SC",
                "Requirement group": "alternative-anchor",
                "Requirement logic": "ANY",
                "Importance": 1.0,
            }
            for residue in ("A:SER10", "A:ASN11")
        ]
    )
    interactions = pd.DataFrame(
        [
            {
                "source_run_id": "run",
                "source_engine": "GNINA",
                "target_run_id": "target",
                "pose_id": "pose-1",
                "compound_id": "CMP",
                "replicate": 1,
                "prediction": "pose 1",
                "selection_criterion": "CNN",
                "interaction_type": "hydrogen bond",
                "protein_chain": "A",
                "protein_residue_name": "SER",
                "protein_residue_number": 10,
                "protein_atom_scope": "SC",
                "analysis_engine": "PLIP",
            }
        ]
    )

    scored = score_candidate_poses(interactions, hypothesis)

    assert scored.iloc[0]["Reference similarity (%)"] == 50.0
    assert bool(scored.iloc[0]["Required interactions met"])
    assert scored.iloc[0]["Missing required interactions"] == ""
    assert "alternative-anchor (ANY)" in scored.iloc[0][
        "Matched required groups"
    ]


def test_all_requirement_group_rejects_pose_missing_either_interaction() -> None:
    hypothesis = pd.DataFrame(
        [
            {
                "Enabled": True,
                "Required": True,
                "Interaction": "hydrogen bond",
                "Protein residue": residue,
                "Protein region": "SC",
                "Requirement group": "mandatory-anchors",
                "Requirement logic": "ALL",
                "Importance": 1.0,
            }
            for residue in ("A:SER277", "A:HIS381")
        ]
    )
    interactions = pd.DataFrame(
        [
            {
                "source_run_id": "run",
                "source_engine": "GNINA",
                "target_run_id": "target",
                "pose_id": pose_id,
                "compound_id": compound_id,
                "replicate": 1,
                "prediction": "pose 1",
                "selection_criterion": "CNN",
                "interaction_type": "hydrogen bond",
                "protein_chain": "A",
                "protein_residue_name": residue_name,
                "protein_residue_number": residue_number,
                "protein_atom_scope": "SC",
                "analysis_engine": "PLIP",
            }
            for pose_id, compound_id, residues in (
                ("pose-both", "BOTH", (("SER", 277), ("HIS", 381))),
                ("pose-ser-only", "SER_ONLY", (("SER", 277),)),
                ("pose-his-only", "HIS_ONLY", (("HIS", 381),)),
            )
            for residue_name, residue_number in residues
        ]
    )

    scored = score_candidate_poses(interactions, hypothesis)
    selected = select_best_candidates(
        scored,
        require_required_interactions=True,
    )
    status = scored.set_index("compound_id")["Required interactions met"]

    assert bool(status["BOTH"])
    assert not bool(status["SER_ONLY"])
    assert not bool(status["HIS_ONLY"])
    assert selected["compound_id"].tolist() == ["BOTH"]
    missing = scored.set_index("compound_id")["Missing required interactions"]
    assert "HIS381" in missing["SER_ONLY"]
    assert "SER277" in missing["HIS_ONLY"]
