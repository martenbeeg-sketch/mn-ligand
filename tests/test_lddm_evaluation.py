from __future__ import annotations

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from mn_ligand.core.jobs import JobRecord
from mn_ligand.workflows.lddm_evaluation import (
    _comparison_rows,
    _fixed_frame_rmsd,
    _pdb_pose_molecule,
)


def _ethanol() -> Chem.Mol:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=11) == 0
    return Chem.RemoveHs(molecule)


def test_fixed_frame_rmsd_keeps_global_translation() -> None:
    source = _ethanol()
    translated = Chem.Mol(source)
    conformer = translated.GetConformer()
    for atom_index in range(translated.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        conformer.SetAtomPosition(
            atom_index, (point.x + 1.0, point.y, point.z)
        )

    assert _fixed_frame_rmsd(source, translated) == 1.0


def test_rosetta_pdb_residue_uses_source_ligand_chemistry(tmp_path: Path) -> None:
    molecule = _ethanol()
    pdb_lines = [
        line
        for line in Chem.MolToPDBBlock(molecule).splitlines()
        if line.startswith("HETATM")
    ]
    pose_path = tmp_path / "pose.pdb"
    pose_path.write_text("\n".join(pdb_lines + ["END", ""]))

    reconstructed = _pdb_pose_molecule(pose_path, "CCO")

    assert reconstructed is not None
    assert Chem.MolToSmiles(reconstructed, isomericSmiles=False) == "CCO"


def test_comparison_reports_rmsd_and_lddm_uncertainty(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    evaluation_dir = tmp_path / "evaluation"
    source_poses = source_dir / "results" / "replicate_001"
    lddm_poses = evaluation_dir / "results" / "replicate_001"
    source_poses.mkdir(parents=True)
    lddm_poses.mkdir(parents=True)
    (source_dir / "input").mkdir()
    (source_dir / "input" / "compounds.tsv").write_text(
        "compound_id\tsmiles\nCMP1\tCCO\n"
    )

    source_molecule = _ethanol()
    source_molecule.SetProp("compound_id", "CMP1")
    writer = Chem.SDWriter(str(source_poses / "CMP1_out.sdf"))
    writer.write(source_molecule)
    writer.close()
    (source_poses / "CMP1_out.pdbqt").write_text(
        "MODEL 1\n"
        "HETATM    1  C1  UNL     1       0.000   0.000   0.000  0.00  0.00     0.000 C\n"
        "ENDMDL\n"
    )

    sampled_molecule = Chem.Mol(source_molecule)
    sampled_molecule.SetProp("compound_id", "CMP1")
    conformer = sampled_molecule.GetConformer()
    for atom_index in range(sampled_molecule.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        conformer.SetAtomPosition(
            atom_index, (point.x + 1.0, point.y, point.z)
        )
    sampled_molecule.SetProp(
        "sigma_x",
        ",".join(["0.25"] * sampled_molecule.GetNumAtoms()),
    )
    writer = Chem.SDWriter(str(lddm_poses / "CMP1_out.sdf"))
    writer.write(sampled_molecule)
    writer.close()

    source_job = JobRecord(
        run_id="source-run",
        task_group="docking",
        run_dir=source_dir,
        status="completed",
        workflow="docking_campaign",
        tool="vina",
        metadata={"engine": "vina", "tool": "vina"},
    )
    comparisons, summaries = _comparison_rows(evaluation_dir, source_job)

    assert len(comparisons) == 1
    assert comparisons[0]["closest_lddm_pose_rmsd_angstrom"] == 1.0
    assert comparisons[0]["closest_lddm_mean_uncertainty"] == 0.25
    assert summaries[0]["source_pose_count"] == 1
