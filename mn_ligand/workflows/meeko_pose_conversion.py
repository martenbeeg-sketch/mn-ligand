from __future__ import annotations

import argparse
from pathlib import Path


def _canonical_heavy_smiles(molecule) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(
        Chem.RemoveHs(molecule),
        isomericSmiles=True,
    )


def convert_pdbqt_to_sdf(
    pdbqt_path: Path,
    template_path: Path,
    output_path: Path,
) -> int:
    from meeko import PDBQTMolecule, RDKitMolCreate
    from rdkit import Chem

    template = next(
        (
            molecule
            for molecule in Chem.SDMolSupplier(
                str(template_path),
                removeHs=False,
            )
            if molecule is not None
        ),
        None,
    )
    if template is None:
        raise ValueError(f"Could not read ligand template: {template_path}")
    Chem.SanitizeMol(template)
    expected_smiles = _canonical_heavy_smiles(template)
    expected_heavy_atoms = template.GetNumHeavyAtoms()

    pdbqt = PDBQTMolecule.from_file(
        str(pdbqt_path),
        is_dlg=False,
        skip_typing=True,
    )
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.sdf")
    writer = Chem.SDWriter(str(temporary_path))
    pose_count = 0
    try:
        converted = RDKitMolCreate.from_pdbqt_mol(pdbqt)
        molecules = converted if isinstance(converted, list) else [converted]
        for molecule in molecules:
            if molecule is None:
                continue
            Chem.SanitizeMol(molecule)
            observed_smiles = _canonical_heavy_smiles(molecule)
            if molecule.GetNumHeavyAtoms() != expected_heavy_atoms:
                raise ValueError(
                    f"Converted molecule has {molecule.GetNumHeavyAtoms()} "
                    f"heavy atoms; expected {expected_heavy_atoms}"
                )
            if observed_smiles != expected_smiles:
                raise ValueError(
                    "Converted pose chemistry differs from its prepared "
                    f"ligand template: {observed_smiles} != {expected_smiles}"
                )
            for conformer in molecule.GetConformers():
                pose_count += 1
                molecule.SetProp(
                    "_Name",
                    f"{pdbqt_path.stem}_pose_{pose_count}",
                )
                molecule.SetIntProp("POSE_RANK", pose_count)
                molecule.SetProp(
                    "PREPARATION_METHOD",
                    "Meeko PDBQT reconstruction",
                )
                writer.write(molecule, confId=conformer.GetId())
    except Exception:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    writer.close()
    if pose_count < 1:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"No docking poses found in {pdbqt_path}")
    temporary_path.replace(output_path)
    return pose_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdbqt", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    count = convert_pdbqt_to_sdf(
        arguments.pdbqt,
        arguments.template,
        arguments.output,
    )
    print(f"Converted {count} pose(s) with Meeko: {arguments.output}")


if __name__ == "__main__":
    main()
