from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, "/opt/boltzina")

from boltzina_main import Boltzina


Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


def _prepare_poses(
    path: Path, output_dir: Path
) -> tuple[list[Path], Path, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[Path] = []
    molecules: dict[str, Chem.Mol] = {}
    compound_ids: list[str] = []
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    for index, source in enumerate(supplier, start=1):
        if source is None:
            continue
        molecule = Chem.RemoveHs(source, sanitize=False)
        name = (
            molecule.GetProp("_Name").strip()
            if molecule.HasProp("_Name")
            else f"pose_{index:07d}"
        )
        canonical_order = AllChem.CanonicalRankAtoms(molecule)
        for atom, canonical_index in zip(molecule.GetAtoms(), canonical_order):
            atom_name = f"{atom.GetSymbol().upper()}{canonical_index + 1}"
            atom.SetProp("name", atom_name)
            info = Chem.AtomPDBResidueInfo()
            info.SetName(atom_name.rjust(4))
            info.SetResidueName("UNL")
            info.SetResidueNumber(1)
            info.SetChainId("A")
            info.SetIsHeteroAtom(True)
            atom.SetMonomerInfo(info)
        pdb_path = output_dir / f"{name}.pdb"
        Chem.MolToPDBFile(molecule, str(pdb_path))
        prepared.append(pdb_path)
        molecules[pdb_path.stem] = molecule
        if molecule.HasProp("source_compound_id"):
            compound_ids.append(molecule.GetProp("source_compound_id").strip())
    if not prepared:
        raise ValueError(f"No readable 3D poses in {path}")
    pickle_path = output_dir / "prepared_mols.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(molecules, handle)
    return prepared, pickle_path, compound_ids


def _prepare_work_dir(
    context_dir: Path, work_dir: Path, compound_ids: list[str]
) -> str:
    source_processed = context_dir / "processed"
    manifest = json.loads((source_processed / "manifest.json").read_text())
    records = list(manifest.get("records") or [])
    if not records:
        raise ValueError("The Boltz-2 context manifest contains no records")
    matching = next(
        (
            record
            for compound_id in compound_ids
            for record in records
            if str(record.get("id") or "") == compound_id
        ),
        records[0],
    )
    work_processed = work_dir / "processed"
    work_processed.mkdir(parents=True, exist_ok=True)
    for child in source_processed.iterdir():
        if child.name == "manifest.json":
            continue
        target = work_processed / child.name
        if not target.exists():
            os.symlink(child, target, target_is_directory=child.is_dir())
    manifest["records"] = [matching]
    (work_processed / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return str(matching["id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--receptor", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--affinity-mw-correction", action="store_true")
    args = parser.parse_args()

    pdb_paths, prepared_mols, compound_ids = _prepare_poses(
        args.poses, args.output_dir / "ligands_prepared"
    )
    fname = _prepare_work_dir(args.context_dir, args.work_dir, compound_ids)
    vina_config = args.output_dir / "unused_vina_config.txt"
    vina_config.parent.mkdir(parents=True, exist_ok=True)
    vina_config.write_text(
        "center_x = 0\ncenter_y = 0\ncenter_z = 0\n"
        "size_x = 20\nsize_y = 20\nsize_z = 20\n"
    )
    scorer = Boltzina(
        receptor_pdb=str(args.receptor),
        output_dir=str(args.output_dir),
        config=str(vina_config),
        work_dir=str(args.work_dir),
        fname=fname,
        seed=args.seed,
        batch_size=max(1, args.batch_size),
        scoring_only=True,
        input_ligand_name="UNL",
        base_ligand_name="MOL",
        use_kernels=True,
        skip_run_structure=True,
        run_trunk_and_structure=True,
        clean_intermediate_files=True,
        prepared_mols_file=str(prepared_mols),
        predict_affinity_args=(
            {"affinity_mw_correction": True}
            if args.affinity_mw_correction
            else None
        ),
    )
    scorer.run([str(path) for path in pdb_paths])
    scorer.save_results_csv()


if __name__ == "__main__":
    main()
