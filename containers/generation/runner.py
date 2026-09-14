from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from rdkit import Chem
from rdkit.Chem import rdDepictor


ENGINE_IMPORTS = {
    "omtra": "omtra",
    "pocketxmol": "models",
    "flowr_root": "flowr",
    "conditar": "models",
    "paopt": "scripts.paOPT.sample_with_opt",
    "drugrpg": "model",
    "pfm": "egnn",
    "pocketflow": "pocket_flow",
    "pgmg": "model.pgmg",
}


def healthcheck() -> int:
    import torch

    engine = os.environ.get("ENGINE", "").strip()
    module = ENGINE_IMPORTS.get(engine)
    if not engine or module is None:
        raise RuntimeError(f"Unknown generation engine: {engine!r}")
    importlib.import_module(module)
    payload = {
        "engine": engine,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "compiled_cuda_architectures": (
            torch.cuda.get_arch_list() if torch.cuda.is_available() else []
        ),
    }
    if torch.cuda.is_available():
        capability = tuple(int(value) for value in torch.cuda.get_device_capability())
        compiled_architectures = set(torch.cuda.get_arch_list())
        blackwell_compiled = bool(
            {"sm_120", "compute_120"} & compiled_architectures
        )
        payload.update(
            {
                "device": torch.cuda.get_device_name(),
                "capability": list(capability),
                "blackwell_compiled": blackwell_compiled,
            }
        )
        if not blackwell_compiled:
            raise RuntimeError(
                "The installed PyTorch stack does not include sm_120/compute_120"
            )
        if (
            os.environ.get("REQUIRE_BLACKWELL_DEVICE", "").strip() == "1"
            and capability < (12, 0)
        ):
            raise RuntimeError(
                f"Strict Blackwell validation requested; active device is {capability}"
            )
    print(json.dumps(payload, indent=2))
    return 0


def pocket_box(path: Path) -> tuple[tuple[float, float, float], float]:
    coordinates: list[tuple[float, float, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            coordinates.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
        except ValueError:
            continue
    if not coordinates:
        raise RuntimeError(f"No atom coordinates found in pocket structure: {path}")
    axes = tuple(zip(*coordinates))
    minima = tuple(min(axis) for axis in axes)
    maxima = tuple(max(axis) for axis in axes)
    center = tuple(
        (minimum + maximum) / 2.0
        for minimum, maximum in zip(minima, maxima, strict=True)
    )
    box_length = max(
        23.0,
        max(
            maximum - minimum
            for minimum, maximum in zip(minima, maxima, strict=True)
        )
        + 6.0,
    )
    return center, box_length


def selected_atom_fragment(
    source: Path,
    atom_indices: list[int],
    destination: Path,
) -> tuple[Path, int]:
    """Write a coordinate-preserving heavy-atom fragment from an input SDF."""
    supplier = Chem.SDMolSupplier(str(source), removeHs=False, sanitize=True)
    molecule = next((value for value in supplier if value is not None), None)
    if molecule is None:
        raise ValueError(f"Could not read reference ligand: {source}")
    molecule = Chem.RemoveHs(molecule)
    selected = sorted(set(int(value) for value in atom_indices))
    if not selected:
        raise ValueError("Fragment growing requires at least one retained atom")
    if selected[-1] >= molecule.GetNumAtoms():
        raise ValueError(
            "Fragment atom index exceeds the reference ligand atom count"
        )
    editable = Chem.RWMol(molecule)
    selected_set = set(selected)
    for atom_index in reversed(range(molecule.GetNumAtoms())):
        if atom_index not in selected_set:
            editable.RemoveAtom(atom_index)
    fragment = editable.GetMol()
    Chem.SanitizeMol(fragment)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(destination))
    writer.write(fragment)
    writer.close()
    return destination, fragment.GetNumAtoms()


def native_command(args: argparse.Namespace) -> list[str]:
    engine = os.environ.get("ENGINE", "").strip()
    common_output = str(args.output.resolve())
    if engine == "omtra":
        command = [
            "omtra",
            "--task",
            args.native_mode,
            "--n_samples",
            str(args.count),
            "--seed",
            str(args.seed),
            "--output_dir",
            common_output,
            "--checkpoint",
            str(args.checkpoint),
            "--n_timesteps",
            str(args.integration_steps),
        ]
        if args.stochastic_sampling:
            command += [
                "--stochastic_sampling",
                "--noise_scaler",
                str(args.noise_scale),
                "--eps",
                str(args.epsilon),
            ]
        if args.ligand_atoms_mean > 0:
            command += [
                "--n_lig_atoms_mean",
                str(args.ligand_atoms_mean),
                "--n_lig_atoms_std",
                str(args.ligand_atoms_std),
            ]
        if args.target:
            command += ["--protein_file", str(args.target)]
        if args.reference_ligand:
            if args.native_mode.startswith("fixed_protein"):
                command += ["--pocket_ligand", str(args.reference_ligand)]
            else:
                command += ["--ligand_file", str(args.reference_ligand)]
        elif args.pocket_structure:
            pocket_center, box_length = pocket_box(args.pocket_structure)
            command += [
                "--pocket_center",
                *(f"{coordinate:.6f}" for coordinate in pocket_center),
                "--bbox_length",
                f"{box_length:.6f}",
            ]
        if args.pharmacophore:
            command += ["--pharmacophore_file", str(args.pharmacophore)]
        return command
    if engine == "pgmg":
        return [
            sys.executable,
            "generate.py",
            str(args.pharmacophore),
            common_output,
            str(args.checkpoint),
            str(args.secondary_reference),
            "--n_mol",
            str(args.count),
            "--batch_size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--device",
            "cuda",
            "--filter",
        ]
    if engine == "pocketflow":
        pocket_structure = prepare_pocketflow_pocket(
            args.pocket_structure, args.output
        )
        command = [
            sys.executable,
            "main_generate.py",
            "--pocket",
            str(pocket_structure),
            "--ckpt",
            str(args.checkpoint),
            "--num_gen",
            str(args.count),
            "--device",
            "cuda:0",
            "--root_path",
            common_output,
            "--name",
            args.run_name,
            "--seed",
            str(args.seed),
            "--atom_temperature",
            str(args.atom_temperature),
            "--bond_temperature",
            str(args.bond_temperature),
            "--max_atom_num",
            str(args.max_atoms),
            "--focus_threshold",
            str(args.focus_threshold),
            "--choose_max",
            "1" if args.focus_strategy == "maximum" else "0",
            "--min_dist_inter_mol",
            str(args.min_protein_distance),
        ]
        return command
    if engine == "pocketxmol":
        task_config, model_config = prepare_pocketxmol_configs(args)
        return [
            sys.executable,
            "scripts/sample_use.py",
            "--config_task",
            str(task_config),
            "--config_model",
            str(model_config),
            "--outdir",
            str(args.output.resolve() / "pocketxmol_outputs"),
            "--device",
            "cuda:0",
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            "0",
        ]
    if engine == "flowr_root":
        if args.pocket_structure is None:
            raise ValueError("FLOWR.root requires a pocket PDB")
        if args.reference_ligand is None:
            raise ValueError(
                "FLOWR.root requires a reference ligand to establish its "
                "pocket/conditioning frame"
            )
        ligand_file = args.reference_ligand
        if args.redesign_mode == "fragment_growing":
            ligand_file, _ = selected_atom_fragment(
                args.reference_ligand,
                args.preserve_atom_index,
                args.output.resolve() / "selected_fragment.sdf",
            )
        command = [
            sys.executable,
            "flowr/gen/generate_from_pdb.py",
            "--pdb_file",
            str(args.pocket_structure),
            "--ligand_file",
            str(ligand_file),
            "--arch",
            "pocket",
            "--pocket_type",
            "holo",
            "--ckpt_path",
            str(args.checkpoint),
            "--save_dir",
            common_output,
            "--sample_n_molecules_per_target",
            str(args.count),
            "--batch_cost",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--gpus",
            "1",
            "--num_workers",
            "0",
            "--filter_valid_unique",
            "--integration_steps",
            str(args.integration_steps),
            "--corrector_iters",
            str(args.corrector_steps),
            "--solver",
            str(args.solver),
        ]
        if args.redesign_mode == "replace_selected_atoms":
            if not args.redesign_atom_index:
                raise ValueError(
                    "FLOWR.root local redesign requires at least one atom index "
                    "to replace"
                )
            command.extend(
                [
                    "--substructure_inpainting",
                    "--substructure",
                    *[str(value) for value in args.redesign_atom_index],
                ]
            )
            if args.filter_conditioned_substructure:
                command.append("--filter_cond_substructure")
        elif args.redesign_mode == "fragment_growing":
            command.extend(
                ["--fragment_growing", "--grow_size", str(args.grow_size)]
            )
            if args.filter_conditioned_substructure:
                command.append("--filter_cond_substructure")
        elif args.redesign_mode == "scaffold_hopping":
            command.append("--scaffold_hopping")
            if args.filter_conditioned_substructure:
                command.append("--filter_cond_substructure")
        if args.use_sde_simulation:
            command.append("--use_sde_simulation")
        if args.sample_molecule_sizes:
            command.append("--sample_mol_sizes")
        if args.filter_diversity:
            command += [
                "--filter_diversity",
                "--diversity_threshold",
                str(args.diversity_threshold),
            ]
        mode = str(args.native_mode or "").lower()
        if "detect interactions" in mode:
            command += ["--interaction_conditional", "--compute_interactions"]
        elif "preserve as scaffold" in mode:
            command += ["--scaffold_elaboration", "--filter_cond_substructure"]
        elif "grow or link fragments" in mode:
            command += ["--fragment_growing", "--grow_size", "10"]
        elif "retaining similarity" in mode:
            command += ["--scaffold_hopping", "--filter_cond_substructure"]
        elif "spatial reference" in mode:
            command += ["--ref_ligand_com_prior"]
        return command
    if engine == "pfm":
        if args.pocket_structure is None:
            raise ValueError("PFM requires a pocket PDB")
        if args.checkpoint is None:
            raise ValueError("PFM requires its official checkpoint")
        model_dir = args.checkpoint.parent
        return [
            sys.executable,
            "scripts/sample_custom_pocket.py",
            "--pocket",
            str(args.pocket_structure),
            "--checkpoint",
            str(args.checkpoint),
            "--x-predictor",
            str(model_dir / "x_predictor.pth"),
            "--h-predictor",
            str(model_dir / "h_predictor.pth"),
            "--training-config",
            str(model_dir / "training.yml"),
            "--output",
            common_output,
            "--count",
            str(args.count),
            "--seed",
            str(args.seed),
            "--device",
            "cuda:0",
        ]
    if engine == "drugrpg":
        return [
            sys.executable,
            "sample_for_pocket.py",
            "--pdb_path",
            str(args.pocket_structure),
            "--num_atom",
            str(args.max_atoms),
            "--num_samples",
            str(args.count),
            "--batch_size",
            str(args.batch_size),
            "--ckpt",
            str(args.checkpoint),
            "--seed",
            str(args.seed),
            "--save_dir",
            str(args.output.resolve() / "sample_batch"),
        ]
    if engine in {"conditar", "paopt"}:
        if args.pocket_structure is None:
            raise ValueError("conDitar requires a pocket PDB")
        if args.checkpoint is None or args.secondary_reference is None:
            raise ValueError(
                "conDitar requires Diff.pt and PocketAE.pt reference files"
            )
        import yaml

        config = yaml.safe_load(
            Path("/opt/engine/configs/sample_container.yml").read_text()
        )
        config["model"]["checkpoint"] = str(args.checkpoint)
        config["model"]["checkpoint_pocket"] = str(args.secondary_reference)
        config["sample"]["seed"] = int(args.seed)
        config["sample"]["num_steps"] = int(args.diffusion_steps)
        config_path = args.output.resolve() / "conditar_sample.yml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        pocket_structure = prepare_conditar_pocket(
            args.pocket_structure, args.output
        )
        if engine == "paopt":
            if args.reference_ligand is None:
                raise ValueError("paOPT requires a reference ligand")
            reference_ligand = args.output.resolve() / "paopt_reference.sdf"
            shutil.copy2(args.reference_ligand, reference_ligand)
            command = [
                sys.executable,
                "-m",
                "scripts.paOPT.sample_with_opt",
                str(config_path),
                "--device",
                "cuda:0",
                "--protein_root",
                str(pocket_structure.parent),
                "--pdb_filename",
                pocket_structure.name,
                "--sdf_filename",
                reference_ligand.name,
                "--result_path",
                common_output,
                "--num_samples",
                str(args.count),
                "--batch_size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                "--opt_steps",
                str(args.optimization_steps),
                "--num_estimates",
                str(args.gradient_estimate_pairs),
                "--pocket_radius",
                str(int(round(args.pocket_radius))),
                "--per_size",
                str(args.perturbation_size),
                "--opt_keys",
                *args.optimize_properties,
            ]
            if args.minimize_properties:
                command.extend(
                    ["--opt_keys_min", *args.minimize_properties]
                )
            else:
                command.extend(["--opt_keys_min", "__none__"])
            return command
        command = [
            sys.executable,
            "-m",
            "scripts.conDitar.sample",
            str(config_path),
            "--device",
            "cuda:0",
            "--protein_root",
            str(pocket_structure.parent),
            "--pdb_filename",
            pocket_structure.name,
            "--result_path",
            common_output,
            "--num_samples",
            str(args.count),
            "--batch_size",
            str(args.batch_size),
            "--pocket_radius",
            str(int(round(args.pocket_radius))),
        ]
        return command
    raise RuntimeError(
        f"{engine} requires a reviewed engine-specific config adapter before native launch"
    )


def prepare_pocketxmol_configs(
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    """Translate an mn-ligand pocket campaign into PocketXMol native YAML.

    PocketXMol's public user entrypoint consumes a task YAML plus a very small
    model YAML.  The checkpoint's matching training configuration remains next
    to the checkpoint in the read-only official reference bundle and is found
    by PocketXMol itself.
    """
    import yaml

    if args.pocket_structure is None or not args.pocket_structure.is_file():
        raise ValueError("PocketXMol requires an existing pocket PDB")
    protein = args.target or args.pocket_structure
    if not protein.is_file():
        raise ValueError("PocketXMol requires an existing target or pocket PDB")
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise ValueError("PocketXMol requires its official checkpoint")

    center, box_length = pocket_box(args.pocket_structure)
    mode = str(args.native_mode or "").lower()
    local_redesign = args.redesign_mode == "replace_selected_atoms"
    fragment_growing = args.redesign_mode == "fragment_growing"
    partial_optimization = args.redesign_mode == "partial_optimization"
    template_name = (
        "opt_partial.yml"
        if local_redesign
        else (
            "growing_fixed_frag.yml"
            if fragment_growing
            else (
                "opt_mol.yml"
                if (
                    partial_optimization
                    or (
                        "optimize" in mode
                        and args.reference_ligand is not None
                    )
                )
                else "sbdd.yml"
            )
        )
    )
    template_path = (
        Path("/opt/engine/configs/sample/examples") / template_name
    )
    task = yaml.safe_load(template_path.read_text())
    task["sample"].update(
        {
            "seed": int(args.seed),
            "batch_size": max(1, int(args.batch_size)),
            "num_mols": int(args.count),
            "save_traj_prob": 0.0,
        }
    )
    task["noise"]["num_steps"] = int(args.diffusion_steps)
    task["data"]["protein_path"] = str(protein.resolve())
    task["data"]["is_pep"] = False
    task["data"]["pocmol_args"] = {
        "data_id": str(args.run_name),
        "pdbid": str(args.run_name),
    }
    task["data"]["pocket_args"] = {
        "pocket_coord": [float(value) for value in center],
        "radius": max(10.0, float(box_length) / 2.0),
        "criterion": "min",
    }
    task.setdefault("transforms", {}).setdefault(
        "featurizer_pocket", {}
    )["center"] = [float(value) for value in center]
    if args.ligand_atoms_mean > 0:
        size_distribution = (
            task.setdefault("transforms", {})
            .setdefault("variable_mol_size", {})
            .setdefault("num_atoms_distri", {})
        )
        size_distribution.setdefault("mean", {})["bias"] = float(
            args.ligand_atoms_mean
        )
        size_distribution.setdefault("std", {})["bias"] = float(
            args.ligand_atoms_std
        )
    if template_name in {
        "opt_mol.yml",
        "opt_partial.yml",
        "growing_fixed_frag.yml",
    }:
        if (
            args.reference_ligand is None
            or not args.reference_ligand.is_file()
        ):
            raise ValueError(
                "PocketXMol optimization requires a reference ligand"
            )
        input_ligand = args.reference_ligand.resolve()
        fragment_atom_count = 0
        if template_name == "growing_fixed_frag.yml":
            input_ligand, fragment_atom_count = selected_atom_fragment(
                args.reference_ligand,
                args.preserve_atom_index,
                args.output.resolve() / "selected_fragment.sdf",
            )
        task["data"]["input_ligand"] = str(input_ligand)
        task["data"]["pocket_args"] = {
            "ref_ligand_path": str(input_ligand),
            "radius": max(10.0, float(box_length) / 2.0),
            "criterion": "min",
        }
        if template_name != "growing_fixed_frag.yml":
            task["noise"]["init_step"] = float(args.optimization_strength)
        else:
            fragment_indices = list(range(fragment_atom_count))
            transforms = task.setdefault("transforms", {})
            transforms.setdefault("variable_mol_size", {})[
                "not_remove"
            ] = fragment_indices
            transform = task.setdefault("task", {}).setdefault(
                "transform", {}
            )
            partition = transform.setdefault("preset_partition", {})
            partition["grouped_node_p1"] = [fragment_indices]
            transform.setdefault("settings", {})["part1_pert"] = {"fixed": 1}
            transforms.setdefault("variable_mol_size", {}).setdefault(
                "num_atoms_distri", {}
            ).setdefault("mean", {})["bias"] = float(
                fragment_atom_count + args.grow_size
            )
    if template_name == "opt_partial.yml":
        preserve = sorted(set(int(value) for value in args.preserve_atom_index))
        redesign = sorted(set(int(value) for value in args.redesign_atom_index))
        anchors = sorted(set(int(value) for value in args.anchor_atom_index))
        if not preserve or not redesign:
            raise ValueError(
                "PocketXMol partial redesign requires both preserved and "
                "redesigned atom indices"
            )
        transforms = task.setdefault("transforms", {})
        transforms.setdefault("variable_mol_size", {})["not_remove"] = preserve
        transform = task.setdefault("task", {}).setdefault(
            "transform", {}
        )
        partition = transform.setdefault("preset_partition", {})
        partition["grouped_node_p1"] = [preserve]
        partition["node_p2"] = redesign
        if anchors:
            partition["grouped_anchor_p1"] = [anchors]
            transform.setdefault("settings", {})["known_anchor"] = {"all": 1}
        transform.setdefault("settings", {})["part1_pert"] = {"fixed": 1}

    task_path = args.output.resolve() / "pocketxmol_task.yml"
    model_path = args.output.resolve() / "pocketxmol_model.yml"
    task_path.write_text(yaml.safe_dump(task, sort_keys=False))
    model_path.write_text(
        yaml.safe_dump(
            {"model": {"checkpoint": str(args.checkpoint.resolve())}},
            sort_keys=False,
        )
    )
    return task_path, model_path


def prepare_pocketflow_pocket(source: Path | None, output_dir: Path) -> Path:
    """Annotate an already extracted pocket as PocketFlow surface input.

    PocketFlow's parser recognizes surface atoms through a trailing ``surf``
    token. A pocket artifact already contains only pocket-facing protein atoms,
    so annotating those records avoids a hidden PyMOL re-detection step and
    preserves the exact upstream pocket membership.
    """
    if source is None or not source.is_file():
        raise ValueError("PocketFlow requires an existing pocket PDB")
    prepared = output_dir / "pocketflow_input.pdb"
    rows: list[str] = []
    atom_count = 0
    for raw in source.read_text(errors="replace").splitlines():
        if raw.startswith(("ATOM  ", "HETATM")):
            rows.append(raw.rstrip() + " surf")
            atom_count += 1
        else:
            rows.append(raw)
    if atom_count == 0:
        raise ValueError(f"PocketFlow pocket contains no atom records: {source}")
    prepared.write_text("\n".join(rows) + "\n")
    return prepared


def prepare_conditar_pocket(source: Path, output_dir: Path) -> Path:
    """Remove explicit hydrogens before conDitar residue-boundary parsing.

    conDitar identifies residues from consecutive N/CA/C/O records before its
    own hydrogen filter runs. Prepared mn-ligand pockets contain explicit
    hydrogens between those records, so retain the exact heavy-atom pocket
    membership while removing only hydrogen ATOM/HETATM rows.
    """
    prepared = output_dir / "conditar_pocket_no_h.pdb"
    rows: list[str] = []
    atom_count = 0
    for raw in source.read_text(errors="replace").splitlines():
        if raw.startswith(("ATOM  ", "HETATM")):
            element = raw[76:78].strip().upper()
            atom_name = raw[12:16].strip().upper()
            if element == "H" or (
                not element and atom_name.lstrip("0123456789").startswith("H")
            ):
                continue
            atom_count += 1
        rows.append(raw)
    if atom_count == 0:
        raise ValueError(f"conDitar pocket contains no heavy atoms: {source}")
    prepared.write_text("\n".join(rows) + "\n")
    return prepared


def _candidate_molecules(
    native_dir: Path,
    *,
    engine: str,
) -> list[tuple[Chem.Mol, Path, int]]:
    molecules: list[tuple[Chem.Mol, Path, int]] = []
    sdf_paths = sorted(native_dir.rglob("*.sdf"))
    if engine == "paopt":
        step_paths: list[tuple[int, Path]] = []
        for path in sdf_paths:
            match = re.search(r"_generated_(\d+)_\d+\.sdf$", path.name)
            if match:
                step_paths.append((int(match.group(1)), path))
        final_step = max((step for step, _ in step_paths), default=-1)
        sdf_paths = [
            path for step, path in step_paths if step == final_step
        ]
    for path in sdf_paths:
        if path.name == "generated_compounds.sdf":
            continue
        if engine == "omtra" and not (
            path.name == "gen_ligands.sdf"
            or (
                path.name.endswith("_lig.sdf")
                and not path.name.endswith(("_lig_xt.sdf", "_lig_xhat.sdf"))
            )
        ):
            continue
        if engine == "pocketflow" and path.name != "generated.sdf":
            continue
        if engine == "pocketxmol":
            if not path.parent.name.endswith("_SDF"):
                continue
            if path.parent.name == "0_inputs" or any(
                value in path.stem for value in ("-bad", "-incomp")
            ):
                continue
        if engine == "flowr_root" and (
            "_hs_" in path.stem
            or path.parent.name in {"ref_pdbs", "gen_complexes_protonated"}
        ):
            continue
        supplier = Chem.SDMolSupplier(
            str(path), removeHs=False, sanitize=True, strictParsing=False
        )
        for index, molecule in enumerate(supplier):
            if molecule is not None:
                molecules.append((molecule, path, index))
    # PGMG is the implemented SMILES-native engine. PocketFlow also writes a
    # convenience .smi beside its authoritative 3D SDF; reading both would
    # double-count every native molecule and discard the 3D preference.
    smiles_files: list[Path] = []
    if engine == "pgmg":
        smiles_files.extend(native_dir.rglob("*.smi"))
        smiles_files.extend(native_dir.rglob("*.smiles"))
        smiles_files.extend(native_dir.rglob("*_result.txt"))
    for path in sorted(set(smiles_files)):
        for index, line in enumerate(path.read_text(errors="replace").splitlines()):
            smiles = line.strip().split()[0] if line.strip() else ""
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None:
                continue
            rdDepictor.Compute2DCoords(molecule)
            molecules.append((molecule, path, index))
    return molecules


def normalize_outputs(
    native_dir: Path,
    normalized_dir: Path,
    *,
    engine: str,
    requested_count: int,
) -> dict[str, object]:
    normalized_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    unique: dict[str, tuple[Chem.Mol, Path, int]] = {}
    candidates = _candidate_molecules(
        native_dir, engine=engine
    )
    for molecule, source, native_index in candidates:
        try:
            smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
        except Exception:
            continue
        if not smiles:
            continue
        unique.setdefault(smiles, (molecule, source, native_index))
    sdf_path = normalized_dir / "generated_compounds.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    for index, (smiles, item) in enumerate(unique.items(), start=1):
        molecule, source, native_index = item
        compound_id = f"{engine}-generated-{index:07d}"
        molecule.SetProp("_Name", compound_id)
        molecule.SetProp("compound_id", compound_id)
        molecule.SetProp("canonical_isomeric_smiles", smiles)
        molecule.SetProp("generation_engine", engine)
        molecule.SetProp(
            "native_source", source.relative_to(native_dir).as_posix()
        )
        molecule.SetIntProp("native_index", int(native_index))
        writer.write(molecule)
        records.append(
            {
                "compound_id": compound_id,
                "canonical_isomeric_smiles": smiles,
                "generation_engine": engine,
                "native_source": source.relative_to(native_dir).as_posix(),
                "native_index": native_index,
                "coordinate_dimension": (
                    3
                    if molecule.GetNumConformers()
                    and molecule.GetConformer().Is3D()
                    else 2
                ),
            }
        )
    writer.close()
    table_path = normalized_dir / "generated_compounds.csv"
    columns = [
        "compound_id",
        "canonical_isomeric_smiles",
        "generation_engine",
        "native_source",
        "native_index",
        "coordinate_dimension",
    ]
    with table_path.open("w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=columns)
        writer_csv.writeheader()
        writer_csv.writerows(records)
    report = {
        "engine": engine,
        "requested_count": int(requested_count),
        "valid_output_count": len(candidates),
        "unique_valid_compound_count": len(records),
        "normalization_policy": (
            "RDKit-valid molecules deduplicated by canonical isomeric SMILES"
        ),
        "success": bool(records),
    }
    (normalized_dir / "generation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="mn-ligand standardized molecular-generation image entrypoint"
    )
    value.add_argument("--healthcheck", action="store_true")
    value.add_argument("--target", type=Path)
    value.add_argument("--pocket-structure", type=Path)
    value.add_argument("--reference-ligand", type=Path)
    value.add_argument("--pharmacophore", type=Path)
    value.add_argument("--checkpoint", type=Path)
    value.add_argument("--secondary-reference", type=Path)
    value.add_argument("--output", type=Path, default=Path("/work/output"))
    value.add_argument(
        "--normalized-output", type=Path, default=Path("/work/normalized")
    )
    value.add_argument("--count", type=int, default=100)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--seed", type=int, default=2026)
    value.add_argument(
        "--optimize-properties",
        action="append",
        default=[],
    )
    value.add_argument(
        "--minimize-properties",
        action="append",
        default=[],
    )
    value.add_argument("--optimization-steps", type=int, default=1)
    value.add_argument("--gradient-estimate-pairs", type=int, default=4)
    value.add_argument("--integration-steps", type=int, default=250)
    value.add_argument("--diffusion-steps", type=int, default=1000)
    value.add_argument("--corrector-steps", type=int, default=0)
    value.add_argument(
        "--solver", choices=("euler", "midpoint"), default="euler"
    )
    value.add_argument("--use-sde-simulation", action="store_true")
    value.add_argument("--sample-molecule-sizes", action="store_true")
    value.add_argument("--filter-diversity", action="store_true")
    value.add_argument("--diversity-threshold", type=float, default=0.9)
    value.add_argument("--stochastic-sampling", action="store_true")
    value.add_argument("--noise-scale", type=float, default=1.0)
    value.add_argument("--epsilon", type=float, default=0.01)
    value.add_argument("--ligand-atoms-mean", type=float, default=0.0)
    value.add_argument("--ligand-atoms-std", type=float, default=2.0)
    value.add_argument("--atom-temperature", type=float, default=1.0)
    value.add_argument("--bond-temperature", type=float, default=1.0)
    value.add_argument("--max-atoms", type=int, default=40)
    value.add_argument(
        "--focus-strategy", choices=("maximum", "sample"), default="maximum"
    )
    value.add_argument("--focus-threshold", type=float, default=0.5)
    value.add_argument("--min-protein-distance", type=float, default=3.0)
    value.add_argument("--pocket-radius", type=float, default=10.0)
    value.add_argument("--perturbation-size", type=float, default=0.03)
    value.add_argument("--optimization-strength", type=float, default=0.5)
    value.add_argument("--grow-size", type=int, default=10)
    value.add_argument(
        "--redesign-mode",
        choices=(
            "",
            "replace_selected_atoms",
            "fragment_growing",
            "scaffold_hopping",
            "partial_optimization",
        ),
        default="",
    )
    value.add_argument(
        "--preserve-atom-index",
        action="append",
        type=int,
        default=[],
    )
    value.add_argument(
        "--redesign-atom-index",
        action="append",
        type=int,
        default=[],
    )
    value.add_argument(
        "--anchor-atom-index",
        action="append",
        type=int,
        default=[],
    )
    value.add_argument(
        "--filter-conditioned-substructure",
        action="store_true",
    )
    value.add_argument("--run-name", default="mn_ligand_generation")
    value.add_argument(
        "--native-mode", default="fixed_protein_ligand_denovo_condensed"
    )
    value.add_argument("--dry-run", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.healthcheck:
        return healthcheck()
    args.output.mkdir(parents=True, exist_ok=True)
    command = native_command(args)
    (args.output / "native_command.json").write_text(
        json.dumps({"argv": command}, indent=2) + "\n"
    )
    if args.dry_run:
        print(json.dumps({"argv": command}, indent=2))
        return 0
    runtime_env = os.environ.copy()
    runtime_env.setdefault("DGLBACKEND", "pytorch")
    runtime_env.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    completed = subprocess.run(command, check=False, env=runtime_env)
    if completed.returncode:
        return int(completed.returncode)
    report = normalize_outputs(
        args.output,
        args.normalized_output,
        engine=os.environ.get("ENGINE", "").strip(),
        requested_count=args.count,
    )
    return 0 if report["success"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
