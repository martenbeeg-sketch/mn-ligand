"""SMILES-first chemical and 3D qualification for generated compounds."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING, Any

import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    QED,
    rdMolDescriptors,
)


if TYPE_CHECKING:
    from posebusters import PoseBusters


QUALIFICATION_SCHEMA_VERSION = 2
REVIEWABLE_POSEBUSTERS_CHECKS = frozenset(
    {
        "non-aromatic_ring_non-flatness",
    }
)
ALLOWED_ATOMIC_NUMBERS = frozenset(
    {
        1,   # H
        5,   # B
        6,   # C
        7,   # N
        8,   # O
        9,   # F
        14,  # Si
        15,  # P
        16,  # S
        17,  # Cl
        34,  # Se
        35,  # Br
        53,  # I
    }
)
SEVERE_REACTIVE_ALERTS = {
    "peroxide": "[O;X2]-[O;X2]",
    "acyl_halide": "[C;X3](=O)[F,Cl,Br,I]",
    "acid_anhydride": "[C;X3](=O)O[C;X3](=O)",
    "isocyanate": "N=C=O",
    "isothiocyanate": "N=C=S",
    "diazonium": "[N+]#N",
}


def _safe_id(value: object, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (text or fallback)[:160]


def _sa_score(molecule: Chem.Mol) -> float:
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError:
        contribution = Path(RDConfig.RDContribDir) / "SA_Score"
        if str(contribution) not in sys.path:
            sys.path.append(str(contribution))
        import sascorer

    return float(sascorer.calculateScore(molecule))


def _chemical_assessment(
    smiles: str,
    *,
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    max_absolute_charge: int,
    max_sa_score: float,
) -> tuple[Chem.Mol | None, dict[str, Any], list[str]]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, {}, ["canonical SMILES could not be parsed"]
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        return None, {}, [f"RDKit sanitization failed: {exc}"]

    heavy_atoms = int(molecule.GetNumHeavyAtoms())
    formal_charge = int(Chem.GetFormalCharge(molecule))
    radicals = int(
        sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    )
    fragments = len(Chem.GetMolFrags(molecule))
    disallowed = sorted(
        {
            atom.GetSymbol()
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() not in ALLOWED_ATOMIC_NUMBERS
        }
    )
    sa_score = _sa_score(molecule)
    alerts = [
        name
        for name, smarts in SEVERE_REACTIVE_ALERTS.items()
        if molecule.HasSubstructMatch(Chem.MolFromSmarts(smarts))
    ]
    ring_info = molecule.GetRingInfo().AtomRings()
    descriptors = {
        "heavy_atom_count": heavy_atoms,
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "formal_charge": formal_charge,
        "fragment_count": fragments,
        "radical_electron_count": radicals,
        "qed": float(QED.qed(molecule)),
        "sa_score": sa_score,
        "logp": float(Crippen.MolLogP(molecule)),
        "hbond_donors": int(Lipinski.NumHDonors(molecule)),
        "hbond_acceptors": int(Lipinski.NumHAcceptors(molecule)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(molecule)),
        "largest_ring_size": max((len(ring) for ring in ring_info), default=0),
        "bridgehead_atom_count": int(
            rdMolDescriptors.CalcNumBridgeheadAtoms(molecule)
        ),
        "spiro_atom_count": int(rdMolDescriptors.CalcNumSpiroAtoms(molecule)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
        "reactive_alerts": "; ".join(alerts),
        "disallowed_elements": "; ".join(disallowed),
    }
    failures: list[str] = []
    if fragments != 1:
        failures.append(f"molecule has {fragments} disconnected fragments")
    if radicals:
        failures.append(f"molecule has {radicals} radical electrons")
    if disallowed:
        failures.append("disallowed elements: " + ", ".join(disallowed))
    if heavy_atoms < int(min_heavy_atoms):
        failures.append(
            f"heavy atom count {heavy_atoms} is below {int(min_heavy_atoms)}"
        )
    if heavy_atoms > int(max_heavy_atoms):
        failures.append(
            f"heavy atom count {heavy_atoms} exceeds {int(max_heavy_atoms)}"
        )
    if abs(formal_charge) > int(max_absolute_charge):
        failures.append(
            f"formal charge {formal_charge:+d} exceeds ±{int(max_absolute_charge)}"
        )
    if sa_score > float(max_sa_score):
        failures.append(
            f"synthetic accessibility {sa_score:.2f} exceeds {float(max_sa_score):.2f}"
        )
    if alerts:
        failures.append("severe reactive alerts: " + ", ".join(alerts))
    return molecule, descriptors, failures


def _standardized_conformer(
    molecule: Chem.Mol,
    *,
    seed: int,
    conformer_count: int,
) -> tuple[Chem.Mol | None, dict[str, Any], str]:
    embedded = Chem.AddHs(Chem.Mol(molecule))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed) & 0x7FFFFFFF
    parameters.pruneRmsThresh = 0.5
    parameters.useRandomCoords = False
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            embedded,
            numConfs=max(1, int(conformer_count)),
            params=parameters,
        )
    )
    if not conformer_ids:
        return None, {}, "ETKDGv3 could not generate a conformer"

    force_field = ""
    results: list[tuple[int, float]] = []
    if AllChem.MMFFHasAllMoleculeParams(embedded):
        force_field = "MMFF94s"
        results = [
            (int(status), float(energy))
            for status, energy in AllChem.MMFFOptimizeMoleculeConfs(
                embedded,
                numThreads=1,
                maxIters=1000,
                mmffVariant="MMFF94s",
            )
        ]
    elif AllChem.UFFHasAllMoleculeParams(embedded):
        force_field = "UFF"
        results = [
            (int(status), float(energy))
            for status, energy in AllChem.UFFOptimizeMoleculeConfs(
                embedded,
                numThreads=1,
                maxIters=1000,
            )
        ]
    else:
        return None, {}, "neither MMFF94s nor UFF supports this molecule"
    if not results:
        return None, {}, f"{force_field} returned no conformer energies"

    converged = [
        (index, energy)
        for index, (status, energy) in enumerate(results)
        if status == 0
    ]
    pool = converged or [
        (index, energy)
        for index, (_, energy) in enumerate(results)
    ]
    selected_index, selected_energy = min(pool, key=lambda item: item[1])
    selected_id = int(conformer_ids[selected_index])

    selected = Chem.Mol(embedded)
    conformer = Chem.Conformer(embedded.GetConformer(selected_id))
    selected.RemoveAllConformers()
    selected.AddConformer(conformer, assignId=True)
    selected = Chem.RemoveHs(selected)
    selected.GetConformer().Set3D(True)
    return (
        selected,
        {
            "geometry_source": "rdkit_etkdgv3",
            "embedding_seed": int(seed) & 0x7FFFFFFF,
            "conformers_attempted": max(1, int(conformer_count)),
            "conformers_embedded": len(conformer_ids),
            "force_field": force_field,
            "force_field_converged": bool(converged),
            "selected_conformer_index": selected_index,
            "selected_energy": selected_energy,
        },
        "",
    )


def _is_boolean_series(series: pd.Series) -> bool:
    values = set(series.dropna().tolist())
    return bool(values) and values.issubset({True, False})


def _applicable_binary_columns(validator: PoseBusters) -> list[str]:
    columns: list[str] = []
    for module in validator.config.get("modules", []):
        suffix = str(module.get("rename_suffix") or "")
        rename_outputs = module.get("rename_outputs") or {}
        for output in module.get("chosen_binary_test_output") or []:
            renamed = str(rename_outputs.get(output) or f"{output}{suffix}")
            columns.append(renamed.lower().replace(" ", "_"))
    return list(dict.fromkeys(columns))


def qualify_molecules(
    input_sdf: Path,
    input_table: Path,
    output_dir: Path,
    *,
    seed: int,
    conformer_count: int,
    max_workers: int,
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    max_absolute_charge: int,
    max_sa_score: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    table = (
        pd.read_csv(input_table).fillna("")
        if input_table.is_file()
        else pd.DataFrame()
    )
    molecules = list(
        Chem.SDMolSupplier(
            str(input_sdf),
            removeHs=False,
            sanitize=False,
            strictParsing=False,
        )
    )
    rows: list[dict[str, Any]] = []
    prepared: dict[str, tuple[Chem.Mol, Path, int]] = {}
    embedding_inputs: dict[int, tuple[Chem.Mol, int]] = {}
    for index, source_molecule in enumerate(molecules):
        source_row = table.iloc[index].to_dict() if index < len(table) else {}
        compound_id = _safe_id(
            source_row.get("compound_id")
            or (
                source_molecule.GetProp("compound_id")
                if source_molecule is not None
                and source_molecule.HasProp("compound_id")
                else ""
            ),
            f"compound_{index + 1:07d}",
        )
        smiles = str(
            source_row.get("canonical_isomeric_smiles")
            or (
                source_molecule.GetProp("canonical_isomeric_smiles")
                if source_molecule is not None
                and source_molecule.HasProp("canonical_isomeric_smiles")
                else ""
            )
        ).strip()
        base = {
            "compound_id": compound_id,
            "canonical_isomeric_smiles": smiles,
            "generation_engine": source_row.get("generation_engine", ""),
            "native_source": source_row.get("native_source", ""),
            "native_index": source_row.get("native_index", ""),
            "source_coordinate_dimension": source_row.get(
                "coordinate_dimension", ""
            ),
            "chemical_pass": False,
            "chemical_failures": "",
            "conformer_generation_pass": False,
            "conformer_generation_error": "",
            "posebusters_pass": False,
            "posebusters_failed_checks": "",
            "posebusters_hard_pass": False,
            "posebusters_hard_failures": "",
            "review_warnings": "",
            "qualification_status": "rejected",
            "qualified_for_docking": False,
        }
        molecule, descriptors, chemical_failures = _chemical_assessment(
            smiles,
            min_heavy_atoms=min_heavy_atoms,
            max_heavy_atoms=max_heavy_atoms,
            max_absolute_charge=max_absolute_charge,
            max_sa_score=max_sa_score,
        )
        base.update(descriptors)
        base["chemical_pass"] = not chemical_failures
        base["chemical_failures"] = "; ".join(chemical_failures)
        if molecule is None or chemical_failures:
            rows.append(base)
            continue
        conformer_seed = (int(seed) + index * 104729) & 0x7FFFFFFF
        base["embedding_seed"] = conformer_seed
        embedding_inputs[len(rows)] = (molecule, conformer_seed)
        rows.append(base)

    embedded_results: dict[
        int, tuple[Chem.Mol | None, dict[str, Any], str]
    ] = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(int(max_workers), len(embedding_inputs) or 1))
    ) as executor:
        futures = {
            executor.submit(
                _standardized_conformer,
                molecule,
                seed=conformer_seed,
                conformer_count=conformer_count,
            ): row_index
            for row_index, (molecule, conformer_seed) in embedding_inputs.items()
        }
        for future in as_completed(futures):
            row_index = futures[future]
            try:
                embedded_results[row_index] = future.result()
            except Exception as exc:
                embedded_results[row_index] = (
                    None,
                    {},
                    f"3D conformer generation failed: {exc}",
                )

    for row_index in sorted(embedding_inputs):
        conformer, geometry, geometry_error = embedded_results[row_index]
        base = rows[row_index]
        compound_id = str(base["compound_id"])
        smiles = str(base["canonical_isomeric_smiles"])
        base.update(geometry)
        base["conformer_generation_pass"] = conformer is not None
        base["conformer_generation_error"] = geometry_error
        if conformer is None:
            continue
        conformer.SetProp("_Name", compound_id)
        conformer.SetProp("compound_id", compound_id)
        conformer.SetProp("canonical_isomeric_smiles", smiles)
        conformer.SetProp("geometry_source", "rdkit_etkdgv3")
        conformer.SetIntProp(
            "embedding_seed",
            int(geometry.get("embedding_seed") or 0),
        )
        candidate_path = candidate_dir / f"{compound_id}.sdf"
        writer = Chem.SDWriter(str(candidate_path))
        writer.write(conformer)
        writer.close()
        prepared[str(candidate_path.resolve())] = (
            conformer,
            candidate_path,
            row_index,
        )

    full = pd.DataFrame()
    applicable_checks: list[str] = []
    if prepared:
        from posebusters import PoseBusters

        validator = PoseBusters(
            config="mol",
            max_workers=max(1, int(max_workers)),
            chunk_size=None,
        )
        full = validator.bust_table(
            pd.DataFrame({"mol_pred": list(prepared)}),
            full_report=True,
        ).reset_index()
        applicable_checks = [
            column
            for column in _applicable_binary_columns(validator)
            if column in full.columns and _is_boolean_series(full[column])
        ]
        for _, result in full.iterrows():
            prepared_item = prepared.get(str(Path(str(result["file"])).resolve()))
            if prepared_item is None:
                continue
            _, _, row_index = prepared_item
            failed = [
                column
                for column in applicable_checks
                if not bool(result.get(column))
            ]
            review_warnings = [
                column
                for column in failed
                if column in REVIEWABLE_POSEBUSTERS_CHECKS
            ]
            hard_failures = [
                column
                for column in failed
                if column not in REVIEWABLE_POSEBUSTERS_CHECKS
            ]
            rows[row_index]["posebusters_pass"] = not failed and bool(
                applicable_checks
            )
            rows[row_index]["posebusters_failed_checks"] = "; ".join(failed)
            rows[row_index]["posebusters_hard_pass"] = (
                not hard_failures and bool(applicable_checks)
            )
            rows[row_index]["posebusters_hard_failures"] = "; ".join(
                hard_failures
            )
            rows[row_index]["review_warnings"] = "; ".join(review_warnings)
            rows[row_index]["qualified_for_docking"] = bool(
                rows[row_index]["chemical_pass"]
                and rows[row_index]["conformer_generation_pass"]
                and rows[row_index]["posebusters_hard_pass"]
            )
            if rows[row_index]["qualified_for_docking"]:
                rows[row_index]["qualification_status"] = (
                    "qualified_with_warning"
                    if review_warnings
                    else "qualified"
                )

    inventory = pd.DataFrame(rows)
    inventory.to_csv(output_dir / "qualification.csv", index=False)
    if not full.empty:
        full.to_csv(output_dir / "posebusters_full.csv", index=False)
    else:
        (output_dir / "posebusters_full.csv").write_text("")

    writer = Chem.SDWriter(str(output_dir / "qualified_compounds.sdf"))
    qualified_count = 0
    for path_text, (molecule, _, row_index) in prepared.items():
        del path_text
        if not rows[row_index]["qualified_for_docking"]:
            continue
        for key, value in rows[row_index].items():
            if value not in ("", None):
                molecule.SetProp(str(key), str(value))
        writer.write(molecule)
        qualified_count += 1
    writer.close()
    report = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "success": True,
        "input_count": len(rows),
        "chemical_pass_count": sum(
            bool(row["chemical_pass"]) for row in rows
        ),
        "conformer_pass_count": sum(
            bool(row["conformer_generation_pass"]) for row in rows
        ),
        "posebusters_pass_count": sum(
            bool(row["posebusters_pass"]) for row in rows
        ),
        "posebusters_hard_pass_count": sum(
            bool(row["posebusters_hard_pass"]) for row in rows
        ),
        "qualified_with_warning_count": sum(
            row["qualification_status"] == "qualified_with_warning"
            for row in rows
        ),
        "qualified_compound_count": qualified_count,
        "policy": {
            "identity_source": "canonical stereochemistry-aware SMILES",
            "geometry_source": "deterministic RDKit ETKDGv3",
            "posebusters_config": "mol",
            "reviewable_posebusters_checks": sorted(
                REVIEWABLE_POSEBUSTERS_CHECKS
            ),
            "conformer_count": max(1, int(conformer_count)),
            "min_heavy_atoms": int(min_heavy_atoms),
            "max_heavy_atoms": int(max_heavy_atoms),
            "max_absolute_charge": int(max_absolute_charge),
            "max_sa_score": float(max_sa_score),
            "severe_reactive_alerts": list(SEVERE_REACTIVE_ALERTS),
        },
        "applicable_posebusters_checks": applicable_checks,
    }
    (output_dir / "qualification_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
