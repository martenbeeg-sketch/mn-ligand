"""Repair user-selected noncanonical protein residues with native MODELLER.

This runner intentionally depends only on MODELLER and the Python standard
library.  It is executed with the isolated ``mn-ligand-modeller`` interpreter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modeller import Alignment, Environ, Model, Selection, log
from modeller.optimizers import ConjugateGradients, MolecularDynamics


def _residue_id(site: dict[str, object]) -> str:
    insertion_code = str(site.get("icode") or "").strip()
    return f"{site['resseq']}{insertion_code}:{site['chain']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--replacements", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    log.none()
    replacements = json.loads(Path(args.replacements).read_text())
    env = Environ()
    env.io.hetatm = True
    env.libs.topology.read(file="$(LIB)/top_heav.lib")
    env.libs.parameters.read(file="$(LIB)/par.lib")
    model = Model(env, file=str(Path(args.input).resolve()))
    alignment = Alignment(env)
    alignment.append_model(model, atom_files=str(Path(args.input).resolve()), align_codes="template")

    selected_indices = []
    applied = []
    for site in replacements:
        residue_id = _residue_id(site)
        residue = model.residues[residue_id]
        before = residue.pdb_name.strip()
        Selection(residue).mutate(residue_type=str(site["target"]).upper())
        selected_indices.append(residue.index)
        applied.append(
            {
                **site,
                "source": before,
                "modeller_residue_id": residue_id,
                "model_index": residue.index,
            }
        )

    alignment.append_model(model, align_codes="repaired")
    model.clear_topology()
    model.generate_topology(alignment["repaired"])
    model.transfer_xyz(alignment)
    model.build(initialize_xyz=False, build_method="INTERNAL_COORDINATES")

    atoms = Selection(*(model.residues[index] for index in selected_indices))
    model.restraints.clear()
    model.restraints.make(atoms, restraint_type="stereo", spline_on_site=False)
    ConjugateGradients().optimize(atoms, max_iterations=200, min_atom_shift=0.01)
    MolecularDynamics(cap_atom_shift=0.2, md_time_step=4.0).optimize(
        atoms,
        temperature=300.0,
        max_iterations=50,
        equilibrate=10,
    )
    ConjugateGradients().optimize(atoms, max_iterations=200, min_atom_shift=0.01)
    model.write(file=args.output)
    Path(args.report).write_text(
        json.dumps({"success": True, "engine": "MODELLER", "replacements": applied}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
