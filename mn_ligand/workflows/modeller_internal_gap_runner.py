"""Native MODELLER runner for internal missing-residue ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modeller import Environ, Selection, log
from modeller.automodel import AutoModel, assess, refine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--ranges", required=True)
    parser.add_argument("--models", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    log.none()
    ranges = json.loads(Path(args.ranges).read_text())

    class InternalGapModel(AutoModel):
        def select_atoms(self):
            residues = list(self.residues)
            selected = []
            for item in ranges:
                selected.extend(
                    residues[int(item["target_start"]) - 1 : int(item["target_end"])]
                )
            return Selection(*selected)

    env = Environ()
    env.io.atom_files_directory = [str(Path(args.template).resolve().parent)]
    model = InternalGapModel(
        env,
        alnfile=args.alignment,
        knowns="template",
        sequence="repaired",
        assess_methods=(assess.DOPE, assess.GA341),
    )
    model.starting_model = 1
    model.ending_model = int(args.models)
    model.md_level = refine.slow
    model.make()
    outputs = [
        {
            "name": str(item.get("name") or ""),
            "failure": str(item.get("failure") or ""),
            "dope": item.get("DOPE score"),
            "ga341": item.get("GA341 score"),
        }
        for item in model.outputs
    ]
    Path(args.output).write_text(
        json.dumps({"models": outputs, "ranges": ranges}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
