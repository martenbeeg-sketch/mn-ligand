"""Native MODELLER runner for a short C-terminal extension.

This file intentionally depends only on MODELLER and the Python standard
library so it can be executed by the isolated mn-ligand-modeller interpreter.
"""

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
    parser.add_argument("--chain", required=True)
    parser.add_argument("--first-new", required=True, type=int)
    parser.add_argument("--last-new", required=True, type=int)
    parser.add_argument("--models", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    log.none()
    chain = args.chain
    extension_length = args.last_new - args.first_new + 1

    class TerminalExtensionModel(AutoModel):
        def select_atoms(self):
            # Target models can be renumbered from 1 even when the template uses
            # author residue numbers (for example, a trimmed chain 17-262).
            # The extension is always the final aligned sequence segment, so select
            # it positionally and restore author numbering during normalization.
            return Selection(*list(self.residues)[-extension_length:])

    env = Environ()
    env.io.atom_files_directory = [str(Path(args.template).resolve().parent)]
    model = TerminalExtensionModel(
        env,
        alnfile=args.alignment,
        knowns="template",
        sequence="extended",
        assess_methods=(assess.DOPE, assess.GA341),
    )
    model.starting_model = 1
    model.ending_model = args.models
    model.md_level = refine.slow
    model.make()
    outputs = []
    for item in model.outputs:
        outputs.append(
            {
                "name": str(item.get("name") or ""),
                "failure": str(item.get("failure") or ""),
                "dope": item.get("DOPE score"),
                "ga341": item.get("GA341 score"),
            }
        )
    Path(args.output).write_text(json.dumps({"models": outputs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
