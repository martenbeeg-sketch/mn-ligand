#!/usr/bin/env python3
"""Repair and validate one prepared-target PDB without publishing it.

The caller supplies explicit staging paths.  Publication (backup, atomic
replacement, and artifact-manifest update) remains a separate operation so a
failed repair can never damage the current target artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mn_ligand.ligandx.lib.chemistry.preparation.protein import ProteinPreparer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = ProteinPreparer().repair_imported_structure(
        args.input.read_text(),
        add_missing_residues=False,
        refine_rebuilt_positions=True,
    )
    report = result["report"]
    if not report["target_validation"]["valid"]:
        raise RuntimeError("Target repair did not pass publication validation")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result["pdb_data"])
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "report": str(args.report),
                "platform": report["refinement"].get("platform"),
                "validation": report["target_validation"],
                "displacement": report["refinement"].get("displacement"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
