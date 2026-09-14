from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
from typing import Any
from uuid import uuid4

from rdkit import Chem

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.docker_runner import (
    DockerMount,
    DockerRunSpec,
    build_docker_command,
    registered_tool,
    write_registered_command_record,
)
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.runtime import cpu_process_limit, runs_root
from mn_ligand.workflows.docking import _RDKIT_3D_SCRIPT, load_compound_records


DEFAULT_OPENVS_IMAGE = "openvs:local"
OPENVS_PROTOCOLS = ("vsh", "vsx", "convergence")
REFERENCE_MODES = ("reference_guided", "pocket_center")
REFERENCE_SUFFIXES = {".sdf", ".mol", ".mol2", ".pdb"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _excluded_compound_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "excluded_compounds.tsv"
    rows_by_id: dict[str, dict[str, str]] = {}
    for payload_path in (run_dir / "result.json", run_dir / "metadata.json"):
        try:
            payload = json.loads(payload_path.read_text())
        except (OSError, TypeError, ValueError):
            payload = {}
        for row in payload.get("exclusion_details", []):
            compound_id = str(row.get("compound_id") or "").strip()
            if compound_id:
                rows_by_id[compound_id] = {
                    str(key): str(value or "") for key, value in row.items()
                }
        for value in payload.get("excluded_compound_ids", []):
            compound_id = str(value or "").strip()
            if compound_id and compound_id not in rows_by_id:
                rows_by_id[compound_id] = {
                    "compound_id": compound_id,
                    "stage": "persisted_engine_exclusion",
                    "reason": "Explicitly excluded from this engine campaign",
                    "log_file": "",
                }
    if path.is_file():
        with path.open(newline="", errors="replace") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                compound_id = str(row.get("compound_id") or "").strip()
                if compound_id:
                    rows_by_id[compound_id] = {
                        str(key): str(value or "") for key, value in row.items()
                    }
    return list(rows_by_id.values())


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")


def _protocol_xml(
    protocol: str,
    *,
    padding: float,
    reference_guided: bool,
) -> str:
    runmode = "VSH" if protocol == "convergence" else protocol.upper()
    reference_options = (
        ""
        if reference_guided
        else ' reference_pool="none" use_pharmacophore="0"'
    )
    if protocol == "convergence":
        protocol_options = (
            ' final_optH_mode="1" pre_optH_relax="1" entropy_method="Simple"'
            ' nrelax="30" nreport="1" final_exact_minimize="sc"'
        )
        stages = """
      <Stage repeats="8" npool="150" rmsdthreshold="2.0" pmut="0.2"
        maxiter="40" pack_cycles="75" smoothing="0.75"
        ramp_schedule="0.1,1.0"/>
      <Stage repeats="8" npool="150" rmsdthreshold="2.0" pmut="0.2"
        maxiter="40" pack_cycles="75" smoothing="0.375"
        ramp_schedule="0.1,1.0"/>"""
    elif protocol == "vsh":
        protocol_options = ' final_optH_mode="1" pre_optH_relax="1" entropy_method="Simple"'
        stages = ""
    else:
        protocol_options = ' aligner_fastmode="1" final_exact_minimize="ligandonly"'
        stages = ""
    return f"""<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="genpot_soft" weights="beta_cart">
      <Reweight scoretype="fa_rep" weight="0.2"/>
    </ScoreFunction>
    <ScoreFunction name="genpot" weights="beta_cart">
      <Reweight scoretype="coordinate_constraint" weight="0.1"/>
    </ScoreFunction>
  </SCOREFXNS>
  <MOVERS>
    <GALigandDock name="dock" scorefxn="genpot_soft" scorefxn_relax="genpot"
      runmode="{runmode}" premin_ligand="1"
      multiple_ligands_file="%%liglist%%" estimate_dG="1"
      padding="{padding:.3f}"{reference_options}{protocol_options}>{stages}
    </GALigandDock>
  </MOVERS>
  <PROTOCOLS>
    <Add mover="dock"/>
  </PROTOCOLS>
  <OUTPUT scorefxn="genpot"/>
</ROSETTASCRIPTS>
"""


_CENTER_PDB_SCRIPT = """from pathlib import Path
import sys

source, target = map(Path, sys.argv[1:3])
center = tuple(float(value) for value in sys.argv[3:6])
lines = source.read_text(errors="replace").splitlines()
coords = []
for line in lines:
    if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 54:
        coords.append(tuple(float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54))))
if not coords:
    raise SystemExit("Anchor PDB contains no coordinates")
centroid = tuple(sum(point[i] for point in coords) / len(coords) for i in range(3))
shift = tuple(center[i] - centroid[i] for i in range(3))
output = []
for line in lines:
    if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 54:
        xyz = tuple(float(line[start:end]) + shift[i] for i, (start, end) in enumerate(((30, 38), (38, 46), (46, 54))))
        line = f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}"
    if not line.startswith("END"):
        output.append(line)
target.write_text("\\n".join(output) + "\\n")
"""


_ROSETTA_PARAMS_VALIDATOR_SCRIPT = r'''"""Reject parameter files that Rosetta cannot load safely."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys


def atom_name_collisions(path: Path) -> dict[str, list[str]]:
    """Return atom names that collide after Rosetta's four-character limit."""
    normalized: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "ATOM":
            atom_name = fields[1]
            normalized[atom_name[:4]].append(atom_name)
    return {
        name: values
        for name, values in normalized.items()
        if len(values) > 1
    }


def main() -> int:
    path = Path(sys.argv[1])
    collisions = atom_name_collisions(path)
    if not collisions:
        return 0
    details = "; ".join(
        f"{name} <- {', '.join(values)}"
        for name, values in sorted(collisions.items())
    )
    print(
        "Rosetta atom-name collision after its four-character normalization: "
        + details,
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


_RUNNER_SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail
cd /workspace
source /opt/conda/etc/profile.d/conda.sh
conda activate openvs

ROSETTA_SCRIPTS="$ROSETTAHOME/source/bin/rosetta_scripts.linuxgccrelease"
EXTRACT_PDBS="$ROSETTAHOME/source/bin/extract_pdbs.linuxgccrelease"
MOL2GEN="$ROSETTAHOME/source/scripts/python/public/generic_potential/mol2genparams.py"
for required in "$ROSETTA_SCRIPTS" "$EXTRACT_PDBS" "$MOL2GEN"; do
  [[ -e "$required" ]] || { echo "Missing RosettaLigand executable: $required" >&2; exit 2; }
done

mkdir -p prepared/mol2 prepared/params prepared/anchor chunks native poses
: > prepared/ligand_list.txt
mkdir -p exclusion_logs
printf 'compound_id\tstage\treason\tlog_file\n' > excluded_compounds.tsv
record_exclusion() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> excluded_compounds.tsv
}

tail -n +2 input/compounds.tsv | while IFS=$'\t' read -r compound_id smiles; do
  [[ -n "$compound_id" && -n "$smiles" ]] || continue
  ligand_log="exclusion_logs/${compound_id}.preparation.log"
  printf '%s %s\n' "$smiles" "$compound_id" > "prepared/mol2/${compound_id}.smi"
  if ! python input/smiles_to_3d.py --smiles "$smiles" --name "$compound_id" \
    --output "prepared/mol2/${compound_id}.sdf" >"$ligand_log" 2>&1 \
    || [[ ! -s "prepared/mol2/${compound_id}.sdf" ]]; then
    record_exclusion "$compound_id" "3d_generation" \
      "RDKit ETKDGv3 did not produce a valid non-empty SDF" "$ligand_log"
    continue
  fi
  if ! obabel "prepared/mol2/${compound_id}.sdf" \
    -omol2 -O "prepared/mol2/${compound_id}.raw.mol2" \
    --partialcharge mmff94 -xl >/dev/null 2>>"$ligand_log" \
    || [[ ! -s "prepared/mol2/${compound_id}.raw.mol2" ]]; then
    record_exclusion "$compound_id" "mol2_conversion" \
      "Open Babel could not serialize the RDKit conformer as MOL2" "$ligand_log"
    continue
  fi
  if [[ "$OPENVS_PRESERVE_INPUT_PROTONATION" == "1" ]]; then
    if ! obabel "prepared/mol2/${compound_id}.raw.mol2" \
      -omol2 -O "prepared/mol2/${compound_id}.mol2" \
      --partialcharge mmff94 --minimize \
      --steps "$OPENVS_MINIMIZATION_STEPS" --sd -xl \
      >/dev/null 2>>"$ligand_log"; then
      record_exclusion "$compound_id" "minimization" \
        "Open Babel ligand minimization failed" "$ligand_log"
      continue
    fi
  else
    if ! obabel "prepared/mol2/${compound_id}.raw.mol2" \
      -omol2 -O "prepared/mol2/${compound_id}.mol2" \
      -p "$OPENVS_PH" --partialcharge mmff94 --minimize \
      --steps "$OPENVS_MINIMIZATION_STEPS" --sd -xl \
      >/dev/null 2>>"$ligand_log"; then
      record_exclusion "$compound_id" "minimization" \
        "Open Babel ligand protonation/minimization failed" "$ligand_log"
      continue
    fi
  fi
  if [[ ! -s "prepared/mol2/${compound_id}.mol2" ]]; then
    record_exclusion "$compound_id" "minimization" \
      "Open Babel produced an empty minimized MOL2" "$ligand_log"
    continue
  fi
  if ! python "$MOL2GEN" -s "prepared/mol2/${compound_id}.mol2" \
    --outdir prepared/params --resname LG1 --typenm "$compound_id" \
    --rename_atoms --infer_atomtypes >>"$ligand_log" 2>&1 \
    || [[ ! -s "prepared/params/${compound_id}.params" ]]; then
    record_exclusion "$compound_id" "rosetta_params" \
      "Rosetta params generation failed" "$ligand_log"
    continue
  fi
  if ! python input/validate_rosetta_params.py \
    "prepared/params/${compound_id}.params" >>"$ligand_log" 2>&1; then
    record_exclusion "$compound_id" "rosetta_params_validation" \
      "Generated Rosetta params contain atom names that collide after Rosetta four-character normalization" \
      "$ligand_log"
    continue
  fi
  printf '%s\n' "$compound_id" >> prepared/ligand_list.txt
done

[[ -s prepared/ligand_list.txt ]] || { echo "No ligands were prepared" >&2; exit 3; }
if [[ "$OPENVS_REFERENCE_MODE" == "reference_guided" ]]; then
  obabel "input/reference_ligand$OPENVS_REFERENCE_SUFFIX" \
    -f 1 -l 1 -omol2 -O prepared/anchor/reference_anchor.mol2 \
    -h --partialcharge mmff94 -xl >/dev/null 2>prepared/anchor/obabel.log
  python "$MOL2GEN" -s prepared/anchor/reference_anchor.mol2 \
    --outdir prepared/anchor --resname LG1 --typenm reference_anchor \
    --rename_atoms --infer_atomtypes >/dev/null
else
  first_ligand="$(head -n 1 prepared/ligand_list.txt)"
  python "$MOL2GEN" -s "prepared/mol2/${first_ligand}.mol2" \
    --outdir prepared/anchor --resname LG1 --typenm reference_anchor \
    --rename_atoms --infer_atomtypes >/dev/null
fi

# mol2genparams names files from the MOL2 molecule title, which may be the
# source compound ID. Normalize the workflow-internal anchor independently of
# that source identity so every placement mode has one stable contract.
generated_anchor_params="$(find prepared/anchor -maxdepth 1 -type f -name '*.params' -print -quit)"
generated_anchor_pdb="$(find prepared/anchor -maxdepth 1 -type f -name '*_0001.pdb' -print -quit)"
[[ -n "$generated_anchor_params" && -n "$generated_anchor_pdb" ]] || {
  echo "Rosetta anchor parameter generation produced no params/PDB files" >&2
  exit 3
}
if [[ "$generated_anchor_params" != "prepared/anchor/reference_anchor.params" ]]; then
  mv "$generated_anchor_params" prepared/anchor/reference_anchor.params
fi
if [[ "$generated_anchor_pdb" != "prepared/anchor/reference_anchor_0001.pdb" ]]; then
  mv "$generated_anchor_pdb" prepared/anchor/reference_anchor_0001.pdb
fi
if ! python input/validate_rosetta_params.py \
  prepared/anchor/reference_anchor.params >>prepared/anchor/obabel.log 2>&1; then
  echo "Rosetta reference/anchor params contain colliding atom names" >&2
  exit 3
fi
if [[ "$OPENVS_REFERENCE_MODE" == "reference_guided" ]]; then
  cp prepared/anchor/reference_anchor_0001.pdb prepared/anchor/anchor.pdb
else
  python input/center_pdb.py prepared/anchor/reference_anchor_0001.pdb \
    prepared/anchor/anchor.pdb "$OPENVS_CENTER_X" "$OPENVS_CENTER_Y" "$OPENVS_CENTER_Z"
fi
[[ -s prepared/anchor/reference_anchor.params && -s prepared/anchor/anchor.pdb ]] || {
  echo "Reference/anchor preparation failed" >&2
  exit 3
}

awk '/^(ATOM  |TER   )/{print}' input/receptor.pdb > prepared/receptor.pdb
awk '/^(ATOM  |HETATM|TER   )/{print}' prepared/anchor/anchor.pdb > prepared/anchor/anchor.clean.pdb
cat prepared/receptor.pdb prepared/anchor/anchor.clean.pdb > prepared/holo.pdb
printf 'END\n' >> prepared/holo.pdb

ligand_count="$(wc -l < prepared/ligand_list.txt)"
chunk_size="$(( (ligand_count + OPENVS_CPU_WORKERS - 1) / OPENVS_CPU_WORKERS ))"
split -d -a 3 -l "$chunk_size" prepared/ligand_list.txt chunks/ligands_

run_chunk() {
  chunk="$1"
  replicate="$2"
  seed="$3"
  replicate_name="$(printf 'replicate_%03d' "$replicate")"
  chunk_name="$(basename "$chunk")"
  outdir="native/$replicate_name/$chunk_name"
  mkdir -p "$outdir"
  flags="$outdir/params.flags"
  : > "$flags"
  while read -r ligand; do
    printf '%s\n' "-extra_res_fa prepared/params/${ligand}.params" >> "$flags"
  done < "$chunk"
  "$ROSETTA_SCRIPTS" \
    -s prepared/holo.pdb \
    -extra_res_fa prepared/anchor/reference_anchor.params \
    @"$flags" \
    -gen_potential -overwrite -beta_cart -no_autogen_cart_improper \
    -constant_seed -jran "$seed" \
    -missing_density_to_jump -multi_cool_annealer 10 \
    -parser:protocol input/dock.xml \
    -score::hb_don_strength hbdon_GENERIC_SC:1.45 \
    -score::hb_acc_strength hbacc_GENERIC_SP2SC:1.19 \
    -score::hb_acc_strength hbacc_GENERIC_SP3SC:1.19 \
    -score::hb_acc_strength hbacc_GENERIC_RINGSC:1.19 \
    -parser:script_vars "liglist=$chunk" \
    -out:prefix "${replicate_name}_${chunk_name}_" \
    -out:file:silent "$outdir/run.out" \
    -out:file:scorefile "$outdir/run.score.sc" \
    >"$outdir/rosetta.log" 2>&1
  "$EXTRACT_PDBS" \
    -in:file:silent "$outdir/run.out" \
    -extra_res_fa prepared/anchor/reference_anchor.params \
    @"$flags" -gen_potential -beta_cart -no_autogen_cart_improper \
    -missing_density_to_jump true \
    -out:prefix "poses/${replicate_name}_${chunk_name}_" \
    >"$outdir/extract.log" 2>&1
}
export -f run_chunk
export ROSETTA_SCRIPTS EXTRACT_PDBS

: > chunks/tasks.tsv
for replicate in ${OPENVS_REPLICATE_IDS:-$(seq "${OPENVS_REPLICATE_START:-1}" "$OPENVS_REPLICATES")}; do
  seed="$(( OPENVS_SEED_START + replicate - 1 ))"
  for chunk in chunks/ligands_*; do
    printf '%s\t%s\t%s\n' "$chunk" "$replicate" "$seed" >> chunks/tasks.tsv
  done
done

status=0
xargs -P "$OPENVS_CPU_WORKERS" -n 3 bash -c 'run_chunk "$1" "$2" "$3"' _ \
  < chunks/tasks.tsv || status=1
[[ "$status" -eq 0 ]] || { echo "One or more RosettaLigand replicate chunks failed" >&2; exit 4; }
"""


def _parse_score_file(path: Path, protocol: str) -> list[dict[str, Any]]:
    header: list[str] = []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("SCORE:"):
            continue
        values = line.split()[1:]
        if not values:
            continue
        if values[0] == "total_score":
            header = values
            continue
        if not header or len(values) != len(header):
            continue
        native = dict(zip(header, values))

        def number(name: str) -> float | None:
            try:
                return float(native[name])
            except (KeyError, TypeError, ValueError):
                return None

        rows.append(
            {
                "compound_id": native.get("ligandname", ""),
                "engine": "openvs",
                "protocol": protocol.upper(),
                "estimated_dg_reu": number("dG"),
                "enthalpy_reu": number("dH"),
                "entropy_term_reu": number("-TdS"),
                "total_score_reu": number("total_score"),
                "ligand_score_reu": number("ligscore"),
                "receptor_score_reu": number("recscore"),
                "ligand_rmsd_angstrom": number("lig_rms"),
                "runtime_seconds": number("time"),
                "description": native.get("description", ""),
                "native_score_file": path.as_posix(),
                "pose_file": "",
            }
        )
    return rows


def _pose_for_row(run_dir: Path, row: dict[str, Any], index: int) -> Path | None:
    description = str(row.get("description") or "")
    candidates = sorted((run_dir / "poses").glob(f"*{description}*.pdb")) if description else []
    if not candidates:
        candidates = sorted((run_dir / "poses").glob("*.pdb"))
        return candidates[index] if index < len(candidates) else None
    return candidates[0]


def _ligand_coordinates(path: Path) -> dict[str, tuple[float, float, float]]:
    coordinates: dict[str, tuple[float, float, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("HETATM") or len(line) < 78:
            continue
        element = line[76:78].strip().upper()
        if element == "H":
            continue
        atom_name = line[12:16].strip()
        if atom_name:
            coordinates[atom_name] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
    if not coordinates:
        raise ValueError(f"No ligand heavy-atom coordinates in {path.name}")
    return coordinates


def _direct_pose_rmsd(first: Path, second: Path) -> float:
    left = _ligand_coordinates(first)
    right = _ligand_coordinates(second)
    if left.keys() != right.keys():
        raise ValueError("RosettaLigand replicate poses do not share the same named heavy atoms")
    square_distance = sum(
        (left[name][axis] - right[name][axis]) ** 2
        for name in left
        for axis in range(3)
    )
    return math.sqrt(square_distance / len(left))


def _ordered_ligand_coordinates(path: Path) -> list[tuple[float, float, float]]:
    coordinates: list[tuple[float, float, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("HETATM") or len(line) < 78:
            continue
        if line[76:78].strip().upper() == "H":
            continue
        coordinates.append(
            (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        )
    return coordinates


def _symmetry_pose_rmsd(
    run_dir: Path,
    compound_id: str,
    first: Path,
    second: Path,
) -> float:
    template_path = run_dir / "prepared" / "mol2" / f"{compound_id}.mol2"
    try:
        template = Chem.MolFromMol2File(
            str(template_path), sanitize=False, removeHs=False
        )
        if template is None:
            raise ValueError("RDKit could not read the prepared MOL2")
        template = Chem.RemoveHs(template, sanitize=False)
        mappings = template.GetSubstructMatches(
            template, uniquify=False, maxMatches=100_000
        )
        left = _ordered_ligand_coordinates(first)
        right = _ordered_ligand_coordinates(second)
        if not mappings or len(left) != template.GetNumAtoms() or len(right) != len(left):
            raise ValueError("Pose/template heavy-atom counts differ")
        return min(
            math.sqrt(
                sum(
                    (left[mapping[index]][axis] - right[index][axis]) ** 2
                    for index in range(len(right))
                    for axis in range(3)
                )
                / len(right)
            )
            for mapping in mappings
        )
    except (OSError, RuntimeError, ValueError):
        return _direct_pose_rmsd(first, second)


def _cluster_replicate_poses(
    run_dir: Path,
    rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> tuple[dict[int, int], list[list[int]]]:
    adjacency = {index: set() for index in range(len(rows))}
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            rmsd = _symmetry_pose_rmsd(
                run_dir,
                str(rows[left].get("compound_id") or ""),
                run_dir / str(rows[left]["pose_file"]),
                run_dir / str(rows[right]["pose_file"]),
            )
            if rmsd <= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)
    clusters: list[list[int]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: list[int] = []
        unseen.remove(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        clusters.append(sorted(component))
    clusters.sort(key=lambda values: (-len(values), values[0]))
    assignments = {
        row_index: cluster_index
        for cluster_index, members in enumerate(clusters, start=1)
        for row_index in members
    }
    return assignments, clusters


def _convergence_tables(
    run_dir: Path,
    rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    running: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    compound_ids = sorted({str(row.get("compound_id") or "") for row in rows})
    for compound_id in compound_ids:
        compound_rows = [
            row for row in rows
            if row.get("compound_id") == compound_id and row.get("pose_file")
        ]
        best_by_replicate: list[dict[str, Any]] = []
        for replicate in sorted({int(row["replicate"]) for row in compound_rows}):
            candidates = [row for row in compound_rows if int(row["replicate"]) == replicate]
            best_by_replicate.append(
                min(
                    candidates,
                    key=lambda row: math.inf
                    if row.get("estimated_dg_reu") is None
                    else float(row["estimated_dg_reu"]),
                )
            )
        values = [
            float(row["estimated_dg_reu"])
            for row in best_by_replicate
            if row.get("estimated_dg_reu") is not None
        ]
        if not values:
            continue
        assignments, clusters = _cluster_replicate_poses(
            run_dir, best_by_replicate, threshold=threshold
        )
        for index, row in enumerate(best_by_replicate):
            row["cluster_id"] = assignments[index]
        dominant = clusters[0]
        dominant_rows = [best_by_replicate[index] for index in dominant]
        dominant_values = [float(row["estimated_dg_reu"]) for row in dominant_rows]
        median = statistics.median(dominant_values)
        representative = min(
            dominant_rows,
            key=lambda row: (
                round(abs(float(row["estimated_dg_reu"]) - median), 12),
                int(row.get("replicate") or 1),
            ),
        )
        representatives.append(representative)
        for count in range(1, len(values) + 1):
            prefix = values[:count]
            running.append(
                {
                    "compound_id": compound_id,
                    "replicate_count": count,
                    "running_mean_dg_reu": statistics.mean(prefix),
                    "running_sample_sd_dg_reu": statistics.stdev(prefix)
                    if count > 1 else 0.0,
                }
            )
        sample_sd = statistics.stdev(values) if len(values) > 1 else 0.0
        first_half_mean = statistics.mean(values[: max(1, len(values) // 2)])
        mean_shift = abs(statistics.mean(values) - first_half_mean)
        dominant_fraction = len(dominant) / len(best_by_replicate)
        converged = (
            len(values) >= 5
            and sample_sd <= 2.0
            and mean_shift <= 1.0
            and dominant_fraction >= 0.8
        )
        summaries.append(
            {
                "compound_id": compound_id,
                "replicate_count": len(values),
                "mean_dg_reu": statistics.mean(values),
                "sample_sd_dg_reu": sample_sd,
                "min_dg_reu": min(values),
                "max_dg_reu": max(values),
                "cluster_threshold_angstrom": threshold,
                "pose_cluster_count": len(clusters),
                "dominant_cluster_size": len(dominant),
                "dominant_cluster_fraction": dominant_fraction,
                "running_mean_shift_reu": mean_shift,
                "converged": converged,
                "representative_replicate": representative["replicate"],
                "representative_dg_reu": representative["estimated_dg_reu"],
                "representative_pose_file": representative["pose_file"],
            }
        )
    return summaries, running, representatives


def finalize_openvs_docking_job(run_dir: Path, *, returncode: int) -> JobRecord:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    protocol = str(metadata.get("protocol") or "vsh").lower()
    compound_count = int(metadata.get("compound_count") or 0)
    replicates = int(metadata.get("replicates") or 1)
    seed_start = int(metadata.get("seed_start") or 1001)
    cluster_threshold = float(metadata.get("cluster_threshold_angstrom") or 2.0)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_path.touch(exist_ok=True)
    stderr_path.touch(exist_ok=True)
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    recovery_run_dirs: list[Path] = []
    for recovery_run_id in metadata.get("recovery_run_ids", []):
        recovery_run_id = str(recovery_run_id or "").strip()
        if not recovery_run_id or Path(recovery_run_id).name != recovery_run_id:
            continue
        recovery_run_dir = (run_dir.parent / recovery_run_id).resolve()
        recovery_metadata_path = recovery_run_dir / "metadata.json"
        if recovery_run_dir.parent != run_dir.parent or not recovery_metadata_path.is_file():
            continue
        recovery_metadata = json.loads(recovery_metadata_path.read_text())
        if (
            str(recovery_metadata.get("recovery_of_run_id") or "") == run_dir.name
            and str(recovery_metadata.get("status") or "") == "completed"
        ):
            recovery_run_dirs.append(recovery_run_dir)

    excluded_rows = _excluded_compound_rows(run_dir)
    for recovery_run_dir in recovery_run_dirs:
        excluded_rows.extend(_excluded_compound_rows(recovery_run_dir))
    excluded_rows = list(
        {
            str(row.get("compound_id") or "").strip(): row
            for row in excluded_rows
            if str(row.get("compound_id") or "").strip()
        }.values()
    )
    excluded_ids = {
        str(row.get("compound_id") or "").strip() for row in excluded_rows
    }
    exclusions_path = run_dir / "excluded_compounds.tsv"
    if excluded_rows:
        with exclusions_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["compound_id", "stage", "reason", "log_file"],
                delimiter="\t",
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(excluded_rows)
    eligible_compound_count = max(0, compound_count - len(excluded_ids))

    rows: list[dict[str, Any]] = []
    for score_path in sorted((run_dir / "native").rglob("run.score.sc")):
        parsed = _parse_score_file(score_path, protocol)
        relative_parts = score_path.relative_to(run_dir / "native").parts
        replicate_match = next(
            (
                re.fullmatch(r"replicate_(\d+)", part)
                for part in relative_parts
                if part.startswith("replicate_")
            ),
            None,
        )
        replicate = int(replicate_match.group(1)) if replicate_match else 1
        ranks: dict[str, int] = {}
        for row in parsed:
            row["native_score_file"] = score_path.relative_to(run_dir).as_posix()
            row["replicate"] = replicate
            row["seed"] = seed_start + replicate - 1
            compound_id = str(row.get("compound_id") or "")
            ranks[compound_id] = ranks.get(compound_id, 0) + 1
            row["pose_rank"] = ranks[compound_id]
            row["cluster_id"] = ""
        rows.extend(parsed)
    used_poses: set[Path] = set()
    for index, row in enumerate(rows):
        source = _pose_for_row(run_dir, row, index)
        if source is None or source in used_poses:
            continue
        used_poses.add(source)
        compound_id = _safe_id(str(row.get("compound_id") or f"compound-{index + 1}"))
        target = results_dir / (
            f"{compound_id}.replicate_{int(row['replicate']):03d}."
            f"pose_{int(row['pose_rank']):02d}.complex.pdb"
        )
        shutil.copy2(source, target)
        row["pose_file"] = target.relative_to(run_dir).as_posix()

    existing_recovery_keys = {
        (
            str(row.get("compound_id") or ""),
            int(row.get("replicate") or 1),
            int(row.get("pose_rank") or 1),
        )
        for row in rows
    }
    for recovery_run_dir in recovery_run_dirs:
        recovery_scores_path = recovery_run_dir / "openvs_scores.csv"
        if not recovery_scores_path.is_file():
            continue
        with recovery_scores_path.open(newline="", errors="replace") as handle:
            recovery_rows = list(csv.DictReader(handle))
        for recovery_row in recovery_rows:
            compound_id = str(recovery_row.get("compound_id") or "").strip()
            replicate = int(recovery_row.get("replicate") or 1)
            pose_rank = int(recovery_row.get("pose_rank") or 1)
            key = (compound_id, replicate, pose_rank)
            source_pose = recovery_run_dir / str(recovery_row.get("pose_file") or "")
            if (
                not compound_id
                or key in existing_recovery_keys
                or not source_pose.is_file()
                or source_pose.stat().st_size <= 0
            ):
                continue
            target = results_dir / (
                f"{_safe_id(compound_id)}.replicate_{replicate:03d}."
                f"pose_{pose_rank:02d}.complex.pdb"
            )
            shutil.copy2(source_pose, target)
            recovery_row["pose_file"] = target.relative_to(run_dir).as_posix()
            recovery_row["native_score_file"] = (
                f"recovery:{recovery_run_dir.name}/"
                f"{recovery_row.get('native_score_file') or ''}"
            )
            rows.append(recovery_row)
            existing_recovery_keys.add(key)

    score_path = run_dir / "openvs_scores.csv"
    fields = [
        "compound_id",
        "engine",
        "protocol",
        "replicate",
        "seed",
        "pose_rank",
        "cluster_id",
        "estimated_dg_reu",
        "enthalpy_reu",
        "entropy_term_reu",
        "total_score_reu",
        "ligand_score_reu",
        "receptor_score_reu",
        "ligand_rmsd_angstrom",
        "runtime_seconds",
        "description",
        "native_score_file",
        "pose_file",
    ]
    ranked = sorted(
        (row for row in rows if row.get("pose_file")),
        key=lambda row: (
            math.inf if row.get("estimated_dg_reu") is None else float(row["estimated_dg_reu"]),
            math.inf if row.get("total_score_reu") is None else float(row["total_score_reu"]),
        ),
    )
    convergence_summaries: list[dict[str, Any]] = []
    convergence_running: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    convergence_path = run_dir / (
        "openvs_convergence.csv"
        if protocol == "convergence"
        else "openvs_replicate_summary.csv"
    )
    running_path = run_dir / "openvs_running_statistics.csv"
    if replicates > 1 and ranked:
        try:
            convergence_summaries, convergence_running, representative_rows = (
                _convergence_tables(
                    run_dir,
                    rows,
                    threshold=cluster_threshold,
                )
            )
        except (OSError, ValueError) as exc:
            (run_dir / "convergence_error.log").write_text(str(exc) + "\n")
            convergence_summaries = []
            representative_rows = []
        if convergence_summaries:
            with convergence_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(convergence_summaries[0])
                )
                writer.writeheader()
                writer.writerows(convergence_summaries)
            with running_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(convergence_running[0])
                )
                writer.writeheader()
                writer.writerows(convergence_running)

    with score_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    best_path = run_dir / "best_openvs_complex.pdb"
    handoff_ranked = sorted(
        representative_rows or ranked,
        key=lambda row: math.inf
        if row.get("estimated_dg_reu") is None
        else float(row["estimated_dg_reu"]),
    )
    if handoff_ranked:
        shutil.copy2(run_dir / handoff_ranked[0]["pose_file"], best_path)
    completed_pairs = {
        (str(row.get("compound_id") or ""), int(row.get("replicate") or 1))
        for row in rows
    }
    complete_rows = len(
        {
            compound_id
            for compound_id in {pair[0] for pair in completed_pairs}
            if sum(pair[0] == compound_id for pair in completed_pairs) == replicates
        }
    )
    pose_count = len(ranked)
    expected_pairs = eligible_compound_count * replicates
    posed_pairs = {
        (str(row.get("compound_id") or ""), int(row.get("replicate") or 1))
        for row in ranked
    }
    complete_success = (
        returncode == 0
        and eligible_compound_count > 0
        and len(completed_pairs) == expected_pairs
        and len(posed_pairs) == expected_pairs
        and (replicates == 1 or bool(convergence_summaries))
    )
    has_usable_output = bool(completed_pairs) and bool(posed_pairs)
    partial_success = bool(excluded_ids) or (
        has_usable_output and not complete_success
    )
    success = complete_success or has_usable_output
    stderr = stderr_path.read_text(errors="replace")
    incomplete_message = stderr[-4000:] or (
        f"RosettaLigand produced {len(completed_pairs)}/{expected_pairs} replicate score "
        f"sets and {len(posed_pairs)}/{expected_pairs} replicate pose sets"
    )
    error = ""
    if not success:
        error = incomplete_message
    warning = incomplete_message if success and not complete_success else ""
    result = {
        "success": success,
        "partial_success": partial_success,
        "returncode": int(returncode),
        "compound_count": compound_count,
        "eligible_compound_count": eligible_compound_count,
        "excluded_compounds": len(excluded_ids),
        "excluded_compound_ids": sorted(excluded_ids),
        "exclusion_details": excluded_rows,
        "replicates": replicates,
        "completed_compounds": complete_rows,
        "pose_count": pose_count,
        "failed_compounds": max(
            0, eligible_compound_count - min(complete_rows, pose_count)
        ),
        "progress": {"completed": len(posed_pairs), "total": expected_pairs},
        "protocol": protocol.upper(),
        "score_units": "Rosetta energy units (REU; relative ranking only)",
        "best_compound_id": handoff_ranked[0]["compound_id"] if handoff_ranked else "",
        "best_estimated_dg_reu": handoff_ranked[0]["estimated_dg_reu"]
        if handoff_ranked else None,
        "convergence_compounds": len(convergence_summaries),
        "replicate_summary_compounds": len(convergence_summaries),
        "converged_compounds": sum(
            1 for row in convergence_summaries if row["converged"]
        ),
        "warning": warning,
        "error": error,
    }
    _write_json(run_dir / "result.json", result)
    artifacts = [
        ArtifactRef.from_path(run_dir, score_path, "screening_result", role="ranked_scores"),
        ArtifactRef.from_path(run_dir, score_path, "docking_scores", role="ranked_scores"),
        ArtifactRef.from_path(run_dir, stdout_path, "job_log", role="stdout", checksum=False),
        ArtifactRef.from_path(run_dir, stderr_path, "job_log", role="stderr", checksum=False),
    ]
    if exclusions_path.is_file():
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                exclusions_path,
                "compound_exclusions",
                role="excluded_compounds",
                metadata={"excluded_compound_count": len(excluded_ids)},
            )
        )
    if convergence_path.is_file():
        artifacts.extend(
            [
                ArtifactRef.from_path(
                    run_dir,
                    convergence_path,
                    "convergence_result",
                    role=(
                        "convergence_summary"
                        if protocol == "convergence"
                        else "replicate_summary"
                    ),
                    metadata={
                        "rmsd_method": (
                            "symmetry-aware heavy-atom RMSD in the fixed receptor frame"
                        )
                    },
                ),
                ArtifactRef.from_path(
                    run_dir,
                    running_path,
                    "convergence_result",
                    role="running_statistics",
                ),
            ]
        )
    for row in ranked:
        pose_path = run_dir / str(row["pose_file"])
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                pose_path,
                "pose_set",
                role="docked_complex",
                label=str(row["compound_id"]),
                metadata={"engine": "openvs", "protocol": protocol.upper()},
            )
        )
    if best_path.is_file():
        artifacts.append(
            ArtifactRef.from_path(
                run_dir,
                best_path,
                "docked_complex",
                role="best_ranked_complex",
                metadata={
                    "compound_id": handoff_ranked[0]["compound_id"],
                    "estimated_dg_reu": handoff_ranked[0]["estimated_dg_reu"],
                    "selection": "dominant-cluster representative"
                    if representative_rows else "lowest estimated dG",
                },
            )
        )
    write_artifact_manifest(run_dir, artifacts)
    completed = _utc_now_iso()
    metadata.update(
        {
            "status": "completed" if success else "failed",
            "partial_success": partial_success,
            "updated_at": completed,
            "completed_at": completed,
        }
    )
    if error:
        metadata["error"] = error
    if warning:
        metadata["warning"] = warning
    _write_json(metadata_path, metadata)
    recovery_of_run_id = str(metadata.get("recovery_of_run_id") or "").strip()
    recovered_indices = metadata.get("recovery_repetition_indices")
    recovered_indices = (
        [int(value) for value in recovered_indices if int(value) > 0]
        if isinstance(recovered_indices, list)
        else []
    )
    if success and recovery_of_run_id and recovered_indices:
        source_metadata_path = run_dir.parent / recovery_of_run_id / "metadata.json"
        if source_metadata_path.is_file():
            source_metadata = json.loads(source_metadata_path.read_text())
            combined_indices = {
                int(value)
                for value in source_metadata.get("recovered_repetition_indices", [])
                if str(value).isdigit() and int(value) > 0
            }
            combined_indices.update(recovered_indices)
            source_metadata["recovered_repetition_indices"] = sorted(combined_indices)
            source_metadata["recovery_run_ids"] = list(
                dict.fromkeys(
                    [
                        *source_metadata.get("recovery_run_ids", []),
                        metadata["run_id"],
                    ]
                )
            )
            source_metadata["updated_at"] = completed
            _write_json(source_metadata_path, source_metadata)
    return JobRecord.load(run_dir, task_group="docking")


def run_openvs_docking_job(
    *,
    receptor_path: Path,
    target_artifact: ArtifactRef,
    compound_paths: Sequence[Path],
    compound_artifacts: Sequence[ArtifactRef],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    box_mode: str = "fixed",
    box_padding_angstrom: float | None = None,
    protocol: str = "vsh",
    reference_mode: str = "reference_guided",
    reference_ligand_path: Path | None = None,
    reference_ligand_artifact: ArtifactRef | None = None,
    image: str = DEFAULT_OPENVS_IMAGE,
    cpu_workers: int | None = None,
    ph: float = 7.4,
    preserve_input_protonation: bool = False,
    conformers: int = 20,
    minimization_steps: int = 2000,
    padding: float = 4.0,
    replicates: int = 1,
    seed_start: int = 1001,
    cluster_threshold_angstrom: float = 2.0,
    maximum_compounds: int = 0,
    launch_campaign_id: str = "",
    launch_campaign_label: str = "",
    campaign_purpose: str = "",
    enqueue_only: bool = False,
) -> JobRecord:
    protocol = str(protocol).strip().lower()
    if protocol not in OPENVS_PROTOCOLS:
        raise ValueError(f"Unsupported RosettaLigand protocol: {protocol}")
    reference_mode = str(reference_mode).strip().lower()
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(f"Unsupported RosettaLigand reference mode: {reference_mode}")
    if reference_mode == "reference_guided" and reference_ligand_path is None:
        raise ValueError("Reference-guided RosettaLigand requires a coordinate-bearing reference ligand")
    reference_suffix = ""
    if reference_ligand_path is not None:
        reference_suffix = reference_ligand_path.suffix.lower()
        if reference_suffix not in REFERENCE_SUFFIXES:
            raise ValueError("RosettaLigand reference ligands must be SDF, MOL, MOL2, or PDB")
    records = load_compound_records(compound_paths, maximum=max(0, int(maximum_compounds)))
    replicates = max(1, min(int(replicates), 100))
    seed_start = int(seed_start)
    if seed_start < 1 or seed_start + replicates >= 2_147_483_647:
        raise ValueError("RosettaLigand seed range must contain positive 32-bit integers")
    if float(cluster_threshold_angstrom) <= 0:
        raise ValueError("RosettaLigand cluster threshold must be positive")
    requested_cpu_workers = cpu_process_limit() if cpu_workers is None else int(cpu_workers)
    cpu_workers = max(
        1,
        min(
            requested_cpu_workers,
            cpu_process_limit(),
            len(records) * replicates,
            128,
        ),
    )
    if not 0.0 <= float(ph) <= 14.0:
        raise ValueError("RosettaLigand preparation pH must be between 0 and 14")
    if float(padding) < 1.0:
        raise ValueError("RosettaLigand search padding must be at least 1 Å")

    run_id = str(uuid4())
    run_dir = runs_root() / "docking" / run_id
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "native").mkdir()
    (run_dir / "results").mkdir()
    receptor_local = input_dir / "receptor.pdb"
    receptor_local.write_bytes(receptor_path.read_bytes())
    if reference_ligand_path is not None:
        (input_dir / f"reference_ligand{reference_suffix}").write_bytes(
            reference_ligand_path.read_bytes()
        )
    compounds_path = input_dir / "compounds.tsv"
    with compounds_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("compound_id", "smiles"))
        writer.writerows((record["compound_id"], record["smiles"]) for record in records)
    (input_dir / "dock.xml").write_text(
        _protocol_xml(
            protocol,
            padding=float(padding),
            reference_guided=reference_mode == "reference_guided",
        )
    )
    (input_dir / "center_pdb.py").write_text(_CENTER_PDB_SCRIPT)
    (input_dir / "smiles_to_3d.py").write_text(_RDKIT_3D_SCRIPT)
    (input_dir / "validate_rosetta_params.py").write_text(
        _ROSETTA_PARAMS_VALIDATOR_SCRIPT
    )
    runner_path = input_dir / "run_openvs.sh"
    runner_path.write_text(_RUNNER_SCRIPT)

    environment = {
        "OPENVS_PROTOCOL": protocol.upper(),
        "OPENVS_REFERENCE_MODE": reference_mode,
        "OPENVS_REFERENCE_SUFFIX": reference_suffix,
        "OPENVS_CPU_WORKERS": str(cpu_workers),
        "OPENVS_REPLICATES": str(replicates),
        "OPENVS_SEED_START": str(seed_start),
        "OPENVS_PH": f"{float(ph):.2f}",
        "OPENVS_PRESERVE_INPUT_PROTONATION": (
            "1" if preserve_input_protonation else "0"
        ),
        "OPENVS_CONFORMERS": str(max(1, int(conformers))),
        "OPENVS_MINIMIZATION_STEPS": str(max(1, int(minimization_steps))),
        "OPENVS_CENTER_X": f"{float(center[0]):.4f}",
        "OPENVS_CENTER_Y": f"{float(center[1]):.4f}",
        "OPENVS_CENTER_Z": f"{float(center[2]):.4f}",
    }
    command = build_docker_command(
        DockerRunSpec(
            tool=registered_tool("openvs", image=image),
            command=("bash", "/workspace/input/run_openvs.sh"),
            mounts=(DockerMount(run_dir, "/workspace"),),
            environment=environment,
            gpu_enabled=False,
            use_host_user=False,
        )
    )
    now = _utc_now_iso()
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "job_type": "openvs_docking",
        "workflow": "openvs_docking",
        "operation": "docking",
        "status": "queued" if enqueue_only else "running",
        "engine": "openvs",
        "tool": "RosettaLigand",
        "parent_run_id": target_artifact.run_id,
        "prepared_target_run_id": target_artifact.run_id,
        "compound_run_ids": list(dict.fromkeys(item.run_id for item in compound_artifacts)),
        "reference_ligand_run_id": reference_ligand_artifact.run_id if reference_ligand_artifact else "",
        "compound_count": len(records),
        "launch_campaign_id": str(launch_campaign_id),
        "launch_campaign_label": str(launch_campaign_label),
        "campaign_purpose": str(campaign_purpose),
        "protocol": protocol,
        "reference_mode": reference_mode,
        "preserve_input_protonation": bool(preserve_input_protonation),
        "center": {axis: float(value) for axis, value in zip("xyz", center)},
        "size": {axis: float(value) for axis, value in zip("xyz", size)},
        "box_mode": str(box_mode),
        "box_padding_angstrom": (
            float(box_padding_angstrom) if box_padding_angstrom is not None else None
        ),
        "padding_angstrom": float(padding),
        "ph": float(ph),
        "conformers": max(1, int(conformers)),
        "minimization_steps": max(1, int(minimization_steps)),
        "cpu_workers": cpu_workers,
        "replicates": replicates,
        "seed_start": seed_start,
        "cluster_threshold_angstrom": float(cluster_threshold_angstrom),
        "docker_image": image,
        "gpu_queued": False,
        "created_at": now,
        "updated_at": now,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "input.json",
        {
            "target_artifact": target_artifact.to_dict(),
            "compound_artifacts": [artifact.to_dict() for artifact in compound_artifacts],
            "reference_ligand_artifact": (
                reference_ligand_artifact.to_dict() if reference_ligand_artifact else None
            ),
            "parameters": {
                key: metadata[key]
                for key in (
                    "protocol",
                    "reference_mode",
                    "compound_count",
                    "center",
                    "size",
                    "box_mode",
                    "box_padding_angstrom",
                    "padding_angstrom",
                    "ph",
                    "conformers",
                    "minimization_steps",
                    "cpu_workers",
                    "replicates",
                    "seed_start",
                    "cluster_threshold_angstrom",
                    "launch_campaign_id",
                    "launch_campaign_label",
                    "campaign_purpose",
                )
            },
            "score_units": "Rosetta energy units (REU; relative ranking only)",
            "command": [
                value.replace(str(run_dir), "<run-dir>") for value in command
            ],
        },
    )
    write_registered_command_record(
        run_dir, tool_id="openvs", commands=(command,), image=image
    )
    if enqueue_only:
        resources = registered_tool("openvs", image=image).resources.to_dict()
        resources.update(
            {
                "gpu": False,
                "cpu_threads": cpu_workers,
                "ram_gb": max(
                    int(resources.get("ram_gb") or 0),
                    cpu_workers * 4,
                ),
            }
        )
        metadata.update(
            {
                "queued_at": now,
                "queued_command": command,
                "resources": resources,
                "worker_finalizer": "openvs_docking",
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        write_artifact_manifest(run_dir, [])
        return JobRecord.load(run_dir, task_group="docking")
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        returncode = int(process.returncode)
        stdout, stderr = process.stdout or "", process.stderr or ""
    except Exception as exc:
        returncode, stdout, stderr = -1, "", str(exc)
    (run_dir / "stdout.log").write_text(stdout)
    (run_dir / "stderr.log").write_text(stderr)
    return finalize_openvs_docking_job(run_dir, returncode=returncode)


def queue_openvs_docking_job(**parameters: Any) -> JobRecord:
    return run_openvs_docking_job(enqueue_only=True, **parameters)
