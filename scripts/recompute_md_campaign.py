#!/usr/bin/env python3
"""Recompute several completed MD analyses sequentially and verify duration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from recompute_md_analysis import _resolve_analysis_dir
from mn_ligand.runtime import runs_root


def _analysis_end_ns(analysis_dir: Path) -> float:
    report = json.loads((analysis_dir / "result.json").read_text())
    ends = [
        max(row.get("time_ns") or [0.0])
        for row in report.get("replica_series") or []
        if isinstance(row, dict)
    ]
    return min(float(value) for value in ends) if ends else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow_ids", nargs="+")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--minimum-end-ns", type=float, default=90.0)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    recompute = project_dir / "scripts" / "recompute_md_analysis.py"
    root = Path(runs_root())
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("NUMEXPR_NUM_THREADS", "1")
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    failures: list[str] = []
    for index, workflow_id in enumerate(args.workflow_ids, start=1):
        print(
            f"[{index}/{len(args.workflow_ids)}] workflow={workflow_id}",
            flush=True,
        )
        command = [
            sys.executable,
            str(recompute),
            workflow_id,
            "--workers",
            str(max(1, args.workers)),
        ]
        completed = subprocess.run(
            command,
            cwd=project_dir,
            env=environment,
            check=False,
        )
        if completed.returncode:
            failures.append(f"{workflow_id}: exit {completed.returncode}")
            continue
        analysis_dir = _resolve_analysis_dir(root, workflow_id)
        end_ns = _analysis_end_ns(analysis_dir)
        print(
            f"verified workflow={workflow_id} minimum_end_ns={end_ns:.3f}",
            flush=True,
        )
        if end_ns < args.minimum_end_ns:
            failures.append(
                f"{workflow_id}: only {end_ns:.3f} ns in refreshed report"
            )
    if failures:
        print("Failures:", flush=True)
        for failure in failures:
            print(f"  {failure}", flush=True)
        return 1
    print("All requested MD analyses were recomputed and verified.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
