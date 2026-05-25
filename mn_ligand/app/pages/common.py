from __future__ import annotations

import os
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st


WORKFLOWS: dict[str, dict[str, Any]] = {
    "structure-preparation": {
        "title": "Ligand structure preparation",
        "container_param": "structure_container",
        "gpu": False,
        "defaults": {
            "structure_container": "ovolig-structure:latest",
        },
        "files": {
            "structure_file": ["pdb", "cif", "mmcif", "sdf", "mol2"],
        },
        "params": {
            "mode": "protein",
        },
    },
    "docking": {
        "title": "Ligand docking",
        "container_param": "docking_container",
        "gpu": False,
        "defaults": {
            "docking_container": "ovolig-docking:latest",
        },
        "files": {
            "protein_pdb": ["pdb"],
            "ligands_sdf": ["sdf", "mol", "mol2"],
        },
        "params": {
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 0.0,
            "size_x": 20.0,
            "size_y": 20.0,
            "size_z": 20.0,
            "exhaustiveness": 8,
        },
    },
    "batch-docking": {
        "title": "Batch ligand docking",
        "container_param": "docking_container",
        "gpu": False,
        "defaults": {
            "docking_container": "ovolig-docking:latest",
        },
        "files": {
            "protein_pdb": ["pdb"],
            "ligands_sdf": ["sdf"],
        },
        "params": {
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 0.0,
            "size_x": 20.0,
            "size_y": 20.0,
            "size_z": 20.0,
            "exhaustiveness": 8,
        },
    },
    "md": {
        "title": "Ligand molecular dynamics",
        "container_param": "md_container",
        "gpu": True,
        "defaults": {
            "md_container": "ovolig-md-cu128:latest",
        },
        "files": {
            "complex_pdb": ["pdb"],
            "ligand_sdf": ["sdf", "mol"],
        },
        "params": {
            "steps": 5000,
            "temperature": 300.0,
        },
    },
    "admet": {
        "title": "Ligand ADMET prediction",
        "container_param": "admet_container",
        "gpu": False,
        "defaults": {
            "admet_container": "ovolig-admet:latest",
        },
        "files": {
            "smiles_file": ["smi", "txt", "csv"],
        },
        "params": {},
    },
    "boltz2": {
        "title": "Ligand Boltz-2 prediction",
        "container_param": "boltz2_container",
        "gpu": True,
        "defaults": {
            "boltz2_container": "ovoex-boltz2:latest",
        },
        "files": {
            "input_yaml": ["yaml", "yml"],
        },
        "params": {
            "accelerator": "gpu",
        },
    },
    "qc": {
        "title": "Ligand quantum chemistry",
        "container_param": "qc_container",
        "gpu": False,
        "defaults": {
            "qc_container": "ovolig-qc:latest",
        },
        "files": {
            "molecule_file": ["sdf", "mol", "xyz"],
        },
        "params": {
            "method": "B3LYP",
            "basis": "def2-SVP",
            "orca_path": "/opt/orca",
        },
    },
    "abfe": {
        "title": "Ligand ABFE",
        "container_param": "abfe_container",
        "gpu": True,
        "defaults": {
            "abfe_container": "ovolig-md-cu128:latest",
        },
        "files": {
            "protein_pdb": ["pdb"],
            "ligand_sdf": ["sdf", "mol"],
        },
        "params": {
            "protocol": "quick-test",
        },
    },
    "rbfe": {
        "title": "Ligand RBFE",
        "container_param": "rbfe_container",
        "gpu": True,
        "defaults": {
            "rbfe_container": "ovolig-md-cu128:latest",
        },
        "files": {
            "protein_pdb": ["pdb"],
            "ligands_sdf": ["sdf"],
        },
        "params": {
            "mapper": "lomap",
            "protocol": "quick-test",
        },
    },
}


def _input_root() -> Path:
    root = Path(os.getenv("MN_LIGAND_INPUT_DIR", "/tmp/mn-ligand-inputs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_root() -> Path:
    # Keep run storage consistent with MD pages/jobs:
    # default to project-local mn-ligand-workdir/workdir/runs.
    default_root = Path(__file__).resolve().parents[3] / "mn-ligand-workdir" / "workdir" / "runs"
    root = Path(os.getenv("MN_LIGAND_RUN_DIR", str(default_root)))
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_run_artifact_path(path_value: str | Path | None, *, must_exist: bool = False) -> Path | None:
    """Resolve artifact paths across legacy and current run-root layouts.

    Supports old absolute paths from:
    .../ovo-ligand/.ovo-home/workdir/runs/<...>
    by remapping them under the current `_run_root()`.
    """
    if path_value is None:
        return None
    raw = str(path_value).strip()
    if not raw:
        return None

    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate

    marker = "/.ovo-home/workdir/runs/"
    mapped: Path | None = None
    normalized = raw.replace("\\", "/")
    if marker in normalized:
        rel = normalized.split(marker, 1)[1].lstrip("/")
        mapped = _run_root() / rel
    elif not candidate.is_absolute():
        mapped = _run_root() / candidate

    if mapped is None:
        return None if must_exist else candidate
    if must_exist and not mapped.exists():
        return None
    return mapped


def _gpu_lock_path() -> Path:
    return _run_root() / ".gpu_job.lock"


def acquire_gpu_job_lock(workflow: str, run_id: str) -> tuple[bool, dict]:
    payload = {"workflow": str(workflow), "run_id": str(run_id)}
    lock_path = _gpu_lock_path()
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(payload, indent=2))
        return True, payload
    except FileExistsError:
        try:
            existing = json.loads(lock_path.read_text())
        except Exception:
            existing = {"workflow": "unknown", "run_id": "unknown"}
        return False, existing


def release_gpu_job_lock(run_id: str) -> None:
    lock_path = _gpu_lock_path()
    if not lock_path.exists():
        return
    try:
        data = json.loads(lock_path.read_text())
        if str(data.get("run_id", "")) != str(run_id):
            return
    except Exception:
        return
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def queue_gpu_job(run_dir: Path, workflow: str, run_id: str, command: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "metadata.json"
    current = {}
    if metadata_path.exists():
        try:
            current = json.loads(metadata_path.read_text())
        except Exception:
            current = {}
    current.update(
        {
            "run_id": run_id,
            "workflow": workflow,
            "status": "queued",
            "gpu_queued": True,
            "queued_command": command,
            "updated_at": current.get("updated_at") or "",
        }
    )
    metadata_path.write_text(json.dumps(current, indent=2))


def try_dispatch_next_queued_gpu_job() -> dict[str, Any] | None:
    """Dispatch queued GPU jobs sequentially while lock is free.

    Returns a summary dict when at least one queued job was run, otherwise None.
    """
    if _gpu_lock_path().exists():
        return None

    dispatched: list[dict[str, Any]] = []
    while True:
        candidates: list[tuple[float, Path, dict[str, Any]]] = []
        for meta in _run_root().glob("**/metadata.json"):
            try:
                payload = json.loads(meta.read_text())
            except Exception:
                continue
            if str(payload.get("status")) != "queued":
                continue
            cmd = payload.get("queued_command")
            if not isinstance(cmd, list) or not cmd:
                continue
            run_dir = meta.parent
            try:
                t = run_dir.stat().st_mtime
            except Exception:
                t = 0.0
            candidates.append((t, run_dir, payload))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        _, run_dir, payload = candidates[0]
        run_id = str(payload.get("run_id") or run_dir.name)
        workflow = str(payload.get("workflow") or "unknown")
        cmd = payload.get("queued_command")
        lock_ok, _ = acquire_gpu_job_lock(workflow, run_id)
        if not lock_ok:
            break

        metadata_path = run_dir / "metadata.json"
        payload["status"] = "running"
        metadata_path.write_text(json.dumps(payload, indent=2))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            payload["status"] = "completed" if result.returncode == 0 else "failed"
            payload["returncode"] = int(result.returncode)
            payload["stdout_tail"] = (result.stdout or "")[-8000:]
            payload["stderr_tail"] = (result.stderr or "")[-8000:]
            metadata_path.write_text(json.dumps(payload, indent=2))
            dispatched.append(
                {"run_id": run_id, "workflow": workflow, "returncode": int(result.returncode)}
            )
        finally:
            release_gpu_job_lock(run_id)

    if not dispatched:
        return None
    return {"count": len(dispatched), "last": dispatched[-1], "runs": dispatched}


def reconcile_run_metadata_status(run_dir: Path) -> bool:
    """Finalize stale run metadata from result.json when possible.

    Returns True when metadata was updated.
    """
    meta_path = run_dir / "metadata.json"
    result_path = run_dir / "result.json"
    if not meta_path.exists() or not result_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        status = str(meta.get("status") or "")
        if status not in {"running", "queued"}:
            return False
        result = json.loads(result_path.read_text())
        final_status = "completed" if bool(result.get("success")) else "failed"
        meta["status"] = final_status
        now_iso = datetime.now(timezone.utc).isoformat()
        meta["completed_at"] = meta.get("completed_at") or now_iso
        meta["updated_at"] = now_iso
        meta_path.write_text(json.dumps(meta, indent=2))
        return True
    except Exception:
        return False


def _save_upload(workflow_key: str, param_name: str, uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    safe_name = Path(uploaded_file.name).name
    target = _input_root() / workflow_key / param_name / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(uploaded_file.getvalue())
    return str(target)


def _render_scalar_input(key: str, value: Any) -> Any:
    if isinstance(value, bool):
        return st.checkbox(key.replace("_", " ").title(), value=value)
    if isinstance(value, int):
        return st.number_input(key.replace("_", " ").title(), value=value, step=1)
    if isinstance(value, float):
        return st.number_input(key.replace("_", " ").title(), value=value)
    return st.text_input(key.replace("_", " ").title(), value=str(value))


def _build_smoke_script(workflow_key: str, params: dict[str, Any], container_inputs: dict[str, str]) -> str:
    metadata = {
        "workflow": workflow_key,
        "inputs": container_inputs,
        "params": {
            key: value
            for key, value in params.items()
            if key not in container_inputs and not key.endswith("_container")
        },
        "status": "container wiring ok",
    }
    result_payload = {
        "success": True,
        "workflow": workflow_key,
        "status": "completed",
        "message": "smoke workflow completed",
        "inputs": container_inputs,
    }
    copy_lines = [
        f"cp {shlex.quote(container_path)} /output/{shlex.quote(param_name + Path(container_path).suffix)}"
        for param_name, container_path in container_inputs.items()
    ]
    return "\n".join(
        [
            "set -eu",
            "mkdir -p /output",
            *copy_lines,
            "cat > /output/summary.json <<'JSON'",
            json.dumps(metadata, indent=2),
            "JSON",
            "cat > /output/result.json <<'JSON'",
            json.dumps(result_payload, indent=2),
            "JSON",
            "chmod -R a+rwX /output >/dev/null 2>&1 || true",
        ]
    )


def _build_admet_script(smiles_file_path: str) -> str:
    return "\n".join(
        [
            "set -eu",
            "mkdir -p /tmp/admet /output",
            "/opt/conda/bin/python - <<'PY'",
            "import csv, json",
            "from pathlib import Path",
            "",
            f"smiles_src = Path({json.dumps(smiles_file_path)})",
            "csv_in = Path('/tmp/admet/in.csv')",
            "rows = []",
            "for raw in smiles_src.read_text().splitlines():",
            "    line = raw.strip()",
            "    if not line or line.startswith('#'):",
            "        continue",
            "    if ',' in line:",
            "        parts = [p.strip() for p in line.split(',', 1)]",
            "        if len(parts) == 2 and parts[1]:",
            "            rows.append({'molecule_name': parts[0] or 'ligand', 'smiles': parts[1]})",
            "            continue",
            "    toks = line.split()",
            "    if len(toks) >= 2:",
            "        rows.append({'molecule_name': toks[0], 'smiles': toks[-1]})",
            "    else:",
            "        rows.append({'molecule_name': 'ligand', 'smiles': toks[0]})",
            "if not rows:",
            "    raise SystemExit('No SMILES entries found in smiles_file')",
            "with csv_in.open('w', newline='') as f:",
            "    w = csv.DictWriter(f, fieldnames=['smiles', 'molecule_name'])",
            "    w.writeheader()",
            "    for r in rows:",
            "        w.writerow({'smiles': r['smiles'], 'molecule_name': r['molecule_name']})",
            "PY",
            "/opt/conda/bin/admet_predict --data_path /tmp/admet/in.csv --save_path /output/admet_predictions.csv >/output/admet_cli.log 2>&1",
            "/opt/conda/bin/python - <<'PY'",
            "import csv, json",
            "from pathlib import Path",
            "",
            "out_csv = Path('/output/admet_predictions.csv')",
            "rows = list(csv.DictReader(out_csv.open()))",
            "if not rows:",
            "    raise SystemExit('No ADMET prediction rows produced')",
            "r = rows[0]",
            "",
            "def g(key, default=''):",
            "    v = r.get(key)",
            "    return default if v is None else str(v)",
            "",
            "result = {",
            "  'success': True,",
            "  'workflow': 'admet',",
            "  'status': 'completed',",
            "  'method': 'admet_predict',",
            "  'n_molecules': len(rows),",
            "  'smiles': g('smiles'),",
            "  'Physicochemical': {",
            "    'Molecular Weight': g('molecular_weight'),",
            "    'LogP': g('logP'),",
            "    'Hydrogen Bond Acceptors': g('hydrogen_bond_acceptors'),",
            "    'Hydrogen Bond Donors': g('hydrogen_bond_donors'),",
            "    'Lipinski Rule of 5 Violations': g('Lipinski'),",
            "    'QED': g('QED'),",
            "    'Stereo Centers': g('stereo_centers'),",
            "    'TPSA': g('tpsa')",
            "  },",
            "  'Absorption': {",
            "    'Human Intestinal Absorption': g('HIA_Hou') + ' (Prob.)',",
            "    'Oral Bioavailability': g('Bioavailability_Ma') + ' (Prob.)',",
            "    'Aqueous Solubility': g('Solubility_AqSolDB') + ' (logS)',",
            "    'Lipophilicity': g('Lipophilicity_AstraZeneca') + ' (logD7.4)',",
            "    'Cell Effective Permeability': g('Caco2_Wang') + ' (logPapp)',",
            "    'P-glycoprotein Inhibition': g('Pgp_Broccatelli') + ' (Prob.)'",
            "  },",
            "  'Distribution': {",
            "    'Blood-Brain Barrier Penetration': g('BBB_Martins') + ' (Prob.)',",
            "    'VDss': g('VDss_Lombardo')",
            "  },",
            "  'Metabolism': {",
            "    'CYP1A2 Inhibition': g('CYP1A2_Veith') + ' (Prob.)',",
            "    'CYP2C19 Inhibition': g('CYP2C19_Veith') + ' (Prob.)',",
            "    'CYP2C9 Inhibition': g('CYP2C9_Veith') + ' (Prob.)',",
            "    'CYP2D6 Inhibition': g('CYP2D6_Veith') + ' (Prob.)',",
            "    'CYP3A4 Inhibition': g('CYP3A4_Veith') + ' (Prob.)',",
            "    'CYP2C9 Substrate': g('CYP2C9_Substrate_CarbonMangels') + ' (Prob.)',",
            "    'CYP2D6 Substrate': g('CYP2D6_Substrate_CarbonMangels') + ' (Prob.)',",
            "    'CYP3A4 Substrate': g('CYP3A4_Substrate_CarbonMangels') + ' (Prob.)'",
            "  },",
            "  'Toxicity': {",
            "    'hERG Blocking': g('hERG') + ' (Prob.)',",
            "    'Clinical Toxicity': g('ClinTox') + ' (Prob.)',",
            "    'Mutagenicity (AMES)': g('AMES') + ' (Prob.)',",
            "    'Drug-Induced Liver Injury': g('DILI') + ' (Prob.)',",
            "    'Carcinogenicity': g('Carcinogens_Lagunin') + ' (Prob.)',",
            "    'Acute Toxicity LD50': g('LD50_Zhu') + ' (log(mol/kg))'",
            "  },",
            "  'raw_prediction_row': r",
            "}",
            "Path('/output/result.json').write_text(json.dumps(result, indent=2))",
            "PY",
            "chmod -R a+rwX /output || true",
        ]
    )


def _build_qc_script(molecule_file_path: str, method: str, basis: str) -> str:
    return "\n".join(
        [
            "set -eu",
            "mkdir -p /output",
            f"cp {shlex.quote(molecule_file_path)} /output/input_molecule{Path(molecule_file_path).suffix}",
            "/opt/conda/bin/python - <<'PY'",
            "import json",
            "from pathlib import Path",
            "",
            f"molecule_file = {json.dumps(molecule_file_path)}",
            f"method = {json.dumps(method)}",
            f"basis = {json.dumps(basis)}",
            "result = {",
            "  'success': True,",
            "  'workflow': 'qc',",
            "  'status': 'completed',",
            "  'mode': 'smoke',",
            "  'input_file': molecule_file,",
            "  'method': method,",
            "  'basis_set': basis,",
            "  'energy_hartree': -123.456789,",
            "  'homo_eV': -5.12,",
            "  'lumo_eV': -0.88,",
            "  'gap_eV': 4.24,",
            "  'dipole_debye': 2.31,",
            "  'message': 'QC smoke test completed (mock values).'",
            "}",
            "Path('/output/result.json').write_text(json.dumps(result, indent=2))",
            "Path('/output/qc_summary.json').write_text(json.dumps({",
            "  'job_type': 'smoke',",
            "  'method': method,",
            "  'basis_set': basis,",
            "  'completed': True",
            "}, indent=2))",
            "PY",
            "chmod -R a+rwX /output >/dev/null 2>&1 || true",
        ]
    )

def _read_text_file(path: str) -> str:
    try:
        return Path(path).read_text()
    except Exception:
        return ""


def _build_openfe_payload(workflow_key: str, params: dict[str, Any], run_id: str) -> dict[str, Any]:
    if workflow_key == "abfe":
        return {
            "job_id": run_id,
            "protein_pdb_data": _read_text_file(str(params.get("protein_pdb") or "")),
            "ligand_sdf_data": _read_text_file(str(params.get("ligand_sdf") or "")),
            "ligand_id": "ligand",
            "protein_id": "protein",
            "protocol_settings": dict(params.get("protocol_settings") or {}),
        }
    if workflow_key == "rbfe":
        sdf_text = _read_text_file(str(params.get("ligands_sdf") or ""))
        blocks = [b.strip() for b in sdf_text.split("$$$$") if b.strip()]
        ligands: list[dict[str, Any]] = []
        for idx, block in enumerate(blocks, start=1):
            ligands.append({"id": f"ligand_{idx}", "data": block + "\n$$$$\n", "format": "sdf"})
        return {
            "job_id": run_id,
            "protein_pdb_data": _read_text_file(str(params.get("protein_pdb") or "")),
            "ligands": ligands,
            "network_topology": str(params.get("network_topology") or "mst"),
            "central_ligand": params.get("central_ligand") or None,
            "atom_mapper": str(params.get("atom_mapper") or "kartograf"),
            "atom_map_hydrogens": bool(params.get("atom_map_hydrogens", True)),
            "lomap_max3d": float(params.get("lomap_max3d", 1.0)),
            "protocol_settings": dict(params.get("protocol_settings") or {}),
            "protein_id": "protein",
        }
    return {}


def _build_docker_command(
    workflow_key: str,
    workflow: dict[str, Any],
    params: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    image = params[workflow["container_param"]]
    command = ["docker", "run", "--rm", "-v", f"{output_dir}:/output"]
    # Keep output file ownership on the host user for local app-managed runs.
    if workflow_key in {"admet", "qc"}:
        command += ["--user", f"{os.getuid()}:{os.getgid()}"]

    if workflow.get("gpu"):
        command += ["--gpus", "all"]

    container_inputs: dict[str, str] = {}
    for param_name in workflow["files"]:
        host_path = params.get(param_name)
        if not host_path:
            continue
        suffix = Path(host_path).suffix
        container_path = f"/input/{param_name}{suffix}"
        command += ["-v", f"{host_path}:{container_path}:ro"]
        container_inputs[param_name] = container_path

    if workflow_key == "boltz2":
        input_yaml = container_inputs.get("input_yaml")
        if not input_yaml:
            raise ValueError("Boltz2 requires --input_yaml")

        cache_dir = str(params.get("boltz_cache_dir") or "").strip()
        msa_repo_dir = str(params.get("boltz_msa_repository_dir") or "").strip()
        if cache_dir:
            command += ["-v", f"{cache_dir}:/cache"]
            command += ["-e", "BOLTZ_CACHE=/cache"]
        if msa_repo_dir:
            command += ["-v", f"{msa_repo_dir}:/msa_repository"]

        boltz_cmd = [
            image,
            "predict",
            input_yaml,
            "--out_dir",
            "/output",
            "--sampling_steps",
            str(int(params.get("sampling_steps", 200))),
            "--recycling_steps",
            str(int(params.get("recycling_steps", 3))),
            "--diffusion_samples",
            str(int(params.get("diffusion_samples", 1))),
            "--sampling_steps_affinity",
            str(int(params.get("sampling_steps_affinity", 200))),
            "--diffusion_samples_affinity",
            str(int(params.get("diffusion_samples_affinity", 5))),
            "--accelerator",
            str(params.get("accelerator", "gpu")),
            "--override",
        ]
        if bool(params.get("use_msa_server", True)):
            boltz_cmd.append("--use_msa_server")
        if bool(params.get("use_potentials", True)):
            boltz_cmd.append("--use_potentials")
        if bool(params.get("affinity_mw_correction", False)):
            boltz_cmd.append("--affinity_mw_correction")
        return command + boltz_cmd

    if workflow_key in {"abfe", "rbfe"}:
        payload = _build_openfe_payload(workflow_key, params, output_dir.name)
        (output_dir / "openfe_input.json").write_text(json.dumps(payload, indent=2))
        entry = (
            "ABFE_OUTPUT_DIR=/output RBFE_OUTPUT_DIR=/output "
            "python -m mn_ligand.ligandx.services.abfe.run_abfe_job --input /output/openfe_input.json --output /output/result.json"
            if workflow_key == "abfe"
            else "ABFE_OUTPUT_DIR=/output RBFE_OUTPUT_DIR=/output "
            "python -m mn_ligand.ligandx.services.rbfe.run_rbfe_job --input /output/openfe_input.json --output /output/result.json"
        )
        return command + [
            "-v",
            f"{Path(__file__).resolve().parents[3]}:/work:ro",
            "-w",
            "/work",
            image,
            "/bin/bash",
            "-lc",
            entry,
        ]

    if workflow_key == "admet":
        smiles_file = container_inputs.get("smiles_file")
        if not smiles_file:
            raise ValueError("ADMET requires smiles_file input")
        script = _build_admet_script(smiles_file)
    elif workflow_key == "qc":
        molecule_file = container_inputs.get("molecule_file")
        if not molecule_file:
            raise ValueError("QC requires molecule_file input")
        script = _build_qc_script(
            molecule_file,
            str(params.get("method") or "B3LYP"),
            str(params.get("basis") or "def2-SVP"),
        )
    else:
        script = _build_smoke_script(workflow_key, params, container_inputs)
    return command + [image, "/bin/sh", "-lc", script]


def _run_docker_workflow(workflow_key: str, workflow: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    run_id = str(uuid4())
    generated_job_code = run_id.replace("-", "")[:3].upper()
    output_dir = _run_root() / workflow_key / run_id
    if workflow_key == "boltz2":
        extra_meta = params.get("metadata")
        if isinstance(extra_meta, dict):
            structure_run_id = str(extra_meta.get("structure_run_id") or "").strip()
            if structure_run_id:
                output_dir = _run_root() / "structure-jobs" / structure_run_id / "boltz2"
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        # Allow container processes with arbitrary UID/GID to write outputs.
        output_dir.chmod(0o777)
    except Exception:
        pass
    metadata_path = output_dir / "metadata.json"
    now_iso = datetime.now(timezone.utc).isoformat()
    metadata_payload = {
        "run_id": run_id,
        "workflow": workflow_key.upper(),
        "workflow_key": workflow_key,
        "job_code": generated_job_code,
        "openfe_job_code": generated_job_code,
        "status": "running",
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    if workflow_key == "qc":
        metadata_payload["qc_method"] = str(params.get("method") or "")
        metadata_payload["qc_basis"] = str(params.get("basis") or "")
    # Preserve any structured metadata passed by the page layer.
    extra_meta = params.get("metadata")
    if isinstance(extra_meta, dict):
        # Preserve source fields without overwriting generated OpenFE run code.
        source_code = extra_meta.get("job_code")
        if source_code:
            metadata_payload["source_job_code"] = str(source_code)
        extra_meta = {k: v for k, v in extra_meta.items() if k != "job_code"}
        metadata_payload.update(extra_meta)
    try:
        metadata_path.write_text(json.dumps(metadata_payload, indent=2))
    except Exception:
        pass

    command = _build_docker_command(workflow_key, workflow, params, output_dir)
    lock_acquired = False
    if workflow.get("gpu"):
        lock_acquired, lock_info = acquire_gpu_job_lock(workflow_key, run_id)
        if not lock_acquired:
            # Queue like MD workflows instead of failing immediately.
            queue_gpu_job(output_dir, workflow_key, run_id, command)
            try:
                queued_meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            except Exception:
                queued_meta = {}
            queued_meta.update(
                {
                    "run_id": run_id,
                    "workflow": workflow_key.upper(),
                    "workflow_key": workflow_key,
                    "status": "queued",
                    "queued_at": now_iso,
                    "updated_at": now_iso,
                }
            )
            if isinstance(extra_meta, dict):
                queued_meta.update(extra_meta)
            try:
                metadata_path.write_text(json.dumps(queued_meta, indent=2))
            except Exception:
                pass
            return {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "command": command,
                "returncode": 0,
                "queued": True,
                "stdout": "",
                "stderr": (
                    "GPU busy; job queued behind active run "
                    f"{lock_info.get('workflow')} ({lock_info.get('run_id')})."
                ),
            }
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        try:
            (output_dir / "stdout.log").write_text(result.stdout or "")
            (output_dir / "stderr.log").write_text(result.stderr or "")
        except Exception:
            pass
        try:
            metadata_payload["status"] = "completed" if result.returncode == 0 else "failed"
            metadata_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            metadata_payload["completed_at"] = datetime.now(timezone.utc).isoformat()
            metadata_path.write_text(json.dumps(metadata_payload, indent=2))
        except Exception:
            pass
        return {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    finally:
        if lock_acquired:
            release_gpu_job_lock(run_id)


def render_workflow_page(
    workflow_key: str,
    *,
    show_title: bool = True,
    show_container_input: bool = True,
    show_command_preview: bool = True,
    show_command_in_result: bool = True,
    title_override: str | None = None,
    intro_text: str | None = None,
    run_button_label: str = "Run Docker workflow",
) -> None:
    workflow = WORKFLOWS[workflow_key]
    if show_title:
        st.title(title_override or workflow["title"])
        st.caption("Direct Docker scaffold")

    if intro_text is None:
        st.write(
            "This page runs a Docker container directly. Most workflows currently run a smoke-test wrapper; replace that wrapper with the ported Ligand-X tool command as each tool is migrated."
        )
    elif intro_text:
        st.write(intro_text)

    params: dict[str, Any] = {}

    container_param = workflow["container_param"]
    if show_container_input:
        st.subheader("Container")
        params[container_param] = st.text_input(
            "Docker image",
            value=workflow["defaults"][container_param],
            help="Docker image tag used by docker run.",
        )
    else:
        params[container_param] = workflow["defaults"][container_param]

    st.subheader("Inputs")
    for param_name, extensions in workflow["files"].items():
        uploaded_file = st.file_uploader(
            param_name.replace("_", " ").title(),
            type=extensions,
            key=f"{workflow_key}_{param_name}",
        )
        saved_path = _save_upload(workflow_key, param_name, uploaded_file)
        if saved_path:
            params[param_name] = saved_path
            st.code(saved_path)

    if workflow["params"]:
        st.subheader("Parameters")
        for param_name, default in workflow["params"].items():
            params[param_name] = _render_scalar_input(param_name, default)

    if show_command_preview:
        with st.expander("Docker command preview", expanded=True):
            try:
                preview_dir = _run_root() / workflow_key / "preview"
                preview_command = _build_docker_command(workflow_key, workflow, params, preview_dir)
                st.code(shlex.join(preview_command))
            except Exception as exc:
                st.info(str(exc))

    if st.button(run_button_label, type="primary"):
        try:
            run = _run_docker_workflow(workflow_key, workflow, params)
            if run["returncode"] == 0:
                st.success(f"Docker run completed: {run['run_id']}")
            else:
                st.error(f"Docker run failed with exit code {run['returncode']}")
            st.write("Output directory")
            st.code(run["output_dir"])
            if show_command_in_result:
                with st.expander("Command"):
                    st.code(shlex.join(run["command"]))
            if run["stdout"]:
                with st.expander("stdout"):
                    st.code(run["stdout"])
            if run["stderr"]:
                with st.expander("stderr"):
                    st.code(run["stderr"])
        except Exception as exc:
            st.error(f"Docker run failed: {exc}")
