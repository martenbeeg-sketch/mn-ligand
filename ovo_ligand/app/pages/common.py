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
    root = Path(os.getenv("OVO_LIGAND_INPUT_DIR", "/tmp/ovo-ligand-inputs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_root() -> Path:
    # Keep run storage consistent with MD pages/jobs:
    # default to project-local .ovo-home/workdir/runs.
    default_root = Path(__file__).resolve().parents[3] / ".ovo-home" / "workdir" / "runs"
    root = Path(os.getenv("OVO_LIGAND_RUN_DIR", str(default_root)))
    root.mkdir(parents=True, exist_ok=True)
    return root


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
    """Dispatch oldest queued GPU job if lock is free.

    Returns a small status dict when a queued job is run, otherwise None.
    """
    if _gpu_lock_path().exists():
        return None

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
        return None
    candidates.sort(key=lambda x: x[0])
    _, run_dir, payload = candidates[0]
    run_id = str(payload.get("run_id") or run_dir.name)
    workflow = str(payload.get("workflow") or "unknown")
    cmd = payload.get("queued_command")
    lock_ok, _ = acquire_gpu_job_lock(workflow, run_id)
    if not lock_ok:
        return None

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
        return {"run_id": run_id, "workflow": workflow, "returncode": int(result.returncode)}
    finally:
        release_gpu_job_lock(run_id)


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
    copy_lines = [
        f"cp {shlex.quote(container_path)} /output/{shlex.quote(param_name + Path(container_path).suffix)}"
        for param_name, container_path in container_inputs.items()
    ]
    return "\n".join(
        [
            "set -euo pipefail",
            "mkdir -p /output",
            *copy_lines,
            "cat > /output/summary.json <<'JSON'",
            json.dumps(metadata, indent=2),
            "JSON",
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
        # ovoex-boltz2 has an ENTRYPOINT that runs `boltz "$@"`, so pass only
        # Boltz CLI arguments here.
        return command + [
            image,
            "predict",
            input_yaml,
            "--out_dir",
            "/output",
            "--accelerator",
            str(params.get("accelerator", "gpu")),
        ]

    if workflow_key in {"abfe", "rbfe"}:
        payload = _build_openfe_payload(workflow_key, params, output_dir.name)
        (output_dir / "openfe_input.json").write_text(json.dumps(payload, indent=2))
        entry = (
            "ABFE_OUTPUT_DIR=/output RBFE_OUTPUT_DIR=/output "
            "python -m ovo_ligand.ligandx.services.abfe.run_abfe_job --input /output/openfe_input.json --output /output/result.json"
            if workflow_key == "abfe"
            else "ABFE_OUTPUT_DIR=/output RBFE_OUTPUT_DIR=/output "
            "python -m ovo_ligand.ligandx.services.rbfe.run_rbfe_job --input /output/openfe_input.json --output /output/result.json"
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

    script = _build_smoke_script(workflow_key, params, container_inputs)
    return command + [image, "/bin/bash", "-lc", script]


def _run_docker_workflow(workflow_key: str, workflow: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    run_id = str(uuid4())
    generated_job_code = run_id.replace("-", "")[:3].upper()
    output_dir = _run_root() / workflow_key / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
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
