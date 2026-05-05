from __future__ import annotations

import os
import json
import shlex
import subprocess
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
            "abfe_container": "ovolig-abfe-cu128:latest",
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
            "rbfe_container": "ovolig-rbfe-cu128:latest",
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
    root = Path(os.getenv("OVO_LIGAND_RUN_DIR", "/tmp/ovo-ligand-runs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


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

    script = _build_smoke_script(workflow_key, params, container_inputs)
    return command + [image, "/bin/bash", "-lc", script]


def _run_docker_workflow(workflow_key: str, workflow: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    run_id = str(uuid4())
    output_dir = _run_root() / workflow_key / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    command = _build_docker_command(workflow_key, workflow, params, output_dir)

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def render_workflow_page(workflow_key: str) -> None:
    workflow = WORKFLOWS[workflow_key]
    st.title(workflow["title"])
    st.caption("Direct Docker scaffold")

    st.write(
        "This page runs a Docker container directly. Most workflows currently run a smoke-test wrapper; replace that wrapper with the ported Ligand-X tool command as each tool is migrated."
    )

    params: dict[str, Any] = {}

    st.subheader("Container")
    container_param = workflow["container_param"]
    params[container_param] = st.text_input(
        "Docker image",
        value=workflow["defaults"][container_param],
        help="Docker image tag used by docker run.",
    )

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

    with st.expander("Docker command preview", expanded=True):
        try:
            preview_dir = _run_root() / workflow_key / "preview"
            preview_command = _build_docker_command(workflow_key, workflow, params, preview_dir)
            st.code(shlex.join(preview_command))
        except Exception as exc:
            st.info(str(exc))

    if st.button("Run Docker workflow", type="primary"):
        try:
            run = _run_docker_workflow(workflow_key, workflow, params)
            if run["returncode"] == 0:
                st.success(f"Docker run completed: {run['run_id']}")
            else:
                st.error(f"Docker run failed with exit code {run['returncode']}")
            st.write("Output directory")
            st.code(run["output_dir"])
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
