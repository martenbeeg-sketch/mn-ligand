from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JOB_SCHEMA_VERSION, JobRecord, short_job_code
from mn_ligand.runtime import resolve_run_dir, runs_root


WORKFLOW_SCHEMA_VERSION = 1
ACTIVE_STATES = frozenset({"queued", "preparing", "running", "paused"})
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "blocked"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class WorkflowChildRef:
    run_id: str
    task_group: str
    step_id: str
    depends_on: tuple[str, ...] = ()
    required: bool = True
    attached_at: str = field(default_factory=_utc_now_iso)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkflowChildRef:
        return cls(
            run_id=str(payload.get("run_id") or ""),
            task_group=str(payload.get("task_group") or ""),
            step_id=str(payload.get("step_id") or ""),
            depends_on=tuple(str(item) for item in payload.get("depends_on") or ()),
            required=bool(payload.get("required", True)),
            attached_at=str(payload.get("attached_at") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_group": self.task_group,
            "step_id": self.step_id,
            "depends_on": list(self.depends_on),
            "required": self.required,
            "attached_at": self.attached_at,
        }


@dataclass(frozen=True)
class WorkflowInput:
    source_task_group: str
    artifact: ArtifactRef

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkflowInput:
        return cls(
            source_task_group=str(payload.get("source_task_group") or ""),
            artifact=ArtifactRef.from_dict(dict(payload.get("artifact") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"source_task_group": self.source_task_group, "artifact": self.artifact.to_dict()}


@dataclass(frozen=True)
class WorkflowRecord:
    workflow_id: str
    workflow_type: str
    name: str
    status: str
    run_dir: Path
    children: tuple[WorkflowChildRef, ...] = ()
    inputs: tuple[WorkflowInput, ...] = ()
    expected_steps: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    schema_version: int = WORKFLOW_SCHEMA_VERSION

    @classmethod
    def load(cls, workflow_id: str) -> WorkflowRecord:
        run_dir = resolve_run_dir("workflows", workflow_id)
        if run_dir is None:
            raise FileNotFoundError(f"Workflow not found: {workflow_id}")
        payload = _read_json(run_dir / "workflow.json")
        if int(payload.get("schema_version") or 0) != WORKFLOW_SCHEMA_VERSION:
            raise ValueError(f"Unsupported workflow schema version: {payload.get('schema_version')}")
        return cls(
            workflow_id=str(payload.get("workflow_id") or workflow_id),
            workflow_type=str(payload.get("workflow_type") or ""),
            name=str(payload.get("name") or ""),
            status=str(payload.get("status") or "unknown"),
            run_dir=run_dir,
            children=tuple(WorkflowChildRef.from_dict(item) for item in payload.get("children") or ()),
            inputs=tuple(WorkflowInput.from_dict(item) for item in payload.get("inputs") or ()),
            expected_steps=tuple(str(item) for item in payload.get("expected_steps") or ()),
            parameters=dict(payload.get("parameters") or {}),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            completed_at=str(payload.get("completed_at") or ""),
            schema_version=int(payload.get("schema_version") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "workflow",
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "parameters": self.parameters,
            "inputs": [item.to_dict() for item in self.inputs],
            "expected_steps": list(self.expected_steps),
            "children": [item.to_dict() for item in self.children],
        }


def create_workflow(
    workflow_type: str,
    *,
    name: str,
    parameters: dict[str, Any] | None = None,
    inputs: Iterable[WorkflowInput] = (),
    expected_steps: Iterable[str] = (),
) -> WorkflowRecord:
    workflow_id = str(uuid4())
    run_dir = runs_root() / "workflows" / workflow_id
    run_dir.mkdir(parents=True, exist_ok=False)
    now = _utc_now_iso()
    workflow = WorkflowRecord(
        workflow_id=workflow_id,
        workflow_type=workflow_type,
        name=name,
        status="queued",
        run_dir=run_dir,
        inputs=tuple(inputs),
        expected_steps=tuple(dict.fromkeys(str(item) for item in expected_steps if str(item))),
        parameters=dict(parameters or {}),
        created_at=now,
        updated_at=now,
    )
    _write_workflow(workflow)
    write_artifact_manifest(run_dir, [])
    return workflow


def add_workflow_input(workflow_id: str, source_task_group: str, artifact: ArtifactRef) -> WorkflowRecord:
    workflow = WorkflowRecord.load(workflow_id)
    key = (source_task_group, artifact.run_id, artifact.artifact_id)
    existing = {(item.source_task_group, item.artifact.run_id, item.artifact.artifact_id) for item in workflow.inputs}
    if key in existing:
        return workflow
    workflow = replace(
        workflow,
        inputs=(*workflow.inputs, WorkflowInput(source_task_group=source_task_group, artifact=artifact)),
        updated_at=_utc_now_iso(),
    )
    _write_workflow(workflow)
    return workflow


def attach_workflow_child(
    workflow_id: str,
    child: JobRecord,
    *,
    step_id: str,
    depends_on: Iterable[str] = (),
    required: bool = True,
    replace_step: bool = False,
    update_child_metadata: bool = True,
) -> WorkflowRecord:
    workflow = WorkflowRecord.load(workflow_id)
    child_ref = WorkflowChildRef(
        run_id=child.run_id,
        task_group=child.task_group,
        step_id=step_id,
        depends_on=tuple(dict.fromkeys(str(item) for item in depends_on if str(item))),
        required=required,
    )
    children: tuple[WorkflowChildRef, ...] = tuple(
        item for item in workflow.children if item.run_id != child.run_id
    )
    if replace_step:
        replaced: list[WorkflowChildRef] = []
        now = _utc_now_iso()
        for item in children:
            if item.step_id != step_id or not item.required:
                replaced.append(item)
                continue
            replaced.append(replace(item, required=False))
            previous_dir = resolve_run_dir(item.task_group, item.run_id)
            if previous_dir is not None:
                previous_metadata_path = previous_dir / "metadata.json"
                previous_metadata = _read_json(previous_metadata_path)
                previous_metadata.update(
                    {
                        "superseded_by_run_id": child.run_id,
                        "superseded_at": now,
                    }
                )
                if str(previous_metadata.get("status") or "") in {
                    "queued",
                    "paused",
                }:
                    previous_metadata.update(
                        {
                            "status": "cancelled",
                            "cancelled_at": now,
                            "completed_at": now,
                            "cancellation_requested": True,
                            "cancellation_requested_at": now,
                            "cancellation_requested_by": "workflow-replacement",
                            "cancellation_reason": (
                                "Superseded before execution by workflow child "
                                f"{child.run_id}"
                            ),
                        }
                    )
                _write_json(previous_metadata_path, previous_metadata)
        children = tuple(replaced)
    children = children + (child_ref,)
    if update_child_metadata:
        metadata_path = child.run_dir / "metadata.json"
        metadata = _read_json(metadata_path)
        metadata.update(
            {
                "workflow_id": workflow_id,
                "workflow_parent_run_id": workflow_id,
                "workflow_step_id": step_id,
            }
        )
        _write_json(metadata_path, metadata)
    workflow = replace(workflow, children=children, updated_at=_utc_now_iso())
    _write_workflow(workflow)
    return refresh_workflow(workflow_id)


def _aggregate_status(
    children: list[tuple[WorkflowChildRef, JobRecord | None]], expected_steps: tuple[str, ...]
) -> str:
    required = [job for child, job in children if child.required]
    if not required:
        return "queued"
    states = [job.status if job is not None else "blocked" for job in required]
    attached_steps = {child.step_id for child, _ in children}
    missing_expected = bool(set(expected_steps) - attached_steps)
    # A queued downstream child must not make a workflow look active after a
    # required upstream child has already failed or become blocked.
    if any(state == "failed" for state in states):
        return "failed"
    if any(state == "blocked" for state in states):
        return "blocked"
    if all(state == "completed" for state in states) and not missing_expected:
        return "completed"
    if any(state in ACTIVE_STATES for state in states):
        return "running"
    if all(state == "cancelled" for state in states):
        return "cancelled"
    return "queued" if missing_expected else "running"


def refresh_workflow(workflow_id: str) -> WorkflowRecord:
    workflow = WorkflowRecord.load(workflow_id)
    resolved: list[tuple[WorkflowChildRef, JobRecord | None]] = []
    child_rows: list[dict[str, Any]] = []
    for child in workflow.children:
        child_dir = resolve_run_dir(child.task_group, child.run_id)
        job = JobRecord.load(child_dir, task_group=child.task_group) if child_dir is not None else None
        resolved.append((child, job))
        child_rows.append({**child.to_dict(), "status": job.status if job else "blocked"})
    status = _aggregate_status(resolved, workflow.expected_steps)
    now = _utc_now_iso()
    completed_at = now if status in TERMINAL_STATES else ""
    workflow = replace(workflow, status=status, updated_at=now, completed_at=completed_at)
    required = [(child, job) for child, job in resolved if child.required]
    total = max(
        len({child.step_id for child, _ in required}),
        len(workflow.expected_steps),
    )
    completed = len(
        {
            child.step_id
            for child, job in required
            if job is not None and job.status == "completed"
        }
    )
    failed = len(
        {
            child.step_id
            for child, job in required
            if job is None or job.status in {"failed", "blocked"}
        }
    )
    result = {
        "success": status == "completed",
        "progress": {
            "completed": completed,
            "total": total,
            "failed": failed,
            "percent": round(100 * completed / total) if total else 0,
        },
        "children": child_rows,
    }
    _write_workflow(workflow, result=result)
    return workflow


def update_workflow_definition(
    workflow_id: str,
    *,
    parameters: dict[str, Any] | None = None,
    expected_steps: Iterable[str] | None = None,
) -> WorkflowRecord:
    """Update mutable orchestration intent without replacing the workflow ID."""
    workflow = WorkflowRecord.load(workflow_id)
    workflow = replace(
        workflow,
        parameters=(dict(parameters) if parameters is not None else workflow.parameters),
        expected_steps=(
            tuple(dict.fromkeys(str(item) for item in expected_steps if str(item)))
            if expected_steps is not None
            else workflow.expected_steps
        ),
        status="queued" if workflow.status in TERMINAL_STATES else workflow.status,
        completed_at="",
        updated_at=_utc_now_iso(),
    )
    _write_workflow(workflow)
    return refresh_workflow(workflow_id)


def _write_workflow(workflow: WorkflowRecord, *, result: dict[str, Any] | None = None) -> None:
    _write_json(workflow.run_dir / "workflow.json", workflow.to_dict())
    metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "run_id": workflow.workflow_id,
        "job_code": short_job_code(workflow.workflow_id),
        "job_type": "workflow",
        "workflow": workflow.workflow_type,
        "workflow_id": workflow.workflow_id,
        "is_workflow_parent": True,
        "name": workflow.name,
        "status": workflow.status,
        "created_at": workflow.created_at,
        "updated_at": workflow.updated_at,
        "completed_at": workflow.completed_at,
        "child_count": len(workflow.children),
        "parameters": workflow.parameters,
    }
    for key in ("target_key", "target_run_id", "target_provenance_key"):
        if workflow.parameters.get(key):
            metadata[key] = workflow.parameters[key]
    _write_json(workflow.run_dir / "metadata.json", metadata)
    if result is None:
        result = _read_json(workflow.run_dir / "result.json") or {
            "success": False,
            "progress": {"completed": 0, "total": len(workflow.children), "failed": 0, "percent": 0},
            "children": [],
        }
    _write_json(workflow.run_dir / "result.json", result)
    _write_json(
        workflow.run_dir / "input.json",
        {"parameters": workflow.parameters, "inputs": [item.to_dict() for item in workflow.inputs]},
    )
