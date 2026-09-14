from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable


TOOL_REGISTRY_SCHEMA_VERSION = 1
_TOOL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
VALID_INTEGRATION_STATUSES = frozenset(
    {"validated", "implemented", "experimental", "compatibility", "disabled", "candidate"}
)


@dataclass(frozen=True)
class ResourceRequest:
    gpu: bool = False
    min_vram_gb: float = 0.0
    cpu_threads: int = 1
    ram_gb: float = 1.0
    scratch_gb: float = 1.0
    exclusive_gpu: bool = False
    cuda_min: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResourceRequest:
        request = cls(
            gpu=bool(payload.get("gpu", False)),
            min_vram_gb=float(payload.get("min_vram_gb") or 0.0),
            cpu_threads=int(payload.get("cpu_threads") or 1),
            ram_gb=float(payload.get("ram_gb") or 1.0),
            scratch_gb=float(payload.get("scratch_gb") or 1.0),
            exclusive_gpu=bool(payload.get("exclusive_gpu", False)),
            cuda_min=str(payload.get("cuda_min") or ""),
        )
        if request.cpu_threads < 1:
            raise ValueError("cpu_threads must be at least 1")
        if min(request.min_vram_gb, request.ram_gb, request.scratch_gb) < 0:
            raise ValueError("resource amounts cannot be negative")
        if not request.gpu and (request.min_vram_gb or request.exclusive_gpu or request.cuda_min):
            raise ValueError("CPU tools cannot declare GPU-only resources")
        return request

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu": self.gpu,
            "min_vram_gb": self.min_vram_gb,
            "cpu_threads": self.cpu_threads,
            "ram_gb": self.ram_gb,
            "scratch_gb": self.scratch_gb,
            "exclusive_gpu": self.exclusive_gpu,
            "cuda_min": self.cuda_min,
        }


@dataclass(frozen=True)
class ReferenceRequirement:
    path: str
    required: bool = True
    description: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReferenceRequirement:
        path = str(payload.get("path") or "").strip().replace("\\", "/")
        candidate = Path(path)
        if not path or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Reference path must be relative and traversal-free: {path!r}")
        return cls(
            path=path,
            required=bool(payload.get("required", True)),
            description=str(payload.get("description") or ""),
        )


@dataclass(frozen=True)
class ToolManifest:
    tool_id: str
    name: str
    version: str
    image: str
    image_digest: str = ""
    integration_status: str = "candidate"
    status_notes: str = ""
    accepted_artifact_types: tuple[str, ...] = ()
    produced_artifact_types: tuple[str, ...] = ()
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    references: tuple[ReferenceRequirement, ...] = ()
    healthcheck: tuple[str, ...] = ()
    code_license: str = "unverified"
    weights_license: str = "not_applicable"
    commercial_use: str = "unverified"
    citation: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ToolManifest:
        tool_id = str(payload.get("tool_id") or "").strip()
        name = str(payload.get("name") or "").strip()
        image = str(payload.get("image") or "").strip()
        if not _TOOL_ID_PATTERN.fullmatch(tool_id):
            raise ValueError(f"Invalid tool_id: {tool_id!r}")
        if not name:
            raise ValueError(f"Tool {tool_id!r} is missing a name")
        if not image or any(character.isspace() for character in image):
            raise ValueError(f"Tool {tool_id!r} has an invalid image tag")
        integration_status = str(payload.get("integration_status") or "candidate").strip().lower()
        if integration_status not in VALID_INTEGRATION_STATUSES:
            raise ValueError(
                f"Tool {tool_id!r} has unsupported integration status: {integration_status!r}"
            )
        accepted = tuple(str(value).strip() for value in payload.get("accepted_artifact_types") or ())
        produced = tuple(str(value).strip() for value in payload.get("produced_artifact_types") or ())
        if not accepted or not produced or any(not value for value in (*accepted, *produced)):
            raise ValueError(f"Tool {tool_id!r} must declare accepted and produced artifact types")
        healthcheck = tuple(str(value) for value in payload.get("healthcheck") or ())
        if not healthcheck:
            raise ValueError(f"Tool {tool_id!r} must declare a healthcheck command")
        return cls(
            tool_id=tool_id,
            name=name,
            version=str(payload.get("version") or "unverified"),
            image=image,
            image_digest=str(payload.get("image_digest") or ""),
            integration_status=integration_status,
            status_notes=str(payload.get("status_notes") or ""),
            accepted_artifact_types=accepted,
            produced_artifact_types=produced,
            resources=ResourceRequest.from_dict(dict(payload.get("resources") or {})),
            references=tuple(
                ReferenceRequirement.from_dict(dict(item)) for item in payload.get("references") or ()
            ),
            healthcheck=healthcheck,
            code_license=str(payload.get("code_license") or "unverified"),
            weights_license=str(payload.get("weights_license") or "not_applicable"),
            commercial_use=str(payload.get("commercial_use") or "unverified"),
            citation=str(payload.get("citation") or ""),
        )

    def resolved_healthcheck(self) -> list[str]:
        return [value.replace("{image}", self.image) for value in self.healthcheck]


@dataclass(frozen=True)
class ToolRegistry:
    tools: tuple[ToolManifest, ...]
    schema_version: int = TOOL_REGISTRY_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ToolRegistry:
        version = int(payload.get("schema_version") or 0)
        if version != TOOL_REGISTRY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported tool registry schema version: {version}")
        tools = tuple(ToolManifest.from_dict(dict(item)) for item in payload.get("tools") or ())
        if not tools:
            raise ValueError("Tool registry contains no tools")
        ids = [tool.tool_id for tool in tools]
        if len(ids) != len(set(ids)):
            raise ValueError("Tool registry contains duplicate tool IDs")
        return cls(tools=tools, schema_version=version)

    def get(self, tool_id: str) -> ToolManifest:
        for tool in self.tools:
            if tool.tool_id == tool_id:
                return tool
        raise KeyError(f"Unknown tool: {tool_id}")

    def select(self, tool_ids: Iterable[str]) -> tuple[ToolManifest, ...]:
        return tuple(self.get(tool_id) for tool_id in tool_ids)


def load_tool_registry(path: Path | None = None) -> ToolRegistry:
    if path is None:
        resource = resources.files("mn_ligand.manifests").joinpath("tools.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Tool registry root must be an object")
    return ToolRegistry.from_dict(payload)
