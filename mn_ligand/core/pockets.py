from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


POCKET_SET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResidueRef:
    chain_id: str
    residue_name: str
    residue_number: str
    insertion_code: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "chain_id": self.chain_id,
            "residue_name": self.residue_name,
            "residue_number": self.residue_number,
            "insertion_code": self.insertion_code,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResidueRef:
        return cls(
            chain_id=str(payload.get("chain_id") or "_"),
            residue_name=str(payload.get("residue_name") or "UNK"),
            residue_number=str(payload.get("residue_number") or ""),
            insertion_code=str(payload.get("insertion_code") or ""),
        )


@dataclass(frozen=True)
class PocketRecord:
    pocket_id: str
    rank: int
    method: str
    center_angstrom: tuple[float, float, float]
    size_angstrom: tuple[float, float, float]
    score: float | None = None
    druggability_score: float | None = None
    residues: tuple[ResidueRef, ...] = ()
    descriptors: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    structure_path: str = ""
    points_path: str = ""

    def __post_init__(self) -> None:
        if not self.pocket_id.strip() or not self.method.strip():
            raise ValueError("Pocket ID and method are required")
        if self.rank < 1:
            raise ValueError("Pocket rank must be positive")
        if len(self.center_angstrom) != 3 or len(self.size_angstrom) != 3:
            raise ValueError("Pocket center and size must contain three values")
        if any(value <= 0 for value in self.size_angstrom):
            raise ValueError("Pocket box dimensions must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pocket_id": self.pocket_id,
            "rank": self.rank,
            "method": self.method,
            "score": self.score,
            "druggability_score": self.druggability_score,
            "center_angstrom": list(self.center_angstrom),
            "size_angstrom": list(self.size_angstrom),
            "residues": [residue.to_dict() for residue in self.residues],
            "descriptors": self.descriptors,
            "metadata": self.metadata,
            "structure_path": self.structure_path,
            "points_path": self.points_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PocketRecord:
        return cls(
            pocket_id=str(payload.get("pocket_id") or ""),
            rank=int(payload.get("rank") or 0),
            method=str(payload.get("method") or ""),
            score=float(payload["score"]) if payload.get("score") is not None else None,
            druggability_score=(
                float(payload["druggability_score"])
                if payload.get("druggability_score") is not None
                else None
            ),
            center_angstrom=tuple(float(value) for value in payload.get("center_angstrom") or ()),
            size_angstrom=tuple(float(value) for value in payload.get("size_angstrom") or ()),
            residues=tuple(ResidueRef.from_dict(item) for item in payload.get("residues") or ()),
            descriptors={str(key): float(value) for key, value in (payload.get("descriptors") or {}).items()},
            metadata=dict(payload.get("metadata") or {}),
            structure_path=str(payload.get("structure_path") or ""),
            points_path=str(payload.get("points_path") or ""),
        )


@dataclass(frozen=True)
class PocketSet:
    method: str
    source_target: dict[str, Any]
    pockets: tuple[PocketRecord, ...]
    source_complex: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    schema_version: int = POCKET_SET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != POCKET_SET_SCHEMA_VERSION:
            raise ValueError(f"Unsupported pocket-set schema version: {self.schema_version}")
        if not self.method.strip():
            raise ValueError("Pocket-set method is required")
        if any(pocket.method != self.method for pocket in self.pockets):
            raise ValueError("Every pocket must use the pocket-set method")
        ranks = [pocket.rank for pocket in self.pockets]
        if ranks != sorted(set(ranks)):
            raise ValueError("Pocket ranks must be unique and ordered")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "pocket_set",
            "schema_version": self.schema_version,
            "method": self.method,
            "source_target": self.source_target,
            "source_complex": self.source_complex,
            "parameters": self.parameters,
            "pockets": [pocket.to_dict() for pocket in self.pockets],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PocketSet:
        return cls(
            method=str(payload.get("method") or ""),
            source_target=dict(payload.get("source_target") or {}),
            source_complex=dict(payload.get("source_complex") or {}),
            parameters=dict(payload.get("parameters") or {}),
            pockets=tuple(PocketRecord.from_dict(item) for item in payload.get("pockets") or ()),
            schema_version=int(payload.get("schema_version") or 0),
        )

    @classmethod
    def read(cls, path: Path) -> PocketSet:
        return cls.from_dict(json.loads(path.read_text()))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
