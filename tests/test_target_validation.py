from __future__ import annotations

import pytest

from mn_ligand.ligandx.lib.chemistry.preparation.target_validation import (
    audit_target_geometry,
    prepare_target_for_publication,
)


def _atom(
    serial: int,
    name: str,
    residue: str,
    number: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {name:^4s} {residue:>3s} A{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}  "
    )


def test_publication_removes_oxt_that_became_internal() -> None:
    pdb = "\n".join(
        [
            _atom(1, "N", "GLU", 250, 0.0, 0.0, 0.0, "N"),
            _atom(2, "CA", "GLU", 250, 1.4, 0.0, 0.0, "C"),
            _atom(3, "C", "GLU", 250, 2.8, 0.0, 0.0, "C"),
            _atom(4, "O", "GLU", 250, 3.4, 1.0, 0.0, "O"),
            _atom(5, "OXT", "GLU", 250, 4.1, 0.0, 0.0, "O"),
            _atom(6, "N", "GLY", 251, 4.13, 0.0, 0.0, "N"),
            _atom(7, "CA", "GLY", 251, 5.5, 0.0, 0.0, "C"),
            _atom(8, "C", "GLY", 251, 6.9, 0.0, 0.0, "C"),
            "TER",
            "END",
            "",
        ]
    )

    repaired, report = prepare_target_for_publication(pdb)

    assert " OXT " not in repaired
    assert report["removed_internal_oxt_count"] == 1
    assert report["valid"] is True


def test_publication_rejects_nonbonded_side_chain_overlap() -> None:
    pdb = "\n".join(
        [
            _atom(1, "N", "GLN", 34, 0.0, 0.0, 0.0, "N"),
            _atom(2, "CA", "GLN", 34, 1.4, 0.0, 0.0, "C"),
            _atom(3, "C", "GLN", 34, 2.8, 0.0, 0.0, "C"),
            _atom(4, "NE2", "GLN", 34, 0.0, 4.0, 0.0, "N"),
            _atom(5, "N", "ALA", 35, 4.13, 0.0, 0.0, "N"),
            _atom(6, "CA", "ALA", 35, 5.5, 0.0, 0.0, "C"),
            _atom(7, "C", "ALA", 35, 6.9, 0.0, 0.0, "C"),
            _atom(8, "N", "LYS", 41, 8.23, 0.0, 0.0, "N"),
            _atom(9, "CA", "LYS", 41, 9.6, 0.0, 0.0, "C"),
            _atom(10, "C", "LYS", 41, 11.0, 0.0, 0.0, "C"),
            _atom(11, "NZ", "LYS", 41, 1.45, 4.0, 0.0, "N"),
            "TER",
            "END",
            "",
        ]
    )

    audit = audit_target_geometry(pdb)
    assert audit["valid"] is False
    assert audit["severe_clash_count"] == 1
    with pytest.raises(ValueError, match="GLN34:NE2.*LYS41:NZ"):
        prepare_target_for_publication(pdb)
