from __future__ import annotations

from pathlib import Path

from mn_ligand.app.viewers import (
    aligned_structure_data,
    persistent_3dmol_html,
)


class _Viewer:
    def _make_html(self) -> str:
        return """
<div id="3dmolviewer_123"></div>
<script>
var viewer_123 = null;
$3Dmolpromise.then(function() {
viewer_123 = $3Dmol.createViewer(
  document.getElementById("3dmolviewer_123"), {}
);
viewer_123.zoomTo();
viewer_123.render();
});
</script>
"""


def test_persistent_3dmol_html_restores_and_saves_camera() -> None:
    html = persistent_3dmol_html(
        _Viewer(), key="job-results:run-1"
    )

    assert "mn-ligand:3dmol-camera:job-results:run-1" in html
    assert "viewer_123.setView(saved)" in html
    assert "viewer_123.getView()" in html
    assert "window.parent.sessionStorage" in html
    assert 'addEventListener(eventName, __mnPersistCamera' in html
    assert html.count("viewer_123.render();") == 1


def test_persistent_3dmol_html_leaves_unknown_html_unchanged() -> None:
    html = "<div>viewer unavailable</div>"
    assert persistent_3dmol_html(_ViewerWithoutVariable(html), key="x") == html


def test_alignment_matches_sequence_when_prediction_renumbers_residues(
    tmp_path: Path,
) -> None:
    names = ("ALA", "GLY", "SER", "THR", "LEU")
    coordinates = (
        (0.0, 0.0, 0.0),
        (1.5, 0.2, 0.1),
        (2.2, 1.4, 0.3),
        (3.1, 1.7, 1.6),
        (4.0, 2.8, 1.1),
    )

    def pdb_text(*, start: int, offset: tuple[float, float, float]) -> str:
        rows = []
        for serial, (name, coordinate) in enumerate(
            zip(names, coordinates),
            start=1,
        ):
            x, y, z = (
                coordinate[index] + offset[index] for index in range(3)
            )
            rows.append(
                f"ATOM  {serial:5d}  CA  {name:>3s} A"
                f"{start + serial - 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                "  1.00 20.00           C"
            )
        return "\n".join(rows) + "\nEND\n"

    reference = tmp_path / "reference.pdb"
    mobile = tmp_path / "mobile.pdb"
    reference.write_text(pdb_text(start=2, offset=(0.0, 0.0, 0.0)))
    mobile.write_text(pdb_text(start=1, offset=(8.0, -3.0, 5.0)))

    _, rmsd, matched = aligned_structure_data(
        str(reference),
        reference.stat().st_mtime_ns,
        str(mobile),
        mobile.stat().st_mtime_ns,
    )

    assert matched == 5
    assert rmsd < 1e-6


def test_alignment_prefers_exact_ids_for_sparse_extracted_pocket(
    tmp_path: Path,
) -> None:
    receptor_rows = []
    pocket_rows = []
    pocket_residues = {2, 5, 9}
    for serial, residue_number in enumerate(range(1, 11), start=1):
        x = float(residue_number)
        y = float((residue_number * residue_number) % 7)
        z = float((residue_number * 3) % 5)
        row = (
            f"ATOM  {serial:5d}  CA  ALA A{residue_number:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            "  1.00 20.00           C"
        )
        receptor_rows.append(row)
        if residue_number in pocket_residues:
            pocket_rows.append(row)
    receptor = tmp_path / "receptor.pdb"
    pocket = tmp_path / "pocket.pdb"
    receptor.write_text("\n".join(receptor_rows) + "\nEND\n")
    pocket.write_text("\n".join(pocket_rows) + "\nEND\n")

    _, rmsd, matched = aligned_structure_data(
        str(pocket),
        pocket.stat().st_mtime_ns,
        str(receptor),
        receptor.stat().st_mtime_ns,
    )

    assert matched == 3
    assert rmsd < 1e-6


def test_alignment_uses_sequence_when_few_incidental_ids_overlap(
    tmp_path: Path,
) -> None:
    def pdb_text(*, start: int, offset: tuple[float, float, float]) -> str:
        rows = []
        for serial in range(1, 11):
            coordinate = (
                float(serial),
                float((serial * serial) % 7),
                float((serial * 3) % 5),
            )
            x, y, z = (
                coordinate[index] + offset[index] for index in range(3)
            )
            rows.append(
                f"ATOM  {serial:5d}  CA  ALA A{start + serial - 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                "  1.00 20.00           C"
            )
        return "\n".join(rows) + "\nEND\n"

    reference = tmp_path / "reference.pdb"
    mobile = tmp_path / "mobile.pdb"
    reference.write_text(pdb_text(start=1, offset=(0.0, 0.0, 0.0)))
    mobile.write_text(pdb_text(start=7, offset=(5.0, -2.0, 4.0)))

    _, rmsd, matched = aligned_structure_data(
        str(reference),
        reference.stat().st_mtime_ns,
        str(mobile),
        mobile.stat().st_mtime_ns,
    )

    assert matched == 10
    assert rmsd < 1e-6


class _ViewerWithoutVariable:
    def __init__(self, html: str) -> None:
        self.html = html

    def _make_html(self) -> str:
        return self.html
