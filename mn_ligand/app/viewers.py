from __future__ import annotations

import json
from difflib import SequenceMatcher
from html import escape
import re
from typing import Any, Sequence

import streamlit as st


def is_mmcif_text(value: object) -> bool:
    text = str(value or "")
    return bool(
        "_atom_site." in text
        and re.search(r"(?m)^data_(?:\S.*)?$", text)
    )


def horizontalize_2d_coordinates(
    coordinates: Any,
    *,
    anchor_groups: Sequence[Sequence[int]] = (),
):
    import numpy as np

    def rotate(values: Any, angle: float):
        cosine = float(np.cos(-angle))
        sine = float(np.sin(-angle))
        matrix = np.asarray([[cosine, -sine], [sine, cosine]])
        return values @ matrix.T

    centered = np.asarray(coordinates, dtype=float).copy()
    if centered.size == 0:
        return centered
    centered -= np.mean(centered, axis=0)
    if centered.shape[0] < 2:
        return centered
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.arctan2(principal_axis[1], principal_axis[0]))
    oriented = rotate(centered, angle)
    valid_groups = [
        [int(index) for index in group if 0 <= int(index) < len(oriented)]
        for group in anchor_groups
    ]
    valid_groups = [group for group in valid_groups if group]
    if len(valid_groups) >= 2:
        centers = np.asarray([
            np.mean(oriented[group], axis=0) for group in valid_groups
        ])
        distances = np.linalg.norm(
            centers[:, np.newaxis, :] - centers[np.newaxis, :, :],
            axis=2,
        )
        start, end = np.unravel_index(
            int(np.argmax(distances)), distances.shape
        )
        reference_span = centers[end] - centers[start]
    else:
        distances = np.linalg.norm(
            oriented[:, np.newaxis, :] - oriented[np.newaxis, :, :],
            axis=2,
        )
        start, end = np.unravel_index(
            int(np.argmax(distances)), distances.shape
        )
        reference_span = oriented[end] - oriented[start]
    span_angle = float(np.arctan2(reference_span[1], reference_span[0]))
    return rotate(oriented, span_angle)


def transform_2d_coordinates(
    coordinates: Any,
    *,
    rotation_degrees: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
):
    import numpy as np

    transformed = np.asarray(coordinates, dtype=float).copy()
    if flip_horizontal:
        transformed[:, 0] *= -1.0
    if flip_vertical:
        transformed[:, 1] *= -1.0
    angle = float(np.deg2rad(rotation_degrees))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return transformed @ rotation.T


def pdb_interaction_atom_coordinates(
    pdb_text: str,
) -> tuple[dict[tuple[str, str, str], Any], dict[str, Any]]:
    import numpy as np

    protein: dict[tuple[str, str, str], Any] = {}
    ligand: dict[str, Any] = {}
    if is_mmcif_text(pdb_text):
        try:
            import gemmi

            block = gemmi.cif.read_string(pdb_text).sole_block()
            atoms = block.find_mmcif_category("_atom_site.")
            for row in atoms:
                record = row["_atom_site.group_PDB"].strip()
                atom_name = row["_atom_site.label_atom_id"].strip().upper()
                atom_serial = row["_atom_site.id"].strip()
                chain = (
                    row["_atom_site.auth_asym_id"].strip()
                    or row["_atom_site.label_asym_id"].strip()
                )
                residue_number = (
                    row["_atom_site.auth_seq_id"].strip()
                    or row["_atom_site.label_seq_id"].strip()
                )
                residue_name = row[
                    "_atom_site.label_comp_id"
                ].strip().upper()
                coordinates = np.asarray(
                    [
                        float(row["_atom_site.Cartn_x"]),
                        float(row["_atom_site.Cartn_y"]),
                        float(row["_atom_site.Cartn_z"]),
                    ],
                    dtype=float,
                )
                if record == "ATOM":
                    protein[
                        (chain, residue_number, atom_name)
                    ] = coordinates
                    if atom_serial:
                        protein[
                            (chain, residue_number, f"#{atom_serial}")
                        ] = coordinates
                elif residue_name not in {"HOH", "WAT", "SOL"}:
                    ligand.setdefault(atom_name, coordinates)
                    if atom_serial:
                        ligand.setdefault(f"#{atom_serial}", coordinates)
            return protein, ligand
        except (ImportError, RuntimeError, ValueError):
            pass
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"} or len(line) < 54:
            continue
        try:
            coordinates = np.asarray(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
                dtype=float,
            )
        except ValueError:
            continue
        atom_name = line[12:16].strip()
        atom_serial = line[6:11].strip()
        chain = line[21:22].strip()
        residue_number = line[22:26].strip()
        residue_name = line[17:20].strip().upper()
        if record == "ATOM":
            protein[(chain, residue_number, atom_name.upper())] = coordinates
            if atom_serial:
                protein[(chain, residue_number, f"#{atom_serial}")] = (
                    coordinates
                )
        elif residue_name not in {"HOH", "WAT", "SOL"}:
            ligand.setdefault(atom_name.upper(), coordinates)
            if atom_serial:
                ligand.setdefault(f"#{atom_serial}", coordinates)
    return protein, ligand


def pdb_ligand_atom_aliases(
    pdb_text: str,
    molecule: Any,
    *,
    tolerance_angstrom: float = 0.25,
) -> dict[str, int]:
    import numpy as np

    if molecule is None or not molecule.GetNumConformers():
        return {}
    conformer = molecule.GetConformer()
    molecule_atoms = [
        (
            atom.GetIdx(),
            atom.GetSymbol().upper(),
            np.asarray(conformer.GetAtomPosition(atom.GetIdx()), dtype=float),
        )
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() != 1
    ]
    aliases: dict[str, int] = {}
    assigned: set[int] = set()
    for line in pdb_text.splitlines():
        if line[:6].strip() != "HETATM" or len(line) < 54:
            continue
        residue_name = line[17:20].strip().upper()
        if residue_name in {"HOH", "WAT", "SOL"}:
            continue
        try:
            coordinates = np.asarray(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
                dtype=float,
            )
        except ValueError:
            continue
        atom_name = line[12:16].strip().upper()
        atom_serial = line[6:11].strip()
        element = line[76:78].strip().upper()
        if not element:
            element = re.sub(r"[^A-Za-z]", "", atom_name).upper()
            if len(element) > 1 and element[:2] not in {
                "CL", "BR",
            }:
                element = element[:1]
        candidates = [
            (
                float(np.linalg.norm(atom_coordinates - coordinates)),
                atom_index,
            )
            for atom_index, atom_element, atom_coordinates in molecule_atoms
            if atom_index not in assigned and atom_element == element
        ]
        if not candidates:
            continue
        distance, atom_index = min(candidates)
        if distance > max(0.0, float(tolerance_angstrom)):
            continue
        assigned.add(atom_index)
        if atom_name:
            aliases.setdefault(atom_name, atom_index)
        if atom_serial:
            aliases.setdefault(f"#{atom_serial}", atom_index)
    return aliases


def closest_residue_ligand_atom_pair(
    protein_coordinates: dict[tuple[str, str, str], Any],
    ligand_coordinates: dict[str, Any],
    *,
    chain: str,
    residue_number: str,
) -> tuple[Any, Any] | None:
    import numpy as np

    protein_atoms = [
        coordinates
        for (atom_chain, atom_residue, atom_name), coordinates
        in protein_coordinates.items()
        if atom_chain == chain
        and atom_residue == residue_number
        and not atom_name.startswith("#")
    ]
    ligand_atoms = [
        coordinates
        for atom_name, coordinates in ligand_coordinates.items()
        if not atom_name.startswith("#")
    ]
    if not protein_atoms or not ligand_atoms:
        return None
    return min(
        (
            (
                float(np.linalg.norm(protein_atom - ligand_atom)),
                protein_atom,
                ligand_atom,
            )
            for protein_atom in protein_atoms
            for ligand_atom in ligand_atoms
        ),
        key=lambda item: item[0],
    )[1:]


def dashed_line_segments(
    start: Any,
    end: Any,
    *,
    segment_count: int = 11,
) -> list[tuple[Any, Any]]:
    import numpy as np

    count = max(3, int(segment_count))
    origin = np.asarray(start, dtype=float)
    delta = np.asarray(end, dtype=float) - origin
    return [
        (
            origin + delta * (segment_index / count),
            origin + delta * (min(segment_index + 1, count) / count),
        )
        for segment_index in range(0, count, 2)
    ]


def distribute_2d_labels(
    anchor_coordinates: Any,
    *,
    x_radius: float,
    y_radius: float,
):
    import numpy as np

    anchors = np.asarray(anchor_coordinates, dtype=float)
    if anchors.size == 0:
        return anchors.reshape((-1, 2))
    x_radius = max(0.1, float(x_radius))
    y_radius = max(0.1, float(y_radius))
    anchor_scale = np.asarray([
        max(float(np.max(np.abs(anchors[:, 0]))), 1.0),
        max(float(np.max(np.abs(anchors[:, 1]))), 1.0),
    ])
    anchor_angles = np.mod(
        np.arctan2(
            anchors[:, 1] / anchor_scale[1],
            anchors[:, 0] / anchor_scale[0],
        ),
        2.0 * np.pi,
    )
    order = np.argsort(anchor_angles, kind="stable")
    count = len(anchors)

    # Equal parameter angles crowd labels at the narrow ends of an ellipse.
    # Sample its perimeter and place labels at equal arc-length intervals.
    sample_angles = np.linspace(0.0, 2.0 * np.pi, 4097)
    sample_points = np.column_stack((
        x_radius * np.cos(sample_angles),
        y_radius * np.sin(sample_angles),
    ))
    cumulative = np.concatenate((
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(sample_points, axis=0), axis=1)),
    ))
    perimeter = float(cumulative[-1])
    fractions = cumulative / perimeter

    # Preserve circular attachment order and choose the phase that minimizes
    # leader-line travel without sacrificing even perimeter spacing.
    best_angles = None
    best_cost = float("inf")
    ordered_anchor_angles = anchor_angles[order]
    for phase in np.linspace(0.0, 1.0 / count, 129, endpoint=False):
        slot_fractions = np.mod(
            phase + np.arange(count, dtype=float) / count,
            1.0,
        )
        slot_angles = np.interp(slot_fractions, fractions, sample_angles)
        circular_difference = np.angle(
            np.exp(1j * (slot_angles - ordered_anchor_angles))
        )
        cost = float(np.sum(circular_difference**2))
        if cost < best_cost:
            best_cost = cost
            best_angles = slot_angles
    positions = np.zeros_like(anchors)
    positions[order, 0] = x_radius * np.cos(best_angles)
    positions[order, 1] = y_radius * np.sin(best_angles)
    return positions


@st.cache_data(show_spinner=False)
def aligned_structure_data(
    reference_path_text: str,
    reference_modified_ns: int,
    mobile_path_text: str,
    mobile_modified_ns: int,
) -> tuple[str, float, int]:
    """Rigidly align a complex to a reference protein using matched Cα atoms."""
    del reference_modified_ns, mobile_modified_ns
    import gemmi
    import numpy as np

    def read_structure(path_text: str):
        from pathlib import Path

        text = Path(path_text).read_text(errors="replace")
        if is_mmcif_text(text):
            return gemmi.make_structure_from_block(
                gemmi.cif.read_string(text).sole_block()
            )
        return gemmi.read_structure(path_text)

    reference = read_structure(reference_path_text)
    mobile = read_structure(mobile_path_text)

    def alpha_carbons(
        structure: Any,
    ) -> tuple[
        dict[tuple[str, int, str, str], Any],
        list[Any],
        dict[str, list[tuple[str, Any]]],
    ]:
        keyed: dict[tuple[str, int, str, str], Any] = {}
        ordered: list[Any] = []
        chains: dict[str, list[tuple[str, Any]]] = {}
        for chain in structure[0]:
            for residue in chain:
                residue_info = gemmi.find_tabulated_residue(residue.name)
                if not residue_info.is_amino_acid():
                    continue
                atom = next(
                    (
                        item
                        for item in residue
                        if item.name.strip() == "CA"
                    ),
                    None,
                )
                if atom is None:
                    continue
                keyed[
                    (
                        chain.name,
                        int(residue.seqid.num),
                        str(residue.seqid.icode).strip(),
                        residue.name.strip().upper(),
                    )
                ] = atom
                ordered.append(atom)
                chains.setdefault(chain.name, []).append(
                    (residue_info.one_letter_code, atom)
                )
        return keyed, ordered, chains

    (
        reference_keyed,
        reference_ordered,
        reference_chains,
    ) = alpha_carbons(reference)
    mobile_keyed, mobile_ordered, mobile_chains = alpha_carbons(mobile)

    # Exact chain/residue identifiers are the strongest correspondence.  This
    # also handles an extracted, non-contiguous pocket correctly: concatenating
    # its sparse residues and sequence-aligning them to the complete receptor can
    # otherwise produce a plausible-looking but scientifically wrong fit.
    common = sorted(set(reference_keyed).intersection(mobile_keyed))
    reference_atoms = [reference_keyed[key] for key in common]
    mobile_atoms = [mobile_keyed[key] for key in common]
    smaller_structure_size = min(len(reference_ordered), len(mobile_ordered))
    exact_id_coverage = (
        len(reference_atoms) / smaller_structure_size
        if smaller_structure_size
        else 0.0
    )
    if len(reference_atoms) < 3 or exact_id_coverage < 0.5:
        chain_candidates: list[
            tuple[int, int, str, str, list[tuple[Any, Any]]]
        ] = []
        for reference_name, reference_residues in reference_chains.items():
            reference_sequence = "".join(
                residue for residue, _ in reference_residues
            )
            for mobile_name, mobile_residues in mobile_chains.items():
                mobile_sequence = "".join(
                    residue for residue, _ in mobile_residues
                )
                matcher = SequenceMatcher(
                    None,
                    reference_sequence,
                    mobile_sequence,
                    autojunk=False,
                )
                pairs: list[tuple[Any, Any]] = []
                for block in matcher.get_matching_blocks():
                    for offset in range(block.size):
                        pairs.append(
                            (
                                reference_residues[block.a + offset][1],
                                mobile_residues[block.b + offset][1],
                            )
                        )
                chain_candidates.append(
                    (
                        len(pairs),
                        int(reference_name == mobile_name),
                        reference_name,
                        mobile_name,
                        pairs,
                    )
                )
        reference_atoms = []
        mobile_atoms = []
        used_reference_chains: set[str] = set()
        used_mobile_chains: set[str] = set()
        for _, _, reference_name, mobile_name, pairs in sorted(
            chain_candidates,
            reverse=True,
        ):
            if (
                reference_name in used_reference_chains
                or mobile_name in used_mobile_chains
                or len(pairs) < 3
            ):
                continue
            used_reference_chains.add(reference_name)
            used_mobile_chains.add(mobile_name)
            reference_atoms.extend(pair[0] for pair in pairs)
            mobile_atoms.extend(pair[1] for pair in pairs)
    if len(reference_atoms) < 3:
        reference_atoms = [reference_keyed[key] for key in common]
        mobile_atoms = [mobile_keyed[key] for key in common]
    if len(reference_atoms) < 3:
        pair_count = min(len(reference_ordered), len(mobile_ordered))
        reference_atoms = reference_ordered[:pair_count]
        mobile_atoms = mobile_ordered[:pair_count]
    if len(reference_atoms) < 3:
        raise ValueError(
            "Fewer than three matched protein Cα atoms are available"
        )

    reference_xyz = np.asarray(
        [
            [atom.pos.x, atom.pos.y, atom.pos.z]
            for atom in reference_atoms
        ],
        dtype=float,
    )
    mobile_xyz = np.asarray(
        [[atom.pos.x, atom.pos.y, atom.pos.z] for atom in mobile_atoms],
        dtype=float,
    )
    reference_center = reference_xyz.mean(axis=0)
    mobile_center = mobile_xyz.mean(axis=0)
    covariance = (mobile_xyz - mobile_center).T @ (
        reference_xyz - reference_center
    )
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1, :] *= -1
        rotation = right_transpose.T @ left.T
    aligned_ca = (
        rotation @ (mobile_xyz - mobile_center).T
    ).T + reference_center
    rmsd = float(
        np.sqrt(
            np.mean(
                np.sum((aligned_ca - reference_xyz) ** 2, axis=1)
            )
        )
    )
    for model in mobile:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coordinate = np.asarray(
                        [atom.pos.x, atom.pos.y, atom.pos.z],
                        dtype=float,
                    )
                    transformed = (
                        rotation @ (coordinate - mobile_center)
                        + reference_center
                    )
                    atom.pos = gemmi.Position(*map(float, transformed))
    return (
        mobile.make_mmcif_document().as_string(),
        rmsd,
        len(reference_atoms),
    )


_VIEWER_VARIABLE = re.compile(r"var (viewer_[A-Za-z0-9_]+) = null;")


def persistent_3dmol_html(viewer: Any, *, key: str) -> str:
    """Add browser-session camera persistence to py3Dmol's generated HTML."""
    html = viewer._make_html()
    match = _VIEWER_VARIABLE.search(html)
    if match is None:
        return html
    variable = match.group(1)
    render_call = f"{variable}.render();"
    render_index = html.rfind(render_call)
    if render_index < 0:
        return html
    storage_key = json.dumps(f"mn-ligand:3dmol-camera:{key}")
    persistence = f"""
const __mnCameraKey = {storage_key};
const __mnCameraStore = (() => {{
  const candidates = [];
  try {{ candidates.push(window.parent.sessionStorage); }} catch (_) {{}}
  try {{ candidates.push(window.sessionStorage); }} catch (_) {{}}
  try {{ candidates.push(window.localStorage); }} catch (_) {{}}
  for (const candidate of candidates) {{
    try {{
      const probe = __mnCameraKey + ":probe";
      candidate.setItem(probe, "1");
      candidate.removeItem(probe);
      return candidate;
    }} catch (_) {{}}
  }}
  return null;
}})();
if (__mnCameraStore) {{
  try {{
    const saved = JSON.parse(__mnCameraStore.getItem(__mnCameraKey));
    if (Array.isArray(saved)) {variable}.setView(saved);
  }} catch (_) {{}}
}}
{variable}.render();
const __mnPersistCamera = () => {{
  if (!__mnCameraStore) return;
  try {{
    __mnCameraStore.setItem(
      __mnCameraKey, JSON.stringify({variable}.getView())
    );
  }} catch (_) {{}}
}};
const __mnViewerElement = document.getElementById(
  {json.dumps(variable.replace("viewer_", "3dmolviewer_"))}
);
if (__mnViewerElement) {{
  for (const eventName of ["pointerup", "mouseup", "touchend", "wheel"]) {{
    __mnViewerElement.addEventListener(eventName, __mnPersistCamera, {{
      passive: true
    }});
  }}
}}
// Persist only after an actual camera interaction.  A periodic getView call
// is unnecessary and can cause visible repaint pressure for complex viewers
// with many interaction sticks and labels.
window.addEventListener("pagehide", __mnPersistCamera, {{ passive: true }});
"""
    return (
        html[:render_index]
        + persistence
        + html[render_index + len(render_call):]
    )


def render_persistent_3dmol(
    viewer: Any,
    *,
    key: str,
    height: int,
    scrolling: bool = False,
    panel_titles: Sequence[str] = (),
    panel_columns: int = 1,
    panel_width: int | None = None,
) -> None:
    html = persistent_3dmol_html(viewer, key=key)
    titles = [str(title).strip() for title in panel_titles]
    if titles:
        columns = max(1, min(int(panel_columns), len(titles)))
        rows = (len(titles) + columns - 1) // columns
        overlay_width = (
            f"min({max(1, int(panel_width))}px, 100vw)"
            if panel_width is not None
            else "100vw"
        )
        cells = "".join(
            (
                '<div class="mn-panel-title-cell">'
                f'<div class="mn-panel-title">{escape(title)}</div>'
                "</div>"
            )
            for title in titles
        )
        overlay = f"""
<style>
.mn-panel-title-grid {{
  position: fixed;
  top: 0;
  bottom: 0;
  left: 50%;
  width: {overlay_width};
  transform: translateX(-50%);
  z-index: 20;
  display: grid;
  grid-template-columns: repeat({columns}, minmax(0, 1fr));
  grid-template-rows: repeat({rows}, minmax(0, 1fr));
  pointer-events: none;
}}
.mn-panel-title-cell {{
  min-width: 0;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 8px 10px;
}}
.mn-panel-title {{
  max-width: calc(100% - 12px);
  padding: 5px 10px;
  border: 1px solid rgba(148, 163, 184, 0.75);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.92);
  color: #0f172a;
  font: 600 13px/1.25 sans-serif;
  text-align: center;
  overflow-wrap: anywhere;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.18);
}}
</style>
<div class="mn-panel-title-grid">{cells}</div>
"""
        body_end = html.lower().rfind("</body>")
        html = (
            html[:body_end] + overlay + html[body_end:]
            if body_end >= 0
            else html + overlay
        )
    st.iframe(
        html,
        height=height,
    )
