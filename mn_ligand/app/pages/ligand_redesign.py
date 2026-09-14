from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from mn_ligand.app.components.molstar_viewer import molstar_custom_component
from mn_ligand.app.components.molstar_viewer.dataclasses import (
    ChainVisualization,
    StructureVisualization,
)
from mn_ligand.app.pages.discover_inputs import (
    ArtifactChoice,
    artifact_options,
    select_artifact,
    select_target_artifact,
)
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import display_job_code, iter_job_records
from mn_ligand.runtime import reference_root, runs_root
from mn_ligand.workflows.generative_design import (
    GENERATOR_BY_ID,
    create_generation_campaign_job,
    pharmacophore_required_contacts,
    queue_generation_job,
)


REDESIGN_ENGINES = ("pocketxmol", "flowr_root")
_ATOM_SELECTION_KEY = "ligand_redesign_atom_indices"
_ATOM_TABLE_VERSION_KEY = "ligand_redesign_atom_table_version"


def _bound_ligand_choice(
    target: ArtifactChoice | None,
) -> ArtifactChoice | None:
    if target is None or target.job.artifact_manifest is None:
        return None
    for artifact in target.job.artifact_manifest.by_type(
        "prepared_ligand_set"
    ):
        if artifact.resolve(target.job.run_dir, must_exist=True) is not None:
            return ArtifactChoice(job=target.job, artifact=artifact)
    return None


def _bound_ligand_identity(target: ArtifactChoice | None) -> str:
    if target is None:
        return "Bound ligand"
    ligand_key = str(target.job.metadata.get("ligand_key") or "")
    requested_id = ligand_key.partition("|")[0].strip()
    ligands = target.job.metadata.get("ligands")
    candidates = (
        [dict(item) for item in ligands if isinstance(item, dict)]
        if isinstance(ligands, list)
        else []
    )
    selected = next(
        (
            item
            for item in candidates
            if requested_id
            and requested_id
            in {
                str(item.get("ccd_id") or ""),
                str(item.get("resname") or ""),
            }
        ),
        candidates[0] if candidates else {},
    )
    ligand_id = str(
        selected.get("ccd_id")
        or selected.get("resname")
        or requested_id
        or "Bound ligand"
    )
    ligand_name = str(selected.get("name") or "").strip()
    return f"{ligand_id} — {ligand_name}" if ligand_name else ligand_id


def _bound_ligand_pocket_choice(
    target: ArtifactChoice | None,
) -> ArtifactChoice | None:
    if target is None:
        return None
    options = artifact_options(
        ("pocket",),
        source_run_id=target.job.run_id,
    )
    candidates = [
        choice
        for choice in options.values()
        if str(
            choice.job.metadata.get("source")
            or choice.job.metadata.get("tool")
            or ""
        ).lower()
        == "bound_ligand"
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda choice: str(choice.job.created_at or ""),
    )


def _reference_molecule(choice: ArtifactChoice):
    from rdkit import Chem

    path = choice.artifact.resolve(choice.job.run_dir, must_exist=True)
    if path is None:
        raise FileNotFoundError("The selected reference-ligand artifact is unavailable")
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    molecule = next((item for item in supplier if item is not None), None)
    if molecule is None:
        raise ValueError("The selected reference-ligand SDF has no readable molecule")
    molecule = Chem.RemoveHs(molecule)
    if not molecule.GetNumConformers():
        raise ValueError("Ligand redesign requires a coordinate-bearing reference ligand")
    return molecule, path


def _atom_rows(
    molecule,
    target_coordinates: tuple[float, float, float] | None,
) -> list[dict[str, object]]:
    coordinates = molecule.GetConformer().GetPositions()
    rows: list[dict[str, object]] = []
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        position = coordinates[index]
        distance = (
            math.dist(position, target_coordinates)
            if target_coordinates is not None
            else None
        )
        rows.append(
            {
                "Atom index": index,
                "Element": atom.GetSymbol(),
                "Aromatic": atom.GetIsAromatic(),
                "Neighbors": ", ".join(
                    f"{neighbor.GetIdx()}:{neighbor.GetSymbol()}"
                    for neighbor in atom.GetNeighbors()
                ),
                "X": round(float(position[0]), 3),
                "Y": round(float(position[1]), 3),
                "Z": round(float(position[2]), 3),
                "Target distance (Å)": (
                    round(float(distance), 3) if distance is not None else None
                ),
            }
        )
    return rows


def _pseudo_atom_residue_pdb(molecule) -> str:
    """Represent every ligand atom as a selectable Mol* pseudo-residue."""
    from rdkit import Chem

    block = Chem.MolToPDBBlock(molecule)
    atom_index = 0
    rows: list[str] = []
    for line in block.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            rows.append(
                line[:17]
                + "ATM"
                + line[20:21]
                + "L"
                + f"{atom_index + 1:4d}"
                + line[26:]
            )
            atom_index += 1
        else:
            rows.append(line)
    return "\n".join(rows) + "\n"


def _component_atom_indices(value: object) -> set[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return set()
    if not isinstance(value, dict):
        return set()
    selections = value.get("sequenceSelections")
    if not isinstance(selections, list):
        return set()
    selected: set[int] = set()
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        if str(selection.get("chainId") or "") != "L":
            continue
        for residue in selection.get("residues") or ():
            try:
                selected.add(int(residue) - 1)
            except (TypeError, ValueError):
                continue
    return {value for value in selected if value >= 0}


def _protein_only_pdb(path: Path) -> str:
    return (
        "\n".join(
            line
            for line in path.read_text(errors="replace").splitlines()
            if line.startswith(("ATOM  ", "TER", "MODEL", "ENDMDL", "END"))
        )
        + "\n"
    )


def _target_contact(
    hypothesis: ArtifactChoice | None,
) -> tuple[str, tuple[float, float, float] | None, list[dict[str, object]]]:
    if hypothesis is None:
        return "No post-pose target selected", None, []
    contacts = pharmacophore_required_contacts(hypothesis.artifact)
    if not contacts:
        return "Selected hypothesis has no post-pose target", None, []
    labels: list[str] = []
    for contact in contacts:
        atom = contact.get("prepared_protein_atom") or contact.get("protein_atom") or {}
        labels.append(
            f"{atom.get('chain', '')}:{atom.get('residue_name', '')}"
            f"{atom.get('residue_number', '')}:{atom.get('atom_name', '')}"
            f" · ligand {contact.get('ligand_role', 'feature')}"
        )
    selected_label = st.selectbox(
        "Target interaction for redesign guidance",
        labels,
        key="ligand_redesign_target_contact",
        help=(
            "This target guides region selection and remains a post-pose review "
            "criterion. PocketXMol and FLOWR.root do not guarantee the contact."
        ),
    )
    selected = contacts[labels.index(selected_label)]
    atom = selected.get("prepared_protein_atom") or selected.get("protein_atom") or {}
    coordinates = atom.get("coordinates")
    point = (
        tuple(float(value) for value in coordinates[:3])
        if isinstance(coordinates, list) and len(coordinates) >= 3
        else None
    )
    return selected_label, point, contacts


def _set_selected_atoms(
    values: set[int],
    *,
    key_prefix: str = "ligand_redesign",
) -> None:
    selection_key = f"{key_prefix}_atom_indices"
    table_version_key = f"{key_prefix}_atom_table_version"
    st.session_state[selection_key] = sorted(values)
    st.session_state[table_version_key] = (
        int(st.session_state.get(table_version_key, 0)) + 1
    )


def _render_atom_selector(
    molecule,
    *,
    pocket_path: Path | None,
    receptor_path: Path | None,
    target_label: str,
    target_coordinates: tuple[float, float, float] | None,
    key_prefix: str = "ligand_redesign",
    selection_heading: str = "Select the part to replace",
    selection_name: str = "Replace",
    selected_description: str = "replacement region",
) -> tuple[list[int], list[int], list[int]]:
    atom_count = molecule.GetNumAtoms()
    selection_key = f"{key_prefix}_atom_indices"
    table_version_key = f"{key_prefix}_atom_table_version"
    selected = {
        int(value)
        for value in st.session_state.get(selection_key, [])
        if 0 <= int(value) < atom_count
    }
    atom_rows = _atom_rows(molecule, target_coordinates)

    st.markdown(f"#### {selection_heading}")
    st.caption(
        f"Click ligand atoms in Mol* to toggle the {selected_description}. "
        "The atom table below is synchronized and provides an exact auditable "
        "selection."
    )
    action_columns = st.columns(4)
    distance_cutoff = action_columns[0].number_input(
        "Target-near cutoff (Å)",
        min_value=2.0,
        max_value=12.0,
        value=5.0,
        step=0.5,
        disabled=target_coordinates is None,
        key=f"{key_prefix}_distance_cutoff",
    )
    if action_columns[1].button(
        "Suggest target-near atoms",
        disabled=target_coordinates is None,
        width="stretch",
    ):
        _set_selected_atoms(
            {
                int(row["Atom index"])
                for row in atom_rows
                if row["Target distance (Å)"] is not None
                and float(row["Target distance (Å)"]) <= float(distance_cutoff)
            },
            key_prefix=key_prefix,
        )
        st.rerun()
    if action_columns[2].button("Invert selection", width="stretch"):
        _set_selected_atoms(
            set(range(atom_count)) - selected,
            key_prefix=key_prefix,
        )
        st.rerun()
    if action_columns[3].button("Clear selection", width="stretch"):
        _set_selected_atoms(set(), key_prefix=key_prefix)
        st.rerun()

    preserve = sorted(set(range(atom_count)) - selected)
    boundary_anchors = sorted(
        {
            neighbor.GetIdx()
            for index in selected
            for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
            if neighbor.GetIdx() in preserve
        }
    )
    chain_layers = []
    if preserve:
        chain_layers.append(
            ChainVisualization(
                chain_id="L",
                color="uniform",
                color_params={"value": "0x06b6d4"},
                representation_type="ball-and-stick",
                residues=[value + 1 for value in preserve],
                label="Retained scaffold",
            )
        )
    if selected:
        chain_layers.append(
            ChainVisualization(
                chain_id="L",
                color="uniform",
                color_params={"value": "0xf97316"},
                representation_type="ball-and-stick",
                residues=[value + 1 for value in sorted(selected)],
                label=selected_description.capitalize(),
            )
        )
    if boundary_anchors:
        chain_layers.append(
            ChainVisualization(
                chain_id="L",
                color="uniform",
                color_params={"value": "0xfacc15"},
                representation_type="ball-and-stick",
                residues=[value + 1 for value in boundary_anchors],
                label="Attachment anchors",
            )
        )
    structures = [
        StructureVisualization(
            pdb=_pseudo_atom_residue_pdb(molecule),
            color="uniform",
            color_params={"value": "0x06b6d4"},
            representation_type="ball-and-stick",
            highlighted_selections=[
                f"L{value + 1}" for value in sorted(selected)
            ],
            chains=chain_layers,
        )
    ]
    context_path = receptor_path if receptor_path is not None else pocket_path
    if context_path is not None:
        structures.append(
            StructureVisualization(
                pdb=_protein_only_pdb(context_path),
                color="uniform",
                color_params={"value": "0x9ca3af"},
                representation_type="cartoon",
            )
        )
    if target_coordinates is not None:
        x, y, z = target_coordinates
        structures.append(
            StructureVisualization(
                pdb=(
                    "HETATM    1  X   TGT T   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    "  1.00 20.00          Au\nEND\n"
                ),
                color="uniform",
                color_params={"value": "0xfacc15"},
                representation_type="ball-and-stick",
            )
        )
    component_value = molstar_custom_component(
        structures=structures,
        key=f"{key_prefix}_atom_viewer",
        height=650,
        width="100%",
        show_controls=True,
        selection_mode=True,
        force_reload=False,
    )
    clicked = _component_atom_indices(component_value)
    if component_value is not None and clicked != selected:
        _set_selected_atoms(clicked, key_prefix=key_prefix)
        st.rerun()
    st.caption(
        f"Mol* selection: unselected atoms cyan; selected {selected_description} "
        "orange/green; attachment anchors yellow; protein context grey. "
        f"Guidance target: {target_label}."
    )

    table = pd.DataFrame(atom_rows)
    table.insert(
        0,
        selection_name,
        table["Atom index"].map(lambda value: int(value) in selected),
    )
    table.insert(
        1,
        "Anchor",
        table["Atom index"].map(lambda value: int(value) in boundary_anchors),
    )
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=[
            column
            for column in table.columns
            if column not in {selection_name}
        ],
        column_config={
            selection_name: st.column_config.CheckboxColumn(
                selection_name,
                help=f"Checked atoms define the {selected_description}.",
            )
        },
        key=(
            f"{key_prefix}_atom_table_"
            f"{st.session_state.get(table_version_key, 0)}"
        ),
    )
    table_selected = {
        int(row["Atom index"])
        for _, row in edited.iterrows()
        if bool(row[selection_name])
    }
    if table_selected != selected:
        _set_selected_atoms(table_selected, key_prefix=key_prefix)
        st.rerun()
    return preserve, sorted(selected), boundary_anchors


def render() -> None:
    st.title("Ligand Redesign")
    st.caption(
        "Retain a coordinate-bearing fragment or scaffold and replace a selected "
        "local region inside its protein pocket using atom-masked native inference."
    )
    target_tab, region_tab, engines_tab, run_tab, results_tab = st.tabs(
        ["Target", "Redesign Region", "Engines", "Run", "Results"]
    )

    with target_tab:
        target = select_target_artifact(
            "Prepared protein–ligand complex",
            ("prepared_complex", "prepared_target", "prepared_receptor"),
            key="ligand_redesign_target",
        )
        reference = _bound_ligand_choice(target)
        pocket = _bound_ligand_pocket_choice(target)
        if target is not None:
            target_code = display_job_code(
                target.job.metadata.get("job_code"), target.job.run_id
            )
            st.metric("Selected target", target_code)
        if reference is not None:
            st.success(
                "Reference ligand: "
                f"{_bound_ligand_identity(target)} from this exact prepared complex."
            )
        else:
            st.warning(
                "The selected target has no coordinate-bearing prepared ligand. "
                "Ligand redesign requires a complex rather than an unrelated compound."
            )
        if pocket is not None:
            st.info(
                "Pocket context: immutable bound-ligand extraction "
                f"{display_job_code(pocket.job.metadata.get('job_code'), pocket.job.run_id)}."
            )
        else:
            st.warning(
                "No bound-ligand pocket is linked to this complex. Create one on "
                "Pocket Detection and Extraction before launching redesign."
            )
        hypothesis = select_artifact(
            "Pharmacophore hypothesis / post-pose review target (optional)",
            ("pharmacophore_hypothesis",),
            key="ligand_redesign_pharmacophore",
            required=False,
        )

    molecule = None
    reference_path: Path | None = None
    if reference is not None:
        try:
            molecule, reference_path = _reference_molecule(reference)
        except Exception as exc:
            st.error(f"Could not load the bound reference ligand: {exc}")

    target_label = "No post-pose interaction target"
    target_coordinates = None
    contacts: list[dict[str, object]] = []
    with region_tab:
        if molecule is None:
            st.info("Select a prepared complex with a bound ligand first.")
            preserve_indices: list[int] = []
            redesign_indices: list[int] = []
            anchor_indices: list[int] = []
        else:
            target_label, target_coordinates, contacts = _target_contact(
                hypothesis
            )
            pocket_path = (
                pocket.artifact.resolve(pocket.job.run_dir, must_exist=True)
                if pocket is not None
                else None
            )
            show_full_receptor = st.checkbox(
                "Show full receptor instead of extracted pocket",
                value=False,
                key="ligand_redesign_show_receptor",
            )
            receptor_path = (
                target.artifact.resolve(target.job.run_dir, must_exist=True)
                if show_full_receptor and target is not None
                else None
            )
            preserve_indices, redesign_indices, anchor_indices = (
                _render_atom_selector(
                    molecule,
                    pocket_path=pocket_path,
                    receptor_path=receptor_path,
                    target_label=target_label,
                    target_coordinates=target_coordinates,
                )
            )
            metrics = st.columns(4)
            metrics[0].metric("Heavy atoms", molecule.GetNumAtoms())
            metrics[1].metric("Retained", len(preserve_indices))
            metrics[2].metric("Replace", len(redesign_indices))
            metrics[3].metric("Attachment anchors", len(anchor_indices))

    engine_settings: dict[str, dict[str, object]] = {}
    selected_engines: list[str] = []
    with engines_tab:
        st.markdown("#### Native local-redesign engines")
        selection_columns = st.columns(2)
        if selection_columns[0].checkbox(
            "PocketXMol",
            value=True,
            key="ligand_redesign_engine_pocketxmol",
            help=(
                "MaskFill partial optimization: retained atoms are part1, selected "
                "atoms are regenerated as part2, and boundary atoms are anchors."
            ),
        ):
            selected_engines.append("pocketxmol")
        if selection_columns[1].checkbox(
            "FLOWR.root",
            value=True,
            key="ligand_redesign_engine_flowr",
            help=(
                "Native substructure inpainting: the selected atom indices are "
                "locally replaced with optional retained-substructure filtering."
            ),
        ):
            selected_engines.append("flowr_root")

        with st.expander(
            "PocketXMol — partial redesign controls",
            expanded="pocketxmol" in selected_engines,
        ):
            px_columns = st.columns(3)
            px_steps = px_columns[0].number_input(
                "Diffusion steps",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                key="ligand_redesign_pxm_steps",
            )
            px_strength = px_columns[1].slider(
                "Redesign strength",
                min_value=0.05,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="ligand_redesign_pxm_strength",
                help=(
                    "Lower values retain more of the input state; higher values "
                    "permit broader atom, bond, and coordinate changes."
                ),
            )
            px_size_std = px_columns[2].number_input(
                "Size-prior SD",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key="ligand_redesign_pxm_size_std",
            )
            engine_settings["pocketxmol"] = {
                "diffusion_steps": int(px_steps),
                "optimization_strength": float(px_strength),
                "ligand_atoms_mean": float(
                    len(preserve_indices) + len(redesign_indices)
                ),
                "ligand_atoms_std": float(px_size_std),
            }
        with st.expander(
            "FLOWR.root — local inpainting controls",
            expanded="flowr_root" in selected_engines,
        ):
            flow_columns = st.columns(3)
            flow_steps = flow_columns[0].number_input(
                "Integration steps",
                min_value=20,
                max_value=500,
                value=100,
                step=10,
                key="ligand_redesign_flow_steps",
            )
            flow_solver = flow_columns[1].selectbox(
                "Solver",
                ("euler", "midpoint"),
                key="ligand_redesign_flow_solver",
            )
            flow_corrector = flow_columns[2].number_input(
                "Corrector iterations",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key="ligand_redesign_flow_corrector",
            )
            strict_substructure = st.checkbox(
                "Require retained-substructure match",
                value=True,
                key="ligand_redesign_flow_filter",
                help=(
                    "FLOWR.root can slightly change nominally retained atoms. "
                    "This native RDKit filter removes outputs that no longer match "
                    "the conditioning substructure."
                ),
            )
            engine_settings["flowr_root"] = {
                "integration_steps": int(flow_steps),
                "solver": flow_solver,
                "corrector_steps": int(flow_corrector),
                "filter_conditioned_substructure": bool(strict_substructure),
            }

    with run_tab:
        render_run_resources(
            requires_gpu=bool(selected_engines),
            selected_gpu="Automatic",
            key="ligand_redesign",
        )
        run_columns = st.columns(4)
        campaign_name = run_columns[0].text_input(
            "Campaign name",
            value="Local ligand redesign",
            key="ligand_redesign_campaign_name",
        )
        requested_count = run_columns[1].number_input(
            "Attempts per engine",
            min_value=1,
            max_value=100000,
            value=100,
            step=10,
            key="ligand_redesign_count",
        )
        batch_size = run_columns[2].number_input(
            "Batch size",
            min_value=1,
            max_value=4096,
            value=32,
            step=1,
            key="ligand_redesign_batch",
        )
        seed = run_columns[3].number_input(
            "Seed",
            min_value=0,
            max_value=2147483647,
            value=2026,
            step=1,
            key="ligand_redesign_seed",
        )
        st.markdown("#### Campaign objectives")
        objective_columns = st.columns(4)
        optimize_affinity = objective_columns[0].checkbox(
            "Affinity / interaction fit",
            value=True,
            key="ligand_redesign_objective_affinity",
        )
        optimize_admet = objective_columns[1].checkbox(
            "ADMET",
            value=False,
            key="ligand_redesign_objective_admet",
        )
        optimize_novelty = objective_columns[2].checkbox(
            "Novelty",
            value=True,
            key="ligand_redesign_objective_novelty",
        )
        optimize_diversity = objective_columns[3].checkbox(
            "Diversity",
            value=True,
            key="ligand_redesign_objective_diversity",
        )
        st.caption(
            "PocketXMol and FLOWR.root consume the pocket and atom mask. These "
            "campaign objectives are recorded for transparent downstream "
            "qualification, ranking, docking, and property evaluation unless a "
            "native control above explicitly implements them."
        )
        blockers: list[str] = []
        if target is None or reference is None:
            blockers.append("Select a prepared complex with its bound ligand.")
        if pocket is None:
            blockers.append("Create/select a bound-ligand coordinate pocket.")
        if not redesign_indices:
            blockers.append("Select at least one ligand atom to replace.")
        if not preserve_indices:
            blockers.append("Retain at least one ligand atom as scaffold context.")
        if not anchor_indices:
            blockers.append(
                "The replacement region must share at least one bond with the retained scaffold."
            )
        if not selected_engines:
            blockers.append("Select PocketXMol and/or FLOWR.root.")
        ref_root = reference_root()
        for engine_id in selected_engines:
            spec = GENERATOR_BY_ID[engine_id]
            missing = [
                value
                for value in spec.reference_paths
                if not (ref_root / value).is_file()
            ]
            if missing:
                blockers.append(
                    f"{spec.name} is missing reference files: {', '.join(missing)}."
                )
            engine_settings.setdefault(engine_id, {}).update(
                {
                    "redesign_mode": "replace_selected_atoms",
                    "preserve_atom_indices": preserve_indices,
                    "redesign_atom_indices": redesign_indices,
                    "anchor_atom_indices": anchor_indices,
                    "requested_count": int(requested_count),
                    "batch_size": int(batch_size),
                    "seed": int(seed),
                }
            )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Engine": GENERATOR_BY_ID[value].name,
                        "Native operation": (
                            "MaskFill partial redesign"
                            if value == "pocketxmol"
                            else "Substructure inpainting"
                        ),
                        "Retained atoms": len(preserve_indices),
                        "Replaced atoms": len(redesign_indices),
                        "Anchors": ", ".join(map(str, anchor_indices)),
                        "Target-contact handling": (
                            "Post-pose docking/cofolding + PLIP/PandaMap review"
                            if contacts
                            else "No target contact selected"
                        ),
                    }
                    for value in selected_engines
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        for blocker in dict.fromkeys(blockers):
            st.info(blocker)
        acknowledge = st.checkbox(
            "I understand atom masks constrain generation, while the intended "
            "protein contact must be verified after docking or cofolding",
            value=False,
            key="ligand_redesign_acknowledge",
        )
        if not acknowledge:
            blockers.append("Acknowledge downstream pose/contact validation.")
        if st.button(
            "Queue ligand-redesign campaign",
            type="primary",
            disabled=bool(blockers),
            key="ligand_redesign_queue",
        ) and target is not None and reference is not None:
            modes = {
                value: "Partial ligand redesign / atom-mask inpainting"
                for value in selected_engines
            }
            objectives = {
                "affinity_interaction_fit": bool(optimize_affinity),
                "admet": bool(optimize_admet),
                "novelty": bool(optimize_novelty),
                "diversity": bool(optimize_diversity),
                "local_redesign": True,
            }
            try:
                campaign = create_generation_campaign_job(
                    name=campaign_name,
                    engine_ids=selected_engines,
                    target_artifact=target.artifact,
                    pocket_artifact=pocket.artifact if pocket else None,
                    reference_artifact=reference.artifact,
                    pharmacophore_artifact=(
                        hypothesis.artifact if hypothesis else None
                    ),
                    engine_modes=modes,
                    objectives=objectives,
                    requested_count=int(requested_count),
                    batch_size=int(batch_size),
                    seed=int(seed),
                    engine_settings=engine_settings,
                )
                children = []
                for engine_id in selected_engines:
                    children.append(
                        queue_generation_job(
                            campaign,
                            engine_id=engine_id,
                            target_artifact=target.artifact,
                            pocket_artifact=pocket.artifact if pocket else None,
                            reference_artifact=reference.artifact,
                            pharmacophore_artifact=(
                                hypothesis.artifact if hypothesis else None
                            ),
                            engine_mode=modes[engine_id],
                            objectives=objectives,
                            requested_count=int(requested_count),
                            batch_size=int(batch_size),
                            seed=int(seed),
                            engine_settings=engine_settings[engine_id],
                        )
                    )
                st.success(
                    "Created immutable redesign campaign "
                    f"{display_job_code(campaign.metadata.get('job_code'), campaign.run_id)} "
                    f"with {len(children)} queued engine job(s)."
                )
            except Exception as exc:
                st.error(f"Could not queue ligand redesign: {exc}")

    with results_tab:
        rows = []
        for job in iter_job_records(
            runs_root(), task_groups=("molecule-generation",)
        ):
            try:
                payload = json.loads((job.run_dir / "input.json").read_text())
            except (OSError, TypeError, ValueError):
                continue
            settings = payload.get("engine_settings")
            if not isinstance(settings, dict) or not settings.get("redesign_mode"):
                continue
            code = display_job_code(
                job.metadata.get("job_code"), job.run_id
            )
            rows.append(
                {
                    "Job": code,
                    "Engine": job.metadata.get("engine_id") or job.tool,
                    "Status": job.status,
                    "Retained atoms": len(
                        settings.get("preserve_atom_indices") or ()
                    ),
                    "Replaced atoms": len(
                        settings.get("redesign_atom_indices") or ()
                    ),
                    "Result": "./job-results?"
                    + urlencode(
                        {
                            "task_group": "molecule-generation",
                            "run_id": job.run_id,
                            "label": code,
                        }
                    ),
                }
            )
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Result": st.column_config.LinkColumn(
                        "Result", display_text="Open"
                    )
                },
            )
        else:
            st.info("No atom-masked ligand-redesign jobs are available yet.")


if __name__ == "__main__":
    render()
