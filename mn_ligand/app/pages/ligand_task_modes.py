from __future__ import annotations

import json
from dataclasses import dataclass
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
    select_artifact,
    select_target_artifact,
)
from mn_ligand.app.pages.ligand_redesign import (
    _bound_ligand_choice,
    _bound_ligand_identity,
    _bound_ligand_pocket_choice,
    _protein_only_pdb,
    _pseudo_atom_residue_pdb,
    _reference_molecule,
    _render_atom_selector,
)
from mn_ligand.app.pages.run_resources import render_run_resources
from mn_ligand.core.jobs import display_job_code, iter_job_records
from mn_ligand.runtime import reference_root, runs_root
from mn_ligand.workflows.generative_design import (
    GENERATOR_BY_ID,
    create_generation_campaign_job,
    queue_generation_job,
)


@dataclass(frozen=True)
class LigandTaskMode:
    mode: str
    title: str
    caption: str
    region_tab: str
    engines: tuple[str, ...]
    native_operations: dict[str, str]


FRAGMENT_GROWING = LigandTaskMode(
    mode="fragment_growing",
    title="Fragment Growing",
    caption=(
        "Select a connected, coordinate-bearing fragment from the bound ligand "
        "and grow new atoms from it inside the linked protein pocket."
    ),
    region_tab="Starting Fragment",
    engines=("pocketxmol", "flowr_root"),
    native_operations={
        "pocketxmol": "MaskFill growing from a fixed fragment pose",
        "flowr_root": "Native fragment growing",
    },
)

SCAFFOLD_HOPPING = LigandTaskMode(
    mode="scaffold_hopping",
    title="Scaffold Hopping",
    caption=(
        "Replace the automatically detected molecular core while retaining "
        "peripheral functionality and the bound-ligand pocket frame."
    ),
    region_tab="Scaffold",
    engines=("flowr_root",),
    native_operations={
        "flowr_root": "Native Bemis–Murcko scaffold hopping",
    },
)

PARTIAL_OPTIMIZATION = LigandTaskMode(
    mode="partial_optimization",
    title="Ligand Optimization",
    caption=(
        "Optimize the complete bound ligand in its pocket while controlling "
        "how far generation may move from the reference molecular state."
    ),
    region_tab="Reference Ligand",
    engines=("pocketxmol",),
    native_operations={
        "pocketxmol": "PocketXMol full-molecule optimization",
    },
)


def _connected_selection(molecule, indices: list[int]) -> bool:
    selected = set(indices)
    if not selected:
        return False
    visited = {next(iter(selected))}
    pending = list(visited)
    while pending:
        atom = molecule.GetAtomWithIdx(pending.pop())
        for neighbor in atom.GetNeighbors():
            index = neighbor.GetIdx()
            if index in selected and index not in visited:
                visited.add(index)
                pending.append(index)
    return visited == selected


def _scaffold_indices(molecule) -> list[int]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    annotated = Chem.Mol(molecule)
    for atom in annotated.GetAtoms():
        atom.SetIntProp("_source_index", atom.GetIdx())
    scaffold = MurckoScaffold.GetScaffoldForMol(annotated)
    return sorted(
        atom.GetIntProp("_source_index")
        for atom in scaffold.GetAtoms()
        if atom.HasProp("_source_index")
    )


def _render_static_partition(
    molecule,
    *,
    selected: list[int],
    selected_label: str,
    pocket_path: Path | None,
    key: str,
) -> None:
    selected_set = set(selected)
    retained = sorted(set(range(molecule.GetNumAtoms())) - selected_set)
    chains = [
        ChainVisualization(
            chain_id="L",
            color="uniform",
            color_params={"value": "0xf97316"},
            representation_type="ball-and-stick",
            residues=[value + 1 for value in selected],
            label=selected_label,
        )
    ]
    if retained:
        chains.append(
            ChainVisualization(
                chain_id="L",
                color="uniform",
                color_params={"value": "0x06b6d4"},
                representation_type="ball-and-stick",
                residues=[value + 1 for value in retained],
                label="Peripheral functionality",
            )
        )
    structures = [
        StructureVisualization(
            pdb=_pseudo_atom_residue_pdb(molecule),
            color="uniform",
            color_params={"value": "0x06b6d4"},
            representation_type="ball-and-stick",
            chains=chains,
        )
    ]
    if pocket_path is not None:
        structures.append(
            StructureVisualization(
                pdb=_protein_only_pdb(pocket_path),
                color="uniform",
                color_params={"value": "0x9ca3af"},
                representation_type="cartoon",
            )
        )
    molstar_custom_component(
        structures=structures,
        key=key,
        height=650,
        width="100%",
        show_controls=True,
        selection_mode=False,
    )


def _target_inputs(
    mode: LigandTaskMode,
) -> tuple[
    ArtifactChoice | None,
    ArtifactChoice | None,
    ArtifactChoice | None,
    ArtifactChoice | None,
]:
    prefix = mode.mode
    target = select_target_artifact(
        "Prepared protein–ligand complex",
        ("prepared_complex", "prepared_target", "prepared_receptor"),
        key=f"{prefix}_target",
    )
    reference = _bound_ligand_choice(target)
    pocket = _bound_ligand_pocket_choice(target)
    hypothesis = select_artifact(
        "Pharmacophore hypothesis / downstream pose-review target (optional)",
        ("pharmacophore_hypothesis",),
        key=f"{prefix}_pharmacophore",
        required=False,
    )
    if target is not None:
        st.metric(
            "Selected target",
            display_job_code(target.job.metadata.get("job_code"), target.job.run_id),
        )
    if reference is not None:
        st.success(
            f"Reference ligand: {_bound_ligand_identity(target)} from this "
            "exact prepared complex."
        )
    else:
        st.warning("The selected target has no coordinate-bearing bound ligand.")
    if pocket is not None:
        st.info(
            "Pocket context: immutable bound-ligand extraction "
            f"{display_job_code(pocket.job.metadata.get('job_code'), pocket.job.run_id)}."
        )
    else:
        st.warning(
            "Create a bound-ligand pocket on Pocket Detection and Extraction "
            "before launching this task."
        )
    return target, reference, pocket, hypothesis


def _engine_controls(
    mode: LigandTaskMode,
    *,
    fragment_size: int,
) -> tuple[list[str], dict[str, dict[str, object]]]:
    selected: list[str] = []
    settings: dict[str, dict[str, object]] = {}
    columns = st.columns(len(mode.engines))
    for column, engine_id in zip(columns, mode.engines, strict=True):
        if column.checkbox(
            GENERATOR_BY_ID[engine_id].name,
            value=True,
            key=f"{mode.mode}_engine_{engine_id}",
            help=mode.native_operations[engine_id],
        ):
            selected.append(engine_id)

    if "pocketxmol" in mode.engines:
        with st.expander(
            "PocketXMol controls",
            expanded="pocketxmol" in selected,
        ):
            values = st.columns(3)
            steps = values[0].number_input(
                "Diffusion steps",
                min_value=10,
                max_value=500,
                value=100 if mode.mode == "fragment_growing" else 50,
                step=10,
                key=f"{mode.mode}_pxm_steps",
            )
            strength = values[1].slider(
                "Optimization strength",
                0.05,
                1.0,
                0.35,
                0.05,
                key=f"{mode.mode}_pxm_strength",
                help=(
                    "For ligand optimization, lower values stay closer to the "
                    "reference and higher values permit broader changes."
                ),
            )
            size_sd = values[2].number_input(
                "Output-size SD",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key=f"{mode.mode}_pxm_size_sd",
            )
            settings["pocketxmol"] = {
                "diffusion_steps": int(steps),
                "optimization_strength": float(strength),
                "ligand_atoms_std": float(size_sd),
            }
    if "flowr_root" in mode.engines:
        with st.expander(
            "FLOWR.root controls",
            expanded="flowr_root" in selected,
        ):
            values = st.columns(3)
            steps = values[0].number_input(
                "Integration steps",
                min_value=20,
                max_value=500,
                value=100,
                step=10,
                key=f"{mode.mode}_flow_steps",
            )
            solver = values[1].selectbox(
                "Solver",
                ("euler", "midpoint"),
                key=f"{mode.mode}_flow_solver",
            )
            correctors = values[2].number_input(
                "Corrector iterations",
                min_value=0,
                max_value=10,
                value=0,
                key=f"{mode.mode}_flow_correctors",
            )
            settings["flowr_root"] = {
                "integration_steps": int(steps),
                "solver": solver,
                "corrector_steps": int(correctors),
                "filter_conditioned_substructure": True,
            }
    if mode.mode == "fragment_growing":
        grow_size = st.number_input(
            "Heavy atoms to add",
            min_value=1,
            max_value=60,
            value=10,
            step=1,
            key="fragment_growing_size",
            help=(
                "Target number of new heavy atoms. PocketXMol treats this as "
                "the output-size mean; FLOWR.root passes it as native grow_size."
            ),
        )
        for engine_id in mode.engines:
            settings.setdefault(engine_id, {})["grow_size"] = int(grow_size)
            settings[engine_id]["ligand_atoms_mean"] = float(
                fragment_size + int(grow_size)
            )
    return selected, settings


def _results(mode: LigandTaskMode) -> None:
    rows = []
    for job in iter_job_records(
        runs_root(), task_groups=("molecule-generation",)
    ):
        try:
            payload = json.loads((job.run_dir / "input.json").read_text())
        except (OSError, TypeError, ValueError):
            continue
        settings = payload.get("engine_settings")
        if not isinstance(settings, dict):
            continue
        if settings.get("redesign_mode") != mode.mode:
            continue
        code = display_job_code(job.metadata.get("job_code"), job.run_id)
        rows.append(
            {
                "Job": code,
                "Engine": payload.get("engine"),
                "Status": job.status,
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
        st.info(f"No {mode.title.lower()} jobs are available yet.")


def render_ligand_task(mode: LigandTaskMode) -> None:
    st.title(mode.title)
    st.caption(mode.caption)
    target_tab, region_tab, engines_tab, run_tab, results_tab = st.tabs(
        ["Target", mode.region_tab, "Engines", "Run", "Results"]
    )
    with target_tab:
        target, reference, pocket, hypothesis = _target_inputs(mode)

    molecule = None
    if reference is not None:
        try:
            molecule, _ = _reference_molecule(reference)
        except Exception as exc:
            st.error(f"Could not load the bound reference ligand: {exc}")
    pocket_path = (
        pocket.artifact.resolve(pocket.job.run_dir, must_exist=True)
        if pocket is not None
        else None
    )
    preserve_indices: list[int] = []
    display_indices: list[int] = []
    with region_tab:
        if molecule is None:
            st.info("Select a prepared complex with a bound ligand first.")
        elif mode.mode == "fragment_growing":
            _, display_indices, _ = _render_atom_selector(
                molecule,
                pocket_path=pocket_path,
                receptor_path=None,
                target_label="No native pharmacophore enforcement",
                target_coordinates=None,
                key_prefix=mode.mode,
                selection_heading="Select the starting fragment",
                selection_name="Keep",
                selected_description="fixed starting fragment",
            )
            preserve_indices = display_indices
            metrics = st.columns(3)
            metrics[0].metric("Reference heavy atoms", molecule.GetNumAtoms())
            metrics[1].metric("Fragment atoms", len(preserve_indices))
            metrics[2].metric(
                "Connected",
                "Yes" if _connected_selection(molecule, preserve_indices) else "No",
            )
        elif mode.mode == "scaffold_hopping":
            display_indices = _scaffold_indices(molecule)
            preserve_indices = sorted(
                set(range(molecule.GetNumAtoms())) - set(display_indices)
            )
            _render_static_partition(
                molecule,
                selected=display_indices,
                selected_label="Automatically detected scaffold to replace",
                pocket_path=pocket_path,
                key="scaffold_hopping_viewer",
            )
            st.caption(
                "Orange is the RDKit Bemis–Murcko core that FLOWR.root will "
                "replace; cyan peripheral functionality is retained. The native "
                "engine independently applies its scaffold-hopping transform."
            )
            metrics = st.columns(3)
            metrics[0].metric("Heavy atoms", molecule.GetNumAtoms())
            metrics[1].metric("Detected core", len(display_indices))
            metrics[2].metric("Peripheral atoms", len(preserve_indices))
        else:
            _render_static_partition(
                molecule,
                selected=list(range(molecule.GetNumAtoms())),
                selected_label="Ligand state to optimize",
                pocket_path=pocket_path,
                key="partial_optimization_viewer",
            )
            st.caption(
                "The full molecular graph and coordinates initialize "
                "PocketXMol optimization; optimization strength controls the "
                "departure from this state."
            )
            preserve_indices = list(range(molecule.GetNumAtoms()))

    with engines_tab:
        selected_engines, engine_settings = _engine_controls(
            mode,
            fragment_size=len(preserve_indices),
        )

    with run_tab:
        render_run_resources(
            requires_gpu=bool(selected_engines),
            selected_gpu="Automatic",
            key=mode.mode,
        )
        columns = st.columns(4)
        campaign_name = columns[0].text_input(
            "Campaign name",
            value=mode.title,
            key=f"{mode.mode}_campaign_name",
        )
        count = columns[1].number_input(
            "Attempts per engine",
            min_value=1,
            max_value=100000,
            value=100,
            step=10,
            key=f"{mode.mode}_count",
        )
        batch = columns[2].number_input(
            "Batch size",
            min_value=1,
            max_value=4096,
            value=32,
            key=f"{mode.mode}_batch",
        )
        seed = columns[3].number_input(
            "Seed",
            min_value=0,
            max_value=2147483647,
            value=2026,
            key=f"{mode.mode}_seed",
        )
        blockers: list[str] = []
        if target is None or reference is None:
            blockers.append("Select a prepared complex with its bound ligand.")
        if pocket is None:
            blockers.append("Create/select a bound-ligand coordinate pocket.")
        if not selected_engines:
            blockers.append("Select at least one native engine.")
        if mode.mode == "fragment_growing":
            if not preserve_indices:
                blockers.append("Select at least one starting-fragment atom.")
            elif molecule is not None and not _connected_selection(
                molecule, preserve_indices
            ):
                blockers.append("The selected starting fragment must be connected.")
        if mode.mode == "scaffold_hopping" and not display_indices:
            blockers.append("No Bemis–Murcko scaffold could be detected.")
        for engine_id in selected_engines:
            missing = [
                value
                for value in GENERATOR_BY_ID[engine_id].reference_paths
                if not (reference_root() / value).is_file()
            ]
            if missing:
                blockers.append(
                    f"{GENERATOR_BY_ID[engine_id].name} is missing: "
                    + ", ".join(missing)
                )
            engine_settings.setdefault(engine_id, {}).update(
                {
                    "redesign_mode": mode.mode,
                    "preserve_atom_indices": preserve_indices,
                    "redesign_atom_indices": display_indices,
                    "requested_count": int(count),
                    "batch_size": int(batch),
                    "seed": int(seed),
                }
            )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Engine": GENERATOR_BY_ID[value].name,
                        "Native operation": mode.native_operations[value],
                        "Reference atoms": (
                            len(preserve_indices)
                            if mode.mode == "fragment_growing"
                            else molecule.GetNumAtoms()
                            if molecule is not None
                            else 0
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
            "I understand generated molecules require chemical/3D qualification "
            "and downstream docking or cofolding validation",
            key=f"{mode.mode}_acknowledge",
        )
        if not acknowledge:
            blockers.append("Acknowledge downstream validation.")
        if st.button(
            f"Queue {mode.title.lower()} campaign",
            type="primary",
            disabled=bool(blockers),
            key=f"{mode.mode}_queue",
        ) and target is not None and reference is not None:
            engine_modes = {
                value: mode.native_operations[value]
                for value in selected_engines
            }
            objectives = {
                mode.mode: True,
                "affinity_interaction_fit": True,
                "chemical_3d_qualification": True,
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
                    engine_modes=engine_modes,
                    objectives=objectives,
                    requested_count=int(count),
                    batch_size=int(batch),
                    seed=int(seed),
                    engine_settings=engine_settings,
                )
                children = [
                    queue_generation_job(
                        campaign,
                        engine_id=engine_id,
                        target_artifact=target.artifact,
                        pocket_artifact=pocket.artifact if pocket else None,
                        reference_artifact=reference.artifact,
                        pharmacophore_artifact=(
                            hypothesis.artifact if hypothesis else None
                        ),
                        engine_mode=engine_modes[engine_id],
                        objectives=objectives,
                        requested_count=int(count),
                        batch_size=int(batch),
                        seed=int(seed),
                        engine_settings=engine_settings[engine_id],
                    )
                    for engine_id in selected_engines
                ]
                st.success(
                    f"Created immutable campaign {display_job_code(campaign.metadata.get('job_code'), campaign.run_id)} "
                    f"with {len(children)} queued engine job(s)."
                )
            except Exception as exc:
                st.error(f"Could not queue {mode.title.lower()}: {exc}")

    with results_tab:
        _results(mode)
