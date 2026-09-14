from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from rdkit import Chem

from mn_ligand.app.pages.discover_inputs import ArtifactChoice
from mn_ligand.app.pages.docking_cofolding import (
    ALL_ENGINES,
    _queue_target_engine_jobs,
)
from mn_ligand.core.artifacts import ArtifactRef
from mn_ligand.core.jobs import JobRecord
from mn_ligand.core.provenance import COMPOUND_DATASET_CAMPAIGN_PURPOSE
from mn_ligand.runtime import runs_root
from mn_ligand.workflows.compound_preparation import (
    create_docking_parent_selection_job,
    parent_duplicate_report,
)
from mn_ligand.workflows.refolding import (
    configured_alphafold3_reference_paths,
    configured_nesso_reference_paths,
)


SOURCE_IMPORT_ID = "9ff15f3c-ea39-47d3-ac98-96cddfdc15b9"
SOURCE_CAMPAIGN_ID = "ac8dcc4f-5c92-4b19-bbc3-02269aec9051"
SOURCE_SCRUB_RUN_ID = "d99f04e9-a29a-47d1-89ec-fb1d0ef8d967"
MSA_RUN_ID = "e91a8cb8-1e82-406f-89d6-d0ee8401dfad"
CAMPAIGN_LABEL = (
    "THR_agonists_28parent_sharedScrub-pH7p4_"
    "16target_xaligned_35x20x20_3rep_20260803"
)


def _job(run_id: str) -> JobRecord:
    root = runs_root()
    matches = [path for group in root.iterdir() if (path := group / run_id).is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one run directory for {run_id}; found {matches}")
    return JobRecord.load(matches[0], task_group=matches[0].parent.name)


def _shared_modeling_parents() -> tuple[JobRecord, ArtifactRef, list[dict]]:
    source_job = _job(SOURCE_IMPORT_ID)
    source_artifact = source_job.artifact_manifest.by_type("compound_set")[0]
    source_path = source_artifact.resolve(source_job.run_dir, must_exist=True)
    rows = pd.read_csv(source_path).fillna("").to_dict("records")
    parents = parent_duplicate_report(rows)["docking_parent_rows"]
    scrub_dir = runs_root() / "docking" / SOURCE_SCRUB_RUN_ID / "prepared_ligands"
    charges: Counter[int] = Counter()
    for parent in parents:
        compound_id = str(parent["representative_compound_id"])
        sdf_path = scrub_dir / f"{compound_id}.sdf"
        molecules = [
            molecule
            for molecule in Chem.SDMolSupplier(str(sdf_path), removeHs=True)
            if molecule is not None
        ]
        if len(molecules) != 1:
            raise RuntimeError(
                f"{compound_id}: expected one Scrub modeling state, found {len(molecules)}"
            )
        molecule = molecules[0]
        parent["modeling_smiles"] = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
        parent["modeling_preparation"] = (
            f"Scrub pH 7.4, tautomers fixed; source run {SOURCE_SCRUB_RUN_ID}"
        )
        parent["modeling_ph"] = 7.4
        charges[int(Chem.GetFormalCharge(molecule))] += 1
    if len(parents) != 28:
        raise RuntimeError(f"Expected 28 unique parents, found {len(parents)}")
    phosphate = next(
        row for row in parents if row["representative_compound_id"] == "HY-19513"
    )
    if phosphate["modeling_smiles"].count("[O-]") != 2:
        raise RuntimeError(
            "HY-19513 did not resolve to the expected phosphate dianion: "
            + phosphate["modeling_smiles"]
        )
    print(
        json.dumps(
            {
                "unique_parents": len(parents),
                "formal_charge_counts": dict(sorted(charges.items())),
                "HY-19513_modeling_smiles": phosphate["modeling_smiles"],
            },
            indent=2,
        )
    )
    return source_job, source_artifact, parents


def _target_contexts() -> list[dict]:
    contexts: dict[str, dict] = {}
    for metadata_path in (runs_root() / "docking").glob("*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("launch_campaign_id") != SOURCE_CAMPAIGN_ID
            or metadata.get("engine") != "vina"
        ):
            continue
        payload = json.loads((metadata_path.parent / "input.json").read_text())
        target_artifact = ArtifactRef.from_dict(payload["target_artifact"])
        reference_artifact = ArtifactRef.from_dict(payload["reference_ligand_artifact"])
        target_job = _job(target_artifact.run_id)
        target_choice = ArtifactChoice(job=target_job, artifact=target_artifact)
        reference_path = reference_artifact.resolve(target_job.run_dir, must_exist=True)
        if reference_path is None:
            raise RuntimeError(f"Missing reference ligand for {target_artifact.run_id}")
        parameters = payload["parameters"]
        center = parameters["center"]
        contexts[target_artifact.run_id] = {
            "target": target_choice,
            "pocket": None,
            "associated_ligand": reference_path,
            "associated_ligand_artifact": reference_artifact,
            # These artifacts are already the x-axis-aligned target/ligand pair.
            "alignment_transform": None,
            "center": tuple(float(center[axis]) for axis in "xyz"),
            "size": (35.0, 20.0, 20.0),
            "box_mode": "fixed",
            "box_padding": None,
        }
    result = sorted(contexts.values(), key=lambda item: item["target"].job.run_id)
    if len(result) != 16:
        raise RuntimeError(f"Expected 16 aligned targets, found {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    source_job, source_artifact, parents = _shared_modeling_parents()
    contexts = _target_contexts()
    print(json.dumps({"aligned_targets": len(contexts), "label": CAMPAIGN_LABEL}, indent=2))
    if not args.launch:
        return
    for metadata_path in runs_root().glob("*/*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("launch_campaign_label") == CAMPAIGN_LABEL:
            raise RuntimeError(f"Campaign label already exists at {metadata_path.parent}")

    selection_job = create_docking_parent_selection_job(
        source_job=source_job,
        source_artifact=source_artifact,
        parent_rows=parents,
        selection_mode=(
            "All 28 unique parents; shared Scrub pH 7.4 modeling state; "
            "29 source records with HY-19513/HY-19513A collapsed by identity"
        ),
        box_size=(35.0, 20.0, 20.0),
    )
    selection_artifact = selection_job.artifact_manifest.by_type("compound_set")[0]
    selection_path = selection_artifact.resolve(selection_job.run_dir, must_exist=True)
    selected = (ArtifactChoice(job=selection_job, artifact=selection_artifact),)
    af3_db, af3_weights, af3_msa = configured_alphafold3_reference_paths()
    nesso_checkpoint, nesso_ccd, nesso_esm = configured_nesso_reference_paths()
    config = {
        "gpu_device": "all",
        "docking_gpu_device": "1",
        "structure_gpu_device": "0",
        "docking_mode": "classic",
        "search_mode": "detail",
        "exhaustiveness": 30,
        "poses": 10,
        "use_scrub": True,
        "scrub_ph": 7.4,
        "scrub_skip_tautomer": True,
        "campaign_replicates": 3,
        "campaign_seed": 1001,
        "maximum_compounds": 0,
        "extra_args_text": "",
        "openvs_protocol": "vsh",
        "openvs_reference_mode": "reference_guided",
        "openvs_workers": 1,
        "openvs_ph": 7.4,
        "openvs_conformers": 20,
        "openvs_steps": 2000,
        "openvs_padding": 4.0,
        "openvs_cluster": 2.0,
        "boltz_max": 0,
        "boltz_recycles": 3,
        "boltz_steps": 200,
        "boltz_samples": 5,
        "af3_db_dir": af3_db,
        "af3_weights_dir": af3_weights,
        "af3_msa_dir": af3_msa,
        "af3_max": 0,
        "af3_batch": 1,
        "af3_recycles": 10,
        "nesso_checkpoint": nesso_checkpoint,
        "nesso_ccd": nesso_ccd,
        "nesso_esm_cache": nesso_esm,
        "nesso_max": 0,
        "nesso_recycles": 5,
        "nesso_refine": 22,
        "nesso_tokens": 256,
        "nesso_affinity": 15,
    }
    campaign_id = str(uuid4())
    queued: list[str] = []
    failures: list[str] = []
    for context in contexts:
        target_queued, target_failures, _ = _queue_target_engine_jobs(
            context=context,
            engines=list(ALL_ENGINES),
            selected_compounds=selected,
            compound_paths=[selection_path],
            reference=None,
            target_ligand_only=False,
            launch_campaign_id=campaign_id,
            launch_campaign_label=CAMPAIGN_LABEL,
            campaign_purpose=COMPOUND_DATASET_CAMPAIGN_PURPOSE,
            config=config,
            msa_dependency_ids=(MSA_RUN_ID,),
        )
        queued.extend(target_queued)
        failures.extend(target_failures)
    print(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "selection_run_id": selection_job.run_id,
                "queued_child_jobs": len(queued),
                "failures": failures,
            },
            indent=2,
        )
    )
    if failures or len(queued) != 16 * len(ALL_ENGINES):
        raise RuntimeError(
            f"Incomplete launch: queued {len(queued)} jobs with failures {failures}"
        )


if __name__ == "__main__":
    main()
