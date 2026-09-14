import json
from pathlib import Path

from mn_ligand.app.pages.discover_inputs import (
    _matches_word_query,
    artifact_box,
    artifact_options,
    bound_ligand_box,
    hide_superseded_target_versions,
    target_coordinate_ligand_box,
    target_inventory,
    target_ligand_path,
)
from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest


PDB_COMPLEX = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 20.00           C
HETATM    3  C1  LIG B 101      20.000  21.000  22.000  1.00 20.00           C
HETATM    4  C2  LIG B 101      22.000  23.000  24.000  1.00 20.00           C
HETATM    5  O1  LIG B 101      21.000  22.000  23.000  1.00 20.00           O
END
"""

PDB_DOCKED_COMPLEX = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 20.00           C
ATOM      3  C1  UNL     1      20.000  21.000  22.000  1.00 20.00           C
ATOM      4  O1  UNL     1      22.000  23.000  24.000  1.00 20.00           O
END
"""


def test_target_filter_words_support_or_and_matching() -> None:
    values = ("4LNW thyroid hormone receptor alpha",)

    assert _matches_word_query("4lnw 2h79", values, require_all=False)
    assert not _matches_word_query("4lnw 2h79", values, require_all=True)
    assert _matches_word_query("thyroid receptor", values, require_all=True)
    assert _matches_word_query("4lnw,2h79", values, require_all=False)


def _write_job(run_dir: Path, metadata: dict, artifacts: list[tuple[Path, str]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps({"status": "completed", **metadata}))
    refs = [ArtifactRef.from_path(run_dir, path, artifact_type) for path, artifact_type in artifacts]
    write_artifact_manifest(run_dir, refs)


def test_target_inventory_combines_origin_preparation_and_source_complex(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    imported = runs / "protein-import" / "import-1"
    imported_path = imported / "artifacts" / "imported.pdb"
    imported_path.parent.mkdir(parents=True)
    imported_path.write_text(PDB_COMPLEX)
    _write_job(
        imported,
        {
            "run_id": "import-1",
            "job_code": "IMP01",
            "source": "pdb",
            "pdb_id": "1ABC",
            "receptor": {
                "title": "Example receptor structure",
                "experimental_method": "X-RAY DIFFRACTION",
                "resolution_angstrom": 1.5,
                "entities": [
                    {
                        "name": "Example receptor",
                        "source_organisms": ["Homo sapiens"],
                        "uniprot_ids": ["P12345"],
                    }
                ],
            },
        },
        [(imported_path, "imported_target")],
    )

    cleaned = runs / "protein-cleaning" / "clean-1"
    receptor = cleaned / "artifacts" / "prepared_target.pdb"
    receptor.parent.mkdir(parents=True)
    receptor.write_text("\n".join(line for line in PDB_COMPLEX.splitlines() if line.startswith("ATOM")) + "\n")
    _write_job(
        cleaned,
        {
            "run_id": "clean-1",
            "job_code": "CLN01",
            "job_type": "protein_cleaning",
            "source": "ligandx-pdbfixer",
            "import_run_id": "import-1",
            "pdb_id": "1ABC",
        },
        [(receptor, "prepared_target")],
    )
    (cleaned / "input.json").write_text(
        json.dumps(
            {
                "parameters": {
                    "clean_protein": True,
                    "map_modified_residues": False,
                    "ph": 7.4,
                    "noncanonical_replacements": [
                        {"key": "CAS|A|10|_", "target": "CYS"}
                    ],
                    "internal_gap_model_count": 10,
                    "internal_gap_definitions": [
                        {
                            "chain": "A",
                            "author_start": 20,
                            "author_end": 21,
                            "sequence": "GG",
                        }
                    ],
                }
            }
        )
    )
    report = cleaned / "artifacts" / "reports" / "repair_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "pdbfixer": {
                    "missing_atoms": [],
                    "refinement": {
                        "engine": "OpenMM LocalEnergyMinimizer",
                        "forcefield": ["amber14-all.xml"],
                        "ligand_context": True,
                        "potential_energy_before_kj_mol": 100.0,
                        "potential_energy_after_kj_mol": 50.0,
                    },
                }
            }
        )
    )

    entries = target_inventory(("prepared_target",))

    assert len(entries) == 1
    assert entries[0].viewer_path == imported_path
    assert entries[0].row["Origin"] == (
        "PDB → MODELLER residue repair → MODELLER gap modeling → "
        "PDBFixer cleaning → OpenMM minimization"
    )
    assert list(entries[0].row)[:14] == [
        "Job",
        "Docking / cofolding",
        "Redocking / refolding",
        "Target",
        "Receptor",
        "Ligands",
        "Tool",
        "Origin",
        "Last step",
        "Residues",
        "Organism",
        "UniProt",
        "Kind",
        "Compound",
    ]
    assert entries[0].row["Job"].endswith(
        "task_group=protein-cleaning&run_id=clean-1&label=CLN01"
    )
    assert entries[0].row["Last step"] == "OpenMM minimization"
    assert entries[0].row["Ligands"] == "LIG"
    assert entries[0].row["Chains"] == "A"
    assert entries[0].row["Receptor"] == "Example receptor"
    assert entries[0].row["Organism"] == "Homo sapiens"
    assert entries[0].row["UniProt"] == "P12345"
    assert entries[0].row["Resolution (A)"] == 1.5
    assert "Cleaned/repaired" in entries[0].row["Preparation"]


def test_target_inventory_orders_created_oldest_first(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    for directory, run_id, created_at in (
        ("a-newer", "newer", "2026-07-30T12:00:00+00:00"),
        ("z-older", "older", "2026-07-29T12:00:00+00:00"),
    ):
        run_dir = runs / "target-trimming" / directory
        target_path = run_dir / "artifacts" / "target.pdb"
        target_path.parent.mkdir(parents=True)
        target_path.write_text(PDB_COMPLEX)
        _write_job(
            run_dir,
            {
                "run_id": run_id,
                "workflow": "target_trimming",
                "pdb_id": run_id.upper(),
                "created_at": created_at,
            },
            [(target_path, "prepared_target")],
        )

    entries = target_inventory(("prepared_target",))

    assert [entry.choice.job.run_id for entry in entries] == [
        "older",
        "newer",
    ]


def test_target_inventory_recovers_full_multistep_lineage(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    source = runs / "structure-jobs" / "source"
    source_path = source / "source.pdb"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(PDB_COMPLEX)
    _write_job(
        source,
        {
            "run_id": "source",
            "job_code": "SRC01",
            "source": "pdb",
            "pdb_id": "1ABC",
            "ligand_key": "TST|B|101|_",
            "ligands": [
                {
                    "ccd_id": "TST",
                    "name": "Test ligand",
                    "formula": "C2H6O",
                    "molecular_weight": 46.07,
                }
            ],
            "receptor": {
                "title": "Lineage receptor",
                "experimental_method": "X-RAY DIFFRACTION",
                "resolution_angstrom": 1.25,
                "entities": [
                    {
                        "name": "Lineage receptor",
                        "source_organisms": ["Homo sapiens"],
                        "uniprot_ids": ["P12345"],
                    }
                ],
            },
        },
        [(source_path, "prepared_complex")],
    )
    trimmed = runs / "target-trimming" / "trimmed"
    _write_job(
        trimmed,
        {
            "run_id": "trimmed",
            "job_code": "TRM01",
            "job_type": "target_trimming",
            "parent_run_id": "source",
            "trim_ranges": {"A": {"start": 1, "end": 1}},
        },
        [],
    )
    repaired = runs / "terminal-repair" / "repaired"
    repaired_path = repaired / "complex_repaired.pdb"
    repaired_path.parent.mkdir(parents=True)
    repaired_path.write_text(PDB_COMPLEX)
    _write_job(
        repaired,
        {
            "run_id": "repaired",
            "job_code": "RPR01",
            "job_type": "terminal_repair",
            "parent_run_id": "trimmed",
            "tool": "MODELLER",
            "chain": "A",
            "extension_sequence": "GG",
        },
        [(repaired_path, "prepared_complex")],
    )

    entries = target_inventory(("prepared_complex",))
    entry = next(
        item
        for item in entries
        if item.choice.job.run_id == "repaired"
    )

    assert entry.row["Target"] == "1ABC"
    assert entry.row["Receptor"] == "Lineage receptor"
    assert entry.row["Organism"] == "Homo sapiens"
    assert entry.row["UniProt"] == "P12345"
    assert entry.row["Ligands"] == "TST"
    assert entry.row["Compound"] == "Test ligand"
    assert entry.row["Formula"] == "C2H6O"
    assert entry.row["MW (Da)"] == 46.07
    assert entry.row["Resolution (A)"] == 1.25
    assert entry.row["Origin"] == "PDB → Target trimming → MODELLER repair"
    assert entry.row["Last step"] == "MODELLER repair"
    assert "Trimmed protein termini A:1-1" in entry.row["Preparation"]
    assert "Extended chain A C-terminus by 2 aa (GG)" in entry.row["Preparation"]
    visible_entries = hide_superseded_target_versions(entries)
    assert {item.choice.job.run_id for item in visible_entries} == {
        "source",
        "repaired",
    }


def test_target_inventory_hides_internal_cleaning_after_structure_publication(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    cleaned = runs / "protein-cleaning" / "cleaning-attempt"
    cleaned_path = cleaned / "artifacts" / "prepared_target.pdb"
    cleaned_path.parent.mkdir(parents=True)
    cleaned_path.write_text(PDB_COMPLEX)
    _write_job(
        cleaned,
        {
            "run_id": "cleaning-attempt",
            "job_type": "protein_cleaning",
            "pdb_id": "3GWS",
        },
        [(cleaned_path, "prepared_target")],
    )
    published = runs / "structure-jobs" / "published-target"
    published_path = published / "artifacts" / "prepared_receptor.pdb"
    published_path.parent.mkdir(parents=True)
    published_path.write_text(PDB_COMPLEX)
    _write_job(
        published,
        {
            "run_id": "published-target",
            "job_type": "structure",
            "pdb_id": "3GWS",
        },
        [(published_path, "prepared_receptor")],
    )

    entries = target_inventory(("prepared_target", "prepared_receptor"))

    assert {entry.choice.job.run_id for entry in entries} == {
        "cleaning-attempt",
        "published-target",
    }
    assert [
        entry.choice.job.run_id
        for entry in hide_superseded_target_versions(entries)
    ] == ["published-target"]


def test_target_inventory_hides_transient_workflow_receptors(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    jobs = (
        (
            "structure-jobs",
            "published",
            {"run_id": "published", "job_code": "PUB01"},
        ),
        (
            "rescoring",
            "pose-selection",
            {
                "run_id": "pose-selection",
                "job_code": "POSE1",
                "workflow": "pose_selection",
                "operation": "rescoring_selection",
            },
        ),
        (
            "target-orientation",
            "orientation",
            {
                "run_id": "orientation",
                "job_code": "AXIS1",
                "workflow": "target_orientation",
                "operation": "preparation",
            },
        ),
        (
            "docking",
            "docking-copy",
            {
                "run_id": "docking-copy",
                "job_code": "DCK01",
                "workflow": "docking_campaign",
                "operation": "docking",
            },
        ),
    )
    for task_group, run_id, metadata in jobs:
        run_dir = runs / task_group / run_id
        target = run_dir / "artifacts" / "receptor.pdb"
        target.parent.mkdir(parents=True)
        target.write_text(PDB_COMPLEX)
        _write_job(
            run_dir,
            metadata,
            [(target, "prepared_target")],
        )

    entries = target_inventory(("prepared_target",))
    transient_entries = target_inventory(
        ("prepared_target",),
        include_transient_targets=True,
    )

    assert [entry.choice.job.run_id for entry in entries] == ["published"]
    assert {entry.choice.job.run_id for entry in transient_entries} == {
        "published",
        "pose-selection",
        "orientation",
        "docking-copy",
    }


def test_target_inventory_links_multi_engine_descendants_as_one_campaign(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    target_dir = runs / "target-trimming" / "target-1"
    target_path = target_dir / "artifacts" / "target.pdb"
    target_path.parent.mkdir(parents=True)
    target_path.write_text(PDB_COMPLEX)
    _write_job(
        target_dir,
        {
            "run_id": "target-1",
            "job_code": "TGT01",
            "workflow": "target_trimming",
        },
        [(target_path, "prepared_target")],
    )
    orientation_dir = runs / "target-orientation" / "orientation-1"
    orientation_path = orientation_dir / "artifacts" / "oriented.pdb"
    orientation_path.parent.mkdir(parents=True)
    orientation_path.write_text(PDB_COMPLEX)
    _write_job(
        orientation_dir,
        {
            "run_id": "orientation-1",
            "job_code": "AXIS1",
            "workflow": "target_orientation",
            "source_target_run_id": "target-1",
            "parent_run_id": "target-1",
        },
        [(orientation_path, "prepared_target")],
    )
    for index, (workflow, tool) in enumerate(
        (
            ("docking_campaign", "GNINA"),
            ("boltz2_refolding", "Boltz-2"),
        ),
        start=1,
    ):
        run_dir = runs / (
            "docking" if workflow == "docking_campaign" else "refolding"
        ) / f"result-{index}"
        _write_job(
            run_dir,
            {
                "run_id": f"result-{index}",
                "job_code": f"RES0{index}",
                "workflow": workflow,
                "operation": (
                    "docking"
                    if workflow == "docking_campaign"
                    else "refolding"
                ),
                "tool": tool,
                "parent_run_id": "orientation-1",
                "launch_campaign_id": "campaign-1",
                "launch_campaign_label": "Named campaign",
                "campaign_purpose": "target_ligand_redocking_refolding",
                "created_at": f"2026-07-30T00:00:0{index}+00:00",
            },
            [],
        )
    imported_campaign_dir = runs / "docking" / "imported-campaign"
    _write_job(
        imported_campaign_dir,
        {
            "run_id": "imported-campaign",
            "job_code": "LIB01",
            "workflow": "docking_campaign",
            "operation": "docking",
            "tool": "GNINA",
            "parent_run_id": "orientation-1",
            "launch_campaign_id": "library-campaign",
            "launch_campaign_label": "Imported library",
            "created_at": "2026-07-30T00:10:00+00:00",
        },
        [],
    )

    entry = target_inventory(("prepared_target",))[0]

    assert entry.choice.job.run_id == "target-1"
    assert entry.row["Redocking / refolding"].endswith(
        "target_run_id=target-1"
        "&campaign_purpose=target_ligand_redocking_refolding&label=Campaigns"
    )
    assert entry.row["Docking / cofolding"].endswith(
        "target_run_id=target-1"
        "&campaign_purpose=compound_dataset_docking_cofolding&label=Campaigns"
    )


def test_target_inventory_reports_pdb_root_docking_and_ligand_preparation(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    source = runs / "structure-jobs" / "pdb-source"
    source_complex = source / "source_complex_refined.pdb"
    source_complex.parent.mkdir(parents=True)
    source_complex.write_text(PDB_COMPLEX)
    _write_job(
        source,
        {
            "run_id": "pdb-source",
            "source": "pdb",
            "pdb_id": "1ABC",
        },
        [(source_complex, "prepared_complex")],
    )
    docked = runs / "structure-jobs" / "gnina-result"
    docked_complex = docked / "docked_complex_refined.pdb"
    docked_complex.parent.mkdir(parents=True)
    docked_complex.write_text(PDB_COMPLEX)
    _write_job(
        docked,
        {
            "run_id": "gnina-result",
            "source": "gnina",
            "engine": "gnina",
            "source_structure_run_id": "pdb-source",
            "use_scrub": True,
            "scrub_ph": 7.4,
        },
        [(docked_complex, "prepared_complex")],
    )

    entry = next(
        item
        for item in target_inventory(("prepared_complex",))
        if item.choice.job.run_id == "gnina-result"
    )

    assert entry.row["Origin"] == "PDB → GNINA docking → Ligand preparation"
    assert "Generated the protein-ligand pose with GNINA" in entry.row["Preparation"]
    assert "Prepared ligand protonation/tautomer" in entry.row["Preparation"]


def test_bound_ligand_and_pocket_boxes_are_typed(tmp_path: Path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    run_dir = runs / "structure-jobs" / "complex-1"
    complex_path = run_dir / "complex.pdb"
    complex_path.parent.mkdir(parents=True)
    complex_path.write_text(PDB_COMPLEX)
    _write_job(
        run_dir,
        {"run_id": "complex-1", "job_code": "CMP01", "pdb_id": "1ABC"},
        [(complex_path, "prepared_complex")],
    )
    choice = target_inventory(("prepared_complex",))[0].choice

    ligand_box = bound_ligand_box(choice, "LIG|B|101|_", padding_angstrom=4.0)
    assert ligand_box == {"center": (21.0, 22.0, 23.0), "size": (10.0, 10.0, 10.0)}

    pocket_ref = ArtifactRef(
        run_id="pocket-1",
        artifact_type="pocket",
        path="pocket.pdb",
        metadata={"center_angstrom": [1, 2, 3], "size_angstrom": [20, 21, 22]},
    )
    pocket_choice = type(choice)(job=choice.job, artifact=pocket_ref)
    assert artifact_box(pocket_choice) == {"center": (1.0, 2.0, 3.0), "size": (20.0, 21.0, 22.0)}


def test_pose_set_beside_receptor_initializes_ligand_box(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    run_dir = runs / "rescoring" / "pose-target"
    receptor = run_dir / "receptor.pdb"
    pose_set = run_dir / "selected_poses.sdf"
    run_dir.mkdir(parents=True)
    receptor.write_text(PDB_COMPLEX)
    pose_set.write_text(
        "pose\n  test\n\n"
        "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
        "   20.0000   30.0000   40.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "   24.0000   36.0000   48.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0\nM  END\n$$$$\n"
    )
    _write_job(
        run_dir,
        {"run_id": "pose-target", "job_code": "POSE1"},
        [(receptor, "prepared_target"), (pose_set, "pose_set")],
    )
    choice = target_inventory(("prepared_target",))[0].choice

    assert target_ligand_path(choice) == pose_set
    assert target_coordinate_ligand_box(choice) == {
        "center": (22.0, 33.0, 44.0),
        "size": (4.0, 6.0, 8.0),
    }


def test_inventory_recognizes_docked_atom_unl_and_uses_ligand_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    run_dir = runs / "structure-jobs" / "dock-1"
    complex_path = run_dir / "docked_complex.pdb"
    complex_path.parent.mkdir(parents=True)
    complex_path.write_text(PDB_DOCKED_COMPLEX)
    _write_job(
        run_dir,
        {
            "run_id": "dock-1",
            "job_code": "DCK01",
            "source": "gnina",
            "ligand_key": "STE|A|200|_",
            "ligand_id": "candidate-42",
            "ligand_smiles": "CCO",
        },
        [(complex_path, "prepared_complex")],
    )

    entry = target_inventory(("prepared_complex",))[0]

    assert entry.row["Ligands"] == "candidate-42"
    assert entry.row["Coordinate ID"] == "UNL"
    assert entry.row["Formula"] == "C2H6O"
    assert entry.row["Residues"] == 1
    assert bound_ligand_box(entry.choice, "STE|A|200|_") == {
        "center": (21.0, 22.0, 23.0),
        "size": (10.0, 10.0, 10.0),
    }


def test_openvs_docked_complex_is_available_for_typed_downstream_handoff(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    run_dir = runs / "docking" / "openvs-1"
    complex_path = run_dir / "best_openvs_complex.pdb"
    complex_path.parent.mkdir(parents=True)
    complex_path.write_text(PDB_COMPLEX)
    _write_job(
        run_dir,
        {
            "run_id": "openvs-1",
            "job_code": "OVS01",
            "workflow": "openvs_docking",
            "tool": "openvs",
            "engine": "openvs",
        },
        [(complex_path, "docked_complex")],
    )

    entry = target_inventory(("prepared_complex", "docked_complex"))[0]

    assert entry.choice.artifact.artifact_type == "docked_complex"
    assert entry.viewer_path == complex_path
    assert entry.row["Kind"] == "Complex"
    assert entry.row["Origin"] == "Docking"
    assert "MD preparation required" in entry.row["Preparation"]


def test_pocket_options_are_limited_to_selected_target(tmp_path: Path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    for index, target_id in enumerate(("target-1", "target-2"), start=1):
        run_dir = runs / "pocket-detection" / f"pocket-{index}"
        pocket = run_dir / "pocket.pdb"
        pocket.parent.mkdir(parents=True)
        pocket.write_text(PDB_COMPLEX)
        _write_job(
            run_dir,
            {
                "run_id": f"pocket-{index}",
                "job_code": f"PKT0{index}",
                "prepared_target_run_id": target_id,
            },
            [(pocket, "pocket")],
        )

    options = artifact_options(("pocket",), source_run_id="target-1")
    assert len(options) == 1
    assert next(iter(options.values())).job.run_id == "pocket-1"


def test_pharmacophore_options_show_saved_name_and_revision_code(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    run_dir = runs / "pharmacophore-hypotheses" / "hypothesis-1"
    hypothesis = run_dir / "pharmacophore.json"
    hypothesis.parent.mkdir(parents=True)
    hypothesis.write_text("{}")
    _write_job(
        run_dir,
        {
            "run_id": "hypothesis-1",
            "job_code": "HYP01",
            "workflow": "pharmacophore_hypothesis",
            "name": "4LNW SER277 side-chain acceptor",
        },
        [(hypothesis, "pharmacophore_hypothesis")],
    )

    options = artifact_options(("pharmacophore_hypothesis",))

    assert list(options) == [
        "4LNW SER277 side-chain acceptor — HYP01"
    ]
