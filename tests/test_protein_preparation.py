from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from mn_ligand.workflows.protein_preparation import (
    _internal_gap_pir,
    _merge_modeled_chain,
    _restore_modeled_author_numbering,
    create_protein_import_job,
    detect_noncanonical_residues,
    detect_modeller_internal_gaps,
    detect_structure_format,
    imported_target,
    inline_structure_preview_data,
    repair_noncanonical_residues_with_modeller,
    resolve_present_protein_chains,
    structure_summary,
    run_protein_cleaning_job,
)
from mn_ligand.workflows.bound_ligand_md import (
    map_modified_residues_to_standard,
    normalize_structure_for_repair,
)
from mn_ligand.ligandx.lib.chemistry.preparation.protein import ProteinPreparer


PDB_DATA = """HEADER    TEST
TITLE     EXAMPLE RECEPTOR COMPLEX
EXPDTA    X-RAY DIFFRACTION
REMARK   2 RESOLUTION.    1.80 ANGSTROMS.
COMPND    MOL_ID: 1;
COMPND   2 MOLECULE: EXAMPLE RECEPTOR;
COMPND   3 CHAIN: A;
SOURCE    MOL_ID: 1;
SOURCE   2 ORGANISM_SCIENTIFIC: HOMO SAPIENS;
SOURCE   3 EXPRESSION_SYSTEM: ESCHERICHIA COLI;
HETNAM     LIG EXAMPLE INHIBITOR
HETSYN     LIG TEST COMPOUND; EXAMPLE LIGAND
FORMUL   2  LIG    C2 H4 O2
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.400   0.000   0.000  1.00 20.00           C
HETATM    3  C1  LIG A 101       3.000   0.000   0.000  1.00 20.00           C
END
"""

PREPARED_DATA = """HEADER    CLEANED
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.400   0.000   0.000  1.00 20.00           C
HETATM    3  C1  LIG A 101       3.000   0.000   0.000  1.00 20.00           C
END
"""
REPAIR_FIXTURE = Path(__file__).parent / "fixtures" / "repair_noncanonical_missing.pdb"


def test_detect_structure_format_rejects_non_structure() -> None:
    with pytest.raises(ValueError):
        detect_structure_format("not a structure", "target.pdb")


def test_structure_summary_extracts_pdb_ligand_identity() -> None:
    summary = structure_summary(PDB_DATA, "pdb")

    assert summary["ligands"] == [
        {
            "key": "LIG|A|101|_",
            "resname": "LIG",
            "chain": "A",
            "resseq": "101",
            "icode": "_",
            "atom_count": 1,
            "heavy_atom_count": 1,
            "center": [3.0, 0.0, 0.0],
            "ccd_id": "",
            "name": "EXAMPLE INHIBITOR",
            "formula": "C2 H4 O2",
            "synonyms": ["TEST COMPOUND", "EXAMPLE LIGAND"],
        }
    ]
    assert summary["receptor"]["title"] == "EXAMPLE RECEPTOR COMPLEX"
    assert summary["receptor"]["experimental_method"] == "X-RAY DIFFRACTION"
    assert summary["receptor"]["resolution_angstrom"] == 1.8
    assert summary["receptor"]["entities"][0]["name"] == "EXAMPLE RECEPTOR"
    assert summary["receptor"]["entities"][0]["source_organisms"] == ["HOMO SAPIENS"]


def test_structure_summary_does_not_count_atom_unl_as_protein() -> None:
    data = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  C1  UNL     1       3.000   0.000   0.000  1.00 20.00           C
END
"""

    summary = structure_summary(data, "pdb")

    assert summary["protein_atom_records"] == 1
    assert summary["protein_residues"] == 1
    assert summary["ligands"][0]["resname"] == "UNL"


def test_protein_import_publishes_immutable_target(tmp_path: Path) -> None:
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        job = create_protein_import_job(PDB_DATA, filename="4lnw.pdb", source="pdb", pdb_id="4lnw")

    target = imported_target(job)
    target_path = target.resolve(job.run_dir, must_exist=True)
    assert job.status == "completed"
    assert job.schema_version == 1
    assert target.path.startswith("artifacts/imported/")
    assert target_path is not None and target_path.read_text() == PDB_DATA
    assert job.result["protein_residues"] == 1
    assert job.metadata["ligands"][0]["name"] == "EXAMPLE INHIBITOR"
    assert target.metadata["ligands"][0]["formula"] == "C2 H4 O2"
    assert job.metadata["receptor"]["entities"][0]["chains"] == ["A"]
    assert not Path(target.path).is_absolute()


def test_supported_noncanonical_mapping_keeps_ligand_and_drops_substituent() -> None:
    mapped, report = map_modified_residues_to_standard(REPAIR_FIXTURE.read_text())

    assert "CAS A   3" not in mapped
    assert "CYS A   3" in mapped
    assert "AS1  CAS" not in mapped
    assert "LIG B 101" in mapped
    assert report["mappings"]["CAS|A|3|_"] == {
        "target": "CYS",
        "kept_atoms": 6,
        "dropped_atoms": 1,
    }


def test_caf_is_authoritatively_mapped_to_cysteine_without_cacodylate() -> None:
    caf = REPAIR_FIXTURE.read_text().replace("CAS", "CAF")

    mapped, report = map_modified_residues_to_standard(caf)

    assert "CAF A   3" not in mapped
    assert "CYS A   3" in mapped
    assert "AS1  CAF" not in mapped
    assert report["mappings"]["CAF|A|3|_"]["target"] == "CYS"
    assert report["mappings"]["CAF|A|3|_"]["dropped_atoms"] == 1


def test_noncanonical_detection_is_site_specific_and_uses_modres_as_suggestion() -> None:
    pdb_data = """HEADER    SITE-SPECIFIC NONCANONICAL TEST
MODRES TEST CAS A  244  CYS  FIRST SITE
MODRES TEST CAS A  388  CYS  SECOND SITE
SEQADV TEST CAS A  388  UNP  P00000    MET   388 ENGINEERED MUTATION
HETATM    1  N   CAS A 244       0.000   0.000   0.000  1.00 20.00           N
HETATM    2  CA  CAS A 244       1.000   0.000   0.000  1.00 20.00           C
HETATM    3  N   CAS A 388       5.000   0.000   0.000  1.00 20.00           N
HETATM    4  CA  CAS A 388       6.000   0.000   0.000  1.00 20.00           C
END
"""

    sites = detect_noncanonical_residues(pdb_data)

    assert [site["key"] for site in sites] == ["CAS|A|244|_", "CAS|A|388|_"]
    assert [site["suggested_target"] for site in sites] == ["CYS", "MET"]
    assert sites[0]["evidence"] == "PDB MODRES"
    assert sites[1]["evidence"] == "PDB MODRES + SEQADV"
    assert "ENGINEERED MUTATION" in sites[1]["evidence_detail"]
    assert structure_summary(pdb_data, "pdb")["ligands"] == []


def test_modeller_repair_accepts_different_targets_for_same_component(
    tmp_path: Path,
) -> None:
    pdb_data = """HEADER    SITE-SPECIFIC NONCANONICAL TEST
MODRES TEST CAS A  244  CYS  FIRST SITE
MODRES TEST CAS A  388  CYS  SECOND SITE
HETATM    1  N   CAS A 244       0.000   0.000   0.000  1.00 20.00           N
HETATM    2  CA  CAS A 244       1.000   0.000   0.000  1.00 20.00           C
HETATM    3  N   CAS A 388       5.000   0.000   0.000  1.00 20.00           N
HETATM    4  CA  CAS A 388       6.000   0.000   0.000  1.00 20.00           C
HETATM    5  C1  LIG B 501       8.000   0.000   0.000  1.00 20.00           C
END
"""

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("--output") + 1])
        report = Path(command[command.index("--report") + 1])
        output.write_text(
            """ATOM      1  N   CYS A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  CYS A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  N   MET A   2       5.000   0.000   0.000  1.00 20.00           N
ATOM      4  CA  MET A   2       6.000   0.000   0.000  1.00 20.00           C
HETATM    5  C1  BLK B   3       8.000   0.000   0.000  1.00 20.00           C
END
"""
        )
        report.write_text(
            json.dumps(
                {
                    "success": True,
                    "engine": "MODELLER",
                    "replacements": [
                        {"key": "CAS|A|244|_", "chain": "A", "resseq": "244", "icode": "", "target": "CYS", "model_index": 1},
                        {"key": "CAS|A|388|_", "chain": "A", "resseq": "388", "icode": "", "target": "MET", "model_index": 2},
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with patch("mn_ligand.workflows.protein_preparation.subprocess.run", side_effect=fake_run):
        repaired, report = repair_noncanonical_residues_with_modeller(
            pdb_data,
            [
                {"key": "CAS|A|244|_", "target": "CYS"},
                {"key": "CAS|A|388|_", "target": "MET"},
            ],
            work_dir=tmp_path,
        )

    assert "CYS A 244" in repaired
    assert "MET A 388" in repaired
    assert "CAS A 244" not in repaired
    assert "LIG B 501" in repaired
    assert report["engine"] == "MODELLER"


def test_missing_residue_policy_skips_terminals_and_long_internal_gaps() -> None:
    class Chain:
        def __init__(self, chain_id: str):
            self.id = chain_id

    class Sequence:
        chainId = "A"
        residues = ["ALA"] * 20

    class Topology:
        @staticmethod
        def chains():
            return iter([Chain("A")])

    class Fixer:
        topology = Topology()
        sequences = [Sequence()]
        missingResidues = {
            (0, 0): ["ALA", "GLY"],
            (0, 4): ["SER"] * 11,
            (0, 6): ["THR"],
        }

    added, skipped = ProteinPreparer._missing_residue_policy(
        Fixer(),
        skip_terminal_missing_residues=True,
        max_internal_gap=10,
    )

    assert added == []
    assert [item["reason"] for item in skipped] == [
        "terminal",
        "gap_exceeds_10",
        "terminal",
    ]
    assert Fixer.missingResidues == {}


def test_modeller_gap_detection_uses_authoritative_missing_residue_numbers() -> None:
    seqres = "ALA GLY " + " ".join(
        ["GLN", "ALA", "PRO", "ILE", "VAL", "ASN", "ALA", "PRO", "GLU", "GLY", "GLY", "LYS"]
    ) + " VAL ASP"
    data = f"""HEADER    MODELLER GAP TEST
SEQRES   1 A   16  {seqres}
REMARK 465     GLN A     3
REMARK 465     ALA A     4
REMARK 465     PRO A     5
REMARK 465     ILE A     6
REMARK 465     VAL A     7
REMARK 465     ASN A     8
REMARK 465     ALA A     9
REMARK 465     PRO A    10
REMARK 465     GLU A    11
REMARK 465     GLY A    12
REMARK 465     GLY A    13
REMARK 465     LYS A    14
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  N   GLY A   2       1.000   0.000   0.000  1.00 20.00           N
ATOM      3  N   VAL A  15       2.000   0.000   0.000  1.00 20.00           N
ATOM      4  N   ASP A  16       3.000   0.000   0.000  1.00 20.00           N
END
"""

    detected = detect_modeller_internal_gaps(data, max_internal_gap=15)

    assert detected["A"]["alignment_method"].startswith("authoritative")
    assert detected["A"]["gaps"] == [
        {
            "chain": "A",
            "target_start": 3,
            "target_end": 14,
            "author_start": 3,
            "author_end": 14,
            "insertion_index": 2,
            "sequence": "QAPIVNAPEGGK",
            "residues": [
                "GLN", "ALA", "PRO", "ILE", "VAL", "ASN",
                "ALA", "PRO", "GLU", "GLY", "GLY", "LYS",
            ],
            "length": 12,
            "terminal": False,
            "eligible": True,
            "reason": "",
            "evidence": "PDB REMARK 465 + SEQRES",
        }
    ]


def test_modeled_chain_merge_removes_stale_conect_records() -> None:
    source = (
        "HEADER    TEST\n"
        "ATOM    100  N   ALA X  10       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM    101  CA  ALA X  10       1.000   0.000   0.000  1.00 20.00           C\n"
        "HETATM  200  C1  LIG X 500       5.000   0.000   0.000  1.00 20.00           C\n"
        "HETATM  201  C2  LIG X 500       6.000   0.000   0.000  1.00 20.00           C\n"
        "CONECT  100  200\n"
        "CONECT  200  201\n"
        "END\n"
    )
    model = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )

    merged, report = _merge_modeled_chain(source, model, chain="X")

    assert "CONECT" not in merged
    assert merged.count("HETATM") == 2
    assert report["retained_heterogen_atom_count"] == 2


def test_internal_gap_alignment_preserves_author_residue_numbers() -> None:
    alignment = _internal_gap_pir(
        template_name="template",
        chain="X",
        start=202,
        end=460,
        template_alignment="AG--VT",
        target_alignment="AGQKVT",
    )

    assert "structureX:template:202:X:460:X" in alignment
    assert "sequence:repaired:202:X:460:X" in alignment


def test_modeled_gap_restores_deposited_author_numbering() -> None:
    source = (
        "ATOM      1  N   GLY X 251       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  N   VAL X 264       4.000   0.000   0.000  1.00 20.00           N\n"
        "END\n"
    )
    model = (
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  N   GLN A   2       1.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      3  N   LYS A   3       2.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      4  N   VAL A   4       4.000   0.000   0.000  1.00  0.00           N\n"
        "END\n"
    )

    restored = _restore_modeled_author_numbering(
        source,
        model,
        chain="X",
        gaps=[
            {
                "insertion_index": 1,
                "length": 2,
                "author_start": 252,
                "author_end": 253,
            }
        ],
    )

    assert [int(line[22:26]) for line in restored.splitlines() if line.startswith("ATOM")] == [
        251,
        252,
        253,
        264,
    ]


def test_historical_chain_rewrite_and_ligand_only_preview_are_recovered() -> None:
    protein = (
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "END\n"
    )
    ligand_only_complex = (
        "HETATM    1  C1  LIG X 500       1.000   0.000   0.000  1.00 20.00           C\n"
        "END\n"
    )

    assert resolve_present_protein_chains(protein, ["X"]) == {"A"}
    assert inline_structure_preview_data(ligand_only_complex, protein) == protein
    assert inline_structure_preview_data(protein + ligand_only_complex, protein) != protein


def test_pdbfixer_output_preserves_chain_and_residue_ids() -> None:
    preparer = ProteinPreparer()
    fixer = type("Fixer", (), {"topology": object(), "positions": object()})()

    with patch(
        "mn_ligand.ligandx.lib.chemistry.preparation.protein.PDBFile",
        create=True,
    ) as pdb_file:
        preparer._fixer_to_pdb_string(fixer)

    assert pdb_file.writeFile.call_args.kwargs["keepIds"] is True


def test_gemmi_mmcif_normalization_preserves_sequence_records() -> None:
    import gemmi

    structure = gemmi.read_pdb_string(REPAIR_FIXTURE.read_text())
    structure.setup_entities()
    mmcif = structure.make_mmcif_document().as_string()

    normalized, report = normalize_structure_for_repair(mmcif)

    assert "SEQRES   1 A    3  ALA GLY CYS" in normalized
    assert "HETATM" in normalized
    assert report["engine"] == "Gemmi"
    assert report["input_format"] == "mmcif"
    assert report["sequence_records_preserved"] is True


def test_cleaning_consumes_import_and_publishes_prepared_target(tmp_path: Path) -> None:
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output_mount = next(command[index + 1] for index, item in enumerate(command) if item == "-v" and command[index + 1].endswith(":/output"))
        output_dir = Path(output_mount.removesuffix(":/output"))
        (output_dir / "native_result.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "prepared_pdb_data": PREPARED_DATA,
                    "protein_cleaned": True,
                    "components": {"protein": 1, "ligands": 1},
                    "modified_residue_mapping": {"enabled": True, "mappings": {}},
                    "repair_report": {
                        "engine": "PDBFixer",
                        "strict": True,
                        "missing_residue_segments": [
                            {
                                "chain_index": 0,
                                "insertion_index": 1,
                                "residues": ["GLY"],
                            }
                        ],
                    },
                    "ligands": [{"key": "LIG|A|101|_"}],
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout="cleaned", stderr="")

    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        imported = create_protein_import_job(PDB_DATA, filename="target.pdb", source="upload")
        with patch("mn_ligand.workflows.protein_preparation.subprocess.run", side_effect=fake_run):
            cleaned, native_payload = run_protein_cleaning_job(imported.run_id)

    assert cleaned.status == "completed"
    assert cleaned.parent_run_id == imported.run_id
    assert native_payload["success"] is True
    assert cleaned.artifact_manifest is not None
    prepared = cleaned.artifact_manifest.by_type("prepared_target")[0]
    prepared_path = prepared.resolve(cleaned.run_dir, must_exist=True)
    assert prepared_path is not None
    assert "HETATM" not in prepared_path.read_text()
    assert cleaned.artifact_manifest.by_type("repair_report")
    report_ref = cleaned.artifact_manifest.by_type("repair_report")[0]
    report_path = report_ref.resolve(cleaned.run_dir, must_exist=True)
    assert report_path is not None
    repair_report = json.loads(report_path.read_text())
    assert repair_report["pdbfixer"]["strict"] is True
    assert repair_report["pdbfixer"]["missing_residue_segments"][0]["residues"] == ["GLY"]
    input_payload = json.loads((cleaned.run_dir / "input.json").read_text())
    assert input_payload["input_artifact"]["run_id"] == imported.run_id
    assert input_payload["parameters"]["skip_terminal_missing_residues"] is True
    assert input_payload["parameters"]["max_internal_gap"] == 15
    assert input_payload["parameters"]["add_missing_residues"] is False
    assert input_payload["parameters"]["internal_gap_engine"] == "MODELLER"
    assert input_payload["parameters"]["refine_rebuilt_positions"] is True
    assert input_payload["parameters"]["biological_assembly_id"] == ""
    assert input_payload["parameters"]["preserve_nonwater_heterogens"] is False
    assert not (cleaned.run_dir / "artifacts" / "imported").exists()
    command_record = json.loads((cleaned.run_dir / "command.json").read_text())
    assert command_record["tool_id"] == "protein_cleaning"
    assert command_record["image"] == "ovolig-md-cu128:latest"


def test_cleaning_records_failed_job_when_container_cannot_start(tmp_path: Path) -> None:
    with patch.dict("os.environ", {"MN_LIGAND_RUN_DIR": str(tmp_path)}, clear=False):
        imported = create_protein_import_job(PDB_DATA, filename="target.pdb", source="upload")
        with patch(
            "mn_ligand.workflows.protein_preparation.subprocess.run",
            side_effect=OSError("docker unavailable"),
        ):
            cleaned, payload = run_protein_cleaning_job(imported.run_id)

    assert cleaned.status == "failed"
    assert payload["success"] is False
    assert "docker unavailable" in cleaned.result["error"]
    assert cleaned.artifact_manifest is not None
    assert cleaned.artifact_manifest.artifacts == ()
