from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from mn_ligand.workflows.compound_preparation import create_compound_import_job
from mn_ligand.workflows.compound_pubchem import (
    accepted_reviewed_compound_rows,
    compare_molecular_formulas,
    create_compound_review_job,
    latest_compound_review_map,
    pubchem_component_options,
    search_pubchem_candidates,
    selected_parent_candidate,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_pubchem_search_returns_provenance_rich_candidates() -> None:
    payload = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": 6450278,
                    "MolecularFormula": "C21H28O2",
                    "SMILES": "CC=C1C(=O)CC2C1CCC2",
                    "ConnectivitySMILES": "CC=C1C(=O)CC2C1CCC2",
                    "InChIKey": "EXAMPLE-INCHIKEY",
                }
            ]
        }
    }
    with patch(
        "mn_ligand.workflows.compound_pubchem.urlopen",
        side_effect=[
            _Response({"IdentifierList": {"CID": [6450278]}}),
            _Response(payload),
        ],
    ) as mocked:
        candidates = search_pubchem_candidates(
            "39025-23-5", query_type="cas"
        )

    assert len(candidates) == 1
    assert candidates[0]["cid"] == 6450278
    assert candidates[0]["query"] == "39025-23-5"
    assert candidates[0]["query_type"] == "cas"
    assert candidates[0]["pubchem_url"].endswith("/6450278")
    assert "39025-23-5" in mocked.call_args_list[0].args[0].full_url


def test_pubchem_name_search_returns_full_and_base_name_candidates() -> None:
    properties = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": cid,
                    "MolecularFormula": formula,
                    "SMILES": smiles,
                    "ConnectivitySMILES": smiles,
                    "InChIKey": f"KEY-{cid}",
                }
                for cid, formula, smiles in (
                    (166638328, "C30H55NO5S", "CCN.CCO"),
                    (11583880, "C27H46O5S", "CCO"),
                    (25208970, "C27H45NaO5S", "CCO.[Na+]"),
                    (163320958, "C27H45NaO5S", "[2H]C.O"),
                )
            ]
        }
    }
    with patch(
        "mn_ligand.workflows.compound_pubchem.urlopen",
        side_effect=[
            _Response({"IdentifierList": {"CID": [166638328]}}),
            _Response(
                {
                    "IdentifierList": {
                        "CID": [
                            11583880,
                            25208970,
                            163320958,
                            166638328,
                        ]
                    }
                }
            ),
            _Response(properties),
        ],
    ) as mocked:
        candidates = search_pubchem_candidates(
            "Larsucosterol (trimethylamine)",
            query_type="name",
        )

    assert [candidate["cid"] for candidate in candidates] == [
        166638328,
        11583880,
        25208970,
        163320958,
    ]
    assert candidates[0]["match_scope"] == "full vendor name"
    assert candidates[1]["match_scope"] == (
        "base name without formulation qualifier"
    )
    assert "Larsucosterol%20%28trimethylamine%29" in (
        mocked.call_args_list[0].args[0].full_url
    )
    assert "/name/Larsucosterol/cids/" in (
        mocked.call_args_list[1].args[0].full_url
    )


@pytest.mark.parametrize(
    ("vendor_formula", "pubchem_formula", "expected_scale"),
    [
        ("C15H10O6.1/4H2O", "C60H42O25", "4"),
        ("C16H22N6O4.3/2C2H4O2", "C38H56N12O14", "2"),
    ],
)
def test_formula_comparison_accepts_scaled_fractional_formulations(
    vendor_formula: str,
    pubchem_formula: str,
    expected_scale: str,
) -> None:
    comparison = compare_molecular_formulas(
        vendor_formula,
        pubchem_formula,
    )

    assert comparison["compatible"] is True
    assert comparison["status"] == "scaled formula unit"
    assert comparison["scale"] == expected_scale


def test_formula_comparison_reports_incompatible_composition() -> None:
    comparison = compare_molecular_formulas("C2H6O", "C2H6O2")

    assert comparison["compatible"] is False
    assert comparison["status"] == "stoichiometry differs"


@pytest.mark.parametrize(
    ("full_smiles", "parent_formula", "parent_occurrences"),
    [
        (
            ".".join(
                ["O=C1C(O)=C(c2ccc(O)c(O)c2)Oc2cc(O)ccc21"] * 4
                + ["O"]
            ),
            "C15H10O6",
            4,
        ),
        (
            ".".join(
                [
                    "O=C(N)[C@H]1N(C([C@H](CC2=CNC=N2)"
                    "NC([C@H](CC3)NC3=O)=O)=O)CCC1"
                ]
                * 2
                + ["CC(=O)O"] * 3
            ),
            "C16H22N6O4",
            2,
        ),
    ],
)
def test_pubchem_formulations_select_one_repeated_parent_component(
    full_smiles: str,
    parent_formula: str,
    parent_occurrences: int,
) -> None:
    options = pubchem_component_options(full_smiles)

    assert options[0]["formula"] == parent_formula
    assert options[0]["occurrences"] == parent_occurrences
    assert options[0]["automatic_parent_candidate"] is True
    selected = selected_parent_candidate(
        {
            "smiles": full_smiles,
            "molecular_formula": "full-record-formula",
        },
        options[0],
    )
    assert selected["molecular_formula"] == parent_formula
    assert "." not in selected["smiles"]
    assert selected["pubchem_record_smiles"] == full_smiles
    assert selected["pubchem_record_formula"] == "full-record-formula"


def test_compound_review_rejects_unselected_multifragment_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    source = create_compound_import_job(
        b"id,smiles\ngood,CCO\nbad,invalid\n",
        filename="review.csv",
        id_column="id",
        smiles_column="smiles",
    )

    with pytest.raises(ValueError, match="exactly one selected parent"):
        create_compound_review_job(
            source,
            source_row=3,
            compound_id="bad",
            decision="accepted",
            candidate={"cid": 1, "smiles": "CCO.O"},
        )


def test_compound_review_jobs_persist_accept_and_skip_decisions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path))
    source = create_compound_import_job(
        b"id,smiles\nvalid,CCO\nbad-one,invalid\nbad-two,also-invalid\n",
        filename="review.csv",
        id_column="id",
        smiles_column="smiles",
    )
    candidate = {
        "cid": 702,
        "smiles": "CCO",
        "connectivity_smiles": "CCO",
        "molecular_formula": "C2H6O",
        "inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "query": "ethanol",
        "query_type": "name",
        "pubchem_url": "https://pubchem.ncbi.nlm.nih.gov/compound/702",
        "retrieved_at": "2026-07-24T00:00:00+00:00",
    }

    accepted = create_compound_review_job(
        source,
        source_row=3,
        compound_id="bad-one",
        decision="accepted",
        query="ethanol",
        query_type="name",
        candidate=candidate,
    )
    skipped = create_compound_review_job(
        source,
        source_row=4,
        compound_id="bad-two",
        decision="rejected",
    )

    assert accepted.artifact_manifest is not None
    review_path = accepted.artifact_manifest.by_type("compound_review")[
        0
    ].resolve(accepted.run_dir, must_exist=True)
    review = json.loads(review_path.read_text())
    assert review["candidate"]["cid"] == 702
    assert review["candidate"]["smiles"] == "CCO"
    assert review["candidate"]["formula"] == "C2H6O"
    assert review["candidate"]["validation_engine"] == "RDKit"
    assert review["candidate"]["structure_origin"] == (
        "PubChem confirmed import"
    )
    assert review["source_run_id"] == source.run_id
    reviewed_path = accepted.artifact_manifest.by_type("reviewed_compound")[
        0
    ].resolve(accepted.run_dir, must_exist=True)
    reviewed = pd.read_csv(reviewed_path, keep_default_na=False)
    assert reviewed.loc[0, "structure_origin"] == "PubChem confirmed import"
    assert reviewed.loc[0, "molecular_weight"] > 0
    assert reviewed.loc[0, "qed"] > 0
    assert skipped.metadata["decision"] == "rejected"
    latest = latest_compound_review_map(source.run_id)
    assert latest["bad-one"].metadata["decision"] == "accepted"
    assert latest["bad-two"].metadata["decision"] == "rejected"
    additions = accepted_reviewed_compound_rows(source.run_id)
    assert len(additions) == 1
    assert additions[0]["compound_id"] == "bad-one"
    assert additions[0]["structure_origin"] == "PubChem confirmed import"
    assert additions[0]["review_job"] == accepted.metadata["job_code"]
