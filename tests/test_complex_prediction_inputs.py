from __future__ import annotations

from pathlib import Path

import pytest

from mn_ligand.workflows.complex_prediction_inputs import (
    create_complex_prediction_inputs,
    normalize_ligand,
    normalize_sequence,
)


def test_sequence_and_single_ligand_inputs_are_typed_and_normalized(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))

    job, target, compounds, proteins = create_complex_prediction_inputs(
        protein_input=">target alpha\nACDEFGHIKLMNPQRSTVWY",
        ligand_input="T3,C(C)O",
    )

    assert job.status == "completed"
    assert target.artifact_type == "target_sequence"
    assert compounds.artifact_type == "compound_set"
    assert proteins == (("A", "ACDEFGHIKLMNPQRSTVWY"),)
    assert target.resolve(job.run_dir, must_exist=True).read_text().startswith(">target")
    assert "CCO" in compounds.resolve(job.run_dir, must_exist=True).read_text()


def test_sequence_ligand_validation_rejects_campaign_style_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        normalize_sequence("ACD")
    with pytest.raises(ValueError, match="LIGAND_ID,SMILES"):
        normalize_ligand("CCO")
    with pytest.raises(ValueError, match="valid SMILES"):
        normalize_ligand("LIG,not-a-smiles")
