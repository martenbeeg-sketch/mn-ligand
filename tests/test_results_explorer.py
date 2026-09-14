from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from mn_ligand.app.results_explorer_index import build_results_index


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _job(
    root: Path,
    group: str,
    run_id: str,
    metadata: dict[str, object],
    *,
    input_payload: dict[str, object] | None = None,
) -> Path:
    run_dir = root / group / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "status": "completed",
                "created_at": "2026-07-27T10:00:00+00:00",
                **metadata,
            }
        )
    )
    (run_dir / "result.json").write_text(json.dumps({"success": True}))
    if input_payload is not None:
        (run_dir / "input.json").write_text(json.dumps(input_payload))
    return run_dir


def test_results_index_keeps_dataset_target_variant_and_evaluations_together(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    _job(
        root,
        "compound-import",
        "dataset-1",
        {"dataset_name": "Fixture set"},
    )
    _job(
        root,
        "protein-import",
        "import-1",
        {
            "pdb_id": "4LNW",
            "receptor": {
                "entities": [{"name": "Thyroid hormone receptor alpha"}]
            },
        },
    )
    _job(
        root,
        "structure-jobs",
        "target-1",
        {"parent_run_id": "import-1", "job_code": "TARG1"},
    )
    docking = _job(
        root,
        "docking",
        "dock-1",
        {
            "parent_run_id": "target-1",
            "engine": "vina",
            "workflow": "docking_campaign",
            "job_code": "DOCK1",
            "launch_campaign_id": "campaign-1",
            "launch_campaign_label": "Fixture campaign",
        },
        input_payload={
            "target_artifact": {
                "run_id": "target-1",
                "label": "target_trimmed.pdb",
            },
            "compound_artifacts": [
                {
                    "run_id": "selection-1",
                    "metadata": {"source_compound_run_id": "dataset-1"},
                }
            ],
        },
    )
    input_dir = docking / "input"
    input_dir.mkdir()
    (input_dir / "compounds.tsv").write_text(
        "candidate_id\tsmiles\ncompound-A\tCC\ncompound-B\tCCC\n"
    )
    _job(
        root,
        "pose-validation",
        "poses-1",
        {
            "parent_run_id": "dock-1",
            "workflow": "posebusters_validation",
            "tool": "PoseBusters",
        },
    )
    _job(
        root,
        "interaction-analysis",
        "plip-1",
        {
            "parent_run_id": "dock-1",
            "workflow": "plip_interactions",
            "interaction_engine": "PLIP",
        },
    )

    predictions, compounds = build_results_index(root)

    assert len(predictions) == 1
    row = predictions.iloc[0]
    assert row["dataset"] == "Fixture set"
    assert row["target_family"] == "4LNW · Thyroid hormone receptor alpha"
    assert row["target_variant"] == "PDB 4LNW · prepared target TARG1"
    assert row["target_artifact"] == "target_trimmed.pdb"
    assert row["target_origin"] == "PDB 4LNW"
    assert row["campaign"] == "Fixture campaign"
    assert row["prediction_engine"] == "AutoDock Vina"
    assert row["posebusters"].startswith("./job-results?")
    assert row["plip"].startswith("./job-results?")
    assert row["pandamap"] == ""
    assert set(compounds["compound_id"]) == {"compound-A", "compound-B"}
    assert all("compound_id=compound-" in url for url in compounds["prediction_result"])


def test_results_explorer_renders_without_results(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))
    page = AppTest.from_file(
        PROJECT_DIR / "mn_ligand/app/pages/results_explorer.py"
    ).run(timeout=20)

    assert not page.exception
    assert page.title[0].value == "Results Explorer"
    assert any("No docking" in item.value for item in page.info)
