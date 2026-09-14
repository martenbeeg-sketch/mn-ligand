from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_settings_page_shows_runtime_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_LIGAND_APP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MN_LIGAND_REFERENCE_DIR", str(tmp_path / "references"))
    monkeypatch.setenv("MN_LIGAND_CONFIG", str(tmp_path / "runtime.json"))

    page = AppTest.from_file(PROJECT_DIR / "mn_ligand/app/pages/settings.py").run(timeout=20)

    assert not page.exception
    assert any(item.label == "Results and jobs directory" for item in page.text_input)
    assert any(item.label == "Reference files directory" for item in page.text_input)
    assert any(
        item.label == "Uni-Dock Pro maximum compounds per batch"
        and item.value == 10_000
        for item in page.number_input
    )
    assert any(
        {"ID", "Label", "Category", "SMILES variants", "SMILES"}.issubset(
            table.value.columns
        )
        for table in page.dataframe
    )
    assert any(item.label == "Active services" for item in page.metric)
    assert any(item.label == "Queued jobs" for item in page.metric)
    assert any(item.label == "Active GPU leases" for item in page.metric)
    assert any(item.label == "Refresh worker status" for item in page.button)
    assert any(item.label == "Run diagnostics" for item in page.button)
