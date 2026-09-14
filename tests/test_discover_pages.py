from pathlib import Path


DISCOVER_PAGES = (
    "structure_prediction.py",
    "pocket_detection.py",
    "docking_cofolding.py",
    "redocking_benchmark.py",
    "virtual_screening.py",
    "generative_design.py",
)


def test_discover_pages_do_not_import_files_or_smiles() -> None:
    page_root = Path(__file__).parents[1] / "mn_ligand" / "app" / "pages"
    for filename in DISCOVER_PAGES:
        source = (page_root / filename).read_text()
        assert "file_uploader" not in source
        assert "text_area" not in source
        assert "SMILES" not in source


def test_discover_launchers_use_typed_artifact_inputs() -> None:
    page_root = Path(__file__).parents[1] / "mn_ligand" / "app" / "pages"
    expected = {
        "structure_prediction.py": "prepared_target",
        "docking_cofolding.py": "prepared_ligand_set",
        "redocking_benchmark.py": "prepared_ligand_set",
        "virtual_screening.py": "compound_set",
        "generative_design.py": "pocket",
    }
    for filename, artifact_type in expected.items():
        source = (page_root / filename).read_text()
        assert any(
            selector in source
            for selector in ("select_artifact", "select_target_artifact", "render_discover_job")
        )
        assert artifact_type in source


def test_pocket_detection_page_queues_worker_owned_jobs() -> None:
    source = (
        Path(__file__).parents[1] / "mn_ligand" / "app" / "pages" / "pocket_detection.py"
    ).read_text()

    assert "queue_pocket_detection_job" in source
    assert "run_pocket_detection_job" not in source


def test_docking_page_queues_all_classical_engines_through_worker() -> None:
    source = (
        Path(__file__).parents[1]
        / "mn_ligand"
        / "app"
        / "pages"
        / "docking_cofolding.py"
    ).read_text()

    assert "queue_docking_campaign_job(" in source
    assert "queue_openvs_docking_job(" in source
    assert "run_docking_campaign_job" not in source


def test_refolding_page_queues_af3_boltz2_and_nesso_through_worker() -> None:
    source = (
        Path(__file__).parents[1]
        / "mn_ligand"
        / "app"
        / "pages"
        / "docking_cofolding.py"
    ).read_text()

    assert "queue_alphafold3_refolding_job(" in source
    assert "queue_boltz2_refolding_job(" in source
    assert "queue_nesso_affinity_job(" in source
    assert "run_alphafold3_refolding_job" not in source
    assert "run_boltz2_refolding_job" not in source
    assert "run_nesso_affinity_job" not in source


def test_redocking_refolding_uses_only_the_target_associated_ligand() -> None:
    page_root = Path(__file__).parents[1] / "mn_ligand" / "app" / "pages"
    shared = (page_root / "docking_cofolding.py").read_text()
    redocking = (page_root / "refolding.py").read_text()

    assert "target_ligand_only=True" in redocking
    assert '"Launch campaign name",' in shared
    assert "Launch campaign name (optional)" not in shared
    assert 'blockers.append("Enter a launch campaign name.")' in shared
    assert "launch_campaign_label = campaign_name.strip()" in shared
    assert '["Target", "Engines", "Run", "Results"]' in shared
    assert "compound_paths = [associated_ligand]" in shared
    assert "compound_artifacts = (" in shared
    assert "Queue selected redocking / refolding engines" in shared


def test_docking_and_redocking_share_one_repetition_setting_across_engines() -> None:
    page_root = Path(__file__).parents[1] / "mn_ligand" / "app" / "pages"
    shared = (page_root / "docking_cofolding.py").read_text()
    redocking = (page_root / "refolding.py").read_text()

    assert shared.count('"Independent runs per structure"') == 1
    assert shared.count('"First seed per structure"') == 1
    assert 'column.checkbox(\n                engine,\n                value=True,' in shared
    assert "column.checkbox(engine, value=True, key=ENGINE_KEYS[engine])" in shared
    assert '"Independent runs per structure",\n                    min_value=1,\n                    max_value=100,\n                    value=1,' in shared
    assert 'st.expander("Shared campaign repetitions", expanded=True)' in shared
    assert 'st.expander("Classical docking preparation", expanded=True)' in shared
    assert "Shared campaign repetitions and classical preparation" not in shared
    assert shared.count('config["campaign_replicates"]') == 5
    assert shared.count('config["campaign_seed"]') == 5
    assert 'key="nesso_replicates"' not in shared
    assert 'key="boltz_seed"' not in shared
    assert 'key="nesso_seed"' not in shared
    assert "target_ligand_only=True" in redocking


def test_docking_and_redocking_support_named_multi_target_engine_matrices() -> None:
    page_root = Path(__file__).parents[1] / "mn_ligand" / "app" / "pages"
    shared = (page_root / "docking_cofolding.py").read_text()

    assert "select_target_artifacts(" in shared
    assert '"Global pocket"' in shared
    assert '"Displayed target model"' in shared
    assert 'context["center"] = tuple(float(value)' in shared
    assert '"Align all target-associated ligand longest axes to X"' in shared
    assert "_build_target_launch_context(" in shared
    assert "st.expander(target_title" not in shared
    assert '"Target pose-view columns"' not in shared
    assert '"Target mode"' in shared
    assert '("Single target", "Multi-target ensemble")' in shared
    assert "maximum=None if ensemble_mode else 1" in shared
    assert 'selection_frame.insert(0, "Selected", False)' in shared
    assert 'st.markdown("#### Engines by target")' in shared
    assert 'selected_engines_by_target' in shared
    assert "for context in target_contexts:" in shared
    assert "_queue_target_engine_jobs(" in shared
    assert "launch_campaign_id = str(uuid4())" in shared
    assert 'blockers.append("Enter a launch campaign name.")' in shared


def test_reference_guided_placement_is_the_selectable_default() -> None:
    page_root = Path(__file__).parents[1] / "mn_ligand" / "app" / "pages"
    shared = (page_root / "docking_cofolding.py").read_text()
    single_engine = (page_root / "docking.py").read_text()

    assert shared.count(
        '"Placement", ("Reference-guided", "Pocket-center (unguided)"),'
    ) == 1
    assert single_engine.count(
        '("Reference-guided", "Pocket-center (unguided)"),'
    ) == 1
    assert shared.count(
        'index=0,\n                key="openvs_reference_mode"'
    ) == 1
    assert single_engine.count(
        'index=0,\n                        key="openvs_reference_mode"'
    ) == 1


def test_docking_and_prediction_have_specialized_modes() -> None:
    page_root = Path(__file__).parents[1] / "mn_ligand" / "app" / "pages"
    docking = (page_root / "docking_cofolding.py").read_text()
    redocking = (page_root / "redocking_benchmark.py").read_text()
    prediction = (page_root / "structure_prediction.py").read_text()
    assert '["Target", "Compounds", "Engines", "Run", "Results"]' in docking
    assert "render_run_resources(" in docking
    assert '"Queue selected docking / cofolding engines"' in docking
    assert '"Run redocking benchmark"' in redocking
    assert "queue_redocking_benchmark(" in redocking
    assert "_imported_compound_options" in docking
    assert "artifact_options" in docking
    assert all(engine in docking for engine in ("AutoDock Vina", "GNINA", "Uni-Dock Pro"))
    assert 'st.tabs(["Structure Prediction", "Refolding"])' not in prediction
    assert all(tool in docking for tool in ("Boltz-2", "AlphaFold 3", "Nesso-1"))
    assert all(tool in prediction for tool in ("Boltz-2", "AlphaFold 3"))


def test_visible_property_pages_select_prepared_artifacts() -> None:
    page_root = Path(__file__).parents[1] / "mn_ligand" / "app" / "pages"
    for filename in ("admet.py", "qc.py"):
        source = (page_root / filename).read_text()
        assert "input_artifact_types=" in source
        assert "file_uploader" not in source
