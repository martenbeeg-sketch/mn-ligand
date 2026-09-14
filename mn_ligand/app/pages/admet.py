from mn_ligand.app.pages.common import render_workflow_page


render_workflow_page(
    "admet",
    title_override="Ligand ADMET",
    intro_text="Run ADMET for a prepared compound artifact and track the result in Jobs.",
    show_container_input=False,
    show_command_preview=False,
    show_command_in_result=False,
    run_button_label="Run ADMET",
    input_artifact_types={"smiles_file": ("compound_set",)},
)
