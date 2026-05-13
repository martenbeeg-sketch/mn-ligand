from ovo_ligand.app.pages.common import render_workflow_page


render_workflow_page(
    "admet",
    title_override="Ligand ADMET",
    intro_text="Minimal ADMET task. Submit ligand SMILES input via file and track runs in Jobs – ADMET.",
    show_container_input=False,
    show_command_preview=False,
    show_command_in_result=False,
    run_button_label="Run ADMET",
)
