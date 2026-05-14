from mn_ligand.app.pages.common import render_workflow_page


render_workflow_page(
    "qc",
    title_override="Quantum Chemistry (QC)",
    intro_text="Minimal QC task. Submit molecule input and key QC parameters, then track runs in Jobs – QC.",
    show_container_input=False,
    show_command_preview=False,
    show_command_in_result=False,
    run_button_label="Run QC",
)
