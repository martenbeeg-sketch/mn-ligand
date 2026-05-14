from mn_ligand.app.pages.common import render_workflow_page


render_workflow_page(
    "rbfe",
    show_container_input=False,
    show_command_preview=False,
    show_command_in_result=False,
    title_override="OpenFE RBFE",
    intro_text="Run RBFE with OpenFE-managed Docker execution.",
    run_button_label="Run OpenFE RBFE",
)
