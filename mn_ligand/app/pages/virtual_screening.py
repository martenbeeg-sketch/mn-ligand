from mn_ligand.app.pages.discover_inputs import InputSpec, render_discover_job


render_discover_job(
    title="Virtual Screening",
    tools=("Vina campaign", "GNINA campaign", "Uni-Dock Pro campaign", "RosettaLigand campaign"),
    inputs=(
        InputSpec("Prepared target", ("prepared_target", "prepared_receptor")),
        InputSpec("Pocket", ("pocket",)),
        InputSpec("Compound set", ("compound_set", "prepared_ligand_set")),
    ),
    task_key="virtual_screening",
)
