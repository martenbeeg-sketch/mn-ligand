from __future__ import annotations

from mn_ligand.app.pages.docking_cofolding import render as render_binding


def render() -> None:
    render_binding(
        title="Redocking / Refolding",
        caption=(
            "Redock or refold the coordinate-matched ligand associated with a "
            "prepared target, using the same grouped engine controls as "
            "Docking / Cofolding."
        ),
        target_ligand_only=True,
    )


if __name__ == "__main__":
    render()
