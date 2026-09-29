"""Shared LDDM configuration used by design and docking workflows."""

import math

LDDM_CHECKPOINTS = {
    "CD+BB (MIT; commercial use allowed)": "generation/lddm/lddm_CDBB.ckpt",
    "CD+BB+BN (CC-BY-NC 4.0; non-commercial)": "generation/lddm/lddm.ckpt",
}


def mean_coordinate_uncertainty(molecule) -> float | None:
    """Return LDDM's mean positive per-atom coordinate uncertainty."""
    values: list[float] = []
    for atom in molecule.GetAtoms():
        if not atom.HasProp("sigma_x"):
            continue
        try:
            value = float(atom.GetProp("sigma_x"))
        except (TypeError, ValueError):
            continue
        if value > 0 and math.isfinite(value):
            values.append(value)

    if not values and molecule.HasProp("sigma_x"):
        for item in molecule.GetProp("sigma_x").split(","):
            try:
                value = float(item)
            except (TypeError, ValueError):
                continue
            if value > 0 and math.isfinite(value):
                values.append(value)

    return sum(values) / len(values) if values else None
