from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mn_ligand.runtime import app_home


NETWORK_APPEARANCE_DEFAULTS: dict[str, float] = {
    "molecule_scale": 1.10,
    "ellipse_width_scale": 1.10,
    "ellipse_height_scale": 1.10,
    "canvas_scale": 1.00,
    "residue_circle_size": 380.0,
    "residue_font_size": 4.25,
    "legend_spacing": 0.06,
}

_NETWORK_APPEARANCE_LIMITS: dict[str, tuple[float, float]] = {
    "molecule_scale": (0.50, 2.00),
    "ellipse_width_scale": (0.50, 2.00),
    "ellipse_height_scale": (0.50, 2.00),
    "canvas_scale": (0.75, 2.00),
    "residue_circle_size": (100.0, 600.0),
    "residue_font_size": (2.0, 8.0),
    "legend_spacing": (0.02, 0.30),
}


def ui_preferences_path() -> Path:
    return app_home() / "config" / "ui_preferences.json"


def load_network_appearance() -> dict[str, float]:
    try:
        payload = json.loads(ui_preferences_path().read_text())
    except (OSError, TypeError, ValueError):
        payload = {}
    stored = (
        payload.get("interaction_network")
        if isinstance(payload, dict)
        else {}
    )
    stored = stored if isinstance(stored, dict) else {}
    preferences = dict(NETWORK_APPEARANCE_DEFAULTS)
    for key, default in NETWORK_APPEARANCE_DEFAULTS.items():
        try:
            value = float(stored.get(key, default))
        except (TypeError, ValueError):
            continue
        lower, upper = _NETWORK_APPEARANCE_LIMITS[key]
        preferences[key] = min(upper, max(lower, value))
    return preferences


def save_network_appearance(values: dict[str, Any]) -> Path:
    current = load_network_appearance()
    for key in NETWORK_APPEARANCE_DEFAULTS:
        if key not in values:
            continue
        try:
            value = float(values[key])
        except (TypeError, ValueError):
            continue
        lower, upper = _NETWORK_APPEARANCE_LIMITS[key]
        current[key] = min(upper, max(lower, value))
    target = ui_preferences_path()
    try:
        payload = json.loads(target.read_text())
    except (OSError, TypeError, ValueError):
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    payload["schema_version"] = 1
    payload["interaction_network"] = current
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    return target
