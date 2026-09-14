from __future__ import annotations

import streamlit as st


FIXED_BOX = "Fixed box"
PADDING_BOX = "Padding"


def render_search_region(
    *,
    center: tuple[float, float, float],
    source_size: tuple[float, float, float],
    source_label: str,
    key: str,
    default_padding_angstrom: float = 15.0,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    str,
    float | None,
]:
    """Render a shared fixed-size or source-padding search-region editor."""
    mode = st.segmented_control(
        "Box sizing",
        (FIXED_BOX, PADDING_BOX),
        default=FIXED_BOX,
        key=f"{key}_mode",
        help=(
            "Fixed box uses explicit x/y/z dimensions. Padding expands the selected "
            "pocket, bound ligand, or stored source region on every side."
        ),
    ) or FIXED_BOX
    st.caption(f"Center/source region: {source_label}.")

    center_columns = st.columns(3)
    effective_center = tuple(
        float(
            column.number_input(
                f"center_{axis}",
                value=float(value),
                step=0.5,
                format="%.3f",
                key=f"{key}_center_{axis}",
            )
        )
        for column, axis, value in zip(center_columns, "xyz", center)
    )

    if mode == PADDING_BOX:
        padding = float(
            st.number_input(
                "Padding on each side (Å)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_padding_angstrom),
                step=0.5,
                key=f"{key}_padding",
            )
        )
        calculated_size = tuple(
            max(1.0, float(value) + 2.0 * padding) for value in source_size
        )
        size_signature = (
            tuple(round(float(value), 6) for value in source_size),
            round(padding, 6),
        )
        signature_key = f"{key}_padding_size_signature"
        size_keys = tuple(f"{key}_padding_size_{axis}" for axis in "xyz")
        if st.session_state.get(signature_key) != size_signature:
            for size_key, value in zip(size_keys, calculated_size):
                st.session_state[size_key] = float(value)
            st.session_state[signature_key] = size_signature
        size_columns = st.columns(3)
        effective_size = tuple(
            float(
                column.number_input(
                    f"size_{axis}",
                    min_value=1.0,
                    step=1.0,
                    format="%.2f",
                    key=size_key,
                )
            )
            for column, axis, size_key in zip(
                size_columns, "xyz", size_keys
            )
        )
        manually_adjusted = any(
            abs(actual - calculated) > 1e-6
            for actual, calculated in zip(effective_size, calculated_size)
        )
        st.caption(
            (
                "The displayed dimensions were initialized from the source-region "
                "extent plus padding and then manually adjusted. Changing padding "
                "or the source region recalculates and overwrites all three sizes."
                if manually_adjusted
                else "Dimensions equal the source-region extent plus padding on "
                "both sides. They can be edited manually; changing padding or the "
                "source region recalculates and overwrites all three sizes."
            )
        )
        return (
            effective_center,
            effective_size,
            "padding_manual" if manually_adjusted else "padding",
            padding,
        )

    size_columns = st.columns(3)
    effective_size = tuple(
        float(
            column.number_input(
                f"size_{axis}",
                value=20.0,
                min_value=1.0,
                step=1.0,
                format="%.2f",
                key=f"{key}_size_{axis}",
            )
        )
        for column, axis in zip(size_columns, "xyz")
    )
    return effective_center, effective_size, "fixed", None
