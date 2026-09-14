from __future__ import annotations

from pathlib import Path

from scripts.recompute_md_analysis import _resolve_replica_file


def test_resolver_prefers_exact_configured_nested_trajectory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "replica-id"
    nested = run_dir / "replica-id"
    nested.mkdir(parents=True)
    legacy = run_dir / "production.dcd"
    extended = nested / "production.dcd"
    legacy.write_bytes(b"1 ns")
    extended.write_bytes(b"100 ns")

    resolved = _resolve_replica_file(
        run_dir,
        "/output/replica-id/production.dcd",
        ("production.dcd",),
    )

    assert resolved == extended
