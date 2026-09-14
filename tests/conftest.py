from __future__ import annotations

import pytest

from mn_ligand.core.resources import GPUCapacity, ResourceSnapshot


@pytest.fixture(autouse=True)
def deterministic_worker_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep worker tests independent of the workstation's live free space/load."""
    snapshot = ResourceSnapshot(
        cpu_threads_total=64,
        ram_available_gb=256,
        ram_total_gb=512,
        scratch_free_gb=500,
        scratch_total_gb=1000,
        gpus=(
            GPUCapacity(0, free_vram_gb=24, total_vram_gb=24),
            GPUCapacity(1, free_vram_gb=24, total_vram_gb=24),
        ),
    )
    monkeypatch.setattr(
        "mn_ligand.core.worker.capture_resource_snapshot",
        lambda _path: snapshot,
    )
