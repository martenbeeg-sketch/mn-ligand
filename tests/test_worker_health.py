from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

from mn_ligand.app.pages.run_resources import resource_rows
from mn_ligand.core.worker_health import inspect_worker_health, record_worker_heartbeat


def test_worker_health_combines_systemd_heartbeat_queue_and_lease(tmp_path: Path) -> None:
    queued = tmp_path / "jobs" / "queued-1"
    queued.mkdir(parents=True)
    (queued / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "queued-1",
                "workflow": "fixture",
                "status": "queued",
                "queued_command": ["true"],
            }
        )
    )
    record_worker_heartbeat(
        tmp_path,
        worker_id="mn-ligand-gpu-0",
        gpu_ids=(0,),
        state="running",
        run_id="active-1",
        selected_gpu=0,
    )
    lock_dir = tmp_path / ".worker" / "locks"
    lock_dir.mkdir(parents=True)
    (lock_dir / "gpu-0.json").write_text(
        json.dumps({"run_id": "active-1", "owner_id": "mn-ligand-gpu-0"})
    )
    for slot in range(4):
        (lock_dir / f"cpu-{slot}.json").write_text(
            json.dumps(
                {
                    "run_id": "active-1",
                    "owner_id": "mn-ligand-gpu-0",
                    "cpu_slot": slot,
                }
            )
        )

    def fake_runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        active = "@0.service" in command[3]
        stdout = (
            "LoadState=loaded\n"
            f"ActiveState={'active' if active else 'inactive'}\n"
            f"SubState={'running' if active else 'dead'}\n"
            f"MainPID={'1234' if active else '0'}\n"
            f"UnitFileState={'enabled' if active else 'disabled'}\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    snapshot = inspect_worker_health((0, 1), run_dir=tmp_path, runner=fake_runner)

    assert snapshot.queued_jobs == 1
    assert snapshot.active_leases == 1
    assert snapshot.cpu_slots_leased == 4
    assert snapshot.cpu_pool_capacity >= 4
    assert snapshot.services[0].active is True
    assert snapshot.services[0].enabled is True
    assert snapshot.services[0].pid == 1234
    assert snapshot.services[0].worker_state == "running"
    assert snapshot.services[0].current_run_id == "active-1"
    assert snapshot.services[0].heartbeat_stale is False
    assert snapshot.services[1].active is False
    rows = resource_rows(snapshot)
    assert rows[0]["GPU availability"] == "Leased"
    assert rows[0]["worker slot"] == "Busy"
    assert rows[0]["current run"] == "active-1"
    assert rows[1]["GPU availability"] == "Offline"

    free_service = replace(
        snapshot.services[0],
        worker_state="idle",
        current_run_id="",
        lease_run_id="",
        heartbeat_stale=False,
    )
    stale_service = replace(free_service, gpu_id=1, heartbeat_stale=True)
    normalized = resource_rows(
        replace(snapshot, services=(free_service, stale_service), active_leases=0)
    )
    assert normalized[0]["GPU availability"] == "Free"
    assert normalized[1]["GPU availability"] == "Unknown (stale heartbeat)"


def test_worker_health_tolerates_unavailable_user_bus(tmp_path: Path) -> None:
    def failed_runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="Failed to connect to bus")

    snapshot = inspect_worker_health((0,), run_dir=tmp_path, runner=failed_runner)

    service = snapshot.services[0]
    assert service.active is False
    assert service.service_state == "unavailable"
    assert service.error == "Failed to connect to bus"
    assert service.heartbeat_stale is True
