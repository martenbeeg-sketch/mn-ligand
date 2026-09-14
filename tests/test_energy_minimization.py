from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mn_ligand.core.artifacts import ArtifactRef, write_artifact_manifest
from mn_ligand.core.jobs import JobRecord
from mn_ligand.workflows import energy_minimization
from mn_ligand.workflows.energy_minimization import minimize_prepared_complex


COMPLEX = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
HETATM    3  C1  LIG X 101       3.000   2.000   0.000  1.00 20.00           C
END
"""


def test_minimization_backfill_publishes_typed_derived_complex(
    tmp_path: Path, monkeypatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("MN_LIGAND_RUN_DIR", str(runs))
    source_dir = runs / "structure-jobs" / "source"
    source_dir.mkdir(parents=True)
    complex_path = source_dir / "complex.pdb"
    complex_path.write_text(COMPLEX)
    ligand_path = source_dir / "ligand.sdf"
    ligand_path.write_text("LIG\n  test\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n")
    (source_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "source",
                "status": "completed",
                "source": "pdb",
                "pdb_id": "1ABC",
                "ligand_key": "LIG|X|101|_",
            }
        )
    )
    manifest = write_artifact_manifest(
        source_dir,
        [
            ArtifactRef.from_path(
                source_dir, complex_path, "prepared_complex", role="complex"
            ),
            ArtifactRef.from_path(
                source_dir, ligand_path, "prepared_ligand_set", role="ligand"
            ),
        ],
    )
    source_job = JobRecord.load(source_dir, task_group="structure-jobs")

    monkeypatch.setattr(
        energy_minimization,
        "_cleaning_command",
        lambda **kwargs: ["fake-minimizer"],
    )

    def fake_run(*args, **kwargs):
        run_dir = next(
            path
            for path in (runs / "structure-jobs").iterdir()
            if path.name != "source"
        )
        (run_dir / "native_result.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "prepared_pdb_data": COMPLEX,
                    "repair_report": {
                        "refinement": {
                            "engine": "OpenMM LocalEnergyMinimizer",
                            "forcefield": ["amber14-all.xml"],
                            "ligand_context": True,
                            "ligand_parameterization": "SMIRNOFF",
                            "movable_atom_count": 2,
                            "frozen_atom_count": 3,
                            "potential_energy_before_kj_mol": 10.0,
                            "potential_energy_after_kj_mol": 2.0,
                            "max_iterations": 1000,
                            "tolerance_kj_mol_nm": 10.0,
                        }
                    },
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(energy_minimization.subprocess, "run", fake_run)
    job = minimize_prepared_complex(
        source_job=source_job,
        source_artifact=manifest.by_type("prepared_complex")[0],
        source_path=complex_path,
    )

    assert job.status == "completed"
    assert job.parent_run_id == "source"
    assert job.metadata["energy_minimized"] is True
    assert job.artifact_manifest is not None
    assert len(job.artifact_manifest.by_type("prepared_complex")) == 1
    assert len(job.artifact_manifest.by_type("prepared_receptor")) == 1
    assert len(job.artifact_manifest.by_type("prepared_ligand_set")) == 1
    report = json.loads(
        job.artifact_manifest.by_type("minimization_report")[0]
        .resolve(job.run_dir, must_exist=True)
        .read_text()
    )
    assert report["potential_energy_before_kj_mol"] == 10.0
    assert report["potential_energy_after_kj_mol"] == 2.0
    assert report["ligand_context"] is True
