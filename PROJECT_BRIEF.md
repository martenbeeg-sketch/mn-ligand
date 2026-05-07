# PROJECT_BRIEF

## 1) App overview
`ovo-ligand` is a Streamlit-based, Docker-backed ligand workflow app. It is intentionally separated from the larger OVO design platform and focuses on small-molecule/protein workflows. The app currently includes:
- Structure preparation (PDB-centric, with cleaning/repair and ligand refinement artifacts)
- MD system preparation
- MD production from prepared systems
- Results and job tracking pages
- Placeholder/task pages for additional workflows (ADMET/QC/ABFE/RBFE and others)

The app is implemented as a Python package with CLI entrypoint `ovo-ligand`, and an app runtime rooted at:
- `/home/user/programs/git-projects/ovo-ligand/.ovo-home`

## 2) Current goal
Stabilize and complete the new modular workflow split:
1. **Structure Preparation** task produces reusable refined artifacts.
2. **MD System Preparation** task performs setup + short equilibration through NPT and writes restart assets.
3. **MD Production** task starts from system-prep outputs, preferably via checkpoint continuation.

The immediate objective is deterministic, reproducible production continuation and clean UX around restart mode selection and run provenance.

## 3) Tech stack (and inspiration folders)
- Python 3.11 package (`pyproject.toml`)
- Streamlit UI pages under `ovo_ligand/app/pages`
- Docker execution (`docker run`) from app pages
- Ligand-X-derived backend modules vendored under `ovo_ligand/ligandx`
- Mol* viewer component vendored under `ovo_ligand/app/components/molstar_viewer`
- MD stack in containers (OpenMM/OpenFF/RDKit-related, depending on image)

Inspiration/reference folders used during implementation:
- OVO source: `/home/user/programs/ovo/original/`
- Ligand-X source context: `/home/user/programs/ligand-x`
- HiQBind workflow reference was also discussed and inspected by user context: `/home/user/programs/HiQBind/workflow/` (use status: partial/To be confirmed)

## 4) Current implementation status
- The app has moved from a single “old landing/wizard” style toward separated tasks/pages.
- New pages exist for:
  - MD system prep jobs
  - MD system preparation run
  - MD production run
- Production restart mode selector exists:
  - `Checkpoint (exact continuation)` (recommended)
  - `NPT-final PDB (coordinate restart)` (less exact)
- Checkpoint-mode bugfixes were applied so production input includes resume paths.
- A path-mapping fix was applied so host paths are mapped to container-visible paths in production restart inputs.

Status caveat:
- Overall UX is still in transition and can show behavior mismatches between legacy and new page flows.

## 5) Most important files/folders
- App entry and navigation:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/run_app.py`
- Main legacy/new workflow pages:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/app/pages/bound_ligand_md.py`
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/app/pages/structure_preparation.py`
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/app/pages/md_system_preparation.py`
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/app/pages/md_production.py`
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/app/pages/jobs.py`
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/app/pages/jobs_md_system.py`
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/app/pages/md_results.py`
- MD workflow wrapper:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/workflows/bound_ligand_md.py`
- Core MD service/config/runner:
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/ligandx/services/md/config.py`
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/ligandx/services/md/service.py`
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/ligandx/services/md/workflow/equilibration_runner.py`
  - `/home/user/programs/git-projects/ovo-ligand/ovo_ligand/ligandx/services/md/workflow/system_builder.py`
- Runtime outputs:
  - `/home/user/programs/git-projects/ovo-ligand/.ovo-home/workdir/runs/`

## 6) Core features
- PDB download/inspection and bound ligand selection flow (legacy and modularized variants)
- Protein cleaning/repair pipeline integration
- Ligand refined chemistry handoff via SDF/SMILES inputs
- MD setup/equilibration in Docker
- MD production launch and results rendering
- Job tables for multiple run classes (structure, MD system prep, MD production/OpenFE pages in varying completeness)
- 3D visualization (Mol* and py3Dmol-based views in different contexts)

## 7) Known bugs or fragile areas
- Legacy/new page routing can still surface unexpected fallback pages in some paths.
- Restart behavior is sensitive to file provenance and container path mapping.
- Coordinate-restart mode is more error-prone than checkpoint restart.
- Some status/report fields can be inconsistent across job types during refactor.
- Production and system-prep file contracts are still evolving (checkpoint, system PDB, NPT final PDB dependencies).
- Optional features (e.g., MM/GBSA in Ligand-X original) remain marked not implemented.

## 8) Immediate next steps
1. Validate checkpoint restart end-to-end on multiple new system-prep runs.
2. Harden contracts:
   - required files for each restart mode
   - explicit errors and no silent fallback
3. Simplify/align MD production UI:
   - concise “what is used” summary
   - clear structure preview source
4. Clean up legacy navigation leftovers and remove confusing internal links from user-facing flow.
5. Add regression tests for:
   - checkpoint mode
   - coordinate restart mode
   - path translation host→container
6. Decide and lock production policy defaults per task (system-prep vs production).

## 9) Rules for future LLMs
- Do **not** reintroduce hidden fallback logic for production continuation without explicit UI/metadata warning.
- Prefer deterministic checkpoint continuation for production.
- If a detail is unknown, record `Unknown` or `To be confirmed` instead of guessing.
- Keep `ovo-ligand` self-contained; do not modify OVO core install unless explicitly requested.
- Preserve run provenance in `input.json`, `result.json`, and `metadata.json`.
- When changing restart logic, inspect a real failed run folder under:
  - `/home/user/programs/git-projects/ovo-ligand/.ovo-home/workdir/runs/`
- Keep container path visibility in mind (`/ovo-ligand` and `/output` mounts).
- Avoid broad refactors during bug-fix passes; make scoped, traceable changes.

## Commands (known)
- Activate env (example):
  - `source /home/user/mambaforge/etc/profile.d/conda.sh`
  - `conda activate ovo-ligand`
- Run app:
  - `ovo-ligand init`
  - `ovo-ligand app`
- Build images:
  - `docker compose build`

Last Updated: 2026-05-07
