# PROJECT_BRIEF

## 1) App overview
`mn-ligand` is a Streamlit-based, Docker-backed ligand workflow app. It is intentionally separated from the larger external design platform and focuses on small-molecule/protein workflows. The app currently includes:
- Structure preparation (PDB-centric and docking-from-prepared-structure paths with cleaning/repair and ligand refinement artifacts)
- MD system preparation
- MD production from prepared systems
- Results and job tracking pages
- Active task pages for ADMET/QC/OpenFE and docking engines (UDP/Vina/Gnina) in the structure workflow

The app is implemented as a Python package with CLI entrypoint `mn-ligand`, and an app runtime rooted at:
- `/home/user/programs/git-projects/mn-ligand/mn-ligand-workdir`

## 2) Current goal
Stabilize and complete the modular workflow split with consistent artifact handoff:
1. **Structure Preparation** task produces reusable refined artifacts (including docking-derived refined outputs).
2. **MD System Preparation** task performs setup + short equilibration through NPT and writes restart assets.
3. **MD Production** task starts from system-prep outputs with explicit provenance and restart controls.
4. **OpenFE/ADMET/QC** tasks run as tracked jobs with consistent run folders and result pages.

Immediate objective: keep all task outputs deterministic, discoverable in job tables, and consumable by downstream tasks without ambiguous fallback behavior.

## 3) Tech stack (and inspiration folders)
- Python 3.11 package (`pyproject.toml`)
- Streamlit UI pages under `mn_ligand/app/pages`
- Docker execution (`docker run`) from app pages
- Ligand-X-derived backend modules vendored under `mn_ligand/ligandx`
- Mol* viewer component vendored under `mn_ligand/app/components/molstar_viewer`
- MD stack in containers (OpenMM/OpenFF/RDKit-related, depending on image)

Inspiration/reference folders used during implementation:
- - Ligand-X source context: `/home/user/programs/ligand-x`
- HiQBind workflow reference was also discussed and inspected by user context: `/home/user/programs/HiQBind/workflow/` (use status: partial/To be confirmed)

## 4) Current implementation status
- The app has moved from a single “old landing/wizard” style toward separated tasks/pages.
- Implemented task flows include:
  - Structure preparation from PDB
  - Structure preparation from docking of prepared structures (UDP/Vina/Gnina)
  - MD system preparation
  - MD production
  - OpenFE ABFE/RBFE job flow
  - ADMET and QC minimal job flows with result pages
- Job tables are in place across major tasks with links on job codes and auto-refresh behavior.
- Structure results include docking metadata and ligand overlay/RMSD-oriented views.
- Production restart handling and run provenance were hardened (input/result/metadata consistency).

Status caveats:
- **Boltz2 prediction route in structure preparation is not completed** (`To be confirmed`).
- **From custom files route (upload + cleaning for uploaded PDB/SDF) is not completed** (`To be confirmed`).
- Some legacy-to-modular UX transitions still have edge-case inconsistencies.

## 5) Most important files/folders
- App entry and navigation:
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/run_app.py`
- Main legacy/new workflow pages:
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/app/pages/bound_ligand_md.py`
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/app/pages/structure_preparation.py`
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/app/pages/md_system_preparation.py`
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/app/pages/md_production.py`
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/app/pages/jobs.py`
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/app/pages/jobs_md_system.py`
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/app/pages/md_results.py`
- MD workflow wrapper:
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/workflows/bound_ligand_md.py`
- Core MD service/config/runner:
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/ligandx/services/md/config.py`
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/ligandx/services/md/service.py`
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/ligandx/services/md/workflow/equilibration_runner.py`
  - `/home/user/programs/git-projects/mn-ligand/mn_ligand/ligandx/services/md/workflow/system_builder.py`
- Runtime outputs:
  - `/home/user/programs/git-projects/mn-ligand/mn-ligand-workdir/workdir/runs/`

## 6) Core features
- PDB download/inspection and bound ligand selection flow (legacy and modularized variants)
- Protein cleaning/repair pipeline integration
- Ligand refined chemistry handoff via SDF/SMILES inputs
- Docking-from-prepared-structure workflow with engine selection (`udp`, `vina`, `gnina`)
- MD setup/equilibration in Docker
- MD production launch and results rendering
- Job tables for multiple run classes (structure, MD system prep, MD production, OpenFE, ADMET, QC)
- 3D visualization (Mol* and py3Dmol-based views in different contexts)

## 7) Known bugs or fragile areas
- Legacy/new page routing can still surface unexpected fallback pages in some paths.
- Restart behavior is sensitive to file provenance and container path mapping.
- Coordinate-restart mode is more error-prone than checkpoint restart.
- Some status/report fields can be inconsistent across job types during refactor.
- Production and system-prep file contracts are still evolving (checkpoint, system PDB, NPT final PDB dependencies).
- Feature parity gaps remain:
  - Boltz2 structure-prep path (`To be confirmed`)
  - Custom-file upload+cleaning route (`To be confirmed`)

## 8) Immediate next steps
1. Finish **Boltz2 prediction** path in structure preparation (`To be confirmed`).
2. Finish **From custom files** path with robust upload validation + cleaning (uploaded PDB/SDF) (`To be confirmed`).
3. Keep downstream compatibility checks strict:
   - structure outputs -> MD system prep
   - structure outputs -> OpenFE
4. Continue hardening run contracts:
   - required files for each task
   - explicit errors and no silent fallback
5. Add regression tests for:
   - docking-derived structure to MD handoff
   - checkpoint continuation
   - host→container path translation

## 9) Rules for future LLMs
- Do **not** reintroduce hidden fallback logic for production continuation without explicit UI/metadata warning.
- Prefer deterministic checkpoint continuation for production.
- If a detail is unknown, record `Unknown` or `To be confirmed` instead of guessing.
- Keep `mn-ligand` self-contained; do not modify external platform installs unless explicitly requested.
- Preserve run provenance in `input.json`, `result.json`, and `metadata.json`.
- When changing restart logic, inspect a real failed run folder under:
  - `/home/user/programs/git-projects/mn-ligand/mn-ligand-workdir/workdir/runs/`
- Keep container path visibility in mind (`/mn-ligand` and `/output` mounts).
- Avoid broad refactors during bug-fix passes; make scoped, traceable changes.

## Commands (known)
- Activate env (example):
  - `source /home/user/mambaforge/etc/profile.d/conda.sh`
  - `conda activate mn-ligand`
- Run app:
  - `mn-ligand init`
  - `mn-ligand app`
- Build images:
  - `docker compose build`

Last Updated: 2026-05-12
